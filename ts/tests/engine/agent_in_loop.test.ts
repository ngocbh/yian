/**
 * Tool-calling AGENT behavior inside a dynamically grown LOOP body: a port of
 * `tests/engine/test_agent_in_loop.py`.
 *
 * Python patches `model_from_config` with a fake chat and registers a fake `ping` tool; here the
 * run is given a `ScriptedModel` through `serving()` and its provider config (`models`,
 * `llmConfig`), and a `FunctionTool` ping in its `tools`. The fold is inline TS code, as the
 * fixtures have no `reply_fold`.
 *
 * Ported: `test_tool_calling_agent_in_loop_runs` and
 * `test_auth_failure_inside_loop_reaches_run_result`.
 *
 * Changed: `test_auth_failure_inside_loop_reaches_run_result`: the refusal is a `ProviderError` of
 * kind `auth`, so `causeType` is `"ProviderError"`, the JS error name, where Python's fake raises
 * an `AuthenticationError`.
 *
 * Ported through `runFlow` and `resumeFlow`, in `hostEnv`:
 * `test_tool_calling_agent_in_loop_retries_whole_node_after_process_loss`. Changed: the process
 * loss is `crashAfter`'s log refusing the append of the second turn's AGENT success, after its
 * tool call ran and its model answered (Python: the model's fourth call raises).
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { IN_LOOP_EVENTS, type RunEvent } from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { NodeStartedRecord, NodeSucceededRecord, readRunLog, RunResumedRecord } from "../../src/runtime/log.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { crashAfter, serving, textReply } from "./run_state_fixtures.ts";
import { NO_ENV } from "./session_fakes.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-in-loop-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A `ping` tool that takes no arguments and answers `pong`. */
function ping(seen: unknown[] = []): FunctionTool {
  return new FunctionTool({
    name: "ping",
    description: "Ping.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: (args) => {
      seen.push(args);
      return "pong";
    },
  });
}

/** One assistant reply requesting a single `name` tool call with `args`. */
function toolCall(name: string, args: Record<string, never>, callId = "1"): Response {
  const part = new ToolCallPart({ id: callId, name, arguments: args });
  return new Response({ message: new Message({ role: "assistant", parts: [part] }), finishReason: "tool_calls" });
}

const FLOW = `
id: t
name: t
input:
  seed: !str
defs:
  body:
    input:
      transcript: !str
      exited: !bool
    nodes:
      reply:
        kind: agent
        tools: [ping]
        input:
          transcript: \${input.transcript}
        output: str
        prompt: "history \${transcript}"
      fold:
        kind: code
        code: |
          return { transcript: inputs.transcript + "|" + inputs.reply, exited: inputs.exited }
        input:
          transcript: \${input.transcript}
          reply:      \${reply.output}
          exited:     \${input.exited}
        output:
          transcript: str
          exited: bool
    output: \${fold.output}
nodes:
  loop:
    kind: loop
    call: body
    input:
      transcript: \${input.seed}
      exited: false
    times: 2
output: \${loop.output.transcript}
`;

test("tool calling agent in loop runs", async () => {
  // Two uninterrupted LOOP turns each complete one local model/tool loop: the model calls the
  // tool once, then answers; two turns are four replies.
  const model = new ScriptedModel(
    [toolCall("ping", {}), textReply("turn1"), toolCall("ping", {}), textReply("turn2")],
    CONFIG,
  );
  const result = await runFlow(await loadConformanceFlow(FLOW), { seed: "s" }, {
    tools: { ping: ping() },
    llmConfig: CONFIG,
    models: serving(model),
    env: NO_ENV,
  });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("s|turn1|turn2");
});

// FLOW's body under a `while:` loop with a `max:` budget.
const BOUNDED_FLOW = FLOW.replace("    times: 2\n", "    while: not exited\n    max: 3\n");

test("auth failure inside loop reaches run result", async () => {
  // A body AGENT's refused credentials stay `provider.auth` through the loop.
  const model = new ScriptedModel([new ProviderError("auth", "unauthorized", { status: 401 })], CONFIG);
  const result = await runFlow(await loadConformanceFlow(BOUNDED_FLOW), { seed: "s" }, {
    tools: { ping: ping() },
    llmConfig: CONFIG,
    models: serving(model),
    env: NO_ENV,
  });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("provider.auth");
  expect(result.failure!.causeType).toBe("ProviderError");
});

test("tool calling agent in loop retries whole node after process loss", async () => {
  // A fresh flow and model retry the one namespaced AGENT the lost process left started.
  const path = join(dir, "agent-loop.jsonl");
  const seen: unknown[] = [];
  const tools = { ping: ping(seen) };
  const interrupted = new ScriptedModel(
    [toolCall("ping", {}, "turn-1"), textReply("turn1"), toolCall("ping", {}, "turn-2-before-crash"), textReply("lost")],
    CONFIG,
  );
  // Record 18 is the second turn's `loop#1/reply` success.
  await crashAfter(18, async () =>
    runFlow(await loadConformanceFlow(FLOW), { seed: "s" }, {
      log: path,
      runId: "agent-loop-recovery",
      tools,
      llmConfig: CONFIG,
      models: serving(interrupted),
      env: hostEnv(dir),
    }),
  );

  const prefix = readRunLog(path).map((envelope) => envelope.record);
  const last = prefix.at(-1);
  expect(last).toBeInstanceOf(NodeStartedRecord);
  const retryId = (last as NodeStartedRecord).nodeId;
  expect(retryId).toBe("loop#1/reply");
  expect(prefix.filter((r) => r instanceof NodeSucceededRecord && r.nodeId.endsWith("/reply")).length).toBe(1);
  expect(seen).toEqual([{}, {}]);

  const recoveredModel = new ScriptedModel([toolCall("ping", {}, "turn-2-retry"), textReply("turn2")], CONFIG);
  const recoveredEvents: RunEvent[] = [];
  const recovered = await resumeFlow(await loadConformanceFlow(FLOW), {
    log: path,
    tools,
    onEvent: (event) => void recoveredEvents.push(event),
    models: serving(recoveredModel),
    env: hostEnv(dir),
  });
  expect(recovered.status, recovered.failure?.message).toBe("succeeded");
  expect(recovered.output).toBe("s|turn1|turn2");
  expect(recovered.runId).toBe("agent-loop-recovery");
  // Only the re-executed AGENT publishes fresh in-loop observers; none are replayed.
  expect(recovered.events).toEqual(
    recoveredEvents.filter((event) => !IN_LOOP_EVENTS.some((type) => event instanceof type)),
  );
  expect(seen).toEqual([{}, {}, {}]);
  const records = readRunLog(path).map((envelope) => envelope.record);
  expect(records.filter((r) => r instanceof RunResumedRecord)).toEqual([new RunResumedRecord([retryId], 8)]);
  expect(records.filter((r) => r instanceof NodeStartedRecord && r.nodeId === retryId).length).toBe(2);
  expect(records.filter((r) => r instanceof NodeSucceededRecord && r.nodeId === retryId).length).toBe(1);

  // Terminal replay builds no model and leaves the log as it is.
  const unexpected = {
    env: NO_ENV,
    createModel: (): never => {
      throw new Error("terminal replay must not build a model");
    },
  };
  const before = readFileSync(path);
  const terminalEvents: RunEvent[] = [];
  const terminal = await resumeFlow(await loadConformanceFlow(FLOW), {
    log: path,
    tools,
    onEvent: (event) => void terminalEvents.push(event),
    models: unexpected,
    env: hostEnv(dir),
  });
  expect(terminal.status, terminal.failure?.message).toBe("succeeded");
  expect(terminal.output).toBe(recovered.output);
  expect(terminal.runId).toBe(recovered.runId);
  expect(terminal.events).toEqual([]);
  expect(terminalEvents).toEqual([]);
  expect(readFileSync(path).equals(before)).toBe(true);
  expect(seen).toEqual([{}, {}, {}]);
});
