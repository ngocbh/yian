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
 * Later (T4-8): needs `resumeFlow`: `test_tool_calling_agent_in_loop_retries_whole_node_after_process_loss`.
 */

import { expect, test } from "bun:test";

import { ProviderError } from "../../src/llm/errors.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { serving, textReply } from "./run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A `ping` tool that takes no arguments and answers `pong`. */
function ping(): FunctionTool {
  return new FunctionTool({
    name: "ping",
    description: "Ping.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "pong",
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
    env: () => undefined,
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
    env: () => undefined,
  });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("provider.auth");
  expect(result.failure!.causeType).toBe("ProviderError");
});
