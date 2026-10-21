/**
 * `ReportedToolError`: an AGENT's model reads it as the call's result; mirrors
 * `tests/engine/test_reported_tool_error.py`.
 *
 * Ported, the AGENT rows: `test_an_agent_model_reads_a_reported_error_and_the_loop_continues`,
 * `test_an_empty_reported_error_reads_as_its_type_name`,
 * `test_an_ordinary_tool_error_still_fails_the_agent_node` and the `session=False` row of
 * `test_a_pending_stop_wins_over_a_reported_error`; through `FlowEngine` with a `SessionMemory`
 * (`runSessionFlow`), the `session=True` row of `test_a_pending_stop_wins_over_a_reported_error`
 * and `test_a_session_records_the_reported_call_completed_with_its_message`.
 *
 * Changed: the rows run an AGENT `worker` through `evalNode` (`drive`) with a `ScriptedModel`
 * from `models` and the tool in the run's `ToolSet`, where Python runs a loaded flow with
 * `run_flow`. The empty error's subclass sets its own `name` (a JS subclass inherits
 * `ReportedToolError`'s); the ordinary error is an `Error` (Python: `RuntimeError`). A pending stop
 * rejects with the signal's reason; TS errors have no implicit context chain, so the reported
 * error isn't checked on it.
 *
 * Later (T4-8), the TOOL rows through the runner: `test_a_tool_node_fails_on_a_reported_error`,
 * `test_tool_retries_retry_a_reported_error` and
 * `test_tool_retries_stop_on_a_reported_error_that_is_also_permanent`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunSucceeded, ToolCallFinished } from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { decodeMessage } from "../../src/nodes/agent/memory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Output } from "../../src/nodes/base.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { ReportedToolError } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { drive, rejected, serving } from "./run_state_fixtures.ts";
import { runSessionFlow, sessionFlow, view } from "./session_fakes.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "test-model" };

/** A `read_file` tool that records each path it is called with in `calls`, then throws `error`. */
function raising(error: unknown, calls: string[], during: () => void = () => {}): FunctionTool {
  return new FunctionTool({
    name: "read_file",
    description: "Read one file.",
    parametersJsonSchema: { type: "object", properties: { path: { type: "string" } } },
    execute: (args) => {
      calls.push(args.path as string);
      during();
      throw error;
    },
  });
}

/** A reply calling `read_file` on `a.txt` as `c0`, then the answer. */
function callThenAnswer(answer = "the file is missing"): Response[] {
  const call = new ToolCallPart({ id: "c0", name: "read_file", arguments: { path: "a.txt" } });
  return [
    new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
    new Response({ message: new Message({ role: "assistant", parts: [new TextPart(answer)] }), finishReason: "stop" }),
  ];
}

/** Evaluate an AGENT `worker` granted `read_file` over `model`, with the other `drive` options. */
function evaluate(model: ScriptedModel, tool: FunctionTool, options: Parameters<typeof drive>[1] = {}) {
  const node = new AgentNode("worker", { prompt: "go", llmConfig: CONFIG, tools: ["read_file"] });
  return drive(node, { models: serving(model), tools: new ToolSet({ read_file: tool }), ...options });
}

/** The tool results of request `index`. */
function toolResults(model: ScriptedModel, index: number): ToolResultPart[] {
  return model.requests[index]!.messages.flatMap((message) =>
    message.role === "tool" ? message.parts.filter((part) => part instanceof ToolResultPart) : [],
  );
}

test("an agent model reads a reported error and the loop continues", async () => {
  const model = new ScriptedModel(callThenAnswer(), CONFIG);
  const calls: string[] = [];
  const events: unknown[] = [];
  const outcome = await evaluate(model, raising(new ReportedToolError("no such file"), calls), {
    emitEvent: (event) => void events.push(event),
  });
  expect((outcome as Output).value).toBe("the file is missing");
  expect(calls).toEqual(["a.txt"]);
  expect(toolResults(model, 1)).toEqual([
    new ToolResultPart({ callId: "c0", name: "read_file", content: "no such file", isError: true }),
  ]);
  const finished = events.filter((event) => event instanceof ToolCallFinished);
  expect(finished.map((event) => [event.error, event.result])).toEqual([["ReportedToolError", null]]);
});

test("an empty reported error reads as its type name", async () => {
  class NotFound extends ReportedToolError {
    override name = "NotFound";
  }
  const model = new ScriptedModel(callThenAnswer(), CONFIG);
  const outcome = await evaluate(model, raising(new NotFound(), []));
  expect((outcome as Output).value).toBe("the file is missing");
  expect(toolResults(model, 1).map((part) => [part.isError, part.content])).toEqual([[true, "NotFound"]]);
});

test("an ordinary tool error still fails the agent node", async () => {
  const model = new ScriptedModel(callThenAnswer(), CONFIG);
  const outcome = await evaluate(model, raising(new Error("disk on fire"), []));
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const failure = (outcome as EvaluationFailure).failure;
  expect([failure.message, failure.causeType]).toEqual(["disk on fire", "Error"]);
  expect(model.requests).toHaveLength(1);
});

test("a pending stop wins over a reported error", async () => {
  const stop = new AbortController();
  const model = new ScriptedModel(callThenAnswer("missing"), CONFIG);
  const tool = raising(new ReportedToolError("no such file"), [], () => stop.abort(new Error("stopped")));
  expect(await rejected(evaluate(model, tool, { signal: stop.signal }))).toBe(stop.signal.reason);
  expect(model.requests).toHaveLength(1);
});

/** A `lookup` tool taking an integer `value` that throws `error(value)`, after `during()`. */
function lookupRaising(error: (value: unknown) => unknown, during: () => void = () => {}): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look one thing up.",
    parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
    execute: (args) => {
      during();
      throw error(args.value);
    },
  });
}

/** A reply calling `lookup` on 1 as `c0`, then each of `answers`. */
function lookupThenAnswer(...answers: string[]): Response[] {
  const call = new ToolCallPart({ id: "c0", name: "lookup", arguments: { value: 1 } });
  return [
    new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
    ...answers.map((answer) => {
      const message = new Message({ role: "assistant", parts: [new TextPart(answer)] });
      return new Response({ message, finishReason: "stop" });
    }),
  ];
}

test("a pending stop wins over a reported error in a session", async () => {
  const stop = new AbortController();
  const memory = new SessionMemory();
  const model = new ScriptedModel(lookupThenAnswer("missing"), CONFIG);
  const tool = lookupRaising(
    () => new ReportedToolError("no such file"),
    () => stop.abort(new Error("stopped")),
  );
  const running = runSessionFlow(sessionFlow({ tools: ["lookup"] }), memory, {
    inputs: { question: "q" },
    tools: new ToolSet({ lookup: tool }),
    models: serving(model),
    signal: stop.signal,
  });
  expect(await rejected(running)).toBe(stop.signal.reason);
  expect(model.requests).toHaveLength(1);
  const stored = view(memory);
  expect(stored.effects.map((effect) => effect.status)).toEqual(["completed", "interrupted"]);
  expect(stored.effects[1]!.failure!.code).toBe("api.tool.interrupted");
});

test("a session records the reported call completed with its message", async () => {
  const memory = new SessionMemory();
  const model = new ScriptedModel(lookupThenAnswer("missing", "still missing"), CONFIG);
  const tools = new ToolSet({ lookup: lookupRaising((value) => new ReportedToolError(`no record ${value}`)) });
  const run = (runId: string, question: string) =>
    runSessionFlow(sessionFlow({ tools: ["lookup"] }), memory, {
      runId,
      inputs: { question },
      tools,
      models: serving(model),
    });
  expect(await run("first", "q")).toEqual(new RunSucceeded({ output: "missing" }));
  const stored = view(memory);
  expect(stored.effects.map((effect) => [effect.kind, effect.status])).toEqual([
    ["model", "completed"],
    ["tool", "completed"],
    ["model", "completed"],
  ]);
  expect(stored.effects[1]!.failure).toBeNull();
  const parts = stored.items.map(decodeMessage).flatMap((message) => message.parts);
  const results = parts.filter((part) => part instanceof ToolResultPart);
  expect(results).toEqual([
    new ToolResultPart({ callId: "c0", name: "lookup", content: "no record 1", isError: true }),
  ]);
  // A later turn on the same session replays the reported error to the model.
  expect(await run("second", "again")).toEqual(new RunSucceeded({ output: "still missing" }));
  expect(toolResults(model, 2)).toEqual(results);
});
