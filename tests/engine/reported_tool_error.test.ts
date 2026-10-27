/**
 * `ReportedToolError`: an AGENT's model reads it as the call's result and the loop continues, and
 * a TOOL node with retries stops on one built with `permanent: true`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ToolCallFinished } from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Output } from "../../src/nodes/base.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { ReportedToolError } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { drive, rejected, serving } from "./run_state_fixtures.ts";

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

// ---------- TOOL nodes ----------

const dir = mkdtempSync(join(tmpdir(), "yian-reported-tool-error-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("tool retries stop on a reported error that is also permanent", async () => {
  class Gone extends ReportedToolError {
    override name = "Gone";
  }
  const calls: string[] = [];
  const tool = new FunctionTool({
    name: "read_file",
    description: "Read one file.",
    parametersJsonSchema: { type: "object", properties: { path: { type: "string" } } },
    execute: (args) => {
      calls.push((args.path as string | undefined) ?? "a.txt");
      throw new Gone("gone", { permanent: true });
    },
  });
  const flow = await loadFlow(
    "id: t\nname: t\nnodes:\n  call:\n    kind: tool\n    tool_id: read_file\n    retries: 2\n" +
      "output: ${call.output}\n",
  );
  const result = await runFlow(flow, {}, {
    tools: { read_file: tool },
    env: hostEnv(dir, { YIAN_TOOL_RETRY_DELAY_SECONDS: "0" }),
  });
  expect(result.status).toBe("failed");
  expect([result.failure!.message, result.failure!.causeType]).toEqual(["gone", "Gone"]);
  expect(calls).toHaveLength(1);
});
