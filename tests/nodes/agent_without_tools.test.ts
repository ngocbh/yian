/**
 * An AGENT without tools or skills makes a single model call; one with tools runs the loop. An
 * `Optional(str)` agent without tools returns the reply text, and an empty reply is `""`, not null.
 */

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunEvent, RunStarted, type StreamChunk } from "../../src/events.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { interrupt, type RunFlowOptions, send } from "../../src/run.ts";
import { NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { drive, runWith, serving } from "../helpers/run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };

function reply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** The body line that gives `chat` the steer inbox `notes`. */
const STEER = "steer_inbox: ${input.notes}";

/** One `chat` agent with no tools, plus the given body lines; `STEER` declares its inbox. */
function chatFlow(...lines: string[]): string {
  const body = lines.map((line) => `    ${line}\n`).join("");
  const input = lines.includes(STEER) ? "input:\n  notes: !Inbox(str)\n" : "";
  return `id: single
name: single
llm_config: {provider: anthropic, model: claude-test}
${input}nodes:
  chat:
    kind: agent
    prompt: Help the user.
${body}output: \${chat.output}
`;
}

/** Load `chatFlow(...lines)` and run it with `model`; return the run's output once it succeeds. */
async function run(lines: string[], model: ScriptedModel, options: RunFlowOptions = {}): Promise<unknown> {
  const result = await runWith(await loadFlow(chatFlow(...lines)), {}, model, options);
  expect(result.status).toBe("succeeded");
  return result.output;
}

test("text agent without tools makes one unbound call", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(["output: !str"], model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("structured agent without tools makes one structured call", async () => {
  const model = new ScriptedModel([reply('{"name": "Ada", "score": 3}')], CONFIG);
  expect(await run(["output: {name: !str, score: !int}"], model)).toEqual({ name: "Ada", score: 3n });
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("skills and a steer inbox need no mode", () => {
  expect(new AgentNode("n", { prompt: "p", skills: ["review"] }).skills).toEqual(["review"]);
  expect(new AgentNode("n", { prompt: "p", steerInbox: true }).steerInbox).toBe(true);
});

test("an agent built in code with tools runs the tool loop", async () => {
  const seen: unknown[] = [];
  const value = new FunctionTool({
    name: "value",
    description: "Price a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } } },
    execute: (args) => {
      seen.push(args);
      return `px:${args.topic}`;
    },
  });
  const call = new ToolCallPart({ id: "1", name: "value", arguments: { topic: "ACME" } });
  const calling = new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
  const model = new ScriptedModel([calling, reply("done")], CONFIG);
  const node = new AgentNode("n", { prompt: "hi", tools: ["value"], llmConfig: CONFIG });

  const outcome = await drive(node, { models: serving(model), tools: new ToolSet({ value }) });

  expect(outcome).toEqual(new Output("done"));
  expect(seen).toEqual([{ topic: "ACME" }]);
  expect(model.requests).toHaveLength(2);
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["value"]);
});

test("a steer joins the single request after the prompt", async () => {
  const model = new ScriptedModel([reply("ok")], CONFIG);
  const log = join(mkdtempSync(join(tmpdir(), "yian-no-tools-")), "log.jsonl");
  const onEvent = (event: RunEvent) => {
    if (event instanceof RunStarted) void send(log, "notes", "be brief");
  };
  expect(await run([STEER, "output: !str"], model, { log, onEvent })).toBe("ok");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.messages.slice(-2).map((message) => [message.role, message.text()])).toEqual([
    ["user", "Help the user."],
    ["user", "be brief"],
  ]);
});

test("an interrupt while streaming records the partial text; an Optional(str) agent ends null", async () => {
  const model = new ScriptedModel(
    [[new TextDelta(0, "hello"), new TextDelta(0, " big"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]],
    CONFIG,
  );
  const log = join(mkdtempSync(join(tmpdir(), "yian-no-tools-")), "log.jsonl");
  const chunks: StreamChunk[] = [];
  const onStream = (chunk: StreamChunk) => {
    chunks.push(chunk);
    // A live delivery lands in this tick, before the next chunk.
    if (chunk.chunk === " big") void interrupt(log, "notes");
  };
  expect(await run([STEER, "output: !Optional(str)"], model, { log, onStream, numWorkers: 0 })).toBeNull();
  expect(chunks.map((chunk) => [chunk.chunk, chunk.final])).toEqual([
    ["hello", false],
    [" big", false],
    ["", true],
  ]);
  const cut = readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord ? [record.step] : []));
  expect(cut.map((step) => [step["kind"], step["finish"] ?? null])).toEqual([
    ["model_call", "interrupted"],
    ["interrupted", null],
  ]);
  expect((cut[0]!["output"] as { parts: { text: string }[] }).parts[0]!.text).toBe("hello big");
});

test("an Optional(str) agent without tools makes one unbound call and returns the text", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(["output: !Optional(str)"], model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
  expect(model.requests[0]!.outputFormat).toBeNull();
});

test("an empty reply from an Optional(str) agent is empty text, not null", async () => {
  const model = new ScriptedModel([reply("")], CONFIG);
  expect(await run(["output: !Optional(str)"], model)).toBe("");
});
