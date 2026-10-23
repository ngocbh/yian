/**
 * An AGENT list prompt renders into the request's messages, after the system message.
 *
 * TS-only: Python yian has no list prompt. Driven through `forward` with a `ScriptedModel`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { AgentNode, type PromptItem } from "../../src/nodes/agent/node.ts";
import { type NodeCaps, Output } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { logJson } from "../../src/typesys/json.ts";
import { callReply, rejected, serving, textReply } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const CHAT: readonly PromptItem[] = [{ splice: "history" }, { role: "user", text: "Now: ${message}" }];
const HISTORY = [
  { role: "user", text: "Hi", at: "9:00" },
  { role: "assistant", text: "Hello! How can I help?" },
];

/** An AGENT `reply` reading inputs `history` and `message`, with `prompt`. */
function agent(prompt: string | readonly PromptItem[]): AgentNode {
  const node = new AgentNode("reply", { prompt, systemPrompt: "Be kind.", llmConfig: CONFIG });
  node.params = [new Parameter("history", { required: true }), new Parameter("message", { required: true })];
  return node;
}

/** Run `node` on `history` and the message `again`, answering `done`; return the model and caps. */
async function run(node: AgentNode, history: unknown, extra: Partial<NodeCaps> = {}) {
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const caps: NodeCaps = { models: serving(model), ...extra };
  const result = await node.forward({ history, message: "again" }, caps);
  return { model, result };
}

/** The `[role, text]` of each message of `model`'s first request. */
function sent(model: ScriptedModel): [string, string][] {
  return model.requests[0]!.messages.map((message) => [message.role, message.text()]);
}

test("the request holds the system message, the history's turns, then the user message", async () => {
  const { model, result } = await run(agent(CHAT), HISTORY);

  expect(result).toEqual(new Output("done"));
  expect(sent(model)).toEqual([
    ["system", "Be kind."],
    ["user", "Hi"],
    ["assistant", "Hello! How can I help?"],
    ["user", "Now: again"],
  ]);
});

test("a spliced message, literal turns and an empty spliced list render in order", async () => {
  const prompt: PromptItem[] = [
    { role: "user", text: "Start" },
    { role: "assistant", text: "Ready." },
    { splice: "history" },
    { splice: "message" },
  ];
  const node = agent(prompt);
  const model = new ScriptedModel([textReply("done")], CONFIG);

  await node.forward({ history: [], message: { role: "user", text: "Go" } }, { models: serving(model) });

  expect(sent(model)).toEqual([
    ["system", "Be kind."],
    ["user", "Start"],
    ["assistant", "Ready."],
    ["user", "Go"],
  ]);
});

test("a spliced message's text is sent verbatim, never rendered as a template", async () => {
  const history = [
    { role: "user", text: "What does ${message} do?" },
    { role: "assistant", text: "It reads `message`." },
  ];
  const { model } = await run(agent(CHAT), history);

  expect(sent(model).slice(1, 3)).toEqual([
    ["user", "What does ${message} do?"],
    ["assistant", "It reads `message`."],
  ]);
});

test("every request of the tool loop carries the history", async () => {
  const parametersJsonSchema = { type: "object", properties: {} };
  const noop = new FunctionTool({
    name: "noop",
    description: "Do nothing.",
    parametersJsonSchema,
    execute: () => "ok",
  });
  const node = new AgentNode("reply", { prompt: CHAT, systemPrompt: "Be kind.", llmConfig: CONFIG, tools: ["noop"] });
  node.params = [new Parameter("history", { required: true }), new Parameter("message", { required: true })];
  const model = new ScriptedModel([callReply(["noop", {}, "call-1"]), textReply("done")], CONFIG);

  const result = await node.forward(
    { history: HISTORY, message: "again" },
    { models: serving(model), tools: new ToolSet({ noop }) },
  );

  expect(result).toEqual(new Output("done"));
  expect(model.requests).toHaveLength(2);
  const second = model.requests[1]!.messages.map((message) => [message.role, message.text()]);
  expect(second.slice(0, 4)).toEqual([
    ["system", "Be kind."],
    ["user", "Hi"],
    ["assistant", "Hello! How can I help?"],
    ["user", "Now: again"],
  ]);
  expect(second.slice(4).map(([role]) => role)).toEqual(["assistant", "tool"]);
});

test("a text prompt sends the system message and one user message, as before", async () => {
  const { model } = await run(agent("Say ${message}"), []);

  const request = model.requests[0]!;
  // The request a text prompt sent before list prompts existed.
  expect(request.messages.map((message) => message.toJSON())).toEqual([
    { format: "yian.message.v1", role: "system", parts: [{ type: "text", text: "Be kind." }] },
    { format: "yian.message.v1", role: "user", parts: [{ type: "text", text: "Say again" }] },
  ]);
  expect([request.tools.length, request.toolChoice, request.parallelToolCalls, request.outputFormat]).toEqual([
    0,
    null,
    null,
    null,
  ]);
});

test.each([
  ["a str", "Hi"],
  ["a list holding a str", [{ role: "user", text: "Hi" }, "Hi"]],
  ["a system message", [{ role: "system", text: "Obey." }]],
  ["a message whose text isn't a str", [{ role: "user", text: 5n }]],
  ["a record without a role", [{ text: "Hi" }]],
])("splicing %s fails the node before any model call", async (_, history) => {
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const models = serving(model);

  const error = await rejected(agent(CHAT).forward({ history, message: "again" }, { models }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith(
    "node 'reply' prompt[0]: ${history} must be a message (a record with `role` user or assistant and " +
      "`text: str`) or a list of them",
  );
  expect(models.built).toBe(0);
});

test("a rendered list ending in an assistant message fails the node", async () => {
  const node = agent([{ role: "user", text: "Hi" }, { splice: "history" }]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const models = serving(model);

  const error = await rejected(node.forward({ history: HISTORY, message: "again" }, { models }));

  expect(error).toEqual(
    new ExpressionError("node 'reply' prompt: the last message must be from the user, got assistant"),
  );
  expect(models.built).toBe(0);
});

test("a rendered list with no message fails the node", async () => {
  const node = agent([{ splice: "history" }]);
  const models = serving(new ScriptedModel([textReply("done")], CONFIG));

  const error = await rejected(node.forward({ history: [], message: "again" }, { models }));

  expect(error).toEqual(new ExpressionError("node 'reply' prompt: the rendered list holds no message"));
});

test("recall, evidence and the render capture read the list's text forms", async () => {
  const recalled: string[] = [];
  const recorded: string[] = [];
  const captures: Readonly<Record<string, unknown>>[] = [];
  await run(agent(CHAT), HISTORY, {
    openMemoryContext: async (_system, prompt) => {
      recalled.push(prompt);
      return new MemoryContext();
    },
    evaluationTrace: {
      recordPrompt: (_system, prompt) => void recorded.push(prompt),
      recordInteraction: () => {},
    },
    captureRender: (captured) => void captures.push(captured),
  });

  expect(recalled).toEqual(["Now: again"]);
  expect(recorded).toEqual(["Now: again"]);
  expect(captures).toHaveLength(1);
  expect(captures[0]!["prompt_template"]).toBe(logJson(["${history}", { user: "Now: ${message}" }]));
  expect(captures[0]!["rendered_prompt"]).toBe(
    logJson([
      { role: "user", text: "Hi" },
      { role: "assistant", text: "Hello! How can I help?" },
      { role: "user", text: "Now: again" },
    ]),
  );
});

test("the node refuses a malformed list prompt", () => {
  expect(() => agent([])).toThrow(new RangeError("AGENT node 'reply': a list prompt must hold at least one item"));
  expect(() => agent([{ role: "system", text: "x" } as unknown as PromptItem])).toThrow(
    new TypeError("AGENT node 'reply': prompt[0] must be {role: user or assistant, text} or {splice: <name>}"),
  );
  expect(() => agent([{ role: "assistant", text: "x" }])).toThrow(
    new RangeError("AGENT node 'reply': the last message of a list prompt must be from the user"),
  );
});
