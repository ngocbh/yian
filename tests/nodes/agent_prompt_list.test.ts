/**
 * An AGENT list prompt renders into the request's messages, after the system message. Driven
 * through `forward` with a `ScriptedModel`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { type AgentPrompt, AgentNode, type PromptItem } from "../../src/nodes/agent/node.ts";
import { type NodeCaps, Output } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import type { Skill } from "../../src/skills/catalog.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { callReply, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

/** The node's private render, which `forward` and `rebuildRequests` share. */
function renderPrompt(
  node: AgentNode,
  inputs: Readonly<Record<string, unknown>>,
  skills: ReadonlyMap<string, Skill> | null,
): AgentPrompt {
  type Inputs = Readonly<Record<string, unknown>>;
  type Render = { renderPrompt(inputs: Inputs, skills: ReadonlyMap<string, Skill> | null): AgentPrompt };
  return (node as unknown as Render).renderPrompt(inputs, skills);
}

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
  ["a Message whose part doesn't fit", [{ role: "user", parts: [{ type: "text" }] }]],
])("splicing %s fails the node before any model call", async (label, history) => {
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const models = serving(model);

  const error = await rejected(agent(CHAT).forward({ history, message: "again" }, { models }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith(
    "node 'reply' prompt[0]: ${history} must be a message (a `Message`, or a record with `role` user or assistant and " +
      "`text: str`) or a list of them",
  );
  // A value read as a `Message` says why it didn't fit.
  if (label === "a Message whose part doesn't fit") expect((error as Error).message).toContain("; ");
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

test("recall and the rendered prompt read the list's text forms", async () => {
  const recalled: string[] = [];
  await run(agent(CHAT), HISTORY, {
    openMemoryContext: async (_system, prompt) => {
      recalled.push(prompt);
      return new MemoryContext();
    },
  });

  expect(recalled).toEqual(["Now: again"]);
  expect(renderPrompt(agent(CHAT), { history: HISTORY, message: "again" }, null).promptText).toBe("Now: again");
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

const TOOL_HISTORY = [
  { role: "user", parts: [{ type: "text", text: "Read a.txt" }] },
  { role: "assistant", parts: [{ type: "tool_call", id: "c1", name: "read_file", arguments: { path: "a.txt" } }] },
  { role: "tool", parts: [{ type: "tool_result", call_id: "c1", name: "read_file", content: "hi" }] },
  { role: "assistant", parts: [{ type: "text", text: "It says hi." }] },
];

test("a spliced list of Messages reaches the request with every part and role", async () => {
  const { model } = await run(agent(CHAT), TOOL_HISTORY);

  expect(model.requests[0]!.messages.slice(1, 5).map((message) => message.toJSON())).toEqual(
    TOOL_HISTORY.map((message) => ({ format: "yian.message.v1", ...message })),
  );
  expect(sent(model).at(-1)).toEqual(["user", "Now: again"]);
});

test("Messages and {role, text} records splice together in order", async () => {
  const history = [{ role: "user", text: "Hi", at: "9:00" }, TOOL_HISTORY[3]];
  const { model } = await run(agent(CHAT), history);
  expect(sent(model).slice(1)).toEqual([["user", "Hi"], ["assistant", "It says hi."], ["user", "Now: again"]]);
});

test("a spliced Message ending the list must be the user's", async () => {
  const node = agent([{ role: "user", text: "Hi" }, { splice: "history" }]);
  const models = serving(new ScriptedModel([textReply("done")], CONFIG));
  const error = await rejected(node.forward({ history: TOOL_HISTORY, message: "x" }, { models }));
  expect(error).toEqual(new ExpressionError("node 'reply' prompt: the last message must be from the user, got assistant"));
  expect(models.built).toBe(0);
});

test("recall reads the last Message's text", async () => {
  const recalled: string[] = [];
  const reply = {
    role: "assistant",
    protocol: "anthropic_messages",
    parts: [{ type: "reasoning", text: "r", provider_payload: { sig: "s" } }, { type: "text", text: "It says hi." }],
  };
  const last = { role: "user", author: null, parts: [{ type: "text", text: "Go" }] };
  await run(agent([{ splice: "history" }]), [TOOL_HISTORY[0], reply, last], {
    openMemoryContext: async (_system, prompt) => (recalled.push(prompt), new MemoryContext()),
  });
  expect(recalled).toEqual(["Go"]);
});

test("a spliced media part fails the node before any model call", async () => {
  const history = [{ role: "user", parts: [{ type: "media", mime: "image/png", file: { uri: "media/a.png", mime: "image/png", name: null } }] }];
  const models = serving(new ScriptedModel([textReply("done")], CONFIG));
  const error = await rejected(agent(CHAT).forward({ history, message: "x" }, { models }));
  expect(error).toEqual(new ExpressionError("node 'reply' prompt[0]: ${history} holds a media part; no model route sends media yet"));
  expect(models.built).toBe(0);
});

test.each([
  ["a history ending on a tool call", CHAT, TOOL_HISTORY.slice(0, 2),
    "node 'reply' prompt[0]: rendered message 1: tool call 'c1' (read_file) has no result before the next user message"],
  ["a tool result with no call", CHAT, [TOOL_HISTORY[0], TOOL_HISTORY[2]],
    "node 'reply' prompt[0]: rendered message 1: tool result 'c1' answers no open tool call"],
  ["a tool call in a user message", CHAT,
    [{ role: "user", parts: [{ type: "tool_call", id: "c", name: "t", arguments: {} }] }],
    "node 'reply' prompt[0]: rendered message 0: user messages can't hold a tool_call part"],
  ["text in a tool message", CHAT,
    [TOOL_HISTORY[0], TOOL_HISTORY[1], { role: "tool", parts: [{ type: "text", text: "x" }] }],
    "node 'reply' prompt[0]: rendered message 2: tool messages can't hold a text part"],
  ["an empty Message", CHAT, [{ role: "assistant", parts: [] }],
    "node 'reply' prompt[0]: rendered message 0: the message has no parts"],
])("%s fails the node before any model call", async (_, prompt, history, text) => {
  const node = agent(prompt);
  const models = serving(new ScriptedModel([textReply("done")], CONFIG));
  const error = await rejected(node.forward({ history, message: "again" }, { models }));
  expect(error).toEqual(new ExpressionError(text));
  expect(models.built).toBe(0);
});

/** An AGENT `reply` like `agent`, reading the inputs `names`. */
function agentWith(prompt: readonly PromptItem[], names: readonly string[]): AgentNode {
  const node = new AgentNode("reply", { prompt, systemPrompt: "Be kind.", llmConfig: CONFIG });
  node.params = names.map((name) => new Parameter(name, { required: true }));
  return node;
}

const SPLIT: readonly PromptItem[] = [{ splice: "calls" }, { splice: "results" }, { role: "user", text: "Go" }];

test("a call and its result from two splice items pass", async () => {
  const model = new ScriptedModel([textReply("done")], CONFIG);
  await agentWith(SPLIT, ["calls", "results"]).forward(
    { calls: TOOL_HISTORY.slice(0, 2), results: [TOOL_HISTORY[2]] },
    { models: serving(model) },
  );
  expect(model.requests[0]!.messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "tool", "user"]);
});

test("a literal user message between a call and its result fails, naming the call's item", async () => {
  const prompt: PromptItem[] = [{ splice: "calls" }, { role: "user", text: "wait" }, { splice: "results" }, { role: "user", text: "Go" }];
  const models = serving(new ScriptedModel([textReply("done")], CONFIG));
  const error = await rejected(
    agentWith(prompt, ["calls", "results"]).forward({ calls: TOOL_HISTORY.slice(0, 2), results: [TOOL_HISTORY[2]] }, { models }),
  );
  expect(error).toEqual(new ExpressionError(
    "node 'reply' prompt[0]: rendered message 1: tool call 'c1' (read_file) has no result before the next user message",
  ));
  expect(models.built).toBe(0);
});

test("a spliced Gemini-style history repeating yian_call_0 across turns passes", async () => {
  const call = (id: string) => ({ type: "tool_call", id, name: "read_file", arguments: {} });
  const answer = (id: string) => ({ role: "tool", parts: [{ type: "tool_result", call_id: id, name: "read_file", content: "ok" }] });
  const history = [
    { role: "user", parts: [{ type: "text", text: "Read two files" }] },
    { role: "assistant", parts: [call("yian_call_0")] }, answer("yian_call_0"),
    { role: "assistant", parts: [call("yian_call_0"), call("yian_call_1")] }, answer("yian_call_1"), answer("yian_call_0"),
    { role: "assistant", parts: [{ type: "text", text: "Both read." }] },
  ];
  const { model } = await run(agent(CHAT), history);
  expect(model.requests[0]!.messages).toHaveLength(9);
});

test("a spliced Message's author reaches the request", async () => {
  const history = [{ role: "user", author: "ana", parts: [{ type: "text", text: "Hi" }] }];
  const { model } = await run(agent(CHAT), history);
  expect(model.requests[0]!.messages[1]!.toJSON().author).toBe("ana");
});

// --- declared fields at run time (#1149) --- //

const REC = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["k", Type.scalar(ValueKind.STRING)],
    ["opt", Type.scalar(ValueKind.STRING).replace({ nullable: true })],
  ]),
  required: new Set(["k"]),
});

/** Render `prompt` over input `r` (`Rec`, declared through `localTypes`) and return the user message. */
async function renderOverR(prompt: string | readonly PromptItem[], r: unknown): Promise<string> {
  const node = new AgentNode("reply", { prompt, systemPrompt: "Be kind.", llmConfig: CONFIG });
  node.params = [new Parameter("r", { required: true })];
  const model = new ScriptedModel([textReply("done")], CONFIG);
  await node.forward({ r }, { models: serving(model), localTypes: new Map([["r", REC]]) });
  return sent(model).at(-1)![1];
}

test("a prompt reads a declared field that is absent as null", async () => {
  expect(await renderOverR("${r.opt :- 'none'}", { k: "a" })).toBe("none");
  expect(await renderOverR([{ role: "user", text: "${r.opt :- 'none'}" }], { k: "a" })).toBe("none");
});

test("a prompt span that reads null through ?. still fails: prompts stay strict", async () => {
  await expect(renderOverR("${r?.k}", null)).rejects.toThrow(
    'null value in strict template: ${r?.k} is null; give it a default, e.g. ${r?.k :- "none"}',
  );
});
