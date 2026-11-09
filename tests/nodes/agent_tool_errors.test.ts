/**
 * An AGENT answers its model's mistakes with error tool results the model reads, and the loop goes
 * on: every tool call of a reply gets a result, so the next request answers each one. A call whose
 * arguments aren't valid JSON, that names a tool the agent wasn't granted, or whose arguments fail
 * their check (the schema, or the validator's issues) never reaches its tool; it runs alone, with
 * started and finished events, and its error result is written as a `tool_result` step. A
 * validator that throws something other than its issues still fails the node.
 *
 * Each path's requests rebuild from the steps it recorded (`rebuildStepRequests`), each matching
 * the digest the call recorded. Rows run `runAgent` directly with sinks for events and steps.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, type Part, Response, TextPart, ToolCallPart, ToolResultPart, ToolSpec } from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { rebuildStepRequests, runAgent } from "../../src/nodes/agent/loop.ts";
import { checkModelConversation } from "../../src/nodes/agent/messages.ts";
import { unavailableText, unparsedText } from "../../src/nodes/agent/results.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { FunctionTool, type StandardSchemaV1 } from "../../src/tools/function_tools.ts";
import { asToolGrant, expandGrants, type GrantedTool, requestToolSpecs } from "../../src/tools/grants.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/**
 * A tool `name` taking an optional int `n`, which logs `<name>` and returns `<name> done`;
 * `validator` is its parameters validator.
 */
function instant(name: string, log: string[], validator: StandardSchemaV1 | null = null): FunctionTool {
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } } },
    parametersValidator: validator,
    execute: () => {
      log.push(name);
      return `${name} done`;
    },
  });
}

/** A call `c<index>` to `name`: with `args` parsed, or with `args` as raw text that didn't parse. */
function call(index: number, name: string, args: JsonObject | string = {}): ToolCallPart {
  const id = `c${index}`;
  if (typeof args === "string") return new ToolCallPart({ id, name, arguments: null, rawArguments: args });
  return new ToolCallPart({ id, name, arguments: args });
}

/** A reply holding `parts`. */
function reply(...parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/**
 * One `runAgent` over `model`, granting `tools`, with sinks for its events and steps. `material()`
 * rebuilds its requests from the recorded steps, with the messages and tool specs the run sent.
 */
function agent(model: ScriptedModel, toolset: ToolSet, tools: readonly string[], options: { maxToolIterations?: number } = {}) {
  const events: unknown[] = [];
  const steps: ValueRecord[] = [];
  const ctx: AgentRunContext = {
    nodeId: "n",
    messages: [userMessage("go")],
    tools,
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: options.maxToolIterations ?? -1,
    steps: new NodeSteps(1, [], (step) => steps.push(step)),
    toolset,
    emitEvent: (event) => void events.push(event),
    systemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
  };
  const material = () => {
    const system = new Message({ role: "system", parts: [new TextPart(DEFAULT_SYSTEM)] });
    const specOf = (granted: GrantedTool) => {
      const tool = toolset.resolve([{ group: granted.group, name: granted.tool }])[0]!;
      return new ToolSpec({ name: tool.name, description: tool.description, parameters: tool.parametersJsonSchema });
    };
    const specs = requestToolSpecs(expandGrants(tools.map(asToolGrant), toolset.scopes()), specOf, null)!;
    return rebuildStepRequests(steps, { messages: [system, ...ctx.messages], tools: specs, structured: async () => null });
  };
  return { running: runAgent(ctx), events, steps, material };
}

/** Check that every request `run` sent matches the one rebuilt from its steps. */
async function expectRebuilt(run: { material: ReturnType<typeof agent>["material"] }, sent: number) {
  const { requests, digestOk } = await run.material();
  expect(digestOk).toBe(true);
  expect(requests.size).toBe(sent);
}

/** The tool events among `events`, as `started <name>` and `finished <name> <result or error>`. */
function toolEvents(events: readonly unknown[]): string[] {
  return events.flatMap((event) => {
    if (event instanceof ToolCallStarted) return [`started ${event.name}`];
    if (event instanceof ToolCallFinished) return [`finished ${event.name} ${event.error ?? event.result}`];
    return [];
  });
}

/** The tool result parts the model's `index`-th request carried. */
function toolResults(model: ScriptedModel, index: number): ToolResultPart[] {
  return model.requests[index]!.messages.filter((message) => message.role === "tool").map(
    (message) => message.parts[0] as ToolResultPart,
  );
}

/** A tool result part for call `c<index>`. */
function result(index: number, name: string, content: string, isError = false): ToolResultPart {
  return new ToolResultPart({ callId: `c${index}`, name, content, isError });
}

/** The `tool_result` steps among `steps`: call id, output and whether an error. */
function resultSteps(steps: readonly ValueRecord[]): unknown[] {
  return steps.filter((step) => step["kind"] === "tool_result").map(({ call_id, output, is_error }) => [call_id, output, is_error]);
}

test("an unparsed call gets an error result the model reads, never reaching its tool, and the loop goes on", async () => {
  const log: string[] = [];
  const model = new ScriptedModel([reply(call(0, "a", "{oops"), call(1, "b", { n: 1 })), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: instant("a", log), b: instant("b", log) }), ["a", "b"]);

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["b"]);
  const text = "Arguments for a are not valid JSON; send them as one JSON object.";
  expect(unparsedText("a")).toBe(text);
  expect(toolResults(model, 1)).toEqual([result(0, "a", text, true), result(1, "b", "b done")]);
  expect(resultSteps(run.steps)).toEqual([
    ["c0", text, true],
    ["c1", "b done", false],
  ]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a ReportedToolError", "started b", "finished b b done"]);
  const started = run.events.filter((event) => event instanceof ToolCallStarted);
  expect(started.map((event) => [event.callIndex, event.arguments])).toEqual([
    [0, null],
    [1, { n: 1 }],
  ]);
  await expectRebuilt(run, 2);
});

test("the request after an unparsed call answers every call, through the Anthropic and OpenAI Chat adapters", async () => {
  const log: string[] = [];
  const calls = [call(0, "a", { n: 1 }), call(1, "a", '{"n": '), call(2, "a", { n: 2 })];
  const model = new ScriptedModel([reply(new TextPart("checking"), ...calls), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: instant("a", log) }), ["a"]);

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["a", "a"]);
  const request = model.requests[1]!;
  expect(() => checkModelConversation(request.messages.slice(1))).not.toThrow();

  const anthropic = new AnthropicMessages({ model: "claude-sonnet-4-5", provider: "anthropic" }).buildBody(request);
  const blocks = (anthropic["messages"] as JsonObject[]).flatMap((message) =>
    Array.isArray(message["content"]) ? (message["content"] as JsonObject[]) : [],
  );
  const ids = (type: string, key: string) => blocks.filter((block) => block["type"] === type).map((block) => block[key]);
  expect(ids("tool_use", "id")).toEqual(["c0", "c1", "c2"]);
  expect(ids("tool_result", "tool_use_id")).toEqual(["c0", "c1", "c2"]);

  const chat = new OpenAIChat({ model: "gpt-5.5", provider: "openai" }).buildBody(request);
  const messages = chat["messages"] as JsonObject[];
  const asked = messages.flatMap((message) => (message["tool_calls"] as JsonObject[] | undefined) ?? []).map((each) => each["id"]);
  const answered = messages.filter((message) => message["role"] === "tool").map((message) => message["tool_call_id"]);
  // The adapter sends the unparsed call after the others; each still has its result.
  expect(asked).toEqual(["c0", "c2", "c1"]);
  expect(answered).toEqual(["c0", "c1", "c2"] as JsonValue[]);
  await expectRebuilt(run, 2);
});

test("a reply whose calls are all unparsed is a tool turn: the model reads the errors and answers", async () => {
  const log: string[] = [];
  const model = new ScriptedModel([reply(new TextPart("let me look"), call(0, "a", "{")), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: instant("a", log) }), ["a"]);

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual([]);
  expect(toolResults(model, 1)).toEqual([result(0, "a", unparsedText("a"), true)]);
  await expectRebuilt(run, 2);
});

for (const [what, name] of [
  ["a tool the agent wasn't granted", "x"],
  ["a tool that doesn't exist", "nope"],
] as const) {
  test(`a call to ${what} gets "Unavailable tool", never reaching a tool, and the loop goes on`, async () => {
    const log: string[] = [];
    const model = new ScriptedModel([reply(call(0, "a"), call(1, name, { n: 1 })), textReply("done")], CONFIG);
    const run = agent(model, new ToolSet({ a: instant("a", log), x: instant("x", log) }), ["a"]);

    expect(((await run.running) as Output).value).toBe("done");
    expect(log).toEqual(["a"]);
    expect(unavailableText(name)).toBe(`Unavailable tool: ${name}`);
    expect(toolResults(model, 1)).toEqual([result(0, "a", "a done"), result(1, name, `Unavailable tool: ${name}`, true)]);
    expect(resultSteps(run.steps)).toEqual([
      ["c0", "a done", false],
      ["c1", `Unavailable tool: ${name}`, true],
    ]);
    expect(toolEvents(run.events)).toEqual(["started a", "finished a a done", `started ${name}`, `finished ${name} ReportedToolError`]);
    const started = run.events.filter((event) => event instanceof ToolCallStarted).at(-1)!;
    expect([started.n, started.callIndex, started.toolCallId, started.arguments]).toEqual([1, 1, "c1", { n: 1 }]);
    await expectRebuilt(run, 2);
  });
}

/** A validator that reports one issue at `n`, or throws `thrown` when given. */
function validator(thrown?: Error): StandardSchemaV1 {
  return {
    "~standard": {
      version: 1,
      vendor: "test",
      validate: () => {
        if (thrown !== undefined) throw thrown;
        return { issues: [{ message: "too big", path: ["n"] }] };
      },
    },
  };
}

for (const [what, args, tool, message] of [
  ["an argument the schema lacks", { bogus: 1 }, null, `tool 'b' received unsupported argument(s) ["bogus"]; allowed: ["n"]`],
  ["an argument of the wrong type", { n: "one" }, null, `tool 'b' argument 'n': "one" does not match declared type int`],
  ["the validator's issues", { n: 9 }, validator(), "tool 'b' arguments failed its validator: n: too big"],
] as const) {
  test(`a failed argument check (${what}): the model reads its message as the call's error result, and the loop goes on`, async () => {
    const log: string[] = [];
    const model = new ScriptedModel([reply(call(0, "a"), call(1, "b", args)), textReply("done")], CONFIG);
    const run = agent(model, new ToolSet({ a: instant("a", log), b: instant("b", log, tool) }), ["a", "b"]);

    expect(((await run.running) as Output).value).toBe("done");
    expect(log).toEqual(["a"]);
    expect(toolResults(model, 1)).toEqual([result(0, "a", "a done"), result(1, "b", message, true)]);
    expect(resultSteps(run.steps)).toEqual([
      ["c0", "a done", false],
      ["c1", message, true],
    ]);
    expect(toolEvents(run.events)).toEqual(["started a", "finished a a done", "started b", "finished b ReportedToolError"]);
    await expectRebuilt(run, 2);
  });
}

test("a validator that throws something other than its issues fails the node once the calls before it ran", async () => {
  const log: string[] = [];
  const fault = new Error("validator down");
  const model = new ScriptedModel([reply(call(0, "a"), call(1, "b")), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: instant("a", log), b: instant("b", log, validator(fault)) }), ["a", "b"]);

  expect(await rejected(run.running)).toBe(fault);
  expect(log).toEqual(["a"]);
  expect(resultSteps(run.steps)).toEqual([["c0", "a done", false]]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a done", "started b", "finished b Error"]);
  await expectRebuilt(run, 1);
});
