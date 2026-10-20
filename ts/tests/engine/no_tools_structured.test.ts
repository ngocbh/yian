/**
 * `runAgent` without tools: one structured call for a constrained output type, the text for a
 * bare `str`; mirrors `tests/engine/test_no_tools_structured.py`, driving `runAgent` directly
 * with a hand-built `AgentRunContext`.
 *
 * Ported: `test_no_tools_uses_structured_output_for_record` and
 * `test_no_tools_bare_str_text_passthrough`.
 *
 * Changed: the model is a `ScriptedModel` rather than a LangChain chat model whose
 * `with_structured_output` is or isn't called; the call's purpose (`structured_native` or
 * `chat`) shows the path instead. `test_no_tools_uses_ollama_structured_output_for_literal`: a
 * protocol-less model takes the native `json_schema` plan whatever its provider, so the row
 * asserts the structured purpose and the value for an `ollama` `ScriptedModel`.
 *
 * TS-only: `systemPromptWithMemory` with no sections returns the base unchanged, and with
 * sections appends Python's text and canonical JSON; a memory transcript is refused before the
 * model is built; a steering inbox's steers join the loop's request after the prompt; a granted
 * tool runs in the loop; a grant without a tool set reads the registry (`new ToolSet()`), so an
 * unregistered grant fails with `resolve`'s `RangeError` before the model call; a granted skill
 * binds `load_skill`; a reply with a parsed tool call and no grant fails at a cap of 1 with
 * `ToolIterationLimitError`, else with the ungranted `AgentLoopError`; an unparsed tool call is
 * ignored; a stop during the call wins over its text or tool-call reply, before any tool call is
 * recorded; `ctx.sent()` returns the built `RouteModel`'s route credentials; the prompt is
 * recorded in the evaluation trace.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted } from "../../src/events.ts";
import { BearerAuth } from "../../src/llm/auth.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import { Message, type Part, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type ConversationTurn,
  type EvaluationTrace,
  MemoryContext,
  MemorySection,
  ModelTextInteraction,
  ToolCallInteraction,
  type VisibleInteraction,
} from "../../src/memory/contracts.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  DEFAULT_SYSTEM,
  ToolIterationLimitError,
} from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { systemPromptWithMemory } from "../../src/nodes/agent/memory.ts";
import type { Steering } from "../../src/nodes/base.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { loadSkillTool } from "../../src/skills/tool.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };

function reply(...parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "stop" });
}

function textReply(body: string): Response {
  return reply(new TextPart(body));
}

/** A hand-built context over `model`, with `overrides`; `built` counts the model builds. */
function context(model: ChatModel, overrides: Partial<AgentRunContext> = {}): AgentRunContext & { built: number } {
  const ctx = {
    built: 0,
    nodeId: "a",
    prompt: "x",
    tools: [],
    llmConfig: null,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: {
      env: () => undefined,
      createModel: () => {
        ctx.built++;
        return model;
      },
    },
    sent: () => [],
    ...overrides,
  };
  return ctx;
}

/** The purposes of the `ModelCallStarted` events in `events`. */
function purposes(events: unknown[]): string[] {
  return events.filter((event) => event instanceof ModelCallStarted).map((event) => event.purpose);
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("the promise resolved");
}

const RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["name", Type.scalar(ValueKind.STRING)],
    ["score", Type.scalar(ValueKind.INTEGER)],
  ]),
  required: new Set(["name", "score"]),
});

test("test_no_tools_uses_structured_output_for_record", async () => {
  const model = new ScriptedModel([textReply('{"name": "Ada", "score": 9}')], NATIVE);
  const events: unknown[] = [];
  const ctx = context(model, { outputType: RECORD, emitEvent: (event) => void events.push(event) });
  const out = await runAgent(ctx);
  expect(out.value).toEqual({ name: "Ada", score: 9n });
  expect(model.requests).toHaveLength(1);
  expect(purposes(events)).toEqual(["structured_native"]);
});

test("test_no_tools_uses_ollama_structured_output_for_literal (Changed: the native plan)", async () => {
  const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "neutral"]) });
  const model = new ScriptedModel([textReply('{"value": "neutral"}')], { provider: "ollama", model: "qwen2.5:72b" });
  const events: unknown[] = [];
  const ctx = context(model, {
    prompt: "classify",
    llmConfig: { provider: "ollama", model: "qwen2.5:72b" },
    outputType: typ,
    emitEvent: (event) => void events.push(event),
  });
  expect((await runAgent(ctx)).value).toBe("neutral");
  expect(purposes(events)).toEqual(["structured_native"]);
  expect(model.requests[0]!.outputFormat).not.toBeNull();
});

test("test_no_tools_bare_str_text_passthrough", async () => {
  const model = new ScriptedModel([textReply("hello")]);
  const events: unknown[] = [];
  const ctx = context(model, { outputType: Type.scalar(ValueKind.STRING), emitEvent: (event) => void events.push(event) });
  expect((await runAgent(ctx)).value).toBe("hello");
  expect(purposes(events)).toEqual(["chat"]);
  const request = model.requests[0]!;
  expect(request.outputFormat).toBeNull();
  expect(request.tools).toEqual([]);
  expect(request.messages.map((message) => [message.role, message.text()])).toEqual([
    ["system", DEFAULT_SYSTEM],
    ["user", "x"],
  ]);
});

test("systemPromptWithMemory without sections returns the base unchanged", () => {
  expect(systemPromptWithMemory("base", [])).toBe("base");
});

test("systemPromptWithMemory appends Python's text and canonical JSON", () => {
  const sections = [new MemorySection("outer", 'notes "é"\n'), new MemorySection("local", "")];
  expect(systemPromptWithMemory("base", sections)).toBe(
    "base\n\nMemory context (canonical JSON; sections are ordered from outer workflow to local agent):\n" +
      '{"order":"outer-to-local","sections":[{"index":1,"memory":"outer","text":"notes \\"é\\"\\n"},' +
      '{"index":2,"memory":"local","text":""}]}',
  );
});

test("the memory sections follow the system prompt in the request", async () => {
  const model = new ScriptedModel([textReply("ok")]);
  const sections = [new MemorySection("m", "remember")];
  await runAgent(context(model, { baseSystemPrompt: "sys", memoryContext: new MemoryContext({ sections }) }));
  expect(model.requests[0]!.messages[0]!.text()).toBe(systemPromptWithMemory("sys", sections));
});

test("a memory transcript is refused before the model is built", async () => {
  const transcript = new MemoryContext({ transcript: {} as ConversationTurn });
  const ctx = context(new ScriptedModel([]), { memoryContext: transcript });
  const error = await rejection(runAgent(ctx));
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toBe("AGENT memory transcripts are not ported yet (T6-6)");
  expect(ctx.built).toBe(0);
});

test("a steering inbox's steers join the loop's request after the prompt", async () => {
  const model = new ScriptedModel([textReply("ok")]);
  const steering: Steering = { interruptSignal: () => new AbortController().signal, drain: () => ["be brief"] };
  expect((await runAgent(context(model, { steering }))).value).toBe("ok");
  expect(model.requests[0]!.messages.map((message) => [message.role, message.text()])).toEqual([
    ["system", DEFAULT_SYSTEM],
    ["user", "x"],
    ["user", "be brief"],
  ]);
});

const CALL = new ToolCallPart({ id: "c1", name: "lookup", arguments: { q: "x" } });

test("a granted tool runs in the loop", async () => {
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look a term up.",
    parametersJsonSchema: { type: "object", properties: { q: { type: "string" } } },
    execute: (args) => `found ${args.q}`,
  });
  const model = new ScriptedModel([reply(CALL), textReply("done")]);
  const ctx = context(model, { tools: ["lookup"], toolset: new ToolSet({ lookup }) });
  expect((await runAgent(ctx)).value).toBe("done");
  expect(model.requests[1]!.messages.at(-1)!.parts).toEqual([
    new ToolResultPart({ callId: "c1", name: "lookup", content: "found x" }),
  ]);
});

test("a granted tool without a tool set fails as unknown to the registry before the model call", async () => {
  const model = new ScriptedModel([textReply("unused")]);
  const error = await rejection(runAgent(context(model, { tools: ["no_such_registered_tool"] })));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toStartWith("unknown tool id 'no_such_registered_tool'; known tools: ");
  expect(model.requests).toEqual([]);
});

test("a granted skill binds load_skill", async () => {
  const model = new ScriptedModel([textReply("done")]);
  const skills = [new Skill({ name: "review", description: "Review code.", instructions: "" })];
  expect((await runAgent(context(model, { skills }))).value).toBe("done");
  expect(model.requests[0]!.tools).toEqual([loadSkillTool()]);
});

test("a parsed tool call at a cap of 1 fails with ToolIterationLimitError", async () => {
  const model = new ScriptedModel([reply(new TextPart("calling"), CALL, CALL)]);
  const error = await rejection(runAgent(context(model, { nodeId: "n", maxToolIterations: 1 })));
  expect(error).toBeInstanceOf(ToolIterationLimitError);
  expect((error as Error).message).toBe(
    "agent node 'n' hit the tool-iteration cap (1) without a final answer; the last reply's 2 tool call(s) were not run",
  );
});

test("a parsed tool call with no grant fails with AgentLoopError", async () => {
  const model = new ScriptedModel([reply(CALL)]);
  const interactions: VisibleInteraction[] = [];
  const trace: EvaluationTrace = { recordPrompt: () => {}, recordInteraction: (i) => void interactions.push(i) };
  const error = await rejection(runAgent(context(model, { nodeId: "n", evaluationTrace: trace })));
  expect(error).toBeInstanceOf(AgentLoopError);
  expect(error).not.toBeInstanceOf(ToolIterationLimitError);
  expect((error as Error).message).toBe("agent node 'n' requested tool 'lookup', but it is not granted; granted tools: []");
  expect(interactions).toEqual([new ToolCallInteraction("lookup", { q: "x" })]);
});

test("an unparsed tool call is ignored", async () => {
  const unparsed = new ToolCallPart({ id: "c1", name: "lookup", arguments: null, rawArguments: "{not json" });
  const model = new ScriptedModel([reply(new TextPart("answer"), unparsed)]);
  expect((await runAgent(context(model, { maxToolIterations: 1 }))).value).toBe("answer");
});

for (const [name, response] of [
  ["text", textReply("hello")],
  ["tool-call", reply(CALL)],
] as const) {
  test(`a stop during the call wins over its ${name} reply`, async () => {
    const stop = new AbortController();
    const interactions: VisibleInteraction[] = [];
    const trace: EvaluationTrace = { recordPrompt: () => {}, recordInteraction: (i) => void interactions.push(i) };
    const ctx = context(new ScriptedModel([response]), {
      signal: stop.signal,
      evaluationTrace: trace,
      emitEvent: (event) => {
        if (event instanceof ModelCallFinished) stop.abort(new Error("stopped"));
      },
    });
    expect(await rejection(runAgent(ctx))).toBe(stop.signal.reason);
    expect(interactions).toEqual([]);
  });
}

test("ctx.sent() returns the built RouteModel's route credentials", async () => {
  const route = new Route({
    provider: "openai",
    baseUrl: "https://api.example",
    path: "x",
    auth: new BearerAuth("sk-test-key-0123456789"),
  });
  const model = new RouteModel({ route, protocol: new OpenAIChat({ model: "gpt-5.5", provider: "openai" }) });
  const stop = new AbortController();
  // A stop right after the build ends the run before any request.
  const ctx = context(model, {
    signal: stop.signal,
    models: {
      env: () => undefined,
      createModel: () => {
        stop.abort(new Error("stopped"));
        return model;
      },
    },
  });
  expect(ctx.sent()).toEqual([]);
  expect(await rejection(runAgent(ctx))).toBe(stop.signal.reason);
  expect(ctx.sent()).toEqual(["sk-test-key-0123456789"]);
});

test("the prompt is recorded in the evaluation trace", async () => {
  const model = new ScriptedModel([textReply("hi")]);
  const prompts: unknown[] = [];
  const interactions: VisibleInteraction[] = [];
  const trace: EvaluationTrace = {
    recordPrompt: (...args) => void prompts.push(args),
    recordInteraction: (interaction) => void interactions.push(interaction),
  };
  const sections = [new MemorySection("m", "remember")];
  await runAgent(
    context(model, { baseSystemPrompt: "sys", prompt: "p", evaluationTrace: trace, memoryContext: new MemoryContext({ sections }) }),
  );
  expect(prompts).toEqual([["sys", "p", sections]]);
  expect(interactions).toEqual([new ModelTextInteraction("hi")]);
});
