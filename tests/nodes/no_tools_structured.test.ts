/**
 * `runAgent` without tools: one structured call for a constrained output type, the text for a bare
 * `str`; driving `runAgent` directly with a hand-built `AgentRunContext`.
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
import { MemoryContext, MemorySection } from "../../src/memory/contracts.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  DEFAULT_SYSTEM,
  ToolIterationLimitError,
} from "../../src/nodes/agent/context.ts";
import { Interrupted } from "../../src/nodes/agent/interrupts.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { systemPromptWithMemory } from "../../src/nodes/agent/memory.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output, SteerInbox } from "../../src/nodes/base.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { loadSkillTool } from "../../src/skills/tool.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

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
    messages: [userMessage("x")],
    promptText: "x",
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
      env: NO_ENV,
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

test("no tools uses structured output for record", async () => {
  const model = new ScriptedModel([textReply('{"name": "Ada", "score": 9}')], NATIVE);
  const events: unknown[] = [];
  const ctx = context(model, { outputType: RECORD, emitEvent: (event) => void events.push(event) });
  const out = await runAgent(ctx);
  expect((out as Output).value).toEqual({ name: "Ada", score: 9n });
  expect(model.requests).toHaveLength(1);
  expect(purposes(events)).toEqual(["structured_native"]);
});

test("no tools uses ollama structured output for a literal", async () => {
  const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "neutral"]) });
  const model = new ScriptedModel([textReply('{"value": "neutral"}')], { provider: "ollama", model: "qwen2.5:72b" });
  const events: unknown[] = [];
  const ctx = context(model, {
    messages: [userMessage("classify")],
    promptText: "classify",
    llmConfig: { provider: "ollama", model: "qwen2.5:72b" },
    outputType: typ,
    emitEvent: (event) => void events.push(event),
  });
  expect(((await runAgent(ctx)) as Output).value).toBe("neutral");
  expect(purposes(events)).toEqual(["structured_native"]);
  expect(model.requests[0]!.outputFormat).not.toBeNull();
});

test("no tools bare str text passthrough", async () => {
  const model = new ScriptedModel([textReply("hello")]);
  const events: unknown[] = [];
  const ctx = context(model, { outputType: Type.scalar(ValueKind.STRING), emitEvent: (event) => void events.push(event) });
  expect(((await runAgent(ctx)) as Output).value).toBe("hello");
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

test("systemPromptWithMemory appends the memory context heading and canonical JSON", () => {
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

test("a steer inbox's queued steers join the loop's request after the prompt", async () => {
  const model = new ScriptedModel([textReply("ok")]);
  let queued: string[] = ["be brief"];
  const steerInbox: SteerInbox = {
    inbox: "notes",
    interrupt: new AbortController().signal, interruptedBy: () => null,
    queued: () => queued.length > 0,
    read: () => {
      if (queued.length === 0) return null;
      const values = queued;
      queued = [];
      return { n: 1, from: [1], values };
    },
  };
  expect(((await runAgent(context(model, { steerInbox }))) as Output).value).toBe("ok");
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
  expect(((await runAgent(ctx)) as Output).value).toBe("done");
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
  expect(((await runAgent(context(model, { skills }))) as Output).value).toBe("done");
  expect(model.requests[0]!.tools).toEqual([loadSkillTool()]);
});

test("a parsed tool call at a cap of 1 fails with ToolIterationLimitError", async () => {
  const second = new ToolCallPart({ id: "c2", name: "lookup", arguments: { q: "y" } });
  const model = new ScriptedModel([reply(new TextPart("calling"), CALL, second)]);
  const error = await rejection(runAgent(context(model, { nodeId: "n", maxToolIterations: 1 })));
  expect(error).toBeInstanceOf(ToolIterationLimitError);
  expect((error as Error).message).toBe(
    "agent node 'n' hit the tool-iteration cap (1) without a final answer; the last reply's 2 tool call(s) were not run",
  );
});

test("a parsed tool call with no grant fails with AgentLoopError", async () => {
  const model = new ScriptedModel([reply(CALL)]);
  const error = await rejection(runAgent(context(model, { nodeId: "n" })));
  expect(error).toBeInstanceOf(AgentLoopError);
  expect(error).not.toBeInstanceOf(ToolIterationLimitError);
  expect((error as Error).message).toBe("agent node 'n' requested tool 'lookup', but it is not granted; granted tools: []");
});

test("an unparsed tool call is ignored", async () => {
  const unparsed = new ToolCallPart({ id: "c1", name: "lookup", arguments: null, rawArguments: "{not json" });
  const model = new ScriptedModel([reply(new TextPart("answer"), unparsed)]);
  expect(((await runAgent(context(model, { maxToolIterations: 1 }))) as Output).value).toBe("answer");
});

for (const [name, response] of [
  ["text", textReply("hello")],
  ["tool-call", reply(CALL)],
] as const) {
  test(`a stop during the call wins over its ${name} reply`, async () => {
    const stop = new AbortController();
    const ctx = context(new ScriptedModel([response]), {
      signal: stop.signal,
      emitEvent: (event) => {
        if (event instanceof ModelCallFinished) stop.abort(new Error("stopped"));
      },
    });
    expect(await rejection(runAgent(ctx))).toBe(stop.signal.reason);
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
      env: NO_ENV,
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

test("the memory sections follow the base system prompt", async () => {
  const model = new ScriptedModel([textReply("hi")]);
  const sections = [new MemorySection("m", "remember")];
  await runAgent(
    context(model, { baseSystemPrompt: "sys", messages: [userMessage("p")], promptText: "p", memoryContext: new MemoryContext({ sections }) }),
  );
  expect(model.requests[0]!.messages[0]!.text()).toBe(systemPromptWithMemory("sys", sections));
  expect(model.requests[0]!.messages[1]!.text()).toBe("p");
});

/** A steer inbox double holding `queued`, whose interrupt signal is `interrupt`, backed by record 9. */
function steerDouble(queued: string[], interrupt = new AbortController().signal): SteerInbox {
  return {
    inbox: "notes",
    interrupt,
    interruptedBy: () => (interrupt.aborted ? 9 : null),
    queued: () => queued.length > 0,
    read: () => (queued.length === 0 ? null : { n: 1, from: [1], values: queued.splice(0) }),
  };
}

test("a tool-free structured agent reads its steer inbox into its structured request", async () => {
  const model = new ScriptedModel([textReply('{"name": "Ada", "score": 9}')], NATIVE);
  const out = await runAgent(context(model, { outputType: RECORD, steerInbox: steerDouble(["be brief"]) }));
  expect((out as Output).value).toEqual({ name: "Ada", score: 9n });
  expect(model.requests[0]!.messages.map((message) => [message.role, message.text()])).toEqual([
    ["system", DEFAULT_SYSTEM],
    ["user", "x"],
    ["user", "be brief"],
  ]);
});

test("an interrupt before a tool-free structured agent's call honors it: no call, steers left queued", async () => {
  const model = new ScriptedModel([textReply('{"name": "Ada", "score": 9}')], NATIVE);
  const interrupt = new AbortController();
  interrupt.abort();
  const queued = ["be brief"];
  const out = await runAgent(context(model, { outputType: RECORD, steerInbox: steerDouble(queued, interrupt.signal) }));
  expect(out).toEqual(new Interrupted(9));
  expect(model.requests).toHaveLength(0);
  expect(queued).toEqual(["be brief"]);
});
