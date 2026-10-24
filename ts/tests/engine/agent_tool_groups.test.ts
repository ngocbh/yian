/**
 * Concurrent tool calls within one AGENT turn (D29, design 6.9); TS-only, no Python counterpart.
 *
 * Python runs a reply's calls one at a time. TS forms groups in call order: consecutive calls
 * whose tools say `isConcurrencySafe` (and `load_skill`) run together, an unsafe call runs alone,
 * and a failed check (an ungranted name, invalid arguments) ends the calls before it as their own
 * group, which runs and commits before the node fails. Results commit in call order whatever order
 * the calls finish in; a failing call aborts its group's siblings, and the node rejects only once
 * every started call has settled. A host interrupt never aborts a running group: it skips the calls
 * not started yet, a failed check after a committed group included, with interrupted brackets,
 * while a run stop still wins.
 *
 * What learning reads of each path comes from the steps it recorded (`stepsMaterial`): a reply's
 * runnable calls, then the results recorded for them in call order. So a failing group shows the
 * results its calls recorded, a failed check shows the call it refused with no result, and calls an
 * interrupt skipped show with no result.
 *
 * The tools here block on a gate the test settles, and log when they start and end; most rows run
 * `runAgent` directly with sinks for events and steps, and the host-callback and run-stop rows
 * run the node through `FlowEngine`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart, ToolSpec } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  MemoryContext,
  ModelTextInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  DEFAULT_SYSTEM,
  ToolIterationLimitError,
} from "../../src/nodes/agent/context.ts";
import { runAgent, stepsMaterial } from "../../src/nodes/agent/loop.ts";
import { systemPromptWithMemory } from "../../src/nodes/agent/memory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output, Steering } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { LOAD_SKILL, loadSkillTool } from "../../src/skills/tool.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { ReportedToolError, ToolCheckError } from "../../src/tools/errors.ts";
import { FunctionTool, type StandardSchemaV1 } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { aborted, engine, graph, rejected, serving, textReply } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** Let every pending promise job run: one macrotask turn. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

/** One tool whose calls block until the test settles them. */
interface Gated {
  readonly tool: FunctionTool;
  /** Resolves with the first call's context once its `execute` begins. */
  readonly entered: Promise<ToolContext>;
  /** The first call's context, once it began. */
  ctx: ToolContext | null;
  /** End the call with `value`. */
  resolve(value: unknown): void;
  /** End the call by throwing `error`. */
  reject(error: unknown): void;
}

/**
 * Return a tool `name` whose call logs `<name> start`, waits for the test (`resolve`/`reject`) or
 * its signal's abort, then logs `<name> end`. `safe` sets `isConcurrencySafe`; `onAbort` replaces
 * the abort: it runs once the signal aborts, and the call ends with what it returns or throws.
 */
function gated(
  name: string,
  log: string[],
  options: {
    safe?: boolean;
    parametersValidator?: StandardSchemaV1 | null;
    onAbort?: (signal: AbortSignal) => Promise<unknown>;
  } = {},
): Gated {
  const { safe = true, parametersValidator = null, onAbort = (signal) => aborted(signal) } = options;
  const gate = Promise.withResolvers<unknown>();
  const entered = Promise.withResolvers<ToolContext>();
  const gatedTool: Gated = {
    tool: new FunctionTool({
      name,
      description: `Tool ${name}.`,
      parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } } },
      isConcurrencySafe: safe ? () => true : null,
      parametersValidator,
      execute: async (_args, ctx) => {
        log.push(`${name} start`);
        gatedTool.ctx = ctx;
        entered.resolve(ctx);
        const abort = Promise.withResolvers<never>();
        const onAbortEvent = () => onAbort(ctx.signal).then(abort.resolve as (value: unknown) => void, abort.reject);
        if (ctx.signal.aborted) onAbortEvent();
        else ctx.signal.addEventListener("abort", onAbortEvent, { once: true });
        try {
          return await Promise.race([gate.promise, abort.promise]);
        } finally {
          ctx.signal.removeEventListener("abort", onAbortEvent);
          log.push(`${name} end`);
        }
      },
    }),
    entered: entered.promise,
    ctx: null,
    resolve: gate.resolve,
    reject: gate.reject,
  };
  return gatedTool;
}

/** A tool `name` that logs `<name>` and returns `<name> done` at once. */
function instant(name: string, log: string[], safe = true): FunctionTool {
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } } },
    isConcurrencySafe: safe ? () => true : null,
    execute: () => {
      log.push(name);
      return `${name} done`;
    },
  });
}

/** A reply calling `calls` (`[name, arguments]`) in order, ids `c0`, `c1`, ... */
function callsReply(...calls: readonly (readonly [string, JsonObject?])[]): Response {
  const parts = calls.map(([name, args = {}], index) => new ToolCallPart({ id: `c${index}`, name, arguments: args }));
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** A validator that logs `<name> validate` and reports one issue. */
function refusing(name: string, log: string[]): StandardSchemaV1 {
  return {
    "~standard": {
      version: 1,
      vendor: "test",
      validate: () => {
        log.push(`${name} validate`);
        return { issues: [{ message: "bad" }] };
      },
    },
  };
}

/**
 * One `runAgent` over `model` with recording sinks; `onEvent` also sees each event. `material()`
 * rebuilds what learning reads from the recorded steps, with the messages and tool specs the run
 * sent.
 */
function agent(
  model: ScriptedModel,
  toolset: ToolSet,
  tools: readonly string[],
  options: {
    signal?: AbortSignal;
    skills?: readonly Skill[];
    nodeId?: string;
    onEvent?: (event: unknown) => void;
    steering?: Steering;
    maxToolIterations?: number;
  } = {},
) {
  const events: unknown[] = [];
  const steps: ValueRecord[] = [];
  const ctx: AgentRunContext = {
    nodeId: options.nodeId ?? "n",
    messages: [userMessage("go")],
    promptText: "go",
    tools,
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: options.maxToolIterations ?? -1,
    signal: options.signal,
    steps: new NodeSteps(1, [], (step) => steps.push(step)),
    memoryContext: new MemoryContext(),
    toolset,
    emitEvent: (event) => {
      events.push(event);
      options.onEvent?.(event);
    },
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: options.skills ?? [],
    steering: options.steering,
    models: serving(model),
    sent: () => [],
  };
  const running = runAgent(ctx);
  let settled = false;
  running.then(
    () => (settled = true),
    () => (settled = true),
  );
  const material = () => {
    const system = new Message({ role: "system", parts: [new TextPart(systemPromptWithMemory(DEFAULT_SYSTEM, []))] });
    const specs = toolset
      .resolve(toolset.expand(tools))
      .map((tool) => new ToolSpec({ name: tool.name, description: tool.description, parameters: tool.parametersJsonSchema }));
    if (ctx.skills.length > 0) specs.push(loadSkillTool());
    return stepsMaterial(steps, { messages: [system, ...ctx.messages], tools: specs, structured: async () => null });
  };
  return { running, events, material, isSettled: () => settled };
}

/** What learning reads of `run`, checking that every recorded request matched its rebuild. */
async function learned(run: { material: ReturnType<typeof agent>["material"] }) {
  const { interactions, digestOk } = await run.material();
  expect(digestOk).toBe(true);
  return interactions;
}

/** A learner's view of a call to `name` with `args`. */
const called = (name: string, args: JsonObject = {}) => new ToolCallInteraction(name, args);

/** A learner's view of a result `text` from `name`. */
const returned = (name: string, text: string) => new ToolResultInteraction(name, text);

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

/** A host inbox with no steers whose `interrupt()` interrupts the node. */
function interruptible(): { steering: Steering; interrupt: () => void } {
  const controller = new AbortController();
  return {
    steering: { interruptSignal: () => controller.signal, drain: () => [] },
    interrupt: () => controller.abort(new Error("interrupted")),
  };
}

/** A tool result part for call `c<index>`. */
function result(index: number, name: string, content: string, isError = false): ToolResultPart {
  return new ToolResultPart({ callId: `c${index}`, name, content, isError });
}

test("two safe calls run together and commit in call order", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log);
  const model = new ScriptedModel([callsReply(["a", { n: 1 }], ["b", { n: 2 }]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  await b.entered;
  expect(log).toEqual(["a start", "b start"]);
  b.resolve("b result");
  await flush();
  a.resolve("a result");

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["a start", "b start", "b end", "a end"]);
  expect(toolResults(model, 1)).toEqual([result(0, "a", "a result"), result(1, "b", "b result")]);
  expect(await learned(run)).toEqual([
    called("a", { n: 1 }),
    called("b", { n: 2 }),
    returned("a", "a result"),
    returned("b", "b result"),
    new ModelTextInteraction("done"),
  ]);
  // Started in call order, finished in finish order.
  expect(toolEvents(run.events)).toEqual(["started a", "started b", "finished b b result", "finished a a result"]);
});

test("a host callback failing a group's second start aborts the first and never starts the third", async () => {
  const log: string[] = [];
  const fault = new Error("host boom");
  const release = Promise.withResolvers<void>();
  const firstAborted = Promise.withResolvers<void>();
  const a = gated("a", log, {
    onAbort: async (signal) => {
      firstAborted.resolve();
      await release.promise;
      throw signal.reason;
    },
  });
  const b = gated("b", log);
  const c = gated("c", log);
  let starts = 0;
  const onEvent = (event: unknown) => {
    if (event instanceof ToolCallStarted && ++starts === 2) throw fault;
  };
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const node = new AgentNode("agent", { prompt: "go", llmConfig: CONFIG, tools: ["a", "b", "c"] });
  const flow = graph([node], [
    [START_ID, "agent"],
    ["agent", END_ID],
  ]);
  const tools = new ToolSet({ a: a.tool, b: b.tool, c: c.tool });
  const running = engine(flow, { tools, models: serving(model), onEvent }).run();
  let settled = false;
  running.then(
    () => (settled = true),
    () => (settled = true),
  );

  await firstAborted.promise;
  await flush();
  expect(settled).toBe(false);
  release.resolve();

  expect(await rejected(running)).toBe(fault);
  expect(a.ctx!.signal.aborted).toBe(true);
  expect(log).toEqual(["a start", "a end"]);
  expect(model.requests).toHaveLength(1);
});

test("an event sink failing a group's second start starts no later call of the group", async () => {
  // Called directly, the sink's throw doesn't stop the node, so only the start guard holds `c`.
  const log: string[] = [];
  const fault = new Error("sink boom");
  const a = gated("a", log);
  const b = gated("b", log);
  const c = gated("c", log);
  let starts = 0;
  const onEvent = (event: unknown) => {
    if (event instanceof ToolCallStarted && ++starts === 2) throw fault;
  };
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool, c: c.tool }), ["a", "b", "c"], { onEvent });

  expect(await rejected(run.running)).toBe(fault);
  expect(a.ctx!.signal.reason).toBe(fault);
  expect(log).toEqual(["a start", "a end"]);
  expect(await learned(run)).toEqual([called("a"), called("b"), called("c")]);
});

test("a reported error in a group doesn't abort its sibling; both commit in call order", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log);
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  await b.entered;
  a.reject(new ReportedToolError("a is missing"));
  await flush();
  expect(b.ctx!.signal.aborted).toBe(false);
  b.resolve("b result");

  expect(((await run.running) as Output).value).toBe("done");
  expect(b.ctx!.signal.aborted).toBe(false);
  expect(toolResults(model, 1)).toEqual([result(0, "a", "a is missing", true), result(1, "b", "b result")]);
});

test("an unsafe call waits for the group before it, and the next call waits for it", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log, { safe: false });
  const c = gated("c", log);
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool, c: c.tool }), ["a", "b", "c"]);

  await a.entered;
  await flush();
  expect(log).toEqual(["a start"]);
  a.resolve("a result");
  await b.entered;
  await flush();
  expect(log).toEqual(["a start", "a end", "b start"]);
  b.resolve("b result");
  await c.entered;
  c.resolve("c result");

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["a start", "a end", "b start", "b end", "c start", "c end"]);
  expect(toolResults(model, 1).map((part) => part.content)).toEqual(["a result", "b result", "c result"]);
});

test("a tool without isConcurrencySafe runs alone", async () => {
  const log: string[] = [];
  const p = gated("p", log, { safe: false });
  const model = new ScriptedModel([callsReply(["p", { n: 1 }], ["p", { n: 2 }]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ p: p.tool }), ["p"]);

  await p.entered;
  await flush();
  expect(log).toEqual(["p start"]);
  // The gate is already settled for the second call, which starts only once the first ended.
  p.resolve("p result");

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["p start", "p end", "p start", "p end"]);
});

test("load_skill joins a group with safe tools; ask_user runs alone", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = instant("b", log);
  const skill = new Skill({ name: "deploy", description: "Ship a release.", instructions: "Tag, then publish." });
  const askUser = () => {
    log.push("ask_user");
    return "yes";
  };
  const model = new ScriptedModel(
    [callsReply(["a"], [LOAD_SKILL, { name: "deploy" }], ["ask_user", { question: "ok?" }], ["b"]), textReply("done")],
    CONFIG,
  );
  const toolset = new ToolSet({ a: a.tool, b, ask_user: askUser });
  const run = agent(model, toolset, ["a", "b", "ask_user"], { skills: [skill] });

  await a.entered;
  await flush();
  // load_skill started (and finished) beside the running `a`; ask_user waits.
  const skillEvents = [`started ${LOAD_SKILL}`, `finished ${LOAD_SKILL} Tag, then publish.`];
  expect(toolEvents(run.events)).toEqual(["started a", ...skillEvents]);
  expect(log).toEqual(["a start"]);
  a.resolve("a result");

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual(["a start", "a end", "ask_user", "b"]);
  expect(toolResults(model, 1).map((part) => part.content)).toEqual(["a result", "Tag, then publish.", "yes", "b done"]);
});

test("an ungranted call fails the node after the group before it ran and committed", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const model = new ScriptedModel([callsReply(["a"], ["x"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, x: instant("x", log) }), ["a"]);

  await a.entered;
  a.resolve("a result");

  const error = await rejected(run.running);
  expect(error).toBeInstanceOf(AgentLoopError);
  expect((error as Error).message).toContain("requested tool 'x', but it is not granted");
  expect(log).toEqual(["a start", "a end"]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a result"]);
  // The reply's calls, then the result the group before the refused call recorded.
  expect(await learned(run)).toEqual([called("a"), called("x"), returned("a", "a result")]);
});

test("invalid arguments fail the node after the group before them ran and committed", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log);
  const model = new ScriptedModel([callsReply(["a"], ["b", { bogus: 1 }]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  a.resolve("a result");

  const error = await rejected(run.running);
  expect(error).toBeInstanceOf(RangeError);
  expect(log).toEqual(["a start", "a end"]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a result", "started b", "finished b RangeError"]);
  expect(await learned(run)).toEqual([called("a"), called("b", { bogus: 1 }), returned("a", "a result")]);
});

test("a call after an unsafe one is validated only once the unsafe call ran", async () => {
  const log: string[] = [];
  const a = gated("a", log, { safe: false });
  const b = gated("b", log, { parametersValidator: refusing("b", log) });
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  await flush();
  expect(log).toEqual(["a start"]);
  a.resolve("a result");

  expect(await rejected(run.running)).toBeInstanceOf(ToolCheckError);
  expect(log).toEqual(["a start", "a end", "b validate"]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a result", "started b", "finished b ToolCheckError"]);
});

test("a reported error from the argument check is the call's error result, and the loop goes on", async () => {
  const log: string[] = [];
  const validator: StandardSchemaV1 = {
    "~standard": {
      version: 1,
      vendor: "test",
      validate: () => {
        throw new ReportedToolError("no such file");
      },
    },
  };
  const a = gated("a", log, { safe: false, parametersValidator: validator });
  const model = new ScriptedModel([callsReply(["a"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool }), ["a"]);

  expect(((await run.running) as Output).value).toBe("done");
  expect(log).toEqual([]);
  expect(toolResults(model, 1)).toEqual([result(0, "a", "no such file", true)]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a ReportedToolError"]);
  expect(await learned(run)).toEqual([called("a"), returned("a", "no such file"), new ModelTextInteraction("done")]);
});

test("nothing in a group starts until the group is formed", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log, { parametersValidator: refusing("b", log) });
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  // B's check ran before A started; it failed, so A runs as its own group.
  expect(log).toEqual(["b validate", "a start"]);
  a.resolve("a result");

  expect(await rejected(run.running)).toBeInstanceOf(ToolCheckError);
  expect(log).toEqual(["b validate", "a start", "a end"]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a result", "started b", "finished b ToolCheckError"]);
});

test("a failing call aborts its siblings, and the node rejects with it once all settled", async () => {
  const log: string[] = [];
  const failure = new Error("b boom");
  const release = Promise.withResolvers<void>();
  const aAborted = Promise.withResolvers<void>();
  // `a` ends with its own error, later than the failure.
  const a = gated("a", log, {
    onAbort: async () => {
      aAborted.resolve();
      await release.promise;
      throw new TypeError("a cancelled");
    },
  });
  const b = gated("b", log);
  const c = gated("c", log);
  const d = gated("d", log, { safe: false });
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"], ["d"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool, c: c.tool, d: d.tool }), ["a", "b", "c", "d"]);

  await a.entered;
  await b.entered;
  await c.entered;
  b.reject(failure);
  await aAborted.promise;
  await flush();
  expect(run.isSettled()).toBe(false);
  expect(a.ctx!.signal.reason).toBe(failure);
  expect(c.ctx!.signal.reason).toBe(failure);
  release.resolve();

  expect(await rejected(run.running)).toBe(failure);
  expect(log).toEqual(["a start", "b start", "c start", "b end", "c end", "a end"]);
  expect(toolEvents(run.events)).toEqual([
    "started a",
    "started b",
    "started c",
    "finished b Error",
    "finished c Error",
    "finished a TypeError",
  ]);
  expect(model.requests).toHaveLength(1);
  expect(await learned(run)).toEqual([called("a"), called("b"), called("c"), called("d")]);
});

test("a failing group shows learning the results its calls recorded before the failure", async () => {
  const log: string[] = [];
  const failure = new Error("b boom");
  const a = gated("a", log);
  const b = gated("b", log);
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"]);

  await a.entered;
  await b.entered;
  a.resolve("a result");
  await flush();
  b.reject(failure);

  expect(await rejected(run.running)).toBe(failure);
  expect(log).toEqual(["a start", "b start", "a end", "b end"]);
  expect(await learned(run)).toEqual([called("a"), called("b"), returned("a", "a result")]);
});

test.each([
  ["the context's signal", (stop: AbortSignal) => ({ toolset: (tools: ToolSet) => tools, signal: stop })],
  [
    "the tool set's run signal",
    (stop: AbortSignal) => ({ toolset: (tools: ToolSet) => tools.forRun("r1", { signal: stop }), signal: undefined }),
  ],
] as const)("a stop after a group's first failure wins over it (%s)", async (_label, wiring) => {
  const log: string[] = [];
  const stop = new AbortController();
  const failure = new Error("b boom");
  const reason = new Error("stop now");
  const release = Promise.withResolvers<void>();
  const aAborted = Promise.withResolvers<void>();
  const a = gated("a", log, {
    onAbort: async (signal) => {
      aAborted.resolve();
      await release.promise;
      throw signal.reason;
    },
  });
  const b = gated("b", log);
  const { toolset, signal } = wiring(stop.signal);
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, toolset(new ToolSet({ a: a.tool, b: b.tool })), ["a", "b"], { signal });

  await a.entered;
  await b.entered;
  b.reject(failure);
  await aAborted.promise;
  // `a` is still pending, aborted by the failure, when the node stops.
  stop.abort(reason);
  await flush();
  expect(run.isSettled()).toBe(false);
  release.resolve();

  expect(await rejected(run.running)).toBe(reason);
  expect(model.requests).toHaveLength(1);
  expect(await learned(run)).toEqual([called("a"), called("b")]);
});

test("a settled group's signal no longer follows the node's signal", async () => {
  const log: string[] = [];
  const stop = new AbortController();
  const a = gated("a", log);
  const b = gated("b", log);
  const model = new ScriptedModel([callsReply(["a"]), callsReply(["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"], { signal: stop.signal });

  a.resolve("a result");
  b.resolve("b result");
  expect(((await run.running) as Output).value).toBe("done");
  stop.abort(new Error("late stop"));

  // A listener left on the node's signal would abort these.
  expect(a.ctx!.signal.aborted).toBe(false);
  expect(b.ctx!.signal.aborted).toBe(false);
  expect(a.ctx!.signal).not.toBe(b.ctx!.signal);
});

test("a run stop during a group awaits its started calls and starts no later group", async () => {
  const log: string[] = [];
  const host = new AbortController();
  const reason = new Error("stop now");
  const release = Promise.withResolvers<void>();
  const bAborted = Promise.withResolvers<void>();
  const a = gated("a", log);
  const b = gated("b", log, {
    onAbort: async (signal) => {
      bAborted.resolve();
      await release.promise;
      throw signal.reason;
    },
  });
  const c = gated("c", log, { safe: false });
  const events: unknown[] = [];
  const aFinished = Promise.withResolvers<void>();
  const onEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ToolCallFinished && event.name === "a") aFinished.resolve();
  };
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const node = new AgentNode("agent", { prompt: "go", llmConfig: CONFIG, tools: ["a", "b", "c"] });
  const flow = graph([node], [
    [START_ID, "agent"],
    ["agent", END_ID],
  ]);
  const tools = new ToolSet({ a: a.tool, b: b.tool, c: c.tool });
  const running = engine(flow, { tools, models: serving(model), onEvent, signal: host.signal }).run();
  let settled = false;
  running.then(
    () => (settled = true),
    () => (settled = true),
  );

  await a.entered;
  await b.entered;
  a.resolve("a result");
  await aFinished.promise;
  host.abort(reason);
  await bAborted.promise;
  await flush();
  expect(settled).toBe(false);
  release.resolve();

  expect(await rejected(running)).toBe(reason);
  expect(log).toEqual(["a start", "b start", "a end", "b end"]);
  expect(toolEvents(events)).toEqual(["started a", "started b", "finished a a result"]);
});

test("each call gets its own call id, from its reply's model_call step, node and run id", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log);
  const stop = new AbortController();
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const toolset = new ToolSet({ a: a.tool, b: b.tool }).forRun("r1");
  const run = agent(model, toolset, ["a", "b"], { signal: stop.signal, nodeId: "writer" });

  await a.entered;
  await b.entered;
  a.resolve("a result");
  b.resolve("b result");
  await run.running;

  // The key names the reply's `model_call` step, its first: `<run>/<node>/<n>/<index>`.
  const fields = (ctx: ToolContext) => [ctx.callId, ctx.node, ctx.runId, ctx.toolName, ctx.attempt];
  expect(fields(a.ctx!)).toEqual(["r1/writer/1/0", "writer", "r1", "a", 1]);
  expect(fields(b.ctx!)).toEqual(["r1/writer/1/1", "writer", "r1", "b", 1]);
  // One group: both calls share its signal, a child of the node's.
  expect(a.ctx!.signal).toBe(b.ctx!.signal);
  expect(a.ctx!.signal).not.toBe(stop.signal);
});

test("an interrupt while a safe group runs lets every call of it finish and skips only later calls", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log);
  const { steering, interrupt } = interruptible();
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool, c: instant("c", log, false) }), ["a", "b", "c"], {
    steering,
  });

  await a.entered;
  await b.entered;
  interrupt();
  await flush();
  expect(a.ctx!.signal.aborted).toBe(false);
  a.resolve("a result");
  b.resolve("b result");

  expect(((await run.running) as Output).value).toBe("");
  expect(log).toEqual(["a start", "b start", "a end", "b end"]);
  expect(model.requests).toHaveLength(1);
  expect(toolEvents(run.events)).toEqual([
    "started a",
    "started b",
    "finished a a result",
    "finished b b result",
    "started c",
    "finished c interrupted",
  ]);
  // The skipped call shows with no result.
  expect(await learned(run)).toEqual([
    called("a"),
    called("b"),
    called("c"),
    returned("a", "a result"),
    returned("b", "b result"),
  ]);
});

test("an interrupt seen at a later call's check skips it and the rest", async () => {
  const log: string[] = [];
  const a = gated("a", log, { safe: false });
  const { steering, interrupt } = interruptible();
  const model = new ScriptedModel([callsReply(["a"], ["b"], ["c"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: instant("b", log), c: instant("c", log) }), ["a", "b", "c"], {
    steering,
  });

  await a.entered;
  interrupt();
  a.resolve("a result");

  expect(((await run.running) as Output).value).toBe("");
  expect(log).toEqual(["a start", "a end"]);
  expect(toolEvents(run.events)).toEqual([
    "started a",
    "finished a a result",
    "started b",
    "finished b interrupted",
    "started c",
    "finished c interrupted",
  ]);
  expect(await learned(run)).toEqual([called("a"), called("b"), called("c"), returned("a", "a result")]);
});

test("an interrupt inside a group before a failed check commits the group and skips the failed call", async () => {
  const log: string[] = [];
  const a = gated("a", log);
  const b = gated("b", log, { parametersValidator: refusing("b", log) });
  const { steering, interrupt } = interruptible();
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"], { steering });

  await a.entered;
  interrupt();
  a.resolve("a result");

  expect(((await run.running) as Output).value).toBe("");
  expect(log).toEqual(["b validate", "a start", "a end"]);
  expect(toolEvents(run.events)).toEqual(["started a", "finished a a result", "started b", "finished b interrupted"]);
  expect(await learned(run)).toEqual([called("a"), called("b"), returned("a", "a result")]);
});

test("a run stop inside a group before a failed check rejects with the stop, and the failed call gets no brackets", async () => {
  const log: string[] = [];
  const reason = new Error("stopped");
  const stop = new AbortController();
  const a = gated("a", log, { onAbort: async () => "a result" });
  const b = gated("b", log, { parametersValidator: refusing("b", log) });
  const model = new ScriptedModel([callsReply(["a"], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: a.tool, b: b.tool }), ["a", "b"], {
    signal: stop.signal,
    steering: interruptible().steering,
  });

  await a.entered;
  stop.abort(reason);

  expect(await rejected(run.running)).toBe(reason);
  expect(log).toEqual(["b validate", "a start", "a end"]);
  expect(toolEvents(run.events)).toEqual(["started a"]);
  expect(await learned(run)).toEqual([called("a"), called("b")]);
});

test("a reply at the tool-iteration cap fails the node, running none of its calls, and learning reads its calls", async () => {
  const log: string[] = [];
  const model = new ScriptedModel([callsReply(["a", { n: 1 }], ["b"]), textReply("done")], CONFIG);
  const run = agent(model, new ToolSet({ a: instant("a", log), b: instant("b", log) }), ["a", "b"], {
    maxToolIterations: 1,
  });

  expect(await rejected(run.running)).toBeInstanceOf(ToolIterationLimitError);
  expect(log).toEqual([]);
  expect(await learned(run)).toEqual([called("a", { n: 1 }), called("b")]);
});
