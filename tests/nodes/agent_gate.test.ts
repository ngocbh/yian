/**
 * The host's check before each tool call an agent's model makes (`beforeToolCall`): it runs a call,
 * with an audit note the model never reads, or refuses it with a text the model reads as the
 * call's error. It is asked about a whole group before any of the group's calls starts, and a call
 * never starts once the node stopped or the agent took an interrupt while the hook decided. Every
 * source is asked about: host, MCP, module, memory and flow tools, a refused flow call starting no
 * flow, and the agents inside a granted flow with the flow calls enclosing them.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runInNewContext } from "node:vm";

import { END_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { CallHeld, NodeFailed, NodeSucceeded, NodeWaiting, NodeWoken, type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, AgentLoopError, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { Interrupted } from "../../src/nodes/agent/interrupts.ts";
import { SKIPPED_TOOL_RESULT, TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/results.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { rebuildMessages } from "../../src/nodes/agent/messages.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { type CallView, NodeSteps } from "../../src/nodes/contracts.ts";
import { withWarnings } from "../../src/platform.ts";
import { advance, agentCall, complete, interrupt, pendingCalls, release, runFlow } from "../../src/run.ts";
import { NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool, tool, type ToolOptions } from "../../src/tools/function_tools.ts";
import { type BeforeToolCall, GATE_FAILED_TEXT, type GatedCall, type GateDecision, type ToolSource } from "../../src/tools/gate.ts";
import { nodeScopeKey } from "../../src/tools/grants.ts";
import { registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { gated } from "../helpers/gate.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { FakeMemory, fileInfo, RecordingSession } from "../helpers/memory_fakes.ts";
import { callReply, crashAfter, interruptible, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-gate-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);
const env = hostEnv(dir);
const CONFIG = { provider: "openai", model: "test-model" };

/** One agent `reply` granted `grants`, reading steer inbox `m`. */
const flowOf = (grants: string) =>
  loadFlow(`
id: gate
name: gate
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [${grants}]
    steer_inbox: \${input.m}
    output: !Optional(str)
output: \${reply.output}
`);

/**
 * Host tools noting each run in `ran`: `lookup` (resumable) and the group `team`'s `a`, safe to
 * run together, and `wait`, which returns "stopped" once its call learns of an interrupt.
 */
function hostTools(ran: string[], began: () => void = () => {}) {
  const safe = (name: string, resumable: boolean) =>
    tool({
      name,
      description: `Tool ${name}.`,
      parameters: { type: "object", properties: { q: { type: "string" } } },
      isConcurrencySafe: () => true,
      resumable,
      execute: (args) => {
        ran.push(name);
        return `${name}:${String(args.q)}`;
      },
    });
  const wait = tool({
    name: "wait",
    description: "Wait until interrupted.",
    parameters: { type: "object", properties: {} },
    resumable: true,
    execute: async (_args, ctx) => {
      ran.push("wait");
      began();
      if (!ctx.interrupted.aborted) await new Promise((resolve) => ctx.interrupted.addEventListener("abort", resolve, { once: true }));
      return "stopped";
    },
  });
  return { lookup: safe("lookup", true), team: [safe("a", false)], wait };
}

/** A hook keeping each call it is asked about in `asked`, and each signal it got in `signals`, answering `decide(call)`. */
function asking(decide: (call: GatedCall) => unknown = () => undefined) {
  const asked: GatedCall[] = [];
  const signals: AbortSignal[] = [];
  const hook: BeforeToolCall = (call, ctx) => {
    asked.push(call);
    signals.push(ctx.signal);
    return decide(call) as GateDecision;
  };
  return { asked, signals, hook };
}

/**
 * Run the flow granting `grants` with `model`, gated by `hook`, at `extra.log` (a fresh one by
 * default), noting tool runs in `extra.ran`; the rest of `extra` adds options.
 */
async function gatedRun(
  grants: string,
  model: ScriptedModel,
  hook: BeforeToolCall | null,
  extra: { log?: string; ran?: string[]; signal?: AbortSignal; onEvent?: (event: RunEvent) => void } = {},
) {
  const { log = freshLog(), ran = [], ...rest } = extra;
  const events: RunEvent[] = [];
  const options = { env, log, runId: "r", tools: hostTools(ran), models: serving(model), onEvent: (event: RunEvent) => void events.push(event) };
  const result = await runFlow(await flowOf(grants), {}, gated({ ...options, ...rest }, hook));
  return { result, log, events, ran, options };
}

/** Run agent `chat` directly over `toolset`, granted `tools`, with `model`; `extra` adds to its context. */
const direct = (model: ScriptedModel, toolset: ToolSet, tools: string[], extra: Partial<AgentRunContext> = {}) =>
  runAgent({
    nodeId: "chat",
    messages: [userMessage("go")],
    tools,
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    toolset,
    systemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
    ...extra,
  });

/** The steps node `node` recorded in the log at `log`. */
const stepsOf = (log: string, node = "reply") =>
  readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.nodeId === node ? [record.step] : []));

/** Each step's kind, a result as its `closed` word when it has one. */
const shape = (log: string, node = "reply") =>
  stepsOf(log, node).map((step) => (typeof step["closed"] === "string" ? step["closed"] : step["kind"]));

/** The tool results the model read in its request `index`: `[call id, content, is error]`. */
const read = (model: ScriptedModel, index: number) =>
  model.requests[index]!.messages
    .flatMap((message) => (message.role === "tool" ? (message.parts as ToolResultPart[]) : []))
    .map((part) => [part.callId, part.content, part.isError]);

/** The tool call brackets among `events`: `[kind, tool call id, result or error]`. */
const brackets = (events: readonly RunEvent[]) =>
  events.flatMap((event) => {
    if (event instanceof ToolCallStarted) return [["started", event.toolCallId]];
    if (event instanceof ToolCallFinished) return [["finished", event.toolCallId, event.error ?? event.result]];
    return [];
  });

test("the hook decides a whole group before any call starts; a refusal is recorded and the model reads it as the call's error", async () => {
  const ran: string[] = [];
  // Set in a callback, so declared wide.
  let ranAtDecision = null as string[] | null;
  const { asked, hook } = asking((call) => {
    if (call.tool === "lookup") return { run: true, note: "rule 7" };
    // Decided later, after the group's other call was allowed.
    return Promise.resolve().then(() => {
      ranAtDecision = [...ran];
      return { refuse: "No team tools." };
    });
  });
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"]), textReply("done")], CONFIG);
  const { result, log, events } = await gatedRun("lookup, team", model, hook, { ran });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ranAtDecision).toEqual([]);
  expect(ran).toEqual(["lookup"]);
  const n = shape(log).indexOf("model_call") + 1;
  expect(asked).toEqual([
    { callId: `r/reply/${n}/0`, node: "reply", tool: "lookup", source: { kind: "host", group: null, tool: "lookup" }, arguments: { q: "x" }, target: null, scopes: [], attempt: 1 },
    { callId: `r/reply/${n}/1`, node: "reply", tool: "team_a", source: { kind: "host", group: "team", tool: "a" }, arguments: { q: "y" }, target: null, scopes: [], attempt: 1 },
  ]);
  // Frozen, so the hook can't change what the tool gets.
  expect([asked[0], asked[0]!.arguments, asked[0]!.source, asked[0]!.scopes].every(Object.isFrozen)).toBe(true);
  const [, refusal, allowed] = stepsOf(log);
  expect(refusal).toEqual({ kind: "tool_result", call_id: "c2", started_at: null, output: "No team tools.", is_error: true, closed: "refused" });
  expect(allowed).toMatchObject({ call_id: "c1", output: "lookup:x", is_error: false, note: "rule 7" });
  expect(read(model, 1)).toEqual([
    ["c1", "lookup:x", false],
    ["c2", "No team tools.", true],
  ]);
  // The note is for audit: a request rebuilt from the steps matches the one sent, and the
  // messages the agent leaves hold no note.
  const view = await agentCall(log, { nodeId: "reply", n: shape(log).lastIndexOf("model_call") + 1 }, { env });
  expect(view.digestMismatch).toBe(false);
  const left = JSON.stringify(rebuildMessages(stepsOf(log), null).toObject());
  expect([left.includes("No team tools."), left.includes("rule 7")]).toEqual([true, false]);
  expect(brackets(events)).toEqual([
    ["started", "c2"],
    ["finished", "c2", "refused"],
    ["started", "c1"],
    ["finished", "c1", "lookup:x"],
  ]);
});

test("a registry tool, which a run without a tools mapping reaches, is a top-level host tool", async () => {
  const weather = tool({ name: "gate_weather", description: "Weather.", parameters: { type: "object", properties: {} }, execute: () => "sunny" });
  registerTool("gate_weather", weather);
  try {
    const { asked, hook } = asking((call) => (call.callId.endsWith("/0") ? undefined : { refuse: "No weather." }));
    const model = new ScriptedModel([callReply(["gate_weather", {}, "c1"], ["gate_weather", {}, "c2"]), textReply("done")], CONFIG);
    const result = await runFlow(await flowOf("gate_weather"), {}, gated({ env, models: serving(model) }, hook));
    expect(result.status, result.failure?.message).toBe("succeeded");
    const source = { kind: "host", group: null, tool: "gate_weather" } as const;
    expect(asked.map((call) => call.source)).toEqual([source, source]);
    expect(read(model, 1)).toEqual([
      ["c1", "sunny", false],
      ["c2", "No weather.", true],
    ]);
  } finally {
    TOOL_REGISTRY.delete("gate_weather");
  }
});

/** A model calling `lookup` once, then answering. */
const oneLookup = () => new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"]), textReply("done")], CONFIG);

test.each<[string, () => unknown, RegExp]>([
  ["a throw", () => { throw new Error("boom"); }, /: boom;/],
  ["a rejection", () => Promise.reject(new Error("later")), /: later;/],
  ["null", () => null, /./],
  ["an empty object", () => ({}), /./],
  ["run: false", () => ({ run: false }), /./],
  ["run with another key", () => ({ run: true, refuse: "x" }), /./],
  ["a class instance", () => new (class { run = true })(), /./],
  ["an empty refusal", () => ({ refuse: "" }), /refuse must be a non-empty string/],
  ["an oversized note", () => ({ run: true, note: "x".repeat(4097) }), /note is 4097 UTF-8 bytes/],
  ["a note that is null", () => ({ run: true, note: null }), /note must be a string/],
  ["a Promise of a bad shape", () => Promise.resolve({ run: "yes" }), /./],
])("a hook giving %s refuses the call with the fixed text and a warning", async (_, decide, message) => {
  const warnings: string[] = [];
  const model = oneLookup();
  const { result, log, ran } = await withWarnings(
    (text) => void warnings.push(text),
    () => gatedRun("lookup", model, asking(decide).hook),
  );
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual([]);
  expect(stepsOf(log)[1]).toMatchObject({ call_id: "c1", output: GATE_FAILED_TEXT, closed: "refused" });
  expect(read(model, 1)).toEqual([["c1", GATE_FAILED_TEXT, true]]);
  expect(warnings).toHaveLength(1);
  expect(warnings[0]).toStartWith("node 'reply': beforeToolCall failed for tool 'lookup': ");
  expect(warnings[0]).toEndWith("; the call is refused");
  expect(warnings[0]).toMatch(message);
});

test("a run decision whose note is undefined runs the call with no note", async () => {
  const model = oneLookup();
  const { log, ran } = await gatedRun("lookup", model, asking(() => ({ run: true, note: undefined })).hook);
  expect(ran).toEqual(["lookup"]);
  expect(Object.hasOwn(stepsOf(log)[1]!, "note")).toBe(false);
});

test("the hook isn't asked about load_skill, ask_user, a call refused before its tool, or a reply at the cap", async () => {
  const { asked, hook } = asking();
  const unparsed = new ToolCallPart({ id: "c5", name: "lookup", arguments: null, rawArguments: "{" });
  const first = new Response({
    message: new Message({
      role: "assistant",
      parts: [
        new ToolCallPart({ id: "c1", name: "load_skill", arguments: { name: "review" } }),
        new ToolCallPart({ id: "c2", name: "ask_user", arguments: { question: "ok?" } }),
        new ToolCallPart({ id: "c3", name: "nope", arguments: {} }),
        new ToolCallPart({ id: "c4", name: "lookup", arguments: { q: 5 } }),
        unparsed,
      ],
    }),
    finishReason: "tool_calls",
  });
  const model = new ScriptedModel([first, callReply(["lookup", { q: "x" }, "c6"]), textReply("done")], CONFIG);
  const ran: string[] = [];
  const toolset = new ToolSet({ ...hostTools(ran), ask: () => "yes" }).withGate(hook).forRun("r");
  const skills = [new Skill({ name: "review", description: "Review.", instructions: "Read it." })];
  const output = await direct(model, toolset, ["lookup", "ask_user"], { maxToolIterations: 2, skills });
  expect((output as { value: unknown }).value).toBe("done");
  expect(read(model, 1).map(([id, , isError]) => [id, isError])).toEqual([
    ["c1", false],
    ["c2", false],
    ["c3", true],
    ["c4", true],
    ["c5", true],
  ]);
  expect(asked).toEqual([]);
  expect(ran).toEqual([]);
});

test("the hook isn't asked about a TOOL node's call, a map's or a loop's", async () => {
  const { asked, hook } = asking();
  const flow = await loadFlow(`
id: nodes
name: nodes
nodes:
  direct: {kind: call, call: lookup, input: {q: one}}
  each: {kind: map, over: [a, b], call: lookup, input: {q: "\${item}"}}
  again: {kind: loop, call: echo, input: {q: two}, times: 1}
output: \${each.output}
`);
  const ran: string[] = [];
  const echo = tool({ name: "echo", description: "Echo.", parameters: { type: "object", properties: { q: { type: "string" } } }, execute: (args) => {
    ran.push("echo");
    return { q: String(args.q) };
  } });
  const result = await runFlow(flow, {}, gated({ env, tools: { ...hostTools(ran), echo } }, hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect([asked, ran.sort()]).toEqual([[], ["echo", "lookup", "lookup", "lookup"]]);
});

test("an interrupt while the hook decides leaves the group unrun and skipped, even when the decision then runs it", async () => {
  const decision = Promise.withResolvers<GateDecision>();
  const askedNow = Promise.withResolvers<void>();
  const { signals, hook } = asking(() => {
    askedNow.resolve();
    return decision.promise;
  });
  const warnings: string[] = [];
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"])], CONFIG);
  const log = freshLog();
  const running = withWarnings(
    (text) => void warnings.push(text),
    () => gatedRun("lookup, team", model, hook, { log }),
  );
  await askedNow.promise;
  await interrupt(log, "m", { env });
  expect(signals.map((signal) => signal.aborted)).toEqual([true, true]);
  decision.resolve(undefined);
  const { result, ran } = await running;
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
  expect(ran).toEqual([]);
  expect(shape(log)).toEqual(["model_call", "skipped", "skipped", "interrupted"]);
  expect(warnings).toEqual([]);
});

test("a hold decided at once and a Promise still out when an interrupt lands write only skips, with no fault", async () => {
  const askedNow = Promise.withResolvers<void>();
  const { hook } = asking((call) => {
    if (call.tool === "lookup") return pending();
    askedNow.resolve();
    return new Promise(() => {});
  });
  const warnings: string[] = [];
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"])], CONFIG);
  const log = freshLog();
  const running = withWarnings(
    (text) => void warnings.push(text),
    () => gatedRun("lookup, team", model, hook, { log }),
  );
  await askedNow.promise;
  await interrupt(log, "m", { env });
  const { result, ran } = await running;
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual([]);
  expect(shape(log)).toEqual(["model_call", "skipped", "skipped", "interrupted"]);
  expect(warnings).toEqual([]);
});

test("a stop while the hook decides writes nothing; the resume asks again about a resumable call and closes the other crashed", async () => {
  const decision = Promise.withResolvers<GateDecision>();
  const askedNow = Promise.withResolvers<void>();
  const first = asking(() => {
    askedNow.resolve();
    return decision.promise;
  });
  const stop = new AbortController();
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"]), textReply("done")], CONFIG);
  const log = freshLog();
  const ran: string[] = [];
  const options = { env, log, runId: "r", tools: hostTools(ran), models: serving(model) };
  const running = runFlow(await flowOf("lookup, team"), {}, gated({ ...options, signal: stop.signal }, first.hook));
  await askedNow.promise;
  stop.abort(new Error("stop"));
  decision.resolve(undefined);
  expect(await rejected(running)).toEqual(new Error("stop"));
  expect(first.signals.map((signal) => signal.aborted)).toEqual([true, true]);
  expect(ran).toEqual([]);
  expect(shape(log)).toEqual(["model_call"]);
  const second = asking();
  const done = await advance(gated({ ...options, log }, second.hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(second.asked.map((call) => [call.tool, call.attempt])).toEqual([["lookup", 2]]);
  expect(ran).toEqual(["lookup"]);
  expect(shape(log)).toEqual(["model_call", "tool_result", "crashed", "model_call"]);
});

test("an interrupt from onEvent while a refusal is written skips the group's allowed call", async () => {
  const { hook } = asking((call) => (call.tool === "lookup" ? undefined : { refuse: "No." }));
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"])], CONFIG);
  const log = freshLog();
  const delivered: Promise<unknown>[] = [];
  const onEvent = (event: RunEvent) => {
    if (event instanceof ToolCallFinished && event.error === "refused") delivered.push(interrupt(log, "m", { env }));
  };
  const { result, ran } = await gatedRun("lookup, team", model, hook, { log, onEvent });
  await Promise.all(delivered);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual([]);
  expect(shape(log)).toEqual(["model_call", "refused", "skipped", "interrupted"]);
});

/** The record count up to and with the first step of `kind` in the log at `log`: a crash cut there keeps that step. */
const through = (log: string, kind: string) =>
  readRunLog(log).findIndex(({ record }) => record instanceof NodeStepRecord && record.step["kind"] === kind) + 1;

/** Crash a gated run of `lookup` at record `at`, then resume it with `hook`; return the resume's result and its log. */
async function crashThenResume(at: number, hook: BeforeToolCall, ran: string[] = []) {
  const log = freshLog();
  const options = { env, log, runId: "r", tools: hostTools(ran), models: serving(oneLookup()) };
  const flow = await flowOf("lookup");
  await crashAfter(at, () => runFlow(flow, {}, gated(options, hook)));
  const done = await advance(gated({ ...options, models: serving(new ScriptedModel([textReply("done")], CONFIG)) }, hook));
  return { done, log };
}

test("a resend after a crash asks the hook again with its attempt; a replayed result isn't asked about", async () => {
  const whole = await gatedRun("lookup", oneLookup(), () => undefined);
  // Cut before the call's result, it is sent again; cut after, it is replayed.
  for (const [kind, attempts] of [["model_call", [1, 2]], ["tool_result", [1]]] as const) {
    const { asked, hook } = asking();
    const { done } = await crashThenResume(through(whole.log, kind), hook);
    expect(done.status, done.failure?.message).toBe("succeeded");
    expect(asked.map((call) => call.attempt)).toEqual([...attempts]);
  }
});

test("a hook refusing a resend writes refused, not crashed", async () => {
  const whole = await gatedRun("lookup", oneLookup(), () => undefined);
  const ran: string[] = [];
  const { asked, hook } = asking((call) => (call.attempt === 1 ? undefined : { refuse: "Not again." }));
  const { done, log } = await crashThenResume(through(whole.log, "model_call"), hook, ran);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([asked.length, ran]).toEqual([2, ["lookup"]]);
  expect(shape(log)).toEqual(["model_call", "refused", "model_call"]);
});

test("a resend with an interrupt pending is closed crashed without asking the hook", async () => {
  /** Run `wait` at `log`: an interrupt reaches it while it runs. */
  const waitRun = async (log: string, hook: BeforeToolCall, ran: string[]) => {
    const delivered: Promise<unknown>[] = [];
    const tools = hostTools(ran, () => void delivered.push(interrupt(log, "m", { env })));
    const model = new ScriptedModel([callReply(["wait", {}, "c1"])], CONFIG);
    try {
      return await runFlow(await flowOf("wait"), {}, gated({ env, log, runId: "r", tools, models: serving(model) }, hook));
    } finally {
      await Promise.allSettled(delivered);
    }
  };
  const uncut = freshLog();
  await waitRun(uncut, () => undefined, []);
  const log = freshLog();
  const ran: string[] = [];
  const { asked, hook } = asking();
  // Cut after the interrupt's record, before the call's result.
  await crashAfter(readRunLog(uncut).findIndex(({ kind }) => kind === "interrupted") + 1, () => waitRun(log, hook, ran));
  expect(shape(log)).toEqual(["model_call"]);
  const done = await advance(gated({ env, log, tools: hostTools(ran), models: serving(new ScriptedModel([], CONFIG)) }, hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBeNull();
  expect([asked.length, ran]).toEqual([1, ["wait"]]);
  expect(shape(log)).toEqual(["model_call", "crashed", "interrupted"]);
});

/**
 * A flow whose agent grants the module tools `price` (one export) and `crm_rate` (a group's), the
 * memory tool `lessons_append`, the flow `shout`, the library's def `billing_quote`, the `uses:`
 * flow `kid` and `lookup`.
 */
const OWN_FLOW = `
id: own
name: own
llm_config: {provider: openai, model: test-model}
input:
  lessons: !FileMemory
uses:
  price: "./quote.ts:quote"
  crm: ./quote.ts
  billing: ./billing
  kid: ./kid/child
defs:
  shout:
    description: Shout the text back.
    input:
      text: !str
    nodes:
      loud:
        kind: code
        input:
          text: \${input.text}
        output: !str
        code: return inputs.text.toUpperCase();
    output: \${loud.output}
nodes:
  reply:
    kind: agent
    input:
      lessons: \${input.lessons}
    prompt: go
    tools: [price, {crm: [rate]}, {lessons: [append]}, shout, {billing: [quote]}, kid, lookup]
output: \${reply.output}
`;

/** Where the tools of `quote.ts` and `kid/quote.ts` note each run. */
const quoted = join(dir, "quoted");

/** A module's source exporting tools `names`, each noting its run in `quoted` as `<note> `. */
const quoteModule = (names: readonly string[], note: (name: string) => string = (name) => name) =>
  `import { appendFileSync } from "node:fs";\n` +
  names
    .map((name) => `export const ${name} = { name: "${name}", description: "${name}.", parameters: { type: "object", properties: { id: { type: "string" } } }, ` +
      `execute: () => { appendFileSync(${JSON.stringify(quoted)}, "${note(name)} "); return 1; } };\n`)
    .join("");

/**
 * Run `OWN_FLOW` with `model`, gated by `hook`, once its files are written: `quote.ts`, whose
 * tools `quote` and `rate` note each run in `quoted` (cleared first), the library `billing`, and
 * the flow `kid/child.yaml`, whose agent `leaf` grants its own module's tool `price`.
 */
async function ownRun(model: ScriptedModel, hook: BeforeToolCall, extra: { env?: typeof env } = {}) {
  rmSync(quoted, { force: true });
  writeFileSync(join(dir, "quote.ts"), quoteModule(["quote", "rate"]));
  mkdirSync(join(dir, "kid"), { recursive: true });
  writeFileSync(join(dir, "kid", "quote.ts"), quoteModule(["quote"], () => "kid"));
  const child = "id: child\nname: child\ndescription: Price a thing.\nllm_config: {provider: openai, model: test-model}\n" +
    'input:\n  thing: !str\nuses:\n  price: "./quote.ts:quote"\nnodes:\n  leaf: {kind: agent, prompt: leaf, tools: [price]}\n' +
    "output: ${leaf.output}\n";
  writeFileSync(join(dir, "kid", "child.yaml"), child);
  const billing = "id: billing\nname: billing\ndefs:\n  quote:\n    description: Price an order.\n    input: {qty: !int}\n" +
    "    kind: code\n    code: return inputs.qty * 3\n    output: !int\n";
  writeFileSync(join(dir, "billing.yaml"), billing);
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/lessons")));
  const log = freshLog();
  const ran: string[] = [];
  const options = { env: extra.env ?? env, log, runId: "r", tools: hostTools(ran), models: serving(model) };
  const result = await runFlow(await loadFlow(OWN_FLOW, { searchPaths: [dir] }), { lessons: memory }, gated(options, hook));
  const applied = memory.sessions.flatMap((session) => session.applied);
  return { result, log, ran, applied, quotes: existsSync(quoted) ? readFileSync(quoted, "utf8") : null };
}

/** Whether the log at `log` grew a flow's scope. */
const expanded = (log: string) => readRunLog(log).some(({ kind }) => kind === "node_expanded");

test.each<[string, string, JsonObject, ToolSource]>([
  ["a one-export module tool", "price", { id: "1" }, { kind: "module", alias: "price", ref: "./quote.ts:quote", export: "quote" }],
  ["a module group's tool", "crm_rate", { id: "1" }, { kind: "module", alias: "crm", ref: "./quote.ts", export: "rate" }],
  ["a memory tool", "lessons_append", { path: "a.md", text: "x" }, { kind: "memory", input: "lessons", tool: "append" }],
  ["a flow tool", "shout", { text: "hi" }, { kind: "flow", flow: "shout", library: null }],
  ["a library's def", "billing_quote", { qty: 2n }, { kind: "flow", flow: "quote", library: "billing" }],
  ["a uses: flow", "kid", { thing: "x" }, { kind: "flow", flow: "kid", library: null }],
])("the hook is asked about a call of %s with its source, and its refusal leaves the tool unrun", async (_, name, args, source) => {
  const { asked, hook } = asking((call) => (call.tool === "lookup" ? undefined : { refuse: "Not yours." }));
  const model = new ScriptedModel([callReply([name, args, "c1"], ["lookup", { q: "x" }, "c2"]), textReply("done")], CONFIG);
  const { result, log, ran, applied, quotes } = await ownRun(model, hook);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const n = shape(log).indexOf("model_call") + 1;
  expect(asked).toEqual([
    { callId: `r/reply/${n}/0`, node: "reply", tool: name, source, arguments: args, target: null, scopes: [], attempt: 1 },
    { callId: `r/reply/${n}/1`, node: "reply", tool: "lookup", source: { kind: "host", group: null, tool: "lookup" }, arguments: { q: "x" }, target: null, scopes: [], attempt: 1 },
  ]);
  expect(Object.isFrozen(asked[0]!.source)).toBe(true);
  // A refused flow call grows no scope.
  expect([ran, quotes, applied, expanded(log)]).toEqual([["lookup"], null, [], false]);
  expect(read(model, 1)).toEqual([
    ["c1", "Not yours.", true],
    ["c2", "lookup:x", false],
  ]);
});

/** A flow whose agent `reply` grants the def `shout`, a code node. */
const SHOUT_FLOW = `
id: shouting
name: shouting
llm_config: {provider: openai, model: test-model}
defs:
  shout:
    description: Shout the text back.
    input: {text: !str}
    kind: code
    code: return inputs.text.toUpperCase()
    output: !str
nodes:
  reply: {kind: agent, prompt: go, tools: [shout]}
output: \${reply.output}
`;

/** The flow call scopes node `node`'s pending steps name in the log at `log`. */
const scopesOf = (log: string, node = "reply") =>
  stepsOf(log, node).flatMap((step) => (step["kind"] === "pending" && typeof step["scope"] === "string" ? [step["scope"]] : []));

/** The `model_call` steps the log at `log` holds, of any node; none before it exists. */
const modelCalls = (log: string) =>
  !existsSync(log) ? 0 : readRunLog(log).filter(({ record }) => record instanceof NodeStepRecord && record.step["kind"] === "model_call").length;

test("a flow call the hook fails on takes no scope number: the allowed call after it is #0, and a resume cut at any record agrees", async () => {
  const flow = await loadFlow(SHOUT_FLOW);
  const replies = () => [callReply(["shout", { text: "no" }, "c1"], ["shout", { text: "yes" }, "c2"]), textReply("done")];
  const hook: BeforeToolCall = (call) => {
    if (call.arguments["text"] === "no") throw new Error("boom");
  };
  const quiet = <T>(fn: () => Promise<T>) => withWarnings(() => {}, fn);
  const run = (log: string, model: ScriptedModel) => runFlow(flow, {}, gated({ env, log, runId: "r", models: serving(model) }, hook));
  const uncut = freshLog();
  const model = new ScriptedModel(replies(), CONFIG);
  expect((await quiet(() => run(uncut, model))).status).toBe("succeeded");
  expect([shape(uncut).slice(0, 3), scopesOf(uncut)]).toEqual([["model_call", "refused", "pending"], ["reply/shout#0"]]);
  expect(read(model, 1)).toEqual([
    ["c1", GATE_FAILED_TEXT, true],
    ["c2", "YES", false],
  ]);
  for (let k = through(uncut, "model_call"); k < readRunLog(uncut).length; k++) {
    const log = freshLog();
    await quiet(() => crashAfter(k, () => run(log, new ScriptedModel(replies(), CONFIG))));
    const rest = new ScriptedModel(replies().slice(modelCalls(log)), CONFIG);
    const done = await quiet(() => advance(gated({ env, log, models: serving(rest) }, hook)));
    expect(done.status, `cut at ${k}: ${done.failure?.message}`).toBe("succeeded");
    expect(scopesOf(log)).toEqual(["reply/shout#0"]);
  }
}, 60_000);

test("a crash after the hook allows a flow call, before its pending step, asks again on resume before the flow's scope grows", async () => {
  const flow = await loadFlow(SHOUT_FLOW);
  const replies = () => [callReply(["shout", { text: "yes" }, "c1"]), textReply("done")];
  const uncut = freshLog();
  await runFlow(flow, {}, gated({ env, log: uncut, models: serving(new ScriptedModel(replies(), CONFIG)) }, () => undefined));
  const log = freshLog();
  const asked: [number, boolean][] = [];
  const hook: BeforeToolCall = (call) => void asked.push([call.attempt, expanded(log)]);
  // Cut after the reply's model_call: the hook allowed the call, and its pending step is lost.
  await crashAfter(through(uncut, "model_call"), () => runFlow(flow, {}, gated({ env, log, models: serving(new ScriptedModel(replies(), CONFIG)) }, hook)));
  expect([asked, shape(log)]).toEqual([[[1, false]], ["model_call"]]);
  const done = await advance(gated({ env, log, models: serving(new ScriptedModel(replies().slice(1), CONFIG)) }, hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(asked).toEqual([[1, false], [2, false]]);
  expect([shape(log), scopesOf(log)]).toEqual([["model_call", "pending", "tool_result", "model_call"], ["reply/shout#0"]]);
});

test("an allowed module, memory or flow call runs with its note on its first record; the flow's answer isn't asked about", async () => {
  const { asked, hook } = asking((call) => ({ run: true, note: `ok ${call.tool}` }));
  const calls: [string, JsonObject, string][] = [["price", { id: "1" }, "c1"], ["lessons_append", { path: "a.md", text: "x" }, "c2"], ["shout", { text: "hi" }, "c3"]];
  const model = new ScriptedModel([callReply(...calls), textReply("done")], CONFIG);
  const { result, log, applied, quotes } = await ownRun(model, hook);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect([asked.map((call) => call.tool), quotes, applied.length]).toEqual([["price", "lessons_append", "shout"], "quote ", 1]);
  const [, price, append, flow, answer] = stepsOf(log);
  expect([price, append]).toMatchObject([{ call_id: "c1", note: "ok price" }, { call_id: "c2", note: "ok lessons_append" }]);
  expect(flow).toMatchObject({ kind: "pending", flow: true, scope: "reply/shout#0", note: "ok shout" });
  // The flow's answer is the agent's, as the run shaped it: it holds no note.
  expect(answer).toEqual({ kind: "tool_result", call_id: "c3", started_at: null, output: "HI", is_error: false });
  expect(read(model, 1).map(([id, , isError]) => [id, isError])).toEqual([["c1", false], ["c2", false], ["c3", false]]);
});

test("an allowed flow call whose arguments are too large to record keeps its note on its error result, which counts it", async () => {
  const model = new ScriptedModel([callReply(["shout", { text: "x".repeat(5000) }, "c1"]), textReply("done")], CONFIG);
  const { result, log } = await ownRun(model, () => ({ run: true, note: "big" }), { env: hostEnv(dir, { YIAN_MAX_EVENT_BYTES: "2000" }) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(stepsOf(log)[1]).toMatchObject({ kind: "tool_result", call_id: "c1", is_error: true, note: "big" });
  const [[, content]] = read(model, 1) as [[string, string]];
  expect(content).toMatch(/^Arguments for shout make a call of \d+ bytes with the host's 3-byte note on it, over the run's limit of 2000 /);
  expect(expanded(log)).toBe(false);
});

test("a module tool's validator runs on its thread before the hook, which reads the model's arguments; a refused call never reaches execute", async () => {
  const marker = join(dir, "checked");
  const validate = `(value) => { appendFileSync(${JSON.stringify(marker)}, "checked "); return { value: { id: "default", ...value } }; }`;
  writeFileSync(
    join(dir, "checked.ts"),
    `import { appendFileSync, readFileSync } from "node:fs";\nexport const quote = { name: "quote", description: "Quote.", ` +
      `parameters: { "~standard": { version: 1, vendor: "test", validate: ${validate} } }, ` +
      `parametersJsonSchema: { type: "object", properties: { id: { type: "string" } } }, ` +
      `execute: (args) => { appendFileSync(${JSON.stringify(marker)}, "ran " + args.id); return 1; } };\n`,
  );
  const seen: string[] = [];
  const { asked, hook } = asking((call) => {
    seen.push(readFileSync(marker, "utf8"));
    return call.callId.endsWith("/0") ? { refuse: "No." } : undefined;
  });
  const flow = await loadFlow(
    `id: checked\nname: checked\nllm_config: {provider: openai, model: test-model}\nuses:\n  quote: "./checked.ts:quote"\n` +
      "nodes:\n  reply: {kind: agent, prompt: go, tools: [quote]}\noutput: ${reply.output}\n",
    { searchPaths: [dir] },
  );
  const model = new ScriptedModel([callReply(["quote", {}, "c1"], ["quote", {}, "c2"]), textReply("done")], CONFIG);
  const result = await runFlow(flow, {}, gated({ env, log: freshLog(), runId: "r", models: serving(model) }, hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(asked.map((call) => [call.source, call.arguments])).toEqual([
    [{ kind: "module", alias: "quote", ref: "./checked.ts:quote", export: "quote" }, {}],
    [{ kind: "module", alias: "quote", ref: "./checked.ts:quote", export: "quote" }, {}],
  ]);
  expect(seen).toEqual(["checked ", "checked checked "]);
  expect(readFileSync(marker, "utf8")).toBe("checked checked ran default");
});

/**
 * A flow whose agent `top` grants `outer`, whose agent `mid` grants `inner`, whose agent `leaf`
 * calls `lookup`; then the authored call `authored` of `plain`, whose agent `solo` calls `lookup`.
 */
const NESTED_FLOW = `
id: nested
name: nested
llm_config: {provider: openai, model: test-model}
defs:
  outer:
    description: Outer.
    nodes:
      mid: {kind: agent, prompt: mid, tools: [inner]}
    output: \${mid.output}
  inner:
    description: Inner.
    nodes:
      leaf: {kind: agent, prompt: leaf, tools: [lookup]}
    output: \${leaf.output}
  plain:
    description: Plain.
    input:
      after: !str
    nodes:
      solo: {kind: agent, prompt: solo, tools: [lookup]}
    output: \${solo.output}
nodes:
  top: {kind: agent, prompt: top, tools: [outer]}
  authored:
    kind: call
    call: plain
    input:
      after: \${top.output}
output: \${authored.output}
`;

/** `NESTED_FLOW`'s model replies, in the order its agents ask. */
const nestedReplies = () => [
  callReply(["outer", {}, "c1"]),
  callReply(["inner", {}, "c2"]),
  callReply(["lookup", { q: "x" }, "c3"]),
  textReply("leaf"),
  textReply("mid"),
  textReply("top"),
  callReply(["lookup", { q: "y" }, "c4"]),
  textReply("solo"),
];

/** The flow calls enclosing `NESTED_FLOW`'s agents `mid` (`outer`) and `leaf` (both). */
const outer = { scope: "top/outer#0", callId: "r/top/1/0" };
const inner = { scope: "top/outer#0/mid/inner#0", callId: "r/top/outer#0/mid/1/0" };

test("an agent inside a granted flow is asked with the flow calls enclosing it, outermost first; an authored call adds none", async () => {
  const flow = await loadFlow(NESTED_FLOW);
  const { asked, hook } = asking();
  const ran: string[] = [];
  const options = { env, log: freshLog(), runId: "r", tools: hostTools(ran), models: serving(new ScriptedModel(nestedReplies(), CONFIG)) };
  const result = await runFlow(flow, {}, gated(options, hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(asked.map(({ node, tool, scopes }) => [node, tool, scopes])).toEqual([
    ["top", "outer", []],
    ["top/outer#0/mid", "inner", [outer]],
    ["top/outer#0/mid/inner#0/leaf", "lookup", [outer, inner]],
    [asked[3]!.node, "lookup", []],
  ]);
  expect(asked[3]!.node).toEndWith("/solo");
  expect(asked.every((call) => Object.isFrozen(call.scopes) && call.scopes.every(Object.isFrozen))).toBe(true);
  expect(ran).toEqual(["lookup", "lookup"]);
});

test("a crash inside a flow call's scope asks the child again on resume, with the same scopes and its next attempt", async () => {
  const flow = await loadFlow(NESTED_FLOW);
  const leaf = "top/outer#0/mid/inner#0/leaf";
  const run = (log: string, hook: BeforeToolCall, replies: Response[]) =>
    gated({ env, log, runId: "r", tools: hostTools([]), models: serving(new ScriptedModel(replies, CONFIG)) }, hook);
  const uncut = freshLog();
  await runFlow(flow, {}, run(uncut, () => undefined, nestedReplies()));
  // Cut after `leaf`'s reply, before its call's result.
  const cut = readRunLog(uncut).findIndex(({ record }) => record instanceof NodeStepRecord && record.nodeId === leaf) + 1;
  const log = freshLog();
  const first = asking();
  await crashAfter(cut, () => runFlow(flow, {}, run(log, first.hook, nestedReplies())));
  const second = asking();
  const done = await advance(run(log, second.hook, nestedReplies().slice(modelCalls(log))));
  expect(done.status, done.failure?.message).toBe("succeeded");
  const leafCalls = (asked: GatedCall[]) => asked.filter((call) => call.node === leaf).map(({ scopes, attempt }) => [scopes, attempt]);
  expect([leafCalls(first.asked), leafCalls(second.asked)]).toEqual([[[[outer, inner], 1]], [[[outer, inner], 2]]]);
});

test("an agent inside a granted uses: flow is asked about its module tool, with the module's source and the flow's scope", async () => {
  const { asked, hook } = asking();
  const replies = [callReply(["kid", { thing: "x" }, "c1"]), callReply(["price", { id: "1" }, "c2"]), textReply("priced"), textReply("done")];
  const { result, quotes } = await ownRun(new ScriptedModel(replies, CONFIG), hook);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(asked.map(({ node, source, scopes }) => [node, source, scopes])).toEqual([
    ["reply", { kind: "flow", flow: "kid", library: null }, []],
    ["reply/kid#0/leaf", { kind: "module", alias: "price", ref: "./quote.ts:quote", export: "quote" }, [{ scope: "reply/kid#0", callId: "r/reply/1/0" }]],
  ]);
  expect(quotes).toBe("kid ");
});

test("a child agent of a flow call never runs its call when its parent's interrupt ends the scope while the hook decides", async () => {
  const flow = await loadFlow(`
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
defs:
  review:
    description: Review a thing.
    input:
      thing: !str
    nodes:
      sub:
        kind: agent
        prompt: check
        tools: [lookup]
        output: !Optional(str)
    output: \${sub.output}
nodes:
  helper:
    kind: agent
    steer_inbox: \${input.m}
    tools: [review]
    prompt: hi
    output: !Optional(str)
output: \${helper.output}
`);
  const askedNow = Promise.withResolvers<void>();
  const decision = Promise.withResolvers<GateDecision>();
  // The parent's flow call runs at once; the child's call waits on the decision.
  const { asked, signals, hook } = asking((call) => {
    if (call.tool === "review") return undefined;
    askedNow.resolve();
    return decision.promise;
  });
  const model = new ScriptedModel([callReply(["review", { thing: "x" }, "c1"]), callReply(["lookup", { q: "y" }, "c2"])], CONFIG);
  const log = freshLog();
  const ran: string[] = [];
  const warnings: string[] = [];
  const running = withWarnings(
    (text) => void warnings.push(text),
    () => runFlow(flow, {}, gated({ env, log, runId: "r", tools: hostTools(ran), models: serving(model) }, hook)),
  );
  await askedNow.promise;
  await interrupt(log, "m", { env });
  decision.resolve(undefined);
  const result = await running;
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(warnings).toEqual([]);
  expect(result.output).toBeNull();
  expect(asked.map((call) => [call.node, call.tool])).toEqual([
    ["helper", "review"],
    ["helper/review#0/sub", "lookup"],
  ]);
  expect((signals[1]!.reason as Error).name).toBe("ScopeEnded");
  expect(ran).toEqual([]);
  expect(stepsOf(log, "helper/review#0/sub").map((step) => step["kind"])).toEqual(["model_call"]);
});

test("a node's own tool named ask_user is asked about, unlike the run's ask handler; without skills, so is one named load_skill", async () => {
  const { asked, hook } = asking();
  const own = (name: string) => tool({ name, description: "Own.", parameters: { type: "object", properties: {} }, execute: () => name });
  const source = (flow: string) => ({ kind: "flow", flow, library: null }) as const;
  const sources = new Map(["ask_user", "load_skill"].map((name) => [nodeScopeKey(null, name), source(name)]));
  const scope = { groups: new Map(), tools: [own("ask_user"), own("load_skill")], sources };
  const toolset = new ToolSet({ ask: () => "yes" }).withGate(hook).forRun("r").withNodeScope(scope);
  const model = new ScriptedModel([callReply(["ask_user", {}, "c1"], ["load_skill", {}, "c2"]), textReply("done")], CONFIG);
  const output = await direct(model, toolset, ["ask_user", "load_skill"]);
  expect((output as { value: unknown }).value).toBe("done");
  expect(asked.map((call) => call.source)).toEqual([source("ask_user"), source("load_skill")]);
  expect(read(model, 1)).toEqual([
    ["c1", "ask_user", false],
    ["c2", "load_skill", false],
  ]);
});

test.each<[string, BeforeToolCall | null]>([
  ["without a hook", null],
  ["with a hook", () => undefined],
])("a call whose started event throws the interrupt's reason fails the agent with it, %s", async (_, hook) => {
  const inbox = interruptible();
  const emitEvent = (event: unknown) => {
    if (!(event instanceof ToolCallStarted)) return;
    // The first call's start takes the interrupt; the second's start event throws its reason.
    if (event.toolCallId === "c1") inbox.interrupt();
    if (event.toolCallId === "c2") throw inbox.steerInbox.interrupt.reason;
  };
  const model = new ScriptedModel([callReply(["lookup", { q: "1" }, "c1"], ["team_a", { q: "2" }, "c2"], ["lookup", { q: "3" }, "c3"])], CONFIG);
  let toolset = new ToolSet(hostTools([]));
  if (hook !== null) toolset = toolset.withGate(hook);
  const error = await rejected(direct(model, toolset.forRun("r"), ["lookup", "team"], { steerInbox: inbox.steerInbox, emitEvent }));
  expect(error).toBe(inbox.steerInbox.interrupt.reason);
});

test("once the agent takes an interrupt while the hook decides, it isn't asked about the group's later calls", async () => {
  const inbox = interruptible();
  const asked: string[] = [];
  const hook: BeforeToolCall = (call) => {
    asked.push(call.callId);
    inbox.interrupt();
    return undefined;
  };
  const ran: string[] = [];
  const model = new ScriptedModel([callReply(["lookup", { q: "1" }, "c1"], ["team_a", { q: "2" }, "c2"])], CONFIG);
  const output = await direct(model, new ToolSet(hostTools(ran)).withGate(hook).forRun("r"), ["lookup", "team"], {
    steerInbox: inbox.steerInbox,
  });
  expect(output).toBeInstanceOf(Interrupted);
  expect([asked, ran]).toEqual([["r/chat/1/0"], []]);
});

test("a gate on a tool set bound to no run fails the agent before the hook is asked", async () => {
  const { asked, hook } = asking();
  const ran: string[] = [];
  const error = await rejected(direct(oneLookup(), new ToolSet(hostTools(ran)).withGate(hook), ["lookup"]));
  expect(error).toEqual(new AgentLoopError("agent node 'chat': beforeToolCall checks tool calls only in a run"));
  expect([asked, ran]).toEqual([[], []]);
});

test("a hook that runs every call at once leaves the run's steps and events as they are without one", async () => {
  /** The steps without their times, the events' kinds, and the brackets of a run with `hook`. */
  const trace = async (hook: BeforeToolCall | null) => {
    const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"]), textReply("done")], CONFIG);
    const { log, events } = await gatedRun("lookup, team", model, hook);
    const steps = stepsOf(log).map(({ started_at: _started, first_token_at: _first, ...rest }) => rest);
    return { steps, events: events.map((event) => event.constructor.name), brackets: brackets(events) };
  };
  expect(await trace(() => undefined)).toEqual(await trace(null));
});

/** An abort signal that keeps the abort listeners on it in `live`, as `tests/platform.test.ts` counts them. */
function counted(): { signal: AbortSignal; live: Set<unknown> } {
  const { signal } = new AbortController();
  const live = new Set<unknown>();
  const add = signal.addEventListener.bind(signal);
  const remove = signal.removeEventListener.bind(signal);
  signal.addEventListener = ((type: string, listener: EventListener, options?: AddEventListenerOptions) => {
    if (type === "abort") live.add(listener);
    add(type, listener, options);
  }) as typeof signal.addEventListener;
  signal.removeEventListener = ((type: string, listener: EventListener, options?: EventListenerOptions) => {
    if (type === "abort") live.delete(listener);
    remove(type, listener, options);
  }) as typeof signal.removeEventListener;
  return { signal, live };
}

test.each<[string, BeforeToolCall | null]>([
  ["no hook", null],
  ["a hook answering at once", () => undefined],
  ["a hook answering later", () => Promise.resolve({ run: true, note: "n" })],
  ["a hook refusing later", () => Promise.resolve({ refuse: "No." })],
])("%s leaves no abort listener on the node's signal or the agent's interrupt across many turns", async (_, hook) => {
  const node = counted();
  const interrupt = counted();
  const turns = Array.from({ length: 50 }, (_, i) => callReply(["lookup", { q: `${i}` }, `a${i}`], ["team_a", { q: `${i}` }, `b${i}`]));
  const model = new ScriptedModel([...turns, textReply("done")], CONFIG);
  let toolset = new ToolSet(hostTools([]));
  if (hook !== null) toolset = toolset.withGate(hook);
  const steerInbox = { inbox: "m", queuedBesidesActions: () => false, read: () => null, interrupt: interrupt.signal, interruptedBy: () => null };
  const output = await direct(model, toolset.forRun("r"), ["lookup", "team"], { signal: node.signal, steerInbox });
  expect((output as { value: unknown }).value).toBe("done");
  expect([node.live.size, interrupt.live.size]).toEqual([0, 0]);
});

// ------------------------------------------------------------------------------------------- //
// Holds: the hook answers pending(). The held call never runs; the calls before it and in its
// group do, and the reply's later calls wait until it is answered.
// ------------------------------------------------------------------------------------------- //

/**
 * Tools noting each run in `ran` as `<tool>:<q>`: `lookup` (safe to run together, resumable), the
 * group `team`'s `a` (safe, not resumable), `later`, which runs alone and isn't resumable, and
 * `again`, which runs alone and is.
 */
function notedTools(ran: string[]) {
  const make = (name: string, safe: boolean, resumable: boolean) =>
    tool({
      name,
      description: `Tool ${name}.`,
      parameters: { type: "object", properties: { q: { type: "string" } } },
      isConcurrencySafe: () => safe,
      resumable,
      execute: (args) => {
        ran.push(`${name}:${String(args.q)}`);
        return `${name}:${String(args.q)}`;
      },
    });
  const again = make("again", false, true);
  return { lookup: make("lookup", true, true), team: [make("a", true, false)], later: make("later", false, false), again };
}

/** The agent granted `notedTools`' tools. */
const NOTED = "lookup, team, later, again";

/**
 * A reply whose call `q: "0"` runs alone, then `1` to `3` together, the hold `2` among them, then
 * the later calls `4`, alone, and `5`; and the answer after.
 */
const heldReplies = () => [
  callReply(
    ["later", { q: "0" }, "c0"],
    ["lookup", { q: "1" }, "c1"],
    ["team_a", { q: "2" }, "c2"],
    ["team_a", { q: "3" }, "c3"],
    ["later", { q: "4" }, "c4"],
    ["lookup", { q: "5" }, "c5"],
  ),
  textReply("done"),
];

/** The call id of `heldReplies`' call `index`, of the reply step 1. */
const heldId = (index: number) => `r/reply/1/${index}`;

/** A hook holding each call whose `q` is one of `held` and running the others, keeping each call it is asked about. */
const holding = (...held: string[]) => asking((call) => (held.includes(String(call.arguments["q"])) ? pending() : undefined));

/** A gated run's options at `log` over `notedTools(ran)`, its model left with `replies`, its events in `events`. */
const notedOptions = (log: string, ran: string[], replies: Response[], events: RunEvent[] = []) => ({
  env,
  log,
  runId: "r",
  tools: notedTools(ran),
  models: serving(new ScriptedModel(replies, CONFIG)),
  onEvent: (event: RunEvent) => void events.push(event),
});

/** The model replies of `calls` (`[tool, q]` each, the call id `c<q>`), then the answer. */
const replying = (...calls: [string, string][]) => () => [callReply(...calls.map(([name, q]) => [name, { q }, `c${q}`] as [string, JsonObject, string])), textReply("done")];

/**
 * Advance the gated run at `log` with `hook` and the model left of `replies` until it ends,
 * answering each hold it waits on with "approved", noted in `ran` as `answer:<q>` first.
 */
async function finish(log: string, ran: string[], hook: BeforeToolCall, replies: () => Response[] = heldReplies) {
  for (let round = 0; round < 6; round++) {
    for (const held of pendingCalls(log).filter((call) => call.held)) {
      ran.push(`answer:${String(held.arguments!["q"])}`);
      await complete(log, held.callId, { output: "approved" }, { env });
    }
    const result = await advance(gated(notedOptions(log, ran, replies().slice(modelCalls(log))), hook));
    if (result.status !== "waiting") return result;
  }
  throw new Error(`the run at ${log} never ended`);
}

test("a held call never runs: the calls before it and in its group run, and the later ones wait for complete, asked about then", async () => {
  const ran: string[] = [];
  const { asked, hook } = holding("2");
  const log = freshLog();
  const events: RunEvent[] = [];
  const first = await runFlow(await flowOf(NOTED), {}, gated(notedOptions(log, ran, heldReplies(), events), hook));
  expect(first.status, first.failure?.message).toBe("waiting");
  expect([ran, asked.map((call) => call.arguments["q"])]).toEqual([["later:0", "lookup:1", "a:3"], ["0", "1", "2", "3"]]);
  expect(stepsOf(log)[2]).toEqual({
    kind: "pending", call_id: heldId(2), tool: "team_a", tool_call_id: "c2", reply: 1n, index: 2n, held: true,
    source: { kind: "host", group: "team", tool: "a" }, group_end: 3n,
  });
  // Published once its hold is durable; a hold has no brackets until it runs.
  expect(events.filter((event) => event instanceof CallHeld)).toEqual([new CallHeld({ nodeId: "reply", callId: heldId(2), tool: "team_a" })]);
  expect(brackets(events).map(([, id]) => id)).not.toContain("c2");
  expect(pendingCalls(log).map((call) => [call.callId, call.held])).toEqual([[heldId(2), true]]);
  await complete(log, heldId(2), { output: "approved" }, { env });
  const model = new ScriptedModel(heldReplies().slice(1), CONFIG);
  const done = await advance(gated({ ...notedOptions(log, ran, []), models: serving(model) }, hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(ran.slice(3)).toEqual(["later:4", "lookup:5"]);
  expect(asked.slice(4).map((call) => [call.arguments["q"], call.attempt])).toEqual([["4", 1], ["5", 1]]);
  expect(read(model, 0)).toEqual([
    ["c0", "later:0", false],
    ["c1", "lookup:1", false],
    ["c2", "approved", false],
    ["c3", "a:3", false],
    ["c4", "later:4", false],
    ["c5", "lookup:5", false],
  ]);
});

/**
 * Cut the life of a gated run of `replies`, holding the calls `held`, at every record from its
 * reply's model call on; resume and finish each cut run (`finish`), and check it. A held call never
 * runs, and has the host's answer when its hold was written, else a crashed closer (a resend can't
 * be held); each call in `once` runs at most once; each call in `after` (a run, by the `q` of the
 * hold it waits for) runs only after that hold's answer; and every call has one result.
 */
async function sweep(replies: () => Response[], held: string[], once: string[], after: Record<string, string>) {
  const flow = await flowOf(NOTED);
  /** Answer each hold the run at `log` waits on, noted in `ran`. */
  const answer = async (log: string, ran: string[]) => {
    for (const call of pendingCalls(log).filter((each) => each.held)) {
      ran.push(`answer:${String(call.arguments!["q"])}`);
      await complete(log, call.callId, { output: "approved" }, { env });
    }
  };
  /** Run segment `i` of the life at `log`: the start, then each advance after the answers. */
  const segment = (log: string, ran: string[], hook: BeforeToolCall, i: number) =>
    i === 0
      ? runFlow(flow, {}, gated(notedOptions(log, ran, replies()), hook))
      : advance(gated(notedOptions(log, ran, replies().slice(modelCalls(log))), hook));
  const uncut = freshLog();
  const whole = holding(...held).hook;
  // Each segment's records in the uncut life.
  const spans: [number, number][] = [];
  for (let i = 0, ended = false; !ended; i++) {
    if (i > 0) await answer(uncut, []);
    const start = i === 0 ? 0 : readRunLog(uncut).length;
    ended = (await segment(uncut, [], whole, i)).status !== "waiting";
    spans.push([start, readRunLog(uncut).length]);
  }
  const ids = (replies()[0]!.message.parts as ToolCallPart[]).map((part) => part.id);
  for (const [i, [start, end]] of spans.entries()) {
    for (let k = Math.max(start, through(uncut, "model_call")); k < end; k++) {
      const log = freshLog();
      const ran: string[] = [];
      const { hook } = holding(...held);
      const done = await withWarnings(() => {}, async () => {
        for (let j = 0; j < i; j++) {
          if (j > 0) await answer(log, ran);
          await segment(log, ran, hook, j);
        }
        if (i > 0) await answer(log, ran);
        await crashAfter(k, () => segment(log, ran, hook, i));
        return await finish(log, ran, hook, replies);
      });
      const where = `segment ${i}, cut at ${k}: ${ran}`;
      expect(done.status, `${where}: ${done.failure?.message}`).toBe("succeeded");
      const steps = stepsOf(log);
      const results = steps.filter((step) => step["kind"] === "tool_result");
      for (const q of held) {
        const answer = results.find((step) => step["call_id"] === `c${q}`)?.["output"];
        const wrote = steps.some((step) => step["held"] === true && step["tool_call_id"] === `c${q}`);
        expect(answer, `${where}: ${q}`).toBe(wrote ? "approved" : TOOL_OUTCOME_UNKNOWN);
        expect(ran.some((name) => name.endsWith(`:${q}`) && !name.startsWith("answer:")), `${where}: ${q} ran`).toBe(false);
      }
      for (const name of once) expect(ran.filter((each) => each === name).length, `${where}: ${name}`).toBeLessThanOrEqual(1);
      // A later call waits for a hold that was written; a cut before it leaves nothing to wait for.
      for (const [name, hold] of Object.entries(after)) {
        const answered = ran.indexOf(`answer:${hold}`);
        const wrote = steps.some((step) => step["held"] === true && step["tool_call_id"] === `c${hold}`);
        if (wrote && ran.includes(name)) expect(answered >= 0 && answered < ran.indexOf(name), `${where}: ${name} before ${hold}`).toBe(true);
      }
      expect(ids.map((id) => results.filter((step) => step["call_id"] === id).length), where).toEqual(ids.map(() => 1));
    }
  }
}

test("a crash at any record of a held reply's life never runs the held call, a call twice unless it is resumable, or a later call before the answer", async () => {
  await sweep(heldReplies, ["2"], ["later:0", "a:3", "later:4"], { "later:4": "2", "lookup:5": "2" });
});

/** A reply with two holds: `b` alone, `h1` held beside `x`, `m` alone, `h2` held beside `y`, then `z`. */
const twoHolds = replying(["later", "b"], ["lookup", "h1"], ["team_a", "x"], ["later", "m"], ["lookup", "h2"], ["lookup", "y"], ["later", "z"]);

test("a crash at any record of a two-hold reply's life keeps each later call after the answer it waits for", async () => {
  await sweep(twoHolds, ["h1", "h2"], ["later:b", "a:x", "later:m", "later:z"], { "later:m": "h1", "lookup:y": "h1", "later:z": "h2" });
}, 60_000);

/** The closers of the log at `log`'s calls (`[call id, word]`), and the calls closed `crashed` that never ran (`ran`). */
const closersOf = (log: string, ran: readonly string[]) => {
  const closers = stepsOf(log).flatMap((step) => (typeof step["closed"] === "string" ? [[step["call_id"], step["closed"]]] : []));
  const neverRan = closers.filter(([id, word]) => word === "crashed" && !ran.some((name) => name.endsWith(`:${String(id).slice(1)}`)));
  return { closers, neverRan };
};

/** The record count of the log at `log` up to and with its `nth` step that `match` holds of, from record `from`. */
const recordAt = (log: string, match: (step: ValueRecord) => boolean, nth = 1, from = 0) =>
  readRunLog(log).findIndex(({ record }, at) => at >= from && record instanceof NodeStepRecord && match(record.step) && --nth === 0) + 1;

test("a crash as the start after an answer writes a flow call's answer leaves the later calls to run: it never reached them", async () => {
  // [shout F] [later h] [later L] [lookup L2]: F's scope answers it while the hold waits.
  const flow = await loadFlow(SHOUT_FLOW.replace("reply: {kind: agent, prompt: go, tools: [shout]}", "reply: {kind: agent, prompt: go, tools: [shout, lookup, later]}"));
  const replies = () => [callReply(["shout", { text: "f" }, "cF"], ["later", { q: "h" }, "ch"], ["later", { q: "L" }, "cL"], ["lookup", { q: "L2" }, "cL2"]), textReply("done")];
  const run = (log: string, ran: string[]) => gated({ ...notedOptions(log, ran, replies().slice(modelCalls(log))) }, holding("h").hook);
  const uncut = freshLog();
  expect((await runFlow(flow, {}, run(uncut, []))).status).toBe("waiting");
  await complete(uncut, heldId(1), { output: "yes" }, { env });
  expect((await advance(run(uncut, []))).status).toBe("succeeded");
  const log = freshLog();
  const ran: string[] = [];
  await runFlow(flow, {}, run(log, ran));
  await complete(log, heldId(1), { output: "yes" }, { env });
  // Cut at the woken start's first step: its record of F's answer.
  const woke = readRunLog(uncut).findIndex(({ kind }) => kind === "node_woken");
  await crashAfter(recordAt(uncut, (step) => step["call_id"] === "cF", 1, woke) - 1, () => advance(run(log, ran)));
  expect((await advance(run(log, ran))).status).toBe("succeeded");
  expect([ran, closersOf(log, ran).closers]).toEqual([["later:L", "lookup:L2"], []]);
});

/** `flowOf`'s flow, its agent granted `grants`, with the defs `shout`, a code node, and `fail`, one that throws. */
const flowCallsOf = (grants: string) =>
  loadFlow(`
id: gate
name: gate
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
defs:
  shout:
    description: Shout the text back.
    input: {text: !str}
    kind: code
    code: return inputs.text.toUpperCase()
    output: !str
  fail:
    description: Fail.
    input: {text: !str}
    kind: code
    code: throw new Error("no " + inputs.text)
    output: !str
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [${grants}]
    steer_inbox: \${input.m}
    output: !Optional(str)
output: \${reply.output}
`);

/**
 * Tools noting each run in `ran`, each running alone and returning once a flow call's scope ended
 * (its END succeeded or it failed), so the flow's answer lands while it runs: `settle`, and `stop`,
 * which then interrupts the run at `log` and returns once its call learns of it. `onEvent` sees the
 * scope end.
 */
function scopeWaiters(log: string, ran: string[]) {
  let ended = () => {};
  const scopeEnded = new Promise<void>((resolve) => (ended = resolve));
  const onEvent = (event: RunEvent) => {
    if (event instanceof NodeFailed ? event.scope !== null : event instanceof NodeSucceeded && event.nodeId.endsWith(`#0/${END_ID}`)) ended();
  };
  const waiter = (name: string, then: (interrupted: AbortSignal) => Promise<unknown>) =>
    tool({
      name,
      description: `Tool ${name}.`,
      parameters: { type: "object", properties: { q: { type: "string" } } },
      execute: async (args, ctx) => {
        ran.push(`${name}:${String(args.q)}`);
        await scopeEnded;
        await then(ctx.interrupted);
        return `${name}:${String(args.q)}`;
      },
    });
  const settle = waiter("settle", async () => {});
  const stop = waiter("stop", async (interrupted) => {
    await interrupt(log, "m", { env });
    if (!interrupted.aborted) await new Promise((resolve) => interrupted.addEventListener("abort", resolve, { once: true }));
  });
  return { tools: { settle, stop }, onEvent };
}

/**
 * Run `replies` at `log` over `notedTools` and `scopeWaiters`, holding the call `h` and answering
 * it as it is announced, so its answer wakes the agent while the reply's flow call `cF` still runs.
 */
async function flowBesideHold(replies: () => Response[], log = freshLog()) {
  const ran: string[] = [];
  const events: RunEvent[] = [];
  const delivered: Promise<unknown>[] = [];
  const waiters = scopeWaiters(log, ran);
  const onEvent = (event: RunEvent) => {
    events.push(event);
    waiters.onEvent(event);
    if (event instanceof CallHeld) delivered.push(complete(log, event.callId, { output: "approved" }, { env }));
  };
  const model = new ScriptedModel(replies(), CONFIG);
  const options = { ...notedOptions(log, ran, []), models: serving(model), tools: { ...notedTools(ran), ...waiters.tools }, onEvent };
  const result = await runFlow(await flowCallsOf(`${NOTED}, settle, stop, shout, fail`), {}, gated(options, holding("h").hook));
  await Promise.all(delivered);
  const waits = events.filter((event) => event instanceof NodeWaiting).map((event) => (event as NodeWaiting).callIds);
  return { result, log, ran, model, waits, results: stepsOf(log).filter((step) => step["kind"] === "tool_result") };
}

test.each<[string, string, [string, string, boolean]]>([
  ["answers", "shout", ["cF", "F", false]],
  ["fails", "fail", ["cF", "Flow fail failed at node fail (node.execute): Error: no f", true]],
])("a flow call whose scope %s while the woken agent runs a later call is waited on again and read on the next wake", async (_, name, read0) => {
  // [F] [later h] [settle L]: h's answer wakes the agent while F's scope runs, and the scope ends
  // while L runs, after the woken start read F open.
  const replies = () => [callReply([name, { text: "f" }, "cF"], ["later", { q: "h" }, "ch"], ["settle", { q: "L" }, "cL"]), textReply("done")];
  const { result, ran, model, waits, results } = await flowBesideHold(replies);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual(["settle:L"]);
  expect(waits).toEqual([[heldId(0), heldId(1)], [heldId(0)]]);
  expect(results.map((step) => step["call_id"])).toEqual(["ch", "cL", "cF"]);
  expect(read(model, 1)).toEqual([read0, ["ch", "approved", false], ["cL", "settle:L", false]]);
});

test.each<[string, string, string | undefined]>([
  ["answer", "shout", undefined],
  ["failure", "fail", "failed"],
])("an interrupt honored after a flow call's scope ended while a later call ran records the flow's %s", async (_, name, closed) => {
  // [F] [later h] [settle L0] [stop L1]: F's scope ends while L0 runs; the interrupt lands while L1 runs.
  const replies = () => [
    callReply([name, { text: "f" }, "cF"], ["later", { q: "h" }, "ch"], ["settle", { q: "L0" }, "cL0"], ["stop", { q: "L1" }, "cL1"]),
  ];
  const { result, ran, results } = await flowBesideHold(replies);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
  expect(ran).toEqual(["settle:L0", "stop:L1"]);
  expect(results.map((step) => [step["call_id"], step["closed"]])).toEqual([["ch", undefined], ["cL0", undefined], ["cL1", undefined], ["cF", closed]]);
});

test("a flow call that failed while its hold waited, cut at its closer on the wake, leaves the later calls to run: the wake never reached them", async () => {
  // [G] [team_a h1, lookup x] [later L0] [again L1]: G's scope fails while the node waits on h1.
  const replies = () => [
    callReply(["fail", { text: "g" }, "cG"], ["team_a", { q: "h1" }, "ch1"], ["lookup", { q: "x" }, "cx"], ["later", { q: "L0" }, "cL0"], ["again", { q: "L1" }, "cL1"]),
    textReply("done"),
  ];
  const flow = await flowCallsOf(`${NOTED}, fail`);
  const run = (log: string, ran: string[]) => gated(notedOptions(log, ran, replies().slice(modelCalls(log))), holding("h1").hook);
  /** Run to the wait, answer h1 and advance, cut at record `k` when given; then advance to the end. */
  const life = async (log: string, ran: string[], k: number | null) => {
    expect((await runFlow(flow, {}, run(log, ran))).status).toBe("waiting");
    await complete(log, heldId(1), { output: "yes" }, { env });
    if (k !== null) await crashAfter(k, () => advance(run(log, ran)));
    return await advance(run(log, ran));
  };
  const uncut = freshLog();
  expect((await life(uncut, [], null)).status).toBe("succeeded");
  const woke = readRunLog(uncut).findIndex(({ kind }) => kind === "node_woken");
  const log = freshLog();
  const ran: string[] = [];
  const done = await life(log, ran, recordAt(uncut, (step) => step["call_id"] === "cG", 1, woke) - 1);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([ran, closersOf(log, ran).closers]).toEqual([["lookup:x", "later:L0", "again:L1"], [["cG", "failed"]]]);
});

test("a host call a resume closed crashed, cut at its closer by the next start, leaves the later calls to run: that start never reached them", async () => {
  // [team_a h1, ticket p, team_a x] [later L1] [again L2]: a crash at node_waiting, h1 answered
  // while the run is down; the resume closes p crashed and is cut at p's closer.
  const ticket = tool({ name: "ticket", description: "Ticket.", parameters: { type: "object", properties: { q: { type: "string" } } },
    isConcurrencySafe: () => true, execute: () => pending() });
  const replies = replying(["team_a", "h1"], ["ticket", "p"], ["team_a", "x"], ["later", "L1"], ["again", "L2"]);
  const flow = await flowOf(`${NOTED}, ticket`);
  const run = (log: string, ran: string[]) =>
    gated({ ...notedOptions(log, ran, replies().slice(modelCalls(log))), tools: { ...notedTools(ran), ticket } }, holding("h1").hook);
  /** Cut the start at record `k1`, answer h1, cut the resume at record `k2` when given; then advance to the end. */
  const life = async (log: string, ran: string[], k1: number, k2: number | null) => {
    await crashAfter(k1, () => runFlow(flow, {}, run(log, ran)));
    await complete(log, heldId(0), { output: "yes" }, { env });
    if (k2 !== null) await crashAfter(k2, () => advance(run(log, ran)));
    return await advance(run(log, ran));
  };
  const uncut = freshLog();
  await runFlow(flow, {}, run(uncut, []));
  const k1 = readRunLog(uncut).findIndex(({ kind }) => kind === "node_waiting");
  const once = freshLog();
  expect((await life(once, [], k1, null)).status).toBe("succeeded");
  const log = freshLog();
  const ran: string[] = [];
  const done = await life(log, ran, k1, recordAt(once, (step) => step["closed"] === "crashed") - 1);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([ran, closersOf(log, ran).closers]).toEqual([["a:x", "later:L1", "again:L2"], [["cp", "crashed"]]]);
});

test.each<[string, () => Response[], (step: ValueRecord) => boolean, string[]]>([
  [
    "its first skip",
    replying(["later", "h"], ["team_a", "a"], ["lookup", "b"], ["team_a", "c"], ["later", "L"]),
    (step) => step["closed"] === "skipped",
    ["skipped", "skipped", "skipped", "skipped"],
  ],
  [
    "the closer of a host call beside the hold",
    replying(["lookup", "h"], ["ticket", "p"], ["team_a", "x"], ["later", "L1"], ["again", "L2"]),
    (step) => step["closed"] === "interrupted",
    ["interrupted", "skipped", "skipped"],
  ],
])("a crash at %s, after an answer and then an interrupt, skips the later calls: the start that began with the interrupt ran none", async (_, replies, cut, words) => {
  const ticket = tool({ name: "ticket", description: "Ticket.", parameters: { type: "object", properties: { q: { type: "string" } } },
    isConcurrencySafe: () => true, execute: () => pending() });
  const flow = await flowOf(`${NOTED}, ticket`);
  const run = (log: string, ran: string[]) =>
    gated({ ...notedOptions(log, ran, replies().slice(modelCalls(log))), tools: { ...notedTools(ran), ticket } }, holding("h").hook);
  /** Run to the wait, answer the hold, interrupt, and advance, cut at record `k` when given. */
  const life = async (log: string, ran: string[], k: number | null) => {
    await runFlow(flow, {}, run(log, ran));
    await complete(log, heldId(0), { output: "yes" }, { env });
    await interrupt(log, "m", { env });
    if (k === null) await advance(run(log, ran));
    else await crashAfter(k, () => advance(run(log, ran)));
  };
  const uncut = freshLog();
  await life(uncut, [], null);
  const log = freshLog();
  const ran: string[] = [];
  await life(log, ran, recordAt(uncut, cut) - 1);
  const ranBefore = [...ran];
  expect((await advance(run(log, ran))).status).toBe("succeeded");
  expect(ran).toEqual(ranBefore);
  expect(closersOf(log, ran).closers.map(([, word]) => word)).toEqual(words);
});

test("two crashes cutting a hold's resumable mate leave the later calls to run, and the later hold to be asked", async () => {
  // [h1 x] [h2] [L0] [L1 L2] [L3]; h1 is answered as it is announced.
  const replies = replying(["lookup", "h1"], ["lookup", "x"], ["again", "h2"], ["later", "L0"], ["team_a", "L1"], ["team_a", "L2"], ["later", "L3"]);
  const flow = await flowOf(NOTED);
  const delivered: Promise<unknown>[] = [];
  const run = (log: string, ran: string[], hook: BeforeToolCall) => {
    const onEvent = (event: RunEvent) => {
      if (event instanceof CallHeld && event.callId === heldId(0)) delivered.push(complete(log, heldId(0), { output: "one" }, { env }));
    };
    return gated({ ...notedOptions(log, ran, replies().slice(modelCalls(log))), onEvent }, hook);
  };
  /** Cut the start at x's result (`k1`), then the resume at x's result again (`k2`) when given. */
  const cut = async (log: string, ran: string[], hook: BeforeToolCall, k1: number, k2: number | null) => {
    await crashAfter(k1, () => runFlow(flow, {}, run(log, ran, hook)));
    await Promise.all(delivered.splice(0));
    if (k2 !== null) await crashAfter(k2, () => advance(run(log, ran, hook)));
  };
  const uncut = freshLog();
  await runFlow(flow, {}, run(uncut, [], holding("h1", "h2").hook));
  await Promise.all(delivered.splice(0));
  const k1 = recordAt(uncut, (step) => step["call_id"] === "cx") - 1;
  const once = freshLog();
  await cut(once, [], holding("h1", "h2").hook, k1, null);
  await advance(run(once, [], holding("h1", "h2").hook));
  const k2 = recordAt(once, (step) => step["call_id"] === "cx") - 1;
  const log = freshLog();
  const ran: string[] = [];
  const { asked, hook } = holding("h1", "h2");
  const warnings: string[] = [];
  const done = await withWarnings((text) => void warnings.push(text), async () => {
    await cut(log, ran, hook, k1, k2);
    return await finish(log, ran, hook, replies);
  });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([warnings, closersOf(log, ran).neverRan]).toEqual([[], []]);
  expect(ran).toEqual(["lookup:x", "lookup:x", "lookup:x", "answer:h2", "later:L0", "a:L1", "a:L2", "later:L3"]);
  // Two evaluations began after h1's answer, which made h2 runnable.
  expect(asked.filter((call) => call.arguments["q"] === "h2").map((call) => call.attempt)).toEqual([2]);
});

test("a call cut alongside a hold is sent again when its tool is resumable, else closed crashed; it never runs fresh", async () => {
  const flow = await flowOf(NOTED);
  const uncut = freshLog();
  await runFlow(flow, {}, gated(notedOptions(uncut, [], heldReplies()), holding("2").hook));
  // Cut right after the hold: its group's other calls started, and their results are lost.
  const log = freshLog();
  const ran: string[] = [];
  const { asked, hook } = holding("2");
  await crashAfter(through(uncut, "pending"), () => runFlow(flow, {}, gated(notedOptions(log, ran, heldReplies()), hook)));
  expect(ran).toEqual(["later:0", "lookup:1", "a:3"]);
  const resumed = await advance(gated(notedOptions(log, ran, heldReplies().slice(1)), hook));
  expect(resumed.status, resumed.failure?.message).toBe("waiting");
  expect(ran).toEqual(["later:0", "lookup:1", "a:3", "lookup:1"]);
  expect(asked.slice(4).map((call) => [call.arguments["q"], call.attempt])).toEqual([["1", 2]]);
  expect(shape(log)).toEqual(["model_call", "tool_result", "pending", "tool_result", "crashed"]);
  expect(pendingCalls(log).map((call) => call.callId)).toEqual([heldId(2)]);
});

test.each([false, true])("calls cut beside a second hold count their attempts from the first answer, and an interrupt (%s) never skips them", async (interrupted) => {
  // [h1] [m] [h2, y, w] [z]. A crash before the first wait makes the start after the answer a
  // second resume, so counting resumes since the reply would give y a third attempt.
  const replies = replying(["lookup", "h1"], ["later", "m"], ["lookup", "h2"], ["lookup", "y"], ["team_a", "w"], ["later", "z"]);
  const flow = await flowOf(NOTED);
  /** The record count up to and with the `nth` hold of the log at `log`. */
  const holdAt = (log: string, nth: number) =>
    readRunLog(log).findIndex(({ record }) => record instanceof NodeStepRecord && record.step["held"] === true && --nth === 0) + 1;
  /** Cut a run at `log` after its first hold, resume it, answer it, and advance; cut there at `k` when given. */
  const run = async (log: string, ran: string[], hook: BeforeToolCall, k: number | null) => {
    await crashAfter(holdAt(uncut, 1), () => runFlow(flow, {}, gated(notedOptions(log, ran, replies()), hook)));
    expect((await advance(gated(notedOptions(log, ran, replies().slice(1)), hook))).status).toBe("waiting");
    await complete(log, heldId(0), { output: "one" }, { env });
    const advancing = () => advance(gated(notedOptions(log, ran, replies().slice(1)), hook));
    if (k === null) expect((await advancing()).status).toBe("waiting");
    else await crashAfter(k, advancing);
  };
  const uncut = freshLog();
  await runFlow(flow, {}, gated(notedOptions(uncut, [], replies()), holding("h1", "h2").hook));
  const whole = freshLog();
  await run(whole, [], holding("h1", "h2").hook, null);
  // Cut after the second hold: y and w started beside it.
  const log = freshLog();
  const ran: string[] = [];
  const { asked, hook } = holding("h1", "h2");
  await run(log, ran, hook, holdAt(whole, 2));
  expect(ran).toEqual(["later:m", "lookup:y", "a:w"]);
  if (interrupted) await interrupt(log, "m", { env });
  const done = await advance(gated(notedOptions(log, ran, replies().slice(1)), hook));
  expect(done.status, done.failure?.message).toBe(interrupted ? "succeeded" : "waiting");
  const closers = stepsOf(log).flatMap((step) => (typeof step["closed"] === "string" ? [[step["call_id"], step["closed"]]] : []));
  if (interrupted) {
    // A resend with an interrupt pending is closed crashed. The resume takes the interrupt as the
    // node opens its inbox, after it read its calls, so the hold pends and is closed at the honor.
    expect([ran, closers]).toEqual([["later:m", "lookup:y", "a:w"], [["cy", "crashed"], ["cw", "crashed"], ["cz", "skipped"], ["ch2", "interrupted"]]]);
  } else {
    expect([ran, closers]).toEqual([["later:m", "lookup:y", "a:w", "lookup:y"], [["cw", "crashed"]]]);
    expect(asked.filter((call) => call.arguments["q"] === "y").map((call) => call.attempt)).toEqual([1, 2]);
  }
});

test.each<[string, () => Response[], string[], string]>([
  ["holds in one group", replying(["lookup", "h1"], ["lookup", "h2"], ["later", "l"]), ["h1"], "l"],
  ["holds in two groups", replying(["lookup", "h1"], ["later", "m"], ["lookup", "h2"], ["later", "z"]), ["h1"], "z"],
])("an interrupt closing one of the %s, after another was answered, skips the waiting calls: none of them ran", async (_, replies, answered, last) => {
  const ran: string[] = [];
  const { hook } = holding("h1", "h2");
  const log = freshLog();
  await runFlow(await flowOf(NOTED), {}, gated(notedOptions(log, ran, replies()), hook));
  for (const q of answered) await complete(log, pendingCalls(log).find((call) => call.arguments!["q"] === q)!.callId, { output: "yes" }, { env });
  expect((await advance(gated(notedOptions(log, ran, replies().slice(1)), hook))).status).toBe("waiting");
  await interrupt(log, "m", { env });
  const done = await advance(gated(notedOptions(log, ran, replies().slice(1)), hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(ran.some((name) => name.endsWith(`:${last}`))).toBe(false);
  expect(stepsOf(log).find((step) => step["call_id"] === `c${last}`)).toMatchObject({ output: SKIPPED_TOOL_RESULT, closed: "skipped" });
});

test("after a hold's answer, an interrupt skips the calls a later start never reached, as without a hold", async () => {
  // [h] [x] [c2] [c3], each alone: the start after the answer runs x, and is cut while c2 runs.
  const replies = replying(["later", "h"], ["lookup", "x"], ["later", "2"], ["later", "3"]);
  const flow = await flowOf(NOTED);
  /** Run to the answer at `log`, then advance, cut at record `k` when given. */
  const run = async (log: string, ran: string[], hook: BeforeToolCall | null, k: number | null) => {
    const options = () => gated(notedOptions(log, ran, replies().slice(modelCalls(log))), hook);
    if (hook === null) return k === null ? await runFlow(flow, {}, options()) : await crashAfter(k, () => runFlow(flow, {}, options()));
    await runFlow(flow, {}, options());
    await complete(log, heldId(0), { output: "one" }, { env });
    return k === null ? await advance(options()) : await crashAfter(k, () => advance(options()));
  };
  for (const hook of [holding("h").hook, null]) {
    const uncut = freshLog();
    await run(uncut, [], hook, null);
    const log = freshLog();
    const ran: string[] = [];
    await run(log, ran, hook, readRunLog(uncut).findIndex(({ record }) => record instanceof NodeStepRecord && record.step["call_id"] === "c2"));
    // Without a hook, the first call ran too.
    const runs = hook === null ? ["later:h", "lookup:x", "later:2"] : ["lookup:x", "later:2"];
    expect(ran).toEqual(runs);
    await interrupt(log, "m", { env });
    const done = await advance(gated(notedOptions(log, ran, []), hook));
    expect(done.status, done.failure?.message).toBe("succeeded");
    const closers = stepsOf(log).flatMap((step) => (typeof step["closed"] === "string" ? [[step["call_id"], step["closed"]]] : []));
    expect([ran, closers]).toEqual([runs, [["c2", "crashed"], ["c3", "skipped"]]]);
  }
});

test.each([false, true])("an interrupt after a hold's answer, before its later calls start (from onEvent: %s), skips them and keeps the answer", async (fromEvent) => {
  const ran: string[] = [];
  const { asked, hook } = holding("2");
  const log = freshLog();
  await runFlow(await flowOf(NOTED), {}, gated(notedOptions(log, ran, heldReplies()), hook));
  await complete(log, heldId(2), { output: "approved" }, { env });
  const delivered: Promise<unknown>[] = [];
  if (!fromEvent) await interrupt(log, "m", { env });
  const onEvent = (event: RunEvent) => {
    if (fromEvent && event instanceof NodeWoken && delivered.length === 0) delivered.push(interrupt(log, "m", { env }));
  };
  const done = await advance(gated({ ...notedOptions(log, ran, []), onEvent }, hook));
  await Promise.all(delivered);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([ran, asked.length]).toEqual([["later:0", "lookup:1", "a:3"], 4]);
  const results = stepsOf(log).flatMap((step) => (step["kind"] === "tool_result" ? [[step["call_id"], step["closed"] ?? step["output"]]] : []));
  expect(results.slice(-3)).toEqual([["c2", "approved"], ["c4", "skipped"], ["c5", "skipped"]]);
});

test("a later call whose check faults waits with the others, and is checked again once the hold is answered", async () => {
  let checks = 0;
  const bad = new FunctionTool({
    name: "bad",
    description: "Bad.",
    parametersJsonSchema: { type: "object", properties: { q: { type: "string" } } },
    parametersValidator: { "~standard": { version: 1, vendor: "test", validate: () => {
      checks += 1;
      throw new Error("no check");
    } } },
    execute: () => "bad",
  });
  const flow = await flowOf("lookup, bad");
  const replies = replying(["lookup", "h"], ["bad", "z"]);
  const ran: string[] = [];
  const { hook } = holding("h");
  const log = freshLog();
  const options = (models: Response[]) => gated({ ...notedOptions(log, ran, models), tools: { ...notedTools(ran), bad } }, hook);
  const first = await runFlow(flow, {}, options(replies()));
  expect([first.status, checks]).toEqual(["waiting", 1]);
  await complete(log, heldId(0), { output: "ok" }, { env });
  const done = await advance(options(replies().slice(1)));
  expect([done.status, checks]).toEqual(["failed", 2]);
  expect(done.failure?.message).toContain("no check");
});

test("a hold beside a tool's own pending() makes the later call wait only for the hold", async () => {
  const ran: string[] = [];
  const ticket = tool({
    name: "ticket",
    description: "Ticket.",
    parameters: { type: "object", properties: { q: { type: "string" } } },
    isConcurrencySafe: () => true,
    execute: (args) => (ran.push(`ticket:${String(args.q)}`), pending()),
  });
  const replies = replying(["ticket", "p"], ["lookup", "h"], ["later", "l"]);
  const { hook } = holding("h");
  const log = freshLog();
  const options = () => gated({ ...notedOptions(log, ran, replies().slice(modelCalls(log))), tools: { ...notedTools(ran), ticket } }, hook);
  expect((await runFlow(await flowOf("ticket, lookup, later"), {}, options())).status).toBe("waiting");
  expect(pendingCalls(log).map((call) => [call.callId, call.held])).toEqual([[heldId(1), true], [heldId(0), false]]);
  await complete(log, heldId(1), { output: "approved" }, { env });
  expect((await advance(options())).status).toBe("waiting");
  expect([ran, pendingCalls(log).map((call) => call.callId)]).toEqual([["ticket:p", "later:l"], [heldId(0)]]);
  await complete(log, heldId(0), { output: "ticket" }, { env });
  expect((await advance(options())).status).toBe("succeeded");
  expect(ran).toEqual(["ticket:p", "later:l"]);
});

test("an interrupt closes a waiting hold, whose closer reads Not run, and skips the later calls; none of them runs", async () => {
  const ran: string[] = [];
  const { hook } = holding("2");
  const log = freshLog();
  await runFlow(await flowOf(NOTED), {}, gated(notedOptions(log, ran, heldReplies()), hook));
  const { calls } = await interrupt(log, "m", { env });
  expect(calls.map((call) => [call.callId, call.held])).toEqual([[heldId(2), true]]);
  const done = await advance(gated(notedOptions(log, ran, []), hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBeNull();
  expect(ran).toEqual(["later:0", "lookup:1", "a:3"]);
  const steps = stepsOf(log);
  expect(steps[5]).toEqual({ kind: "tool_result", call_id: "c2", started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "interrupted" });
  expect(shape(log).slice(5)).toEqual(["interrupted", "skipped", "skipped", "interrupted"]);
});

test("an honored interrupt that closed a hold and skipped the later calls stands across a crash at every record after it", async () => {
  const flow = await flowOf(NOTED);
  /** A run at `log` waiting on its hold, then interrupted while it waits. */
  const interrupted = async (log: string, ran: string[]) => {
    await runFlow(flow, {}, gated(notedOptions(log, ran, heldReplies()), holding("2").hook));
    await interrupt(log, "m", { env });
  };
  const uncut = freshLog();
  await interrupted(uncut, []);
  const start = readRunLog(uncut).length;
  expect((await advance(gated(notedOptions(uncut, [], []), holding("2").hook))).status).toBe("succeeded");
  for (let k = start; k < readRunLog(uncut).length; k++) {
    const log = freshLog();
    const ran: string[] = [];
    await interrupted(log, ran);
    await crashAfter(k, () => advance(gated(notedOptions(log, ran, []), holding("2").hook)));
    const done = await advance(gated(notedOptions(log, ran, []), holding("2").hook));
    expect(done.status, `cut at ${k}: ${done.failure?.message}`).toBe("succeeded");
    expect(ran, `cut at ${k}`).toEqual(["later:0", "lookup:1", "a:3"]);
    const results = stepsOf(log).filter((step) => step["kind"] === "tool_result").map((step) => step["call_id"]);
    expect(results.toSorted(), `cut at ${k}`).toEqual(["c0", "c1", "c2", "c3", "c4", "c5"]);
  }
});

test.each<[string, (event: RunEvent) => boolean, boolean, string[], string[]]>([
  [
    "its hold is announced",
    (event) => event instanceof CallHeld,
    false,
    ["later:0"],
    ["skipped", "interrupted", "skipped", "skipped", "skipped", "interrupted"],
  ],
  [
    "its hold is announced and answered",
    (event) => event instanceof CallHeld,
    true,
    ["later:0"],
    ["tool_result", "skipped", "skipped", "skipped", "skipped", "interrupted"],
  ],
  [
    "a call beside it starts",
    (event) => event instanceof ToolCallStarted && event.toolCallId === "c1",
    false,
    ["later:0", "lookup:1", "a:3"],
    ["tool_result", "tool_result", "interrupted", "skipped", "skipped", "interrupted"],
  ],
])("an interrupt delivered from onEvent as %s closes the hold or keeps its answer, and no later call starts", async (_, when, answer, runs, rest) => {
  const ran: string[] = [];
  const log = freshLog();
  const delivered: Promise<unknown>[] = [];
  const options = notedOptions(log, ran, heldReplies());
  const onEvent = (event: RunEvent) => {
    if (!when(event) || delivered.length > 0) return;
    if (answer) delivered.push(complete(log, heldId(2), { output: "approved" }, { env }));
    delivered.push(interrupt(log, "m", { env }));
  };
  const result = await runFlow(await flowOf(NOTED), {}, gated({ ...options, onEvent }, holding("2").hook));
  await Promise.all(delivered);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual(runs);
  expect(shape(log)).toEqual(["model_call", "tool_result", "pending", ...rest]);
  const closer = stepsOf(log).find((step) => step["call_id"] === "c2" && step["kind"] === "tool_result");
  expect(closer).toMatchObject(answer ? { output: "approved" } : { output: SKIPPED_TOOL_RESULT, closed: "interrupted" });
});

test("a hold answered between its node's wake and the node reading its calls pends again; the later call then runs once, across a crash too", async () => {
  // Two holds in one group, then `later`. The first hold's answer, given as it is announced, wakes
  // the node; the second's lands from onEvent(NodeWoken), before the woken node reads its calls.
  const flow = await flowOf(NOTED);
  const replies = () => [callReply(["lookup", { q: "1" }, "c1"], ["team_a", { q: "2" }, "c2"], ["later", { q: "3" }, "c3"]), textReply("done")];
  /** Run the flow at `log`, answering both holds in flight, cut at record `k` when given. */
  const run = async (log: string, ran: string[], k: number | null) => {
    const { asked, hook } = holding("1", "2");
    const events: RunEvent[] = [];
    const delivered: Promise<unknown>[] = [];
    const onEvent = (event: RunEvent) => {
      events.push(event);
      if (event instanceof CallHeld && event.callId === heldId(0)) delivered.push(complete(log, heldId(0), { output: "one" }, { env }));
      if (event instanceof NodeWoken && delivered.length === 1) delivered.push(complete(log, heldId(1), { output: "two" }, { env }));
    };
    const running = () => runFlow(flow, {}, gated({ ...notedOptions(log, ran, replies()), onEvent }, hook));
    if (k === null) expect((await running()).status).toBe("succeeded");
    else await crashAfter(k, running);
    await Promise.all(delivered);
    return { asked, hook, events };
  };
  const uncut = freshLog();
  const ran: string[] = [];
  const { asked, events } = await run(uncut, ran, null);
  // The wake saw the second answer recorded after it began, so it waited on that call again.
  const waits = events.filter((event) => event instanceof NodeWaiting).map((event) => (event as NodeWaiting).callIds);
  expect(waits).toEqual([[heldId(0), heldId(1)], [heldId(1)]]);
  expect(ran).toEqual(["later:3"]);
  expect(asked.slice(2).map((call) => [call.tool, call.attempt])).toEqual([["later", 1]]);
  // A crash right after the second answer: the next start is the first evaluation to see it.
  const answered = readRunLog(uncut).findIndex(({ record }) => record instanceof NodeStepRecord && record.step["call_id"] === "c2") + 1;
  expect(readRunLog(uncut)[answered - 2]!.kind).toBe("node_woken");
  const log = freshLog();
  const cut: string[] = [];
  const { hook } = await run(log, cut, answered);
  expect(cut).toEqual([]);
  const done = await advance(gated(notedOptions(log, cut, replies().slice(1)), hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(cut).toEqual(["later:3"]);
});

test("a held flow call grows no scope: complete answers it, and its flow never runs", async () => {
  const flow = await loadFlow(SHOUT_FLOW);
  const log = freshLog();
  const hold: BeforeToolCall = () => pending();
  const first = await runFlow(flow, {}, gated({ env, log, runId: "r", models: serving(new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"])], CONFIG)) }, hold));
  expect(first.status, first.failure?.message).toBe("waiting");
  const [call] = pendingCalls(log);
  expect([call!.held, call!.source]).toEqual([true, { kind: "flow", flow: "shout", library: null }]);
  await complete(log, call!.callId, { output: "quiet" }, { env });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const done = await advance(gated({ env, log, models: serving(model) }, hold));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(read(model, 0)).toEqual([["c1", "quiet", false]]);
  expect([expanded(log), scopesOf(log)]).toEqual([false, []]);
});

test.each(["pending", "node_waiting"])("a held module tool call survives a crash after its %s record, and complete answers it with the tool never run", async (kind) => {
  rmSync(quoted, { force: true });
  writeFileSync(join(dir, "held.ts"), quoteModule(["quote"]));
  const flow = await loadFlow(
    `id: held\nname: held\nllm_config: {provider: openai, model: test-model}\nuses:\n  price: "./held.ts:quote"\n` +
      "nodes:\n  reply: {kind: agent, prompt: go, tools: [price]}\noutput: ${reply.output}\n",
    { searchPaths: [dir] },
  );
  const hold: BeforeToolCall = () => pending();
  const options = (log: string, replies: Response[]) => gated({ env, log, runId: "r", models: serving(new ScriptedModel(replies, CONFIG)) }, hold);
  const replies = () => [callReply(["price", { id: "1" }, "c1"]), textReply("done")];
  const uncut = freshLog();
  expect((await runFlow(flow, {}, options(uncut, replies()))).status).toBe("waiting");
  const at = readRunLog(uncut).findIndex(({ kind: recorded, record }) =>
    recorded === kind || (record instanceof NodeStepRecord && record.step["kind"] === kind)) + 1;
  const log = freshLog();
  await crashAfter(at, () => runFlow(flow, {}, options(log, replies())));
  expect((await advance(options(log, []))).status).toBe("waiting");
  const [call] = pendingCalls(log);
  expect([call!.held, call!.source]).toEqual([true, { kind: "module", alias: "price", ref: "./held.ts:quote", export: "quote" }]);
  await complete(log, call!.callId, { output: 7 }, { env });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  expect((await advance({ ...options(log, []), models: serving(model) })).status).toBe("succeeded");
  expect(read(model, 0)).toEqual([["c1", "7", false]]);
  expect(existsSync(quoted)).toBe(false);
});

test("a replayed call whose view is a host's answer its steps don't hold fails the agent as a capability fault", async () => {
  const toolset = new ToolSet(notedTools([])).forRun("r");
  const written: ValueRecord[] = [];
  const first = new ScriptedModel([callReply(["lookup", { q: "1" }, "c1"]), textReply("done")], CONFIG);
  await direct(first, toolset, ["lookup"], { steps: new NodeSteps(1, [], (step) => written.push(step)) });
  // The reply as an earlier start recorded it, without the call's result, which a host's answer says it has.
  const view = {
    callId: "r/chat/1/0", tool: "lookup", resultKey: "c1", state: "answered", closedReason: null,
    result: null, deadline: null, scope: null, failure: null, held: false,
  } as const;
  const resumed: ValueRecord[] = [];
  const steps = new NodeSteps(2, written.slice(0, 1), (step) => 1 + resumed.push(step), [view]);
  const error = await rejected(direct(new ScriptedModel([], CONFIG), toolset, ["lookup"], { steps }));
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe("node 'chat''s call 'r/chat/1/0' is answered with no result recorded");
  expect(resumed).toEqual([]);
});

test.each<[string, "hold" | "run"]>([
  ["the hold's group was being written (skipFrom)", "hold"],
  ["its group-mate ran (closePended, at the honor)", "run"],
])("a release in flight, then an interrupt, skips the released call, which never started, once %s", async (_, when) => {
  // [lookup h (held), lookup x]: the release and the interrupt land as the hold is recorded, or
  // while x runs. Each path settles the reply's pended calls, and finds h released.
  const interrupt = new AbortController();
  let current: CallView[] = [];
  const land = () => {
    current = [{ callId: "r/chat/1/0", tool: "lookup", resultKey: "ch", state: "released", closedReason: null, result: null, deadline: null, scope: null, failure: null, held: true }];
    interrupt.abort();
  };
  const lookup = tool({
    name: "lookup",
    description: "Look up.",
    parameters: { type: "object", properties: { q: { type: "string" } } },
    isConcurrencySafe: () => true,
    execute: (args) => {
      if (when === "run") land();
      return `found ${String(args.q)}`;
    },
  });
  const written: ValueRecord[] = [];
  const record = (step: ValueRecord) => {
    written.push(step);
    if (when === "hold" && step["held"] === true) land();
    return written.length;
  };
  const steps = new NodeSteps(1, [], record, [], undefined, true, { currentCalls: () => current });
  const steerInbox = { inbox: "m", queuedBesidesActions: () => false, read: () => null, interrupt: interrupt.signal, interruptedBy: () => 7 };
  const toolset = new ToolSet({ lookup }).withGate(holding("h").hook).forRun("r");
  const model = new ScriptedModel([callReply(["lookup", { q: "h" }, "ch"], ["lookup", { q: "x" }, "cx"])], CONFIG);
  expect(await direct(model, toolset, ["lookup"], { steps, steerInbox })).toEqual(new Interrupted(7));
  const results = written.flatMap((step) => (step["kind"] === "tool_result" ? [[step["call_id"], step["closed"] ?? step["output"]]] : []));
  expect(results).toEqual(when === "hold" ? [["ch", "skipped"], ["cx", "skipped"]] : [["cx", "found x"], ["ch", "skipped"]]);
});

test("a hold outside a durable run fails the agent before anything of its group is written", async () => {
  const ran: string[] = [];
  const { hook } = asking((call) => (call.tool === "lookup" ? { refuse: "No." } : pending()));
  const model = new ScriptedModel([callReply(["lookup", { q: "x" }, "c1"], ["team_a", { q: "y" }, "c2"])], CONFIG);
  const events: RunEvent[] = [];
  const flow = await loadFlow("id: bare\nname: bare\nllm_config: {provider: openai, model: test-model}\nnodes:\n  reply: {kind: agent, prompt: go, tools: [lookup, team]}\noutput: ${reply.output}\n");
  const options = { env, runId: "r", tools: hostTools(ran), models: serving(model), onEvent: (event: RunEvent) => void events.push(event) };
  const result = await runFlow(flow, {}, gated(options, hook));
  expect(result.status).toBe("failed");
  expect(result.failure?.message).toContain("agent node 'reply': beforeToolCall held tool 'team_a' outside a durable run; give the run a log");
  expect([ran, brackets(events)]).toEqual([[], []]);
});

test("a resend the hook holds is closed crashed with a warning, since it may have run before its cut", async () => {
  const whole = await gatedRun("lookup", oneLookup(), () => undefined);
  const ran: string[] = [];
  const warnings: string[] = [];
  const { asked, hook } = asking((call) => (call.attempt === 1 ? undefined : pending()));
  const { done, log } = await withWarnings(
    (text) => void warnings.push(text),
    () => crashThenResume(through(whole.log, "model_call"), hook, ran),
  );
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([asked.length, ran]).toEqual([2, ["lookup"]]);
  expect(shape(log)).toEqual(["model_call", "crashed", "model_call"]);
  expect(warnings).toEqual(["node 'reply': beforeToolCall held tool 'lookup', sent again after a cut; it may have run, so it is closed crashed"]);
});

// ------------------------------------------------------------------------------------------- //
// What a call acts on (its tool's `gateTarget`), and a tool that never runs without the hook
// (`refuseUngated`).
// ------------------------------------------------------------------------------------------- //

/** What the host knows of each call `approve` may approve, by id; the model writes none of it. */
const HELD_CALLS: Record<string, JsonObject> = {
  h1: { tool: "push", arguments: { branch: "main" } },
  h2: { tool: "rm", arguments: { path: "/" } },
  h3: { tool: "push", arguments: { branch: "dev" } },
};

/**
 * `approve`, noting each run in `ran`, whose target is the host's record of the call it approves,
 * noting the call id it was read for in `seen`; `extra` replaces options.
 */
const approveTool = (ran: string[], seen: string[] = [], extra: Partial<ToolOptions> = {}) =>
  tool({
    name: "approve",
    description: "Approve a held call.",
    parameters: { type: "object", properties: { call: { type: "string" }, why: { type: "string" } }, required: ["call"] },
    gateTarget: (args, ctx) => {
      seen.push(ctx.callId);
      return { call: String(args.call), held: HELD_CALLS[String(args.call)] ?? null };
    },
    execute: (args) => {
      ran.push(`approve:${String(args.call)}`);
      return "approved";
    },
    ...extra,
  });

/** A run's options at `log` over `approve`, with `model`. */
const approveOptions = (log: string, approve: FunctionTool, model: ScriptedModel) => ({
  env, log, runId: "r", tools: { approve }, models: serving(model),
});

test("the hook sees what a call acts on as its tool's target, never the model's words; a run, a refusal and a hold all work", async () => {
  const ran: string[] = [];
  const seen: string[] = [];
  const { asked, hook } = asking((call) => {
    const held = call.target!["held"] as JsonObject;
    if (held["tool"] === "rm") return { refuse: "Never rm." };
    return (held["arguments"] as JsonObject)["branch"] === "main" ? pending() : undefined;
  });
  // The model's words describe harmless calls; the targets hold what the host knows of them.
  const replies = () => [
    callReply(["approve", { call: "h3", why: "a read" }, "c1"], ["approve", { call: "h2" }, "c2"], ["approve", { call: "h1", why: '{"tool": "read"}' }, "c3"]),
    textReply("done"),
  ];
  const log = freshLog();
  const approve = approveTool(ran, seen);
  const first = await runFlow(await flowOf("approve"), {}, gated(approveOptions(log, approve, new ScriptedModel(replies(), CONFIG)), hook));
  expect(first.status, first.failure?.message).toBe("waiting");
  expect(ran).toEqual(["approve:h3"]);
  const targets = ["h3", "h2", "h1"].map((id) => ({ call: id, held: HELD_CALLS[id]! }));
  expect(asked.map((call) => call.target)).toEqual(targets);
  expect(asked[2]!.arguments).toEqual({ call: "h1", why: '{"tool": "read"}' });
  // Read with the context the call's tool would get.
  expect(seen).toEqual(asked.map((call) => call.callId));
  const target = asked[0]!.target!;
  expect([target, target["held"], (target["held"] as JsonObject)["arguments"]].every(Object.isFrozen)).toBe(true);
  expect(stepsOf(log)[2]).toMatchObject({ call_id: "c2", output: "Never rm.", closed: "refused" });
  // The hold records it, so a host rebuilds its card from the log after a restart.
  expect(stepsOf(log)[3]).toMatchObject({ held: true, target: targets[2] });
  expect(pendingCalls(log).map((call) => [call.callId, call.target])).toEqual([["r/reply/1/2", targets[2]!]]);
  await release(log, "r/reply/1/2", { env });
  const model = new ScriptedModel(replies().slice(1), CONFIG);
  const done = await advance(gated(approveOptions(log, approve, model), hook));
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect([ran, asked.length]).toEqual([["approve:h3", "approve:h1"], 3]);
  expect(read(model, 0)).toEqual([["c1", "approved", false], ["c2", "Never rm.", true], ["c3", "approved", false]]);
});

test("a tool without a target, or one giving null, shows the hook null, and a hold records none", async () => {
  const { asked, hook } = asking(() => pending());
  const model = new ScriptedModel([callReply(["approve", { call: "h9" }, "c1"])], CONFIG);
  const log = freshLog();
  const approve = approveTool([], [], { gateTarget: () => null });
  const { status } = await runFlow(await flowOf("approve"), {}, gated(approveOptions(log, approve, model), hook));
  expect(status).toBe("waiting");
  expect(asked.map((call) => call.target)).toEqual([null]);
  expect(Object.hasOwn(stepsOf(log)[1]!, "target")).toBe(false);
  expect(pendingCalls(log).map((call) => call.target)).toEqual([null]);
  const lookups = asking();
  await gatedRun("lookup", oneLookup(), lookups.hook);
  expect(lookups.asked.map((call) => call.target)).toEqual([null]);
});

/** The start of the warning a call's failed target gives. */
const TARGET_FAILED = "node 'reply': tool 'approve''s gateTarget failed: ";

/**
 * Run one call of `approve` with `extra` options in a gated run whose hook runs every call, with
 * the environment `vars` beside the test's; return what happened and the warnings.
 */
async function targetRun(extra: Partial<ToolOptions>, vars: Record<string, string> = {}) {
  const warnings: string[] = [];
  const ran: string[] = [];
  const { asked, hook } = asking();
  const model = new ScriptedModel([callReply(["approve", { call: "h1" }, "c1"]), textReply("done")], CONFIG);
  const log = freshLog();
  const options = { ...approveOptions(log, approveTool(ran, [], extra), model), env: hostEnv(dir, vars) };
  const result = await withWarnings(
    (text) => void warnings.push(text),
    async () => runFlow(await flowOf("approve"), {}, gated(options, hook)),
  );
  return { result, warnings, ran, asked, log };
}

/** Expect `run` refused its call with the failed target's `message`, the model reading `text`, and nothing asked or run. */
function expectTargetRefused(run: Awaited<ReturnType<typeof targetRun>>, text: string, message: RegExp) {
  expect(run.result.status, run.result.failure?.message).toBe("succeeded");
  expect([run.ran, run.asked]).toEqual([[], []]);
  expect(stepsOf(run.log)[1]).toMatchObject({ call_id: "c1", output: text, closed: "refused" });
  expect(run.warnings).toHaveLength(1);
  expect(run.warnings[0]).toStartWith(TARGET_FAILED);
  expect(run.warnings[0]).toEndWith("; the call is refused");
  expect(run.warnings[0]).toMatch(message);
}

test.each<[string, () => unknown, RegExp]>([
  ["a throw", () => { throw new Error("boom"); }, /failed: boom;/],
  ["a list", () => ["x"], /failed: its result must be a JSON object or null, got Array;/],
  ["a Promise", () => Promise.resolve({ a: 1 }), /failed: its result must be a JSON object or null, got Promise;/],
  ["another realm's rejected Promise", () => runInNewContext("Promise.reject(new Error('foreign'))"), /got Promise;/],
  ["a non-JSON value", () => ({ when: new Date(0) }), /failed: its result\.when: Date is not a JSON value;/],
])("a target that is %s refuses the call with a warning, the model reading the failed check's text or the tool's ungated refusal; the hook isn't asked", async (_, gateTarget, message) => {
  for (const refuseUngated of [null, "Ask the user."]) {
    const run = await targetRun({ gateTarget: gateTarget as ToolOptions["gateTarget"], refuseUngated });
    expectTargetRefused(run, refuseUngated ?? GATE_FAILED_TEXT, message);
  }
  // A rejection nothing handled would surface by now.
  await new Promise((resolve) => setTimeout(resolve, 20));
});

test("a target is never cut short: one the hold step fits is shown to the hook, held and read back whole", async () => {
  const big = { text: "x".repeat(10 * 1024) };
  const { asked, hook } = asking(() => pending());
  const model = new ScriptedModel([callReply(["approve", { call: "h1" }, "c1"])], CONFIG);
  const log = freshLog();
  const approve = approveTool([], [], { gateTarget: () => big });
  const { status } = await runFlow(await flowOf("approve"), {}, gated(approveOptions(log, approve, model), hook));
  expect(status).toBe("waiting");
  expect([asked[0]!.target, pendingCalls(log)[0]!.target]).toEqual([big, big]);
});

test("a target that would make the hold step over the run's YIAN_MAX_EVENT_BYTES refuses the call, whatever the hook would say", async () => {
  const message = /failed: its target makes a hold of 3\d\d\d bytes, over the run's limit of 1500 \(YIAN_MAX_EVENT_BYTES\);/;
  for (const refuseUngated of [null, "Ask the user."]) {
    const run = await targetRun({ gateTarget: () => ({ text: "x".repeat(3000) }), refuseUngated }, { YIAN_MAX_EVENT_BYTES: "1500" });
    expectTargetRefused(run, refuseUngated ?? GATE_FAILED_TEXT, message);
  }
});

test("a tool's target reads the validator's output, the very argument its execute gets", async () => {
  const targets: unknown[] = [];
  const executed: unknown[] = [];
  const validate = (value: unknown) => ({ value: { call: String((value as JsonObject)["call"]).trim().toUpperCase() } });
  const approve = new FunctionTool({
    name: "approve",
    description: "Approve a held call.",
    parametersJsonSchema: { type: "object", properties: { call: { type: "string" } }, required: ["call"] },
    parametersValidator: { "~standard": { version: 1, vendor: "test", validate } },
    gateTarget: (args) => {
      targets.push(args);
      return { call: String(args["call"]) };
    },
    execute: (args) => {
      executed.push(args);
      return "approved";
    },
  });
  const { asked, hook } = asking();
  const model = new ScriptedModel([callReply(["approve", { call: "  h1 " }, "c1"]), textReply("done")], CONFIG);
  const result = await runFlow(await flowOf("approve"), {}, gated(approveOptions(freshLog(), approve, model), hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect([asked[0]!.arguments, asked[0]!.target]).toEqual([{ call: "  h1 " }, { call: "H1" }]);
  expect(targets).toHaveLength(1);
  expect(targets[0]).toBe(executed[0]);
});

test("a tool that refuses to run ungated answers a call in a run without the hook with its text, and never runs; a hook is asked as usual", async () => {
  const ran: string[] = [];
  const seen: string[] = [];
  const text = "Only the user can approve or refuse a held call: ask them in the chat.";
  const approve = approveTool(ran, seen, { refuseUngated: text });
  // Refused whatever its arguments: here they miss `call`.
  const replies = () => [callReply(["approve", { why: "x" }, "c1"], ["approve", { call: "h3" }, "c2"]), textReply("done")];
  const model = new ScriptedModel(replies(), CONFIG);
  const events: RunEvent[] = [];
  const log = freshLog();
  const options = { ...approveOptions(log, approve, model), onEvent: (event: RunEvent) => void events.push(event) };
  const ungated = await runFlow(await flowOf("approve"), {}, gated(options, null));
  expect(ungated.status, ungated.failure?.message).toBe("succeeded");
  expect([ran, seen]).toEqual([[], []]);
  expect(read(model, 1)).toEqual([["c1", text, true], ["c2", text, true]]);
  expect(stepsOf(log)[1]).toMatchObject({ call_id: "c1", started_at: null, output: text, is_error: true });
  const refusal = "ReportedToolError";
  expect(brackets(events)).toEqual([["started", "c1"], ["finished", "c1", refusal], ["started", "c2"], ["finished", "c2", refusal]]);

  const { asked, hook } = asking();
  const gatedModel = new ScriptedModel(replies(), CONFIG);
  const withHook = await runFlow(await flowOf("approve"), {}, gated(approveOptions(freshLog(), approve, gatedModel), hook));
  expect(withHook.status, withHook.failure?.message).toBe("succeeded");
  expect([ran, asked.map((call) => call.callId)]).toEqual([["approve:h3"], ["r/reply/1/1"]]);
  expect(read(gatedModel, 1)[1]).toEqual(["c2", "approved", false]);
});

test("a tool that refuses to run ungated never runs from a TOOL node, gated or not, so a flow wrapping it can't reach it", async () => {
  const flow = await loadFlow(`
id: wrapper
name: wrapper
llm_config: {provider: openai, model: test-model}
defs:
  wrap:
    description: Wrap approve.
    input:
      call: !str
    nodes:
      go: {kind: call, call: approve, input: {call: "\${input.call}"}}
    output: \${go.output}
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [wrap]
    output: !Optional(str)
output: \${reply.output}
`);
  const text = "Only the user can approve.";
  for (const hook of [null, asking().hook]) {
    const ran: string[] = [];
    const model = new ScriptedModel([callReply(["wrap", { call: "h1" }, "c1"]), textReply("done")], CONFIG);
    const approve = approveTool(ran, [], { refuseUngated: text });
    const result = await runFlow(flow, {}, gated(approveOptions(freshLog(), approve, model), hook));
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(ran).toEqual([]);
    expect(read(model, 1)).toEqual([
      ["c1", `Flow wrap failed at node go (node.execute): tool 'approve' never runs unchecked, and nothing checks a TOOL node's call: ${text}`, true],
    ]);
  }
});
