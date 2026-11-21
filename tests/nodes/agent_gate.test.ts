/**
 * The host's check before each tool call an agent's model makes (`beforeToolCall`): it runs a call,
 * with an audit note the model never reads, or refuses it with a text the model reads as the
 * call's error. It is asked about a whole group before any of the group's calls starts, and a call
 * never starts once the node stopped or the agent took an interrupt while the hook decided. Every
 * source is asked about: host, MCP, module, memory and flow tools, a refused flow call starting no
 * flow, and the agents inside a granted flow with the flow calls enclosing them.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, AgentLoopError, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { Interrupted } from "../../src/nodes/agent/interrupts.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { rebuildMessages } from "../../src/nodes/agent/messages.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { withWarnings } from "../../src/platform.ts";
import { advance, agentCall, interrupt, runFlow } from "../../src/run.ts";
import { NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { type BeforeToolCall, GATE_FAILED_TEXT, type GatedCall, type GateDecision, type ToolSource } from "../../src/tools/gate.ts";
import { registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
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
    { callId: `r/reply/${n}/0`, node: "reply", tool: "lookup", source: { kind: "host", group: null, tool: "lookup" }, arguments: { q: "x" }, scopes: [], attempt: 1 },
    { callId: `r/reply/${n}/1`, node: "reply", tool: "team_a", source: { kind: "host", group: "team", tool: "a" }, arguments: { q: "y" }, scopes: [], attempt: 1 },
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
  ["pending()", () => pending(), /a hold \(pending\(\)\) isn't supported yet/],
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

test("a fault decided at once and a Promise still out when an interrupt lands write only skips, with no fault", async () => {
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
 * memory tool `lessons_append`, the flow `shout`, the library's def `billing_quote` and `lookup`.
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
    tools: [price, {crm: [rate]}, {lessons: [append]}, shout, {billing: [quote]}, lookup]
output: \${reply.output}
`;

/** Where `quote.ts`'s tools note each run. */
const quoted = join(dir, "quoted");

/**
 * Run `OWN_FLOW` with `model`, gated by `hook`, once its files are written: `quote.ts`, whose
 * tools `quote` and `rate` note each run in `quoted` (cleared first), and the library `billing`.
 */
async function ownRun(model: ScriptedModel, hook: BeforeToolCall, extra: { env?: typeof env } = {}) {
  rmSync(quoted, { force: true });
  const each = (name: string) =>
    `export const ${name} = { name: "${name}", description: "${name}.", parameters: { type: "object", properties: { id: { type: "string" } } }, ` +
    `execute: () => { appendFileSync(${JSON.stringify(quoted)}, "${name} "); return 1; } };\n`;
  writeFileSync(join(dir, "quote.ts"), `import { appendFileSync } from "node:fs";\n${each("quote")}${each("rate")}`);
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
])("the hook is asked about a call of %s with its source, and its refusal leaves the tool unrun", async (_, name, args, source) => {
  const { asked, hook } = asking((call) => (call.tool === "lookup" ? undefined : { refuse: "Not yours." }));
  const model = new ScriptedModel([callReply([name, args, "c1"], ["lookup", { q: "x" }, "c2"]), textReply("done")], CONFIG);
  const { result, log, ran, applied, quotes } = await ownRun(model, hook);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const n = shape(log).indexOf("model_call") + 1;
  expect(asked).toEqual([
    { callId: `r/reply/${n}/0`, node: "reply", tool: name, source, arguments: args, scopes: [], attempt: 1 },
    { callId: `r/reply/${n}/1`, node: "reply", tool: "lookup", source: { kind: "host", group: null, tool: "lookup" }, arguments: { q: "x" }, scopes: [], attempt: 1 },
  ]);
  expect(Object.isFrozen(asked[0]!.source)).toBe(true);
  // A refused flow call grows no scope.
  expect([ran, quotes, applied, expanded(log)]).toEqual([["lookup"], null, [], false]);
  expect(read(model, 1)).toEqual([
    ["c1", "Not yours.", true],
    ["c2", "lookup:x", false],
  ]);
});

test("a hook that fails on a flow call refuses it, and its flow never starts", async () => {
  const model = new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"]), textReply("done")], CONFIG);
  const warnings: string[] = [];
  const { result, log } = await withWarnings(
    (text) => void warnings.push(text),
    () => ownRun(model, () => { throw new Error("boom"); }),
  );
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect([shape(log), expanded(log), warnings.length]).toEqual([["model_call", "refused", "model_call"], false, 1]);
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

test("an allowed flow call whose arguments are too large to record keeps its note on its error result", async () => {
  const model = new ScriptedModel([callReply(["shout", { text: "x".repeat(5000) }, "c1"]), textReply("done")], CONFIG);
  const { result, log } = await ownRun(model, () => ({ run: true, note: "big" }), { env: hostEnv(dir, { YIAN_MAX_EVENT_BYTES: "2000" }) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(stepsOf(log)[1]).toMatchObject({ kind: "tool_result", call_id: "c1", is_error: true, note: "big" });
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

test("an agent inside a granted flow is asked with the flow calls enclosing it, outermost first; an authored call adds none", async () => {
  const flow = await loadFlow(`
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
`);
  const { asked, hook } = asking();
  const replies = [
    callReply(["outer", {}, "c1"]),
    callReply(["inner", {}, "c2"]),
    callReply(["lookup", { q: "x" }, "c3"]),
    textReply("leaf"),
    textReply("mid"),
    textReply("top"),
    callReply(["lookup", { q: "y" }, "c4"]),
    textReply("solo"),
  ];
  const ran: string[] = [];
  const options = { env, log: freshLog(), runId: "r", tools: hostTools(ran), models: serving(new ScriptedModel(replies, CONFIG)) };
  const result = await runFlow(flow, {}, gated(options, hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outer = { scope: "top/outer#0", callId: "r/top/1/0" };
  const inner = { scope: "top/outer#0/mid/inner#0", callId: "r/top/outer#0/mid/1/0" };
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

test("a node's own tools named ask_user and load_skill are asked about: the exemption follows the tool the name reaches", async () => {
  const { asked, hook } = asking();
  const own = (name: string) => tool({ name, description: "Own.", parameters: { type: "object", properties: {} }, execute: () => name });
  const source = (flow: string) => ({ kind: "flow", flow, library: null }) as const;
  const sources = new Map(["ask_user", "load_skill"].map((name) => [JSON.stringify([null, name]), source(name)]));
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
  const steerInbox = { inbox: "m", queued: () => false, read: () => null, interrupt: interrupt.signal, interruptedBy: () => null };
  const output = await direct(model, toolset.forRun("r"), ["lookup", "team"], { signal: node.signal, steerInbox });
  expect((output as { value: unknown }).value).toBe("done");
  expect([node.live.size, interrupt.live.size]).toEqual([0, 0]);
});
