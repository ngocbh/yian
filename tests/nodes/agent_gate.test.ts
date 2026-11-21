/**
 * The host's check before each tool call an agent's model makes (`beforeToolCall`): it runs a call,
 * with an audit note the model never reads, or refuses it with a text the model reads as the
 * call's error. It is asked about a whole group before any of the group's calls starts, and a call
 * never starts once the node stopped or the agent took an interrupt while the hook decided.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { rebuildMessages } from "../../src/nodes/agent/messages.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { withWarnings } from "../../src/platform.ts";
import { advance, agentCall, interrupt, runFlow } from "../../src/run.ts";
import { NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { type BeforeToolCall, GATE_FAILED_TEXT, type GatedCall, type GateDecision } from "../../src/tools/gate.ts";
import { registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
import { gated } from "../helpers/gate.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { callReply, crashAfter, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

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
  hook: BeforeToolCall,
  extra: { log?: string; ran?: string[]; signal?: AbortSignal; onEvent?: (event: RunEvent) => void } = {},
) {
  const { log = freshLog(), ran = [], ...rest } = extra;
  const events: RunEvent[] = [];
  const options = { env, log, runId: "r", tools: hostTools(ran), models: serving(model), onEvent: (event: RunEvent) => void events.push(event) };
  const result = await runFlow(await flowOf(grants), {}, gated({ ...options, ...rest }, hook));
  return { result, log, events, ran, options };
}

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
    const { asked, hook } = asking(() => ({ refuse: "No weather." }));
    const model = new ScriptedModel([callReply(["gate_weather", {}, "c1"]), textReply("done")], CONFIG);
    const result = await runFlow(await flowOf("gate_weather"), {}, gated({ env, models: serving(model) }, hook));
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(asked.map((call) => call.source)).toEqual([{ kind: "host", group: null, tool: "gate_weather" }]);
    expect(read(model, 1)).toEqual([["c1", "No weather.", true]]);
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
  ["an unknown key", () => ({ run: true, why: "x" }), /./],
  ["run and refuse", () => ({ run: true, refuse: "x" }), /./],
  ["a class instance", () => new (class { run = true })(), /./],
  ["an empty refusal", () => ({ refuse: "" }), /refuse must be a non-empty string/],
  ["a refusal that isn't text", () => ({ refuse: 5 }), /refuse must be a string/],
  ["an oversized refusal", () => ({ refuse: "é".repeat(2049) }), /over the limit of 4096/],
  ["an oversized note", () => ({ run: true, note: "x".repeat(4097) }), /note is 4097 UTF-8 bytes/],
  ["a note with a lone surrogate", () => ({ run: true, note: "a\uD800" }), /lone surrogate/],
  ["a Promise of a bad shape", () => Promise.resolve({ run: "yes" }), /./],
  ["pending()", () => pending(), /pending\(\)/],
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
  const skill = new Skill({ name: "review", description: "Review.", instructions: "Read it." });
  const output = await runAgent({
    nodeId: "chat",
    messages: [userMessage("go")],
    tools: ["lookup", "ask_user"],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: 2,
    toolset,
    systemPrompt: DEFAULT_SYSTEM,
    skills: [skill],
    models: serving(model),
    sent: () => [],
  });
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

test("the hook isn't asked about a TOOL node's call or a map's", async () => {
  const { asked, hook } = asking();
  const flow = await loadFlow(`
id: nodes
name: nodes
nodes:
  direct: {kind: call, call: lookup, input: {q: one}}
  each: {kind: map, over: [a, b], call: lookup, input: {q: "\${item}"}}
output: \${each.output}
`);
  const ran: string[] = [];
  const result = await runFlow(flow, {}, gated({ env, tools: hostTools(ran) }, hook));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect([asked, ran]).toEqual([[], ["lookup", "lookup", "lookup"]]);
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

test("a resend after a crash asks the hook again with its attempt; with an interrupt pending it is closed crashed without asking", async () => {
  const kindAt = (log: string, kind: string) =>
    readRunLog(log).findIndex(({ record }) => record instanceof NodeStepRecord && record.step["kind"] === kind) + 1;
  const whole = await gatedRun("lookup", oneLookup(), () => undefined);
  const flow = await flowOf("lookup");
  // Cut before the call's result, it is sent again; cut after, it is replayed.
  for (const [at, attempts] of [[kindAt(whole.log, "model_call"), [1, 2]], [kindAt(whole.log, "tool_result"), [1]]] as const) {
    const log = freshLog();
    const { asked, hook } = asking();
    const options = { env, log, runId: "r", tools: hostTools([]), models: serving(oneLookup()) };
    await crashAfter(at, () => runFlow(flow, {}, gated(options, hook)));
    const done = await advance(gated({ ...options, models: serving(new ScriptedModel([textReply("done")], CONFIG)) }, hook));
    expect(done.status, done.failure?.message).toBe("succeeded");
    expect(asked.map((call) => call.attempt)).toEqual([...attempts]);
  }
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
  const { asked, signals, hook } = asking(() => {
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
  expect(asked.map((call) => [call.node, call.tool])).toEqual([["helper/review#0/sub", "lookup"]]);
  expect((signals[0]!.reason as Error).name).toBe("ScopeEnded");
  expect(ran).toEqual([]);
  expect(stepsOf(log, "helper/review#0/sub").map((step) => step["kind"])).toEqual(["model_call"]);
});
