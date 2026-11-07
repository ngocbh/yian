/**
 * An AGENT's tool calls replayed across cuts, exhaustively over small replies. Each reply makes two
 * or three calls, each to a tool that is unsafe and resumable, safe and resumable, safe and not
 * resumable, or that pends, resumable or not. Its run is cut at every point of the reply's tool
 * phase and resumed, across up to three starts. Each later start runs with no interrupt, or with
 * one delivered as it begins or while it sends its first or second call again. A run that waits on
 * a pended call gets its answer and goes on. Every finished run must hold:
 * - a call that ran in any start is never recorded `skipped`;
 * - once the interrupt is pending, no call that hadn't run before it runs;
 * - each call ends with exactly one `tool_result` step;
 * - each start runs calls in call order, each at most once;
 * - with no interrupt, nothing is skipped, each resumable call that doesn't pend ends with the
 *   result it gave, a call that isn't resumable runs at most once, and the agent answers.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { advance, complete, interrupt, pendingCalls, type RunResult, runFlow } from "../../src/run.ts";
import { decodeRecord, NodeStepRecord, type RecordEnvelope } from "../../src/runtime/log.ts";
import { pending, type ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { NO_ENV, textReply } from "../helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-replay-convergence-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 10, 7, 10, 0, 0, 0);
/** The most starts a scenario runs, wakes after an answer aside. */
const STARTS = 3;

/** The kinds of tool a call can go to. */
type Kind = "unsafe" | "safe" | "plain" | "pends" | "waits";

const KINDS: Record<Kind, { name: string; safe: boolean; resumable: boolean; pends: boolean }> = {
  unsafe: { name: "save", safe: false, resumable: true, pends: false },
  safe: { name: "look", safe: true, resumable: true, pends: false },
  plain: { name: "note", safe: true, resumable: false, pends: false },
  pends: { name: "hold", safe: false, resumable: false, pends: true },
  // A later start sends it again when it was cut before its pending step, and writes that step.
  waits: { name: "wait", safe: false, resumable: true, pends: true },
};

/** When a start delivers the interrupt: not at all, as it begins, or while it sends its `n`-th call again. */
type Mode = { readonly when: "never" } | { readonly when: "begin" } | { readonly when: "resend"; readonly n: number };

const MODES: readonly Mode[] = [{ when: "never" }, { when: "begin" }, { when: "resend", n: 1 }, { when: "resend", n: 2 }];

/** One call of a tool: the start that made it, the call's place in the reply, and its attempt. */
interface Invoked {
  readonly start: number;
  readonly index: number;
  readonly attempt: number;
}

/** Where a scenario stands between starts. */
interface World {
  /** The run log so far; `null` before the first start. */
  readonly log: Buffer | null;
  readonly invoked: readonly Invoked[];
  /** How many calls had been made when the interrupt was delivered, or `null`. */
  readonly interruptAt: number | null;
  /** The starts so far, the first `runFlow` and each `advance`. */
  readonly starts: number;
}

/** What the tools and the model of the start running now update. */
interface Live {
  invoked: Invoked[];
  interruptAt: number | null;
  /** The calls this start sent again. */
  resends: number;
  readonly start: number;
  readonly mode: Mode;
  readonly path: string;
  readonly delivered: Promise<unknown>[];
}

let live: Live | null = null;

/** Deliver the interrupt to the running start, once. */
function deliver(state: Live): void {
  if (state.interruptAt !== null) return;
  state.interruptAt = state.invoked.length;
  state.delivered.push(interrupt(state.path, "notes"));
}

/** The tool for `kind`: it records each call, and pends or answers `<name>:<index>`. */
function kindTool(kind: Kind): FunctionTool {
  const { name, safe, resumable, pends } = KINDS[kind];
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    isConcurrencySafe: safe ? () => true : null,
    resumable,
    execute: (_args, ctx: ToolContext) => {
      const state = live!;
      const index = Number(ctx.callId.split("/").at(-1));
      state.invoked.push({ start: state.start, index, attempt: ctx.attempt });
      if (ctx.attempt >= 2) {
        state.resends += 1;
        if (state.mode.when === "resend" && state.mode.n === state.resends) deliver(state);
      }
      return pends ? pending() : `${name}:${index}`;
    },
  });
}

const TOOLS = Object.fromEntries(Object.keys(KINDS).map((kind) => [KINDS[kind as Kind].name, kindTool(kind as Kind)]));

const FLOW = `
id: replay
name: replay
llm_config: {provider: anthropic, model: claude-test}
input:
  notes: !Inbox(str)
nodes:
  writer:
    kind: agent
    tools: [save, look, note, hold, wait]
    steer_inbox: \${input.notes}
    prompt: hi
    output: Optional(str)
output: \${writer.output}
`;

/** The reply calling a tool of each of `kinds`, in order; call `i` has the id `c<i>`. */
function reply(kinds: readonly Kind[]): Response {
  const parts = kinds.map((kind, index) => new ToolCallPart({ id: `c${index}`, name: KINDS[kind].name, arguments: { topic: "t" } }));
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** The records of a log's bytes. */
function decoded(log: Buffer): RecordEnvelope[] {
  return log
    .toString("utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => decodeRecord(line));
}

/** The `node_step` of `writer` an envelope holds, or `null`. */
function writerStep(envelope: RecordEnvelope): Record<string, unknown> | null {
  const { record } = envelope;
  return record instanceof NodeStepRecord && record.nodeId === "writer" ? record.step : null;
}

let paths = 0;

/** A fresh log path holding `log`, if given. */
function logAt(log: Buffer | null): string {
  const dir = join(root, `run-${++paths}`);
  mkdirSync(dir);
  const path = join(dir, "log.jsonl");
  if (log !== null) writeFileSync(path, log);
  return path;
}

/**
 * Run one start of the scenario at `world` in `mode`: the first `runFlow`, or an `advance`. With
 * `cutAt`, the append of record `cutAt` crashes it. Returns the new world, the run's result
 * (`null` when cut) and how many calls the start sent again.
 */
async function runStart(
  world: World,
  flow: ComposedFlow,
  first: Response,
  mode: Mode,
  cutAt: number | null,
): Promise<{ world: World; result: RunResult | null; resends: number }> {
  const path = logAt(world.log);
  const state: Live = {
    invoked: [...world.invoked],
    interruptAt: world.interruptAt,
    resends: 0,
    start: world.starts + 1,
    mode,
    path,
    delivered: [],
  };
  live = state;
  const models = {
    env: NO_ENV,
    createModel: () => {
      if (mode.when === "begin") deliver(state);
      return new ScriptedModel(world.log === null ? [first, textReply("done")] : [textReply("done")], CONFIG);
    },
  };
  const options = {
    env: NO_ENV,
    scheduler: virtualScheduler(START + world.starts * 60_000),
    random: () => 1,
    models,
    tools: TOOLS,
    log: path,
  };
  const go = () => (world.log === null ? runFlow(flow, {}, { ...options, runId: "r" }) : advance({ ...options, flow }));
  let result: RunResult | null = null;
  const crash = new Error(`crash at record ${cutAt}`);
  if (cutAt === null) {
    result = await go();
  } else {
    const refuse = (log: { path: string | null; nextSequence: number }) => {
      if (log.path !== null && log.nextSequence >= cutAt) throw crash;
    };
    const outcome = await withAppendHook(refuse, () => go().then(() => null, (error: unknown) => error));
    expect(outcome).toBe(crash);
  }
  for (const delivery of await Promise.allSettled(state.delivered)) {
    if (delivery.status === "fulfilled") continue;
    // The cut refused the interrupt's own record, so it never landed.
    expect(delivery.reason).toBe(crash);
    state.interruptAt = world.interruptAt;
  }
  live = null;
  const next = { log: readFileSync(path), invoked: state.invoked, interruptAt: state.interruptAt, starts: state.start };
  return { world: next, result, resends: state.resends };
}

/**
 * Run the scenario at `world` to its end: one uncut start in `mode`, then, while the run waits on
 * a pended call, its answer and another start. Returns the end, its result, and the first start's
 * records, first seq and calls sent again.
 */
async function finish(
  world: World,
  flow: ComposedFlow,
  first: Response,
  mode: Mode,
): Promise<{ end: World; result: RunResult; records: RecordEnvelope[]; from: number; resends: number }> {
  const from = world.log === null ? 0 : decoded(world.log).length;
  let { world: end, result, resends } = await runStart(world, flow, first, mode, null);
  const records = decoded(end.log!);
  for (let wakes = 0; result!.status === "waiting"; wakes += 1) {
    expect(wakes).toBeLessThan(4);
    const path = logAt(end.log);
    for (const call of pendingCalls(path)) await complete(path, call.callId, { output: "held" }, { flow, env: NO_ENV });
    ({ world: end, result } = await runStart({ ...end, log: readFileSync(path) }, flow, first, { when: "never" }, null));
  }
  return { end, result: result!, records, from, resends };
}

/**
 * The records a start that began at seq `from` may be cut at: from after the reply's `model_call`,
 * and after a resume's `node_started` (a cut before it records no start, as if none had run), to
 * the end of the reply's tool phase: the agent's next model call, its honor of an interrupt, its
 * wait or its output.
 */
function cutPoints(records: readonly RecordEnvelope[], from: number): number[] {
  const reply = records.find((envelope) => writerStep(envelope)?.["kind"] === "model_call");
  if (reply === undefined) return [];
  const low = Math.max(from === 0 ? 0 : from + 2, reply.seq + 1);
  const end = records.find((envelope) => {
    if (envelope.seq < low) return false;
    const step = writerStep(envelope);
    if (step !== null) return step["kind"] === "model_call" || step["kind"] === "interrupted";
    const nodeId = (envelope.record as { nodeId?: string }).nodeId;
    return envelope.kind === "node_waiting" || (envelope.kind === "pool_commit" && nodeId === "writer");
  });
  const high = end?.seq ?? records.at(-1)!.seq;
  return Array.from({ length: Math.max(0, high - low + 1) }, (_, i) => low + i);
}

/** Check a finished scenario's invariants; `label` names it in failures. */
function checkEnd(label: string, kinds: readonly Kind[], end: World, result: RunResult): void {
  expect(result.status, `${label}: ${result.failure?.message}`).toBe("succeeded");
  const writer = decoded(end.log!).flatMap((envelope) => {
    const step = writerStep(envelope);
    return step === null ? [] : [step];
  });
  const steps = writer.filter((step) => step["kind"] === "tool_result");
  const interrupted = end.interruptAt !== null;
  expect(result.output, label).toBe(interrupted ? null : "done");
  for (const [index, kind] of kinds.entries()) {
    const own = steps.filter((step) => step["call_id"] === `c${index}`);
    // Each call ends with exactly one result.
    expect(own.length, `${label}: c${index} results`).toBe(1);
    const closed = own[0]!["closed"];
    // A call that ran is never skipped.
    if (end.invoked.some((each) => each.index === index)) expect(closed, `${label}: c${index} ran`).not.toBe("skipped");
    if (!interrupted) {
      expect(closed, `${label}: c${index} without an interrupt`).not.toBe("skipped");
      if (KINDS[kind].resumable && !KINDS[kind].pends) expect(closed, `${label}: c${index} is resumable`).toBeUndefined();
      else if (!KINDS[kind].resumable) expect(end.invoked.filter((each) => each.index === index).length, `${label}: c${index} runs once`).toBeLessThanOrEqual(1);
      else if (closed !== undefined) {
        // A resumable call that pends is closed only once it pended and a resume closed its call:
        // one cut before its pending step is sent again.
        expect(closed, `${label}: c${index} waits, closed`).toBe("crashed");
        const pended = writer.filter((step) => step["kind"] === "pending" && String(step["call_id"]).endsWith(`/${index}`));
        expect(pended.length, `${label}: c${index} crashed with no pending step`).toBe(1);
      }
    }
  }
  // Once the interrupt is pending, only calls that had run before it run.
  if (end.interruptAt !== null) {
    const before = new Set(end.invoked.slice(0, end.interruptAt).map((each) => each.index));
    for (const each of end.invoked.slice(end.interruptAt)) {
      expect(before.has(each.index), `${label}: c${each.index} ran after the interrupt`).toBe(true);
    }
  }
  // Each start runs calls in call order, each at most once.
  for (let start = 1; start <= end.starts; start += 1) {
    const order = end.invoked.filter((each) => each.start === start).map((each) => each.index);
    expect(order.every((index, at) => at === 0 || index > order[at - 1]!), `${label}: start ${start} ran ${order}`).toBe(true);
  }
}

/**
 * Every reply of two calls over the kinds, and of three over the resumable kinds and the pending
 * ones, with at most one call that pends. A third call adds a group before or after a pair, which a
 * second call that is never sent again (the non-resumable safe kind, or another pending one) only
 * closes, as the one already there does.
 */
function replies(): Kind[][] {
  const pairs = (kinds: readonly Kind[]) => kinds.flatMap((a) => kinds.map((b) => [a, b]));
  const three: Kind[] = ["unsafe", "safe", "pends", "waits"];
  const triples = pairs(three).flatMap((pair) => three.map((c) => [...pair, c]));
  return [...pairs(Object.keys(KINDS) as Kind[]), ...triples.filter((kinds) => kinds.filter((kind) => KINDS[kind].pends).length <= 1)];
}

test("an agent's tool calls replayed across up to three starts, cut anywhere in the tool phase, converge", async () => {
  const flow = new ComposedFlow((await loadFlow(FLOW)).compiled);
  let scenarios = 0;
  let starts = 0;
  /**
   * From `world`, run the next start in each mode to its end, a scenario to check, and cut it at
   * each point to go on from there. A mode whose interrupt never came is the scenario of no
   * interrupt, so it is left out.
   */
  const explore = async (kinds: Kind[], world: World, cuts: readonly string[]): Promise<void> => {
    const first = reply(kinds);
    // The first start has no interrupt: it makes the reply, which nothing could interrupt yet.
    const modes = world.log === null || world.interruptAt !== null ? MODES.slice(0, 1) : MODES;
    // A start runs as with no interrupt until the interrupt lands, so one sent during a call sent
    // again needs that many calls sent again with none.
    let sentAgain = 0;
    for (const mode of modes) {
      if (mode.when === "resend" && mode.n > sentAgain) continue;
      const { end, result, records, from, resends } = await finish(world, flow, first, mode);
      starts += end.starts - world.starts;
      if (mode.when === "never") sentAgain = resends;
      if (mode.when !== "never" && end.interruptAt === world.interruptAt) continue;
      const label = `[${kinds.join(", ")}] ${[...cuts, JSON.stringify(mode)].join(" ")}`;
      scenarios += 1;
      checkEnd(label, kinds, end, result);
      if (world.starts + 1 === STARTS) continue;
      // A cut before the interrupt's record is the start with no interrupt, explored already.
      const landed = mode.when === "never" ? -1 : records.find((each) => each.kind === "interrupted" && each.seq >= from)!.seq;
      for (const k of cutPoints(records, from).filter((each) => each > landed)) {
        const { world: cut } = await runStart(world, flow, first, mode, k);
        starts += 1;
        // Cut before its interrupt came, the start is the one with no interrupt, explored already.
        if (mode.when !== "never" && cut.interruptAt === world.interruptAt) continue;
        await explore(kinds, cut, [...cuts, `${JSON.stringify(mode)} cut at ${k}`]);
      }
    }
  };
  const all = replies();
  for (const kinds of all) await explore(kinds, { log: null, invoked: [], interruptAt: null, starts: 0 }, []);
  console.log(`replay convergence: ${scenarios} scenarios over ${all.length} replies, ${starts} starts`);
  expect(scenarios).toBeGreaterThan(0);
}, 120_000);
