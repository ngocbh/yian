/**
 * Builders shared by the inbox and wait reducer tests (`waits_reducer.test.ts`,
 * `wait_scenarios.test.ts`): a flow whose START declares an inbox, exact TOOL nodes for `take`,
 * `sleep` and `ask`, and a JSONL log and state past the run start. A `*.test.ts` imported by
 * another test file registers its tests twice, so these live here.
 */
import { mkdtempSync } from "node:fs";
import { dirname, join } from "node:path";

import { stampExecutionScopes } from "../../src/compile/execution_layout.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { JsonlRunLog, readRunLog, type RunLog, type TakeCommit, type WaitEntry } from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  prepareInboxClosed,
  prepareInterrupt,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareNodeWaiting,
  prepareNodeWoken,
  prepareMessagesCommit,
  preparePoolCommit,
  type PreparedTransition,
  replayRecords,
  prepareRunResumed,
  prepareRunStarted,
  prepareRunWaiting,
  prepareSend,
  prepareTakeCommit,
  prepareWithdraw,
  RunState,
} from "../../src/runtime/state.ts";
import type { RetainedTrace } from "../../src/runtime/trace.ts";
import { toolCallId } from "../../src/tools/context.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { InboxRef, inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { FINGERPRINT, FIXED_CLOCK, forwardCapture, noGrowth, replayTraceSnapshot, RUN_ID } from "./run_state_fixtures.ts";

/** The START's inbox param `msgs: Inbox(str)`. */
export const MSGS = new Parameter("msgs", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** `node`'s TOOL call id in the fixtures' run. */
export const call = (node: string) => toolCallId(RUN_ID, node);

/** The take output type `{events: list(str), closed: bool}`, built in code (the type grammar has no inline record). */
export const TAKE_OUTPUT = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["events", typeFor("list(str)", new Map())],
    ["closed", Type.scalar(ValueKind.BOOLEAN)],
  ]),
  required: new Set(["events", "closed"]),
});

/** An exact ToolNode for `toolId`; `take` gets the take output contract. */
export function tool(id: string, toolId: string): ToolNode {
  const node = new ToolNode(id, { toolId });
  if (toolId === "take") node.outputContract = new ValueContract(TAKE_OUTPUT);
  return node;
}

/** A side branch: an exact CodeNode (never run; the reducer only sees records). */
export const work = () => new CodeNode("work", { code: "return 1" });

/** START(msgs: Inbox(str), ...params) -> each node -> END (END joins them all). */
export function waitFlow(...nodes: Node[]): CompiledFlow {
  return paramFlow([MSGS], ...nodes);
}

/** START(msgs: Inbox(str)) -> an AGENT `chat` reading steer inbox `msgs` -> END. */
export function steerFlow(): CompiledFlow {
  const chat = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  const wiring = new Map([["chat", new Map([[STEER_INBOX_KEY, new Binding("${input.msgs}")]])]]);
  return paramFlow([MSGS], { wiring }, chat);
}

/** START(...params) -> each node -> END (END joins them all), with `options.wiring` when given first. */
export function paramFlow(
  params: readonly Parameter[],
  ...rest: [{ wiring: ReadonlyMap<string, ReadonlyMap<string, unknown>> }, ...Node[]] | Node[]
): CompiledFlow {
  const options = rest[0] instanceof Node || rest.length === 0 ? {} : (rest.shift() as { wiring: ReadonlyMap<string, ReadonlyMap<string, unknown>> });
  const nodes = rest as Node[];
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params })],
      ...nodes.map((node) => [node.id, node] as const),
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      ...nodes.map((node) => new Edge(`start-${node.id}`, START_ID, node.id)),
      ...nodes.map((node) => new Edge(`${node.id}-end`, node.id, END_ID)),
    ],
    options,
  );
}

/**
 * Commit `nodeId`'s output, its messages when it produces them (an AGENT), and its success; a
 * learnable state records a literal capture.
 */
export function succeed(log: RunLog, state: RunState, nodeId: string, output: unknown): void {
  const node = state.flow.nodes.get(nodeId)!;
  commitPrepared(log, state, preparePoolCommit(state, nodeId, output));
  if (node.producesMessages) commitPrepared(log, state, prepareMessagesCommit(state, nodeId));
  commitPrepared(
    log,
    state,
    prepareNodeSucceeded(state, nodeId, { forwardCapture: state.learnable ? forwardCapture(node, {}) : null }),
  );
}

/**
 * A JSONL log and state of `flow` after run start and START's success (holding each inbox's
 * handle). `fingerprint` defaults to `flowFingerprint(flow)`; a flow holding a test double passes
 * `FINGERPRINT` and is never replayed through `replayRunLog`. A learnable run gets the execution
 * layout after fingerprinting, as `learnableStarted` does. `numWorkers` defaults to 0; an AGENT's
 * model config comes from `effectiveLlmByStaticPath` (none by default).
 */
export function started(
  flow: CompiledFlow,
  options: {
    learnable?: boolean;
    defaults?: Record<string, bigint>;
    fingerprint?: string;
    numWorkers?: number;
    effectiveLlmByStaticPath?: Record<string, JsonObject>;
  } = {},
) {
  const fingerprint = options.fingerprint ?? flowFingerprint(flow);
  if (options.learnable) stampExecutionScopes(flow);
  // Beside the test's HOME, in the per-test dir the preload removes once the run ends.
  const path = join(mkdtempSync(join(dirname(process.env.HOME!), "waits-")), "log.jsonl");
  const log = JsonlRunLog.create(path, RUN_ID, { clock: FIXED_CLOCK });
  const state = new RunState(flow);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: fingerprint,
      startWiringOverride: {},
      effectiveLlmByStaticPath: options.effectiveLlmByStaticPath ?? {},
      numWorkers: options.numWorkers ?? 0,
      learnable: options.learnable ?? false,
      defaults: options.defaults ?? {},
    }),
  );
  const initial = new VariablePool(START_ID);
  initial.set(START_ID, {});
  initial.addSystem("run_id", RUN_ID);
  commitInitialPool(log, state, initial);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const start = state.flow.nodes.get(START_ID)!;
  const handles = (start.params ?? []).filter((param) => param.contract?.type.kind === ValueKind.INBOX);
  succeed(log, state, START_ID, Object.fromEntries(handles.map((param) => [param.name, new InboxRef(param.name)])));
  return { path, log, state, composed: new ComposedFlow(flow) };
}
export { FINGERPRINT };

/** Reopen a closed scenario log for an offline append, as a host does. */
export const reopen = (path: string) => JsonlRunLog.resume(path, { clock: FIXED_CLOCK });

/** The trace a learnable replay must reproduce: `replayTraceSnapshot` plus the parked starts. */
export const traceParts = (trace: RetainedTrace) => ({ ...replayTraceSnapshot(trace), parked: [...trace._parked] });

/** A fresh state of `flow` replayed through the records at `path` (all, or the first `count`). */
export function replayed(flow: CompiledFlow, path: string, count?: number): RunState {
  const envelopes = readRunLog(path);
  return replayRecords(new RunState(copyGraph(flow)), envelopes.slice(0, count ?? envelopes.length), { growBuilder: noGrowth });
}

/** The deadline the scenarios' timed waits record. */
export const DEADLINE = "2026-10-21T00:00:00.000Z";

/** `node`'s take wait entry on `msgs`. */
export function takeEntry(node: string, deadline: string | null = null): WaitEntry {
  return { callId: call(node), tool: "take", inbox: "msgs", mode: "one", deadline };
}

/** `node`'s sleep wait entry until `DEADLINE`. */
export function sleepEntry(node: string): WaitEntry {
  return { callId: call(node), tool: "sleep", inbox: null, mode: null, deadline: DEADLINE };
}

/** One closed scenario log, its flow and the live state that wrote it; `learnable` logs are compared through `replayRecords` only. */
export interface ScenarioLog {
  readonly name: string;
  readonly path: string;
  readonly flow: CompiledFlow;
  readonly composed: ComposedFlow;
  readonly learnable: boolean;
  /** The state after the last record, as the writer left it. */
  readonly state: RunState;
  /** The seq of each named record. */
  readonly marks: Readonly<Record<string, number>>;
}

/** A scenario being written: a live state and its JSONL log, closed and reopened as a host would. */
class Scenario {
  readonly marks: Record<string, number> = {};
  readonly name: string;
  readonly flow: CompiledFlow;
  readonly learnable: boolean;
  readonly path: string;
  readonly composed: ComposedFlow;
  log: RunLog;
  state: RunState;

  constructor(
    name: string,
    flow: CompiledFlow,
    learnable: boolean,
    numWorkers = 0,
    effectiveLlmByStaticPath: Record<string, JsonObject> = {},
  ) {
    this.name = name;
    this.flow = flow;
    this.learnable = learnable;
    const run = started(flow, { learnable, numWorkers, effectiveLlmByStaticPath });
    this.path = run.path;
    this.composed = run.composed;
    this.log = run.log;
    this.state = run.state;
  }

  /** Commit `prepared`, marking its seq as `mark` when given. */
  commit(prepared: PreparedTransition, mark?: string): this {
    commitPrepared(this.log, this.state, prepared);
    if (mark !== undefined) this.marks[mark] = prepared.expectedSequence;
    return this;
  }

  /** The capture an outcome of `node` carries: a literal one while learning. */
  capture(node: string) {
    return this.learnable ? forwardCapture(this.state.flow.nodes.get(node)!, {}) : null;
  }

  start(node: string, mark?: string): this {
    return this.commit(prepareNodeStarted(this.state, node), mark);
  }
  wait(node: string, entry: WaitEntry | null, mark?: string): this {
    const waits = entry === null ? [] : [entry];
    return this.commit(prepareNodeWaiting(this.state, node, { callIds: [call(node)], waits }), mark);
  }
  runWaiting(mark?: string): this {
    return this.commit(prepareRunWaiting(this.state), mark);
  }
  send(event: unknown, mark?: string): this {
    return this.commit(prepareSend(this.state, "msgs", event), mark);
  }
  close(mark?: string): this {
    return this.commit(prepareInboxClosed(this.state, "msgs")!, mark);
  }
  withdraw(mark?: string): this {
    return this.commit(prepareWithdraw(this.state, "msgs")!, mark);
  }
  /** An offline interrupt of `msgs`. */
  interrupt(mark?: string): this {
    return this.commit(prepareInterrupt(this.state, "msgs")!, mark);
  }
  woken(node: string, timedOut: readonly string[] = [], mark?: string): this {
    return this.commit(prepareNodeWoken(this.state, node, timedOut), mark);
  }
  step(node: string, body: Record<string, unknown>, mark?: string): this {
    return this.commit(prepareNodeStep(this.state, node, body), mark);
  }
  take(node: string, take: TakeCommit, mark?: string): this {
    this.commit(prepareTakeCommit(this.state, node, take), mark);
    return this.commit(prepareNodeSucceeded(this.state, node, { forwardCapture: this.capture(node) }));
  }
  succeed(node: string, output: unknown, mark?: string): this {
    this.commit(preparePoolCommit(this.state, node, output), mark);
    if (this.state.flow.nodes.get(node)!.producesMessages) this.commit(prepareMessagesCommit(this.state, node));
    return this.commit(prepareNodeSucceeded(this.state, node, { forwardCapture: this.capture(node) }));
  }
  resume(mark?: string): this {
    return this.commit(prepareRunResumed(this.state, { numWorkers: this.state.numWorkers }), mark);
  }
  /** Start END and commit its output; the run succeeds. */
  end(): this {
    return this.start(END_ID).succeed(END_ID, "done", "end");
  }

  /** Close the log, as a process that ends (or crashes) does, then reopen it with a replayed state. */
  reopen(): this {
    this.log.close();
    this.state = replayed(this.flow, this.path);
    this.log = reopen(this.path);
    return this;
  }

  /** Close the log and return the scenario. */
  done(): ScenarioLog {
    this.log.close();
    const { name, path, flow, composed, learnable, state, marks } = this;
    return { name, path, flow, composed, learnable, state, marks };
  }
}

/** The send seqs a scenario holds, by mark, for take commits. */
const seqOf = (scenario: Scenario, mark: string) => scenario.marks[mark]!;

/** An `ask` pending step for `node`'s call. */
export const askPending = (node: string) => ({ kind: "pending", call_id: call(node), tool: "ask", arguments: { prompt: "ok?" } });

/** A `tool_result` step answering `node`'s call with `answer`. */
export const askAnswer = (node: string, answer: string) => ({
  kind: "tool_result",
  call_id: call(node),
  output: { kind: "str", value: answer },
});

/**
 * Write every wait scenario log, plain and learnable, and return them:
 *
 * - `a`: a take waits, the run waits, a send arrives offline, the take wakes, takes it, and the
 *   run ends; `a-inline`: the send comes first, so the take commits inline.
 * - `b`: as `a` to the wake, then a crash; a send while down; the resumed take commits its
 *   assigned event, not the new one. `b-timeout`: a take woken by its timeout commits nothing
 *   after the crash, though an event is unread; `b-sleep`: a sleep woken by its deadline commits
 *   `null` after the crash.
 * - `c`: two `ask` nodes wait on host calls, answered b then a; `c-crash`: a crash after the
 *   pending step, and the restart waits on the same call.
 * - `d`: a sleep waits while another branch runs, then times out.
 * - `e`: the inbox closes while a take waits with the host down; the take commits closed.
 * - `f`: an AGENT reads two sends as one steer step, a third send arrives, then a crash; the
 *   restarted agent succeeds and the third send stays unread.
 * - `g`: a send to a waiting take is withdrawn, so the take waits again and takes a later send.
 * - `h`: an empty Stop is recorded while a node that reads no inbox works; the take that then
 *   waits ends it, and takes a later send.
 */
export function scenarioLogs(): ScenarioLog[] {
  return [false, true].flatMap((learnable) => [
    scenarioA(learnable),
    scenarioAInline(learnable),
    scenarioB(learnable),
    scenarioBTimeout(learnable),
    scenarioBSleep(learnable),
    scenarioC(learnable),
    scenarioCCrash(learnable),
    scenarioD(learnable),
    scenarioE(learnable),
    scenarioF(learnable),
    scenarioG(learnable),
    scenarioH(learnable),
  ]);
}

function scenarioA(learnable: boolean): ScenarioLog {
  const run = new Scenario("a", waitFlow(tool("heard", "take")), learnable);
  run.start("heard").wait("heard", takeEntry("heard"), "waiting").runWaiting("runWaiting");
  run.reopen().send("hi", "send");
  run.reopen().woken("heard", [], "woken").take("heard", { inbox: "msgs", events: [seqOf(run, "send")], closed: false }, "take");
  return run.end().done();
}

function scenarioAInline(learnable: boolean): ScenarioLog {
  const run = new Scenario("a-inline", waitFlow(tool("heard", "take")), learnable);
  run.send("hi", "send").start("heard");
  run.take("heard", { inbox: "msgs", events: [seqOf(run, "send")], closed: false }, "take");
  return run.end().done();
}

function scenarioB(learnable: boolean): ScenarioLog {
  const run = new Scenario("b", waitFlow(tool("heard", "take")), learnable);
  run.start("heard").wait("heard", takeEntry("heard")).runWaiting();
  run.reopen().send("hi", "send");
  run.reopen().woken("heard", [], "woken");
  run.reopen().send("again", "downtime");
  run.resume("resumed").start("heard", "restarted");
  run.take("heard", { inbox: "msgs", events: [seqOf(run, "send")], closed: false }, "take");
  return run.end().done();
}

function scenarioBTimeout(learnable: boolean): ScenarioLog {
  const run = new Scenario("b-timeout", waitFlow(tool("heard", "take")), learnable);
  run.start("heard").wait("heard", takeEntry("heard", DEADLINE)).runWaiting();
  run.reopen().woken("heard", [call("heard")], "woken");
  run.reopen().send("late", "downtime");
  run.resume().start("heard", "restarted").take("heard", { inbox: "msgs", events: [], closed: false }, "take");
  return run.end().done();
}

function scenarioBSleep(learnable: boolean): ScenarioLog {
  const run = new Scenario("b-sleep", waitFlow(tool("nap", "sleep")), learnable);
  run.start("nap").wait("nap", sleepEntry("nap")).runWaiting();
  run.reopen().woken("nap", [call("nap")], "woken");
  run.reopen().resume().start("nap", "restarted").succeed("nap", null, "slept");
  return run.end().done();
}

function scenarioC(learnable: boolean): ScenarioLog {
  const run = new Scenario("c", waitFlow(tool("a", "ask"), tool("b", "ask")), learnable, 2);
  run.start("a").start("b").step("a", askPending("a")).step("b", askPending("b"));
  run.wait("a", null).wait("b", null).runWaiting("runWaiting");
  run.reopen().step("b", askAnswer("b", "no")).step("a", askAnswer("a", "yes"), "answered");
  run.reopen().woken("a").succeed("a", "yes").woken("b").succeed("b", "no");
  return run.end().done();
}

function scenarioCCrash(learnable: boolean): ScenarioLog {
  const run = new Scenario("c-crash", waitFlow(tool("a", "ask")), learnable);
  run.start("a").step("a", askPending("a"), "pending");
  run.reopen().resume().start("a", "restarted").wait("a", null, "rewaited").runWaiting();
  run.reopen().step("a", askAnswer("a", "yes")).woken("a").succeed("a", "yes");
  return run.end().done();
}

function scenarioD(learnable: boolean): ScenarioLog {
  const run = new Scenario("d", waitFlow(tool("nap", "sleep"), work()), learnable);
  run.start("nap").start("work").wait("nap", sleepEntry("nap"), "napping").succeed("work", 1n, "worked").runWaiting("runWaiting");
  run.reopen().woken("nap", [call("nap")], "woken").succeed("nap", null);
  return run.end().done();
}

function scenarioE(learnable: boolean): ScenarioLog {
  const run = new Scenario("e", waitFlow(tool("heard", "take")), learnable);
  run.start("heard").wait("heard", takeEntry("heard")).runWaiting();
  run.reopen().close("closed");
  run.reopen().woken("heard").take("heard", { inbox: "msgs", events: [], closed: true }, "take");
  return run.end().done();
}

function scenarioF(learnable: boolean): ScenarioLog {
  const run = new Scenario("f", steerFlow(), learnable, 0, { chat: { provider: "openai", model: "gpt-5.5" } });
  run.send("a", "sa").send("b", "sb").start("chat");
  const from = [BigInt(seqOf(run, "sa")), BigInt(seqOf(run, "sb"))];
  run.step("chat", { kind: "steer", inbox: "msgs", from }, "steer").send("c", "sc");
  run.reopen().resume("resumed").start("chat", "restarted").succeed("chat", "ok");
  return run.end().done();
}

function scenarioG(learnable: boolean): ScenarioLog {
  const run = new Scenario("g", waitFlow(tool("heard", "take")), learnable);
  run.start("heard").wait("heard", takeEntry("heard"), "waiting").runWaiting();
  run.reopen().send("x", "sx").withdraw("withdrawn");
  run.reopen().send("y", "sy");
  run.reopen().woken("heard").take("heard", { inbox: "msgs", events: [seqOf(run, "sy")], closed: false }, "take");
  return run.end().done();
}

function scenarioH(learnable: boolean): ScenarioLog {
  const run = new Scenario("h", waitFlow(work(), tool("heard", "take")), learnable);
  run.start("work").interrupt("stop").succeed("work", 1n).start("heard").wait("heard", takeEntry("heard"), "waiting").runWaiting();
  run.reopen().send("x", "sx");
  run.reopen().woken("heard").take("heard", { inbox: "msgs", events: [seqOf(run, "sx")], closed: false }, "take");
  return run.end().done();
}
