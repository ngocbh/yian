/**
 * TOOL `retries:` across crashes, resume and replay: a port of
 * `tests/engine/test_tool_retries_resume.py`.
 *
 * TS has no `_sleep` or `_jitter` to patch: each run and resume gets a virtual scheduler that
 * records every wait, in milliseconds where Python records seconds, and a draw of 1. A crash is
 * `crashAfter`'s log refusing an append, so a crash "inside attempt k" is the refused outcome of
 * that attempt (the tool itself returns), and a crash "during the backoff wait" is the refused
 * `started` step after the wait. JS has no `TimeoutError`, so the rows throw their own. Every run
 * sees only the test's credential store in its environment.
 *
 * Ported through `runFlow` and `resumeFlow`:
 * `test_a_crash_inside_attempt_two_resumes_at_attempt_three_with_one_retry_used`,
 * `test_a_crash_inside_attempt_two_with_no_retry_left_fails_on_resume`,
 * `test_a_retries_zero_crash_resumes_at_attempt_two_without_steps`,
 * `test_a_crash_during_the_backoff_wait_rewaits_and_resumes_at_attempt_two`,
 * `test_a_crash_during_the_second_backoff_rewaits_the_last_recorded_delay`,
 * `test_two_crashes_rewait_the_recorded_delay_each_resume`,
 * `test_a_crash_mid_retry_resumes_across_worker_modes` (both rows),
 * `test_an_interrupted_first_attempt_shows_in_the_final_message`,
 * `test_a_learnable_run_with_a_retried_tool_replays_and_runs_backward` and
 * `test_two_tools_retrying_at_once_in_a_pooled_map_resume_with_their_own_steps` (the crash refuses
 * each child TOOL's outcome once both have recorded their retry's start; a promise barrier holds
 * both attempts together, Python's `threading.Barrier`).
 *
 * Ported (adapted) through `FlowEngine` over a capturing memory log, since TS can't patch
 * `commit_prepared`: the log snapshots the state's node starts, node steps and in-flight ids at
 * every append, which is the state after the commit before it (the last one taken when `run()`
 * settles). Each prefix replays through `replayRunLog`. The live half of
 * `test_a_finished_log_with_retries_replays_to_the_live_steps`, and
 * `test_a_long_loop_keeps_no_steps_for_finished_iterations` (50 iterations, Python's flow).
 *
 * The two backoff rows also have the cases
 * `resume/tool_retries_resume__a_crash_in_the_backoff_resumes_at_attempt_two` and
 * `resume/tool_retries_resume__a_crash_during_the_second_backoff_rewaits_the_last_delay`, which pin
 * the trace and the tool calls but not the waits.
 *
 * Covered by `resume/tool_retries_resume__a_finished_run_with_retries_resumes_read_only`: the resume half of
 * `test_a_finished_log_with_retries_replays_to_the_live_steps` (two read-only resumes, the log
 * unchanged, the tool not called again).
 *
 * Changed: `test_an_old_tool_crash_log_resumes_at_attempt_two` and
 * `test_an_old_loop_tool_crash_log_resumes_with_its_growth` resume Python's committed schema-9
 * logs; TS reads only schema 10, so each resume rejects with a `RangeError` naming Python yian,
 * calls no tool and leaves the log unchanged.
 *
 * Changed (vocabulary): a TS resume also publishes `NodeAttemptResumed` for each retried node; no
 * row here reads the events.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildGrow } from "../../src/compile/expand.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";
import { replayRunLog, resumeFlow, runFlow } from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import {
  BackwardFinishedRecord,
  JsonlRunLog,
  MemoryRunLog,
  NodeStepRecord,
  NodeSucceededRecord,
  readRunLog,
  type RecordEnvelope,
  type RunRecord,
} from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { TransientToolError } from "../../src/tools/errors.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { barrier, Crash, crashAfter, rejection, startedRun } from "./run_state_fixtures.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets: the
// engine rows and the replays read the process's values.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const dir = mkdtempSync(join(tmpdir(), "yian-tool-retries-resume-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const PY_FIXTURES = fileURLToPath(new URL("../../../tests/fixtures", import.meta.url));

/** JS has no built-in `TimeoutError`; the rows throw this one, as Python's throw its own. */
class TimeoutError extends Error {
  override name = "TimeoutError";
}

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** Load a one-TOOL flow calling `x` with `retries:`. */
const flowOf = (retries: number) =>
  loadFlow(`id: t\nname: t\nnodes:\n  call:\n    kind: tool\n    tool_id: x\n    retries: ${retries}\noutput: \${call.output}\n`);

/** What a tool's context showed for one call. */
interface Seen {
  attempt: number;
  callId: string;
  node: string;
}

const seenOf = (ctx: ToolContext): Seen => ({ attempt: ctx.attempt, callId: ctx.callId, node: ctx.node });

/** A tool `x` throwing the scripted errors in order, then returning `"ok"`; `seen` records each call. */
class Script {
  readonly seen: Seen[] = [];
  outcomes: Error[];
  readonly tool: FunctionTool;

  constructor(...outcomes: Error[]) {
    this.outcomes = outcomes;
    this.tool = tool({
      name: "x",
      description: "Do one scripted thing.",
      parameters: { type: "object", properties: {} },
      execute: (_args, ctx) => {
        this.seen.push(seenOf(ctx));
        const next = this.outcomes.shift();
        if (next !== undefined) throw next;
        return "ok";
      },
    });
  }

  /** The run's tools, the scripted tool as `x`. */
  tools(): Record<string, FunctionTool> {
    return { x: this.tool };
  }

  attempts(): number[] {
    return this.seen.map((call) => call.attempt);
  }

  callIds(): Set<string> {
    return new Set(this.seen.map((call) => call.callId));
  }
}

/** A virtual scheduler recording each wait, in milliseconds, across a run and its resumes. */
function clock(): { scheduler: Scheduler; waited: number[] } {
  const inner = virtualScheduler();
  const waited: number[] = [];
  return {
    waited,
    scheduler: {
      now: () => inner.now(),
      sleep(ms, signal) {
        waited.push(ms);
        return inner.sleep(ms, signal);
      },
    },
  };
}

/** The run options every row shares: the test's environment, the clock and a draw of 1. */
const host = (scheduler: Scheduler) => ({ env: hostEnv(dir), scheduler, random: () => 1 });

/** The node steps a durable log recorded, in order. */
const steps = (path: string) =>
  readRunLog(path)
    .map((envelope) => envelope.record)
    .filter((record) => record instanceof NodeStepRecord)
    .map((record) => (record as NodeStepRecord).step);

/** Each recorded node step with its node id, grouped by node id. */
function stepsByNode(path: string): Map<string, unknown[]> {
  const byNode = new Map<string, unknown[]>();
  for (const { record } of readRunLog(path)) {
    if (!(record instanceof NodeStepRecord)) continue;
    byNode.set(record.nodeId, [...(byNode.get(record.nodeId) ?? []), record.step]);
  }
  return byNode;
}

/** The recorded step of a failed attempt that threw `type`. */
const failed = (attempt: number, message: string, retryIn: number, type = "TimeoutError") => ({
  event: "failed",
  attempt: BigInt(attempt),
  error: { type, message },
  retry_in: retryIn,
});

/** The recorded step of a retried attempt's start. */
const started = (attempt: number) => ({ event: "started", attempt: BigInt(attempt) });

test("a crash inside attempt two resumes at attempt three with one retry used", async () => {
  const path = freshLog();
  const flow = await flowOf(3);
  const script = new Script(new TimeoutError("a"));
  const { scheduler, waited } = clock();

  // Record 6 is attempt 2's outcome.
  await crashAfter(6, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );

  expect(script.attempts()).toEqual([1, 2]);
  expect(steps(path)).toEqual([failed(1, "a", 1), started(2)]);

  // Failures used = 1, so two retries remain: attempts 3 and 4 fail, attempt 5 succeeds.
  script.outcomes = [new TimeoutError("c"), new TimeoutError("d")];
  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(script.attempts()).toEqual([1, 2, 3, 4, 5]);
  expect(script.callIds()).toEqual(new Set(["r1/call"]));
  expect(waited).toEqual([1000, 2000, 4000]);
  expect(steps(path)).toEqual([
    failed(1, "a", 1),
    started(2),
    failed(3, "c", 2),
    started(4),
    failed(4, "d", 4),
    started(5),
  ]);
});

test("a crash inside attempt two with no retry left fails on resume", async () => {
  const path = freshLog();
  const flow = await flowOf(1);
  const script = new Script(new TimeoutError("a"));
  const { scheduler } = clock();

  await crashAfter(6, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );

  script.outcomes = [new TimeoutError("c")];
  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.message).toBe(
    "tool 'x' failed after 3 attempts: attempt 1: TimeoutError: a; attempt 2: interrupted; attempt 3: TimeoutError: c",
  );
});

test("a retries zero crash resumes at attempt two without steps", async () => {
  const path = freshLog();
  const flow = await flowOf(0);
  const script = new Script();
  const { scheduler, waited } = clock();

  // Record 4 is attempt 1's outcome.
  await crashAfter(4, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(script.attempts()).toEqual([1, 2]);
  expect(script.callIds()).toEqual(new Set(["r1/call"]));
  expect(steps(path)).toEqual([]);
  expect(waited).toEqual([]);
});

test("a crash during the backoff wait rewaits and resumes at attempt two", async () => {
  const path = freshLog();
  const flow = await flowOf(3);
  // `retryAfter` makes the recorded delay 7, where a recomputed backoff would be 1.
  const script = new Script(new TransientToolError("a", { retryAfter: 7 }));
  const { scheduler, waited } = clock();

  // Record 5 is attempt 2's `started` step, after the wait.
  await crashAfter(5, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  expect(steps(path)).toEqual([failed(1, "a", 7, "TransientToolError")]);

  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(script.attempts()).toEqual([1, 2]);
  expect(script.callIds()).toEqual(new Set(["r1/call"]));
  expect(waited).toEqual([7000, 7000]);
  // The resumed attempt 2 is the node's own start, so it records no `started` step.
  expect(steps(path)).toEqual([failed(1, "a", 7, "TransientToolError")]);
});

test("a crash during the second backoff rewaits the last recorded delay", async () => {
  const path = freshLog();
  const flow = await flowOf(3);
  // Recorded delays 7 then 5; a recomputed second backoff would be 2.
  const script = new Script(
    new TransientToolError("a", { retryAfter: 7 }),
    new TransientToolError("b", { retryAfter: 5 }),
  );
  const { scheduler, waited } = clock();

  // Record 7 is attempt 3's `started` step, after the second wait.
  await crashAfter(7, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(script.attempts()).toEqual([1, 2, 3]);
  expect(script.callIds()).toEqual(new Set(["r1/call"]));
  expect(waited).toEqual([7000, 5000, 5000]);
  expect(steps(path)).toEqual([
    failed(1, "a", 7, "TransientToolError"),
    started(2),
    failed(2, "b", 5, "TransientToolError"),
  ]);
});

test("two crashes rewait the recorded delay each resume", async () => {
  const path = freshLog();
  const flow = await flowOf(3);
  const script = new Script(new TransientToolError("a", { retryAfter: 7 }));
  const { scheduler, waited } = clock();

  // Record 5 is attempt 2's `started` step, after the wait.
  await crashAfter(5, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  // Records 5 and 6 are `RunResumed` and the resumed start; record 7 is attempt 2's outcome.
  await crashAfter(7, () => resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) }));

  expect(script.attempts()).toEqual([1, 2]);
  expect(waited).toEqual([7000, 7000]);

  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(script.attempts()).toEqual([1, 2, 3]);
  expect(script.callIds()).toEqual(new Set(["r1/call"]));
  // The first wait, then the recorded delay again on each of the two resumes.
  expect(waited).toEqual([7000, 7000, 7000]);
  expect(steps(path)).toEqual([failed(1, "a", 7, "TransientToolError")]);
});

for (const [name, first, then] of [
  ["pooled then serial", 8, 0],
  ["serial then pooled", 0, 8],
] as const) {
  test(`a crash mid retry resumes across worker modes: ${name}`, async () => {
    const path = freshLog();
    const flow = await flowOf(3);
    const script = new Script(new TimeoutError("a"));
    const { scheduler, waited } = clock();

    await crashAfter(6, () =>
      runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: first, ...host(scheduler) }),
    );
    script.outcomes = [new TimeoutError("c")];
    const result = await resumeFlow(flow, { log: path, tools: script.tools(), numWorkers: then, ...host(scheduler) });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(script.attempts()).toEqual([1, 2, 3, 4]);
    expect(script.callIds()).toEqual(new Set(["r1/call"]));
    expect(waited).toEqual([1000, 2000]);
    expect(steps(path)).toEqual([failed(1, "a", 1), started(2), failed(3, "c", 2), started(4)]);
  });
}

test("an interrupted first attempt shows in the final message", async () => {
  const path = freshLog();
  const flow = await flowOf(1);
  const script = new Script();
  const { scheduler, waited } = clock();

  await crashAfter(4, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  script.outcomes = [new TimeoutError("b"), new TimeoutError("c")];
  const result = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.causeType).toBe("ToolAttemptsFailed");
  expect(result.failure!.message).toBe(
    "tool 'x' failed after 3 attempts: attempt 1: interrupted; attempt 2: TimeoutError: b; attempt 3: TimeoutError: c",
  );
  expect(script.attempts()).toEqual([1, 2, 3]);
  expect(waited).toEqual([1000]);
});

test("a learnable run with a retried tool replays and runs backward", async () => {
  const path = freshLog();
  const flow = await flowOf(2);
  const script = new Script(new TimeoutError("a"));
  const { scheduler } = clock();

  const result = await runFlow(flow, {}, {
    log: path,
    runId: "r1",
    tools: script.tools(),
    learnable: true,
    ...host(scheduler),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(steps(path)).toEqual([failed(1, "a", 1), started(2)]);

  const resumed = await resumeFlow(flow, { log: path, tools: script.tools(), ...host(scheduler) });

  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("ok");
  expect(script.seen).toHaveLength(2);
  const outcome = await resumed.backward("Review.");
  expect(outcome.status).toBe("succeeded");
  expect(readRunLog(path).at(-1)!.record).toBeInstanceOf(BackwardFinishedRecord);
});

const MAP_FLOW = `
id: map_tools
name: map_tools
defs:
  one:
    input:
      n: !int
    nodes:
      place:
        kind: tool
        tool_id: x
        retries: 3
        input: {n: "\${input.n}"}
    output: \${place.output}
nodes:
  each:
    kind: map
    call: one
    over: \${[0, 1]}
    input:
      n: \${item}
output: \${each.output}
`;

test("two tools retrying at once in a pooled map resume with their own steps", async () => {
  const path = freshLog();
  const flow = await loadFlow(MAP_FLOW);
  const seen: [number, Seen][] = [];
  const firstTogether = barrier(2);
  const secondTogether = barrier(2);
  let crashed = false;
  const x = tool({
    name: "x",
    description: "Fail attempt 1, crash in attempt 2 once, then fail once more before succeeding.",
    parameters: { type: "object", properties: { n: { type: "integer" } }, required: ["n"] },
    execute: async (args, ctx) => {
      const n = Number(args.n);
      seen.push([n, seenOf(ctx)]);
      if (ctx.attempt === 1) {
        await firstTogether();
        throw new TimeoutError(`a${n}`);
      }
      // Both nodes have recorded their failure and the retry's start; the outcome is refused.
      if (ctx.attempt === 2 && !crashed) await secondTogether();
      if (ctx.attempt === 3) throw new TimeoutError(`c${n}`);
      return `ok${n}`;
    },
  });
  const { scheduler } = clock();

  // The crash: each child TOOL's outcome is refused, so both stay in flight.
  const append = JsonlRunLog.prototype.append;
  const crash = new Crash("process lost");
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (record instanceof NodeSucceededRecord && record.nodeId.endsWith("/place")) throw crash;
    return append.call(this, record);
  };
  try {
    const error = await rejection(
      runFlow(flow, {}, { log: path, runId: "r1", tools: { x }, numWorkers: 4, ...host(scheduler) }),
    );
    expect(error).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }

  crashed = true;
  const ids = new Map(seen.filter(([, ctx]) => ctx.attempt === 1).map(([n, ctx]) => [n, ctx.node]));
  expect(new Set(ids.values()).size).toBe(2);
  let byNode = stepsByNode(path);
  for (const [n, nodeId] of ids) expect(byNode.get(nodeId)).toEqual([failed(1, `a${n}`, 1), started(2)]);

  seen.length = 0;
  const result = await resumeFlow(flow, { log: path, tools: { x }, numWorkers: 4, ...host(scheduler) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual(["ok0", "ok1"]);
  for (const [n, nodeId] of ids) {
    const mine = seen.filter(([m]) => m === n).map(([, ctx]) => ctx);
    expect(mine.map((ctx) => ctx.attempt)).toEqual([3, 4]);
    expect(new Set(mine.map((ctx) => ctx.callId))).toEqual(new Set([`r1/${nodeId}`]));
    expect(new Set(mine.map((ctx) => ctx.node))).toEqual(new Set([nodeId]));
  }
  byNode = stepsByNode(path);
  for (const [n, nodeId] of ids) {
    expect(byNode.get(nodeId)).toEqual([failed(1, `a${n}`, 1), started(2), failed(3, `c${n}`, 2), started(4)]);
  }
});

// --- live state against replay ---------------------------------------------------------- //

/** The state's node starts, node steps and in-flight ids at one point. */
interface Live {
  starts: Map<string, number>;
  steps: Map<string, readonly unknown[]>;
  inFlight: string[];
}

const liveOf = (state: RunState): Live => ({
  starts: new Map(state.nodeStarts),
  steps: new Map([...state.nodeSteps].map(([id, recorded]) => [id, [...recorded]])),
  inFlight: [...state.inFlight],
});

/** A memory log that snapshots `state` before each append, once `state` is set. */
class CapturingLog extends MemoryRunLog {
  state: RunState | null = null;
  readonly before: Live[] = [];

  override append(record: RunRecord): RecordEnvelope {
    if (this.state !== null) this.before.push(liveOf(this.state));
    return super.append(record);
  }
}

/**
 * Run `loaded` serially through `FlowEngine` over a capturing log with `tools`; return the log's
 * records and the live state after each commit past `RunStarted` (Python's `_capture_live`).
 */
async function liveRun(
  loaded: ComposedFlow,
  tools: Record<string, FunctionTool>,
): Promise<{ records: readonly RecordEnvelope[]; live: Live[]; state: RunState }> {
  const log = new CapturingLog("r1");
  const state = startedRun(loaded, log);
  log.state = state;
  const { scheduler } = clock();
  const engine = new FlowEngine(state, log, {
    growBuilder: buildGrow,
    numWorkers: 0,
    tools: new ToolSet(tools),
    scheduler,
    random: () => 1,
  });
  await engine.run();
  // The snapshot before record k is the state after record k - 1; the first is after RunStarted.
  return { records: log.records(), live: [...log.before.slice(1), liveOf(state)], state };
}

/** The replayed state of the first `length` records. */
const replayPrefix = (loaded: ComposedFlow, records: readonly RecordEnvelope[], length: number) =>
  replayRunLog(loaded, MemoryRunLog.fromRecords(records.slice(0, length)));

test("a finished log with retries replays to the live steps", async () => {
  const flow = await flowOf(2);
  const script = new Script(new TimeoutError("a"), new TimeoutError("b"));

  const { records, live, state } = await liveRun(flow, script.tools());

  expect(state.result?.constructor.name).toBe("RunSucceeded");
  // RunStarted is written before the engine runs; every later record is one live commit.
  expect(live).toHaveLength(records.length - 1);
  // Every prefix replays to the live node starts and steps after that commit.
  for (const [i, expected] of live.entries()) {
    const length = i + 2;
    expect(liveOf(replayPrefix(flow, records, length)), `prefix ${length}`).toEqual(expected);
  }
  expect(live.some(({ steps }) => (steps.get("call")?.length ?? 0) > 0)).toBe(true);
  expect(live.at(-1)!.starts).toEqual(new Map());
  expect(live.at(-1)!.steps).toEqual(new Map());
  expect(script.seen).toHaveLength(3);
});

const LOOP_FLOW = `
id: loop_tools
name: loop_tools
defs:
  turn:
    input:
      count: !int
    nodes:
      place:
        kind: tool
        tool_id: x
        retries: 1
        input: {n: "\${input.count}"}
    output:
      count: \${input.count + 1}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
    while: count < 50
    max: 60
output: \${loop.output}
`;

test("a long loop keeps no steps for finished iterations", async () => {
  const flow = await loadFlow(LOOP_FLOW);
  const calls: [number, number][] = [];
  const x = tool({
    name: "x",
    description: "Fail each iteration's first attempt.",
    parameters: { type: "object", properties: { n: { type: "integer" } }, required: ["n"] },
    execute: (args, ctx) => {
      calls.push([Number(args.n), ctx.attempt]);
      if (ctx.attempt === 1) throw new TimeoutError(`a${args.n}`);
      return "ok";
    },
  });

  const { records, live, state } = await liveRun(flow, { x });

  expect(state.result?.constructor.name).toBe("RunSucceeded");
  expect(state.result).toHaveProperty("output", { count: 50n });
  expect(calls).toEqual(Array.from({ length: 50 }, (_, n) => [[n, 1], [n, 2]] as [number, number][]).flat());
  expect(records.filter(({ record }) => record instanceof NodeStepRecord)).toHaveLength(100);
  // Live: only in-flight nodes hold a start count or steps, so a finished iteration leaves
  // nothing behind.
  for (const { starts, steps, inFlight } of live) {
    expect([...starts.keys()].every((id) => inFlight.includes(id))).toBe(true);
    expect([...steps.keys()].every((id) => inFlight.includes(id))).toBe(true);
    expect([...steps.values()].every((recorded) => recorded.length <= 2)).toBe(true);
  }
  expect(live.at(-1)!.starts).toEqual(new Map());
  expect(live.at(-1)!.steps).toEqual(new Map());

  const replayed = replayPrefix(flow, records, records.length);
  expect(replayed.nodeStarts).toEqual(new Map());
  expect(replayed.nodeSteps).toEqual(new Map());
  // Replay of a prefix ending inside an iteration holds that iteration's steps only.
  const stepPrefixes = records.flatMap(({ record }, i) => (record instanceof NodeStepRecord ? [i + 1] : []));
  for (const length of stepPrefixes.filter((_, i) => i % 10 === 0)) {
    const { starts, steps } = live[length - 2]!;
    const state = replayPrefix(flow, records, length);
    expect(state.nodeStarts).toEqual(starts);
    expect(state.nodeSteps).toEqual(steps as Map<string, readonly never[]>);
    expect(state.nodeSteps.size).toBe(1);
  }
});

// --- Python's old logs ------------------------------------------------------------------ //

const OLD_TOOL_FLOW = `id: old_tool_crash
name: old_tool_crash
nodes:
  place:
    kind: tool
    tool_id: place
    input: {symbol: "ACME"}
output: \${place.output}
`;

const OLD_LOOP_FLOW = `id: old_loop_tool_crash
name: old_loop_tool_crash
defs:
  turn:
    input:
      count: !int
    nodes:
      place:
        kind: tool
        tool_id: place
        input: {n: "\${input.count}"}
    output:
      count: \${input.count + 1}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
    while: count < 3
    max: 5
output: \${loop.output}
`;

for (const [name, fixture, source] of [
  ["an old tool crash log", "old_tool_crash.jsonl", OLD_TOOL_FLOW],
  ["an old loop tool crash log", "old_loop_tool_crash.jsonl", OLD_LOOP_FLOW],
] as const) {
  test(`${name} is Python's to resume`, async () => {
    const path = freshLog();
    copyFileSync(join(PY_FIXTURES, fixture), path);
    const before = readFileSync(path);
    const calls: unknown[] = [];
    const place = tool({
      name: "place",
      description: "Place an order.",
      parameters: { type: "object", properties: {} },
      execute: (args) => {
        calls.push(args);
        return "placed";
      },
    });
    const { scheduler } = clock();

    const error = await rejection(resumeFlow(await loadFlow(source), { log: path, tools: { place }, ...host(scheduler) }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(
      `run log ${path} line 1 (seq 0): run log schema 9 was written by Python yian; resume it with Python yian`,
    );
    expect(calls).toEqual([]);
    expect(readFileSync(path).equals(before)).toBe(true);
  });
}
