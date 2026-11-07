/**
 * TOOL `retries:` across crashes, resume and replay. Each run gets a virtual scheduler that records
 * every wait in milliseconds and draws 1; a crash is `crashAfter`'s log refusing an append, so a
 * crash "inside attempt k" is that attempt's refused outcome (the tool itself returns).
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
import { replayRunLog, advance, runFlow } from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import {
  MemoryRunLog,
  NodeStepRecord,
  NodeSucceededRecord,
  readRunLog,
  type RecordEnvelope,
  type RunLog,
  type RunRecord,
} from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { TransientToolError } from "../../src/tools/errors.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { barrier, Crash, crashAfter, rejected, startedRun } from "../helpers/run_state_fixtures.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets: the
// engine rows and the replays read the process's values.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const dir = mkdtempSync(join(tmpdir(), "yian-tool-retries-resume-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const SCHEMA9_FIXTURES = fileURLToPath(new URL("../fixtures/schema9", import.meta.url));

/** JS has no built-in `TimeoutError`, so the rows throw this one. */
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
  kind: "tool_attempt",
  event: "failed",
  attempt: BigInt(attempt),
  error: { type, message },
  retry_in: retryIn,
});

/** The recorded step of a retried attempt's start. */
const started = (attempt: number) => ({ kind: "tool_attempt", event: "started", attempt: BigInt(attempt) });

test("a crash inside attempt two resumes at attempt three with one retry used", async () => {
  const path = freshLog();
  const flow = await flowOf(3);
  const script = new Script(new TimeoutError("a"));
  const { scheduler, waited } = clock();

  // Record 11 is attempt 2's outcome (its commit), after the run start's five records and START's three.
  await crashAfter(11, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );

  expect(script.attempts()).toEqual([1, 2]);
  expect(steps(path)).toEqual([failed(1, "a", 1), started(2)]);

  // Failures used = 1, so two retries remain: attempts 3 and 4 fail, attempt 5 succeeds.
  script.outcomes = [new TimeoutError("c"), new TimeoutError("d")];
  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

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

  await crashAfter(11, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );

  script.outcomes = [new TimeoutError("c")];
  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.message).toBe(
    "tool 'x' failed after 3 attempts: attempt 1: TimeoutError: a; attempt 2: unfinished; attempt 3: TimeoutError: c",
  );
});

test("a retries zero crash resumes at attempt two without steps", async () => {
  const path = freshLog();
  const flow = await flowOf(0);
  const script = new Script();
  const { scheduler, waited } = clock();

  // Record 9 is attempt 1's outcome (its commit), after the run start's five records and START's three.
  await crashAfter(9, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

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

  // Record 10 is attempt 2's `started` step, after the wait.
  await crashAfter(10, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  expect(steps(path)).toEqual([failed(1, "a", 7, "TransientToolError")]);

  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

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

  // Record 12 is attempt 3's `started` step, after the second wait.
  await crashAfter(12, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

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

  // Record 10 is attempt 2's `started` step, after the wait.
  await crashAfter(10, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  // Records 10 and 11 are `RunResumed` and the resumed start; record 12 is attempt 2's outcome.
  await crashAfter(12, () => advance({ log: path, tools: script.tools(), ...host(scheduler) }));

  expect(script.attempts()).toEqual([1, 2]);
  expect(waited).toEqual([7000, 7000]);

  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

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

    await crashAfter(11, () =>
      runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: first, ...host(scheduler) }),
    );
    script.outcomes = [new TimeoutError("c")];
    const result = await advance({ log: path, tools: script.tools(), numWorkers: then, ...host(scheduler) });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(script.attempts()).toEqual([1, 2, 3, 4]);
    expect(script.callIds()).toEqual(new Set(["r1/call"]));
    expect(waited).toEqual([1000, 2000]);
    expect(steps(path)).toEqual([failed(1, "a", 1), started(2), failed(3, "c", 2), started(4)]);
  });
}

test("an unfinished first attempt shows in the final message", async () => {
  const path = freshLog();
  const flow = await flowOf(1);
  const script = new Script();
  const { scheduler, waited } = clock();

  await crashAfter(9, () =>
    runFlow(flow, {}, { log: path, runId: "r1", tools: script.tools(), numWorkers: 0, ...host(scheduler) }),
  );
  script.outcomes = [new TimeoutError("b"), new TimeoutError("c")];
  const result = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.causeType).toBe("ToolAttemptsFailed");
  expect(result.failure!.message).toBe(
    "tool 'x' failed after 3 attempts: attempt 1: unfinished; attempt 2: TimeoutError: b; attempt 3: TimeoutError: c",
  );
  expect(script.attempts()).toEqual([1, 2, 3]);
  expect(waited).toEqual([1000]);
});

test("a run with a retried tool replays", async () => {
  const path = freshLog();
  const flow = await flowOf(2);
  const script = new Script(new TimeoutError("a"));
  const { scheduler } = clock();

  const result = await runFlow(flow, {}, {
    log: path,
    runId: "r1",
    tools: script.tools(),
    ...host(scheduler),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(steps(path)).toEqual([failed(1, "a", 1), started(2)]);

  const resumed = await advance({ log: path, tools: script.tools(), ...host(scheduler) });

  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("ok");
  expect(script.seen).toHaveLength(2);
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
  const crash = new Crash("process lost");
  const refuse = (_log: RunLog, record: RunRecord) => {
    if (record instanceof NodeSucceededRecord && record.nodeId.endsWith("/place")) throw crash;
  };
  const error = await withAppendHook(refuse, () =>
    rejected(runFlow(flow, {}, { log: path, runId: "r1", tools: { x }, numWorkers: 4, ...host(scheduler) })),
  );
  expect(error).toBe(crash);

  crashed = true;
  const ids = new Map(seen.filter(([, ctx]) => ctx.attempt === 1).map(([n, ctx]) => [n, ctx.node]));
  expect(new Set(ids.values()).size).toBe(2);
  let byNode = stepsByNode(path);
  for (const [n, nodeId] of ids) expect(byNode.get(nodeId)).toEqual([failed(1, `a${n}`, 1), started(2)]);

  seen.length = 0;
  const result = await advance({ log: path, tools: { x }, numWorkers: 4, ...host(scheduler) });

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
 * records and the live state after each commit past `RunStarted`.
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
  // The snapshot before record k is the state after record k - 1; the first is after the run start.
  return { records: log.records(), live: [...log.before.slice(1), liveOf(state)], state };
}

/** The number of the run start's records (RunStarted and its commits): those before the first node start. */
const runStartLength = (records: readonly RecordEnvelope[]) => records.findIndex(({ kind }) => kind === "node_started");

/** The replayed state of the first `length` records. */
const replayPrefix = (loaded: ComposedFlow, records: readonly RecordEnvelope[], length: number) =>
  replayRunLog(loaded, MemoryRunLog.fromRecords(records.slice(0, length)));

test("a finished log with retries replays to the live steps", async () => {
  const flow = await flowOf(2);
  const script = new Script(new TimeoutError("a"), new TimeoutError("b"));

  const { records, live, state } = await liveRun(flow, script.tools());

  expect(state.result?.constructor.name).toBe("RunSucceeded");
  // The run start is written before the engine runs; every later record is one live commit.
  const started = runStartLength(records);
  expect(live).toHaveLength(records.length - started);
  // Every prefix replays to the live node starts and steps after that commit.
  for (const [i, expected] of live.entries()) {
    const length = started + 1 + i;
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
  const started = runStartLength(records);
  for (const length of stepPrefixes.filter((_, i) => i % 10 === 0)) {
    const { starts, steps } = live[length - started - 1]!;
    const state = replayPrefix(flow, records, length);
    expect(state.nodeStarts).toEqual(starts);
    expect(state.nodeSteps).toEqual(steps as Map<string, readonly never[]>);
    expect(state.nodeSteps.size).toBe(1);
  }
});

// --- Schema-9 logs ---------------------------------------------------------------------- //

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
  test(`${name} is refused untouched`, async () => {
    const path = freshLog();
    copyFileSync(join(SCHEMA9_FIXTURES, fixture), path);
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

    const error = await rejected(advance({ log: path, tools: { place }, ...host(scheduler) }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(
      `run log ${path} line 1 (seq 0): unsupported run-record schema: 9`,
    );
    expect(calls).toEqual([]);
    expect(readFileSync(path).equals(before)).toBe(true);
  });
}
