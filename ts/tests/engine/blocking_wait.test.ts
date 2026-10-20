/**
 * Blocking WAIT semantics and cooperative wakeups: a port of `tests/engine/test_blocking_wait.py`.
 *
 * TS has no stop event and no `_HostStop`: the wait sleeps on the node's `caps.scheduler` with its
 * `caps.signal`, and a stop rejects with the signal's reason. The direct rows run the node on a
 * scheduler whose clock the test sets, in place of patching `_utc_now`, and read each sleep in
 * milliseconds where Python reads `Event.wait` timeouts in seconds.
 *
 * Ported (adapted): `test_date_means_midnight_utc`, `test_naive_datetime_means_utc`,
 * `test_aware_datetime_is_normalized_to_utc` and
 * `test_deadline_accepts_z_short_fraction_and_space_separator` read the wait as the sleeps on a
 * clock that keeps time: a wait longer than `SLEEP_CHUNK_MS` is whole chunks, then the rest, where
 * Python waits once.
 *
 * Changed:
 * - `test_deadline_at_or_before_now_returns_immediately` runs with an already-aborted signal, which
 *   the wait doesn't check then, as Python doesn't check its event.
 * - `test_wait_uses_remaining_absolute_duration` drops the `needs_stop_event` check: every node
 *   gets its signal.
 * - `test_far_future_deadline_waits_in_timeout_max_chunks` and
 *   `test_host_stop_between_timeout_max_chunks` chunk at `SLEEP_CHUNK_MS` (one hour), where Python
 *   chunks at `TIMEOUT_MAX` (about 49 days); the far-future row's clock jumps past a chunk, as
 *   after a sleeping machine, and each sleep is `min(deadline - now(), SLEEP_CHUNK_MS)`.
 * - The engine rows wait on the real clock for `9999-12-31` through a scheduler that reports the
 *   first sleep; the abort clears the timer, so no thread check is needed.
 *   `test_host_stop_wakes_wait_and_discards_its_outcome` checks that `run()` rejects with the host
 *   signal's reason (no `primary` flag) and that `RunStopped` is recorded;
 *   `test_observer_failure_wakes_wait_and_preserves_callback_error` and
 *   `test_sibling_failure_wakes_wait_and_remains_terminal_cause` check the run's signal aborted in
 *   place of the stop event being set; their siblings are `BlockingNode`s gated on the first
 *   sleep, as in `blocking_workers.test.ts`.
 *
 * TS-only: microseconds are truncated to milliseconds.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeFailed, NodeSucceeded, type RunEvent, RunFailed } from "../../src/events.ts";
import { type Node, Output } from "../../src/nodes/base.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { realScheduler, type Scheduler, SLEEP_CHUNK_MS, virtualScheduler } from "../../src/platform.ts";
import { RunStoppedRecord } from "../../src/runtime/log.ts";
import { BlockingNode, engine, graph, rejected } from "./run_state_fixtures.ts";

const NOW = Date.UTC(2026, 0, 1);
const FAR_FUTURE = "9999-12-31";

/** A virtual scheduler starting at `NOW` that records each sleep it is asked for. */
function recordingScheduler(): { scheduler: Scheduler; sleeps: number[] } {
  const inner = virtualScheduler(NOW);
  const sleeps: number[] = [];
  const scheduler: Scheduler = {
    now: () => inner.now(),
    sleep(ms, signal) {
      sleeps.push(ms);
      return inner.sleep(ms, signal);
    },
  };
  return { scheduler, sleeps };
}

/**
 * A scheduler whose clock reads `nows` in turn and whose sleeps return at once, recording each
 * sleep and, at each clock read, the sleeps so far. `onSleep` runs with the sleep count before the
 * sleep checks its signal.
 */
function steppedScheduler(
  nows: number[],
  onSleep: (count: number) => void = () => {},
): { scheduler: Scheduler; sleeps: number[]; clockReads: number[][] } {
  const sleeps: number[] = [];
  const clockReads: number[][] = [];
  const scheduler: Scheduler = {
    now() {
      clockReads.push([...sleeps]);
      return nows.shift()!;
    },
    async sleep(ms, signal) {
      sleeps.push(ms);
      onSleep(sleeps.length);
      signal?.throwIfAborted();
    },
  };
  return { scheduler, sleeps, clockReads };
}

/** The real scheduler, with `entered` resolving at the first sleep. */
function observedScheduler(): { scheduler: Scheduler; entered: Promise<void> } {
  const { promise, resolve } = Promise.withResolvers<void>();
  const scheduler: Scheduler = {
    now: () => realScheduler.now(),
    sleep(ms, signal) {
      resolve();
      return realScheduler.sleep(ms, signal);
    },
  };
  return { scheduler, entered: promise };
}

/** The sleeps a wait of `ms` takes on a clock that keeps time: whole chunks, then the rest. */
function chunked(ms: number): number[] {
  const sleeps = Array<number>(Math.floor(ms / SLEEP_CHUNK_MS)).fill(SLEEP_CHUNK_MS);
  return ms % SLEEP_CHUNK_MS > 0 ? [...sleeps, ms % SLEEP_CHUNK_MS] : sleeps;
}

/** Run a WAIT for `until` directly on a recording scheduler; return its result and sleeps. */
async function runDirect(until: string, signal?: AbortSignal): Promise<{ result: unknown; sleeps: number[] }> {
  const { scheduler, sleeps } = recordingScheduler();
  const result = await new WaitNode("wait").run({ until }, { scheduler, signal });
  return { result, sleeps };
}

/** Return `START -> {wait, ...nodes} -> END`, the WAIT waiting for `FAR_FUTURE`. */
function waitGraph(...nodes: Node[]) {
  const body = [new WaitNode("wait"), ...nodes];
  const edges = [
    ...body.map((node) => [START_ID, node.id] as const),
    ...body.map((node) => [node.id, END_ID] as const),
  ];
  return graph(body, edges, { wiring: { wait: { until: FAR_FUTURE } } });
}

/** Whether `events` hold a `NodeSucceeded` or a `NodeFailed` for the WAIT. */
function waitSettled(events: readonly RunEvent[]): { succeeded: boolean; failed: boolean } {
  return {
    succeeded: events.some((event) => event instanceof NodeSucceeded && event.nodeId === "wait"),
    failed: events.some((event) => event instanceof NodeFailed && event.failure.nodeId === "wait"),
  };
}

test("a date means midnight UTC", async () => {
  const { result, sleeps } = await runDirect("2026-01-02");

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual(chunked(86_400_000));
});

test("a naive datetime means UTC", async () => {
  const { result, sleeps } = await runDirect("2026-01-01T01:02:03");

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual(chunked(3_723_000));
});

test("an aware datetime is normalized to UTC", async () => {
  const { result, sleeps } = await runDirect("2026-01-02T05:30:00+05:30");

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual(chunked(86_400_000));
});

test.each([
  ["z-suffix", "2026-01-02T00:00:00Z", 86_400_000],
  ["short-fraction", "2026-01-01T00:00:00.5Z", 500],
  ["space-separator", "2026-01-01 01:00", 3_600_000],
  ["microseconds-truncated", "2026-01-01T00:00:00.001999Z", 1],
] as const)("the deadline accepts Z, a short fraction and a space separator (%s)", async (_, deadline, ms) => {
  const { result, sleeps } = await runDirect(deadline);

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual(chunked(ms));
});

test.each([
  ["past", "2025-12-31T23:59:59+00:00"],
  ["exact-boundary", "2026-01-01T00:00:00+00:00"],
])("a deadline at or before now returns at once (%s)", async (_, deadline) => {
  const { result, sleeps } = await runDirect(deadline, AbortSignal.abort(new Error("stopped")));

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual([]);
});

test("the wait is the remaining absolute duration", async () => {
  const { result, sleeps } = await runDirect("2026-01-01T00:00:12.500000+00:00");

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual([12_500]);
});

test("a far-future deadline waits in SLEEP_CHUNK_MS chunks, re-reading the clock", async () => {
  const deadline = NOW + 3 * SLEEP_CHUNK_MS + 12;
  // The second read jumps a chunk further than the sleep, as after a sleeping machine.
  const { scheduler, sleeps, clockReads } = steppedScheduler([
    NOW,
    NOW + 2 * SLEEP_CHUNK_MS,
    NOW + 3 * SLEEP_CHUNK_MS,
    deadline,
  ]);

  const result = await new WaitNode("wait").run({ until: new Date(deadline).toISOString() }, { scheduler });

  expect(result).toEqual(new Output(null));
  expect(sleeps).toEqual([SLEEP_CHUNK_MS, SLEEP_CHUNK_MS, 12]);
  expect(clockReads).toEqual([
    [],
    [SLEEP_CHUNK_MS],
    [SLEEP_CHUNK_MS, SLEEP_CHUNK_MS],
    [SLEEP_CHUNK_MS, SLEEP_CHUNK_MS, 12],
  ]);
});

test("a host stop between SLEEP_CHUNK_MS chunks rejects with the signal's reason", async () => {
  const host = new AbortController();
  const reason = new Error("stop now");
  const deadline = NOW + 2 * SLEEP_CHUNK_MS + 12;
  const { scheduler, sleeps } = steppedScheduler([NOW, NOW + SLEEP_CHUNK_MS], (count) => {
    if (count === 2) host.abort(reason);
  });

  const error = await rejected(
    new WaitNode("wait").run({ until: new Date(deadline).toISOString() }, { scheduler, signal: host.signal }),
  );

  expect(error).toBe(reason);
  expect(sleeps).toEqual([SLEEP_CHUNK_MS, SLEEP_CHUNK_MS]);
});

for (const numWorkers of [0, 1]) {
  test(`a host stop wakes the wait and discards its outcome (numWorkers ${numWorkers})`, async () => {
    const { scheduler, entered } = observedScheduler();
    const host = new AbortController();
    const reason = new Error("stop now");
    const events: RunEvent[] = [];
    const built = engine(waitGraph(), {
      numWorkers,
      signal: host.signal,
      scheduler,
      onEvent: (event) => {
        events.push(event);
      },
    });

    const running = built.run();
    await entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(waitSettled(events)).toEqual({ succeeded: false, failed: false });
    const stops = built.log
      .records()
      .map((envelope) => envelope.record)
      .filter((record) => record instanceof RunStoppedRecord);
    expect(stops).toEqual([new RunStoppedRecord("signal", "stop now", ["wait"])]);
  });
}

test("an observer failure wakes the wait and stays the run's error", async () => {
  const { scheduler, entered } = observedScheduler();
  const sibling = new BlockingNode("sibling", entered.then(() => "done"));
  const original = new Error("observer failed");
  const observed: RunEvent[] = [];
  const built = engine(waitGraph(sibling), {
    numWorkers: 2,
    scheduler,
    onEvent: (event) => {
      observed.push(event);
      if (event instanceof NodeSucceeded && event.nodeId === "sibling") throw original;
    },
  });

  expect(await rejected(built.run())).toBe(original);
  expect(built.signal.aborted).toBe(true);
  expect(waitSettled(observed).failed).toBe(false);
});

test("a sibling failure wakes the wait and stays the terminal cause", async () => {
  const { scheduler, entered } = observedScheduler();
  const fail = new BlockingNode(
    "fail",
    entered.then(() => {
      throw new Error("sibling failed");
    }),
  );
  const built = engine(waitGraph(fail), { numWorkers: 2, scheduler });

  const events = await built.run();

  expect(events.at(-1)).toBeInstanceOf(RunFailed);
  expect((events.at(-1) as RunFailed).failure.message).toBe("sibling failed");
  expect(built.signal.aborted).toBe(true);
  expect(waitSettled(events)).toEqual({ succeeded: false, failed: false });
});
