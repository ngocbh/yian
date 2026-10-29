/**
 * Explicit backward attempts over a retained forward run, through `runFlow`, `advance` and
 * `RunResult.backward`, with memory keys served by the shared `FakeMemory`. An unfinished backward
 * attempt fails `backward.unfinished`.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { type ChildProcess, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { DefaultsMismatchError } from "../../src/defaults.ts";
import { ModelCallRetried, RunStarted, RunSucceeded } from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryOperationError, MemorySetupError } from "../../src/memory/contracts.ts";
import {
  replayRunLog,
  type AdvanceOptions,
  type RunFlowOptions,
  type RunResult,
  advance,
  RunBusyError,
  runFlow,
} from "../../src/run.ts";
import { FileLockError, lockFile, virtualScheduler } from "../../src/platform.ts";
import { computeBackward } from "../../src/runtime/backward.ts";
import { MemoryProposal } from "../../src/runtime/contracts.ts";
import { memoryCommitId } from "../../src/runtime/learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  JsonlRunLog,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  MemoryRunLog,
  readRunLog,
  type RecordEnvelope,
  type RunLog,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { commitPrepared, prepareRecord } from "../../src/runtime/state.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";

import { withAppendHook } from "../helpers/append_hook.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { FakeMemory, FakeSession } from "../helpers/memory_fakes.ts";
import { backwardReply, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const IDENTITY = "id: identity\nname: identity\ninput:\n  text: !str\nnodes: {}\noutput: ${input.text}\n";
const WITH_MEMORY = IDENTITY.replace("name: identity", "name: identity\nlong_term_memory: test");
const TS_ROOT = fileURLToPath(new URL("../../", import.meta.url));

let dir: string;
let cwd: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-backward-"));
  cwd = process.cwd();
});

afterEach(() => {
  process.chdir(cwd);
  rmSync(dir, { recursive: true, force: true });
});

/** `runFlow` in `hostEnv(dir)` unless `options` sets `env`. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(dir), ...options });
}

/** `advance` in `hostEnv(dir)` unless `options` sets `env`. */
function resume(flow: ComposedFlow, options: AdvanceOptions) {
  return advance({ env: hostEnv(dir), ...options });
}

/** A `Map` serving the memory key `test` with `backend`. */
function testMemory(backend: unknown): Map<string, FakeMemory> {
  return new Map([["test", backend as FakeMemory]]);
}

/** The records of the closed durable log at `path`, proving its lock was released. */
function records(path: string): RunRecord[] {
  const log = JsonlRunLog.resume(path);
  try {
    return log.records().map((envelope) => envelope.record);
  } finally {
    log.close();
  }
}

/** The attempt ids of the `BackwardStarted` records at `path`. */
function startedAttempts(path: string): number[] {
  return records(path).flatMap((record) => (record instanceof BackwardStartedRecord ? [record.attemptId] : []));
}

/** A learner refusal the attempt records as failed. */
function declined(): MemoryOperationError {
  return new MemoryOperationError("learner", "declined", "Try again.", true);
}

/** Replace `owner.prototype[name]` with `replacement(original)` while `fn` runs. */
async function patched<T>(
  owner: { prototype: object },
  name: string,
  replacement: (original: Function) => Function,
  fn: () => Promise<T>,
): Promise<T> {
  const prototype = owner.prototype as Record<string, Function>;
  const original = prototype[name]!;
  prototype[name] = replacement(original);
  try {
    return await fn();
  } finally {
    prototype[name] = original;
  }
}

/** Whether `log` holds a `BackwardStarted` record. */
function hasBackwardStarted(log: { records(): readonly RecordEnvelope[] }, original: Function): boolean {
  return (original.call(log) as readonly RecordEnvelope[]).some((envelope) => envelope.record instanceof BackwardStartedRecord);
}

test.each(["memory", "durable"])("a %s log stamps backward records from the run's clock", async (kind) => {
  const log = kind === "durable" ? join(dir, "run.jsonl") : null;
  const scheduler = virtualScheduler(Date.UTC(2026, 9, 20, 14, 0));
  const result = await run(await loadFlow(IDENTITY), { text: "hello" }, { learnable: true, log, scheduler });
  const appended = new Set<RunLog>();
  const outcome = await withAppendHook((into) => appended.add(into), () => result.backward("Too long."));
  expect(outcome.status).toBe("succeeded");
  const backward = (log === null ? [...appended].flatMap((into) => into.records()) : readRunLog(log))
    .filter((envelope) => envelope.kind.startsWith("backward_"))
    .map((envelope) => envelope.at);
  expect(backward.length).toBeGreaterThan(0);
  expect(new Set(backward)).toEqual(new Set(["2026-10-20T14:00:00.000Z"]));
});

test("backward without memory is explicit and repeatable (memory)", async () => {
  const result = await run(await loadFlow(IDENTITY), { text: "hello" }, { learnable: true });
  expect(result.backwardOutcome).toBeNull();
  expect(result.learning).toBeNull();
  const before = [result.input, result.status, result.output, result.failure, [...result.events]];

  const outcome = await result.backward("Too long.");
  expect(outcome.status).toBe("succeeded");
  expect(outcome.feedback!.inputFeedback).toEqual({ text: ["Too long."] });
  expect(result.backwardOutcome).toBe(outcome);
  expect(result.learning).toBeNull();
  expect([result.input, result.status, result.output, result.failure, result.events]).toEqual(before);
  expect(Object.isFrozen(outcome)).toBe(true);
  const again = await result.backward("Again.");
  expect(again.status).toBe("succeeded");
  expect(again.feedback!.inputFeedback).toEqual({ text: ["Again."] });
  expect(result.backwardOutcome).toBe(again);
});

test("forward returns before any learning or commit (the backward half)", async () => {
  const path = join(dir, "run.jsonl");
  const memory = new FakeMemory();
  const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
    learnable: true,
    log: path,
    memory: testMemory(memory),
  });
  expect(memory.learnContexts).toEqual([]);
  expect(memory.commitCalls).toEqual([]);
  expect(memory.closeCount).toBe(1);
  const forward = records(path);

  await result.backward("Review.");

  expect(memory.learnContexts).toHaveLength(1);
  expect(memory.commitCalls).toHaveLength(1);
  expect(memory.learnContexts[0]!.episodes).toHaveLength(2);
  expect(result.learning!.status).toBe("succeeded");
  const after = records(path);
  expect(after.slice(0, forward.length)).toEqual(forward);
  expect(after.slice(forward.length).map((record) => record.constructor)).toEqual([
    BackwardStartedRecord,
    BackwardReadyRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    BackwardFinishedRecord,
  ]);
});

test("same handle rejects nested backward before a second commit", async () => {
  const rejected: unknown[] = [];
  let result: RunResult;
  const backend = new FakeMemory({
    learnOutcomes: [
      () => {
        result.backward("Nested review.").catch((error: unknown) => rejected.push(error));
        return [];
      },
    ],
  });
  result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, { learnable: true, memory: testMemory(backend) });

  expect((await result.backward("Outer review.")).status).toBe("succeeded");

  expect(backend.commitCalls).toHaveLength(1);
  expect(backend.learnContexts).toHaveLength(1);
  expect(rejected).toHaveLength(1);
  expect((rejected[0] as Error).message).toBe("a backward call on this result is already in progress");
  expect(backend.closeCount).toBe(2);
  expect((await result.backward("Later review.")).status).toBe("succeeded");
  expect(backend.commitCalls).toHaveLength(2);
  expect(backend.closeCount).toBe(3);
});

for (const durable of [false, true]) {
  test(`same handle rejects concurrent backward without waiting (${durable})`, async () => {
    let enter!: () => void;
    const entered = new Promise<void>((resolve) => {
      enter = resolve;
    });
    let release!: () => void;
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    const backend = new FakeMemory({
      learnOutcomes: [
        async () => {
          enter();
          await released;
          return [];
        },
      ],
    });
    const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
      learnable: true,
      log: durable ? join(dir, "run.jsonl") : null,
      memory: testMemory(backend),
    });

    const pending = result.backward("First review.");
    try {
      await entered;
      const error = await rejected(result.backward("Concurrent review."));
      expect((error as Error).message).toBe("a backward call on this result is already in progress");
      expect(backend.learnContexts).toHaveLength(1);
      expect(backend.commitCalls).toEqual([]);
    } finally {
      release();
    }
    expect((await pending).status).toBe("succeeded");
    expect(backend.commitCalls).toHaveLength(1);
    expect(backend.closeCount).toBe(2);
  });
}

for (const acquisition of ["fresh_event", "fresh_after", "resume_event", "resume_after"]) {
  test(`delayed backward anchors the log at acquisition (${acquisition})`, async () => {
    const acquired = join(dir, "acquired");
    const elsewhere = join(dir, "elsewhere");
    mkdirSync(acquired);
    mkdirSync(elsewhere);
    process.chdir(acquired);
    const flow = await loadFlow(IDENTITY);
    const moveAfterForward = (event: unknown) => {
      if (event instanceof RunSucceeded) process.chdir(elsewhere);
    };

    let result: RunResult;
    if (acquisition === "resume_event") {
      const interrupted = new Error("interrupt before forward evaluation");
      const stopAfterStart = (event: unknown) => {
        if (event instanceof RunStarted) throw interrupted;
      };
      expect(
        await rejected(run(flow, { text: "hello" }, { learnable: true, log: "run.jsonl", onEvent: stopAfterStart })),
      ).toBe(interrupted);
      result = await resume(flow, { log: "run.jsonl", onEvent: moveAfterForward });
    } else if (acquisition === "resume_after") {
      await run(flow, { text: "hello" }, { learnable: true, log: "run.jsonl" });
      result = await resume(flow, { log: "run.jsonl" });
    } else {
      result = await run(flow, { text: "hello" }, {
        learnable: true,
        log: "run.jsonl",
        onEvent: acquisition === "fresh_event" ? moveAfterForward : null,
      });
    }
    process.chdir(elsewhere);

    expect(result.log).toBe("run.jsonl");
    expect((await result.backward("Review.")).feedback!.inputFeedback).toEqual({ text: ["Review."] });
    expect(records(join(acquired, "run.jsonl")).at(-1)).toBeInstanceOf(BackwardFinishedRecord);
    expect(existsSync(join(elsewhere, "run.jsonl"))).toBe(false);
  });
}

for (const opening of ["create", "resume"] as const) {
  test(`acquisition uses the captured locator if opener changes cwd (${opening})`, async () => {
    const acquired = join(dir, "acquired");
    const elsewhere = join(dir, "elsewhere");
    mkdirSync(acquired);
    mkdirSync(elsewhere);
    process.chdir(acquired);
    const flow = await loadFlow(IDENTITY);
    let otherPrefix: string | null = null;
    if (opening === "resume") {
      await run(flow, { text: "hello" }, { learnable: true, log: "run.jsonl" });
      await run(flow, { text: "other run" }, { learnable: true, log: join(elsewhere, "run.jsonl") });
      otherPrefix = readFileSync(join(elsewhere, "run.jsonl"), "utf8");
    }
    const original = JsonlRunLog[opening];
    const spy = spyOn(JsonlRunLog, opening).mockImplementation(((path: string, runId?: string) => {
      process.chdir(elsewhere);
      return (original as (path: string, runId?: string) => JsonlRunLog).call(JsonlRunLog, path, runId);
    }) as never);
    let result: RunResult;
    try {
      result =
        opening === "create"
          ? await run(flow, { text: "hello" }, { learnable: true, log: "run.jsonl" })
          : await resume(flow, { log: "run.jsonl" });
    } finally {
      spy.mockRestore();
    }

    expect(result.output).toBe("hello");
    expect(result.log).toBe("run.jsonl");
    expect((await result.backward("Review.")).feedback!.inputFeedback).toEqual({ text: ["Review."] });
    expect(records(join(acquired, "run.jsonl")).at(-1)).toBeInstanceOf(BackwardFinishedRecord);
    if (otherPrefix === null) expect(existsSync(join(elsewhere, "run.jsonl"))).toBe(false);
    else expect(readFileSync(join(elsewhere, "run.jsonl"), "utf8")).toBe(otherPrefix);
  });
}

for (const durable of [false, true]) {
  for (const phase of ["learner", "commit", "finished"]) {
    test(`prefix export failure preserves primary and prevents stale retry (${durable}-${phase})`, async () => {
      const primary = new Error(phase === "learner" ? "learner failed" : "backend failed");
      const exportError = new Error("prefix export failed");
      const closeError = new Error("log close failed");
      const backend = new FakeMemory({
        learnOutcomes: phase === "learner" ? [primary] : [],
        commitOutcomes: phase === "commit" ? [primary] : [],
      });
      const flow = await loadFlow(WITH_MEMORY);
      const path = durable ? join(dir, "export.jsonl") : null;
      const result = await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(backend) });
      const logType = durable ? JsonlRunLog : MemoryRunLog;
      const closed: unknown[] = [];

      const error = await patched(
        logType,
        "records",
        (original) =>
          function (this: MemoryRunLog) {
            if (hasBackwardStarted(this, original)) throw exportError;
            return original.call(this);
          },
        () =>
          patched(
            logType,
            "close",
            (original) =>
              function (this: MemoryRunLog) {
                original.call(this);
                closed.push(this);
                throw closeError;
              },
            () => rejected(result.backward("Review.")),
          ),
      );

      expect(error).toBe(phase === "finished" ? exportError : primary);
      expect(closed).toHaveLength(1);
      expect(backend.closeCount).toBe(2);
      expect(result.backwardOutcome).not.toBeNull();
      expect(result.backwardOutcome!.status).toBe(phase === "finished" ? "succeeded" : "failed");
      if (phase === "commit") expect(result.learning!.policies[0]!.status).toBe("unknown");
      const [learned, committed] = [backend.learnContexts.length, backend.commitCalls.length];

      const stale = await rejected(result.backward("Do not retry stale history."));
      expect((stale as Error).message).toBe("backward handle is unavailable after an accepted-prefix refresh failure");
      expect([backend.learnContexts.length, backend.commitCalls.length]).toEqual([learned, committed]);
      if (durable) expect((await resume(flow, { log: path! })).backwardOutcome).toEqual(result.backwardOutcome);
    });
  }
}

const ACCEPTED_APPENDS = [
  [BackwardStartedRecord, "unrequested"],
  [MemoryCommitRequestedRecord, "unknown"],
  [MemoryCommitSucceededRecord, "succeeded"],
  [BackwardFinishedRecord, "finished"],
] as const;

for (const durable of [false, true]) {
  for (const [recordType, disposition] of ACCEPTED_APPENDS) {
    test(`accepted append error refreshes the actual prefix (${durable}-${recordType.name})`, async () => {
      const backend = new FakeMemory();
      const path = durable ? join(dir, "accepted.jsonl") : null;
      const flow = await loadFlow(WITH_MEMORY);
      const result = await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(backend) });
      const failure = new Error("append accepted before returning");
      let failOnce = true;

      const error = await patched(
        durable ? JsonlRunLog : MemoryRunLog,
        "append",
        (original) =>
          function (this: MemoryRunLog, record: RunRecord) {
            const envelope = original.call(this, record);
            if (failOnce && record instanceof recordType) {
              failOnce = false;
              throw failure;
            }
            return envelope;
          },
        () => rejected(result.backward("Review.")),
      );

      expect(error).toBe(failure);
      expect(result.backwardOutcome).not.toBeNull();
      if (disposition === "finished") {
        expect(result.backwardOutcome!.status).toBe("succeeded");
        expect(result.learning!.status).toBe("succeeded");
      } else {
        expect(result.backwardOutcome!.failure!.code).toBe("backward.unfinished");
        expect(result.learning!.status).toBe("incomplete");
        expect(result.learning!.policies[0]!.status).toBe(disposition);
      }
      if (durable) expect((await resume(flow, { log: path! })).backwardOutcome).toEqual(result.backwardOutcome);
      const [learned, committed] = [backend.learnContexts.length, backend.commitCalls.length];
      if (disposition === "unrequested") expect([learned, committed]).toEqual([0, 0]);
      expect((await result.backward("Later whole pass.")).status).toBe("succeeded");
      expect([backend.learnContexts.length, backend.commitCalls.length]).toEqual([learned + 1, committed + 1]);
    });
  }
}

test("each explicit attempt restores fresh sessions from recorded snapshots", async () => {
  const backend = new FakeMemory({ learnOutcomes: [declined(), []] });
  const replacement = new FakeMemory();
  const mapping = testMemory(backend);
  const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, { learnable: true, memory: mapping });
  expect(backend.sessions).toHaveLength(1);
  expect(backend.sessions[0]!.closed).toBe(true);
  mapping.set("test", replacement);

  expect((await result.backward("First review.")).status).toBe("failed");
  expect((await result.backward("Fresh review.")).status).toBe("succeeded");

  expect(backend.openSnapshots).toEqual([null, { namespace: "test" }, { namespace: "test" }]);
  expect(new Set(backend.sessions).size).toBe(3);
  expect(backend.sessions.every((session) => session.closed)).toBe(true);
  expect(backend.learnContexts).toHaveLength(2);
  expect(backend.commitCalls).toHaveLength(1);
  expect(replacement.sessions).toEqual([]);
});

test("terminal resume retains mapping references without opening them", async () => {
  const path = join(dir, "terminal.jsonl");
  const flow = await loadFlow(WITH_MEMORY);
  const original = new FakeMemory();
  await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(original) });
  const missing = await resume(flow, { log: path });
  const prefix = readFileSync(path);

  expect(await rejected(missing.backward("Review."))).toBeInstanceOf(MemorySetupError);

  expect(readFileSync(path)).toEqual(prefix);
  expect(missing.backwardOutcome).toBeNull();
  const backend = new FakeMemory();
  const replacement = new FakeMemory();
  const mapping = testMemory(backend);
  const inspected = await resume(flow, { log: path, memory: mapping });
  expect(backend.sessions).toEqual([]);
  expect(readFileSync(path)).toEqual(prefix);
  mapping.set("test", replacement);

  expect((await inspected.backward("Review.")).status).toBe("succeeded");

  expect(backend.openSnapshots).toEqual([{ namespace: "test" }]);
  expect(backend.sessions).toHaveLength(1);
  expect(backend.sessions[0]!.closed).toBe(true);
  expect(replacement.sessions).toEqual([]);
  expect(original.learnContexts).toEqual([]);
});

test("failed snapshot restoration keeps interrupted prefix and observation", async () => {
  const path = join(dir, "restore.jsonl");
  const interrupted = new Error("interrupted learner");
  const backend = new FakeMemory({ learnOutcomes: [interrupted, []] });
  const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
    learnable: true,
    log: path,
    memory: testMemory(backend),
  });
  expect(await rejected(result.backward("First review."))).toBe(interrupted);
  const [previous, learning, prefix] = [result.backwardOutcome, result.learning, readFileSync(path)];
  const openRun = backend.openRun;
  backend.openRun = (options = {}) => {
    openRun.call(backend, options);
    const session = new FakeSession(backend, { namespace: "wrong snapshot" });
    backend.sessions.push(session);
    return session;
  };

  try {
    expect(await rejected(result.backward("Review after restoration."))).toBeInstanceOf(MemorySetupError);
  } finally {
    backend.openRun = openRun;
  }

  expect(result.backwardOutcome).toBe(previous);
  expect(result.learning).toBe(learning);
  expect(readFileSync(path)).toEqual(prefix);
  expect(backend.sessions.at(-1)!.closed).toBe(true);
  expect(backend.learnContexts).toHaveLength(1);
  expect(backend.commitCalls).toEqual([]);
  expect((await result.backward("Fresh review.")).status).toBe("succeeded");
  expect(startedAttempts(path)).toEqual([1, 2]);
});

for (const durable of [false, true]) {
  test(`delayed backward uses private program and canonical values (${durable})`, async () => {
    const flow = await loadFlow(IDENTITY.replace("text: !str", "text: !object"));
    const supplied = { text: { items: [1, 2] } };
    const result = await run(flow, supplied, { learnable: true, log: durable ? join(dir, "private.jsonl") : null });
    supplied.text.items.push(3);
    expect(result.input).toEqual({ text: { items: [1n, 2n] } });
    (result.input as { text: { items: unknown[] } }).text.items.push(4n);
    (result.output as { items: unknown[] }).items.push(5n);
    flow.compiled.wiring.set(flow.compiled.endId, new Map());
    flow.compiled.nodes.clear();

    const outcome = await result.backward({ items: ["Review first.", null] });

    expect(outcome.status).toBe("succeeded");
    expect(outcome.feedback!.inputFeedback).toEqual({ text: [{ items: ["Review first.", null] }] });
    expect(result.output).toEqual({ items: [1n, 2n, 5n] });
    expect(result.input).toEqual({ text: { items: [1n, 2n, 4n] } });
  });
}

test("distinct durable handles keep file lock exclusion", async () => {
  const path = join(dir, "locked.jsonl");
  const errors: unknown[] = [];
  let second: RunResult;
  const flow = await loadFlow(WITH_MEMORY);
  const backend = new FakeMemory({
    learnOutcomes: [
      async () => {
        errors.push(await rejected(second.backward("Nested on another handle.")));
        return [];
      },
    ],
  });
  const first = await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(backend) });
  second = await resume(flow, { log: path, memory: testMemory(backend) });

  expect((await first.backward("Review.")).status).toBe("succeeded");

  expect(errors).toHaveLength(1);
  expect(errors[0]).toBeInstanceOf(RunBusyError);
  expect((errors[0] as Error).message).toBe(`run log ${path} is locked by another writer (is the run still active?)`);
  expect(backend.learnContexts).toHaveLength(1);
  expect(backend.commitCalls).toHaveLength(1);
  expect(second.backwardOutcome).toBeNull();
  expect((await resume(flow, { log: path })).backwardOutcome).toEqual(first.backwardOutcome);
});

for (const phase of ["export", "close"]) {
  test(`handle guard covers prefix export and log cleanup (${phase})`, async () => {
    const backend = new FakeMemory();
    const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
      learnable: true,
      memory: testMemory(backend),
    });
    const nested: Promise<unknown>[] = [];
    const readRecords = MemoryRunLog.prototype.records;

    const outcome = await patched(
      MemoryRunLog,
      phase === "export" ? "records" : "close",
      (original) =>
        function (this: MemoryRunLog) {
          if (hasBackwardStarted(this, readRecords)) nested.push(rejected(result.backward("Nested during cleanup.")));
          return original.call(this);
        },
      () => result.backward("Review."),
    );

    expect(outcome.status).toBe("succeeded");
    const errors = await Promise.all(nested);
    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toBe("a backward call on this result is already in progress");
    expect(backend.learnContexts).toHaveLength(1);
    expect(backend.commitCalls).toHaveLength(1);
  });
}

for (const resource of ["session", "log"]) {
  for (const learnerFails of [false, true]) {
    test(`cleanup error preserves finished authority and releases guard (${learnerFails}-${resource})`, async () => {
      const backend = new FakeMemory({ learnOutcomes: learnerFails ? [declined(), []] : [] });
      const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
        learnable: true,
        memory: testMemory(backend),
      });
      const failure = new Error("cleanup failed");

      const error = await patched(
        resource === "session" ? FakeSession : MemoryRunLog,
        "close",
        (original) =>
          function (this: unknown) {
            original.call(this);
            throw failure;
          },
        () => rejected(result.backward("Review.")),
      );

      expect(error).toBe(failure);
      expect(result.backwardOutcome!.status).toBe(learnerFails ? "failed" : "succeeded");
      expect(backend.sessions.every((session) => session.closed)).toBe(true);
      expect((await result.backward("Fresh review.")).status).toBe("succeeded");
      expect(backend.learnContexts).toHaveLength(2);
      expect(backend.commitCalls).toHaveLength(learnerFails ? 1 : 2);
    });
  }
}

test("cancelled backward retains started and releases resources", async () => {
  const failure = new DOMException("cancelled", "AbortError");
  const backend = new FakeMemory({ learnOutcomes: [failure, []] });
  const result = await run(await loadFlow(WITH_MEMORY), { text: "hello" }, {
    learnable: true,
    memory: testMemory(backend),
  });

  expect(await rejected(result.backward("Interrupted review."))).toBe(failure);

  expect(result.backwardOutcome!.failure!.code).toBe("backward.unfinished");
  expect(result.backwardOutcome!.feedback).toBeNull();
  expect(backend.sessions.every((session) => session.closed)).toBe(true);
  expect(backend.commitCalls).toEqual([]);
  expect((await result.backward("Fresh whole pass.")).status).toBe("succeeded");
  expect(backend.learnContexts).toHaveLength(2);
  expect(backend.commitCalls).toHaveLength(1);
});

for (const previousFailure of [false, true]) {
  test(`invalid feedback on stale handle preserves its observation (${previousFailure})`, async () => {
    const path = join(dir, "stale.jsonl");
    const flow = await loadFlow(previousFailure ? WITH_MEMORY : IDENTITY);
    const backend = new FakeMemory({ learnOutcomes: [declined(), []] });
    const memory = previousFailure ? testMemory(backend) : null;
    const first = await run(flow, { text: "hello" }, { learnable: true, log: path, memory });
    if (previousFailure) expect((await first.backward("First review.")).status).toBe("failed");
    const [previous, previousLearning] = [first.backwardOutcome, first.learning];
    const second = await resume(flow, { log: path, memory });
    expect((await second.backward("New review.")).status).toBe("succeeded");
    const prefix = readFileSync(path);
    const [learned, committed] = [backend.learnContexts.length, backend.commitCalls.length];

    expect(await rejected(first.backward([null]))).toBeInstanceOf(RangeError);

    expect(first.backwardOutcome).toBe(previous);
    expect(first.learning).toBe(previousLearning);
    expect(readFileSync(path)).toEqual(prefix);
    expect([backend.learnContexts.length, backend.commitCalls.length]).toEqual([learned, committed]);
  });
}

test("stale handle publishes its new accepted prefix on host failure", async () => {
  const path = join(dir, "accepted.jsonl");
  const flow = await loadFlow(WITH_MEMORY);
  const failure = new Error("learner allocation failed");
  const backend = new FakeMemory({ learnOutcomes: [declined(), failure] });
  const memory = testMemory(backend);
  const first = await run(flow, { text: "hello" }, { learnable: true, log: path, memory });
  const second = await resume(flow, { log: path, memory });
  expect((await second.backward("First review.")).status).toBe("failed");
  expect(first.backwardOutcome).toBeNull();
  expect(first.learning).toBeNull();

  expect(await rejected(first.backward("New review."))).toBe(failure);

  expect(first.backwardOutcome!.failure!.code).toBe("backward.unfinished");
  expect(first.learning!.status).toBe("incomplete");
  const last = records(path).at(-1);
  expect(last).toBeInstanceOf(BackwardStartedRecord);
  expect((last as BackwardStartedRecord).attemptId).toBe(2);
  const inspected = await resume(flow, { log: path });
  expect(first.backwardOutcome).toEqual(inspected.backwardOutcome);
  expect(first.learning).toEqual(inspected.learning);
});

for (const otherFinishes of [false, true]) {
  test(`resumed interrupted observation keeps identity on invalid feedback (${otherFinishes})`, async () => {
    const path = join(dir, "interrupted.jsonl");
    const flow = await loadFlow(WITH_MEMORY);
    const interrupted = new Error("interrupted learner");
    const backend = new FakeMemory({ learnOutcomes: [interrupted, []] });
    const memory = testMemory(backend);
    const original = await run(flow, { text: "hello" }, { learnable: true, log: path, memory });
    expect(await rejected(original.backward("First review."))).toBe(interrupted);
    const resumed = await resume(flow, { log: path, memory });
    const [previous, previousLearning] = [resumed.backwardOutcome, resumed.learning];
    expect(previous!.failure!.code).toBe("backward.unfinished");
    expect(previousLearning!.status).toBe("incomplete");
    if (otherFinishes) {
      const other = await resume(flow, { log: path, memory });
      expect((await other.backward("New review.")).status).toBe("succeeded");
    }
    const prefix = readFileSync(path);
    const [learned, committed] = [backend.learnContexts.length, backend.commitCalls.length];

    expect(await rejected(resumed.backward([null]))).toBeInstanceOf(RangeError);

    expect(resumed.backwardOutcome).toBe(previous);
    expect(resumed.learning).toBe(previousLearning);
    expect(readFileSync(path)).toEqual(prefix);
    expect([backend.learnContexts.length, backend.commitCalls.length]).toEqual([learned, committed]);
  });
}

test("backward while another writer holds the log is RunBusyError and appends nothing", async () => {
  const path = join(dir, "held.jsonl");
  const result = await run(await loadFlow(IDENTITY), { text: "hello" }, { learnable: true, log: path });
  const before = readFileSync(path);
  const holder = lockFile(`${realpathSync(path)}.lock`);
  try {
    const error = (await rejected(result.backward("Too long."))) as RunBusyError;
    expect(error).toBeInstanceOf(RunBusyError);
    expect([error.log, error.cause instanceof FileLockError]).toEqual([path, true]);
  } finally {
    holder.release();
  }
  expect(readFileSync(path).equals(before)).toBe(true);
  expect(result.backwardOutcome).toBeNull();
  expect((await result.backward("Too long.")).status).toBe("succeeded");
});

test("nonretained result rejects backward", async () => {
  const result = await run(await loadFlow(IDENTITY), { text: "hello" });

  const error = await rejected(result.backward("Review."));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe("backward requires a run retained with learnable: true");
  expect(result.learning).toBeNull();
  expect(result.backwardOutcome).toBeNull();
});

test("terminal resume only reconstructs backward outcome", async () => {
  const path = join(dir, "run.jsonl");
  const flow = await loadFlow(WITH_MEMORY);
  const memory = new FakeMemory();
  const result = await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(memory) });
  let before = records(path);
  let inspected = await resume(flow, { log: path });
  expect(inspected.backwardOutcome).toBeNull();
  expect(inspected.learning).toBeNull();
  expect(records(path)).toEqual(before);
  expect(memory.openSnapshots).toEqual([null]);

  const outcome = await result.backward("Review.");
  before = records(path);
  inspected = await resume(flow, { log: path, memory: testMemory({}) });

  expect(inspected.backwardOutcome).toEqual(outcome);
  expect(inspected.learning).toEqual(outcome.learning);
  expect(inspected.events).toEqual([]);
  expect(inspected.output).toBe("hello");
  expect(records(path)).toEqual(before);
  expect(memory.openSnapshots).toEqual([null, { namespace: "test" }]);
});

for (const prefix of ["started", "ready", "requested", "confirmed"]) {
  test(`terminal resume inspects every unfinished backward prefix (${prefix})`, async () => {
    const path = join(dir, "prefix.jsonl");
    const flow = await loadFlow(WITH_MEMORY);
    const memory = new FakeMemory();
    await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(memory) });
    const log = JsonlRunLog.resume(path);
    let expected: unknown;
    try {
      const state = replayRunLog(flow, log);
      const [root, attributed] = await computeBackward(state.trace!, "Review.");
      commitPrepared(log, state, prepareRecord(state, new BackwardStartedRecord(1, "Review.")));
      if (prefix !== "started") {
        const proposals = state.memories!.map((spec) => new MemoryProposal(spec.key, []));
        commitPrepared(log, state, prepareRecord(state, new BackwardReadyRecord(1, root, attributed, proposals)));
      }
      if (prefix === "requested" || prefix === "confirmed") {
        const identity = memoryCommitId(state.runId!, "test", [], 1);
        commitPrepared(log, state, prepareRecord(state, new MemoryCommitRequestedRecord("test", identity, [], 1)));
        if (prefix === "confirmed") {
          commitPrepared(log, state, prepareRecord(state, new MemoryCommitSucceededRecord("test", identity, 1)));
        }
      }
      expected = state.backwardState.outcome;
    } finally {
      log.close();
    }
    const before = readFileSync(path);

    const inspected = await resume(flow, { log: path });

    expect(inspected.backwardOutcome).toEqual(expected as never);
    expect(inspected.backwardOutcome!.failure!.code).toBe("backward.unfinished");
    expect(inspected.learning!.status).toBe("incomplete");
    expect(inspected.learning!.policies[0]!.status).toBe(
      ({ started: "unrequested", ready: "unrequested", requested: "unknown", confirmed: "succeeded" } as const)[
        prefix as "started"
      ],
    );
    expect(inspected.events).toEqual([]);
    expect(readFileSync(path)).toEqual(before);
    expect(memory.openSnapshots).toEqual([null]);
    expect(memory.learnContexts).toEqual([]);
    expect(memory.commitCalls).toEqual([]);
  });
}

for (const phase of ["learn", "commit"]) {
  test(
    `killed attempt does not block a later attempt (${phase})`,
    async () => {
      const path = join(dir, "run.jsonl");
      const marker = join(dir, "blocked");
      const flow = await loadFlow(WITH_MEMORY);
      const first = new FakeMemory();
      const firstResult = await run(flow, { text: "hello" }, { learnable: true, log: path, memory: testMemory(first) });
      expect((await firstResult.backward("First review.")).status).toBe("succeeded");

      const script = join(dir, "child.mjs");
      const url = (relative: string) => JSON.stringify(pathToFileURL(join(TS_ROOT, relative)).href);
      writeFileSync(
        script,
        `const { writeFileSync } = await import("node:fs");
const { loadFlow } = await import(${url("src/compose/loader.ts")});
const { advance } = await import(${url("src/run.ts")});
const { FakeMemory, FakeSession } = await import(${url("tests/helpers/memory_fakes.ts")});
const auth = ${JSON.stringify(join(dir, "auth.json"))};
const block = () => {
  writeFileSync(${JSON.stringify(marker)}, ${JSON.stringify(phase)});
  return new Promise(() => {});
};
const backend = new FakeMemory(${phase === "learn" ? "{ learnOutcomes: [block] }" : "{}"});
${phase === "commit" ? "FakeSession.prototype.commit = block;" : ""}
const flow = await loadFlow(${JSON.stringify(WITH_MEMORY)});
const env = (name) => (name === "YIAN_AUTH_FILE" ? auth : undefined);
const resumed = await advance({ log: ${JSON.stringify(path)}, memory: new Map([["test", backend]]), env });
await resumed.backward("Second review.");
`,
      );
      const child: ChildProcess = spawn(process.execPath, [script], { cwd: TS_ROOT, stdio: ["ignore", "pipe", "pipe"] });
      let output = "";
      child.stdout!.setEncoding("utf8").on("data", (chunk: string) => (output += chunk));
      child.stderr!.setEncoding("utf8").on("data", (chunk: string) => (output += chunk));
      const closed = new Promise((resolve) => child.on("close", resolve));
      try {
        const deadline = Date.now() + 15_000;
        while (!existsSync(marker)) {
          if (child.exitCode !== null) throw new Error(`the child exited before blocking:\n${output}`);
          if (Date.now() > deadline) throw new Error(`the child never reached the blocking call:\n${output}`);
          await new Promise((resolve) => setTimeout(resolve, 20));
        }
      } finally {
        child.kill("SIGKILL");
        await closed;
      }

      const third = new FakeMemory();
      const resumed = await resume(flow, { log: path, memory: testMemory(third) });
      const projected = resumed.backwardOutcome!;
      expect(projected.status).toBe("failed");
      expect(projected.failure!.code).toBe("backward.unfinished");
      const outcome = await resumed.backward("Third review.");
      expect(outcome.status).toBe("succeeded");
      expect(resumed.backwardOutcome).toBe(outcome);
      const all = records(path);
      expect(startedAttempts(path)).toEqual([1, 2, 3]);
      const finished = new Map(
        all.flatMap((record) => (record instanceof BackwardFinishedRecord ? [[record.attemptId, record.outcome]] : [])),
      );
      expect(finished.get(1)!.status).toBe("succeeded");
      expect(finished.get(3)!.status).toBe("succeeded");
      expect(finished.get(2)).toEqual(projected);
      expect(finished.get(2)!.learning!.policies[0]!.status).toBe(phase === "commit" ? "unknown" : "unrequested");
      const requests = all.filter((record) => record instanceof MemoryCommitRequestedRecord) as MemoryCommitRequestedRecord[];
      expect(requests.map((request) => request.attemptId)).toEqual(phase === "commit" ? [1, 2, 3] : [1, 3]);
      expect(new Set(requests.map((request) => request.commitId)).size).toBe(requests.length);
      expect(third.commitCalls.map(([commitId]) => commitId)).toEqual([requests.at(-1)!.commitId]);
    },
    30_000,
  );
}

test("a later backward replays in the run's env", async () => {
  const path = join(dir, "env.jsonl");
  const env = hostEnv(dir, { YIAN_MAX_VALUE_NESTING: "50" });
  const flow = await loadFlow(IDENTITY, { env });
  const result = await run(flow, { text: "hello" }, { learnable: true, log: path, env });

  expect((await result.backward("Review.")).status).toBe("succeeded");
  // The feedback is bounded by the run's limit too, before anything is appended.
  let deep: unknown = "too deep";
  for (let level = 0; level < 51; level++) deep = [deep];
  const before = readFileSync(path);
  const refused = await rejected(result.backward(deep));
  expect(refused).toBeInstanceOf(TypeCheckError);
  expect((refused as Error).message).toContain("50");
  expect(readFileSync(path)).toEqual(before);

  // The value is recorded: a resume in another env is refused, one in the run's env replays.
  expect(await rejected(resume(flow, { log: path }))).toBeInstanceOf(DefaultsMismatchError);
  const resumed = await resume(flow, { log: path, env });
  expect((await resumed.backward("Again.")).status).toBe("succeeded");
  expect(startedAttempts(path)).toEqual([1, 2]);
});

test("result.backward publishes an AGENT backward model retry to onEvent", async () => {
  const flow = await loadFlow(
    "id: agent\nname: agent\nllm_config: {provider: openai, model: test-model}\ninput:\n  text: !str\n" +
      "nodes:\n  agent:\n    kind: agent\n    prompt: ${text}\n    input:\n      text: ${input.text}\n" +
      "    output: str\noutput: ${agent.output}\n",
  );
  const chat = new ScriptedModel([
    textReply("Answer."),
    new ProviderError("server", "busy", { retryAfter: 0.001 }),
    backwardReply("Be brief."),
  ]);
  // The run's env allows one retry, and a later backward replays in it.
  const env = hostEnv(dir, { YIAN_MODEL_RETRIES: "1" });
  const result = await run(flow, { text: "hello" }, { learnable: true, models: serving(chat), env });
  const seen: unknown[] = [];
  const outcome = await result.backward("Review.", { onEvent: (event) => void seen.push(event) });

  expect(outcome.status).toBe("succeeded");
  expect(chat.requests).toHaveLength(3);
  expect(seen).toHaveLength(1);
  const event = seen[0] as ModelCallRetried;
  expect(event).toBeInstanceOf(ModelCallRetried);
  expect([event.nodeId, event.consumer, event.attempt, event.error.kind]).toEqual(["agent", "backward", 1, "server"]);
});
