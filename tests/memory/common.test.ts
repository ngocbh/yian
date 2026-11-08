/**
 * What the memory types share: this process's queue on a memory's lock, the calls a session's
 * `close` waits for, and taking a memory's location out of an error.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { MemoryBusyError } from "../../src/memory/base.ts";
import { locked, type MemoryLock, PendingCalls, withoutLocation } from "../../src/memory/common.ts";
import { realScheduler } from "../../src/platform.ts";

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-mc-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

/** A lock on the memory at `dir`, with `timeoutMs`; every lock a test makes shares one queue. */
function lockOf(timeoutMs: number): MemoryLock {
  return { key: dir, path: join(dir, "lock"), timeoutMs, scheduler: realScheduler, check: () => {} };
}

/** A promise and the function that resolves it. */
function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

/** Return what `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("calls on one memory run one at a time, in the order they were made", async () => {
  const ran: string[] = [];
  const first = deferred();
  const calls = [
    locked(lockOf(5_000), undefined, async () => {
      ran.push("a start");
      await first.promise;
      ran.push("a end");
    }),
    locked(lockOf(5_000), undefined, () => ran.push("b")),
    locked(lockOf(5_000), undefined, () => ran.push("c")),
  ];
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(ran).toEqual(["a start"]);
  first.resolve();
  await Promise.all(calls);
  expect(ran).toEqual(["a start", "a end", "b", "c"]);
});

test("an abort while queued rejects at once; the body never runs, and the calls after it do", async () => {
  const ran: string[] = [];
  const ahead = deferred();
  const first = locked(lockOf(5_000), undefined, () => ahead.promise);
  const stop = new AbortController();
  const aborted = locked(lockOf(5_000), stop.signal, () => ran.push("aborted"));
  const after = locked(lockOf(5_000), undefined, () => ran.push("after"));
  stop.abort(new Error("stopped"));
  expect(((await rejection(aborted)) as Error).message).toBe("stopped");
  ahead.resolve();
  await first;
  await after;
  expect(ran).toEqual(["after"]);
});

test("a queued call is busy at its deadline while the turn ahead still runs, and its body never runs", async () => {
  const ran: string[] = [];
  const ahead = deferred();
  const first = locked(lockOf(10_000), undefined, () => ahead.promise);
  const started = performance.now();
  const late = locked(lockOf(100), undefined, () => ran.push("late"));
  expect(await rejection(late)).toBeInstanceOf(MemoryBusyError);
  expect(performance.now() - started).toBeLessThan(2_000);
  ahead.resolve();
  await first;
  // Its turn has come and gone: the calls queued after it ran, and it didn't.
  await locked(lockOf(5_000), undefined, () => ran.push("next"));
  expect(ran).toEqual(["next"]);
});

test("PendingCalls settles once every call tracked has, a call tracked while it waits included", async () => {
  const calls = new PendingCalls();
  const first = deferred();
  const second = deferred();
  calls.track(first.promise);
  let settled = false;
  const closing = calls.settled().then(() => {
    settled = true;
  });
  // A call that rejects counts as settled too.
  calls.track(Promise.reject(new Error("failed"))).catch(() => {});
  // Tracked after settled() began waiting, as a change's second turn is.
  calls.track(second.promise);
  first.resolve();
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(settled).toBe(false);
  second.resolve();
  await closing;
  expect(settled).toBe(true);
});

test("PendingCalls doesn't wait for the turn of a call that left the queue, which never runs", async () => {
  const calls = new PendingCalls();
  const ahead = deferred();
  // Another instance's turn, which the session doesn't track.
  const first = locked(lockOf(10_000), undefined, () => ahead.promise);
  const busy = calls.track(locked(lockOf(50), undefined, () => "ran"));
  expect(await rejection(busy)).toBeInstanceOf(MemoryBusyError);
  const closed = await Promise.race([
    calls.settled().then(() => true),
    new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 1_000)),
  ]);
  expect(closed).toBe(true);
  ahead.resolve();
  await first;
});

test("withoutLocation replaces each text in order, so the longer path goes before its folder", () => {
  const error = Object.assign(new Error("EACCES: permission denied, open '/srv/m/cases.db' in /srv/m"), {
    code: "EACCES",
    errno: -13,
    syscall: "open",
    path: "/srv/m/cases.db",
  });
  const replacements: [string, string][] = [
    ["/srv/m/cases.db", "the memory's store"],
    ["/srv/m", "the memory's folder"],
  ];
  const scrubbed = withoutLocation(error, replacements) as NodeJS.ErrnoException;
  expect(scrubbed.message).toBe("EACCES: permission denied, open 'the memory's store' in the memory's folder");
  expect([scrubbed.name, scrubbed.code, scrubbed.errno, scrubbed.syscall]).toEqual(["Error", "EACCES", -13, "open"]);
  // The fields that hold the location aren't copied.
  expect(scrubbed.path).toBeUndefined();
  // An error that names no location is returned as it is.
  const other = new Error("nothing here");
  expect(withoutLocation(other, replacements)).toBe(other);
});
