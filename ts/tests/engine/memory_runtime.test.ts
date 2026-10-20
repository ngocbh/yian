/**
 * A port of `tests/engine/test_memory_runtime.py`.
 *
 * Ported: `test_open_validates_the_memory_table_and_its_origins`,
 * `test_open_uses_key_order_and_captures_snapshots`,
 * `test_open_restores_exact_recorded_snapshot_per_key`,
 * `test_restore_rejects_noncanonical_snapshot_order_before_opening`,
 * `test_restore_rejects_a_session_that_changes_the_recorded_snapshot`,
 * `test_registry_rejects_aliases_and_attributes_missing_key_origin`,
 * `test_later_open_failure_closes_prior_session_and_preserves_origin` and
 * `test_frame_era_session_is_closed_before_setup_failure`, with the `_Session`, `_Memory`,
 * `_table`, `_origins` and `_open` helpers.
 *
 * Changed: Python's tuples are arrays and its dictionaries are `Map`s; each refusal pins its class
 * and full message (Python's `ValueError` is a `RangeError`, matched in part there); every origin
 * checked, the missing key's included, is the very `OriginRef` passed in (`toBe`); the restore
 * test also checks the captured snapshots; `open` and `close` return promises; a wrapped setup
 * failure's `cause` is the backend's error (Python's `raise ... from`).
 *
 * TS-only: a plain-object table item; origins as a plain object, with a numeric key, or with a key
 * emptied after construction (`OriginRef`'s own `RangeError`); each registry name message (a
 * numeric key, `""`, a lone surrogate); a value without the `Memory` methods; a registry that
 * isn't a `Map` (a plain object, `null`, an array); each snapshot message (a plain object, an item
 * that isn't an envelope, a repeated key, and a repeated key before a non-envelope item); every
 * refusal opens nothing; async `openRun` and `close` are awaited; a snapshot getter throwing a
 * `MemoryOperationError`; a snapshot that is a `Promise` or holds a `Map`; a session lacking
 * `close` isn't closed; close failures without a primary (the first is thrown, later ones noted on
 * it), with an error primary (all noted on it) and with a primitive primary (notes lost); a
 * concurrent or later `close()` waits for the first, never throws and adds no notes; a reentrant
 * close from inside a session's close doesn't close again; the caller's array and `Map`s edited
 * during `open` change nothing; a subclass's `open` returns the subclass.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  type JsonObject,
  type LearnContext,
  type Memory,
  type MemoryContent,
  type MemoryMutation,
  type MemorySession,
  MemoryOperationError,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  type RecallContext,
} from "../../src/memory/contracts.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";

type Events = unknown[][];
type ErrorClass = new (...args: never[]) => Error;

/** One observable fake session for one memory key. */
class FakeSession implements MemorySession {
  readonly name: string;
  readonly events: Events;
  value: string | MemoryContent | Error;
  learnValue: readonly MemoryMutation[] | Error | null;
  commitValue: Error | null;
  storedSnapshot: JsonObject;
  closeError: Error | null;
  readonly calls: RecallContext[] = [];
  readonly learnCalls: LearnContext[] = [];

  constructor(
    name: string,
    events: Events,
    options: {
      value?: string | MemoryContent | Error;
      learnValue?: readonly MemoryMutation[] | Error | null;
      commitValue?: Error | null;
      snapshot?: JsonObject | null;
      closeError?: Error | null;
    } = {},
  ) {
    this.name = name;
    this.events = events;
    this.value = options.value ?? "";
    this.learnValue = options.learnValue ?? null;
    this.commitValue = options.commitValue ?? null;
    this.storedSnapshot = options.snapshot ?? { policy: name };
    this.closeError = options.closeError ?? null;
  }

  get snapshot(): JsonObject {
    this.events.push(["snapshot", this.name]);
    return this.storedSnapshot;
  }

  recall(context?: RecallContext | null): string | MemoryContent {
    expect(context).toBeTruthy();
    this.calls.push(context!);
    if (this.value instanceof Error) throw this.value;
    return this.value;
  }

  learn(context: LearnContext): readonly MemoryMutation[] {
    this.learnCalls.push(context);
    if (this.learnValue instanceof Error) throw this.learnValue;
    return this.learnValue ?? [];
  }

  commit(commitId: string, mutations: readonly MemoryMutation[]): void {
    this.events.push(["commit", this.name, commitId, mutations]);
    if (this.commitValue !== null) throw this.commitValue;
  }

  close(): void {
    this.events.push(["close", this.name]);
    if (this.closeError !== null) throw this.closeError;
  }
}

/** Tiny `Memory` implementation returning one configured session. */
class FakeMemory implements Memory {
  readonly name: string;
  readonly events: Events;
  readonly openError: Error | null;
  readonly restoreSnapshot: boolean;
  readonly session: FakeSession;

  constructor(
    name: string,
    events: Events,
    options: {
      value?: string | MemoryContent | Error;
      learnValue?: readonly MemoryMutation[] | Error | null;
      commitValue?: Error | null;
      sessionSnapshot?: JsonObject | null;
      openError?: Error | null;
      closeError?: Error | null;
      restoreSnapshot?: boolean;
    } = {},
  ) {
    this.name = name;
    this.events = events;
    this.openError = options.openError ?? null;
    this.restoreSnapshot = options.restoreSnapshot ?? true;
    this.session = new FakeSession(name, events, {
      value: options.value,
      learnValue: options.learnValue,
      commitValue: options.commitValue,
      snapshot: options.sessionSnapshot,
      closeError: options.closeError,
    });
  }

  openRun(options: { snapshot?: JsonObject | null } = {}): FakeSession {
    const snapshot = options.snapshot ?? null;
    this.events.push(["open_run", this.name, snapshot]);
    if (this.openError !== null) throw this.openError;
    if (snapshot !== null && this.restoreSnapshot) this.session.storedSnapshot = snapshot;
    return this.session;
  }

  reset(): void {
    this.events.push(["reset", this.name]);
  }
}

/** Return the recorded table: AGENT-owned alpha and flow-owned beta. */
function table(): MemorySpec[] {
  return [
    new MemorySpec("alpha", "alpha", {}, [new DeclSite([], ["root"], "agent")], true),
    new MemorySpec("beta", "beta", { limit: 2n }, [new DeclSite([], ["root"])], false),
  ];
}

/** Return one authored origin per key. */
function origins(): Map<string, OriginRef | null> {
  return new Map([
    ["alpha", new OriginRef("local-site")],
    ["beta", new OriginRef("outer-site")],
  ]);
}

/** Open the two-key test table over `memory`. */
function open(
  memory: ReadonlyMap<string, Memory>,
  options: { origins?: ReadonlyMap<string, OriginRef | null>; snapshots?: readonly MemorySnapshotEnvelope[] | null } = {},
): Promise<MemoryRuntime> {
  return MemoryRuntime.open({
    memories: table(),
    origins: options.origins ?? origins(),
    memory,
    snapshots: options.snapshots,
  });
}

/** Return the two fake memories under their own keys, logging into `events`. */
function pair(events: Events): Map<string, Memory> {
  return new Map([
    ["alpha", new FakeMemory("alpha", events)],
    ["beta", new FakeMemory("beta", events)],
  ]);
}

/** Assert `run` rejects with an error of exactly class `cls` and exactly `message`. */
async function expectError(run: () => Promise<unknown>, cls: ErrorClass, message: string): Promise<unknown> {
  let caught: unknown;
  try {
    await run();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught;
}

test("open validates the memory table and its origins", async () => {
  const events: Events = [];
  const memory = pair(events);
  const [alpha, beta] = table() as [MemorySpec, MemorySpec];
  await expectError(
    () => MemoryRuntime.open({ memories: new Set(table()) as never, origins: origins(), memory }),
    TypeError,
    "memories must be an array of MemorySpec values",
  );
  await expectError(
    () => MemoryRuntime.open({ memories: [beta, alpha], origins: origins(), memory }),
    RangeError,
    "memories must be sorted by key and unique",
  );
  await expectError(
    () => MemoryRuntime.open({ memories: [alpha, alpha], origins: new Map([["alpha", null]]), memory }),
    RangeError,
    "memories must be sorted by key and unique",
  );
  await expectError(
    () => MemoryRuntime.open({ memories: table(), origins: new Map([["alpha", null]]), memory }),
    RangeError,
    "memory origins must exactly match the memory keys",
  );
  await expectError(
    () =>
      MemoryRuntime.open({
        memories: table(),
        origins: new Map<string, unknown>([["alpha", "site"], ["beta", null]]) as never,
        memory,
      }),
    TypeError,
    "memory origins must be OriginRef values or null",
  );
  expect(events).toEqual([]);
  const empty = await MemoryRuntime.open({ memories: [], origins: new Map(), memory: new Map() });
  expect(empty.memories).toEqual([]);
  expect(empty.snapshots).toEqual([]);
  await empty.close();
});

test("open uses key order and captures snapshots", async () => {
  const events: Events = [];
  const runtime = await open(
    new Map([
      ["beta", new FakeMemory("beta", events)],
      ["alpha", new FakeMemory("alpha", events)],
    ]),
  );

  expect(events).toEqual([
    ["open_run", "alpha", null],
    ["open_run", "beta", null],
    ["snapshot", "alpha"],
    ["snapshot", "beta"],
  ]);
  expect(runtime.memories).toEqual(table());
  expect(runtime.snapshots).toEqual([
    new MemorySnapshotEnvelope("alpha", { policy: "alpha" }),
    new MemorySnapshotEnvelope("beta", { policy: "beta" }),
  ]);

  expect(runtime.closed).toBe(false);
  await runtime.close();
  expect(runtime.closed).toBe(true);
  await runtime.close();
  expect(events.slice(4)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("restore rejects noncanonical snapshot order before opening", async () => {
  const events: Events = [];
  const snapshots = [
    new MemorySnapshotEnvelope("beta", { revision: 3n }),
    new MemorySnapshotEnvelope("alpha", { revision: 2n }),
  ];
  await expectError(
    () => open(pair(events), { snapshots }),
    RangeError,
    "memory snapshots must exactly match the memory keys in canonical order",
  );
  expect(events).toEqual([]);
});

test("open restores exact recorded snapshot per key", async () => {
  const events: Events = [];
  const snapshots = [
    new MemorySnapshotEnvelope("alpha", { revision: 2n }),
    new MemorySnapshotEnvelope("beta", { revision: 3n }),
  ];

  const runtime = await open(pair(events), { snapshots });

  expect(events.slice(0, 2)).toEqual([
    ["open_run", "alpha", { revision: 2n }],
    ["open_run", "beta", { revision: 3n }],
  ]);
  expect(runtime.snapshots).toEqual(snapshots);
  await runtime.close();
});

test("restore rejects a session that changes the recorded snapshot", async () => {
  const events: Events = [];
  const snapshots = [
    new MemorySnapshotEnvelope("alpha", { revision: 2n }),
    new MemorySnapshotEnvelope("beta", { revision: 3n }),
  ];
  const authored = origins();

  const error = await expectError(
    () =>
      open(
        new Map([
          ["alpha", new FakeMemory("alpha", events)],
          ["beta", new FakeMemory("beta", events, { restoreSnapshot: false })],
        ]),
        { origins: authored, snapshots },
      ),
    MemorySetupError,
    "restored memory session did not reproduce its recorded snapshot",
  );

  expect((error as MemorySetupError).origin).toBe(authored.get("beta")!);
  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("registry rejects aliases and attributes missing key origin", async () => {
  const events: Events = [];
  const shared = new FakeMemory("shared", events);
  await expectError(
    () => open(new Map([["alpha", shared], ["beta", shared]])),
    RangeError,
    "one memory object cannot be registered under multiple names: 'alpha' and 'beta'",
  );

  const authored = origins();
  const error = await expectError(
    () => open(new Map([["alpha", new FakeMemory("alpha", events)]]), { origins: authored }),
    MemorySetupError,
    "memory 'beta' was not supplied by the host",
  );
  expect((error as MemorySetupError).origin).toBe(authored.get("beta")!);
  expect(events).toEqual([]);
});

test("a plain-object table item is not a MemorySpec", async () => {
  const events: Events = [];
  const [alpha, beta] = table() as [MemorySpec, MemorySpec];
  await expectError(
    () => MemoryRuntime.open({ memories: [{ ...alpha }, beta] as never, origins: origins(), memory: pair(events) }),
    TypeError,
    "memories must be an array of MemorySpec values",
  );
  expect(events).toEqual([]);
});

test("origins must be a Map of string keys to valid origins", async () => {
  const events: Events = [];
  const memory = pair(events);
  await expectError(
    () =>
      MemoryRuntime.open({
        memories: table(),
        origins: { alpha: null, beta: null } as never,
        memory,
      }),
    TypeError,
    "memory origins must be a Map",
  );
  await expectError(
    () =>
      MemoryRuntime.open({
        memories: table(),
        origins: new Map<unknown, null>([["alpha", null], [1, null], ["beta", null]]) as never,
        memory,
      }),
    TypeError,
    "memory origin keys must be strings",
  );
  const emptied = new OriginRef("site");
  (emptied as { key: string }).key = "";
  await expectError(
    () => open(memory, { origins: new Map([["alpha", null], ["beta", emptied]]) }),
    RangeError,
    "origin key must be a non-empty string",
  );
  expect(events).toEqual([]);
});

test("registry names must be non-empty well-formed strings", async () => {
  const events: Events = [];
  const backend = new FakeMemory("alpha", events);
  for (const [name, cls, message] of [
    [1, TypeError, "memory name must be a string"],
    ["", RangeError, "memory name must be non-empty"],
    ["\ud800", RangeError, "memory name must be valid UTF-8"],
  ] as const) {
    await expectError(() => open(new Map<unknown, Memory>([[name, backend]]) as never), cls, message);
  }
  expect(events).toEqual([]);
});

test("registry values must implement the Memory protocol", async () => {
  const events: Events = [];
  await expectError(
    () => open(new Map<string, unknown>([["alpha", { openRun() {} }]]) as never),
    TypeError,
    "memory 'alpha' must implement the Memory protocol",
  );
  expect(events).toEqual([]);
});

test("registry must be a Map", async () => {
  const events: Events = [];
  for (const memory of [{ alpha: new FakeMemory("alpha", events) }, null, []]) {
    await expectError(() => open(memory as never), TypeError, "memory must be a Map of memory keys to backends");
  }
  expect(events).toEqual([]);
});

test("snapshots must be an array of unique envelopes", async () => {
  const events: Events = [];
  const envA = new MemorySnapshotEnvelope("alpha", { revision: 2n });
  for (const [snapshots, cls, message] of [
    [{ alpha: envA }, TypeError, "memory snapshots must be an array or null"],
    [[{ memory: "alpha", snapshot: {} }], TypeError, "memory snapshots must contain MemorySnapshotEnvelope values"],
    [[envA, envA], RangeError, "memory snapshot keys must be unique"],
    // Each item is checked in turn, so the repeated key is found before the bad item.
    [[envA, envA, "x"], RangeError, "memory snapshot keys must be unique"],
  ] as const) {
    await expectError(() => open(pair(events), { snapshots: snapshots as never }), cls, message);
  }
  expect(events).toEqual([]);
});

test("later open failure closes prior session and preserves origin", async () => {
  const events: Events = [];
  const failure = new MemoryOperationError("open_run", "beta.unavailable", "temporarily unavailable", true);
  const authored = origins();

  const error = await expectError(
    () =>
      open(
        new Map([
          ["alpha", new FakeMemory("alpha", events)],
          ["beta", new FakeMemory("beta", events, { openError: failure })],
        ]),
        { origins: authored },
      ),
    MemorySetupError,
    "memory 'beta' could not open its run session: temporarily unavailable",
  );

  expect((error as MemorySetupError).origin).toBe(authored.get("beta")!);
  expect((error as Error).cause).toBe(failure);
  expect(events).toEqual([
    ["open_run", "alpha", null],
    ["open_run", "beta", null],
    ["close", "alpha"],
  ]);
});

test("frame-era session is closed before setup failure", async () => {
  const events: Events = [];
  // A session with the removed frame surface and no recall or learn.
  const frameSession = {
    snapshot: {},
    openFrame(spec: unknown): never {
      throw new Error(String(spec));
    },
    commit(commitId: string): never {
      throw new Error(commitId);
    },
    close(): void {
      events.push(["close", "alpha-invalid"]);
    },
  };
  const frameMemory = {
    openRun(options: { snapshot?: JsonObject | null } = {}) {
      events.push(["open_run", "alpha-invalid", options.snapshot ?? null]);
      return frameSession;
    },
    reset(): void {},
  };
  const authored = origins();

  const error = await expectError(
    () =>
      open(new Map<string, unknown>([["alpha", frameMemory], ["beta", new FakeMemory("beta", events)]]) as never, {
        origins: authored,
      }),
    MemorySetupError,
    "memory 'alpha' returned an invalid run session",
  );

  expect((error as MemorySetupError).origin).toBe(authored.get("alpha")!);
  expect(events).toEqual([
    ["open_run", "alpha-invalid", null],
    ["close", "alpha-invalid"],
  ]);
});

/** A memory whose `openRun` logs `open_run` and returns `session`. */
function memoryOf(name: string, events: Events, session: unknown): Memory {
  return {
    openRun() {
      events.push(["open_run", name, null]);
      return session as MemorySession;
    },
    reset() {},
  };
}

/** A fake session whose `close` logs `closed` only after a timer tick. */
class AsyncCloseSession extends FakeSession {
  override async close(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 0));
    this.events.push(["closed", this.name]);
    if (this.closeError !== null) throw this.closeError;
  }
}

/**
 * Return alpha and beta memories whose sessions close after a timer tick; alpha's close fails with
 * `alphaError`.
 */
function asyncPair(events: Events, alphaError: Error | null = null): Map<string, Memory> {
  return new Map([
    ["alpha", memoryOf("alpha", events, new AsyncCloseSession("alpha", events, { closeError: alphaError }))],
    ["beta", memoryOf("beta", events, new AsyncCloseSession("beta", events))],
  ]);
}

class StorageError extends Error {}

test("async openRun and close are awaited", async () => {
  const events: Events = [];
  const asyncMemory = (name: string): Memory => ({
    async openRun() {
      await new Promise((resolve) => setTimeout(resolve, 0));
      events.push(["opened", name]);
      return new AsyncCloseSession(name, events);
    },
    reset() {},
  });

  const runtime = await open(new Map([["alpha", asyncMemory("alpha")], ["beta", asyncMemory("beta")]]));
  expect(events).toEqual([
    ["opened", "alpha"],
    ["opened", "beta"],
    ["snapshot", "alpha"],
    ["snapshot", "beta"],
  ]);

  await runtime.close();
  expect(events.slice(4)).toEqual([
    ["closed", "beta"],
    ["closed", "alpha"],
  ]);
});

test("a snapshot getter's operation error is a setup failure that closes every session", async () => {
  const events: Events = [];
  const failure = new MemoryOperationError("snapshot", "alpha.export", "export unavailable", false);
  class FailingSnapshot extends FakeSession {
    override get snapshot(): JsonObject {
      throw failure;
    }
  }
  const authored = origins();

  const error = await expectError(
    () =>
      open(
        new Map([
          ["alpha", memoryOf("alpha", events, new FailingSnapshot("alpha", events))],
          ["beta", new FakeMemory("beta", events)],
        ]),
        { origins: authored },
      ),
    MemorySetupError,
    "memory 'alpha' could not export its run snapshot: export unavailable",
  );

  expect((error as MemorySetupError).origin).toBe(authored.get("alpha")!);
  expect((error as Error).cause).toBe(failure);
  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("a snapshot that is a Promise or holds a Map is an invalid run snapshot", async () => {
  for (const snapshot of [Promise.resolve({}), { index: new Map() }]) {
    const events: Events = [];
    const authored = origins();
    const error = await expectError(
      () =>
        open(
          new Map([
            ["alpha", new FakeMemory("alpha", events)],
            ["beta", new FakeMemory("beta", events, { sessionSnapshot: snapshot as never })],
          ]),
          { origins: authored },
        ),
      MemorySetupError,
      "memory 'beta' exported an invalid run snapshot",
    );
    expect((error as MemorySetupError).origin).toBe(authored.get("beta")!);
    expect((error as Error).cause).toBeInstanceOf(TypeError);
    expect(events.slice(-2)).toEqual([
      ["close", "beta"],
      ["close", "alpha"],
    ]);
  }
});

test("a session lacking close is not closed on a later failure", async () => {
  const events: Events = [];
  const noClose = { snapshot: {}, recall() {}, learn() {}, commit() {} };
  const error = await expectError(
    () =>
      open(
        new Map([
          ["alpha", new FakeMemory("alpha", events)],
          ["beta", memoryOf("beta", events, noClose)],
        ]),
      ),
    MemorySetupError,
    "memory 'beta' returned an invalid run session",
  );
  expect((error as MemorySetupError).cause).toBeUndefined();
  expect(Object.hasOwn(error as object, "notes")).toBe(false);
  expect(events).toEqual([
    ["open_run", "alpha", null],
    ["open_run", "beta", null],
    ["close", "alpha"],
  ]);
});

test("without a primary the first close failure is thrown after every close, noting later ones", async () => {
  const events: Events = [];
  const betaError = new Error("network down");
  const runtime = await open(
    new Map([
      ["alpha", new FakeMemory("alpha", events, { closeError: new StorageError("disk full") })],
      ["beta", new FakeMemory("beta", events, { closeError: betaError })],
    ]),
  );

  let caught: unknown;
  await runtime.close().catch((error: unknown) => {
    caught = error;
  });

  expect(caught).toBe(betaError);
  expect((betaError as Error & { notes?: string[] }).notes).toEqual([
    "memory 'alpha' session close also raised StorageError: disk full",
  ]);
  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("closePreserving notes every close failure on the primary and resolves", async () => {
  const events: Events = [];
  const runtime = await open(
    new Map([
      ["alpha", new FakeMemory("alpha", events, { closeError: new StorageError("disk full") })],
      ["beta", new FakeMemory("beta", events, { closeError: new Error("network down") })],
    ]),
  );
  const error = new Error("backward failed");

  await runtime.closePreserving({ error });

  expect((error as Error & { notes?: string[] }).notes).toEqual([
    "memory 'beta' session close also raised Error: network down",
    "memory 'alpha' session close also raised StorageError: disk full",
  ]);
  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("a primitive primary loses the close notes and close resolves", async () => {
  const events: Events = [];
  const runtime = await open(
    new Map([
      ["alpha", new FakeMemory("alpha", events, { closeError: new StorageError("disk full") })],
      ["beta", new FakeMemory("beta", events)],
    ]),
  );

  await runtime.closePreserving({ error: "x" });

  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("a concurrent second close resolves only after the sessions closed", async () => {
  const events: Events = [];
  const runtime = await open(asyncPair(events));

  const first = runtime.close();
  const second = runtime.close();
  await second;

  expect(events.slice(4)).toEqual([
    ["closed", "beta"],
    ["closed", "alpha"],
  ]);
  await first;
});

test("after a rejected close, later closes resolve and add no notes", async () => {
  const events: Events = [];
  const failure = new StorageError("disk full");
  const runtime = await open(asyncPair(events, failure));

  let caught: unknown;
  await runtime.close().catch((error: unknown) => {
    caught = error;
  });
  expect(caught).toBe(failure);

  const error = new Error("backward failed");
  await runtime.close();
  await runtime.closePreserving({ error });

  expect(Object.hasOwn(error, "notes")).toBe(false);
  expect(Object.hasOwn(failure, "notes")).toBe(false);
  expect(events.slice(4)).toEqual([
    ["closed", "beta"],
    ["closed", "alpha"],
  ]);
});

test("a reentrant close from inside a session's close doesn't close again", async () => {
  const events: Events = [];
  let runtime!: MemoryRuntime;
  let closes = 0;
  // Beta closes first, synchronously inside the first close() call.
  const session = {
    snapshot: {},
    recall() {},
    learn() {},
    commit() {},
    close() {
      closes++;
      void runtime.close();
    },
  };
  runtime = await open(
    new Map([
      ["alpha", new FakeMemory("alpha", events)],
      ["beta", memoryOf("beta", events, session)],
    ]),
  );

  await runtime.close();

  expect(closes).toBe(1);
  expect(events.filter((event) => event[0] === "close")).toEqual([["close", "alpha"]]);
});

test("the caller's array and Maps edited during open change nothing", async () => {
  const events: Events = [];
  const memories = table();
  const authored = origins();
  const memory = pair(events);
  const snapshots = [
    new MemorySnapshotEnvelope("alpha", { policy: "alpha" }),
    new MemorySnapshotEnvelope("beta", { policy: "beta" }),
  ];

  const opening = MemoryRuntime.open({ memories, origins: authored, memory, snapshots });
  memories.pop();
  authored.delete("beta");
  memory.delete("beta");
  snapshots.reverse();
  const runtime = await opening;

  expect(runtime.memories).toEqual(table());
  expect(runtime.snapshots).toEqual([
    new MemorySnapshotEnvelope("alpha", { policy: "alpha" }),
    new MemorySnapshotEnvelope("beta", { policy: "beta" }),
  ]);
  expect(Object.isFrozen(runtime.memories)).toBe(true);
  expect(Object.isFrozen(runtime.snapshots)).toBe(true);
  await runtime.close();
  expect(events.slice(-2)).toEqual([
    ["close", "beta"],
    ["close", "alpha"],
  ]);
});

test("a subclass's open returns the subclass", async () => {
  class Sub extends MemoryRuntime {}
  const runtime = await Sub.open({ memories: table(), origins: origins(), memory: pair([]) });
  expect(runtime).toBeInstanceOf(Sub);
  await runtime.close();
});
