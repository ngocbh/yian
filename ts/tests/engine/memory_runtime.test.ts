/**
 * A port of `tests/engine/test_memory_runtime.py`.
 *
 * Ported: `test_open_validates_the_memory_table_and_its_origins` (its empty-table tail opens a
 * runtime, which comes with the rest of `open`),
 * `test_restore_rejects_noncanonical_snapshot_order_before_opening` and
 * `test_registry_rejects_aliases_and_attributes_missing_key_origin`, with the `_Session`,
 * `_Memory`, `_table`, `_origins` and `_open` helpers.
 *
 * Changed: Python's tuples are arrays and its dictionaries are `Map`s; each refusal pins its class
 * and full message (Python's `ValueError` is a `RangeError`, matched in part there); the missing
 * key's origin is the very `OriginRef` passed in (`toBe`).
 *
 * TS-only: a plain-object table item; origins as a plain object, with a numeric key, or with a key
 * emptied after construction (`OriginRef`'s own `RangeError`); each registry name message (a
 * numeric key, `""`, a lone surrogate); a value without the `Memory` methods; a registry that
 * isn't a `Map` (a plain object, `null`, an array); each snapshot message (a plain object, an item
 * that isn't an envelope, a repeated key, and a repeated key before a non-envelope item); every
 * refusal opens nothing.
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
  for (const memory of [{ alpha: new FakeMemory("alpha", []) }, null, []]) {
    await expectError(() => open(memory as never), TypeError, "memory must be a Map of memory keys to backends");
  }
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
