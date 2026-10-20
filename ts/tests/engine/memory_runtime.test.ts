/**
 * A port of `tests/engine/test_memory_runtime.py`.
 *
 * Ported: `test_open_validates_the_memory_table_and_its_origins`,
 * `test_open_uses_key_order_and_captures_snapshots`,
 * `test_open_restores_exact_recorded_snapshot_per_key`,
 * `test_restore_rejects_noncanonical_snapshot_order_before_opening`,
 * `test_restore_rejects_a_session_that_changes_the_recorded_snapshot`,
 * `test_registry_rejects_aliases_and_attributes_missing_key_origin`,
 * `test_later_open_failure_closes_prior_session_and_preserves_origin`,
 * `test_frame_era_session_is_closed_before_setup_failure`,
 * `test_context_recalls_every_key_in_stack_order_and_opens_the_eligible_transcript`,
 * `test_context_validates_scope_request_and_full_opened_coverage`,
 * `test_recall_returns_nonempty_outer_to_local_sections_with_exact_context`,
 * `test_recall_wraps_typed_failure_with_the_key_origin`,
 * `test_recall_wraps_noncanonical_text_without_exposing_it`,
 * `test_context_ignores_uncontributed_session_capabilities`,
 * `test_transcript_admission_requires_invocation_before_opening_source`,
 * `test_stop_during_recall_prevents_transcript_source_acquisition`,
 * `test_context_admits_a_transcript_only_from_an_eligible_memory` and
 * `test_context_rejects_a_second_transcript_for_one_agent`, with the `_Session`, `_Memory`,
 * `_table`, `_origins`, `_open`, `_scope`, `_context` and `_recall_sections` (`recallSections`)
 * helpers.
 *
 * Changed: Python's tuples are arrays and its dictionaries are `Map`s; each refusal pins its class
 * and full message (Python's `ValueError` is a `RangeError`, matched in part there); every origin
 * checked, the missing key's included, is the very `OriginRef` passed in (`toBe`); the restore
 * test also checks the captured snapshots; `open` and `close` return promises; a wrapped setup
 * failure's `cause` is the backend's error (Python's `raise ... from`). `openContext` returns a
 * promise of the context instead of a context manager, and takes `{ signal }` instead of
 * `check_stopped`; a transcript source is a plain object whose `openTurn` returns a fake turn;
 * Python's `str` subclass is a `String` object; the second-transcript test uses a two-part path to
 * pin the `/` join. The stop test has no `_HostStop`: the recall aborts
 * the controller with a reason object, and `openContext` rejects with that very object. The
 * context validation test tries four non-signal options (`undefined`, `{}`, `{ signal: null }`, a
 * fake signal) where Python tries one `check_stopped=None`, and checks nothing was recalled; the
 * recall context is the very `RecallContext` (`toBe`, where Python uses `==`); an invalid-content
 * error's `cause` is a `TypeError` (Python checks no cause).
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
 * during `open` change nothing; a subclass's `open` returns the subclass; `openContext` is a
 * `MemoryContextOpener`; an async `recall` is awaited; `openTurn` receives the same signal and a
 * `ConversationRequest` with the scope's path, the invocation and the compatibility; an `openTurn`
 * rejection rejects `openContext` with it unchanged; a number from `recall` is invalid content; an
 * already-aborted signal with text-only memories still recalls, then rejects with the reason.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  ConversationInvocation,
  ConversationRequest,
  type ConversationTurn,
  DeclSite,
  type JsonObject,
  type LearnContext,
  type Memory,
  MemoryContent,
  MemoryContextRequest,
  type MemoryMutation,
  type MemorySession,
  MemoryOperationError,
  MemoryRecallError,
  MemoryScope,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  RecallContext,
  type TranscriptSource,
} from "../../src/memory/contracts.ts";
import type { MemoryContextOpener } from "../../src/runtime/eval_node.ts";
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

  // The interface's return type, so a test may swap in an async recall.
  recall(context?: RecallContext | null): string | MemoryContent | Promise<string | MemoryContent> {
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

/** Return the AGENT consumer scope: flow key beta outside AGENT key alpha. */
function scope(): MemoryScope {
  return new MemoryScope(["agent"], { memories: ["beta", "alpha"] });
}

/** Return one exact visible AGENT recall context. */
function context(): RecallContext {
  return new RecallContext("base system", "rendered prompt", { question: "why" }, ["agent"]);
}

/** Return options with a signal that never aborts. */
function live(): { signal: AbortSignal } {
  return { signal: new AbortController().signal };
}

/** Return the text-only sections `runtime` admits for `scope` and `recall`. */
async function recallSections(runtime: MemoryRuntime, scope: MemoryScope, recall: RecallContext) {
  const admitted = await runtime.openContext(scope, new MemoryContextRequest(recall, null, {}), live());
  return admitted.sections;
}

/** A transcript source whose `openTurn` records `[request, signal]` in `calls` and returns `turn`. */
function source(turn: unknown, calls: unknown[][]): TranscriptSource {
  return {
    openTurn(request, { signal }) {
      calls.push([request, signal]);
      return turn as ConversationTurn;
    },
  };
}

/** Return the two fake memories, each recalling its own name as text. */
function named(): Map<string, FakeMemory> {
  return new Map([
    ["alpha", new FakeMemory("alpha", [], { value: "alpha" })],
    ["beta", new FakeMemory("beta", [], { value: "beta" })],
  ]);
}

test("context recalls every key in stack order and opens the eligible transcript", async () => {
  const policies = named();
  const runtime = await open(policies);
  const calls: unknown[][] = [];
  const turn = { close() {} };
  policies.get("alpha")!.session.value = new MemoryContent({ text: "alpha", transcript: source(turn, calls) });
  try {
    const rows: [MemoryScope, unknown, string[]][] = [
      [scope(), turn, ["beta", "alpha"]],
      // A sibling sees the flow key, never the first leaf's AGENT key.
      [new MemoryScope(["sibling"], { memories: ["beta"] }), null, ["beta"]],
    ];
    for (const [consumer, transcript, texts] of rows) {
      const request = new MemoryContextRequest(
        new RecallContext("system", "prompt", {}, consumer.staticPath),
        new ConversationInvocation("run", 1),
        {},
      );
      const admitted = await runtime.openContext(consumer, request, live());
      expect(admitted.transcript).toBe(transcript as never);
      expect(admitted.sections.map((section) => section.text)).toEqual(texts);
      expect(admitted.sections.map((section) => section.memory)).toEqual(texts);
    }
    expect(policies.get("beta")!.session.calls).toHaveLength(2);
    expect(policies.get("alpha")!.session.calls).toHaveLength(1);
    expect(calls.map(([request]) => (request as ConversationRequest).consumerStaticPath)).toEqual([["agent"]]);
  } finally {
    await runtime.close();
  }
});

test("context validates scope request and full opened coverage", async () => {
  const alpha = new FakeMemory("alpha", []);
  const runtime = await MemoryRuntime.open({
    memories: table().slice(0, 1),
    origins: new Map([["alpha", new OriginRef("local-site")]]),
    memory: new Map([["alpha", alpha]]),
  });
  const consumer = scope();
  const request = new MemoryContextRequest(
    new RecallContext("system", "prompt", {}, consumer.staticPath),
    new ConversationInvocation("run", 1),
    {},
  );
  try {
    await expectError(
      () => runtime.openContext(null as never, request, live()),
      TypeError,
      "memory context scope must be a MemoryScope",
    );
    await expectError(
      () => runtime.openContext(consumer, null as never, live()),
      TypeError,
      "memory request must be a MemoryContextRequest",
    );
    for (const options of [undefined, {}, { signal: null }, { signal: { aborted: false } }]) {
      await expectError(
        () => runtime.openContext(consumer, request, options as never),
        TypeError,
        "signal must be an AbortSignal",
      );
    }
    const wrong = new MemoryContextRequest(new RecallContext("system", "prompt", {}, ["other"]), request.invocation, {});
    await expectError(
      () => runtime.openContext(consumer, wrong, live()),
      RangeError,
      "memory context request does not match the consumer scope",
    );
    await expectError(
      () => runtime.openContext(consumer, request, live()),
      Error,
      "memory scope names a memory outside the opened table",
    );
    expect(alpha.session.calls).toEqual([]);
  } finally {
    await runtime.close();
  }
  await expectError(() => runtime.openContext(consumer, request, live()), Error, "memory runtime is closed");
});

test("recall returns nonempty outer to local sections with exact context", async () => {
  const events: Events = [];
  const alpha = new FakeMemory("alpha", events, { value: "local lesson" });
  const beta = new FakeMemory("beta", events, { value: "" });
  const runtime = await open(new Map([["alpha", alpha], ["beta", beta]]));
  const recall = context();

  const sections = await recallSections(runtime, scope(), recall);

  // Empty outer recall is omitted but the local key stays explicit.
  expect(sections).toHaveLength(1);
  expect(sections[0]!.memory).toBe("alpha");
  expect(sections[0]!.text).toBe("local lesson");
  expect(beta.session.calls).toHaveLength(1);
  expect(beta.session.calls[0]).toBe(recall);
  expect(alpha.session.calls).toHaveLength(1);
  expect(alpha.session.calls[0]).toBe(recall);
  await runtime.close();
});

test("recall wraps typed failure with the key origin", async () => {
  const events: Events = [];
  const failure = new MemoryOperationError("recall", "beta.search", "index unavailable", true);
  const authored = origins();
  const runtime = await open(
    new Map([
      ["alpha", new FakeMemory("alpha", events)],
      ["beta", new FakeMemory("beta", events, { value: failure })],
    ]),
    { origins: authored },
  );

  const error = await expectError(
    () => recallSections(runtime, scope(), context()),
    MemoryRecallError,
    "memory 'beta' recall failed: index unavailable",
  );

  expect((error as MemoryRecallError).origin).toBe(authored.get("beta")!);
  expect((error as Error).cause).toBe(failure);
  await runtime.close();
});

test("recall wraps noncanonical text without exposing it", async () => {
  for (const value of [new String("secret"), 42]) {
    const events: Events = [];
    const authored = origins();
    const runtime = await open(
      new Map([
        ["alpha", new FakeMemory("alpha", events)],
        ["beta", new FakeMemory("beta", events, { value: value as never })],
      ]),
      { origins: authored },
    );

    const error = await expectError(
      () => recallSections(runtime, scope(), context()),
      MemoryRecallError,
      "memory 'beta' returned invalid recall content",
    );

    expect((error as MemoryRecallError).origin).toBe(authored.get("beta")!);
    expect((error as Error).cause).toBeInstanceOf(TypeError);
    await runtime.close();
  }
});

test("context ignores uncontributed session capabilities", async () => {
  const alpha = new FakeMemory("alpha", [], { value: "local notes" });
  const beta = new FakeMemory("beta", [], { value: "outer notes" });
  const runtime = await open(new Map([["alpha", alpha], ["beta", beta]]));
  const forbidden: unknown[] = [];
  // A session's incidental openTurn is not a recall contribution.
  Object.assign(alpha.session, { openTurn: (...args: unknown[]) => forbidden.push(args) });
  try {
    const admitted = await runtime.openContext(scope(), new MemoryContextRequest(context(), null, {}), live());
    expect(admitted.transcript).toBeNull();
    expect(admitted.sections.map((section) => section.text)).toEqual(["outer notes", "local notes"]);
    expect(forbidden).toEqual([]);
  } finally {
    await runtime.close();
  }
});

test("transcript admission requires invocation before opening source", async () => {
  const calls: unknown[][] = [];
  const alpha = new FakeMemory("alpha", [], { value: new MemoryContent({ transcript: source({}, calls) }) });
  const runtime = await open(new Map([["alpha", alpha], ["beta", new FakeMemory("beta", [])]]));
  try {
    await expectError(
      () => runtime.openContext(scope(), new MemoryContextRequest(context(), null, {}), live()),
      RangeError,
      "transcript context requires an accepted invocation",
    );
    expect(calls).toEqual([]);
  } finally {
    await runtime.close();
  }
});

test("stop during recall prevents transcript source acquisition", async () => {
  const alpha = new FakeMemory("alpha", []);
  const runtime = await open(new Map([["alpha", alpha], ["beta", new FakeMemory("beta", [])]]));
  const controller = new AbortController();
  const reason = { stop: "host" };
  const calls: unknown[][] = [];
  alpha.session.recall = () => {
    controller.abort(reason);
    return new MemoryContent({ text: "notes", transcript: source({}, calls) });
  };
  const request = new MemoryContextRequest(context(), new ConversationInvocation("run", 1), {});
  try {
    let caught: unknown;
    await runtime.openContext(scope(), request, { signal: controller.signal }).catch((error: unknown) => {
      caught = error;
    });
    expect(caught).toBe(reason);
    expect(calls).toEqual([]);
  } finally {
    await runtime.close();
  }
});

test("context admits a transcript only from an eligible memory", async () => {
  for (const contributor of ["alpha", "beta"]) {
    const policies = named();
    const authored = origins();
    const runtime = await open(policies, { origins: authored });
    const opened: unknown[][] = [];
    const turn = { close() {} };
    try {
      policies.get(contributor)!.session.value = new MemoryContent({
        text: contributor,
        transcript: source(turn, opened),
      });
      const request = new MemoryContextRequest(context(), new ConversationInvocation("run", 1), {});
      if (contributor === "alpha") {
        const admitted = await runtime.openContext(scope(), request, live());
        expect(admitted.transcript).toBe(turn as never);
        expect(opened).toHaveLength(1);
      } else {
        const error = await expectError(
          () => runtime.openContext(scope(), request, live()),
          MemoryRecallError,
          "memory 'beta' returned a transcript but is not conversation-eligible; only a memory with one AGENT owner may hold a conversation",
        );
        expect((error as MemoryRecallError).origin).toBe(authored.get("beta")!);
        expect(opened).toEqual([]);
      }
    } finally {
      await runtime.close();
    }
  }
});

test("context rejects a second transcript for one agent", async () => {
  const agent = [new DeclSite([], ["root"], "agent")];
  const calls: unknown[][] = [];
  const policies = new Map(
    ["alpha", "gamma"].map((name) => [
      name,
      new FakeMemory(name, [], { value: new MemoryContent({ text: name, transcript: source({}, calls) }) }),
    ]),
  );
  const gammaSite = new OriginRef("gamma-site");
  const runtime = await MemoryRuntime.open({
    memories: [new MemorySpec("alpha", "alpha", {}, agent, true), new MemorySpec("gamma", "gamma", {}, agent, true)],
    origins: new Map([["alpha", null], ["gamma", gammaSite]]),
    memory: policies,
  });
  const recall = new RecallContext("base system", "rendered prompt", {}, ["team", "agent"]);
  const request = new MemoryContextRequest(recall, new ConversationInvocation("run", 1), {});
  try {
    const error = await expectError(
      () => runtime.openContext(new MemoryScope(["team", "agent"], { memories: ["alpha", "gamma"] }), request, live()),
      MemoryRecallError,
      "memory 'gamma' returned a second transcript for 'team/agent'; one AGENT may hold at most one conversation",
    );
    expect((error as MemoryRecallError).origin).toBe(gammaSite);
    expect(calls).toEqual([]);
  } finally {
    await runtime.close();
  }
});

test("openContext is a MemoryContextOpener", async () => {
  const runtime = await open(named());
  const opener: MemoryContextOpener = (s, r, o) => runtime.openContext(s, r, o);

  const admitted = await opener(scope(), new MemoryContextRequest(context(), null, {}), live());

  expect(admitted.sections.map((section) => section.text)).toEqual(["beta", "alpha"]);
  expect(admitted.transcript).toBeNull();
  await runtime.close();
});

test("an async recall is awaited", async () => {
  const policies = named();
  const runtime = await open(policies);
  policies.get("beta")!.session.recall = async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    return "outer";
  };
  policies.get("alpha")!.session.recall = async () => new MemoryContent({ text: "local" });

  const sections = await recallSections(runtime, scope(), context());

  expect(sections.map((section) => section.text)).toEqual(["outer", "local"]);
  await runtime.close();
});

test("openTurn receives the same signal and the scope's conversation request", async () => {
  const policies = named();
  const runtime = await open(policies);
  const calls: unknown[][] = [];
  const turn = { close() {} };
  policies.get("alpha")!.session.value = new MemoryContent({
    transcript: {
      async openTurn(request, { signal }) {
        calls.push([request, signal]);
        return turn as never;
      },
    },
  });
  const controller = new AbortController();
  const invocation = new ConversationInvocation("run", 7);

  const admitted = await runtime.openContext(
    scope(),
    new MemoryContextRequest(context(), invocation, { model: "m" }),
    { signal: controller.signal },
  );

  expect(admitted.transcript).toBe(turn as never);
  expect(admitted.sections.map((section) => section.text)).toEqual(["beta"]);
  expect(calls).toHaveLength(1);
  const [request, signal] = calls[0] as [ConversationRequest, AbortSignal];
  expect(signal).toBe(controller.signal);
  expect(Object.getPrototypeOf(request)).toBe(ConversationRequest.prototype);
  expect(request.consumerStaticPath).toEqual(["agent"]);
  expect(request.invocation).toBe(invocation);
  expect(request.compatibility).toEqual({ model: "m" });
  await runtime.close();
});

test("an openTurn rejection rejects openContext unchanged", async () => {
  const policies = named();
  const runtime = await open(policies);
  const failure = new StorageError("lease lost");
  policies.get("alpha")!.session.value = new MemoryContent({
    transcript: {
      async openTurn() {
        throw failure;
      },
    },
  });
  const request = new MemoryContextRequest(context(), new ConversationInvocation("run", 1), {});

  let caught: unknown;
  await runtime.openContext(scope(), request, live()).catch((error: unknown) => {
    caught = error;
  });

  expect(caught).toBe(failure);
  await runtime.close();
});

test("an already-aborted signal still recalls text-only memories, then rejects with the reason", async () => {
  const policies = named();
  const runtime = await open(policies);
  const controller = new AbortController();
  const reason = new Error("stopped");
  controller.abort(reason);

  let caught: unknown;
  await runtime
    .openContext(scope(), new MemoryContextRequest(context(), null, {}), { signal: controller.signal })
    .catch((error: unknown) => {
      caught = error;
    });

  expect(caught).toBe(reason);
  expect(policies.get("beta")!.session.calls).toHaveLength(1);
  expect(policies.get("alpha")!.session.calls).toHaveLength(1);
  await runtime.close();
});
