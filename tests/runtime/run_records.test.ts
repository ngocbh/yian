/**
 * Run records and their canonical wire codec: record construction and freezing, the seed and
 * typed-value helpers, envelope and payload decoding, and replay of live log lines through the run state.
 */

import { expect, test } from "bun:test";

import { copyGraph, START_ID } from "../../src/compile/model.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { MemoryMutation, MemoryOperationFailure } from "../../src/memory/contracts.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { ForwardCapture } from "../../src/runtime/contracts.ts";
import {
  canonicalSeed,
  canonicalTyped,
  decodeRecord,
  encodeRecord,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RecordEnvelope,
  RUN_RECORD_TYPES,
  RunCancelledRecord,
  RunFailedRecord,
  RUN_STOP_REASONS,
  type RunLog,
  RunLogSchemaError,
  type RunRecord,
  RunResumedRecord,
  type RunStartedFields,
  RunStartedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import * as runLog from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  HistoricalLogCursor,
  PreparedTransition,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareRecord,
  prepareRunStarted,
  replayRecords,
  requireUnchangedRecord,
  RunState,
} from "../../src/runtime/state.ts";
import {
  FileRef,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
} from "../../src/typesys/index.ts";
import { JSONDecodeError, type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import {
  commitStartSuccess,
  flow,
  forwardCapture,
  fragmentBuilder,
  learnableStarted,
  pool,
  startData,
  stateSnapshot,
  terminalSkipRouteFlow,
  untimed,
} from "../helpers/run_state_fixtures.ts";

/** Assert `fn` throws a `TypeCheckError` with exactly `message` at `path`. */
function expectTypeCheck(fn: () => unknown, message: string, path: readonly (string | number)[]): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(TypeCheckError);
  expect((caught as TypeCheckError).message).toBe(message);
  expect((caught as TypeCheckError).path).toEqual(path);
}

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`, and return it. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): Error {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught as Error;
}

const TOPOLOGY_DIGEST = "b".repeat(64);

test("a typed object output rejects a non-JSON member", () => {
  expectTypeCheck(
    () => canonicalTyped({ items: [1n, 2n], nested: { at: new Date(Date.UTC(2026, 7, 13)) } }, ValueKind.OBJECT),
    "<Date object> is not a JSON value",
    ["nested", "at"],
  );
});

test("a typed object output records a nested file as its mapping", () => {
  const file = new FileRef({ uri: "s3://bucket/report" });

  const [wire, decoded] = canonicalTyped({ items: [1n, 2n], nested: { at: file } }, ValueKind.OBJECT);

  const mapping = { uri: "s3://bucket/report", mime: null, name: null };
  expect(wire).toEqual({ kind: "object", value: { items: [1n, 2n], nested: { at: mapping } } });
  expect(decoded.toObject()).toEqual(wire.value);
});

test("a recorded object past the nesting limit still replays", () => {
  let value: Value = "leaf";
  for (let i = 0; i < 150; i++) value = { a: value };
  const recorded = TypedValue.fromWire({ kind: "object", value });

  const [wire, decoded] = canonicalTyped(recorded, ValueKind.OBJECT);

  expect(wire).toEqual({ kind: "object", value });
  expect(decoded).toEqual(recorded);
});

test("a seed canonicalizes a file ref", () => {
  const seed = { carried: [{ day: "2026-08-13" }, 3n], file: new FileRef({ uri: "memory://answer" }) };

  const [wire, decoded] = canonicalSeed(seed);

  expect(wire).toEqual({
    carried: [{ day: "2026-08-13" }, 3n],
    file: { mime: null, name: null, uri: "memory://answer" },
  });
  expect(Object.keys((wire as Record<string, Record<string, unknown>>)["file"]!)).toEqual(["mime", "name", "uri"]);
  expect(decoded).toEqual(wire);
  expect(canonicalSeed(decoded)[0]).toEqual(wire);
});

test("a seed keeps supported values and independent canonical copies", () => {
  const value = { files: [new FileRef({ uri: "memory://answer" })], count: 4n, ratio: 0.5 };
  const [wire, decoded] = canonicalSeed(value);
  type Seed = { files: { uri: string }[]; count: unknown };
  expect(wire).toEqual(decoded);
  expect(typeof (decoded as Seed).count).toBe("bigint");
  (decoded as Seed).files[0]!.uri = "changed";
  expect((wire as Seed).files[0]!.uri).toBe("memory://answer");
  expect(value.files[0]!.uri).toBe("memory://answer");
});

test("a seed keeps an infinite float", () => {
  const [wire, decoded] = canonicalSeed({ high: Infinity, low: [-Infinity] });
  expect(wire).toEqual({ high: Infinity, low: [-Infinity] });
  expect(decoded).toEqual(wire);
});

test.each<[string, () => unknown, (string | number)[]]>([
  ["raw", () => canonicalSeed(NaN), []],
  ["dict", () => canonicalSeed({ nested: [NaN] }), ["nested", 0]],
  ["typed", () => canonicalTyped(new TypedValue(ValueKind.OBJECT, { nested: NaN }), null), ["nested"]],
])("a seed rejects NaN: %s", (_name, fn, path) => {
  expectTypeCheck(fn, "NaN is not a yian value", path);
});

test("a typed output rejects NaN at its field path", () => {
  expectTypeCheck(() => canonicalTyped({ a: [1.5, NaN] }, null), "NaN is not a yian value", ["a", 1]);
});

test("the canonical helpers reject unordered containers", () => {
  expectTypeCheck(() => canonicalSeed({ values: new Set(["alpha", "beta"]) }), "<Set object> is not a JSON value", [
    "values",
  ]);
  expectTypeCheck(
    () => canonicalTyped({ values: new Set(["alpha", "beta"]) }, ValueKind.OBJECT),
    "<Set object> is not a JSON value",
    ["values"],
  );
});

test("the canonical helpers reject cyclic node data", () => {
  const cyclic: unknown[] = [];
  cyclic.push(cyclic);

  expectTypeCheck(() => canonicalSeed(cyclic), "value contains itself", [0]);
  expectTypeCheck(() => canonicalTyped(cyclic, null), "value contains itself", [0]);
});

/** A minimal structural implementation that pins the `RunLog` interface. */
class ProtocolLog {
  readonly runId = "run-protocol";
  readonly path: string | null = null;
  nextSequence = 0;
  closeCount = 0;
  #records: RecordEnvelope[] = [];
  #closed = false;

  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return this.#records[Symbol.iterator]();
  }

  records(): readonly RecordEnvelope[] {
    return Object.freeze([...this.#records]);
  }

  append(record: RunRecord): RecordEnvelope {
    const envelope = new RecordEnvelope(SCHEMA_VERSION, this.runId, this.nextSequence, record);
    this.#records.push(envelope);
    this.nextSequence += 1;
    return envelope;
  }

  close(): void {
    if (!this.#closed) {
      this.#closed = true;
      this.closeCount += 1;
    }
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

test("the run log protocol pins iteration, append, close and context management", () => {
  const log = new ProtocolLog();
  {
    const entered: RunLog = log;
    try {
      expect(entered).toBe(log);
      const envelope = log.append(new NodeStartedRecord("work"));
      expect(envelope.seq).toBe(0);
      expect([...log]).toEqual([envelope]);
      expect(log.nextSequence).toBe(1);
    } finally {
      entered[Symbol.dispose]();
    }
  }

  log.close();
  expect(log.closeCount).toBe(1);
});

const FAILURE = new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
const OPERATION_FAILURE = new MemoryOperationFailure("backend", "failed", "Failed.", true);

/** Return a run-start record with one agent's LLM config and `fields` over it. */
function runStarted(fields: Partial<RunStartedFields> = {}): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: { agent: { model: "gpt-test" } },
    numWorkers: 0,
    defaults: { MAX_TOTAL_NODES: 5000n },
    ...fields,
  });
}

test.each<[string, () => unknown]>([
  ["run resumed", () => ((new RunResumedRecord(["a"], 1) as { numWorkers: number }).numWorkers = 2)],
  ["run stopped", () => ((new RunStoppedRecord("signal", "stop", []) as { reason: string }).reason = "callback")],
  ["node started", () => ((new NodeStartedRecord("work") as { nodeId: string }).nodeId = "changed")],
  ["node step", () => ((new NodeStepRecord("a", 1, { kind: "test" }) as { nodeId: string }).nodeId = "b")],
  ["node succeeded", () => ((new NodeSucceededRecord("a", 1) as { commit: number }).commit = 2)],
  ["pool commit", () => ((new PoolCommitRecord("a", "a", { kind: "int", value: 1n }) as { key: string }).key = "b")],
  ["node routed", () => ((new NodeRoutedRecord("a", "next") as { handle: string }).handle = "other")],
  [
    "node expanded",
    () => ((new NodeExpandedRecord("grow", [], TOPOLOGY_DIGEST) as { spawnerId: string }).spawnerId = "other"),
  ],
  ["node failed", () => ((new NodeFailedRecord(FAILURE) as { failure: unknown }).failure = null)],
  [
    "the envelope",
    () => ((new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work")) as { seq: number }).seq = 1),
  ],
  ["run resumed retry ids", () => (new RunResumedRecord(["a"], 1).retryNodeIds as string[]).push("b")],
  ["run resumed closed calls", () => (new RunResumedRecord(["a"], 1, ["c"]).closedCalls as string[]).push("d")],
  ["run stopped node ids", () => (new RunStoppedRecord("signal", "stop", ["a"]).nodeIds as string[]).push("b")],
  ["a nested step value", () => ((new NodeStepRecord("a", 1, { kind: "test", n: { m: 1n } }).step["n"] as { m: unknown }).m = 2n)],
  [
    "a committed value",
    () => ((new PoolCommitRecord("a", "a", { kind: "int", value: 1n }).value as { value: unknown }).value = 2n),
  ],
  ["prune ids", () => (new NodeExpandedRecord("grow", ["a"], TOPOLOGY_DIGEST).pruneIds as string[]).push("b")],
  ["a failure", () => ((new NodeFailedRecord(FAILURE).failure as { message: string }).message = "changed")],
  ["run started", () => ((runStarted() as { numWorkers: number }).numWorkers = 2)],
  [
    "a nested start wiring override",
    () => ((runStarted({ startWiringOverride: { a: [1n] } }).startWiringOverride["a"] as unknown[]).push(2n)),
  ],
  [
    "a nested LLM config",
    () => ((runStarted().effectiveLlmByStaticPath["agent"] as { model: string }).model = "other"),
  ],
  ["run-started defaults", () => ((runStarted().defaults as Record<string, bigint>)["MAX_TOTAL_NODES"] = 1n)],
  [
    "a nested call-time LLM config",
    () =>
      ((runStarted({ callTimeLlm: { selection: {}, agents: { chat: {} } } }).callTimeLlm!["agents"] as {
        chat: unknown;
      }).chat = null),
  ],
  [
    "run-started memories",
    () => (runStarted({ memories: [], memoryStacks: [], memorySnapshots: [] }).memories as unknown[]).push(null),
  ],
  [
    "run-started memory stacks",
    () => (runStarted({ memories: [], memoryStacks: [], memorySnapshots: [] }).memoryStacks as unknown[]).push(null),
  ],
  [
    "run-started memory snapshots",
    () =>
      (runStarted({ memories: [], memoryStacks: [], memorySnapshots: [] }).memorySnapshots as unknown[]).push(null),
  ],
  [
    "memory commit requested",
    () => ((new MemoryCommitRequestedRecord("notes", "c", [], 1) as { attemptId: number }).attemptId = 2),
  ],
  [
    "commit mutations",
    () => (new MemoryCommitRequestedRecord("notes", "c", [], 1).mutations as unknown[]).push(null),
  ],
  [
    "memory commit succeeded",
    () => ((new MemoryCommitSucceededRecord("notes", "c", 1) as { commitId: string }).commitId = "d"),
  ],
  [
    "memory commit failed",
    () => ((new MemoryCommitFailedRecord("notes", "c", OPERATION_FAILURE, 1) as { failure: unknown }).failure = null),
  ],
])("records and envelopes are frozen: %s", (_name, assign) => {
  expect(assign).toThrow(TypeError);
});

test("the envelope names its record's wire kind", () => {
  expect(new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work")).kind).toBe("node_started");
});

test("RUN_RECORD_TYPES holds every record class", () => {
  expect(RUN_RECORD_TYPES).toEqual([
    RunStartedRecord,
    RunResumedRecord,
    RunStoppedRecord,
    NodeWaitingRecord,
    NodeWokenRecord,
    RunWaitingRecord,
    InboxClosedRecord,
    InboxWithdrawnRecord,
    InterruptedRecord,
    RunCancelledRecord,
    RunFailedRecord,
    NodeStartedRecord,
    NodeStepRecord,
    PoolCommitRecord,
    NodeSucceededRecord,
    NodeRoutedRecord,
    NodeExpandedRecord,
    NodeFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitFailedRecord,
  ]);
  expect(Object.isFrozen(RUN_RECORD_TYPES)).toBe(true);
});

test("records hold their carriers by reference and frozen copies of their values", () => {
  const capture = new ForwardCapture(ValueRoute.record([]));
  const terminal = new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "a" });
  const wire = { kind: "object", value: { b: [1n], a: { file: { uri: "s3://x", mime: null, name: null } } } };

  const succeeded = new NodeSucceededRecord("a", 1, { forwardCapture: capture });
  const committed = new PoolCommitRecord("a", "a", wire as never);
  const routed = new NodeRoutedRecord("a", "next", { terminalFailure: terminal });
  const failed = new NodeFailedRecord(FAILURE);
  wire.value.b.push(2n);

  expect(succeeded.forwardCapture).toBe(capture);
  expect(committed.value).toEqual({ kind: "object", value: { a: { file: wire.value.a.file }, b: [1n] } });
  expect(Object.keys(committed.value!.value as object)).toEqual(["a", "b"]);
  expect(routed.terminalFailure).toEqual(terminal);
  expect(routed.terminalFailure).not.toBe(terminal);
  expect(Object.isFrozen(routed.terminalFailure)).toBe(true);
  expect(failed.failure).toEqual(FAILURE);
  expect(failed.failure).not.toBe(FAILURE);
  expect(Object.isFrozen(failed.failure)).toBe(true);
});

test.each<[string, Readonly<{ kind: string; value: unknown }>, string]>([
  ["an int", { kind: "int", value: 2n }, '"kind":"int","node_id":"a","value":2'],
  ["a float", { kind: "float", value: 2 }, '"kind":"float","node_id":"a","value":2.0'],
  ["an int above 2^53", { kind: "int", value: 2n ** 60n }, `"kind":"int","node_id":"a","value":${2n ** 60n}`],
])("a pool commit encodes flat and decodes back equal: %s", (_name, wire, text) => {
  const record = new PoolCommitRecord("a", "a", wire as never);
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 1, record);
  const encoded = encodeRecord(envelope);
  expect(encoded).toContain(`"payload":{"key":"a",${text}}`);
  const decoded = decodeRecord(encoded);
  expect(decoded).toEqual(envelope);
  expect((decoded.record as PoolCommitRecord).value).toEqual(wire as never);
});

test("a derived pool commit holds from and no value: an object with sorted keys that decodes back to a map", () => {
  const from = new Map([
    ["b", 7],
    ["a", 4],
    ["system.now", 1],
  ]);
  const record = new PoolCommitRecord("e", "e", null, { from });
  expect([...record.from!]).toEqual([
    ["a", 4],
    ["b", 7],
    ["system.now", 1],
  ]);
  expect(record.value).toBeNull();
  expect(Object.isFrozen(record.from)).toBe(true);
  from.set("c", 9);
  expect(record.from!.has("c")).toBe(false);
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 8, record);
  const encoded = encodeRecord(envelope);
  expect(encoded).toContain('"payload":{"from":{"a":4,"b":7,"system.now":1},"key":"e","node_id":"e"}');
  const decoded = decodeRecord(encoded);
  expect(decoded).toEqual(envelope);
  expect([...(decoded.record as PoolCommitRecord).from!]).toEqual([...record.from!]);
  expect(new PoolCommitRecord("e", "e", { kind: "int", value: 2n }).from).toBeNull();
  expect(new PoolCommitRecord("e", "e", null, { from: new Map() }).from!.size).toBe(0);
  for (const [value, both] of [
    [{ kind: "int", value: 2n }, new Map()],
    [null, null],
  ] as const) {
    expectError(
      () => new PoolCommitRecord("e", "e", value as never, { from: both }),
      RangeError,
      "a pool_commit holds exactly one of value, from, take and steps",
    );
  }
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["a seq that isn't a positive integer", new Map([["a", 0]]), RangeError, "from seq of 'a' must be a positive integer"],
  ["an empty name", new Map([["", 3]]), RangeError, "from names must be non-empty"],
  ["a plain object", { a: 3 }, TypeError, "from must be a Map or null"],
])("a pool commit refuses a from with %s", (_name, from, cls, message) => {
  expectError(() => new PoolCommitRecord("e", "e", null, { from: from as never }), cls, message);
});

test("a node success names its commit and holds no value", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 6, new NodeSucceededRecord("a", 5));
  expect(envelope.payload).toEqual({ node_id: "a", commit: 5n, messages: null, forward_capture: null });
  for (const commit of [0, -1, 1.5, Number.NaN, 2 ** 53, "5"]) {
    expectError(() => new NodeSucceededRecord("a", commit as never), RangeError, "commit must be a positive integer");
  }
});

test.each<[string, string, string]>([
  ["node_succeeded", "value", '{"kind":"int","value":1}'],
  ["node_succeeded", "commit_target", '"a"'],
])("decode rejects a %s payload holding the retired %s", (kind, field, json) => {
  const line =
    `{"at":"2026-01-01T00:00:00.000Z","kind":"${kind}","payload":{"commit":5,` +
    `"forward_capture":null,"messages":null,"node_id":"a","${field}":${json}},"run_id":"run","schema":${SCHEMA_VERSION},"seq":6}`;
  expectError(() => decodeRecord(line), RangeError, `invalid ${kind} payload fields; missing=[], extra=["${field}"]`);
});

test("record constructors validate programmatically constructed values", () => {
  expectError(() => new NodeStartedRecord(3 as never), TypeError, "node_id must be a non-empty string");
  expectError(() => new NodeFailedRecord({} as never), TypeError, "NodeFailedRecord failure must be a Failure");
});

test.each<[string, () => unknown, new (...args: never[]) => Error, string]>([
  [
    "a failure without a node id",
    () => new NodeFailedRecord(new Failure({ code: "node.execute", message: "", nodeId: null })),
    RangeError,
    "NodeFailedRecord failure node_id must be a non-empty string",
  ],
  [
    "a terminal failure without a node id",
    () =>
      new NodeRoutedRecord("a", "next", {
        terminalFailure: new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "" }),
      }),
    RangeError,
    "NodeRoutedRecord failure node_id must be a non-empty string",
  ],
  [
    "a capture of another class",
    () => new NodeExpandedRecord("grow", [], TOPOLOGY_DIGEST, { forwardCapture: {} as never }),
    TypeError,
    "forward_capture must be a ForwardCapture or null",
  ],
  ["negative workers", () => new RunResumedRecord([], -1), RangeError, "num_workers must be non-negative"],
  ["fractional workers", () => new RunResumedRecord([], 1.5), TypeError, "num_workers must be an integer"],
  ["bigint workers", () => new RunResumedRecord([], 1n as never), TypeError, "num_workers must be an integer"],
  ["boolean workers", () => new RunResumedRecord([], true as never), TypeError, "num_workers must be an integer"],
  [
    "a non-text digest",
    () => new NodeExpandedRecord("grow", [], 3 as never),
    TypeError,
    "topology_digest must be a string",
  ],
  [
    "a digest that isn't 64 lowercase hex digits",
    () => new NodeExpandedRecord("grow", [], "B".repeat(64)),
    RangeError,
    "topology_digest must be 64 lowercase hex digits",
  ],
  [
    "an empty commit key",
    () => new PoolCommitRecord("a", "", { kind: "int", value: 1n }),
    RangeError,
    "key must be a non-empty string",
  ],
  ["an empty handle", () => new NodeRoutedRecord("a", ""), RangeError, "handle must be a non-empty string"],
  [
    "an empty spawner id",
    () => new NodeExpandedRecord("", [], TOPOLOGY_DIGEST),
    RangeError,
    "spawner_id must be a non-empty string",
  ],
  [
    "a bad typed-value wire",
    () => new PoolCommitRecord("a", "a", { kind: "int", value: "1" }),
    RangeError,
    'invalid int typed value: expected a bigint, got "1"',
  ],
  [
    "a negative envelope seq",
    () => new RecordEnvelope(SCHEMA_VERSION, "run", -1, new NodeStartedRecord("work")),
    RangeError,
    "run-record envelope seq must be a non-negative integer",
  ],
  [
    "a fractional envelope seq",
    () => new RecordEnvelope(SCHEMA_VERSION, "run", 1.5, new NodeStartedRecord("work")),
    RangeError,
    "run-record envelope seq must be a non-negative integer",
  ],
  ["a non-text stop reason", () => new RunStoppedRecord(3 as never, "stop", []), TypeError, "reason must be a string"],
  [
    "a non-text flow fingerprint",
    () => runStarted({ flowFingerprint: 3 as never }),
    TypeError,
    "flow_fingerprint must be a string",
  ],
  [
    "a flow fingerprint that isn't 64 lowercase hex digits",
    () => runStarted({ flowFingerprint: "A".repeat(64) }),
    RangeError,
    "flow_fingerprint must be 64 lowercase hex digits",
  ],
  ["negative run-start workers", () => runStarted({ numWorkers: -1 }), RangeError, "num_workers must be non-negative"],
  [
    "bigint run-start workers",
    () => runStarted({ numWorkers: 1n as never }),
    TypeError,
    "num_workers must be an integer",
  ],
  [
    "a NaN in the start wiring override",
    () => runStarted({ startWiringOverride: { x: [NaN] } }),
    TypeCheckError,
    "NaN is not a yian value",
  ],
  [
    "a start wiring override that isn't an object",
    () => runStarted({ startWiringOverride: [] as never }),
    TypeError,
    "start_wiring_override must be a mapping",
  ],
  [
    "an agent configuration that isn't an object",
    () => runStarted({ effectiveLlmByStaticPath: "openai" as never }),
    TypeError,
    "effective_llm_by_static_path must be a mapping",
  ],
  [
    "one agent's configuration that isn't an object",
    () => runStarted({ effectiveLlmByStaticPath: { agent: ["openai"] as never } }),
    TypeError,
    "effective_llm_by_static_path value for 'agent' must be a mapping",
  ],
  [
    "an empty commit memory",
    () => new MemoryCommitSucceededRecord("", "commit", 1),
    RangeError,
    "memory must be a non-empty string",
  ],
  [
    "a non-text commit id",
    () => new MemoryCommitRequestedRecord("notes", 3 as never, [], 1),
    TypeError,
    "commit_id must be a non-empty string",
  ],
  [
    "commit mutations that aren't an array",
    () => new MemoryCommitRequestedRecord("notes", "commit", new Set() as never, 1),
    TypeError,
    "mutations must be an array",
  ],
  [
    "a commit failure of another class",
    () => new MemoryCommitFailedRecord("notes", "commit", {} as never, 1),
    TypeError,
    "failure must be a MemoryOperationFailure",
  ],
])("record constructors refuse %s", (_name, fn, cls, message) => {
  expectError(fn, cls, message);
});

test.each<[string, () => unknown, string]>([
  ["text retry ids", () => new RunResumedRecord("ab" as never, 1), "retry_node_ids must be an array of strings"],
  [
    "a non-text retry id",
    () => new RunResumedRecord(["a", 2] as never, 1),
    "retry_node_ids must contain only non-empty strings",
  ],
  ["an empty retry id", () => new RunResumedRecord([""], 1), "retry_node_ids must contain only non-empty strings"],
  ["a repeated retry id", () => new RunResumedRecord(["a", "a"], 1), "retry_node_ids must contain unique IDs"],
  ["a repeated closed call", () => new RunResumedRecord(["a"], 1, ["c", "c"]), "closed_calls must contain unique IDs"],
  ["an empty closed call", () => new RunResumedRecord(["a"], 1, [""]), "closed_calls must contain only non-empty strings"],
  [
    "text prune ids",
    () => new NodeExpandedRecord("grow", "ab" as never, TOPOLOGY_DIGEST),
    "prune_ids must be an array of strings",
  ],
  [
    "a non-text prune id",
    () => new NodeExpandedRecord("grow", ["a", 2] as never, TOPOLOGY_DIGEST),
    "prune_ids must contain only non-empty strings",
  ],
  [
    "an empty prune id",
    () => new NodeExpandedRecord("grow", [""], TOPOLOGY_DIGEST),
    "prune_ids must contain only non-empty strings",
  ],
  [
    "a repeated prune id",
    () => new NodeExpandedRecord("grow", ["a", "a"], TOPOLOGY_DIGEST),
    "prune_ids must contain unique IDs",
  ],
  [
    "unsorted prune ids",
    () => new NodeExpandedRecord("grow", ["b", "a"], TOPOLOGY_DIGEST),
    "prune_ids must be sorted",
  ],
])("record constructors reject invalid id arrays: %s", (_name, fn, message) => {
  expectError(fn, RangeError, message);
});

test.each<[string, unknown, string]>([
  ["7", 7, "unsupported run-record schema: 7"],
  ["8", 8, "unsupported run-record schema: 8"],
  ["9", 9, "unsupported run-record schema: 9"],
  ["10", 10, "run log schema 10 predates schema 27 and is not supported"],
  ["11", 11, "run log schema 11 predates schema 27 and is not supported"],
  ["12", 12, "run log schema 12 predates schema 27 and is not supported"],
  ["13", 13, "run log schema 13 predates schema 27 and is not supported"],
  ["14", 14, "run log schema 14 predates schema 27 and is not supported"],
  ["15", 15, "run log schema 15 predates schema 27 and is not supported"],
  ["16", 16, "run log schema 16 predates schema 27 and is not supported"],
  ["17", 17, "run log schema 17 predates schema 27 and is not supported"],
  ["18", 18, "run log schema 18 predates schema 27 and is not supported"],
  ["19", 19, "run log schema 19 predates schema 27 and is not supported"],
  ["20", 20, "run log schema 20 predates schema 27 and is not supported"],
  ["21", 21, "run log schema 21 predates schema 27 and is not supported"],
  ["22", 22, "run log schema 22 predates schema 27 and is not supported"],
  ["23", 23, "run log schema 23 predates schema 27 and is not supported"],
  ["24", 24, "run log schema 24 predates schema 27 and is not supported"],
  ["25", 25, "run log schema 25 predates schema 27 and is not supported"],
  ["26", 26, "run log schema 26 predates schema 27 and is not supported"],
  ["28", 28, "unsupported run-record schema: 28"],
  ["true", true, "unsupported run-record schema: true"],
  ["27n", 27n, "unsupported run-record schema: 27n"],
])("the envelope refuses schema %s", (_name, schema, message) => {
  expectError(() => new RecordEnvelope(schema as never, "run", 0, new NodeStartedRecord("work")), RunLogSchemaError, message);
});

test("the envelope checks its schema before any other field", () => {
  expectError(
    () => new RecordEnvelope(9, "", -1, {} as never),
    RunLogSchemaError,
    "unsupported run-record schema: 9",
  );
});

test.each<[string, string, string]>([
  ["empty", "", "run log run_id must be a non-empty string"],
  ["a lone surrogate", "run-\ud800", "Unicode surrogate is not valid canonical JSON"],
])("the envelope refuses a run id: %s", (_name, runId, message) => {
  expectError(() => new RecordEnvelope(SCHEMA_VERSION, runId, 0, new NodeStartedRecord("work")), RangeError, message);
});

test("the envelope refuses an unsupported record", () => {
  expectError(
    () => new RecordEnvelope(SCHEMA_VERSION, "run", 0, {} as never),
    TypeError,
    "unsupported run record: Object",
  );
});

test("the encoder refuses a record of a class that isn't a run-record kind", () => {
  // The constructor refuses such a record, so the envelope is forged to reach the encoder.
  class StrayRecord {}
  const forged = Object.create(RecordEnvelope.prototype, { record: { value: new StrayRecord() } }) as RecordEnvelope;
  expectError(() => forged.payload, TypeError, "unsupported run record: StrayRecord");
});

test("a run-stopped record keeps its reason, message and in-flight node ids", () => {
  const record = new RunStoppedRecord("callback", "the host callback threw", ["b", "a"]);
  expect(RUN_STOP_REASONS).toEqual(["signal", "callback"]);
  expect([record.reason, record.message, record.nodeIds]).toEqual(["callback", "the host callback threw", ["b", "a"]]);
  expect(new RunStoppedRecord("signal", "", []).nodeIds).toEqual([]);
  expect(new RecordEnvelope(SCHEMA_VERSION, "run", 3, record).kind).toBe("run_stopped");
});

test.each<[string, () => unknown, new (...args: never[]) => Error, string]>([
  [
    "an unknown reason",
    () => new RunStoppedRecord("timeout" as never, "stop", []),
    RangeError,
    `reason must be one of ["signal","callback"], got 'timeout'`,
  ],
  [
    "an empty node id",
    () => new RunStoppedRecord("signal", "stop", [""]),
    RangeError,
    "node_ids must contain only non-empty strings",
  ],
  [
    "a repeated node id",
    () => new RunStoppedRecord("signal", "stop", ["a", "a"]),
    RangeError,
    "node_ids must contain unique IDs",
  ],
  ["a non-text message", () => new RunStoppedRecord("signal", 3 as never, []), TypeError, "message must be a string"],
])("a run-stopped record refuses %s", (_name, fn, cls, message) => {
  expectError(fn, cls, message);
});

/** The run-start record of the round trips: a one-entry pool, a wiring override and one agent. */
function roundTripRunStarted(): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: { name: "Ada" },
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test" } },
    numWorkers: 2,
  });
}

/** A run-start record holding the flow files, skills, tool specs and groups, version and parent. */
function sourcedRunStarted(): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    flow: {
      root: "flow.yaml",
      dir: "/x",
      codeRoot: "/x/code",
      files: {
        "flow.yaml": { text: "id: a\n", refs: { "lib/b": "lib/b.yaml" } },
        "lib/b.yaml": { text: "id: b\n", refs: {} },
      },
    },
    skills: { s: { description: "d", instructions: "i" } },
    tools: { t: { name: "t", description: "d", parameters: { type: "object" } } },
    toolGroups: { g: ["t"] },
    yianVersion: "0.0.0",
    parent: { runId: "p", node: "n" },
  });
}

const ROUND_TRIP_RECORDS: [string, RunRecord][] = [
  ["RunStartedRecord", roundTripRunStarted()],
  ["RunStartedRecord with its sources", sourcedRunStarted()],
  ["RunResumedRecord", new RunResumedRecord(["slow", "later"], 4)],
  ["RunResumedRecord closing calls", new RunResumedRecord(["slow"], 4, ["run/slow/1/0", "run/slow/1/1"])],
  ["RunStoppedRecord", new RunStoppedRecord("callback", "the host callback threw", ["b", "a"])],
  [
    "NodeWaitingRecord",
    new NodeWaitingRecord("heard", ["run/heard/1"], {
      waits: [{ callId: "run/heard/1", tool: "take", inbox: "msgs", mode: "all", deadline: "2026-10-24T09:00:00.000Z" }],
    }),
  ],
  ["NodeWokenRecord", new NodeWokenRecord("heard", ["run/heard/1"])],
  ["RunWaitingRecord", new RunWaitingRecord(["heard", "ask"])],
  ["InboxClosedRecord", new InboxClosedRecord("msgs")],
  ["InboxWithdrawnRecord", new InboxWithdrawnRecord("msgs", [3])],
  ["InterruptedRecord", new InterruptedRecord("owner", ["reply", "side"])],
  ["RunCancelledRecord", new RunCancelledRecord()],
  [
    "RunFailedRecord",
    new RunFailedRecord(new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "choice" })),
  ],
  ["NodeStartedRecord", new NodeStartedRecord("agent")],
  ["NodeStepRecord", new NodeStepRecord("agent", 2, { kind: "tool_attempt", event: "started", attempt: 2n })],
  ["PoolCommitRecord", new PoolCommitRecord("agent", "agent", { kind: "str", value: "done" })],
  ["PoolCommitRecord, a take", new PoolCommitRecord("heard", "heard", null, { take: { inbox: "msgs", events: [4, 6], closed: true } })],
  ["NodeSucceededRecord", new NodeSucceededRecord("agent", 3)],
  ["NodeSucceededRecord with its messages commit", new NodeSucceededRecord("agent", 3, { messages: 4 })],
  ["PoolCommitRecord, a messages commit", new PoolCommitRecord("agent", "agent.new_messages", null, { steps: [2, 3] })],
  ["NodeRoutedRecord", new NodeRoutedRecord("choice", "approved")],
  [
    "NodeExpandedRecord",
    new NodeExpandedRecord("each", ["each#old/end", "each#old/start"], TOPOLOGY_DIGEST),
  ],
  [
    "NodeFailedRecord",
    new NodeFailedRecord(
      new Failure({
        code: "node.execute",
        message: "boom",
        nodeId: "broken",
        trace: [new OriginRef("root/nodes/broken/code")],
        causeType: "ValueError",
        traceback: "Traceback (most recent call last): ...",
      }),
    ),
  ],
  [
    "MemoryCommitRequestedRecord",
    new MemoryCommitRequestedRecord("history", "commit-1", [new MemoryMutation("m1", { text: "Remember." })], 1),
  ],
  ["MemoryCommitSucceededRecord", new MemoryCommitSucceededRecord("history", "commit-1", 1)],
  [
    "MemoryCommitFailedRecord",
    new MemoryCommitFailedRecord(
      "history",
      "commit-1",
      new MemoryOperationFailure("backend", "backend.failed", "Failed.", true),
      1,
    ),
  ],
];

test.each<[string, Record<string, unknown>]>([
  ["memory_commit_requested", { mutations: [] }],
  ["memory_commit_succeeded", {}],
  ["memory_commit_failed", { failure: { stage: "commit", code: "declined", message: "Declined.", retryable: true } }],
])("the memory commit wire requires a positive attempt: %s", (kind, extra) => {
  const wire = {
    schema: 27,
    run_id: "run",
    seq: 0,
    kind,
    payload: { memory: "history", commit_id: "commit-1", attempt_id: null as number | null, ...extra },
  };
  expectError(() => decodeRecord(JSON.stringify(wire)), TypeError, `${kind} payload attempt_id must be an integer`);
  for (const attemptId of [0, -1]) {
    wire.payload.attempt_id = attemptId;
    expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, "attempt_id must be positive");
  }
});

// A TS record refuses NaN when it is built, so no log ever sees it.
test.each<[string, () => unknown, (string | number)[]]>([
  [
    "a run-start LLM config",
    () => runStarted({ effectiveLlmByStaticPath: { a: { temperature: NaN } } }),
    ["a", "temperature"],
  ],
])("a record holding NaN is refused before it is appended: %s", (_name, build, path) => {
  expectTypeCheck(build, "NaN is not a yian value", path);
});

const RUN_STARTED_WIRE = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, roundTripRunStarted()));

test.each<[string, (payload: Record<string, unknown>) => void, new (...args: never[]) => Error, string]>([
  [
    "a missing field",
    (payload) => delete payload["defaults"],
    RangeError,
    'invalid run_started payload fields; missing=["defaults"], extra=[]',
  ],
  [
    "a non-text flow fingerprint",
    (payload) => (payload["flow_fingerprint"] = 7n),
    TypeError,
    "run_started payload flow_fingerprint must be a string",
  ],
  [
    "the retired initial_pool",
    (payload) => (payload["initial_pool"] = {}),
    RangeError,
    'invalid run_started payload fields; missing=[], extra=["initial_pool"]',
  ],
  [
    "a float worker count",
    (payload) => (payload["num_workers"] = 2),
    TypeError,
    "run_started payload num_workers must be an integer",
  ],
  [
    "a negative worker count",
    (payload) => (payload["num_workers"] = -1n),
    RangeError,
    "num_workers must be non-negative",
  ],
  [
    "a non-boolean learnable",
    (payload) => (payload["learnable"] = 1n),
    TypeError,
    "run_started payload learnable must be a boolean",
  ],
  [
    "memories that aren't an array",
    (payload) => (payload["memories"] = {}),
    TypeError,
    "run_started payload memories must be an array or null",
  ],
  [
    "defaults that aren't an object",
    (payload) => (payload["defaults"] = []),
    TypeError,
    "run_started payload defaults must be an object",
  ],
  [
    "a parent missing its node",
    (payload) => (payload["parent"] = { run_id: "p" }),
    RangeError,
    'invalid run_started payload parent fields; missing=["node"], extra=[]',
  ],
  [
    "an empty parent node",
    (payload) => (payload["parent"] = { run_id: "p", node: "" }),
    RangeError,
    "RunStarted parent node must be a non-empty string",
  ],
  [
    "a flow missing its root",
    (payload) => (payload["flow"] = { dir: "/x", code_root: null, files: { f: { text: "", refs: {} } } }),
    RangeError,
    'invalid run_started payload flow fields; missing=["root"], extra=[]',
  ],
  [
    "a flow whose files lack its root",
    (payload) => (payload["flow"] = { root: "f", dir: "/x", code_root: null, files: {} }),
    RangeError,
    "RunStarted flow files must hold the root 'f'",
  ],
  [
    "a flow file missing its refs",
    (payload) => (payload["flow"] = { root: "f", dir: "/x", code_root: null, files: { f: { text: "" } } }),
    RangeError,
    `invalid run_started payload flow files['f'] fields; missing=["refs"], extra=[]`,
  ],
  [
    "a ref to a file the flow doesn't hold",
    (payload) =>
      (payload["flow"] = { root: "f", dir: "/x", code_root: null, files: { f: { text: "", refs: { g: "g.yaml" } } } }),
    RangeError,
    `RunStarted flow files['f'] refs['g'] names no recorded file: 'g.yaml'`,
  ],
  [
    "a flow missing its code root",
    (payload) => (payload["flow"] = { root: "f", dir: "/x", files: { f: { text: "", refs: {} } } }),
    RangeError,
    'invalid run_started payload flow fields; missing=["code_root"], extra=[]',
  ],
  [
    "a flow with an empty code root",
    (payload) => (payload["flow"] = { root: "f", dir: "/x", code_root: "", files: { f: { text: "", refs: {} } } }),
    RangeError,
    "RunStarted flow codeRoot must be a non-empty string",
  ],
  [
    "a skill missing its instructions",
    (payload) => (payload["skills"] = { s: { description: "d" } }),
    RangeError,
    `invalid run_started payload skills['s'] fields; missing=["instructions"], extra=[]`,
  ],
  [
    "a tool missing its name",
    (payload) => (payload["tools"] = { t: { description: "d", parameters: {} } }),
    RangeError,
    `invalid run_started payload tools['t'] fields; missing=["name"], extra=[]`,
  ],
  [
    "a tool whose parameters aren't an object",
    (payload) => (payload["tools"] = { t: { name: "t", description: "d", parameters: [] } }),
    TypeError,
    "RunStarted tools['t'] parameters must be an object",
  ],
  [
    "a tool group that isn't a list of names",
    (payload) => (payload["tool_groups"] = { g: "t" }),
    TypeError,
    "RunStarted tool_groups['g'] must be an array of tool names",
  ],
  [
    "an empty tool name in a group",
    (payload) => (payload["tool_groups"] = { g: [""] }),
    RangeError,
    "RunStarted tool_groups['g'][0] must be a non-empty string",
  ],
  [
    "a tool group naming a tool the specs lack",
    (payload) => (payload["tool_groups"] = { g: ["missing"] }),
    RangeError,
    "RunStarted tool_groups['g'] names 'missing', which tools doesn't hold",
  ],
  [
    "an empty yian version",
    (payload) => (payload["yian_version"] = ""),
    RangeError,
    "RunStarted yian_version must be a non-empty string",
  ],
])("decode rejects a run_started payload with %s", (_name, edit, cls, message) => {
  const wire = parseJson(RUN_STARTED_WIRE, "codec") as { payload: Record<string, unknown> };
  edit(wire.payload);
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test.each(ROUND_TRIP_RECORDS)("every record round trips through a validated envelope: %s", (_name, record) => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-123", 0, record);

  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);

  expect(decoded).toEqual(envelope);
  expect(decoded.record.constructor).toBe(record.constructor);
  expect(encodeRecord(decoded)).toBe(encoded);
  expect(decoded.kind).toBe(envelope.kind);
  expect(decoded.payload).toEqual((parseJson(encoded, "codec") as JsonObject)["payload"] as JsonObject);
  expect(Object.isFrozen(decoded.payload)).toBe(true);
  expect(decodeRecord(new TextEncoder().encode(encoded))).toEqual(envelope);
});

test("the record wire examples cover the closed record inventory", () => {
  expect(new Set(ROUND_TRIP_RECORDS.map(([, record]) => record.constructor))).toEqual(new Set(RUN_RECORD_TYPES));
});

test.each([
  ["memory_frame_learning_started", { occurrences: [] }],
  ["memory_frame_learned", { mutations: [] }],
  [
    "memory_frame_learning_failed",
    { failure: { stage: "learn", code: "declined", message: "Declined.", retryable: true } },
  ],
])("removed frame-learning wire kinds are rejected: %s", (kind, extra) => {
  const wire = {
    schema: SCHEMA_VERSION,
    run_id: "run",
    seq: 0,
    kind,
    payload: {
      policy_name: "history",
      frame_ref: { root_flow_id: "root", static_path: ["root"], owner: "flow" },
      ...extra,
    },
  };
  const message = `run-record kind '${kind}' is unsupported in schema 27`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, message);
});

test.each([
  ["conversation_admitted", "ConversationAdmittedRecord"],
  ["conversation_effect_requested", "ConversationEffectRequestedRecord"],
  ["conversation_effect_observed", "ConversationEffectObservedRecord"],
  ["conversation_closed", "ConversationClosedRecord"],
])("removed conversation wire kinds are rejected: %s", (kind, recordName) => {
  const wire = { schema: SCHEMA_VERSION, run_id: "run", seq: 0, kind, payload: { checkpoint: {} } };
  const message = `run-record kind '${kind}' is unsupported in schema 27`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, message);
  expect(Object.hasOwn(runLog, recordName)).toBe(false);
});

test("record encoding is canonical, newline-free and byte-stable", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-canonical", 0, new NodeStartedRecord("work"));

  const encoded = encodeRecord(envelope);

  expect(encoded).toBe(
    '{"kind":"node_started","payload":{"node_id":"work"},"run_id":"run-canonical","schema":27,"seq":0}',
  );
  expect(encoded.endsWith("\n")).toBe(false);
  expect(encodeRecord(decodeRecord(encoded))).toBe(encoded);
});

test("a node failure encoding nests one canonical failure", () => {
  const envelope = new RecordEnvelope(
    SCHEMA_VERSION,
    "run-failure",
    7,
    new NodeFailedRecord(
      new Failure({
        code: "contract.input",
        message: "bad input",
        nodeId: "__start__",
        trace: [new OriginRef("root/input/window"), new OriginRef("root/type/window")],
        causeType: "TypeCheckError",
        traceback: null,
      }),
    ),
  );

  const encoded = encodeRecord(envelope);

  expect(JSON.parse(encoded)).toEqual({
    schema: 27,
    run_id: "run-failure",
    seq: 7,
    kind: "node_failed",
    payload: {
      forward_capture: null,
      failure: {
        code: "contract.input",
        message: "bad input",
        node_id: "__start__",
        trace: [{ key: "root/input/window" }, { key: "root/type/window" }],
        cause_type: "TypeCheckError",
        traceback: null,
      },
    },
  });
  expect(decodeRecord(encoded)).toEqual(envelope);
});

test("a route encoding persists only its optional terminal failure", () => {
  const failure = new Failure({
    code: "graph.terminal_unreachable",
    message: "terminal output 'result' skipped",
    nodeId: "choice",
    trace: [new OriginRef("root/nodes/choice")],
  });
  const causal = new RecordEnvelope(
    SCHEMA_VERSION,
    "run-route",
    3,
    new NodeRoutedRecord("choice", "default", { terminalFailure: failure }),
  );
  const noncausal = new RecordEnvelope(SCHEMA_VERSION, "run-route", 4, new NodeRoutedRecord("other", "yes"));

  expect(JSON.parse(encodeRecord(causal)).payload).toEqual({
    forward_capture: null,
    node_id: "choice",
    handle: "default",
    terminal_failure: {
      code: "graph.terminal_unreachable",
      message: "terminal output 'result' skipped",
      node_id: "choice",
      trace: [{ key: "root/nodes/choice" }],
      cause_type: null,
      traceback: null,
    },
  });
  expect(JSON.parse(encodeRecord(noncausal)).payload).toEqual({
    forward_capture: null,
    node_id: "other",
    handle: "yes",
    terminal_failure: null,
  });
  expect(decodeRecord(encodeRecord(causal))).toEqual(causal);
  expect(decodeRecord(encodeRecord(noncausal))).toEqual(noncausal);
});

// Counters nested in a capture (route sizes, spans and path indexes, an item index, an occurrence
// index, the envelope's seq) are written as ints, never as floats such as `1.0`.
test("a node success with a capture encodes its counters as ints", () => {
  const head = new ValueRoute("source", { source: new SourceRef("run", 1, ["items", 0]), expression: "a.items[0]" });
  const tail = new ValueRoute("source", { source: new SourceRef("run", 2), expression: "b" });
  const joined = new ValueRoute("concat", { children: [head, tail], sizes: [1, 2], expression: "(a.items[0] + b)" });
  const text = new ValueRoute("interpolation", {
    children: [tail],
    valueJson: '"n=3"',
    spans: [[2, 3]],
    expression: "n=${b}",
  });
  const picked = new ValueRoute("project", { children: [joined], path: [1], expression: "(a.items[0] + b)[1]" });
  const route = ValueRoute.record([
    ["joined", joined],
    ["text", text],
    ["picked", picked],
  ]);
  const item = new ForwardCapture(ValueRoute.record([["x", ValueRoute.literal(1n)]]), {
    evaluatedKeys: new Set(["x"]),
  });
  const capture = new ForwardCapture(route, {
    evaluatedKeys: new Set(["text", "joined"]),
    items: [[2, item]],
    scope: new ExecutionScope(["each", "work"], {
      flowPath: ["each"],
      occurrencePath: [new ExecutionPosition("map_item", ["each"], 3)],
    }),
  });
  const envelope = new RecordEnvelope(
    SCHEMA_VERSION,
    "run",
    5,
    new NodeSucceededRecord("work", 4, { forwardCapture: capture }),
  );

  const encoded = encodeRecord(envelope);

  const literal =
    '{"children":[],"expression":"1","kind":"literal","names":[],"path":[],"result_type":null,' +
    '"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":"1"}';
  const source = (seq: string, path: string, expression: string) =>
    `{"children":[],"expression":"${expression}","kind":"source","names":[],"path":[],"result_type":null,` +
    `"sizes":[],"snapshot":null,"source":{"execution_seq":${seq},"path":${path},"run_id":"run"},` +
    '"spans":[],"value_json":null}';
  const tailWire = source("2", "[]", "b");
  const joinedWire =
    `{"children":[${source("1", '["items",0]', "a.items[0]")},${tailWire}],"expression":"(a.items[0] + b)",` +
    '"kind":"concat","names":[],"path":[],"result_type":null,"sizes":[1,2],"snapshot":null,"source":null,' +
    '"spans":[],"value_json":null}';
  const textWire =
    `{"children":[${tailWire}],"expression":"n=\${b}","kind":"interpolation","names":[],"path":[],` +
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[[2,3]],"value_json":null}';
  const pickedWire =
    `{"children":[${joinedWire}],"expression":"(a.items[0] + b)[1]","kind":"project","names":[],"path":[1],` +
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null}';
  const itemWire =
    '{"bound":true,"evaluated_keys":["x"],"input_names":[],"items":[],' +
    `"route":{"children":[${literal}],"expression":"{x: 1}","kind":"record","names":["x"],"path":[],` +
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},' +
    '"scope":null,"variable_names":[]}';
  expect(encoded).toBe(
    '{"kind":"node_succeeded","payload":{"commit":4,"forward_capture":{"bound":true,' +
      '"evaluated_keys":["joined","text"],' +
      '"input_names":[],' +
      `"items":[{"capture":${itemWire},"index":2}],` +
      `"route":{"children":[${joinedWire},${textWire},${pickedWire}],` +
      '"expression":"{joined: (a.items[0] + b), text: n=${b}, picked: (a.items[0] + b)[1]}",' +
      '"kind":"record","names":["joined","text","picked"],"path":[],"result_type":null,"sizes":[],"snapshot":null,' +
      '"source":null,"spans":[],"value_json":null},"scope":{"boundary":null,"flow_path":["each"],' +
      '"occurrence_path":[{"index":3,"kind":"map_item","static_path":["each"]}],"static_path":["each","work"]},' +
      '"variable_names":[]},"messages":null,"node_id":"work"},"run_id":"run","schema":27,"seq":5}',
  );
  // The interpolation's text isn't recorded; learning rebuilds it.
  const recorded = ValueRoute.record([
    ["joined", joined],
    ["text", text.with({ valueJson: null })],
    ["picked", picked],
  ]);
  const { evaluatedKeys, items, scope } = capture;
  const decoded = new ForwardCapture(recorded, { evaluatedKeys, items, scope });
  expect(decodeRecord(encoded)).toEqual(
    new RecordEnvelope(SCHEMA_VERSION, "run", 5, new NodeSucceededRecord("work", 4, { forwardCapture: decoded })),
  );
  expect(encodeRecord(decodeRecord(encoded))).toBe(encoded);
});

test("a forward capture records whether its bound inputs are usable, and decode requires it", () => {
  const failure = new Failure({ code: "binding.evaluate", message: "Missing.", nodeId: "work" });
  const capture = new ForwardCapture(ValueRoute.record([]), { bound: false });
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 3, new NodeFailedRecord(failure, { forwardCapture: capture }));
  const encoded = encodeRecord(envelope);
  expect(encoded).toContain('"forward_capture":{"bound":false,');
  expect(decodeRecord(encoded)).toEqual(envelope);
  expect(new ForwardCapture(ValueRoute.record([])).bound).toBe(true);
  expectError(() => new ForwardCapture(ValueRoute.record([]), { bound: 1 as never }), TypeError, "bound must be a boolean");

  const wire = JSON.parse(encoded) as { payload: { forward_capture: Record<string, unknown> } };
  wire.payload.forward_capture["bound"] = "no";
  expectError(() => decodeRecord(JSON.stringify(wire)), TypeError, "node_failed payload forward_capture bound must be a boolean");
  delete wire.payload.forward_capture["bound"];
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    'invalid node_failed payload forward_capture fields; missing=["bound"], extra=[]',
  );
});

test("decode requires an explicit route terminal failure field", () => {
  const wire = {
    schema: 27,
    run_id: "run",
    seq: 0,
    kind: "node_routed",
    payload: { node_id: "choice", handle: "default" },
  };
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    'invalid node_routed payload fields; missing=["forward_capture","terminal_failure"], extra=[]',
  );
});

test("decode rejects an extra nested route terminal failure field", () => {
  const wire = {
    schema: 27,
    run_id: "run",
    seq: 0,
    kind: "node_routed",
    payload: {
      forward_capture: null,
      node_id: "choice",
      handle: "default",
      terminal_failure: {
        code: "graph.terminal_unreachable",
        message: "terminal output 'result' skipped",
        node_id: "choice",
        trace: [],
        cause_type: null,
        traceback: null,
        extra: true,
      },
    },
  };
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    'invalid node_routed payload terminal_failure fields; missing=[], extra=["extra"]',
  );
});

const NODE_STARTED = '"kind":"node_started","payload":{"node_id":"work"}';

test.each<[string, string, new (...args: never[]) => Error, string]>([
  ["schema 1", `{"schema":1,"run_id":"run","seq":0,${NODE_STARTED}}`, RunLogSchemaError, "unsupported run-record schema: 1"],
  [
    "schema true",
    `{"schema":true,"run_id":"run","seq":0,${NODE_STARTED}}`,
    RunLogSchemaError,
    "unsupported run-record schema: true",
  ],
  // A float token reads as a number, never as a schema; it is shown as the log writes floats.
  [
    "schema 27.0",
    `{"schema":27.0,"run_id":"run","seq":0,${NODE_STARTED}}`,
    RunLogSchemaError,
    "unsupported run-record schema: 27.0",
  ],
  [
    "an empty run id",
    `{"schema":27,"run_id":"","seq":0,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope run_id must be a non-empty string",
  ],
  [
    "seq -1",
    `{"schema":27,"run_id":"run","seq":-1,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope seq must be a non-negative integer",
  ],
  [
    "seq true",
    `{"schema":27,"run_id":"run","seq":true,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope seq must be an integer",
  ],
  [
    "an unknown kind",
    '{"schema":27,"run_id":"run","seq":0,"kind":"not_a_record","payload":{}}',
    RangeError,
    "run-record kind 'not_a_record' is unsupported in schema 27",
  ],
  [
    "an extra key",
    `{"schema":27,"run_id":"run","seq":0,${NODE_STARTED},"unexpected":true}`,
    RangeError,
    'invalid run-record envelope fields; missing=[], extra=["unexpected"]',
  ],
  [
    "a missing payload",
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_started"}',
    RangeError,
    'invalid run-record envelope fields; missing=["payload"], extra=[]',
  ],
  [
    "seq 0.0",
    `{"schema":27,"run_id":"run","seq":0.0,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope seq must be an integer",
  ],
  [
    "seq past 2^53 - 1",
    `{"schema":27,"run_id":"run","seq":9007199254740992,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope seq must be a safe integer",
  ],
  [
    "a non-text run id",
    `{"schema":27,"run_id":7,"seq":0,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope run_id must be a non-empty string",
  ],
  [
    "a non-text kind",
    '{"schema":27,"run_id":"run","seq":0,"kind":7,"payload":{}}',
    TypeError,
    "run-record envelope kind must be a string",
  ],
  [
    "a payload that isn't an object",
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_started","payload":[]}',
    TypeError,
    "run-record envelope payload must be an object",
  ],
  ["an envelope that isn't an object", "[10]", TypeError, "run-record envelope must be an object"],
  [
    "a missing schema",
    `{"run_id":"run","seq":0,${NODE_STARTED}}`,
    RangeError,
    'invalid run-record envelope fields; missing=["schema"], extra=[]',
  ],
  [
    "an old schema whatever else the line holds",
    '{"schema":9,"seq":"x","kind":"node_frobbed","extra":1}',
    RunLogSchemaError,
    "unsupported run-record schema: 9",
  ],
])("decode rejects an invalid envelope: %s", (_name, text, cls, message) => {
  expectError(() => decodeRecord(text), cls, message);
});

test.each<[string, string, unknown, new (...args: never[]) => Error, string]>([
  [
    "an unknown node_started field",
    "node_started",
    { node_id: "work", unexpected: true },
    RangeError,
    'invalid node_started payload fields; missing=[], extra=["unexpected"]',
  ],
  [
    "a missing node_started field",
    "node_started",
    {},
    RangeError,
    'invalid node_started payload fields; missing=["node_id"], extra=[]',
  ],
  [
    "a non-text node id",
    "node_started",
    { node_id: 7 },
    TypeError,
    "node_started payload node_id must be a non-empty string",
  ],
  [
    "a node_succeeded payload without its carrier fields",
    "node_succeeded",
    { node_id: "work", commit: 7 },
    RangeError,
    'invalid node_succeeded payload fields; missing=["forward_capture","messages"], extra=[]',
  ],
  [
    "a legacy node_failed locator without carriers",
    "node_failed",
    {
      failure: {
        code: "node.execute",
        message: "boom",
        node_id: "work",
        trace: [],
        cause_type: "ValueError",
        traceback: null,
      },
      locator: null,
    },
    RangeError,
    'invalid node_failed payload fields; missing=["forward_capture"], extra=["locator"]',
  ],
  [
    "an empty node id",
    "node_started",
    { node_id: "" },
    RangeError,
    "node_started payload node_id must be a non-empty string",
  ],
])("decode rejects unknown, missing and wrong payload fields: %s", (_name, kind, payload, cls, message) => {
  const wire = { schema: 27, run_id: "run", seq: 0, kind, payload };
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

type FailureWire = {
  schema: number;
  run_id: string;
  seq: number;
  kind: string;
  payload: Record<string, unknown> & { failure: Record<string, unknown> };
};

/** Return one independently mutable valid node-failure wire object. */
function nodeFailureWire(): FailureWire {
  return {
    schema: 27,
    run_id: "run",
    seq: 0,
    kind: "node_failed",
    payload: {
      forward_capture: null,
      failure: {
        code: "node.execute",
        message: "boom",
        node_id: "work",
        trace: [{ key: "root/nodes/work/code" }],
        cause_type: "ValueError",
        traceback: "trace",
      },
    },
  };
}

test("a valid node-failure wire decodes", () => {
  const decoded = decodeRecord(JSON.stringify(nodeFailureWire()));
  expect((decoded.record as NodeFailedRecord).failure.trace[0]!.key).toBe("root/nodes/work/code");
});

const FAILURE_LABEL = "node_failed payload failure";

test.each<[string, (payload: FailureWire["payload"]) => void, new (...args: never[]) => Error, string]>([
  [
    "missing-failure",
    (p) => delete (p as Record<string, unknown>)["failure"],
    RangeError,
    'invalid node_failed payload fields; missing=["failure"], extra=[]',
  ],
  [
    "extra-payload",
    (p) => (p["extra"] = true),
    RangeError,
    'invalid node_failed payload fields; missing=[], extra=["extra"]',
  ],
  [
    "missing-code",
    (p) => delete p.failure["code"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["code"], extra=[]`,
  ],
  [
    "missing-message",
    (p) => delete p.failure["message"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["message"], extra=[]`,
  ],
  [
    "missing-node-id",
    (p) => delete p.failure["node_id"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["node_id"], extra=[]`,
  ],
  [
    "missing-trace",
    (p) => delete p.failure["trace"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["trace"], extra=[]`,
  ],
  [
    "missing-cause-type",
    (p) => delete p.failure["cause_type"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["cause_type"], extra=[]`,
  ],
  [
    "missing-traceback",
    (p) => delete p.failure["traceback"],
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=["traceback"], extra=[]`,
  ],
  ["unknown-code", (p) => (p.failure["code"] = "node.unknown"), RangeError, "unknown failure code 'node.unknown'"],
  ["wrong-message", (p) => (p.failure["message"] = 7), TypeError, `${FAILURE_LABEL} message must be a string`],
  [
    "null-node-id",
    (p) => (p.failure["node_id"] = null),
    TypeError,
    `${FAILURE_LABEL} node_id must be a non-empty string`,
  ],
  [
    "empty-node-id",
    (p) => (p.failure["node_id"] = ""),
    RangeError,
    `${FAILURE_LABEL} node_id must be a non-empty string`,
  ],
  [
    "wrong-trace-container",
    (p) => (p.failure["trace"] = "origin"),
    TypeError,
    `${FAILURE_LABEL} trace must be an array`,
  ],
  [
    "missing-origin-key",
    (p) => (p.failure["trace"] = [{}]),
    RangeError,
    `invalid ${FAILURE_LABEL} trace item fields; missing=["key"], extra=[]`,
  ],
  [
    "empty-origin-key",
    (p) => (p.failure["trace"] = [{ key: "" }]),
    RangeError,
    `${FAILURE_LABEL} trace item key must be a non-empty string`,
  ],
  [
    "wrong-origin-key",
    (p) => (p.failure["trace"] = [{ key: 7 }]),
    TypeError,
    `${FAILURE_LABEL} trace item key must be a non-empty string`,
  ],
  [
    "extra-origin-field",
    (p) => (p.failure["trace"] = [{ key: "origin", site: "legacy" }]),
    RangeError,
    `invalid ${FAILURE_LABEL} trace item fields; missing=[], extra=["site"]`,
  ],
  [
    "extra-failure-field",
    (p) => (p.failure["error"] = "legacy"),
    RangeError,
    `invalid ${FAILURE_LABEL} fields; missing=[], extra=["error"]`,
  ],
  [
    "wrong-cause-type",
    (p) => (p.failure["cause_type"] = 7),
    TypeError,
    `${FAILURE_LABEL} cause_type must be a string or null`,
  ],
  [
    "wrong-traceback",
    (p) => (p.failure["traceback"] = []),
    TypeError,
    `${FAILURE_LABEL} traceback must be a string or null`,
  ],
  ["wrong-code", (p) => (p.failure["code"] = 7), TypeError, `${FAILURE_LABEL} code must be a string`],
  ["failure-not-object", (p) => (p.failure = [] as never), TypeError, `${FAILURE_LABEL} must be an object`],
  [
    "origin-not-object",
    (p) => (p.failure["trace"] = ["origin"]),
    TypeError,
    `${FAILURE_LABEL} trace item must be an object`,
  ],
])("decode rejects a malformed nested failure: %s", (_name, mutate, cls, message) => {
  const wire = nodeFailureWire();
  mutate(wire.payload);
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

test.each(["locator", "range"])("a node failure rejects the legacy location field %s", (field) => {
  const wire = nodeFailureWire();
  wire.payload[field] = null;
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    `invalid node_failed payload fields; missing=[], extra=[${JSON.stringify(field)}]`,
  );
});

test.each([1, 2, 3, 4, 5, 6, 7, 8, 9])("decode rejects old schema %d even with a nested failure", (schema) => {
  const wire = nodeFailureWire();
  wire.schema = schema;
  const message = `unsupported run-record schema: ${schema}`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RunLogSchemaError, message);
});

test("decode rejects flattened failure aliases", () => {
  const wire = nodeFailureWire();
  wire.payload = {
    node_id: "work",
    error: "boom",
    error_type: "ValueError",
    locator: null,
    range: null,
    traceback: "trace",
  } as never;
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    'invalid node_failed payload fields; missing=["failure","forward_capture"], ' +
      'extra=["error","error_type","locator","node_id","range","traceback"]',
  );
});

/** Assert decoding `data` throws RangeError `invalid run-record JSON`; return its cause. */
function invalidJsonCause(data: string | Uint8Array): Error {
  return expectError(() => decodeRecord(data), RangeError, "invalid run-record JSON").cause as Error;
}

test("decode rejects duplicate keys at any object depth", () => {
  const payloadKey = invalidJsonCause(
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"first","node_id":"second"}}',
  );
  expect(payloadKey).toBeInstanceOf(JSONDecodeError);
  expect(payloadKey.message).toStartWith('duplicate JSON object key: "node_id"');

  const originKey = invalidJsonCause(
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_failed",' +
      '"payload":{"failure":{"code":"node.execute","message":"boom",' +
      '"node_id":"work","trace":[{"key":"one","key":"two"}],' +
      '"cause_type":null,"traceback":null}}}',
  );
  expect(originKey).toBeInstanceOf(JSONDecodeError);
  expect(originKey.message).toStartWith('duplicate JSON object key: "key"');
});

test("encode validates programmatically constructed envelopes", () => {
  const forged = { schema: SCHEMA_VERSION, runId: "run", seq: 0, record: new NodeStartedRecord("work") };
  expectError(() => encodeRecord(forged as never), TypeError, "encodeRecord expects a RecordEnvelope");
});

test("canonical JSON rejects lone Unicode surrogates", () => {
  const surrogate = "\ud800";

  expectTypeCheck(() => canonicalSeed({ nested: [surrogate] }), "text holding a lone surrogate is not a yian value", [
    "nested",
    0,
  ]);
  expectTypeCheck(
    () => canonicalSeed({ [surrogate]: "nested key" }),
    "text holding a lone surrogate is not a yian value",
    [],
  );
  expectError(
    () => new RecordEnvelope(SCHEMA_VERSION, surrogate, 0, new NodeStartedRecord("work")),
    RangeError,
    "Unicode surrogate is not valid canonical JSON",
  );
  const cause = invalidJsonCause(
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"\\ud800"}}',
  );
  expect(cause).toBeInstanceOf(JSONDecodeError);
  expect(cause.message).toStartWith("string has an unpaired surrogate");
});

test("invalid record JSON keeps its message", () => {
  const cause = invalidJsonCause(new TextEncoder().encode('{"schema": 9, '));
  expect(cause).toBeInstanceOf(JSONDecodeError);
  expect(cause.message).toStartWith("Expecting property name");
});

test("a line with a UTF-8 BOM is invalid record JSON", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work"));
  const encoded = encodeRecord(envelope);
  for (const data of [new Uint8Array([0xef, 0xbb, 0xbf, ...new TextEncoder().encode(encoded)]), `\ufeff${encoded}`]) {
    const cause = invalidJsonCause(data);
    expect(cause).toBeInstanceOf(JSONDecodeError);
    expect(cause.message).toStartWith("Unexpected UTF-8 BOM");
  }
});

test("a line that isn't UTF-8 is invalid record JSON", () => {
  const bytes = new TextEncoder().encode(
    '{"schema":27,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"x"}}',
  );
  bytes[bytes.indexOf(0x78)] = 0xff;
  // Only the class is checked: the decoder's message differs between Bun and Node.
  expect(invalidJsonCause(bytes).constructor).toBe(TypeError);
});

test("decode refuses data that is neither text nor bytes", () => {
  expectError(() => decodeRecord(7 as never), TypeError, "decodeRecord expects a string or a Uint8Array");
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  [
    "an unknown reason",
    { reason: "timeout", message: "stop", node_ids: [] },
    RangeError,
    `reason must be one of ["signal","callback"], got 'timeout'`,
  ],
  [
    "an extra field",
    { reason: "signal", message: "stop", node_ids: [], extra: true },
    RangeError,
    'invalid run_stopped payload fields; missing=[], extra=["extra"]',
  ],
  [
    "a non-text reason",
    { reason: 7, message: "stop", node_ids: [] },
    TypeError,
    "run_stopped payload reason must be a string",
  ],
  [
    "a non-text node id",
    { reason: "signal", message: "stop", node_ids: [7] },
    TypeError,
    "run_stopped payload node_ids must be an array of strings",
  ],
  [
    "a repeated node id",
    { reason: "signal", message: "stop", node_ids: ["a", "a"] },
    RangeError,
    "node_ids must contain unique IDs",
  ],
])("decode rejects a run_stopped payload with %s", (_name, payload, cls, message) => {
  const wire = { schema: 27, run_id: "run", seq: 0, kind: "run_stopped", payload };
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

test.each<[string, string, new (...args: never[]) => Error, string]>([
  [
    "float workers",
    '{"closed_calls":[],"retry_node_ids":[],"num_workers":1.0}',
    TypeError,
    "run_resumed payload num_workers must be an integer",
  ],
  ["negative workers", '{"closed_calls":[],"retry_node_ids":[],"num_workers":-1}', RangeError, "num_workers must be non-negative"],
  [
    "text retry ids",
    '{"closed_calls":[],"retry_node_ids":"ab","num_workers":1}',
    TypeError,
    "run_resumed payload retry_node_ids must be an array of strings",
  ],
  [
    "no closed calls",
    '{"retry_node_ids":[],"num_workers":1}',
    RangeError,
    'invalid run_resumed payload fields; missing=["closed_calls"], extra=[]',
  ],
  [
    "text closed calls",
    '{"closed_calls":"c","retry_node_ids":[],"num_workers":1}',
    TypeError,
    "run_resumed payload closed_calls must be an array of strings",
  ],
])("decode rejects a run_resumed payload with %s", (_name, payload, cls, message) => {
  const text = `{"schema":27,"run_id":"run","seq":0,"kind":"run_resumed","payload":${payload}}`;
  expectError(() => decodeRecord(text), cls, message);
});

const TAMPERED = ["payload", "run id", "sequence"];

test.each(TAMPERED)("the prepared record tamper check uses the frozen schema: %s", (change) => {
  const state = new RunState(flow());
  let prepared = prepareRunStarted(state, startData());
  const before = stateSnapshot(state);
  requireUnchangedRecord(prepared, prepared._plan.envelopeJson);
  const { record, expectedRunId, expectedSequence, _plan: plan, observers } = prepared;
  if (change === "payload") {
    const started = record as RunStartedRecord;
    const tampered = new RunStartedRecord({
      ...started,
      startWiringOverride: { ...started.startWiringOverride, new: "tampered" },
    });
    prepared = new PreparedTransition(tampered, expectedRunId, expectedSequence, plan, observers);
  } else if (change === "run id") {
    prepared = new PreparedTransition(record, "other", expectedSequence, plan, observers);
  } else {
    prepared = new PreparedTransition(record, expectedRunId, 1, plan, observers);
  }
  expectError(
    () => requireUnchangedRecord(prepared, prepared._plan.envelopeJson),
    RangeError,
    change === "payload" ? "prepared record changed after validation" : "prepared coordinates changed after validation",
  );
  expect(stateSnapshot(state)).toEqual(before);
});

test("an append result of the wrong schema is a protocol fault", () => {
  /** Retain the record, then return its envelope with schema 9. */
  class WrongSchemaLog extends MemoryRunLog {
    override append(record: RunRecord): RecordEnvelope {
      const accepted = super.append(record);
      expect(accepted.schema).toBe(SCHEMA_VERSION);
      const forged = Object.create(RecordEnvelope.prototype) as RecordEnvelope;
      return Object.assign(forged, { ...accepted, schema: 9 });
    }
  }

  const state = new RunState(flow());
  const prepared = prepareRunStarted(state, { ...startData(), learnable: true });
  const before = stateSnapshot(state);
  const log = new WrongSchemaLog(prepared.expectedRunId);
  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    "run-log append result does not match prepared transition",
  );
  expect(stateSnapshot(state)).toEqual(before);
  expect(state.trace).toBeNull();
  expect(log.nextSequence).toBe(1);
  expect(untimed(encodeRecord(log.records()[0]!))).toBe(prepared._plan.envelopeJson);
});

test.each(["started", "resumed"])("a historical frontier prepares the recorded schema's line: %s", (kind) => {
  const live = new RunState(flow());
  const data = startData();
  const log = new MemoryRunLog(data.runId);
  commitPrepared(log, live, prepareRunStarted(live, data));
  commitInitialPool(log, live, pool(live.flow));

  const records = log.records();
  const state = replayRecords(new RunState(flow()), records, { growBuilder: fragmentBuilder });
  const record = kind === "resumed" ? new RunResumedRecord([], 0) : new NodeStartedRecord(state.flow.startId);
  const prepared = prepareRecord(state, record);

  const expected = new RecordEnvelope(SCHEMA_VERSION, state.runId!, records.length, record);
  expect(prepared._plan.envelopeJson).toBe(encodeRecord(expected));
  expect(state.nextSequence).toBe(records.length);
});

test("a historical route keeps the schema in its trace line", () => {
  const live = learnableStarted(terminalSkipRouteFlow());
  const original = copyGraph(live.state.flow);
  commitPrepared(live.log, live.state, prepareNodeStarted(live.state, START_ID));
  commitStartSuccess(live.state, live.log, {});
  commitPrepared(live.log, live.state, prepareNodeStarted(live.state, "router"));
  const router = live.state.flow.nodes.get("router")!;
  commitPrepared(
    live.log,
    live.state,
    prepareNodeRouted(live.state, "router", "go", { forwardCapture: forwardCapture(router, {}) }),
  );
  const wires = live.log.records().map(encodeRecord);
  const last = wires.at(-1)!;

  const state = replayRecords(new RunState(original), live.log.records().slice(0, -1), {
    growBuilder: fragmentBuilder,
  });
  const before = [...state.trace!.occurrences];
  const prepared = prepareRecord(state, decodeRecord(last).record, { growBuilder: fragmentBuilder });
  expect(prepared._plan.envelopeJson).toBe(untimed(last));
  expect(state.trace!.occurrences).toEqual(before);
  const cursor = new HistoricalLogCursor(wires);
  for (const wire of wires.slice(0, -1)) cursor.append(decodeRecord(wire).record);
  commitPrepared(cursor, state, prepared);

  expect(state.trace!.occurrences.slice(0, -1)).toEqual(before);
  const occurrence = state.trace!.occurrences.at(-1)!;
  expect(occurrence.recordJson).toBe(untimed(last));
  expect(decodeRecord(occurrence.recordJson).schema).toBe(SCHEMA_VERSION);
});
