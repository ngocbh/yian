/**
 * Run records and their canonical wire codec: a port of `tests/engine/test_run_records.py` and of
 * `test_long_int_old_logs.py::test_invalid_record_json_keeps_its_message`.
 *
 * Ported, the canonical helper tests:
 * `test_typed_object_output_records_a_nested_file_as_its_mapping`,
 * `test_recorded_object_past_the_nesting_limit_still_replays`,
 * `test_seed_keeps_supported_values_and_independent_canonical_copies`,
 * `test_pool_decoder_rejects_unknown_fields_and_wrong_types` (3 rows),
 * `test_run_log_protocol_pins_iteration_append_close_and_context_management` (a class assigned
 * to `RunLog`, closed through `using`),
 * `test_records_and_envelopes_are_frozen` (every record of this module and the envelope, nested
 * values and held arrays too; assignment is a `TypeError`). Python's `ValueError` is a
 * `RangeError` here.
 *
 * Ported, the wire codec tests (each row with its exact class and message; a field of the wrong
 * JSON type is a `TypeError`): `test_record_encoding_is_canonical_newline_free_and_byte_stable`
 * (`"schema":10`), `test_node_failure_encoding_nests_one_canonical_failure`,
 * `test_removed_frame_learning_wire_kinds_are_rejected`,
 * `test_removed_conversation_wire_kinds_are_rejected` (the module has no such export),
 * `test_decode_rejects_invalid_envelopes` (literal JSON text, so `10.0` stays a float token),
 * `test_decode_rejects_unknown_missing_and_wrong_payload_fields`,
 * `test_route_encoding_persists_only_its_optional_terminal_failure`,
 * `test_decode_requires_explicit_route_terminal_failure_field`,
 * `test_decode_rejects_extra_nested_route_terminal_failure_field`,
 * `test_decode_rejects_malformed_nested_failures` (20 rows),
 * `test_node_failure_rejects_legacy_location_fields`,
 * `test_decode_rejects_old_schemas_even_with_a_nested_failure` (1-8, and 9 naming Python yian),
 * `test_decode_rejects_flattened_failure_aliases`,
 * `test_decode_rejects_duplicate_keys_at_any_object_depth` (`invalid run-record JSON`, caused by
 * the reader's `JSONDecodeError` `duplicate JSON object key: "node_id"`, double-quoted),
 * `test_invalid_record_json_keeps_its_message`,
 * `test_every_record_round_trips_through_a_validated_envelope` (each row also decoded from
 * bytes) and
 * `test_record_wire_examples_cover_the_closed_record_inventory` (against `RUN_RECORD_TYPES`).
 *
 * TS-only: a seed keeps `±Infinity` as a float (a TS yian value, unlike Python's); a seed's
 * record keys come out in code-unit order; the envelope's schema rows (7, 8 and 11 are
 * unsupported, 9 names Python yian, a boolean is unsupported, and the schema is checked first),
 * its run-id rows (empty, a lone surrogate) and its unsupported-record row; the `RunStopped`
 * rows (an unknown reason, an empty or repeated node id, a non-text message; no node ids is
 * valid); a failure without a node id, an evidence or capture of another class, a bad worker
 * count, digest, commit target, handle, spawner id, typed-value wire or envelope seq, refused by
 * the record's constructor; a bad flow fingerprint, run-start worker count or initial pool, a
 * start wiring override holding NaN, a wiring override, agent configuration or one agent's
 * configuration that isn't a JSON object, a bad commit identity, mutation batch or failure, and
 * a backward carrier of another class, refused by the run-start, memory-commit and backward
 * records; `RUN_RECORD_TYPES`; carriers held by reference and failures and values held as
 * frozen copies. On the wire: a line starting with a UTF-8 BOM (bytes or text) and a line that
 * isn't UTF-8 are `invalid run-record JSON`; data that is neither text nor bytes; the `run_stopped`
 * round trip and its unknown-reason, extra-field, non-text and repeated-id rows; the
 * `run_resumed` float-counter, negative-counter and text-ids rows; envelope rows for a float or
 * too-large `seq`, a non-text run id or kind, a non-object payload or envelope, a missing schema,
 * and schema 9 named before anything else is checked; an empty `node_started` node id, a
 * non-text failure code, a failure or origin that isn't an object; a byte-pinned `node_succeeded`
 * line whose capture holds route sizes, spans and path indexes, an item index and an occurrence
 * index, all written as ints; a memory commit's `attempt_id` of 0 or -1; `run_started` rows for a
 * missing field, a non-text fingerprint, a pool, memory table or defaults of the wrong JSON type, a
 * float or negative worker count and a non-boolean `learnable`.
 *
 * Changed:
 * - `test_encode_validates_programmatically_constructed_envelopes`: the record's constructor
 *   refuses the bad node id and the non-`Failure` (Python refused them at encode); `encodeRecord`
 *   refuses an object that isn't a `RecordEnvelope`.
 * - `test_canonical_json_rejects_lone_unicode_surrogates_as_value_errors`: the seed rows are the
 *   value layer's `TypeCheckError`, the run-id row the envelope's `RangeError`, and the decode
 *   row reads `invalid run-record JSON` (the reader refuses the surrogate).
 * - Python's retired schemas 7 and 8 read `unsupported run-record schema: N` like every schema
 *   below 10.
 * - `test_encode_rejects_invalid_id_tuple_carriers`: the 9 rows are constructor rejections
 *   (Python refused them at encode); "tuple" reads "array".
 * - `test_typed_object_output_rejects_a_non_json_member`: Python's date and datetime rows are one
 *   JS `Date` row, refused by the value layer as `<Date object> is not a JSON value`.
 * - `test_seed_canonicalizes_tuple_date_and_file_ref`: only the `FileRef` half (TS has no tuples
 *   or dates among its values).
 * - `test_seed_rejects_nonfinite_before_json_mode_normalization`: the NaN rows `raw`, `dict` and
 *   `typed`, refused with the value layer's `NaN is not a yian value` at Python's paths. A TS seed
 *   is a plain value, never a `TypedValue`, so the `typed` row checks an unchecked `TypedValue`
 *   holding NaN through `canonicalTyped`.
 * - `test_typed_output_rejects_nonfinite_at_its_field_path`: the TS wording, at `["a", 1]`.
 * - `test_canonical_helpers_reject_unordered_containers`: a JS `Set`, refused by the value layer
 *   as `<Set object> is not a JSON value` by both helpers.
 * - `test_canonical_helpers_reject_cyclic_node_data`: both helpers give the value layer's `value
 *   contains itself` (Python's seed helper says `cyclic container`).
 * - `test_pool_codec_is_canonical_and_lossless_for_typed_values`: no date value; the tuple is an
 *   array.
 * - `test_memory_commit_wire_requires_a_positive_attempt`: a `null` attempt is a `TypeError` (a
 *   field of the wrong JSON type), with its message pinned.
 * - `test_expansion_record_with_nonfinite_seed_does_not_append` and
 *   `test_run_started_record_with_nonfinite_llm_config_does_not_append`: the record's constructor
 *   refuses the NaN (the value layer's `NaN is not a yian value`, at its path), so no log is
 *   involved.
 *
 * Not ported: the two list rows of `test_encode_rejects_invalid_id_tuple_carriers` (a TS id
 * carrier is an array); the `inf` and `-inf` rows (a TS float may be infinite) and the tuple,
 * model and dataclass rows of the non-finite seed test; the pydantic-extras tests; the dataclass
 * row of the unordered-container test;
 * `test_canonical_seed_rejects_keys_before_json_stringification` (JS object keys are always
 * text); `test_typed_object_output_canonicalizes_a_tuple` (TS has no tuples); the
 * recursion-depth tests (the TS codec is iterative); `test_pool_codec_rejects_nonfinite_values`
 * (`inf` is a TS float, and a NaN fails at `pool.set`).
 *
 * Ported, through the run state: `test_prepared_record_tamper_check_uses_frozen_schema` (3 rows,
 * calling the internal `requireUnchangedRecord` as Python does; the record is frozen, so the
 * `payload` row forges a `PreparedTransition` pairing a changed record with the original plan, and
 * the coordinate rows forge one with another run id or sequence).
 *
 * Changed, through the run state: `test_append_result_wrong_schema_is_protocol_fault`: TS refuses
 * to build an envelope of another schema, so the log returns one built on the envelope prototype
 * with schema 9; `encodeRecord` doesn't check the schema, so it fails as `run-log append result
 * does not match prepared transition` where Python names the old schema.
 *
 * Not ported: `test_historical_record_requires_an_exact_supported_schema` (`prepareRecord` has no
 * schema parameter, since schema 10 is the only one).
 *
 * Later (T4-5): `test_historical_frontier_prepares_recorded_schema_bytes` (`replayRecords`).
 * Later (T4-5, T4-8): `test_historical_outcomes_keep_schema_in_trace_bytes` (the replay fold and
 * `run.ts`'s log replay).
 */

import { expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import {
  EvaluationEvidence,
  LearningResult,
  MemoryMutation,
  MemoryOperationFailure,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardOutcome,
  ForwardCapture,
  MemoryProposal,
} from "../../src/runtime/contracts.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalSeed,
  canonicalTyped,
  decodePool,
  decodeRecord,
  encodePool,
  encodeRecord,
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
  RecordEnvelope,
  RUN_RECORD_TYPES,
  RUN_STOP_REASONS,
  type RunLog,
  type RunRecord,
  RunResumedRecord,
  type RunStartedFields,
  RunStartedRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import * as runLog from "../../src/runtime/log.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareRunStarted,
  requireUnchangedRecord,
  RunState,
} from "../../src/runtime/state.ts";
import {
  FileRef,
  type PoolWire,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
  VariablePool,
} from "../../src/typesys/index.ts";
import { JSONDecodeError, type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import { flow, startData, stateSnapshot } from "./run_state_fixtures.ts";

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

test("the pool codec is canonical and lossless for typed values", () => {
  const pool = new VariablePool("start");
  pool.set("start", { items: [1n, 2n], name: "x" });
  pool.set("file", new FileRef({ uri: "s3://bucket/input", name: "input.txt" }));
  pool.addSystem("run_id", "run-pool");

  const wire = encodePool(pool);
  const decoded = decodePool(wire);

  expect(wire.store["start"]!.value).toEqual({ items: [1n, 2n], name: "x" });
  expect(decoded.get("start")).toEqual(wire.store["start"]!.value);
  expect(decoded.get("file")).toEqual(new FileRef({ uri: "s3://bucket/input", mime: null, name: "input.txt" }));
  expect(encodePool(decoded)).toEqual(wire);
});

test.each<[string, unknown, string]>([
  [
    "an extra pool field",
    { store: {}, system: {}, start_id: "start", extra: true },
    'invalid variable pool fields; missing=[], extra=["extra"]',
  ],
  [
    "an int typed value holding text",
    { store: { node: { kind: "int", value: "1" } }, system: {}, start_id: "start" },
    'invalid int typed value: expected a bigint, got "1"',
  ],
  [
    "an extra typed-value field",
    { store: { node: { kind: "int", value: 1n, extra: true } }, system: {}, start_id: "start" },
    'invalid typed value fields; missing=[], extra=["extra"]',
  ],
])("the pool decoder rejects %s", (_name, wire, message) => {
  let caught: unknown;
  try {
    decodePool(wire as PoolWire);
  } catch (error) {
    caught = error;
  }
  expect((caught as Error).constructor).toBe(RangeError);
  expect((caught as Error).message).toBe(message);
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
    using entered: RunLog = log;
    expect(entered).toBe(log);
    const envelope = log.append(new NodeStartedRecord("work"));
    expect(envelope.seq).toBe(0);
    expect([...log]).toEqual([envelope]);
    expect(log.nextSequence).toBe(1);
  }

  log.close();
  expect(log.closeCount).toBe(1);
});

const FAILURE = new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
const OPERATION_FAILURE = new MemoryOperationFailure("backend", "failed", "Failed.", true);
const SUCCEEDED = new BackwardOutcome("succeeded");

/** Return a run-start record with an empty pool, one agent's LLM config and `fields` over them. */
function runStarted(fields: Partial<RunStartedFields> = {}): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(new VariablePool("start")),
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
  ["node step", () => ((new NodeStepRecord("a", {}) as { nodeId: string }).nodeId = "b")],
  [
    "node succeeded",
    () => {
      const record = new NodeSucceededRecord("a", "a", { kind: "int", value: 1n });
      (record as { commitTarget: string }).commitTarget = "b";
    },
  ],
  ["node routed", () => ((new NodeRoutedRecord("a", "next") as { handle: string }).handle = "other")],
  [
    "node expanded",
    () => ((new NodeExpandedRecord("grow", {}, [], TOPOLOGY_DIGEST) as { spawnerId: string }).spawnerId = "other"),
  ],
  ["node failed", () => ((new NodeFailedRecord(FAILURE) as { failure: unknown }).failure = null)],
  [
    "the envelope",
    () => ((new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work")) as { seq: number }).seq = 1),
  ],
  ["run resumed retry ids", () => (new RunResumedRecord(["a"], 1).retryNodeIds as string[]).push("b")],
  ["run stopped node ids", () => (new RunStoppedRecord("signal", "stop", ["a"]).nodeIds as string[]).push("b")],
  ["a nested step value", () => ((new NodeStepRecord("a", { n: { m: 1n } }).step["n"] as { m: unknown }).m = 2n)],
  [
    "a succeeded value",
    () => ((new NodeSucceededRecord("a", "a", { kind: "int", value: 1n }).value as { value: unknown }).value = 2n),
  ],
  [
    "a nested seed value",
    () => {
      const record = new NodeExpandedRecord("grow", { items: [1n] }, [], TOPOLOGY_DIGEST);
      (record.seed as { items: unknown[] }).items.push(2n);
    },
  ],
  ["prune ids", () => (new NodeExpandedRecord("grow", {}, ["a"], TOPOLOGY_DIGEST).pruneIds as string[]).push("b")],
  ["a failure", () => ((new NodeFailedRecord(FAILURE).failure as { message: string }).message = "changed")],
  ["run started", () => ((runStarted() as { numWorkers: number }).numWorkers = 2)],
  [
    "a run-started pool value",
    () => ((runStarted().initialPool.system as Record<string, unknown>)["run_id"] = null),
  ],
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
  ["backward started", () => ((new BackwardStartedRecord(1, null) as { attemptId: number }).attemptId = 2)],
  [
    "a nested backward feedback",
    () => ((new BackwardStartedRecord(1, { parts: ["a"] }).feedback as { parts: unknown[] }).parts.push("b")),
  ],
  [
    "backward ready",
    () => ((new BackwardReadyRecord(1, new BackwardResult(), [], []) as { attemptId: number }).attemptId = 2),
  ],
  [
    "ready attributed results",
    () => (new BackwardReadyRecord(1, new BackwardResult(), [], []).attributedResults as unknown[]).push(null),
  ],
  [
    "ready proposals",
    () => (new BackwardReadyRecord(1, new BackwardResult(), [], []).proposals as unknown[]).push(null),
  ],
  [
    "backward finished",
    () => ((new BackwardFinishedRecord(1, SUCCEEDED) as { outcome: unknown }).outcome = null),
  ],
  [
    "finished attributed results",
    () => (new BackwardFinishedRecord(1, SUCCEEDED).attributedResults as unknown[]).push(null),
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
    NodeStartedRecord,
    NodeStepRecord,
    NodeSucceededRecord,
    NodeRoutedRecord,
    NodeExpandedRecord,
    NodeFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
    MemoryCommitFailedRecord,
    BackwardStartedRecord,
    BackwardReadyRecord,
    BackwardFinishedRecord,
  ]);
  expect(Object.isFrozen(RUN_RECORD_TYPES)).toBe(true);
});

test("records hold their carriers by reference and frozen copies of their values", () => {
  const evidence = new EvaluationEvidence(null, { q: "why" }, null, null);
  const capture = new ForwardCapture(null, ValueRoute.record([]));
  const terminal = new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "a" });
  const wire = { kind: "object", value: { b: [1n], a: { file: { uri: "s3://x", mime: null, name: null } } } };

  const succeeded = new NodeSucceededRecord("a", "a", wire as never, { evidence, forwardCapture: capture });
  const routed = new NodeRoutedRecord("a", "next", { terminalFailure: terminal });
  const failed = new NodeFailedRecord(FAILURE);
  wire.value.b.push(2n);

  expect(succeeded.evidence).toBe(evidence);
  expect(succeeded.forwardCapture).toBe(capture);
  expect(succeeded.value).toEqual({ kind: "object", value: { a: { file: wire.value.a.file }, b: [1n] } });
  expect(Object.keys(succeeded.value.value as object)).toEqual(["a", "b"]);
  expect(routed.terminalFailure).toEqual(terminal);
  expect(routed.terminalFailure).not.toBe(terminal);
  expect(Object.isFrozen(routed.terminalFailure)).toBe(true);
  expect(failed.failure).toEqual(FAILURE);
  expect(failed.failure).not.toBe(FAILURE);
  expect(Object.isFrozen(failed.failure)).toBe(true);
});

test("record constructors validate programmatically constructed values", () => {
  expectError(() => new NodeStartedRecord(3 as never), TypeError, "node_id must be a string");
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
    "evidence of another class",
    () => new NodeSucceededRecord("a", "a", { kind: "int", value: 1n }, { evidence: {} as never }),
    TypeError,
    "evidence must be an EvaluationEvidence or null",
  ],
  [
    "a capture of another class",
    () => new NodeExpandedRecord("grow", {}, [], TOPOLOGY_DIGEST, { forwardCapture: {} as never }),
    TypeError,
    "forward_capture must be a ForwardCapture or null",
  ],
  ["negative workers", () => new RunResumedRecord([], -1), RangeError, "num_workers must be non-negative"],
  ["fractional workers", () => new RunResumedRecord([], 1.5), TypeError, "num_workers must be an integer"],
  ["bigint workers", () => new RunResumedRecord([], 1n as never), TypeError, "num_workers must be an integer"],
  ["boolean workers", () => new RunResumedRecord([], true as never), TypeError, "num_workers must be an integer"],
  [
    "a non-text digest",
    () => new NodeExpandedRecord("grow", {}, [], 3 as never),
    TypeError,
    "topology_digest must be a string",
  ],
  [
    "a digest that isn't 64 lowercase hex digits",
    () => new NodeExpandedRecord("grow", {}, [], "B".repeat(64)),
    RangeError,
    "topology_digest must be 64 lowercase hex digits",
  ],
  [
    "an empty commit target",
    () => new NodeSucceededRecord("a", "", { kind: "int", value: 1n }),
    RangeError,
    "commit_target must be non-empty",
  ],
  ["an empty handle", () => new NodeRoutedRecord("a", ""), RangeError, "handle must be non-empty"],
  [
    "an empty spawner id",
    () => new NodeExpandedRecord("", {}, [], TOPOLOGY_DIGEST),
    RangeError,
    "spawner_id must be non-empty",
  ],
  [
    "a bad typed-value wire",
    () => new NodeSucceededRecord("a", "a", { kind: "int", value: "1" }),
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
    "a bad initial pool",
    () => runStarted({ initialPool: { store: {}, system: {} } as never }),
    RangeError,
    'invalid variable pool fields; missing=["start_id"], extra=[]',
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
    "memory must be non-empty",
  ],
  [
    "a non-text commit id",
    () => new MemoryCommitRequestedRecord("notes", 3 as never, [], 1),
    TypeError,
    "commit_id must be a string",
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
  [
    "a ready root result of another class",
    () => new BackwardReadyRecord(1, {} as never, [], []),
    TypeError,
    "result must be an exact BackwardResult",
  ],
  [
    "ready attributed results that aren't an array",
    () => new BackwardReadyRecord(1, new BackwardResult(), new Set() as never, []),
    TypeError,
    "attributed_results must be an array",
  ],
  [
    "ready proposals that aren't an array",
    () => new BackwardReadyRecord(1, new BackwardResult(), [], new Set() as never),
    TypeError,
    "proposals must be an array",
  ],
  [
    "a finished outcome of another class",
    () => new BackwardFinishedRecord(1, {} as never),
    TypeError,
    "outcome must be an exact BackwardOutcome",
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
  [
    "text prune ids",
    () => new NodeExpandedRecord("grow", {}, "ab" as never, TOPOLOGY_DIGEST),
    "prune_ids must be an array of strings",
  ],
  [
    "a non-text prune id",
    () => new NodeExpandedRecord("grow", {}, ["a", 2] as never, TOPOLOGY_DIGEST),
    "prune_ids must contain only non-empty strings",
  ],
  [
    "an empty prune id",
    () => new NodeExpandedRecord("grow", {}, [""], TOPOLOGY_DIGEST),
    "prune_ids must contain only non-empty strings",
  ],
  [
    "a repeated prune id",
    () => new NodeExpandedRecord("grow", {}, ["a", "a"], TOPOLOGY_DIGEST),
    "prune_ids must contain unique IDs",
  ],
  [
    "unsorted prune ids",
    () => new NodeExpandedRecord("grow", {}, ["b", "a"], TOPOLOGY_DIGEST),
    "prune_ids must be sorted",
  ],
])("record constructors reject invalid id arrays: %s", (_name, fn, message) => {
  expectError(fn, RangeError, message);
});

test.each<[string, unknown, string]>([
  ["7", 7, "unsupported run-record schema: 7"],
  ["8", 8, "unsupported run-record schema: 8"],
  ["9", 9, "run log schema 9 was written by Python yian; resume it with Python yian"],
  ["11", 11, "unsupported run-record schema: 11"],
  ["true", true, "unsupported run-record schema: true"],
  ["10n", 10n, "unsupported run-record schema: 10n"],
])("the envelope refuses schema %s", (_name, schema, message) => {
  expectError(() => new RecordEnvelope(schema as never, "run", 0, new NodeStartedRecord("work")), RangeError, message);
});

test("the envelope checks its schema before any other field", () => {
  expectError(
    () => new RecordEnvelope(9, "", -1, {} as never),
    RangeError,
    "run log schema 9 was written by Python yian; resume it with Python yian",
  );
});

test.each<[string, string, string]>([
  ["empty", "", "run-record envelope run_id must be a non-empty string"],
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
  const pool = new VariablePool("start");
  pool.set("start", { name: "Ada" });
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(pool),
    startWiringOverride: { name: "Ada" },
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test" } },
    numWorkers: 2,
  });
}

/** The three-channel result of the backward round trips. */
function backwardResult(): BackwardResult {
  return new BackwardResult({
    inputFeedback: { topic: [[null, { uri: "Review." }]] },
    selfFeedback: ["Behavior."],
    variableFeedback: { tone: ["Concise."] },
  });
}

/** The invoked occurrence of the backward round trips, with a root attribution. */
function backwardOccurrences(): AttributedBackwardResult[] {
  const root = new BackwardAttribution(null, null, 0, null, []);
  return [new AttributedBackwardResult(new SourceRef("run", 1), ["Review."], [root], backwardResult(), true)];
}

const ROUND_TRIP_RECORDS: [string, RunRecord][] = [
  ["RunStartedRecord", roundTripRunStarted()],
  ["RunResumedRecord", new RunResumedRecord(["slow", "later"], 4)],
  ["RunStoppedRecord", new RunStoppedRecord("callback", "the host callback threw", ["b", "a"])],
  ["NodeStartedRecord", new NodeStartedRecord("agent")],
  ["NodeStepRecord", new NodeStepRecord("agent", { event: "started", attempt: 2n })],
  ["NodeSucceededRecord", new NodeSucceededRecord("agent", "agent", { kind: "str", value: "done" })],
  ["NodeRoutedRecord", new NodeRoutedRecord("choice", "approved")],
  [
    "NodeExpandedRecord",
    new NodeExpandedRecord("each", [{ item: 1n }, { item: 2n }], ["each#old/end", "each#old/start"], TOPOLOGY_DIGEST),
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
  ["BackwardStartedRecord", new BackwardStartedRecord(1, [null, { uri: "Review." }])],
  [
    "BackwardReadyRecord",
    new BackwardReadyRecord(1, backwardResult(), backwardOccurrences(), [
      new MemoryProposal("agent", [new MemoryMutation("m1", { notes: ["one"] })]),
    ]),
  ],
  [
    "BackwardFinishedRecord",
    new BackwardFinishedRecord(
      1,
      new BackwardOutcome("succeeded", {
        feedback: backwardResult(),
        learning: new LearningResult("succeeded", {
          policies: [new PolicyCommitOutcome("history", "succeeded", { commitId: "commit-1" })],
        }),
      }),
      backwardOccurrences(),
    ),
  ],
];

test.each<[string, Record<string, unknown>]>([
  ["memory_commit_requested", { mutations: [] }],
  ["memory_commit_succeeded", {}],
  ["memory_commit_failed", { failure: { stage: "commit", code: "declined", message: "Declined.", retryable: true } }],
])("the memory commit wire requires a positive attempt: %s", (kind, extra) => {
  const wire = {
    schema: 10,
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
  ["an expansion seed", () => new NodeExpandedRecord("n", { limit: NaN }, [], "0".repeat(64)), ["limit"]],
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
    "a pool that isn't an object",
    (payload) => (payload["initial_pool"] = []),
    TypeError,
    "run_started payload initial_pool must be an object",
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
])("decode rejects a run_started payload with %s (TS-only)", (_name, edit, cls, message) => {
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
  const message = `run-record kind '${kind}' is unsupported in schema 10`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, message);
});

test.each([
  ["conversation_admitted", "ConversationAdmittedRecord"],
  ["conversation_effect_requested", "ConversationEffectRequestedRecord"],
  ["conversation_effect_observed", "ConversationEffectObservedRecord"],
  ["conversation_closed", "ConversationClosedRecord"],
])("removed conversation wire kinds are rejected: %s", (kind, recordName) => {
  const wire = { schema: SCHEMA_VERSION, run_id: "run", seq: 0, kind, payload: { checkpoint: {} } };
  const message = `run-record kind '${kind}' is unsupported in schema 10`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, message);
  expect(Object.hasOwn(runLog, recordName)).toBe(false);
});

test("record encoding is canonical, newline-free and byte-stable", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-canonical", 0, new NodeStartedRecord("work"));

  const encoded = encodeRecord(envelope);

  expect(encoded).toBe(
    '{"kind":"node_started","payload":{"node_id":"work"},"run_id":"run-canonical","schema":10,"seq":0}',
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
    schema: 10,
    run_id: "run-failure",
    seq: 7,
    kind: "node_failed",
    payload: {
      evidence: null,
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
    evidence: null,
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
    evidence: null,
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
  const item = new ForwardCapture({ x: 1n }, ValueRoute.record([["x", ValueRoute.literal(1n)]]), {
    evaluatedKeys: new Set(["x"]),
  });
  const capture = new ForwardCapture({ joined: [1n, 2n, 3n], text: "n=3", picked: 2n }, route, {
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
    new NodeSucceededRecord("work", "work", { kind: "int", value: 7n }, { forwardCapture: capture }),
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
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[[2,3]],"value_json":"\\"n=3\\""}';
  const pickedWire =
    `{"children":[${joinedWire}],"expression":"(a.items[0] + b)[1]","kind":"project","names":[],"path":[1],` +
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null}';
  const itemWire =
    '{"bound_inputs":{"x":1},"evaluated_keys":["x"],"input_names":[],"items":[],"node_data":null,' +
    `"route":{"children":[${literal}],"expression":"{x: 1}","kind":"record","names":["x"],"path":[],` +
    '"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},' +
    '"scope":null,"variable_names":[]}';
  expect(encoded).toBe(
    '{"kind":"node_succeeded","payload":{"commit_target":"work","evidence":null,"forward_capture":{' +
      '"bound_inputs":{"joined":[1,2,3],"picked":2,"text":"n=3"},"evaluated_keys":["joined","text"],' +
      '"input_names":[],' +
      `"items":[{"capture":${itemWire},"index":2}],"node_data":null,` +
      `"route":{"children":[${joinedWire},${textWire},${pickedWire}],` +
      '"expression":"{joined: (a.items[0] + b), text: n=${b}, picked: (a.items[0] + b)[1]}",' +
      '"kind":"record","names":["joined","text","picked"],"path":[],"result_type":null,"sizes":[],"snapshot":null,' +
      '"source":null,"spans":[],"value_json":null},"scope":{"boundary":null,"flow_path":["each"],' +
      '"occurrence_path":[{"index":3,"kind":"map_item","static_path":["each"]}],"static_path":["each","work"]},' +
      '"variable_names":[]},"node_id":"work","value":{"kind":"int","value":7}},"run_id":"run","schema":10,"seq":5}',
  );
  expect(decodeRecord(encoded)).toEqual(envelope);
  expect(encodeRecord(decodeRecord(encoded))).toBe(encoded);
});

test("decode requires an explicit route terminal failure field", () => {
  const wire = {
    schema: 10,
    run_id: "run",
    seq: 0,
    kind: "node_routed",
    payload: { node_id: "choice", handle: "default", evidence: null },
  };
  expectError(
    () => decodeRecord(JSON.stringify(wire)),
    RangeError,
    'invalid node_routed payload fields; missing=["forward_capture","terminal_failure"], extra=[]',
  );
});

test("decode rejects an extra nested route terminal failure field", () => {
  const wire = {
    schema: 10,
    run_id: "run",
    seq: 0,
    kind: "node_routed",
    payload: {
      evidence: null,
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
  ["schema 1", `{"schema":1,"run_id":"run","seq":0,${NODE_STARTED}}`, RangeError, "unsupported run-record schema: 1"],
  [
    "schema true",
    `{"schema":true,"run_id":"run","seq":0,${NODE_STARTED}}`,
    RangeError,
    "unsupported run-record schema: true",
  ],
  // A float token reads as a number, never as a schema; it is shown as the log writes floats.
  [
    "schema 10.0",
    `{"schema":10.0,"run_id":"run","seq":0,${NODE_STARTED}}`,
    RangeError,
    "unsupported run-record schema: 10.0",
  ],
  [
    "an empty run id",
    `{"schema":10,"run_id":"","seq":0,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope run_id must be a non-empty string",
  ],
  [
    "seq -1",
    `{"schema":10,"run_id":"run","seq":-1,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope seq must be a non-negative integer",
  ],
  [
    "seq true",
    `{"schema":10,"run_id":"run","seq":true,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope seq must be an integer",
  ],
  [
    "an unknown kind",
    '{"schema":10,"run_id":"run","seq":0,"kind":"not_a_record","payload":{}}',
    RangeError,
    "run-record kind 'not_a_record' is unsupported in schema 10",
  ],
  [
    "an extra key",
    `{"schema":10,"run_id":"run","seq":0,${NODE_STARTED},"unexpected":true}`,
    RangeError,
    'invalid run-record envelope fields; missing=[], extra=["unexpected"]',
  ],
  [
    "a missing payload",
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_started"}',
    RangeError,
    'invalid run-record envelope fields; missing=["payload"], extra=[]',
  ],
  // TS-only rows.
  [
    "seq 0.0",
    `{"schema":10,"run_id":"run","seq":0.0,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope seq must be an integer",
  ],
  [
    "seq past 2^53 - 1",
    `{"schema":10,"run_id":"run","seq":9007199254740992,${NODE_STARTED}}`,
    RangeError,
    "run-record envelope seq must be a safe integer",
  ],
  [
    "a non-text run id",
    `{"schema":10,"run_id":7,"seq":0,${NODE_STARTED}}`,
    TypeError,
    "run-record envelope run_id must be a non-empty string",
  ],
  [
    "a non-text kind",
    '{"schema":10,"run_id":"run","seq":0,"kind":7,"payload":{}}',
    TypeError,
    "run-record envelope kind must be a string",
  ],
  [
    "a payload that isn't an object",
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_started","payload":[]}',
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
    "a Python schema whatever else the line holds",
    '{"schema":9,"seq":"x","kind":"node_frobbed","extra":1}',
    RangeError,
    "run log schema 9 was written by Python yian; resume it with Python yian",
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
    { node_id: "work", commit_target: "work", value: { kind: "int", value: "7" } },
    RangeError,
    'invalid node_succeeded payload fields; missing=["evidence","forward_capture"], extra=[]',
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
    'invalid node_failed payload fields; missing=["evidence","forward_capture"], extra=["locator"]',
  ],
  // TS-only.
  [
    "an empty node id",
    "node_started",
    { node_id: "" },
    RangeError,
    "node_started payload node_id must be a non-empty string",
  ],
])("decode rejects unknown, missing and wrong payload fields: %s", (_name, kind, payload, cls, message) => {
  const wire = { schema: 10, run_id: "run", seq: 0, kind, payload };
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
    schema: 10,
    run_id: "run",
    seq: 0,
    kind: "node_failed",
    payload: {
      evidence: null,
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
  // TS-only.
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
  const message =
    schema === 9
      ? "run log schema 9 was written by Python yian; resume it with Python yian"
      : `unsupported run-record schema: ${schema}`;
  expectError(() => decodeRecord(JSON.stringify(wire)), RangeError, message);
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
    'invalid node_failed payload fields; missing=["evidence","failure","forward_capture"], ' +
      'extra=["error","error_type","locator","node_id","range","traceback"]',
  );
});

/** Assert decoding `data` throws RangeError `invalid run-record JSON`; return its cause. */
function invalidJsonCause(data: string | Uint8Array): Error {
  return expectError(() => decodeRecord(data), RangeError, "invalid run-record JSON").cause as Error;
}

test("decode rejects duplicate keys at any object depth", () => {
  const payloadKey = invalidJsonCause(
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"first","node_id":"second"}}',
  );
  expect(payloadKey).toBeInstanceOf(JSONDecodeError);
  expect(payloadKey.message).toStartWith('duplicate JSON object key: "node_id"');

  const originKey = invalidJsonCause(
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_failed",' +
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
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"\\ud800"}}',
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
    '{"schema":10,"run_id":"run","seq":0,"kind":"node_started","payload":{"node_id":"x"}}',
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
  const wire = { schema: 10, run_id: "run", seq: 0, kind: "run_stopped", payload };
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

test.each<[string, string, new (...args: never[]) => Error, string]>([
  [
    "float workers",
    '{"retry_node_ids":[],"num_workers":1.0}',
    TypeError,
    "run_resumed payload num_workers must be an integer",
  ],
  ["negative workers", '{"retry_node_ids":[],"num_workers":-1}', RangeError, "num_workers must be non-negative"],
  [
    "text retry ids",
    '{"retry_node_ids":"ab","num_workers":1}',
    TypeError,
    "run_resumed payload retry_node_ids must be an array of strings",
  ],
])("decode rejects a run_resumed payload with %s (TS-only)", (_name, payload, cls, message) => {
  const text = `{"schema":10,"run_id":"run","seq":0,"kind":"run_resumed","payload":${payload}}`;
  expectError(() => decodeRecord(text), cls, message);
});

const TAMPERED = ["payload", "run id", "sequence"];

test.each(TAMPERED)("the prepared record tamper check uses the frozen schema: %s", (change) => {
  const state = new RunState(flow());
  let prepared = prepareRunStarted(state, startData(state.flow));
  const before = stateSnapshot(state);
  requireUnchangedRecord(prepared, prepared._plan.envelopeJson);
  const { record, expectedRunId, expectedSequence, _plan: plan, observers } = prepared;
  if (change === "payload") {
    const started = record as RunStartedRecord;
    const tampered = new RunStartedRecord({
      ...(started as unknown as RunStartedFields),
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
  const prepared = prepareRunStarted(state, { ...startData(state.flow), learnable: true });
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
  expect(encodeRecord(log.records()[0]!)).toBe(prepared._plan.envelopeJson);
});
