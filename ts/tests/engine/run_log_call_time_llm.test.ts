/**
 * `RunStarted.callTimeLlm`: the log half of `tests/engine/test_run_log_call_time_llm.py`, the
 * record of agents whose LLM config resolves when they run. Python's `ValueError` is a
 * `RangeError` here.
 *
 * Ported: `test_the_record_keeps_its_own_copy`, `test_a_malformed_record_is_rejected` (11 rows,
 * each message and class pinned; Python matches `call_time_llm` only), `test_call_time_llm_round_trips`.
 *
 * TS-only: the copy is deep-frozen with its keys in code-unit order, `-0` becomes `0`; omitted, the
 * field is `null`; a NaN refusal's `cause` is the value layer's `TypeCheckError`.
 *
 * Changed:
 * - `test_the_schema_stays_nine`: TS writes schema 13, so the constant and the encoded line's
 *   `schema` are 13.
 * - The NaN row's message is the value layer's (`RunStarted call_time_llm: NaN is not a yian
 *   value`), where Python's canonical JSON copy says `non-finite numbers are not valid canonical
 *   JSON`.
 * - `test_a_run_without_call_time_agents_writes_the_same_bytes`: schema 13 always writes both
 *   fields, so the golden line holds `"call_time_llm":null,"defaults":{}` and `"schema":17`.
 * - `test_a_malformed_payload_is_rejected`: `null` is valid in schema 13 (no call-time agents), so
 *   its row is dropped; the other 10 rows are pinned with the record's class and message (a part of
 *   the wrong JSON type is a `TypeError`, where Python's pydantic raises `ValueError`).
 *
 * Ported, through the run state: `test_run_start_records_and_applies_call_time_llm`,
 * `test_run_start_without_call_time_agents_leaves_it_none` and
 * `test_run_start_requires_a_pristine_call_time_llm` (its message pinned).
 */

import { expect, test } from "bun:test";

import {
  decodeRecord,
  encodeRecord,
  MemoryRunLog,
  RecordEnvelope,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { commitPrepared, type PreparedTransition, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { TypeCheckError } from "../../src/typesys/index.ts";
import { type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import { flow } from "./run_state_fixtures.ts";

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

const CALL_TIME = {
  selection: {
    defaults: { provider: "openai", model: "gpt-saved" },
    provider_models: { openai: "gpt-saved" },
    connections: {
      openai: {
        method: "api_key",
        source: "missing",
        credential_id: null,
        endpoint: "https://api.openai.com/v1",
        store_path: null,
      },
    },
    api_endpoint: "https://api.openai.com/v1",
    stray_model: false,
    host: null,
  },
  agents: { chat: { provider: "openai" }, "c/a": {} },
};

/** Return a run-start record with the given call-time LLM record. */
function record(callTimeLlm?: unknown): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: { name: "${input.name}" },
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test" } },
    numWorkers: 2,
    callTimeLlm: callTimeLlm as Record<string, unknown> | null,
  });
}

/** Encode `record` as the first line of run `r1`. */
function encode(record: RunStartedRecord): string {
  return encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r1", 0, record));
}

// A run without call-time agents writes `call_time_llm` as null.
const GOLDEN =
  '{"kind":"run_started","payload":{"call_time_llm":null,"defaults":{},"effective_llm_by_static_path":{"agent":' +
  '{"model":"gpt-test","provider":"openai"}},"flow":null,"flow_fingerprint":"' +
  "a".repeat(64) +
  '","learnable":false,' +
  '"memories":null,"memory_snapshots":null,"memory_stacks":null,"num_workers":2,"parent":null,"skills":{},' +
  '"start_wiring_override":{"name":"${input.name}"},"tools":{},"yian_version":"0.0.0"},"run_id":"r1",' +
  '"schema":17,"seq":0}';

test("the schema is seventeen", () => {
  expect(SCHEMA_VERSION).toBe(17);
  expect((parseJson(encode(record(CALL_TIME)), "codec") as JsonObject)["schema"]).toBe(17n);
});

test("call_time_llm round trips", () => {
  const data = encode(record(structuredClone(CALL_TIME)));
  expect(((parseJson(data, "codec") as JsonObject)["payload"] as JsonObject)["call_time_llm"]).toEqual(CALL_TIME);
  const decoded = decodeRecord(data).record as RunStartedRecord;
  expect(decoded.callTimeLlm).toEqual(CALL_TIME);
  expect(decoded).toEqual(record(CALL_TIME));
  expect(encode(decoded)).toBe(data);
});

test("a run without call-time agents writes the same bytes", () => {
  const plain = record();
  expect(plain.callTimeLlm).toBeNull();
  expect(encode(plain)).toBe(GOLDEN);
  expect((decodeRecord(GOLDEN).record as RunStartedRecord).callTimeLlm).toBeNull();
});

test("the record keeps its own copy", () => {
  const given = structuredClone(CALL_TIME);
  const kept = record(given);
  given.agents.chat.provider = "anthropic";
  expect(kept.callTimeLlm).toEqual(CALL_TIME);
  expect(() => {
    (kept.callTimeLlm!["agents"] as { chat: { provider: string } }).chat.provider = "anthropic";
  }).toThrow(TypeError);
});

test("the record holds a canonical copy, and null when omitted", () => {
  const kept = record({ selection: {}, agents: { chat: { top: Infinity, temperature: -0 } } });
  expect(Object.keys(kept.callTimeLlm!)).toEqual(["agents", "selection"]);
  const chat = (kept.callTimeLlm!["agents"] as Record<string, Record<string, number>>)["chat"]!;
  expect(Object.keys(chat)).toEqual(["temperature", "top"]);
  expect(Object.is(chat["temperature"], 0)).toBe(true);
  expect(chat["top"]).toBe(Infinity);
  expect(record().callTimeLlm).toBeNull();
});

const NOT_EXACTLY = "RunStarted call_time_llm must have exactly selection and agents";

const MALFORMED: [string, unknown, new (...args: never[]) => Error, string][] = [
  ["an array", [], TypeError, "RunStarted call_time_llm must be a dict"],
  ["an empty object", {}, RangeError, NOT_EXACTLY],
  ["no selection", { agents: CALL_TIME.agents }, RangeError, NOT_EXACTLY],
  ["no agents", { selection: CALL_TIME.selection }, RangeError, NOT_EXACTLY],
  ["an extra field", { ...CALL_TIME, extra: 1n }, RangeError, NOT_EXACTLY],
  ["no agent", { ...CALL_TIME, agents: {} }, RangeError, "RunStarted call_time_llm agents must not be empty"],
  [
    "a base that isn't an object",
    { ...CALL_TIME, agents: { chat: "openai" } },
    TypeError,
    "RunStarted call_time_llm base for 'chat' must be a dict",
  ],
  [
    "an empty agent path",
    { ...CALL_TIME, agents: { "": {} } },
    RangeError,
    "RunStarted call_time_llm agent paths must be non-empty strings",
  ],
  ["agents in an array", { ...CALL_TIME, agents: [] }, TypeError, "RunStarted call_time_llm agents must be a dict"],
  [
    "a text selection",
    { ...CALL_TIME, selection: "x" },
    TypeError,
    "RunStarted call_time_llm selection must be a dict",
  ],
  [
    "a NaN",
    { ...CALL_TIME, agents: { chat: { temperature: NaN } } },
    RangeError,
    "RunStarted call_time_llm: NaN is not a yian value",
  ],
];

test.each(MALFORMED)("a malformed record is rejected: %s", (_name, value, cls, message) => {
  expectError(() => record(value), cls, message);
});

// The log can't hold a NaN, so its row has no wire form.
const MALFORMED_WIRE = MALFORMED.filter(([name]) => name !== "a NaN");

test.each(MALFORMED_WIRE)("a malformed payload is rejected: %s", (_name, value, cls, message) => {
  const wire = parseJson(encode(record()), "codec") as { payload: Record<string, unknown> };
  wire.payload["call_time_llm"] = value;
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test("a NaN in the record keeps the value layer's error as its cause", () => {
  const caught = expectError(
    () => record({ ...CALL_TIME, agents: { chat: { temperature: NaN } } }),
    RangeError,
    "RunStarted call_time_llm: NaN is not a yian value",
  );
  expect(caught.cause).toBeInstanceOf(TypeCheckError);
});

/** Prepare the start of run `r1` over `state` with an empty START record and `callTimeLlm`. */
function start(state: RunState, callTimeLlm: Record<string, unknown> | null): PreparedTransition {
  return prepareRunStarted(state, {
    runId: "r1",
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test" } },
    numWorkers: 1,
    callTimeLlm,
  });
}

test("run start records and applies call_time_llm", () => {
  const state = new RunState(flow());
  expect(state.callTimeLlm).toBeNull();
  const authority = new MemoryRunLog("r1");
  commitPrepared(authority, state, start(state, structuredClone(CALL_TIME)));
  expect(state.callTimeLlm).toEqual(CALL_TIME);
  expect((authority.records()[0]!.record as RunStartedRecord).callTimeLlm).toEqual(CALL_TIME);
});

test("run start without call-time agents leaves it null", () => {
  const state = new RunState(flow());
  commitPrepared(new MemoryRunLog("r1"), state, start(state, null));
  expect(state.callTimeLlm).toBeNull();
});

test("run start requires a pristine call_time_llm", () => {
  const state = new RunState(flow());
  state.callTimeLlm = structuredClone(CALL_TIME);
  expectError(() => start(state, null), RangeError, "RunStarted requires a pristine RunState at sequence 0");
});
