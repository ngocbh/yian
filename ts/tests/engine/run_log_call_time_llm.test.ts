/**
 * `RunStarted.callTimeLlm`: the log half of `tests/engine/test_run_log_call_time_llm.py`, the
 * record of agents whose LLM config resolves when they run. Python's `ValueError` is a
 * `RangeError` here.
 *
 * Ported: `test_the_record_keeps_its_own_copy`, `test_a_malformed_record_is_rejected` (11 rows,
 * each message and class pinned; Python matches `call_time_llm` only).
 *
 * TS-only: the copy is deep-frozen with its keys in code-unit order, `-0` becomes `0`; omitted, the
 * field is `null`; the NaN row's `cause` is the value layer's `TypeCheckError`.
 *
 * Changed:
 * - `test_the_schema_stays_nine`: TS writes schema 10, so the constant is 10 (its encoded half
 *   comes with the record codec).
 * - The NaN row's message is the value layer's (`RunStarted call_time_llm: NaN is not a yian
 *   value`), where Python's canonical JSON copy says `non-finite numbers are not valid canonical
 *   JSON`.
 *
 * Later: the round-trip, golden-bytes and payload tests, with the record codec; the run-start
 * tests, with the run state.
 */

import { expect, test } from "bun:test";

import { encodePool, RunStartedRecord, SCHEMA_VERSION } from "../../src/runtime/log.ts";
import { TypeCheckError, VariablePool } from "../../src/typesys/index.ts";

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
  const pool = new VariablePool("start");
  pool.set("start", { name: "Ada" });
  pool.addSystem("run_id", "r1");
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(pool),
    startWiringOverride: { name: "${input.name}" },
    effectiveLlmByStaticPath: { agent: { provider: "openai", model: "gpt-test" } },
    numWorkers: 2,
    callTimeLlm: callTimeLlm as Record<string, unknown> | null,
  });
}

test("the schema is ten", () => {
  expect(SCHEMA_VERSION).toBe(10);
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

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
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
])("a malformed record is rejected: %s", (_name, value, cls, message) => {
  const caught = expectError(() => record(value), cls, message);
  if (message.endsWith("NaN is not a yian value")) expect(caught.cause).toBeInstanceOf(TypeCheckError);
});
