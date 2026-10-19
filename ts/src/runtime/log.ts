/**
 * Closed run-record vocabulary and canonical JSON codecs.
 *
 * The dispatcher is the sole producer of these durable records. Records contain only transition
 * data; their enclosing `RecordEnvelope` owns run identity, schema version and sequence. This
 * module deliberately knows nothing about the scheduler or reducer, so the same validated
 * carriers can back memory and JSONL logs without creating an authority cycle.
 *
 * Knows about:   diagnostics, expr/lineage, memory/contracts, nodes/base, nodes/contracts,
 *                platform, runtime/contracts, typesys.
 * Never imports: state, trace, engine, run, compile, compose, or the CLI.
 */

import { type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import {
  buildValue,
  buildValueAs,
  type PoolWire,
  type Type,
  TypedValue,
  type ValueKind,
  VariablePool,
} from "../typesys/index.ts";
import { jsonPart } from "../typesys/values.ts";

/** The only run-record schema that this version writes and reads. */
export const SCHEMA_VERSION = 10;

/** A typed value's wire object `{kind, value}`. */
type TypedWire = ReturnType<TypedValue["toWire"]>;

/**
 * Canonicalize a node value through its durable typed JSON representation.
 *
 * `value` is a live node output, or an existing `TypedValue`; `declared` is the target output
 * type, or `null` to infer the natural storage type (`buildValue`; else `buildValueAs`).
 *
 * Returns the canonical typed wire object (a fresh JSON copy, keys in code-unit order, each
 * `FileRef` as its mapping) and the value decoded from an independent copy of that exact
 * object.
 *
 * @throws TypeCheckError at the rejected value's path when the value doesn't satisfy
 *   `declared`, or isn't made of JSON values and `FileRef`s (NaN, text holding a lone surrogate,
 *   a container inside itself, a `Date`, a `Set`, ...).
 */
export function canonicalTyped(value: unknown, declared: ValueKind | Type | null): [TypedWire, TypedValue] {
  const typed = declared === null ? buildValue(value) : buildValueAs(declared, value);
  const wire = canonicalCopy(typed.toWire()) as TypedWire;
  return [wire, TypedValue.fromWire(canonicalCopy(wire))];
}

/**
 * Canonicalize a seed value (builder input or invocation data) as JSON.
 *
 * Returns independent, mutable canonical JSON copies, the wire and the decoded value, keys in
 * code-unit order; each `FileRef` becomes its mapping and `±Infinity` stays a float. Live
 * expansion building must consume the decoded second value.
 *
 * @throws TypeCheckError at the rejected value's path when the value isn't a JSON value (see
 *   `jsonPart`): NaN, text holding a lone surrogate, a container inside itself, a `Date`, ...
 */
export function canonicalSeed(value: unknown): [JsonValue, JsonValue] {
  const wire = canonicalCopy(jsonPart(value, [], "mapping"));
  return [wire, canonicalCopy(wire)];
}

/**
 * Encode a variable pool to its canonical JSON object, a `RunStartedRecord`'s `initial_pool`.
 *
 * @throws TypeCheckError when a stored value isn't a JSON value (see `TypedValue.toWire`).
 * @throws RangeError when a stored typed value doesn't hold its kind (`TypedValue.fromWire`).
 */
export function encodePool(pool: VariablePool): PoolWire {
  const wire = canonicalCopy(pool.toWire()) as unknown as PoolWire;
  // The unchecked `TypedValue` constructor can hold a value of the wrong kind.
  decodePool(wire);
  return wire;
}

/**
 * Decode and strictly validate a canonical variable-pool wire object.
 *
 * `wire` must have exactly `store`, `system` and `start_id` (see `VariablePool.fromWire`). The
 * pool's values are decoded from a fresh copy, so they share nothing with `wire`.
 *
 * @throws RangeError when the object has missing or extra fields or invalid typed values.
 * @throws TypeError when it holds data outside canonical JSON (the writer's error).
 */
export function decodePool(wire: PoolWire): VariablePool {
  return VariablePool.fromWire(canonicalCopy(wire));
}

/**
 * Return a fresh canonical copy of `value`, keys in code-unit order, as the run log reads it back.
 *
 * @throws TypeError when `value` isn't canonical JSON (the writer's error).
 */
function canonicalCopy(value: unknown): JsonValue {
  return parseJson(logJson(value), "codec");
}

/**
 * Freeze `root` and every container in it, in place, and return it.
 *
 * `root` must be a fresh value `parseJson` returned, so it is a tree of plain containers; nothing
 * is checked. An explicit stack walks it, so any depth works.
 */
function freezeParsed<T extends JsonValue>(root: T): T {
  const stack: JsonValue[] = [root];
  while (stack.length > 0) {
    const value = stack.pop()!;
    if (typeof value !== "object" || value === null) continue;
    Object.freeze(value);
    for (const item of Object.values(value)) stack.push(item);
  }
  return root;
}
