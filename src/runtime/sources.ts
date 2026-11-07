/**
 * The values a binding reads from the pool and the producers they come from: the source entries
 * of accepted values, the system-value key prefix, and the root read with its source.
 *
 * Typed values are held as their canonical log text, and each read decodes a fresh value.
 *
 * Knows about:   expr/lineage, nodes/contracts, typesys.
 * Never imports: runtime state, engine or run, compile, compose, or the CLI.
 */

import { type RootRead, type RoutePath, SourceRef, sameSource } from "../expr/lineage.ts";
import { sameOptional } from "../nodes/contracts.ts";
import { type JsonValue, parseJson } from "../typesys/json.ts";
import { referenceRoot, type VariablePool } from "../typesys/pool.ts";
import {
  isList,
  isRecord,
  normalizeTypedFiles,
  type Type,
  TypedValue,
  typeEquals,
  type Value,
} from "../typesys/values.ts";

/** One value `rebind` reads for a source: the typed value a pool name held, and the producer it names. */
export interface ReboundSource {
  /** The value the pool name held when the binding read it. */
  readonly value: TypedValue;
  /** The producer the input route names for it, or `null` for none. */
  readonly source: SourceRef | null;
}

/** @internal Decode a typed value from its canonical log text, a fresh copy. */
export function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
}

/**
 * A typed accepted value and its producer identity read as one unit. Frozen.
 *
 * `files` is the type the commit target reads the value through to restore its files, as the pool
 * does; `null` reads the value as stored.
 */
export class SourceEntry {
  /**
   * The producer's accepted start; for an AGENT's new messages, its messages commit; `null` for an
   * initial pool value without one.
   */
  readonly source: SourceRef | null;
  /** The sequence of the record that accepted the value. */
  readonly outcomeSeq: number;
  /** The value's canonical typed wire `{kind, value}` as log text. */
  readonly valueJson: string;
  /** The type the value is read through to restore its files, or `null`. */
  readonly files: Type | null;
  /**
   * The value's JSON, parsed on the first `requirePath` and kept: each item of a wide MAP checks
   * a path into the same list, which parsing per check made quadratic.
   */
  #json: { value: JsonValue } | null = null;

  /** Hold the entry's fields as given. */
  constructor(source: SourceRef | null, outcomeSeq: number, valueJson: string, files: Type | null = null) {
    this.source = source;
    this.outcomeSeq = outcomeSeq;
    this.valueJson = valueJson;
    this.files = files;
    Object.freeze(this);
  }

  /** A fresh decoded value with the pool's semantics: files restored through `files`. */
  get value(): Value {
    return normalizeTypedFiles(this.files, decodeTyped(this.valueJson)).toObject();
  }

  /** A fresh decoded value as stored, each nested file as its JSON form. */
  get storedValue(): Value {
    return decodeTyped(this.valueJson).toObject();
  }

  /**
   * Require `path` inside the value.
   *
   * @throws RangeError when the path is absent.
   */
  requirePath(path: RoutePath): void {
    this.#json ??= parseJson(this.valueJson, "codec") as { value: JsonValue };
    atPath(this.#json.value, path);
  }
}

/**
 * @internal The JSON at `path` inside `value`.
 *
 * @throws RangeError when the path is absent.
 */
export function atPath(value: JsonValue, path: RoutePath): JsonValue {
  for (const part of path) {
    if (typeof part === "string" && isRecord(value) && Object.hasOwn(value, part)) {
      value = value[part] as JsonValue;
    } else if (typeof part === "number" && isList(value) && part >= 0 && part < value.length) {
      value = value[part] as JsonValue;
    } else {
      throw new RangeError("source path is absent from its accepted value");
    }
  }
  return value;
}

/** Whether two entries have the same source (by key), outcome sequence, value text and files type. */
export function sameSourceEntry(a: SourceEntry, b: SourceEntry): boolean {
  return (
    sameOptional(a.source, b.source, sameSource) &&
    a.outcomeSeq === b.outcomeSeq &&
    a.valueJson === b.valueJson &&
    sameOptional(a.files, b.files, typeEquals)
  );
}

/** The key prefix of a system value's run-start commit: `system.<name>`. */
export const SYSTEM_PREFIX = "system.";

/**
 * Read a reference's root in `pool` (`VariablePool.readRoot`) with the source `sourceOf` names for
 * the pool name it reads (`referenceRoot`); an `input.<k>` root's source is the START's, at path
 * `[k]`. The evaluator walks the names after the root.
 *
 * @throws whatever `sourceOf` throws.
 */
export function resolvePoolPath(
  names: readonly string[],
  pool: VariablePool,
  sourceOf: (name: string) => SourceRef | null,
): RootRead {
  const read = pool.readRoot(names);
  if (!read.present) return read;
  const root = referenceRoot(names, pool.startId, (name) => pool.store.has(name));
  const source = sourceOf(root.name!);
  const path = root.key === undefined ? [] : [root.key];
  return { ...read, source: source === null ? null : new SourceRef(source.runId, source.executionSeq, path) };
}
