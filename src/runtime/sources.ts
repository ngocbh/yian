/**
 * The values a binding reads from the pool and the producers they come from: decoding a typed
 * value from its log text, the system-value key prefix, and the root read with its source.
 *
 * Knows about: expr/lineage, typesys.
 */

import { type RootRead, SourceRef } from "../expr/lineage.ts";
import { parseJson } from "../typesys/json.ts";
import { referenceRoot, type VariablePool } from "../typesys/pool.ts";
import { TypedValue } from "../typesys/values.ts";

/** @internal Decode a typed value from its canonical log text, a fresh copy. */
export function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
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
