/**
 * Reading typed values back from the run log: decoding a typed value from its log text, and the
 * system-value key prefix.
 *
 * Knows about: typesys.
 */

import { parseJson } from "../typesys/json.ts";
import { TypedValue } from "../typesys/values.ts";

/** @internal Decode a typed value from its canonical log text, a fresh copy. */
export function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
}

/** The key prefix of a system value's run-start commit: `system.<name>`. */
export const SYSTEM_PREFIX = "system.";
