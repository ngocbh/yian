/**
 * The flow loader's plain-scalar rules: YAML 1.2 core schema plus dates and merges.
 *
 * The flow loader parses with eemeli/yaml under its failsafe schema, which leaves every
 * scalar as text, and resolves plain scalars with yian's own table: the YAML 1.2 core
 * schema, where `yes`/`no`/`on`/`off` and `1:30` are text and `017` is decimal, plus two
 * YAML 1.1 extensions, unquoted timestamps and `<<` merge keys. Ints are `bigint`, of any
 * length. `.inf` forms are infinities; `.nan` and a float that overflows are refused,
 * since neither is a yian value. A date or timestamp stays text, checked with yian's
 * `date`/`datetime` grammar, so a date-shaped plain scalar that is not a real date or
 * time, such as `2026-13-45` or an offset of `+05:99`, is an error that says to quote it.
 *
 * Knows about:   yaml, defaults, diagnostics, typesys.
 * Never imports: compose helpers or phases, nodes, compile, runtime, the run boundary,
 *                CLI modules.
 */

import type { Node } from "yaml";

import { reprText } from "../../diagnostics.ts";
import { parseDate, parseDatetime } from "../../typesys/temporal.ts";

export const STR_TAG = "tag:yaml.org,2002:str";
export const NULL_TAG = "tag:yaml.org,2002:null";
export const BOOL_TAG = "tag:yaml.org,2002:bool";
export const INT_TAG = "tag:yaml.org,2002:int";
export const FLOAT_TAG = "tag:yaml.org,2002:float";
export const TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp";
export const MERGE_TAG = "tag:yaml.org,2002:merge";
export const SEQ_TAG = "tag:yaml.org,2002:seq";
export const MAP_TAG = "tag:yaml.org,2002:map";

/**
 * The eemeli/yaml options the flow loader parses with: every scalar stays text for the
 * table below.
 */
export const FLOW_OPTIONS = {
  version: "1.2",
  schema: "failsafe",
  resolveKnownTags: false,
  uniqueKeys: false,
  prettyErrors: false,
} as const;

/** A scalar the flow loader builds: text, int, float, boolean, or null. */
export type ScalarValue = string | bigint | number | boolean | null;

// No `m` flag on any pattern, so `$` is the end of the text: a trailing newline fails,
// where Python's `$` would match before it.
const BOOL = /^(?:true|True|TRUE|false|False|FALSE)$/;
const INT = /^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$/;
const FLOAT = new RegExp(
  "^(?:[-+]?(?:\\.[0-9]+|[0-9]+(?:\\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?" +
    "|[-+]?\\.(?:inf|Inf|INF)" +
    "|\\.(?:nan|NaN|NAN))$",
);
const NULL = /^(?:~|null|Null|NULL|)$/;
const MERGE = /^<<$/;
// PyYAML's timestamp patterns share the time and offset after the date.
const TIME =
  "(?:[Tt]|[ \\t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\\.[0-9]*)?" +
  "(?:[ \\t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?";
// PyYAML's implicit resolver pattern for unquoted timestamps; a date alone needs a 2-digit
// month and day.
const TIMESTAMP = new RegExp(
  "^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}" +
    "|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}" + TIME + ")$",
);
// PyYAML's timestamp constructor pattern, where the month and day take 1 or 2 digits; the
// group captures the time, if any.
const TIMESTAMP_PARTS = new RegExp("^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(" + TIME + ")?$");

// Each pattern's first characters are exactly its PyYAML first-character bucket, so this
// ordered list is the same table; int precedes float because the float pattern also
// matches integers.
const IMPLICIT: readonly (readonly [RegExp, string])[] = [
  [BOOL, BOOL_TAG],
  [INT, INT_TAG],
  [FLOAT, FLOAT_TAG],
  [NULL, NULL_TAG],
  [MERGE, MERGE_TAG],
  [TIMESTAMP, TIMESTAMP_TAG],
];

/**
 * A YAML node the flow loader refuses; `node` is the node at fault, or `null` when the
 * caller knows it.
 */
export class FlowLoaderError extends RangeError {
  override name = "FlowLoaderError";
  readonly node: Node | null;

  constructor(message: string, node: Node | null = null) {
    super(message);
    this.node = node;
  }
}

/**
 * A plain scalar whose resolved tag can't build its text; quoting keeps it as text.
 * `reason` omits the text.
 */
export class ScalarValueError extends FlowLoaderError {
  override name = "ScalarValueError";
  readonly reason: string;

  constructor(reason: string, node: Node | null = null) {
    super(`${reason}; quote it to keep it as text`, node);
    this.reason = reason;
  }
}

/**
 * Return the tag the table resolves a plain scalar's text to.
 *
 * Example: `resolvePlain("017")` is `INT_TAG`, and `resolvePlain("yes")` is `STR_TAG`.
 */
export function resolvePlain(text: string): string {
  for (const [pattern, tag] of IMPLICIT) {
    if (pattern.test(text)) return tag;
  }
  return STR_TAG;
}

/**
 * Build a scalar's value from its text under its resolved or explicit tag.
 *
 * Explicit `!!int`, `!!float` and `!!bool` tags accept only the forms the table resolves.
 * A timestamp is checked with yian's date grammar and kept as its text. `plain` says the
 * scalar was unquoted, so a refused value says to quote it.
 *
 * Example: `constructScalar(INT_TAG, "017", true)` is `17n`.
 *
 * @throws ScalarValueError when `plain` and the tag can't build the text, such as `.nan`
 *   or `2026-13-45`.
 * @throws FlowLoaderError when the text is not the tag's core form, when a quoted value
 *   can't be built, or when the tag has no constructor.
 */
export function constructScalar(tag: string, text: string, plain: boolean): ScalarValue {
  const refuse = (reason: string): FlowLoaderError =>
    plain ? new ScalarValueError(reason) : new FlowLoaderError(reason);
  switch (tag) {
    case STR_TAG:
      return text;
    case NULL_TAG:
      return null;
    case BOOL_TAG:
      coreCheck(text, BOOL, "boolean");
      return text.toLowerCase() === "true";
    case INT_TAG:
      coreCheck(text, INT, "integer");
      // BigInt reads a decimal with leading zeros as decimal, and `0o`/`0x` as their bases.
      return BigInt(text);
    case FLOAT_TAG: {
      coreCheck(text, FLOAT, "float");
      const lowered = text.toLowerCase();
      if (lowered.endsWith(".inf")) return text[0] === "-" ? -Infinity : Infinity;
      if (lowered === ".nan") throw refuse("NaN is not a yian value");
      const value = Number(text);
      if (!Number.isFinite(value)) throw refuse("number is too large for a float");
      // yian has no negative zero, so `-0.0` and `-1e-400` read as 0.
      return value === 0 ? 0 : value;
    }
    case TIMESTAMP_TAG: {
      const match = TIMESTAMP_PARTS.exec(text);
      if (match === null) {
        throw new FlowLoaderError("expected a date or timestamp, but found " + reprText(text));
      }
      try {
        if (match[1] === undefined) parseDate(text);
        else parseDatetime(text);
      } catch (error) {
        if (error instanceof RangeError) throw refuse(error.message);
        throw error;
      }
      return text;
    }
    default:
      throw new FlowLoaderError("could not determine a constructor for the tag " + reprText(tag));
  }
}

/** Refuse `text` unless all of it is the YAML 1.2 core form `pattern` describes. */
function coreCheck(text: string, pattern: RegExp, kind: string): void {
  if (!pattern.test(text)) {
    throw new FlowLoaderError(`expected a YAML 1.2 core ${kind}, but found ${reprText(text)}`);
  }
}
