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
 * `composeFlowDocument` composes the text and resolves its aliases; a document nested deeper
 * than `current().maxYamlNesting` levels is a located `FlowLoaderError`.
 *
 * Knows about:   yaml, defaults, diagnostics, typesys.
 * Never imports: compose helpers or phases, nodes, compile, runtime, the run boundary,
 *                CLI modules.
 */

import {
  type Alias,
  type Document,
  isAlias,
  isMap,
  isSeq,
  type Node,
  type ParsedNode,
  parseDocument,
  type ParseOptions,
  type YAMLError,
} from "yaml";

import { current } from "../../defaults.ts";
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

/** One composed flow document and where each of its aliases points. */
export interface FlowDocument {
  readonly document: Document.Parsed;
  readonly aliasTargets: ReadonlyMap<Alias, Node>;
}

/**
 * Options a caller adds to `FLOW_OPTIONS`: a `LineCounter` for ranges, source tokens for
 * properties.
 */
export type FlowParseOptions = Pick<ParseOptions, "lineCounter" | "keepSourceTokens">;

/**
 * Compose `text` as one YAML document under `FLOW_OPTIONS` plus `options`, and map each alias
 * to the latest node anchored with its name before it.
 *
 * Example: in `[&a 1, &a 2, *a]`, `aliasTargets` maps the alias to the `2`.
 *
 * @throws YAMLError when the text is not valid YAML and its first error comes before any
 *   problem below.
 * @throws FlowLoaderError at an alias with no anchor before it, or at the first mapping or
 *   list nested deeper than `current().maxYamlNesting` levels.
 */
export function composeFlowDocument(text: string, options: FlowParseOptions = {}): FlowDocument {
  const document = parseDocument(text, { ...FLOW_OPTIONS, ...options });
  const limits = current();
  const anchors = new Map<string, ParsedNode>();
  const aliasTargets = new Map<Alias, Node>();
  // The check runs on the composed tree: eemeli catches its own stack overflow
  // (RESOURCE_EXHAUSTION) far deeper than the limit and keeps the levels above it, and the
  // order check below puts the nesting error first when it comes first in the text.
  let refused: { error: FlowLoaderError; at: number } | null = null;
  // Each entry: a node and its parent collection's level. Children are pushed in reverse,
  // so they pop in source order and an anchor is seen before the aliases after it.
  const stack: [ParsedNode, number][] = document.contents ? [[document.contents, 0]] : [];
  while (stack.length > 0) {
    const [node, parentLevel] = stack.pop()!;
    if (isAlias(node)) {
      const target = anchors.get(node.source);
      if (target === undefined) {
        const error = new FlowLoaderError("found undefined alias " + reprText(node.source), node);
        refused = { error, at: node.range[0] };
        break;
      }
      aliasTargets.set(node, target);
      continue;
    }
    // Registered before the children, so `&a [*a]` names its own list.
    if (node.anchor !== undefined) anchors.set(node.anchor, node);
    if (!isMap(node) && !isSeq(node)) continue;
    const level = parentLevel + 1;
    if (level > limits.maxYamlNesting) {
      const kind = isSeq(node) ? "list" : "mapping";
      const error = new FlowLoaderError(
        `this ${kind} nests deeper than ${limits.maxYamlNesting} levels, the limit for a ` +
          "YAML document (each mapping and list is a level; a yian value nests at most " +
          `${limits.maxValueNesting}); flatten it`,
        node,
      );
      refused = { error, at: node.range[0] };
      break;
    }
    const children = isSeq(node)
      ? node.items
      : node.items.flatMap((pair) => [pair.key, pair.value]);
    for (let i = children.length - 1; i >= 0; i--) {
      const child = children[i];
      if (child !== null && child !== undefined) stack.push([child as ParsedNode, level]);
    }
  }
  // The library's errors are not in source order.
  let first: YAMLError | undefined;
  for (const error of document.errors) {
    if (first === undefined || error.pos[0] < first.pos[0]) first = error;
  }
  if (refused !== null && (first === undefined || refused.at < first.pos[0])) {
    throw refused.error;
  }
  if (first !== undefined) throw first;
  return { document, aliasTargets };
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
