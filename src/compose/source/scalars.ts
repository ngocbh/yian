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
 * than `current().maxYamlNesting` levels, or one that expands to more than
 * `current().maxYamlNodes` nodes when each alias counts as a copy of what it names, is a
 * located `FlowLoaderError`.
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
  isNode,
  isSeq,
  type Node,
  type ParsedNode,
  parseDocument,
  type ParseOptions,
  type Scalar,
  type YAMLError,
  type YAMLMap,
  type YAMLSeq,
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

/** A node an alias can name. */
export type AnchoredNode = Scalar.Parsed | YAMLMap.Parsed | YAMLSeq.Parsed;

// No `m` flag on any pattern, so `$` is the end of the text: a trailing newline fails.
const BOOL = /^(?:true|True|TRUE|false|False|FALSE)$/;
const INT = /^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$/;
const FLOAT = new RegExp(
  "^(?:[-+]?(?:\\.[0-9]+|[0-9]+(?:\\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?" +
    "|[-+]?\\.(?:inf|Inf|INF)" +
    "|\\.(?:nan|NaN|NAN))$",
);
const NULL = /^(?:~|null|Null|NULL|)$/;
const MERGE = /^<<$/;
// The YAML 1.1 timestamp patterns share the time and offset after the date.
const TIME =
  "(?:[Tt]|[ \\t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\\.[0-9]*)?" +
  "(?:[ \\t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?";
// The YAML 1.1 implicit pattern for unquoted timestamps; a date alone needs a 2-digit
// month and day.
const TIMESTAMP = new RegExp(
  "^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}" +
    "|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}" + TIME + ")$",
);
// The timestamp parts pattern, where the month and day take 1 or 2 digits; the
// group captures the time, if any.
const TIMESTAMP_PARTS = new RegExp("^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(" + TIME + ")?$");

// The patterns are tried in this order; int precedes float because the float pattern also
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
  readonly aliasTargets: ReadonlyMap<Alias, AnchoredNode>;
}

/**
 * Options a caller adds to `FLOW_OPTIONS`: a `LineCounter` for ranges, source tokens for
 * properties.
 */
export type FlowParseOptions = Pick<ParseOptions, "lineCounter" | "keepSourceTokens">;

/** Return `text` with each lone CR turned into LF: the same length, so offsets hold. */
export function normalizeBreaks(text: string): string {
  return text.replace(/\r(?!\n)/g, "\n");
}

/**
 * Compose `normalizeBreaks(text)` as one YAML document under `FLOW_OPTIONS` plus `options`,
 * and map each alias to the latest node anchored with its name before it. eemeli/yaml breaks
 * lines only at LF and CRLF, so a lone CR would otherwise stay inside a scalar.
 *
 * Example: in `[&a 1, &a 2, *a]`, `aliasTargets` maps the alias to the `2`.
 *
 * @throws YAMLError when the text is not valid YAML and its first error comes before any
 *   problem below.
 * @throws FlowLoaderError at an alias with no anchor before it, or at the first mapping or
 *   list nested deeper than `current().maxYamlNesting` levels. Otherwise at the first mapping
 *   or list, in post-order, that holds more than `current().maxYamlNodes` nodes, counting
 *   each alias as a copy of what it names (each mapping, list, and scalar, mapping keys
 *   included, counts once per place it appears); under a recursive alias, at the deepest
 *   node on the path already over the limit once the count would need repeating.
 */
export function composeFlowDocument(text: string, options: FlowParseOptions = {}): FlowDocument {
  const document = parseDocument(normalizeBreaks(text), { ...FLOW_OPTIONS, ...options });
  const limits = current();
  const anchors = new Map<string, AnchoredNode>();
  const aliasTargets = new Map<Alias, AnchoredNode>();
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
      if (child !== null && child !== undefined) stack.push([child, level]);
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

  // Aliases let a short document name exponentially many nodes, and every later step
  // (construction, walking, the run log) costs per expanded node, so the composed
  // graph is counted before anything else touches it.
  //
  // A recursive alias is a back edge to a node still on the path. It counts 1 (the
  // alias itself), and the source layer reports it when something descends through
  // it. Its count makes every node on the cycle, from its target down to it,
  // path-dependent: reached from outside the cycle, the loader's walkers expand that
  // target in full, and reached from inside it, the target is the back edge. So a node
  // is memoized only when no back edge below it reaches it or above it (its `low`, the
  // shallowest path depth such an edge reaches, is deeper than its own depth), and a
  // path-dependent node is counted again at each visit. Every step adds at least 1 to
  // `counted`, the nodes counted so far, so once that passes the limit the document
  // is over it; a recount then would only repeat work, so the walk stops there and
  // reports the deepest node on the path already holding more than the limit.
  const root = document.contents;
  if (root !== null) {
    const limit = limits.maxYamlNodes;
    const counts = new Map<Node, number>();
    const pathDependent = new Set<Node>();
    // The depth of each node on the path.
    const depths = new Map<Node, number>([[root, 0]]);
    let counted = 1;
    let over: Node | null = null;
    // Each frame: the node, its children, the next child's index, its count so far,
    // and its `low`, which starts one deeper than the node.
    const stack: [
      node: Node,
      children: (Node | null)[],
      index: number,
      total: number,
      low: number,
    ][] = [
      [root, countedChildren(root, aliasTargets), 0, 1, 1],
    ];
    while (stack.length > 0) {
      const frame = stack.at(-1)!;
      const [node, nodeChildren, index, total, low] = frame;
      if (index < nodeChildren.length) {
        frame[2] = index + 1;
        const child = nodeChildren[index] ?? null;
        // A missing key or value is an empty scalar: one node, with no children.
        if (child === null) {
          frame[3] = total + 1;
          counted += 1;
        } else if (counts.has(child)) {
          const count = counts.get(child)!;
          frame[3] = total + count;
          counted += count;
        } else if (depths.has(child)) {
          frame[3] = total + 1;
          frame[4] = Math.min(low, depths.get(child)!);
          counted += 1;
        } else if (counted > limit && pathDependent.has(child)) {
          let partial = 0;
          for (let i = stack.length - 1; i >= 0; i--) {
            const held = stack[i]!;
            partial += held[3];
            if (partial > limit) {
              over = held[0];
              break;
            }
          }
          break;
        } else {
          depths.set(child, stack.length);
          stack.push([child, countedChildren(child, aliasTargets), 0, 1, stack.length + 1]);
          counted += 1;
        }
        continue;
      }
      stack.pop();
      depths.delete(node);
      if (low > stack.length) counts.set(node, total);
      else pathDependent.add(node);
      if (total > limit) {
        over = node;
        break;
      }
      const parent = stack.at(-1);
      if (parent !== undefined) {
        parent[3] += total;
        parent[4] = Math.min(parent[4], low);
      }
    }
    if (over !== null) {
      const kind = isSeq(over) ? "list" : "mapping";
      throw new FlowLoaderError(
        `this ${kind} holds more than ${limit} YAML nodes, counting each alias as a copy ` +
          `of what it names; the limit for a YAML document is ${limit} ` +
          "(YIAN_MAX_YAML_NODES; each mapping, list, and scalar is a node)",
        over,
      );
    }
  }
  return { document, aliasTargets };
}

/**
 * Return a node's children for the node count: a list's items, or each pair's key and
 * value, with each alias replaced by the node it names and `null` for a missing key or
 * value; none for a scalar.
 */
function countedChildren(node: Node, aliasTargets: ReadonlyMap<Alias, Node>): (Node | null)[] {
  const items: unknown[] = isSeq(node)
    ? node.items
    : isMap(node)
      ? node.items.flatMap((pair) => [pair.key, pair.value])
      : [];
  return items.map((item) => {
    const child = isNode(item) ? item : null;
    return isAlias(child) ? aliasTargets.get(child)! : child;
  });
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
 * A timestamp is checked with yian's date grammar and kept as its text. `plain` means the
 * scalar was written unquoted, so a refused value says to quote it. Under an explicit tag the
 * quoting hint is still given, although quoting doesn't remove the tag.
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
      try {
        return BigInt(text);
      } catch (error) {
        // A runtime refuses an enormous int with its own RangeError.
        if (error instanceof RangeError) throw refuse("integer is too large for this runtime");
        throw error;
      }
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
