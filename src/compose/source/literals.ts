/** Parse isolated YAML literals outside source-document ownership. */

import {
  type Alias,
  isAlias,
  isMap,
  isScalar,
  isSeq,
  type Pair,
  type ParsedNode,
  Scalar,
  type YAMLMap,
  type YAMLSeq,
  YAMLError,
} from "yaml";

import { reprText } from "../../diagnostics.ts";
import type { Value, ValueRecord } from "../../typesys/index.ts";
import { setRecordKey, SourceSyntaxError } from "./common.ts";
import {
  type AnchoredNode,
  BOOL_TAG,
  composeFlowDocument,
  constructScalar,
  FLOAT_TAG,
  FlowLoaderError,
  INT_TAG,
  MAP_TAG,
  MERGE_TAG,
  NULL_TAG,
  resolvePlain,
  ScalarValueError,
  type ScalarValue,
  SEQ_TAG,
  STR_TAG,
  TIMESTAMP_TAG,
} from "./scalars.ts";

/** A composed mapping's pair. */
export type MapPair = Pair<ParsedNode, ParsedNode | null>;

/** The core scalar tags; on a list or mapping they say `expected a scalar node`. */
const SCALAR_TAGS: ReadonlySet<string> = new Set([
  STR_TAG,
  NULL_TAG,
  BOOL_TAG,
  INT_TAG,
  FLOAT_TAG,
  TIMESTAMP_TAG,
]);

/**
 * Safely parse one location-free YAML literal.
 *
 * Plain scalars read with the flow loader's scalar rules, the same as in a source document,
 * but no document graph is retained: input defaults need a value, not a graph. Mapping keys
 * must be text. An alias shares its target's value, so a recursive alias builds a recursive
 * list or record. A `<<` key merges a mapping, or a list of mappings, into its own mapping,
 * whose own keys win.
 *
 * Example: `parseYamlLiteral("{a: 017, b: yes}")` is `{a: 17n, b: "yes"}`.
 *
 * @throws SourceSyntaxError, with no range, when the text is not valid YAML for the flow
 *   loader.
 */
export function parseYamlLiteral(text: string): Value {
  try {
    const { document, aliasTargets } = composeFlowDocument(text);
    return construct(document.contents, aliasTargets);
  } catch (error) {
    if (error instanceof YAMLError || error instanceof FlowLoaderError) {
      throw new SourceSyntaxError(error.message);
    }
    throw error;
  }
}

/**
 * Read one plain scalar's text with the flow loader's scalar rules.
 *
 * The text is one stripped, single-line plain scalar, never YAML structure: the flow
 * loader's table picks its tag and builds the value, so `1e3` is a float and `~` is null
 * as in a source document, while `[1]`, `a: b`, or `&a` stay text. Text that spans lines,
 * and `<<`, are returned as they are.
 *
 * Example: `readPlainScalar("0x1F")` is `31n`, and `readPlainScalar("yes")` is `"yes"`.
 *
 * @throws SourceSyntaxError when the resolved tag can't build the text, such as an
 *   impossible date; the message is the reason, without the text or the quoting fix.
 */
export function readPlainScalar(text: string): ScalarValue {
  const tag = resolvePlain(text);
  if (tag === MERGE_TAG) return text; // `<<` merges only as a mapping key; as a value it is text
  try {
    return constructScalar(tag, text, true);
  } catch (error) {
    // The caller adds the quoting fix.
    if (error instanceof ScalarValueError) throw new SourceSyntaxError(error.reason);
    if (error instanceof FlowLoaderError) throw new SourceSyntaxError(error.message);
    throw error;
  }
}

/**
 * Build the value of a composed document's root, as SafeLoader constructs a document.
 *
 * A list or mapping is allocated when first reached and filled later, first-in first-out,
 * so a container's scalars are built (and their errors raised) before the containers nested
 * in it, and an alias to a container still being filled shares it.
 *
 * @throws FlowLoaderError when a node can't be built.
 */
function construct(root: ParsedNode | null, aliasTargets: ReadonlyMap<Alias, AnchoredNode>): Value {
  const values = new Map<ParsedNode, Value>();
  const fills: (() => void)[] = [];
  const target = (written: ParsedNode | null): AnchoredNode | null => {
    if (written === null) return null; // a missing key or value
    return isAlias(written) ? aliasTargets.get(written)! : written;
  };
  const valueOf = (written: ParsedNode | null): Value => {
    const node = target(written);
    if (node === null) return null;
    const built = values.get(node);
    if (built !== undefined) return built;
    // The tag picks the constructor: `!!seq` builds a list and `!!map` a record
    // whatever the node's kind.
    if (isScalar(node) && node.tag !== SEQ_TAG && node.tag !== MAP_TAG) {
      const value = scalarValue(node);
      values.set(node, value);
      return value;
    }
    if (!isScalar(node)) {
      const message = collectionTagError(node.tag, node);
      if (message !== null) throw new FlowLoaderError(message, node);
    }
    // SafeLoader's `construct_yaml_seq` and `construct_yaml_map` yield an empty container and
    // check the node's kind only when filling it, so a mismatch comes after sibling scalars.
    if (node.tag === SEQ_TAG || (node.tag !== MAP_TAG && isSeq(node))) {
      const list: Value[] = [];
      values.set(node, list);
      fills.push(() => {
        if (!isSeq(node)) {
          throw new FlowLoaderError(`expected a sequence node, but found ${nodeId(node)}`, node);
        }
        for (const item of node.items) list.push(valueOf(item));
      });
      return list;
    }
    const record: ValueRecord = {};
    values.set(node, record);
    fills.push(() => {
      if (!isMap(node)) {
        throw new FlowLoaderError(`expected a mapping node, but found ${nodeId(node)}`, node);
      }
      for (const pair of flatten(node)) {
        const key = valueOf(pair.key);
        if (typeof key !== "string") throw nonTextKey(key, target(pair.key));
        const value = valueOf(pair.value);
        // A repeated key keeps its first position and takes the last value, as in
        // SafeLoader.
        setRecordKey(record, key, value);
      }
    });
    return record;
  };
  const flattened = new Map<YAMLMap.Parsed, MapPair[]>();
  /**
   * Return the pairs SafeLoader's `flatten_mapping` leaves in `root`: each merge source's
   * flattened pairs, in the order of the merge keys, then the mapping's own other pairs.
   *
   * The walk is post-order over merge sources on an explicit stack, since a raised
   * `YIAN_MAX_YAML_NODES` admits merge chains deep enough to exhaust a recursive one. A
   * source still being flattened (a merge cycle) contributes its own other pairs.
   */
  const flatten = (root: YAMLMap.Parsed): MapPair[] => {
    const done = flattened.get(root);
    if (done !== undefined) return done;
    let pairs: MapPair[] = [];
    const stack: [YAMLMap.Parsed, Generator<YAMLMap.Parsed, YAMLMap.Parsed[]>][] = [];
    const merging = new Set<YAMLMap.Parsed>();
    const enter = (node: YAMLMap.Parsed): void => {
      merging.add(node);
      stack.push([node, mergeSources(node, target)]);
    };
    enter(root);
    for (let frame = stack.at(-1); frame !== undefined; frame = stack.at(-1)) {
      const [node, sources] = frame;
      const next = sources.next();
      if (!next.done) {
        if (!flattened.has(next.value) && !merging.has(next.value)) enter(next.value);
        continue;
      }
      stack.pop();
      merging.delete(node);
      pairs = next.value.flatMap((source) => flattened.get(source) ?? ownPairs(source, target));
      // A loop, not `push(...)`: spreading a document-sized array into call arguments
      // overflows the stack.
      for (const pair of ownPairs(node, target)) pairs.push(pair);
      flattened.set(node, pairs);
    }
    return pairs; // the root's: it leaves the stack last
  };
  const value = valueOf(root);
  // The loop also reaches the fills pushed while it runs: an array iterator rereads the
  // length at each step.
  for (const fill of fills) fill();
  return value;
}

/**
 * Build a scalar's value: an untagged plain scalar by the table, an untagged quoted or block
 * scalar, or one with the bare `!` tag, as its text, and a tagged one by its tag.
 */
function scalarValue(node: Scalar.Parsed): ScalarValue {
  const text = node.source;
  const plain = node.type === Scalar.PLAIN;
  switch (node.tag) {
    case undefined:
      return plain ? constructScalar(resolvePlain(text), text, true) : text;
    case "!":
      return text; // YAML 1.2: the bare tag keeps a scalar as text
    default:
      return constructScalar(node.tag, text, plain);
  }
}

/**
 * Return the error for the tag `tag` on a list or mapping, or null when the tag is absent, the
 * bare `!`, `!!seq`, or `!!map`.
 */
export function collectionTagError(
  tag: string | undefined,
  node: YAMLSeq.Parsed | YAMLMap.Parsed,
): string | null {
  if (tag === undefined || tag === "!" || tag === SEQ_TAG || tag === MAP_TAG) return null;
  return SCALAR_TAGS.has(tag)
    ? `expected a scalar node, but found ${nodeId(node)}`
    : "could not determine a constructor for the tag " + reprText(tag);
}

/**
 * Yield a mapping's merge sources as `flatten_mapping` reaches them, a list's items in their
 * written order, and return them in merge order, each list's items reversed so its first
 * mapping comes last and wins.
 *
 * At each merge pair, `checkSite` first sees the value as written (null when missing) and,
 * when it is a list written there, each of its items, with the pair's key, before any kind
 * check or yield for that pair.
 *
 * @throws FlowLoaderError at the first merge value that is not a mapping or a list of
 *   mappings.
 */
export function* mergeSources(
  node: YAMLMap.Parsed,
  target: (written: ParsedNode | null) => AnchoredNode | null,
  checkSite?: (written: ParsedNode | null, key: ParsedNode) => void,
): Generator<YAMLMap.Parsed, YAMLMap.Parsed[]> {
  const merged: YAMLMap.Parsed[] = [];
  for (const pair of node.items) {
    if (!isMergeKey(target(pair.key))) continue;
    if (checkSite !== undefined) {
      checkSite(pair.value, pair.key);
      if (isSeq(pair.value)) for (const item of pair.value.items) checkSite(item, pair.key);
    }
    // Past `checkSite`, the value's tag is ignored; only the node's kind
    // counts.
    const value = target(pair.value);
    if (isMap(value)) {
      yield value;
      merged.push(value);
    } else if (isSeq(value)) {
      const items: YAMLMap.Parsed[] = [];
      for (const item of value.items) {
        const source = target(item);
        if (!isMap(source)) {
          throw new FlowLoaderError(
            `expected a mapping for merging, but found ${nodeId(source)}`,
            source,
          );
        }
        yield source;
        items.push(source);
      }
      // A loop, not `push(...)`: spreading a document-sized array into call arguments
      // overflows the stack.
      for (const item of items.reverse()) merged.push(item);
    } else {
      // A missing value has no node, so the mapping stands in.
      throw new FlowLoaderError(mergeValueError(value), value ?? node);
    }
  }
  return merged;
}

/** Return the error for a merge value, `null` when missing, that is not a mapping or list. */
export function mergeValueError(value: AnchoredNode | null): string {
  return `expected a mapping or list of mappings for merging, but found ${nodeId(value)}`;
}

/** Return a mapping's pairs other than its merge pairs. */
export function ownPairs(
  node: YAMLMap.Parsed,
  target: (written: ParsedNode | null) => AnchoredNode | null,
): MapPair[] {
  return node.items.filter((pair) => !isMergeKey(target(pair.key)));
}

/**
 * Say whether a key is a merge key: a scalar tagged `!!merge`, or an untagged plain one
 * the table resolves to it. The bare `!` keeps `<<` text.
 */
export function isMergeKey(key: AnchoredNode | null): boolean {
  if (!isScalar(key)) return false;
  if (key.tag === undefined) {
    return key.type === Scalar.PLAIN && resolvePlain(key.source) === MERGE_TAG;
  }
  return key.tag === MERGE_TAG;
}

/** Name a node's kind (`scalar`, `sequence` or `mapping`); a missing node is an empty scalar. */
export function nodeId(node: AnchoredNode | null): string {
  return isSeq(node) ? "sequence" : isMap(node) ? "mapping" : "scalar";
}

/** Return the error for a mapping key that YAML read as `key`, not text, from `node`. */
function nonTextKey(key: Exclude<Value, string>, node: AnchoredNode | null): FlowLoaderError {
  if (typeof key === "object" && key !== null) {
    return new FlowLoaderError(
      "a list or mapping can't be a mapping key; mapping keys must be text",
      node,
    );
  }
  const written = isScalar(node) ? node.source : "";
  if (key === null && written === "") {
    return new FlowLoaderError(
      "an empty key was read by YAML as null, but mapping keys must be text",
      node,
    );
  }
  const readAs =
    key === null
      ? "null"
      : typeof key === "bigint"
        ? "an integer"
        : typeof key === "number"
          ? "a float"
          : "a boolean";
  return new FlowLoaderError(
    `the key \`${written}\` was read by YAML as ${readAs}, but mapping keys must be text; ` +
      "quote it",
    node,
  );
}
