/**
 * Reading one plain scalar on its own, and the merge and collection-tag rules a document's build
 * pass follows as SafeLoader does.
 */

import { isMap, isScalar, isSeq, type Pair, type ParsedNode, Scalar, type YAMLMap, type YAMLSeq } from "yaml";

import { reprText } from "../../diagnostics.ts";
import type { Value } from "../../typesys/index.ts";
import { SourceSyntaxError } from "./common.ts";
import {
  type AnchoredNode,
  BOOL_TAG,
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

/**
 * Return why a mapping key that YAML read as `key`, not text, was refused; `written` is the key
 * as written, empty for an empty key.
 */
export function nonTextKeyMessage(key: Exclude<Value, string>, written: string): string {
  if (key === null && written === "") return "an empty key was read by YAML as null, but mapping keys must be text";
  const readAs =
    key === null
      ? "null"
      : typeof key === "bigint"
        ? "an integer"
        : typeof key === "number"
          ? "a float"
          : "a boolean";
  return `the key \`${written}\` was read by YAML as ${readAs}, but mapping keys must be text; quote it`;
}
