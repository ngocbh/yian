/** Parse isolated YAML literals outside source-document ownership. */

import {
  type Alias,
  isAlias,
  isMap,
  isNode,
  isScalar,
  isSeq,
  type Node,
  Scalar,
  type YAMLMap,
  type YAMLSeq,
  YAMLError,
} from "yaml";

import { reprText } from "../../diagnostics.ts";
import type { Value, ValueRecord } from "../../typesys/index.ts";
import { SourceSyntaxError } from "./common.ts";
import {
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
 * list or record.
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
function construct(root: Node | null, aliasTargets: ReadonlyMap<Alias, Node>): Value {
  const values = new Map<Node, Value>();
  const fills: (() => void)[] = [];
  const target = (written: unknown): Node | null => {
    if (!isNode(written)) return null; // a missing key or value
    return isAlias(written) ? aliasTargets.get(written)! : written;
  };
  const valueOf = (written: unknown): Value => {
    const node = target(written);
    if (node === null) return null;
    const built = values.get(node);
    if (built !== undefined) return built;
    if (isScalar(node)) {
      const value = scalarValue(node);
      values.set(node, value);
      return value;
    }
    if (isSeq(node)) {
      checkCollectionTag(node, "sequence");
      const list: Value[] = [];
      values.set(node, list);
      fills.push(() => {
        for (const item of node.items) list.push(valueOf(item));
      });
      return list;
    }
    if (isMap(node)) {
      checkCollectionTag(node, "mapping");
      const record: ValueRecord = {};
      values.set(node, record);
      fills.push(() => {
        for (const pair of node.items) {
          const key = valueOf(pair.key);
          if (typeof key !== "string") throw nonTextKey(key, target(pair.key));
          const value = valueOf(pair.value);
          // A repeated key keeps its first position and takes the last value, as in
          // SafeLoader; defining `__proto__` keeps it data instead of setting the prototype.
          if (key === "__proto__") {
            Object.defineProperty(record, key, {
              value,
              writable: true,
              enumerable: true,
              configurable: true,
            });
          } else {
            record[key] = value;
          }
        }
      });
      return record;
    }
    // composeFlowDocument maps each alias to an anchored node, never to another alias.
    throw new Error("an alias names an alias");
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
function scalarValue(node: Scalar): ScalarValue {
  // A parsed scalar always has its source.
  const text = node.source!;
  const plain = node.type === Scalar.PLAIN;
  switch (node.tag) {
    case undefined:
      return plain ? constructScalar(resolvePlain(text), text, true) : text;
    case "!":
      return text; // YAML 1.2: the bare tag keeps a scalar as text
    case SEQ_TAG:
      throw new FlowLoaderError("expected a sequence node, but found scalar", node);
    case MAP_TAG:
      throw new FlowLoaderError("expected a mapping node, but found scalar", node);
    default:
      return constructScalar(node.tag, text, plain);
  }
}

/** Refuse a list's or mapping's tag unless it is absent, the bare `!`, or its own. */
function checkCollectionTag(node: YAMLSeq | YAMLMap, kind: "sequence" | "mapping"): void {
  const tag = node.tag;
  if (tag === undefined || tag === "!" || tag === (kind === "sequence" ? SEQ_TAG : MAP_TAG)) {
    return;
  }
  if (SCALAR_TAGS.has(tag)) {
    throw new FlowLoaderError(`expected a scalar node, but found ${kind}`, node);
  }
  if (tag === SEQ_TAG || tag === MAP_TAG) {
    const expected = tag === SEQ_TAG ? "sequence" : "mapping";
    throw new FlowLoaderError(`expected a ${expected} node, but found ${kind}`, node);
  }
  throw new FlowLoaderError("could not determine a constructor for the tag " + reprText(tag), node);
}

/** Return the error for a mapping key that YAML read as `key`, not text, from `node`. */
function nonTextKey(key: Exclude<Value, string>, node: Node | null): FlowLoaderError {
  if (typeof key === "object" && key !== null) {
    return new FlowLoaderError(
      "a list or mapping can't be a mapping key; mapping keys must be text",
      node,
    );
  }
  const written = isScalar(node) ? node.source! : "";
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
