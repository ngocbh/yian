/**
 * One-pass parsing and private retention of a Compose YAML document.
 *
 * Lone CRs read as LF, eemeli/yaml composes the text (via `composeFlowDocument`), the CST
 * locates properties, and every reachable node becomes one frozen definition. Values are yian
 * values with text keys: a key YAML reads as non-text, such as `1:` or `true:`, is kept as
 * written, and the document reports it.
 */

import {
  type Alias,
  CST,
  isAlias,
  isMap,
  isScalar,
  isSeq,
  type Node,
  Parser,
  type ParsedNode,
  Scalar,
  YAMLError,
  type YAMLMap,
} from "yaml";

import { reprText, SourceRange } from "../../diagnostics.ts";
import type { Value, ValueRecord } from "../../typesys/index.ts";
import {
  type ArcSequence,
  type Definition,
  freezeValue,
  type MapArc,
  type MapOccurrence,
  type OccurrencePath,
  type ScalarDef,
  type ScalarStyle,
  setRecordKey,
  SourceSyntaxError,
} from "./common.ts";
import { SourceCursor } from "./cursor.ts";
import {
  collectionTagError,
  isMergeKey,
  type MapPair,
  mergeSources,
  nodeId,
  ownPairs,
} from "./literals.ts";
import {
  type AnchoredNode,
  composeFlowDocument,
  constructScalar,
  type FlowDocument,
  FlowLoaderError,
  MAP_TAG,
  MERGE_TAG,
  normalizeBreaks,
  resolvePlain,
  type ScalarValue,
  SEQ_TAG,
  STR_TAG,
} from "./scalars.ts";

/**
 * The hint added to a YAML error at a `,`, `[`, `]`, `{` or `}` glued to a tag, which ends the
 * tag there: a type with several arguments, or written with brackets, can't be a tag.
 */
export const TAG_HINT =
  "hint: a YAML tag ends at `,`, `[`, `]`, `{` or `}`; name a type with several arguments in " +
  "`typedefs:` and tag with its name (e.g. `Decision: Union(Buy, Sell)`, then `!Decision`), " +
  "and write `list(str)`, not `list[str]`";

/** The first key a mapping repeats among its own pairs. */
export interface RepeatedKey {
  readonly key: string;
  readonly first: SourceRange;
  readonly second: SourceRange;
}

/** The first mapping key, in document order, that YAML read as non-text. */
export interface NonTextKey {
  readonly key: Exclude<ScalarValue, string>;
  readonly text: string;
  readonly range: SourceRange;
}

// PyYAML's printable characters; with the `u` flag a lone surrogate is one code point,
// outside every range here.
const NON_PRINTABLE = /[^\t\n\r\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\u{10000}-\u{10FFFF}]/u;

const STYLES: Readonly<Record<Scalar.Type, ScalarStyle>> = {
  [Scalar.PLAIN]: "plain",
  [Scalar.QUOTE_SINGLE]: "single",
  [Scalar.QUOTE_DOUBLE]: "double",
  [Scalar.BLOCK_LITERAL]: "literal",
  [Scalar.BLOCK_FOLDED]: "folded",
};

const NO_PARTS: readonly never[] = Object.freeze([]);

const CORE_TAG_PREFIX = "tag:yaml.org,2002:";

// A tag glued to the flow indicator that ends a line's text.
const GLUED_TAG = /(?:^|[\s[{,])![^\s,[\]{}]*[,[\]{}]$/;

const ONLY_SPACES = /^ *$/;

/**
 * Offsets from `start` to the exclusive `end`; `bare` says the node's own properties hold the
 * bare tag `!`.
 */
interface Span {
  readonly start: number;
  readonly end: number;
  readonly bare: boolean;
}

/** Return a composed node's span: its range, from its anchor or tag when it has one. */
type SpanOf = (node: Node) => Span;

/**
 * A rope of a mapping's raw pairs, as `ArcSequence` is of arcs: a leaf (pairs, no parts) or a
 * concatenation (parts at partOffsets, each part's first ordinal, no pairs).
 */
interface PairSequence {
  readonly pairs: readonly MapPair[];
  readonly parts: readonly PairSequence[];
  readonly partOffsets: readonly number[];
  readonly flatPairCount: number;
}

/** A mapping rope's keys by first occurrence, and each key's last occurrence. */
interface Summary {
  readonly order: readonly string[];
  readonly last: ReadonlyMap<string, MapOccurrence>;
}

/** An arc rope and its summary. */
interface ArcRope {
  readonly sequence: ArcSequence;
  readonly summary: Summary;
}

const EMPTY_PAIRS: PairSequence = {
  pairs: NO_PARTS,
  parts: NO_PARTS,
  partOffsets: NO_PARTS,
  flatPairCount: 0,
};

/**
 * One safely constructed YAML document with its source graph kept private.
 *
 * `text` is the YAML as written and `label` names it to later diagnostics. The constructed
 * value and the definition graph stay private, and no eemeli/yaml tree is kept.
 */
export class SourceDocument {
  readonly text: string;
  readonly label: string;
  /**
   * The first key a mapping repeats among its own pairs, with the first pair's key range
   * and the repeating pair's. YAML lookup keeps the last pair; Compose refuses the document.
   */
  readonly repeatedKey: RepeatedKey | null;
  /**
   * The first mapping key, in document order, that YAML read as non-text: the key YAML
   * built, its text as written, and its range. Flow names must be text, so Compose refuses
   * the document.
   */
  readonly nonTextKey: NonTextKey | null;
  /** @internal The constructed value, deep-frozen. */
  readonly _value: Value;
  /** @internal Every reachable node's definition, by id. */
  readonly _definitions: readonly Definition[];
  /** @internal The root's definition id. */
  readonly _rootDefinition: number;

  private constructor(
    text: string,
    label: string,
    definitions: readonly Definition[],
    repeatedKey: RepeatedKey | null,
    nonTextKey: NonTextKey | null,
  ) {
    this.text = text;
    this.label = label;
    this.repeatedKey = repeatedKey;
    this.nonTextKey = nonTextKey;
    this._definitions = Object.freeze(definitions);
    this._rootDefinition = 0;
    this._value = definitions[this._rootDefinition]!.value;
    Object.freeze(this);
  }

  /** A cursor at the document root. */
  get root(): SourceCursor {
    return SourceCursor._physical(this, this._rootDefinition);
  }

  /**
   * @internal Replay an occurrence path from the root, one step at a time.
   *
   * @throws SourceAccessError when a step doesn't fit the document.
   */
  _resolveOccurrence(path: OccurrencePath): SourceCursor {
    let cursor = this.root;
    for (const step of path) cursor = cursor._descendOccurrenceStep(step);
    return cursor;
  }

  /**
   * Compose and construct exactly one YAML document in one pass.
   *
   * @throws SourceSyntaxError when the text holds a character YAML refuses, is not one valid
   *   YAML document, or holds a node the flow loader can't build or a tag that is bare or
   *   neither local nor core; `range` locates the problem. A YAML error at a character glued
   *   to a tag ends with `TAG_HINT`.
   */
  static parse(text: string, label = "<flow>"): SourceDocument {
    const source = normalizeBreaks(text);
    const starts = lineStarts(source);
    const refused = NON_PRINTABLE.exec(source);
    if (refused !== null) {
      const code = source.codePointAt(refused.index)!;
      throw new SourceSyntaxError(
        `unacceptable character #x${code.toString(16).padStart(4, "0")}: special characters ` +
          "are not allowed",
        pointRange(starts, refused.index),
      );
    }
    const spanOf = propertySpans(source);
    let flow: FlowDocument;
    let built: Built;
    try {
      flow = composeFlowDocument(source);
      // A merge error comes from `mergeSources`, a `FlowLoaderError`.
      built = build(flow, source, starts, spanOf);
    } catch (error) {
      if (error instanceof YAMLError) {
        let message =
          error.code === "MULTIPLE_DOCS"
            ? "expected a single document in the stream, but found another document"
            : error.message;
        if (gluedToTag(error, source)) message += "\n" + TAG_HINT;
        throw new SourceSyntaxError(message, pointRange(starts, error.pos[0]));
      }
      if (error instanceof FlowLoaderError) {
        const node = error.node;
        const range = node?.range ? pointRange(starts, spanOf(node).start) : null;
        throw new SourceSyntaxError(error.message, range);
      }
      throw error;
    }
    const { definitions, ids, nonTextKey } = built;
    return new SourceDocument(
      text,
      label,
      definitions,
      firstRepeatedKey(flow, ids, definitions),
      nonTextKey,
    );
  }
}

/** What one pass over a composed document builds. */
interface Built {
  readonly definitions: Definition[];
  readonly ids: ReadonlyMap<Node, number>;
  readonly nonTextKey: NonTextKey | null;
}

/**
 * Intern every reachable node of `flow` once, as SafeLoader constructs a document.
 *
 * A list or mapping is allocated when first reached and filled later, first-in first-out, so
 * a container's scalars are built (and their errors raised) before the containers nested in
 * it, and an alias to a container still being filled shares it. The root is definition 0.
 *
 * `source` is the normalized text the CST and `starts` index.
 *
 * @throws SourceSyntaxError at a scalar the flow loader can't build, at a list or mapping key,
 *   at a bare tag or one neither local nor core, or at a core tag that can't build its node.
 */
function build(
  flow: FlowDocument,
  source: string,
  starts: readonly number[],
  spanOf: SpanOf,
): Built {
  const { document, aliasTargets } = flow;
  const ids = new Map<Node, number>();
  if (document.contents === null) {
    const empty: ScalarDef = {
      kind: "scalar",
      value: null,
      range: null,
      raw: null,
      constructedLineRanges: null,
      style: null,
      tag: null,
    };
    return { definitions: [Object.freeze(empty)], ids, nonTextKey: null };
  }
  const definitions: (Definition | null)[] = [];
  const values: Value[] = [];
  const fills: (() => void)[] = [];
  let nonTextKey: NonTextKey | null = null;
  let nonTextKeyAt = 0;

  const reserve = (node: AnchoredNode, value: Value): number => {
    const id = definitions.length;
    ids.set(node, id);
    definitions.push(null);
    values.push(value);
    return id;
  };
  /** Intern a missing value, as in `{a}`: an empty plain scalar at offset `at`. */
  const internMissing = (at: number): number => {
    const id = definitions.length;
    definitions.push(
      Object.freeze({
        kind: "scalar",
        value: null,
        range: toRange(starts, at, at),
        raw: "",
        constructedLineRanges: null,
        style: "plain",
        tag: null,
      }),
    );
    values.push(null);
    return id;
  };
  const targetOf = (written: ParsedNode | null): AnchoredNode | null =>
    written === null ? null : target(aliasTargets, written);
  /**
   * Refuse a tag written at a `<<:` merge site, as Python's `_reject_merge_source_tag`: YAML
   * never builds a merge source, so a local tag there checks nothing. An alias is skipped: its
   * target's tag sits at the anchor and checks the value there.
   */
  const checkSite = (written: ParsedNode | null): void => {
    if (written === null || isAlias(written)) return;
    const span = spanOf(written);
    const range = toRange(starts, span.start, span.end);
    if (localTag(written, span, range) !== null) {
      throw new SourceSyntaxError(
        "a tag on a `<<:` merge source checks nothing; tag the mapping that merges it",
        range,
      );
    }
  };
  const ownLeaves = new Map<YAMLMap.Parsed, PairSequence>();
  /** Return a mapping's non-merge pairs as a leaf, one object per mapping. */
  const ownLeaf = (map: YAMLMap.Parsed): PairSequence => {
    let leaf = ownLeaves.get(map);
    if (leaf === undefined) {
      leaf = pairLeaf(ownPairs(map, targetOf));
      ownLeaves.set(map, leaf);
    }
    return leaf;
  };
  const flattened = new Map<YAMLMap.Parsed, PairSequence>();
  /**
   * Return the pairs SafeLoader's `flatten_mapping` leaves in `root` as a rope that shares each
   * merge source's: the sources' ropes in merge order, then the mapping's own leaf.
   *
   * The walk is post-order over merge sources on an explicit stack, as `literals.ts`'s
   * `flatten`, since a raised node limit admits merge chains too deep for a recursive one. A
   * source still being flattened (a merge cycle) contributes its own leaf.
   *
   * @throws SourceSyntaxError at a tag written at a merge site, and FlowLoaderError at a merge
   *   value that is not a mapping or a list of mappings.
   */
  const flatten = (root: YAMLMap.Parsed): PairSequence => {
    const done = flattened.get(root);
    if (done !== undefined) return done;
    const stack: [YAMLMap.Parsed, Generator<YAMLMap.Parsed, YAMLMap.Parsed[]>][] = [];
    const merging = new Set<YAMLMap.Parsed>();
    const enter = (node: YAMLMap.Parsed): void => {
      merging.add(node);
      stack.push([node, mergeSources(node, targetOf, checkSite)]);
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
      const parts = next.value.map((source) => flattened.get(source) ?? ownLeaf(source));
      parts.push(ownLeaf(node));
      flattened.set(node, concat(parts));
    }
    return flattened.get(root)!;
  };
  /**
   * Intern a leaf's pairs, each key before its value, as an arc leaf.
   *
   * @throws SourceSyntaxError at a list or mapping key, or at a node that can't be built.
   */
  const leafArcs = (pairs: readonly MapPair[]): ArcSequence => {
    const arcs: MapArc[] = [];
    for (const pair of pairs) {
      const keyNode = target(aliasTargets, pair.key);
      if (!isScalar(keyNode)) {
        throw new SourceSyntaxError(
          "a list or mapping can't be a mapping key; mapping keys must be text",
          pointRange(starts, spanOf(keyNode).start),
        );
      }
      const keyDefinition = intern(keyNode);
      // A missing value sits where the key ends.
      const valueDefinition =
        pair.value === null ? internMissing(pair.key.range[1]) : intern(pair.value);
      const keyScalar = definitions[keyDefinition] as ScalarDef;
      const key = keyText(keyScalar);
      // Fills don't run in document order, so the key that starts first wins; an alias key
      // is its anchored node.
      const keyAt = spanOf(keyNode).start;
      if (typeof keyScalar.value !== "string" && (nonTextKey === null || keyAt < nonTextKeyAt)) {
        const keyValue = keyScalar.value as NonTextKey["key"];
        nonTextKey = { key: keyValue, text: key, range: keyScalar.range! };
        nonTextKeyAt = keyAt;
      }
      arcs.push(Object.freeze({ key, keyDefinition, valueDefinition }));
    }
    return Object.freeze({
      arcs: Object.freeze(arcs),
      parts: NO_PARTS,
      partOffsets: NO_PARTS,
      flatPairCount: arcs.length,
    });
  };
  const arcRopes = new Map<PairSequence, ArcRope>();
  /**
   * Return a pair rope as an arc rope of the same shape with its summary (Python's
   * `effective_sequence` and its first-key order), post-order on an explicit stack and cached
   * per rope: each distinct leaf is interned once, leaves in first-occurrence order, and a
   * shared rope is summarized once.
   */
  const arcRope = (pairSequence: PairSequence): ArcRope => {
    const stack: [PairSequence, boolean][] = [[pairSequence, false]];
    while (stack.length > 0) {
      const [current, ready] = stack.pop()!;
      if (arcRopes.has(current)) continue;
      if (current.parts.length === 0) {
        const sequence = leafArcs(current.pairs);
        arcRopes.set(current, { sequence, summary: leafSummary(sequence.arcs) });
      } else if (!ready) {
        stack.push([current, true]);
        for (let i = current.parts.length - 1; i >= 0; i--) stack.push([current.parts[i]!, false]);
      } else {
        const parts = current.parts.map((part) => arcRopes.get(part)!);
        const sequence: ArcSequence = Object.freeze({
          arcs: NO_PARTS,
          parts: Object.freeze(parts.map((part) => part.sequence)),
          partOffsets: Object.freeze(current.partOffsets),
          flatPairCount: current.flatPairCount,
        });
        arcRopes.set(current, { sequence, summary: concatSummary(sequence, parts) });
      }
    }
    return arcRopes.get(pairSequence)!;
  };
  const intern = (written: ParsedNode): number => {
    const node = target(aliasTargets, written);
    const known = ids.get(node);
    if (known !== undefined) return known;
    const span = spanOf(node);
    const range = toRange(starts, span.start, span.end);
    const at = pointRange(starts, span.start);
    const tag = localTag(node, span, range);
    // A local tag reads its node as if untagged.
    const core = tag === null ? node.tag : undefined;
    // The tag picks the constructor, as in PyYAML: `!!seq` builds a list and `!!map` a record
    // whatever the node's kind.
    const picksCollection = core === SEQ_TAG || core === MAP_TAG;
    if (isScalar(node) && !picksCollection) {
      const text = node.source;
      const plain = node.type === Scalar.PLAIN;
      let value: ScalarValue;
      try {
        value = scalarValue(text, plain, core, tag !== null);
      } catch (error) {
        if (error instanceof FlowLoaderError) throw new SourceSyntaxError(error.message, at);
        throw error;
      }
      const id = reserve(node, value);
      definitions[id] = Object.freeze({
        kind: "scalar",
        value,
        range,
        raw: text,
        constructedLineRanges: literalLineRanges(source, starts, node, value),
        style: STYLES[node.type!],
        tag,
      });
      return id;
    }
    if (!isScalar(node)) {
      const message = collectionTagError(core, node);
      if (message !== null) throw new SourceSyntaxError(message, at);
    }
    // SafeLoader's `construct_yaml_seq` and `construct_yaml_map` yield an empty container and
    // check the node's kind only when filling it, so a mismatch comes after sibling scalars.
    if (picksCollection ? core === SEQ_TAG : isSeq(node)) {
      const list: Value[] = [];
      const id = reserve(node, list);
      fills.push(() => {
        if (!isSeq(node)) {
          throw new SourceSyntaxError(`expected a sequence node, but found ${nodeId(node)}`, at);
        }
        const itemDefinitions: number[] = [];
        for (const item of node.items) {
          const itemId = intern(item);
          itemDefinitions.push(itemId);
          list.push(values[itemId]!);
        }
        definitions[id] = Object.freeze({
          kind: "sequence",
          value: list,
          range,
          itemDefinitions: Object.freeze(itemDefinitions),
          tag,
        });
      });
      return id;
    }
    const record: ValueRecord = {};
    const id = reserve(node, record);
    fills.push(() => {
      if (!isMap(node)) {
        throw new SourceSyntaxError(`expected a mapping node, but found ${nodeId(node)}`, at);
      }
      // Flattened first, so every merge error comes before any pair is interned.
      const { sequence: pairSequence, summary } = arcRope(flatten(node));
      const { order, last } = summary;
      // A repeated key keeps its first position and takes the last value, as in SafeLoader.
      for (const key of order) {
        setRecordKey(record, key, values[last.get(key)!.arc.valueDefinition]!);
      }
      definitions[id] = Object.freeze({
        kind: "mapping",
        value: record,
        range,
        pairSequence,
        effectiveIndex: last,
        tag,
      });
    });
    return id;
  };

  intern(document.contents);
  // The loop also reaches the fills pushed while it runs: an array iterator rereads the
  // length at each step.
  for (const fill of fills) fill();
  const built = definitions as Definition[];
  // A value only a repeated key's earlier pair holds is not reachable from the root, so
  // every definition's value is frozen, the root's first.
  for (const definition of built) {
    if (typeof definition.value === "object" && !Object.isFrozen(definition.value)) {
      freezeValue(definition.value);
    }
  }
  return { definitions: built, ids, nonTextKey };
}

/** Return `pairs` as a leaf rope; no pairs is the empty rope. */
function pairLeaf(pairs: readonly MapPair[]): PairSequence {
  if (pairs.length === 0) return EMPTY_PAIRS;
  return { pairs, parts: NO_PARTS, partOffsets: NO_PARTS, flatPairCount: pairs.length };
}

/** Return the concatenation of `parts` without the empty ones: the only one, when one is left. */
function concat(parts: readonly PairSequence[]): PairSequence {
  const kept = parts.filter((part) => part.flatPairCount > 0);
  if (kept.length === 0) return EMPTY_PAIRS;
  if (kept.length === 1) return kept[0]!;
  const partOffsets: number[] = [];
  let count = 0;
  for (const part of kept) {
    partOffsets.push(count);
    count += part.flatPairCount;
  }
  return { pairs: NO_PARTS, parts: kept, partOffsets, flatPairCount: count };
}

/** Return an arc leaf's summary: its keys by first occurrence, and each key's last arc. */
function leafSummary(arcs: readonly MapArc[]): Summary {
  const order: string[] = [];
  const last = new Map<string, MapOccurrence>();
  for (const [index, arc] of arcs.entries()) {
    if (!last.has(arc.key)) order.push(arc.key);
    last.set(arc.key, Object.freeze({ arc, pairOrdinal: index }));
  }
  return { order, last };
}

/**
 * Return a concatenation's summary from its parts, each distinct part read once: a key's first
 * occurrence is in the leftmost part that holds it, its last in the rightmost.
 */
function concatSummary(sequence: ArcSequence, parts: readonly ArcRope[]): Summary {
  const order: string[] = [];
  const known = new Set<string>();
  const firstParts = new Set<ArcRope>();
  for (const part of parts) {
    if (firstParts.has(part)) continue;
    firstParts.add(part);
    for (const key of part.summary.order) {
      if (known.has(key)) continue;
      known.add(key);
      order.push(key);
    }
  }
  const last = new Map<string, MapOccurrence>();
  const lastParts = new Set<ArcRope>();
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i]!;
    if (lastParts.has(part)) continue;
    lastParts.add(part);
    const offset = sequence.partOffsets[i]!;
    for (const [key, occurrence] of part.summary.last) {
      if (last.has(key)) continue;
      const pairOrdinal = occurrence.pairOrdinal + offset;
      last.set(key, Object.freeze({ arc: occurrence.arc, pairOrdinal }));
    }
  }
  return { order, last };
}

/**
 * Return the first key a mapping repeats among its own pairs, in a pre-order walk; two pairs
 * keyed by one alias target are not a repeat.
 */
function firstRepeatedKey(
  flow: FlowDocument,
  ids: ReadonlyMap<Node, number>,
  definitions: readonly Definition[],
): RepeatedKey | null {
  const { document, aliasTargets } = flow;
  const stack: ParsedNode[] = document.contents === null ? [] : [document.contents];
  const seen = new Set<Node>();
  while (stack.length > 0) {
    const written = stack.pop()!;
    const node = target(aliasTargets, written);
    if (seen.has(node)) continue;
    seen.add(node);
    if (isSeq(node)) {
      for (let i = node.items.length - 1; i >= 0; i--) stack.push(node.items[i]!);
      continue;
    }
    if (!isMap(node)) continue;
    const firstKeys = new Map<string, Node>();
    for (const pair of node.items) {
      const keyNode = target(aliasTargets, pair.key);
      // Keys a merge brings in override; only a mapping's own pairs count.
      if (isMergeKey(keyNode)) continue;
      const keyScalar = definitions[ids.get(keyNode)!] as ScalarDef;
      const key = keyText(keyScalar);
      const first = firstKeys.get(key);
      if (first === undefined) {
        firstKeys.set(key, keyNode);
      } else if (first !== keyNode) {
        const firstScalar = definitions[ids.get(first)!]!;
        return { key, first: firstScalar.range!, second: keyScalar.range! };
      }
    }
    for (let i = node.items.length - 1; i >= 0; i--) {
      const pair = node.items[i]!;
      if (pair.value !== null) stack.push(pair.value);
      stack.push(pair.key);
    }
  }
  return null;
}

/**
 * Lex `source` and return each composed node's span.
 *
 * The composed tree keeps no property positions, so the CST records them. Each span of
 * properties is stored twice: under the offset where the properties end, to find an empty
 * node, and under the `tokenKey` of the token they precede, whose collection flag tells a
 * block mapping from its first key, which shares its offset. An empty node has no token, or a
 * zero-length `scalar` one before a `,` in a flow collection; eemeli places it after its
 * properties and any spaces that follow them.
 */
function propertySpans(source: string): SpanOf {
  const byEnd = new Map<number, Span>();
  const byToken = new Map<string, Span>();
  const note = (props: readonly CST.SourceToken[], token?: CST.Token | null): void => {
    const span = propertiesSpan(props);
    if (span === null) return;
    byEnd.set(span.end, span);
    if (token) byToken.set(tokenKey(token.offset, CST.isCollection(token)), span);
  };
  const stack: CST.Token[] = [];
  const document = [...new Parser().parse(source)].find(
    (token): token is CST.Document => token.type === "document",
  );
  if (document !== undefined) {
    // The root's properties sit in the document's start.
    note(document.start, document.value);
    if (document.value) stack.push(document.value);
  }
  while (stack.length > 0) {
    const token = stack.pop()!;
    if (token.type === "block-seq") {
      for (const item of token.items) {
        note(item.start, item.value);
        if (item.value) stack.push(item.value);
      }
    } else if (token.type === "block-map" || token.type === "flow-collection") {
      for (const item of token.items) {
        if (item.key !== undefined || item.sep !== undefined) {
          note(item.start, item.key);
          note(afterValueIndicator(item.sep ?? []), item.value);
        } else {
          note(item.start, item.value);
        }
        if (item.key) stack.push(item.key);
        if (item.value) stack.push(item.value);
      }
      // A block mapping starts at its first key's properties, as `&k a: 1`, unless its own
      // come earlier; after an explicit `?` they follow the mapping's start.
      const first = token.type === "block-map" ? token.items[0] : undefined;
      const keySpan = first === undefined ? null : propertiesSpan(first.start);
      if (keySpan !== null && keySpan.start < token.offset) {
        const key = tokenKey(token.offset, true);
        const known = byToken.get(key);
        if (known === undefined || known.start > keySpan.start) {
          // A bare `!` on the first key (`! a: 1`) belongs to the key, so the mapping keeps
          // only its own `bare` flag.
          byToken.set(key, { ...keySpan, bare: known?.bare ?? false });
        }
      }
    }
  }
  const spanOf: SpanOf = (node) => {
    const [start, end] = node.range!;
    if (start === end) {
      let propertiesEnd = start;
      while (
        propertiesEnd > 0 &&
        (source[propertiesEnd - 1] === " " || source[propertiesEnd - 1] === "\t")
      ) {
        propertiesEnd--;
      }
      const none: Span = { start, end: start, bare: false };
      return byToken.get(tokenKey(start, false)) ?? byEnd.get(propertiesEnd) ?? none;
    }
    // Only a token with properties has an entry.
    const own = byToken.get(tokenKey(start, isMap(node) || isSeq(node)));
    let from = own?.start ?? start;
    // A one-pair mapping in a flow sequence, as `[&k a: 1]`, has no token, and eemeli starts
    // it at its key, after the key's properties; a braced `{...}` mapping already starts
    // before its key, so `Math.min` leaves it unchanged.
    const key = isMap(node) && node.flow ? node.items[0]?.key : undefined;
    if (isScalar(key)) from = Math.min(from, spanOf(key).start);
    return { start: from, end, bare: own?.bare ?? false };
  };
  return spanOf;
}

/** Return the key under which `byToken` records the properties before a token. */
function tokenKey(offset: number, collection: boolean): string {
  return `${offset}:${collection}`;
}

/** Return the span of the anchor and tag among `props`, or null when there are none. */
function propertiesSpan(props: readonly CST.SourceToken[]): Span | null {
  const marks = props.filter((token) => token.type === "anchor" || token.type === "tag");
  const first = marks[0];
  const last = marks[marks.length - 1];
  if (first === undefined || last === undefined) return null;
  const bare = marks.some((token) => token.type === "tag" && token.source === "!");
  return { start: first.offset, end: last.offset + last.source.length, bare };
}

/** Return the tokens after the last `:` indicator in `sep`; none without one. */
function afterValueIndicator(sep: readonly CST.SourceToken[]): readonly CST.SourceToken[] {
  const at = sep.findLastIndex((token) => token.type === "map-value-ind");
  return at === -1 ? [] : sep.slice(at + 1);
}

/**
 * Return a node's local tag without its `!`, or null when it has none or a core tag.
 *
 * @throws SourceSyntaxError over `range` at the bare tag `!`, which eemeli keeps on a scalar
 *   but resolves away on a list or mapping, or at a tag neither local nor core.
 */
function localTag(node: AnchoredNode, span: Span, range: SourceRange): string | null {
  const tag = node.tag;
  if (isScalar(node) ? tag === "!" : span.bare) {
    throw new SourceSyntaxError("a tag names a type, e.g. `!int`", range);
  }
  if (tag === undefined || tag.startsWith(CORE_TAG_PREFIX)) return null;
  if (tag.startsWith("!")) return tag.slice(1);
  throw new SourceSyntaxError(
    `unknown YAML tag ${reprText(tag)}; a type tag is local, e.g. \`!int\``,
    range,
  );
}

/**
 * Build a scalar's value: under the core tag `core` by that tag, else (untagged or locally
 * tagged) a plain scalar by the table and a quoted or block one as its text.
 *
 * @throws FlowLoaderError when the tag can't build the text.
 */
function scalarValue(
  text: string,
  plain: boolean,
  core: string | undefined,
  local: boolean,
): ScalarValue {
  if (core !== undefined) return constructScalar(core, text, plain);
  if (!plain) return text;
  const resolved = resolvePlain(text);
  // `<<` merges only as an untagged key; under a local tag it is text.
  return constructScalar(local && resolved === MERGE_TAG ? STR_TAG : resolved, text, true);
}

/**
 * Return the source range of each line of a literal block scalar's value, or null when the
 * lines don't align with the text.
 *
 * Each constructed line must end its physical line after spaces only, the non-empty ones at
 * one indentation, and the physical lines after the last constructed one hold only spaces.
 * `source` is the normalized text, whose only line breaks are LF and CRLF.
 */
function literalLineRanges(
  source: string,
  starts: readonly number[],
  node: Scalar.Parsed,
  value: ScalarValue,
): SourceRange[] | null {
  if (node.type !== Scalar.BLOCK_LITERAL || value !== node.source) return null;
  const [header, end] = node.range;
  // The body starts after the header's line break.
  const headerBreak = source.indexOf("\n", header);
  if (headerBreak === -1 || headerBreak >= end) return value === "" ? [] : null;
  // Each physical line's start and the offset before its line break.
  const physical: [number, number][] = [];
  for (let at = headerBreak + 1; at < end; ) {
    const lineBreak = source.indexOf("\n", at);
    const next = lineBreak === -1 || lineBreak >= end ? end : lineBreak;
    const lineEnd = next > at && source[next - 1] === "\r" ? next - 1 : next;
    physical.push([at, lineEnd]);
    at = next + 1;
  }
  const lines = value.split("\n");
  if (lines.at(-1) === "") lines.pop(); // no phantom line after the last break
  if (lines.length > physical.length) return null;
  const ranges: SourceRange[] = [];
  let indent: number | null = null;
  for (const [i, line] of lines.entries()) {
    const [start, lineEnd] = physical[i]!;
    const contentStart = lineEnd - line.length;
    if (
      contentStart < start ||
      source.slice(contentStart, lineEnd) !== line ||
      !ONLY_SPACES.test(source.slice(start, contentStart))
    ) {
      return null;
    }
    if (line !== "") {
      indent ??= contentStart - start;
      if (contentStart - start !== indent) return null;
    }
    ranges.push(toRange(starts, contentStart, lineEnd));
  }
  const rest = physical.slice(lines.length);
  if (!rest.every(([start, lineEnd]) => ONLY_SPACES.test(source.slice(start, lineEnd)))) {
    return null;
  }
  return ranges;
}

/** Say whether a YAML error is at a `,`, `[`, `]`, `{` or `}` glued to the tag before it. */
function gluedToTag(error: YAMLError, source: string): boolean {
  // eemeli ends a tag at a flow indicator even in block context and reports MISSING_CHAR, a
  // code it also uses for an anchor glued to the next token and an unclosed flow collection,
  // so the error must sit on a flow indicator right after a `!` tag.
  if (error.code !== "MISSING_CHAR") return false;
  const at = error.pos[0];
  const lineStart = source.lastIndexOf("\n", at - 1) + 1;
  return GLUED_TAG.test(source.slice(lineStart, at + 1));
}

/** Return the node `written` names: an alias's target, else `written` itself. */
function target(
  aliasTargets: ReadonlyMap<Alias, AnchoredNode>,
  written: ParsedNode,
): AnchoredNode {
  return isAlias(written) ? aliasTargets.get(written)! : written;
}

/** Return a mapping key's text: the value YAML built when text, else the key as written. */
function keyText(key: ScalarDef): string {
  return typeof key.value === "string" ? key.value : key.raw!;
}

/** Return the offsets where lines start in normalized text: 0, and after each LF. */
function lineStarts(text: string): number[] {
  const starts = [0];
  for (let i = text.indexOf("\n"); i !== -1; i = text.indexOf("\n", i + 1)) starts.push(i + 1);
  return starts;
}

/** Return the 1-based line and 1-based UTF-16 column of `offset`. */
function position(starts: readonly number[], offset: number): [number, number] {
  let low = 0;
  let high = starts.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (starts[middle]! <= offset) low = middle;
    else high = middle - 1;
  }
  return [low + 1, offset - starts[low]! + 1];
}

/** Return the range from offset `start` to the exclusive offset `end`. */
function toRange(starts: readonly number[], start: number, end: number): SourceRange {
  const [startLine, startCol] = position(starts, start);
  const [endLine, endCol] = position(starts, end);
  return new SourceRange(startLine, startCol, endLine, endCol);
}

/** Return the one-character range at `offset`. */
function pointRange(starts: readonly number[], offset: number): SourceRange {
  const [line, column] = position(starts, offset);
  return new SourceRange(line, column, line, column + 1);
}
