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
  type TaggedNode,
} from "./common.ts";
import { SourceCursor } from "./cursor.ts";
import {
  collectionTagError,
  isMergeKey,
  type MapPair,
  mergeSources,
  mergeValueError,
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
  parseTokens,
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
  /**
   * The front of a value a `{...}` mapping's comma split before either copy of the key, if one
   * did.
   */
  readonly cutValue?: string;
}

/** The first mapping key, in document order, that YAML read as non-text. */
export interface NonTextKey {
  readonly key: Exclude<ScalarValue, string>;
  readonly text: string;
  readonly range: SourceRange;
  /** The front of a value a `{...}` mapping's comma split before the key, if one did. */
  readonly cutValue?: string;
}

// YAML's printable characters; with the `u` flag a lone surrogate is one code point,
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

/** The local tag whose scalar names a file to read the value's text from. */
const INCLUDE_TAG = "include";

const NO_INCLUDES: ReadonlyMap<string, string> = new Map();

/** An `!include` value a parse wasn't given the text of. */
export interface IncludeSite {
  /** The path as written after the tag. */
  readonly path: string;
  /** The `!include` value's range, its tag included. */
  readonly range: SourceRange;
}

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
  /** Whether a leaf's pairs are written in a `{...}` mapping. */
  readonly flow: boolean;
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
  flow: false,
  parts: NO_PARTS,
  partOffsets: NO_PARTS,
  flatPairCount: 0,
};

/** One authored key path step linked to its parent: a key as written, or an item index. */
interface KeyLink {
  readonly parent: KeyLink | null;
  readonly step: string | number;
}

/** One pending occurrence of the tagged-node walk. */
interface WalkFrame {
  readonly cursor: SourceCursor;
  /** The key definition selecting the occurrence; null for the root or a sequence item. */
  readonly keyDefinition: number | null;
  readonly link: KeyLink | null;
}

/** Spell a linked authored key path out from the root. */
function spellKeyLink(link: KeyLink | null): (string | number)[] {
  const steps: (string | number)[] = [];
  for (let current = link; current !== null; current = current.parent) steps.push(current.step);
  return steps.reverse();
}

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
  /**
   * Each `!include` value the parse wasn't given the text of, in document order. Each stays a
   * scalar tagged `!include` that holds its path.
   */
  readonly includes: readonly IncludeSite[];
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
    includes: readonly IncludeSite[],
  ) {
    this.text = text;
    this.label = label;
    this.repeatedKey = repeatedKey;
    this.nonTextKey = nonTextKey;
    this.includes = Object.freeze(includes);
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
   * Return every node that carries a local tag, in document order.
   *
   * An aliased node is listed once, with the first occurrence path that reaches it.
   * `skipTopLevel` picks top-level keys, as written, whose pairs are not walked: a node
   * reached only through them is not listed; a node also reached elsewhere is listed with its
   * first path outside them.
   *
   * Example: `SourceDocument.parse("a: !int 1\nb: [!str x]\n").taggedNodes()` lists the
   * tags `"int"` and `"str"`.
   */
  taggedNodes(options: { skipTopLevel?: (key: string) => boolean } = {}): readonly TaggedNode[] {
    if (!this._definitions.some((definition) => definition.tag !== null)) return [];
    const skipTopLevel = options.skipTopLevel ?? null;
    // Each definition's first cursor and authored key link; paths and key paths are spelled
    // out only for tagged nodes, so the walk stays linear in the document size.
    const firstCursors = new Map<number, SourceCursor>();
    const keyLinks = new Map<number, KeyLink | null>();
    const keyIds = new Set<number>();
    const walkedParts = new Set<ArcSequence>();
    const stack: WalkFrame[] = [{ cursor: this.root, keyDefinition: null, link: null }];
    while (stack.length > 0) {
      const { cursor, keyDefinition, link } = stack.pop()!;
      if (keyDefinition !== null && !firstCursors.has(keyDefinition)) {
        firstCursors.set(keyDefinition, cursor);
        keyLinks.set(keyDefinition, link);
        keyIds.add(keyDefinition);
      }
      const definitionId = cursor._definition!;
      if (firstCursors.has(definitionId)) continue;
      firstCursors.set(definitionId, cursor);
      keyLinks.set(definitionId, link);
      const definition = this._definitions[definitionId]!;
      const skip = definitionId === this._rootDefinition ? skipTopLevel : null;
      const children: WalkFrame[] = [];
      if (definition.kind === "mapping") {
        // A rope part shared by several mappings (a `<<:` source) is walked once per
        // document: the first walk already reached every node in it, so merge-heavy
        // documents stay linear.
        const parts: [ArcSequence, number][] = [[definition.pairSequence, 0]];
        while (parts.length > 0) {
          const [part, base] = parts.pop()!;
          // The root is walked once; with skipped keys, its parts stay unmarked so another
          // mapping sharing them walks every pair.
          if (skip === null) {
            if (walkedParts.has(part)) continue;
            walkedParts.add(part);
          }
          for (let index = part.parts.length - 1; index >= 0; index--) {
            parts.push([part.parts[index]!, base + part.partOffsets[index]!]);
          }
          part.arcs.forEach((arc, index) => {
            if (skip !== null && skip(arc.key)) return;
            children.push({
              cursor: cursor._descendMappingOccurrence({ arc, pairOrdinal: base + index }),
              keyDefinition: arc.keyDefinition,
              link: { parent: link, step: arc.key },
            });
          });
        }
      } else if (definition.kind === "sequence") {
        definition.itemDefinitions.forEach((item, index) => {
          children.push({
            cursor: cursor._descendSequenceItem(index, item),
            keyDefinition: null,
            link: { parent: link, step: index },
          });
        });
      }
      for (let index = children.length - 1; index >= 0; index--) stack.push(children[index]!);
    }

    const tagged: TaggedNode[] = [];
    this._definitions.forEach((definition, definitionId) => {
      if (definition.tag === null) return;
      const first = firstCursors.get(definitionId);
      if (skipTopLevel !== null && first === undefined) return;
      tagged.push(
        Object.freeze({
          definitionId,
          tag: definition.tag,
          range: definition.range,
          path: first?.path ?? [],
          isKey: keyIds.has(definitionId),
          keyPath: spellKeyLink(keyLinks.get(definitionId) ?? null),
        }),
      );
    });
    // Nodes without a range sort last.
    tagged.sort((first, second) => {
      const a = first.range;
      const b = second.range;
      if (a === null || b === null) {
        if (a !== b) return a === null ? 1 : -1;
      } else if (a.startLine !== b.startLine) {
        return a.startLine - b.startLine;
      } else if (a.startCol !== b.startCol) {
        return a.startCol - b.startCol;
      }
      return first.definitionId - second.definitionId;
    });
    return Object.freeze(tagged);
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
   * `included` maps an `!include` path, as written, to the file's text. Such a value reads as
   * that text: a string, never resolved as a plain scalar, untagged, marked `included`, and
   * located at the `!include` value. A path `included` lacks is listed in `includes`.
   *
   * @throws SourceSyntaxError when the text holds a character YAML refuses, is not one valid
   *   YAML document, or holds a node the flow loader can't build, a tag that is bare or
   *   neither local nor core, or `!include` on a mapping key, a list, a mapping or a `<<:` merge
   *   source; `range` locates the problem. A YAML error at a character glued to a tag ends with
   *   `TAG_HINT`.
   */
  static parse(
    text: string,
    label = "<flow>",
    included: ReadonlyMap<string, string> = NO_INCLUDES,
  ): SourceDocument {
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
      built = build(flow, source, starts, spanOf, included);
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
    const { definitions, ids, nonTextKey, includes } = built;
    return new SourceDocument(
      text,
      label,
      definitions,
      firstRepeatedKey(flow, ids, definitions, starts),
      nonTextKey,
      includes,
    );
  }
}

/** What one pass over a composed document builds. */
interface Built {
  readonly definitions: Definition[];
  readonly ids: ReadonlyMap<Node, number>;
  readonly nonTextKey: NonTextKey | null;
  readonly includes: readonly IncludeSite[];
}

/**
 * Intern every reachable node of `flow` once, as SafeLoader constructs a document.
 *
 * A list or mapping is allocated when first reached and filled later, first-in first-out, so
 * a container's scalars are built (and their errors raised) before the containers nested in
 * it, and an alias to a container still being filled shares it. The root is definition 0.
 *
 * `source` is the normalized text the CST and `starts` index. An `!include` scalar reads as its
 * path's text in `included`, else as its path, still tagged and listed in `includes`.
 *
 * @throws SourceSyntaxError at a scalar the flow loader can't build, at a list or mapping key,
 *   at a bare tag or one neither local nor core, at a core tag that can't build its node, or at
 *   an `!include` on a key, a list, a mapping or a merge source.
 */
function build(
  flow: FlowDocument,
  source: string,
  starts: readonly number[],
  spanOf: SpanOf,
  included: ReadonlyMap<string, string>,
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
      included: false,
    };
    return { definitions: [Object.freeze(empty)], ids, nonTextKey: null, includes: [] };
  }
  const definitions: (Definition | null)[] = [];
  const values: Value[] = [];
  const fills: (() => void)[] = [];
  let nonTextKey: NonTextKey | null = null;
  let nonTextKeyAt = 0;
  const includes: IncludeSite[] = [];

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
        included: false,
      }),
    );
    values.push(null);
    return id;
  };
  const targetOf = (written: ParsedNode | null): AnchoredNode | null =>
    written === null ? null : target(aliasTargets, written);
  /**
   * Refuse a tag written at a `<<:` merge site: YAML
   * never builds a merge source, so a local tag there checks nothing. An alias is skipped: its
   * target's tag sits at the anchor and checks the value there. A missing value fails here,
   * where its key `key` ends, as `internMissing` places a missing value.
   */
  const checkSite = (written: ParsedNode | null, key: ParsedNode): void => {
    if (written === null) {
      throw new SourceSyntaxError(mergeValueError(null), pointRange(starts, key.range[1]));
    }
    if (isAlias(written)) return;
    const span = spanOf(written);
    const range = toRange(starts, span.start, span.end);
    const tag = localTag(written, span, starts);
    if (tag === INCLUDE_TAG) {
      throw new SourceSyntaxError(
        "`!include` gives text, and `<<:` merges a mapping; write the mapping to merge here",
        range,
      );
    }
    if (tag !== null) {
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
      leaf = pairLeaf(ownPairs(map, targetOf), map.flow === true);
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
  const leafArcs = ({ pairs, flow }: PairSequence): ArcSequence => {
    const arcs: MapArc[] = [];
    for (const [index, pair] of pairs.entries()) {
      const keyNode = target(aliasTargets, pair.key);
      // An alias key is located where it is written, not at its anchor: the fix goes on the
      // key, and quoting the anchored node would change its value instead.
      const keyAlias = isAlias(pair.key) ? pair.key : null;
      const keyAt = keyAlias === null ? spanOf(keyNode).start : keyAlias.range![0];
      const collectionKey = (): SourceSyntaxError =>
        new SourceSyntaxError(
          "a list or mapping can't be a mapping key; mapping keys must be text",
          pointRange(starts, keyAt),
        );
      if (!isScalar(keyNode)) throw collectionKey();
      const keyDefinition = intern(keyNode);
      const keyScalar = definitions[keyDefinition];
      // A scalar tagged `!!seq` or `!!map` builds a list or record, which can't be
      // a key; it fails here, before the value is built.
      if (keyScalar?.kind !== "scalar") throw collectionKey();
      if (keyScalar.tag === INCLUDE_TAG || keyScalar.included) {
        throw new SourceSyntaxError(
          "`!include` can't be on a mapping key; it gives a value's text, as in " +
            "`prompt: !include prompts/answer.md`",
          pointRange(starts, keyAt),
        );
      }
      // A missing value sits where the key ends.
      const valueDefinition =
        pair.value === null ? internMissing(pair.key.range[1]) : intern(pair.value);
      const key = keyText(keyScalar);
      // Fills don't run in document order, so the key that starts first wins.
      if (typeof keyScalar.value !== "string" && (nonTextKey === null || keyAt < nonTextKeyAt)) {
        const keyValue = keyScalar.value as NonTextKey["key"];
        const range =
          keyAlias === null
            ? keyScalar.range!
            : toRange(starts, keyAlias.range![0], keyAlias.range![1]);
        const cutValue = flow ? cutValueBefore(pairs, index) : null;
        nonTextKey = Object.freeze({
          key: keyValue,
          text: key,
          range,
          ...(cutValue === null ? {} : { cutValue }),
        });
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
   * Return a pair rope as an arc rope of the same shape with its summary (its
   * pairs in first-key order), post-order on an explicit stack and cached
   * per rope: each distinct leaf is interned once, leaves in first-occurrence order, and a
   * shared rope is summarized once.
   */
  const arcRope = (pairSequence: PairSequence): ArcRope => {
    const stack: [PairSequence, boolean][] = [[pairSequence, false]];
    while (stack.length > 0) {
      const [current, ready] = stack.pop()!;
      if (arcRopes.has(current)) continue;
      if (current.parts.length === 0) {
        const sequence = leafArcs(current);
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
  /**
   * Intern an `!include` scalar: its path's text, or its path when `included` lacks it.
   *
   * @throws SourceSyntaxError at an `!include` on a list or mapping.
   */
  const internInclude = (node: AnchoredNode, range: SourceRange, at: SourceRange): number => {
    if (!isScalar(node)) {
      throw new SourceSyntaxError(
        "`!include` reads a file's text into a string, so it goes on a text value, not a " +
          (isSeq(node) ? "list" : "mapping"),
        at,
      );
    }
    // The path is the text as written: a plain `3` or `true` names a file too.
    const path = node.source;
    const text = included.get(path);
    if (text === undefined) includes.push(Object.freeze({ path, range }));
    const value = text ?? path;
    const id = reserve(node, value);
    definitions[id] = Object.freeze({
      kind: "scalar",
      value,
      range,
      // An included value is written as its `!include`, so a message never quotes the file.
      raw: text === undefined ? path : `!include ${source.slice(node.range![0], node.range![1])}`,
      constructedLineRanges: null,
      style: text === undefined ? STYLES[node.type!] : "literal",
      tag: text === undefined ? INCLUDE_TAG : null,
      included: text !== undefined,
    });
    return id;
  };
  const intern = (written: ParsedNode): number => {
    const node = target(aliasTargets, written);
    const known = ids.get(node);
    if (known !== undefined) return known;
    const span = spanOf(node);
    const range = toRange(starts, span.start, span.end);
    const at = pointRange(starts, span.start);
    const tag = localTag(node, span, starts);
    if (tag === INCLUDE_TAG) return internInclude(node, range, at);
    // A local tag reads its node as if untagged.
    const core = tag === null ? node.tag : undefined;
    // The tag picks the constructor: `!!seq` builds a list and `!!map` a record
    // whatever the node's kind.
    const picksCollection = core === SEQ_TAG || core === MAP_TAG;
    if (isScalar(node) && !picksCollection) {
      const text = node.source;
      const plain = node.type === Scalar.PLAIN;
      let value: ScalarValue;
      try {
        value = scalarValue(text, plain, core);
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
        included: false,
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
  includes.sort(
    (a, b) => a.range.startLine - b.range.startLine || a.range.startCol - b.range.startCol,
  );
  return { definitions: built, ids, nonTextKey, includes };
}

/**
 * Return `pairs`, written in a `{...}` mapping when `flow`, as a leaf rope; no pairs is the
 * empty rope.
 */
function pairLeaf(pairs: readonly MapPair[], flow: boolean): PairSequence {
  if (pairs.length === 0) return EMPTY_PAIRS;
  return { pairs, flow, parts: NO_PARTS, partOffsets: NO_PARTS, flatPairCount: pairs.length };
}

/** Return how many more `(` than `)` `text` holds. */
function openParens(text: string): number {
  return text.split("(").length - text.split(")").length;
}

/**
 * Return the front of a value a `{...}` mapping's comma split before `pairs[index]`, or null.
 *
 * YAML reads `{role: Literal(a, true, b)}` as the value `Literal(a` and the keys `true` and
 * `b)` with no value. The front is the nearest earlier plain value, past plain keys with no
 * value; it is a split value's when it and those keys open more `(` than they close.
 */
function cutValueBefore(pairs: readonly MapPair[], index: number): string | null {
  let open = 0;
  for (let i = index - 1; i >= 0; i--) {
    const { key, value } = pairs[i]!;
    const piece = value ?? key;
    if (!isScalar(piece) || piece.type !== Scalar.PLAIN) return null;
    open += openParens(piece.source);
    if (value !== null) return open > 0 ? piece.source : null;
  }
  return null;
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
  return { pairs: NO_PARTS, flow: false, parts: kept, partOffsets, flatPairCount: count };
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
 * Return the first key a mapping repeats among its own pairs, in a pre-order walk, by key
 * text; an alias of an earlier key is a repeat too. Each key is located where it is written.
 */
function firstRepeatedKey(
  flow: FlowDocument,
  ids: ReadonlyMap<Node, number>,
  definitions: readonly Definition[],
  starts: readonly number[],
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
    const firstKeys = new Map<string, { readonly range: SourceRange; readonly index: number }>();
    for (const [index, pair] of node.items.entries()) {
      const keyNode = target(aliasTargets, pair.key);
      // Keys a merge brings in override; only a mapping's own pairs count.
      if (isMergeKey(keyNode)) continue;
      const keyScalar = definitions[ids.get(keyNode)!] as ScalarDef;
      const key = keyText(keyScalar);
      // An alias key is located where it is written, not at its anchor, so a key repeated
      // through an alias of itself points at the repeat.
      const written = pair.key;
      const range = isAlias(written)
        ? toRange(starts, written.range![0], written.range![1])
        : keyScalar.range!;
      const first = firstKeys.get(key);
      if (first !== undefined) {
        // Either copy of the key can be a piece of the split value.
        const cutValue =
          node.flow === true
            ? (cutValueBefore(node.items, index) ?? cutValueBefore(node.items, first.index))
            : null;
        return Object.freeze({
          key,
          first: first.range,
          second: range,
          ...(cutValue === null ? {} : { cutValue }),
        });
      }
      firstKeys.set(key, { range, index });
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
  const document = [...parseTokens(source)].find(
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
    return { start: from, end: pyyamlEnd(source, node, end), bare: own?.bare ?? false };
  };
  return spanOf;
}

/**
 * Return where a source range ends a non-empty node that eemeli ends at `end`.
 *
 * A range ends a block mapping or list at the block-end token that closes it, which the scanner
 * emits at the next token, past blank lines and comments, or at the end of the text; it ends a
 * `|` or `>` scalar after the blank lines that follow its text. eemeli ends both at their last
 * content, so the scan moves the end on; every other node already ends there.
 */
function pyyamlEnd(source: string, node: Node, end: number): number {
  let at = end;
  if ((isMap(node) || isSeq(node)) && !node.flow) {
    while (at < source.length) {
      const char = source[at];
      if (char === "#") {
        const lineBreak = source.indexOf("\n", at);
        at = lineBreak === -1 ? source.length : lineBreak;
      } else if (char === " " || char === "\t" || char === "\n" || char === "\r") {
        at++;
      } else {
        break;
      }
    }
  } else if (
    isScalar(node) &&
    (node.type === Scalar.BLOCK_LITERAL || node.type === Scalar.BLOCK_FOLDED)
  ) {
    for (;;) {
      let lineEnd = at;
      while (source[lineEnd] === " ") lineEnd++;
      if (source[lineEnd] === "\r" && source[lineEnd + 1] === "\n") lineEnd++;
      if (source[lineEnd] !== "\n") break;
      at = lineEnd + 1;
    }
  }
  return at;
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
 * `starts` are the line starts of the text `span` indexes.
 *
 * @throws SourceSyntaxError at the bare tag `!`, which eemeli keeps on a scalar but resolves
 *   away on a list or mapping, or over the node at a tag neither local nor core.
 */
function localTag(node: AnchoredNode, span: Span, starts: readonly number[]): string | null {
  const tag = node.tag;
  if (isScalar(node) ? tag === "!" : span.bare) {
    // The bare tag is refused at the node's start: a scalar's ends with the
    // scalar, a flow collection's after its opening bracket, a block one's at its first token.
    const end = isScalar(node) ? span.end : node.flow ? node.range![0] + 1 : node.range![0];
    const range = toRange(starts, span.start, end);
    throw new SourceSyntaxError("a tag names a type, e.g. `!int`", range);
  }
  if (tag === undefined || tag.startsWith(CORE_TAG_PREFIX)) return null;
  if (tag.startsWith("!")) return tag.slice(1);
  throw new SourceSyntaxError(
    `unknown YAML tag ${reprText(tag)}; a type tag is local, e.g. \`!int\``,
    toRange(starts, span.start, span.end),
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
): ScalarValue {
  if (core !== undefined) return constructScalar(core, text, plain);
  if (!plain) return text;
  const resolved = resolvePlain(text);
  // `<<` merges only as an untagged key, which never reaches here; anywhere else it is text.
  return constructScalar(resolved === MERGE_TAG ? STR_TAG : resolved, text, true);
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
): readonly SourceRange[] | null {
  if (node.type !== Scalar.BLOCK_LITERAL || value !== node.source) return null;
  const [header, end] = node.range;
  // The body starts after the header's line break.
  const headerBreak = source.indexOf("\n", header);
  if (headerBreak === -1 || headerBreak >= end) return value === "" ? Object.freeze([]) : null;
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
  return Object.freeze(ranges);
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
  return Object.freeze(new SourceRange(startLine, startCol, endLine, endCol));
}

/** Return the one-character range at `offset`. */
function pointRange(starts: readonly number[], offset: number): SourceRange {
  const [line, column] = position(starts, offset);
  return Object.freeze(new SourceRange(line, column, line, column + 1));
}
