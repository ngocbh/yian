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
} from "yaml";

import { SourceRange } from "../../diagnostics.ts";
import type { Value, ValueRecord } from "../../typesys/index.ts";
import {
  type ArcSequence,
  type Definition,
  freezeValue,
  type MapArc,
  type MapOccurrence,
  type ScalarDef,
  type ScalarStyle,
  setRecordKey,
  SourceSyntaxError,
} from "./common.ts";
import {
  type AnchoredNode,
  composeFlowDocument,
  constructScalar,
  type FlowDocument,
  FlowLoaderError,
  normalizeBreaks,
  resolvePlain,
  type ScalarValue,
} from "./scalars.ts";

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

/** Offsets from `start` to the exclusive `end`. */
interface Span {
  readonly start: number;
  readonly end: number;
}

/** Return a composed node's span: its range, from its anchor or tag when it has one. */
type SpanOf = (node: Node) => Span;

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

  /**
   * Compose and construct exactly one YAML document in one pass.
   *
   * @throws SourceSyntaxError when the text holds a character YAML refuses, is not one valid
   *   YAML document, or holds a node the flow loader can't build; `range` locates the problem.
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
    try {
      flow = composeFlowDocument(source);
    } catch (error) {
      if (error instanceof YAMLError) {
        const message =
          error.code === "MULTIPLE_DOCS"
            ? "expected a single document in the stream, but found another document"
            : error.message;
        throw new SourceSyntaxError(message, pointRange(starts, error.pos[0]));
      }
      if (error instanceof FlowLoaderError) {
        const node = error.node;
        const range = node?.range ? pointRange(starts, spanOf(node).start) : null;
        throw new SourceSyntaxError(error.message, range);
      }
      throw error;
    }
    const { definitions, ids, nonTextKey } = build(flow, starts, spanOf);
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
 * @throws SourceSyntaxError at a scalar the flow loader can't build, or at a list or mapping
 *   key.
 */
function build(flow: FlowDocument, starts: readonly number[], spanOf: SpanOf): Built {
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
  const intern = (written: ParsedNode): number => {
    const node = target(aliasTargets, written);
    const known = ids.get(node);
    if (known !== undefined) return known;
    const span = spanOf(node);
    const range = toRange(starts, span.start, span.end);
    if (isScalar(node)) {
      const text = node.source;
      const plain = node.type === Scalar.PLAIN;
      let value: ScalarValue;
      try {
        value = plain ? constructScalar(resolvePlain(text), text, true) : text;
      } catch (error) {
        if (error instanceof FlowLoaderError) {
          throw new SourceSyntaxError(error.message, pointRange(starts, span.start));
        }
        throw error;
      }
      const id = reserve(node, value);
      definitions[id] = Object.freeze({
        kind: "scalar",
        value,
        range,
        raw: text,
        constructedLineRanges: null,
        style: STYLES[node.type!],
        tag: null,
      });
      return id;
    }
    if (isSeq(node)) {
      const list: Value[] = [];
      const id = reserve(node, list);
      fills.push(() => {
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
          tag: null,
        });
      });
      return id;
    }
    const record: ValueRecord = {};
    const id = reserve(node, record);
    fills.push(() => {
      const arcs: MapArc[] = [];
      const effectiveIndex = new Map<string, MapOccurrence>();
      for (const pair of node.items) {
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
        // Fills don't run in document order, so the key that starts first wins; an alias
        // key is its anchored node.
        const keyAt = spanOf(keyNode).start;
        if (typeof keyScalar.value !== "string" && (nonTextKey === null || keyAt < nonTextKeyAt)) {
          const keyValue = keyScalar.value as NonTextKey["key"];
          nonTextKey = { key: keyValue, text: key, range: keyScalar.range! };
          nonTextKeyAt = keyAt;
        }
        const arc: MapArc = Object.freeze({ key, keyDefinition, valueDefinition });
        // A repeated key keeps its first position and takes the last value, as in SafeLoader.
        effectiveIndex.set(key, Object.freeze({ arc, pairOrdinal: arcs.length }));
        arcs.push(arc);
        setRecordKey(record, key, values[valueDefinition]!);
      }
      const pairSequence: ArcSequence = Object.freeze({
        arcs: Object.freeze(arcs),
        parts: NO_PARTS,
        partOffsets: NO_PARTS,
        flatPairCount: arcs.length,
      });
      definitions[id] = Object.freeze({
        kind: "mapping",
        value: record,
        range,
        pairSequence,
        effectiveIndex,
        tag: null,
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
 * The composed tree keeps no property positions, so the CST records them: under the end of
 * the properties, and under the token they precede by `offset:is a collection` (a block
 * mapping and its first key share an offset). An empty node has no token, or a zero-length
 * `scalar` one before a `,` in a flow collection; eemeli places it after its properties and
 * any spaces that follow them.
 */
function propertySpans(source: string): SpanOf {
  const byEnd = new Map<number, Span>();
  const byToken = new Map<string, Span>();
  const note = (props: readonly CST.SourceToken[], token?: CST.Token | null): void => {
    const span = propertiesSpan(props);
    if (span === null) return;
    byEnd.set(span.end, span);
    if (token) byToken.set(`${token.offset}:${CST.isCollection(token)}`, span);
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
        const key = `${token.offset}:true`;
        const known = byToken.get(key);
        if (known === undefined || known.start > keySpan.start) byToken.set(key, keySpan);
      }
    }
  }
  return (node) => {
    const [start, end] = node.range!;
    if (start === end) {
      let p = start;
      while (p > 0 && (source[p - 1] === " " || source[p - 1] === "\t")) p--;
      return byToken.get(`${start}:false`) ?? byEnd.get(p) ?? { start, end: start };
    }
    if (isMap(node) || isSeq(node)) {
      return { start: byToken.get(`${start}:true`)?.start ?? start, end };
    }
    if (node.anchor !== undefined || node.tag !== undefined) {
      return { start: byToken.get(`${start}:false`)?.start ?? start, end };
    }
    return { start, end };
  };
}

/** Return the span of the anchor and tag among `props`, or null when there are none. */
function propertiesSpan(props: readonly CST.SourceToken[]): Span | null {
  const marks = props.filter((token) => token.type === "anchor" || token.type === "tag");
  const first = marks[0];
  const last = marks[marks.length - 1];
  if (first === undefined || last === undefined) return null;
  return { start: first.offset, end: last.offset + last.source.length };
}

/** Return the tokens after the last `:` indicator in `sep`; none without one. */
function afterValueIndicator(sep: readonly CST.SourceToken[]): readonly CST.SourceToken[] {
  const at = sep.findLastIndex((token) => token.type === "map-value-ind");
  return at === -1 ? [] : sep.slice(at + 1);
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
