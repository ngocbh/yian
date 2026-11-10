/**
 * Shared contracts for source documents and their occurrence cursors: errors, provenance,
 * occurrence paths, the private definition records, and YAML-value snapshots.
 */

import type { SourceRange } from "../../diagnostics.ts";
import { isList, isRecord, jsTypeName, type Value, type ValueRecord } from "../../typesys/index.ts";
// Type-only: `document.ts` imports this module at runtime.
import type { SourceDocument } from "./document.ts";

/**
 * A source cursor can't satisfy a requested navigation or typed access. `range` is the
 * closest concrete source location, else `null` for a virtual or otherwise unlocated value.
 */
export class SourceAccessError extends RangeError {
  override name = "SourceAccessError";
  readonly range: SourceRange | null;

  constructor(message: string, range: SourceRange | null = null) {
    super(message);
    this.range = range;
  }
}

/**
 * A YAML source can't be composed or constructed safely. `range` locates the problem when
 * known, else `null`.
 */
export class SourceSyntaxError extends RangeError {
  override name = "SourceSyntaxError";
  readonly range: SourceRange | null;

  constructor(message: string, range: SourceRange | null = null) {
    super(message);
    this.range = range;
  }
}

/** Source provenance for one unchanged text scalar. */
export interface ScalarSource {
  /** The whole authored scalar, used when exact line selection is unavailable. */
  readonly range: SourceRange;
  /**
   * Zero-based ranges of a literal block scalar's constructed line contents. `null` means
   * the scalar has physical source but its constructed lines can't be selected exactly; `[]`
   * means an exactly mapped scalar with no constructed lines.
   */
  readonly constructedLineRanges: readonly SourceRange[] | null;
}

/**
 * One mapping-pair step of an occurrence path: the pair's key and its zero-based position in
 * the flattened pairs, duplicates and merged pairs included.
 */
export interface MapStep {
  readonly key: string;
  readonly pairOrdinal: number;
}

/** One zero-based sequence-item step of an occurrence path. */
export interface SeqStep {
  readonly index: number;
}

/** One step of an occurrence path. */
export type OccurrenceStep = MapStep | SeqStep;

/** The steps that reach one occurrence from the document root. */
export type OccurrencePath = readonly OccurrenceStep[];

/**
 * The file a typed `!include(T) PATH` value was read from: `T` as written, which is the value's
 * tag, and the path as written. `data` is the file read as a YAML value, whose ranges locate a
 * part of the value in the file; `null` for `!include(str)`, whose value is the file's text.
 */
export interface TypedInclude {
  readonly type: string;
  readonly path: string;
  readonly data: SourceDocument | null;
}

/** One YAML node that carries a local tag. */
export interface TaggedNode {
  /** The document-local id of the tagged definition; aliases share it. */
  readonly definitionId: number;
  /** The tag text without the leading `!`, e.g. `"list(str)"`. */
  readonly tag: string;
  /** The tagged node's range. */
  readonly range: SourceRange | null;
  /**
   * The first occurrence path that reaches the node from the document root; for a tagged
   * mapping key, the path of the pair the key selects.
   */
  readonly path: OccurrencePath;
  /** Whether that first occurrence is a mapping key. */
  readonly isKey: boolean;
  /** `path` as authored: each mapping key as written and each sequence item's index. */
  readonly keyPath: readonly (string | number)[];
  /** The file the node's value was read from when it is an `!include(T)`, else `null`. */
  readonly include: TypedInclude | null;
}

/** @internal One lazily materialized occurrence-path link. */
export interface PathLink {
  readonly parent: PathLink | null;
  readonly step: OccurrenceStep | null;
  readonly definitionId: number;
}

/** @internal One authored non-merge mapping pair, kept without duplicate collapse. */
export interface MapArc {
  readonly key: string;
  readonly keyDefinition: number;
  readonly valueDefinition: number;
}

/** @internal One local arc at its logical position in a flattened merge view. */
export interface MapOccurrence {
  readonly arc: MapArc;
  readonly pairOrdinal: number;
}

/**
 * @internal A compact persistent sequence of mapping arcs: a leaf (arcs, no parts) or a
 * concatenation (parts at partOffsets, each child's logical start ordinal, no arcs). Counts
 * and ordinals are exact while the node limit bounds them.
 */
export interface ArcSequence {
  readonly arcs: readonly MapArc[];
  readonly parts: readonly ArcSequence[];
  readonly partOffsets: readonly number[];
  readonly flatPairCount: number;
}

/** A scalar's authored style. */
export type ScalarStyle = "plain" | "single" | "double" | "literal" | "folded";

/**
 * @internal One scalar-shaped YAML definition. `style` is `null` when `raw` is; `tag` is the
 * local tag text without the leading `!`, `null` when the scalar is untagged or carries a
 * YAML core tag. `included` says the value is a file's text an `!include` read: untagged, its
 * style `literal` and its `raw` the `!include` as written. `verbatim` says the value is text
 * that is never a template: the text `!include(str)` read, or a string in a value `!include(T)`
 * read.
 */
export interface ScalarDef {
  readonly kind: "scalar";
  readonly value: Value;
  readonly range: SourceRange | null;
  readonly raw: string | null;
  readonly constructedLineRanges: readonly SourceRange[] | null;
  readonly style: ScalarStyle | null;
  readonly tag: string | null;
  readonly included: boolean;
  readonly verbatim: boolean;
}

/**
 * @internal One mapping definition backed by a compact persistent pair sequence. `tag` is the
 * local tag text without the leading `!`, `null` when the mapping is untagged or carries a
 * YAML core tag.
 */
export interface MappingDef {
  readonly kind: "mapping";
  readonly value: ValueRecord;
  readonly range: SourceRange | null;
  readonly pairSequence: ArcSequence;
  readonly effectiveIndex: ReadonlyMap<string, MapOccurrence>;
  readonly tag: string | null;
}

/**
 * @internal One sequence-shaped YAML definition and its item definitions. `tag` is the local
 * tag text without the leading `!`, `null` when the sequence is untagged or carries a YAML
 * core tag.
 */
export interface SequenceDef {
  readonly kind: "sequence";
  readonly value: Value[];
  readonly range: SourceRange | null;
  readonly itemDefinitions: readonly number[];
  readonly tag: string | null;
}

/** @internal One YAML definition. */
export type Definition = ScalarDef | MappingDef | SequenceDef;

/** Set `record[key]`; a `__proto__` key is defined as data, not the prototype. */
export function setRecordKey(record: ValueRecord, key: string, value: Value): void {
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

/**
 * Return an alias-preserving deep copy of a YAML value, iteratively; throws TypeError for
 * anything else.
 *
 * A YAML value is text, a `bigint`, a `number`, a `boolean`, `null`, a plain array, or a
 * plain record. Each container is copied once, when first reached, so shared and recursive
 * aliases are kept, and an explicit stack keeps any depth off the call stack.
 *
 * @throws TypeError for anything else anywhere in the value: a class instance, a `Map`, a
 *   `Date`, `undefined` (a sparse array's hole), a function, an array subclass.
 */
export function snapshotValue(value: unknown): Value {
  const copies = new Map<object, Value[] | ValueRecord>();
  const pending: unknown[] = [value];
  while (pending.length > 0) {
    const current = pending.pop();
    if (isAtom(current)) continue;
    if (copies.has(current as object)) continue;
    // Each original is read once: the copy first holds the original children, and the loop
    // below swaps the containers among them for their copies.
    if (isList(current)) {
      const list: unknown[] = [];
      // An index loop, not for-of, so a sparse array's hole reads as `undefined` and is
      // rejected below rather than skipped.
      for (let index = 0; index < current.length; index++) list.push(current[index]);
      copies.set(current, list as Value[]);
      for (const item of list) pending.push(item);
    } else if (isRecord(current)) {
      const record: ValueRecord = {};
      for (const key of Object.keys(current)) setRecordKey(record, key, current[key]!);
      copies.set(current, record);
      for (const key of Object.keys(record)) pending.push(record[key]);
    } else {
      throw new TypeError(
        "source cursors accept only YAML values (text, numbers, booleans, null, lists and " +
          `records); got ${jsTypeName(current)}`,
      );
    }
  }
  const copied = (item: Value): Value => (isAtom(item) ? item : copies.get(item)!);
  for (const copy of copies.values()) {
    if (Array.isArray(copy)) {
      for (let index = 0; index < copy.length; index++) copy[index] = copied(copy[index]!);
    } else {
      for (const key of Object.keys(copy)) setRecordKey(copy, key, copied(copy[key]!));
    }
  }
  return copied(value as Value);
}

/** Deep-freeze a YAML value in place, iteratively, and return it. */
export function freezeValue<T extends Value>(value: T): T {
  const seen = new Set<object>();
  const pending: Value[] = [value];
  while (pending.length > 0) {
    const current = pending.pop()!;
    if (!(isList(current) || isRecord(current)) || seen.has(current)) continue;
    seen.add(current);
    Object.freeze(current);
    if (isList(current)) {
      for (const item of current) pending.push(item as Value);
    } else {
      for (const key of Object.keys(current)) pending.push(current[key]!);
    }
  }
  return value;
}

/** Say whether `value` is a YAML atom: text, a `bigint`, a `number`, a `boolean` or `null`. */
function isAtom(value: unknown): value is string | bigint | number | boolean | null {
  const type = typeof value;
  return (
    value === null ||
    type === "string" ||
    type === "bigint" ||
    type === "number" ||
    type === "boolean"
  );
}
