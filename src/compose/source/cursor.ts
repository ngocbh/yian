/** Immutable cursors over physical and virtual YAML source values. */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { YianTypeError } from "../../typesys/errors.ts";
import {
  isList,
  isRecord,
  typeName,
  type Value,
  type ValueRecord,
} from "../../typesys/index.ts";
import {
  type ArcSequence,
  freezeValue,
  type MapArc,
  type MapOccurrence,
  type MappingDef,
  type OccurrencePath,
  type OccurrenceStep,
  type PathLink,
  type ScalarDef,
  type ScalarSource,
  type ScalarStyle,
  type SequenceDef,
  setRecordKey,
  snapshotValue,
  SourceAccessError,
  type TypedInclude,
} from "./common.ts";
// Type-only: `document.ts` imports this module at runtime.
import type { SourceDocument } from "./document.ts";

/** @internal One projected mapping entry and its optional physical source cursor. */
interface ProjectedEntry {
  readonly key: string;
  readonly source: SourceCursor | null;
}

/**
 * Each projection's child cursors, by entry, built on first read: a cursor is immutable, so one
 * child serves every read, and a projection read key by key (a node's thousands of `input:`
 * bindings) builds each child once instead of once per read.
 */
const projectedChildren = new WeakMap<SourceCursor, Map<ProjectedEntry, SourceCursor>>();

/** @internal The fields a cursor is built from. */
interface CursorFields {
  readonly document: SourceDocument | null;
  readonly definition: number | null;
  readonly value: Value;
  readonly pathLink: PathLink | null;
  readonly keyDefinition: number | null;
  readonly cycleCut: boolean;
  readonly overlay: boolean;
  readonly projectedEntries: readonly ProjectedEntry[] | null;
  readonly projectedIndex: ReadonlyMap<string, number> | null;
  readonly sequenceSources: readonly (SourceCursor | null)[] | null;
}

/**
 * Navigate one immutable, occurrence-aware YAML source value.
 *
 * Physical cursors keep a document definition, an occurrence path, and ranges. Virtual
 * cursors keep only a frozen snapshot of their value. Projected cursors show a replacement
 * value while keeping the child sources they were given. Cursors compare by identity.
 *
 * Example: `SourceCursor.virtual({ items: ["first"] }).child("items").child(0).value` is
 * `"first"`.
 */
export class SourceCursor {
  /** @internal The owning document, or null for a virtual cursor. */
  readonly _document: SourceDocument | null;
  /** @internal The value's definition id, or null for a virtual cursor. */
  readonly _definition: number | null;
  /** @internal The value, deep-frozen. */
  readonly _value: Value;
  /** @internal The last link of the occurrence path, or null for a virtual cursor. */
  readonly _pathLink: PathLink | null;
  /** @internal The selecting key's definition id; null for a root or a sequence item. */
  readonly _keyDefinition: number | null;
  /** @internal Whether this occurrence repeats an ancestor's definition (a recursive alias). */
  readonly _cycleCut: boolean;
  /** @internal Whether the value replaces the definition's, so the definition can't navigate. */
  readonly _overlay: boolean;
  /** @internal A mapping projection's entries, in order. */
  readonly _projectedEntries: readonly ProjectedEntry[] | null;
  /** @internal Each projected key's last entry position. */
  readonly _projectedIndex: ReadonlyMap<string, number> | null;
  /** @internal A replacement sequence's items' source cursors, by index. */
  readonly _sequenceSources: readonly (SourceCursor | null)[] | null;

  private constructor(fields: CursorFields) {
    this._document = fields.document;
    this._definition = fields.definition;
    this._value = fields.value;
    this._pathLink = fields.pathLink;
    this._keyDefinition = fields.keyDefinition;
    this._cycleCut = fields.cycleCut;
    this._overlay = fields.overlay;
    this._projectedEntries = fields.projectedEntries;
    this._projectedIndex = fields.projectedIndex;
    this._sequenceSources = fields.sequenceSources;
    Object.freeze(this);
  }

  /** @internal Build a cursor at definition `definition` of `document`. */
  static _physical(
    document: SourceDocument,
    definition: number,
    options: { pathLink?: PathLink; keyDefinition?: number | null; cycleCut?: boolean } = {},
  ): SourceCursor {
    return new SourceCursor({
      document,
      definition,
      value: document._definitions[definition]!.value,
      pathLink:
        options.pathLink ?? Object.freeze({ parent: null, step: null, definitionId: definition }),
      keyDefinition: options.keyDefinition ?? null,
      cycleCut: options.cycleCut ?? false,
      overlay: false,
      projectedEntries: null,
      projectedIndex: null,
      sequenceSources: null,
    });
  }

  /**
   * Build a location-free cursor over a frozen snapshot of `value`.
   *
   * @throws TypeError when `value` holds anything but a YAML value.
   */
  static virtual(value: Value): SourceCursor {
    return SourceCursor._adopt(freezeValue(snapshotValue(value)));
  }

  /**
   * @internal Build a location-free cursor over `value` without copying it; `value` must
   * already be deep-frozen, as a part of another cursor's value is.
   */
  static _adopt(value: Value): SourceCursor {
    return new SourceCursor({
      document: null,
      definition: null,
      value,
      pathLink: null,
      keyDefinition: null,
      cycleCut: false,
      overlay: false,
      projectedEntries: null,
      projectedIndex: null,
      sequenceSources: null,
    });
  }

  /** The owning document, or null for a virtual cursor. */
  get document(): SourceDocument | null {
    return this._document;
  }

  /** The occurrence path from the document root, built on demand; empty for a virtual cursor. */
  get path(): OccurrencePath {
    const steps: OccurrenceStep[] = [];
    for (let link = this._pathLink; link !== null; link = link.parent) {
      if (link.step !== null) steps.push(link.step);
    }
    return steps.reverse();
  }

  /** The value definition's range, at its anchor for an alias; null for a virtual cursor. */
  get range(): SourceRange | null {
    if (this._document === null || this._definition === null) return null;
    return this._document._definitions[this._definition]!.range;
  }

  /**
   * The source of an unchanged text scalar: its whole range, plus its constructed lines'
   * ranges when it is a literal block. Null for a virtual cursor, a value that isn't text, or
   * a value that differs from the authored one.
   *
   * Example: `SourceDocument.parse("code: |-\n  pass\n").root.child("code").scalarSource`
   * has one constructed line range, on line 2.
   */
  get scalarSource(): ScalarSource | null {
    const definition = this._sourceScalarDefinition();
    if (
      definition === null ||
      definition.range === null ||
      typeof definition.value !== "string" ||
      this._value !== definition.value
    ) {
      return null;
    }
    return Object.freeze({
      range: definition.range,
      constructedLineRanges: definition.constructedLineRanges,
    });
  }

  /**
   * The authored style of an unchanged source scalar; null for a mapping, a sequence, a
   * virtual value, or a value that differs from the authored scalar.
   *
   * Example: `SourceDocument.parse("a: 'x = 1'\n").root.child("a").scalarStyle` is
   * `"single"`.
   */
  get scalarStyle(): ScalarStyle | null {
    return this._unchangedScalarDefinition()?.style ?? null;
  }

  /**
   * The local tag of an unchanged source value, without the leading `!` (`"list(str)"`);
   * null when the value is untagged or core-tagged, virtual, or differs from the authored
   * value.
   *
   * Example: `SourceDocument.parse("a: !int 5\n").root.child("a").tag` is `"int"`.
   */
  get tag(): string | null {
    if (this._document === null || this._definition === null) return null;
    const definition = this._document._definitions[this._definition]!;
    if (definition.tag === null || !sameValue(this._value, definition.value)) return null;
    return definition.tag;
  }

  /**
   * Whether this is an unchanged source scalar whose text `!include` read from a file, which
   * can't be a type.
   */
  get included(): boolean {
    return this._unchangedScalarDefinition()?.included ?? false;
  }

  /**
   * Whether this is an unchanged source string that is never a template: the text
   * `!include(str)` read, or a string in a value `!include(T)` read.
   */
  get verbatim(): boolean {
    return this._unchangedScalarDefinition()?.verbatim ?? false;
  }

  /**
   * The file an unchanged `!include(T)` value was read from, whose `T` is this value's `tag`;
   * null for any other value.
   */
  get typedInclude(): TypedInclude | null {
    if (this.tag === null) return null;
    return this._document!._typedIncludes.get(this._definition!) ?? null;
  }

  /**
   * The document-local id of the tagged definition this cursor reads, which aliases of one
   * tagged node share; null whenever `tag` is.
   */
  get tagDefinitionId(): number | null {
    return this.tag === null ? null : this._definition;
  }

  /**
   * The authored text of an unchanged source scalar, before YAML resolves its type: a plain
   * `yes` reads `"yes"`, and an included value its `!include` (`"!include a.md"`). Null where
   * `scalarStyle` is.
   */
  get scalarText(): string | null {
    return this._unchangedScalarDefinition()?.raw ?? null;
  }

  /**
   * The authored text of the mapping key selecting this occurrence; null for a root, a
   * sequence item, or a virtual cursor.
   */
  get keyText(): string | null {
    return this._sourceKeyText();
  }

  /**
   * Whether the source wrote a value, not just a tag: false only for an unchanged empty plain
   * scalar that carries a local tag, as in `a: !int`.
   */
  get hasValue(): boolean {
    const definition = this._sourceScalarDefinition();
    return !(
      definition !== null &&
      definition.raw === "" &&
      definition.style === "plain" &&
      this.tag !== null
    );
  }

  /** The selecting mapping key's range; null for a root, a sequence item, or a virtual cursor. */
  get keyRange(): SourceRange | null {
    if (this._document === null || this._keyDefinition === null) return null;
    return this._document._definitions[this._keyDefinition]!.range;
  }

  /** A fresh, unfrozen copy of the value that keeps shared and recursive aliases. */
  get value(): Value {
    return snapshotValue(this._value);
  }

  /**
   * Say whether this cursor's value is exactly `value`.
   *
   * Atoms compare with `===`, so `1n`, `1` and `true` differ; lists compare by items and
   * records by keys, in any order, and values. A recursive alias compares once per pair of
   * containers, so a cyclic value ends.
   */
  holds(value: Value): boolean {
    return sameValue(this._value, value);
  }

  /** @internal Return the frozen value itself, aliases kept. */
  _valueView(): Value {
    return this._value;
  }

  /**
   * Return the mapping value at `key`, or the sequence item at index `key`.
   *
   * A mapping key is text; a sequence index is a non-negative integer, never text.
   *
   * @throws SourceAccessError for a missing key, an invalid or out-of-range index, a scalar,
   *   or a cursor at a recursive alias.
   */
  child(key: string | number): SourceCursor {
    this._ensureTraversable();
    if (this._projectedEntries !== null) {
      const entry = typeof key === "string" ? this._projectedEntry(key) : null;
      if (entry === null) throw new SourceAccessError(missingField(key), this.range);
      return this._projectedChild(entry);
    }
    if (this._sequenceSources !== null) return this._alignedSequenceChild(key);
    if (this._overlay || this._document === null) return this._valueOnlyChild(key);
    const mapping = this._mappingDefinition();
    if (mapping !== null) {
      const occurrence = typeof key === "string" ? this._effectiveOccurrence(key) : null;
      if (occurrence === null) throw new SourceAccessError(missingField(key), this.range);
      return this._descendMappingOccurrence(occurrence);
    }
    const sequence = this._sequenceDefinition();
    if (sequence !== null) {
      checkIndex(key, sequence.itemDefinitions.length, this.range);
      return this._descendSequenceItem(key, sequence.itemDefinitions[key]!);
    }
    throw new SourceAccessError("cannot select a child from a scalar value", this.range);
  }

  /**
   * Return a mapping's values in key order, each from its last pair, or a sequence's items in
   * index order; a scalar has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  children(): readonly SourceCursor[] {
    this._ensureTraversable();
    if (this._projectedEntries !== null) {
      return this._projectedEntries.map((entry) => this._projectedChild(entry));
    }
    if (this._sequenceSources !== null) {
      return this._sequenceSources.map((_, index) => this._alignedSequenceChild(index));
    }
    const value = this._value;
    if (this._overlay || this._document === null) {
      if (isRecord(value)) {
        return Object.keys(value).map((key) => SourceCursor._adopt(value[key]!));
      }
      if (isList(value)) return value.map((item) => SourceCursor._adopt(item as Value));
      return [];
    }
    const mapping = this._mappingDefinition();
    if (mapping !== null) {
      const children: SourceCursor[] = [];
      for (const key of Object.keys(value as ValueRecord)) {
        const occurrence = this._effectiveOccurrence(key);
        if (occurrence !== null) children.push(this._descendMappingOccurrence(occurrence));
      }
      return children;
    }
    const sequence = this._sequenceDefinition();
    if (sequence !== null) {
      return sequence.itemDefinitions.map((id, index) => this._descendSequenceItem(index, id));
    }
    return [];
  }

  /**
   * Return a mapping's pair values in flattened pair order, repeated keys and merged pairs
   * included; only the pairs keyed `key` when given. A sequence or scalar has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  mappingOccurrences(key?: string): readonly SourceCursor[] {
    this._ensureTraversable();
    if (this._projectedEntries !== null) {
      return this._projectedEntries
        .filter((entry) => key === undefined || entry.key === key)
        .map((entry) => this._projectedChild(entry));
    }
    const value = this._value;
    if (this._overlay || this._document === null) {
      if (!isRecord(value)) return [];
      return Object.keys(value)
        .filter((name) => key === undefined || name === key)
        .map((name) => SourceCursor._adopt(value[name]!));
    }
    const definition = this._mappingDefinition();
    if (definition === null) return [];
    if (key !== undefined && !Object.hasOwn(value as ValueRecord, key)) return [];
    const occurrences: SourceCursor[] = [];
    for (const occurrence of this._iterFlatOccurrences(definition.pairSequence)) {
      if (key === undefined || occurrence.arc.key === key) {
        occurrences.push(this._descendMappingOccurrence(occurrence));
      }
    }
    return occurrences;
  }

  /**
   * Say whether the value is a mapping that holds `key`.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  has(key: string): boolean {
    this._ensureTraversable();
    if (this._projectedEntries !== null) return this._projectedEntry(key) !== null;
    if (this._overlay || this._document === null) {
      return isRecord(this._value) && Object.hasOwn(this._value, key);
    }
    return this._effectiveOccurrence(key) !== null;
  }

  /**
   * Return a fresh copy of the mapping value at `key`, or of `fallback` when it is absent.
   *
   * @throws SourceAccessError at a recursive alias, and TypeError when an absent key's
   *   `fallback` holds anything but a YAML value.
   */
  get(key: string, fallback: Value = null): Value {
    if (!this.has(key)) return snapshotValue(fallback);
    return this.child(key).value;
  }

  /**
   * Return the required text field `key`.
   *
   * @throws SourceAccessError when the field is absent, isn't text, or can't be reached.
   */
  str(key: string): string {
    const child = this.child(key);
    const value = child._value;
    if (typeof value !== "string") throw fieldTypeError(key, "a string", value, child.range);
    return value;
  }

  /**
   * Return the required integer field `key`; a boolean or a float is refused.
   *
   * @throws SourceAccessError when the field is absent, isn't an integer, or can't be reached.
   */
  int(key: string): bigint {
    const child = this.child(key);
    const value = child._value;
    if (typeof value !== "bigint") throw fieldTypeError(key, "an integer", value, child.range);
    return value;
  }

  /**
   * Return a fresh, unfrozen copy of the required mapping field `key`.
   *
   * @throws SourceAccessError when the field is absent, isn't a mapping, or can't be reached.
   */
  mapping(key: string): ValueRecord {
    const child = this.child(key);
    const value = child._value;
    if (!isRecord(value)) throw fieldTypeError(key, "a mapping", value, child.range);
    return snapshotValue(value) as ValueRecord;
  }

  /**
   * Return a fresh, unfrozen copy of the required sequence field `key`.
   *
   * @throws SourceAccessError when the field is absent, isn't a sequence, or can't be reached.
   */
  sequence(key: string): Value[] {
    const child = this.child(key);
    const value = child._value;
    if (!isList(value)) throw fieldTypeError(key, "a sequence", value, child.range);
    return snapshotValue(value) as Value[];
  }

  /**
   * Return the range of the required child at mapping key or sequence index `key`; null when
   * the child has no source.
   *
   * @throws SourceAccessError when `key` can't be selected or reached.
   */
  rangeOf(key: string | number): SourceRange | null {
    return this.child(key).range;
  }

  /**
   * Return the key range of the mapping child at `key`; null when it is absent or has no
   * source.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  keyRangeOf(key: string): SourceRange | null {
    this._ensureTraversable();
    if (this._projectedEntries !== null) {
      const entry = this._projectedEntry(key);
      return entry?.source?.keyRange ?? null;
    }
    if (this._overlay || this._document === null) return null;
    const occurrence = this._effectiveOccurrence(key);
    if (occurrence === null) return null;
    return this._document._definitions[occurrence.arc.keyDefinition]!.range;
  }

  /**
   * Return the ranges of the scalar items of the sequence field `key`, by authored text; a
   * repeated text keeps its last range. An absent, unlocated or non-sequence field has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  itemRanges(key: string): Map<string, SourceRange> {
    this._ensureTraversable();
    return this._optionalChild(key)?._ownItemRanges() ?? new Map();
  }

  /**
   * Return one range per item of the sequence field `key`, repeats kept, null for an item that
   * isn't a scalar. An absent or unlocated field has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  itemRangeList(key: string): (SourceRange | null)[] {
    this._ensureTraversable();
    return this._optionalChild(key)?._ownItemRangeList() ?? [];
  }

  /**
   * Return the value ranges of the mapping field `key`'s children, by authored key text. An
   * absent, unlocated or non-mapping field has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  mappingValueRanges(key: string): Map<string, SourceRange> {
    this._ensureTraversable();
    return this._optionalChild(key)?._ownMappingValueRanges() ?? new Map();
  }

  /**
   * Return one range per item of the sequence field `sequenceKey`, in order: the item's
   * `field` value range, or null when the item isn't a mapping or lacks `field`. An absent,
   * unlocated or non-sequence field has none.
   *
   * @throws SourceAccessError at a recursive alias.
   */
  sequenceFieldRanges(sequenceKey: string, field: string): (SourceRange | null)[] {
    this._ensureTraversable();
    return this._optionalChild(sequenceKey)?._ownSequenceFieldRanges(field) ?? [];
  }

  /**
   * The built value of the mapping key selecting this occurrence (`1n` for `1:`); null for a
   * root, a sequence item, or a virtual cursor. Pair it with `keyRange` to tell a YAML null
   * key apart.
   */
  get sourceKeyValue(): Value {
    if (this._document === null || this._keyDefinition === null) return null;
    const definition = this._document._definitions[this._keyDefinition]!;
    return definition.kind === "scalar" ? definition.value : null;
  }

  /**
   * Return a cursor over a frozen snapshot of `value` that keeps this cursor's document, path
   * and ranges. Mapping keys and sequence indexes that still exist keep their child sources,
   * physical or projected; new nested values have none.
   *
   * @throws TypeError when `value` holds anything but a YAML value.
   */
  withValue(value: Value): SourceCursor {
    return this._withSnapshot(freezeValue(snapshotValue(value)));
  }

  /**
   * @internal `withValue` over `snapshot` without copying it; `snapshot` must already be
   * deep-frozen, as a part of another cursor's value is.
   */
  _withSnapshot(snapshot: Value): SourceCursor {
    let projectedEntries: ProjectedEntry[] | null = null;
    let sequenceSources: (SourceCursor | null)[] | null = null;
    if (isRecord(snapshot)) {
      projectedEntries = Object.keys(snapshot).map((key) => {
        let source: SourceCursor | null = null;
        if (this._projectedEntries !== null) {
          // The retained child source, not its projected overlay: a projection may show a
          // placeholder value that no longer lines up with this replacement's children.
          source = this._projectedEntry(key)?.source ?? null;
        } else if (this._document !== null && !this._cycleCut && isRecord(this._value)) {
          source = this._optionalChild(key);
        }
        return Object.freeze({ key, source });
      });
    } else if (isList(snapshot)) {
      const current = this._value;
      const retained = this._sequenceSources;
      const aligned = this._document !== null && !this._cycleCut && isList(current);
      sequenceSources = snapshot.map((_, index) => {
        // The retained item source, not its projected overlay, as for a mapping above.
        if (retained !== null) return retained[index] ?? null;
        if (!aligned || index >= current.length) return null;
        try {
          return this.child(index);
        } catch (error) {
          if (!(error instanceof SourceAccessError)) throw error;
          return null;
        }
      });
    }
    return this.projection(snapshot, projectedEntries, sequenceSources);
  }

  /**
   * Return a mapping projection that keeps this cursor's document, path and ranges: each
   * displayed key shows a frozen snapshot of its value and keeps its source cursor's document,
   * path and ranges, or none when the source is null. Keys follow record key order, so
   * integer-like keys come first.
   *
   * @throws TypeError for a source that isn't a SourceCursor or null, or a value that isn't a
   *   YAML value.
   */
  projectMapping(
    entries: ReadonlyMap<string, readonly [Value, SourceCursor | null]>,
  ): SourceCursor {
    const displayed: ValueRecord = {};
    const sources = new Map<string, SourceCursor | null>();
    for (const [key, [value, source]] of entries) {
      if (source !== null && !(source instanceof SourceCursor)) {
        throw new YianTypeError("projected entry source must be a SourceCursor or null");
      }
      setRecordKey(displayed, key, value);
      sources.set(key, source);
    }
    const snapshot = freezeValue(snapshotValue(displayed)) as ValueRecord;
    const projectedEntries = Object.keys(snapshot).map((key) =>
      Object.freeze({ key, source: sources.get(key) ?? null }),
    );
    return this.projection(snapshot, projectedEntries, null);
  }

  /**
   * Return a mapping projection with each key in `names` shown under its new name; other keys
   * keep theirs, and every child keeps its source.
   *
   * @throws SourceAccessError when the value isn't a mapping, at a recursive alias, or when
   *   two keys would share one name.
   */
  renamed(names: ReadonlyMap<string, string>): SourceCursor {
    this._ensureTraversable();
    const value = this._value;
    if (!isRecord(value)) {
      throw new SourceAccessError("cannot rename keys on a non-mapping value", this.range);
    }
    const entries = new Map<string, readonly [Value, SourceCursor | null]>();
    const origins = new Map<string, string>();
    for (const key of Object.keys(value)) {
      const renamedKey = names.get(key) ?? key;
      const origin = origins.get(renamedKey);
      if (origin !== undefined) {
        // Keeping either entry would silently drop the other authored value.
        const [moved, other] = names.has(key) ? [key, origin] : [origin, key];
        throw new SourceAccessError(
          `key ${this._authoredKey(moved)} and key ${this._authoredKey(other)} both mean ` +
            `${reprText(renamedKey)}; keep one`,
          this.keyRangeOf(moved),
        );
      }
      origins.set(renamedKey, key);
      entries.set(renamedKey, [value[key]!, this.child(key)]);
    }
    return this.projectMapping(entries);
  }

  /** @internal Return the child at `key`, or null when it is absent or can't be reached. */
  _optionalChild(key: string): SourceCursor | null {
    try {
      if (this.has(key)) return this.child(key);
    } catch (error) {
      if (!(error instanceof SourceAccessError)) throw error;
    }
    return null;
  }

  /** @internal Return the scalar definition behind this cursor, or null. */
  _sourceScalarDefinition(): ScalarDef | null {
    if (this._document === null || this._definition === null) return null;
    const definition = this._document._definitions[this._definition]!;
    return definition.kind === "scalar" ? definition : null;
  }

  /** @internal Return the authored scalar definition while this value still equals it. */
  _unchangedScalarDefinition(): ScalarDef | null {
    const definition = this._sourceScalarDefinition();
    if (definition === null || definition.raw === null || this._value !== definition.value) {
      return null;
    }
    return definition;
  }

  /** @internal Return the authored text of this cursor's selecting key, or null. */
  _sourceKeyText(): string | null {
    if (this._document === null || this._keyDefinition === null) return null;
    const definition = this._document._definitions[this._keyDefinition]!;
    return definition.kind === "scalar" ? definition.raw : null;
  }

  /**
   * @internal Return mapping key `key` as written, in backticks and in its quotes when
   * quoted; its `reprText` when it has no authored scalar.
   */
  _authoredKey(key: string): string {
    const child = this._optionalChild(key);
    if (child !== null && child._document !== null && child._keyDefinition !== null) {
      const definition = child._document._definitions[child._keyDefinition]!;
      if (definition.kind === "scalar" && definition.raw !== null) {
        const style = definition.style;
        const quote = style === "single" ? "'" : style === "double" ? '"' : "";
        return `\`${quote}${definition.raw}${quote}\``;
      }
    }
    return reprText(key);
  }

  /** @internal Return this sequence's scalar item ranges by authored text, the last kept. */
  _ownItemRanges(): Map<string, SourceRange> {
    this._ensureTraversable();
    const ranges = new Map<string, SourceRange>();
    if (this._sequenceSources !== null) {
      for (const source of this._sequenceSources) {
        if (source === null) continue;
        const definition = source._sourceScalarDefinition();
        if (definition !== null && definition.raw !== null && source.range !== null) {
          ranges.set(definition.raw, source.range);
        }
      }
      return ranges;
    }
    const sequence = this._sequenceDefinition();
    if (sequence === null) return ranges;
    for (const id of sequence.itemDefinitions) {
      const definition = this._document!._definitions[id]!;
      if (definition.kind === "scalar" && definition.raw !== null && definition.range !== null) {
        ranges.set(definition.raw, definition.range);
      }
    }
    return ranges;
  }

  /** @internal Return this sequence's item ranges by position, null for a non-scalar item. */
  _ownItemRangeList(): (SourceRange | null)[] {
    this._ensureTraversable();
    if (this._sequenceSources !== null) {
      return this._sequenceSources.map((source) =>
        source !== null && source._sourceScalarDefinition() !== null ? source.range : null,
      );
    }
    const sequence = this._sequenceDefinition();
    if (sequence === null) return [];
    return sequence.itemDefinitions.map((id) => {
      const definition = this._document!._definitions[id]!;
      return definition.kind === "scalar" ? definition.range : null;
    });
  }

  /**
   * @internal Return this mapping's child value ranges by key, each key at its first place
   * with its last pair's range.
   */
  _ownMappingValueRanges(): Map<string, SourceRange> {
    this._ensureTraversable();
    if (this._projectedEntries !== null) {
      const ranges = new Map<string, SourceRange>();
      for (const { key, source } of this._projectedEntries) {
        if (source !== null && source.range !== null) ranges.set(key, source.range);
      }
      return ranges;
    }
    const definition = this._mappingDefinition();
    if (definition === null) return new Map();
    // Each distinct rope leaf once, so a shared merge source is read once, not once per use.
    const lastRanges = new Map<string, SourceRange>();
    for (const { arc } of this._iterUniqueOccurrences(definition.pairSequence, true)) {
      const range = this._document!._definitions[arc.valueDefinition]!.range;
      if (range !== null && !lastRanges.has(arc.key)) lastRanges.set(arc.key, range);
    }
    const ranges = new Map<string, SourceRange>();
    for (const { arc } of this._iterUniqueOccurrences(definition.pairSequence)) {
      const located = lastRanges.get(arc.key);
      if (located !== undefined && !ranges.has(arc.key)) ranges.set(arc.key, located);
    }
    return ranges;
  }

  /**
   * @internal Return one range per item of this sequence: the item's effective `field` value
   * range, or null when the item isn't a mapping or lacks `field`.
   */
  _ownSequenceFieldRanges(field: string): (SourceRange | null)[] {
    this._ensureTraversable();
    if (this._sequenceSources !== null) {
      return this._sequenceSources.map(
        (_, index) => this._alignedSequenceChild(index)._optionalChild(field)?.range ?? null,
      );
    }
    const sequence = this._sequenceDefinition();
    if (sequence === null) return [];
    const definitions = this._document!._definitions;
    return sequence.itemDefinitions.map((id) => {
      const definition = definitions[id]!;
      if (definition.kind !== "mapping" || !Object.hasOwn(definition.value, field)) return null;
      const occurrence = definition.effectiveIndex.get(field);
      return occurrence === undefined ? null : definitions[occurrence.arc.valueDefinition]!.range;
    });
  }

  /** @internal Return the projected entry `key` reads, or null. */
  _projectedEntry(key: string): ProjectedEntry | null {
    const index = this._projectedIndex?.get(key);
    return index === undefined ? null : this._projectedEntries![index]!;
  }

  /** @internal Return a projected value's cursor, with its entry's source when it has one. */
  _projectedChild(entry: ProjectedEntry): SourceCursor {
    let built = projectedChildren.get(this);
    if (built === undefined) {
      built = new Map();
      projectedChildren.set(this, built);
    }
    let child = built.get(entry);
    if (child === undefined) {
      const value = (this._value as ValueRecord)[entry.key]!;
      child = entry.source === null ? SourceCursor._adopt(value) : entry.source._withSnapshot(value);
      built.set(entry, child);
    }
    return child;
  }

  /**
   * @internal Return a replacement sequence's item `index`, with its aligned source item when
   * it has one.
   *
   * @throws SourceAccessError when `index` is not a non-negative integer or is out of range.
   */
  _alignedSequenceChild(index: string | number): SourceCursor {
    checkIndex(index, this._sequenceSources!.length, this.range);
    const value = (this._value as Value[])[index]!;
    const source = this._sequenceSources![index] ?? null;
    return source === null ? SourceCursor._adopt(value) : source._withSnapshot(value);
  }

  /** @internal Navigate a value with no navigable definition: virtual, or replaced. */
  _valueOnlyChild(key: string | number): SourceCursor {
    const value = this._value;
    if (isRecord(value)) {
      if (typeof key !== "string" || !Object.hasOwn(value, key)) {
        throw new SourceAccessError(missingField(key), this.range);
      }
      return SourceCursor._adopt(value[key]!);
    }
    if (isList(value)) {
      checkIndex(key, value.length, this.range);
      return SourceCursor._adopt(value[key] as Value);
    }
    throw new SourceAccessError("cannot select a child from a scalar value", this.range);
  }

  /** @internal Refuse to go past a recursive alias occurrence. */
  _ensureTraversable(): void {
    if (this._cycleCut) {
      throw new SourceAccessError(
        "cannot descend through a recursive alias occurrence",
        this.range,
      );
    }
  }

  /** @internal Return the navigable physical mapping definition, or null. */
  _mappingDefinition(): MappingDef | null {
    if (this._overlay || this._document === null || this._definition === null) return null;
    const definition = this._document._definitions[this._definition]!;
    return definition.kind === "mapping" && isRecord(this._value) ? definition : null;
  }

  /** @internal Return the navigable physical sequence definition, or null. */
  _sequenceDefinition(): SequenceDef | null {
    if (this._overlay || this._document === null || this._definition === null) return null;
    const definition = this._document._definitions[this._definition]!;
    return definition.kind === "sequence" && isList(this._value) ? definition : null;
  }

  /** @internal Return the last physical pair keyed `key`, from the effective index, or null. */
  _effectiveOccurrence(key: string): MapOccurrence | null {
    const definition = this._mappingDefinition();
    if (definition === null || !Object.hasOwn(this._value as ValueRecord, key)) return null;
    return definition.effectiveIndex.get(key) ?? null;
  }

  /**
   * @internal Yield every pair of `sequence` in flattened order with its ordinal, on an
   * explicit stack over the merge DAG.
   */
  *_iterFlatOccurrences(sequence: ArcSequence): Generator<MapOccurrence> {
    const stack: (ArcSequence | MapArc)[] = [sequence];
    let pairOrdinal = 0;
    while (stack.length > 0) {
      const item = stack.pop()!;
      if ("valueDefinition" in item) {
        yield { arc: item, pairOrdinal };
        pairOrdinal++;
        continue;
      }
      // A rope holds arcs or parts, never both.
      for (let i = item.arcs.length - 1; i >= 0; i--) stack.push(item.arcs[i]!);
      for (let i = item.parts.length - 1; i >= 0; i--) stack.push(item.parts[i]!);
    }
  }

  /**
   * @internal Yield the pairs of each distinct leaf of `sequence` once, at its first logical
   * place, or at its last and backwards when `reverse`.
   */
  *_iterUniqueOccurrences(sequence: ArcSequence, reverse = false): Generator<MapOccurrence> {
    const seen = new Set<ArcSequence>();
    const stack: [ArcSequence, number][] = [[sequence, 0]];
    while (stack.length > 0) {
      const [current, base] = stack.pop()!;
      if (seen.has(current)) continue;
      seen.add(current);
      const { parts, partOffsets, arcs } = current;
      if (parts.length > 0) {
        // Pushed so the first part (the last when `reverse`) is popped first.
        for (let n = 0; n < parts.length; n++) {
          const i = reverse ? n : parts.length - 1 - n;
          stack.push([parts[i]!, base + partOffsets[i]!]);
        }
        continue;
      }
      for (let n = 0; n < arcs.length; n++) {
        const i = reverse ? arcs.length - 1 - n : n;
        yield { arc: arcs[i]!, pairOrdinal: base + i };
      }
    }
  }

  /**
   * @internal Return the arc at flattened ordinal `ordinal` of `sequence`, without
   * enumerating the pairs before it.
   *
   * @throws SourceAccessError when no pair has that ordinal.
   */
  _flatArcAt(sequence: ArcSequence, ordinal: number): MapArc {
    let current = sequence;
    let local = ordinal;
    while (current.parts.length > 0) {
      // The last part starting at or before `local`, by binary search.
      const offsets = current.partOffsets;
      let low = 0;
      let high = offsets.length;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (offsets[middle]! <= local) low = middle + 1;
        else high = middle;
      }
      if (low === 0) throw new SourceAccessError("invalid mapping occurrence path", this.range);
      local -= offsets[low - 1]!;
      current = current.parts[low - 1]!;
    }
    const arc = current.arcs[local];
    if (arc === undefined) {
      throw new SourceAccessError("invalid mapping occurrence path", this.range);
    }
    return arc;
  }

  /** @internal Return the cursor at one physical mapping occurrence of this mapping. */
  _descendMappingOccurrence(occurrence: MapOccurrence): SourceCursor {
    const { arc, pairOrdinal } = occurrence;
    const step = Object.freeze({ key: arc.key, pairOrdinal });
    return SourceCursor._physical(this._document!, arc.valueDefinition, {
      pathLink: this.childLink(step, arc.valueDefinition),
      keyDefinition: arc.keyDefinition,
      cycleCut: this.onPath(arc.valueDefinition),
    });
  }

  /** @internal Return the cursor at item `index`, definition `definitionId`, of this sequence. */
  _descendSequenceItem(index: number, definitionId: number): SourceCursor {
    return SourceCursor._physical(this._document!, definitionId, {
      pathLink: this.childLink(Object.freeze({ index }), definitionId),
      cycleCut: this.onPath(definitionId),
    });
  }

  /**
   * @internal Replay one occurrence-path step.
   *
   * @throws SourceAccessError when the step doesn't fit this cursor's definition.
   */
  _descendOccurrenceStep(step: OccurrenceStep): SourceCursor {
    this._ensureTraversable();
    if ("key" in step) {
      const definition = this._mappingDefinition();
      const { pairOrdinal } = step;
      if (
        definition === null ||
        !Number.isInteger(pairOrdinal) ||
        pairOrdinal < 0 ||
        pairOrdinal >= definition.pairSequence.flatPairCount
      ) {
        throw new SourceAccessError("invalid mapping occurrence path", this.range);
      }
      const arc = this._flatArcAt(definition.pairSequence, pairOrdinal);
      if (arc.key !== step.key) {
        throw new SourceAccessError("mapping occurrence path key does not match", this.range);
      }
      return this._descendMappingOccurrence({ arc, pairOrdinal });
    }
    const definition = this._sequenceDefinition();
    const { index } = step;
    if (
      definition === null ||
      !Number.isInteger(index) ||
      index < 0 ||
      index >= definition.itemDefinitions.length
    ) {
      throw new SourceAccessError("invalid sequence occurrence path", this.range);
    }
    return this._descendSequenceItem(index, definition.itemDefinitions[index]!);
  }

  /** Return a cursor over replacement value `value` that keeps this cursor's source. */
  private projection(
    value: Value,
    projectedEntries: readonly ProjectedEntry[] | null,
    sequenceSources: readonly (SourceCursor | null)[] | null,
  ): SourceCursor {
    return new SourceCursor({
      document: this._document,
      definition: this._definition,
      value,
      pathLink: this._pathLink,
      keyDefinition: this._keyDefinition,
      cycleCut: this._cycleCut,
      overlay: this._document !== null,
      projectedEntries: projectedEntries && Object.freeze(projectedEntries),
      // A snapshot's record keys are distinct, so each key's last position is its only one.
      projectedIndex:
        projectedEntries === null
          ? null
          : new Map(projectedEntries.map((entry, index) => [entry.key, index])),
      sequenceSources: sequenceSources && Object.freeze(sequenceSources),
    });
  }

  /** Return the path link one step below this cursor's. */
  private childLink(step: OccurrenceStep, definitionId: number): PathLink {
    return Object.freeze({ parent: this._pathLink, step, definitionId });
  }

  /** Say whether this cursor's path, itself included, already reaches `definitionId`. */
  private onPath(definitionId: number): boolean {
    for (let link = this._pathLink; link !== null; link = link.parent) {
      if (link.definitionId === definitionId) return true;
    }
    return false;
  }
}

/** Return the message for a missing mapping key. */
function missingField(key: string | number): string {
  return `missing required field ${typeof key === "string" ? reprText(key) : String(key)}`;
}

/**
 * Return the error for a field `key` whose `value` isn't `expected`, naming the value's type by
 * its type name (`int`, `NoneType`).
 */
function fieldTypeError(
  key: string,
  expected: string,
  value: Value,
  range: SourceRange | null,
): SourceAccessError {
  return new SourceAccessError(
    `field ${reprText(key)} must be ${expected}, got ${typeName(value)}`,
    range,
  );
}

/**
 * Check that `key` indexes a sequence of `length` items.
 *
 * @throws SourceAccessError over `range` when `key` is not a non-negative integer or is out
 *   of range.
 */
function checkIndex(
  key: string | number,
  length: number,
  range: SourceRange | null,
): asserts key is number {
  if (typeof key !== "number" || !Number.isInteger(key) || key < 0) {
    throw new SourceAccessError("sequence index must be a non-negative integer", range);
  }
  if (key >= length) throw new SourceAccessError(`sequence index ${key} is out of range`, range);
}

/**
 * Compare two YAML values deeply, on an explicit stack: atoms with `===`, lists by length and
 * items, records by key sets and values; a list never equals a record. A pair of containers
 * compares once, so cyclic values end.
 */
function sameValue(left: Value, right: Value): boolean {
  const stack: [Value, Value][] = [[left, right]];
  const seen = new Map<object, Set<object>>();
  /** Say whether the pair `first`, `second` was compared before, and note it. */
  const compared = (first: object, second: object): boolean => {
    let partners = seen.get(first);
    if (partners === undefined) {
      partners = new Set();
      seen.set(first, partners);
    }
    if (partners.has(second)) return true;
    partners.add(second);
    return false;
  };
  while (stack.length > 0) {
    const [first, second] = stack.pop()!;
    if (first === second) continue;
    if (isList(first) && isList(second)) {
      if (compared(first, second)) continue;
      if (first.length !== second.length) return false;
      for (let i = 0; i < first.length; i++) stack.push([first[i] as Value, second[i] as Value]);
    } else if (isRecord(first) && isRecord(second)) {
      if (compared(first, second)) continue;
      const keys = Object.keys(first);
      if (keys.length !== Object.keys(second).length) return false;
      for (const key of keys) {
        if (!Object.hasOwn(second, key)) return false;
        stack.push([first[key]!, second[key]!]);
      }
    } else {
      return false;
    }
  }
  return true;
}
