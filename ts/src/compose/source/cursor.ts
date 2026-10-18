/** Immutable cursors over physical and virtual YAML source values. */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { isList, isRecord, type Value, type ValueRecord } from "../../typesys/index.ts";
import {
  type ArcSequence,
  freezeValue,
  type MapArc,
  type MapOccurrence,
  type MappingDef,
  type OccurrencePath,
  type OccurrenceStep,
  type PathLink,
  type SequenceDef,
  snapshotValue,
  SourceAccessError,
} from "./common.ts";
// Type-only: `document.ts` imports this module at runtime.
import type { SourceDocument } from "./document.ts";

/** @internal One projected mapping entry and its optional physical source cursor. */
interface ProjectedEntry {
  readonly key: string;
  readonly source: SourceCursor | null;
}

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
 * cursors keep only a frozen snapshot of their value. Cursors compare by identity.
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
    return new SourceCursor({
      document: null,
      definition: null,
      value: freezeValue(snapshotValue(value)),
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
    const value = this._value;
    if (this._overlay || this._document === null) {
      if (isRecord(value)) {
        return Object.keys(value).map((key) => SourceCursor.virtual(value[key]!));
      }
      if (isList(value)) return value.map((item) => SourceCursor.virtual(item as Value));
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
    const value = this._value;
    if (this._overlay || this._document === null) {
      if (!isRecord(value)) return [];
      return Object.keys(value)
        .filter((name) => key === undefined || name === key)
        .map((name) => SourceCursor.virtual(value[name]!));
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

  /** @internal Navigate a value with no navigable definition: virtual, or replaced. */
  _valueOnlyChild(key: string | number): SourceCursor {
    const value = this._value;
    if (isRecord(value)) {
      if (typeof key !== "string" || !Object.hasOwn(value, key)) {
        throw new SourceAccessError(missingField(key), this.range);
      }
      return SourceCursor.virtual(value[key]!);
    }
    if (isList(value)) {
      checkIndex(key, value.length, this.range);
      return SourceCursor.virtual(value[key] as Value);
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
      // The last part starting at or before `local`, as Python's `bisect_right` minus one.
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
      pathLink: this._childLink(step, arc.valueDefinition),
      keyDefinition: arc.keyDefinition,
      cycleCut: this._onPath(arc.valueDefinition),
    });
  }

  /** @internal Return the cursor at item `index`, definition `definitionId`, of this sequence. */
  _descendSequenceItem(index: number, definitionId: number): SourceCursor {
    return SourceCursor._physical(this._document!, definitionId, {
      pathLink: this._childLink(Object.freeze({ index }), definitionId),
      cycleCut: this._onPath(definitionId),
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

  /** Return the path link one step below this cursor's. */
  private _childLink(step: OccurrenceStep, definitionId: number): PathLink {
    return Object.freeze({ parent: this._pathLink, step, definitionId });
  }

  /** Say whether this cursor's path, itself included, already reaches `definitionId`. */
  private _onPath(definitionId: number): boolean {
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
