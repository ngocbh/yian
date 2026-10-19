/**
 * Parse-free source helpers for the Compose front end.
 *
 * This module projects synthesized boundary-input ranges from an existing source cursor,
 * slices authored text out of a document, and reads positional metadata. It never parses,
 * keeping the helper layer below parsing. Assertion origins and temporary ranges are
 * deliberately absent: the build conductor derives both from one retained item cursor.
 */

import type { SourceRange } from "../../diagnostics.ts";
import { isRecord } from "../../typesys/index.ts";
import { SourceAccessError, type SourceCursor, type SourceDocument } from "../source/index.ts";

/** Return whether a top-level key is an `x-*` extension key, which a load ignores. */
export function isExtensionKey(key: string): boolean {
  return key.startsWith("x-");
}

/**
 * Return the `index`-th item of positional parse metadata, or `null` past either end.
 * The metadata is aligned with a list, such as a case's `thenRanges`, and is empty or
 * short for a source-free descriptor.
 */
export function nth<T>(items: readonly T[], index: number): T | null {
  return 0 <= index && index < items.length ? items[index]! : null;
}

/**
 * Return the authored text a one-line range covers, as written; `null` when there is no
 * document or range, the range spans more than one line, or it lies past the text.
 */
export function oneLineText(
  document: SourceDocument | null,
  range: SourceRange | null,
): string | null {
  if (document === null || range === null || range.startLine !== range.endLine) return null;
  // Lines break as ranges count them (`rule: columns`); no empty line after a final break.
  const lines = document.text.split(/\r\n|\r|\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  if (!(1 <= range.startLine && range.startLine <= lines.length)) return null;
  return lines[range.startLine - 1]!.slice(range.startCol - 1, range.endCol - 1);
}

/**
 * Return Compose-owned ranges for authored flow input declarations: each top-level
 * `input:` (or `inputs:`) declaration name -> its declaration value's range.
 *
 * Assembly uses the map while compiling flow parameters and discards it before semantic
 * graph construction. Flow-level `asserts:` are not here: assembly derives their ranges and
 * origins from the retained assertion cursor. Empty for a spanless or in-memory flow. This
 * projection never throws a `SourceAccessError`.
 */
export function boundaryFieldRanges(root: SourceCursor): Map<string, SourceRange> {
  try {
    const value = root._valueView();
    const out = new Map<string, SourceRange>();
    if (!isRecord(value)) return out;
    for (const key of ["input", "inputs"]) {
      if (isRecord(value[key])) {
        for (const [name, range] of root.mappingValueRanges(key)) out.set(name, range);
      }
    }
    return out;
  } catch (error) {
    if (error instanceof SourceAccessError) return new Map();
    throw error;
  }
}
