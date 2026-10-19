/**
 * Reject a written-out value that nests deeper than a yian value may, or holds a lone
 * surrogate, as a load error.
 *
 * A value written in a flow file is checked at load with the counting the run-time check
 * applies to it (`rejectDeepNesting`), so it fails before any recursive load walker reads it,
 * even when YAML aliases build its depth while the text stays shallow. `${...}` text counts as
 * a scalar; the run-time check still covers computed values. Text or a record key holding a
 * lone surrogate (only a double-quoted `\uD800` escape can write one) is refused too, since it
 * is not a yian value. An `env:` value that holds itself through a recursive alias is rejected
 * the same way, since the run's fingerprint cannot encode it.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  recursiveEntryPath,
  rejectDeepNesting,
  rejectLoneSurrogates,
  renderValuePath,
  TypeCheckError,
  type ValueRecord,
} from "../../typesys/values.ts";
import type { LoadDiagnostics } from "./errors.ts";

/**
 * Reject `value` if it nests deeper than `current().maxValueNesting` levels, or if text or a
 * record key in it holds a lone surrogate (checked after the depth).
 *
 * @param where Names the site for the message, e.g. `"node 'use' input 'topic'"`.
 * @param range Where the error points.
 * @param options.enclosingLevels Levels the value sits in at run time, as for
 *   `rejectDeepNesting` (default 0).
 * @throws LoadError at `range` with the run-time message, its `TypeCheckError` as `cause`.
 */
export function checkWrittenValue(
  value: unknown,
  where: string,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
  options: { enclosingLevels?: number } = {},
): void {
  try {
    rejectDeepNesting(value, where, { enclosingLevels: options.enclosingLevels ?? 0 });
    rejectLoneSurrogates(value, where);
  } catch (error) {
    if (error instanceof TypeCheckError) throw diagnostics.error(error.message, { range, cause: error });
    throw error;
  }
}

/**
 * Reject an `env:` key holding a lone surrogate, or a value that nests too deep, holds a lone
 * surrogate, or holds itself through an alias, checked in that order per entry.
 *
 * @param prefix Starts the message before `env '<key>'`, e.g. `"flow "`.
 * @param envCursor Cursor over the `env:` mapping, for ranges.
 * @throws LoadError at the entry's key (else at `env:`) when the key holds a lone surrogate:
 *   `<prefix>env '<key>': text holding a lone surrogate is not a yian value`.
 * @throws LoadError at the entry's value (else at `env:`) as `checkWrittenValue` throws it, or
 *   when an entry in the value refers back to a container enclosing it: `<prefix>env '<key>' at
 *   <path>: cannot descend through a recursive alias occurrence`.
 */
export function checkEnvValues(
  env: ValueRecord,
  prefix: string,
  envCursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
): void {
  for (const [key, value] of Object.entries(env)) {
    const where = `${prefix}env ${reprText(key)}`;
    // The key is text, so this checks only its lone surrogates.
    checkWrittenValue(key, where, envCursor?.keyRangeOf(key) ?? envCursor?.range ?? null, diagnostics);
    const range = envCursor?.rangeOf(key) ?? envCursor?.range ?? null;
    checkWrittenValue(value, where, range, diagnostics);
    const path = recursiveEntryPath(value);
    if (path !== null) {
      throw diagnostics.error(
        `${where} at ${renderValuePath(path)}: cannot descend through a recursive alias occurrence`,
        { range },
      );
    }
  }
}
