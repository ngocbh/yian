/** Source-aware parsing of authored `long_term_memory:` declarations. */

import { compareCodeUnits, getCloseMatches, reprText, type SourceRange } from "../../diagnostics.ts";
import { type DeclSite, MemoryDecl } from "../../memory/contracts.ts";
import { isRecord, type Value, type ValueRecord } from "../../typesys/index.ts";
import { writeAsText } from "../../typesys/values.ts";
import type { SourceCursor } from "../source/index.ts";
import type { LoadDiagnostics } from "./errors.ts";
import { CLOSE_MATCH_CUTOFF } from "./tags.ts";

const LONG_FORM_KEYS: readonly string[] = ["type", "config"];
// Reserved so the retired `{name:, config:}` form and a stray top-level `type:` can never
// parse as memories.
const RESERVED_NAMES: readonly string[] = ["config", "name", "type"];

/** Where `parseMemoryDeclarations` reports errors and what it stamps on each declaration. */
export interface MemoryParseOptions {
  /** Source context for errors. */
  diagnostics: LoadDiagnostics;
  /** The authored value, which locates each error; `null` leaves errors unlocated. */
  cursor: SourceCursor | null;
  /** Definition site stamped on every declaration. */
  site: DeclSite;
}

/**
 * Parse one present, non-null `long_term_memory:` value into declarations, as a frozen array.
 *
 * A type name is sugar for `{<type>: <type>}`. A record declares one memory per entry, as
 * `name: type` or `name: {type: ..., config: {...}}`, in authored order. Origins are attached
 * later by the caller, which treats absence and `null` as no declaration and never passes them
 * here.
 *
 * @throws LoadError when the value has the wrong shape or a declaration is invalid, located at
 *   the offending entry, key or field (the whole value for a shape or sugar error).
 */
export function parseMemoryDeclarations(value: Value, options: MemoryParseOptions): readonly MemoryDecl[] {
  const { diagnostics, cursor } = options;
  const whole = cursor?.range ?? null;
  if (typeof value === "string") {
    rejectReserved(value, whole, diagnostics);
    return Object.freeze([declaration(value, value, {}, whole, options)]);
  }
  if (!isRecord(value)) {
    throw diagnostics.error(
      "long_term_memory must be a type name or a mapping of memory names to types",
      { range: whole },
    );
  }
  const entries = Object.entries(value);
  if (entries.length === 0) {
    throw diagnostics.error("long_term_memory: {} declares no memory; remove the key or name one", {
      range: whole,
    });
  }
  const declarations: MemoryDecl[] = [];
  for (const [name, entry] of entries) {
    const at = cursor?.child(name) ?? null;
    const entryRange = at?.range ?? whole;
    rejectReserved(name, cursor?.keyRangeOf(name) ?? entryRange, diagnostics);
    if (typeof entry === "string") {
      declarations.push(declaration(name, entry, {}, entryRange, options));
      continue;
    }
    if (!isRecord(entry)) {
      throw diagnostics.error(
        `long_term_memory ${reprText(name)} must be a type name or a mapping ` +
          "with 'type' and optional 'config'",
        { range: entryRange },
      );
    }
    const extras = Object.keys(entry)
      .filter((key) => !LONG_FORM_KEYS.includes(key))
      .sort(compareCodeUnits);
    if (extras.length > 0) {
      const first = extras[0]!;
      const close = getCloseMatches(first, LONG_FORM_KEYS, 1, CLOSE_MATCH_CUTOFF)[0];
      throw diagnostics.error(
        `long_term_memory ${reprText(name)} has unknown key(s) ${writeAsText(extras)} ` +
          "(allowed: type, config)" +
          (close === undefined ? "" : `; did you mean ${reprText(close)}?`),
        { range: at?.keyRangeOf(first) ?? entryRange },
      );
    }
    if (!Object.hasOwn(entry, "type")) {
      throw diagnostics.error(`long_term_memory ${reprText(name)} requires 'type'`, { range: entryRange });
    }
    const type = entry["type"];
    if (typeof type !== "string") {
      throw diagnostics.error(`long_term_memory ${reprText(name)} type must be a string`, {
        range: at?.rangeOf("type") ?? entryRange,
      });
    }
    const config = Object.hasOwn(entry, "config") ? entry["config"] : {};
    if (!isRecord(config)) {
      throw diagnostics.error(`long_term_memory ${reprText(name)} config must be a mapping`, {
        range: at?.rangeOf("config") ?? entryRange,
      });
    }
    declarations.push(declaration(name, type, config, entryRange, options));
  }
  return Object.freeze(declarations);
}

/** Throw a load error at `range` when `name` is one of the reserved memory names. */
function rejectReserved(name: string, range: SourceRange | null, diagnostics: LoadDiagnostics): void {
  if (RESERVED_NAMES.includes(name)) {
    throw diagnostics.error(
      `long_term_memory name ${reprText(name)} is reserved (reserved: ${RESERVED_NAMES.join(", ")})`,
      { range },
    );
  }
}

/** Build one declaration, reporting a shape failure at `range`, the authored entry. */
function declaration(
  name: string,
  type: string,
  config: ValueRecord,
  range: SourceRange | null,
  { diagnostics, site }: MemoryParseOptions,
): MemoryDecl {
  try {
    return new MemoryDecl(name, type, config, site);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) {
      throw diagnostics.error(`long_term_memory ${reprText(name)}: ${error.message}`, {
        range,
        cause: error,
      });
    }
    throw error;
  }
}
