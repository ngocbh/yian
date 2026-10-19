/** Source-aware parsing of authored `long_term_memory:` declarations. */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { type DeclSite, MemoryDecl } from "../../memory/contracts.ts";
import { isRecord, type Value, type ValueRecord, writeAsText } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "./errors.ts";

const LONG_FORM_KEYS: ReadonlySet<string> = new Set(["type", "config"]);
// Reserved so the retired `{name:, config:}` form and a stray top-level `type:` can never
// parse as memories.
const RESERVED_NAMES: readonly string[] = ["config", "name", "type"];

/** Where `parseMemoryDeclarations` reports errors and what it stamps on each declaration. */
interface MemoryParseOptions {
  /** Source context for errors. */
  diagnostics: LoadDiagnostics;
  /** Range of the whole value. */
  sourceRange: SourceRange | null;
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
 * @throws LoadError at `sourceRange` when the value has the wrong shape or a declaration is
 *   invalid.
 */
export function parseMemoryDeclarations(value: Value, options: MemoryParseOptions): readonly MemoryDecl[] {
  const { diagnostics, sourceRange } = options;
  if (typeof value === "string") {
    rejectReserved(value, options);
    return Object.freeze([declaration(value, value, {}, options)]);
  }
  if (!isRecord(value)) {
    throw diagnostics.error(
      "long_term_memory must be a type name or a mapping of memory names to types",
      { range: sourceRange },
    );
  }
  const entries = Object.entries(value);
  if (entries.length === 0) {
    throw diagnostics.error("long_term_memory: {} declares no memory; remove the key or name one", {
      range: sourceRange,
    });
  }
  const declarations: MemoryDecl[] = [];
  for (const [name, entry] of entries) {
    rejectReserved(name, options);
    if (typeof entry === "string") {
      declarations.push(declaration(name, entry, {}, options));
      continue;
    }
    if (!isRecord(entry)) {
      throw diagnostics.error(
        `long_term_memory ${reprText(name)} must be a type name or a mapping ` +
          "with 'type' and optional 'config'",
        { range: sourceRange },
      );
    }
    const extras = Object.keys(entry)
      .filter((key) => !LONG_FORM_KEYS.has(key))
      .sort(compareCodeUnits);
    if (extras.length > 0) {
      throw diagnostics.error(
        `long_term_memory ${reprText(name)} has unknown key(s) ${writeAsText(extras)} ` +
          "(allowed: type, config)",
        { range: sourceRange },
      );
    }
    if (!Object.hasOwn(entry, "type")) {
      throw diagnostics.error(`long_term_memory ${reprText(name)} requires 'type'`, { range: sourceRange });
    }
    const type = entry["type"];
    if (typeof type !== "string") {
      throw diagnostics.error(`long_term_memory ${reprText(name)} type must be a string`, {
        range: sourceRange,
      });
    }
    const config = Object.hasOwn(entry, "config") ? entry["config"] : {};
    if (!isRecord(config)) {
      throw diagnostics.error(`long_term_memory ${reprText(name)} config must be a mapping`, {
        range: sourceRange,
      });
    }
    declarations.push(declaration(name, type, config, options));
  }
  return Object.freeze(declarations);
}

/** Throw a load error when `name` is one of the reserved memory names. */
function rejectReserved(name: string, { diagnostics, sourceRange }: MemoryParseOptions): void {
  if (RESERVED_NAMES.includes(name)) {
    throw diagnostics.error(
      `long_term_memory name ${reprText(name)} is reserved (reserved: ${RESERVED_NAMES.join(", ")})`,
      { range: sourceRange },
    );
  }
}

/** Build one declaration, reporting a shape failure at the authored value. */
function declaration(
  name: string,
  type: string,
  config: ValueRecord,
  { diagnostics, sourceRange, site }: MemoryParseOptions,
): MemoryDecl {
  try {
    return new MemoryDecl(name, type, config, site);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) {
      throw diagnostics.error(`long_term_memory ${reprText(name)}: ${error.message}`, {
        range: sourceRange,
        cause: error,
      });
    }
    throw error;
  }
}
