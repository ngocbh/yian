/** The memory types a load knows, and how a flow input's tag reads one. */

import { reprText } from "../../diagnostics.ts";
import { BUILTIN_MEMORY_TOOLS, type MemoryBase } from "../../memory/base.ts";
import { parseType, type TypeExpr, type TypeRegistry, withMemoryTypes } from "../../typesys/types.ts";
import { TypeCheckError } from "../../typesys/values.ts";
import type { SourceCursor } from "../source/index.ts";
import type { LoadDiagnostics } from "./errors.ts";

/** A memory class a host passes to a load: its constructor, which declares the contract. */
export type MemoryClass = typeof MemoryBase;

/** How one file's flow inputs read memory types beyond the built-ins. */
export interface MemoryTypeTable {
  /** Each host memory type's tool names, in spec order, by its contract type. */
  readonly tools: ReadonlyMap<string, readonly string[]>;
  /**
   * Whether an unknown name on a flow input reads as a memory type: true only when a recorded
   * flow is compiled again, whose start already checked its memories.
   */
  readonly opaque: boolean;
  /** The host memory types this file also declares as typedefs; an input may name neither. */
  readonly clashes: ReadonlySet<string>;
}

/** The table of a load that knows only the built-in memory types. */
export const NO_MEMORY_TYPES: MemoryTypeTable = Object.freeze({
  tools: new Map(),
  opaque: false,
  clashes: new Set<string>(),
});

/**
 * Return the tool names of each memory class a host passes in `memoryTypes`, by contract type.
 * A class implementing a built-in contract is skipped: the built-in names it already.
 *
 * @throws TypeError when `memoryTypes` isn't a list of classes each declaring a contract with a
 *   type a tag can write and named tools, or lists two classes for one type with different
 *   tools.
 */
export function memoryTypeTools(memoryTypes: unknown): ReadonlyMap<string, readonly string[]> {
  if (!Array.isArray(memoryTypes)) throw new TypeError("memoryTypes must be a list of memory classes");
  const tools = new Map<string, readonly string[]>();
  for (const cls of memoryTypes) {
    if (typeof cls !== "function") throw new TypeError("memoryTypes must hold memory classes");
    const contract: unknown = (cls as Partial<MemoryClass>).contract;
    const name = (cls as { name?: unknown }).name;
    if (typeof contract !== "object" || contract === null) {
      throw new TypeError(`${String(name)} declares no memory contract; give it a static contract`);
    }
    const { type, tools: specs } = contract as { type?: unknown; tools?: unknown };
    if (typeof type !== "string" || !Array.isArray(specs)) {
      throw new TypeError(`${String(name)}'s contract needs a type and a list of tools`);
    }
    const names = specs.map((spec: unknown) => (spec as { name?: unknown } | null)?.name);
    if (!names.every((tool): tool is string => typeof tool === "string" && tool !== "")) {
      throw new TypeError(`${String(name)}'s contract tools need names`);
    }
    if (Object.hasOwn(BUILTIN_MEMORY_TOOLS, type)) continue;
    try {
      withMemoryTypes(new Map(), [type]);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw new TypeError(`${String(name)}: ${error.message}`, { cause: error });
    }
    const known = tools.get(type);
    if (known !== undefined && known.join() !== names.join()) {
      throw new TypeError(`memoryTypes lists two classes for ${reprText(type)} with different tools`);
    }
    tools.set(type, Object.freeze(names));
  }
  return tools;
}

/**
 * Return `registry` with the memory types a flow input's tag `tag` may name added: under
 * `table.opaque`, the name the tag holds whole or in `Optional(...)` when the registry and the
 * built-ins don't know it.
 *
 * @throws LoadError when that name is a host memory type the file also declares as a typedef,
 *   framed at the typedef's key and then at the input.
 */
export function inputTagRegistry(
  tag: string,
  registry: TypeRegistry,
  table: MemoryTypeTable,
  options: { label: string; cursor: SourceCursor; diagnostics: LoadDiagnostics },
): TypeRegistry {
  const name = wholeName(tag);
  if (name === null) return registry;
  const { label, cursor, diagnostics } = options;
  if (table.clashes.has(name)) {
    const typedefs = cursor.document?.root.child("typedefs") ?? null;
    const frames = [
      diagnostics.frame(typedefs?.keyRangeOf(name) ?? null, { note: `typedef ${reprText(name)}` }),
      diagnostics.frame(cursor.range),
    ].filter((frame) => frame !== null);
    throw diagnostics.error(
      `${reprText(label)}: the memory type \`!${name}\` is also a typedef name; rename the typedef or ` +
        "the memory type",
      { frames },
    );
  }
  if (!table.opaque || registry.has(name)) return registry;
  try {
    return withMemoryTypes(registry, [name]);
  } catch (error) {
    // A name no host type could take (a scalar, a built-in record) keeps its own meaning.
    if (error instanceof TypeCheckError) return registry;
    throw error;
  }
}

/** Return the name type text `tag` is, alone or in `Optional(...)`, or `null`. */
function wholeName(tag: string): string | null {
  let parsed: TypeExpr;
  try {
    parsed = parseType(tag);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
  while (parsed.kind === "optional") parsed = parsed.inner;
  return parsed.kind === "ref" ? parsed.name : null;
}
