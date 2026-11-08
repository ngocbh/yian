/** The memory types a load knows, and how a flow input's tag reads one. */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { BUILTIN_MEMORY_TOOLS, type MemoryBase } from "../../memory/base.ts";
import { MEMORY_KEY_PREFIX } from "../../nodes/base.ts";
import { checkVisibleNames, expandGrants, GrantError, type GrantScope, type ToolGrant } from "../../tools/grants.ts";
import { RESERVED_TOOL_NAMES } from "../../tools/registry.ts";
import { parseType, type TypeExpr, type TypeRegistry, withMemoryTypes } from "../../typesys/types.ts";
import { isMemoryType, type Type, TypeCheckError } from "../../typesys/values.ts";
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
 *   type a tag can write and named tools, each once, or lists two classes for one type with
 *   different tools.
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
    const twice = names.find((tool, i) => names.indexOf(tool) !== i);
    if (twice !== undefined) throw new TypeError(`${String(name)}'s contract lists the tool ${twice} twice`);
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

/**
 * Return the tool names of memory type `typ`, in spec order: a built-in contract's, or a host
 * type's from `table`; `null` for a type this load doesn't know, a recorded flow's opaque one.
 */
export function memoryToolNames(
  typ: Type & { readonly memory: string },
  table: MemoryTypeTable,
): readonly string[] | null {
  if (Object.hasOwn(BUILTIN_MEMORY_TOOLS, typ.memory)) {
    return BUILTIN_MEMORY_TOOLS[typ.memory as keyof typeof BUILTIN_MEMORY_TOOLS];
  }
  return table.tools.get(typ.memory) ?? null;
}

/**
 * Check grants of the memory inputs `flowInputs` types as memories: each tool a grant lists is
 * one of its type's, and each tool shows to a model under a name it can call, which yian doesn't
 * reserve and no other memory tool of the same grants shows. A grant on a type this load doesn't
 * know (a recorded flow's) is left unchecked: the run's start checked it.
 *
 * @throws GrantError naming the grant and the fix.
 */
export function checkMemoryGrants(
  grants: readonly ToolGrant[],
  flowInputs: ReadonlyMap<string, Type>,
  table: MemoryTypeTable,
): void {
  const names = (input: string): readonly string[] | null => {
    const typ = flowInputs.get(input);
    return isMemoryType(typ) ? memoryToolNames(typ, table) : null;
  };
  const known = grants.filter((grant) => names(grant.kind === "name" ? grant.name : grant.group) !== null);
  const scope: GrantScope = {
    label: "this flow's memories",
    group: names,
    hasTool: () => false,
    groupsHolding: () => [],
    groupNames: () => [...flowInputs.keys()].filter((input) => names(input) !== null),
    // Structured, so two tools that would show one name stay two tools for the name check.
    identity: (group, tool) => JSON.stringify([group, tool]),
    bareMembers: false,
  };
  const granted = expandGrants(known, [scope]);
  checkVisibleNames(granted);
  for (const tool of granted) {
    if (!RESERVED_TOOL_NAMES.has(tool.visible)) continue;
    throw new GrantError(
      `the tool name \`${tool.visible}\` of \`{${tool.group}: [${tool.tool}]}\` is yian's; rename the input`,
    );
  }
}

/**
 * Return `error`'s message after `where` (`node 'a' (kind=agent)`), joined as each kind of grant
 * error reads: `… grants {x: [y]}: …`, or `…: the tool name … is yian's; …`.
 */
export function grantErrorText(where: string, error: GrantError): string {
  return `${where}${error.message.startsWith("grants ") ? " " : ": "}${error.message}`;
}

/** Return the source a memory grant's reserved wiring binds: the whole `${input.<input>}`. */
export function memoryBinding(input: string): string {
  return `\${input.${input}}`;
}

/**
 * Refuse a node input named like a memory grant's reserved wiring key (`__memory_<name>`).
 *
 * @param rangeOf The authored range of input `name`, or `null`.
 * @throws LoadError at the first such input.
 */
export function rejectMemoryKeyInputs(
  names: Iterable<string>,
  rangeOf: (name: string) => SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  for (const name of names) {
    if (!name.startsWith(MEMORY_KEY_PREFIX)) continue;
    throw diagnostics.error(`input ${reprText(name)} is reserved for memory grants; rename the input`, {
      range: rangeOf(name),
    });
  }
}
