/**
 * Compose type declarations -> source-neutral runtime contracts.
 *
 * A node here is a leaf type string (`"float"`, `"list(str)"`, a registry name), a map of
 * field -> node (a record, nested natively), or a list of single-key maps (the list-of-fields
 * style). Resolution of leaf strings and nested records is the type layer's job (`typeFor`
 * recurses over aliases/lists/Optional); this reader walks native record syntax and wraps any
 * type error loudly. A record counts its levels against `current().maxTypeNesting`: each
 * record is a level and a field's type adds its own (see `typeLevels`), however YAML text or
 * aliases built the record, so a deep alias chain is a load error before the walk recurses
 * past the limit. `readType` is the source-free semantic view. `readValueContract` performs
 * the same recursion over a retained cursor and compiles authored root/nested sites into
 * opaque `ContractOrigins` without leaking YAML concepts below Compose.
 */

import { type OriginRef, reprText } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import { comparePaths, ContractOrigins, ValueContract, type ValuePath } from "../../nodes/contracts.ts";
import type { SourceCatalogBuilder, SourceSitePath } from "../source/catalog.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";
import { encodesAsUtf8, pythonTypeName, type TypeRegistry, typeFor } from "../../typesys/types.ts";
import {
  isRecord,
  renderValuePath,
  Type,
  TypeCheckError,
  typeLevels,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import { LoadDiagnostics } from "./errors.ts";

/** One physically authored record field: its path from the declaration's root and source. */
type Site = readonly [ValuePath, SourceCursor];

/** Return the message for a type at `path` nesting past `limit` levels. */
function tooDeep(path: ValuePath, limit: number): string {
  const location = path.length > 0 ? `type at ${renderValuePath(path)}` : "type";
  return (
    `${location} nests deeper than ${limit} levels, ` +
    "the limit for a type (each record and list is a level)"
  );
}

/**
 * Read normalized `[field, value, cursor]` entries into one record type.
 *
 * `levels` counts the records enclosing each field, this one included.
 *
 * @throws LoadError when a field name is empty or holds a Unicode surrogate, or a field's
 *   type can't be read.
 */
function readRecordType(
  entries: readonly (readonly [string, unknown, SourceCursor | null])[],
  registry: TypeRegistry,
  options: { path: ValuePath; diagnostics: LoadDiagnostics; levels: number },
): [Type, Site[]] {
  const { path, diagnostics, levels } = options;
  const fields = new Map<string, Type>();
  const sites: Site[] = [];
  for (const [key, value, childCursor] of entries) {
    // Checked without a source too, where Python only checks with one: a source-free read
    // then gets the same load error rather than the raw error `Type` gives such a name.
    const range = childCursor === null ? null : (childCursor.keyRange ?? childCursor.range);
    if (!key) {
      throw diagnostics.error("record field names must be non-empty strings", { range });
    }
    if (!encodesAsUtf8(key)) {
      throw diagnostics.error(
        `record field name ${reprText(key)}: field names must not contain a Unicode surrogate`,
        { range },
      );
    }
    const childPath = [...path, key];
    if (childCursor !== null) sites.push([childPath, childCursor]);
    const [fieldType, childSites] = readTypeAndSites(value, registry, {
      cursor: childCursor,
      path: childPath,
      diagnostics,
      levels,
    });
    fields.set(key, fieldType);
    sites.push(...childSites);
  }
  const required = new Set([...fields].filter(([, fieldType]) => !fieldType.nullable).map(([key]) => key));
  return [new Type({ kind: ValueKind.OBJECT, fields, required }), sites];
}

/**
 * Read one type node and collect each physically authored record-field site.
 *
 * @param node Type text, a record mapping, or a list of single-key maps.
 * @param registry The typedefs type text resolves against.
 * @param options.cursor The node's source, or `null` for a source-free read.
 * @param options.path The node's field path from the declaration's root.
 * @param options.diagnostics Builds the load errors.
 * @param options.levels The records enclosing `node` (default 0).
 * @returns The type and the authored record-field sites inside it.
 * @throws LoadError when the node is not a type, or a record or a field's type nests past
 *   `current().maxTypeNesting` levels with the records around it.
 */
function readTypeAndSites(
  node: unknown,
  registry: TypeRegistry,
  options: {
    cursor: SourceCursor | null;
    path: ValuePath;
    diagnostics: LoadDiagnostics;
    levels?: number;
  },
): [Type, Site[]] {
  const { cursor, path, diagnostics, levels = 0 } = options;
  const limit = current().maxTypeNesting;
  const range = cursor?.range ?? null;
  if (typeof node === "string") {
    let typ: Type;
    try {
      typ = typeFor(node, registry);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw diagnostics.error(`bad type expression ${reprText(node)}: ${error.message}`, {
        range,
        cause: error,
      });
    }
    if (levels) {
      const added = typeLevels(typ);
      if (levels + added > limit) {
        throw diagnostics.error(`${tooDeep(path, limit)}; its type ${reprText(node)} adds ${added}`, {
          range,
        });
      }
    }
    return [typ, []];
  }

  if ((isRecord(node) || Array.isArray(node)) && levels + 1 > limit) {
    // Checked before reading the entries: an alias chain can nest a record far deeper than
    // this walk could recurse.
    throw diagnostics.error(tooDeep(path, limit), { range });
  }

  if (isRecord(node)) {
    return readRecordType(
      Object.entries(node).map(([key, value]) => [key, value, cursor?.child(key) ?? null] as const),
      registry,
      { path, diagnostics, levels: levels + 1 },
    );
  }

  if (Array.isArray(node)) {
    // The list syntax is another spelling of one record: the first occurrence of a key
    // keeps its place and the last one its value, along with that value's source.
    const merged = new Map<string, readonly [unknown, SourceCursor | null]>();
    for (const [index, element] of (node as unknown[]).entries()) {
      const itemCursor = cursor?.child(index) ?? null;
      if (!(isRecord(element) && Object.keys(element).length === 1)) {
        throw diagnostics.error(`a list of fields must hold single-key maps, got ${valueRepr(element)}`, {
          range: itemCursor?.range ?? null,
        });
      }
      const [key, value] = Object.entries(element)[0]!;
      merged.set(key, [value, itemCursor?.child(key) ?? null]);
    }
    return readRecordType(
      [...merged].map(([key, [value, childCursor]]) => [key, value, childCursor] as const),
      registry,
      { path, diagnostics, levels: levels + 1 },
    );
  }

  throw diagnostics.error(`cannot read type from ${valueRepr(node)} (type ${pythonTypeName(node)})`, {
    range,
  });
}

/**
 * Read a Compose source node into one runtime `Type` (recursively).
 *
 * @throws LoadError, unranged, when the node is not a type or nests too deep.
 */
export function readType(
  node: unknown,
  registry: TypeRegistry,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Type {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const [typ] = readTypeAndSites(node, registry, { cursor: null, path: [], diagnostics });
  return typ;
}

/**
 * Compile one authored type declaration and its nested source sites.
 *
 * `sitePath` is the semantic prefix for this contract. The reader appends `root` for the
 * whole declaration and `path/<segments...>` for each physically authored record field. A
 * scalar list/alias has no interior YAML declaration, so element failures correctly fall back
 * to the root.
 *
 * @throws LoadError, located, when the declaration is not a type or nests too deep, or its
 *   source can't be read or registered.
 */
export function readValueContract(
  cursor: SourceCursor,
  registry: TypeRegistry,
  options: {
    sitePath: SourceSitePath;
    catalogBuilder: SourceCatalogBuilder;
    diagnostics?: LoadDiagnostics | null;
  },
): ValueContract {
  const { sitePath, catalogBuilder } = options;
  const diagnostics = options.diagnostics ?? LoadDiagnostics.fromCursor(cursor);
  try {
    const [typ, sites] = readTypeAndSites(cursor.value, registry, { cursor, path: [], diagnostics });
    const fallback = catalogBuilder.register([...sitePath, "root"], cursor);
    const entries: [ValuePath, OriginRef][] = [];
    for (const [path, fieldCursor] of sites.sort(([a], [b]) => comparePaths(a, b))) {
      const origin = catalogBuilder.register([...sitePath, "path", ...path], fieldCursor);
      if (origin !== null) entries.push([path, origin]);
    }
    return new ValueContract(typ, new ContractOrigins({ default: fallback, entries }));
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
}
