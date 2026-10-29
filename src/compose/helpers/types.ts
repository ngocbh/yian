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

import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import {
  comparePaths,
  ContractOrigins,
  Parameter,
  ValueContract,
  type ValuePath,
} from "../../nodes/contracts.ts";
import type { SourceCatalogBuilder, SourceSitePath } from "../source/catalog.ts";
import {
  parseYamlLiteral,
  SourceAccessError,
  type SourceCursor,
  SourceSyntaxError,
} from "../source/index.ts";
import {
  bareConstructorHint,
  BUILTIN_TYPE_NAMES,
  encodesAsUtf8,
  typeName,
  type TypeRegistry,
  typeFor,
} from "../../typesys/types.ts";
import {
  boundText,
  isInboxType,
  isRecord,
  renderValuePath,
  Type,
  TypeCheckError,
  typeLevels,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError } from "./errors.ts";
import {
  closeTypeName,
  oldFormSuggestion,
  readTaggedValue,
  resolveTypeText,
  spellTypeText,
  splitTypedText,
  suggestTag,
  yamlQuoted,
} from "./tags.ts";
import { checkWrittenValue } from "./value_nesting.ts";

const HEAD_NAME = /^[A-Za-z_][A-Za-z0-9_]*/;
// Text this plain is safe to show unquoted after a tag, once it reads back as text.
const PLAIN_TEXT = /^[A-Za-z][A-Za-z0-9 _.,'-]*$/;

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
    // Checked without a source too, so a source-free read
    // gets the same load error rather than the raw error `Type` gives such a name.
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
    if (isInboxType(typ)) {
      throw diagnostics.error(`an output can't be an inbox (\`Inbox(...)\` declares a flow input)`, {
        range,
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

  throw diagnostics.error(`cannot read type from ${valueRepr(node)} (type ${typeName(node)})`, {
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

// ---------- flow `input:` declarations ----------

const SOURCE_FREE_DEFAULT = "source-free declarations take a type only; write defaults in YAML with a tag";

/**
 * Read one source-free input declaration, which is type text with no default.
 *
 * @throws LoadError, unranged, when the value is not text, or holds a default (`T = default`)
 *   or is not a type.
 */
function readTypeOnly(
  name: string,
  value: unknown,
  registry: TypeRegistry,
  options: { diagnostics: LoadDiagnostics },
): Parameter {
  const { diagnostics } = options;
  if (typeof value !== "string") {
    throw diagnostics.error(
      `input ${reprText(name)}: a source-free declaration is type text, got ${typeName(value)}`,
    );
  }
  let typ: Type;
  try {
    typ = typeFor(value, registry);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    if (value.includes("=")) throw diagnostics.error(`input ${reprText(name)}: ${SOURCE_FREE_DEFAULT}`);
    throw diagnostics.error(
      `input ${reprText(name)}: bad type expression ${reprText(value)}: ${error.message}`,
      { cause: error },
    );
  }
  return new Parameter(name, {
    contract: new ValueContract(typ),
    required: !typ.nullable,
    coerceStrings: true,
  });
}

/**
 * Read source-free `input:` declarations, each a type text such as `Optional(int)`.
 *
 * A source-free declaration has no default: defaults are written in YAML after a tag. This
 * entry point deliberately emits no origins.
 *
 * @param mapping The `input:` section, or `null`/`undefined` for none.
 * @param ranges Each declaration's range, given to an error the declaration raises without
 *   one.
 * @throws LoadError when a declaration is not type text or holds a default.
 */
export function readFlowInputs(
  mapping: Readonly<Record<string, unknown>> | null | undefined,
  registry: TypeRegistry,
  ranges: ReadonlyMap<string, SourceRange | null> | null = null,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Parameter[] {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const parameters: Parameter[] = [];
  for (const [name, value] of Object.entries(mapping ?? {})) {
    const sourceRange = ranges?.get(name) ?? null;
    try {
      parameters.push(readTypeOnly(name, value, registry, { diagnostics }));
    } catch (error) {
      if (!(error instanceof LoadError) || error.diagnostic.frames.length > 0 || sourceRange === null) {
        throw error;
      }
      throw diagnostics.prepend(error, { range: sourceRange, cause: error });
    }
  }
  return parameters;
}

/**
 * Read source-free local declarations: only an empty mapping succeeds.
 *
 * A variable needs a default, and a source-free declaration can't carry one; defaults are
 * written in YAML after a tag (`n: !int 4`) and read by `readSourceParameters`.
 *
 * @param mapping The `variables:` section.
 * @param registry The flow's typedefs; unused, since no declaration is read.
 * @param options.diagnostics Factory for load errors.
 * @returns Always empty.
 * @throws LoadError, unranged, when `mapping` is not a mapping, a name is empty, or it declares
 *   any variable.
 */
export function readVariables(
  mapping: unknown,
  registry: TypeRegistry,
  options: { diagnostics?: LoadDiagnostics | null } = {},
): Parameter[] {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  if (!isRecord(mapping)) throw diagnostics.error("variables must be a mapping");
  for (const name of Object.keys(mapping)) {
    // A record's keys are always text, so only the empty name is checked.
    if (name === "") throw diagnostics.error("variable names must be non-empty strings");
    throw diagnostics.error(`variable ${reprText(name)}: a variable needs a default; ${SOURCE_FREE_DEFAULT}`);
  }
  return [];
}

/**
 * Read one tagged declaration: the tag is its type, a value after it its default.
 *
 * A declaration with no value after the tag (`qty: !int`) has no default and is required
 * unless its type is nullable. No default may nest deeper than a yian value may.
 *
 * @throws LoadError when the default nests too deep or holds a lone surrogate, the tag or the
 *   default can't be read, or a required default is missing.
 */
function readTaggedParameter(
  name: string,
  cursor: SourceCursor,
  registry: TypeRegistry,
  options: {
    parameterPath: SourceSitePath;
    parameterOrigin: OriginRef | null;
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
    declarationLabel: string;
    requireDefault: boolean;
  },
): Parameter {
  const { parameterPath, parameterOrigin, catalogBuilder, diagnostics } = options;
  const { declarationLabel, requireDefault } = options;
  if (cursor.hasValue) {
    // First: the tag read below recurses through the value.
    const where = `${declarationLabel} ${reprText(name)} default`;
    checkWrittenValue(cursor.value, where, cursor.range, diagnostics);
  }
  const slot = readTaggedValue(cursor, registry, {
    label: name,
    // The same contract prefix an untagged declaration's type registers under.
    sitePath: [...parameterPath, "contract"],
    catalogBuilder,
    diagnostics,
    ledger: catalogBuilder.consumedTags,
    literalOnly: true,
    inbox: declarationLabel === "input",
  });
  const contract = slot.contract!;
  if (requireDefault && !slot.hasValue) {
    const spelled = suggestTag(name, contract.type, "<default>", { typeText: cursor.tag });
    throw diagnostics.error(
      `${declarationLabel} ${reprText(name)}: a ${declarationLabel} needs a default: ${spelled}`,
      { range: cursor.range },
    );
  }
  return new Parameter(name, {
    contract,
    required: !contract.type.nullable && !slot.hasValue,
    hasDefault: slot.hasValue,
    default: slot.value,
    coerceStrings: true,
    origin: parameterOrigin,
    defaultOrigin: slot.hasValue ? catalogBuilder.register([...parameterPath, "default"], cursor) : null,
  });
}

/**
 * Explain how to write an untagged input or variable declaration with a tag.
 *
 * @param name The declared name.
 * @param value The untagged value: old `T` or `T = default` text, a record map, or anything
 *   else.
 * @param registry The flow's typedefs the old type text resolves against.
 * @param options.declarationLabel `"input"` or `"variable"`.
 * @param options.requireDefault Whether the declaration needs a default.
 * @returns The error message.
 */
function untaggedDeclarationError(
  name: string,
  value: unknown,
  registry: TypeRegistry,
  options: { declarationLabel: string; requireDefault: boolean },
): string {
  const { declarationLabel, requireDefault } = options;
  const noun = declarationLabel === "input" ? "flow input" : declarationLabel;
  const head = `${noun} ${reprText(name)}`;
  if (isRecord(value)) {
    const record = name
      .split("_")
      .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
      .join("");
    return (
      `${head}: name the record in \`typedefs:\` and tag the ${declarationLabel} ` +
      `with it, e.g. \`${name}: !${record || "Record"}\``
    );
  }
  if (typeof value === "string") {
    // The whole text first: a type such as `Literal("a = b")` may hold ` = `.
    const split = splitTypedText(value);
    const candidates: [string, string][] = [[value.trim(), ""], ...(split !== null ? [split] : [])];
    for (const [text, fallback] of candidates) {
      const typeText = typeTextWithKnownHead(text, registry);
      if (typeText === null) continue; // value text, not a type
      // A bare `list` has no tag to suggest; say what it takes instead.
      const bare = bareConstructorHint(typeText);
      if (bare !== null) return `${head}: ${bare}`;
      const resolved = resolveTypeText(typeText, registry);
      // Text whose leading name is a type but whose arguments name none still gets its tag,
      // so pasting it back reaches the tag reader's unknown-type error.
      const [typ, spelled] = resolved ?? [null, spellTypeText(typeText)];
      if (spelled !== null) {
        const suggestion = oldFormSuggestion(name, typ, spelled, fallback, {
          registry,
          placeholder: requireDefault ? "<default>" : null,
        });
        return `${head}: write the type as a tag: ${suggestion}`;
      }
    }
  }
  const [tag, example] = exampleDefault(value);
  if (requireDefault) {
    return `${head}: a ${declarationLabel} is a tag with a default, e.g. \`${name}: !${tag} ${example}\``;
  }
  return (
    `${head}: an ${declarationLabel} is a tag with an optional default, ` +
    `e.g. \`${name}: !${tag}\` or \`${name}: !${tag} ${example}\``
  );
}

/**
 * Return type text whose leading name is a type name, else `null`.
 *
 * A leading name close to a builtin or typedef name is replaced with that name (`strr` gives
 * `str`); text whose leading name is neither is a value.
 *
 * @param text Authored text that may be a type.
 * @param registry The flow's typedefs, offered with the builtins.
 * @returns The text with a type name in front, or `null`.
 */
function typeTextWithKnownHead(text: string, registry: TypeRegistry): string | null {
  const trimmed = text.trim();
  const head = HEAD_NAME.exec(trimmed)?.[0];
  if (head === undefined) return null;
  if (BUILTIN_TYPE_NAMES.has(head) || registry.has(head)) return trimmed;
  const close = closeTypeName(trimmed, registry);
  return close === null ? null : close + trimmed.slice(head.length);
}

/**
 * Spell a tag and a default for an untagged value, as in `!str hello`.
 *
 * A finite float is spelled in its shortest form with a fraction digit or exponent (`2.0`,
 * `1e-07`) and infinity as `.inf` or `-.inf`, which YAML reads back as a float.
 *
 * @param value The untagged value.
 * @returns The tag's type and the default written after it: the value itself when it is text
 *   or a scalar, else `int` and `30`.
 */
function exampleDefault(value: unknown): [tag: string, example: string] {
  if (typeof value === "boolean") return ["bool", value ? "true" : "false"];
  if (typeof value === "bigint") return ["int", value.toString()];
  if (typeof value === "number") {
    return ["float", Number.isFinite(value) ? boundText(value) : value > 0 ? ".inf" : "-.inf"];
  }
  if (typeof value === "string" && value.trim()) {
    return ["str", readsAsText(value) ? value : yamlQuoted(value)];
  }
  return ["int", "30"];
}

/** Whether `text` written unquoted after a tag reads back as exactly that text. */
function readsAsText(text: string): boolean {
  if (!PLAIN_TEXT.test(text)) return false;
  try {
    return parseYamlLiteral(text) === text;
  } catch (error) {
    if (error instanceof SourceSyntaxError) return false;
    throw error;
  }
}

/**
 * Compile authored input or variable declarations into located parameters.
 *
 * A declaration takes its type from its tag and its default from the value after it
 * (`n: !int(1..12) 4`); with no value (`qty: !int`) it has no default. An untagged declaration
 * is an error that shows the tagged spelling. `sitePath` identifies the containing owner;
 * `fieldName` its section (default `input`). `declarationLabel` (default `input`) names
 * individual errors, and `requireDefault` (default `false`) selects declarations whose values
 * must include a typed literal default. Each parameter receives stable child identities for
 * its key declaration, contract tree, and packed default. The resulting carriers retain only
 * `OriginRef` values.
 *
 * @throws LoadError, located, when the section is not a mapping, a name is empty, a
 *   declaration is untagged or can't be read, or the source can't be read or registered.
 */
export function readSourceParameters(
  cursor: SourceCursor,
  registry: TypeRegistry,
  options: {
    sitePath: SourceSitePath;
    catalogBuilder: SourceCatalogBuilder;
    diagnostics?: LoadDiagnostics | null;
    fieldName?: string;
    declarationLabel?: string;
    requireDefault?: boolean;
  },
): Parameter[] {
  const { sitePath, catalogBuilder, fieldName = "input" } = options;
  const { declarationLabel = "input", requireDefault = false } = options;
  const diagnostics = options.diagnostics ?? LoadDiagnostics.fromCursor(cursor);
  const mapping = cursor.value;
  const sectionLabel = fieldName === "input" ? "flow input" : fieldName;
  if (!isRecord(mapping)) {
    throw diagnostics.error(`${sectionLabel} must be a mapping, got ${typeName(mapping)}`, {
      range: cursor.range,
    });
  }

  const parameters: Parameter[] = [];
  try {
    for (const [name, value] of Object.entries(mapping)) {
      const declarationCursor = cursor.child(name);
      // Keys are text here; the loader refuses a non-text key (`document.nonTextKey`).
      if (name === "") {
        throw diagnostics.error(`${sectionLabel} names must be non-empty strings`, {
          range: declarationCursor.keyRange ?? declarationCursor.range,
        });
      }
      const parameterPath = [...sitePath, fieldName, name];
      const parameterOrigin = catalogBuilder.register(
        [...parameterPath, "parameter"],
        declarationCursor,
        "key",
      );

      try {
        if (declarationCursor.tag === null) {
          throw diagnostics.error(
            untaggedDeclarationError(name, value, registry, { declarationLabel, requireDefault }),
            { range: declarationCursor.range },
          );
        }
        parameters.push(
          readTaggedParameter(name, declarationCursor, registry, {
            parameterPath,
            parameterOrigin,
            catalogBuilder,
            diagnostics,
            declarationLabel,
            requireDefault,
          }),
        );
      } catch (error) {
        if (
          !(error instanceof LoadError) ||
          error.diagnostic.frames.length > 0 ||
          declarationCursor.range === null
        ) {
          throw error;
        }
        throw diagnostics.prepend(error, { range: declarationCursor.range, cause: error });
      }
    }
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
  return parameters;
}
