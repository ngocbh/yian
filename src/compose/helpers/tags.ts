/**
 * Read a slot's type and value through its YAML tags.
 *
 * A slot's type is a local YAML tag on its value (`qty: !int ${size.output.qty}`). A tag ends
 * at the first space and, in strict YAML 1.2, can't contain `,`, `[`, `]`, `{` or `}`, so a
 * type with several arguments is named in `typedefs:` and tagged by its name. `readTagType`
 * turns tag text into a `Type` and explains each way a tag goes wrong; `suggestTag` spells a
 * tagged slot for error messages; `readTaggedValue` reads one slot's authored value, with tags
 * inside it, into a binding source or a literal, and records each tag it reads in the load's
 * ledger. `rejectUnreadTags` then rejects every tag no slot read (on `kind:`, a typedef, a
 * mapping key, ...), and `rejectTaggedOutput` a tag on a node's `output:` type declaration.
 *
 * A tag only checks a value and declares its type: it never adds or changes a field.
 */

import {
  compareCodeUnits,
  getCloseMatches,
  isPrintable,
  type OriginRef,
  reprText,
  type SourceRange,
} from "../../diagnostics.ts";
import { ExpressionError, MISSING_ROOT } from "../../expr/expressions.ts";
import { evalBinding, scanTemplate, Span, Verbatim } from "../../expr/template.ts";
import {
  Binding,
  ContractOrigins,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import {
  BUILTIN_TYPE_NAMES,
  formatTypeExpr,
  legacyRespelling,
  parseType,
  type TypeRegistry,
  typeFor,
} from "../../typesys/types.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isInboxType,
  isMemoryType,
  isList,
  isListKind,
  isRecord,
  requiresFile,
  Type,
  TypeCheckError,
  typeEquals,
  ValueKind,
  type Value,
  valueRepr,
} from "../../typesys/values.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { ScalarStyle } from "../source/common.ts";
import {
  includeTag,
  parseYamlLiteral,
  SourceAccessError,
  type SourceCursor,
  type SourceDocument,
  SourceSyntaxError,
  type TypedInclude,
} from "../source/index.ts";
import { CALL_HEAD } from "./directives.ts";
import type { LoadDiagnostics } from "./errors.ts";
import { inputTagRegistry, type MemoryTypeTable, NO_MEMORY_TYPES } from "./memory.ts";
import { isExtensionKey } from "./source.ts";
import { formerBooleanFix, valueFits } from "./yaml_words.ts";

const HEAD_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*/;
/** The `getCloseMatches` ratio a name must reach to be offered as a close match. */
export const CLOSE_MATCH_CUTOFF = 0.6;
const FLOW_INDICATORS = /[[\]{}]/;
// The characters YAML reads into a tag, minus YAML 1.2's flow indicators, `!`, and the `%`
// that starts an escape: text made only of these reads back as one tag.
const TAG_CHARACTERS = /^[A-Za-z0-9\-;/?:@&=+$_.~*'()]+$/;
const TYPEDEF_EXAMPLE = "(e.g. `Decision: Union(Buy, Sell)`) and tag with its name (`!Decision`)";
/**
 * YAML scalar styles that make a value text, never a `call(...)` directive. Typed to take a
 * cursor's `scalarStyle` as is: `null` (not a scalar) is not in it.
 */
export const QUOTED_STYLES: ReadonlySet<ScalarStyle | null> = new Set<ScalarStyle | null>([
  "single",
  "double",
  "literal",
  "folded",
]);
const OLD_SEPARATOR = " = ";
// YAML drops the trailing space of `qty: int = `, delivering "int =".
const OLD_TRAILING_SEPARATOR = " =";

/** Return how messages name the file an `!include` (or `!include(T)`, for `type` `T`) reads. */
export function includeFile(path: string, type: string | null = null): string {
  return `${includeTag(type)} file ${reprText(path)}`;
}

/**
 * Resolve a slot's tag text into its type.
 *
 * @param tag The tag text without the leading `!`, e.g. `"list(Buy)"`.
 * @param registry The flow's typedefs the tag resolves against.
 * @param options.label The slot's name, used in error messages.
 * @param options.diagnostics Factory for located load errors.
 * @param options.range The tagged value's location, attached to any error.
 * @param options.include The path an `!include(T)` names, whose `T` is `tag`: messages name the
 *   file and the type `T`, not a tag `!T`. `null` (the default) for a tag.
 * @returns The type the tag names.
 * @throws LoadError when the tag holds a bracket or brace (the message shows the
 *   parenthesized spelling), is cut short by a space (it ends in `,` or has an unclosed `(`),
 *   holds a `,`, or does not name a type (the message offers a close match).
 */
export function readTagType(
  tag: string,
  registry: TypeRegistry,
  options: { label: string; diagnostics: LoadDiagnostics; range?: SourceRange | null; include?: string | null },
): Type {
  const { label, diagnostics, range = null, include = null } = options;
  // How the messages below name the tag.
  const shown = include === null ? `\`!${tag}\`` : `${includeFile(include, tag)}: \`${tag}\``;
  if (FLOW_INDICATORS.test(tag)) {
    let message =
      `${reprText(label)}: a tag can't contain \`[\`, \`]\`, \`{\` or \`}\` (strict YAML 1.2)`;
    const spelled = respelled(tag, registry);
    if (spelled !== null) {
      message += writableAsTag(spelled)
        ? `; write \`!${spelled}\``
        : `; name \`${spelled}\` in \`typedefs:\` and tag with its name`;
    }
    throw diagnostics.error(message, { range });
  }
  let resolved: Type | TypeCheckError;
  try {
    resolved = typeFor(tag, registry);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    resolved = error;
  }
  // Parens are counted only for a tag that names no type, so one inside a quoted `Literal`
  // member (`!Literal('(')`) doesn't read as cut short.
  const unclosed = tag.split("(").length > tag.split(")").length;
  if (tag.endsWith(",") || (resolved instanceof TypeCheckError && unclosed)) {
    throw diagnostics.error(
      `${reprText(label)}: ${shown} is not a complete type: a YAML tag ends at the first ` +
        "space and can't contain `,`; name a type with several arguments in " +
        `\`typedefs:\` ${TYPEDEF_EXAMPLE}`,
      { range },
    );
  }
  if (tag.includes(",")) {
    throw diagnostics.error(
      `${reprText(label)}: ${shown}: a tag can't contain \`,\`; name it in \`typedefs:\` ` +
        TYPEDEF_EXAMPLE,
      { range },
    );
  }
  if (resolved instanceof TypeCheckError) {
    let message = `${reprText(label)}: ${shown} is not a type (${resolved.message})`;
    const match = closeTypeName(tag, registry);
    if (match !== null) message += `. Did you mean \`${match}\`?`;
    throw diagnostics.error(message, { range });
  }
  return resolved;
}

/**
 * Spell bracket-style type text in the parenthesized spelling, if it resolves.
 *
 * A `dict[K, V]` becomes `object`, the lenient mapping it resolves to.
 */
function respelled(text: string, registry: TypeRegistry): string | null {
  try {
    const spelled = legacyRespelling(text);
    typeFor(spelled, registry);
    return spelled;
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/**
 * Split text at its first ` = ` into the part before and the part after.
 *
 * A `call(...)` directive is not split, since its arguments may hold ` = `.
 *
 * @param value The authored value.
 * @returns The trimmed parts before and after the separator (the second empty for `int =`),
 *   or `null` when `value` is not text, is a `call(...)`, or holds no separator.
 */
export function splitTypedText(value: unknown): [before: string, after: string] | null {
  if (typeof value !== "string" || CALL_HEAD.test(value.trim())) return null;
  let left: string;
  let right: string;
  const at = value.indexOf(OLD_SEPARATOR);
  if (at >= 0) {
    left = value.slice(0, at);
    right = value.slice(at + OLD_SEPARATOR.length);
  } else if (value.endsWith(OLD_TRAILING_SEPARATOR)) {
    left = value.slice(0, -OLD_TRAILING_SEPARATOR.length);
    right = "";
  } else {
    return null;
  }
  return [left.trim(), right.trim()];
}

/**
 * Recognize the old `Type = value` text, whose type is now written as a tag.
 *
 * Only text whose part before the first ` = ` names a type (in either spelling) is the old
 * form; any other text holding ` = ` is plain text, and so is a `call(...)` directive, whose
 * arguments may hold ` = `. The caller decides whether the value's YAML style makes it text.
 *
 * @param value The authored value.
 * @param registry The flow's typedefs the left side resolves against.
 * @returns The type, its parenthesized spelling, and the text after the separator (empty for
 *   `int =`), or `null` when `value` is not the old form.
 */
export function oldTypedText(
  value: unknown,
  registry: TypeRegistry,
): [Type, spelled: string, after: string] | null {
  const split = splitTypedText(value);
  if (split === null) return null;
  const resolved = resolveTypeText(split[0], registry);
  if (resolved === null) return null;
  return [resolved[0], resolved[1], split[1]];
}

/**
 * Resolve type text written in either the parenthesized or the bracket spelling.
 *
 * @param text Authored type text, e.g. `"int"` or `"Optional[int]"`.
 * @param registry The flow's typedefs the text resolves against.
 * @returns The type and its parenthesized spelling, or `null` when the text names no type in
 *   either spelling.
 */
export function resolveTypeText(
  text: string,
  registry: TypeRegistry,
): [Type, spelled: string] | null {
  const trimmed = text.trim();
  const spelled = resolves(trimmed, registry) ? trimmed : respelled(trimmed, registry);
  if (spelled === null) return null;
  return [typeFor(spelled, registry), spelled];
}

/**
 * Spell text that parses as a type, in either spelling, with parentheses.
 *
 * Names are not resolved, so `list[message]` gives `list(message)` even with no `message`
 * typedef.
 *
 * @param text Authored type text.
 * @returns The parenthesized spelling, or `null` when the text does not parse as a type
 *   expression.
 */
export function spellTypeText(text: string): string | null {
  const trimmed = text.trim();
  try {
    return formatTypeExpr(parseType(trimmed));
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
  }
  try {
    return legacyRespelling(trimmed);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/**
 * Spell a literal for a tag suggestion so that it fits `typ`, where that is cheap.
 *
 * A former YAML boolean word becomes the boolean it meant when that fits, with the
 * `formerBooleanFix` note; a non-string literal whose text fits becomes quoted text (`30` for
 * `str` gives `"30"`). Templates, directives, text that doesn't parse as YAML, and anything
 * else are kept as written.
 *
 * @param text The literal text after the type.
 * @param typ The type the tag declares.
 * @returns The text to show after the tag, and a note to show after the suggestion, if any.
 */
export function suggestedLiteral(text: string, typ: Type): [shown: string, note: string | null] {
  if (!text || hasSpan(text) || CALL_HEAD.test(text)) return [text, null];
  let parsed: Value;
  try {
    parsed = parseYamlLiteral(text);
  } catch (error) {
    if (error instanceof SourceSyntaxError) return [text, null];
    throw error;
  }
  if (valueFits(typ, parsed)) return [text, null];
  const fix = formerBooleanFix(parsed, typ);
  if (fix !== null) return [fix[0] ? "true" : "false", fix[1]];
  if (typeof parsed !== "string" && valueFits(typ, text)) return [yamlQuoted(text), null];
  return [text, null];
}

/**
 * Spell the tagged slot an old `name: T = value` form should be written as.
 *
 * @param name The slot's name.
 * @param typ The type, or `null` when its text names no type.
 * @param typeText The type in the parenthesized spelling.
 * @param value The authored text after ` = `, or `""` for none.
 * @param options.registry The flow's typedefs.
 * @param options.placeholder The text shown when there is no value, or `null` to show the tag
 *   alone.
 * @returns The suggestion from `suggestTag`, with a literal spelled to fit the type where that
 *   is cheap, and the reason when the literal changed.
 */
export function oldFormSuggestion(
  name: string,
  typ: Type | null,
  typeText: string,
  value: string,
  options: { registry: TypeRegistry; placeholder: string | null },
): string {
  let shown = value;
  let note: string | null = null;
  if (typ !== null && value) [shown, note] = suggestedLiteral(value, typ);
  const suggestion = suggestTag(name, typ, shown || options.placeholder, {
    registry: options.registry,
    typeText,
  });
  return note === null ? suggestion : `${suggestion} (${note})`;
}

/** Return whether `text` names a type. */
function resolves(text: string, registry: TypeRegistry): boolean {
  try {
    typeFor(text, registry);
  } catch (error) {
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
  return true;
}

/**
 * Return the type name closest to the name `text` starts with, if any.
 *
 * @param text Authored type text that failed to resolve.
 * @param registry The flow's typedefs, offered with the builtins.
 * @returns A builtin or typedef name close to the leading name, or `null` when there is no
 *   leading name, it is already a type name (the error lies later in the text, e.g.
 *   `list(Strr)`), or nothing is close.
 */
export function closeTypeName(text: string, registry: TypeRegistry): string | null {
  const head = HEAD_IDENTIFIER.exec(text)?.[0];
  const names = new Set([...BUILTIN_TYPE_NAMES, ...registry.keys()]);
  if (head === undefined || names.has(head)) return null;
  const sorted = [...names].sort(compareCodeUnits);
  const matches = getCloseMatches(head, sorted, 1, CLOSE_MATCH_CUTOFF);
  if (matches.length > 0) return matches[0]!;
  // A wrong capital (`Strr`, `INT`) costs a whole character in the ratio, so a case-blind
  // second pass finds what the first misses.
  const byLower = new Map<string, string>();
  for (const name of sorted) {
    const lower = name.toLowerCase();
    if (!byLower.has(lower)) byLower.set(lower, name);
  }
  const lowered = getCloseMatches(
    head.toLowerCase(),
    [...byLower.keys()].sort(compareCodeUnits),
    1,
    CLOSE_MATCH_CUTOFF,
  );
  return lowered.length > 0 ? byLower.get(lowered[0]!)! : null;
}

/**
 * Spell a tagged slot for an error message.
 *
 * @param name The slot's name.
 * @param typ The slot's type, or `null` when its text names no type; `typeText` is then shown
 *   as written.
 * @param value The value text after the tag, or `null` for none.
 * @param options.registry The flow's typedefs; a typedef that resolves to `typ` (or to its
 *   list element or non-null form) is spelled by its name.
 * @param options.typeText The authored spelling of `typ`, used when a tag can write it and in
 *   the advice to name it otherwise.
 * @returns `` `name: !<type> value` `` in backticks, or, for a type a tag can't spell (an
 *   unnamed record or union, or anything needing a `,` or quotes), advice to name the type in
 *   `typedefs:` and tag with its name.
 * @throws TypeError when both `typ` and `options.typeText` are `null`.
 */
export function suggestTag(
  name: string,
  typ: Type | null,
  value: string | null,
  options: { registry?: TypeRegistry | null; typeText?: string | null } = {},
): string {
  let { typeText = null } = options;
  const text = tagText(typ, { registry: options.registry, typeText });
  if (text === null) {
    if (typeText === null) {
      if (typ === null) throw new TypeError("suggestTag needs a type or its text");
      typeText = canonicalTypeLabel(typ);
    }
    return `name \`${typeText}\` in \`typedefs:\` and tag with its name`;
  }
  if (value === null) return `\`${name}: !${text}\``;
  return `\`${name}: !${text} ${value}\``;
}

/**
 * Advise how to fill a slot of type `sink` from an `Any` source, when that can work.
 *
 * Load can't tell which JSON value an `Any` holds, so a tag checks it at runtime or a `case`
 * narrows it. There is no advice when the source isn't `Any`, the slot is `Any` or
 * `Optional(Any)` (only nullability failed), or every non-null value of the slot holds a file:
 * load treats such a slot as disjoint from `Any`, although a `file` slot reads a file's JSON
 * form at runtime, where load can't tell.
 *
 * @param name The slot's name.
 * @param source The source's type.
 * @param sink The slot's type.
 * @param options.registry The flow's typedefs, used to spell the tag by a typedef name.
 * @returns Advice to tag the slot with `sink` or to narrow the value with `isinstance` in a
 *   `case`, or `null` when neither can help.
 */
export function anySourceHint(
  name: string,
  source: Type,
  sink: Type,
  options: { registry?: TypeRegistry | null } = {},
): string | null {
  if (source.kind !== ValueKind.ANY || sink.kind === ValueKind.ANY || requiresFile(sink)) {
    return null;
  }
  const tag = suggestTag(name, sink, "${...}", { registry: options.registry });
  return (
    `tag the slot (${tag}) to check it at runtime, or narrow it with \`isinstance\` ` +
    "in a `case`"
  );
}

/**
 * Spell a type as the text after `!`, or return `null` when no tag can spell it.
 *
 * @param typ The type, or `null` when its text names no type.
 * @param options.registry The flow's typedefs; a typedef that resolves to `typ` (or to its
 *   list element or non-null form) is spelled by its name.
 * @param options.typeText The authored spelling of `typ`, used as written when a tag can
 *   write it.
 * @returns Such as `"list(int)"` or `"Reply"`, or `null` for a type a tag can't spell (an
 *   unnamed record or union, or anything needing a `,` or quotes).
 */
export function tagText(
  typ: Type | null,
  options: { registry?: TypeRegistry | null; typeText?: string | null } = {},
): string | null {
  const { registry, typeText = null } = options;
  if (typeText !== null && writableAsTag(typeText)) return typeText;
  if (typ === null) return null;
  return tagTextOf(typ, registry ?? new Map());
}

/**
 * Spell `typ` as tag text, or return `null` when no tag can spell it.
 *
 * The type is spelled directly where a tag can; a typedef name stands in only where it can't
 * (a record, a union, a multi-value `Literal`, anything needing a comma or quotes).
 */
function tagTextOf(typ: Type, registry: TypeRegistry): string | null {
  const direct = directTagText(typ, registry);
  if (direct !== null) return direct;
  const named = typedefName(typ.replace({ nullable: false }), registry);
  if (named === null) return null;
  const text = typ.nullable ? `Optional(${named})` : named;
  return writableAsTag(text) ? text : null;
}

/**
 * Spell `typ` by its structure; only a list element may fall back to a typedef name. An inbox or
 * a memory has no spelling here.
 */
function directTagText(typ: Type, registry: TypeRegistry): string | null {
  // An inbox or a memory is only ever a flow input's type, so no other slot is told to tag one.
  if (typ.members !== undefined || typ.fields !== undefined || isInboxType(typ) || isMemoryType(typ)) return null;
  let text: string;
  if (typ.tags !== undefined) {
    if (typ.tags.size !== 1) return null;
    text = formatTypeExpr({ kind: "enum", tags: [...typ.tags] });
  } else if (isListKind(typ.kind) && typ.element !== undefined) {
    const element = tagTextOf(typ.element, registry);
    if (element === null) return null;
    text = `list(${element})`;
  } else {
    // A flat list, bounded number, or plain scalar has no part a tag can't spell except its
    // own characters, checked below.
    text = canonicalTypeLabel(new Type({ kind: typ.kind, bounds: typ.bounds }));
  }
  if (typ.nullable) text = `Optional(${text})`;
  return writableAsTag(text) ? text : null;
}

/**
 * Label a type for a message, naming a record or union by its typedef.
 *
 * @param typ The type to label.
 * @param registry The flow's typedefs.
 * @returns The typedef's name for a record or union one resolves to exactly (also under
 *   `Optional` or as a list element, as in `list(Buy)`), else `canonicalTypeLabel`.
 */
export function namedTypeLabel(typ: Type, registry: TypeRegistry): string {
  return namedRecord(typ, registry) ?? canonicalTypeLabel(typ);
}

/** Spell a record or union type (or a list of one) by its typedef, else `null`. */
function namedRecord(typ: Type, registry: TypeRegistry): string | null {
  let text: string | null;
  if (isListKind(typ.kind) && typ.element !== undefined) {
    const inner = namedRecord(typ.element, registry);
    text = inner === null ? null : `list(${inner})`;
  } else if (typ.fields !== undefined || typ.members !== undefined) {
    text = typedefName(typ.replace({ nullable: false }), registry);
  } else {
    text = null;
  }
  if (text === null) return null;
  return typ.nullable ? `Optional(${text})` : text;
}

/** Return the first typedef a tag can name that resolves to exactly `typ`. */
function typedefName(typ: Type, registry: TypeRegistry): string | null {
  for (const name of registry.keys()) {
    if (!writableAsTag(name)) continue;
    try {
      if (typeEquals(typeFor(name, registry), typ)) return name;
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
    }
  }
  return null;
}

/** Return whether `text` reads back as exactly one YAML tag when written after `!`. */
function writableAsTag(text: string): boolean {
  return TAG_CHARACTERS.test(text);
}

/** Return the index of the `)` that closes the `(` at `text[0]`, or -1 when it isn't closed. */
function closingParen(text: string): number {
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "(") depth++;
    else if (text[i] === ")" && --depth === 0) return i;
  }
  return -1;
}

/** One slot's authored value, read through its tags. Frozen. */
export class TaggedSlot {
  /**
   * The binding source or literal: a template string, a `ValueTemplate`, or a plain value;
   * `null` when `hasValue` is false.
   */
  readonly value: unknown;
  /** The top-level tag's contract, or `null` for an untagged value. */
  readonly contract: ValueContract | null;
  /** False for `name: !T` with nothing after the tag. */
  readonly hasValue: boolean;

  constructor(value: unknown, contract: ValueContract | null, hasValue: boolean) {
    this.value = value;
    this.contract = contract;
    this.hasValue = hasValue;
    Object.freeze(this);
  }
}

/** How `readTaggedValue` reads one slot. */
export interface TaggedValueOptions {
  /** The slot's name, used in messages and suggested spellings. */
  label: string;
  /** Catalog site prefix for this slot. */
  sitePath: readonly (string | number)[];
  /** File-owned origin catalog builder. */
  catalogBuilder: SourceCatalogBuilder;
  /** Factory for located load errors. */
  diagnostics: LoadDiagnostics;
  /**
   * Consumed tags; the reader adds the definition id (`SourceCursor.tagDefinitionId`, matching
   * `TaggedNode.definitionId`) of every tag it reads.
   */
  ledger: Set<number>;
  /**
   * Read a literal (a default): strings are text, never templates, and are not unescaped;
   * nested tags are checked and dropped, so the value is plain. Defaults to `false`.
   */
  literalOnly?: boolean;
  /**
   * The fix an unquoted nested string starting with `call(` is told to make; defaults to binding
   * it at the top level.
   */
  nestedCallFix?: string;
  /**
   * Whether the slot is a flow input, which may be an `Inbox(...)` or a memory; defaults to
   * `false`.
   */
  inbox?: boolean;
  /** How a flow input's tag reads memory types beyond the built-ins; defaults to none. */
  memoryTypes?: MemoryTypeTable;
}

/**
 * Read one slot's authored value and the tags on and inside it.
 *
 * The top-level tag becomes the slot's contract. A top-level string is kept as written (a
 * template, a `call(...)`, or text the binder unescapes). A tagged node inside a mapping or list
 * becomes a typed `Binding` at its position, and a container holding one, or holding a string
 * with a `${...}` span, becomes a `ValueTemplate` with raw leaf strings; any other container is
 * a plain literal with its leaf strings `$$`-unescaped once. A string an `!include(T)` read
 * (`SourceCursor.verbatim`) is never a template: it stays as written, as `Verbatim` text where a
 * string would be a template. Every span-free tagged literal is checked against its tag here; a
 * tag never fills in a missing field.
 *
 * Catalog sites: the top-level contract at `[...sitePath, "type", "root"]`, a nested binding at
 * `[...sitePath, "type", "path", ...path]`, and a template's leaf string at
 * `[...sitePath, "path", ...path]`.
 *
 * @param cursor The slot's value.
 * @param registry The flow's typedefs the tags resolve against.
 * @returns The value, the top-level contract, and whether a value follows the tag.
 * @throws LoadError when a tag is malformed or names no type, a value is missing after a nested
 *   tag, a space splits a tag from its arguments, an unquoted nested string starts with
 *   `call(`, a span-free tagged literal does not fit its tag, or the value can't be walked (a
 *   recursive alias).
 */
export function readTaggedValue(
  cursor: SourceCursor,
  registry: TypeRegistry,
  options: TaggedValueOptions,
): TaggedSlot {
  const reader = new TaggedValueReader(registry, options);
  try {
    return reader.read(cursor);
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw options.diagnostics.error(`${reprText(options.label)}: ${error.message}`, {
      range: error.range,
      cause: error,
    });
  }
}

/** One walked node: its tree, and what in it needs the binder. */
class Walked {
  tree: unknown;
  hasSpan: boolean;
  hasBinding: boolean;

  constructor(tree: unknown, options: { hasSpan?: boolean; hasBinding?: boolean } = {}) {
    this.tree = tree;
    this.hasSpan = options.hasSpan ?? false;
    this.hasBinding = options.hasBinding ?? false;
  }

  /** Whether the node must be a `ValueTemplate` part rather than a literal. */
  get dynamic(): boolean {
    return this.hasSpan || this.hasBinding;
  }

  /** Carry a child's span and binding flags up to this container. */
  absorb(part: Walked): void {
    this.hasSpan ||= part.hasSpan;
    this.hasBinding ||= part.hasBinding;
  }
}

/** Walk one slot's value with its cursors, reading the tags inside it. */
class TaggedValueReader {
  readonly registry: TypeRegistry;
  readonly label: string;
  readonly sitePath: readonly (string | number)[];
  readonly catalogBuilder: SourceCatalogBuilder;
  readonly diagnostics: LoadDiagnostics;
  readonly ledger: Set<number>;
  readonly literalOnly: boolean;
  readonly nestedCallFix: string;
  readonly inbox: boolean;
  readonly memoryTypes: MemoryTypeTable;
  readonly leafSites: [ValuePath, SourceCursor][] = [];

  constructor(registry: TypeRegistry, options: TaggedValueOptions) {
    this.registry = registry;
    this.label = options.label;
    this.sitePath = options.sitePath;
    this.catalogBuilder = options.catalogBuilder;
    this.diagnostics = options.diagnostics;
    this.ledger = options.ledger;
    this.literalOnly = options.literalOnly ?? false;
    this.nestedCallFix = options.nestedCallFix ?? "bind it at the top level, or reword the text";
    this.inbox = options.inbox ?? false;
    this.memoryTypes = options.memoryTypes ?? NO_MEMORY_TYPES;
  }

  /** Read the slot at `cursor`; see `readTaggedValue`. */
  read(cursor: SourceCursor): TaggedSlot {
    let contract: ValueContract | null = null;
    if (cursor.tag !== null) {
      // A space-split tag (`!Optional (str)`) usually names no type on its own, so the spacing
      // is reported before the type is resolved.
      this.rejectSpacedTag(cursor, []);
      const typ = this.tagType(cursor, []);
      if (isInboxType(typ) && cursor.hasValue) {
        throw this.diagnostics.error(`${reprText(this.label)}: an inbox input takes no default`, {
          range: cursor.range,
        });
      }
      if (isMemoryType(typ) && cursor.hasValue) {
        throw this.diagnostics.error(
          `${reprText(this.label)}: a memory input takes no default; the runner binds it`,
          { range: cursor.range },
        );
      }
      const origin = this.catalogBuilder.register([...this.sitePath, "type", "root"], cursor);
      contract = new ValueContract(typ, new ContractOrigins({ default: origin }));
      if (!cursor.hasValue) return new TaggedSlot(null, contract, false);
    }
    const walked = this.walk(cursor, cursor.value, []);
    const text = typeof walked.tree === "string" || walked.tree instanceof Verbatim;
    if (text || !walked.dynamic || this.literalOnly) {
      let literal = walked.tree;
      if (!this.isTemplateText(literal, cursor)) {
        literal = this.literal(literal);
        if (contract !== null) this.check(cursor, [], contract.type, literal);
      }
      // A top-level string stays as written: the binder unescapes it, and reads `Verbatim` text as is.
      return new TaggedSlot(text ? walked.tree : literal, contract, true);
    }

    if (contract !== null && !walked.hasSpan) {
      this.check(cursor, [], contract.type, this.literal(walked.tree));
    }
    const origins: [ValuePath, OriginRef][] = [];
    for (const [path, leaf] of this.leafSites) {
      const origin = this.catalogBuilder.register([...this.sitePath, "path", ...path], leaf);
      if (origin !== null) origins.push([path, origin]);
    }
    // Only a container walks dynamic: a top-level string returned above.
    const tree = walked.tree as Record<string, unknown> | unknown[];
    return new TaggedSlot(new ValueTemplate(tree, origins), contract, true);
  }

  /** Walk one node, reading a tag on it unless it is the slot itself. */
  walk(cursor: SourceCursor, value: unknown, path: ValuePath): Walked {
    if (path.length > 0 && cursor.tag !== null) return this.walkTagged(cursor, value, path);
    return this.walkValue(cursor, value, path);
  }

  /** Read a nested tagged node into a typed `Binding`, or check and fold it. */
  walkTagged(cursor: SourceCursor, value: unknown, path: ValuePath): Walked {
    this.rejectSpacedTag(cursor, path);
    const typ = this.tagType(cursor, path);
    if (!cursor.hasValue) {
      throw this.diagnostics.error(
        `${reprText(pathLabel(this.label, path))}: a nested tag needs a value: ` +
          `\`!${cursor.tag} <value>\``,
        { range: cursor.range },
      );
    }
    const inner = this.walkValue(cursor, value, path);
    if (!inner.hasSpan) this.check(cursor, path, typ, this.literal(inner.tree));
    if (this.literalOnly) return inner;
    const origin = this.catalogBuilder.register(
      [...this.sitePath, "type", "path", ...path],
      cursor,
    );
    const binding = new Binding(inner.tree, {
      origin,
      contract: new ValueContract(typ, new ContractOrigins({ default: origin })),
    });
    return new Walked(binding, { hasSpan: inner.hasSpan, hasBinding: true });
  }

  /** Walk one node's own value, reading the tags on its children. */
  walkValue(cursor: SourceCursor, value: unknown, path: ValuePath): Walked {
    if (typeof value === "string") {
      // A file's text or data is never a template; a literal's strings are text anyway.
      if (cursor.verbatim) return new Walked(this.literalOnly ? value : new Verbatim(value));
      if (this.literalOnly || path.length === 0) return new Walked(value);
      // Quoted, it is text, as at the top level (`isTemplateText`).
      if (!QUOTED_STYLES.has(cursor.scalarStyle) && CALL_HEAD.test(value.trim())) {
        throw this.diagnostics.error(
          `${reprText(pathLabel(this.label, path))}: a \`call(...)\` inside a value isn't ` +
            `supported; ${this.nestedCallFix}`,
          { range: cursor.range },
        );
      }
      this.leafSites.push([path, cursor]);
      return new Walked(value, { hasSpan: hasSpan(value) });
    }
    if (isRecord(value)) {
      const entries: [string, unknown][] = [];
      const walked = new Walked(null);
      for (const [key, item] of Object.entries(value)) {
        const part = this.walk(cursor.child(key), item, [...path, key]);
        entries.push([key, part.tree]);
        walked.absorb(part);
      }
      // `fromEntries` keeps a `__proto__` key as data.
      walked.tree = Object.fromEntries(entries);
      return walked;
    }
    if (Array.isArray(value)) {
      const items: unknown[] = [];
      const walked = new Walked(items);
      for (const [index, item] of value.entries()) {
        const part = this.walk(cursor.child(index), item, [...path, index]);
        items.push(part.tree);
        walked.absorb(part);
      }
      return walked;
    }
    return new Walked(value);
  }

  /**
   * Record the tag on `cursor` as consumed and return the type it names.
   *
   * @throws LoadError when the tag names an inbox or a memory anywhere but a flow input, or a
   *   flow input's tag names a host memory type the file also declares as a typedef.
   */
  tagType(cursor: SourceCursor, path: ValuePath): Type {
    const definitionId = cursor.tagDefinitionId;
    if (definitionId !== null) this.ledger.add(definitionId);
    const label = pathLabel(this.label, path);
    const input = path.length === 0 && this.inbox;
    const registry = input
      ? inputTagRegistry(cursor.tag!, this.registry, this.memoryTypes, { label, cursor, diagnostics: this.diagnostics })
      : this.registry;
    const typ = readTagType(cursor.tag!, registry, {
      label,
      diagnostics: this.diagnostics,
      range: cursor.range,
      include: cursor.typedInclude?.path ?? null,
    });
    if (isInboxType(typ) && !input) {
      throw this.diagnostics.error(
        `${reprText(label)}: \`!${cursor.tag}\` declares an inbox, which only a flow input can be`,
        { range: cursor.range },
      );
    }
    if (isMemoryType(typ) && !input) {
      throw this.diagnostics.error(
        `${reprText(label)}: \`!${cursor.tag}\` declares a memory, which only a flow input can be`,
        { range: cursor.range },
      );
    }
    return typ;
  }

  /**
   * Reject plain text starting with `(` right after a tag.
   *
   * When the tag and the parenthesized part together name a type (`!int (1..12) 5`), a space
   * cut the tag's arguments off; otherwise the text really starts with `(`
   * (`!str (555) 123-4567`) and must be quoted so a reader doesn't take it for arguments. A tag
   * that names no type either way is left for `readTagType` to report.
   */
  rejectSpacedTag(cursor: SourceCursor, path: ValuePath): void {
    if (cursor.scalarStyle !== "plain") return;
    const value = cursor.value;
    if (typeof value !== "string" || !value.startsWith("(")) return;
    const tag = cursor.tag!;
    const close = closingParen(value);
    const joined = tag + (close < 0 ? value : value.slice(0, close + 1));
    const label = pathLabel(this.label, path);
    let message: string;
    if (writableAsTag(joined) && resolves(joined, this.registry)) {
      message =
        `${reprText(label)}: a YAML tag ends at the first space; ` +
        `write \`!${joined}\` without spaces`;
    } else if (resolves(tag, this.registry)) {
      const quoted = `!${tag} ${yamlQuoted(value)}`;
      message =
        `${reprText(label)}: a YAML tag ends at the first space; quote the text: ` +
        (path.length === 0 ? `\`${this.label}: ${quoted}\`` : `\`${quoted}\``);
    } else {
      return;
    }
    throw this.diagnostics.error(message, { range: cursor.range });
  }

  /**
   * Whether a top-level value is binding text resolved once sources are known.
   *
   * Unquoted text starting with `call(` is left to the inline-call desugar, which requires
   * exactly one well-formed directive; quoted, it is text.
   */
  isTemplateText(value: unknown, cursor: SourceCursor): boolean {
    if (this.literalOnly || typeof value !== "string") return false;
    if (hasSpan(value)) return true;
    return cursor.scalarStyle === "plain" && CALL_HEAD.test(value.trim());
  }

  /** Return the value a span-free tree binds: no bindings, strings unescaped, `Verbatim` text as is. */
  literal(tree: unknown): unknown {
    if (tree instanceof Binding) return this.literal(tree.source);
    if (tree instanceof Verbatim) return tree.text;
    if (typeof tree === "string") {
      return this.literalOnly ? tree : evalBinding(tree, MISSING_ROOT);
    }
    if (isRecord(tree)) {
      return Object.fromEntries(
        Object.entries(tree).map(([key, item]) => [key, this.literal(item)]),
      );
    }
    if (Array.isArray(tree)) return tree.map((item) => this.literal(item));
    return tree;
  }

  /**
   * Check a span-free literal against its tag's type, explaining a mismatch.
   *
   * @throws LoadError when the literal does not fit `typ`.
   */
  check(cursor: SourceCursor, path: ValuePath, typ: Type, value: unknown): void {
    try {
      buildValueAs(typ, value);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw this.diagnostics.error(this.mismatch(cursor, path, typ, value, error), {
        range: cursor.range,
      });
    }
  }

  /**
   * Explain why a tagged literal does not fit, suggesting a rewrite when one helps.
   *
   * A former YAML boolean word such as `yes`, now text, written bare gets the `true`/`false` to
   * write when that boolean fits; a non-string scalar whose text would fit gets the quoted form.
   * A value an `!include(T)` read names its file, and the place and the line in the file the
   * check failed at.
   */
  mismatch(
    cursor: SourceCursor,
    path: ValuePath,
    typ: Type,
    value: unknown,
    error: TypeCheckError,
  ): string {
    const label = pathLabel(this.label, path);
    const tag = cursor.tag;
    const include = cursor.typedInclude;
    if (include !== null) {
      const places = error.path.length > 0 ? [`at \`${pathLabel(label, error.path)}\``] : [];
      const line = includedLine(include, error.path);
      if (line !== null) places.push(`line ${line} of the file`);
      const message = `${reprText(label)}: ${includeFile(include.path, include.type)} is not a valid ${tag}: ${error.message}`;
      return places.length === 0 ? message : `${message} (${places.join(", ")})`;
    }
    const text = cursor.scalarText;
    if (text === null) {
      let message = `${reprText(label)}: the value is not a valid ${tag}: ${error.message}`;
      if (error.path.length > 0) message += ` (at \`${pathLabel(label, error.path)}\`)`;
      return message;
    }
    // Quoted and block text is shown quoted, so its line breaks stay escaped.
    const written = cursor.scalarStyle === "plain" ? text : yamlQuoted(text);
    let message = `${reprText(label)}: \`${written}\` is not a valid ${tag}`;
    if (error.path.length > 0) {
      message += ` (${error.message})`;
    } else {
      const reason = mismatchReason(error, value);
      if (reason !== null) message += ` (${reason})`;
    }
    const fix = formerBooleanFix(value, typ);
    if (fix !== null) return `${message}; ${fix[1]}`;
    // Quoting only helps when YAML read the text as a non-string and the text itself would
    // fit, e.g. `!str 30`.
    if (typeof value === "string") return message;
    if (value === null && typ.kind === ValueKind.ANY && !typ.nullable) {
      // `Any` would read the quoted text as the string "null", which is rarely what a null
      // under `!Any` means.
      const optional = `!Optional(${tag}) null`;
      return (
        message +
        "; to allow null, write " +
        (path.length === 0 ? `\`${this.label}: ${optional}\`` : `\`${optional}\``)
      );
    }
    if (!valueFits(typ, text)) return message;
    const quoted = `!${tag} ${yamlQuoted(text)}`;
    message +=
      "; quote it to keep it as text: " +
      (path.length === 0 ? `\`${this.label}: ${quoted}\`` : `\`${quoted}\``);
    if (typeof value === "boolean") message += " (YAML reads unquoted true/false as booleans)";
    return message;
  }
}

/**
 * Return the reason a root type check gives beyond the plain mismatch, else `null`.
 *
 * A date or datetime check, for example, says what the text must look like.
 *
 * @param error The failed check of `value`, at the root.
 * @param value The checked literal.
 * @returns The reason, or `null` when the error only says the value does not match.
 */
function mismatchReason(error: TypeCheckError, value: unknown): string | null {
  const head = `${valueRepr(value)} does not match declared type `;
  const text = error.message;
  if (!text.startsWith(head)) return null;
  const rest = text.slice(head.length);
  const at = rest.indexOf(": ");
  const reason = at < 0 ? "" : rest.slice(at + 2);
  return reason || null;
}

/**
 * Return the line, in the file an `!include(T)` read, of the deepest part of its value on `path`;
 * `null` for text `!include(str)` read, or an empty file.
 */
function includedLine(include: TypedInclude, path: ValuePath): number | null {
  if (include.data === null) return null;
  let cursor = include.data.root;
  for (const step of path) {
    // A missing field's path ends past the value; the line is the record's.
    if (typeof step === "string" ? !cursor.has(step) : !isList(cursor.value) || step >= cursor.value.length) break;
    cursor = cursor.child(step);
  }
  return cursor.range?.startLine ?? null;
}

/**
 * Reject a tag or text `!include` read on a node's `output:` type declaration or inside it.
 *
 * An `output:` declaration is a type, so a tag there would type nothing.
 *
 * @param cursor The node's `output:` value.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError at the first tag or `!include` on or inside the declaration.
 */
export function rejectTaggedOutput(cursor: SourceCursor, diagnostics: LoadDiagnostics): void {
  // Each entry: a cursor, the key it sits under, and whether it is a list item.
  const stack: [SourceCursor, string, boolean][] = [[cursor, "output", false]];
  while (stack.length > 0) {
    const [current, key, isItem] = stack.pop()!;
    const tag = current.tag;
    const value = current.value;
    if (current.included || current.typedInclude !== null) {
      const type = current.typedInclude?.type ?? null;
      throw diagnostics.error(`\`output:\` declares a type, which ${includeTag(type)} can't give; write the type`, {
        range: current.range,
      });
    }
    if (tag !== null) {
      let message: string;
      if (isItem) {
        message =
          `\`output:\` declares a type; an item of \`${key}:\` is tagged \`!${tag}\`; ` +
          "write the type without a tag";
      } else {
        const written = current.hasValue && typeof value === "string" ? value : tag;
        message = `\`output:\` declares a type; write \`${key}: ${written}\`, without a tag`;
      }
      throw diagnostics.error(message, { range: current.range });
    }
    if (isRecord(value)) {
      const children = Object.keys(value).map((name) => current.child(name));
      for (const child of children.reverse()) stack.push([child, authoredKey(child), false]);
    } else if (Array.isArray(value)) {
      for (let index = value.length - 1; index >= 0; index--) {
        stack.push([current.child(index), key, true]);
      }
    }
  }
}

/** Spell the mapping key selecting `cursor` as written. */
function authoredKey(cursor: SourceCursor): string {
  return cursor.keyText ?? String(cursor.sourceKeyValue);
}

/**
 * Reject the first tag in `document` that no slot reader consumed.
 *
 * A tag types a flow input, a binding, a flow output, an agent variable, an `llm_config` value
 * (the whole record or a field), a value entry in `nodes:`, or a value inside one; anywhere
 * else it would be silently ignored. Tags reached only through top-level `x-*` keys are exempt.
 * Tags under `typedefs:` and on mapping keys get their own messages. Keys are spelled as
 * written, so `on:` stays `on:`.
 *
 * @param document The loaded document.
 * @param consumed Definition ids of the tags slot readers read.
 * @param diagnostics Factory for located load errors.
 * @param options.typedefsOnly Check only tags under `typedefs:`, which are read before any
 *   slot. Defaults to `false`.
 * @throws LoadError at the first offending tag, in document order.
 */
export function rejectUnreadTags(
  document: SourceDocument,
  consumed: ReadonlySet<number>,
  diagnostics: LoadDiagnostics,
  options: { typedefsOnly?: boolean } = {},
): void {
  const typedefsOnly = options.typedefsOnly ?? false;
  for (const node of document.taggedNodes({ skipTopLevel: isExtensionKey })) {
    const path = node.keyPath;
    const inTypedefs = path.length >= 2 && path[0] === "typedefs";
    if (typedefsOnly && !inTypedefs) continue;
    const key = path.findLast((step): step is string => typeof step === "string") ?? null;
    if (node.isKey && key !== null) {
      throw diagnostics.error(
        `\`!${node.tag} ${key}:\` tags a mapping key; a tag types a value, so put it after ` +
          `the colon: \`${key}: !${node.tag} <value>\``,
        { range: node.range },
      );
    }
    if (inTypedefs) {
      const name = path[1]!;
      const how = node.include === null ? "without a tag" : `which ${includeTag(node.include.type)} can't give`;
      throw diagnostics.error(
        `typedef ${typeof name === "string" ? reprText(name) : name}: a type definition is ` +
          `written as a type, ${how}`,
        { range: node.range },
      );
    }
    if (consumed.has(node.definitionId)) continue;
    const where = key !== null ? `\`${key}:\`` : "the document root";
    if (node.include !== null) {
      const { type, path: file } = node.include;
      throw diagnostics.error(
        `${includeFile(file, type)}: a value read with \`!include(T)\` goes where a tag goes — a flow ` +
          "input, a binding, a flow output, an agent variable, an `llm_config` value, a value entry in " +
          "`nodes:`, or a value inside one — and `!include(str)` text also on an agent's `prompt:`, " +
          `\`system_prompt:\` or prompt message; ${where} is none of these`,
        { range: node.range },
      );
    }
    throw diagnostics.error(
      "a tag types a value — a flow input, a binding, a flow output, an agent variable, an " +
        "`llm_config` value, a value entry in `nodes:`, or a value inside one; " +
        `${where} is not one`,
      { range: node.range },
    );
  }
}

/** Spell a slot name and a path inside its value, as in `x.items[0]`. */
export function pathLabel(name: string, path: ValuePath): string {
  return name + path.map((part) => (typeof part === "string" ? `.${part}` : `[${part}]`)).join("");
}

/** Return whether `text` holds a `${...}` span; malformed text counts, for later checks. */
export function hasSpan(text: string): boolean {
  try {
    return scanTemplate(text).some((segment) => segment instanceof Span);
  } catch (error) {
    if (error instanceof ExpressionError) return true;
    throw error;
  }
}

/** Quote `text` as a YAML scalar that reads back as exactly `text`. */
export function yamlQuoted(text: string): string {
  // A single-quoted scalar folds line breaks and cannot escape, so text with a control
  // character needs a double-quoted scalar with escapes.
  const chars = [...text];
  if (!chars.every((ch) => isPrintable(ch.codePointAt(0)!))) {
    return '"' + chars.map(yamlEscaped).join("") + '"';
  }
  // A double-quoted scalar treats backslashes as escapes; a single-quoted one only needs its
  // own quote doubled.
  if (text.includes('"') || text.includes("\\")) return "'" + text.replaceAll("'", "''") + "'";
  return `"${text}"`;
}

const SHORT_ESCAPES: ReadonlyMap<string, string> = new Map([
  ['"', '\\"'],
  ["\\", "\\\\"],
  ["\n", "\\n"],
  ["\r", "\\r"],
  ["\t", "\\t"],
  ["\b", "\\b"],
  ["\f", "\\f"],
]);

/** Escape one character for a YAML double-quoted scalar. */
function yamlEscaped(ch: string): string {
  const cp = ch.codePointAt(0)!;
  if (isPrintable(cp) && ch !== '"' && ch !== "\\") return ch;
  if (cp > 0xffff) {
    // JSON would write a surrogate pair, which YAML does not join.
    return "\\U" + cp.toString(16).toUpperCase().padStart(8, "0");
  }
  // JSON's escapes for the basic plane are all valid YAML escapes.
  return SHORT_ESCAPES.get(ch) ?? "\\u" + cp.toString(16).padStart(4, "0");
}
