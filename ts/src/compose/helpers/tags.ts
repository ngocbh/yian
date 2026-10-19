/**
 * Read a slot's type and value through its YAML tags.
 *
 * A slot's type is a local YAML tag on its value (`qty: !int ${size.output.qty}`). A tag ends
 * at the first space and, in strict YAML 1.2, can't contain `,`, `[`, `]`, `{` or `}`, so a
 * type with several arguments is named in `typedefs:` and tagged by its name. `readTagType`
 * turns tag text into a `Type` and explains each way a tag goes wrong; `suggestTag` spells a
 * tagged slot for error messages.
 *
 * A tag only checks a value and declares its type: it never adds or changes a field.
 */

import { getCloseMatches, isPrintable, reprText, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { scanTemplate, Span } from "../../expr/template.ts";
import type { ValuePath } from "../../nodes/contracts.ts";
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
  isListKind,
  requiresFile,
  Type,
  TypeCheckError,
  typeEquals,
  ValueKind,
  type Value,
} from "../../typesys/values.ts";
import type { ScalarStyle } from "../source/common.ts";
import { parseYamlLiteral, SourceSyntaxError } from "../source/index.ts";
import { CALL_HEAD } from "./directives.ts";
import type { LoadDiagnostics } from "./errors.ts";
import { formerBooleanFix } from "./yaml_words.ts";

const HEAD_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*/;
// The `difflib` ratio a name must reach to be offered as a close match.
export const CLOSE_MATCH_CUTOFF = 0.6;
const FLOW_INDICATORS = /[[\]{}]/;
// The characters PyYAML reads into a tag, minus YAML 1.2's flow indicators, `!`, and the `%`
// that starts an escape: text made only of these reads back as one tag.
const TAG_CHARACTERS = /^[A-Za-z0-9\-;/?:@&=+$_.~*'()]+$/;
const TYPEDEF_EXAMPLE = "(e.g. `Decision: Union(Buy, Sell)`) and tag with its name (`!Decision`)";
/** YAML scalar styles that make a value text, never a `call(...)` directive. */
export const QUOTED_STYLES: ReadonlySet<ScalarStyle> = new Set([
  "single",
  "double",
  "literal",
  "folded",
]);
const OLD_SEPARATOR = " = ";
// YAML drops the trailing space of `qty: int = `, delivering "int =".
const OLD_TRAILING_SEPARATOR = " =";

/**
 * Resolve a slot's tag text into its type.
 *
 * @param tag The tag text without the leading `!`, e.g. `"list(Buy)"`.
 * @param registry The flow's typedefs the tag resolves against.
 * @param options.label The slot's name, used in error messages.
 * @param options.diagnostics Factory for located load errors.
 * @param options.range The tagged value's location, attached to any error.
 * @returns The type the tag names.
 * @throws LoadError when the tag holds a bracket or brace (the message shows the
 *   parenthesized spelling), is cut short by a space (it ends in `,` or has an unclosed `(`),
 *   holds a `,`, or does not name a type (the message offers a close match).
 */
export function readTagType(
  tag: string,
  registry: TypeRegistry,
  options: { label: string; diagnostics: LoadDiagnostics; range?: SourceRange | null },
): Type {
  const { label, diagnostics, range = null } = options;
  if (FLOW_INDICATORS.test(tag)) {
    let message =
      `${reprText(label)}: a tag can't contain \`[\`, \`]\`, \`{\` or \`}\` (strict YAML 1.2)`;
    const spelled = respelled(tag, registry);
    if (spelled !== null && writableAsTag(spelled)) {
      message += `; write \`!${spelled}\``;
    } else if (spelled !== null) {
      message += `; name \`${spelled}\` in \`typedefs:\` and tag with its name`;
    }
    throw diagnostics.error(message, { range });
  }
  if (tag.endsWith(",") || tag.split("(").length > tag.split(")").length) {
    throw diagnostics.error(
      `${reprText(label)}: \`!${tag}\` is not a complete type: a YAML tag ends at the first ` +
        "space and can't contain `,`; name a type with several arguments in " +
        `\`typedefs:\` ${TYPEDEF_EXAMPLE}`,
      { range },
    );
  }
  if (tag.includes(",")) {
    throw diagnostics.error(
      `${reprText(label)}: \`!${tag}\`: a tag can't contain \`,\`; name it in \`typedefs:\` ` +
        TYPEDEF_EXAMPLE,
      { range },
    );
  }
  try {
    return typeFor(tag, registry);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    let message = `${reprText(label)}: \`!${tag}\` is not a type (${error.message})`;
    const match = closeTypeName(tag, registry);
    if (match !== null) message += `. Did you mean \`${match}\`?`;
    throw diagnostics.error(message, { range });
  }
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
  if (fits(typ, parsed)) return [text, null];
  const fix = formerBooleanFix(parsed, typ);
  if (fix !== null) return [fix[0] ? "true" : "false", fix[1]];
  if (typeof parsed !== "string") return fits(typ, text) ? [yamlQuoted(text), null] : [text, null];
  return [text, null];
}

/** Return whether `value` passes the write-boundary check for `typ`. */
function fits(typ: Type, value: unknown): boolean {
  try {
    buildValueAs(typ, value);
  } catch (error) {
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
  return true;
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
  const sorted = [...names].sort();
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
    [...byLower.keys()].sort(),
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
  const text = tagText(typ, { registry: options.registry ?? null, typeText });
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
  const tag = suggestTag(name, sink, "${...}", { registry: options.registry ?? null });
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
  const { registry = null, typeText = null } = options;
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

/** Spell `typ` by its structure; only a list element may fall back to a typedef name. */
function directTagText(typ: Type, registry: TypeRegistry): string | null {
  if (typ.variants !== undefined || typ.fields !== undefined) return null;
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
  } else if (typ.fields !== undefined || typ.variants !== undefined) {
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
  // JSON's escapes for the basic plane, as Python's `json.dumps` writes them, are all valid
  // YAML escapes.
  return SHORT_ESCAPES.get(ch) ?? "\\u" + cp.toString(16).padStart(4, "0");
}
