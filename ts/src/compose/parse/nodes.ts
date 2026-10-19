/**
 * Graph-stage: parse the `nodes:` mapping into typed per-kind descriptors.
 *
 * The first cross-node assembly pass. Consumes a source cursor over the flow's `nodes:` mapping
 * and classifies each entry by its YAML tag: a tagged entry is a value
 * (`total: !int ${a.output + b.output}`) handed to the caller's `readValue` reader, and an
 * untagged mapping with `kind:` is a node body dispatched to its kind's `parse` via the `KINDS`
 * table. Any other entry is a located error that says to add `kind:` for a node or a tag for a
 * value.
 *
 * Only the generic, kind-independent pre-checks live here (node id shape and reserved ids, entry
 * classification, the `input:`/`output:` to plural back-map, `kind:` resolution); each kind's own
 * `parse` owns its field legality and its descriptor-local `SourceRange` recording (each field
 * keyed `siteKey("field", name)`). Two kind-blind enrichments run after every kind parse: each
 * `input:` binding's own range is stamped into the descriptor's `fieldRanges` under
 * `siteKey("input", name)`, and each `asserts:` item range is appended positionally to its
 * `assertRanges`. This metadata stays inside Compose; `_finish` copies node and field locations
 * to the `BuildResult` for validation.
 *
 * Knows about: diagnostics, compose source, helpers and node kinds, compile model, expr and
 * typesys.
 * Never imports: the runtime, the run boundary, the CLI or provider SDKs.
 */

import { END_ID, START_ID } from "../../compile/model.ts";
import { compareCodeUnits, reprText } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { EXPRESSION_KEYWORDS } from "../../expr/grammar.ts";
import { scanTemplate, Span } from "../../expr/template.ts";
import {
  buildValue,
  isList,
  isListKind,
  isRecord,
  listKindForElement,
  Type,
  TypeCheckError,
  typeEquals,
  ValueKind,
} from "../../typesys/values.ts";
import { CALL_HEAD } from "../helpers/directives.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { oneLineText } from "../helpers/source.ts";
import { QUOTED_STYLES, suggestTag, yamlQuoted } from "../helpers/tags.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { type NodeDescriptor, notTextError, siteKey } from "../nodes/base.ts";
import { KINDS, REMOVED_KINDS } from "../nodes/index.ts";
import type { ScalarStyle } from "../source/common.ts";
import {
  parseYamlLiteral,
  SourceAccessError,
  type SourceCursor,
  SourceSyntaxError,
} from "../source/index.ts";

// Singular source keys are projected onto the descriptors' plural field names while keeping
// their original child sources.
const INPUT_OUTPUT_REMAP: ReadonlyMap<string, string> = new Map([
  ["input", "inputs"],
  ["output", "outputs"],
]);

// Explicit ASCII classes, matching the expression `NAME` terminal: `${...}` can't reference a
// non-ASCII letter (`café`).
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NON_IDENTIFIER_CHAR = /[^A-Za-z0-9_]/gu;

// Item types a list of numbers joins to `float`.
const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);
// A null while list items are joined: its type is the other items' type, made optional.
const NULL = new Type({ kind: ValueKind.NONE, nullable: true });
// The item type of a list whose items share no type.
const ANY = Type.scalar(ValueKind.ANY);

/** The identifier rule, as node id messages state it. */
export const IDENTIFIER_RULE = "ASCII letters, digits, and _, not starting with a digit";

/** Whether `name` is an identifier `${...}` can reference: text matching the ASCII rule. */
export function isIdentifier(name: unknown): boolean {
  return typeof name === "string" && IDENTIFIER.test(name);
}

/**
 * Spell `name` as an identifier, for a rename hint: each code point outside `[A-Za-z0-9_]`
 * becomes `_`, and a `_` is prefixed before a leading digit; `""` for an empty name.
 */
export function identifierSuggestion(name: string): string {
  const suggestion = name.replace(NON_IDENTIFIER_CHAR, "_");
  return /^[0-9]/.test(suggestion) ? `_${suggestion}` : suggestion;
}

/** The message for a `nodes:` entry that is neither a node body nor a tagged value. */
function nodeOrValue(nodeId: string): string {
  return (
    `node ${reprText(nodeId)}: an entry in \`nodes:\` is a node or a value; add \`kind:\` for a ` +
    "node, or a tag (`!object`, `!Buy`, ...) for a value"
  );
}

/** Reads one tagged `nodes:` entry, given its id and cursor, into its descriptor. */
export type ValueReader = (nodeId: string, cursor: SourceCursor) => NodeDescriptor;

/**
 * Parse one `nodes:` source cursor into typed per-kind descriptors.
 *
 * The source is physical for a full graph, source-aligned for compact or def projections, or
 * virtual for a programmatic or test mapping. A tagged entry is a value and is read by
 * `readValue`; every other entry must be a node body with `kind:`.
 *
 * @param nodesCursor Cursor over the `nodes:` mapping; a non-mapping value parses to no nodes.
 * @param options.readValue Reads one tagged entry into its descriptor; may be omitted only for a
 *   mapping with no tagged entry.
 * @returns The descriptors keyed by entry id, in source order, each anchored on its entry's key
 *   line whenever an authored source exists.
 * @throws LoadError at the entry for a bad or reserved id, an untagged entry that is not a node
 *   body, or a node body that fails its kind's parse; at the `kind:` value for a non-text or
 *   unknown kind.
 * @throws TypeError when a tagged entry is found and `readValue` is absent.
 */
export function parseNodes(
  nodesCursor: SourceCursor,
  options: { readValue?: ValueReader } = {},
): Map<string, NodeDescriptor> {
  const diagnostics = LoadDiagnostics.fromCursor(nodesCursor);
  const out = new Map<string, NodeDescriptor>();
  try {
    const nodes = nodesCursor._valueView();
    if (!isRecord(nodes)) return out;
    for (const nodeId of Object.keys(nodes)) {
      let desc = parseNode(nodeId, nodesCursor.child(nodeId), options.readValue);
      // Anchor the node-level range on the node's key, not its body: the anchor a cycle, an
      // unresolved ref, or a namespaced runtime failure boxes.
      const header = nodesCursor.keyRangeOf(nodeId);
      if (header !== null && desc.range !== null) desc = desc.replace({ range: header });
      out.set(nodeId, desc);
    }
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
  return out;
}

/**
 * Read one `nodes:` entry into its descriptor: a tagged entry goes to `readValue`; an untagged
 * one must be a node body with `kind:` and is dispatched to its kind's `parse`.
 */
function parseNode(
  nodeId: string,
  cursor: SourceCursor,
  readValue: ValueReader | undefined,
): NodeDescriptor {
  const diagnostics = LoadDiagnostics.fromCursor(cursor);
  rejectReservedNodeId(nodeId, cursor);
  // A tag makes the entry a value even when its mapping has a `kind:` key.
  if (cursor.tag !== null) {
    if (readValue === undefined) {
      throw new TypeError(
        `node ${reprText(nodeId)} is a tagged value entry; parseNodes needs readValue to read it`,
      );
    }
    return readValue(nodeId, cursor);
  }
  if (!isRecord(cursor._valueView())) rejectUntaggedValue(nodeId, cursor, diagnostics);
  const normalized = normalizeNodeSignatureCursor(cursor);

  const kind = normalized.get("kind");
  if (kind === null) {
    throw diagnostics.error(nodeOrValue(nodeId), {
      range: normalized.range ?? normalized.keyRange,
    });
  }
  if (typeof kind !== "string") {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: ${notTextError("kind", kind, `write one of ${kindNames()}`)}`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const removed = REMOVED_KINDS.get(kind);
  if (removed !== undefined) {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: kind ${reprText(kind)} was removed; ${removed}`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const cls = KINDS.get(kind);
  if (cls === undefined) {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: unknown kind ${reprText(kind)} (allowed: ${kindNames()})`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const desc = cls.parse(nodeId, normalized);
  // Compose-only enrichments, read from the normalized cursor (`input:` is `inputs` there):
  // each `input:` binding's own range, keyed apart from a like-named field, and each `asserts:`
  // item's range by position, since two identical asserts still need distinct carets. Both
  // no-op for a source-free cursor.
  for (const [name, range] of normalized.mappingValueRanges("inputs")) {
    desc.fieldRanges.set(siteKey("input", name), range);
  }
  desc.assertRanges.push(...normalized.itemRangeList("asserts"));
  return desc;
}

/** The registered kind names, sorted and comma-joined, for a bad-kind message. */
function kindNames(): string {
  return [...KINDS.keys()].sort(compareCodeUnits).join(", ");
}

/**
 * Reject a node id that isn't a referenceable identifier or collides with a reserved name, at
 * the node body: an expression keyword, the synthesized boundary or a resolver head; `#`, `/`
 * and `.` get their own reserved-separator message.
 *
 * @throws LoadError for a bad id.
 */
function rejectReservedNodeId(nodeId: string, cursor: SourceCursor): void {
  const diagnostics = LoadDiagnostics.fromCursor(cursor);
  const range = cursor.range;
  const id = reprText(nodeId);
  if (nodeId.includes("#") || nodeId.includes("/") || nodeId.includes(".")) {
    throw diagnostics.error(
      `node id ${id} uses a reserved separator ('#' and '/' are reserved for runtime graph ` +
        "expansion; '.' is reserved for reference paths)",
      { range },
    );
  }
  if (!isIdentifier(nodeId)) {
    const reference = nodeId ? ` so \${${nodeId}.output} can reference it` : "";
    const suggestion = identifierSuggestion(nodeId);
    const hint = suggestion ? `; try ${reprText(suggestion)}` : "";
    throw diagnostics.error(
      `node id ${id} must be an identifier (${IDENTIFIER_RULE})${reference}${hint}`,
      { range },
    );
  }
  if (nodeId === START_ID || nodeId === END_ID) {
    throw diagnostics.error(
      `node id ${id} is reserved for the synthesized START_ID/END_ID boundary; authors write ` +
        "`input:`/`output:`, not a boundary node",
      { range },
    );
  }
  if (EXPRESSION_KEYWORDS.has(nodeId)) {
    throw diagnostics.error(
      `node id ${id} is an expression keyword, so \${${nodeId}.output} can't reference it; ` +
        "pick a different node id",
      { range },
    );
  }
  // The four singular head literals keep the node-first ref `${<node>.output.k}` unambiguous;
  // the plural forms catch migration typos.
  if (["input", "output", "system", "item"].includes(nodeId)) {
    throw diagnostics.error(
      `node id ${id} is reserved (used as \`\${${nodeId}.X}\` as a resolver head — pick a ` +
        "different node id)",
      { range },
    );
  }
  if (nodeId === "inputs" || nodeId === "outputs") {
    throw diagnostics.error(
      `node id ${id} is reserved (the plural head is retired — rename to ` +
        `\`${nodeId.slice(0, -1)}\` or pick a different node id)`,
      { range },
    );
  }
}

/**
 * Reject an untagged scalar, list or null entry, showing the tag to write, at the entry.
 *
 * A literal gets an example of the entry with its tag. A `${...}` or `call(...)` value is not a
 * literal, and no tag holds a value that contains itself, so neither gets an example.
 *
 * @throws LoadError always; one nested deeper than a yian value may report the nesting limit
 *   instead.
 */
function rejectUntaggedValue(
  nodeId: string,
  cursor: SourceCursor,
  diagnostics: LoadDiagnostics,
): never {
  const value = cursor.value;
  const range = cursor.range ?? cursor.keyRange;
  // First: the checks below walk the value recursively, and one past the limit couldn't load as
  // the suggested entry either.
  checkWrittenValue(value, `value entry ${reprText(nodeId)}`, range, diagnostics);
  let message = nodeOrValue(nodeId);
  if (!isReference(value, cursor.scalarStyle) && !containsItself(value)) {
    const typ = exampleType(value);
    if (typ !== null) message += `, e.g. ${suggestTag(nodeId, typ, exampleText(value, cursor))}`;
  }
  throw diagnostics.error(message, { range });
}

/** Return whether an untagged value is a `${...}` template or a `call(...)` directive. */
function isReference(value: unknown, style: ScalarStyle | null): boolean {
  if (typeof value !== "string") return false;
  if (isBindingText(value)) return true;
  // A quoted `call(...)` is text, as in a binding.
  return !QUOTED_STYLES.has(style) && CALL_HEAD.test(value.trim());
}

/**
 * Return whether `value` is template text holding at least one `${...}` span: plain text, a
 * `$${...}` escape (literal text) and malformed template text are not.
 */
function isBindingText(value: string): boolean {
  try {
    return scanTemplate(value).some((segment) => segment instanceof Span);
  } catch (error) {
    if (error instanceof ExpressionError) return false;
    throw error;
  }
}

/**
 * Return the widened type to tag an authored value with in an example.
 *
 * `Optional(str)` for null, `object` for a record, a list of its items' joined type, else the
 * widened type of the value (`float` for an infinity); `null` for a value no type holds. Items
 * join recursively: a null makes the type optional, ints and floats join to `float`, and an
 * empty list joins any list, so `[[1], [null]]` is `list(list(Optional(int)))` and `[[], [1]]`
 * is `list(list(int))`. A list of nulls only is `list(Optional(str))`. Items that share no type
 * are `Any`, or `Optional(Any)` when one of them is null, so `[1, a]` is `list(Any)` and
 * `[null, 1, a]` is `list(Optional(Any))`.
 *
 * @param value An authored YAML value that does not contain itself.
 */
export function exampleType(value: unknown): Type | null {
  const typ = itemType(value);
  return typ === null ? null : settled(typ);
}

/** Return `exampleType(value)`, but with `NULL` for a null left to be joined. */
function itemType(value: unknown): Type | null {
  if (value === null) return NULL;
  if (isRecord(value)) return Type.scalar(ValueKind.OBJECT);
  if (isList(value) && value.length > 0) {
    const types = value.map(itemType);
    if (types.some((typ) => typ === null)) return null;
    let element: Type | null = types[0]!;
    for (const typ of types.slice(1)) {
      element = element === null ? null : joined(element, typ!);
    }
    if (element !== null) return listOf(element);
    return listOf(ANY.replace({ nullable: value.includes(null) }));
  }
  try {
    // An empty list is a `list(Optional(Any))` with no element type, which joins any list.
    return Type.scalar(buildValue(value).kind);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/** Return the one type two item types widen to, or `null` when there is none. */
function joined(first: Type, second: Type): Type | null {
  if (first.kind === ValueKind.NONE) return second.replace({ nullable: true });
  if (second.kind === ValueKind.NONE) return first.replace({ nullable: true });
  const nullable = first.nullable || second.nullable;
  const a = first.replace({ nullable: false });
  const b = second.replace({ nullable: false });
  const isNumber = (typ: Type) => typeEquals(typ, INT) || typeEquals(typ, FLOAT);
  let result: Type;
  if (typeEquals(a, b)) {
    result = a;
  } else if (isNumber(a) && isNumber(b)) {
    result = FLOAT;
  } else if (isListKind(a.kind) && isListKind(b.kind)) {
    if (a.element === undefined || b.element === undefined) {
      result = a.element === undefined ? b : a;
    } else {
      const element = joined(a.element, b.element);
      if (element === null) return null;
      result = listOf(element);
    }
  } else {
    return null;
  }
  return nullable ? result.replace({ nullable: true }) : result;
}

/** Return `typ` with each `NULL` left in it read as `Optional(str)`. */
function settled(typ: Type): Type {
  if (typ.kind === ValueKind.NONE) return new Type({ kind: ValueKind.STRING, nullable: true });
  if (typ.element === undefined) return typ;
  const element = settled(typ.element);
  return typ.replace({ kind: listKindForElement(element), element });
}

/** Return the list type of an element type. */
function listOf(element: Type): Type {
  return new Type({ kind: listKindForElement(element), element });
}

/**
 * Return whether a list or record holds itself at some depth, through a YAML alias
 * (`&R [*R]`).
 *
 * @param value An authored YAML value or a part of one.
 * @param active The lists and records enclosing `value`.
 */
export function containsItself(value: unknown, active: ReadonlySet<object> = new Set()): boolean {
  if (!isList(value) && !isRecord(value)) return false;
  if (active.has(value)) return true;
  const inner = new Set(active).add(value);
  const items = isList(value) ? value : Object.values(value);
  return items.some((item) => containsItself(item, inner));
}

/**
 * Spell an untagged entry's value for its example: a value written over several lines (a block
 * list) as `...`, and text YAML would not read back as the same text quoted (`!!str 3` as
 * `"3"`).
 */
function exampleText(value: unknown, cursor: SourceCursor): string {
  if (value === null) return "null";
  const text = cursor.scalarText;
  if (typeof value === "string") {
    if (cursor.scalarStyle === "plain" && text !== null && readsAs(text, value)) return text;
    return yamlQuoted(value);
  }
  if (text !== null) return text;
  return oneLineText(cursor.document, cursor.range) || "...";
}

/** Return whether the plain scalar `text` reads back as the string `value`. */
function readsAs(text: string, value: string): boolean {
  try {
    return parseYamlLiteral(text) === value;
  } catch (error) {
    if (error instanceof SourceSyntaxError) return false;
    throw error;
  }
}

/**
 * Project authored node-signature keys onto descriptor field names: `input:` to `inputs` and
 * `output:` to `outputs`, keeping each child's source.
 *
 * The projection is the single crossing from the singular Compose surface to the plural carrier
 * fields; later phases that correlate a descriptor with its source reuse this cursor so collision
 * and source-selection rules can't drift from parsing.
 *
 * @throws SourceAccessError when `cursor` is not a traversable mapping, or when both spellings of
 *   one key are written.
 */
export function normalizeNodeSignatureCursor(cursor: SourceCursor): SourceCursor {
  return cursor.renamed(INPUT_OUTPUT_REMAP);
}
