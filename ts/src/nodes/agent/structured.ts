/**
 * Structured AGENT output: derive the JSON Schema of a declared `output:` Type.
 *
 * An AGENT declares `output:` as a typed `Type`. For a bare `str` the agent stays a text
 * producer: `typeToSchema` returns `null` and the AGENT loop takes the plain-text path. For any
 * constrained or richer type (a `Literal(...)` variant, record, union, scalar `int`/`float`/`bool`,
 * or list) it returns the JSON Schema the model is asked to fill: what Python yian's pydantic model
 * emits (`model_json_schema()`), with ints as `bigint`.
 *
 * Scalars and lists can't be a top-level schema on their own, so they are wrapped in a
 * single-field object (`ScalarWrapper` with `value`, `ListWrapper` with `items`), and so is a
 * union. A record maps one property per declared field. A nested record goes to `$defs` under
 * its position name (`meta`, `item`, `lines_item`; `Record` for a nullable root record) and is
 * referenced by `$ref`. A union is an `anyOf` of its members' `$ref`s, each member named by its
 * tag or numbered (`buy`, `item_Member1`); members and every record inside them are closed
 * (`additionalProperties: false`), so a reply can't match a member by ignoring fields.
 *
 * A `$defs` key is the position name with every code point outside `[A-Za-z0-9_-]` replaced by
 * `_`; the definition's `title` stays the raw name. Different definitions that want one key get
 * `<key>__1`, `<key>__2`, ... in definition order, skipping any key another definition wants;
 * identical ones share it.
 *
 * Where pydantic's output is an accident of Python, it differs: keys are sorted at every level
 * but a `properties` map (pydantic leaves unsorted the schema of a field named `properties` or
 * `default` and of a definition named `properties`); a `properties` map follows JS key order,
 * so integer-like field names come first, then the others in field order; fields starting with
 * `_` are kept, `__proto__` included (pydantic drops them); a `.` in a `$defs` key becomes `_`
 * (pydantic keeps the text after the last one), and a clashing key is `<key>__<n>` (pydantic
 * qualifies it with its module, `yian__nodes__agent__structured__a_b__1`).
 *
 * Only the schema is here. Generating a value and checking the reply are not ported yet.
 *
 * Knows about:   diagnostics and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { compareCodeUnits } from "../../diagnostics.ts";
import { type JsonObject, type JsonValue, logJson } from "../../typesys/json.ts";
import {
  type Bounds,
  IDENTIFIER_PATTERN,
  isListKind,
  listElementKind,
  setField,
  type Type,
  unionTagFields,
  ValueKind,
} from "../../typesys/values.ts";

// ValueKind -> JSON Schema type of a scalar slot. date and datetime persist as ISO strings.
const SCALAR_TYPES: ReadonlyMap<ValueKind, string> = new Map([
  [ValueKind.STRING, "string"],
  [ValueKind.INTEGER, "integer"],
  [ValueKind.NUMBER, "number"],
  [ValueKind.BOOLEAN, "boolean"],
  [ValueKind.DATE, "string"],
  [ValueKind.DATETIME, "string"],
]);

// Python's `str.isidentifier()`.
const IDENTIFIER = new RegExp(`^(?:${IDENTIFIER_PATTERN})$`, "u");

/**
 * A nested record's definition, collected while one schema is built: the `$defs` key it wants,
 * its schema (and that schema's text, to find an identical one), and the `$ref`s to it, which
 * get the final key once every definition is known.
 */
interface Definition {
  key: string;
  schema: JsonObject;
  text: string;
  refs: JsonObject[];
}

/** The definitions of one schema, in the order they are made (a record after those inside it). */
type Defs = Definition[];

/** Return an open object's schema: any mapping. */
function openObject(): JsonObject {
  return { additionalProperties: true, type: "object" };
}

/** Return a bare list's schema: any items. */
function bareList(): JsonObject {
  return { items: {}, type: "array" };
}

/** Return `Any`'s schema: any non-null JSON value, one arm per JSON type. */
function anySchema(): JsonObject {
  return {
    anyOf: [
      { type: "boolean" },
      { type: "integer" },
      { type: "number" },
      { type: "string" },
      openObject(),
      bareList(),
    ],
  };
}

/**
 * Return the schema of a value of `typ` at `position`; a nullable `typ` also accepts null.
 *
 * A `STRING` with `tags` is a `const` or `enum`; a nested `OBJECT` with `fields` goes to `defs`
 * under its position name and is referenced by `$ref`; a union is an `anyOf` of `$ref`s to its
 * closed members, in member order; a `LIST_*` is an array of its element schema; `Any` is an
 * `anyOf` of the JSON types.
 *
 * @param position The slot's position name, built by `nested`; `""` is the schema root.
 * @param closed Whether records reject unknown fields. Union members and everything inside
 *   them are always closed.
 * @throws RangeError when `typ` has no structured mapping (`NONE`, `FILE`), so a new
 *   `ValueKind` is loud, or a scalar's bounds are malformed.
 */
function schemaFor(typ: Type, defs: Defs, position: string, closed: boolean): JsonObject {
  const schema = nonNullSchema(typ, defs, position, closed);
  return typ.nullable ? orNull(schema) : schema;
}

/** `schemaFor` without the nullable widening. */
function nonNullSchema(typ: Type, defs: Defs, position: string, closed: boolean): JsonObject {
  const kind = typ.kind;
  if (kind === ValueKind.ANY) return anySchema();
  if (kind === ValueKind.STRING && typ.tags !== undefined && typ.tags.size > 0) {
    const tags = [...typ.tags].sort(compareCodeUnits);
    return tags.length === 1 ? { const: tags[0]!, type: "string" } : { enum: tags, type: "string" };
  }
  if (SCALAR_TYPES.has(kind)) {
    if (typ.bounds !== undefined) typ.validateShape();
    return scalarSchema(kind, typ.bounds);
  }
  if (kind === ValueKind.OBJECT && typ.variants !== undefined) {
    const names = memberNames(typ, position);
    return {
      anyOf: typ.variants.map((member, index) =>
        define(defs, names[index]!, recordSchema(member, defs, names[index]!, true)),
      ),
    };
  }
  if (kind === ValueKind.OBJECT) {
    if (!isRecord(typ)) return openObject();
    return define(defs, position, recordSchema(typ, defs, position, closed));
  }
  if (kind === ValueKind.LIST_ANY && typ.element === undefined) return bareList();
  if (isListKind(kind)) {
    if (typ.element !== undefined) {
      const items = schemaFor(typ.element, defs, nested(position, "item"), closed);
      return { items, type: "array" };
    }
    const scalar = listElementKind(kind);
    if (scalar === null || !SCALAR_TYPES.has(scalar)) return bareList();
    return { items: scalarSchema(scalar, undefined), type: "array" };
  }
  throw new RangeError(`type_to_schema: no structured mapping for segment type ${kind}`);
}

/**
 * Return the schema of a scalar slot of `kind`, within inclusive `bounds`.
 *
 * An open (`null`) end has no limit, so the schema omits that end's `minimum` or `maximum`.
 * A bound keeps the form the type stores it in (`bigint` or `number`).
 */
function scalarSchema(kind: ValueKind, bounds: Bounds | undefined): JsonObject {
  const schema: JsonObject = { type: SCALAR_TYPES.get(kind)! };
  const [minimum, maximum] = bounds ?? [null, null];
  if (minimum !== null) schema.minimum = minimum;
  if (maximum !== null) schema.maximum = maximum;
  return schema;
}

/** Return `schema` widened to accept null; the null arm joins an `anyOf` rather than nesting it. */
function orNull(schema: JsonObject): JsonObject {
  const arms = schema.anyOf;
  if (!Array.isArray(arms)) return { anyOf: [schema, { type: "null" }] };
  if (arms.some((arm) => (arm as JsonObject).type === "null")) return schema;
  return { ...schema, anyOf: [...arms, { type: "null" }] };
}

/** Return the position name of `segment` (a field, `item`, or a member) inside `position`. */
function nested(position: string, segment: string): string {
  return position ? `${position}_${segment}` : segment;
}

/**
 * Return the position names of the members of the union `typ` at `position`, in member order.
 *
 * A tagged union whose tag values are all identifiers (Python's `str.isidentifier()`) names
 * each member after its tag (`buy`, `sell`); otherwise the members are `Member1`, `Member2`, ...
 */
function memberNames(typ: Type, position: string): string[] {
  const variants = typ.variants ?? [];
  const [tagField] = unionTagFields(typ);
  let tags =
    tagField === undefined
      ? []
      : variants.map((member) => [...member.fields!.get(tagField)!.tags!][0]!);
  if (tags.length === 0 || !tags.every((tag) => IDENTIFIER.test(tag))) {
    tags = variants.map((_, index) => `Member${index + 1}`);
  }
  return tags.map((tag) => nested(position, tag));
}

/** Whether `typ` is a record with fields, whose schema is a `$ref` to its own definition. */
function isRecord(typ: Type): boolean {
  return (
    typ.kind === ValueKind.OBJECT &&
    typ.variants === undefined &&
    typ.fields !== undefined &&
    typ.fields.size > 0
  );
}

/**
 * Return the object schema of the record `typ` at `position`.
 *
 * Each field is a property, required iff named in `required` and not nullable; any other
 * field also accepts null and defaults to it. A field holding a record carries no `title`,
 * since its definition has one.
 *
 * @param position The record's position name, which is its title and prefixes the records
 *   inside it; `""` is the schema root, titled `Record`.
 * @param closed Whether the record and the records inside it reject unknown fields.
 */
function recordSchema(typ: Type, defs: Defs, position: string, closed: boolean): JsonObject {
  const properties: JsonObject = {};
  const required: string[] = [];
  for (const [name, sub] of typ.fields ?? []) {
    let schema = schemaFor(sub, defs, nested(position, name), closed);
    if (typ.required?.has(name) && !sub.nullable) {
      required.push(name);
    } else {
      schema = { ...orNull(schema), default: null };
    }
    setField(properties, name, titled(schema, name, sub));
  }
  const schema: JsonObject = { properties, title: position || "Record", type: "object" };
  if (required.length > 0) schema.required = required;
  if (closed) schema.additionalProperties = false;
  return schema;
}

/**
 * Add the record `schema` at `position` to `defs` and return a `$ref` to it. An identical
 * definition that wants the same key is shared.
 */
function define(defs: Defs, position: string, schema: JsonObject): JsonObject {
  const key = (position || "Record").replace(/[^A-Za-z0-9_-]/gu, "_");
  // Key order doesn't matter (Python compares dicts); the text sorts keys.
  const text = logJson(schema);
  let index = defs.findIndex((definition) => definition.key === key && definition.text === text);
  if (index < 0) index = defs.push({ key, schema, text, refs: [] }) - 1;
  // The index stands in for the key until `definitions` knows every clash; it also keeps the
  // text of records referring to different definitions apart.
  const ref: JsonObject = { $ref: `#/$defs/${index}` };
  defs[index]!.refs.push(ref);
  return ref;
}

/**
 * Return the `$defs` map of `defs`, pointing every `$ref` at its definition's final key.
 *
 * A key wanted by one definition is used as is; definitions sharing a key get `<key>__1`,
 * `<key>__2`, ... in order, skipping any key some definition wants or already got.
 */
function definitions(defs: Defs): JsonObject {
  const counts = new Map<string, number>();
  for (const { key } of defs) counts.set(key, (counts.get(key) ?? 0) + 1);
  const taken = new Set(counts.keys());
  const suffixes = new Map<string, number>();
  const $defs: JsonObject = {};
  for (const definition of defs) {
    let key = definition.key;
    if (counts.get(key)! > 1) {
      let suffix = suffixes.get(definition.key) ?? 0;
      do {
        suffix += 1;
        key = `${definition.key}__${suffix}`;
      } while (taken.has(key));
      suffixes.set(definition.key, suffix);
      taken.add(key);
    }
    for (const ref of definition.refs) ref.$ref = `#/$defs/${key}`;
    setField($defs, key, definition.schema);
  }
  return $defs;
}

/** Return the property schema of field `name` of type `sub`: `schema` with the field's title. */
function titled(schema: JsonObject, name: string, sub: Type): JsonObject {
  return isRecord(sub) ? schema : { ...schema, title: fieldTitle(name) };
}

/** Return a root schema wrapping `typ` as its one required property `field`. */
function wrapper(title: string, field: string, typ: Type, defs: Defs): JsonObject {
  const properties: JsonObject = {};
  setField(properties, field, titled(schemaFor(typ, defs, "", false), field, typ));
  return { properties, required: [field], title, type: "object" };
}

/**
 * Return the JSON Schema of a value of `typ`, or `null` when the agent should stay a text
 * producer (an untagged `str`, nullable or not).
 *
 * A non-nullable record (`OBJECT` with `fields`) maps directly to an object schema titled
 * `Record`. A scalar, a union, a list, or a nullable record can't be a standalone schema, so it
 * is wrapped in a single required property: `items` for a list (`ListWrapper`), else `value`
 * (`ScalarWrapper`). Nested records are collected under `$defs`.
 *
 * @throws RangeError when `typ` holds a kind with no structured mapping (`file`, `None`) or a
 *   scalar with malformed bounds.
 */
export function typeToSchema(typ: Type): JsonObject | null {
  if (typ.kind === ValueKind.STRING && (typ.tags === undefined || typ.tags.size === 0)) {
    return null; // unconstrained text stays on the ordinary generation path
  }
  const defs: Defs = [];
  let schema: JsonObject;
  if (isRecord(typ) && !typ.nullable) {
    schema = recordSchema(typ, defs, "", false);
  } else if (isListKind(typ.kind)) {
    schema = wrapper("ListWrapper", "items", typ, defs);
  } else {
    schema = wrapper("ScalarWrapper", "value", typ, defs);
  }
  if (defs.length > 0) schema.$defs = definitions(defs);
  return sortKeys(schema);
}

/**
 * Return a copy of `schema` with its keys in code-unit order at every level but a `properties`
 * map, which keeps its order. Entries are written with `setField`, so a `__proto__` name stays
 * data.
 */
function sortKeys(schema: JsonObject): JsonObject {
  const sorted: JsonObject = {};
  for (const key of Object.keys(schema).sort(compareCodeUnits)) {
    const value = schema[key]!;
    if (key === "properties") {
      setField(sorted, key, sortSchemas(value as JsonObject, Object.keys(value as JsonObject)));
    } else if (key === "$defs") {
      // Its keys are names, not keywords: a definition named `properties` is a schema.
      const names = Object.keys(value as JsonObject).sort(compareCodeUnits);
      setField(sorted, key, sortSchemas(value as JsonObject, names));
    } else {
      setField(sorted, key, sortValue(value));
    }
  }
  return sorted;
}

/** Return a copy of the map `schemas` (name to schema) with its `names` in order, each sorted. */
function sortSchemas(schemas: JsonObject, names: string[]): JsonObject {
  const sorted: JsonObject = {};
  for (const name of names) setField(sorted, name, sortKeys(schemas[name] as JsonObject));
  return sorted;
}

/** Return `value` with every schema inside it key-sorted by `sortKeys`. */
function sortValue(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortValue);
  if (typeof value === "object" && value !== null) return sortKeys(value);
  return value;
}

// --------------------------------------------------------------------------------------- //
// Field titles: pydantic's `name.title().replace("_", " ").strip()`
// --------------------------------------------------------------------------------------- //

const CASED = /\p{Cased}/u;

// The titlecase letters (`\p{Lt}`), keyed by their lowercase form: a digraph or a Greek letter
// with ypogegrammeni title-cases to one of these, not to its uppercase.
const TITLE_LETTERS: ReadonlyMap<string, string> = new Map(
  [..."ǅǈǋǲᾈᾉᾊᾋᾌᾍᾎᾏᾘᾙᾚᾛᾜᾝᾞᾟᾨᾩᾪᾫᾬᾭᾮᾯᾼῌῼ"].map((letter) => [letter.toLowerCase(), letter]),
);

// Georgian Mkhedruli letters are their own title case, though they upper-case to Mtavruli.
const GEORGIAN_LOWER = /^[\u10D0-\u10FA\u10FD-\u10FF]$/u;

// The characters Python's `str.strip()` removes (`str.isspace()`).
const PY_SPACE =
  "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");

/**
 * Return the title of field `name`, as pydantic derives it: Python's `str.title()`, then `_` to
 * a space, then Python's `strip()`. Example: `lesson_id` gives `Lesson Id`, `_x` gives `X`.
 */
function fieldTitle(name: string): string {
  let title = "";
  let previousCased = false;
  let index = 0;
  for (const ch of name) {
    title += previousCased ? lowerCase(ch, name, index) : titleCase(ch);
    previousCased = CASED.test(ch);
    index += ch.length;
  }
  return title.replaceAll("_", " ").replace(PY_STRIP, "");
}

/** Return the full title case of the character `ch`, as Python's `str.title()` maps it. */
function titleCase(ch: string): string {
  const letter = TITLE_LETTERS.get(ch.toLowerCase());
  if (letter !== undefined) return letter;
  if (GEORGIAN_LOWER.test(ch)) return ch;
  const upper = ch.toUpperCase();
  // A Greek letter with ypogegrammeni keeps it in title case; uppercase spells it as Ι.
  if (ch !== "\u0345" && ch.normalize("NFD").endsWith("\u0345")) {
    return `${upper.slice(0, -1)}\u0345`;
  }
  // A multi-character uppercase (`ß` -> `SS`, `ŉ` -> `ʼN`) stays upper only up to its first
  // cased character.
  const parts = /^(\P{Cased}*\p{Cased})(.*)$/su.exec(upper);
  return parts === null ? upper : parts[1]! + parts[2]!.toLowerCase();
}

/** Return the full lower case of `ch` at `index` of `text`, with Python's final-sigma rule. */
function lowerCase(ch: string, text: string, index: number): string {
  if (ch !== "Σ") return ch.toLowerCase();
  // Python skips case-ignorable characters on both sides, then asks whether a cased one
  // precedes and none follows.
  const before = text.slice(0, index).replace(/\p{Case_Ignorable}*$/u, "");
  const after = text.slice(index + 1).replace(/^\p{Case_Ignorable}*/u, "");
  return /\p{Cased}$/u.test(before) && !/^\p{Cased}/u.test(after) ? "ς" : "σ";
}
