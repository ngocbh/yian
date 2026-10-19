/**
 * Pure JSON Schema transforms for structured output.
 *
 * Each function reads a schema as `typeToSchema` emits it (or one written by hand) and returns
 * what a route sends: whether the schema has an open part, the forced tool LangChain builds from
 * it, or the schema rewritten for a provider's subset. None changes its input. Ints in a schema
 * are `bigint`s and floats numbers, so numbers are compared as JSON values, never across kinds.
 * The transforms are ported from Python yian and the SDKs it calls (each function names its source).
 */

import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { encodeBody } from "./body.ts";
import { copyJson, isPlainObject, put, ToolSpec } from "./messages.ts";

/**
 * Return whether a JSON schema has an open part: an empty `{}` subschema (any value) or
 * `additionalProperties: true`, anywhere, `$defs` included.
 *
 * These are the shapes `typeToSchema` emits for an `object`, `Any` or elementless-list output;
 * strict structured-output modes reject them. It reads the root and the subschemas under `items`,
 * `additionalProperties`, the `properties` and `$defs` maps, and the `anyOf`, `oneOf` and `allOf`
 * lists; other keywords (`prefixItems`, `not`) are not inspected. Ported from yian's
 * `capabilities.has_open_schema`.
 *
 * @param schema The JSON schema; it is only read.
 * @returns `true` when it has an open part; `false` when it has none or is not an object.
 */
export function hasOpenSchema(schema: JsonValue): boolean {
  const stack: JsonValue[] = [schema];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (!isPlainObject(node)) continue;
    if (Object.keys(node).length === 0 || node["additionalProperties"] === true) return true;
    for (const key of ["items", "additionalProperties"]) {
      const value = node[key];
      if (isPlainObject(value)) stack.push(value);
    }
    // Walk the maps by value, so an empty `properties: {}` is not itself open.
    for (const key of ["properties", "$defs"]) {
      const value = node[key];
      if (isPlainObject(value)) stack.push(...Object.values(value));
    }
    for (const key of ["anyOf", "oneOf", "allOf"]) {
      const value = node[key];
      if (Array.isArray(value)) stack.push(...value);
    }
  }
  return false;
}

/**
 * Return the function tool LangChain's `convert_to_openai_tool` builds from a JSON schema.
 *
 * Every `$ref` is inlined (`$defs` values are copied, not walked), the root's `$defs` and
 * `definitions` are dropped, the root's `title` names the tool and its `description` describes it
 * (`""` when absent), and every `title` keyword is removed except a property named `title`. Like
 * LangChain, title removal walks objects only, so a title inside an `anyOf`, `oneOf`, `allOf` or
 * `prefixItems` member is kept. Ported from langchain-core's `utils/function_calling.py`
 * (`_convert_json_schema_to_openai_function`, `_rm_titles`) and `utils/json_schema.py`
 * (`dereference_refs`).
 *
 * @param schema The JSON schema; it is not changed.
 * @param strict The tool's `strict` flag, passed through.
 * @returns The tool.
 * @throws {TypeError} For a `$ref` that is not a `#` fragment, or one that names nothing in the schema.
 */
export function functionTool(schema: JsonObject, strict: boolean | null): ToolSpec {
  // `dereference` builds a new tree, so the deletes below never reach `schema`.
  const parameters = dereference(schema, schema, new Set()) as JsonObject;
  delete parameters["definitions"];
  delete parameters["$defs"];
  const name = parameters["title"] ?? "";
  delete parameters["title"];
  const description = parameters["description"] ?? "";
  delete parameters["description"];
  return new ToolSpec({
    name: name as string,
    description: description as string,
    parameters: removeTitles(parameters, ""),
    strict,
  });
}

/**
 * Return a copy of `value` with every `$ref` object replaced by its resolved definition, the
 * object's other keys merged over it. A ref already being resolved on the current path (a
 * cycle) gives only those other keys. A `$defs` value is copied as is.
 */
function dereference(value: JsonValue, root: JsonObject, active: Set<JsonValue>): JsonValue {
  if (Array.isArray(value)) return value.map((item) => dereference(item, root, active));
  if (!isPlainObject(value)) return value;
  if (!Object.hasOwn(value, "$ref")) return dereferenceMembers(value, root, active);
  const ref = value["$ref"]!;
  const siblings: JsonObject = {};
  for (const key of Object.keys(value)) if (key !== "$ref") put(siblings, key, value[key]!);
  if (active.has(ref)) return dereferenceMembers(siblings, root, active);
  active.add(ref);
  const resolved = dereference(resolveRef(ref, root), root, active);
  active.delete(ref);
  if (Object.keys(siblings).length === 0) return resolved;
  const merged: JsonObject = {};
  if (isPlainObject(resolved)) for (const key of Object.keys(resolved)) put(merged, key, resolved[key]!);
  const extra = dereferenceMembers(siblings, root, active);
  for (const key of Object.keys(extra)) put(merged, key, extra[key]!);
  return merged;
}

/** Return a copy of `object` with each member dereferenced, except `$defs`, which is copied as is. */
function dereferenceMembers(object: JsonObject, root: JsonObject, active: Set<JsonValue>): JsonObject {
  const result: JsonObject = {};
  for (const key of Object.keys(object)) {
    const value = object[key]!;
    put(result, key, key === "$defs" ? copyJson(value) : dereference(value, root, active));
  }
  return result;
}

/** Return what a `#/a/b` ref names in `root`; a digit component indexes an array. */
function resolveRef(ref: JsonValue, root: JsonObject): JsonValue {
  const components = typeof ref === "string" ? ref.split("/") : [];
  if (components[0] !== "#") {
    throw new TypeError("ref paths are expected to be URI fragments, meaning they should start with #.");
  }
  let node: JsonValue = root;
  for (const component of components.slice(1)) {
    if (isPlainObject(node) && Object.hasOwn(node, component)) {
      node = node[component]!;
    } else if (Array.isArray(node) && /^\d+$/.test(component) && Number(component) < node.length) {
      node = node[Number(component)]!;
    } else {
      throw new TypeError(`Reference '${String(ref)}' not found.`);
    }
  }
  return node;
}

/**
 * Return a copy of `object` without `title` keywords, keeping a property named `title` (an object
 * directly under `properties`). It walks object values only; arrays and scalars are kept as they are.
 */
function removeTitles(object: JsonObject, parentKey: string): JsonObject {
  const result: JsonObject = {};
  for (const key of Object.keys(object)) {
    const value = object[key]!;
    if (key === "title" && !(isPlainObject(value) && parentKey === "properties")) continue;
    put(result, key, isPlainObject(value) ? removeTitles(value, key) : value);
  }
  return result;
}

/**
 * Return a copy of a JSON schema with each string or number `const` written as a one-value `enum`.
 *
 * Gemini's `responseJsonSchema` lists `enum` (for strings and numbers) but not `const`, and
 * `{const: v}` and `{enum: [v]}` accept the same values. A boolean or `null` `const` is left as
 * written, and so is one whose existing `enum` lacks `v` (members compared as JSON values, an int
 * never equal to a float). It rewrites the root and the subschemas under `properties`, `$defs`,
 * `items`, `prefixItems`, `additionalProperties`, `anyOf`, `oneOf` and `allOf`. Ported from yian's
 * `google_client._const_as_enum`.
 *
 * @param schema The JSON schema; it is not changed.
 * @returns The rewritten copy.
 */
export function constAsEnum(schema: JsonObject): JsonObject {
  const result = copyJson(schema);
  const stack: JsonValue[] = [result];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (!isPlainObject(node)) continue;
    const value = node["const"];
    if (typeof value === "string" || typeof value === "bigint" || typeof value === "number") {
      // `logJson` writes an int and a float apart (`1` and `1.0`), as Python's `json.dumps` does.
      const wanted = logJson(value);
      const members = node["enum"];
      const matches = Array.isArray(members) && members.some((member) => logJson(member) === wanted);
      if (members === undefined || members === null || matches) {
        delete node["const"];
        node["enum"] = [value];
      }
    }
    stack.push(...childSchemas(node));
  }
  return result;
}

/**
 * Return the subschemas `constAsEnum` and `moveAzureKeywords` visit under `node`: the values of
 * `properties` and `$defs`, the `items` and `additionalProperties` objects, and the members of
 * `prefixItems`, `anyOf`, `oneOf` and `allOf`.
 */
function childSchemas(node: JsonObject): JsonValue[] {
  const children: JsonValue[] = [];
  for (const key of ["properties", "$defs"]) {
    const map = node[key];
    if (isPlainObject(map)) children.push(...Object.values(map));
  }
  for (const key of ["items", "additionalProperties"]) {
    const sub = node[key];
    if (isPlainObject(sub)) children.push(sub);
  }
  for (const key of ["prefixItems", "anyOf", "oneOf", "allOf"]) {
    const list = node[key];
    if (Array.isArray(list)) children.push(...list);
  }
  return children;
}

/**
 * Return a copy of a JSON schema in the form OpenAI's strict structured output accepts.
 *
 * Every `type: "object"` without `additionalProperties` gets `additionalProperties: false` (an
 * explicit `true` stays, so an open schema stays open); every `properties` map makes all its keys
 * `required`; a `default: null` is dropped; a one-entry `allOf` is merged into its node, the
 * entry's keys winning; a `$ref` with sibling keys is replaced by its definition with the
 * siblings kept over it. A bare `{$ref}` and the `$defs` stay. It visits the `$defs` and
 * `definitions` values, the `properties` values, an `items` object and the `anyOf` and `allOf`
 * members. Ported from the openai SDK's `lib/_pydantic.py` (`_ensure_strict_json_schema`).
 *
 * @param schema The JSON schema; it is not changed.
 * @returns The strict copy.
 * @throws {TypeError} For a non-object where a schema is expected, a mixed `$ref` that is not a
 *   `#/` path, or one that names no object in the schema.
 */
export function strictJsonSchema(schema: JsonObject): JsonObject {
  const result = copyJson(schema);
  return makeStrict(result, "#", result);
}

/** Make `node`, the subschema of `root` at `path`, strict in place, and return it. */
function makeStrict(node: JsonValue, path: string, root: JsonObject): JsonObject {
  const object = expectSchema(node, path);
  for (const key of ["$defs", "definitions"]) {
    const map = object[key];
    if (isPlainObject(map)) for (const name of Object.keys(map)) makeStrict(map[name]!, `${path}/${key}/${name}`, root);
  }
  if (object["type"] === "object" && !Object.hasOwn(object, "additionalProperties")) {
    object["additionalProperties"] = false;
  }
  const properties = object["properties"];
  if (isPlainObject(properties)) {
    object["required"] = Object.keys(properties);
    for (const key of Object.keys(properties)) makeStrict(properties[key]!, `${path}/properties/${key}`, root);
  }
  const items = object["items"];
  if (isPlainObject(items)) makeStrict(items, `${path}/items`, root);
  const anyOf = object["anyOf"];
  if (Array.isArray(anyOf)) anyOf.forEach((member, i) => makeStrict(member, `${path}/anyOf/${i}`, root));
  const allOf = object["allOf"];
  if (Array.isArray(allOf) && allOf.length === 1) {
    const entry = makeStrict(allOf[0]!, `${path}/allOf/0`, root);
    for (const key of Object.keys(entry)) put(object, key, entry[key]!);
    delete object["allOf"];
  } else if (Array.isArray(allOf)) {
    allOf.forEach((entry, i) => makeStrict(entry, `${path}/allOf/${i}`, root));
  }
  if (object["default"] === null) delete object["default"];
  if (Object.hasOwn(object, "$ref") && Object.keys(object).length > 1) {
    // Strict mode rejects a `$ref` with siblings, so the definition is inlined under them; the
    // inlined copy may not be strict yet, so the node is walked again.
    const definition = copyJson(strictRefTarget(object["$ref"]!, root));
    for (const key of Object.keys(definition)) if (!Object.hasOwn(object, key)) put(object, key, definition[key]!);
    delete object["$ref"];
    return makeStrict(object, path, root);
  }
  return object;
}

/** Return the object a `#/a/b` ref names in `root`. */
function strictRefTarget(ref: JsonValue, root: JsonObject): JsonObject {
  if (typeof ref !== "string" || !ref.startsWith("#/")) {
    throw new TypeError(`Unexpected $ref format ${encodeBody(ref)}; Does not start with #/`);
  }
  let node: JsonObject = root;
  for (const key of ref.slice(2).split("/")) {
    const next = Object.hasOwn(node, key) ? node[key] : undefined;
    if (!isPlainObject(next)) throw new TypeError(`Reference '${ref}' not found.`);
    node = next;
  }
  return node;
}

/**
 * Return `value` when it is an object, such as a schema or a `$defs` or `properties` map.
 *
 * @throws {TypeError} `expected a JSON Schema object at <path>` otherwise.
 */
function expectSchema(value: JsonValue, path: string): JsonObject {
  if (!isPlainObject(value)) throw new TypeError(`expected a JSON Schema object at ${path}`);
  return value;
}

/** Return whether `value` is falsy in Python: `null`, `false`, a zero, `""`, `[]` or `{}`. */
function isFalsy(value: JsonValue): boolean {
  if (Array.isArray(value)) return value.length === 0;
  if (isPlainObject(value)) return Object.keys(value).length === 0;
  return value === null || value === false || value === "" || value === 0 || value === 0n;
}

/** Return the `{key: value, …}` note for keywords moved into a description, values as JSON, in key order. */
function keywordNote(keywords: JsonObject): string {
  return `{${Object.keys(keywords)
    .map((key) => `${key}: ${encodeBody(keywords[key])}`)
    .join(", ")}}`;
}

/** The keywords Azure OpenAI's strict structured output rejects, in the order their note lists them. */
export const AZURE_STRICT_UNSUPPORTED: readonly string[] = [
  "minLength",
  "maxLength",
  "pattern",
  "format",
  "minimum",
  "maximum",
  "multipleOf",
  "patternProperties",
  "unevaluatedProperties",
  "propertyNames",
  "minProperties",
  "maxProperties",
  "unevaluatedItems",
  "contains",
  "minContains",
  "maxContains",
  "minItems",
  "maxItems",
  "uniqueItems",
];

/**
 * Return a copy of a JSON schema with the keywords Azure's strict mode rejects moved into
 * descriptions, or `null` when it has none.
 *
 * Each subschema loses the `AZURE_STRICT_UNSUPPORTED` keywords it holds and states them, in that
 * list's order, as a `{key: <JSON value>, …}` note: its `description` when that is absent, empty or
 * not a string, else after the description and a blank line. It visits the root and the subschemas
 * under `properties`, `$defs`, `items`, `prefixItems`, `additionalProperties`, `anyOf`, `oneOf` and
 * `allOf`, so a property named like a keyword is kept. Ported from yian's
 * `azure_client._strict_schema_without`.
 *
 * @param schema The JSON schema, usually `strictJsonSchema`'s result; it is not changed.
 * @returns The rewritten copy, or `null` when no subschema holds a listed keyword.
 */
export function moveAzureKeywords(schema: JsonObject): JsonObject | null {
  const result = copyJson(schema);
  let moved = false;
  const stack: JsonValue[] = [result];
  while (stack.length > 0) {
    const node = stack.pop()!;
    if (!isPlainObject(node)) continue;
    const removed: JsonObject = {};
    for (const key of AZURE_STRICT_UNSUPPORTED) {
      if (!Object.hasOwn(node, key)) continue;
      removed[key] = node[key]!;
      delete node[key];
    }
    if (Object.keys(removed).length > 0) {
      moved = true;
      const description = node["description"];
      const note = keywordNote(removed);
      node["description"] = typeof description === "string" && description !== "" ? `${description}\n\n${note}` : note;
    }
    stack.push(...childSchemas(node));
  }
  return moved ? result : null;
}

/** The string formats Anthropic's structured output accepts. */
const ANTHROPIC_STRING_FORMATS = new Set([
  "date-time",
  "time",
  "date",
  "duration",
  "email",
  "hostname",
  "uri",
  "ipv4",
  "ipv6",
  "uuid",
]);

/**
 * Return a JSON schema rebuilt in the subset Anthropic's structured output accepts.
 *
 * Keywords are kept in a fixed order: `$defs` (each definition rebuilt), then either a `$ref`
 * alone (its siblings dropped) or `anyOf` (from `anyOf`, else `oneOf`), `allOf` or `type`, then
 * an array `enum`, `description` and `title`. An `object` gets its rebuilt `properties` (`{}` when
 * absent), `additionalProperties: false` (an open object is closed) and its `required`; a
 * `string` keeps a supported `format` and drops a falsy one (as Python does); an `array` keeps
 * its rebuilt `items` and a `minItems` of 0 or 1. Every other keyword is stated in the description
 * as a `{key: <JSON value>, …}` note, in the schema's key order, after the description and a blank
 * line when there is a string one.
 * Ported from the anthropic SDK's `lib/_parse/_transform.py` (`transform_schema`).
 *
 * @param schema The JSON schema; it is not changed.
 * @returns The rebuilt schema.
 * @throws {TypeError} For a non-object where a schema is expected, a subschema with none of
 *   `type`, `anyOf`, `oneOf` and `allOf` (such as the `{}` of an `Any`), or a `type` that is not
 *   one of JSON Schema's type names.
 */
export function anthropicOutputSchema(schema: JsonObject): JsonObject {
  return anthropicSchema(schema, "#");
}

/** Return `anthropicOutputSchema` of `schema`, the subschema at `path`. */
function anthropicSchema(schema: JsonValue, path: string): JsonObject {
  const out: JsonObject = {};
  const src: JsonObject = { ...expectSchema(schema, path) };
  /** Remove `key` from `src` and return its value, `null` when absent (Python's `pop(key, None)`). */
  const take = (key: string): JsonValue => {
    const value = src[key] ?? null;
    delete src[key];
    return value;
  };
  const defs = take("$defs");
  if (defs !== null) {
    const map = expectSchema(defs, `${path}/$defs`);
    const result: JsonObject = {};
    for (const name of Object.keys(map)) put(result, name, anthropicSchema(map[name]!, `${path}/$defs/${name}`));
    out["$defs"] = result;
  }
  const ref = take("$ref");
  if (ref !== null) {
    out["$ref"] = ref;
    return out;
  }
  const type = take("type");
  const anyOf = take("anyOf");
  const oneOf = take("oneOf");
  const allOf = take("allOf");
  const members = (list: JsonValue[], key: string) => list.map((m, i) => anthropicSchema(m, `${path}/${key}/${i}`));
  if (Array.isArray(anyOf)) out["anyOf"] = members(anyOf, "anyOf");
  else if (Array.isArray(oneOf)) out["anyOf"] = members(oneOf, "oneOf");
  else if (Array.isArray(allOf)) out["allOf"] = members(allOf, "allOf");
  else if (type === null) throw new TypeError("Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field.");
  else out["type"] = type;
  const enumValues = take("enum");
  if (Array.isArray(enumValues)) out["enum"] = enumValues;
  const description = take("description");
  if (description !== null) out["description"] = description;
  const title = take("title");
  if (title !== null) out["title"] = title;
  switch (type) {
    case "object": {
      const properties = expectSchema(take("properties") ?? {}, `${path}/properties`);
      const result: JsonObject = {};
      for (const key of Object.keys(properties)) {
        put(result, key, anthropicSchema(properties[key]!, `${path}/properties/${key}`));
      }
      out["properties"] = result;
      take("additionalProperties");
      out["additionalProperties"] = false;
      const required = take("required");
      if (required !== null) out["required"] = required;
      break;
    }
    case "string": {
      // An unsupported format goes back into `src`, last, to be noted with the other keywords.
      const format = take("format");
      if (typeof format === "string" && ANTHROPIC_STRING_FORMATS.has(format)) out["format"] = format;
      else if (!isFalsy(format)) src["format"] = format;
      break;
    }
    case "array": {
      const items = take("items");
      if (items !== null) out["items"] = anthropicSchema(items, `${path}/items`);
      const minItems = take("minItems");
      const numeric = typeof minItems === "bigint" || typeof minItems === "number";
      if (numeric && (Number(minItems) === 0 || Number(minItems) === 1)) out["minItems"] = minItems;
      else if (minItems !== null) src["minItems"] = minItems;
      break;
    }
    case null:
    case "boolean":
    case "integer":
    case "number":
    case "null":
      break;
    default:
      throw new TypeError(`unsupported JSON Schema type ${encodeBody(type)}`);
  }
  if (Object.keys(src).length > 0) {
    const prefix = typeof description === "string" ? `${description}\n\n` : "";
    out["description"] = prefix + keywordNote(src);
  }
  return out;
}
