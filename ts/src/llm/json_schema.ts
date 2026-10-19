/**
 * Pure JSON Schema transforms for structured output.
 *
 * Each function reads a schema as `typeToSchema` emits it (or one written by hand) and returns
 * what a route sends: whether the schema has an open part, the forced tool LangChain builds from
 * it, or the schema rewritten for a provider's subset. None changes its input. Ints in a schema
 * are `bigint`s and floats numbers, so numbers are compared as JSON values, never across kinds.
 */

import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { copyJson, isPlainObject, put, ToolSpec } from "./messages.ts";

/**
 * Return whether a JSON schema has an open part: an empty `{}` subschema (any value) or
 * `additionalProperties: true`, anywhere, `$defs` included.
 *
 * These are the shapes `typeToSchema` emits for an `object`, `Any` or elementless-list output;
 * strict structured-output modes reject them. It reads the root and the subschemas under `items`,
 * `additionalProperties`, the `properties` and `$defs` maps, and the `anyOf`, `oneOf` and `allOf`
 * lists; other keywords (`prefixItems`, `not`) are not inspected.
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
 * `prefixItems` member is kept.
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
 * `items`, `prefixItems`, `additionalProperties`, `anyOf`, `oneOf` and `allOf`.
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
    for (const key of ["properties", "$defs"]) {
      const map = node[key];
      if (isPlainObject(map)) stack.push(...Object.values(map));
    }
    for (const key of ["items", "additionalProperties"]) {
      const sub = node[key];
      if (isPlainObject(sub)) stack.push(sub);
    }
    for (const key of ["prefixItems", "anyOf", "oneOf", "allOf"]) {
      const list = node[key];
      if (Array.isArray(list)) stack.push(...list);
    }
  }
  return result;
}
