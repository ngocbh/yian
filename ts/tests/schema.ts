/**
 * A small JSON Schema 2020-12 validator for the schemas in `conformance/schema/`: it implements
 * only the keywords those schemas use, and `lintSchema` keeps a schema to them.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

/** A JSON Schema subschema, as parsed. */
export type Schema = Record<string, unknown>;

/** How `validate` reads a schema. */
export interface ValidateOptions {
  /** Refuse any key a subschema's `properties` doesn't list, as if it had `additionalProperties: false`. */
  closed?: boolean;
}

/** The keywords the validator implements. */
export const KEYWORDS = new Set([
  "$ref",
  "type",
  "const",
  "enum",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "oneOf",
  "anyOf",
  "allOf",
  "if",
  "then",
  "minimum",
  "minItems",
  "maxItems",
  "minLength",
]);
/** Annotations the validator ignores. */
export const ANNOTATIONS = new Set(["$schema", "title", "description", "$defs"]);
export const TYPES = new Set(["null", "boolean", "integer", "number", "string", "array", "object"]);

/** Return the schema `conformance/schema/<name>`, parsed. */
export function loadSchema(name: string): Schema {
  const path = fileURLToPath(new URL(`../../conformance/schema/${name}`, import.meta.url));
  return JSON.parse(readFileSync(path, "utf8")) as Schema;
}

/** Return the JSON Schema type names `value` has (an integer, `bigint` included, is also a number). */
export function jsonTypes(value: unknown): string[] {
  if (value === null) return ["null"];
  if (Array.isArray(value)) return ["array"];
  if (typeof value === "bigint") return ["integer", "number"];
  if (typeof value === "number") return Number.isInteger(value) ? ["integer", "number"] : ["number"];
  return [typeof value];
}

/** Return `value` as problem text, with any `bigint` in it written as digits. */
function show(value: unknown): string {
  // JSON.stringify throws on a bigint, even one nested inside the value.
  return JSON.stringify(value, (_, v) => (typeof v === "bigint" ? String(v) : v));
}

/** Return the `root.$defs` entry a `#/$defs/<name>` reference names, or `undefined`. */
export function resolveRef(root: Schema, ref: unknown): Schema | undefined {
  if (typeof ref !== "string" || !ref.startsWith("#/$defs/")) return undefined;
  const defs = (root.$defs ?? {}) as Record<string, Schema>;
  const name = ref.slice("#/$defs/".length);
  return Object.hasOwn(defs, name) ? defs[name] : undefined;
}

/**
 * Validate `value` against `schema`, a subschema of `root`, for the keywords in `KEYWORDS`,
 * returning `path: problem` strings. `additionalProperties` supports only `false`.
 */
export function validate(
  root: Schema,
  schema: Schema,
  value: unknown,
  options: ValidateOptions = {},
  path = "$",
): string[] {
  const problems: string[] = [];
  const types = jsonTypes(value);
  if ("$ref" in schema) problems.push(...validate(root, resolveRef(root, schema.$ref)!, value, options, path));
  if ("type" in schema) {
    const allowed = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((type) => allowed.includes(type))) {
      return [...problems, `${path}: ${types[0]} is not ${allowed.join(" or ")}`];
    }
  }
  if ("const" in schema && !isDeepStrictEqual(value, schema.const)) {
    problems.push(`${path}: ${show(value)} is not ${show(schema.const)}`);
  }
  if ("enum" in schema) {
    const members = schema.enum as unknown[];
    if (!members.some((member) => isDeepStrictEqual(value, member))) {
      problems.push(`${path}: ${show(value)} is not one of the enum`);
    }
  }
  if (
    "minimum" in schema &&
    (typeof value === "number" || typeof value === "bigint") &&
    value < (schema.minimum as number)
  ) {
    problems.push(`${path}: ${value} is below ${schema.minimum}`);
  }
  // JSON Schema counts a string's length in code points.
  if (
    "minLength" in schema &&
    typeof value === "string" &&
    [...value].length < (schema.minLength as number)
  ) {
    problems.push(`${path}: shorter than ${schema.minLength} characters`);
  }
  if (types[0] === "object") {
    const object = value as Record<string, unknown>;
    for (const key of (schema.required as string[] | undefined) ?? []) {
      if (!Object.hasOwn(object, key)) problems.push(`${path}: missing ${key}`);
    }
    const properties = (schema.properties ?? {}) as Record<string, Schema>;
    const closed = schema.additionalProperties === false || (options.closed === true && "properties" in schema);
    for (const [key, inner] of Object.entries(object)) {
      if (Object.hasOwn(properties, key)) {
        problems.push(...validate(root, properties[key]!, inner, options, `${path}.${key}`));
      } else if (closed) {
        problems.push(`${path}: undeclared key ${key}`);
      }
    }
  }
  if (types[0] === "array") {
    const items = value as unknown[];
    if ("minItems" in schema && items.length < (schema.minItems as number)) {
      problems.push(`${path}: fewer than ${schema.minItems} items`);
    }
    if ("maxItems" in schema && items.length > (schema.maxItems as number)) {
      problems.push(`${path}: more than ${schema.maxItems} items`);
    }
    if ("items" in schema) {
      items.forEach((item, index) => {
        const at = `${path}[${index}]`;
        problems.push(...validate(root, schema.items as Schema, item, options, at));
      });
    }
  }
  if ("oneOf" in schema) {
    const passing = (schema.oneOf as Schema[]).filter(
      (branch) => validate(root, branch, value, options, path).length === 0,
    ).length;
    if (passing !== 1) problems.push(`${path}: ${passing} oneOf branches match, not 1`);
  }
  if ("anyOf" in schema) {
    const branches = schema.anyOf as Schema[];
    if (!branches.some((branch) => validate(root, branch, value, options, path).length === 0)) {
      problems.push(`${path}: no anyOf branch matches`);
    }
  }
  for (const branch of (schema.allOf as Schema[] | undefined) ?? []) {
    problems.push(...validate(root, branch, value, options, path));
  }
  if (
    "if" in schema &&
    "then" in schema &&
    validate(root, schema.if as Schema, value, options, path).length === 0
  ) {
    problems.push(...validate(root, schema.then as Schema, value, options, path));
  }
  return problems;
}

/**
 * Return `path: problem` strings for keywords outside the validator's list, an
 * `additionalProperties` other than `false`, a length bound that isn't a non-negative integer,
 * and `$ref`s that don't resolve in `root`.
 */
export function lintSchema(root: Schema, schema: unknown, path = "$"): string[] {
  if (schema === null || typeof schema !== "object" || Array.isArray(schema)) {
    return [`${path}: not a schema object`];
  }
  const problems: string[] = [];
  for (const [key, inner] of Object.entries(schema)) {
    const at = `${path}.${key}`;
    if (!KEYWORDS.has(key) && !ANNOTATIONS.has(key)) problems.push(`${at}: unknown keyword`);
    if (key === "$ref" && resolveRef(root, inner) === undefined) problems.push(`${at}: unresolved`);
    if (key === "type") {
      const types = Array.isArray(inner) ? inner : [inner];
      for (const type of types) if (!TYPES.has(type)) problems.push(`${at}: unknown type`);
    }
    if (key === "additionalProperties" && inner !== false) problems.push(`${at}: only false is supported`);
    const bound = ["minItems", "maxItems", "minLength"].includes(key);
    if (bound && !(Number.isInteger(inner) && (inner as number) >= 0)) {
      problems.push(`${at}: not a non-negative integer`);
    }
    if (["items", "if", "then"].includes(key)) problems.push(...lintSchema(root, inner, at));
    if (key === "oneOf" || key === "anyOf" || key === "allOf") {
      (inner as unknown[]).forEach((branch, index) => {
        problems.push(...lintSchema(root, branch, `${at}[${index}]`));
      });
    }
    if (key === "properties" || key === "$defs") {
      for (const [name, sub] of Object.entries(inner as Record<string, unknown>)) {
        problems.push(...lintSchema(root, sub, `${at}.${name}`));
      }
    }
  }
  return problems;
}
