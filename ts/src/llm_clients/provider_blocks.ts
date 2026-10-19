/**
 * Provider blocks of `llm_config`: request-body fields checked against the API yian calls.
 *
 * Each block mirrors the request body of one provider API with the same names, nesting and
 * allowed values. A block is described by plain field specs (`Field`, `ObjectSchema`), and
 * the value pass checks a value against them the way Python yian's pydantic models do in
 * strict mode: no coercion (`"5"` is not an integer), no non-finite numbers, unknown keys
 * refused, unset (`null`) fields left out of the result at every level, and errors that
 * never echo values. Issues carry pydantic's `loc`, `type` and `msg`.
 */

import { compareCodeUnits } from "../diagnostics.ts";
import { isList, isRecord } from "../typesys/index.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";

/** A path into a value: keys, list indices and pydantic's union labels and variant tags. */
type Loc = readonly (string | number)[];

/** What a value of one block field must be. */
export type Field =
  | { readonly kind: "str"; readonly minLength?: number; readonly maxLength?: number; readonly pattern?: string }
  | { readonly kind: "int"; readonly ge?: bigint; readonly le?: bigint; readonly gt?: bigint; readonly lt?: bigint }
  | { readonly kind: "float"; readonly ge?: number; readonly le?: number; readonly gt?: number; readonly lt?: number }
  | { readonly kind: "bool" }
  | { readonly kind: "literal"; readonly values: readonly string[] }
  | { readonly kind: "list"; readonly item: Field; readonly maxLength?: number }
  | { readonly kind: "map"; readonly key: Field; readonly value: Field; readonly maxLength?: number }
  | { readonly kind: "object"; readonly schema: ObjectSchema }
  | { readonly kind: "union"; readonly members: readonly Field[] }
  /** Objects told apart by their `type` key, which names the variant. */
  | { readonly kind: "tagged"; readonly variants: Readonly<Record<string, ObjectSchema>> }
  /** Any JSON value (Python's `Dict[str, Any]` values). */
  | { readonly kind: "json" };

/** The fields of an object, in declaration order; the ones not in `required` are optional. */
export interface ObjectSchema {
  readonly fields: Readonly<Record<string, Field>>;
  readonly required: readonly string[];
}

/** One problem the value pass found, shaped like a pydantic error. */
export interface Issue {
  readonly loc: Loc;
  readonly type: string;
  readonly msg: string;
  /** For a `value_error`: the validator's own message, without "Value error, " (Python's `ctx["error"]`). */
  readonly error?: string;
}

/** A value a block refuses; the message keeps pydantic's layout without the help URL and type. */
export class ValidationError extends RangeError {
  override name = "ValidationError";
  /** What was validated, e.g. the block's Python class name. */
  readonly title: string;
  readonly issues: readonly Issue[];

  constructor(title: string, issues: readonly Issue[]) {
    const lines = [`${issues.length} validation error${issues.length === 1 ? "" : "s"} for ${title}`];
    for (const { loc, msg } of issues) {
      if (loc.length > 0) lines.push(loc.join("."));
      lines.push(`  ${msg}`);
    }
    super(lines.join("\n"));
    this.title = title;
    this.issues = Object.freeze(issues.map((issue) => Object.freeze({ ...issue, loc: Object.freeze([...issue.loc]) })));
  }
}

/** A checked value (meaningless when there are issues) and the issues found. */
interface Checked {
  value: JsonValue;
  issues: Issue[];
}

type Options<K extends Field["kind"]> = Omit<Extract<Field, { kind: K }>, "kind">;

/** Return `obj` as a frozen null-prototype object, so no inherited key reads as an entry. */
function table<T>(obj: Record<string, T>): Readonly<Record<string, T>> {
  return Object.freeze(Object.assign(Object.create(null) as Record<string, T>, obj));
}

const str = (options: Options<"str"> = {}): Field => Object.freeze({ kind: "str", ...options });
const int = (options: Options<"int"> = {}): Field => Object.freeze({ kind: "int", ...options });
const float = (options: Options<"float"> = {}): Field => Object.freeze({ kind: "float", ...options });
const bool = (): Field => Object.freeze({ kind: "bool" });
const literal = (...values: string[]): Field => Object.freeze({ kind: "literal", values: Object.freeze(values) });
const list = (item: Field, options: Omit<Options<"list">, "item"> = {}): Field =>
  Object.freeze({ kind: "list", item, ...options });
const map = (key: Field, value: Field, options: Omit<Options<"map">, "key" | "value"> = {}): Field =>
  Object.freeze({ kind: "map", key, value, ...options });
const schema = (fields: Record<string, Field>, required: readonly string[] = []): ObjectSchema =>
  Object.freeze({ fields: table(fields), required: Object.freeze([...required]) });
const object = (of: ObjectSchema): Field => Object.freeze({ kind: "object", schema: of });
const union = (...members: Field[]): Field => Object.freeze({ kind: "union", members: Object.freeze(members) });
const tagged = (variants: Record<string, ObjectSchema>): Field =>
  Object.freeze({ kind: "tagged", variants: table(variants) });
const json = (): Field => Object.freeze({ kind: "json" });

// A JSON array or record is checked as a list or map of JSON values.
const JSON_LIST: Extract<Field, { kind: "list" }> = Object.freeze({ kind: "list", item: json() });
const JSON_MAP: Extract<Field, { kind: "map" }> = Object.freeze({ kind: "map", key: str(), value: json() });

const ok = (value: JsonValue): Checked => ({ value, issues: [] });
const fail = (loc: Loc, type: string, msg: string): Checked => ({ value: null, issues: [{ loc, type, msg }] });

/** The value pass of one object: pydantic's model validation and dump, without the key walk. */
export function checkObject(schema: ObjectSchema, data: unknown): { value: JsonObject | null; issues: Issue[] } {
  const { value, issues } = checkFields(schema, data, []);
  return issues.length > 0 ? { value: null, issues } : { value: value as JsonObject, issues };
}

/** Check `value` at `loc` against `field`, returning a fresh copy of it. */
function checkField(field: Field, value: unknown, loc: Loc): Checked {
  switch (field.kind) {
    case "str":
      return checkStr(field, value, loc);
    case "int":
      if (typeof value === "bigint") return checkBounds(field, value, loc);
      return fail(loc, "int_type", "Input should be a valid integer");
    case "float":
      if (typeof value === "number") {
        if (Number.isFinite(value)) return checkBounds(field, value, loc);
        return fail(loc, "finite_number", "Input should be a finite number");
      }
      // A whole number in a float slot stays an int, unless a double can't hold it.
      if (typeof value === "bigint" && Number.isFinite(Number(value))) return checkBounds(field, value, loc);
      return fail(loc, "float_type", "Input should be a valid number");
    case "bool":
      return typeof value === "boolean" ? ok(value) : fail(loc, "bool_type", "Input should be a valid boolean");
    case "literal":
      return typeof value === "string" && field.values.includes(value)
        ? ok(value)
        : fail(loc, "literal_error", `Input should be ${choices(field.values)}`);
    case "list":
      return checkList(field, value, loc);
    case "map":
      return checkMap(field, value, loc);
    case "object":
      return checkFields(field.schema, value, loc);
    case "union":
      return checkUnion(field.members, value, loc);
    case "tagged":
      return checkTagged(field.variants, value, loc);
    case "json":
      return checkJson(value, loc);
  }
}

/** Check a string's length in code points, then its pattern; only the first failure counts. */
function checkStr(field: Extract<Field, { kind: "str" }>, value: unknown, loc: Loc): Checked {
  if (typeof value !== "string") return fail(loc, "string_type", "Input should be a valid string");
  const length = [...value].length;
  if (field.minLength !== undefined && length < field.minLength) {
    return fail(loc, "string_too_short", `String should have at least ${field.minLength} characters`);
  }
  if (field.maxLength !== undefined && length > field.maxLength) {
    return fail(loc, "string_too_long", `String should have at most ${field.maxLength} characters`);
  }
  if (field.pattern !== undefined && !new RegExp(field.pattern, "u").test(value)) {
    return fail(loc, "string_pattern_mismatch", `String should match pattern '${field.pattern}'`);
  }
  return ok(value);
}

/** Check a number against its field's bounds; only the first failure counts. */
function checkBounds(bounds: Options<"int"> | Options<"float">, value: bigint | number, loc: Loc): Checked {
  const { ge, gt, le, lt } = bounds;
  if (ge !== undefined && value < ge) {
    return fail(loc, "greater_than_equal", `Input should be greater than or equal to ${String(ge)}`);
  }
  if (gt !== undefined && value <= gt) return fail(loc, "greater_than", `Input should be greater than ${String(gt)}`);
  if (le !== undefined && value > le) {
    return fail(loc, "less_than_equal", `Input should be less than or equal to ${String(le)}`);
  }
  if (lt !== undefined && value >= lt) return fail(loc, "less_than", `Input should be less than ${String(lt)}`);
  return ok(value);
}

/** Return pydantic's list of literal choices: `'a', 'b' or 'c'`. */
function choices(values: readonly string[]): string {
  const quoted = values.map((v) => `'${v}'`);
  return quoted.length > 1 ? `${quoted.slice(0, -1).join(", ")} or ${quoted.at(-1)!}` : quoted.join("");
}

/** Check a list's length, then each item at its index. */
function checkList(field: Extract<Field, { kind: "list" }>, value: unknown, loc: Loc): Checked {
  if (!isList(value)) return fail(loc, "list_type", "Input should be a valid list");
  // pydantic refuses an over-long list before looking at its items.
  if (field.maxLength !== undefined && value.length > field.maxLength) {
    const msg = `List should have at most ${field.maxLength} items after validation, not ${value.length}`;
    return fail(loc, "too_long", msg);
  }
  const items: JsonValue[] = [];
  const issues: Issue[] = [];
  for (let i = 0; i < value.length; i++) {
    const checked = checkField(field.item, value[i], [...loc, i]);
    items.push(checked.value);
    issues.push(...checked.issues);
  }
  return { value: items, issues };
}

/** Check each entry's key and value, then the mapping's length. */
function checkMap(field: Extract<Field, { kind: "map" }>, value: unknown, loc: Loc): Checked {
  if (!isRecord(value)) return fail(loc, "dict_type", "Input should be a valid dictionary");
  const entries: [string, JsonValue][] = [];
  const issues: Issue[] = [];
  for (const [key, item] of Object.entries(value)) {
    if (item === undefined) continue;
    issues.push(...checkField(field.key, key, [...loc, key, "[key]"]).issues);
    const checked = checkField(field.value, item, [...loc, key]);
    issues.push(...checked.issues);
    entries.push([key, checked.value]);
  }
  if (issues.length > 0) return { value: null, issues };
  // pydantic checks a mapping's length only once its keys and values pass.
  if (field.maxLength !== undefined && entries.length > field.maxLength) {
    const msg = `Dictionary should have at most ${field.maxLength} items after validation, not ${entries.length}`;
    return fail(loc, "too_long", msg);
  }
  return ok(Object.fromEntries(entries));
}

/** Check an object's fields in declaration order, then refuse its other keys in written order. */
function checkFields(schema: ObjectSchema, value: unknown, loc: Loc): Checked {
  if (!isRecord(value)) return fail(loc, "model_type", "Input should be a valid dictionary");
  const entries: [string, JsonValue][] = [];
  const issues: Issue[] = [];
  for (const [name, field] of Object.entries(schema.fields)) {
    const given = Object.hasOwn(value, name) ? value[name] : undefined;
    const required = schema.required.includes(name);
    if (given === undefined) {
      if (required) issues.push({ loc: [...loc, name], type: "missing", msg: "Field required" });
      continue;
    }
    if (given === null && !required) continue;
    const checked = checkField(field, given, [...loc, name]);
    issues.push(...checked.issues);
    entries.push([name, checked.value]);
  }
  for (const [key, item] of Object.entries(value)) {
    if (item !== undefined && !Object.hasOwn(schema.fields, key)) {
      issues.push({ loc: [...loc, key], type: "extra_forbidden", msg: "Extra inputs are not permitted" });
    }
  }
  return { value: Object.fromEntries(entries), issues };
}

/** Return the first member that takes `value`, else every member's issues under its label. */
function checkUnion(members: readonly Field[], value: unknown, loc: Loc): Checked {
  const issues: Issue[] = [];
  for (const member of members) {
    const checked = checkField(member, value, [...loc, label(member)]);
    if (checked.issues.length === 0) return checked;
    issues.push(...checked.issues);
  }
  return { value: null, issues };
}

/** Return pydantic's label for a union member, e.g. `list[str]` or `literal['a','b']`. */
function label(field: Field): string {
  switch (field.kind) {
    case "list":
      return `list[${label(field.item)}]`;
    case "literal":
      return `literal[${field.values.map((v) => `'${v}'`).join(",")}]`;
    default:
      return field.kind;
  }
}

/** Check a record against the variant its `type` names, under that tag. */
function checkTagged(variants: Readonly<Record<string, ObjectSchema>>, value: unknown, loc: Loc): Checked {
  const tag = isRecord(value) && Object.hasOwn(value, "type") ? value["type"] : undefined;
  if (typeof tag !== "string" || !Object.hasOwn(variants, tag)) {
    // Python yian's own message: it lists the tags and never echoes the value.
    const tags = Object.keys(variants).sort(compareCodeUnits).join(", ");
    return fail(loc, "unknown_type", `\`type\` must be one of: ${tags}`);
  }
  return checkFields(variants[tag]!, value, [...loc, tag]);
}

/** Check a JSON value at any depth, refusing non-finite numbers and anything that isn't JSON. */
function checkJson(value: unknown, loc: Loc): Checked {
  if (value === null || typeof value === "boolean" || typeof value === "string" || typeof value === "bigint") {
    return ok(value);
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? ok(value) : fail(loc, "finite_number", "Input should be a finite number");
  }
  if (isList(value)) return checkList(JSON_LIST, value, loc);
  if (isRecord(value)) return checkMap(JSON_MAP, value, loc);
  return fail(loc, "json_type", "Input should be a JSON value");
}
