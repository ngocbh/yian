/**
 * Provider blocks of `llm_config`: request-body fields checked against the API yian calls.
 *
 * Each block mirrors the request body of one provider API with the same names, nesting and
 * allowed values. A block is described by plain field specs (`Field`, `ObjectSchema`), and
 * the value pass checks a value against them strictly: no coercion (`"5"` is not an integer), no non-finite numbers, unknown keys
 * refused, unset (`null`) fields left out of the result at every level, and errors that
 * never echo values. Each issue carries a location `loc`, an error `type` and a message `msg`. Before the value pass,
 * the key walk refuses the paths yian builds itself (`reserved`) and unknown keys, naming each.
 * `locatedBlockProblems` checks a block whose leaves may be values resolved later, and
 * `blockFieldAnnotation` gives the simple type a value for one field must have.
 */

import { compareCodeUnits } from "../diagnostics.ts";
import { isList, isRecord } from "../typesys/index.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { valueRepr } from "../typesys/values.ts";

/** A path into a value: keys, list indices, union labels and variant tags. */
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
  /** Any JSON value. */
  | { readonly kind: "json" };

/** The fields of an object, in declaration order; the ones not in `required` are optional. */
export interface ObjectSchema {
  readonly fields: Readonly<Record<string, Field>>;
  readonly required: readonly string[];
}

/** One problem the value pass found: where, its error type and its message. */
export interface Issue {
  readonly loc: Loc;
  readonly type: string;
  readonly msg: string;
  /** For a `value_error`: the validator's own message, without "Value error, ". */
  readonly error?: string;
}

/**
 * A value a block refuses. The message is a count header ("N validation errors for <title>"), then
 * each issue's dotted location, if any, and its indented message.
 */
export class ValidationError extends RangeError {
  override name = "ValidationError";
  /** What was validated, e.g. the block's schema name. */
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
function table<T>(obj: Readonly<Record<string, T>>): Readonly<Record<string, T>> {
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
const schema = (fields: Readonly<Record<string, Field>>, required: readonly string[] = []): ObjectSchema =>
  Object.freeze({ fields: table(fields), required: Object.freeze([...required]) });
const object = (of: ObjectSchema): Field => Object.freeze({ kind: "object", schema: of });
const union = (...members: Field[]): Field => Object.freeze({ kind: "union", members: Object.freeze(members) });
const tagged = (variants: Readonly<Record<string, ObjectSchema>>): Field =>
  Object.freeze({ kind: "tagged", variants: table(variants) });
const json = (): Field => Object.freeze({ kind: "json" });

// A JSON array or record is checked as a list or map of JSON values.
const JSON_LIST: Extract<Field, { kind: "list" }> = Object.freeze({ kind: "list", item: json() });
const JSON_MAP: Extract<Field, { kind: "map" }> = Object.freeze({ kind: "map", key: str(), value: json() });

const ok = (value: JsonValue): Checked => ({ value, issues: [] });
const fail = (loc: Loc, type: string, msg: string): Checked => ({ value: null, issues: [{ loc, type, msg }] });

/** The value pass of one object: validate it and drop unset fields, without the key walk. */
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

/** Return a list of literal choices spelled as JSON strings: `"a", "b" or "c"`. */
export function choices(values: readonly string[]): string {
  const quoted = values.map((v) => JSON.stringify(v));
  return quoted.length > 1 ? `${quoted.slice(0, -1).join(", ")} or ${quoted.at(-1)!}` : quoted.join("");
}

/** Check a list's length, then each item at its index. */
function checkList(field: Extract<Field, { kind: "list" }>, value: unknown, loc: Loc): Checked {
  if (!isList(value)) return fail(loc, "list_type", "Input should be a valid list");
  // An over-long list is refused before its items are checked.
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
  // A mapping's length is checked only once its keys and values pass.
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

/** Return the label for a union member, e.g. `list[str]` or `literal["a","b"]`. */
function label(field: Field): string {
  switch (field.kind) {
    case "list":
      return `list[${label(field.item)}]`;
    case "literal":
      return `literal[${field.values.map((v) => JSON.stringify(v)).join(",")}]`;
    default:
      return field.kind;
  }
}

/** Check a record against the variant its `type` names, under that tag. */
function checkTagged(variants: Readonly<Record<string, ObjectSchema>>, value: unknown, loc: Loc): Checked {
  const tag = isRecord(value) && Object.hasOwn(value, "type") ? value["type"] : undefined;
  if (typeof tag !== "string" || !Object.hasOwn(variants, tag)) {
    // The message lists the tags and never echoes the value.
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

// --------------------------------------------------------------------------------------- //
// Blocks and the key walk
// --------------------------------------------------------------------------------------- //

/** A top-level block: an object schema that names itself and refuses reserved and unknown keys. */
export interface BlockSchema extends ObjectSchema {
  /** The schema name, e.g. "OpenAIResponsesBlock", for the error header. */
  readonly title: string;
  /** The block name used in messages, e.g. "qwen-cn". */
  readonly name: string;
  /** How an unknown key is described, e.g. "is not a supported field of the X request". */
  readonly api: string;
  /** Dotted key path -> what yian builds there; setting one is an error. */
  readonly reserved: Readonly<Record<string, string>>;
}

/** Check `data` against a block: the key walk, then the value pass. */
export function checkBlock(schema: BlockSchema, data: unknown): { value: JsonObject | null; issues: Issue[] } {
  if (isRecord(data)) {
    const problems: string[] = [];
    keyProblems(schema, schema, data, "", problems);
    if (problems.length > 0) {
      const joined = problems.join("; ");
      return { value: null, issues: [{ loc: [], type: "value_error", msg: `Value error, ${joined}`, error: joined }] };
    }
  }
  return checkObject(schema, data);
}

/** Return `data` checked against a block, without unset fields; throws `ValidationError` on any issue. */
export function validateBlock(schema: BlockSchema, data: unknown): JsonObject {
  const { value, issues } = checkBlock(schema, data);
  if (issues.length > 0) throw new ValidationError(schema.title, issues);
  return value!;
}

/**
 * Append a message for each reserved or unknown key of `data`, recursing into parts.
 *
 * `model` is the schema `data` is checked against; `top` is the block, which supplies the
 * block name, the reserved paths and the unknown-key wording.
 */
function keyProblems(
  top: BlockSchema,
  model: ObjectSchema,
  data: Readonly<Record<string, unknown>>,
  prefix: string,
  problems: string[],
): void {
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined) continue;
    const path = `${prefix}${key}`;
    const problem = badKey(top, path, key, model.fields);
    if (problem !== null) problems.push(problem);
    else valueProblems(top, model.fields[key]!, value, path, problems);
  }
}

/**
 * Return why `key` at `path` is not allowed, or `null` when it is.
 *
 * `allowed` holds the keys the enclosing object accepts.
 */
function badKey(top: BlockSchema, path: string, key: string, allowed: Readonly<Record<string, Field>>): string | null {
  if (Object.hasOwn(top.reserved, path)) {
    return `\`${top.name}.${path}\` is built by yian (${top.reserved[path]!}); remove it`;
  }
  if (!Object.hasOwn(allowed, key)) {
    return `\`${top.name}.${path}\` ${top.api}; allowed: ${Object.keys(allowed).sort(compareCodeUnits).join(", ")}`;
  }
  return null;
}

/**
 * Check the keys of a nested object, or of each object in a list, at `path`.
 *
 * Objects inside a map field are not walked; none exists, and the value pass's
 * `extra_forbidden` would still refuse an unknown key there.
 */
function valueProblems(top: BlockSchema, field: Field, value: unknown, path: string, problems: string[]): void {
  if (isRecord(value)) {
    const part = nestedSchema(field, value);
    if (part !== null) keyProblems(top, part, value, `${path}.`, problems);
  } else if (isList(value)) {
    const item = listItem(field);
    if (item !== null) value.forEach((entry, index) => valueProblems(top, item, entry, `${path}[${index}]`, problems));
  }
}

/** Return what an item of a list `field` may be (a union for several list members), if it is a list. */
function listItem(field: Field): Field | null {
  const items = members(field).flatMap((member) => (member.kind === "list" ? [member.item] : []));
  if (items.length === 0) return null;
  return items.length === 1 ? items[0]! : union(...items);
}

/** Return the fields a union allows, flattening nested unions. */
function members(field: Field): Field[] {
  return field.kind === "union" ? field.members.flatMap(members) : [field];
}

/**
 * Return the schema to check the record `value` against, if one can be chosen.
 *
 * A tagged field is told apart by `type`; when `value` names no variant there is none,
 * and the value pass reports the problem instead.
 */
function nestedSchema(field: Field, value: Readonly<Record<string, unknown>>): ObjectSchema | null {
  if (field.kind === "tagged") {
    const tag = Object.hasOwn(value, "type") ? value["type"] : undefined;
    return typeof tag === "string" && Object.hasOwn(field.variants, tag) ? field.variants[tag]! : null;
  }
  const objects = members(field).flatMap((member) => (member.kind === "object" ? [member.schema] : []));
  return objects.length === 1 ? objects[0]! : null;
}

/** What `block` builds a block from. */
interface BlockOptions {
  readonly title: string;
  readonly name: string;
  readonly api: string;
  readonly reserved: Readonly<Record<string, string>>;
  readonly fields: Readonly<Record<string, Field>>;
}

/** Return a frozen block schema. */
function block({ title, name, api, reserved, fields }: BlockOptions): BlockSchema {
  return Object.freeze({ ...schema(fields), title, name, api, reserved: table(reserved) });
}

// --------------------------------------------------------------------------------------- //
// OpenAI Responses API
// --------------------------------------------------------------------------------------- //

/** The `reasoning` object of an OpenAI Responses API request. */
const OpenAIReasoning = schema({
  context: literal("auto", "current_turn", "all_turns"),
  effort: literal("none", "minimal", "low", "medium", "high", "xhigh", "max"),
  // The SDK also types `mode` as any string; these are the values it documents.
  mode: literal("standard", "pro"),
  summary: literal("auto", "concise", "detailed"),
});

/** The `text` object of an OpenAI Responses API request (`format` is reserved). */
const OpenAIText = schema({ verbosity: literal("low", "medium", "high") });

/** The `access_programs` object of an OpenAI Responses API request. */
const OpenAIAccessPrograms = schema({ cyber: literal("standard", "daybreak_blue", "daybreak_red") });

/** How moderation treats the input or the output: score it, or block it. */
const OpenAIModerationMode = schema({ mode: literal("score", "block") }, ["mode"]);

/** The `moderation.policy` object of an OpenAI Responses API request. */
const OpenAIModerationPolicy = schema({ input: object(OpenAIModerationMode), output: object(OpenAIModerationMode) });

/** The `moderation` object of an OpenAI Responses API request. */
const OpenAIModeration = schema({ model: str(), policy: object(OpenAIModerationPolicy) }, ["model"]);

/**
 * The `prompt_cache_options` object of an OpenAI Responses API request.
 *
 * `comparison_response_id` is left out because it names one earlier response, and
 * `prewarm` because it suppresses the reply yian reads.
 */
const OpenAIPromptCacheOptions = schema({ mode: literal("implicit", "explicit"), ttl: literal("30m") });

const OPENAI_METADATA = map(str({ maxLength: 64 }), str({ maxLength: 512 }), { maxLength: 16 });

/** Responses API request fields that every OpenAI transport yian uses accepts. */
const OPENAI_RESPONSES_SHARED: Readonly<Record<string, Field>> = Object.freeze({
  access_programs: object(OpenAIAccessPrograms),
  max_tool_calls: int({ ge: 1n }),
  metadata: OPENAI_METADATA,
  moderation: object(OpenAIModeration),
  prompt_cache_key: str(),
  prompt_cache_options: object(OpenAIPromptCacheOptions),
  prompt_cache_retention: literal("in_memory", "24h"),
  reasoning: object(OpenAIReasoning),
  safety_identifier: str({ maxLength: 64 }),
  service_tier: literal("auto", "default", "flex", "scale", "priority", "fast", "ultrafast"),
  text: object(OpenAIText),
  top_logprobs: int({ ge: 0n, le: 20n }),
  top_p: float({ ge: 0, le: 1 }),
  truncation: literal("auto", "disabled"),
  user: str(),
});

/** The paths of a Responses API request that yian builds, with what it builds there. */
const OPENAI_RESERVED: Readonly<Record<string, string>> = Object.freeze({
  model: "llm_config.model",
  input: "messages",
  instructions: "system prompt",
  prompt: "messages",
  tools: "tools",
  tool_choice: "tools",
  parallel_tool_calls: "tools",
  "text.format": "structured output",
  stream: "streaming",
  stream_options: "streaming",
  previous_response_id: "conversation state",
  conversation: "conversation state",
  store: "conversation state",
  include: "response items",
  background: "request lifecycle",
  context_management: "conversation state",
});

/**
 * The `openai` block: the request body of the OpenAI Responses API (`POST /v1/responses`).
 *
 * Every body keyword of the request is a field here or reserved.
 */
export const OpenAIResponsesBlock: BlockSchema = block({
  title: "OpenAIResponsesBlock",
  name: "openai",
  api: "is not a supported field of the OpenAI Responses API request",
  reserved: OPENAI_RESERVED,
  fields: { ...OPENAI_RESPONSES_SHARED, max_output_tokens: int({ ge: 1n }), temperature: float({ ge: 0, le: 2 }) },
});

/**
 * The `openai` block when the call goes to the ChatGPT subscription endpoint.
 *
 * The endpoint accepts a subset of the Responses API request: everything in
 * `OpenAIResponsesBlock` except `temperature` and `max_output_tokens`, which it refuses.
 */
export const OpenAISubscriptionBlock: BlockSchema = block({
  title: "OpenAISubscriptionBlock",
  name: "openai",
  api: "is not accepted by the ChatGPT subscription endpoint (a subset of the Responses API)",
  reserved: OPENAI_RESERVED,
  fields: OPENAI_RESPONSES_SHARED,
});

// --------------------------------------------------------------------------------------- //
// Chat Completions APIs
// --------------------------------------------------------------------------------------- //

/** The Chat Completions body fields every vendor below reserves, with what yian builds there. */
const CHAT_RESERVED: Readonly<Record<string, string>> = Object.freeze({
  model: "llm_config.model",
  messages: "messages",
  tools: "tools",
  tool_choice: "tools",
  parallel_tool_calls: "tools",
  response_format: "structured output",
  stream: "streaming",
  stream_options: "streaming",
  n: "one choice per call",
});

const STOP4 = union(str(), list(str(), { maxLength: 4 }));

// A `logit_bias` key: a token id as a digit string. A YAML integer key reaches here as
// text already, since record keys are always text.
const TOKEN_ID = str({ pattern: "^[0-9]+$" });

/** `temperature` in [0, 2] and `top_p` in [0, 1], as most vendors take them. */
const CHAT_SAMPLING: Readonly<Record<string, Field>> = Object.freeze({
  temperature: float({ ge: 0, le: 2 }),
  top_p: float({ ge: 0, le: 1 }),
});

/** `frequency_penalty` and `presence_penalty`, each in [-2, 2]. */
const CHAT_PENALTIES: Readonly<Record<string, Field>> = Object.freeze({
  frequency_penalty: float({ ge: -2, le: 2 }),
  presence_penalty: float({ ge: -2, le: 2 }),
});

/** `max_tokens` (older) and `max_completion_tokens` output limits, each at least 1. */
const CHAT_LENGTH: Readonly<Record<string, Field>> = Object.freeze({
  max_completion_tokens: int({ ge: 1n }),
  max_tokens: int({ ge: 1n }),
});

/** `logprobs` and `top_logprobs` in [0, 20]. */
const CHAT_LOGPROBS: Readonly<Record<string, Field>> = Object.freeze({
  logprobs: bool(),
  top_logprobs: int({ ge: 0n, le: 20n }),
});

// Blocks spread the shared field records last base first, then their own fields, which sets the
// field order; an own field replaces a shared one in place.

/**
 * The `xai` block: the request body of the xAI Chat Completions API.
 *
 * Mirrors `POST /v1/chat/completions`
 * (https://docs.x.ai/developers/rest-api-reference/inference/chat-completions), checked
 * 2026-10-13. `logit_bias` is left out because the reference marks it unsupported, and
 * `reasoning_effort` is any string because the accepted values depend on the model.
 */
export const XAIChatBlock: BlockSchema = block({
  title: "XAIChatBlock",
  name: "xai",
  api: "is not a supported field of the xAI Chat Completions request",
  reserved: {
    ...CHAT_RESERVED,
    deferred: "request lifecycle",
    search_parameters: "tools: server-side search",
    web_search_options: "tools: server-side search",
  },
  fields: {
    ...CHAT_LOGPROBS,
    ...CHAT_LENGTH,
    ...CHAT_PENALTIES,
    ...CHAT_SAMPLING,
    prompt_cache_key: str(),
    reasoning_effort: str(),
    safety_identifier: str(),
    seed: int(),
    service_tier: literal("default", "priority", "fast"),
    stop: list(str(), { maxLength: 4 }),
    top_logprobs: int({ ge: 0n, le: 8n }),
    user: str(),
  },
});

/** The `thinking` object of a Z.AI GLM chat request. */
const GLMThinking = schema({ type: literal("enabled", "disabled"), clear_thinking: bool() });

/**
 * The `glm` block: the request body of the Z.AI GLM Chat Completion API.
 *
 * Mirrors `POST /api/paas/v4/chat/completions`
 * (https://docs.z.ai/api-reference/llm/chat-completion), checked 2026-10-13.
 */
export const GLMChatBlock: BlockSchema = block({
  title: "GLMChatBlock",
  name: "glm",
  api: "is not a supported field of the Z.AI GLM Chat Completion request",
  reserved: { ...CHAT_RESERVED, tool_stream: "streaming" },
  fields: {
    do_sample: bool(),
    max_tokens: int({ ge: 1n, le: 131072n }),
    reasoning_effort: literal("max", "xhigh", "high", "medium", "low", "minimal", "none"),
    stop: list(str(), { maxLength: 4 }),
    temperature: float({ ge: 0, le: 1 }),
    thinking: object(GLMThinking),
    top_p: float({ ge: 0.01, le: 1 }),
    user_id: str({ minLength: 6, maxLength: 128 }),
  },
});

/** The `glm-cn` block: `GLMChatBlock` for the mainland China endpoint. */
export const GLMCNChatBlock: BlockSchema = Object.freeze({
  ...GLMChatBlock,
  title: "GLMCNChatBlock",
  name: "glm-cn",
});

/** The `thinking` object of a MiniMax chat request. */
const MiniMaxThinking = schema({ type: literal("disabled", "adaptive") });

/**
 * The `minimax` block: the request body of MiniMax's OpenAI-compatible API.
 *
 * Mirrors the Chat Completions parameters MiniMax lists
 * (https://platform.minimax.io/docs/api-reference/text-openai-api), checked 2026-10-13.
 * MiniMax ignores the other OpenAI parameters, so they are not fields here.
 */
export const MiniMaxChatBlock: BlockSchema = block({
  title: "MiniMaxChatBlock",
  name: "minimax",
  api: "is not a supported field of the MiniMax Chat Completions request",
  reserved: { ...CHAT_RESERVED, reasoning_split: "keeps reasoning out of the message content" },
  fields: {
    ...CHAT_LENGTH,
    ...CHAT_SAMPLING,
    reasoning_effort: literal("low", "medium", "high", "xhigh", "max"),
    service_tier: literal("standard", "priority"),
    thinking: object(MiniMaxThinking),
  },
});

/** The `minimax-cn` block: `MiniMaxChatBlock` for the mainland China endpoint. */
export const MiniMaxCNChatBlock: BlockSchema = Object.freeze({
  ...MiniMaxChatBlock,
  title: "MiniMaxCNChatBlock",
  name: "minimax-cn",
});

/** The `thinking` object of a DeepSeek chat request. */
const DeepSeekThinking = schema({ type: literal("enabled", "disabled") });

/**
 * The `deepseek` block: the request body of the DeepSeek Chat Completion API.
 *
 * Mirrors `POST /chat/completions`
 * (https://api-docs.deepseek.com/api/create-chat-completion), checked 2026-10-13.
 * `frequency_penalty` and `presence_penalty` are left out because the reference says
 * they no longer take effect.
 */
export const DeepSeekChatBlock: BlockSchema = block({
  title: "DeepSeekChatBlock",
  name: "deepseek",
  api: "is not a supported field of the DeepSeek Chat Completion request",
  reserved: CHAT_RESERVED,
  fields: {
    ...CHAT_LOGPROBS,
    max_tokens: int({ ge: 1n, le: 393216n }),
    reasoning_effort: literal("none", "low", "high", "max"),
    stop: union(str(), list(str(), { maxLength: 16 })),
    temperature: float({ ge: 0, le: 2 }),
    thinking: object(DeepSeekThinking),
    top_p: float({ gt: 0, le: 1 }),
    user_id: str({ maxLength: 512, pattern: "^[a-zA-Z0-9_-]+$" }),
  },
});

/** The `search_options` object of a Qwen chat request. */
const QwenSearchOptions = schema({
  enable_search_extension: bool(),
  forced_search: bool(),
  search_strategy: literal("turbo", "max", "agent"),
});

/**
 * The `qwen` block: the request body of Model Studio's OpenAI-compatible chat.
 *
 * Mirrors Alibaba Cloud Model Studio's OpenAI-compatible Chat Completions request
 * (https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions),
 * checked 2026-10-13. `thinking` and `clear_thinking` are for the MiniMax and GLM models
 * Model Studio also serves.
 */
export const QwenChatBlock: BlockSchema = block({
  title: "QwenChatBlock",
  name: "qwen",
  api: "is not a supported field of the Model Studio Chat Completions request",
  reserved: {
    ...CHAT_RESERVED,
    tool_stream: "streaming",
    modalities: "output format",
    audio: "output format",
    enable_code_interpreter: "tools: server-side code interpreter",
  },
  fields: {
    ...CHAT_LENGTH,
    clear_thinking: bool(),
    enable_search: bool(),
    enable_thinking: bool(),
    logprobs: bool(),
    preserve_thinking: bool(),
    presence_penalty: float({ ge: -2, le: 2 }),
    reasoning_effort: literal("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"),
    repetition_penalty: float({ gt: 0 }),
    search_options: object(QwenSearchOptions),
    seed: int({ ge: 0n, le: 2n ** 31n - 1n }),
    // A stop list holds strings or token ids, never both.
    stop: union(str(), list(str()), list(int())),
    temperature: float({ ge: 0, lt: 2 }),
    thinking: object(MiniMaxThinking),
    thinking_budget: int({ ge: 1n }),
    top_k: int({ ge: 0n }),
    top_logprobs: int({ ge: 0n, le: 5n }),
    top_p: float({ gt: 0, le: 1 }),
    use_multichannel: bool(),
    vl_high_resolution_images: bool(),
  },
});

/** The `qwen-cn` block: `QwenChatBlock` for the mainland China endpoint. */
export const QwenCNChatBlock: BlockSchema = Object.freeze({
  ...QwenChatBlock,
  title: "QwenCNChatBlock",
  name: "qwen-cn",
});

/** The `reasoning` object of an OpenRouter chat request. */
const OpenRouterReasoning = schema({
  effort: literal("max", "xhigh", "high", "medium", "low", "minimal", "none"),
  enabled: bool(),
  exclude: bool(),
  max_tokens: int({ ge: 0n }),
  summary: literal("auto", "concise", "detailed"),
});

/** The top-level `cache_control` object of an OpenRouter chat request. */
const OpenRouterCacheControl = schema({ type: literal("ephemeral"), ttl: literal("5m", "1h") });

/** The `prompt_cache_options` object of an OpenRouter chat request. */
const OpenRouterPromptCacheOptions = schema({ mode: literal("explicit"), ttl: str() });

/**
 * The `openrouter` block: the request body of the OpenRouter Chat Completions API.
 *
 * Mirrors `POST /api/v1/chat/completions`
 * (https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion), with
 * bounds from https://openrouter.ai/docs/api/reference/parameters and the `reasoning`
 * fields from https://openrouter.ai/docs/guides/best-practices/reasoning-tokens, checked
 * 2026-10-13. `provider` (routing preferences) is passed through as written.
 * `session_id` and `trace` are left out because they identify one conversation or
 * trace, so a value fixed in the flow would group every run together.
 */
export const OpenRouterChatBlock: BlockSchema = block({
  title: "OpenRouterChatBlock",
  name: "openrouter",
  api: "is not a supported field of the OpenRouter Chat Completions request",
  reserved: {
    ...CHAT_RESERVED,
    plugins: "tools",
    web_search_options: "tools: server-side search",
    stop_server_tools_when: "tools",
    deferred_tools: "tools",
    structured_outputs: "structured output",
    modalities: "output format",
    image_config: "output format",
    debug: "streaming",
  },
  fields: {
    ...CHAT_LOGPROBS,
    ...CHAT_LENGTH,
    ...CHAT_PENALTIES,
    ...CHAT_SAMPLING,
    cache_control: object(OpenRouterCacheControl),
    logit_bias: map(TOKEN_ID, float({ ge: -100, le: 100 })),
    metadata: OPENAI_METADATA,
    min_p: float({ ge: 0, le: 1 }),
    models: list(str()),
    prompt_cache_key: str(),
    prompt_cache_options: object(OpenRouterPromptCacheOptions),
    provider: map(str(), json()),
    reasoning: object(OpenRouterReasoning),
    reasoning_effort: literal("max", "xhigh", "high", "medium", "low", "minimal", "none"),
    repetition_penalty: float({ ge: 0, le: 2 }),
    seed: int(),
    service_tier: literal("auto", "default", "fast", "flex", "priority", "scale", "ultrafast"),
    stop: STOP4,
    top_a: float({ ge: 0, le: 1 }),
    top_k: int({ ge: 0n }),
    user: str(),
    verbosity: literal("low", "medium", "high", "xhigh", "max"),
  },
});

/**
 * The `vllm` block: the request body of a vLLM server's Chat Completions API.
 *
 * Mirrors vLLM's `ChatCompletionRequest` with its extra sampling parameters
 * (https://docs.vllm.ai/en/latest/serving/online_serving/), with bounds from vLLM's
 * `SamplingParams` checks, checked 2026-10-13. `top_logprobs` is capped at 20, the
 * server's default `max_logprobs`.
 */
export const VLLMChatBlock: BlockSchema = block({
  title: "VLLMChatBlock",
  name: "vllm",
  api: "is not a supported field of the vLLM Chat Completions request",
  reserved: {
    ...CHAT_RESERVED,
    guided_json: "structured output",
    guided_regex: "structured output",
    guided_choice: "structured output",
    guided_grammar: "structured output",
    guided_decoding_backend: "structured output",
    guided_whitespace_pattern: "structured output",
    structured_outputs: "structured output",
    chat_template: "messages",
    add_generation_prompt: "messages",
    continue_final_message: "messages",
    echo: "messages",
  },
  fields: {
    ...CHAT_LOGPROBS,
    ...CHAT_LENGTH,
    ...CHAT_PENALTIES,
    chat_template_kwargs: map(str(), json()),
    ignore_eos: bool(),
    include_stop_str_in_output: bool(),
    length_penalty: float(),
    logit_bias: map(TOKEN_ID, float()),
    min_p: float({ ge: 0, le: 1 }),
    min_tokens: int({ ge: 0n }),
    priority: int(),
    reasoning_effort: literal("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    repetition_penalty: float({ gt: 0 }),
    seed: int({ ge: -(2n ** 63n), le: 2n ** 63n - 1n }),
    skip_special_tokens: bool(),
    spaces_between_special_tokens: bool(),
    stop: union(str(), list(str())),
    stop_token_ids: list(int()),
    temperature: float({ ge: 0, le: 2 }),
    // -1 means no limit.
    thinking_token_budget: int({ ge: -1n }),
    // -1 and 0 both turn top-k off.
    top_k: int({ ge: -1n }),
    top_p: float({ gt: 0, le: 1 }),
    user: str(),
  },
});

/** The `user_security_context` object of an Azure OpenAI chat request. */
const AzureUserSecurityContext = schema({
  application_name: str(),
  end_user_id: str(),
  end_user_tenant_id: str(),
  source_ip: str(),
});

/**
 * The `azure` block: the request body of the Azure OpenAI Chat Completions API.
 *
 * Mirrors the chat completions request in the Azure OpenAI reference
 * (https://learn.microsoft.com/azure/ai-foundry/openai/latest) and the matching
 * `openai.types.chat.completion_create_params` fields, checked 2026-10-13.
 * `service_tier`, `prompt_cache_options` and `moderation` are left out: the `openai`
 * SDK sends them, but the Azure reference does not list them.
 */
export const AzureChatBlock: BlockSchema = block({
  title: "AzureChatBlock",
  name: "azure",
  api: "is not a supported field of the Azure OpenAI Chat Completions request",
  reserved: {
    ...CHAT_RESERVED,
    data_sources: "tools: On Your Data retrieval",
    functions: "tools",
    function_call: "tools",
    modalities: "output format",
    audio: "output format",
  },
  fields: {
    ...CHAT_LOGPROBS,
    ...CHAT_LENGTH,
    ...CHAT_PENALTIES,
    ...CHAT_SAMPLING,
    logit_bias: map(TOKEN_ID, int({ ge: -100n, le: 100n })),
    metadata: OPENAI_METADATA,
    prompt_cache_key: str(),
    prompt_cache_retention: literal("in_memory", "24h"),
    reasoning_effort: literal("none", "minimal", "low", "medium", "high", "xhigh"),
    safety_identifier: str({ maxLength: 64 }),
    seed: int(),
    stop: STOP4,
    store: bool(),
    user: str(),
    user_security_context: object(AzureUserSecurityContext),
    verbosity: literal("low", "medium", "high"),
  },
});

// --------------------------------------------------------------------------------------- //
// Anthropic Messages API
// --------------------------------------------------------------------------------------- //

/** A `CacheControlEphemeral` object of an Anthropic Messages request. */
const AnthropicCacheControl = schema({ type: literal("ephemeral"), ttl: literal("5m", "1h") }, ["type"]);

/** The `metadata` object of an Anthropic Messages request. */
const AnthropicMetadata = schema({ user_id: str({ maxLength: 512 }) });

/** The `output_config` object of an Anthropic Messages request (`format` is reserved). */
const AnthropicOutputConfig = schema({ effort: literal("low", "medium", "high", "xhigh", "max") });

const THINKING_DISPLAY = literal("summarized", "omitted");

/** Manual extended thinking with a token budget of at least 1024. */
const AnthropicThinkingEnabled = schema(
  { type: literal("enabled"), budget_tokens: int({ ge: 1024n }), display: THINKING_DISPLAY },
  ["type", "budget_tokens"],
);

/** Thinking turned off. */
const AnthropicThinkingDisabled = schema({ type: literal("disabled") }, ["type"]);

/** Thinking only between tool calls. */
const AnthropicThinkingBetweenTools = schema({ type: literal("between_tools") }, ["type"]);

/** Thinking whose amount the model decides. */
const AnthropicThinkingAdaptive = schema({ type: literal("adaptive"), display: THINKING_DISPLAY }, ["type"]);

/**
 * The `anthropic` block: the request body of the Anthropic Messages API (`POST /v1/messages`).
 *
 * `temperature`, `top_k` and `top_p` are deprecated for newer models but still body
 * fields; a model that refuses them answers with a provider error. `max_tokens` starts at
 * 1 because a zero-token call only warms the cache and yian reads the reply.
 */
export const AnthropicMessagesBlock: BlockSchema = block({
  title: "AnthropicMessagesBlock",
  name: "anthropic",
  api: "is not a supported field of the Anthropic Messages API request",
  reserved: {
    model: "llm_config.model",
    messages: "messages",
    system: "system prompt",
    tools: "tools",
    tool_choice: "tools",
    stream: "streaming",
    "output_config.format": "structured output",
    container: "tools",
    mcp_servers: "tools",
    diagnostics: "conversation state",
  },
  fields: {
    cache_control: object(AnthropicCacheControl),
    inference_geo: str(),
    max_tokens: int({ ge: 1n }),
    metadata: object(AnthropicMetadata),
    output_config: object(AnthropicOutputConfig),
    service_tier: literal("auto", "standard_only"),
    stop_sequences: list(str()),
    temperature: float({ ge: 0, le: 1 }),
    thinking: tagged({
      enabled: AnthropicThinkingEnabled,
      disabled: AnthropicThinkingDisabled,
      between_tools: AnthropicThinkingBetweenTools,
      adaptive: AnthropicThinkingAdaptive,
    }),
    top_k: int({ ge: 0n }),
    top_p: float({ ge: 0, le: 1 }),
  },
});

// --------------------------------------------------------------------------------------- //
// Gemini generateContent
// --------------------------------------------------------------------------------------- //

/** The `generationConfig.thinkingConfig` object of a Gemini request. */
const GeminiThinkingConfig = schema({
  includeThoughts: bool(),
  // -1 asks for a dynamic budget and 0 turns thinking off on models that allow it.
  thinkingBudget: int(),
  thinkingLevel: literal("MINIMAL", "LOW", "MEDIUM", "HIGH"),
});

/** The `generationConfig` object of a Gemini request, without its output-format fields. */
const GeminiGenerationConfig = schema({
  enableEnhancedCivicAnswers: bool(),
  frequencyPenalty: float(),
  logprobs: int({ ge: 0n, le: 20n }),
  maxOutputTokens: int({ ge: 1n }),
  mediaResolution: literal("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"),
  presencePenalty: float(),
  responseLogprobs: bool(),
  seed: int(),
  stopSequences: list(str(), { maxLength: 5 }),
  temperature: float({ ge: 0, le: 2 }),
  thinkingConfig: object(GeminiThinkingConfig),
  topK: int(),
  topP: float({ ge: 0, le: 1 }),
});

/** One `safetySettings` entry of a Gemini request: a category and its block threshold. */
const GeminiSafetySetting = schema(
  {
    category: literal(
      "HARM_CATEGORY_HARASSMENT",
      "HARM_CATEGORY_HATE_SPEECH",
      "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "HARM_CATEGORY_DANGEROUS_CONTENT",
      "HARM_CATEGORY_CIVIC_INTEGRITY",
      "HARM_CATEGORY_JAILBREAK",
    ),
    threshold: literal("BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE", "OFF"),
  },
  ["category", "threshold"],
);

/**
 * The `google` block: the request body of the Gemini API `generateContent` method.
 *
 * Uses the REST (camelCase) names and enum spellings. Output-modality fields (speech,
 * image, audio, translation) are left out because yian reads a text reply, and `labels`
 * because the Gemini Developer API refuses it.
 */
export const GeminiGenerateContentBlock: BlockSchema = block({
  title: "GeminiGenerateContentBlock",
  name: "google",
  api: "is not a supported field of the Gemini generateContent request",
  reserved: {
    model: "llm_config.model",
    contents: "messages",
    systemInstruction: "system prompt",
    tools: "tools",
    toolConfig: "tools",
    cachedContent: "system prompt and tools",
    continuationToken: "conversation state",
    "generationConfig.responseMimeType": "structured output",
    "generationConfig.responseSchema": "structured output",
    "generationConfig.responseJsonSchema": "structured output",
    "generationConfig._responseJsonSchema": "structured output",
    "generationConfig.responseFormat": "structured output",
    "generationConfig.candidateCount": "yian reads one reply",
  },
  fields: {
    generationConfig: object(GeminiGenerationConfig),
    safetySettings: list(object(GeminiSafetySetting)),
    serviceTier: literal("standard", "flex", "priority"),
    store: bool(),
  },
});

// --------------------------------------------------------------------------------------- //
// Ollama /api/chat
// --------------------------------------------------------------------------------------- //

/**
 * The `options` object of an Ollama chat request.
 *
 * Has the fields of `ollama.Options` plus `min_p`, which the /api/chat reference lists;
 * the client sends `options` as a plain mapping.
 */
const OllamaOptions = schema({
  // Load-time options.
  numa: bool(),
  num_ctx: int(),
  num_batch: int(),
  num_gpu: int(),
  main_gpu: int(),
  low_vram: bool(),
  f16_kv: bool(),
  logits_all: bool(),
  vocab_only: bool(),
  use_mmap: bool(),
  use_mlock: bool(),
  embedding_only: bool(),
  num_thread: int(),

  // Runtime options.
  num_keep: int(),
  seed: int(),
  num_predict: int(),
  top_k: int(),
  top_p: float(),
  tfs_z: float(),
  typical_p: float(),
  repeat_last_n: int(),
  temperature: float(),
  repeat_penalty: float(),
  presence_penalty: float(),
  frequency_penalty: float(),
  mirostat: int(),
  mirostat_tau: float(),
  mirostat_eta: float(),
  min_p: float(),
  penalize_newline: bool(),
  stop: list(str()),
});

/**
 * The `ollama` block: the request body of Ollama's native chat API.
 *
 * Mirrors `POST /api/chat`
 * (https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion),
 * checked 2026-10-13, limited to what `ollama.Client.chat` sends (`ollama` 0.6.3).
 * `think` takes the levels the reference lists; the client passes any string through.
 */
export const OllamaChatBlock: BlockSchema = block({
  title: "OllamaChatBlock",
  name: "ollama",
  api: "is not a supported field of the Ollama /api/chat request",
  reserved: {
    model: "llm_config.model",
    messages: "messages",
    tools: "tools",
    stream: "streaming",
    format: "structured output",
  },
  fields: {
    // A duration string such as "5m", or a number of seconds; negative keeps the model loaded.
    keep_alive: union(float(), str()),
    logprobs: bool(),
    options: object(OllamaOptions),
    think: union(bool(), literal("low", "medium", "high", "max")),
    top_logprobs: int(),
  },
});

// --------------------------------------------------------------------------------------- //
// Registry
// --------------------------------------------------------------------------------------- //

/** Provider name -> the block of the API a call to it reaches. */
export const PROVIDER_BLOCKS: Readonly<Record<string, BlockSchema>> = table({
  openai: OpenAIResponsesBlock,
  azure: AzureChatBlock,
  xai: XAIChatBlock,
  deepseek: DeepSeekChatBlock,
  qwen: QwenChatBlock,
  "qwen-cn": QwenCNChatBlock,
  glm: GLMChatBlock,
  "glm-cn": GLMCNChatBlock,
  minimax: MiniMaxChatBlock,
  "minimax-cn": MiniMaxCNChatBlock,
  openrouter: OpenRouterChatBlock,
  vllm: VLLMChatBlock,
  ollama: OllamaChatBlock,
  anthropic: AnthropicMessagesBlock,
  google: GeminiGenerateContentBlock,
});

/**
 * Return the block for the API a call to `provider` actually reaches.
 *
 * @param provider A key of `PROVIDER_BLOCKS`, e.g. `"openai"`.
 * @param connection The resolved OpenAI connection, whose own `method` says how the call is
 *   authenticated; `undefined` means no connection.
 * @param apiKey The host-supplied key. A key sends the call to the Responses API even when the
 *   connection is a subscription.
 * @returns `OpenAISubscriptionBlock` for an `openai` call on a `subscription` connection
 *   without `apiKey`; otherwise `PROVIDER_BLOCKS[provider]`.
 * @throws {RangeError} If `provider` has no block.
 */
export function blockSchema(provider: string, connection?: unknown, apiKey?: unknown): BlockSchema {
  if (
    provider === "openai" &&
    isRecord(connection) &&
    Object.hasOwn(connection, "method") &&
    connection["method"] === "subscription" &&
    (apiKey === null || apiKey === undefined)
  ) {
    return OpenAISubscriptionBlock;
  }
  if (!Object.hasOwn(PROVIDER_BLOCKS, provider)) {
    throw new RangeError(`no provider block named ${valueRepr(provider)}`);
  }
  return PROVIDER_BLOCKS[provider]!;
}

// --------------------------------------------------------------------------------------- //
// Blocks that hold values
// --------------------------------------------------------------------------------------- //

/** One problem `locatedBlockProblems` finds in a provider block. */
export class BlockProblem {
  /**
   * The deepest written key path inside the block the problem names, with list indices as
   * numbers; empty for the block itself.
   */
  readonly keys: readonly (string | number)[];
  /** The message, starting with the backticked key path. */
  readonly message: string;

  constructor(keys: readonly (string | number)[], message: string) {
    this.keys = Object.freeze([...keys]);
    this.message = message;
    Object.freeze(this);
  }
}

/** The type of a provider-block field simple enough to check a value's source against. */
export interface BlockFieldType {
  readonly base: "str" | "int" | "float" | "bool";
  /** The values a literal field allows; empty for any other. */
  readonly literals: readonly string[];
  /** The inclusive lower bound, if any; a `bigint` for an int field. */
  readonly ge: bigint | number | null;
  /** The inclusive upper bound, if any; a `bigint` for an int field. */
  readonly le: bigint | number | null;
}

/** What a walk of a block holding values shares at every level. */
interface PartialWalk {
  readonly top: BlockSchema;
  readonly isValue: (leaf: unknown) => boolean;
  readonly problems: BlockProblem[];
}

/**
 * Check a provider block whose leaves may be values that resolve later.
 *
 * Reserved and unknown key paths are reported as `blockSchema`'s block reports them. A value
 * is opaque: it is never checked, a required field given as one counts as present, and a
 * nested object whose `type` (the variant tag) is one is not walked. Every other leaf is
 * checked against its field's type and constraints. A list holding a value has its object
 * entries walked for key paths and its other entries left for when it resolves; a mapping
 * field holding one has its value-free items checked and its item count left for then.
 * Messages name the key path and never echo a value.
 *
 * @param block A key of `PROVIDER_BLOCKS`, e.g. `"anthropic"`.
 * @param tree The block as written, with values in place of some leaves or subtrees.
 * @param isValue Whether a leaf or subtree is a value.
 * @returns One problem each, with its key path, in the order written; empty when none.
 * @throws {RangeError} If `block` has no block.
 */
export function locatedBlockProblems(
  block: string,
  tree: unknown,
  isValue: (leaf: unknown) => boolean,
): BlockProblem[] {
  const top = blockSchema(block);
  if (isValue(tree)) return [];
  if (!isRecord(tree)) return [new BlockProblem([], `\`${top.name}\` must be a mapping`)];
  const walk: PartialWalk = { top, isValue, problems: [] };
  partialProblems(walk, top, tree, "", []);
  return walk.problems;
}

/**
 * Append the problems of the partial object `data` checked against `model`.
 *
 * `prefix` is the written key path of `data` as messages show it, and `keys` the same path as parts.
 */
function partialProblems(
  walk: PartialWalk,
  model: ObjectSchema,
  data: Readonly<Record<string, unknown>>,
  prefix: string,
  keys: Loc,
): void {
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined) continue;
    const path = `${prefix}${key}`;
    const problem = badKey(walk.top, path, key, model.fields);
    if (problem !== null) walk.problems.push(new BlockProblem([...keys, key], problem));
    else partialField(walk, model, key, value, path, [...keys, key]);
  }
  for (const key of Object.keys(model.fields)) {
    const given = Object.hasOwn(data, key) ? data[key] : undefined;
    if (given === undefined && model.required.includes(key)) {
      walk.problems.push(new BlockProblem([...keys, key], `\`${walk.top.name}.${prefix}${key}\` is required`));
    }
  }
}

/** Append the problems of field `name`'s partial `value`; values are never checked. */
function partialField(
  walk: PartialWalk,
  model: ObjectSchema,
  name: string,
  value: unknown,
  path: string,
  keys: Loc,
): void {
  const { isValue } = walk;
  if (isValue(value)) return;
  const field = model.fields[name]!;
  if (isRecord(value)) {
    const part = nestedSchema(field, value);
    if (part !== null) {
      partialProblems(walk, part, value, `${path}.`, keys);
      return;
    }
    if (holdsValue(value, isValue)) {
      if (members(field).some((member) => member.kind === "map")) {
        // Only the value-free items can be checked; the item count waits for the rest.
        const written = Object.fromEntries(Object.entries(value).filter(([, item]) => !holdsValue(item, isValue)));
        checkLiteral(walk, model, name, written, path, keys);
      } else if (field.kind === "tagged") {
        // Several variants and no literal tag: only the tag itself can be checked.
        const tag = Object.hasOwn(value, "type") ? value["type"] : undefined;
        if (!isValue(tag)) checkLiteral(walk, model, name, tag === undefined ? {} : { type: tag }, path, keys);
      } else if (!members(field).some((member) => member.kind === "object" || member.kind === "json")) {
        // A field that takes no mapping: an empty one gives its own type's message, whatever the
        // values resolve to.
        checkLiteral(walk, model, name, {}, path, keys);
      }
      return;
    }
  } else if (isList(value) && holdsValue(value, isValue)) {
    const item = listItem(field);
    if (item === null) {
      // A field that takes no list, as for a mapping above.
      if (!members(field).some((member) => member.kind === "json")) checkLiteral(walk, model, name, [], path, keys);
      return;
    }
    value.forEach((entry, index) => {
      if (!isRecord(entry) || isValue(entry)) return;
      const part = nestedSchema(item, entry);
      if (part !== null) partialProblems(walk, part, entry, `${path}[${index}].`, [...keys, index]);
    });
    return;
  }
  checkLiteral(walk, model, name, value, path, keys);
}

/**
 * Append why the value-free `value` is not valid for field `name` of `model`.
 *
 * Each message is located at the deepest key path of `value` its issue names; union labels
 * and variant tags in an issue's location are dropped.
 */
function checkLiteral(
  walk: PartialWalk,
  model: ObjectSchema,
  name: string,
  value: unknown,
  path: string,
  keys: Loc,
): void {
  const { top, problems } = walk;
  const add = (parts: Loc, message: string): void => {
    // Keys hold only strings and numbers, so equal JSON means equal keys.
    const keysJson = JSON.stringify(parts);
    if (!problems.some((problem) => problem.message === message && JSON.stringify(problem.keys) === keysJson)) {
      problems.push(new BlockProblem(parts, message));
    }
  };
  if (model === top) {
    // The field is checked as part of the block, whose key walk runs first and stops on a problem.
    const found: string[] = [];
    keyProblems(top, top, { [name]: value }, "", found);
    if (found.length > 0) {
      add(keys, `\`${top.name}.${path}\`: Value error, ${found.join("; ")}`);
      return;
    }
  }
  // An optional field given null is unset, as in the object pass.
  if (value === null && !model.required.includes(name)) return;
  for (const issue of checkField(model.fields[name]!, value, []).issues) {
    const [written, parts] = writtenPath(path, keys, value, issue.loc);
    add(parts, `\`${top.name}.${written}\`: ${issue.msg}`);
  }
}

/** Extend `path` and its parts `keys` by the parts of `loc` written in `value`. */
function writtenPath(path: string, keys: Loc, value: unknown, loc: Loc): [string, Loc] {
  const parts = [...keys];
  let node = value;
  for (const part of loc) {
    if (typeof part === "number" && isList(node) && part < node.length) {
      path += `[${part}]`;
      node = node[part];
    } else if (typeof part === "string" && isRecord(node) && Object.hasOwn(node, part) && node[part] !== undefined) {
      path += `.${part}`;
      node = node[part];
    } else {
      continue;
    }
    parts.push(part);
  }
  return [path, parts];
}

/** Return whether `tree` is, or anywhere holds, a value. */
function holdsValue(tree: unknown, isValue: (leaf: unknown) => boolean): boolean {
  if (isValue(tree)) return true;
  if (isRecord(tree)) return Object.values(tree).some((item) => holdsValue(item, isValue));
  if (isList(tree)) return tree.some((item) => holdsValue(item, isValue));
  return false;
}

/**
 * Return the simple type of the block field at `path`, if it has one.
 *
 * A nested object with several variants is told apart by the literal `type` written at its
 * place in `tree`.
 *
 * @param block A key of `PROVIDER_BLOCKS`, e.g. `"anthropic"`.
 * @param path The field's key path inside the block, e.g. `["thinking", "budget_tokens"]`.
 * @param tree The block as written, which may hold values.
 * @returns The field's type when it is a `str`, `int`, `float` or `bool` (with any inclusive
 *   bounds) or a literal; `null` for any other field (an object, list, mapping or union), an
 *   unknown path, or a variant that `tree` does not name.
 * @throws {RangeError} If `block` has no block.
 */
export function blockFieldAnnotation(block: string, path: readonly string[], tree: unknown): BlockFieldType | null {
  let model: ObjectSchema = blockSchema(block);
  let node = tree;
  for (const [index, key] of path.entries()) {
    if (!Object.hasOwn(model.fields, key)) return null;
    const field = model.fields[key]!;
    if (index === path.length - 1) return simpleType(field);
    node = isRecord(node) && Object.hasOwn(node, key) ? node[key] : undefined;
    const part = nestedSchema(field, isRecord(node) ? node : {});
    if (part === null) return null;
    model = part;
  }
  return null;
}

/**
 * Return the `BlockFieldType` of a field, if it is simple.
 *
 * Only inclusive bounds (`ge`, `le`) are carried; exclusive bounds and string length and
 * pattern constraints are checked only when the whole block is validated after its values resolve.
 */
function simpleType(field: Field): BlockFieldType | null {
  switch (field.kind) {
    case "str":
    case "bool":
      return Object.freeze({ base: field.kind, literals: Object.freeze([]), ge: null, le: null });
    case "int":
    case "float":
      return Object.freeze({
        base: field.kind,
        literals: Object.freeze([]),
        ge: field.ge ?? null,
        le: field.le ?? null,
      });
    case "literal":
      return Object.freeze({ base: "str", literals: field.values, ge: null, le: null });
    default:
      return null;
  }
}
