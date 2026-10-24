/**
 * Structured AGENT output: derive the JSON Schema of a declared `output:` Type and generate a
 * value that conforms to it.
 *
 * An AGENT declares `output:` as a typed `Type`. For an untagged `str`, nullable or not (`str`,
 * `Optional(str)`), the agent stays a text producer: `typeToSchema` returns `null` and the AGENT
 * loop takes the plain-text path. For any constrained or richer type (a `Literal(...)` variant,
 * record, union, scalar `int`/`float`/`bool`, or list) it returns the JSON Schema the model is
 * asked to fill: what Python yian's pydantic model emits (`model_json_schema()`), with ints as
 * `bigint`.
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
 * `generateStructured` sends the schema the way `planStructured` picks for the model, and checks
 * the parsed reply with yian's value check (`checkReply`): the wrapper is stripped, and the value
 * keeps the shape Python's pydantic model gives it (an open record drops unknown fields, an absent
 * optional field reads `null`, records follow the declared field order). An invalid reply is
 * corrected under `retries:`; once they are spent the node fails with `OutputInvalidError`, whose
 * `path` locates the last error inside the output when it can.
 *
 * Knows about:   defaults, diagnostics, llm, memory contracts, nodes/contracts, typesys, peer
 *                AGENT modules.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { current } from "../../defaults.ts";
import { compareCodeUnits } from "../../diagnostics.ts";
import type { ChatModel } from "../../llm/events.ts";
import { Message, Request, TextPart } from "../../llm/messages.ts";
import { planStructured, StructuredOutputError, structuredTarget } from "../../llm/structured.ts";
import {
  type EvaluationTrace,
  ModelTextInteraction,
  ModelValueInteraction,
  type VisibleInteraction,
} from "../../memory/contracts.ts";
import { type JsonObject, type JsonValue, logJson } from "../../typesys/json.ts";
import {
  type Bounds,
  buildValueAs,
  IDENTIFIER_PATTERN,
  isListKind,
  isRecord as isValueRecord,
  listElementKind,
  rejectDeepNesting,
  renderValuePath,
  setField,
  type Type,
  TypeCheckError,
  unionTagFields,
  type Value,
  type ValueRecord,
  ValueKind,
} from "../../typesys/values.ts";
import type { ValuePath } from "../contracts.ts";
import { type AgentSteps, recordedResponse, requestSha256, type RetrySettings } from "./context.ts";
import { callModel, type LiveSinks } from "./live.ts";
import { pushVisible, recordModelText, recordModelValue } from "./text.ts";

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
function schemaFor(typ: Type, defs: Defs, position: string, closed = false): JsonObject {
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
      anyOf: typ.variants.map((member, index) => {
        const name = names[index]!;
        return define(defs, name, recordSchema(member, defs, name, true));
      }),
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
 * A bound keeps the form the type stores it in (`bigint` or `number`); a `-0` bound becomes `0`.
 */
function scalarSchema(kind: ValueKind, bounds: Bounds | undefined): JsonObject {
  const schema: JsonObject = { type: SCALAR_TYPES.get(kind)! };
  const [minimum, maximum] = bounds ?? [null, null];
  // `define` compares schemas by their `logJson` text, which throws on `-0`; only a hand-built
  // Type has one. `-0 === 0`, so the ternary writes `0`, and a `bigint` passes through.
  if (minimum !== null) schema.minimum = minimum === 0 ? 0 : minimum;
  if (maximum !== null) schema.maximum = maximum === 0 ? 0 : maximum;
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
  const [tagField] = unionTagFields(typ);
  const tags = typ.variants!.map((member) =>
    tagField === undefined ? undefined : [...member.fields!.get(tagField)!.tags!][0]!,
  );
  const named = tags.every((tag) => tag !== undefined && IDENTIFIER.test(tag));
  return tags.map((tag, index) => nested(position, named ? tag! : `Member${index + 1}`));
}

/** Return the name of the record at `position`: its title, and its `$defs` key unnormalized. */
function recordName(position: string): string {
  return position || "Record";
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
function recordSchema(typ: Type, defs: Defs, position: string, closed = false): JsonObject {
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
  const schema: JsonObject = { properties, title: recordName(position), type: "object" };
  if (required.length > 0) schema.required = required;
  if (closed) schema.additionalProperties = false;
  return schema;
}

/**
 * Add the record `schema` at `position` to `defs` and return a `$ref` to it. An identical
 * definition that wants the same key is shared.
 */
function define(defs: Defs, position: string, schema: JsonObject): JsonObject {
  const key = recordName(position).replace(/[^A-Za-z0-9_-]/gu, "_");
  // Key order doesn't matter (Python compares dicts); the text sorts keys. Unlike Python, an
  // int and a float bound of equal value differ here, which only costs a separate definition.
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
    const wanted = definition.key;
    let key = wanted;
    if (counts.get(wanted)! > 1) {
      let suffix = suffixes.get(wanted) ?? 0;
      do {
        suffix += 1;
        key = `${wanted}__${suffix}`;
      } while (taken.has(key));
      suffixes.set(wanted, suffix);
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
  setField(properties, field, titled(schemaFor(typ, defs, ""), field, typ));
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
  if (isBareRecord(typ)) {
    schema = recordSchema(typ, defs, "");
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
// Generation: the structured call, its corrections and the reply check
// --------------------------------------------------------------------------------------- //

/**
 * A structured reply was still invalid once the `retries:` corrections were spent. The message is
 * the last error's, and that error is the `cause`.
 */
export class OutputInvalidError extends Error {
  override name = "OutputInvalidError";
  /**
   * Where inside the output the last error is, relative to the output contract (`[]` is its root);
   * `null` when the error isn't the value check's or has no place in the contract.
   */
  readonly path: ValuePath | null;

  constructor(message: string, options: { cause: unknown; path: ValuePath | null }) {
    super(message, { cause: options.cause });
    this.path = options.path === null ? null : Object.freeze([...options.path]);
  }
}

/**
 * A reply's value failed the value check; its `path` is relative to the output type. Distinct from
 * the nesting check's `TypeCheckError`, so `outputPath` locates only value-check errors.
 */
class ReplyCheckError extends TypeCheckError {}

/** Return a user message holding `text`. */
export function userMessage(text: string): Message {
  return new Message({ role: "user", parts: [new TextPart(text)] });
}

/**
 * Generate a value of `typ` with `model`, correcting an invalid reply up to `retries` times.
 *
 * The schema is sent the way `planStructured` picks for the model; the prompt strategy appends its
 * instruction as a user message. Each call goes through `callModel` with purpose
 * `structured_fallback` for the prompt strategy, else `structured_native`. A reply
 * holding no valid answer (`StructuredOutputError`) or failing `checkReply` is not kept in the
 * conversation: the next call gets the earlier messages and a correction naming the error. The
 * prompt strategy records every reply's text in `evaluationTrace`, the other strategies the
 * returned value.
 *
 * @param messages The conversation so far; not changed.
 * @param typ The declared output type; `typeToSchema` must give it a schema.
 * @param options `retries` defaults to `YIAN_AGENT_RETRIES`; `live` gets the call events;
 *   `signal` stops the run before each call and wins over the reply of a call it stopped during;
 *   `retry` is each call's retry settings; with `steps`, each call is written as a `model_call`
 *   step (`callModel`).
 * @returns The checked value, unwrapped.
 * @throws OutputInvalidError once `retries + 1` replies were invalid.
 * @throws Whatever a model call throws (a provider error, a stop's reason), unchanged.
 */
export async function generateStructured(
  model: ChatModel,
  messages: readonly Message[],
  typ: Type,
  options: {
    retries?: number;
    live?: LiveSinks | null;
    signal?: AbortSignal;
    evaluationTrace?: EvaluationTrace | null;
    retry?: RetrySettings;
    steps?: AgentSteps;
  } = {},
): Promise<Value> {
  const { live = null, signal, evaluationTrace, retry, steps } = options;
  const retries = options.retries ?? current().agentRetries;
  const plan = planStructured(structuredTarget(model), typeToSchema(typ)!);
  const prompted = plan.strategy === "prompt";
  const purpose = prompted ? "structured_fallback" : "structured_native";
  const instruction = prompted ? [userMessage(plan.instruction())] : [];
  let conversation = [...messages, ...instruction];
  for (let call = 0; ; call++) {
    signal?.throwIfAborted();
    const request = plan.apply(new Request({ messages: conversation }));
    const { response } = await callModel(live, model, request, {
      purpose,
      signal,
      retry,
      // No steers to leave out: a steerable agent's output is `str`, so it never calls here.
      record: steps === undefined ? undefined : { steps, digest: requestSha256(request, new Set()) },
    });
    // A stop that lands during the call wins over its reply.
    signal?.throwIfAborted();
    if (prompted) recordModelText(evaluationTrace, response.message.text());
    let value: Value;
    try {
      value = checkReply(plan.parse(response), typ);
    } catch (error) {
      if (!(error instanceof StructuredOutputError || error instanceof TypeCheckError)) throw error;
      if (call >= retries) {
        throw new OutputInvalidError(error.message, { cause: error, path: outputPath(error, typ) });
      }
      const correction = userMessage(plan.correction(error.message));
      conversation = [...conversation, correction];
      continue;
    }
    if (!prompted) recordModelValue(evaluationTrace, value);
    return value;
  }
}

/**
 * Rebuild what `generateStructured(model, messages, typ)` showed a learner from its recorded
 * `model_call` steps, in order: each prompt-strategy reply's text, or the other strategies'
 * checked value. A failed attempt shows nothing; an invalid reply adds the correction the next
 * request carries.
 *
 * @param steps The structured calls' `model_call` steps.
 * @returns The interactions, and whether each step recorded the purpose and the digest of the
 *   request rebuilt for it.
 */
export function structuredMaterial(
  model: ChatModel,
  messages: readonly Message[],
  typ: Type,
  steps: readonly ValueRecord[],
): { interactions: VisibleInteraction[]; digestOk: boolean } {
  const plan = planStructured(structuredTarget(model), typeToSchema(typ)!);
  const prompted = plan.strategy === "prompt";
  const purpose = prompted ? "structured_fallback" : "structured_native";
  let conversation = [...messages, ...(prompted ? [userMessage(plan.instruction())] : [])];
  const interactions: VisibleInteraction[] = [];
  let digestOk = true;
  for (const step of steps) {
    const request = plan.apply(new Request({ messages: conversation }));
    if (step["purpose"] !== purpose || step["request_sha256"] !== requestSha256(request, new Set())) digestOk = false;
    if (Object.hasOwn(step, "error")) continue;
    const response = recordedResponse(step);
    if (prompted) pushVisible(interactions, () => new ModelTextInteraction(response.message.text()));
    try {
      const value = checkReply(plan.parse(response), typ);
      if (!prompted) pushVisible(interactions, () => new ModelValueInteraction(value));
    } catch (error) {
      if (!(error instanceof StructuredOutputError || error instanceof TypeCheckError)) throw error;
      conversation = [...conversation, userMessage(plan.correction(error.message))];
    }
  }
  return { interactions, digestOk };
}

/** Whether `typ` is a non-nullable record, whose schema is not wrapped. */
function isBareRecord(typ: Type): boolean {
  return isRecord(typ) && !typ.nullable;
}

/**
 * Return the value the parsed structured reply `data` carries for `typ`, checked and in the shape
 * downstream nodes read.
 *
 * A bare record (`isBareRecord`) is `data` itself; any other type is `data`'s wrapper field
 * (`items` for a list, else `value`). The value must not nest past `YIAN_MAX_VALUE_NESTING`; it is
 * then normalized the way Python's pydantic model dumps it (`normalize`) and checked with
 * `buildValueAs`.
 *
 * @throws TypeCheckError reading `structured output[ at <path>]: <reason>`, `path` inside the
 *   value; a missing wrapper field is refused at the root.
 */
export function checkReply(data: JsonObject, typ: Type): Value {
  let value: Value = data;
  if (!isBareRecord(typ)) {
    const field = isListKind(typ.kind) ? "items" : "value";
    if (!Object.hasOwn(data, field)) {
      throw new ReplyCheckError(`structured output: record missing required fields: ["${field}"]`);
    }
    value = data[field]!;
  }
  rejectDeepNesting(value, "structured output");
  try {
    return buildValueAs(typ, normalize(value, typ)).value;
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    const at = error.path.length > 0 ? ` at ${renderValuePath(error.path)}` : "";
    throw new ReplyCheckError(`structured output${at}: ${error.message}`, { path: error.path });
  }
}

/**
 * Return `value` in the shape Python's pydantic model for `typ` dumps it, for `buildValueAs` to
 * check.
 *
 * A record is rebuilt in its declared field order: an absent nullable field becomes `null`, and
 * an unknown field is dropped from an open record but kept in a closed one, so the check refuses
 * it. Union members and everything inside them are closed; a tagged union reads the member its tag
 * names, an untagged one the first member whose normalized value passes the check. `Any`, a
 * field-less `object`, a flat list and a value of the wrong shape are kept as given.
 */
function normalize(value: Value, typ: Type, closed = false): Value {
  if (typ.variants !== undefined) return isValueRecord(value) ? normalizeUnion(value, typ) : value;
  if (isRecord(typ)) {
    if (!isValueRecord(value)) return value;
    const fields = typ.fields!;
    const record: ValueRecord = {};
    for (const [name, field] of fields) {
      if (Object.hasOwn(value, name)) {
        setField(record, name, normalize(value[name]!, field, closed));
      } else if (field.nullable) {
        // A field that is only not required stays absent: `null` would break its type.
        setField(record, name, null);
      }
    }
    if (closed) {
      for (const name of Object.keys(value)) if (!fields.has(name)) setField(record, name, value[name]!);
    }
    return record;
  }
  if (isListKind(typ.kind) && typ.element !== undefined && Array.isArray(value)) {
    return value.map((item) => normalize(item, typ.element!, closed));
  }
  return value;
}

/** `normalize` for a record `value` of the union `typ`; `value` itself when no member fits. */
function normalizeUnion(value: ValueRecord, typ: Type): Value {
  const members = typ.variants!;
  const [tagField] = unionTagFields(typ);
  if (tagField !== undefined) {
    const tag = value[tagField];
    const member =
      typeof tag === "string" ? members.find((each) => each.fields!.get(tagField)!.tags!.has(tag)) : undefined;
    return member === undefined ? value : normalize(value, member, true);
  }
  for (const member of members) {
    const normalized = normalize(value, member, true);
    try {
      buildValueAs(member, normalized);
    } catch (error) {
      if (error instanceof TypeCheckError) continue;
      throw error;
    }
    return normalized;
  }
  return value;
}

/**
 * Return where inside the output of `typ` the generation error `error` is, relative to the output
 * contract.
 *
 * Only the value check's errors have a place: the error's path when it is a contract path, else
 * its parent when that is one (in practice an unknown field of a closed union member, which
 * belongs to the record refusing it), else `null`.
 */
function outputPath(error: Error, typ: Type): ValuePath | null {
  if (!(error instanceof ReplyCheckError)) return null;
  const path = error.path;
  if (isContractPath(path, typ)) return path;
  const parent = path.slice(0, -1);
  return path.length > 0 && isContractPath(parent, typ) ? parent : null;
}

/**
 * Whether `path` names a position of the output contract of `typ`: declared record fields,
 * non-negative list indices, through any union member.
 */
export function isContractPath(path: ValuePath, typ: Type): boolean {
  if (path.length === 0) return true;
  if (typ.variants !== undefined) return typ.variants.some((member) => isContractPath(path, member));
  const [head, ...tail] = path;
  if (isRecord(typ)) {
    const field = typeof head === "string" ? typ.fields!.get(head) : undefined;
    return field !== undefined && isContractPath(tail, field);
  }
  if (isListKind(typ.kind)) {
    if (typeof head !== "number" || !Number.isInteger(head) || head < 0) return false;
    return typ.element === undefined ? tail.length === 0 : isContractPath(tail, typ.element);
  }
  return false;
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
