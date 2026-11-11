/**
 * Read a tool's declared input and return types as yian `Type`s, from its argument and output
 * JSON Schemas.
 *
 * The translation is never narrower than what the schema accepts; it may widen. That keeps a
 * run-start check against the result from reporting a binding as disjoint when the tool would
 * accept it, or an output as disjoint when the tool may return it. Inputs and outputs are both
 * read open, as JSON Schema reads them and as a call's checks do. A schema with no yian
 * equivalent translates to `null`, which means "do not check".
 *
 * Knows about: typesys, `tools/context` and, for types only, `tools/function_tools`.
 */

import {
  canonicalUnionOrder,
  isRecord,
  listKindForElement,
  Type,
  ValueKind,
  type ValueRecord,
} from "../typesys/index.ts";
import { type Bound, type Bounds, typeEquals } from "../typesys/values.ts";
import { ASK_USER } from "./context.ts";
import type { FunctionTool } from "./function_tools.ts";

const SCALAR_KINDS: ReadonlyMap<string, ValueKind> = new Map([
  ["string", ValueKind.STRING],
  ["integer", ValueKind.INTEGER],
  ["number", ValueKind.NUMBER],
  ["boolean", ValueKind.BOOLEAN],
  ["null", ValueKind.NONE],
]);

const REF_PREFIXES = ["#/$defs/", "#/definitions/"] as const;

/** Return the own property `key` of `schema`, or `undefined` when it has none. */
function get(schema: ValueRecord, key: string): unknown {
  return Object.hasOwn(schema, key) ? schema[key] : undefined;
}

/**
 * Translate a JSON Schema into a yian `Type` that accepts at least what it accepts.
 *
 * `closed` says whether an object schema with `properties` describes only those keys; tools read
 * theirs with `false`, as JSON Schema does. An object with `additionalProperties: false` is closed
 * either way. `root` is the document `$ref` pointers
 * resolve against; absent, it is `schema`. Returns `null` when the schema has no yian
 * equivalent or is too big to translate (`MAX_NODES` nodes, counting each path to a def that
 * sits on a `$ref` cycle).
 *
 * @example
 * `{"anyOf": [{"type": "integer", "minimum": 1, "exclusiveMaximum": 10}, {"type": "null"}]}`
 * gives `Optional(int(1..9))` either way. An object with `properties` gives a record when
 * `closed` is `true` and plain `object` when it is `false`.
 */
export function typeFromJsonSchema(
  schema: unknown,
  options: { closed: boolean; root?: unknown },
): Type | null {
  const walk: Walk = {
    closed: options.closed,
    root: options.root ?? schema,
    refTypes: new Map(),
    nodes: 0,
    cycles: 0,
  };
  let typ: Type | null;
  try {
    typ = translate(schema, walk, []);
  } catch (error) {
    if (error === TOO_BIG) return null;
    throw error;
  }
  if (typ === null) return null;
  try {
    typ.validateShape();
  } catch (error) {
    // A RangeError here means the shape is invalid, so "do not check".
    if (error instanceof TypeError || error instanceof RangeError) return null;
    throw error;
  }
  return typ;
}

/** Open translations of frozen schemas, by root and then by schema. */
const INPUT_TYPES = new WeakMap<object, WeakMap<object, Type | null>>();

/**
 * Translate one tool input's or output's JSON Schema as an open schema, or return `null` when it
 * nests too deeply.
 *
 * The result for a frozen schema and root is kept and returned again, since a tool's schemas are
 * read on every call.
 */
export function inputTypeFromJsonSchema(schema: unknown, root?: unknown): Type | null {
  const doc = root ?? schema;
  // yian deep-freezes a tool's schemas, so a frozen one is read as one that won't change.
  const kept = isRecord(schema) && isRecord(doc) && Object.isFrozen(schema) && Object.isFrozen(doc);
  const byRoot = kept ? INPUT_TYPES.get(doc) : undefined;
  if (byRoot?.has(schema as object)) return byRoot.get(schema as object)!;
  let typ: Type | null;
  try {
    typ = typeFromJsonSchema(schema, { closed: false, root });
  } catch (error) {
    // A host or MCP schema can nest deeper than the translator recurses. A RangeError (the
    // runtime's stack overflow, or a shape the IR refuses)
    // leaves the input unchecked rather than failing the run.
    if (!(error instanceof RangeError)) throw error;
    // Kept like any other result: a stack overflow depends on the depth of the first call, but
    // `null` only leaves the input unchecked, so keeping it can only widen the check.
    typ = null;
  }
  if (kept) {
    let map = byRoot;
    if (map === undefined) INPUT_TYPES.set(doc as object, (map = new WeakMap()));
    map.set(schema as object, typ);
  }
  return typ;
}

/** The most schema nodes one translation visits; a bigger schema is not checked. */
const MAX_NODES = 100_000;

/** Thrown when a translation passes `MAX_NODES`. */
const TOO_BIG = Symbol("too big");

/** The state of one top-level translation. */
interface Walk {
  readonly closed: boolean;
  /** The document `$ref` pointers resolve against. */
  readonly root: unknown;
  /** Each `$ref` translated so far whose translation met no cycle, with its type. */
  readonly refTypes: Map<unknown, Type | null>;
  /** The schema nodes visited so far. */
  nodes: number;
  /** The `$ref` cycles met so far. */
  cycles: number;
}

/**
 * Translate one schema node; `refs` holds the `$ref`s being resolved above it.
 *
 * @throws {typeof TOO_BIG} The translation passed `MAX_NODES`.
 */
function translate(schema: unknown, walk: Walk, refs: readonly unknown[]): Type | null {
  if (!isRecord(schema)) return null;
  // Translating a cyclic schema that reaches a def many ways can't reuse those defs' types.
  if (++walk.nodes > MAX_NODES) throw TOO_BIG;
  const closed = walk.closed;

  const sub = (node: unknown, moreRefs: readonly unknown[] = []): Type | null =>
    translate(node, walk, [...refs, ...moreRefs]);

  if (Object.hasOwn(schema, "$ref")) {
    const ref = schema.$ref;
    if (refs.includes(ref)) {
      walk.cycles += 1;
      return null;
    }
    if (walk.refTypes.has(ref)) return walk.refTypes.get(ref)!;
    const target = resolveRef(ref, walk.root);
    if (target === undefined) return null;
    const cycles = walk.cycles;
    const typ = sub(target, [ref]);
    // A type that met no cycle doesn't depend on the refs above it, so another path reaching
    // this def reuses it; without this a def held twice by each def before it costs 2^depth.
    if (walk.cycles === cycles) walk.refTypes.set(ref, typ);
    return typ;
  }

  if (Object.hasOwn(schema, "allOf")) {
    const members = schema.allOf;
    if (Array.isArray(members) && members.length === 1) return sub(members[0]);
    return null;
  }

  for (const keyword of ["anyOf", "oneOf"]) {
    if (Object.hasOwn(schema, keyword)) {
      const members = schema[keyword];
      if (!Array.isArray(members) || members.length === 0) return null;
      return union(members.map((member) => sub(member)));
    }
  }

  const tags = stringTags(schema);
  if (tags !== null) return new Type({ kind: ValueKind.STRING, tags });

  const jsonType = get(schema, "type");
  if (Array.isArray(jsonType)) {
    if (jsonType.length === 0) return null;
    return union(jsonType.map((member) => sub({ ...schema, type: member })));
  }
  if (jsonType === "array") {
    const items = get(schema, "items");
    // Tuple positions have no yian spelling; a list with no element type
    // (`list(Optional(Any))`) accepts every tuple.
    const element =
      Object.hasOwn(schema, "prefixItems") || !isRecord(items) ? null : sub(items);
    if (element === null) return Type.scalar(ValueKind.LIST_ANY);
    return new Type({ kind: listKindForElement(element), element });
  }
  if (jsonType === "object") return object(schema, closed, sub);
  const kind = typeof jsonType === "string" ? SCALAR_KINDS.get(jsonType) : undefined;
  if (kind === undefined) return null;
  if (kind === ValueKind.INTEGER || kind === ValueKind.NUMBER) {
    return new Type({ kind, bounds: bounds(schema, kind) });
  }
  return Type.scalar(kind);
}

/** Return the `$defs` / `definitions` entry `ref` points at, else `undefined`. */
function resolveRef(ref: unknown, root: unknown): unknown {
  if (typeof ref !== "string") return undefined;
  for (const prefix of REF_PREFIXES) {
    if (ref.startsWith(prefix)) {
      const defs = isRecord(root) ? get(root, prefix.slice(2, -1)) : undefined;
      const name = ref.slice(prefix.length).replaceAll("~1", "/").replaceAll("~0", "~");
      return isRecord(defs) ? get(defs, name) : undefined;
    }
  }
  return undefined;
}

/** Return the tags of a string `const` or `enum`, else `null`. */
function stringTags(schema: ValueRecord): ReadonlySet<string> | null {
  const enumValues = get(schema, "enum");
  let values: unknown[];
  if (Object.hasOwn(schema, "const")) {
    values = [schema.const];
  } else if (Array.isArray(enumValues)) {
    values = enumValues;
  } else {
    return null;
  }
  // A non-string or empty value has no `Literal` spelling; falling back to `type` widens
  // instead.
  if (values.length === 0 || values.some((value) => typeof value !== "string" || value === "")) {
    return null;
  }
  return new Set(values as string[]);
}

/**
 * Return one numeric end of a schema: an int (a `bigint` or a safe-integer `number`) as a
 * `bigint`, any other finite `number` as itself, and anything else as `null`.
 */
function endpoint(schema: ValueRecord, keyword: string): Bound {
  const value = get(schema, keyword);
  if (typeof value === "bigint") return value;
  // A safe integer is an int, so `{"minimum": 0}` labels `float(0..)`, not `float(0.0..)`.
  if (Number.isSafeInteger(value)) return BigInt(value as number);
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return null;
}

/** Return the least integer at or above `end`. */
function ceil(end: bigint | number): bigint {
  return typeof end === "bigint" ? end : BigInt(Math.ceil(end));
}

/** Return the greatest integer at or below `end`. */
function floor(end: bigint | number): bigint {
  return typeof end === "bigint" ? end : BigInt(Math.floor(end));
}

/** Return inclusive bounds with `null` for an open end, or `undefined` when both are open. */
function bounds(schema: ValueRecord, kind: ValueKind): Bounds | undefined {
  const low = endpoint(schema, "minimum");
  const high = endpoint(schema, "maximum");
  const exclusiveLow = endpoint(schema, "exclusiveMinimum");
  const exclusiveHigh = endpoint(schema, "exclusiveMaximum");
  let lows: (bigint | number)[];
  let highs: (bigint | number)[];
  if (kind === ValueKind.INTEGER) {
    lows = low === null ? [] : [ceil(low)];
    highs = high === null ? [] : [floor(high)];
    if (exclusiveLow !== null) lows.push(floor(exclusiveLow) + 1n);
    if (exclusiveHigh !== null) highs.push(ceil(exclusiveHigh) - 1n);
  } else {
    // yian float ends are inclusive, so an exclusive end widens to the same value; an end
    // that is not an exact double is dropped, which also widens.
    lows = [low, exclusiveLow].filter(exactFloat);
    highs = [high, exclusiveHigh].filter(exactFloat);
  }
  const result: Bounds = [extreme(lows, (a, b) => a > b), extreme(highs, (a, b) => a < b)];
  if (result[0] === null && result[1] === null) return undefined;
  // Crossed ends accept no number at all; unbounded is the widening fallback.
  if (result[0] !== null && result[1] !== null && result[0] > result[1]) return undefined;
  return result;
}

/**
 * Return the first of `ends` no other end beats by `better`, or `null` when there are none.
 */
function extreme(
  ends: readonly (bigint | number)[],
  better: (a: bigint | number, b: bigint | number) => boolean,
): bigint | number | null {
  let best: bigint | number | null = null;
  // `<` and `>` compare a bigint with a number exactly.
  for (const end of ends) if (best === null || better(end, best)) best = end;
  return best;
}

/** Return whether `value` is a number a float holds exactly. */
function exactFloat(value: Bound): value is bigint | number {
  if (value === null) return false;
  if (typeof value === "number") return true;
  const represented = Number(value);
  return Number.isFinite(represented) && BigInt(represented) === value;
}

/** Return whether `value` is truthy, with empty lists and records false. */
function truthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return Boolean(value);
}

/**
 * Translate an object schema to a record, or to plain `object` when open.
 *
 * A property with no yian type also gives plain `object`, which still rejects non-objects.
 */
function object(
  schema: ValueRecord,
  closed: boolean,
  sub: (node: unknown) => Type | null,
): Type {
  const plain = Type.scalar(ValueKind.OBJECT);
  const properties = get(schema, "properties");
  if (!isRecord(properties)) return plain;
  const extra = get(schema, "additionalProperties");
  // An absent key and an explicit `null` read alike.
  if (extra !== false && !(closed && (extra === undefined || extra === null))) return plain;
  if (truthy(get(schema, "patternProperties"))) return plain;
  const required = Object.hasOwn(schema, "required") ? schema.required : [];
  if (
    !Array.isArray(required) ||
    required.some((name) => typeof name !== "string") ||
    !required.every((name) => Object.hasOwn(properties, name as string))
  ) {
    // A required key outside `properties` would be rejected by the record.
    return plain;
  }
  const fields = new Map<string, Type>();
  for (const [name, prop] of Object.entries(properties)) {
    const field = sub(prop);
    if (field === null) return plain;
    fields.set(name, field);
  }
  return new Type({
    kind: ValueKind.OBJECT,
    fields,
    required: new Set((required as string[]).filter((name) => !fields.get(name)!.nullable)),
  });
}

/**
 * Combine `anyOf` members into one type, or `null` when yian has no spelling.
 *
 * Numeric ranges of one kind combine into their hull.
 */
function union(members: readonly (Type | null)[]): Type | null {
  if (members.some((member) => member === null)) return null;
  const present = members as readonly Type[];
  let nullable = present.some((member) => member.nullable);
  let others: Type[] = [];
  const none = Type.scalar(ValueKind.NONE);
  for (const member of present) {
    if (typeEquals(member, none)) {
      nullable = true;
      continue;
    }
    for (const variant of member.members ?? [member]) {
      const part = variant.replace({ nullable: false });
      if (!others.some((other) => typeEquals(other, part))) others.push(part);
    }
  }
  if (others.length === 0) return none;
  const kinds = new Set(others.map((other) => other.kind));
  if (kinds.size === 1 && (kinds.has(ValueKind.INTEGER) || kinds.has(ValueKind.NUMBER))) {
    // Members of one numeric kind differ only by range; one range covering them all is at
    // least as wide as their union, so the check stays sound.
    others = [hull(others)];
  }
  if (others.length === 1) return others[0]!.replace({ nullable });
  if (others.some((other) => other.kind !== ValueKind.OBJECT)) return null;
  if (others.some((other) => other.fields === undefined)) {
    return new Type({ kind: ValueKind.OBJECT, nullable });
  }
  return new Type({ kind: ValueKind.OBJECT, members: canonicalUnionOrder(others), nullable });
}

/** Return the smallest range of the members' kind that holds every member's range. */
function hull(ranges: readonly Type[]): Type {
  const lows = ranges.map((typ) => (typ.bounds === undefined ? null : typ.bounds[0]));
  const highs = ranges.map((typ) => (typ.bounds === undefined ? null : typ.bounds[1]));
  const low = lows.includes(null) ? null : extreme(lows as (bigint | number)[], (a, b) => a < b);
  const high = highs.includes(null) ? null : extreme(highs as (bigint | number)[], (a, b) => a > b);
  return new Type({
    kind: ranges[0]!.kind,
    bounds: low === null && high === null ? undefined : [low, high],
  });
}

/** The types a tool declares for its inputs and its return value. */
export interface ToolTypes {
  /**
   * The model-visible inputs by name. `null` when the argument schema cannot be read, so no input
   * is checked. A `null` value means that input is not checked.
   */
  readonly inputs: ReadonlyMap<string, Type | null> | null;
  /** The names of the inputs a call must pass. */
  readonly required: ReadonlySet<string>;
  /**
   * The type of an input not in `inputs`, from `additionalProperties`: `undefined` when the tool
   * takes no such input, and `null` when it takes any (`true`, or a schema yian can't translate).
   */
  readonly extra?: Type | null;
  /**
   * The return type, read from `outputJsonSchema` as `inputTypeFromJsonSchema` reads a schema (the
   * reading a call's result check uses), or `null` when it is not checked: the tool has no output
   * schema, or yian can't translate it.
   */
  readonly output: Type | null;
}

/**
 * Read the declared input and return types of `tool` from its `parametersJsonSchema` and
 * `outputJsonSchema`.
 *
 * Each part is `null` where it cannot be read without risking a narrower type than the tool
 * accepts or returns.
 */
export function toolTypes(tool: FunctionTool): ToolTypes {
  const output = outputType(tool);
  const schema = tool.parametersJsonSchema;
  const properties = get(schema, "properties");
  if (!isRecord(properties)) return { inputs: null, required: new Set(), output };
  const inputs = new Map<string, Type | null>();
  for (const [name, prop] of Object.entries(properties)) {
    // The run cleans `ask_user`'s `options` before checking it, so any shape is accepted.
    const unchecked = tool.name === ASK_USER && name === "options";
    inputs.set(name, unchecked ? null : inputTypeFromJsonSchema(prop, schema));
  }
  const required = get(schema, "required");
  const names = Array.isArray(required) ? required : [];
  // As a call's argument check reads it: only `true` or a schema lets other keys in.
  const additional = get(schema, "additionalProperties");
  const extra =
    additional === true ? null : isRecord(additional) ? inputTypeFromJsonSchema(additional, schema) : undefined;
  return {
    inputs,
    required: new Set(names.filter((name): name is string => typeof name === "string")),
    output,
    ...(extra === undefined ? {} : { extra }),
  };
}

/**
 * Translate `tool`'s output JSON Schema as a call's result check reads it, or return `null` when
 * it has none.
 */
function outputType(tool: FunctionTool): Type | null {
  // `??`: a tool built by an older copy of yian has no `outputJsonSchema` field.
  const schema = tool.outputJsonSchema ?? null;
  return schema === null ? null : inputTypeFromJsonSchema(schema);
}
