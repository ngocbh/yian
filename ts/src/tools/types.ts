/**
 * Read a tool's declared input and return types as yian `Type`s.
 *
 * The translation is never narrower than what the schema accepts; it may widen. That keeps a
 * run-start check against the result from reporting a binding as disjoint when the tool would
 * accept it. A schema with no yian equivalent translates to `null`, which means "do not check".
 *
 * Knows about: typesys and `tools/context`. Never imports: nodes, compile, compose, runtime, the run boundary, or
 * CLI.
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
 * `closed` says whether an object schema with `properties` describes only those keys: `false`
 * for tool inputs, which accept extra keys, and `true` for tool outputs. An object with
 * `additionalProperties: false` is closed either way. `root` is the document `$ref` pointers
 * resolve against; absent, it is `schema`. Returns `null` when the schema has no yian
 * equivalent.
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
  const typ = translate(schema, options.closed, options.root ?? schema, []);
  if (typ === null) return null;
  try {
    typ.validateShape();
  } catch (error) {
    // A RangeError here stands for Python's ValueError: the shape is invalid, so "do not check".
    if (error instanceof TypeError || error instanceof RangeError) return null;
    throw error;
  }
  return typ;
}

/**
 * Translate one tool input's JSON Schema as an open schema, or return `null` when it nests
 * too deeply.
 */
export function inputTypeFromJsonSchema(schema: unknown, root?: unknown): Type | null {
  try {
    return typeFromJsonSchema(schema, { closed: false, root });
  } catch (error) {
    // A host or MCP schema can nest deeper than the translator recurses; the runtime's stack
    // overflow is a RangeError, standing for Python's RecursionError. Such an input is left
    // unchecked rather than failing the run.
    if (error instanceof RangeError) return null;
    throw error;
  }
}

/** Translate one schema node; `refs` holds the `$ref`s being resolved above it. */
function translate(
  schema: unknown,
  closed: boolean,
  root: unknown,
  refs: readonly unknown[],
): Type | null {
  if (!isRecord(schema)) return null;

  const sub = (node: unknown, moreRefs: readonly unknown[] = []): Type | null =>
    translate(node, closed, root, [...refs, ...moreRefs]);

  if (Object.hasOwn(schema, "$ref")) {
    const ref = schema.$ref;
    if (refs.includes(ref)) return null;
    const target = resolveRef(ref, root);
    return target === undefined ? null : sub(target, [ref]);
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
  // A safe integer is an int, so `{"minimum": 0}` labels `float(0..)` as Python's int does.
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
 * Return the first of `ends` no other end beats by `better`, or `null` when there are none,
 * as Python's `max`/`min` pick.
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

/** Return whether `value` is true as Python reads it: empty containers are false. */
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
  // An absent key and an explicit `null` both read as Python's `None`.
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
    for (const variant of member.variants ?? [member]) {
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
  return new Type({ kind: ValueKind.OBJECT, variants: canonicalUnionOrder(others), nullable });
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
  inputs: ReadonlyMap<string, Type | null> | null;
  /** The names of the inputs a call must pass. */
  required: ReadonlySet<string>;
  /** The return type, or `null` when it is not checked. */
  output: Type | null;
}

/**
 * Read the declared input and return types of `tool` from its `parametersJsonSchema`.
 *
 * Each part is `null` where it cannot be read without risking a narrower type than the tool
 * accepts or returns.
 */
export function toolTypes(tool: FunctionTool): ToolTypes {
  const schema = tool.parametersJsonSchema;
  const properties = get(schema, "properties");
  if (!isRecord(properties)) return { inputs: null, required: new Set(), output: null };
  const inputs = new Map<string, Type | null>();
  for (const [name, prop] of Object.entries(properties)) {
    // The run cleans `ask_user`'s `options` before checking it, so any shape is accepted.
    const unchecked = tool.name === ASK_USER && name === "options";
    inputs.set(name, unchecked ? null : inputTypeFromJsonSchema(prop, schema));
  }
  const required = get(schema, "required");
  const names = Array.isArray(required) ? required : [];
  return {
    inputs,
    required: new Set(names.filter((name): name is string => typeof name === "string")),
    output: null,
  };
}
