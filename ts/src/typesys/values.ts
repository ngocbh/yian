/**
 * The typed value system's type half: the type vocabulary `types.ts` builds on.
 *
 * `ValueKind` is the closed set of storage tags, and `Type` the resolved structure of a
 * declared type (records, `Literal` tags, typed lists, numeric ranges and unions) that
 * the write boundary checks values against, with its canonical labels, union member
 * order and overlap check.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { parseDate, parseDatetime } from "./temporal.ts";

/**
 * A value violates a declared type at one source-neutral structural path.
 *
 * `path` is relative to the checked type: strings name record fields and numbers are
 * actual list indices. Type-language errors and root mismatches use the empty path.
 */
export class TypeCheckError extends RangeError {
  override name = "TypeCheckError";
  /** The frozen record-field/list-index path to the rejected value. */
  readonly path: readonly (string | number)[];

  constructor(
    message?: string,
    options: ErrorOptions & { path?: readonly (string | number)[] } = {},
  ) {
    super(message, options);
    this.path = Object.freeze([...(options.path ?? [])]);
  }
}

// --------------------------------------------------------------------------------------- //
// Type vocabulary
// --------------------------------------------------------------------------------------- //

/**
 * Closed type vocabulary for runtime values.
 *
 * Each value IS the one surface vocabulary: at once the serialized discriminator tag,
 * the author-written `type:` token and the displayed name (`str`, never `string`).
 *
 * Every kind but `ANY` is a storage tag and wire discriminator. `ANY` is type-only: no
 * carrier stores it, and an `Any` value persists under its natural kind (`str`,
 * `object`, `list[Any]`, ...). The list kinds keep the bracket storage spelling
 * (`list[str]`); type labels spell them `list(str)`. `date` and `datetime` are stored
 * as strings and never inferred; `file` is reserved: never inferred, not authorable yet.
 */
export const ValueKind = {
  NONE: "None",
  STRING: "str",
  INTEGER: "int",
  NUMBER: "float",
  DATE: "date",
  DATETIME: "datetime",
  BOOLEAN: "bool",
  OBJECT: "object",
  FILE: "file",
  ANY: "Any",
  LIST_ANY: "list[Any]",
  LIST_STRING: "list[str]",
  LIST_INTEGER: "list[int]",
  LIST_NUMBER: "list[float]",
  LIST_BOOLEAN: "list[bool]",
  LIST_OBJECT: "list[object]",
} as const;

export type ValueKind = (typeof ValueKind)[keyof typeof ValueKind];

const VALUE_KINDS: ReadonlySet<string> = new Set(Object.values(ValueKind));

/** Return whether `kind` is one of the six `list[...]` kinds. */
export function isListKind(kind: ValueKind): boolean {
  return LIST_ELEMENT_KIND.has(kind);
}

/** Return the scalar element kind of a list kind (`null` for `LIST_ANY` and scalars). */
export function listElementKind(kind: ValueKind): ValueKind | null {
  return LIST_ELEMENT_KIND.get(kind) ?? null;
}

// --------------------------------------------------------------------------------------- //
// Lookup tables
// --------------------------------------------------------------------------------------- //

const LIST_ELEMENT_KIND: ReadonlyMap<ValueKind, ValueKind | null> = new Map([
  [ValueKind.LIST_ANY, null],
  [ValueKind.LIST_STRING, ValueKind.STRING],
  [ValueKind.LIST_INTEGER, ValueKind.INTEGER],
  [ValueKind.LIST_NUMBER, ValueKind.NUMBER],
  [ValueKind.LIST_BOOLEAN, ValueKind.BOOLEAN],
  [ValueKind.LIST_OBJECT, ValueKind.OBJECT],
]);

// --------------------------------------------------------------------------------------- //
// Structural type (records / variants / typed lists)
// --------------------------------------------------------------------------------------- //

/** One inclusive numeric range end: a `bigint` or `number`, or `null` for an open end. */
export type Bound = bigint | number | null;

/** A numeric type's inclusive `[minimum, maximum]`. */
export type Bounds = readonly [Bound, Bound];

/**
 * The properties a `Type` is built from. An absent refinement is left out (`undefined`), never
 * `null`: decode a JSON `null` to `undefined` before building a `Type`.
 */
export interface TypeInit {
  readonly kind: ValueKind;
  readonly fields?: ReadonlyMap<string, Type> | undefined;
  readonly required?: ReadonlySet<string> | undefined;
  readonly tags?: ReadonlySet<string> | undefined;
  readonly element?: Type | undefined;
  readonly nullable?: boolean | undefined;
  readonly bounds?: Bounds | undefined;
  readonly variants?: readonly Type[] | undefined;
}

/**
 * Resolved runtime structure of a declared type; drives the write-boundary check.
 *
 * `kind` is the storage tag (what the value persists as). The optional properties
 * refine validation beyond the bare tag:
 *   - record:  kind=OBJECT, fields={name: Type}, required={names}
 *   - variant: kind=STRING, tags={labels} (a `Literal(...)` of strings)
 *   - list:    kind=LIST_*, element=<element Type>
 *   - number:  kind=INTEGER/NUMBER, bounds=[inclusive minimum, maximum]; `null` for an
 *              open end
 *   - union:   kind=OBJECT, variants=[record member, ...] (a `Union(...)` of records)
 * A refined specialized list uses the element's matching storage tag; `LIST_ANY` is the
 * generic carrier and may refine any element type. A plain scalar or flat list uses only
 * `kind` (see `Type.scalar`). A `LIST_ANY` with no `element` is a list whose items
 * weren't proven (a tool's untyped array, a mixed or empty literal); it is labelled
 * `list(Optional(Any))` and its items read as `Optional(Any)`. `kind=ANY` is type-only:
 * an `Any` value is stored under its natural kind.
 * Despite the name, `variants` holds union members, not the string tags of a "variant"
 * above. Members are non-nullable records; a nullable union sets `nullable` on the
 * union itself. `nullable` marks an `Optional(X)` slot, which accepts null.
 * A `Type` must not be changed after it is built (its `fields` map included), so copies
 * of a flow share it.
 */
export class Type {
  readonly kind: ValueKind;
  readonly fields: ReadonlyMap<string, Type> | undefined;
  readonly required: ReadonlySet<string> | undefined;
  readonly tags: ReadonlySet<string> | undefined;
  readonly element: Type | undefined;
  readonly nullable: boolean;
  readonly bounds: Bounds | undefined;
  readonly variants: readonly Type[] | undefined;

  /** Build a type from `init` as given; `validateShape` checks it. */
  constructor(init: TypeInit) {
    this.kind = init.kind;
    this.fields = init.fields;
    this.required = init.required;
    this.tags = init.tags;
    this.element = init.element;
    // Not `??`: a forged `null` must reach validateShape's bool check.
    this.nullable = init.nullable === undefined ? false : init.nullable;
    this.bounds = init.bounds;
    this.variants = init.variants;
  }

  /** Build a plain `Type` carrying only a storage tag, without refinements. */
  static scalar(kind: ValueKind): Type {
    return new Type({ kind });
  }

  /** Return a new `Type` with the properties in `changes` replaced. */
  replace(changes: Partial<TypeInit>): Type {
    return new Type({ ...this, ...changes });
  }

  /**
   * Require recursively well-formed structural type metadata.
   *
   * This is an execution-boundary invariant check, not value validation. It catches a
   * corrupted or manually assembled type before value rejection is classified against
   * that contract.
   *
   * @throws TypeError when a property or nested carrier has the wrong runtime type.
   * @throws RangeError when refinements conflict, cannot be canonicalized, name an
   *   absent record field, use incompatible storage, or form a cycle.
   */
  validateShape(): void {
    validateTypeShape(this, new Set(), new Set());
  }
}

/**
 * Return how many record and list levels a type nests.
 *
 * Each record (a type with `fields`) and each list kind is one level, on top of the
 * deepest type inside it; a union counts as its deepest member and a scalar as 0. The
 * walk uses an explicit stack, so a deep type needs no recursion, and measures a `Type`
 * shared by several fields once. `typ`'s graph must be acyclic.
 *
 * Example: 0 for `int`, 1 for `list(int)`, 2 for a record holding a record.
 */
export function typeLevels(typ: Type): number {
  const levels = new Map<Type, number>();
  const stack: [Type, boolean][] = [[typ, false]];
  while (stack.length > 0) {
    const [node, expanded] = stack.pop()!;
    if (levels.has(node)) continue;
    const children = [
      ...(node.fields?.values() ?? []),
      ...(node.element !== undefined ? [node.element] : []),
      ...(node.variants ?? []),
    ];
    if (!expanded) {
      stack.push([node, true]);
      for (const child of children) if (!levels.has(child)) stack.push([child, false]);
      continue;
    }
    const own = node.fields !== undefined || isListKind(node.kind) ? 1 : 0;
    let deepest = 0;
    for (const child of children) deepest = Math.max(deepest, levels.get(child)!);
    levels.set(node, own + deepest);
  }
  return levels.get(typ)!;
}

/** Validate one recursive type graph while permitting shared child objects. */
function validateTypeShape(typ: unknown, active: Set<Type>, validated: Set<Type>): void {
  if (!(typ instanceof Type)) throw new TypeError("nested type must be a Type");
  if (active.has(typ)) throw new RangeError("type refinements must form an acyclic graph");
  if (validated.has(typ)) return;
  if (!VALUE_KINDS.has(typ.kind)) throw new TypeError("type kind must be a ValueKind");
  if (typeof typ.nullable !== "boolean") throw new TypeError("type nullable must be a bool");
  if (typ.bounds !== undefined) validateNumericBounds(typ.kind, typ.bounds);

  active.add(typ);
  try {
    const fields = typ.fields;
    if (fields !== undefined) {
      if (!(fields instanceof Map)) throw new TypeError("type fields must be a Map or undefined");
      if (typ.kind !== ValueKind.OBJECT) {
        throw new RangeError("only object types may declare record fields");
      }
      for (const [name, fieldType] of fields) {
        if (!isNonEmptyString(name)) {
          throw new TypeError("type field names must be non-empty strings");
        }
        validateTypeMetadataString(name);
        validateTypeShape(fieldType, active, validated);
      }
    }

    const required = typ.required;
    if (required !== undefined) {
      if (!(required instanceof Set) || ![...required].every(isNonEmptyString)) {
        throw new TypeError(
          "type required fields must be a Set of non-empty strings or undefined",
        );
      }
      for (const name of required) validateTypeMetadataString(name);
      if (fields === undefined) {
        throw new RangeError("type required fields need a record field mapping");
      }
      const unknown = [...required].filter((name) => !fields.has(name)).sort(compareCodeUnits);
      if (unknown.length > 0) {
        throw new RangeError(`type requires unknown record fields: ${JSON.stringify(unknown)}`);
      }
    }

    const tags = typ.tags;
    if (tags !== undefined) {
      if (!(tags instanceof Set) || ![...tags].every(isNonEmptyString)) {
        throw new TypeError("type tags must be a Set of non-empty strings or undefined");
      }
      for (const tag of tags) validateTypeMetadataString(tag);
      if (typ.kind !== ValueKind.STRING) {
        throw new RangeError("only string types may declare variant tags");
      }
    }

    const element = typ.element;
    if (element !== undefined) {
      if (!isListKind(typ.kind)) {
        throw new RangeError("only list types may declare an element type");
      }
      validateTypeShape(element, active, validated);
      const expectedKind = listKindForElement(element);
      if (typ.kind !== ValueKind.LIST_ANY && typ.kind !== expectedKind) {
        throw new RangeError(
          `list storage kind ${reprText(typ.kind)} does not match ` +
            `element type; expected ${reprText(expectedKind)}`,
        );
      }
    }

    const variants = typ.variants;
    if (variants !== undefined) {
      if (!Array.isArray(variants)) {
        throw new TypeError("type variants must be an array or undefined");
      }
      if (typ.kind !== ValueKind.OBJECT) {
        throw new RangeError("only object types may declare union variants");
      }
      if (variants.length < 2) throw new RangeError("a union needs at least two members");
      for (const member of variants) {
        validateTypeShape(member, active, validated);
        if (member.fields === undefined) {
          throw new RangeError("every union member must be a record type");
        }
        if (member.nullable) throw new RangeError("union members must not be nullable");
      }
    }

    const refinements = [fields, tags, element, typ.bounds, variants].filter(
      (value) => value !== undefined,
    ).length;
    if (refinements > 1) {
      throw new RangeError("a type may declare only one structural refinement");
    }
  } finally {
    active.delete(typ);
  }
  validated.add(typ);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value !== "";
}

/**
 * Require inclusive bounds with at least one end; `null` is an open end.
 *
 * Present ends must be exact numbers (`bigint` for `int`; `bigint` or `number` for
 * `float`, lossless and finite as a float), and ordered when both are present.
 *
 * @throws TypeError when `bounds` is not a two-item array or an end has the wrong type.
 * @throws RangeError when `kind` takes no bounds, both ends are open, a float end is not
 *   finite or lossless, or the ends are out of order.
 */
export function validateNumericBounds(kind: ValueKind, bounds: Bounds): void {
  if (kind !== ValueKind.INTEGER && kind !== ValueKind.NUMBER) {
    throw new RangeError("only int and float types may declare bounds");
  }
  if (!Array.isArray(bounds) || bounds.length !== 2) {
    throw new TypeError("type bounds must be a two-item array");
  }
  const endpoints = bounds.filter((endpoint) => endpoint !== null);
  if (endpoints.length === 0) throw new RangeError("type bounds need at least one endpoint");
  const exact = (endpoint: unknown) =>
    typeof endpoint === "bigint" || (kind === ValueKind.NUMBER && typeof endpoint === "number");
  if (!endpoints.every(exact)) {
    throw new TypeError(`${kind} bounds require exact numeric endpoints, excluding bool`);
  }
  if (kind === ValueKind.NUMBER) {
    for (const endpoint of endpoints) {
      const represented = Number(endpoint);
      if (!Number.isFinite(represented)) {
        throw new RangeError("float bounds require finite endpoints");
      }
      if (typeof endpoint === "bigint" && BigInt(represented) !== endpoint) {
        throw new RangeError("float bounds require lossless float endpoints");
      }
    }
  }
  // `>` compares a bigint with a number exactly.
  if (endpoints.length === 2 && endpoints[0]! > endpoints[1]!) {
    throw new RangeError("type bounds must be ordered from minimum to maximum");
  }
}

/** Require a type-owned name to be representable in canonical JSON. */
function validateTypeMetadataString(value: string): void {
  if (!value.isWellFormed()) {
    throw new RangeError("type metadata contains a Unicode surrogate");
  }
}

/**
 * Return the list storage kind the type resolver picks for an element type.
 *
 * A `Literal` element stores as `list[str]`, a record or union element as
 * `list[object]`, a scalar element as its own list kind, and anything else (`Any` and
 * `Optional(Any)` included) as `list[Any]`.
 */
export function listKindForElement(element: Type): ValueKind {
  let semanticKind = element.kind;
  if (element.tags !== undefined) {
    semanticKind = ValueKind.STRING;
  } else if (element.fields !== undefined || element.variants !== undefined) {
    semanticKind = ValueKind.OBJECT;
  }
  for (const [listKind, elementKind] of LIST_ELEMENT_KIND) {
    if (elementKind === semanticKind) return listKind;
  }
  return ValueKind.LIST_ANY;
}

/**
 * Return whether two types are structurally equal, as Python's dataclass `==` compares
 * them: maps and sets by content (a record's field order doesn't count), bounds by
 * numeric value (`1n` equals `1`), union members in order.
 */
export function typeEquals(a: Type, b: Type): boolean {
  if (a === b) return true;
  return (
    a.kind === b.kind &&
    a.nullable === b.nullable &&
    optionalEquals(a.fields, b.fields, mapEquals) &&
    optionalEquals(a.required, b.required, setEquals) &&
    optionalEquals(a.tags, b.tags, setEquals) &&
    optionalEquals(a.element, b.element, typeEquals) &&
    optionalEquals(
      a.bounds,
      b.bounds,
      (x, y) => boundEquals(x[0], y[0]) && boundEquals(x[1], y[1]),
    ) &&
    optionalEquals(
      a.variants,
      b.variants,
      (x, y) => x.length === y.length && x.every((member, i) => typeEquals(member, y[i]!)),
    )
  );
}

/** Return whether `a` and `b` are both absent, or both present and `equals`. */
function optionalEquals<T>(
  a: T | undefined,
  b: T | undefined,
  equals: (a: T, b: T) => boolean,
): boolean {
  return a === undefined || b === undefined ? a === b : equals(a, b);
}

function mapEquals(a: ReadonlyMap<string, Type>, b: ReadonlyMap<string, Type>): boolean {
  if (a.size !== b.size) return false;
  for (const [name, typ] of a) {
    const other = b.get(name);
    if (other === undefined || !typeEquals(typ, other)) return false;
  }
  return true;
}

function setEquals(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  return a.size === b.size && [...a].every((item) => b.has(item));
}

function boundEquals(a: Bound, b: Bound): boolean {
  // Loose `==` compares a bigint with a number by value; `null` equals only `null` here.
  return a === null || b === null ? a === b : a == b;
}

// --------------------------------------------------------------------------------------- //
// Type labels
// --------------------------------------------------------------------------------------- //

/**
 * Render one deterministic, unambiguous label for a structural type.
 *
 * Record keys use compact JSON quoting; variant tags print bare when they read back as
 * names, else JSON-quoted (`Literal(go, "no go")`). Structurally refined lists render
 * from their element; a valid flat-list type renders from its storage kind
 * (`list(str)`), and a `LIST_ANY` with no element as `list(Optional(Any))`. A union
 * lists its members in canonical order.
 *
 * Example: `Optional(int(1..12))`.
 *
 * @throws RangeError or TypeError when a numeric range's bounds are malformed.
 */
export function canonicalTypeLabel(typ: Type): string {
  return labels(typ)[0];
}

/**
 * Return `[label, order key]` for `typ`.
 *
 * The label is the parenthesized spelling; its variant tags print bare when they read
 * back as names, else JSON-quoted. The order key is the same type in the older bracket
 * spelling (`Optional[int[1,12]]`, `Literal["go","no_go"]`); union members sort by it,
 * so member order, JSON Schema `anyOf` order and manifest fingerprints did not move
 * when the spelling changed. Both come from one walk so nested unions stay linear.
 */
function labels(typ: Type): [string, string] {
  const [label, key] = canonicalNonnullableLabel(typ);
  return typ.nullable ? [`Optional(${label})`, `Optional[${key}]`] : [label, key];
}

/** Return `[label, order key]` for `typ` without its outer nullable wrapper. */
function canonicalNonnullableLabel(typ: Type): [string, string] {
  if (typ.variants !== undefined) {
    const ordered = orderedMemberLabels(typ.variants);
    return [
      "Union(" + ordered.map(([, label]) => label).join(",") + ")",
      "Union[" + ordered.map(([, , key]) => key).join(",") + "]",
    ];
  }

  if (typ.fields !== undefined) {
    const fields = typ.fields;
    const pairs = [...fields.keys()]
      .sort(compareCodeUnits)
      .map((name) => [JSON.stringify(name), labels(fields.get(name)!)] as const);
    return [
      "{" + pairs.map(([name, [label]]) => `${name}:${label}`).join(",") + "}",
      "{" + pairs.map(([name, [, key]]) => `${name}:${key}`).join(",") + "}",
    ];
  }

  if (typ.tags !== undefined) {
    const tags = [...typ.tags].sort(compareCodeUnits);
    const key = "Literal[" + tags.map((tag) => JSON.stringify(tag)).join(",") + "]";
    return [literalLabel(tags), key];
  }

  if (isListKind(typ.kind)) {
    if (typ.element === undefined) {
      if (typ.kind === ValueKind.LIST_ANY) {
        // Items not proven may be any JSON value, null included.
        return ["list(Optional(Any))", "list[Optional[Any]]"];
      }
      return [kindLabel(typ.kind), typ.kind];
    }
    const [label, key] = labels(typ.element);
    return [`list(${label})`, `list[${key}]`];
  }

  if (typ.bounds !== undefined) {
    validateNumericBounds(typ.kind, typ.bounds);
    // An open end renders empty in the label (`int(0..)`) and as null in the key.
    const [low, high] = typ.bounds;
    const keyEnd = (end: Bound) => (end === null ? "null" : boundText(end));
    return [
      `${typ.kind}(${boundText(low)}..${boundText(high)})`,
      `${typ.kind}[${keyEnd(low)},${keyEnd(high)}]`,
    ];
  }

  return [typ.kind, typ.kind];
}

/**
 * Spell a list storage kind as a type for messages, e.g. `list(str)` for `LIST_STRING`.
 *
 * `LIST_ANY` reads `list(Optional(Any))`, since its items may be any JSON value.
 */
function kindLabel(kind: ValueKind): string {
  const element = listElementKind(kind);
  return `list(${element === null ? "Optional(Any)" : element})`;
}

/**
 * Spell one numeric range end as type text: a `bigint` as its digits, a `number` as
 * Python's `repr` of the float (`1.0`, `1e-05`), and an open end as `""`.
 *
 * Example: `boundText(1n)` is `1`, `boundText(1e16)` is `1e+16`.
 */
export function boundText(end: Bound): string {
  if (end === null) return "";
  return typeof end === "bigint" ? end.toString() : pythonFloatRepr(end);
}

/**
 * Return Python's `repr` of a float: the shortest round-trip digits, positional when
 * the decimal exponent is in `-4..15`, else exponent form with at least two exponent
 * digits (`1e-05`, `1.5e+16`). A non-finite value prints as JavaScript spells it.
 */
function pythonFloatRepr(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  // `toExponential()` with no argument gives the shortest digits that round-trip.
  const [mantissa, exponentText] = Math.abs(x).toExponential().split("e") as [string, string];
  const exponent = Number(exponentText);
  const digits = mantissa.replace(".", "");
  const sign = x < 0 ? "-" : "";
  if (exponent < -4 || exponent > 15) {
    const exponentDigits = String(Math.abs(exponent)).padStart(2, "0");
    return `${sign}${mantissa}e${exponent < 0 ? "-" : "+"}${exponentDigits}`;
  }
  if (exponent < 0) return `${sign}0.${"0".repeat(-exponent - 1)}${digits}`;
  const whole = digits.slice(0, exponent + 1).padEnd(exponent + 1, "0");
  return `${sign}${whole}.${digits.slice(exponent + 1) || "0"}`;
}

/**
 * The type tokenizer's NAME token, Python's `[^\W\d]\w*` in Unicode properties; a
 * `Literal` member printed bare must match it. Use it with the `u` flag.
 */
export const NAME_PATTERN = "[\\p{L}\\p{Nl}\\p{No}_][\\p{L}\\p{N}_]*";
const NAME_TOKEN = new RegExp(`^(?:${NAME_PATTERN})$`, "u");
/** Python's identifier, as `str.isidentifier()` checks it. Use it with the `u` flag. */
export const IDENTIFIER_PATTERN = "[\\p{XID_Start}_]\\p{XID_Continue}*";
const IDENTIFIER = new RegExp(`^(?:${IDENTIFIER_PATTERN})$`, "u");

/** Names the type parser reads as constants, never as `Literal` members. */
export const BUILTIN_CONSTANTS: ReadonlySet<string> = new Set(["True", "False", "None"]);

/**
 * Spell a `Literal` type from its members, in the given order.
 *
 * A member prints bare when it reads back as a name, else JSON-quoted; members are
 * separated by `", "`. The type parser reads the result back.
 *
 * Example: `literalLabel(["go", "no go", "True"])` gives `Literal(go, "no go", "True")`.
 */
export function literalLabel(tags: Iterable<string>): string {
  return "Literal(" + Array.from(tags, literalMember).join(", ") + ")";
}

/** Spell one `Literal` member bare when it reads back as a name, else quoted. */
function literalMember(tag: string): string {
  // The tokenizer's NAME and `str.isidentifier` disagree on a few Unicode characters;
  // a bare member must pass both to read back as the same single name.
  if (NAME_TOKEN.test(tag) && IDENTIFIER.test(tag) && !BUILTIN_CONSTANTS.has(tag)) {
    return tag;
  }
  return JSON.stringify(tag);
}

// --------------------------------------------------------------------------------------- //
// Unions of records
// --------------------------------------------------------------------------------------- //

/** Return the one value of a single-value `Literal`, else `null`. */
function singleTag(typ: Type): string | null {
  if (typ.tags === undefined || typ.tags.size !== 1) return null;
  return typ.tags.values().next().value!;
}

/** Return the tag fields shared by `members`, sorted by name. */
function membersTagFields(members: readonly Type[]): string[] {
  const first = members[0];
  if (first?.fields === undefined || members.some((member) => member.fields === undefined)) {
    return [];
  }
  const qualifying: string[] = [];
  for (const name of [...first.fields.keys()].sort(compareCodeUnits)) {
    const values = new Set<string>();
    const isTag = members.every((member) => {
      const fieldType = member.fields!.get(name);
      if (fieldType === undefined || fieldType.nullable || !member.required?.has(name)) {
        return false;
      }
      const tag = singleTag(fieldType);
      if (tag === null) return false;
      values.add(tag);
      return true;
    });
    if (isTag && values.size === members.length) qualifying.push(name);
  }
  return qualifying;
}

/**
 * Return the fields that tell a union's members apart, sorted by name.
 *
 * A field is a tag when every member declares it as required, not nullable, and a
 * single-value `Literal`, and the values are pairwise distinct. Any tag field selects
 * the same member. A type that is not a union has no tag fields.
 *
 * Example: a union of `{action: Literal(buy), qty: int}` and
 * `{action: Literal(sell), price: float}` gives `["action"]`.
 */
export function unionTagFields(typ: Type): readonly string[] {
  if (typ.variants === undefined) return [];
  return membersTagFields(typ.variants);
}

/**
 * Return union members in canonical order.
 *
 * Members with a tag field sort by the value of the first tag field by name; otherwise
 * they sort by their canonical label in the bracket spelling (`Optional[int]`), which
 * predates the parenthesized one.
 */
export function canonicalUnionOrder(members: Iterable<Type>): Type[] {
  const all = [...members];
  const tagFields = membersTagFields(all);
  if (tagFields.length > 0) {
    // Tag order reads best in labels and manifests, and needs no member labels.
    const field = tagFields[0]!;
    return all.sort((a, b) => compareCodeUnits(memberTag(a, field), memberTag(b, field)));
  }
  return orderedMemberLabels(all).map(([member]) => member);
}

/** Return the single tag a tagged union member holds in `field`. */
function memberTag(member: Type, field: string): string {
  return singleTag(member.fields!.get(field)!)!;
}

/**
 * Return `[member, label, order key]` triples in canonical order, labelling each member
 * once.
 *
 * Labelling once keeps nested untagged unions linear: their order depends on the labels,
 * so ordering and rendering share them. Untagged members sort by their bracket-spelled
 * order key (see `labels`).
 */
function orderedMemberLabels(members: readonly Type[]): [Type, string, string][] {
  const labelled = members.map((member): [Type, string, string] => [member, ...labels(member)]);
  const tagFields = membersTagFields(members);
  if (tagFields.length > 0) {
    const field = tagFields[0]!;
    return labelled.sort(([a], [b]) => compareCodeUnits(memberTag(a, field), memberTag(b, field)));
  }
  return labelled.sort(([, , a], [, , b]) => compareCodeUnits(a, b));
}

/** Whether `null` satisfies `typ`. */
function admitsNone(typ: Type): boolean {
  return typ.nullable || typ.kind === ValueKind.NONE;
}

/** Whether `typ` is `object` with no record or union refinement. */
function isPlainObject(typ: Type): boolean {
  return typ.kind === ValueKind.OBJECT && typ.fields === undefined && typ.variants === undefined;
}

/** Whether two closed records can match the same value. */
function recordsOverlap(a: Type, b: Type): boolean {
  const aFields = a.fields!;
  const bFields = b.fields!;
  const aRequired = a.required ?? new Set<string>();
  const bRequired = b.required ?? new Set<string>();
  if (![...aRequired].every((name) => bFields.has(name))) return false;
  if (![...bRequired].every((name) => aFields.has(name))) return false;
  return [...new Set([...aRequired, ...bRequired])].every((name) =>
    typesOverlap(aFields.get(name)!, bFields.get(name)!),
  );
}

/** Whether two numeric types' inclusive intervals share a point; `null` is an open end. */
function boundsIntersect(a: Type, b: Type): boolean {
  const [aLow, aHigh] = a.bounds ?? [null, null];
  const [bLow, bHigh] = b.bounds ?? [null, null];
  // `<=` compares a bigint with a number exactly.
  return (
    (aLow === null || bHigh === null || aLow <= bHigh) &&
    (bLow === null || aHigh === null || bLow <= aHigh)
  );
}

/** Whether `tag` is a valid `date` or `datetime` string, as `kind` says. */
function matchesTemporal(kind: ValueKind, tag: string): boolean {
  try {
    if (kind === ValueKind.DATE) parseDate(tag);
    else parseDatetime(tag);
  } catch (error) {
    if (error instanceof RangeError) return false;
    throw error;
  }
  return true;
}

const TEXT_KINDS: ReadonlySet<ValueKind> = new Set([
  ValueKind.STRING,
  ValueKind.DATE,
  ValueKind.DATETIME,
]);
const NUMERIC_KINDS: ReadonlySet<ValueKind> = new Set([ValueKind.INTEGER, ValueKind.NUMBER]);
const OPTIONAL_STR = new Type({ kind: ValueKind.STRING, nullable: true });

/**
 * The record a file's JSON form satisfies; a record overlapping it overlaps `file`. Its
 * fields also type a file's dotted reads.
 */
export const FILE_FORM = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["uri", Type.scalar(ValueKind.STRING)],
    ["mime", OPTIONAL_STR],
    ["name", OPTIONAL_STR],
  ]),
  required: new Set(["uri", "mime", "name"]),
});

/**
 * Whether some value could satisfy both `a` and `b`.
 *
 * The check is conservative: it may say two types overlap when no value satisfies both,
 * but never the reverse. It decides whether an untagged union's members can be told
 * apart from the data alone. `Any` overlaps every type except `None`. Since a `file`
 * slot reads a file's JSON form `{uri, mime, name}`, `file` overlaps plain `object`,
 * `Any`, and every record that admits that mapping.
 *
 * Example: `{action: Literal(buy), qty: int}` and `{action: Literal(sell), price: float}`
 * do not overlap; `{conviction: float}` overlaps itself.
 */
export function typesOverlap(a: Type, b: Type): boolean {
  if (admitsNone(a) && admitsNone(b)) return true;
  if (a.kind === ValueKind.ANY || b.kind === ValueKind.ANY) {
    const other = a.kind === ValueKind.ANY ? b : a;
    return other.kind !== ValueKind.NONE;
  }
  if (a.variants !== undefined) return a.variants.some((member) => typesOverlap(member, b));
  if (b.variants !== undefined) return b.variants.some((member) => typesOverlap(a, member));
  if (a.fields !== undefined && b.fields !== undefined) return recordsOverlap(a, b);
  if (a.kind === ValueKind.OBJECT && b.kind === ValueKind.OBJECT) {
    // Only plain `object` remains on at least one side, and it accepts any record.
    return isPlainObject(a) || isPlainObject(b);
  }
  if (isListKind(a.kind) && isListKind(b.kind)) return true;
  if (a.tags !== undefined && b.tags !== undefined) {
    return [...a.tags].some((tag) => b.tags!.has(tag));
  }
  if (a.tags !== undefined || b.tags !== undefined) {
    const [literal, other] = a.tags !== undefined ? [a, b] : [b, a];
    if (other.kind === ValueKind.STRING) return true;
    if (other.kind === ValueKind.DATE || other.kind === ValueKind.DATETIME) {
      return [...literal.tags!].some((tag) => matchesTemporal(other.kind, tag));
    }
    return false;
  }
  if (NUMERIC_KINDS.has(a.kind) && NUMERIC_KINDS.has(b.kind)) return boundsIntersect(a, b);
  if (TEXT_KINDS.has(a.kind) && TEXT_KINDS.has(b.kind)) {
    return a.kind === b.kind || a.kind === ValueKind.STRING || b.kind === ValueKind.STRING;
  }
  if ((a.kind === ValueKind.FILE || b.kind === ValueKind.FILE) && a.kind !== b.kind) {
    const other = a.kind === ValueKind.FILE ? b : a;
    if (isPlainObject(other)) return true;
    return other.fields !== undefined && recordsOverlap(other, FILE_FORM);
  }
  return a.kind === b.kind;
}
