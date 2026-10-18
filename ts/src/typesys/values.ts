/**
 * The typed value system's type half: the type vocabulary `types.ts` builds on.
 *
 * `ValueKind` is the closed set of storage tags, and `Type` the resolved structure of a
 * declared type (records, `Literal` tags, typed lists, numeric ranges and unions) that
 * the write boundary checks values against. The value half arrives later.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";

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

/** The properties a `Type` is built from; absent refinements are left out. */
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
