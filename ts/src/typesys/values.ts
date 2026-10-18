/**
 * The typed value system: yian's values, their typed carriers and wire form, the write
 * boundary that checks a value against a declared type, host value intake, and the type
 * vocabulary those checks use.
 *
 * A value is plain JS data (`Value`): `null`, a `boolean`, a `bigint` int, a `number` float, a
 * `string`, a `FileRef`, an array or a plain record. A `TypedValue` pairs a value with its
 * storage kind; the kind tag keeps an int an int and a float a float through the run log's
 * JSON, so the log can rebuild the same pool on replay. The write boundary wraps a value in
 * its natural kind or checks it against a declared `Type`; host intake reads a value a host
 * hands in (a tool or CODE result, an input) into a fresh yian value. Messages show a value
 * with `valueRepr`, cut to the shown-value cap.
 *
 * `ValueKind` is the closed set of storage tags, and `Type` the resolved structure of a
 * declared type (records, `Literal` tags, typed lists, numeric ranges and unions), with its
 * canonical labels, union member order and overlap check.
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

/**
 * The most characters a value or key shown in a message keeps.
 *
 * Becomes `current().maxShownValueChars` once the defaults module exists.
 */
function shownValueChars(): number {
  return 200;
}

/**
 * Cut text to the shown-value cap (200) for a message, counting code points.
 *
 * Returns `text` when it fits; else its first cap - 1 code points and `…`. A surrogate pair
 * is never split.
 *
 * Example: `shownText("x".repeat(500))` gives 199 x's then `…`.
 */
export function shownText(text: string): string {
  const cap = shownValueChars();
  // A string has at least as many code units as code points.
  if (text.length <= cap) return text;
  let index = 0;
  let cut = 0;
  for (let count = 0; count < cap; count++) {
    if (index >= text.length) return text;
    if (count === cap - 1) cut = index;
    index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  }
  return index >= text.length ? text : text.slice(0, cut) + "…";
}

/**
 * Return a value's text form for a message, cut to the shown-value cap (see `shownText`).
 *
 * The text form is compact JSON with JS number spelling: `true`, `null`, `2.5`, `2` for the
 * float 2, `Infinity`, an int's digits, text JSON-quoted, a `FileRef` as its mapping
 * `{"uri":…,"mime":…,"name":…}`, records in JS property order. A part that isn't a yian
 * value (a `Date`, a `Map`, a class instance, a symbol, a function, `undefined`) shows as
 * `<Name object>`, and an int the runtime can't stringify as `<int too large to display>`.
 * It never throws: when a getter or proxy trap throws, the whole value shows as
 * `<Name object>`. The writer stops once it has more than the cap can show, so a long list,
 * a deep value or a cyclic one costs time bounded by the cap.
 */
export function valueRepr(value: unknown): string {
  let text: string;
  try {
    text = textForm(value, 2 * shownValueChars() + 2);
  } catch {
    // A getter or proxy trap is user code; a broken one must not crash the message.
    try {
      return `<${jsTypeName(value)} object>`;
    } catch {
      return "<object object>";
    }
  }
  return shownText(text);
}

/** An open container while writing: its keys (`null` for an array) and the next index. */
interface TextCursor {
  readonly container: { readonly [key: string]: unknown } | readonly unknown[];
  readonly keys: readonly string[] | null;
  index: number;
}

/**
 * Return the text form of `root` (see `valueRepr`), stopping once it holds at least `budget`
 * code units. The walk uses an explicit stack and reads an array's items as it writes them.
 */
function textForm(root: unknown, budget: number): string {
  let out = "";
  const stack: ({ value: unknown } | TextCursor)[] = [{ value: root }];
  while (stack.length > 0 && out.length < budget) {
    const step = stack[stack.length - 1]!;
    if ("container" in step) {
      const size = step.keys === null ? step.container.length : step.keys.length;
      if (step.index === size) {
        stack.pop();
        out += step.keys === null ? "]" : "}";
        continue;
      }
      const index = step.index++;
      if (index > 0) out += ",";
      if (step.keys === null) {
        stack.push({ value: (step.container as readonly unknown[])[index] });
      } else {
        const key = step.keys[index]!;
        out += quoteHead(key, budget) + ":";
        stack.push({ value: (step.container as { readonly [key: string]: unknown })[key] });
      }
      continue;
    }
    stack.pop();
    const value = step.value;
    if (value === null || typeof value === "boolean" || typeof value === "number") {
      out += String(value);
    } else if (typeof value === "bigint") {
      out += bigintText(value);
    } else if (typeof value === "string") {
      out += quoteHead(value, budget);
    } else if (value instanceof FileRef) {
      out += "{";
      const container = value as unknown as { readonly [key: string]: unknown };
      stack.push({ container, keys: FILE_FIELDS, index: 0 });
    } else if (Array.isArray(value) && Object.getPrototypeOf(value) === Array.prototype) {
      out += "[";
      stack.push({ container: value, keys: null, index: 0 });
    } else if (isRecord(value)) {
      out += "{";
      stack.push({ container: value, keys: Object.keys(value), index: 0 });
    } else {
      out += `<${jsTypeName(value)} object>`;
    }
  }
  return out;
}

/** Return `text` JSON-quoted, quoting only its first `budget` code units when it is longer. */
function quoteHead(text: string, budget: number): string {
  // The rendering is cut far before `budget` code units, so the rest can't show.
  return JSON.stringify(text.length > budget ? text.slice(0, budget) : text);
}

/** Return an int's digits, or a placeholder when the runtime can't stringify it. */
function bigintText(value: bigint): string {
  try {
    return value.toString();
  } catch (error) {
    // JavaScriptCore reports a bigint too large to stringify as a RangeError.
    if (error instanceof RangeError) return "<int too large to display>";
    throw error;
  }
}

/**
 * Name a JS value's type for a message: the name of its prototype's own `constructor`
 * (`Date`, `Map`, `Foo`; not `value.constructor`, which the value can fake), else its
 * `typeof` (`symbol`, `function`, `undefined`), else `object`.
 */
function jsTypeName(value: unknown): string {
  if (typeof value !== "object" || value === null) return typeof value;
  const proto: unknown = Object.getPrototypeOf(value);
  if (proto === null) return "object";
  const constructor: unknown = Object.getOwnPropertyDescriptor(proto, "constructor")?.value;
  if (typeof constructor !== "function") return "object";
  const name: unknown = Object.getOwnPropertyDescriptor(constructor, "name")?.value;
  return typeof name === "string" && name !== "" ? name : "object";
}

/**
 * Render a value path as `a[1].b`, quoting a key that isn't an identifier as `['my key']`.
 *
 * Each key is cut with `shownText`, and a rendering longer than four times the shown-value
 * cap keeps its start and end around `…` (counting code points).
 */
export function renderValuePath(path: readonly (string | number)[]): string {
  const parts: string[] = [];
  for (const segment of path) {
    if (typeof segment === "number") {
      parts.push(`[${segment}]`);
    } else if (IDENTIFIER.test(segment)) {
      const shown = shownText(segment);
      parts.push(parts.length > 0 ? `.${shown}` : shown);
    } else {
      parts.push(`[${shownText(reprText(segment))}]`);
    }
  }
  const text = parts.join("");
  const limit = 4 * shownValueChars();
  if (text.length <= limit) return text;
  const points = Array.from(text);
  if (points.length <= limit) return text;
  const head = Math.floor(limit / 2);
  const tail = points.slice(points.length - (limit - head - 1));
  return points.slice(0, head).join("") + "…" + tail.join("");
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

/** A kind a value is stored under: every `ValueKind` but the type-only `Any`. */
export type StorageKind = Exclude<ValueKind, typeof ValueKind.ANY>;

const STORAGE_KINDS: ReadonlySet<string> = new Set(
  Object.values(ValueKind).filter((kind) => kind !== ValueKind.ANY),
);

/**
 * A yian value as JS holds it: `null`, a `boolean`, an int as a `bigint`, a float as a
 * `number` (never NaN), a `string` (well-formed), a `FileRef`, an array or a `ValueRecord`.
 */
export type Value = null | boolean | bigint | number | string | FileRef | Value[] | ValueRecord;

/**
 * A record: a plain object (prototype `Object.prototype` or `null`) whose own enumerable
 * string keys are its fields. A record yian builds sets a `__proto__` key with
 * `Object.defineProperty`, so it stays data.
 */
export interface ValueRecord {
  [key: string]: Value;
}

/** Return whether `kind` is one of the six `list[...]` kinds. */
export function isListKind(kind: ValueKind): boolean {
  return LIST_ELEMENT_KIND.has(kind);
}

/** Return the scalar element kind of a list kind (`null` for `LIST_ANY` and scalars). */
export function listElementKind(kind: ValueKind): ValueKind | null {
  return LIST_ELEMENT_KIND.get(kind) ?? null;
}

// --------------------------------------------------------------------------------------- //
// Reserved FILE placeholder
// --------------------------------------------------------------------------------------- //

/** The fields a `FileRef` is built from; `mime` and `name` default to `null`. */
interface FileRefInit {
  readonly uri: string;
  readonly mime?: string | null;
  readonly name?: string | null;
}

/**
 * A minimal, opaque file reference: a reserved placeholder.
 *
 * Carries only what is needed to round-trip through the typed-value wire form. Its JSON
 * form is the mapping `{"uri": ..., "mime": ..., "name": ...}`: a `file` slot reads that
 * mapping as a `FileRef`, and an `object` slot reads a nested `FileRef` as that mapping.
 * Storage, transfer, size and markdown rendering are absent until file handling is built.
 * A `FileRef` is frozen.
 */
export class FileRef {
  readonly uri: string;
  readonly mime: string | null;
  readonly name: string | null;

  /**
   * Build a file reference from its fields.
   *
   * @throws RangeError when `uri` isn't text, or `mime` or `name` isn't text or null, or a
   *   field holds a lone surrogate.
   */
  constructor({ uri, mime = null, name = null }: FileRefInit) {
    checkFileField("uri", uri, false);
    checkFileField("mime", mime, true);
    checkFileField("name", name, true);
    this.uri = uri;
    this.mime = mime;
    this.name = name;
    Object.freeze(this);
  }
}

/** Require file field `field` to be well-formed text, or null when `nullable`. */
function checkFileField(field: string, value: unknown, nullable: boolean): void {
  if (nullable && value === null) return;
  if (typeof value !== "string") {
    const expected = nullable ? "text or null" : "text";
    throw new RangeError(`file ${field} must be ${expected}, got ${valueRepr(value)}`);
  }
  if (!value.isWellFormed()) {
    throw new RangeError(`file ${field} holds a lone surrogate`);
  }
}

/** A file's fields: the keys of its JSON form and the only names a dotted read gives. */
const FILE_FIELDS: readonly ("uri" | "mime" | "name")[] = ["uri", "mime", "name"];

/**
 * Read field `name` of `value`, as a dotted `${x.name}` read sees it.
 *
 * A record gives its own key `name`; a `FileRef` gives its `uri`, `mime` or `name`. Nothing
 * else is read, so a dotted read never reaches an inherited property such as `constructor`,
 * `toString` or `__proto__` (unless the record holds it as an own key).
 *
 * Returns `[true, field value]` when `value` has the field, else `[false, null]`. A present
 * field may hold `null`, as a file's unset `mime` does.
 *
 * Example: `fieldOf(new FileRef({uri: "a"}), "uri")` gives `[true, "a"]`;
 * `fieldOf({a: 1n}, "constructor")` gives `[false, null]`.
 */
export function fieldOf(value: Value, name: string): [present: boolean, value: Value] {
  if (isRecord(value)) return Object.hasOwn(value, name) ? [true, value[name] as Value] : [false, null];
  if (value instanceof FileRef) {
    const field = FILE_FIELDS.find((candidate) => candidate === name);
    if (field !== undefined) return [true, value[field]];
  }
  return [false, null];
}

/**
 * Return the type of reading field `name` on a `file`, as load checks it.
 *
 * `uri` is `str`; `mime` and `name` are `Optional(str)`, since a file may leave them unset. A
 * file has no other field, so any other name gives `null`.
 *
 * Example: `fileFieldType("uri")` gives `Type.scalar(ValueKind.STRING)`;
 * `fileFieldType("size")` gives `null`.
 */
export function fileFieldType(name: string): Type | null {
  return FILE_FORM.fields!.get(name) ?? null;
}

/** Whether `value` is a record: a plain object with prototype `Object.prototype` or `null`. */
function isRecord(value: unknown): value is ValueRecord {
  if (typeof value !== "object" || value === null) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

// --------------------------------------------------------------------------------------- //
// Typed values
// --------------------------------------------------------------------------------------- //

// ---- Open owner question: a whole number in a `float` slot ---------------------------- //

/**
 * Whether a `float` carrier may hold `value`: a `number` other than NaN.
 *
 * Open owner question. Design 3.2.1 says a whole number filling a `float` slot stays an
 * int; Python yian and the shared case `values/int_fills_float` widen it to a float, which
 * is what is implemented: a `float` value is always a `number`. Under widening, a `bigint`
 * whose `Number()` is infinite (`10n ** 400n`) fails with `int is too large for float`, where
 * design 3.2.2's mixing rule would give `±Infinity`. If the owner picks "stays an int", this
 * function and `intInFloatSlot` change together, and a stored value takes either the
 * `float` kind holding a `bigint` or the `int` kind.
 */
function floatSlotHolds(value: unknown): value is number {
  return typeof value === "number" && !Number.isNaN(value);
}

// ---- End of the open owner question --------------------------------------------------- //

/** What a typed value of each storage kind holds, for the decoder's messages. */
const KIND_HOLDS: ReadonlyMap<StorageKind, string> = new Map([
  [ValueKind.NONE, "null"],
  [ValueKind.STRING, "a string"],
  [ValueKind.INTEGER, "a bigint"],
  [ValueKind.NUMBER, "a number other than NaN"],
  [ValueKind.DATE, "a string"],
  [ValueKind.DATETIME, "a string"],
  [ValueKind.BOOLEAN, "a boolean"],
  [ValueKind.OBJECT, "a plain record"],
  [ValueKind.LIST_ANY, "an array"],
  [ValueKind.LIST_STRING, "an array of strings or null"],
  [ValueKind.LIST_INTEGER, "an array of bigints or null"],
  [ValueKind.LIST_NUMBER, "an array of numbers other than NaN, or null"],
  [ValueKind.LIST_BOOLEAN, "an array of booleans or null"],
  [ValueKind.LIST_OBJECT, "an array of plain records or null"],
]);

/**
 * Whether `value` is a scalar of storage kind `kind` (`None`, `str`, `int`, `float`, `bool`,
 * `object`); `false` for any other kind.
 */
function holdsScalar(kind: ValueKind, value: unknown): boolean {
  switch (kind) {
    case ValueKind.NONE:
      return value === null;
    case ValueKind.STRING:
      return typeof value === "string";
    case ValueKind.INTEGER:
      return typeof value === "bigint";
    case ValueKind.NUMBER:
      return floatSlotHolds(value);
    case ValueKind.BOOLEAN:
      return typeof value === "boolean";
    case ValueKind.OBJECT:
      return isRecord(value);
    default:
      return false;
  }
}

/**
 * A value tagged with the storage kind it persists as; frozen.
 *
 * The tag is what makes the run log lossless: on a JSON round trip `fromWire` reads it back,
 * so an int stays an int and a float a float whatever JSON's number spelling. The constructor
 * doesn't check that `value` fits `kind`; the write boundary and `fromWire` are the checked
 * ways to build one.
 */
export class TypedValue {
  readonly kind: StorageKind;
  readonly value: Value;

  /** Pair `value` with `kind` as given, unchecked. */
  constructor(kind: StorageKind, value: Value) {
    this.kind = kind;
    this.value = value;
    Object.freeze(this);
  }

  /** Return the plain value (what expressions and node code see). */
  toObject(): Value {
    return this.value;
  }

  /**
   * Return the wire form `{kind, value}`, with every `FileRef` in the value (the root or
   * nested) written as its mapping, as the run log's encoder takes it.
   */
  toWire(): { kind: StorageKind; value: Value } {
    return { kind: this.kind, value: filesAsMappings(this.value) };
  }

  /**
   * Decode a typed value's wire form strictly.
   *
   * `data` must be a plain record with exactly `kind` and `value`; `kind` a storage kind (not
   * `Any`); and `value` of that kind: `None` null, `str` a string, `int` a `bigint`, `float` a
   * `number` other than NaN (an int is refused: the log writes a float with `.0`), `date` and
   * `datetime` text in their grammar, `bool` a boolean, `object` a plain record, `file` a
   * file's mapping with exactly `uri`, `mime` and `name` (read as a `FileRef`), `list[Any]`
   * an array, and `list[X]` an array of `X` or null. Contents aren't walked: the lossless
   * reader made the data.
   *
   * @throws RangeError naming the kind when `data` isn't such a wire form.
   */
  static fromWire(data: unknown): TypedValue {
    if (!isRecord(data)) throw new RangeError("typed-value wire must be a JSON object");
    requireFields(data, ["kind", "value"], "typed value");
    const kind: unknown = data["kind"];
    if (typeof kind !== "string" || !STORAGE_KINDS.has(kind)) {
      throw new RangeError(`typed-value kind must be a storage kind, got ${valueRepr(kind)}`);
    }
    const storage = kind as StorageKind;
    const value = data["value"] as Value;
    if (storage === ValueKind.FILE) {
      if (!isRecord(value)) throw new RangeError("file typed value must contain a JSON object");
      requireFields(value, FILE_FIELDS, "file reference");
      try {
        return new TypedValue(storage, new FileRef(value as unknown as FileRefInit));
      } catch (error) {
        if (error instanceof RangeError) {
          throw new RangeError(`invalid file typed value: ${error.message}`);
        }
        throw error;
      }
    }
    if (storage === ValueKind.DATE || storage === ValueKind.DATETIME) {
      if (typeof value === "string") {
        try {
          if (storage === ValueKind.DATE) parseDate(value);
          else parseDatetime(value);
        } catch (error) {
          if (error instanceof RangeError) {
            throw new RangeError(`invalid ${storage} typed value: ${error.message}`);
          }
          throw error;
        }
        return new TypedValue(storage, value);
      }
    } else if (holdsScalar(storage, value)) {
      return new TypedValue(storage, value);
    } else if (isListKind(storage) && Array.isArray(value)) {
      const element = listElementKind(storage);
      if (element === null || value.every((item) => item === null || holdsScalar(element, item))) {
        return new TypedValue(storage, value);
      }
    }
    const expected = KIND_HOLDS.get(storage)!;
    throw new RangeError(
      `invalid ${storage} typed value: expected ${expected}, got ${valueRepr(value)}`,
    );
  }
}

/**
 * Require `record`'s own keys to be exactly `expected`.
 *
 * @throws RangeError listing the missing and extra keys, each sorted.
 */
function requireFields(record: ValueRecord, expected: readonly string[], label: string): void {
  const actual = Object.keys(record);
  const missing = expected.filter((key) => !Object.hasOwn(record, key)).sort(compareCodeUnits);
  const extra = actual.filter((key) => !expected.includes(key)).sort(compareCodeUnits);
  if (missing.length > 0 || extra.length > 0) {
    throw new RangeError(
      `invalid ${label} fields; missing=${pythonList(missing)}, extra=${pythonList(extra)}`,
    );
  }
}

/** Spell a list of names as Python prints it: `['a', 'b']`. */
function pythonList(names: readonly string[]): string {
  return "[" + names.map(reprText).join(", ") + "]";
}

/**
 * Return `root` with every `FileRef` in it as its mapping, copying the containers.
 *
 * The walk uses an explicit stack, so any depth copies, and copies a shared container once.
 * `root` must be acyclic, as every yian value is.
 */
function filesAsMappings(root: Value): Value {
  const copies = new Map<Value[] | ValueRecord, Value>();
  const copied = (value: Value): Value => {
    if (value instanceof FileRef) return { uri: value.uri, mime: value.mime, name: value.name };
    if (typeof value === "object" && value !== null) return copies.get(value)!;
    return value;
  };
  const stack: [Value[] | ValueRecord, boolean][] = [];
  const visit = (value: Value): void => {
    if (typeof value === "object" && value !== null && !(value instanceof FileRef)) {
      if (!copies.has(value)) stack.push([value, false]);
    }
  };
  visit(root);
  while (stack.length > 0) {
    const [node, children] = stack.pop()!;
    if (copies.has(node)) continue;
    const items = Array.isArray(node) ? node : Object.values(node);
    if (!children) {
      stack.push([node, true]);
      items.forEach(visit);
      continue;
    }
    if (Array.isArray(node)) {
      copies.set(node, node.map(copied));
    } else {
      const record: ValueRecord = {};
      for (const [key, item] of Object.entries(node)) setField(record, key, copied(item));
      copies.set(node, record);
    }
  }
  return copied(root);
}

/** Set `record[key]` as an own data property, so a `__proto__` key stays data. */
function setField(record: ValueRecord, key: string, value: Value): void {
  if (key === "__proto__") {
    Object.defineProperty(record, key, {
      value,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  } else {
    record[key] = value;
  }
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
