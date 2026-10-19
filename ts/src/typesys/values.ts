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

import { types } from "node:util";

import { current } from "../defaults.ts";
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
  readonly path: ValuePath;

  constructor(message?: string, options: ErrorOptions & { path?: ValuePath } = {}) {
    super(message, options);
    this.path = Object.freeze([...(options.path ?? [])]);
  }
}

/** A record-field/list-index path to a value, as `TypeCheckError.path` holds it. */
type ValuePath = readonly (string | number)[];

/**
 * Cut text to `YIAN_MAX_SHOWN_VALUE_CHARS` (200 by default) for a message, counting code points.
 *
 * Returns `text` when it fits; else its first cap - 1 code points and `…`. A surrogate pair
 * is never split.
 *
 * Example: `shownText("x".repeat(500))` gives 199 x's then `…`.
 */
export function shownText(text: string): string {
  const cap = current().maxShownValueChars;
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
 * The value never makes it throw: when a getter or proxy trap throws, the whole value shows as
 * `<Name object>`. The writer stops once it has more than the cap can show, so a long list,
 * a deep value or a cyclic one costs time bounded by the cap.
 *
 * @throws DefaultsError when a `YIAN_*` variable holds a value its entry does not accept.
 */
export function valueRepr(value: unknown): string {
  // Read before the `try`, so a bad `YIAN_*` value is thrown, not turned into a placeholder.
  const cap = current().maxShownValueChars;
  let text: string;
  try {
    text = textForm(value, 2 * cap + 2);
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
    } else if (FileRef.isFileRef(value)) {
      out += "{";
      const container = value as unknown as { readonly [key: string]: unknown };
      stack.push({ container, keys: FILE_FIELDS, index: 0 });
    } else if (isList(value)) {
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
 * `typeof` (`symbol`, `function`, `undefined`), else `object`. A `Proxy` names `Proxy`, and a
 * prototype or constructor that is a `Proxy` names `object`, so naming a value runs none of
 * its traps.
 */
export function jsTypeName(value: unknown): string {
  if (types.isProxy(value)) return "Proxy";
  if (typeof value !== "object" || value === null) return typeof value;
  const proto: unknown = Object.getPrototypeOf(value);
  if (proto === null || types.isProxy(proto)) return "object";
  const constructor: unknown = Object.getOwnPropertyDescriptor(proto, "constructor")?.value;
  if (typeof constructor !== "function" || types.isProxy(constructor)) return "object";
  const name: unknown = Object.getOwnPropertyDescriptor(constructor, "name")?.value;
  return typeof name === "string" && name !== "" ? name : "object";
}

// The writers below give a yian value's whole text, for a template or builtin to embed. They
// are separate from `textForm`, which writes any JS value for a message and stops at the cap.

/**
 * Return a value's text form (design 3.2.3): text as itself, anything else as `writeAsJson`'s
 * compact JSON.
 *
 * @throws TypeError as `writeAsJson` does.
 */
export function writeAsText(value: Value): string {
  return typeof value === "string" ? value : writeAsJson(value);
}

/**
 * Return a value as JSON, laid out as Python's `json.dumps(value, indent=indent)`.
 *
 * With no `indent` the JSON is compact (`,` and `:`, no spaces). With an int `indent` of 0 or
 * more, each item sits on its own line indented by `indent` spaces per level, items end with
 * `,`, and a key is followed by `": "`; an empty list or record is `[]` or `{}`. Text is
 * JSON-quoted with non-ASCII kept, an int is its digits, a float its JS spelling (`1` for the
 * float 1, bare `Infinity`), and a `FileRef` its mapping `{"uri", "mime", "name"}`; records keep
 * JS property order.
 *
 * @throws TypeError `integer is too large for this runtime` when the runtime can't stringify an
 *   int, or `text is too long for this runtime` when the text passes the engine's string limit.
 */
export function writeAsJson(value: Value, indent: number | null = null): string {
  try {
    return jsonText(value, indent, 0);
  } catch (error) {
    // Bun and Node word an over-long string differently; both throw a RangeError.
    if (error instanceof RangeError) throw new TypeError("text is too long for this runtime");
    throw error;
  }
}

/** Write `value` at nesting `level` (see `writeAsJson`). */
function jsonText(value: Value, indent: number | null, level: number): string {
  if (value === null || typeof value === "boolean" || typeof value === "number") return String(value);
  if (typeof value === "bigint") {
    try {
      return value.toString();
    } catch (error) {
      // JavaScriptCore reports a bigint too large to stringify as a RangeError.
      if (error instanceof RangeError) throw new TypeError("integer is too large for this runtime");
      throw error;
    }
  }
  if (typeof value === "string") return JSON.stringify(value);
  const [open, close] = Array.isArray(value) ? (["[", "]"] as const) : (["{", "}"] as const);
  let items: string[];
  if (Array.isArray(value)) {
    items = value.map((item) => jsonText(item, indent, level + 1));
  } else {
    const record = value as ValueRecord;
    const keys = FileRef.isFileRef(value) ? FILE_FIELDS : Object.keys(record);
    const colon = indent === null ? ":" : ": ";
    items = keys.map((key) => JSON.stringify(key) + colon + jsonText(record[key] as Value, indent, level + 1));
  }
  if (items.length === 0) return open + close;
  if (indent === null) return open + items.join(",") + close;
  const inner = "\n" + " ".repeat(indent * (level + 1));
  return open + inner + items.join("," + inner) + "\n" + " ".repeat(indent * level) + close;
}

/**
 * Render a value path as `a[1].b`, quoting a key that isn't an identifier as `["my key"]`.
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
      parts.push(`[${shownText(JSON.stringify(segment))}]`);
    }
  }
  const text = parts.join("");
  const limit = 4 * current().maxShownValueChars;
  if (text.length <= limit) return text;
  const points = Array.from(text);
  if (points.length <= limit) return text;
  const head = Math.floor(limit / 2);
  const tail = points.slice(points.length - (limit - head - 1));
  return points.slice(0, head).join("") + "…" + tail.join("");
}

/**
 * Throw if a value nests deeper than `YIAN_MAX_VALUE_NESTING` levels (100 by default).
 *
 * Each array, record and `FileRef` is one level; a `TypedValue` is unwrapped and adds none.
 * The walk is depth first and in order. A container already on the current path is skipped;
 * the codecs refuse cycles.
 *
 * @param value The value to check.
 * @param where Names the boundary for the message, e.g. `"output"`.
 * @param options.enclosingLevels Levels counted around the value (default 0); -1 lets the
 *   value itself go one level over, as for a START's input record.
 * @throws TypeCheckError at the first container past the limit, reading `<where>[ at <path>]:
 *   value nests deeper than <limit> levels, the limit for a yian value (each record and list
 *   is a level)`.
 */
export function rejectDeepNesting(
  value: unknown,
  where: string,
  { enclosingLevels = 0 }: { enclosingLevels?: number } = {},
): void {
  const limit = current().maxValueNesting;
  if (1 + enclosingLevels > limit && isNestingLevel(unwrapTyped(value))) {
    throw tooDeep(where, [], limit);
  }
  for (const [stack, key, , descends] of walkNesting(value)) {
    if (descends && stack.length + 1 + enclosingLevels > limit) {
      throw tooDeep(where, [...framePath(stack), key], limit);
    }
  }
}

/**
 * Return the path of the first entry that refers back to a container enclosing it.
 *
 * Walks the containers `rejectDeepNesting` walks, depth first and in order, and unwraps a
 * `TypedValue`. A container repeated outside its own path does not count.
 *
 * Returns the entry's path from `value`, or `null` if no entry holds a container enclosing it.
 */
export function recursiveEntryPath(value: unknown): ValuePath | null {
  for (const [stack, key, child, descends] of walkNesting(value)) {
    if (!descends && isNestingLevel(child)) return [...framePath(stack), key];
  }
  return null;
}

/**
 * Throw if text in a value, or a record key, holds a lone surrogate. Root text is checked
 * itself; inside containers the walk is `walkNesting`'s, depth first and in order.
 *
 * @throws TypeCheckError `<where>[ at <path>]: text holding a lone surrogate is not a yian value`
 *   (a key: at its record's path, as `fromHost` reports it), with `path`.
 */
export function rejectLoneSurrogates(value: unknown, where: string): void {
  const root = unwrapTyped(value);
  if (typeof root === "string" && !root.isWellFormed()) {
    throw locatedError(where, [], LONE_SURROGATE_REASON);
  }
  for (const [stack, key, child] of walkNesting(value)) {
    if (typeof key === "string" && !key.isWellFormed()) {
      throw locatedError(where, framePath(stack), LONE_SURROGATE_REASON);
    }
    if (typeof child === "string" && !child.isWellFormed()) {
      throw locatedError(where, [...framePath(stack), key], LONE_SURROGATE_REASON);
    }
  }
}

/** An open container in the nesting walk. */
interface NestingFrame {
  readonly container: object;
  /** The keys of a record or `FileRef`, or `null` for an array, read up to its length. */
  readonly keys: readonly string[] | null;
  /** The container's key or index in its parent (`null` for the root). */
  readonly key: string | number | null;
  index: number;
}

/**
 * One entry of the nesting walk: the frames of the containers on the current path, root
 * first, with the entry's container last; the entry's key in that container; its value,
 * unwrapped; and whether the walk goes into the entry next.
 */
type NestingEntry = [
  stack: readonly NestingFrame[],
  key: string | number,
  child: unknown,
  descends: boolean,
];

/**
 * Yield each entry of a value's containers, depth first and in order.
 *
 * Unwraps a `TypedValue` wherever it sits. The walk goes into an entry that is an array,
 * record or `FileRef` not already on the path. A caller reads the stack before resuming the
 * walk, which changes it. A leaf yields nothing.
 */
function* walkNesting(value: unknown): Generator<NestingEntry> {
  const root = unwrapTyped(value);
  if (!isNestingLevel(root)) return;
  // An explicit stack, not recursion, so any depth works.
  const stack: NestingFrame[] = [nestingFrame(root, null)];
  const active = new Set<object>([root]);
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    const container = frame.container as Record<string | number, unknown>;
    const size = frame.keys === null ? (frame.container as unknown[]).length : frame.keys.length;
    if (frame.index >= size) {
      stack.pop();
      active.delete(frame.container);
      continue;
    }
    const index = frame.index++;
    const key = frame.keys === null ? index : frame.keys[index]!;
    const child = unwrapTyped(container[key]);
    const descends = isNestingLevel(child) && !active.has(child);
    yield [stack, key, child, descends];
    if (descends) {
      stack.push(nestingFrame(child, key));
      active.add(child);
    }
  }
}

/** Return the open frame of `container`, an array, record or `FileRef`, at `key`. */
function nestingFrame(container: object, key: string | number | null): NestingFrame {
  let keys: readonly string[] | null = null;
  if (FileRef.isFileRef(container)) keys = FILE_FIELDS;
  else if (!isList(container)) keys = Object.keys(container);
  return { container, keys, key, index: 0 };
}

/** Return the path from the root to the last frame's container; each frame holds its key. */
function framePath(
  stack: readonly { readonly key: string | number | null }[],
): (string | number)[] {
  return stack.slice(1).map((frame) => frame.key!);
}

/** Whether `value` is a nesting level: an array, a record or a `FileRef`. */
function isNestingLevel(value: unknown): value is object {
  return isList(value) || isRecord(value) || FileRef.isFileRef(value);
}

/** Return `value` with any `TypedValue` wrapping it removed. */
function unwrapTyped(value: unknown): unknown {
  while (value instanceof TypedValue) value = value.value;
  return value;
}

/** Build the error for a container past the nesting limit at `path`. */
function tooDeep(where: string, path: ValuePath, limit: number): TypeCheckError {
  return locatedError(
    where,
    path,
    `value nests deeper than ${limit} levels, ` +
      "the limit for a yian value (each record and list is a level)",
  );
}

/** Build the `TypeCheckError` reading `<where>[ at <path>]: <reason>`, with `path`. */
function locatedError(where: string, path: ValuePath, reason: string): TypeCheckError {
  const location = path.length > 0 ? `${where} at ${renderValuePath(path)}` : where;
  return new TypeCheckError(`${location}: ${reason}`, { path });
}

/** Why text or a record key holding a lone surrogate is not a yian value. */
const LONE_SURROGATE_REASON = "text holding a lone surrogate is not a yian value";

/** An open array or record in host intake, with the fresh copy being filled. */
interface HostFrame {
  readonly source: Record<string | number, unknown>;
  /** The record's keys, or `null` for an array. */
  readonly keys: readonly string[] | null;
  /** The number of entries: the array's length or the record's key count, read once. */
  readonly size: number;
  /** The container's key or index in its parent (`null` for the root). */
  readonly key: string | number | null;
  readonly copy: Value[] | ValueRecord;
  index: number;
}

/**
 * Return a host value (a tool's result, a CODE body's return value) as a fresh yian value.
 *
 * `null`, a boolean, well-formed text, a `bigint` and a `FileRef` are taken as they are. A
 * `number` that is an integer within ±(2^53−1) becomes an int (`-0` becomes `0n`); any other
 * number but NaN (`±Infinity`, `1e20`, `2.5`, `9007199254740992`) is a float. A plain array
 * (prototype `Array.prototype`) and a plain record (prototype `Object.prototype` or `null`)
 * become a fresh array and record, their entries taken the same way; like `JSON.stringify`,
 * symbol keys, non-enumerable keys and extra array properties are skipped. A record key is
 * set as an own data property, so a `__proto__` key stays data.
 *
 * Intake reads only own data properties, each once, and refuses accessors and Proxies, so no
 * host code runs while it reads a value and the value checked is the value recorded. A value
 * shared by two parents is copied once, so sharing is kept and a diamond costs linear time.
 * The walk uses an explicit stack, so any depth works; depth is left to `rejectDeepNesting`.
 *
 * Example: `fromHost({a: [1, 2.5]}, "output")` gives `{a: [1n, 2.5]}`.
 *
 * @throws TypeCheckError at the first bad part, depth first and in order, reading `<where>[ at
 *   <path>]: <reason>`: `undefined` or an array hole is `undefined is not a yian value; use
 *   null`; NaN is `NaN is not a yian value`; text or a record key holding a lone surrogate
 *   (a key: at its record's path) is `text holding a lone surrogate is not a yian value`; a
 *   container inside itself is `value contains itself` at the entry holding it; an accessor
 *   property (a getter or a setter) on a record key or array index is `accessor property is
 *   not a yian value` at that entry; a `Proxy` is `Proxy is not a yian value`; anything else
 *   (a `Date`, a `Map`, a class instance, a `TypedValue`, a symbol, a function) is
 *   `<Name> is not a yian value`.
 */
export function fromHost(value: unknown, where: string): Value {
  const stack: HostFrame[] = [];
  const active = new Set<object>();
  const copies = new Map<object, Value[] | ValueRecord>();
  /** Throw `reason` at the top frame's entry `keys` (none: the top frame itself). */
  const fail = (reason: string, ...keys: (string | number)[]): never => {
    // The path is built only on a throw.
    throw locatedError(where, [...framePath(stack), ...keys], reason);
  };

  /** Return `source`'s own data property `key` (`undefined` for a hole); refuse an accessor. */
  const dataProperty = (source: object, key: string | number): unknown => {
    const descriptor = Object.getOwnPropertyDescriptor(source, key);
    if (descriptor === undefined || "value" in descriptor) return descriptor?.value;
    return fail("accessor property is not a yian value", key);
  };

  /** Return what `item`, the top frame's entry `key`, stores as; push a new container's frame. */
  const enter = (item: unknown, key: string | number | null): Value => {
    const at = key === null ? [] : [key];
    switch (typeof item) {
      case "boolean":
      case "bigint":
        return item;
      case "string":
        return item.isWellFormed() ? item : fail(LONE_SURROGATE_REASON, ...at);
      case "number":
        if (Number.isNaN(item)) fail("NaN is not a yian value", ...at);
        return Number.isSafeInteger(item) ? BigInt(item) : item;
      case "undefined":
        return fail("undefined is not a yian value; use null", ...at);
    }
    // Checked first: every later check (FileRef.isFileRef, isList, isRecord, jsTypeName) could
    // run a trap.
    if (types.isProxy(item)) fail("Proxy is not a yian value", ...at);
    if (item === null || FileRef.isFileRef(item)) return item;
    const list = isList(item);
    if (list || isRecord(item)) {
      if (active.has(item)) fail("value contains itself", ...at);
      const shared = copies.get(item);
      if (shared !== undefined) return shared;
      const source = item as Record<string | number, unknown>;
      const keys = list ? null : Object.keys(item);
      const size = keys === null ? (item as unknown[]).length : keys.length;
      const copy: Value[] | ValueRecord = list ? [] : {};
      copies.set(item, copy);
      active.add(item);
      stack.push({ source, keys, size, key, copy, index: 0 });
      return copy;
    }
    return fail(`${jsTypeName(item)} is not a yian value`, ...at);
  };

  const root = enter(value, null);
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.index === frame.size) {
      stack.pop();
      active.delete(frame.source);
      continue;
    }
    const index = frame.index++;
    if (frame.keys === null) {
      (frame.copy as Value[]).push(enter(dataProperty(frame.source, index), index));
    } else {
      const key = frame.keys[index]!;
      if (!key.isWellFormed()) fail(LONE_SURROGATE_REASON);
      setField(frame.copy as ValueRecord, key, enter(dataProperty(frame.source, key), key));
    }
  }
  return root;
}

/** An open array or record in `toHost`, with the fresh copy being filled. */
interface ToHostFrame {
  readonly source: Value[] | ValueRecord;
  /** The record's keys, or `null` for an array. */
  readonly keys: readonly string[] | null;
  /** The number of entries: the array's length or the record's key count, read once. */
  readonly size: number;
  readonly copy: Value[] | ValueRecord;
  index: number;
}

/**
 * Return a yian value as a fresh host value, by design 3.2.1's host boundary (D9A).
 *
 * An int within ±(2^53−1) becomes a `number`; a bigger one stays a `bigint`. A float
 * (`±Infinity` included), text, a boolean, `null` and a `FileRef` are handed over as they are.
 * Arrays and records become fresh arrays and records, their entries taken the same way; a
 * record key is set as an own data property, so a `__proto__` key stays data. A value shared
 * by two parents is copied once, and the walk uses an explicit stack, so any depth works.
 *
 * Example: `toHost({a: [1n, 2.5]})` gives `{a: [1, 2.5]}`.
 */
export function toHost(value: Value): unknown {
  const stack: ToHostFrame[] = [];
  const copies = new Map<object, Value[] | ValueRecord>();

  /** Return what `item` is handed over as; push a new container's frame. */
  const enter = (item: Value): Value => {
    if (typeof item === "bigint") {
      const n = Number(item);
      return Number.isSafeInteger(n) ? n : item;
    }
    if (item === null || typeof item !== "object" || FileRef.isFileRef(item)) return item;
    const shared = copies.get(item);
    if (shared !== undefined) return shared;
    const keys = Array.isArray(item) ? null : Object.keys(item);
    const size = keys === null ? (item as Value[]).length : keys.length;
    const copy: Value[] | ValueRecord = keys === null ? [] : {};
    copies.set(item, copy);
    stack.push({ source: item, keys, size, copy, index: 0 });
    return copy;
  };

  const root = enter(value);
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.index === frame.size) {
      stack.pop();
      continue;
    }
    const index = frame.index++;
    if (frame.keys === null) {
      (frame.copy as Value[]).push(enter((frame.source as Value[])[index]!));
    } else {
      const key = frame.keys[index]!;
      setField(frame.copy as ValueRecord, key, enter((frame.source as ValueRecord)[key]!));
    }
  }
  return root;
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
  // Set only by the constructor, so an object merely given this prototype isn't a FileRef.
  readonly #built = true;

  /** Whether `value` is a `FileRef` its constructor built (not one forged from its prototype). */
  static isFileRef(value: unknown): value is FileRef {
    return typeof value === "object" && value !== null && #built in value;
  }

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
  if (FileRef.isFileRef(value)) {
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

/**
 * Whether `value` is a record: a plain object with prototype `Object.prototype` or `null`. An
 * array is never a record, whatever its prototype.
 */
export function isRecord(value: unknown): value is ValueRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** Whether `value` is a list: a plain array (prototype `Array.prototype`). */
export function isList(value: unknown): value is unknown[] {
  return Array.isArray(value) && Object.getPrototypeOf(value) === Array.prototype;
}

// --------------------------------------------------------------------------------------- //
// Typed values
// --------------------------------------------------------------------------------------- //

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
  [ValueKind.LIST_NUMBER, "an array of bigints, numbers other than NaN, or null"],
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
      return typeof value === "number" && !Number.isNaN(value);
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
   * nested) written as its mapping, as the run log's encoder takes it. The value is copied
   * only on the paths to its files.
   *
   * @throws TypeCheckError when the value isn't a JSON value (see `jsonPart`): a non-JSON part,
   *   NaN, text holding a lone surrogate, or a container inside itself. The unchecked
   *   constructor can hold such a value.
   */
  toWire(): { kind: StorageKind; value: Value } {
    return { kind: this.kind, value: jsonPart(this.value, [], "mapping") };
  }

  /**
   * Decode a typed value's wire form strictly.
   *
   * `data` must be a plain record with exactly `kind` and `value`; `kind` a storage kind (not
   * `Any`); and `value` of that kind: `None` null, `str` a string, `int` a `bigint`, `float` a
   * `number` other than NaN (an int is refused: a `float` slot given an int stores kind
   * `int`), `date` and `datetime` text in their grammar, `bool` a boolean, `object` a plain
   * record, `file` a file's mapping with exactly `uri`, `mime` and `name` (read as a
   * `FileRef`), `list[Any]` an array, and `list[X]` an array of `X` or null, where a
   * `list[float]` item may also be a `bigint` (a `list(float)` slot keeps its int items).
   * Contents aren't walked: the lossless reader made the data.
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
    } else if (isListKind(storage) && isList(value)) {
      const element = listElementKind(storage);
      // An index loop, not `every`, so a hole is checked as `undefined`.
      let holds = true;
      for (let index = 0; holds && element !== null && index < value.length; index++) {
        const item = value[index];
        holds =
          item === null ||
          holdsScalar(element, item) ||
          (element === ValueKind.NUMBER && typeof item === "bigint");
      }
      if (holds) return new TypedValue(storage, value);
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
export function requireFields(
  record: ValueRecord,
  expected: readonly string[],
  label: string,
): void {
  const actual = Object.keys(record);
  const missing = expected.filter((key) => !Object.hasOwn(record, key)).sort(compareCodeUnits);
  const extra = actual.filter((key) => !expected.includes(key)).sort(compareCodeUnits);
  if (missing.length > 0 || extra.length > 0) {
    throw new RangeError(
      `invalid ${label} fields; missing=${writeAsText(missing)}, extra=${writeAsText(extra)}`,
    );
  }
}

/** Set `record[key]` as an own data property, so a `__proto__` key stays data. */
export function setField(record: ValueRecord, key: string, value: Value): void {
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

/**
 * Every union member has the field; `type` is what a read gives, or `null` when the members'
 * types differ and the read is untyped.
 */
export interface FieldRead {
  readonly kind: "read";
  readonly type: Type | null;
}

/** Only some union members have the field: `members`, in the union's member order. */
export interface FieldPartial {
  readonly kind: "partial";
  readonly members: readonly Type[];
}

/** No union member has the field. */
export interface FieldAbsent {
  readonly kind: "absent";
}

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

/**
 * Pick a union's tag field: `field` when given, else the first by name.
 *
 * @throws RangeError when `typ` is not a tagged union, or `field` is not one of its tag fields.
 */
function tagField(typ: Type, field: string | undefined): string {
  const tagFields = unionTagFields(typ);
  if (tagFields.length === 0) throw new RangeError("the type is not a tagged union");
  if (field === undefined) return tagFields[0]!;
  if (!tagFields.includes(field)) {
    throw new RangeError(`${reprText(field)} is not a tag field of the union`);
  }
  return field;
}

/**
 * Return the member of the tagged union `typ` whose tag field equals `tag`, or `null` when no
 * member has that tag (a non-string `tag` included). `field` is the tag field to match on; by
 * default the first tag field by name.
 *
 * @throws RangeError when `typ` is not a tagged union, or `field` is not one of its tag fields.
 */
export function unionMemberForTag(typ: Type, tag: unknown, field?: string): Type | null {
  return memberWithTag(typ.variants!, tagField(typ, field), tag);
}

/** Return the member whose single-value tag field `field` equals `tag`, or `null`. */
function memberWithTag(members: readonly Type[], field: string, tag: unknown): Type | null {
  const index = tagIndex(members, field, tag);
  return index === null ? null : members[index]!;
}

/**
 * Return the index of the member whose tag field `field` equals `tag`, or `null`; a non-string
 * tag finds no member.
 */
function tagIndex(members: readonly Type[], field: string, tag: unknown): number | null {
  if (typeof tag !== "string") return null;
  const index = members.findIndex((member) => member.fields!.get(field)!.tags!.has(tag));
  return index === -1 ? null : index;
}

/**
 * Return the part of the union `typ` that `keep` selects, keeping its nullability.
 *
 * Each item of `keep` names a member by its tag value (tagged unions only) or is a member
 * `Type` (compared with `typeEquals`); `field` is the tag field tag values refer to, by
 * default the first by name. One selected member gives that member record; several give a
 * smaller union with its members in canonical order. Either is nullable when `typ` is.
 *
 * Example: restricting `Union(Buy, Hold, Sell)` to `["buy", "sell"]` gives `Union(Buy, Sell)`.
 *
 * @throws TypeError when `keep` is a bare string rather than a collection of tags.
 * @throws RangeError when `typ` is not a union, `keep` is empty, or an item is neither a tag
 *   nor a member of `typ`.
 */
export function restrictUnion(typ: Type, keep: Iterable<string | Type>, field?: string): Type {
  if (typeof keep === "string") {
    throw new TypeError("restrictUnion keeps a collection of tags, not a bare string");
  }
  if (typ.variants === undefined) throw new RangeError("only a union can be restricted");
  const members = typ.variants;
  let selectedField: string | null = null;
  const selected = new Set<number>();
  for (const item of keep) {
    let index: number | null;
    if (item instanceof Type) {
      index = members.findIndex((member) => typeEquals(member, item));
      if (index === -1) {
        throw new RangeError("restrictUnion got a type that is not a union member");
      }
    } else {
      selectedField ??= tagField(typ, field);
      index = tagIndex(members, selectedField, item);
      if (index === null) throw new RangeError(`${reprText(item)} is not a tag of the union`);
    }
    selected.add(index);
  }
  if (selected.size === 0) throw new RangeError("restrictUnion needs at least one member");
  const indexes = [...selected].sort((a, b) => a - b);
  if (indexes.length === 1) return members[indexes[0]!]!.replace({ nullable: typ.nullable });
  const kept = canonicalUnionOrder(indexes.map((index) => members[index]!));
  return new Type({ kind: ValueKind.OBJECT, variants: kept, nullable: typ.nullable });
}

/**
 * Describe reading field `name` on a value of the union type `typ`.
 *
 * When every member has the field, the read's type is the members' type if they are equal
 * ignoring `nullable`, the merged `Literal` if they are all string `Literal`s, and untyped
 * otherwise. It is nullable when any member's field is `Optional` or not required.
 *
 * @returns A `FieldRead` when every member has the field, a `FieldPartial` with the members
 *   that have it when only some do, and a `FieldAbsent` when none do.
 * @throws RangeError when `typ` is not a union.
 */
export function unionField(typ: Type, name: string): FieldRead | FieldPartial | FieldAbsent {
  if (typ.variants === undefined) throw new RangeError("unionField needs a union type");
  const having = typ.variants.filter((member) => member.fields!.has(name));
  if (having.length === 0) return { kind: "absent" };
  if (having.length < typ.variants.length) return { kind: "partial", members: having };
  const fieldTypes = having.map((member) => member.fields!.get(name)!);
  const nullable = having.some(
    (member, i) => fieldTypes[i]!.nullable || !member.required?.has(name),
  );
  const strict = fieldTypes.map((fieldType) => fieldType.replace({ nullable: false }));
  if (strict.every((fieldType) => typeEquals(fieldType, strict[0]!))) {
    return { kind: "read", type: strict[0]!.replace({ nullable }) };
  }
  if (strict.every((fieldType) => fieldType.tags !== undefined)) {
    const merged = new Set(strict.flatMap((fieldType) => [...fieldType.tags!]));
    return { kind: "read", type: new Type({ kind: ValueKind.STRING, tags: merged, nullable }) };
  }
  return { kind: "read", type: null };
}

/** Whether `null` satisfies `typ`. */
function admitsNone(typ: Type): boolean {
  return typ.nullable || typ.kind === ValueKind.NONE;
}

/**
 * Whether `file` appears anywhere in `typ`: itself, a field, the element, or a member, so some
 * value of `typ` may hold a file. A type without `file` holds only JSON values, so every value
 * of it is an `Any` value.
 */
export function hasFile(typ: Type): boolean {
  if (typ.kind === ValueKind.FILE) return true;
  if (typ.element !== undefined && hasFile(typ.element)) return true;
  if (typ.fields !== undefined && [...typ.fields.values()].some(hasFile)) return true;
  return typ.variants !== undefined && typ.variants.some(hasFile);
}

/**
 * Whether every non-null value of `typ` holds a file, so none is an `Any` value.
 *
 * A list counts when its element is a non-nullable type that requires a file: the empty list
 * is shared by every list type and is not counted as a shared value. A record counts when a
 * required, non-nullable field requires a file; a union when every member does. Load treats
 * such a type as disjoint from `Any`, although a `file` slot reads a file's JSON form (an
 * `Any` value) at runtime, where load can't tell.
 */
export function requiresFile(typ: Type): boolean {
  if (typ.kind === ValueKind.FILE) return true;
  if (typ.variants !== undefined) return typ.variants.every(requiresFile);
  if (isListKind(typ.kind)) {
    const element = typ.element;
    return element !== undefined && !element.nullable && requiresFile(element);
  }
  if (typ.fields !== undefined) {
    return [...typ.fields].some(
      ([name, field]) => typ.required?.has(name) === true && !field.nullable && requiresFile(field),
    );
  }
  return false;
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
      return [...literal.tags!].some((tag) => scalarMatches(other.kind, tag));
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

// --------------------------------------------------------------------------------------- //
// Scalar type checks / coercion
// --------------------------------------------------------------------------------------- //

/** Return the storage kind a scalar, file or record infers as, or `null` for anything else. */
function inferScalarKind(value: unknown): StorageKind | null {
  if (value === null) return ValueKind.NONE;
  if (typeof value === "boolean") return ValueKind.BOOLEAN;
  if (typeof value === "bigint") return ValueKind.INTEGER;
  if (typeof value === "number") return ValueKind.NUMBER;
  if (typeof value === "string") return ValueKind.STRING;
  if (FileRef.isFileRef(value)) return ValueKind.FILE;
  if (isRecord(value)) return ValueKind.OBJECT;
  return null;
}

/**
 * Return why `value` is not a valid `declared` (`date` or `datetime`) string, or `null` when
 * it parses.
 */
function temporalError(declared: ValueKind, value: string): string | null {
  try {
    if (declared === ValueKind.DATE) parseDate(value);
    else parseDatetime(value);
  } catch (error) {
    if (error instanceof RangeError) return error.message;
    throw error;
  }
  return null;
}

/**
 * Whether `value` fits the scalar kind `declared`. A `float` slot takes an int (which stays
 * an int); a `file` slot takes a `FileRef` or a file's JSON form.
 */
function scalarMatches(declared: ValueKind, value: unknown): boolean {
  switch (declared) {
    case ValueKind.NONE:
      return value === null;
    case ValueKind.STRING:
      return typeof value === "string";
    case ValueKind.BOOLEAN:
      return typeof value === "boolean";
    case ValueKind.INTEGER:
      return typeof value === "bigint";
    case ValueKind.NUMBER:
      return typeof value === "bigint" || typeof value === "number";
    case ValueKind.DATE:
    case ValueKind.DATETIME:
      return typeof value === "string" && temporalError(declared, value) === null;
    case ValueKind.OBJECT:
      return isRecord(value);
    case ValueKind.FILE:
      return FileRef.isFileRef(value) || isFileForm(value);
    default:
      return false;
  }
}

/**
 * Return `value` as a `declared` slot stores it: `-0` in a `float` slot becomes `0`; a file's
 * JSON form in a `file` slot becomes a `FileRef`. Any other value (an int in a `float` slot
 * too) is returned unchanged.
 *
 * @throws TypeCheckError at `path` when a file form's text holds a lone surrogate.
 */
function coerceScalar(declared: ValueKind, value: unknown, path: ValuePath): unknown {
  if (declared === ValueKind.NUMBER && Object.is(value, -0)) return 0;
  if (declared === ValueKind.FILE && isFileForm(value)) {
    for (const field of FILE_FIELDS) rejectNonValue(value[field], [...path, field]);
    return new FileRef(value);
  }
  return value;
}

/**
 * Whether `value` is a file's JSON form: a record with exactly the keys `uri` (text), `mime`
 * and `name` (each text or null).
 */
function isFileForm(value: unknown): value is ValueRecord & FileRefInit {
  if (!isRecord(value) || Object.keys(value).length !== FILE_FIELDS.length) return false;
  if (!FILE_FIELDS.every((field) => Object.hasOwn(value, field))) return false;
  const { uri, mime, name } = value;
  return (
    typeof uri === "string" &&
    (mime === null || typeof mime === "string") &&
    (name === null || typeof name === "string")
  );
}

/**
 * Return why `value` itself can't be a yian value (NaN, or text holding a lone surrogate), or
 * `null`. A container's contents aren't looked at.
 */
function nonValueReason(value: unknown): string | null {
  if (typeof value === "number" && Number.isNaN(value)) return "NaN is not a yian value";
  if (typeof value === "string" && !value.isWellFormed()) {
    return LONE_SURROGATE_REASON;
  }
  return null;
}

/** Refuse NaN or text holding a lone surrogate at `path`; a container passes. */
function rejectNonValue(value: unknown, path: ValuePath): void {
  const reason = nonValueReason(value);
  if (reason !== null) throw new TypeCheckError(reason, { path });
}

/** What the JSON walk does with a `FileRef`: keep it, write it as its mapping, or refuse it. */
type FileMode = "keep" | "mapping" | "refuse";

/** An open list or record in the JSON walk. */
interface JsonFrame {
  readonly container: unknown[] | ValueRecord;
  /** The record's keys, or `null` for a list. */
  readonly keys: readonly string[] | null;
  /** The container's key or index in its parent (`null` for the root). */
  readonly key: string | number | null;
  index: number;
  /** The copy holding the entries read so far, made at the first entry that changes. */
  copy: Value[] | ValueRecord | null;
}

/**
 * Return `root` as a JSON value: `null`, a boolean, a `bigint`, a `number`, text, and lists and
 * records of those, with `FileRef`s handled as `files` says.
 *
 * `-0` becomes `0`, and under `"mapping"` each `FileRef` (the root too) becomes its mapping
 * `{uri, mime, name}`. Only the containers on the path to a change are copied; `root` itself is
 * returned when nothing changes, and a container shared by two parents is walked once. The
 * walk uses an explicit stack, so any depth works; nesting depth is left to the boundaries.
 *
 * @throws TypeCheckError at the bad part's path, the first depth-first in order: NaN or text
 *   holding a lone surrogate (a key: at its record's path) is not a yian value; `undefined` or
 *   a list hole is `undefined is not a JSON value`; a container inside itself is `value
 *   contains itself` at the entry holding it; anything else (a `Date`, a `Map`, a class
 *   instance, a `FileRef` under `"refuse"`) is `<valueRepr> is not a JSON value`.
 */
export function jsonPart(root: unknown, path: ValuePath, files: FileMode): Value {
  const stack: JsonFrame[] = [];
  const active = new Set<object>();
  const done = new Map<object, Value>();
  const pathTo = (...keys: (string | number)[]): ValuePath => [
    ...path,
    ...framePath(stack),
    ...keys,
  ];
  /** Throw `message` at the entry `key` of the top frame (`null`: the root). */
  const fail = (key: string | number | null, message: string): never => {
    // The path is built only on a throw.
    throw new TypeCheckError(message, { path: key === null ? path : pathTo(key) });
  };

  /** Return what `value` stores as, or `undefined` once its container's frame is pushed. */
  const enter = (value: unknown, key: string | number | null): Value | undefined => {
    const reason = nonValueReason(value);
    if (reason !== null) fail(key, reason);
    if (value === null || typeof value === "boolean" || typeof value === "bigint") return value;
    if (typeof value === "string") return value;
    if (typeof value === "number") return Object.is(value, -0) ? 0 : value;
    if (FileRef.isFileRef(value)) {
      if (files === "keep") return value;
      if (files === "mapping") return { uri: value.uri, mime: value.mime, name: value.name };
      // Under "refuse" a file is not a JSON value.
      return fail(key, `${valueRepr(value)} is not a JSON value`);
    }
    if (isList(value) || isRecord(value)) {
      if (active.has(value)) fail(key, "value contains itself");
      const finished = done.get(value);
      if (finished !== undefined) return finished;
      const keys = isList(value) ? null : Object.keys(value);
      stack.push({ container: value, keys, key, index: 0, copy: null });
      active.add(value);
      return undefined;
    }
    if (value === undefined) fail(key, "undefined is not a JSON value");
    return fail(key, `${valueRepr(value)} is not a JSON value`);
  };

  /** Record that `frame`'s last-read entry, holding `original`, stores `stored`. */
  const settle = (frame: JsonFrame, original: unknown, stored: Value): void => {
    const index = frame.index - 1;
    if (frame.copy === null) {
      if (Object.is(stored, original)) return;
      if (frame.keys === null) {
        frame.copy = (frame.container as Value[]).slice(0, index);
      } else {
        const copy: ValueRecord = {};
        const container = frame.container as ValueRecord;
        for (const name of frame.keys.slice(0, index)) setField(copy, name, container[name]!);
        frame.copy = copy;
      }
    }
    if (Array.isArray(frame.copy)) frame.copy.push(stored);
    else setField(frame.copy, frame.keys![index]!, stored);
  };

  const rootValue = enter(root, null);
  if (rootValue !== undefined) return rootValue;
  for (;;) {
    const frame = stack[stack.length - 1]!;
    const size = frame.keys === null ? (frame.container as unknown[]).length : frame.keys.length;
    if (frame.index < size) {
      const index = frame.index++;
      const key = frame.keys === null ? index : frame.keys[index]!;
      if (typeof key === "string" && !key.isWellFormed()) {
        throw new TypeCheckError(LONE_SURROGATE_REASON, { path: pathTo() });
      }
      const child = (frame.container as Record<string | number, unknown>)[key];
      const stored = enter(child, key);
      if (stored !== undefined) settle(frame, child, stored);
      continue;
    }
    stack.pop();
    active.delete(frame.container);
    const result = frame.copy ?? (frame.container as Value);
    done.set(frame.container, result);
    const parent = stack[stack.length - 1];
    if (parent === undefined) return result;
    settle(parent, frame.container, result);
  }
}

// --------------------------------------------------------------------------------------- //
// Constructors
// --------------------------------------------------------------------------------------- //

/**
 * Wrap a value in the `TypedValue` of its natural kind, inferring the kind.
 *
 * A `TypedValue` is returned as is, so the call is idempotent. Any other value is first walked
 * as a JSON value with files kept (see `jsonPart`): `-0` becomes `0` at any depth, and a part
 * that isn't a yian JSON value is refused. `file` is never inferred from a record or text;
 * only a `FileRef` is a file. A list of text, bools, ints or records infers its list kind, a
 * list of ints and floats is `list[float]` (each int stays an int), and an empty or mixed list
 * is `list[Any]`.
 *
 * Example: `buildValue([1n, 2.5])` gives a `list[float]` holding `[1n, 2.5]`.
 *
 * @throws TypeCheckError when the value is not a scalar, `FileRef`, list or record (`cannot
 *   wrap value of type 'Date'`), or a part of it is not a JSON value.
 */
export function buildValue(value: unknown): TypedValue {
  if (value instanceof TypedValue) return value;
  if (inferScalarKind(value) === null && !isList(value)) {
    throw new TypeCheckError(`cannot wrap value of type ${reprText(jsTypeName(value))}`);
  }
  const json = jsonPart(value, [], "keep");
  const kind = inferScalarKind(json);
  return kind === null ? inferList(json as Value[]) : new TypedValue(kind, json);
}

/** Wrap a list in the list kind its items infer (see `buildValue`). */
function inferList(items: Value[]): TypedValue {
  if (items.length === 0) return new TypedValue(ValueKind.LIST_ANY, items);
  const kinds = new Set(items.map(inferScalarKind));
  const only = (kind: StorageKind) => kinds.size === 1 && kinds.has(kind);
  if (kinds.has(null)) return new TypedValue(ValueKind.LIST_ANY, items);
  if (only(ValueKind.STRING)) return new TypedValue(ValueKind.LIST_STRING, items);
  if (only(ValueKind.BOOLEAN)) return new TypedValue(ValueKind.LIST_BOOLEAN, items);
  if (only(ValueKind.INTEGER)) return new TypedValue(ValueKind.LIST_INTEGER, items);
  if ([...kinds].every((kind) => kind === ValueKind.INTEGER || kind === ValueKind.NUMBER)) {
    return new TypedValue(ValueKind.LIST_NUMBER, items);
  }
  if (only(ValueKind.OBJECT)) return new TypedValue(ValueKind.LIST_OBJECT, items);
  return new TypedValue(ValueKind.LIST_ANY, items);
}

/**
 * Wrap `value` as a `TypedValue` of `declared`, throwing on a type mismatch.
 *
 * This is the write-boundary check against each declared type, so a node returning the wrong
 * type fails at the write rather than downstream. NaN and text holding a lone surrogate are
 * refused at any depth. An int filling a `float` slot stays an int at any depth: a `float`
 * slot given an int stores kind `int`, a `list(float)` keeps its int items, and a record's
 * `float` field keeps the int. Conversions apply at any depth (a record field, a list item):
 * `-0` in a `float` slot becomes `0`; a file's JSON form `{uri, mime, name}` (exactly those
 * keys) filling a `file` slot becomes a `FileRef`; and a `FileRef` nested in an `object`
 * value or `list(object)` item becomes that mapping, copying only the containers on its
 * path.
 *
 * `Any` accepts any non-null JSON value and stores it in its natural kind (a list as
 * `list[Any]`); its one conversion is `-0` to `0` at any depth, as in an `object` value. An
 * `object` value, a `list(object)` item and a `list[Any]` item must be JSON values (see
 * `jsonPart`); a `FileRef` is not one, so it fails anywhere in an `Any` or `list[Any]` value.
 * A record keeps the value's key order.
 *
 * `declared` is a `Type`, or a bare `ValueKind` for a scalar or flat list. A `TypedValue`
 * input is unwrapped to its value first.
 *
 * Example: `buildValueAs(ValueKind.NUMBER, 7n)` gives an `int` holding `7n`.
 *
 * @throws TypeCheckError at the failing value's path when the value does not satisfy
 *   `declared`: a wrong scalar, a non-member variant tag, an out-of-range number, a record
 *   with missing or unknown fields, a union value with a missing or unknown tag or matching no
 *   member, a bad list element, or a part that is not a JSON value where one is required.
 */
export function buildValueAs(declared: ValueKind | Type, value: unknown): TypedValue {
  const typ = declared instanceof Type ? declared : Type.scalar(declared);
  return buildForType(typ, value instanceof TypedValue ? value.value : value, []);
}

/**
 * Check `value` against `typ` at `path` (relative to the checked root) and return it as `typ`
 * stores it.
 *
 * @throws TypeCheckError at `path` or the failing descendant's path.
 */
function buildForType(typ: Type, value: unknown, path: ValuePath): TypedValue {
  rejectNonValue(value, path);
  if (typ.bounds !== undefined) {
    validateNumericBounds(typ.kind, typ.bounds);
    const [low, high] = typ.bounds;
    // `<` and `>` compare a bigint with a number exactly; an open end admits its infinity.
    if (scalarMatches(typ.kind, value)) {
      const number = value as bigint | number;
      const below = low !== null && number < low;
      const above = high !== null && number > high;
      if (below || above) {
        let message: string;
        if (low !== null && high !== null) {
          const bounds = `(${boundText(low)}, ${boundText(high)})`;
          message = `is outside inclusive ${typ.kind} bounds ${bounds}`;
        } else if (below) {
          message = `is below the inclusive ${typ.kind} minimum ${valueRepr(low)}`;
        } else {
          message = `is above the inclusive ${typ.kind} maximum ${valueRepr(high)}`;
        }
        throw new TypeCheckError(`${valueRepr(value)} ${message}`, { path });
      }
    }
  }

  if (value === null && typ.nullable) return new TypedValue(ValueKind.NONE, null);

  if (typ.variants !== undefined) return buildForUnion(typ, value, path);

  if (typ.tags !== undefined) {
    if (typeof value !== "string" || !typ.tags.has(value)) {
      const tags = writeAsText([...typ.tags].sort(compareCodeUnits));
      throw new TypeCheckError(`${valueRepr(value)} is not a member of variant ${tags}`, { path });
    }
    return new TypedValue(ValueKind.STRING, value);
  }

  if (typ.fields !== undefined) {
    const fields = typ.fields;
    if (!isRecord(value)) {
      throw new TypeCheckError(`${valueRepr(value)} is not an object for a record type`, { path });
    }
    const missing = [...(typ.required ?? [])]
      .filter((name) => !Object.hasOwn(value, name))
      .sort(compareCodeUnits);
    if (missing.length > 0) {
      throw new TypeCheckError(`record missing required fields: ${writeAsText(missing)}`, {
        path: [...path, missing[0]!],
      });
    }
    const names = Object.keys(value);
    const unknown = names.filter((name) => !fields.has(name)).sort(compareCodeUnits);
    if (unknown.length > 0) {
      const shown = shownText(unknown.map((name) => JSON.stringify(name)).join(","));
      throw new TypeCheckError(`record has unknown fields: [${shown}]`, {
        path: [...path, unknown[0]!],
      });
    }
    // Each field stores its checked value, so a nested slot converts as a top one does.
    const record: ValueRecord = {};
    for (const name of names) {
      setField(record, name, buildForType(fields.get(name)!, value[name], [...path, name]).value);
    }
    return new TypedValue(ValueKind.OBJECT, record);
  }

  if (isListKind(typ.kind) && typ.element !== undefined) {
    if (!isList(value)) {
      const label = canonicalTypeLabel(typ);
      throw new TypeCheckError(`${valueRepr(value)} is not a list for declared ${label}`, {
        path,
      });
    }
    const items: Value[] = [];
    // An index loop, not `map`, so a hole is checked as `undefined`.
    for (let index = 0; index < value.length; index++) {
      items.push(buildForType(typ.element, value[index], [...path, index]).value);
    }
    return new TypedValue(typ.kind as StorageKind, items);
  }

  if (typ.kind === ValueKind.ANY) {
    if (value === null) {
      throw new TypeCheckError("null does not match declared type Any", { path });
    }
    const json = jsonPart(value, path, "refuse");
    return new TypedValue(isList(json) ? ValueKind.LIST_ANY : inferScalarKind(json)!, json);
  }

  return buildScalarOrList(typ.kind, value, path);
}

/**
 * Check `value` against the member of the union `typ` it belongs to, at `path`, and return it
 * stored as an object.
 *
 * A tagged union selects the member by its first tag field; an untagged union takes the first
 * member the value satisfies.
 *
 * @throws TypeCheckError when the value is not a record, its tag is missing or unknown, or it
 *   fails the selected member (or every member, a `NoUnionMember`).
 */
function buildForUnion(typ: Type, value: unknown, path: ValuePath): TypedValue {
  if (!isRecord(value)) {
    throw new TypeCheckError(`${valueRepr(value)} is not an object for a union type`, { path });
  }
  const members = typ.variants!;
  const tagged = taggedMember(members, value);
  if (tagged !== null) {
    const [field, member] = tagged;
    if (!Object.hasOwn(value, field)) {
      const expected = writeAsText(tagValues(members, field));
      throw new TypeCheckError(
        `missing tag field ${reprText(field)} (expected one of ${expected})`,
        { path: [...path, field] },
      );
    }
    if (member === null) {
      const expected = writeAsText(tagValues(members, field));
      throw new TypeCheckError(
        `${valueRepr(value[field])} is not a tag of the union (expected one of ${expected})`,
        { path: [...path, field] },
      );
    }
    return buildForType(member, value, path);
  }
  const reasons: string[] = [];
  for (const member of members) {
    try {
      return buildForType(member, value, path);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      // A nested union contributes its summary, so the message stays bounded.
      const reason = error instanceof NoUnionMember ? error.summary : error.message;
      reasons.push(`${canonicalTypeLabel(member)}: ${reason}`);
    }
  }
  const summary = `value matches no member of ${canonicalTypeLabel(typ)}`;
  throw new NoUnionMember(`${summary}: ${reasons.join("; ")}`, summary, path);
}

/**
 * Return a tagged union's first tag field and the member `value`'s tag selects (`null` when
 * that field is missing or holds an unknown tag), or `null` when the members share no tag
 * field (an untagged union).
 */
function taggedMember(
  members: readonly Type[],
  value: ValueRecord,
): [string, Type | null] | null {
  const tagFields = membersTagFields(members);
  if (tagFields.length === 0) return null;
  const field = tagFields[0]!;
  const member = Object.hasOwn(value, field) ? memberWithTag(members, field, value[field]) : null;
  return [field, member];
}

/** Return the sorted values of one tag field across union members. */
function tagValues(members: readonly Type[], field: string): string[] {
  return members.map((member) => singleTag(member.fields!.get(field)!)!).sort(compareCodeUnits);
}

/**
 * A value matches no member of an untagged union; `summary` is the message without the
 * per-member reasons. Its `name` stays `TypeCheckError`, so it reads as that class.
 */
class NoUnionMember extends TypeCheckError {
  readonly summary: string;

  constructor(message: string, summary: string, path: ValuePath) {
    super(message, { path });
    this.summary = summary;
  }
}

/**
 * Check `value` against the plain scalar or flat-list kind `declared` at `path` and return it
 * as `declared` stores it.
 *
 * @throws TypeCheckError at `path`, or at an element's index for a flat-list element.
 */
function buildScalarOrList(declared: ValueKind, value: unknown, path: ValuePath): TypedValue {
  if (isListKind(declared)) {
    const label = kindLabel(declared);
    if (!isList(value)) {
      throw new TypeCheckError(`${valueRepr(value)} is not a list for declared ${label}`, {
        path,
      });
    }
    const kind = declared as StorageKind;
    const element = listElementKind(declared);
    if (element === null) return new TypedValue(kind, jsonPart(value, path, "refuse"));
    const items: unknown[] = [];
    for (let index = 0; index < value.length; index++) {
      let item = value[index];
      rejectNonValue(item, [...path, index]);
      if (!scalarMatches(element, item)) {
        throw new TypeCheckError(
          `list element ${valueRepr(item)} is not ${element} (declared ${label})`,
          { path: [...path, index] },
        );
      }
      if (element === ValueKind.OBJECT) item = jsonPart(item, [...path, index], "mapping");
      items.push(item);
    }
    // Converting in a second pass keeps Python's error order: every item is checked first.
    const coerced = items.map((item, index) => coerceScalar(element, item, [...path, index]));
    return new TypedValue(kind, coerced as Value[]);
  }

  if (!scalarMatches(declared, value)) {
    let message = `${valueRepr(value)} does not match declared type ${declared}`;
    const temporal = declared === ValueKind.DATE || declared === ValueKind.DATETIME;
    if (temporal && typeof value === "string") message += `: ${temporalError(declared, value)}`;
    throw new TypeCheckError(message, { path });
  }
  // The stored kind follows the value, so a `float` carrier never holds a bigint.
  if (declared === ValueKind.NUMBER && typeof value === "bigint") {
    return new TypedValue(ValueKind.INTEGER, value);
  }
  const stored = declared === ValueKind.OBJECT ? jsonPart(value, path, "mapping") : value;
  return new TypedValue(declared as StorageKind, coerceScalar(declared, stored, path) as Value);
}

/**
 * Return `value` with only its file and object representation read as `typ` reads it.
 *
 * At a `file` position a file's JSON form becomes a `FileRef`; at an `object` position (a
 * plain `object` value, a `list(object)` item, an `object` record field) each nested `FileRef`
 * becomes its mapping. Nothing else converts: an int at a `float` position stays an int, and a
 * value that doesn't fit `typ` is walked as far as it fits and otherwise kept. A union member
 * is picked as `buildValueAs` picks it: by its tag, else the first member the value
 * satisfies. Only the containers on the path to a change are copied; `value` itself is
 * returned when nothing changes (always so when `typ` has no `file` and no `object` position).
 *
 * An `object` position reads its value with the write check's JSON walk, so a `-0` there
 * becomes `0` as the write check stores it, and a value there that isn't a JSON value (NaN, a
 * container inside itself, a non-JSON part) is kept whole.
 *
 * Example: with `Rec` the record `{f: file, n: float}`,
 * `normalizeFiles(Rec, {f: {uri: "a", mime: null, name: null}, n: 1n})` gives
 * `{f: new FileRef({uri: "a"}), n: 1n}`.
 */
export function normalizeFiles(typ: Type, value: Value): Value {
  if (value === null || !readsFiles(typ)) return value;
  if (typ.variants !== undefined) {
    const member = unionMemberOf(typ, value);
    return member === null ? value : normalizeFiles(member, value);
  }
  if (typ.fields !== undefined) {
    if (!isRecord(value)) return value;
    const changed = new Map<string, Value>();
    for (const [name, field] of typ.fields) {
      if (!Object.hasOwn(value, name)) continue;
      const normalized = normalizeFiles(field, value[name]!);
      if (!Object.is(normalized, value[name])) changed.set(name, normalized);
    }
    if (changed.size === 0) return value;
    const copy: ValueRecord = {};
    for (const name of Object.keys(value)) {
      setField(copy, name, changed.has(name) ? changed.get(name)! : value[name]!);
    }
    return copy;
  }
  if (isListKind(typ.kind)) {
    if (!isList(value)) return value;
    // `readsFiles` admits a list only with an element type or as a flat `list(object)`.
    const element = typ.element ?? Type.scalar(ValueKind.OBJECT);
    const items = (value as Value[]).map((item) => normalizeFiles(element, item));
    return items.every((item, index) => Object.is(item, value[index])) ? value : items;
  }
  if (typ.kind === ValueKind.FILE) {
    if (!isFileForm(value)) return value;
    // A form whose text holds a lone surrogate isn't a yian value, so it is kept.
    const fits = FILE_FIELDS.every((name) => nonValueReason(value[name]) === null);
    return fits ? new FileRef(value) : value;
  }
  if (typ.kind === ValueKind.OBJECT && (isList(value) || isRecord(value))) {
    try {
      return jsonPart(value, [], "mapping");
    } catch (error) {
      if (error instanceof TypeCheckError) return value;
      throw error;
    }
  }
  return value;
}

/**
 * Return `typed` with its value read through `typ` by `normalizeFiles`; a `null` `typ` reads
 * nothing.
 *
 * A decoded value holds each file as its JSON form unless its whole kind is `file`; reading it
 * through the type it was written as turns each one back into a `FileRef`. The result keeps
 * `typed`'s kind and writes the same wire form. `typed` itself is returned when nothing
 * changes.
 *
 * Example: with `typ` the type `list(file)` and `form` a file's JSON form,
 * `normalizeTypedFiles(typ, new TypedValue("list[Any]", [form]))` holds `[new FileRef(form)]`.
 */
export function normalizeTypedFiles(typ: Type | null, typed: TypedValue): TypedValue {
  if (typ === null) return typed;
  const value = normalizeFiles(typ, typed.value);
  return Object.is(value, typed.value) ? typed : new TypedValue(typed.kind, value);
}

/**
 * Whether `typ` has a `file` or an unrefined `object` position at any depth, so reading a
 * value through it may change how a file is held. Unlike `hasFile`, it also counts plain
 * `object` and flat `list(object)`, whose nested `FileRef`s are read as mappings.
 */
function readsFiles(typ: Type): boolean {
  if (typ.kind === ValueKind.FILE || isPlainObject(typ)) return true;
  if (typ.kind === ValueKind.LIST_OBJECT && typ.element === undefined) return true;
  if (typ.element !== undefined && readsFiles(typ.element)) return true;
  if (typ.fields !== undefined && [...typ.fields.values()].some(readsFiles)) return true;
  return typ.variants !== undefined && typ.variants.some(readsFiles);
}

/**
 * Return the member of the union `typ` that `buildValueAs` checks `value` against, or `null`
 * when the value is not a record, has a missing or unknown tag, or matches no member.
 *
 * A tagged union's member comes from `taggedMember`, as in `buildForUnion`. For an untagged
 * union it builds the value against each member again, deliberately: the first member it
 * satisfies is the one the write check took.
 */
function unionMemberOf(typ: Type, value: Value): Type | null {
  if (!isRecord(value)) return null;
  const tagged = taggedMember(typ.variants!, value);
  if (tagged !== null) return tagged[1];
  for (const member of typ.variants!) {
    try {
      buildForType(member, value, []);
    } catch (error) {
      if (error instanceof TypeCheckError) continue;
      throw error;
    }
    return member;
  }
  return null;
}
