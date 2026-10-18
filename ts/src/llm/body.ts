/**
 * Merge a provider block into a request body yian built, and encode the body as JSON.
 *
 * A provider block mirrors the request body of the API it reaches. Adapters build their own
 * fields first, then merge the block as written: objects key by key, arrays and scalars whole,
 * `null` meaning unset.
 */

import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { typeName } from "./messages.ts";

/**
 * Deep-merge a provider block into a request body, in place.
 *
 * Objects merge key by key; arrays and scalars from `block` replace the body's value whole; a
 * `null` in `block` means unset and leaves the body alone. `block` is copied, never mutated or
 * shared. Keys are written as own data properties, so `"__proto__"` is an ordinary key.
 *
 * @param body The request body yian built. Updated in place.
 * @param block The provider block, as written.
 * @returns `body`.
 */
export function mergeRequestBody(body: JsonObject, block: JsonObject): JsonObject {
  for (const key of Object.keys(block)) {
    const value = block[key]!;
    if (value === null) continue;
    if (isObject(value)) {
      let current = Object.hasOwn(body, key) ? body[key] : undefined;
      if (!isObject(current)) {
        current = {};
        put(body, key, current);
      }
      mergeRequestBody(current, value);
    } else {
      put(body, key, copy(value));
    }
  }
  return body;
}

/** Return whether a JSON value is an object (not an array or `null`). */
function isObject(value: JsonValue | undefined): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Set `target[key]` as an own data property, so a `"__proto__"` key never sets the prototype. */
function put(target: JsonObject, key: string, value: JsonValue): void {
  Object.defineProperty(target, key, { value, writable: true, enumerable: true, configurable: true });
}

/** Copy a block value so the body never shares a container with the block. */
function copy(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(copy);
  if (!isObject(value)) return value;
  const out: JsonObject = {};
  for (const key of Object.keys(value)) put(out, key, copy(value[key]!));
  return out;
}

/** One step of encoding: a value with the comma and key before it, or a container closing. */
type Task = { value: unknown; comma: boolean; key?: string } | { leave: object; close: string };

/**
 * Return `value` as compact JSON, keys in insertion order.
 *
 * A `bigint` is written as its digits and a finite number as `JSON.stringify` writes it (`-0`
 * as `0`); an `undefined` property is left out. The walk uses an explicit stack, so a value of
 * any depth encodes.
 *
 * @param value The value to encode.
 * @param what What the value is, starting each error message.
 * @returns The JSON text.
 * @throws {TypeError} For NaN or an infinity, an unpaired surrogate in a string or key (with its
 *   UTF-16 index in the output), a non-plain object or other non-JSON value, `undefined` in an
 *   array, a symbol key, or a cycle.
 */
export function encodeBody(value: unknown, what = "request body"): string {
  const notJson = (problem: string) => new TypeError(`${what} is not JSON: ${problem}`);
  let out = "";
  // Ancestors only, so a shared value writes twice and only a cycle is refused.
  const active = new Set<object>();
  const stack: Task[] = [{ value, comma: false }];
  while (stack.length > 0) {
    const task = stack.pop()!;
    if ("leave" in task) {
      out += task.close;
      active.delete(task.leave);
      continue;
    }
    if (task.comma) out += ",";
    if (task.key !== undefined) out += quote(task.key, out.length, what) + ":";
    const item = task.value;
    if (item === null) {
      out += "null";
    } else if (typeof item === "boolean") {
      out += item ? "true" : "false";
    } else if (typeof item === "string") {
      out += quote(item, out.length, what);
    } else if (typeof item === "bigint") {
      out += item.toString();
    } else if (typeof item === "number") {
      if (!Number.isFinite(item)) throw notJson(`${String(item)} is not a JSON number`);
      out += String(item);
    } else if (Array.isArray(item)) {
      if (active.has(item)) throw notJson("it contains a cycle");
      active.add(item);
      stack.push({ leave: item, close: "]" });
      // Pushed last to first so they pop in order; a hole reads as `undefined` and is refused.
      for (let i = item.length - 1; i >= 0; i--) stack.push({ value: item[i], comma: i > 0 });
      out += "[";
    } else if (typeof item === "object" && typeName(item) === "object") {
      if (Object.getOwnPropertySymbols(item).length > 0) throw notJson("an object has a symbol key");
      if (active.has(item)) throw notJson("it contains a cycle");
      active.add(item);
      const record = item as Record<string, unknown>;
      // Each value is read once, so a getter runs once.
      const entries = Object.keys(record)
        .map((key): [string, unknown] => [key, record[key]])
        .filter(([, each]) => each !== undefined);
      stack.push({ leave: item, close: "}" });
      for (let i = entries.length - 1; i >= 0; i--) {
        const [key, each] = entries[i]!;
        stack.push({ value: each, comma: i > 0, key });
      }
      out += "{";
    } else {
      throw notJson(`${typeName(item)} is not a JSON value`);
    }
  }
  return out;
}

/** Return `text` as a JSON string written at `at` in the output, refusing an unpaired surrogate. */
function quote(text: string, at: number, what: string): string {
  if (text.isWellFormed()) return JSON.stringify(text);
  let i = 0;
  for (;;) {
    const unit = text.charCodeAt(i);
    const next = text.charCodeAt(i + 1);
    if (unit >= 0xd800 && unit <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) i += 2;
    else if (unit >= 0xd800 && unit <= 0xdfff) break;
    else i++;
  }
  // The escaped prefix, less its closing quote, ends where the surrogate would be written.
  const position = at + JSON.stringify(text.slice(0, i)).length - 1;
  throw new TypeError(`${what} has an unpaired surrogate at position ${position}`);
}
