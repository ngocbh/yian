/**
 * Build request bodies the same way in every adapter: merge a provider block into a body yian
 * built, encode the body as JSON, and render a user message's author.
 *
 * A provider block mirrors the request body of the API it reaches. Adapters build their own
 * fields first, then merge the block as written: objects key by key, arrays and scalars whole,
 * `null` meaning unset.
 *
 * A user message's author goes to the model as an `<authored_message author="...">` …
 * `</authored_message>` wrapper around its parts, and the tag is escaped in every user and tool
 * text, so only the adapter can write it.
 */

import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { copyJson, isPlainObject, Message, type Part, put, TextPart, ToolResultPart, typeName } from "./messages.ts";

/**
 * Deep-merge a provider block into a request body, in place.
 *
 * Objects merge key by key; arrays and scalars from `block` replace the body's value whole; a
 * `null` in `block` means unset and leaves the body alone. `block` is copied, never mutated or
 * shared. Keys are written as own data properties, so `"__proto__"` is an ordinary key. It
 * recurses once per nesting level: provider blocks are shallow config, checked at
 * load.
 *
 * @param body The request body yian built. Updated in place.
 * @param block The provider block, as written.
 * @returns `body`.
 */
export function mergeRequestBody(body: JsonObject, block: JsonObject): JsonObject {
  for (const key of Object.keys(block)) {
    const value = block[key]!;
    if (value === null) continue;
    if (isPlainObject(value)) {
      let current = Object.hasOwn(body, key) ? body[key] : undefined;
      if (!isPlainObject(current)) {
        current = {};
        put(body, key, current);
      }
      mergeRequestBody(current, value);
    } else {
      put(body, key, copyJson(value));
    }
  }
  return body;
}

/** One step of encoding: a value with the comma and key before it, or a container closing. */
type Task =
  | {
      value: unknown;
      comma: boolean;
      /** Quoted when the task runs, since an unpaired surrogate's position needs `out.length` then. */
      key?: string;
    }
  | { leave: object; close: string };

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
      out += digits(item, notJson);
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
    } else if (isPlainObject(item)) {
      if (Object.getOwnPropertySymbols(item).length > 0) throw notJson("an object has a symbol key");
      if (active.has(item)) throw notJson("it contains a cycle");
      active.add(item);
      // Each value is read once, so a getter runs once.
      const entries = Object.keys(item)
        .map((key): [string, unknown] => [key, item[key]])
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

/** Return a `bigint`'s digits, refusing one too large for the runtime to write. */
function digits(n: bigint, notJson: (problem: string) => TypeError): string {
  try {
    return n.toString();
  } catch (error) {
    // A runtime may refuse to stringify a huge bigint (JavaScriptCore reports running out of
    // memory as a RangeError).
    if (error instanceof RangeError) throw notJson("integer is too large for this runtime");
    throw error;
  }
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

/** The tag name of the wrapper that carries a user message's author. */
export const AUTHOR_TAG = "authored_message";

// Any run of whitespace, backslashes, slashes, format characters and other default-ignorable
// characters (zero-width, bidi controls, variation selectors, Hangul fillers) may sit between
// `<` and the name, because `<\/authored_message>` (the JSON escaped-slash form a tool may return
// as text) and an invisible character after `<` read as the tag. A `<` is escaped too when the
// rest of its text is such a run and the start of the name, because adapters join parts and
// Anthropic merges consecutive user and tool messages, so the next part or message can finish
// the tag. Global regexes are stateful under `.test()`, so a change is detected by comparing the
// replaced string with the original.
const FILLER = String.raw`[\s\\\/\p{Cf}\p{Default_Ignorable_Code_Point}]`;
const NAME_STARTS = Array.from({ length: AUTHOR_TAG.length }, (_, i) => AUTHOR_TAG.slice(0, AUTHOR_TAG.length - 1 - i));
const TAG_OPENING = new RegExp(`<(?=${FILLER}*(?:${AUTHOR_TAG}|(?:${NAME_STARTS.join("|")})$))`, "giu");

/**
 * Return `text` with the `<` of every opening of the author tag, in any case, escaped as `&lt;`,
 * and of a `<` that ends the text with only filler and the start of the name after it.
 */
export function escapeAuthorTags(text: string): string {
  return text.replace(TAG_OPENING, "&lt;");
}

/** Return `value` escaped for an XML attribute: `&`, `<`, `>`, `"` and line breaks. */
function attr(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("\n", "&#10;")
    .replaceAll("\r", "&#13;");
}

/** Return the wrapper's opening tag naming `author`. */
export function authorOpen(author: string): string {
  return `<${AUTHOR_TAG} author="${attr(author)}">`;
}

/** Return the wrapper's closing tag. */
export function authorClose(): string {
  return `</${AUTHOR_TAG}>`;
}

/** One value still to escape in `escapeAuthorTagsInJson`, and where its copy goes. */
interface EscapeTask {
  item: JsonValue;
  into: JsonValue[] | JsonObject;
  key: string | number;
}

/**
 * Return `value` with every string and object key passed through `escapeAuthorTags`, at any
 * depth, or `value` itself when nothing changes. When two keys of one object escape to the same
 * text, the first in key order is kept and the other dropped.
 */
export function escapeAuthorTagsInJson(value: JsonValue): JsonValue {
  // An explicit stack, as `checkJsonValue` uses, since the value comes from `parseJson`, which
  // reads any depth.
  const root: JsonValue[] = [];
  const stack: EscapeTask[] = [{ item: value, into: root, key: 0 }];
  let changed = false;
  while (stack.length > 0) {
    const { item, into, key } = stack.pop()!;
    if (typeof item === "string") {
      const escaped = escapeAuthorTags(item);
      if (escaped !== item) changed = true;
      put(into, key, escaped);
    } else if (Array.isArray(item)) {
      const copy: JsonValue[] = [];
      put(into, key, copy);
      for (let i = item.length - 1; i >= 0; i--) stack.push({ item: item[i]!, into: copy, key: i });
    } else if (item !== null && typeof item === "object") {
      const copy: JsonObject = {};
      put(into, key, copy);
      const seen = new Set<string>();
      const children: EscapeTask[] = [];
      for (const each of Object.keys(item)) {
        const escaped = escapeAuthorTags(each);
        if (escaped !== each) changed = true;
        if (seen.has(escaped)) continue;
        seen.add(escaped);
        children.push({ item: item[each]!, into: copy, key: escaped });
      }
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i]!);
    } else {
      put(into, key, item);
    }
  }
  return changed ? root[0]! : value;
}

/** Return `part` with its text escaped, or `part` itself when nothing changes. */
function escapeText(part: TextPart): TextPart {
  const text = escapeAuthorTags(part.text);
  return text === part.text ? part : new TextPart(text, part.providerPayload);
}

/** Return `part` with its content escaped, or `part` itself when nothing changes. */
function escapeResult(part: ToolResultPart): ToolResultPart {
  const content = escapeAuthorTags(part.content);
  if (content === part.content) return part;
  return new ToolResultPart({
    callId: part.callId,
    name: part.name,
    content,
    isError: part.isError,
    providerPayload: part.providerPayload,
  });
}

/** Return `part` with `open + "\n"` before its text, or `"\n" + close` after it. */
function foldTags(part: TextPart, open: string | null, close: string | null): TextPart {
  const text = (open === null ? "" : `${open}\n`) + part.text + (close === null ? "" : `\n${close}`);
  return new TextPart(text, part.providerPayload);
}

/**
 * Return `message` as an adapter sends it: a user message's texts and a tool message's results
 * with the author tag escaped, and an authored user message's parts wrapped in `authorOpen` and
 * `authorClose` text parts.
 *
 * A user message is wrapped only when it has an author, `wrap` isn't `false`, and some text part
 * holds more than whitespace; an unauthored one stays unmarked, as the flow's own user. With
 * `inline`, the tags are folded into the first and last parts with `"\n"` when those are text,
 * and added as parts of their own otherwise, for routes that send a one-text message as a
 * string. System and assistant messages, and any message with nothing to change, are returned
 * as they are; a copy keeps the role, author, protocol and payload.
 */
export function withAuthorWrapper(message: Message, options: { wrap?: boolean; inline?: boolean } = {}): Message {
  // Every user and tool text is escaped, author or not, because a forged wrapper claims an
  // author even where no real one exists.
  let parts: Part[];
  if (message.role === "user") {
    parts = message.parts.map((part) => (part instanceof TextPart ? escapeText(part) : part));
    const author = message.author;
    const sendable = message.parts.some((part) => part instanceof TextPart && part.text.trim() !== "");
    // The wrapper always closes: Anthropic merges consecutive user turns, and the next message's
    // text must not read as this author's.
    if (author !== null && options.wrap !== false && sendable) {
      const open = authorOpen(author);
      const close = authorClose();
      if (!options.inline) {
        parts = [new TextPart(open), ...parts, new TextPart(close)];
      } else {
        const first = parts[0];
        if (first instanceof TextPart) parts[0] = foldTags(first, open, null);
        else parts.unshift(new TextPart(open));
        const end = parts.length - 1;
        const last = parts[end];
        if (last instanceof TextPart) parts[end] = foldTags(last, null, close);
        else parts.push(new TextPart(close));
      }
    }
  } else if (message.role === "tool") {
    parts = message.parts.map((part) => (part instanceof ToolResultPart ? escapeResult(part) : part));
  } else {
    return message;
  }
  if (parts.length === message.parts.length && parts.every((part, i) => part === message.parts[i])) return message;
  return new Message({
    role: message.role,
    parts,
    author: message.author,
    protocol: message.protocol,
    providerPayload: message.providerPayload,
  });
}
