/**
 * Provider-neutral messages with a stable JSON shape.
 *
 * A `Message` holds typed parts; `Message.toJSON` writes the `yian.message.v1` JSON shape and
 * `Message.fromJSON` reads it back strictly. Every class checks its values on construction, so
 * an invalid message cannot exist: opaque provider data and tool-call arguments must be plain
 * JSON within yian's nesting limit. The check walks values iteratively, so a deep value throws
 * `MessageFormatError`, never `RangeError`. Fields are camelCase; JSON keys are snake_case.
 */

import { current } from "../defaults.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";

/** The format tag every message's JSON carries. */
export const MESSAGE_FORMAT = "yian.message.v1";

/** Who wrote a message. */
export type Role = "system" | "user" | "assistant" | "tool";

const ROLES: ReadonlySet<unknown> = new Set(["system", "user", "assistant", "tool"]);
const MESSAGE_KEYS: ReadonlySet<string> = new Set(["format", "role", "parts", "protocol", "provider_payload"]);
const PLAIN_KEY = /^[A-Za-z_][A-Za-z0-9_]*$/;
const UNPAIRED_SURROGATE = "string has an unpaired surrogate";

/**
 * A message, part or value that does not fit the `yian.message.v1` shape.
 *
 * The text never echoes payload values, which may hold provider secrets.
 */
export class MessageFormatError extends Error {
  override name = "MessageFormatError";
  /** JSON path of the offending value, e.g. `parts[2].arguments`; empty for the value itself. */
  readonly path: string;
  /** What is wrong with it. */
  readonly reason: string;

  constructor(path: string, reason: string) {
    super(path ? `${path}: ${reason}` : reason);
    this.path = path;
    this.reason = reason;
  }

  /** Return the same error with its path placed under `prefix`, the path of the containing value. */
  within(prefix: string): MessageFormatError {
    return new MessageFormatError(join(prefix, this.path), this.reason);
  }
}

/** Join two JSON paths, e.g. `parts[1]` and `text` into `parts[1].text`. */
function join(prefix: string, path: string): string {
  if (!prefix) return path;
  if (!path) return prefix;
  return path.startsWith("[") ? prefix + path : `${prefix}.${path}`;
}

/** Return a key's path segment without a leading dot: `key`, or `["odd key"]`. */
function segment(key: string): string {
  return PLAIN_KEY.test(key) ? key : `[${JSON.stringify(key)}]`;
}

/** Return a value's type name for error text: a `typeof` name, `null`, `array`, `object` or a class name. */
function typeName(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value !== "object") return typeof value;
  const proto: unknown = Object.getPrototypeOf(value);
  if (proto === Object.prototype || proto === null) return "object";
  const name: unknown = (value as { constructor?: { name?: unknown } }).constructor?.name;
  return typeof name === "string" && name !== "" ? name : "object";
}

/** Set `target[key]`, defining a `"__proto__"` key so it stays data rather than setting the prototype. */
function put(target: JsonValue[] | JsonObject, key: string | number, value: JsonValue): void {
  if (key === "__proto__") {
    Object.defineProperty(target, key, { value, writable: true, enumerable: true, configurable: true });
  } else {
    (target as Record<string | number, JsonValue>)[key] = value;
  }
}

/** A path in `checkJsonValue`: the root path, or a parent and a key or index. */
type Where = string | { parent: Where; segment: string | number };

/** Return the JSON path text for a `checkJsonValue` parent chain. */
function formatPath(where: Where): string {
  const segments: string[] = [];
  while (typeof where !== "string") {
    const step = where.segment;
    segments.push(typeof step === "number" ? `[${step}]` : segment(step));
    where = where.parent;
  }
  return segments.reduceRight(join, where);
}

/** One value still to check, and where its copy goes. */
interface Task {
  item: unknown;
  where: Where;
  level: number;
  into: JsonValue[] | JsonObject;
  key: string | number;
}

/**
 * Check that `value` is plain JSON within yian's nesting limit and return a deep copy.
 *
 * Accepted: plain objects (prototype `Object.prototype` or `null`, no symbol keys), whose own
 * enumerable string keys are copied as own data properties; arrays without holes; strings;
 * `bigint`s; finite numbers (`-0` copied as `0`); booleans; `null`. No string or key may hold an
 * unpaired surrogate. Each object and array is one level of `current().maxValueNesting`. The
 * walk uses an explicit stack, so a value of any depth throws `MessageFormatError`.
 *
 * @param value The value to check.
 * @param path JSON path of `value`, used in the error.
 * @returns A deep copy of `value`.
 * @throws {MessageFormatError} Naming the path of the first offending value in document order.
 */
export function checkJsonValue(value: unknown, path: string): JsonValue {
  const limit = current().maxValueNesting;
  const root: JsonValue[] = [];
  // Each task keeps its parent chain and the path text is built only on failure, so a large
  // valid value costs O(nodes) rather than O(nodes x depth).
  const stack: Task[] = [{ item: value, where: path, level: 1, into: root, key: 0 }];
  while (stack.length > 0) {
    const { item, where, level, into, key } = stack.pop()!;
    if (typeof item === "string") {
      if (!item.isWellFormed()) throw new MessageFormatError(formatPath(where), UNPAIRED_SURROGATE);
      put(into, key, item);
    } else if (typeof item === "number") {
      if (!Number.isFinite(item)) throw new MessageFormatError(formatPath(where), "float is not finite");
      // yian has no negative zero, as `parseJson` reads it.
      put(into, key, item === 0 ? 0 : item);
    } else if (item === null || typeof item === "boolean" || typeof item === "bigint") {
      put(into, key, item);
    } else if (Array.isArray(item)) {
      if (level > limit) throw new MessageFormatError(formatPath(where), `value nests deeper than ${limit} levels`);
      const copy: JsonValue[] = [];
      put(into, key, copy);
      // Pushed last to first so they pop in order; a hole reads as `undefined` and is refused.
      for (let i = item.length - 1; i >= 0; i--) {
        stack.push({ item: item[i], where: { parent: where, segment: i }, level: level + 1, into: copy, key: i });
      }
    } else if (typeName(item) === "object") {
      if (level > limit) throw new MessageFormatError(formatPath(where), `value nests deeper than ${limit} levels`);
      const object = item as Record<string, unknown>;
      if (Object.getOwnPropertySymbols(object).length > 0) {
        throw new MessageFormatError(formatPath(where), "object key is symbol, not string");
      }
      const keys = Object.keys(object);
      for (const each of keys) {
        if (!each.isWellFormed()) throw new MessageFormatError(formatPath(where), "object key has an unpaired surrogate");
      }
      const copy: JsonObject = {};
      put(into, key, copy);
      // Pushed last to first so the first offending value in document order is reported.
      for (let i = keys.length - 1; i >= 0; i--) {
        const each = keys[i]!;
        stack.push({ item: object[each], where: { parent: where, segment: each }, level: level + 1, into: copy, key: each });
      }
    } else {
      throw new MessageFormatError(formatPath(where), `${typeName(item)} is not a JSON value`);
    }
  }
  return root[0]!;
}

/** Return a deep copy of a value `checkJsonValue` accepted; its nesting limit bounds the recursion. */
function copyJson<T extends JsonValue>(value: T): T {
  if (Array.isArray(value)) return value.map(copyJson) as T;
  if (typeof value !== "object" || value === null) return value;
  const copy: JsonObject = {};
  for (const key of Object.keys(value)) put(copy, key, copyJson(value[key]!));
  return copy as T;
}

/** Return `value` if it is a string with no unpaired surrogate, which could not be sent as JSON. */
function checkString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new MessageFormatError(path, `expected string, got ${typeName(value)}`);
  if (!value.isWellFormed()) throw new MessageFormatError(path, UNPAIRED_SURROGATE);
  return value;
}

/** Return `value` as `checkString` does, or `null` when it is `null` or `undefined`. */
function optionalString(value: unknown, path: string): string | null {
  return value === null || value === undefined ? null : checkString(value, path);
}

/** Return a copy of `value` if it is a JSON object, or `null` when it is `null` or `undefined`. */
function optionalObject(value: unknown, path: string): JsonObject | null {
  if (value === null || value === undefined) return null;
  const kind = typeName(value);
  if (kind !== "object") throw new MessageFormatError(path, `expected an object, got ${kind}`);
  return checkJsonValue(value, path) as JsonObject;
}

/** Return the JSON of a part: `type` first, then each field whose value isn't `null`, payloads copied. */
function partJSON(type: string, fields: Record<string, JsonValue>): JsonObject {
  const data: JsonObject = { type };
  for (const [key, value] of Object.entries(fields)) {
    if (value !== null) data[key] = copyJson(value);
  }
  return data;
}

/** Text the model wrote or the user sent. */
export class TextPart {
  readonly type = "text" as const;
  /** The text. */
  readonly text: string;
  /** Opaque provider data, kept untouched. */
  readonly providerPayload: JsonObject | null;

  constructor(text: string, providerPayload?: JsonObject | null) {
    this.text = checkString(text, "text");
    this.providerPayload = optionalObject(providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields. */
  toJSON(): JsonObject {
    return partJSON(this.type, { text: this.text, provider_payload: this.providerPayload });
  }
}

/** An image or file, given inline as base64 or by URL; exactly one of the two. */
export class MediaPart {
  readonly type = "media" as const;
  /** The media type, e.g. `image/png`. */
  readonly mime: string;
  /** Base64 content. */
  readonly data: string | null;
  /** Where the content lives. */
  readonly url: string | null;
  /** A file name, if any. */
  readonly name: string | null;
  /** Opaque provider data, e.g. OpenAI's image `detail`. */
  readonly providerPayload: JsonObject | null;

  constructor(options: {
    mime: string;
    data?: string | null;
    url?: string | null;
    name?: string | null;
    providerPayload?: JsonObject | null;
  }) {
    this.mime = checkString(options.mime, "mime");
    this.data = optionalString(options.data, "data");
    this.url = optionalString(options.url, "url");
    this.name = optionalString(options.name, "name");
    if ((this.data === null) === (this.url === null)) {
      throw new MessageFormatError("", "media needs exactly one of data or url");
    }
    this.providerPayload = optionalObject(options.providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields. */
  toJSON(): JsonObject {
    return partJSON(this.type, {
      mime: this.mime,
      data: this.data,
      url: this.url,
      name: this.name,
      provider_payload: this.providerPayload,
    });
  }
}

/**
 * A tool call the model asked for.
 *
 * By adapter convention (not enforced), `arguments` is `null` exactly when the raw text failed
 * to parse; both may be set when the adapter keeps the raw text.
 */
export class ToolCallPart {
  readonly type = "tool_call" as const;
  /** The call id that the tool result answers. */
  readonly id: string;
  /** The tool name. */
  readonly name: string;
  /** The parsed arguments. */
  readonly arguments: JsonObject | null;
  /** The arguments text as the model sent it. */
  readonly rawArguments: string | null;
  /** Opaque provider data, e.g. a Responses item id. */
  readonly providerPayload: JsonObject | null;

  constructor(options: {
    id: string;
    name: string;
    arguments: JsonObject | null;
    rawArguments?: string | null;
    providerPayload?: JsonObject | null;
  }) {
    this.id = checkString(options.id, "id");
    this.name = checkString(options.name, "name");
    this.rawArguments = optionalString(options.rawArguments, "raw_arguments");
    // Read as a property: a binding named `arguments` is a SyntaxError in strict code.
    const args = options.arguments;
    if ((args === null || args === undefined) && this.rawArguments === null) {
      throw new MessageFormatError("arguments", "a tool call needs arguments or raw_arguments");
    }
    this.arguments = optionalObject(args, "arguments");
    this.providerPayload = optionalObject(options.providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields. */
  toJSON(): JsonObject {
    return partJSON(this.type, {
      id: this.id,
      name: this.name,
      arguments: this.arguments,
      raw_arguments: this.rawArguments,
      provider_payload: this.providerPayload,
    });
  }
}

/** The result of a tool call, sent back to the model. */
export class ToolResultPart {
  readonly type = "tool_result" as const;
  /** The id of the call this answers. */
  readonly callId: string;
  /** The tool name. */
  readonly name: string;
  /** The result text. */
  readonly content: string;
  /** Whether the tool failed. */
  readonly isError: boolean;
  /** Opaque provider data, kept untouched. */
  readonly providerPayload: JsonObject | null;

  constructor(options: {
    callId: string;
    name: string;
    content: string;
    isError?: boolean;
    providerPayload?: JsonObject | null;
  }) {
    this.callId = checkString(options.callId, "call_id");
    this.name = checkString(options.name, "name");
    this.content = checkString(options.content, "content");
    // Only an absent `isError` defaults: `is_error` has no `null` form, so `null` is refused.
    const isError: unknown = options.isError === undefined ? false : options.isError;
    if (typeof isError !== "boolean") {
      throw new MessageFormatError("is_error", `expected boolean, got ${typeName(isError)}`);
    }
    this.isError = isError;
    this.providerPayload = optionalObject(options.providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields and `is_error: false`. */
  toJSON(): JsonObject {
    return partJSON(this.type, {
      call_id: this.callId,
      name: this.name,
      content: this.content,
      is_error: this.isError || null,
      provider_payload: this.providerPayload,
    });
  }
}

/** The model's reasoning text. */
export class ReasoningPart {
  readonly type = "reasoning" as const;
  /** The reasoning, or a summary of it; may be empty. */
  readonly text: string;
  /** Opaque provider data, e.g. an Anthropic signature or Responses `encrypted_content`. */
  readonly providerPayload: JsonObject | null;

  constructor(text: string, providerPayload?: JsonObject | null) {
    this.text = checkString(text, "text");
    this.providerPayload = optionalObject(providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields. */
  toJSON(): JsonObject {
    return partJSON(this.type, { text: this.text, provider_payload: this.providerPayload });
  }
}

/** The model's refusal to answer, kept apart from its text. */
export class RefusalPart {
  readonly type = "refusal" as const;
  /** The refusal message; may be empty. */
  readonly text: string;
  /** Opaque provider data, e.g. the Responses message id and phase. */
  readonly providerPayload: JsonObject | null;

  constructor(text: string, providerPayload?: JsonObject | null) {
    this.text = checkString(text, "text");
    this.providerPayload = optionalObject(providerPayload, "provider_payload");
  }

  /** Return the part's JSON, omitting `null` fields. */
  toJSON(): JsonObject {
    return partJSON(this.type, { text: this.text, provider_payload: this.providerPayload });
  }
}

/** One piece of a message's content. */
export type Part = TextPart | MediaPart | ToolCallPart | ToolResultPart | ReasoningPart | RefusalPart;

/** Return whether `value` is an instance of one of the part classes. */
export function isPart(value: unknown): value is Part {
  return (
    value instanceof TextPart ||
    value instanceof MediaPart ||
    value instanceof ToolCallPart ||
    value instanceof ToolResultPart ||
    value instanceof ReasoningPart ||
    value instanceof RefusalPart
  );
}

/** How to read one part type's JSON. */
interface PartReader {
  /** Every key besides `type`. */
  keys: readonly string[];
  /** The keys the JSON must hold; the others are omitted when `null` (or, for `is_error`, `false`). */
  required: readonly string[];
  make: (data: Record<string, unknown>) => Part;
}

// Constructor arguments come straight from the JSON; each constructor checks them.
const PART_READERS: ReadonlyMap<string, PartReader> = new Map<string, PartReader>([
  [
    "text",
    {
      keys: ["text", "provider_payload"],
      required: ["text"],
      make: (d) => new TextPart(d["text"] as string, d["provider_payload"] as JsonObject),
    },
  ],
  [
    "media",
    {
      keys: ["mime", "data", "url", "name", "provider_payload"],
      required: ["mime"],
      make: (d) =>
        new MediaPart({
          mime: d["mime"] as string,
          data: d["data"] as string,
          url: d["url"] as string,
          name: d["name"] as string,
          providerPayload: d["provider_payload"] as JsonObject,
        }),
    },
  ],
  [
    "tool_call",
    {
      keys: ["id", "name", "arguments", "raw_arguments", "provider_payload"],
      required: ["id", "name"],
      make: (d) =>
        new ToolCallPart({
          id: d["id"] as string,
          name: d["name"] as string,
          arguments: (d["arguments"] ?? null) as JsonObject | null,
          rawArguments: d["raw_arguments"] as string,
          providerPayload: d["provider_payload"] as JsonObject,
        }),
    },
  ],
  [
    "tool_result",
    {
      keys: ["call_id", "name", "content", "is_error", "provider_payload"],
      required: ["call_id", "name", "content"],
      make: (d) =>
        new ToolResultPart({
          callId: d["call_id"] as string,
          name: d["name"] as string,
          content: d["content"] as string,
          isError: d["is_error"] as boolean,
          providerPayload: d["provider_payload"] as JsonObject,
        }),
    },
  ],
  [
    "reasoning",
    {
      keys: ["text", "provider_payload"],
      required: ["text"],
      make: (d) => new ReasoningPart(d["text"] as string, d["provider_payload"] as JsonObject),
    },
  ],
  [
    "refusal",
    {
      keys: ["text", "provider_payload"],
      required: ["text"],
      make: (d) => new RefusalPart(d["text"] as string, d["provider_payload"] as JsonObject),
    },
  ],
]);

// Keys `toJSON` omits when `null`, so an explicit `null` is not JSON `toJSON` writes.
const OBJECT_KEYS: ReadonlySet<string> = new Set(["provider_payload", "arguments"]);
const NULLABLE_KEYS: ReadonlySet<string> = new Set([
  "protocol",
  "provider_payload",
  "data",
  "url",
  "name",
  "arguments",
  "raw_arguments",
]);

/** Throw for the first of `keys` that `data` holds as an explicit `null` where `toJSON` would omit it. */
function refuseNulls(data: Record<string, unknown>, keys: readonly string[], path: string): void {
  for (const key of keys) {
    if (NULLABLE_KEYS.has(key) && Object.hasOwn(data, key) && data[key] === null) {
      const expected = OBJECT_KEYS.has(key) ? "an object" : "string";
      throw new MessageFormatError(join(path, key), `expected ${expected}, got null`);
    }
  }
}

/** Read one part's JSON strictly; errors name paths under `path`. */
function partFromJSON(data: unknown, path: string): Part {
  const kind = typeName(data);
  if (kind !== "object") throw new MessageFormatError(path, `expected an object, got ${kind}`);
  const record = data as Record<string, unknown>;
  if (!Object.hasOwn(record, "type")) throw new MessageFormatError(join(path, "type"), "missing");
  const type = record["type"];
  const reader = typeof type === "string" ? PART_READERS.get(type) : undefined;
  if (reader === undefined) throw new MessageFormatError(join(path, "type"), "unknown part type");
  for (const key of Object.keys(record)) {
    if (key !== "type" && !reader.keys.includes(key)) {
      throw new MessageFormatError(join(path, segment(key)), "unknown key");
    }
  }
  for (const key of reader.required) {
    if (!Object.hasOwn(record, key)) throw new MessageFormatError(join(path, key), "missing");
  }
  refuseNulls(record, reader.keys, path);
  try {
    return reader.make(record);
  } catch (error) {
    if (error instanceof MessageFormatError) throw error.within(path);
    throw error;
  }
}

/** One message of a conversation. */
export class Message {
  /** `system`, `user`, `assistant` or `tool`. */
  readonly role: Role;
  /** The content, a frozen copy of the array given. */
  readonly parts: readonly Part[];
  /**
   * Protocol id of the adapter that produced this message, e.g. `openai_responses` or
   * `catalog:<provider>`. Every provider payload in the message belongs to it, so an adapter
   * can drop foreign payloads after a model switch.
   */
  readonly protocol: string | null;
  /** Message-level opaque provider data. */
  readonly providerPayload: JsonObject | null;

  constructor(options: {
    role: Role;
    parts: readonly Part[];
    protocol?: string | null;
    providerPayload?: JsonObject | null;
  }) {
    if (!ROLES.has(options.role)) throw new MessageFormatError("role", "expected system, user, assistant or tool");
    this.role = options.role;
    const parts: unknown = options.parts;
    if (!Array.isArray(parts)) throw new MessageFormatError("parts", `expected an array, got ${typeName(parts)}`);
    parts.forEach((part: unknown, index) => {
      if (!isPart(part)) throw new MessageFormatError(`parts[${index}]`, `expected a part, got ${typeName(part)}`);
    });
    this.parts = Object.freeze([...(parts as Part[])]);
    this.protocol = optionalString(options.protocol, "protocol");
    this.providerPayload = optionalObject(options.providerPayload, "provider_payload");
  }

  /**
   * Return the message as `yian.message.v1` JSON: `format`, `role`, `parts`, then `protocol` and
   * `provider_payload` unless `null`. Payloads are copies, so changing the result leaves the
   * message unchanged.
   */
  toJSON(): JsonObject {
    const data: JsonObject = {
      format: MESSAGE_FORMAT,
      role: this.role,
      parts: this.parts.map((part) => part.toJSON()),
    };
    if (this.protocol !== null) data["protocol"] = this.protocol;
    if (this.providerPayload !== null) data["provider_payload"] = copyJson(this.providerPayload);
    return data;
  }

  /**
   * Read a message from `yian.message.v1` JSON, strictly.
   *
   * @param data The JSON object, as `toJSON` writes it.
   * @returns The message.
   * @throws {MessageFormatError} Naming the JSON path, for a missing or wrong format, an unknown
   *   key or part type, a wrong type, a missing field, an explicit `null` where `toJSON` omits
   *   the key, or a value that is not plain JSON within yian's limits.
   */
  static fromJSON(data: unknown): Message {
    const kind = typeName(data);
    if (kind !== "object") throw new MessageFormatError("", `expected an object, got ${kind}`);
    const record = data as Record<string, unknown>;
    if (!Object.hasOwn(record, "format")) throw new MessageFormatError("format", "missing");
    // Python's repr quoting, kept so both ports word the error alike.
    if (record["format"] !== MESSAGE_FORMAT) throw new MessageFormatError("format", `expected '${MESSAGE_FORMAT}'`);
    for (const key of Object.keys(record)) {
      if (!MESSAGE_KEYS.has(key)) throw new MessageFormatError(segment(key), "unknown key");
    }
    for (const key of ["role", "parts"]) {
      if (!Object.hasOwn(record, key)) throw new MessageFormatError(key, "missing");
    }
    const rawParts = record["parts"];
    if (!Array.isArray(rawParts)) {
      throw new MessageFormatError("parts", `expected an array, got ${typeName(rawParts)}`);
    }
    refuseNulls(record, ["protocol", "provider_payload"], "");
    const parts: Part[] = [];
    for (let i = 0; i < rawParts.length; i++) parts.push(partFromJSON(rawParts[i], `parts[${i}]`));
    return new Message({
      role: record["role"] as Role,
      parts,
      protocol: record["protocol"] as string,
      providerPayload: record["provider_payload"] as JsonObject,
    });
  }

  /** Return the text parts' texts joined in order with `"\n"`; reasoning and refusal parts are left out. */
  text(): string {
    return this.parts
      .filter((part) => part instanceof TextPart)
      .map((part) => part.text)
      .join("\n");
  }

  /** Return the tool-call parts in order. */
  toolCalls(): ToolCallPart[] {
    return this.parts.filter((part) => part instanceof ToolCallPart);
  }
}
