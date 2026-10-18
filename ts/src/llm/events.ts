/**
 * Stream events a model emits, and how they add up to a `Response`.
 *
 * An adapter turns a provider's stream into these events: deltas per part index, an
 * authoritative `PartDone` per part when the provider gives one, usage, and one `Finish`
 * last. Each event has a `type` discriminant equal to its class name. `ReplyAssembler` and
 * `collect` assemble them into a `Response`.
 */

import { JSONDecodeError, parseJson, type JsonObject } from "../typesys/json.ts";
import { ProviderError } from "./errors.ts";
import {
  checkJsonValue,
  isPart,
  Message,
  MessageFormatError,
  ReasoningPart,
  Response,
  TextPart,
  ToolCallPart,
  type Part,
  type Usage,
} from "./messages.ts";

/** More text for the text part at `index`. */
export class TextDelta {
  readonly type = "TextDelta" as const;
  /** The part's position in the reply. */
  readonly index: number;
  /** The new text. */
  readonly text: string;

  constructor(index: number, text: string) {
    this.index = index;
    this.text = text;
  }
}

/** More reasoning text for the reasoning part at `index`. */
export class ReasoningDelta {
  readonly type = "ReasoningDelta" as const;
  /** The part's position in the reply. */
  readonly index: number;
  /** The new text. */
  readonly text: string;

  constructor(index: number, text: string) {
    this.index = index;
    this.text = text;
  }
}

/** More of the tool call at `index`. */
export class ToolCallDelta {
  readonly type = "ToolCallDelta" as const;
  /** The part's position in the reply. */
  readonly index: number;
  /** The call id, when this delta carries it. */
  readonly id: string | null;
  /** The tool name, when this delta carries it. */
  readonly name: string | null;
  /** More arguments text. */
  readonly arguments: string;

  constructor(options: { index: number; id: string | null; name: string | null; arguments: string }) {
    this.index = options.index;
    this.id = options.id;
    this.name = options.name;
    // Read as a property: a binding named `arguments` is a SyntaxError in strict code.
    this.arguments = options.arguments;
  }
}

/** The final part at `index`; it replaces whatever its deltas built. */
export class PartDone {
  readonly type = "PartDone" as const;
  /** The part's position in the reply. */
  readonly index: number;
  /** The complete part. */
  readonly part: Part;

  constructor(index: number, part: Part) {
    this.index = index;
    this.part = part;
  }
}

/** Token counts so far; the last report wins. */
export class UsageReport {
  readonly type = "UsageReport" as const;
  /** The counts. */
  readonly usage: Usage;

  constructor(usage: Usage) {
    this.usage = usage;
  }
}

/** The reply ended; a stream's last event. */
export class Finish {
  readonly type = "Finish" as const;
  /** The finish reason, e.g. `stop` or `tool_calls`. */
  readonly reason: string | null;
  /** The provider's id for the reply. */
  readonly responseId: string | null;
  /** The adapter's protocol id, for `Message.protocol`. */
  readonly protocol: string | null;
  /** Message-level opaque data, for `Message.providerPayload`. */
  readonly providerPayload: JsonObject | null;

  constructor(options: {
    reason: string | null;
    responseId?: string | null;
    protocol?: string | null;
    providerPayload?: JsonObject | null;
  }) {
    this.reason = options.reason;
    this.responseId = options.responseId ?? null;
    this.protocol = options.protocol ?? null;
    this.providerPayload = options.providerPayload ?? null;
  }
}

/** One stream event. */
export type Event = TextDelta | ReasoningDelta | ToolCallDelta | PartDone | UsageReport | Finish;

/** Return `text` with each raw U+0000 to U+001F inside a JSON string escaped as `\u00XX`. */
function escapeControls(text: string): string {
  let out = "";
  let start = 0;
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (!inString) {
      if (code === 0x22) inString = true;
    } else if (code === 0x5c) {
      // The escaped character can't end the string.
      i++;
    } else if (code === 0x22) {
      inString = false;
    } else if (code < 0x20) {
      out += text.slice(start, i) + "\\u" + code.toString(16).padStart(4, "0");
      start = i + 1;
    }
  }
  return out + text.slice(start);
}

/**
 * Parse a tool call's complete arguments text.
 *
 * An empty text is an empty object. Otherwise the text must be a JSON object that
 * `parseJson` reads in plain mode (ints as `bigint`; no `NaN` or `Infinity`, overflowing
 * float, duplicate key or unpaired surrogate), nested at most `current().maxValueNesting`
 * levels. Raw control characters such as tabs and newlines are accepted inside strings.
 *
 * @param text The arguments text as the model sent it.
 * @returns `[object, null]` on success, `[null, text]` on any failure, so the caller can
 *   apply its own checks to the raw text.
 */
export function parseArguments(text: string): [JsonObject | null, string | null] {
  if (text === "") return [{}, null];
  try {
    // Models send raw tabs and newlines inside string values, which the provider SDKs
    // accept; a strict parse would reject working calls.
    const value = parseJson(escapeControls(text), "plain");
    if (typeof value !== "object" || value === null || Array.isArray(value)) return [null, text];
    checkJsonValue(value, "");
    return [value, null];
  } catch (error) {
    if (error instanceof JSONDecodeError || error instanceof MessageFormatError) return [null, text];
    throw error;
  }
}

/** The deltas seen so far at one index. */
interface Assembly {
  kind: "TextDelta" | "ReasoningDelta" | "ToolCallDelta";
  chunks: string[];
  id: string | null;
  name: string | null;
}

/** Return the type name of a value that is not an event: its class name, or `typeof` for a non-object. */
function typeName(value: unknown): string {
  if (value === null) return "null";
  if (typeof value !== "object") return typeof value;
  const name: unknown = (value as { constructor?: { name?: unknown } }).constructor?.name;
  return typeof name === "string" && name !== "" ? name : "object";
}

/**
 * Build the part the deltas at `index` describe.
 *
 * @throws {ProviderError} A tool call has no id or no name (kind `protocol`).
 * @throws {MessageFormatError} The joined text is not valid, e.g. it has an unpaired surrogate.
 */
function assemblyPart(assembly: Assembly, index: number): Part {
  const text = assembly.chunks.join("");
  if (assembly.kind === "TextDelta") return new TextPart(text);
  if (assembly.kind === "ReasoningDelta") return new ReasoningPart(text);
  if (assembly.id === null) throw new ProviderError("protocol", `tool call ${index} has no id`);
  if (assembly.name === null) throw new ProviderError("protocol", `tool call ${index} has no name`);
  const [args, raw] = parseArguments(text);
  return new ToolCallPart({ id: assembly.id, name: assembly.name, arguments: args, rawArguments: raw });
}

/**
 * Assembles a reply from its stream events, added in stream order.
 *
 * Parts are ordered by index. A `PartDone` part wins; otherwise the part is built from its
 * deltas, with a tool call's id and name taken from the first non-`null` value and its
 * arguments parsed by `parseArguments`. The stream must end with exactly one `Finish`.
 */
export class ReplyAssembler {
  #added = false;
  #done = new Map<number, Part>();
  #assemblies = new Map<number, Assembly>();
  #usage: Usage | null = null;
  #finish: Finish | null = null;

  /**
   * Add the next event.
   *
   * @throws {TypeError} `event` is not a stream event.
   * @throws {Error} An event follows the `Finish`, or one index has deltas of different kinds.
   */
  add(event: Event): void {
    const value: unknown = event;
    if (
      !(
        value instanceof TextDelta ||
        value instanceof ReasoningDelta ||
        value instanceof ToolCallDelta ||
        value instanceof PartDone ||
        value instanceof UsageReport ||
        value instanceof Finish
      )
    ) {
      throw new TypeError(`not a stream event: ${typeName(value)}`);
    }
    if (this.#finish !== null) throw new Error("event after finish");
    this.#added = true;
    if (event instanceof PartDone) {
      this.#done.set(event.index, event.part);
    } else if (event instanceof UsageReport) {
      this.#usage = event.usage;
    } else if (event instanceof Finish) {
      this.#finish = event;
    } else {
      let assembly = this.#assemblies.get(event.index);
      if (assembly === undefined) {
        assembly = { kind: event.type, chunks: [], id: null, name: null };
        this.#assemblies.set(event.index, assembly);
      } else if (assembly.kind !== event.type) {
        throw new Error(`index ${event.index} mixes ${assembly.kind} and ${event.type}`);
      }
      if (event instanceof ToolCallDelta) {
        assembly.chunks.push(event.arguments);
        assembly.id ??= event.id;
        assembly.name ??= event.name;
      } else {
        assembly.chunks.push(event.text);
      }
    }
  }

  /**
   * Return the reply. The message's protocol and provider payload, the finish reason and the
   * response id come from the `Finish`; usage comes from the last `UsageReport`.
   *
   * @throws {ProviderError} No `Finish` arrived (kind `incomplete`, retryable), or a tool call
   *   built from deltas has no id or no name (kind `protocol`).
   * @throws {MessageFormatError} A part built from deltas holds text that is not valid (for
   *   example an unpaired surrogate), so it could never be sent back.
   */
  response(): Response {
    const finish = this.#finish;
    if (finish === null) {
      throw new ProviderError("incomplete", "the stream ended before its finish event", { retryable: true });
    }
    const parts = this.#indices().map((index) => this.#done.get(index) ?? assemblyPart(this.#assemblies.get(index)!, index));
    const message = new Message({
      role: "assistant",
      parts,
      protocol: finish.protocol,
      providerPayload: finish.providerPayload,
    });
    return new Response({ message, usage: this.#usage, finishReason: finish.reason, responseId: finish.responseId });
  }

  /**
   * Return an assistant message of the parts that arrived so far, or `null` before any event:
   * done parts, text and reasoning built from their deltas, and tool calls that have both an
   * id and a name. A part that can't be built, such as text with an unpaired surrogate, is
   * left out. Never throws.
   */
  partial(): Message | null {
    if (!this.#added) return null;
    const parts: Part[] = [];
    for (const index of this.#indices()) {
      const done = this.#done.get(index);
      if (done !== undefined) {
        if (isPart(done)) parts.push(done);
        continue;
      }
      const assembly = this.#assemblies.get(index)!;
      if (assembly.kind === "ToolCallDelta" && (assembly.id === null || assembly.name === null)) continue;
      try {
        parts.push(assemblyPart(assembly, index));
      } catch (error) {
        if (!(error instanceof MessageFormatError)) throw error;
      }
    }
    return new Message({ role: "assistant", parts });
  }

  /** Return every index with a part or deltas, in numeric order. */
  #indices(): number[] {
    return Array.from(new Set([...this.#done.keys(), ...this.#assemblies.keys()])).sort((a, b) => a - b);
  }
}

/**
 * Assemble a reply from its stream events with a `ReplyAssembler`.
 *
 * When adding an event throws, `for await` closes the source, so a generator or
 * `EventStream` releases its connection.
 *
 * @param events The events, in stream order, ending with one `Finish`.
 * @returns The reply, with an assistant message.
 * @throws As `ReplyAssembler.add` and `ReplyAssembler.response` do, or as the source does.
 */
export async function collect(events: AsyncIterable<Event> | Iterable<Event>): Promise<Response> {
  const assembler = new ReplyAssembler();
  for await (const event of events) assembler.add(event);
  return assembler.response();
}
