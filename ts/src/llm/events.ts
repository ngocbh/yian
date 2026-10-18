/**
 * Stream events a model emits, and how they add up to a `Response`.
 *
 * An adapter turns a provider's stream into these events: deltas per part index, an
 * authoritative `PartDone` per part when the provider gives one, usage, and one `Finish`
 * last. Each event has a `type` discriminant equal to its class name.
 */

import { JSONDecodeError, parseJson, type JsonObject } from "../typesys/json.ts";
import { checkJsonValue, MessageFormatError, type Part, type Usage } from "./messages.ts";

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
