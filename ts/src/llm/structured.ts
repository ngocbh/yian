/**
 * Structured output: choose how a call is sent per route, model and schema, and read the JSON
 * object from the reply.
 *
 * `planStructured` picks a strategy (a forced or unforced tool, the route's JSON-schema output
 * format, or a prompt for a model with no native structured output) from the first matching row
 * of a fixed table, and the request fields it needs. The resulting `StructuredPlan` `apply`s
 * those fields to a request, gives the prompt strategy's `instruction` and the retry
 * `correction` texts, and `parse`s the reply's JSON object; `parse` throws
 * `StructuredOutputError`, or its subclass `MissingToolCallError` for a tool strategy. The caller
 * validates replies against the schema it passed. Ported from Python yian's
 * `with_structured_output` overrides in `llm_clients/*_client.py`, the native-structured gate and
 * code-fence strip in `nodes/agent/structured.py`, and `missing_tool_call` in
 * `llm_clients/base_client.py`.
 */

import type { JsonObject } from "../typesys/json.ts";
import { encodeBody } from "./body.ts";
import { getCapabilities, supportsNativeStructured } from "./capabilities.ts";
import { type ChatModel, parseArguments } from "./events.ts";
import { RouteModel } from "./executor.ts";
import {
  anthropicOutputSchema,
  constAsEnum,
  functionTool,
  hasOpenSchema,
  moveAzureKeywords,
  strictJsonSchema,
} from "./json_schema.ts";
import {
  copyJson,
  isPlainObject,
  NamedTool,
  OutputFormat,
  RefusalPart,
  Request,
  type Response,
  type ToolChoice,
  ToolCallPart,
  type ToolSpec,
} from "./messages.ts";
import { AnthropicMessages } from "./protocols/anthropic_messages.ts";
import { OpenAIResponses } from "./protocols/openai_responses.ts";

/**
 * A structured reply that holds no valid answer: no JSON object, a refusal, or
 * (`MissingToolCallError`) no call of the schema's tool.
 */
export class StructuredOutputError extends Error {
  override name = "StructuredOutputError";
}

/** A structured reply that did not call the schema's tool. */
export class MissingToolCallError extends StructuredOutputError {
  override name = "MissingToolCallError";
}

/**
 * How a structured call is sent: a tool the model must call, a tool it may call, the route's
 * JSON-schema output format, or an instruction in the prompt.
 */
export type Strategy = "forced_tool" | "unforced_tool" | "json_schema" | "prompt";

/** What the strategy choice reads off a model. */
export interface StructuredTarget {
  /** The provider name, e.g. `azure`. */
  readonly provider: string;
  /** The model name (the deployment name on Azure). */
  readonly model: string;
  /** The wire protocol's name, e.g. `anthropic_messages`; `null` for a model without one. */
  readonly protocol: string | null;
  /** The provider block, read for Anthropic's `thinking`; `null` when the protocol exposes none. */
  readonly block: JsonObject | null;
  /** Whether the route is the ChatGPT subscription. */
  readonly subscription: boolean;
}

/** Model id prefixes the Anthropic Messages API refuses a forced tool choice for. */
export const FORCED_TOOL_REFUSING_PREFIXES: readonly string[] = Object.freeze([
  "claude-fable-5-1",
  "claude-opus-5-5",
  "claude-sonnet-5-5",
  "claude-mythos-5-1",
]);

/** Return whether the Anthropic model `model` accepts a forced tool choice. */
export function forcesTool(model: string): boolean {
  return !FORCED_TOOL_REFUSING_PREFIXES.some((prefix) => model.startsWith(prefix));
}

/**
 * Return whether the Azure deployment `model` is a legacy one that gets its schema as a forced
 * tool rather than a JSON-schema output format (LangChain's check for `gpt-3*`, `gpt-4-*` and
 * `gpt-4`).
 */
export function sendsJsonSchemaAsTool(model: string): boolean {
  return model.startsWith("gpt-3") || model.startsWith("gpt-4-") || model === "gpt-4";
}

/**
 * Return the target `planStructured` reads off `model`.
 *
 * A `RouteModel` gives its protocol's name, an `AnthropicMessages` block and an `OpenAIResponses`
 * subscription flag; any other `ChatModel` has no wire protocol.
 */
export function structuredTarget(model: ChatModel): StructuredTarget {
  if (!(model instanceof RouteModel)) {
    return { provider: model.provider, model: model.model, protocol: null, block: null, subscription: false };
  }
  const protocol = model.protocol;
  return {
    provider: model.provider,
    model: model.model,
    protocol: protocol.name,
    block: protocol instanceof AnthropicMessages ? protocol.block : null,
    subscription: protocol instanceof OpenAIResponses && protocol.subscription,
  };
}

/** The request fields a plan sets; omitted ones are empty. */
interface PlanFields {
  tools?: readonly ToolSpec[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean | null;
  outputFormat?: OutputFormat | null;
}

/** How one structured call is sent: its strategy and the request fields it sets. */
export class StructuredPlan {
  /** How the call is sent. */
  readonly strategy: Strategy;
  /** The schema's name, its root `title`: the tool name and the output format name. */
  readonly name: string;
  /** The JSON schema the reply must match, a copy. */
  readonly schema: JsonObject;
  /** The tools the request carries: the schema's tool, or none. */
  readonly tools: readonly ToolSpec[];
  /** The request's tool choice. */
  readonly toolChoice: ToolChoice;
  /** The request's `parallelToolCalls`; `null` leaves it unsent. */
  readonly parallelToolCalls: boolean | null;
  /** The request's output format, or `null`. */
  readonly outputFormat: OutputFormat | null;

  constructor(options: { strategy: Strategy; name: string; schema: JsonObject } & PlanFields) {
    this.strategy = options.strategy;
    this.name = options.name;
    this.schema = copyJson(options.schema);
    this.tools = Object.freeze([...(options.tools ?? [])]);
    this.toolChoice = options.toolChoice ?? null;
    this.parallelToolCalls = options.parallelToolCalls ?? null;
    this.outputFormat = options.outputFormat ?? null;
  }

  /**
   * Return `request` with this plan's tools, tool choice, parallel calls flag and output format;
   * its messages are kept. The prompt strategy returns `request` itself: the caller appends
   * `instruction()` as a user message.
   *
   * @throws {TypeError} When `request` already has tools or an output format, whatever the strategy.
   */
  apply(request: Request): Request {
    if (request.tools.length > 0 || request.outputFormat !== null) {
      throw new TypeError("a structured request carries only the schema's tool");
    }
    if (this.strategy === "prompt") return request;
    return new Request({
      messages: request.messages,
      tools: this.tools,
      toolChoice: this.toolChoice,
      parallelToolCalls: this.parallelToolCalls,
      outputFormat: this.outputFormat,
    });
  }

  /**
   * Return the user message that asks a prompt-strategy model for JSON matching the schema.
   *
   * @throws {Error} When the strategy is not `prompt`.
   */
  instruction(): string {
    if (this.strategy !== "prompt") throw new Error("only the prompt strategy has an instruction");
    const schema = encodeBody(this.schema);
    return `Respond with ONLY a JSON object matching this JSON schema (no prose, no code fences):\n${schema}`;
  }

  /** Return the user message that asks for a new reply after an invalid one, `detail` saying why. */
  correction(detail: string): string {
    const ask =
      this.strategy === "prompt"
        ? "Respond with ONLY a JSON object matching the schema."
        : "Respond with valid data matching the schema.";
    return `Your previous output was invalid: ${detail}. ${ask}`;
  }

  /**
   * Return the JSON object `response` answers with, unvalidated (ints as `bigint`).
   *
   * The tool strategies read the first call of the schema's tool whose arguments parsed. The
   * text strategies read the message's text, stripped of whitespace and backticks, or else the
   * text inside its first code fence up to the last one; either must be one JSON object, with no
   * repair.
   *
   * @throws {MissingToolCallError} A tool strategy's reply has no valid call of the schema's tool.
   * @throws {StructuredOutputError} A text strategy's reply holds no JSON object; the message
   *   names the model's refusal, if any, else the text.
   */
  parse(response: Response): JsonObject {
    const parts = response.message.parts;
    if (this.strategy === "forced_tool" || this.strategy === "unforced_tool") {
      for (const part of parts) {
        if (part instanceof ToolCallPart && part.name === this.name && part.arguments !== null) return part.arguments;
      }
      throw new MissingToolCallError(`the reply did not call the ${this.name} tool; call it with the answer`);
    }
    const text = response.message.text();
    const value = parseObject(strip(text)) ?? parseObject(fenced(text));
    if (value !== null) return value;
    const refusals = parts.filter((part) => part instanceof RefusalPart);
    if (refusals.length > 0) {
      throw new StructuredOutputError(`the model refused: ${refusals.map((part) => part.text).join("\n")}`);
    }
    throw new StructuredOutputError(`Invalid json output: ${text.trim()}`);
  }
}

const FENCE = "```";
/** LangChain's strip set: Python's `" \n\r\t`"`. */
const STRIPPED = " \n\r\t`";

/** Return `text` without the whitespace and backticks around it. */
function strip(text: string): string {
  // A loop, not a `[...]+$` regex, which backtracks quadratically over a long inner run.
  let start = 0;
  let end = text.length;
  while (start < end && STRIPPED.includes(text[start]!)) start++;
  while (end > start && STRIPPED.includes(text[end - 1]!)) end--;
  return text.slice(start, end);
}

/**
 * Return the text inside the first code fence of `text`, without its info string (e.g. `json`),
 * up to the last fence after it or the end, stripped like `strip`; `""` when `text` has no
 * fence. Taking the last fence keeps fences inside the JSON's strings.
 */
function fenced(text: string): string {
  const start = text.indexOf(FENCE);
  if (start === -1) return "";
  const body = text.slice(start + FENCE.length).replace(/^[A-Za-z0-9_+.-]*/, "");
  const end = body.lastIndexOf(FENCE);
  return strip(end === -1 ? body : body.slice(0, end));
}

/**
 * Return the JSON object `text` holds, or `null`; an empty text holds none (checked here because
 * `parseArguments` reads `""` as `{}`).
 */
function parseObject(text: string): JsonObject | null {
  return text === "" ? null : parseArguments(text)[0];
}

/**
 * Return how a structured call for `schema` is sent to `target`: the first matching row wins.
 *
 * 1. no native structured output (neither the subscription nor `supportsNativeStructured`):
 *    `prompt`;
 * 1a. no wire protocol: `json_schema` with the tool's parameters (the provider-neutral form);
 * 2. Anthropic with `thinking` on and an open schema: the tool, unforced (thinking forbids a
 *    forced tool, and the output format's grammar rejects an open schema);
 * 3. Anthropic with `thinking` on: `json_schema` with `anthropicOutputSchema`;
 * 4. Anthropic, a model that `forcesTool`: the tool, forced;
 * 5. Anthropic otherwise: the tool, unforced;
 * 6. Ollama: `json_schema` with the schema as given;
 * 7. Gemini: `json_schema` with `constAsEnum`;
 * 8. Azure, an open schema: the tool, forced (`strict: false` on Responses);
 * 9. Azure, a legacy deployment (`sendsJsonSchemaAsTool`): the tool, forced, keywords kept;
 * 10. Azure otherwise: strict `json_schema` with the rejected keywords moved into descriptions;
 * 11. any other route: the tool (`strict: false` for an open schema on Responses), forced when
 *    the route is the subscription or the model accepts a tool choice, else unforced.
 *
 * Azure's tool rows send `parallelToolCalls: false` on Chat and leave it unsent on Responses;
 * row 11 sends `false`.
 *
 * @param target The route and model, from `structuredTarget`.
 * @param schema The JSON schema; it is not changed.
 * @returns The plan.
 * @throws {TypeError} When the schema has no non-empty string `title`, its name.
 */
export function planStructured(target: StructuredTarget, schema: JsonObject): StructuredPlan {
  const name = schema["title"];
  if (typeof name !== "string" || name === "") {
    throw new TypeError("a structured output schema needs a title, its name");
  }
  const plan = (strategy: Strategy, fields: PlanFields = {}) =>
    new StructuredPlan({ ...fields, strategy, name, schema });
  const format = (formatSchema: JsonObject, strict: boolean | null = null) =>
    new OutputFormat({ name, schema: formatSchema, strict });
  const tools = (strict: boolean | null) => [functionTool(schema, strict)];

  if (!(target.subscription || supportsNativeStructured(target.provider, target.model))) return plan("prompt");
  if (target.protocol === null) {
    return plan("json_schema", { outputFormat: format(functionTool(schema, null).parameters) });
  }
  const open = hasOpenSchema(schema);
  const responses = target.protocol === "openai_responses";

  if (target.protocol === "anthropic_messages") {
    const thinking = target.block?.["thinking"];
    const thinkingOn = isPlainObject(thinking) && thinking["type"] !== "disabled";
    if (thinkingOn && open) return plan("unforced_tool", { tools: tools(null) });
    if (thinkingOn) return plan("json_schema", { outputFormat: format(anthropicOutputSchema(schema)) });
    if (forcesTool(target.model)) {
      return plan("forced_tool", { tools: tools(null), toolChoice: new NamedTool(name) });
    }
    return plan("unforced_tool", { tools: tools(null) });
  }
  if (target.protocol === "ollama_chat") return plan("json_schema", { outputFormat: format(schema) });
  if (target.protocol === "gemini") return plan("json_schema", { outputFormat: format(constAsEnum(schema)) });

  if (target.provider === "azure") {
    const parallelToolCalls = responses ? null : false;
    if (open) {
      const strict = responses ? false : null;
      return plan("forced_tool", { tools: tools(strict), toolChoice: new NamedTool(name), parallelToolCalls });
    }
    if (sendsJsonSchemaAsTool(target.model)) {
      return plan("forced_tool", { tools: tools(null), toolChoice: new NamedTool(name), parallelToolCalls });
    }
    const strictSchema = strictJsonSchema(schema);
    return plan("json_schema", { outputFormat: format(moveAzureKeywords(strictSchema) ?? strictSchema, true) });
  }

  const strict = responses && open ? false : null;
  if (target.subscription || getCapabilities(target.model).supportsToolChoice) {
    return plan("forced_tool", { tools: tools(strict), toolChoice: new NamedTool(name), parallelToolCalls: false });
  }
  return plan("unforced_tool", { tools: tools(strict), parallelToolCalls: false });
}
