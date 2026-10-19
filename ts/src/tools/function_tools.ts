/**
 * Host tools given as a name, a description, a JSON Schema and a call.
 *
 * Knows about: `tools/context`, `llm/messages` (the JSON check), typesys and diagnostics.
 */

import { reprText } from "../diagnostics.ts";
import { checkJsonValue, MessageFormatError } from "../llm/messages.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isRecord, jsTypeName } from "../typesys/values.ts";
import type { ToolContext } from "./context.ts";

/**
 * The brand every `FunctionTool` carries. The registry checks it rather than `instanceof`, so a
 * tool built against another copy of yian (an npm install beside a workspace one) is accepted.
 */
const FUNCTION_TOOL = Symbol.for("yian.function_tool");

/**
 * A host tool given by its name, description, argument JSON Schema and call.
 *
 * A tool set checks each call's arguments against `parametersJsonSchema` before calling
 * `execute`: unknown keys (unless `additionalProperties` is `true` or a schema), missing required
 * keys, and values whose JSON Schema yian can translate to a type. A property yian can't
 * translate is passed unchecked.
 *
 * @throws {TypeError} `name` or `description` is not a string, `parametersJsonSchema` is not a
 *   plain object or not JSON, `execute` is not a function, or `isConcurrencySafe` is neither a
 *   function nor `null`.
 * @throws {RangeError} `name` is empty, or `parametersJsonSchema`'s `type` is not `"object"`.
 *
 * @example
 * const echo = new FunctionTool({
 *   name: "echo",
 *   description: "Echo the text.",
 *   parametersJsonSchema: {
 *     type: "object",
 *     properties: { text: { type: "string" } },
 *     required: ["text"],
 *   },
 *   execute: (args) => args.text,
 * });
 */
export class FunctionTool {
  /** The tool name. A tool set key or registered id must equal it. */
  readonly name: string;
  /** The description the model sees. */
  readonly description: string;
  /** The arguments' JSON Schema, a deep-frozen copy of the one given; its `type` is `"object"`. */
  readonly parametersJsonSchema: JsonObject;
  // `any`, not `unknown`: a tool typed for its own context (`ToolContext<Broker>`) must be accepted.
  /**
   * Receives a deep copy of the checked arguments and the call's `ToolContext`, and returns the
   * result or a Promise of it. Changing that copy doesn't change the caller's values.
   */
  readonly execute: (args: Record<string, unknown>, ctx: ToolContext<any>) => unknown;
  /**
   * Says whether a call with these arguments may run beside other calls; `null` when none was
   * given. It must be synchronous: a Promise result counts as unsafe.
   */
  readonly isConcurrencySafe: ((args: Record<string, unknown>) => boolean) | null;

  /** Check the fields and build a frozen tool. */
  constructor(options: {
    name: string;
    description: string;
    parametersJsonSchema: JsonObject;
    execute: (args: Record<string, unknown>, ctx: ToolContext<any>) => unknown;
    isConcurrencySafe?: ((args: Record<string, unknown>) => boolean) | null;
  }) {
    const { name, description, parametersJsonSchema, execute, isConcurrencySafe } = options;
    if (typeof name !== "string") {
      throw new TypeError(`FunctionTool name must be a string, got ${jsTypeName(name)}`);
    }
    if (name === "") throw new RangeError("FunctionTool name must not be empty");
    const prefix = `FunctionTool ${reprText(name)}`;
    if (typeof description !== "string") {
      throw new TypeError(`${prefix} description must be a string, got ${jsTypeName(description)}`);
    }
    if (!isRecord(parametersJsonSchema)) {
      throw new TypeError(
        `${prefix} parametersJsonSchema must be an object, got ${jsTypeName(parametersJsonSchema)}`,
      );
    }
    if (!Object.hasOwn(parametersJsonSchema, "type") || parametersJsonSchema.type !== "object") {
      throw new RangeError(`${prefix} parametersJsonSchema must have type 'object'`);
    }
    let schema: JsonObject;
    try {
      // The model sees the schema, so it must be JSON: no Infinity, Date or runaway nesting.
      schema = checkJsonValue(parametersJsonSchema, "parametersJsonSchema") as JsonObject;
    } catch (error) {
      if (error instanceof MessageFormatError) throw new TypeError(`${prefix} ${error.message}`);
      throw error;
    }
    if (typeof execute !== "function") throw new TypeError(`${prefix} execute must be a function`);
    if (
      isConcurrencySafe !== undefined &&
      isConcurrencySafe !== null &&
      typeof isConcurrencySafe !== "function"
    ) {
      throw new TypeError(`${prefix} isConcurrencySafe must be a function`);
    }
    this.name = name;
    this.description = description;
    this.parametersJsonSchema = deepFreeze(schema);
    this.execute = execute;
    this.isConcurrencySafe = isConcurrencySafe ?? null;
    Object.freeze(this);
  }
}

Object.defineProperty(FunctionTool.prototype, FUNCTION_TOOL, { value: true });

/**
 * Return whether `value` is a `FunctionTool` from any copy of yian: it carries the brand and a
 * tool's shape. Never throws, so a hostile object is not a tool.
 */
export function isFunctionTool(value: unknown): value is FunctionTool {
  if (typeof value !== "object" || value === null) return false;
  try {
    const tool = value as Partial<Record<keyof FunctionTool, unknown>>;
    return (
      (value as { [FUNCTION_TOOL]?: unknown })[FUNCTION_TOOL] === true &&
      typeof tool.name === "string" &&
      typeof tool.execute === "function" &&
      isRecord(tool.parametersJsonSchema)
    );
  } catch {
    return false;
  }
}

/** Freeze `value` and every object and array in it, and return it; `value` is plain JSON. */
function deepFreeze<T>(value: T): T {
  // An explicit stack, as `checkJsonValue` walks, so any nesting it accepted is frozen.
  const stack: unknown[] = [value];
  while (stack.length > 0) {
    const item = stack.pop();
    if (typeof item !== "object" || item === null) continue;
    Object.freeze(item);
    for (const child of Object.values(item)) stack.push(child);
  }
  return value;
}
