/**
 * Host tools given as a name, a description, a JSON Schema and a call.
 *
 * `FunctionTool` describes a tool by its argument JSON Schema and an `execute` function, and may
 * carry Standard Schema v1 validators and an output JSON Schema. `tool()` builds one from a
 * validator (zod, valibot, ArkType, ...) or a raw JSON Schema; yian imports no validator library.
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

/** The members of a Standard Schema v1 validator yian reads (standardschema.dev). */
export interface StandardSchemaV1<Input = unknown, Output = Input> {
  readonly "~standard": {
    readonly version: 1;
    readonly vendor: string;
    readonly validate: (value: unknown) => StandardResult<Output> | Promise<StandardResult<Output>>;
    readonly types?: { readonly input: Input; readonly output: Output } | undefined;
  };
}

/** A validation result: the output, or the issues. */
export type StandardResult<Output> =
  | { readonly value: Output; readonly issues?: undefined }
  | { readonly issues: ReadonlyArray<StandardIssue> };

/** One issue: a message and the path to the value it is about. */
export interface StandardIssue {
  readonly message: string;
  readonly path?: ReadonlyArray<PropertyKey | { readonly key: PropertyKey }> | undefined;
}

// any (in both types below): the validator's other type parameter is its own; only the inferred one
// is used here.
/** The arguments `execute` gets: the validator's output type, else a plain record. */
type ArgsOf<P> = P extends StandardSchemaV1<any, infer O> ? O : Record<string, unknown>;
/** What `execute` returns: the output validator's input type, since it parses the result, else `unknown`. */
type ResultOf<O> = O extends StandardSchemaV1<infer I, any> ? I : unknown;

/**
 * A host tool given by its name, description, argument JSON Schema and call.
 *
 * A tool set checks each call's arguments against `parametersJsonSchema` before calling
 * `execute`: unknown keys (unless `additionalProperties` is `true` or a schema), missing required
 * keys, and values whose JSON Schema yian can translate to a type. A property yian can't
 * translate is passed unchecked.
 *
 * @throws {TypeError} `name` or `description` is not a string, `parametersJsonSchema` or
 *   `outputJsonSchema` is not a plain object or not JSON, `parametersValidator` or
 *   `outputValidator` is not a Standard Schema v1 validator, `execute` is not a function,
 *   `isConcurrencySafe` is neither a function nor `null`, or `resumable` is not a boolean.
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
   * Receives the call's argument and its `ToolContext`, and returns the result or a Promise of it.
   * The argument is a deep copy of the checked arguments, so changing it doesn't change the
   * caller's values; for a tool with a `parametersValidator` it is the validator's output, the
   * same object for every attempt of one checked call.
   */
  readonly execute: (args: Record<string, unknown>, ctx: ToolContext<any>) => unknown;
  /**
   * Says whether a call may run beside other calls; `null` when none was given. It receives what
   * `execute` gets: a deep copy of the checked arguments, or the validator's output, the same
   * object every attempt gets. It must be synchronous: a Promise result counts as unsafe.
   */
  readonly isConcurrencySafe: ((args: Record<string, unknown>) => boolean) | null;
  /** The Standard Schema v1 validator of the arguments, or `null` when none was given. */
  readonly parametersValidator: StandardSchemaV1 | null;
  /** The result's JSON Schema, a deep-frozen copy of the one given, or `null` when none was given. */
  readonly outputJsonSchema: JsonObject | null;
  /** The Standard Schema v1 validator of the result, or `null` when none was given. */
  readonly outputValidator: StandardSchemaV1 | null;
  /**
   * Whether a resumed AGENT may send a call that a crash or a stop cut off again, with the same
   * `ctx.callId`. Set it only for a tool that treats `callId` as an idempotency key.
   */
  readonly resumable: boolean;

  /** Check the fields and build a frozen tool. */
  constructor(options: {
    name: string;
    description: string;
    parametersJsonSchema: JsonObject;
    execute: (args: Record<string, unknown>, ctx: ToolContext<any>) => unknown;
    isConcurrencySafe?: ((args: Record<string, unknown>) => boolean) | null;
    parametersValidator?: StandardSchemaV1 | null;
    outputJsonSchema?: JsonObject | null;
    outputValidator?: StandardSchemaV1 | null;
    resumable?: boolean;
  }) {
    const { name, description, parametersJsonSchema, execute, isConcurrencySafe } = options;
    const { parametersValidator, outputJsonSchema, outputValidator, resumable } = options;
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
    // The model sees the schema, so it must be JSON: no Infinity, Date or runaway nesting.
    const schema = frozenJsonCopy(prefix, parametersJsonSchema, "parametersJsonSchema");
    let output: JsonObject | null = null;
    if (outputJsonSchema !== undefined && outputJsonSchema !== null) {
      if (!isRecord(outputJsonSchema)) {
        throw new TypeError(`${prefix} outputJsonSchema must be an object, got ${jsTypeName(outputJsonSchema)}`);
      }
      output = frozenJsonCopy(prefix, outputJsonSchema, "outputJsonSchema");
    }
    for (const [option, validator] of [
      ["parametersValidator", parametersValidator],
      ["outputValidator", outputValidator],
    ] as const) {
      if (validator !== undefined && validator !== null && !isStandardSchema(validator)) {
        throw new TypeError(`${prefix} ${option} must be a Standard Schema v1 validator`);
      }
    }
    if (typeof execute !== "function") throw new TypeError(`${prefix} execute must be a function`);
    if (
      isConcurrencySafe !== undefined &&
      isConcurrencySafe !== null &&
      typeof isConcurrencySafe !== "function"
    ) {
      throw new TypeError(`${prefix} isConcurrencySafe must be a function`);
    }
    if (resumable !== undefined && typeof resumable !== "boolean") {
      throw new TypeError(`${prefix} resumable must be a boolean, got ${jsTypeName(resumable)}`);
    }
    this.name = name;
    this.description = description;
    this.parametersJsonSchema = schema;
    this.execute = execute;
    this.isConcurrencySafe = isConcurrencySafe ?? null;
    this.parametersValidator = parametersValidator ?? null;
    this.outputJsonSchema = output;
    this.outputValidator = outputValidator ?? null;
    this.resumable = resumable ?? false;
    Object.freeze(this);
  }
}

Object.defineProperty(FunctionTool.prototype, FUNCTION_TOOL, { value: true });

/**
 * @internal Return a `FunctionTool` like `original` in every field but its call, which is
 * `execute`: for a host that wraps a tool's calls (to count them, say).
 */
export function withExecute(original: FunctionTool, execute: FunctionTool["execute"]): FunctionTool {
  return new FunctionTool({
    name: original.name,
    description: original.description,
    parametersJsonSchema: original.parametersJsonSchema,
    isConcurrencySafe: original.isConcurrencySafe,
    parametersValidator: original.parametersValidator,
    outputJsonSchema: original.outputJsonSchema,
    outputValidator: original.outputValidator,
    resumable: original.resumable,
    execute,
  });
}

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

/**
 * Build a `FunctionTool` from a Standard Schema v1 validator or a raw JSON Schema.
 *
 * `parameters` and `output` each take a raw JSON Schema object, or a validator. A validator's JSON
 * Schema is `parametersJsonSchema` (`outputJsonSchema`) when given, else the one its Standard JSON
 * Schema interface (`~standard.jsonSchema.input`) gives for draft 2020-12. The tool keeps the
 * validators as `parametersValidator` and `outputValidator`.
 *
 * @throws {TypeError} `parameters` (a given `output`) is neither a plain object nor a validator, a
 *   validator has no Standard JSON Schema and no JSON Schema was passed beside it, its converter
 *   throws (the error is the `cause`), `parametersJsonSchema` (`outputJsonSchema`) is given beside
 *   a raw JSON Schema, or `outputJsonSchema` is given without `output`; and as `FunctionTool` throws.
 * @throws {RangeError} as `FunctionTool` throws, for example when the arguments' schema is not of
 *   type `"object"`.
 *
 * @example
 * const place = tool({
 *   name: "place",
 *   description: "Place an order.",
 *   parameters: z.object({ sym: z.string(), qty: z.number().int().min(1) }),
 *   output: z.object({ filled: z.number().int() }),
 *   execute: (args, ctx) => ctx.context.submit(args.sym, args.qty),
 * });
 */
export function tool<
  P extends StandardSchemaV1 | JsonObject,
  O extends StandardSchemaV1 | JsonObject | undefined = undefined,
>(options: {
  name: string;
  description: string;
  parameters: P;
  parametersJsonSchema?: JsonObject;
  output?: O;
  outputJsonSchema?: JsonObject;
  execute: (args: ArgsOf<P>, ctx: ToolContext<any>) => ResultOf<O> | Promise<ResultOf<O>>;
  isConcurrencySafe?: ((args: ArgsOf<P>) => boolean) | null;
  resumable?: boolean;
}): FunctionTool {
  const { name, description, execute, isConcurrencySafe, resumable } = options;
  // `String`: the prefix must not throw on a non-string name; `FunctionTool` reports a bad name
  // when nothing fails first.
  const prefix = `tool ${reprText(String(name))}`;
  const [parametersJsonSchema, parametersValidator] = schemaOf(
    prefix,
    "parameters",
    options.parameters,
    options.parametersJsonSchema,
  );
  const [outputJsonSchema, outputValidator] = schemaOf(prefix, "output", options.output, options.outputJsonSchema);
  return new FunctionTool({
    name,
    description,
    parametersJsonSchema: parametersJsonSchema as JsonObject,
    // The typed signatures narrow the arguments to what the validator describes.
    execute: execute as FunctionTool["execute"],
    isConcurrencySafe: isConcurrencySafe as FunctionTool["isConcurrencySafe"] | undefined,
    parametersValidator,
    // `undefined` when no output was given, which `FunctionTool` stores as `null`.
    outputJsonSchema: outputJsonSchema as JsonObject | undefined,
    outputValidator,
    resumable,
  });
}

/**
 * Return the JSON Schema and the validator `value` gives for `role`: a raw JSON Schema is its own
 * schema with no validator; a validator's schema is `given`, else its Standard JSON Schema; an
 * absent output is neither.
 */
function schemaOf(
  prefix: string,
  role: "parameters" | "output",
  value: unknown,
  given: JsonObject | undefined,
): [unknown, StandardSchemaV1 | null] {
  const option = `${role}JsonSchema`;
  const isValidator =
    ((typeof value === "object" && value !== null) || typeof value === "function") && "~standard" in value;
  if (!isValidator) {
    if (role === "output" && value === undefined) {
      if (given !== undefined) throw new TypeError(`${prefix} ${option} needs an output validator`);
      return [undefined, null];
    }
    if (!isRecord(value)) {
      throw new TypeError(`${prefix} ${role} must be a JSON Schema object or a Standard Schema v1 validator`);
    }
    if (given !== undefined) throw new TypeError(`${prefix} ${option} is only for a validator`);
    return [value, null];
  }
  if (!isStandardSchema(value)) throw new TypeError(`${prefix} ${role} must be a Standard Schema v1 validator`);
  const validator = value;
  if (given !== undefined) return [given, validator];
  const { vendor, jsonSchema } = validator["~standard"] as { vendor: unknown; jsonSchema?: unknown };
  const input =
    typeof jsonSchema === "object" && jsonSchema !== null ? (jsonSchema as { input?: unknown }).input : undefined;
  if (typeof input !== "function") {
    throw new TypeError(
      `${prefix} ${role}: the ${String(vendor)} validator has no Standard JSON Schema (~standard.jsonSchema); ` +
        `pass ${option} too`,
    );
  }
  try {
    return [input.call(jsonSchema, { target: "draft-2020-12" }), validator];
  } catch (error) {
    const what = role === "parameters" ? "an argument" : "an output";
    const message = error instanceof Error ? error.message : String(error);
    throw new TypeError(`${prefix} has ${what} type with no JSON Schema: ${message}`, { cause: error });
  }
}

/** Return whether `value` is a Standard Schema v1 validator: an object or function with a v1 `~standard`. */
function isStandardSchema(value: unknown): value is StandardSchemaV1 {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") return false;
  const standard: unknown = (value as { "~standard"?: unknown })["~standard"];
  if (typeof standard !== "object" || standard === null) return false;
  const props = standard as { version?: unknown; validate?: unknown };
  return props.version === 1 && typeof props.validate === "function";
}

/** Return a deep-frozen JSON copy of the `label` option; a non-JSON value is a TypeError naming it. */
function frozenJsonCopy(prefix: string, schema: JsonObject, label: string): JsonObject {
  try {
    return deepFreeze(checkJsonValue(schema, label) as JsonObject);
  } catch (error) {
    if (error instanceof MessageFormatError) throw new TypeError(`${prefix} ${error.message}`);
    throw error;
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
