/**
 * `FunctionTool` construction, `tool()`, the types `toolTypes` reads from a tool's JSON Schemas,
 * and a `FunctionTool` in a `ToolSet` and the registry, a port of
 * `tests/engine/test_function_tool.py` and of the JSON Schema and output rows of
 * `test_tool_types.py` that read a tool. The `function_tool` rows are ported as `tool()` rows, with
 * a Standard Schema validator or a JSON Schema in place of the function's signature; the two
 * invocation rows, `test_function_tool_result_validates_like_a_function_and_passes_the_context` and
 * `test_function_tool_without_context_gets_its_validated_arguments`, are in the checking-order
 * rows, where the validator plays pydantic's role. Not ported: the decorator form and the docstring
 * defaults (TS has no runtime signatures), the async refusal (TS awaits an async `execute`), the
 * unresolvable annotation row (TS has no runtime annotations), Python's lax `"2"` to 2 coercion
 * (yian's argument check refuses `"2"` for an integer first),
 * `test_replace_drops_the_function_schema` (no `dataclasses.replace`),
 * `test_run_entry_points_and_tool_set_annotate_function_tool_entries` (needs the run entry points),
 * `test_function_tool_without_a_signature_cannot_be_called_directly`, the second half of
 * `test_function_tool_without_context_gets_its_validated_arguments` (calling `on_invoke` directly
 * validates; calling `execute` directly runs no validator), and the LangChain binding rows. The
 * exports row, `test_tool_names_are_exported_from_yian_and_yian_tools`, is in
 * `tool_errors.test.ts`. With `YIAN_TEST_NODE` set to a Node binary, the built tools code on Node,
 * beside a second copy of its leaves.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import * as v from "valibot";
import { z } from "zod";

import { ToolContext } from "../../src/tools/context.ts";
import { isToolCheckError, ToolCheckError } from "../../src/tools/errors.ts";
import {
  FunctionTool,
  isFunctionTool,
  type StandardResult,
  type StandardSchemaV1,
  tool,
} from "../../src/tools/function_tools.ts";
import { registeredTools, registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
import { toolTypes } from "../../src/tools/types.ts";
import { type JsonObject, parseJson } from "../../src/typesys/json.ts";
import { canonicalTypeLabel, Type, TypeCheckError, typeEquals, ValueKind } from "../../src/typesys/values.ts";

const ECHO_SCHEMA: JsonObject = {
  type: "object",
  properties: { text: { type: "string" } },
  required: ["text"],
};

const FIELDS = {
  name: "echo",
  description: "Echo.",
  parametersJsonSchema: ECHO_SCHEMA,
  execute: () => null,
};

/** Build a `FunctionTool` from the echo fields with `changes` applied. */
function build(changes: Record<string, unknown>): FunctionTool {
  return new FunctionTool({ ...FIELDS, ...changes } as ConstructorParameters<typeof FunctionTool>[0]);
}

/** Return the canonical label of each input type, `null` for an unchecked one. */
function inputLabels(tool: FunctionTool): Record<string, string | null> | null {
  const { inputs } = toolTypes(tool);
  if (inputs === null) return null;
  return Object.fromEntries(
    [...inputs].map(([name, typ]) => [name, typ === null ? null : canonicalTypeLabel(typ)]),
  );
}

/** Return the error `promise` rejects with, after checking its class. */
async function rejection(promise: Promise<unknown>, kind: abstract new (...args: never[]) => Error): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    expect((error as Error).constructor).toBe(kind);
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** Parameters yian's check accepts where the validator refines, fills a default or transforms. */
const CHECKED_PARAMETERS = z.object({
  sym: z.string().refine((s) => s === s.toUpperCase(), "must be upper case"),
  qty: z.number().int().min(1).default(1),
  note: z
    .string()
    .transform((s) => s.trim())
    .optional(),
});

/** What `CHECKED_PARAMETERS` gives `execute`. */
type Checked = z.output<typeof CHECKED_PARAMETERS>;

/** A call `checkedPlace`'s `execute` saw: its arguments and context. */
type SeenCall = [Checked, ToolContext<any>];

/**
 * Build the `place` tool on `CHECKED_PARAMETERS` behind a hand-built async validator that counts
 * its calls in `count` and runs `during` while it validates. `execute` records each call in `seen`.
 */
function checkedPlace(
  options: { during?: () => void; isConcurrencySafe?: (args: Checked) => boolean } = {},
): { placed: FunctionTool; seen: SeenCall[]; count: { calls: number } } {
  const seen: SeenCall[] = [];
  const count = { calls: 0 };
  const zod = CHECKED_PARAMETERS["~standard"];
  const validator: StandardSchemaV1<unknown, Checked> = {
    "~standard": {
      version: 1,
      vendor: "hand",
      validate: async (value) => {
        count.calls++;
        options.during?.();
        await Promise.resolve();
        return (await zod.validate(value)) as StandardResult<Checked>;
      },
    },
  };
  const placed = tool({
    name: "place",
    description: "Place an order.",
    parameters: validator,
    parametersJsonSchema: zod.jsonSchema.input({ target: "draft-2020-12" }) as JsonObject,
    execute: (args, ctx) => {
      seen.push([args, ctx]);
      return `${args.sym}:${args.qty}`;
    },
    isConcurrencySafe: options.isConcurrencySafe,
  });
  return { placed, seen, count };
}

describe("FunctionTool construction", () => {
  const ROWS: [string, Record<string, unknown>, typeof TypeError | typeof RangeError, string][] = [
    ["an empty name", { name: "" }, RangeError, "FunctionTool name must not be empty"],
    ["a non-string name", { name: 3 }, TypeError, "FunctionTool name must be a string, got number"],
    [
      "a non-string description",
      { description: null },
      TypeError,
      "FunctionTool 'echo' description must be a string, got object",
    ],
    [
      "a non-object schema",
      { parametersJsonSchema: [1] },
      TypeError,
      "FunctionTool 'echo' parametersJsonSchema must be an object, got Array",
    ],
    [
      "a schema of another type",
      { parametersJsonSchema: { type: "string" } },
      RangeError,
      "FunctionTool 'echo' parametersJsonSchema must have type 'object'",
    ],
    [
      "a schema with no type",
      { parametersJsonSchema: { properties: {} } },
      RangeError,
      "FunctionTool 'echo' parametersJsonSchema must have type 'object'",
    ],
    [
      "a non-function execute",
      { execute: "not callable" },
      TypeError,
      "FunctionTool 'echo' execute must be a function",
    ],
    [
      "an infinite number in the schema",
      { parametersJsonSchema: { type: "object", properties: { x: { type: "number", maximum: Infinity } } } },
      TypeError,
      "FunctionTool 'echo' parametersJsonSchema.properties.x.maximum: float is not finite",
    ],
    [
      "a non-function isConcurrencySafe",
      { isConcurrencySafe: true },
      TypeError,
      "FunctionTool 'echo' isConcurrencySafe must be a function",
    ],
    [
      "a non-object output schema",
      { outputJsonSchema: [1] },
      TypeError,
      "FunctionTool 'echo' outputJsonSchema must be an object, got Array",
    ],
    [
      "an infinite number in the output schema",
      { outputJsonSchema: { maximum: Infinity } },
      TypeError,
      "FunctionTool 'echo' outputJsonSchema.maximum: float is not finite",
    ],
  ];

  for (const [what, changes, error, message] of ROWS) {
    test(`refuses ${what}`, () => {
      let thrown: unknown;
      try {
        build(changes);
      } catch (caught) {
        thrown = caught;
      }
      expect(thrown).toBeInstanceOf(error);
      expect((thrown as Error).message).toBe(message);
    });
  }

  for (const option of ["parametersValidator", "outputValidator"]) {
    const bad: [string, unknown][] = [
      ["an object with no ~standard", {}],
      ["a version 2 ~standard", { "~standard": { version: 2, validate() {} } }],
      ["a ~standard with no validate", { "~standard": { version: 1 } }],
      ["a string", "zod"],
    ];
    for (const [what, validator] of bad) {
      test(`refuses ${what} as ${option}`, () => {
        expect(() => build({ [option]: validator })).toThrow(
          new TypeError(`FunctionTool 'echo' ${option} must be a Standard Schema v1 validator`),
        );
      });
    }
  }

  test("accepts a callable validator, as ArkType's types are", () => {
    const validator = Object.assign(() => null, {
      "~standard": { version: 1 as const, vendor: "fn", validate: (value: unknown) => ({ value }) },
    });
    expect(build({ parametersValidator: validator }).parametersValidator).toBe(validator);
  });

  test("its validator and output fields are null by default, and it stays frozen", () => {
    const tool = build({});
    expect(tool.parametersValidator).toBeNull();
    expect(tool.outputJsonSchema).toBeNull();
    expect(tool.outputValidator).toBeNull();
    const nulls = build({ parametersValidator: null, outputJsonSchema: null, outputValidator: null });
    expect(nulls.outputJsonSchema).toBeNull();
    expect(Object.isFrozen(tool)).toBe(true);
  });

  test("stores a frozen copy of the output schema", () => {
    const schema = { type: "object", properties: { n: { type: "integer" } } };
    const tool = build({ outputJsonSchema: schema });
    schema.properties.n.type = "string";
    expect(tool.outputJsonSchema).toEqual({ type: "object", properties: { n: { type: "integer" } } });
    const properties = tool.outputJsonSchema!.properties as Record<string, JsonObject>;
    expect(() => {
      properties.n!.type = "string";
    }).toThrow(TypeError);
    expect(Object.isFrozen(tool.outputJsonSchema)).toBe(true);
  });

  test("refuses a Date in the schema", () => {
    const schema = { type: "object", properties: { x: { type: "string", default: new Date(0) } } };
    expect(() => build({ parametersJsonSchema: schema })).toThrow(TypeError);
    expect(() => build({ parametersJsonSchema: schema })).toThrow(
      /^FunctionTool 'echo' parametersJsonSchema\.properties\.x\.default: /,
    );
  });

  test("refuses a schema nested past the value nesting limit", () => {
    // Python leaves such an input unchecked; a TS tool's schema must be JSON the model can see.
    let deep: JsonObject = { type: "integer" };
    for (let i = 0; i < 150; i++) deep = { type: "array", items: deep };
    const schema = { type: "object", properties: { x: deep, y: { type: "string" } } };
    expect(() => build({ parametersJsonSchema: schema })).toThrow(TypeError);
    expect(() => build({ parametersJsonSchema: schema })).toThrow(/value nests deeper than 100 levels/);
  });

  test("keeps its fields", () => {
    const execute = (args: Record<string, unknown>) => args.text;
    const isConcurrencySafe = () => true;
    const tool = new FunctionTool({
      name: "echo",
      description: "Echo.",
      parametersJsonSchema: ECHO_SCHEMA,
      execute,
      isConcurrencySafe,
    });
    expect(tool.name).toBe("echo");
    expect(tool.description).toBe("Echo.");
    expect(tool.parametersJsonSchema).toEqual(ECHO_SCHEMA);
    expect(tool.execute).toBe(execute);
    expect(tool.isConcurrencySafe).toBe(isConcurrencySafe);
    expect(build({}).isConcurrencySafe).toBeNull();
    expect(build({ isConcurrencySafe: null }).isConcurrencySafe).toBeNull();
  });

  test("accepts an async execute", () => {
    const execute = async () => "done";
    expect(build({ execute }).execute).toBe(execute);
  });

  test("stores a copy of the schema", () => {
    const schema = { type: "object", properties: { text: { type: "string" } } };
    const tool = build({ parametersJsonSchema: schema });
    schema.properties.text.type = "integer";
    expect(tool.parametersJsonSchema).not.toBe(schema);
    expect(tool.parametersJsonSchema).toEqual({ type: "object", properties: { text: { type: "string" } } });
  });

  test("freezes its schema copy", () => {
    const tool = build({ parametersJsonSchema: { type: "object", properties: { text: { type: "string" } } } });
    const properties = tool.parametersJsonSchema.properties as Record<string, JsonObject>;
    expect(() => {
      properties.text!.type = "integer";
    }).toThrow(TypeError);
    expect(() => {
      properties.extra = { type: "string" };
    }).toThrow(TypeError);
    expect(tool.parametersJsonSchema).toEqual({ type: "object", properties: { text: { type: "string" } } });
  });

  test("a second module copy's FunctionTool passes isFunctionTool", async () => {
    // A computed specifier, so `tsc` doesn't resolve it. Bun shares one instance for a `file:` URL
    // whatever its query, but loads a second instance for a relative specifier with a query.
    const specifier = "../../src/tools/function_tools.ts?copy=2";
    const copy = (await import(specifier)) as typeof import("../../src/tools/function_tools.ts");
    expect(copy.FunctionTool).not.toBe(FunctionTool);
    const tool = new copy.FunctionTool(FIELDS);
    expect(tool).not.toBeInstanceOf(FunctionTool);
    expect(isFunctionTool(tool)).toBe(true);
    expect(isFunctionTool(build({}))).toBe(true);
    expect(isFunctionTool({ ...FIELDS })).toBe(false);
    expect(isFunctionTool(null)).toBe(false);
  });
});

describe("toolTypes", () => {
  test("reads a FunctionTool's JSON Schema", () => {
    const types = toolTypes(build({}));
    expect([...types.inputs!.keys()]).toEqual(["text"]);
    expect(typeEquals(types.inputs!.get("text")!, Type.scalar(ValueKind.STRING))).toBe(true);
    expect(types.required).toEqual(new Set(["text"]));
    expect(types.output).toBeNull();
  });

  test("translates the inputs of a JSON Schema with a $defs ref and bounds", () => {
    const tool = build({
      name: "json_schema_tool",
      parametersJsonSchema: {
        type: "object",
        $defs: { Mode: { enum: ["x", "y"], type: "string" } },
        properties: {
          value: { type: "integer", minimum: 0, maximum: 5 },
          mode: { $ref: "#/$defs/Mode" },
        },
        required: ["value"],
        additionalProperties: false,
      },
    });
    const types = toolTypes(tool);
    expect(inputLabels(tool)).toEqual({ value: "int(0..5)", mode: "Literal(x, y)" });
    expect(types.required).toEqual(new Set(["value"]));
    expect(types.output).toBeNull();
  });

  test("reads no inputs when properties is not an object", () => {
    const types = toolTypes(build({ parametersJsonSchema: { type: "object", properties: [1] } }));
    expect(types.inputs).toBeNull();
    expect(types.required).toEqual(new Set());
  });

  test("leaves the ask_user handler's options unchecked", () => {
    // The run cleans `options` before checking it, so any shape is accepted.
    const tool = build({
      name: "ask_user",
      parametersJsonSchema: {
        type: "object",
        properties: { question: { type: "string" }, options: { type: "array", items: { type: "string" } } },
        required: ["question"],
      },
    });
    expect(inputLabels(tool)).toEqual({ question: "str", options: null });
    expect(toolTypes(tool).required).toEqual(new Set(["question"]));
  });
});

describe("FunctionTool in a ToolSet and the registry", () => {
  let saved: [string, FunctionTool][] = [];

  // Keep process-global registrations isolated between tests.
  beforeEach(() => {
    saved = [...TOOL_REGISTRY];
  });
  afterEach(() => {
    TOOL_REGISTRY.clear();
    for (const [id, tool] of saved) TOOL_REGISTRY.set(id, tool);
  });

  test("its key must match its name", () => {
    expect(() => new ToolSet({ other: build({}) })).toThrow(
      new RangeError("tool 'other' is a FunctionTool named 'echo'; its name must match its key"),
    );
  });

  test("a group item is named by its name", () => {
    const tool = build({});
    const tools = new ToolSet({ box: [tool] });
    expect(tools.expand(["box"])).toEqual(["echo"]);
    expect(tools.resolve(["echo"])).toEqual([tool]);
  });

  test("it can't be the ask_user handler", () => {
    expect(() => new ToolSet({ ask_user: build({ name: "ask_user" }) as never })).toThrow(
      new TypeError('tools["ask_user"] must be a plain function handler; the engine supplies its schema'),
    );
  });

  test("a hostile object or a branded object of the wrong shape gets the registry's own TypeError", () => {
    const hostile = new Proxy(
      {},
      {
        get() {
          throw new Error("trap");
        },
      },
    );
    expect(isFunctionTool(hostile)).toBe(false);
    expect(() => new ToolSet({ x: hostile as never })).toThrow(
      new TypeError('tools["x"] must be a FunctionTool, an MCP server, or an array of FunctionTools, got Object'),
    );
    expect(() => registerTool("x", hostile as never)).toThrow(
      new TypeError(
        "registerTool('x') got a Object; it accepts a FunctionTool. " +
          "Pass anything else in the run's tools mapping instead",
      ),
    );
    const brand = Symbol.for("yian.function_tool");
    const good = { name: "x", description: "X.", parametersJsonSchema: { type: "object" }, execute: () => null };
    expect(isFunctionTool({ ...good, [brand]: true })).toBe(true);
    for (const forged of [
      { ...good, name: 3 },
      { ...good, execute: "run" },
      { ...good, parametersJsonSchema: null },
      { [brand]: true },
    ]) {
      const branded = { ...forged, [brand]: true };
      expect(isFunctionTool(branded)).toBe(false);
      expect(() => new ToolSet({ box: [branded as never] })).toThrow(
        new TypeError('tools["box"][0] must be a FunctionTool, got Object'),
      );
    }
  });

  test("registerTool registers it and returns it unchanged", () => {
    const tool = build({});
    expect(registerTool("echo", tool)).toBe(tool);
    expect(TOOL_REGISTRY.get("echo")).toBe(tool);
    expect(new ToolSet().resolve(["echo"])).toEqual([tool]);
  });

  test("registerTool rejects it under another name", () => {
    expect(() => registerTool("other", build({}))).toThrow(
      new RangeError("tool id 'other' names a FunctionTool named 'echo'; its name must match its id"),
    );
    expect(TOOL_REGISTRY.has("other")).toBe(false);
  });

  test("registerTool stores the description override on a copy", () => {
    const tool = build({});
    expect(registerTool("echo", tool, { descriptionOverride: "New." })).toBe(tool);
    const stored = TOOL_REGISTRY.get("echo")!;
    expect(stored).not.toBe(tool);
    expect(isFunctionTool(stored)).toBe(true);
    expect(stored.description).toBe("New.");
    expect(stored.name).toBe("echo");
    expect(stored.execute).toBe(tool.execute);
    expect(stored.parametersJsonSchema).toEqual(tool.parametersJsonSchema);
    expect(tool.description).toBe("Echo.");
  });
});

describe("FunctionTool invocation", () => {
  let saved: [string, FunctionTool][] = [];

  // Keep process-global registrations isolated between tests.
  beforeEach(() => {
    saved = [...TOOL_REGISTRY];
  });
  afterEach(() => {
    TOOL_REGISTRY.clear();
    for (const [id, tool] of saved) TOOL_REGISTRY.set(id, tool);
  });

  /** Build an echo tool with `schema` that records each call, and the list it records into. */
  function echo(
    schema: JsonObject = ECHO_SCHEMA,
    name = "echo",
  ): [FunctionTool, [ToolContext<any>, Record<string, unknown>][]] {
    const seen: [ToolContext<any>, Record<string, unknown>][] = [];
    const tool = new FunctionTool({
      name,
      description: "Echo.",
      parametersJsonSchema: schema,
      execute: (args, ctx) => {
        seen.push([ctx, args]);
        return "done";
      },
    });
    return [tool, seen];
  }

  test("test_function_tool_is_invoked_with_a_context_and_its_arguments", async () => {
    const [ft, seen] = echo();
    const inputs = { text: "hi" };

    expect(await new ToolSet({ echo: ft }).forRun("r").invoke("echo", inputs, { node: "n" })).toBe("done");

    expect(seen).toHaveLength(1);
    const [ctx, args] = seen[0]!;
    expect(ctx).toBeInstanceOf(ToolContext);
    expect(ctx.toolName).toBe("echo");
    expect(ctx.node).toBe("n");
    expect(ctx.callId).toBe("r/n");
    expect(args).toEqual({ text: "hi" });
    expect(Object.getPrototypeOf(args)).toBe(Object.prototype);
    expect(args).not.toBe(inputs);
  });

  test("test_function_tool_rejects_an_unknown_argument_before_on_invoke", async () => {
    const [ft, seen] = echo();

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { text: "hi", extra: 1 }), RangeError);
    expect(error.message).toBe('tool \'echo\' received unsupported argument(s) ["extra"]; allowed: ["text"]');
    expect(seen).toEqual([]);
  });

  test("an inherited property name is an unknown argument", async () => {
    const [ft, seen] = echo();

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { text: "hi", toString: 1 }), RangeError);
    expect(error.message).toBe('tool \'echo\' received unsupported argument(s) ["toString"]; allowed: ["text"]');
    expect(seen).toEqual([]);
  });

  for (const extra of [true, { type: "integer" }]) {
    const label = JSON.stringify(extra);
    test(`test_function_tool_accepts_unknown_arguments_when_the_schema_allows_them[${label}]`, async () => {
      const [ft, seen] = echo({ ...ECHO_SCHEMA, additionalProperties: extra });

      await new ToolSet({ echo: ft }).invoke("echo", { text: "hi", extra: 1 });

      expect(seen[0]![1]).toEqual({ text: "hi", extra: 1 });
    });
  }

  test("test_function_tool_checks_extra_arguments_against_an_additional_properties_schema", async () => {
    const [ft, seen] = echo({ ...ECHO_SCHEMA, additionalProperties: { type: "integer" } });

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { text: "hi", extra: "x" }), TypeCheckError);
    expect(error.message).toStartWith("tool 'echo' argument 'extra': ");
    expect((error as TypeCheckError).path).toEqual(["extra"]);
    expect(seen).toEqual([]);
  });

  test("test_function_tool_rejects_a_missing_required_argument", async () => {
    const [ft, seen] = echo();

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", {}), RangeError);
    expect(error.message).toBe('tool \'echo\' is missing required argument(s) ["text"]');
    expect(seen).toEqual([]);
  });

  test("missing required arguments keep the schema's order", async () => {
    const [ft] = echo({
      type: "object",
      properties: { a: { type: "string" }, b: { type: "string" }, c: { type: "string" } },
      required: ["c", "a", "b"],
    });

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { a: "x" }), RangeError);
    expect(error.message).toBe('tool \'echo\' is missing required argument(s) ["c","b"]');
  });

  test("test_function_tool_rejects_a_value_of_the_wrong_type", async () => {
    const [ft, seen] = echo();

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { text: 3 }), TypeCheckError);
    expect(error.message).toStartWith("tool 'echo' argument 'text': ");
    expect((error as TypeCheckError).path).toEqual(["text"]);
    expect(error.cause).toBeInstanceOf(TypeCheckError);
    expect(seen).toEqual([]);
  });

  test("test_function_tool_passes_an_untranslatable_property_through_unchecked", async () => {
    const [ft, seen] = echo({
      type: "object",
      properties: { v: { anyOf: [{ type: "string" }, { type: "integer" }] } },
    });

    await new ToolSet({ echo: ft }).invoke("echo", { v: [1, 2] });

    expect(seen[0]![1]).toEqual({ v: [1, 2] });
  });

  test("test_function_tool_accepts_an_integral_float_for_an_integer_as_given", async () => {
    const [ft, seen] = echo({ type: "object", properties: { n: { type: "integer", minimum: 1 } } });
    const tools = new ToolSet({ echo: ft });

    await tools.invoke("echo", parseJson('{"n": 3.0}', "plain") as Record<string, unknown>);

    expect(seen[0]![1]).toEqual({ n: 3 });
    await rejection(tools.invoke("echo", { n: 3.5 }), TypeCheckError);
    await rejection(tools.invoke("echo", { n: 0 }), TypeCheckError);
  });

  test("test_function_tool_trusted_invocation_validates_the_same_way", async () => {
    const [ft] = echo();
    const tools = new ToolSet({ echo: ft });

    expect(await tools.invokeTrusted("echo", { text: "hi" })).toBe("done");
    const error = await rejection(tools.invokeTrusted("echo", { text: "hi", extra: 1 }), RangeError);
    expect(error.message).toBe('tool \'echo\' received unsupported argument(s) ["extra"]; allowed: ["text"]');
  });

  test("a schema with a $ref checks the referenced type and is kept as given", async () => {
    const schema: JsonObject = {
      type: "object",
      title: "Order",
      $defs: { Side: { type: "string", enum: ["buy", "sell"], title: "Side" } },
      properties: { side: { $ref: "#/$defs/Side" }, qty: { type: "integer" } },
      required: ["side"],
    };
    const [ft] = echo(schema, "order");
    const tools = new ToolSet({ order: ft });

    expect(tools.resolve(["order"])[0]!.parametersJsonSchema).toEqual(schema);
    await tools.invoke("order", { side: "buy" });
    await rejection(tools.invoke("order", { side: "hold" }), TypeCheckError);
  });

  test("test_on_invoke_gets_a_deep_copy_of_the_arguments", async () => {
    const ft = new FunctionTool({
      name: "grow",
      description: "",
      parametersJsonSchema: { type: "object", properties: { items: { type: "array" } } },
      execute: (args) => {
        (args.items as unknown[]).push("added");
        args.new = 1;
        return (args.items as unknown[]).length;
      },
    });
    const items = ["a"];
    const inputs = { items };

    expect(await new ToolSet({ grow: ft }).invoke("grow", inputs)).toBe(2);
    expect(items).toEqual(["a"]);
    expect(inputs).toEqual({ items: ["a"] });
  });

  test("test_function_tool_accepts_integral_floats_in_an_integer_array_as_given", async () => {
    const [ft, seen] = echo({ type: "object", properties: { ns: { type: "array", items: { type: "integer" } } } });
    const tools = new ToolSet({ echo: ft });

    await tools.invoke("echo", parseJson('{"ns": [1.0, 2]}', "plain") as Record<string, unknown>);

    expect(seen[0]![1]).toEqual({ ns: [1, 2] });
    await rejection(tools.invoke("echo", { ns: [1.5] }), TypeCheckError);
  });

  test("test_function_tool_accepts_integral_floats_in_a_closed_record", async () => {
    const [ft, seen] = echo({
      type: "object",
      properties: {
        pos: {
          type: "object",
          properties: { x: { type: "integer" } },
          required: ["x"],
          additionalProperties: false,
        },
      },
    });
    const tools = new ToolSet({ echo: ft });

    await tools.invoke("echo", parseJson('{"pos": {"x": 4.0}}', "plain") as Record<string, unknown>);

    expect(seen[0]![1]).toEqual({ pos: { x: 4 } });
    const error = await rejection(tools.invoke("echo", { pos: { x: 4.5 } }), TypeCheckError);
    expect((error as TypeCheckError).path).toEqual(["pos", "x"]);
  });

  test("the tool receives the host form of its arguments", async () => {
    const [ft, seen] = echo({ type: "object", properties: {}, additionalProperties: true });

    await new ToolSet({ echo: ft }).invoke("echo", { small: 2n, big: 2n ** 60n, inf: Infinity });

    expect(seen[0]![1]).toEqual({ small: 2, big: 2n ** 60n, inf: Infinity });
  });

  test("a non-yian value in the arguments is refused", async () => {
    const [ft, seen] = echo();

    const error = await rejection(new ToolSet({ echo: ft }).invoke("echo", { text: undefined }), TypeCheckError);
    expect(error.message).toBe("tool 'echo' arguments at text: undefined is not a yian value; use null");
    expect(seen).toEqual([]);
  });

  test("an async execute is awaited and an undefined result is null", async () => {
    const run = (execute: () => unknown) =>
      new ToolSet({ echo: new FunctionTool({ ...FIELDS, execute }) }).invoke("echo", { text: "hi" });

    expect(await run(async () => "later")).toBe("later");
    expect(await run(() => undefined)).toBeNull();
    expect(await run(async () => undefined)).toBeNull();
  });

  test("a sync throw and a rejection both reject invoke, unchanged", async () => {
    const boom = new Error("boom");
    for (const execute of [
      () => {
        throw boom;
      },
      async () => {
        throw boom;
      },
    ]) {
      const tools = new ToolSet({ echo: new FunctionTool({ ...FIELDS, execute }) });
      const promise = tools.invoke("echo", { text: "hi" });
      expect(promise).toBeInstanceOf(Promise);
      expect(await rejection(promise, Error)).toBe(boom);
    }
  });
});

describe("tool()", () => {
  let saved: [string, FunctionTool][] = [];

  // Keep process-global registrations isolated between tests.
  beforeEach(() => {
    saved = [...TOOL_REGISTRY];
  });
  afterEach(() => {
    TOOL_REGISTRY.clear();
    for (const [id, tool] of saved) TOOL_REGISTRY.set(id, tool);
  });

  const PLACE_PARAMETERS = z.object({
    sym: z.string().describe("The ticker."),
    qty: z.number().int().min(1),
    mode: z.enum(["a", "b"]).optional(),
  });
  const FILLED = z.object({ filled: z.number().int() });

  /** Build the zod `place` tool. */
  function place(): FunctionTool {
    return tool({
      name: "place",
      description: "Place an order.",
      parameters: PLACE_PARAMETERS,
      output: FILLED,
      execute: (args) => ({ filled: args.qty }),
    });
  }

  /** Check `place`'s types as `toolTypes` reads them. */
  function expectPlaceTypes(placed: FunctionTool): void {
    const types = toolTypes(placed);
    expect(inputLabels(placed)).toEqual({ sym: "str", qty: "int(1..9007199254740991)", mode: "Literal(a, b)" });
    expect(types.required).toEqual(new Set(["sym", "qty"]));
    const filled = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["filled", new Type({ kind: ValueKind.INTEGER, bounds: [-9007199254740991n, 9007199254740991n] })],
      ]),
      required: new Set(["filled"]),
    });
    expect(types.output).not.toBeNull();
    expect(typeEquals(types.output!, filled)).toBe(true);
  }

  /** Return the error `build` throws, after checking its class. */
  function thrown(build: () => unknown, kind: abstract new (...args: never[]) => Error): Error {
    try {
      build();
    } catch (error) {
      expect((error as Error).constructor).toBe(kind);
      return error as Error;
    }
    throw new Error("expected a throw");
  }

  test("test_function_tool_builds_a_function_tool_from_a_function", () => {
    const placed = place();

    expect(isFunctionTool(placed)).toBe(true);
    expect(placed.name).toBe("place");
    expect(placed.description).toBe("Place an order.");
    expect(placed.parametersJsonSchema).toEqual(
      PLACE_PARAMETERS["~standard"].jsonSchema.input({ target: "draft-2020-12" }) as JsonObject,
    );
    expect(placed.parametersValidator).toBe(PLACE_PARAMETERS);
    expect(placed.outputJsonSchema).toEqual(
      FILLED["~standard"].jsonSchema.input({ target: "draft-2020-12" }) as JsonObject,
    );
    expect(placed.outputValidator).toBe(FILLED);
  });

  test("a description in the validator reaches the JSON Schema", () => {
    const properties = place().parametersJsonSchema.properties as Record<string, JsonObject>;
    expect(properties.sym!.description).toBe("The ticker.");
  });

  test("test_function_tool_result_binds_and_reads_types_like_a_function, test_function_tool_types", () => {
    expectPlaceTypes(place());
  });

  test("test_function_tool_registers_with_register_tool: registry, mapping and group read the same types", () => {
    const placed = place();
    registerTool("place", placed);

    expectPlaceTypes(TOOL_REGISTRY.get("place")!);
    expectPlaceTypes(new ToolSet().resolve(["place"])[0]!);
    expectPlaceTypes(new ToolSet({ place: placed }).resolve(["place"])[0]!);
    expectPlaceTypes(new ToolSet({ g: [placed] }).resolve(["place"])[0]!);
  });

  test("test_register_tool_description_override_on_a_function_tool_from_a_function", async () => {
    const placed = place();
    registerTool("place", placed, { descriptionOverride: "Buy." });

    const stored = TOOL_REGISTRY.get("place")!;
    expect(stored).not.toBe(placed);
    expect(stored.description).toBe("Buy.");
    expect(stored.parametersValidator).toBe(PLACE_PARAMETERS);
    expect(stored.outputValidator).toBe(FILLED);
    expect(stored.outputJsonSchema).toEqual(placed.outputJsonSchema!);
    expectPlaceTypes(stored);

    // The copy carries the validator, so a call through the registry runs it.
    TOOL_REGISTRY.delete("place");
    const { placed: checked, seen } = checkedPlace();
    registerTool("place", checked, { descriptionOverride: "Buy." });
    const error = await rejection(new ToolSet().invoke("place", { sym: "abc" }), ToolCheckError);
    expect(error.message).toBe("tool 'place' arguments failed its validator: sym: must be upper case");
    expect(await new ToolSet().invoke("place", { sym: "X" })).toBe("X:1");
    expect(seen.map(([args]) => args)).toEqual([{ sym: "X", qty: 1 }]);
  });

  test("test_registered_tools_copy_keeps_the_output_type", () => {
    registerTool("place", place());

    expectPlaceTypes(new ToolSet(registeredTools()).resolve(["place"])[0]!);
  });

  test("a raw JSON Schema has no validators and its output type is read", () => {
    const echoed = tool({
      name: "echo",
      description: "Echo.",
      parameters: ECHO_SCHEMA,
      output: { type: "string" },
      execute: (args) => String(args.text),
    });

    expect(echoed.parametersJsonSchema).toEqual(ECHO_SCHEMA);
    expect(echoed.parametersValidator).toBeNull();
    expect(echoed.outputValidator).toBeNull();
    expect(echoed.outputJsonSchema).toEqual({ type: "string" });
    expect(typeEquals(toolTypes(echoed).output!, Type.scalar(ValueKind.STRING))).toBe(true);
    const plain = tool({ name: "echo", description: "", parameters: ECHO_SCHEMA, execute: () => null });
    expect(plain.outputJsonSchema).toBeNull();
  });

  test("test_explicit_none_return_is_none_kind", () => {
    const nothing = tool({
      name: "echo",
      description: "",
      parameters: ECHO_SCHEMA,
      output: { type: "null" },
      execute: () => null,
    });

    expect(typeEquals(toolTypes(nothing).output!, Type.scalar(ValueKind.NONE))).toBe(true);
  });

  for (const output of [{}, { allOf: [{ type: "string" }, { minLength: 1 }] }] as JsonObject[]) {
    test(`test_return_without_a_json_like_type_has_no_output[${JSON.stringify(output)}]`, () => {
      const untyped = tool({ name: "echo", description: "", parameters: ECHO_SCHEMA, output, execute: () => null });

      expect(untyped.outputJsonSchema).toEqual(output);
      expect(toolTypes(untyped).output).toBeNull();
    });
  }

  test("a validator with no Standard JSON Schema needs parametersJsonSchema", () => {
    const hand: StandardSchemaV1 = { "~standard": { version: 1, vendor: "hand", validate: (value) => ({ value }) } };

    const error = thrown(() => tool({ name: "x", description: "", parameters: hand, execute: () => null }), TypeError);
    expect(error.message).toBe(
      "tool 'x' parameters: the hand validator has no Standard JSON Schema (~standard.jsonSchema); " +
        "pass parametersJsonSchema too",
    );
    const built = tool({
      name: "x",
      description: "",
      parameters: hand,
      parametersJsonSchema: ECHO_SCHEMA,
      execute: () => null,
    });
    expect(built.parametersJsonSchema).toEqual(ECHO_SCHEMA);
    expect(built.parametersValidator).toBe(hand);

    const outputError = thrown(
      () => tool({ name: "x", description: "", parameters: ECHO_SCHEMA, output: hand, execute: () => null }),
      TypeError,
    );
    expect(outputError.message).toBe(
      "tool 'x' output: the hand validator has no Standard JSON Schema (~standard.jsonSchema); " +
        "pass outputJsonSchema too",
    );
    const withOutput = tool({
      name: "x",
      description: "",
      parameters: ECHO_SCHEMA,
      output: hand,
      outputJsonSchema: { type: "string" },
      execute: () => null,
    });
    expect(withOutput.outputJsonSchema).toEqual({ type: "string" });
    expect(withOutput.outputValidator).toBe(hand);
  });

  for (const [what, bad] of [
    ["a non-object ~standard", { "~standard": 5 }],
    ["a version 2 ~standard", { "~standard": { version: 2, validate() {} } }],
  ] as unknown as [string, StandardSchemaV1][]) {
    test(`refuses ${what} as parameters`, () => {
      expect(() => tool({ name: "x", description: "", parameters: bad, execute: () => null })).toThrow(
        new TypeError("tool 'x' parameters must be a Standard Schema v1 validator"),
      );
      // Refused up front, even with a JSON Schema beside it.
      expect(() =>
        tool({ name: "x", description: "", parameters: bad, parametersJsonSchema: ECHO_SCHEMA, execute: () => null }),
      ).toThrow(new TypeError("tool 'x' parameters must be a Standard Schema v1 validator"));
    });
  }

  test("refuses a non-v1 validator as output", () => {
    const bad = { "~standard": { version: 2, validate() {} } } as unknown as StandardSchemaV1;
    expect(() =>
      tool({ name: "x", description: "", parameters: ECHO_SCHEMA, output: bad, execute: () => null }),
    ).toThrow(new TypeError("tool 'x' output must be a Standard Schema v1 validator"));
  });

  test("a plain valibot schema needs parametersJsonSchema", () => {
    const error = thrown(
      () => tool({ name: "x", description: "", parameters: v.object({ symbol: v.string() }), execute: () => null }),
      TypeError,
    );
    expect(error.message).toBe(
      "tool 'x' parameters: the valibot validator has no Standard JSON Schema (~standard.jsonSchema); " +
        "pass parametersJsonSchema too",
    );
  });

  test("test_function_tool_names_the_tool_when_an_argument_has_no_json_schema", () => {
    const error = thrown(
      () =>
        tool({ name: "when", description: "", parameters: z.object({ when: z.coerce.date() }), execute: () => null }),
      TypeError,
    );
    expect(error.message).toBe(
      "tool 'when' has an argument type with no JSON Schema: Date cannot be represented in JSON Schema",
    );
    expect(error.cause).toBeInstanceOf(Error);

    const outputError = thrown(
      () => tool({ name: "x", description: "", parameters: ECHO_SCHEMA, output: z.date(), execute: () => new Date(0) }),
      TypeError,
    );
    expect(outputError.message).toBe(
      "tool 'x' has an output type with no JSON Schema: Date cannot be represented in JSON Schema",
    );
  });

  test("parametersJsonSchema and outputJsonSchema are only for a validator", () => {
    expect(() =>
      tool({
        name: "x",
        description: "",
        parameters: ECHO_SCHEMA,
        parametersJsonSchema: ECHO_SCHEMA,
        execute: () => null,
      }),
    ).toThrow(new TypeError("tool 'x' parametersJsonSchema is only for a validator"));
    expect(() =>
      tool({
        name: "x",
        description: "",
        parameters: ECHO_SCHEMA,
        output: { type: "string" },
        outputJsonSchema: { type: "string" },
        execute: () => null,
      }),
    ).toThrow(new TypeError("tool 'x' outputJsonSchema is only for a validator"));
  });

  test("parametersJsonSchema given with a zod validator replaces zod's schema", () => {
    const built = tool({
      name: "x",
      description: "",
      parameters: PLACE_PARAMETERS,
      parametersJsonSchema: ECHO_SCHEMA,
      execute: () => null,
    });

    expect(built.parametersJsonSchema).toEqual(ECHO_SCHEMA);
    expect(built.parametersValidator).toBe(PLACE_PARAMETERS);
  });

  test("a zod parameter that isn't an object gets FunctionTool's check", () => {
    expect(() => tool({ name: "x", description: "", parameters: z.string(), execute: () => null })).toThrow(
      new RangeError("FunctionTool 'x' parametersJsonSchema must have type 'object'"),
    );
  });

  test("execute and isConcurrencySafe get the validator's output type", () => {
    const counted = tool({
      name: "count",
      description: "Count.",
      parameters: z.object({ n: z.number() }),
      output: z.number(),
      execute: (args) => {
        // @ts-expect-error `missing` is not a parameter.
        void args.missing;
        return args.n + 1;
      },
      isConcurrencySafe: (args) => args.n > 0,
    });
    expect(counted.isConcurrencySafe!({ n: 1 })).toBe(true);
    // @ts-expect-error a string is not the output type.
    tool({ name: "bad", description: "", parameters: ECHO_SCHEMA, output: z.number(), execute: () => "no" });
  });
});

describe("tool() checking order", () => {
  /** Build a tool `x` on the echo schema whose validator is `validate`. */
  function handTool(validate: StandardSchemaV1["~standard"]["validate"]): FunctionTool {
    return tool({
      name: "x",
      description: "",
      parameters: { "~standard": { version: 1, vendor: "hand", validate } },
      parametersJsonSchema: ECHO_SCHEMA,
      execute: (args) => args,
    });
  }

  test("yian's check runs first and the validator is never called", async () => {
    const { placed, seen, count } = checkedPlace();
    const tools = new ToolSet({ place: placed });

    const unknown = await rejection(tools.validate("place", { sym: "X", ctx: "forged" }), RangeError);
    expect(unknown.message).toBe(
      'tool \'place\' received unsupported argument(s) ["ctx"]; allowed: ["note","qty","sym"]',
    );
    await rejection(tools.invoke("place", { sym: "X", qty: "2" }), TypeCheckError);
    expect(count.calls).toBe(0);
    expect(seen).toEqual([]);
  });

  test("the validator's issues fail the call as a ToolCheckError", async () => {
    const { placed, seen } = checkedPlace();

    const error = await rejection(new ToolSet({ place: placed }).invoke("place", { sym: "abc" }), ToolCheckError);
    expect(isToolCheckError(error)).toBe(true);
    expect(error.message).toBe("tool 'place' arguments failed its validator: sym: must be upper case");
    expect(seen).toEqual([]);
  });

  test("each issue is shown with its path, when it has one", async () => {
    const issues = [
      { message: "no path" },
      { message: "empty path", path: [] },
      { message: "keyed", path: ["a", { key: 0 }, { key: "b" }] },
      { message: "symbol", path: [Symbol("s")] },
    ];
    const tools = new ToolSet({ x: handTool(() => ({ issues })) });

    const error = await rejection(tools.validate("x", { text: "hi" }), ToolCheckError);
    expect(error.message).toBe(
      'tool \'x\' arguments failed its validator: no path; empty path; a[0].b: keyed; ["Symbol(s)"]: symbol',
    );
    const empty = new ToolSet({ x: handTool(() => ({ issues: [] })) });
    const none = await rejection(empty.validate("x", { text: "hi" }), ToolCheckError);
    expect(none.message).toBe("tool 'x' arguments failed its validator");
  });

  test("the validator runs once in validate and every attempt gets its output", async () => {
    const { placed, seen, count } = checkedPlace();
    const tools = new ToolSet({ place: placed });

    const call = await tools.validate("place", { sym: "X", note: " hi " });
    expect(count.calls).toBe(1);
    for (const attempt of [1, 2, 3]) expect(await tools.invoke(call, { attempt })).toBe("X:1");

    expect(count.calls).toBe(1);
    expect(seen.map(([, ctx]) => ctx.attempt)).toEqual([1, 2, 3]);
    expect(seen[0]![0]).toEqual({ sym: "X", qty: 1, note: "hi" });
    expect(seen[1]![0]).toBe(seen[0]![0]);
    expect(seen[2]![0]).toBe(seen[0]![0]);
  });

  test("a one-shot invoke runs the validator once per call", async () => {
    const { placed, count } = checkedPlace();
    const tools = new ToolSet({ place: placed });

    await tools.invoke("place", { sym: "X" });
    await tools.invoke("place", { sym: "Y", qty: 2 });

    expect(count.calls).toBe(2);
  });

  test("an async validator is awaited, and a throwing one propagates unchanged", async () => {
    const delayed = handTool(async (value) => {
      await Promise.resolve();
      return { value: { ...(value as object), late: true } };
    });
    expect(await new ToolSet({ x: delayed }).invoke("x", { text: "hi" })).toEqual({ text: "hi", late: true });

    const boom = new Error("boom");
    const throwing = handTool(() => {
      throw boom;
    });
    expect(await rejection(new ToolSet({ x: throwing }).validate("x", { text: "hi" }), Error)).toBe(boom);
  });

  test("an already-aborted signal stops validate before the validator runs", async () => {
    const { placed, count } = checkedPlace();
    const controller = new AbortController();
    const reason = new Error("stopped");
    controller.abort(reason);

    const tools = new ToolSet({ place: placed });
    await expect(tools.validate("place", { sym: "X" }, { signal: controller.signal })).rejects.toBe(reason);
    expect(count.calls).toBe(0);
    // With no signal passed, the one `forRun` bound is used.
    await expect(tools.forRun("r", { signal: controller.signal }).validate("place", { sym: "X" })).rejects.toBe(reason);
    await expect(tools.forRun("r", { signal: controller.signal }).invoke("place", { sym: "X" })).rejects.toBe(reason);
    expect(count.calls).toBe(0);
  });

  test("a stop while the validator runs rejects after it returns", async () => {
    const controller = new AbortController();
    const reason = new Error("stopped");
    const { placed, seen, count } = checkedPlace({ during: () => controller.abort(reason) });

    const tools = new ToolSet({ place: placed });
    await expect(tools.invoke("place", { sym: "X" }, { signal: controller.signal })).rejects.toBe(reason);
    expect(count.calls).toBe(1);
    expect(seen).toEqual([]);
  });

  test("isConcurrencySafe receives the object execute gets", async () => {
    const safe: Checked[] = [];
    const { placed, seen } = checkedPlace({
      isConcurrencySafe: (args) => {
        safe.push(args);
        return true;
      },
    });
    const tools = new ToolSet({ place: placed });
    const call = await tools.validate("place", { sym: "X" });

    expect(tools.isConcurrencySafe(call)).toBe(true);
    await tools.invoke(call);

    expect(safe[0]).toEqual({ sym: "X", qty: 1 });
    expect(seen[0]![0]).toBe(safe[0]!);
  });

  test("invokeTrusted runs the validator too", async () => {
    const { placed, count } = checkedPlace();
    const tools = new ToolSet({ place: placed });

    await rejection(tools.invokeTrusted("place", { sym: "abc" }), ToolCheckError);
    expect(await tools.invokeTrusted("place", { sym: "X" })).toBe("X:1");
    expect(count.calls).toBe(2);
  });

  test("test_function_tool_result_validates_like_a_function_and_passes_the_context", async () => {
    const app = { name: "broker" };
    const { placed, seen } = checkedPlace();
    const tools = new ToolSet({ place: placed }, { context: app }).forRun("r");

    expect(await tools.invoke("place", { sym: "X", qty: 2 }, { node: "n" })).toBe("X:2");

    const [args, ctx] = seen[0]!;
    expect(args).toEqual({ sym: "X", qty: 2 });
    expect(ctx.callId).toBe("r/n");
    expect(ctx.context).toBe(app);
    await rejection(tools.invoke("place", { sym: "X", ctx: "forged" }), RangeError);
    await rejection(tools.invoke("place", { sym: "X", qty: "many" }), TypeCheckError);
  });

  test("test_function_tool_without_context_gets_its_validated_arguments", async () => {
    const add = tool({
      name: "add",
      description: "Add two integers.",
      parameters: z.object({ a: z.number().int(), b: z.number().int().default(4) }),
      execute: (args) => args.a + args.b,
    });

    expect(await new ToolSet({ add }).invoke("add", { a: 3 })).toBe(7);
  });
});

/** The `ts/` directory. */
const ROOT = resolve(import.meta.dir, "../..");
const BUILT = join(ROOT, "dist/src/tools/index.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/platform.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

test.skipIf(!NODE)(
  "the built tools code runs on Node beside a second copy of its leaves",
  () => {
    const url = (file: string) => JSON.stringify(pathToFileURL(join(ROOT, file)).href);
    // Node keys modules by full URL, so a query loads a second instance of that module. Copying
    // `index.js` would re-run only `index.js` and share its leaves, so the leaves are copied.
    const script = `
      import assert from "node:assert/strict";

      const tools = await import(${url("dist/src/tools/index.js")});
      const copyErrors = await import(${url("dist/src/tools/errors.js")} + "?copy=2");
      const copyTools = await import(${url("dist/src/tools/function_tools.js")} + "?copy=2");

      const schema = { type: "object", properties: { n: { type: "integer" } }, required: ["n"] };
      const count = new tools.FunctionTool({
        name: "count",
        description: "Count.",
        parametersJsonSchema: schema,
        execute: (args) => args.n,
      });
      const set = new tools.ToolSet({ count });
      assert.equal(await set.invoke("count", { n: 3 }), 3);
      assert.equal(await set.invoke("count", { n: 3n }), 3);
      assert.equal(await set.invoke("count", { n: 2n ** 70n }), 2n ** 70n);
      await assert.rejects(set.invoke("count", { n: 2.5 }), { name: "TypeCheckError" });

      assert.notEqual(copyErrors.PermanentToolError, tools.PermanentToolError);
      assert.notEqual(copyTools.FunctionTool, tools.FunctionTool);
      const permanent = new copyErrors.PermanentToolError("no");
      assert.ok(!(permanent instanceof tools.PermanentToolError));
      assert.ok(tools.isPermanentToolError(permanent));
      const reported = new copyErrors.ReportedToolError("x", { permanent: true });
      assert.deepEqual(tools.toolErrorKinds(reported), ["reported", "permanent"]);

      const echo = new copyTools.FunctionTool({
        name: "echo",
        description: "Echo.",
        parametersJsonSchema: { type: "object", properties: { text: { type: "string" } } },
        execute: (args, ctx) => args.text + " from " + ctx.toolName,
      });
      assert.ok(!(echo instanceof tools.FunctionTool));
      assert.equal(await new tools.ToolSet({ echo }).invoke("echo", { text: "hi" }), "hi from echo");
      console.log("ok");
    `;
    expect(execFileSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8" })).toBe("ok\n");
  },
  30_000,
);
