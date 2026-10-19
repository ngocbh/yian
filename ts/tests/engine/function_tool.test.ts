/**
 * `FunctionTool` construction, the types `toolTypes` reads from its JSON Schema, and a
 * `FunctionTool` in a `ToolSet` and the registry, a port of `tests/engine/test_function_tool.py`
 * and of the JSON Schema rows of `test_tool_types.py` that read a tool. Not ported: the
 * `function_tool` and decorator rows and `test_replace_drops_the_function_schema` (a later
 * `tool()` builds a tool from a function),
 * `test_run_entry_points_and_tool_set_annotate_function_tool_entries` (needs the run entry
 * points), `test_function_tool_without_a_signature_cannot_be_called_directly`, the async-refusal
 * rows (TS accepts and awaits an async `execute`), and the LangChain binding rows.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool, isFunctionTool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
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
      new TypeError("tools['ask_user'] must be a plain function handler; the engine supplies its schema"),
    );
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

  /** Return the error `promise` rejects with, after checking its class. */
  async function rejection(promise: Promise<unknown>, kind: abstract new (...args: never[]) => Error): Promise<Error> {
    try {
      await promise;
    } catch (error) {
      expect(error).toBeInstanceOf(kind);
      return error as Error;
    }
    throw new Error("expected a rejection");
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
