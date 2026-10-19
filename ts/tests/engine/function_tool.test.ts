/**
 * `FunctionTool` construction and the types `toolTypes` reads from its JSON Schema, a port of
 * `tests/engine/test_function_tool.py` (construction and types rows) and of the JSON Schema rows
 * of `test_tool_types.py` that read a tool. The invocation, `ToolSet` and `registerTool` rows land
 * with the tool set and registry. Not ported: the `function_tool` and decorator rows (a later
 * `tool()` builds a tool from a function), the async-refusal rows (TS accepts an async
 * `execute`), and the LangChain binding rows.
 */

import { describe, expect, test } from "bun:test";

import { FunctionTool, isFunctionTool } from "../../src/tools/function_tools.ts";
import { toolTypes } from "../../src/tools/types.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { canonicalTypeLabel, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

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
