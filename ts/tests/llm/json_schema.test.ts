/**
 * Tests for the structured-output schema transforms in `src/llm/json_schema.ts`: `hasOpenSchema`,
 * `functionTool` and `constAsEnum`, checked against Python yian's (and LangChain's) results for
 * the schemas `typeToSchema` emits, and on hand-written schemas.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import { constAsEnum, functionTool, hasOpenSchema } from "../../src/llm/json_schema.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

/** A schema Python's `type_to_schema(...).model_json_schema()` emits, and its `has_open_schema`. */
interface Fixture {
  open: boolean;
  schema: JsonObject;
}

/**
 * Python's schemas for the open-schema test's types (`open_schema_test:*`) and the structured-output
 * corpus (`s5:*`): ints are `bigint`s, floats numbers.
 */
const FIXTURES: Record<string, Fixture> = {
  "open_schema_test:object": {
    open: true,
    schema: {
      properties: { value: { additionalProperties: true, title: "Value", type: "object" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:Any": {
    open: true,
    schema: {
      properties: {
        value: {
          anyOf: [
            { type: "boolean" },
            { type: "integer" },
            { type: "number" },
            { type: "string" },
            { additionalProperties: true, type: "object" },
            { items: {}, type: "array" },
          ],
          title: "Value",
        },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:Optional(Any)": {
    open: true,
    schema: {
      properties: {
        value: {
          anyOf: [
            { type: "boolean" },
            { type: "integer" },
            { type: "number" },
            { type: "string" },
            { additionalProperties: true, type: "object" },
            { items: {}, type: "array" },
            { type: "null" },
          ],
          title: "Value",
        },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(object)": {
    open: true,
    schema: {
      properties: {
        items: { items: { additionalProperties: true, type: "object" }, title: "Items", type: "array" },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(Any)": {
    open: true,
    schema: {
      properties: {
        items: {
          items: {
            anyOf: [
              { type: "boolean" },
              { type: "integer" },
              { type: "number" },
              { type: "string" },
              { additionalProperties: true, type: "object" },
              { items: {}, type: "array" },
            ],
          },
          title: "Items",
          type: "array",
        },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(Optional(Any))": {
    open: true,
    schema: {
      properties: {
        items: {
          items: {
            anyOf: [
              { type: "boolean" },
              { type: "integer" },
              { type: "number" },
              { type: "string" },
              { additionalProperties: true, type: "object" },
              { items: {}, type: "array" },
              { type: "null" },
            ],
          },
          title: "Items",
          type: "array",
        },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:Optional(object)": {
    open: true,
    schema: {
      properties: {
        value: { anyOf: [{ additionalProperties: true, type: "object" }, { type: "null" }], title: "Value" },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(list(object))": {
    open: true,
    schema: {
      properties: {
        items: {
          items: { items: { additionalProperties: true, type: "object" }, type: "array" },
          title: "Items",
          type: "array",
        },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:Optional(R)": {
    open: true,
    schema: {
      $defs: {
        Record: {
          properties: {
            a: { title: "A", type: "integer" },
            m: { additionalProperties: true, title: "M", type: "object" },
          },
          required: ["a", "m"],
          title: "Record",
          type: "object",
        },
      },
      properties: { value: { anyOf: [{ $ref: "#/$defs/Record" }, { type: "null" }] } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(R)": {
    open: true,
    schema: {
      $defs: {
        item: {
          properties: {
            a: { title: "A", type: "integer" },
            m: { additionalProperties: true, title: "M", type: "object" },
          },
          required: ["a", "m"],
          title: "item",
          type: "object",
        },
      },
      properties: { items: { items: { $ref: "#/$defs/item" }, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:R": {
    open: true,
    schema: {
      properties: {
        a: { title: "A", type: "integer" },
        m: { additionalProperties: true, title: "M", type: "object" },
      },
      required: ["a", "m"],
      title: "Record",
      type: "object",
    },
  },
  "open_schema_test:D": {
    open: true,
    schema: {
      $defs: {
        buy: {
          additionalProperties: false,
          properties: {
            kind: { const: "buy", title: "Kind", type: "string" },
            x: {
              anyOf: [
                { type: "boolean" },
                { type: "integer" },
                { type: "number" },
                { type: "string" },
                { additionalProperties: true, type: "object" },
                { items: {}, type: "array" },
              ],
              title: "X",
            },
          },
          required: ["kind", "x"],
          title: "buy",
          type: "object",
        },
        sell: {
          additionalProperties: false,
          properties: {
            kind: { const: "sell", title: "Kind", type: "string" },
            n: { title: "N", type: "integer" },
          },
          required: ["kind", "n"],
          title: "sell",
          type: "object",
        },
      },
      properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:Outer": {
    open: true,
    schema: {
      $defs: {
        inner: {
          properties: {
            a: { title: "A", type: "integer" },
            m: { additionalProperties: true, title: "M", type: "object" },
          },
          required: ["a", "m"],
          title: "inner",
          type: "object",
        },
      },
      properties: { inner: { $ref: "#/$defs/inner" }, n: { title: "N", type: "integer" } },
      required: ["inner", "n"],
      title: "Record",
      type: "object",
    },
  },
  "open_schema_test:int": {
    open: false,
    schema: {
      properties: { value: { title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:float(0.0..1.0)": {
    open: false,
    schema: {
      properties: { value: { maximum: 1.0, minimum: 0.0, title: "Value", type: "number" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:bool": {
    open: false,
    schema: {
      properties: { value: { title: "Value", type: "boolean" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:date": {
    open: false,
    schema: {
      properties: { value: { title: "Value", type: "string" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:Literal(a, b)": {
    open: false,
    schema: {
      properties: { value: { enum: ["a", "b"], title: "Value", type: "string" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(str)": {
    open: false,
    schema: {
      properties: { items: { items: { type: "string" }, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:list(int)": {
    open: false,
    schema: {
      properties: { items: { items: { type: "integer" }, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "open_schema_test:C": {
    open: false,
    schema: {
      properties: {
        a: { title: "A", type: "integer" },
        b: { items: { type: "string" }, title: "B", type: "array" },
      },
      required: ["a", "b"],
      title: "Record",
      type: "object",
    },
  },
  "open_schema_test:CU": {
    open: false,
    schema: {
      $defs: {
        c2: {
          additionalProperties: false,
          properties: { kind: { const: "c2", title: "Kind", type: "string" }, a: { title: "A", type: "integer" } },
          required: ["kind", "a"],
          title: "c2",
          type: "object",
        },
        c3: {
          additionalProperties: false,
          properties: { kind: { const: "c3", title: "Kind", type: "string" }, b: { title: "B", type: "string" } },
          required: ["kind", "b"],
          title: "c3",
          type: "object",
        },
      },
      properties: { value: { anyOf: [{ $ref: "#/$defs/c2" }, { $ref: "#/$defs/c3" }], title: "Value" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:Optional(int)": {
    open: false,
    schema: {
      properties: { value: { anyOf: [{ type: "integer" }, { type: "null" }], title: "Value" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "open_schema_test:elementless list": {
    open: true,
    schema: {
      properties: { items: { items: {}, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "s5:Rec": {
    open: false,
    schema: {
      properties: { answer: { title: "Answer", type: "string" }, score: { title: "Score", type: "integer" } },
      required: ["answer", "score"],
      title: "Record",
      type: "object",
    },
  },
  "s5:Bounded": {
    open: false,
    schema: {
      properties: {
        answer: { title: "Answer", type: "string" },
        score: { maximum: 10n, minimum: 0n, title: "Score", type: "integer" },
      },
      required: ["answer", "score"],
      title: "Record",
      type: "object",
    },
  },
  "s5:Open": {
    open: true,
    schema: {
      properties: {
        answer: { title: "Answer", type: "string" },
        data: { additionalProperties: true, title: "Data", type: "object" },
      },
      required: ["answer", "data"],
      title: "Record",
      type: "object",
    },
  },
  "s5:Union": {
    open: false,
    schema: {
      $defs: {
        buy: {
          additionalProperties: false,
          properties: { kind: { const: "buy", title: "Kind", type: "string" }, x: { title: "X", type: "integer" } },
          required: ["kind", "x"],
          title: "buy",
          type: "object",
        },
        sell: {
          additionalProperties: false,
          properties: {
            kind: { const: "sell", title: "Kind", type: "string" },
            n: { title: "N", type: "integer" },
          },
          required: ["kind", "n"],
          title: "sell",
          type: "object",
        },
      },
      properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "s5:_ReflectionResponse": {
    open: false,
    schema: {
      $defs: {
        _ReflectionAction: {
          additionalProperties: false,
          description: "One closed model-proposed lesson action.",
          properties: {
            action: { enum: ["add", "revise", "remove", "noop"], title: "Action", type: "string" },
            lesson_id: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Lesson Id" },
            text: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Text" },
          },
          required: ["action"],
          title: "_ReflectionAction",
          type: "object",
        },
      },
      additionalProperties: false,
      description: "Strict structured result from one reflective model call.",
      properties: {
        actions: { items: { $ref: "#/$defs/_ReflectionAction" }, minItems: 1n, title: "Actions", type: "array" },
      },
      required: ["actions"],
      title: "_ReflectionResponse",
      type: "object",
    },
  },
  "s5:int(0..10)": {
    open: false,
    schema: {
      properties: { value: { maximum: 10n, minimum: 0n, title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "s5:Nested (nullable+list of records, float, date, bool)": {
    open: false,
    schema: {
      $defs: {
        inner: {
          properties: { a: { title: "A", type: "integer" } },
          required: ["a"],
          title: "inner",
          type: "object",
        },
        items_item: {
          properties: { a: { title: "A", type: "integer" } },
          required: ["a"],
          title: "items_item",
          type: "object",
        },
      },
      properties: {
        inner: { anyOf: [{ $ref: "#/$defs/inner" }, { type: "null" }], default: null },
        items: { items: { $ref: "#/$defs/items_item" }, title: "Items", type: "array" },
        f: { maximum: 1.0, minimum: 0.5, title: "F", type: "number" },
        d: { title: "D", type: "string" },
        b: { title: "B", type: "boolean" },
      },
      required: ["items", "f", "d", "b"],
      title: "Record",
      type: "object",
    },
  },
  "s5:Opt (optional field)": {
    open: false,
    schema: {
      properties: {
        req: { title: "Req", type: "string" },
        maybe: { anyOf: [{ type: "integer" }, { type: "null" }], default: null, title: "Maybe" },
      },
      required: ["req"],
      title: "Record",
      type: "object",
    },
  },
  "s5:list(int)": {
    open: false,
    schema: {
      properties: { items: { items: { type: "integer" }, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "s5:Any": {
    open: true,
    schema: {
      properties: {
        value: {
          anyOf: [
            { type: "boolean" },
            { type: "integer" },
            { type: "number" },
            { type: "string" },
            { additionalProperties: true, type: "object" },
            { items: {}, type: "array" },
          ],
          title: "Value",
        },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "s5:object": {
    open: true,
    schema: {
      properties: { value: { additionalProperties: true, title: "Value", type: "object" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "s5:list(Any)": {
    open: true,
    schema: {
      properties: {
        items: {
          items: {
            anyOf: [
              { type: "boolean" },
              { type: "integer" },
              { type: "number" },
              { type: "string" },
              { additionalProperties: true, type: "object" },
              { items: {}, type: "array" },
            ],
          },
          title: "Items",
          type: "array",
        },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "s5:Union(Inner, Opt) untagged": {
    open: false,
    schema: {
      $defs: {
        Member1: {
          additionalProperties: false,
          properties: { a: { title: "A", type: "integer" } },
          required: ["a"],
          title: "Member1",
          type: "object",
        },
        Member2: {
          additionalProperties: false,
          properties: {
            req: { title: "Req", type: "string" },
            maybe: { anyOf: [{ type: "integer" }, { type: "null" }], default: null, title: "Maybe" },
          },
          required: ["req"],
          title: "Member2",
          type: "object",
        },
      },
      properties: { value: { anyOf: [{ $ref: "#/$defs/Member1" }, { $ref: "#/$defs/Member2" }], title: "Value" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
};

/** Python's `convert_to_openai_tool(...)["function"]` for the schemas the structured-output plan names. */
const FUNCTION_TOOLS: Record<string, { name: string; description: string; parameters: JsonObject }> = {
  "s5:Rec": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, score: { type: "integer" } },
      required: ["answer", "score"],
      type: "object",
    },
  },
  "s5:Bounded": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, score: { maximum: 10n, minimum: 0n, type: "integer" } },
      required: ["answer", "score"],
      type: "object",
    },
  },
  "s5:Open": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, data: { additionalProperties: true, type: "object" } },
      required: ["answer", "data"],
      type: "object",
    },
  },
  "s5:Union": {
    name: "ScalarWrapper",
    description: "",
    parameters: {
      properties: {
        value: {
          anyOf: [
            {
              additionalProperties: false,
              properties: {
                kind: { const: "buy", title: "Kind", type: "string" },
                x: { title: "X", type: "integer" },
              },
              required: ["kind", "x"],
              title: "buy",
              type: "object",
            },
            {
              additionalProperties: false,
              properties: {
                kind: { const: "sell", title: "Kind", type: "string" },
                n: { title: "N", type: "integer" },
              },
              required: ["kind", "n"],
              title: "sell",
              type: "object",
            },
          ],
        },
      },
      required: ["value"],
      type: "object",
    },
  },
  "s5:_ReflectionResponse": {
    name: "_ReflectionResponse",
    description: "Strict structured result from one reflective model call.",
    parameters: {
      additionalProperties: false,
      properties: {
        actions: {
          items: {
            additionalProperties: false,
            description: "One closed model-proposed lesson action.",
            properties: {
              action: { enum: ["add", "revise", "remove", "noop"], type: "string" },
              lesson_id: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
              text: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
            },
            required: ["action"],
            type: "object",
          },
          minItems: 1n,
          type: "array",
        },
      },
      required: ["actions"],
      type: "object",
    },
  },
  "s5:Nested (nullable+list of records, float, date, bool)": {
    name: "Record",
    description: "",
    parameters: {
      properties: {
        inner: {
          anyOf: [
            { properties: { a: { title: "A", type: "integer" } }, required: ["a"], title: "inner", type: "object" },
            { type: "null" },
          ],
          default: null,
        },
        items: {
          items: { properties: { a: { type: "integer" } }, required: ["a"], type: "object" },
          type: "array",
        },
        f: { maximum: 1.0, minimum: 0.5, type: "number" },
        d: { type: "string" },
        b: { type: "boolean" },
      },
      required: ["items", "f", "d", "b"],
      type: "object",
    },
  },
  "s5:Union(Inner, Opt) untagged": {
    name: "ScalarWrapper",
    description: "",
    parameters: {
      properties: {
        value: {
          anyOf: [
            {
              additionalProperties: false,
              properties: { a: { title: "A", type: "integer" } },
              required: ["a"],
              title: "Member1",
              type: "object",
            },
            {
              additionalProperties: false,
              properties: {
                req: { title: "Req", type: "string" },
                maybe: { anyOf: [{ type: "integer" }, { type: "null" }], default: null, title: "Maybe" },
              },
              required: ["req"],
              title: "Member2",
              type: "object",
            },
          ],
        },
      },
      required: ["value"],
      type: "object",
    },
  },
};

/** Python's `_const_as_enum` of `s5:Union`: each `kind` const becomes a one-value enum, appended last. */
const G_UNION: JsonObject = {
  $defs: {
    buy: {
      additionalProperties: false,
      properties: { kind: { title: "Kind", type: "string", enum: ["buy"] }, x: { title: "X", type: "integer" } },
      required: ["kind", "x"],
      title: "buy",
      type: "object",
    },
    sell: {
      additionalProperties: false,
      properties: { kind: { title: "Kind", type: "string", enum: ["sell"] }, n: { title: "N", type: "integer" } },
      required: ["kind", "n"],
      title: "sell",
      type: "object",
    },
  },
  properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
  required: ["value"],
  title: "ScalarWrapper",
  type: "object",
};

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Return the `functionTool` fields a test compares. */
function fields(schema: JsonObject, strict: boolean | null = null): JsonObject {
  const tool = functionTool(schema, strict);
  return { name: tool.name, description: tool.description, parameters: tool.parameters, strict: tool.strict };
}

describe("hasOpenSchema", () => {
  test.each(Object.keys(FIXTURES))("%s", (name) => {
    const { open, schema } = FIXTURES[name]!;
    expect(hasOpenSchema(schema)).toBe(open);
  });

  test.each<[JsonObject, boolean]>([
    [{ type: "object", additionalProperties: true }, true],
    [{ type: "object", properties: {}, additionalProperties: false }, false],
    [{ type: "array", items: {} }, true],
  ])("reads a hand-written schema as given: %j", (schema, expected) => {
    expect(hasOpenSchema(schema)).toBe(expected);
  });

  test("reads only items, additionalProperties, properties, $defs, anyOf, oneOf and allOf", () => {
    expect(hasOpenSchema({})).toBe(true);
    expect(hasOpenSchema({ $defs: { A: { type: "object", additionalProperties: true } } })).toBe(true);
    expect(hasOpenSchema({ oneOf: [{ type: "string" }, {}] })).toBe(true);
    expect(hasOpenSchema({ allOf: [{ items: { additionalProperties: true } }] })).toBe(true);
    expect(hasOpenSchema({ type: "array", prefixItems: [{}] })).toBe(false);
    expect(hasOpenSchema({ type: "string", not: {} })).toBe(false);
    expect(hasOpenSchema({ type: "object", additionalProperties: "true" })).toBe(false);
  });

  test("a non-object has no open schema", () => {
    expect(hasOpenSchema(null)).toBe(false);
    expect(hasOpenSchema("{}")).toBe(false);
    expect(hasOpenSchema([{}])).toBe(false);
  });
});

describe("functionTool", () => {
  test.each(Object.keys(FUNCTION_TOOLS))("%s", (name) => {
    expect(fields(FIXTURES[name]!.schema)).toEqual({ ...FUNCTION_TOOLS[name]!, strict: null });
  });

  test("keeps the root's key order and the dereferenced definition's", () => {
    const tool = functionTool(FIXTURES["s5:Union"]!.schema, null);
    expect(encodeBody(tool.parameters)).toBe(encodeBody(FUNCTION_TOOLS["s5:Union"]!.parameters));
  });

  test("passes strict through", () => {
    expect(functionTool(FIXTURES["s5:Rec"]!.schema, false).strict).toBe(false);
    expect(functionTool(FIXTURES["s5:Rec"]!.schema, true).strict).toBe(true);
    expect(functionTool(FIXTURES["s5:Rec"]!.schema, null).strict).toBeNull();
  });

  test("gives an empty name and description when the root has neither", () => {
    expect(fields({ type: "object", definitions: { A: {} } })).toEqual({
      name: "",
      description: "",
      parameters: { type: "object" },
      strict: null,
    });
  });

  test("keeps a property named title and drops every title keyword", () => {
    const schema: JsonObject = {
      title: "Doc",
      type: "object",
      properties: { title: { title: "Title", type: "string" }, other: { title: "Other", type: "string" } },
      required: ["title"],
    };
    expect(fields(schema).parameters).toEqual({
      type: "object",
      properties: { title: { type: "string" }, other: { type: "string" } },
      required: ["title"],
    });
  });

  test("does not walk array members when it drops titles", () => {
    const schema: JsonObject = { anyOf: [{ title: "A", type: "string" }], items: { title: "I", type: "string" } };
    expect(fields(schema).parameters).toEqual({ anyOf: [{ title: "A", type: "string" }], items: { type: "string" } });
  });

  test("merges a mixed ref's siblings over the definition", () => {
    const schema: JsonObject = {
      properties: { a: { $ref: "#/$defs/A", description: "d", type: "integer" } },
      $defs: { A: { type: "string", minLength: 1n } },
    };
    const parameters = fields(schema).parameters as JsonObject;
    const a = (parameters["properties"] as JsonObject)["a"] as JsonObject;
    expect(Object.keys(a)).toEqual(["type", "minLength", "description"]);
    expect(a).toEqual({ type: "integer", minLength: 1n, description: "d" });
  });

  test("resolves a ref against the root, a digit indexing an array", () => {
    const schema: JsonObject = {
      properties: { a: { $ref: "#/$defs/L/1" }, b: { $ref: "#/$defs/B" } },
      $defs: { L: [{ type: "string" }, { type: "integer" }], B: { items: { $ref: "#/$defs/L/0" } } },
    };
    expect(fields(schema).parameters).toEqual({
      properties: { a: { type: "integer" }, b: { items: { type: "string" } } },
    });
  });

  test("a cyclic ref gives only its siblings", () => {
    const schema: JsonObject = {
      title: "Node",
      properties: { next: { $ref: "#/$defs/Node" } },
      $defs: {
        Node: {
          type: "object",
          properties: { next: { $ref: "#/$defs/Node", description: "again" }, loop: { $ref: "#/$defs/Node" } },
        },
      },
    };
    expect(fields(schema).parameters).toEqual({
      properties: { next: { type: "object", properties: { next: { description: "again" }, loop: {} } } },
    });
  });

  test("refuses a ref that is not a URI fragment", () => {
    const error = thrown(() => functionTool({ properties: { a: { $ref: "defs/A" } } }, null));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("ref paths are expected to be URI fragments, meaning they should start with #.");
  });

  test("refuses a ref that is not in the schema", () => {
    const schema: JsonObject = { properties: { a: { $ref: "#/$defs/B" } }, $defs: { A: {} } };
    const error = thrown(() => functionTool(schema, null));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("Reference '#/$defs/B' not found.");
    const list: JsonObject = { properties: { a: { $ref: "#/$defs/L/2" } }, $defs: { L: [{}, {}] } };
    expect(thrown(() => functionTool(list, null)).message).toBe("Reference '#/$defs/L/2' not found.");
  });
});

describe("constAsEnum", () => {
  // Python's `_const_as_enum` table; each row is named by its encoded input (`%j` can't write a `bigint`).
  const rows: [JsonObject, JsonObject][] = [
    [{ const: "a", enum: ["a", "b"] }, { enum: ["a"] }],
    [{ const: "a", enum: ["b"] }, { const: "a", enum: ["b"] }],
    [{ const: 1n, enum: [true, "b"] }, { const: 1n, enum: [true, "b"] }],
    [{ const: true, enum: [1n] }, { const: true, enum: [1n] }],
    [{ const: 3n, type: "integer" }, { enum: [3n], type: "integer" }],
    [{ const: 0.5 }, { enum: [0.5] }],
    [{ const: true }, { const: true }],
    [{ const: null }, { const: null }],
    [{ type: "array", items: { const: "a" } }, { type: "array", items: { enum: ["a"] } }],
    [{ prefixItems: [{ const: "a" }] }, { prefixItems: [{ enum: ["a"] }] }],
    [{ additionalProperties: { const: 1n } }, { additionalProperties: { enum: [1n] } }],
    [
      { allOf: [{ const: "a" }], oneOf: [{ const: "b" }] },
      { allOf: [{ enum: ["a"] }], oneOf: [{ enum: ["b"] }] },
    ],
    [{ anyOf: [{ const: "a" }, { type: "null" }] }, { anyOf: [{ enum: ["a"] }, { type: "null" }] }],
    [{ $defs: { A: { const: "a" } } }, { $defs: { A: { enum: ["a"] } } }],
    [{ properties: { const: { const: 1n } } }, { properties: { const: { enum: [1n] } } }],
    [{ default: { const: 1n } }, { default: { const: 1n } }],
  ];

  test.each(rows.map(([schema, sent]) => [encodeBody(schema), schema, sent] as const))("%s", (before, schema, sent) => {
    expect(constAsEnum(schema)).toEqual(sent);
    expect(encodeBody(schema)).toBe(before);
  });

  test("turns the tagged union's consts into enums, appended last", () => {
    const result = constAsEnum(FIXTURES["s5:Union"]!.schema);
    expect(encodeBody(result)).toBe(encodeBody(G_UNION));
  });

  test("compares a const with the enum's members as JSON values", () => {
    expect(constAsEnum({ const: 0.5, enum: [0.5] })).toEqual({ enum: [0.5] });
    expect(constAsEnum({ const: 1n, enum: [1.5] })).toEqual({ const: 1n, enum: [1.5] });
    expect(constAsEnum({ const: 1n, enum: [1] })).toEqual({ const: 1n, enum: [1] });
    expect(constAsEnum({ const: 2n ** 60n })).toEqual({ enum: [2n ** 60n] });
    expect(constAsEnum({ const: 2n ** 60n, enum: [2n ** 60n, 1n] })).toEqual({ enum: [2n ** 60n] });
  });
});

describe("every transform", () => {
  const transforms: [string, (schema: JsonObject) => unknown][] = [
    ["hasOpenSchema", hasOpenSchema],
    ["functionTool", (schema) => functionTool(schema, null)],
    ["constAsEnum", constAsEnum],
  ];
  const cases = transforms.flatMap(([label, run]) =>
    Object.keys(FIXTURES).map((name): [string, string, (schema: JsonObject) => unknown] => [label, name, run]),
  );

  test.each(cases)("%s leaves %s unchanged and gives the same result twice", (_label, name, run) => {
    const schema = FIXTURES[name]!.schema;
    const before = encodeBody(schema);
    const first = run(schema);
    expect(encodeBody(schema)).toBe(before);
    expect(run(schema)).toEqual(first);
  });
});

