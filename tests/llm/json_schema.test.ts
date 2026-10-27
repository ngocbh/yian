/**
 * Tests for the structured-output schema transforms in `src/llm/json_schema.ts`: `hasOpenSchema`,
 * `functionTool`, `constAsEnum`, `strictJsonSchema`, `moveAzureKeywords` and
 * `anthropicOutputSchema`, checked against recorded results (LangChain's, openai's and
 * anthropic's among them) for the schemas `typeToSchema` emits, and on hand-written schemas.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  AZURE_STRICT_UNSUPPORTED,
  anthropicOutputSchema,
  constAsEnum,
  functionTool,
  hasOpenSchema,
  moveAzureKeywords,
  strictJsonSchema,
} from "../../src/llm/json_schema.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

/** A schema `typeToSchema` emits, and whether it is open. */
interface Fixture {
  open: boolean;
  schema: JsonObject;
}

/**
 * The schemas for the open-schema test's types (`open_schema_test:*`), the Azure keyword test's
 * types (`azure_test:*`) and the record, bounded, open, union, reflection and other schemas
 * `typeToSchema` builds (`corpus:*`): ints are `bigint`s, floats numbers.
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
  "azure_test:int(0..10)": {
    open: false,
    schema: {
      properties: { value: { maximum: 10n, minimum: 0n, title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "azure_test:list(int(1..5))": {
    open: false,
    schema: {
      properties: {
        items: { items: { maximum: 5n, minimum: 1n, type: "integer" }, title: "Items", type: "array" },
      },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "azure_test:Rec": {
    open: false,
    schema: {
      properties: {
        n: { maximum: 5n, minimum: 0n, title: "N", type: "integer" },
        s: { title: "S", type: "string" },
      },
      required: ["n", "s"],
      title: "Record",
      type: "object",
    },
  },
  "azure_test:CU": {
    open: false,
    schema: {
      $defs: {
        c2: {
          additionalProperties: false,
          properties: {
            kind: { const: "c2", title: "Kind", type: "string" },
            a: { maximum: 4n, minimum: 0n, title: "A", type: "integer" },
          },
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
  "azure_test:Optional(int(0..3))": {
    open: false,
    schema: {
      properties: {
        value: { anyOf: [{ maximum: 3n, minimum: 0n, type: "integer" }, { type: "null" }], title: "Value" },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "azure_test:float(0.0..1.0)": {
    open: false,
    schema: {
      properties: { value: { maximum: 1.0, minimum: 0.0, title: "Value", type: "number" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "azure_test:list(Rec)": {
    open: false,
    schema: {
      $defs: {
        item: {
          properties: {
            n: { maximum: 5n, minimum: 0n, title: "N", type: "integer" },
            s: { title: "S", type: "string" },
          },
          required: ["n", "s"],
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
  "azure_test:int(0..)": {
    open: false,
    schema: {
      properties: { value: { minimum: 0n, title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "azure_test:M": {
    open: false,
    schema: {
      properties: { minimum: { maximum: 3n, minimum: 0n, title: "Minimum", type: "integer" } },
      required: ["minimum"],
      title: "Record",
      type: "object",
    },
  },
  "azure_test:C": {
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
  "azure_test:int": {
    open: false,
    schema: {
      properties: { value: { title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "azure_test:OpenRec": {
    open: true,
    schema: {
      properties: {
        n: { maximum: 5n, minimum: 0n, title: "N", type: "integer" },
        m: { additionalProperties: true, title: "M", type: "object" },
      },
      required: ["n", "m"],
      title: "Record",
      type: "object",
    },
  },
  "azure_test:_Described": {
    open: false,
    schema: {
      properties: { count: { description: "How many.", minimum: 1n, title: "Count", type: "integer" } },
      required: ["count"],
      title: "_Described",
      type: "object",
    },
  },
  "corpus:Rec": {
    open: false,
    schema: {
      properties: { answer: { title: "Answer", type: "string" }, score: { title: "Score", type: "integer" } },
      required: ["answer", "score"],
      title: "Record",
      type: "object",
    },
  },
  "corpus:Bounded": {
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
  "corpus:Open": {
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
  "corpus:Union": {
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
  "corpus:_ReflectionResponse": {
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
  "corpus:int(0..10)": {
    open: false,
    schema: {
      properties: { value: { maximum: 10n, minimum: 0n, title: "Value", type: "integer" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "corpus:Nested (nullable+list of records, float, date, bool)": {
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
  "corpus:Opt (optional field)": {
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
  "corpus:list(int)": {
    open: false,
    schema: {
      properties: { items: { items: { type: "integer" }, title: "Items", type: "array" } },
      required: ["items"],
      title: "ListWrapper",
      type: "object",
    },
  },
  "corpus:Any": {
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
  "corpus:object": {
    open: true,
    schema: {
      properties: { value: { additionalProperties: true, title: "Value", type: "object" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    },
  },
  "corpus:list(Any)": {
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
  "corpus:Union(Inner, Opt) untagged": {
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

/** The OpenAI function tool (`functionTool`) for each `corpus:*` fixture. */
const FUNCTION_TOOLS: Record<string, { name: string; description: string; parameters: JsonObject }> = {
  "corpus:Rec": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, score: { type: "integer" } },
      required: ["answer", "score"],
      type: "object",
    },
  },
  "corpus:Bounded": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, score: { maximum: 10n, minimum: 0n, type: "integer" } },
      required: ["answer", "score"],
      type: "object",
    },
  },
  "corpus:Open": {
    name: "Record",
    description: "",
    parameters: {
      properties: { answer: { type: "string" }, data: { additionalProperties: true, type: "object" } },
      required: ["answer", "data"],
      type: "object",
    },
  },
  "corpus:Union": {
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
  "corpus:_ReflectionResponse": {
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
  "corpus:Nested (nullable+list of records, float, date, bool)": {
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
  "corpus:Union(Inner, Opt) untagged": {
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

/** `constAsEnum` of `corpus:Union`: each `kind` const becomes a one-value enum, appended last. */
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

/** openai's `to_strict_json_schema` of the `corpus:*` fixtures. */
const STRICT: Record<string, JsonObject> = {
  "corpus:Rec": {
    properties: { answer: { title: "Answer", type: "string" }, score: { title: "Score", type: "integer" } },
    required: ["answer", "score"],
    title: "Record",
    type: "object",
    additionalProperties: false,
  },
  "corpus:Bounded": {
    properties: {
      answer: { title: "Answer", type: "string" },
      score: { maximum: 10n, minimum: 0n, title: "Score", type: "integer" },
    },
    required: ["answer", "score"],
    title: "Record",
    type: "object",
    additionalProperties: false,
  },
  "corpus:Union": {
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
        properties: { kind: { const: "sell", title: "Kind", type: "string" }, n: { title: "N", type: "integer" } },
        required: ["kind", "n"],
        title: "sell",
        type: "object",
      },
    },
    properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
    additionalProperties: false,
  },
  "corpus:_ReflectionResponse": {
    $defs: {
      _ReflectionAction: {
        additionalProperties: false,
        description: "One closed model-proposed lesson action.",
        properties: {
          action: { enum: ["add", "revise", "remove", "noop"], title: "Action", type: "string" },
          lesson_id: { anyOf: [{ type: "string" }, { type: "null" }], title: "Lesson Id" },
          text: { anyOf: [{ type: "string" }, { type: "null" }], title: "Text" },
        },
        required: ["action", "lesson_id", "text"],
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
};

/**
 * yian's `_strict_schema_without` (Azure's rejected keywords moved into descriptions) of every Azure
 * keyword test type and the reflection response, with notes spelled as JSON (float bounds as JS
 * numbers); `null` when nothing moves.
 */
const MOVED: Record<string, JsonObject | null> = {
  "azure_test:int(0..10)": {
    properties: { value: { title: "Value", type: "integer", description: "{minimum: 0, maximum: 10}" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:list(int(1..5))": {
    properties: {
      items: { items: { type: "integer", description: "{minimum: 1, maximum: 5}" }, title: "Items", type: "array" },
    },
    required: ["items"],
    title: "ListWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:Rec": {
    properties: {
      n: { title: "N", type: "integer", description: "{minimum: 0, maximum: 5}" },
      s: { title: "S", type: "string" },
    },
    required: ["n", "s"],
    title: "Record",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:CU": {
    $defs: {
      c2: {
        additionalProperties: false,
        properties: {
          kind: { const: "c2", title: "Kind", type: "string" },
          a: { title: "A", type: "integer", description: "{minimum: 0, maximum: 4}" },
        },
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
    additionalProperties: false,
  },
  "azure_test:Optional(int(0..3))": {
    properties: {
      value: {
        anyOf: [{ type: "integer", description: "{minimum: 0, maximum: 3}" }, { type: "null" }],
        title: "Value",
      },
    },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:float(0.0..1.0)": {
    properties: { value: { title: "Value", type: "number", description: "{minimum: 0, maximum: 1}" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:list(Rec)": {
    $defs: {
      item: {
        properties: {
          n: { title: "N", type: "integer", description: "{minimum: 0, maximum: 5}" },
          s: { title: "S", type: "string" },
        },
        required: ["n", "s"],
        title: "item",
        type: "object",
        additionalProperties: false,
      },
    },
    properties: { items: { items: { $ref: "#/$defs/item" }, title: "Items", type: "array" } },
    required: ["items"],
    title: "ListWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:int(0..)": {
    properties: { value: { title: "Value", type: "integer", description: "{minimum: 0}" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:M": {
    properties: { minimum: { title: "Minimum", type: "integer", description: "{minimum: 0, maximum: 3}" } },
    required: ["minimum"],
    title: "Record",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:C": null,
  "azure_test:int": null,
  "azure_test:OpenRec": {
    properties: {
      n: { title: "N", type: "integer", description: "{minimum: 0, maximum: 5}" },
      m: { additionalProperties: true, title: "M", type: "object" },
    },
    required: ["n", "m"],
    title: "Record",
    type: "object",
    additionalProperties: false,
  },
  "azure_test:_Described": {
    properties: { count: { description: "How many.\n\n{minimum: 1}", title: "Count", type: "integer" } },
    required: ["count"],
    title: "_Described",
    type: "object",
    additionalProperties: false,
  },
  "corpus:_ReflectionResponse": {
    $defs: {
      _ReflectionAction: {
        additionalProperties: false,
        description: "One closed model-proposed lesson action.",
        properties: {
          action: { enum: ["add", "revise", "remove", "noop"], title: "Action", type: "string" },
          lesson_id: { anyOf: [{ type: "string" }, { type: "null" }], title: "Lesson Id" },
          text: { anyOf: [{ type: "string" }, { type: "null" }], title: "Text" },
        },
        required: ["action", "lesson_id", "text"],
        title: "_ReflectionAction",
        type: "object",
      },
    },
    additionalProperties: false,
    description: "Strict structured result from one reflective model call.",
    properties: {
      actions: {
        items: { $ref: "#/$defs/_ReflectionAction" },
        title: "Actions",
        type: "array",
        description: "{minItems: 1}",
      },
    },
    required: ["actions"],
    title: "_ReflectionResponse",
    type: "object",
  },
};

/** anthropic's `transform_schema` of the `corpus:*` fixtures, with notes spelled as JSON. */
const ANTHROPIC: Record<string, JsonObject> = {
  "corpus:Rec": {
    type: "object",
    title: "Record",
    properties: { answer: { type: "string", title: "Answer" }, score: { type: "integer", title: "Score" } },
    additionalProperties: false,
    required: ["answer", "score"],
  },
  "corpus:Bounded": {
    type: "object",
    title: "Record",
    properties: {
      answer: { type: "string", title: "Answer" },
      score: { type: "integer", title: "Score", description: "{maximum: 10, minimum: 0}" },
    },
    additionalProperties: false,
    required: ["answer", "score"],
  },
  "corpus:Union": {
    $defs: {
      buy: {
        type: "object",
        title: "buy",
        properties: {
          kind: { type: "string", title: "Kind", description: "{const: \"buy\"}" },
          x: { type: "integer", title: "X" },
        },
        additionalProperties: false,
        required: ["kind", "x"],
      },
      sell: {
        type: "object",
        title: "sell",
        properties: {
          kind: { type: "string", title: "Kind", description: "{const: \"sell\"}" },
          n: { type: "integer", title: "N" },
        },
        additionalProperties: false,
        required: ["kind", "n"],
      },
    },
    type: "object",
    title: "ScalarWrapper",
    properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
    additionalProperties: false,
    required: ["value"],
  },
  "corpus:_ReflectionResponse": {
    $defs: {
      _ReflectionAction: {
        type: "object",
        description: "One closed model-proposed lesson action.",
        title: "_ReflectionAction",
        properties: {
          action: { type: "string", enum: ["add", "revise", "remove", "noop"], title: "Action" },
          lesson_id: {
            anyOf: [{ type: "string" }, { type: "null" }],
            title: "Lesson Id",
            description: "{default: null}",
          },
          text: { anyOf: [{ type: "string" }, { type: "null" }], title: "Text", description: "{default: null}" },
        },
        additionalProperties: false,
        required: ["action"],
      },
    },
    type: "object",
    description: "Strict structured result from one reflective model call.",
    title: "_ReflectionResponse",
    properties: {
      actions: { type: "array", title: "Actions", items: { $ref: "#/$defs/_ReflectionAction" }, minItems: 1n },
    },
    additionalProperties: false,
    required: ["actions"],
  },
  "corpus:object": {
    type: "object",
    title: "ScalarWrapper",
    properties: { value: { type: "object", title: "Value", properties: {}, additionalProperties: false } },
    additionalProperties: false,
    required: ["value"],
  },
  "corpus:Open": {
    type: "object",
    title: "Record",
    properties: {
      answer: { type: "string", title: "Answer" },
      data: { type: "object", title: "Data", properties: {}, additionalProperties: false },
    },
    additionalProperties: false,
    required: ["answer", "data"],
  },
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

/** Assert `got` equals `want` as JSON values, ints and floats apart, with the same key order. */
function expectSame(got: unknown, want: unknown): void {
  expect(got).toEqual(want);
  expect(encodeBody(got)).toBe(encodeBody(want));
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
    const tool = functionTool(FIXTURES["corpus:Union"]!.schema, null);
    expect(encodeBody(tool.parameters)).toBe(encodeBody(FUNCTION_TOOLS["corpus:Union"]!.parameters));
  });

  test("passes strict through", () => {
    expect(functionTool(FIXTURES["corpus:Rec"]!.schema, false).strict).toBe(false);
    expect(functionTool(FIXTURES["corpus:Rec"]!.schema, true).strict).toBe(true);
    expect(functionTool(FIXTURES["corpus:Rec"]!.schema, null).strict).toBeNull();
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
    for (const ref of ["#foo", 5n]) {
      const other = thrown(() => functionTool({ properties: { a: { $ref: ref } } }, null));
      expect(other).toBeInstanceOf(TypeError);
      expect(other.message).toBe(error.message);
    }
  });

  test("copies an unused definition without resolving its refs", () => {
    const schema: JsonObject = { properties: { a: { type: "string" } }, $defs: { A: { $ref: "#/$defs/Gone" } } };
    expect(fields(schema).parameters).toEqual({ properties: { a: { type: "string" } } });
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
  // Each row is named by its encoded input (`%j` can't write a `bigint`).
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
    const result = constAsEnum(FIXTURES["corpus:Union"]!.schema);
    expect(encodeBody(result)).toBe(encodeBody(G_UNION));
  });

  test("compares a const with the enum's members as JSON values", () => {
    expect(constAsEnum({ const: 0.5, enum: [0.5] })).toEqual({ enum: [0.5] });
    expect(constAsEnum({ const: 1n, enum: [1.5] })).toEqual({ const: 1n, enum: [1.5] });
    expect(constAsEnum({ const: 1n, enum: [1] })).toEqual({ const: 1n, enum: [1] });
    expect(constAsEnum({ const: 2n ** 60n })).toEqual({ enum: [2n ** 60n] });
    expect(constAsEnum({ const: 2n ** 60n, enum: [2n ** 60n, 1n] })).toEqual({ enum: [2n ** 60n] });
  });

  test("reads an enum of null as no enum", () => {
    expectSame(constAsEnum({ const: "a", enum: null }), { enum: ["a"] });
  });
});

describe("strictJsonSchema", () => {
  test.each(Object.keys(STRICT))("%s", (name) => {
    expectSame(strictJsonSchema(FIXTURES[name]!.schema), STRICT[name]);
  });

  test("closes the root, keeps the closed members, their $defs and the bare refs", () => {
    const union = FIXTURES["corpus:Union"]!.schema;
    expectSame(strictJsonSchema(union), { ...union, additionalProperties: false });
  });

  test("drops null defaults and requires every property", () => {
    const strict = strictJsonSchema(FIXTURES["corpus:_ReflectionResponse"]!.schema);
    const action = (strict["$defs"] as JsonObject)["_ReflectionAction"] as JsonObject;
    expect(action["required"]).toEqual(["action", "lesson_id", "text"]);
    expect(encodeBody(action)).not.toContain("default");
  });

  test("keeps an open object open and a non-null default", () => {
    const schema: JsonObject = {
      type: "object",
      additionalProperties: true,
      properties: { a: { type: "string", default: "x" } },
    };
    expectSame(strictJsonSchema(schema), { ...schema, required: ["a"] });
  });

  test("walks definitions, items and anyOf members", () => {
    const schema: JsonObject = {
      definitions: { A: { type: "object" } },
      items: { type: "object" },
      anyOf: [{ type: "object" }],
    };
    const closed = { type: "object", additionalProperties: false };
    expectSame(strictJsonSchema(schema), { definitions: { A: closed }, items: closed, anyOf: [closed] });
  });

  test("resolves a mixed ref, the node's keys first, and walks the result", () => {
    const schema: JsonObject = {
      properties: { a: { $ref: "#/$defs/A", description: "d" } },
      $defs: { A: { type: "object", properties: { x: { type: "integer", default: null } } } },
    };
    const a = { type: "object", properties: { x: { type: "integer" } }, additionalProperties: false, required: ["x"] };
    expectSame(strictJsonSchema(schema), {
      properties: { a: { description: "d", ...a } },
      $defs: { A: a },
      required: ["a"],
    });
  });

  test("merges a one-entry allOf into the node and walks several entries", () => {
    const one: JsonObject = { allOf: [{ type: "object", properties: { a: { type: "string" } } }], description: "d" };
    expectSame(strictJsonSchema(one), {
      description: "d",
      type: "object",
      properties: { a: { type: "string" } },
      additionalProperties: false,
      required: ["a"],
    });
    const two: JsonObject = { allOf: [{ type: "object" }, { properties: { a: { default: null } } }] };
    expectSame(strictJsonSchema(two), {
      allOf: [{ type: "object", additionalProperties: false }, { properties: { a: {} }, required: ["a"] }],
    });
  });

  test("refuses a mixed ref that is not a #/ path", () => {
    for (const [ref, text] of [
      ["defs/A", '"defs/A"'],
      ["#A", '"#A"'],
      [5n, "5"],
    ] as const) {
      const error = thrown(() => strictJsonSchema({ properties: { a: { $ref: ref, description: "d" } } }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(`Unexpected $ref format ${text}; Does not start with #/`);
    }
  });

  test("refuses a mixed ref that names nothing", () => {
    const schema: JsonObject = { properties: { a: { $ref: "#/$defs/B", description: "d" } }, $defs: { A: {} } };
    const error = thrown(() => strictJsonSchema(schema));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("Reference '#/$defs/B' not found.");
  });

  test("refuses a non-object where a schema is expected", () => {
    for (const [schema, path] of [
      [{ properties: { a: true } }, "#/properties/a"],
      [{ anyOf: ["x"] }, "#/anyOf/0"],
      [{ $defs: { A: null } }, "#/$defs/A"],
      [[] as unknown as JsonObject, "#"],
    ] as const) {
      const error = thrown(() => strictJsonSchema(schema as JsonObject));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(`expected a JSON Schema object at ${path}`);
    }
  });
});

describe("moveAzureKeywords", () => {
  test("lists Azure's rejected keywords in a fixed order", () => {
    expect(AZURE_STRICT_UNSUPPORTED).toEqual([
      "minLength",
      "maxLength",
      "pattern",
      "format",
      "minimum",
      "maximum",
      "multipleOf",
      "patternProperties",
      "unevaluatedProperties",
      "propertyNames",
      "minProperties",
      "maxProperties",
      "unevaluatedItems",
      "contains",
      "minContains",
      "maxContains",
      "minItems",
      "maxItems",
      "uniqueItems",
    ]);
    expect(Object.isFrozen(AZURE_STRICT_UNSUPPORTED)).toBe(true);
  });

  test.each(Object.keys(MOVED))("%s", (name) => {
    expectSame(moveAzureKeywords(strictJsonSchema(FIXTURES[name]!.schema)), MOVED[name]);
  });

  /** Return the moved strict schema of fixture `name`. */
  const moved = (name: string) => moveAzureKeywords(strictJsonSchema(FIXTURES[name]!.schema))!;
  /** Return the `properties` of the moved strict schema of fixture `name`. */
  const properties = (name: string) => moved(name)["properties"] as JsonObject;

  test("a bounded int moves its bounds into the description", () => {
    expectSame(moved("azure_test:int(0..10)"), {
      properties: { value: { title: "Value", type: "integer", description: "{minimum: 0, maximum: 10}" } },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
      additionalProperties: false,
    });
  });

  test("a bound on an anyOf member moves on that member", () => {
    expectSame(properties("azure_test:Optional(int(0..3))")["value"], {
      anyOf: [{ type: "integer", description: "{minimum: 0, maximum: 3}" }, { type: "null" }],
      title: "Value",
    });
  });

  test("float bounds are spelled as JS numbers", () => {
    expect((properties("azure_test:float(0.0..1.0)")["value"] as JsonObject)["description"]).toBe(
      "{minimum: 0, maximum: 1}",
    );
  });

  test("an open end gives only the given bound", () => {
    expectSame(properties("azure_test:int(0..)")["value"], {
      title: "Value",
      type: "integer",
      description: "{minimum: 0}",
    });
  });

  test("an existing description is kept before the note", () => {
    expect((properties("azure_test:_Described")["count"] as JsonObject)["description"]).toBe(
      "How many.\n\n{minimum: 1}",
    );
  });

  test("a field named like a keyword is kept", () => {
    expectSame(properties("azure_test:M"), {
      minimum: { title: "Minimum", type: "integer", description: "{minimum: 0, maximum: 3}" },
    });
  });

  test("moves keywords inside items and inside $defs", () => {
    expectSame((properties("azure_test:list(int(1..5))")["items"] as JsonObject)["items"], {
      type: "integer",
      description: "{minimum: 1, maximum: 5}",
    });
    for (const [name, def, key, note] of [
      ["azure_test:CU", "c2", "a", "{minimum: 0, maximum: 4}"],
      ["azure_test:list(Rec)", "item", "n", "{minimum: 0, maximum: 5}"],
    ] as const) {
      const defs = moved(name)["$defs"] as JsonObject;
      const field = ((defs[def] as JsonObject)["properties"] as JsonObject)[key] as JsonObject;
      expect(field["description"]).toBe(note);
    }
  });

  test("the reflection response's minItems moves into the description", () => {
    const actions = properties("corpus:_ReflectionResponse")["actions"] as JsonObject;
    expect(actions["minItems"]).toBeUndefined();
    expect(actions["description"]).toBe("{minItems: 1}");
  });

  test("closed unbounded schemas need no change", () => {
    expect(moveAzureKeywords(strictJsonSchema(FIXTURES["azure_test:C"]!.schema))).toBeNull();
    expect(moveAzureKeywords(strictJsonSchema(FIXTURES["azure_test:int"]!.schema))).toBeNull();
  });

  test("an empty or non-string description counts as none", () => {
    expectSame(moveAzureKeywords({ type: "string", description: "", minLength: 1n }), {
      type: "string",
      description: "{minLength: 1}",
    });
    expectSame(moveAzureKeywords({ type: "string", description: null, minLength: 1n }), {
      type: "string",
      description: "{minLength: 1}",
    });
    expectSame(moveAzureKeywords({ type: "string", description: 5n, minLength: 1n }), {
      type: "string",
      description: "{minLength: 1}",
    });
  });

  test("notes keywords in the list's order, values as JSON", () => {
    const schema: JsonObject = { type: "array", maxItems: 5n, minItems: 1n, uniqueItems: true, pattern: "^a" };
    expectSame(moveAzureKeywords(schema), {
      type: "array",
      description: '{pattern: "^a", minItems: 1, maxItems: 5, uniqueItems: true}',
    });
  });

  test("walks prefixItems, oneOf and allOf members and an additionalProperties object only", () => {
    const schema: JsonObject = {
      prefixItems: [{ minimum: 1n }],
      oneOf: [{ maximum: 2n }],
      allOf: [{ format: "date" }],
      additionalProperties: { minLength: 3n },
    };
    expectSame(moveAzureKeywords(schema), {
      prefixItems: [{ description: "{minimum: 1}" }],
      oneOf: [{ description: "{maximum: 2}" }],
      allOf: [{ description: '{format: "date"}' }],
      additionalProperties: { description: "{minLength: 3}" },
    });
    expect(moveAzureKeywords({ not: { minimum: 1n }, default: { minimum: 1n } })).toBeNull();
  });
});

describe("anthropicOutputSchema", () => {
  test.each(Object.keys(ANTHROPIC))("%s", (name) => {
    expectSame(anthropicOutputSchema(FIXTURES[name]!.schema), ANTHROPIC[name]);
  });

  const noType = "Schema must have a 'type', 'anyOf', 'oneOf', or 'allOf' field.";

  test.each(["corpus:Any", "corpus:list(Any)", "open_schema_test:D", "open_schema_test:elementless list"])(
    "%s has a type-less subschema",
    (name) => {
      const error = thrown(() => anthropicOutputSchema(FIXTURES[name]!.schema));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(noType);
    },
  );

  test("refuses a type-less schema", () => {
    const error = thrown(() => anthropicOutputSchema({}));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe(noType);
  });

  test("refuses a non-object where a schema is expected", () => {
    const error = thrown(() => anthropicOutputSchema({ type: "array", items: true }));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("expected a JSON Schema object at #/items");
  });

  test("refuses an unknown type", () => {
    for (const [type, text] of [
      ["date", '"date"'],
      [["string", "null"], '["string","null"]'],
    ] as const) {
      const error = thrown(() => anthropicOutputSchema({ type: type as JsonObject["type"] }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(`unsupported JSON Schema type ${text}`);
    }
  });

  test("writes oneOf as anyOf and drops a non-array enum", () => {
    expectSame(anthropicOutputSchema({ oneOf: [{ type: "string" }, { type: "integer" }] }), {
      anyOf: [{ type: "string" }, { type: "integer" }],
    });
    expectSame(anthropicOutputSchema({ type: "string", enum: "a" }), { type: "string" });
  });

  test("keeps a ref and the root's $defs, dropping the ref's siblings", () => {
    const schema: JsonObject = { $ref: "#/$defs/A", description: "d", $defs: { A: { type: "string" } } };
    expectSame(anthropicOutputSchema(schema), { $defs: { A: { type: "string" } }, $ref: "#/$defs/A" });
  });

  test("keeps a supported string format, notes another and drops a falsy one", () => {
    expectSame(anthropicOutputSchema({ type: "string", format: "date" }), { type: "string", format: "date" });
    expectSame(anthropicOutputSchema({ type: "string", format: "regex", description: "d" }), {
      type: "string",
      description: 'd\n\n{format: "regex"}',
    });
    for (const format of ["", null, false, 0n, 0, [], {}] as JsonValue[]) {
      expectSame(anthropicOutputSchema({ type: "string", format }), { type: "string" });
    }
    expectSame(anthropicOutputSchema({ type: "string", format: 1n }), { type: "string", description: "{format: 1}" });
  });

  test("keeps a minItems of 0 or 1 and notes another, re-added last", () => {
    expectSame(anthropicOutputSchema({ type: "array", minItems: 1n }), { type: "array", minItems: 1n });
    expectSame(anthropicOutputSchema({ type: "array", minItems: 1 }), { type: "array", minItems: 1 });
    expectSame(anthropicOutputSchema({ type: "array", minItems: 0n, items: { type: "string" } }), {
      type: "array",
      items: { type: "string" },
      minItems: 0n,
    });
    expectSame(anthropicOutputSchema({ type: "array", minItems: 2n, maxItems: 5n }), {
      type: "array",
      description: "{maxItems: 5, minItems: 2}",
    });
    expectSame(anthropicOutputSchema({ type: "array", minItems: "1" }), {
      type: "array",
      description: '{minItems: "1"}',
    });
  });

  test("notes unknown keywords as JSON, after an existing description", () => {
    expectSame(anthropicOutputSchema({ type: "string", minLength: 1n, "x-foo": [1n, "a"] }), {
      type: "string",
      description: '{minLength: 1, x-foo: [1,"a"]}',
    });
    expectSame(anthropicOutputSchema({ type: "integer", description: "", minimum: 1n }), {
      type: "integer",
      description: "\n\n{minimum: 1}",
    });
    expectSame(anthropicOutputSchema({ type: "integer", description: 5n, minimum: 1n }), {
      type: "integer",
      description: "{minimum: 1}",
    });
  });
});

describe("every transform", () => {
  const transforms: [string, (schema: JsonObject) => unknown][] = [
    ["hasOpenSchema", hasOpenSchema],
    ["functionTool", (schema) => functionTool(schema, null)],
    ["constAsEnum", constAsEnum],
    ["strictJsonSchema", strictJsonSchema],
    ["moveAzureKeywords", moveAzureKeywords],
    ["anthropicOutputSchema", anthropicOutputSchema],
  ];
  /** Return what `run` gives for `schema`: its result, or the message it throws. */
  const outcome = (run: (schema: JsonObject) => unknown, schema: JsonObject) => {
    try {
      return { result: run(schema) };
    } catch (error) {
      return { error: (error as Error).message };
    }
  };
  const cases = transforms.flatMap(([label, run]) =>
    Object.keys(FIXTURES).map((name): [string, string, (schema: JsonObject) => unknown] => [label, name, run]),
  );

  test.each(cases)("%s leaves %s unchanged and gives the same result (or error) twice", (_label, name, run) => {
    const schema = FIXTURES[name]!.schema;
    const before = encodeBody(schema);
    const first = outcome(run, schema);
    expect(encodeBody(schema)).toBe(before);
    expect(outcome(run, schema)).toEqual(first);
  });
});

