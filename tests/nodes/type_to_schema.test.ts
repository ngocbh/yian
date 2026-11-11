/**
 * `typeToSchema`: the JSON Schema an AGENT asks the model to fill for its declared `output:`, and
 * `checkReply`, the check of a reply against it (also through `generateStructured` and `runFlow`).
 *
 * Each schema is pinned exactly against a captured JSON text read with `parseJson`, so its ints
 * are `bigint` as in the built schema, and a float bound spelled as an int (`float(-0.5..1)`'s
 * `1`) is a `bigint` on both sides. Keys are sorted at every level, a `properties` map's own
 * schema too; in `properties`, integer-like field names come first (JS key order), then the others
 * in field order; fields starting with `_` are kept, `__proto__` too; a `.` in a `$defs` key
 * becomes `_`, and a clashing key is `<key>__<n>`. A root `Optional(str)` gives `null`.
 *
 * An int in a `float` slot stays an int; the value check's error paths are relative to the output
 * (`[0, "scores", 0, 1]`); an open record drops unknown fields and comes back in declared field
 * order.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";

import { compareCodeUnits } from "../../src/diagnostics.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  checkReply,
  generateStructured,
  OutputInvalidError,
  typeToSchema,
} from "../../src/nodes/agent/structured.ts";
import { questionListType } from "../../src/nodes/tool/questions.ts";
import { runFlow } from "../../src/run.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  isListKind,
  Type,
  TypeCheckError,
  ValueKind,
} from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { serving } from "../helpers/run_state_fixtures.ts";

const NO_TYPEDEFS = readTypedefs({});
const POINT = readTypedefs({ Point: { x: "int" } });
const OPEN = readTypedefs({
  R: { a: "int", m: "object" },
  Buy: { kind: "Literal(buy)", x: "Any" },
  Sell: { kind: "Literal(sell)", n: "int" },
  D: "Union(Buy, Sell)",
  Outer: { inner: "R", n: "int" },
  C: { a: "int", b: "list(str)" },
  C2: { kind: "Literal(c2)", a: "int" },
  C3: { kind: "Literal(c3)", b: "str" },
  CU: "Union(C2, C3)",
});
const TRADE_DEFS = {
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Text: { text: "str" },
  Call: { name: "str", input: "object" },
};
const TRADES = readTypedefs(TRADE_DEFS);
const BUY_SELL_DEFS = { Buy: { action: "Literal(buy)" }, Sell: { action: "Literal(sell)" } };
const BUY_SELL = readTypedefs(BUY_SELL_DEFS);
const NOW_LATER = readTypedefs({
  Now: { when: "Literal('right now')" },
  Later: { when: "Literal(later)" },
});
const TICKETS = readTypedefs({
  ...TRADE_DEFS,
  Limit: { px: "float" },
  Ticket: { kind: "Literal(ticket)", trade: "Union(Buy, Sell)", limit: "Limit" },
  Skip: { kind: "Literal(skip)", trade: "Union(Buy, Sell)" },
});
const ORDER = readTypedefs({ ...TRADE_DEFS, Order: { trade: "Union(Buy, Sell)", note: "str" } });
const LIMITED = readTypedefs({
  Limit: { px: "float" },
  Buy: { action: "Literal(buy)", limit: "Limit" },
  Sell: { action: "Literal(sell)" },
});
const HOLDER = readTypedefs({
  Point: { x: "int" },
  ...BUY_SELL_DEFS,
  Holder: { p: "Optional(Point)", u: "Optional(Union(Buy, Sell))", q: "Point" },
});
const ZED_ALPHA = readTypedefs({
  Zed: { t: "Literal(z)", u: "Union(Buy, Sell)" },
  Alpha: { t: "Literal(a)", p: "P" },
  P: { x: "int" },
  ...BUY_SELL_DEFS,
});

/** The type `expression` names. */
function t(expression: string, registry = NO_TYPEDEFS): Type {
  return typeFor(expression, registry);
}

/** A record type over `fields`, with the names in `required` required. */
function record(fields: Record<string, Type>, required: string[] = []): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(required),
  });
}

/** The schema of `typ`, which must not be `null`. */
function schemaOf(typ: Type): JsonObject {
  const schema = typeToSchema(typ);
  expect(schema).not.toBeNull();
  return schema!;
}

/** `value` as an object, for reading nested schema parts. */
function obj(value: JsonValue | undefined): JsonObject {
  expect(typeof value === "object" && value !== null && !Array.isArray(value)).toBe(true);
  return value as JsonObject;
}

/** The `anyOf` arms of `schema`. */
function arms(schema: JsonValue | undefined): JsonObject[] {
  const list = obj(schema).anyOf;
  expect(Array.isArray(list)).toBe(true);
  return list as JsonObject[];
}

/** The key order JS gives an object written with `keys` sorted (integer-like keys first). */
function sortedOrder(keys: string[]): string[] {
  return Object.keys(Object.fromEntries([...keys].sort(compareCodeUnits).map((key) => [key, 0])));
}

/** Expect the keys of `schema` sorted at every level but a `properties` map. */
function expectSortedKeys(schema: JsonValue): void {
  if (Array.isArray(schema)) {
    schema.forEach(expectSortedKeys);
    return;
  }
  if (typeof schema !== "object" || schema === null) return;
  expect(Object.keys(schema)).toEqual(sortedOrder(Object.keys(schema)));
  for (const [key, value] of Object.entries(schema)) {
    if (key === "properties") {
      Object.values(obj(value)).forEach(expectSortedKeys);
    } else if (key === "$defs") {
      // Its keys are names: a definition named `properties` is a schema, not a properties map.
      expect(Object.keys(obj(value))).toEqual(sortedOrder(Object.keys(obj(value))));
      Object.values(obj(value)).forEach(expectSortedKeys);
    } else {
      expectSortedKeys(value);
    }
  }
}

/** The wrapper property of a root schema: `items` for a list, else `value`. */
function wrapped(typ: Type): JsonObject {
  return obj(obj(schemaOf(typ).properties)[isListKind(typ.kind) ? "items" : "value"]);
}

/** Expect `typ`'s schema to be `expected` exactly, keys sorted but in `properties`. */
function expectSchema(typ: Type, expected: JsonValue): void {
  const schema = schemaOf(typ);
  expect(schema).toStrictEqual(expected as JsonObject);
  expectSortedKeys(schema);
}

// The expected schema per type, as captured (compact JSON, in its captured key order).
const CAPTURES: [string, () => Type, string][] = [
  [
    "variant str",
    () => new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]) }),
    `{"properties":{"value":{"enum":["a","b"],"title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Literal(a, b)",
    () => t("Literal(a, b)"),
    `{"properties":{"value":{"enum":["a","b"],"title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int",
    () => t("int"),
    `{"properties":{"value":{"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int(1..12)",
    () => t("int(1..12)"),
    `{"properties":{"value":{"maximum":12,"minimum":1,"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "float(-0.5..1)",
    () => t("float(-0.5..1)"),
    `{"properties":{"value":{"maximum":1,"minimum":-0.5,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int(4..4)",
    () => t("int(4..4)"),
    `{"properties":{"value":{"maximum":4,"minimum":4,"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(Rating)",
    () =>
      t(
        "list(Rating)",
        readTypedefs({
          Rating: {
            count: "int(1..12)",
            scores: "list(list(float(0..1)))",
            confidence: "Optional(float(0..1))",
          },
        }),
      ),
    `{"$defs":{"item":{"properties":{"count":{"maximum":12,"minimum":1,"title":"Count","type":"integer"},"scores":{"items":{"items":{"maximum":1,"minimum":0,"type":"number"},"type":"array"},"title":"Scores","type":"array"},"confidence":{"anyOf":[{"maximum":1,"minimum":0,"type":"number"},{"type":"null"}],"default":null,"title":"Confidence"}},"required":["count","scores"],"title":"item","type":"object"}},"properties":{"items":{"items":{"$ref":"#/$defs/item"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(Optional(int(1..12)))",
    () => t("list(Optional(int(1..12)))"),
    `{"properties":{"items":{"items":{"anyOf":[{"maximum":12,"minimum":1,"type":"integer"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "float(2**53..2**53+2)",
    () => t("float(9007199254740992..9007199254740994)"),
    `{"properties":{"value":{"maximum":9007199254740994,"minimum":9007199254740992,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "float(-(2**53+2)..-(2**53))",
    () => t("float(-9007199254740994..-9007199254740992)"),
    `{"properties":{"value":{"maximum":-9007199254740992,"minimum":-9007199254740994,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int(0..)",
    () => t("int(0..)"),
    `{"properties":{"value":{"minimum":0,"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int(..100)",
    () => t("int(..100)"),
    `{"properties":{"value":{"maximum":100,"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "float(0.0..)",
    () => t("float(0.0..)"),
    `{"properties":{"value":{"minimum":0.0,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "float(..1)",
    () => t("float(..1)"),
    `{"properties":{"value":{"maximum":1,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Order (Optional(list(Line)))",
    () =>
      t(
        "Order",
        readTypedefs({ Line: { qty: "int(1..)" }, Order: { lines: "Optional(list(Line))" } }),
      ),
    `{"$defs":{"lines_item":{"properties":{"qty":{"minimum":1,"title":"Qty","type":"integer"}},"required":["qty"],"title":"lines_item","type":"object"}},"properties":{"lines":{"anyOf":[{"items":{"$ref":"#/$defs/lines_item"},"type":"array"},{"type":"null"}],"default":null,"title":"Lines"}},"title":"Record","type":"object"}`,
  ],
  [
    "list(list(int))",
    () => t("list(list(int))"),
    `{"properties":{"items":{"items":{"items":{"type":"integer"},"type":"array"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "bare list",
    () => Type.scalar(ValueKind.LIST_ANY),
    `{"properties":{"items":{"items":{},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "test_record_type's record",
    () =>
      record({ name: Type.scalar(ValueKind.STRING), score: Type.scalar(ValueKind.NUMBER) }, [
        "name",
      ]),
    `{"properties":{"name":{"title":"Name","type":"string"},"score":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"title":"Score"}},"required":["name"],"title":"Record","type":"object"}`,
  ],
  [
    "test_list_of_records' list",
    () =>
      new Type({
        kind: ValueKind.LIST_OBJECT,
        element: record({ x: Type.scalar(ValueKind.INTEGER) }, ["x"]),
      }),
    `{"$defs":{"item":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"item","type":"object"}},"properties":{"items":{"items":{"$ref":"#/$defs/item"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Optional(int)",
    () => t("Optional(int)"),
    `{"properties":{"value":{"anyOf":[{"type":"integer"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(bool)",
    () => t("Optional(bool)"),
    `{"properties":{"value":{"anyOf":[{"type":"boolean"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(Literal(a, b))",
    () => t("Optional(Literal(a, b))"),
    `{"properties":{"value":{"anyOf":[{"enum":["a","b"],"type":"string"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(object)",
    () => t("Optional(object)"),
    `{"properties":{"value":{"anyOf":[{"additionalProperties":true,"type":"object"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(list(int))",
    () => t("Optional(list(int))"),
    `{"properties":{"items":{"anyOf":[{"items":{"type":"integer"},"type":"array"},{"type":"null"}],"title":"Items"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(int)",
    () => t("list(int)"),
    `{"properties":{"items":{"items":{"type":"integer"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Optional(Point)",
    () => t("Optional(Point)", POINT),
    `{"$defs":{"Record":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"Record","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Record"},{"type":"null"}]}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Point",
    () => t("Point", POINT),
    `{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"Record","type":"object"}`,
  ],
  [
    "list(Optional(int))",
    () => t("list(Optional(int))"),
    `{"properties":{"items":{"items":{"anyOf":[{"type":"integer"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(Optional(Literal(a, b)))",
    () => t("list(Optional(Literal(a, b)))"),
    `{"properties":{"items":{"items":{"anyOf":[{"enum":["a","b"],"type":"string"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(Optional(Point))",
    () => t("list(Optional(Point))", POINT),
    `{"$defs":{"item":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"item","type":"object"}},"properties":{"items":{"items":{"anyOf":[{"$ref":"#/$defs/item"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Any",
    () => t("Any"),
    `{"properties":{"value":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(Any)",
    () => t("Optional(Any)"),
    `{"properties":{"value":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(Any)",
    () => t("list(Any)"),
    `{"properties":{"items":{"items":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(Optional(Any))",
    () => t("list(Optional(Any))"),
    `{"properties":{"items":{"items":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Box {v: Any}",
    () => t("Box", readTypedefs({ Box: { v: "Any" } })),
    `{"properties":{"v":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"}],"title":"V"}},"required":["v"],"title":"Record","type":"object"}`,
  ],
  [
    "Box {v: Optional(Any), n: int}",
    () => t("Box", readTypedefs({ Box: { v: "Optional(Any)", n: "int" } })),
    `{"properties":{"v":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"},{"type":"null"}],"default":null,"title":"V"},"n":{"title":"N","type":"integer"}},"required":["n"],"title":"Record","type":"object"}`,
  ],
  [
    "object",
    () => t("object"),
    `{"properties":{"value":{"additionalProperties":true,"title":"Value","type":"object"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "date",
    () => t("date"),
    `{"properties":{"value":{"description":"A date: YYYY-MM-DD, such as 2026-05-21.","format":"date","title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "datetime",
    () => t("datetime"),
    `{"properties":{"value":{"description":"A date and time: YYYY-MM-DDTHH:MM[:SS[.ffffff]][Z|+HH:MM|-HH:MM], such as 2026-05-21T14:30:00Z. Seconds, a fraction of one to six digits and the offset are optional.","title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(date)",
    () => t("list(date)"),
    `{"properties":{"items":{"items":{"description":"A date: YYYY-MM-DD, such as 2026-05-21.","format":"date","type":"string"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Optional(datetime)",
    () => t("Optional(datetime)"),
    `{"properties":{"value":{"anyOf":[{"description":"A date and time: YYYY-MM-DDTHH:MM[:SS[.ffffff]][Z|+HH:MM|-HH:MM], such as 2026-05-21T14:30:00Z. Seconds, a fraction of one to six digits and the offset are optional.","type":"string"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "bool",
    () => t("bool"),
    `{"properties":{"value":{"title":"Value","type":"boolean"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "float",
    () => t("float"),
    `{"properties":{"value":{"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Lesson",
    () =>
      t(
        "Lesson",
        readTypedefs({
          Meta: { due_date: "date", score: "Optional(int(0..10))" },
          Lesson: { lesson_id: "str", meta: "Meta", meta_list: "list(Meta)" },
        }),
      ),
    `{"$defs":{"meta":{"properties":{"due_date":{"description":"A date: YYYY-MM-DD, such as 2026-05-21.","format":"date","title":"Due Date","type":"string"},"score":{"anyOf":[{"maximum":10,"minimum":0,"type":"integer"},{"type":"null"}],"default":null,"title":"Score"}},"required":["due_date"],"title":"meta","type":"object"},"meta_list_item":{"properties":{"due_date":{"description":"A date: YYYY-MM-DD, such as 2026-05-21.","format":"date","title":"Due Date","type":"string"},"score":{"anyOf":[{"maximum":10,"minimum":0,"type":"integer"},{"type":"null"}],"default":null,"title":"Score"}},"required":["due_date"],"title":"meta_list_item","type":"object"}},"properties":{"lesson_id":{"title":"Lesson Id","type":"string"},"meta":{"$ref":"#/$defs/meta"},"meta_list":{"items":{"$ref":"#/$defs/meta_list_item"},"title":"Meta List","type":"array"}},"required":["lesson_id","meta","meta_list"],"title":"Record","type":"object"}`,
  ],
  [
    // The `properties` field's schema is sorted too (checked by key order).
    "Odd (fields named like schema keywords)",
    () =>
      t(
        "Odd",
        readTypedefs({
          Odd: {
            properties: "int(0..3)",
            default: "Optional(int(1..2))",
            minimum: "str",
            title: "Literal(z, a)",
          },
        }),
      ),
    `{"properties":{"properties":{"type":"integer","maximum":3,"minimum":0,"title":"Properties"},"default":{"anyOf":[{"maximum":2,"minimum":1,"type":"integer"},{"type":"null"}],"default":null,"title":"Default"},"minimum":{"title":"Minimum","type":"string"},"title":{"enum":["a","z"],"title":"Title","type":"string"}},"required":["properties","minimum","title"],"title":"Record","type":"object"}`,
  ],
  [
    // The `properties` definition is sorted too (checked by key order).
    "Odd2 (a record field named properties)",
    () => t("Odd2", readTypedefs({ P: { x: "int" }, Odd2: { properties: "P", n: "int" } })),
    `{"$defs":{"properties":{"type":"object","properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"properties"}},"properties":{"properties":{"$ref":"#/$defs/properties"},"n":{"title":"N","type":"integer"}},"required":["properties","n"],"title":"Record","type":"object"}`,
  ],
  [
    "Odd3 (an optional record field named properties)",
    () => t("Odd3", readTypedefs({ P: { x: "int" }, Odd3: { properties: "Optional(P)" } })),
    `{"$defs":{"properties":{"type":"object","properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"properties"}},"properties":{"properties":{"anyOf":[{"$ref":"#/$defs/properties"},{"type":"null"}],"default":null}},"title":"Record","type":"object"}`,
  ],
  [
    "Optional(Pair)",
    () => t("Optional(Pair)", readTypedefs({ Pair: { b_side: "int", a_side: "Optional(str)" } })),
    `{"$defs":{"Record":{"properties":{"b_side":{"title":"B Side","type":"integer"},"a_side":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"A Side"}},"required":["b_side"],"title":"Record","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Record"},{"type":"null"}]}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "int(0..10)",
    () => t("int(0..10)"),
    `{"properties":{"value":{"maximum":10,"minimum":0,"title":"Value","type":"integer"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Optional(int(0..3))",
    () => t("Optional(int(0..3))"),
    `{"properties":{"value":{"anyOf":[{"maximum":3,"minimum":0,"type":"integer"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Literal(c, a, b)",
    () => t("Literal(c, a, b)"),
    `{"properties":{"value":{"enum":["a","b","c"],"title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(str)",
    () => t("list(str)"),
    `{"properties":{"items":{"items":{"type":"string"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(object)",
    () => t("list(object)"),
    `{"properties":{"items":{"items":{"additionalProperties":true,"type":"object"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Optional(float(0..1))",
    () => t("Optional(float(0..1))"),
    `{"properties":{"value":{"anyOf":[{"maximum":1,"minimum":0,"type":"number"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "R (open typedefs)",
    () => t("R", OPEN),
    `{"properties":{"a":{"title":"A","type":"integer"},"m":{"additionalProperties":true,"title":"M","type":"object"}},"required":["a","m"],"title":"Record","type":"object"}`,
  ],
  [
    "Outer (open typedefs)",
    () => t("Outer", OPEN),
    `{"$defs":{"inner":{"properties":{"a":{"title":"A","type":"integer"},"m":{"additionalProperties":true,"title":"M","type":"object"}},"required":["a","m"],"title":"inner","type":"object"}},"properties":{"inner":{"$ref":"#/$defs/inner"},"n":{"title":"N","type":"integer"}},"required":["inner","n"],"title":"Record","type":"object"}`,
  ],
  [
    "C (open typedefs)",
    () => t("C", OPEN),
    `{"properties":{"a":{"title":"A","type":"integer"},"b":{"items":{"type":"string"},"title":"B","type":"array"}},"required":["a","b"],"title":"Record","type":"object"}`,
  ],
  [
    "list(list(object))",
    () => t("list(list(object))", OPEN),
    `{"properties":{"items":{"items":{"items":{"additionalProperties":true,"type":"object"},"type":"array"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Optional(R) (open typedefs)",
    () => t("Optional(R)", OPEN),
    `{"$defs":{"Record":{"properties":{"a":{"title":"A","type":"integer"},"m":{"additionalProperties":true,"title":"M","type":"object"}},"required":["a","m"],"title":"Record","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Record"},{"type":"null"}]}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(R) (open typedefs)",
    () => t("list(R)", OPEN),
    `{"$defs":{"item":{"properties":{"a":{"title":"A","type":"integer"},"m":{"additionalProperties":true,"title":"M","type":"object"}},"required":["a","m"],"title":"item","type":"object"}},"properties":{"items":{"items":{"$ref":"#/$defs/item"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "float(0.0..1.0)",
    () => t("float(0.0..1.0)"),
    `{"properties":{"value":{"maximum":1.0,"minimum":0.0,"title":"Value","type":"number"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "the question-list type",
    questionListType,
    `{"$defs":{"item":{"properties":{"question":{"title":"Question","type":"string"},"header":{"title":"Header","type":"string"},"options":{"anyOf":[{"items":{"$ref":"#/$defs/item_options_item"},"type":"array"},{"type":"null"}],"default":null,"title":"Options"},"multi_select":{"anyOf":[{"type":"boolean"},{"type":"null"}],"default":null,"title":"Multi Select"}},"required":["question","header"],"title":"item","type":"object"},"item_options_item":{"properties":{"label":{"title":"Label","type":"string"},"description":{"title":"Description","type":"string"}},"required":["label","description"],"title":"item_options_item","type":"object"}},"properties":{"items":{"items":{"$ref":"#/$defs/item"},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Names (title rule)",
    () =>
      t(
        "Names",
        readTypedefs({ Names: { camelCase: "int", a1b: "int", HTTPCode: "int", x__y: "int" } }),
      ),
    `{"properties":{"camelCase":{"title":"Camelcase","type":"integer"},"a1b":{"title":"A1B","type":"integer"},"HTTPCode":{"title":"Httpcode","type":"integer"},"x__y":{"title":"X  Y","type":"integer"}},"required":["camelCase","a1b","HTTPCode","x__y"],"title":"Record","type":"object"}`,
  ],
  [
    "Optional(list(Point))",
    () => t("Optional(list(Point))", POINT),
    `{"$defs":{"item":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"item","type":"object"}},"properties":{"items":{"anyOf":[{"items":{"$ref":"#/$defs/item"},"type":"array"},{"type":"null"}],"title":"Items"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "list(Optional(list(int)))",
    () => t("list(Optional(list(int)))"),
    `{"properties":{"items":{"items":{"anyOf":[{"items":{"type":"integer"},"type":"array"},{"type":"null"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Literal(b)",
    () => t("Literal(b)"),
    `{"properties":{"value":{"const":"b","title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Union(Buy, Sell)",
    () => t("Union(Buy, Sell)", TRADES),
    `{"$defs":{"buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"qty":{"title":"Qty","type":"integer"}},"required":["action","qty"],"title":"buy","type":"object"},"sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"},"price":{"title":"Price","type":"number"}},"required":["action","price"],"title":"sell","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/buy"},{"$ref":"#/$defs/sell"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Union(Now, Later)",
    () => t("Union(Now, Later)", NOW_LATER),
    `{"$defs":{"Member1":{"additionalProperties":false,"properties":{"when":{"const":"later","title":"When","type":"string"}},"required":["when"],"title":"Member1","type":"object"},"Member2":{"additionalProperties":false,"properties":{"when":{"const":"right now","title":"When","type":"string"}},"required":["when"],"title":"Member2","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Member1"},{"$ref":"#/$defs/Member2"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "list(Union(Ticket, Skip))",
    () => t("list(Union(Ticket, Skip))", TICKETS),
    `{"$defs":{"item_skip":{"additionalProperties":false,"properties":{"kind":{"const":"skip","title":"Kind","type":"string"},"trade":{"anyOf":[{"$ref":"#/$defs/item_skip_trade_buy"},{"$ref":"#/$defs/item_skip_trade_sell"}],"title":"Trade"}},"required":["kind","trade"],"title":"item_skip","type":"object"},"item_skip_trade_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"qty":{"title":"Qty","type":"integer"}},"required":["action","qty"],"title":"item_skip_trade_buy","type":"object"},"item_skip_trade_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"},"price":{"title":"Price","type":"number"}},"required":["action","price"],"title":"item_skip_trade_sell","type":"object"},"item_ticket":{"additionalProperties":false,"properties":{"kind":{"const":"ticket","title":"Kind","type":"string"},"trade":{"anyOf":[{"$ref":"#/$defs/item_ticket_trade_buy"},{"$ref":"#/$defs/item_ticket_trade_sell"}],"title":"Trade"},"limit":{"$ref":"#/$defs/item_ticket_limit"}},"required":["kind","trade","limit"],"title":"item_ticket","type":"object"},"item_ticket_limit":{"additionalProperties":false,"properties":{"px":{"title":"Px","type":"number"}},"required":["px"],"title":"item_ticket_limit","type":"object"},"item_ticket_trade_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"qty":{"title":"Qty","type":"integer"}},"required":["action","qty"],"title":"item_ticket_trade_buy","type":"object"},"item_ticket_trade_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"},"price":{"title":"Price","type":"number"}},"required":["action","price"],"title":"item_ticket_trade_sell","type":"object"}},"properties":{"items":{"items":{"anyOf":[{"$ref":"#/$defs/item_skip"},{"$ref":"#/$defs/item_ticket"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Union(Text, Call)",
    () => t("Union(Text, Call)", TRADES),
    `{"$defs":{"Member1":{"additionalProperties":false,"properties":{"name":{"title":"Name","type":"string"},"input":{"additionalProperties":true,"title":"Input","type":"object"}},"required":["name","input"],"title":"Member1","type":"object"},"Member2":{"additionalProperties":false,"properties":{"text":{"title":"Text","type":"string"}},"required":["text"],"title":"Member2","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Member1"},{"$ref":"#/$defs/Member2"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Order (a union field)",
    () => t("Order", ORDER),
    `{"$defs":{"trade_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"qty":{"title":"Qty","type":"integer"}},"required":["action","qty"],"title":"trade_buy","type":"object"},"trade_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"},"price":{"title":"Price","type":"number"}},"required":["action","price"],"title":"trade_sell","type":"object"}},"properties":{"trade":{"anyOf":[{"$ref":"#/$defs/trade_buy"},{"$ref":"#/$defs/trade_sell"}],"title":"Trade"},"note":{"title":"Note","type":"string"}},"required":["trade","note"],"title":"Record","type":"object"}`,
  ],
  [
    "list(Union(Buy, Sell))",
    () => t("list(Union(Buy, Sell))", TRADES),
    `{"$defs":{"item_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"qty":{"title":"Qty","type":"integer"}},"required":["action","qty"],"title":"item_buy","type":"object"},"item_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"},"price":{"title":"Price","type":"number"}},"required":["action","price"],"title":"item_sell","type":"object"}},"properties":{"items":{"items":{"anyOf":[{"$ref":"#/$defs/item_buy"},{"$ref":"#/$defs/item_sell"}]},"title":"Items","type":"array"}},"required":["items"],"title":"ListWrapper","type":"object"}`,
  ],
  [
    "Union(Buy, Sell) (a record inside a member)",
    () => t("Union(Buy, Sell)", LIMITED),
    `{"$defs":{"buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"},"limit":{"$ref":"#/$defs/buy_limit"}},"required":["action","limit"],"title":"buy","type":"object"},"buy_limit":{"additionalProperties":false,"properties":{"px":{"title":"Px","type":"number"}},"required":["px"],"title":"buy_limit","type":"object"},"sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"}},"required":["action"],"title":"sell","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/buy"},{"$ref":"#/$defs/sell"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "D (open typedefs)",
    () => t("D", OPEN),
    `{"$defs":{"buy":{"additionalProperties":false,"properties":{"kind":{"const":"buy","title":"Kind","type":"string"},"x":{"anyOf":[{"type":"boolean"},{"type":"integer"},{"type":"number"},{"type":"string"},{"additionalProperties":true,"type":"object"},{"items":{},"type":"array"}],"title":"X"}},"required":["kind","x"],"title":"buy","type":"object"},"sell":{"additionalProperties":false,"properties":{"kind":{"const":"sell","title":"Kind","type":"string"},"n":{"title":"N","type":"integer"}},"required":["kind","n"],"title":"sell","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/buy"},{"$ref":"#/$defs/sell"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "CU (open typedefs)",
    () => t("CU", OPEN),
    `{"$defs":{"c2":{"additionalProperties":false,"properties":{"kind":{"const":"c2","title":"Kind","type":"string"},"a":{"title":"A","type":"integer"}},"required":["kind","a"],"title":"c2","type":"object"},"c3":{"additionalProperties":false,"properties":{"kind":{"const":"c3","title":"Kind","type":"string"},"b":{"title":"B","type":"string"}},"required":["kind","b"],"title":"c3","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/c2"},{"$ref":"#/$defs/c3"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Holder (nullable record and union fields)",
    () => t("Holder", HOLDER),
    `{"$defs":{"p":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"p","type":"object"},"q":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"q","type":"object"},"u_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"}},"required":["action"],"title":"u_buy","type":"object"},"u_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"}},"required":["action"],"title":"u_sell","type":"object"}},"properties":{"p":{"anyOf":[{"$ref":"#/$defs/p"},{"type":"null"}],"default":null},"u":{"anyOf":[{"$ref":"#/$defs/u_buy"},{"$ref":"#/$defs/u_sell"},{"type":"null"}],"default":null,"title":"U"},"q":{"$ref":"#/$defs/q"}},"required":["q"],"title":"Record","type":"object"}`,
  ],
  [
    "Optional(Union(Buy, Sell))",
    () => t("Optional(Union(Buy, Sell))", BUY_SELL),
    `{"$defs":{"buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"}},"required":["action"],"title":"buy","type":"object"},"sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"}},"required":["action"],"title":"sell","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/buy"},{"$ref":"#/$defs/sell"},{"type":"null"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "Union(Zed, Alpha) (members in canonical order)",
    () => t("Union(Zed, Alpha)", ZED_ALPHA),
    `{"$defs":{"a":{"additionalProperties":false,"properties":{"t":{"const":"a","title":"T","type":"string"},"p":{"$ref":"#/$defs/a_p"}},"required":["t","p"],"title":"a","type":"object"},"a_p":{"additionalProperties":false,"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"a_p","type":"object"},"z":{"additionalProperties":false,"properties":{"t":{"const":"z","title":"T","type":"string"},"u":{"anyOf":[{"$ref":"#/$defs/z_u_buy"},{"$ref":"#/$defs/z_u_sell"}],"title":"U"}},"required":["t","u"],"title":"z","type":"object"},"z_u_buy":{"additionalProperties":false,"properties":{"action":{"const":"buy","title":"Action","type":"string"}},"required":["action"],"title":"z_u_buy","type":"object"},"z_u_sell":{"additionalProperties":false,"properties":{"action":{"const":"sell","title":"Action","type":"string"}},"required":["action"],"title":"z_u_sell","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/a"},{"$ref":"#/$defs/z"}],"title":"Value"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
];

describe("captured schemas", () => {
  test.each(CAPTURES)("%s", (_label, build, json) => {
    expectSchema(build(), parseJson(json, "plain"));
  });

  test("int(10**400..10**400+2) keeps its exact bounds", () => {
    const lower = 10n ** 400n;
    expectSchema(t(`int(${lower}..${lower + 2n})`), {
      properties: {
        value: { maximum: lower + 2n, minimum: lower, title: "Value", type: "integer" },
      },
      required: ["value"],
      title: "ScalarWrapper",
      type: "object",
    });
  });
});

test("bare str returns none", () => {
  expect(typeToSchema(Type.scalar(ValueKind.STRING))).toBeNull();
});

test("test_variant_str_gets_scalar_schema (schema half)", () => {
  const schema = schemaOf(new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]) }));
  expect(obj(schema.properties).value).toStrictEqual({
    enum: ["a", "b"],
    title: "Value",
    type: "string",
  });
});

test("test_scalar_int_gets_schema (schema half)", () => {
  const schema = schemaOf(Type.scalar(ValueKind.INTEGER));
  expect(schema.title).toBe("ScalarWrapper");
  expect(Object.keys(obj(schema.properties))).toEqual(["value"]);
});

describe("test_bounded_scalar_schema_exposes_and_enforces_inclusive_limits (schema half)", () => {
  test.each([
    ["int(1..12)", 1n, 12n],
    ["float(-0.5..1)", -0.5, 1n],
    ["int(4..4)", 4n, 4n],
  ] as [string, bigint | number, bigint | number][])("%s", (expression, lower, upper) => {
    const value = obj(obj(schemaOf(t(expression)).properties).value);
    expect([value.minimum, value.maximum]).toEqual([lower, upper]);
  });
});

test("a hand-built -0 bound becomes 0, also inside a record", () => {
  const typ = new Type({ kind: ValueKind.NUMBER, bounds: [-0, 1] });
  const value = obj(obj(schemaOf(typ).properties).value);
  expect(Object.is(value.minimum, 0)).toBe(true);
  const outer = schemaOf(record({ r: record({ x: typ }, ["x"]) }, ["r"]));
  const field = obj(obj(obj(obj(outer.$defs).r).properties).x);
  expect(Object.is(field.minimum, 0)).toBe(true);
});

test("test_bounded_numeric_schemas_preserve_nested_lists_records_and_nullable_fields (schema half)", () => {
  const registry = readTypedefs({
    Rating: {
      count: "int(1..12)",
      scores: "list(list(float(0..1)))",
      confidence: "Optional(float(0..1))",
    },
  });
  const schema = schemaOf(t("list(Rating)", registry));
  const ref = obj(obj(obj(schema.properties).items).items).$ref as string;
  const fields = obj(obj(obj(schema.$defs)[ref.split("/").at(-1)!]).properties);
  expect([obj(fields.count).minimum, obj(fields.count).maximum]).toEqual([1n, 12n]);
  const scalar = obj(obj(obj(fields.scores).items).items);
  expect([scalar.minimum, scalar.maximum]).toEqual([0n, 1n]);
  expect(arms(fields.confidence)).toContainEqual({ maximum: 1n, minimum: 0n, type: "number" });
});

test("test_nullable_bounded_scalars_keep_explicit_null_in_schema (schema half)", () => {
  const schema = schemaOf(t("list(Optional(int(1..12)))"));
  const items = obj(obj(obj(schema.properties).items).items);
  expect(arms(items)).toContainEqual({ maximum: 12n, minimum: 1n, type: "integer" });
  expect(arms(items)).toContainEqual({ type: "null" });
});

describe("test_large_numeric_schema_bounds_match_resolved_type (schema and buildValueAs halves)", () => {
  test.each([
    ["int", 10n ** 400n, 10n ** 400n + 2n, 10n ** 400n + 3n],
    ["float", 2n ** 53n, 2n ** 53n + 2n, 2n ** 53n + 4n],
    ["float", -(2n ** 53n + 2n), -(2n ** 53n), -(2n ** 53n + 4n)],
  ])("%s(%p..%p)", (kind, lower, upper, outside) => {
    const typ = t(`${kind}(${lower}..${upper})`);
    typ.validateShape();
    const value = obj(obj(schemaOf(typ).properties).value);
    expect([value.minimum, value.maximum]).toEqual([lower, upper]);
    for (const bound of [lower, upper]) {
      expect(buildValueAs(typ, bound).toObject()).toBe(bound);
    }
    expect(() => buildValueAs(typ, outside)).toThrow(TypeCheckError);
  });
});

describe("test_one_sided_schema_has_only_that_limit (schema half)", () => {
  test.each([
    ["int(0..)", "minimum", 0n],
    ["int(..100)", "maximum", 100n],
    ["float(0.0..)", "minimum", 0],
    ["float(..1)", "maximum", 1n],
  ] as [string, string, bigint | number][])("%s", (expression, keyword, limit) => {
    const value = obj(obj(schemaOf(t(expression)).properties).value);
    const other = keyword === "minimum" ? "maximum" : "minimum";
    expect(value[keyword]).toBe(limit);
    expect(other in value).toBe(false);
  });
});

test("test_one_sided_limit_survives_nested_records_and_lists (schema half)", () => {
  const registry = readTypedefs({
    Line: { qty: "int(1..)" },
    Order: { lines: "Optional(list(Line))" },
  });
  const qty = obj(obj(obj(obj(schemaOf(t("Order", registry)).$defs).lines_item).properties).qty);
  expect(qty.minimum).toBe(1n);
  expect("maximum" in qty).toBe(false);
});

test("test_generic_list_schema_honors_existing_unbounded_element_refinements (schema half)", () => {
  const nested = schemaOf(t("list(list(int))"));
  expect(obj(obj(obj(nested.properties).items).items).items).toStrictEqual({ type: "integer" });
  const bare = schemaOf(Type.scalar(ValueKind.LIST_ANY));
  expect(obj(obj(bare.properties).items).items).toStrictEqual({});
});

test("test_record_type and test_list_of_records (schema titles)", () => {
  const point = record({ x: Type.scalar(ValueKind.INTEGER) }, ["x"]);
  expect(schemaOf(point).title).toBe("Record");
  expect(schemaOf(new Type({ kind: ValueKind.LIST_OBJECT, element: point })).title).toBe(
    "ListWrapper",
  );
});

/** The member definitions the `anyOf` arms of `armSchema` refer to, in order. */
function unionArms(schema: JsonObject, armSchema: JsonValue | undefined): JsonObject[] {
  expect("oneOf" in obj(armSchema) || "discriminator" in obj(armSchema)).toBe(false);
  return arms(armSchema)
    .filter((arm) => "$ref" in arm)
    .map((arm) => obj(obj(schema.$defs)[(arm.$ref as string).split("/").at(-1)!]));
}

/** Every key of a JSON schema, at any depth. */
function schemaKeys(node: JsonValue): string[] {
  if (Array.isArray(node)) return node.flatMap(schemaKeys);
  if (typeof node !== "object" || node === null) return [];
  return Object.entries(node).flatMap(([key, value]) => [key, ...schemaKeys(value)]);
}

/** Expect `size` closed member definitions with distinct titles, and no `oneOf` anywhere. */
function expectMemberDefs(schema: JsonObject, members: JsonObject[], size: number): void {
  expect(members).toHaveLength(size);
  expect(members.every((member) => member.additionalProperties === false)).toBe(true);
  expect(new Set(members.map((member) => member.title)).size).toBe(size);
  const keys = new Set(schemaKeys(schema));
  expect(keys.has("oneOf") || keys.has("discriminator")).toBe(false);
}

/** The `const` of each member's `action` property. */
function actions(members: JsonObject[]): JsonValue[] {
  return members.map((member) => obj(obj(member.properties).action).const!);
}

test("tagged union schema is any of closed members with const tags", () => {
  const schema = schemaOf(t("Union(Buy, Sell)", TRADES));
  expect(schema.title).toBe("ScalarWrapper");
  const members = unionArms(schema, obj(schema.properties).value);
  expectMemberDefs(schema, members, 2);
  expect(actions(members)).toEqual(["buy", "sell"]);
  expect(members.map((member) => member.title)).toEqual(["buy", "sell"]);
});

test("union members with non identifier tags are numbered", () => {
  const schema = schemaOf(t("Union(Now, Later)", NOW_LATER));
  const members = unionArms(schema, obj(schema.properties).value);
  expect(members.map((member) => member.title)).toEqual(["Member1", "Member2"]);
});

test("union nested in a member gets position named defs", () => {
  const defs = obj(schemaOf(t("list(Union(Ticket, Skip))", TICKETS)).$defs);
  expect(Object.keys(defs).filter((key) => key.includes("__"))).toEqual([]);
  const titles = new Set(Object.values(defs).map((definition) => obj(definition).title));
  expect(titles.size).toBe(Object.keys(defs).length);
  expect(new Set(Object.keys(defs))).toEqual(
    new Set([
      "item_skip",
      "item_skip_trade_buy",
      "item_skip_trade_sell",
      "item_ticket",
      "item_ticket_limit",
      "item_ticket_trade_buy",
      "item_ticket_trade_sell",
    ]),
  );
});

test("untagged union schema is any of closed members", () => {
  const schema = schemaOf(t("Union(Text, Call)", TRADES));
  const members = unionArms(schema, obj(schema.properties).value);
  expectMemberDefs(schema, members, 2);
  const names = members.map((member) => Object.keys(obj(member.properties)).sort());
  expect(names.sort()).toEqual([["input", "name"], ["text"]]);
});

test("record field of union type gets member schemas", () => {
  const schema = schemaOf(t("Order", ORDER));
  const members = unionArms(schema, obj(schema.properties).trade);
  expectMemberDefs(schema, members, 2);
  expect(actions(members)).toEqual(["buy", "sell"]);
});

test("list of union schema gets member schemas", () => {
  const schema = schemaOf(t("list(Union(Buy, Sell))", TRADES));
  expect(schema.title).toBe("ListWrapper");
  const members = unionArms(schema, obj(obj(schema.properties).items).items);
  expectMemberDefs(schema, members, 2);
});

test("test_records_nested_in_union_members_reject_unknown_fields (schema half)", () => {
  const schema = schemaOf(t("Union(Buy, Sell)", LIMITED));
  expect(obj(obj(schema.$defs).buy_limit).additionalProperties).toBe(false);
});

describe("test_nullable_root_schema_accepts_null (schema half)", () => {
  test.each([
    "Optional(int)",
    "Optional(bool)",
    "Optional(Literal(a, b))",
    "Optional(object)",
    "Optional(list(int))",
  ])("%s", (expression) => {
    expect(arms(wrapped(t(expression)))).toContainEqual({ type: "null" });
  });
});

test("test_non_nullable_root_schema_rejects_null (schema half)", () => {
  for (const expression of ["int", "Literal(a, b)", "list(int)"]) {
    expect("anyOf" in wrapped(t(expression))).toBe(false);
  }
});

test("test_nullable_list_elements_accept_null_items (schema half)", () => {
  for (const expression of [
    "list(Optional(int))",
    "list(Optional(Literal(a, b)))",
    "list(Optional(Point))",
  ]) {
    expect(arms(wrapped(t(expression, POINT)).items)).toContainEqual({ type: "null" });
  }
  expect("anyOf" in obj(wrapped(t("list(int)")).items)).toBe(false);
});

test("test_nullable_record_root_is_wrapped_so_it_can_be_null (schema half)", () => {
  expect(schemaOf(t("Optional(Point)", POINT)).title).toBe("ScalarWrapper");
  expect(schemaOf(t("Point", POINT)).title).toBe("Record");
});

test("test_any_schema_accepts_every_non_null_json_value_unchanged (schema half)", () => {
  expect(schemaOf(Type.scalar(ValueKind.ANY)).title).toBe("ScalarWrapper");
});

test("test_any_schema_lists_the_json_types_and_optional_any_adds_null (schema half)", () => {
  const jsonTypes = (expression: string) =>
    new Set(arms(wrapped(t(expression))).map((arm) => arm.type));
  const kinds = ["object", "array", "string", "integer", "number", "boolean"];
  expect(jsonTypes("Any")).toEqual(new Set(kinds));
  expect(jsonTypes("Optional(Any)")).toEqual(new Set([...kinds, "null"]));
});

test("test_list_of_any_items_reject_null_and_list_of_optional_any_items_accept_it (schema half)", () => {
  expect(arms(wrapped(t("list(Any)")).items)).not.toContainEqual({ type: "null" });
  expect(arms(wrapped(t("list(Optional(Any))")).items)).toContainEqual({ type: "null" });
});

test("test_flow_fingerprint_preserves_structural_field_order_used_by_agent_schema (schema half)", () => {
  const first = record({
    first: Type.scalar(ValueKind.INTEGER),
    second: Type.scalar(ValueKind.STRING),
  });
  const second = record({
    second: Type.scalar(ValueKind.STRING),
    first: Type.scalar(ValueKind.INTEGER),
  });
  const firstField = `{"anyOf":[{"type":"integer"},{"type":"null"}],"default":null,"title":"First"}`;
  const secondField = `{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Second"}`;
  expectSchema(
    first,
    parseJson(
      `{"properties":{"first":${firstField},"second":${secondField}},"title":"Record","type":"object"}`,
      "plain",
    ),
  );
  expectSchema(
    second,
    parseJson(
      `{"properties":{"second":${secondField},"first":${firstField}},"title":"Record","type":"object"}`,
      "plain",
    ),
  );
  expect(Object.keys(obj(schemaOf(first).properties))).toEqual(["first", "second"]);
  expect(Object.keys(obj(schemaOf(second).properties))).toEqual(["second", "first"]);
});

/** The `TypeCheckError` `checkReply` throws for `data` against `typ`. */
function refused(typ: Type, data: JsonObject): TypeCheckError {
  try {
    checkReply(data, typ);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("checkReply accepted the reply");
}

/** A native model whose every reply's text is `payload` (JSON). */
function native(payload: string, calls = 1): ScriptedModel {
  const text = new Response({ message: new Message({ role: "assistant", parts: [new TextPart(payload)] }) });
  return new ScriptedModel(Array.from({ length: calls }, () => text), { provider: "openai", model: "gpt-5.5" });
}

describe("an integral float in an int slot of a model reply", () => {
  test("is read as an int, in a field, a list or an optional", () => {
    expect(checkReply({ value: 2 }, t("int"))).toBe(2n);
    expect(checkReply({ value: -0 }, t("int"))).toBe(0n);
    expect(checkReply({ items: [1, 2n] }, t("list(int)"))).toEqual([1n, 2n]);
    const typ = t("Order", readTypedefs({ Line: { qty: "int(1..)" }, Order: { lines: "list(Line)", n: "Optional(int)" } }));
    expect(checkReply({ lines: [{ qty: 3 }], n: 4 }, typ)).toEqual({ lines: [{ qty: 3n }], n: 4n });
  });

  test("a fraction, an unsafe integer or one out of bounds is still refused", () => {
    expect(refused(t("int"), { value: 2.5 }).message).toBe("structured output: 2.5 does not match declared type int");
    refused(t("int"), { value: 2 ** 53 });
    refused(t("int(1..12)"), { value: 13 });
  });

  test("a float slot keeps the float", () => {
    expect(checkReply({ value: 2 }, t("float"))).toBe(2);
  });

  test("a reply's 2.0 fills an int output", async () => {
    expect(await generateStructured(native('{"value": 2.0}'), [], t("int"), { retries: 0 })).toBe(2n);
  });
});

describe("validate and unwrap halves, through checkReply", () => {
  test("test_variant_str_gets_scalar_schema (validate half)", () => {
    const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]) });
    expect(checkReply({ value: "a" }, typ)).toBe("a");
    refused(typ, { value: "not-a-tag" });
  });

  test("test_scalar_int_gets_schema (validate half)", () => {
    expect(checkReply({ value: 7n }, Type.scalar(ValueKind.INTEGER))).toBe(7n);
  });

  test.each([
    ["int(1..12)", [1n, 4n, 12n], 13n],
    ["float(-0.5..1)", [-0.5, 0.4, 1n], 1.1],
    ["int(4..4)", [4n], 3n],
  ] as [string, (bigint | number)[], bigint | number][])(
    "test_bounded_scalar_schema_exposes_and_enforces_inclusive_limits (validate half) %s",
    (expression, inside, outside) => {
      for (const value of inside) expect(checkReply({ value }, t(expression))).toBe(value);
      refused(t(expression), { value: outside });
    },
  );

  test("test_bounded_numeric_schemas_preserve_nested_lists_records_and_nullable_fields (validate half)", () => {
    const typ = t(
      "list(Rating)",
      readTypedefs({
        Rating: {
          count: "int(1..12)",
          scores: "list(list(float(0..1)))",
          confidence: "Optional(float(0..1))",
        },
      }),
    );
    const items = [{ count: 4n, scores: [[0n, 1n]], confidence: null }];
    expect(checkReply({ items }, typ)).toEqual(items);
    expect(refused(typ, { items: [{ count: 4n, scores: [[0n, 1.1]] }] }).path).toEqual([0, "scores", 0, 1]);
  });

  test("test_nullable_bounded_scalars_keep_explicit_null_in_schema (validate half)", () => {
    expect(checkReply({ items: [null, 4n] }, t("list(Optional(int(1..12)))"))).toEqual([null, 4n]);
  });

  test.each([
    ["int", 10n ** 400n, 10n ** 400n + 2n, 10n ** 400n + 3n],
    ["float", 2n ** 53n, 2n ** 53n + 2n, 2n ** 53n + 4n],
    ["float", -(2n ** 53n + 2n), -(2n ** 53n), -(2n ** 53n + 4n)],
  ])("test_large_numeric_schema_bounds_match_resolved_type (validate half) %s(%p..%p)", (kind, lower, upper, outside) => {
    const typ = t(`${kind}(${lower}..${upper})`);
    for (const bound of [lower, upper]) expect(checkReply({ value: bound }, typ)).toBe(bound);
    refused(typ, { value: outside });
  });

  test.each([
    ["int(0..)", 10n ** 30n, -1n],
    ["int(..100)", -(10n ** 30n), 101n],
    ["float(0.0..)", 1e300, -0.5],
    ["float(..1)", -2.5, 1.5],
  ] as [string, bigint | number, bigint | number][])(
    "test_one_sided_schema_has_only_that_limit (validate half) %s",
    (expression, inside, outside) => {
      expect(checkReply({ value: inside }, t(expression))).toBe(inside);
      refused(t(expression), { value: outside });
    },
  );

  test("test_one_sided_limit_survives_nested_records_and_lists (validate half)", () => {
    const typ = t("Order", readTypedefs({ Line: { qty: "int(1..)" }, Order: { lines: "Optional(list(Line))" } }));
    refused(typ, { lines: [{ qty: 0n }] });
    expect(checkReply({ lines: [{ qty: 10n ** 30n }] }, typ)).toEqual({ lines: [{ qty: 10n ** 30n }] });
    expect(checkReply({ lines: null }, typ)).toEqual({ lines: null });
  });

  test("test_generic_list_schema_honors_existing_unbounded_element_refinements (validate half)", () => {
    const nested = t("list(list(int))");
    expect(checkReply({ items: [[1n], [2n]] }, nested)).toEqual([[1n], [2n]]);
    expect(refused(nested, { items: [[1n], ["not-an-int"]] }).path).toEqual([1, 0]);
    const bare = Type.scalar(ValueKind.LIST_ANY);
    expect(checkReply({ items: [1n, "free", {}] }, bare)).toEqual([1n, "free", {}]);
  });

  test("test_record_type (validate half)", () => {
    const typ = record({ name: Type.scalar(ValueKind.STRING), score: Type.scalar(ValueKind.NUMBER) }, ["name"]);
    expect(checkReply({ name: "a", score: 1.5 }, typ)).toEqual({ name: "a", score: 1.5 });
  });

  test("test_list_of_records (validate half)", () => {
    const typ = new Type({ kind: ValueKind.LIST_OBJECT, element: record({ x: Type.scalar(ValueKind.INTEGER) }, ["x"]) });
    expect(checkReply({ items: [{ x: 1n }, { x: 2n }] }, typ)).toEqual([{ x: 1n }, { x: 2n }]);
  });

  test("union members reject unknown fields and unwrap to the member dict", () => {
    const union = t("Union(Buy, Sell)", TRADES);
    expect(checkReply({ value: { action: "sell", price: 1.5 } }, union)).toEqual({ action: "sell", price: 1.5 });
    refused(union, { value: { action: "buy", qty: 2n, zzz: 1n } });
  });

  test("test_records_nested_in_union_members_reject_unknown_fields (validate half)", () => {
    refused(t("Union(Buy, Sell)", LIMITED), { value: { action: "buy", limit: { px: 1.0, zzz: 1n } } });
  });

  test.each([
    ["Optional(int)", 4n],
    ["Optional(bool)", true],
    ["Optional(Literal(a, b))", "a"],
    ["Optional(object)", { k: 1n }],
    ["Optional(list(int))", [1n]],
  ] as [string, JsonValue][])("test_nullable_root_schema_accepts_null (validate half) %s", (expression, value) => {
    const typ = t(expression);
    const field = isListKind(typ.kind) ? "items" : "value";
    expect(checkReply({ [field]: null }, typ)).toBeNull();
    expect(checkReply({ [field]: value }, typ)).toEqual(value);
  });

  test("test_non_nullable_root_schema_rejects_null (validate half)", () => {
    for (const expression of ["int", "Literal(a, b)", "list(int)"]) {
      const typ = t(expression);
      refused(typ, { [isListKind(typ.kind) ? "items" : "value"]: null });
    }
  });

  test("test_nullable_list_elements_accept_null_items (validate half)", () => {
    for (const [expression, items] of [
      ["list(Optional(int))", [1n, null]],
      ["list(Optional(Literal(a, b)))", ["a", null]],
      ["list(Optional(Point))", [{ x: 1n }, null]],
    ] as [string, JsonValue[]][]) {
      expect(checkReply({ items }, t(expression, POINT))).toEqual(items);
    }
    refused(t("list(int)"), { items: [1n, null] });
  });

  test("test_nullable_record_root_is_wrapped_so_it_can_be_null (validate half)", () => {
    const typ = t("Optional(Point)", POINT);
    expect(checkReply({ value: null }, typ)).toBeNull();
    expect(checkReply({ value: { x: 1n } }, typ)).toEqual({ x: 1n });
    expect(checkReply({ x: 1n }, t("Point", POINT))).toEqual({ x: 1n });
  });

  test("test_any_schema_accepts_every_non_null_json_value_unchanged (validate half)", () => {
    const typ = Type.scalar(ValueKind.ANY);
    for (const value of ["x", 3n, 2n ** 80n, 2.5, 1.0, true, "3", { a: 1n }, [1n, "a"], { a: 1.0, b: true }] as JsonValue[]) {
      expect(checkReply({ value }, typ)).toStrictEqual(value);
    }
    refused(typ, { value: null });
  });

  test("test_any_schema_lists_the_json_types_and_optional_any_adds_null (unwrap half)", () => {
    expect(checkReply({ value: null }, t("Optional(Any)"))).toBeNull();
  });

  test("test_list_of_any_items_reject_null_and_list_of_optional_any_items_accept_it (validate half)", () => {
    const items = [1n, "a", { k: [null] }];
    expect(checkReply({ items }, t("list(Any)"))).toEqual(items);
    refused(t("list(Any)"), { items: [1n, null] });
    expect(checkReply({ items: [1n, null] }, t("list(Optional(Any))"))).toEqual([1n, null]);
  });

  test.each([
    ["Box", '{"v": null}', ["v"]],
    ["Any", '{"value": null}', []],
    ["list(Any)", '{"items": [1, null]}', [1]],
  ] as [string, string, (string | number)[]][])(
    "test_null_at_an_any_position_is_reported_at_that_position %s",
    async (expression, payload, path) => {
      const typ = t(expression, readTypedefs({ Box: { v: "Any" } }));
      const error = await generateStructured(native(payload), [], typ, { retries: 0 }).catch((caught: unknown) => caught);
      expect(error).toBeInstanceOf(OutputInvalidError);
      expect((error as OutputInvalidError).path).toEqual(path);
    },
  );

  test("absent optional any record field reads none", async () => {
    const typ = t("Box", readTypedefs({ Box: { v: "Optional(Any)", n: "int" } }));
    const value = await generateStructured(native('{"n": 1}'), [], typ, { retries: 0 });
    expect(value).toEqual({ v: null, n: 1n });
    expect(Object.keys(value as JsonObject)).toEqual(["v", "n"]);
  });

  test("an open record drops unknown fields and comes back in declared field order", () => {
    const typ = record({ a: Type.scalar(ValueKind.INTEGER), b: Type.scalar(ValueKind.STRING) }, ["a", "b"]);
    const value = checkReply({ zz: 1n, b: "x", a: 1n }, typ);
    expect(value).toEqual({ a: 1n, b: "x" });
    expect(Object.keys(value as JsonObject)).toEqual(["a", "b"]);
  });

  test("an absent field that is not required but not nullable stays absent", () => {
    const typ = record({ a: Type.scalar(ValueKind.INTEGER), b: Type.scalar(ValueKind.NUMBER) }, ["a"]);
    expect(checkReply({ a: 1n }, typ)).toEqual({ a: 1n });
  });
});

describe("TS-only", () => {
  test("a root Optional(str) gives null; a nullable tagged string is wrapped", () => {
    expect(typeToSchema(t("Optional(str)"))).toBeNull();
    expect(schemaOf(t("Optional(Literal(a))")).title).toBe("ScalarWrapper");
  });

  test("titles title-case each word, then turn _ into a space, then strip", () => {
    // Each run of cased letters starts in title case and goes on in lower case, with Unicode's
    // full mappings (`ß` to `Ss`, a final sigma `ς`).
    const titles: [string, string][] = [
      ["ªb", "ªb"],
      ["ßa", "Ssa"],
      ["ǆx", "ǅx"],
      ["Ǆx", "ǅx"],
      ["xǆ", "Xǆ"],
      ["lessonID", "Lessonid"],
      ["x1y", "X1Y"],
      ["ΑΣ", "Ας"],
      ["ΑΣ_b", "Ας B"],
      ["aΣ", "Aς"],
      ["ΣΑ", "Σα"],
      ["ﬁx", "Fix"],
      ["İx", "İx"],
      ["xİ", "Xi̇"],
      ["é_à", "É À"],
      ["ŉa", "ʼNa"],
      ["სახელი", "სახელი"],
      ["ᾀx", "ᾈx"],
      ["xᾀ", "Xᾀ"],
      ["ᾷb", "ᾼ͂b"],
      ["_", ""],
      ["a_", "A"],
      ["x_", "X"],
    ];
    const typ = record(
      Object.fromEntries(titles.map(([name]) => [name, Type.scalar(ValueKind.INTEGER)])),
      titles.map(([name]) => name),
    );
    const properties = obj(schemaOf(typ).properties);
    expect(titles.map(([name]) => obj(properties[name]).title)).toEqual(
      titles.map(([, title]) => title),
    );
  });

  test("fields starting with _ are kept, __proto__ as an own property", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["_x", Type.scalar(ValueKind.INTEGER)],
        ["__proto__", Type.scalar(ValueKind.STRING)],
      ]),
      required: new Set(["_x", "__proto__"]),
    });
    const schema = schemaOf(typ);
    const properties = obj(schema.properties);
    expect(Object.keys(properties)).toEqual(["_x", "__proto__"]);
    expect(Object.hasOwn(properties, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(properties)).toBe(Object.prototype);
    expect(properties._x).toStrictEqual({ title: "X", type: "integer" });
    expect(properties.__proto__).toStrictEqual({ title: "Proto", type: "string" });
    expect(schema.required).toEqual(["_x", "__proto__"]);
  });

  test("integer-like field names come first in properties; required keeps field order", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["b", Type.scalar(ValueKind.INTEGER)],
        ["10", Type.scalar(ValueKind.INTEGER)],
        ["2", Type.scalar(ValueKind.INTEGER)],
      ]),
      required: new Set(["b", "10", "2"]),
    });
    const schema = schemaOf(typ);
    expect(Object.keys(obj(schema.properties))).toEqual(["2", "10", "b"]);
    expect(schema.required).toEqual(["b", "10", "2"]);
  });

  test("a field named properties has its schema keys sorted", () => {
    const schema = schemaOf(t("Odd", readTypedefs({ Odd: { properties: "int(0..3)" } })));
    expect(Object.keys(obj(obj(schema.properties).properties))).toEqual([
      "maximum",
      "minimum",
      "title",
      "type",
    ]);
  });

  test("a scalar with bounds is shape-checked first", () => {
    expect(() => typeToSchema(new Type({ kind: ValueKind.DATE, bounds: [1n, 2n] }))).toThrow(
      new RangeError("only int and float types may declare bounds"),
    );
    expect(() => typeToSchema(new Type({ kind: ValueKind.INTEGER, bounds: [3n, 2n] }))).toThrow(
      new RangeError("type bounds must be ordered from minimum to maximum"),
    );
  });

  describe("$defs keys and clashes", () => {
    const P = { x: "int" };
    const Q = { y: "str" };
    /** The JSON text of P's definition titled `title`. */
    const defX = (title: string) =>
      `{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"${title}","type":"object"}`;
    /** The JSON text of Q's definition titled `title`. */
    const defY = (title: string) =>
      `{"properties":{"y":{"title":"Y","type":"string"}},"required":["y"],"title":"${title}","type":"object"}`;

    test("a key replaces each code point outside [A-Za-z0-9_-] with _; the title stays raw", () => {
      expectSchema(
        t("R", readTypedefs({ P, R: { "my key": "P", "a/b": "P", café: "P" } })),
        parseJson(
          `{"$defs":{"a_b":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"a/b","type":"object"},"caf_":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"café","type":"object"},"my_key":{"properties":{"x":{"title":"X","type":"integer"}},"required":["x"],"title":"my key","type":"object"}},"properties":{"my key":{"$ref":"#/$defs/my_key"},"a/b":{"$ref":"#/$defs/a_b"},"café":{"$ref":"#/$defs/caf_"}},"required":["my key","a/b","café"],"title":"Record","type":"object"}`,
          "plain",
        ),
      );
      // A `.` becomes `_` in the key too.
      const dotted = schemaOf(t("R", readTypedefs({ P, R: { "a.b": "P" } })));
      expect(obj(dotted.properties)["a.b"]).toStrictEqual({ $ref: "#/$defs/a_b" });
      expect(obj(obj(dotted.$defs).a_b).title).toBe("a.b");
      // One `_` for the code point, not one per UTF-16 half.
      const emoji = schemaOf(t("R", readTypedefs({ P, R: { "😀": "P" } })));
      expect(obj(emoji.properties)["😀"]).toStrictEqual({ $ref: "#/$defs/_" });
      expect(obj(obj(emoji.$defs)._).title).toBe("😀");
    });

    test("different records wanting one key are numbered in definition order", () => {
      expectSchema(
        t("C", readTypedefs({ P, Q, I: { b: "Q" }, C: { a_b: "P", a: "I" } })),
        parseJson(
          `{"$defs":{"a":{"properties":{"b":{"$ref":"#/$defs/a_b__2"}},"required":["b"],"title":"a","type":"object"},"a_b__1":${defX("a_b")},"a_b__2":${defY("a_b")}},"properties":{"a_b":{"$ref":"#/$defs/a_b__1"},"a":{"$ref":"#/$defs/a"}},"required":["a_b","a"],"title":"Record","type":"object"}`,
          "plain",
        ),
      );
    });

    test("identical records wanting one key share it", () => {
      expectSchema(
        t("C", readTypedefs({ P, I: { b: "P" }, C: { a_b: "P", a: "I" } })),
        parseJson(
          `{"$defs":{"a":{"properties":{"b":{"$ref":"#/$defs/a_b"}},"required":["b"],"title":"a","type":"object"},"a_b":${defX("a_b")}},"properties":{"a_b":{"$ref":"#/$defs/a_b"},"a":{"$ref":"#/$defs/a"}},"required":["a_b","a"],"title":"Record","type":"object"}`,
          "plain",
        ),
      );
    });

    test("a numbered key skips a key another record wants", () => {
      const schema = schemaOf(
        t("C", readTypedefs({ P, Q, I: { b: "Q" }, C: { a_b: "P", a: "I", a_b__1: "Q" } })),
      );
      expect(obj(schema.properties).a_b).toStrictEqual({ $ref: "#/$defs/a_b__2" });
      expect(obj(schema.properties).a_b__1).toStrictEqual({ $ref: "#/$defs/a_b__1" });
      const defs = obj(schema.$defs);
      expect(obj(defs.a).properties).toStrictEqual({ b: { $ref: "#/$defs/a_b__3" } });
      expect(Object.keys(defs)).toEqual(["a", "a_b__1", "a_b__2", "a_b__3"]);
      expect(Object.values(defs).map((definition) => obj(definition).title)).toEqual([
        "a",
        "a_b__1",
        "a_b",
        "a_b",
      ]);
    });

    test("a/b beside a_b clash after normalization, though only their titles differ", () => {
      expectSchema(
        t("C", readTypedefs({ P, C: { "a/b": "P", a_b: "P" } })),
        parseJson(
          `{"$defs":{"a_b__1":${defX("a/b")},"a_b__2":${defX("a_b")}},"properties":{"a/b":{"$ref":"#/$defs/a_b__1"},"a_b":{"$ref":"#/$defs/a_b__2"}},"required":["a/b","a_b"],"title":"Record","type":"object"}`,
          "plain",
        ),
      );
    });

    test("the root position's Record takes part in clashes", () => {
      expectSchema(
        t("Optional(C)", readTypedefs({ P, C: { Record: "P" } })),
        parseJson(
          `{"$defs":{"Record__1":${defX("Record")},"Record__2":{"properties":{"Record":{"$ref":"#/$defs/Record__1"}},"required":["Record"],"title":"Record","type":"object"}},"properties":{"value":{"anyOf":[{"$ref":"#/$defs/Record__2"},{"type":"null"}]}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
          "plain",
        ),
      );
    });

    test("integer-like $defs keys come first, then the others sorted", () => {
      const schema = schemaOf(t("C", readTypedefs({ P, C: { "10": "P", "2": "P", b: "P" } })));
      expect(Object.keys(obj(schema.$defs))).toEqual(["2", "10", "b"]);
    });

    test("a __proto__ record field's definition survives the key sort", () => {
      const typ = new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([["__proto__", t("P", readTypedefs({ P }))]]),
        required: new Set(["__proto__"]),
      });
      const schema = schemaOf(typ);
      expect(obj(schema.properties).__proto__).toStrictEqual({ $ref: "#/$defs/__proto__" });
      const defs = obj(schema.$defs);
      expect(Object.hasOwn(defs, "__proto__")).toBe(true);
      expect(Object.getPrototypeOf(defs)).toBe(Object.prototype);
      expect(defs.__proto__).toStrictEqual(parseJson(defX("__proto__"), "plain"));
    });
  });

  test("file and None have no structured mapping", () => {
    expect(() => typeToSchema(Type.scalar(ValueKind.FILE))).toThrow(
      new RangeError("type_to_schema: no structured mapping for segment type file"),
    );
    expect(() => typeToSchema(Type.scalar(ValueKind.NONE))).toThrow(
      new RangeError("type_to_schema: no structured mapping for segment type None"),
    );
    expect(() => typeToSchema(record({ f: Type.scalar(ValueKind.FILE) }, ["f"]))).toThrow(
      RangeError,
    );
  });
});

// ---------- an AGENT with an `Any` output, through the runner ----------

describe("an agent with an Any output loads and returns the value", () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-type-to-schema-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));
  const flow = `
id: guess
name: guess
nodes:
  a:
    kind: agent
    prompt: say anything
    output: !Any
output: \${a.output}
`;

  test.each<[string, unknown]>([
    ['"hello"', "hello"],
    ["3", 3n],
    ["true", true],
    ['[1, "a"]', [1n, "a"]],
  ])("%s", async (reply, value) => {
    const result = await runFlow(await loadFlow(flow), {}, {
      llmConfig: { provider: "openai", model: "gpt-5.5" },
      models: serving(native(`{"value": ${reply}}`)),
      env: hostEnv(dir),
    });
    expect(result.failure).toBeNull();
    expect(result.output).toEqual(value);
  });
});

describe("a tagged union reply, through checkReply", () => {
  const typ = t("Union(Buy, Sell)", TRADES);

  test("a reply without its tag, with a non-string tag or with an unknown tag is refused at the tag", () => {
    expect(refused(typ, { value: { qty: 1n } }).message).toStartWith(
      "structured output at action: missing tag field 'action'",
    );
    expect(refused(typ, { value: { action: 3n, qty: 1n } }).message).toStartWith(
      "structured output at action: 3 is not a tag of the union",
    );
    expect(refused(typ, { value: { action: "wait", qty: 1n } }).message).toStartWith(
      'structured output at action: "wait" is not a tag of the union',
    );
  });

  test("many items of a large tagged union read member fields a bounded number of times per item", () => {
    // Counting field reads, not time: normalizing each item must not look its member up by
    // scanning every member.
    const gets = { count: 0 };
    class CountingMap extends Map<string, Type> {
      override get(key: string): Type | undefined {
        gets.count += 1;
        return super.get(key);
      }
    }
    const size = 50;
    const members = Array.from({ length: size }, (_, i) => {
      return new Type({
        kind: ValueKind.OBJECT,
        fields: new CountingMap([
          ["tag", new Type({ kind: ValueKind.STRING, tags: new Set([`t${i}`]) })],
          [`f${i}`, Type.scalar(ValueKind.INTEGER)],
        ]),
        required: new Set(["tag", `f${i}`]),
      });
    });
    const union = new Type({ kind: ValueKind.OBJECT, members });
    const list = new Type({ kind: ValueKind.LIST_OBJECT, element: union });
    const items = 1000;
    const replies = Array.from({ length: items }, (_, i) => {
      const k = i % size;
      return { tag: `t${k}`, [`f${k}`]: BigInt(i) };
    });
    expect(checkReply({ items: replies }, list)).toEqual(replies);
    expect(gets.count).toBeLessThan(items * 12 + size * size);
  });
});
