/**
 * `typeToSchema`: the JSON Schema an AGENT asks the model to fill for its declared `output:`, a
 * port of the schema halves of `tests/engine/test_type_to_schema.py` and of the schema half of
 * `test_flow_fingerprint_preserves_structural_field_order_used_by_agent_schema`
 * (`tests/engine/test_flow_identity.py`). Python reads a pydantic model's `__name__` and
 * `model_json_schema()`; TS reads the schema and its `title`.
 *
 * Every non-union schema Python yian emits (`model_json_schema()`, captured with pydantic
 * 2.13.5) is pinned exactly: the captured JSON text is read with `parseJson`, so its ints are
 * `bigint` as in TS's schema, and a float bound spelled as an int (`float(-0.5..1)`'s `1`) is a
 * `bigint` on both sides. The question-list type of `nodes/human_input/questions.py` is built by
 * hand.
 *
 * Differences from pydantic, by design: keys are sorted at every level but a `properties` map
 * (pydantic leaves the schema of a field or definition named `properties` unsorted); in
 * `properties`, integer-like field names come first (JS key order), then the others in field
 * order; fields starting with `_` are kept, `__proto__` too (pydantic drops them).
 *
 * TS-only: a root `Optional(str)` gives `null`; titles follow Python's `str.title()` (the
 * expected titles are Python's); `_x` and `__proto__` fields; integer-like field names; a field
 * named `properties`; bounds on a non-numeric scalar; `file` and `None` throwing.
 *
 * Later (T5-11a): the validate and unwrap halves (`model_validate`, `_unwrap`, the
 * `model_validate` half of `test_large_numeric_schema_bounds_match_resolved_type`),
 * `test_union_members_reject_unknown_fields_and_unwrap_to_the_member_dict`,
 * `test_null_at_an_any_position_is_reported_at_that_position` and
 * `test_absent_optional_any_record_field_reads_none`. Later (T5-11a, T3-22):
 * `test_agent_with_any_output_loads_and_returns_the_value`. Later: the union rows
 * (`test_tagged_union_schema_is_any_of_closed_members_with_const_tags` through
 * `test_list_of_union_schema_gets_member_schemas`, the shape half of
 * `test_records_nested_in_union_members_reject_unknown_fields`, and the union captures), with
 * union schemas.
 */

import { describe, expect, test } from "bun:test";

import { compareCodeUnits } from "../../src/diagnostics.ts";
import { typeToSchema } from "../../src/nodes/agent/structured.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  isListKind,
  Type,
  TypeCheckError,
  ValueKind,
} from "../../src/typesys/values.ts";

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

/** The question-list type of Python's `nodes/human_input/questions.py`, built by hand. */
function questionListType(): Type {
  const option = record(
    { label: Type.scalar(ValueKind.STRING), description: Type.scalar(ValueKind.STRING) },
    ["label", "description"],
  );
  const question = record(
    {
      question: Type.scalar(ValueKind.STRING),
      header: Type.scalar(ValueKind.STRING),
      options: new Type({ kind: ValueKind.LIST_OBJECT, element: option }),
      multi_select: Type.scalar(ValueKind.BOOLEAN),
    },
    ["question", "header"],
  );
  return new Type({ kind: ValueKind.LIST_OBJECT, element: question });
}

// Python's `model_json_schema()` per type, as captured (compact JSON, pydantic's key order).
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
    `{"properties":{"value":{"title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
  ],
  [
    "datetime",
    () => t("datetime"),
    `{"properties":{"value":{"title":"Value","type":"string"}},"required":["value"],"title":"ScalarWrapper","type":"object"}`,
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
    `{"$defs":{"meta":{"properties":{"due_date":{"title":"Due Date","type":"string"},"score":{"anyOf":[{"maximum":10,"minimum":0,"type":"integer"},{"type":"null"}],"default":null,"title":"Score"}},"required":["due_date"],"title":"meta","type":"object"},"meta_list_item":{"properties":{"due_date":{"title":"Due Date","type":"string"},"score":{"anyOf":[{"maximum":10,"minimum":0,"type":"integer"},{"type":"null"}],"default":null,"title":"Score"}},"required":["due_date"],"title":"meta_list_item","type":"object"}},"properties":{"lesson_id":{"title":"Lesson Id","type":"string"},"meta":{"$ref":"#/$defs/meta"},"meta_list":{"items":{"$ref":"#/$defs/meta_list_item"},"title":"Meta List","type":"array"}},"required":["lesson_id","meta","meta_list"],"title":"Record","type":"object"}`,
  ],
  [
    // pydantic leaves the `properties` field's schema unsorted; TS sorts it (checked by key order).
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
    // pydantic leaves the `properties` definition unsorted; TS sorts it (checked by key order).
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
];

describe("Python's captured schemas", () => {
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

test("test_bare_str_returns_none", () => {
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

describe("TS-only", () => {
  test("a root Optional(str) gives null; a nullable tagged string is wrapped", () => {
    expect(typeToSchema(t("Optional(str)"))).toBeNull();
    expect(schemaOf(t("Optional(Literal(a))")).title).toBe("ScalarWrapper");
  });

  test("titles follow Python's str.title(), then _ to space, then strip", () => {
    // Expected titles captured from Python: `name.title().replace("_", " ").strip()`.
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
