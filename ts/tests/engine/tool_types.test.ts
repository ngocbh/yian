/**
 * A tool's JSON Schema translates into a yian `Type` that is never narrower than it, a port of
 * the JSON Schema parts of `tests/engine/test_tool_types.py`.
 *
 * The cases are raw JSON Schema objects, so the same rules cover function tools and MCP tools
 * that only publish a schema. Python ints beyond 2^53 are `bigint`s here.
 */

import { describe, expect, test } from "bun:test";

import { inputTypeFromJsonSchema, typeFromJsonSchema } from "../../src/tools/types.ts";
import { canonicalTypeLabel, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

type Schema = Record<string, unknown>;

function label(schema: Schema, closed = true): string | null {
  const typ = typeFromJsonSchema(schema, { closed });
  return typ === null ? null : canonicalTypeLabel(typ);
}

const PLAIN_OBJECT = Type.scalar(ValueKind.OBJECT);

/** Assert `typ` equals `expected` structurally. */
function expectType(typ: Type | null, expected: Type): void {
  expect(typ).not.toBeNull();
  expect(typeEquals(typ!, expected)).toBe(true);
}

const BUY: Schema = {
  type: "object",
  properties: {
    action: { const: "buy", title: "Action", type: "string" },
    qty: { title: "Qty", type: "integer" },
  },
  required: ["action", "qty"],
  title: "Buy",
};
const SELL: Schema = {
  type: "object",
  properties: {
    action: { const: "sell", title: "Action", type: "string" },
    price: { title: "Price", type: "number" },
  },
  required: ["action", "price"],
  title: "Sell",
};
const TAGGED_UNION: Schema = {
  $defs: { Buy: BUY, Sell: SELL },
  anyOf: [{ $ref: "#/$defs/Sell" }, { $ref: "#/$defs/Buy" }],
};

const LABELS: [Schema, string | null][] = [
  [{ type: "string" }, "str"],
  [{ type: "string", format: "date" }, "str"],
  [{ type: "string", format: "date-time" }, "str"],
  [{ type: "string", format: "email", minLength: 1, pattern: "@" }, "str"],
  [{ enum: ["b", "a"] }, "Literal(a, b)"],
  [{ enum: ["a", "b"], type: "string" }, "Literal(a, b)"],
  [{ const: "buy" }, "Literal(buy)"],
  [{ enum: [1, 2], type: "integer" }, "int"],
  [{ const: 3, type: "number" }, "float"],
  [{ enum: [1, 2] }, null],
  [{ type: "integer" }, "int"],
  [{ type: "number" }, "float"],
  [{ type: "integer", minimum: 1, maximum: 10 }, "int(1..10)"],
  [{ type: "integer", minimum: 1, exclusiveMaximum: 10 }, "int(1..9)"],
  [{ type: "integer", exclusiveMinimum: 0, maximum: 5 }, "int(1..5)"],
  [{ type: "integer", minimum: 0.5, maximum: 3.7 }, "int(1..3)"],
  [{ type: "integer", minimum: 1 }, "int(1..)"],
  [{ type: "integer", maximum: 1 }, "int(..1)"],
  [{ type: "integer", exclusiveMinimum: 0 }, "int(1..)"],
  [{ type: "integer", exclusiveMaximum: 10 }, "int(..9)"],
  [{ type: "integer", minimum: 0.5 }, "int(1..)"],
  [{ type: "integer", maximum: -0.5 }, "int(..-1)"],
  [{ type: "integer", minimum: 0, exclusiveMinimum: 3 }, "int(4..)"],
  [{ type: "number", minimum: 0, maximum: 1.5 }, "float(0..1.5)"],
  [{ type: "number", exclusiveMinimum: 0, maximum: 1 }, "float(0..1)"],
  [{ type: "number", exclusiveMinimum: 0 }, "float(0..)"],
  [{ type: "number", exclusiveMaximum: 2.5 }, "float(..2.5)"],
  [{ type: "number", minimum: 1, exclusiveMinimum: 3 }, "float(3..)"],
  [{ type: "number", exclusiveMinimum: 2n ** 53n + 1n }, "float"],
  [{ type: "number", minimum: 0 }, "float(0..)"],
  [{ type: "number", maximum: 2.5 }, "float(..2.5)"],
  [{ type: "number", minimum: 2n ** 53n + 1n, maximum: 1e300 }, "float(..1e+300)"],
  [{ type: "integer", minimum: 0.5, maximum: 0.7 }, "int"],
  [{ type: "integer", minimum: Infinity }, "int"],
  [{ type: "integer", minimum: true }, "int"],
  [{ type: ["integer", "null"], minimum: 0 }, "Optional(int(0..))"],
  [{ type: "array", items: { type: "integer", maximum: 9 } }, "list(int(..9))"],
  [{ anyOf: [{ type: "integer", minimum: 0 }, { type: "integer" }] }, "int"],
  [{ anyOf: [{ type: "integer", minimum: 0 }, { type: "integer", maximum: -5 }] }, "int"],
  [
    {
      anyOf: [
        { type: "integer", minimum: 0, maximum: 3 },
        { type: "integer", minimum: 5, maximum: 9 },
      ],
    },
    "int(0..9)",
  ],
  [
    {
      anyOf: [{ type: "integer", minimum: 5 }, { type: "integer", minimum: 0 }, { type: "null" }],
    },
    "Optional(int(0..))",
  ],
  [{ anyOf: [{ type: "number", maximum: 1 }, { type: "number", maximum: 2.5 }] }, "float(..2.5)"],
  [{ anyOf: [{ type: "integer", minimum: 0 }, { type: "number" }] }, null],
  [{ type: "boolean" }, "bool"],
  [{ type: "null" }, "None"],
  [{ type: "array", items: { type: "integer" } }, "list(int)"],
  [{ type: "array", items: { enum: ["x", "y"] } }, "list(Literal(x, y))"],
  [{ type: "array", items: { type: "array", items: { type: "string" } } }, "list(list(str))"],
  [
    { type: "array", items: { anyOf: [{ type: "integer" }, { type: "null" }] } },
    "list(Optional(int))",
  ],
  [{ type: "array" }, "list(Optional(Any))"],
  [{ type: "array", items: {} }, "list(Optional(Any))"],
  [{ type: "object" }, "object"],
  [{ type: "object", additionalProperties: { type: "integer" } }, "object"],
  [{ anyOf: [{ type: "integer" }, { type: "null" }] }, "Optional(int)"],
  [{ oneOf: [{ type: "null" }, { type: "string" }] }, "Optional(str)"],
  [{ type: ["integer", "null"] }, "Optional(int)"],
  [{ type: ["integer", "null"], minimum: 0, maximum: 3 }, "Optional(int(0..3))"],
  [{ anyOf: [{ type: "integer" }, { type: "string" }] }, null],
  [{ type: ["integer", "string"] }, null],
  [{ anyOf: [{ type: "object" }, BUY] }, "object"],
  [{ anyOf: [{ type: "integer" }, {}] }, null],
  [{ anyOf: [{ type: "object" }, { type: "integer" }] }, null],
  [{ type: ["object", "string"] }, null],
  [{ anyOf: [BUY, { type: "string" }] }, null],
  [{ allOf: [{ type: "integer" }] }, "int"],
  [{ allOf: [{ type: "integer" }, { minimum: 0 }] }, null],
  [{}, null],
  [{ not: { type: "string" } }, null],
  [{ type: "string", not: { const: "x" } }, "str"],
  [{ type: "array", prefixItems: [{ type: "integer" }] }, "list(Optional(Any))"],
  [
    { type: "array", prefixItems: [{ type: "integer" }], items: { type: "integer" } },
    "list(Optional(Any))",
  ],
  [{ minimum: 0 }, null],
  [
    {
      type: "integer",
      title: "N",
      description: "d",
      default: 1,
      examples: [1],
      multipleOf: 2,
    },
    "int",
  ],
  [{ type: "string", minLength: 1, maxLength: 9 }, "str"],
  [
    { anyOf: [BUY, SELL], discriminator: { propertyName: "action" } },
    'Union({"action":Literal(buy),"qty":int},{"action":Literal(sell),"price":float})',
  ],
  [
    { type: "array", items: { type: "string" }, minItems: 1, maxItems: 3, uniqueItems: true },
    "list(str)",
  ],
];

/** Spell `schema` as one line of JSON, a `bigint` as its digits and `n`. */
function show(schema: Schema): string {
  return JSON.stringify(schema, (_key, value) => (typeof value === "bigint" ? `${value}n` : value));
}

describe("schema translates to label", () => {
  test.each(LABELS.map(([schema, expected]) => [show(schema), schema, expected] as const))(
    "%s",
    (_name, schema, expected) => {
      expect(label(schema)).toBe(expected);
    },
  );
});

test("list kinds follow the type resolver", () => {
  const kind = (items: Schema) =>
    typeFromJsonSchema({ type: "array", items }, { closed: true })!.kind;

  expect(kind({ enum: ["a"] })).toBe(ValueKind.LIST_STRING);
  expect(kind({ type: "object", properties: { a: { type: "integer" } } })).toBe(
    ValueKind.LIST_OBJECT,
  );
});

const OBJECT: Schema = {
  type: "object",
  properties: {
    a: { anyOf: [{ type: "integer" }, { type: "null" }], default: null },
    b: { type: "array", items: { type: "string" } },
    c: { enum: ["x", "y"], type: "string" },
    d: { type: "integer" },
  },
  required: ["a", "b", "c"],
};

test("open object with properties is plain object", () => {
  expectType(typeFromJsonSchema(OBJECT, { closed: false }), PLAIN_OBJECT);
});

describe("closed object is a record", () => {
  test.each([
    [OBJECT, true],
    [{ ...OBJECT, additionalProperties: false }, false],
  ] as [Schema, boolean][])("%j closed=%p", (schema, closed) => {
    const typ = typeFromJsonSchema(schema, { closed });
    expect(typ).not.toBeNull();
    expect(typ!.kind).toBe(ValueKind.OBJECT);
    const fields = [...typ!.fields!].map(([name, field]) => [name, canonicalTypeLabel(field)]);
    expect(Object.fromEntries(fields)).toEqual({
      a: "Optional(int)",
      b: "list(str)",
      c: "Literal(x, y)",
      d: "int",
    });
    // `a` is nullable and `d` is not listed in JSON `required`.
    expect(typ!.required).toEqual(new Set(["b", "c"]));
  });
});

test("closed object with an untranslatable property is plain object", () => {
  const schema = { type: "object", properties: { a: {} }, required: ["a"] };
  expectType(typeFromJsonSchema(schema, { closed: true }), PLAIN_OBJECT);
});

test("open object with an untranslatable property is plain object", () => {
  const schema = { type: "object", properties: { a: {} } };
  expectType(typeFromJsonSchema(schema, { closed: false }), PLAIN_OBJECT);
});

test("nested objects follow the direction", () => {
  const schema = {
    type: "object",
    properties: { inner: { type: "object", properties: { x: { type: "integer" } } } },
    required: ["inner"],
    additionalProperties: false,
  };
  expect(label(schema, true)).toBe('{"inner":{"x":int}}');
  expect(label(schema, false)).toBe('{"inner":object}');
});

test("tagged union of $defs is a union of records when closed", () => {
  const typ = typeFromJsonSchema(TAGGED_UNION, { closed: true });
  expect(typ?.variants).toBeDefined();
  expect(typ!.variants!.map(canonicalTypeLabel)).toEqual([
    '{"action":Literal(buy),"qty":int}',
    '{"action":Literal(sell),"price":float}',
  ]);
  for (const member of typ!.variants!) {
    expect(member.required).toEqual(new Set(member.fields!.keys()));
  }
});

test("tagged union of $defs is plain object when open", () => {
  expectType(typeFromJsonSchema(TAGGED_UNION, { closed: false }), PLAIN_OBJECT);
});

test("nullable union of records", () => {
  const anyOf = [...(TAGGED_UNION.anyOf as unknown[]), { type: "null" }];
  const schema = { ...TAGGED_UNION, anyOf };
  const typ = typeFromJsonSchema(schema, { closed: true });
  expect(typ).not.toBeNull();
  expect(typ!.nullable).toBe(true);
  expect(typ!.variants).toBeDefined();
  expect(typ!.variants!.some((member) => member.nullable)).toBe(false);
});

test("definitions refs resolve against root", () => {
  const root = { definitions: { N: { type: "integer" } }, type: "array" };
  expectType(
    typeFromJsonSchema({ $ref: "#/definitions/N" }, { closed: true, root }),
    Type.scalar(ValueKind.INTEGER),
  );
});

test("record with a self-referential field is plain object", () => {
  const schema = {
    $defs: {
      Node: {
        type: "object",
        properties: { child: { $ref: "#/$defs/Node" } },
        additionalProperties: false,
      },
    },
    $ref: "#/$defs/Node",
  };
  expectType(typeFromJsonSchema(schema, { closed: true }), PLAIN_OBJECT);
});

test("ref to itself has no type", () => {
  const schema = { $defs: { A: { $ref: "#/$defs/A" } }, $ref: "#/$defs/A" };
  expect(typeFromJsonSchema(schema, { closed: true })).toBeNull();
});

describe("unresolvable ref has no type", () => {
  test.each(["#/$defs/Missing", "https://example.com/s.json", "#"])("%s", (ref) => {
    expect(typeFromJsonSchema({ $defs: {}, $ref: ref }, { closed: true })).toBeNull();
  });
});

test("an invalid shape has no type", () => {
  // An empty property name is not a valid record field.
  const schema = { type: "object", properties: { "": { type: "integer" } } };
  expect(typeFromJsonSchema(schema, { closed: true })).toBeNull();
});

// The schema pydantic generates for a field `v: Union[_Inner, str]`, `_Inner` holding `x: int`.
const UNION_OF_A_MODEL_AND_A_SCALAR: Schema = {
  $defs: {
    _Inner: {
      properties: { x: { title: "X", type: "integer" } },
      required: ["x"],
      title: "_Inner",
      type: "object",
    },
  },
  anyOf: [{ $ref: "#/$defs/_Inner" }, { type: "string" }],
  title: "V",
};

describe("pydantic union of a model and a scalar has no type", () => {
  test.each([false, true])("closed=%p", (closed) => {
    expect(typeFromJsonSchema(UNION_OF_A_MODEL_AND_A_SCALAR, { closed })).toBeNull();
  });
});

describe("huge bounds never raise", () => {
  const huge = 10n ** 400n;
  test.each([
    [
      "integer",
      { type: "integer", minimum: huge, maximum: 10n ** 401n },
      `int(${huge}..${10n ** 401n})`,
    ],
    ["number", { type: "number", minimum: 0, maximum: huge }, "float(0..)"],
    [
      "exclusive-integer",
      { type: "integer", minimum: 0, exclusiveMaximum: huge },
      `int(0..${huge - 1n})`,
    ],
    ["one-sided-integer", { type: "integer", minimum: huge }, `int(${huge}..)`],
    ["one-sided-number", { type: "number", maximum: -huge }, "float"],
  ] as [string, Schema, string][])("%s", (_id, schema, expected) => {
    expect(label(schema)).toBe(expected);
  });
});

test("malformed required is plain object", () => {
  const schema = {
    type: "object",
    properties: { a: { type: "integer" } },
    required: [{ x: 1 }],
  };
  expectType(typeFromJsonSchema(schema, { closed: true }), PLAIN_OBJECT);
});

describe("closed object that admits other keys is plain object", () => {
  test.each([
    { additionalProperties: true },
    { additionalProperties: { type: "integer" } },
    { patternProperties: { "^x_": { type: "integer" } } },
    { required: ["a", "z"] },
  ] as Schema[])("%j", (extra) => {
    const schema = { type: "object", properties: { a: { type: "integer" } }, ...extra };
    expectType(typeFromJsonSchema(schema, { closed: true }), PLAIN_OBJECT);
  });
});

test("too deeply nested input schema is unchecked", () => {
  let deep: Schema = { type: "integer" };
  for (let i = 0; i < 100_000; i++) deep = { type: "array", items: deep };
  expect(inputTypeFromJsonSchema(deep)).toBeNull();
});
