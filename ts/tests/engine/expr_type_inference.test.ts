/**
 * Conservative static type inference over unified expression trees, a port of
 * `tests/engine/test_expr_type_inference.py`, plus `inf`, which is a float.
 */

import { expect, test } from "bun:test";

import { elementType, inferExprType } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";
import { listKindForElement, typeEquals } from "../../src/typesys/values.ts";

/** A reference lookup no test here should reach. */
function unexpectedRef(path: string): Type | null {
  throw new Error(`inference unexpectedly resolved reference '${path}'`);
}

test.each([
  ["1", ValueKind.INTEGER],
  ["1.5", ValueKind.NUMBER],
  ['"text"', ValueKind.STRING],
  ["true", ValueKind.BOOLEAN],
  ["false", ValueKind.BOOLEAN],
  ["null", ValueKind.NONE],
  ["none", ValueKind.NONE],
  ["inf", ValueKind.NUMBER],
] as [string, ValueKind][])("the literal %s has a proven scalar type", (source, kind) => {
  expect(inferExprType(parseExpr(source), unexpectedRef)).toEqual(Type.scalar(kind));
});

test("an exact reference uses the injected dotted-path type", () => {
  const expected = Type.scalar(ValueKind.STRING);
  const paths: string[] = [];
  const inferred = inferExprType(parseExpr("node#0.output.answer"), (path) => {
    paths.push(path);
    return expected;
  });
  expect(inferred).toBe(expected);
  expect(paths).toEqual(["node#0.output.answer"]);
});

test("an unknown exact reference stays unknown", () => {
  expect(inferExprType(parseExpr("missing.output"), () => null)).toBeNull();
});

class ReferenceFailure extends Error {}

test("a reference callback's exception propagates", () => {
  const fail = (): Type | null => {
    throw new ReferenceFailure("lookup failed");
  };
  expect(() => inferExprType(parseExpr("producer.output"), fail)).toThrow(ReferenceFailure);
  expect(() => inferExprType(parseExpr("producer.output"), fail)).toThrow("lookup failed");
});

test.each(["left == right", "item in items", "item not in items", "not value", "left and right", "left or right"])(
  "%s is a non-nullable bool",
  (source) => {
    const inferred = inferExprType(parseExpr(source), unexpectedRef);
    expect(inferred).toEqual(Type.scalar(ValueKind.BOOLEAN));
    expect(inferred!.nullable).toBe(false);
  },
);

test.each([
  "left + right",
  "left - right",
  "left * right",
  "left / right",
  "left % right",
  "left ** right",
  "-value",
  "[left, right]",
  "left | right",
  "left :- right",
  'left :? "required"',
  "upper(value)",
])("%s stays unknown", (source) => {
  expect(inferExprType(parseExpr(source), unexpectedRef)).toBeNull();
});

test("a builtin call is not mistaken for a reference", () => {
  const calls: string[] = [];
  const inferred = inferExprType(parseExpr("upper(value)"), (path) => {
    calls.push(path);
    return Type.scalar(ValueKind.STRING);
  });
  expect(inferred).toBeNull();
  expect(calls).toEqual([]);
});

test("a ?. read is Optional of its field", () => {
  const str = Type.scalar(ValueKind.STRING);
  expect(inferExprType(parseExpr("a.output?.k"), () => str)?.nullable).toBe(true);
  expect(inferExprType(parseExpr("a.output.k"), () => str)?.nullable).toBe(false);
  const Rec = new Type({ kind: ValueKind.OBJECT, fields: new Map([["k", str]]), required: new Set(["k"]) });
  expect(inferExprType(parseExpr("R(k='x').k"), () => null, (n) => (n === "R" ? Rec : null))?.nullable).toBe(false);
  expect(inferExprType(parseExpr("R(k='x')?.k"), () => null, (n) => (n === "R" ? Rec : null))?.nullable).toBe(true);
});

const str = Type.scalar(ValueKind.STRING);
const listOf = (e: Type) => new Type({ kind: listKindForElement(e), element: e });
const Msg = new Type({ kind: ValueKind.OBJECT, fields: new Map([["text", str]]), required: new Set(["text"]) });
const Other = new Type({ kind: ValueKind.OBJECT, fields: new Map([["n", str]]), required: new Set(["n"]) });

test.each([
  [listOf(Msg), Msg, false],
  [listOf(Msg.replace({ nullable: true })), Msg, true],
  [listOf(Msg).replace({ nullable: true }), Msg, false],
  [new Type({ kind: ValueKind.ANY }), null, false],
  [new Type({ kind: listKindForElement(str) }), null, false],
  [null, null, false],
  // A union's members are records, never lists, so iterating one has no element type.
  [new Type({ kind: ValueKind.OBJECT, variants: [Msg, Other] }), null, false],
] as [Type | null, Type | null, boolean][])("elementType(%#)", (iterable, element, nullable) => {
  const got = elementType(iterable);
  if (element === null) expect(got).toBeNull();
  else {
    expect(typeEquals(got!.replace({ nullable: false }), element)).toBe(true);
    expect(got!.nullable).toBe(nullable);
  }
});
