/**
 * Load-time compatibility of `Union(...)` types in `typesCompatible`: a port of every test of
 * `tests/engine/test_union_compat.py`.
 */

import { expect, test } from "bun:test";

import { typesCompatible } from "../../src/compile/validation.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Hold: { action: "Literal(hold)" },
  BuyText: { action: "Literal(buy)", qty: "str" },
  Either: { action: "Literal(buy, sell)", qty: "int", price: "float" },
  Note: { text: "str" },
  Count: { n: "int" },
  Score: { n: "float" },
  Mark: { flag: "bool" },
  SmallBuy: { action: "Literal(buy)", qty: "int(0..10)" },
  LooseBuy: { action: "str", qty: "int" },
  Order: { trade: "Union(Buy, Sell)" },
  WideOrder: { trade: "Union(Buy, Sell, Hold)" },
});

const t = (text: string): Type => typeFor(text, REGISTRY);

test("a union fits a wider union", () => {
  expect(typesCompatible(t("Union(Buy, Sell)"), t("Union(Buy, Sell, Hold)"))).toBe(true);
});

test("a union does not fit a narrower union", () => {
  expect(typesCompatible(t("Union(Buy, Sell, Hold)"), t("Union(Buy, Sell)"))).toBe(false);
});

test("the same tags with an incompatible member do not fit", () => {
  expect(typesCompatible(t("Union(BuyText, Sell)"), t("Union(Buy, Sell)"))).toBe(false);
});

test("a member record fits the union", () => {
  expect(typesCompatible(t("Buy"), t("Union(Buy, Sell)"))).toBe(true);
});

test("a record with a multi-tag Literal does not fit the union", () => {
  expect(typesCompatible(t("Either"), t("Union(Buy, Sell)"))).toBe(false);
});

test("a union does not fit a member record", () => {
  expect(typesCompatible(t("Union(Buy, Sell)"), t("Buy"))).toBe(false);
});

test("a union fits object", () => {
  expect(typesCompatible(t("Union(Buy, Sell)"), Type.scalar(ValueKind.OBJECT))).toBe(true);
});

test("object does not fit a union", () => {
  expect(typesCompatible(Type.scalar(ValueKind.OBJECT), t("Union(Buy, Sell)"))).toBe(false);
});

test("a nullable union does not fit a strict union", () => {
  expect(typesCompatible(t("Optional(Union(Buy, Sell))"), t("Union(Buy, Sell)"))).toBe(false);
});

test("a nullable union fits a nullable union", () => {
  expect(
    typesCompatible(t("Optional(Union(Buy, Sell))"), t("Optional(Union(Buy, Sell, Hold))")),
  ).toBe(true);
});

test("a nullable member record fits a nullable union", () => {
  expect(typesCompatible(t("Optional(Buy)"), t("Optional(Union(Buy, Sell))"))).toBe(true);
});

test("a nullable member record does not fit a strict union", () => {
  expect(typesCompatible(t("Optional(Buy)"), t("Union(Buy, Sell)"))).toBe(false);
});

test("None fits an optional union", () => {
  expect(typesCompatible(Type.scalar(ValueKind.NONE), t("Optional(Union(Buy, Sell))"))).toBe(
    true,
  );
});

test("None does not fit a strict union", () => {
  expect(typesCompatible(Type.scalar(ValueKind.NONE), t("Union(Buy, Sell)"))).toBe(false);
});

test("a scalar does not fit a union", () => {
  expect(typesCompatible(Type.scalar(ValueKind.STRING), t("Union(Buy, Sell)"))).toBe(false);
});

test("a list of unions follows its members", () => {
  expect(typesCompatible(t("list(Union(Buy, Sell))"), t("list(Union(Buy, Sell, Hold))"))).toBe(
    true,
  );
  expect(typesCompatible(t("list(Union(Buy, Sell, Hold))"), t("list(Union(Buy, Sell))"))).toBe(
    false,
  );
});

test("a list of members fits a list of unions", () => {
  expect(typesCompatible(t("list(Buy)"), t("list(Union(Buy, Sell))"))).toBe(true);
});

test("an untagged union fits a wider untagged union", () => {
  expect(typesCompatible(t("Union(Note, Count)"), t("Union(Note, Count, Mark)"))).toBe(true);
  expect(typesCompatible(t("Union(Note, Count, Mark)"), t("Union(Note, Count)"))).toBe(false);
});

test("an untagged union member fits by widening", () => {
  expect(typesCompatible(t("Union(Note, Count)"), t("Union(Note, Score)"))).toBe(true);
  expect(typesCompatible(t("Union(Note, Score)"), t("Union(Note, Count)"))).toBe(false);
});

test("an untagged member record fits the union", () => {
  expect(typesCompatible(t("Count"), t("Union(Note, Count)"))).toBe(true);
  expect(typesCompatible(t("Mark"), t("Union(Note, Count)"))).toBe(false);
});

test("a member record does not fit a bounded member", () => {
  expect(typesCompatible(t("Buy"), t("Union(SmallBuy, Sell)"))).toBe(false);
});

test("a strict union fits a wider optional union", () => {
  expect(typesCompatible(t("Union(Buy, Sell)"), t("Optional(Union(Buy, Sell, Hold))"))).toBe(
    true,
  );
});

test("a list of optional unions does not fit a list of unions", () => {
  expect(
    typesCompatible(t("list(Optional(Union(Buy, Sell)))"), t("list(Union(Buy, Sell))")),
  ).toBe(false);
});

test("a union field follows its members", () => {
  expect(typesCompatible(t("Order"), t("WideOrder"))).toBe(true);
  expect(typesCompatible(t("WideOrder"), t("Order"))).toBe(false);
});

test("a plain str tag does not fit the union", () => {
  expect(typesCompatible(t("LooseBuy"), t("Union(Buy, Sell)"))).toBe(false);
});
