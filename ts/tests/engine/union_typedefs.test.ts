/**
 * `Union(...)` in the type grammar, a port of the parsing tests of
 * `tests/engine/test_union_typedefs.py`; resolution, `readTypedefs` and flow loading come
 * later.
 */

import { describe, expect, test } from "bun:test";

import { parseType } from "../../src/typesys/types.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";

/** Return the message of the `TypeCheckError` `parseType(text)` throws. */
function parseError(text: string): string {
  try {
    parseType(text);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error(`parseType(${JSON.stringify(text)}) did not throw`);
}

describe("parsing", () => {
  test("parses a union", () => {
    expect(parseType("Union(Buy, Sell)")).toEqual({
      kind: "union",
      members: [
        { kind: "ref", name: "Buy" },
        { kind: "ref", name: "Sell" },
      ],
    });
  });

  test("parses a union inside Optional", () => {
    expect(parseType("Optional(Union(Buy, Sell))")).toEqual({
      kind: "optional",
      inner: {
        kind: "union",
        members: [
          { kind: "ref", name: "Buy" },
          { kind: "ref", name: "Sell" },
        ],
      },
    });
  });

  test.each(["Buy | Sell", "Optional(Buy | Sell)"])("the bar operator hints Union: %p", (text) => {
    expect(parseError(text)).toContain("use Union(Buy, Sell) (| is not a type operator)");
  });

  test("a bar chain hints every member", () => {
    expect(parseError("Buy | Sell | Hold")).toBe(
      "malformed type expression 'Buy | Sell | Hold': " +
        "use Union(Buy, Sell, Hold) (| is not a type operator)",
    );
  });

  test("a None member hints Optional", () => {
    expect(parseError("Union(Buy, None)")).toBe(
      "None is not a union member in 'Union(Buy, None)'; use Optional(Union(...))",
    );
  });

  test.each([
    ["Union()", "malformed type expression 'Union()': Union(...) needs a type"],
    ["Union(())", "malformed type expression 'Union(())'"],
  ])("an empty union is malformed: %p", (text, message) => {
    expect(parseError(text)).toBe(message);
  });
});
