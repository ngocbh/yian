/**
 * Stream events: argument parsing, a port of `tests/llm/test_events.py`, plus JavaScript's own
 * cases: ints of any size and raw control characters inside strings.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import { parseArguments } from "../../src/llm/events.ts";

afterEach(() => install(BUILTIN));

describe("parseArguments", () => {
  test("empty text is an empty object", () => {
    expect(parseArguments("")).toEqual([{}, null]);
  });

  test("object", () => {
    expect(parseArguments('{"a": [1, 2.5, null, true]}')).toEqual([{ a: [1n, 2.5, null, true] }, null]);
  });

  test.each(["[1, 2]", "{bad", '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '"s"', "1e400"])(
    "rejects %p",
    (text) => {
      expect(parseArguments(text)).toEqual([null, text]);
    },
  );

  test("a 400-digit int parses as a bigint", () => {
    const digits = "9".repeat(400);
    expect(parseArguments(`{"n": -${digits}}`)).toEqual([{ n: -BigInt(digits) }, null]);
  });

  test("nesting limit", () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    expect(parseArguments('{"a": {"b": [1]}}')).toEqual([{ a: { b: [1n] } }, null]);
    expect(parseArguments('{"a": {"b": [[1]]}}')).toEqual([null, '{"a": {"b": [[1]]}}']);
  });

  test.each(['{"a": "\\ud800"}', '{"\\udfff": 1}', '{"a": "\ud800"}'])("unpaired surrogate %p is a failure", (text) => {
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("escaped surrogate pair parses", () => {
    expect(parseArguments('{"a": "\\ud83d\\ude00"}')).toEqual([{ a: "\u{1f600}" }, null]);
  });

  test("deep brackets fail without a stack overflow", () => {
    let text = "[".repeat(100_000);
    expect(parseArguments(text)).toEqual([null, text]);
    text = '{"a":'.repeat(100_000);
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("raw tab and newline inside a string parse", () => {
    expect(parseArguments('{"a": "x\ty\nz"}')).toEqual([{ a: "x\ty\nz" }, null]);
  });

  test("raw newlines between tokens parse", () => {
    expect(parseArguments('{\n"a":\r\n1\n}\n')).toEqual([{ a: 1n }, null]);
  });

  test("a raw control character outside a string fails", () => {
    const text = '{"a": 1\u0001}';
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("an escaped backslash before a closing quote ends the string", () => {
    expect(parseArguments('{"a": "\\\\", "b": "\t"}')).toEqual([{ a: "\\", b: "\t" }, null]);
    // Read as still inside the string, the newline after the closing quote would be escaped and fail.
    expect(parseArguments('{"a": "\\\\"\n}')).toEqual([{ a: "\\" }, null]);
  });

  test("a duplicate key fails", () => {
    const text = '{"a": 1, "a": 2}';
    expect(parseArguments(text)).toEqual([null, text]);
  });
});
