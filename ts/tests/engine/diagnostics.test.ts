/**
 * Contracts for source-neutral diagnostic values and origin resolution, a port of
 * `tests/engine/test_diagnostics.py`, plus the text helpers yian's messages need.
 */

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";

import { compareCodeUnits, reprText } from "../../src/diagnostics.ts";

describe("compareCodeUnits", () => {
  test("sorts by UTF-16 code units", () => {
    // UTF-16 order puts the surrogate pair of U+1F600 before U+E000.
    expect(["b", "a", "B", "\u{1F600}", "\uE000", "aa"].sort(compareCodeUnits)).toEqual([
      "B",
      "a",
      "aa",
      "b",
      "\u{1F600}",
      "\uE000",
    ]);
  });

  test("returns zero for equal strings and a negative number for a smaller one", () => {
    expect(compareCodeUnits("a", "a")).toBe(0);
    expect(Math.sign(compareCodeUnits("a", "b"))).toBe(-1);
  });
});

describe("reprText", () => {
  // `expected` is CPython 3.12.14's `repr(input)`; regenerate with
  // `print(json.dumps([[s, repr(s)] for s in inputs]))`, where `inputs` is the first column.
  const cases: [string, string][] = [
    ["", "''"],
    ["abc", "'abc'"],
    ["it's", "\"it's\""],
    ["say \"hi\"", "'say \"hi\"'"],
    ["both ' and \"", "'both \\' and \"'"],
    ["back\\slash", "'back\\\\slash'"],
    ["tab\there", "'tab\\there'"],
    ["nl\ncr\r", "'nl\\ncr\\r'"],
    ["\u0000\u0001\u001f\u007f", "'\\x00\\x01\\x1f\\x7f'"],
    ["\u0080\u009f\u00a0\u00ad", "'\\x80\\x9f\\xa0\\xad'"],
    ["caf\u00e9", "'caf\u00e9'"],
    ["e\u0301", "'e\u0301'"],
    ["\u200b\u2028\u2029\u3000", "'\\u200b\\u2028\\u2029\\u3000'"],
    ["\u0378", "'\\u0378'"],
    ["\ue000", "'\\ue000'"],
    ["\ud800", "'\\ud800'"],
    ["\udfff", "'\\udfff'"],
    ["\ud83d\ude00", "'\ud83d\ude00'"],
    ["\udbff\udfff", "'\\U0010ffff'"],
    ["\udb40\udc01", "'\\U000e0001'"],
    ["\udb40\udd00", "'\udb40\udd00'"],
    ["\ufeff", "'\\ufeff'"],
    ["\uffff", "'\\uffff'"],
    ["\ud7ff", "'\\ud7ff'"],
    ["\ud840\udc00", "'\ud840\udc00'"],
    ["\ud884\udf4a", "'\ud884\udf4a'"],
    ["\ud884\udf4b", "'\\U0003134b'"],
    ["x' \"y\\", "'x\\' \"y\\\\'"],
  ];

  test.each(cases)("reprText(%j) is %j", (input, expected) => {
    expect(reprText(input)).toBe(expected);
  });

  test("its printable table is CPython 3.12.14's str.isprintable", () => {
    // CPython 3.12.14's
    // hashlib.sha256(''.join('1' if chr(c).isprintable() else '0'
    //                        for c in range(0x110000)).encode()).hexdigest()
    // `\` is printable but escaped, and surrogates are never printable, as in Python.
    const bits: string[] = [];
    for (let cp = 0; cp < 0x110000; cp++) {
      const ch = String.fromCodePoint(cp);
      if (ch === "\\") bits.push("1");
      else if (cp >= 0xd800 && cp <= 0xdfff) bits.push("0");
      else {
        const repr = reprText(ch);
        bits.push(repr.length === ch.length + 2 && repr.slice(1, -1) === ch ? "1" : "0");
      }
    }
    expect(createHash("sha256").update(bits.join("")).digest("hex")).toBe(
      "0326b64c15e91a0583a1d9f8bc53f4bb534adc91a5ae096b55adaf19525ed98d",
    );
  });
});
