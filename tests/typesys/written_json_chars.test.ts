/**
 * `writtenJsonChars` gives the characters (code points) `writeAsJson` writes, without writing
 * them, for every indent and every kind of value, escapes and astral characters included.
 */

import { expect, test } from "bun:test";

import { FileRef, InboxRef, type Value } from "../../src/typesys/index.ts";
import { codePointCount, writeAsJson, writtenJsonChars } from "../../src/typesys/values.ts";

const TEXTS = [
  "",
  "plain",
  'a "quoted" \\ back\\slash',
  "\b\f\n\r\t",
  "\u0000\u0001\u001f\u007f",
  "héllo 😀 🎉",
  "  ",
  "lone \ud800 high and \udc00 low",
  "\ud83d",
];

const VALUES: Value[] = [
  null,
  true,
  false,
  0n,
  -12345678901234567890n,
  10n ** 300n,
  1.5,
  -0.25,
  1e21,
  5e-324,
  Infinity,
  ...TEXTS,
  [],
  {},
  [1n, "two", [3.5, [null, {}]], []],
  { a: 1n, "b c": ["x", { d: [] }], [TEXTS[3]!]: TEXTS[2]!, "😀": {} },
  new FileRef({ uri: "s3://b/k", mime: "text/plain", name: 'n"1' }),
  [new FileRef({ uri: "f" }), { inbox: new InboxRef("notes") }],
  TEXTS,
];

test.each([null, 0, 1, 2, 10])("the count matches the text written with indent %p", (indent) => {
  for (const value of VALUES) {
    expect(writtenJsonChars(value, indent)).toBe(codePointCount(writeAsJson(value, indent)));
  }
});

test("a value that shares its parts is counted once per part and level, not once per copy", () => {
  // 60 levels that each hold the level below twice: 2^60 copies of "x" when written out.
  let value: Value = ["x"];
  for (let i = 0; i < 60; i++) value = [value, { k: value }];
  const started = performance.now();
  for (const indent of [null, 2]) expect(writtenJsonChars(value, indent)).toBeGreaterThan(2 ** 60);
  expect(performance.now() - started).toBeLessThan(1_000);
  let small: Value = ["x"];
  for (let i = 0; i < 3; i++) small = [small, { k: small }];
  for (const indent of [null, 2]) {
    expect(writtenJsonChars(small, indent)).toBe(codePointCount(writeAsJson(small, indent)));
  }
});
