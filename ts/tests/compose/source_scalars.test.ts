/**
 * Plain scalars read with YAML 1.2 core rules, keeping timestamps and merge keys: a port of
 * `tests/compose/test_source_scalars.py`.
 *
 * TS-era differences: dates stay text, checked with yian's date grammar; `.nan` and an
 * overflowing float are refused; ints are `bigint` of any length, so an int of any length
 * replaces Python's digit-limit tests. The `SourceDocument` rows come with `document.ts`.
 * `test_pyyaml_globals_are_untouched` has no counterpart, since eemeli/yaml has no global
 * tables.
 */

import { describe, expect, test } from "bun:test";

import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { readPlainScalar } from "../../src/compose/source/literals.ts";
import {
  BOOL_TAG,
  constructScalar,
  FLOAT_TAG,
  FlowLoaderError,
  INT_TAG,
  NULL_TAG,
  ScalarValueError,
  type ScalarValue,
  STR_TAG,
  TIMESTAMP_TAG,
} from "../../src/compose/source/scalars.ts";
import { reprText } from "../../src/diagnostics.ts";
import { parseDatetime } from "../../src/typesys/temporal.ts";

const DATETIME_MSG = "datetime must look like 2026-05-21T14:30[:00][Z|+07:00]";

const WORDS = ["on", "off", "yes", "no"].flatMap((base) => [
  base,
  base[0]!.toUpperCase() + base.slice(1),
  base.toUpperCase(),
]);

const ROWS: [string, ScalarValue][] = [
  ...WORDS.map((word): [string, ScalarValue] => [word, word]),
  ...["y", "n", "Y", "N", "oN"].map((word): [string, ScalarValue] => [word, word]),
  ["true", true],
  ["True", true],
  ["TRUE", true],
  ["false", false],
  ["False", false],
  ["FALSE", false],
  ["1:30", "1:30"],
  ["017", 17n],
  ["0o17", 15n],
  ["0x1F", 31n],
  ["1_000", "1_000"],
  ["0b101", "0b101"],
  ["1e3", 1000],
  ["-1.5E-2", -0.015],
  [".5", 0.5],
  ["+12", 12n],
  ["2026-10-14", "2026-10-14"],
  ["2026-10-14T10:00:00Z", "2026-10-14T10:00:00Z"],
  ["~", null],
  ["null", null],
  ["NULL", null],
  ["=", "="],
  [".inf", Infinity],
  ["-.inf", -Infinity],
  ["+.Inf", Infinity],
  [".INF", Infinity],
  ["-.INF", -Infinity],
  ["-.nan", "-.nan"],
  // `toBe` compares with `Object.is`, so this row fails on `-0`.
  ["-0.0", 0],
  ["1e-400", 0],
  ["2026-10-14 10:00:00", "2026-10-14 10:00:00"],
];

/** Return the error `run` throws, failing unless it is an instance of `kind`. */
function thrown<T extends Error>(kind: abstract new (...args: never[]) => T, run: () => unknown): T {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(kind);
    return error as T;
  }
  throw new Error(`expected a ${kind.name}`);
}

describe("readPlainScalar", () => {
  test.each<[string, ScalarValue]>([
    ...ROWS,
    ["", null],
    ["None", "None"],
    ["none", "none"],
    ["NONE", "NONE"],
    ["tRUE", "tRUE"],
    ["<<", "<<"],
  ])("readPlainScalar follows the table: %p", (text, expected) => {
    expect(readPlainScalar(text)).toBe(expected);
  });

  test.each([".nan", ".NaN", ".NAN"])("readPlainScalar refuses NaN: %p", (text) => {
    const error = thrown(SourceSyntaxError, () => readPlainScalar(text));
    expect(error.message).toBe("NaN is not a yian value");
    expect(error.range).toBeNull();
  });

  test.each(["1e400", "-1e400"])("readPlainScalar refuses an overflowing float: %p", (text) => {
    const error = thrown(SourceSyntaxError, () => readPlainScalar(text));
    expect(error.message).toBe("number is too large for a float");
  });

  test.each(["a: b", "[1]", "{a: 1}", "- x", "a #b", "&a", "*a", "!x", "|", "hello world", "1\n"])(
    "readPlainScalar keeps YAML structure as text: %p",
    (text) => {
      expect(readPlainScalar(text)).toBe(text);
    },
  );

  test.each([
    ["2026-13-45", "not a valid date: month must be in 1..12"],
    // YAML's spaced offset is not in yian's datetime grammar.
    ["2026-01-01 10:00:00 +05:00", DATETIME_MSG],
  ])("readPlainScalar rejects an unbuildable form: %p", (text, message) => {
    // The inline-call error adds the quoting fix and must not echo the value.
    expect(thrown(SourceSyntaxError, () => readPlainScalar(text)).message).toBe(message);
  });

  test.each(["+05:99", "+07:60", "+23:60", "+24:00", "+99:00", "+99:99"])(
    "the loader and the typed grammar reject an offset for the same reason: %p",
    (offset) => {
      const text = `2026-05-21T14:30:00${offset}`; // YAML needs the seconds to read a timestamp
      const typed = thrown(RangeError, () => parseDatetime(text));
      const loaded = thrown(SourceSyntaxError, () => readPlainScalar(text));
      expect(loaded.message).toBe(typed.message);
    },
  );

  test("in-range offsets build", () => {
    for (const text of ["2026-01-01T10:00:00+05:59", "2026-01-01T10:00:00-23:59"]) {
      expect(readPlainScalar(text)).toBe(text);
    }
    // Python reads +05:00; yian's datetime grammar needs the minutes.
    const error = thrown(SourceSyntaxError, () => readPlainScalar("2026-01-01T10:00:00+5"));
    expect(error.message).toBe(DATETIME_MSG);
  });

  test("an int of any length reads exactly", () => {
    for (const text of [
      "1".repeat(5000),
      "-" + "1".repeat(5000),
      "0x" + "f".repeat(4000),
      "0o" + "7".repeat(5000),
    ]) {
      expect(readPlainScalar(text)).toBe(BigInt(text));
    }
    expect(readPlainScalar("0".repeat(4999) + "1")).toBe(1n);
    const text = "1".repeat(5000);
    expect(constructScalar(INT_TAG, text, true)).toBe(BigInt(text));
  });
});

describe("constructScalar", () => {
  test.each<[string, string, boolean, ScalarValue]>([
    [INT_TAG, "017", true, 17n],
    [FLOAT_TAG, "1", true, 1],
    [STR_TAG, "1", false, "1"],
    [NULL_TAG, "x", true, null],
  ])("constructScalar under an explicit tag reads a core form: %p %p", (tag, text, plain, expected) => {
    expect(constructScalar(tag, text, plain)).toBe(expected);
  });

  test.each([
    [INT_TAG, "1_000"],
    [INT_TAG, "1.0"],
    [BOOL_TAG, "yes"],
    [BOOL_TAG, "true\n"],
    [FLOAT_TAG, ".inf\n"],
  ])("constructScalar under an explicit tag rejects a non-core form: %p %p", (tag, found) => {
    const error = thrown(FlowLoaderError, () => constructScalar(tag, found, true));
    expect(error).not.toBeInstanceOf(ScalarValueError);
    expect(error.message).toContain(reprText(found));
  });

  test.each([
    [FLOAT_TAG, ".nan", "NaN is not a yian value"],
    [TIMESTAMP_TAG, "2026-13-45", "not a valid date: month must be in 1..12"],
  ])("a refused value says to quote it only when plain: %p %p", (tag, text, reason) => {
    const plain = thrown(ScalarValueError, () => constructScalar(tag, text, true));
    expect(plain.message).toBe(`${reason}; quote it to keep it as text`);
    expect(plain.reason).toBe(reason);
    const tagged = thrown(FlowLoaderError, () => constructScalar(tag, text, false));
    expect(tagged).not.toBeInstanceOf(ScalarValueError);
    expect(tagged.message).toBe(reason);
  });

  test("!!timestamp on non-date text names the text", () => {
    const error = thrown(FlowLoaderError, () => constructScalar(TIMESTAMP_TAG, "hello", true));
    expect(error.message).toBe("expected a date or timestamp, but found 'hello'");
    expect(error.message).not.toContain("quote it");
    const short = thrown(ScalarValueError, () => constructScalar(TIMESTAMP_TAG, "2026-1-1", true));
    expect(short.message).toBe("date must look like 2026-05-21; quote it to keep it as text");
  });

  test("an unknown tag has no constructor", () => {
    const error = thrown(FlowLoaderError, () => constructScalar("!int", "5", true));
    expect(error.message).toBe("could not determine a constructor for the tag '!int'");
  });
});
