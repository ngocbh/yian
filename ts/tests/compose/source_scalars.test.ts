/**
 * Plain scalars read with YAML 1.2 core rules, keeping timestamps and merge keys: a port of
 * `tests/compose/test_source_scalars.py`.
 *
 * TS-era differences: dates stay text, checked with yian's date grammar; `.nan` and an
 * overflowing float are refused; ints are `bigint`, so 'an int of any length reads exactly'
 * replaces Python's digit-limit tests. A literal's mapping keys must be text, local and
 * unknown tags are refused, and `! 5` is the text `"5"`. A document keeps such a key as written
 * and reports it as `nonTextKey`. Lone CRs are line breaks for both entry points. Both entry
 * points, `SourceDocument.parse` and `parseYamlLiteral`, read the shared rows, explicit core
 * tags included. The literal's merge rows quote Python's `017:` key; the document's keep it.
 * `test_pyyaml_globals_are_untouched` has no counterpart, since eemeli/yaml has no global
 * tables. The harvested `yaml/compose_source_scalars__*` cases are pinned at the end through
 * `parseSourceDocument`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { parseYamlLiteral, readPlainScalar } from "../../src/compose/source/literals.ts";
import {
  BOOL_TAG,
  composeFlowDocument,
  constructScalar,
  FLOAT_TAG,
  FlowLoaderError,
  INT_TAG,
  MAP_TAG,
  MERGE_TAG,
  NULL_TAG,
  ScalarValueError,
  type ScalarValue,
  SEQ_TAG,
  STR_TAG,
  TIMESTAMP_TAG,
} from "../../src/compose/source/scalars.ts";
import { reprText, SourceRange } from "../../src/diagnostics.ts";
import { isRecord, type Value } from "../../src/typesys/index.ts";
import { forget } from "../../src/defaults.ts";
import { parseDatetime } from "../../src/typesys/temporal.ts";
import { runTextCase } from "./declarations.ts";

afterEach(() => forget());

const DATETIME_MSG = "datetime must look like 2026-05-21T14:30[:00][Z|+07:00]";
const QUOTE_FIX = "; quote it to keep it as text";

const WORDS = ["on", "off", "yes", "no"].flatMap((base) => [
  base,
  base[0]!.toUpperCase() + base.slice(1),
  base.toUpperCase(),
]);

const LONG_INTS = [
  "1".repeat(5000),
  "-" + "1".repeat(5000),
  "0x" + "f".repeat(4000),
  "0o" + "7".repeat(5000),
];

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
function thrown<T extends Error>(
  kind: abstract new (...args: never[]) => T,
  run: () => unknown,
): T {
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
  ])("follows the table: %p", (text, expected) => {
    expect(readPlainScalar(text)).toBe(expected);
  });

  test.each([".nan", ".NaN", ".NAN"])("refuses NaN: %p", (text) => {
    const error = thrown(SourceSyntaxError, () => readPlainScalar(text));
    expect(error.message).toBe("NaN is not a yian value");
    expect(error.range).toBeNull();
  });

  test.each(["1e400", "-1e400"])("refuses an overflowing float: %p", (text) => {
    const error = thrown(SourceSyntaxError, () => readPlainScalar(text));
    expect(error.message).toBe("number is too large for a float");
  });

  test.each(["a: b", "[1]", "{a: 1}", "- x", "a #b", "&a", "*a", "!x", "|", "hello world", "1\n"])(
    "keeps YAML structure as text: %p",
    (text) => {
      expect(readPlainScalar(text)).toBe(text);
    },
  );

  test.each([
    ["2026-13-45", "not a valid date: month must be in 1..12"],
    // YAML's spaced offset is not in yian's datetime grammar.
    ["2026-01-01 10:00:00 +05:00", DATETIME_MSG],
  ])("rejects an unbuildable form: %p", (text, message) => {
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
    for (const text of LONG_INTS) expect(readPlainScalar(text)).toBe(BigInt(text));
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
  ])("under an explicit tag reads a core form: %p %p", (tag, text, plain, expected) => {
    expect(constructScalar(tag, text, plain)).toBe(expected);
  });

  test.each([
    [INT_TAG, "1_000"],
    [INT_TAG, "1.0"],
    [BOOL_TAG, "yes"],
    [BOOL_TAG, "true\n"],
    [FLOAT_TAG, ".inf\n"],
  ])("under an explicit tag rejects a non-core form: %p %p", (tag, found) => {
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

/** Return the phrase for what YAML read a non-text key as. */
function readAs(value: ScalarValue): string {
  if (typeof value === "bigint") return "an integer";
  if (typeof value === "number") return "a float";
  if (typeof value === "boolean") return "a boolean";
  return "null";
}

/** Return the message `parseYamlLiteral(text)` throws, failing unless it is a syntax error. */
function literalError(text: string): string {
  return thrown(SourceSyntaxError, () => parseYamlLiteral(text)).message;
}

/** The two YAML entry points, which share one scalar table. */
const READERS: [string, (text: string) => Value][] = [
  ["document", (text) => SourceDocument.parse(text)._value],
  ["literal", parseYamlLiteral],
];

describe.each(READERS)("both entry points: %s", (name, read) => {
  test.each(ROWS)("a plain scalar reads the same as a value: %p", (text, expected) => {
    expect(read("value: " + text + "\n")).toEqual({ value: expected });
  });

  test("an empty plain scalar is null", () => {
    expect(read("value:\n")).toEqual({ value: null });
    expect(read("")).toBeNull();
  });

  test.each([
    ["'yes'", "yes"],
    ['"017"', "017"],
    ["|\n  yes\n", "yes\n"],
    [">\n  017\n", "017\n"],
  ])("quoted and block scalars stay text: %p", (text, expected) => {
    expect(read("value: " + text)).toEqual({ value: expected });
  });

  test("nested collections read the same", () => {
    // Python's `true: 1:30` key is refused by a literal, whose keys must be text.
    const text = "outer:\n  inner: [yes, 017, on]\n  map: {no: 0x1F, on: 1:30}\n";

    expect(read(text)).toEqual({
      outer: { inner: ["yes", 17n, "on"], map: { no: 31n, on: "1:30" } },
    });
  });

  test("lone CRs are line breaks", () => {
    expect(read("a: x\r  y\r")).toEqual({ a: "x y" });
  });

  test.each<[string, ScalarValue]>([
    ["!!int 017", 17n],
    ["!!float 1", 1],
    ["!!str 1", "1"],
    ["!!bool true", true],
  ])("an explicit core tag reads a core form: %p", (text, expected) => {
    expect(read("value: " + text + "\n")).toEqual({ value: expected });
  });

  test.each([
    ["!!int 1_000", "1_000"],
    ["!!int 1.0", "1.0"],
    ["!!bool yes", "yes"],
    ["!!bool |\n  true\n", "true\n"],
    ['!!float ".inf\\n"', ".inf\n"],
  ])("an explicit core tag rejects a non-core form: %p", (text, found) => {
    const error = thrown(SourceSyntaxError, () => read("value: " + text + "\n"));

    expect(error.message).toContain(reprText(found));
  });

  test("!!timestamp on non-date text names the text", () => {
    const error = thrown(SourceSyntaxError, () => read("value: !!timestamp hello\n"));

    expect(error.message).toBe("expected a date or timestamp, but found 'hello'");
  });

  test.each([
    ['"2026-13-45"', "not a valid date: month must be in 1..12"],
    ['"2026-01-01T10:00:00+05:99"', "not a valid datetime: UTC offset minutes must be under 60"],
  ])("a quoted unbuildable date under an explicit tag gets no quoting fix: %p", (text, reason) => {
    const error = thrown(SourceSyntaxError, () => read("value: !!timestamp " + text + "\n"));

    expect(error.message).toBe(reason);
  });

  test.each([
    ["2026-13-45", "not a valid date: month must be in 1..12"],
    ["2026-02-30", "not a valid date: day is out of range for month"],
    ["0000-01-01", "not a valid date: year 0 is out of range"],
    ["2026-01-01T25:00:00", "not a valid datetime: hour must be in 0..23"],
    // Python's spaced offsets and one-digit offset hour are not in yian's datetime grammar.
    ["2026-01-01 10:00:00 +99:00", DATETIME_MSG],
    ["2026-01-01 10:00:00 +05:99", DATETIME_MSG],
    ["2026-01-01T10:00:00+05:60", "not a valid datetime: UTC offset minutes must be under 60"],
    ["2026-01-01t10:00:00-5:99", DATETIME_MSG],
    ["2026-01-01T10:00:00+23:60", "not a valid datetime: UTC offset minutes must be under 60"],
  ])("an unbuildable date is an error with the fix: %p", (text, reason) => {
    const error = thrown(SourceSyntaxError, () => read("value: " + text + "\n"));

    expect(error.message).toBe(reason + QUOTE_FIX);
    if (name === "document") {
      expect([error.range?.startLine, error.range?.startCol]).toEqual([1, 8]);
    }
  });
});

describe("SourceDocument", () => {
  test.each(ROWS)("keeps a key as written and reports a non-text one: %p", (text, expected) => {
    const document = SourceDocument.parse(text + ": value\n");
    const written = typeof expected === "string" ? expected : text;

    expect(Object.keys(document._value ?? {})).toEqual([written]);
    expect(document.nonTextKey).toEqual(
      typeof expected === "string"
        ? null
        : { key: expected, text, range: new SourceRange(1, 1, 1, 1 + text.length) },
    );
  });

  test("keeps a true key in a nested mapping", () => {
    const document = SourceDocument.parse("map: {no: 0x1F, true: 1:30}\n");

    expect(document._value).toEqual({ map: { no: 31n, true: "1:30" } });
    expect(document.nonTextKey?.key).toBe(true);
  });

  test.each(["~", "null", "NULL", ""])("keeps a null form as a key: %p", (text) => {
    const document = SourceDocument.parse(`? ${text}\n: value\n`);

    expect(document._value).toEqual({ [text]: "value" });
    expect(document.nonTextKey?.key).toBeNull();
    expect(document.nonTextKey?.text).toBe(text);
  });

  test("merge keys still merge and override", () => {
    const text = "base: &base\n  on: x\n  yes: 1\n  017: a\ntarget:\n  <<: *base\n  yes: 2\n";
    const document = SourceDocument.parse(text);

    expect(document._value).toEqual({
      base: { on: "x", yes: 1n, "017": "a" },
      target: { on: "x", yes: 2n, "017": "a" },
    });
    expect(document.nonTextKey).toEqual({
      key: 17n,
      text: "017",
      range: new SourceRange(4, 3, 4, 6),
    });
  });

  test.each([
    ["in a list", "[1, 2026-13-45]\n", 1, 5],
    ["as a key", "a:\n  2026-13-45: 1\n", 2, 3],
    ["offset in a list", "[1, 2026-01-01T10:00:00+05:99]\n", 1, 5],
    ["offset as a key", "a:\n  2026-01-01T10:00:00+05:99: 1\n", 2, 3],
  ])("locates an unbuildable date %s", (_, text, line, column) => {
    const error = thrown(SourceSyntaxError, () => SourceDocument.parse(text));

    // Also matches "not a valid datetime".
    expect(error.message).toContain("not a valid date");
    expect([error.range?.startLine, error.range?.startCol]).toEqual([line, column]);
  });
});

describe("parseYamlLiteral", () => {
  test.each(ROWS)("a text key reads as text; a non-text key is refused: %p", (text, expected) => {
    if (typeof expected === "string") {
      expect(Object.keys(parseYamlLiteral(text + ": value\n") ?? {})).toEqual([expected]);
      return;
    }
    const literal = expected === null ? `? ${text}\n: value\n` : text + ": value\n";
    expect(literalError(literal)).toBe(
      `the key \`${text}\` was read by YAML as ${readAs(expected)}, but mapping keys must ` +
        "be text; quote it",
    );
  });

  test("an empty key or a list key is refused", () => {
    expect(literalError("? \n: value\n")).toBe(
      "an empty key was read by YAML as null, but mapping keys must be text",
    );
    expect(literalError("? [1]\n: v\n")).toBe(
      "a list or mapping can't be a mapping key; mapping keys must be text",
    );
  });

  test("NaN and overflowing floats are refused", () => {
    expect(literalError("value: .nan\n")).toBe("NaN is not a yian value" + QUOTE_FIX);
    expect(literalError("value: 1e400\n")).toBe("number is too large for a float" + QUOTE_FIX);
    expect(literalError('value: !!float ".nan"\n')).toBe("NaN is not a yian value");
  });

  test("an int of any length reads exactly", () => {
    for (const text of LONG_INTS) {
      expect(parseYamlLiteral("value: " + text + "\n")).toEqual({ value: BigInt(text) });
    }
    expect(parseYamlLiteral("value: " + "0".repeat(4999) + "1\n")).toEqual({ value: 1n });
    const digits = "1".repeat(5000);
    expect(parseYamlLiteral("value: !!int " + digits + "\n")).toEqual({ value: BigInt(digits) });
  });

  test.each(["[1, 2026-13-45]\n", "a:\n  2026-13-45: 1\n"])(
    "an unbuildable date in a list or as a key is refused: %p",
    (text) => {
      expect(literalError(text)).toBe("not a valid date: month must be in 1..12" + QUOTE_FIX);
    },
  );

  test("an in-range offset builds; a one-digit offset is refused", () => {
    for (const text of ["2026-01-01T10:00:00+05:59", "2026-01-01T10:00:00-23:59"]) {
      expect(parseYamlLiteral(text)).toBe(text);
    }
    // Python reads +05:00; yian's datetime grammar needs the minutes.
    expect(literalError("2026-01-01T10:00:00+5")).toBe(DATETIME_MSG + QUOTE_FIX);
  });

  test.each(["+05:99", "+07:60", "+23:60", "+24:00", "+99:00", "+99:99"])(
    "the literal and the typed grammar reject an offset for the same reason: %p",
    (offset) => {
      const text = `2026-05-21T14:30:00${offset}`; // YAML needs the seconds to read a timestamp
      const typed = thrown(RangeError, () => parseDatetime(text));
      expect(literalError(text)).toContain(typed.message);
    },
  );

  const TAGS = [
    STR_TAG,
    NULL_TAG,
    BOOL_TAG,
    INT_TAG,
    FLOAT_TAG,
    TIMESTAMP_TAG,
    SEQ_TAG,
    MAP_TAG,
    MERGE_TAG,
    ...["binary", "set", "omap", "pairs", "value"].map((name) => "tag:yaml.org,2002:" + name),
    "!x",
  ];
  test.each(TAGS.flatMap((tag) => ["hello", "2026-13-45", "''"].map((text) => [tag, text])))(
    "no constructor raises a non-YAML error: %p %p",
    (tag, text) => {
      // Consumers turn only syntax errors into load errors.
      try {
        parseYamlLiteral("value: !<" + tag + "> " + text + "\n");
      } catch (error) {
        expect(error).toBeInstanceOf(SourceSyntaxError);
      }
    },
  );

  test("tags", () => {
    const noConstructor = (tag: string) =>
      "could not determine a constructor for the tag " + reprText(tag);

    expect(literalError("value: !int 5\n")).toBe(noConstructor("!int"));
    expect(literalError("value: !!binary aGk=\n")).toBe(noConstructor("tag:yaml.org,2002:binary"));
    // YAML 1.2: the bare tag keeps the scalar as text, where Python reads 5.
    expect(parseYamlLiteral("value: ! 5\n")).toEqual({ value: "5" });
    // The bare tag keeps "<<" as a text key too, where Python merges.
    expect(parseYamlLiteral("{! <<: {a: 1}}")).toEqual({ "<<": { a: 1n } });
    expect(literalError("value: !!seq {a: 1}\n")).toBe(
      "expected a sequence node, but found mapping",
    );
    expect(literalError("value: !!str [1]\n")).toBe("expected a scalar node, but found sequence");
    expect(literalError("value: !!map [1]\n")).toBe("expected a mapping node, but found sequence");
    expect(literalError("value: !!seq 1\n")).toBe("expected a sequence node, but found scalar");
    expect(literalError("value: <<\n")).toBe(noConstructor(MERGE_TAG));
  });

  test.each(["[!!seq {a: 1}, 2026-13-45]", "[!!seq 1, 2026-13-45]"])(
    "a !!seq node of the wrong kind is refused after its sibling scalars: %p",
    (text) => {
      expect(literalError(text)).toBe("not a valid date: month must be in 1..12" + QUOTE_FIX);
    },
  );

  test("aliases", () => {
    const shared = parseYamlLiteral("{a: &x [1], b: *x}");
    expect(shared).toEqual({ a: [1n], b: [1n] });
    expect(isRecord(shared) && shared["a"] === shared["b"]).toBe(true);

    const recursive = parseYamlLiteral("&a [*a]");
    expect(Array.isArray(recursive) && recursive.length === 1 && recursive[0] === recursive)
      .toBe(true);

    expect(literalError("[*nope]")).toBe("found undefined alias 'nope'");
  });

  test("a repeated key keeps the last value at its first position", () => {
    expect(parseYamlLiteral("{a: 1, a: 2}")).toEqual({ a: 2n });
    expect(Object.entries(parseYamlLiteral("{a: 1, b: 0, a: 2}") ?? {})).toEqual([
      ["a", 2n],
      ["b", 0n],
    ]);
  });

  test("a __proto__ key stays data", () => {
    const value = parseYamlLiteral("__proto__: 1\n");

    expect(Object.getOwnPropertyDescriptor(value, "__proto__")?.value).toBe(1n);
    expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
  });

  test("merge keys merge and own keys override", () => {
    // Python's `017: a` key is quoted here: a literal's keys must be text.
    const text = 'base: &base\n  on: x\n  yes: 1\n  "017": a\ntarget:\n  <<: *base\n  yes: 2\n';

    expect(parseYamlLiteral(text)).toEqual({
      base: { on: "x", yes: 1n, "017": "a" },
      target: { on: "x", yes: 2n, "017": "a" },
    });
  });

  test("merged keys come before the mapping's own keys", () => {
    const value = parseYamlLiteral("{a: 1, <<: {b: 2, a: 0}}");

    expect(Object.keys(value ?? {})).toEqual(["b", "a"]);
    expect(value).toEqual({ b: 2n, a: 1n });
  });

  test("a merge list's first mapping wins; of two merge keys, the later wins", () => {
    expect(parseYamlLiteral("{<<: [{a: 1}, {a: 2}]}")).toEqual({ a: 1n });
    expect(parseYamlLiteral("{<<: {a: 1}, <<: {a: 2}}")).toEqual({ a: 2n });
  });

  test("a merge chain of 200 anchors reads every key of the last link", () => {
    const links = ["&m0 {k0: 1}"];
    for (let i = 1; i < 200; i++) links.push(`&m${i} {<<: *m${i - 1}, k${i}: 1}`);

    const value = parseYamlLiteral(`[${links.join(", ")}]`);

    const keys = Array.from({ length: 200 }, (_, i) => `k${i}`);
    expect(Array.isArray(value) ? value[199] : null).toEqual(
      Object.fromEntries(keys.map((key) => [key, 1n])),
    );
  });

  test.each([
    ["{<<: 1}", "expected a mapping or list of mappings for merging, but found scalar"],
    ["{<<: [1]}", "expected a mapping for merging, but found scalar"],
    ["{<<: [[1]]}", "expected a mapping for merging, but found sequence"],
    ["{<<: }", "expected a mapping or list of mappings for merging, but found scalar"],
  ])("a merge source that is not a mapping or list of mappings is refused: %p", (text, problem) => {
    expect(literalError(text)).toBe(problem);
  });

  test.each([
    // The merge is checked before the mapping's own keys are built.
    [
      "{a: 2026-13-45, <<: 1}",
      "expected a mapping or list of mappings for merging, but found scalar",
    ],
    // The first source's error comes before the second merge key's.
    [
      "{<<: {<<: 1}, <<: [2]}",
      "expected a mapping or list of mappings for merging, but found scalar",
    ],
    // Sibling scalars are built before the mappings nested beside them.
    [
      "[{<<: 1}, 2026-13-45]",
      "not a valid date: month must be in 1..12; quote it to keep it as text",
    ],
  ])("merge errors come in Python's order: %p", (text, problem) => {
    expect(literalError(text)).toBe(problem);
  });

  test("a merge cycle ends", () => {
    expect(parseYamlLiteral("&m {<<: *m, a: 1}")).toEqual({ a: 1n });
  });

  test("a merge source's tag is ignored", () => {
    expect(parseYamlLiteral("{<<: !!str {a: 1}}")).toEqual({ a: 1n });
  });

  test("an explicit !!merge key merges", () => {
    expect(parseYamlLiteral("{!!merge x: {a: 1}}")).toEqual({ a: 1n });
  });

  test.each(["value: [1, 2\n", "a\n---\nb\n"])(
    "YAML errors are SourceSyntaxErrors without a range: %p",
    (text) => {
      const library = thrown(Error, () => composeFlowDocument(text));
      const error = thrown(SourceSyntaxError, () => parseYamlLiteral(text));

      expect(error.range).toBeNull();
      expect(error.message).toBe(library.message);
    },
  );
});

test.each([
  "yaml/compose_source_scalars__explicit_bool_tag_refuses_yes",
  "yaml/compose_source_scalars__explicit_int_tag_refuses_non_core_form",
  "yaml/compose_source_scalars__unbuildable_date_as_a_key",
])("parseSourceDocument: %s", (id) => runTextCase({ id, call: parseSourceDocument }));
