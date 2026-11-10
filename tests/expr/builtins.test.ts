/**
 * The shipped builtins: each one's results, its null and wrong-type errors, and its result type.
 *
 * Where Python has the same operation, yian gives Python's result. The rows of
 * `tests/fixtures/python_builtins.json` are what CPython 3.12 gives (written by
 * `tests/fixtures/python_builtins.py`): the whitespace set, `split`, `strip`, `lstrip`, `rstrip`
 * and `replace` over astral and whitespace characters.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { builtinResultType, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import { evalExpr, type RootResolver } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";
import { fieldOf, Type, ValueKind } from "../../src/typesys/values.ts";

/** What CPython gave, by table. */
const PYTHON = JSON.parse(readFileSync(new URL("../fixtures/python_builtins.json", import.meta.url), "utf8")) as Record<
  string,
  unknown[]
>;

/** A resolver over `values`: a name reads its own field, and a missing one is absent. */
function fromRecord(values: ValueRecord): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(values, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Evaluate `text` with `values` in scope. */
function run(text: string, values: ValueRecord = {}): Value {
  return evalExpr(parseExpr(text), fromRecord(values));
}

/** Return the message `run(text, values)` fails with, failing the test if it doesn't. */
function failure(text: string, values: ValueRecord = {}): string {
  try {
    run(text, values);
  } catch (error) {
    return (error as Error).message;
  }
  throw new Error(`${text} did not fail`);
}

/** Call builtin `name` on `args`, bound to `a0`, `a1`, ... */
function call(name: string, ...args: Value[]): Value {
  const names = args.map((_, i) => `a${i}`);
  return run(`${name}(${names.join(", ")})`, Object.fromEntries(args.map((arg, i) => [names[i], arg])));
}

describe("whitespace", () => {
  test("strip removes exactly Python's whitespace set, and nothing else", () => {
    const strip = TEMPLATE_FNS.get("strip")!;
    const stripped: number[] = [];
    for (let c = 0; c <= 0x10ffff; c++) {
      // A lone surrogate is no yian text.
      if (c >= 0xd800 && c <= 0xdfff) continue;
      if (strip.call(String.fromCodePoint(c), null) === "") stripped.push(c);
    }
    expect(stripped).toEqual(PYTHON["whitespace"] as number[]);
  });
});

describe("text", () => {
  test.each(PYTHON["split"] as [string, string | null, string[]][])("split(%j, %j) is Python's", (s, sep, want) => {
    expect(call("split", s, sep)).toEqual(want);
  });

  test.each(PYTHON["strip"] as [string, string | null, string, string, string][])(
    "strip, lstrip and rstrip of %j by %j are Python's",
    (s, chars, both, left, right) => {
      const got = [call("strip", s, chars), call("lstrip", s, chars), call("rstrip", s, chars)];
      expect(got).toEqual([both, left, right]);
    },
  );

  test.each(PYTHON["replace"] as [string, string, string, string][])(
    "replace(%j, %j, %j) is Python's",
    (s, old, by, want) => {
      expect(call("replace", s, old, by)).toBe(want);
    },
  );

  test("an empty old never splits a surrogate pair", () => {
    expect(call("replace", "a😀", "", "-")).toBe("-a-😀-");
    expect(call("replace", "", "", "-")).toBe("-");
  });

  test.each<[string, boolean]>([
    ['startswith("hello", "he")', true],
    ['startswith("hello", "")', true],
    ['startswith("he", "hello")', false],
    ['endswith("hello", "lo")', true],
    ['endswith(s="😀x", suffix="x")', true],
    ['endswith("hello", "he")', false],
  ])("%s", (text, want) => {
    expect(run(text)).toBe(want);
  });

  test("the separators and the characters to strip are keyword arguments too", () => {
    expect(run('split(s="a-b", sep="-")')).toEqual(["a", "b"]);
    expect(run('strip("xax", chars="x")')).toBe("a");
    expect(run('replace(s="aa", old="a", new="b")')).toBe("bb");
  });

  test.each<[string, ValueRecord, string]>([
    ["startswith(s, 'a')", {}, 'startswith() got null; give a default, e.g. `startswith(x :- "")`'],
    ["startswith('a', p)", {}, "startswith() takes str for prefix, got null"],
    ["endswith(n, 'a')", { n: 3n }, "endswith() takes str, got int"],
    ["split(xs)", { xs: ["a"] }, "split() takes str, got list"],
    ["split('a', 1)", {}, "split() takes str for sep, got int"],
    ["split('a', '')", {}, "split() got an empty sep; omit it to split on whitespace"],
    ["strip(s)", {}, 'strip() got null; give a default, e.g. `strip(x :- "")`'],
    ["lstrip('a', 1)", {}, "lstrip() takes str for chars, got int"],
    ["rstrip(1.5)", {}, "rstrip() takes str, got float"],
    ["replace('a', 'a', n)", {}, "replace() takes str for new, got null"],
    ["replace('a', 1, 'b')", {}, "replace() takes str for old, got int"],
  ])("%s fails", (text, values, message) => {
    const name = text.slice(0, text.indexOf("("));
    expect(failure(text, values)).toBe(`expression function '${name}' failed: ${message}`);
  });

  test("a null separator or set of characters is the default", () => {
    expect(run("split(' a  b ', none)")).toEqual(["a", "b"]);
    expect(run("strip(' a ', null)")).toBe("a");
  });

  test.each<[string, Type]>([
    ["startswith", Type.scalar(ValueKind.BOOLEAN)],
    ["endswith", Type.scalar(ValueKind.BOOLEAN)],
    ["split", new Type({ kind: ValueKind.LIST_STRING, element: Type.scalar(ValueKind.STRING) })],
    ["strip", Type.scalar(ValueKind.STRING)],
    ["lstrip", Type.scalar(ValueKind.STRING)],
    ["rstrip", Type.scalar(ValueKind.STRING)],
    ["replace", Type.scalar(ValueKind.STRING)],
  ])("%s has a fixed result type", (name, want) => {
    expect(builtinResultType(name)).toEqual(want);
  });
});
