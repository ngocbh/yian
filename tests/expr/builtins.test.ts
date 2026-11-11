/**
 * The shipped builtins: each one's results, its null and wrong-type errors, and its result type.
 *
 * Where Python has the same operation, yian gives Python's result. The rows of
 * `tests/fixtures/python_builtins.json` are what CPython 3.12 gives (written by
 * `tests/fixtures/python_builtins.py`): the whitespace set, `split`, `strip`, `lstrip`, `rstrip`
 * and `replace` over astral and whitespace characters, `int` and `float` of text, `int` of a float
 * and `float` of an int, `round` of floats and ints, `sum` of floats, ints and bools, and `min` and
 * `max` of texts beyond the BMP.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { builtinResultType, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import { evalExpr, inferExprType, type RootResolver } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { evalBinding } from "../../src/expr/template.ts";
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

/** A result CPython gave: an int, a float's `repr`, or the class of the error it raised. */
type Python = { int: string } | { error: string } | string | boolean;

/** Decode a float's `repr`. */
function pythonFloat(text: string): number {
  if (text === "inf") return Infinity;
  if (text === "-inf") return -Infinity;
  return Number(text);
}

/** Decode a value CPython gave, or `undefined` for an error. */
function pythonValue(want: Python): Value | undefined {
  if (typeof want === "boolean") return want;
  if (typeof want === "string") return pythonFloat(want);
  return "int" in want ? BigInt(want.int) : undefined;
}

/** Expect `got()` to give what CPython gave (`Object.is`, so `-0.0` counts), or to fail where it failed. */
function expectPython(got: () => Value, want: Python): void {
  const value = pythonValue(want);
  if (value === undefined) expect(got).toThrow();
  else expect(got()).toBe(value);
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

describe("conversions", () => {
  // Python reads other decimal digits too (`int("\u0661")` is 1); yian takes ASCII digits only.
  const space = new Set(PYTHON["whitespace"] as number[]);
  const asciiDigits = (text: string) => [...text].every((ch) => ch.codePointAt(0)! < 0x80 || space.has(ch.codePointAt(0)!));

  test.each(PYTHON["int_text"] as [string, Python][])("int(%j) is Python's (ASCII digits only)", (text, want) => {
    expectPython(() => call("int", text), asciiDigits(text) ? want : { error: "ValueError" });
  });

  test.each(PYTHON["float_text"] as [string, Python][])("float(%j) is Python's (no nan)", (text, want) => {
    expectPython(() => call("float", text), asciiDigits(text) && want !== "nan" ? want : { error: "ValueError" });
  });

  test.each(PYTHON["int_float"] as [string, Python][])("int(%s) truncates toward zero", (x, want) => {
    expectPython(() => call("int", pythonFloat(x)), want);
  });

  test.each(PYTHON["float_int"] as [string, Python][])("float(%s) rounds to the nearest float", (n, want) => {
    expectPython(() => call("float", BigInt(n)), want);
  });

  test("a bool is 0 or 1", () => {
    expect(run("[int(true), int(false), float(true), float(false)]")).toEqual([1n, 0n, 1, 0]);
  });

  test("int and float give back their own kind as it is", () => {
    expect(run("[int(7), float(2.5), float(inf)]")).toEqual([7n, 2.5, Infinity]);
  });

  test.each<[string, Value]>([
    ["str(s)", "text"],
    ["str(n)", "12"],
    ["str(f)", "1"],
    ["str(g)", "2.5"],
    ["str(xs)", '[1,"a"]'],
    ["str(r)", '{"a":true}'],
    ["str(i)", "Infinity"],
  ])("%s is the text the value writes into text", (text, want) => {
    const values = { s: "text", n: 12n, f: 1, g: 2.5, xs: [1n, "a"], r: { a: true }, i: Infinity };
    expect(run(text, values)).toBe(want);
    const name = text.slice(4, -1);
    expect(evalBinding(`<\${${name}}>`, fromRecord(values))).toBe(`<${String(want)}>`);
  });

  test.each<[string, ValueRecord, string]>([
    ["str(x)", {}, 'str() got null; give a default, e.g. `str(x :- "")`'],
    ["int(x)", {}, "int() got null; give a default, e.g. `int(x :- 0)`"],
    ["int(x)", { x: "1.5" }, 'int() can\'t convert "1.5" to an int'],
    ["int(x)", { x: "\u0661" }, 'int() can\'t convert "\u0661" to an int'],
    ["int(x)", { x: Infinity }, "int() can't convert Infinity to an int"],
    ["int(x)", { x: [1n] }, "int() takes str, int, float or bool, got list"],
    ["float(x)", {}, "float() got null; give a default, e.g. `float(x :- 0.0)`"],
    ["float(x)", { x: "nan" }, 'float() can\'t convert "nan": yian has no NaN'],
    ["float(x)", { x: "1,5" }, 'float() can\'t convert "1,5" to a float'],
    ["float(x)", { x: 10n ** 400n }, "float() got an int too large for a float"],
    ["float(x)", { x: { a: 1n } }, "float() takes str, int, float or bool, got record"],
  ])("%s fails (case %#)", (text, values, message) => {
    const name = text.slice(0, text.indexOf("("));
    expect(failure(text, values)).toBe(`expression function '${name}' failed: ${message}`);
  });

  test.each<[string, Type]>([
    ["str(x)", Type.scalar(ValueKind.STRING)],
    ["int(x)", Type.scalar(ValueKind.INTEGER)],
    ["float(x)", Type.scalar(ValueKind.NUMBER)],
  ])("%s has a fixed result type", (text, want) => {
    expect(inferExprType(parseExpr(text), () => null)).toEqual(want);
  });
});

describe("round", () => {
  test.each(PYTHON["round"] as [string | { int: string }, number | null, Python][])(
    "round(%j, %j) is Python's",
    (x, n, want) => {
      const value = typeof x === "string" ? pythonFloat(x) : BigInt(x.int);
      expectPython(() => (n === null ? call("round", value) : call("round", value, BigInt(n))), want);
    },
  );

  test("round(x) gives an int, round(x, n) a float for a float", () => {
    const rounded = run("[round(2.5), round(3.5), round(-2.5), round(2.675, 2), round(2.5, 0)]");
    expect(rounded).toEqual([2n, 4n, -2n, 2.67, 2]);
    expect(typeof run("round(2.5, 0)")).toBe("number");
  });

  test("a null n is the default, and a bool n counts as 0 or 1", () => {
    expect(run("round(2.5, null)")).toBe(2n);
    expect(run("round(2.25, true)")).toBe(2.2);
  });

  test("a huge n gives the float back, a very negative one a zero with its sign", () => {
    expect(run("round(1.5, 10 ** 30)")).toBe(1.5);
    expect(run("round(-1.5, -(10 ** 30))")).toBe(-0);
    expect(run("round(15, -(10 ** 30))")).toBe(0n);
  });

  test.each<[string, string]>([
    ["round(x)", "round() got null; give a default, e.g. `round(x :- 0)`"],
    ['round("1")', "round() takes a number, got str"],
    ["round(1.5, 1.0)", "round() takes int for n, got float"],
    ["round(inf)", "round() can't round Infinity to an int"],
    ["round(1.7976931348623157 * 10 ** 308, -308)", "round() of 1.7976931348623157e+308 is too large for a float"],
  ])("%s fails", (text, message) => {
    expect(failure(text)).toBe(`expression function 'round' failed: ${message}`);
  });

  test.each<[string, Type | null]>([
    ["round(f)", Type.scalar(ValueKind.INTEGER)],
    ["round(f, null)", Type.scalar(ValueKind.INTEGER)],
    ["round(i, 2)", Type.scalar(ValueKind.INTEGER)],
    ["round(f, 2)", Type.scalar(ValueKind.NUMBER)],
    ["round(x=f, n=2)", Type.scalar(ValueKind.NUMBER)],
    ["round(f, o)", null],
    ["round(u, 2)", null],
  ])("%s is typed at load", (text, want) => {
    const types: Record<string, Type> = {
      f: Type.scalar(ValueKind.NUMBER),
      i: Type.scalar(ValueKind.INTEGER),
      o: Type.scalar(ValueKind.INTEGER).replace({ nullable: true }),
    };
    expect(inferExprType(parseExpr(text), (path) => types[path] ?? null)).toEqual(want);
  });
});

describe("numbers and lists", () => {
  /** Decode one item of a `sum` row. */
  const item = (x: Python): Value => (typeof x === "boolean" ? x : pythonValue(x)!);

  test.each(PYTHON["sum"] as [Python[], Python][])("sum(%j) is Python's (no nan)", (xs, want) => {
    expectPython(() => call("sum", xs.map(item)), want === "nan" ? { error: "ValueError" } : want);
  });

  test.each(PYTHON["minmax_text"] as [string[], string, string][])(
    "min and max of %j compare code points",
    (xs, least, most) => {
      expect([call("min", xs), call("max", xs)]).toEqual([least, most]);
    },
  );

  test.each<[string, Value]>([
    ["sum([])", 0n],
    ["sum([1, 2, true])", 4n],
    ["sum([1, 2.5])", 3.5],
    ["sum([0.1, 0.2, 0.3])", 0.6],
    ["min([3, 1.5, 2])", 1.5],
    ["max([1, 1.0, true])", 1n],
    ["min([1.0, 1])", 1],
    ["min([true, 2])", true],
    ["max(['b', 'a', 'c'])", "c"],
    ["abs(-3)", 3n],
    ["abs(-2.5)", 2.5],
    ["abs(true)", 1n],
    ["any([0, '', [], 'x'])", true],
    ["any([])", false],
    ["all([1, 'a', [0]])", true],
    ["all([1, null])", false],
    ["all([])", true],
  ])("%s", (text, want) => {
    expect(run(text)).toBe(want);
  });

  test("a whole float stays a float", () => {
    expect(run("[sum([0.5, 0.5]), abs(-2.0), max([1.0, 0])]")).toEqual([1, 2, 1]);
    expect(typeof run("sum([0.5, 0.5])")).toBe("number");
  });

  test.each<[string, string]>([
    ["sum(x)", "sum() got null; give a default, e.g. `sum(x :- [])`"],
    ["sum('12')", "sum() takes a list, got str"],
    ["sum([1, '2'])", "sum() takes a list of numbers, got a str item"],
    ["sum([1, null])", "sum() takes a list of numbers, got a null item"],
    ["sum([inf, -inf])", "sum() has no numeric result: it adds inf and -inf"],
    ["sum([10 ** 400, 0.5])", "sum() got an int too large for a float"],
    ["min([])", "min() got an empty list"],
    ["max([1, 'a'])", "max() can't compare int and str"],
    ["min([1, null])", "min() got a null item"],
    ["max([[1]])", "max() takes a list of numbers or texts, got a list item"],
    ["abs('1')", "abs() takes a number, got str"],
    ["abs(x)", "abs() got null; give a default, e.g. `abs(x :- 0)`"],
    ["any('ab')", "any() takes a list, got str"],
    ["all(x)", "all() got null; give a default, e.g. `all(x :- [])`"],
  ])("%s fails", (text, message) => {
    const name = text.slice(0, text.indexOf("("));
    expect(failure(text)).toBe(`expression function '${name}' failed: ${message}`);
  });

  const LISTS: Record<string, Type> = {
    ints: new Type({ kind: ValueKind.LIST_INTEGER, element: Type.scalar(ValueKind.INTEGER) }),
    floats: new Type({ kind: ValueKind.LIST_NUMBER, element: Type.scalar(ValueKind.NUMBER) }),
    texts: new Type({ kind: ValueKind.LIST_STRING, element: Type.scalar(ValueKind.STRING) }),
    maybe: new Type({
      kind: ValueKind.LIST_INTEGER,
      element: Type.scalar(ValueKind.INTEGER).replace({ nullable: true }),
    }),
    i: Type.scalar(ValueKind.INTEGER),
    f: Type.scalar(ValueKind.NUMBER),
  };

  test.each<[string, Type | null]>([
    ["any(ints)", Type.scalar(ValueKind.BOOLEAN)],
    ["all(x)", Type.scalar(ValueKind.BOOLEAN)],
    ["min(ints)", Type.scalar(ValueKind.INTEGER)],
    ["max(texts)", Type.scalar(ValueKind.STRING)],
    ["min(maybe)", Type.scalar(ValueKind.INTEGER)],
    ["max(x)", null],
    ["sum(ints)", Type.scalar(ValueKind.INTEGER)],
    ["sum(floats)", Type.scalar(ValueKind.NUMBER)],
    ["sum(texts)", null],
    ["abs(i)", Type.scalar(ValueKind.INTEGER)],
    ["abs(f)", Type.scalar(ValueKind.NUMBER)],
    ["abs(x)", null],
  ])("%s is typed at load", (text, want) => {
    expect(inferExprType(parseExpr(text), (path) => LISTS[path] ?? null)).toEqual(want);
  });
});

describe("dates", () => {
  test.each<[string, Value]>([
    ['add_days("2026-02-27", 2)', "2026-03-01"],
    ['add_days("2024-02-28", 1)', "2024-02-29"],
    ['add_days("2026-01-01", -1)', "2025-12-31"],
    ['add_days("2026-05-21", 0)', "2026-05-21"],
    ['add_days("0001-01-01", 3652058)', "9999-12-31"],
    // A datetime keeps its time and offset as written.
    ['add_days("2026-05-21T14:30Z", 1)', "2026-05-22T14:30Z"],
    ['add_days("2026-12-31 23:59:59.5-07:00", 1)', "2027-01-01 23:59:59.5-07:00"],
    ['days_between("2026-01-01", "2026-03-01")', 59n],
    ['days_between("2026-03-01", "2026-01-01")', -59n],
    ['days_between("2024-01-01", "2025-01-01")', 366n],
    // Datetimes count whole days in UTC, rounded down, as Python's `timedelta.days` does.
    ['days_between("2026-01-01T12:00Z", "2026-01-02T11:59Z")', 0n],
    ['days_between("2026-01-02T11:59Z", "2026-01-01T12:00Z")', -1n],
    ['days_between("2026-01-01T00:00:00.000001", "2026-01-02T00:00")', 0n],
    ['days_between("2026-01-01T23:00-02:00", "2026-01-02T01:00Z")', 0n],
    // A date is its midnight in UTC, and a datetime without an offset is UTC.
    ['days_between("2026-01-01", "2026-01-03T00:00")', 2n],
  ])("%s", (text, want) => {
    expect(run(text)).toBe(want);
  });

  test("add_days reads system.today as any date", () => {
    expect(run("add_days(today, -7)", { today: "2026-11-10" })).toBe("2026-11-03");
  });

  const SHAPE = "must be a date (2026-05-21) or datetime (2026-05-21T14:30[:00][Z|+07:00])";

  test.each<[string, ValueRecord, string]>([
    ["add_days(d, 1)", {}, "add_days() got null; give a default, e.g. `add_days(x :- system.today)`"],
    [
      'add_days("2026-13-01", 1)',
      {},
      'add_days() takes a date or datetime, got "2026-13-01": not a valid date: month must be in 1..12',
    ],
    ['add_days("soon", 1)', {}, `add_days() takes a date or datetime, got "soon": ${SHAPE}`],
    ["add_days(n, 1)", { n: 3n }, "add_days() takes a date or datetime, got int"],
    ['add_days("2026-01-01", 1.5)', {}, "add_days() takes int for n, got float"],
    ['add_days("2026-01-01", n)', {}, "add_days() takes int for n, got null"],
    ['add_days("9999-12-31", 1)', {}, "add_days() gives a date past the year 9999"],
    ['add_days("0001-01-01", -1)', {}, "add_days() gives a date before the year 1"],
    ['add_days("2026-01-01", 10 ** 20)', {}, "add_days() gives a date past the year 9999"],
    [
      'days_between("2026-01-01", b)',
      {},
      "days_between() got null for b; give a default, e.g. `days_between(a, b :- system.today)`",
    ],
    ['days_between("2026-01-01", "x")', {}, `days_between() takes a date or datetime for b, got "x": ${SHAPE}`],
  ])("%s fails", (text, values, message) => {
    const name = text.slice(0, text.indexOf("("));
    expect(failure(text, values)).toBe(`expression function '${name}' failed: ${message}`);
  });

  const TYPES: Record<string, Type> = {
    d: Type.scalar(ValueKind.DATE),
    t: Type.scalar(ValueKind.DATETIME),
    od: Type.scalar(ValueKind.DATE).replace({ nullable: true }),
    s: Type.scalar(ValueKind.STRING),
  };

  test.each<[string, Type | null]>([
    ["add_days(d, 1)", Type.scalar(ValueKind.DATE)],
    ["add_days(od, 1)", Type.scalar(ValueKind.DATE)],
    ["add_days(t, 1)", Type.scalar(ValueKind.DATETIME)],
    ["add_days(s, 1)", null],
    ["days_between(d, t)", Type.scalar(ValueKind.INTEGER)],
  ])("%s is typed at load", (text, want) => {
    expect(inferExprType(parseExpr(text), (path) => TYPES[path] ?? null)).toEqual(want);
  });
});
