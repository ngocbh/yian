/**
 * The caps that bound one expression's work: the items its comprehensions bind
 * (`YIAN_MAX_COMPREHENSION_ITEMS`), the digits of an int an operator or a builtin gives
 * (`YIAN_MAX_INT_DIGITS`), the indent `render_as_json` takes (`YIAN_MAX_JSON_INDENT`), and the
 * characters of a text an operator or a builtin gives (`YIAN_MAX_TEXT_CHARS`). Going over one is an
 * `ExpressionError` that names it, raised before the work it would take.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, type Defaults, install } from "../../src/defaults.ts";
import { registerTemplateFn, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import { ExpressionError, evalExpr, type RootResolver } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";
import { fieldOf } from "../../src/typesys/values.ts";

afterEach(() => install(BUILTIN));

/** Return a resolver over `mapping`; a miss is `null`. */
function fromRecord(mapping: ValueRecord): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(mapping, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Evaluate `text` over `mapping` with the limits `limits` changes. */
function evaluate(text: string, mapping: ValueRecord = {}, limits: Partial<Defaults> = {}): Value {
  install({ ...BUILTIN, ...limits });
  return evalExpr(parseExpr(text), fromRecord(mapping));
}

/** Return the message of the `ExpressionError` evaluating `text` throws. */
function messageOf(text: string, mapping: ValueRecord = {}, limits: Partial<Defaults> = {}): string {
  try {
    evaluate(text, mapping, limits);
  } catch (error) {
    expect(error).toBeInstanceOf(ExpressionError);
    return (error as Error).message;
  }
  throw new Error(`${text} evaluated without an error`);
}

const FOUR = { xs: [1n, 2n, 3n, 4n] };

describe("YIAN_MAX_COMPREHENSION_ITEMS", () => {
  test("every for clause's bindings count, up to the cap", () => {
    // 4 bindings of a, then 4 of b for each: 20 in all.
    expect(evaluate("[a * b for a in xs for b in xs]", FOUR, { maxComprehensionItems: 20 })).toHaveLength(16);
    expect(messageOf("[a * b for a in xs for b in xs]", FOUR, { maxComprehensionItems: 19 })).toBe(
      "comprehension `for b in xs` goes over the limit of 19 items bound in one expression " +
        "(YIAN_MAX_COMPREHENSION_ITEMS)",
    );
  });

  test("an item an if clause drops still counts", () => {
    expect(messageOf("[a for a in xs if false]", FOUR, { maxComprehensionItems: 3 })).toBe(
      "comprehension `for a in xs` goes over the limit of 3 items bound in one expression " +
        "(YIAN_MAX_COMPREHENSION_ITEMS)",
    );
  });

  test("nested and sibling comprehensions share one count", () => {
    expect(evaluate("[[b for b in xs] for a in xs]", FOUR, { maxComprehensionItems: 20 })).toHaveLength(4);
    expect(() => evaluate("[[b for b in xs] for a in xs]", FOUR, { maxComprehensionItems: 19 })).toThrow(
      "(YIAN_MAX_COMPREHENSION_ITEMS)",
    );
    expect(evaluate("[a for a in xs] + [b for b in xs]", FOUR, { maxComprehensionItems: 8 })).toHaveLength(8);
    expect(() => evaluate("[a for a in xs] + [b for b in xs]", FOUR, { maxComprehensionItems: 7 })).toThrow(
      "(YIAN_MAX_COMPREHENSION_ITEMS)",
    );
  });

  test("each evaluation starts its own count", () => {
    const tree = parseExpr("[a for a in xs]");
    install({ ...BUILTIN, maxComprehensionItems: 4 });
    for (let i = 0; i < 3; i++) expect(evalExpr(tree, fromRecord(FOUR))).toHaveLength(4);
  });

  test("by default, three nested clauses over 300 items stop at the cap", () => {
    const xs = Array.from({ length: 300 }, (_, i) => BigInt(i));
    const started = performance.now();
    expect(messageOf("[a for a in xs for b in xs for c in xs]", { xs })).toBe(
      "comprehension `for c in xs` goes over the limit of 1000000 items bound in one expression " +
        "(YIAN_MAX_COMPREHENSION_ITEMS)",
    );
    expect(performance.now() - started).toBeLessThan(5_000);
  });
});

describe("YIAN_MAX_INT_DIGITS", () => {
  const OVER = "gives an int over the limit of 4300 digits (YIAN_MAX_INT_DIGITS)";

  test("an int result may have the cap's digits, and not one more", () => {
    expect(String(evaluate("10 ** 4299"))).toHaveLength(4300);
    expect(String(evaluate("-(10 ** 4299)"))).toHaveLength(4301);
    expect(messageOf("10 ** 4300")).toBe(`10 ** 4300 ${OVER}`);
    expect(messageOf("9 * 10 ** 4299 + 10 ** 4299")).toStartWith("9");
    expect(messageOf("9 * 10 ** 4299 + 10 ** 4299")).toEndWith(` + 1${"0".repeat(198)}… ${OVER}`);
  });

  test("a power or product too large is refused before it is computed", () => {
    const started = performance.now();
    expect(messageOf("7 ** 300000000")).toBe(`7 ** 300000000 ${OVER}`);
    expect(messageOf("2 ** (2 ** 40)")).toBe(`2 ** 1099511627776 ${OVER}`);
    const big = { x: 10n ** 4000n };
    expect(messageOf("x * x", big)).toEndWith(OVER);
    expect(messageOf("x * x * x * x * x * x * x * x", big)).toEndWith(OVER);
    expect(performance.now() - started).toBeLessThan(5_000);
  });

  test("an operator on an int over the cap is refused, but reading or comparing it is not", () => {
    const big = { x: 10n ** 5000n };
    expect(evaluate("x", big)).toBe(10n ** 5000n);
    expect(evaluate("x > 0", big)).toBe(true);
    expect(evaluate("x % 7", big)).toBe(10n ** 5000n % 7n);
    expect(messageOf("x + 1", big)).toEndWith(OVER);
    expect(messageOf("-x", big)).toBe(`unary minus on ${String(10n ** 5000n).slice(0, 199)}… ${OVER}`);
  });

  test("a float is not an int", () => {
    expect(evaluate("2.0 ** 5000")).toBe(Infinity);
    expect(evaluate("10 ** 4299 * 1.0")).toBe(Infinity);
  });

  test("a builtin's int result is held to the cap", () => {
    expect(evaluate(`int("${"9".repeat(4300)}")`)).toBe(10n ** 4300n - 1n);
    expect(evaluate(`int("${"0".repeat(5000)}1")`)).toBe(1n);
    expect(messageOf(`int("${"1".repeat(4301)}")`)).toBe(`expression function 'int' failed: int() ${OVER}`);
    expect(messageOf("sum([9 * 10 ** 4299, 10 ** 4299])")).toBe(`expression function 'sum' failed: sum() ${OVER}`);
    expect(messageOf("abs(x)", { x: -(10n ** 5000n) })).toBe(`expression function 'abs' failed: abs() ${OVER}`);
  });

  test("a host function's int result is held to the cap", () => {
    const saved = TEMPLATE_FNS.get("huge");
    registerTemplateFn("huge", { params: [], call: () => 10n ** 5000n });
    try {
      expect(messageOf("huge()")).toBe(`expression function 'huge' failed: huge() ${OVER}`);
    } finally {
      if (saved === undefined) TEMPLATE_FNS.delete("huge");
      else TEMPLATE_FNS.set("huge", saved);
    }
  });

  test("a raised cap lets a larger int through", () => {
    expect(String(evaluate("10 ** 4999", {}, { maxIntDigits: 5000 }))).toHaveLength(5000);
    expect(messageOf("10 ** 5000", {}, { maxIntDigits: 5000 })).toBe(
      "10 ** 5000 gives an int over the limit of 5000 digits (YIAN_MAX_INT_DIGITS)",
    );
  });

  test("past the cap, the runtime's own refusal still names itself", () => {
    expect(messageOf("2 ** (2 ** 40)", {}, { maxIntDigits: Number.MAX_SAFE_INTEGER })).toBe(
      "integer result is too large for this runtime",
    );
  });
});

describe("YIAN_MAX_JSON_INDENT", () => {
  test("render_as_json takes an indent up to the cap", () => {
    expect(evaluate("render_as_json([1], 10)")).toBe(`[\n${" ".repeat(10)}1\n]`);
    expect(messageOf("render_as_json([1], 11)")).toBe(
      "expression function 'render_as_json' failed: render_as_json() takes an indent of at most 10 " +
        "(YIAN_MAX_JSON_INDENT), got 11",
    );
  });

  test("a huge indent is refused before any text is written", () => {
    const started = performance.now();
    expect(messageOf("render_as_json(xs, 100000000)", FOUR)).toEndWith("(YIAN_MAX_JSON_INDENT), got 100000000");
    expect(performance.now() - started).toBeLessThan(1_000);
  });

  test("a raised cap takes a wider indent", () => {
    expect(evaluate("render_as_json([1], 12)", {}, { maxJsonIndent: 12 })).toBe(`[\n${" ".repeat(12)}1\n]`);
  });
});

describe("YIAN_MAX_TEXT_CHARS", () => {
  const over = (cap: number) => `gives a text over the limit of ${cap} characters (YIAN_MAX_TEXT_CHARS)`;
  const TEN = { maxTextChars: 10 };

  test("text + may give the cap's characters, and not one more", () => {
    expect(evaluate('"abcde" + "fghij"', {}, TEN)).toBe("abcdefghij");
    expect(messageOf('"abcde" + "fghij" + "k"', {}, TEN)).toBe(`"abcdefghij" + "k" ${over(10)}`);
    // Characters are code points: ten emoji fit, though they take twenty UTF-16 units.
    expect(evaluate(`"${"😀".repeat(6)}" + "${"😀".repeat(4)}"`, {}, TEN)).toBe("😀".repeat(10));
  });

  test("text that doubles through comprehension variables stops at the cap", () => {
    // Each level binds one item and doubles its text: 2^24 copies of "ab" at the top.
    let expr = "x";
    for (let i = 0; i < 24; i++) expr = `[y${i} + y${i} for y${i} in [${expr}]][0]`;
    const started = performance.now();
    expect(messageOf(expr, { x: "ab" })).toEndWith(over(10_000_000));
    expect(performance.now() - started).toBeLessThan(5_000);
  });

  test("replace is sized from its operands before it builds its text", () => {
    expect(evaluate('replace("aaaa", "a", "bb")', {}, { maxTextChars: 8 })).toBe("bbbbbbbb");
    expect(messageOf('replace("aaaa", "a", "bb")', {}, { maxTextChars: 7 })).toBe(
      `expression function 'replace' failed: replace() ${over(7)}`,
    );
    // An empty `old` puts `new` before each character and at the end.
    expect(evaluate('replace("ab", "", "xy")', {}, { maxTextChars: 8 })).toBe("xyaxybxy");
    expect(messageOf('replace("ab", "", "xy")', {}, { maxTextChars: 7 })).toEndWith(over(7));
    expect(evaluate('replace("😀😀", "😀", "é")', {}, { maxTextChars: 2 })).toBe("éé");
    const s = "a".repeat(700);
    const started = performance.now();
    expect(messageOf('replace(replace(s, "", s), "", s)', { s })).toBe(
      `expression function 'replace' failed: replace() ${over(10_000_000)}`,
    );
    expect(performance.now() - started).toBeLessThan(1_000);
  });

  test("join is sized from its items and sep before it builds its text", () => {
    expect(evaluate('join(["ab", "cd"], "--")', {}, { maxTextChars: 6 })).toBe("ab--cd");
    expect(messageOf('join(["ab", "cd"], "--")', {}, { maxTextChars: 5 })).toBe(
      `expression function 'join' failed: join() ${over(5)}`,
    );
    // A list item is written as compact JSON.
    expect(evaluate('join([[1, 2]], "")', {}, { maxTextChars: 5 })).toBe("[1,2]");
    expect(messageOf('join([[1, 2]], "")', {}, { maxTextChars: 4 })).toEndWith(over(4));
    const xs = Array.from({ length: 1000 }, () => "x");
    expect(messageOf("join(xs, sep)", { xs, sep: "-".repeat(20_000) })).toEndWith(over(10_000_000));
  });

  test("str and render_as_json are sized before they write, so nested escaping can't double past the cap", () => {
    expect(evaluate("render_as_json([1, 2])", {}, { maxTextChars: 12 })).toBe("[\n  1,\n  2\n]");
    expect(messageOf("render_as_json([1, 2])", {}, { maxTextChars: 11 })).toBe(
      `expression function 'render_as_json' failed: render_as_json() ${over(11)}`,
    );
    // `a\` is written `["a\\"]`: seven characters.
    expect(evaluate("str([s])", { s: "a\\" }, { maxTextChars: 7 })).toBe('["a\\\\"]');
    expect(messageOf("str([s])", { s: "a\\" }, { maxTextChars: 6 })).toBe(
      `expression function 'str' failed: str() ${over(6)}`,
    );
    // Each level writes the one below as a quoted text, doubling its backslashes.
    let expr = "s";
    for (let i = 0; i < 40; i++) expr = `render_as_json([${expr}], 0)`;
    const started = performance.now();
    expect(messageOf(expr, { s: "\\" })).toEndWith(over(10_000_000));
    expect(performance.now() - started).toBeLessThan(5_000);
  });

  test("any builtin's text result is held to the cap", () => {
    // Upper-casing ß gives SS, so the text grows.
    expect(messageOf(`upper("${"ß".repeat(6)}")`, {}, TEN)).toBe(`expression function 'upper' failed: upper() ${over(10)}`);
    const saved = TEMPLATE_FNS.get("long_text");
    registerTemplateFn("long_text", { params: [], call: () => "x".repeat(11) });
    try {
      expect(messageOf("long_text()", {}, TEN)).toBe(`expression function 'long_text' failed: long_text() ${over(10)}`);
    } finally {
      if (saved === undefined) TEMPLATE_FNS.delete("long_text");
      else TEMPLATE_FNS.set("long_text", saved);
    }
  });

  test("reading, comparing or measuring a longer text is fine", () => {
    const long = { x: "y".repeat(20) };
    expect(evaluate("x", long, TEN)).toBe("y".repeat(20));
    expect(evaluate('x == "y"', long, TEN)).toBe(false);
    expect(evaluate("len(x)", long, TEN)).toBe(20n);
    expect(evaluate("x[0:5]", long, TEN)).toBe("yyyyy");
  });
});
