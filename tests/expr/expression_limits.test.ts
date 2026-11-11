/**
 * The caps that bound one expression's work: the items its comprehensions bind
 * (`YIAN_MAX_COMPREHENSION_ITEMS`), the digits of an int an operator or a builtin gives
 * (`YIAN_MAX_INT_DIGITS`), and the indent `render_as_json` takes (`YIAN_MAX_JSON_INDENT`). Going
 * over one is an `ExpressionError` that names it, raised before the work it would take.
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
