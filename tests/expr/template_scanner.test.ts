/**
 * The template scanner over the one expression parser.
 *
 * `scanTemplate(text)` splits a text into `Literal` runs (raw text, `$$` already collapsed to
 * `$`) and `Span`s whose interior went through `parseExpr`. Only span interiors are parsed, so
 * operator characters (`|`, `+`, `[`) in literal text stay literal, which keeps free text like
 * `"stance (positive|negative|neutral)"` safe.
 *
 * `evalTemplate(segments, resolve, {item, mode})` gives the typed value of a whole single span,
 * else writes each span's value in the text form and joins it with the literal runs.
 * `templateRefs(segments)` unions the span references in source order.
 *
 * Ints are `bigint`s with no digit limit, so a huge int renders its digits, and an int the runtime
 * can't stringify fails with its own message. A text-form table checks what an embedded span
 * writes.
 */

import { expect, test } from "bun:test";

import { ExpressionError, MISSING_ROOT, ResolveMode, type RootResolver } from "../../src/expr/expressions.ts";
import {
  evalTemplate,
  Literal,
  renderTemplateRecord,
  scanTemplate,
  Span,
  templateRefs,
} from "../../src/expr/template.ts";
import { FileRef, type Value, type ValueRecord } from "../../src/typesys/index.ts";
import { fieldOf } from "../../src/typesys/values.ts";

/** Return a `resolve` over `values`: the head, then `fieldOf` per dotted step; a miss is `null`. */
function from(values: ValueRecord): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(values, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Return the error `run` throws, failing if it returns. */
function errorOf(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("no error was thrown");
}

/** Expect `run` to throw an `ExpressionError` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string): void {
  const error = errorOf(run);
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe(message);
}

/** Scan and evaluate `text`. */
function evaluate(
  text: string,
  resolve: RootResolver = MISSING_ROOT,
  options: { item?: Value; mode?: ResolveMode } = {},
): Value {
  return evalTemplate(scanTemplate(text), resolve, options);
}

// --- a whole single span keeps its type --- //

test("a whole single span keeps a float", () => {
  expect(evaluate("${score}", from({ score: 0.5 }))).toBe(0.5);
});

test("a whole single span keeps a list", () => {
  expect(evaluate("${items}", from({ items: [1n, 2n, 3n] }))).toEqual([1n, 2n, 3n]);
});

test("a whole single span that misses is null", () => {
  expect(evaluate("${gone}")).toBeNull();
});

// --- an embedded span is written as text --- //

test("an embedded span is written as text", () => {
  expect(evaluate("n=${score}", from({ score: 0.5 }))).toBe("n=0.5");
});

test("two embedded spans keep the literal runs between them", () => {
  expect(evaluate("a ${b} c ${d}", from({ b: "B", d: "D" }))).toBe("a B c D");
});

test("an embedded span that misses is empty text", () => {
  expect(evaluate("x=${gone}!")).toBe("x=!");
});

// --- literal text is never parsed --- //

test("a pure literal keeps its operator characters", () => {
  const text = "stance (positive|negative|neutral)";
  expect(evaluate(text)).toBe(text);
});

test("$$ collapses to one $", () => {
  expect(evaluate("$$")).toBe("$");
});

test("a lone $ stays literal", () => {
  expect(evaluate("cost is $5")).toBe("cost is $5");
});

test("operator characters outside a span stay literal with a span present", () => {
  expect(evaluate("pick a|b|c: ${choice} + [x]", from({ choice: "b" }))).toBe("pick a|b|c: b + [x]");
});

test("a span interior is parsed", () => {
  // Inside the span `|` is a coalesce.
  expect(evaluate("${a | b}", from({ b: "fallback" }))).toBe("fallback");
});

// --- segment shapes --- //

test("a pure literal is one Literal", () => {
  expect(scanTemplate("just text")).toEqual([new Literal("just text")]);
});

test("$$ collapses inside a literal", () => {
  expect(scanTemplate("a$$b")).toEqual([new Literal("a$b")]);
});

test("two spans and literals", () => {
  const segments = scanTemplate("a ${b} c ${d}");
  expect(segments.map((s) => s.constructor.name)).toEqual(["Literal", "Span", "Literal", "Span"]);
  expect((segments[0] as Literal).text).toBe("a ");
  expect((segments[2] as Literal).text).toBe(" c ");
});

test("a whole single span is one Span", () => {
  const segments = scanTemplate("${score}");
  expect(segments).toHaveLength(1);
  expect(segments[0]).toBeInstanceOf(Span);
});

test("an unbalanced span raises", () => {
  expectError(() => scanTemplate("${a"), "unbalanced '${' in template '${a'");
});

test("an empty template has no segments and evaluates to empty text", () => {
  expect(scanTemplate("")).toEqual([]);
  expect(evaluate("")).toBe("");
});

test("an empty span raises", () => {
  expect(() => scanTemplate("${}")).toThrow(ExpressionError);
  expect(() => scanTemplate("${   }")).toThrow(ExpressionError);
});

test("adjacent spans are two spans and concatenate", () => {
  // Not one whole span, so each is written as text.
  expect(scanTemplate("${a}${b}").map((s) => s.constructor.name)).toEqual(["Span", "Span"]);
  expect(evaluate("${a}${b}", from({ a: 1n, b: 2n }))).toBe("12");
});

// --- templateRefs --- //

test("templateRefs keeps source order without dedupe", () => {
  expect(templateRefs(scanTemplate("${a} and ${b.c} and ${a}"))).toEqual(["a", "b.c", "a"]);
});

test("templateRefs of a pure literal is empty", () => {
  expect(templateRefs(scanTemplate("no refs (a|b)"))).toEqual([]);
});

test("templateRefs ignores the literals between spans", () => {
  expect(templateRefs(scanTemplate("x ${one} y ${two}"))).toEqual(["one", "two"]);
});

// --- item scope and mode --- //

test("evalTemplate reads the item scope", () => {
  expect(evaluate("${item.title}", undefined, { item: { title: "T" } })).toBe("T");
});

test("evalTemplate in strict mode raises on a miss", () => {
  expect(() => evaluate("${gone}", undefined, { mode: ResolveMode.STRICT_RAISE })).toThrow(ExpressionError);
});

// --- the text form of an embedded span --- //

const HUGE = 10n ** 5000n;

test("an embedded huge int renders its digits", () => {
  expect(evaluate("x=${n}", from({ n: HUGE }))).toBe(`x=${HUGE.toString()}`);
});

test("an embedded list holding a huge int renders it", () => {
  expect(evaluate("x=${n}", from({ n: [HUGE] }))).toBe(`x=[${HUGE.toString()}]`);
});

test("a whole single span keeps a huge int typed", () => {
  expect(evaluate("${n}", from({ n: HUGE }))).toBe(HUGE);
});

test("renderTemplateRecord writes a whole-span huge int's digits", () => {
  expect(renderTemplateRecord("${n}", { n: HUGE })).toBe(HUGE.toString());
});

test("an int the runtime can't stringify is an ExpressionError", () => {
  const toString = Object.getOwnPropertyDescriptor(BigInt.prototype, "toString")!;
  Object.defineProperty(BigInt.prototype, "toString", {
    ...toString,
    value: () => {
      throw new RangeError("Maximum BigInt size exceeded");
    },
  });
  try {
    expectError(() => evaluate("x=${n}", from({ n: 1n })), "integer is too large for this runtime");
  } finally {
    Object.defineProperty(BigInt.prototype, "toString", toString);
  }
});

test.each<[string, Value, string]>([
  ["true", true, "true"],
  ["the float 1.0", 1.0, "1"],
  ["inf", Infinity, "Infinity"],
  ["-inf", -Infinity, "-Infinity"],
  ["a large float", 1e21, "1e+21"],
  ["a small float", 1e-7, "1e-7"],
  ["minus zero", -0, "0"],
  ["a big int", 10n ** 25n, "10000000000000000000000000"],
  ["a list", [1n, "a"], '[1,"a"]'],
  ["a record", { k: "v" }, '{"k":"v"}'],
  ["a record in JS key order", { b: 1n, "2": 2n, a: 3n }, '{"2":2,"b":1,"a":3}'],
  ["a file", new FileRef({ uri: "s3://b/k" }), '{"uri":"s3://b/k","mime":null,"name":null}'],
  ["a nested null", [null], "[null]"],
  ["a date text", "2024-01-02", "2024-01-02"],
])("an embedded span writes %s in the text form", (_label, value, text) => {
  expect(evaluate("v=${x}", from({ x: value }))).toBe(`v=${text}`);
});
