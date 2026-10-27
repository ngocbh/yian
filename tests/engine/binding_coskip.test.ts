/**
 * The co-skip classifier, a port of `tests/engine/test_binding_coskip.py`: does a binding
 * co-skip when all its producers are skipped (`bindingCoSkips`)?
 */

import { expect, test } from "bun:test";

import { bindingCoSkips } from "../../src/expr/index.ts";

test("a plain reference co-skips", () => {
  expect(bindingCoSkips("${a.output}")).toBe(true);
});

test("a coalesce of references co-skips", () => {
  expect(bindingCoSkips("${a.output | b.output}")).toBe(true);
});

test("a reference default co-skips", () => {
  expect(bindingCoSkips("${a.output :- b.output}")).toBe(true); // the default is a reference
});

test("a literal default does not co-skip", () => {
  expect(bindingCoSkips("${a.output:-null}")).toBe(false);
});

test("a literal coalesce operand does not co-skip", () => {
  expect(bindingCoSkips('${a.output | "fallback"}')).toBe(false);
});

test("a required does not co-skip", () => {
  expect(bindingCoSkips("${a.output:?missing}")).toBe(false);
});

test("embedded text does not co-skip", () => {
  expect(bindingCoSkips("got ${a.output} here")).toBe(false);
});

test("a value that isn't text, or isn't a reference, does not co-skip", () => {
  expect(bindingCoSkips(30)).toBe(false);
  expect(bindingCoSkips("plain literal")).toBe(false);
  expect(bindingCoSkips("${input.x}")).toBe(true); // a whole-text reference still co-skips structurally
});

test("an arithmetic span does not co-skip", () => {
  // A computed span (not a pure reference group) is not a hard data dependency.
  expect(bindingCoSkips("${a.output + 1}")).toBe(false);
});

test("a builtin call span does not co-skip", () => {
  // A builtin call span is a computed value, not a pure reference, so it does not co-skip.
  expect(bindingCoSkips("${upper(a.output)}")).toBe(false);
});

test("a bare reference default co-skips", () => {
  // `:-b.output` (a bare-reference default) keeps co-skipping, like `:-${b.output}`.
  expect(bindingCoSkips("${a.output:-b.output}")).toBe(true);
});
