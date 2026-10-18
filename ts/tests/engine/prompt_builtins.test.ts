/**
 * Prompt builtins, a port of `tests/engine/test_prompt_builtins.py`: `renderTemplateRecord`
 * evaluates a prompt span as a plain reference, a builtin call (positional and keyword
 * arguments, optional dotted access on the result), or a larger expression, and raises on a miss
 * or a final `null`; a call-result dotted miss coalesces, defaults or is falsy outside strict
 * mode; `promptRefs` collects the references a prompt reads and rejects an unknown builtin; and
 * the result type a shipped builtin always returns (`builtinResultType`).
 *
 * The values follow the TS number model (ints are `bigint`s). Python's `_mkrec`, registered for
 * good, is a stand-in registered for one test each.
 *
 * Left for later: the loader tests land with compose, and
 * `test_prompt_default_fires_in_a_running_flow` with `run.ts`.
 */

import { afterEach, expect, test } from "bun:test";

import {
  builtinResultType,
  registerTemplateFn,
  TEMPLATE_FNS,
  type TemplateFn,
  type TemplateParam,
} from "../../src/expr/builtins.ts";
import { ExpressionError, evalExpr, RequiredError, ResolveMode, resolveInRecord } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { evalBinding, promptRefs, renderTemplateRecord } from "../../src/expr/template.ts";
import { Type, type Value, ValueKind, type ValueRecord } from "../../src/typesys/index.ts";

const REC: ValueRecord = { briefs: ["ab", "adf"], name: "zeta", sig: { value: 0.8 } };

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` taking `params` and running `call`, for this test only. */
function standIn(name: string, params: TemplateParam[], call: (...args: Value[]) => unknown): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** Register `_mkrec(x)`, giving `{field: "F-<x>"}`, for this test only. */
function mkrec(): void {
  standIn("_mkrec", [{ name: "x" }], (x) => ({ field: `F-${String(x)}` }));
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

/** Expect `renderTemplateRecord(template, record)` to throw an `ExpressionError`. */
function expectRenderError(template: string, record: ValueRecord = REC): Error {
  const error = errorOf(() => renderTemplateRecord(template, record));
  expect(error).toBeInstanceOf(ExpressionError);
  return error;
}

// --- render: builtin calls --- //

test("render_as_json with a positional reference and literal", () => {
  // `briefs` gives the typed list, `4` the int `indent`.
  expect(renderTemplateRecord("J=${render_as_json(briefs, 4)}", REC)).toBe('J=[\n    "ab",\n    "adf"\n]');
});

test("render_as_json in the keyword form", () => {
  expect(renderTemplateRecord("J=${render_as_json(value=briefs, indent=4)}", REC)).toBe(
    'J=[\n    "ab",\n    "adf"\n]',
  );
});

test("join with a keyword sep", () => {
  expect(renderTemplateRecord("${join(briefs, sep=', ')}", REC)).toBe("ab, adf");
});

test("upper and lower over a reference", () => {
  expect(renderTemplateRecord("${upper(name)}/${lower('HI')}", REC)).toBe("ZETA/hi");
});

test("dotted access on a call's result", () => {
  mkrec();
  expect(renderTemplateRecord("${_mkrec(name).field}", REC)).toBe("F-zeta");
});

// --- render: arithmetic and the dollar escape --- //

test("arithmetic in a prompt span", () => {
  expect(renderTemplateRecord("total ${a + 1}", { a: 4n })).toBe("total 5");
});

test("$$ renders one $", () => {
  expect(renderTemplateRecord("cost is $$5", {})).toBe("cost is $5");
});

// --- render: the error floor --- //

test("an unknown builtin raises", () => {
  expectRenderError("${frobnicate(name)}");
});

test("a bare word call argument is a reference", () => {
  expect(renderTemplateRecord("${render_as_json(briefs, 4)}", REC)).toBe('[\n    "ab",\n    "adf"\n]');
});

test("a call argument reference is strict on a miss", () => {
  expectRenderError("${render_as_json(ghost)}");
});

test("a dotted miss on a call's result raises", () => {
  mkrec();
  expectRenderError("${_mkrec(name).nope}");
});

test("an embedded plain reference's dotted miss raises", () => {
  // Never a silent blank in the multi-span path.
  expectRenderError("x ${sig.nope} y");
});

test("an embedded call result's dotted miss raises", () => {
  mkrec();
  expectRenderError("x ${_mkrec(name).nope} y");
});

test.each(["null", "null | null"])("a strict prompt rejects a final null, whole or embedded: %s", (expression) => {
  const messages = [`\${${expression}}`, `value=\${${expression}}`].map(
    (template) => expectRenderError(template, {}).message,
  );
  expect(messages).toEqual(["null value in strict template", "null value in strict template"]);
});

test("a binding's embedded null is still empty text", () => {
  expect(evalBinding("${null}", () => null)).toBeNull();
  expect(evalBinding("value=${null}", () => null)).toBe("value=");
});

test("a strict prompt allows a null intermediate with a non-null result", () => {
  expect(renderTemplateRecord("value=${null | 'fallback'}", {})).toBe("value=fallback");
});

test.each<[string, ValueRecord, string]>([
  ['Note: ${note :- "none"}', { note: null }, "Note: none"],
  ['Note: ${note :- "none"}', {}, "Note: none"],
  ['Note: ${note :- "none"}', { note: "hi" }, "Note: hi"],
  ["v=${sig.nope :- 0}", REC, "v=0"],
  ["Hi ${x | y}", { x: null, y: "Y" }, "Hi Y"],
  ["Hi ${x | y | 'z'}", { x: null, y: null }, "Hi z"],
  ["${x | y}", { x: null, y: "Y" }, "Y"],
])("a strict prompt's default and coalesce fire on a miss: %s", (template, record, expected) => {
  expect(renderTemplateRecord(template, record)).toBe(expected);
});

test("a strict prompt's required raises its own message on a miss", () => {
  const error = errorOf(() => renderTemplateRecord('Hi ${x :? "need x"}', { x: null }));
  expect(error).toBeInstanceOf(RequiredError);
  expect(error.message).toContain("need x");
});

test.each(["${x :- y}", "Hi ${x | y}", "${x :- null}"])(
  "a strict prompt still raises when the fallback misses: %s",
  (template) => {
    expectRenderError(template, { x: null, y: null });
  },
);

test("a strict prompt's miss error names the reference", () => {
  expect(expectRenderError("x ${sig.nope} y").message).toContain("${sig.nope}");
  expect(expectRenderError("x ${upper(ghost)} y").message).toContain("${ghost}");
});

// --- render: plain references --- //

test("plain and dotted references render", () => {
  expect(renderTemplateRecord("${name} v=${sig.value}", REC)).toBe("zeta v=0.8");
});

test("a plain reference that is unknown or null raises", () => {
  expectRenderError("${ghost}");
  expectRenderError("${t}", { t: null });
});

test("a span nested in a call shows the rewrite", () => {
  // Inside a span, a builtin's argument names the input directly.
  expect(expectRenderError("[${render_as_json(${briefs})}]").message).toContain("write `${render_as_json(briefs)}`");
});

// --- promptRefs, the compile-time companion --- //

test("promptRefs collects plain and argument references", () => {
  expect(promptRefs("a ${name} b ${render_as_json(briefs, 4)} c ${join(sig.value)}")).toEqual([
    "name",
    "briefs",
    "sig.value",
  ]);
});

test("promptRefs skips literals", () => {
  expect(promptRefs("${render_as_json(briefs, 4)} ${lower('HI')}")).toEqual(["briefs"]);
});

test("promptRefs rejects an unknown builtin", () => {
  expect(() => promptRefs("${frobnicate(name)}")).toThrow(ExpressionError);
});

// --- mode interaction for call-result dotted misses --- //
// A call-result dotted miss is a miss: non-strict modes may coalesce, default, or treat it as
// falsy; only strict mode raises.

/** Return a `resolve` doing the dotted walk over `record` (a miss is `null`). */
function recordResolve(record: ValueRecord): (path: string) => Value {
  return (path) => resolveInRecord(path, record);
}

test("a binding coalesce falls through a call-result miss", () => {
  mkrec();
  expect(evalBinding("${_mkrec(name).nope | 'fallback'}", recordResolve(REC))).toBe("fallback");
});

test("a binding default fires on a call-result miss", () => {
  mkrec();
  expect(evalBinding('${_mkrec(name).nope :- "x"}', recordResolve(REC))).toBe("x");
});

test("a condition's call-result miss stays falsy", () => {
  mkrec();
  const tree = parseExpr("_mkrec(name).nope == 1");
  expect(evalExpr(tree, recordResolve(REC), { mode: ResolveMode.CONDITION_FALSY })).toBe(false);
});

// --- result types --- //

test.each(["render_as_json", "join", "upper", "lower"])("the result type of %s is text", (name) => {
  expect(builtinResultType(name)).toEqual(Type.scalar(ValueKind.STRING));
});

test("an unknown name has no result type", () => {
  expect(builtinResultType("frobnicate")).toBeNull();
});

test("a host override has no result type", () => {
  standIn("upper", [{ name: "s" }], (s) => ({ f: s }));
  expect(builtinResultType("upper")).toBeNull();
});
