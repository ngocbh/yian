/**
 * The CODE node's source classification and its inline line-origin selector: a port of
 * `tests/engine/test_code_inline.py`.
 *
 * Ported (adapted): `test_classify_reference_inline_reject` and
 * `test_wrapped_line_selector_uses_exact_whole_then_node_fallback` (its `True` line is `true`).
 *
 * Changed: a reference is a relative path to a `.ts`, `.js` or `.mjs` file plus `:fn`
 * (`./mod.ts:fn`); any other value is an inline body, so there is no reject bucket and
 * `./helpers.ts`, `pkg.mod:fn` and `./x.py:fn` classify as inline. The selector takes the line of
 * the authored body (line 1 is the body's first line), not the line of Python's
 * `def main(inputs):` wrapper.
 *
 * TS-only: the constructor compiles nothing, so a body that doesn't parse constructs; the node's
 * `mode`; a non-integer line selects no line; the selector's origin shape checks; `run` rejects as
 * not ported yet; the built-in identity form a value entry lowers to (its `"builtin"` mode, its
 * frozen copy, and the shape and type checks on `code`).
 *
 * Not ported: `test_inline_stdlib_import_in_body_works_in_process` (a Python `import`);
 * `test_no_return_body_rejected_at_construction` and `test_return_only_in_nested_def_still_rejected`
 * (Python `ast`; TS has no load-time `return` check);
 * `test_structurally_equal_foreign_code_cannot_impersonate_owned_code` (Python code objects);
 * `test_dynamic_same_filename_code_falls_back_to_owned_call_site` (Python `compile` and `exec`);
 * `test_reference_failure_does_not_inspect_origin_truthiness` (Python `__bool__`).
 *
 * Later (T4-9): `test_inline_reads_inputs_dict_and_returns`,
 * `test_inline_calling_convention_matches_reference`, `test_inline_multi_statement_body_with_helpers`,
 * `test_inline_body_that_prints_still_returns_value`,
 * `test_inline_exception_selects_deepest_owned_code_line`,
 * `test_inline_non_owned_failure_uses_whole_code_origin`,
 * `test_reference_failure_uses_code_then_node_fallback`, `test_inline_return_comes_back_unchecked`,
 * `test_reference_mode_still_dispatches_in_process`, `test_inline_flow_loads_and_runs` and
 * `test_inline_raise_becomes_failed_run`.
 *
 * Later (T4-9), adapted: `test_syntax_error_rejected_at_construction`; the syntax check moves to
 * the awaited load step.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { classifyCodeSource, CodeNode, selectInlineCodeOrigin } from "../../src/nodes/code/node.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

test("classify reference inline reject", () => {
  expect(classifyCodeSource("./mod.ts:fn")).toBe("reference");
  expect(classifyCodeSource("../lib/x.mjs:run")).toBe("reference");
  expect(classifyCodeSource("./a/b.js:f")).toBe("reference");
  expect(classifyCodeSource("return 1")).toBe("inline");
  expect(classifyCodeSource("const x = 1\nreturn x")).toBe("inline");
  // No `:fn`, a Python-style token and a non-TS file are bodies, not references.
  expect(classifyCodeSource("./helpers.ts")).toBe("inline");
  expect(classifyCodeSource("pkg.mod:fn")).toBe("inline");
  expect(classifyCodeSource("./x.py:fn")).toBe("inline");
});

test("wrapped line selector uses exact, whole, then node fallback", () => {
  const whole = new OriginRef("code");
  const lines = [new OriginRef("line:0"), new OriginRef("line:1")];
  const node = new OriginRef("node");

  expect(selectInlineCodeOrigin(2, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: node })).toBe(lines[1]!);
  expect(selectInlineCodeOrigin(0, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: node })).toBe(whole);
  expect(selectInlineCodeOrigin(3, { codeOrigin: whole, codeLineOrigins: lines.slice(0, 1), nodeOrigin: node })).toBe(
    whole,
  );
  expect(selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: node })).toBe(node);
  expect(selectInlineCodeOrigin(true, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: null })).toBeNull();
});

// --- TS-only rows --- //

test("a non-integer line selects no line", () => {
  const whole = new OriginRef("code");
  const lines = [new OriginRef("line:0"), new OriginRef("line:1")];

  for (const line of [1.5, "1", null]) {
    expect(selectInlineCodeOrigin(line, { codeOrigin: whole, codeLineOrigins: lines, nodeOrigin: null })).toBe(whole);
  }
});

test("the selector rejects origin carriers of the wrong shape", () => {
  const line = new OriginRef("line:0");
  const bad = "not-an-origin" as unknown as OriginRef;

  expect(() => selectInlineCodeOrigin(1, { codeOrigin: bad, codeLineOrigins: [], nodeOrigin: null })).toThrow(
    new TypeError("CODE source origin must be an OriginRef or null"),
  );
  expect(() => selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [], nodeOrigin: bad })).toThrow(
    new TypeError("node origin must be an OriginRef or null"),
  );
  expect(() =>
    selectInlineCodeOrigin(1, {
      codeOrigin: null,
      codeLineOrigins: line as unknown as OriginRef[],
      nodeOrigin: null,
    }),
  ).toThrow(new TypeError("CODE line origins must be an array"));
  expect(() => selectInlineCodeOrigin(1, { codeOrigin: null, codeLineOrigins: [line, bad], nodeOrigin: null })).toThrow(
    new TypeError("CODE line origins must contain only OriginRef values"),
  );
});

test("the constructor compiles nothing: a body that doesn't parse constructs", () => {
  const node = new CodeNode("n", { code: "return (" });

  expect(node.code).toBe("return (");
  expect(node.mode).toBe("inline");
});

test("the node's mode follows the classification", () => {
  expect(new CodeNode("r", { code: "./mod.ts:fn" }).mode).toBe("reference");
  expect(new CodeNode("i", { code: "return inputs.v" }).mode).toBe("inline");
  expect(new CodeNode("p", { code: "./helpers.ts" }).mode).toBe("inline");
});

test("a CODE forward is not ported yet", async () => {
  await expect(new CodeNode("c", { code: "return 1" }).run({})).rejects.toThrow(
    "node 'c' (CodeNode) forward is not ported yet (T4-9)",
  );
});

test("the built-in identity form has builtin mode and keeps a frozen copy", () => {
  const form = { builtin: "identity" as const, field: "x" };
  const node = new CodeNode("x", { code: form });

  expect(node.mode).toBe("builtin");
  expect(node.code).not.toBe(form);
  expect(node.code).toEqual(form);
  expect(Object.isFrozen(node.code)).toBe(true);
});

test("a CODE built-in of the wrong shape is rejected", () => {
  const message = 'CODE built-in must be {builtin: "identity", field: <input name>}';
  for (const code of [
    { builtin: "other", field: "x" },
    { builtin: "identity" },
    { builtin: "identity", field: "x", extra: 1 },
    { builtin: "identity", field: 1 },
  ]) {
    expectError(() => new CodeNode("x", { code: code as never }), TypeError, message);
  }
  expectError(
    () => new CodeNode("x", { code: { builtin: "identity", field: "" } }),
    RangeError,
    "CODE built-in field must be non-empty",
  );
});

test("a CODE code that is neither text nor a built-in is rejected", () => {
  expectError(
    () => new CodeNode("x", { code: 42 as never }),
    TypeError,
    "CODE code must be text or a built-in operation, got number",
  );
  expectError(
    () => new CodeNode("x", { code: [] as never }),
    TypeError,
    "CODE code must be text or a built-in operation, got Array",
  );
});
