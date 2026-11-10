/**
 * The span call check: every call in a `${...}` must name a builtin that takes its arguments
 * (`checkTemplateCalls`), and a prompt gets the same check through `promptRefs`. The
 * unknown-prompt-function message is pinned in full, which pins the order of the builtin names and
 * that no test registration leaks into `TEMPLATE_FNS`; a prompt can't call `len`, which computes
 * rather than formats.
 */

import { expect, test } from "bun:test";

import { checkTemplateCalls, ExpressionError, promptRefs } from "../../src/expr/index.ts";

const UPPER = "expression function 'upper' takes (s): too many positional arguments";

/** Return the error `run` throws, failing if it returns. */
function errorOf(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw an `ExpressionError` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string): void {
  const error = errorOf(run);
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe(message);
}

test.each([
  ["${upper(v, 1)}", UPPER],
  ["hi ${upper(v, 1)} there", UPPER],
  ["${v :- upper(v, 1)}", UPPER],
  ["${upper()}", "expression function 'upper' takes (s): missing a required argument: 's'"],
  [
    "${render_as_json(v, width=4)}",
    "expression function 'render_as_json' takes (value, indent=2): got an unexpected keyword argument 'width'",
  ],
  ["${upper(s=v, s=w)}", "upper(...) repeats the keyword 's'"],
])("the template call check rejects a bad call: %s", (text, message) => {
  expectError(() => checkTemplateCalls(text), message);
});

test.each([
  "plain text",
  "${v}",
  "${upper(v)} and ${join(xs, sep=',')}",
  "${isinstance(v, int)}",
  "${render_as_json(v, indent=4)}",
  "$${upper(v, 1)}",
  "${Buy(v)}",
  "${Buy(action='buy', qty=upper(v))}",
])("the template call check accepts good calls: %s", (text) => {
  checkTemplateCalls(text);
});

test("the template call check raises a parse error", () => {
  expect(() => checkTemplateCalls("${v +}")).toThrow(ExpressionError);
  expect(() => checkTemplateCalls("${v +}")).toThrow("could not parse");
});

test.each([
  ["Say ${upper(t, 1)}", UPPER],
  ["${render_as_json(t, width=4)}", "expression function 'render_as_json' takes"],
])("a prompt with a mis-called builtin is rejected: %s", (text, message) => {
  const error = errorOf(() => promptRefs(text));
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toContain(message);
});

test("a prompt's unknown function keeps its message", () => {
  const error = errorOf(() => promptRefs("${nosuch(t)}"));
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message.startsWith("unknown prompt function 'nosuch';")).toBe(true);
});

test("a prompt's unknown function lists the builtins in order", () => {
  expectError(
    () => promptRefs("${nosuch(t)}"),
    "unknown prompt function 'nosuch'; prompts call only formatting builtins (join, lower, lstrip, render_as_json, replace, rstrip, strip, upper); " +
      "build or check a typed value with a `typedefs:` constructor `T(...)` in an input binding",
  );
});

test("len is not a prompt function", () => {
  expectError(
    () => promptRefs("${len(t)}"),
    "unknown prompt function 'len'; prompts call only formatting builtins (join, lower, lstrip, render_as_json, replace, rstrip, strip, upper); " +
      "build or check a typed value with a `typedefs:` constructor `T(...)` in an input binding",
  );
});
