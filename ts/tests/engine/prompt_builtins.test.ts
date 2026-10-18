/**
 * Prompt builtins, a port of `tests/engine/test_prompt_builtins.py`: the result type a shipped
 * builtin always returns (`builtinResultType`).
 *
 * Left for later: the render and `prompt_refs` tests land with `template.ts`, the loader tests
 * with compose.
 */

import { afterEach, expect, test } from "bun:test";

import {
  builtinResultType,
  registerTemplateFn,
  TEMPLATE_FNS,
  type TemplateFn,
  type TemplateParam,
} from "../../src/expr/builtins.ts";
import { Type, type Value, ValueKind } from "../../src/typesys/index.ts";

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
