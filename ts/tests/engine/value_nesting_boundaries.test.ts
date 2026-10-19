/**
 * Value nesting at the binding boundary, a port of
 * `test_apply_defaults_rejects_a_default_over_the_limit` and
 * `test_bind_params_rejects_a_filled_default_over_the_limit` of
 * `tests/engine/test_value_nesting_boundaries.py`.
 *
 * Later: the other 39 tests run flows, agents or loads and land with the runner, the node kinds
 * and compose.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { applyDefaults, BindingError, bindParams } from "../../src/nodes/binding.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { type Value, VariablePool } from "../../src/typesys/index.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** Return `"leaf"` wrapped in `levels` records under `a`. */
function nestRecord(levels: number): Value {
  let value: Value = "leaf";
  for (let i = 0; i < levels; i++) value = { a: value };
  return value;
}

/** Return the `BindingError` `call` throws. */
function bindingError(call: () => unknown): BindingError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("call did not throw");
}

test("applyDefaults rejects a default over the limit", () => {
  const error = bindingError(() =>
    applyDefaults([new Parameter("p", { hasDefault: true, default: nestRecord(101) })], {}),
  );
  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});

test("bindParams rejects a filled default over the limit", () => {
  const parameter = new Parameter("p", { hasDefault: true, default: nestRecord(101) });

  const error = bindingError(() => bindParams([parameter], new Map(), new VariablePool()));

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});
