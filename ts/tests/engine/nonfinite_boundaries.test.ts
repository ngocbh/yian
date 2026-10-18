/**
 * Non-finite numbers at the binding boundary, a port of the two `apply_defaults` tests of
 * `tests/engine/test_nonfinite_boundaries.py`.
 *
 * Changed (rule: infinity and minus infinity are float values, NaN is not a yian value):
 * `test_apply_defaults_rejects_a_nonfinite_default` now shows an `Infinity` default accepted
 * and keeps the rejection as a NaN row; a NaN default fails with host intake's message
 * (`default for 'p': NaN is not a yian value`), also before its contract is checked.
 *
 * Later: the other 20 tests run flows or agents and land with the runner and the node kinds.
 */

import { expect, test } from "bun:test";

import { applyDefaults, BindingError } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";

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

test("applyDefaults accepts an infinite default and rejects a NaN default", () => {
  expect(applyDefaults([new Parameter("p", { hasDefault: true, default: Infinity })], {})).toEqual({ p: Infinity });

  const error = bindingError(() => applyDefaults([new Parameter("p", { hasDefault: true, default: NaN })], {}));
  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message).toBe("default for 'p': NaN is not a yian value");
});

test("applyDefaults rejects a NaN default before its contract", () => {
  const parameter = new Parameter("p", {
    contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
    hasDefault: true,
    default: NaN,
  });

  const error = bindingError(() => applyDefaults([parameter], {}));
  expect(error.message).toBe("default for 'p': NaN is not a yian value");
});
