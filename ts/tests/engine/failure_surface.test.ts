/**
 * Failure producers carry semantic origins, never YAML coordinates: a port of
 * `tests/engine/test_failure_surface.py`. Ported: the instance half of
 * `test_failure_producers_have_no_location_parameters_or_attributes`.
 *
 * Later (T4-x / T4-8): `test_source_span_is_absent_from_runtime_events`,
 * `test_failure_carriers_have_no_parallel_location_fields` and
 * `test_observers_share_the_same_failure_authority`.
 *
 * Not ported: the `inspect.signature` half of
 * `test_failure_producers_have_no_location_parameters_or_attributes` (TypeScript has no keyword
 * parameters to inspect; an unknown option fails the type check).
 */

import { expect, test } from "bun:test";

import { NodeAssertFailed } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";

test("failure producers have no location attributes", () => {
  const binding = new BindingError("bad", { failureCode: "binding.evaluate" });
  const assertion = new NodeAssertFailed("bad");

  for (const error of [binding, assertion]) {
    expect("locator" in error).toBe(false);
    expect("range" in error).toBe(false);
  }
});
