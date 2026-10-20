/**
 * Flow assertions lowered onto the synthesized boundaries: a port of
 * `tests/engine/test_merge_asserts.py::test_system_ref_in_flow_assert_is_load_error`, through
 * `assembleText` where Python calls `load_flow`, pinning the whole message where Python only
 * expects a `LoadError`.
 *
 * TS-only: the harvested `load-leaves/engine_merge_asserts__system_ref_in_a_flow_assert`, pinned at
 * the end.
 *
 * Later (T4): `test_call_bare_internal_assert_succeeds`, `test_map_bare_internal_assert_all_satisfy_succeeds`,
 * `test_map_bare_internal_assert_one_element_violates_fails_loudly`,
 * `test_loop_bare_internal_assert_succeeds`, `test_braced_boundary_assert_fail_fast_and_pass`,
 * `test_bare_record_scoped_boundary_assert_fail_fast_and_pass`,
 * `test_optional_edge_taken_branch_completes` and `test_ordered_assert_on_skipped_producer_fails_loudly`
 * (runs).
 */

import { expect, test } from "bun:test";

import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

const BOUNDARY_ASSERT_ECHO_BODY = `
nodes:
  echo:
    kind: code
    input:
      age: \${input.age}
    output: int
    code: |
      return inputs["age"]
output: \${echo.output}
`;

test("a system ref in a flow assert is a load error", () => {
  // A top-level flow assert rejects the ambient `system` namespace at load.
  const flow = `
id: sysf
name: sysf
input:
  age: !int
${BOUNDARY_ASSERT_ECHO_BODY}
asserts:
  - "system.run_id != null"
`;
  expect(loadError(() => assembleText(flow)).message).toBe(
    "assert 'system.run_id != null': the `system` namespace is not allowed in asserts (it is ambient — " +
      "there is no producer to draw an edge from; move the check into a body node)",
  );
});

// ---------- harvested conformance cases ----------

test("assembled: load-leaves/engine_merge_asserts__system_ref_in_a_flow_assert", () => {
  runCase({ id: "load-leaves/engine_merge_asserts__system_ref_in_a_flow_assert", call: assembled });
});
