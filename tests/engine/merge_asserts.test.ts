/**
 * Flow assertions lowered onto the synthesized boundaries: a port of
 * `tests/engine/test_merge_asserts.py`.
 *
 * Ported through `loadFlow` and `runFlow`: `test_map_bare_internal_assert_all_satisfy_succeeds`,
 * `test_braced_boundary_assert_fail_fast_and_pass` and
 * `test_bare_record_scoped_boundary_assert_fail_fast_and_pass`, each failing run checked for
 * `assertion.failed` at `__start__` where Python checks the status. Ported through `assembleText`
 * where Python calls `load_flow`: `test_system_ref_in_flow_assert_is_load_error`, pinning the
 * whole message where Python only expects a `LoadError`.
 *
 * Covered by `run-leaves` cases running the same flows on the same inputs:
 * `test_call_bare_internal_assert_succeeds`
 * (`engine_merge_asserts__a_child_flow_assert_passes_through_a_call`),
 * `test_map_bare_internal_assert_one_element_violates_fails_loudly`
 * (`engine_merge_asserts__a_child_flow_assert_fails_one_map_element`),
 * `test_loop_bare_internal_assert_succeeds`
 * (`engine_merge_asserts__a_child_flow_assert_passes_each_loop_turn`),
 * `test_optional_edge_taken_branch_completes`
 * (`engine_merge_asserts__a_flow_assert_on_the_taken_branch_waits_for_it`) and
 * `test_ordered_assert_on_skipped_producer_fails_loudly`
 * (`engine_merge_asserts__a_flow_assert_on_a_skipped_producer_fails`).
 *
 * TS-only: the harvested `load-leaves/engine_merge_asserts__system_ref_in_a_flow_assert`, pinned at
 * the end.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

// ---------- child bare assertions over internal producers ----------

test("a map succeeds when every child's internal assertion holds", async () => {
  const flow = `
id: c2map
name: c2_map
input:
  seeds: !list(int)

defs:
  child:
    input:
      seed: !int
    nodes:
      emit:
        kind: code
        input:
          seed: \${input.seed}
        output: {n: int}
        code: |
          return {"n": inputs["seed"] + 1}
    output: \${emit.output}
    asserts:
      - "emit.output.n >= 1"

nodes:
  fan:
    kind: map
    call: child
    over: \${input.seeds}
    input:
      seed: \${item}
output: \${fan.output}
`;
  const result = await runFlow(await loadFlow(flow), { seeds: [5, 6, 7] });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual([{ n: 6n }, { n: 7n }, { n: 8n }]);
});

// ---------- boundary assertion reference forms ----------

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

/** A flow echoing its input `age`, with the flow assert `assertion`. */
const boundaryAssertFlow = (assertion: string) => `
id: c3
name: c3
input:
  age: !int
${BOUNDARY_ASSERT_ECHO_BODY}
asserts:
  - "${assertion}"
`;

test.each([
  ["braced", "input.age >= 18"],
  ["bare record-scoped", "age >= 18"],
])("a %s boundary assert fails fast and passes", async (_form, assertion) => {
  const loaded = await loadFlow(boundaryAssertFlow(assertion));

  const young = await runFlow(loaded, { age: 10 });
  expect(young.status).toBe("failed");
  expect(young.failure?.code).toBe("assertion.failed");
  expect(young.failure?.nodeId).toBe("__start__");
  expect((await runFlow(loaded, { age: 20 })).status).toBe("succeeded");
});

// ---------- the system namespace ----------

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
