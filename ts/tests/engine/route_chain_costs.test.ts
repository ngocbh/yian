/**
 * Backward context reads: `test_downstream_context_reads_are_unaliased` of
 * `tests/engine/test_route_chain_costs.py`.
 *
 * Changed: reads return the frozen copy the constructor made, not a fresh copy per read, so the
 * test checks that a shared subtree is copied once, that the caller's objects aren't aliased, and
 * that the stored value is frozen.
 *
 * Later (T4-x / T4-8): the other 8 run flows or encode route payloads:
 * `test_backward_results_match_the_recorded_results`, `test_route_errors_on_multi_error_trees`,
 * `test_route_encode_and_decode_check_each_node_once`,
 * `test_run_and_resume_route_checks_grow_linearly`, `test_backward_lookups_grow_near_linearly`,
 * `test_backward_context_copies_grow_near_linearly`, `test_backward_freeze_work_grows_near_linearly`
 * and `test_backward_peak_memory_grows_near_linearly`.
 */

import { expect, test } from "bun:test";

import { BackwardContext } from "../../src/nodes/base.ts";

test("downstream context reads are unaliased", () => {
  const shared = { note: ["kept"] };
  const items = [{ first: shared, again: shared }, { second: shared }];
  const context = new BackwardContext({}, { downstreamContext: items });

  const read = context.downstreamContext;
  expect(read).toEqual(items);
  expect(read[0]!.first).toBe(read[0]!.again!);
  expect(read[0]!.first).toBe(read[1]!.second!);
  expect(read[0]!.first).not.toBe(shared);
  shared.note.push("changed");
  expect(read[1]!.second).toEqual({ note: ["kept"] });
  expect(Object.isFrozen(read)).toBe(true);
  expect(Object.isFrozen(read[0])).toBe(true);
  expect(() => {
    (read[0]!.first as { note: string[] }).note.push("changed");
  }).toThrow(TypeError);
  expect(context.downstreamContext).toBe(read);
});
