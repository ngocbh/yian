/**
 * Seed-first growth: a port of `tests/engine/test_seed_first_growth.py`.
 *
 * Ported: `test_loop_respawn_preserves_diagnostic_origin_separately_from_origin_id`.
 *
 * Later (T4-10): `test_call_map_and_loop_return_seed_only`.
 *
 * Later (T3-22): `test_compile_builder_derives_prefix_from_each_spawner_failure_trace`,
 * `test_source_free_spawners_do_not_invent_generated_context`,
 * `test_compile_builder_propagates_each_spawner_memory_occurrence`,
 * `test_nested_map_then_call_extends_context_only_when_inner_expands`,
 * `test_compile_builder_reconstructs_every_growth_kind`,
 * `test_compile_builder_marks_only_explicit_legal_seed_rejection`,
 * `test_compile_builder_keeps_an_unsupported_kind_outside_growth_rejection` and
 * `test_compile_builder_preserves_helper_fault_identity_after_seed_validation`.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";

test("loop respawn preserves diagnostic origin separately from origin id", () => {
  const loop = new LoopNode("loop", { flowId: "child", originId: "runtime-loop" });
  const authoredOrigin = new OriginRef("root/node/loop");
  const outerOrigin = new OriginRef("root/node/call");
  loop.origin = authoredOrigin;
  loop.originContext = [outerOrigin];

  const clone = loop.respawn(2);

  expect(clone.origin).toBe(authoredOrigin);
  expect(clone.originContext).toEqual([outerOrigin]);
  expect(clone.failureTrace()).toEqual([outerOrigin, authoredOrigin]);
  expect(clone.originId).toBe("runtime-loop");
});
