/**
 * `LoopNode`, the self-respawning `while:`/`until:`/`times:` driver: a port of
 * `tests/engine/test_loop_node.py`.
 *
 * Ported: `test_loop_is_a_spawner_kind`, `test_compiled_loop_origin_is_self`,
 * `test_should_stop_is_the_max_iters_budget` and `test_should_stop_without_max_never_stops`.
 *
 * Changed: `NodeKind` is a const object of strings, so a member is its value and has no `.value`;
 * a trait is read on an instance, not the class. The child is a stub object.
 *
 * TS-only: the traits on an instance; `exceptionCode` classifies `LoopMaxExceeded` as
 * `loop.max_exceeded` and defers a plain `Error` to `node.execute`; `skippedOutputTarget()` is the
 * origin id on a node built with `originId`; a baked LOOP's `run` rejects as not ported yet.
 *
 * Covered elsewhere: `test_node_origin_id_defaults_none` (`base_node.test.ts`).
 *
 * Later (T3-8, respawn): `test_respawn_preserves_assertion_carriers`,
 * `test_respawn_preserves_the_full_output_contract_carrier`,
 * `test_respawn_preserves_the_derived_base_memory_scope`,
 * `test_respawn_defaults_omitted_fields_from_the_carried_record`,
 * `test_partial_respawn_requires_the_carried_record`,
 * `test_respawn_keeps_a_null_omitted_value_as_its_default` and `test_total_respawn_keeps_params`.
 *
 * Later (T4-10): `test_run_grows_iteration_zero_with_next_driver_when_continue`,
 * `test_run_commits_seed_when_while_predicate_false`,
 * `test_clone_driver_self_prunes_previous_body_and_self`,
 * `test_respawned_driver_binds_merged_record`, `test_until_do_while_continues_on_iteration_zero`,
 * `test_times_stops_at_count`, `test_runaway_guard_raises_loop_max_exceeded` and
 * `test_unlimited_driver_far_past_any_cap_still_grows`.
 *
 * Later (T3-22 / T4-10): `test_build_grow_rebuilds_body_and_next_driver_on_origin`,
 * `test_build_grow_matches_live_clone_and_origin_replay_byte_for_byte`,
 * `test_later_loop_growth_does_not_duplicate_the_loop_origin` and
 * `test_build_grow_partial_loop_matches_live_and_replay_digest`.
 */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";
import { LoopMaxExceeded, LoopNode } from "../../src/nodes/loop/node.ts";

test("loop is a spawner kind", () => {
  const node = new LoopNode("lp", { flowId: "f", child: {} });
  expect(node.isSpawner).toBe(true);
  expect(NodeKind.LOOP).toBe("loop");
});

test("compiled loop origin is self", () => {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "${x}",
    maxIters: 10,
  });
  expect(node.originId).toBe("lp");
  expect(node.iteration).toBe(0);
});

test("should stop is the max iters budget", () => {
  const node = new LoopNode("l", { flowId: "f", child: {}, predicateKind: "while", predicate: "${x}", maxIters: 3 });
  expect(node.shouldStop(0)).toBe(false);
  expect(node.shouldStop(2)).toBe(false);
  expect(node.shouldStop(3)).toBe(true);
  expect(node.shouldStop(4)).toBe(true);
});

test("should stop without max never stops", () => {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "${x}",
    maxIters: null,
  });
  expect(node.shouldStop(0)).toBe(false);
  expect(node.shouldStop(10 ** 9)).toBe(false);
});

test("a loop's traits are read on an instance", () => {
  const node = new LoopNode("lp", { flowId: "f" });
  expect(node.kind).toBe(NodeKind.LOOP);
  expect(node.isSpawner).toBe(true);
  expect(node.isLoop).toBe(true);
  expect(node.growDepthDelta).toBe(0);
});

test("the runaway guard is classified loop.max_exceeded", () => {
  const node = new LoopNode("lp", { flowId: "f" });
  const guard = new LoopMaxExceeded("x");
  expect(guard).toBeInstanceOf(Error);
  expect(guard.name).toBe("LoopMaxExceeded");
  expect(node.exceptionCode(guard)).toBe("loop.max_exceeded");
  expect(node.exceptionCode(new Error("x"))).toBe("node.execute");
});

test("a driver's skipped output targets its origin", () => {
  const clone = new LoopNode("lp~2", { flowId: "f", iteration: 2, originId: "lp" });
  expect(clone.originId).toBe("lp");
  expect(clone.skippedOutputTarget()).toBe("lp");
  expect(new LoopNode("lp", { flowId: "f" }).skippedOutputTarget()).toBe("lp");
});

test("a baked loop's run rejects as not ported yet", async () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicate: "not exited", maxIters: 3 });
  await expect(node.run({ exited: false })).rejects.toThrow("node 'lp' (LoopNode) forward is not ported yet (T4-10)");
});
