/**
 * `LoopNode`, the self-respawning `while:`/`until:`/`times:` driver: a port of
 * `tests/engine/test_loop_node.py`.
 *
 * Ported: `test_loop_is_a_spawner_kind`, `test_compiled_loop_origin_is_self`,
 * `test_respawn_preserves_assertion_carriers`,
 * `test_respawn_preserves_the_full_output_contract_carrier`,
 * `test_respawn_preserves_the_derived_base_memory_scope`,
 * `test_respawn_defaults_omitted_fields_from_the_carried_record`,
 * `test_partial_respawn_requires_the_carried_record`,
 * `test_respawn_keeps_a_null_omitted_value_as_its_default`, `test_total_respawn_keeps_params`,
 * `test_should_stop_is_the_max_iters_budget` and `test_should_stop_without_max_never_stops`.
 *
 * Changed: `NodeKind` is a const object of strings, so a member is its value and has no `.value`;
 * a trait is read on an instance, not the class. The child is a stub object, in place of
 * `_child_flow()`. Python ints in carried values are `bigint`.
 *
 * TS-only: the other traits (`kind`, `isLoop`, `growDepthDelta`) on an instance; `exceptionCode`
 * classifies `LoopMaxExceeded` as `loop.max_exceeded` and defers a plain `Error` to
 * `node.execute`; `skippedOutputTarget()` is the origin id on a node built with `originId` and on a
 * respawned driver; a baked LOOP's `run` rejects as not ported yet; `respawn` refuses an execution
 * scope of another class and one that doesn't end in the driver's position, and moves a scope that
 * does to index `k`.
 *
 * Covered elsewhere: `test_node_origin_id_defaults_none` (`base_node.test.ts`).
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

import { OriginRef } from "../../src/diagnostics.ts";
import { MemoryScope } from "../../src/memory/contracts.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import {
  Assertion,
  ContractOrigins,
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { LoopMaxExceeded, LoopNode } from "../../src/nodes/loop/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

/** A `while` driver `loop` over a stub body, guarded at 3 runs. */
function whileLoop(): LoopNode {
  return new LoopNode("loop", { flowId: "body", child: {}, predicateKind: "while", predicate: "${x}", maxIters: 3 });
}

/** A `times` loop `lp` whose body returns only `n` of its carried `n` and `person`. */
function partialLoop(): LoopNode {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "times",
    times: 1,
    maxIters: 1,
    returnedFields: ["n"],
  });
  node.params = [new Parameter("n"), new Parameter("person")];
  return node;
}

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

test("respawn preserves assertion carriers", () => {
  const node = whileLoop();
  const origin = new OriginRef("nodes/loop/asserts/0");
  node.preAsserts = [new Assertion("x > 0", "x must be positive", { origin })];

  const clone = node.respawn(1);

  expect(clone.preAsserts).toEqual(node.preAsserts);
  expect(clone.preAsserts).not.toBe(node.preAsserts);
  expect(clone.preAsserts[0]!.origin).toBe(origin);
});

test("respawn preserves the full output contract carrier", () => {
  const node = whileLoop();
  const origins = new ContractOrigins({
    default: new OriginRef("nodes/loop/output"),
    entries: [[["answer"], new OriginRef("nodes/loop/output/answer")]],
  });
  const contract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["answer"]),
    }),
    origins,
  );
  node.outputContract = contract;

  const clone = node.respawn(1);

  expect(clone.outputContract).toBe(contract);
  expect((clone.outputContract as ValueContract).origins).toBe(origins);
  expect((clone.outputContract as ValueContract).type).toBe(contract.type);
});

test("respawn preserves the derived base memory scope", () => {
  const node = whileLoop();
  node.memoryScope = new MemoryScope(["loop"], { memories: ["outer"] });

  const clone = node.respawn(1);

  expect(clone.memoryScope).toBe(node.memoryScope);
});

test("respawn defaults omitted fields from the carried record", () => {
  const node = partialLoop();
  const originParams = node.params;
  const originSnapshot = [...node.params!];

  const clone = node.respawn(1, { n: 1n, person: { name: "A" } });

  const person = clone.params!.find((param) => param.name === "person")!;
  const n = clone.params!.find((param) => param.name === "n")!;
  expect(person.hasDefault).toBe(true);
  expect(person.default).toEqual({ name: "A" });
  expect(person.required).toBe(false);
  expect(n.hasDefault).toBe(false);
  expect(clone.returnedFields).toEqual(["n"]);
  expect(node.params).toBe(originParams);
  expect(node.params).toEqual(originSnapshot);
  expect(node.params!.every((param) => param.hasDefault === false)).toBe(true);
});

test("partial respawn requires the carried record", () => {
  expectError(
    () => partialLoop().respawn(1),
    RangeError,
    "loop 'lp' returns a subset of its record; respawn needs the carried record",
  );
});

test("respawn keeps a null omitted value as its default", () => {
  const clone = partialLoop().respawn(1, { n: 1n, person: null });
  const person = clone.params!.find((param) => param.name === "person")!;
  expect(person.hasDefault).toBe(true);
  expect(person.default).toBeNull();
});

test("total respawn keeps params", () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "times", times: 1, maxIters: 1 });
  node.params = [new Parameter("n"), new Parameter("person")];

  const clone = node.respawn(1, { n: 1n, person: { name: "A" } });

  expect(clone.params).toBe(node.params);
  expect(clone.returnedFields).toBeNull();
});

test("a respawned driver is named for its origin and targets it", () => {
  const clone = whileLoop().respawn(2).respawn(3);
  expect(clone.id).toBe("loop~3");
  expect(clone.iteration).toBe(3);
  expect(clone.originId).toBe("loop");
  expect(clone.skippedOutputTarget()).toBe("loop");
});

test("respawn moves a scope that ends in the driver's position to index k", () => {
  const node = whileLoop();
  const outer = new ExecutionPosition("call", ["c"]);
  node.executionScope = new ExecutionScope(["c", "loop"], {
    flowPath: ["c"],
    occurrencePath: [outer, new ExecutionPosition("loop_driver", ["c", "loop"], 0)],
  });

  const scope = node.respawn(2).executionScope!;

  expect(scope).toBeInstanceOf(ExecutionScope);
  expect(scope.staticPath).toEqual(["c", "loop"]);
  expect(scope.flowPath).toEqual(["c"]);
  expect(scope.boundary).toBeNull();
  expect(scope.occurrencePath[0]).toBe(outer);
  expect(scope.occurrencePath[1]).toEqual(new ExecutionPosition("loop_driver", ["c", "loop"], 2));
});

test("respawn refuses an execution scope of another class", () => {
  class Subscope extends ExecutionScope {}
  const node = whileLoop();
  node.executionScope = new Subscope(["loop"], { occurrencePath: [new ExecutionPosition("loop_driver", ["loop"], 0)] });
  expectError(() => node.respawn(1), TypeError, "loop execution scope must be an ExecutionScope");
});

test("respawn refuses a scope that doesn't end in the driver's position", () => {
  const node = whileLoop();
  for (const occurrencePath of [
    [],
    [new ExecutionPosition("loop_driver", ["loop"], 1)],
    [new ExecutionPosition("loop_body", ["loop"], 0)],
    [new ExecutionPosition("loop_driver", ["other"], 0)],
  ]) {
    node.executionScope = new ExecutionScope(["loop"], { occurrencePath });
    expectError(() => node.respawn(1), RangeError, "loop scope must end in its current driver position");
  }
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

test("a loop's other traits are read on an instance", () => {
  const node = new LoopNode("lp", { flowId: "f" });
  expect(node.kind).toBe(NodeKind.LOOP);
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
