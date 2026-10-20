/**
 * `evalNode` accepts only trait-declared spawner `Grow` results: a port of
 * `tests/engine/test_eval_node_grow.py`.
 *
 * A node whose `isSpawner` trait is true may return `Grow` for `FlowEngine` to interpret. A
 * non-spawner returning `Grow` becomes an `EvaluationFailure`.
 *
 * Ported: `test_spawner_grow_remains_a_raw_grow_result` and `test_leaf_grow_is_an_evaluation_failure`.
 */

import { expect, test } from "bun:test";

import { Grow, Node, NodeKind, type NodeResult } from "../../src/nodes/base.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

/** A non-spawner whose run returns a `Grow`, which must fail: only spawners grow. */
class FakeLeaf extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Grow({});
  }
}

/** A minimal spawner whose run returns a `Grow`, independent of any real spawner kind. */
class FakeSpawner extends FakeLeaf {
  override readonly isSpawner = true;
}

test("spawner grow remains a raw grow result", async () => {
  const { outcome } = await evalNode(new FakeSpawner("s"), null, new VariablePool());

  expect(outcome).toEqual(new Grow({}));
});

test("leaf grow is an evaluation failure", async () => {
  const { outcome } = await evalNode(new FakeLeaf("l"), null, new VariablePool());

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect((outcome as EvaluationFailure).failure.message).toContain("not a spawner");
});
