/**
 * `commitAs` on `Output` and on the `Node` base: a port of `tests/engine/test_commit_as.py`.
 *
 * `test_node_succeeded_commit_as_default_and_set` is in `events.test.ts`.
 */

import { expect, test } from "bun:test";

import { Node, NodeKind, Output } from "../../src/nodes/base.ts";

/** Minimal Node subclass to check the base `commitAs` default and assignment. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;
}

test("output commit as default and set", () => {
  expect(new Output(1n).commitAs).toBeNull();
  expect(new Output(1n, { commitAs: "s" }).commitAs).toBe("s");
});

test("node commit as default and assignment", () => {
  const node = new BareNode("n");
  expect(node.commitAs).toBeNull();
  node.commitAs = "s";
  expect(node.commitAs).toBe("s");
});
