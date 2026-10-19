/**
 * The flow boundary ids owned by the START and END node classes: a port of
 * `tests/engine/test_boundary_ids.py`. Ported: `test_pool_start_id_default_mirrors_startnode_id`.
 *
 * Changed: `test_start_end_ids_are_owned_by_the_node_classes` compares the ids with their literals;
 * the compile model's `START_ID`/`END_ID` re-export doesn't exist yet.
 *
 * Later (T3-10): `test_compiledflow_start_end_id_accessors`, and the `START_ID`/`END_ID` half of
 * `test_start_end_ids_are_owned_by_the_node_classes`.
 */

import { expect, test } from "bun:test";

import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

test("start end ids are owned by the node classes", () => {
  expect(StartNode.ID).toBe("__start__");
  expect(EndNode.ID).toBe("__end__");
});

test("pool start id default mirrors startnode id", () => {
  // The pool keeps a literal default (it sits below nodes and can't import the node class); this
  // pins it to StartNode.ID so the duplicated literal can never diverge.
  expect(new VariablePool().startId).toBe(StartNode.ID);
});
