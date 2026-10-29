/**
 * The flow boundary ids owned by the START and END node classes, and the pool's and
 * `CompiledFlow`'s accessors for them.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

test("start end ids are owned by the node classes", () => {
  // The boundary node ids are owned canonically by the node classes; compile/model re-exports
  // them as START_ID/END_ID.
  expect(StartNode.ID).toBe("__start__");
  expect(START_ID).toBe(StartNode.ID);
  expect(EndNode.ID).toBe("__end__");
  expect(END_ID).toBe(EndNode.ID);
});

test("pool start id default mirrors startnode id", () => {
  // The pool keeps a literal default (it sits below nodes and can't import the node class); this
  // pins it to StartNode.ID so the duplicated literal can never diverge.
  expect(new VariablePool().startId).toBe(StartNode.ID);
});

test("compiledflow start end id accessors", () => {
  const n = new CodeNode("a", { code: "return 1" });
  const flow = new CompiledFlow(new Map([["a", n]]), [new Edge("s", START_ID, "a"), new Edge("t", "a", END_ID)]);
  expect(flow.startId).toBe(START_ID);
  expect(flow.endId).toBe(END_ID);
});
