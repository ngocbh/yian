/**
 * `ChildFlowNode`, the base CALL, MAP and LOOP share: the baked child, its signature's origins
 * and the refusal to run unbaked, named by the node's kind.
 */
import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { ChildFlowNode } from "../../src/nodes/child_flow.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";

const KINDS = [
  ["CALL", (child: unknown) => new CallNode("n", { flowId: "f", child })],
  ["MAP", (child: unknown) => new MapNode("n", { flowId: "f", child })],
  ["LOOP", (child: unknown) => new LoopNode("n", { flowId: "f", child, predicate: "true" })],
] as const;

test.each(KINDS)("%s refuses to run unbaked, naming its kind", (label, make) => {
  const node = make(null);
  expect(node).toBeInstanceOf(ChildFlowNode);
  expect(() => node.validateExecutionSetup()).toThrow(`${label} node 'n': child flow 'f' not baked`);
  expect(node.originChildren()).toEqual([]);
});

test.each(KINDS)("%s exposes its baked child and walks and rewrites its signature's origins", (_label, make) => {
  const baked = {};
  const node = make(baked);
  expect(node.originChildren()).toEqual([baked]);
  const origin = new OriginRef("input-x");
  node.childInputs = [new Parameter("x", { origin })];
  expect([...node.iterLocalOrigins()]).toContain(origin);
  const moved = new OriginRef("copied-x");
  node.rewriteLocalOrigins((ref) => (ref === origin ? moved : ref));
  expect(node.childInputs[0]!.origin).toBe(moved);
});
