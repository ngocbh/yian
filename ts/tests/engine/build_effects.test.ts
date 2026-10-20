/**
 * Building the effect leaves: a port of `tests/engine/test_build_effects.py`'s HUMAN_INPUT and
 * WAIT rows, and `test_wait_until_output_ref_makes_data_edge` (data edges).
 *
 * Changed: the CODE producer takes a TS-style `code:` value (`./m.ts:f`) in place of `m:f`.
 */

import { expect, test } from "bun:test";

import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import type { Node } from "../../src/nodes/base.ts";
import { HumanInputNode } from "../../src/nodes/human_input/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";

/** Build a leaf descriptor and return its node and wiring. */
function build(desc: NodeDescriptor): [Node, Map<string, unknown>] {
  const result = desc.build(new BuildContext({ registry: new Map() }));
  return [result.node, result.wiring];
}

test("build human_input node", () => {
  const desc = new HumanInputDescriptor({
    id: "approve",
    prompt: "ok? ${a}",
    inputs: { a: "${p.output}" },
    outputs: "str",
  });
  const [node, wiring] = build(desc);

  expect(node).toBeInstanceOf(HumanInputNode);
  // `str` is enforced at the write boundary.
  expect(node.outputContract).not.toBeNull();
  // The node-side signature, then the flow-owned source.
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["a"]);
  expect(wiring).toEqual(new Map([["a", "${p.output}"]]));
});

test("build wait node", () => {
  const [node, wiring] = build(new WaitDescriptor({ id: "settle", until: "${input.settle_at}" }));

  expect(node).toBeInstanceOf(WaitNode);
  // The `until` source rides the wiring.
  expect(wiring).toEqual(new Map([["until", "${input.settle_at}"]]));
  expect(node.outputContract).toBeNull();
});

test("a wait until output ref makes a data edge", () => {
  // A wait whose `until` reads ${X.output} must order after X.
  const descs = new Map<string, NodeDescriptor>([
    ["clock", new CodeDescriptor({ id: "clock", code: "./m.ts:f", outputs: "datetime" })],
    ["settle", new WaitDescriptor({ id: "settle", until: "${clock.output}" })],
  ]);
  const flowWiring = new Map([...descs].map(([id, desc]) => [id, build(desc)[1]]));

  const edges = inferDataEdges(descs, flowWiring);

  expect(edges.some((edge) => edge.from === "clock" && edge.to === "settle")).toBe(true);
});
