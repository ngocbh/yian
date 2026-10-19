/**
 * Building the effect leaves: a port of `tests/engine/test_build_effects.py`'s WAIT rows.
 *
 * Later (T3-15, the HUMAN_INPUT build): `test_build_human_input_node`.
 * Later (T3-17): `test_wait_until_output_ref_makes_data_edge` (data edges); its CODE producer
 * then takes a TS-style `code:` value in place of `m:f`.
 */

import { expect, test } from "bun:test";

import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import type { Node } from "../../src/nodes/base.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";

/** Build a leaf descriptor and return its node and wiring. */
function build(desc: NodeDescriptor): [Node, Map<string, unknown>] {
  const result = desc.build(new BuildContext({ registry: new Map() }));
  return [result.node, result.wiring];
}

test("build wait node", () => {
  const [node, wiring] = build(new WaitDescriptor({ id: "settle", until: "${input.settle_at}" }));

  expect(node).toBeInstanceOf(WaitNode);
  // The `until` source rides the wiring.
  expect(wiring).toEqual(new Map([["until", "${input.settle_at}"]]));
  expect(node.outputContract).toBeNull();
});
