/**
 * The pure MAP builder `mapSubgraph`, which pins its output to the MAP growth's clone and fan-in
 * shape.
 *
 * `mapSubgraph(child, spawnerId, records)` clones the child once per element (namespaced under
 * `mapCallsite(spawnerId, i)`), synthesizes a `<spawner>/__start__` StartNode that fans out to every
 * element START through ordering edges, plus the list-END fan-in (an `EndNode.list` carrying
 * `commitAs = spawnerId`, fed one `e<i>` group per element), and returns a `Flow`. With no records
 * the only body node is the list-END, and the synthetic START wires straight to it, so it
 * schedules and emits `[]`.
 */

import { expect, test } from "bun:test";

import { mapSubgraph } from "../../src/compile/expand.ts";
import { END_ID, Flow, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { Binding, ChildFlowBoundary, SeedConnection } from "../../src/nodes/contracts.ts";
import { mapCallsite, ns } from "../../src/nodes/ids.ts";
import { childFlow } from "../helpers/run_state_fixtures.ts";

test("map subgraph matches grow map type", () => {
  const child = childFlow();
  const records = [{ x: 1n }, { x: 2n }];
  const sg = mapSubgraph(child, "m0", records);

  expect(sg).toBeInstanceOf(Flow);
  const mapStartId = ns("m0", START_ID);
  const mapEndId = ns("m0", END_ID);

  // Two child clones (one per element callsite), the synthetic start and one synthesized list-END.
  for (const i of [0, 1]) {
    const callsite = mapCallsite("m0", i);
    for (const nid of child.nodes.keys()) expect(sg.nodes.has(ns(callsite, nid))).toBe(true);
  }
  expect(sg.nodes.has(mapStartId)).toBe(true);
  expect(sg.nodes.has(mapEndId)).toBe(true);

  // startId is the synthetic fan-out start; endId is the list collector.
  expect(sg.startId).toBe(mapStartId);
  expect(sg.endId).toBe(mapEndId);

  // The list-END commits its list Output under the spawner id.
  expect(sg.nodes.get(mapEndId)!.commitAs).toBe("m0");

  // The fan-in wiring has one e<i> group per element.
  const endWiring = sg.wiring.get(mapEndId)!;
  expect(new Set(endWiring.keys())).toEqual(new Set(["e0", "e1"]));
  for (const i of [0, 1]) {
    const outId = ns(mapCallsite("m0", i), child.endId);
    expect(endWiring.get(`e${i}`)).toEqual(new Binding(`\${${outId}.output}`));
  }

  // The synthetic start fans out to every element START through an ordering edge (its sole gate).
  for (const i of [0, 1]) {
    const elemStart = ns(mapCallsite("m0", i), child.startId);
    const matched = sg.edges.filter((e) => e.from === mapStartId && e.to === elemStart);
    expect(matched).toHaveLength(1);
    const edge = matched[0]!;
    // Ordering, not data: no phantom input group.
    expect(edge.ordering).toBe(true);
    expect(edge.inputGroup).toBeNull();
    // As `depends_on`, mirroring the top-level start -> root gate.
    expect(edge.optional).toBe(false);
  }

  // With records, the synthetic start doesn't wire straight to the list-END (it fans in over the
  // clones).
  expect(sg.edges.some((e) => e.from === mapStartId && e.to === mapEndId)).toBe(false);
  expect(sg.inputConnections).toEqual(
    [0, 1].map((i) => new SeedConnection(ns(mapCallsite("m0", i), child.startId), "x", [i, "x"])),
  );
  expect(sg.childBoundaries).toEqual(
    [0, 1].map(
      (i) => new ChildFlowBoundary(ns(mapCallsite("m0", i), child.startId), ns(mapCallsite("m0", i), child.endId)),
    ),
  );
});

test("map subgraph n ge 2 start fans out to all element starts", () => {
  // With N >= 2 the synthetic start has an ordering edge to all N element starts (the fan-out
  // that replaced the old N-root list).
  const child = childFlow();
  const records = [0n, 1n, 2n, 3n].map((x) => ({ x }));
  const sg = mapSubgraph(child, "m0", records);

  const mapStartId = ns("m0", START_ID);
  const fanout = new Set(sg.edges.filter((e) => e.from === mapStartId && e.ordering).map((e) => e.to));
  const expected = new Set([0, 1, 2, 3].map((i) => ns(mapCallsite("m0", i), child.startId)));
  expect(fanout).toEqual(expected);
});

test("map subgraph shares context across distinct clones and boundaries", () => {
  const child = childFlow();
  const context = [new OriginRef("root/nodes/map")];

  const sg = mapSubgraph(child, "m0", [{ x: 1n }, { x: 2n }], { originContext: context });

  const firstBody = ns(mapCallsite("m0", 0), "head");
  const secondBody = ns(mapCallsite("m0", 1), "head");
  expect(firstBody).not.toBe(secondBody);
  expect(sg.nodes.get(firstBody)!.originContext).toEqual(context);
  expect(sg.nodes.get(secondBody)!.originContext).toEqual(context);
  expect(sg.nodes.get(ns("m0", START_ID))!.originContext).toEqual(context);
  expect(sg.nodes.get(ns("m0", END_ID))!.originContext).toEqual(context);
});

test("map subgraph n0 wires start to collector", () => {
  const child = childFlow();
  const context = [new OriginRef("root/nodes/map")];
  const sg = mapSubgraph(child, "m0", [], { originContext: context });

  const mapStartId = ns("m0", START_ID);
  const mapEndId = ns("m0", END_ID);
  // The only nodes are the synthetic start and the list-END; the start wires straight to the END
  // (an ordering edge), so the collector schedules off the fan-out and emits [].
  expect(new Set(sg.nodes.keys())).toEqual(new Set([mapStartId, mapEndId]));
  expect(sg.startId).toBe(mapStartId);
  expect(sg.endId).toBe(mapEndId);
  const edgesToEnd = sg.edges.filter((e) => e.to === mapEndId);
  expect(edgesToEnd).toHaveLength(1);
  expect(edgesToEnd[0]!.from).toBe(mapStartId);
  expect(edgesToEnd[0]!.ordering).toBe(true);
  expect(sg.nodes.get(mapEndId)!.commitAs).toBe("m0");
  expect(sg.nodes.get(mapStartId)!.originContext).toEqual(context);
  expect(sg.nodes.get(mapEndId)!.originContext).toEqual(context);
  expect(sg.inputConnections).toEqual([]);
  expect(sg.childBoundaries).toEqual([]);
});

