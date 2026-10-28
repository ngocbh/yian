/**
 * The MAP child-run seam, end to end over CODE-only children.
 *
 * A `map` node runs its baked child once per element of `over:`, binding `${item}` per element,
 * and collects each child's single value into `list(U)`. `parallel` overlaps the element runs
 * but keeps their order.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Grow, NodeKind } from "../../src/nodes/base.ts";
import type { CodeNode } from "../../src/nodes/code/node.ts";
import { mapCallsite, ns } from "../../src/nodes/ids.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { runFlow } from "../../src/run.ts";
import type { JsonValue } from "../../src/typesys/json.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { childFlow, childResolver, engine } from "../helpers/run_state_fixtures.ts";

// topic -> the topic (echo), so MAP results == the over list.
const ECHO_CHILD = `
id: echo-one
name: echo_one
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./fixtures/compose_codefns.ts:echo
output: \${emit.output}
`;

const BOOM_CHILD = `
id: boom-child
name: boom_child
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./fixtures/compose_codefns.ts:boom
output: \${emit.output}
`;

function mapFlow(childId: string, parallel: boolean): string {
  const par = parallel ? "\n    parallel: true" : "";
  return `
id: map-parent
name: map_parent
input:
  topics: !list(str)
uses:
  ${childId}: ${childId}
nodes:
  each:
    kind: map
    call: ${childId}
    over: \${input.topics}${par}
    input:
      topic: \${item}
output: \${each.output}
`;
}

test("map runs child per element collecting results", async () => {
  const flow = await loadConformanceFlow(mapFlow("echo-one", false), {
    childResolver: childResolver({ "echo-one": ECHO_CHILD }),
  });
  const result = await runFlow(flow, { topics: ["ACME", "BETA", "GOOG"] });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(["ACME", "BETA", "GOOG"]);
});

const OPAQUE_OUTPUT_MAP = `
id: opaque-output-map
name: opaque_output_map
input:
  values: !list(int)
defs:
  increment:
    input:
      value: !int
    nodes: {}
    output: \${input.value + 1}
nodes:
  each:
    kind: map
    call: increment
    over: \${input.values}
    input:
      value: \${item}
output: \${each.output}
`;

test("map distinguishes declared opaque output from no output", async () => {
  // A declared but unproven child output maps without a false list contract.
  const loaded = await loadConformanceFlow(OPAQUE_OUTPUT_MAP);

  expect(loaded.compiled.nodes.get("each")!.outputContract).toBeNull();

  const result = await runFlow(loaded, { values: [1n, 4n] });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([2n, 5n]);
});

test("map empty over yields empty list", async () => {
  const flow = await loadConformanceFlow(mapFlow("echo-one", false), {
    childResolver: childResolver({ "echo-one": ECHO_CHILD }),
  });
  const result = await runFlow(flow, { topics: [] });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([]);
});

test("map parallel preserves element order", async () => {
  const flow = await loadConformanceFlow(mapFlow("echo-one", true), {
    childResolver: childResolver({ "echo-one": ECHO_CHILD }),
  });
  const result = await runFlow(flow, { topics: ["A", "B", "C", "D"] });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(["A", "B", "C", "D"]); // order preserved under parallel
});

test("map child failure fails the run", async () => {
  const flow = await loadConformanceFlow(mapFlow("boom-child", false), {
    childResolver: childResolver({ "boom-child": BOOM_CHILD }),
  });
  const result = await runFlow(flow, { topics: ["ACME"] });
  expect(result.status).not.toBe("succeeded");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.message).toContain("boom");
  const mapped = flow.compiled.nodes.get("each") as MapNode;
  expect(result.failure!.nodeId).toBe("each#0/emit");
  const emit = (mapped.child as CompiledFlow).nodes.get("emit") as CodeNode;
  expect(result.failure!.trace).toEqual([mapped.origin!, emit.codeOrigin!]);
});

// `over:` with a `|` coalesce resolves at run through the same template layer it validated
// through at load.
const MAP_OVER_COALESCE = `
id: map-coalesce-over
name: map_coalesce_over
input:
  topics: !Optional(list(str))
  alt: !list(str)
uses:
  echo-one: echo-one
nodes:
  each:
    kind: map
    call: echo-one
    over: \${input.topics | input.alt}
    input:
      topic: \${item}
output: \${each.output}
`;

test("map over with coalesce resolves at run", async () => {
  // A run grows its compiled flow in place, so each independent run loads a fresh flow.
  // Omitted topics -> coalesce falls through to alt.
  const flow = await loadConformanceFlow(MAP_OVER_COALESCE, {
    childResolver: childResolver({ "echo-one": ECHO_CHILD }),
  });
  const result = await runFlow(flow, { alt: ["X", "Y"] });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(["X", "Y"]);
  // Present topics -> the first branch wins.
  const flow2 = await loadConformanceFlow(MAP_OVER_COALESCE, {
    childResolver: childResolver({ "echo-one": ECHO_CHILD }),
  });
  const result2 = await runFlow(flow2, { topics: ["A"], alt: ["Z"] });
  expect(result2.status).toBe("succeeded");
  expect(result2.output).toEqual(["A"]);
});

// --- MapNode.run -> seed; the compile builder reconstructs the ordered END-list fan-in --- //
test("map run returns grow", async () => {
  const n = new MapNode("m", { flowId: "c", child: childFlow(), childInputs: [] });
  const out = await n.run({ over: ["A", "B"] }, { bindItem: (element) => ({ x: element }) });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual([{ x: "A" }, { x: "B" }]);
  const subgraph = buildGrow(n, (out as Grow).seed as JsonValue);
  expect(subgraph.startId).toBe(ns("m", START_ID));
  for (let i = 0; i < 2; i++) {
    expect(subgraph.nodes.has(ns(mapCallsite("m", i), childFlow().startId))).toBe(true);
  }
});

test("map empty over returns grow with lone list end", async () => {
  const n = new MapNode("m", { flowId: "c", child: childFlow(), childInputs: [] });
  const out = await n.run({ over: [] }, { bindItem: () => ({}) });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual([]);
  const subgraph = buildGrow(n, (out as Grow).seed as JsonValue);
  expect(new Set(subgraph.nodes.keys())).toEqual(new Set([ns("m", START_ID), ns("m", END_ID)]));
  expect(subgraph.startId).toBe(ns("m", START_ID));
});

for (const numWorkers of [0, 4]) {
  test(`map parallel via num workers preserves order (numWorkers ${numWorkers})`, async () => {
    const flow = await loadConformanceFlow(mapFlow("echo-one", false), {
      childResolver: childResolver({ "echo-one": ECHO_CHILD }),
    });
    const eng = engine(flow.compiled, { inputs: { topics: ["A", "B", "C", "D"] }, numWorkers, growBuilder: buildGrow });
    // The over-order join (the END-list invariant).
    expect(((await eng.run()).at(-1) as RunSucceeded).output).toEqual(["A", "B", "C", "D"]);
  });
}

for (const numWorkers of [0, 4]) {
  test(`map n zero synthesizes end list emitting empty (numWorkers ${numWorkers})`, async () => {
    const flow = await loadConformanceFlow(mapFlow("echo-one", false), {
      childResolver: childResolver({ "echo-one": ECHO_CHILD }),
    });
    const eng = engine(flow.compiled, { inputs: { topics: [] }, numWorkers, growBuilder: buildGrow });
    // N=0 -> the synthetic map START -> the list END -> [].
    expect(((await eng.run()).at(-1) as RunSucceeded).output).toEqual([]);
  });
}

for (const numWorkers of [0, 4]) {
  test(`map synthesizes one end list node (numWorkers ${numWorkers})`, async () => {
    // The MAP fan-in is one list END: N=2 inputs e0/e1 wired to each element's child END,
    // committed as the spawner.
    const flow = await loadConformanceFlow(mapFlow("echo-one", false), {
      childResolver: childResolver({ "echo-one": ECHO_CHILD }),
    });
    const eng = engine(flow.compiled, { inputs: { topics: ["A", "B"] }, numWorkers, growBuilder: buildGrow });
    expect(((await eng.run()).at(-1) as RunSucceeded).output).toEqual(["A", "B"]);
    const mapEndId = "each/__end__"; // ns(spawner, END_ID)
    expect(eng.state.flow.nodes.get(mapEndId)!.kind).toBe(NodeKind.END);
    const intoEnd = eng.state.flow.edges
      .filter((edge) => edge.to === mapEndId)
      .map((edge) => [edge.from, edge.inputGroup])
      .sort();
    expect(intoEnd).toEqual([
      ["each#0/__end__", "e0"],
      ["each#1/__end__", "e1"],
    ]);
    expect(eng.state.flow.nodes.get(mapEndId)!.commitAs).toBe("each");
  });
}
