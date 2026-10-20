/**
 * Data-edge inference and its input groups (`compose/edges/data.ts`): a port of
 * `tests/engine/test_infer.py`.
 *
 * A flow-level pass: for every sink binding in the flow wiring, each `${<id>.output[.…]}`
 * reference becomes a data `Edge(<id>, <consumer>, {inputGroup: <sink key>})`. A `case` node is
 * skipped (its build emits its own `__rN`/`__on` edges). A coalesce `${a | b}` sink yields two
 * edges sharing one input group. An `${input.X}` read mints a `START_ID -> reader` edge;
 * `${system.X}`/`${item}` mint none; boundary assembly, not this pass, makes the `END_ID` edges.
 *
 * Changed: the seed rows read seeds 01, 02, 06 and 10 through the conformance cases
 * `load-agent/seed_01_structured_agent`, `load-case/seed_02_case`, `load-case/seed_06_case_on` and
 * `load-leaves/seed_10_asserts_arithmetic`, where Python reads the seed files; leaf wiring comes
 * from `build` with the case's typedefs. The CODE reference is TS's (`./m.ts:f`).
 *
 * TS-only: a WAIT with no `until` wiring throws a `RangeError` (Python's `KeyError`).
 */

import { expect, test } from "bun:test";

import { type Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { flowNodesCursor, parseFile } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import type { TypeRegistry } from "../../src/typesys/types.ts";
import { caseDocument, caseRegistry } from "../compose/declarations.ts";

/** Return the flow-owned wiring of a built leaf descriptor. */
function wiring(desc: NodeDescriptor, registry: TypeRegistry = new Map()): Map<string, unknown> {
  return desc.build(new BuildContext({ registry })).wiring;
}

/** Return the descriptors of `document`'s nodes and the wiring of its AGENT and CODE nodes. */
function leaves(document: SourceDocument): [Map<string, NodeDescriptor>, Map<string, Map<string, unknown>>] {
  const descriptors = parseNodes(flowNodesCursor(document, parseFile(document).nodes));
  const registry = caseRegistry(document);
  const flowWiring = new Map<string, Map<string, unknown>>();
  for (const [id, desc] of descriptors) {
    if (desc instanceof AgentDescriptor || desc instanceof CodeDescriptor) flowWiring.set(id, wiring(desc, registry));
  }
  return [descriptors, flowWiring];
}

/** Return the data edges of conformance case `id`'s seed flow. */
function seedEdges(id: string): Edge[] {
  return inferDataEdges(...leaves(caseDocument(id).document));
}

/** Return each edge as `[from, to, inputGroup]`. */
function edgeTuples(edges: readonly Edge[]): [string, string, string | null][] {
  return edges.map((edge) => [edge.from, edge.to, edge.inputGroup]);
}

const SEED01 = "load-agent/seed_01_structured_agent";

// ---------- seed 01: plain data edges, input group = sink field name ----------

test("seed 01 data edges carry their input group", () => {
  const edges = seedEdges(SEED01);
  const tuples = edgeTuples(edges);

  // verdict reads score.rating + score.rationale (the input group is the sink field).
  expect(tuples).toContainEqual(["score", "verdict", "rating"]);
  expect(tuples).toContainEqual(["score", "verdict", "rationale"]);
  // score reads only ${input.topic}: its only incoming edge is the START_ID input-producer edge.
  expect(edges.some((edge) => edge.to === "score" && edge.from !== START_ID)).toBe(false);
  expect(tuples).toContainEqual(["__start__", "score", "topic"]);
});

test("inference mints START input edges, never END edges", () => {
  const edges = seedEdges(SEED01);

  expect(edges.some((edge) => edge.from === START_ID)).toBe(true);
  for (const edge of edges) {
    expect(edge.to).not.toBe(END_ID);
    expect(edge.from).not.toBe(END_ID);
  }
});

test("edge ids are synthesized", () => {
  const edges = seedEdges(SEED01);

  // Ids are unique and follow the `from->to#i` convention.
  const ids = edges.map((edge) => edge.id);
  expect(new Set(ids).size).toBe(ids.length);
  for (const edge of edges) expect(edge.id).toStartWith(`${edge.from}->${edge.to}#`);
});

// ---------- coalesce sink: two edges sharing one input group ----------

test("a coalesce sink's edges share one input group", () => {
  // A node input bound to ${pro.output | con.output} gives two edges, same group.
  const descriptors = parseNodes(
    SourceCursor.virtual({
      pro: { kind: "agent", outputs: "str", prompt: "x" },
      con: { kind: "agent", outputs: "str", prompt: "x" },
      merge: { kind: "code", inputs: { claim: "${pro.output | con.output}" }, outputs: "str", code: "./m.ts:f" },
    }),
  );
  const flowWiring = new Map([...descriptors].map(([id, desc]) => [id, wiring(desc)]));

  const coalesce = inferDataEdges(descriptors, flowWiring).filter((edge) => edge.to === "merge");

  expect(coalesce.length).toBe(2);
  expect(new Set(coalesce.map((edge) => edge.from))).toEqual(new Set(["pro", "con"]));
  // Both edges share one input group (the sink input name).
  expect(new Set(coalesce.map((edge) => edge.inputGroup))).toEqual(new Set(["claim"]));
});

// ---------- mapped call (over:) produces a data edge ----------

test("a map's over ref produces a data edge", () => {
  const descriptors = parseNodes(
    SourceCursor.virtual({
      bundle: { kind: "code", outputs: "list(str)", code: "./m.ts:f" },
      each: { kind: "map", call: "child", over: "${bundle.output}", inputs: { topic: "${item}" } },
    }),
  );
  // A MAP build needs a resolver, so its wiring is given directly: the `over` source rides the
  // reserved "over" key, over-then-inputs order.
  const flowWiring = new Map([
    ["bundle", wiring(descriptors.get("bundle")!)],
    [
      "each",
      new Map([
        ["over", "${bundle.output}"],
        ["topic", "${item}"],
      ]),
    ],
  ]);

  const edges = inferDataEdges(descriptors, flowWiring);

  expect(edgeTuples(edges)).toContainEqual(["bundle", "each", "over"]);
  // ${item} is a body-local scope, not a pool ref: no edge.
  expect(edges.some((edge) => edge.from === "item")).toBe(false);
});

// ---------- a case node's data edges are not inferred here ----------
// The case owns its lowering in `CaseDescriptor.build`, which emits the reconciled `__rN`/`__on`
// data edges directly, so `inferDataEdges` emits no edge to or from a case node (the case
// data-edge coverage lives in `case.test.ts`).

test("a case on: ref gives no inferred edge", () => {
  // seed 06: route is a case with `on: ${classify.output}`; the classify->route edge is the case
  // build's.
  expect(seedEdges("load-case/seed_06_case_on").some((edge) => edge.to === "route")).toBe(false);
});

test("a case searched when: ref gives no inferred edge", () => {
  // seed 02: gate is a case.
  expect(seedEdges("load-case/seed_02_case").some((edge) => edge.to === "gate")).toBe(false);
});

test("a case multi-ref when: gives no inferred edge", () => {
  // seed 10: size is a case.
  expect(seedEdges("load-leaves/seed_10_asserts_arithmetic").some((edge) => edge.to === "size")).toBe(false);
});

test.each(["score.output >= 0.5", "${score.output >= 0.5}"])(
  "a searched when gives no inferred edge in any spelling: %s",
  (when) => {
    const text =
      "id: f\nname: f\ninput:\n  x: !float\nnodes:\n" +
      "  score:\n    kind: code\n    input:\n      x: ${input.x}\n" +
      "    output: float\n    code: ./m.ts:fn\n" +
      "  gate:\n    kind: case\n    cases:\n" +
      `      - when: "${when}"\n        then: hi\n    else: lo\n` +
      '  hi:\n    kind: agent\n    prompt: "hi"\n    output: str\n' +
      '  lo:\n    kind: agent\n    prompt: "lo"\n    output: str\n' +
      "output: ${hi.output | lo.output}\n";

    const edges = inferDataEdges(...leaves(SourceDocument.parse(text)));

    expect(edges.some((edge) => edge.to === "gate")).toBe(false);
  },
);

test("no END edges for a case seed", () => {
  // Input readers get START_ID input-producer edges, but no END_ID edge is minted here.
  for (const edge of seedEdges("load-case/seed_02_case")) {
    expect(edge.to).not.toBe(END_ID);
    expect(edge.from).not.toBe(END_ID);
  }
});

test("a WAIT with no until wiring is a compiler bug", () => {
  // TS-only: Python raises a bare `KeyError` on the missing wiring entry.
  const descriptors = parseNodes(
    SourceCursor.virtual({ w: { kind: "wait", until: "${input.at}" } }),
  ) as Map<string, NodeDescriptor>;

  expect(() => inferDataEdges(descriptors, new Map())).toThrow(new RangeError("WAIT node 'w' has no until wiring"));
});
