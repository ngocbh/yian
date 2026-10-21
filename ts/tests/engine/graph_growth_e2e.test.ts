/**
 * The engine's graph growth end to end: a port of `tests/engine/test_graph_growth_e2e.py`.
 *
 * A MAP run grows the live graph (each element splices its namespaced child START..END, the MAP
 * fans into one list END, and every synthesized reference is a bare `${<id>.output}`), a
 * HUMAN_INPUT inside a called child passes its namespaced id to the host and completes in one run,
 * and the representative seeds load with a runnable entry.
 *
 * Ported: `test_live_graph_grows_under_a_map_run`, `test_start_end_splice_is_exercised`,
 * `test_synthesized_refs_are_bare_outputs`, `test_nested_human_input_reaches_terminal` and
 * `test_representative_seed_loads_with_structure` (all seven seeds; their `load-*` cases pin only
 * that they load). Integers are bigints (decision 4a); the CODE reference is the TS fixture's
 * (`./fixtures/compose_codefns.ts:make_report`).
 *
 * Changed: the seeds come from the conformance cases `load-core/seed_00_hello_agent`,
 * `load-map-call/seed_04_call`, `load-map-call/seed_05_call_map`, `load-agent/seed_14_agent_tools`,
 * `load-leaves/seed_17_effects_human_wait`, `load-core/seed_18_research_pipeline` and
 * `load-map-call/seed_25_nested_suspension`, loaded with the conformance directory as `baseDir`
 * and a search path holding seed 03 (`03-research-one.yaml` of the seed 04 case), written into a
 * temporary directory linked to the conformance fixtures. A wiring value is a `Binding`, so the
 * bare-output check reads each text `source`.
 *
 * Not ported:
 * - `test_scratch_is_eliminated_structurally` and `test_no_node_attr_or_method_reads_scratch`:
 *   guards against Python's removed node scratch, which TS never had.
 * - `test_no_remaining_engine_xfails`: a pytest collection check.
 *
 * Later (T4-8): `test_f0_goldens_hold` (it re-runs rows of `test_golden_run_results.py`, ported
 * with that file).
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { RunSucceeded } from "../../src/events.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { runFlow } from "../../src/run.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { caseText, linkConformanceFixtures, loadConformanceFlow } from "../compose/declarations.ts";
import { childResolver, engine } from "./run_state_fixtures.ts";

// --- CODE-child CALL/MAP fixtures --- //

const CHILD = `
id: child-one
name: child_one
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output:
      report: str
      n: int
    code: ./fixtures/compose_codefns.ts:make_report
output:
  report: \${emit.output.report}
  n: \${emit.output.n}
`;

const MAP = `
id: map-parent
name: map_parent
input:
  topics: !list(str)
uses:
  child-one: child-one
nodes:
  research_each:
    kind: map
    call: child-one
    over: \${input.topics}
    input:
      topic: \${item}
output: \${research_each.output}
`;

/** Drive the MAP through the engine and return it with its events, for live-graph inspection. */
async function runMap(topics: string[]) {
  const parent = await loadConformanceFlow(MAP, { childResolver: childResolver({ "child-one": CHILD }) });
  const eng = engine(parent.compiled, { inputs: { topics }, growBuilder: buildGrow });
  const events = await eng.run();
  return { eng, events };
}

/** The search path holding seed 03, as `03-research-one.yaml`. */
let seeds: string;

beforeAll(() => {
  seeds = mkdtempSync(join(tmpdir(), "yian-graph-growth-"));
  writeFileSync(
    join(seeds, "03-research-one.yaml"),
    caseText("load-map-call/seed_04_call", { file: "03-research-one.yaml" }).text,
  );
  linkConformanceFixtures(seeds);
});

afterAll(() => rmSync(seeds, { recursive: true, force: true }));

/** Load the flow of conformance case `id` with seed 03 on the search path. */
function loadSeed(id: string) {
  const { text, label } = caseText(id);
  return loadConformanceFlow(text, { searchPaths: [seeds], sourceLabel: label });
}

// --- the live graph grows under a MAP run --- //

test("live graph grows under a map run", async () => {
  const parent = await loadConformanceFlow(MAP, { childResolver: childResolver({ "child-one": CHILD }) });
  const eng = engine(parent.compiled, { inputs: { topics: ["ACME", "BETA"] }, growBuilder: buildGrow });
  const before = eng.state.flow.nodes.size;
  const events = await eng.run();
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  // The growth spliced a child clone per element into the live graph.
  expect(eng.state.flow.nodes.size).toBeGreaterThan(before);
});

test("start end splice is exercised", async () => {
  const { eng, events } = await runMap(["ACME", "BETA"]);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  const ids = [...eng.state.flow.nodes.keys()];
  // Each element splices its namespaced child START..END; MAP also synthesizes one
  // `research_each/__start__` fan-out start and one list END under the spawner namespace.
  const starts = ids.filter((id) => id.endsWith("/__start__")).sort();
  expect(starts).toEqual(["research_each#0/__start__", "research_each#1/__start__", "research_each/__start__"]);
  const mapStartId = "research_each/__start__"; // ns(spawner, START_ID)
  // The synthetic start fans out to each element START through an ordering edge.
  const fanout = new Set(
    eng.state.flow.edges.filter((edge) => edge.from === mapStartId && edge.ordering).map((edge) => edge.to),
  );
  expect(fanout).toEqual(new Set(["research_each#0/__start__", "research_each#1/__start__"]));
  const mapEndId = "research_each/__end__"; // ns(spawner, END_ID)
  expect(eng.state.flow.nodes.get(mapEndId)!.kind).toBe(NodeKind.END);
  const intoEnd = eng.state.flow.edges
    .filter((edge) => edge.to === mapEndId)
    .map((edge) => [edge.from, edge.inputGroup])
    .sort();
  expect(intoEnd).toEqual([
    ["research_each#0/__end__", "e0"],
    ["research_each#1/__end__", "e1"],
  ]);
  // The list END joins in over order (element index).
  expect((events.at(-1) as RunSucceeded).output).toEqual([
    { report: "report for ACME", n: 4n },
    { report: "report for BETA", n: 4n },
  ]);
});

// A trailing `.output` after a reference, which a synthesized reference never carries (a dotted
// field read like `${<id>/emit.output.report}` is allowed).
const SPURIOUS_OUTPUT = /\}\.output\b|outputs\.[^}]*\.output\b/;

test("synthesized refs are bare outputs", async () => {
  const { eng } = await runMap(["ACME", "BETA"]);
  for (const [nodeId, nodeWiring] of eng.state.flow.wiring) {
    if (!nodeId.includes("/") && !nodeId.includes("#")) continue; // a static node, not grown
    for (const binding of nodeWiring.values()) {
      if (typeof binding.source === "string") expect(binding.source).not.toMatch(SPURIOUS_OUTPUT);
    }
  }
});

// --- a nested blocking interaction reaches terminal --- //

test("nested human input reaches terminal", async () => {
  // Seed 25: a HUMAN_INPUT inside a called child invokes the host with its live namespaced id and
  // completes in one run.
  const loaded = await loadSeed("load-map-call/seed_25_nested_suspension");
  const requests: AskUserRequest[] = [];
  const result = await runFlow(
    loaded,
    { action: "order 10 units" },
    {
      tools: {
        ask_user: (request) => {
          requests.push(request);
          return "approve";
        },
      },
    },
  );

  expect(requests[0]!.nodeId).toBe("gate/approve");
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("approve"); // confirm_action({answer: "approve"})
});

// --- representative seeds load with structure --- //

const REPRESENTATIVE = [
  "load-core/seed_00_hello_agent",
  "load-map-call/seed_04_call",
  "load-map-call/seed_05_call_map",
  "load-agent/seed_14_agent_tools",
  "load-leaves/seed_17_effects_human_wait",
  "load-core/seed_18_research_pipeline",
  "load-map-call/seed_25_nested_suspension",
];

for (const seed of REPRESENTATIVE) {
  test(`representative seed loads with structure (${seed})`, async () => {
    // The LLM-terminating seeds are checked for load and structure only.
    const loaded = await loadSeed(seed);
    expect(loaded.compiled.nodes.size).toBeGreaterThan(0); // loaded, with a body
    expect(loaded.compiled.nodes.has(loaded.compiled.startId)).toBe(true); // a runnable entry
  });
}
