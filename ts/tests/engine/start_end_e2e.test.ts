/**
 * Every flow is `START_ID -> body -> END_ID`: a port of `tests/engine/test_start_end_e2e.py`.
 *
 * Each representative seed compiles to one START and one END node whose ids are `__start__` and
 * `__end__`, every compiled kind is a current `NodeKind`, and every edge touching a boundary id
 * leaves a START or enters an END.
 *
 * Ported: `test_exactly_one_start_and_one_end_per_flow`,
 * `test_no_retired_node_kinds_in_compiled_ir` and `test_representative_seed_loads_with_structure`,
 * each over the seven seeds. Changed: the seeds come from the conformance cases
 * `load-core/seed_00_hello_agent`, `load-agent/seed_03_research_one`, `load-map-call/seed_04_call`,
 * `load-map-call/seed_05_call_map`, `load-agent/seed_14_agent_tools`,
 * `load-leaves/seed_17_effects_human_wait` and `load-core/seed_18_research_pipeline`, loaded with
 * the conformance directory as `baseDir` and a search path holding seed 03 (`03-research-one.yaml`
 * of the seed 04 case), as `graph_growth_e2e.test.ts` loads them.
 *
 * Covered by `run-core/engine_start_end_e2e__inputs_default_resolves_via_start`:
 * `test_inputs_default_resolves_via_start_d_b_closed` (its trace pins the `RunStarted` inputs
 * `RunResult.input` reads).
 *
 * Already ported: `test_author_identical_run_results_hold`, which re-runs rows of
 * `test_golden_run_results.py`: the CALL and MAP run results in `graph_growth_e2e.test.ts`'s
 * `f0 goldens hold`, the HUMAN_INPUT-then-WAIT run and the three output arities in
 * `golden_run_results.test.ts`.
 *
 * Not ported: `test_nodekind_has_no_retired_members`, `test_retired_node_modules_are_gone`,
 * `test_engine_source_has_no_live_reference_to_retired_constructs` and
 * `test_grep_clean_guard_actually_bites`: guards (and the AST scan behind them) against Python's
 * retired `RefNode`, OUTPUT_RESOLVER and COLLECTOR constructs and their modules, which TS never had.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { caseText, linkConformanceFixtures, loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const REPRESENTATIVE = [
  "load-core/seed_00_hello_agent",
  "load-agent/seed_03_research_one",
  "load-map-call/seed_04_call",
  "load-map-call/seed_05_call_map",
  "load-agent/seed_14_agent_tools",
  "load-leaves/seed_17_effects_human_wait",
  "load-core/seed_18_research_pipeline",
];

/** The search path holding seed 03, as `03-research-one.yaml`. */
let seeds: string;

beforeAll(() => {
  seeds = mkdtempSync(join(tmpdir(), "yian-start-end-e2e-"));
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
  return loadConformanceFlow(text, { searchPaths: [seeds], sourceLabel: label, env: hostEnv(seeds) });
}

// --- exactly one START_ID and one END_ID per representative flow --- //

test.each(REPRESENTATIVE)("exactly one start and one end per flow: %s", async (seed) => {
  const compiled = (await loadSeed(seed)).compiled;
  const ids = (kind: NodeKind) => [...compiled.nodes].filter(([, node]) => node.kind === kind).map(([id]) => id);
  expect(ids(NodeKind.START)).toEqual([START_ID]);
  expect(ids(NodeKind.END)).toEqual([END_ID]);
  expect([compiled.startId, compiled.endId]).toEqual([START_ID, END_ID]);
});

// --- no retired constructs in the compiled IR --- //

test.each(REPRESENTATIVE)("no retired node kinds in compiled ir: %s", async (seed) => {
  const compiled = (await loadSeed(seed)).compiled;
  const kinds = new Set([...compiled.nodes.values()].map((node) => node.kind));
  // Every compiled kind belongs to the closed current set.
  const current: readonly string[] = Object.values(NodeKind);
  for (const kind of kinds) expect(current).toContain(kind);
  // `__start__` and `__end__` name only the boundary nodes, and every edge touching them is to or
  // from a real boundary node.
  for (const [id, node] of compiled.nodes) {
    if (id === START_ID || id === END_ID) expect<string[]>([NodeKind.START, NodeKind.END]).toContain(node.kind);
  }
  const boundary = new Set([START_ID, END_ID]);
  for (const edge of compiled.edges) {
    if (boundary.has(edge.from)) expect(compiled.nodes.get(edge.from)!.kind).toBe(NodeKind.START);
    if (boundary.has(edge.to)) expect(compiled.nodes.get(edge.to)!.kind).toBe(NodeKind.END);
  }
});

// --- the representative seeds load with a runnable entry --- //

test.each(REPRESENTATIVE)("representative seed loads with structure: %s", async (seed) => {
  const loaded = await loadSeed(seed);
  expect(loaded.compiled.nodes.size).toBeGreaterThan(0); // loaded, with a body
  expect(loaded.compiled.nodes.has(loaded.compiled.startId)).toBe(true); // a runnable entry
});
