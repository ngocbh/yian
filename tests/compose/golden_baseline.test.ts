/**
 * Golden baselines (a characterization test) for the edge ids of representative flows.
 *
 * Pins the edge ids, and the ordered `[id, inputGroup, optional]` projection, of representative
 * flows, so a change to the wiring or the edge inference proves no drift against a recorded
 * baseline rather than a remembered one. If a later change legitimately changes the topology,
 * update the literals here in the same commit and explain why.
 *
 * Coverage across the three seeds: data edges, CASE control edges and skip-flood targets, the
 * `START_ID` input-producer edges, terminal (`->__end__#0`) edges, the ordering separator (`~>`),
 * and a mapped `call` (MAP) node.
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  type CaseCall,
  caseDocument,
  caseRegistry,
  handChild,
  parametersFromYaml,
} from "../helpers/declarations.ts";

/** Seed 05's `research-one` def: `{report, asof}` from its `synth` agent and its `as_of` input. */
const researchOne = handChild({
  inputs: parametersFromYaml('input:\n  topic: !str\n  as_of: !date "2026-01-01"\n', new Map()),
  producers: { synth: new ValueContract(Type.scalar(ValueKind.STRING)) },
  output: { report: "${synth.output}", asof: "${input.as_of}" },
});

/** Each seed's case and the call that assembles it. */
const SEEDS: Record<string, readonly [id: string, call: CaseCall]> = {
  "02-case.yaml": ["load-case/seed_02_case", assembled],
  "05-call-map.yaml": ["load-map-call/seed_05_call_map", assembledWith({ "research-one": researchOne })],
  "17-effects-human-wait.yaml": ["load-leaves/seed_17_effects_human_wait", assembled],
};

/** Return seed `name`'s assembled flow. */
function compiledOf(name: string): CompiledFlow {
  const [id, call] = SEEDS[name]!;
  const { document } = caseDocument(id);
  const flow = call(document, caseRegistry(document), LoadDiagnostics.fromDocument(document), new SourceCatalogBuilder());
  return (flow as ComposedFlow).compiled;
}

// An `${input.X}` reader gets a `START_ID->reader#i` data edge (inputGroup the param, `over` or
// `until` name) instead of a bare `__start__->reader` root edge. A case branch target that also
// reads `${input.X}` therefore carries both its control edge and the START_ID data edge (the
// control veto still skip-floods it). The END_ID producer edges carry the output name.
const GOLDEN_EDGE_IDS: Record<string, string[]> = {
  // case: data (score->gate), control (gate->positive/cautious), START_ID input-producer edges,
  // terminals
  "02-case.yaml": [
    "__start__->cautious#0",
    "__start__->positive#0",
    "__start__->score#0",
    "cautious->__end__#0",
    "gate->cautious#0",
    "gate->positive#0",
    "positive->__end__#0",
    "score->gate#0",
  ],
  // mapped call (MAP) + fan-in; research_each reads ${input.topics} (over) + ${input.as_of} (opt)
  "05-call-map.yaml": [
    "__start__->research_each#0",
    "__start__->research_each#1",
    "compare->__end__#0",
    "research_each->__end__#0",
    "research_each->compare#0",
  ],
  // ask + absolute sleep: CASE gate control edges + the `~>` ordering separator
  "17-effects-human-wait.yaml": [
    "__start__->propose#0",
    "__start__->settle#0",
    "abort->__end__#0",
    "approve->gate#0",
    "confirm->__end__#0",
    "gate->abort#0",
    "gate->settle#0",
    "propose->abort#0",
    "propose->approve#0",
    "propose->confirm#0",
    "settle~>confirm#0",
  ],
};

test.each(Object.keys(GOLDEN_EDGE_IDS).sort())("%s edge ids match the baseline", (name) => {
  expect(
    compiledOf(name)
      .edges.map((edge) => edge.id)
      .sort(),
  ).toEqual(GOLDEN_EDGE_IDS[name]!);
});

// Stronger than the sorted ids: catches an emission-order, inputGroup or optional drift that keeps
// the id set. The END_ID producer edges carry `inputGroup` = the output name and come before the
// START_ID root edges (`synthesizeBoundaryGraph` returns END_ID edges then START_ID edges, in
// output-declaration order).
const GOLDEN_EDGE_TUPLES: Record<string, [id: string, inputGroup: string | null, optional: boolean][]> = {
  "02-case.yaml": [
    ["__start__->score#0", "topic", false],
    ["__start__->positive#0", "topic", false],
    ["__start__->cautious#0", "topic", false],
    ["score->gate#0", "__r0", false],
    ["gate->positive#0", null, false],
    ["gate->cautious#0", null, false],
    ["positive->__end__#0", "result", false],
    ["cautious->__end__#0", "result", false],
  ],
  "05-call-map.yaml": [
    ["__start__->research_each#0", "over", false],
    ["__start__->research_each#1", "as_of", true],
    ["research_each->compare#0", "briefs", false],
    ["compare->__end__#0", "report", false],
    ["research_each->__end__#0", "briefs", false],
  ],
  "17-effects-human-wait.yaml": [
    ["__start__->propose#0", "topic", false],
    // The ask's prompt embeds `${propose.output}` in text, so its edge doesn't co-skip.
    ["propose->approve#0", "prompt", true],
    ["__start__->settle#0", "until", false],
    ["propose->confirm#0", "action", false],
    ["propose->abort#0", "action", false],
    ["approve->gate#0", "__on", false],
    ["gate->settle#0", null, false],
    ["gate->abort#0", null, false],
    ["settle~>confirm#0", null, false],
    ["confirm->__end__#0", "result", false],
    ["abort->__end__#0", "result", false],
  ],
};

test.each(Object.keys(GOLDEN_EDGE_TUPLES).sort())("%s edge tuples match the baseline", (name) => {
  expect(compiledOf(name).edges.map((edge) => [edge.id, edge.inputGroup, edge.optional])).toEqual(
    GOLDEN_EDGE_TUPLES[name]!,
  );
});
