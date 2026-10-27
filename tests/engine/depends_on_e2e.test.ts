/**
 * `depends_on` co-skip versus `runs_after` pure ordering, and the load errors for an unknown
 * target and an ordering cycle.
 */

import { expect, test } from "bun:test";

import { NodeSucceeded } from "../../src/events.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { assembled, assembleText, type CaseRow, loadConformanceFlow, loadError, runCase } from "../compose/declarations.ts";

const FNS = "./fixtures/compose_codefns.ts";

const FLOW = `
id: ordering-e2e
name: ordering_e2e
input:
  score: !float
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: warm
  warm:
    kind: code
    code: ${FNS}:pick_pro
    input:
      topic: \${input.topic}
    output: str
  dep_fetch:
    kind: code
    depends_on: [warm]
    code: ${FNS}:positive
    input:
      topic: \${input.topic}
    output: str
  after_fetch:
    kind: code
    runs_after: [warm]
    code: ${FNS}:cautious
    input:
      topic: \${input.topic}
    output: str
output: \${dep_fetch.output | after_fetch.output}
`;

/** The ids of the nodes that succeeded in a run. */
const succeeded = (result: RunResult) =>
  result.events.filter((event) => event instanceof NodeSucceeded).map((event) => (event as NodeSucceeded).nodeId);

test("warm runs, both dependents run", async () => {
  // score >= 0.5: warm runs, both ordering edges are satisfied and both dependents run.
  const result = await runFlow(await loadConformanceFlow(FLOW), { score: 0.9, topic: "ACME" });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(succeeded(result)).toEqual(expect.arrayContaining(["warm", "dep_fetch", "after_fetch"]));
  // The coalesce picks the first non-null producer (dep_fetch ran).
  expect(result.output).toBe("pro case for ACME");
});

// ---------- ordering edges from a case: the routed case settled, so they hold ----------

/**
 * A `gate` case routing to `pro`/`con`, with an ordering edge `field: [gate]` on a third node
 * `note` (or, with `onBranch`, on the `pro` branch target itself).
 */
function caseSourceFlow(field: string, onBranch: boolean): string {
  const branchDep = onBranch ? `\n    ${field}: [gate]` : "";
  const noteDep = onBranch ? "" : `\n    ${field}: [gate]`;
  return `
id: case-source
name: case_source
input:
  score: !float
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: pro
    else: con
  pro:
    kind: code${branchDep}
    code: ${FNS}:pick_pro
    input:
      topic: \${input.topic}
    output: str
  con:
    kind: code
    code: ${FNS}:pick_con
    input:
      topic: \${input.topic}
    output: str
  note:
    kind: code${noteDep}
    code: ${FNS}:cautious
    input:
      topic: \${input.topic}
    output: str
output:
  pick: "\${pro.output | con.output}"
  note: \${note.output}
`;
}

const FIELDS = ["depends_on", "runs_after"];

test.each(FIELDS.flatMap((field) => [0.9, 0.2].map((score) => [field, score] as const)))(
  "ordering on a case runs on every branch: %s, score %p",
  async (field, score) => {
    // The case routes either way, so it settles; an ordering edge from it is satisfied.
    const result = await runFlow(await loadConformanceFlow(caseSourceFlow(field, false)), { score, topic: "ACME" });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(succeeded(result)).toContain("note");
    expect((result.output as Record<string, unknown>)["note"]).toBe("cautious note for ACME");
  },
);

test.each(FIELDS)("a branch target ordered on its case runs when taken: %s", async (field) => {
  const flow = await loadConformanceFlow(caseSourceFlow(field, true));
  const taken = await runFlow(flow, { score: 0.9, topic: "ACME" });
  expect(taken.status, taken.failure?.message).toBe("succeeded");
  expect((taken.output as Record<string, unknown>)["pick"]).toBe("pro: ACME");
  expect(succeeded(taken)).not.toContain("con");
  const lost = await runFlow(flow, { score: 0.2, topic: "ACME" });
  expect(lost.status, lost.failure?.message).toBe("succeeded");
  expect((lost.output as Record<string, unknown>)["pick"]).toBe("con: ACME");
  expect(succeeded(lost)).not.toContain("pro");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-core/engine_depends_on_e2e__unknown_depends_on_target", call: assembled },
  { id: "load-core/engine_depends_on_e2e__ordering_cycle_is_loud", call: assembled, full: true },
])("assemble: $id", runCase);

test("an unknown depends_on target is reported before an unknown reference", () => {
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  a:\n    kind: code\n    output: int\n    depends_on: [ghost]\n    code: return 1\n" +
    "  b:\n    kind: code\n    output: int\n    input:\n      v: ${nope.output}\n    code: return 1\n" +
    "output: ${a.output}\n";
  expect(loadError(() => assembleText(text)).message).toContain("`depends_on` references unknown node 'ghost'");
});
