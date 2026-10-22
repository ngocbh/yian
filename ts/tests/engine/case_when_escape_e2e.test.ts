/**
 * A searched CASE whose `when:` reads a skipped branch under `:-`, `|` and `:?`: a port of
 * `tests/engine/test_case_when_escape_e2e.py`.
 *
 * `gate` routes on `input.score` to `pro` or `con` (only one runs). `gate2` is a searched CASE
 * whose `when:` reads `pro.output`, routing to `hot` or `cold`. With `score: 0.2`, `pro` is
 * skipped: a plain read skips `gate2` with it, while a read under an escape runs `gate2` and lets
 * the escape supply the value, as a binding of that escape would. The `when:` stays lazy: a later
 * case's `:?` or call never runs when an earlier case matches.
 *
 * Ported: `test_a_literal_coalesce_runs_the_gate_on_a_skipped_branch`,
 * `test_a_required_read_fails_the_gate_on_a_skipped_branch`,
 * `test_a_coalesce_of_two_branches_reads_the_one_that_ran`,
 * `test_a_call_on_a_skipped_branch_in_a_later_case_is_not_evaluated`,
 * `test_a_default_reads_the_branch_when_it_ran` and both rows of
 * `test_an_unknown_namespace_inside_a_co_skipping_escape_is_reported_once`. The CODE references
 * are TS's (`./fixtures/compose_codefns.ts:pick_pro`).
 *
 * Changed: the failed `when:` of `test_a_required_read_fails_the_gate_on_a_skipped_branch` fails
 * with code `condition.evaluate`, where Python's is `node.execute` (decision 36).
 *
 * Covered by `run-case/engine_case_when_escape_e2e__a_default_runs_the_gate_on_a_skipped_branch`:
 * `test_a_default_runs_the_gate_on_a_skipped_branch`; by
 * `run-case/engine_case_when_escape_e2e__a_plain_read_skips_the_gate_with_the_branch`:
 * `test_a_plain_read_skips_the_gate_with_the_branch`; by
 * `run-case/engine_case_when_escape_e2e__a_later_required_read_is_lazy`:
 * `test_a_later_case_required_read_is_not_evaluated_when_an_earlier_case_matches`; by
 * `run-case/engine_case_when_escape_e2e__a_call_on_a_skipped_branch_fails_the_gate` (TS expect,
 * code `condition.evaluate`, decision 36):
 * `test_a_call_on_a_skipped_branch_under_a_default_fails_in_the_predicate`; by
 * `run-case/engine_case_when_escape_e2e__a_co_skipping_default_skips_the_gate`:
 * `test_a_co_skipping_default_skips_the_gate_when_its_producers_skip`; by
 * `load-case/engine_case_when_escape_e2e__an_unknown_field_inside_an_escape` (pinned in
 * `validate.test.ts`; its whole message holds neither `__r` nor `:- null`):
 * `test_an_unknown_field_inside_an_escape_names_the_authored_reference`; by
 * `load-case/engine_case_when_escape_e2e__an_on_case_with_no_cases_still_checks_its_on_value`
 * (pinned in `validate.test.ts`): `test_an_on_case_with_no_cases_still_checks_its_on_value`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type RunResult, runFlow } from "../../src/run.ts";
import { loadConformanceFlow, loadErrorAsync } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-case-when-escape-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const FLOW = `
id: when_escape
name: when_escape
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
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_pro
    input:
      topic: \${input.topic}
    output: str
  con:
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_con
    input:
      topic: \${input.topic}
    output: str
  gate2:
    kind: case
    cases:
<CASES>
    else: cold
  hot:
    kind: code
    code: ./fixtures/compose_codefns.ts:positive
    input:
      topic: \${input.topic}
    output: str
  cold:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      topic: \${input.topic}
    output: str
output: \${gate2.output}
`;

const HOT = "pro case for ACME";
const COLD = "cautious note for ACME";

/** `FLOW` with `gate2`'s `[when, then]` cases spliced in. */
function flow(...cases: [string, string][]): string {
  const body = cases.map(([when, then]) => `      - when: ${when}\n        then: ${then}\n`).join("");
  return FLOW.replace("<CASES>\n", body);
}

/** Load and run the flow with `cases`, on a score that skips `pro` unless raised. */
async function run(cases: [string, string][], score = 0.2): Promise<RunResult> {
  return runFlow(await loadConformanceFlow(flow(...cases), { env }), { score, topic: "ACME" }, { env });
}

test("a literal coalesce runs the gate on a skipped branch", async () => {
  const out = await run([[`'(pro.output | "none") == "none"'`, "hot"]]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a required read fails the gate on a skipped branch", async () => {
  const out = await run([[`'(pro.output :? "pro must run") == "x"'`, "hot"]]);
  expect(out.status).toBe("failed");
  expect(out.failure!.code).toBe("condition.evaluate");
  expect(out.failure!.message).toBe("pro must run");
  expect(out.failure!.nodeId).toBe("gate2");
  expect(out.failure!.causeType).toBe("RequiredError");
  expect(out.failure!.trace[0]!.key).toEndWith("/s:case/i:0/s:when");
});

test("a coalesce of two branches reads the one that ran", async () => {
  const out = await run([[`'(pro.output | con.output) == "con: ACME"'`, "hot"]]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a call on a skipped branch in a later case is not evaluated", async () => {
  const out = await run([
    [`"input.score < 0.5"`, "hot"],
    [`'(upper(pro.output) :- con.output) == "con: ACME"'`, "cold"],
  ]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a default reads the branch when it ran", async () => {
  const out = await run([[`'(pro.output :- "none") == "none"'`, "hot"]], 0.9);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(COLD);
});

test.each([
  [`"(nope.output | pro.output) == 1"`, "${nope.output}"],
  [`"(pro.output :- nope.output.x) == 1"`, "${nope.output.x}"],
])("an unknown namespace inside a co skipping escape is reported once: %s", async (when, reference) => {
  const error = await loadErrorAsync(loadConformanceFlow(flow([when, "hot"]), { env }));
  expect(error.message).toBe(
    "flow has unresolved references:\n" +
      `  node 'gate2' cases[0].when from: reference ${reference} uses unknown namespace 'nope'`,
  );
});
