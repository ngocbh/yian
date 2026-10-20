/**
 * `depends_on` co-skip versus `runs_after` pure ordering: a port of
 * `tests/engine/test_depends_on_e2e.py`'s load rows.
 *
 * Ported: `test_unknown_depends_on_target_is_located_load_error`, through its harvested case
 * `load-core/engine_depends_on_e2e__unknown_depends_on_target`, whose error `inferOrderingEdges`
 * raises during `assemble`; and `test_ordering_cycle_is_loud`, through its harvested case
 * `load-core/engine_depends_on_e2e__ordering_cycle_is_loud`, with its notes and both frames.
 *
 * Changed: the flow's CODE reference is TS's (`./fixtures/compose_codefns.ts:pick_pro`), as the
 * case's `flow` names it. The case pins the whole message and range, where Python checks the
 * message holds `depends_on` and `nope` and the last frame has a line, or that the cycle message
 * holds `cycle`. The cycle message writes its node list in the text form (rule `messages`).
 *
 * Later (T4): the run rows (`test_warm_runs_both_dependents_run`,
 * `test_warm_skipped_depends_on_co_skips_runs_after_survives`,
 * `test_ordering_on_a_case_runs_on_every_branch`,
 * `test_branch_target_ordered_on_its_case_runs_when_taken`,
 * `test_depends_on_a_case_with_no_match_and_no_else_still_runs`).
 */

import { expect, test } from "bun:test";

import { assembled, assembleText, type CaseRow, caseText, loadError, runCase } from "../compose/declarations.ts";

const ORDERING_CYCLE = "load-core/engine_depends_on_e2e__ordering_cycle_is_loud";

test.each<CaseRow>([
  { id: "load-core/engine_depends_on_e2e__unknown_depends_on_target", call: assembled },
  { id: ORDERING_CYCLE, call: assembled },
])("assemble: $id", runCase);

test("an ordering cycle frames both nodes and notes both ordering edges", () => {
  const error = loadError(() => assembleText(caseText(ORDERING_CYCLE).text));
  expect(error.diagnostic.frames.map(({ range }) => [range.startLine, range.startCol, range.endLine, range.endCol])).toEqual([
    [7, 3, 7, 4],
    [14, 3, 14, 4],
  ]);
  expect(error.diagnostic.notes).toEqual(["a depends on b (ordering)", "b depends on a (ordering)"]);
});
