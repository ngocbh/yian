/**
 * `depends_on` co-skip versus `runs_after` pure ordering: a port of
 * `tests/engine/test_depends_on_e2e.py`'s load rows.
 *
 * Ported: `test_unknown_depends_on_target_is_located_load_error`, through its harvested case
 * `load-core/engine_depends_on_e2e__unknown_depends_on_target`, whose error `inferOrderingEdges`
 * raises during `assemble`; and `test_ordering_cycle_is_loud`, through its harvested case
 * `load-core/engine_depends_on_e2e__ordering_cycle_is_loud`, with `full` (every frame and the
 * notes).
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

import { test } from "bun:test";

import { assembled, type CaseRow, runCase } from "../compose/declarations.ts";

test.each<CaseRow>([
  { id: "load-core/engine_depends_on_e2e__unknown_depends_on_target", call: assembled },
  { id: "load-core/engine_depends_on_e2e__ordering_cycle_is_loud", call: assembled, full: true },
])("assemble: $id", runCase);
