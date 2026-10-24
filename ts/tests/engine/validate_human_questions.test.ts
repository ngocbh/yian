/**
 * The load-time human_input questions check through assembly: a port of
 * `tests/engine/test_validate_human_questions.py`.
 *
 * Ported: `test_rejects_bad_literal`, `test_rejects_ref_to_undeclared_input` and
 * `test_rejects_pool_ref_in_questions_field`, through `assembleText` where Python calls
 * `load_flow`. The last is refused by the human_input reader before the check runs, as in Python.
 * Changed: its CODE reference is a TS fixture (`./fixtures/compose_codefns.ts:make_plan`), not
 * Python's `tests.seeds.fns:questions_seed`; the reader refuses the flow before the code matters.
 *
 * The harvested case from this file, pinned at the end:
 * `load-leaves/engine_validate_human_questions__duplicate_question_headers`.
 */

import { afterEach, expect, test } from "bun:test";

import { forget } from "../../src/defaults.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../compose/declarations.ts";

afterEach(() => forget());

const BAD_LITERAL = `
id: f
name: f
nodes:
  ask:
    kind: human_input
    questions:
      - {question: "dup", header: "H"}
      - {question: "dup2", header: "H"}
output: \${ask.output}
`;

const REF_UNDECLARED = `
id: f
name: f
nodes:
  ask: {kind: human_input, questions: "\${qs}"}
output: \${ask.output}
`;

const POOL_REF_IN_FIELD = `
id: f
name: f
nodes:
  a: {kind: code, output: list(object), code: ./fixtures/compose_codefns.ts:make_plan, input: {seed: "\${input.s}"}}
  ask: {kind: human_input, questions: "\${a.output}"}
input:
  s: !str
output: \${ask.output}
`;

test("a bad literal questions list is refused", () => {
  expect(loadError(() => assembleText(BAD_LITERAL)).message).toContain("header");
});

test("a questions reference to an undeclared input is refused", () => {
  loadError(() => assembleText(REF_UNDECLARED));
});

test("a node-output reference in the questions field is refused", () => {
  // A node-output reference belongs in `input:` wiring.
  loadError(() => assembleText(POOL_REF_IN_FIELD));
});

test.each<CaseRow>([
  { id: "load-leaves/engine_validate_human_questions__duplicate_question_headers", call: assembled },
])("case $id", (row) => runCase(row));
