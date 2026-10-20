/**
 * A bad adaptive brief is reported on the gate at the brief: a port of
 * `tests/engine/test_adaptive_brief_errors.py`.
 *
 * Ported: `test_a_clean_brief_loads`.
 *
 * Changed: the flow assembles with `assembleText`, where Python loads it.
 *
 * Later (T3-18): `test_bad_brief_is_named_and_located_at_the_brief` and
 * `test_brief_and_gate_prompt_errors_are_both_reported` (the reference validator's prompt-scope
 * check). Later (T3-22): `test_compact_flow_brief_error_names_the_flow_node` (a compact flow),
 * `test_aliased_brief_shares_the_anchor_line` (the reference validator too) and
 * `test_defs_body_brief_error` (a `defs:` body).
 */

import { expect, test } from "bun:test";

import { assembleText } from "../compose/declarations.ts";

/** Return the adaptive-gate flow whose gate has `prompt:` line `gatePrompt` and the brief `brief`. */
const flow = (brief: string, gatePrompt = '    prompt: "Pick one for ${proj}"\n') => `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
${gatePrompt}    adaptive_questions:
      prompt: "${brief}"
output: \${ask.output}
`;

test("a clean brief loads", () => {
  expect(assembleText(flow("Q ${proj}")).flow.compiled.nodes.has("ask__compose")).toBe(true);
});
