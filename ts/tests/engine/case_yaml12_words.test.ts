/**
 * CASE `when:` words that YAML 1.1 read as booleans are text under YAML 1.2 rules, a port of
 * `tests/engine/test_case_yaml12_words.py`.
 *
 * Ported: `test_when_yes_on_a_bool_is_a_load_error`, and the load half of
 * `test_yes_and_no_cover_a_literal_input_without_else`.
 *
 * Changed: the flows assemble (`assembleText`), where Python loads them.
 *
 * Later (T4-8 follow-up): the run half of `test_yes_and_no_cover_a_literal_input_without_else`.
 *
 * The harvested case is pinned at the end.
 */

import { expect, test } from "bun:test";

import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

/** Return a flow whose `route` case matches `input.cmd` and routes to echo nodes. */
function flow(cmdType: string, caseTail: string): string {
  return (
    "id: f\nname: f\ntypedefs:\n  Cmd: Literal(yes, no)\ninput:\n" +
    `  cmd: !${cmdType}\n` +
    "nodes:\n  route:\n    kind: case\n    on: ${input.cmd}\n    cases:\n" +
    "      - when: yes\n        then: agree\n" +
    caseTail +
    "  agree:\n    kind: code\n    code: tests.engine._compose_codefns:echo\n" +
    "    input:\n      topic: agreed\n    output: str\n" +
    "  refuse:\n    kind: code\n    code: tests.engine._compose_codefns:echo\n" +
    "    input:\n      topic: refused\n    output: str\n" +
    "output: ${agree.output | refuse.output}\n"
  );
}

test("yes and no cover a literal input without else: assembles", () => {
  assembleText(flow("Cmd", "      - when: no\n        then: refuse\n"));
});

test("when yes on a bool is a load error", () => {
  const text = flow("bool", "    else: refuse\n");

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "case node 'route': `when: yes` is text but `on: ${input.cmd}` is `bool`; `yes` is text; write `true`",
  );
  const line = text.split("\n").findIndex((row) => row.includes("when: yes")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

// ---------- harvested conformance cases ----------

test("assembled: yaml/engine_case_yaml12_words__when_yes_on_a_bool_is_a_load_error", () => {
  runCase({ id: "yaml/engine_case_yaml12_words__when_yes_on_a_bool_is_a_load_error", call: assembled });
});
