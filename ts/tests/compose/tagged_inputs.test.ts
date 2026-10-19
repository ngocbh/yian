/**
 * Flow inputs declared with YAML tags (`qty: !int`, `history: !list(Message) []`): a port of
 * the declaration tests of `tests/compose/test_tagged_inputs.py`, reading the declarations with
 * `parametersFromYaml` (`readSourceParameters`) where Python loads a flow. Ints are `bigint`;
 * a missing-field list is written in the text form (`["kind"]`).
 *
 * Ported: `test_a_tag_does_not_fill_in_a_missing_field`,
 * `test_tagged_parameters_read_required_and_default_from_the_value`,
 * `test_source_free_inputs_accept_the_parenthesized_spelling`, and the declarations of
 * `test_a_tagged_default_must_fit_its_tag`, read once rather than at each placement. An unquoted
 * date is text (rule `dates`), so `d: !str 2026-01-01` loads where Python asks for quotes.
 *
 * TS-only: the harvested-case table, which makes the loader's call on the `input:` declarations
 * of the conformance cases these readers decide, comparing the message and the last frame.
 *
 * Later (T3-22): the `load_flow` placements and runs
 * (`test_tagged_inputs_load_and_run_with_their_defaults`,
 * `test_a_tagged_input_without_a_value_is_required`), the origins
 * (`test_tagged_parameter_origins_point_at_the_declaration`), the single-node `input:` list
 * (`test_a_single_node_input_list_shows_the_tagged_spelling`), and
 * `test_a_tagged_default_with_a_non_string_key_is_a_load_error`: TS reads `{true: 1}`'s key as
 * `"true"` and records it in `document.nonTextKey`, which the loader refuses, so
 * `readSourceParameters` alone accepts it.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { readFlowInputs } from "../../src/compose/helpers/types.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { isListKind } from "../../src/typesys/values.ts";
import { type CaseRow, inputsAt, loadError, parametersFromYaml, runCase } from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({
  Cfg: { kind: "Literal(cfg)", a: "int" },
  Message: { role: "str", content: "str" },
});

const DECLARATIONS = `\
qty: !int
history: !list(Message) []
model: !Optional(str)
note: !Optional(str) null
cfg: !Cfg {kind: cfg, a: 1}
s: !str \${x}
`;

/** Indent each line of `text` by two spaces. */
function indented(text: string): string {
  return text.replace(/^(?=.)/gm, "  ");
}

describe("tagged declarations", () => {
  test("a tag does not fill in a missing field", () => {
    const error = loadError(() => parametersFromYaml("input:\n  cfg: !Cfg {a: 1}\n", REG));
    expect(error.message).toContain('missing required fields: ["kind"]');
    const [parameter] = parametersFromYaml("input:\n  cfg: !Cfg {kind: cfg, a: 1}\n", REG);
    expect(parameter!.default).toEqual({ kind: "cfg", a: 1n });
  });

  test("a tagged parameter reads required and its default from the value", () => {
    const parameters = parametersFromYaml("input:\n" + indented(DECLARATIONS), REG);
    const shape = Object.fromEntries(
      parameters.map((parameter) => [
        parameter.name,
        [parameter.required, parameter.hasDefault, parameter.default],
      ]),
    );
    expect(shape).toEqual({
      qty: [true, false, null],
      history: [false, true, []],
      model: [false, false, null],
      note: [false, true, null],
      cfg: [false, true, { kind: "cfg", a: 1n }],
      s: [false, true, "${x}"],
    });
    const types = new Map(parameters.map((parameter) => [parameter.name, parameter.contract!.type]));
    expect(types.get("model")!.nullable).toBe(true);
    expect(isListKind(types.get("history")!.kind)).toBe(true);
    expect(parameters.every((parameter) => parameter.coerceStrings)).toBe(true);
  });

  test("a source-free input accepts the parenthesized spelling", () => {
    const [parameter] = readFlowInputs({ x: "Optional(int)" }, new Map());
    expect(parameter!.contract!.type.nullable).toBe(true);
    expect(parameter!.required).toBe(false);
  });

  test.each([
    [
      "cfg: !Cfg {a: 1}",
      "'cfg': the value is not a valid Cfg: record missing required fields: [\"kind\"]",
    ],
    ["n: !int(1..12) 13", "'n': `13` is not a valid int(1..12)"],
  ])("a tagged default must fit its tag: %p", (declaration, fragment) => {
    const text = `input:\n  qty: !int\n  ${declaration}\n`;
    expect(loadError(() => parametersFromYaml(text, REG)).message).toContain(fragment);
  });

  test("an unquoted date fits a str tag, as text", () => {
    const parameters = parametersFromYaml("input:\n  qty: !int\n  d: !str 2026-01-01\n", REG);
    expect(parameters[1]!.default).toBe("2026-01-01");
  });
});

// ---------- harvested conformance cases ----------

const rootInput = inputsAt(["input"]);

test.each<CaseRow>([
  { id: "load-types/compose_tagged_inputs__a_tagged_default_must_fit_its_range", call: rootInput },
  {
    id: "load-types/compose_tagged_inputs__a_tagged_record_default_must_have_every_field",
    call: rootInput,
  },
  {
    id: "load-types/engine_date_grammar__flow_default_outside_the_grammar_fails_at_load",
    call: rootInput,
  },
  { id: "types/compose_type_name_checks__empty_literal_member_in_a_tag", call: rootInput },
])("readSourceFlowInputs: $id", runCase);
