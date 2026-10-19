/**
 * Agent variables declared with YAML tags (`num_sentences: !int(1..12) 4`): a port of the
 * declaration tests of `tests/compose/test_tagged_variables.py`, reading the declarations with
 * `parametersFromYaml` (`readSourceParameters`) where Python loads a flow. Ints are `bigint`.
 *
 * Ported: `test_tagged_variables_read_their_types_and_defaults`, and the declarations of
 * `test_a_tagged_variable_needs_a_default_that_fits`, read once rather than at each placement.
 *
 * TS-only: the harvested-case table, which makes the loader's call on the `variables:`
 * declarations of the conformance case this reader decides, comparing the message and the last
 * frame.
 *
 * Later (T3-15, T3-22): the agent runs at each placement
 * (`test_tagged_variables_render_their_defaults`) and the `load_flow` placements.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { type CaseRow, loadError, parametersFromYaml, runCase, variablesAt } from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({ Tone: "Literal(concise, detailed)" });

const VARIABLES = `\
num_sentences: !int(1..12) 4
tone: !Tone concise
instruction: !str "Summarize clearly"
`;

/** Return the shared variables with `declaration` replacing (or added to) its name. */
function withDeclaration(declaration: string): string {
  const name = declaration.split(":", 1)[0]!;
  const kept = VARIABLES.split(/(?<=\n)/).filter((line) => !line.startsWith(`${name}:`));
  return kept.join("") + declaration;
}

/** Read `variables` (unindented declarations) as agent variables. */
function variablesFrom(variables: string) {
  const text = "variables:\n" + variables.replace(/^(?=.)/gm, "  ");
  return parametersFromYaml(text, REG, { field: "variables", requireDefault: true });
}

describe("tagged variables", () => {
  test.each([
    ["x: !int\n", "variable 'x': a variable needs a default: `x: !int <default>`"],
    ["num_sentences: !int(1..12) 13\n", "'num_sentences': `13` is not a valid int(1..12)"],
    ["tone: !Tone loud\n", "'tone': `loud` is not a valid Tone"],
  ])("a tagged variable needs a default that fits: %p", (declaration, fragment) => {
    expect(loadError(() => variablesFrom(withDeclaration(declaration))).message).toContain(fragment);
  });

  test("tagged variables read their types and defaults", () => {
    const variables = variablesFrom(VARIABLES);
    expect(Object.fromEntries(variables.map((variable) => [variable.name, variable.default]))).toEqual({
      num_sentences: 4n,
      tone: "concise",
      instruction: "Summarize clearly",
    });
    expect(variables.every((variable) => variable.hasDefault && !variable.required)).toBe(true);
    expect(variables[0]!.contract!.type.bounds).toEqual([1n, 12n]);
    expect(variables[1]!.contract!.type.tags).toEqual(new Set(["concise", "detailed"]));
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  {
    id: "load-types/compose_tagged_variables__a_variable_needs_a_default",
    // A compact flow's one node is named by the flow's `id:`.
    call: variablesAt(["variables"], ["node", "tagged_variables"]),
  },
])("readSourceParameters: $id", runCase);
