/**
 * Agent variables declared with YAML tags (`num_sentences: !int(1..12) 4`): a port of the
 * declaration tests of `tests/compose/test_tagged_variables.py`, reading the declarations with
 * `parametersFromYaml` (`readSourceParameters`) where Python loads a flow. Ints are `bigint`.
 *
 * Ported: `test_tagged_variables_read_their_types_and_defaults`, and
 * `test_a_tagged_variable_needs_a_default_that_fits`: its declarations read on their own, and
 * its compact and inline placements (through `assembleText`; the inline one pins the whole
 * message).
 *
 * TS-only: the harvested-case table, which assembles the conformance case's flow as the loader
 * calls `assemble`, comparing the message and the last frame.
 *
 * Ported: `test_tagged_variables_render_their_defaults`, its compact placement: the assembled AGENT
 * runs through `forward` with a `ScriptedModel` (its environment sets the default model, as
 * Python's test configuration does), the request's prompt checked where Python's patched
 * `run_agent` returns it as the run's output.
 *
 * Later (T4-8, T3-22): the definition, flat-definition and inline placements of
 * `test_tagged_variables_render_their_defaults` (a loaded flow run through CALL).
 * Later (T3-22): the definition and flat-definition placements of
 * `test_a_tagged_variable_needs_a_default_that_fits`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { assembled, assembleText, type CaseRow, loadError, parametersFromYaml, runCase } from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

/** The default model Python's test configuration sets; the compact flow has no `llm_config`. */
const DEFAULT_MODEL = { provider: "openai", model: "gpt-5.5" };

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

const HEADER = `\
id: tagged_variables
name: tagged_variables
typedefs:
  Tone: Literal(concise, detailed)
`;

/** Spell one AGENT body declaring `variables` (unindented), its prompt rendering each. */
function agentBody(variables: string): string {
  return (
    "kind: agent\nvariables:\n" +
    variables.replace(/^(?=.)/gm, "  ") +
    'prompt: "${instruction} in ${num_sentences} sentences, ${tone}."\n' +
    "output: str\n"
  );
}

/** Spell a compact flow: the AGENT body is the flow. */
function compactFlow(variables: string): string {
  return HEADER + agentBody(variables);
}

/** Spell a flow whose inline CALL `invoke` holds the AGENT as its node `writer`. */
function inlineFlow(variables: string): string {
  return (
    HEADER +
    "nodes:\n  invoke:\n    kind: call\n    nodes:\n      writer:\n" +
    agentBody(variables).replace(/^(?=.)/gm, "        ") +
    "    output: ${writer.output}\noutput: ${invoke.output}\n"
  );
}

/** Read `variables` (unindented declarations) as agent variables. */
function variablesFrom(variables: string) {
  const text = "variables:\n" + variables.replace(/^(?=.)/gm, "  ");
  return parametersFromYaml(text, REG, { field: "variables", requireDefault: true });
}

/** `(declaration, fragment)`: a declaration and a fragment of the error it raises. */
const MISFITS: [string, string][] = [
  ["x: !int\n", "variable 'x': a variable needs a default: `x: !int <default>`"],
  ["num_sentences: !int(1..12) 13\n", "'num_sentences': `13` is not a valid int(1..12)"],
  ["tone: !Tone loud\n", "'tone': `loud` is not a valid Tone"],
];

describe("tagged variables", () => {
  test.each(MISFITS)("a tagged variable needs a default that fits: %p", (declaration, fragment) => {
    expect(loadError(() => variablesFrom(withDeclaration(declaration))).message).toContain(fragment);
  });

  test.each(MISFITS)("a tagged variable in a compact flow needs a default that fits: %p", (declaration, fragment) => {
    expect(loadError(() => assembleText(compactFlow(withDeclaration(declaration)))).message).toContain(fragment);
  });

  test.each(MISFITS)("a tagged variable in an inline body needs a default that fits: %p", (declaration, message) => {
    // The child's assembly reads the variables, so the message is the fragment alone.
    expect(loadError(() => assembleText(inlineFlow(withDeclaration(declaration)))).message).toBe(message);
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

  test("tagged variables render their defaults", async () => {
    const reply = new Message({ role: "assistant", parts: [new TextPart("done")] });
    const model = new ScriptedModel([new Response({ message: reply, finishReason: "stop" })], DEFAULT_MODEL);
    const node = assembleText(compactFlow(VARIABLES)).flow.compiled.nodes.get("tagged_variables")!;
    const defaults: Readonly<Record<string, string>> = {
      YIAN_DEFAULT_PROVIDER: DEFAULT_MODEL.provider,
      YIAN_DEFAULT_MODEL: DEFAULT_MODEL.model,
    };
    const env = (name: string) => defaults[name];

    await node.forward({}, { models: { env, createModel: () => model } });

    expect(model.requests[0]!.messages.at(-1)!.text()).toBe("Summarize clearly in 4 sentences, concise.");
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_tagged_variables__a_variable_needs_a_default", call: assembled },
])("assemble: $id", runCase);
