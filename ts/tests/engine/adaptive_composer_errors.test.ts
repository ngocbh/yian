/**
 * A load error in an adaptive gate's composer names the gate and points at the authored text: a
 * port of `tests/engine/test_adaptive_composer_errors.py`.
 *
 * Ported: `test_bad_llm_config_is_named_and_located_at_the_llm_config`,
 * `test_block_llm_config_error_is_located_at_its_first_key`, `test_valid_llm_config_loads`,
 * `test_literal_inputs_still_load`, and the cycle rows
 * (`test_self_reading_gate_cycle_names_only_the_gate`,
 * `test_cycle_through_a_gate_names_only_authored_nodes`,
 * `test_ordering_cycle_through_a_gate_names_only_authored_nodes`,
 * `test_cycle_through_a_direct_composer_reader_keeps_the_id`,
 * `test_composer_read_outside_the_cycle_still_names_only_the_gate`,
 * `test_two_self_reading_gates_each_name_only_the_gate` and
 * `test_gate_reading_its_own_composer_keeps_the_id`); the harvested
 * `load-agent/engine_adaptive_composer_errors__*` cases the composer's build or the cycle check
 * raises, pinned at the end, the cycle cases with their notes and every frame.
 *
 * Changed: the flows assemble with `assembleText`, where Python loads them. A text value reads
 * `"false"` where Python's repr writes `'false'`, and the unknown-key error carries the issue's
 * message ("Extra inputs are not permitted") where pydantic adds its `extra_forbidden` type. Each
 * bad-config row pins the whole message where Python matches fragments. The cycle message writes
 * its node list in the text form (`["ask","b"]`, rule `messages`), and each cycle row pins the
 * whole message where Python checks its start. The inline CODE body is TypeScript.
 *
 * Later (T3-18): `test_bad_input_is_reported_once_on_the_gate`,
 * `test_two_inputs_with_the_same_bad_source_are_both_reported`,
 * `test_bad_brief_and_bad_input_are_each_reported_once`,
 * `test_bad_input_is_listed_before_a_bad_gate_prompt` (the reference validator),
 * `test_isinstance_unknown_type_names_the_gate` (the type-table pass, `fillTypeTables`).
 * Later (T3-19): `test_typed_input_mismatch_names_the_gate`.
 * Later (T3-20): `test_skipped_operand_arithmetic_names_the_gate`.
 * Later (T3-22): `test_compact_flow_llm_config_error_names_the_flow_node`,
 * `test_defs_body_input_error_names_the_gate` and `test_defs_child_cycle_names_only_the_gate`.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { assembled, assembleText, type CaseRow, caseText, loadError, runCase } from "../compose/declarations.ts";

const LLM_SITE = "node 'ask' adaptive_questions.llm_config";

/** Return the adaptive-gate flow with the `llm_config:` lines `llmConfig`; its value starts on line 10. */
const flow = (llmConfig = "") => `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
${llmConfig}output: \${ask.output}
`;

/** Return the 1-based line of the error's last frame. */
const lineOf = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

test.each([
  ["inherit", "{inherit: 'false'}", `${LLM_SITE} \`inherit:\` must be true or false, got "false"`],
  [
    "model-type",
    "{model: 5}",
    `${LLM_SITE}: \`model:\` must be text, but YAML read it as an integer; put it in quotes`,
  ],
  [
    "unknown-key",
    "{temparature: 1}",
    `${LLM_SITE}: 1 validation error for LLMConfig\ntemparature\n  Extra inputs are not permitted`,
  ],
  ["not-mapping", "5", `${LLM_SITE} must be a mapping`],
  [
    "api-key",
    "{api_key: x}",
    `${LLM_SITE}.api_key is a host credential; pass it in the run's llm_config, not in a flow file`,
  ],
])("bad llm config is named and located at the llm config: %s", (_id, config, message) => {
  const error = loadError(() => assembleText(flow(`      llm_config: ${config}\n`)));
  expect(error.message).toBe(message);
  expect(lineOf(error)).toBe(10);
});

test("block llm config error is located at its first key", () => {
  const error = loadError(() => assembleText(flow("      llm_config:\n        model: m\n        bogus: 1\n")));
  expect(error.message).toBe(
    `${LLM_SITE}: 1 validation error for LLMConfig\nbogus\n  Extra inputs are not permitted`,
  );
  expect(lineOf(error)).toBe(11);
});

test("valid llm config loads", () => {
  expect(assembleText(flow("      llm_config: {model: m}\n")).flow.compiled.nodes.has("ask__compose")).toBe(true);
});

/** Return the flow whose gate `ask` has the `input:` lines `inputs` and the brief `brief`. */
const inputFlow = (inputs: string, brief: string) => `\
id: f
name: f
input: {proj: !str }
nodes:
  up:
    kind: code
    output: int
    code: "return 1"
  ask:
    kind: human_input
    input:
${inputs}    adaptive_questions:
      prompt: "${brief}"
output: \${ask.output}
`;

test("literal inputs still load", () => {
  const inputs = "      cfg: {a: 1}\n      ls: [1, 2]\n      n: !list(int) [1]\n";
  expect(assembleText(inputFlow(inputs, "Q ${cfg} ${ls} ${n}")).flow.compiled.nodes.has("ask__compose")).toBe(true);
  const agentFlow = `\
id: f
name: f
nodes:
  a:
    kind: agent
    input:
${inputs}    prompt: "Q \${cfg} \${ls} \${n}"
    output: str
output: \${a.output}
`;
  expect(assembleText(agentFlow).flow.compiled.nodes.has("a")).toBe(true);
});

// ---------- cycles: the gate is named, not its composer, unless an author reads the composer ----------

/** Return the cycle message naming `names`. */
const cycleMessage = (names: string[]) => `flow has a cycle involving ${JSON.stringify(names)}; flows must be acyclic`;
const GENERATED = "ask__compose is generated from node 'ask'";

const SELF_READING = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${ask.output}'}
    adaptive_questions: {prompt: "Q \${proj}"}
`;

/** Return the 1-based line of the first line of `text` that starts with `needle`. */
function lineStarting(text: string, needle: string): number {
  const index = text.split("\n").findIndex((line) => line.startsWith(needle));
  if (index < 0) throw new Error(`${needle} not in flow`);
  return index + 1;
}

/** Return the start line of each frame of `error`, in order. */
const frameLines = (error: LoadError) => error.diagnostic.frames.map((frame) => frame.range.startLine);

test("a self-reading gate's cycle names only the gate", () => {
  const text = SELF_READING + "output: ${ask.output}\n";
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on ask (ask.input.proj)"]);
});

/** Return the flow where `b` reads `bReads` and the gate `ask`, with `gateExtra` lines, reads `gateReads`. */
const throughB = (bReads: string, gateExtra: string, gateReads: string) => `\
id: f
name: f
input: {proj: !str }
nodes:
  b:
    kind: agent
    input: {a: '\${${bReads}}'}
    prompt: "x \${a}"
  ask:
    kind: human_input
${gateExtra}    input: {proj: '\${${gateReads}}'}
    adaptive_questions: {prompt: "Q \${proj}"}
output: \${ask.output}
`;

test("a cycle through a gate names only authored nodes", () => {
  const text = throughB("ask.output", "", "b.output");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask", "b"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  b:"), lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on b (ask.input.proj)", "b depends on ask (b.input.a)"]);
});

test("an ordering cycle through a gate names only authored nodes", () => {
  const error = loadError(() => assembleText(throughB("ask.output", "    depends_on: [b]\n", "input.proj")));
  expect(error.message).toBe(cycleMessage(["ask", "b"]));
  expect(error.diagnostic.notes).toEqual(["ask depends on b (ordering)", "b depends on ask (b.input.a)"]);
});

test("a cycle through a direct composer reader keeps the id", () => {
  const error = loadError(() => assembleText(throughB("ask__compose.output", "", "b.output")));
  expect(error.message).toBe(cycleMessage(["ask__compose", "b"]));
  expect(error.diagnostic.notes).toContain("b depends on ask__compose (b.input.a)");
  expect(error.diagnostic.notes).toContain(GENERATED);
});

test("a composer read outside the cycle still names only the gate", () => {
  const text = `${SELF_READING}\
  log:
    kind: code
    input:
      q: \${ask__compose.output}
    output: int
    code: return inputs["q"].length
output: {a: '\${ask.output}', q: '\${ask__compose.output}'}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on ask (ask.input.proj)"]);
});

test("two self-reading gates each name only the gate", () => {
  const text = `${SELF_READING}\
  ask2:
    kind: human_input
    input: {proj: '\${ask2.output}'}
    adaptive_questions: {prompt: "Q \${proj}"}
output: {a: '\${ask.output}', b: '\${ask2.output}'}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask", "ask2"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:"), lineStarting(text, "  ask2:")]);
  expect(error.diagnostic.notes).toEqual([
    "ask depends on ask (ask.input.proj)",
    "ask2 depends on ask2 (ask2.input.proj)",
  ]);
});

test("a gate reading its own composer keeps the id", () => {
  // The composer copies the gate's `x`, so it reads itself: the cycle is its self-edge.
  const text = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}', x: '\${ask__compose.output}'}
    adaptive_questions: {prompt: "Q \${proj} \${x}"}
output: \${ask.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask__compose"]));
  expect(error.diagnostic.notes).toEqual(["ask__compose depends on ask__compose (ask__compose.input.x)", GENERATED]);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-agent/engine_adaptive_composer_errors__api_key_in_the_composer_llm_config", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__unknown_key_in_the_composer_llm_config", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__valid_llm_config_loads", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__composer_reader_cycle_keeps_the_id", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__cycle_through_a_gate_names_only_authored_nodes", call: assembled },
])("assemble: $id", runCase);

test.each([
  [
    "composer_reader_cycle_keeps_the_id",
    [
      "ask__compose depends on b (ask__compose.input.proj)",
      GENERATED,
      "b depends on ask__compose (b.input.a)",
    ],
  ],
  [
    "cycle_through_a_gate_names_only_authored_nodes",
    ["ask depends on b (ask.input.proj)", "b depends on ask (b.input.a)"],
  ],
])("cycle case %s carries its notes and both frames", (name, notes) => {
  const text = caseText(`load-agent/engine_adaptive_composer_errors__${name}`).text;
  const error = loadError(() => assembleText(text));
  expect(error.diagnostic.frames.map(({ range }) => [range.startLine, range.startCol, range.endLine, range.endCol])).toEqual([
    [5, 3, 5, 4],
    [9, 3, 9, 6],
  ]);
  expect(error.diagnostic.notes).toEqual(notes);
});
