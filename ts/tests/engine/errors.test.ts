/**
 * Negative fixtures and their located source diagnostics: a port of `tests/engine/test_errors.py`.
 * The rows the parse phase decides run through `parseFile`, `flowNodesCursor` and `parseNodes`
 * with the registered kinds.
 *
 * The five parse gallery rows read conformance cases holding Python's `tests/seeds/errors/` flows:
 * `load-core/seed_e09_unknown_top_level_key`, `seed_e10_unknown_node_kind` and
 * `seed_e42_removed_model_kind`, and `load-leaves/seed_e11_field_not_allowed` and
 * `seed_e12_missing_required_field`. The `env` collision row reads
 * `load-core/seed_e33_input_named_like_field`.
 *
 * `test_e02_cycle_located` assembles conformance case `load-core/seed_e02_cycle` with `assembled`
 * (`assemble` as the loader calls it), with `full` (every frame and the notes). Changed: its
 * message writes the node list in the text form (`["a","b"]`, rule `messages`).
 *
 * `test_e06_cross_flow_type_mismatch_is_loud` assembles conformance case
 * `load-core/seed_e06_type_mismatch_ref` (`assembledWith`), its callee `research-one` (seed 03)
 * built by hand (`handChild`) where Python loads the seed with a child resolver; it pins the
 * whole message, where Python matches fragments.
 *
 * `test_e04_exhaustiveness_mechanism_loud` and `test_e04_else_satisfies_coverage` assemble their
 * inline flows (`assembleText`), where Python loads them; the seed `e17` case
 * (`load-case/seed_e17_case_nonexhaustive`, the same mechanism) is pinned with `assembled`. Changed:
 * its message writes the uncovered tags in the text form (`["flat"]`, rule `messages`).
 *
 * The load rows assemble their seed's conformance case with `assembleText` where Python calls
 * `load_flow`, a `defs:` entry a hand-built child (`handChild`) with the def's signature:
 * `test_e01_dangling_ref_located`, `test_e03_unknown_field_located`,
 * `test_negative_fixtures_are_located`, the e13, e14, e15 and e17 rows of `test_compile_error_gallery`,
 * `test_invalid_authored_default_carries_input_range`, `test_bad_input_ref_locates_the_offending_entry`,
 * `test_bad_map_per_element_input_ref_locates_the_offending_entry`,
 * `test_map_over_ref_locates_the_over_line`, the `load_flow` half of
 * `test_input_named_like_a_field_does_not_collide`, `test_llm_config_error_carries_the_field_range`,
 * `test_agent_controls_field_is_retired_and_located`, `test_agent_mode_field_is_removed_and_located`,
 * `test_synth_node_locates_at_its_call_directive_line` and `test_case_when_ref_locates_at_when_line`.
 * The reference cases are pinned at the end: `load-core/seed_e01_undeclared_ref`,
 * `load-types/seed_e03_unknown_field`, `load-core/seed_e14_unknown_system_ambient`,
 * `load-agent/seed_e15_prompt_undeclared_input`, `load-core/seed_e33_input_named_like_field`,
 * `load-map-call/seed_e35_bad_map_over_ref`,
 * `load-map-call/seed_e36_bad_map_input_ref`, `load-map-call/seed_e37_synth_node_inherits_host_ranges`
 * and `load-case/seed_e38_case_when_ref_locates_node_header`. Changed: the invalid-default row's
 * CODE body is inline JavaScript, where Python names a fixture function.
 *
 * Later (T4-9): the e30, e31 and e32 rows of `test_compile_error_gallery` and
 * `test_inline_code_construction_errors_carry_ranges` (compiling the CODE body; the TS case
 * `load-leaves/seed_e31_inline_no_return` loads).
 * Later (T3-22), as they need the loader itself: `test_single_node_def_model_kind_is_removed`
 * (the loader parses the `defs:` entry) and the e16 row of `test_compile_error_gallery` (the loader
 * reports the typedef error as `bad typedefs: ...`).
 * Later (T4-8): `test_runtime_error_gallery`, `test_e23_unknown_tool_fails_at_run_start`,
 * `test_flow_input_type_failure_origin_resolves_to_declaration`,
 * `test_false_boundary_assert_origin_resolves_to_assertion`,
 * `test_duplicate_asserts_caret_at_first_occurrence`, `test_structured_output_origin_is_exact`.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { siteKey } from "../../src/compose/nodes/base.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  caseText,
  type ChildFlows,
  handChild,
  loadError,
  nodesOf,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

/** The flow text of conformance case `id`. */
function seed(id: string): () => string {
  return () => caseText(id).text;
}

/** A hand-built child over `inputs` (a YAML `input:` section) returning one `type` producer. */
const child = (inputs: string, type: string) =>
  handChild({
    inputs: parametersFromYaml(inputs, new Map()),
    producers: { p: new ValueContract(typeFor(type, new Map())) },
    output: "${p.output}",
  });

/** The `defs:` entries of the MAP and CALL seeds, by case. */
const SEED_DEFS: Readonly<Record<string, ChildFlows>> = {
  "load-map-call/seed_e35_bad_map_over_ref": { body: child("input:\n  item: !str\n", "int") },
  "load-map-call/seed_e36_bad_map_input_ref": {
    body: child("input:\n  item: !str\n  n: !int\n  extra: !int\n", "int"),
  },
  "load-map-call/seed_e37_synth_node_inherits_host_ranges": { enrich: child("input:\n  topic: !str\n", "list(str)") },
};

/** Return the `LoadError` assembling conformance case `id`'s flow throws. */
function assembleError(id: string): LoadError {
  const { text, label } = caseText(id);
  return loadError(() => assembleText(text, label, { children: SEED_DEFS[id] ?? {} }));
}

/** Return the sole frame's range of a single-site error. */
function onlyRange(error: LoadError) {
  expect(error.diagnostic.frames).toHaveLength(1);
  return error.diagnostic.frames[0]!.range;
}

test.each([
  ["e09", seed("load-core/seed_e09_unknown_top_level_key"), "unknown top-level key 'bogus'", 14],
  // Per-kind parse anchors these at the precise offending field (not the node body):
  ["e10", seed("load-core/seed_e10_unknown_node_kind"), "unknown kind 'llm'", 8], // the value
  ["e42", seed("load-core/seed_e42_removed_model_kind"), "kind 'model' was removed", 8], // the line
  ["e11", seed("load-leaves/seed_e11_field_not_allowed"), "field 'prompt' is not allowed", 13],
  ["e12", seed("load-leaves/seed_e12_missing_required_field"), "missing required field 'code'", 8],
] as const)("compile error gallery: %s", (_, flow, substr, line) => {
  const error = loadError(() => nodesOf(flow()));

  expect(error.message).toContain(substr);
  expect(error.diagnostic.frames[0]!.range.startLine).toBe(line);
});

test("input named like a field does not collide", () => {
  // e33 has an `env:` field on L32 and an input named `env` on L35. `siteKey` keys keep
  // separate source ranges for the two.
  const flow = caseText("load-core/seed_e33_input_named_like_field").text;
  const document = SourceDocument.parse(flow.replace("${nope.output}", '"lit"'));
  const fr = parseNodes(document.root.child("nodes")).get("calc")!.fieldRanges;

  expect(fr.get(siteKey("field", "env"))!.startLine).toBe(32); // the `env:` field's VALUE
  expect(fr.get(siteKey("input", "env"))!.startLine).toBe(35); // the input NAMED `env`
  expect(fr.get(siteKey("field", "env"))).not.toEqual(fr.get(siteKey("input", "env")));

  const error = assembleError("load-core/seed_e33_input_named_like_field");
  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("'env'");
  // Per-entry resolution lands on the offending `env:` input (L35), never the `env:` field (L32).
  expect(onlyRange(error).startLine).toBe(35);
});

test("e01: a dangling ref is located", () => {
  const error = assembleError("load-core/seed_e01_undeclared_ref");

  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("scor"); // names the dangling ref
  // `output: ${scor.output}` is line 19 of the fixture (the offending line).
  expect(onlyRange(error).startLine).toBe(19);
});

test("e03: an unknown field is located", () => {
  const error = assembleError("load-types/seed_e03_unknown_field");

  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("confidence"); // names the missing record field
  // `output: ${score.output.confidence}` is line 21 of the fixture.
  expect(onlyRange(error).startLine).toBe(21);
});

test.each([
  ["load-core/seed_e01_undeclared_ref", 19],
  ["load-core/seed_e02_cycle", 12],
  ["load-types/seed_e03_unknown_field", 21],
] as const)("the negative fixture %s is located", (id, line) => {
  expect(assembleError(id).diagnostic.frames[0]!.range.startLine).toBe(line);
});

test.each([
  ["e13", "load-types/seed_e13_bad_type_expression", "bad type expression 'list('", 11],
  ["e14", "load-core/seed_e14_unknown_system_ambient", "bogus", 11],
  ["e15", "load-agent/seed_e15_prompt_undeclared_input", "is not a declared input", 12],
  ["e17", "load-case/seed_e17_case_nonexhaustive", "non-exhaustive", 13],
] as const)("compile error gallery (assembled): %s", (_, id, substr, line) => {
  const error = assembleError(id);

  expect(error.message).toContain(substr);
  expect(error.diagnostic.frames[0]!.range.startLine).toBe(line);
});

test("an invalid authored default carries the input's range", () => {
  const source = `id: invalid-default
name: invalid_default
input:
  value: !int null
nodes:
  echo:
    kind: code
    code: |
      return inputs.x
    input:
      x: \${input.value}
    output: int
output: \${echo.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toContain("'value': `null` is not a valid int");
  expect(onlyRange(error).startLine).toBe(4);
});

test("a bad input ref locates the offending entry", () => {
  // e14's `input:` block spans lines 10-12, but only `rid:` on line 11 is bad (`topic:` on 10 is
  // valid; 12 is `output: str`). The range must box line 11 alone. e14 is the only seed with a bad
  // node-input ref, and its bad entry is the second, so the lookup is per entry, not "first wins".
  const error = assembleError("load-core/seed_e14_unknown_system_ambient");

  expect(error.message).toContain("bogus");
  const range = onlyRange(error);
  expect([range.startLine, range.endLine]).toEqual([11, 11]);
});

test("a bad MAP per-element input ref locates the offending entry", () => {
  // e36's MAP per-element branch is its own code path: it reads the node wiring, allows the
  // `${item}` scope and skips the reserved `over` key. `extra:` is on line 33; `input:` spans 31-33.
  const error = assembleError("load-map-call/seed_e36_bad_map_input_ref");

  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("map input 'extra'");
  const range = onlyRange(error);
  expect([range.startLine, range.endLine]).toEqual([33, 33]);
});

test("a MAP over: ref locates the over: line", () => {
  // e35 has a dangling ref in MAP's reserved `over:` key on line 26; its range boxes `over:`
  // itself, not the `input:` block on lines 27-28.
  const error = assembleError("load-map-call/seed_e35_bad_map_over_ref");

  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("map over"); // names the reserved key, not an input
  const range = onlyRange(error);
  expect([range.startLine, range.endLine]).toEqual([26, 26]);
});

test.each([
  ["e39", "load-agent/seed_e39_llm_config_no_range", ["llm_config", "provider"], 27],
  ["e40", "load-agent/seed_e40_agent_controls_no_range", ["field 'controls' is not allowed"], 13],
  ["e43", "load-agent/seed_e43_agent_mode_removed", ["`mode:` was removed", "will now be used"], 13],
] as const)("an AGENT field error is located: %s", (_, id, substrs, line) => {
  // e39: the first offending `llm_config:` entry (L27), not the node header; e40: the retired
  // `controls:` field at its value; e43: a stale `mode:` at its key line.
  const error = assembleError(id);

  for (const substr of substrs) expect(error.message).toContain(substr);
  expect(onlyRange(error).startLine).toBe(line);
});

test("a synth node locates at its call directive's line", () => {
  // e37's inline `call(...)` on L30 desugars to `__call_0`; directive-owned provenance locates the
  // bad `topic` binding at L30, not the host node's like-named input on L29.
  const error = assembleError("load-map-call/seed_e37_synth_node_inherits_host_ranges");

  expect(error.message).toContain("unresolved references");
  expect(error.message).toContain("node 'summary' input 'news' call(enrich) argument 'topic' from:");
  expect(error.message).not.toContain("__call_");
  expect(onlyRange(error).startLine).toBe(30);
});

test("a case when ref locates at the when line", () => {
  // e38's dangling ref in `cases[1].when` (L36) hoists to `__r1`; predicate-owned provenance keeps
  // the caret and the author-facing label on `cases[1].when`.
  const error = assembleError("load-case/seed_e38_case_when_ref_locates_node_header");

  expect(error.message).toContain("unresolved references");
  expect(error.message).not.toContain("'__r1'");
  expect(error.message).toContain("cases[1].when");
  expect(onlyRange(error).startLine).toBe(36);
});

test.each<CaseRow>([
  { id: "load-core/seed_e02_cycle", call: assembled, full: true },
  { id: "load-case/seed_e17_case_nonexhaustive", call: assembled },
  { id: "load-core/seed_e01_undeclared_ref", call: assembled, full: true },
  { id: "load-types/seed_e03_unknown_field", call: assembled, full: true },
  { id: "load-core/seed_e14_unknown_system_ambient", call: assembled, full: true },
  { id: "load-agent/seed_e15_prompt_undeclared_input", call: assembled, full: true },
  { id: "load-core/seed_e33_input_named_like_field", call: assembled, full: true },
  ...[
    "load-map-call/seed_e35_bad_map_over_ref",
    "load-map-call/seed_e36_bad_map_input_ref",
    "load-map-call/seed_e37_synth_node_inherits_host_ranges",
  ].map((id) => ({ id, call: assembledWith(SEED_DEFS[id]!), full: true })),
  { id: "load-case/seed_e38_case_when_ref_locates_node_header", call: assembled, full: true },
])("assemble: $id", runCase);

// ---------- e06: a cross-flow type mismatch at a `call` boundary ----------

const E06 = "load-core/seed_e06_type_mismatch_ref";

/** Seed 03 (`research-one`, the e06 case's file), built by hand: its parameters, producers and output. */
function researchOne(): ComposedFlow {
  const { text } = caseText(E06, { file: "research-one.yaml" });
  const facts = typeFor("Facts", readTypedefs({ Facts: { values: "list(float)", news: "list(str)" } }));
  return handChild({
    inputs: parametersFromYaml(text, new Map()),
    producers: { fetch: new ValueContract(facts), synth: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: { report: "${synth.output}", asof: "${input.as_of}" },
  });
}

test("e06 cross-flow type mismatch is loud", () => {
  // A float output bound to the callee's str input names the binding and both types.
  const { text, label } = caseText(E06);
  const error = loadError(() => assembleText(text, label, { children: { "research-one": researchOne() } }));

  expect(error.message).toBe("call node 'call': binding 'topic' — child expects 'str', source is 'float'");
});

test(`assemble: ${E06}`, () => {
  runCase({ id: E06, call: assembledWith({ "research-one": researchOne() }) });
});

// ---------- e04: case exhaustiveness ----------

// classify gives a Literal(pro, con, flat); route covers only pro and con with no `else:`, so the
// `flat` tag is uncovered.
const E04_NONEXHAUSTIVE = `
id: e04-constructed
name: e04_constructed

input:
  topic: !str

nodes:
  classify:
    kind: agent
    input:
      topic: \${input.topic}
    output: Literal(pro, con, flat)
    prompt: "Classify \${topic}."
  route:
    kind: case
    on: \${classify.output}
    cases:
      - when: pro
        then: pro_note
      - when: con
        then: con_note
  pro_note:
    kind: agent
    input:
      topic: \${input.topic}
    output: str
    prompt: "Pro note for \${topic}."
  con_note:
    kind: agent
    input:
      topic: \${input.topic}
    output: str
    prompt: "Con note for \${topic}."

output: \${pro_note.output | con_note.output}
`;

// The same flow, with an `else:` that covers the remaining tag.
const E04_WITH_ELSE = E04_NONEXHAUSTIVE.replace(
  "        then: con_note\n",
  "        then: con_note\n    else: flat_note\n",
).replace(
  "  pro_note:",
  "  flat_note:\n" +
    "    kind: agent\n" +
    "    input:\n" +
    "      topic: ${input.topic}\n" +
    "    output: str\n" +
    '    prompt: "Flat note for ${topic}."\n' +
    "  pro_note:",
);

test("e04: the exhaustiveness mechanism is loud", () => {
  const message = loadError(() => assembleText(E04_NONEXHAUSTIVE)).message;

  expect(message).toContain("non-exhaustive");
  expect(message).toContain("flat"); // names the uncovered enum tag
});

test("e04: an else satisfies coverage", () => {
  assembleText(E04_WITH_ELSE);
});

