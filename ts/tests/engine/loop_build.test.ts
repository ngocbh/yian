/**
 * `LoopDescriptor.build`: the driver, the counts, the carried record's field-name contract, the
 * predicate's names, checks and field reads, and the never-ending check, a port of rows of
 * `tests/engine/test_loop_build.py`.
 *
 * Ported: `test_predicate_loop_without_max_builds_unlimited` (3),
 * `test_predicate_loop_with_max_keeps_its_limit`, `test_times_with_null_max_builds`,
 * `test_good_loop_bakes_loopnode`, `test_body_may_return_a_subset_of_carried_fields`,
 * `test_body_output_field_outside_carried_record_is_rejected`, `test_one_field_body_record_builds`,
 * the build halves of `test_one_field_body_record_returns_only_that_carried_field` and
 * `test_value_body_returns_the_referenced_record_beside_a_carried_result`,
 * `test_body_returning_no_carried_field_is_rejected` (2),
 * `test_total_loop_keeps_body_contract_and_no_returned_fields`, the contract half of
 * `test_merged_contract_types_omitted_field_from_its_seed`,
 * `test_merged_contract_keeps_literal_seed_opaque`, `test_partial_body_with_opaque_returned_field`,
 * the contract of `test_loop_accepts_proven_null_for_nullable_carried_field`,
 * `test_multi_segment_template_output_types_as_str`, `test_until_loop_builds`,
 * `test_times_loop_builds`, `test_exactly_one_predicate_required` (2),
 * `test_times_with_max_is_rejected`, `test_times_count_must_be_positive_int`,
 * `test_max_below_one_is_rejected`, `test_non_int_max_is_rejected` (3),
 * `test_while_predicate_unknown_name_is_rejected`, `test_while_predicate_all_spellings_build` (2),
 * `test_while_predicate_unknown_name_rejected_in_all_spellings` (2),
 * `test_loop_whose_condition_reads_only_unreturned_fields_is_a_load_error`,
 * `test_while_loop_whose_condition_reads_only_unreturned_fields_is_a_load_error`,
 * `test_never_ending_error_names_every_unreturned_field_read`,
 * `test_never_ending_check_reads_every_condition_spelling` (4),
 * `test_null_max_still_gets_the_never_ending_error`, `test_dotted_condition_ref_names_the_carried_field`,
 * `test_never_ending_error_does_not_stack_on_other_loop_errors` (5),
 * `test_never_changing_condition_with_max_loads`, `test_condition_reading_one_returned_field_loads`,
 * `test_unknown_returned_fields_get_only_the_record_error`; the build halves of
 * `test_predicate_nested_field_read_builds` (2),
 * `test_predicate_tagged_member_field_error_points_at_isinstance` and
 * `test_predicate_isinstance_check_on_an_object_field_checks_reads_on_the_type`;
 * `test_predicate_unknown_nested_field_is_rejected` (3),
 * `test_predicate_field_read_on_scalar_is_rejected`,
 * `test_predicate_unknown_field_on_omitted_typed_seed_is_rejected`,
 * `test_predicate_field_on_omitted_literal_seed_is_not_checked`,
 * `test_predicate_unguarded_member_field_read_is_rejected` (3),
 * `test_predicate_read_of_a_field_the_checked_member_lacks_is_rejected` (2),
 * `test_predicate_read_after_a_check_the_type_always_decides_is_checked_unnarrowed`,
 * `test_predicate_check_that_can_never_be_true_is_rejected`,
 * `test_predicate_read_after_a_reported_check_is_not_checked` (2),
 * `test_predicate_tag_comparison_with_a_value_that_is_no_tag_is_rejected`,
 * `test_predicate_reports_only_the_impossible_check` (4),
 * `test_predicate_read_past_a_check_the_checks_before_it_decide_is_checked_under_them`,
 * `test_predicate_read_after_a_value_that_is_no_tag_is_not_checked`,
 * `test_predicate_impossible_on_the_seed_and_the_body_is_rejected` (3),
 * `test_predicate_isinstance_of_a_partly_shared_union_rejects_a_member_only_field` and
 * `test_predicate_isinstance_check_with_an_unknown_type_reports_the_type`. Through
 * `assembleText`: `test_inline_loop_body_may_return_a_subset`,
 * `test_inline_body_loop_whose_condition_never_changes_is_a_load_error`,
 * `test_loop_seeded_from_partial_loop_sees_merged_type` and
 * `test_omitted_field_seed_cycle_is_located`.
 *
 * Changed: a row whose Python flow calls a `defs:` body parses the loop node (`nodesOf`) and
 * builds it over a `BuildContext` whose resolver returns the body built by hand (`handChild`: the
 * def's parameters read from its `input:` text, its typed nodes as producers with their output
 * contracts), where Python loads the flow (def bodies load in T3-22). Such a context has no
 * binding-contract reader: the merged-contract rows give it a test-local `bindingContractReader`
 * typing the omitted fields' seeds and recording what the LOOP asks; without one an untagged seed
 * stays opaque, as Python's literal seed does, so a `while:` reading it gets no seed type: none of
 * its checks is reported impossible and none narrows a read, and a plain field read is walked on
 * the carried record. `test_loop_seeded_from_partial_loop_sees_merged_type` and
 * `test_omitted_field_seed_cycle_is_located` assemble the flow (`assembleText`, the def built by
 * hand), so the loader's reader types the omitted seeds, building the seeding loop on demand. The
 * rows whose Python flow has an inline `nodes:` body assemble that flow, its inline CODE written in
 * TS as the harvested cases write it. The typed-seed rows of
 * `test_predicate_impossible_on_the_seed_and_the_body_is_rejected` assemble the flow
 * (`assembleText` with the def built by hand), since a tagged seed's declared type comes from
 * `assemble`; parsing alone drops the tag. The error rows pin the whole message, where Python
 * matches fragments; lists of names are JSON text (`["exited","n"]`) and values the text form
 * (`(got "5")`, `(got true)`, a tag comparison's `with "dnoe"` and `did you mean "done"?`).
 *
 * TS-only: the counts are YAML ints (`bigint`) converted to numbers: `times: 3` gives `3`, a
 * `max:` or `times:` past `Number.MAX_SAFE_INTEGER` is refused with the at-most message (the
 * largest safe integer builds), and a float `max: 2.0` is refused as `(got 2)`; `times:` is
 * refused below 1 and as a float, a bool or text, as `max:` is; `times:` beside `while:` is a
 * second driver; a body input that isn't a carried field is refused; a count error and a
 * field-name error are joined at the node, as are a count error and a predicate that doesn't
 * parse; and a hand-constructed descriptor's `number` count is refused.
 *
 * Pinned at the end (`assembledWith`, each def built by hand): the
 * `load-loop/engine_loop_build__*` cases whose error is a driver, count, field-name, predicate-name,
 * predicate field-read or never-ending error; then (`assembled`, the body inline)
 * `load-loop/engine_loop_build__{inline_body_condition_that_never_changes_is_rejected,
 * predicate_isinstance_of_a_disjoint_type_is_rejected,
 * predicate_isinstance_on_an_object_checks_reads,
 * predicate_isinstance_with_an_unknown_type,
 * predicate_isinstance_ruled_out_by_an_earlier_check,
 * predicate_negated_isinstance_of_the_declared_type_is_rejected,
 * predicate_read_of_a_field_the_checked_member_lacks,predicate_tag_comparison_with_a_non_tag,
 * predicate_tagged_member_field_error_points_at_isinstance,
 * predicate_unguarded_member_field_read_is_rejected,
 * predicate_unknown_field_on_omitted_typed_seed_is_rejected}`, and (`assembledWith`, the def built
 * by hand) `load-loop/engine_loop_build__omitted_field_seed_cycle_is_located`.
 *
 * Later (T3-18 PR 3):
 * `test_partial_loop_contract_rejects_unknown_downstream_field_at_load` and the downstream half of
 * `test_merged_contract_types_omitted_field_from_its_seed`. Later (T3-19):
 * `test_returned_field_type_mismatch_still_rejected_in_partial_body`,
 * `test_body_output_types_must_equal_carried`,
 * `test_opaque_body_field_does_not_erase_proven_sibling_type_mismatch`,
 * `test_constant_seed_that_cannot_fill_the_body_input_is_a_load_error`,
 * `test_seed_is_checked_even_when_a_while_loop_never_runs`,
 * `test_text_seed_into_int_input_of_a_full_loop_is_a_load_error`,
 * `test_reference_or_typed_seed_that_cannot_fill_the_body_input_is_a_load_error` and
 * `test_reference_seed_is_checked_even_when_a_while_loop_never_runs`. Later (T4): the run halves
 * of the rows above, `test_predicate_field_on_omitted_typed_seed_builds`,
 * `test_predicate_member_field_read_guarded_by_isinstance_builds`,
 * `test_predicate_while_read_guarded_through_or_and_not_builds`,
 * `test_predicate_tag_comparison_narrows_a_read`, `test_predicate_while_is_checked_against_the_seed_too`,
 * `test_predicate_isinstance_of_a_partly_shared_union_reads_the_shared_fields`,
 * `test_constant_seed_that_fits_the_body_input_loads_and_runs`,
 * `test_inline_body_constant_seed_fits_its_generated_parameter`,
 * `test_reference_or_typed_seed_that_fits_the_body_input_loads_and_runs`,
 * `test_carried_field_the_body_does_not_read_is_not_checked`,
 * `test_inline_body_reference_seed_fits_its_generated_parameter`,
 * `test_while_null_check_lets_an_optional_seed_fill_a_strict_body_input` and
 * `test_loop_seed_reads_the_case_narrowed_type`.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { type BindingContractReader, BuildContext, type BuildResult } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { isExactly, type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  nodesOf,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));
const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/** Replace `old` in a fixture, failing if it is absent (a stale needle). */
function swap(text: string, old: string, replacement: string): string {
  if (!text.includes(old)) throw new Error(`stale needle: ${old}`);
  return text.replace(old, replacement);
}

/**
 * The contract of a record of `fields` (field name to type text), as a typed node declares it;
 * its types may name `typedefs`.
 */
function record(
  fields: Readonly<Record<string, string>>,
  typedefs: Readonly<Record<string, unknown>> = {},
): ValueContract {
  return new ValueContract(typeFor("Rec", readTypedefs({ ...typedefs, Rec: fields })));
}

/** The parameters a def declares in `input` (the lines under its `input:` key). */
function params(input: string, registry: TypeRegistry = new Map()) {
  return parametersFromYaml(`input:\n${input}`, registry);
}

/**
 * Parse loop `nodeId` of `text` and build it over a resolver returning `child`, with `registry`'s
 * typedefs (default none).
 */
function build(
  text: string,
  child: ComposedFlow,
  options: { nodeId?: string; bindingContractReader?: BindingContractReader; registry?: TypeRegistry } = {},
): BuildResult {
  const { nodeId = "chat_loop", bindingContractReader = null, registry = new Map() } = options;
  const desc = nodesOf(text).get(nodeId)!;
  return desc.build(
    new BuildContext({
      registry,
      resolver: () => child,
      diagnostics: new LoadDiagnostics("<flow>", text),
      bindingContractReader,
    }),
  );
}

/** Build as `build` does and return the built LOOP node. */
function loopOf(text: string, child: ComposedFlow, options: Parameters<typeof build>[2] = {}): LoopNode {
  const node = build(text, child, options).node;
  expect(node).toBeInstanceOf(LoopNode);
  return node as LoopNode;
}

/** The ordered field names of an exact or partial record contract. */
function fieldNames(contract: OutputContract | null): string[] {
  if (isExactly(contract, PartialRecordContract)) return [...contract.names];
  return [...(contract as ValueContract).type.fields!.keys()];
}

/** One field's type of exact record contract `contract`. */
function fieldType(contract: OutputContract | null, name: string): Type {
  expect(isExactly(contract, ValueContract)).toBe(true);
  return (contract as ValueContract).type.fields!.get(name)!;
}

/** The `bump` def of `GOOD`: it reads `{n, exited}` and returns the whole carried record. */
function goodBody(stepOutput: Readonly<Record<string, string>> = { n: "int", exited: "bool" }): ComposedFlow {
  return handChild({
    inputs: params("  n: !int\n  exited: !bool\n"),
    producers: { step: record(stepOutput) },
    output: "${step.output}",
  });
}

// `chat_loop` carries {n, exited}, loops while `not exited`, capped at max: 5.
const GOOD = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
`;

const UNTIL_GOOD = swap(GOOD, "    while: not exited\n", "    until: ${exited}\n");
const TIMES_GOOD = swap(GOOD, "    while: not exited\n    max: 5\n", "    times: 3\n");
const NO_MAX = swap(GOOD, "    max: 5\n", "");

// ---------- drivers and counts ----------

test.each([
  ["while, no max", NO_MAX, "while"],
  ["until, no max", swap(UNTIL_GOOD, "    max: 5\n", ""), "until"],
  ["while, null max", swap(GOOD, "    max: 5\n", "    max: null\n"), "while"],
])("a predicate loop without max builds unlimited: %s", (_name, text, kind) => {
  const node = loopOf(text, goodBody());

  expect(node.predicateKind).toBe(kind as "while" | "until");
  expect(node.maxIters).toBeNull();
});

test("a predicate loop with max keeps its limit", () => {
  const node = loopOf(GOOD, goodBody());

  expect(node.predicateKind).toBe("while");
  expect(node.predicate).toBe("not exited");
  expect(node.maxIters).toBe(5);
});

test("times with a null max builds", () => {
  const node = loopOf(swap(TIMES_GOOD, "    times: 3\n", "    times: 3\n    max: null\n"), goodBody());

  expect(node.times).toBe(3);
  expect(node.maxIters).toBe(3);
});

test("an until loop builds", () => {
  const node = loopOf(UNTIL_GOOD, goodBody());

  expect(node.predicateKind).toBe("until");
  expect(node.predicate).toBe("${exited}");
});

test("a times loop builds with number counts", () => {
  const node = loopOf(TIMES_GOOD, goodBody());

  expect(node.predicateKind).toBe("times");
  expect(node.predicate).toBeNull();
  // YAML reads `3` as the bigint 3n; the node holds the number.
  expect(node.times).toBe(3);
  expect(node.maxIters).toBe(3);
});

test.each([
  ["both", swap(GOOD, "    while: not exited\n", "    while: not exited\n    until: ${exited}\n"), '["until","while"]'],
  ["neither", swap(GOOD, "    while: not exited\n", ""), "[]"],
  ["times and while", swap(GOOD, "    max: 5\n", "    times: 2\n"), '["times","while"]'],
])("exactly one driver is required: %s", (_name, text, got) => {
  const error = loadError(() => build(text, goodBody()));

  expect(error.message).toBe(`loop node 'chat_loop': exactly one of while:/until:/times: is required (got ${got})`);
});

test("times with max is rejected", () => {
  const error = loadError(() => build(swap(GOOD, "    while: not exited\n", "    times: 3\n"), goodBody()));

  expect(error.message).toBe("loop node 'chat_loop': max: is redundant with times: (times bounds the count)");
});

test.each([
  ["times: 0", "`times:` must be >= 1 (got 0)"],
  ["times: -2", "`times:` must be >= 1 (got -2)"],
  ["times: 2.5", "`times:` must be an integer (got 2.5)"],
  ["times: true", "`times:` must be an integer (got true)"],
  ['times: "3"', '`times:` must be an integer (got "3")'],
  ["times: 9007199254740992", "`times:` must be at most 9007199254740991 (got 9007199254740992)"],
])("a times count must be a positive safe integer: %s", (times, message) => {
  const error = loadError(() => build(swap(GOOD, "    while: not exited\n    max: 5\n", `    ${times}\n`), goodBody()));

  expect(error.message).toBe(`loop node 'chat_loop': ${message}`);
});

test("a max below one is rejected", () => {
  const error = loadError(() => build(swap(GOOD, "    max: 5\n", "    max: 0\n"), goodBody()));

  expect(error.message).toBe("loop node 'chat_loop': `max:` must be >= 1 (got 0)");
});

test.each([
  ['max: "5"', '(got "5")'],
  ["max: 2.5", "(got 2.5)"],
  ["max: true", "(got true)"],
  // A float 2 is spelled `2` in the text form.
  ["max: 2.0", "(got 2)"],
])("a non-int max is rejected: %s", (max, got) => {
  const error = loadError(() => build(swap(GOOD, "    max: 5\n", `    ${max}\n`), goodBody()));

  expect(error.message).toBe(`loop node 'chat_loop': \`max:\` must be an integer ${got}`);
});

test("a max at the largest safe integer builds", () => {
  const node = loopOf(swap(GOOD, "    max: 5\n", "    max: 9007199254740991\n"), goodBody());

  expect(node.maxIters).toBe(Number.MAX_SAFE_INTEGER);
});

test("a max past the largest safe integer is rejected", () => {
  const error = loadError(() => build(swap(GOOD, "    max: 5\n", "    max: 9007199254740992\n"), goodBody()));

  expect(error.message).toBe("loop node 'chat_loop': `max:` must be at most 9007199254740991 (got 9007199254740992)");
});

test("a constructed descriptor's number count is refused, as a YAML float is", () => {
  // Counts are YAML ints, so a descriptor built by hand passes a bigint (`1n`).
  const desc = new LoopDescriptor({ id: "lp", call: "bump", inputs: { n: 0n, exited: false }, times: 1 });

  const error = loadError(() => desc.build(new BuildContext({ registry: new Map(), resolver: () => goodBody() })));

  expect(error.message).toBe("loop node 'lp': `times:` must be an integer (got 1)");
});

// ---------- the carried record's field names ----------

test("a good loop bakes a LoopNode", () => {
  const child = goodBody();

  const result = build(GOOD, child);

  const node = result.node as LoopNode;
  expect(node).toBeInstanceOf(LoopNode);
  expect(node.flowId).toBe("bump");
  expect(node.child).toBe(child.compiled);
  expect(node.childInputs).toStrictEqual(child.input);
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["n", "exited"]);
  expect(result.wiring).toEqual(
    new Map<string, unknown>([
      ["n", 0n],
      ["exited", false],
    ]),
  );
  // The output contract carries the body codomain.
  expect(new Set(fieldNames(node.outputContract))).toEqual(new Set(["n", "exited"]));
});

test("a total loop keeps the body contract and no returned fields", () => {
  const node = loopOf(GOOD, goodBody());

  expect(node.returnedFields).toBeNull();
  expect(isExactly(node.outputContract, ValueContract)).toBe(true);
  expect(fieldNames(node.outputContract)).toEqual(["n", "exited"]);
  expect(fieldType(node.outputContract, "n").kind).toBe(ValueKind.INTEGER);
  expect(fieldType(node.outputContract, "exited").kind).toBe(ValueKind.BOOLEAN);
});

test("a body may return a subset of the carried fields", () => {
  // The body omits `exited`, which carries over.
  const node = loopOf(GOOD, goodBody({ n: "int" }));

  expect(node.returnedFields).toEqual(["n"]);
  expect(fieldNames(node.outputContract)).toEqual(["n", "exited"]);
});

// An inline body returning only `n` of the carried `{n, exited}`.
const INLINE_SUBSET = `
id: loop-inline-subset
name: loop_inline_subset
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: 0
      exited: false
    nodes:
      bump:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
    output: \${bump.output}
output: \${repeat.output}
`;

test("an inline loop body may return a subset", () => {
  const node = assembleText(INLINE_SUBSET).flow.compiled.nodes.get("repeat") as LoopNode;

  expect(node.returnedFields).toEqual(["n"]);
  expect(fieldNames(node.outputContract)).toEqual(["n", "exited"]);
});

/** The body of `OUTSIDE_CARRIED`: it reads `n` and returns `{n, extra}`. */
function outsideBody(): ComposedFlow {
  return handChild({
    inputs: params("  n: !int\n"),
    output: { n: "${input.n}", extra: "${input.n}" },
  });
}

test("a body output field outside the carried record is rejected", () => {
  const error = loadError(() => build(GOOD, outsideBody()));

  expect(error.message).toBe(
    'loop node \'chat_loop\': body output field(s) ["extra"] are not carried record fields ["exited","n"] ' +
      "(a body returns a subset of the carried record)",
  );
  // Located at the node.
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test("a one-field body record builds", () => {
  const body = handChild({ inputs: params("  n: !int\n"), output: { n: "${input.n}" } });

  const node = loopOf(GOOD, body);

  expect(node.returnedFields).toEqual(["n"]);
  expect(fieldNames(node.outputContract)).toEqual(["n", "exited"]);
});

test("a one-field body record returns only that carried field", () => {
  // The one output `state` holds a record `{n}`: only `state` is returned, not `n`.
  const text = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      state: init
    times: 2
`;
  const body = handChild({
    inputs: params("  n: !int\n"),
    producers: { step: record({ n: "int" }) },
    output: { state: "${step.output}" },
  });

  expect(loopOf(text, body).returnedFields).toEqual(["state"]);
});

const REJECTED_BODY = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
    times: 2
`;

test.each([
  ["scalar value", "${step.output}"],
  ["empty record", {}],
])("a body returning no carried field is rejected: %s", (_name, output) => {
  const body = handChild({ inputs: params("  n: !int\n"), producers: { step: INT }, output });

  const error = loadError(() => build(REJECTED_BODY, body));

  expect(error.message).toBe(
    "loop node 'chat_loop': body callable 'bump' must return a record with at least one carried field " +
      '(a subset of the carried record ["n"]); write `output: {field: ...}` or a whole reference to a node ' +
      "with a record output",
  );
});

test("a value body returns the referenced record beside a carried result", () => {
  const text = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      result: start
      n: 0
    times: 2
`;
  const body = handChild({
    inputs: params("  n: !int\n"),
    producers: { step: record({ n: "int" }) },
    output: "${step.output}",
  });

  expect(loopOf(text, body).returnedFields).toEqual(["n"]);
});

test("a body input outside the carried record is rejected", () => {
  const body = handChild({
    inputs: params("  n: !int\n  x: !int\n  a: !int\n"),
    producers: { step: record({ n: "int", exited: "bool" }) },
    output: "${step.output}",
  });

  const error = loadError(() => build(GOOD, body));

  expect(error.message).toBe(
    'loop node \'chat_loop\': body input(s) ["a","x"] are not carried record fields ["exited","n"] ' +
      "(the body reads a subset of the carried record)",
  );
});

test("driver and field-name errors are reported together at the node", () => {
  const error = loadError(() => build(swap(GOOD, "    max: 5\n", "    max: 0\n"), outsideBody()));

  expect(error.message).toBe(
    "loop node 'chat_loop': `max:` must be >= 1 (got 0)\n" +
      '  loop node \'chat_loop\': body output field(s) ["extra"] are not carried record fields ["exited","n"] ' +
      "(a body returns a subset of the carried record)",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

// ---------- the merged carried contract ----------

// The body reads `person`/`talking_to` but returns only `{exited, reply}`.
const CHAT = `id: l
name: l
nodes:
  chat:
    kind: loop
    call: turn
    input:
      person: \${input.person}
      talking_to: \${input.talking_to}
      exited: false
      reply: ""
    while: not exited
    max: 3
`;

test("the merged contract types an omitted field from its seed", () => {
  const registry = readTypedefs({ Person: { name: "str", handle: "str" } });
  const person = new ValueContract(typeFor("Person", registry));
  const body = handChild({
    inputs: params("  person: !Person\n  talking_to: !str\n  exited: !bool\n  reply: !str\n", registry),
    output: { exited: "${input.exited}", reply: "${input.talking_to}" },
  });
  const seeds = new Map<string, OutputContract>([
    ["chat.input.person", person],
    ["chat.input.talking_to", STR],
  ]);
  const read: string[] = [];
  const reader: BindingContractReader = (label, binding, range) => {
    read.push(`${label} = ${String(binding)} at line ${range?.startLine}`);
    return seeds.get(label) ?? null;
  };

  const node = loopOf(CHAT, body, { nodeId: "chat", bindingContractReader: reader });

  // The omitted fields' seeds are read for the merged contract, then the seed of `exited`, which
  // the `while:` reads (it is first evaluated on the seeds).
  expect(read).toEqual([
    "chat.input.person = ${input.person} at line 8",
    "chat.input.talking_to = ${input.talking_to} at line 9",
    "chat.input.exited = false at line 10",
  ]);
  expect(node.returnedFields).toEqual(["exited", "reply"]);
  expect(isExactly(node.outputContract, ValueContract)).toBe(true);
  expect(fieldNames(node.outputContract)).toEqual(["person", "talking_to", "exited", "reply"]);
  const personType = fieldType(node.outputContract, "person");
  expect(personType.kind).toBe(ValueKind.OBJECT);
  expect(new Set(personType.fields!.keys())).toEqual(new Set(["name", "handle"]));
  expect(fieldType(node.outputContract, "talking_to").kind).toBe(ValueKind.STRING);
  expect(fieldType(node.outputContract, "exited").kind).toBe(ValueKind.BOOLEAN);
});

test("the merged contract keeps a literal seed opaque", () => {
  const text = swap(GOOD, "      exited: false\n    while:", "      exited: false\n      keep: 3\n    while:");

  const node = loopOf(text, goodBody({ n: "int" }));

  expect(node.returnedFields).toEqual(["n"]);
  const contract = node.outputContract as PartialRecordContract;
  expect(contract).toBeInstanceOf(PartialRecordContract);
  expect(fieldNames(contract)).toEqual(["n", "exited", "keep"]);
  const fields = new Map(contract.fields);
  expect(fields.get("keep")).toBeNull();
  expect(fields.get("n")!.type.kind).toBe(ValueKind.INTEGER);
});

// `second` seeds `person` and `talking_to` from `first`, a partial loop whose `turn` def returns
// only `exited` and `reply`; `second` is written first, so its seeds build `first` on demand.
const CHAINED = `id: loop-chained
name: loop_chained
typedefs:
  Person:
    name: str
    handle: str
input:
  person: !Person
  talking_to: !str
nodes:
  second:
    kind: loop
    call: turn
    input:
      person: \${first.output.person}
      talking_to: \${first.output.talking_to}
      exited: false
      reply: ""
    times: 1
  first:
    kind: loop
    call: turn
    input:
      person: \${input.person}
      talking_to: \${input.talking_to}
      exited: false
      reply: ""
    times: 1
output: \${second.output.person.name}
`;

test("a loop seeded from a partial loop sees its merged type", () => {
  const registry = readTypedefs({ Person: { name: "str", handle: "str" } });
  const turn = handChild({
    inputs: params("  person: !Person\n  talking_to: !str\n  exited: !bool\n  reply: !str\n", registry),
    output: { exited: "${input.exited}", reply: "${input.talking_to}" },
  });

  const { flow } = assembleText(CHAINED, "flow.yaml", { children: { turn } });

  const person = fieldType(flow.compiled.nodes.get("second")!.outputContract, "person");
  expect(person.kind).toBe(ValueKind.OBJECT);
  expect(new Set(person.fields!.keys())).toEqual(new Set(["name", "handle"]));
});

// `first` and `second` each omit `keep` and seed it from the other's output, so resolving either
// merged contract needs the other built first.
const SEED_CYCLE = `id: loop-seed-cycle
name: loop_seed_cycle
nodes:
  first:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
      keep: \${second.output.keep}
    times: 1
  second:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
      keep: \${first.output.keep}
    times: 1
output: \${second.output}
`;

test("an omitted field's seed cycle is located", () => {
  const error = loadError(() => assembleText(SEED_CYCLE, "flow.yaml", { children: { bump: goodBody({ n: "int" }) } }));

  expect(error.message).toBe("node build dependency cycle: first.input.keep -> second.input.keep -> first");
  // The binding that closes the cycle carries the caret, not the coarse `input:` span.
  const closingLine = SEED_CYCLE.split("\n").indexOf("      keep: ${first.output.keep}") + 1;
  const range = error.diagnostic.frames.at(-1)!.range;
  expect([range.startLine, range.endLine]).toEqual([closingLine, closingLine]);
});

test("a partial body with an opaque returned field", () => {
  // `step` declares no output, so `n` projects an untyped producer value.
  const text = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 1
      note: \${input.note0}
      keep: \${input.k0}
    times: 1
`;
  const body = handChild({
    inputs: params("  n: !int\n  note: !str\n"),
    producers: { step: null },
    output: { n: "${step.output.n}", note: "${input.note}" },
  });
  const reader: BindingContractReader = (label) => (label === "chat_loop.input.keep" ? INT : null);

  const node = loopOf(text, body, { bindingContractReader: reader });

  expect(node.returnedFields).toEqual(["n", "note"]);
  const contract = node.outputContract as PartialRecordContract;
  expect(contract).toBeInstanceOf(PartialRecordContract);
  expect(fieldNames(contract)).toEqual(["n", "note", "keep"]);
  const fields = new Map(contract.fields);
  expect(fields.get("n")).toBeNull();
  expect(fields.get("note")!.type.kind).toBe(ValueKind.STRING);
  expect(fields.get("keep")!.type.kind).toBe(ValueKind.INTEGER);
});

test("a loop keeps a nullable carried field's contract", () => {
  const text = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: keep
    input:
      maybe: \${null}
      done: false
    times: 1
`;
  const body = handChild({
    inputs: params("  maybe: !Optional(str)\n  done: !bool\n"),
    output: { maybe: "${input.maybe}", done: "${input.done}" },
  });

  const maybe = fieldType(loopOf(text, body).outputContract, "maybe");

  expect(maybe.kind).toBe(ValueKind.STRING);
  expect(maybe.nullable).toBe(true);
});

test("a multi-segment template output types as str", () => {
  const text = `id: l
name: l
nodes:
  chat_loop:
    kind: loop
    call: turn
    input:
      transcript: \${input.opening}
      exited: false
    while: not exited
    max: 5
`;
  const body = handChild({
    inputs: params("  transcript: !str\n  exited: !bool\n"),
    producers: { step: record({ n: "int", exited: "bool" }) },
    output: { transcript: "${input.transcript}\nturn: ${step.output.n}", exited: "${step.output.exited}" },
  });

  const node = loopOf(text, body);

  expect(fieldType(node.outputContract, "transcript").kind).toBe(ValueKind.STRING);
});

// ---------- the predicate's names and the never-ending check ----------

test("a while predicate's unknown name is rejected", () => {
  const error = loadError(() => build(swap(GOOD, "    while: not exited\n", "    while: not exted\n"), goodBody()));

  expect(error.message).toBe(
    'loop node \'chat_loop\': `while:` references unknown name(s) ["exted"] — the predicate reads the carried ' +
      'record ["exited","n"]',
  );
  // Located at the node.
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test.each([
  ["bare", GOOD],
  ["whole", swap(GOOD, "    while: not exited\n", "    while: ${not exited}\n")],
])("a while predicate builds in every spelling: %s", (_name, text) => {
  expect(loopOf(text, goodBody()).predicateKind).toBe("while");
});

test.each([
  ["bare", "    while: not exted\n"],
  ["whole", "    while: ${not exted}\n"],
])("a while predicate's unknown name is rejected in every spelling: %s", (_name, predicate) => {
  const error = loadError(() => build(swap(GOOD, "    while: not exited\n", predicate), goodBody()));

  expect(error.message).toBe(
    'loop node \'chat_loop\': `while:` references unknown name(s) ["exted"] — the predicate reads the carried ' +
      'record ["exited","n"]',
  );
});

/** The `body` def of the never-ending cases: it reads `{done, attempts}` and returns `attempts`. */
function neverEndsBody(): ComposedFlow {
  return handChild({
    inputs: params("  done: !bool\n  attempts: !int\n"),
    output: { attempts: "${input.attempts + 1}" },
  });
}

// `until: done` reads only `done`, which the body never returns, and there is no `max:`, so the
// loop can never end.
const NEVER_ENDS = `id: loop-never-ends
name: loop_never_ends
nodes:
  wait:
    kind: loop
    call: body
    input:
      done: false
      attempts: 0
    until: done
`;

/** The body of the two-field never-ending loop: it reads `{a, b, n}` and returns `n`. */
function twoReadBody(): ComposedFlow {
  return handChild({ inputs: params("  a: !bool\n  b: !bool\n  n: !int\n"), output: { n: "${input.n + 1}" } });
}

/** Build loop `wait` of `text` over the never-ending body, as `build` does. */
function buildWait(text: string, child: ComposedFlow = neverEndsBody()): BuildResult {
  return build(text, child, { nodeId: "wait" });
}

/** The never-ending error of loop `id` whose `condition` reads `names`, none returned. */
function neverEnds(id: string, condition: string, names: string[]): string {
  const quoted = names.map((name) => `\`${name}\``);
  return (
    `loop node '${id}': this loop can never end: \`${condition}\` reads ${quoted.join(", ")}, which the body ` +
    `never updates; return ${quoted.join(" or ")} from the body, or add \`max:\``
  );
}

test("a loop whose condition reads only unreturned fields is a load error", () => {
  const error = loadError(() => buildWait(NEVER_ENDS));

  expect(error.message).toBe(neverEnds("wait", "until: done", ["done"]));
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test("a while loop whose condition reads only unreturned fields is a load error", () => {
  // The body returns only `n`; `while: not exited` reads only `exited`.
  const error = loadError(() => build(NO_MAX, goodBody({ n: "int" })));

  expect(error.message).toBe(neverEnds("chat_loop", "while: not exited", ["exited"]));
});

test("an inline body loop whose condition never changes is a load error", () => {
  // INLINE_SUBSET's body returns only `n`; `until: exited` reads only `exited`.
  const error = loadError(() => assembleText(swap(INLINE_SUBSET, "    times: 2\n", "    until: exited\n")));

  expect(error.message).toBe(neverEnds("repeat", "until: exited", ["exited"]));
});

test("the never-ending error names every unreturned field read", () => {
  const seeded = swap(
    NEVER_ENDS,
    "      done: false\n      attempts: 0\n",
    "      a: false\n      b: false\n      n: 0\n",
  );
  const text = swap(seeded, "    until: done\n", "    until: a or b\n");

  const error = loadError(() => buildWait(text, twoReadBody()));

  expect(error.message).toBe(neverEnds("wait", "until: a or b", ["a", "b"]));
});

test.each(["done", "${done}", "${not done}", "not done"])(
  "the never-ending check reads every condition spelling: %s",
  (condition) => {
    const error = loadError(() => buildWait(swap(NEVER_ENDS, "    until: done\n", `    until: "${condition}"\n`)));

    expect(error.message).toBe(neverEnds("wait", `until: ${condition}`, ["done"]));
  },
);

test("a null max still gets the never-ending error", () => {
  const error = loadError(() => buildWait(swap(NEVER_ENDS, "    until: done\n", "    until: done\n    max: null\n")));

  expect(error.message).toBe(neverEnds("wait", "until: done", ["done"]));
});

test("a dotted condition reference names the carried field", () => {
  const seeded = swap(NEVER_ENDS, "      done: false\n", "      st: {ok: false}\n");
  const text = swap(seeded, "    until: done\n", "    until: st.ok\n");
  const body = handChild({
    inputs: params("  st: !Any\n  attempts: !int\n"),
    output: { attempts: "${input.attempts + 1}" },
  });

  const error = loadError(() => buildWait(text, body));

  expect(error.message).toBe(neverEnds("wait", "until: st.ok", ["st"]));
});

test.each([
  [
    "two drivers",
    "    until: done\n    while: not done\n",
    'loop node \'wait\': exactly one of while:/until:/times: is required (got ["until","while"])',
  ],
  [
    "times and while",
    "    times: 2\n    while: not done\n",
    'loop node \'wait\': exactly one of while:/until:/times: is required (got ["times","while"])',
  ],
  [
    "unknown name",
    "    until: done or nope\n",
    'loop node \'wait\': `until:` references unknown name(s) ["nope"] — the predicate reads the carried record ' +
      '["attempts","done"]',
  ],
  ["bad max", "    until: done\n    max: 0\n", "loop node 'wait': `max:` must be >= 1 (got 0)"],
  [
    "parse error",
    '    until: "done and"\n',
    "loop node 'wait': `until:` is not a valid boolean expression: could not parse expression 'done and': " +
      "unexpected end of expression at line 1, column 9",
  ],
])("the never-ending error does not stack on other loop errors: %s", (_name, driver, message) => {
  const error = loadError(() => buildWait(swap(NEVER_ENDS, "    until: done\n", driver)));

  expect(error.message).toBe(message);
});

test("a predicate that doesn't parse beside another error joins it at the node", () => {
  const text = swap(NEVER_ENDS, "    until: done\n", '    until: "done and"\n    max: 0\n');

  const error = loadError(() => buildWait(text));

  expect(error.message).toBe(
    "loop node 'wait': `max:` must be >= 1 (got 0)\n" +
      "  loop node 'wait': `until:` is not a valid boolean expression: could not parse expression 'done and': " +
      "unexpected end of expression at line 1, column 9",
  );
  // At the node (the `wait:` line), not the `until:` line.
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test("a never-changing condition with max loads", () => {
  const text = swap(NEVER_ENDS, "    until: done\n", "    until: done\n    max: 3\n");

  const node = loopOf(text, neverEndsBody(), { nodeId: "wait" });

  expect(node.maxIters).toBe(3);
  expect(node.returnedFields).toEqual(["attempts"]);
});

test("a condition reading one returned field loads", () => {
  // `done` is never returned, but `attempts` is, so the condition can change.
  const text = swap(NEVER_ENDS, "    until: done\n", "    until: done or attempts >= 3\n");

  const node = loopOf(text, neverEndsBody(), { nodeId: "wait" });

  expect(node.maxIters).toBeNull();
});

test("unknown returned fields get only the record error", () => {
  // A value body over an untyped producer: its returned field set is unknown at load.
  const body = handChild({
    inputs: params("  done: !bool\n  attempts: !int\n"),
    producers: { step: null },
    output: "${step.output}",
  });

  const error = loadError(() => buildWait(NEVER_ENDS, body));

  expect(error.message).toBe(
    "loop node 'wait': body callable 'body' must return a record with at least one carried field " +
      '(a subset of the carried record ["attempts","done"]); write `output: {field: ...}` or a whole reference ' +
      "to a node with a record output",
  );
});

// ---------- the predicate's field reads and checks ----------

// `repeat` carries `{state}`, a record holding `count`; its inline body returns `{state}`. PRED
// stands for the driver.
const NESTED_STATE = `id: loop-nested
name: loop_nested
nodes:
  repeat:
    kind: loop
    PRED
    input:
      state: {count: 0}
    nodes:
      bump:
        kind: code
        input:
          s: \${input.state}
        output:
          state:
            count: int
        code: |
          return {"state": {"count": inputs["s"]["count"] + 1n}}
    output: \${bump.output}
output: \${repeat.output}
`;

/** Assemble `text` and return its loop `nodeId`. */
function assembledLoop(text: string, nodeId: string): LoopNode {
  const node = assembleText(text).flow.compiled.nodes.get(nodeId);
  expect(node).toBeInstanceOf(LoopNode);
  return node as LoopNode;
}

/** Assemble `NESTED_STATE` with driver `pred` and return its loop `repeat`. */
function buildNested(pred: string): LoopNode {
  return assembledLoop(swap(NESTED_STATE, "PRED", pred), "repeat");
}

test.each(["while: state.count < 3", "until: state.count >= 3"])(
  "a predicate's nested field read builds: %s",
  (pred) => {
    buildNested(pred);
  },
);

test.each([
  ["while: state.cuont < 3", "while"],
  ["until: state.cuont >= 3", "until"],
  ["while: ${state.cuont < 3}", "while"],
])("a predicate's unknown nested field is rejected: %s", (pred, key) => {
  const error = loadError(() => buildNested(pred));

  expect(error.message).toBe(
    `loop node 'repeat': \`${key}:\` reference \${state.cuont} reads unknown field 'cuont' on record`,
  );
  // Located at the node.
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test("a predicate's field read on a scalar is rejected", () => {
  const error = loadError(() => build(swap(GOOD, "    while: not exited\n", "    while: n.value < 3\n"), goodBody()));

  expect(error.message).toBe(
    "loop node 'chat_loop': `while:` reference ${n.value} reads field 'value' on type int, which has no fields",
  );
});

// `repeat` carries `{n, cfg}`; its inline body reads `n` and returns only `{n}`, so `cfg` carries
// over, typed from its seed. PRED stands for the driver and SEED for `cfg`'s seed.
const OMITTED_SEED = `id: loop-omitted
name: loop_omitted
typedefs:
  Cfg: {limit: int}
input:
  cfg: !Cfg
nodes:
  repeat:
    kind: loop
    PRED
    input:
      n: 0
      cfg: SEED
    nodes:
      inc:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1n
    output:
      n: \${inc.output}
output: \${repeat.output}
`;

test("a predicate's unknown field on an omitted typed seed is rejected", () => {
  // The flow's `cfg: !Cfg` input types the seed `${input.cfg}`, read by the loader's reader.
  const text = swap(swap(OMITTED_SEED, "PRED", "while: n < cfg.limt"), "SEED", "${input.cfg}");

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe("loop node 'repeat': `while:` reference ${cfg.limt} reads unknown field 'limt' on record");
});

test("a predicate's field on an omitted literal seed is not checked", () => {
  const text = swap(swap(OMITTED_SEED, "PRED", "while: n < cfg.other\n    max: 3"), "SEED", "{limit: 2}");

  assembledLoop(text, "repeat");
});

// `ask` carries `{n, r}`, `r` a `Step`: a `Done` or a `More`; its inline body returns both. PRED
// stands for the driver.
const UNION_STATE = `id: loop-union
name: loop_union
typedefs:
  Done: {answer: str}
  More: {hint: str}
  Step: Union(Done, More)
nodes:
  ask:
    kind: loop
    PRED
    input:
      n: 0
      r: !Step {"hint":"start"}
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
          r: Step
        code: |
          const n = inputs["n"] + 1n
          return {"n": n, "r": n >= 3n ? {"answer": "ok"} : {"hint": "again"}}
    output: \${step.output}
output: \${ask.output}
`;

// The same loop whose union members are tagged by `kind`.
const TAGGED_STATE = (
  [
    [
      "  Done: {answer: str}\n  More: {hint: str}\n",
      "  Done: {kind: 'Literal(done)', answer: str}\n  More: {kind: 'Literal(more)', hint: str}\n",
    ],
    ['{"hint":"start"}', '{"kind":"more","hint":"start"}'],
    ['{"answer": "ok"}', '{"kind": "done", "answer": "ok"}'],
    ['{"hint": "again"}', '{"kind": "more", "hint": "again"}'],
  ] as const
).reduce<string>((text, [old, replacement]) => swap(text, old, replacement), UNION_STATE);

// The same loop carrying a `Done` record rather than the union.
const DONE_STATE = swap(
  swap(UNION_STATE, 'r: !Step {"hint":"start"}', 'r: !Done {"answer":"x"}'),
  "          r: Step\n",
  "          r: Done\n",
);

// The same loop carrying an `object`.
const OBJECT_STATE = swap(
  swap(UNION_STATE, 'r: !Step {"hint":"start"}', 'r: !object {"hint":"start"}'),
  "          r: Step\n",
  "          r: object\n",
);

// `Step` and `Finished` share `Done` and `Skip`; `Finished` also has `Other`, which no `Step`
// value can be, so `isinstance(r, Finished)` narrows `r` to `Union(Done, Skip)`.
const PARTLY_SHARED_STATE = swap(
  swap(UNION_STATE, "id: loop-union\nname: loop_union\n", "id: loop-partly-shared\nname: loop_partly_shared\n"),
  "  Step: Union(Done, More)\n",
  "  Skip: {answer: str, why: str}\n  Other: {x: int}\n  Step: Union(Done, More, Skip)\n" +
    "  Finished: Union(Done, Skip, Other)\n",
);

/** Assemble the `ask` loop `state` with driver `pred` and return the loop. */
function buildAsk(state: string, pred: string): LoopNode {
  return assembledLoop(swap(state, "PRED", pred), "ask");
}

/** The message of `buildAsk`'s load error. */
function askError(state: string, pred: string): string {
  return loadError(() => buildAsk(state, pred)).message;
}

const MEMBER_ONLY =
  "loop node 'ask': `until:` reference ${r.answer} reads field 'answer', which only 1 of 2 union members have; " +
  "check the member first with `isinstance(r, <Type>) and ...`";

test.each([
  "until: r.answer == 'ok'",
  // The check must come before the read.
  "until: r.answer == 'ok' and isinstance(r, Done)",
  // `n > 5` may be what made the `and` false, so `r` may still be either member.
  "until: not (n > 5 and isinstance(r, More)) and r.answer == 'ok'",
])("a predicate's unguarded member field read is rejected: %s", (pred) => {
  expect(askError(UNION_STATE, pred)).toBe(MEMBER_ONLY);
});

test.each(["until: isinstance(r, More) and r.answer == 'ok'", "until: isinstance(r, Done) or r.answer == 'ok'"])(
  "a predicate's read of a field the checked member lacks is rejected: %s",
  (pred) => {
    expect(askError(UNION_STATE, pred)).toBe(
      "loop node 'ask': `until:` reference ${r.answer} reads field 'answer', which More doesn't have; r is More " +
        "here, by the checks before this read",
    );
  },
);

test("a predicate's read after a check the type always decides is checked unnarrowed", () => {
  expect(askError(UNION_STATE, "until: isinstance(r, Step) or r.nosuch == 'x'")).toBe(
    "loop node 'ask': `until:` reference ${r.nosuch} reads unknown field 'nosuch' on record",
  );
  buildAsk(UNION_STATE, "until: isinstance(r, Step) or n > 2");
});

test("a predicate's check that can never be true is rejected", () => {
  const pred = "until: isinstance(r, Done) and isinstance(r, More) and r.nosuch == 'x'";

  // The read after it is reached by no value, so it is not checked as well.
  expect(askError(UNION_STATE, pred)).toBe(
    "loop node 'ask': `until:` `isinstance(r, More)` can never be true here: r is Done, by the checks before it",
  );
});

test.each([
  [
    "no overlap",
    "until: isinstance(r, int) and r.zz == 1",
    "`isinstance(r, int)` can never be true: r is declared Step, and no Step value fits int",
  ],
  [
    "always",
    "until: not isinstance(n, int) and n.zz == 1",
    "`not isinstance(n, int)` can never be true: n is declared int, so `isinstance(n, int)` is always true",
  ],
])("a predicate's read after a reported check is not checked: %s", (_name, pred, expected) => {
  expect(askError(UNION_STATE, pred)).toBe(`loop node 'ask': \`until:\` ${expected}`);
});

test("a predicate's tagged member field error points at isinstance", () => {
  expect(askError(TAGGED_STATE, "until: r.answer == 'ok'")).toBe(
    "loop node 'ask': `until:` reference ${r.answer} reads field 'answer', which only union members with kind " +
      "done have; check the member first with `isinstance(r, <Type>) and ...`",
  );
  buildAsk(TAGGED_STATE, "until: isinstance(r, Done) and r.answer == 'ok'");
});

/** The message of a tag comparison against `"dnoe"`, which is no tag of `r.kind`. */
const NOT_A_TAG =
  'loop node \'ask\': `until:` `r.kind == "dnoe"` compares r.kind with "dnoe", which is not one of its tags ' +
  '(done, more); did you mean "done"?';

test("a predicate's tag comparison with a value that is no tag is rejected", () => {
  expect(askError(TAGGED_STATE, "until: r.kind == 'dnoe'")).toBe(NOT_A_TAG);
});

test.each([
  [
    "always",
    DONE_STATE,
    "until: not isinstance(r, Done) and r.nosuch == 'x'",
    "`not isinstance(r, Done)` can never be true: r is declared Done, so `isinstance(r, Done)` is always true",
  ],
  [
    "always, union",
    DONE_STATE,
    "until: not isinstance(r, Step) and r.nosuch == 'x'",
    "`not isinstance(r, Step)` can never be true: r is declared Done, so `isinstance(r, Step)` is always true",
  ],
  [
    "no overlap",
    DONE_STATE,
    "until: isinstance(r, More) and r.nosuch == 'x'",
    "`isinstance(r, More)` can never be true: r is declared Done, and no Done value fits More",
  ],
  [
    "no overlap, union",
    UNION_STATE,
    "until: isinstance(r, str) and r.nosuch == 'x'",
    "`isinstance(r, str)` can never be true: r is declared Step, and no Step value fits str",
  ],
])("a predicate reports only the impossible check: %s", (_name, state, pred, expected) => {
  expect(askError(state, pred)).toBe(`loop node 'ask': \`until:\` ${expected}`);
});

test("a predicate's read past a check the checks before it decide is checked under them", () => {
  const pred = "until: isinstance(r, Done) and (isinstance(r, Done) or r.answer == 'ok')";

  buildAsk(UNION_STATE, pred);
});

test("a predicate's read after a value that is no tag is not checked", () => {
  expect(askError(TAGGED_STATE, "until: r.kind == 'dnoe' and r.zz == 'x'")).toBe(NOT_A_TAG);
});

// `ask` calls the `body` def, which returns `r: Done`; the type contract doesn't force the seed to
// fit it, so a `while:` (first run on the seed) is checked against the seed's type too. PRED
// stands for the driver and SEED for `r`'s seed.
const SEEDED_BODY = `id: loop-seed
name: loop_seed
typedefs:
  Done: {answer: str}
  More: {hint: str}
nodes:
  ask:
    kind: loop
    call: body
    PRED
    max: 5
    input:
      n: 0
      r: SEED
output: \${ask.output}
`;

test.each([
  [
    "typed seed",
    '!Done {"answer": "x"}',
    "while: isinstance(r, More)",
    "`while:` `isinstance(r, More)` can never be true: r is declared Done, and no Done value fits More",
  ],
  [
    "typed seed, always",
    '!Done {"answer": "x"}',
    "while: not isinstance(r, Done)",
    "`while:` `not isinstance(r, Done)` can never be true: r is declared Done, so `isinstance(r, Done)` is always true",
  ],
  [
    "until sees no seed",
    '{"hint": "start"}',
    "until: isinstance(r, More)",
    "`until:` `isinstance(r, More)` can never be true: r is declared Done, and no Done value fits More",
  ],
])("a predicate impossible on the seed and the body is rejected: %s", (_name, seed, pred, expected) => {
  const body = handChild({
    inputs: params("  n: !int\n"),
    producers: { step: record({ n: "int", r: "Done" }, { Done: { answer: "str" } }) },
    output: "${step.output}",
  });
  const text = swap(swap(SEEDED_BODY, "PRED", pred), "SEED", seed);

  // A tagged seed's declared type comes from `assemble`, so the flow is assembled.
  const error = loadError(() => assembleText(text, "flow.yaml", { children: { body } }));

  expect(error.message).toBe(`loop node 'ask': ${expected}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});

test("a predicate's isinstance check on an object field checks reads on the type", () => {
  const message =
    "loop node 'ask': `until:` reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done here, " +
    "by the checks before this read; did you mean 'answer'?";

  for (const pred of [
    "until: isinstance(r, Done) and r.answr == 'ok'",
    "until: not isinstance(r, Done) or r.answr == 'x'",
  ]) {
    expect(askError(OBJECT_STATE, pred)).toBe(message);
  }
  buildAsk(OBJECT_STATE, "until: isinstance(r, Done) and r.answer == 'ok'");
});

test("a predicate's isinstance of a partly shared union rejects a member-only field", () => {
  expect(askError(PARTLY_SHARED_STATE, "until: isinstance(r, Finished) and r.why == 'x'")).toBe(
    "loop node 'ask': `until:` reference ${r.why} reads field 'why', which only 1 of 2 union members have; " +
      "check the member first with `isinstance(r, <Type>) and ...`",
  );
});

test("a predicate's isinstance check with an unknown type reports the type", () => {
  // A typo in the checked type is reported as that typo, not as an unguarded read.
  expect(askError(UNION_STATE, "until: isinstance(r, Dnoe) and r.answer == 'ok'")).toBe(
    "loop node 'ask' until: unknown type 'Dnoe'; isinstance takes a typedefs: name or one of str, int, float, " +
      "bool, date, datetime, object",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-loop/engine_loop_build__a_loop_without_a_driver_is_rejected", call: assembledWith({ bump: goodBody() }) },
  {
    id: "load-loop/engine_loop_build__body_output_field_outside_carried_record_is_rejected",
    call: assembledWith({ bump: outsideBody() }),
  },
  {
    id: "load-loop/engine_loop_build__body_returning_a_scalar_is_rejected",
    call: assembledWith({
      bump: handChild({ inputs: params("  n: !int\n"), producers: { step: INT }, output: "${step.output}" }),
    }),
  },
  {
    id: "load-loop/engine_loop_build__condition_reading_only_unreturned_fields_is_rejected",
    call: assembledWith({ body: neverEndsBody() }),
  },
  { id: "load-loop/engine_loop_build__float_max_is_rejected", call: assembledWith({ bump: goodBody() }) },
  { id: "load-loop/engine_loop_build__max_of_zero_is_rejected", call: assembledWith({ body: neverEndsBody() }) },
  {
    id: "load-loop/engine_loop_build__never_ending_error_names_every_unreturned_field_read",
    call: assembledWith({ body: twoReadBody() }),
  },
  {
    id: "load-loop/engine_loop_build__predicate_field_read_on_scalar_is_rejected",
    call: assembledWith({ bump: goodBody() }),
  },
  {
    id: "load-loop/engine_loop_build__predicate_unknown_name_is_rejected",
    call: assembledWith({ body: neverEndsBody() }),
  },
  { id: "load-loop/engine_loop_build__text_max_is_rejected", call: assembledWith({ bump: goodBody() }) },
  { id: "load-loop/engine_loop_build__times_with_max_is_rejected", call: assembledWith({ bump: goodBody() }) },
  { id: "load-loop/engine_loop_build__two_drivers_are_rejected", call: assembledWith({ body: neverEndsBody() }) },
])("assemble: $id", runCase);

// Each body is inline, so the child build types its seeds.
test.each<CaseRow>(
  [
    "inline_body_condition_that_never_changes_is_rejected",
    "predicate_isinstance_of_a_disjoint_type_is_rejected",
    "predicate_isinstance_on_an_object_checks_reads",
    "predicate_isinstance_ruled_out_by_an_earlier_check",
    "predicate_isinstance_with_an_unknown_type",
    "predicate_negated_isinstance_of_the_declared_type_is_rejected",
    "predicate_read_of_a_field_the_checked_member_lacks",
    "predicate_tag_comparison_with_a_non_tag",
    "predicate_tagged_member_field_error_points_at_isinstance",
    "predicate_unguarded_member_field_read_is_rejected",
    "predicate_unknown_field_on_omitted_typed_seed_is_rejected",
  ].map((name) => ({ id: `load-loop/engine_loop_build__${name}`, call: assembled })),
)("assemble: $id", runCase);

test.each<CaseRow>([
  {
    // Only the reader typing the omitted `keep` field reaches the cycle.
    id: "load-loop/engine_loop_build__omitted_field_seed_cycle_is_located",
    call: assembledWith({ bump: goodBody({ n: "int" }) }),
  },
])("assemble: $id", runCase);
