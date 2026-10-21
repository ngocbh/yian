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
 * `test_omitted_field_seed_cycle_is_located`. The carried types and the seed check (the loop type
 * contract and the cross-flow binding check on a LOOP seed), through `assembleText` with each def
 * built by hand: `test_returned_field_type_mismatch_still_rejected_in_partial_body`,
 * `test_body_output_types_must_equal_carried`,
 * `test_opaque_body_field_does_not_erase_proven_sibling_type_mismatch`,
 * `test_constant_seed_that_cannot_fill_the_body_input_is_a_load_error` (7, and the date row below),
 * `test_seed_is_checked_even_when_a_while_loop_never_runs`,
 * `test_text_seed_into_int_input_of_a_full_loop_is_a_load_error`,
 * `test_reference_or_typed_seed_that_cannot_fill_the_body_input_is_a_load_error` (6),
 * `test_reference_seed_is_checked_even_when_a_while_loop_never_runs`, and the load halves of
 * `test_constant_seed_that_fits_the_body_input_loads_and_runs` (6),
 * `test_reference_or_typed_seed_that_fits_the_body_input_loads_and_runs` (8),
 * `test_carried_field_the_body_does_not_read_is_not_checked`,
 * `test_inline_body_constant_seed_fits_its_generated_parameter`,
 * `test_inline_body_reference_seed_fits_its_generated_parameter` and
 * `test_while_null_check_lets_an_optional_seed_fill_a_strict_body_input` (2); both loads of
 * `test_loop_seed_reads_the_case_narrowed_type` (the CASE narrowing types the seed).
 *
 * Changed: a row whose Python flow calls a `defs:` body parses the loop node (`nodesOf`) and builds
 * it over a `BuildContext` whose resolver returns the body built by hand (`handChild`: the def's
 * parameters read from its `input:` text, its typed nodes as producers with their output
 * contracts), where Python loads the flow; these rows test the LOOP build on its own. Such a
 * context has no binding-contract reader: the merged-contract rows give it a test-local
 * `bindingContractReader` typing the omitted fields' seeds and recording what the LOOP asks;
 * without one an untagged seed stays opaque, as Python's literal seed does, so a `while:` reading
 * it gets no seed type: none of its checks is reported impossible and none narrows a read, and a
 * plain field read is walked on the carried record. `test_loop_seeded_from_partial_loop_sees_merged_type` and
 * `test_omitted_field_seed_cycle_is_located` assemble the flow (`assembleText`, the def built by
 * hand), so the loader's reader types the omitted seeds, building the seeding loop on demand. The
 * rows whose Python flow has an inline `nodes:` body assemble that flow, its inline CODE written in
 * TS as the harvested cases write it. The typed-seed rows of
 * `test_predicate_impossible_on_the_seed_and_the_body_is_rejected` assemble the flow
 * (`assembleText` with the def built by hand), since a tagged seed's declared type comes from
 * `assemble`; parsing alone drops the tag. The error rows pin the whole message, where Python
 * matches fragments; lists of names are JSON text (`["exited","n"]`) and values the text form
 * (`(got "5")`, `(got true)`, a tag comparison's `with "dnoe"` and `did you mean "done"?`). The
 * `("str", "2026-01-01")` row of `test_constant_seed_that_cannot_fill_the_body_input_is_a_load_error`
 * loads (rule `dates`): an unquoted date is text in TS, so it fits a `str` input, where Python reads
 * a date and asks to quote it.
 *
 * TS-only: the counts are YAML ints (`bigint`) converted to numbers: `times: 3` gives `3`, a
 * `max:` or `times:` past `Number.MAX_SAFE_INTEGER` is refused with the at-most message (the
 * largest safe integer builds), and a float `max: 2.0` is refused as `(got 2)`; `times:` is
 * refused below 1 and as a float, a bool or text, as `max:` is; `times:` beside `while:` is a
 * second driver; a body input that isn't a carried field is refused; a count error and a
 * field-name error are joined at the node, as are a count error and a predicate that doesn't
 * parse; and a hand-constructed descriptor's `number` count is refused.
 *
 * Pinned at the end (`assembled`, each case's `defs:` or inline body built by the loader's
 * callable scope): the `load-loop/engine_loop_build__*` cases whose error is a driver, count,
 * field-name, predicate-name, predicate field-read, never-ending, predicate check or seed error,
 * `load-loop/engine_loop_build__omitted_field_seed_cycle_is_located`,
 * `load-loop/engine_loop_build__date_seed_for_a_str_input_loads_as_text` (`ok`, rule `dates`) and
 * the reference check's `load-loop/engine_loop_build__partial_loop_output_rejects_an_unknown_field`.
 *
 * The reference rows assemble with `assembleText`, each def built by hand:
 * `test_partial_loop_contract_rejects_unknown_downstream_field_at_load` (its `body` def's `note`
 * output an untyped producer's, so the loop's contract is a partial record) and the downstream
 * half of `test_merged_contract_types_omitted_field_from_its_seed`.
 *
 * The runs, each loading the flow with the defs Python writes in it (`withDefs`, or the flow's own
 * inline body) and running it through `runFlow`: `test_predicate_nested_field_read_builds` (2),
 * `test_predicate_member_field_read_guarded_by_isinstance_builds` (4),
 * `test_predicate_while_read_guarded_through_or_and_not_builds` (2), the run halves of
 * `test_predicate_tagged_member_field_error_points_at_isinstance` and
 * `test_predicate_isinstance_check_on_an_object_field_checks_reads_on_the_type`,
 * `test_predicate_tag_comparison_narrows_a_read` (2),
 * `test_predicate_while_is_checked_against_the_seed_too` (6),
 * `test_predicate_isinstance_of_a_partly_shared_union_reads_the_shared_fields`, the run halves of
 * `test_constant_seed_that_fits_the_body_input_loads_and_runs` (6),
 * `test_inline_body_constant_seed_fits_its_generated_parameter`,
 * `test_reference_or_typed_seed_that_fits_the_body_input_loads_and_runs` (8),
 * `test_carried_field_the_body_does_not_read_is_not_checked`,
 * `test_inline_body_reference_seed_fits_its_generated_parameter`,
 * `test_while_null_check_lets_an_optional_seed_fill_a_strict_body_input` (both body inputs, both
 * runs each) and `test_loop_seed_reads_the_case_narrowed_type`. The CODE bodies are TS, ints
 * arriving as `number` (decision 45); the runs' ints are bigints (decision 4a). Changed: the float
 * rows of the two fitting-seed groups give an int (`("float", "5")` outputs `5n`, decision 38).
 *
 * Not ported: the run halves of `test_one_field_body_record_returns_only_that_carried_field`,
 * covered by `run-loop/engine_loop_build__one_field_body_record_returns_only_that_field`, and of
 * `test_value_body_returns_the_referenced_record_beside_a_carried_result`, covered by
 * `run-loop/engine_loop_build__value_body_returns_the_referenced_record`;
 * `test_predicate_field_on_omitted_typed_seed_builds`, covered by
 * `run-loop/engine_loop_build__while_reads_a_carried_over_field`.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { type BindingContractReader, BuildContext, type BuildResult } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { isExactly, type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
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
 * `text` with `defs` (each entry indented under `defs:`) before its top-level `nodes:`: the runs
 * load the flow with the defs Python writes in it, where the build rows build each def by hand.
 */
const withDefs = (text: string, defs: string) => swap(text, "\nnodes:\n", `\ndefs:\n${defs}nodes:\n`);

/** Load `text` and run it on `inputs`, expecting it to succeed; return its output. */
async function runOutput(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text), inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
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

test("the merged contract types an omitted field from its seed (a downstream read)", () => {
  const registry = readTypedefs({ Person: { name: "str", handle: "str" } });
  const turn = handChild({
    inputs: params("  person: !Person\n  talking_to: !str\n  exited: !bool\n  reply: !str\n", registry),
    output: { exited: "${input.exited}", reply: "${input.talking_to}" },
  });
  const text = `id: loop-chat
name: loop_chat
typedefs:
  Person:
    name: str
    handle: str
input:
  person: !Person
  talking_to: !str
${CHAT.split("\n").slice(2).join("\n")}  sink:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: \${chat.output.person.missing}
    output: str
output: \${sink.output}
`;

  expect(loadError(() => assembleText(text, "flow.yaml", { children: { turn } })).message).toContain(
    "unknown field 'missing'",
  );
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

/** The `body` def: `n` and `note` in, `{n, note}` out, `note` from a producer with no contract. */
const partialBody = () =>
  handChild({
    inputs: params("  n: !int\n  note: !str\n"),
    producers: { shout: null },
    output: { n: "${input.n}", note: "${shout.output}" },
  });

test("a partial loop contract rejects an unknown downstream field at load", () => {
  const text = `id: loop-partial-reference
name: loop_partial_reference
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 1
      note: seed
    times: 1
  sink:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: \${loop.output.missing}
    output: str
output: \${sink.output}
`;

  expect(loadError(() => assembleText(text, "flow.yaml", { children: { body: partialBody() } })).message).toContain(
    "unknown field 'missing'",
  );
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
          return {"state": {"count": inputs["s"]["count"] + 1}}
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

test.each(["while: state.count < 3", "until: state.count >= 3"])("predicate nested field read builds: %s", async (pred) => {
  expect(await runOutput(swap(NESTED_STATE, "PRED", pred))).toEqual({ state: { count: 3n } });
});

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
          const n = inputs["n"] + 1
          return {"n": n, "r": n >= 3 ? {"answer": "ok"} : {"hint": "again"}}
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
  "until: isinstance(r, Done) and r.answer == 'ok'",
  "until: ${isinstance(r, Done) and r.answer == 'ok'}",
  // `or` and `not` narrow too: each read runs only once the checks before it decide.
  "until: n > 5 or isinstance(r, Done) and r.answer == 'ok'",
  "until: not isinstance(r, More) and r.answer == 'ok'",
])("predicate member field read guarded by isinstance builds: %s", async (pred) => {
  expect(await runOutput(swap(UNION_STATE, "PRED", pred))).toEqual({ n: 3n, r: { answer: "ok" } });
});

test.each(["while: not (isinstance(r, Done) and r.answer == 'ok')", "while: isinstance(r, More) or r.answer != 'ok'"])(
  "predicate while read guarded through or and not builds: %s",
  async (pred) => {
    expect(await runOutput(swap(UNION_STATE, "PRED", pred))).toEqual({ n: 3n, r: { answer: "ok" } });
  },
);

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

test("predicate tagged member field error points at isinstance (runs)", async () => {
  const output = await runOutput(swap(TAGGED_STATE, "PRED", "until: isinstance(r, Done) and r.answer == 'ok'"));

  expect(output).toEqual({ n: 3n, r: { kind: "done", answer: "ok" } });
});

test.each(["until: r.kind == 'done' and r.answer == 'ok'", "while: r.kind != 'done' or r.answer != 'ok'"])(
  "predicate tag comparison narrows a read: %s",
  async (pred) => {
    expect(await runOutput(swap(TAGGED_STATE, "PRED", pred))).toEqual({ n: 3n, r: { kind: "done", answer: "ok" } });
  },
);

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

// `SEEDED_BODY` with the `body` def Python writes and the `prev` node a seed may reference.
const SEEDED_FLOW = swap(
  SEEDED_BODY,
  "\nnodes:\n",
  `
defs:
  body:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
          r: Done
        code: |
          return {"n": inputs["n"] + 1, "r": {"answer": "ok"}}
    output: \${step.output}
nodes:
  prev:
    kind: code
    code: 'return {hint: "x"}'
`,
);

test.each([
  ["null", "while: not isinstance(r, Done)"],
  ['{"hint": "start"}', "while: not isinstance(r, Done)"],
  ["${prev.output}", "while: not isinstance(r, Done)"],
  ['{"hint": "start"}', "while: isinstance(r, More)"],
  ["${prev.output}", "while: isinstance(r, More)"],
  ['{"hint": "start"}', "while: isinstance(r, More) and r.hint == 'start'"],
])("predicate while is checked against the seed too: %s %s", async (seed, pred) => {
  const output = await runOutput(swap(swap(SEEDED_FLOW, "PRED", pred), "SEED", seed));

  expect(output).toEqual({ n: 1n, r: { answer: "ok" } });
});

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

test("predicate isinstance check on an object field checks reads on the type (runs)", async () => {
  const output = await runOutput(swap(OBJECT_STATE, "PRED", "until: isinstance(r, Done) and r.answer == 'ok'"));

  expect(output).toEqual({ n: 3n, r: { answer: "ok" } });
});

test("predicate isinstance of a partly shared union reads the shared fields", async () => {
  const pred = "until: isinstance(r, Finished) and r.answer == 'ok'";

  expect(await runOutput(swap(PARTLY_SHARED_STATE, "PRED", pred))).toEqual({ n: 3n, r: { answer: "ok" } });
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

// ---------- the carried types and the seed check ----------

// `chat_loop` of `GOOD`, its carried `n` seeded from the int flow input `n0`, so the carried-type
// check has a seed type to compare (a literal seed stays lenient there).
const BAD_TYPES = swap(
  swap(GOOD, "name: l\n", "name: l\ninput:\n  n0: !int\n"),
  "      n: 0\n",
  "      n: ${input.n0}\n",
);

const CARRIED_N =
  "loop node 'chat_loop': carried field 'n' — body output is 'str', carried source is 'int' " +
  "(a returned field must keep its carried type)";

test("a returned field's type mismatch is still rejected in a partial body", () => {
  const error = loadError(() => assembleText(BAD_TYPES, "flow.yaml", { children: { bump: goodBody({ n: "str" }) } }));

  expect(error.message).toBe(CARRIED_N);
});

test("body output types must equal the carried types", () => {
  const error = loadError(() =>
    assembleText(BAD_TYPES, "flow.yaml", { children: { bump: goodBody({ n: "str", exited: "bool" }) } }),
  );

  expect(error.message).toBe(CARRIED_N);
  // Located at the loop's `input:` mapping, which starts at its first key.
  const inputStart = BAD_TYPES.split("\n").indexOf("      n: ${input.n0}") + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(inputStart);
});

test("an opaque body field does not erase a proven sibling's type mismatch", () => {
  // `note + "!"` is outside static inference, while `input.note` proves the body returns `str`
  // for the carried `n`, seeded as `int`.
  const text = `id: l
name: l
input:
  n0: !int
  note0: !str
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: \${input.n0}
      note: \${input.note0}
    times: 1
output: \${chat_loop.output}
`;
  const bump = handChild({
    inputs: params("  n: !int\n  note: !str\n"),
    output: { n: "${input.note}", note: '${input.note + "!"}' },
  });

  const error = loadError(() => assembleText(text, "flow.yaml", { children: { bump } }));

  expect(error.message).toBe(CARRIED_N);
});

/**
 * A loop over the `body` def (`seededBody`), seeding `n` with `seed`, `k` with 0 and `extra` (no
 * body input, so never checked) with `extra`, driven by `driver`; with `inputs`, the flow takes
 * `s: str`, `m: Optional(int)` and `i: int` for reference seeds.
 */
function seeded(seed: string, options: { driver?: string; inputs?: boolean; extra?: string } = {}): string {
  const { driver = "times: 1", inputs = false, extra = '"abc"' } = options;
  return `id: p
name: p
${inputs ? "input:\n  s: !str\n  m: !Optional(int)\n  i: !int\n" : ""}nodes:
  loop:
    kind: loop
    ${driver}
    input:
      n: ${seed}
      k: 0
      extra: ${extra}
    call: body
output: \${loop.output}
`;
}

/** The `body` def: `n: <nType>` and `k: int` in, its `step` node's `{k: int}` out, so `n` carries over. */
function seededBody(nType: string): ComposedFlow {
  return handChild({
    inputs: params(`  n: !${nType}\n  k: !int\n`),
    producers: { step: record({ k: "int" }) },
    output: "${step.output}",
  });
}

/** The runnable `body` def of `seeded`: `n: <nType>` and `k: int` in, `{k: k + 1}` out. */
const seededDef = (nType: string) => `  body:
    input:
      n: !${nType}
      k: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
          k: \${input.k}
        output:
          k: int
        code: 'return {k: inputs["k"] + 1}'
    output: \${step.output}
`;

/** Assemble `text` with `body` as the `body` def. */
function loadSeeded(text: string, body: ComposedFlow): ComposedFlow {
  return assembleText(text, "flow.yaml", { children: { body } }).flow;
}

/**
 * Expect `text` (seeding `n` with `seed`) to fail with the seed error `message`, located like a
 * CALL argument: at the loop's whole `input:` mapping, which starts at its first key.
 */
function expectSeedError(text: string, seed: string, nType: string, message: string): void {
  const error = loadError(() => loadSeeded(text, seededBody(nType)));

  expect(error.message).toBe(`loop node 'loop': binding 'n' — ${message}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(`      n: ${seed}`) + 1);
}

test.each([
  ["int", '"abc"', 'child expects \'int\', but the value "abc" does not fit'],
  [
    "int",
    '"1"',
    'child expects \'int\', but the value "1" does not fit; ' +
      "text is not converted between flows, so write 1 without quotes",
  ],
  ["int", "null", "child expects 'int', but the value null does not fit"],
  ["int 5", "null", "child expects 'int', but the value null does not fit"],
  ["int", "${null}", "child expects 'int', source is 'None'"],
  ["int", "${'1'}", "child expects 'int', source is 'str'"],
  ["int", '"x${1}"', "child expects 'int', source is 'str'"],
])("a constant seed that cannot fill the body input is a load error: %s %s", (nType, seed, message) => {
  expectSeedError(seeded(seed), seed, nType, message);
});

test("a date seed for a str input loads as text", () => {
  // Changed: Python reads `2026-01-01` as a date and asks to quote it; TS reads it as text.
  expect(loadSeeded(seeded("2026-01-01"), seededBody("str")).compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
});

test("a seed is checked even when a while loop never runs", () => {
  // `k: 0` fails `while:` at once, so the body never binds `n`; like a CALL on an untaken branch,
  // the provable mismatch still fails at load.
  const error = loadError(() => loadSeeded(seeded('"abc"', { driver: "while: k > 5" }), seededBody("int")));

  expect(error.message).toBe(
    'loop node \'loop\': binding \'n\' — child expects \'int\', but the value "abc" does not fit',
  );
});

test.each([
  ["int", "1"],
  ["float", "5"],
  ["Optional(int)", "null"],
  ["Optional(int)", "${null}"],
  ["str", '"2026-01-01"'],
  ["int", "${1 + 2}"],
])("a constant seed that fits the body input loads: %s %s", (nType, seed) => {
  expect(loadSeeded(seeded(seed), seededBody(nType)).compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
});

test.each<[string, string, unknown]>([
  ["int", "1", 1n],
  // A whole number in a float slot stays an int.
  ["float", "5", 5n],
  ["Optional(int)", "null", null],
  ["Optional(int)", "${null}", null],
  ["str", '"2026-01-01"', "2026-01-01"],
  ["int", "${1 + 2}", 3n],
])("constant seed that fits the body input loads and runs: %s %s", async (nType, seed, n) => {
  expect(await runOutput(withDefs(seeded(seed), seededDef(nType)))).toEqual({ n, k: 1n, extra: "abc" });
});

// An inline body over a text seed: its generated parameter `n` is typed `str` from the seed.
const INLINE_CONSTANT_SEED = `id: p
name: p
nodes:
  loop:
    kind: loop
    times: 1
    input:
      n: "abc"
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output:
          n: str
        code: 'return {n: inputs["n"]}'
    output: \${step.output}
output: \${loop.output}
`;

test("an inline body's constant seed fits its generated parameter", () => {
  const { flow } = assembleText(INLINE_CONSTANT_SEED);

  expect(flow.compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
});

test("inline body constant seed fits its generated parameter", async () => {
  expect(await runOutput(INLINE_CONSTANT_SEED)).toEqual({ n: "abc" });
});

test("a text seed into the int input of a full loop is a load error", () => {
  const text = `id: p
name: p
nodes:
  loop:
    kind: loop
    times: 2
    input:
      n: "abc"
    call: bump
output: \${loop.output}
`;
  const bump = handChild({
    inputs: params("  n: !int\n"),
    producers: { step: record({ n: "int" }) },
    output: "${step.output}",
  });

  const error = loadError(() => assembleText(text, "flow.yaml", { children: { bump } }));

  expect(error.message).toBe(
    'loop node \'loop\': binding \'n\' — child expects \'int\', but the value "abc" does not fit',
  );
});

test.each([
  ["int", "${input.s}", "child expects 'int', source is 'str'"],
  ["int", '!str "abc"', "child expects 'int', source is 'str'"],
  // An `Optional` source does not fit a strict input, as for a CALL argument.
  ["int", "${input.m}", "child expects 'int', source is 'Optional(int)'"],
  ["int", "!Optional(int) ${input.m}", "child expects 'int', source is 'Optional(int)'"],
  ["int", "${input.s :- 0}", "child expects 'int', source is 'str'"],
  ["int", "${input.m :- null}", "child expects 'int', but the default null does not fit"],
])("a reference or typed seed that cannot fill the body input is a load error: %s %s", (nType, seed, message) => {
  expectSeedError(seeded(seed, { inputs: true }), seed, nType, message);
});

test("a reference seed is checked even when a while loop never runs", () => {
  const error = loadError(() =>
    loadSeeded(seeded("${input.s}", { inputs: true, driver: "while: k > 5" }), seededBody("int")),
  );

  expect(error.message).toBe("loop node 'loop': binding 'n' — child expects 'int', source is 'str'");
});

test.each([
  ["int", "${input.i}"],
  ["float", "${input.i}"],
  ["Optional(int)", "${input.m}"],
  // A typed binding accepts the narrowing and checks it when the body starts.
  ["int", "!int ${input.m}"],
  ["int", "${input.m :- 0}"],
  ["int", "${input.m | 0}"],
  ["str", '"x${input.i}"'],
  // A non-null literal fills a strict input whatever its tag, as for a CALL argument.
  ["int", "!Optional(int) 5"],
])("a reference or typed seed that fits the body input loads: %s %s", (nType, seed) => {
  expect(loadSeeded(seeded(seed, { inputs: true }), seededBody(nType)).compiled.nodes.get("loop")).toBeInstanceOf(
    LoopNode,
  );
});

/** The flow inputs of the reference-seed runs. */
const SEED_INPUTS = { s: "x", m: 3n, i: 2n };

test.each<[string, string, unknown]>([
  ["int", "${input.i}", 2n],
  // A whole number in a float slot stays an int.
  ["float", "${input.i}", 2n],
  ["Optional(int)", "${input.m}", 3n],
  // A typed binding accepts the narrowing and checks it when the body starts.
  ["int", "!int ${input.m}", 3n],
  ["int", "${input.m :- 0}", 3n],
  ["int", "${input.m | 0}", 3n],
  ["str", '"x${input.i}"', "x2"],
  // A non-null literal fills a strict input whatever its tag, as for a CALL argument.
  ["int", "!Optional(int) 5", 5n],
])("reference or typed seed that fits the body input loads and runs: %s %s", async (nType, seed, n) => {
  const output = await runOutput(withDefs(seeded(seed, { inputs: true }), seededDef(nType)), SEED_INPUTS);

  expect(output).toEqual({ n, k: 1n, extra: "abc" });
});

test("a carried field the body does not read is not checked", () => {
  const flow = loadSeeded(seeded("1", { inputs: true, extra: "${input.m}" }), seededBody("int"));

  expect(flow.compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
});

test("carried field the body does not read is not checked", async () => {
  const text = withDefs(seeded("1", { inputs: true, extra: "${input.m}" }), seededDef("int"));

  expect(await runOutput(text, SEED_INPUTS)).toEqual({ n: 1n, k: 1n, extra: 3n });
});

// An inline body over a reference seed: its generated parameter `n` is typed `Optional(int)`.
const INLINE_REFERENCE_SEED = `id: p
name: p
input:
  m: !Optional(int)
nodes:
  loop:
    kind: loop
    times: 1
    input:
      n: \${input.m}
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output:
          n: Optional(int)
        code: 'return {n: inputs["n"]}'
    output: \${step.output}
output: \${loop.output}
`;

test("an inline body's reference seed fits its generated parameter", () => {
  const { flow } = assembleText(INLINE_REFERENCE_SEED);

  expect(flow.compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
});

test("inline body reference seed fits its generated parameter", async () => {
  expect(await runOutput(INLINE_REFERENCE_SEED, { m: 3n })).toEqual({ n: 3n });
});

// Pagination: `while:` keeps a null cursor out of the `page` def.
const PAGES = `id: p
name: p
input:
  start: !Optional(str)
nodes:
  loop:
    kind: loop
    while: cursor != null
    max: 10
    input:
      cursor: \${input.start}
      seen: 0
    call: page
output: \${loop.output}
`;

test.each(["str", "Optional(str)"])(
  "a while null check lets an optional seed fill a strict body input: cursor %s",
  (cursor) => {
    // `while:` keeps a null cursor out of the body, so the `Optional` seed fills a strict input, and
    // an `Optional` body input works as before.
    const page = handChild({
      inputs: params(`  cursor: !${cursor}\n  seen: !int\n`),
      producers: { step: record({ cursor: "Optional(str)", seen: "int" }) },
      output: "${step.output}",
    });

    const { flow } = assembleText(PAGES, "flow.yaml", { children: { page } });

    expect(flow.compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
  },
);

test("while null check lets an optional seed fill a strict body input", async () => {
  // Each body input runs with a cursor and without one.
  for (const cursor of ["str", "Optional(str)"]) {
    const page = `  page:
    input:
      cursor: !${cursor}
      seen: !int
    nodes:
      step:
        kind: code
        input:
          cursor: !str \${input.cursor}
          seen: \${input.seen}
        output:
          cursor: Optional(str)
          seen: int
        code: 'return {cursor: null, seen: inputs["seen"] + 1}'
    output: \${step.output}
`;
    const flow = await loadConformanceFlow(withDefs(PAGES, page));
    for (const [start, seen] of [
      ["c1", 1n],
      [null, 0n],
    ] as const) {
      const result = await runFlow(flow, { start });
      expect(result.status, result.failure?.message).toBe("succeeded");
      expect(result.output).toEqual({ cursor: null, seen });
    }
  }
});

/**
 * A LOOP routed by a `case` on a `Union(Buy, Hold)` producer, seeding the whole record into a body
 * that takes a `Buy`; `then` and `other` pick the branch the LOOP is on.
 */
function narrowedLoop(then: string, other: string): string {
  return `id: p
name: p
typedefs:
  Buy: {kind: 'Literal(buy)', qty: int}
  Hold: {kind: 'Literal(hold)', note: str}
input:
  s: !str
nodes:
  d:
    kind: code
    input:
      s: \${input.s}
    output: Union(Buy, Hold)
    code: 'return inputs["s"] === "b" ? {kind: "buy", qty: 2n} : {kind: "hold", note: "n"}'
  route:
    kind: case
    on: \${d.output.kind}
    cases:
      - when: buy
        then: ${then}
    else: ${other}
  loop:
    kind: loop
    times: 1
    input:
      order: \${d.output}
      k: 0
    call: body
  e:
    kind: code
    code: return 1n
output: \${loop.output.k}
`;
}

test("a loop seed reads the case-narrowed type", () => {
  const body = handChild({
    inputs: params("  order: !Buy\n  k: !int\n", readTypedefs({ Buy: { kind: "Literal(buy)", qty: "int" } })),
    producers: { step: record({ k: "int" }) },
    output: "${step.output}",
  });

  // Under `when: buy` the seed is narrowed to `Buy`, as a CALL argument would be.
  expect(loadSeeded(narrowedLoop("loop", "e"), body).compiled.nodes.get("loop")).toBeInstanceOf(LoopNode);
  const error = loadError(() => loadSeeded(narrowedLoop("e", "loop"), body));

  expect(error.message).toBe(
    "loop node 'loop': binding 'order' — child expects '{\"kind\":Literal(buy),\"qty\":int}', " +
      "source is '{\"kind\":Literal(hold),\"note\":str}'",
  );
});

test("loop seed reads the case narrowed type", async () => {
  const body = `  body:
    input:
      order: !Buy
      k: !int
    nodes:
      step:
        kind: code
        input:
          order: \${input.order}
          k: \${input.k}
        output:
          k: int
        code: 'return {k: inputs["k"] + inputs["order"]["qty"]}'
    output: \${step.output}
`;

  expect(await runOutput(withDefs(narrowedLoop("loop", "e"), body), { s: "b" })).toBe(2n);
});

// ---------- harvested conformance cases ----------

// Each case's `defs:` body or inline body is built by the loader's callable scope, which types the
// seeds.
test.each<CaseRow>(
  [
    "a_loop_without_a_driver_is_rejected",
    "body_output_field_outside_carried_record_is_rejected",
    "body_returning_a_scalar_is_rejected",
    "condition_reading_only_unreturned_fields_is_rejected",
    "float_max_is_rejected",
    "max_of_zero_is_rejected",
    "never_ending_error_names_every_unreturned_field_read",
    "predicate_field_read_on_scalar_is_rejected",
    "predicate_unknown_name_is_rejected",
    "text_max_is_rejected",
    "times_with_max_is_rejected",
    "two_drivers_are_rejected",
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
    "quoted_number_seed_for_an_int_input",
    "seed_default_null_for_an_int_input",
    "text_seed_for_an_int_input",
    // Changed (rule `dates`): Python reads the seed as a date and errors; TS reads it as text.
    "date_seed_for_a_str_input_loads_as_text",
    "omitted_field_seed_cycle_is_located",
    "partial_loop_output_rejects_an_unknown_field",
  ].map((name) => ({ id: `load-loop/engine_loop_build__${name}`, call: assembled })),
)("assemble: $id", runCase);
