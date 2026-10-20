/**
 * A flow output that reads a node CASE routing may skip gets the explicit null default, so it
 * returns null when that node is skipped: a port of the load rows of
 * `tests/engine/test_skipped_flow_outputs.py`, assembling each flow with `assembleText` where
 * Python calls `load_flow`.
 *
 * Ported, the compiled output sources (Python's `_sources`), whose `_run` half is Later (T4):
 * `test_record_output_of_a_skipped_branch_is_null`,
 * `test_coalesce_over_a_routing_only_case_is_null`, `test_value_mode_output_is_null`,
 * `test_dotted_output_is_null`, `test_ref_default_group_over_a_routing_only_case_is_null`,
 * `test_depends_on_a_case_runs_on_every_branch`, `test_coalesce_over_an_exhaustive_case_is_unchanged`,
 * `test_case_output_of_an_exhaustive_case_is_unchanged`,
 * `test_case_output_over_a_value_entry_branch_is_unchanged`,
 * `test_case_output_over_an_inner_case_is_unchanged`,
 * `test_case_output_over_an_inner_case_and_a_value_entry_is_unchanged`,
 * `test_root_node_output_is_unchanged`, `test_reader_of_an_exhaustive_join_is_unchanged`,
 * `test_literal_output_is_unchanged`, `test_embedded_template_is_unchanged`,
 * `test_comparison_is_unchanged`, `test_on_over_a_nullable_literal_may_match_nothing` (2),
 * `test_on_over_a_literal_input` (3) and `test_map_body_routing_on_a_literal_item_is_not_rewritten`
 * (the inline MAP body's outputs). Also `test_rewritten_output_fingerprints_as_the_explicit_null_default`
 * (`flowFingerprint`).
 *
 * Changed: CODE bodies are TS; load doesn't run them. The literal output `5` reads as `5n`. The
 * fingerprint row drops its second half, which stubs `null_skipped_outputs` in the pipeline
 * module to fingerprint the flow without the rewrite: a test can't stub a function another ES
 * module imports.
 *
 * Later (T3-20), decided by the skipped-arithmetic check:
 * `test_inline_loop_over_an_untyped_seed_asks_for_its_type`,
 * `test_arithmetic_on_a_skipped_output_is_a_load_error`,
 * `test_arithmetic_in_a_node_input_that_still_runs_is_a_load_error`,
 * `test_arithmetic_in_an_embedded_template_is_a_load_error`,
 * `test_arithmetic_in_a_case_on_is_a_load_error`,
 * `test_unary_minus_and_power_on_a_skipped_output_are_load_errors`,
 * `test_arithmetic_on_a_coalesce_of_routing_only_targets_is_a_load_error`,
 * `test_comparison_on_a_skipped_output_loads`, `test_hint_names_a_placeholder_default`,
 * `test_hint_replaces_a_null_default`, `test_arithmetic_in_a_searched_when_under_a_default_is_a_load_error`,
 * `test_arithmetic_on_a_null_default_in_a_searched_when_is_a_load_error`,
 * `test_arithmetic_in_a_map_over_names_map_over`, `test_arithmetic_in_a_wait_until_names_wait_until`,
 * `test_arithmetic_in_a_default_is_checked_as_if_it_always_runs`; and the load halves of
 * `test_arithmetic_in_a_node_that_co_skips_loads`,
 * `test_arithmetic_in_a_node_routed_by_a_case_on_the_value_loads`,
 * `test_arithmetic_on_a_node_that_depends_on_a_case_loads`, `test_arithmetic_on_a_defaulted_value_loads`,
 * `test_arithmetic_on_an_exhaustive_join_loads`, `test_arithmetic_on_an_input_loads`,
 * `test_arithmetic_on_a_node_that_cannot_be_skipped_runs`,
 * `test_arithmetic_on_a_defaulted_value_in_a_searched_when_loads`,
 * `test_plain_arithmetic_in_a_searched_when_loads` and
 * `test_arithmetic_on_a_join_over_a_literal_input_loads`.
 *
 * Later (T3-22), reading a `defs:` child's own output contract:
 * `test_call_output_of_a_skipped_branch_is_optional`,
 * `test_call_output_of_a_coalesce_over_a_routing_only_case_is_optional`,
 * `test_call_output_of_a_dotted_skipped_read_is_optional`,
 * `test_passing_a_skipped_child_output_to_a_str_input_is_a_load_error`; and the `_sources` half of
 * `test_defs_child_returns_null_to_its_caller`.
 *
 * Later (T4), through `run_flow`: `test_nested_case_target_is_null_when_either_case_skips_it`,
 * `test_depends_on_a_skipped_target_is_null`, `test_then_call_target_is_null`,
 * `test_call_node_target_is_null`, `test_loop_node_target_is_null`, `test_map_node_target_is_null`,
 * `test_defs_child_returns_null_to_its_caller`, `test_typed_output_over_a_skipped_branch_fails_at_runtime`,
 * `test_optional_typed_output_over_a_skipped_branch_is_null`,
 * `test_typed_argument_narrows_a_skipped_child_output_at_runtime`,
 * `test_inline_loop_sets_a_skipped_field_to_null`,
 * `test_inline_loop_over_a_typed_str_seed_rejects_the_null_next_iteration`,
 * `test_inline_loop_over_an_optional_seed_carries_the_null`,
 * `test_named_loop_body_sets_a_skipped_field_to_null`,
 * `test_map_inline_body_returns_null_per_skipped_element`, `test_replay_returns_the_same_null`,
 * `test_isinstance_guard_in_a_searched_when_over_a_skippable_union_routes`,
 * `test_child_routing_on_a_literal_input_passes_to_a_str_input`, the run halves of the `*_loads`
 * rows above, and the `_run` halves of the ported rows.
 */

import { expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { assembleText } from "../compose/declarations.ts";

/** Return a flow text with the given `typedefs:`, `input:`, `nodes:` and `output:` blocks. */
function flow(nodes: string, output: string, options: { inputs?: string; typedefs?: string } = {}): string {
  const { inputs = "approve: !bool", typedefs = "" } = options;
  return `id: f\nname: f\n${typedefs}input:\n  ${inputs}\nnodes:\n${nodes}output:${output}\n`;
}

/** Return an inline CODE node returning the expression `value`. */
function code(node: string, value: string, output = "str", extra = ""): string {
  return `  ${node}:\n    kind: code\n    output: ${output}\n${extra}    code: |\n      return ${value}\n`;
}

/** Return a CASE that routes to `then` when `input.<when>` holds, else to `else_`. */
function gate(then: string, else_: string | null = null, name = "gate", when = "approve"): string {
  const text = `  ${name}:\n    kind: case\n    cases:\n      - when: \${input.${when}}\n        then: ${then}\n`;
  return text + (else_ !== null ? `    else: ${else_}\n` : "");
}

// then: order / else: rejected
const R1 = gate("order", "rejected") + code("order", '"filled"') + code("rejected", '"no"');
// two `when:` cases and no `else:`: neither matches when `approve` and `other` are false
const R2 =
  "  gate:\n    kind: case\n    cases:\n" +
  "      - when: ${input.approve}\n        then: a\n" +
  "      - when: ${input.other}\n        then: b\n" +
  code("a", '"A"') +
  code("b", '"B"');
const R2_INPUTS = "approve: !bool\n  other: !bool";

/** Return the assembled flow's output sources in declaration order. */
function sources(text: string): unknown[] {
  return assembleText(text).flow.compiled.outputs.map((output) => output.binding.source);
}

test("a record output of a skipped branch is null", () => {
  expect(sources(flow(R1, "\n  order: ${order.output}"))).toEqual(["${order.output:-null}"]);
});

test("a coalesce over a routing-only case is null", () => {
  const text = flow(R2, "\n  v: ${a.output | b.output}", { inputs: R2_INPUTS });
  expect(sources(text)).toEqual(["${(a.output | b.output) :- null}"]);
});

test("a value-mode output is null", () => {
  expect(sources(flow(R1, " ${order.output}"))).toEqual(["${order.output:-null}"]);
});

test("a dotted output is null", () => {
  const nodes = gate("order", "rejected") + code("order", '{"status": "filled"}', "\n      status: str") + code("rejected", '"no"');
  expect(sources(flow(nodes, "\n  s: ${order.output.status}"))).toEqual(["${order.output.status:-null}"]);
});

test("a ref default group over a routing-only case is null", () => {
  const text = flow(R2, "\n  v: ${a.output :- b.output}", { inputs: R2_INPUTS });
  expect(sources(text)).toEqual(["${(a.output :- b.output) :- null}"]);
});

test("depends_on a case runs on every branch", () => {
  // `depends_on` a CASE runs whichever branch the CASE takes, so its output needs no null.
  const nodes = R1 + code("after", '"after"', "str", "    depends_on: [gate]\n");
  expect(sources(flow(nodes, "\n  a: ${after.output}"))).toEqual(["${after.output}"]);
});

// then: pro / else: caut (always takes a branch)
const JOIN = gate("pro", "caut") + code("pro", '"pro"') + code("caut", '"caut"');

test("a coalesce over an exhaustive case is unchanged", () => {
  expect(sources(flow(JOIN, "\n  v: ${pro.output | caut.output}"))).toEqual(["${pro.output | caut.output}"]);
});

test("the case output of an exhaustive case is unchanged", () => {
  expect(sources(flow(JOIN, "\n  v: ${gate.output}"))).toEqual(["${pro.output | caut.output}"]);
});

// A CASE routing to a value entry: then: pro (`!str pro`) / else: caut (always takes a branch)
const ENTRY_JOIN = gate("pro", "caut", "route") + "  pro: !str pro\n" + code("caut", '"caut"');

test("a case output over a value entry branch is unchanged", () => {
  expect(sources(flow(ENTRY_JOIN, "\n  v: ${route.output}"))).toEqual(["${pro.output | caut.output}"]);
});

test("a case output over an inner case is unchanged", () => {
  // An exhaustive CASE whose branch is an exhaustive CASE keeps a plain coalesce.
  const nodes = gate("route", "hold", "gate", "other") + ENTRY_JOIN + code("hold", '"held"');
  const text = flow(nodes, "\n  v: ${gate.output}", { inputs: R2_INPUTS });
  expect(sources(text)).toEqual(["${pro.output | caut.output | hold.output}"]);
});

test("a case output over an inner case and a value entry is unchanged", () => {
  const nodes = gate("route", "held", "gate", "other") + "  held: !str held\n" + ENTRY_JOIN;
  const text = flow(nodes, "\n  v: ${gate.output}", { inputs: R2_INPUTS });
  expect(sources(text)).toEqual(["${pro.output | caut.output | held.output}"]);
});

test("a root node output is unchanged", () => {
  expect(sources(flow(R1 + code("root", '"r"'), "\n  r: ${root.output}"))).toEqual(["${root.output}"]);
});

test("a reader of an exhaustive join is unchanged", () => {
  // A node that reads an exhaustive join always runs.
  const extra = "    input:\n      v: ${pro.output | caut.output}\n";
  expect(sources(flow(JOIN + code("j", 'inputs["v"]', "str", extra), "\n  j: ${j.output}"))).toEqual(["${j.output}"]);
});

test("a literal output is unchanged", () => {
  expect(sources(flow(R1, "\n  k: 5"))).toEqual([5n]);
});

test("an embedded template is unchanged", () => {
  // Embedded text renders a skipped reference as an empty string, as before.
  const text = flow(R1, "\n  t: '${order.output} and ${rejected.output}'");
  expect(sources(text)).toEqual(["${order.output} and ${rejected.output}"]);
});

test("a comparison is unchanged", () => {
  // A comparison over a skipped reference is false, as before.
  expect(sources(flow(R1, '\n  c: ${order.output == "filled"}'))).toEqual(['${order.output == "filled"}']);
});

test.each(["null", '"a"'])("an on over a nullable Literal may match nothing (cls returns %s)", (value) => {
  // A null `on:` value matches no tag, so the targets' join may be skipped.
  const nodes =
    code("cls", value, "Optional(Literal(a, b))") +
    "  route:\n    kind: case\n    on: ${cls.output}\n    cases:\n" +
    "      - when: a\n        then: na\n      - when: b\n        then: nb\n" +
    code("na", '"A"') +
    code("nb", '"B"');
  expect(sources(flow(nodes, "\n  v: ${na.output | nb.output}"))).toEqual(["${(na.output | nb.output) :- null}"]);
});

test("the rewritten output fingerprints as the explicit null default", () => {
  // The rewrite changes the flow's fingerprint to that of the authored `:-null` form.
  const rewritten = flowFingerprint(assembleText(flow(R1, "\n  order: ${order.output}")).flow.compiled);
  const explicit = assembleText(flow(R1, "\n  order: ${order.output:-null}")).flow.compiled;
  expect(rewritten).toBe(flowFingerprint(explicit));
});

// The Literal input type `onInput` routes on; a two-member Literal can't sit in a tag.
const K = "typedefs:\n  K: Literal(x, y)\n";

/** Return a CASE routing on `input.k` to `a`/`b` for each tag in `tags`. */
function onInput(tags = "x y"): string {
  const cases = tags
    .split(" ")
    .map((tag, index) => `      - when: ${tag}\n        then: ${"ab"[index]}\n`)
    .join("");
  return (
    "  sw:\n    kind: case\n    on: ${input.k}\n    cases:\n" + cases + code("a", "1", "int") + code("b", "2", "int")
  );
}

test.each([
  ["x y", "k: !K", "${a.output | b.output}", "${a.output | b.output}"],
  ["x", "k: !str", "${a.output}", "${a.output:-null}"],
  ["x y", "k: !Optional(K)", "${a.output | b.output}", "${(a.output | b.output) :- null}"],
])("an on over a Literal input (tags %s, %s)", (tags, inputs, source, expected) => {
  // A fully listed non-nullable Literal input always takes a branch; a `str` or nullable input
  // may not, so its join defaults to null.
  const text = flow(onInput(tags), `\n  v: ${source}`, { inputs, typedefs: K });
  expect(sources(text)).toEqual([expected]);
});

test("a map body routing on a Literal item is not rewritten", () => {
  // A MAP body routing on a Literal element always takes a branch, so its join stays.
  const body = onInput()
    .split("\n")
    .map((line) => (line === "" ? line : `    ${line}`))
    .join("\n");
  const text = flow(
    "  each:\n    kind: map\n    over: ${input.ks}\n    input:\n      k: ${item}\n" +
      `    nodes:\n${body}    output: \${a.output | b.output}\n`,
    " ${each.output}",
    { inputs: "ks: !list(K)", typedefs: K },
  );
  const each = assembleText(text).flow.compiled.nodes.get("each") as MapNode;
  const child = each.child as CompiledFlow;
  expect(child.outputs.map((output) => output.binding.source)).toEqual(["${a.output | b.output}"]);
});
