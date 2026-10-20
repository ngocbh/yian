/**
 * A flow output that reads a node CASE routing may skip gets the explicit null default, so it
 * returns null when that node is skipped, and arithmetic on such a value is a load error: a port
 * of the load rows of `tests/engine/test_skipped_flow_outputs.py`, assembling each flow with
 * `assembleText` where Python calls `load_flow`.
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
 * Ported, the load-time rows (all errors but the comparison, which loads):
 * `test_inline_loop_over_an_untyped_seed_asks_for_its_type` (the loop
 * type contract raises it, since the skip makes the body field `Optional(str)`),
 * `test_arithmetic_on_a_skipped_output_is_a_load_error`,
 * `test_arithmetic_in_a_node_input_that_still_runs_is_a_load_error`,
 * `test_arithmetic_in_an_embedded_template_is_a_load_error`,
 * `test_arithmetic_in_a_case_on_is_a_load_error`,
 * `test_unary_minus_and_power_on_a_skipped_output_are_load_errors` (2),
 * `test_arithmetic_on_a_coalesce_of_routing_only_targets_is_a_load_error`,
 * `test_comparison_on_a_skipped_output_loads`, `test_hint_names_a_placeholder_default`,
 * `test_hint_replaces_a_null_default`, `test_arithmetic_in_a_searched_when_under_a_default_is_a_load_error`,
 * `test_arithmetic_on_a_null_default_in_a_searched_when_is_a_load_error`,
 * `test_arithmetic_in_a_map_over_names_map_over`, `test_arithmetic_in_a_wait_until_names_wait_until`
 * and `test_arithmetic_in_a_default_is_checked_as_if_it_always_runs`. Ported, the load halves
 * (they assemble, with no skipped-arithmetic error) of `test_arithmetic_in_a_node_that_co_skips_loads`,
 * `test_arithmetic_in_a_node_routed_by_a_case_on_the_value_loads`,
 * `test_arithmetic_on_a_node_that_depends_on_a_case_loads`, `test_arithmetic_on_a_defaulted_value_loads`,
 * `test_arithmetic_on_an_exhaustive_join_loads`, `test_arithmetic_on_an_input_loads`,
 * `test_arithmetic_on_a_node_that_cannot_be_skipped_runs`,
 * `test_arithmetic_on_a_defaulted_value_in_a_searched_when_loads`,
 * `test_plain_arithmetic_in_a_searched_when_loads`,
 * `test_isinstance_guard_in_a_searched_when_over_a_skippable_union_routes` and
 * `test_arithmetic_on_a_join_over_a_literal_input_loads`.
 *
 * The harvested `load-case/engine_skipped_flow_outputs__*` cases are pinned at the end; the MAP
 * `over:` one with its `child` def built by hand (`handChild`), since def bodies load in T3-22.
 *
 * Changed: CODE bodies are TS; load doesn't run them. The literal output `5` reads as `5n`. The
 * fingerprint row's second half stubs `null_skipped_outputs` in Python's pipeline module to
 * fingerprint the flow without the rewrite; a test can't stub a function another ES module
 * imports, so it puts the authored output source back on a fresh assembly by hand (END's edge
 * keeps the optional stance the rewrite gave it, so only the output source differs). The MAP
 * `over:` row builds its `child` def by hand. The error rows pin the whole message where Python
 * checks fragments.
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
 * `test_child_routing_on_a_literal_input_passes_to_a_str_input`, and the run halves of the rows
 * above.
 */

import { expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { type CompiledFlow, FlowOutput } from "../../src/compile/model.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

/** Return a flow text with the given `typedefs:`, `input:`, `defs:`, `nodes:` and `output:` blocks. */
function flow(
  nodes: string,
  output: string,
  options: { inputs?: string; defs?: string; typedefs?: string } = {},
): string {
  const { inputs = "approve: !bool", defs = "", typedefs = "" } = options;
  return `id: f\nname: f\n${typedefs}input:\n  ${inputs}\n${defs}nodes:\n${nodes}output:${output}\n`;
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

// A LOOP body that makes a note only when `flip` holds (a CASE with no `else:`).
const FLIP_BODY = `      gate:
        kind: case
        cases:
          - when: \${input.flip}
            then: mk
      mk:
        kind: code
        output: str
        code: |
          return "made"
      bump:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1n
`;

/** Return `text` with each non-empty line indented by `prefix`. */
const indent = (text: string, prefix: string) =>
  text
    .split("\n")
    .map((line) => (line === "" ? line : prefix + line))
    .join("\n");

/** Return a flow whose inline LOOP body returns `note: ${mk.output}`. */
function inlineLoop(noteSeed: string, times: number): string {
  const nodes =
    `  repeat:\n    kind: loop\n    times: ${times}\n    input:\n      n: 0\n` +
    `      note: ${noteSeed}\n      flip: \${input.flip}\n    nodes:\n` +
    indent(FLIP_BODY, "  ") +
    "    output:\n      n: ${bump.output}\n      note: ${mk.output}\n";
  return flow(nodes, " ${repeat.output}", { inputs: "flip: !bool" });
}

test("an inline loop over an untyped seed asks for its type", () => {
  // The body field is `Optional(str)`, so an untyped `"start"` seed must be typed.
  expect(loadError(() => assembleText(inlineLoop('"start"', 1))).message).toBe(
    "loop node 'repeat': the body returns 'note' as Optional(str), but the carried field is typed str from its " +
      'seed; declare the seed\'s type: `note: !Optional(str) "start"`',
  );
});

// then: order / else: rejected, where `order` returns `{qty: int}`
const QTY = gate("order", "rejected") + code("order", '{"qty": 2}', "\n      qty: int") + code("rejected", '"no"');

/** Return the text of the skipped-arithmetic error on `operand` (null when `producers` skip). */
function arithmetic(source: string, operand: string, producers: string, hint = `(${operand} :- <default>)`): string {
  return (
    `${source} does arithmetic on ${operand}, which is null when ${producers} skipped; ` +
    `give it a default with :-, e.g. ${hint}`
  );
}

/** Return the lines of `text` the error's innermost frame covers. */
const errorLines = (error: LoadError, text: string) => error.diagnostic.frames.at(-1)!.range.coveredLines(text);

/** Return the 1-based number of the first line of `text` that is exactly `line`. */
const lineOf = (text: string, line: string) => text.split("\n").indexOf(line) + 1;

test("arithmetic on a skipped output is a load error", () => {
  // `${order.output.qty + 1}` in an output asks for a `:-` default, at the output line.
  const text = flow(QTY, "\n  q: ${order.output.qty + 1}");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    `output 'q': ${arithmetic("${order.output.qty + 1}", "order.output.qty", "node 'order' is")}`,
  );
  expect(errorLines(error, text)).toEqual([lineOf(text, "  q: ${order.output.qty + 1}")]);
});

test("arithmetic in a node input that still runs is a load error", () => {
  // A node that still runs when `order` is skipped can't do arithmetic on its output.
  const use = code("use", 'inputs["q"]', "int", "    input:\n      q: ${order.output.qty * 2}\n");
  const text = flow(QTY + use, "\n  u: ${use.output}");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    `node 'use' input 'q': ${arithmetic("${order.output.qty * 2}", "order.output.qty", "node 'order' is")}`,
  );
  expect(errorLines(error, text)).toEqual([lineOf(text, "      q: ${order.output.qty * 2}")]);
});

test("arithmetic in an embedded template is a load error", () => {
  // Arithmetic inside surrounding text is checked too.
  const error = loadError(() => assembleText(flow(QTY, "\n  t: 'total ${order.output.qty + 1}'")));
  expect(error.message).toBe(
    `output 't': ${arithmetic("total ${order.output.qty + 1}", "order.output.qty", "node 'order' is")}`,
  );
});

test("arithmetic in a case on: is a load error", () => {
  // A CASE `on:` that does arithmetic on a skipped value is rejected, naming `on:`.
  const nodes =
    QTY +
    "  route:\n    kind: case\n    on: ${order.output.qty + 1}\n    cases:\n" +
    "      - when: 3\n        then: three\n    else: other\n" +
    code("three", '"3"') +
    code("other", '"?"');
  const error = loadError(() => assembleText(flow(nodes, "\n  v: ${three.output | other.output}")));
  expect(error.message).toBe(
    `node 'route' on: ${arithmetic("${order.output.qty + 1}", "order.output.qty", "node 'order' is")}`,
  );
});

test.each(["-order.output.qty", "order.output.qty ** 2"])(
  "unary minus and power on a skipped output are load errors: %s",
  (expr) => {
    // Unary minus and `**` are arithmetic too.
    const error = loadError(() => assembleText(flow(QTY, `\n  q: \${${expr}}`)));
    expect(error.message).toBe(`output 'q': ${arithmetic(`\${${expr}}`, "order.output.qty", "node 'order' is")}`);
  },
);

test("arithmetic on a coalesce of routing-only targets is a load error", () => {
  // `${(a.output | b.output) + 1}` is null when the CASE matches nothing.
  const nodes =
    "  gate:\n    kind: case\n    cases:\n" +
    "      - when: ${input.approve}\n        then: a\n" +
    "      - when: ${input.other}\n        then: b\n" +
    code("a", "1n", "int") +
    code("b", "2n", "int");
  const text = flow(nodes, "\n  v: ${(a.output | b.output) + 1}", { inputs: R2_INPUTS });
  expect(loadError(() => assembleText(text)).message).toBe(
    `output 'v': ${arithmetic("${(a.output | b.output) + 1}", "(a.output | b.output)", "nodes 'a' and 'b' are")}`,
  );
});

test("arithmetic in a node that co-skips loads", () => {
  // A node that also reads `${order.output}` plainly is skipped with it.
  const extra = "    input:\n      q: ${order.output.qty * 2}\n      o: ${order.output}\n";
  assembleText(flow(QTY + code("use", 'inputs["q"]', "int", extra), "\n  u: ${use.output}"));
});

test("arithmetic in a node routed by a case on the value loads", () => {
  // A node behind a CASE whose `when:` reads `order` never runs when `order` is skipped.
  const nodes =
    QTY +
    "  check:\n    kind: case\n    cases:\n" +
    "      - when: ${order.output.qty > 1}\n        then: use\n" +
    code("use", 'inputs["q"]', "int", "    input:\n      q: ${order.output.qty * 2}\n");
  assembleText(flow(nodes, "\n  u: ${use.output}"));
});

test("arithmetic on a node that depends on a case loads", () => {
  // A node with `depends_on` a CASE runs on every branch, so its output is never null.
  const note = code("note", "1n", "int", "    depends_on: [gate]\n");
  assembleText(flow(QTY + note, "\n  q: ${order.output.qty :- (note.output + 1)}"));
});

test("arithmetic on a defaulted value loads", () => {
  // `(order.output.qty :- 0) + 1` gives an explicit default.
  assembleText(flow(QTY, "\n  q: ${(order.output.qty :- 0) + 1}"));
});

test("a comparison on a skipped output loads", () => {
  // A comparison is not arithmetic.
  assembleText(flow(QTY, "\n  c: ${order.output.qty > 1}"));
});

test("arithmetic on an exhaustive join loads", () => {
  // `(pro.output | caut.output) + 1` over then/else is never null from a skip.
  const nodes = gate("pro", "caut") + code("pro", "1n", "int") + code("caut", "2n", "int");
  assembleText(flow(nodes, "\n  v: ${(pro.output | caut.output) + 1}"));
});

test("arithmetic on an input loads", () => {
  // A flow input is never skipped.
  assembleText(flow(QTY, "\n  v: ${input.x + 1}", { inputs: "approve: !bool\n  x: !int" }));
});

test("arithmetic on a node that cannot be skipped loads", () => {
  // Arithmetic over a root node in a flow with a CASE loads as before.
  const nodes = QTY + code("root", "5n", "int");
  assembleText(flow(nodes, "\n  v: ${root.output * 2 + 1}\n  w: ${-root.output}"));
});

test("the rewritten output fingerprints as the explicit null default", () => {
  // The rewrite changes the flow's fingerprint to that of the authored `:-null` form.
  const rewritten = flowFingerprint(assembleText(flow(R1, "\n  order: ${order.output}")).flow.compiled);
  const explicit = assembleText(flow(R1, "\n  order: ${order.output:-null}")).flow.compiled;
  expect(rewritten).toBe(flowFingerprint(explicit));
  // Python reloads with the rewrite switched off; an ES module import can't be swapped, so a fresh
  // assembly gets its authored output source back by hand.
  const plain = assembleText(flow(R1, "\n  order: ${order.output}")).flow.compiled;
  plain.outputs = plain.outputs.map((output) => new FlowOutput(output.name, "${order.output}"));
  expect(plain.outputs[0]!.binding.source).toBe("${order.output}");
  expect(rewritten).not.toBe(flowFingerprint(plain));
});

test("the hint names a placeholder default", () => {
  // The hint asks for a default without guessing its value.
  expect(loadError(() => assembleText(flow(QTY, "\n  q: ${order.output.qty + 1}"))).message).toEndWith(
    "give it a default with :-, e.g. (order.output.qty :- <default>)",
  );
});

test("the hint replaces a null default", () => {
  // An operand already defaulted to null gets the placeholder in place of `null`.
  expect(loadError(() => assembleText(flow(QTY, "\n  q: ${(order.output.qty :- null) + 1}"))).message).toEndWith(
    "e.g. (order.output.qty :- <default>)",
  );
});

// `a` is skippable (then: a / else: rejected); `gate2` is a searched case that reads it.
const A = gate("a", "rejected") + code("a", '{"n": 2}', "\n      n: int") + code("rejected", '"no"');

/** Return the flow where searched case `gate2` routes to `hot` when `when` holds. */
function searchedGate(when: string): string {
  const nodes =
    A +
    `  gate2:\n    kind: case\n    cases:\n      - when: "${when}"\n        then: hot\n` +
    "    else: cold\n" +
    code("hot", '"hot"') +
    code("cold", '"cold"');
  return flow(nodes, "\n  v: ${hot.output | cold.output}");
}

test("arithmetic in a searched when: under a default is a load error", () => {
  // A searched `when:` runs with `a` skipped under `:-`, so arithmetic inside is rejected.
  const when = "((a.output.n + 1) :- 0) > 1";
  const text = searchedGate(when);
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(`node 'gate2' cases[0].when: ${arithmetic(when, "a.output.n", "node 'a' is")}`);
  expect(errorLines(error, text)).toEqual([lineOf(text, `      - when: "${when}"`)]);
});

test("arithmetic on a null default in a searched when: is a load error", () => {
  // `(a.output.n :- null) + 1` in a searched `when:` asks for a real default.
  const when = "(a.output.n :- null) + 1 > 1";
  expect(loadError(() => assembleText(searchedGate(when))).message).toBe(
    `node 'gate2' cases[0].when: ${arithmetic(when, "(a.output.n :- null)", "node 'a' is", "(a.output.n :- <default>)")}`,
  );
});

test("arithmetic on a defaulted value in a searched when: loads", () => {
  // `(a.output.n :- 0) + 1` in a searched `when:` gives the default.
  assembleText(searchedGate("(a.output.n :- 0) + 1 > 1"));
});

test("plain arithmetic in a searched when: loads", () => {
  // A plain read skips `gate2` with `a`, so its arithmetic never sees a skipped null.
  assembleText(searchedGate("a.output.n + 1 > 1"));
});

const UNION_FLOW = `id: f
name: f
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int}
  Decision: Union(Buy, Sell)
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: \${input.approve}
        then: d
    else: rejected
  d:
    kind: code
    output: Decision
    code: |
      return {"action": "sell", "qty": 9n}
  rejected:
    kind: code
    output: str
    code: |
      return "no"
  gate2:
    kind: case
    cases:
      - when: "isinstance(d.output, Buy) and ((d.output.qty + 1) :- 0) > 5"
        then: hot
    else: cold
  hot:
    kind: code
    output: str
    code: |
      return "hot"
  cold:
    kind: code
    output: str
    code: |
      return "cold"
output:
  v: \${hot.output | cold.output}
`;

test("an isinstance guard in a searched when: over a skippable union loads", () => {
  // The plain `isinstance(d.output, Buy)` read skips `gate2` with `d`, so the arithmetic never sees
  // a skipped null.
  assembleText(UNION_FLOW);
});

const CHILD =
  "defs:\n  child:\n    input:\n      n: !int\n    nodes:\n      k:\n        kind: code\n" +
  "        input:\n          n: ${input.n}\n        output: int\n        code: |\n" +
  "          return inputs['n']\n    output: ${k.output}\n";

/** The `child` def built by hand: input `n: !int`, one `k: int` producer, returned. */
const HAND_CHILD = () =>
  handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n", new Map()),
    producers: { k: new ValueContract(Type.scalar(ValueKind.INTEGER)) },
    output: "${k.output}",
  });

// then: order / else: rejected, where `order` returns a list field and a str field
const ITEMS =
  gate("order", "rejected") +
  code("order", '{"items": [1n], "s": "2020"}', "\n      items: list(int)\n      s: str") +
  code("rejected", '"no"');

test("arithmetic in a map over: names map over", () => {
  // A MAP `over:` that does arithmetic on a skipped value is rejected as `map over`.
  const over = "    over: ${order.output.items + [3]}";
  const nodes = `${ITEMS}  m:\n    kind: map\n${over}\n    call: child\n    input:\n      n: \${item}\n`;
  const text = flow(nodes, "\n  r: ${rejected.output}", { defs: CHILD });
  const error = loadError(() => assembleText(text, "flow.yaml", { children: { child: HAND_CHILD() } }));
  expect(error.message).toBe(
    `node 'm' map over: ${arithmetic("${order.output.items + [3]}", "order.output.items", "node 'order' is")}`,
  );
  expect(errorLines(error, text)).toEqual([lineOf(text, over)]);
});

test("arithmetic in a wait until: names wait until", () => {
  // A WAIT `until:` that does arithmetic on a skipped value is rejected as `wait until`.
  const until = "    until: ${order.output.s + '-01-01'}";
  const text = flow(`${ITEMS}  w:\n    kind: wait\n${until}\n`, "\n  r: ${rejected.output}");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    `node 'w' wait until: ${arithmetic("${order.output.s + '-01-01'}", "order.output.s", "node 'order' is")}`,
  );
  expect(errorLines(error, text)).toEqual([lineOf(text, until)]);
});

test("arithmetic in a default is checked as if it always runs", () => {
  // `a.output :- (b.output + 1)` is rejected though `b` runs whenever `a` is skipped.
  const nodes = gate("a", "b") + code("a", "1n", "int") + code("b", "2n", "int");
  expect(loadError(() => assembleText(flow(nodes, "\n  v: ${a.output :- (b.output + 1)}"))).message).toBe(
    `output 'v': ${arithmetic("${a.output :- (b.output + 1)}", "b.output", "node 'b' is")}`,
  );
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

test("arithmetic on a join over a Literal input loads", () => {
  // The join of a fully listed Literal input is never null from a skip.
  assembleText(flow(onInput(), "\n  v: ${(a.output | b.output) + 1}", { inputs: "k: !K", typedefs: K }));
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

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/engine_skipped_flow_outputs__inline_loop_over_an_untyped_seed_asks_for_its_type", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_on_a_skipped_output", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_node_input_that_still_runs", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_case_on", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__unary_minus_on_a_skipped_output", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_on_a_coalesce_of_routing_only_targets", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__comparison_on_a_skipped_output_loads", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__hint_replaces_a_null_default", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_searched_when_under_a_default", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_wait_until", call: assembled },
  { id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_default_is_checked", call: assembled },
])("assembled: $id", runCase);

test("assembledWith: load-case/engine_skipped_flow_outputs__arithmetic_in_a_map_over", () => {
  runCase({ id: "load-case/engine_skipped_flow_outputs__arithmetic_in_a_map_over", call: assembledWith({ child: HAND_CHILD() }) });
});
