/**
 * The CASE `on:` form's typed `when:` values: the lowering rows of
 * `tests/engine/test_case_on_typed.py`. Each `when:` value lowers to a condition that compares it
 * exactly with the `on:` value: text as text, a number numerically and never with a bool, a bool
 * only with a bool.
 *
 * Ported: `test_lowered_conditions`, `test_large_and_small_floats_lower_exactly`,
 * `test_non_finite_when_is_a_load_error` (flipped), `test_list_or_record_when_is_a_load_error`
 * and `test_on_match_key` (adapted).
 *
 * Changed: the flow rows assemble the flow (`assembleText`), where Python loads it.
 * `test_large_and_small_floats_lower_exactly` builds an `on:` case holding the value, where Python
 * calls `_match_condition`, which stays module-private; its values are those TS YAML can read too,
 * but YAML reads `-0.0` as the float `0` (lowered `0.0`; yian has no negative zero), so the `-0.0`
 * value is a constructed one. `.inf` and `-.inf` are float values (decision 34): a `when: .inf`
 * lowers to `__on == inf ...` and routes, where Python refuses it; `.nan` is refused by the YAML
 * layer ("NaN is not a yian value"), not by the CASE build. `onMatchKey` returns a string, and
 * the date half of `test_on_match_key` is dropped (TS YAML reads an unquoted date as text); a
 * bigint keeps its digits, so `2n ** 53n + 1n` and the float `2 ** 53` keep distinct keys.
 *
 * TS-only: the exact spellings of the float lowering (positional, no exponent, `.0` when
 * integral, `inf`), a YAML `when: -0` (the int `0n`) lowering to `0`, and an unquoted datetime
 * `when:` lowering to its text.
 *
 * Dropped: `test_unquoted_datetime_when_is_a_load_error` (TS YAML reads an unquoted datetime as
 * text, so it matches that text).
 *
 * Later (T4): `test_int_on_matches_int_when`, `test_negative_int_when`,
 * `test_float_on_matches_numerically`, `test_bool_on_matches_bool_when`,
 * `test_quoted_when_is_text`, `test_no_else_int_on_routes_at_runtime`,
 * `test_bool_never_equals_a_number_on_an_untyped_on`, `test_date_when_is_iso_text`,
 * `test_untyped_on_compares_exactly`, `test_date_when_on_a_date_input`,
 * `test_unquoted_1e3_when_is_a_float`, `test_bool_value_case_with_true_and_false_needs_no_else`,
 * `test_routing_only_bool_join_is_exhaustive`,
 * `test_literal_value_case_listing_every_tag_needs_no_else`,
 * `test_unquoted_date_when_narrows_a_date_shaped_union_tag`.
 *
 * Later (T3-20, `case_exhaustive`): `test_int_when_on_str_is_a_migration_error`,
 * `test_text_when_on_int_is_an_error`, `test_bool_when_on_int_is_an_error`,
 * `test_int_when_on_bool_is_an_error`, `test_float_when_on_int_is_an_error`,
 * `test_out_of_bounds_int_when`, `test_typo_next_to_else_on_a_literal_node_output`,
 * `test_typo_on_a_literal_flow_input`, `test_int_when_on_a_literal_is_an_error`,
 * `test_quote_fix_on_a_literal_with_a_numeric_tag`,
 * `test_tags_that_read_as_other_values_are_listed_quoted`, `test_unknown_on_type_is_not_checked`,
 * `test_object_on_type_is_not_checked`, `test_optional_on_type_checks_the_non_null_part`,
 * `test_bool_when_on_str_quote_fix_keeps_spelling`, `test_integral_float_when_on_int`,
 * `test_no_drop_quotes_hint_for_text_yaml_reads_as_text`, `test_former_boolean_word_on_a_bool`,
 * `test_invalid_date_text_hint_does_not_crash`, `test_no_drop_quotes_for_null_or_nonfinite_text`,
 * `test_unquoted_date_on_int_is_called_a_date`, `test_expression_on_is_typed`,
 * `test_quote_fix_skips_an_anchor_or_tag`, `test_drop_quotes_fix_needs_quotes_and_strips_the_text`,
 * `test_malformed_date_text_on_a_date_is_not_a_value`, `test_repeated_int_when_is_a_load_error`,
 * `test_int_and_float_when_are_duplicates`,
 * `test_int_and_float_duplicate_reported_before_the_int_fix`,
 * `test_float_then_int_duplicate_reported_before_the_int_fix`, `test_duplicate_on_an_untyped_on`,
 * `test_duplicate_text_when`, `test_date_and_its_iso_text_are_duplicates`,
 * `test_bool_number_and_text_are_distinct`, `test_duplicate_next_to_else_and_same_target`,
 * `test_duplicate_message_without_a_source_range`,
 * `test_duplicate_through_an_alias_shows_the_spelling`,
 * `test_routing_only_literal_flow_input_missing_a_tag`,
 * `test_routing_only_literal_flow_input_with_else_loads`,
 * `test_routing_only_literal_flow_input_listing_every_tag_loads`. Later (T3-20, T3-17): the inline
 * body rows `test_loop_body_carried_field`, `test_map_child_input`,
 * `test_routing_only_loop_carried_literal_missing_a_tag` and
 * `test_routing_only_map_child_literal_input_missing_a_tag`, and the value-case rows
 * `test_bool_value_case_missing_false_is_not_exhaustive`, `test_optional_bool_is_not_exhaustive`
 * and `test_int_value_case_reason_names_literal_or_bool`. Later (T3-20, T3-22):
 * `test_routing_only_call_child_literal_input_missing_a_tag`.
 *
 * The harvested case the CASE build decides is pinned at the end.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor, onMatchKey } from "../../src/compose/nodes/case.ts";
import { evaluateWhenRecord } from "../../src/expr/expressions.ts";
import type { CaseNode } from "../../src/nodes/case/node.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { assembled, assembleText, loadError, nodesOf, runCase } from "../compose/declarations.ts";

/** A build context with no typedefs; a CASE build reads none. */
const CTX = new BuildContext({ registry: new Map() });

/** The type table `isinstance(__on, bool)` reads. */
const TYPES = new Map([["bool", typeFor("bool", new Map())]]);

/** Return the 1-based number of the first line of `text` holding `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Return the start line of a load error's innermost frame. */
function errorLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return a CODE node `name` that returns its own name. */
function branch(name: string): string {
  return `  ${name}:\n    kind: code\n    code: |\n      return "${name}"\n    output: str\n`;
}

/**
 * Return a flow whose `route` case matches `on:` against `whens` (`[when, then]` pairs).
 *
 * With `onType`, `on:` reads the flow input `n: !<onType>`; with `null` it reads a CODE producer
 * `src` over the input `pick`.
 */
function flow(
  whens: readonly (readonly [string, string])[],
  options: { onType?: string | null; else_?: string | null } = {},
): string {
  const { onType = "int", else_ = "other" } = options;
  const targets = [...new Set(whens.map(([, then]) => then))];
  if (else_ !== null) targets.push(else_);
  let text: string;
  let on: string;
  if (onType !== null) {
    text = `id: f\nname: f\ninput:\n  n: !${onType}\nnodes:\n`;
    on = "${input.n}";
  } else {
    text =
      "id: f\nname: f\ninput:\n  pick: !str\nnodes:\n  src:\n    kind: code\n    input:\n" +
      '      p: ${input.pick}\n    code: |\n      return True if inputs["p"] == "t" else 1\n';
    on = "${src.output}";
  }
  text += `  route:\n    kind: case\n    on: ${on}\n    cases:\n`;
  for (const [when, then] of whens) text += `      - when: ${when}\n        then: ${then}\n`;
  if (else_ !== null) text += `    else: ${else_}\n`;
  text += targets.map(branch).join("");
  return `${text}output: \${${targets.map((name) => `${name}.output`).join(" | ")}}\n`;
}

/** Return the `when:` conditions of the built `route` of flow `text`, assembled. */
function loweredWhens(text: string): (string | null)[] {
  const { flow: composed } = assembleText(text);
  return (composed.compiled.nodes.get("route") as CaseNode).cases.map((branch) => branch.when);
}

/** Return the condition an `on:` case lowers one `when:` value to. */
function lowered(value: unknown): string {
  const desc = new CaseDescriptor({ id: "route", on: "${input.n}", cases: [{ when: value, then: "one" }] });
  return (desc.build(CTX).node as CaseNode).cases[0]!.when!;
}

test("lowered conditions", () => {
  const text = flow(
    [
      ["1", "one"],
      ["1.5", "two"],
      ["true", "three"],
      ['"x"', "four"],
    ],
    { onType: null },
  );

  expect(loweredWhens(text)).toEqual([
    "__on == 1 and not isinstance(__on, bool)",
    "__on == 1.5 and not isinstance(__on, bool)",
    "__on == true and isinstance(__on, bool)",
    '__on == "x"',
  ]);
});

test.each([1e20, 1.5e-7, -2.5, -0, 5e-324])("large and small floats lower exactly: %p", (value) => {
  const condition = lowered(value);

  expect(evaluateWhenRecord(condition, { __on: value }, TYPES)).toBe(true);
  if (value * 2 !== value) expect(evaluateWhenRecord(condition, { __on: value * 2 }, TYPES)).toBe(false);
});

test.each<[bigint | number, string]>([
  [1e20, "100000000000000000000.0"],
  [1e21, "1000000000000000000000.0"],
  [1.5e-7, "0.00000015"],
  [-2.5, "-2.5"],
  [0.001, "0.001"],
  [2, "2.0"],
  [0, "0.0"],
  [-0, "-0.0"],
  [5e-324, `0.${"0".repeat(323)}5`],
  [Infinity, "inf"],
  [-Infinity, "-inf"],
  [1n, "1"],
  [-5n, "-5"],
  [2n ** 64n, "18446744073709551616"],
])("a number lowers positionally: %p", (value, text) => {
  expect(lowered(value)).toBe(`__on == ${text} and not isinstance(__on, bool)`);
});

test.each([
  ["1e20", "100000000000000000000.0"],
  // YAML reads `-0` as the int 0, and `-0.0` as the float 0: yian has no negative zero.
  ["-0", "0"],
  ["-0.0", "0.0"],
  ["5e-324", `0.${"0".repeat(323)}5`],
])("a YAML number lowers positionally: when: %s", (when, text) => {
  expect(loweredWhens(flow([[when, "one"]], { onType: "float" }))).toEqual([
    `__on == ${text} and not isinstance(__on, bool)`,
  ]);
});

test.each([
  [".inf", "inf", Infinity],
  ["-.inf", "-inf", -Infinity],
])("an infinite when lowers and routes: when: %s", (authored, shown, value) => {
  const [condition] = loweredWhens(flow([[authored, "one"]], { onType: "float" }));

  expect(condition).toBe(`__on == ${shown} and not isinstance(__on, bool)`);
  expect(evaluateWhenRecord(condition!, { __on: value }, TYPES)).toBe(true);
  expect(evaluateWhenRecord(condition!, { __on: 1.5 }, TYPES)).toBe(false);
});

test("a NaN when is refused by the YAML layer", () => {
  const text = flow([[".nan", "one"]], { onType: "float" });

  const error = loadError(() => loweredWhens(text));

  expect(error.message).toBe("invalid YAML: NaN is not a yian value; quote it to keep it as text");
  expect(errorLine(error)).toBe(lineOf(text, "when: .nan"));
});

test.each([
  ["[a]", "a list"],
  ["{a: 1}", "a record"],
])("a list or record when is a load error: %s", (authored, kind) => {
  const text = flow([[authored, "one"]]);

  const error = loadError(() => loweredWhens(text));

  expect(error.message).toBe(`case node 'route': an on: case's when: takes text, a number, or a bool, got ${kind}`);
  expect(errorLine(error)).toBe(lineOf(text, `when: ${authored}`));
});

test("an unquoted datetime when lowers to its text", () => {
  // TS YAML reads an unquoted timestamp as text, so it matches that text exactly.
  expect(loweredWhens(flow([["2026-01-02T10:00:00Z", "one"]], { onType: "str" }))).toEqual([
    '__on == "2026-01-02T10:00:00Z"',
  ]);
});

test("onMatchKey", () => {
  expect(onMatchKey(1n)).toBe(onMatchKey(1.0));
  expect(onMatchKey(true)).not.toBe(onMatchKey(1n));
  expect(onMatchKey("1")).not.toBe(onMatchKey(1n));
  expect(onMatchKey(false)).not.toBe(onMatchKey(0n));
  expect(new Set([onMatchKey(1n), onMatchKey(1.0)]).size).toBe(1);
  expect(onMatchKey(-0)).toBe(onMatchKey(0n));
  expect(onMatchKey(2n ** 53n + 1n)).not.toBe(onMatchKey(2 ** 53));
  expect(onMatchKey(1.5)).not.toBe(onMatchKey(1n));
  expect(onMatchKey(Infinity)).not.toBe(onMatchKey(-Infinity));
});

// ---------- harvested conformance cases ----------

test("assembled: load-case/engine_case_on_typed__infinite_when_on_a_float", () => {
  runCase({ id: "load-case/engine_case_on_typed__infinite_when_on_a_float", call: assembled });
});
