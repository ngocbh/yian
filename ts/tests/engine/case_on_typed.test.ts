/**
 * The CASE `on:` form's typed `when:` values: the lowering and `when:` value rows of
 * `tests/engine/test_case_on_typed.py`. Each `when:` value lowers to a condition that compares it
 * exactly with the `on:` value: text as text, a number numerically and never with a bool, a bool
 * only with a bool. Where the `on:` type is known, a `when:` value that can never be a member of
 * it is a located load error naming the value, the type and the fix, and a `when:` value equal to
 * an earlier one is a load error at the later one.
 *
 * Ported: `test_lowered_conditions`, `test_large_and_small_floats_lower_exactly`,
 * `test_non_finite_when_is_a_load_error` (flipped), `test_list_or_record_when_is_a_load_error`
 * and `test_on_match_key` (adapted); the `when:` value rows `test_int_when_on_str_is_a_migration_error`,
 * `test_text_when_on_int_is_an_error`, `test_bool_when_on_int_is_an_error`,
 * `test_int_when_on_bool_is_an_error`, `test_float_when_on_int_is_an_error`,
 * `test_out_of_bounds_int_when`, `test_typo_next_to_else_on_a_literal_node_output`,
 * `test_typo_on_a_literal_flow_input`, `test_int_when_on_a_literal_is_an_error`,
 * `test_quote_fix_on_a_literal_with_a_numeric_tag`,
 * `test_tags_that_read_as_other_values_are_listed_quoted`, `test_map_child_input`,
 * `test_unknown_on_type_is_not_checked`, `test_object_on_type_is_not_checked`,
 * `test_optional_on_type_checks_the_non_null_part`,
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
 * `test_duplicate_through_an_alias_shows_the_spelling` and `test_loop_body_carried_field`; and the
 * exhaustiveness rows `test_routing_only_literal_flow_input_missing_a_tag`,
 * `test_routing_only_literal_flow_input_with_else_loads`,
 * `test_routing_only_literal_flow_input_listing_every_tag_loads`,
 * `test_bool_value_case_missing_false_is_not_exhaustive`, `test_optional_bool_is_not_exhaustive`,
 * `test_int_value_case_reason_names_literal_or_bool`,
 * `test_routing_only_loop_carried_literal_missing_a_tag` and
 * `test_routing_only_map_child_literal_input_missing_a_tag`.
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
 * The `".inf"` row of `test_no_drop_quotes_for_null_or_nonfinite_text` gets "drop the quotes:
 * `when: .inf`", since `.inf` is a float value. `test_unquoted_date_on_int_is_called_a_date` reads
 * "`when: 2026-01-02` is text": TS YAML reads an unquoted date as text, so
 * `test_date_and_its_iso_text_are_duplicates` holds two equal texts (the message is Python's). The
 * tagged half of `test_quote_fix_skips_an_anchor_or_tag` suggests `when: "1"`: `!!float 1` is the
 * float `1`, spelled `1` as JSON spells it, where Python spells `1.0`.
 * `test_duplicate_message_without_a_source_range` calls the exported `duplicateWhenMessage` (Python's
 * `_duplicate_when_message`). An uncovered tag list is written in the text form, `["b"]` (Python
 * `['b']`).
 *
 * TS-only: the exact spellings of the float lowering (positional, no exponent, `.0` when
 * integral, `inf`), a YAML `when: -0` (the int `0n`) lowering to `0`, and an unquoted datetime
 * `when:` lowering to its text.
 *
 * Dropped: `test_unquoted_datetime_when_is_a_load_error` (TS YAML reads an unquoted datetime as
 * text, so it matches that text).
 *
 * Later (T4-8 follow-up): `test_int_on_matches_int_when`, `test_negative_int_when`,
 * `test_float_on_matches_numerically`, `test_bool_on_matches_bool_when`,
 * `test_quoted_when_is_text`, `test_no_else_int_on_routes_at_runtime`,
 * `test_bool_never_equals_a_number_on_an_untyped_on`, `test_date_when_is_iso_text`,
 * `test_untyped_on_compares_exactly`, `test_date_when_on_a_date_input`,
 * `test_unquoted_1e3_when_is_a_float`, `test_bool_value_case_with_true_and_false_needs_no_else`,
 * `test_routing_only_bool_join_is_exhaustive`,
 * `test_literal_value_case_listing_every_tag_needs_no_else`,
 * `test_unquoted_date_when_narrows_a_date_shaped_union_tag`.
 *
 * Ported through `loadFlow`: `test_routing_only_call_child_literal_input_missing_a_tag`, which
 * composes a `defs:` entry; its harvested case is pinned at the end with `full` (the definition
 * frame and the loader's `defs entry 'child': ` prefix included).
 *
 * The harvested cases the CASE build, the `when:` value check and the exhaustiveness check decide
 * are pinned at the end.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor, onMatchKey } from "../../src/compose/nodes/case.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { duplicateWhenMessage } from "../../src/compose/validate/case_exhaustive.ts";
import { evaluateWhenRecord } from "../../src/expr/expressions.ts";
import type { CaseNode } from "../../src/nodes/case/node.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  loadErrorAsync,
  nodesOf,
  runCase,
} from "../compose/declarations.ts";

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
 * Return the `input:` section declaring `n: !<onType>`.
 *
 * A type a tag can't hold (one with a comma, space or quote) is named `N` in `typedefs:` and
 * tagged by that name.
 */
function declareN(onType: string): string {
  if (/[, "']/.test(onType)) return `typedefs:\n  N: ${onType}\ninput:\n  n: !N\n`;
  return `input:\n  n: !${onType}\n`;
}

/**
 * Return a flow whose `route` case matches `on:` against `whens` (`[when, then]` pairs).
 *
 * With `onType`, `on:` reads the flow input `n: !<onType>`; with `null` it reads a CODE producer
 * `src` that runs `srcCode` over the input `pick`, typed by `srcOutput` (untyped when `null`).
 */
function flow(
  whens: readonly (readonly [string, string])[],
  options: { onType?: string | null; else_?: string | null; srcCode?: string; srcOutput?: string | null } = {},
): string {
  const {
    onType = "int",
    else_ = "other",
    srcCode = 'return True if inputs["p"] == "t" else 1',
    srcOutput = null,
  } = options;
  const targets = [...new Set(whens.map(([, then]) => then))];
  if (else_ !== null) targets.push(else_);
  let text: string;
  let on: string;
  if (onType !== null) {
    text = `id: f\nname: f\n${declareN(onType)}nodes:\n`;
    on = "${input.n}";
  } else {
    text =
      "id: f\nname: f\ninput:\n  pick: !str\nnodes:\n  src:\n    kind: code\n    input:\n" +
      `      p: \${input.pick}\n    code: |\n      ${srcCode}\n`;
    if (srcOutput !== null) text += `    output: ${srcOutput}\n`;
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

// ---------- when: values ----------

/** Return the message of the load error `text` raises, asserting it is at `when: <when>`. */
function whenError(text: string, when: string): string {
  const error = loadError(() => assembleText(text));
  expect(errorLine(error)).toBe(lineOf(text, `when: ${when}`));
  return error.message;
}

/** Return the canonical label of the type `text` names. */
function label(text: string): string {
  return canonicalTypeLabel(typeFor(text, new Map()));
}

test("int when on str is a migration error", () => {
  const text = flow([["1", "one"]], { onType: "str" });

  expect(whenError(text, "1")).toBe(
    "case node 'route': `when: 1` is an int but `on: ${input.n}` is `str`; " + 'quote it: `when: "1"`',
  );
});

test("text when on int is an error", () => {
  const text = flow([['"1"', "one"]]);

  expect(whenError(text, '"1"')).toBe(
    "case node 'route': `when: \"1\"` is text but `on: ${input.n}` is `int`; drop the quotes: `when: 1`",
  );
});

test("bool when on int is an error", () => {
  const text = flow([["true", "one"]]);

  expect(whenError(text, "true")).toBe("case node 'route': `when: true` is a bool but `on: ${input.n}` is `int`");
});

test("int when on bool is an error", () => {
  const text = flow([["1", "one"]], { onType: "bool" });

  expect(whenError(text, "1")).toBe("case node 'route': `when: 1` is an int but `on: ${input.n}` is `bool`");
});

test("float when on int is an error", () => {
  const text = flow([["1.5", "one"]]);

  expect(whenError(text, "1.5")).toBe("case node 'route': `when: 1.5` is a float but `on: ${input.n}` is `int`");
});

test("out of bounds int when", () => {
  const text = flow(
    [
      ["1", "one"],
      ["13", "two"],
    ],
    { onType: "int(1..12)" },
  );

  expect(whenError(text, "13")).toBe(
    "case node 'route': `when: 13` is not a value of `on: ${input.n}` (`int(1..12)`): " +
      "13 is outside inclusive int bounds (1, 12)",
  );
});

test("typo next to else on a literal node output", () => {
  const text = flow(
    [
      ["go", "one"],
      ["nogo", "two"],
    ],
    { onType: null, srcCode: 'return "go"', srcOutput: "Literal(go, no_go)" },
  );

  expect(whenError(text, "nogo")).toBe(
    `case node 'route': \`when: nogo\` is not a tag of \`on: \${src.output}\` (\`${label("Literal(go, no_go)")}\`); ` +
      "use one of: go, no_go",
  );
});

test("typo on a literal flow input", () => {
  const text = flow(
    [
      ["a", "one"],
      ["c", "two"],
    ],
    { onType: "Literal(a, b)" },
  );

  expect(whenError(text, "c")).toBe(
    `case node 'route': \`when: c\` is not a tag of \`on: \${input.n}\` (\`${label("Literal(a, b)")}\`); ` +
      "use one of: a, b",
  );
});

test("int when on a literal is an error", () => {
  const text = flow([["1", "one"]], { onType: "Literal(a, b)" });

  expect(whenError(text, "1")).toBe(
    `case node 'route': \`when: 1\` is an int but \`on: \${input.n}\` is \`${label("Literal(a, b)")}\``,
  );
});

test("quote fix on a literal with a numeric tag", () => {
  const text = flow([["1", "one"]], { onType: 'Literal("1", "2")' });

  expect(whenError(text, "1")).toBe(
    `case node 'route': \`when: 1\` is an int but \`on: \${input.n}\` is \`${label('Literal("1", "2")')}\`; ` +
      'quote it: `when: "1"`',
  );
});

test("tags that read as other values are listed quoted", () => {
  const text = flow([['"3"', "one"]], { onType: 'Literal("1", "2")' });

  expect(whenError(text, '"3"')).toBe(
    `case node 'route': \`when: "3"\` is not a tag of \`on: \${input.n}\` (\`${label('Literal("1", "2")')}\`); ` +
      'use one of: "1", "2"',
  );
});

/** Return an inline-body CODE node `name` returning `value` as `output`, indented six. */
function branchBody(name: string, output: string, value: string): string {
  return `      ${name}:\n        kind: code\n        code: |\n          return ${value}\n        output: ${output}\n`;
}

test("map child input", () => {
  const text =
    "id: f\nname: f\ninput:\n  xs: !list(int)\nnodes:\n" +
    "  each:\n    kind: map\n    over: ${input.xs}\n    input:\n      x: ${item}\n    nodes:\n" +
    "      route:\n        kind: case\n        on: ${input.x}\n        cases:\n" +
    '          - when: "1"\n            then: one\n        else: other\n' +
    branchBody("one", "str", '"one"') +
    branchBody("other", "str", '"other"') +
    "    output: ${one.output | other.output}\noutput: ${each.output}\n";

  expect(whenError(text, '"1"')).toEndWith(
    "case node 'route': `when: \"1\"` is text but `on: ${input.x}` is `int`; drop the quotes: `when: 1`",
  );
});

test("loop body carried field", () => {
  const text =
    "id: f\nname: f\ninput:\n  s: !int\nnodes:\n" +
    "  rep:\n    kind: loop\n    times: 2\n    input:\n      n: !int 0\n    nodes:\n" +
    "      route:\n        kind: case\n        on: ${input.n}\n        cases:\n" +
    '          - when: "1"\n            then: one\n        else: other\n' +
    branchBody("one", "int", "2") +
    branchBody("other", "int", "1") +
    "    output:\n      n: ${one.output | other.output}\noutput: ${rep.output}\n";

  expect(whenError(text, '"1"')).toEndWith(
    "case node 'route': `when: \"1\"` is text but `on: ${input.n}` is `int`; drop the quotes: `when: 1`",
  );
});

test("unknown on type is not checked", () => {
  assembleText(
    flow(
      [
        ["1", "one"],
        ['"a"', "two"],
        ["true", "three"],
      ],
      { onType: null },
    ),
  );
});

test("object on type is not checked", () => {
  assembleText(flow([["1", "one"]], { onType: "object" }));
});

test("optional on type checks the non-null part", () => {
  assembleText(flow([["1", "one"]], { onType: "Optional(int)" }));
  const text = flow([['"1"', "one"]], { onType: "Optional(int)" });

  expect(whenError(text, '"1"')).toContain("`Optional(int)`");
});

test("bool when on str quote fix keeps spelling", () => {
  const title = flow([["True", "one"]], { onType: "str" });
  const lower = flow([["true", "one"]], { onType: "str" });

  expect(whenError(title, "True")).toEndWith('; quote it: `when: "True"`');
  expect(whenError(lower, "true")).toEndWith('; quote it: `when: "true"`');
});

test("integral float when on int", () => {
  const text = flow([["1.0", "one"]]);

  expect(whenError(text, "1.0")).toBe(
    "case node 'route': `when: 1.0` is a float but `on: ${input.n}` is `int`; write it as an int: `when: 1`",
  );
});

test("no drop-quotes hint for text YAML reads as text", () => {
  const text = flow([['"1_000"', "one"]]);

  expect(whenError(text, '"1_000"')).toBe(
    "case node 'route': `when: \"1_000\"` is text but `on: ${input.n}` is `int`",
  );
});

test.each([
  ["off", "bool", "`off` is text; write `false`"],
  ['"yes"', "Optional(bool)", "`yes` is text; write `true`"],
])("former boolean word on a bool: when: %s", (when, onType, fix) => {
  const text = flow([[when, "one"]], { onType });

  expect(whenError(text, when)).toBe(
    `case node 'route': \`when: ${when}\` is text but \`on: \${input.n}\` is \`${onType}\`; ${fix}`,
  );
});

test("invalid date text hint does not crash", () => {
  const text = flow([['"2026-13-01"', "one"]]);

  expect(whenError(text, '"2026-13-01"')).toBe(
    "case node 'route': `when: \"2026-13-01\"` is text but `on: ${input.n}` is `int`",
  );
});

test.each([
  ['""', "Optional(int)"],
  ['"null"', "Optional(int)"],
])("no drop-quotes fix for null text: when: %s", (when, onType) => {
  const message = whenError(flow([[when, "one"]], { onType }), when);

  expect(message).toContain(`\`when: ${when}\` is text but \`on: \${input.n}\` is \`${onType}\``);
  expect(message).not.toContain("drop the quotes");
});

test("infinity text on a float gets the drop-quotes fix", () => {
  // `.inf` is a float value, so dropping the quotes gives a member.
  const text = flow([['".inf"', "one"]], { onType: "float" });

  expect(whenError(text, '".inf"')).toBe(
    "case node 'route': `when: \".inf\"` is text but `on: ${input.n}` is `float`; drop the quotes: `when: .inf`",
  );
});

test("unquoted date on int is called text", () => {
  const text = flow([["2026-01-02", "one"]]);

  expect(whenError(text, "2026-01-02")).toBe(
    "case node 'route': `when: 2026-01-02` is text but `on: ${input.n}` is `int`",
  );
});

test("expression on is typed", () => {
  const text = flow([['"x"', "one"]]).replace("on: ${input.n}", "on: ${input.n > 1}");
  const untyped = flow([['"2"', "one"]]).replace("on: ${input.n}", "on: ${input.n + 1}");

  expect(whenError(text, '"x"')).toBe("case node 'route': `when: \"x\"` is text but `on: ${input.n > 1}` is `bool`");
  assembleText(untyped);
});

test("quote fix skips an anchor or tag", () => {
  const anchored = flow([["&a 1", "one"]], { onType: "str" });
  const tagged = flow([["!!float 1", "one"]], { onType: "str" });

  expect(whenError(anchored, "&a 1")).toEndWith('; quote it: `when: "1"`');
  // JSON spells the float `1` as `1`.
  expect(whenError(tagged, "!!float 1")).toEndWith('; quote it: `when: "1"`');
});

test("drop-quotes fix needs quotes and strips the text", () => {
  const tagged = flow([["!!str 1", "one"]]);
  const padded = flow([['" 1"', "one"]]);

  expect(whenError(tagged, "!!str 1")).toBe(
    "case node 'route': `when: !!str 1` is text but `on: ${input.n}` is `int`",
  );
  expect(whenError(padded, '" 1"')).toEndWith("; drop the quotes: `when: 1`");
});

test("malformed date text on a date is not a value", () => {
  const text = flow([['"2026-1-2"', "one"]], { onType: "date" });

  expect(whenError(text, '"2026-1-2"')).toStartWith(
    "case node 'route': `when: \"2026-1-2\"` is not a value of `on: ${input.n}` (`date`): ",
  );
});

// ---------- duplicate when: values ----------

/** Return the 1-based number of the `n`-th (1-based) line of `text` holding `needle`. */
function nthLine(text: string, needle: string, n: number): number {
  const lines = text
    .split("\n")
    .map((line, index) => [line, index + 1] as const)
    .filter(([line]) => line.includes(needle));
  return lines[n - 1]![1];
}

/** The message for `when: <later>` repeating `when: <first>` first written at `line`. */
function duplicate(later: string, first: string, line: number): string {
  return (
    `case node 'route': \`when: ${later}\` duplicates \`when: ${first}\` (first at line ${line}), ` +
    "so it never matches; remove one"
  );
}

test("repeated int when is a load error", () => {
  const text = flow(
    [
      ["1", "one"],
      ["1", "two"],
    ],
    { onType: "int" },
  );

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(duplicate("1", "1", nthLine(text, "when: 1", 1)));
  expect(errorLine(error)).toBe(nthLine(text, "when: 1", 2));
});

test("int and float when are duplicates", () => {
  const text = flow(
    [
      ["1", "one"],
      ["1.0", "two"],
    ],
    { onType: "float" },
  );

  expect(whenError(text, "1.0")).toBe(duplicate("1.0", "1", lineOf(text, "when: 1")));
});

test("int and float duplicate reported before the int fix", () => {
  const text = flow(
    [
      ["1", "one"],
      ["1.0", "two"],
    ],
    { onType: "int" },
  );

  expect(whenError(text, "1.0")).toBe(duplicate("1.0", "1", lineOf(text, "when: 1")));
});

test("float then int duplicate reported before the int fix", () => {
  const text = flow(
    [
      ["1.0", "one"],
      ["1", "two"],
    ],
    { onType: "int" },
  );

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(duplicate("1", "1.0", lineOf(text, "when: 1.0")));
  expect(errorLine(error)).toBe(nthLine(text, "when: 1", 2));
});

test("duplicate on an untyped on", () => {
  const text = flow(
    [
      ["2", "one"],
      ["2.0", "two"],
    ],
    { onType: null },
  );

  expect(whenError(text, "2.0")).toBe(duplicate("2.0", "2", lineOf(text, "when: 2")));
});

test("duplicate text when", () => {
  const text = flow(
    [
      ["a", "one"],
      ['"a"', "two"],
      ["b", "three"],
    ],
    { onType: "Literal(a, b)" },
  );

  expect(whenError(text, '"a"')).toBe(duplicate('"a"', "a", lineOf(text, "when: a")));
});

test("date and its ISO text are duplicates", () => {
  const text = flow(
    [
      ["2026-01-02", "one"],
      ['"2026-01-02"', "two"],
    ],
    { onType: "str" },
  );

  expect(whenError(text, '"2026-01-02"')).toBe(
    duplicate('"2026-01-02"', "2026-01-02", lineOf(text, "when: 2026-01-02")),
  );
});

test("bool, number and text are distinct", () => {
  assembleText(
    flow(
      [
        ["1", "one"],
        ["true", "two"],
        ['"1"', "three"],
      ],
      { onType: null },
    ),
  );
});

test("duplicate next to else and same target", () => {
  const text = flow(
    [
      ["3", "one"],
      ["3", "one"],
    ],
    { onType: "int" },
  );

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(duplicate("3", "3", nthLine(text, "when: 3", 1)));
  expect(errorLine(error)).toBe(nthLine(text, "when: 3", 2));
});

test("duplicate message without a source range", () => {
  const message = duplicateWhenMessage(1n, null, 1n, null, { firstRange: null, firstIndex: 0 });

  expect(message).toEndWith("(first at cases[0]), so it never matches; remove one");
});

test("duplicate through an alias shows the spelling", () => {
  const text = flow(
    [
      ["&a 1", "one"],
      ["*a", "two"],
    ],
    { onType: "int" },
  );

  expect(loadError(() => assembleText(text)).message).toBe(duplicate("1", "1", lineOf(text, "when: &a 1")));
});

// ---------- value-producing cases ----------

/** The message for a value-producing case `route` that is not exhaustive, for `reason`. */
function notExhaustive(reason: string): string {
  return `case node 'route' produces a value, so it must be exhaustive: add an \`else:\` (${reason})`;
}

/**
 * Return a flow whose read case `route` routes `${input.n}` per `whens`, no else.
 *
 * Each `[when, value]` routes to a value entry `v<i>: !str <value>`.
 */
function valueFlow(whens: readonly (readonly [string, string])[], onType: string): string {
  let text = `id: f\nname: f\n${declareN(onType)}nodes:\n  route:\n    kind: case\n    on: \${input.n}\n    cases:\n`;
  for (const [index, [when]] of whens.entries()) text += `      - when: ${when}\n        then: v${index}\n`;
  for (const [index, [, value]] of whens.entries()) text += `  v${index}: !str "${value}"\n`;
  return `${text}output: \${route.output}\n`;
}

test("bool value case missing false is not exhaustive", () => {
  const error = loadError(() => assembleText(valueFlow([["true", "yes"]], "bool")));

  expect(error.message).toBe(notExhaustive("bool value(s) [false] not covered"));
});

test("optional bool is not exhaustive", () => {
  const text = valueFlow(
    [
      ["true", "yes"],
      ["false", "no"],
    ],
    "Optional(bool)",
  );

  expect(loadError(() => assembleText(text)).message).toBe(notExhaustive("${input.n} may be null"));
});

test("int value case reason names Literal or bool", () => {
  const error = loadError(() => assembleText(valueFlow([["1", "one"]], "int")));

  expect(error.message).toBe(notExhaustive("on: ${input.n} is int, not a Literal or bool"));
});

// ---------- routing-only enum coverage ----------

/** The message for case `route` over `on` leaving the tag `b` uncovered. */
function uncovered(on: string): string {
  return (
    `case node 'route' on ${on}: non-exhaustive — enum tag(s) ["b"] not covered by a case ` +
    "`when:` or an `else:`"
  );
}

test("routing-only Literal flow input missing a tag", () => {
  const text = flow([["a", "one"]], { onType: "Literal(a, b)", else_: null });

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(uncovered("${input.n}"));
  expect(errorLine(error)).toBe(lineOf(text, "  route:"));
});

test("routing-only Literal flow input with else loads", () => {
  assembleText(flow([["a", "one"]], { onType: "Literal(a, b)", else_: "other" }));
});

test("routing-only Literal flow input listing every tag loads", () => {
  assembleText(
    flow(
      [
        ["a", "one"],
        ["b", "two"],
      ],
      { onType: "Literal(a, b)", else_: null },
    ),
  );
});

test("routing-only loop-carried Literal missing a tag", () => {
  const text =
    "id: f\nname: f\ntypedefs:\n  N: Literal(a, b)\ninput:\n  s: !int\nnodes:\n" +
    "  rep:\n    kind: loop\n    times: 2\n    input:\n      n: !N a\n    nodes:\n" +
    "      route:\n        kind: case\n        on: ${input.n}\n        cases:\n" +
    "          - when: a\n            then: one\n" +
    branchBody("one", "Literal(a, b)", '"b"') +
    "    output:\n      n: ${one.output}\noutput: ${rep.output}\n";

  const error = loadError(() => assembleText(text));

  expect(error.message).toEndWith(uncovered("${input.n}"));
  expect(errorLine(error)).toBe(lineOf(text, "      route:"));
});

test("routing-only map child Literal input missing a tag", () => {
  const text =
    "id: f\nname: f\ntypedefs:\n  Xs: list(Literal(a, b))\ninput:\n  xs: !Xs\nnodes:\n" +
    "  each:\n    kind: map\n    over: ${input.xs}\n    input:\n      x: ${item}\n    nodes:\n" +
    "      route:\n        kind: case\n        on: ${input.x}\n        cases:\n" +
    "          - when: a\n            then: one\n" +
    branchBody("one", "str", '"one"') +
    "    output: ${one.output}\noutput: ${each.output}\n";

  const error = loadError(() => assembleText(text));

  expect(error.message).toEndWith(uncovered("${input.x}"));
  expect(errorLine(error)).toBe(lineOf(text, "      route:"));
});

test("routing-only call child Literal input missing a tag", async () => {
  const text =
    "id: f\nname: f\ntypedefs:\n  K: Literal(a, b)\ninput:\n  k: !K\n" +
    "defs:\n  child:\n    input:\n      k: !K\n    nodes:\n" +
    "      route:\n        kind: case\n        on: ${input.k}\n        cases:\n" +
    "          - when: a\n            then: one\n" +
    branchBody("one", "str", '"one"') +
    "    output: ${one.output}\n" +
    "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      k: ${input.k}\n" +
    "output: ${c.output}\n";

  const error = await loadErrorAsync(loadFlow(text));

  expect(error.message).toEndWith(uncovered("${input.k}"));
  expect(errorLine(error)).toBe(lineOf(text, "      route:"));
});

// ---------- harvested conformance cases ----------

test("assembled: load-case/engine_case_on_typed__infinite_when_on_a_float", () => {
  runCase({ id: "load-case/engine_case_on_typed__infinite_when_on_a_float", call: assembled });
});

test.each([
  "load-case/engine_case_on_typed__bool_when_on_int_is_an_error",
  "load-case/engine_case_on_typed__expression_on_is_typed",
  "load-case/engine_case_on_typed__int_when_on_str_asks_for_quotes",
  "load-case/engine_case_on_typed__integral_float_when_on_int",
  "load-case/engine_case_on_typed__malformed_date_text_on_a_date_is_not_a_value",
  "load-case/engine_case_on_typed__out_of_bounds_int_when",
  "load-case/engine_case_on_typed__repeated_int_when",
  "load-case/engine_case_on_typed__tags_that_read_as_other_values_are_listed_quoted",
  "load-case/engine_case_on_typed__text_when_on_int_in_a_map_body",
  "load-case/engine_case_on_typed__text_when_on_int_is_an_error",
  "load-case/engine_case_on_typed__typo_on_a_literal_flow_input",
])("assembled: %s", (id) => {
  runCase({ id, call: assembled });
});

test.each<CaseRow>(
  [
    "bool_value_case_missing_false_is_not_exhaustive",
    "int_value_case_reason_names_literal_or_bool",
    "optional_bool_is_not_exhaustive",
    "routing_only_map_child_literal_missing_a_tag",
    "routing_only_literal_flow_input_with_else_loads",
    "routing_only_literal_flow_input_listing_every_tag_loads",
  ].map((name) => ({ id: `load-case/engine_case_on_typed__${name}`, call: assembled })),
)("assembled: $id", runCase);

test("assembled: load-case/engine_case_on_typed__routing_only_call_child_literal_missing_a_tag", () => {
  runCase({
    id: "load-case/engine_case_on_typed__routing_only_call_child_literal_missing_a_tag",
    call: assembled,
    full: true,
  });
});
