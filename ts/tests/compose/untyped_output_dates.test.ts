/**
 * Literals in an untyped flow output: a port of `tests/compose/test_untyped_output_dates.py`,
 * through `assembleText` where Python loads the flow.
 *
 * Changed: Python refuses a literal in an untyped output that isn't a JSON value. In TS an
 * unquoted date or datetime is text (rule `dates`), and so is a date under `!!timestamp`, so each
 * loads and the output binding holds the text: the rows of
 * `test_untagged_date_in_an_untyped_output_field_is_a_load_error`,
 * `test_nested_untagged_date_names_its_path`, `test_a_nested_date_is_located_at_the_value`,
 * `test_a_datetime_or_timestamp_in_an_untyped_output_is_a_load_error`,
 * `test_untagged_date_as_the_whole_output_is_a_load_error` and
 * `test_an_aliased_date_names_the_output_field`, and `!Any 2026-01-01` in
 * `test_quoted_or_tagged_dates_keep_their_behavior`. Infinity is a float value (rule `infinity`),
 * so `.inf` and its spellings in `test_a_non_finite_number_in_an_untyped_output_is_a_load_error`
 * load. The YAML reader refuses the rest before any output check: `!!set` and `!!binary`
 * (`test_set_or_binary_literal_in_an_untyped_output_is_a_load_error`), `!!omap` and `!!pairs`
 * (`test_a_non_json_value_in_an_omap_pair_is_a_load_error`, and the `!!omap` row of
 * `test_other_untagged_literals_in_untyped_outputs_still_load`), `.nan` and a float too large to
 * read (the other rows of `test_a_non_finite_number_in_an_untyped_output_is_a_load_error`,
 * `test_a_nested_or_core_tagged_non_finite_number_is_a_load_error`, whose `!!omap [{a: .nan}]` row
 * is refused as `!!omap`, and `test_a_non_finite_number_as_the_whole_output_is_a_load_error`), and
 * `!float .nan`, which Python refuses at run (the load half of
 * `test_quoted_tagged_or_finite_numbers_keep_their_behavior`). So do the inline CALL, MAP and LOOP
 * rows of `test_untagged_date_in_a_body_output_is_a_load_error`: the date in the child's output
 * binding is text.
 *
 * The harvested `yaml/compose_untyped_output_dates__*` cases are pinned at the end;
 * `any_tag_on_a_date` is pinned in `tags.test.ts`.
 *
 * Later (T3-22): the `defs:` body rows of `test_untagged_date_in_a_body_output_is_a_load_error`,
 * `test_a_body_output_date_is_located_at_the_value` and
 * `test_a_non_finite_number_in_a_body_output_is_a_load_error`.
 * Later (T4): the runs of `test_quoted_or_tagged_dates_keep_their_behavior`,
 * `test_other_untagged_literals_in_untyped_outputs_still_load` and
 * `test_quoted_tagged_or_finite_numbers_keep_their_behavior`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase, runTextCase } from "./declarations.ts";

afterEach(() => forget());

const HEAD = "id: p\nname: p\nnodes:\n  b:\n    kind: code\n    code: return 1\n";
const NAN = "invalid YAML: NaN is not a yian value; quote it to keep it as text";
const TOO_LARGE = "invalid YAML: number is too large for a float; quote it to keep it as text";

/** Return a flow with producer `b` and the block `output:` lines `output`. */
function flow(output: string): string {
  return HEAD + "output:\n" + output.split("\n").map((line) => `  ${line}\n`).join("");
}

/** Load `text` and return its flow. */
const load = (text: string): ComposedFlow => assembleText(text).flow;

/** Return each flow output's binding source by name. */
function outputSources(text: string): Record<string, unknown> {
  return Object.fromEntries(
    load(text).compiled.outputs.map((output) => [output.name, output.binding.source]),
  );
}

/** Load `text`, expecting `message` at the line spelled `line`. */
function errorAt(text: string, message: string, line: string): void {
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(line) + 1);
}

describe("dates", () => {
  test.each([
    ["as_of: 2026-01-01", { as_of: "2026-01-01" }],
    ["g: {d: 2026-01-01}", { g: { d: "2026-01-01" } }],
    ["g: [1, 2026-01-01]", { g: [1n, "2026-01-01"] }],
    ["g: {x: [{d: 2026-01-01}]}", { g: { x: [{ d: "2026-01-01" }] } }],
    ["g:\n  x: 1\n  d: 2026-01-01", { g: { x: 1n, d: "2026-01-01" } }],
    ["as_of: 2026-01-01T10:00:00", { as_of: "2026-01-01T10:00:00" }],
    ["as_of: !!timestamp '2026-01-01'", { as_of: "2026-01-01" }],
  ])("an unquoted date in an untyped output field is text: %j", (output, expected) => {
    expect(outputSources(flow(`${output}\nh: \${b.output}`))).toStrictEqual({ ...expected, h: "${b.output}" });
  });

  test.each([
    ["2026-01-01", "2026-01-01"],
    ["[2026-01-01]", ["2026-01-01"]],
  ])("an unquoted date as the whole output is text: %s", (output, expected) => {
    expect(outputSources(HEAD + `output: ${output}\n`)).toStrictEqual({ result: expected });
  });

  test("an aliased date is text", () => {
    const text =
      "id: p\nname: p\nnodes:\n  b:\n    kind: code\n" +
      "    input:\n      d: &d 2026-01-01\n    code: return 1\n" +
      "output:\n  g: *d\n  h: ${b.output}\n";

    expect(outputSources(text)).toStrictEqual({ g: "2026-01-01", h: "${b.output}" });
  });

  test.each([
    ['as_of: "2026-01-01"', "2026-01-01"],
    ["as_of: !Any 2026-01-01", "2026-01-01"],
    ['as_of: !date "2026-01-01"', "2026-01-01"],
    ['as_of: "${b.output}-2026-01-01"', "${b.output}-2026-01-01"],
    ["as_of: ${b.output}-2026-01-01", "${b.output}-2026-01-01"],
  ])("a quoted or tagged date loads: %j", (output, expected) => {
    expect(outputSources(flow(`${output}\nh: \${b.output}`))["as_of"]).toBe(expected);
  });
});

describe("inline body outputs", () => {
  const inlineCall = `id: p
name: p
nodes:
  c:
    kind: call
    nodes:
      b:
        kind: code
        code: return 1
    output:
      d: 2026-01-01
      h: \${b.output}
output: \${c.output}
`;
  const inlineMap = `id: p
name: p
nodes:
  m:
    kind: map
    over: [1, 2]
    nodes:
      b:
        kind: code
        code: return 1
    output:
      d: 2026-01-01
      h: \${b.output}
output: \${m.output}
`;
  const inlineLoop = `id: p
name: p
nodes:
  l:
    kind: loop
    times: 1
    input:
      n: 0
      d: !Any "x"
    nodes:
      b:
        kind: code
        input:
          n: \${input.n}
        code: return inputs['n'] + 1n
    output:
      n: \${b.output}
      d: 2026-01-01
output: \${l.output}
`;

  test.each([
    ["inline-call", inlineCall, "c", { d: "2026-01-01", h: "${b.output}" }],
    ["inline-map", inlineMap, "m", { d: "2026-01-01", h: "${b.output}" }],
    ["inline-loop", inlineLoop, "l", { n: "${b.output}", d: "2026-01-01" }],
  ])("an unquoted date in an inline body's untyped output is text: %s", (_name, text, nodeId, expected) => {
    const child = (load(text).compiled.nodes.get(nodeId) as CallNode | MapNode | LoopNode).child as CompiledFlow;

    expect(Object.fromEntries(child.outputs.map((output) => [output.name, output.binding.source]))).toStrictEqual(
      expected,
    );
  });
});

describe("other literals", () => {
  test("JSON literals in an untyped output load", () => {
    expect(outputSources(flow("n: 1\ns: x\nl: [1, null]\nm: {k: true}\nh: ${b.output}"))).toStrictEqual({
      n: 1n,
      s: "x",
      l: [1n, null],
      m: { k: true },
      h: "${b.output}",
    });
  });

  test.each([
    ["g: !!set {a: null}", "set"],
    ["g: !!binary aGVsbG8=", "binary"],
    ["g: !!omap [a: 1]", "omap"],
    ["g: !!omap [a: 2026-01-01]", "omap"],
    ["g: !!omap [{a: .nan}]", "omap"],
    ["g: !!pairs [a: !!set {x: null}]", "pairs"],
  ])("the YAML reader refuses %j", (output, tag) => {
    const text = flow(`${output}\nh: \${b.output}`);

    errorAt(text, `invalid YAML: could not determine a constructor for the tag 'tag:yaml.org,2002:${tag}'`, `  ${output}`);
  });
});

describe("numbers that are not finite", () => {
  test.each([
    [".inf", Infinity],
    ["-.inf", -Infinity],
    [".Inf", Infinity],
    ["+.inf", Infinity],
    ["-.Inf", -Infinity],
  ])("an infinity in an untyped output loads: %s", (g, expected) => {
    expect(outputSources(flow(`g: ${g}\nh: \${b.output}`))["g"]).toBe(expected);
  });

  test.each([
    ["g: .nan", NAN],
    ["g: .NaN", NAN],
    ["g: 1e400", TOO_LARGE],
    ["g: 1.0e+400", TOO_LARGE],
    ["g: [1, .nan]", NAN],
    ["g: {x: .nan}", NAN],
    ["g: !!float .nan", NAN],
    ["g: !float .nan", NAN],
  ])("the YAML reader refuses %j", (output, message) => {
    const text = flow(`${output}\nh: \${b.output}`);

    errorAt(text, message, `  ${output}`);
  });

  test.each([".nan", "[1, .nan]"])("a NaN as the whole output is refused by the YAML reader: %s", (output) => {
    const text = HEAD + `output: ${output}\n`;

    errorAt(text, NAN, `output: ${output}`);
  });

  test("quoted or !!str NaN is text, and finite numbers load", () => {
    expect(outputSources(flow('g: ".nan"\nh: ${b.output}'))["g"]).toBe(".nan");
    expect(outputSources(flow("g: !!str .nan\nh: ${b.output}"))["g"]).toBe(".nan");
    expect(outputSources(flow("g: [1.5, -2e10, 1e300]\nh: ${b.output}"))["g"]).toStrictEqual([1.5, -2e10, 1e300]);
  });
});

describe("harvested cases", () => {
  test.each<CaseRow>(
    [
      "an_aliased_date_names_the_output_field",
      "core_timestamp_tag_on_quoted_text",
      "date_in_an_inline_call_output",
      "date_in_an_output_field",
      "date_tag_on_quoted_text_loads",
      "datetime_in_an_output_field",
      "inf_in_an_output_field",
      "nested_date_names_its_path",
    ].map((name) => ({ id: `yaml/compose_untyped_output_dates__${name}`, call: assembled })),
  )("$id", runCase);

  // The YAML reader refuses these while the document is parsed; the cases leave out its message.
  test.each([
    ["core_float_tag_on_nan", NAN],
    ["nan_in_an_output_field", NAN],
    ["overflowing_float_in_an_output_field", TOO_LARGE],
  ])("yaml/compose_untyped_output_dates__%s", (name, message) => {
    runTextCase({
      id: `yaml/compose_untyped_output_dates__${name}`,
      call: (text, label) => assembleText(text, label),
      message,
    });
  });
});
