/**
 * Flows that use `Any`: the rows the cross-flow binding check and the loop type contract decide, a
 * port of rows of `tests/compose/test_any_flows.py`, assembling each flow with `assembleText` where
 * Python calls `load_flow`.
 *
 * `Any` is any JSON value except null; `Optional(Any)` adds null. Wiring an `Any` into a narrower
 * child input is a load error unless a tag or an `isinstance` branch narrows it.
 *
 * Ported: `test_an_optional_str_output_into_a_child_any_input_is_a_load_error`,
 * `test_an_untagged_any_into_a_child_str_input_loads_inside_an_isinstance_branch` (load half),
 * `test_the_same_untagged_wiring_outside_the_branch_is_a_load_error`,
 * `test_a_map_over_a_list_of_optional_any_into_a_child_any_input_is_a_load_error`,
 * `test_a_map_over_a_mixed_literal_list_still_checks_each_element_at_load`,
 * `test_a_child_any_output_into_a_parent_int_input_is_a_load_error_with_the_hint`,
 * `test_an_untyped_seed_into_an_any_body_input_rejects_an_optional_any_body_field`; the load halves
 * of `test_a_tagged_any_binding_into_a_child_str_input_fails_at_runtime_for_an_int`,
 * `test_a_tagged_any_binding_into_a_child_str_input_runs_for_text`,
 * `test_a_str_output_into_a_child_any_input_loads_and_runs`,
 * `test_a_map_over_a_list_of_any_runs_items_into_a_child_any_input`,
 * `test_a_map_over_a_list_of_optional_any_into_a_child_optional_any_input_runs`,
 * `test_a_map_over_a_mixed_literal_list_into_a_child_any_input_runs`,
 * `test_a_reader_tagging_a_case_result_any_gets_the_taken_branch_value` and
 * `test_a_child_any_output_into_a_parent_any_consumer_runs`,
 * `test_a_loop_carrying_an_any_seed_with_an_any_body_field_runs`,
 * `test_a_typed_loop_seed_accepts_an_optional_any_body_field` and
 * `test_a_null_from_an_optional_any_body_field_fails_the_next_turn`;
 * `test_a_case_output_any_is_a_load_error_suggesting_the_reader_tag`, which the case-branch check
 * decides; and the harvested cases
 * `load-types/compose_any_flows__a_map_over_a_mixed_literal_list_checks_each_element` and
 * `load-types/compose_any_flows__an_untyped_seed_rejects_an_optional_any_body_field`.
 *
 * Changed: the `child` and `mk_any` defs are built by hand (`handChild`: the def's parameters read
 * from its `input:` text, its nodes as producers with their output contracts, `mk_any`'s output a
 * `Binding` declared `Any`), since def bodies load in T3-22. Rows asserting fragments pin the
 * whole message. CODE bodies are TS; load doesn't run them.
 *
 * Not ported (not T3-19 rows): `test_a_typed_any_literal_loads_and_runs` (4) and
 * `test_a_null_any_literal_is_a_load_error`, decided by the tags reader (the null row's case is
 * pinned in `tags.test.ts`); `test_a_list_of_any_into_an_untagged_inline_body_parameter_runs` (2),
 * decided by T3-17's inline inputs; their run halves are T4's.
 * Python-only (dates and non-text keys): `test_an_object_output_rejects_a_date_member`,
 * `test_an_object_output_rejects_a_non_text_key_at_the_output`,
 * `test_an_object_flow_input_rejects_a_non_text_key_naming_the_input`,
 * `test_a_non_text_key_nested_in_a_flow_input_names_its_path`,
 * `test_a_date_inside_an_object_flow_input_arrives_as_text`,
 * `test_an_unquoted_datetime_when_on_an_any_value_keeps_its_quote_error` (dropped in
 * `case_on_typed.test.ts`) and `test_an_unquoted_date_any_literal_gets_the_quote_hint` (TS loads
 * it, as `tags.test.ts` notes).
 *
 * Later (T3-18): `test_a_field_read_on_an_any_output_loads_in_a_binding`,
 * `test_a_field_read_on_an_any_output_loads_in_a_flow_output`,
 * `test_a_field_read_on_an_any_output_loads_in_a_case_condition`,
 * `test_a_field_read_on_an_any_value_entry_is_not_checked` and
 * `test_a_field_read_on_an_object_value_entry_is_still_checked`. Later (T3-20):
 * `test_a_field_read_on_an_any_output_loads_in_an_assert`,
 * `test_an_int_when_on_an_any_value_loads_and_matches`,
 * `test_a_text_when_on_an_any_value_loads_and_matches`. Later (T4):
 * `test_a_list_of_any_output_returns_mixed_items`, `test_an_any_output_returns_any_json_value` (4),
 * `test_an_any_output_rejects_null`, `test_an_optional_any_output_returns_null`,
 * `test_an_any_flow_input_rejects_null_naming_the_input`,
 * `test_an_unquoted_date_when_on_an_any_value_matches_its_text`, and the run halves of the rows
 * above.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/**
 * A flow taking `s: !str` whose CODE node `mk` returns `code`, typed `output` (untyped when
 * `null`), then the `rest` nodes; `head` holds top-level blocks before `input:`.
 */
function flow(
  output: string | null,
  rest: string,
  options: { code?: string; out?: string; head?: string } = {},
): string {
  const { code = "null", out = "${mk.output}", head = "" } = options;
  const typed = output === null ? "" : `    output: ${output}\n`;
  return (
    `id: p\nname: p\n${head}input:\n  s: !str\nnodes:\n  mk:\n    kind: code\n` +
    `    input:\n      s: \${input.s}\n${typed}    code: |\n      return ${code}\n${rest}output: ${out}\n`
  );
}

/** A `defs:` entry `name` taking `x: !<inputType>` and returning its text. */
const childDef = (name: string, inputType: string) =>
  `  ${name}:\n    input:\n      x: !${inputType}\n    nodes:\n      n:\n        kind: code\n` +
  "        input:\n          x: ${input.x}\n        output: str\n" +
  "        code: return JSON.stringify(inputs['x'])\n    output: ${n.output}\n";

/** The `child` def built by hand: input `x: !<inputType>`, one `n: str` producer, returned. */
const child = (inputType: string) =>
  handChild({
    inputs: parametersFromYaml(`input:\n  x: !${inputType}\n`, new Map()),
    producers: { n: STR },
    output: "${n.output}",
  });

/** A CALL node `nodeId` calling `callee` with `x: <binding>`. */
const call = (binding: string, nodeId = "c", callee = "child") =>
  `  ${nodeId}:\n    kind: call\n    call: ${callee}\n    input:\n      x: ${binding}\n`;

/** Assemble `text` with `children` as its defs. */
const load = (text: string, children: Record<string, ComposedFlow>) => assembleText(text, "flow.yaml", { children });

// ---------- CALL wiring ----------

test.each([
  ["fails at runtime for an int", "3n"],
  ["runs for text", '"hi"'],
])("a tagged Any binding into a child str input %s (loads)", (_name, code) => {
  const text = flow("Any", call("!str ${mk.output}"), {
    code,
    out: "${c.output}",
    head: `defs:\n${childDef("child", "str")}`,
  });

  load(text, { child: child("str") });
});

test("a str output into a child Any input loads and runs (loads)", () => {
  const text = flow("str", call("${mk.output}"), {
    code: '"hi"',
    out: "${c.output}",
    head: `defs:\n${childDef("child", "Any")}`,
  });

  load(text, { child: child("Any") });
});

test("an optional str output into a child Any input is a load error", () => {
  const text = flow("Optional(str)", call("${mk.output}"), {
    out: "${c.output}",
    head: `defs:\n${childDef("child", "Any")}`,
  });

  const error = loadError(() => load(text, { child: child("Any") }));

  expect(error.message).toBe("call node 'c': binding 'x' — child expects 'Any', source is 'Optional(str)'");
});

// ---------- narrowing with isinstance ----------

/** A flow routing on `isinstance(mk.output, str)` with `bindingNode` as `go`. */
function narrowed(bindingNode: string): string {
  const rest =
    "  route:\n    kind: case\n    cases:\n" +
    "      - when: isinstance(mk.output, str)\n        then: go\n" +
    "    else: other\n" +
    bindingNode +
    "  other:\n    kind: code\n    output: str\n    code: return 'other'\n";
  return flow("Any", rest, {
    code: '"hi"',
    out: "${go.output | other.output}",
    head: `defs:\n${childDef("child", "str")}`,
  });
}

test("an untagged Any into a child str input loads inside an isinstance branch (loads)", () => {
  load(narrowed(call("${mk.output}", "go")), { child: child("str") });
});

test("the same untagged wiring outside the branch is a load error", () => {
  const text = narrowed(call("${mk.output}", "go") + call("${mk.output}", "c2"));

  const message = loadError(() => load(text, { child: child("str") })).message;

  // `go` is inside the branch, so only `c2` fails.
  expect(message).toBe(
    "call node 'c2': binding 'x' — child expects 'str', source is 'Any'; tag the slot " +
      "(`x: !str ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
  );
});

// ---------- MAP over lists of Any ----------

/** A flow mapping `child` (taking `x: !<childType>`) over `over`. */
function map(overOutput: string | null, childType: string, code: string, over = "${mk.output}"): string {
  const rest = `  m:\n    kind: map\n    over: ${over}\n    call: child\n    input:\n      x: \${item}\n`;
  return flow(overOutput, rest, { code, out: "${m.output}", head: `defs:\n${childDef("child", childType)}` });
}

test.each<[string, string | null, string, string, string | undefined]>([
  ["a list of Any runs items into a child Any input", "list(Any)", "Any", '[1n, "two"]', undefined],
  [
    "a list of optional Any into a child optional Any input runs",
    "list(Optional(Any))",
    "Optional(Any)",
    "[1n, null]",
    undefined,
  ],
  ["a mixed literal list into a child Any input runs", null, "Any", "null", "['a', 1]"],
])("a map over %s (loads)", (_name, overOutput, childType, code, over) => {
  load(map(overOutput, childType, code, over), { child: child(childType) });
});

test("a map over a list of optional Any into a child Any input is a load error", () => {
  const error = loadError(() => load(map("list(Optional(Any))", "Any", "[1n]"), { child: child("Any") }));

  expect(error.message).toBe("call node 'm': binding 'x' — child expects 'Any', source is 'Optional(Any)'");
});

test("a map over a mixed literal list still checks each element at load", () => {
  const error = loadError(() => load(map(null, "str", "null", "['a', 1]"), { child: child("str") }));

  expect(error.message).toBe(
    "call node 'm': binding 'x' — child expects 'str', but the element 1 does not fit; " +
      'quote it to keep it as text: "1"',
  );
});

// ---------- LOOP carried Any ----------

/** A flow with an inline loop carrying `acc`, seeded `seed`, from a body node typed `bodyOutput`. */
const inlineLoop = (seed: string, bodyOutput: string, code: string) =>
  "id: p\nname: p\nnodes:\n" +
  `  loop:\n    kind: loop\n    times: 2\n    input:\n      acc: ${seed}\n    nodes:\n` +
  "      step:\n        kind: code\n        input:\n          acc: ${input.acc}\n" +
  `        output: ${bodyOutput}\n        code: |\n          return ${code}\n` +
  "    output:\n      acc: ${step.output}\noutput: ${loop.output}\n";

test.each([
  ["a loop carrying an Any seed with an Any body field runs", "Any", "[inputs['acc']]"],
  ["a typed loop seed accepts an optional Any body field", "Optional(Any)", "inputs['acc'] + 1n"],
  ["a null from an optional Any body field fails the next turn", "Optional(Any)", "null"],
])("%s (loads)", (_name, bodyOutput, code) => {
  assembleText(inlineLoop("!Any 0", bodyOutput, code));
});

test("an untyped seed into an Any body input rejects an optional Any body field", () => {
  const text = inlineLoop("${mk.output}", "Optional(Any)", "inputs['acc']").replace(
    "nodes:\n",
    "nodes:\n  mk:\n    kind: code\n    output: Any\n    code: return 0\n",
  );

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "loop node 'loop': the body returns 'acc' as Optional(Any), but the carried field is typed Any from its seed; " +
      "declare the seed's type: `acc: !Optional(Any) ${mk.output}`",
  );
});

// ---------- CASE output Any over branches of different types ----------

/** Return a flow whose `route` case picks the value entry `one` or `a`, read by `use`. */
const routed = (caseOutput = "", reader = "!Any ${route.output}") =>
  "id: p\nname: p\ninput:\n  s: !str\nnodes:\n" +
  `  route:\n    kind: case\n${caseOutput}    cases:\n` +
  "      - when: ${input.s == 'i'}\n        then: one\n" +
  "    else: a\n" +
  "  one: !int 1\n" +
  '  a: !str "a"\n' +
  `  use:\n    kind: code\n    input:\n      v: ${reader}\n` +
  "    code: return inputs['v']\n" +
  "output: ${use.output}\n";

test("a reader tagging a case result Any gets the taken branch value (loads)", () => {
  assembleText(routed());
});

test("a case output Any is a load error suggesting the reader tag", () => {
  const error = loadError(() => assembleText(routed("    output: Any\n", "${route.output}")));

  expect(error.message).toBe(
    "case node 'route': a `case` routes and has no `output:`; tag its result where you read it, " +
      "e.g. `x: !Any ${route.output}`",
  );
});

// ---------- child flow output Any ----------

const ANY_CHILD_DEF =
  "  mk_any:\n    input:\n      x: !str\n    nodes:\n" +
  "      n:\n        kind: code\n        input:\n          x: ${input.x}\n" +
  "        code: return 3n\n    output: !Any ${n.output}\n";

/** The `mk_any` def built by hand: input `x: !str`, an untyped `n`, output `!Any ${n.output}`. */
const MK_ANY = handChild({
  inputs: parametersFromYaml("input:\n  x: !str\n", new Map()),
  producers: { n: null },
  output: new Binding("${n.output}", { contract: new ValueContract(Type.scalar(ValueKind.ANY)) }),
});

/** A flow calling `mk_any`, then `child` (taking `consumerType`) on its output. */
function fromAnyChild(consumerType: string): string {
  const rest = call("${input.s}", "g", "mk_any") + call("${g.output}");
  return flow(null, rest, { out: "${c.output}", head: `defs:\n${ANY_CHILD_DEF}${childDef("child", consumerType)}` });
}

test("a child Any output into a parent Any consumer runs (loads)", () => {
  load(fromAnyChild("Any"), { mk_any: MK_ANY, child: child("Any") });
});

test("a child Any output into a parent int input is a load error with the hint", () => {
  const error = loadError(() => load(fromAnyChild("int"), { mk_any: MK_ANY, child: child("int") }));

  expect(error.message).toBe(
    "call node 'c': binding 'x' — child expects 'int', source is 'Any'; " +
      "tag the slot (`x: !int ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
  );
});

// ---------- harvested conformance cases ----------

test("assembledWith: load-types/compose_any_flows__a_map_over_a_mixed_literal_list_checks_each_element", () => {
  runCase({
    id: "load-types/compose_any_flows__a_map_over_a_mixed_literal_list_checks_each_element",
    call: assembledWith({ child: child("str") }),
  });
});

test("assembled: load-types/compose_any_flows__an_untyped_seed_rejects_an_optional_any_body_field", () => {
  runCase({ id: "load-types/compose_any_flows__an_untyped_seed_rejects_an_optional_any_body_field", call: assembled });
});
