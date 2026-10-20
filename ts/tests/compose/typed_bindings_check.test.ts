/**
 * Checking typed bindings at load: literals exactly, references by inferred type; a port of
 * `tests/compose/test_typed_bindings_check.py`, loading each flow with `assembleText` (`assemble`
 * as the loader calls it) where Python calls `load_flow`.
 *
 * A typed binding whose value can never fit its declared type is a load error located at the
 * binding. A literal value is fully known, so it is checked exactly; a reference or expression is
 * rejected only when its inferred type is disjoint from the declared one, while a narrower inferred
 * type loads and is checked at runtime.
 *
 * Ported: `test_disjoint_node_input_fails_at_its_line`,
 * `test_disjoint_flow_output_field_fails_at_its_line`, `test_disjoint_single_flow_output_fails`,
 * `test_record_flow_output_field_named_result_keeps_its_name`,
 * `test_embedded_template_is_text_and_disjoint_from_int`,
 * `test_record_with_an_extra_required_field_is_disjoint_from_a_smaller_record`,
 * `test_record_missing_a_required_field_names_the_field`,
 * `test_missing_field_is_not_named_when_another_field_mismatches`, `test_fitting_literals_load`
 * (6), the load half of `test_optional_into_non_optional_loads_and_fails_at_runtime`,
 * `test_str_into_literal_loads`, `test_int_into_float_loads`, `test_unproven_sources_load` (3),
 * `test_inline_call_source_that_returns_str_is_disjoint_from_int`,
 * `test_disjoint_binding_into_an_inline_body_fails`, `test_disjoint_loop_seed_fails`,
 * `test_disjoint_reference_into_a_call_reports_the_binding_not_the_child`,
 * `test_disjoint_item_binding_on_a_named_map_fails_at_its_line`,
 * `test_disjoint_item_binding_on_an_inline_map_fails`, `test_fitting_item_binding_loads`,
 * `test_disjoint_item_field_binding_fails`, `test_item_binding_over_an_untyped_list_loads` and
 * `test_disjoint_binding_inside_a_defs_body_names_the_body` (the `defs entry` prefix comes from the
 * scope's def loading).
 *
 * Changed: the error rows pin the whole message where Python matches its start, and the last frame's label and range where Python
 * checks the first frame's line. CODE bodies are written in TS and code refs name
 * `./fixtures/compose_codefns.ts`, as the harvested cases write them; load doesn't run them.
 *
 * Not ported here: the literal rows the tags reader (`readTaggedValue`, tested in `tags.test.ts`)
 * decides while the binding is split, before this check runs. Pinned in `tags.test.ts` through
 * their cases: `test_text_literal_for_int_fails_at_its_line`,
 * `test_number_literal_for_str_suggests_quoting`, `test_true_literal_for_str_explains_yaml_booleans`
 * and `test_null_literal_for_a_record_output_field_fails_at_its_line`. Decided there with no case:
 * `test_former_boolean_word_for_bool_names_the_boolean_to_write`,
 * `test_former_boolean_word_for_a_bool_output_names_the_boolean_to_write`,
 * `test_former_boolean_word_in_a_list_literal_gets_no_fix`,
 * `test_escaped_template_is_text_and_does_not_fit_int`,
 * `test_bad_literal_into_a_call_reports_the_binding_not_the_child` and
 * `test_bad_literal_loop_seed_reports_the_binding_not_the_carried_type`.
 *
 * Ported from `tests/compose/test_nested_template_types.py` (its mirror doesn't exist yet):
 * `test_nested_typed_binding_is_checked_at_its_own_line`.
 *
 * TS-only: `shownLiteral` rows (compact JSON, ints as digits); `localInputContracts` rows over
 * hand-built maps (Python has no direct test; the reference and assert checks read it); and the
 * harvested cases these checks decide (`runCase`, `assembled`), in the "Pinned harvested cases"
 * block, then those from test files whose mirrors don't exist yet (`test_nested_template_types.py`,
 * `test_tagged_bindings.py`) in the "Pinned harvested cases (TS-only placement)" block.
 *
 * Later (T4): `test_unquoted_yes_or_no_literal_for_str_is_text` (2),
 * `test_escaped_template_binds_its_text`, and the run half of
 * `test_optional_into_non_optional_loads_and_fails_at_runtime`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildResult } from "../../src/compose/nodes/base.ts";
import { LocalInput, localInputContracts, shownLiteral } from "../../src/compose/validate/typed_bindings.ts";
import { forget } from "../../src/defaults.ts";
import { Binding, type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));
const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/** Return a flow whose `use` node binds `x` as `binding` after `producers`. */
function flow(binding: string, options: { producers?: string; typedefs?: string } = {}): string {
  const { producers = "", typedefs = "" } = options;
  return `
id: checked
name: checked
${typedefs}nodes:
${producers}  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: ${binding}
output: \${use.output}
`;
}

/** Return a CODE node `nodeId` returning `value`, declaring `output` when given. */
function producer(nodeId: string, value: string, output: string | null): string {
  const declared = output !== null ? `    output: ${output}\n` : "";
  return `  ${nodeId}:\n    kind: code\n    code: |\n      return ${value}\n${declared}`;
}

/** Load `text`, expecting a load error, and return it. */
const loadErrorOf = (text: string): LoadError => loadError(() => assembleText(text));

/** Load `text`. */
const load = (text: string): void => {
  assembleText(text);
};

/** Return the last frame of a load error as `[label, [startLine, startCol, endLine, endCol]]`. */
function lastFrame(error: LoadError): [string | null, number[]] {
  const frame = error.diagnostic.frames.at(-1)!;
  const range = frame.range;
  return [frame.label, [range.startLine, range.startCol, range.endLine, range.endCol]];
}

/** Return `flow.yaml` and the range of the value after `key: ` on `line` of `text`. */
function valueFrame(text: string, line: string): [string, number[]] {
  const number = text.split("\n").indexOf(line) + 1;
  return ["flow.yaml", [number, line.indexOf(": ") + 3, number, line.length + 1]];
}

// ---------- disjoint references ----------

test("a disjoint node input fails at its line", () => {
  const text = flow("!int ${note.output}", { producers: producer("note", '"hi"', "str") });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'use' input 'x': declared int, but the value is str");
  expect(lastFrame(error)).toEqual(valueFrame(text, "      x: !int ${note.output}"));
});

test("a disjoint flow output field fails at its line", () => {
  const text = `
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output:
  x: !int \${note.output}
`;
  const error = loadErrorOf(text);

  expect(error.message).toBe("output 'x': declared int, but the value is str");
  expect(lastFrame(error)).toEqual(valueFrame(text, "  x: !int ${note.output}"));
});

test("a disjoint single flow output fails", () => {
  const text = `
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output: !int \${note.output}
`;
  const error = loadErrorOf(text);

  expect(error.message).toBe("output: declared int, but the value is str");
  expect(lastFrame(error)).toEqual(valueFrame(text, "output: !int ${note.output}"));
});

test("a record flow output field named result keeps its name", () => {
  const text = `
id: checked
name: checked
nodes:
  note:
    kind: code
    code: |
      return "hi"
    output: str
output:
  result: !int \${note.output}
`;
  const error = loadErrorOf(text);

  expect(error.message).toBe("output 'result': declared int, but the value is str");
});

test("an embedded template is text and disjoint from int", () => {
  const text = flow("!int n=${a.output}", { producers: producer("a", "1", "int") });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'use' input 'x': declared int, but the value is str");
});

test("a record with an extra required field is disjoint from a smaller record", () => {
  const typedefs = "typedefs:\n  Big:\n    a: int\n    b: str\n  Small:\n    a: int\n";
  const text = flow("!Small ${big.output}", { producers: producer("big", '{a: 1n, b: "x"}', "Big"), typedefs });

  const error = loadErrorOf(text);

  expect(error.message).toBe('node \'use\' input \'x\': declared Small, but the value is {"a":int,"b":str}');
});

const BUY_TYPEDEFS = (conviction: string) =>
  `typedefs:\n  Buy:\n    action: Literal(buy)\n    conviction: float\n  Conv:\n    conviction: ${conviction}\n`;

test("a record missing a required field names the field", () => {
  const text = flow("!Buy ${d.output}", {
    producers: producer("d", "{conviction: 0.5}", "Conv"),
    typedefs: BUY_TYPEDEFS("float"),
  });

  const error = loadErrorOf(text);

  expect(error.message).toBe(
    "node 'use' input 'x': declared Buy, but the value is " +
      '{"conviction":float} (missing required `action`)',
  );
});

test("a missing field is not named when another field mismatches", () => {
  const text = flow("!Buy ${d.output}", {
    producers: producer("d", '{conviction: "x"}', "Conv"),
    typedefs: BUY_TYPEDEFS("str"),
  });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'use' input 'x': declared Buy, but the value is {\"conviction\":str}");
});

// Ported from `test_nested_template_types.py::test_nested_typed_binding_is_checked_at_its_own_line`.
test("a disjoint binding nested in a value template names its path", () => {
  const text = flow("{a: !int '${note.output}'}", { producers: producer("note", '"hi"', "str") });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'use' input 'x.a': declared int, but the value is str");
  const line = "      x: {a: !int '${note.output}'}";
  const number = text.split("\n").indexOf(line) + 1;
  // The nested binding's own tagged value, `!int '${note.output}'`.
  expect(lastFrame(error)).toEqual(["flow.yaml", [number, line.indexOf("!") + 1, number, line.length]]);
});

// ---------- literals ----------

test.each(["!Optional(int) null", "!int 3", "!list(str) []", "!float 3", "!str abc", "!str $${x}"])(
  "fitting literals load: %s",
  (binding) => {
    load(flow(binding));
  },
);

// ---------- narrower, opaque, and fitting references ----------

test("an optional into a non-optional loads (the runtime check rejects the null)", () => {
  load(flow("!int ${maybe.output}", { producers: producer("maybe", "null", "Optional(int)") }));
});

test("a str into a literal loads", () => {
  load(
    flow("!AB ${s.output}", {
      producers: producer("s", '"a"', "str"),
      typedefs: "typedefs:\n  AB: Literal(a, b)\n",
    }),
  );
});

test("an int into a float loads", () => {
  load(flow("!float ${i.output}", { producers: producer("i", "1", "int") }));
});

test.each([
  ["!int ${a.output + 1}", producer("a", "1", null)],
  ["!int ${p.output}", producer("p", "1", null)],
  ["!str ${system.run_id}", ""],
])("unproven sources load: %s", (binding, producers) => {
  load(flow(binding, { producers }));
});

// ---------- sources from calls, bodies, and loops ----------

const STR_CHILD = `
defs:
  shout:
    input:
      y: !str
    nodes:
      up:
        kind: code
        input:
          y: \${input.y}
        output: str
        code: |
          return inputs["y"].toUpperCase()
    output: \${up.output}
`;

test("a disjoint binding inside a defs body names the body", () => {
  const text = `
id: checked
name: checked
defs:
  child:
    nodes:
      note:
        kind: code
        output: str
        code: |
          return "hi"
      use:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: !int \${note.output}
    output: \${use.output}
nodes:
  run:
    kind: call
    call: child
output: \${run.output}
`;
  const error = loadErrorOf(text);

  expect(error.diagnostic.message).toBe("defs entry 'child': node 'use' input 'x': declared int, but the value is str");
  // The first frame is the enclosing definition; the last is the binding.
  expect(lastFrame(error)).toEqual(valueFrame(text, "          x: !int ${note.output}"));
});

test("a disjoint binding into an inline body fails", () => {
  const text = `
id: checked
name: checked
nodes:
  note:
    kind: code
    output: str
    code: |
      return "hi"
  wrap:
    kind: call
    input:
      n: !int \${note.output}
    nodes:
      twice:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] * 2n
    output: \${twice.output}
output: \${wrap.output}
`;
  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'wrap' input 'n': declared int, but the value is str");
  expect(lastFrame(error)).toEqual(valueFrame(text, "      n: !int ${note.output}"));
});

/** Return a flow whose LOOP seeds `n` as `seed` into `bump`, a body returning `n: int`. */
function loopFlow(seed: string): string {
  return `
id: checked
name: checked
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1n
    output:
      n: \${step.output}
nodes:
  note:
    kind: code
    output: str
    code: |
      return "hi"
  loop:
    kind: loop
    times: 2
    input:
      n: ${seed}
    call: bump
output: \${loop.output}
`;
}

test("a disjoint loop seed fails", () => {
  const error = loadErrorOf(loopFlow("!int ${note.output}"));

  expect(error.message).toBe("node 'loop' input 'n': declared int, but the value is str");
});

// ---------- the typed binding is reported before downstream checks ----------

/** Return a flow that calls `shout` (taking `y: str`) with `y` bound as `binding`. */
function callStrChild(binding: string): string {
  return `
id: checked
name: checked
${STR_CHILD}nodes:
  a:
    kind: code
    output: str
    code: |
      return "hi"
  run:
    kind: call
    call: shout
    input:
      y: ${binding}
output: \${run.output}
`;
}

test("a disjoint reference into a call reports the binding, not the child", () => {
  const error = loadErrorOf(callStrChild("!int ${a.output}"));

  expect(error.message).toBe("node 'run' input 'y': declared int, but the value is str");
});

// ---------- MAP item bindings ----------

/** Return a flow that maps the named `child` over `over`, binding `t` as `binding`. */
function namedMapFlow(
  binding: string,
  options: { over: string; paramType: string; typedefs?: string; input?: string },
): string {
  const { over, paramType, typedefs = "", input = "words: !list(int)" } = options;
  return `
id: checked
name: checked
${typedefs}input:
  ${input}
defs:
  child:
    input:
      t: ${paramType}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.t}
    output: \${echo.output}
nodes:
  opaque:
    kind: code
    code: |
      return [1n, 2n]
  m:
    kind: map
    over: ${over}
    call: child
    input:
      t: ${binding}
output: \${m.output}
`;
}

test("a disjoint item binding on a named map fails at its line", () => {
  const text = namedMapFlow("!str ${item}", { over: "${input.words}", paramType: "!str" });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'm' input 't': declared str, but the value is int");
  expect(lastFrame(error)).toEqual(valueFrame(text, "      t: !str ${item}"));
});

test("a disjoint item binding on an inline map fails", () => {
  const text = `
id: checked
name: checked
input:
  words: !list(int)
nodes:
  m:
    kind: map
    over: \${input.words}
    input:
      t: !str \${item}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.t}
    output: \${echo.output}
output: \${m.output}
`;
  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'm' input 't': declared str, but the value is int");
});

test("a fitting item binding loads", () => {
  load(namedMapFlow("!int ${item}", { over: "${input.words}", paramType: "!int" }));
});

test("a disjoint item field binding fails", () => {
  const text = namedMapFlow("!int ${item.name}", {
    over: "${input.recs}",
    paramType: "!int",
    typedefs: "typedefs:\n  Rec:\n    name: str\n",
    input: "recs: !list(Rec)",
  });

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'm' input 't': declared int, but the value is str");
});

test("an item binding over an untyped list loads", () => {
  load(namedMapFlow("!str ${item}", { over: "${opaque.output}", paramType: "!str" }));
});

// ---------- shownLiteral (TS-only) ----------

describe("shownLiteral writes a literal as compact JSON", () => {
  test.each<{ value: unknown; shown: string }>([
    { value: 5n, shown: "5" },
    { value: 1.5, shown: "1.5" },
    { value: "a", shown: '"a"' },
    { value: null, shown: "null" },
    { value: true, shown: "true" },
    { value: [1n, "a", { k: 2n }], shown: '[1,"a",{"k":2}]' },
  ])("$shown", ({ value, shown }) => {
    expect(shownLiteral(value)).toBe(shown);
  });
});

// ---------- localInputContracts (TS-only) ----------

describe("localInputContracts", () => {
  const OPTIONAL_INT = new ValueContract(Type.scalar(ValueKind.INTEGER).replace({ nullable: true }));
  const FLOAT = new ValueContract(Type.scalar(ValueKind.NUMBER));
  const PRODUCERS = new Map<string, OutputContract>([
    ["d", INT],
    ["o", OPTIONAL_INT],
    ["t", STR],
  ]);

  const contracts = (
    wiring: Record<string, Record<string, unknown>>,
    buildResults: Map<string, BuildResult> = new Map(),
  ): Map<string, Map<string, LocalInput>> =>
    localInputContracts(
      new Map(Object.entries(wiring).map(([nodeId, names]) => [nodeId, new Map(Object.entries(names))])),
      PRODUCERS,
      new Map(),
      buildResults,
    );

  const built = (node: MapNode | WaitNode | LoopNode, wiring: Record<string, unknown>): Map<string, BuildResult> =>
    new Map([[node.id, new BuildResult({ node, wiring: new Map(Object.entries(wiring)) })]]);

  test("a typed binding keeps the declared contract, then the inferred one when it fits", () => {
    const names = contracts({
      n: {
        fits: new Binding("${d.output}", { contract: OPTIONAL_INT }),
        wider: new Binding("${o.output}", { contract: INT }),
      },
    }).get("n");
    expect(names?.get("fits")).toEqual(new LocalInput([OPTIONAL_INT, INT], "d.output"));
    expect(names?.get("wider")).toEqual(new LocalInput([INT], "o.output"));
  });

  test("a literal is widened", () => {
    expect(contracts({ n: { s: "abc" } }).get("n")?.get("s")).toEqual(new LocalInput([STR], null));
  });

  test("an unproven binding is left out", () => {
    expect(contracts({ n: { u: "${nobody.output}", k: "${d.output}" } }).get("n")).toEqual(
      new Map([["k", new LocalInput([INT], "d.output")]]),
    );
    expect(contracts({ n: { u: "${nobody.output}" } })).toEqual(new Map());
  });

  test("a map node and reserved keys are skipped", () => {
    const wiring = { over: "${d.output}", x: "${d.output}" };
    expect(contracts({ m: wiring }, built(new MapNode("m", { flowId: "child" }), wiring))).toEqual(new Map());
    const waitWiring = { until: "${t.output}", x: "${d.output}" };
    expect(contracts({ w: waitWiring }, built(new WaitNode("w"), waitWiring))).toEqual(
      new Map([["w", new Map([["x", new LocalInput([INT], "d.output")]])]]),
    );
  });

  test("a loop name adds the carried field contract, and is dropped when the carried field is unproven", () => {
    const loop = new LoopNode("l", { flowId: "body" });
    loop.outputContract = new PartialRecordContract([
      ["n", FLOAT],
      ["note", null],
    ]);
    const wiring = { n: "${d.output}", note: "${t.output}", extra: "${d.output}" };
    expect(contracts({ l: wiring }, built(loop, wiring))).toEqual(
      new Map([["l", new Map([["n", new LocalInput([INT, FLOAT], "d.output")]])]]),
    );
  });

  test.each<{ source: string; bound: string | null }>([
    { source: "${d.output}", bound: "d.output" },
    { source: "${d.output :- null}", bound: "d.output" },
    { source: "${isinstance(d.output, int)}", bound: null },
    { source: "n=${d.output}", bound: null },
    { source: "abc", bound: null },
  ])("source of $source is $bound", ({ source, bound }) => {
    const local = contracts({ n: { p: source } }).get("n")?.get("p");
    expect(local).toBeInstanceOf(LocalInput);
    expect(local?.source).toBe(bound);
  });
});

// ---------- Pinned harvested cases ----------

// The open-range case comes from `test_open_range_flows.py`, whose mirror doesn't exist yet.

test.each<CaseRow>([
  ...[
    "disjoint_node_input_fails_at_its_line",
    "disjoint_flow_output_field_fails_at_its_line",
    "disjoint_single_flow_output_fails",
    "disjoint_binding_into_an_inline_body_fails",
    "an_unproven_source_loads",
    "int_into_float_loads",
    "a_mismatched_field_hides_the_missing_one",
    "a_record_with_an_extra_field_is_disjoint",
    "record_missing_a_required_field_names_the_field",
  ].map((name) => ({ id: `load-types/compose_typed_bindings_check__${name}`, call: assembled })),
  { id: "types/compose_open_range_flows__tagged_binding_disjoint_from_an_open_range", call: assembled },
])("assemble: $id", runCase);

// ---------- Pinned harvested cases (TS-only placement) ----------

// From `test_nested_template_types.py` and `test_tagged_bindings.py`, whose mirrors don't exist
// yet; the record and template messages decide them.

test.each<CaseRow>([
  {
    id: "load-types/compose_nested_template_types__nested_item_read_is_typed_from_the_map_element",
    call: assembled,
  },
  ...["tagged_template_with_a_disjoint_field_fails", "wrong_literal_value_under_a_tag_fails_at_load"].map((name) => ({
    id: `load-types/compose_nested_template_types__${name}`,
    call: assembled,
  })),
  ...["list_element_missing_a_field_names_its_path", "nested_record_missing_a_field_names_its_path"].map((name) => ({
    id: `load-types/compose_tagged_bindings__${name}`,
    call: assembled,
  })),
])("assemble: $id", runCase);
