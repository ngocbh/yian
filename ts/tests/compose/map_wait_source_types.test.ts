/**
 * A MAP `over:` or WAIT `until:` of the wrong type is a load error at that value: a port of
 * `tests/compose/test_map_wait_source_types.py`, loading each flow with `assembleText` where
 * Python calls `load_flow`; a `defs:` entry a flow calls is a hand-built child (`handChild`) with
 * the def's signature.
 *
 * Literals are checked where the node is parsed; references and expressions by their inferred type
 * once the graph is built.
 *
 * Ported: `test_value_type_error_wording`, `test_non_list_over_literal_is_a_load_error` (9),
 * `test_null_over_is_empty`, `test_text_over_is_a_load_error` (7), `test_list_over_literals_load`
 * (2), `test_call_directive_over_is_left_to_the_graph_stage`,
 * `test_call_text_under_until_is_not_a_directive`, `test_non_date_until_literal_is_a_load_error`
 * (9), `test_null_until_is_empty`, `test_text_until_must_parse_as_a_date` (3),
 * `test_date_until_literals_load` (5), the inline half of
 * `test_literal_check_reaches_defs_and_inline_bodies`, the first two thirds of
 * `test_tagged_wrong_literal_reports_the_type_first`; `test_disjoint_over_reference_is_a_load_error`
 * (10), `test_disjoint_until_reference_is_a_load_error` (8),
 * `test_the_issue_flow_is_a_load_error_at_until`, `test_fitting_or_unproven_sources_load` (13),
 * `test_call_directive_over_with_a_str_output_is_a_load_error`, `test_isinstance_keeps_its_own_error`,
 * the first half of `test_tagged_wrong_reference_reports_the_type_first`,
 * `test_record_typedef_is_named`, `test_none_typed_source_is_a_load_error` and
 * `test_type_error_comes_before_the_cycle_error`.
 *
 * Changed: TS YAML reads an unquoted date or timestamp as text, so `over: 2026-01-01` and
 * `over: 2026-01-01T10:00:00` are read as text (Python: "a date", "a date and time"), and the
 * unquoted `until:` dates load as text that parses as a date.
 *
 * Pinned elsewhere: `kind_map_parse.test.ts` and `kind_wait_parse.test.ts` pin this file's nine
 * parse-stage cases (`compose_map_wait_source_types__{date_over_literal,int_over_literal,
 * null_over_is_empty,quoted_call_over_is_text,call_text_under_until_is_not_a_directive,
 * integer_until_literal,month_13_text_until_is_not_a_date,null_until_is_empty,
 * literal_until_check_in_an_inline_body}`).
 *
 * Later (T3-18 PR 3): `test_dangling_reference_keeps_its_error` (the reference validator).
 * Later (T3-20): `test_a_case_branch_narrows_the_source` (`computeNarrowing`).
 * Later (T3-22): the `defs` half of `test_literal_check_reaches_defs_and_inline_bodies` and
 * `test_reference_check_reaches_defs` (the def's own assembly); the last third of `test_tagged_wrong_literal_reports_the_type_first` and the second
 * half of `test_tagged_wrong_reference_reports_the_type_first` ("a tag types a value",
 * `reject_unread_tags`).
 */

import { afterEach, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { valueTypeError } from "../../src/compose/nodes/base.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import type { Type } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

const OVER_FIX = "write a list, or a reference such as `over: ${input.items}`";
const UNTIL_FIX = "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`";

const overMessage = (readAs: string, node = "m") =>
  `node '${node}' (kind=map): \`over:\` must be a list, but ${readAs}; ${OVER_FIX}`;

const untilMessage = (readAs: string, node = "w") =>
  `node '${node}' (kind=wait): \`until:\` must be a date or datetime, but ${readAs}; ${UNTIL_FIX}`;

/** The `child` def the MAP calls: one `x: !Any` input, an untyped producer, END over it. */
const CHILD = handChild({
  inputs: parametersFromYaml("input:\n  x: !Any\n", new Map()),
  producers: { echo: null },
  output: "${echo.output}",
});

/** A def with no inputs whose one producer `c` has `typ`, returned. */
const returning = (typ: Type) =>
  handChild({ producers: { c: new ValueContract(typ) }, output: "${c.output}" });

/** The defs `mapFlow` declares: `child` and `items` (returning `list(str)`). */
const MAP_DEFS: ChildFlows = {
  child: CHILD,
  items: returning(typeFor("list(str)", new Map())),
};

/** The defs `mapRef` declares: `child` and `words` (returning `str`). */
const MAP_REF_DEFS: ChildFlows = { child: CHILD, words: returning(typeFor("str", new Map())) };

/** Load `text` with `children` built by hand, expecting a load error, and return it. */
const errorOf = (text: string, children: ChildFlows = {}): LoadError =>
  loadError(() => assembleText(text, "flow.yaml", { children }));

/** Load `text` with `children` built by hand. */
const load = (text: string, children: ChildFlows = {}): void => {
  assembleText(text, "flow.yaml", { children });
};

/** Return the 1-based line of the first line of `text` holding `marker`. */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${JSON.stringify(marker)} not in text`);
  return index + 1;
}

/** Return the start line of the error's last frame. */
const startLine = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

const ITEMS_DEF = `  items:
    nodes:
      c:
        kind: code
        output: list(str)
        code: |
          return ["a", "b"]
    output: \${c.output}
`;

/**
 * A flow whose MAP node `m` has `over: <value>` and calls `child`. Each flow builder substitutes
 * through a function, since a replacement string reads `$$` as `$`.
 */
function mapFlow(value: string): string {
  return (
    `id: f
name: f
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${echo.output}
` +
    ITEMS_DEF +
    `nodes:
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: \${m.output}
`
  ).replace("@V", () => value);
}

/** A flow whose WAIT node `w` has `until: <value>`, followed by a code node. */
function waitFlow(value: string): string {
  return (
    `id: f
name: f
input:
  d: !date
defs:
` +
    ITEMS_DEF +
    `nodes:
  w:
    kind: wait
    until: @V
  after:
    kind: code
    depends_on: [w]
    output: int
    code: |
      return 1
output: \${after.output}
`
  ).replace("@V", () => value);
}

// ---------- literals, checked where the node is parsed ----------

test("value type error wording", () => {
  expect(valueTypeError("over", "a list", 5n, "F")).toBe("`over:` must be a list, but YAML read it as an integer; F");
  expect(valueTypeError("over", "a list", null, "F")).toBe("`over:` must be a list, but it is empty; F");
});

test.each<[string, string]>([
  ["5", "an integer"],
  ["0", "an integer"],
  ["true", "a boolean"],
  ["false", "a boolean"],
  ["1.5", "a float"],
  ["{x: 1}", "a mapping"],
  ["{}", "a mapping"],
  ["2026-01-01", "text"],
  ["2026-01-01T10:00:00", "text"],
])("a non-list over literal %s is a load error", (value, phrase) => {
  const text = mapFlow(value);
  const error = errorOf(text, MAP_DEFS);
  expect(error.diagnostic.message).toBe(overMessage(`YAML read it as ${phrase}`));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("a null over is empty", () => {
  const text = mapFlow("null");
  const error = errorOf(text, MAP_DEFS);
  expect(error.diagnostic.message).toBe(overMessage("it is empty"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each<[string, string]>([
  ["word", "ok"],
  ["int", "'5'"],
  ["date", "'2026-01-01'"],
  ["escape", "'$${x}'"],
  ["quoted-call", "'call(items)'"],
  ["spaced-call", "' call(items)'"],
  ["block-call", "|\n      call(items)"],
])("a text over is a load error (%s)", (_id, value) => {
  const text = mapFlow(value);
  const error = errorOf(text, MAP_DEFS);
  expect(error.diagnostic.message).toBe(overMessage("YAML read it as text"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each(["[1, 2]", "[]"])("a list over literal %s loads", (value) => {
  load(mapFlow(value), MAP_DEFS);
});

test("a call directive over is left to the graph stage", () => {
  load(mapFlow("call(items)"), MAP_DEFS);
});

test("call text under until is not a directive", () => {
  const text = waitFlow("call(items)");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilMessage("YAML read it as text that is not a date or datetime"));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test.each<[string, string]>([
  ["5", "an integer"],
  ["0", "an integer"],
  ["true", "a boolean"],
  ["false", "a boolean"],
  ["1.5", "a float"],
  ["[a]", "a list"],
  ["[]", "a list"],
  ["{x: 1}", "a mapping"],
  ["{}", "a mapping"],
])("a non-date until literal %s is a load error", (value, phrase) => {
  const text = waitFlow(value);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilMessage(`YAML read it as ${phrase}`));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test("a null until is empty", () => {
  const text = waitFlow("null");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilMessage("it is empty"));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test.each(["ok", "'2026-13-01'", "'$${x}'"])("a text until %s must parse as a date", (value) => {
  const text = waitFlow(value);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilMessage("YAML read it as text that is not a date or datetime"));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test.each([
  "2026-01-01",
  "2026-01-01T10:00:00",
  "'2026-01-01'",
  "'2026-01-01T10:00'",
  "'2026-01-01T10:00:00+07:00'",
])("a date until literal %s loads", (value) => {
  load(waitFlow(value));
});

test("the literal check reaches inline bodies", () => {
  const text = `id: f
name: f
nodes:
  a:
    kind: call
    nodes:
      pause:
        kind: wait
        until: 5
      worker:
        kind: code
        depends_on: [pause]
        output: int
        code: |
          return 1
    output: \${worker.output}
output: \${a.output}
`;
  const error = errorOf(text);
  expect(error.message).toContain(untilMessage("YAML read it as an integer", "pause"));
  expect(startLine(error)).toBe(lineOf(text, "until: 5"));
});

test("a tagged wrong literal reports the type first", () => {
  expect(errorOf(mapFlow("!int 5"), MAP_DEFS).diagnostic.message).toBe(overMessage("YAML read it as an integer"));
  expect(errorOf(waitFlow("!int 5")).diagnostic.message).toBe(untilMessage("YAML read it as an integer"));
});

// ---------- references and expressions, checked by their inferred type ----------

const MODE = "typedefs:\n  Mode: Literal(ok, no)\n";

const overRefMessage = (label: string, node = "m") =>
  `node '${node}' (kind=map): \`over:\` must be a list, but the value is ${label}; write a reference to a list`;

const untilRefMessage = (label: string, node = "w") =>
  `node '${node}' (kind=wait): \`until:\` must be a date or datetime, but the value is ${label}; ` +
  "write a reference to a date, datetime, or text";

/**
 * A flow with `inputs` whose MAP node `m` has `over: <over>` and calls `child`. A code node `c`
 * with `output: int` and one `u` with no declared output are available as producers.
 */
function mapRef(over: string, inputs = "  s: !str\n", typedefs = ""): string {
  return (
    "id: f\nname: f\n" +
    typedefs +
    "input:\n" +
    inputs +
    `defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${echo.output}
  words:
    nodes:
      c:
        kind: code
        output: str
        code: |
          return "a"
    output: \${c.output}
nodes:
  c:
    kind: code
    output: int
    code: |
      return 1
  u:
    kind: code
    code: |
      return [1]
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: \${m.output}
`
  ).replace("@V", () => over);
}

/**
 * A flow with `inputs` whose WAIT node `w` has `until: <until>`, then a code node. A code node `c`
 * with `output: int` and one `u` with no declared output are available as producers.
 */
function waitRef(until: string, inputs = "  s: !str\n", typedefs = ""): string {
  return (
    "id: f\nname: f\n" +
    typedefs +
    "input:\n" +
    inputs +
    `nodes:
  c:
    kind: code
    output: int
    code: |
      return 1
  u:
    kind: code
    code: |
      return "2026-01-01"
  w:
    kind: wait
    until: @V
  after:
    kind: code
    depends_on: [w]
    output: int
    code: |
      return 1
output: \${after.output}
`
  ).replace("@V", () => until);
}

test.each<[string, string, string, string]>([
  ["str", "${input.a}", "  a: !str\n", "str"],
  ["int", "${input.a}", "  a: !int\n", "int"],
  ["bool", "${input.a}", "  a: !bool\n", "bool"],
  ["object", "${input.a}", "  a: !object\n", "object"],
  ["date", "${input.a}", "  a: !date\n", "date"],
  ["optional-int", "${input.a}", "  a: !Optional(int)\n", "Optional(int)"],
  ["optional-str", "${input.a}", "  a: !Optional(str)\n", "Optional(str)"],
  ["literal", "${input.a}", "  a: !Mode\n", "Literal(no, ok)"],
  ["template", "'a ${input.s}'", "  s: !str\n", "str"],
  ["typed-producer", "${c.output}", "  s: !str\n", "int"],
])("a disjoint over reference is a load error (%s)", (_id, over, inputs, label) => {
  const text = mapRef(over, inputs, MODE);
  const error = errorOf(text, MAP_REF_DEFS);
  expect(error.diagnostic.message).toBe(overRefMessage(label));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each<[string, string, string, string]>([
  ["int", "${input.a}", "  a: !int\n", "int"],
  ["bool", "${input.a}", "  a: !bool\n", "bool"],
  ["float", "${input.a}", "  a: !float\n", "float"],
  ["object", "${input.a}", "  a: !object\n", "object"],
  ["list", "${input.a}", "  a: !list(str)\n", "list(str)"],
  ["optional-int", "${input.a}", "  a: !Optional(int)\n", "Optional(int)"],
  ["literal", "${input.a}", "  a: !Mode\n", "Literal(no, ok)"],
  ["typed-producer", "${c.output}", "  s: !str\n", "int"],
])("a disjoint until reference is a load error (%s)", (_id, until, inputs, label) => {
  const text = waitRef(until, inputs, MODE);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilRefMessage(label));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test("the issue flow is a load error at until", () => {
  const text = `id: f
name: f
input:
  n: !int
nodes:
  w:
    kind: wait
    until: \${input.n}
  b:
    kind: code
    depends_on: [w]
    output: str
    code: return 'x'
output: \${b.output}
`;
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilRefMessage("int"));
  expect(startLine(error)).toBe(8);
});

test.each<[string, string]>([
  ["over-list", mapRef("${input.a}", "  a: !list(str)\n")],
  ["over-optional-list", mapRef("${input.a}", "  a: !Optional(list(str))\n")],
  ["over-any", mapRef("${input.a}", "  a: !Any\n")],
  ["over-list-expression", mapRef("${[input.s]}")],
  ["over-arithmetic", mapRef("${input.n + 1}", "  n: !int\n")],
  ["over-untyped-producer", mapRef("${u.output}")],
  ["until-str", waitRef("${input.a}", "  a: !str\n")],
  ["until-date", waitRef("${input.a}", "  a: !date\n")],
  ["until-datetime", waitRef("${input.a}", "  a: !datetime\n")],
  ["until-optional-date", waitRef("${input.a}", "  a: !Optional(date)\n")],
  ["until-any", waitRef("${input.a}", "  a: !Any\n")],
  ["until-untyped-producer", waitRef("${u.output}")],
  ["until-overlapping-literal", waitRef("${input.a}", "  a: !Day\n", "typedefs:\n  Day: Literal(ok, '2026-01-01')\n")],
])("a fitting or unproven source loads (%s)", (_id, text) => {
  load(text, MAP_REF_DEFS);
});

test("a call directive over with a str output is a load error", () => {
  const text = mapRef("call(words)");
  const error = errorOf(text, MAP_REF_DEFS);
  expect(error.diagnostic.message).toBe(overRefMessage("str"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("isinstance keeps its own error", () => {
  let text = mapRef("${isinstance(input.a, int)}", "  a: !int\n");
  let error = errorOf(text, MAP_REF_DEFS);
  expect(error.message).toContain("isinstance is not supported in a map node's over:");
  expect(startLine(error)).toBe(lineOf(text, "over: "));

  text = waitRef("${isinstance(input.a, int)}", "  a: !int\n");
  error = errorOf(text);
  expect(error.message).toContain("isinstance is not supported in a wait node's until:");
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test("a tagged wrong reference reports the type first", () => {
  const text = waitRef("!list(str) ${input.n}", "  n: !int\n");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(untilRefMessage("int"));
  expect(startLine(error)).toBe(lineOf(text, "until: "));
});

test("a record typedef is named", () => {
  const text = mapRef("${input.r}", "  r: !Buy\n", "typedefs:\n  Buy: {action: str}\n");
  const error = errorOf(text, MAP_REF_DEFS);
  expect(error.diagnostic.message).toBe(overRefMessage("Buy"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("a None-typed source is a load error", () => {
  expect(errorOf(mapRef("${input.z}", "  z: !None\n"), MAP_REF_DEFS).diagnostic.message).toBe(overRefMessage("None"));
  expect(errorOf(waitRef("${input.z}", "  z: !None\n")).diagnostic.message).toBe(untilRefMessage("None"));
});

test("the type error comes before the cycle error", () => {
  // `after` waits on `w`, so reading it in `until:` is also a cycle.
  expect(errorOf(waitRef("${after.output}")).diagnostic.message).toBe(untilRefMessage("int"));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-leaves/compose_map_wait_source_types__int_until_reference", call: assembled },
  {
    id: "load-map-call/compose_map_wait_source_types__call_directive_over_with_a_str_output",
    call: assembledWith(MAP_REF_DEFS),
  },
  { id: "load-map-call/compose_map_wait_source_types__int_over_reference", call: assembledWith(MAP_REF_DEFS) },
  {
    id: "load-map-call/compose_map_wait_source_types__record_typedef_over_reference_names_the_typedef",
    call: assembledWith(MAP_REF_DEFS),
  },
])("assemble: $id", runCase);
