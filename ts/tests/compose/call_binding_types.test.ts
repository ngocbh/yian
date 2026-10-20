/**
 * CALL/MAP bindings from `${item}`, literals and null escapes are type-checked at load: a port of
 * `tests/compose/test_call_binding_types.py`, assembling each flow with `assembleText` where Python
 * calls `load_flow`, its `child` def built by hand (`handChild`: an input `t` of the row's type, read
 * with the flow's typedefs, one `n: str` producer, returning `${n.output}`).
 *
 * Ported: `test_unprovable_fit_is_a_load_error` (31 of 35), asserted with `toContain` as Python
 * asserts `in`; `test_no_rewrite_hint_when_rewriting_cannot_help` (6), asserted on the first line's
 * end as Python asserts `endswith`;
 * `test_inline_call_with_a_former_boolean_word_names_the_boolean_to_write`; the load halves of
 * `test_bindings_that_fit_load_and_run` (16),
 * `test_text_from_an_unproven_source_is_not_converted_at_the_child_start` and
 * `test_text_supplied_to_the_top_level_flow_is_still_converted`.
 *
 * Changed: the four date params of `test_unprovable_fit_is_a_load_error`
 * (`("date", "2026-01-01")`, `("datetime", "2026-01-01T10:00:00")`, `("str", "2026-01-01")` and
 * `("list(date)", "[2026-01-01]")`) load: an unquoted date is text in TS (rule `dates`), so it fits
 * a `str` input, and a `date` or `datetime` input when it parses as one. Its `("float", '"5"')`
 * param hints `write 5 without quotes` (Python: `5.0`): TS reads `"5"` for a `float` input as the
 * int 5, which a float input keeps (rule `messages`). The list row of
 * `test_no_rewrite_hint_when_rewriting_cannot_help` shows `[true,"no"]` and `"no"` (Python:
 * `[true, "no"]` and `'no'`; rule `messages`).
 * `test_no_rewrite_hint_for_text_spelling_an_int_past_the_digit_limit` gets the hint: TS has no
 * int digit limit, so the text reads as that int. The CODE bodies are TS.
 *
 * TS-only: a float spelling past the double range (`"1e400"`) and `"inf"` (YAML reads an unquoted
 * inf as text) for a `float` input get no hint; the
 * harvested `load-map-call/compose_call_binding_types__*` cases, pinned at the end
 * (`…__date_literal_into_a_str_input` is `ok`, rule `dates`); and, as "Pinned harvested cases
 * (TS-only placement)", the cases this check decides whose Python file has no TS mirror:
 * `load-map-call/compose_call_argument_labels__an_argument_literal_type_mismatch` and
 * `types/compose_open_range_flows__call_with_a_wider_open_range_source`.
 *
 * Later (T3-20): `load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none`
 * (its `source is 'None'` needs the CASE narrowing; its Python file has no TS mirror).
 * Later (T4): the run halves of the load rows above.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembledWith,
  assembleText,
  type CaseRow,
  caseRegistry,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

/** The parent flow: `c` binds `t` of a `child` def taking `t: !<type>`, as a CALL, or a MAP over `over`. */
function flow(type: string, binding: string, over: string | null = null): string {
  return `id: p
name: p
input:
  m: !Optional(str)
  n: !Optional(int)
  s: !str
  xs: !list(int)
  ds: !list(str)
  rs: !list(R)
  os: !list(Optional(str))
  d: !Optional(date)
typedefs:
  R:
    a: int
  AB: Literal(a, b)
defs:
  child:
    input:
      t: !${type}
    nodes:
      n:
        kind: code
        input:
          t: \${input.t}
        output: str
        code: return String(inputs["t"])
    output: \${n.output}
nodes:
  c:
    kind: ${over === null ? "call" : "map"}
    ${over === null ? "" : `over: ${over}`}
    call: child
    input:
      t: ${binding}
output: \${c.output}
`;
}

/**
 * A def built by hand: input `name: !<type>` read with `registry`, one producer `producer` of type
 * `output`, returned.
 */
function def(
  name: string,
  type: string,
  producer: string,
  output: string,
  registry: TypeRegistry = new Map(),
): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n  ${name}: !${type}\n`, registry),
    producers: { [producer]: new ValueContract(typeFor(output, new Map())) },
    output: `\${${producer}.output}`,
  });
}

/** The parent flow's typedefs (`R`, `AB`), which the `child` def's input type may name. */
const FLOW_TYPEDEFS = caseRegistry(SourceDocument.parse(flow("str", "null")));

/** The `child` def built by hand: input `t: !<type>`, one `n: str` producer, returned. */
const child = (type: string) => def("t", type, "n", "str", FLOW_TYPEDEFS);

/** Assemble `text` with `child` as its `child` def. */
const load = (text: string, childFlow: ComposedFlow) =>
  assembleText(text, "flow.yaml", { children: { child: childFlow } });

test.each<[string, string, string | null, string]>([
  ["str", "${item}", "${input.xs}", "child expects 'str', source is 'int'"],
  ["str", "${item.a}", "${input.rs}", "child expects 'str', source is 'int'"],
  ["date", "${item}", "${input.ds}", "child expects 'date', source is 'str'"],
  ["str", "${item}", "${input.os}", "child expects 'str', source is 'Optional(str)'"],
  ["int", '"abc"', null, "child expects 'int', but the value \"abc\" does not fit"],
  ["int(1..3)", "9", null, "child expects 'int(1..3)', but the value 9 does not fit"],
  ["str", "null", null, "child expects 'str', but the value null does not fit"],
  ["str", '${input.n:-"x"}', null, "child expects 'str', source is 'int'"],
  ["str", "${input.n:?missing}", null, "child expects 'str', source is 'int'"],
  ["str", "${input.m:-null}", null, "child expects 'str', but the default null does not fit"],
  ["date", '${input.m:-"2026-01-01"}', null, "child expects 'date', source is 'str'"],
  ["date", '${input.d:-"today"}', null, "child expects 'date', but the default \"today\" does not fit"],
  ["str", "${input.m:-input.n}", null, "child expects 'str', source is 'Optional(int)'"],
  ["int", "${item:-3}", '[null, "1"]', "child expects 'int', but the element \"1\" does not fit"],
  ["int", "${input.n :- input.m}", null, "child expects 'int', source is 'Optional(str)'"],
  ["str", "5", null, 'child expects \'str\', but the value 5 does not fit; quote it to keep it as text: "5"'],
  [
    "int",
    '"3"',
    null,
    "child expects 'int', but the value \"3\" does not fit; text is not converted between flows, so write 3 without quotes",
  ],
  // Changed: TS reads "5" for a float input as the int 5, which the input keeps (Python: 5.0).
  [
    "float",
    '"5"',
    null,
    "child expects 'float', but the value \"5\" does not fit; text is not converted between flows, so write 5 without quotes",
  ],
  ["bool", '"yes"', null, "child expects 'bool', but the value \"yes\" does not fit; `yes` is text; write `true`"],
  ["bool", "yes", null, "child expects 'bool', but the value \"yes\" does not fit; `yes` is text; write `true`"],
  ["bool", "yes", "[1, 2]", "child expects 'bool', but the value \"yes\" does not fit; `yes` is text; write `true`"],
  [
    "Optional(bool)",
    "Off",
    null,
    "child expects 'Optional(bool)', but the value \"Off\" does not fit; `Off` is text; write `false`",
  ],
  [
    "bool",
    '"true"',
    null,
    "child expects 'bool', but the value \"true\" does not fit; text is not converted between flows, so write true without quotes",
  ],
  [
    "str",
    "true",
    null,
    "child expects 'str', but the value true does not fit; quote it to keep it as text (YAML reads unquoted true/false as booleans)",
  ],
  [
    "int",
    '${input.n:-"3"}',
    null,
    "child expects 'int', but the default \"3\" does not fit; text is not converted between flows, so write 3 without quotes",
  ],
  ["str", "${input.m:-5}", null, 'child expects \'str\', but the default 5 does not fit; quote it to keep it as text: "5"'],
  [
    "str",
    "${item:-5}",
    "${input.os}",
    'child expects \'str\', but the default 5 does not fit; quote it to keep it as text: "5"',
  ],
  [
    "int",
    "${item}",
    '["1", "2"]',
    "child expects 'int', but the element \"1\" does not fit; text is not converted between flows, so write 1 without quotes",
  ],
  [
    "int",
    "${item.a}",
    '[{a: 2}, {a: "3"}]',
    "child expects 'int', but the element \"3\" does not fit; text is not converted between flows, so write 3 without quotes",
  ],
  [
    "int",
    '${input.n | "3"}',
    null,
    "child expects 'int', but the default \"3\" does not fit; text is not converted between flows, so write 3 without quotes",
  ],
  [
    "str",
    "${input.m:-1.50}",
    null,
    'child expects \'str\', but the default 1.50 does not fit; quote it to keep it as text: "1.50"',
  ],
  // TS-only: the `inf` constant is shown as written, as a number is.
  [
    "str",
    "${input.m:-inf}",
    null,
    'child expects \'str\', but the default inf does not fit; quote it to keep it as text: "inf"',
  ],
])("an unprovable fit is a load error: %s from %s over %p", (type, binding, over, message) => {
  const error = loadError(() => load(flow(type, binding, over), child(type)));

  expect(error.message).toContain(`binding 't' — ${message}`);
});

/** The first line of `error`'s message. */
const firstLine = (error: Error) => error.message.split("\n")[0]!;

test.each<[string, string, string]>([
  ["float", '"nan"', "child expects 'float', but the value \"nan\" does not fit"],
  ["str", "${input.m:-2026-01-01}", "child expects 'str', but the default 2024 does not fit"],
  ["str", "${input.m:-true}", "child expects 'str', but the default true does not fit; quote it to keep it as text"],
  ["int", "yes", "child expects 'int', but the value \"yes\" does not fit"],
  ["AB", "no", "child expects 'Literal(a, b)', but the value \"no\" does not fit"],
  // Changed: the text form shows [true,"no"] and "no" (Python: [true, "no"] and 'no').
  [
    "list(bool)",
    "[true, no]",
    'child expects \'list(bool)\', but the value [true,"no"] does not fit ("no" does not match declared type bool)',
  ],
  // TS-only: a float spelling past the double range fits no float however it's written.
  ["float", '"1e400"', "child expects 'float', but the value \"1e400\" does not fit"],
  // TS-only: "inf" reads as infinity, but the hint would spell it `Infinity`, which YAML reads as text.
  ["float", '"inf"', "child expects 'float', but the value \"inf\" does not fit"],
])("no rewrite hint when rewriting cannot help: %s from %s", (type, binding, message) => {
  const error = loadError(() => load(flow(type, binding), child(type)));

  expect(firstLine(error).endsWith(`binding 't' — ${message}`)).toBe(true);
});

test("text spelling an int past Python's digit limit gets the rewrite hint (Changed)", () => {
  // Python's int-from-text digit limit refuses this text, so Python gives no hint; TS has no limit.
  const digits = "1".repeat(5000);
  const error = loadError(() => load(flow("int", `"${digits}"`), child("int")));

  expect(
    firstLine(error).endsWith(
      `binding 't' — child expects 'int', but the value "${digits}" does not fit; ` +
        `text is not converted between flows, so write ${digits} without quotes`,
    ),
  ).toBe(true);
});

test("an inline call with a former boolean word names the boolean to write", () => {
  const text = flow("bool", "null").replace(
    "  c:\n    kind: call\n    \n    call: child\n    input:\n      t: null\n",
    "  c:\n    kind: code\n    input:\n      x: call(child, t=yes)\n    code: return 1\n",
  );
  const error = loadError(() => load(text, child("bool")));

  const message = firstLine(error);
  expect(
    message.endsWith(
      "node 'c' input 'x' call(child) argument 't' — child expects 'bool', but the value " +
        '"yes" does not fit; `yes` is text; write `true`',
    ),
  ).toBe(true);
  expect(message).not.toContain("without quotes");
});

// test_unprovable_fit_is_a_load_error (date params)
test.each<[string, string]>([
  ["date", "2026-01-01"],
  ["datetime", "2026-01-01T10:00:00"],
  ["str", "2026-01-01"],
  ["list(date)", "[2026-01-01]"],
])("an unquoted date is text, which fits a %s input: %s", (type, binding) => {
  load(flow(type, binding), child(type));
});

test.each<[string, string, string | null]>([
  ["int", "${item}", "${input.xs}"],
  ["float", "${item}", "${input.xs}"],
  ["int", "${item.a}", "${input.rs}"],
  ["float", "5", null],
  ["date", '${input.d:-"2026-01-01"}', null],
  ["date", '"2026-01-01"', null],
  ["int(1..3)", "2", null],
  ["Optional(str)", "null", null],
  ["str", '${input.m:-"z"}', null],
  ["int", "${input.n:-5}", null],
  ["str", "${input.m:-input.s}", null],
  ["Optional(str)", "${input.m:-null}", null],
  ["str", '${item:-"z"}', "${input.os}"],
  ["date", "${item}", '["2026-01-01"]'],
  ["int", "${item:-3}", "[null, 1]"],
  ["int", "${input.n | 3}", null],
])("bindings that fit load: %s from %s over %p", (type, binding, over) => {
  const { flow: composed } = load(flow(type, binding, over), child(type));

  expect(composed.compiled.nodes.has("c")).toBe(true);
});

test("text from an unproven source is not converted at the child start (loads)", () => {
  // `g` has no declared output, so its "3" is unproven at load; the child's start refuses it.
  const text = `id: p
name: p
defs:
  child:
    input:
      t: !int
    nodes:
      n:
        kind: code
        input:
          t: \${input.t}
        output: str
        code: return String(inputs["t"])
    output: \${n.output}
nodes:
  g:
    kind: code
    code: return "3"
  c:
    kind: call
    call: child
    input:
      t: \${g.output}
output: \${c.output}
`;

  load(text, child("int"));
});

test("text supplied to the top-level flow is still converted (loads)", () => {
  const text = `id: p
name: p
input:
  n: !int
  f: !float
  b: !bool
defs:
  child:
    input:
      n: !int
      f: !float
      b: !bool
    nodes:
      e:
        kind: code
        input:
          n: \${input.n}
          f: \${input.f}
          b: \${input.b}
        output: str
        code: return JSON.stringify([inputs["n"], inputs["f"], inputs["b"]])
    output: \${e.output}
nodes:
  c:
    kind: call
    call: child
    input:
      n: \${input.n}
      f: \${input.f}
      b: \${input.b}
output: \${c.output}
`;
  const childFlow = handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n  f: !float\n  b: !bool\n", new Map()),
    producers: { e: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${e.output}",
  });

  load(text, childFlow);
});

// ---------- harvested conformance cases ----------

test.each<[string, string]>([
  ["load-map-call/compose_call_binding_types__arithmetic_default_has_no_rewrite_hint", "str"],
  ["load-map-call/compose_call_binding_types__bool_default_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__bool_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__date_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__float_default_keeps_its_spelling", "str"],
  ["load-map-call/compose_call_binding_types__inline_call_with_a_yaml11_bool_word", "bool"],
  ["load-map-call/compose_call_binding_types__int_default_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__int_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__list_with_a_yaml11_word_has_no_rewrite_hint", "list(bool)"],
  ["load-map-call/compose_call_binding_types__map_item_source_type_mismatch", "str"],
  ["load-map-call/compose_call_binding_types__map_over_literal_element_does_not_fit", "int"],
  ["load-map-call/compose_call_binding_types__null_default_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__null_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__quoted_number_default_into_an_int_input", "int"],
  ["load-map-call/compose_call_binding_types__quoted_number_into_a_float_input", "float"],
  ["load-map-call/compose_call_binding_types__quoted_number_into_an_int_input", "int"],
  ["load-map-call/compose_call_binding_types__text_literal_into_an_int_input", "int"],
  ["load-map-call/compose_call_binding_types__yaml11_bool_word_into_a_bool_input", "bool"],
])("assembledWith: %s", (id, type) => {
  runCase({ id, call: assembledWith({ child: child(type) }) });
});

// ---------- Pinned harvested cases (TS-only placement) ----------

test.each<CaseRow>([
  {
    id: "load-map-call/compose_call_argument_labels__an_argument_literal_type_mismatch",
    call: assembledWith({ child: def("y", "int", "x", "int") }),
  },
  {
    id: "types/compose_open_range_flows__call_with_a_wider_open_range_source",
    call: assembledWith({ child: def("t", "int(0..100)", "echo", "int") }),
  },
])("assembledWith: $id", runCase);
