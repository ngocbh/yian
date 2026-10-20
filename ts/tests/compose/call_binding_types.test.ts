/**
 * CALL/MAP bindings from `${item}`, literals and null escapes are type-checked at load: a port of
 * `tests/compose/test_call_binding_types.py`, assembling each flow with `assembleText` where Python
 * calls `load_flow`, its `child` def built by hand (`handChild`: an input `t` of the row's type, one
 * `n: str` producer, returning `${n.output}`).
 *
 * Ported: the params of `test_unprovable_fit_is_a_load_error` whose message gives no rewrite hint
 * (`source is …`, or `does not fit` with nothing after it), asserted with `toContain` as Python
 * asserts `in`; the load halves of `test_bindings_that_fit_load_and_run` (16),
 * `test_text_from_an_unproven_source_is_not_converted_at_the_child_start` and
 * `test_text_supplied_to_the_top_level_flow_is_still_converted`.
 *
 * Changed: the four date params of `test_unprovable_fit_is_a_load_error`
 * (`("date", "2026-01-01")`, `("datetime", "2026-01-01T10:00:00")`, `("str", "2026-01-01")` and
 * `("list(date)", "[2026-01-01]")`) load: an unquoted date is text in TS (rule `dates`), so it fits
 * a `str` input, and a `date` or `datetime` input when it parses as one. The CODE bodies are TS.
 *
 * TS-only: the harvested `load-map-call/compose_call_binding_types__*` cases whose message gives no
 * rewrite hint, pinned at the end (`…__date_literal_into_a_str_input` is `ok`, rule `dates`); and,
 * as "Pinned harvested cases (TS-only placement)", the cases this check decides whose Python file
 * has no TS mirror: `load-map-call/compose_call_argument_labels__an_argument_literal_type_mismatch`
 * and `types/compose_open_range_flows__call_with_a_wider_open_range_source`.
 *
 * Later (T3-19, rewrite hints): the params of `test_unprovable_fit_is_a_load_error` whose message
 * gives a hint, `test_no_rewrite_hint_when_rewriting_cannot_help` (6),
 * `test_no_rewrite_hint_for_text_spelling_an_int_past_the_digit_limit`,
 * `test_inline_call_with_a_former_boolean_word_names_the_boolean_to_write`, and the harvested
 * `load-map-call/compose_call_binding_types__*` cases whose message gives one.
 * Later (T3-20): `load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none`
 * (its `source is 'None'` needs the CASE narrowing; its Python file has no TS mirror).
 * Later (T4): the run halves of the load rows above.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembledWith,
  assembleText,
  type CaseRow,
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

/** The `child` def built by hand: input `t: !<type>`, one `n: str` producer, returned. */
function child(type: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n  t: !${type}\n`, new Map()),
    producers: { n: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${n.output}",
  });
}

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
])("an unprovable fit is a load error: %s from %s over %p", (type, binding, over, message) => {
  const error = loadError(() => load(flow(type, binding, over), child(type)));

  expect(error.message).toContain(`binding 't' — ${message}`);
});

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
  ["load-map-call/compose_call_binding_types__date_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__map_item_source_type_mismatch", "str"],
  ["load-map-call/compose_call_binding_types__null_default_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__null_literal_into_a_str_input", "str"],
  ["load-map-call/compose_call_binding_types__text_literal_into_an_int_input", "int"],
])("assembledWith: %s", (id, type) => {
  runCase({ id, call: assembledWith({ child: child(type) }) });
});

// ---------- Pinned harvested cases (TS-only placement) ----------

/** A def built by hand: input `name: !<type>`, one producer `producer` of type `output`, returned. */
function def(name: string, type: string, producer: string, output: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n  ${name}: !${type}\n`, new Map()),
    producers: { [producer]: new ValueContract(typeFor(output, new Map())) },
    output: `\${${producer}.output}`,
  });
}

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
