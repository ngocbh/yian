/** CALL/MAP bindings from `${item}`, literals and null escapes are type-checked at load. */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  assembledWith,
  assembleText,
  type CaseRow,
  caseRegistry,
  handChild,
  loadConformanceFlow,
  loadError,
  parametersFromYaml,
  runCase,
} from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-call-binding-types-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

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
  TY: Literal('true', yes)
  FN: Literal('false', no)
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
        code: return JSON.stringify(inputs["t"])
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

/** The parent flow's typedefs (`R`, `AB`, `TY`, `FN`), which the `child` def's input type may name. */
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
  // "5" for a float input reads as the int 5, which the input keeps.
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
    "FN",
    "false",
    null,
    "child expects 'Literal(false, no)', but the value false does not fit; quote it to keep it as text (YAML reads unquoted true/false as booleans)",
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
  // An element the run can't read `${item...}` on fails, whatever falls back after it.
  [
    "int",
    "${item.x :- 0}",
    "[{x: 1}, {y: 2}]",
    "${item.x} fails on element 1 of over:, where item has no field 'x'; add it to every element, or write item?.x to read null where it is missing",
  ],
  ["int", "${item.a.b}", "[{a: {b: 1}}, {a: 3}]", "${item.a.b} fails on element 1 of over:, where item.a is int, which has no fields"],
  ["int", "${item.a.b}", "[{a: null}]", "${item.a.b} fails on element 0 of over:, where item.a is null; write item.a?.b to read null instead"],
  ["int", "${item?.a.b :- 0}", "[{c: 1}]", "${item?.a.b} fails on element 0 of over:, where item?.a is null; write item?.a?.b to read null instead"],
  [
    "str",
    "${input.m:-1.50}",
    null,
    'child expects \'str\', but the default 1.50 does not fit; quote it to keep it as text: "1.50"',
  ],
  // The `inf` constant is shown as written, as a number is.
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
  ["TY", "false", "child expects 'Literal(true, yes)', but the value false does not fit"],
  // The text form shows [true,"no"] and "no".
  [
    "list(bool)",
    "[true, no]",
    'child expects \'list(bool)\', but the value [true,"no"] does not fit ("no" does not match declared type bool)',
  ],
  // A float spelling past the double range fits no float however it's written.
  ["float", '"1e400"', "child expects 'float', but the value \"1e400\" does not fit"],
  // "inf" reads as infinity, but the hint would spell it `Infinity`, which YAML reads as text.
  ["float", '"inf"', "child expects 'float', but the value \"inf\" does not fit"],
])("no rewrite hint when rewriting cannot help: %s from %s", (type, binding, message) => {
  const error = loadError(() => load(flow(type, binding), child(type)));

  expect(firstLine(error).endsWith(`binding 't' — ${message}`)).toBe(true);
});

test("text spelling a 5000-digit int gets the rewrite hint", () => {
  // Int text has no digit limit.
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

/** The run inputs of the fit rows; `m`, `n` and `d` are left out, so they are null. */
const INPUTS = { s: "x", xs: [1n], ds: ["2026-01-01"], rs: [{ a: 1n }], os: ["x"] };

test.each<[string, string, string | null, unknown]>([
  ["int", "${item}", "${input.xs}", ["1"]],
  ["float", "${item}", "${input.xs}", ["1"]],
  ["int", "${item.a}", "${input.rs}", ["1"]],
  ["float", "5", null, "5"],
  ["date", '${input.d:-"2026-01-01"}', null, '"2026-01-01"'],
  ["date", '"2026-01-01"', null, '"2026-01-01"'],
  ["int(1..3)", "2", null, "2"],
  ["Optional(str)", "null", null, "null"],
  ["str", '${input.m:-"z"}', null, '"z"'],
  ["int", "${input.n:-5}", null, "5"],
  ["str", "${input.m:-input.s}", null, '"x"'],
  ["Optional(str)", "${input.m:-null}", null, "null"],
  ["str", '${item:-"z"}', "${input.os}", ['"x"']],
  ["date", "${item}", '["2026-01-01"]', ['"2026-01-01"']],
  ["int", "${item:-3}", "[null, 1]", ["3", "1"]],
  ["int", "${input.n | 3}", null, "3"],
  ["int", "${item?.x :- 0}", "[{x: 1}, {y: 2}]", ["1", "0"]],
  ["int", "${item.a?.b :- 0}", "[{a: {b: 1}}, {a: null}, {a: {}}]", ["1", "0", "0"]],
])("bindings that fit load and run: %s from %s over %p", async (type, binding, over, output) => {
  const result = await runFlow(await loadConformanceFlow(flow(type, binding, over)), INPUTS, { env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual(output);
});

/** A CALL whose binding comes from `g`, which has no declared output. */
const OPAQUE_TEXT = `id: p
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
        code: return JSON.stringify(inputs["t"])
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

test("text from an unproven source is not converted at the child start (loads)", () => {
  // `g` has no declared output, so its "3" is unproven at load; the child's start refuses it.
  load(OPAQUE_TEXT, child("int"));
});

test("text from an unproven source is not converted at the child start", async () => {
  const result = await runFlow(await loadConformanceFlow(OPAQUE_TEXT), {}, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toContain('"3" does not match declared type int');
});

/** A CALL passing the top-level flow's `int`, `float` and `bool` inputs on to its child. */
const TOP_LEVEL = `id: p
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

test("text supplied to the top-level flow is still converted (loads)", () => {
  const childFlow = handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n  f: !float\n  b: !bool\n", new Map()),
    producers: { e: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${e.output}",
  });

  load(TOP_LEVEL, childFlow);
});

test("text supplied to the top-level flow is still converted", async () => {
  const result = await runFlow(await loadConformanceFlow(TOP_LEVEL), { n: " 3 ", f: "2.5", b: "yes" }, { env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("[3,2.5,true]");
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

// ---------- Pinned conformance cases ----------

test.each<CaseRow>([
  {
    id: "types/compose_open_range_flows__call_with_a_wider_open_range_source",
    call: assembledWith({ child: def("t", "int(0..100)", "echo", "int") }),
  },
])("assembledWith: $id", runCase);

/**
 * A flow whose code node `mk` makes a `Made` record, passed whole to the `child` def's `s: !Small`
 * input through `kind` (a `call`, or a `map` over two of them, read as `${item}`).
 */
function recordFlow(made: string, kind: "call" | "map" = "call"): string {
  const source = kind === "call" ? "${mk.output}" : "${item}";
  return `id: rec
name: rec
typedefs:
  Made: {${made}}
  Small:
    a: int
defs:
  child:
    input:
      s: !Small
    nodes:
      n:
        kind: code
        input:
          s: \${input.s}
        output: int
        code: return inputs.s.a
    output: \${n.output}
nodes:
  mk:
    kind: code
    output: ${kind === "call" ? "Made" : "list(Made)"}
    code: ${kind === "call" ? '"return {a: 1}"' : '"return [{a: 1}, {a: 2}]"'}
  c:
    kind: ${kind}
    ${kind === "map" ? "over: ${mk.output}" : ""}
    call: child
    input:
      s: ${source}
output: \${c.output}
`;
}

test.each<[string, "call" | "map"]>([
  ["a: int, b: int", "call"],
  ["a: int, b: Optional(int)", "call"],
  ["a: int, b: int", "map"],
  ["a: Optional(int)", "call"],
])("a record that may not fit the child's closed record is a load error: {%s} by %s", async (made, kind) => {
  const error = loadError(() => assembleText(recordFlow(made, kind), "flow.yaml"));

  expect(error.message).toContain(`${kind} node 'c': binding 's' — child expects '{"a":int}', source is`);
});

test("a record with the child's fields loads and runs", async () => {
  const result = await runFlow(await loadConformanceFlow(recordFlow("a: int")), {}, { env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(1n);
});

test("a date passed to a str input loads and runs as its ISO text", async () => {
  const text = `id: dt
name: dt
input:
  d: !date
defs:
  child:
    input:
      b: !str
    nodes:
      n:
        kind: code
        input:
          b: \${input.b}
        output: str
        code: return inputs.b
    output:
      b: \${n.output}
nodes:
  c:
    kind: call
    call: child
    input:
      b: \${input.d}
output: \${c.output}
`;
  const result = await runFlow(await loadConformanceFlow(text), { d: "2026-01-02" }, { env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ b: "2026-01-02" });
});

test("a loop seed record with a field the body's input lacks is a load error", () => {
  const text = `id: lp
name: lp
typedefs:
  Big: {a: int, b: Optional(int)}
  Small: {a: int}
input:
  z: !Big
defs:
  step:
    input:
      s: !Small
    nodes:
      bump:
        kind: code
        input:
          s: \${input.s}
        output:
          s: Small
        code: "return {s: inputs.s}"
    output: \${bump.output}
nodes:
  drive:
    kind: loop
    call: step
    times: 2
    input:
      s: \${input.z}
output: \${drive.output}
`;
  const error = loadError(() => assembleText(text, "flow.yaml"));

  expect(error.message).toContain(`loop node 'drive': binding 's' — child expects '{"a":int}', source is`);
});
