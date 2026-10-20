/**
 * Load-time checks and types for dotted reads of a file's uri, mime and name: a port of
 * `tests/compose/test_file_field_reads.py`, loading each flow with `assembleText` (`assemble` as
 * the loader calls it) where Python calls `load_flow`.
 *
 * Ported: `test_file_field_type_gives_the_type_of_each_file_field`,
 * `test_project_output_contract_types_the_fields_of_a_file`,
 * `test_another_field_of_a_file_is_a_located_load_error` (but its `flow-assert` id),
 * `test_another_field_of_a_top_level_file_output_is_a_load_error`,
 * `test_another_field_of_an_optional_file_is_a_load_error`, `test_a_field_of_a_file_uri_is_a_load_error`,
 * `test_another_field_of_a_file_flow_input_is_a_load_error`,
 * `test_another_field_of_a_carried_file_in_a_loop_predicate_is_a_load_error`,
 * `test_a_carried_file_uri_in_a_loop_predicate_loads` (the load, not the run),
 * `test_another_field_of_a_file_item_in_a_named_map_is_a_load_error`,
 * `test_another_field_of_a_record_item_file_in_a_named_map_is_a_load_error`,
 * `test_another_field_of_a_file_every_union_member_has_is_a_load_error`,
 * `test_another_field_of_a_file_after_a_constructor_is_a_load_error`,
 * `test_a_constructor_trailer_on_an_optional_file_is_nullable`, the load half of
 * `test_the_uri_of_a_file_after_a_constructor_is_typed_str` (the `!int` binding's error),
 * `test_another_field_of_a_file_in_an_inline_body_is_a_located_load_error` (pinning the whole
 * message), `test_the_uri_of_a_file_into_an_int_typed_binding_is_a_load_error`,
 * `test_a_str_typed_binding_of_a_file_field_loads_like_a_record_field`,
 * `test_the_uri_of_an_optional_file_is_optional_str`, `test_reads_on_object_and_any_stay_unchecked`
 * and `test_a_file_field_into_a_child_input_is_checked_at_load` (4, the cross-flow binding check),
 * pinning the whole message where Python matches a fragment.
 *
 * Changed: the `child` and `child_int` defs are hand-built children (`handChild`) with the defs'
 * signatures and output contracts. The CODE bodies stay Python's text: the flows only load.
 *
 * Pinned at the end, the harvested `load-core/compose_file_field_reads__*` cases the reference
 * validator and the typed-binding check decide, the `child` def built by hand (`assembledWith`).
 *
 * Later (T3-20): the `flow-assert` id of `test_another_field_of_a_file_is_a_located_load_error`
 * and `test_a_node_assert_reading_another_field_of_its_file_output_is_a_load_error` (the assert
 * passes), and `test_a_case_on_the_uri_of_a_file_with_an_int_when_is_a_load_error` (the CASE
 * `when` value check).
 *
 * Later (T4): the flow runs (`test_a_node_assert_reads_the_uri_of_its_file_output`,
 * `test_the_uri_of_a_file_item_in_a_named_map_feeds_a_str_input`,
 * `test_the_uri_of_a_file_every_union_member_has_loads_and_reads`,
 * the run half of `test_the_uri_of_a_file_after_a_constructor_is_typed_str`,
 * `test_an_inline_call_body_reads_the_uri_of_a_file_input`,
 * `test_an_inline_map_body_reads_the_fields_of_a_file_item`,
 * `test_an_inline_call_body_input_from_a_file_uri_is_typed_str`,
 * `test_the_uri_of_a_file_into_a_str_child_input_runs`, `test_the_fields_of_a_file_load_and_read_at_run`),
 * with the case `run-core/compose_file_field_reads__an_inline_call_body_reads_a_file_uri`, and
 * `test_a_growth_fragment_reading_another_field_of_a_file_is_rejected` (runtime growth).
 */

import { expect, test } from "bun:test";

import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { inferExprType } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { isExactly, ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { fileFieldType, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
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

const STR = Type.scalar(ValueKind.STRING);
const OPTIONAL_STR = new Type({ kind: ValueKind.STRING, nullable: true });

test.each<[string, Type | null]>([
  ["uri", STR],
  ["mime", OPTIONAL_STR],
  ["name", OPTIONAL_STR],
  ["urii", null],
  ["__class__", null],
  ["", null],
  // TS-only: a file has no inherited JS property as a field.
  ["constructor", null],
  ["__proto__", null],
])("fileFieldType gives the type of each file field: %p", (name, expected) => {
  expect(fileFieldType(name)).toEqual(expected);
});

test("projectOutputContract types the fields of a file", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["f", Type.scalar(ValueKind.FILE)],
      ["g", new Type({ kind: ValueKind.FILE, nullable: true })],
    ]),
    required: new Set(["f", "g"]),
  });
  const contract = new ValueContract(record);

  /** The projected type of `fields`, requiring the path to project to an exact `ValueContract`. */
  const projected = (...fields: string[]): Type => {
    const result = projectOutputContract(contract, fields);
    expect(isExactly(result, ValueContract)).toBe(true);
    return (result as ValueContract).type;
  };

  expect(typeEquals(projected("f", "uri"), STR)).toBe(true);
  expect(typeEquals(projected("f", "mime"), OPTIONAL_STR)).toBe(true);
  expect(typeEquals(projected("g", "uri"), OPTIONAL_STR)).toBe(true);
  expect(projectOutputContract(contract, ["f", "urii"])).toBe(null);
});

// ---------- other names are located load errors at every reader ----------

const F = "__import__('yian.typesys.values', fromlist=['FileRef']).FileRef(uri='f')";
const NAMED_F =
  "__import__('yian.typesys.values', fromlist=['FileRef']).FileRef(uri='f', mime='text/plain', name='a.txt')";
const URII = "reads field 'urii' on a file, which has only uri, mime, and name";

/**
 * The producer `a`: a file, an absent `Optional(file)`, a record with an `Optional(str)` field,
 * and the opaque `object` and `Any` slots.
 */
const PRODUCER =
  "  a:\n    kind: code\n    output:\n" +
  "      f: file\n      g: Optional(file)\n      r: {opt: Optional(str), s: str}\n      o: object\n      y: Any\n" +
  `    code: |\n      return {'f': ${NAMED_F}, 'g': None, 'r': {'opt': None, 's': 'x'}, 'o': {'k': 1}, 'y': 1}\n`;

/** A flow whose nodes are the producer `a`, then `nodes`, after `head`. */
const flow = (nodes: string, output = "${b.output}", head = "") =>
  `id: p\nname: p\n${head}nodes:\n${PRODUCER}${nodes}output: ${output}\n`;

/** A CODE node `b` binding `x: <binding>` and returning it. */
const reader = (binding: string) =>
  `  b:\n    kind: code\n    input:\n      x: ${binding}\n    code: return inputs['x']\n`;

/** Load `text` with `children` as its defs. */
const load = (text: string, children: ChildFlows = {}) => assembleText(text, "flow.yaml", { children });

/** Expect loading `text` (with `children`) to fail with `message`, located at the line holding `marker`. */
function expectLocated(text: string, message: string, marker: string, children: ChildFlows = {}): void {
  const error = loadError(() => load(text, children));
  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes(marker)) + 1,
  );
}

/** A CASE `gate` holding `cases`, routing to `b` or else `c`. */
const caseGate = (cases: string) =>
  `  gate:\n    kind: case\n${cases}\n    else: c\n` +
  '  b: {kind: code, code: "return 1", output: int}\n  c: {kind: code, code: "return 2", output: int}\n';

test.each([
  ["input", reader("${a.output.f.urii}"), "${b.output}", "x: ${a.output.f.urii}"],
  ["template", reader('"u=${a.output.f.urii}"'), "${b.output}", "u=${a.output.f.urii}"],
  [
    "case-when",
    caseGate("    cases:\n      - when: \"a.output.f.urii == 'f'\"\n        then: b"),
    "${b.output | c.output}",
    'when: "a.output.f.urii',
  ],
  [
    "case-on",
    caseGate("    on: ${a.output.f.urii}\n    cases:\n      - when: f\n        then: b"),
    "${b.output | c.output}",
    "on: ${a.output.f.urii}",
  ],
  ["flow-output", reader("1"), "${a.output.f.urii}", "output: ${a.output.f.urii}"],
])("another field of a file is a located load error: %s", (_id, nodes, output, marker) => {
  expectLocated(flow(nodes, output), URII, marker);
});

test("another field of a top-level file output is a load error", () => {
  const text =
    `id: p\nname: p\nnodes:\n  a:\n    kind: code\n    output: file\n    code: return ${F}\n` +
    reader("${a.output.urii}") +
    "output: ${b.output}\n";
  expectLocated(text, URII, "x: ${a.output.urii}");
});

test("another field of an optional file is a load error", () => {
  expectLocated(flow(reader("${a.output.g.urii}")), URII, "x: ${a.output.g.urii}");
});

test("a field of a file uri is a load error", () => {
  expectLocated(
    flow(reader("${a.output.f.uri.x}")),
    "reads field 'x' on type str, which has no fields",
    "x: ${a.output.f.uri.x}",
  );
});

test("another field of a file flow input is a load error", () => {
  const text = "id: p\nname: p\ninput:\n  f: !file\nnodes:\n" + reader("${input.f.urii}") + "output: ${b.output}\n";
  expectLocated(text, URII, "x: ${input.f.urii}");
});

/** A flow whose inline LOOP `repeat` carries the file `f` until `until`. */
const carriedFile = (until: string) =>
  flow(
    `  repeat:\n    kind: loop\n    until: "${until}"\n    max: 3\n` +
      "    input:\n      f: ${a.output.f}\n" +
      "    nodes:\n      bump:\n        kind: code\n        input:\n          f: ${input.f}\n" +
      "        output:\n          f: file\n        code: \"return {'f': inputs['f']}\"\n" +
      "    output: ${bump.output}\n",
    "${repeat.output.f.uri}",
  );

test("another field of a carried file in a loop predicate is a load error", () => {
  expect(loadError(() => load(carriedFile("f.urii == 'f'"))).message).toContain(
    `\`until:\` reference \${f.urii} ${URII}`,
  );
});

test("a carried file uri in a loop predicate loads", () => {
  load(carriedFile("f.uri == 'f'"));
});

// ---------- MAP items, unions and constructors ----------

/** A def over the one input `s: !<type>` whose node `k` returns `type`, built by hand. */
const def = (type: string) =>
  handChild({
    inputs: parametersFromYaml(`input:\n  s: !${type}\n`, new Map()),
    producers: { k: new ValueContract(typeFor(type, new Map())) },
    output: "${k.output}",
  });

/** The `child` (`s: !str`) and `child_int` (`s: !int`) defs. */
const children = (): ChildFlows => ({ child: def("str"), child_int: def("int") });

/** A flow whose named MAP `b` calls `child` over `input.<over>: !<type>`, binding `s` to `binding`. */
const namedMap = (head: string, over: string, binding: string) =>
  `id: p\nname: p\n${head}input:\n  ${over}\nnodes:\n` +
  `  b:\n    kind: map\n    over: \${input.${over.split(":")[0]}}\n    call: child\n` +
  `    input:\n      s: ${binding}\noutput: \${b.output}\n`;

test("another field of a file item in a named map is a load error", () => {
  expectLocated(namedMap("", "fs: !list(file)", "${item.urii}"), URII, "s: ${item.urii}", children());
});

test("another field of a record item file in a named map is a load error", () => {
  const text = namedMap("typedefs:\n  Item: {f: file}\n", "rs: !list(Item)", "${item.f.urii}");
  expectLocated(text, URII, "s: ${item.f.urii}", children());
});

const UNION_HEAD = "typedefs:\n  A: {f: file, k: int}\n  B: {f: file, j: str}\n  AB: Union(A, B)\n";
const UNION_PRODUCER = `  u:\n    kind: code\n    output: AB\n    code: |\n      return {'f': ${F}, 'k': 1}\n`;

test("another field of a file every union member has is a load error", () => {
  const text = flow(UNION_PRODUCER + reader("${u.output.f.urii}"), "${b.output}", UNION_HEAD);
  expectLocated(text, URII, "x: ${u.output.f.urii}");
});

const REC_HEAD = "typedefs:\n  Rec: {f: file}\n";

test("another field of a file after a constructor is a load error", () => {
  const text = flow(reader("${Rec(f=a.output.f).f.urii}"), "${b.output}", REC_HEAD);
  expect(loadError(() => load(text)).message).toContain(URII);
});

test("the uri of a file after a constructor is typed str (load half)", () => {
  const text = flow(reader("!int ${Rec(f=a.output.f).f.uri}"), "${b.output}", REC_HEAD);
  expect(loadError(() => load(text)).message).toContain("declared int, but the value is str");
});

test("a constructor trailer on an optional file is nullable", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["f", new Type({ kind: ValueKind.FILE, nullable: true })]]),
    required: new Set(["f"]),
  });
  const infer = (text: string) => inferExprType(parseExpr(text), () => null, (name) => (name === "R" ? record : null));

  expect(typeEquals(infer("R(x).f.uri")!, OPTIONAL_STR)).toBe(true);
  expect(typeEquals(infer("R(x).f.mime")!, OPTIONAL_STR)).toBe(true);
  expect(infer("R(x).f.urii")).toBeNull();
});

// ---------- the types of uri, mime and name ----------

test("the uri of a file into an int-typed binding is a load error", () => {
  expect(loadError(() => load(flow(reader("!int ${a.output.f.uri}")))).message).toContain(
    "node 'b' input 'x': declared int, but the value is str",
  );
});

test.each([["a.output.f.uri"], ["a.output.f.mime"], ["a.output.r.opt"]])(
  "a str-typed binding of a file field loads like a record field: %s",
  (path) => {
    load(flow(reader(`!str \${${path}}`)));
  },
);

test("the uri of an optional file is Optional(str)", () => {
  expect(loadError(() => load(flow(reader("!int ${a.output.g.uri}")))).message).toContain(
    "declared int, but the value is Optional(str)",
  );
});

// ---------- guards: opaque slots stay unchecked ----------

test.each([["a.output.o.zzz"], ["a.output.y.zzz"], ["a.output.o.f.urii"]])(
  "reads on object and Any stay unchecked: %s",
  (path) => {
    load(flow(reader(`\${${path}}`)));
  },
);

// ---------- inline bodies ----------

/** A flow whose inline CALL `b` binds `s` to `binding`, over a file input `f`. */
function inlineCall(binding: string): string {
  return (
    "id: p\nname: p\ninput:\n  f: !file\nnodes:\n" +
    `  b:\n    kind: call\n    input:\n      s: ${binding}\n    nodes:\n` +
    '      k: {kind: code, input: {s: "${input.s}"}, code: "return inputs[\'s\']"}\n' +
    "    output: ${k.output}\noutput: ${b.output}\n"
  );
}

/** A flow whose inline MAP `b` over `xs` (typed `over`) binds `s` to `binding`. */
function inlineMap(over: string, binding: string): string {
  return (
    `id: p\nname: p\ntypedefs:\n  Item: {f: file}\ninput:\n  xs: !${over}\nnodes:\n` +
    `  b:\n    kind: map\n    over: \${input.xs}\n    input:\n      s: ${binding}\n` +
    "    nodes:\n" +
    '      k: {kind: code, input: {s: "${input.s}"}, code: "return inputs[\'s\']"}\n' +
    "    output: ${k.output}\noutput: ${b.output}\n"
  );
}

test.each([
  ["call-input", inlineCall("${input.f.urii}"), "${input.f.urii}"],
  ["map-record-item", inlineMap("list(Item)", "${item.f.urii}"), "${item.f.urii}"],
  ["map-file-item", inlineMap("list(file)", "${item.urii}"), "${item.urii}"],
])("another field of a file in an inline body is a located load error: %s", (_name, text, reference) => {
  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    `cannot infer type for inline binding 'b.input.s': reference ${reference} reads field 'urii' on a ` +
      "file, which has only uri, mime, and name; declare its type with a tag (`s: !<type> ...`) or use an " +
      "explicitly typed named `defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes("s: ${")) + 1,
  );
});

// ---------- a file field into a child input ----------

const CHILD_DEFS = `defs:
  child:
    input:
      s: !str
    nodes:
      k: {kind: code, input: {s: "\${input.s}"}, code: "return inputs['s']", output: str}
    output: \${k.output}
  child_int:
    input:
      s: !int
    nodes:
      k: {kind: code, input: {s: "\${input.s}"}, code: "return 1n", output: int}
    output: \${k.output}
`;

test.each([
  ["child", "a.output.f.mime", "child expects 'str', source is 'Optional(str)'"],
  ["child", "a.output.f.name", "child expects 'str', source is 'Optional(str)'"],
  ["child", "a.output.r.opt", "child expects 'str', source is 'Optional(str)'"],
  ["child_int", "a.output.f.uri", "child expects 'int', source is 'str'"],
])("a file field into a child input is checked at load: %s ${%s}", (child, path, message) => {
  const text =
    `id: p\nname: p\n${CHILD_DEFS}nodes:\n${PRODUCER}` +
    `  b:\n    kind: call\n    call: ${child}\n    input:\n      s: \${${path}}\n` +
    "output: ${b.output}\n";

  const error = loadError(() => load(text, children()));

  expect(error.message).toBe(`call node 'b': binding 's' — ${message}`);
});

test.each<CaseRow>([
  { id: "load-core/compose_file_field_reads__another_field_of_a_file_in_an_inline_body", call: assembled },
  { id: "load-core/compose_file_field_reads__the_uri_of_a_file_is_str", call: assembled },
  { id: "load-core/compose_file_field_reads__another_field_of_a_file_input", call: assembled },
  { id: "load-core/compose_file_field_reads__another_field_of_a_map_file_item", call: assembledWith(children()) },
])("assemble: $id", runCase);
