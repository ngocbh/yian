/**
 * A LOOP `while:` null check and what enters the body: the loop-entry helpers of
 * `compose/nodes/loop.ts`, a port of rows of `tests/engine/test_loop_null_narrowing.py`.
 *
 * Ported: `test_while_proves_the_null_checked_paths` (9), `test_until_and_times_prove_nothing` (3)
 * and `test_body_inputs_keep_their_declared_types_after_load`.
 *
 * Changed: `whileNonNullPaths` returns the proven paths as distinct field-name lists in path order,
 * where Python returns a `frozenset` of tuples. The declared-types row parses the loop node
 * (`nodesOf`) and builds it over the body built by hand (`handChild`), where Python loads the
 * flow whose def is the body.
 *
 * TS-only: `nullableAt` on a record field, a path through a union, a missing field or a scalar,
 * and the null type; `loopEntryInputs` relaxing only the parameters under a `while:` path, keeping
 * their origins, and returning an unchanged parameter as is; `carriedFieldTypes` on an exact
 * record, a partly typed record (its opaque fields skipped) and anything else; `whileNonNullPaths`
 * giving a path checked twice once and in order, and nothing for a condition that doesn't parse.
 *
 * Later (T3-19): `test_until_null_check_narrows_neither_the_seed_nor_a_returned_field`,
 * `test_null_check_under_or_narrows_nothing`,
 * `test_returned_field_still_keeps_its_carried_type_under_a_while_null_check`,
 * `test_nested_null_check_keeps_the_carried_type_of_the_returned_record`,
 * `test_while_null_check_rejects_a_record_seed_the_zero_iteration_output_cannot_carry`,
 * `test_while_null_check_rejects_a_null_seed_the_zero_iteration_output_cannot_carry`,
 * `test_seed_of_the_wrong_type_names_the_declared_input_under_a_while_null_check` and
 * `test_returned_field_of_the_wrong_type_names_the_declared_input_under_a_while_check`.
 * Later (T3-17, T3-19): `test_inline_body_reads_a_while_null_checked_seed_as_its_seed_types_it`,
 * `test_inline_returned_field_of_the_wrong_type_names_the_seed_type` and
 * `test_inline_body_under_until_reads_the_seed_as_optional`. Later (T3-17, T4):
 * `test_a_tag_in_the_inline_body_admits_the_while_null_checked_seed`,
 * `test_inline_body_returns_null_into_a_literal_seed_under_a_while_null_check` and
 * `test_inline_body_returning_a_while_null_checked_field_unchanged_runs`. Later (T4):
 * `test_optional_seed_fills_a_strict_body_input_under_a_while_null_check`,
 * `test_optional_seed_pagination_resumes_after_its_first_expansion`,
 * `test_every_seed_shape_is_narrowed_by_a_while_null_check`,
 * `test_while_null_check_keeps_the_case_narrowed_seed` (and T3-19/T3-20),
 * `test_returned_null_fits_a_strict_body_input_under_a_while_null_check`,
 * `test_partial_loop_seed_check_alone_decides_a_carried_over_field` (and T3-19),
 * `test_nested_null_check_narrows_the_record_above_it`,
 * `test_while_null_check_lets_a_seed_record_omit_the_checked_field` (and T3-19) and
 * `test_while_null_check_admits_a_null_seed_the_body_output_can_carry`.
 */

import { expect, test } from "bun:test";

import { BuildContext } from "../../src/compose/nodes/base.ts";
import { carriedFieldTypes, loopEntryInputs, nullableAt, whileNonNullPaths } from "../../src/compose/nodes/loop.ts";
import { PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, Type, ValueKind } from "../../src/typesys/values.ts";
import { handChild, nodesOf, parametersFromYaml } from "../compose/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);

// ---------- the paths a while: proves not null ----------

test.each<[string, string[][]]>([
  ["cursor != null", [["cursor"]]],
  ["n < 3 and cursor != null", [["cursor"]]],
  ["not (cursor == null)", [["cursor"]]],
  ["page.next != null", [["page", "next"]]],
  ["${cursor != null}", [["cursor"]]],
  ["cursor != null or n > 3", []],
  ["isinstance(item, Item)", []],
  ["cursor != null and other != null", [["cursor"], ["other"]]],
  ["(a != null and b != null) or (a != null and n > 1)", [["a"]]],
])("while proves the null-checked paths: %s", (predicate, paths) => {
  expect(whileNonNullPaths("while", predicate)).toEqual(paths);
});

test.each<["until" | "times", string | null]>([
  ["until", "cursor == null"],
  ["until", "cursor != null"],
  ["times", null],
])("until and times prove nothing: %s %s", (kind, predicate) => {
  expect(whileNonNullPaths(kind, predicate)).toEqual([]);
});

test("a path checked twice is given once, the paths in order", () => {
  expect(whileNonNullPaths("while", "b != null and a.x != null and b != null and a != null")).toEqual([
    ["a"],
    ["a", "x"],
    ["b"],
  ]);
});

test("a while that doesn't parse proves nothing", () => {
  expect(whileNonNullPaths("while", "cursor != null and")).toEqual([]);
});

// ---------- the body's declared inputs ----------

// Pagination over the `fetch_page` def, which takes `cursor: str`.
const PAGE = `id: pages
name: pages
nodes:
  pages:
    kind: loop
    while: cursor != null
    input:
      cursor: \${input.start}
    call: fetch_page
`;

/** The parameters a def declares in `input` (the lines under its `input:` key). */
function params(input: string) {
  return parametersFromYaml(`input:\n${input}`, new Map());
}

test("body inputs keep their declared types after load", () => {
  // Only the checks read a null-checked input as nullable; the node keeps `str`, which the body's
  // start checks each value against.
  const body = handChild({
    inputs: params("  cursor: !str\n"),
    producers: { get: new ValueContract(STR.replace({ nullable: true })) },
    output: { cursor: "${get.output}" },
  });
  const desc = nodesOf(PAGE).get("pages")!;

  const built = desc.build(new BuildContext({ registry: new Map(), resolver: () => body })).node;
  expect(built).toBeInstanceOf(LoopNode);
  const node = built as LoopNode;

  const [cursor] = node.childInputs;
  expect(node.childInputs).toHaveLength(1);
  expect(canonicalTypeLabel(cursor!.contract!.type)).toBe("str");
  expect(cursor!.required).toBe(true);
  // The entry checks read it as nullable.
  expect(canonicalTypeLabel(loopEntryInputs(node)[0]!.contract!.type)).toBe("Optional(str)");
});

// ---------- nullableAt ----------

/** A record of `fields`, each required. */
function recordType(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

test("nullableAt makes the value at a path and each record above it nullable", () => {
  const page = recordType({ next: STR, n: INT });

  const relaxed = nullableAt(page, ["next"]);

  expect(relaxed).toStrictEqual(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["next", STR.replace({ nullable: true })],
        ["n", INT],
      ]),
      required: new Set(["n"]),
      nullable: true,
    }),
  );
  expect(canonicalTypeLabel(relaxed)).toBe('Optional({"n":int,"next":Optional(str)})');
  // The type itself is not changed.
  expect(page.nullable).toBe(false);
  expect(page.required).toEqual(new Set(["next", "n"]));
});

test("nullableAt follows a path through nested records", () => {
  const outer = recordType({ page: recordType({ next: STR }) });

  const relaxed = nullableAt(outer, ["page", "next"]);

  expect(relaxed.nullable).toBe(true);
  expect(relaxed.required).toEqual(new Set());
  const page = relaxed.fields!.get("page")!;
  expect(page.nullable).toBe(true);
  expect(page.required).toEqual(new Set());
  expect(page.fields!.get("next")).toStrictEqual(STR.replace({ nullable: true }));
});

test("nullableAt stops at a union and still makes it nullable", () => {
  const registry = readTypedefs({ Done: { answer: "str" }, More: { hint: "str" }, Step: "Union(Done, More)" });
  const step = typeFor("Step", registry);
  const holder = recordType({ r: step });

  const relaxed = nullableAt(holder, ["r", "answer"]);

  // The union's members are left as they are.
  expect(relaxed.fields!.get("r")).toStrictEqual(step.replace({ nullable: true }));
  expect(relaxed.required).toEqual(new Set());
  expect(relaxed.nullable).toBe(true);
});

test("nullableAt stops at a missing field and still makes the record nullable", () => {
  const page = recordType({ next: STR });

  const relaxed = nullableAt(page, ["other", "x"]);

  expect(relaxed).toStrictEqual(page.replace({ nullable: true }));
});

test("nullableAt stops at a scalar and still makes it nullable", () => {
  expect(nullableAt(STR, ["x"])).toStrictEqual(STR.replace({ nullable: true }));
  expect(nullableAt(STR, [])).toStrictEqual(STR.replace({ nullable: true }));
});

test("nullableAt keeps the null type as it is", () => {
  const none = Type.scalar(ValueKind.NONE);

  expect(nullableAt(none, [])).toBe(none);
  expect(nullableAt(recordType({ x: none }), ["x"]).fields!.get("x")).toBe(none);
});

// ---------- loopEntryInputs ----------

/** A LOOP node over a body taking `cursor: str`, `page: {next: str}` and `n: int`, driven by `kind`. */
function pagesLoop(kind: "while" | "until" | "times", predicate: string | null): LoopNode {
  return new LoopNode("pages", {
    flowId: "fetch_page",
    childInputs: parametersFromYaml(
      "input:\n  cursor: !str\n  page: !Page\n  n: !int\n",
      readTypedefs({ Page: { next: "str" } }),
    ),
    predicateKind: kind,
    predicate,
  });
}

test("loopEntryInputs relaxes only the parameters under a while path", () => {
  const node = pagesLoop("while", "cursor != null and page.next != null and n < 3");
  const [cursor, page, n] = node.childInputs;

  const inputs = loopEntryInputs(node);

  expect(inputs.map((parameter) => parameter.name)).toEqual(["cursor", "page", "n"]);
  expect(canonicalTypeLabel(inputs[0]!.contract!.type)).toBe("Optional(str)");
  expect(canonicalTypeLabel(inputs[1]!.contract!.type)).toBe('Optional({"next":Optional(str)})');
  // An input under no path is the same parameter.
  expect(inputs[2]).toBe(n!);
  // A relaxed copy keeps its origins and its other fields.
  for (const [relaxed, declared] of [
    [inputs[0]!, cursor!],
    [inputs[1]!, page!],
  ] as const) {
    expect(relaxed).not.toBe(declared);
    expect(relaxed.origin).not.toBeNull();
    expect(relaxed.origin).toBe(declared.origin);
    expect(relaxed.contract!.origins).toBe(declared.contract!.origins);
    expect(relaxed.required).toBe(declared.required);
  }
  // The node's inputs stay as declared.
  expect(canonicalTypeLabel(node.childInputs[0]!.contract!.type)).toBe("str");
});

test.each<["until" | "times", string | null]>([
  ["until", "cursor != null"],
  ["times", null],
])("loopEntryInputs changes nothing under %s", (kind, predicate) => {
  const node = pagesLoop(kind, predicate);

  const inputs = loopEntryInputs(node);

  expect(inputs).toHaveLength(3);
  inputs.forEach((parameter, i) => expect(parameter).toBe(node.childInputs[i]!));
});

// ---------- carriedFieldTypes ----------

test("carriedFieldTypes gives an exact record's field types", () => {
  const contract = new ValueContract(recordType({ n: INT, cursor: STR }));

  const types = carriedFieldTypes(contract);

  expect(types).toStrictEqual(
    new Map([
      ["n", INT],
      ["cursor", STR],
    ]),
  );
  expect([...types.keys()]).toEqual(["n", "cursor"]);
});

test("carriedFieldTypes gives a partly typed record's proven field types", () => {
  const contract = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["keep", null],
    ["cursor", new ValueContract(STR)],
  ]);

  const types = carriedFieldTypes(contract);

  expect([...types.keys()]).toEqual(["n", "cursor"]);
  expect(types.get("n")).toBe(INT);
  expect(types.get("cursor")).toBe(STR);
});

test("carriedFieldTypes proves no field of anything else", () => {
  expect(carriedFieldTypes(null)).toEqual(new Map());
  expect(carriedFieldTypes(new ValueContract(STR))).toEqual(new Map());
});
