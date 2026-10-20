/**
 * Reading fields of a union of records through an output contract, a port of
 * `tests/compose/test_union_field_access.py::test_shared_walk_allows_member_only_fields_when_asked`,
 * `::test_member_only_hint_names_the_union_root` and
 * `::test_shared_field_one_member_does_not_require_reads_as_optional` and
 * `::test_human_input_answer_schema_for_a_union_is_one_object_entry`; and the child-signature
 * rows `::test_tag_field_reads_as_the_merged_literal`,
 * `::test_shared_field_of_one_type_reads_as_that_type`,
 * `::test_shared_field_optional_in_one_member_reads_as_optional`,
 * `::test_shared_field_through_an_optional_union_reads_as_optional` and
 * `::test_shared_field_of_differing_types_is_untyped_and_loads` (the signature, not the load);
 * `::test_map_over_a_union_output_gives_a_list_of_objects` and
 * `::test_loop_carrying_a_union_keeps_the_non_record_error`. `typeFor` stands in for compose's
 * `read_type`. The child-signature rows build the child with `handChild` (a producer `d` of the
 * given type and a record output), where Python loads a flow; the MAP row builds a
 * `MapDescriptor` over a hand-built child returning its `Decision` producer, where Python loads a
 * flow whose def is that child. The LOOP row loads Python's flow with `assembleText` (without the
 * unused `Text`, `Num` and `Reply` typedefs) and pins the whole message; its harvested case
 * `load-types/compose_union_field_access__loop_carrying_a_union_keeps_the_non_record_error` is
 * pinned at the end.
 *
 * TS-only: the member-only fix hints, which Python checks only through flows the loader builds
 * (a tag comparison hint read from the local path, through a union root and below a shared
 * field; an `isinstance` hint; an untagged union keeping the case hint under both hints), the
 * shared-field walk (continuing into a nested union, stopping at an untyped shared field, and
 * reading below a merged `Literal` field), and reading below a file field.
 *
 * Later: the rest needs compose and the runtime.
 */

import { expect, test } from "bun:test";

import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { answerSchema } from "../../src/compose/nodes/human_input.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { childSignature } from "../../src/compose/nodes/resolver.ts";
import {
  isExactly,
  outputContractFieldError,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

const DECISION_TYPEDEFS = {
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Decision: "Union(Buy, Sell)",
};

/** The tagged union `Decision` of `Buy {action, qty}` and `Sell {action, price}`. */
function decision(): Type {
  return typeFor("Decision", readTypedefs(DECISION_TYPEDEFS));
}

/** The untagged union `Reply` of `Text {text}` and `Num {num}`. */
function reply(): Type {
  return typeFor("Reply", readTypedefs({ Text: { text: "str" }, Num: { num: "int" }, Reply: "Union(Text, Num)" }));
}

test("shared walk allows member-only fields when asked", () => {
  const contract = new ValueContract(decision());
  expect(outputContractFieldError(contract, ["qty"], "d.output.qty")).toBe(
    "reference ${d.output.qty} reads field 'qty', which only union members with " +
      "action buy have; read it inside a case on ${d.output.action}",
  );
  expect(outputContractFieldError(contract, ["qty"], "d.output.qty", { partialUnionFields: true })).toBe(null);
  expect(outputContractFieldError(contract, ["zzz"], "d.output.zzz", { partialUnionFields: true })).toBe(
    "reference ${d.output.zzz} reads unknown field 'zzz' on record",
  );
});

test("human_input answer schema for a union is one object entry", () => {
  expect(answerSchema(decision())).toEqual([{ name: "answer", type: "object", required: true }]);
});

test("member-only hint names the union root", () => {
  const contract = new ValueContract(decision());
  expect(outputContractFieldError(contract, ["qty"], "v.qty", { unionRoot: "d.output" })).toBe(
    "reference ${v.qty} reads field 'qty', which only union members with " +
      "action buy have; read it inside a case on ${d.output.action}",
  );
  const nested = new PartialRecordContract([
    ["order", contract],
    ["note", null],
  ]);
  expect(outputContractFieldError(nested, ["order", "qty"], "v.order.qty", { unionRoot: "d.output" })).toBe(
    "reference ${v.order.qty} reads field 'qty', which only union members with " +
      "action buy have; read it inside a case on ${d.output.order.action}",
  );
  const untagged = new ValueContract(reply());
  expect(outputContractFieldError(untagged, ["text"], "v.text", { unionRoot: "s.output" })).toBe(
    "reference ${v.text} reads field 'text', which only 1 of 2 union members have; " +
      "check the member with isinstance(s.output, <Type>) in a case first",
  );
});

test("member-only hints point at a tag comparison or an isinstance conjunct", () => {
  const contract = new ValueContract(decision());
  const buyFirst = "reference ${v.qty} reads field 'qty', which only union members with action buy have; ";
  expect(outputContractFieldError(contract, ["qty"], "v.qty", { tagHint: true, unionRoot: "d.output" })).toBe(
    buyFirst + 'check the member first with `v.action == "buy" and ...`',
  );
  expect(outputContractFieldError(contract, ["qty"], "v.qty", { isinstanceHint: true, unionRoot: "d.output" })).toBe(
    buyFirst + "check the member first with `isinstance(d.output, <Type>) and ...`",
  );
  expect(outputContractFieldError(contract, ["qty"], "v.qty", { isinstanceHint: true, tagHint: true })).toBe(
    buyFirst + 'check the member first with `v.action == "buy" and ...`',
  );

  const untagged = new ValueContract(reply());
  const caseFirst =
    "reference ${v.text} reads field 'text', which only 1 of 2 union members have; " +
    "check the member with isinstance(s.output, <Type>) in a case first";
  expect(outputContractFieldError(untagged, ["text"], "v.text", { tagHint: true, unionRoot: "s.output" })).toBe(
    caseFirst,
  );
  const bothHints = { tagHint: true, isinstanceHint: true, unionRoot: "s.output" };
  expect(outputContractFieldError(untagged, ["text"], "v.text", bothHints)).toBe(caseFirst);
});

test("a shared union field continues the walk with its shared type", () => {
  const registry = readTypedefs({
    ...DECISION_TYPEDEFS,
    BoxA: { kind: "Literal(a)", order: "Decision" },
    BoxB: { kind: "Literal(b)", order: "Decision" },
    Boxes: "Union(BoxA, BoxB)",
    Word: { v: "str" },
    Count: { v: "int" },
    Mixed: "Union(Word, Count)",
  });
  const boxes = new ValueContract(typeFor("Boxes", registry));
  const orderBuyFirst = "reference ${v.order.qty} reads field 'qty', which only union members with action buy have; ";
  expect(outputContractFieldError(boxes, ["order", "qty"], "v.order.qty", { unionRoot: "d.output" })).toBe(
    orderBuyFirst + "read it inside a case on ${d.output.order.action}",
  );
  const tagHint = { tagHint: true, unionRoot: "d.output" };
  expect(outputContractFieldError(boxes, ["order", "qty"], "v.order.qty", tagHint)).toBe(
    orderBuyFirst + 'check the member first with `v.order.action == "buy" and ...`',
  );
  expect(outputContractFieldError(boxes, ["order", "zzz"], "v.order.zzz")).toBe(
    "reference ${v.order.zzz} reads unknown field 'zzz' on record",
  );

  const contract = new ValueContract(decision());
  expect(outputContractFieldError(contract, ["action"], "d.output.action")).toBe(null);
  expect(outputContractFieldError(contract, ["action", "x"], "d.output.action.x")).toBe(
    "reference ${d.output.action.x} reads field 'x' on type Literal(buy, sell), which has no fields",
  );
  // `v` is `str` in one member and `int` in the other, so the read is untyped and not checked below.
  const mixed = new ValueContract(typeFor("Mixed", registry));
  expect(outputContractFieldError(mixed, ["v", "x"], "m.output.v.x")).toBe(null);
});

test("a file field read continues with the field's type", () => {
  const file = new ValueContract(Type.scalar(ValueKind.FILE));
  expect(outputContractFieldError(file, ["uri"], "f.uri")).toBe(null);
  expect(outputContractFieldError(file, ["uri", "x"], "f.uri.x")).toBe(
    "reference ${f.uri.x} reads field 'x' on type str, which has no fields",
  );
});

test("a shared field one member does not require reads as optional", () => {
  const text = Type.scalar(ValueKind.STRING);
  const tag = (value: string) => new Type({ kind: ValueKind.STRING, tags: new Set([value]) });
  const union = new Type({
    kind: ValueKind.OBJECT,
    variants: [
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([
          ["kind", tag("a")],
          ["note", text],
        ]),
        required: new Set(["kind", "note"]),
      }),
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([
          ["kind", tag("b")],
          ["note", text],
        ]),
        required: new Set(["kind"]),
      }),
    ],
  });
  const projected = projectOutputContract(new ValueContract(union), ["note"]);
  expect(isExactly(projected, ValueContract)).toBe(true);
  const optionalText = new Type({ kind: ValueKind.STRING, nullable: true });
  expect(typeEquals((projected as ValueContract).type, optionalText)).toBe(true);
});

const TRADE_TYPEDEFS = {
  Buy: { action: "Literal(buy)", qty: "int", note: "str" },
  Hold: { action: "Literal(hold)", note: "str" },
  Sell: { action: "Literal(sell)", qty: "int", price: "float", note: "str" },
  Decision: "Union(Buy, Hold, Sell)",
};

/**
 * Return the contract of output field `name` of a child whose producer `d` is typed `output`
 * over `typedefs` and whose record output reads `source` (`null` when untyped).
 */
function outputField(
  name: string,
  source: string,
  options: { output?: string; typedefs?: Readonly<Record<string, unknown>> } = {},
): ValueContract | null {
  const { output = "Decision", typedefs = TRADE_TYPEDEFS } = options;
  const child = handChild({
    producers: { d: new ValueContract(typeFor(output, readTypedefs(typedefs))) },
    output: { [name]: source },
  });
  const contract = childSignature(child).output;
  if (isExactly(contract, PartialRecordContract)) {
    return contract.fields.find(([field]) => field === name)![1];
  }
  expect(isExactly(contract, ValueContract)).toBe(true);
  return new ValueContract((contract as ValueContract).type.fields!.get(name)!);
}

test("a tag field reads as the merged literal", () => {
  const contract = outputField("a", "${d.output.action}");
  const merged = new Type({ kind: ValueKind.STRING, tags: new Set(["buy", "hold", "sell"]) });
  expect(typeEquals(contract!.type, merged)).toBe(true);
});

test("a shared field of one type reads as that type", () => {
  const contract = outputField("n", "${d.output.note}");
  expect(typeEquals(contract!.type, Type.scalar(ValueKind.STRING))).toBe(true);
});

test("a shared field optional in one member reads as optional", () => {
  const typedefs = { ...TRADE_TYPEDEFS, Hold: { action: "Literal(hold)", note: "Optional(str)" } };
  const contract = outputField("n", "${d.output.note}", { typedefs });
  expect(typeEquals(contract!.type, new Type({ kind: ValueKind.STRING, nullable: true }))).toBe(true);
});

test("a shared field through an optional union reads as optional", () => {
  const contract = outputField("n", "${d.output.note}", { output: "Optional(Decision)" });
  expect(typeEquals(contract!.type, new Type({ kind: ValueKind.STRING, nullable: true }))).toBe(true);
});

test("a shared field of differing types is untyped", () => {
  const typedefs = {
    A: { kind: "Literal(a)", v: "int" },
    B: { kind: "Literal(b)", v: "str" },
    AB: "Union(A, B)",
  };
  expect(outputField("v", "${d.output.v}", { output: "AB", typedefs })).toBeNull();
});

test("a map over a union output gives a list of objects", () => {
  const each = handChild({
    inputs: parametersFromYaml("input:\n  v: !str\n", new Map()),
    producers: { inner: new ValueContract(typeFor("Decision", readTypedefs(TRADE_TYPEDEFS))) },
    output: "${inner.output}",
  });
  const desc = new MapDescriptor({ id: "m", call: "each", over: "${input.xs}", inputs: { v: "${item}" } });

  const contract = desc.build(new BuildContext({ registry: new Map(), resolver: () => each })).node
    .outputContract as ValueContract;

  expect(contract.type.kind).toBe(ValueKind.LIST_OBJECT);
  expect(contract.type.element!.variants).toBeDefined();
});

test("a loop carrying a union keeps the non-record error", () => {
  const text =
    "id: p\nname: p\n" +
    "typedefs:\n" +
    "  Buy: {action: 'Literal(buy)', qty: int, note: str}\n" +
    "  Hold: {action: 'Literal(hold)', note: str}\n" +
    "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n" +
    "  Decision: Union(Buy, Hold, Sell)\n" +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  drive:\n" +
    "    kind: loop\n" +
    "    times: 2\n" +
    "    input:\n      action: buy\n" +
    "    nodes:\n" +
    "      step:\n" +
    "        kind: code\n" +
    "        input:\n          action: ${input.action}\n" +
    "        output: Decision\n" +
    "        code: return null\n" +
    "    output: ${step.output}\n" +
    "output: ${drive.output}\n";

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "loop node 'drive': body inline body of node 'drive' must return a record with at least one carried " +
      'field (a subset of the carried record ["action"]); write `output: {field: ...}` or a whole reference ' +
      "to a node with a record output",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_union_field_access__loop_carrying_a_union_keeps_the_non_record_error", call: assembled },
])("assemble: $id", runCase);
