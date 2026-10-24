/**
 * Reading fields of a union of records through an output contract, a port of
 * `tests/compose/test_union_field_access.py::test_shared_walk_allows_member_only_fields_when_asked`,
 * `::test_member_only_hint_names_the_union_root` and
 * `::test_shared_field_one_member_does_not_require_reads_as_optional` and
 * `::test_human_input_answer_schema_for_a_union_is_one_object_entry` (an ask's); and the child-signature
 * rows `::test_tag_field_reads_as_the_merged_literal`,
 * `::test_shared_field_of_one_type_reads_as_that_type`,
 * `::test_shared_field_optional_in_one_member_reads_as_optional`,
 * `::test_shared_field_through_an_optional_union_reads_as_optional` and
 * `::test_shared_field_of_differing_types_is_untyped_and_loads` (the signature; its load below);
 * `::test_map_over_a_union_output_gives_a_list_of_objects`,
 * `::test_loop_carrying_a_union_keeps_the_non_record_error` and
 * `::test_non_exhaustive_on_over_the_tag_names_the_missing_member`. `typeFor` stands in for compose's
 * `read_type`. The child-signature rows build the child with `handChild` (a producer `d` of the
 * given type and a record output), where Python loads a flow; the MAP row builds a
 * `MapDescriptor` over a hand-built child returning its `Decision` producer, where Python loads a
 * flow whose def is that child. The LOOP row loads Python's flow with `assembleText` (without the
 * unused `Text`, `Num` and `Reply` typedefs) and pins the whole message; its harvested case
 * `load-types/compose_union_field_access__loop_carrying_a_union_keeps_the_non_record_error` is
 * pinned at the end. The non-exhaustive row loads Python's flow with `assembleText` (without the
 * unused `Text`, `Num` and `Reply` typedefs); the uncovered tags are written as JSON text,
 * `["hold"]` (Python `['hold']`).
 *
 * TS-only: the member-only fix hints, which Python checks only through flows the loader builds
 * (a tag comparison hint read from the local path, through a union root and below a shared
 * field; an `isinstance` hint; an untagged union keeping the case hint under both hints), the
 * shared-field walk (continuing into a nested union, stopping at an untyped shared field, and
 * reading below a merged `Literal` field), and reading below a file field.
 *
 * Ported through `assembleText` (`assemble` as the loader calls it) where Python calls
 * `load_flow`, the member-only rows reference validation decides:
 * `::test_member_only_field_of_a_tagged_union_is_a_load_error`,
 * `::test_member_only_field_of_an_untagged_union_is_a_load_error`,
 * `::test_member_only_field_in_a_flow_assert_is_a_load_error` (the whole message),
 * `::test_member_only_field_in_a_searched_case_when_is_a_load_error`,
 * `::test_field_no_member_has_keeps_the_unknown_field_message`,
 * `::test_member_only_field_through_a_record_field_suggests_the_union_path`,
 * `::test_member_only_field_of_a_typed_flow_input_is_a_load_error`,
 * `::test_member_only_field_of_a_map_item_is_a_load_error` (its `each` def built by hand),
 * `::test_member_only_field_in_a_node_assert_is_a_load_error` (the whole message),
 * `::test_shared_field_in_a_node_assert_loads`,
 * `::test_member_only_field_in_an_inline_child_body_is_a_load_error`,
 * `::test_shared_field_in_an_inline_child_body_loads` and the load half of
 * `::test_shared_field_of_differing_types_is_untyped_and_loads`; and
 * `::test_canonicalize_inline_type_keeps_variants`. Their harvested
 * `load-types/compose_union_field_access__member_only_*` cases and
 * `__shared_field_in_a_node_assert_loads` are pinned at the end, the map item's `each` def built by
 * hand (`assembledWith`).
 *
 * Ported through `prepareExpanded`: `::test_runtime_growth_allows_member_only_fields`, where Python
 * calls the private `_validate_growth_wiring_refs`. The grown fragment's START carries the
 * `Decision` contract and its END wires `value` from it (the `run_state_fixtures.ts` spawner and
 * fragment); a member-only read prepares, and an unknown field is a `GrowthRejected`.
 */

import { expect, test } from "bun:test";

import { Edge, GrowthRejected } from "../../src/compile/model.ts";
import { canonicalizeInlineType } from "../../src/compose/build/inline_inputs.ts";
import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { childSignature } from "../../src/compose/nodes/resolver.ts";
import { answerSchema } from "../../src/compose/nodes/tool.ts";
import { Grow } from "../../src/nodes/base.ts";
import {
  Binding,
  isExactly,
  outputContractFieldError,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { prepareExpanded } from "../../src/runtime/state.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { fragment, initializedSpawner } from "../engine/run_state_fixtures.ts";
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

test("ask answer schema for a union is one object entry", () => {
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

test("a non-exhaustive on over the tag names the missing member", () => {
  const text =
    "id: p\nname: p\n" +
    "typedefs:\n" +
    "  Buy: {action: 'Literal(buy)', qty: int, note: str}\n" +
    "  Hold: {action: 'Literal(hold)', note: str}\n" +
    "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n" +
    "  Decision: Union(Buy, Hold, Sell)\n" +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  d:\n    kind: code\n    input:\n      x: ${input.s}\n    output: Decision\n    code: return None\n" +
    "  route:\n    kind: case\n    on: ${d.output.action}\n    cases:\n" +
    "      - when: buy\n        then: sink\n" +
    "      - when: sell\n        then: sink\n" +
    "  sink:\n    kind: code\n    input:\n      x: ${input.s}\n    output: str\n    code: return 'x'\n" +
    "output: ${sink.output}\n";

  expect(loadError(() => assembleText(text)).message).toContain('enum tag(s) ["hold"] not covered');
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

test("runtime growth allows member-only fields", () => {
  /** Prepare growing a fragment whose END reads `field` of its START's `Decision` output. */
  const grow = (field: string) => {
    const { state } = initializedSpawner();
    return prepareExpanded(state, "spawn", new Grow({}), () => {
      const built = fragment("spawn");
      built.nodes.get(built.startId)!.outputContract = new ValueContract(decision());
      const end = EndNode.value(built.endId, { outputName: "value" });
      end.commitAs = "spawn";
      built.nodes.set(built.endId, end);
      built.wiring.set(built.endId, new Map([["value", new Binding(`\${${built.startId}.output.${field}}`)]]));
      built.edges[0] = new Edge(built.edges[0]!.id, built.startId, built.endId, { inputGroup: "value" });
      return built;
    });
  };

  grow("qty");
  expect(() => grow("zzz")).toThrow(GrowthRejected);
  expect(() => grow("zzz")).toThrow("reads unknown field 'zzz' on record");
});

test("canonicalizeInlineType keeps variants", () => {
  const canonical = canonicalizeInlineType(decision());
  expect(typeEquals(canonical, decision())).toBe(true);
  expect(canonical.variants).toBeDefined();
});

// ---------- member-only reads at load ----------

const FLOW_TYPEDEFS =
  "typedefs:\n" +
  "  Buy: {action: 'Literal(buy)', qty: int, note: str}\n" +
  "  Hold: {action: 'Literal(hold)', note: str}\n" +
  "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n" +
  "  Decision: Union(Buy, Hold, Sell)\n" +
  "  Text: {text: str}\n" +
  "  Num: {num: int}\n" +
  "  Reply: Union(Text, Num)\n";

const TAGGED_QTY =
  "reference ${d.output.qty} reads field 'qty', which only union members with action " +
  "buy, sell have; read it inside a case on ${d.output.action}";

/** A flow over `typedefs` with one CODE node `d` typed `output`, plus `rest` (YAML tail). */
const flow = (rest: string, output = "Decision", typedefs = FLOW_TYPEDEFS) =>
  "id: p\nname: p\n" +
  typedefs +
  "input:\n  s: !str\nnodes:\n" +
  `  d:\n    kind: code\n    input:\n      x: \${input.s}\n    output: ${output}\n    code: return None\n` +
  rest;

/** Return the message of the `LoadError` loading `text` with `children` as its defs raises. */
const failure = (text: string, children: ChildFlows = {}) =>
  loadError(() => assembleText(text, "flow.yaml", { children })).message;

const SINK = "  sink:\n    kind: code\n    input:\n      x: ${input.s}\n    output: str\n    code: return 'x'\n";

test("a member-only field of a tagged union is a load error", () => {
  expect(failure(flow("output:\n  q: ${d.output.qty}\n"))).toContain(`flow output from: ${TAGGED_QTY}`);
});

test("a member-only field of an untagged union is a load error", () => {
  expect(failure(flow("output:\n  t: ${d.output.text}\n", "Reply"))).toContain(
    "reference ${d.output.text} reads field 'text', which only 1 of 2 union members " +
      "have; check the member with isinstance(d.output, <Type>) in a case first",
  );
});

test("a member-only field in a flow assert is a load error", () => {
  expect(failure(flow("asserts:\n  - d.output.qty > 0\noutput: ${d.output}\n"))).toBe(
    "assert 'd.output.qty > 0': reference ${d.output.qty} reads field 'qty', which only union members with " +
      "action buy, sell have; check the member first with `isinstance(d.output, <Type>) and ...`",
  );
});

test("a member-only field in a searched case when is a load error", () => {
  const rest =
    "  route:\n    kind: case\n    cases:\n      - when: ${d.output.qty > 5}\n        then: sink\n    else: sink\n" +
    SINK +
    "output: ${sink.output}\n";
  expect(failure(flow(rest))).toContain(`node 'route' cases[0].when from: ${TAGGED_QTY}`);
});

test("a field no member has keeps the unknown field message", () => {
  expect(failure(flow("output:\n  z: ${d.output.zzz}\n"))).toContain(
    "reference ${d.output.zzz} reads unknown field 'zzz' on record",
  );
});

test("a member-only field through a record field suggests the union path", () => {
  expect(failure(flow("output:\n  q: ${d.output.decision.qty}\n", "{decision: Decision}"))).toContain(
    "reference ${d.output.decision.qty} reads field 'qty', which only union members " +
      "with action buy, sell have; read it inside a case on ${d.output.decision.action}",
  );
});

test("a member-only field of a typed flow input is a load error", () => {
  const text =
    "id: p\nname: p\n" +
    FLOW_TYPEDEFS +
    "input:\n  d: !Decision\nnodes:\n" +
    "  step:\n    kind: code\n    input:\n      v: ${input.d.qty}\n    output: str\n    code: return 'x'\n" +
    "output: ${step.output}\n";
  expect(failure(text)).toContain(
    "node 'step' input 'v' from: reference ${input.d.qty} reads field 'qty', which only " +
      "union members with action buy, sell have; read it inside a case on ${input.d.action}",
  );
});

/** The `each` def: input `v: !int` into `inner`, whose `str` it returns. */
const each = (): ChildFlows => ({
  each: handChild({
    inputs: parametersFromYaml("input:\n  v: !int\n", new Map()),
    producers: { inner: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${inner.output}",
  }),
});

test("a member-only field of a map item is a load error", () => {
  const text =
    "id: p\nname: p\n" +
    FLOW_TYPEDEFS +
    "input:\n  xs: !list(Decision)\nnodes:\n" +
    "  m:\n    kind: map\n    call: each\n    over: ${input.xs}\n    input:\n      v: ${item.qty}\n" +
    "output: ${m.output}\n";
  expect(failure(text, each())).toContain(
    "node 'm' map input 'v' from: reference ${item.qty} reads field 'qty', which only " +
      "union members with action buy, sell have; read it inside a case on ${item.action}",
  );
});

/** A flow whose CODE node `d` asserts `assertion`. */
const assertFlow = (assertion: string) => flow(`    asserts:\n      - ${assertion}\noutput: \${d.output}\n`);

test("a member-only field in a node assert is a load error", () => {
  expect(failure(assertFlow("output.qty > 0"))).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'd' assert: reference ${output.qty} reads field 'qty', which only union members with " +
      "action buy, sell have; check the member first with `isinstance(output, <Type>) and ...`",
  );
});

test("a shared field in a node assert loads", () => {
  assembleText(assertFlow("output.note != ''"));
});

/** A flow whose inline CALL `c` binds `v` to `d`'s union and reads `${input.v.<field>}` inside. */
const inlineChild = (field: string) =>
  flow(
    "  c:\n    kind: call\n    input:\n      v: ${d.output}\n    nodes:\n" +
      `      inner:\n        kind: code\n        input:\n          w: \${input.v.${field}}\n` +
      "        output: str\n        code: return 'x'\n" +
      "    output: ${inner.output}\noutput: ${c.output}\n",
  );

test("a member-only field in an inline child body is a load error", () => {
  expect(failure(inlineChild("qty"))).toContain(
    "reference ${input.v.qty} reads field 'qty', which only union members with action " +
      "buy, sell have; read it inside a case on ${input.v.action}",
  );
});

test("a shared field in an inline child body loads", () => {
  assembleText(inlineChild("note"));
});

test("a shared field of differing types loads", () => {
  const typedefs =
    "typedefs:\n  A: {kind: 'Literal(a)', v: int}\n  B: {kind: 'Literal(b)', v: str}\n  AB: Union(A, B)\n";
  assembleText(flow("output:\n  v: ${d.output.v}\n", "AB", typedefs));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  ...[
    "loop_carrying_a_union_keeps_the_non_record_error",
    "member_only_field_in_a_case_when",
    "member_only_field_in_a_flow_assert",
    "member_only_field_in_a_flow_output",
    "member_only_field_in_a_node_assert",
    "member_only_field_in_an_inline_body",
    "member_only_field_of_a_typed_flow_input",
    "member_only_field_of_an_untagged_union",
    "shared_field_in_a_node_assert_loads",
  ].map((name) => ({ id: `load-types/compose_union_field_access__${name}`, call: assembled })),
  {
    id: "load-types/compose_union_field_access__member_only_field_of_a_map_item",
    call: assembledWith(each()),
  },
])("assemble: $id", runCase);
