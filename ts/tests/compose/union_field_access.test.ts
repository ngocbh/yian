/**
 * Reading fields of a union of records through an output contract, a port of
 * `tests/compose/test_union_field_access.py::test_shared_walk_allows_member_only_fields_when_asked`
 * and `::test_member_only_hint_names_the_union_root`. `typeFor` stands in for compose's
 * `read_type`.
 *
 * Later: the rest needs compose and the runtime.
 */

import { expect, test } from "bun:test";

import { outputContractFieldError, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import type { Type } from "../../src/typesys/values.ts";

/** The tagged union `Decision` of `Buy {action, qty}` and `Sell {action, price}`. */
function decision(): Type {
  const registry = readTypedefs({
    Buy: { action: "Literal(buy)", qty: "int" },
    Sell: { action: "Literal(sell)", price: "float" },
    Decision: "Union(Buy, Sell)",
  });
  return typeFor("Decision", registry);
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
  const registry = readTypedefs({ Text: { text: "str" }, Num: { num: "int" }, Reply: "Union(Text, Num)" });
  const untagged = new ValueContract(typeFor("Reply", registry));
  expect(outputContractFieldError(untagged, ["text"], "v.text", { unionRoot: "s.output" })).toBe(
    "reference ${v.text} reads field 'text', which only 1 of 2 union members have; " +
      "check the member with isinstance(s.output, <Type>) in a case first",
  );
});
