/**
 * The declared-type functions in `compile/output_contracts.ts`, which the loader and the run time
 * share: binding alternatives, the MAP item contract, and the type a contract gives field reads.
 */

import { expect, test } from "bun:test";

import {
  bindingAlternatives,
  contractType,
  itemContract,
  resolveOutputBindingContract,
} from "../../src/compile/output_contracts.ts";
import type { Node } from "../../src/nodes/base.ts";
import { PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { listKindForElement, Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

const str = Type.scalar(ValueKind.STRING);
const strs = new Type({ kind: listKindForElement(str), element: str });
const producers = new Map([["a", new ValueContract(strs)]]);

test("bindingAlternatives types a binding from compile/", () => {
  const alternatives = bindingAlternatives("${a.output}", producers, new Map(), null)!;
  expect(alternatives).toHaveLength(1);
  expect(typeEquals((alternatives[0] as ValueContract).type, strs)).toBe(true);
});

test("itemContract reads a node's item source, given the node", () => {
  const map = { itemSourceKey: "over" } as unknown as Node;
  expect(typeEquals(itemContract(map, new Map([["over", "${a.output}"]]), producers, new Map())!.type, str)).toBe(true);
  expect(itemContract(null, new Map(), producers, new Map())).toBeNull();
});

test("contractType gives a partial record its names, opaque ones untyped", () => {
  const typ = contractType(
    new PartialRecordContract([
      ["n", new ValueContract(str)],
      ["o", null],
    ]),
  )!;
  expect([...typ.fields!.keys()]).toEqual(["n", "o"]);
  expect(typ.fields!.get("o")!.kind).toBe(ValueKind.ANY);
  expect(contractType(null)).toBeNull();
  expect(typeEquals(contractType(new ValueContract(strs))!, strs)).toBe(true);
});

test("a whole-span ?. read is typed Optional, as inferExprType types it", () => {
  const rec = new Type({ kind: ValueKind.OBJECT, fields: new Map([["k", str]]), required: new Set(["k"]) });
  const recs = new Map([["a", new ValueContract(rec)]]);
  const read = resolveOutputBindingContract("${a.output?.k}", recs, new Map()) as ValueContract;
  expect(typeEquals(read.type, str.replace({ nullable: true }))).toBe(true);
  const plain = resolveOutputBindingContract("${a.output.k}", recs, new Map()) as ValueContract;
  expect(typeEquals(plain.type, str)).toBe(true);
});
