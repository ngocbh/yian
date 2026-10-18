/**
 * A binding's declared contract: the four carrier tests of
 * `tests/engine/test_binding_contract_identity.py` (the contract is the third, optional field;
 * only an exact `ValueContract` is accepted; `mapOrigins` rebases the contract and keeps its type;
 * `iterOrigins` yields the binding's origin before its contract's).
 *
 * Changed: `fields()` order is the own-key order of an instance.
 *
 * Left for later: the eight flow tests (fingerprints, manifests, renamespacing, origin traversal,
 * CALL expansion, never hashing a record-typed binding) need compile, compose and the runtime, and
 * land with the compile layer, compose and the runner.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { Binding, ContractOrigins, ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/index.ts";

/** Return an exact int contract with an optional root origin. */
function intContract(origin: OriginRef | null = null): ValueContract {
  return new ValueContract(typeFor("int", new Map()), new ContractOrigins({ default: origin }));
}

test("binding fields end with an optional contract", () => {
  expect(Object.keys(new Binding("${x}"))).toEqual(["source", "origin", "contract"]);
  expect(new Binding("${x}").contract).toBeNull();
  const contract = intContract();
  expect(new Binding("${x}", { contract }).contract).toBe(contract);
});

test("a binding rejects a non-value contract", () => {
  expect(() => new Binding("${x}", { contract: typeFor("int", new Map()) as never })).toThrow(TypeError);
  expect(() => new Binding("${x}", { contract: typeFor("int", new Map()) as never })).toThrow(/ValueContract/);
  expect(() => new Binding("${x}", { contract: "int" as never })).toThrow(TypeError);
  expect(() => new Binding("${x}", { contract: "int" as never })).toThrow(/ValueContract/);
});

test("mapOrigins keeps and rebases the contract", () => {
  const contract = intContract(new OriginRef("type-site"));
  const binding = new Binding("${x}", { origin: new OriginRef("binding-site"), contract });

  const mapped = binding.mapOrigins((origin) => new OriginRef(`mapped:${origin.key}`));

  expect(mapped.source).toBe("${x}");
  expect(mapped.origin).toEqual(new OriginRef("mapped:binding-site"));
  expect(mapped.contract).not.toBeNull();
  expect(mapped.contract!.type).toBe(contract.type);
  expect(mapped.contract!.origins.default).toEqual(new OriginRef("mapped:type-site"));
  expect(new Binding("${x}").mapOrigins((origin) => origin).contract).toBeNull();
});

test("iterOrigins yields the binding's origin, then its contract's", () => {
  const contract = new ValueContract(
    typeFor("int", new Map()),
    new ContractOrigins({ default: new OriginRef("type-site") }),
  );
  const binding = new Binding("${x}", { origin: new OriginRef("binding-site"), contract });

  expect([...binding.iterOrigins()]).toEqual([new OriginRef("binding-site"), new OriginRef("type-site")]);
  expect([...new Binding("${x}", { contract }).iterOrigins()]).toEqual([new OriginRef("type-site")]);
});
