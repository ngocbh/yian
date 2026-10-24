/** The START of every flow binds each inbox input to its handle, a constant. */
import { expect, test } from "bun:test";

import { START_ID } from "../../src/compile/model.ts";
import { synthesizeBoundaryGraph } from "../../src/compose/build/boundary.ts";
import { bindParams } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { InboxRef, inboxType, Type, ValueKind } from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });

test("START wires an inbox input to its handle", () => {
  const [, , wiring] = synthesizeBoundaryGraph([inboxParam("m"), new Parameter("n")], [], new Set(), [], null, {
    record: true,
  });
  expect(wiring.get(START_ID)!.get("m")).toEqual({ inbox: "m" });
  expect(wiring.get(START_ID)!.get("n")).toBe("${input.n}");
});

test("START binds the constant to an InboxRef", () => {
  const record = bindParams([inboxParam("m")], new Map([["m", { inbox: "m" }]]), new VariablePool(START_ID));
  expect(InboxRef.isInboxRef(record["m"])).toBe(true);
  expect((record["m"] as InboxRef).inbox).toBe("m");
});
