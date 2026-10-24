/** A loop driver carries an inbox field into its body and leaves it out of the stopping output. */
import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { Grow, Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { InboxRef, inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { handChild } from "../compose/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });

/** A `while: not closed` chat loop whose body takes inbox `m` and returns `closed`. */
function chatLoop(): LoopNode {
  const childInputs = [inboxParam("m"), new Parameter("closed", { required: true, contract: new ValueContract(BOOL) })];
  const node = new LoopNode("chat", {
    flowId: "turn",
    child: handChild({ inputs: childInputs, output: { closed: "${input.closed}" } }).compiled,
    childInputs,
    predicateKind: "while",
    predicate: "not closed",
    returnedFields: ["closed"],
  });
  node.params = [new Parameter("m"), new Parameter("closed")];
  return node;
}

test("a stopping loop drops the inbox field it carries", () => {
  const out = chatLoop().forward({ m: new InboxRef("m"), closed: true });
  expect(out).toBeInstanceOf(Output);
  expect((out as Output).value).toEqual({ closed: true });
});

test("a continuing loop keeps the inbox field in its seed", () => {
  const out = chatLoop().forward({ m: new InboxRef("m"), closed: false });
  expect(out).toBeInstanceOf(Grow);
  const [carried] = (out as Grow).seed as [Record<string, unknown>, bigint];
  expect(InboxRef.isInboxRef(carried["m"]) || (carried["m"] as { inbox?: string }).inbox === "m").toBe(true);
});

/** A one-node flow holding `node`, so its manifest can be digested. */
const singleNodeFlow = (node: LoopNode) =>
  new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map()]]),
    startId: node.id,
    endId: node.id,
  });

test("a partial loop driver's inbox default has a manifest form", () => {
  const driver = chatLoop().respawn(1, { m: { inbox: "m" }, closed: false });
  const m = driver.params!.find((p) => p.name === "m")!;
  expect(InboxRef.isInboxRef(m.default)).toBe(true);
  const again = chatLoop().respawn(1, { m: { inbox: "m" }, closed: false });
  expect(flowFingerprint(singleNodeFlow(driver))).toBe(flowFingerprint(singleNodeFlow(again)));
  expect(expansionDigest(singleNodeFlow(driver))).toBe(expansionDigest(singleNodeFlow(again)));
  const other = chatLoop().respawn(1, { m: { inbox: "n" }, closed: false });
  expect(flowFingerprint(singleNodeFlow(other))).not.toBe(flowFingerprint(singleNodeFlow(driver)));
});
