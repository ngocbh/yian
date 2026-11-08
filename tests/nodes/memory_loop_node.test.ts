/** A loop driver carries a memory field into its body and leaves it out of the stopping output. */
import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { Grow, Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { jsonPart, MemoryRef, memoryType, Type, ValueKind } from "../../src/typesys/values.ts";
import { handChild } from "../helpers/declarations.ts";

const BOOL = Type.scalar(ValueKind.BOOLEAN);
const memory = (identity: string) =>
  new MemoryRef({
    identity,
    type: "FileMemory",
    settings: {},
    version: "v1",
    recall: { text: "notes", cut: null, hint: null },
    tools: [{ name: "read", write: false, description: "Read a file.", parameters: { type: "object" } }],
  });

/** A `while: not done` loop whose body takes memory `m` and returns `done`. */
function memoryLoop(): LoopNode {
  const childInputs = [
    new Parameter("m", { contract: new ValueContract(memoryType("FileMemory", null)), required: true }),
    new Parameter("done", { required: true, contract: new ValueContract(BOOL) }),
  ];
  const node = new LoopNode("work", {
    flowId: "step",
    child: handChild({ inputs: childInputs, output: { done: "${input.done}" } }).compiled,
    childInputs,
    predicateKind: "while",
    predicate: "not done",
    returnedFields: ["done"],
  });
  node.params = [new Parameter("m"), new Parameter("done")];
  return node;
}

test("a stopping loop drops the memory field it carries", () => {
  const out = memoryLoop().forward({ m: memory("t/a"), done: true });
  expect(out).toBeInstanceOf(Output);
  expect((out as Output).value).toEqual({ done: true });
});

test("a continuing loop keeps the memory field in its seed", () => {
  const out = memoryLoop().forward({ m: memory("t/a"), done: false });
  expect(out).toBeInstanceOf(Grow);
  const [carried] = (out as Grow).seed as [Record<string, unknown>, bigint];
  expect(carried["m"]).toEqual(memory("t/a"));
});

/** A one-node flow holding `node`, so its manifest can be digested. */
const singleNodeFlow = (node: LoopNode) =>
  new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map()]]),
    startId: node.id,
    endId: node.id,
  });

test("a partial loop driver's memory default has a manifest form", () => {
  const driver = memoryLoop().respawn(1, { m: memory("t/a"), done: false });
  const m = driver.params!.find((p) => p.name === "m")!;
  expect(MemoryRef.isMemoryRef(m.default)).toBe(true);
  const again = memoryLoop().respawn(1, { m: memory("t/a"), done: false });
  expect(flowFingerprint(singleNodeFlow(driver))).toBe(flowFingerprint(singleNodeFlow(again)));
  expect(expansionDigest(singleNodeFlow(driver))).toBe(expansionDigest(singleNodeFlow(again)));
  const other = memoryLoop().respawn(1, { m: memory("t/b"), done: false });
  expect(flowFingerprint(singleNodeFlow(other))).not.toBe(flowFingerprint(singleNodeFlow(driver)));
});

test("a driver respawned from the seed's JSON form, as a resume reads it, holds a MemoryRef", () => {
  const driver = memoryLoop().respawn(1, { m: jsonPart(memory("t/a"), [], "mapping"), done: false });
  expect(driver.params!.find((p) => p.name === "m")!.default).toEqual(memory("t/a"));
});
