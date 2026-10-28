/**
 * The END boundary node of `nodes/end/node.ts`: its record, value and list modes, their params and
 * captured output paths, the constructor's checks, and post asserts through `evalNode`.
 */

import { expect, test } from "bun:test";

import { NodeKind, Output } from "../../src/nodes/base.ts";
import { Assertion, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, nodeFlow } from "../helpers/run_state_fixtures.ts";

/** A value END named `n` whose post-assert `expression` reads the synth param `param`. */
function endWithSynthAssert(param: string, expression: string, display: string): EndNode {
  const node = EndNode.value("__end__", { outputName: "n" });
  node.postAsserts = [new Assertion(expression, display)];
  node.params = [...node.params!, new Parameter(param)];
  return node;
}

test("end kind", () => {
  expect(EndNode.record("__end__", { outputNames: ["a"] }).kind).toBe(NodeKind.END);
});

// RECORD mode (a flow's `output: {...}`): always a record keyed by the declared names.
test("end record zero outputs returns empty record", async () => {
  const out = await EndNode.record("__end__", { outputNames: [] }).run({});
  expect(out).toBeInstanceOf(Output);
  expect((out as Output).value).toEqual({});
});

test("end record one output returns record", async () => {
  const out = await EndNode.record("__end__", { outputNames: ["only"] }).run({ only: 42n });
  expect((out as Output).value).toEqual({ only: 42n });
});

test("end record one output captures a keyed path", () => {
  const node = EndNode.record("__end__", { outputNames: ["only"] });
  expect(node.captureBackwardContext()).toEqual({ input_output_paths: { only: ["only"] } });
});

test("end record two outputs returns keyed object", async () => {
  const node = EndNode.record("__end__", { outputNames: ["report", "n"] });
  const out = await node.run({ report: "r", n: 4n });
  expect((out as Output).value).toEqual({ report: "r", n: 4n });
});

test("end record params are output names", () => {
  const node = EndNode.record("__end__", { outputNames: ["report", "n"] });
  expect(node.params!.map((p) => p.name)).toEqual(["report", "n"]);
});

// VALUE mode (a flow's `output: ${x}`, or no output): returns the one value or null.
test("end value returns the bare value", async () => {
  const node = EndNode.value("__end__", { outputName: "only" });
  expect(node.params!.map((p) => p.name)).toEqual(["only"]);
  expect(node.returnsRecord).toBe(false);
  expect(((await node.run({ only: 42n })) as Output).value).toBe(42n);
});

test("end value without a name has no params and returns none", async () => {
  const node = EndNode.value("__end__", { outputName: null });
  expect(node.params).toEqual([]);
  expect(((await node.run({})) as Output).value).toBeNull();
});

test("end value captures the whole output path", () => {
  const node = EndNode.value("__end__", { outputName: "only" });
  expect(node.backwardInputNames()).toEqual(["only"]);
  expect(node.captureBackwardContext()).toEqual({ input_output_paths: { only: [] } });
});

test("end record returns record", () => {
  expect(EndNode.record("__end__", { outputNames: ["a", "b"] }).returnsRecord).toBe(true);
});

test("end value rejects more than one name", () => {
  expect(() => new EndNode("__end__", { outputNames: ["a", "b"], value: true })).toThrow(RangeError);
  expect(() => new EndNode("__end__", { n: 1, value: true })).toThrow(RangeError);
});

test("end takes exactly one of outputNames or n", () => {
  const message = "EndNode takes exactly one of outputNames (value/record) or n (list)";
  expect(() => new EndNode("__end__")).toThrow(new RangeError(message));
  expect(() => new EndNode("__end__", { outputNames: ["a"], n: 1 })).toThrow(new RangeError(message));
});

test("end record missing an input rejects", async () => {
  const node = EndNode.record("__end__", { outputNames: ["report", "n"] });
  await expect(node.run({ report: "r" })).rejects.toThrow(new RangeError("END input 'n' is missing"));
});

test("end output named constructor reads its own value", async () => {
  const node = EndNode.record("__end__", { outputNames: ["constructor"] });
  expect(((await node.run({ constructor: 7n })) as Output).value).toEqual({ constructor: 7n });
});

// LIST mode (the MAP fan-in): joins in over order (index).
test("end list joins in over order", async () => {
  const node = EndNode.list("__end__", { n: 3 });
  const out = await node.run({ e0: "ACME", e1: "BETA", e2: "GOOG" });
  expect((out as Output).value).toEqual(["ACME", "BETA", "GOOG"]);
});

test("end list n zero returns empty list", async () => {
  const out = await EndNode.list("__end__", { n: 0 }).run({});
  expect((out as Output).value).toEqual([]);
});

test("end list params are e0 to en", () => {
  const node = EndNode.list("__end__", { n: 2 });
  expect(node.params!.map((p) => p.name)).toEqual(["e0", "e1"]);
});

test("end post asserts fire record scoped through eval node", async () => {
  // The producer ref is lowered to a synth param fed by a `${emit.output.n:-null}` wiring, so
  // binding puts it into the record and the assert reads it node-locally.
  const pool = new VariablePool();
  pool.set("emit", { n: 4n });
  const wiring = { n: "${emit.output.n}", __assert_emit: "${emit.output.n:-null}" };

  const node = endWithSynthAssert("__assert_emit", "__assert_emit > 0", "emit must be positive");
  expect(await drive(node, { pool, flow: nodeFlow(node, wiring) })).toEqual(new Output(4n));

  const bad = endWithSynthAssert("__assert_emit", "__assert_emit > 100", "emit must exceed 100");
  const failed = await drive(bad, { pool, flow: nodeFlow(bad, wiring) });
  expect(failed).toBeInstanceOf(EvaluationFailure);
  expect((failed as EvaluationFailure).failure.message).toContain("post-assert failed");
});

test("end post asserts resolve namespaced cross node ref via synth param", async () => {
  // A namespaced cross-node ref (the shape expansion gives a fanned-in child END) is lowered the
  // same way: the assert holds or fails record-scoped, with no END special case.
  const pool = new VariablePool();
  pool.set("each#0/n", { score: 7n });
  const wiring = { n: "${each#0/n.output.score}", __assert_score: "${each#0/n.output.score:-null}" };

  const okNode = endWithSynthAssert("__assert_score", "__assert_score > 0", "score must be positive");
  expect(await drive(okNode, { pool, flow: nodeFlow(okNode, wiring) })).toEqual(new Output(7n));

  const badNode = endWithSynthAssert("__assert_score", "__assert_score > 100", "score must exceed 100");
  const failed = await drive(badNode, { pool, flow: nodeFlow(badNode, wiring) });
  expect(failed).toBeInstanceOf(EvaluationFailure);
  expect((failed as EvaluationFailure).failure.message).toContain("post-assert failed");
});
