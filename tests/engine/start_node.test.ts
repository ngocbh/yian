/**
 * The START boundary node of `nodes/start/node.ts`: its kind and params, binding and defaults
 * through `evalNode` and `run`, and its constructor's checks.
 */

import { expect, test } from "bun:test";

import { START_ID } from "../../src/compile/model.ts";
import { NodeKind, Output } from "../../src/nodes/base.ts";
import { isExactly, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, nodeFlow } from "./run_state_fixtures.ts";

const INT = Type.scalar(ValueKind.INTEGER);

/** A coercing parameter with a scalar contract (STRING unless `type` is given). */
function parameter(
  name: string,
  options: { default?: unknown; required?: boolean; type?: Type } = {},
): Parameter {
  const hasDefault = Object.hasOwn(options, "default");
  return new Parameter(name, {
    contract: new ValueContract(options.type ?? Type.scalar(ValueKind.STRING)),
    required: options.required ?? false,
    hasDefault,
    default: hasDefault ? options.default : null,
    coerceStrings: true,
  });
}

test("start kind and params are input names", () => {
  const node = new StartNode("__start__", { params: [parameter("topic"), parameter("window")] });
  expect(node.kind).toBe(NodeKind.START);
  // params = the input NAMES (so the evaluator's binder reads their wired sources).
  expect(node.params!.map((p) => p.name)).toEqual(["topic", "window"]);
  expect(node.params!.every((p) => isExactly(p, Parameter))).toBe(true);
});

test("start through eval node fills omitted default", async () => {
  // An input the caller omitted is filled by START's own run template from its declared
  // default, not by the CALL/MAP driver.
  const node = new StartNode("__start__", {
    params: [parameter("topic"), parameter("window", { default: 30n, type: Type.scalar(ValueKind.INTEGER) })],
  });
  expect(await node.run({ topic: "ACME" })).toEqual(new Output({ topic: "ACME", window: 30n }));
});

test("start params default to an empty list", () => {
  expect(new StartNode("__start__").params).toEqual([]);
});

test("start params must be an exact list", () => {
  expect(() => new StartNode("__start__", { params: new Set() as never })).toThrow(
    new TypeError("START params must be an exact list, got Set"),
  );
});

test("start params must contain Parameter values", () => {
  expect(() => new StartNode("__start__", { params: ["topic"] as never })).toThrow(
    new TypeError("START params must contain Parameter values, got string"),
  );
});

test("start commits its input record", () => {
  expect(new StartNode("__start__").commitsInputRecord).toBe(true);
});

test("start through eval node defaults into a record without converting text", async () => {
  const node = new StartNode("__start__", {
    params: [parameter("topic"), parameter("window", { default: 30n, type: INT }), parameter("offset", { type: INT })],
  });
  const pool = new VariablePool();
  pool.set(START_ID, { topic: "ACME", offset: 7n });
  const flow = nodeFlow(node, { topic: "${input.topic}", offset: "${input.offset}" });

  expect(await drive(node, { pool, flow })).toEqual(new Output({ topic: "ACME", offset: 7n, window: 30n }));

  // Text is converted only by `runFlow` before START is seeded, never by START itself.
  pool.set(START_ID, { topic: "ACME", offset: "7" });
  const outcome = await drive(node, { pool, flow });
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("contract.input");
  expect(failure.message).toContain(`input 'offset': "7" does not match declared type int`);
});

test("start e08 type mismatch returns generic input failure", async () => {
  const node = new StartNode("__start__", { params: [parameter("n", { required: true, type: INT })] });
  const pool = new VariablePool();
  pool.set(START_ID, { n: ["not", "an", "int"] });

  const outcome = await drive(node, { pool, flow: nodeFlow(node, { n: "${input.n}" }) });

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("contract.input");
  expect(failure.message).toContain("does not match declared type int");
  expect(failure.trace).toEqual([]);
});

test("start through eval node binds from wiring", async () => {
  // START runs through the engine seam: binding reads its sources from the flow's wiring for
  // START, coerces, and evalNode returns the raw Output record.
  const node = new StartNode("__start__", { params: [parameter("topic")] });
  const pool = new VariablePool();
  pool.set(START_ID, { topic: "ACME" });

  expect(await drive(node, { pool, flow: nodeFlow(node, { topic: "${input.topic}" }) })).toEqual(
    new Output({ topic: "ACME" }),
  );
});

test("start through eval node required unbound fails", async () => {
  // A required input that resolves unbound at the boundary fails loudly (a runtime backstop to
  // the load-time reference check).
  const node = new StartNode("__start__", { params: [parameter("n", { required: true, type: INT })] });

  expect(await drive(node, { flow: nodeFlow(node, {}) })).toBeInstanceOf(EvaluationFailure);
});
