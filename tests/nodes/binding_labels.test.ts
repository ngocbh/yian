/**
 * A node's binding labels replace a param's name in its binding errors; without a label each
 * message keeps its default `input '<p>'` / `binding '<p>'` text.
 */

import { expect, test } from "bun:test";

import { BindingError, bindParams, resolveSource } from "../../src/nodes/binding.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { Type, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { drive, FuncNode } from "../helpers/run_state_fixtures.ts";

const START_ID = "__start__";
const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));
const N_RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["n", Type.scalar(ValueKind.INTEGER)]]),
  required: new Set(["n"]),
});
const RECORD = new ValueContract(N_RECORD);
const ITEMS = new ValueContract(new Type({ kind: ValueKind.LIST_OBJECT, element: N_RECORD }));
const BAD_ITEMS = { items: [{ n: 1n }, { n: "x" }] };

/** Return a pool whose flow input holds `inputs`. */
function poolOf(inputs: Record<string, unknown>): VariablePool {
  const pool = new VariablePool();
  pool.set(START_ID, inputs);
  return pool;
}

/** Bind one parameter (`p` by default) to `source` and return the error thrown. */
function fail(
  source: unknown,
  pool: VariablePool,
  { param = null, labels = null }: { param?: Parameter | null; labels?: ReadonlyMap<string, string> | null } = {},
): BindingError {
  const parameter = param ?? new Parameter("p");
  try {
    bindParams([parameter], new Map([[parameter.name, source]]), pool, { labels });
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("bindParams did not throw");
}

type Case = [
  id: string,
  source: unknown,
  inputs: Record<string, unknown>,
  param: Parameter | null,
  labeled: string,
  unlabeled: string,
  code: string,
];

const CASES: Case[] = [
  [
    "nonfinite",
    NaN,
    {},
    null,
    "flow output: NaN is not a yian value",
    "input 'p': NaN is not a yian value",
    "binding.evaluate",
  ],
  [
    "nonfinite-at-path",
    [1.5, NaN],
    {},
    null,
    "flow output at [1]: NaN is not a yian value",
    "input 'p' at [1]: NaN is not a yian value",
    "binding.evaluate",
  ],
  [
    "typed-binding",
    new Binding("${input.s}", { contract: INT }),
    { s: "x" },
    null,
    'flow output: "x" does not match declared type int',
    "binding 'p': \"x\" does not match declared type int",
    "binding.type",
  ],
  [
    "template-leaf",
    new ValueTemplate({ a: "${input.x / input.z}" }),
    { x: 1.0, z: 0.0 },
    null,
    "flow output at a: division by zero: 1 / 0",
    "binding 'p.a': division by zero: 1 / 0",
    "binding.evaluate",
  ],
  [
    "nested-typed-binding",
    new ValueTemplate({ a: new Binding("${input.s}", { contract: INT }) }),
    { s: "x" },
    null,
    'flow output at a: "x" does not match declared type int',
    "binding 'p.a': \"x\" does not match declared type int",
    "binding.type",
  ],
  [
    "contract-input",
    "${input.s}",
    { s: "x" },
    new Parameter("p", { contract: INT }),
    'flow output: "x" does not match declared type int',
    "input 'p': \"x\" does not match declared type int",
    "contract.input",
  ],
  [
    "typed-binding-at-path",
    new Binding("${input.items}", { contract: ITEMS }),
    BAD_ITEMS,
    null,
    'flow output at [1].n: "x" does not match declared type int',
    "binding 'p' at [1].n: \"x\" does not match declared type int",
    "binding.type",
  ],
  [
    "nested-typed-binding-at-path",
    new ValueTemplate({ a: new Binding("${input.items}", { contract: ITEMS }) }),
    BAD_ITEMS,
    null,
    'flow output at a[1].n: "x" does not match declared type int',
    "binding 'p.a' at [1].n: \"x\" does not match declared type int",
    "binding.type",
  ],
  [
    "typed-binding-unknown-key",
    new Binding("${input.r}", { contract: RECORD }),
    { r: { n: 1n, "a.b": 1n } },
    null,
    'flow output at ["a.b"]: record has unknown fields: ["a.b"]',
    "binding 'p' at [\"a.b\"]: record has unknown fields: [\"a.b\"]",
    "binding.type",
  ],
  [
    "contract-input-at-path",
    "${input.items}",
    BAD_ITEMS,
    new Parameter("p", { contract: ITEMS }),
    'flow output at [1].n: "x" does not match declared type int',
    "input 'p' at [1].n: \"x\" does not match declared type int",
    "contract.input",
  ],
  [
    "contract-input-missing-field",
    "${input.r}",
    { r: {} },
    new Parameter("p", { contract: RECORD }),
    'flow output at n: record missing required fields: ["n"]',
    "input 'p' at n: record missing required fields: [\"n\"]",
    "contract.input",
  ],
];

test.each(CASES)("a label replaces the param name: %s", (_id, source, inputs, param, labeled, _unlabeled, code) => {
  const error = fail(source, poolOf(inputs), { param, labels: new Map([["p", "flow output"]]) });

  expect([error.message, error.failureCode]).toEqual([labeled, code]);
});

test.each(CASES)(
  "without a label the default text stays: %s",
  (_id, source, inputs, param, _labeled, unlabeled, code) => {
    const plain = fail(source, poolOf(inputs), { param });
    const other = fail(source, poolOf(inputs), { param, labels: new Map([["q", "flow output"]]) });

    expect([plain.message, plain.failureCode]).toEqual([unlabeled, code]);
    expect(other.message).toBe(unlabeled);
  },
);

test("a fresh node has no binding labels", () => {
  expect(new CodeNode("n", { code: "return 1" }).bindingLabels.size).toBe(0);
});

test("an infinite result binds", () => {
  const record = bindParams(
    [new Parameter("p")],
    new Map([["p", "${input.x * 10}"]]),
    poolOf({ x: 1e308 }),
    { labels: new Map([["p", "flow output"]]) },
  );

  expect(record).toEqual({ p: Infinity });
});

test("a function label names a failing template leaf by its path", () => {
  const source = new ValueTemplate({ a: { b: "${input.x / input.z}" } });
  let error: unknown;
  try {
    resolveSource(source, poolOf({ x: 1.0, z: 0.0 }), { name: "p", label: (path) => `field ${path.join(".")}` });
  } catch (caught) {
    error = caught;
  }

  expect(error).toBeInstanceOf(BindingError);
  const failure = error as BindingError;
  expect([failure.message, failure.failureCode]).toEqual(["field a.b: division by zero: 1 / 0", "binding.evaluate"]);
});

test("a function label names a typed binding's path, then the failing place in its value", () => {
  const source = new ValueTemplate({ a: new Binding("${input.items}", { contract: ITEMS }) });
  let error: unknown;
  try {
    resolveSource(source, poolOf(BAD_ITEMS), { name: "p", label: (path) => `field ${path.join(".")}` });
  } catch (caught) {
    error = caught;
  }

  expect(error).toBeInstanceOf(BindingError);
  const failure = error as BindingError;
  expect([failure.message, failure.failureCode]).toEqual([
    'field a at [1].n: "x" does not match declared type int',
    "binding.type",
  ]);
});

test("eval node uses the node binding labels", async () => {
  const node = new FuncNode("work", (inputs) => inputs.p, { reads: { p: NaN } });
  node.bindingLabels = new Map([["p", "X"]]);

  const outcome = await drive(node);

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect((outcome as EvaluationFailure).failure.message).toBe("X: NaN is not a yian value");
});
