/**
 * A node's binding labels replace a param's name in its binding errors: a port of
 * `test_label_replaces_the_param_name` and `test_without_a_label_the_default_text_stays` of
 * `tests/engine/test_binding_labels.py`. Without a label each message keeps its default
 * `input '<p>'` / `binding '<p>'` text.
 *
 * Changed (rule: infinity is a float value, NaN is not a yian value): the `nonfinite` and
 * `nonfinite-at-path` cases computed infinity (`${input.x * 10}` with `1e308`), which now binds;
 * they keep failing at the same check with a literal NaN source, reading
 * `input 'p': NaN is not a yian value` and `input 'p' at [1]: NaN is not a yian value`. Changed
 * (values in messages use the text form): `"x" does not match declared type int` and
 * `division by zero: 1 / 0`.
 *
 * TS-only: `${input.x * 10}` with `1e308` delivers `Infinity`, and a function label passed to
 * `resolveSource` names a failing template leaf by its path.
 *
 * Later: `test_fresh_node_has_no_binding_labels` and `test_eval_node_uses_the_node_binding_labels`
 * land with the node kinds and the runtime's `evalNode`.
 */

import { expect, test } from "bun:test";

import { BindingError, bindParams, resolveSource } from "../../src/nodes/binding.ts";
import { Binding, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { Type, ValueKind, VariablePool } from "../../src/typesys/index.ts";

const START_ID = "__start__";
const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));

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

// --- TS-only rows --- //

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
