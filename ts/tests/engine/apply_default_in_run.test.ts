/**
 * The base `run` template fills parameter defaults before the pre-asserts: a port of
 * `tests/engine/test_apply_default_in_run.py`. A pre-assert may read an input the caller omitted
 * but its `Parameter` supplies; a caller's explicit `null` shadows the default (`f(x=None)`).
 *
 * Ported: `test_omitted_input_defaulted_before_pre_assert`, `test_bound_null_shadows_default`,
 * `test_explicit_null_default_fills_when_omitted`,
 * `test_compatible_stored_default_is_not_replaced_by_validation`,
 * `test_incompatible_stored_default_fails_without_string_coercion`,
 * `test_apply_default_idempotent_no_double_fill` (both halves) and
 * `test_node_run_rejects_noncanonical_parameter_elements`.
 *
 * TS-only: an `Array` subclass and a `Set` as `params` reject, in place of the tuple rows.
 *
 * Not ported: `test_node_run_rejects_noncanonical_parameter_containers`: both of its rows are
 * tuples, which TS doesn't have.
 */

import { expect, test } from "bun:test";

import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { applyDefaults, BindingError } from "../../src/nodes/binding.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

const INTEGER = new ValueContract(Type.scalar(ValueKind.INTEGER));
const NUMBER = new ValueContract(Type.scalar(ValueKind.NUMBER));
const NULLABLE_INTEGER = new ValueContract(new Type({ kind: ValueKind.INTEGER, nullable: true }));

/** Near-canonical fixture proving direct runs reject carrier subclasses. */
class ParameterSubclass extends Parameter {}

/** An `Array` subclass, which isn't the exact array the template requires. */
class ParameterList extends Array<Parameter> {}

/**
 * A node implementing `forward`, so the base template (defaults, then the asserts) is under
 * test. `forward` records the record it saw, so a test can check what the template filled.
 */
class ImplNode extends Node {
  readonly kind = NodeKind.CODE;
  seen: Record<string, unknown> | null = null;

  override forward(inputs: Record<string, unknown>): NodeResult {
    this.seen = { ...inputs };
    return new Output(inputs);
  }
}

/** Return the parameter `x: int = 5`. */
function defaultFive(): Parameter {
  return new Parameter("x", { contract: INTEGER, hasDefault: true, default: 5n });
}

test("omitted input defaulted before pre-assert", async () => {
  const node = new ImplNode("n");
  node.params = [defaultFive()];
  node.preAsserts = [new Assertion("x > 0", "x > 0")];

  const result = await node.run({});

  expect(result).toBeInstanceOf(Output);
  expect(node.seen).toEqual({ x: 5n });
  expect((result as Output).value).toEqual({ x: 5n });
});

test("bound null shadows default", async () => {
  const node = new ImplNode("n");
  node.params = [defaultFive()];

  const result = await node.run({ x: null });

  expect(node.seen).toEqual({ x: null });
  expect((result as Output).value).toEqual({ x: null });
});

test("explicit null default fills when omitted", async () => {
  const node = new ImplNode("n");
  node.params = [new Parameter("x", { contract: NULLABLE_INTEGER, hasDefault: true, default: null })];

  const result = await node.run({});

  expect(node.seen).toEqual({ x: null });
  expect((result as Output).value).toEqual({ x: null });
});

test("compatible stored default is not replaced by validation", async () => {
  const node = new ImplNode("n");
  const parameter = new Parameter("x", { contract: NUMBER, hasDefault: true, default: 30n });
  node.params = [parameter];

  const result = await node.run({});

  expect(typeof parameter.default).toBe("bigint");
  expect(typeof ((result as Output).value as Record<string, unknown>).x).toBe("bigint");
});

test("incompatible stored default fails without string coercion", async () => {
  const node = new ImplNode("n");
  const parameter = new Parameter("x", { contract: INTEGER, hasDefault: true, default: "30", coerceStrings: true });
  node.params = [parameter];

  await expect(node.run({})).rejects.toBeInstanceOf(BindingError);
  await expect(node.run({})).rejects.toMatchObject({ failureCode: "binding.evaluate" });
  expect(parameter.default).toBe("30");
  expect(node.seen).toBeNull();
});

test("apply default idempotent no double fill", async () => {
  const node = new ImplNode("n");
  node.params = [defaultFive()];

  // The template's fill, then a second fill a START body might repeat: the value stays 5.
  const once = applyDefaults(node.params, {});
  const twice = applyDefaults(node.params, once);
  expect(once).toEqual({ x: 5n });
  expect(twice).toEqual({ x: 5n });

  const result = await node.run({});
  expect((result as Output).value).toEqual({ x: 5n });
});

test.each([
  ["an Array subclass", ParameterList.from([new Parameter("x")])],
  ["a Set", new Set([new Parameter("x")])],
])("node run rejects noncanonical parameter containers: %s", async (_label, params) => {
  const node = new ImplNode("n");
  node.params = params as unknown as Parameter[];

  await expect(node.run({})).rejects.toThrow(TypeError);
  await expect(node.run({})).rejects.toThrow("node 'n' params must be an exact list");
});

test.each([
  ["a string", ["x"]],
  ["a Parameter subclass", [new ParameterSubclass("x")]],
])("node run rejects noncanonical parameter elements: %s", async (_label, params) => {
  const node = new ImplNode("n");
  node.params = params as Parameter[];

  await expect(node.run({})).rejects.toThrow(TypeError);
  await expect(node.run({})).rejects.toThrow("node 'n' params must contain Parameter values");
});
