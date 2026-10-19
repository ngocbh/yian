/**
 * The START boundary node of `nodes/start/node.ts`: a port of `tests/engine/test_start_node.py`.
 * Ported: `test_start_kind_and_params_are_input_names`.
 *
 * Changed: `test_start_through_eval_node_fills_omitted_default` runs the node's own `run`, whose
 * template fills the omitted default through `applyDefaults`, instead of `eval_node` with a
 * wiring that leaves the input unbound.
 *
 * TS-only: omitted params are an empty list, the constructor's two `TypeError`s (params not an
 * exact array, an item not an exact `Parameter`) and the `commitsInputRecord` trait read on an
 * instance.
 *
 * Not ported: `tests/engine/test_node_range.py::test_node_has_no_compose_range_metadata`, which
 * guards against Python-era range fields TS nodes never had.
 *
 * Later (T4-6): `test_start_through_eval_node_defaults_into_a_record_without_converting_text`,
 * `test_start_e08_type_mismatch_returns_generic_input_failure`,
 * `test_start_through_eval_node_binds_from_wiring`,
 * `test_start_through_eval_node_required_unbound_fails`.
 */

import { expect, test } from "bun:test";

import { NodeKind, Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

/** A coercing parameter with a scalar contract (STRING unless `type` is given). */
function parameter(name: string, options: { default?: unknown; type?: Type } = {}): Parameter {
  const hasDefault = Object.hasOwn(options, "default");
  return new Parameter(name, {
    contract: new ValueContract(options.type ?? Type.scalar(ValueKind.STRING)),
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
  expect(node.params!.every((p) => p instanceof Parameter)).toBe(true);
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
