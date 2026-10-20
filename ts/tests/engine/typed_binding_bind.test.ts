/**
 * The binder checks a typed binding's value against its declared type: a port of
 * `tests/engine/test_typed_binding_bind.py`. A typed binding checks the resolved value,
 * converting only its file and object representation, and fails with `binding.type`; the
 * receiving parameter's own contract still runs afterwards.
 *
 * Ported (adapted): `test_binding_type_failure_keeps_the_route_without_usable_inputs` passes
 * Python's `wiring_override` as the wiring of a one-node flow (`nodeFlow`), since an override is a
 * JSON record and this one holds a `Binding`; Python's bare `Node` is a minimal subclass.
 *
 * Later: the three run tests
 * (`test_run_fails_the_consumer_with_binding_type`, `test_map_run_fails_an_item_with_binding_type`,
 * `test_run_fails_a_typed_flow_output_with_binding_type`) land with compose and the runner.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { BindingError, bindParams } from "../../src/nodes/binding.ts";
import { Binding, ContractOrigins, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { Type, type Value, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { nodeFlow } from "./run_state_fixtures.ts";

const INTEGER = Type.scalar(ValueKind.INTEGER);
const NUMBER = Type.scalar(ValueKind.NUMBER);
const OPTIONAL_INTEGER = new Type({ kind: ValueKind.INTEGER, nullable: true });

/** A minimal concrete node; it fails before its body here. */
class CodeNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
}

/** Return a pool holding each entry of `outputs` as that node's output. */
function poolOf(outputs: Record<string, unknown> = {}): VariablePool {
  const pool = new VariablePool();
  for (const [nodeId, value] of Object.entries(outputs)) pool.set(nodeId, value);
  return pool;
}

/** Bind one parameter (`qty` by default) to a typed binding of `source`. */
function bind(
  source: unknown,
  contract: ValueContract,
  pool: VariablePool,
  { param = null, item = null }: { param?: Parameter | null; item?: Value } = {},
): ValueRecord {
  const parameter = param ?? new Parameter("qty");
  return bindParams([parameter], new Map([[parameter.name, new Binding(source, { contract })]]), pool, { item });
}

/** Return the `BindingError` `call` throws. */
function bindingError(call: () => unknown): BindingError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("call did not throw");
}

test("a matching value is delivered unchanged", () => {
  const record = bind("${a.output}", new ValueContract(NUMBER), poolOf({ a: 30n }));

  expect(record).toEqual({ qty: 30n });
  expect(typeof record["qty"]).toBe("bigint");
});

test("a mismatch fails with binding.type at the contract origin", () => {
  const contractOrigin = new OriginRef("contract:qty");
  const bindingOrigin = new OriginRef("binding:qty");
  const contract = new ValueContract(INTEGER, new ContractOrigins({ default: contractOrigin }));

  const error = bindingError(() =>
    bindParams(
      [new Parameter("qty")],
      new Map([["qty", new Binding("${a.output}", { origin: bindingOrigin, contract })]]),
      poolOf({ a: "many" }),
    ),
  );

  expect(error.failureCode).toBe("binding.type");
  expect(error.message.startsWith("binding 'qty':")).toBe(true);
  expect(error.origin).toEqual(contractOrigin);
});

test("a nested mismatch uses the nearest contract origin", () => {
  const rootOrigin = new OriginRef("contract:root");
  const fieldOrigin = new OriginRef("contract:n");
  const recordType = new Type({ kind: ValueKind.OBJECT, fields: new Map([["n", INTEGER]]), required: new Set(["n"]) });
  const contract = new ValueContract(
    recordType,
    new ContractOrigins({ default: rootOrigin, entries: [[["n"], fieldOrigin]] }),
  );

  const error = bindingError(() => bind("${a.output}", contract, poolOf({ a: { n: "x" } })));

  expect(error.failureCode).toBe("binding.type");
  expect(error.origin).toEqual(fieldOrigin);
});

test("a mismatch falls back to the binding origin", () => {
  const bindingOrigin = new OriginRef("binding:qty");

  const error = bindingError(() =>
    bindParams(
      [new Parameter("qty")],
      new Map([["qty", new Binding("${a.output}", { origin: bindingOrigin, contract: new ValueContract(INTEGER) })]]),
      poolOf({ a: "many" }),
    ),
  );

  expect(error.failureCode).toBe("binding.type");
  expect(error.origin).toEqual(bindingOrigin);
});

test("null is rejected for int and accepted for Optional(int)", () => {
  const error = bindingError(() => bind("${a.output}", new ValueContract(INTEGER), poolOf({ a: null })));
  expect(error.failureCode).toBe("binding.type");

  expect(bind("${a.output}", new ValueContract(OPTIONAL_INTEGER), poolOf({ a: null }))).toEqual({ qty: null });
});

test("a string is rejected before parameter coercion", () => {
  const param = new Parameter("qty", { contract: new ValueContract(INTEGER), coerceStrings: true });

  const error = bindingError(() => bind("${a.output}", new ValueContract(INTEGER), poolOf({ a: "30" }), { param }));

  expect(error.failureCode).toBe("binding.type");
});

test("the parameter contract still runs after a passing binding check", () => {
  const param = new Parameter("qty", { contract: new ValueContract(INTEGER) });

  const error = bindingError(() =>
    bind("${a.output}", new ValueContract(OPTIONAL_INTEGER), poolOf({ a: null }), { param }),
  );

  expect(error.failureCode).toBe("contract.input");
});

test("the MAP item scope is checked per item", () => {
  const contract = new ValueContract(INTEGER);

  expect(bind("${item}", contract, poolOf(), { item: 3n })).toEqual({ qty: 3n });
  const error = bindingError(() => bind("${item}", contract, poolOf(), { item: "x" }));
  expect(error.failureCode).toBe("binding.type");
});

test("a typed binding checks its evaluated value template", () => {
  const recordType = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["n", INTEGER],
      ["label", Type.scalar(ValueKind.STRING)],
    ]),
    required: new Set(["n", "label"]),
  });
  const template = new ValueTemplate({ n: "${a.output}", label: "n=${a.output}" });

  expect(bind(template, new ValueContract(recordType), poolOf({ a: 2n }))).toEqual({ qty: { n: 2n, label: "n=2" } });
  const error = bindingError(() => bind(template, new ValueContract(recordType), poolOf({ a: "two" })));
  expect(error.failureCode).toBe("binding.type");
});

test("a typed binding checks a non-string literal", () => {
  const error = bindingError(() => bind(30n, new ValueContract(Type.scalar(ValueKind.STRING)), poolOf()));
  expect(error.failureCode).toBe("binding.type");

  expect(bind(30n, new ValueContract(INTEGER), poolOf())).toEqual({ qty: 30n });
});

test("a binding type failure keeps the route without usable inputs", async () => {
  const origin = new OriginRef("contract:bad");
  const node = new CodeNode("node");
  node.params = [new Parameter("ok"), new Parameter("bad")];
  const wiring = {
    ok: { safe: [1n] },
    bad: new Binding("wrong", { contract: new ValueContract(INTEGER, new ContractOrigins({ default: origin })) }),
  };

  const evaluated = await evalNode(node, nodeFlow(node, wiring), poolOf(), { captureBindings: true });

  expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = evaluated.outcome as EvaluationFailure;
  expect(failure.code).toBe("binding.type");
  expect(failure.trace).toEqual([origin]);
  expect(evaluated.bindings!.boundInputs).toBeNull();
  expect(evaluated.bindings!.route.names).toEqual(["ok", "bad"]);
});

test("an untyped binding and raw sources are unchanged", () => {
  const pool = poolOf({ a: "many" });

  expect(bindParams([new Parameter("qty")], new Map([["qty", new Binding("${a.output}")]]), pool)).toEqual({
    qty: "many",
  });
  expect(bindParams([new Parameter("qty")], new Map([["qty", "${a.output}"]]), pool)).toEqual({ qty: "many" });
});
