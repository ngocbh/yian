/**
 * The bind grammar, resolving each declared param's source into a typed record. The local
 * `InputBinding`/`bindInputs` adapter phrases each case as a `Parameter` plus a wiring entry, so
 * the grammar (coalesce, `:-` default, `:?` required, escapes, type checks, deep copies, the
 * `${item}` scope) is tested through `bindParams`, the engine's read boundary; also the binding
 * capture and `BindingCollector`.
 */

import { describe, expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { TEMPLATE_FNS, registerTemplateFn } from "../../src/expr/builtins.ts";
import { RequiredError } from "../../src/expr/index.ts";
import { type RootRead, SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { Grow, Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import {
  applyDefaults,
  BindingCapture,
  BindingCollector,
  BindingError,
  bindParams,
  type ReservedBinder,
} from "../../src/nodes/binding.ts";
import { Assertion, Binding, ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { FileRef, Type, type Value, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { nodeFlow } from "../helpers/run_state_fixtures.ts";

const START_ID = "__start__";

/** One test-local parameter plus its flow-owned source; a missing or null `source` is an omitted input. */
interface InputBinding {
  name: string;
  contract?: ValueContract | null;
  source?: unknown;
  required?: boolean;
  hasDefault?: boolean;
  default?: unknown;
  coerceStrings?: boolean;
}

const STRING = new ValueContract(Type.scalar(ValueKind.STRING));
const INTEGER = new ValueContract(Type.scalar(ValueKind.INTEGER));
const NUMBER = new ValueContract(Type.scalar(ValueKind.NUMBER));
const BOOLEAN = new ValueContract(Type.scalar(ValueKind.BOOLEAN));
const OBJECT = new ValueContract(Type.scalar(ValueKind.OBJECT));
const NONE = new ValueContract(Type.scalar(ValueKind.NONE));
const NULLABLE_STRING = new ValueContract(new Type({ kind: ValueKind.STRING, nullable: true }));
const NULLABLE_INTEGER = new ValueContract(new Type({ kind: ValueKind.INTEGER, nullable: true }));

/** Near-canonical fixture proving the read seam requires the exact carrier. */
class ParameterSubclass extends Parameter {}

/**
 * Drive the live `bindParams` from test-local parameter/source pairs.
 *
 * A binding without a `source` models an omitted input (no wiring entry), so the parameter's
 * presence-based `default`/`required` applies. An entry that merely resolves to null (e.g.
 * `${system.w}` with the pool empty) is bound-null, which shadows a parameter default.
 */
function bindInputs(bindings: InputBinding[], pool: VariablePool, { item = null }: { item?: Value } = {}): ValueRecord {
  const params = bindings.map(
    (b) =>
      new Parameter(b.name, {
        contract: b.contract ?? null,
        required: b.required ?? false,
        hasDefault: b.hasDefault ?? false,
        default: b.default ?? null,
        coerceStrings: b.coerceStrings ?? false,
      }),
  );
  const wiring = new Map(
    bindings.filter((b) => b.source !== undefined && b.source !== null).map((b) => [b.name, b.source]),
  );
  return bindParams(params, wiring, pool, { item });
}

/** Return a pool holding each entry of `system` as a `${system.<key>}` ambient. */
function poolOf(system: Record<string, unknown> = {}): VariablePool {
  const pool = new VariablePool();
  for (const [key, value] of Object.entries(system)) pool.addSystem(key, value);
  return pool;
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

/** Return the error `call` throws. */
function errorOf(call: () => unknown): Error {
  try {
    call();
  } catch (error) {
    return error as Error;
  }
  throw new Error("call did not throw");
}

test("binding capture matches the value actually delivered", () => {
  const original: ValueRecord = { items: ["before"] };
  const reads: string[] = [];
  const calls: Value[] = [];

  const resolve = (names: readonly string[]): RootRead => {
    const path = names.join(".");
    reads.push(path);
    const value = path === "count" ? 12n : original;
    return { used: 1, present: true, value, why: null, source: new SourceRef("run", 1, [path]) };
  };

  const previous = TEMPLATE_FNS.get("observe");
  registerTemplateFn("observe", {
    params: [{ name: "value" }],
    call: (value) => {
      calls.push(value);
      return value;
    },
  });
  try {
    const collector = new BindingCollector();
    const delivered = bindParams(
      [new Parameter("count", { contract: NUMBER }), new Parameter("payload")],
      new Map([
        ["count", "${count}"],
        ["payload", "${observe(payload)}"],
      ]),
      poolOf(),
      { collector, sourceResolver: resolve },
    );
    const captured = collector.freeze();
    expect(captured.boundInputs).toEqual(delivered);
    expect(delivered).toEqual({ count: 12n, payload: { items: ["before"] } });
    expect(typeof delivered["count"]).toBe("bigint");
    expect(reads).toEqual(["count", "payload"]);
    expect(calls).toEqual([original]);
    ((delivered["payload"] as ValueRecord)["items"] as Value[]).push("node mutation");
    (original["items"] as Value[]).push("pool mutation");
    expect(captured.boundInputs!["payload"]).toEqual({ items: ["before"] });
  } finally {
    if (previous === undefined) TEMPLATE_FNS.delete("observe");
    else TEMPLATE_FNS.set("observe", previous);
  }
});

test("an omitted default has no upstream source", () => {
  const collector = new BindingCollector();
  const delivered = bindParams(
    [
      new Parameter("defaulted", { contract: NUMBER, hasDefault: true, default: 30n }),
      new Parameter("null", { hasDefault: true, default: "unused" }),
      new Parameter("omitted"),
    ],
    new Map([["null", "${present}"]]),
    poolOf(),
    {
      collector,
      sourceResolver: () => ({ used: 1, present: true, value: null, why: null, source: new SourceRef("run", 3) }),
    },
  );
  const captured = collector.freeze();
  expect(captured.boundInputs).toEqual(delivered);
  expect(delivered).toEqual({ defaulted: 30n, null: null });
  expect(typeof captured.boundInputs!["defaulted"]).toBe("bigint");
  expect(captured.route.names).toEqual(["defaulted", "null"]);
  expect(captured.evaluatedKeys).toEqual(new Set(["null"]));
  const [defaulted, read] = captured.route.children as [ValueRoute, ValueRoute];
  expect(defaulted.kind).toBe("literal");
  expect(read.source).toEqual(new SourceRef("run", 3));
});

/** A minimal concrete node; it fails before its body in these tests. */
class CodeNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
}

/** A node reserving `at`, whose normalizer refuses anything but a date string as `contract.input`. */
class ReservedAtNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;

  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(["at"]);
  }

  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    _pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    const source = nodeWiring.get("at");
    const normalize = (value: unknown) => {
      if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
      throw new BindingError("`at` is not a date", {
        failureCode: "contract.input",
        origin: source instanceof Binding ? source.origin : null,
      });
    };
    return { at: options.bind!("at", source, normalize) };
  }
}

describe("binding failure keeps partial context without usable inputs", () => {
  const origin = new OriginRef("bad-input");
  const rows: [string, () => [Node, Record<string, unknown>, string[]]][] = [
    [
      "ordinary",
      () => {
        const node = new CodeNode("node");
        node.params = [new Parameter("ok"), new Parameter("bad", { contract: INTEGER, origin })];
        return [node, { ok: { safe: [1n] }, bad: "wrong" }, ["ok", "bad"]];
      },
    ],
    [
      "reserved",
      () => {
        const node = new ReservedAtNode("node");
        node.params = [new Parameter("ok")];
        return [node, { ok: { safe: [1n] }, at: new Binding("wrong", { origin }) }, ["ok", "at"]];
      },
    ],
    [
      "deferred",
      () => {
        const node = new MapNode("node", { flowId: "child", child: {} });
        node.params = [new Parameter("ok"), new Parameter("bad", { contract: INTEGER, origin })];
        return [node, { over: [1n, "wrong"], ok: { safe: [1n] }, bad: "${item}" }, ["over"]];
      },
    ],
    [
      "wrapper_default",
      () => {
        // The default fails before the child flow is read.
        const node = new MapNode("node", { flowId: "child", child: {} });
        node.params = [
          new Parameter("bad", { contract: INTEGER, hasDefault: true, default: "wrong", defaultOrigin: origin }),
        ];
        return [node, { over: [] }, ["over"]];
      },
    ],
  ];
  for (const [stage, build] of rows) {
    test(stage, async () => {
      const [node, wiring, expectedNames] = build();
      const evaluated = await evalNode(node, nodeFlow(node, wiring), poolOf(), { captureBindings: true });
      expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
      const { failure } = evaluated.outcome as EvaluationFailure;
      expect(failure.code).toBe(stage === "wrapper_default" ? "binding.evaluate" : "contract.input");
      expect(failure.trace).toEqual([origin]);
      expect(evaluated.bindings!.boundInputs).toBeNull();
      expect(evaluated.bindings!.route.names).toEqual(expectedNames);
      if (stage === "deferred") {
        expect(evaluated.bindings!.items.map(([index]) => index)).toEqual([0, 1]);
        for (const [, item] of evaluated.bindings!.items) {
          expect(item.boundInputs).toBeNull();
          expect(item.route.names).toEqual(["ok", "bad"]);
        }
      }
    });
  }
});

test("evaluator capture preserves typed values before node mutation", async () => {
  class MutatingNode extends Node {
    readonly kind = NodeKind.CODE;

    override forward(inputs: Record<string, unknown>): NodeResult {
      ((inputs["payload"] as ValueRecord[])[0]!["items"] as Value[]).push("after");
      return new Output(inputs);
    }
  }

  const file = new FileRef({ uri: "file:///input.txt", name: "input.txt" });
  const node = new MutatingNode("mutate");
  node.params = [new Parameter("payload")];
  const flow = nodeFlow(node, { payload: [{ items: [file] }] });
  const evaluated = await evalNode(node, flow, poolOf(), { captureBindings: true });
  expect(evaluated.bindings!.boundInputs).toEqual({ payload: [{ items: [file] }] });
  const captured = evaluated.bindings!.boundInputs!["payload"] as ValueRecord[];
  expect((captured[0]!["items"] as Value[])[0]).toBe(file);
  const delivered = (evaluated.outcome as Output).value as ValueRecord;
  expect((delivered["payload"] as ValueRecord[])[0]!["items"]).toEqual([file, "after"]);
  expect((await evalNode(node, flow, poolOf())).bindings).toBeNull();
});

test("capture includes materialized reserved values from node hooks", async () => {
  class ReservedNode extends Node {
    readonly kind = NodeKind.CODE;

    override bindReserved(): Record<string, unknown> {
      return { payload: { items: [1n] } };
    }

    override forward(inputs: Record<string, unknown>): NodeResult {
      ((inputs["payload"] as ValueRecord)["items"] as Value[]).push(2n);
      return new Output(inputs);
    }
  }

  const evaluated = await evalNode(new ReservedNode("node"), null, poolOf(), { captureBindings: true });
  expect(evaluated.bindings!.boundInputs).toEqual({ payload: { items: [1n] } });
  expect(evaluated.bindings!.route.names).toEqual(["payload"]);
  expect(evaluated.bindings!.route.children[0]!.kind).toBe("literal");
});

describe("per-item wrapper capture includes defaults before body and asserts", () => {
  for (const mode of ["return", "mutate", "preassert"]) {
    test(mode, async () => {
      class PerItemNode extends Node {
        readonly kind = NodeKind.CODE;
        override readonly bindsPerItem = true;

        override forward(inputs: Record<string, unknown>): NodeResult {
          if (mode === "mutate") ((inputs["fallback"] as ValueRecord)["items"] as Value[]).push(2n);
          return new Output(inputs);
        }
      }

      const node = new PerItemNode("node");
      node.params = [new Parameter("fallback", { hasDefault: true, default: { items: [1n] } })];
      if (mode === "preassert") node.preAsserts = [new Assertion("false", "must hold")];
      const evaluated = await evalNode(node, null, poolOf(), { captureBindings: true });
      expect(evaluated.bindings!.boundInputs).toEqual({ fallback: { items: [1n] } });
      expect(evaluated.bindings!.route.names).toEqual(["fallback"]);
      expect(evaluated.bindings!.route.children[0]!.kind).toBe("literal");
      expect(evaluated.bindings!.evaluatedKeys).toEqual(new Set());
      if (mode === "preassert") {
        expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
        expect((evaluated.outcome as EvaluationFailure).failure.code).toBe("assertion.failed");
      } else {
        const expected = mode === "mutate" ? [1n, 2n] : [1n];
        expect((evaluated.outcome as Output).value).toEqual({ fallback: { items: expected } });
      }
    });
  }
});

test("map wrapper capture includes valid defaults and item routes", async () => {
  const node = new MapNode("map", { flowId: "child", child: {} });
  node.params = [new Parameter("fallback", { contract: NUMBER, hasDefault: true, default: 30n })];
  const evaluated = await evalNode(node, null, poolOf(), { wiringOverride: { over: [1n] }, captureBindings: true });
  expect(evaluated.bindings!.boundInputs).toEqual({ over: [1n], fallback: 30n });
  expect(typeof evaluated.bindings!.boundInputs!["fallback"]).toBe("bigint");
  expect(evaluated.bindings!.route.names).toEqual(["over", "fallback"]);
  expect(evaluated.bindings!.route.children.at(-1)!.kind).toBe("literal");
  expect((evaluated.outcome as Grow).seed).toEqual([{ fallback: 30n }]);
  const [index, item] = evaluated.bindings!.items[0]!;
  expect(index).toBe(0);
  expect(item.boundInputs).toEqual({ fallback: 30n });
  expect(item.route.names).toEqual(["fallback"]);
  expect(item.route.children[0]!.kind).toBe("literal");
});

test("bind a reference and a literal", () => {
  const pool = poolOf({ topics: "ACME" });
  pool.set("up", { pe: 21 });
  const rec = bindInputs(
    [
      { name: "topic", contract: STRING, source: "${system.topics}" },
      { name: "pe", contract: NUMBER, source: "${up.output.pe}" },
      { name: "k", contract: NUMBER, source: 90n }, // literal
    ],
    pool,
  );
  expect(rec).toEqual({ topic: "ACME", pe: 21, k: 90n });
});

test("the default is applied when the input is omitted", () => {
  // The param `default` is the omit-default (presence-based): it fills only when the input has
  // no wiring entry at all. (Resolve-fallback for a present-but-null source is the `:-` grammar.)
  const rec = bindInputs([{ name: "w", contract: INTEGER, hasDefault: true, default: 30n }], poolOf());
  expect(rec).toEqual({ w: 30n });
});

test("a compatible default is validated without replacement", () => {
  // A typed default keeps its authored payload after validation.
  const rec = bindInputs([{ name: "w", contract: NUMBER, hasDefault: true, default: 30n }], poolOf());
  expect(typeof rec["w"]).toBe("bigint");
});

test("a present null shadows the default but fails a non-nullable contract", () => {
  // An entry that is present but resolves to null is bound-null: it shadows the param default
  // rather than filling 30, but the non-nullable type still rejects the explicit null.
  const error = bindingError(() =>
    bindInputs(
      [{ name: "w", contract: INTEGER, source: "${system.w}", hasDefault: true, default: 30n }],
      poolOf(),
    ),
  );
  expect(error.failureCode).toBe("contract.input");
});

test("a present null shadows the default when the contract is nullable", () => {
  const rec = bindInputs(
    [{ name: "w", contract: NULLABLE_INTEGER, source: "${system.w}", hasDefault: true, default: 30n }],
    poolOf(),
  );
  expect(rec).toEqual({ w: null });
});

test("a present null is valid for a none contract", () => {
  expect(bindInputs([{ name: "w", contract: NONE, source: "${system.w}" }], poolOf())).toEqual({ w: null });
});

test("an explicit null default fills when omitted", () => {
  // `hasDefault` owns presence, so an authored nullable `default: null` is a real default and
  // stays distinct from a parameter with no default.
  const rec = bindInputs([{ name: "w", contract: NULLABLE_INTEGER, hasDefault: true, default: null }], poolOf());
  expect(rec).toEqual({ w: null });
});

test("an omitted input with no default is absent", () => {
  // A truly omitted optional input with no default leaves its key absent from the record (not
  // present as null) so the wrapper's default can fill it.
  expect(bindInputs([{ name: "x", contract: INTEGER }], poolOf())).toEqual({});
});

test("an omitted required input raises", () => {
  // Param `required` is presence-based too: an omitted required input (no entry) fails loud.
  const error = bindingError(() => bindInputs([{ name: "t", contract: STRING, required: true }], poolOf()));
  expect(error.failureCode).toBe("binding.evaluate");
});

test("binding errors carry the stage code and the operation origin", () => {
  // Expression/required/default work and supplied contracts stay distinct.
  const bindingOrigin = new OriginRef("binding:x");
  const parameterOrigin = new OriginRef("parameter:x");
  const defaultOrigin = new OriginRef("default:x");
  const contractOrigin = new OriginRef("contract:x");

  const expression = bindingError(() =>
    bindParams(
      [new Parameter("x", { origin: parameterOrigin })],
      new Map([["x", new Binding("${missing.value:?x is required}", { origin: bindingOrigin })]]),
      poolOf(),
    ),
  );
  expect([expression.failureCode, expression.origin]).toEqual(["binding.evaluate", bindingOrigin]);

  const required = bindingError(() =>
    bindParams([new Parameter("x", { required: true, origin: parameterOrigin })], new Map(), poolOf()),
  );
  expect([required.failureCode, required.origin]).toEqual(["binding.evaluate", parameterOrigin]);
  expect(required.message).toBe("required input 'x' is unbound");

  const integerContract = new ValueContract(
    Type.scalar(ValueKind.INTEGER),
    new ContractOrigins({ default: contractOrigin }),
  );
  const invalidDefault = bindingError(() =>
    bindParams(
      [
        new Parameter("x", {
          contract: integerContract,
          hasDefault: true,
          default: "bad",
          origin: parameterOrigin,
          defaultOrigin,
        }),
      ],
      new Map(),
      poolOf(),
    ),
  );
  expect([invalidDefault.failureCode, invalidDefault.origin]).toEqual(["binding.evaluate", defaultOrigin]);

  const contract = bindingError(() =>
    bindParams(
      [new Parameter("x", { contract: integerContract, origin: parameterOrigin })],
      new Map([["x", new Binding("bad", { origin: bindingOrigin })]]),
      poolOf(),
    ),
  );
  expect([contract.failureCode, contract.origin]).toEqual(["contract.input", contractOrigin]);
});

test("a nested input contract failure selects the exact, then the ancestor origin", () => {
  // Supplied values resolve their structural error path through the contract.
  const exactOrigin = new OriginRef("contract:value/items/0/score");
  const itemsOrigin = new OriginRef("contract:value/items");
  const rootOrigin = new OriginRef("contract:value");
  const score = Type.scalar(ValueKind.INTEGER);
  const item = new Type({ kind: ValueKind.OBJECT, fields: new Map([["score", score]]), required: new Set(["score"]) });
  const valueType = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["items", new Type({ kind: ValueKind.LIST_OBJECT, element: item })]]),
    required: new Set(["items"]),
  });
  const contract = new ValueContract(
    valueType,
    new ContractOrigins({
      default: rootOrigin,
      entries: [
        [["items"], itemsOrigin],
        [["items", 0, "score"], exactOrigin],
      ],
    }),
  );
  const parameter = new Parameter("value", { contract });

  const exact = bindingError(() =>
    bindParams([parameter], new Map([["value", new Binding({ items: [{ score: "bad" }] })]]), poolOf()),
  );
  expect(exact.origin).toEqual(exactOrigin);

  const ancestor = bindingError(() =>
    bindParams(
      [parameter],
      new Map([["value", new Binding({ items: [{ score: 1n }, { score: "bad" }] })]]),
      poolOf(),
    ),
  );
  expect(ancestor.origin).toEqual(itemsOrigin);
});

test("a nested input contract fallback keeps the default's precedence", () => {
  // Missing contract sites fall back, while authored defaults keep ownership.
  const parameterOrigin = new OriginRef("parameter:value");
  const defaultOrigin = new OriginRef("default:value");
  const item = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["score", Type.scalar(ValueKind.INTEGER)]]),
    required: new Set(["score"]),
  });
  const sourceFree = new ValueContract(new Type({ kind: ValueKind.LIST_OBJECT, element: item }));

  const supplied = bindingError(() =>
    bindParams(
      [new Parameter("value", { contract: sourceFree, origin: parameterOrigin })],
      new Map([["value", new Binding([{ score: "bad" }])]]),
      poolOf(),
    ),
  );
  expect(supplied.failureCode).toBe("contract.input");
  expect(supplied.origin).toEqual(parameterOrigin);
  expect(supplied.message).toBe('input \'value\' at [0].score: "bad" does not match declared type int');

  const defaulted = bindingError(() =>
    bindParams(
      [
        new Parameter("value", {
          contract: sourceFree,
          hasDefault: true,
          default: [{ score: "bad" }],
          origin: parameterOrigin,
          defaultOrigin,
        }),
      ],
      new Map(),
      poolOf(),
    ),
  );
  expect(defaulted.failureCode).toBe("binding.evaluate");
  expect(defaulted.origin).toEqual(defaultOrigin);
  // A failing default names the place inside it the same way.
  expect(defaulted.message).toBe('input \'value\' at [0].score: "bad" does not match declared type int');
});

test("applyDefaults marks an invalid authored default as binding evaluation", () => {
  // The wrapper's default backstop keeps default-site semantics.
  const defaultOrigin = new OriginRef("default:x");
  const parameter = new Parameter("x", { contract: INTEGER, hasDefault: true, default: "bad", defaultOrigin });

  const error = bindingError(() => applyDefaults([parameter], {}));

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.origin).toEqual(defaultOrigin);
});

test("bindParams rejects a raw declaration bypass", () => {
  // The engine read seam enforces the canonical node params shape.
  const error = errorOf(() => bindParams(["x"] as never, new Map(), poolOf()));
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toContain("Parameter values");
});

test("bindParams rejects Parameter subclasses", () => {
  const error = errorOf(() => bindParams([new ParameterSubclass("x")], new Map(), poolOf()));
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toContain("Parameter values");
});

test("a nullable reference that is missing binds null", () => {
  expect(bindInputs([{ name: "t", contract: NULLABLE_STRING, source: "${system.t}" }], poolOf())).toEqual({
    t: null,
  });
});

test("a type mismatch raises", () => {
  const error = bindingError(() =>
    bindInputs([{ name: "n", contract: NUMBER, source: "${system.n}" }], poolOf({ n: "not-a-number" })),
  );
  expect(error.failureCode).toBe("contract.input");
});

test("binding never converts a supplied string", () => {
  // Only the run boundary (`coerceInputs`) converts text; a child START bound across a
  // CALL/MAP/LOOP edge validates the value as is, whatever its policy says.
  const pool = poolOf({ n: "30" });
  for (const coerceStrings of [false, true]) {
    const error = bindingError(() =>
      bindInputs([{ name: "n", contract: INTEGER, source: "${system.n}", coerceStrings }], pool),
    );
    expect(error.message).toContain('"30" does not match declared type int');
  }
});

test("an incompatible stored default is not coerced", () => {
  // String coercion is caller-input policy. It must not silently rewrite the already-stored
  // default, even when that policy is enabled.
  const error = bindingError(() =>
    bindInputs([{ name: "n", contract: INTEGER, hasDefault: true, default: "30", coerceStrings: true }], poolOf()),
  );
  expect(error.failureCode).toBe("binding.evaluate");
});

test("an untyped value is lenient", () => {
  // No contract means the read boundary accepts any value.
  expect(bindInputs([{ name: "p", source: "${system.p}" }], poolOf({ p: { k: 1n } }))).toEqual({ p: { k: 1n } });
});

test("a falsy default is applied", () => {
  // A falsy param default (0 / false) still fills when the input is omitted (no entry).
  const rec = bindInputs(
    [
      { name: "n", contract: INTEGER, hasDefault: true, default: 0n },
      { name: "f", contract: BOOLEAN, hasDefault: true, default: false },
    ],
    poolOf(),
  );
  expect(rec).toEqual({ n: 0n, f: false });
});

test("binding isolates an object from the pool", () => {
  // The leaf must see only its own record: mutating it must not write the pool.
  const pool = poolOf();
  pool.set("up", { k: 1n, nested: { a: 2n } });
  const rec = bindInputs([{ name: "x", contract: OBJECT, source: "${up.output}" }], pool);
  const x = rec["x"] as ValueRecord;
  x["k"] = 999n;
  (x["nested"] as ValueRecord)["a"] = 999n; // nested mutation must not leak either
  expect(pool.get("up")).toEqual({ k: 1n, nested: { a: 2n } });
});

test("binding isolates an untyped value from the pool", () => {
  // The untyped path must also isolate the value.
  const pool = poolOf({ p: { k: 1n } });
  const rec = bindInputs([{ name: "p", source: "${system.p}" }], pool);
  (rec["p"] as ValueRecord)["k"] = 999n;
  expect(pool.system.get("p")!.toObject()).toEqual({ k: 1n });
});

test("an untyped binding resolves and isolates", () => {
  // No contract: resolve and deep-copy, with no type check.
  const pool = poolOf({ topic: "ACME" });
  pool.set("up", { k: 1n });
  const rec = bindInputs(
    [
      { name: "topic", source: "${system.topic}" },
      { name: "cfg", source: "${up.output}" },
      { name: "lit", source: 5n },
    ],
    pool,
  );
  expect(rec).toEqual({ topic: "ACME", cfg: { k: 1n }, lit: 5n });
  (rec["cfg"] as ValueRecord)["k"] = 999n; // mutation must not leak (deep-copied)
  expect(pool.get("up")).toEqual({ k: 1n });
});

// --- coalesce, `:-` default and `:?` required --- //

test("coalesce takes the first non-null", () => {
  // `${a | b}` branch-join: the first non-null among peers (refs).
  const pool = new VariablePool();
  pool.set("pro", "pro-text");
  // a present: a wins
  expect(bindInputs([{ name: "t", source: "${pro.output | con.output}" }], pool)).toEqual({ t: "pro-text" });
  // a absent: falls through to b
  const pool2 = new VariablePool();
  pool2.set("con", "con-text");
  expect(bindInputs([{ name: "t", source: "${pro.output | con.output}" }], pool2)).toEqual({ t: "con-text" });
});

test("an n-ary coalesce and a literal last segment", () => {
  const pool = new VariablePool();
  pool.set("c", "c-text");
  // 3-way, first two absent: the third ref wins
  expect(bindInputs([{ name: "t", source: "${a.output | b.output | c.output}" }], pool)).toEqual({ t: "c-text" });
  // all refs absent, a quoted or number literal last segment survives
  const empty = new VariablePool();
  expect(bindInputs([{ name: "t", source: '${a.output | "fallback"}' }], empty)).toEqual({ t: "fallback" });
  expect(bindInputs([{ name: "n", source: "${a.output | 0.0}" }], empty)).toEqual({ n: 0 });
});

test("the default operator's quoted right side is a literal", () => {
  // The `:-` right side is a full expression: a quoted string is a literal, a number or null is
  // its literal, and a bare word is a reference, so quote a literal default.
  const empty = new VariablePool();
  expect(bindInputs([{ name: "d", source: '${input.as_of:-"today"}' }], empty)).toEqual({ d: "today" });
  expect(bindInputs([{ name: "n", source: "${input.n:-30}" }], empty)).toEqual({ n: 30n });
  expect(bindInputs([{ name: "z", source: "${input.z:-null}" }], empty)).toEqual({ z: null });
  // a bare word right side is a ref: unbound gives null (a miss), no literal string fallback.
  expect(bindInputs([{ name: "d", source: "${input.as_of:-today}" }], empty)).toEqual({ d: null });
  // a present value wins over the default
  const pool = new VariablePool();
  pool.set(START_ID, { as_of: "2026-06-12" });
  expect(bindInputs([{ name: "d", source: '${input.as_of:-"today"}' }], pool)).toEqual({ d: "2026-06-12" });
});

test("the required operator fails loud when unbound", () => {
  const empty = new VariablePool();
  const error = bindingError(() => bindInputs([{ name: "t", source: '${input.topic:?"a topic is required"}' }], empty));
  expect((error.cause as Error).constructor).toBe(RequiredError);
  expect((error.cause as Error).message).toBe("a topic is required");
  // present: returns the value, no raise
  const pool = new VariablePool();
  pool.set(START_ID, { topic: "ACME" });
  expect(bindInputs([{ name: "t", source: "${input.topic:?required}" }], pool)).toEqual({ t: "ACME" });
});

test("a plain ref and a literal are unchanged by the coalesce parser", () => {
  // A plain single ref and a non-`${}` literal resolve as before.
  const pool = new VariablePool();
  pool.set("up", { pe: 21 });
  expect(bindInputs([{ name: "pe", source: "${up.output.pe}" }], pool)).toEqual({ pe: 21 });
  expect(bindInputs([{ name: "k", source: 90n }], pool)).toEqual({ k: 90n });
});

test("a falsy but present value wins a coalesce", () => {
  // Only null is "no value". A present 0 / false / "" / [] must win a coalesce, not fall through.
  for (const falsy of [0n, false, "", []]) {
    const pool = new VariablePool();
    pool.set("a", falsy);
    pool.set("b", "b-text");
    const rec = bindInputs([{ name: "t", source: "${a.output | b.output}" }], pool);
    expect(rec["t"]).toEqual(falsy);
    expect(rec["t"]).not.toBe("b-text");
  }
});

test("required does not raise on a present falsy value", () => {
  // `:?` fires only on null: a present false / 0 / "" is a value, returned, no raise.
  for (const falsy of [false, 0n, ""]) {
    const pool = new VariablePool();
    pool.set(START_ID, { flag: falsy });
    expect(bindInputs([{ name: "f", source: "${input.flag:?req}" }], pool)).toEqual({ f: falsy });
  }
});

test("the default is suppressed by a present falsy value", () => {
  // `:-` present-value-wins must hold for a falsy present value (0 must not become 99).
  const pool = new VariablePool();
  pool.set(START_ID, { n: 0n });
  expect(bindInputs([{ name: "n", source: "${input.n:-99}" }], pool)).toEqual({ n: 0n });
});

test("a quote-aware literal keeps its pipe", () => {
  // A quoted literal operand holding `|` is not split mid-token.
  const empty = new VariablePool();
  expect(bindInputs([{ name: "t", source: '${a.output | "x|y"}' }], empty)).toEqual({ t: "x|y" });
  // and a default value holding `:` inside quotes survives (the first operator is the `:-`)
  expect(bindInputs([{ name: "d", source: '${input.x:-"a:b"}' }], empty)).toEqual({ d: "a:b" });
});

test("a default ref resolves and a nested span raises", () => {
  // `${input.as_of :- system.today}` falls back to the system date; a `${...}` written inside
  // the expression is a loud error.
  const pool = new VariablePool();
  pool.addSystem("today", "2026-06-12");
  // as_of unbound: falls through to system.today
  expect(bindInputs([{ name: "d", source: "${input.as_of :- system.today}" }], pool)).toEqual({ d: "2026-06-12" });
  // as_of present: wins; the default is not consulted
  pool.set(START_ID, { as_of: "2026-01-01" });
  expect(bindInputs([{ name: "d", source: "${input.as_of :- system.today}" }], pool)).toEqual({ d: "2026-01-01" });
  const error = bindingError(() => bindInputs([{ name: "d", source: "${input.as_of:-${system.today}}" }], pool));
  expect(error.message).toContain("write `${input.as_of:-system.today}`");
});

test("the item scope combines with coalesce and default", () => {
  // `${item?.x | b.output}`: an item without `x` falls to the ref peer; a present item wins.
  const pool = new VariablePool();
  pool.set("b", "b-text");
  expect(bindInputs([{ name: "v", source: "${item?.x | b.output}" }], pool, { item: { y: 1n } })).toEqual({
    v: "b-text",
  });
  expect(bindInputs([{ name: "v", source: "${item?.x | b.output}" }], pool, { item: { x: 7n } })).toEqual({ v: 7n });
  // `.` on a key the untyped item lacks raises; the coalesce never sees it.
  expect(() => bindInputs([{ name: "v", source: "${item.x | b.output}" }], pool, { item: { y: 1n } })).toThrow(
    "cannot read ${item.x}: item has no field 'x'",
  );
});

test("a whole ref is typed, an embedded one stringified, and $$ escapes", () => {
  // A value that is exactly one `${...}` gives the typed value; a `${...}` embedded in text is
  // stringified; `$$` is a literal `$`.
  const pool = new VariablePool();
  pool.set("a", 0.7);
  pool.set("b", ["ACME", "BETA"]);
  // a whole `${...}` gives the float, not the string "0.7"
  const rec = bindInputs([{ name: "x", source: "${a.output}" }], pool);
  expect(rec).toEqual({ x: 0.7 });
  expect(typeof rec["x"]).toBe("number");
  // a whole `${...}` of a list gives the list
  expect(bindInputs([{ name: "x", source: "${b.output}" }], pool)).toEqual({ x: ["ACME", "BETA"] });
  // an embedded `${...}` in text gives a string with the value interpolated
  expect(bindInputs([{ name: "x", source: "pe=${a.output}" }], pool)).toEqual({ x: "pe=0.7" });
  // `$$` gives a literal `$` (no interpolation)
  expect(bindInputs([{ name: "x", source: "cost is $$5" }], pool)).toEqual({ x: "cost is $5" });
});

test("bindInputs gives a typed record from the declared inputs", () => {
  // The read boundary the engine seam (evalNode) uses to turn a node's declared inputs into a
  // typed record.
  expect(bindInputs([{ name: "t", contract: STRING, source: "${system.t}" }], poolOf({ t: "X" }))).toEqual({
    t: "X",
  });
});

test("bindInputs resolves item from the local scope", () => {
  const rec = bindInputs([{ name: "topic", contract: STRING, source: "${item}" }], new VariablePool(), {
    item: "ACME",
  });
  expect(rec).toEqual({ topic: "ACME" });
});

test("bindInputs walks a dotted item path", () => {
  const rec = bindInputs([{ name: "v", source: "${item.value}" }], new VariablePool(), { item: { value: 7n } });
  expect(rec).toEqual({ v: 7n });
});

test("bindInputs reads an unset item outside a MAP as null", () => {
  // `${item}` with no item resolves to null (the compile-time scan forbids `${item}` outside a
  // MAP body, so this only guards a defensive path).
  expect(bindInputs([{ name: "x", source: "${item}" }], new VariablePool())).toEqual({ x: null });
});

test("bindInputs still reads a non-item ref from the pool with an item present", () => {
  const pool = new VariablePool();
  pool.set(START_ID, { as_of: "2026-01-01" });
  expect(bindInputs([{ name: "d", source: "${input.as_of}" }], pool, { item: "X" })).toEqual({ d: "2026-01-01" });
});

test("bindParams refuses a Set as params", () => {
  const error = errorOf(() => bindParams(new Set() as never, new Map(), poolOf()));
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toBe("node params must be an exact list, got Set");
});

test("a bound -0 is written as 0", () => {
  const rec = bindInputs(
    [
      { name: "literal", source: -0 },
      { name: "evaluated", source: "${-0.0}" },
    ],
    poolOf(),
  );
  expect(Object.is(rec["literal"], 0)).toBe(true);
  expect(Object.is(rec["evaluated"], 0)).toBe(true);
});

test("a container shared inside a bound value stays shared in the copy", () => {
  const shared = [1n];
  const rec = bindInputs([{ name: "x", source: [shared, shared] }], poolOf());
  const [first, second] = rec["x"] as Value[];
  expect(first).toBe(second!);
  expect(first).not.toBe(shared);
  expect(first).toEqual(shared);
});

test("a FileRef inside a bound record is kept by identity", () => {
  const ref = new FileRef({ uri: "f" });
  const rec = bindInputs([{ name: "x", source: { f: ref } }], poolOf());
  expect((rec["x"] as ValueRecord)["f"]).toBe(ref);
});

test("an own __proto__ key in a bound record stays data", () => {
  const record = {};
  Object.defineProperty(record, "__proto__", { value: 1n, writable: true, enumerable: true, configurable: true });
  const rec = bindInputs([{ name: "x", source: record }], poolOf());
  const x = rec["x"] as ValueRecord;
  expect(x).not.toBe(record);
  expect(Object.hasOwn(x, "__proto__")).toBe(true);
  expect(Object.getPrototypeOf(x)).toBe(Object.prototype);
  expect(Object.getOwnPropertyDescriptor(x, "__proto__")?.value).toBe(1n);
});

test("forItem refuses a negative, non-integer or repeated index", () => {
  const collector = new BindingCollector();
  for (const index of [-1, 1.5, Number.NaN]) {
    const error = errorOf(() => collector.forItem(index));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe("binding item index must be a nonnegative integer");
  }
  collector.forItem(3);
  const error = errorOf(() => collector.forItem(3));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("binding item index 3 was already recorded");
});

test("an invalidated capture keeps partial routes without usable inputs", () => {
  const collector = new BindingCollector();
  collector.forItem(0).finish({ i: 2n });
  // bindParams invalidates the collector first, then fails before it finishes.
  bindingError(() =>
    bindParams(
      [new Parameter("a"), new Parameter("b", { required: true })],
      new Map([["a", "${x}"]]),
      poolOf(),
      {
        collector,
        sourceResolver: () => ({ used: 1, present: true, value: 1n, why: null, source: new SourceRef("run", 1) }),
      },
    ),
  );
  const captured = collector.freeze();
  expect(captured.boundInputs).toBeNull();
  expect(captured.route.names).toEqual(["a"]);
  expect(captured.route.children[0]!.kind).toBe("source");
  expect(captured.evaluatedKeys).toEqual(new Set(["a"]));
  expect(captured.items.map(([index]) => index)).toEqual([0]);
  const [, item] = captured.items[0]!;
  expect(item.boundInputs).toBeNull();
  expect(item.route.names).toEqual(["i"]);
});

test("a collector freezes once", () => {
  const collector = new BindingCollector();
  collector.freeze();
  const error = errorOf(() => collector.freeze());
  expect(error.constructor).toBe(Error);
  expect(error.message).toBe("binding capture is already frozen");
});

test("boundInputs gives a fresh copy on each read", () => {
  const collector = new BindingCollector();
  bindParams([new Parameter("x")], new Map([["x", { items: [1n] }]]), poolOf(), { collector });
  const captured = collector.freeze();
  const first = captured.boundInputs!;
  expect(captured.boundInputs).not.toBe(first);
  ((first["x"] as ValueRecord)["items"] as Value[]).push(2n);
  expect(captured.boundInputs).toEqual({ x: { items: [1n] } });
  expect(Object.isFrozen(captured)).toBe(true);
});

test("evaluatedKeys gives a fresh copy on each read", () => {
  const collector = new BindingCollector();
  bindParams([new Parameter("a")], new Map([["a", "${system.x}"]]), poolOf({ x: 1n }), { collector });
  const captured = collector.freeze();
  (captured.evaluatedKeys as Set<string>).add("zzz");
  expect(captured.evaluatedKeys).toEqual(new Set(["a"]));
});

test("a capture built directly freezes its item pairs and copies its evaluated keys", () => {
  const inner = new BindingCapture({ i: 1n }, ValueRoute.record([]), new Set());
  const keys = new Set(["a"]);
  const capture = new BindingCapture(null, ValueRoute.record([]), keys, [[0, inner]]);
  expect(Object.isFrozen(capture.items)).toBe(true);
  expect(Object.isFrozen(capture.items[0])).toBe(true);
  keys.add("b");
  expect(capture.evaluatedKeys).toEqual(new Set(["a"]));
});
