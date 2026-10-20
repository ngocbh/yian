/**
 * Raw node evaluation for the append-only dispatcher: a port of
 * `tests/engine/test_eval_node_raw.py`.
 *
 * Ported: `test_raw_evaluation_returns_closed_success_route_and_grow_values`.
 *
 * Ported (adapted): `test_public_evaluation_always_returns_frozen_slotted_wrapper` and
 * `test_raw_failure_is_frozen_slotted_and_preserves_diagnostics`: the field names are the own
 * keys, frozen means `Object.isFrozen` and a throwing write, and the traceback is the binder
 * error's stack.
 *
 * Not ported: the `__slots__` (no `__dict__`) checks; a TS class has no slots.
 *
 * TS-only: the `EvaluatedOutcome` constructor checks, `requireSteering`, and the steering
 * capability's checks on what the host inbox returns.
 *
 * Changed: `raw` evaluates the node in a one-node `CompiledFlow` wired from a `FuncNode`'s
 * `reads`, as Python's stub flow derives it.
 *
 * Later (T4-6): the other rows.
 */

import { describe, expect, test } from "bun:test";

import { CompiledFlow } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import {
  CapabilityFault,
  Grow,
  Node,
  NodeKind,
  type NodeResult,
  Output,
  Route,
  type Steering,
} from "../../src/nodes/base.ts";
import { BindingCollector } from "../../src/nodes/binding.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { BackwardCapture } from "../../src/runtime/contracts.ts";
import { EvaluationEvidence } from "../../src/memory/contracts.ts";
import {
  EvaluatedOutcome,
  EvaluationFailure,
  type EvalNodeOptions,
  evalNode,
  requireSteering,
  SteeringCapability,
  type WorkerOutcome,
} from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FuncNode } from "./run_state_fixtures.ts";

/** Evaluate `node` in a one-node flow wired from a `FuncNode`'s `reads`. */
function evaluate(node: Node, options: EvalNodeOptions = {}): Promise<EvaluatedOutcome> {
  const wiring = node instanceof FuncNode ? node.reads : {};
  const flow = CompiledFlow.fromParts(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map(Object.entries(wiring))]]),
  });
  return evalNode(node, flow, new VariablePool(), options);
}

/** Return the raw outcome of evaluating `node`. */
async function raw(node: Node, options: EvalNodeOptions = {}): Promise<WorkerOutcome> {
  return (await evaluate(node, options)).outcome;
}

/** Return one configured result from its body. */
class ResultNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  readonly result: unknown;

  constructor(nodeId: string, result: unknown) {
    super(nodeId);
    this.result = result;
  }

  override forward(): NodeResult {
    return this.result as NodeResult;
  }
}

/** A minimal spawner for the raw `Grow` result. */
class Spawner extends ResultNode {
  override readonly isSpawner = true;
}

/** Return the value `run` throws, or fail when it returns. */
function thrown(run: () => unknown): unknown {
  try {
    run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** Return the value `promise` rejects with, or fail when it resolves. */
async function rejected(promise: unknown): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Return a `node.execute` failure for node `work`. */
function failure(): Failure {
  return new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
}

test("public evaluation always returns frozen slotted wrapper", async () => {
  const evaluated = await evaluate(new FuncNode("work", () => "done"));

  expect(evaluated).toBeInstanceOf(EvaluatedOutcome);
  expect(evaluated).toEqual(new EvaluatedOutcome(new Output("done"), null));
  expect(Object.keys(evaluated)).toEqual(["outcome", "evidence", "bindings", "backwardCapture"]);
  expect(Object.isFrozen(evaluated)).toBe(true);
  expect(() => {
    (evaluated as { evidence: unknown }).evidence = null;
  }).toThrow(TypeError);
});

test("raw evaluation returns closed success route and grow values", async () => {
  expect(await raw(new FuncNode("work", () => "done"))).toEqual(new Output("done"));
  expect(await raw(new ResultNode("branch", new Route("chosen")))).toEqual(new Route("chosen"));
  expect(await raw(new Spawner("spawn", new Grow({ value: 1n })))).toEqual(new Grow({ value: 1n }));
});

test("raw failure is frozen slotted and preserves diagnostics", async () => {
  const node = new FuncNode("work", (inputs) => inputs);
  node.params = [new Parameter("required", { required: true })];

  const result = await raw(node);

  expect(result).toBeInstanceOf(EvaluationFailure);
  const retained = (result as EvaluationFailure).failure;
  expect(retained.nodeId).toBe("work");
  expect(retained.causeType).toBe("BindingError");
  expect(retained.traceback).toStartWith("BindingError: required input 'required' is unbound");
  expect(Object.keys(result)).toEqual(["failure"]);
  for (const alias of ["nodeId", "error", "errorType", "traceback"]) expect(alias in result).toBe(false);
  expect(Object.isFrozen(result)).toBe(true);
  expect(() => {
    (result as { failure: unknown }).failure = retained;
  }).toThrow(TypeError);
});

describe("EvaluatedOutcome checks its fields", () => {
  test("every worker outcome is accepted", () => {
    for (const outcome of [new Output("v"), new Route("a"), new Grow({}), new EvaluationFailure(failure())]) {
      expect(new EvaluatedOutcome(outcome, null).outcome).toBe(outcome);
    }
  });

  test("an outcome outside the closed sum is refused", () => {
    for (const outcome of [{ value: "v" }, "done", null]) {
      expect(thrown(() => new EvaluatedOutcome(outcome as Output, null))).toEqual(
        new TypeError("outcome must be a WorkerOutcome"),
      );
    }
  });

  test("evidence must be exactly an EvaluationEvidence", () => {
    const evidence = new EvaluationEvidence(null, null, null, null);
    expect(new EvaluatedOutcome(new Output(), evidence).evidence).toBe(evidence);

    class Subclass extends EvaluationEvidence {}
    for (const bad of [new Subclass(null, null, null, null), {}]) {
      expect(thrown(() => new EvaluatedOutcome(new Output(), bad as EvaluationEvidence))).toEqual(
        new TypeError("evidence must be an EvaluationEvidence or null"),
      );
    }
  });

  test("bindings and backward capture travel together", () => {
    const bindings = new BindingCollector().freeze();
    const capture = new BackwardCapture({ inputNames: ["x"] });

    const evaluated = new EvaluatedOutcome(new Output(), null, bindings, capture);
    expect(evaluated.bindings).toBe(bindings);
    expect(evaluated.backwardCapture).toBe(capture);

    const together = new RangeError("binding capture and backward invocation metadata must be retained together");
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, null))).toEqual(together);
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, null, capture))).toEqual(together);
  });

  test("bindings must be exactly a BindingCapture", () => {
    const capture = new BackwardCapture();
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, {} as never, capture))).toEqual(
      new TypeError("bindings must be a BindingCapture or null"),
    );
  });

  test("backward capture must be exactly a BackwardCapture and is checked again", () => {
    const bindings = new BindingCollector().freeze();
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, {} as never))).toEqual(
      new TypeError("backward_capture must be a BackwardCapture or null"),
    );

    // A capture built without its constructor is checked as the constructor would.
    const forged = Object.assign(Object.create(BackwardCapture.prototype) as BackwardCapture, {
      inputNames: ["x", "x"],
      variableNames: [],
      scope: null,
      nodeData: {},
    });
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, forged))).toEqual(
      new RangeError("input_names must contain unique names"),
    );
  });
});

describe("requireSteering", () => {
  const inbox: Steering = {
    interruptSignal: () => new AbortController().signal,
    drain: () => [],
  };

  test("returns null or the inbox itself", () => {
    expect(requireSteering(null)).toBeNull();
    expect(requireSteering(inbox)).toBe(inbox);
  });

  test("refuses anything without both methods", () => {
    const message = "steering must provide interruptSignal(nodeId) and drain(nodeId), or be null";
    for (const bad of [
      undefined,
      {},
      "steering",
      { drain: inbox.drain },
      { interruptSignal: inbox.interruptSignal },
      { interruptSignal: inbox.interruptSignal, drain: [] },
    ]) {
      expect(thrown(() => requireSteering(bad))).toEqual(new TypeError(message));
    }
  });
});

describe("the steering capability", () => {
  /** Wrap an inbox whose methods are `interruptSignal` and `drain`. */
  function wrap(methods: { interruptSignal?: (id: string) => unknown; drain?: (id: string) => unknown }): Steering {
    return new SteeringCapability({
      interruptSignal: methods.interruptSignal ?? (() => new AbortController().signal),
      drain: methods.drain ?? (() => []),
    } as Steering);
  }

  test("passes the node id and returns the host's signal", () => {
    const signal = new AbortController().signal;
    const seen: string[] = [];
    const steering = wrap({ interruptSignal: (id) => (seen.push(id), signal) });

    expect(steering.interruptSignal("n")).toBe(signal);
    expect(seen).toEqual(["n"]);
  });

  test("an interrupt signal that isn't an AbortSignal is a capability fault", () => {
    for (const bad of [true, null, {}]) {
      const fault = thrown(() => wrap({ interruptSignal: () => bad }).interruptSignal("n"));
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toEqual(
        new TypeError("steering.interruptSignal must return an AbortSignal"),
      );
    }
  });

  test("drain returns a fresh array of the host's messages", () => {
    const messages = ["a", "b"];
    const drained = wrap({ drain: () => messages }).drain("n");

    expect(drained).toEqual(["a", "b"]);
    expect(drained).not.toBe(messages);
  });

  test("drain may return a promise of the messages", async () => {
    const drained = wrap({ drain: async () => ["a"] }).drain("n");

    expect(drained).toBeInstanceOf(Promise);
    expect(await drained).toEqual(["a"]);
  });

  test("drain messages that aren't an array of strings are a capability fault", async () => {
    const shape = new TypeError("steering.drain must return an array of strings");
    for (const bad of ["a", ["a", 1], null, new Set(["a"])]) {
      const fault = thrown(() => wrap({ drain: () => bad }).drain("n"));
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toEqual(shape);

      const late = await rejected(wrap({ drain: async () => bad }).drain("n"));
      expect(late).toBeInstanceOf(CapabilityFault);
      expect((late as CapabilityFault).original).toEqual(shape);
    }
  });

  test("a host throw or rejection is a capability fault keeping the original", async () => {
    const expected = new Error("inbox failed");
    for (const fault of [
      thrown(() => wrap({ interruptSignal: () => { throw expected; } }).interruptSignal("n")),
      thrown(() => wrap({ drain: () => { throw expected; } }).drain("n")),
      await rejected(wrap({ drain: () => Promise.reject(expected) }).drain("n")),
    ]) {
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toBe(expected);
    }
  });

  test("a capability fault from the host passes through", async () => {
    const carried = new CapabilityFault(new Error("inner"));
    expect(thrown(() => wrap({ drain: () => { throw carried; } }).drain("n"))).toBe(carried);
    expect(await rejected(wrap({ drain: () => Promise.reject(carried) }).drain("n"))).toBe(carried);
  });
});
