/**
 * The base `run` template checks the pre- and post-asserts against the bound record only. The
 * tests drive the template directly, on a small node that implements `forward`.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import {
  AssertionStageError,
  BackwardContext,
  BackwardResult,
  BindingStageError,
  CapabilityFault,
  DeferredBindingFault,
  Grow,
  Node,
  NodeAssertFailed,
  NodeBodyError,
  NodeKind,
  type NodeResult,
  Output,
  Route,
} from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { BranchNode, drive } from "../helpers/run_state_fixtures.ts";

/**
 * A node that implements `forward` (not `run`), so the base template is under test. `forward`
 * records that it ran, so a pre-assert failure can be shown to stop it.
 */
class ImplNode extends Node {
  readonly kind = NodeKind.CODE;
  value: unknown;
  ran = false;

  constructor(nodeId: string, value: unknown) {
    super(nodeId);
    this.value = value;
  }

  override forward(): NodeResult {
    this.ran = true;
    return new Output(this.value);
  }
}

/** A spawner-like fake whose body grows with its inputs as the seed. */
class GrowNode extends Node {
  readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;

  override forward(inputs: Record<string, unknown>): NodeResult {
    return new Grow(inputs);
  }
}

/** A node whose body throws one given value. */
class ThrowNode extends Node {
  readonly kind = NodeKind.CODE;
  error: unknown;

  constructor(nodeId: string, error: unknown) {
    super(nodeId);
    this.error = error;
  }

  override forward(): NodeResult {
    throw this.error;
  }
}

/** A node whose async body awaits once, then throws `error` if given, else returns `value`. */
class AsyncNode extends Node {
  readonly kind = NodeKind.CODE;
  value: unknown;
  error: unknown;

  constructor(nodeId: string, value: unknown, error?: unknown) {
    super(nodeId);
    this.value = value;
    this.error = error;
  }

  override async forward(): Promise<NodeResult> {
    await null;
    if (this.error !== undefined) throw this.error;
    return new Output(this.value);
  }
}

/** A node whose body changes the record it receives before returning it. */
class MutatingNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(inputs: Record<string, unknown>): NodeResult {
    (inputs.x as Record<string, unknown>).n = 0n;
    inputs.x = null;
    return new Output(1n);
  }
}

/** A node with no body. */
class NoForwardNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** Return what `promise` rejects with, failing if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** A parameter whose stored default violates its `int` contract. */
function badDefault(): Parameter {
  return new Parameter("x", {
    contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
    hasDefault: true,
    default: "30",
  });
}

test("default backward stops feedback with empty channels", () => {
  const node = new ImplNode("n", null);

  expect(node.backward(new BackwardContext({}), ["unused"])).toEqual(new BackwardResult());
});

test("run returns forward output when asserts hold", async () => {
  const node = new ImplNode("n", { output: 42n });
  node.preAsserts = [new Assertion("x > 0", "x is positive")];
  node.postAsserts = [new Assertion("output.output == 42", "output is 42")];

  const result = await node.run({ x: 5n });

  expect(result).toBeInstanceOf(Output);
  expect((result as Output).value).toEqual({ output: 42n });
  expect(node.ran).toBe(true);
});

test("pre-assert failure raises before forward runs", async () => {
  const node = new ImplNode("n", { output: 1n });
  const assertion = new Assertion("x > 100", "x is greater than 100", { origin: new OriginRef("assert:n:0") });
  node.preAsserts = [assertion];

  const error = await rejection(node.run({ x: 5n }));

  expect(error).toBeInstanceOf(NodeAssertFailed);
  expect(node.ran).toBe(false);
  expect((error as NodeAssertFailed).assertion).toBe(assertion);
  expect((error as Error).message).toContain("pre-assert failed");
  expect((error as Error).message).toContain("'n'");
});

test("post-assert failure on output raises", async () => {
  const node = new ImplNode("n", { output: 7n });
  node.postAsserts = [new Assertion("output.output > 100", "output is greater than 100")];

  const error = await rejection(node.run({}));

  expect(error).toBeInstanceOf(NodeAssertFailed);
  expect(node.ran).toBe(true);
  expect((error as Error).message).toContain("post-assert failed");
  expect((error as Error).message).toContain("'n'");
});

test("assertion evaluates expression and reports display", async () => {
  const node = new ImplNode("n", null);
  const assertion = new Assertion("__assert_calc > 100", "authored text that is not an expression");
  node.preAsserts = [assertion];

  const error = await rejection(node.run({ __assert_calc: 5n }));

  expect(error).toBeInstanceOf(NodeAssertFailed);
  expect((error as Error).message).toContain("authored text that is not an expression");
  expect((error as NodeAssertFailed).assertion).toBe(assertion);
});

test("assertion display is not evaluated when expression holds", async () => {
  const node = new ImplNode("n", null);
  node.preAsserts = [new Assertion("x > 0", "{{ definitely not executable")];

  expect(await node.run({ x: 1n })).toEqual(new Output(null));
});

test("duplicate assertions retain first occurrence origin", async () => {
  const node = new ImplNode("n", null);
  const first = new Assertion("1 == 2", "must hold", { origin: new OriginRef("assert:first") });
  const second = new Assertion("1 == 2", "must hold", { origin: new OriginRef("assert:second") });
  node.preAsserts = [first, second];

  const error = await rejection(node.run({}));

  expect((error as NodeAssertFailed).assertion).toBe(first);
});

test("wrapper delegates for route kind", async () => {
  const node = new BranchNode("b", "yes");

  expect(BranchNode.prototype.run).toBe(Node.prototype.run);
  const result = await node.run({});

  expect(result).toBeInstanceOf(Route);
  expect((result as Route).handle).toBe("yes");
});

test("unparseable pre-assert raises its parse error before forward", async () => {
  const node = new ImplNode("n", { output: 1n });
  node.preAsserts = [new Assertion("${x} > 0", "x positive")];

  const error = await rejection(node.run({ x: 5n }));

  expect(error).toBeInstanceOf(ExpressionError);
  const message = (error as Error).message;
  expect(message).toStartWith("node 'n' pre-assert 'x positive' (`${x} > 0`) is not a valid boolean expression: ");
  expect(message).toContain("write `x > 0`");
  expect(node.ran).toBe(false);
});

test("unparseable post-assert raises before forward runs", async () => {
  const node = new ImplNode("n", { output: 1n });
  node.preAsserts = [new Assertion("x > 0", "x positive")];
  node.postAsserts = [new Assertion("output.output >=", "output.output >=")];

  const error = await rejection(node.run({ x: 5n }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith(
    "node 'n' post-assert 'output.output >=' is not a valid boolean expression: ",
  );
  expect(node.ran).toBe(false);
});

test("assert evaluation error still fails the assert", async () => {
  const node = new ImplNode("n", { output: 1n });
  const assertion = new Assertion("output > 0", "output is positive");
  node.postAsserts = [assertion];

  const error = await rejection(node.run({}));

  expect(error).toBeInstanceOf(NodeAssertFailed);
  const failed = error as NodeAssertFailed;
  expect(failed.failureCode).toBe("condition.evaluate");
  expect(failed.assertion).toBe(assertion);
  expect(failed.cause).toBeInstanceOf(ExpressionError);
  expect(failed.message).toBe(
    "node 'n' post-assert 'output is positive' (`output > 0`) could not be evaluated: " +
      (failed.cause as Error).message,
  );
});

test("a failing pre-assert rejects the promise rather than throwing", async () => {
  const node = new ImplNode("n", null);
  node.preAsserts = [new Assertion("x > 100", "x is large")];

  const promise = node.run({ x: 5n });

  await expect(promise).rejects.toBeInstanceOf(NodeAssertFailed);
});

test("a node without forward rejects", async () => {
  const node = new NoForwardNode("n");

  await expect(node.run({})).rejects.toThrow("node 'n' (NoForwardNode) defines no forward");
});

test("a body changing its input record doesn't change what a post-assert sees", async () => {
  const node = new MutatingNode("n");
  node.postAsserts = [new Assertion("x.n == 5 and output == 1", "x is as bound")];
  const inputs = { x: { n: 5n } };

  expect(await node.run(inputs)).toEqual(new Output(1n));
});

test("a false assert has the failure code assertion.failed", async () => {
  const node = new ImplNode("n", 1n);
  node.postAsserts = [new Assertion("output > 5", "output is large")];

  const error = await rejection(node.run({}));

  expect((error as NodeAssertFailed).failureCode).toBe("assertion.failed");
  expect((error as Error).message).toBe("node 'n' post-assert failed: output is large");
  expect((error as Error).cause).toBeUndefined();
});

test("a Route and a Grow pass a failing post-assert untouched", async () => {
  const branch = new BranchNode("b", "yes");
  branch.postAsserts = [new Assertion("output == 'never'", "output is never")];
  const grow = new GrowNode("g");
  grow.postAsserts = [new Assertion("output == 'never'", "output is never")];

  expect(await branch.run({})).toEqual(new Route("yes"));
  const result = await grow.run({ topic: "ACME" });
  expect(result).toBeInstanceOf(Grow);
  expect((result as Grow).seed).toEqual({ topic: "ACME" });
});

test("runForEvaluator wraps a body error, keeping a non-Error original", async () => {
  const thrown = new Error("boom");
  const error = await rejection(new ThrowNode("n", thrown).runForEvaluator({}));
  expect(error).toBeInstanceOf(NodeBodyError);
  expect((error as NodeBodyError).original).toBe(thrown);

  const text = await rejection(new ThrowNode("n", "boom").runForEvaluator({}));
  expect(text).toBeInstanceOf(NodeBodyError);
  expect((text as NodeBodyError).original).toBe("boom");
  expect((text as Error).message).toBe("boom");

  // A thrown value with no text still wraps, kept by identity.
  const bare: unknown = Object.create(null);
  const textless = await rejection(new ThrowNode("n", bare).runForEvaluator({}));
  expect(textless).toBeInstanceOf(NodeBodyError);
  expect((textless as NodeBodyError).original).toBe(bare);
});

test("an async forward that throws after an await rejects, wrapped by runForEvaluator", async () => {
  const thrown = new Error("late boom");

  expect(await rejection(new AsyncNode("n", null, thrown).run({}))).toBe(thrown);
  const error = await rejection(new AsyncNode("n", null, thrown).runForEvaluator({}));
  expect(error).toBeInstanceOf(NodeBodyError);
  expect((error as NodeBodyError).original).toBe(thrown);
});

test("an async forward's Output gets its post-asserts", async () => {
  const holding = new AsyncNode("n", 42n);
  holding.postAsserts = [new Assertion("output == 42", "output is 42")];
  const failing = new AsyncNode("n", 7n);
  failing.postAsserts = [new Assertion("output > 100", "output is large")];

  expect(await holding.run({})).toEqual(new Output(42n));
  const error = await rejection(failing.run({}));
  expect(error).toBeInstanceOf(NodeAssertFailed);
  expect((error as Error).message).toContain("post-assert failed");
  const marked = await rejection(failing.runForEvaluator({}));
  expect(marked).toBeInstanceOf(AssertionStageError);
});

test("runForEvaluator wraps a default's binding error", async () => {
  const node = new ImplNode("n", null);
  node.params = [badDefault()];

  const error = await rejection(node.runForEvaluator({}));

  expect(error).toBeInstanceOf(BindingStageError);
  expect((error as BindingStageError).original).toBeInstanceOf(BindingError);
  expect(node.ran).toBe(false);
});

test("runForEvaluator wraps an assert failure", async () => {
  const pre = new ImplNode("n", null);
  pre.preAsserts = [new Assertion("x > 100", "x is large")];
  const post = new ImplNode("n", 1n);
  post.postAsserts = [new Assertion("output > 5", "output is large")];

  const preError = await rejection(pre.runForEvaluator({ x: 5n }));
  const postError = await rejection(post.runForEvaluator({}));

  expect(preError).toBeInstanceOf(AssertionStageError);
  expect((preError as AssertionStageError).original).toBeInstanceOf(NodeAssertFailed);
  expect((preError as AssertionStageError).original.assertion).toBe(pre.preAsserts[0]!);
  expect(postError).toBeInstanceOf(AssertionStageError);
  expect((postError as AssertionStageError).original.assertion).toBe(post.postAsserts[0]!);
});

test("runForEvaluator passes the evaluator's markers through and wraps a body's assertion marker", async () => {
  const passed = [
    new BindingStageError(new BindingError("bad item", { failureCode: "binding.evaluate" })),
    new DeferredBindingFault(new Error("binder down")),
    new CapabilityFault(new Error("host down")),
    new NodeBodyError(new Error("inner body")),
  ];
  const marker = new AssertionStageError(new NodeAssertFailed("bad"));

  for (const fault of passed) {
    expect(await rejection(new ThrowNode("n", fault).runForEvaluator({}))).toBe(fault);
  }
  const wrapped = await rejection(new ThrowNode("n", marker).runForEvaluator({}));
  expect(wrapped).toBeInstanceOf(NodeBodyError);
  expect((wrapped as NodeBodyError).original).toBe(marker);
});

test("runForEvaluator captures the defaulted inputs before it returns, ahead of the pre-asserts", async () => {
  const node = new ImplNode("n", null);
  node.params = [new Parameter("x", { hasDefault: true, default: 5n })];
  node.preAsserts = [new Assertion("x > 100", "x is large")];
  const captured: Record<string, unknown>[] = [];

  const promise = node.runForEvaluator({}, {}, (inputs) => captured.push(inputs));

  expect(captured).toEqual([{ x: 5n }]);
  await expect(promise).rejects.toBeInstanceOf(AssertionStageError);
});

test("run rejects with the original errors", async () => {
  const thrown = new Error("boom");
  const bound = new ImplNode("n", null);
  bound.params = [badDefault()];
  const asserted = new ImplNode("n", null);
  asserted.preAsserts = [new Assertion("x > 100", "x is large")];

  expect(await rejection(new ThrowNode("n", thrown).run({}))).toBe(thrown);
  expect(await rejection(new ThrowNode("n", "boom").run({}))).toBe("boom");
  expect(await rejection(bound.run({}))).toBeInstanceOf(BindingError);
  expect(await rejection(asserted.run({ x: 5n }))).toBeInstanceOf(NodeAssertFailed);
});

test("wrapper delegates to real code node forward", async () => {
  const node = new CodeNode("c", { code: "return inputs.v" });
  // The kind defines no `run`, so the base template runs the kind's `forward`.
  expect(node.run).toBe(Node.prototype.run);

  const result = await node.run({ v: "hi" });

  expect(result).toBeInstanceOf(Output);
  expect((result as Output).value).toBe("hi");
});

test("wrapper delegates for grow kind", async () => {
  // A spawner (CALL -> Grow): `run` is the inherited template and resolves to `forward`'s Grow.
  const node = new CallNode("r", { flowId: "c", child: {}, childInputs: [] });
  expect(node.run).toBe(Node.prototype.run);
  const result = await node.run({ topic: "ACME" });

  expect(result).toBeInstanceOf(Grow);
  expect((result as Grow).seed).toEqual({ topic: "ACME" });
});

test("runtime does not drive backward", async () => {
  // Forward execution remains the runtime's only node-body call.
  class BackwardTrapNode extends ImplNode {
    override backward(): BackwardResult {
      throw new Error("backward must remain dormant");
    }
  }

  expect(await drive(new BackwardTrapNode("n", "ok"))).toEqual(new Output("ok"));
});

test("eval node checks a post-assert through the template", async () => {
  // evalNode doesn't check asserts itself: the template around `forward` does.
  const node = new ImplNode("n", "v");
  node.postAsserts = [new Assertion("output == 'never'", "output must equal never")];

  const result = await drive(node);

  expect(result).toBeInstanceOf(EvaluationFailure);
  expect((result as EvaluationFailure).failure.message).toContain("post-assert");
});
