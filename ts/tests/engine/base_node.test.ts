/**
 * The `Node` base class of `nodes/base.ts`: a port of `tests/engine/test_base_node.py`. Ported:
 * `test_node_stores_one_canonical_output_contract` and
 * `test_node_rejects_invalid_output_contract_inputs`. From `tests/engine/test_node_run_wrapper.py`:
 * `test_validate_assertions_names_the_given_path`.
 *
 * TS-only: a new node's defaults, a subclass overriding traits, the constructor's origin check, the
 * parse error's `cause`, the node's type table reaching the assertion parse, `parseNamedCondition`
 * refusing every constructor when given no table, `checkNamedPrompt`'s prefix and `cause`,
 * `validatePrompts` being a no-op on the base, `validateExecutionSetup` (a missing `forward`, then
 * the assertion parse), the stage markers' `original`, `cause` and message (a thrown value with no
 * text is named by its type), `NodeAssertFailed`'s defaults, `hostOutput` (a host value taken in as
 * a yian value, or a `HostOutputError` at the refused path), and the base `exceptionCode` and
 * `exceptionOrigin` giving a `HostOutputError` `contract.output` at the output contract's origin
 * for its path while any other error, a plain `TypeCheckError` too, stays `node.execute` at the
 * node's origin.
 *
 * Ported (adapted), through the shared `drive`: `test_result_node_returns_output`,
 * `test_streaming_node_forwards_chunks_and_returns_output` (the fake streams through
 * `caps.emitStream`), `test_raising_node_returns_evaluation_failure` (cause type `Error` for
 * Python's `RuntimeError`), `test_policy_exception_returns_typed_evaluation_failure` (the `Policy`
 * error sets its `name`, which is the cause type), `test_branch_node_returns_route_handle` and
 * `test_node_does_not_write_pool`.
 *
 * Changed: Python's `vars(node)` is `Object.hasOwn`, and `hasattr` is the `in` operator.
 *
 * Not ported: `test_node_rejects_removed_output_type_constructor_keyword`: TS rejects an unknown
 * option when it type-checks the call.
 */

import { describe, expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { StreamChunk } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import {
  AssertionStageError,
  BindingStageError,
  CapabilityFault,
  checkNamedPrompt,
  DeferredBindingFault,
  HostOutputError,
  hostOutput,
  Node,
  NodeAssertFailed,
  NodeBodyError,
  NodeKind,
  type NodeCaps,
  Output,
  parseNamedCondition,
  Route,
} from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import {
  Assertion,
  ContractOrigins,
  type OutputContract,
  outputContractOrigin,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { BranchNode, drive, FailNode, FuncNode as FakeFuncNode, StreamNode } from "./run_state_fixtures.ts";

/** A minimal concrete node; the base constructor is the subject. */
class FuncNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A minimal node with a body, so the setup check reaches the assertions. */
class BodyNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): Output {
    return new Output(null);
  }
}

/** Overrides two traits, so the type check pins that a kind may set a trait to `true`. */
class ToolSpawner extends Node {
  readonly kind = NodeKind.CALL;
  override readonly needsTools = true;
  override readonly isSpawner = true;
}

/** Near-canonical fixture proving nodes require the exact IR carrier. */
class ValueContractSubclass extends ValueContract {}

test("node stores one canonical output contract", () => {
  const untyped = new FuncNode("untyped");
  const contractType = Type.scalar(ValueKind.NUMBER);
  const contract = new ValueContract(contractType, new ContractOrigins({ default: new OriginRef("node/output") }));

  const node = new FuncNode("n", { outputContract: contract });

  expect(untyped.outputContract).toBeNull();
  expect(Object.hasOwn(untyped, "outputContract")).toBe(true);
  expect(Object.hasOwn(untyped, "outputType")).toBe(false);
  expect(node.outputContract).toBe(contract);
  expect((node.outputContract as ValueContract).type).toBe(contractType);
  expect(Object.hasOwn(node, "outputContract")).toBe(true);
  expect(Object.hasOwn(node, "outputType")).toBe(false);
  expect("outputType" in Node.prototype).toBe(false);
  expect("outputType" in node).toBe(false);
});

test.each([
  ["a bare Type", Type.scalar(ValueKind.INTEGER) as unknown as OutputContract],
  ["a ValueContract subclass", new ValueContractSubclass(Type.scalar(ValueKind.INTEGER))],
])("node rejects invalid output contract inputs: %s", (_label, value) => {
  expect(() => new FuncNode("n", { outputContract: value })).toThrow(TypeError);
  expect(() => new FuncNode("n", { outputContract: value })).toThrow(/output_contract.*ValueContract/);
});

test("a new node starts with empty fields and every trait off", () => {
  const node = new FuncNode("n");

  expect([node.id, node.title, node.outputContract, node.params]).toEqual(["n", null, null, null]);
  expect([node.preAsserts, node.postAsserts, node.originContext]).toEqual([[], [], []]);
  expect([node.commitAs, node.originId, node.origin]).toEqual([null, null, null]);
  expect([node.memoryScope, node.executionScope]).toEqual([null, null]);
  expect(node.env).toEqual({});
  expect([node.types, node.wiringLabels, node.bindingLabels]).toEqual([new Map(), new Map(), new Map()]);
  expect([
    node.isSpawner,
    node.commitsInputRecord,
    node.bindsPerItem,
    node.isLoop,
    node.needsMemoryContext,
    node.needsSkills,
    node.needsTools,
    node.needsLiveEvents,
    node.needsSteps,
    node.readsSteerInbox,
    node.needsModels,
    node.derived,
    node.producesMessages,
  ]).toEqual(Array(13).fill(false));
  expect([node.itemSourceKey, node.growDepthDelta]).toEqual([null, null]);
});

test("a node that produces no messages refuses the message methods", () => {
  const node = new FuncNode("n");
  expect(() => node.messageSteps([])).toThrow(new TypeError("node 'n' produces no messages"));
  expect(() => node.rebuildMessages([], null)).toThrow(new TypeError("node 'n' produces no messages"));
});

test("a node opts into its steer inbox through readsSteerInbox; the base node doesn't", () => {
  expect(new FuncNode("f").readsSteerInbox).toBe(false);
  const caps: NodeCaps = { openSteerInbox: (inbox) => ({ inbox, queued: () => false, read: () => null, interrupt: new AbortController().signal, interruptedBy: () => null }) };
  expect(caps.openSteerInbox!("m").read()).toBeNull();
});

test("node rejects an origin that isn't an OriginRef", () => {
  expect(() => new FuncNode("n", { origin: "x" as never })).toThrow(TypeError);
  expect(() => new FuncNode("n", { origin: "x" as never })).toThrow("node origin must be an OriginRef or null");
});

test("a kind overrides its traits", () => {
  const node = new ToolSpawner("s");

  expect([node.needsTools, node.isSpawner, node.isLoop]).toEqual([true, true, false]);
});

test("validate assertions names the given path", () => {
  const node = new FuncNode("inner");
  node.preAsserts = [new Assertion("${x} > 0", "${x} > 0")];

  expect(() => node.validateAssertions({ path: "call/inner" })).toThrow(ExpressionError);
  expect(() => node.validateAssertions({ path: "call/inner" })).toThrow(
    /^node 'call\/inner' pre-assert '\$\{x\} > 0' is not/,
  );
  let error: unknown;
  try {
    node.validateAssertions();
  } catch (caught) {
    error = caught;
  }
  expect((error as Error).message).toStartWith(
    "node 'inner' pre-assert '${x} > 0' is not a valid boolean expression: ",
  );
  expect((error as Error).cause).toBeInstanceOf(ExpressionError);
});

test("assertions parse against the node's type table", () => {
  const node = new FuncNode("n");
  node.preAsserts = [new Assertion("Size(x) == x", "Size(x) == x")];

  expect(() => node.validateAssertions()).toThrow(/^node 'n' pre-assert 'Size\(x\) == x' is not/);
  node.types = new Map([["Size", Type.scalar(ValueKind.INTEGER)]]);
  expect(node.validateAssertions()).toBeUndefined();
});

test("a condition parsed without a type table refuses every constructor", () => {
  expect(() => parseNamedCondition("Size(x) == x", "case 'c'")).toThrow(
    /^case 'c' is not a valid boolean expression: /,
  );
  expect(parseNamedCondition("Size(x) == x", "case 'c'", new Map([["Size", Type.scalar(ValueKind.INTEGER)]])))
    .toBeUndefined();
});

test("a prompt check names its subject and keeps the template error as the cause", () => {
  let error: unknown;
  try {
    checkNamedPrompt("${", "node 'n' prompt");
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith("node 'n' prompt: ");
  expect((error as Error).cause).toBeInstanceOf(ExpressionError);
});

test("validatePrompts checks nothing on the base", () => {
  const node = new FuncNode("n");
  // An unparseable assert: the prompt check must not read the assertions.
  node.preAsserts = [new Assertion("${x} > 0", "${x} > 0")];

  expect(node.validatePrompts()).toBeUndefined();
  expect(node.validatePrompts({ path: "call/n" })).toBeUndefined();
});

test("the setup check rejects a node that defines no forward", () => {
  const node = new FuncNode("n");

  expect(() => node.validateExecutionSetup()).toThrow(Error);
  expect(() => node.validateExecutionSetup()).toThrow("node 'n' (FuncNode) defines no forward");
  expect(() => node.forward({}, {})).toThrow("node 'n' (FuncNode) defines no forward");
});

test("the setup check parses the assertions of a node with a body", () => {
  const node = new BodyNode("n");
  node.postAsserts = [new Assertion("output > 0", "output is positive")];

  expect(node.validateExecutionSetup()).toBeUndefined();
  node.postAsserts = [new Assertion("output >", "output >")];
  let error: unknown;
  try {
    node.validateExecutionSetup();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).name).toBe("ExpressionError");
  expect((error as Error).message).toStartWith("node 'n' post-assert 'output >' is not a valid boolean expression: ");
});

test.each([
  ["NodeBodyError", NodeBodyError],
  ["BindingStageError", BindingStageError],
  ["AssertionStageError", AssertionStageError],
  ["DeferredBindingFault", DeferredBindingFault],
  ["CapabilityFault", CapabilityFault],
] as const)("the %s marker keeps its original", (name, marker) => {
  const Marker = marker as new (original: unknown) => Error & { original: unknown };
  const original = name === "BindingStageError"
    ? new BindingError("bad default", { failureCode: "binding.evaluate" })
    : new NodeAssertFailed("bad default");
  const wrapped = new Marker(original);

  expect(wrapped).toBeInstanceOf(Error);
  expect(wrapped.name).toBe(name);
  expect(wrapped.original).toBe(original);
  expect(wrapped.cause).toBe(original);
  expect(wrapped.message).toBe("bad default");

  const thrown = new Marker("boom");
  expect(thrown.original).toBe("boom");
  expect(thrown.cause).toBe("boom");
  expect(thrown.message).toBe("boom");
  expect(new Marker(7n).message).toBe("7");

  // A value with no text still wraps, named by its type.
  const bare: unknown = Object.create(null);
  const textless = new Marker(bare);
  expect(textless.original).toBe(bare);
  expect(textless.message).toBe("object");
});

test("NodeAssertFailed defaults its failure code and keeps its assertion and cause", () => {
  const plain = new NodeAssertFailed("bad");

  expect(plain).toBeInstanceOf(Error);
  expect(plain.name).toBe("NodeAssertFailed");
  expect(plain.message).toBe("bad");
  expect(plain.failureCode).toBe("assertion.failed");
  expect(plain.assertion).toBeNull();
  expect("cause" in plain).toBe(false);

  const assertion = new Assertion("output > 0", "output is positive");
  const cause = new ExpressionError("cannot compare");
  const failed = new NodeAssertFailed("could not be evaluated", {
    assertion,
    failureCode: "condition.evaluate",
    cause,
  });
  expect(failed.assertion).toBe(assertion);
  expect(failed.failureCode).toBe("condition.evaluate");
  expect(failed.cause).toBe(cause);
});

test("hostOutput takes a host value in as a yian value", () => {
  expect(hostOutput(5)).toBe(5n);
  expect(hostOutput({ a: [1, 2.5] })).toEqual({ a: [1n, 2.5] });
});

test("hostOutput refuses a host value that isn't a yian value", () => {
  let error: unknown;
  try {
    hostOutput({ a: new Date(0) });
  } catch (caught) {
    error = caught;
  }

  expect(error).toBeInstanceOf(HostOutputError);
  expect(error).toBeInstanceOf(TypeCheckError);
  const refused = error as HostOutputError;
  expect(refused.name).toBe("TypeCheckError");
  expect(refused.message).toBe("output at a: Date is not a yian value");
  expect(refused.path).toEqual(["a"]);
  expect(refused.cause).toBeInstanceOf(TypeCheckError);
});

describe("a host output error is a contract.output failure", () => {
  const nodeOrigin = new OriginRef("node");
  const fieldOrigin = new OriginRef("node/output/a");
  const contract = new ValueContract(
    Type.scalar(ValueKind.OBJECT),
    new ContractOrigins({ default: new OriginRef("node/output"), entries: [[["a"], fieldOrigin]] }),
  );
  const refused = new HostOutputError("output at a: Date is not a yian value", { path: ["a"] });

  test("the code is contract.output, else node.execute", () => {
    const node = new FuncNode("n", { origin: nodeOrigin });

    expect(node.exceptionCode(refused)).toBe("contract.output");
    expect(node.exceptionCode(new Error("boom"))).toBe("node.execute");
  });

  test("the origin is the output contract's at the error path, else the node's", () => {
    const typed = new FuncNode("n", { origin: nodeOrigin, outputContract: contract });
    const untyped = new FuncNode("n", { origin: nodeOrigin });

    expect(typed.exceptionOrigin(refused)).toBe(outputContractOrigin(contract, ["a"]));
    expect(typed.exceptionOrigin(refused)).toBe(fieldOrigin);
    expect(untyped.exceptionOrigin(refused)).toBeNull();
    expect(typed.exceptionOrigin(new Error("boom"))).toBe(nodeOrigin);
  });

  test("a plain TypeCheckError stays a node.execute failure at the node", () => {
    const node = new FuncNode("n", { origin: nodeOrigin, outputContract: contract });
    const mistyped = new TypeCheckError("argument at a: expected int", { path: ["a"] });

    expect(node.exceptionCode(mistyped)).toBe("node.execute");
    expect(node.exceptionOrigin(mistyped)).toBe(nodeOrigin);
  });
});

describe("evaluation through drive", () => {
  /** Named so the boundary yields a failure with cause type `Policy`. */
  class Policy extends Error {
    override name = "Policy";
  }

  /** Throws a `Policy` error from its body. */
  class ReturnsFailedNode extends Node {
    readonly kind = NodeKind.CODE;

    override forward(): Output {
      throw new Policy("declined");
    }
  }

  test("result node returns output", async () => {
    expect(await drive(new FakeFuncNode("n", () => ({ output: "ok" })))).toEqual(new Output({ output: "ok" }));
  });

  test("streaming node forwards chunks and returns output", async () => {
    const chunks: StreamChunk[] = [];

    const result = await drive(new StreamNode("n", ["a", "b", "c"]), { emitStream: (chunk) => chunks.push(chunk) });

    expect(chunks.map((chunk) => chunk.chunk)).toEqual(["a", "b", "c"]);
    expect(chunks.every((chunk) => chunk instanceof StreamChunk)).toBe(true);
    expect(result).toEqual(new Output({ output: "abc" }));
  });

  test("raising node returns evaluation failure", async () => {
    const result = await drive(new FailNode("n", "kaboom"));

    expect(result).toBeInstanceOf(EvaluationFailure);
    const { failure } = result as EvaluationFailure;
    expect([failure.message, failure.causeType]).toEqual(["kaboom", "Error"]);
  });

  test("policy exception returns typed evaluation failure", async () => {
    const result = await drive(new ReturnsFailedNode("n"));

    expect(result).toBeInstanceOf(EvaluationFailure);
    const { failure } = result as EvaluationFailure;
    expect([failure.message, failure.causeType]).toEqual(["declined", "Policy"]);
  });

  test("branch node returns route handle", async () => {
    expect(await drive(new BranchNode("n", "case_a"))).toEqual(new Route("case_a"));
  });

  test("node does not write pool", async () => {
    const pool = new VariablePool();

    expect(await drive(new FakeFuncNode("n", () => ({ output: "x" })), { pool })).toEqual(new Output({ output: "x" }));
    expect(pool.get("n")).toBeNull();
  });
});
