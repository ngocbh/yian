/**
 * The raw `evalNode` read/execution boundary: a port of `tests/engine/test_eval_node.py`.
 *
 * The evaluator binds from flow-owned wiring, runs the node, and resolves to exactly one
 * `Output | Route | Grow | EvaluationFailure`.
 *
 * Ported: `test_success_returns_output_without_observer_events`,
 * `test_baked_commit_target_remains_engine_owned`,
 * `test_node_chosen_commit_target_remains_on_output`,
 * `test_binding_failure_returns_diagnostic_evaluation_failure`,
 * `test_binding_carrier_resolves_at_the_node_read_boundary`,
 * `test_routing_returns_route_without_output_value`,
 * `test_non_spawner_grow_returns_clear_evaluation_failure`,
 * `test_bad_return_type_returns_clear_evaluation_failure`,
 * `test_reserved_input_rejection_is_an_input_contract_failure` (2 rows),
 * `test_case_rejects_a_non_origin_predicate_carrier`, `test_node_raise_returns_evaluation_failure`,
 * `test_post_assert_isolated_from_node_input_mutation`,
 * `test_wrapper_post_assertion_uses_assertion_stage_and_origin`,
 * `test_body_raised_assertion_exception_remains_node_execution`,
 * `test_node_execution_prefers_operation_origin_then_node_fallback` (2 rows),
 * `test_node_execution_uses_the_node_exception_code` (2 rows),
 * `test_exception_code_is_asked_only_for_the_body` (3 rows) and
 * `test_per_item_wrapper_default_keeps_binding_stage`.
 *
 * Ported (adapted): `test_map_over_accepts_a_binding_carrier`,
 * `test_wait_until_accepts_a_binding_carrier`,
 * `test_unbaked_composite_is_a_setup_fault`,
 * `test_case_without_predicate_is_a_setup_fault`,
 * `test_case_with_non_string_predicate_is_a_setup_fault`,
 * `test_map_over_not_a_list_message_is_byte_identical` (on the node and through the evaluator),
 * `test_missing_reserved_wiring_is_an_ir_fault` (2 rows; the node's plain `Error`, where Python
 * raises `KeyError`), `test_malformed_parameter_carrier_is_an_ir_fault` (`params` that isn't an
 * array, where Python uses a tuple), `test_wrapper_assertion_failure_uses_assertion_stage_and_origin`
 * (the traceback is the assertion error's stack, with no marker name),
 * `test_body_raised_binding_exception_remains_node_execution` (Python's generator row is an async
 * `forward` that rejects), `test_base_exception_code_is_node_execution` (on a minimal subclass:
 * `Node` is abstract), `test_run_override_exception_is_node_execution` (5 rows; the override is
 * `async`), `test_missing_node_implementation_is_a_setup_fault` (a subclass without `forward`),
 * `test_malformed_assertion_carrier_is_an_ir_fault` (a `TypeError`, where Python raises
 * `AttributeError`), `test_map_deferred_item_binding_keeps_input_contract_stage` and
 * `test_map_deferred_expression_keeps_binding_stage` (on a local node that binds per item, since
 * MAP's body is ported later), and `test_map_deferred_unexpected_binder_fault_escapes` (on the same
 * node; a getter on the item throws the host fault Python's `__deepcopy__` bomb raises).
 *
 * Not ported: `test_malformed_result_context_is_an_ir_fault`: a template string can't fail on a
 * malformed `kind`.
 *
 * Changed: the shared `drive` evaluates the node in a one-node `CompiledFlow` (`nodeFlow`) whose
 * wiring is a `FuncNode`'s `reads` or the given sources, as Python's stub flow derives it. Python's `RuntimeError` is a
 * plain `Error` (cause type `Error`) and its `ValueError` a `RangeError`. The non-NodeResult
 * message names the JS type (`Object`). In `test_binding_carrier_unwraps_exactly_one_layer` the
 * inner carrier isn't a yian value, so binding refuses it (`binding.evaluate`) where Python passes
 * it through as data. The MAP and WAIT carrier rows check the resolved `over` and `until` through
 * `bindReserved`, not the `Grow` or `Output` the bodies build from them, and the setup faults go
 * through `run`, which rejects with the node's own error; the non-string `when` is a cast. The
 * not-a-list message is pinned as the golden format's text.
 *
 * TS-only: `test_map_over_not_a_list_message_is_byte_identical` also checks the error's `origin`
 * (the carrier's, else `null`); `evalNode`'s argument checks; a node that records steps and whose
 * reserved key fails to bind after an earlier start recorded steps rejects with a `RangeError`,
 * while a node that doesn't read its steps gets the ordinary binding failure.
 *
 * Later (T4-9): `test_case_predicate_failure_selects_the_active_ordered_origin` and
 * `test_source_free_case_predicate_failure_falls_back_to_the_node_origin`.
 *
 * Later (T4-10): the run-result half of `test_map_over_accepts_a_binding_carrier` (the `Grow`
 * through the evaluator). Later (T4-11): the run-result half of
 * `test_wait_until_accepts_a_binding_carrier` (the `Output` through the evaluator).
 */

import { describe, expect, test } from "bun:test";

import { START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ConversationInvocation } from "../../src/memory/contracts.ts";
import {
  AssertionStageError,
  BindingStageError,
  Grow,
  Node,
  NodeAssertFailed,
  type NodeCaps,
  NodeKind,
  type NodeResult,
  Output,
  Route,
} from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  NodeSteps,
  Parameter,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { type EvalNodeOptions, EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, type Value, ValueKind } from "../../src/typesys/values.ts";
import { drive, FuncNode, nodeFlow, rejected } from "./run_state_fixtures.ts";

/** Return the failure `drive` resolves to, or fail when it resolves to anything else. */
async function driveFailure(
  node: Node,
  options: Parameters<typeof drive>[1] = {},
): Promise<EvaluationFailure["failure"]> {
  const outcome = await drive(node, options);
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure;
}

/** Return one configured result from its body. */
class ResultNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  readonly result: unknown;

  constructor(nodeId: string, result: unknown, options: { origin?: OriginRef | null } = {}) {
    super(nodeId, options);
    this.result = result;
  }

  override forward(): NodeResult {
    return this.result as NodeResult;
  }
}

/** Return one configured route handle (a CASE-shaped router). */
class RouteNode extends ResultNode {
  override readonly kind = NodeKind.CASE;
}

test("map over accepts a binding carrier", () => {
  // MAP's reserved iteration source uses the ordinary binding read seam.
  const node = new MapNode("m", { flowId: "child", child: {} });
  const wiring = new Map([["over", new Binding(["first", "second"], { origin: new OriginRef("map-over") })]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ over: ["first", "second"] });
});

test("wait until accepts a binding carrier", () => {
  // WAIT's dedicated resolver accepts the carrier shape it bypasses elsewhere.
  const node = new WaitNode("wait");
  const source = new Binding("2000-01-01T00:00:00+00:00", { origin: new OriginRef("wait-until") });
  const wiring = new Map([["until", source]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ until: "2000-01-01T00:00:00+00:00" });
});

for (const [label, make, inputs] of [
  ["call", () => new CallNode("call", { flowId: "child" }), {}],
  ["map", () => new MapNode("map", { flowId: "child" }), { over: [] }],
  ["loop", () => new LoopNode("loop", { flowId: "child", predicate: "true", maxIters: 1 }), {}],
] as const) {
  test(`unbaked composite is a setup fault (${label})`, async () => {
    // A missing compiled child is malformed setup, not expected node rejection.
    const node: Node = make();
    node.preAsserts = [new Assertion("false", "must not mask malformed setup")];

    await expect(node.run(inputs)).rejects.toThrow("baked");
  });
}

test("map over not a list message is byte identical", () => {
  // Raw and carrier-wrapped sources produce the same origin-neutral error text.
  const pool = new VariablePool();
  pool.set("__start__", { bad: "notalist" });
  const rawSource = "${input.bad}";
  const origin = new OriginRef("map-over");
  for (const source of [rawSource, new Binding(rawSource, { origin })]) {
    const node = new MapNode("m", { flowId: "f" });
    let caught: unknown = null;
    try {
      node.bindReserved(new Map([["over", source]]), pool);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(BindingError);
    expect((caught as BindingError).message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
    expect((caught as BindingError).origin).toBe(source instanceof Binding ? origin : null);
  }
});

test("case without predicate is a setup fault", async () => {
  // A compiled CASE branch must have its validated predicate.
  const node = new CaseNode("case", [new Case("branch")]);

  await expect(node.run({})).rejects.toThrow(RangeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' has no `when` expression");
});

test("case with non string predicate is a setup fault", async () => {
  // Malformed code-built CASE state must not look like node execution.
  const malformed = new Case("branch", { when: 42 as unknown as string });
  const node = new CaseNode("case", [malformed]);

  await expect(node.run({})).rejects.toThrow(TypeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' `when` must be a string");
});

test("case rejects a non origin predicate carrier", () => {
  // Code-built CASE provenance must stay source-neutral.
  const make = () => new Case("branch", { when: "true", origin: "flow.yaml:12" as unknown as OriginRef });

  expect(make).toThrow(TypeError);
  expect(make).toThrow("CASE predicate origin");
});

test("success returns output without observer events", async () => {
  expect(await drive(new FuncNode("n", () => ({ output: "ok" })))).toEqual(new Output({ output: "ok" }));
});

test("baked commit target remains engine owned", async () => {
  const node = new ResultNode("n", new Output("v"));
  node.commitAs = "spawner";

  const result = await drive(node);

  expect(result).toEqual(new Output("v"));
  expect((result as Output).commitAs).toBeNull();
});

test("node chosen commit target remains on output", async () => {
  const node = new ResultNode("n", new Output("v", { commitAs: "chosen" }));
  node.commitAs = "spawner";

  expect(await drive(node)).toEqual(new Output("v", { commitAs: "chosen" }));
});

test("binding failure returns diagnostic evaluation failure", async () => {
  const node = new FuncNode("n", (inputs) => inputs, { origin: new OriginRef("node:n") });
  node.params = [new Parameter("x", { required: true })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("BindingError");
});

test("binding carrier resolves at the node read boundary", async () => {
  // Ordinary parameter binding evaluates the carrier source without changing it.
  const pool = new VariablePool();
  pool.set(START_ID, { value: "resolved" });
  const origin = new OriginRef("binding");
  const wrapped = new Binding("${input.value}", { origin });
  const node = new FuncNode("n", (inputs) => inputs.x, { reads: { x: wrapped } });

  expect(await drive(node, { pool })).toEqual(new Output("resolved"));
  expect(node.reads.x).toBe(wrapped);
  expect(wrapped.origin).toBe(origin);
});

test("binding carrier unwraps exactly one layer", async () => {
  // A nested carrier stays an opaque payload after the outer layer, never a second source.
  const pool = new VariablePool();
  pool.set(START_ID, { value: "must-not-resolve" });
  const inner = new Binding("${input.value}", { origin: new OriginRef("inner") });
  const outer = new Binding(inner, { origin: new OriginRef("outer") });
  const node = new FuncNode("n", (inputs) => inputs.x, { reads: { x: outer } });

  const failure = await driveFailure(node, { pool });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.message).not.toContain("must-not-resolve");
});

test("routing returns route without output value", async () => {
  const result = await drive(new RouteNode("n", new Route("case_a")));

  expect(result).toEqual(new Route("case_a"));
  expect("value" in (result as object)).toBe(false);
});

test("non spawner grow returns clear evaluation failure", async () => {
  const node = new ResultNode("n", new Grow({}), { origin: new OriginRef("node:n") });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("contract.node_result");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("Error");
  expect(failure.message).toBe(
    "node 'n' (kind code) returned a Grow but is not a spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph",
  );
});

test("bad return type returns clear evaluation failure", async () => {
  const failure = await driveFailure(new ResultNode("n", { not: "a NodeResult" }));

  expect(failure.code).toBe("contract.node_result");
  expect(failure.trace).toEqual([]);
  expect(failure.causeType).toBe("Error");
  expect(failure.message).toBe("node 'n' run() returned Object, not a NodeResult");
});

for (const [label, make, message] of [
  ["map", () => new MapNode("map", { flowId: "child", child: {} }), "MAP node 'map' has no `over` wiring"],
  ["wait", () => new WaitNode("wait"), "WAIT node 'wait' has no `until` wiring"],
] as const) {
  test(`missing reserved wiring is an ir fault (${label})`, async () => {
    // A validated compiled graph has every trait-declared reserved key.
    expect(await rejected(drive(make()))).toEqual(new Error(message));
  });
}

test("malformed parameter carrier is an ir fault", async () => {
  // Malformed compiled setup escapes instead of becoming a flow failure.
  const node = new FuncNode("n", (inputs) => inputs);
  node.params = { 0: new Parameter("x"), length: 1 } as unknown as Parameter[];

  expect(await rejected(drive(node))).toEqual(new TypeError("node params must be an exact list, got Object"));
});

for (const [label, make, field, value] of [
  ["map", () => new MapNode("map", { flowId: "child", child: {} }), "over", "not-a-list"],
  ["wait", () => new WaitNode("wait"), "until", "not-an-iso-date"],
] as const) {
  test(`reserved input rejection is an input contract failure (${label})`, async () => {
    // An expected reserved-value rejection keeps its authored binding origin.
    const node: Node = make();
    const origin = new OriginRef(`${node.id}:${field}`);

    const failure = await driveFailure(node, { flow: nodeFlow(node, { [field]: new Binding(value, { origin }) }) });

    expect(failure.code).toBe("contract.input");
    expect(failure.trace).toEqual([origin]);
  });
}

test("map over not a list message is byte identical through the evaluator", async () => {
  const pool = new VariablePool();
  pool.set(START_ID, { bad: "notalist" });
  const rawSource = "${input.bad}";
  for (const source of [rawSource, new Binding(rawSource, { origin: new OriginRef("map-over") })]) {
    const node = new MapNode("m", { flowId: "f" });
    const failure = await driveFailure(node, { pool, flow: nodeFlow(node, { over: source }) });

    expect(failure.message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
  }
});

/** Reserve key `k`, whose binding always fails, and record steps. */
class StepsReservedNode extends ResultNode {
  override readonly needsSteps: boolean = true;

  override bindReserved(): Record<string, unknown> {
    throw new BindingError("node 'n' reserved key failed", { failureCode: "contract.input" });
  }
}

test("a reserved binding failure after an earlier start recorded steps is a host fault", async () => {
  // The earlier start resolved the key from the same restored pool, so the flow or yian changed.
  const node = new StepsReservedNode("n", new Output("unused"));
  const steps = new NodeSteps(2, [{ step: 1n }], () => {});

  expect(await rejected(drive(node, { steps }))).toEqual(
    new RangeError(
      "node 'n' reserved key failed; an earlier start of this node resolved it, so resume the run with the " +
        "flow and yian version that started it",
    ),
  );
});

test("a reserved binding failure with no prior steps stays a node failure", async () => {
  const node = new StepsReservedNode("n", new Output("unused"));

  for (const steps of [null, new NodeSteps(1, [], () => {})]) {
    expect((await driveFailure(node, { steps })).code).toBe("contract.input");
  }
});

test("a reserved binding failure of a node that doesn't read its steps stays a node failure", async () => {
  // Only a node reading its steps saw the earlier start's resolution.
  class NoStepsReservedNode extends StepsReservedNode {
    override readonly needsSteps = false;
  }
  const steps = new NodeSteps(2, [{ step: 1n }], () => {});

  expect((await driveFailure(new NoStepsReservedNode("n", new Output("unused")), { steps })).code).toBe(
    "contract.input",
  );
});

describe("evalNode checks its arguments", () => {
  const node = new FuncNode("n", () => "unused");
  const pool = new VariablePool();

  for (const [label, options, error] of [
    ["capture_evidence", { captureEvidence: 1 }, new TypeError("capture_evidence must be a bool")],
    ["capture_bindings", { captureBindings: "yes" }, new TypeError("capture_bindings must be a bool")],
    ["emit_stream", { emitStream: "sink" }, new TypeError("emit_stream must be callable or null")],
    ["emit_event", { emitEvent: {} }, new TypeError("emit_event must be callable or null")],
    ["source_resolver", { sourceResolver: 1 }, new TypeError("source_resolver must be callable or null")],
    [
      "memory_context_opener",
      { memoryContextOpener: {} },
      new TypeError("memory_context_opener must be callable or null"),
    ],
    ["tools", { tools: new Map() }, new TypeError("tools must be a ToolSet or null")],
    ["skills", { skills: { s: {} } }, new TypeError("skills must be a map or null")],
    ["steps", { steps: { starts: 1, prior: [], record() {} } }, new TypeError("steps must be a NodeSteps or null")],
    [
      "memory_invocation",
      { memoryInvocation: { runId: "r", startSequence: 1 } },
      new TypeError("memory_invocation must be a ConversationInvocation or null"),
    ],
    [
      "effective_llm_by_static_path",
      { effectiveLlmByStaticPath: new Map() },
      new TypeError("effective_llm_by_static_path must be a record or null"),
    ],
    ["signal", { signal: new AbortController() }, new TypeError("signal must be an AbortSignal or null")],
    [
      "steering",
      { steering: { drain: () => [] } },
      new TypeError("steering must provide interruptSignal(nodeId) and drain(nodeId), or be null"),
    ],
  ] as const) {
    test(label, async () => {
      expect(await rejected(evalNode(node, null, pool, options as unknown as EvalNodeOptions))).toEqual(error);
    });
  }

  test("well-formed options are accepted", async () => {
    const options: EvalNodeOptions = {
      captureEvidence: false,
      captureBindings: false,
      emitStream: () => {},
      emitEvent: () => {},
      memoryInvocation: new ConversationInvocation("run", 1),
      skills: new Map(),
      tools: new ToolSet(),
      steps: new NodeSteps(1, [], () => {}),
      effectiveLlmByStaticPath: {},
      signal: new AbortController().signal,
      steering: { interruptSignal: () => new AbortController().signal, drain: () => [] },
    };

    expect((await evalNode(node, null, pool, options)).outcome).toEqual(new Output("unused"));
  });
});

/** Throw `error` from its body, after an `await` when `async`. */
class RaiseNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly error: unknown;
  readonly async: boolean;

  constructor(nodeId: string, error: unknown, options: { origin?: OriginRef | null; async?: boolean } = {}) {
    const { async = false, ...rest } = options;
    super(nodeId, rest);
    this.error = error;
    this.async = async;
  }

  override forward(): NodeResult | Promise<NodeResult> {
    if (!this.async) throw this.error;
    return Promise.resolve().then(() => {
      throw this.error;
    });
  }
}

test("node raise returns evaluation failure", async () => {
  const node = new RaiseNode("n", new Error("boom"), { origin: new OriginRef("node:n") });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("node.execute");
  expect(failure.trace).toEqual([node.origin!]);
  expect([failure.message, failure.causeType]).toEqual(["boom", "Error"]);
});

/** Change its input record, to test the template's post-assert snapshot. */
class MutatingNode extends Node {
  readonly kind = NodeKind.CODE;

  constructor(nodeId: string) {
    super(nodeId);
    this.params = [new Parameter("x")];
  }

  override forward(inputs: Record<string, unknown>): NodeResult {
    inputs.x = 999n;
    return new Output("done");
  }
}

test("post assert isolated from node input mutation", async () => {
  const pool = new VariablePool();
  pool.set(START_ID, { x: 1n });
  const node = new MutatingNode("n");
  node.postAsserts = [new Assertion("x == 1", "x == 1")];

  expect(await drive(node, { pool, flow: nodeFlow(node, { x: "${input.x}" }) })).toEqual(new Output("done"));
});

test("wrapper assertion failure uses assertion stage and origin", async () => {
  // Only the template's own authored assertion gets `assertion.failed`.
  const assertionOrigin = new OriginRef("assert:n:0");
  const node = new FuncNode("n", () => "never", { origin: new OriginRef("node:n") });
  node.preAsserts = [new Assertion("1 == 2", "must hold", { origin: assertionOrigin })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("assertion.failed");
  expect(failure.trace).toEqual([assertionOrigin]);
  expect(failure.causeType).toBe("NodeAssertFailed");
  expect(failure.traceback).toStartWith("NodeAssertFailed: node 'n' pre-assert failed: must hold\n");
  expect(failure.traceback).not.toContain("AssertionStageError");
});

test("wrapper post assertion uses assertion stage and origin", async () => {
  // Post-asserts keep the same template-owned classification as pre-asserts.
  const assertionOrigin = new OriginRef("assert:n:post:0");
  const node = new FuncNode("n", () => "value", { origin: new OriginRef("node:n") });
  node.postAsserts = [new Assertion("output == 'other'", "output must match", { origin: assertionOrigin })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("assertion.failed");
  expect(failure.trace).toEqual([assertionOrigin]);
  expect(failure.causeType).toBe("NodeAssertFailed");
});

test("body raised assertion exception remains node execution", async () => {
  // The classification follows the body stage, not the error's class.
  const node = new RaiseNode("n", new NodeAssertFailed("body chose this exception type"), {
    origin: new OriginRef("node:n"),
  });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("node.execute");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("NodeAssertFailed");
});

for (const async of [false, true]) {
  test(`body raised binding exception remains node execution (${async ? "async" : "body"})`, async () => {
    // Only evaluator-owned binding operations may keep the binder's classification.
    const error = new BindingError("body chose this exception type", {
      failureCode: "contract.input",
      origin: new OriginRef("spoofed-binding-origin"),
    });
    const node = new RaiseNode("n", error, { origin: new OriginRef("node:n"), async });

    const failure = await driveFailure(node);

    expect(failure.code).toBe("node.execute");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe("BindingError");
  });
}

/** Throw from the body while naming a finer operation origin. */
class OperationOriginNode extends RaiseNode {
  readonly operationOrigin: OriginRef | null;

  constructor(nodeId: string, operationOrigin: OriginRef | null) {
    super(nodeId, new Error("operation failed"), { origin: new OriginRef(`node:${nodeId}`) });
    this.operationOrigin = operationOrigin;
  }

  override exceptionOrigin(): OriginRef | null {
    return this.operationOrigin;
  }
}

for (const [operationOrigin, expectedKey] of [
  [new OriginRef("operation:n"), "operation:n"],
  [null, "node:n"],
] as const) {
  test(`node execution prefers operation origin then node fallback (${expectedKey})`, async () => {
    // A body failure appends the selected local origin after the outer ancestry.
    const outer = new OriginRef("outer:call");
    const node = new OperationOriginNode("n", operationOrigin);
    node.originContext = [outer];

    expect((await driveFailure(node)).trace).toEqual([outer, new OriginRef(expectedKey)]);
  });
}

/** Return or throw a configured body outcome while classifying every body error as `provider.auth`. */
class ExceptionCodeNode extends ResultNode {
  constructor(nodeId: string, result: unknown) {
    super(nodeId, result, { origin: new OriginRef(`node:${nodeId}`) });
  }

  override forward(): NodeResult {
    if (this.result instanceof Error) throw this.result;
    return this.result as NodeResult;
  }

  override exceptionCode(): "provider.auth" {
    return "provider.auth";
  }
}

/** Bypass the base template with an `async` `run` that rejects with `error`. */
class OverrideRaisesNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly error: unknown;

  constructor(nodeId: string, error: unknown) {
    super(nodeId, { origin: new OriginRef(`node:${nodeId}`) });
    this.error = error;
  }

  override forward(): NodeResult {
    return new Output("unused");
  }

  override async run(): Promise<NodeResult> {
    throw this.error;
  }
}

/** Bypass the base template while classifying every body error as `provider.auth`. */
class OverrideRaisesWithCode extends OverrideRaisesNode {
  override exceptionCode(): "provider.auth" {
    return "provider.auth";
  }
}

for (const [label, make] of [
  ["body", () => new ExceptionCodeNode("n", new Error("unauthorized"))],
  ["run override", () => new OverrideRaisesWithCode("n", new Error("unauthorized"))],
] as const) {
  test(`node execution uses the node exception code (${label})`, async () => {
    // A body error takes the code the node's hook selects.
    const node: Node = make();

    const failure = await driveFailure(node);

    expect(failure.code).toBe("provider.auth");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe("Error");
  });
}

test("base exception code is node execution", () => {
  // A node without a finer classification reports `node.execute`.
  expect(new ResultNode("n", null).exceptionCode(new Error("boom"))).toBe("node.execute");
});

for (const [label, result, configure, expectedCode] of [
  [
    "unbound parameter",
    new Output("unused"),
    (node: Node) => {
      node.params = [new Parameter("x", { required: true })];
    },
    "binding.evaluate",
  ],
  [
    "failing pre-assert",
    new Output("unused"),
    (node: Node) => {
      node.preAsserts = [new Assertion("1 == 2", "must hold")];
    },
    "assertion.failed",
  ],
  ["bad result", { not: "a NodeResult" }, () => {}, "contract.node_result"],
] as const) {
  test(`exception code is asked only for the body (${label})`, async () => {
    // The binding, assertion and result-contract stages keep their own codes.
    const node = new ExceptionCodeNode("n", result);
    configure(node);

    expect((await driveFailure(node)).code).toBe(expectedCode);
  });
}

for (const [label, error, expectedCauseType] of [
  ["ordinary", new Error("ordinary override failure"), "Error"],
  ["binding error", new BindingError("spoofed", { failureCode: "binding.evaluate" }), "BindingError"],
  [
    "assertion error",
    new NodeAssertFailed("spoofed", { assertion: new Assertion("true", "true", { origin: new OriginRef("spoofed") }) }),
    "NodeAssertFailed",
  ],
  [
    "binding stage marker",
    new BindingStageError(new BindingError("wrapped", { failureCode: "binding.evaluate" })),
    "BindingError",
  ],
  [
    "assertion stage marker",
    new AssertionStageError(new NodeAssertFailed("wrapped", { assertion: new Assertion("true", "true") })),
    "NodeAssertFailed",
  ],
] as const) {
  test(`run override exception is node execution (${label})`, async () => {
    // A custom run is one undivided node-body stage.
    const node = new OverrideRaisesNode("n", error);

    const failure = await driveFailure(node);

    expect(failure.code).toBe("node.execute");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe(expectedCauseType);
    expect(failure.traceback).not.toContain("BindingStageError");
    expect(failure.traceback).not.toContain("AssertionStageError");
  });
}

/** A node class that defines no `forward`. */
class NoForwardNode extends Node {
  readonly kind = NodeKind.CODE;
}

test("missing node implementation is a setup fault", async () => {
  // The base placeholder body is malformed executable setup.
  expect(await rejected(drive(new NoForwardNode("n")))).toEqual(
    new Error("node 'n' (NoForwardNode) defines no forward"),
  );
});

test("malformed assertion carrier is an ir fault", async () => {
  // An unexpected template-state fault is outside expected flow failure.
  const node = new FuncNode("n", () => "never");
  node.preAsserts = [{} as Assertion];

  expect(await rejected(drive(node))).toBeInstanceOf(TypeError);
});

/** Bind one record per item through `caps.bindItem`, as MAP does, and return them. */
class PerItemNode extends Node {
  readonly kind = NodeKind.MAP;
  override readonly bindsPerItem = true;
  readonly items: readonly Value[];

  constructor(nodeId: string, items: readonly Value[]) {
    super(nodeId);
    this.items = items;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    return new Output(this.items.map((item, index) => caps.bindItem!(item, index)));
  }
}

test("map deferred item binding keeps input contract stage", async () => {
  // The per-item binder capability doesn't become a node-body failure.
  const contractOrigin = new OriginRef("map:x:contract");
  const node = new PerItemNode("m", ["bad"]);
  node.params = [
    new Parameter("x", {
      contract: new ValueContract(Type.scalar(ValueKind.INTEGER), new ContractOrigins({ default: contractOrigin })),
    }),
  ];

  const failure = await driveFailure(node, {
    flow: nodeFlow(node, { x: new Binding("${item}", { origin: new OriginRef("map:x:binding") }) }),
  });

  expect(failure.code).toBe("contract.input");
  expect(failure.trace).toEqual([contractOrigin]);
});

test("map deferred expression keeps binding stage", async () => {
  // A per-item expression failure stays evaluator-owned binding work.
  const bindingOrigin = new OriginRef("map:x:binding");
  const node = new PerItemNode("m", [{}]);
  node.params = [new Parameter("x")];

  const failure = await driveFailure(node, {
    flow: nodeFlow(node, { x: new Binding('${item.missing :? "x is required"}', { origin: bindingOrigin }) }),
  });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([bindingOrigin]);
});

test("map deferred unexpected binder fault escapes", async () => {
  // Deferred binding keeps the ordinary binder's host and setup boundary.
  const expected = new Error("item read failed");
  const item = Object.defineProperty({}, "x", {
    enumerable: true,
    get() {
      throw expected;
    },
  }) as Value;
  const node = new PerItemNode("m", [item]);
  node.params = [new Parameter("x")];

  expect(await rejected(drive(node, { flow: nodeFlow(node, { x: "${item}" }) }))).toBe(expected);
});

test("per item wrapper default keeps binding stage", async () => {
  // The template's per-item default backstop keeps the default's origin.
  const defaultOrigin = new OriginRef("map:x:default");
  const node = new MapNode("m", { flowId: "child", child: {} });
  node.params = [
    new Parameter("x", {
      contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
      hasDefault: true,
      default: "bad",
      defaultOrigin,
    }),
  ];

  const failure = await driveFailure(node, { flow: nodeFlow(node, { over: [] }) });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([defaultOrigin]);
});
