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
 * `test_reserved_input_rejection_is_an_input_contract_failure` (2 rows) and
 * `test_case_rejects_a_non_origin_predicate_carrier`.
 *
 * Ported (adapted): `test_map_over_accepts_a_binding_carrier`,
 * `test_wait_until_accepts_a_binding_carrier`,
 * `test_unbaked_composite_is_a_setup_fault`,
 * `test_case_without_predicate_is_a_setup_fault`,
 * `test_case_with_non_string_predicate_is_a_setup_fault`,
 * `test_map_over_not_a_list_message_is_byte_identical` (on the node and through the evaluator),
 * `test_missing_reserved_wiring_is_an_ir_fault` (2 rows; the node's plain `Error`, where Python
 * raises `KeyError`) and `test_malformed_parameter_carrier_is_an_ir_fault` (`params` that isn't
 * an array, where Python uses a tuple).
 *
 * Changed: `drive` evaluates the node in a one-node `CompiledFlow` whose wiring is a `FuncNode`'s
 * `reads` or the given sources, as Python's stub flow derives it. Python's `RuntimeError` is a
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
 * reserved key fails to bind after an earlier start recorded steps rejects with a `RangeError`.
 *
 * Later (T4-9): `test_case_predicate_failure_selects_the_active_ordered_origin` and
 * `test_source_free_case_predicate_failure_falls_back_to_the_node_origin`.
 *
 * Later (T4-10): the run-result half of `test_map_over_accepts_a_binding_carrier` (the `Grow`
 * through the evaluator). Later (T4-11): the run-result half of
 * `test_wait_until_accepts_a_binding_carrier` (the `Output` through the evaluator).
 *
 * Later (T4-6): the stage classification rows.
 */

import { describe, expect, test } from "bun:test";

import { CompiledFlow, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ConversationInvocation } from "../../src/memory/contracts.ts";
import { Grow, Node, NodeKind, type NodeResult, Output, Route } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { Assertion, Binding, NodeSteps, Parameter } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import {
  type EvalNodeOptions,
  EvaluationFailure,
  evalNode,
  type WorkerOutcome,
} from "../../src/runtime/eval_node.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FuncNode } from "./run_state_fixtures.ts";

/**
 * Evaluate `node` in a one-node flow wired from `wiring` (a `FuncNode`'s `reads` by default) and
 * return its raw outcome.
 */
async function drive(
  node: Node,
  options: { pool?: VariablePool; wiring?: Readonly<Record<string, unknown>> } & EvalNodeOptions = {},
): Promise<WorkerOutcome> {
  const { pool = new VariablePool(), wiring = node instanceof FuncNode ? node.reads : {}, ...rest } = options;
  const flow = CompiledFlow.fromParts(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map(Object.entries(wiring))]]),
  });
  return (await evalNode(node, flow, pool, rest)).outcome;
}

/** Return the failure `drive` resolves to, or fail when it resolves to anything else. */
async function driveFailure(
  node: Node,
  options: Parameters<typeof drive>[1] = {},
): Promise<EvaluationFailure["failure"]> {
  const outcome = await drive(node, options);
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure;
}

/** Return the value `promise` rejects with, or fail when it resolves. */
async function rejected(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
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

    const failure = await driveFailure(node, { wiring: { [field]: new Binding(value, { origin }) } });

    expect(failure.code).toBe("contract.input");
    expect(failure.trace).toEqual([origin]);
  });
}

test("map over not a list message is byte identical through the evaluator", async () => {
  const pool = new VariablePool();
  pool.set(START_ID, { bad: "notalist" });
  const rawSource = "${input.bad}";
  for (const source of [rawSource, new Binding(rawSource, { origin: new OriginRef("map-over") })]) {
    const failure = await driveFailure(new MapNode("m", { flowId: "f" }), { pool, wiring: { over: source } });

    expect(failure.message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
  }
});

/** Reserve key `k`, whose binding always fails, and record steps. */
class StepsReservedNode extends ResultNode {
  override readonly needsSteps = true;

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
