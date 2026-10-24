/**
 * Default and structural backward on nodes: a port of `tests/engine/test_backward_structural.py`.
 * Ported: `test_behavioral_backward_stops_with_empty_channels_and_static_capture` (Python's bare
 * `Node` is a minimal subclass), `test_case_backward_inventory_excludes_generated_control_inputs`,
 * `test_list_end_keeps_whole_list_text_unchanged_and_records_each_output_position`,
 * `test_failed_boundaries_forward_text_only_when_inputs_and_layout_were_captured`,
 * `test_successful_null_output_is_a_bare_contributor_but_zero_output_has_none`,
 * `test_stopped_loop_failure_does_not_route_an_unaccepted_carried_output`.
 *
 * Changed: `test_structural_output_paths_must_exist_in_the_available_accepted_output` builds an
 * END in Python; here a fake whose `backward` calls `projectStructuralFeedback`, as END, START and
 * LOOP do. `test_start_projects_named_delivered_inputs_including_defaults_but_not_absence` binds
 * the inputs with `applyDefaults` and builds the output with START's `forward` instead of
 * `eval_node`. `test_end_backward_uses_captured_layout_after_live_node_fields_change` changes END's
 * public `names`, `mode` and `n` (Python's private `_names`, `_mode`, `_n`).
 * `test_end_projects_its_zero_one_and_many_output_layouts` appends an END's extra assertion param
 * by replacing `params`, which TS types as read-only.
 * `test_stopping_loop_projects_all_carried_fields_without_reevaluating_predicate` swaps the
 * predicate for one that doesn't parse after `forward` stops, so backward would throw if it
 * evaluated it (Python's monkeypatch). The params are emptied by assigning `[]`.
 *
 * Ported (through `evalNode`): `test_input_inventory_capture_preserves_canonical_parameter_setup_faults`
 * (Python's `object()` entry is `{}`),
 * `test_early_binding_failure_keeps_inventories_without_usable_inputs_or_fabricated_layout`,
 * `test_static_capture_exception_escapes_unchanged_and_capture_off_calls_no_hook` (the hook throws
 * a `RangeError` for Python's `ValueError`),
 * `test_static_layout_is_captured_before_a_preassert_failure_after_defaults` (it empties END's
 * public `names` for Python's `_names.clear()`),
 * `test_custom_run_uses_its_existing_pre_run_capture_boundary` and
 * `test_agent_capture_keeps_variable_inventory_separate_from_delivered_inputs` (a `ScriptedModel`
 * from `models` replies the rendered prompt, for Python's patched `run_agent`).
 *
 * Changed: TS records no node data, so an evaluation calls no static capture hook and these rows
 * read the node data learning rebuilds (`rebuildNodeData`, the AGENT's `renderCapture`) instead of
 * a capture: the static capture faults are the rebuild's errors, not host faults before the body;
 * `test_static_backward_capture_and_names_are_detached_for_the_actual_invocation` checks the names
 * and scope only, and that the rebuild reads the node as it is then.
 * `test_static_capture_serialization_faults_are_host_faults_before_the_body` refuses NaN for
 * Python's infinity, which is a value here, and a class instance for Python's `object()`.
 */

import { expect, test } from "bun:test";

import { Failure } from "../../src/diagnostics.ts";
import type { FeedbackValue } from "../../src/expr/index.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import {
  BackwardContext,
  BackwardResult,
  Node,
  NodeKind,
  type NodeResult,
  Output,
  projectStructuralFeedback,
  validateBackwardResult,
} from "../../src/nodes/base.ts";
import { applyDefaults } from "../../src/nodes/binding.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { Assertion, ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { EvaluationFailure, evalNode, rebuildNodeData } from "../../src/runtime/eval_node.ts";
import { captureBindings } from "../../src/runtime/trace.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import { NO_ENV, rejected } from "./run_state_fixtures.ts";

/** A backward context over the node's declared inputs and captured layout. */
function contextFor(node: Node, inputs: Record<string, unknown>, output: unknown): BackwardContext {
  return new BackwardContext(inputs, {
    inputNames: node.backwardInputNames(),
    output,
    nodeData: node.captureBackwardContext(),
  });
}

/** A minimal concrete node with the base backward hooks. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A class instance: not JSON material. */
class Opaque {}

/**
 * A node with declared inputs, variables and static material; its body changes all three and its
 * scope, so a capture taken before the body keeps what the invocation started with.
 */
class CaptureNode extends Node {
  readonly kind = NodeKind.CODE;
  variables: Parameter[] = [new Parameter("instruction")];
  material: unknown = { settings: { active: true } };
  captures = 0;
  ran = false;

  constructor() {
    super("work");
    this.params = [new Parameter("topic"), new Parameter("limit", { hasDefault: true, default: 4n })];
  }

  override declaredVariables(): readonly Parameter[] {
    return [...this.variables];
  }

  override captureBackwardContext(): Record<string, unknown> {
    this.captures += 1;
    return this.material as Record<string, unknown>;
  }

  override forward(inputs: Record<string, unknown>): NodeResult {
    this.ran = true;
    this.params = [new Parameter("changed_input"), this.params![1]!];
    this.variables[0] = new Parameter("changed_variable");
    ((this.material as Record<string, unknown>)["settings"] as Record<string, unknown>)["active"] = false;
    this.executionScope = null;
    return new Output(inputs["topic"]);
  }
}

/** A structural node: its backward projects feedback through the captured output paths. */
class StructuralNode extends Node {
  readonly kind = NodeKind.END;

  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    return projectStructuralFeedback(context, feedback);
  }
}

test("behavioral backward stops with empty channels and static capture", () => {
  const node = new BareNode("ordinary");
  node.params = [new Parameter("topic")];
  const context = new BackwardContext({ topic: "reefs" }, { inputNames: ["topic"], output: "answer" });

  expect(node.backward(context, ["Be more precise."])).toEqual(new BackwardResult());
  expect(node.captureBackwardContext()).toEqual({});
  expect(node.backwardInputNames()).toEqual(["topic"]);
  expect(new BareNode("empty").backwardInputNames()).toEqual([]);
});

test("case backward inventory excludes generated control inputs", () => {
  const node = new CaseNode("choose", []);
  node.params = ["topic", "__r0", "__on", "__r12", "__remarks"].map((name) => new Parameter(name));

  expect(node.backwardInputNames()).toEqual(["topic", "__remarks"]);
});

test.each([
  ["a record", { topic: new Parameter("topic") }],
  ["a non-Parameter entry", [{}]],
])("input inventory capture preserves canonical parameter setup faults: %s", async (_label, params) => {
  const node = new BareNode("invalid");
  node.params = params as unknown as Parameter[];
  for (const captureBindings of [false, true]) {
    const error = await rejected(evalNode(node, null, new VariablePool(), { captureBindings }));
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toMatch(/params/);
  }
});

test.each([
  ["a missing field", { actual: 1n }, ["missing"]],
  ["an index into a record", { actual: 1n }, [0n]],
  ["an index past the end", [1n], [1n]],
])("structural output paths must exist in the available accepted output: %s", (_label, output, path) => {
  const node = new StructuralNode("end");
  const context = new BackwardContext(
    { result: 1n },
    { inputNames: ["result"], output, nodeData: { input_output_paths: { result: path } } },
  );

  expect(() => node.backward(context, ["Review."])).toThrow(RangeError);
  expect(() => node.backward(context, ["Review."])).toThrow(/output path does not exist/);
});

test("start projects named delivered inputs including defaults but not absence", () => {
  const node = new StartNode("start", {
    params: [
      new Parameter("topic"),
      new Parameter("limit", { hasDefault: true, default: 4n }),
      new Parameter("omitted"),
    ],
  });
  const inputs = applyDefaults(node.params!, { topic: "reefs" });
  const context = contextFor(node, inputs, node.forward(inputs).value);

  const result = node.backward(context, ["Review the request.", { limit: "Use fewer." }, null]);

  expect(inputs).toEqual({ topic: "reefs", limit: 4n });
  expect(result.inputFeedback).toEqual({
    topic: ["Review the request."],
    limit: ["Review the request.", "Use fewer."],
  });
  expect(context.nodeData).toEqual({
    input_output_paths: { topic: ["topic"], limit: ["limit"], omitted: ["omitted"] },
  });
  validateBackwardResult(result, context);
});

test.each([
  ["value, no output", false, [], {}, ["No return value."], {}, {}],
  [
    "value, one output",
    false,
    ["result"],
    { result: { items: ["a", "b"] } },
    [{ items: [null, "Review."] }],
    { result: [{ items: [null, "Review."] }] },
    { result: [] },
  ],
  ["record, no field", true, [], {}, ["No fields."], {}, {}],
  [
    "record, one field",
    true,
    ["only"],
    { only: { items: ["a", "b"] } },
    ["Review it.", { only: { items: [null, "Review."] } }],
    { only: ["Review it.", { items: [null, "Review."] }] },
    { only: ["only"] },
  ],
  [
    "record, two fields",
    true,
    ["left", "right"],
    { left: 1n, right: 2n },
    ["Review both.", { right: "Check." }],
    { left: ["Review both."], right: ["Review both.", "Check."] },
    { left: ["left"], right: ["right"] },
  ],
] as [
  string,
  boolean,
  string[],
  Record<string, unknown>,
  FeedbackValue[],
  Record<string, FeedbackValue[]>,
  Record<string, unknown[]>,
][])(
  "end projects its zero one and many output layouts: %s",
  (_label, record, names, inputs, feedback, expected, paths) => {
    const node = record
      ? EndNode.record("end", { outputNames: names })
      : EndNode.value("end", { outputName: names[0] ?? null });
    node.params = [...node.params!, new Parameter("assertion_input")];
    const delivered = { ...inputs, assertion_input: "diagnostic-only" };
    const context = contextFor(node, delivered, node.forward(delivered).value);

    const result = node.backward(context, feedback);

    expect(context.inputNames).toEqual(names);
    expect(context.nodeData).toEqual({ input_output_paths: paths });
    expect(result.inputFeedback).toEqual(expected);
    validateBackwardResult(result, context);
  },
);

test.each([
  ["empty", []],
  ["two equal items", [["same"], ["same"]]],
] as [string, string[][]][])(
  "list end keeps whole list text unchanged and records each output position: %s",
  (_label, items) => {
    const node = EndNode.list("collector", { n: items.length });
    const inputs = Object.fromEntries(items.map((item, index) => [`e${index}`, item]));
    const context = contextFor(node, inputs, node.forward(inputs).value);
    const text = "The complete list repeats itself.";

    const result = node.backward(context, [text, items.map(() => null)]);

    expect(result.inputFeedback).toEqual(Object.fromEntries(Object.keys(inputs).map((name) => [name, [text]])));
    expect(context.nodeData).toEqual({
      input_output_paths: Object.fromEntries(items.map((_item, index) => [`e${index}`, [BigInt(index)]])),
    });
    validateBackwardResult(result, context);
    if (items.length > 0) {
      const positional = node.backward(context, [[null, ["Review the second item."]]]);
      expect(positional.inputFeedback).toEqual({ e1: [["Review the second item."]] });
      expect(() => node.backward(context, [["too short"]])).toThrow(RangeError);
      expect(() => node.backward(context, [["too short"]])).toThrow(/exactly 2 items/);
    }
  },
);

test("end backward uses captured layout after live node fields change", () => {
  const node = EndNode.value("end", { outputName: "result" });
  const inputs = { result: { nested: "answer" } };
  const context = contextFor(node, inputs, node.forward(inputs).value);
  node.names = ["changed", "fields"];
  node.mode = "list";
  node.n = 0;
  node.params = [];

  const result = node.backward(context, [{ nested: "Explain." }]);

  expect(result.inputFeedback).toEqual({ result: [{ nested: "Explain." }] });
  expect(context.nodeData).toEqual({ input_output_paths: { result: [] } });
});

test("stopping loop projects all carried fields without reevaluating predicate", () => {
  const node = new LoopNode("loop", {
    flowId: "body",
    predicate: "false",
    maxIters: 2,
    childInputs: [new Parameter("value")],
  });
  node.params = [new Parameter("value"), new Parameter("carried_only")];
  const inputs = { value: 7n, carried_only: ["kept"] };
  const context = contextFor(node, inputs, (node.forward(inputs) as Output).value);

  // A predicate that doesn't parse: evaluating it in backward would throw.
  node.predicate = "${unparseable";
  node.params = [];
  const result = node.backward(context, [{ carried_only: ["Review."] }]);
  expect(result.inputFeedback).toEqual({ carried_only: [["Review."]] });
  validateBackwardResult(result, context);
  const continuing = new BackwardContext(inputs, { inputNames: context.inputNames, nodeData: context.nodeData });
  expect(node.backward(continuing, ["Growth is handled by its connections."])).toEqual(new BackwardResult());
});

test.each([
  ["START", new StartNode("start", { params: [new Parameter("value")] })],
  ["list END", EndNode.list("end", { n: 1 })],
] as [string, StartNode | EndNode][])(
  "failed boundaries forward text only when inputs and layout were captured: %s",
  (_label, node) => {
    const [name] = node.backwardInputNames() as [string];
    const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: node.id });
    const context = new BackwardContext(
      { [name]: ["delivered"] },
      { inputNames: [name], failure, nodeData: node.captureBackwardContext() },
    );
    const result = node.backward(context, ["Review the supplied value."]);
    expect(result.inputFeedback).toEqual({ [name]: ["Review the supplied value."] });
    expect(context.output).toBeNull();
    expect(() => node.backward(context, [{ [name]: "No accepted output shape." }])).toThrow(RangeError);
    const unavailable = new BackwardContext(null, { inputNames: [name], failure });
    expect(node.backward(unavailable, ["No usable binding."])).toEqual(new BackwardResult());
  },
);

test("successful null output is a bare contributor but zero output has none", () => {
  const one = EndNode.value("one", { outputName: "result" });
  const zero = EndNode.value("zero", { outputName: null });
  expect(one.backward(contextFor(one, { result: null }, null), ["Review null."]).inputFeedback).toEqual({
    result: ["Review null."],
  });
  expect(zero.backward(contextFor(zero, {}, null), ["Review null."])).toEqual(new BackwardResult());
});

test("stopped loop failure does not route an unaccepted carried output", () => {
  const node = new LoopNode("loop", { flowId: "body", predicate: "false", maxIters: 2 });
  node.params = [new Parameter("value")];
  const context = new BackwardContext(
    { value: 1n },
    {
      inputNames: ["value"],
      nodeData: node.captureBackwardContext(),
      failure: new Failure({ code: "assertion.failed", message: "Stopped output was rejected.", nodeId: node.id }),
    },
  );
  expect(node.backward(context, ["Review the loop."])).toEqual(new BackwardResult());
});

test("backward names are detached for the actual invocation, and no static material is captured", async () => {
  const node = new CaptureNode();
  const scope = new ExecutionScope(["authored"]);
  node.executionScope = scope;
  const evaluated = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { topic: "reefs" },
    captureBindings: true,
  });
  const snapshot = evaluated.backwardCapture!;
  const capture = captureBindings(evaluated.bindings!, snapshot);

  expect(node.captures).toBe(0);
  expect(node.ran).toBe(true);
  expect(snapshot.inputNames).toEqual(["topic", "limit"]);
  expect(snapshot.variableNames).toEqual(["instruction"]);
  expect(capture.inputNames).toEqual(snapshot.inputNames);
  expect(capture.variableNames).toEqual(snapshot.variableNames);
  expect(capture.scope).toEqual(scope);
  expect(node.executionScope).toBeNull();
  expect(evaluated.bindings!.boundInputs).toEqual({ topic: "reefs", limit: 4n });
  // Learning rebuilds the node data from the node as it is then.
  expect(rebuildNodeData(node, evaluated.bindings!.boundInputs, {})).toEqual({ settings: { active: false } });
});

test("early binding failure keeps inventories without usable inputs or fabricated layout", async () => {
  const node = new CaptureNode();
  node.params = [new Parameter("topic", { required: true }), node.params![1]!];
  const evaluated = await evalNode(node, null, new VariablePool(), { captureBindings: true });
  const capture = captureBindings(evaluated.bindings!, evaluated.backwardCapture!);

  expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
  expect(evaluated.bindings!.boundInputs).toBeNull();
  expect(capture.inputNames).toEqual(["topic", "limit"]);
  expect(capture.variableNames).toEqual(["instruction"]);
  expect(rebuildNodeData(node, evaluated.bindings!.boundInputs, {})).toEqual({});
  expect(node.captures).toBe(0);
  expect(node.ran).toBe(false);
});

const unserializableMaterials: [string, unknown][] = [
  ["a class instance", new Opaque()],
  ["a record holding a class instance", { bad: new Opaque() }],
  ["NaN", { bad: NaN }],
  ["a lone surrogate", { bad: "\ud800" }],
];

test.each(unserializableMaterials)(
  "static material that isn't JSON fails the node data rebuild: %s",
  (_label, material) => {
    const node = new CaptureNode();
    node.material = material;
    let error: unknown = null;
    try {
      rebuildNodeData(node, { topic: "reefs", limit: 4n }, {});
    } catch (caught) {
      error = caught;
    }
    expect(error instanceof TypeError || error instanceof RangeError).toBe(true);
  },
);

test("a static capture exception escapes the node data rebuild unchanged, and evaluation calls no hook", async () => {
  const expected = new RangeError("capture failed");
  const node = new CaptureNode();
  node.captureBackwardContext = () => {
    throw expected;
  };
  expect(() => rebuildNodeData(node, { topic: "reefs", limit: 4n }, {})).toThrow(expected);
  const retained = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { topic: "reefs" },
    captureBindings: true,
  });
  expect(node.ran).toBe(true);
  expect((retained.outcome as Output).value).toBe("reefs");
  expect(retained.bindings!.boundInputs).toEqual({ topic: "reefs", limit: 4n });
});

test("static layout is rebuilt for a preassert failure after defaults", async () => {
  const node = EndNode.value("end", { outputName: "result" });
  node.params = [new Parameter("result", { hasDefault: true, default: { items: [1n] } })];
  node.preAsserts = [new Assertion("false", "reject before body")];
  const evaluated = await evalNode(node, null, new VariablePool(), { captureBindings: true });
  const capture = captureBindings(evaluated.bindings!, evaluated.backwardCapture!);

  expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = evaluated.outcome as EvaluationFailure;
  expect(failure.code).toBe("assertion.failed");
  expect(evaluated.bindings!.boundInputs).toEqual({ result: { items: [1n] } });
  const nodeData = rebuildNodeData(node, evaluated.bindings!.boundInputs, {});
  expect(nodeData).toEqual({ input_output_paths: { result: [] } });
  node.names = [];
  const context = new BackwardContext(evaluated.bindings!.boundInputs, {
    inputNames: capture.inputNames,
    nodeData,
    failure,
  });
  expect(node.backward(context, ["Review the default."]).inputFeedback).toEqual({
    result: ["Review the default."],
  });
});

test("custom run captures its inputs at its existing pre-run boundary", async () => {
  class CustomRun extends CaptureNode {
    override async run(inputs: Readonly<Record<string, unknown>>): Promise<NodeResult> {
      return this.forward(inputs as Record<string, unknown>);
    }
  }

  const node = new CustomRun();
  const evaluated = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { topic: "reefs" },
    captureBindings: true,
  });
  expect((evaluated.outcome as Output).value).toBe("reefs");
  expect(evaluated.bindings!.boundInputs).toEqual({ topic: "reefs", limit: 4n });
  expect(evaluated.backwardCapture!.inputNames).toEqual(["topic", "limit"]);
  expect(node.captures).toBe(0);
});

test("an agent capture keeps the variable inventory separate from the delivered inputs", async () => {
  const config = { provider: "openai", model: "gpt-5.5" };
  const variables = parametersFromYaml("variables:\n  instruction: !str Summarize clearly\n", new Map(), {
    field: "variables",
    requireDefault: true,
  });
  const node = new AgentNode("writer", { prompt: "${topic}: ${instruction}", llmConfig: config, variables });
  node.params = [new Parameter("topic")];
  const reply = new Message({ role: "assistant", parts: [new TextPart("reefs: Summarize clearly")] });
  const model = new ScriptedModel([new Response({ message: reply, finishReason: "stop" })], config);

  const evaluated = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { topic: "reefs" },
    captureBindings: true,
    models: { env: NO_ENV, createModel: () => model },
  });

  expect((evaluated.outcome as Output).value).toBe("reefs: Summarize clearly");
  expect(model.requests[0]!.messages.at(-1)!.text()).toBe("reefs: Summarize clearly");
  expect(evaluated.backwardCapture!.variableNames).toEqual(["instruction"]);
  const render = node.renderCapture(evaluated.bindings!.boundInputs!);
  expect(render["variables"]).toEqual({ instruction: "Summarize clearly" });
  expect(render["rendered_prompt"]).toBe("reefs: Summarize clearly");
  expect(evaluated.bindings!.boundInputs).toEqual({ topic: "reefs" });
});
