/**
 * The backward context a node's retained invocation material builds: a port of the node-neutral
 * rows of `tests/engine/test_agent_backward.py`.
 *
 * Ported: `test_backward_context_rejects_captured_reserved_evidence_collision` (the node's
 * `captureBackwardContext` replaced on the instance; the error is a `RangeError` with its full
 * message, where Python matches a part of a `ValueError`) and
 * `test_backward_context_allows_absent_node_data_without_changing_the_record`.
 *
 * Ported, through AGENT's forward in a learnable `FlowEngine` run with a `ScriptedModel` from
 * `models`: `test_backward_context_exposes_only_accepted_evidence_and_replays_without_agent_calls`.
 * The run records the agent's model map by hand (Python's run start derives it). Before the replay
 * the node's `forward` and `backward` (on the run's node and the replayed copy) are replaced with functions that throw, and the model access
 * must build no second model, where Python patches the render, variable resolution, backward and
 * the model factory to fail.
 *
 * Later (T6-6), with AGENT's backward: `test_agent_backward_uses_recorded_model_and_no_tools`,
 * `test_agent_backward_keeps_visible_tool_evidence_inert_and_separate`,
 * `test_agent_backward_skips_empty_effective_feedback`,
 * `test_agent_backward_validates_all_feedback_before_skipping_or_calling`,
 * `test_agent_backward_empty_string_is_effective_and_nested_arrays_keep_their_shape`,
 * `test_agent_backward_rejects_missing_retained_material_without_fallback`,
 * `test_agent_backward_rejects_malformed_retained_capture`,
 * `test_agent_backward_uses_partial_failure_capture_without_inventing_locals`,
 * `test_agent_backward_failure_is_attributed_without_forward_changes`,
 * `test_agent_backward_reads_no_live_render_model_or_variable_state` and
 * `test_agent_backward_host_controls_preserve_original_identity` (the last three also run AGENT's
 * forward).
 */

import { expect, test } from "bun:test";

import { copyGraph, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { evidencePayload } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  prepareNodeSucceeded,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import { engine, forwardCapture, FuncNode, graph, noGrowth } from "./run_state_fixtures.ts";

test("the backward context rejects a captured reserved evidence collision", async () => {
  const node = new FuncNode("custom", () => "done");
  node.captureBackwardContext = () => ({ _evaluation_evidence: null });
  const built = engine(graph([node], [[START_ID, "custom"]], { outputs: [new FlowOutput("result", "${custom.output}")] }), {
    learnable: true,
  });

  await built.run();

  const trace = built.state.trace!;
  const occurrence = trace.occurrences.find((item) => item.nodeId === "custom")!;
  expect(() => trace.backwardContext(occurrence.source)).toThrow(
    new RangeError("captured node data uses reserved _evaluation_evidence"),
  );
});

test("the backward context allows absent node data without changing the record", () => {
  const built = engine(graph([], []), { learnable: true });
  const { state, log } = built;
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(
    log,
    state,
    prepareNodeSucceeded(state, START_ID, {}, { forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}) }),
  );

  const trace = state.trace!;
  expect(trace.occurrences).toHaveLength(1);
  const occurrence = trace.occurrences[0]!;
  expect(occurrence.record.forwardCapture!.nodeData).toBeNull();
  expect(trace.backwardContext(occurrence.source).nodeData).toEqual({ _evaluation_evidence: null });
  expect(occurrence.record.forwardCapture!.nodeData).toBeNull();
});

test("the backward context exposes only accepted evidence and replays without agent calls", async () => {
  const llmConfig = { provider: "test", model: "recorded" };
  const registry = readTypedefs({ Tone: "Literal(concise, detailed)" });
  const variables = parametersFromYaml("variables:\n  tone: !Tone concise\n", registry, {
    field: "variables",
    requireDefault: true,
  });
  const node = new AgentNode("writer", { prompt: "${topic}: ${tone}", llmConfig, variables });
  node.params = [new Parameter("topic"), new Parameter("tags")];
  const flow = graph([node], [[START_ID, "writer"]], {
    wiring: { writer: { topic: "${input.topic}", tags: "${input.tags}" } },
    outputs: [new FlowOutput("result", "${writer.output}")],
  });
  const reply = new Message({ role: "assistant", parts: [new TextPart("forward answer")] });
  const model = new ScriptedModel([new Response({ message: reply, finishReason: "stop" })], llmConfig);
  let built = 0;
  const run = engine(flow, {
    inputs: { topic: "reefs", tags: ["a", "b"] },
    learnable: true,
    effectiveLlmByStaticPath: { writer: llmConfig },
    models: {
      env: () => undefined,
      createModel: () => {
        built++;
        return model;
      },
    },
  });
  const original = copyGraph(run.state.flow);
  await run.run();

  const occurrence = run.state.trace!.occurrences.find((item) => item.nodeId === "writer")!;
  const accepted = occurrence.record.forwardCapture!.nodeData!;
  const expected = { ...accepted, _evaluation_evidence: evidencePayload(occurrence.record.evidence) };
  expect(Object.hasOwn(accepted, "_evaluation_evidence")).toBe(false);
  expect(run.state.trace!.backwardContext(occurrence.source).nodeData).toEqual(expected);
  node.prompt = "later prompt";
  node.llmConfig = { ...llmConfig, model: "later model" };
  const forbidden = (): never => {
    throw new Error("replay invoked AGENT work");
  };
  // On the run's node and on the copy the replay runs over, as Python patches the class.
  for (const target of [node, original.nodes.get("writer")!]) {
    target.forward = forbidden;
    target.backward = forbidden;
  }

  const replay = replayRecords(new RunState(original), run.log.records(), { growBuilder: noGrowth });

  expect(replay.trace!.backwardContext(occurrence.source).nodeData).toEqual(expected);
  expect(replay.trace!.occurrence(occurrence.source).record.forwardCapture!.nodeData).toEqual(accepted);
  expect(built).toBe(1);
  // Clearing the expected copy leaves the replayed context's evidence whole.
  (expected._evaluation_evidence as { interactions: unknown[] }).interactions.length = 0;
  const evidence = replay.trace!.backwardContext(occurrence.source).nodeData!["_evaluation_evidence"];
  expect((evidence as { interactions: unknown[] }).interactions).not.toEqual([]);
});
