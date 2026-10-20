/**
 * Host steering of a running AGENT: mirrors `tests/engine/test_agent_steering.py`.
 *
 * Ported (split): the `run({})` half of
 * `test_python_built_steerable_agent_with_structured_output_fails_before_the_model`; its
 * `eval_node` half needs the evaluator (T4).
 *
 * Ported, through a hand-built `START -> chat -> END` flow run by `FlowEngine` with a `ScriptedModel`
 * from `models` (Python: `load_flow` and `run_flow`): `test_non_steerable_agent_never_calls_the_inbox`,
 * on the plain path.
 *
 * Changed: the structured output contract is built by hand where Python reads it from YAML through
 * `load_flow`; the node runs with no model, so there is no model to check for calls. Python's
 * `ValueError` is a `RangeError`.
 *
 * Later (T4): `test_run_boundary_rejects_an_object_without_the_inbox_methods` (the run boundary).
 *
 * Later (T5-11b), a steerable AGENT runs the tool loop:
 * `test_run_context_receives_steering_only_when_steerable` and the steer and interrupt tests
 * (`test_steers_are_sent_as_user_messages_at_the_top_of_each_iteration` through
 * `test_retry_after_a_mid_node_kill_sees_the_recorded_steer`; the resume and loop ones also need
 * T3-22 and T4).
 *
 * Not ported: `test_inbox_satisfies_the_public_protocol` (it checks the test's inbox against a
 * `typing.Protocol`; TS checks `Steering` statically).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Steering } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { engine, graph } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A host inbox with steers queued for `chat` and its interrupt aborted; `calls` records each call. */
function inbox(): Steering & { calls: [string, string][] } {
  const calls: [string, string][] = [];
  const interrupted = new AbortController();
  interrupted.abort(new Error("interrupted"));
  return {
    calls,
    interruptSignal: (nodeId) => {
      calls.push(["interruptSignal", nodeId]);
      return interrupted.signal;
    },
    drain: (nodeId) => {
      calls.push(["drain", nodeId]);
      return ["ignored"];
    },
  };
}

test("a non-steerable agent never calls the inbox", async () => {
  const model = new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [new TextPart("done")] }), finishReason: "stop" })],
    CONFIG,
  );
  const steering = inbox();
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG });
  const flow = graph([chat], [[START_ID, "chat"]], { outputs: [new FlowOutput("output", "${chat.output}")] });

  const events = await engine(flow, { steering, models: { env: () => undefined, createModel: () => model } }).run();

  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  expect((last as RunSucceeded).output).toBe("done");
  expect(steering.calls).toEqual([]);
});

test("python built steerable agent with structured output fails before the model (run half)", async () => {
  const contract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["answer"]),
    }),
  );
  const llmConfig = { provider: "anthropic", model: "claude-test" };
  const node = new AgentNode("chat", { prompt: "hi", llmConfig, steerable: true });
  node.outputContract = contract;
  const run = node.run({});
  await expect(run).rejects.toBeInstanceOf(RangeError);
  await expect(run).rejects.toThrow("AGENT node 'chat': steerable needs `str` output");
});
