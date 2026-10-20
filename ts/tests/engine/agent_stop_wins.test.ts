/**
 * A stop pending when a model call fails stops the node cleanly: mirrors
 * `tests/engine/test_agent_stop_wins.py`.
 *
 * Changed: the rows run an AGENT through `evalNode` (`drive`) with a model from `models` and a
 * stop signal, where Python runs `_run_flow_controlled` with a stop event; the stop is the signal,
 * so a stopped node rejects with the signal's reason. Ported this way, the rows without session
 * memory: `test_a_stop_during_a_failing_model_call_stops_the_run`,
 * `test_a_stop_during_a_failing_structured_call_stops_the_run` and
 * `test_a_host_callback_fault_escapes_a_pending_stop`.
 * `test_a_stop_raised_while_observing_a_failure_wins` (the model row): TS has no memory turn
 * observing the call, so the observer is the event sink, which aborts the stop when it sees the
 * failed call's `ModelCallFinished`; the stop wins over the failure.
 * `test_a_stop_failing_a_call_yields_to_an_observed_stop_only_if_derived` (the model rows): TS has
 * one stop, the signal's reason, with no initiating or derived kind; a call the stop fails emits no
 * finished event and the node rejects with the reason.
 *
 * Later (T6-6), session memory: the session rows of
 * `test_a_stop_during_a_failing_model_call_stops_the_run` and
 * `test_a_host_callback_fault_escapes_a_pending_stop`, and the interrupted-effect records of
 * `test_a_stop_during_a_failing_structured_call_stops_the_run`.
 *
 * Later (T5-11b), the tool loop: `test_a_stop_during_a_failing_tool_call_stops_the_run` and the
 * tool and representation rows of `test_a_stop_raised_while_observing_a_failure_wins` and
 * `test_a_stop_failing_a_call_yields_to_an_observed_stop_only_if_derived`.
 *
 * Not ported: `test_a_stop_wins_over_a_malformed_native_carrier` (TS has no LangChain structured
 * carrier).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted } from "../../src/events.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, rejected } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };

/** A model whose call runs `during`, then throws `error`. */
function failing(error: unknown, during: () => void, config = CONFIG): ChatModel {
  return {
    ...config,
    stream: () =>
      new EventStream(
        (async function* (): AsyncGenerator<Event> {
          during();
          throw error;
        })(),
      ),
  };
}

/** Evaluate `node` (by default an AGENT `writer`) over `model` with the other `drive` options. */
function evaluate(model: ChatModel, options: Parameters<typeof drive>[1] & { node?: AgentNode } = {}) {
  const { node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG }), ...rest } = options;
  return drive(node, { models: { env: () => undefined, createModel: () => model }, ...rest });
}

test("a stop during a failing model call stops the node", async () => {
  const stop = new AbortController();
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")));
  expect(await rejected(evaluate(model, { signal: stop.signal }))).toBe(stop.signal.reason);
});

test("a stop during a failing structured call stops the node", async () => {
  const stop = new AbortController();
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: PROMPTED, retries: 2 });
  node.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")), PROMPTED);
  expect(await rejected(evaluate(model, { node, signal: stop.signal }))).toBe(stop.signal.reason);
});

test("a stop raised while observing a failed model call wins", async () => {
  const stop = new AbortController();
  const observed: string[] = [];
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallFinished) {
      observed.push(event.error ?? "");
      stop.abort(new Error("stopped"));
    }
  };
  const model = failing(new Error("boom"), () => {});
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(stop.signal.reason);
  expect(observed).toEqual(["Error"]);
});

test("a stop failing a model call is the node's rejection, with no finished event", async () => {
  const stop = new AbortController();
  const events: string[] = [];
  const emitEvent = (event: unknown) => void events.push((event as { type: string }).type);
  // The call fails with the stop's reason, as a call the signal aborted does.
  const model = failing(new Error("unreached"), () => {
    stop.abort(new Error("stopped"));
    stop.signal.throwIfAborted();
  });
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(stop.signal.reason);
  expect(events).toEqual(["ModelCallStarted"]);
});

test("a host callback fault escapes a pending stop", async () => {
  const stop = new AbortController();
  const fault = new RangeError("sink boom");
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) {
      stop.abort(new Error("stopped"));
      throw fault;
    }
  };
  const reply = new Response({ message: new Message({ role: "assistant", parts: [new TextPart("fine")] }), finishReason: "stop" });
  const model = new ScriptedModel([reply], CONFIG);
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(fault);
});
