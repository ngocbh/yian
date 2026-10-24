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
 * the node's `forward` and `backward` (on the run's node and the replayed copy) are replaced with
 * functions that throw, and the model access must build no second model, where Python patches the
 * render, variable resolution, backward and the model factory to fail.
 *
 * Ported, with AGENT's backward (`node.backward` or `backwardAgent` called directly, the model a
 * `ScriptedModel` served by the `models` option where Python patches `model_from_config`; prompts
 * are compared as text, the text form of the material, never read with `JSON.parse`):
 * `test_agent_backward_uses_recorded_model_and_no_tools` (the whole prompt text pinned),
 * `test_agent_backward_keeps_visible_tool_evidence_inert_and_separate`,
 * `test_agent_backward_skips_empty_effective_feedback` (4 rows),
 * `test_agent_backward_empty_string_is_effective_and_nested_arrays_keep_their_shape` (the result is
 * frozen, so the reader's write throws a `TypeError` where Python's lands on a copy),
 * `test_agent_backward_rejects_missing_retained_material_without_fallback` (6 rows; the full
 * `RangeError` message pinned), `test_agent_backward_rejects_malformed_retained_capture` (13 rows)
 * and `test_agent_backward_uses_partial_failure_capture_without_inventing_locals`.
 * `test_agent_backward_validates_all_feedback_before_skipping_or_calling` (4 rows; the last, a
 * list where Python wants a tuple, becomes feedback that isn't an array, since an array is the
 * only list).
 *
 * Ported, through AGENT's forward in a learnable `FlowEngine` run, then `computeBackward` with the
 * backward model in its `models` option: `test_agent_backward_failure_is_attributed_without_forward_changes`
 * (13 rows; the duplicate-key row fails in `parseJson`, with its message, and the model error is an
 * `Error`, Python's `ValueError`) and `test_agent_backward_reads_no_live_render_model_or_variable_state`
 * (the node's `forward`, `declaredVariables`, `captureBackwardContext` and private
 * `resolveVariables` are replaced on the instance; ES modules can't patch the render, type label
 * and memory functions Python patches).
 *
 * Changed: `test_agent_backward_host_controls_preserve_original_identity`. There is no
 * host-control unwrapping (`computeBackward` attributes every handler throw), and TS has no
 * `MemoryError`, `_HostStop` or cancellation classes, so each row (an abort `DOMException`, a
 * nested `CapabilityFault`, an `Error` wrapping another as `cause`; thrown by `createModel` or by
 * the call) is a `BackwardComputeError` coded `backward.execute` whose `cause` is the thrown value
 * itself, scrubbed of the node's own `api_key`, with the log, result and trace unchanged.
 *
 * Ported from `tests/engine/test_openai_connection_recovery.py`:
 * `test_old_backward_config_gets_api_semantics_before_factory` (it builds its context with this
 * file's `context`; it calls `backwardAgent` directly).
 *
 * TS-only: `__llm_config` is left out of the prompt's inputs; a retried call publishes one
 * `ModelCallRetried` (consumer `backward`, the error scrubbed) through `onEvent`; `createModel`
 * receives the node id, consumer `backward` and a signal that isn't aborted, and each
 * `onReplayDegraded` reason is published once as `ReplayDegraded`; a model error is scrubbed of the
 * key the call sent; a context that isn't a `BackwardContext` is refused.
 */

import { expect, test } from "bun:test";

import { copyGraph, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Failure } from "../../src/diagnostics.ts";
import { ModelCallRetried, ReplayDegraded } from "../../src/events.ts";
import { legacyConnectionConfig } from "../../src/llm_clients/connection.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { sourceKey } from "../../src/expr/lineage.ts";
import { backwardAgent } from "../../src/nodes/agent/backward.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { LLM_CONFIG_KEY } from "../../src/nodes/agent/llm_values.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { BackwardContext, BackwardResult, CapabilityFault, validateBackwardResult } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { BackwardComputeError, computeBackward, learningMaterials } from "../../src/runtime/backward.ts";
import { validateAttributedBackward } from "../../src/runtime/learning.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { type Value, writeAsText } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import {
  engine,
  forwardCapture,
  FuncNode,
  graph,
  NO_ENV,
  noGrowth,
  prepareOutput,
  rejected,
  serving,
  traceSnapshot,
} from "./run_state_fixtures.ts";

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
    prepareOutput(log, state, START_ID, {}, { forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}) }),
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
      env: NO_ENV,
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
  const evidence = {
    scope: null,
    bound_inputs: { topic: "reefs", tags: ["a", "b"] },
    base_system_prompt: DEFAULT_SYSTEM,
    prompt: "reefs: concise",
    memory_sections: [],
    interactions: [{ kind: "model_text", text: "forward answer" }],
  };
  const expected = { ...accepted, _evaluation_evidence: evidence };
  expect(Object.hasOwn(accepted, "_evaluation_evidence")).toBe(false);
  /** The context backward reads for the occurrence, its material rebuilt from `state`'s trace. */
  const contextOf = async (state: RunState) => {
    const options = { memory: null, models: null, expandTools: null, onEvent: null };
    const material = (await learningMaterials(state, options)).get(sourceKey(occurrence.source)) ?? null;
    return state.trace!.backwardContext(occurrence.source, { material });
  };
  expect((await contextOf(run.state)).nodeData).toEqual(expected);
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

  expect((await contextOf(replay)).nodeData).toEqual(expected);
  expect(replay.trace!.occurrence(occurrence.source).record.forwardCapture!.nodeData).toEqual(accepted);
  expect(built).toBe(1);
});

// ---------- AGENT backward ----------

type ContextFields = ConstructorParameters<typeof BackwardContext>[1] & {
  inputs?: Readonly<Record<string, unknown>> | null;
};

/** Python's `_context`: a recorded AGENT invocation, with `overrides` replacing its fields. */
function context(overrides: ContextFields = {}): BackwardContext {
  const { inputs = { topic: "reefs", tags: ["a", "b"] }, ...fields } = overrides;
  return new BackwardContext(inputs, {
    inputNames: ["topic", "tags"],
    variableNames: ["tone"],
    output: { answer: "recorded answer", tags: ["a", "b"] },
    runId: "recorded-run",
    nodeId: "recorded-node",
    nodeStartedSeq: 3,
    nodeData: nodeData(),
    downstreamContext: [{ message_index: 0, contexts: [{ expression: "items[1]" }] }],
    ...fields,
  });
}

/** A fresh copy of `context()`'s recorded node data, to change before building a context. */
function nodeData(): Record<string, unknown> {
  return {
    prompt_template: "${topic}: ${tone}",
    variables: { tone: "concise" },
    variable_domains: { tone: 'Literal("concise","detailed")' },
    rendered_prompt: "reefs: concise",
    effective_llm_config: { provider: "anthropic", model: "recorded", anthropic: { temperature: 0.4 } },
    _evaluation_evidence: null,
  };
}

/** A model reply with `text` as its only text. */
function text(reply: string): Response {
  const message = new Message({ role: "assistant", parts: [new TextPart(reply)] });
  return new Response({ message, finishReason: "stop" });
}

/** Python's `_reply`: the three channels, empty unless `overrides` fills them. */
function reply(overrides: Record<string, unknown> = {}): Response {
  return text(JSON.stringify({ input_feedback: {}, self_feedback: [], variable_feedback: {}, ...overrides }));
}

/** A model access that serves `model` and keeps a copy of each config and the options it is given. */
function access(model: ChatModel): ModelAccess & { configs: Record<string, unknown>[]; options: unknown[] } {
  const configs: Record<string, unknown>[] = [];
  const options: unknown[] = [];
  return {
    configs,
    options,
    env: NO_ENV,
    createModel: (config, given) => {
      configs.push(structuredClone(config) as Record<string, unknown>);
      options.push(given);
      return model;
    },
  };
}

/** A model access whose build fails the test. */
const NO_MODEL: ModelAccess = {
  env: NO_ENV,
  createModel: () => {
    throw new Error("backward constructed a model");
  },
};

/** The text of the user message of `model`'s request `index`. */
function promptOf(model: ScriptedModel, index = 0): string {
  return model.requests[index]!.messages[1]!.text();
}

test("AGENT backward uses the recorded model and no tools", async () => {
  const node = new AgentNode("mutable-node", {
    prompt: "mutable prompt",
    tools: ["never-granted"],
    llmConfig: { provider: "ambient", model: "wrong" },
  });
  const recorded = context();
  const feedback = [{ tags: [null, "Check the second tag."] }, "Explain the answer."];
  const chat = new ScriptedModel([
    reply({
      input_feedback: { tags: [[null, "Use a relevant tag."]] },
      self_feedback: ["Preserve uncertainty."],
      variable_feedback: { tone: ["Be precise."] },
    }),
  ]);
  const models = access(chat);
  const build = models.createModel;
  // As Python's factory, mutate the config the model is built from; the record must not change.
  models.createModel = (config, given) => {
    const built = build(config, given);
    (config as Record<string, unknown>)["model"] = "factory mutation";
    return built;
  };

  const result = await node.backward(recorded, feedback, { models });

  expect(result).toEqual(
    new BackwardResult({
      inputFeedback: { tags: [[null, "Use a relevant tag."]] },
      selfFeedback: ["Preserve uncertainty."],
      variableFeedback: { tone: ["Be precise."] },
    }),
  );
  validateBackwardResult(result, recorded);
  // A record without a connection replays on its provider's key variable.
  const config = recorded.nodeData["effective_llm_config"] as Record<string, unknown>;
  expect(models.configs).toEqual([legacyConnectionConfig(config)]);
  expect(chat.requests).toHaveLength(1);
  const request = chat.requests[0]!;
  expect(request.tools).toEqual([]);
  expect(request.messages.map((message) => message.role)).toEqual(["system", "user"]);
  expect(promptOf(chat)).toBe(
    '{"forward":{"run_id":"recorded-run","node_id":"recorded-node","node_started_seq":3,' +
      '"inputs":{"topic":"reefs","tags":["a","b"]},"input_names":["topic","tags"],' +
      '"output":{"answer":"recorded answer","tags":["a","b"]},"failure":null,"variable_names":["tone"],' +
      '"prompt_template":"${topic}: ${tone}","rendered_prompt":"reefs: concise","variables":{"tone":"concise"},' +
      '"variable_domains":{"tone":"Literal(\\"concise\\",\\"detailed\\")"},"evaluation_evidence":null},' +
      '"feedback":[{"tags":[null,"Check the second tag."]},"Explain the answer."],' +
      '"downstream_context":[{"message_index":0,"contexts":[{"expression":"items[1]"}]}]}',
  );
  expect((recorded.nodeData["effective_llm_config"] as Record<string, unknown>)["model"]).toBe("recorded");
  expect(node.prompt).toBe("mutable prompt");
  expect(node.tools).toEqual(["never-granted"]);
});

test("AGENT backward keeps visible tool evidence inert and separate", async () => {
  const evidence = {
    scope: null,
    bound_inputs: {},
    base_system_prompt: "Recorded base system.",
    prompt: "Recorded prompt.",
    memory_sections: [],
    interactions: [
      { kind: "tool_call", name: "search", arguments: { query: "reefs" } },
      { kind: "tool_result", name: "search", text: "Found reefs." },
      { kind: "model_text", text: "Recorded visible answer." },
    ],
  };
  const data = nodeData();
  data["_evaluation_evidence"] = evidence;
  const chat = new ScriptedModel([reply()]);

  const node = new AgentNode("unused", { tools: ["search"] });
  const result = await node.backward(context({ nodeData: data }), ["Review."], { models: serving(chat) });

  expect(result).toEqual(new BackwardResult());
  expect(chat.requests).toHaveLength(1);
  expect(chat.requests[0]!.tools).toEqual([]);
  expect(chat.requests[0]!.messages.map((message) => message.role)).toEqual(["system", "user"]);
  const prompt = promptOf(chat);
  expect(prompt).toContain(`"evaluation_evidence":${writeAsText(evidence as unknown as Value)}}`);
  expect(prompt).toContain('"rendered_prompt":"reefs: concise"');
});

for (const feedback of [[], [null], [{}], [{ tags: [null, null] }]]) {
  test(`AGENT backward skips empty effective feedback ${JSON.stringify(feedback)}`, async () => {
    const result = await new AgentNode("unused").backward(context({ nodeData: {} }), feedback, { models: NO_MODEL });
    expect(result).toEqual(new BackwardResult());
  });
}

for (const feedback of [[[null]], [{ tags: [null] }], ["review", { unknown: null }], "review"]) {
  test(`AGENT backward validates all feedback before skipping or calling ${JSON.stringify(feedback)}`, async () => {
    const error = await rejected(new AgentNode("unused").backward(context(), feedback as never, { models: NO_MODEL }));
    expect(error instanceof TypeError || error instanceof RangeError).toBe(true);
  });
}

test("AGENT backward treats the empty string as effective, and nested arrays keep their shape", async () => {
  const chat = new ScriptedModel([
    reply({
      input_feedback: { tags: [[null, ""]] },
      self_feedback: ["", "second"],
      variable_feedback: { tone: ["first", "second"] },
    }),
  ]);
  const recorded = context();

  const result = await new AgentNode("unused").backward(recorded, [""], { models: serving(chat) });

  expect(chat.requests).toHaveLength(1);
  expect(result.inputFeedback).toEqual({ tags: [[null, ""]] });
  expect(result.selfFeedback).toEqual(["", "second"]);
  expect(result.variableFeedback).toEqual({ tone: ["first", "second"] });
  validateBackwardResult(result, recorded);
  expect(() => {
    (result.inputFeedback["tags"]![0] as unknown[])[1] = "reader mutation";
  }).toThrow(TypeError);
  expect(result.inputFeedback).toEqual({ tags: [[null, ""]] });
});

for (const key of [
  "prompt_template",
  "rendered_prompt",
  "variables",
  "variable_domains",
  "effective_llm_config",
  "_evaluation_evidence",
]) {
  test(`AGENT backward rejects missing retained ${key} without a fallback`, async () => {
    const data = nodeData();
    delete data[key];
    const node = new AgentNode("unused", { llmConfig: { provider: "ambient", model: "fallback" } });
    const error = await rejected(node.backward(context({ nodeData: data }), ["review"], { models: NO_MODEL }));
    expect(error).toEqual(new RangeError(`AGENT backward requires recorded capture fields: ['${key}']`));
  });
}

for (const [key, value] of [
  ["prompt_template", null],
  ["rendered_prompt", []],
  ["variables", []],
  ["variables", { foreign: "value" }],
  ["variables", null],
  ["variable_domains", { tone: 1 }],
  ["variable_domains", {}],
  ["variable_domains", null],
  ["effective_llm_config", null],
  ["effective_llm_config", {}],
  ["effective_llm_config", { provider: "test", model: " " }],
  ["effective_llm_config", { provider: null, model: "recorded" }],
  ["_evaluation_evidence", []],
] as const) {
  test(`AGENT backward rejects a malformed retained ${key} ${JSON.stringify(value)}`, async () => {
    const data = nodeData();
    data[key] = value;
    const node = new AgentNode("unused");
    const error = await rejected(node.backward(context({ nodeData: data }), ["review"], { models: NO_MODEL }));
    expect(error instanceof TypeError || error instanceof RangeError).toBe(true);
  });
}

test("AGENT backward uses a partial failure capture without inventing locals", async () => {
  const data = { ...nodeData(), variables: null, rendered_prompt: null };
  const recorded = context({
    inputs: null,
    output: null,
    failure: new Failure({ code: "node.execute", message: "resolution failed", nodeId: "recorded-node" }),
    nodeData: data,
  });
  const chat = new ScriptedModel([
    reply({ self_feedback: ["Check resolution."], variable_feedback: { tone: ["Use a valid default."] } }),
  ]);

  const result = await new AgentNode("unused").backward(recorded, ["Explain the failure."], { models: serving(chat) });

  validateBackwardResult(result, recorded);
  const prompt = promptOf(chat);
  expect(prompt).toContain('"inputs":null');
  expect(prompt).toContain('"rendered_prompt":null,"variables":null');
  expect(prompt).toContain('"failure":{"code":"node.execute","message":"resolution failed","cause_type":null}');
  expect(result.inputFeedback).toEqual({});
});

test("AGENT backward leaves __llm_config out of the prompt's inputs", async () => {
  const chat = new ScriptedModel([reply()]);
  const inputs = { topic: "reefs", tags: ["a", "b"], [LLM_CONFIG_KEY]: { provider: "test", model: "m" } };
  const recorded = context({ inputs });

  await backwardAgent(recorded, ["Review."], { models: serving(chat) });

  expect(promptOf(chat)).toContain('"inputs":{"topic":"reefs","tags":["a","b"]},"input_names"');
});

test("AGENT backward refuses a context that isn't a BackwardContext", async () => {
  const error = await rejected(backwardAgent({} as BackwardContext, ["Review."], { models: NO_MODEL }));
  expect(error).toEqual(new TypeError("AGENT backward requires a BackwardContext and a feedback array"));
});

test("AGENT backward builds its model as consumer backward and publishes each replay reason once", async () => {
  const chat = new ScriptedModel([reply()]);
  const seen: unknown[] = [];
  const given: { nodeId: string; consumer?: string; signal: AbortSignal }[] = [];
  const models: ModelAccess = {
    env: NO_ENV,
    createModel: (_config, options) => {
      given.push(options);
      options.onReplayDegraded?.("other_account");
      options.onReplayDegraded?.("other_account");
      options.onReplayDegraded?.("other_provider");
      return chat;
    },
  };

  await backwardAgent(context(), ["Review."], { models, onEvent: (event) => void seen.push(event) });

  expect(given).toHaveLength(1);
  expect(given[0]!.nodeId).toBe("recorded-node");
  expect(given[0]!.consumer).toBe("backward");
  expect(given[0]!.signal.aborted).toBe(false);
  expect(seen).toEqual([
    new ReplayDegraded({ nodeId: "recorded-node", reason: "other_account" }),
    new ReplayDegraded({ nodeId: "recorded-node", reason: "other_provider" }),
  ]);
});

test("AGENT backward publishes a retried call as ModelCallRetried, scrubbed", async () => {
  const failure = new ProviderError("server", "boom for sk-backward-secret", { retryAfter: 0.001 });
  const chat = new ScriptedModel([failure, reply({ self_feedback: ["Retried."] })]);
  const seen: unknown[] = [];
  // One retry whatever YIAN_MODEL_RETRIES the environment sets.
  install({ ...BUILTIN, modelRetries: 1 });
  let result: BackwardResult;
  try {
    result = await backwardAgent(context(), ["Review."], {
      models: serving(chat),
      apiKey: "sk-backward-secret",
      onEvent: (event) => void seen.push(event),
    });
  } finally {
    forget();
  }

  expect(result.selfFeedback).toEqual(["Retried."]);
  expect(chat.requests).toHaveLength(2);
  expect(seen).toHaveLength(1);
  const event = seen[0] as ModelCallRetried;
  expect(event).toBeInstanceOf(ModelCallRetried);
  expect([event.nodeId, event.consumer, event.attempt, event.error.kind, event.retryIn]).toEqual([
    "recorded-node",
    "backward",
    1,
    "server",
    0.001,
  ]);
  expect(event.error.message).toStartWith("boom for ");
  expect(event.error.message).not.toContain("sk-backward-secret");
});

test("AGENT backward scrubs the key the call sent from a model error", async () => {
  const chat = new ScriptedModel([new Error("refused sk-backward-secret")]);
  const error = await rejected(
    backwardAgent(context(), ["Review."], { models: serving(chat), apiKey: "sk-backward-secret" }),
  );
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toStartWith("refused ");
  expect((error as Error).message).not.toContain("sk-backward-secret");
});

test("an old backward config gets API semantics before the factory", async () => {
  const data = nodeData();
  data["effective_llm_config"] = { provider: "openai", model: "old-model" };
  const models = access(new ScriptedModel([reply()]));

  await backwardAgent(context({ nodeData: data }), [{ answer: "Review" }], { models });

  const connection = models.configs[0]!["_connection"] as Record<string, unknown>;
  expect(connection["method"]).toBe("api_key");
  expect(connection["source"]).toBe("legacy_environment");
});

/** Python's `_retained_engine`: AGENT `writer` ran once in a learnable run, its model map recorded. */
async function retainedEngine(llmConfig: Record<string, unknown> = { provider: "test", model: "recorded" }) {
  const recorded = { provider: "test", model: "recorded" };
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
  const run = engine(flow, {
    inputs: { topic: "reefs", tags: ["a", "b"] },
    learnable: true,
    effectiveLlmByStaticPath: { writer: recorded },
    models: serving(new ScriptedModel([text("forward answer")])),
  });
  await run.run();
  const trace = run.state.trace!;
  const occurrence = trace.occurrences.find((item) => item.nodeId === "writer")!;
  const snapshot = () => [run.log.records(), run.state.result, traceSnapshot(trace), node.variables];
  return { run, node, trace, occurrence, snapshot };
}

const DUPLICATE = '{"input_feedback":{},"self_feedback":[],"self_feedback":[],"variable_feedback":{}}';
const EXTRA = '{"input_feedback":{},"self_feedback":[],"variable_feedback":{},"other":null}';
const SURROGATE = '{"input_feedback":{},"self_feedback":["\\ud800"],"variable_feedback":{}}';
const FAILURE_ROWS: [string, Response | Error, string][] = [
  ["not JSON", text("not JSON"), "backward.execute"],
  ["an array", text("[]"), "backward.execute"],
  ["an extra channel", text(EXTRA), "backward.execute"],
  ["a duplicate channel", text(DUPLICATE), "backward.execute"],
  ["a lone surrogate", text(SURROGATE), "backward.execute"],
  ["a flat batch", reply({ input_feedback: { tags: "flat batch" } }), "backward.execute"],
  ["a null self message", reply({ self_feedback: [null] }), "backward.execute"],
  ["an int variable message", reply({ variable_feedback: { tone: [1] } }), "backward.execute"],
  ["a boolean input message", reply({ input_feedback: { tags: [false] } }), "backward.execute"],
  ["a foreign input", reply({ input_feedback: { foreign: ["bad"] } }), "backward.result"],
  ["a foreign variable", reply({ variable_feedback: { foreign: ["bad"] } }), "backward.result"],
  ["a short positional message", reply({ input_feedback: { tags: [[null]] } }), "backward.result"],
  ["a model error", new Error("model failed"), "backward.execute"],
];

for (const [label, response, code] of FAILURE_ROWS) {
  test(`AGENT backward's failure on ${label} is attributed without forward changes`, async () => {
    const { run, trace, occurrence, snapshot } = await retainedEngine();
    const before = snapshot();
    const chat = new ScriptedModel([response]);

    const error = await rejected(computeBackward(trace, "Review.", { models: serving(chat) }));

    expect(error).toBeInstanceOf(BackwardComputeError);
    const failure = (error as BackwardComputeError).failure;
    expect(failure.code).toBe(code);
    expect(failure.nodeId).toBe("writer");
    expect(failure.nodeStartedSeq).toBe(occurrence.source.executionSeq);
    expect(failure.runId).toBe(run.state.runId);
    expect(failure.scope).toEqual(occurrence.record.forwardCapture!.scope);
    expect(chat.requests).toHaveLength(1);
    expect(chat.requests[0]!.tools).toEqual([]);
    expect(snapshot()).toEqual(before);
  });
}

test("AGENT backward reads no live render, model or variable state", async () => {
  const { node, trace } = await retainedEngine();
  node.prompt = "wrong";
  node.llmConfig = { provider: "wrong", model: "wrong" };
  node.variables = [];
  node.params = [];
  const forbidden = (): never => {
    throw new Error("backward reconstructed live forward material");
  };
  node.forward = forbidden;
  node.declaredVariables = forbidden;
  node.captureBackwardContext = forbidden;
  (node as unknown as { resolveVariables: () => never }).resolveVariables = forbidden;
  const chat = new ScriptedModel([
    reply({ input_feedback: { topic: ["Use a specific reef."] }, variable_feedback: { tone: ["Be precise."] } }),
  ]);
  const models = access(chat);

  const [root, attributed] = await computeBackward(trace, "Review.", { models });

  expect(root.inputFeedback).toEqual({ topic: ["Use a specific reef."] });
  expect(models.configs).toEqual([{ provider: "test", model: "recorded" }]);
  validateAttributedBackward(trace, "Review.", root, attributed);
  expect(chat.requests).toHaveLength(1);
});

const HOST_FAULTS: [string, () => unknown][] = [
  ["an abort", () => new DOMException("aborted for sk-node-secret", "AbortError")],
  ["a nested capability fault", () => new CapabilityFault(new CapabilityFault(new Error("original sk-node-secret")))],
  ["a wrapped error", () => new Error("model wrapper", { cause: new Error("wrapped sk-node-secret") })],
];

for (const stage of ["factory", "invoke"] as const) {
  for (const [kind, make] of HOST_FAULTS) {
    test(`AGENT backward keeps ${kind} thrown at ${stage} as backward.execute`, async () => {
      const llmConfig = { provider: "test", model: "recorded", api_key: "sk-node-secret" };
      const { trace, snapshot } = await retainedEngine(llmConfig);
      const before = snapshot();
      const raised = make();
      const chat = new ScriptedModel([raised as Error]);
      const models: ModelAccess = {
        env: NO_ENV,
        createModel: () => {
          if (stage === "factory") throw raised;
          return chat;
        },
      };

      const error = await rejected(computeBackward(trace, "Review.", { models }));

      expect(error).toBeInstanceOf(BackwardComputeError);
      expect((error as BackwardComputeError).failure.code).toBe("backward.execute");
      expect((error as Error).cause).toBe(raised);
      let inner: unknown = raised;
      while (inner instanceof Error) {
        expect(inner.message).not.toContain("sk-node-secret");
        inner = inner instanceof CapabilityFault ? inner.original : inner.cause;
      }
      expect(chat.requests).toHaveLength(stage === "invoke" ? 1 : 0);
      expect(snapshot()).toEqual(before);
    });
  }
}
