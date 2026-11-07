/**
 * The backward context a node's retained invocation material builds, and AGENT's backward: the
 * recorded model and no tools, feedback checks, failure attribution, and secrets scrubbed.
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
import { Message, RefusalPart, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { sourceKey } from "../../src/expr/lineage.ts";
import { backwardAgent } from "../../src/nodes/agent/backward.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { LLM_CONFIG_KEY } from "../../src/nodes/agent/llm_values.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { BackwardContext, BackwardResult, CapabilityFault, validateBackwardResult } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { BackwardComputeError, computeBackward, learningMaterials } from "../../src/runtime/backward.ts";
import { recordedGrants } from "../../src/run.ts";
import { validateAttributedBackward } from "../../src/runtime/learning.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { type Value, writeAsText } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../helpers/declarations.ts";
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
} from "../helpers/run_state_fixtures.ts";

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
    new RangeError("node data uses reserved _evaluation_evidence"),
  );
});

test("the backward context rebuilds the node data the record doesn't hold", () => {
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
  expect(trace.backwardContext(occurrence.source).nodeData).toEqual({ input_output_paths: {}, _evaluation_evidence: null });
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
  const accepted = {
    effective_llm_config: llmConfig,
    prompt_template: "${topic}: ${tone}",
    rendered_prompt: "reefs: concise",
    variable_domains: { tone: "Literal(concise, detailed)" },
    variables: { tone: "concise" },
  };
  expect(occurrence.nodeData).toEqual(accepted);
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
    const options = { grants: recordedGrants(state), memory: null, models: null, onEvent: null };
    const material = (await learningMaterials(state, options)).get(sourceKey(occurrence.source)) ?? null;
    return state.trace!.backwardContext(occurrence.source, { material });
  };
  expect((await contextOf(run.state)).nodeData).toEqual(expected);
  node.prompt = "later prompt";
  node.llmConfig = { ...llmConfig, model: "later model" };
  const forbidden = (): never => {
    throw new Error("replay invoked AGENT work");
  };
  // On the run's node and on the copy the replay runs over.
  for (const target of [node, original.nodes.get("writer")!]) {
    target.forward = forbidden;
    target.backward = forbidden;
  }

  const replay = replayRecords(new RunState(original), run.log.records(), { growBuilder: noGrowth });

  expect((await contextOf(replay)).nodeData).toEqual(expected);
  expect(replay.trace!.occurrence(occurrence.source).nodeData).toEqual(accepted);
  expect(built).toBe(1);
});

// ---------- AGENT backward ----------

type ContextFields = ConstructorParameters<typeof BackwardContext>[1] & {
  inputs?: Readonly<Record<string, unknown>> | null;
};

/** A recorded AGENT invocation, with `overrides` replacing its fields. */
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

/** The three channels, empty unless `overrides` fills them. */
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
  // Mutate the config the model is built from; the record must not change.
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
  expect(node.tools).toEqual([{ kind: "name", name: "never-granted" }]);
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

/** AGENT `writer` ran once in a learnable run, its model map recorded. */
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
  ["a fence around no JSON", text("```json\nnot JSON\n```"), "backward.execute"],
  ["a fenced duplicate channel", text("```json\n" + DUPLICATE + "\n```"), "backward.execute"],
  ["a fenced lone surrogate", text("```json\n" + SURROGATE + "\n```"), "backward.execute"],
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

const FENCED_BODY = JSON.stringify({
  input_feedback: { topic: ["Name the reef."] },
  self_feedback: ["Lead with steps."],
  variable_feedback: {},
});

test.each([
  ["a json fence", "```json\n" + FENCED_BODY + "\n```"],
  ["a bare fence", "```\n" + FENCED_BODY + "\n```"],
  ["prose before the fence", "Here it is:\n```json\n" + FENCED_BODY + "\n```"],
  ["whitespace around the object", "\n  " + FENCED_BODY + "  \n"],
])("AGENT backward reads a reply holding its object in %s", async (_label, body) => {
  const chat = new ScriptedModel([text(body)]);

  const result = await backwardAgent(context(), ["Review."], { models: serving(chat) });

  expect(result.inputFeedback).toEqual({ topic: ["Name the reef."] });
  expect(result.selfFeedback).toEqual(["Lead with steps."]);
});

const NO_OBJECT = "AGENT backward reply is not a JSON object with input_feedback, self_feedback and variable_feedback";

test.each([
  ["no JSON", "not JSON"],
  ["a fence around no JSON", "```json\nnot JSON\n```"],
  ["an array", "[]"],
  ["a fenced duplicate channel", "```json\n" + DUPLICATE + "\n```"],
  ["a fenced lone surrogate", "```json\n" + SURROGATE + "\n```"],
])("AGENT backward names the expected channels when the reply holds %s", async (_label, body) => {
  const chat = new ScriptedModel([text(body)]);

  const error = await rejected(backwardAgent(context(), ["Review."], { models: serving(chat) }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(NO_OBJECT);
});

test("AGENT backward names the model's refusal when the reply holds no JSON object", async () => {
  const message = new Message({ role: "assistant", parts: [new RefusalPart("I can't review this.")] });
  const chat = new ScriptedModel([new Response({ message, finishReason: "stop" })]);

  const error = await rejected(backwardAgent(context(), ["Review."], { models: serving(chat) }));

  expect((error as Error).message).toBe(`${NO_OBJECT}; the model refused: I can't review this.`);
});

test("a fenced AGENT backward reply routes its feedback through the backward pass", async () => {
  const { trace, occurrence } = await retainedEngine();
  const chat = new ScriptedModel([text("```json\n" + FENCED_BODY + "\n```")]);

  const [root, attributed] = await computeBackward(trace, "Review.", { models: serving(chat) });

  expect(root.inputFeedback).toEqual({ topic: ["Name the reef."] });
  const writer = attributed.find((item) => item.source.executionSeq === occurrence.source.executionSeq)!;
  expect(writer.result.selfFeedback).toEqual(["Lead with steps."]);
  validateAttributedBackward(trace, "Review.", root, attributed);
});

test("AGENT backward reads no live render, model or variable state", async () => {
  const { node, trace } = await retainedEngine();
  // The bound inputs and node data are rebuilt from the node, so they are read before it changes.
  for (const occurrence of trace.occurrences) void [occurrence.boundInputs, occurrence.nodeData];
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
