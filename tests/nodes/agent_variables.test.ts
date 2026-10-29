/**
 * AGENT local variables: declarations, defaults, rendering, the retained render capture, and
 * keeping variables out of wiring and the pool.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type MaterialCaps, Node, NodeKind, Output } from "../../src/nodes/base.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { driveBackward } from "../../src/runtime/backward.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import {
  decodeRecord,
  encodeRecord,
  NodeFailedRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { NodeDataError } from "../../src/runtime/trace.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { FileRef, Type, TypeCheckError, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { engine, graph, NO_ENV, rejected, textReply } from "../helpers/run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
/** The default model for an agent without `llmConfig`. */
const DEFAULT_ENV: Readonly<Record<string, string>> = { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" };

/** Read an agent's `variables:` declarations from their YAML body. */
function variables(body: string, registry: TypeRegistry = new Map()): Parameter[] {
  const text = "variables:\n" + body.replace(/^(?=.)/gm, "  ");
  return parametersFromYaml(text, registry, { field: "variables", requireDefault: true });
}

/**
 * A model access whose model answers each request with the request's last message, the rendered
 * prompt; `prompts` holds each one and
 * `onCreate` runs when a model is built.
 */
function echo(onCreate: (config: Readonly<Record<string, unknown>>) => void = () => {}) {
  const prompts: string[] = [];
  const access: ModelAccess & { prompts: string[] } = {
    prompts,
    env: (name) => DEFAULT_ENV[name] ?? NO_ENV(name),
    createModel: (config): ChatModel => {
      onCreate(config);
      return {
        ...CONFIG,
        stream: (request, options) => {
          const prompt = request.messages.at(-1)!.text();
          prompts.push(prompt);
          const reply = new Message({ role: "assistant", parts: [new TextPart(prompt)] });
          return new ScriptedModel([new Response({ message: reply, finishReason: "stop" })], CONFIG).stream(
            request,
            options,
          );
        },
      };
    },
  };
  return access;
}

/**
 * Wrap the node's private `resolveVariables`: return the
 * record each call resolved, or run `replace` in its place.
 */
function spyResolve(node: AgentNode, replace?: () => ValueRecord): ValueRecord[] {
  const records: ValueRecord[] = [];
  const target = node as unknown as { resolveVariables: () => ValueRecord };
  const original = target.resolveVariables.bind(node);
  target.resolveVariables = () => {
    const record = (replace ?? original)();
    records.push(record);
    return record;
  };
  return records;
}

/**
 * Run `node` as the only node of a learnable flow over START `inputs` (each its parameter); the
 * run records the node's `llmConfig` (or `{}`) as its model configuration. Return the engine, the
 * node's accepted record and the model access.
 */
async function retainedAgent(node: AgentNode, inputs: Record<string, unknown> = {}, models = echo()) {
  node.params = Object.keys(inputs).map((name) => new Parameter(name));
  const flow = graph([node], [[START_ID, node.id]], {
    wiring: { [node.id]: Object.fromEntries(Object.keys(inputs).map((name) => [name, `\${input.${name}}`])) },
    outputs: [new FlowOutput("answer", `\${${node.id}.output}`)],
  });
  const built = engine(flow, {
    inputs,
    learnable: true,
    models,
    effectiveLlmByStaticPath: { [node.id]: { ...node.llmConfig } as JsonObject },
  });
  await built.run();
  const occurrence = built.state.trace!.occurrences.find((item) => item.nodeId === node.id)!;
  return { built, record: occurrence.record, occurrence, models };
}

/** A minimal concrete node; the base's `declaredVariables` is the subject. */
class FuncNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A class instance, which no `object` value may hold. */
class Extras {
  readonly fields: Record<string, unknown>;

  constructor(fields: Record<string, unknown>) {
    this.fields = fields;
  }
}

test("an object variable default holding a class instance is rejected", () => {
  // An `object` holds JSON values only, so an instance inside one fails before any run.
  const variable = () =>
    new Parameter("settings", {
      contract: new ValueContract(Type.scalar(ValueKind.OBJECT)),
      hasDefault: true,
      default: { nested: new Extras({ child: new Extras({ lost: 1.0 }) }) },
    });
  expect(() => new AgentNode("writer", { prompt: "ready", variables: [variable()] })).toThrow(TypeCheckError);
  expect(() => new AgentNode("writer", { prompt: "ready", variables: [variable()] })).toThrow("is not a JSON value");
});

test("declared variable inventory preserves canonical declarations", () => {
  const declarations = [
    new Parameter("tone", {
      contract: new ValueContract(Type.scalar(ValueKind.STRING)),
      hasDefault: true,
      default: "concise",
    }),
    new Parameter("count", {
      contract: new ValueContract(new Type({ kind: ValueKind.INTEGER, bounds: [1n, 12n] })),
      hasDefault: true,
      default: 4n,
    }),
  ];
  const node = new AgentNode("writer", { variables: declarations });

  const inventory = node.declaredVariables();

  expect(Array.isArray(inventory)).toBe(true);
  expect(inventory).not.toBe(node.variables);
  expect(inventory.length).toBe(2);
  inventory.forEach((actual, index) => expect(actual).toBe(declarations[index]!));
  expect(new FuncNode("ordinary").declaredVariables()).toEqual([]);
  expect(new AgentNode("empty").declaredVariables()).toEqual([]);
});

test("a retained agent's node data is its render locals, rendered again", async () => {
  const llmConfig = { provider: "openai", model: "chosen", openai: { temperature: 0.8 } };
  const node = new AgentNode("writer", {
    prompt: "${instruction}: ${topic}; ${count}/${count}",
    llmConfig,
    variables: variables("count: !float(0..4) 4\ninstruction: !str ${input.topic}\n"),
  });
  const resolved = spyResolve(node);

  const { built, occurrence, models } = await retainedAgent(node, { topic: "reefs" });
  expect(resolved).toHaveLength(1);

  // Learning renders the occurrence again, which resolves the variables again.
  const data = occurrence.nodeData;
  expect(data).toEqual({
    prompt_template: "${instruction}: ${topic}; ${count}/${count}",
    // The float variable's int default stays an int.
    variables: { count: 4n, instruction: "${input.topic}" },
    variable_domains: { count: "float(0..4)", instruction: "str" },
    rendered_prompt: "${input.topic}: reefs; 4/4",
    effective_llm_config: llmConfig,
  });
  expect(resolved).toHaveLength(2);
  expect(models.prompts).toEqual(["${input.topic}: reefs; 4/4"]);
  expect(occurrence.boundInputs).toEqual({ topic: "reefs" });
  expect(built.state.effectiveLlmByStaticPath["writer"]).toEqual(data["effective_llm_config"] as JsonObject);
});

test.each(["render", "empty"] as const)(
  "a retained agent distinguishes partial and empty render locals: %s",
  async (stage) => {
    const node = new AgentNode("writer", {
      prompt: stage === "empty" ? "" : "${note}",
      variables: stage === "empty" ? [] : variables("note: !Optional(str) null\n"),
    });

    const { record, occurrence, models } = await retainedAgent(node);

    const data = occurrence.nodeData;
    expect(data["prompt_template"]).toBe(node.prompt as string);
    expect(data["variable_domains"]).toEqual(stage === "empty" ? {} : { note: "Optional(str)" });
    expect(data["variables"]).toEqual(stage === "empty" ? {} : { note: null });
    expect(data["rendered_prompt"]).toBe(stage === "empty" ? "" : null);
    expect(occurrence.boundInputs).toEqual({});
    expect(models.prompts).toEqual(stage === "empty" ? [""] : []);
    if (stage !== "empty") {
      expect(record).toBeInstanceOf(NodeFailedRecord);
      expect((record as NodeFailedRecord).failure.code).toBe("node.execute");
    }
  },
);

/** A retained run of an agent whose variables fail to resolve with `expected`. */
async function unresolvable(expected: Error) {
  const node = new AgentNode("writer", { prompt: "${note}", variables: variables("note: !Optional(str) null\n") });
  spyResolve(node, () => {
    throw expected;
  });
  return await retainedAgent(node);
}

test("a retained agent whose variables fail to resolve fails the node, and its node data names the node", async () => {
  const expected = new TypeError("variable resolution failed");

  const { record, occurrence } = await unresolvable(expected);

  expect(record).toBeInstanceOf(NodeFailedRecord);
  expect((record as NodeFailedRecord).failure.code).toBe("node.execute");
  let error: unknown = null;
  try {
    void occurrence.nodeData;
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(NodeDataError);
  const { message, cause, nodeId, nodeStartedSeq, runId } = error as NodeDataError;
  expect(message).toBe("node 'writer': its node data can't be rebuilt: variable resolution failed");
  expect(cause).toBe(expected);
  expect([nodeId, nodeStartedSeq, runId]).toEqual(["writer", occurrence.source.executionSeq, occurrence.source.runId]);
});

test("an agent whose pre-assert failed is not rendered again, so its node data rebuilds", async () => {
  const node = new AgentNode("writer", { prompt: "${note}", variables: variables("note: !Optional(str) null\n") });
  node.preAsserts = [new Assertion("false", "never")];
  spyResolve(node, () => {
    throw new TypeError("variable resolution failed");
  });

  const { record, occurrence } = await retainedAgent(node);

  expect((record as NodeFailedRecord).failure.code).toBe("assertion.failed");
  expect(occurrence.nodeData).not.toHaveProperty("rendered_prompt");
});

test("backward over an agent whose node data can't be rebuilt finishes with a failure attributed to it", async () => {
  const { built, occurrence } = await unresolvable(new TypeError("variable resolution failed"));

  const outcome = await driveBackward(built.state, built.log, "Review.");

  expect(outcome.status).toBe("failed");
  expect(outcome.failure!.code).toBe("backward.execute");
  expect(outcome.failure!.message).toBe("node 'writer': its node data can't be rebuilt: variable resolution failed");
  expect(outcome.failure!.nodeId).toBe("writer");
  expect(outcome.failure!.nodeStartedSeq).toBe(occurrence.source.executionSeq);
  expect(outcome.failure!.runId).toBe(built.state.runId);
  expect(built.state.backwardState.finished!.outcome).toBe(outcome);
});

test.each([
  ["a colliding variable", { topic: "reefs" }, {}, { variable_domains: null, variables: null }],
  ["a missing value", {}, {}, { variable_domains: { tone: "str" }, variables: { tone: "concise" } }],
  ["a granted skill", { topic: "reefs" }, { skills: ["review"] }, { variable_domains: { tone: "str" }, variables: { tone: "concise" } }],
] as const)("the render capture keeps what rendered before a refusal: %s", (name, inputs, extra, expected) => {
  const node = new AgentNode("writer", {
    prompt: "${topic}: ${tone}",
    variables: variables(name === "a colliding variable" ? "topic: !str concise\n" : "tone: !str concise\n"),
    ...extra,
  });

  const capture = node.renderCapture(inputs);

  expect(capture["prompt_template"]).toBe("${topic}: ${tone}");
  expect(capture["variable_domains"]).toEqual(expected.variable_domains);
  expect(capture["variables"]).toEqual(expected.variables);
  expect(capture["rendered_prompt"]).toBe(name === "a granted skill" ? "reefs: concise" : null);
});

test("the render capture rethrows a render fault that isn't a refusal", () => {
  const node = new AgentNode("writer", { prompt: "${tone}", variables: variables("tone: !str concise\n") });
  const expected = new TypeError("variable resolution failed");
  spyResolve(node, () => {
    throw expected;
  });

  expect(() => node.renderCapture({})).toThrow(expected);
});

test("a retained agent canonicalizes a supported file variable without changing its default", async () => {
  const file = new FileRef({ uri: "file:///tmp/artifact.txt" });
  const node = new AgentNode("writer", {
    prompt: "Inspect the attachment.",
    variables: [
      new Parameter("artifact", {
        contract: new ValueContract(Type.scalar(ValueKind.FILE)),
        hasDefault: true,
        default: file,
      }),
    ],
  });

  const { record, occurrence } = await retainedAgent(node);

  expect(occurrence.nodeData["variables"]).toEqual({
    artifact: { uri: "file:///tmp/artifact.txt", mime: null, name: null },
  });
  expect(occurrence.nodeData["variable_domains"]).toEqual({ artifact: "file" });
  expect(node.variables[0]!.default).toBe(file);
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 3, record);
  expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
});

test.each(["domains", "copy"] as const)(
  "render capture preparation faults are capability faults carrying the original: %s",
  async (phase) => {
    const expected = new Error("capture preparation failed");
    const node = new AgentNode("writer", {
      prompt: "${settings}",
      variables: variables("settings: !object {items: [1]}\n"),
    });
    if (phase === "domains") {
      // Reading the declaration's contract fails.
      const settings = node.variables[0]!;
      node.variables = [
        Object.create(settings, {
          contract: {
            get() {
              throw expected;
            },
          },
        }) as Parameter,
      ];
    } else {
      // The resolved record can't be copied.
      spyResolve(node, () =>
        Object.defineProperty({}, "settings", {
          enumerable: true,
          get() {
            throw expected;
          },
        }),
      );
    }
    let error: unknown = null;
    try {
      node.renderCapture({});
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(CapabilityFault);
    expect((error as CapabilityFault).original).toBe(expected);
  },
);

test("a retained agent keeps early binding unavailable and capture off dormant", async () => {
  const node = new AgentNode("writer", { prompt: "${topic}", variables: variables("tone: !str concise\n") });
  node.params = [new Parameter("topic", { required: true })];
  const resolved = spyResolve(node);
  const models = echo();

  const missing = await evalNode(node, null, new VariablePool(), { captureBindings: true, models });
  expect(missing.outcome).toBeInstanceOf(EvaluationFailure);
  expect(missing.bindings!.boundInputs).toBeNull();
  expect(missing.backwardCapture!.variableNames).toEqual(["tone"]);
  expect(resolved).toEqual([]);

  const ordinary = await evalNode(node, null, new VariablePool(), { wiringOverride: { topic: "reefs" }, models });
  const retained = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { topic: "reefs" },
    captureBindings: true,
    models,
  });
  expect(ordinary.outcome).toEqual(new Output("reefs"));
  expect(retained.outcome).toEqual(ordinary.outcome);
  expect(ordinary.backwardCapture).toBeNull();
  expect(resolved).toHaveLength(2);
});

test("an agent renders inputs and variable defaults without execution or memory scope", async () => {
  const node = new AgentNode("writer", {
    prompt: "${instruction}: ${topic}, ${num_sentences} sentences, ${tone}.",
    variables: variables(
      'num_sentences: !int(1..12) 4\ntone: !Tone concise\ninstruction: !str "Summarize clearly"\n',
      readTypedefs({ Tone: "Literal(concise, detailed, technical)" }),
    ),
  });
  node.params = [new Parameter("topic")];
  const models = echo();
  expect(node.executionScope).toBeNull();
  expect(node.memoryScope).toBeNull();

  const output = await node.forward({ topic: "coral reefs" }, { models });

  expect((output as Output).value).toBe("Summarize clearly: coral reefs, 4 sentences, concise.");
  expect(models.prompts).toHaveLength(1);
});

test("an agent resolves and renders once with repeated literal default references", async () => {
  const node = new AgentNode("writer", {
    prompt: "${count}/${count}: ${instruction}/${empty}",
    variables: variables('count: !float(0..4) 4\ninstruction: !str ${input.topic}\nempty: !str ""\n'),
  });
  const resolved = spyResolve(node);
  const models = echo();

  expect(((await node.forward({}, { models })) as Output).value).toBe("4/4: ${input.topic}/");
  expect(resolved).toEqual([{ count: 4n, instruction: "${input.topic}", empty: "" }]);
  expect(models.prompts).toHaveLength(1);
});

test("mutable variable defaults are detached between invocations", async () => {
  const node = new AgentNode("writer", {
    prompt: "${settings}",
    variables: variables("settings: !object {nested: {items: [a, b]}}\n"),
  });
  const preset = node.variables[0]!.default as { nested: { items: string[] } };
  const resolved = spyResolve(node) as unknown as { settings: { nested: { items: string[] } } }[];
  const models = echo();

  const first = ((await node.forward({}, { models })) as Output).value;
  resolved[0]!.settings.nested.items.push("changed");
  const second = ((await node.forward({}, { models })) as Output).value;

  expect(second).toBe(first);
  expect(preset).toEqual({ nested: { items: ["a", "b"] } });
  expect(resolved[0]!.settings.nested).not.toBe(resolved[1]!.settings.nested);
  expect(resolved[1]!.settings.nested.items).not.toBe(preset.nested.items);
});

test("nullable variable defaults follow the existing strict prompt rules", async () => {
  const node = new AgentNode("writer", {
    prompt: "${null | 'fallback'}",
    variables: variables("note: !Optional(str) null\n"),
  });
  const resolved = spyResolve(node);
  const models = echo();

  expect(((await node.forward({}, { models })) as Output).value).toBe("fallback");
  expect(resolved).toEqual([{ note: null }]);
  node.prompt = "${note | 'fallback'}";
  expect(((await node.forward({}, { models })) as Output).value).toBe("fallback");
  node.prompt = "${note}";
  const error = await rejected(node.forward({}, { models }));
  expect(error).toBeInstanceOf(ExpressionError);
  // A present null is a value: the strict prompt refuses the null span, naming it.
  expect((error as Error).message).toBe(
    'null value in strict template: ${note} is null; give it a default, e.g. ${note :- "none"}',
  );
  expect(models.prompts).toHaveLength(2);
  expect(node.variables[0]!.default).toBeNull();
});

/** What learning gives `learningMaterial`: no skills or tools, no model, and `recall` for recall. */
function materialCaps(recall: () => Promise<MemoryContext> = async () => new MemoryContext()): MaterialCaps {
  return { skills: new Map(), toolSpecs: () => [], models: null, openMemoryContext: recall };
}

test("learning: a variable colliding with an input refuses the rebuild; another RangeError is a fault", async () => {
  const node = new AgentNode("writer", { prompt: "${tone}", variables: variables("tone: !str concise\n") });
  const steps = [{ kind: "model_call" }];

  const refused = await node.learningMaterial({ tone: "input" }, steps, materialCaps());
  expect(refused.prompt).toBeNull();
  expect(refused.digestOk).toBe(false);

  const fault = new RangeError("broken recall");
  const failing = materialCaps(async () => {
    throw fault;
  });
  expect(await rejected(node.learningMaterial({}, steps, failing))).toBe(fault);
});

test.each([
  ["a delivered input", [], { tone: "input" }],
  ["a declared input", [new Parameter("tone")], {}],
] as const)("a direct variable and input overlap is rejected before render or run: %s", async (_label, declared, inputs) => {
  const node = new AgentNode("writer", { prompt: "${tone}", variables: variables("tone: !str concise\n") });
  node.params = [...declared];
  const resolved = spyResolve(node);
  const models = echo();

  const error = await rejected(node.forward({ ...inputs }, { models }));

  expect(error).toEqual(new RangeError("variable 'tone' collides with an input name"));
  expect(resolved).toEqual([]);
  expect(models.prompts).toEqual([]);
});

test("an ordinary run keeps variables out of wiring and pool and backward dormant", async () => {
  const flow = (
    await loadFlow(`id: local_render
name: local_render
input:
  topic: !str
nodes:
  writer:
    kind: agent
    input: {topic: '\${input.topic}'}
    variables:
      instruction: !str \${absent.output}
      tone: !str concise
    prompt: '\${topic}: \${tone}, \${instruction}'
    output: str
output: \${writer.output}
`)
  ).compiled;

  expect([...flow.wiring.get("writer")!.keys()]).toEqual(["topic"]);
  expect(flow.nodes.get("writer")!.params!.map((parameter) => parameter.name)).toEqual(["topic"]);
  expect(flow.edges.every((edge) => edge.from !== "absent")).toBe(true);
  const built = engine(flow, { inputs: { topic: "reefs" }, learnable: false, models: echo() });
  await built.run();

  const expected = "reefs: concise, ${absent.output}";
  expect(built.state.pool.get(END_ID)).toBe(expected);
  expect(new Set(built.state.pool.store.keys())).toEqual(new Set([START_ID, "writer", "writer.new_messages", END_ID]));
  expect(built.state.pool.get(START_ID)).toEqual({ topic: "reefs" });
  expect(built.state.pool.get("writer")).toBe(expected);
  expect(built.state.trace).toBeNull();
});

test("a temperature variable renders independently of the effective model configuration", async () => {
  const loaded = await loadFlow(`id: variable_temperature
name: variable_temperature
input: {topic: !str }
llm_config: {provider: ollama, model: test, ollama: {options: {temperature: 0.8}}}
nodes:
  writer:
    kind: agent
    input: {topic: '\${input.topic}'}
    variables:
      temperature: !float(0..1) 0.4
    prompt: '\${topic}: \${temperature}'
    output: str
output: \${writer.output}
`);
  const dir = mkdtempSync(join(tmpdir(), "yian-agent-variables-"));
  const model = new ScriptedModel([textReply("answer")]);
  const configurations: Readonly<Record<string, unknown>>[] = [];
  const scopes: unknown[][] = [];
  const target = AgentNode.prototype as unknown as { resolveVariables: (this: AgentNode) => ValueRecord };
  const resolve = target.resolveVariables;
  const spy = spyOn(target, "resolveVariables").mockImplementation(function (this: AgentNode) {
    scopes.push([this.executionScope, this.memoryScope]);
    return resolve.call(this);
  });
  try {
    const result = await runFlow(loaded, { topic: "reefs" }, {
      learnable: false,
      env: hostEnv(dir),
      models: {
        env: hostEnv(dir),
        createModel: (config) => {
          configurations.push(config);
          return model;
        },
      },
    });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("answer");
    expect(result.learning).toBeNull();
    expect(configurations).toEqual([{ provider: "ollama", model: "test", ollama: { options: { temperature: 0.8 } } }]);
    expect(model.requests.at(-1)!.messages.at(-1)!.text()).toBe("reefs: 0.4");
    expect(model.requests.length).toBe(1);
    expect(scopes).toEqual([[null, null]]);
  } finally {
    spy.mockRestore();
    rmSync(dir, { recursive: true, force: true });
  }
});
