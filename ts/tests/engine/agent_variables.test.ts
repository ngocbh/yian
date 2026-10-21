/**
 * AGENT local variables: mirrors `tests/engine/test_agent_variables.py`.
 *
 * Ported: `test_an_object_variable_default_holding_a_model_is_rejected`,
 * `test_declared_variable_inventory_preserves_canonical_declarations`, the retained render capture
 * tests through a learnable `FlowEngine` run (`test_retained_agent_captures_actual_render_locals_once`,
 * `test_retained_agent_distinguishes_partial_and_empty_render_locals`,
 * `test_retained_agent_canonicalizes_supported_file_variable_without_changing_default`),
 * `test_retained_agent_keeps_early_binding_unavailable_and_capture_off_dormant` through `evalNode`,
 * and the rendering tests through `forward`
 * (`test_agent_renders_inputs_and_variable_defaults_without_execution_or_memory_scope`,
 * `test_agent_resolves_and_renders_once_with_repeated_literal_default_references`,
 * `test_mutable_variable_defaults_are_detached_between_invocations`,
 * `test_nullable_variable_defaults_follow_existing_strict_prompt_rules`,
 * `test_direct_variable_input_overlap_is_rejected_before_render_or_run`).
 *
 * Changed: the object default holds a class instance where Python's holds a pydantic model; the
 * first tests build their declarations by hand where Python reads them from YAML; the base `Node`
 * is abstract, so a minimal concrete node stands in for `Node("ordinary")`. The node rows' model
 * echoes the rendered prompt, for Python's patched `run_agent`, and its environment sets the
 * default model Python's test configuration sets. TS can't replace a module function, so a
 * spy on the node's private `resolveVariables` stands in for the patched `apply_defaults` and
 * `_resolve_variables`: it counts resolutions and returns the resolved records (the render record's
 * values), and no row counts renders; "nothing rendered" reads as nothing resolved and no model
 * call. The learnable run records the node's `llmConfig` as its model map by hand, where Python's
 * run start derives it. The float variable's int default is a `bigint` (Python: `type is int`).
 * `test_retained_render_locals_are_detached_before_render_and_model_callbacks`: only the model
 * build mutates (the node's prompt and the config it gets); TS can't patch the render, a
 * `Parameter`'s default is frozen, and the captured locals are frozen, so the reader's write throws.
 * `test_render_capture_preparation_faults_escape_with_original_host_identity`: the domain fault is
 * a declaration whose `contract` read throws (for Python's patched `canonical_type_label`), the
 * copy fault a resolved record whose field read throws (for Python's patched `deepcopy`).
 * A variable colliding with an input is a `RangeError` with its full message.
 *
 * Ported through `loadFlow` and a `FlowEngine` run (`engine`), as Python's `_engine`:
 * `test_ordinary_run_keeps_variables_out_of_wiring_and_pool_and_backward_dormant` (the echoing
 * model stands in for the patched `run_agent`; `AgentNode.backward` throws until it is ported, so a
 * backward call would fail the run, for Python's patched `backward`).
 *
 * Later (T4-8), through the loader and `runFlow` (the flow's `llm_config` cascade is the subject):
 * `test_temperature_variable_renders_independently_of_effective_model_configuration`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, NodeKind, Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import {
  decodeRecord,
  encodeRecord,
  NodeFailedRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { FileRef, Type, TypeCheckError, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import { engine, graph, rejected } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
/** The default model Python's test configuration sets, for an agent without `llmConfig`. */
const DEFAULT_ENV: Readonly<Record<string, string>> = { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" };

/** Read an agent's `variables:` declarations from their YAML body. */
function variables(body: string, registry: TypeRegistry = new Map()): Parameter[] {
  const text = "variables:\n" + body.replace(/^(?=.)/gm, "  ");
  return parametersFromYaml(text, registry, { field: "variables", requireDefault: true });
}

/**
 * A model access whose model answers each request with the request's last message, the rendered
 * prompt, as Python's patched `run_agent` returns `ctx.prompt`; `prompts` holds each one and
 * `onCreate` runs when a model is built.
 */
function echo(onCreate: (config: Readonly<Record<string, unknown>>) => void = () => {}) {
  const prompts: string[] = [];
  const access: ModelAccess & { prompts: string[] } = {
    prompts,
    env: (name) => DEFAULT_ENV[name],
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
 * Wrap the node's private `resolveVariables` (Python patches `_resolve_variables`): return the
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
  return { built, record: occurrence.record, models };
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

test("a retained agent captures its actual render locals once", async () => {
  const llmConfig = { provider: "openai", model: "chosen", openai: { temperature: 0.8 } };
  const node = new AgentNode("writer", {
    prompt: "${instruction}: ${topic}; ${count}/${count}",
    llmConfig,
    variables: variables("count: !float(0..4) 4\ninstruction: !str ${input.topic}\n"),
  });
  const resolved = spyResolve(node);

  const { built, record, models } = await retainedAgent(node, { topic: "reefs" });

  const data = record.forwardCapture!.nodeData!;
  expect(data).toEqual({
    prompt_template: "${instruction}: ${topic}; ${count}/${count}",
    // The float variable's int default stays an int.
    variables: { count: 4n, instruction: "${input.topic}" },
    variable_domains: { count: "float(0..4)", instruction: "str" },
    rendered_prompt: "${input.topic}: reefs; 4/4",
    effective_llm_config: llmConfig,
  });
  expect(resolved).toHaveLength(1);
  expect(models.prompts).toEqual(["${input.topic}: reefs; 4/4"]);
  expect(record.forwardCapture!.boundInputs).toEqual({ topic: "reefs" });
  expect(built.state.effectiveLlmByStaticPath["writer"]).toEqual(data["effective_llm_config"] as JsonObject);
});

test("retained render locals are detached before the model call", async () => {
  const node = new AgentNode("writer", {
    prompt: "${settings}",
    llmConfig: { provider: "test", model: "original" },
    variables: variables("settings: !object {items: [original]}\n"),
  });
  const models = echo((config) => {
    node.prompt = "different template";
    (config as Record<string, unknown>)["model"] = "different model";
  });

  const { built, record } = await retainedAgent(node, {}, models);

  const data = record.forwardCapture!.nodeData! as Record<string, unknown>;
  expect(data["variables"]).toEqual({ settings: { items: ["original"] } });
  expect(data["prompt_template"]).toBe("${settings}");
  expect(data["effective_llm_config"]).toEqual({ provider: "test", model: "original" });
  expect(built.state.effectiveLlmByStaticPath["writer"]).toEqual({ provider: "test", model: "original" });
  const items = (data["variables"] as { settings: { items: string[] } }).settings.items;
  expect(() => items.push("reader mutation")).toThrow(TypeError);
  expect(record.forwardCapture!.nodeData!["variables"]).toEqual({ settings: { items: ["original"] } });
});

test.each(["resolve", "render", "empty"] as const)(
  "a retained agent distinguishes partial and empty render locals: %s",
  async (stage) => {
    const node = new AgentNode("writer", {
      prompt: stage === "empty" ? "" : "${note}",
      variables: stage === "empty" ? [] : variables("note: !Optional(str) null\n"),
    });
    if (stage === "resolve") {
      spyResolve(node, () => {
        throw new RangeError("variable resolution failed");
      });
    }

    const { record, models } = await retainedAgent(node);

    const data = record.forwardCapture!.nodeData!;
    expect(data["prompt_template"]).toBe(node.prompt);
    expect(data["variable_domains"]).toEqual(stage === "empty" ? {} : { note: "Optional(str)" });
    expect(data["variables"]).toEqual(stage === "resolve" ? null : stage === "empty" ? {} : { note: null });
    expect(data["rendered_prompt"]).toBe(stage === "empty" ? "" : null);
    expect(record.forwardCapture!.boundInputs).toEqual({});
    expect(models.prompts).toEqual(stage === "empty" ? [""] : []);
    if (stage !== "empty") {
      expect(record).toBeInstanceOf(NodeFailedRecord);
      expect((record as NodeFailedRecord).failure.code).toBe("node.execute");
    }
  },
);

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

  const { record } = await retainedAgent(node);

  expect(record.forwardCapture!.nodeData!["variables"]).toEqual({
    artifact: { uri: "file:///tmp/artifact.txt", mime: null, name: null },
  });
  expect(record.forwardCapture!.nodeData!["variable_domains"]).toEqual({ artifact: "file" });
  expect(node.variables[0]!.default).toBe(file);
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 3, record);
  expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
});

test.each(["domains", "copy"] as const)(
  "render capture preparation faults escape with the original host identity: %s",
  async (phase) => {
    const expected = new Error("capture preparation failed");
    const node = new AgentNode("writer", {
      prompt: "${settings}",
      variables: variables("settings: !object {items: [1]}\n"),
    });
    if (phase === "domains") {
      // Reading the declaration's contract fails, for Python's patched `canonical_type_label`.
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
      // The resolved record can't be copied, for Python's patched `deepcopy`.
      spyResolve(node, () =>
        Object.defineProperty({}, "settings", {
          enumerable: true,
          get() {
            throw expected;
          },
        }),
      );
    }
    const models = echo();

    const error = await rejected(evalNode(node, null, new VariablePool(), { captureBindings: true, models }));

    expect(error).toBe(expected);
    expect(models.prompts).toEqual([]);
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
  expect(missing.backwardCapture!.nodeData).toEqual({});
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
  expect(retained.backwardCapture!.nodeData["effective_llm_config"]).toBeNull();
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

  expect(output.value).toBe("Summarize clearly: coral reefs, 4 sentences, concise.");
  expect(models.prompts).toHaveLength(1);
});

test("an agent resolves and renders once with repeated literal default references", async () => {
  const node = new AgentNode("writer", {
    prompt: "${count}/${count}: ${instruction}/${empty}",
    variables: variables('count: !float(0..4) 4\ninstruction: !str ${input.topic}\nempty: !str ""\n'),
  });
  const resolved = spyResolve(node);
  const models = echo();

  expect((await node.forward({}, { models })).value).toBe("4/4: ${input.topic}/");
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

  const first = (await node.forward({}, { models })).value;
  resolved[0]!.settings.nested.items.push("changed");
  const second = (await node.forward({}, { models })).value;

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

  expect((await node.forward({}, { models })).value).toBe("fallback");
  expect(resolved).toEqual([{ note: null }]);
  node.prompt = "${note | 'fallback'}";
  expect((await node.forward({}, { models })).value).toBe("fallback");
  node.prompt = "${note}";
  const error = await rejected(node.forward({}, { models }));
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toMatch(/unresolved reference \$\{note\}/);
  expect(models.prompts).toHaveLength(2);
  expect(node.variables[0]!.default).toBeNull();
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
  expect(new Set(built.state.pool.store.keys())).toEqual(new Set([START_ID, "writer", END_ID]));
  expect(built.state.pool.get(START_ID)).toEqual({ topic: "reefs" });
  expect(built.state.pool.get("writer")).toBe(expected);
  expect(built.state.trace).toBeNull();
});
