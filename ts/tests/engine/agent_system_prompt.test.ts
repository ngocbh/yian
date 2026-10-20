/**
 * An authored AGENT `system_prompt:` replaces `DEFAULT_SYSTEM` on every model path: mirrors
 * `tests/engine/test_agent_system_prompt.py`.
 *
 * Ported through `evalNode` with a `ScriptedModel` from `models`: the tool-free rows of
 * `test_every_path_sends_the_rendered_system_prompt` (plain and structured),
 * `test_system_prompt_reads_the_node_variables`, `test_memory_sections_append_after_the_authored_prompt`
 * (a hand-made memory context opener) and `test_a_null_system_prompt_value_fails_the_node_before_any_model_call`;
 * the tool-free row of `test_omitted_system_prompt_keeps_default_system` through `forward`.
 *
 * Changed: the bound inputs come from a one-node `CompiledFlow`'s wiring over a pool, where Python
 * passes a stub flow; "no model call" reads as the model access building no model.
 *
 * Later (T5-11b), the tool loop: the tool rows of `test_every_path_sends_the_rendered_system_prompt`
 * and `test_omitted_system_prompt_keeps_default_system`, and
 * `test_skills_append_after_the_authored_prompt_and_recall_skips_them` (a skill grant runs it).
 *
 * Later (T4-8, T3-22), through `loadFlow`, `runFlow` and `resumeFlow`:
 * `test_run_flow_sends_the_rendered_persona`,
 * `test_resume_replays_a_finished_node_with_its_recorded_system_prompt`,
 * `test_retry_after_a_mid_node_crash_renders_the_same_system_prompt` and
 * `test_resume_refuses_a_flow_whose_system_prompt_changed`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  MemoryContext,
  type MemoryContextRequest,
  MemoryScope,
  MemorySection,
} from "../../src/memory/contracts.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { systemPromptWithMemory } from "../../src/nodes/agent/memory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure, evalNode, type EvalNodeOptions } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import { nodeFlow, serving, textReply } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const TEMPLATE = "You are ${name}, ${title}. Speak as ${name} in the first person.";
const RENDERED = "You are Ada, CTO. Speak as Ada in the first person.";

/** A model replying each of `texts` in turn. */
function scripted(...texts: string[]): ScriptedModel {
  return new ScriptedModel(texts.map(textReply), CONFIG);
}

/** An AGENT `agent` whose system prompt (`TEMPLATE` by default) and prompt read inputs `name` and `title`. */
function agent(options: { structured?: boolean; systemPrompt?: string; variables?: Parameter[] } = {}): AgentNode {
  const { structured = false, systemPrompt = TEMPLATE, variables } = options;
  const node = new AgentNode("agent", { prompt: "Hello from ${name}", systemPrompt, llmConfig: CONFIG, variables });
  node.params = [new Parameter("name", { required: true }), new Parameter("title", { required: true })];
  if (structured) node.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  return node;
}

/** Evaluate `node` with `name=Ada` and `title=CTO` bound from the pool. */
function evaluate(node: AgentNode, options: EvalNodeOptions = {}) {
  const pool = new VariablePool();
  pool.set("person", { name: "Ada", title: "CTO" });
  const flow = nodeFlow(node, { name: "${person.output.name}", title: "${person.output.title}" });
  return evalNode(node, flow, pool, options);
}

/** The system text of each request `model` received. */
function systems(model: ScriptedModel): string[] {
  return model.requests.map((request) => request.messages[0]!.text());
}

test.each([false, true])("every tool-free path sends the rendered system prompt (structured: %p)", async (structured) => {
  const model = scripted(structured ? '{"value": 7}' : "done");

  const evaluated = await evaluate(agent({ structured }), { models: serving(model), captureEvidence: true });

  expect(evaluated.outcome).toEqual(new Output(structured ? 7n : "done"));
  expect(systems(model)).toEqual([RENDERED]);
  const user = model.requests[0]!.messages[1]!;
  expect([user.role, user.text()]).toEqual(["user", "Hello from Ada"]);
  expect(evaluated.evidence!.baseSystemPrompt).toBe(RENDERED);
});

test("the system prompt reads the node variables", async () => {
  const model = scripted("done");
  const variables = parametersFromYaml("variables:\n  tone: !str warm\n", new Map(), {
    field: "variables",
    requireDefault: true,
  });
  await evaluate(agent({ systemPrompt: "${name}, be ${tone}.", variables }), { models: serving(model) });
  expect(systems(model)).toEqual(["Ada, be warm."]);
});

test("memory sections append after the authored prompt", async () => {
  const model = scripted("done");
  const node = agent();
  node.memoryScope = new MemoryScope(["flow", "agent"], { memories: ["outer", "local"] });
  const sections = [new MemorySection("outer", "outer facts"), new MemorySection("local", "local facts")];
  const recalled: string[] = [];
  const memoryContextOpener = async (_scope: MemoryScope, request: MemoryContextRequest) => {
    recalled.push(request.recall.baseSystemPrompt);
    return new MemoryContext({ sections });
  };

  const evaluated = await evaluate(node, { models: serving(model), captureEvidence: true, memoryContextOpener });

  expect(evaluated.outcome).toEqual(new Output("done"));
  expect(recalled).toEqual([RENDERED]);
  expect(systems(model)).toEqual([systemPromptWithMemory(RENDERED, sections)]);
  expect(systems(model)[0]!.startsWith(RENDERED + "\n\nMemory context")).toBe(true);
  expect(evaluated.evidence!.baseSystemPrompt).toBe(RENDERED);
  expect(evaluated.evidence!.memorySections).toEqual(sections);
});

test("an omitted system prompt keeps DEFAULT_SYSTEM", async () => {
  const model = scripted("done");
  const seen: string[] = [];
  const openMemoryContext = async (system: string) => {
    seen.push(system);
    return new MemoryContext();
  };
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG });

  expect(await node.forward({}, { openMemoryContext, models: serving(model) })).toEqual(new Output("done"));

  expect(seen).toEqual([DEFAULT_SYSTEM]);
  expect(systems(model)).toEqual([DEFAULT_SYSTEM]);
});

test("a null system prompt value fails the node before any model call", async () => {
  const node = new AgentNode("agent", { prompt: "p", systemPrompt: "You are ${name}.", llmConfig: CONFIG });
  node.params = [new Parameter("name", { required: true })];
  const pool = new VariablePool();
  pool.set("person", null);
  const models = serving(scripted());

  const evaluated = await evalNode(node, nodeFlow(node, { name: "${person.output}" }), pool, { models });

  expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
  expect(models.built).toBe(0);
});
