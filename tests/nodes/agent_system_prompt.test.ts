/**
 * An authored AGENT `system_prompt:` replaces `DEFAULT_SYSTEM` on every model path, reads the
 * node's variables, and is replayed and fingerprinted on resume.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext, MemoryScope, MemorySection, type RecallContext } from "../../src/memory/contracts.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { systemPromptWithMemory } from "../../src/nodes/agent/memory.ts";
import { type AgentPrompt, AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { advance, runFlow } from "../../src/run.ts";
import { EvaluationFailure, evalNode, type EvalNodeOptions } from "../../src/runtime/eval_node.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { systemPromptWithSkills } from "../../src/skills/tool.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { parametersFromYaml } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { crashAfter, editRecordedFlow, nodeFlow, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

/** The node's private render, which `forward` and `learningMaterial` share. */
function renderPrompt(
  node: AgentNode,
  inputs: Readonly<Record<string, unknown>>,
  skills: ReadonlyMap<string, Skill> | null,
): AgentPrompt {
  type Inputs = Readonly<Record<string, unknown>>;
  type Render = { renderPrompt(inputs: Inputs, skills: ReadonlyMap<string, Skill> | null): AgentPrompt };
  return (node as unknown as Render).renderPrompt(inputs, skills);
}

const dir = mkdtempSync(join(tmpdir(), "yian-agent-system-prompt-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const TEMPLATE = "You are ${name}, ${title}. Speak as ${name} in the first person.";
const RENDERED = "You are Ada, CTO. Speak as Ada in the first person.";

/** A model replying each of `texts` in turn. */
function scripted(...texts: string[]): ScriptedModel {
  return new ScriptedModel(texts.map(textReply), CONFIG);
}

/** A run tool set holding `noop`, which takes no arguments. */
function noopTools(): ToolSet {
  const parametersJsonSchema = { type: "object", properties: {} };
  return new ToolSet({ noop: new FunctionTool({ name: "noop", description: "Do nothing.", parametersJsonSchema, execute: () => "ok" }) });
}

/**
 * An AGENT `agent` whose system prompt (`TEMPLATE` by default) and prompt read inputs `name` and
 * `title`, granted `tools`.
 */
function agent(
  options: { structured?: boolean; systemPrompt?: string; variables?: Parameter[]; tools?: string[] } = {},
): AgentNode {
  const { structured = false, systemPrompt = TEMPLATE, variables, tools } = options;
  const node = new AgentNode("agent", { prompt: "Hello from ${name}", systemPrompt, llmConfig: CONFIG, variables, tools });
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

test.each([
  [false, false],
  [true, false],
  [false, true],
  [true, true],
])("every path sends the rendered system prompt (structured: %p, tools: %p)", async (structured, tools) => {
  const answers = structured ? ['{"value": 7}'] : ["done"];
  // With tools the loop asks first; the structured value then comes in its own request.
  const model = scripted(...(tools && structured ? ["done", ...answers] : answers));
  const node = agent({ structured, tools: tools ? ["noop"] : undefined });

  const evaluated = await evaluate(node, { models: serving(model), tools: noopTools() });

  expect(evaluated.outcome).toEqual(new Output(structured ? 7n : "done"));
  expect(systems(model)).toEqual(Array(model.requests.length).fill(RENDERED));
  expect(model.requests).toHaveLength(tools && structured ? 2 : 1);
  const user = model.requests[0]!.messages[1]!;
  expect([user.role, user.text()]).toEqual(["user", "Hello from Ada"]);
  expect(renderPrompt(node, { name: "Ada", title: "CTO" }, null).baseSystemPrompt).toBe(RENDERED);
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
  const memoryContextOpener = async (_scope: MemoryScope, recall: RecallContext) => {
    recalled.push(recall.baseSystemPrompt);
    return new MemoryContext({ sections });
  };

  const evaluated = await evaluate(node, { models: serving(model), memoryContextOpener });

  expect(evaluated.outcome).toEqual(new Output("done"));
  expect(recalled).toEqual([RENDERED]);
  expect(systems(model)).toEqual([systemPromptWithMemory(RENDERED, sections)]);
  expect(systems(model)[0]!.startsWith(RENDERED + "\n\nMemory context")).toBe(true);
});

test("skills append after the authored prompt and recall skips them", async () => {
  const model = scripted("done");
  const skill = new Skill({ name: "review", description: "Review code changes.", instructions: "Read the diff." });
  const seen: string[] = [];
  const openMemoryContext = async (system: string) => {
    seen.push(system);
    return new MemoryContext();
  };
  const node = new AgentNode("agent", { prompt: "p", systemPrompt: "You review code.", llmConfig: CONFIG, skills: ["review"] });

  const skills = new Map([["review", skill]]);
  expect(await node.forward({}, { openMemoryContext, skills, models: serving(model) })).toEqual(new Output("done"));

  expect(seen).toEqual(["You review code."]);
  expect(systems(model)).toEqual([systemPromptWithSkills("You review code.", [skill])]);
});

test.each([
  ["a text prompt", "Hi ${name}" as const],
  ["a list prompt", [{ splice: "history" }, { role: "user", text: "Now: ${name}" }] as const],
])("the render returns what forward sends (%s)", async (_label, prompt) => {
  const model = scripted("done");
  const skill = new Skill({ name: "review", description: "Review code changes.", instructions: "Read the diff." });
  const skills = new Map([["review", skill]]);
  const node = new AgentNode("agent", { prompt, systemPrompt: "You are ${name}.", llmConfig: CONFIG, skills: ["review"] });
  node.params = [new Parameter("name", { required: true }), new Parameter("history", { required: false })];
  const inputs = { name: "Ada", history: [{ role: "user", text: "earlier" }, { role: "assistant", text: "noted" }] };

  expect(await node.forward(inputs, { skills, models: serving(model) })).toEqual(new Output("done"));
  const rendered = renderPrompt(node, inputs, skills);

  const [system, ...sent] = model.requests[0]!.messages;
  expect(rendered.baseSystemPrompt).toBe(system!.text());
  expect(rendered.baseSystemPrompt).toBe(systemPromptWithSkills("You are Ada.", [skill]));
  expect(rendered.systemPrompt).toBe("You are Ada.");
  expect(rendered.modelMessages().map((message) => message.toJSON())).toEqual(sent.map((message) => message.toJSON()));
  expect(rendered.promptText).toBe(sent.at(-1)!.text());
  expect(rendered.skills).toEqual([skill]);
});

test.each([false, true])("an omitted system prompt keeps DEFAULT_SYSTEM (tools: %p)", async (tools) => {
  const model = scripted("done");
  const seen: string[] = [];
  const openMemoryContext = async (system: string) => {
    seen.push(system);
    return new MemoryContext();
  };
  const node = new AgentNode("agent", { prompt: "p", llmConfig: CONFIG, tools: tools ? ["noop"] : undefined });

  const caps = { openMemoryContext, models: serving(model), tools: noopTools() };
  expect(await node.forward({}, caps)).toEqual(new Output("done"));

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

// --- through YAML, the run log, and recovery --- //

const CHAT = `\
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
input:
  name: !str
  title: !str
  message: !str
nodes:
  reply:
    kind: agent
    input: {name: "\${input.name}", title: "\${input.title}", message: "\${input.message}"}
    system_prompt: "SYSTEM"
    prompt: \${message}
    output: str
  after:
    kind: agent
    input: {draft: "\${reply.output}"}
    prompt: "polish: \${draft}"
    output: str
output: \${after.output}
`;
const INPUTS = { name: "Ada", title: "CTO", message: "How are we doing?" };
// The records before `reply`'s output and before `after`'s: the run start (RunStarted and four
// commits), START's start, commit and success, then each agent's start (and `reply`'s commit and
// success before `after`'s start).
const BEFORE_REPLY_SUCCESS = 9;
const BEFORE_AFTER_SUCCESS = 12;

/** Load the chat flow with `template` as `reply`'s system prompt. */
function chatFlow(template: string = TEMPLATE) {
  return loadFlow(CHAT.replace('"SYSTEM"', JSON.stringify(template)));
}

/** Run options serving `model` in `hostEnv`. */
function served(model: ScriptedModel) {
  return { models: serving(model), env: hostEnv(dir) };
}

test("run flow sends the rendered persona", async () => {
  const model = scripted("draft", "final");
  const result = await runFlow(await chatFlow(), INPUTS, served(model));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("final");
  expect(systems(model)).toEqual([RENDERED, DEFAULT_SYSTEM]);
});

test("resume replays a finished node with its recorded system prompt", async () => {
  const path = join(dir, "replays.jsonl");
  const flow = await chatFlow();
  await crashAfter(BEFORE_AFTER_SUCCESS, () =>
    runFlow(flow, INPUTS, { log: path, ...served(scripted("draft", "lost")) }),
  );

  const model = scripted("final");
  const resumed = await advance({ log: path, ...served(model) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("final");
  expect(model.requests.map((request) => request.messages.at(-1)!.text())).toEqual(["polish: draft"]);
});

test("retry after a mid node crash renders the same system prompt", async () => {
  const path = join(dir, "retry.jsonl");
  const flow = await chatFlow();
  const first = scripted("lost");
  await crashAfter(BEFORE_REPLY_SUCCESS, () => runFlow(flow, INPUTS, { log: path, ...served(first) }));
  expect(systems(first)[0]).toBe(RENDERED);

  const retry = scripted("draft", "final");
  const resumed = await advance({ log: path, ...served(retry) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("final");
  expect(systems(retry)[0]).toBe(RENDERED);
});

test("resume refuses a flow whose system prompt changed", async () => {
  const path = join(dir, "changed.jsonl");
  await crashAfter(BEFORE_AFTER_SUCCESS, async () =>
    runFlow(await chatFlow(), INPUTS, { log: path, ...served(scripted("draft", "lost")) }),
  );

  // Resume compiles the recorded flow, so the change is to the recorded text.
  editRecordedFlow(path, (text) => text.replace(JSON.stringify(TEMPLATE), JSON.stringify("You are ${name}.")));
  const error = await rejected(advance({ log: path, ...served(scripted()) }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("flow fingerprint mismatch");
});
