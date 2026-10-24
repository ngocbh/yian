/**
 * Learning reads an AGENT occurrence's steps, render and recall (`learningMaterials`), and what it
 * builds from them for AGENT backward (`evaluationEvidence`) and memory episodes. TS-only.
 *
 * Each run is learnable and logged; its log is replayed, and the material of each occurrence is
 * rebuilt from the replayed trace. Interactions follow step order: a reply's calls, then their
 * results in call order, so two unsafe calls read A, B, result A, result B. A grant naming a tool
 * group is expanded with the groups run_started records, so a finished run reopened learns the same.
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ReplayDegraded } from "../../src/events.ts";
import { sourceKey } from "../../src/expr/lineage.ts";
import type { Response } from "../../src/llm/messages.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  MemorySection,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import { advance, replayRunLog, runFlow } from "../../src/run.ts";
import { learningMaterials } from "../../src/runtime/backward.ts";
import { JsonlRunLog } from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { evaluationEvidence } from "../../src/runtime/trace.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { LOAD_SKILL, systemPromptWithSkills } from "../../src/skills/tool.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { logJson } from "../../src/typesys/json.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { callReply, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-learning-from-steps-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let count = 0;

/** A tool taking a `topic`, answering `prefix:<topic>`; concurrency-safe when `safe`. */
function topicTool(name: string, prefix: string, safe = false): FunctionTool {
  return new FunctionTool({
    name,
    description: `${name} a topic.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `${prefix}:${String(args["topic"])}`,
    isConcurrencySafe: safe ? () => true : null,
  });
}

/**
 * Run `text` learnably with a model replying `replies` (and `memory` under key `test`), then
 * replay its log and rebuild every occurrence's material.
 */
async function rebuilt(
  text: string,
  replies: Response[],
  options: {
    tools?: Record<string, FunctionTool>;
    memory?: FakeMemory;
    provider?: string;
    model?: string;
    skills?: ReadonlyMap<string, Skill>;
  } = {},
) {
  const path = join(dir, `run-${++count}.jsonl`);
  const flow: ComposedFlow = await loadFlow(text);
  const model = new ScriptedModel(replies, {
    provider: options.provider ?? "anthropic",
    model: options.model ?? "claude-test",
  });
  const registry = options.memory === undefined ? null : new Map([["test", options.memory]]);
  const env = hostEnv(dir);
  const result = await runFlow(flow, {}, {
    log: path,
    learnable: true,
    models: serving(model),
    tools: options.tools ?? {},
    memory: registry,
    skills: options.skills ?? null,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return await materialsOf(flow, path, registry, model);
}

/**
 * Replay the log at `path` and rebuild its occurrences' material, with the reports it makes;
 * `model` is served, or a `ModelAccess` stands in for it.
 */
async function materialsOf(
  flow: ComposedFlow,
  path: string,
  registry: Map<string, FakeMemory> | null,
  model: ScriptedModel | ModelAccess,
) {
  const log = JsonlRunLog.resume(path);
  let state;
  try {
    state = replayRunLog(flow, log);
  } finally {
    log[Symbol.dispose]();
  }
  const memories = state.memories ?? [];
  const memory =
    registry === null || memories.length === 0
      ? null
      : await MemoryRuntime.open({
          memories,
          origins: new Map(memories.map((spec) => [spec.key, null])),
          memory: registry,
          snapshots: state.memorySnapshots,
        });
  const events: unknown[] = [];
  try {
    const materials = await learningMaterials(state, {
      memory,
      models: model instanceof ScriptedModel ? serving(model) : model,
      onEvent: (event) => events.push(event),
    });
    const trace = state.trace!;
    const agent = (nodeId: string) => {
      const occurrence = trace.occurrences.find((item) => item.nodeId === nodeId)!;
      const material = materials.get(sourceKey(occurrence.source)) ?? null;
      return { occurrence, material, evidence: evaluationEvidence(occurrence, material) };
    };
    return { agent, events };
  } finally {
    await memory?.close();
  }
}

const GROUPS = `\
id: groups
name: groups
llm_config: {provider: anthropic, model: claude-test}
nodes:
  writer:
    kind: agent
    tools: [lookup, save]
    system_prompt: Help.
    prompt: hi
    output: str
output: \${writer.output}
`;

const GROUP_REPLIES = [
  callReply(["save", { topic: "a" }, "call-a"], ["save", { topic: "b" }, "call-b"]),
  callReply(["lookup", { topic: "x" }, "call-x"], ["lookup", { topic: "y" }, "call-y"], ["save", { topic: "z" }, "call-z"]),
  textReply("all done"),
];

const GROUP_TOOLS = () => ({ lookup: topicTool("lookup", "found", true), save: topicTool("save", "saved") });

test("the prompts come from the render", async () => {
  const { agent } = await rebuilt(GROUPS, GROUP_REPLIES, { tools: GROUP_TOOLS() });
  const { material, evidence } = agent("writer");
  expect(material!.baseSystemPrompt).toBe("Help.");
  expect(material!.prompt).toBe("hi");
  const { interactions: _interactions, ...rest } = evidence!;
  expect(rest).toEqual({
    scope: null,
    bound_inputs: {},
    base_system_prompt: "Help.",
    prompt: "hi",
    memory_sections: [],
  });
});

test("interactions follow step order: a reply's calls, then their results in call order", async () => {
  const { agent } = await rebuilt(GROUPS, GROUP_REPLIES, { tools: GROUP_TOOLS() });
  const call = (name: string, topic: string) => new ToolCallInteraction(name, { topic });
  const result = (name: string, text: string) => new ToolResultInteraction(name, text);
  // The run made two unsafe calls one at a time; the steps hold both calls before their results.
  expect(agent("writer").material!.interactions).toEqual([
    call("save", "a"),
    call("save", "b"),
    result("save", "saved:a"),
    result("save", "saved:b"),
    call("lookup", "x"),
    call("lookup", "y"),
    call("save", "z"),
    result("lookup", "found:x"),
    result("lookup", "found:y"),
    result("save", "saved:z"),
    new ModelTextInteraction("all done"),
  ]);
});

test.each([
  ["the prompt strategy shows each reply's text", "vllm", "no-structured-sentinel"],
  ["a native strategy shows the checked value", "openai", "gpt-5.5"],
])("a structured agent: %s", async (_label, provider, model) => {
  // The scripted model's provider and model pick the strategy.
  const text = `\
id: counted
name: counted
llm_config: {provider: openai, model: gpt-5.5}
nodes:
  count:
    kind: agent
    prompt: count
    output: int
output: \${count.output}
`;
  const replies = [textReply('{"value": "seven"}'), textReply('{"value": 7}')];
  const { agent } = await rebuilt(text, replies, { provider, model });
  const { material } = agent("count");
  expect(material!.digestOk).toBe(true);
  expect(material!.interactions).toEqual(
    provider === "vllm"
      ? [new ModelTextInteraction('{"value": "seven"}'), new ModelTextInteraction('{"value": 7}')]
      : [new ModelValueInteraction(7n)],
  );
});

test.each([
  ["a refused build leaves it out and is reported", new RangeError("no credentials")],
  ["a TypeError is a fault, which rejects", new TypeError("broken builder")],
])("a structured agent whose model can't be built now: %s", async (_label, error) => {
  const text = `\
id: counted
name: counted
llm_config: {provider: openai, model: gpt-5.5}
nodes:
  count:
    kind: agent
    prompt: count
    output: int
output: \${count.output}
`;
  const path = join(dir, `unbuilt-${++count}.jsonl`);
  const flow = await loadFlow(text);
  const model = new ScriptedModel([textReply('{"value": 7}')], { provider: "openai", model: "gpt-5.5" });
  const result = await runFlow(flow, {}, { log: path, learnable: true, models: serving(model), env: hostEnv(dir) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const refusing: ModelAccess = {
    env: hostEnv(dir),
    createModel: () => {
      throw error;
    },
  };
  if (error instanceof TypeError) {
    await expect(materialsOf(flow, path, null, refusing)).rejects.toBe(error);
    return;
  }
  const rebuiltMaterial = await materialsOf(flow, path, null, refusing);
  expect(rebuiltMaterial.agent("count").material).toBeNull();
  expect(rebuiltMaterial.events).toEqual([new ReplayDegraded({ nodeId: "count", reason: "rebuilt_request_differs" })]);
});

test("a skill grant: the render lists the recorded skills and every request carries load_skill", async () => {
  const text = `\
id: skilled
name: skilled
llm_config: {provider: anthropic, model: claude-test}
nodes:
  review:
    kind: agent
    skills: [style]
    prompt: review it
    output: str
output: \${review.output}
`;
  const skill = new Skill({ name: "style", description: "House style.", instructions: "Prefer small diffs." });
  const replies = [callReply([LOAD_SKILL, { name: "style" }, "call-1"]), textReply("reviewed")];
  const { agent } = await rebuilt(text, replies, { skills: new Map([["style", skill]]) });
  const { material } = agent("review");
  expect(material!.digestOk).toBe(true);
  expect(material!.baseSystemPrompt).toBe(systemPromptWithSkills(DEFAULT_SYSTEM, [skill]));
  expect(material!.interactions).toEqual([
    new ToolCallInteraction(LOAD_SKILL, { name: "style" }),
    new ToolResultInteraction(LOAD_SKILL, "Prefer small diffs."),
    new ModelTextInteraction("reviewed"),
  ]);
});

test("memory sections are recalled against the recorded snapshot", async () => {
  const text = `\
id: recalled
name: recalled
llm_config: {provider: anthropic, model: claude-test}
nodes:
  work:
    kind: agent
    long_term_memory: test
    prompt: hi
    output: str
output: \${work.output}
`;
  const memory = new FakeMemory({ recallText: "notes" });
  const { agent } = await rebuilt(text, [textReply("done")], { memory });
  const { material, evidence } = agent("work");
  expect(material!.memorySections).toEqual([new MemorySection("test", "notes")]);
  expect(evidence!["memory_sections"]).toEqual([{ memory: "test", text: "notes" }]);
});

test("a request that no longer matches its digest leaves the occurrence out and is reported", async () => {
  const path = join(dir, "tampered.jsonl");
  const flow = await loadFlow(GROUPS);
  const model = new ScriptedModel(GROUP_REPLIES, { provider: "anthropic", model: "claude-test" });
  const result = await runFlow(flow, {}, {
    log: path,
    learnable: true,
    models: serving(model),
    tools: GROUP_TOOLS(),
    env: hostEnv(dir),
  });
  expect(result.status).toBe("succeeded");
  const untouched = await materialsOf(flow, path, null, model);
  expect(untouched.agent("writer").material!.digestOk).toBe(true);
  expect(untouched.events).toEqual([]);

  const text = readFileSync(path, "utf8");
  const digest = /"request_sha256":"([0-9a-f]{64})"/.exec(text)![1]!;
  writeFileSync(path, text.replace(digest, "0".repeat(64)));
  const tampered = await materialsOf(flow, path, null, model);
  const { material, evidence } = tampered.agent("writer");
  expect(material).toBeNull();
  expect(evidence).toBeNull();
  expect(tampered.events).toEqual([new ReplayDegraded({ nodeId: "writer", reason: "rebuilt_request_differs" })]);
});

const GROUPED = `\
id: grouped
name: grouped
llm_config: {provider: anthropic, model: claude-test}
nodes:
  writer:
    kind: agent
    long_term_memory: test
    tools: [toolbox]
    prompt: hi
    output: str
output: \${writer.output}
`;

/** The reply of an AGENT backward call: no feedback for inputs or variables. */
const backwardReply = () => textReply(JSON.stringify({ input_feedback: {}, self_feedback: [], variable_feedback: {} }));

/**
 * Run `GROUPED` learnably, its grant naming the group `toolbox`, with `backend` as memory `test`;
 * the model has one more reply, for a backward call.
 */
async function groupedRun(backend: FakeMemory) {
  const path = join(dir, `grouped-${++count}.jsonl`);
  const flow = await loadFlow(GROUPED);
  const replies = [callReply(["lookup", { topic: "x" }, "call-x"]), textReply("done"), backwardReply()];
  const model = new ScriptedModel(replies, { provider: "anthropic", model: "claude-test" });
  const result = await runFlow(flow, {}, {
    log: path,
    learnable: true,
    models: serving(model),
    tools: { toolbox: [topicTool("lookup", "found"), topicTool("save", "saved")] },
    memory: new Map([["test", backend]]),
    env: hostEnv(dir),
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return { path, flow, result };
}

test("a grant naming a tool group is expanded with the run's tools when the same process learns", async () => {
  const backend = new FakeMemory();
  const { result } = await groupedRun(backend);
  const events: unknown[] = [];
  const outcome = await result.backward("Shorter.", { onEvent: (event) => events.push(event) });
  expect(outcome.status, JSON.stringify(outcome.failure)).toBe("succeeded");
  const [episode] = backend.learnContexts[0]!.episodes;
  expect(episode!.prompt).toBe("hi");
  expect(episode!.interactions).toEqual([
    new ToolCallInteraction("lookup", { topic: "x" }),
    new ToolResultInteraction("lookup", "found:x"),
    new ModelTextInteraction("done"),
  ]);
  expect(events).toEqual([]);
});

test("run_started records each granted tool group's members", async () => {
  const { path } = await groupedRun(new FakeMemory());
  const start = JSON.parse(readFileSync(path, "utf8").split("\n")[0]!);
  expect(start.payload.tool_groups).toEqual({ toolbox: ["lookup", "save"] });
  expect(Object.keys(start.payload.tools).sort()).toEqual(["lookup", "save"]);
});

test("a finished run reopened with no tools expands its grant from the recorded tool groups", async () => {
  const { path } = await groupedRun(new FakeMemory());
  const backend = new FakeMemory();
  // A finished run opens no tools, as in a fresh process: the groups come from the log.
  const model = new ScriptedModel([backwardReply()], { provider: "anthropic", model: "claude-test" });
  const reopened = await advance({
    log: path,
    memory: new Map([["test", backend]]),
    models: serving(model),
    env: hostEnv(dir),
  });
  const events: unknown[] = [];
  const outcome = await reopened.backward("Shorter.", { onEvent: (event) => events.push(event) });
  expect(outcome.status, JSON.stringify(outcome.failure)).toBe("succeeded");
  const context = backend.learnContexts[0]!;
  const [episode] = context.episodes;
  expect(episode!.prompt).toBe("hi");
  expect(episode!.interactions).toEqual([
    new ToolCallInteraction("lookup", { topic: "x" }),
    new ToolResultInteraction("lookup", "found:x"),
    new ModelTextInteraction("done"),
  ]);
  expect(logJson(context.feedback)).toContain("Shorter.");
  // AGENT backward reads the occurrence's evaluation evidence.
  const prompt = model.requests[0]!.messages[1]!.text();
  expect(prompt).toContain("Shorter.");
  expect(prompt).not.toMatch(/"evaluation_evidence": ?null/);
  expect(events).toEqual([]);
});
