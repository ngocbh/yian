/**
 * Learning reads an AGENT occurrence's steps, render and recall (`learningMaterials`), and what it
 * builds from them for AGENT backward (`evaluationEvidence`) and memory episodes. TS-only.
 *
 * Each run is learnable and logged; its log is replayed, and the material of each occurrence is
 * rebuilt from the replayed trace. Interactions follow step order: a reply's calls, then their
 * results in call order, so two unsafe calls read A, B, result A, result B.
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
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  MemorySection,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { learningMaterials } from "../../src/runtime/backward.ts";
import { evidencePayload, JsonlRunLog } from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { evaluationEvidence } from "../../src/runtime/trace.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
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
  options: { tools?: Record<string, FunctionTool>; memory?: FakeMemory; provider?: string; model?: string } = {},
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
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return await materialsOf(flow, path, registry, model);
}

/** Replay the log at `path` and rebuild its occurrences' material, with the reports it makes. */
async function materialsOf(
  flow: ComposedFlow,
  path: string,
  registry: Map<string, FakeMemory> | null,
  model: ScriptedModel,
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
      models: serving(model),
      expandTools: null,
      onEvent: (event) => events.push(event),
    });
    const trace = state.trace!;
    const agent = (nodeId: string) => {
      const occurrence = trace.occurrences.find((item) => item.nodeId === nodeId)!;
      const material = materials.get(sourceKey(occurrence.source)) ?? null;
      const evidence = evaluationEvidence(occurrence, material);
      return { occurrence, material, evidence, recorded: evidencePayload(occurrence.record.evidence) };
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

test("the prompts come from the render with the recorded skills", async () => {
  const { agent } = await rebuilt(GROUPS, GROUP_REPLIES, { tools: GROUP_TOOLS() });
  const { material, evidence, recorded } = agent("writer");
  expect(material!.baseSystemPrompt).toBe("Help.");
  expect(material!.prompt).toBe("hi");
  const { interactions: _rebuilt, ...rest } = evidence!;
  const { interactions: _recorded, ...recordedRest } = recorded!;
  expect(rest).toEqual(recordedRest);
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
  // The recorded evidence ran each unsafe call's result before the next call.
  expect((agent("writer").recorded!["interactions"] as { kind: string }[]).map((item) => item.kind)).toEqual([
    "tool_call",
    "tool_result",
    "tool_call",
    "tool_result",
    "tool_call",
    "tool_call",
    "tool_result",
    "tool_result",
    "tool_call",
    "tool_result",
    "model_text",
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
  const { material, evidence, recorded } = agent("count");
  expect(material!.digestOk).toBe(true);
  expect(evidence).toEqual(recorded);
  expect(material!.interactions).toEqual(
    provider === "vllm"
      ? [new ModelTextInteraction('{"value": "seven"}'), new ModelTextInteraction('{"value": 7}')]
      : [new ModelValueInteraction(7n)],
  );
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
  const { material, evidence, recorded } = agent("work");
  expect(evidence).toEqual(recorded);
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
