/**
 * Every shipped example loads and compiles (no live LLM): mirrors
 * `tests/examples/test_examples.py`.
 *
 * Ported: `test_example_loads_and_compiles` (one test per `examples/*.yaml`, loaded with the
 * examples directory as the search path), `test_examples_present` and
 * `test_debate_agents_are_tool_free`. TS-only: the shape of `human-approval.yaml`'s `ask` gate,
 * `chat.yaml` on its `Inbox(str)` input, run end to end on a scripted model, and
 * `compact_history.yaml` alone (pass-through, `keep`, a failed summary).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { ASK } from "../../src/tools/context.ts";
import { isInboxType, ValueKind } from "../../src/typesys/values.ts";
import { NO_ENV, PromptModel, serving, textReply } from "../engine/run_state_fixtures.ts";

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const EXAMPLES = readdirSync(EXAMPLES_DIR)
  .filter((name) => name.endsWith(".yaml"))
  .sort();

const load = (name: string) =>
  loadFlow(readFileSync(join(EXAMPLES_DIR, name), "utf8"), { searchPaths: [EXAMPLES_DIR] });

test.each(EXAMPLES)("example %s loads and compiles", async (name) => {
  const loaded = await load(name);
  expect(loaded.compiled).toBeDefined();
  expect(loaded.input).toBeDefined();
});

test("examples present", () => {
  for (const name of ["hello.yaml", "summarize.yaml", "classify.yaml"]) {
    expect(EXAMPLES).toContain(name);
  }
});

test("debate agents are tool-free", async () => {
  const loaded = await load("debate.yaml");
  for (const nodeId of ["frame", "for_case", "against_case", "verdict"]) {
    const node = loaded.compiled.nodes.get(nodeId) as AgentNode;
    expect(node.tools).toEqual([]);
  }
});

test("human-approval's approve is a TOOL ask answered with the Approval enum", async () => {
  const loaded = await load("human-approval.yaml");
  const node = loaded.compiled.nodes.get("approve");
  expect(node).toBeInstanceOf(ToolNode);
  expect((node as ToolNode).builtin).toBe(ASK);
  expect((node as ToolNode).answerSchema).toEqual([
    { name: "answer", type: "str", required: true, enum: ["approve", "revise"] },
  ]);
});

test("chat's one input is the `messages` inbox of text", async () => {
  const loaded = await load("chat.yaml");
  const input = loaded.input!;
  expect(input.map((param) => param.name)).toEqual(["messages"]);
  const type = input[0]!.contract!.type;
  expect(isInboxType(type)).toBe(true);
  expect(type!.element?.kind).toBe(ValueKind.STRING);
});

const tmp = mkdtempSync(join(tmpdir(), "yian-examples-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

test("chat waits for each message, answers it, and ends when its inbox closes", async () => {
  const loaded = await load("chat.yaml");
  const config = { provider: "anthropic", model: "claude-test" };
  const models = serving(new ScriptedModel([textReply("hello")], config));
  const log = join(tmp, "chat.jsonl");
  const options = { env: NO_ENV, numWorkers: 0, models };
  let result = await runFlow(loaded, {}, { ...options, log, llmConfig: config });
  expect(result.status).toBe("waiting");
  await send(log, "messages", "hi", { env: NO_ENV });
  result = await advance({ ...options, log });
  expect(result.status).toBe("waiting");
  await close(log, "messages", { env: NO_ENV });
  result = await advance({ ...options, log });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([
    { role: "user", text: "hi" },
    { role: "assistant", text: "hello" },
  ]);
}, 30_000);

test("chat joins the messages one take reads with newlines before the model sees them", async () => {
  const loaded = await load("chat.yaml");
  const config = { provider: "anthropic", model: "claude-test" };
  const models = serving(new ScriptedModel([textReply("ok")], config));
  const log = join(tmp, "chat-joined.jsonl");
  const options = { env: NO_ENV, numWorkers: 0, models };
  expect((await runFlow(loaded, {}, { ...options, log, llmConfig: config })).status).toBe("waiting");
  await send(log, "messages", "a", { env: NO_ENV });
  await send(log, "messages", "b", { env: NO_ENV });
  await close(log, "messages", { env: NO_ENV });
  const result = await advance({ ...options, log });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([
    { role: "user", text: "a\nb" },
    { role: "assistant", text: "ok" },
  ]);
}, 30_000);

const config = { provider: "anthropic", model: "claude-test" };
const opts = { env: NO_ENV, numWorkers: 0, llmConfig: config };
const turn = (u: string, a: string) => [
  { role: "user", text: u },
  { role: "assistant", text: a },
];

test("compact_history passes through under budget with no model call", async () => {
  const flow = await load("compact_history.yaml");
  const history = [turn("hi", "hello")];
  const result = await runFlow(flow, { summary: "s", history }, { ...opts, models: serving(new ScriptedModel([], config)) });
  expect(result.output).toEqual({ summary: "s", history });
});

test("compact_history keeps up to `keep` turns even over budget", async () => {
  const flow = await load("compact_history.yaml");
  const history = [turn("a1", "b1"), turn("a2", "b2"), turn("a3", "b3")]; // over budget 10, 3 <= keep 4
  const result = await runFlow(
    flow,
    { summary: "s", history, budget: 10n },
    { ...opts, models: serving(new ScriptedModel([], config)) },
  );
  expect(result.output).toEqual({ summary: "s", history });
});

test.each([
  [0n, 0],
  [1n, 1],
  [-1n, 0],
])("compact_history over budget with keep %p keeps %p turns", async (keep, kept) => {
  const flow = await load("compact_history.yaml");
  const model = new PromptModel({ reply: "unused", summary: "S" });
  const history = [turn("a1", "b1"), turn("a2", "b2"), turn("a3", "b3")];
  const result = await runFlow(flow, { summary: "old", history, budget: 1n, keep }, { ...opts, models: serving(model) });
  expect(result.output).toEqual({ summary: "S", history: history.slice(history.length - kept) });
  // The summarizer saw the old summary and exactly the folded turns, as `role: text` lines.
  const prompt = model.requests[0]!.messages.at(-1)!.text();
  expect(prompt).toContain("old");
  for (const t of history.slice(0, history.length - kept)) expect(prompt).toContain(`user: ${t[0]!.text}`);
  for (const t of history.slice(history.length - kept)) expect(prompt).not.toContain(`user: ${t[0]!.text}`);
});

test("a failed summary fails the run with the provider's code", async () => {
  const flow = await load("compact_history.yaml");
  const model = new ScriptedModel([new ProviderError("auth", "unauthorized", { status: 401 })], config);
  const result = await runFlow(
    flow,
    { summary: "", history: [turn("a", "b")], budget: 1n, keep: 0n },
    { ...opts, models: serving(model) },
  );
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("provider.auth");
});
