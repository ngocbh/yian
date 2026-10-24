/**
 * Every shipped example loads and compiles (no live LLM): mirrors
 * `tests/examples/test_examples.py`.
 *
 * Ported: `test_example_loads_and_compiles` (one test per `examples/*.yaml`, loaded with the
 * examples directory as the search path), `test_examples_present` and
 * `test_debate_agents_are_tool_free`. TS-only: the shape of `human-approval.yaml`'s `ask` gate,
 * `chat.yaml` on its `Inbox(str)` input, run end to end on a scripted model, and
 * `compact_history.yaml` alone (pass-through, `keep`, a failed summary), and a learnable chat
 * whose reply feedback reaches the summarizer.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { BackwardReadyRecord, JsonlRunLog, NodeStartedRecord } from "../../src/runtime/log.ts";
import { ASK } from "../../src/tools/context.ts";
import { isInboxType, ValueKind } from "../../src/typesys/values.ts";
import {
  NO_ENV,
  PromptModel,
  REPLY_FIRST_LINE,
  serving,
  SUMMARIZER_FIRST_LINE,
  systemText,
  textReply,
} from "../engine/run_state_fixtures.ts";

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
  expect(result.output).toEqual({
    summary: "",
    history: [
      { role: "user", text: "hi" },
      { role: "assistant", text: "hello" },
    ],
  });
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
  expect(result.output).toEqual({
    summary: "",
    history: [
      { role: "user", text: "a\nb" },
      { role: "assistant", text: "ok" },
    ],
  });
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

test("chat folds the older turns into a summary once nine long turns pass the budget", async () => {
  const loaded = await load("chat.yaml");
  const SUMMARY = "s".repeat(2000);
  const model = new PromptModel({ reply: "r".repeat(2000), summary: SUMMARY });
  const log = join(tmp, "chat-fold.jsonl");
  const options = { env: NO_ENV, numWorkers: 0, models: serving(model) };
  const message = (n: number) => `${n}:`.padEnd(2000, "m");
  expect((await runFlow(loaded, {}, { ...options, log, llmConfig: config })).status).toBe("waiting");
  for (let n = 1; n <= 10; n++) {
    await send(log, "messages", message(n), { env: NO_ENV });
    expect((await advance({ ...options, log })).status).toBe("waiting");
  }
  // Nine replies, the summary the 10th turn's `compacted` writes (9 turns are 36,000 characters),
  // then the 10th reply.
  const kinds: string[] = model.requests.map((r) =>
    systemText(r).startsWith(SUMMARIZER_FIRST_LINE) ? "summary" : "reply",
  );
  expect(kinds).toEqual([...Array<string>(9).fill("reply"), "summary", "reply"]);
  const replies = model.requests.filter((r) => systemText(r).startsWith(REPLY_FIRST_LINE));
  expect(replies).toHaveLength(10);
  for (const r of replies.slice(0, 9)) expect(systemText(r).trimEnd().endsWith("Conversation so far:")).toBe(true);
  expect(systemText(replies[9]!).trimEnd().endsWith(`Conversation so far: ${SUMMARY}`)).toBe(true);
  // The 10th reply sees turns 6-9 verbatim, then its own message.
  const messages = replies[9]!.messages.filter((m) => m.role !== "system");
  const kept = [6, 7, 8, 9].flatMap((n) => [
    ["user", message(n)],
    ["assistant", "r".repeat(2000)],
  ]);
  expect(messages.map((m) => [m.role, m.text()])).toEqual([...kept, ["user", message(10)]]);
}, 60_000);

test("in a learnable chat, feedback on a reply reaches the summarizer through the summary", async () => {
  // A copy of the chat with a 200-character budget, beside an unchanged compact_history.yaml.
  const dir = mkdtempSync(join(tmp, "small-budget-"));
  const chatText = readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8");
  const compactedInput = `input: {summary: "\${input.summary}", history: "\${input.history}"}`;
  expect(chatText).toContain(compactedInput);
  writeFileSync(join(dir, "chat.yaml"), chatText.replace(compactedInput, `${compactedInput.slice(0, -1)}, budget: 200}`));
  writeFileSync(join(dir, "compact_history.yaml"), readFileSync(join(EXAMPLES_DIR, "compact_history.yaml"), "utf8"));
  const loaded = await loadFlow(readFileSync(join(dir, "chat.yaml"), "utf8"), { searchPaths: [dir] });
  // A backward input_feedback value is a list of messages; a bare string fails the attempt.
  const KEEP_NAME = JSON.stringify({
    input_feedback: { summary: ["Keep the user's name."] },
    self_feedback: [],
    variable_feedback: {},
  });
  const model = new PromptModel({ reply: "x".repeat(30), summary: "S", backward: [textReply(KEEP_NAME)] });
  const log = join(dir, "chat.jsonl");
  const options = { env: NO_ENV, numWorkers: 0, models: serving(model) };
  let result = await runFlow(loaded, {}, { ...options, log, learnable: true, llmConfig: config });
  // Five 60-character turns pass 200, so the sixth turn's `compacted` folds; the closing turn
  // folds again.
  for (let n = 1; n <= 6; n++) {
    expect(result.status).toBe("waiting");
    await send(log, "messages", `${n}:`.padEnd(30, "m"), { env: NO_ENV });
    result = await advance({ ...options, log });
  }
  await close(log, "messages", { env: NO_ENV });
  result = await advance({ ...options, log });
  expect(result.status).toBe("succeeded");
  const history = (result.output as { history: unknown[] }).history;
  expect(history).toHaveLength(8);
  const feedback = [...Array<null>(history.length - 1).fill(null), "Mention the user's name."];
  const outcome = await result.backward({ history: feedback });
  expect(outcome.status).toBe("succeeded");
  const reader = JsonlRunLog.resume(log);
  const records = (() => {
    try {
      return reader.records();
    } finally {
      reader.close();
    }
  })();
  const starts = new Map(
    records.flatMap((e) => (e.record instanceof NodeStartedRecord ? [[e.seq, e.record.nodeId] as const] : [])),
  );
  const ready = records.map((e) => e.record).filter((r) => r instanceof BackwardReadyRecord);
  expect(ready).toHaveLength(1);
  const summarized = ready[0]!.attributedResults.filter((item) =>
    starts.get(item.source.executionSeq)!.endsWith("/summarize"),
  );
  const reached = summarized.filter((item) => item.invoked && item.feedback.length > 0);
  expect(reached.map((item) => starts.get(item.source.executionSeq))).toEqual(["chat#5/compacted/folded/summarize"]);
}, 60_000);
