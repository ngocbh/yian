/**
 * History summaries change the active context without replacing retained history: mirrors
 * `tests/engine/test_session_summary_execution.py`, through `runFlow`.
 *
 * Ported: `test_soft_budget_keeps_four_prior_turns_and_all_raw_messages`,
 * `test_summary_failure_is_a_node_failure_without_answer_or_history_deletion` and
 * `test_new_summary_uses_current_model_without_a_second_model_factory`.
 *
 * Changed: the model factory Python patches is `models.createModel`, which records each config
 * it builds from. The summary request's input is read as its messages' texts joined.
 *
 * Later (T6-6, with resumeFlow): `test_interrupted_summary_can_resume_against_existing_memory`.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, type Request, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { JsonObject } from "../../src/memory/contracts.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { runWith, snapshot } from "./session_fakes.ts";

const FLOW = `id: summarized-session
name: summarized_session
llm_config: {provider: openai, model: old-model}
input:
  question: !str
nodes:
  answer:
    kind: agent
    long_term_memory: session
    input: {question: "\${input.question}"}
    prompt: \${question}
    output: str
output: \${answer.output}
`;
const SUMMARY = "Retained summary of the earliest conversation. ".repeat(3);
const QUESTIONS = [0, 1, 2, 3, 4].map((index) => `prior question ${index}: ${"x".repeat(30)}`);

/** A reply holding only `text`. */
const reply = (text: string): Response =>
  new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });

/** The five prior answers. */
const replies = (): Response[] => [0, 1, 2, 3, 4].map((index) => reply(`prior answer ${index}`));

/** The conversation part of `memory`'s current snapshot. */
const conversation = (memory: SessionMemory): JsonObject => snapshot(memory)["conversation"] as JsonObject;

/** The texts of `request`'s messages. */
const texts = (request: Request): string[] => request.messages.map((message) => message.text());

/** Run the five prior questions, each answered and none learned. */
async function seed(flow: ComposedFlow, memory: SessionMemory, model: ChatModel, options = {}): Promise<void> {
  for (const [index, question] of QUESTIONS.entries()) {
    const result = await runWith(flow, { question }, memory, model, options);
    expect(result.output).toBe(`prior answer ${index}`);
    expect(result.learning).toBeNull();
  }
}

test("a soft budget keeps four prior turns and all raw messages", async () => {
  const flow = await loadFlow(FLOW);
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 4 });
  const model = new ScriptedModel([...replies(), reply(SUMMARY), reply("current answer")]);
  await seed(flow, memory, model);
  // No eligible older turn yet, despite the budget.
  expect(model.requests).toHaveLength(5);
  const before = conversation(memory)["items"] as JsonObject[];

  const result = await runWith(flow, { question: "current question" }, memory, model);

  expect(result.output).toBe("current answer");
  expect(result.learning).toBeNull();
  // One oversized summary, without a retry, then the answer.
  expect(model.requests).toHaveLength(7);
  const summaryInput = texts(model.requests[5]!).join("\n");
  expect(summaryInput).toContain(QUESTIONS[0]!);
  expect(summaryInput).toContain("prior answer 0");
  for (const question of QUESTIONS.slice(1)) expect(summaryInput).not.toContain(question);
  expect(summaryInput).not.toContain("current question");
  const answerContext = texts(model.requests[6]!);
  expect(answerContext.some((text) => text.includes(SUMMARY))).toBe(true);
  expect(answerContext).not.toContain(QUESTIONS[0]!);
  expect(answerContext).not.toContain("prior answer 0");
  for (let index = 1; index < 5; index++) {
    expect(answerContext).toContain(QUESTIONS[index]!);
    expect(answerContext).toContain(`prior answer ${index}`);
  }
  expect(answerContext.at(-1)).toBe("current question");
  const stored = snapshot(memory);
  const items = (stored["conversation"] as JsonObject)["items"] as JsonObject[];
  expect(items.slice(0, before.length)).toEqual(before);
  expect(items).toHaveLength(18);
  expect(JSON.stringify(items)).not.toContain(SUMMARY);
  expect(stored["notes"]).toEqual([]);
});

test("a summary failure is a node failure without an answer or history deletion", async () => {
  const flow = await loadFlow(FLOW);
  const memory = new SessionMemory({ historyBudgetChars: 1 });
  const model = new ScriptedModel([...replies(), new Error("summary request failed")]);
  await seed(flow, memory, model);
  const before = conversation(memory);

  const failed = await runWith(flow, { question: "current question" }, memory, model);

  expect(failed.failure!.code).toBe("node.execute");
  expect(model.requests).toHaveLength(6);
  const after = conversation(memory);
  expect(after["items"]).toEqual(before["items"]);
  expect(after["effects"]).toEqual(before["effects"]);
  expect(((after["closed_invocations"] as JsonObject[]).at(-1) as JsonObject)["status"]).toBe("failed");
});

test("a new summary uses the current model without a second model factory", async () => {
  const flow = await loadFlow(FLOW);
  const memory = new SessionMemory({ historyBudgetChars: 1 });
  const old = new ScriptedModel(replies());
  const current = new ScriptedModel([reply(SUMMARY), reply("new answer")]);
  const configs: Readonly<Record<string, unknown>>[] = [];
  const models = {
    env: () => undefined,
    createModel: (config: Readonly<Record<string, unknown>>) => {
      configs.push(config);
      return config["model"] === "new-model" ? current : old;
    },
  };
  await seed(flow, memory, old, { models });
  const changed = await loadFlow(FLOW.replace("old-model", "new-model"));
  const result = await runWith(changed, { question: "new question" }, memory, current, { models });

  expect(result.output).toBe("new answer");
  expect(configs).toHaveLength(6);
  expect(configs.at(-1)!["model"]).toBe("new-model");
  expect([old.requests.length, current.requests.length]).toEqual([5, 2]);
  expect(texts(current.requests[0]!).join("\n")).toContain(QUESTIONS[0]!);
});
