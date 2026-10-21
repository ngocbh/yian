/**
 * Explicit API backward feedback becomes conversation-local recalled notes: mirrors
 * `tests/engine/test_agent_session_notes.py`.
 *
 * Ported, through `runFlow` and `RunResult.backward`, the model a `ScriptedModel` from `models`
 * (Python patches `model_from_config`): `test_backward_reuses_self_feedback_without_an_extra_model_call`,
 * `test_backward_without_feedback_adds_no_notes_or_model_call`,
 * `test_forward_without_learning_keeps_history_but_adds_no_notes`,
 * `test_learned_notes_do_not_cross_sibling_conversations` and
 * `test_admitted_failed_forward_can_learn_without_rewriting_history` (the forward's model error is
 * an `Error`, Python's `RuntimeError`).
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { backwardReply, reply, runWith, SESSION_CONFIG, sentTexts, snapshot } from "./session_fakes.ts";

const NOTE = "Check the measurement units before answering.";

const FLOW = `id: learned-session
name: learned_session
llm_config: {provider: openai, model: test-model}
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

/** The texts of every message of `model`'s request `index`, its system prompt included. */
function allTexts(model: ScriptedModel, index = -1): string[] {
  return model.requests.at(index)!.messages.map((message) => message.text());
}

test("backward reuses self feedback without an extra model call", async () => {
  const flow = await loadFlow(FLOW);
  const model = new ScriptedModel([reply("first answer"), backwardReply(NOTE), reply("next answer")], SESSION_CONFIG);
  const memory = new SessionMemory();
  const first = await runWith(flow, { question: "first question" }, memory, model, { learnable: true });
  expect(first.status).toBe("succeeded");
  const history = snapshot(memory)["conversation"];

  const outcome = await first.backward("The answer used inconsistent units.");
  expect(outcome.status).toBe("succeeded");
  // The forward call and the backward call; no note-generation call.
  expect(model.requests).toHaveLength(2);
  expect(snapshot(memory)["conversation"]).toEqual(history);

  const second = await runWith(flow, { question: "next question" }, memory, model);
  expect(second.status).toBe("succeeded");
  expect(second.output).toBe("next answer");
  expect(model.requests).toHaveLength(3);
  expect(allTexts(model).reduce((count, text) => count + text.split(NOTE).length - 1, 0)).toBe(1);
  expect(snapshot(memory)["notes"]).toHaveLength(1);
});

test("backward without feedback adds no notes or model call", async () => {
  const model = new ScriptedModel([reply("answer")], SESSION_CONFIG);
  const memory = new SessionMemory();
  const result = await runWith(await loadFlow(FLOW), { question: "question" }, memory, model, { learnable: true });
  const history = snapshot(memory)["conversation"];
  expect((await result.backward()).status).toBe("succeeded");
  expect(model.requests).toHaveLength(1);
  expect(snapshot(memory)["conversation"]).toEqual(history);
  expect(snapshot(memory)["notes"] ?? []).toEqual([]);
});

test("forward without learning keeps history but adds no notes", async () => {
  const model = new ScriptedModel([reply("first answer"), reply("next answer")], SESSION_CONFIG);
  const memory = new SessionMemory();
  const flow = await loadFlow(FLOW);
  for (const question of ["first question", "next question"]) {
    const result = await runWith(flow, { question }, memory, model);
    expect(result.status).toBe("succeeded");
    expect(result.learning).toBeNull();
  }
  expect(model.requests).toHaveLength(2);
  expect(sentTexts(model)).toContain("first answer");
  expect(snapshot(memory)["notes"] ?? []).toEqual([]);
});

test("learned notes do not cross sibling conversations", async () => {
  const flow = await loadFlow(`id: sibling-notes
name: sibling_notes
llm_config: {provider: openai, model: test-model}
input:
  question: !str
nodes:
  left:
    kind: agent
    long_term_memory: left_chat
    input: {question: "\${input.question}"}
    prompt: "left: \${question}"
    output: str
  right:
    kind: agent
    long_term_memory: right_chat
    input: {question: "\${input.question}"}
    prompt: "right: \${question}"
    output: str
output:
  left: \${left.output}
  right: \${right.output}
`);
  const model = new ScriptedModel(
    [reply("answer"), reply("answer"), backwardReply(NOTE), reply("answer"), reply("answer")],
    SESSION_CONFIG,
  );
  const left = new SessionMemory();
  const right = new SessionMemory();
  const memory = new Map([
    ["left_chat", left],
    ["right_chat", right],
  ]);
  const first = await runWith(flow, { question: "first question" }, memory, model, { learnable: true });
  expect(first.status).toBe("succeeded");
  expect((await first.backward({ left: "Check units.", right: null })).status).toBe("succeeded");
  expect(model.requests).toHaveLength(3);
  expect((await runWith(flow, { question: "next question" }, memory, model)).status).toBe("succeeded");
  expect(model.requests).toHaveLength(5);
  for (const index of [3, 4]) {
    const texts = allTexts(model, index);
    const leaf = texts.at(-1)!.split(":", 1)[0];
    expect(["left", "right"]).toContain(leaf!);
    expect(texts.some((text) => text.includes(NOTE))).toBe(leaf === "left");
  }
  expect(snapshot(left)["notes"]).toHaveLength(1);
  expect(snapshot(right)["notes"] ?? []).toEqual([]);
});

test("an admitted failed forward can learn without rewriting history", async () => {
  const model = new ScriptedModel(
    [new Error("provider could not answer"), backwardReply(NOTE), reply("recovered")],
    SESSION_CONFIG,
  );
  const memory = new SessionMemory();
  const flow = await loadFlow(FLOW);
  const first = await runWith(flow, { question: "first question" }, memory, model, { learnable: true });
  expect(first.status).toBe("failed");
  const history = snapshot(memory)["conversation"];
  expect((await first.backward("Check the input units.")).status).toBe("succeeded");
  expect(snapshot(memory)["conversation"]).toEqual(history);
  const second = await runWith(flow, { question: "next question" }, memory, model);
  expect(second.status).toBe("succeeded");
  expect(model.requests).toHaveLength(3);
  expect(allTexts(model).some((text) => text.includes(NOTE))).toBe(true);
});
