/**
 * Explicit API backward feedback becomes conversation-local recalled notes: mirrors
 * `tests/engine/test_agent_session_notes.py`.
 *
 * Ported, through `runFlow`: `test_forward_without_learning_keeps_history_but_adds_no_notes`.
 *
 * Later (T6-6, with RunResult.backward): `test_backward_reuses_self_feedback_without_an_extra_model_call`,
 * `test_backward_without_feedback_adds_no_notes_or_model_call`,
 * `test_learned_notes_do_not_cross_sibling_conversations` and
 * `test_admitted_failed_forward_can_learn_without_rewriting_history`.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { reply, runWith, SESSION_CONFIG, sentTexts, snapshot } from "./session_fakes.ts";

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
