/**
 * End-to-end blocking answers across the three `human_input` question surfaces: a port of
 * `tests/engine/test_human_questions_e2e.py`.
 *
 * A `human_input` gate calls the run's `ask_user` handler with a validated `questions` list and
 * returns the handler's record keyed by each question's `header`. The gate's questions are static
 * (a literal `questions:` list, seed 26), upstream (a CODE node produces the list, fed through
 * `questions: ${qs}`), or adaptive (an `adaptive_questions:` block that desugars at load into a
 * composer AGENT and the gate; the agent authors the list).
 *
 * Ported: `test_static_questions_block_and_return_answer` (seed 26's TS flow, read from
 * `run-leaves/seed_26_human_questions`), `test_manual_ref_questions_block_and_return_answer` (the
 * CODE reference is TS's, `./fixtures/seeds.ts:questions_seed`) and
 * `test_adaptive_questions_block_and_return_answer`. The composer's model is a `ScriptedModel`
 * given as `models`, named by the run's `llmConfig`, where Python patches `model_from_config`; it
 * answers the composer's `list(Question)` output as the `{"items": [...]}` wrapper. Every run sees
 * only the test's credential store in its environment.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { caseText, loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-human-questions-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** One question of an `ask_user` request. */
type Question = { question: string; header: string; options?: { label: string }[] };

/** Return an `ask_user` handler answering `answer`, and the requests it received. */
function answering(answer: unknown) {
  const requests: AskUserRequest[] = [];
  const handler = (request: AskUserRequest) => {
    requests.push(request);
    return answer;
  };
  return { handler, requests };
}

/** The questions of a request. */
const questionsOf = (request: AskUserRequest) => request.questions as Question[];

// --- static: a literal questions list in the seed ------------------------------------- //

test("static questions block and return answer", async () => {
  // Seed 26 renders its static questions and routes the host answer.
  const loaded = await loadConformanceFlow(caseText("run-leaves/seed_26_human_questions").text, { env });
  const { handler, requests } = answering({ Framework: "React", Notes: "ship it" });

  const result = await runFlow(loaded, { proj: "Atlas" }, { env, tools: { ask_user: handler } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  const questions = questionsOf(requests[0]!);
  expect(questions.length).toBeGreaterThan(0);
  expect(questions.map((q) => q.header)).toEqual(["Framework", "Notes"]);
  const framework = questions.find((q) => q.header === "Framework")!;
  expect(framework.question).toContain("Atlas"); // ${proj} rendered
  expect(result.output).toBe("React"); // routed on the returned label
});

// --- upstream: a code node produces the questions list -------------------------------- //

const MANUAL_REF = `
id: manual_ref
name: manual_ref
input:
  seed: !str
nodes:
  src:
    kind: code
    input: {seed: "\${input.seed}"}
    output: list(object)
    code: ./fixtures/seeds.ts:questions_seed
  ask:
    kind: human_input
    input: {qs: "\${src.output}"}
    questions: "\${qs}"
output: \${ask.output}
`;

test("manual ref questions block and return answer", async () => {
  // A CODE node builds the questions list; the gate reads it through `questions: ${qs}`, passes
  // the code-produced question to the host, and returns its record.
  const loaded = await loadConformanceFlow(MANUAL_REF, { env });
  const { handler, requests } = answering({ H: "A" });

  const result = await runFlow(loaded, { seed: "Which option?" }, { env, tools: { ask_user: handler } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(questionsOf(requests[0]!)).toHaveLength(1);
  const q = questionsOf(requests[0]!)[0]!;
  expect(q.question).toBe("Which option?"); // the seed input flowed through the CODE node
  expect(q.header).toBe("H");
  expect(result.output).toEqual({ H: "A" });
});

// --- adaptive: an LLM composes the questions, desugared into agent + gate -------------- //

const ADAPTIVE = `
id: adaptive
name: adaptive
input:
  ctx: !str
nodes:
  ask:
    kind: human_input
    input: {ctx: "\${input.ctx}"}
    prompt: "Help me decide on \${ctx}."
    adaptive_questions:
      prompt: "Design 1-3 questions with options for: \${ctx}"
output: \${ask.output}
`;

// The questions the composer "authors": one single-select choice.
const COMPOSED = [
  {
    question: "Which database?",
    header: "DB",
    options: [
      { label: "Postgres", description: "Relational." },
      { label: "SQLite", description: "Embedded." },
    ],
    multi_select: false,
  },
];

test("adaptive questions block and return answer", async () => {
  // An `adaptive_questions:` block desugars into a composer AGENT and the gate. The agent composes
  // the questions list and the gate passes exactly those questions to the host.
  const loaded = await loadConformanceFlow(ADAPTIVE, { env });
  const model = new ScriptedModel([textReply(JSON.stringify({ items: COMPOSED }))]);
  const { handler, requests } = answering({ DB: "Postgres" });

  const result = await runFlow(loaded, { ctx: "storage" }, {
    env,
    models: serving(model, env),
    llmConfig: { provider: "openai", model: "m" },
    tools: { ask_user: handler },
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  // The composer takes the native structured path, where Python's `_ComposeChat` refuses plain text.
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.outputFormat).not.toBeNull();
  const questions = questionsOf(requests[0]!);
  expect(questions.map((q) => q.header)).toEqual(["DB"]);
  expect(questions[0]!.question).toBe("Which database?");
  expect(questions[0]!.options!.map((o) => o.label)).toEqual(["Postgres", "SQLite"]);
  expect(result.output).toEqual({ DB: "Postgres" });
});
