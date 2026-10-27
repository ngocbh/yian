/**
 * Public flow integration for the built-in episodic-memory backend: a backward pass commits an
 * episode that a fresh run recalls into its system prompt.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { memorySections, NO_ENV, reply, runWith } from "./run_state_fixtures.ts";

const FLOW = `id: episodic-history
name: episodic_history
input:
  topic: !str
nodes:
  answer:
    kind: agent
    long_term_memory:
      episodic: {type: episodic, config: {max_episodes: 4, recall_limit: 2}}
    input: {topic: "\${input.topic}"}
    prompt: Explain \${topic}.
    output: str
output: \${answer.output}
`;

const ANSWER = "Use a queue to preserve arrival order.";

const VARS = new Map([
  ["YIAN_DEFAULT_PROVIDER", "openai"],
  ["YIAN_DEFAULT_MODEL", "test-model"],
]);
const env: Env = (name) => VARS.get(name) ?? NO_ENV(name);

/** The memory section texts the system prompt of `model`'s last request shows. */
function recalled(model: ScriptedModel): string[] {
  return memorySections(model.requests.at(-1)!.messages).map((section) => section.text);
}

test("backward commits an episode recalled by a fresh run", async () => {
  const flow = await loadFlow(FLOW);
  const model = new ScriptedModel([reply(ANSWER), reply(ANSWER), reply(ANSWER)]);
  const memory = new Map([["episodic", new EpisodicMemory()]]);
  const inputs = { topic: "queue ordering" };

  const first = await runWith(flow, inputs, memory, model, { learnable: true, env });
  expect(first.status).toBe("succeeded");
  expect(recalled(model)).toEqual([]);

  // Forward retains the episode but does not update memory.
  const beforeBackward = await runWith(flow, inputs, memory, model, { env });
  expect(beforeBackward.status).toBe("succeeded");
  expect(recalled(model)).toEqual([]);

  const outcome = await first.backward();
  expect(outcome.status).toBe("succeeded");
  expect(outcome.learning).not.toBeNull();
  expect(outcome.learning!.status).toBe("succeeded");

  const later = await runWith(flow, inputs, memory, model, { env });
  expect(later.status).toBe("succeeded");
  expect(model.requests).toHaveLength(3);
  const sections = recalled(model);
  expect(sections).toHaveLength(1);
  const episode = JSON.parse(sections[0]!).episode;
  expect(episode.run_id).toBe(first.runId);
  expect(episode.inputs).toEqual({ topic: "queue ordering" });
  expect(episode.output).toBe(ANSWER);
});
