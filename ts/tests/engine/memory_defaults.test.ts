/**
 * Memory defaults come from `defaults.ts`: a port of `tests/engine/test_memory_defaults.py`.
 *
 * Ported: `test_episodic_limits_default_from_the_environment`,
 * `test_session_memory_settings_default_from_the_environment` (the settings are read through the
 * memory's private members, as Python reads `_history_budget_chars`),
 * `test_reflective_limits_default_from_the_environment`,
 * `test_reflection_lesson_material_follows_its_budget`,
 * `test_reflection_episode_material_follows_its_budget` and
 * `test_profile_memory_budgets_default_from_the_environment`.
 *
 * TS-only: none.
 *
 * Changed:
 * - `reload(env)` with a map stands in for `set_defaults`; reflective backends also get `baseEnv()`,
 *   an environment map standing in for Python's `conftest.py` variables.
 * - The reflection rows' `createModel` returns a `ScriptedModel` answering a no-op, and the prompt
 *   is read off its recorded request, where Python patches `model_from_config`.
 *
 * Not ported: `test_memory_nesting_bound_follows_the_value_nesting_limit` (TS has no memory nesting
 * limit); `test_reflection_limit_material_follows_its_budget` (the `{"max_items": N}` material can't
 * reach the 256 floor, and an ESM call can't be spied on; `reflective_memory.test.ts`'s
 * `boundedJsonMaterial` test checks the cut at the floor).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { BUILTIN, type Env, install, reload } from "../../src/defaults.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type JsonObject,
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
} from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";

let home: string;

beforeEach(() => {
  install(BUILTIN);
  home = fs.mkdtempSync(path.join(os.tmpdir(), "yian-memory-defaults-"));
});

afterEach(() => {
  install(BUILTIN);
  fs.rmSync(home, { recursive: true, force: true });
});

const CONFIG = { provider: "ollama", model: "qwen" };

/** Python's `conftest.py` environment: an empty private store and an explicit model default. */
function baseEnv(): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", path.join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
  return (name) => vars.get(name);
}

/** Install the defaults read from `vars` alone. */
function setDefaults(vars: Record<string, string>): void {
  const env = new Map(Object.entries(vars));
  reload((name) => env.get(name));
}

/** Return the pinned snapshot of one run of `memory`, optionally restoring `snapshot`. */
function snapshotOf(memory: EpisodicMemory | ReflectiveMemory, snapshot?: JsonObject): JsonObject {
  const session = memory.openRun({ snapshot });
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

test("episodic limits default from the environment", () => {
  const env = new Map([
    ["YIAN_EPISODIC_MAX_EPISODES", "3"],
    ["YIAN_EPISODIC_RECALL_LIMIT", "2"],
  ]);
  reload((name) => env.get(name));
  expect(snapshotOf(new EpisodicMemory()).config).toEqual({ max_episodes: 3n, recall_limit: 2n });
  const recorded = snapshotOf(new EpisodicMemory({ maxEpisodes: 7, recallLimit: 5 }));
  expect(recorded.config).toEqual({ max_episodes: 7n, recall_limit: 5n });
  expect(snapshotOf(new EpisodicMemory(), recorded).config).toEqual({ max_episodes: 7n, recall_limit: 5n });
});

test("session memory settings default from the environment", () => {
  const env = new Map([
    ["YIAN_SESSION_HISTORY_BUDGET_CHARS", "10"],
    ["YIAN_SESSION_RECENT_TURNS", "1"],
  ]);
  reload((name) => env.get(name));
  const memory = new SessionMemory();
  expect([memory["historyBudgetChars"], memory["recentTurns"]]).toEqual([10, 1]);
  const explicit = new SessionMemory({ historyBudgetChars: 20, recentTurns: 3 });
  expect([explicit["historyBudgetChars"], explicit["recentTurns"]]).toEqual([20, 3]);
});

test("reflective limits default from the environment", () => {
  setDefaults({ YIAN_REFLECTIVE_MAX_ITEMS: "3", YIAN_REFLECTIVE_RECALL_LIMIT: "2" });
  const memory = (options: ConstructorParameters<typeof ReflectiveMemory>[0] = {}) =>
    new ReflectiveMemory({ env: baseEnv(), ...options });
  expect(snapshotOf(memory({ llmConfig: CONFIG })).config).toEqual({ max_items: 3n, recall_limit: 2n });
  const recorded = snapshotOf(memory({ llmConfig: CONFIG, maxItems: 7, recallLimit: 5 }));
  expect(recorded.config).toEqual({ max_items: 7n, recall_limit: 5n });
  expect(snapshotOf(memory(), recorded).config).toEqual({ max_items: 7n, recall_limit: 5n });
});

/** A reflective backend whose model answers a no-op, and the user prompt it last received. */
function noopLearner(): { memory: ReflectiveMemory; prompt: () => string } {
  const models: ScriptedModel[] = [];
  const noop = { actions: [{ action: "noop", lesson_id: null, text: null }] };
  const memory = new ReflectiveMemory({
    llmConfig: CONFIG,
    env: baseEnv(),
    createModel: () => {
      const message = new Message({ role: "assistant", parts: [new TextPart(JSON.stringify(noop))] });
      const model = new ScriptedModel([new Response({ message })], CONFIG);
      models.push(model);
      return model;
    },
  });
  return { memory, prompt: () => models.at(-1)!.requests.at(-1)!.messages.at(-1)!.text() };
}

/** One learning context whose single succeeded episode echoes `text`. */
function context(text = "hello"): LearnContext {
  const episode = new MemoryEpisode("run-1", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text }, {
    output: { answer: text.toUpperCase() },
  });
  return new LearnContext("run-1", "reflective", [episode]);
}

/** Split a reflection prompt into its four labelled lines, keyed by label. */
function promptParts(prompt: string): Record<string, string> {
  const labels = ["Memory: ", "Limits: ", "Current lessons: ", "Episodes: "];
  const lines = prompt.split("\n");
  expect(labels.map((label, i) => lines[i]!.slice(0, label.length))).toEqual(labels);
  return Object.fromEntries(labels.map((label, i) => [label, lines[i]!.slice(label.length)]));
}

test("reflection lesson material follows its budget", async () => {
  setDefaults({ YIAN_REFLECTIVE_MAX_LESSON_CHARS: "300" });
  const { memory, prompt } = noopLearner();
  const payload = { action: "add", lesson_id: "big", source: { run_id: "run-1" }, text: "x".repeat(5_000) };
  const big = new MemoryMutation("big", payload);
  const session = memory.openRun();
  session.commit("big", [big]);
  session.close();
  const learning = memory.openRun();
  expect(await learning.learn(context())).toEqual([]);
  learning.close();
  const lessons = promptParts(prompt())["Current lessons: "]!;
  expect(Array.from(lessons).length).toBeLessThanOrEqual(300);
  expect(JSON.parse(lessons).truncated).toBe(true);
});

test("reflection episode material follows its budget", async () => {
  setDefaults({ YIAN_REFLECTIVE_MAX_EPISODE_CHARS: "400" });
  const { memory, prompt } = noopLearner();
  const session = memory.openRun();
  expect(await session.learn(context("y".repeat(5_000)))).toEqual([]);
  session.close();
  const episodes = promptParts(prompt())["Episodes: "]!;
  expect(Array.from(episodes).length).toBeLessThanOrEqual(400);
  expect(JSON.parse(episodes).truncated).toBe(true);
});

test("profile memory budgets default from the environment", () => {
  setDefaults({ YIAN_PROFILE_MAX_CHARS: "100", YIAN_PROFILE_INPUT_BUDGET_CHARS: "50" });
  const memory = new ProfileMemory(new FileMarkdownStore(path.join(home, "ada.md")));
  expect([memory.maxChars, memory.inputBudgetChars]).toEqual([100, 50]);
  const explicit = new ProfileMemory(new FileMarkdownStore(path.join(home, "bob.md")), {
    maxChars: 200,
    inputBudgetChars: 70,
  });
  expect([explicit.maxChars, explicit.inputBudgetChars]).toEqual([200, 70]);
});
