/**
 * Memory defaults come from `defaults.ts`: a port of `tests/engine/test_memory_defaults.py`.
 *
 * Ported: `test_episodic_limits_default_from_the_environment`.
 *
 * TS-only: none.
 *
 * Changed: none.
 *
 * Later: `test_session_memory_settings_default_from_the_environment` (with `memory/session.ts`).
 *
 * Later (T6-5): `test_reflective_limits_default_from_the_environment`,
 * `test_reflection_lesson_material_follows_its_budget`,
 * `test_reflection_episode_material_follows_its_budget`,
 * `test_reflection_limit_material_follows_its_budget` and
 * `test_profile_memory_budgets_default_from_the_environment` (with the reflective and profile
 * backends).
 *
 * Not ported: `test_memory_nesting_bound_follows_the_value_nesting_limit` (TS has no memory nesting
 * limit).
 */

import { afterEach, expect, test } from "bun:test";

import { BUILTIN, install, reload } from "../../src/defaults.ts";
import type { JsonObject } from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";

afterEach(() => install(BUILTIN));

/** Return the pinned snapshot of one run of `memory`, optionally restoring `snapshot`. */
function snapshotOf(memory: EpisodicMemory, snapshot?: JsonObject): JsonObject {
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
