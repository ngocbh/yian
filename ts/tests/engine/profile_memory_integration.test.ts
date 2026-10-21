/**
 * Public runs, backward attempts and recovery with `ProfileMemory`: mirrors
 * `tests/engine/test_profile_memory_integration.py`.
 *
 * Ported, through `runFlow`: `test_sibling_agents_recall_one_shared_profile_document`.
 *
 * Changed: one `ScriptedModel` serves both AGENTs, where Python builds a chat per AGENT; the run's
 * environment names the default model (conftest's in Python), since the flow names none.
 *
 * Later (T6-6, with RunResult.backward): `test_recall_learn_and_commit_through_a_public_run`,
 * `test_nested_declarations_of_one_key_show_and_rewrite_the_document_once`,
 * `test_profile_notes_and_a_session_transcript_share_one_agent_context`,
 * `test_human_edit_during_learning_fails_the_commit_without_a_model`,
 * `test_a_new_attempt_after_a_mismatch_learns_from_the_edited_document`, and, through
 * `resumeFlow` too, `test_resume_reopens_the_recorded_version_for_backward` and
 * `test_an_attempt_killed_after_its_write_is_unknown_and_findable_in_provenance`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import { reply, runWith } from "./session_fakes.ts";

const SIBLINGS = `id: siblings
name: siblings
input:
  message: !str
nodes:
  first:
    kind: agent
    long_term_memory: profile
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
  second:
    kind: agent
    long_term_memory: profile
    input:
      message: \${input.message}
    prompt: \${message}
    output: str
output: \${first.output}
`;

let dir: string;
let env: Env;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-profile-integration-"));
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(dir, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "test-model"],
  ]);
  env = (name) => vars.get(name);
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** Python's `_profile`: Ada's document in a file store, learned by a local model. */
async function profile(): Promise<ProfileMemory> {
  const store = new FileMarkdownStore(join(dir, "ada.md"));
  await store.write("# Ada\n- Works on reefs.\n", { baseVersion: (await store.read())[1] });
  return new ProfileMemory(store, { llmConfig: { provider: "ollama", model: "local-profiler" }, env });
}

test("sibling agents recall one shared profile document", async () => {
  const model = new ScriptedModel([reply("a"), reply("b")]);
  const memory = new Map([["profile", await profile()]]);
  const result = await runWith(await loadFlow(SIBLINGS), { message: "hi" }, memory, model, { env });
  expect(result.status).toBe("succeeded");
  const systems = model.requests.map((request) => request.messages[0]!);
  expect(systems).toHaveLength(2);
  for (const system of systems) {
    expect(system.role).toBe("system");
    expect(system.text().split("Works on reefs.")).toHaveLength(2);
  }
});
