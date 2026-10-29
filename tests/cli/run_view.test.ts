/**
 * A child run's view (`run_view.ts`): its progress read from a real run log as it grows, and its
 * screen, in colour and without.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ChildStatus } from "../../src/cli/chat_runs.ts";
import { RunProgress, runScreen } from "../../src/cli/run_view.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { callReply, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-run-view-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const COLOR = { isTTY: true };
const PLAIN = { isTTY: false };
/** Drops the colours, for rows compared as the screen shows them. */
const shown = (rows: readonly string[]) => rows.map((row) => row.replace(/\x1b\[[0-9;]*m/g, ""));

/** Pick a colour with an agent that looks up the options, then ask the user to confirm it. */
const PICK = `id: pick
name: pick
nodes:
  choose:
    kind: agent
    prompt: Pick a colour.
    tools: [options]
    output: str
  route:
    kind: case
    on: \${choose.output}
    cases:
      - when: blue
        then: sure
    else: other
  other: !str not blue
  sure:
    kind: tool
    tool_id: ask
    input:
      prompt: "Is \${choose.output} right?"
    output: str
output: \${sure.output}
`;

/** Run PICK to its question on a virtual clock, the agent scripted; return its log. */
async function pickToQuestion(): Promise<string> {
  const log = join(tmp, "log.jsonl");
  const flow = await loadFlow(PICK, { env: NO_ENV });
  const model = new ScriptedModel(
    [callReply(["options", {}, "c1"]), textReply("blue")],
    { provider: "xai", model: "grok-4.3" },
  );
  const options = new FunctionTool({
    name: "options",
    description: "the colours",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "red\nblue",
  });
  const result = await runFlow(flow, {}, {
    log,
    tools: { options, ask: () => pending() },
    models: serving(model),
    llmConfig: { provider: "xai", model: "grok-4.3" },
    env: (name) => (name === "XAI_API_KEY" ? "test-key" : NO_ENV(name)),
    scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")),
  });
  expect(result.status).toBe("waiting");
  return log;
}

describe("a run's progress", () => {
  test(
    "draws each node that ended, an agent's tool calls and text, and a node still waiting",
    async () => {
      const log = await pickToQuestion();
      const progress = new RunProgress(log, () => 60, COLOR, NO_ENV);
      const rows = progress.rows(Date.parse("2026-01-01T00:01:05Z"));
      expect(shown(rows)).toEqual([
        "⏺ options()",
        "  ⎿  red · blue",
        "⏺ blue",
        "✓ choose [0s]",
        "✓ route → sure [0s]",
        "◷ sure [1m05s] waiting",
      ]);
      expect(rows[3]).toContain("\x1b[32m✓ choose");
      // Read again, it adds only what the log gained: nothing here.
      expect(progress.rows(Date.parse("2026-01-01T00:01:05Z"))).toEqual(rows);
    },
    20_000,
  );

  test("of a log not written yet is empty", () => {
    const progress = new RunProgress(join(tmp, "none.jsonl"), () => 60, PLAIN, NO_ENV);
    expect(progress.rows(0)).toEqual([]);
  });
});

describe("a run's screen", () => {
  const status: ChildStatus = {
    run: "r1",
    status: "waiting",
    seq: 3,
    log: "/runs/chat-r1/log.jsonl",
    asks: [{ call: "chat-r1/sure", question: "Is blue right?" }],
  };

  test("has the run, its flow, status and log, its progress and the question it waits on", () => {
    const rows = runScreen({ id: "r1", name: "pick.yaml", status }, ["✓ choose [0s]"], 60, 24, COLOR, NO_ENV);
    expect(shown(rows)).toEqual([
      "r1  pick.yaml  ● asks",
      "log  /runs/chat-r1/log.jsonl",
      "",
      "✓ choose [0s]",
      "",
      "? Is blue right?",
      "",
      "esc to go back to the chat",
    ]);
    expect(rows[0]).toContain("\x1b[33m● asks");
  });

  test("keeps the last rows of a long progress, under a count of those left out", () => {
    const progress = Array.from({ length: 30 }, (_, i) => `✓ n${i}`);
    const done: ChildStatus = { run: "r1", status: "succeeded", seq: 9, log: "l", output: "ok" };
    const rows = runScreen({ id: "r1", name: "x.yaml", status: done }, progress, 60, 10, PLAIN, NO_ENV);
    const kept = ["… 26 earlier rows", "✓ n26", "✓ n27", "✓ n28", "✓ n29"];
    expect(rows).toEqual(["r1  x.yaml  ● succeeded", "log  l", "", ...kept, "", "esc to go back to the chat"]);
    expect(rows.length).toBe(10);
  });
});
