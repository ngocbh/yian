/**
 * A child run's view (`run_view.ts`): its progress read from a real run log as it grows, and its
 * screen, in colour and without.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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

/** A CODE node that fails with a long message holding a tab. */
const BOOM = `id: boom
name: boom
nodes:
  boom:
    kind: code
    output: str
    code: |
      throw new Error("a\\tb " + "word ".repeat(12));
output: \${boom.output}
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
      const { lines: rows, earlier } = progress.rows(Date.parse("2026-01-01T00:01:05Z"));
      expect(earlier).toBe(0);
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
      expect(progress.rows(Date.parse("2026-01-01T00:01:05Z")).lines).toEqual(rows);
    },
    20_000,
  );

  test(
    "reads only what the log gained, not a line still being written, and all again once it is rewritten",
    async () => {
      const full = readFileSync(await pickToQuestion(), "utf8");
      const lines = full.split("\n").filter(Boolean);
      const log = join(tmp, "grow.jsonl");
      const at = lines.findIndex((line) => line.includes('"node_succeeded"') && line.includes('"choose"'));
      // Up to the agent's success, then half of the next line.
      writeFileSync(log, `${lines.slice(0, at).join("\n")}\n${lines[at]!.slice(0, 20)}`);
      const progress = new RunProgress(log, () => 60, PLAIN, NO_ENV);
      const now = Date.parse("2026-01-01T00:00:00Z");
      expect(progress.rows(now).lines).toEqual(["⏺ options()", "  ⎿  red · blue", "⏺ blue", "· choose [0s]"]);
      writeFileSync(log, full);
      expect(progress.rows(now).lines).toEqual([
        "⏺ options()",
        "  ⎿  red · blue",
        "⏺ blue",
        "✓ choose [0s]",
        "✓ route → sure [0s]",
        "◷ sure [0s] waiting",
      ]);
      // Shorter than what was read: read again from the start, at once.
      writeFileSync(log, `${lines.slice(0, at).join("\n")}\n`);
      expect(progress.rows(now).lines).toEqual(["⏺ options()", "  ⎿  red · blue", "⏺ blue", "· choose [0s]"]);
    },
    20_000,
  );

  test(
    "wraps a long failure under its indent at the screen's width, its tabs expanded",
    async () => {
      const log = join(tmp, "fail.jsonl");
      const flow = await loadFlow(BOOM, { env: NO_ENV });
      const result = await runFlow(flow, {}, { log, env: NO_ENV });
      expect(result.status).toBe("failed");
      const { lines } = new RunProgress(log, () => 30, PLAIN, NO_ENV).rows(0);
      expect(lines[0]).toMatch(/^✗ boom \[\d+s\]$/);
      expect(lines.slice(1).every((line) => line.startsWith("    ") && line.length <= 30)).toBe(true);
      expect(lines.join(" ")).toContain("a    b");
      expect(lines.join("")).not.toContain("\t");
    },
    20_000,
  );

  test("of a log not written yet is empty", () => {
    const progress = new RunProgress(join(tmp, "none.jsonl"), () => 60, PLAIN, NO_ENV);
    expect(progress.rows(0)).toEqual({ lines: [], earlier: 0 });
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
    const progress = { lines: ["✓ choose [0s]"], earlier: 0 };
    const rows = runScreen({ id: "r1", name: "pick.yaml", status }, progress, 60, 24, COLOR, NO_ENV);
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

  test("says in yellow when the chat waits on the user behind it", () => {
    const rows = runScreen({ id: "r1", name: "pick.yaml", status }, { lines: [], earlier: 0 }, 60, 24, COLOR, NO_ENV, true);
    expect(shown(rows).at(-1)).toBe("the chat waits on you · esc to answer");
    expect(rows.at(-1)).toContain("\x1b[33m");
  });

  test("keeps the last rows of a long progress, under a count of those left out", () => {
    const lines = Array.from({ length: 30 }, (_, i) => `✓ n${i}`);
    const done: ChildStatus = { run: "r1", status: "succeeded", seq: 9, log: "l", output: "ok" };
    const rows = runScreen({ id: "r1", name: "x.yaml", status: done }, { lines, earlier: 0 }, 60, 10, PLAIN, NO_ENV);
    const kept = ["… 26 earlier rows", "✓ n26", "✓ n27", "✓ n28", "✓ n29"];
    expect(rows).toEqual(["r1  x.yaml  ● succeeded", "log  l", "", ...kept, "", "esc to go back to the chat"]);
    expect(rows.length).toBe(10);
    // The rows the progress no longer keeps count too.
    const few = runScreen({ id: "r1", name: "x.yaml", status: done }, { lines: ["✓ n"], earlier: 5 }, 60, 10, PLAIN, NO_ENV);
    expect(few.slice(3, 5)).toEqual(["… 5 earlier rows", "✓ n"]);
  });
});
