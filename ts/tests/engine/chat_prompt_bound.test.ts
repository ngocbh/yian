/**
 * TS-only: a long `examples/chat.yaml` conversation keeps every reply request bounded, learnable
 * or not, and its run log stays linear while compact_history folds older turns into a summary.
 *
 * 80 turns of 2,000-character messages and 2,000-character replies (4,000 characters a turn),
 * with the chat's defaults (a 32,000-character budget, 4 turns kept). The `compacted` beside turn
 * n sees n - 1 turns and folds when they pass the budget: first at 9 turns (before turn 10), then
 * every fifth turn, so folds come before turns 10, 15, ..., 80, and none on the closing turn. Since
 * `compacted` runs before `answered`, every reply request holds at most the budget, the message,
 * the summary and the system text.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Request } from "../../src/llm/messages.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { NO_ENV, PromptModel, serving, SUMMARIZER_FIRST_LINE, systemText } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-chat-prompt-bound-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const CONFIG = { provider: "anthropic", model: "claude-test" };
const TURNS = 80;
const BUDGET = 32_000;
const MESSAGE = 2_000;
const SUMMARY_SIZE = 2_000;
const SUMMARY = "s".repeat(SUMMARY_SIZE);

/** A request's size: the characters of its messages' text. */
const size = (request: Request): number => request.messages.reduce((sum, m) => sum + m.text().length, 0);

/**
 * Run `examples/chat.yaml` for 80 turns, recording the log's size right after turns 20, 40 and 80
 * are advanced, then close its inbox; return the model, those sizes and the flow's output.
 */
async function chat(learnable: boolean) {
  const flow = await loadFlow(readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8"), { searchPaths: [EXAMPLES_DIR] });
  const model = new PromptModel({ reply: "r".repeat(MESSAGE), summary: SUMMARY });
  const models = serving(model);
  const log = join(tmp, `chat-${learnable}.jsonl`);
  const bytesAt = new Map<number, number>();
  let result = await runFlow(flow, {}, { env: NO_ENV, log, numWorkers: 0, learnable, llmConfig: CONFIG, models });
  for (let turn = 1; turn <= TURNS; turn += 1) {
    expect(result.status).toBe("waiting");
    await send(log, "messages", `${turn}:`.padEnd(MESSAGE, "m"), { env: NO_ENV });
    result = await advance({ log, env: NO_ENV, numWorkers: 0, models });
    if (turn === 20 || turn === 40 || turn === 80) bytesAt.set(turn, statSync(log).size);
  }
  await close(log, "messages", { env: NO_ENV });
  result = await advance({ log, env: NO_ENV, numWorkers: 0, models });
  expect(result.status).toBe("succeeded");
  return { model, bytesAt, output: result.output };
}

for (const learnable of [false, true]) {
  test(`an 80-turn chat folds 15 times and its reply prompt stays bounded${learnable ? " (learnable)" : ""}`, async () => {
    const started = performance.now();
    const { model, bytesAt, output } = await chat(learnable);
    const replies = model.requests.filter((r) => !systemText(r).startsWith(SUMMARIZER_FIRST_LINE));
    expect(replies).toHaveLength(80);
    expect(model.requests.length - replies.length).toBe(15); // folds before turns 10, 15, ..., 80
    const most = (from: number, to: number) => Math.max(...replies.slice(from, to).map(size));
    expect(most(0, 80)).toBeLessThanOrEqual(BUDGET + MESSAGE + SUMMARY_SIZE + 200); // 200: the system text
    expect(most(40, 80)).toBeLessThanOrEqual(1.05 * most(20, 40));
    // The fold before turn 80 keeps turns 76-79; turn 80 is appended; the closing turn's 5 turns
    // (20,000 characters) are under budget, so it doesn't fold.
    expect(output).toEqual({ summary: SUMMARY, history: expect.any(Array) });
    expect((output as { history: unknown[] }).history).toHaveLength(10);
    // The log stays linear, as chat_log_growth.test.ts checks, measured within the one run.
    const [at20, at40, at80] = [bytesAt.get(20)!, bytesAt.get(40)!, bytesAt.get(80)!];
    console.log(
      [
        `learnable: ${learnable} (${((performance.now() - started) / 1000).toFixed(1)} s)`,
        `largest reply request: turns 1-20 ${most(0, 20)}, 21-40 ${most(20, 40)}, 41-80 ${most(40, 80)} characters`,
        `log bytes: turn 20 ${at20}, turn 40 ${at40}, turn 80 ${at80}`,
        `bytes per turn: 21-40 ${((at40 - at20) / 20).toFixed(0)}, 41-80 ${((at80 - at40) / 40).toFixed(0)}`,
      ].join("\n"),
    );
    expect((at80 - at40) / 40).toBeLessThanOrEqual(1.15 * ((at40 - at20) / 20));
  }, 180_000);
}
