/**
 * TS-only: an 80-turn `examples/chat.yaml` run log grows linearly with the conversation, learnable
 * or not.
 *
 * Each turn's AGENT outputs only its reply, and `respond`'s output binding appends the turn's two
 * messages to the history. The bindings' values (each turn's END, the expansion's child STARTs,
 * the stopping LOOP driver) are derived commits holding `from` pointers, not the history, so a
 * turn adds about the same bytes however long the conversation is. A learnable run's forward
 * captures hold routes, not the bound inputs or the rendered prompt, which learning rebuilds.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import { NO_ENV, serving, textReply } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-chat-log-growth-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const CONFIG = { provider: "anthropic", model: "claude-test" };
const MESSAGE = 100;
const REPLY = 200;

/**
 * Run `examples/chat.yaml` for `turns` turns of fixed-size messages and replies, then `/exit`;
 * return the log's size.
 */
async function logBytes(turns: number, learnable: boolean): Promise<number> {
  const flow = await loadFlow(readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8"), { searchPaths: [EXAMPLES_DIR] });
  const replies = Array.from({ length: turns }, (_, k) => textReply(`${k}`.padEnd(REPLY, "r")));
  let asked = 0;
  const log = join(tmp, `chat-${turns}-${learnable}.jsonl`);
  const result = await runFlow(flow, {}, {
    env: NO_ENV,
    log,
    numWorkers: 0,
    learnable,
    llmConfig: CONFIG,
    models: serving(new ScriptedModel(replies, CONFIG)),
    tools: { ask_user: () => (asked < turns ? `${asked++}`.padEnd(MESSAGE, "m") : "/exit") },
  });
  expect(result.status).toBe("succeeded");
  expect((result.output as unknown[]).length).toBe(2 * turns);
  return statSync(log).size;
}

for (const learnable of [false, true]) {
  test(`an 80-turn chat log grows linearly${learnable ? " in a learnable run" : ""}`, async () => {
    const sizes = new Map<number, number>();
    for (const turns of [10, 20, 40, 80]) sizes.set(turns, await logBytes(turns, learnable));
    const rows = [...sizes].map(([turns, bytes]) => {
      const transcript = turns * (MESSAGE + REPLY);
      return `${String(turns).padStart(5)} ${String(transcript).padStart(10)} ${String(bytes).padStart(10)} ${(bytes / transcript).toFixed(2).padStart(6)}`;
    });
    console.log([`learnable: ${learnable}`, "turns transcript  log bytes  ratio", ...rows].join("\n"));
    const inc1 = (sizes.get(40)! - sizes.get(20)!) / 20;
    const inc2 = (sizes.get(80)! - sizes.get(40)!) / 40;
    // A log that copies the history every turn grows quadratically, and doubles this increment.
    expect(inc2).toBeLessThanOrEqual(1.15 * inc1);
  }, 120_000);
}
