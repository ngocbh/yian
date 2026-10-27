/**
 * An 80-turn `examples/chat.yaml` run log grows linearly with the conversation, learnable or not.
 *
 * Each turn's message is sent to the run's `owner_messages` inbox as a user `Message` and the
 * run advanced; closing the inbox ends it. One 80-turn run per case, its log measured after turns
 * 10, 20, 40 and 80.
 *
 * Each turn's AGENT records its reply in the messages commit (`reply.new_messages`), and the
 * turn's output binding appends the turn (the take's events, then those new messages) to
 * `compacted.output.history`; the turn's END points at the take commit and the messages commit.
 * Each turn also adds the `compacted` expansion (START and END are
 * derived; CASE records only a handle); 100- and 200-character turns (24,000 characters in 80
 * turns) never reach compact_history's 32,000-character budget, so nothing folds. The bindings'
 * values (each turn's END, the expansions' child STARTs, the stopping LOOP driver) are derived
 * commits holding `from` pointers, not the history, so a turn adds about the same bytes however
 * long the conversation is. A learnable run's forward captures hold routes, not the bound inputs
 * or the rendered prompt, which learning rebuilds.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { NO_ENV, serving, textReply, userText } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-chat-log-growth-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const EXAMPLES_DIR = fileURLToPath(new URL("../../examples", import.meta.url));
const CONFIG = { provider: "anthropic", model: "claude-test" };
const MESSAGE = 100;
const REPLY = 200;

const TURNS = [10, 20, 40, 80];

/**
 * Run `examples/chat.yaml` for 80 turns of fixed-size messages and replies, then close its inbox;
 * return the log's size right after each of `TURNS` is advanced.
 */
async function logBytes(learnable: boolean): Promise<Map<number, number>> {
  const flow = await loadFlow(readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8"), { searchPaths: [EXAMPLES_DIR] });
  const turns = TURNS.at(-1)!;
  const replies = Array.from({ length: turns }, (_, k) => textReply(`${k}`.padEnd(REPLY, "r")));
  const log = join(tmp, `chat-${learnable}.jsonl`);
  const models = serving(new ScriptedModel(replies, CONFIG));
  const sizes = new Map<number, number>();
  let result = await runFlow(flow, {}, { env: NO_ENV, log, numWorkers: 0, learnable, llmConfig: CONFIG, models });
  for (let k = 0; k < turns; k += 1) {
    expect(result.status).toBe("waiting");
    await send(log, "owner_messages", userText(`${k}`.padEnd(MESSAGE, "m")), { env: NO_ENV });
    result = await advance({ log, env: NO_ENV, numWorkers: 0, models });
    if (TURNS.includes(k + 1)) sizes.set(k + 1, statSync(log).size);
  }
  await close(log, "owner_messages", { env: NO_ENV });
  result = await advance({ log, env: NO_ENV, numWorkers: 0, models });
  expect(result.status).toBe("succeeded");
  expect((result.output as { history: unknown[] }).history.length).toBe(2 * turns);
  return sizes;
}

for (const learnable of [false, true]) {
  test(`an 80-turn chat log grows linearly${learnable ? " in a learnable run" : ""}`, async () => {
    const sizes = await logBytes(learnable);
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
