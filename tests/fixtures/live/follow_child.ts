/**
 * The writing side of `tests/engine/live_follow.test.ts`'s two-process test: a host that runs a
 * flow and forwards its live channel over stdout, with the pipe standing in for the host's channel.
 *
 * Run as `bun --no-install tests/fixtures/live/follow_child.ts <log>` (or with Node). It runs a MAP
 * of two streaming AGENTs over the topics `a` and `b`, whose model streams one delta per gate, with
 * its log at `<log>`, and writes one `logJson` line per message on stdout:
 * - `{"type":"ready"}` once both calls have started;
 * - for each stdin line `release <topic>`, which opens that call's next gate, `{"type":"ack"}` once
 *   the chunk or record it lets out was sent;
 * - for the stdin line `follow`, every frame `LiveChannel.follow` sends (the snapshot first), then
 *   `{"type":"ack"}`; a later `follow` replaces it;
 * - `{"type":"done","status":...}` when the run ends.
 *
 * It exits when stdin closes, and after 60 seconds in any case.
 */

import { createInterface } from "node:readline";

import { loadFlow } from "../../../src/compose/loader.ts";
import { runFlow } from "../../../src/run.ts";
import { LiveChannel } from "../../../src/runtime/live.ts";
import { logJson } from "../../../src/typesys/json.ts";
import { GatedModel, MAP_AGENTS, until } from "../../engine/live_fixtures.ts";

// A bound, so a stuck run can't outlive its test.
setTimeout(() => process.exit(2), 60_000).unref();

const [log] = process.argv.slice(2);
const write = (message: unknown): void => void process.stdout.write(`${logJson(message)}\n`);

const live = new LiveChannel();
const model = new GatedModel({ a: ["a0", "a1", "a2"], b: ["b0", "b1"] });
let sent = 0;
live.subscribe((frame) => void (frame.type !== "start" && frame.type !== "end" && sent++));
const env = (): undefined => undefined;
const run = runFlow(await loadFlow(MAP_AGENTS), { topics: ["a", "b"] }, {
  log: log!,
  env,
  models: { env, createModel: () => model },
  numWorkers: 2,
  live,
});
void run.then(
  (result) => write({ type: "done", status: result.status }),
  (error: unknown) => write({ type: "done", status: String(error) }),
);
await until(() => live.snapshot().calls.length === 2);
write({ type: "ready" });

let stop = (): void => undefined;
for await (const line of createInterface({ input: process.stdin })) {
  if (line === "follow") {
    // Following again replaces the earlier follow, so no frame is written twice.
    stop();
    stop = live.follow(write);
  } else if (line.startsWith("release ")) {
    const before = sent;
    model.release(line.slice("release ".length));
    await until(() => sent > before);
  }
  write({ type: "ack" });
}
