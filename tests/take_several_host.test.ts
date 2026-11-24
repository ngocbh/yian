/**
 * TS-only: a take whose `from:` lists several inboxes, through the host calls: it waits on each,
 * names each message's inbox by the flow's own input name, closes only when each is closed, and a
 * withdraw from one inbox leaves the other's message with it.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { advance, close, inspectRun, runFlow, send, withdraw } from "../src/run.ts";
import { hostEnv } from "./helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-take-several-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
const env = hostEnv(dir);

/** Run `text` to its first wait; return its log and the options `advance` takes. */
async function waiting(text: string) {
  const log = join(dir, `run-${++logs}.jsonl`);
  const options = { env, log, runId: "r" };
  const result = await runFlow(await loadFlow(text), {}, options);
  expect(result.status, result.failure?.message).toBe("waiting");
  return { log, options, result };
}

/** Collects `messages` and `queued` until both close, each message as `<inbox>:<text>`. */
const COLLECT = `
id: collect
name: collect
input:
  messages: !Inbox(str)
  queued: !Inbox(str)
nodes:
  gather:
    kind: loop
    input:
      messages: \${input.messages}
      queued: \${input.queued}
      seen: !list(str) []
      closed: false
    while: not closed
    max: 10
    nodes:
      heard:
        kind: call
        call: take
        input:
          from:
            - \${input.messages}
            - \${input.queued}
          take: all
    output:
      seen: \${input.seen + [e.inbox + ":" + e.event for e in heard.output.events]}
      closed: \${heard.output.closed}
output: \${gather.output.seen}
`;

test("a take of two inboxes wakes for either, in seq order, and closes once both are closed", async () => {
  const { log, options, result } = await waiting(COLLECT);
  expect(result.waits.map((wait) => wait.inbox)).toEqual([["messages", "queued"]]);
  await send(log, "queued", "q1", { env });
  await send(log, "messages", "m1", { env });
  expect((await advance(options)).status).toBe("waiting");
  await close(log, "messages", { env });
  // One inbox still open: the take waits on.
  expect((await advance(options)).status).toBe("waiting");
  await send(log, "queued", "q2", { env });
  await close(log, "queued", { env });
  const done = await advance(options);
  expect([done.status, done.output]).toEqual(["succeeded", ["queued:q1", "messages:m1", "queued:q2"]]);
});

/** A def that takes one message from `chat` or `later`, passed `messages` and `queued`. */
const RENAMED = `
id: renamed
name: renamed
input:
  messages: !Inbox(str)
  queued: !Inbox(str)
defs:
  turn:
    input:
      chat: !Inbox(str)
      later: !Inbox(str)
    nodes:
      heard:
        kind: call
        call: take
        input:
          from:
            - \${input.chat}
            - \${input.later}
    output: \${heard.output}
nodes:
  first:
    kind: call
    call: turn
    input:
      chat: \${input.messages}
      later: \${input.queued}
output: \${first.output}
`;

test("a def's take names each inbox by the def's own input", async () => {
  const { log, options } = await waiting(RENAMED);
  await send(log, "queued", "soon", { env });
  const done = await advance(options);
  expect(done.output).toEqual({ events: [{ inbox: "later", event: "soon" }], closed: false, closed_inboxes: [] });
});

test("one inbox passed under both names fails the take", async () => {
  const text = RENAMED.replace("later: ${input.queued}", "later: ${input.messages}").replace(
    "nodes:\n  first:",
    "nodes:\n  keep:\n    kind: call\n    call: take\n    input:\n      from: ${input.queued}\n      timeout: 1s\n  first:",
  );
  const result = await runFlow(await loadFlow(text), {}, { env, log: join(dir, `run-${++logs}.jsonl`), runId: "r" });
  expect([result.status, result.failure?.code]).toEqual(["failed", "contract.input"]);
  expect(result.failure?.message).toContain("take `from` lists inbox 'messages' twice");
});

test("a withdraw from one inbox takes back only its own message from a take of two", async () => {
  const { log } = await waiting(RENAMED);
  const queued = await send(log, "queued", "soon", { env });
  expect(inspectRun(log).unread.map((sent) => [sent.inbox, sent.seq])).toEqual([["queued", queued]]);
  expect(await withdraw(log, "messages", { env })).toEqual([]);
  expect(await withdraw(log, "queued", { env })).toEqual([{ seq: queued, value: "soon", by: null }]);
  expect(inspectRun(log).unread).toEqual([]);
});
