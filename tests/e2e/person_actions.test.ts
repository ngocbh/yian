/**
 * A person's action on an agent's tool, sent to a chat flow's inbox: the flow files it without a
 * turn (`p.type == "action"`), and it reaches the next model call signed by the person, as the host
 * redacted it, with nothing it left out in the run log, the events or the request.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { RunEvent } from "../../src/events.ts";
import { withAuthorWrapper } from "../../src/llm/body.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { send } from "../../src/run.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { resumeWith, runWith, textReply } from "../helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-person-actions-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** A chat turn per message: an action-only message is filed in `noted`, any other is answered after the noted ones. */
const FLOW = `
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
input:
  messages: !Inbox(Message)
defs:
  turn:
    input:
      noted: !list(Message)
      messages: !Inbox(Message)
    nodes:
      heard:
        kind: call
        call: take
        input: {from: "\${input.messages}", take: one}
      route:
        kind: case
        cases:
          - when: \${not all([len(m.parts) > 0 and all([p.type == "action" for p in m.parts]) for m in heard.output.events])}
            then: reply
      reply:
        kind: agent
        input: {noted: "\${input.noted}", said: "\${heard.output.events}"}
        steer_inbox: \${input.messages}
        prompt:
          - \${noted}
          - \${said}
        output: !str
    output:
      noted: \${input.noted + heard.output.events if reply.output == null else []}
nodes:
  chat:
    kind: loop
    call: turn
    input: {noted: [], messages: "\${input.messages}"}
    times: 2
output: \${chat.output.noted}
`;

/** What the person typed into the shell; the host leaves it out of the action. */
const SECRET = "hunter2";
const ACTION = { role: "user", author: "Ngoc", parts: [{ type: "action", tool: "shell_type", summary: "Ngoc typed into shell 1" }] };

test("an action is filed without a model call, then joins the next turn signed and redacted", async () => {
  const log = join(mkdtempSync(join(tmp, "run-")), "log.jsonl");
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const events: RunEvent[] = [];
  const options = { log, onEvent: (event: RunEvent) => events.push(event) };
  const flow = await loadFlow(FLOW);
  expect((await runWith(flow, {}, model, options)).status).toBe("waiting");

  await send(log, "messages", ACTION);
  const filed = await resumeWith(flow, log, model, options);
  expect(filed.status, filed.failure?.message).toBe("waiting");
  expect(model.requests).toHaveLength(0);

  // A delivery that would echo the payload is refused, and the log keeps nothing of it.
  const before = readFileSync(log, "utf8");
  const raw = { ...ACTION, parts: [{ type: "action", tool: "shell_type", arguments: SECRET }] };
  await expect(send(log, "messages", raw)).rejects.toThrow(DeliveryRefused);
  expect(readFileSync(log, "utf8")).toBe(before);

  await send(log, "messages", { role: "user", parts: [{ type: "text", text: "what did I run?" }] });
  const done = await resumeWith(flow, log, model, options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toEqual([]);
  expect(model.requests).toHaveLength(1);
  const [action, said] = model.requests[0]!.messages.slice(-2).map((message) => withAuthorWrapper(message).toJSON());
  expect(action).toEqual({
    format: "yian.message.v1",
    role: "user",
    author: "Ngoc",
    parts: [
      { type: "text", text: '<authored_message author="Ngoc">' },
      { type: "text", text: 'action: {"tool":"shell_type","summary":"Ngoc typed into shell 1"}' },
      { type: "text", text: "</authored_message>" },
    ],
  });
  expect(said).toEqual({ format: "yian.message.v1", role: "user", parts: [{ type: "text", text: "what did I run?" }] });
  const text = (value: unknown) => JSON.stringify(value, (_, each: unknown) => (typeof each === "bigint" ? String(each) : each));
  for (const where of [readFileSync(log, "utf8"), text(events), text(model.requests[0]!.messages)]) {
    expect(where).not.toContain(SECRET);
  }
});
