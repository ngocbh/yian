/**
 * The earlier chats `/resume` lists (`chatSessions`) and what a new chat takes from one
 * (`sessionSeed`): bundled chat runs made in a temporary runs folder with a scripted model.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadChatFlow } from "../../src/cli/chat.ts";
import {
  chatSessions,
  chosenModel,
  modelMessage,
  sessionSeed,
  sessionTitle,
} from "../../src/cli/chat_sessions.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, close, runFlow, runLogPath, send } from "../../src/run.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

const TIMEOUT = 20_000;

let tmp: string;
let runsDir: string;
let root: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-sessions-")));
  runsDir = join(tmp, "runs");
  root = join(tmp, "project");
  mkdirSync(runsDir);
  mkdirSync(root);
});

afterEach(() => rmSync(tmp, { recursive: true, force: true }));

/** A stand-in for each tool the reply agent is granted. */
function stubs() {
  const names = [
    "list_flows", "read_flow", "check_flow", "write_flow", "read_docs", "start_flow",
    "flow_status", "wait_flow", "answer_flow", "message_flow", "close_inbox", "stop_flow",
    "report_issue", "author_flow", "review_flow",
  ];
  const parameters = { type: "object", properties: {} };
  return Object.fromEntries(
    names.map((name) => [name, tool({ name, description: name, parameters, execute: () => "" })]),
  );
}

const SKILL = new Skill({ name: "composing-agents", description: "Compose.", instructions: "." });

/** A user's line. */
const line = (text: string) => ({ role: "user", parts: [{ type: "text", text }] });

/**
 * Run a chat `runId` in `folder`: send each of `sent` (a line, or a message) and advance after
 * each, the model replying `re: <n>`; close it when `end`. Returns its log.
 */
async function chatRun(
  runId: string,
  folder: string,
  sent: readonly (string | Record<string, unknown>)[],
  { end = true, parent = null }: { end?: boolean; parent?: string | null } = {},
): Promise<string> {
  const replies = Array.from({ length: sent.length }, (_, i) => textReply(`re: ${i + 1}`));
  const model = new ScriptedModel(replies, { provider: "xai", model: "grok-4.3" });
  const options = {
    env: NO_ENV,
    tools: stubs(),
    skills: new Map([[SKILL.name, SKILL]]),
    llmConfig: { provider: "xai", model: "grok-4.3" },
    models: serving(model),
  };
  const log = runLogPath(runsDir, runId);
  const flow = await loadChatFlow(NO_ENV, folder);
  await runFlow(flow, { flow: null, focus: null }, {
    ...options,
    log,
    runId,
    parent: parent === null ? null : { runId: parent },
  });
  for (const each of sent) {
    await send(log, "owner_messages", typeof each === "string" ? line(each) : each);
    await advance({ ...options, log });
  }
  if (end) {
    await close(log, "owner_messages");
    expect((await advance({ ...options, log })).status).toBe("succeeded");
  }
  return log;
}

test("a model message reads back as its choice, and no other message does", () => {
  const choice = { provider: "openai", model: "gpt-5.5" };
  expect(chosenModel(modelMessage(choice))).toEqual(choice);
  // A user's line isn't a choice, even one written as one.
  expect(chosenModel(line('/model {"provider":"a","model":"b"}'))).toBeNull();
  expect(chosenModel({ ...line("hi"), author: "yian" })).toBeNull();
  expect(chosenModel({ ...line("/model {not json"), author: "yian" })).toBeNull();
  // A name with a space or a newline goes through whole.
  const odd = { provider: "vllm", model: "my model\nv2" };
  expect(chosenModel(modelMessage(odd))).toEqual(odd);
});

test(
  "the folder's chats are listed newest first, with their first line and model",
  async () => {
    await chatRun("s1", root, ["first chat", modelMessage({ provider: "xai", model: "grok-4" })]);
    await chatRun("s2", join(tmp), ["another folder"]);
    await chatRun("s3", root, [], { end: true });
    await chatRun("s4", root, ["second chat", "more"], { end: false });
    const sessions = chatSessions(runsDir, root);
    // A chat shows the model it chose, else the one its last reply used; only a choice carries.
    const grok4 = { provider: "xai", model: "grok-4" };
    expect(sessions.map((each) => [each.runId, each.firstLine, each.model, each.chosen])).toEqual([
      ["s4", "second chat", { provider: "xai", model: "grok-4.3" }, null],
      ["s1", "first chat", grok4, grok4],
    ]);
    const title = /^\d{4}-\d\d-\d\d \d\d:\d\d · xai\/grok-4 · first chat$/;
    expect(sessionTitle(sessions[1]!, 80)).toMatch(title);
    expect(sessionTitle(sessions[1]!, 30)).toBe(`${sessionTitle(sessions[1]!, 80).slice(0, 29)}…`);
    expect(chatSessions(runsDir, join(tmp, "nowhere"))).toEqual([]);
  },
  TIMEOUT,
);

test(
  "a seed holds a chat's transcript, from a finished one or one left waiting, and its model",
  async () => {
    await chatRun("done", root, ["remember 7", modelMessage({ provider: "xai", model: "grok-4" })]);
    await chatRun("left", root, ["remember 9"], { end: false });
    const [left, done] = chatSessions(runsDir, root);
    const finished = await sessionSeed(done!, runsDir, NO_ENV);
    expect(finished.model).toEqual({ provider: "xai", model: "grok-4" });
    expect(finished.transcript.summary).toBe("");
    // Turn by turn.
    const texts = (seed: typeof finished) =>
      seed.transcript.history.map((turn) =>
        turn.map((m) => (m as { parts: { text?: string }[] }).parts[0]!.text),
      );
    expect(texts(finished)).toEqual([["remember 7", "re: 1"]]);
    const waiting = await sessionSeed(left!, runsDir, NO_ENV);
    expect(texts(waiting)).toEqual([["remember 9", "re: 1"]]);
    // It ran on the defaults, which a chat going on from it uses too.
    expect(waiting.model).toBeNull();
  },
  TIMEOUT,
);

test(
  "a seed lists the earlier chat's children left waiting, not those that ended",
  async () => {
    await chatRun("chat", root, ["hi"]);
    const waiting = await chatRun("chat-r1", tmp, ["child"], { end: false, parent: "chat" });
    await chatRun("chat-r2", tmp, ["child"], { end: true, parent: "chat" });
    const [session] = chatSessions(runsDir, root);
    const seed = await sessionSeed(session!, runsDir, NO_ENV);
    expect(seed.waiting).toEqual([
      { run: "chat-r1", kind: "inbox", log: waiting },
    ]);
  },
  TIMEOUT,
);
