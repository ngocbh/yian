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
  sessionChoice,
  sessionSeed,
  spokenCount,
  timeAgo,
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
 * each, the model replying `re: <n>` to the n-th, and naming the chat `name` after its first
 * reply; close it when `end`. `inputs` go to its start. Returns its log.
 */
async function chatRun(
  runId: string,
  folder: string,
  sent: readonly (string | Record<string, unknown>)[],
  {
    end = true,
    parent = null,
    name = "A named chat",
    inputs = {},
  }: {
    end?: boolean;
    parent?: string | null;
    name?: string;
    inputs?: Record<string, unknown>;
  } = {},
): Promise<string> {
  const replies = sent.flatMap((each, i) => {
    if (typeof each !== "string" && chosenModel(each) !== null) return [];
    return [textReply(`re: ${i + 1}`)];
  });
  // The first reply has the chat named after it, unless it starts with a name.
  if (replies.length > 0 && inputs["title"] == null) replies.splice(1, 0, textReply(name));
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
  await runFlow(flow, { flow: null, focus: null, ...inputs }, {
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
    expect(chatSessions(runsDir, join(tmp, "nowhere"))).toEqual([]);
  },
  TIMEOUT,
);

test(
  "a chat is named once, from its log, and lists its messages; until then its first line stands in",
  async () => {
    await chatRun("named", root, ["fix my parser", "and its tests"], { name: "Fixing the parser" });
    // A name that's empty once its quotes go leaves the chat unnamed.
    await chatRun("unnamed", root, ["hello"], { name: '""' });
    await chatRun("seeded", root, ["go on"], { inputs: { title: "Fixing the parser" } });
    const sessions = chatSessions(runsDir, root);
    const shown = sessions.map((each) => [each.runId, each.title, each.firstLine, each.messages]);
    expect(shown).toEqual([
      // A chat going on from a named one keeps its name, with no call to name it.
      ["seeded", "Fixing the parser", "go on", 2],
      ["unnamed", null, "hello", 2],
      // The user's two lines and the two replies.
      ["named", "Fixing the parser", "fix my parser", 4],
    ]);
    const seed = await sessionSeed(sessions[2]!, runsDir, NO_ENV);
    expect(seed.title).toBe("Fixing the parser");
  },
  TIMEOUT,
);

test("a transcript's spoken messages are the user's and assistant's with text", () => {
  const tool = { role: "assistant", parts: [{ type: "tool_call", name: "read_docs" }] };
  const result = { role: "tool", parts: [{ type: "tool_result", content: "docs" }] };
  const reply = { role: "assistant", parts: [{ type: "text", text: "use map" }] };
  expect(spokenCount([[line("how?"), tool, result, reply], [line("thanks")]])).toBe(3);
  expect(spokenCount(null)).toBe(0);
});

test(
  "a chat going on from another counts the messages it started with",
  async () => {
    const reply = { role: "assistant", parts: [{ type: "text", text: "sure" }] };
    const transcript = { summary: "", history: [[line("earlier"), reply]] };
    await chatRun("on", root, ["and now"], { inputs: { transcript, title: "Earlier" } });
    expect(chatSessions(runsDir, root).map((each) => each.messages)).toEqual([4]);
  },
  TIMEOUT,
);

test("a time ago counts seconds, minutes, hours and days, then gives the UTC date", () => {
  const at = Date.parse("2026-10-31T12:00:00.000Z");
  const ago = (seconds: number) => timeAgo("2026-10-31T12:00:00.000Z", at + seconds * 1000);
  expect([ago(0), ago(59), ago(60), ago(3599), ago(3600), ago(86_399), ago(86_400)]).toEqual([
    "0s ago", "59s ago", "1m ago", "59m ago", "1h ago", "23h ago", "1d ago",
  ]);
  expect(ago(29 * 86_400)).toBe("29d ago");
  expect(ago(30 * 86_400)).toBe("2026-10-31");
  // A start a skewed clock puts ahead of now is just now.
  expect(ago(-5)).toBe("0s ago");
  expect(timeAgo("not a time", at)).toBe("not a time");
});

test("a session's picker choice is its name, then when, its messages and model", () => {
  const session = {
    runId: "r",
    log: "/runs/r/log.jsonl",
    startedAt: "2026-10-31T10:00:00.000Z",
    firstLine: "fix my\nparser",
    title: "Fixing the parser",
    messages: 4,
    model: { provider: "xai", model: "grok-4" },
    chosen: null,
  };
  const now = Date.parse("2026-10-31T12:00:00.000Z");
  expect(sessionChoice(session, now)).toEqual({
    title: "Fixing the parser",
    detail: "2h ago · 4 messages · xai/grok-4",
    also: "fix my parser",
  });
  // Unnamed, its first line stands in; one message, and no model the log names.
  const first = { ...session, title: null, messages: 1, model: null };
  const unnamed = { title: "fix my parser", detail: "2h ago · 1 message" };
  expect(sessionChoice(first, now)).toEqual(unnamed);
});

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
