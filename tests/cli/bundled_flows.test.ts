/**
 * The flows `yian chat` bundles (`src/cli/flows/`): the assistant and the compaction it calls
 * load, and the assistant's reply agent has the tools, skill and steering the chat gives it.
 */

import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { advance, runFlow, send, staticNodes } from "../../src/run.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { builtinType } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals } from "../../src/typesys/values.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

/** The skill the reply agent is granted, as a stand-in. */
const SKILL = new Skill({ name: "composing-agents", description: "Compose flows.", instructions: "Compose." });

const FLOWS = fileURLToPath(new URL("../../src/cli/flows/", import.meta.url));

/** The text of the bundled flow file `name`. */
const flowText = (name: string) => readFileSync(join(FLOWS, name), "utf8");

/** The bundled chat flow, its `uses:` resolved to the bundled compaction flow. */
async function loadChat() {
  const compaction = await loadFlow(flowText("compact_history.yaml"), {
    sourceLabel: "compact_history.yaml",
    env: NO_ENV,
  });
  return loadFlow(flowText("chat.yaml"), {
    sourceLabel: "chat.yaml",
    env: NO_ENV,
    childResolver: async (ref) => {
      expect(ref).toBe("compact_history");
      return compaction;
    },
  });
}

test("the bundled compaction flow is examples/compact_history.yaml", () => {
  const example = fileURLToPath(new URL("../../examples/compact_history.yaml", import.meta.url));
  expect(flowText("compact_history.yaml")).toBe(readFileSync(example, "utf8"));
});

test("the bundled chat loads, with an Inbox(Message), optional flow and focus, model and transcript", async () => {
  const flow = await loadChat();
  expect(flow.compiled.flowId).toBe("yian-chat");
  const [inbox, ...rest] = flow.input;
  expect([inbox!.name, inbox!.required]).toEqual(["owner_messages", true]);
  const type = inbox!.contract!.type;
  expect(isInboxType(type) && typeEquals(type!.element!, builtinType("Message")!)).toBe(true);
  const inputs = rest.map((param) => [
    param.name,
    canonicalTypeLabel(param.contract!.type),
    param.required,
  ]);
  expect(inputs.slice(0, 3)).toEqual([
    ["flow", "Optional(str)", false],
    ["focus", "Optional(str)", false],
    ["model", 'Optional({"model":str,"provider":str})', false],
  ]);
  expect([inputs[3]![0], inputs[3]![2]]).toEqual(["transcript", false]);
});

test("the reply agent gets the tools, the skill and steering, with no tool-call cap", async () => {
  const flow = await loadChat();
  const replies = staticNodes(flow).filter(([path]) => path.endsWith("/reply"));
  expect(replies).toHaveLength(1);
  const reply = replies[0]![1] as AgentNode;
  expect(reply.kind).toBe("agent");
  expect(reply.tools).toEqual([
    "list_flows",
    "read_flow",
    "check_flow",
    "write_flow",
    "read_docs",
    "start_flow",
    "flow_status",
    "wait_flow",
    "answer_flow",
    "message_flow",
    "close_inbox",
    "stop_flow",
    "report_issue",
    "author_flow",
    "review_flow",
  ]);
  expect(reply.skills).toEqual(["composing-agents"]);
  expect(reply.steerInbox).toBe(true);
  expect(Object.hasOwn(reply.env, "max_tool_iterations")).toBe(false);
});

test("the prompt reads before explaining, checks before runs, and relays inbox runs", async () => {
  // Read with its lines joined: the prompt wraps.
  const text = flowText("chat.yaml").replace(/\s+/g, " ");
  expect(text).toContain("Read a flow before you explain or change it, and check it before you run it.");
  expect(text).not.toContain("Check a flow before you write it");
  for (const phrase of [
    "message_flow",
    "close_inbox when the user is done with it",
    "what a run said",
    "offer to report it with report_issue",
    "if the user pastes one, tell them to revoke it",
    "Build a new flow or make a major change with author_flow; make a small edit with " +
      "write_flow directly.",
  ]) {
    expect(text).toContain(phrase);
  }
});

/**
 * Run the bundled report flow on `problem` and `homes`, its draft answered with `report` by a
 * scripted model; return what reached file_issue and the flow's output.
 */
async function runReport(report: { title: string; body: string }, homes: string[] = []) {
  const flow = await loadFlow(flowText("report_issue.yaml"), {
    sourceLabel: "report_issue.yaml",
    env: NO_ENV,
  });
  const filed: Record<string, unknown>[] = [];
  const fileIssue = tool({
    name: "file_issue",
    description: "Take the report.",
    parameters: {
      type: "object",
      properties: { title: { type: "string" }, body: { type: "string" } },
      required: ["title", "body"],
    },
    execute: (args) => {
      filed.push(args);
      return "not filed: saved only";
    },
  });
  // With a tool granted, the agent's tool loop ends in text, and a structured call follows.
  const replies = [textReply("Drafted."), textReply(JSON.stringify(report))];
  const model = new ScriptedModel(replies, { provider: "xai", model: "grok-4.3" });
  const result = await runFlow(
    flow,
    { problem: "check crashed", context: "yian version: 0.0.4", homes },
    {
      env: NO_ENV,
      tools: { file_issue: fileIssue, ask: () => pending() },
      llmConfig: { provider: "xai", model: "grok-4.3" },
      models: serving(model),
    },
  );
  return { filed, result, model };
}

test("the report flow drafts, scrubs and hands the report to file_issue", async () => {
  const { filed, result, model } = await runReport({ title: "check crashes", body: "It crashed." });
  expect(result.failure?.message ?? null).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("not filed: saved only");
  expect(filed).toEqual([{ title: "check crashes", body: "It crashed." }]);
  const system = model.requests[0]!.messages[0]!.text();
  expect(system).toContain("never paste a log's contents");
  expect(model.requests[0]!.tools.map((each) => each.name)).toContain("ask_user");
});

test("the report flow's scrub redacts each kind of secret and shows the home as ~", async () => {
  const secrets = [
    ["sk-proj-abcdefghijklmnopqrstuv", "[redacted]"],
    ["sk-ant-api03-abcdefghijklmnop_QRST", "[redacted]"],
    ["ghp_abcdefghijklmnopqrstuvwxyz0123456789", "[redacted]"],
    ["gho_abcdefghijklmnopqrstuvwxyz0123456789", "[redacted]"],
    ["github_pat_11ABCDEFG0123456789_abcdefghijklmnop", "[redacted]"],
    ["xoxb-1234567890-abcdefghij", "[redacted]"],
    ["AKIAABCDEFGHIJKLMNOP", "[redacted]"],
    ["AIzaSyA1234567890abcdefghijklmnopqrstuv", "[redacted]"],
    ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abcdefghijklmnop", "[redacted]"],
    ["Authorization: token abc123", "Authorization: [redacted]"],
    ["curl -H 'Bearer abc.def-123'", "curl -H 'Bearer [redacted]'"],
    ["password=hunter2", "password=[redacted]"],
    ['"api_key": "abc123"', '"api_key": [redacted]'],
    ["token: abc123", "token: [redacted]"],
    ["https://ada:hunter2@example.com/x", "https://[redacted]@example.com/x"],
    ["DATABASE_URL=postgres://u:pw@db/x", "DATABASE_URL=postgres://[redacted]@db/x"],
    ["x-api-key: abc123def", "x-api-key: [redacted]"],
    ["OPENAI_API_KEY=abcdefghijklmnopqrstu", "OPENAI_API_KEY=[redacted]"],
    ["AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG", "AWS_SECRET_ACCESS_KEY=[redacted]"],
    ["AZURE_OPENAI_KEY=abcdef", "AZURE_OPENAI_KEY=[redacted]"],
    ["ANTHROPIC_API_KEY = 'abc def'", "ANTHROPIC_API_KEY = [redacted]"],
    ["export GITHUB_TOKEN=ghp_short", "export GITHUB_TOKEN=[redacted]"],
    ["client_secret: abc", "client_secret: [redacted]"],
    ["pass: hunter2", "pass: [redacted]"],
    ["pwd=hunter2", "pwd=[redacted]"],
    ['{"api_key":"abc"}', '{"api_key":[redacted]}'],
    ["https://x.io/?key=abc123&x=1", "https://x.io/?key=[redacted]&x=1"],
    ["--api-key abc123", "--api-key [redacted]"],
    ["--token=abc123", "--token=[redacted]"],
    ["Cookie: session=abcdef123", "Cookie: [redacted]"],
    ["set-cookie: id=abc; Path=/", "set-cookie: [redacted]"],
    ["Bearer: abc", "Bearer: [redacted]"],
    ["sk_live_abcdefghijklmnopqrstuv", "[redacted]"],
    ["hf_abcdefghijklmnopqrstuvwxyz123456", "[redacted]"],
    ["npm_abcdefghijklmnopqrstuvwxyz0123456789", "[redacted]"],
    ["QmFzZTY0U2VjcmV0VmFsdWVUaGF0SXNMb25nRW5vdWdoVG9CZUFLZXk=", "[redacted]"],
    ["AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEf", "[redacted]"],
    [
      "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
      "[redacted private key]",
    ],
  ];
  const body = [
    ...secrets.map(([secret]) => secret),
    "log: /home/ada/.yian/runs/9eccdeca88574a9f96cba63e7564b67e/log.jsonl",
    "real: /gpfs/real/ada/x.yaml",
    "not hers: /home/adam/x",
    // Kept: a commit SHA, a sha256 digest, a child run's id, long identifiers.
    "commit 3e8c4a51e2b0c9d8f7a6b5c4d3e2f1a0b9c8d7e6",
    "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "run 9eccdeca88574a9f96cba63e7564b67e-r10-r2",
    "node chat_answered_reply_with_a_long_name_and_digit_1_in_it",
    "path src/cli/flows/report_issue_with_a_much_longer_name_v2.yaml",
    "yian check nap.yaml",
  ].join("\n");
  const title = "crash with ghp_abcdefghijklmnopqrstuvwxyz0123456789";
  const { filed } = await runReport({ title, body }, ["/home/ada", "/gpfs/real/ada/"]);
  expect(filed[0]!["title"]).toBe("crash with [redacted]");
  expect((filed[0]!["body"] as string).split("\n")).toEqual([
    ...secrets.flatMap(([, redacted]) => redacted!.split("\n")),
    "log: ~/.yian/runs/9eccdeca88574a9f96cba63e7564b67e/log.jsonl",
    "real: ~/x.yaml",
    "not hers: /home/adam/x",
    "commit 3e8c4a51e2b0c9d8f7a6b5c4d3e2f1a0b9c8d7e6",
    "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "run 9eccdeca88574a9f96cba63e7564b67e-r10-r2",
    "node chat_answered_reply_with_a_long_name_and_digit_1_in_it",
    "path src/cli/flows/report_issue_with_a_much_longer_name_v2.yaml",
    "yian check nap.yaml",
  ]);
});

/** A stand-in for each tool the chat's reply agent is granted; none is called here. */
function replyToolStubs() {
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

/** A user message holding `text`; authored `yian` when `yian`. */
function said(text: string, yian = false) {
  return { role: "user", ...(yian ? { author: "yian" } : {}), parts: [{ type: "text", text }] };
}

test(
  "a /model message chooses the replies' model, is not answered, and a resume replays it",
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "yian-chat-model-"));
    try {
      const flow = await loadChat();
      const log = join(dir, "log.jsonl");
      const model = new ScriptedModel([textReply("one"), textReply("two")], {
        provider: "xai",
        model: "grok-4.3",
      });
      const access = serving(model);
      const options = {
        env: NO_ENV,
        tools: replyToolStubs(),
        skills: new Map([[SKILL.name, SKILL]]),
        llmConfig: { provider: "xai", model: "grok-4.3" },
        models: access,
      };
      const first = await runFlow(flow, { flow: null, focus: null }, { ...options, log });
      expect(first.status).toBe("waiting");
      await send(log, "owner_messages", said("/model openai gpt-5.5", true));
      await send(log, "owner_messages", said("hi"));
      expect((await advance({ ...options, log })).status).toBe("waiting");
      // The choice isn't a line to answer: one request, the user's line, on the chosen model.
      expect(model.requests).toHaveLength(1);
      expect(JSON.stringify(model.requests[0]!.messages.map((m) => m.toJSON()))).not.toContain("/model");
      expect(access.configs.map((c) => [c["provider"], c["model"]])).toEqual([["openai", "gpt-5.5"]]);
      // Each advance replays the log: the next turn still uses the choice.
      await send(log, "owner_messages", said("again"));
      expect((await advance({ ...options, log })).status).toBe("waiting");
      expect(access.configs.map((c) => [c["provider"], c["model"]])).toEqual([
        ["openai", "gpt-5.5"],
        ["openai", "gpt-5.5"],
      ]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  20_000,
);

test("a transcript and a model given at the start seed the chat's first reply", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-chat-seed-"));
  try {
    const flow = await loadChat();
    const log = join(dir, "log.jsonl");
    const model = new ScriptedModel([textReply("sure")], { provider: "xai", model: "grok-4.3" });
    const access = serving(model);
    const options = {
      env: NO_ENV,
      tools: replyToolStubs(),
      skills: new Map([[SKILL.name, SKILL]]),
      llmConfig: { provider: "xai", model: "grok-4.3" },
      models: access,
    };
    const transcript = {
      summary: "The user is called Ada.",
      history: [said("my flow is hello.yaml"), { role: "assistant", parts: [{ type: "text", text: "Noted." }] }],
    };
    const inputs = { flow: null, focus: null, transcript, model: { provider: "xai", model: "grok-4" } };
    expect((await runFlow(flow, inputs, { ...options, log })).status).toBe("waiting");
    await send(log, "owner_messages", said("what did I say?"));
    await advance({ ...options, log });
    const request = JSON.stringify(model.requests[0]!.messages.map((m) => m.toJSON()));
    expect(request).toContain("The user is called Ada.");
    expect(request).toContain("my flow is hello.yaml");
    expect(request).toContain("Noted.");
    expect(access.configs.map((c) => [c["provider"], c["model"]])).toEqual([["xai", "grok-4"]]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
