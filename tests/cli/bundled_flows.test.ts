/**
 * The flows `yian chat` bundles (`src/cli/flows/`): the assistant and the compaction it calls
 * load, and the assistant's reply agent has the tools, skill and steering the chat gives it.
 */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { runFlow, staticNodes } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { builtinType } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals } from "../../src/typesys/values.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

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

test("the bundled chat loads, with an Inbox(Message) and optional flow and focus", async () => {
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
  expect(inputs).toEqual([
    ["flow", "Optional(str)", false],
    ["focus", "Optional(str)", false],
  ]);
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
  ]) {
    expect(text).toContain(phrase);
  }
});

/**
 * Run the bundled report flow on `problem` and `home`, its draft answered with `report` by a
 * scripted model; return what reached file_issue and the flow's output.
 */
async function runReport(report: { title: string; body: string }, home = "") {
  const flow = await loadFlow(flowText("report_issue.yaml"), { sourceLabel: "report_issue.yaml", env: NO_ENV });
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
    { problem: "check crashed", context: "yian version: 0.0.4", home },
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
    [
      "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
      "[redacted private key]",
    ],
  ];
  const body = [
    ...secrets.map(([secret]) => secret),
    "log: /home/ada/.yian/runs/r1/log.jsonl",
    "not hers: /home/adam/x",
  ].join("\n");
  const title = "crash with ghp_abcdefghijklmnopqrstuvwxyz0123456789";
  const { filed } = await runReport({ title, body }, "/home/ada");
  expect(filed[0]!["title"]).toBe("crash with [redacted]");
  expect((filed[0]!["body"] as string).split("\n")).toEqual([
    ...secrets.flatMap(([, redacted]) => redacted!.split("\n")),
    "log: ~/.yian/runs/r1/log.jsonl",
    "not hers: /home/adam/x",
  ]);
});
