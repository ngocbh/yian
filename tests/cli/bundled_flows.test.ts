/**
 * The flows `yian chat` bundles (`src/cli/flows/`): the assistant and the compaction it calls
 * load, and the assistant's reply agent has the tools, skill and steering the chat gives it.
 */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { staticNodes } from "../../src/run.ts";
import { builtinType } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals } from "../../src/typesys/values.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

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
  ]);
  expect(reply.skills).toEqual(["composing-agents"]);
  expect(reply.steerInbox).toBe(true);
  expect(Object.hasOwn(reply.env, "max_tool_iterations")).toBe(false);
});

test("the prompt reads before explaining, checks before runs, and relays inbox runs", async () => {
  // Read with its lines joined: the prompt wraps.
  const text = flowText("chat.yaml").replace(/\s+/g, " ");
  expect(text).toContain("write_flow refuses content that doesn't load; fix what it reports.");
  expect(text).toContain("Check a flow before you run it.");
  expect(text).toContain(
    "To explain or change a flow, read it with read_flow first; " +
      "check_flow only says whether it loads and lists its inputs.",
  );
  expect(text).toContain("refuse a path outside it");
  expect(text).not.toContain("Check a flow before you write it");
  for (const phrase of [
    "message_flow",
    "close_inbox when the user is done with it",
    "what a run said",
  ]) {
    expect(text).toContain(phrase);
  }
});
