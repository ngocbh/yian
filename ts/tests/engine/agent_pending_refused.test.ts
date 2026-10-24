/**
 * An AGENT's tool call can't wait for a later answer in this version: a tool (the blocking
 * `ask_user` or a host tool) that returns `pending()` fails the agent with an explicit error,
 * while one that answers inline works as before. TS-only: Python yian has no `pending()`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { runFlow } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { NO_ENV, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-pending-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** A flow whose one AGENT node `agent` grants `granted`. */
const agentFlow = (granted: string) => `
id: agent_pending
name: agent_pending
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [${granted}]
output: \${agent.output}
`;

/** A model that calls `name` with `args` once, then answers with the call's result text. */
function callingChat(name: string, args: JsonObject): ChatModel {
  return {
    provider: "openai",
    model: "test-model",
    stream(request: Request) {
      const last = request.messages.at(-1)!.parts.at(-1);
      const reply =
        last instanceof ToolResultPart
          ? textReply(`got ${last.content}`)
          : new Response({
              message: new Message({ role: "assistant", parts: [new ToolCallPart({ id: "c1", name, arguments: args })] }),
              finishReason: "tool_calls",
            });
      return new EventStream(
        (async function* (): AsyncGenerator<Event> {
          yield* new ScriptedModel([reply], { provider: "openai", model: "test-model" }).stream(request);
        })(),
      );
    },
  };
}

/** Run the agent flow granting `granted`, its model calling it with `args`, with `tools`. */
async function runAgent(granted: string, args: JsonObject, tools: Record<string, ToolEntry>) {
  const models: ModelAccess = { env: NO_ENV, createModel: () => callingChat(granted, args) };
  return await runFlow(await loadFlow(agentFlow(granted)), {}, { env: hostEnv(dir), log: freshLog(), tools, models });
}

test("an ask_user answered inline works as before", async () => {
  const result = await runAgent("ask_user", { question: "Ship?" }, { ask_user: () => "yes" });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("got yes");
});

const refused = (name: string) =>
  `agent node 'agent': tool '${name}' returned pending(); an AGENT's tool call can't wait for a later answer ` +
  "in this version, so answer it inline or ask from a TOOL node";

test("an ask_user handler that returns pending() fails the agent", async () => {
  const result = await runAgent("ask_user", { question: "Ship?" }, { ask_user: () => pending() });

  expect(result.status).toBe("failed");
  expect(result.failure?.nodeId).toBe("agent");
  expect(result.failure?.message).toBe(refused("ask_user"));
});

test("a host tool that returns pending() fails the agent", async () => {
  const codexSend = tool({
    name: "codex_send",
    description: "Send text.",
    parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: () => pending(),
  });

  const result = await runAgent("codex_send", { text: "hi" }, { codex_send: codexSend });

  expect(result.status).toBe("failed");
  expect(result.failure?.nodeId).toBe("agent");
  expect(result.failure?.message).toBe(refused("codex_send"));
});
