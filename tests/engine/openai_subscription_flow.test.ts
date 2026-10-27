/**
 * A port of `tests/engine/test_openai_subscription_flow.py`: direct subscription responses use
 * yian's ordinary tools.
 *
 * Ported: `test_tool_loop_and_restored_memory_use_direct_model` (one run: TS yian has no session
 * memory to restore) and `test_partial_stream_fails_before_composer_executes_tool`, whole. Not
 * ported: `test_rich_responses_summary_keeps_shared_memory_working` (session memory and its history
 * summary). The model calls go through the real
 * `chatModelFromConfig` with a recording `send` (`recording`) that answers as Python's
 * `ResponsesServer` does (a reasoning item, then one tool call or the text `6`, as Responses
 * streams); the subscription is a saved login in the row's own credential store, where Python
 * builds `create_subscription_model` with a synthetic access source and patches
 * `model_from_config` to return it. Each run's environment is the row's (`hostEnv`). A stream counts as closed once it is
 * read to its end or cancelled (Python's `EventStream.close`).
 *
 * Changed: `test_partial_stream_fails_before_composer_executes_tool`: a stream cut short is
 * retried (decisions 19, 26), so with the default `YIAN_MODEL_RETRIES` (6) the `None` row's 7
 * attempts all break mid-stream, each published as a `ModelCallStarted` and a `ModelCallFinished`
 * naming its error; the `incomplete` row's `response.incomplete` is a whole reply saying it did not
 * finish, not a transport failure, so its one attempt fails the call. Both fail the run with
 * `provider.unavailable` before the tool runs; the backoff waits run on a virtual scheduler with
 * fixed jitter. The tool is a `FunctionTool` passed to the run (Python's `TOOL_REGISTRY` entry).
 * Python's patched `subprocess.Popen` has no counterpart: no TS OpenAI route starts a process.
 */

import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted, type RunEvent } from "../../src/events.ts";
import { AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { PLAIN_FLOW, recording, type Replier, sse } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-openai-subscription-flow-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const TOOL_FLOW = PLAIN_FLOW.replace("test-model", "gpt-5.5").replace(
  "    kind: agent\n",
  "    kind: agent\n    tools: [double]\n",
);

let files = 0;

/** A fresh environment with its own credential store and `vars` set. */
function fresh(vars: Record<string, string> = {}): Env {
  const root = join(dir, String(files++));
  // The credential store refuses a directory others can read.
  mkdirSync(root, { mode: 0o700 });
  return hostEnv(root, vars);
}

/** Save a subscription login for `openai` in `env`'s store. */
const connect = (env: Env) =>
  new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({ accessToken: "synthetic", refreshToken: "refresh", expiresAt: 4102444800, accountId: "account" }),
    "gpt-5.5",
  );

/** Python's `double` tool, recording each value it doubles. */
function doubler(): { tool: FunctionTool; calls: unknown[] } {
  const calls: unknown[] = [];
  const tool = new FunctionTool({
    name: "double",
    description: "Double an integer.",
    parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
    execute: (args) => {
      calls.push(args["value"]);
      return (args["value"] as number) * 2;
    },
  });
  return { tool, calls };
}

const REASONING = { id: "rs_test", type: "reasoning", summary: [], encrypted_content: "synthetic-continuation" };
const TOOL_CALL = {
  id: "fc_test",
  type: "function_call",
  name: "double",
  arguments: '{"value": 3}',
  call_id: "call_test",
  status: "completed",
};
const ANSWER = {
  id: "msg_test",
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: "6", annotations: [] }],
};

/**
 * Python's `ResponsesServer`: each reply streams a reasoning item, then a call of `double` (the
 * first request, with `toolFirst`) or the text `6`, then a `response.<terminal>` event (none when
 * `terminal` is `null`). `closed[i]` turns true once the `i`th stream is read to its end or cancelled.
 */
function responsesServer(
  options: { terminal?: string | null; toolFirst?: boolean } = {},
): { reply: Replier; closed: boolean[] } {
  const { terminal = "completed", toolFirst = true } = options;
  const closed: boolean[] = [];
  const reply: Replier = (_sent, number) => {
    const isTool = toolFirst && number === 1;
    const response = {
      id: `resp_${number}`,
      object: "response",
      created_at: 0,
      status: terminal ?? "in_progress",
      model: "gpt-5.5",
      output: [REASONING, isTool ? TOOL_CALL : ANSWER],
      error: null,
      incomplete_details: null,
      usage: null,
    };
    const events: [string, Record<string, unknown>][] = [
      ["response.created", { response: { ...response, output: [] } }],
      ["response.output_item.added", { output_index: 0, item: { ...REASONING, encrypted_content: null } }],
      ["response.output_item.done", { output_index: 0, item: REASONING }],
    ];
    if (isTool) {
      events.push(
        ["response.output_item.added", { output_index: 1, item: { ...TOOL_CALL, arguments: "", status: "in_progress" } }],
        ["response.function_call_arguments.delta", { output_index: 1, item_id: "fc_test", delta: '{"value":' }],
        ["response.function_call_arguments.delta", { output_index: 1, item_id: "fc_test", delta: " 3}" }],
      );
    } else {
      events.push(
        ["response.output_text.delta", { output_index: 1, content_index: 0, item_id: "msg_test", delta: "6" }],
        ["response.output_text.done", { output_index: 1, content_index: 0, item_id: "msg_test", text: "6" }],
      );
    }
    if (terminal !== null) events.push([`response.${terminal}`, { response }]);
    const chunks = sse(events);
    const index = closed.push(false) - 1;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        const chunk = chunks.shift();
        if (chunk === undefined) {
          closed[index] = true;
          controller.close();
        } else controller.enqueue(new TextEncoder().encode(chunk));
      },
      cancel() {
        closed[index] = true;
      },
    });
    return new globalThis.Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  return { reply, closed };
}

/** Where a subscription's Responses requests go. */
const SUBSCRIPTION_URL = "https://chatgpt.com/backend-api/codex/responses";

/** The `input` items of a request body. */
const inputOf = (body: Record<string, unknown>) => body["input"] as Record<string, unknown>[];

test("a tool loop uses the direct model", async () => {
  const env = fresh();
  await connect(env);
  const { tool, calls } = doubler();
  const server = responsesServer();
  const models = recording(env, server.reply);
  const flow = await loadFlow(TOOL_FLOW, { env });
  const result = await runFlow(flow, { question: "Double 3." }, { tools: { double: tool }, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("6");

  expect(calls).toEqual([3]);
  expect(models.requests.map((request) => request.url)).toEqual(Array(2).fill(SUBSCRIPTION_URL));
  const { body } = models.requests[1]!;
  expect(inputOf(body).some((item) => item["encrypted_content"] === "synthetic-continuation")).toBe(true);
  expect(inputOf(body).some((item) => item["type"] === "function_call_output" && item["output"] === "6")).toBe(true);
});

describe("partial stream fails before composer executes tool", () => {
  // A stream cut short is retried; a `response.incomplete` is a whole reply that says it did not
  // finish, so its one attempt fails the call.
  for (const [terminal, attempts] of [
    [null, 7],
    ["incomplete", 1],
  ] as const) {
    test(String(terminal), async () => {
      const env = fresh();
      await connect(env);
      const { tool, calls } = doubler();
      const server = responsesServer({ terminal });
      const models = recording(env, server.reply);
      const events: RunEvent[] = [];
      const result = await runFlow(await loadFlow(TOOL_FLOW, { env }), { question: "Double 3." }, {
        tools: { double: tool },
        models,
        env,
        onEvent: (event) => void events.push(event),
        scheduler: virtualScheduler(),
        random: () => 0,
      });
      expect(result.status).toBe("failed");
      expect(result.failure!.message).toContain("complete");
      expect(result.failure!.code).toBe("provider.unavailable");
      expect(models.requests).toHaveLength(attempts);
      const started = events.filter((event) => event instanceof ModelCallStarted);
      const finished = events.filter((event) => event instanceof ModelCallFinished);
      // Each attempt is its own model_call step.
      expect(started.map((event) => event.n)).toEqual(Array.from({ length: attempts }, (_, index) => index + 1));
      expect(finished.map((event) => [event.n, event.error !== null])).toEqual(started.map((event) => [event.n, true]));
      expect(calls).toEqual([]);
      expect(server.closed[0]).toBe(true);
    });
  }
});
