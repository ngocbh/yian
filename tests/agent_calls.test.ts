/**
 * A host reads a run log's AGENT model calls (`agentCalls`) and one call's view (`agentCall`): the
 * request rebuilt from the recorded flow and the steps before it, checked against the recorded
 * digest, then the attempts, the reply and its tool results as recorded.
 *
 * Each run is logged with a scripted model, which keeps every request it received; a rebuilt
 * request must equal the one the model received.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../src/compose/loader.ts";
import { ProviderError } from "../src/llm/errors.ts";
import { type ChatModel, EventStream, Finish, TextDelta } from "../src/llm/events.ts";
import { NoAuth } from "../src/llm/auth.ts";
import { RouteModel } from "../src/llm/executor.ts";
import { Message as ModelMessage, NamedTool, type Request, Response, ToolCallPart, Usage } from "../src/llm/messages.ts";
import { AnthropicMessages } from "../src/llm/protocols/anthropic_messages.ts";
import { Route } from "../src/llm/route.ts";
import { BindingError } from "../src/nodes/binding.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { AuthError } from "../src/llm_clients/auth/index.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../src/memory/base.ts";
import { FileMemory } from "../src/memory/file.ts";
import type { ModelAccess } from "../src/llm_clients/factory.ts";
import { fromModelMessage } from "../src/nodes/agent/messages.ts";
import { AgentNode } from "../src/nodes/agent/node.ts";
import { virtualScheduler } from "../src/platform.ts";
import {
  advance,
  type AgentCallView,
  agentCall,
  agentCalls,
  complete,
  interrupt,
  pendingCalls,
  type Message,
  type RunFlowOptions,
  runFlow,
  send,
} from "../src/run.ts";
import { pending } from "../src/tools/context.ts";
import { Skill } from "../src/skills/catalog.ts";
import { LOAD_SKILL } from "../src/skills/tool.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import { logJson } from "../src/typesys/json.ts";
import { RecordingSession } from "./helpers/memory_fakes.ts";
import { callReply, crashAfter, NO_ENV, serving, textReply } from "./helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-calls-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

const START = Date.UTC(2026, 9, 22, 10, 0, 0, 0);

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(root, "run-")), "log.jsonl");
}

/** A tool `name` answering `<name>:<topic>`; `onCall` runs first. */
function topicTool(name: string, onCall: () => void = () => {}): FunctionTool {
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      onCall();
      return `${name}:${String(args["topic"])}`;
    },
  });
}

/** The tools `lookup` and `save`. */
const toolsOf = () => ({ lookup: topicTool("lookup"), save: topicTool("save") });

/** Run `text` over `model` with a log; the other options go to `runFlow`. */
async function run(
  text: string,
  model: ChatModel,
  inputs: Record<string, unknown> = {},
  options: RunFlowOptions & { log?: string } = {},
) {
  const log = options.log ?? runLog();
  const result = await runFlow(await loadFlow(text), inputs, { models: serving(model), env: NO_ENV, ...options, log });
  return { log, result };
}

/** What the view's request holds of `sent`, the request the model received. */
function sentView(sent: Request) {
  const [system, ...messages] = sent.messages;
  const choice = sent.toolChoice;
  const format = sent.outputFormat;
  return {
    system: system!.text(),
    messages: messages.map(fromModelMessage),
    tools: sent.tools.map(({ name, description, parameters, strict }) => ({ name, description, parameters, strict })),
    toolChoice: choice instanceof NamedTool ? { name: choice.name } : choice,
    parallelToolCalls: sent.parallelToolCalls,
    outputFormat: format === null ? null : { name: format.name, schema: format.schema, strict: format.strict },
  };
}

/** `view`'s request, which must have been rebuilt. */
function requestOf(view: AgentCallView) {
  expect(view.request).not.toBeNull();
  return view.request!;
}

const TOOLS = `
id: tools
name: tools
llm_config: {provider: anthropic, model: claude-test}
input:
  topic: !str
nodes:
  writer:
    kind: agent
    input:
      topic: \${input.topic}
    tools: [lookup, save]
    system_prompt: Be exact.
    prompt: Write about \${topic}.
    output: str
output: \${writer.output}
`;

test("a tool loop: each call's request is the one sent, its reply and its two results as recorded", async () => {
  const model = new ScriptedModel(
    [callReply(["lookup", { topic: "a" }, "call-a"], ["save", { topic: "b" }, "call-b"]), textReply("done")],
    { provider: "anthropic", model: "claude-test" },
  );
  const tools = { lookup: topicTool("lookup"), save: topicTool("save") };
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, { tools });
  expect(result.status, result.failure?.message).toBe("succeeded");

  const listed = (n: number, finish: string) => ({
    nodeId: "writer",
    n,
    purpose: "chat",
    model: "claude-test",
    startedAt: expect.any(String),
    attempts: 1,
    finish,
  });
  expect(agentCalls(log)).toEqual([listed(1, "tool_calls"), listed(4, "stop")]);

  const first = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(first.digestMismatch).toBe(false);
  expect(requestOf(first)).toMatchObject(sentView(model.requests[0]!));
  expect(requestOf(first).system).toBe("Be exact.");
  expect(requestOf(first).messages).toEqual([{ role: "user", parts: [{ type: "text", text: "Write about owls." }] }]);
  expect(requestOf(first).tools.map((tool) => tool.name)).toEqual(["lookup", "save"]);
  expect(first.reply).toEqual(fromModelMessage(model.requests[1]!.messages[2]!));
  const ran = (callId: string, name: string, topic: string) => ({
    callId,
    name,
    arguments: { topic },
    output: `${name}:${topic}`,
    isError: false,
    closed: null,
    startedAt: expect.any(String),
  });
  expect(first.toolResults).toEqual([ran("call-a", "lookup", "a"), ran("call-b", "save", "b")]);
  expect(first).toMatchObject({
    nodeId: "writer",
    n: 1,
    purpose: "chat",
    finish: "tool_calls",
    interrupted: false,
  });
  const time = expect.any(String);
  expect(first.attempts).toEqual([
    { n: 1, startedAt: time, firstTokenAt: time, finishedAt: time, error: null, retryIn: null, usage: null },
  ]);

  const second = await agentCall(log, { nodeId: "writer", n: 4 }, { env: NO_ENV });
  expect(second.digestMismatch).toBe(false);
  expect(requestOf(second)).toMatchObject(sentView(model.requests[1]!));
  expect(second.reply).toEqual({ role: "assistant", parts: [{ type: "text", text: "done" }] });
  expect(second.toolResults).toEqual([]);
});

test("an attempt's usage holds every count its step recorded, cache writes included", async () => {
  const usage = new Usage({ inputTokens: 9, outputTokens: 2, totalTokens: 11, cacheReadTokens: 4, cacheCreationTokens: 3 });
  const reply = new Response({ message: textReply("done").message, usage, finishReason: "stop" });
  const model = new ScriptedModel([reply], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, { tools: toolsOf() });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(view.attempts[0]!.usage).toEqual({
    inputTokens: 9,
    outputTokens: 2,
    cacheReadTokens: 4,
    cacheCreationTokens: 3,
    reasoningTokens: null,
  });
});

/** A chat flow: a list prompt splicing `history`, a steer inbox, the tool `lookup`, output `output`. */
const chat = (output = "str") => `
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
input:
  history: !list(Message)
  message: !str
  notes: !Inbox(str)
nodes:
  reply:
    kind: agent
    input:
      history: \${input.history}
      message: \${input.message}
    steer_inbox: \${input.notes}
    tools: [lookup]
    system_prompt: Help.
    prompt:
      - \${history}
      - user: \${message}
    output: ${output}
output: \${reply.output}
`;

const HISTORY: Message[] = [
  { role: "user", parts: [{ type: "text", text: "Hi" }] },
  { role: "assistant", parts: [{ type: "text", text: "Hello." }] },
];

test("a list-prompt agent: the history is spliced and a steer read joins the next request", async () => {
  const log = runLog();
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const tools = { lookup: topicTool("lookup", () => void send(log, "notes", "only 2024")) };
  const { result } = await run(chat(), model, { history: HISTORY, message: "again" }, { log, tools });
  expect(result.status, result.failure?.message).toBe("succeeded");

  const calls = agentCalls(log);
  expect(calls.map(({ nodeId, n }) => [nodeId, n])).toEqual([
    ["reply", 1],
    ["reply", 4],
  ]);
  const first = await agentCall(log, { nodeId: "reply", n: 1 }, { env: NO_ENV });
  expect(requestOf(first).messages).toEqual([...HISTORY, { role: "user", parts: [{ type: "text", text: "again" }] }]);
  const second = await agentCall(log, { nodeId: "reply", n: 4 }, { env: NO_ENV });
  expect(second.digestMismatch).toBe(false);
  expect(requestOf(second)).toMatchObject(sentView(model.requests[1]!));
  expect(requestOf(second).messages.at(-1)).toEqual({ role: "user", parts: [{ type: "text", text: "only 2024" }] });
});

/** An environment with no host setting but `YIAN_MODEL_RETRIES=1`. */
const retryEnv = (name: string) => (name === "YIAN_MODEL_RETRIES" ? "1" : NO_ENV(name));

test("a retried call: its attempts are one call, which the n of either attempt names", async () => {
  const model = new ScriptedModel([new ProviderError("rate_limit", "slow down", { status: 429 }), textReply("ok")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, {
    env: retryEnv,
    scheduler: virtualScheduler(START),
    random: () => 1,
    tools: { lookup: topicTool("lookup"), save: topicTool("save") },
  });
  expect(result.status, result.failure?.message).toBe("succeeded");

  expect(agentCalls(log)).toEqual([
    {
      nodeId: "writer",
      n: 2,
      purpose: "chat",
      model: "claude-test",
      startedAt: new Date(START).toISOString(),
      attempts: 2,
      finish: "stop",
    },
  ]);
  const byFirst = await agentCall(log, { nodeId: "writer", n: 1 }, { env: retryEnv });
  const byLast = await agentCall(log, { nodeId: "writer", n: 2 }, { env: retryEnv });
  expect(byFirst).toEqual(byLast);
  expect(byLast.n).toBe(2);
  expect(byLast.digestMismatch).toBe(false);
  expect(requestOf(byLast)).toMatchObject(sentView(model.requests[1]!));
  expect(byLast.attempts.map(({ n, error, retryIn }) => ({ n, error, retryIn }))).toEqual([
    { n: 1, error: { kind: "rate_limit", message: "slow down" }, retryIn: 0.375 },
    { n: 2, error: null, retryIn: null },
  ]);
  expect(byLast.finish).toBe("stop");
});

/** `model` with `hook(k)` run after the `k`-th event of each stream (from 1), and `hook(Infinity)` at its end. */
function hooked(model: ChatModel, hook: (k: number) => void): ChatModel {
  return {
    provider: model.provider,
    model: model.model,
    stream(request: Request, options: { signal?: AbortSignal } = {}) {
      const inner = model.stream(request, options);
      async function* events() {
        let k = 0;
        for await (const event of inner) {
          yield event;
          hook(++k);
        }
        hook(Infinity);
      }
      return new EventStream(events());
    },
  };
}

test("an interrupted call: a reply the interrupt cut, and a reply whose calls it skipped", async () => {
  const cutLog = runLog();
  const cut = hooked(
    new ScriptedModel([[new TextDelta(0, "hel"), new TextDelta(0, "lo"), new Finish({ reason: "stop" })]], {
      provider: "anthropic",
      model: "claude-test",
    }),
    (k) => {
      if (k === 1) void interrupt(cutLog, "notes");
    },
  );
  const inputs = { history: [], message: "go" };
  const tools = { lookup: topicTool("lookup") };
  const { result } = await run(chat("Optional(str)"), cut, inputs, { log: cutLog, tools });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await agentCall(cutLog, { nodeId: "reply", n: 1 }, { env: NO_ENV });
  expect(view).toMatchObject({ finish: "interrupted", interrupted: true, digestMismatch: false });
  expect(view.reply).toEqual({ role: "assistant", parts: [{ type: "text", text: "hel" }] });

  const skipLog = runLog();
  const skip = hooked(
    new ScriptedModel([callReply(["lookup", { topic: "a" }, "c1"])], { provider: "anthropic", model: "claude-test" }),
    (k) => {
      if (k === Infinity) void interrupt(skipLog, "notes");
    },
  );
  await run(chat("Optional(str)"), skip, inputs, { log: skipLog, tools });
  const skipped = await agentCall(skipLog, { nodeId: "reply", n: 1 }, { env: NO_ENV });
  expect(skipped).toMatchObject({ finish: "tool_calls", interrupted: true });
  expect(skipped.toolResults).toEqual([
    {
      callId: "c1",
      name: "lookup",
      arguments: { topic: "a" },
      output: "Not run: interrupted by the user.",
      isError: true,
      closed: "skipped",
      startedAt: null,
    },
  ]);
});

test("a run whose flow files changed on disk is read with the recorded flow", async () => {
  const dir = mkdtempSync(join(root, "files-"));
  mkdirSync(join(dir, "lib"));
  const child = (system: string) =>
    `id: child\nname: child\ninput:\n  q: !str\nnodes:\n  writer:\n    kind: agent\n` +
    `    input:\n      q: \${input.q}\n` +
    `    system_prompt: ${system}\n    prompt: Answer \${q}.\n    output: str\noutput: \${writer.output}\n`;
  writeFileSync(join(dir, "lib", "child.yaml"), child("Old system."));
  const rootPath = join(dir, "flow.yaml");
  writeFileSync(
    rootPath,
    "id: root\nname: root\nllm_config: {provider: anthropic, model: claude-test}\ninput:\n  q: !str\n" +
      "uses:\n  child: lib/child\nnodes:\n  c:\n    kind: call\n    call: child\n    input:\n      q: ${input.q}\n" +
      "output: ${c.output}\n",
  );
  const flow = await loadFlow(readFileSync(rootPath, "utf8"), { searchPaths: [dir], sourceLabel: basename(rootPath) });
  const model = new ScriptedModel([textReply("42")], { provider: "anthropic", model: "claude-test" });
  const log = runLog();
  const result = await runFlow(flow, { q: "x" }, { log, models: serving(model), env: NO_ENV });
  expect(result.status, result.failure?.message).toBe("succeeded");
  writeFileSync(join(dir, "lib", "child.yaml"), child("New system."));
  writeFileSync(rootPath, "not a flow");

  const [entry, ...rest] = agentCalls(log);
  expect(rest).toEqual([]);
  expect(entry!.nodeId).toMatch(/\/writer$/);
  const view = await agentCall(log, entry!, { env: NO_ENV });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view).system).toBe("Old system.");
  expect(requestOf(view)).toMatchObject(sentView(model.requests[0]!));
});

test("a request that no longer matches its digest is returned flagged, not thrown", async () => {
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log } = await run(TOOLS, model, { topic: "owls" }, { tools: toolsOf() });
  const text = readFileSync(log, "utf8");
  const digest = /"request_sha256":"([0-9a-f]{64})"/.exec(text)![1]!;
  writeFileSync(log, text.replace(digest, "0".repeat(64)));
  const view = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(view.digestMismatch).toBe(true);
  expect(requestOf(view)).toMatchObject(sentView(model.requests[0]!));
});

/** Fail when `view` holds the host's key or header value, or a credential field. */
function expectNoSecrets(view: AgentCallView): void {
  const json = logJson(JSON.parse(JSON.stringify(view)));
  for (const secret of [KEY, HEADER, "api_key", "headers", "_host_key", "_header_names", "_connection"]) {
    expect(json).not.toContain(secret);
  }
}

test("the view holds the model config without the host's key, headers or connection", async () => {
  const key = KEY;
  const header = HEADER;
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const llmConfig = { provider: "anthropic", model: "claude-test", api_key: key, headers: { "x-trace": header } };
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, {
    llmConfig,
    tools: { lookup: topicTool("lookup"), save: topicTool("save") },
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(requestOf(view).model).toBe("claude-test");
  expect(requestOf(view).params).toEqual({ provider: "anthropic", model: "claude-test" });
  expectNoSecrets(view);
  const credentials = { apiKey: key, headers: { "x-trace": header } };
  const keyed = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV, ...credentials });
  expect(keyed).toEqual(view);
});

test("a waiting run's call shows a pending tool call without a result, then the host's answer", async () => {
  const model = new ScriptedModel(
    [callReply(["lookup", { topic: "a" }, "call-a"], ["save", { topic: "b" }, "call-b"]), textReply("done")],
    { provider: "anthropic", model: "claude-test" },
  );
  const save = new FunctionTool({
    name: "save",
    description: "Tool save.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: () => pending(),
  });
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, { tools: { lookup: topicTool("lookup"), save } });
  expect(result.status, result.failure?.message).toBe("waiting");
  const waiting = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(waiting.digestMismatch).toBe(false);
  expect(waiting.toolResults.map(({ callId, output }) => [callId, output])).toEqual([
    ["call-a", "lookup:a"],
    ["call-b", null],
  ]);
  await complete(log, pendingCalls(log)[0]!.callId, { output: "saved" }, { env: NO_ENV });
  const answered = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(answered.toolResults.map(({ callId, output }) => [callId, output])).toEqual([
    ["call-a", "lookup:a"],
    ["call-b", "saved"],
  ]);
});

const CALL_TIME = `
id: call_time
name: call_time
llm_config: {provider: anthropic, model: claude-test}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
  chat:
    kind: agent
    prompt: Go.
    output: str
    llm_config: {model: '\${pick.output}'}
output: \${chat.output}
`;

const KEY = "sk-sentinel-agent-call-7f3a";
const HEADER = "hdr-sentinel-agent-call-91c2";

test("a call-time agent's params come from its llm_config step, with or without a host key", async () => {
  for (const keyed of [false, true]) {
    const model = new ScriptedModel([textReply("claude-other"), textReply("done")], {
      provider: "anthropic",
      model: "claude-test",
    });
    const llmConfig = keyed ? { provider: "anthropic", api_key: KEY, headers: { "x-trace": HEADER } } : undefined;
    const { log, result } = await run(CALL_TIME, model, {}, llmConfig === undefined ? {} : { llmConfig });
    expect(result.status, result.failure?.message).toBe("succeeded");
    const credentials = keyed ? { apiKey: KEY, headers: { "x-trace": HEADER } } : {};
    const view = await agentCall(log, { nodeId: "chat", n: 2 }, { env: NO_ENV, ...credentials });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view).params).toEqual({ provider: "anthropic", model: "claude-other" });
    expect(requestOf(view)).toMatchObject(sentView(model.requests[1]!));
    expectNoSecrets(view);
    if (keyed) {
      // Its config can't be bound again without the key the run used.
      await expect(agentCall(log, { nodeId: "chat", n: 2 }, { env: NO_ENV })).rejects.toThrow(AuthError);
    }
  }
});

const STRUCTURED = `
id: structured
name: structured
llm_config: {provider: anthropic, model: claude-test}
nodes:
  count:
    kind: agent
    prompt: How many?
    output: int
output: \${count.output}
`;

test("a structured call's request is rebuilt with the model the host builds", async () => {
  const model = new ScriptedModel([textReply('{"value": 3}')], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(STRUCTURED, model);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [entry] = agentCalls(log);
  expect(entry!.purpose).not.toBe("chat");
  const view = await agentCall(log, entry!, { env: NO_ENV, models: serving(model) });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view)).toMatchObject(sentView(model.requests[0]!));
});

/** A `RouteModel` on Anthropic Messages (`claude-test`, so the forced tool) that replays `replies`. */
class ScriptedRoute extends RouteModel {
  readonly script: ScriptedModel;

  constructor(replies: Response[]) {
    super({
      route: new Route({ provider: "anthropic", baseUrl: "https://api.example", path: "x", auth: new NoAuth() }),
      protocol: new AnthropicMessages({ model: "claude-test", provider: "anthropic" }),
    });
    this.script = new ScriptedModel(replies, { provider: "anthropic", model: "claude-test" });
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    return this.script.stream(request, options);
  }
}

test("a structured call's request is rebuilt by the default model access without any credential", async () => {
  const call = new ToolCallPart({ id: "toolu_1", name: "ScalarWrapper", arguments: { value: 3n } });
  const reply = new Response({ message: new ModelMessage({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
  const model = new ScriptedRoute([reply]);
  const { log, result } = await run(STRUCTURED, model);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [entry] = agentCalls(log);
  // No model access is passed and no key is set, so nothing could build the model.
  const view = await agentCall(log, entry!, { env: NO_ENV });
  expect(view.digestMismatch).toBe(false);
  const sent = model.script.requests[0]!;
  expect(sent.toolChoice).toEqual(new NamedTool("ScalarWrapper"));
  expect(requestOf(view)).toMatchObject(sentView(sent));
});

test("a structured call in a host-key run is rebuilt with the key, and is null without it", async () => {
  const model = new ScriptedModel([textReply('{"value": 3}')], { provider: "anthropic", model: "claude-test" });
  // As a real model build does, it refuses a config marked for the host key that lacks the key.
  const keyed: ModelAccess = {
    env: NO_ENV,
    createModel: (config) => {
      if (config["_host_key"] === true && config["api_key"] !== KEY) throw new Error("the host key is missing");
      return model;
    },
  };
  const llmConfig = { provider: "anthropic", api_key: KEY };
  const { log, result } = await run(STRUCTURED, model, {}, { llmConfig, models: keyed });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [entry] = agentCalls(log);
  const view = await agentCall(log, entry!, { env: NO_ENV, models: keyed, apiKey: KEY });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view)).toMatchObject(sentView(model.requests[0]!));
  expectNoSecrets(view);
  const keyless = await agentCall(log, entry!, { env: NO_ENV, models: keyed });
  expect(keyless.request).toBeNull();
  expect(keyless.digestMismatch).toBe(false);
});

const CHAT_THEN_STRUCTURED = `
id: chat_then_structured
name: chat_then_structured
llm_config: {provider: anthropic, model: claude-test}
nodes:
  count:
    kind: agent
    tools: [lookup]
    prompt: How many?
    output: int
output: \${count.output}
`;

test("a chat loop then its final structured call: each request is the one sent", async () => {
  const model = new ScriptedModel(
    [callReply(["lookup", { topic: "a" }, "call-a"]), textReply("Three."), textReply('{"value": 3}')],
    { provider: "anthropic", model: "claude-test" },
  );
  const { log, result } = await run(CHAT_THEN_STRUCTURED, model, {}, { tools: { lookup: topicTool("lookup") } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const calls = agentCalls(log);
  expect(calls.map(({ n, purpose }) => [n, purpose === "chat"])).toEqual([
    [1, true],
    [3, true],
    [4, false],
  ]);
  for (const [index, entry] of calls.entries()) {
    const view = await agentCall(log, entry, { env: NO_ENV, models: serving(model) });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view)).toMatchObject(sentView(model.requests[index]!));
  }
});

test("a skill grant: the request lists the recorded skills and carries load_skill", async () => {
  const text = `
id: skilled
name: skilled
llm_config: {provider: anthropic, model: claude-test}
nodes:
  review:
    kind: agent
    skills: [style]
    prompt: review it
    output: str
output: \${review.output}
`;
  const skill = new Skill({ name: "style", description: "House style.", instructions: "Prefer small diffs." });
  const model = new ScriptedModel([callReply([LOAD_SKILL, { name: "style" }, "call-1"]), textReply("reviewed")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log, result } = await run(text, model, {}, { skills: new Map([["style", skill]]) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const first = await agentCall(log, { nodeId: "review", n: 1 }, { env: NO_ENV });
  expect(requestOf(first).tools.map((tool) => tool.name)).toEqual([LOAD_SKILL]);
  expect(requestOf(first).system).toContain("House style.");
  expect(first.toolResults.map(({ name, output }) => [name, output])).toEqual([[LOAD_SKILL, "Prefer small diffs."]]);
  const second = await agentCall(log, { nodeId: "review", n: 3 }, { env: NO_ENV });
  expect(second.digestMismatch).toBe(false);
  expect(requestOf(second)).toMatchObject(sentView(model.requests[1]!));
});

test("a grant naming a tool group carries the group's recorded tools", async () => {
  const text = TOOLS.replace("tools: [lookup, save]", "tools: [toolbox]");
  const model = new ScriptedModel([callReply(["toolbox_save", { topic: "b" }, "call-b"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const tools = { toolbox: [topicTool("lookup"), topicTool("save")] };
  const { log, result } = await run(text, model, { topic: "owls" }, { tools });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await agentCall(log, { nodeId: "writer", n: 3 }, { env: NO_ENV });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view).tools.map((tool) => tool.name)).toEqual(["toolbox_lookup", "toolbox_save"]);
  expect(requestOf(view)).toMatchObject(sentView(model.requests[1]!));
});

test.each<[string, string, Record<string, FunctionTool[]>, string, string[]]>([
  ["some tools of a group", "tools: [{toolbox: [save]}]", { toolbox: [topicTool("lookup"), topicTool("save")] }, "toolbox_save", ["toolbox_save"]],
  [
    "a tool shared by two granted groups",
    "tools: [company, toolbox]",
    (() => {
      const save = topicTool("save");
      return { toolbox: [topicTool("lookup"), save], company: [save, topicTool("file")] };
    })(),
    "company_save",
    ["company_save", "company_file", "toolbox_lookup"],
  ],
])("a grant of %s rebuilds exactly the sent request", async (_name, line, tools, call, names) => {
  const text = TOOLS.replace("tools: [lookup, save]", line);
  const model = new ScriptedModel([callReply([call, { topic: "b" }, "call-b"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log, result } = await run(text, model, { topic: "owls" }, { tools });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(names);
  for (const n of [1, 3]) {
    const view = await agentCall(log, { nodeId: "writer", n }, { env: NO_ENV });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view)).toMatchObject(sentView(model.requests[(n - 1) / 2]!));
  }
});

test("a run cut off mid-agent, then resumed: each start's calls are read", async () => {
  const replies = () => [callReply(["lookup", { topic: "a" }, "call-a"]), textReply("done")];
  const options = { tools: { lookup: topicTool("lookup"), save: topicTool("save") } };
  const scripted = () => new ScriptedModel(replies(), { provider: "anthropic", model: "claude-test" });
  const uncut = await run(TOOLS, scripted(), { topic: "owls" }, options);
  // Cut right after the tool result, before the second model call.
  const k = readFileSync(uncut.log, "utf8").split("\n").findIndex((line) => line.includes('"tool_result"')) + 1;
  const log = runLog();
  await crashAfter(k, () => run(TOOLS, scripted(), { topic: "owls" }, { ...options, log }));
  expect(agentCalls(log).map(({ n }) => n)).toEqual([1]);
  const cut = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(cut.digestMismatch).toBe(false);
  expect(cut.toolResults.map(({ output }) => output)).toEqual(["lookup:a"]);

  const second = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const resumed = await advance({ log, models: serving(second), env: NO_ENV, ...options });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(agentCalls(log).map(({ n }) => n)).toEqual([1, 3]);
  const after = await agentCall(log, { nodeId: "writer", n: 3 }, { env: NO_ENV });
  expect(after.digestMismatch).toBe(false);
  expect(requestOf(after)).toMatchObject(sentView(second.requests[0]!));
  expect(await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV })).toEqual(cut);
});

test("agentCall refuses a step that is no model call", async () => {
  const model = new ScriptedModel([callReply(["lookup", { topic: "a" }, "call-a"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log } = await run(TOOLS, model, { topic: "owls" }, { tools: toolsOf() });
  for (const at of [{ nodeId: "writer", n: 2 }, { nodeId: "writer", n: 9 }, { nodeId: "other", n: 1 }]) {
    await expect(agentCall(log, at, { env: NO_ENV })).rejects.toThrow(
      new RangeError(`run log ${log} has no model call ${at.n} of node '${at.nodeId}'`),
    );
  }
});

/** A flow whose agent binds `!<tag> ${input.data}`. */
const bound = (tag: string) => `
id: bound
name: bound
llm_config: {provider: anthropic, model: claude-test}
input:
  data: !Any
nodes:
  writer:
    kind: agent
    input:
      topic: !${tag} \${input.data}
    prompt: Write about \${topic}.
    output: str
output: \${writer.output}
`;

test("agentCall refuses a node that failed at binding, and one whose binding no longer finishes", async () => {
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log: failedLog, result: failed } = await run(bound("int"), model, { data: "owls" });
  expect(failed.failure?.code).toBe("binding.type");
  await expect(agentCall(failedLog, { nodeId: "writer", n: 1 }, { env: NO_ENV })).rejects.toThrow(
    new RangeError(`run log ${failedLog} has no model call 1 of node 'writer'`),
  );

  // The run bound a str; an edited log gives an int, which the binding now refuses.
  const { log, result } = await run(bound("str"), model, { data: "owls-sentinel" });
  expect(result.status, result.failure?.message).toBe("succeeded");
  writeFileSync(log, readFileSync(log, "utf8").replaceAll('"owls-sentinel"', "5"));
  const error = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV }).catch((e) => e);
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(`run log ${log}: node 'writer''s binding no longer finishes, so no request is rebuilt`);
  expect(error.cause).toBeInstanceOf(BindingError);
});

test("agentCall refuses bad credentials with advance's error", async () => {
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const llmConfig = { provider: "anthropic", model: "claude-test", api_key: KEY, headers: { "x-trace": HEADER } };
  const { log, result } = await run(TOOLS, model, { topic: "owls" }, { llmConfig, tools: toolsOf() });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const refusedHeaders: Record<string, string>[] = [{ "x-other": HEADER }, { "x-trace": "bad\nvalue" }];
  for (const headers of refusedHeaders) {
    const credentials = { apiKey: KEY, headers };
    const refused = await advance({ log, env: NO_ENV, tools: toolsOf(), ...credentials }).catch((e) => e);
    expect(refused).toBeInstanceOf(RangeError);
    await expect(agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV, ...credentials })).rejects.toThrow(refused);
  }
});

test("only the last of a node's calls before its honor of an interrupt is interrupted", async () => {
  const log = runLog();
  let ends = 0;
  const model = hooked(
    new ScriptedModel([callReply(["lookup", { topic: "a" }, "c1"]), callReply(["lookup", { topic: "b" }, "c2"])], {
      provider: "anthropic",
      model: "claude-test",
    }),
    (k) => {
      if (k === Infinity && ++ends === 2) void interrupt(log, "notes");
    },
  );
  const tools = { lookup: topicTool("lookup") };
  const { result } = await run(chat("Optional(str)"), model, { history: [], message: "go" }, { log, tools });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const calls = agentCalls(log);
  expect(calls).toHaveLength(2);
  const views = await Promise.all(calls.map((entry) => agentCall(log, entry, { env: NO_ENV })));
  expect(views.map(({ interrupted }) => interrupted)).toEqual([false, true]);
  expect(views.map(({ finish }) => finish)).toEqual(["tool_calls", "tool_calls"]);
});

test("a steer read after the run resumed joins the request after it", async () => {
  const replies = () => [callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")];
  const scripted = () => new ScriptedModel(replies(), { provider: "anthropic", model: "claude-test" });
  const inputs = { history: [], message: "go" };
  const tools = { lookup: topicTool("lookup") };
  const uncut = await run(chat(), scripted(), inputs, { tools });
  // Cut right after the tool result, before the second model call.
  const k = readFileSync(uncut.log, "utf8").split("\n").findIndex((line) => line.includes('"tool_result"')) + 1;
  const log = runLog();
  await crashAfter(k, () => run(chat(), scripted(), inputs, { log, tools }));
  await send(log, "notes", "only 2024", { env: NO_ENV });

  const second = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const resumed = await advance({ log, models: serving(second), env: NO_ENV, tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const text = readFileSync(log, "utf8");
  expect(text.indexOf('"kind":"steer"')).toBeGreaterThan(text.indexOf('"run_resumed"'));
  const [, after] = agentCalls(log);
  const view = await agentCall(log, after!, { env: NO_ENV });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view)).toMatchObject(sentView(second.requests[0]!));
  expect(requestOf(view).messages.at(-1)).toEqual({ role: "user", parts: [{ type: "text", text: "only 2024" }] });
});

test("agentCall refuses the memory option and its view has no memorySections", async () => {
  const model = new ScriptedModel([callReply(["lookup", { topic: "a" }, "call-a"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log } = await run(TOOLS, model, { topic: "owls" }, { tools: toolsOf() });
  const view = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(view).not.toHaveProperty("memorySections");
  expect(view.digestMismatch).toBe(false);
  await expect(
    // @ts-expect-error: the option is gone.
    agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV, memory: new Map() }),
  ).rejects.toThrow("the memory option was removed: pass each memory as a flow input");
});

test("an agent's rebuildRequests returns each request and whether all digests matched", async () => {
  const node = new AgentNode("writer", { prompt: "hi" });
  const caps = { skills: new Map(), toolSpecs: () => [], models: null };
  const material = await node.rebuildRequests({}, [], caps);
  expect(Object.keys(material!).sort()).toEqual(["digestOk", "requests"]);
});

/** A flow whose agent holds `lookup`, `read` of the memory `lessons` and the skill `style`. */
const REMEMBER = (tag: string, grant = "{lessons: [read]}") => `
id: remember
name: remember
llm_config: {provider: anthropic, model: claude-test}
input:
  lessons: !${tag}
nodes:
  writer:
    kind: agent
    input:
      lessons: \${input.lessons}
    tools: [lookup, ${grant}]
    skills: [style]
    prompt: "<memory>\${lessons}</memory> write"
    output: str
output: \${writer.output}
`;

const STYLE = new Map([["style", new Skill({ name: "style", description: "House style.", instructions: "Be brief." })]]);

test("a memory grant: the rebuilt request holds the recorded recall and the memory's tools, as sent", async () => {
  const folder = mkdtempSync(join(root, "memory-"));
  writeFileSync(join(folder, "MEMORY.md"), "Prefer short answers.");
  writeFileSync(join(folder, "b.md"), "B.");
  const lessons = new FileMemory({ root: folder, id: "t/lessons" });
  const model = new ScriptedModel([callReply(["lessons_read", { path: "b.md" }, "call-1"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const { log, result } = await run(REMEMBER("FileMemory"), model, { lessons }, { tools: toolsOf(), skills: STYLE });
  expect(result.status, result.failure?.message).toBe("succeeded");
  // The folder changes after the run; the rebuild reads only the log.
  writeFileSync(join(folder, "MEMORY.md"), "Changed.");
  for (const [n, index] of [[1, 0], [3, 1]] as const) {
    const view = await agentCall(log, { nodeId: "writer", n }, { env: NO_ENV });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view).tools.map((tool) => tool.name)).toEqual(["lookup", "lessons_read", "load_skill"]);
    expect(requestOf(view)).toMatchObject(sentView(model.requests[index]!));
  }
  const first = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(requestOf(first).messages[0]).toEqual({
    role: "user",
    parts: [{ type: "text", text: "<memory>Prefer short answers.\nRead the rest with lessons_read.</memory> write" }],
  });
});

test("a recorded flow whose memory type is a host's own, not passed again, rebuilds the same", async () => {
  const tools = [
    { name: "find", write: false, description: "Find a trade.", parameters: { type: "object", properties: {} } },
    { name: "keep", write: true, description: "Keep a trade.", parameters: { type: "object", properties: {} } },
  ];
  class Journal extends MemoryBase {
    static override readonly contract: MemoryContract = { type: "Journal", tools };
    override async open(): Promise<MemorySession> {
      const session = new RecordingSession({ type: "Journal", identity: "t/journal", settings: { k: 1n }, tools });
      session.onSnapshot = () => ({ version: "v1", recall: { text: "two trades", cut: null, hint: null } });
      session.onCall = () => ({ trades: 2n });
      return session;
    }
    override reset(): Promise<void> {
      throw new Error("not reset");
    }
  }
  const model = new ScriptedModel([callReply(["lessons_find", {}, "call-1"]), textReply("done")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const flow = await loadFlow(REMEMBER("Journal", "lessons"), { memoryTypes: [Journal] });
  const log = runLog();
  const result = await runFlow(flow, { lessons: new Journal() }, {
    models: serving(model),
    env: NO_ENV,
    log,
    tools: toolsOf(),
    skills: STYLE,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  for (const [n, index] of [[1, 0], [3, 1]] as const) {
    const view = await agentCall(log, { nodeId: "writer", n }, { env: NO_ENV });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view).tools.map((tool) => tool.name)).toEqual(["lookup", "lessons_find", "lessons_keep", "load_skill"]);
    expect(requestOf(view)).toMatchObject(sentView(model.requests[index]!));
  }
});

test("a granted Optional memory left out rebuilds with none of its tools and an empty recall", async () => {
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(REMEMBER("Optional(FileMemory)"), model, {}, { tools: toolsOf(), skills: STYLE });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await agentCall(log, { nodeId: "writer", n: 1 }, { env: NO_ENV });
  expect(view.digestMismatch).toBe(false);
  expect(requestOf(view).tools.map((tool) => tool.name)).toEqual(["lookup", "load_skill"]);
  expect(requestOf(view).messages[0]).toEqual({ role: "user", parts: [{ type: "text", text: "<memory></memory> write" }] });
  expect(requestOf(view)).toMatchObject(sentView(model.requests[0]!));
});
