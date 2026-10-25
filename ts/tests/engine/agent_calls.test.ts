/**
 * TS-only: a host reads a run log's AGENT model calls (`agentCalls`) and one call's view
 * (`agentCall`): the request rebuilt from the recorded flow and the steps before it, checked
 * against the recorded digest, then the attempts, the reply and its tool results as recorded.
 *
 * Each run is logged with a scripted model, which keeps every request it received; a rebuilt
 * request must equal the one the model received. No Python counterpart.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, EventStream, Finish, TextDelta } from "../../src/llm/events.ts";
import { NamedTool, type Request } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { fromModelMessage } from "../../src/nodes/agent/messages.ts";
import { virtualScheduler } from "../../src/platform.ts";
import {
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
} from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { logJson } from "../../src/typesys/json.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { callReply, NO_ENV, serving, textReply } from "./run_state_fixtures.ts";

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
    memorySections: [],
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

test("the view holds the model config without the host's key, headers or connection", async () => {
  const key = "sk-sentinel-agent-call-7f3a";
  const header = "hdr-sentinel-agent-call-91c2";
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
  const json = logJson(JSON.parse(JSON.stringify(view)));
  for (const secret of [key, header, "api_key", "headers", "_host_key", "_header_names", "_connection"]) {
    expect(json).not.toContain(secret);
  }
});

test("a learnable run and one that isn't give the same view", async () => {
  const views = [];
  for (const learnable of [false, true]) {
    const model = new ScriptedModel([callReply(["lookup", { topic: "a" }, "call-a"]), textReply("done")], {
      provider: "anthropic",
      model: "claude-test",
    });
    const tools = { lookup: topicTool("lookup"), save: topicTool("save") };
    const { log } = await run(TOOLS, model, { topic: "owls" }, { tools, learnable });
    const view = await agentCall(log, { nodeId: "writer", n: 3 }, { env: NO_ENV });
    expect(view.digestMismatch).toBe(false);
    expect(requestOf(view)).toMatchObject(sentView(model.requests[1]!));
    views.push({ ...view, attempts: [], toolResults: [] });
  }
  expect(views[0]).toEqual(views[1]);
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

const RECALLED = `
id: recalled
name: recalled
llm_config: {provider: anthropic, model: claude-test}
nodes:
  work:
    kind: agent
    long_term_memory: test
    prompt: hi
    output: str
output: \${work.output}
`;

test("memory sections are recalled against the recorded snapshot only when memory is given", async () => {
  const memory = new FakeMemory({ recallText: "notes" });
  const registry = new Map([["test", memory]]);
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(RECALLED, model, {}, { memory: registry });
  expect(result.status, result.failure?.message).toBe("succeeded");

  const recalled = await agentCall(log, { nodeId: "work", n: 1 }, { env: NO_ENV, memory: registry });
  expect(recalled.memorySections).toEqual([{ memory: "test", text: "notes" }]);
  expect(requestOf(recalled)).toMatchObject(sentView(model.requests[0]!));
  expect(recalled.digestMismatch).toBe(false);

  const plain = await agentCall(log, { nodeId: "work", n: 1 }, { env: NO_ENV });
  expect(plain.memorySections).toBeNull();
  expect(requestOf(plain).system).not.toContain("notes");
  expect(plain.digestMismatch).toBe(false);
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
