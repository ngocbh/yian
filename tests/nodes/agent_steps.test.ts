/**
 * An AGENT's model calls and tool results are `node_step` records, written as the agent runs. Each
 * successful or failed model call is a `model_call` step, written once the call returns and before
 * any of its tools runs; each tool result is a `tool_result` step, written as that call finishes
 * (in finish order) and before the next model call. A `model_call`'s output holds media as files
 * beside the run, and its `request_sha256` hashes the request in that form, steers left out. Failed
 * calls carry their usage and a scrubbed error; a call a stop or abort cut writes nothing. The
 * step's `n` is its number within the node, which a tool call's idempotency key also uses.
 *
 * Most rows run a YAML flow through `runFlow` with a JSONL log and read the log's records;
 * `AgentSteps` is also tested alone over a recording `NodeSteps`.
 */

import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted, type RunEvent } from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, EventStream, Finish, TextDelta, UsageReport } from "../../src/llm/events.ts";
import {
  MediaPart,
  Message,
  NamedTool,
  OutputFormat,
  Request,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
  Usage,
} from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentSteps, RequestDigest, requestSha256 } from "../../src/nodes/agent/context.ts";
import { rebuildStepRequests } from "../../src/nodes/agent/loop.ts";
import { SKIPPED_TOOL_RESULT } from "../../src/nodes/agent/results.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { decodeRecord, InterruptedRecord, NodeStepRecord, readRunLog, type RecordEnvelope } from "../../src/runtime/log.ts";
import { advance, complete, interrupt, pendingCalls, runFlow } from "../../src/run.ts";
import { virtualScheduler, withWarnings } from "../../src/platform.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { pending, type ToolContext } from "../../src/tools/context.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { crashAfter, NO_ENV, rejected, serving, textReply, thrownBy } from "../helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-steps-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 9, 22, 10, 0, 0, 0);
const TIME = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/;

/** A flow whose agent `writer` may call `tools`; `steerInbox` gives it a steer inbox, so a host may interrupt it. */
function flowText(tools: readonly string[] = ["lookup", "save"], steerInbox = false): string {
  return `
id: steps
name: steps
${steerInbox ? "input:\n  notes: !Inbox(str)\n" : ""}nodes:
  writer:
    kind: agent
    tools: [${tools.join(", ")}]
${steerInbox ? "    steer_inbox: ${input.notes}\n" : ""}    prompt: hi
    output: ${steerInbox ? "Optional(str)" : "str"}
output: \${writer.output}
`;
}

/** A tool `name` that returns `<name>:<topic>`; `safe` marks it concurrency safe; `log` sees each call. */
function topicTool(
  name: string,
  options: { safe?: boolean; log?: string[]; run?: (topic: string, ctx: ToolContext) => unknown } = {},
): FunctionTool {
  const { safe = false, log = [], run = (topic: string) => `${name}:${topic}` } = options;
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    isConcurrencySafe: safe ? () => true : null,
    execute: (args, ctx) => {
      log.push(`${name} ${String(args["topic"])}`);
      return run(String(args["topic"]), ctx);
    },
  });
}

/** A reply calling `calls` (`[id, name, topic]`) in order. */
function calls(...parts: [string, string, string][]): Response {
  return new Response({
    message: new Message({
      role: "assistant",
      parts: parts.map(([id, name, topic]) => new ToolCallPart({ id, name, arguments: { topic } })),
    }),
    finishReason: "tool_calls",
  });
}

/** A fresh run folder and its log path. */
function runLog(): string {
  const dir = mkdtempSync(join(root, "run-"));
  return join(dir, "log.jsonl");
}

/** The decoded records of the log at `path`. */
function records(path: string): RecordEnvelope[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => decodeRecord(line));
}

/** The steps the log at `path` holds for `nodeId`, in order, each as `{n, ...step}`. */
function steps(path: string, nodeId = "writer"): ValueRecord[] {
  return records(path).flatMap(({ record }) =>
    record instanceof NodeStepRecord && record.nodeId === nodeId ? [{ n: record.n, ...record.step }] : [],
  );
}

/** The kinds of the log's records about `nodeId`, a step as `<kind> <n>`. */
function shape(path: string, nodeId = "writer"): string[] {
  return records(path).flatMap(({ record }) => {
    if ((record as { nodeId?: string }).nodeId !== nodeId && record.constructor.name !== "NodeFailedRecord") return [];
    if (record instanceof NodeStepRecord) return [`${String(record.step["kind"])} ${record.n}`];
    return [record.constructor.name];
  });
}

/** An environment with no host setting but `YIAN_MODEL_RETRIES`. */
function retriesEnv(modelRetries: number): (name: string) => string | undefined {
  return (name) => (name === "YIAN_MODEL_RETRIES" ? String(modelRetries) : NO_ENV(name));
}

/**
 * Run `flow` over `model` with a log, `modelRetries` model retries (default 0) and a virtual clock
 * at `START`; the other options go to `runFlow`.
 */
async function run(
  model: ChatModel,
  options: Parameters<typeof runFlow>[2] & { flow?: string; modelRetries?: number } = {},
): Promise<{ log: string; result: Awaited<ReturnType<typeof runFlow>> }> {
  const { flow = flowText(), modelRetries = 0, ...rest } = options;
  const log = (rest.log as string | undefined) ?? runLog();
  const env = retriesEnv(modelRetries);
  const result = await runFlow(await loadFlow(flow), {}, {
    llmConfig: CONFIG,
    env,
    scheduler: virtualScheduler(START),
    models: serving(model),
    ...rest,
    log,
  });
  return { log, result };
}

/** The tool result parts the model's `index`-th request carried, in order. */
function sentResults(model: ScriptedModel, index: number): ToolResultPart[] {
  return model.requests[index]!.messages.flatMap((message) =>
    message.role === "tool" ? [message.parts[0] as ToolResultPart] : [],
  );
}

/** The three replies of the grouped AGENT case: two unsafe calls, two safe and an unsafe one, the text. */
function groupsModel(): ScriptedModel {
  return new ScriptedModel(
    [
      calls(["call-a", "save", "a"], ["call-b", "save", "b"]),
      calls(["call-x", "lookup", "x"], ["call-y", "lookup", "y"], ["call-z", "save", "z"]),
      textReply("all done"),
    ],
    CONFIG,
  );
}

// --- the steps of a run, in order --- //

test("an AGENT writes each model call, then each of its tool results, before the next call", async () => {
  const model = groupsModel();
  const tools = { lookup: topicTool("lookup", { safe: true }), save: topicTool("save") };
  const { log, result } = await run(model, { tools });
  expect(result.status, result.failure?.message).toBe("succeeded");

  expect(shape(log)).toEqual([
    "NodeStartedRecord",
    "model_call 1",
    "tool_result 2",
    "tool_result 3",
    "model_call 4",
    "tool_result 5",
    "tool_result 6",
    "tool_result 7",
    "model_call 8",
    "PoolCommitRecord",
    "PoolCommitRecord",
    "NodeSucceededRecord",
  ]);
  const written = steps(log);
  expect(written.filter((step) => step["kind"] === "tool_result").map((step) => step["call_id"])).toEqual([
    "call-a",
    "call-b",
    "call-x",
    "call-y",
    "call-z",
  ]);
  // Each result's output is exactly what the next request gives the model.
  const results = written.filter((step) => step["kind"] === "tool_result");
  const sent = [...sentResults(model, 1), ...sentResults(model, 2).slice(2)];
  expect(results.map((step) => [step["call_id"], step["output"], step["is_error"]])).toEqual(
    sent.map((part) => [part.callId, part.content, part.isError]),
  );
  const first = written[0]!;
  expect(first).toMatchObject({
    kind: "model_call",
    purpose: "chat",
    model: "claude-test",
    output: calls(["call-a", "save", "a"], ["call-b", "save", "b"]).message.toJSON(),
    finish: "tool_calls",
    response_id: null,
    usage: null,
  });
  expect(first["request_sha256"]).toBe(requestSha256(model.requests[0]!));
  expect(Object.hasOwn(written[1]!, "closed")).toBe(false);
});

test("each step has its start time, and a model call the time of its first event", async () => {
  const model = groupsModel();
  const tools = { lookup: topicTool("lookup", { safe: true }), save: topicTool("save") };
  const { log } = await run(model, { tools });
  const written = steps(log);
  for (const step of written) expect(step["started_at"]).toMatch(TIME);
  for (const step of written.filter((each) => each["kind"] === "model_call")) {
    expect(step["first_token_at"]).toMatch(TIME);
  }
  expect(written[0]!["started_at"]).toBe(new Date(START).toISOString());
});

test("safe calls write their results in finish order; the next request lists them in call order", async () => {
  const bDone = Promise.withResolvers<void>();
  const lookup = topicTool("lookup", {
    safe: true,
    run: async (topic) => {
      // `x` finishes only after `y`.
      if (topic === "x") await bDone.promise;
      else queueMicrotask(() => bDone.resolve());
      return `found:${topic}`;
    },
  });
  const model = new ScriptedModel(
    [calls(["call-x", "lookup", "x"], ["call-y", "lookup", "y"]), textReply("done")],
    CONFIG,
  );
  const { log, result } = await run(model, { tools: { lookup }, flow: flowText(["lookup"]) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const results = steps(log).filter((step) => step["kind"] === "tool_result");
  expect(results.map((step) => [step["n"], step["call_id"]])).toEqual([
    [2, "call-y"],
    [3, "call-x"],
  ]);
  expect(sentResults(model, 1).map((part) => part.callId)).toEqual(["call-x", "call-y"]);
});

// --- written before acting --- //

/** The seq of the first record of the uncut run at `log` whose step kind is `kind`. */
function seqOf(log: string, kind: string): number {
  const found = records(log).find(
    ({ record }) => record instanceof NodeStepRecord && record.step["kind"] === kind,
  );
  return found!.seq;
}

test("a crash at a model_call record leaves its tools never called", async () => {
  const uncut = await run(groupsModel(), { tools: { lookup: topicTool("lookup"), save: topicTool("save") } });
  const ran: string[] = [];
  const tools = { lookup: topicTool("lookup", { log: ran }), save: topicTool("save", { log: ran }) };
  await crashAfter(seqOf(uncut.log, "model_call"), () => run(groupsModel(), { tools }));
  expect(ran).toEqual([]);
});

test("a crash at a tool_result record leaves the next model call never made", async () => {
  const uncut = await run(groupsModel(), { tools: { lookup: topicTool("lookup"), save: topicTool("save") } });
  const model = groupsModel();
  await crashAfter(seqOf(uncut.log, "tool_result"), () =>
    run(model, { tools: { lookup: topicTool("lookup"), save: topicTool("save") } }),
  );
  expect(model.requests.length).toBe(1);
});

test("a resumed AGENT's tool call gets an idempotency key no earlier start used", async () => {
  const keys: string[] = [];
  const save = topicTool("save", { run: (topic, ctx) => (keys.push(ctx.callId), `saved:${topic}`) });
  const flow = flowText(["save"]);
  const replies = () => [calls(["call-a", "save", "a"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools: { save }, flow, runId: "r" });
  keys.length = 0;
  const log = runLog();
  // Cut at the second model_call: the first start wrote its reply (n 1) and the tool's result (n 2).
  const cut = seqOf(uncut.log, "tool_result") + 1;
  await crashAfter(cut, () => run(new ScriptedModel(replies(), CONFIG), { log, tools: { save }, flow, runId: "r" }));
  const resumed = await advance({
    log,
    tools: { save },
    env: retriesEnv(0),
    scheduler: virtualScheduler(START),
    // The first reply and its result replay; the next live reply calls `save` again.
    models: serving(new ScriptedModel([calls(["call-b", "save", "b"]), textReply("done")], CONFIG)),
  });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // Steps outlive the first start, so the live reply is step 3 and its key is new.
  expect(keys).toEqual(["r/writer/1/0", "r/writer/3/0"]);
});

test("a reply whose arguments nest too deep still writes its model_call before the node fails", async () => {
  const deep = '{"a": '.repeat(100) + '"leaf"' + "}".repeat(100);
  const call = new ToolCallPart({ id: "call-1", name: "save", arguments: null, rawArguments: `{"topic": ${deep}}` });
  const model = new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" })],
    CONFIG,
  );
  const { log, result } = await run(model, { tools: { save: topicTool("save") }, flow: flowText(["save"]) });
  expect(result.status).toBe("failed");
  expect(shape(log)).toEqual(["NodeStartedRecord", "model_call 1", "NodeFailedRecord"]);
});

test("a reply that repeats a tool call id fails the node after its model_call, running no tool", async () => {
  const seen: string[] = [];
  const model = new ScriptedModel([calls(["call-1", "lookup", "a"], ["call-1", "lookup", "b"]), textReply("done")], CONFIG);
  const { log, result } = await run(model, { tools: { lookup: topicTool("lookup", { log: seen }) }, flow: flowText(["lookup"]) });
  expect(result.status).toBe("failed");
  expect(result.failure?.message).toContain(
    "the reply has two tool calls with id 'call-1' (lookup and lookup); each tool call in a reply needs its own id",
  );
  expect(seen).toEqual([]);
  expect(shape(log)).toEqual(["NodeStartedRecord", "model_call 1", "NodeFailedRecord"]);
});

// --- failures --- //

test("a retried call writes a failed model_call with its wait, then the successful one", async () => {
  const model = new ScriptedModel([new ProviderError("rate_limit", "slow down", { status: 429 }), textReply("ok")], CONFIG);
  const { log, result } = await run(model, { random: () => 1, modelRetries: 1, flow: flowText([]) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [failed, succeeded, ...rest] = steps(log);
  expect(rest).toEqual([]);
  expect(failed).toEqual({
    n: 1,
    kind: "model_call",
    purpose: "chat",
    model: "claude-test",
    request_sha256: succeeded!["request_sha256"]!,
    started_at: new Date(START).toISOString(),
    first_token_at: null,
    output: null,
    usage: null,
    error: { kind: "rate_limit", message: "slow down" },
    retry_in: 0.375,
  });
  expect(succeeded).toMatchObject({ n: 2, kind: "model_call", finish: "stop" });
  expect(readFileSync(log, "utf8")).not.toContain("model_attempt");
});

test("a key in a failed call's error is scrubbed from both failed model_calls", async () => {
  const key = "sk-sentinel-steps-zqxw";
  const llmConfig = { provider: "vllm", model: "m", api_key: key };
  const failure = () => new ProviderError("server", `upstream saw ${key}`, { status: 500 });
  const model = new ScriptedModel([failure(), failure()], llmConfig);
  const { log, result } = await run(model, { llmConfig, random: () => 1, modelRetries: 1, flow: flowText([]) });
  expect(result.status).toBe("failed");
  const failed = steps(log);
  expect(failed.map((step) => [step["n"], step["error"], step["retry_in"]])).toEqual([
    [1, { kind: "server", message: "upstream saw [REDACTED]" }, 0.375],
    [2, { kind: "server", message: "upstream saw [REDACTED]" }, null],
  ]);
  expect(readFileSync(log, "utf8")).not.toContain(key);
});

test("a failure with no retry left writes a failed model_call before node_failed", async () => {
  const model = new ScriptedModel([new ProviderError("server", "boom", { status: 500 })], CONFIG);
  const { log, result } = await run(model, { flow: flowText([]) });
  expect(result.status).toBe("failed");
  expect(shape(log)).toEqual(["NodeStartedRecord", "model_call 1", "NodeFailedRecord"]);
  expect(steps(log)[0]).toMatchObject({ error: { kind: "server", message: "boom" }, retry_in: null, output: null });
});

test("a reply whose media can't be stored writes a failed model_call and ends its call's bracket", async () => {
  const media = new Message({ role: "assistant", parts: [new MediaPart({ mime: "image/png", data: "ab-_" })] });
  const model = new ScriptedModel([new Response({ message: media, finishReason: "stop" })], CONFIG);
  const events: string[] = [];
  const onEvent = (event: RunEvent) => {
    if (event instanceof ModelCallStarted) events.push(event.type);
    if (event instanceof ModelCallFinished) events.push(`${event.type} ${event.error}`);
  };
  const { log, result } = await run(model, { flow: flowText([]), onEvent });
  expect([result.status, result.failure?.code]).toEqual(["failed", "node.execute"]);
  expect(shape(log)).toEqual(["NodeStartedRecord", "model_call 1", "NodeFailedRecord"]);
  expect(steps(log)[0]).toMatchObject({
    error: { kind: "RangeError", message: "image/png media data isn't base64" },
    retry_in: null,
    output: null,
  });
  expect(events).toEqual(["ModelCallStarted", "ModelCallFinished RangeError"]);
});

test("a failed call writes what arrived and the usage reported before it failed", async () => {
  const usage = new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 3 });
  const cut = [new TextDelta(0, "half"), new UsageReport(usage), new ProviderError("server", "cut", { status: 500 })];
  const model = new ScriptedModel([cut], CONFIG);
  const { log } = await run(model, { flow: flowText([]) });
  expect(steps(log)[0]).toMatchObject({
    output: new Message({ role: "assistant", parts: [new TextPart("half")] }).toJSON(),
    usage: { input_tokens: 7n, output_tokens: 2n, cache_read_tokens: 3n, reasoning_tokens: null },
    retry_in: null,
  });
});

/**
 * Return `model` with `hook(k)` run after the `k`-th event of each stream is read (from 1), and
 * after the last as `hook(Infinity)` when the stream ends.
 */
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

/** The seq of the log's `interrupted` record. */
const interruptSeq = (log: string) => readRunLog(log).find(({ record }) => record instanceof InterruptedRecord)!.seq;

test("a host interrupt that cuts a reply writes it with finish interrupted, then the honor step", async () => {
  const log = runLog();
  const reply = [new TextDelta(0, "hel"), new TextDelta(0, "lo"), new Finish({ reason: "stop" })];
  const model = hooked(new ScriptedModel([reply], CONFIG), (k) => {
    if (k === 1) void interrupt(log, "notes");
  });
  const { result } = await run(model, { log, flow: flowText([], true) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
  expect(steps(log)).toEqual([
    expect.objectContaining({
      n: 1,
      kind: "model_call",
      finish: "interrupted",
      output: new Message({ role: "assistant", parts: [new TextPart("hel")] }).toJSON(),
    }),
    { n: 2, kind: "interrupted", by: BigInt(interruptSeq(log)) },
  ]);
});

test("calls a host interrupt skips each write a skipped result", async () => {
  const log = runLog();
  const ran: string[] = [];
  const model = hooked(new ScriptedModel([calls(["call-a", "save", "a"], ["call-b", "save", "b"])], CONFIG), (k) => {
    if (k === Infinity) void interrupt(log, "notes");
  });
  const tools = { save: topicTool("save", { log: ran }) };
  const { result } = await run(model, { log, tools, flow: flowText(["save"], true) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual([]);
  const skipped = { kind: "tool_result", started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" };
  expect(steps(log).slice(1)).toEqual([
    { n: 2, call_id: "call-a", ...skipped },
    { n: 3, call_id: "call-b", ...skipped },
    { n: 4, kind: "interrupted", by: BigInt(interruptSeq(log)) },
  ]);
});

test("a stop after a call's reply arrived still writes its model_call, and nothing more", async () => {
  const stop = new AbortController();
  const reason = new Error("stop");
  const ran: string[] = [];
  const model = hooked(new ScriptedModel([calls(["call-a", "save", "a"])], CONFIG), (k) => {
    if (k === Infinity) stop.abort(reason);
  });
  const log = runLog();
  const tools = { save: topicTool("save", { log: ran }) };
  expect(await rejected(run(model, { log, signal: stop.signal, tools, flow: flowText(["save"]) }))).toBe(reason);
  expect(ran).toEqual([]);
  expect(shape(log)).toEqual(["NodeStartedRecord", "model_call 1"]);
});

test("a stop during a call writes nothing for it", async () => {
  const stop = new AbortController();
  const reason = new Error("stop");
  const model = hooked(new ScriptedModel([[new TextDelta(0, "hel"), new TextDelta(0, "lo")]], CONFIG), (k) => {
    if (k === 1) stop.abort(reason);
  });
  const log = runLog();
  expect(await rejected(run(model, { log, signal: stop.signal, flow: flowText([]) }))).toBe(reason);
  expect(steps(log)).toEqual([]);
});

// --- the request digest --- //

test("requestSha256 hashes the request's log JSON, every message", () => {
  const steer = new Message({ role: "user", parts: [new TextPart("also this")] });
  const user = new Message({ role: "user", parts: [new TextPart("hi")] });
  const tool = new ToolSpec({ name: "save", description: "Save.", parameters: { type: "object" } });
  const request = new Request({ messages: [user], tools: [tool] });
  const expected = createHash("sha256")
    .update(
      logJson({
        messages: [user.toJSON()],
        tools: [{ name: "save", description: "Save.", parameters: { type: "object" }, strict: null }],
        tool_choice: null,
        parallel_tool_calls: null,
        output_format: null,
      }),
    )
    .digest("hex");
  expect(requestSha256(request)).toBe(expected);
  const steered = new Request({ messages: [user, steer], tools: [tool] });
  expect(requestSha256(steered)).not.toBe(expected);
  const forced = new Request({ messages: [user], tools: [tool], toolChoice: new NamedTool("save") });
  expect(requestSha256(forced)).not.toBe(expected);
  // A long int argument, which canonical (JCS) JSON can't write, still hashes.
  const call = new ToolCallPart({ id: "c1", name: "save", arguments: { n: 2n ** 70n } });
  const long = new Request({ messages: [user, new Message({ role: "assistant", parts: [call] })] });
  expect(requestSha256(long)).toMatch(/^[0-9a-f]{64}$/);
});

/** The digest a step records for `request`, hashed whole from its log JSON. */
function fullDigest(request: Request): string {
  const { messages } = request;
  const choice = request.toolChoice;
  const format = request.outputFormat;
  const json = {
    messages: messages.map((message) => message.toJSON()),
    tools: request.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      strict: tool.strict,
    })),
    tool_choice: choice instanceof NamedTool ? { name: choice.name } : choice,
    parallel_tool_calls: request.parallelToolCalls,
    output_format: format === null ? null : { name: format.name, schema: format.schema, strict: format.strict },
  };
  return createHash("sha256").update(logJson(json), "utf8").digest("hex");
}

test("an incremental request digest matches the full digest on every round of a growing conversation", () => {
  const tools = [
    new ToolSpec({ name: "save", description: "Save.", parameters: { type: "object" } }),
    new ToolSpec({ name: "lookup", description: "Look up ✓.", parameters: { type: "object" }, strict: true }),
  ];
  const format = new OutputFormat({ name: "answer", schema: { type: "object" }, strict: true });
  const shapes: ConstructorParameters<typeof Request>[0][] = [
    { messages: [], tools },
    { messages: [], tools, toolChoice: new NamedTool("save") },
    { messages: [], outputFormat: format },
    { messages: [], tools, toolChoice: "required", parallelToolCalls: false },
  ];
  // A system message with an author and a provider payload, which the digest keeps.
  const system = new Message({
    role: "system",
    parts: [new TextPart("Be brief.")],
    author: "host",
    providerPayload: { cache: true },
  });
  {
    const digest = new RequestDigest();
    const check = (request: Request) => expect(digest.digest(request)).toBe(fullDigest(request));
    check(new Request({ messages: [] }));
    const user = new Message({ role: "user", parts: [new TextPart("hi 日本 😀 there")], author: "ann" });
    const messages = [system, user];
    for (let round = 0; round < 8; round++) {
      check(new Request({ ...shapes[round % shapes.length]!, messages }));
      const call = new ToolCallPart({ id: `c${round}`, name: "save", arguments: { n: 2n ** 70n, round } });
      const parts: (MediaPart | ToolCallPart)[] = [call];
      if (round === 2) parts.unshift(new MediaPart({ mime: "image/png", file: { uri: "media/a.png" } }));
      const result = new ToolResultPart({ callId: `c${round}`, name: "save", content: "ok" });
      messages.push(
        new Message({ role: "assistant", parts, protocol: "catalog:openai", providerPayload: { id: `r${round}` } }),
        new Message({ role: "tool", parts: [result] }),
      );
    }
    // A structured correction extends the conversation by one message it doesn't keep.
    const correction = new Message({ role: "user", parts: [new TextPart("That broke the schema; try again.")] });
    check(new Request({ messages: [...messages, correction], outputFormat: format }));
    check(new Request({ messages: [...messages, correction], outputFormat: format }));
    // Requests that don't extend the last one: a shorter one, another system message, none.
    check(new Request({ messages: messages.slice(0, -1), tools }));
    const other = new Message({ role: "system", parts: [new TextPart("Other.")] });
    check(new Request({ messages: [other, ...messages.slice(1)] }));
    check(new Request({ messages: messages.slice(1), tools }));
    check(new Request({ messages: [] }));
    check(new Request({ messages, tools }));
  }
});

test("a 200-turn tool loop serializes each message a bounded number of times: live, in its request rebuild, on replay", async () => {
  const turns = 200;
  const replies = Array.from({ length: turns }, (_, i) => calls([`c${i}`, "lookup", `t${i}`]));
  replies.push(calls(["h", "hold", "h"]));
  const model = new ScriptedModel(replies, CONFIG);
  const tools = { lookup: topicTool("lookup"), hold: topicTool("hold", { run: () => pending() }) };
  // Past the default cap of 50 turns.
  const flow = flowText(["lookup", "hold"]).replace("    prompt: hi\n", "    prompt: hi\n    env: {max_tool_iterations: -1}\n");
  const spy = spyOn(Message.prototype, "toJSON");
  try {
    const { log, result } = await run(model, { tools, flow });
    expect(result.status, result.failure?.message).toBe("waiting");
    expect(spy.mock.calls.length).toBeLessThanOrEqual(10 * turns);

    spy.mockClear();
    const recorded = steps(log);
    const [request] = model.requests;
    const rebuilt = await rebuildStepRequests(recorded, {
      messages: request!.messages,
      tools: request!.tools,
      structured: async () => null,
    });
    expect(rebuilt.digestOk).toBe(true);
    expect(spy.mock.calls.length).toBeLessThanOrEqual(10 * turns);

    const [held] = pendingCalls(log);
    await complete(log, held!.callId, { output: "held" });
    spy.mockClear();
    const resumed = await advance({
      log,
      tools,
      env: retriesEnv(0),
      scheduler: virtualScheduler(START),
      models: serving(new ScriptedModel([textReply("done")], CONFIG)),
    });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(spy.mock.calls.length).toBeLessThanOrEqual(10 * turns);
  } finally {
    spy.mockRestore();
  }
});

const PNG = Buffer.from("not really a png").toString("base64");

/** A reply holding an inline image, then a call to `save`. */
function imageReply(): Response {
  return new Response({
    message: new Message({
      role: "assistant",
      parts: [
        new MediaPart({ mime: "image/png", data: PNG }),
        new ToolCallPart({ id: "call-a", name: "save", arguments: { topic: "a" } }),
      ],
    }),
    finishReason: "tool_calls",
  });
}

test("a reply's media is a run file; the model_call and the next request hold its file form", async () => {
  const model = new ScriptedModel([imageReply(), textReply("done")], CONFIG);
  const { log, result } = await run(model, { tools: { save: topicTool("save") }, flow: flowText(["save"]) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const name = `${createHash("sha256").update(Buffer.from(PNG, "base64")).digest("hex")}.png`;
  expect(readFileSync(join(log, "..", "media", name)).toString("base64")).toBe(PNG);
  const [first, , second] = steps(log);
  const media = ((first!["output"] as JsonObject)["parts"] as JsonObject[])[0]!;
  expect(media).toMatchObject({ type: "media", file: { uri: `media/${name}`, mime: "image/png", name } });
  expect(Object.hasOwn(media, "data")).toBe(false);
  // The provider got the bytes inline, while the digest hashed the file form.
  const sent = model.requests[1]!.messages.find((message) => message.role === "assistant")!;
  expect((sent.parts[0] as MediaPart).data).toBe(PNG);
  const filed = new Request({
    messages: model.requests[1]!.messages.map((message) =>
      message.role === "assistant" ? Message.fromJSON(first!["output"] as JsonObject) : message,
    ),
    tools: model.requests[1]!.tools,
  });
  expect(second!["request_sha256"]).toBe(requestSha256(filed));
  expect(second!["request_sha256"]).not.toBe(requestSha256(model.requests[1]!));
});

test("a reply's media that isn't a run-folder file is warned about: inline without a log, or by URL", async () => {
  const pictured = (media: MediaPart) =>
    new Response({ message: new Message({ role: "assistant", parts: [new TextPart("see"), media] }), finishReason: "stop" });
  const inline = () => new MediaPart({ mime: "image/png", data: PNG });
  const linked = () => new MediaPart({ mime: "image/png", url: "https://example.com/a.png" });
  const INLINE =
    "node 'writer' replied with media given inline, not as a run-folder file (a run without a log keeps a reply's " +
    "media inline); new_messages leaves it out";
  const LINKED = "node 'writer' replied with media given by URL, not as a run-folder file; new_messages leaves it out";
  for (const [log, media, expected] of [
    [null, inline(), [INLINE]],
    [runLog(), inline(), []],
    [runLog(), linked(), [LINKED]],
  ] as const) {
    const warnings: string[] = [];
    const flow = await loadFlow(flowText([]));
    const options = {
      llmConfig: CONFIG,
      env: retriesEnv(0),
      scheduler: virtualScheduler(START),
      models: serving(new ScriptedModel([pictured(media)], CONFIG)),
      log,
    };
    const result = await withWarnings((message) => warnings.push(message), () => runFlow(flow, {}, options));
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(warnings).toEqual([...expected]);
  }
});

test("an agent that records no steps writes no messages commit, so its media isn't warned about", () => {
  const steps = new AgentSteps(undefined, () => START, null, (error) => error, "writer");
  const output = new Message({ role: "assistant", parts: [new MediaPart({ mime: "image/png", data: PNG })] });
  const warnings: string[] = [];
  withWarnings(
    (message) => warnings.push(message),
    () =>
      steps.modelCall({
        purpose: "chat",
        model: "m",
        requestSha256: "d".repeat(64),
        startedAt: START,
        firstTokenAt: null,
        output,
        responseId: null,
        finish: "stop",
        usage: null,
      }),
  );
  expect(warnings).toEqual([]);
});

// --- AgentSteps alone --- //

/**
 * `AgentSteps` over a recording `NodeSteps` with `prior` steps, numbering each recorded step after
 * them as the engine does, a clock at `START` and no run folder.
 */
function recordingSteps(prior: ValueRecord[] = []): { steps: AgentSteps; recorded: [ValueRecord, unknown][] } {
  const recorded: [ValueRecord, unknown][] = [];
  const nodeSteps = new NodeSteps(1, prior, (step, options) => prior.length + recorded.push([step, options]));
  const scrub = (error: unknown) => {
    if (error instanceof Error) error.message = error.message.replaceAll("secret", "[REDACTED]");
    return error;
  };
  return { steps: new AgentSteps(nodeSteps, () => START, null, scrub, "writer"), recorded };
}

test("AgentSteps writes model_call and tool_result payloads and returns each n", () => {
  const { steps, recorded } = recordingSteps([{ kind: "llm_config" }]);
  const output = new Message({ role: "assistant", parts: [new TextPart("ok")] });
  const usage = new Usage({ inputTokens: 5, outputTokens: 1, totalTokens: 6, cacheCreationTokens: 4, reasoningTokens: 0 });
  const n = steps.modelCall(
    {
      purpose: "chat",
      model: "m",
      requestSha256: "d".repeat(64),
      startedAt: START,
      firstTokenAt: START + 5,
      output,
      responseId: "resp-1",
      finish: "stop",
      usage,
    },
    { settled: true },
  );
  expect(n).toBe(2);
  expect(steps.toolResult({ callId: "c1", startedAt: START, output: "done", isError: false })).toBe(3);
  expect(steps.toolResult({ callId: "c2", startedAt: null, output: "x", isError: true, closed: "skipped" })).toBe(4);
  expect(recorded).toEqual([
    [
      {
        kind: "model_call",
        purpose: "chat",
        model: "m",
        request_sha256: "d".repeat(64),
        started_at: "2026-10-22T10:00:00.000Z",
        first_token_at: "2026-10-22T10:00:00.005Z",
        output: output.toJSON(),
        response_id: "resp-1",
        finish: "stop",
        usage: { input_tokens: 5n, output_tokens: 1n, cache_read_tokens: null, cache_creation_tokens: 4n, reasoning_tokens: 0n },
      },
      { settled: true },
    ],
    [{ kind: "tool_result", call_id: "c1", started_at: "2026-10-22T10:00:00.000Z", output: "done", is_error: false }, undefined],
    [
      { kind: "tool_result", call_id: "c2", started_at: null, output: "x", is_error: true, closed: "skipped" },
      undefined,
    ],
  ]);
});

test("AgentSteps scrubs a failed call's error and names its kind", () => {
  const { steps, recorded } = recordingSteps();
  const fields = {
    purpose: "chat" as const,
    model: "m",
    requestSha256: "d".repeat(64),
    startedAt: START,
    firstTokenAt: null,
    output: null,
    usage: null,
    retryIn: 1.5,
  };
  expect(steps.failedModelCall(fields, new ProviderError("server", "saw secret", { status: 500 }))).toBe(1);
  steps.failedModelCall({ ...fields, retryIn: null }, new TypeError("bad secret"));
  expect(recorded.map(([step]) => [step["error"], step["retry_in"]])).toEqual([
    [{ kind: "server", message: "saw [REDACTED]" }, 1.5],
    [{ kind: "TypeError", message: "bad [REDACTED]" }, null],
  ]);
});

/** A recorded steer step reading `values`, as the engine's copy carries them. */
const steerStep = (values: string[]): ValueRecord => ({ kind: "steer", inbox: "notes", from: values.map((_, i) => i), values });

/** A recorded successful chat `model_call` step replying `reply`. */
function callStep(reply: Message, digest = "d".repeat(64)): ValueRecord {
  return {
    kind: "model_call",
    purpose: "chat",
    model: "m",
    request_sha256: digest,
    started_at: "2026-10-22T10:00:00.000Z",
    first_token_at: null,
    output: reply.toJSON(),
    response_id: null,
    finish: "stop",
    usage: null,
  };
}

const replyText = (text: string) => new Message({ role: "assistant", parts: [new TextPart(text)] });
const replyCall = (id: string) =>
  new Message({ role: "assistant", parts: [new ToolCallPart({ id, name: "save", arguments: { topic: "a" } })] });
const resultStep = (id: string): ValueRecord => ({ kind: "tool_result", call_id: id, started_at: null, output: "ok", is_error: false });

test("AgentSteps hands back the steers recorded before each call, once", () => {
  const prior = [steerStep(["a"]), callStep(replyCall("c1")), resultStep("c1"), steerStep(["b"]), callStep(replyText("done"))];
  const { steps } = recordingSteps(prior);
  expect(steps.takeRecordedSteers()).toEqual([{ n: 1, values: ["a"] }]);
  expect(steps.takeRecordedSteers()).toEqual([]);
  expect(steps.takeRecorded("chat", "d".repeat(64))!.n).toBe(2);
  expect(steps.takeRecordedSteers()).toEqual([{ n: 4, values: ["b"] }]);
  expect(steps.takeRecorded("chat", "d".repeat(64))!.n).toBe(5);
  expect(steps.takeRecordedSteers()).toEqual([]);
});

test("every steer recorded before one call is handed back, in order", () => {
  const prior = [steerStep(["a"]), steerStep(["b"]), callStep(replyText("done")), steerStep(["c"]), steerStep(["d"])];
  const { steps } = recordingSteps(prior);
  expect(steps.takeRecordedSteers()).toEqual([
    { n: 1, values: ["a"] },
    { n: 2, values: ["b"] },
  ]);
  expect(steps.takeRecorded("chat", "d".repeat(64))!.n).toBe(3);
  expect(steps.hasRecordedSteer()).toBe(true);
  expect(steps.takeRecordedSteers()).toEqual([
    { n: 4, values: ["c"] },
    { n: 5, values: ["d"] },
  ]);
  expect(steps.hasRecordedSteer()).toBe(false);
});

test("a trailing steer is handed back after the last call", () => {
  const { steps } = recordingSteps([callStep(replyText("done")), steerStep(["late"])]);
  expect(steps.takeRecordedSteers()).toEqual([]);
  steps.takeRecorded("chat", "d".repeat(64));
  expect(steps.replaying()).toBe(false);
  expect(steps.takeRecordedSteers()).toEqual([{ n: 2, values: ["late"] }]);
});

test("hasRecordedSteer tells whether a steer was recorded before the next call, without taking it", () => {
  const { steps } = recordingSteps([callStep(replyText("first")), steerStep(["late"]), callStep(replyText("second"))]);
  expect(steps.hasRecordedSteer()).toBe(false);
  steps.takeRecorded("chat", "d".repeat(64));
  expect(steps.hasRecordedSteer()).toBe(true);
  expect(steps.hasRecordedSteer()).toBe(true);
  expect(steps.takeRecordedSteers()).toEqual([{ n: 2, values: ["late"] }]);
  expect(steps.hasRecordedSteer()).toBe(false);
});

test("continuesCall is true while recorded failed attempts continue the next call", () => {
  const failed: ValueRecord = {
    ...callStep(replyText("x")),
    output: null,
    error: { kind: "server", message: "busy" },
    retry_in: 2,
  };
  const { steps } = recordingSteps([steerStep(["a"]), failed]);
  expect(steps.replaying()).toBe(false);
  expect(steps.continuesCall()).toBe(true);
  expect(steps.takeRecordedSteers()).toEqual([{ n: 1, values: ["a"] }]);
  expect(steps.priorAttempts("chat", "d".repeat(64))).toEqual({ attempts: 1, retryIn: 2 });
  expect(steps.continuesCall()).toBe(false);
});

test("a steer copy without values is a host fault", () => {
  const { steps } = recordingSteps([{ kind: "steer", inbox: "notes", from: [3] }]);
  const error = thrownBy(() => steps.takeRecordedSteers());
  expect(error).toBeInstanceOf(CapabilityFault);
  expect(((error as CapabilityFault).original as Error).message).toBe(
    "node 'writer' step 1: a steer step without its values, which only the engine's copy of the step carries; resume through the engine",
  );
});

test("AgentSteps without node steps records nothing and still numbers its steps", () => {
  const steps = new AgentSteps(undefined, () => START, null, (error) => error, "writer");
  expect(steps.toolResult({ callId: "c1", startedAt: null, output: "x", isError: false })).toBe(1);
  expect(steps.toolResult({ callId: "c2", startedAt: null, output: "y", isError: false })).toBe(2);
});
