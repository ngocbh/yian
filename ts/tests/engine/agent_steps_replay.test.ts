/**
 * TS-only: a resumed AGENT replays the steps its earlier starts recorded. Each recorded
 * `model_call` reply is taken in order instead of calling the model, after its purpose and
 * request digest are checked against the rebuilt request (a mismatch is a host fault that writes
 * nothing); each tool call with a recorded `tool_result` gets that result without running. A
 * replayed call with no result is closed lazily (`closed: "crashed"`, outcome unknown) once its
 * check passes. Failed attempts after the last reply make the first live call wait the recorded
 * delay and continue the attempt count. Steers aren't read until the first live call.
 *
 * Most rows run a YAML flow through `runFlow` with a JSONL log, cut it with `crashAfter`, and
 * resume it with `resumeFlow`; the replay cursor is also tested alone over a fake `NodeSteps`.
 */

import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallStarted, type RunEvent } from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, EventStream, Finish, TextDelta } from "../../src/llm/events.ts";
import {
  MediaPart,
  Message,
  type Request,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  Usage,
} from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentSteps, requestSha256 } from "../../src/nodes/agent/context.ts";
import { SKIPPED_TOOL_RESULT, TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { CapabilityFault, type Steering } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { decodeRecord, NodeStepRecord, type RecordEnvelope } from "../../src/runtime/log.ts";
import { inspectRun, resumeFlow, runFlow } from "../../src/run.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { crashAfter, NO_ENV, rejected, serving, textReply } from "./run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-replay-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 9, 22, 10, 0, 0, 0);
const RUN_ID = "r";

/** A flow whose agent `writer` may call `tools`; `steerable` gives it a host inbox; `output` its type. */
function flowText(tools: readonly string[] = ["save"], options: { steerable?: boolean; output?: string } = {}): string {
  const { steerable = false, output = "str" } = options;
  return `
id: replay
name: replay
nodes:
  writer:
    kind: agent
    tools: [${tools.join(", ")}]
    steerable: ${steerable}
    prompt: hi
    output: ${output}
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

/** A reply calling `parts` (`[id, name, topic]`) in order. */
function calls(...parts: [string, string, string][]): Response {
  return new Response({
    message: new Message({
      role: "assistant",
      parts: parts.map(([id, name, topic]) => new ToolCallPart({ id, name, arguments: { topic } })),
    }),
    finishReason: "tool_calls",
  });
}

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(root, "run-")), "log.jsonl");
}

/** The decoded records of the log at `path`. */
function records(path: string): RecordEnvelope[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => decodeRecord(line));
}

/** The steps the log at `path` holds for `writer`, in order, each as `{n, ...step}`. */
function steps(path: string): ValueRecord[] {
  return records(path).flatMap(({ record }) =>
    record instanceof NodeStepRecord && record.nodeId === "writer" ? [{ n: record.n, ...record.step }] : [],
  );
}

/** The kinds of the log's records from seq `from` on, a step as `<kind> <n>`. */
function kinds(path: string, from = 0): string[] {
  return records(path)
    .slice(from)
    .map(({ kind, record }) => (record instanceof NodeStepRecord ? `${String(record.step["kind"])} ${record.n}` : kind));
}

/** The seq of the `index`-th (from 0) record of the log at `log` whose step kind is `kind`. */
function seqOf(log: string, kind: string, index = 0): number {
  const found = records(log).filter(({ record }) => record instanceof NodeStepRecord && record.step["kind"] === kind);
  return found[index]!.seq;
}

/** An environment with no host setting but `YIAN_MODEL_RETRIES`. */
function retriesEnv(modelRetries: number): (name: string) => string | undefined {
  return (name) => (name === "YIAN_MODEL_RETRIES" ? String(modelRetries) : NO_ENV(name));
}

type Tools = Record<string, FunctionTool>;

/** Run `flow` over `model` with a log, run id `r`, a virtual clock at `START` and `modelRetries`. */
async function run(
  model: ChatModel,
  options: { log?: string; flow?: string; tools?: Tools; modelRetries?: number; steering?: Steering } = {},
): Promise<{ log: string; result: Awaited<ReturnType<typeof runFlow>> }> {
  const { log = runLog(), flow = flowText(), tools = {}, modelRetries = 0, steering } = options;
  const result = await runFlow(await loadFlow(flow), {}, {
    llmConfig: CONFIG,
    env: retriesEnv(modelRetries),
    scheduler: virtualScheduler(START),
    random: () => 1,
    models: serving(model),
    runId: RUN_ID,
    tools,
    steering,
    log,
  });
  return { log, result };
}

/** Run `flow` over `model` cut after record `k - 1`, returning the log. */
async function cutRun(
  k: number,
  model: ChatModel,
  options: { flow?: string; tools?: Tools; modelRetries?: number; steering?: Steering } = {},
): Promise<string> {
  const log = runLog();
  await crashAfter(k, () => run(model, { ...options, log }));
  return log;
}

/** Resume the run at `log` over `model`; `at` is the resumed clock's start. */
function resume(
  log: string,
  model: ChatModel,
  options: { tools?: Tools; modelRetries?: number; steering?: Steering; at?: number; events?: RunEvent[] } = {},
) {
  const { tools = {}, modelRetries = 0, steering, at = START + 60_000, events } = options;
  return resumeFlow({
    log,
    tools,
    steering,
    env: retriesEnv(modelRetries),
    scheduler: virtualScheduler(at),
    random: () => 1,
    models: serving(model),
    onEvent: events === undefined ? null : (event) => void events.push(event),
  });
}

/** The tool result parts the model's `index`-th request carried, in order. */
function sentResults(model: ScriptedModel, index: number): ToolResultPart[] {
  return model.requests[index]!.messages.flatMap((message) =>
    message.role === "tool" ? [message.parts[0] as ToolResultPart] : [],
  );
}

/**
 * The log at `path` as comparable entries: run_resumed records and the later starts of `writer`
 * left out, record and step times masked, and a success's commit given as its commit's entry.
 */
function comparable(path: string): unknown[] {
  let started = false;
  const kept = records(path).filter(({ kind, record }) => {
    if (kind === "run_resumed") return false;
    if (kind !== "node_started" || (record as { nodeId?: string }).nodeId !== "writer") return true;
    const again = started;
    started = true;
    return !again;
  });
  const bySeq = new Map(records(path).map((envelope) => [BigInt(envelope.seq), envelope]));
  const entry = ({ kind, payload }: RecordEnvelope): unknown => {
    const copy = structuredClone(payload) as JsonObject;
    const step = copy["step"] as JsonObject | undefined;
    if (step !== undefined) {
      delete step["started_at"];
      delete step["first_token_at"];
    }
    if (kind === "node_succeeded") copy["commit"] = logJson(entry(bySeq.get(copy["commit"] as bigint)!) as JsonObject);
    return { kind, ...copy };
  };
  return kept.map(entry);
}

// --- the replay cursor alone --- //

/** A successful `model_call` step for `purpose` with `digest`, replying `text`. */
function callStep(digest: string, text: string, purpose = "chat"): ValueRecord {
  return {
    kind: "model_call",
    purpose,
    model: "m",
    request_sha256: digest,
    started_at: "2026-10-22T10:00:00.000Z",
    first_token_at: null,
    output: new Message({ role: "assistant", parts: [new TextPart(text)] }).toJSON(),
    response_id: `resp-${text}`,
    finish: "stop",
    usage: { input_tokens: 5n, output_tokens: 1n, cache_read_tokens: null, reasoning_tokens: null },
  };
}

/** A failed `model_call` step with `digest`, retried after `retryIn` seconds (`null`: not retried). */
function failedStep(digest: string, retryIn: number | null): ValueRecord {
  return {
    kind: "model_call",
    purpose: "chat",
    model: "m",
    request_sha256: digest,
    started_at: "2026-10-22T10:00:00.000Z",
    first_token_at: null,
    output: null,
    usage: null,
    error: { kind: "server", message: "boom" },
    retry_in: retryIn,
  };
}

/** A `tool_result` step for `callId`. */
function resultStep(callId: string, output: string, closed?: string): ValueRecord {
  const step: ValueRecord = { kind: "tool_result", call_id: callId, started_at: null, output, is_error: false };
  if (closed !== undefined) step["closed"] = closed;
  return step;
}

/** `AgentSteps` for node `writer` over a `NodeSteps` holding `prior`, numbering new steps after them. */
function replaying(prior: ValueRecord[]): AgentSteps {
  const nodeSteps = new NodeSteps(2, prior, () => prior.length + 1);
  return new AgentSteps(nodeSteps, () => START, null, (error) => error, "writer");
}

const D1 = "1".repeat(64);
const D2 = "2".repeat(64);

test("the replay cursor takes the recorded replies in order and finds each result by call id", () => {
  const steps = replaying([
    { kind: "llm_config", effective_llm_config: {} },
    callStep(D1, "first"),
    resultStep("c1", "one"),
    failedStep(D2, 0.5),
    { kind: "some_later_kind" },
    callStep(D2, "second"),
    resultStep("c2", "two"),
  ]);
  expect(steps.replaying()).toBe(true);
  const first = steps.takeRecorded("chat", D1) as { n: number; response: Response };
  expect(first.n).toBe(2);
  expect(first.response.message.text()).toBe("first");
  expect(first.response.responseId).toBe("resp-first");
  expect(first.response.finishReason).toBe("stop");
  expect(first.response.usage).toEqual(new Usage({ inputTokens: 5, outputTokens: 1, totalTokens: 6 }));
  expect(steps.replaying()).toBe(true);
  // The failed attempt before the second reply is part of that call, so its reply is step 6.
  const second = steps.takeRecorded("chat", D2) as { n: number; response: Response };
  expect(second.n).toBe(6);
  expect(second.response.message.text()).toBe("second");
  expect(steps.replaying()).toBe(false);
  expect(steps.takeRecorded("chat", D1)).toBeNull();
  expect(steps.priorAttempts("chat", D1)).toBeNull();
  expect(steps.recordedResult("c1")).toEqual({ output: "one", isError: false, closed: null });
  expect(steps.recordedResult("c2")).toEqual({ output: "two", isError: false, closed: null });
  expect(steps.recordedResult("c3")).toBeNull();
});

test("failed attempts after the last reply are the next call's prior attempts, taken once", () => {
  const steps = replaying([callStep(D1, "first"), failedStep(D2, 1.5), failedStep(D2, 3)]);
  expect(steps.takeRecorded("chat", D1)).not.toBeNull();
  expect(steps.replaying()).toBe(false);
  expect(steps.takeRecorded("chat", D2)).toBeNull();
  expect(steps.priorAttempts("chat", D2)).toEqual({ attempts: 2, retryIn: 3 });
  expect(steps.priorAttempts("chat", D2)).toBeNull();
});

test("an attempt that wasn't to be retried is a recorded call that failed, replayed as its failure", () => {
  const steps = replaying([callStep(D1, "first"), failedStep(D2, 1), failedStep(D2, null)]);
  expect(steps.takeRecorded("chat", D1)).not.toBeNull();
  expect(steps.replaying()).toBe(true);
  expect(steps.takeRecorded("chat", D2)).toEqual({ n: 3, failure: { kind: "server", message: "boom" } });
  expect(steps.replaying()).toBe(false);
  expect(steps.priorAttempts("chat", D2)).toBeNull();
});

test("a recorded call whose digest or purpose differs is a host fault naming the node, n and both", () => {
  const steps = replaying([failedStep(D1, 1), callStep(D1, "first")]);
  const fault = (() => {
    try {
      steps.takeRecorded("chat", D2);
    } catch (error) {
      return error;
    }
  })();
  expect(fault).toBeInstanceOf(CapabilityFault);
  const cause = (fault as CapabilityFault).cause;
  expect(cause).toBeInstanceOf(RangeError);
  expect((cause as Error).message).toBe(
    `node 'writer' step 1: recorded request_sha256 ${D1}, rebuilt ${D2}; resume with the host that started the run`,
  );
  const purposes = replaying([callStep(D1, "first", "structured_native")]);
  expect(() => purposes.takeRecorded("chat", D1)).toThrow(
    "node 'writer' step 1: recorded purpose structured_native, rebuilt chat; resume with the host that started the run",
  );
  const trailing = replaying([failedStep(D1, 1)]);
  expect(() => trailing.priorAttempts("chat", D2)).toThrow(`recorded request_sha256 ${D1}, rebuilt ${D2}`);
});

// --- resuming a cut AGENT run --- //

/** The two replies of a one-tool run: a call to `save`, then the text. */
function saveModel(): ScriptedModel {
  return new ScriptedModel([calls(["call-a", "save", "a"]), textReply("done")], CONFIG);
}

test("a crash after a tool result replays the reply: the model and the tool run once over both runs", async () => {
  const uncut = await run(saveModel(), { tools: { save: topicTool("save") } });
  const ran: string[] = [];
  const tools = { save: topicTool("save", { log: ran }) };
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, saveModel(), { tools });
  expect(ran).toEqual(["save a"]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
  expect(ran).toEqual(["save a"]);
  expect(model.requests.length).toBe(1);
  // The live call read the replayed reply and its recorded result.
  expect(sentResults(model, 0).map((part) => [part.callId, part.content])).toEqual([["call-a", "save:a"]]);
  expect(comparable(log)).toEqual(comparable(uncut.log));
});

test("a crash mid-call makes the call again under the same n", async () => {
  const uncut = await run(saveModel(), { tools: { save: topicTool("save") } });
  const tools = { save: topicTool("save") };
  // Cut before the second model_call record.
  const log = await cutRun(seqOf(uncut.log, "model_call", 1), saveModel(), { tools });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(model.requests.length).toBe(1);
  expect(steps(log).map((step) => [step["n"], step["kind"]])).toEqual([
    [1, "model_call"],
    [2, "tool_result"],
    [3, "model_call"],
  ]);
  expect(comparable(log)).toEqual(comparable(uncut.log));
});

test("a crash while tools run closes each call with no result lazily, without running it", async () => {
  const replies = () => [calls(["call-a", "save", "a"], ["call-b", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools: { save: topicTool("save") } });
  const ran: string[] = [];
  const tools = { save: topicTool("save", { log: ran }) };
  // Cut before the first tool_result: `save a` ran, its result was never written.
  const log = await cutRun(seqOf(uncut.log, "tool_result"), new ScriptedModel(replies(), CONFIG), { tools });
  const before = records(log).length;
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(ran).toEqual(["save a"]);
  const closer = { kind: "tool_result", started_at: null, output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" };
  expect(steps(log).slice(1, 3)).toEqual([
    { n: 2, call_id: "call-a", ...closer },
    { n: 3, call_id: "call-b", ...closer },
  ]);
  expect(kinds(log, before)).toEqual([
    "run_resumed",
    "node_started",
    "tool_result 2",
    "tool_result 3",
    "model_call 4",
    "pool_commit",
    "node_succeeded",
    "node_started",
    "pool_commit",
    "node_succeeded",
  ]);
  expect(sentResults(model, 0).map((part) => [part.callId, part.content, part.isError])).toEqual([
    ["call-a", TOOL_OUTCOME_UNKNOWN, true],
    ["call-b", TOOL_OUTCOME_UNKNOWN, true],
  ]);
});

test("a replayed call with no result whose check fails fails the node as live, with no closer", async () => {
  // A call to a tool the agent isn't granted fails its check.
  const reply = () => new ScriptedModel([calls(["call-a", "drop", "a"]), textReply("done")], CONFIG);
  const uncut = await run(reply(), { tools: { save: topicTool("save") } });
  expect(uncut.result.status).toBe("failed");
  // Cut after the model_call, before the node's failure.
  const log = await cutRun(seqOf(uncut.log, "model_call") + 1, reply(), { tools: { save: topicTool("save") } });
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools: { save: topicTool("save") } });
  expect(resumed.status).toBe("failed");
  expect(resumed.failure?.message).toBe(uncut.result.failure?.message);
  expect(model.requests.length).toBe(0);
  expect(steps(log).map((step) => step["kind"])).toEqual(["model_call"]);
});

test("a crash in a retry wait waits the recorded delay again, and the attempt count continues", async () => {
  const slow = () => new ProviderError("rate_limit", "slow down", { status: 429, retryAfter: 2 });
  const uncut = await run(new ScriptedModel([slow(), textReply("ok")], CONFIG), { flow: flowText([]), modelRetries: 1 });
  expect(steps(uncut.log)[0]!["retry_in"]).toBe(2);
  // Cut right after the failed model_call: the run died in its wait.
  const log = await cutRun(seqOf(uncut.log, "model_call") + 1, new ScriptedModel([slow(), textReply("ok")], CONFIG), {
    flow: flowText([]),
    modelRetries: 1,
  });
  const events: RunEvent[] = [];
  const at = START + 60_000;
  const resumed = await resume(log, new ScriptedModel([textReply("ok")], CONFIG), { modelRetries: 1, events, at });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const started = events.filter((event): event is ModelCallStarted => event instanceof ModelCallStarted);
  expect(started.map((event) => event.attempt)).toEqual([2]);
  expect(steps(log)[1]).toMatchObject({ n: 2, finish: "stop", started_at: new Date(at + 2000).toISOString() });

  // With the budget spent by the recorded attempt, one more failure fails the node.
  const again = await cutRun(seqOf(uncut.log, "model_call") + 1, new ScriptedModel([slow(), textReply("ok")], CONFIG), {
    flow: flowText([]),
    modelRetries: 1,
  });
  const failing = new ScriptedModel([slow(), textReply("never")], CONFIG);
  const failed = await resume(again, failing, { modelRetries: 1 });
  expect(failed.status).toBe("failed");
  expect(failing.requests.length).toBe(1);
  expect(steps(again).map((step) => [step["n"], step["retry_in"]])).toEqual([
    [1, 2],
    [2, null],
  ]);
});

test("a crash after a call's last failed attempt fails the node with the recorded error, no model call", async () => {
  const failure = () => new ProviderError("context_length", "too long", { status: 400 });
  const uncut = await run(new ScriptedModel([failure()], CONFIG), { flow: flowText([]) });
  expect(uncut.result.status).toBe("failed");
  const log = await cutRun(seqOf(uncut.log, "model_call") + 1, new ScriptedModel([failure()], CONFIG), {
    flow: flowText([]),
  });
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model);
  expect(resumed.status).toBe("failed");
  expect(model.requests.length).toBe(0);
  const { code, message, causeType } = resumed.failure!;
  expect({ code, message, causeType }).toEqual({
    code: uncut.result.failure!.code,
    message: uncut.result.failure!.message,
    causeType: "ProviderError",
  });
  expect(code).toBe("provider.context_length");
  // A non-provider error keeps its name; its code is the one any model failure gets.
  const odd = () => Object.assign(new Error("socket closed"), { name: "ConnectionError" });
  const oddRun = await run(new ScriptedModel([odd()], CONFIG), { flow: flowText([]) });
  const oddLog = await cutRun(seqOf(oddRun.log, "model_call") + 1, new ScriptedModel([odd()], CONFIG), {
    flow: flowText([]),
  });
  const oddResumed = await resume(oddLog, new ScriptedModel([], CONFIG));
  expect([oddResumed.failure!.code, oddResumed.failure!.message, oddResumed.failure!.causeType]).toEqual([
    oddRun.result.failure!.code,
    "socket closed",
    "ConnectionError",
  ]);
});

test("a request that doesn't match the recorded digest is a host fault; the run stays resumable", async () => {
  const uncut = await run(saveModel(), { tools: { save: topicTool("save") } });
  const tools = { save: topicTool("save") };
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, saveModel(), { tools });
  const digest = steps(log)[0]!["request_sha256"] as string;
  const forged = "f".repeat(64);
  const original = readFileSync(log, "utf8");
  writeFileSync(log, original.replace(digest, forged));
  const before = records(log).length;
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const error = await rejected(resume(log, model, { tools }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(
    `node 'writer' step 1: recorded request_sha256 ${forged}, rebuilt ${digest}; resume with the host that started the run`,
  );
  expect(model.requests.length).toBe(0);
  // Only the resume's own start is written: no step, no failure.
  expect(kinds(log, before)).toEqual(["run_resumed", "node_started"]);
  expect(inspectRun(log).status).toBe("interrupted");
  writeFileSync(log, readFileSync(log, "utf8").replace(forged, digest));
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
});

/** A host inbox whose drains return `batches` in turn (then none), logging each drain. */
function inbox(batches: string[][], drains: number[] = []): Steering {
  return {
    interruptSignal: () => new AbortController().signal,
    drain: () => {
      drains.push(drains.length + 1);
      return batches.shift() ?? [];
    },
  };
}

/** The text of every user message in `request`. */
function userTexts(request: Request): string[] {
  return request.messages.filter((message) => message.role === "user").map((message) => message.text());
}

test("a steer queued while the run is down joins the first live call; a replayed steered call matches", async () => {
  const replies = () => [calls(["call-a", "save", "a"]), calls(["call-b", "save", "b"]), textReply("done")];
  const flow = flowText(["save"], { steerable: true });
  const tools = { save: topicTool("save") };
  // The second call is steered: its request holds the steer, which its digest leaves out.
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools, steering: inbox([[], ["first"]]) });
  // Cut after the second call's tool result, before the third call.
  const log = await cutRun(seqOf(uncut.log, "tool_result", 1) + 1, new ScriptedModel(replies(), CONFIG), {
    flow,
    tools,
    steering: inbox([[], ["first"]]),
  });
  const drains: number[] = [];
  let drainedBeforeLive = -1;
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const watched: ChatModel = {
    provider: model.provider,
    model: model.model,
    stream: (request, options) => {
      drainedBeforeLive = drains.length;
      return model.stream(request, options);
    },
  };
  const resumed = await resume(log, watched, { tools, steering: inbox([["while down"]], drains) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // Neither replayed call drained the inbox; the live call drained it once, first.
  expect(drainedBeforeLive).toBe(1);
  expect(drains).toEqual([1]);
  // The steer read before the crash is lost; the one queued while down joins the live call.
  expect(userTexts(model.requests[0]!)).toEqual(["hi", "while down"]);
  expect(sentResults(model, 0).map((part) => part.callId)).toEqual(["call-a", "call-b"]);
});

test("replay uses the recorded tool call ids, and a re-run call's idempotency key is the original's", async () => {
  const keys: string[] = [];
  const save = topicTool("save", { run: (topic, ctx) => (keys.push(ctx.callId), `saved:${topic}`) });
  const replies = () => [calls(["call-a", "save", "a"]), calls(["call-b", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools: { save } });
  expect(keys).toEqual(["r/writer/1/0", "r/writer/3/0"]);
  keys.length = 0;
  // Cut before the second model_call: the second call is made again.
  const log = await cutRun(seqOf(uncut.log, "model_call", 1), new ScriptedModel(replies(), CONFIG), { tools: { save } });
  expect(keys).toEqual(["r/writer/1/0"]);
  keys.length = 0;
  // The resumed model would give the first call other ids; it is never asked for it.
  const model = new ScriptedModel([calls(["call-b", "save", "b"]), textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools: { save } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(keys).toEqual(["r/writer/3/0"]);
  const replayed = model.requests[0]!.messages.find((message) => message.role === "assistant")!;
  expect((replayed.parts[0] as ToolCallPart).id).toBe("call-a");
  expect(sentResults(model, 0).map((part) => part.callId)).toEqual(["call-a"]);
});

/** A host inbox with no steers whose `interrupt()` interrupts every node. */
function interruptible(): { steering: Steering; interrupt: () => void } {
  const controller = new AbortController();
  return {
    steering: { interruptSignal: () => controller.signal, drain: () => [] },
    interrupt: () => controller.abort(new Error("interrupted")),
  };
}

/** Return `model` with `hook(k)` run after the `k`-th event of each stream (from 1), `Infinity` at its end. */
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

test("a recorded interrupted reply replays as the interrupt: its text is the output, no model is called", async () => {
  const flow = flowText([], { steerable: true });
  const cutReply = () => [new TextDelta(0, "hel"), new TextDelta(0, "lo"), new Finish({ reason: "stop" })];
  const first = interruptible();
  const uncut = await run(
    hooked(new ScriptedModel([cutReply()], CONFIG), (k) => k === 1 && first.interrupt()),
    { flow, steering: first.steering },
  );
  expect(uncut.result.output).toBe("hel");
  const second = interruptible();
  const log = await cutRun(
    seqOf(uncut.log, "model_call") + 1,
    hooked(new ScriptedModel([cutReply()], CONFIG), (k) => k === 1 && second.interrupt()),
    { flow, steering: second.steering },
  );
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { steering: interruptible().steering });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("hel");
  expect(model.requests.length).toBe(0);
});

test("recorded skipped results replay the host interrupt that skipped them", async () => {
  const flow = flowText(["save"], { steerable: true });
  const reply = () => [calls(["call-a", "save", "a"], ["call-b", "save", "b"])];
  const ran: string[] = [];
  const tools = { save: topicTool("save", { log: ran }) };
  const first = interruptible();
  const uncut = await run(
    hooked(new ScriptedModel(reply(), CONFIG), (k) => k === Infinity && first.interrupt()),
    { flow, tools, steering: first.steering },
  );
  expect(uncut.result.status).toBe("succeeded");
  const second = interruptible();
  // Cut after both skipped results, before the node's commit.
  const log = await cutRun(
    seqOf(uncut.log, "tool_result", 1) + 1,
    hooked(new ScriptedModel(reply(), CONFIG), (k) => k === Infinity && second.interrupt()),
    { flow, tools, steering: second.steering },
  );
  expect(steps(log).map((step) => step["closed"] ?? null)).toEqual([null, "skipped", "skipped"]);
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools, steering: interruptible().steering });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe(uncut.result.output);
  expect(model.requests.length).toBe(0);
  expect(ran).toEqual([]);
  expect(steps(log).map((step) => step["output"])).toContain(SKIPPED_TOOL_RESULT);
});

test("a reply whose arguments nest too deep fails the resumed node again after its step", async () => {
  const deep = '{"a": '.repeat(100) + '"leaf"' + "}".repeat(100);
  const call = new ToolCallPart({ id: "call-1", name: "save", arguments: null, rawArguments: `{"topic": ${deep}}` });
  const reply = () =>
    new ScriptedModel(
      [new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" })],
      CONFIG,
    );
  const uncut = await run(reply(), { tools: { save: topicTool("save") } });
  expect(uncut.result.status).toBe("failed");
  const log = await cutRun(seqOf(uncut.log, "model_call") + 1, reply(), { tools: { save: topicTool("save") } });
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools: { save: topicTool("save") } });
  expect(resumed.status).toBe("failed");
  expect(resumed.failure?.message).toBe(uncut.result.failure?.message);
  expect(model.requests.length).toBe(0);
});

test("a structured reply replays without a model call", async () => {
  const flow = flowText([], { output: "{answer: str}" });
  const reply = () =>
    new ScriptedModel([new Response({ message: new Message({ role: "assistant", parts: [new TextPart('{"answer": "yes"}')] }), finishReason: "stop" })], CONFIG);
  const uncut = await run(reply(), { flow });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  const log = await cutRun(seqOf(uncut.log, "model_call") + 1, reply(), { flow });
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model);
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toEqual(uncut.result.output);
  expect(model.requests.length).toBe(0);
});

const PNG = Buffer.from("not really a png").toString("base64");

/** A reply holding an inline image, then a call to `save`, then the text. */
function imageModel(): ScriptedModel {
  const image = new Response({
    message: new Message({
      role: "assistant",
      parts: [
        new MediaPart({ mime: "image/png", data: PNG }),
        new ToolCallPart({ id: "call-a", name: "save", arguments: { topic: "a" } }),
      ],
    }),
    finishReason: "tool_calls",
  });
  return new ScriptedModel([image, textReply("done")], CONFIG);
}

test("a replayed reply's media matches its digest and goes to the next live call inline", async () => {
  const uncutModel = imageModel();
  const uncut = await run(uncutModel, { tools: { save: topicTool("save") } });
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, imageModel(), { tools: { save: topicTool("save") } });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools: { save: topicTool("save") } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const media = (request: Request) =>
    (request.messages.find((message) => message.role === "assistant")!.parts[0] as MediaPart).data;
  expect(media(model.requests[0]!)).toBe(PNG);
  expect(media(model.requests[0]!)).toBe(media(uncutModel.requests[1]!));
  expect(steps(log)[2]!["request_sha256"]).toBe(steps(uncut.log)[2]!["request_sha256"]!);
  const name = `${createHash("sha256").update(Buffer.from(PNG, "base64")).digest("hex")}.png`;
  expect(readFileSync(join(log, "..", "media", name)).toString("base64")).toBe(PNG);
});

test("an AGENT run resumed at every cut point ends as the uncut run, running no tool call twice", async () => {
  const replies = () => [
    calls(["call-a", "save", "a"], ["call-b", "save", "b"]),
    calls(["call-x", "lookup", "x"], ["call-y", "lookup", "y"], ["call-z", "save", "z"]),
    textReply("all done"),
  ];
  const flow = flowText(["lookup", "save"]);
  const toolsFor = (ran: string[]) => ({
    lookup: topicTool("lookup", { safe: true, log: ran }),
    save: topicTool("save", { log: ran }),
  });
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools: toolsFor([]) });
  const count = records(uncut.log).length;
  // From the first cut that keeps the whole run start.
  const first = records(uncut.log).findIndex(({ kind }) => kind === "node_started");
  for (let k = first; k < count - 1; k += 1) {
    const ran: string[] = [];
    const log = await cutRun(k, new ScriptedModel(replies(), CONFIG), { flow, tools: toolsFor(ran) });
    // The replies the cut run's log holds are replayed; the rest are served live.
    const kept = steps(log).filter((step) => step["kind"] === "model_call").length;
    const resumed = await resume(log, new ScriptedModel(replies().slice(kept), CONFIG), { tools: toolsFor(ran) });
    expect(resumed.status, `cut at ${k}: ${resumed.failure?.message}`).toBe("succeeded");
    expect(resumed.output).toBe("all done");
    expect(new Set(ran).size, `cut at ${k}: ${ran.join(", ")}`).toBe(ran.length);
  }
});
