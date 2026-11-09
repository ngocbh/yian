/**
 * A resumed AGENT replays the steps its earlier starts recorded. Each recorded `model_call` reply
 * is taken in order instead of calling the model, after its purpose and request digest are checked
 * against the rebuilt request (a mismatch is a host fault that writes nothing); each tool call with
 * a recorded `tool_result` gets that result without running. A replayed call with no result is
 * closed lazily (`closed: "crashed"`, outcome unknown) once its check passes. Failed attempts after
 * the last reply make the first live call wait the recorded delay and continue the attempt count.
 * Steers aren't read until the first live call.
 *
 * Most rows run a YAML flow through `runFlow` with a JSONL log, cut it with `crashAfter`, and
 * resume it with `advance`; the replay cursor is also tested alone over a fake `NodeSteps`.
 */

import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallStarted, type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
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
import { SKIPPED_TOOL_RESULT, TOOL_OUTCOME_UNKNOWN, unparsedText } from "../../src/nodes/agent/results.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { INTERRUPTED_MARKER } from "../../src/nodes/agent/messages.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { decodeRecord, InterruptedRecord, NodeStepRecord, PoolCommitRecord, type RecordEnvelope } from "../../src/runtime/log.ts";
import { advance, foldRunLog, inspectRun, interrupt, runFlow, send } from "../../src/run.ts";
import { pending, type ToolContext } from "../../src/tools/context.ts";
import { ReportedToolError } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { crashAfter, NO_ENV, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-replay-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 9, 22, 10, 0, 0, 0);
const RUN_ID = "r";

/** A flow whose agent `writer` may call `tools`; `steerInbox` gives it a steer inbox `notes`; `output` its type. */
function flowText(tools: readonly string[] = ["save"], options: { steerInbox?: boolean; output?: string } = {}): string {
  const { steerInbox = false, output = "str" } = options;
  return `
id: replay
name: replay
${steerInbox ? "input:\n  notes: !Inbox(str)\n" : ""}nodes:
  writer:
    kind: agent
    tools: [${tools.join(", ")}]
${steerInbox ? "    steer_inbox: ${input.notes}\n" : ""}    prompt: hi
    output: ${output}
output: \${writer.output}
`;
}

/**
 * A tool `name` that returns `<name>:<topic>`; `safe` marks it concurrency safe, `resumable`
 * resumable; `log` sees each call.
 */
function topicTool(
  name: string,
  options: { safe?: boolean; resumable?: boolean; log?: string[]; run?: (topic: string, ctx: ToolContext) => unknown } = {},
): FunctionTool {
  const { safe = false, resumable = false, log = [], run = (topic: string) => `${name}:${topic}` } = options;
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    isConcurrencySafe: safe ? () => true : null,
    resumable,
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

/** The log's `tool_result` steps for `writer`: call id, output, whether an error, and the closer word if any. */
function toolResults(path: string): Record<string, unknown>[] {
  return steps(path)
    .filter((step) => step["kind"] === "tool_result")
    .map(({ call_id, output, is_error, closed }) => ({ call_id, output, is_error, ...(closed === undefined ? {} : { closed }) }));
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
  options: {
    log?: string;
    flow?: string;
    tools?: Tools;
    modelRetries?: number;
    models?: ModelAccess;
    signal?: AbortSignal;
  } = {},
): Promise<{ log: string; result: Awaited<ReturnType<typeof runFlow>> }> {
  const { log = runLog(), flow = flowText(), tools = {}, modelRetries = 0, models = serving(model), signal } = options;
  const result = await runFlow(await loadFlow(flow), {}, {
    signal,
    llmConfig: CONFIG,
    env: retriesEnv(modelRetries),
    scheduler: virtualScheduler(START),
    random: () => 1,
    models,
    runId: RUN_ID,
    tools,
    log,
  });
  return { log, result };
}

/** Run `flow` over `model` cut after record `k - 1`, returning the log. */
async function cutRun(
  k: number,
  model: ChatModel,
  options: { flow?: string; tools?: Tools; modelRetries?: number; log?: string } = {},
): Promise<string> {
  const log = options.log ?? runLog();
  await crashAfter(k, () => run(model, { ...options, log }));
  return log;
}

/** Resume the run at `log` over `model`; `at` is the resumed clock's start. */
function resume(
  log: string,
  model: ChatModel,
  options: { tools?: Tools; modelRetries?: number; at?: number; events?: RunEvent[]; models?: ModelAccess } = {},
) {
  const { tools = {}, modelRetries = 0, at = START + 60_000, events, models = serving(model) } = options;
  return advance({
    log,
    tools,
    env: retriesEnv(modelRetries),
    scheduler: virtualScheduler(at),
    random: () => 1,
    models,
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
 * left out, record and step times masked, a success's commit and messages commit given as their
 * entries, a messages commit's steps as the steps they name, and a capture source's
 * `execution_seq` as the kind and node of the record it names.
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
    if (kind === "node_succeeded" && copy["messages"] !== null) {
      copy["messages"] = logJson(entry(bySeq.get(copy["messages"] as bigint)!) as JsonObject);
    }
    if (kind === "pool_commit" && Array.isArray(copy["steps"])) {
      copy["steps"] = (copy["steps"] as bigint[]).map((seq) => logJson(entry(bySeq.get(seq)!) as JsonObject));
    }
    const named = (value: unknown): void => {
      if (typeof value !== "object" || value === null) return;
      const record = value as Record<string, unknown>;
      for (const [key, item] of Object.entries(record)) {
        if (key !== "execution_seq") named(item);
        else {
          const source = bySeq.get(item as bigint)!;
          record[key] = `${source.kind} ${String((source.record as { nodeId?: string }).nodeId)}`;
        }
      }
    };
    named(copy);
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
  expect(steps.recordedResult(2, "c1")).toEqual({ output: "one", isError: false, closed: null, n: 3 });
  expect(steps.recordedResult(6, "c2")).toEqual({ output: "two", isError: false, closed: null, n: 7 });
  // A result belongs to the reply before it.
  expect(steps.recordedResult(6, "c1")).toBeNull();
  expect(steps.recordedResult(2, "c3")).toBeNull();
});

test("the replay cursor finds the step an earlier start pended a call in", () => {
  const steps = replaying([
    callStep(D1, "first"),
    { kind: "pending", call_id: "r/writer/1/0", tool: "hold", tool_call_id: "c1", reply: 1n, index: 0n },
  ]);
  expect(steps.recordedPending("r/writer/1/0")).toBe(2);
  expect(steps.recordedPending("r/writer/1/1")).toBeNull();
});

test("a recorded reply's usage reads back its cache writes; a step written without them reads null", () => {
  const counts = { input_tokens: 5n, output_tokens: 1n, cache_read_tokens: 2n, cache_creation_tokens: 3n, reasoning_tokens: 0n };
  const steps = replaying([{ ...callStep(D1, "first"), usage: counts }, callStep(D2, "second")]);
  const first = steps.takeRecorded("chat", D1) as { response: Response };
  expect(first.response.usage).toEqual(
    new Usage({ inputTokens: 5, outputTokens: 1, totalTokens: 6, cacheReadTokens: 2, cacheCreationTokens: 3, reasoningTokens: 0 }),
  );
  const second = steps.takeRecorded("chat", D2) as { response: Response };
  expect(second.response.usage!.cacheCreationTokens).toBeNull();
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

test("a replayed call with a recorded result keeps it without checking its arguments again", async () => {
  const tools = { save: topicTool("save") };
  const uncut = await run(saveModel(), { tools });
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, saveModel(), { tools });
  const validate = spyOn(ToolSet.prototype, "validate");
  try {
    const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(validate).not.toHaveBeenCalled();
  } finally {
    validate.mockRestore();
  }
});

test("a replayed reply whose arguments hold ints and floats matches its digest", async () => {
  const reply = () =>
    new ScriptedModel(
      [
        new Response({
          message: new Message({
            role: "assistant",
            parts: [new ToolCallPart({ id: "call-a", name: "save", arguments: { topic: "a", n: 2n, f: 1.5, w: 3 } })],
          }),
          finishReason: "tool_calls",
        }),
        textReply("done"),
      ],
      CONFIG,
    );
  const save = () =>
    new FunctionTool({
      name: "save",
      description: "Tool save.",
      parametersJsonSchema: {
        type: "object",
        properties: { topic: { type: "string" }, n: { type: "integer" }, f: { type: "number" }, w: { type: "number" } },
      },
      execute: (args) => logJson(args),
    });
  const uncut = await run(reply(), { tools: { save: save() } });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, reply(), { tools: { save: save() } });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools: { save: save() } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(steps(log)[2]!["request_sha256"]).toBe(steps(uncut.log)[2]!["request_sha256"]!);
});

test("a call id two replies share finds each reply's own result", async () => {
  // Gemini makes up `yian_call_<i>` per reply, so a later reply's first call has the first's id.
  const replies = () => [calls(["c1", "save", "a"]), calls(["c1", "save", "b"]), textReply("done")];
  const tools = { save: topicTool("save") };
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools });
  // Cut before the second reply's result: its call must be closed, not given the first's result.
  const log = await cutRun(seqOf(uncut.log, "tool_result", 1), new ScriptedModel(replies(), CONFIG), { tools });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(sentResults(model, 0).map((part) => [part.callId, part.content])).toEqual([
    ["c1", "save:a"],
    ["c1", TOOL_OUTCOME_UNKNOWN],
  ]);
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

test("an agent's tool call sees attempt 1 live", async () => {
  const seen: number[] = [];
  const save = topicTool("save", { run: (_t, ctx) => (seen.push(ctx.attempt), "ok") });
  await run(new ScriptedModel([calls(["c1", "save", "a"]), textReply("done")], CONFIG), { tools: { save } });
  expect(seen).toEqual([1]);
});

test("a call of a reply made after a resume sees attempt 1", async () => {
  const seen: [string, number][] = [];
  const save = topicTool("save", { run: (_t, ctx) => (seen.push([ctx.callId, ctx.attempt]), "ok") });
  const replies = () => [calls(["c1", "save", "a"]), calls(["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools: { save } });
  seen.length = 0;
  // Cut before the second model_call record: the resumed start makes the second reply live.
  const log = await cutRun(seqOf(uncut.log, "model_call", 1), new ScriptedModel(replies(), CONFIG), { tools: { save } });
  const resumed = await resume(log, new ScriptedModel(replies().slice(1), CONFIG), { tools: { save } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(seen).toEqual([
    ["r/writer/1/0", 1],
    ["r/writer/3/0", 1],
  ]);
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

test("a call cut off before its unparsed-arguments result gets the same error again, not a closer", async () => {
  const unparsed = new ToolCallPart({ id: "call-a", name: "save", arguments: null, rawArguments: '{"topic": ' });
  const reply = new Response({ message: new Message({ role: "assistant", parts: [unparsed] }), finishReason: "tool_calls" });
  const model = () => new ScriptedModel([reply, textReply("done")], CONFIG);
  const ran: string[] = [];
  const tools = { save: topicTool("save", { log: ran }) };
  const uncut = await run(model(), { tools });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  const log = await cutRun(afterCall(uncut.log), model(), { tools });
  const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(ran).toEqual([]);
  expect(toolResults(log)).toEqual([{ call_id: "call-a", output: unparsedText("save"), is_error: true }]);
  expect(comparable(log)).toEqual(comparable(uncut.log));
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
  // The live attempt is the step after the recorded failed one.
  expect(started.map((event) => event.n)).toEqual([2]);
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
  expect(inspectRun(log).status).toBe("crashed");
  writeFileSync(log, readFileSync(log, "utf8").replace(forged, digest));
  const resumed = await resume(log, model, { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
});

/** The text parts of every user message in `request`, a message's parts joined by `|`. */
function userTexts(request: Request): string[] {
  return request.messages
    .filter((message) => message.role === "user")
    .map((message) => message.parts.map((part) => (part as TextPart).text).join("|"));
}

test("a send made while the run is down joins the first live call with the one read before the crash", async () => {
  const replies = () => [calls(["call-a", "save", "a"]), calls(["call-b", "save", "b"]), textReply("done")];
  const flow = flowText(["save"], { steerInbox: true });
  // The second call's tool sends a steer to the run, which the third call reads.
  const sending = (log: string) => ({
    save: topicTool("save", {
      run: async (topic) => {
        if (topic === "b") await send(log, "notes", "first");
        return `save:${topic}`;
      },
    }),
  });
  const first = runLog();
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { log: first, flow, tools: sending(first) });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  // Cut after the second call's tool result, before the steer read and the third call.
  const log = runLog();
  await crashAfter(seqOf(uncut.log, "tool_result", 1) + 1, () =>
    run(new ScriptedModel(replies(), CONFIG), { log, flow, tools: sending(log) }),
  );
  await send(log, "notes", "while down", { env: retriesEnv(0) });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resume(log, model, { tools: sending(log) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // Both unread sends join the live call as one read.
  expect(userTexts(model.requests[0]!)).toEqual(["hi", "first|while down"]);
  expect(sentResults(model, 0).map((part) => part.callId)).toEqual(["call-a", "call-b"]);
  expect(resumed.unread).toEqual(new Map());
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

/** Deliver an interrupt of `notes` to the live run at `log`; it lands in this tick. */
const interruptAt = (log: string) => void interrupt(log, "notes");

/** The seq of the `interrupted` record in the log at `path`. */
const interruptSeq = (path: string) => records(path).find(({ record }) => record instanceof InterruptedRecord)!.seq;

/** The honor steps of `writer` in the log at `path`, as their `by`. */
const honors = (path: string) => steps(path).flatMap((step) => (step["kind"] === "interrupted" ? [Number(step["by"])] : []));

/** `writer.new_messages` of the run at `path`, as `[role, text or part types]` per message. */
async function newMessages(path: string): Promise<[string, string][]> {
  const folded = await foldRunLog(path, { env: retriesEnv(0) });
  const value = folded.records.findLast(
    ({ entry }) => entry.envelope.record instanceof PoolCommitRecord && entry.envelope.record.key === "writer.new_messages",
  )!.value!.toObject() as { role: string; parts: { type: string; text?: string }[] }[];
  return value.map((message) => [message.role, message.parts.map((part) => part.text ?? part.type).join("|")]);
}

const cutReply = () => [new TextDelta(0, "hel"), new TextDelta(0, "lo"), new Finish({ reason: "stop" })];

test("a recorded cut and honor step replay to null; no model is called", async () => {
  const flow = flowText([], { steerInbox: true, output: "Optional(str)" });
  const uncutLog = runLog();
  const uncut = await run(hooked(new ScriptedModel([cutReply()], CONFIG), (k) => k === 1 && interruptAt(uncutLog)), {
    flow,
    log: uncutLog,
  });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  expect(uncut.result.output).toBeNull();
  expect(honors(uncutLog)).toEqual([interruptSeq(uncutLog)]);
  expect(await newMessages(uncutLog)).toEqual([
    ["assistant", "hel"],
    ["user", INTERRUPTED_MARKER],
  ]);
  for (const kind of ["interrupted", "model_call"]) {
    // After the honor step, or after the cut call and before the honor step.
    const log = runLog();
    await cutRun(seqOf(uncutLog, kind) + 1, hooked(new ScriptedModel([cutReply()], CONFIG), (k) => k === 1 && interruptAt(log)), {
      flow,
      log,
    });
    const model = new ScriptedModel([], CONFIG);
    const resumed = await resume(log, model);
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBeNull();
    expect(model.requests.length).toBe(0);
    expect(honors(log)).toEqual([interruptSeq(log)]);
  }
});

/** `model`, running `hook` as each call's first event is asked for, before the model sends it. */
function atFirstRead(model: ChatModel, hook: () => void): ChatModel {
  return {
    provider: model.provider,
    model: model.model,
    stream(request: Request, options: { signal?: AbortSignal } = {}) {
      const inner = model.stream(request, options);
      async function* events() {
        hook();
        yield* inner;
      }
      return new EventStream(events(), () => inner.close());
    },
  };
}

test("a call cut before its first event records an empty reply and replays to null; no model is called", async () => {
  const flow = flowText([], { steerInbox: true, output: "Optional(str)" });
  const uncutLog = runLog();
  const cutting = (log: string) => atFirstRead(new ScriptedModel([cutReply()], CONFIG), () => interruptAt(log));
  const uncut = await run(cutting(uncutLog), { flow, log: uncutLog });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  expect(uncut.result.output).toBeNull();
  expect(steps(uncutLog).map((step) => [step["kind"], step["finish"] ?? null])).toEqual([
    ["model_call", "interrupted"],
    ["interrupted", null],
  ]);
  expect(Message.fromJSON(steps(uncutLog)[0]!["output"]).parts).toEqual([]);
  expect(await newMessages(uncutLog)).toEqual([["user", INTERRUPTED_MARKER]]);
  for (const kind of ["interrupted", "model_call"]) {
    // After the honor step, or after the cut call and before the honor step.
    const log = runLog();
    await cutRun(seqOf(uncutLog, kind) + 1, cutting(log), { flow, log });
    const model = new ScriptedModel([], CONFIG);
    const resumed = await resume(log, model);
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBeNull();
    expect(model.requests.length).toBe(0);
    expect(honors(log)).toEqual([interruptSeq(log)]);
    expect(await newMessages(log)).toEqual([["user", INTERRUPTED_MARKER]]);
  }
});

test("a cut with no text leaves only the marker in new_messages", async () => {
  const flow = flowText([], { steerInbox: true, output: "Optional(str)" });
  const log = runLog();
  // Interrupted before the first event's text: a later empty-text event can't cut; the second text does.
  const reply = [new TextDelta(0, ""), new TextDelta(0, "x"), new Finish({ reason: "stop" })];
  await run(hooked(new ScriptedModel([reply], CONFIG), (k) => k === 1 && interruptAt(log)), { flow, log });
  expect(await newMessages(log)).toEqual([["user", INTERRUPTED_MARKER]]);
});

test("an interrupt delivered during replay is honored after it, with no live model call", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const checking = new Response({
    message: new Message({
      role: "assistant",
      parts: [new TextPart("Checking."), new ToolCallPart({ id: "call-a", name: "save", arguments: { topic: "a" } })],
    }),
    finishReason: "tool_calls",
  });
  const replies = () => [checking, textReply("done")];
  const tools = { save: topicTool("save") };
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, new ScriptedModel(replies(), CONFIG), { flow, tools });
  const model = new ScriptedModel([], CONFIG);
  const base = serving(model);
  // The agent opens its steer inbox, then builds its model: the interrupt lands before the replay.
  const models = { ...base, createModel: (config: Readonly<Record<string, unknown>>, opts: never) => (interruptAt(log), base.createModel(config, opts)) };
  const resumed = await resume(log, model, { tools, models });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(model.requests.length).toBe(0);
  expect(steps(log).map((step) => step["kind"])).toEqual(["model_call", "tool_result", "interrupted"]);
});

test("a recorded steer read before an interrupt honored on resume stays in new_messages", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const replies = () => [calls(["call-a", "save", "a"]), textReply("done")];
  const tools = { save: topicTool("save") };
  const uncutLog = runLog();
  // The steer is sent while `save` runs, so the next model call reads it.
  const sending = (log: string) => ({ save: topicTool("save", { run: (topic) => (void send(log, "notes", "use B"), `save:${topic}`) }) });
  await run(new ScriptedModel(replies(), CONFIG), { flow, tools: sending(uncutLog), log: uncutLog });
  const log = runLog();
  await cutRun(seqOf(uncutLog, "steer") + 1, new ScriptedModel(replies(), CONFIG), { flow, tools: sending(log), log });
  const model = new ScriptedModel([], CONFIG);
  const base = serving(model);
  const models = { ...base, createModel: (config: Readonly<Record<string, unknown>>, opts: never) => (interruptAt(log), base.createModel(config, opts)) };
  const resumed = await resume(log, model, { tools, models });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(model.requests.length).toBe(0);
  expect(await newMessages(log)).toEqual([
    ["assistant", "tool_call"],
    ["tool", "tool_result"],
    ["user", "use B"],
    ["user", INTERRUPTED_MARKER],
  ]);
});

test("recorded skipped results replay the host interrupt that skipped them", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const reply = () => [calls(["call-a", "save", "a"], ["call-b", "save", "b"])];
  const ran: string[] = [];
  const tools = { save: topicTool("save", { log: ran }) };
  const uncutLog = runLog();
  const uncut = await run(hooked(new ScriptedModel(reply(), CONFIG), (k) => k === Infinity && interruptAt(uncutLog)), {
    flow,
    tools,
    log: uncutLog,
  });
  expect(uncut.result.status).toBe("succeeded");
  expect(await newMessages(uncutLog)).toEqual([
    ["assistant", "tool_call|tool_call"],
    ["tool", "tool_result"],
    ["tool", "tool_result"],
    ["user", INTERRUPTED_MARKER],
  ]);
  for (const cut of [seqOf(uncutLog, "tool_result", 1) + 1, seqOf(uncutLog, "interrupted") + 1]) {
    // After both skipped results (before the honor step), or after the honor step.
    const log = runLog();
    await cutRun(cut, hooked(new ScriptedModel(reply(), CONFIG), (k) => k === Infinity && interruptAt(log)), { flow, tools, log });
    const model = new ScriptedModel([], CONFIG);
    const resumed = await resume(log, model, { tools });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBeNull();
    expect(model.requests.length).toBe(0);
    expect(steps(log).map((step) => step["closed"] ?? step["kind"])).toEqual(["model_call", "skipped", "skipped", "interrupted"]);
    expect(honors(log)).toEqual([interruptSeq(log)]);
  }
  expect(ran).toEqual([]);
});

test("calls a group committed before a host interrupt skipped the rest replay to the uncut run's success", async () => {
  const flow = flowText(["lookup", "save"], { steerInbox: true, output: "Optional(str)" });
  const replies = () => [calls(["call-x", "lookup", "x"], ["call-y", "lookup", "y"], ["call-z", "save", "z"])];
  const options = (log: string) => {
    // The interrupt lands while the safe group runs, so only `save z` is skipped.
    const lookup = topicTool("lookup", { safe: true, run: (topic) => (interruptAt(log), `lookup:${topic}`) });
    return { flow, tools: { lookup, save: topicTool("save") }, log };
  };
  const uncutLog = runLog();
  const uncut = await run(new ScriptedModel(replies(), CONFIG), options(uncutLog));
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  expect(steps(uncut.log).map((step) => step["closed"] ?? null)).toEqual([null, null, null, "skipped", null]);
  // Cut after the honor step, before the node's commit.
  const log = runLog();
  await cutRun(seqOf(uncutLog, "interrupted") + 1, new ScriptedModel(replies(), CONFIG), options(log));
  const resumed = await resume(log, new ScriptedModel([], CONFIG), { tools: options(log).tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // The node's success is the uncut run's.
  expect(comparable(log)).toEqual(comparable(uncutLog));
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

test("with save resumable, a run resumed at every cut point ends as the uncut run, save running at most twice", async () => {
  const replies = () => [
    calls(["call-a", "save", "a"], ["call-b", "save", "b"]),
    calls(["call-x", "lookup", "x"], ["call-y", "lookup", "y"], ["call-z", "save", "z"]),
    textReply("all done"),
  ];
  const flow = flowText(["lookup", "save"]);
  const toolsFor = (ran: string[]) => ({
    lookup: topicTool("lookup", { safe: true, log: ran }),
    save: topicTool("save", { resumable: true, log: ran }),
  });
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools: toolsFor([]) });
  const count = records(uncut.log).length;
  const first = records(uncut.log).findIndex(({ kind }) => kind === "node_started");
  for (let k = first; k < count - 1; k += 1) {
    const ran: string[] = [];
    const log = await cutRun(k, new ScriptedModel(replies(), CONFIG), { flow, tools: toolsFor(ran) });
    const kept = steps(log).filter((step) => step["kind"] === "model_call").length;
    const resumed = await resume(log, new ScriptedModel(replies().slice(kept), CONFIG), { tools: toolsFor(ran) });
    expect(resumed.status, `cut at ${k}: ${resumed.failure?.message}`).toBe("succeeded");
    expect(resumed.output).toBe("all done");
    for (const topic of ["a", "b", "z"]) {
      expect(ran.filter((each) => each === `save ${topic}`).length, `cut at ${k}: ${ran.join(", ")}`).toBeLessThanOrEqual(2);
    }
    expect(ran.filter((each) => each.startsWith("lookup")).length, `cut at ${k}`).toBe(new Set(ran.filter((each) => each.startsWith("lookup"))).size);
    // Every save call ends with the result it gave; none is closed.
    const saves = toolResults(log).filter(({ call_id }) => ["call-a", "call-b", "call-z"].includes(call_id as string));
    expect(saves, `cut at ${k}`).toEqual([
      { call_id: "call-a", output: "save:a", is_error: false },
      { call_id: "call-b", output: "save:b", is_error: false },
      { call_id: "call-z", output: "save:z", is_error: false },
    ]);
  }
});

// ---------- resumable calls sent again ----------

/** The record count that cuts after the reply's `model_call` step and before its first `tool_result`. */
function afterCall(uncut: string): number {
  return seqOf(uncut, "model_call") + 1;
}

test("a crash mid-call sends a resumable call again with the same call id, once, attempt 2", async () => {
  const seen: [string, number][] = [];
  const save = topicTool("save", { resumable: true, run: (t, ctx) => (seen.push([ctx.callId, ctx.attempt]), `saved ${t}`) });
  const model = () => new ScriptedModel([calls(["c1", "save", "a"]), textReply("done")], CONFIG);
  const uncut = await run(model(), { tools: { save } });
  seen.length = 0;
  const log = await cutRun(afterCall(uncut.log), model(), { tools: { save } });
  const result = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools: { save } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("done");
  expect(seen).toEqual([
    ["r/writer/1/0", 1],
    ["r/writer/1/0", 2],
  ]);
  expect(toolResults(log)).toEqual([{ call_id: "c1", output: "saved a", is_error: false }]);
  expect(comparable(log)).toEqual(comparable(uncut.log));
});

test("a stop mid-call sends a resumable call again on the next advance, attempt 2", async () => {
  const seen: [string, number][] = [];
  const entered = Promise.withResolvers<void>();
  const save = topicTool("save", {
    resumable: true,
    run: async (t, ctx) => {
      seen.push([ctx.callId, ctx.attempt]);
      if (ctx.attempt === 1) {
        entered.resolve();
        await new Promise((_resolve, reject) => ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason)));
      }
      return `saved ${t}`;
    },
  });
  const controller = new AbortController();
  const log = runLog();
  const running = run(new ScriptedModel([calls(["c1", "save", "a"]), textReply("done")], CONFIG), {
    tools: { save },
    log,
    signal: controller.signal,
  });
  await entered.promise;
  controller.abort(new Error("host stop"));
  await running.catch(() => undefined);
  expect(kinds(log)).toContain("run_stopped");
  expect(toolResults(log)).toEqual([]);

  const result = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools: { save } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen).toEqual([
    ["r/writer/1/0", 1],
    ["r/writer/1/0", 2],
  ]);
  expect(toolResults(log)).toEqual([{ call_id: "c1", output: "saved a", is_error: false }]);
});

test("a mixed reply re-sends its resumable call and closes the other crashed, in call order", async () => {
  const ran: string[] = [];
  const tools = { note: topicTool("note", { log: ran }), save: topicTool("save", { resumable: true, log: ran }) };
  const flow = flowText(["note", "save"]);
  const replies = () => [calls(["c1", "note", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  // `note a` ran; the crash cut its result, and `save b` never started.
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  expect(ran).toEqual(["note a"]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const result = await resume(log, model, { tools });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(ran).toEqual(["note a", "save b"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" },
    { call_id: "c2", output: "save:b", is_error: false },
  ]);
  expect(sentResults(model, 0).map((part) => [part.callId, part.content])).toEqual([
    ["c1", TOOL_OUTCOME_UNKNOWN],
    ["c2", "save:b"],
  ]);
});

test("re-sent calls run one at a time even when their tool is concurrency-safe", async () => {
  let active = 0;
  let most = 0;
  const save = topicTool("save", {
    safe: true,
    resumable: true,
    run: async (t) => {
      active += 1;
      most = Math.max(most, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      return `saved ${t}`;
    },
  });
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools: { save } });
  // Live, the two calls run together.
  expect(most).toBe(2);
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { tools: { save } });
  most = 0;
  const result = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools: { save } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(most).toBe(1);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "saved a", is_error: false },
    { call_id: "c2", output: "saved b", is_error: false },
  ]);
});

test("a pending interrupt is honored after the re-sent call, which still runs", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = { save: topicTool("save", { resumable: true, log: ran }) };
  const replies = () => [calls(["c1", "save", "a"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  const model = new ScriptedModel([], CONFIG);
  const base = serving(model);
  // The agent opens its steer inbox, then builds its model: the interrupt lands before the replay.
  const models = { ...base, createModel: (config: Readonly<Record<string, unknown>>, opts: never) => (interruptAt(log), base.createModel(config, opts)) };
  const resumed = await resume(log, model, { tools, models });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(model.requests.length).toBe(0);
  expect(ran).toEqual(["save a", "save a"]);
  expect(steps(log).map((step) => step["kind"])).toEqual(["model_call", "tool_result", "interrupted"]);
  expect(toolResults(log)).toEqual([{ call_id: "c1", output: "save:a", is_error: false }]);
});

test("a resumable tool's call whose argument check reported an error gets the same error again, not a closer", async () => {
  let checks = 0;
  const save = new FunctionTool({
    name: "save",
    description: "Tool save.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    parametersValidator: {
      "~standard": {
        version: 1,
        vendor: "test",
        validate: () => {
          checks += 1;
          throw new ReportedToolError("no such topic");
        },
      },
    },
    execute: () => "never",
    resumable: true,
  });
  const uncut = await run(saveModel(), { tools: { save } });
  expect(uncut.result.status, uncut.result.failure?.message).toBe("succeeded");
  const log = await cutRun(afterCall(uncut.log), saveModel(), { tools: { save } });
  checks = 0;
  const result = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools: { save } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(checks).toBe(1);
  expect(toolResults(log)).toEqual(toolResults(uncut.log));
  expect(toolResults(log)[0]).not.toHaveProperty("closed");
  expect(comparable(log)).toEqual(comparable(uncut.log));
});

/** A model whose agent's interrupt on `notes` lands at the resume, before the replay. */
function interruptedAtResume(log: string, model: ScriptedModel): ModelAccess {
  const base = serving(model);
  // The agent opens its steer inbox, then builds its model: the interrupt lands before the replay.
  return { ...base, createModel: (config: Readonly<Record<string, unknown>>, opts: never) => (interruptAt(log), base.createModel(config, opts)) };
}

test("with an interrupt pending, a resume sends again only the cut-off group, and skips the later calls", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = { save: topicTool("save", { resumable: true, log: ran }) };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  // `save a` ran; the crash cut its result, and `save b`, a later group, never started.
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  expect(ran).toEqual(["save a"]);
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(model.requests.length).toBe(0);
  expect(ran).toEqual(["save a", "save a"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
  expect(steps(log).map((step) => step["kind"])).toEqual(["model_call", "tool_result", "tool_result", "interrupted"]);
});

test("with an interrupt pending, the cut-off group's safe calls are all sent again before the skip", async () => {
  const flow = flowText(["look", "save"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = {
    look: topicTool("look", { safe: true, resumable: true, log: ran }),
    save: topicTool("save", { resumable: true, log: ran }),
  };
  const replies = () => [calls(["c1", "look", "a"], ["c2", "look", "b"], ["c3", "save", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  // Both `look` calls ran together; the crash cut their results, and `save c` never started.
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  expect(ran).toEqual(["look a", "look b"]);
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(ran).toEqual(["look a", "look b", "look a", "look b"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "look:a", is_error: false },
    { call_id: "c2", output: "look:b", is_error: false },
    { call_id: "c3", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
});

/**
 * A run of the reply [c1 save (unsafe), c2 look (safe), c3 look (safe)], whose groups are [c1] and
 * [c2, c3], cut twice: the first start during c1, then a resume, which sends c1 again and records
 * it, during c2. `look` is resumable when `lookResumable`. Returns the log and what each start ran.
 */
async function cutTwice(lookResumable: boolean, ran: string[]) {
  const flow = flowText(["save", "look"], { steerInbox: true, output: "Optional(str)" });
  const tools = {
    save: topicTool("save", { resumable: true, log: ran }),
    look: topicTool("look", { safe: true, resumable: lookResumable, log: ran }),
  };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "look", "b"], ["c3", "look", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  // The resume writes run_resumed, node_started and c1's result, then is cut on c2's record.
  const before = records(log).length;
  await crashAfter(before + 3, () => resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools }));
  expect(kinds(log, before)).toEqual(["run_resumed", "node_started", "tool_result 2"]);
  return { log, tools };
}

test.each([true, false])(
  "with an interrupt pending after two cuts, only the call the last start was sending may have started (look resumable %p)",
  async (lookResumable) => {
    const ran: string[] = [];
    const { log, tools } = await cutTwice(lookResumable, ran);
    // The first start ran `save a`; the resume sent it again, then started `look b` alone.
    expect(ran).toEqual(lookResumable ? ["save a", "save a", "look b"] : ["save a", "save a"]);
    ran.length = 0;
    const model = new ScriptedModel([], CONFIG);
    const resumed = await resume(log, model, { tools, models: interruptedAtResume(log, model) });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBeNull();
    expect(model.requests.length).toBe(0);
    // `look c` never started: no start sent it, so the interrupt skips it. A `look b` that isn't
    // resumable never ran either, since a resume only closes such a call.
    expect(ran).toEqual(lookResumable ? ["look b"] : []);
    expect(toolResults(log)).toEqual([
      { call_id: "c1", output: "save:a", is_error: false },
      lookResumable
        ? { call_id: "c2", output: "look:b", is_error: false }
        : { call_id: "c2", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
      { call_id: "c3", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
    ]);
  },
);

test("an interrupt that arrives while a resume sends a call again still skips the calls no start sent", async () => {
  const flow = flowText(["save", "look"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  let armed = false;
  let target = "";
  const tools = {
    save: topicTool("save", { resumable: true, log: ran }),
    look: topicTool("look", {
      safe: true,
      resumable: true,
      log: ran,
      run: (topic) => {
        // The interrupt lands while `look b` is sent again, after the replay placed `save a`.
        if (armed && topic === "b") {
          armed = false;
          interruptAt(target);
        }
        return `look:${topic}`;
      },
    }),
  };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "look", "b"], ["c3", "look", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  target = log;
  const before = records(log).length;
  await crashAfter(before + 3, () => resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools }));
  ran.length = 0;
  armed = true;
  const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(ran).toEqual(["look b"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: "look:b", is_error: false },
    { call_id: "c3", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
});

test("a call that pended in the reply's own start and was closed by a later one places the group as that start ran it", async () => {
  const flow = flowText(["hold", "look"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = {
    hold: topicTool("hold", { log: ran, run: () => pending() }),
    look: topicTool("look", { safe: true, resumable: true, log: ran }),
  };
  // Groups [c1 hold], which pends, then [c2 look, c3 look].
  const replies = () => [calls(["c1", "hold", "a"], ["c2", "look", "b"], ["c3", "look", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  ran.length = 0;
  // Cut after c1's pending step: c2 and c3 ran together, and neither result was written.
  const log = await cutRun(seqOf(uncut.log, "pending") + 1, new ScriptedModel(replies(), CONFIG), { flow, tools });
  expect(ran).toEqual(["hold a", "look b", "look c"]);
  // The resume closes c1 crashed, then is cut while it sends c2 again.
  const before = records(log).length;
  await crashAfter(before + 3, () => resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools }));
  ran.length = 0;
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  // Both look calls ran in the first start, so both may have started: neither is skipped.
  expect(ran).toEqual(["look b", "look c"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" },
    { call_id: "c2", output: "look:b", is_error: false },
    { call_id: "c3", output: "look:c", is_error: false },
  ]);
});

test("a call a start skipped for a pending interrupt, cut before the skip was written, is skipped again, not sent", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = { save: topicTool("save", { resumable: true, log: ran }) };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  // The resume takes the interrupt, sends `save a` again and is cut writing c2's skip.
  const before = records(log).length;
  const first = new ScriptedModel([], CONFIG);
  await crashAfter(before + 4, () => resume(log, first, { tools, models: interruptedAtResume(log, first) }));
  expect(kinds(log, before)).toEqual(["run_resumed", "node_started", "interrupted", "tool_result 2"]);
  ran.length = 0;
  const resumed = await resume(log, new ScriptedModel([], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(ran).toEqual([]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
});

test("skips a cut left half written are finished by the next start", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const tools = { save: topicTool("save", { resumable: true }) };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"], ["c3", "save", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  // The resume takes the interrupt, sends `save a` again, skips c2 and is cut skipping c3.
  const before = records(log).length;
  const first = new ScriptedModel([], CONFIG);
  await crashAfter(before + 5, () => resume(log, first, { tools, models: interruptedAtResume(log, first) }));
  expect(kinds(log, before)).toEqual(["run_resumed", "node_started", "interrupted", "tool_result 2", "tool_result 3"]);
  const resumed = await resume(log, new ScriptedModel([], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
    { call_id: "c3", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
});

test("a call a later start pended places the next call where it pended, not where it was closed", async () => {
  const flow = flowText(["save", "hold"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  const tools = {
    save: topicTool("save", { resumable: true, log: ran }),
    hold: topicTool("hold", { resumable: true, log: ran, run: () => pending() }),
  };
  const replies = () => [calls(["c0", "save", "a"], ["c1", "hold", "b"], ["c2", "save", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools });
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools });
  // The resume writes c0's result and c1's pending step, runs `save c`, and is cut on its result.
  const before = records(log).length;
  ran.length = 0;
  await crashAfter(before + 4, () => resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools }));
  expect(kinds(log, before)).toEqual(["run_resumed", "node_started", "tool_result 2", "pending 3"]);
  expect(ran).toEqual(["save a", "hold b", "save c"]);
  ran.length = 0;
  // The next resume closes c1, after the interrupt; `save c` ran before it, so it is sent again.
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBeNull();
  expect(ran).toEqual(["save c"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c0", output: "save:a", is_error: false },
    { call_id: "c1", output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" },
    { call_id: "c2", output: "save:c", is_error: false },
  ]);
});

test("without an interrupt after two cuts, the resume sends the rest in order", async () => {
  const ran: string[] = [];
  const { log, tools } = await cutTwice(true, ran);
  ran.length = 0;
  const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
  expect(ran).toEqual(["look b", "look c"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: "look:b", is_error: false },
    { call_id: "c3", output: "look:c", is_error: false },
  ]);
});

test("with an interrupt pending, a call past the cut-off group is skipped before its arguments are checked", async () => {
  const flow = flowText(["save"], { steerInbox: true, output: "Optional(str)" });
  const checked: string[] = [];
  const ran: string[] = [];
  const save = new FunctionTool({
    name: "save",
    description: "Tool save.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    parametersValidator: {
      "~standard": {
        version: 1,
        vendor: "test",
        validate: (value) => (checked.push(String((value as { topic: unknown }).topic)), { value }),
      },
    },
    execute: (args) => (ran.push(String(args["topic"])), `save:${String(args["topic"])}`),
    resumable: true,
  });
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools: { save } });
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { flow, tools: { save } });
  checked.length = 0;
  ran.length = 0;
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resume(log, model, { tools: { save }, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(checked).toEqual(["a"]);
  expect(ran).toEqual(["a"]);
});

test("a skip passes over a call that has a recorded result, even when the call's group changed on resume", async () => {
  const flow = flowText(["look"], { steerInbox: true, output: "Optional(str)" });
  const ran: string[] = [];
  // c3 finishes first, so its result is written while c1 and c2 still run.
  const look = (safe: boolean) =>
    topicTool("look", {
      safe,
      resumable: true,
      log: ran,
      run: async (topic) => {
        if (topic !== "c") await new Promise((resolve) => setTimeout(resolve, 5));
        return `look:${topic}`;
      },
    });
  const replies = () => [calls(["c1", "look", "a"], ["c2", "look", "b"], ["c3", "look", "c"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { flow, tools: { look: look(true) } });
  expect(toolResults(uncut.log).map(({ call_id }) => call_id)).toEqual(["c3", "c1", "c2"]);
  const log = await cutRun(seqOf(uncut.log, "tool_result") + 1, new ScriptedModel(replies(), CONFIG), { flow, tools: { look: look(true) } });
  ran.length = 0;
  const model = new ScriptedModel([], CONFIG);
  // On resume `look` no longer runs beside others, so c1 is a group of its own and c2 is skipped.
  const resumed = await resume(log, model, { tools: { look: look(false) }, models: interruptedAtResume(log, model) });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(ran).toEqual(["look a"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c3", output: "look:c", is_error: false },
    { call_id: "c1", output: "look:a", is_error: false },
    { call_id: "c2", output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" },
  ]);
});

test("without an interrupt, a resume sends the cut-off call again and then the later group's, in order", async () => {
  const ran: string[] = [];
  const tools = { save: topicTool("save", { resumable: true, log: ran }) };
  const replies = () => [calls(["c1", "save", "a"], ["c2", "save", "b"]), textReply("done")];
  const uncut = await run(new ScriptedModel(replies(), CONFIG), { tools });
  ran.length = 0;
  const log = await cutRun(afterCall(uncut.log), new ScriptedModel(replies(), CONFIG), { tools });
  const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(ran).toEqual(["save a", "save a", "save b"]);
  expect(toolResults(log)).toEqual([
    { call_id: "c1", output: "save:a", is_error: false },
    { call_id: "c2", output: "save:b", is_error: false },
  ]);
});

test("a re-sent call emits ToolCallStarted and ToolCallFinished", async () => {
  const tools = { save: topicTool("save", { resumable: true }) };
  const uncut = await run(saveModel(), { tools });
  const log = await cutRun(afterCall(uncut.log), saveModel(), { tools });
  const events: RunEvent[] = [];
  const result = await resume(log, new ScriptedModel([textReply("done")], CONFIG), { tools, events });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const brackets = events.filter((event) => event instanceof ToolCallStarted || event instanceof ToolCallFinished);
  expect(brackets.map((event) => [event.constructor.name, (event as ToolCallStarted).toolCallId])).toEqual([
    ["ToolCallStarted", "call-a"],
    ["ToolCallFinished", "call-a"],
  ]);
  expect((brackets[1] as ToolCallFinished).result).toBe("save:a");
});
