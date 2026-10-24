/**
 * Late steers: a steer queued when the model gives its final answer gets one more model call;
 * every reply counts toward max_tool_iterations, and at the cap the answer stands and the steers
 * stay queued. TS-only: Python yian has no steer inbox.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";

import { type ChatModel, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext, type ModelTextInteraction } from "../../src/memory/contracts.ts";
import { type AgentRunContext, DEFAULT_SYSTEM, requestSha256, ToolIterationLimitError } from "../../src/nodes/agent/context.ts";
import { runAgent, stepsMaterial } from "../../src/nodes/agent/loop.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output, SteerInbox } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { withWarnings } from "../../src/platform.ts";
import { send } from "../../src/run.ts";
import { NodeStepRecord, PoolCommitRecord, readRunLog } from "../../src/runtime/log.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { callReply, crashAfter, NO_ENV, resumeWith, runWith, serving, textReply } from "./run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };
const SYSTEM = new Message({ role: "system", parts: [new TextPart(DEFAULT_SYSTEM)] });

/** A steer inbox double: `queue` is what a read takes; `reads` logs each read's values, `null` for none. */
function inboxDouble(queue: string[] = []) {
  const reads: (readonly unknown[] | null)[] = [];
  let seq = 100;
  const inbox: SteerInbox = {
    inbox: "notes",
    interrupt: new AbortController().signal,
    interruptedBy: () => null,
    queued: () => queue.length > 0,
    read: () => {
      if (queue.length === 0) {
        reads.push(null);
        return null;
      }
      const values = queue.splice(0);
      reads.push(values);
      return { n: 0, from: values.map(() => seq++), values };
    },
  };
  return { inbox, queue, reads };
}

/**
 * A scripted model that runs `during(i)` while it answers its `i`th request (from 1), as a steer
 * sent mid-reply; the reply's events follow once what `during` returns settles.
 */
function hooked(model: ScriptedModel, during: (request: number) => unknown): ChatModel & { requests: Request[] } {
  return {
    provider: model.provider,
    model: model.model,
    requests: model.requests,
    stream(request: Request, options?: { signal?: AbortSignal }): EventStream {
      const inner = model.stream(request, options);
      const settled = during(model.requests.length);
      return new EventStream(
        (async function* () {
          await settled;
          yield* inner;
        })(),
        () => inner.close(),
      );
    },
  };
}

/** A lookup tool returning `found <topic>`. */
const lookup = () =>
  new FunctionTool({
    name: "lookup",
    description: "Look up a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `found ${String(args["topic"])}`,
  });

/** A direct loop context: agent `chat` prompted "go", granted `lookup` with `withLookup`. */
function loopContext(model: ChatModel, extra: Partial<AgentRunContext> & { withLookup?: boolean } = {}): AgentRunContext {
  const { withLookup = false, ...rest } = extra;
  return {
    nodeId: "chat",
    messages: [userMessage("go")],
    promptText: "go",
    tools: withLookup ? ["lookup"] : [],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: withLookup ? new ToolSet({ lookup: lookup() }) : null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
    ...rest,
  };
}

/** Each message of `request` as `[role, text]`. */
const shown = (request: Request) =>
  request.messages.map((message) => [message.role, message.parts.map((part) => (part as TextPart).text ?? "").join("|")]);

/** Run the loop and return its output value. */
const output = async (ctx: AgentRunContext) => ((await runAgent(ctx)) as Output).value;

test("a steer queued while the final reply streams gets one more model call", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) => {
    if (i === 1) queue.push("use 2024");
  });
  expect(await output(loopContext(model, { steerInbox: inbox, withLookup: true }))).toBe("final");
  expect(model.requests).toHaveLength(2);
  expect(shown(model.requests[1]!).slice(-2)).toEqual([
    ["assistant", "draft"],
    ["user", "use 2024"],
  ]);
  expect(reads).toEqual([null, ["use 2024"]]);
});

test("no steer queued at the final answer: one model call, no read after it", async () => {
  const { inbox, reads } = inboxDouble();
  const model = new ScriptedModel([textReply("final")], CONFIG);
  expect(await output(loopContext(model, { steerInbox: inbox, withLookup: true }))).toBe("final");
  expect(model.requests).toHaveLength(1);
  expect(reads).toEqual([null]);
});

test("a late steer on a tool-free agent works the same", async () => {
  const { inbox, queue } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) => {
    if (i === 1) queue.push("shorter");
  });
  expect(await output(loopContext(model, { steerInbox: inbox }))).toBe("final");
  expect(shown(model.requests[1]!).at(-1)).toEqual(["user", "shorter"]);
});

test("replies after a late steer count toward the cap", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("one"), textReply("two")], CONFIG), (i) => {
    queue.push(`steer ${i}`);
  });
  expect(await output(loopContext(model, { steerInbox: inbox, maxToolIterations: 2 }))).toBe("two");
  expect(model.requests).toHaveLength(2);
  expect(queue).toEqual(["steer 2"]);
  expect(reads).toEqual([null, ["steer 1"]]);
});

test("at the cap the answer stands and nothing is read", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("one")], CONFIG), () => queue.push("late"));
  expect(await output(loopContext(model, { steerInbox: inbox, maxToolIterations: 1 }))).toBe("one");
  expect(queue).toEqual(["late"]);
  expect(reads).toEqual([null]);
});

test("-1 is unbounded", async () => {
  const { inbox, queue } = inboxDouble();
  const replies = ["a", "b", "c", "d"].map((text) => textReply(text));
  const model = hooked(new ScriptedModel(replies, CONFIG), (i) => {
    if (i <= 3) queue.push(`steer ${i}`);
  });
  expect(await output(loopContext(model, { steerInbox: inbox }))).toBe("d");
  expect(model.requests).toHaveLength(4);
});

test("a reply with tool calls at the cap still fails", async () => {
  const { inbox } = inboxDouble();
  const model = new ScriptedModel([callReply(["lookup", { topic: "a" }, "c1"])], CONFIG);
  const failure = await runAgent(loopContext(model, { steerInbox: inbox, withLookup: true, maxToolIterations: 1 })).then(
    () => null,
    (error: unknown) => error,
  );
  expect(failure).toBeInstanceOf(ToolIterationLimitError);
});

test("an interrupt that lands during the final reply beats a queued steer: the answer stands, the steer stays", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const interrupt = new AbortController();
  const model = hooked(new ScriptedModel([textReply("final")], CONFIG), () => {
    queue.push("late");
    interrupt.abort();
  });
  const steering = { interruptSignal: () => interrupt.signal };
  expect(await output(loopContext(model, { steerInbox: inbox, steering }))).toBe("final");
  expect(model.requests).toHaveLength(1);
  expect(queue).toEqual(["late"]);
  expect(reads).toEqual([null]);
});

test("stepsMaterial rebuilds a late-steered occurrence with a matching digest", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) => {
    if (i === 1) queue.push("use 2024");
  });
  const recorded: ValueRecord[] = [];
  const steps = new NodeSteps(1, [], (step) => recorded.push(step));
  expect(await output(loopContext(model, { steerInbox: inbox, steps }))).toBe("final");
  // The engine writes the steer step a read makes; here it goes between the two calls by hand.
  const occurrence = [recorded[0]!, { kind: "steer", inbox: "notes", from: [100n], values: reads[1]! } as unknown as ValueRecord, recorded[1]!];
  const options = { messages: [SYSTEM, userMessage("go")], tools: [], structured: async () => null, baseSystemPrompt: DEFAULT_SYSTEM };
  const { interactions, digestOk } = await stepsMaterial(occurrence, options);
  expect(digestOk).toBe(true);
  expect(interactions.map((each) => (each as ModelTextInteraction).text)).toEqual(["draft", "final"]);
  expect((await stepsMaterial([recorded[0]!, recorded[1]!], options)).digestOk).toBe(false);
});

// --- through the engine, and across a crash --- //

const root = mkdtempSync(join(tmpdir(), "yian-late-steer-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

/** A fresh run folder's log path. */
const freshLog = () => join(mkdtempSync(join(root, "run-")), "log.jsonl");

/** One tool-free agent `chat` reading steer inbox `notes`. */
const FLOW = `
id: late
name: late
llm_config: {provider: anthropic, model: claude-test}
input:
  notes: !Inbox(str)
nodes:
  chat:
    kind: agent
    steer_inbox: \${input.notes}
    system_prompt: Help.
    prompt: Help the user.
    output: str
output: \${chat.output}
`;

/** `NO_ENV` with `YIAN_AGENT_MAX_TOOL_ITERATIONS` set to `cap`. */
const capped = (cap: number) => (name: string) => (name === "YIAN_AGENT_MAX_TOOL_ITERATIONS" ? String(cap) : NO_ENV(name));

/** Each node step of the log at `log`: a steer as its body, any other as its kind. */
const steps = (log: string) =>
  readRunLog(log).flatMap(({ record }) =>
    record instanceof NodeStepRecord ? [record.step["kind"] === "steer" ? record.step : String(record.step["kind"])] : [],
  );

/** The seqs of the sends to `notes` in the log at `log`. */
const sendSeqs = (log: string) =>
  readRunLog(log).flatMap(({ seq, record }) => (record instanceof PoolCommitRecord && record.key === "__start__.notes" ? [seq] : []));

/** A model answering "draft" then "final", sending "and cite" to `log`'s `notes` during the first reply. */
const sendingModel = (log: string) =>
  hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) =>
    i === 1 ? send(log, "notes", "and cite") : undefined,
  );

test("through the engine: a send during the final reply writes a steer step and a second model_call", async () => {
  const log = freshLog();
  const model = sendingModel(log);
  const result = await runWith(await loadFlow(FLOW), {}, null, model, { log });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("final");
  const [seq] = sendSeqs(log);
  expect(steps(log)).toEqual(["model_call", { kind: "steer", inbox: "notes", from: [BigInt(seq!)] }, "model_call"]);
  expect(shown(model.requests[1]!).slice(-2)).toEqual([
    ["assistant", "draft"],
    ["user", "and cite"],
  ]);
  expect(result.unread).toEqual(new Map());
});

test("at the cap through the engine the event stays unread and no steer step is written", async () => {
  const log = freshLog();
  const result = await runWith(await loadFlow(FLOW), {}, null, sendingModel(log), { log, env: capped(1) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("draft");
  expect(steps(log)).toEqual(["model_call"]);
  expect(result.unread).toEqual(new Map([["notes", sendSeqs(log)]]));
});

/** The seq of the `i`th (from 0) node step of `kind` in the log at `log`. */
const stepSeq = (log: string, kind: string, i = 0) =>
  readRunLog(log).filter(({ record }) => record instanceof NodeStepRecord && record.step["kind"] === kind)[i]!.seq;

test("a crash between the late steer and its model call replays to the same point", async () => {
  const uncut = freshLog();
  const uncutModel = sendingModel(uncut);
  await runWith(await loadFlow(FLOW), {}, null, uncutModel, { log: uncut });
  const log = freshLog();
  await crashAfter(stepSeq(uncut, "steer") + 1, async () => runWith(await loadFlow(FLOW), {}, null, sendingModel(log), { log }));
  expect(steps(log).at(-1)).toMatchObject({ kind: "steer" });
  const model = new ScriptedModel([textReply("final")], CONFIG);
  const resumed = await resumeWith(await loadFlow(FLOW), log, null, model);
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("final");
  expect(steps(log).filter((step) => typeof step !== "string")).toHaveLength(1);
  expect(model.requests.map(shown)).toEqual([shown(uncutModel.requests[1]!)]);
  const digests = readRunLog(log).flatMap(({ record }) =>
    record instanceof NodeStepRecord && record.step["kind"] === "model_call" ? [record.step["request_sha256"]] : [],
  );
  expect(digests[1]).toBe(requestSha256(model.requests[0]!));
});

test("a crash after the final reply, before any steer: a send while down yields a late steer on resume", async () => {
  const uncut = freshLog();
  await runWith(await loadFlow(FLOW), {}, null, new ScriptedModel([textReply("draft")], CONFIG), { log: uncut });
  const log = freshLog();
  await crashAfter(stepSeq(uncut, "model_call") + 1, async () =>
    runWith(await loadFlow(FLOW), {}, null, new ScriptedModel([textReply("draft")], CONFIG), { log }),
  );
  const seq = await send(log, "notes", "one more thing", { env: NO_ENV });
  const model = new ScriptedModel([textReply("final")], CONFIG);
  const resumed = await resumeWith(await loadFlow(FLOW), log, null, model);
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("final");
  expect(steps(log)).toEqual(["model_call", { kind: "steer", inbox: "notes", from: [BigInt(seq)] }, "model_call"]);
  expect(shown(model.requests[0]!).slice(-2)).toEqual([
    ["assistant", "draft"],
    ["user", "one more thing"],
  ]);
});

test("a crash after the late model_call: replay takes it with a matching digest", async () => {
  const uncut = freshLog();
  await runWith(await loadFlow(FLOW), {}, null, sendingModel(uncut), { log: uncut });
  const log = freshLog();
  await crashAfter(stepSeq(uncut, "model_call", 1) + 1, async () => runWith(await loadFlow(FLOW), {}, null, sendingModel(log), { log }));
  const model = new ScriptedModel([], CONFIG);
  const resumed = await resumeWith(await loadFlow(FLOW), log, null, model);
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("final");
  expect(model.requests).toEqual([]);
  expect(steps(log).filter((step) => step === "model_call")).toHaveLength(2);
});

test("a loaded render whose field is null at run time falls back for that event; the steer step stands", async () => {
  const flow = `
typedefs:
  Note:
    text: Optional(str)
id: render
name: render
llm_config: {provider: anthropic, model: claude-test}
input:
  notes: !Inbox(Note)
nodes:
  chat:
    kind: agent
    steer_inbox:
      from: \${input.notes}
      render: |-
        Note: \${event.text}
    system_prompt: Help.
    prompt: Help the user.
    output: str
output: \${chat.output}
`;
  const log = freshLog();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), async (i) => {
    if (i !== 1) return;
    await send(log, "notes", { text: "use 2024" });
    await send(log, "notes", { text: null });
  });
  const warned: string[] = [];
  const result = await withWarnings(
    (message) => warned.push(message),
    async () => runWith(await loadFlow(flow), {}, null, model, { log }),
  );
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(model.requests[1]!.messages.at(-1)!.parts.map((part) => (part as TextPart).text)).toEqual([
    "Note: use 2024",
    '{"text":null}',
  ]);
  expect(steps(log)).toEqual(["model_call", { kind: "steer", inbox: "notes", from: sendSeqs(log).map(BigInt) }, "model_call"]);
  expect(warned.filter((message) => message.includes("didn't render"))).toHaveLength(1);
});
