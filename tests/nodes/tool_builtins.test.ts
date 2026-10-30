/**
 * The TOOL built-ins at run time: `take` and `sleep` return the wait the engine decides, a host
 * tool or `ask` that returns `pending()` records its call and waits, a held call is reused, and
 * `waitResultStep` shapes a host's answer.
 */

import { describe, expect, test } from "bun:test";

import { Output, Pending } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { type CallView, NodeSteps, ValueContract } from "../../src/nodes/contracts.ts";
import { QuestionSpecError } from "../../src/nodes/tool/questions.ts";
import { CompletedError, ToolNode, ToolTimeoutError } from "../../src/nodes/tool/node.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { pending } from "../../src/tools/context.ts";
import { ToolCheckError } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type AskRequest, ToolSet } from "../../src/tools/registry.ts";
import { eventBytes } from "../../src/typesys/json.ts";
import { deadlineUtc, formatRecordTime } from "../../src/typesys/temporal.ts";
import { InboxRef, Type, TypeCheckError, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";

/** Build a node's caps: recorded steps, `calls`, `tools`, and a virtual clock at 2026-10-23T00:00Z. */
function capsWith(calls: CallView[] = [], tools = new ToolSet({}).forRun("r1"), maxEventBytes?: number) {
  const recorded: ValueRecord[] = [];
  const steps = new NodeSteps(1, [], (step) => (recorded.push(step), recorded.length), calls, maxEventBytes);
  return { recorded, caps: { steps, tools, scheduler: virtualScheduler(Date.UTC(2026, 9, 23)) } };
}

/** Return the error `promise` rejects with; fail if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Expect `promise` to reject with a `contract.input` `BindingError` whose message is `message`. */
async function expectContractInput(promise: Promise<unknown>, message: string): Promise<void> {
  const error = await rejection(promise);
  expect(error).toBeInstanceOf(BindingError);
  expect((error as BindingError).failureCode).toBe("contract.input");
  expect((error as BindingError).message).toBe(message);
}

describe("temporal", () => {
  test("deadlineUtc reads a date at midnight UTC and normalises an offset", () => {
    expect(deadlineUtc("2026-10-24")).toBe(Date.UTC(2026, 9, 24));
    expect(deadlineUtc("2026-10-24T09:30+07:00")).toBe(Date.UTC(2026, 9, 24, 2, 30));
  });

  test("deadlineUtc refuses an instant before year 1", () => {
    expect(() => deadlineUtc("0001-01-01T00:00+01:00")).toThrow(new RangeError("date value out of range"));
  });

  test("formatRecordTime writes a record time and refuses one out of range", () => {
    expect(formatRecordTime(Date.UTC(2026, 9, 24, 9))).toBe("2026-10-24T09:00:00.000Z");
    expect(() => formatRecordTime(Number.NaN)).toThrow(
      new RangeError("run-log clock time NaN is outside the years 0000 to 9999"),
    );
    expect(() => formatRecordTime(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});

describe("take and sleep", () => {
  const take = new ToolNode("heard", { toolId: "take" });
  const sleep = new ToolNode("nap", { toolId: "sleep" });

  test("take returns a take request; the engine decides", async () => {
    const { caps } = capsWith();
    expect(await take.forward({ from: new InboxRef("m"), take: "all" }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "all", timeoutSeconds: null } }),
    );
  });

  test("take reads from as a handle or its {inbox} form, mode one by default, timeout int or text", async () => {
    const { caps } = capsWith();
    expect(await take.forward({ from: { inbox: "m" } }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: null } }),
    );
    expect(await take.forward({ from: new InboxRef("m"), take: "one", timeout: 1800n }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: 1800 } }),
    );
    expect(await take.forward({ from: new InboxRef("m"), timeout: "30m" }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: 1800 } }),
    );
  });

  test("take refuses a bad mode, from or timeout as contract.input", async () => {
    const { caps } = capsWith();
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), take: "some" }, caps),
      'take `take` must be one or all, got "some"',
    );
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), timeout: "0s" }, caps),
      'take `timeout` "0s": a duration is a positive integer and a unit (90s | 30m | 24h | 2d)',
    );
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), timeout: 0n }, caps),
      "take `timeout` must be a duration, got 0",
    );
    await expectContractInput(take.forward({ from: "m" }, caps), 'take `from` must be an inbox, got "m"');
    await expectContractInput(take.forward({}, caps), "take `from` must be an inbox, got null");
  });

  test("a take timeout or sleep for whose deadline is past the year 9999 is contract.input", async () => {
    const { caps } = capsWith();
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), timeout: "3000000d" }, caps),
      'take `timeout` "3000000d": the deadline falls after the year 9999',
    );
    await expectContractInput(
      sleep.forward({ for: "3000000d" }, caps),
      'sleep `for` "3000000d": the deadline falls after the year 9999',
    );
  });

  test("take and sleep write no step and call no tool", async () => {
    const { recorded, caps } = capsWith();
    await take.forward({ from: new InboxRef("m") }, caps);
    await sleep.forward({ for: "1s" }, caps);
    expect(recorded).toEqual([]);
  });

  test("sleep for and until", async () => {
    const { caps } = capsWith();
    const slept = (wait: { until: string | null; forSeconds: number | null }) =>
      new Pending([], { wait: { tool: "sleep", ...wait } });
    expect(await sleep.forward({ for: "90s" }, caps)).toEqual(slept({ until: null, forSeconds: 90 }));
    expect(await sleep.forward({ for: 90n, until: null }, caps)).toEqual(slept({ until: null, forSeconds: 90 }));
    expect(await sleep.forward({ until: "2026-10-24" }, caps)).toEqual(
      slept({ until: "2026-10-24T00:00:00.000Z", forSeconds: null }),
    );
    expect(await sleep.forward({ until: "2026-10-24T09:30+07:00" }, caps)).toEqual(
      slept({ until: "2026-10-24T02:30:00.000Z", forSeconds: null }),
    );
    // A deadline already past is still a request; the engine commits it at once.
    expect(await sleep.forward({ until: "2000-01-01" }, caps)).toEqual(
      slept({ until: "2000-01-01T00:00:00.000Z", forSeconds: null }),
    );
  });

  test("sleep needs exactly one of until and for; a bad until is contract.input", async () => {
    const { caps } = capsWith();
    const one = "sleep takes exactly one of `until` and `for`";
    await expectContractInput(sleep.forward({}, caps), one);
    await expectContractInput(sleep.forward({ until: "2026-10-24", for: "1s" }, caps), one);
    await expectContractInput(
      sleep.forward({ until: "2026-13-01" }, caps),
      'sleep `until` "2026-13-01": not a valid date: month must be in 1..12',
    );
    await expectContractInput(
      sleep.forward({ until: 5n }, caps),
      "sleep `until` did not resolve to a date/datetime: 5",
    );
  });

  test("a held take or sleep call still returns its request (the engine reuses the call)", async () => {
    const held: CallView = {
      callId: "r1/heard",
      tool: "take",
      resultKey: "r1/heard",
      state: "answered",
      closedReason: null,
      result: null,
      deadline: null,
    };
    const { caps } = capsWith([held]);
    expect(await take.forward({ from: new InboxRef("m") }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: null } }),
    );
  });
});

/** A host tool `codex_send` taking `text`, whose handler pushes each call id and returns `answer()`. */
function sendTool(ids: string[], answer: () => unknown = () => pending()): FunctionTool {
  return new FunctionTool({
    name: "codex_send",
    description: "",
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } } },
    execute: (_args, ctx) => (ids.push(ctx.callId), answer()),
  });
}

/** One held call of node `send` (or `node`), open unless `fields` says otherwise. */
function heldCall(fields: Partial<CallView> = {}): CallView {
  const callId = fields.callId ?? "r1/send";
  return { callId, tool: "codex_send", resultKey: callId, state: "open", closedReason: null, result: null, deadline: null, ...fields };
}

const APPROVAL = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["decision", Type.scalar(ValueKind.STRING)]]),
  required: new Set(["decision"]),
});

describe("host calls", () => {
  test("a host tool that returns pending() records the call and waits", async () => {
    const ids: string[] = [];
    const { recorded, caps } = capsWith([], new ToolSet({ codex_send: sendTool(ids) }).forRun("r1"));
    expect(await new ToolNode("send", { toolId: "codex_send" }).forward({ text: "hi" }, caps)).toEqual(
      new Pending(["r1/send"]),
    );
    expect(recorded).toEqual([{ kind: "pending", call_id: "r1/send", tool: "codex_send", arguments: { text: "hi" } }]);
    expect(ids).toEqual(["r1/send"]);
  });

  test("a re-run with no held call calls the handler again with the same call id", async () => {
    const ids: string[] = [];
    const tools = new ToolSet({ codex_send: sendTool(ids) }).forRun("r1");
    const node = new ToolNode("send", { toolId: "codex_send" });
    await node.forward({ text: "hi" }, capsWith([], tools).caps);
    await node.forward({ text: "hi" }, capsWith([], tools).caps);
    expect(ids).toEqual(["r1/send", "r1/send"]);
  });

  test("an open held call waits again without calling the handler", async () => {
    const ids: string[] = [];
    const { recorded, caps } = capsWith([heldCall()], new ToolSet({ codex_send: sendTool(ids) }).forRun("r1"));
    expect(await new ToolNode("send", { toolId: "codex_send" }).forward({ text: "hi" }, caps)).toEqual(
      new Pending(["r1/send"]),
    );
    expect(ids).toEqual([]);
    expect(recorded).toEqual([]);
  });

  test("an answered held call returns its output, or fails with its error for good", async () => {
    const ids: string[] = [];
    const tools = new ToolSet({ codex_send: sendTool(ids) }).forRun("r1");
    const node = new ToolNode("send", { toolId: "codex_send", retries: 3 });
    const answered = heldCall({
      state: "answered",
      result: { kind: "tool_result", call_id: "r1/send", output: { kind: "str", value: "done" } },
    });
    expect(await node.forward({ text: "hi" }, capsWith([answered], tools).caps)).toEqual(new Output("done"));
    const failed = heldCall({
      state: "answered",
      result: { kind: "tool_result", call_id: "r1/send", error: { type: "CompletedError", message: "no" } },
    });
    const { recorded, caps } = capsWith([failed], tools);
    const error = await rejection(node.forward({ text: "hi" }, caps));
    expect(error).toBeInstanceOf(CompletedError);
    expect((error as Error).message).toBe("no");
    expect(ids).toEqual([]);
    expect(recorded).toEqual([]);
  });

  test("a pending step over the event size limit fails the node before anything is written", async () => {
    const step = { kind: "pending", call_id: "r1/send", tool: "codex_send", arguments: { text: "hi" } };
    const size = eventBytes(step);
    const tools = new ToolSet({ codex_send: sendTool([]) }).forRun("r1");
    const node = new ToolNode("send", { toolId: "codex_send" });
    const over = capsWith([], tools, size - 1);
    const error = await rejection(node.forward({ text: "hi" }, over.caps));
    expect(error).toBeInstanceOf(ToolCheckError);
    expect((error as Error).message).toBe(
      `tool 'codex_send' pending call step is ${size} bytes, over the run's limit of ${size - 1} (YIAN_MAX_EVENT_BYTES)`,
    );
    expect(over.recorded).toEqual([]);
    // The whole step is measured, not only its arguments: exactly at the limit it is recorded.
    const at = capsWith([], tools, size);
    expect(await node.forward({ text: "hi" }, at.caps)).toEqual(new Pending(["r1/send"]));
    expect(at.recorded).toEqual([step]);
  });

  test("pending() outside a durable run is an error naming the tool", async () => {
    const tools = new ToolSet({ codex_send: sendTool([]) }).forRun("r1");
    const error = await rejection(new ToolNode("send", { toolId: "codex_send" }).forward({ text: "hi" }, { tools }));
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toBe(
      "tool 'codex_send' returned pending() outside a durable run; give the run a log",
    );
  });

  test("an inline answer is still the output", async () => {
    const tools = new ToolSet({ codex_send: sendTool([], () => "sent") }).forRun("r1");
    const { recorded, caps } = capsWith([], tools);
    expect(await new ToolNode("send", { toolId: "codex_send" }).forward({ text: "hi" }, caps)).toEqual(
      new Output("sent"),
    );
    expect(recorded).toEqual([]);
  });

  test("waitResultStep shapes complete()", () => {
    const node = new ToolNode("send", { toolId: "codex_send" });
    node.outputContract = new ValueContract(Type.scalar(ValueKind.STRING));
    const view = heldCall();
    expect(node.waitResultStep(view, { output: "ok" })).toEqual({
      kind: "tool_result",
      call_id: "r1/send",
      output: { kind: "str", value: "ok" },
    });
    expect(node.waitResultStep(view, { error: "boom" })).toEqual({
      kind: "tool_result",
      call_id: "r1/send",
      error: { type: "CompletedError", message: "boom" },
    });
    expect(() => node.waitResultStep(view, { output: 5 })).toThrow(TypeCheckError);
    // A failure inside the value names where, as host intake does; a root one stays bare.
    const listed = new ToolNode("send", { toolId: "codex_send" });
    listed.outputContract = new ValueContract(Type.scalar(ValueKind.LIST_INTEGER));
    expect(() => listed.waitResultStep(view, { output: [1, "x"] })).toThrow(
      "complete('r1/send') output at [1]: list element \"x\" is not int (declared list(int))",
    );
    expect(() => node.waitResultStep(view, { output: 5 })).toThrow(/^5 does not match declared type str$/);
    // With no declared output the value keeps its natural kind.
    const free = new ToolNode("send", { toolId: "codex_send" });
    expect(free.waitResultStep(view, { output: 5n })).toEqual({
      kind: "tool_result",
      call_id: "r1/send",
      output: { kind: "int", value: 5n },
    });
  });
});

describe("ask", () => {
  /** A run's tools whose ask handler pushes each request and returns `answer(request)`. */
  function askTools(seen: AskRequest[], answer: (request: AskRequest) => unknown = () => pending()) {
    return new ToolSet({
      ask: { handler: (request: AskRequest) => (seen.push(request), answer(request)), inputs: { to: { type: "string" } } },
    }).forRun("r1");
  }

  test("ask sends the base surface, extras, call id and deadline; pending() records the deadline", async () => {
    const seen: AskRequest[] = [];
    const node = new ToolNode("lead", {
      toolId: "ask",
      answerSchema: [{ name: "decision", type: "str", required: true }],
    });
    const { recorded, caps } = capsWith([], askTools(seen));
    expect(await node.forward({ prompt: "Approve?", timeout: 86400n, to: "lead" }, caps)).toEqual(
      new Pending(["r1/lead"]),
    );
    expect(seen[0]).toMatchObject({
      question: "Approve?",
      callId: "r1/lead",
      deadline: "2026-10-24T00:00:00.000Z",
      extra: { to: "lead" },
      nodeId: "lead",
      answerSchema: [{ name: "decision", type: "str", required: true }],
    });
    expect(recorded).toEqual([
      {
        kind: "pending",
        call_id: "r1/lead",
        tool: "ask",
        arguments: { prompt: "Approve?", timeout: 86400n, to: "lead" },
        deadline: "2026-10-24T00:00:00.000Z",
      },
    ]);
  });

  test("ask parses questions or __questions before the handler; an inline answer is the output", async () => {
    const seen: AskRequest[] = [];
    const node = new ToolNode("pick", { toolId: "ask" });
    const tools = askTools(seen, () => ({ h: "a" }));
    const questions = [{ question: "Pick", header: "h", options: [{ label: "a" }] }];
    const parsed = [{ question: "Pick", header: "h", options: [{ label: "a", description: "" }], multiSelect: false }];
    expect(await node.forward({ questions }, capsWith([], tools).caps)).toEqual(new Output({ h: "a" }));
    expect(await node.forward({ __questions: questions }, capsWith([], tools).caps)).toEqual(new Output({ h: "a" }));
    expect(seen.map((request) => [request.questions, request.deadline, request.question])).toEqual([
      [parsed, null, ""],
      [parsed, null, ""],
    ]);
    const { recorded, caps } = capsWith([], tools);
    expect(await rejection(node.forward({ questions: [{ question: "no header" }] }, caps))).toBeInstanceOf(
      QuestionSpecError,
    );
    expect(seen).toHaveLength(2);
    expect(recorded).toEqual([]);
  });

  test("a timeout whose deadline is past the year 9999 is contract.input naming the timeout", async () => {
    const seen: AskRequest[] = [];
    const { recorded, caps } = capsWith([], askTools(seen));
    await expectContractInput(
      new ToolNode("lead", { toolId: "ask" }).forward({ prompt: "?", timeout: "999999999d" }, caps),
      'ask `timeout` "999999999d": the deadline falls after the year 9999',
    );
    expect(seen).toEqual([]);
    expect(recorded).toEqual([]);
  });

  test("an ask with an open held call waits again without asking", async () => {
    const seen: AskRequest[] = [];
    const held = heldCall({ callId: "r1/lead", tool: "ask", deadline: "2026-10-24T00:00:00.000Z" });
    const { recorded, caps } = capsWith([held], askTools(seen));
    expect(await new ToolNode("lead", { toolId: "ask" }).forward({ prompt: "?" }, caps)).toEqual(
      new Pending(["r1/lead"]),
    );
    expect(seen).toEqual([]);
    expect(recorded).toEqual([]);
  });

  test("an ask closed timed_out returns null for Optional(T) and fails tool.timeout otherwise, no output: included", async () => {
    const held = heldCall({
      callId: "r1/lead",
      tool: "ask",
      state: "closed",
      closedReason: "timed_out",
      deadline: "2026-10-24T00:00:00.000Z",
    });
    const tools = askTools([]);
    const optional = new ToolNode("lead", { toolId: "ask" });
    optional.outputContract = new ValueContract(APPROVAL.replace({ nullable: true }));
    expect(await optional.forward({ prompt: "?" }, capsWith([held], tools).caps)).toEqual(new Output(null));

    const strict = new ToolNode("lead", { toolId: "ask" });
    strict.outputContract = new ValueContract(APPROVAL);
    const error = await rejection(strict.forward({ prompt: "?" }, capsWith([held], tools).caps));
    expect(error).toBeInstanceOf(ToolTimeoutError);
    expect((error as Error).message).toBe(
      "TOOL node 'lead' (ask): nobody answered by 2026-10-24T00:00:00.000Z; declare output: Optional(...) to route on a timeout",
    );
    expect(strict.exceptionCode(error)).toBe("tool.timeout");

    // No `output:` reads as `str`, which a timeout can't give.
    const free = new ToolNode("lead", { toolId: "ask" });
    expect(await rejection(free.forward({ prompt: "?" }, capsWith([held], tools).caps))).toBeInstanceOf(ToolTimeoutError);
  });

  test("ask's waitResultStep checks the answer type without null", () => {
    const node = new ToolNode("lead", { toolId: "ask" });
    node.outputContract = new ValueContract(APPROVAL.replace({ nullable: true }));
    const view = heldCall({ callId: "r1/lead", tool: "ask" });
    expect(() => node.waitResultStep(view, { output: null })).toThrow(TypeCheckError);
    // With no declared output null is still refused: it means nobody answered in time.
    const free = new ToolNode("lead", { toolId: "ask" });
    expect(() => free.waitResultStep(view, { output: null })).toThrow(
      new TypeCheckError("complete('r1/lead') output: an ask's answer can't be null; null means nobody answered in time"),
    );
    expect(node.waitResultStep(view, { output: { decision: "approve" } })).toEqual({
      kind: "tool_result",
      call_id: "r1/lead",
      output: { kind: "object", value: { decision: "approve" } },
    });
  });
});
