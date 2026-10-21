/**
 * Tool effects recorded in the AGENT's transcript: mirrors
 * `tests/engine/test_agent_session_tools.py`.
 *
 * Ported, at the adapter level: the `request` and `observe` stages of
 * `test_tool_memory_fault_cannot_repeat_external_action`,
 * `test_tool_error_identity_survives_failed_observation` (2 rows), and
 * `test_restored_unclosed_invocation_reports_unknown_completion_without_changing_history`. Through
 * `FlowEngine` with a `SessionMemory` (`runSessionFlow`), three runs with distinct run ids:
 * `test_two_runs_restore_tool_exchanges_without_executing_old_calls`, which also checks that the
 * evaluator closed both turns `completed`.
 *
 * Changed: a host stop is the aborted signal, its reason what the tool throws, rather than a
 * `_HostStop`; an observation failure's note is on the error's own `notes` array. The restored
 * test checks the exact status text.
 *
 * TS-only: a result is a `completed` tool message under the snake_case request payload; a reported
 * error is a `completed` error result; a stop pending at a reported error records `interrupted`
 * and throws the stop; a skipped call is requested and observed `interrupted` with
 * `SKIPPED_TOOL_RESULT`; a representation error leaves the effect `completed` without items and
 * propagates; `interruptTurn` calls a turn's optional `interrupt`, and a failing one is a
 * `CapabilityFault`; a stop the turn throws after publishing wins over a representation error (a
 * spied `fromHost` throws it), which gets no observation note; a `CapabilityFault` with a stop
 * pending is recorded `failed` and thrown, without the stop check; a skip under an already aborted
 * signal throws the stop, not a `CapabilityFault`, and records nothing, since a session turn
 * checks the stop before publishing a request. The history projection: a call without its result
 * gets a context-only error result with each of the three repair texts (effect `completed`,
 * `failed`, or another status such as `interrupted` or none recorded), a non-`completed` effect is
 * listed in the status message; an unpaired tool result and a tool message with another part
 * are inert `unpaired tool result` text; a reply with an unparsed call, an empty name or id,
 * repeated ids, or a tool result, and a native structured reply, are inert `structured or
 * unpairable model reply` text; a stored system message is dropped. Through `runAgent`: an
 * interrupt skipping a reply's calls records each `interrupted` with `SKIPPED_TOOL_RESULT` in call
 * order and asks the turn to `interrupt()`; a granted call whose argument check fails is requested
 * and observed `failed` (its `interrupted` branch under a stop is `invokeToolEffect`'s, covered by
 * the rows above); an ungranted call records nothing.
 *
 * TS-only, a reply's concurrency-safe calls run as one group through `runAgent`: each is requested
 * before it starts (both pending while they run), and results are stored in call order whatever
 * the finish order; a failing group's finished sibling stays `completed` with its result and the
 * failing call is `failed`; with a stop pending at settle, a returned call stays `completed` and a
 * failed or reported one is `interrupted`, the stop thrown; a request failing mid-group starts no
 * later call, awaits and observes the started one, and throws its `CapabilityFault`; a request
 * that throws the stop does the same, then throws the stop; a sibling failing while a call's
 * request is stored keeps that call from starting, its effect `pending`; a result `toolText`
 * refuses in a failing group is `completed` without items and the failure wins; without a failure
 * it is thrown once the earlier calls' results are committed, their evidence kept; an observation
 * fault mid-group leaves it and every later call `pending` and throws a `CapabilityFault` of the
 * fault, or of a failed call's error with the fault noted on it, winning over a later call's
 * failure, while an earlier call's failure wins over it; a pending stop wins over the fault.
 *
 * Not ported: the `encode` stage of `test_tool_memory_fault_cannot_repeat_external_action`: a TS
 * `Message` always encodes, and the `observe` stage covers a fault after the call.
 *
 * Later (T6-6 PR 3), the rows that need `run_flow`:
 * `test_recalled_transcript_and_current_notes_share_one_context`,
 * `test_changed_grants_keep_history_but_reject_new_ungranted_call`,
 * `test_missing_tool_result_is_repaired_only_in_context_with_truthful_status`,
 * `test_unpairable_provider_call_shapes_become_inert_context_only`,
 * `test_mismatched_responses_call_blocks_are_inert_history`,
 * `test_stop_boundaries_retain_surfaced_tool_truth_before_unwind` and
 * `test_session_tool_calls_keep_public_argument_validation`.
 */

import { expect, spyOn, test } from "bun:test";

import { ModelCallFinished, RunSucceeded } from "../../src/events.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import {
  Message,
  type Part,
  ReasoningPart,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type ConversationTurn,
  type ConversationView,
  type EvaluationTrace,
  type JsonObject,
  ToolResultInteraction,
  type VisibleInteraction,
} from "../../src/memory/contracts.ts";
import { SessionMemory, type Turn } from "../../src/memory/session.ts";
import { AgentLoopError, type AgentRunContext } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import {
  decodeMessage,
  encodeMessage,
  historyMessages,
  interruptTurn,
  invokeModel,
  invokeToolEffect,
  recordSkippedTool,
  SKIPPED_TOOL_RESULT,
} from "../../src/nodes/agent/memory.ts";
import { CapabilityFault, type Steering } from "../../src/nodes/base.ts";
import { ReportedToolError } from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { logJson } from "../../src/typesys/json.ts";
import * as values from "../../src/typesys/values.ts";
import {
  agentContext,
  fakeTurn,
  type Mutable,
  NEVER,
  rejected,
  request,
  runSessionFlow,
  sessionFlow,
  snapshot,
  view,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

const call = { name: "lookup", arguments: { value: 1 }, modelExchangeId: "model", callIndex: 0, toolCallId: "provider" };

const toolMessage = (content: string, isError = false): Message =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId: "provider", name: "lookup", content, isError })] });

/** Run `fn` in a turn of a fresh session, returning its outcome and the turn's last view. */
async function inTurn<T>(
  fn: (turn: Turn) => Promise<T>,
  signal: AbortSignal = NEVER,
): Promise<{ result?: T; error?: unknown; view: ConversationView }> {
  return withHarness(async ({ session }) => {
    let view: ConversationView | null = null;
    const outcome: { result?: T; error?: unknown } = {};
    try {
      outcome.result = await withTurn(
        session,
        request(),
        async (turn) => {
          try {
            return await fn(turn);
          } finally {
            view = turn.view;
          }
        },
        signal,
      );
    } catch (error) {
      outcome.error = error;
    }
    return { ...outcome, view: view! };
  });
}

for (const stage of ["request", "observe"] as const) {
  test(`tool memory fault cannot repeat external action: ${stage}`, async () => {
    const error = new RangeError("memory refused");
    const calls: string[] = [];
    const observations: string[] = [];
    const turn = fakeTurn(
      () => {
        if (stage === "request") throw error;
        return 1;
      },
      (_ordinal, status) => {
        observations.push(status);
        if (stage === "observe") throw error;
      },
    );
    const invoke = async () => {
      calls.push("tool");
      return "actual";
    };
    const caught = await rejected(invokeToolEffect(turn, invoke, { ...call, signal: NEVER }));
    expect(caught).toBeInstanceOf(CapabilityFault);
    expect((caught as CapabilityFault).original).toBe(error);
    expect(calls).toEqual(stage === "request" ? [] : ["tool"]);
    expect(observations).toEqual(stage === "request" ? [] : ["completed"]);
  });
}

for (const hostStop of [false, true]) {
  test(`tool error identity survives failed observation: host stop ${hostStop}`, async () => {
    const controller = new AbortController();
    const primary = new Error(hostStop ? "stopped" : "tool failed");
    const seen: string[] = [];
    const invoke = async (): Promise<unknown> => {
      if (hostStop) controller.abort(primary);
      throw primary;
    };
    const turn = fakeTurn(
      () => 1,
      (_ordinal, status) => {
        seen.push(status);
        throw new Error("cannot save failure");
      },
    );
    const caught = await rejected(invokeToolEffect(turn, invoke, { ...call, arguments: {}, signal: controller.signal }));
    if (hostStop) {
      expect(caught).toBe(primary);
    } else {
      expect(caught).toBeInstanceOf(CapabilityFault);
      expect((caught as CapabilityFault).original).toBe(primary);
    }
    expect(seen).toEqual([hostStop ? "interrupted" : "failed"]);
    expect((primary as Error & { notes?: string[] }).notes).toEqual([
      "Conversation observation failed: Error: cannot save failure",
    ]);
  });
}

test("a result is a completed tool message under the snake_case request payload", async () => {
  const { result, view } = await inTurn((turn) => invokeToolEffect(turn, async () => ({ a: 1 }), call));
  expect(result).toEqual(toolMessage('{"a":1}'));
  const [effect] = view.effects;
  expect(effect!.kind).toBe("tool");
  expect(effect!.request).toEqual({
    name: "lookup",
    arguments: { value: 1 },
    model_exchange_id: "model",
    call_index: 0,
    tool_call_id: "provider",
  });
  expect(effect!.status).toBe("completed");
  expect(view.items.map(decodeMessage)).toEqual([result!]);
  expect(view.items[0]!.payload["exchange_id"]).toBe("model");
});

test("a reported error is a completed error result", async () => {
  const reported = new ReportedToolError("no such row");
  const { result, view } = await inTurn((turn) => invokeToolEffect(turn, () => Promise.reject(reported), call));
  expect(result).toEqual(toolMessage("no such row", true));
  expect(view.effects[0]!.status).toBe("completed");
  expect(view.items.map(decodeMessage)).toEqual([result!]);
});

test("a stop pending at a reported error records interrupted and throws the stop", async () => {
  const controller = new AbortController();
  const stop = new Error("stopped");
  const invoke = async (): Promise<unknown> => {
    controller.abort(stop);
    throw new ReportedToolError("no such row");
  };
  const { error, view } = await inTurn(
    (turn) => invokeToolEffect(turn, invoke, { ...call, signal: controller.signal }),
    controller.signal,
  );
  expect(error).toBe(stop);
  expect(view.effects[0]!.status).toBe("interrupted");
  expect(view.effects[0]!.failure!.message).toBe("Tool invocation interrupted (Error).");
  expect(view.items).toHaveLength(0);
});

test("a skipped call is requested and observed interrupted with SKIPPED_TOOL_RESULT", async () => {
  const { result, view } = await inTurn((turn) => recordSkippedTool(turn, call));
  expect(result).toEqual(toolMessage(SKIPPED_TOOL_RESULT, true));
  expect(view.effects[0]!.request["tool_call_id"]).toBe("provider");
  expect(view.effects[0]!.status).toBe("interrupted");
  expect(view.effects[0]!.failure).toBeNull();
  expect(view.items.map(decodeMessage)).toEqual([result!]);
  expect(await recordSkippedTool(null, call)).toEqual(result!);
});

test("a representation error leaves the effect completed without items and propagates", async () => {
  const { error, view } = await inTurn((turn) => invokeToolEffect(turn, async () => () => "not a value", call));
  expect(error).toBeInstanceOf(values.TypeCheckError);
  expect(view.effects[0]!.status).toBe("completed");
  expect(view.items).toHaveLength(0);
});

test("interruptTurn calls a turn's optional interrupt, and a failing one is a CapabilityFault", async () => {
  await interruptTurn(null);
  await interruptTurn(fakeTurn(() => 1, () => {}));
  const calls: string[] = [];
  await interruptTurn({ interrupt: () => calls.push("interrupt") } as unknown as ConversationTurn);
  expect(calls).toEqual(["interrupt"]);
  const error = new Error("cannot interrupt");
  const caught = await rejected(
    interruptTurn({
      interrupt: () => {
        throw error;
      },
    } as unknown as ConversationTurn),
  );
  expect((caught as CapabilityFault).original).toBe(error);
});

test("a stop the turn throws after publishing wins over a representation error", async () => {
  const controller = new AbortController();
  const stop = new Error("stopped");
  // The representation error itself, to check it gets no observation note.
  const representation = new TypeError("not representable");
  const spy = spyOn(values, "fromHost").mockImplementation(() => {
    throw representation;
  });
  const invoke = async (): Promise<unknown> => {
    controller.abort(stop);
    return "actual";
  };
  try {
    const { error, view } = await inTurn(
      (turn) => invokeToolEffect(turn, invoke, { ...call, signal: controller.signal }),
      controller.signal,
    );
    expect(error).toBe(stop);
    expect(view.effects[0]!.status).toBe("completed");
    expect(view.items).toHaveLength(0);
    expect(spy).toHaveBeenCalled();
  } finally {
    spy.mockRestore();
  }
  expect((representation as Error & { notes?: string[] }).notes).toBeUndefined();
});

test("a CapabilityFault with a stop pending is recorded failed and thrown, without the stop check", async () => {
  const controller = new AbortController();
  const fault = new CapabilityFault(new Error("backend"));
  const invoke = async (): Promise<unknown> => {
    controller.abort(new Error("stopped"));
    throw fault;
  };
  const { error, view } = await inTurn(
    (turn) => invokeToolEffect(turn, invoke, { ...call, signal: controller.signal }),
    controller.signal,
  );
  expect(error).toBe(fault);
  expect(view.effects[0]!.status).toBe("failed");
  expect(view.effects[0]!.failure!.message).toBe("Tool invocation failed (CapabilityFault).");
});

test("a skip under an already aborted signal throws the stop and records nothing", async () => {
  const controller = new AbortController();
  const stop = new Error("stopped");
  const { error, view } = await inTurn(async (turn) => {
    controller.abort(stop);
    return recordSkippedTool(turn, { ...call, signal: controller.signal });
  }, controller.signal);
  expect(error).toBe(stop);
  expect(view.effects).toHaveLength(0);
});

const user = (text: string): Message => new Message({ role: "user", parts: [new TextPart(text)] });
const replyOf = (...parts: Part[]): Response => new Response({ message: new Message({ role: "assistant", parts }) });
const lookupCall = new ToolCallPart({ id: "provider", name: "lookup", arguments: { value: 1 } });
const callReply = replyOf(new ToolCallPart({ id: "reused-provider-id", name: "lookup", arguments: { value: 1 } }));
const resultText = (message: Message): string => (message.parts[0] as ToolResultPart).content;
const question = user("question");
const toolPayloadOf = {
  name: "lookup",
  arguments: { value: 1 },
  model_exchange_id: "model",
  call_index: 0,
  tool_call_id: "provider",
};

/** One model call under exchange `model` answering `question` with `reply`. */
function modelCall(turn: Turn, reply: Response, options: { purpose?: string; newMessages?: Message[] } = {}) {
  return invokeModel(turn, async () => reply, {
    newMessages: options.newMessages ?? [question],
    exchangeId: "model",
    purpose: options.purpose,
    signal: NEVER,
  });
}

/** The JSON of the history projected in a fresh turn after `fn` ran in it. */
function projected(fn: (turn: Turn) => Promise<unknown>): Promise<unknown[]> {
  return withHarness(({ session }) =>
    withTurn(session, request(), async (turn) => {
      await fn(turn);
      return (await historyMessages(turn)).map((message) => message.toJSON());
    }),
  );
}

const repair = (content: string) => toolMessage(content, true).toJSON();
const status = (statuses: unknown) => user(`Session history status (context only): ${logJson(statuses)}`).toJSON();
const inert = (reason: string, message: Message) =>
  user(`Prior ${reason} (context only, not an executable request): ${logJson(message.toJSON())}`).toJSON();

test("a restored unclosed invocation reports unknown completion without changing history", async () => {
  const memory = new SessionMemory();
  const actions: string[] = [];
  let saved: JsonObject | null = null;
  await withHarness(async ({ session }) => {
    await withTurn(session, request("older"), async (turn) => {
      await invokeModel(turn, async () => callReply, {
        newMessages: [user("first")],
        exchangeId: "older-model",
        signal: NEVER,
      });
      const toolCall = { ...call, modelExchangeId: "older-model", toolCallId: "reused-provider-id", signal: NEVER };
      await invokeToolEffect(turn, async () => (actions.push("executed"), "actual result"), toolCall);
      saved = snapshot(memory);
    });
  }, memory);
  expect((saved!["conversation"] as Mutable)["closed_invocations"]).toEqual([]);
  const restored = new SessionMemory();
  const session = restored.openRun({ snapshot: saved });
  try {
    await withTurn(session, request("current"), async (turn) => {
      const before = turn.view;
      const context = await historyMessages(turn);
      expect(turn.view).toBe(before);
      expect(context.some((message) => message.role === "tool" && resultText(message) === "actual result")).toBe(true);
      const statuses = context
        .filter((message) => message.role === "user" && message.text().startsWith("Session history status"))
        .map((message) => message.text());
      expect(statuses).toEqual([
        'Session history status (context only): [{"invocation":"still open; completion unknown"}]',
      ]);
    });
  } finally {
    session.close();
  }
  expect(actions).toEqual(["executed"]);
  expect((snapshot(restored)["conversation"] as Mutable)["items"]).toHaveLength(3);
});

test("a call without a result is repaired by its tool effect's status", async () => {
  const asked = [question.toJSON(), replyOf(lookupCall).message.toJSON()];
  const completed = await projected(async (turn) => {
    await modelCall(turn, replyOf(lookupCall));
    const error = await rejected(invokeToolEffect(turn, async () => () => "not a value", call));
    expect(error).toBeInstanceOf(values.TypeCheckError);
  });
  expect(completed).toEqual([...asked, repair("Tool execution completed; represented result unavailable.")]);
  const failed = await projected(async (turn) => {
    await modelCall(turn, replyOf(lookupCall));
    const error = await rejected(invokeToolEffect(turn, async () => Promise.reject(new Error("tool failed")), call));
    expect((error as Error).message).toBe("tool failed");
  });
  expect(failed).toEqual([
    ...asked,
    repair("Tool invocation failed; result unavailable."),
    status([{ effect: "tool", status: "failed" }]),
  ]);
  const unrecorded = await projected((turn) => modelCall(turn, replyOf(lookupCall)));
  expect(unrecorded).toEqual([
    ...asked,
    repair("Tool result unavailable; execution outcome unknown (status: unrecorded)."),
  ]);
  const interrupted = await projected(async (turn) => {
    await modelCall(turn, replyOf(lookupCall));
    const controller = new AbortController();
    const stop = new Error("stopped");
    const invoke = async () => {
      controller.abort(stop);
      throw stop;
    };
    expect(await rejected(invokeToolEffect(turn, invoke, { ...call, signal: controller.signal }))).toBe(stop);
  });
  expect(interrupted).toEqual([
    ...asked,
    repair("Tool result unavailable; execution outcome unknown (status: interrupted)."),
    status([{ effect: "tool", status: "interrupted" }]),
  ]);
});

test("an unpaired tool result is inert context, and so is a tool message with other parts", async () => {
  const unpaired = await projected(async (turn) => {
    await modelCall(turn, replyOf(new TextPart("no call")));
    await invokeToolEffect(turn, async () => "actual", call);
  });
  expect(unpaired).toEqual([
    question.toJSON(),
    replyOf(new TextPart("no call")).message.toJSON(),
    inert("unpaired tool result", toolMessage("actual")),
  ]);
  const mixed = new Message({
    role: "tool",
    parts: [new ToolResultPart({ callId: "provider", name: "lookup", content: "actual" }), new TextPart("extra")],
  });
  const extra = await projected(async (turn) => {
    await modelCall(turn, replyOf(lookupCall));
    const ordinal = turn.request("tool", toolPayloadOf);
    turn.observe(ordinal, "completed", { items: [encodeMessage(mixed, "model")] });
  });
  expect(extra).toEqual([
    question.toJSON(),
    replyOf(lookupCall).message.toJSON(),
    repair("Tool execution completed; represented result unavailable."),
    inert("unpaired tool result", mixed),
  ]);
});

const unpairable: [string, Part[]][] = [
  ["an unparsed call", [new ToolCallPart({ id: "provider", name: "lookup", arguments: null, rawArguments: "{" })]],
  ["an empty name", [new ToolCallPart({ id: "provider", name: "", arguments: {} })]],
  ["an empty id", [new ToolCallPart({ id: "", name: "lookup", arguments: {} })]],
  ["repeated ids", [lookupCall, new ToolCallPart({ id: "provider", name: "other", arguments: {} })]],
  ["a tool result", [new ToolResultPart({ callId: "provider", name: "lookup", content: "result" })]],
];

for (const [label, parts] of unpairable) {
  test(`a reply with ${label} is inert context`, async () => {
    const reply = replyOf(...parts);
    expect(await projected((turn) => modelCall(turn, reply))).toEqual([
      question.toJSON(),
      inert("structured or unpairable model reply", reply.message),
    ]);
  });
}

test("a native structured reply is inert context, and a stored system message is dropped", async () => {
  const reply = replyOf(lookupCall);
  expect(await projected((turn) => modelCall(turn, reply, { purpose: "structured_native" }))).toEqual([
    question.toJSON(),
    inert("structured or unpairable model reply", reply.message),
  ]);
  const system = new Message({ role: "system", parts: [new TextPart("OLD SYSTEM")] });
  const answer = replyOf(new TextPart("answer"));
  expect(await projected((turn) => modelCall(turn, answer, { newMessages: [system, question] }))).toEqual([
    question.toJSON(),
    answer.message.toJSON(),
  ]);
});

/** `agentContext` granted `lookup`, with `overrides`. */
function context(model: ChatModel, transcript: ConversationTurn, overrides: Partial<AgentRunContext> = {}) {
  const toolset = new ToolSet({ lookup: lookupTool() });
  return agentContext(model, transcript, { tools: ["lookup"], toolset, ...overrides });
}

/** A `lookup` tool taking an integer `value`; it must not run. */
function lookupTool(): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look a value up.",
    parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
    execute: () => {
      throw new Error("the tool ran");
    },
  });
}

/** A reply with `text` and one tool call per `[name, arguments, id]`. */
function callsReply(
  text: string,
  ...calls: (readonly [string, NonNullable<ToolCallPart["arguments"]>, string])[]
): Response {
  const toolCalls = calls.map(([name, args, id]) => new ToolCallPart({ id, name, arguments: args }));
  const parts: Part[] = [new TextPart(text), ...toolCalls];
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** The kind, status and request payload of each effect of `view`, the model's exchange id dropped. */
function effects(view: ConversationView): [string, string, unknown][] {
  return view.effects.map((effect) => {
    const { exchange_id: _exchange, model_exchange_id: _model, ...payload } = effect.request as Mutable;
    return [effect.kind, effect.status, payload];
  });
}

test("an interrupt skipping calls records each interrupted in call order and asks the turn to interrupt", async () => {
  const answer = callsReply("Checking both.", ["lookup", { value: 1n }, "call-1"], ["lookup", { value: 2n }, "call-2"]);
  const model = new ScriptedModel([answer]);
  const interrupt = new AbortController();
  const steering: Steering = { interruptSignal: () => interrupt.signal, drain: () => [] };
  // Interrupted once the reply arrived, so its calls are skipped rather than the reply cut.
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallFinished) interrupt.abort(new Error("interrupted"));
  };
  let asked: ReturnType<typeof spyOn> | null = null;
  const { result, view } = await inTurn((turn) => {
    asked = spyOn(turn, "interrupt");
    return runAgent(context(model, turn, { steering, emitEvent }));
  });
  expect(result!.value).toBe("Checking both.");
  expect(asked!).toHaveBeenCalledTimes(1);
  expect(effects(view)).toEqual([
    ["model", "completed", { purpose: "chat" }],
    ["tool", "interrupted", { name: "lookup", arguments: { value: 1n }, call_index: 0, tool_call_id: "call-1" }],
    ["tool", "interrupted", { name: "lookup", arguments: { value: 2n }, call_index: 1, tool_call_id: "call-2" }],
  ]);
  const exchangeId = view.effects[0]!.request["exchange_id"];
  expect(view.effects.slice(1).map((effect) => effect.request["model_exchange_id"])).toEqual([exchangeId, exchangeId]);
  const skipped = view.items.slice(-2).map(decodeMessage);
  expect(skipped.map((message) => message.parts[0])).toEqual([
    new ToolResultPart({ callId: "call-1", name: "lookup", content: SKIPPED_TOOL_RESULT, isError: true }),
    new ToolResultPart({ callId: "call-2", name: "lookup", content: SKIPPED_TOOL_RESULT, isError: true }),
  ]);
});

test("a granted call whose argument check fails is requested and observed failed", async () => {
  const model = new ScriptedModel([callsReply("", ["lookup", { value: "not a number" }, "call-1"])]);
  const { error, view } = await inTurn((turn) => runAgent(context(model, turn)));
  expect(error).toBeInstanceOf(values.TypeCheckError);
  expect(effects(view)).toEqual([
    ["model", "completed", { purpose: "chat" }],
    ["tool", "failed", { name: "lookup", arguments: { value: "not a number" }, call_index: 0, tool_call_id: "call-1" }],
  ]);
  expect(view.effects[1]!.failure!.message).toBe("Tool invocation failed (TypeCheckError).");
});

test("an ungranted call records nothing", async () => {
  const model = new ScriptedModel([callsReply("", ["other", { value: 1n }, "call-1"])]);
  const { error, view } = await inTurn((turn) => runAgent(context(model, turn)));
  expect(error).toBeInstanceOf(AgentLoopError);
  expect(effects(view)).toEqual([["model", "completed", { purpose: "chat" }]]);
});

/** A model access serving `model`. */
const serve = (model: ChatModel) => ({ env: () => undefined, createModel: () => model });

/** Python's `_call`: a reply with kept reasoning calling `lookup` on `value`, reusing one provider id. */
function callWithReasoning(value: number): Response {
  const parts = [
    new ReasoningPart("keep reasoning"),
    new ToolCallPart({ id: "reused-provider-id", name: "lookup", arguments: { value } }),
  ];
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** A reply holding only `text`. */
const textOnly = (text: string): Response => replyOf(new TextPart(text));

test("two runs restore tool exchanges without executing old calls", async () => {
  const memory = new SessionMemory();
  const calls: unknown[] = [];
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look a value up.",
    parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
    execute: (args) => {
      // The call is requested, after the reply that made it, before it runs.
      const current = view(memory);
      expect([current.effects.at(-1)!.kind, current.effects.at(-1)!.status]).toEqual(["tool", "pending"]);
      expect(decodeMessage(current.items.at(-1)!).parts.some((part) => part instanceof ToolCallPart)).toBe(true);
      calls.push(args);
      return `result:${args.value}`;
    },
  });
  const tools = new ToolSet({ lookup });
  const flow = sessionFlow({ tools: ["lookup"] });
  const model = new ScriptedModel([callWithReasoning(1), textOnly("one"), callWithReasoning(2), textOnly("two")]);
  const run = (runId: string, question: string, chat: ChatModel = model) =>
    runSessionFlow(flow, memory, { runId, inputs: { question }, tools, models: serve(chat) });
  expect(await run("first-run", "first")).toEqual(new RunSucceeded({ output: "one" }));
  expect(calls).toEqual([{ value: 1 }]);
  expect(await run("second-run", "second")).toEqual(new RunSucceeded({ output: "two" }));
  expect(calls).toEqual([{ value: 1 }, { value: 2 }]);
  const history = model.requests[2]!.messages;
  expect(history[2]!.toJSON()).toEqual(callWithReasoning(1).message.toJSON());
  expect(history[3]!.parts).toEqual([
    new ToolResultPart({ callId: "reused-provider-id", name: "lookup", content: "result:1" }),
  ]);
  const stored = view(memory);
  expect(stored.effects.map((effect) => effect.kind)).toEqual(["model", "tool", "model", "model", "tool", "model"]);
  // No repeated provider-input history or duplicated tool results.
  expect(stored.items).toHaveLength(10);
  const requests = stored.effects.filter((effect) => effect.kind === "tool").map((effect) => effect.request);
  expect(requests[0]!["model_exchange_id"]).not.toBe(requests[1]!["model_exchange_id"]);
  expect(stored.effects.every((effect) => effect.status === "completed")).toBe(true);
  // The evaluator closed both turns normally.
  expect(stored.closedInvocations.map((closed) => closed.status)).toEqual(["completed", "completed"]);
  const third = new ScriptedModel([textOnly("third")]);
  await run("third-run", "third", third);
  const restored = third.requests[0]!.messages.filter((message) => message.role === "tool").map(resultText);
  expect(restored).toEqual(["result:1", "result:2"]);
  expect(calls).toEqual([{ value: 1 }, { value: 2 }]);
});

// A reply's concurrency-safe calls run as one group: each is requested right before it starts, and
// every started call is observed in call order once the group settles.

/** Let every pending promise job run: one macrotask turn. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

/** One concurrency-safe tool whose call waits until the test settles it, whatever its signal does. */
interface Held {
  readonly tool: FunctionTool;
  /** Resolves once the call began. */
  readonly entered: Promise<void>;
  /** Whether the call began. */
  started: boolean;
  resolve(value: unknown): void;
  reject(error: unknown): void;
}

/** A held tool `name`. */
function held(name: string): Held {
  const gate = Promise.withResolvers<unknown>();
  const entered = Promise.withResolvers<void>();
  const state: Held = {
    tool: new FunctionTool({
      name,
      description: `Tool ${name}.`,
      parametersJsonSchema: { type: "object", properties: {} },
      isConcurrencySafe: () => true,
      execute: () => {
        state.started = true;
        entered.resolve();
        return gate.promise;
      },
    }),
    entered: entered.promise,
    started: false,
    resolve: gate.resolve,
    reject: gate.reject,
  };
  return state;
}

/** A reply calling each of `names` with no arguments, ids `c0`, `c1`, ... */
function groupReply(...names: string[]): Response {
  const parts = names.map((name, index) => new ToolCallPart({ id: `c${index}`, name, arguments: {} }));
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/**
 * Start `runAgent` in a turn of a fresh session over a model answering `replies`, granted the held
 * tools; `signal` is both the turn's and the node's. `arrange` sees the turn before the run starts.
 */
function groupRun(
  tools: Readonly<Record<string, Held>>,
  replies: Response[],
  options: { signal?: AbortSignal; arrange?: (turn: Turn) => void; evaluationTrace?: EvaluationTrace } = {},
) {
  const model = new ScriptedModel(replies);
  const toolset = new ToolSet(Object.fromEntries(Object.entries(tools).map(([name, { tool }]) => [name, tool])));
  let settled = false;
  let current: Turn | null = null;
  const running = inTurn((turn) => {
    current = turn;
    options.arrange?.(turn);
    const { signal, evaluationTrace } = options;
    const ctx = agentContext(model, turn, { tools: Object.keys(tools), toolset, signal, evaluationTrace });
    return runAgent(ctx).finally(() => (settled = true));
  }, options.signal);
  return { running, model, isSettled: () => settled, turn: () => current! };
}

/** The tool effects of `view` as `[tool_call_id, status]`, in request order. */
function toolStatuses(view: ConversationView): [unknown, string][] {
  return view.effects
    .filter((effect) => effect.kind === "tool")
    .map((effect) => [effect.request["tool_call_id"], effect.status]);
}

/** The stored tool results of `view`, in storing order, as `[callId, content, isError]`. */
function storedResults(view: ConversationView): [string, string, boolean][] {
  return view.items
    .map(decodeMessage)
    .filter((message) => message.role === "tool")
    .map((message) => {
      const part = message.parts[0] as ToolResultPart;
      return [part.callId, part.content, part.isError];
    });
}

/**
 * Make `turn`'s observation of call `toolCallId` throw `fault` without publishing; other
 * observations publish as usual.
 */
function faultObservation(turn: Turn, toolCallId: string, fault: unknown): void {
  const observe = turn.observe.bind(turn);
  spyOn(turn, "observe").mockImplementation((ordinal, status, options) => {
    const effect = turn.view.effects.find((candidate) => candidate.effectOrdinal === ordinal);
    if (effect?.request["tool_call_id"] === toolCallId) throw fault;
    return observe(ordinal, status, options);
  });
}

test("a group's results are committed in call order whatever the finish order", async () => {
  const a = held("a");
  const b = held("b");
  const run = groupRun({ a, b }, [groupReply("a", "b"), textOnly("done")]);
  await Promise.all([a.entered, b.entered]);
  // Both calls were requested before either finished.
  expect(toolStatuses(run.turn().view)).toEqual([
    ["c0", "pending"],
    ["c1", "pending"],
  ]);
  b.resolve("b result");
  await flush();
  a.resolve("a result");
  const { result, view } = await run.running;
  expect(result!.value).toBe("done");
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "completed"],
  ]);
  expect(storedResults(view)).toEqual([
    ["c0", "a result", false],
    ["c1", "b result", false],
  ]);
  const sent = run.model.requests[1]!.messages.filter((message) => message.role === "tool").map(resultText);
  expect(sent).toEqual(["a result", "b result"]);
});

test("a failing group's finished sibling stays completed with its result, and the failing call failed", async () => {
  const a = held("a");
  const b = held("b");
  const failure = new RangeError("b failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")]);
  await Promise.all([a.entered, b.entered]);
  a.resolve("a result");
  await flush();
  b.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBe(failure);
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "failed"],
  ]);
  expect(view.effects.at(-1)!.failure!.message).toBe("Tool invocation failed (RangeError).");
  expect(storedResults(view)).toEqual([["c0", "a result", false]]);
  expect(run.model.requests).toHaveLength(1);
});

test("a stop pending at settle: returned calls completed, failed and reported ones interrupted", async () => {
  const a = held("a");
  const b = held("b");
  const c = held("c");
  const controller = new AbortController();
  const stop = new RangeError("stopped");
  const run = groupRun({ a, b, c }, [groupReply("a", "b", "c")], { signal: controller.signal });
  await Promise.all([a.entered, b.entered, c.entered]);
  a.resolve("a result");
  b.reject(new Error("b failed"));
  await flush();
  controller.abort(stop);
  c.reject(new ReportedToolError("c reported"));
  const { error, view } = await run.running;
  expect(error).toBe(stop);
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "interrupted"],
    ["c2", "interrupted"],
  ]);
  const failures = view.effects.slice(-2).map((effect) => effect.failure!.message);
  expect(failures).toEqual(["Tool invocation interrupted (RangeError).", "Tool invocation interrupted (RangeError)."]);
  expect(storedResults(view)).toEqual([["c0", "a result", false]]);
});

test("a request failing mid-group starts no later call; the started one is awaited and observed", async () => {
  const a = held("a");
  const b = held("b");
  const c = held("c");
  const fault = new Error("memory refused");
  const run = groupRun({ a, b, c }, [groupReply("a", "b", "c")], {
    arrange: (turn) => {
      const requestOf = turn.request.bind(turn);
      spyOn(turn, "request").mockImplementation((kind, payload, options) => {
        if (payload["tool_call_id"] === "c1") throw fault;
        return requestOf(kind, payload, options);
      });
    },
  });
  await a.entered;
  await flush();
  expect(run.isSettled()).toBe(false);
  a.resolve("a result");
  const { error, view } = await run.running;
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBe(fault);
  expect([b.started, c.started]).toEqual([false, false]);
  expect(toolStatuses(view)).toEqual([["c0", "completed"]]);
  expect(storedResults(view)).toEqual([["c0", "a result", false]]);
});

test("a request throwing the stop mid-group starts no later call; the started one is observed", async () => {
  const [a, b] = [held("a"), held("b")];
  const controller = new AbortController();
  const stop = new RangeError("stopped");
  const run = groupRun({ a, b }, [groupReply("a", "b")], {
    signal: controller.signal,
    arrange: (turn) => {
      const requestOf = turn.request.bind(turn);
      spyOn(turn, "request").mockImplementation((kind, payload, options) => {
        // The session turn checks the stop before publishing, so it throws the stop and stores nothing.
        if (payload["tool_call_id"] === "c1") controller.abort(stop);
        return requestOf(kind, payload, options);
      });
    },
  });
  await a.entered;
  await flush();
  expect(run.isSettled()).toBe(false);
  a.resolve("a result");
  const { error, view } = await run.running;
  expect(error).toBe(stop);
  expect(b.started).toBe(false);
  expect(toolStatuses(view)).toEqual([["c0", "completed"]]);
  expect(storedResults(view)).toEqual([["c0", "a result", false]]);
});

test("a sibling failing while a call's request is stored keeps that call from starting", async () => {
  const [a, b] = [held("a"), held("b")];
  const failure = new RangeError("a failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")], {
    arrange: (turn) => {
      const requestOf = turn.request.bind(turn);
      // A host turn may store a request asynchronously; this one takes a macrotask for `c1`.
      spyOn(turn, "request").mockImplementation(((kind, payload, options) => {
        if (payload["tool_call_id"] !== "c1") return requestOf(kind, payload, options);
        return flush().then(() => requestOf(kind, payload, options));
      }) as Turn["request"]);
    },
  });
  await a.entered;
  a.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBe(failure);
  expect(b.started).toBe(false);
  expect(toolStatuses(view)).toEqual([
    ["c0", "failed"],
    ["c1", "pending"],
  ]);
});

test("a result toolText refuses in a failing group is observed completed without items; the failure wins", async () => {
  const a = held("a");
  const b = held("b");
  const failure = new RangeError("b failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")]);
  await Promise.all([a.entered, b.entered]);
  a.resolve(() => "not a value");
  await flush();
  b.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBe(failure);
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "failed"],
  ]);
  expect(storedResults(view)).toEqual([]);
});

test("a result toolText refuses is thrown once the earlier results are committed, their evidence kept", async () => {
  const [a, b, c] = [held("a"), held("b"), held("c")];
  const interactions: VisibleInteraction[] = [];
  const evaluationTrace: EvaluationTrace = {
    recordPrompt: () => {},
    recordInteraction: (interaction) => void interactions.push(interaction),
  };
  const run = groupRun({ a, b, c }, [groupReply("a", "b", "c")], { evaluationTrace });
  await Promise.all([a.entered, b.entered, c.entered]);
  a.resolve("a result");
  b.resolve(() => "not a value");
  c.resolve("c result");
  const { error, view } = await run.running;
  expect(error).toBeInstanceOf(values.TypeCheckError);
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "completed"],
    ["c2", "completed"],
  ]);
  expect(storedResults(view)).toEqual([
    ["c0", "a result", false],
    ["c2", "c result", false],
  ]);
  const results = interactions.filter((interaction) => interaction instanceof ToolResultInteraction);
  expect(results).toEqual([new ToolResultInteraction("a", "a result")]);
});

test("an observation fault mid-group stops observing: later calls stay pending, and a CapabilityFault", async () => {
  const [a, b, c] = [held("a"), held("b"), held("c")];
  const fault = new Error("cannot save");
  const run = groupRun({ a, b, c }, [groupReply("a", "b", "c")], {
    arrange: (turn) => faultObservation(turn, "c1", fault),
  });
  await Promise.all([a.entered, b.entered, c.entered]);
  for (const [tool, value] of [[a, "a result"], [b, "b result"], [c, "c result"]] as const) tool.resolve(value);
  const { error, view } = await run.running;
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBe(fault);
  expect(toolStatuses(view)).toEqual([
    ["c0", "completed"],
    ["c1", "pending"],
    ["c2", "pending"],
  ]);
});

test("an observation fault on a failed call is noted on its error, which is thrown wrapped", async () => {
  const [a, b] = [held("a"), held("b")];
  const fault = new Error("cannot save");
  const failure = new RangeError("a failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")], { arrange: (turn) => faultObservation(turn, "c0", fault) });
  await Promise.all([a.entered, b.entered]);
  b.resolve("b result");
  await flush();
  a.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBe(failure);
  expect((failure as RangeError & { notes?: string[] }).notes).toEqual([
    "Conversation observation failed: Error: cannot save",
  ]);
  expect(toolStatuses(view)).toEqual([
    ["c0", "pending"],
    ["c1", "pending"],
  ]);
});

test("an observation fault wins over a later call's failure", async () => {
  const [a, b] = [held("a"), held("b")];
  const fault = new Error("cannot save");
  const failure = new RangeError("b failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")], { arrange: (turn) => faultObservation(turn, "c0", fault) });
  await Promise.all([a.entered, b.entered]);
  a.resolve("a result");
  await flush();
  b.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBe(fault);
  expect(failure as RangeError & { notes?: string[] }).not.toHaveProperty("notes");
  expect(toolStatuses(view)).toEqual([
    ["c0", "pending"],
    ["c1", "pending"],
  ]);
});

test("an earlier call's failure wins over a later observation fault", async () => {
  const [a, b] = [held("a"), held("b")];
  const failure = new RangeError("a failed");
  const run = groupRun({ a, b }, [groupReply("a", "b")], {
    arrange: (turn) => faultObservation(turn, "c1", new Error("cannot save")),
  });
  await Promise.all([a.entered, b.entered]);
  b.resolve("b result");
  await flush();
  a.reject(failure);
  const { error, view } = await run.running;
  expect(error).toBe(failure);
  expect(failure as RangeError & { notes?: string[] }).not.toHaveProperty("notes");
  expect(toolStatuses(view)).toEqual([
    ["c0", "failed"],
    ["c1", "pending"],
  ]);
});

test("a pending stop wins over an observation fault", async () => {
  const [a, b] = [held("a"), held("b")];
  const controller = new AbortController();
  const stop = new RangeError("stopped");
  const run = groupRun({ a, b }, [groupReply("a", "b")], {
    signal: controller.signal,
    arrange: (turn) => faultObservation(turn, "c0", new Error("cannot save")),
  });
  await Promise.all([a.entered, b.entered]);
  a.resolve("a result");
  controller.abort(stop);
  b.resolve("b result");
  const { error, view } = await run.running;
  expect(error).toBe(stop);
  expect(toolStatuses(view)).toEqual([
    ["c0", "pending"],
    ["c1", "pending"],
  ]);
});
