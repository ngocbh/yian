/**
 * Tool effects recorded in the AGENT's transcript: mirrors
 * `tests/engine/test_agent_session_tools.py`.
 *
 * Ported, at the adapter level: the `request` and `observe` stages of
 * `test_tool_memory_fault_cannot_repeat_external_action`, and
 * `test_tool_error_identity_survives_failed_observation` (2 rows).
 *
 * Changed: a host stop is the aborted signal, its reason what the tool throws, rather than a
 * `_HostStop`; an observation failure's note is on the error's own `notes` array.
 *
 * TS-only: a result is a `completed` tool message under the snake_case request payload; a reported
 * error is a `completed` error result; a stop pending at a reported error records `interrupted`
 * and throws the stop; a skipped call is requested and observed `interrupted` with
 * `SKIPPED_TOOL_RESULT`; a representation error leaves the effect `completed` without items and
 * propagates; `interruptTurn` calls a turn's optional `interrupt`, and a failing one is a
 * `CapabilityFault`.
 *
 * Not ported: the `encode` stage of `test_tool_memory_fault_cannot_repeat_external_action`: a TS
 * `Message` always encodes, and the `observe` stage covers a fault after the call.
 *
 * Later (T6-6): `test_restored_unclosed_invocation_reports_unknown_completion_without_changing_history`
 * (the history projection), `test_two_runs_restore_tool_exchanges_without_executing_old_calls`
 * (the loop wiring), and the rows that need `run_flow`:
 * `test_recalled_transcript_and_current_notes_share_one_context`,
 * `test_changed_grants_keep_history_but_reject_new_ungranted_call`,
 * `test_missing_tool_result_is_repaired_only_in_context_with_truthful_status`,
 * `test_unpairable_provider_call_shapes_become_inert_context_only`,
 * `test_mismatched_responses_call_blocks_are_inert_history`,
 * `test_stop_boundaries_retain_surfaced_tool_truth_before_unwind` and
 * `test_session_tool_calls_keep_public_argument_validation`.
 */

import { expect, test } from "bun:test";

import { Message, ToolResultPart } from "../../src/llm/messages.ts";
import type { ConversationTurn, ConversationView } from "../../src/memory/contracts.ts";
import type { Turn } from "../../src/memory/session.ts";
import {
  decodeMessage,
  interruptTurn,
  invokeToolEffect,
  recordSkippedTool,
  SKIPPED_TOOL_RESULT,
} from "../../src/nodes/agent/memory.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { ReportedToolError } from "../../src/tools/errors.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";
import { NEVER, rejected, request, withHarness, withTurn } from "./session_fakes.ts";

/** A hand-made turn of only `request` and `observe`, as Python's `SimpleNamespace` turns. */
function fakeTurn(
  request: () => number | Promise<number>,
  observe: (ordinal: number, status: string) => void | Promise<void>,
): ConversationTurn {
  return { request, observe } as unknown as ConversationTurn;
}

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
    expect((primary as Error & { notes?: string[] }).notes![0]).toContain("cannot save failure");
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
  expect(error).toBeInstanceOf(TypeCheckError);
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
