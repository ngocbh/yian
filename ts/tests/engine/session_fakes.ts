/**
 * A unit-only backend-owned turn harness without a provider or run log: a port of
 * `tests/engine/_session_fakes.py`, plus the promise helpers the session tests share.
 */

import {
  ConversationInvocation,
  ConversationItem,
  ConversationRequest,
  type JsonObject,
} from "../../src/memory/contracts.ts";
import { type Session, SessionMemory, type Turn } from "../../src/memory/session.ts";

/** A signal that never aborts, for Python's `check_stopped=lambda: None`. */
export const NEVER: AbortSignal = new AbortController().signal;

/** One fake-format request for invocation `[runId, sequence]`. */
export function request(
  runId = "a",
  sequence = 1,
  consumer: readonly string[] = ["root", "agent"],
  compatibility?: JsonObject,
): ConversationRequest {
  return new ConversationRequest(
    consumer,
    new ConversationInvocation(runId, sequence),
    compatibility ?? { format: "fake-v1" },
  );
}

/** One fake-format message item. */
export function item(itemId: string, text: string, role = "assistant"): ConversationItem {
  return new ConversationItem(itemId, "fake-v1", { role, content: text });
}

/** The current snapshot of `memory`, read through a fresh run. */
export function snapshot(memory: SessionMemory): JsonObject {
  const opened = memory.openRun();
  try {
    return opened.snapshot;
  } finally {
    opened.close();
  }
}

/** One open run of a memory; `frame` is the session, as Python's harness names it. */
export interface Harness {
  memory: SessionMemory;
  session: Session;
  frame: Session;
}

/** Open a run of `memory` (a fresh one by default), await `fn` with it, then close the run. */
export async function withHarness<T>(
  fn: (harness: Harness) => T | Promise<T>,
  memory: SessionMemory = new SessionMemory(),
): Promise<T> {
  const session = memory.openRun();
  try {
    return await fn({ memory, session, frame: session });
  } finally {
    session.close();
  }
}

/** Persist one model request and its completed reply item. */
export function appendReply(turn: Turn, itemId: string, text: string): void {
  const ordinal = turn.request("model", { prompt: text });
  turn.observe(ordinal, "completed", { items: [item(itemId, text)] });
}

/**
 * Python's `with source.open_turn(request) as turn:` block: open the turn, await `fn`, then close
 * it with the thrown value (rethrown) or normally.
 */
export async function withTurn<T>(
  source: Session,
  conversationRequest: ConversationRequest,
  fn: (turn: Turn) => T | Promise<T>,
  signal: AbortSignal = NEVER,
): Promise<T> {
  const turn = await source.openTurn(conversationRequest, { signal });
  let result: T;
  try {
    result = await fn(turn);
  } catch (error) {
    turn.close(error);
    throw error;
  }
  turn.close();
  return result;
}

/** A promise with its outcome recorded as it settles. */
export interface Settled<T> {
  readonly promise: Promise<T>;
  settled: boolean;
  value?: T;
  error?: unknown;
}

/**
 * Track `promise`, attaching its handlers at once, since Bun fails a test when a promise rejects
 * before a handler is attached.
 */
export function settle<T>(promise: Promise<T>): Settled<T> {
  const state: Settled<T> = { promise, settled: false };
  promise.then(
    (value) => {
      state.settled = true;
      state.value = value;
    },
    (error: unknown) => {
      state.settled = true;
      state.error = error;
    },
  );
  return state;
}
