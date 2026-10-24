/**
 * The inbox, call and run-status state a run's records build. One fold maintains it for the
 * reducer, `inspectRun` and the run index, so each reads the same waits from the same records.
 *
 * Knows about: runtime/log (record types), events (`CloseReason`).
 * Never imports: runtime/state, engine, run.
 */

import type { TakeCommit } from "./log.ts";

export { CLOSE_REASONS, type CloseReason } from "../events.ts";
import type { CloseReason } from "../events.ts";

/** Where a registered call stands. */
export type CallStateKind = "open" | "answered" | "timed_out" | "closed";

/** One call a waiting or in-flight node registered: a host call, a take or a sleep. */
export interface Call {
  /** The call's id. */
  readonly callId: string;
  /** The node that registered it. */
  readonly nodeId: string;
  /** `"take"`, `"sleep"`, or the host tool's id (`"ask"`, ...). */
  readonly tool: string;
  /** The key the call's answer is stored under: the agent's tool call id, else the call id. */
  readonly resultKey: string;
  /** The inbox a take reads; `null` for any other call. */
  readonly inbox: string | null;
  /** A take's mode; `null` for any other call. */
  readonly mode: "one" | "all" | null;
  /** When the call times out, as a record time; `null` when it doesn't. */
  readonly deadline: string | null;
  /** The seq of the record that registered it: its `pending` step or its `node_waiting`. */
  readonly seq: number;
  /** A host call's `pending` step seq; `null` for a take or sleep. */
  readonly pendingSeq: number | null;
  state: CallStateKind;
  /** The seq of the `tool_result` step that answered it, or `null`. */
  resultSeq: number | null;
  /** A take's assigned send seqs, in assignment order. */
  assigned: number[];
  /** Whether a take was answered by its inbox closing with nothing assigned. */
  closedInbox: boolean;
  /** Why the call was closed, when its state is `closed`. */
  closedReason: CloseReason | null;
}

/** One inbox's queue: unread send seqs, whether it is closed, and the takes absorbing its sends. */
export interface InboxState {
  /** Send seqs no take has been assigned, oldest first. */
  readonly unread: number[];
  closed: boolean;
  /** The ids of the open takes sends go to, oldest first. */
  readonly absorbing: string[];
}

/** A node in the waiting set. */
export interface WaitingNode {
  /** The seq of its `node_waiting`. */
  readonly seq: number;
  /** The calls it waits on. */
  readonly callIds: readonly string[];
  /** Its steer inbox; always null until agents wait on one. */
  readonly steerInbox: string | null;
}

/** One call a host can see the run waiting on. */
export interface WaitInfo {
  readonly node: string;
  readonly callId: string;
  readonly tool: string;
  readonly inbox: string | null;
  readonly deadline: string | null;
  /** Whether the call is already answered, timed out or closed, so its node is about to wake. */
  readonly settled: boolean;
}

/**
 * The waits state a run's records build: the waiting set, the registered calls, the inboxes, and
 * the records-only facts a reader needs for the run's status.
 */
export class WaitsState {
  /** The waiting nodes, in `node_waiting` order. */
  readonly waiting = new Map<string, WaitingNode>();
  /** The registered calls, in registration order; kept until their node settles. */
  readonly calls = new Map<string, Call>();
  /** The inboxes, created on their first send, take wait or close. */
  readonly inboxes = new Map<string, InboxState>();
  /** Interrupts by inbox; declared so the serialized form is stable before anything sets it. */
  readonly interrupts = new Map<string, number>();
  /**
   * The wire kind of the last frontier record (start, wake, wait, resume, stop, run_waiting,
   * outcome); `null` before any.
   */
  lastFrontier: string | null = null;
  /** The in-flight nodes and the seq that put each in flight (`node_started` or `node_woken`), in order. */
  readonly starts = new Map<string, number>();
  /** The seq of the last `run_stopped` since the last `run_resumed`, or `null`. */
  lastStop: number | null = null;
  /** Whether END succeeded. */
  endSucceeded = false;
  /** Whether a route made the terminal output unreachable. */
  unreachable = false;
  /** Whether a node failed. */
  nodeFailed = false;
  /** Whether a `run_failed` ended the run. */
  runFailed = false;
  /** Whether a `run_cancelled` ended the run. */
  cancelled = false;
  /** @internal Whether the run-start input commit was read. */
  _inputsCommitted = false;
  /** @internal Each unsettled node's last `node_woken` seq. */
  readonly _lastWoken = new Map<string, number>();
  /** @internal Each in-flight node's take commit that no success has accepted yet. */
  readonly _pendingTakes = new Map<string, { readonly seq: number } & TakeCommit>();
}
