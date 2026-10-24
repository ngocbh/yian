/**
 * The inbox and call state a run's records build. One fold maintains it for the reducer,
 * `inspectRun` and the run index, so each reads the same waits from the same records; run status
 * stays with the run scan (`runtime/run_index.ts`).
 *
 * Knows about: runtime/log (record types, record times), events (`CloseReason`), nodes/start (the
 * START id).
 * Never imports: runtime/state, engine, run.
 */

import { StartNode } from "../nodes/start/node.ts";
import {
  InboxClosedRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
  type TakeCommit,
} from "./log.ts";

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

/** The waits state a run's records build: the waiting set, the registered calls and the inboxes. */
export class WaitsState {
  /** The waiting nodes, in `node_waiting` order. */
  readonly waiting = new Map<string, WaitingNode>();
  /** The registered calls, in registration order; kept until their node settles. */
  readonly calls = new Map<string, Call>();
  /** The inboxes, created on their first send, take wait or close. */
  readonly inboxes = new Map<string, InboxState>();
  /** Pending interrupts by node; declared so the serialized form is stable before anything sets it. */
  readonly interrupts = new Map<string, number>();
  /**
   * The wire kind of the last frontier record (start, wake, wait, resume, stop, run_waiting,
   * outcome); `null` before any.
   */
  lastFrontier: string | null = null;
  /** @internal Whether the run-start input commit was read. */
  _inputsCommitted = false;
  /** @internal Each unsettled node's last `node_woken` seq. */
  readonly _lastWoken = new Map<string, number>();
  /** @internal Each in-flight node's take commit that no success has accepted yet. */
  readonly _pendingTakes = new Map<string, { readonly seq: number } & TakeCommit>();
}

/** The prefix of a send's pool key: the START's id and a dot, then the inbox's name. */
const SEND_PREFIX = `${StartNode.ID}.`;

/** The wire kinds that move the frontier, so `lastFrontier` names the last one. */
const FRONTIER_KINDS: ReadonlyMap<Function, string> = new Map<Function, string>([
  [NodeStartedRecord, "node_started"],
  [NodeWaitingRecord, "node_waiting"],
  [NodeWokenRecord, "node_woken"],
  [NodeSucceededRecord, "node_succeeded"],
  [NodeRoutedRecord, "node_routed"],
  [NodeExpandedRecord, "node_expanded"],
  [NodeFailedRecord, "node_failed"],
  [RunStoppedRecord, "run_stopped"],
  [RunResumedRecord, "run_resumed"],
  [RunWaitingRecord, "run_waiting"],
]);

/**
 * Fold one accepted record into `w`, in log order. Records only: the reducer validates each record
 * before it is folded, so the fold trusts what it reads and changes nothing for a record it has no
 * rule for. Of a step it reads only `kind`, `call_id`, `tool`, `tool_call_id` and `deadline`.
 *
 * @param envelope The record and its seq; a `RecordEnvelope` fits.
 */
export function applyWaitsRecord(w: WaitsState, envelope: { readonly seq: number; readonly record: RunRecord }): void {
  const { seq, record } = envelope;
  if (record instanceof PoolCommitRecord) applyCommit(w, seq, record);
  else if (record instanceof NodeStepRecord) applyStep(w, seq, record);
  else if (record instanceof NodeWaitingRecord) applyWaiting(w, seq, record);
  else if (record instanceof NodeWokenRecord) applyWoken(w, seq, record);
  else if (record instanceof NodeSucceededRecord) applySuccess(w, record);
  else if (record instanceof NodeRoutedRecord) settle(w, record.nodeId);
  else if (record instanceof NodeExpandedRecord) settle(w, record.spawnerId);
  else if (record instanceof NodeFailedRecord) {
    closeOpenCalls(w, "failed");
    settle(w, record.failure.nodeId!);
  } else if (record instanceof RunResumedRecord) {
    // A resume drops every orphan commit, an inline take's among them, so its events are free again.
    w._pendingTakes.clear();
    for (const callId of record.closedCalls) close(w.calls.get(callId), "crashed");
  } else if (record instanceof InboxClosedRecord) applyClose(w, record.inbox);
  else if (record instanceof RunCancelledRecord) closeOpenCalls(w, "cancelled");
  else if (record instanceof RunFailedRecord) closeOpenCalls(w, "failed");
  const frontier = FRONTIER_KINDS.get(record.constructor);
  if (frontier !== undefined) w.lastFrontier = frontier;
}

/** Return `name`'s inbox, creating an empty open one. */
function inboxOf(w: WaitsState, name: string): InboxState {
  let inbox = w.inboxes.get(name);
  if (inbox === undefined) {
    inbox = { unread: [], closed: false, absorbing: [] };
    w.inboxes.set(name, inbox);
  }
  return inbox;
}

/** Remove `item` from `items` in place, if it is there. */
function remove<T>(items: T[], item: T): void {
  const index = items.indexOf(item);
  if (index >= 0) items.splice(index, 1);
}

/**
 * Apply a `pool_commit`: the run start's input commit, a send (assigned to the oldest absorbing
 * take, else unread), or an inline take commit, held until its node's success.
 */
function applyCommit(w: WaitsState, seq: number, record: PoolCommitRecord): void {
  if (!w._inputsCommitted) {
    // The run start's commits come first; the START inputs' is its last.
    if (record.key === StartNode.ID) w._inputsCommitted = true;
    return;
  }
  if (record.take !== null) {
    w._pendingTakes.set(record.nodeId, { seq, ...record.take });
    return;
  }
  if (!record.key.startsWith(SEND_PREFIX)) return;
  const inbox = inboxOf(w, record.key.slice(SEND_PREFIX.length));
  const takeId = inbox.absorbing[0];
  const take = takeId === undefined ? undefined : w.calls.get(takeId);
  if (take === undefined) {
    inbox.unread.push(seq);
    return;
  }
  take.assigned.push(seq);
  take.state = "answered";
  if (take.mode === "one") inbox.absorbing.shift();
}

/** Apply a node step: a `pending` step registers a host call, a `tool_result` answers one. */
function applyStep(w: WaitsState, seq: number, record: NodeStepRecord): void {
  const step = record.step;
  const callId = step["call_id"];
  const toolCallId = step["tool_call_id"];
  if (step["kind"] === "pending") {
    const tool = step["tool"];
    const deadline = step["deadline"];
    if (typeof callId !== "string" || typeof tool !== "string") return;
    w.calls.set(callId, {
      callId,
      nodeId: record.nodeId,
      tool,
      resultKey: typeof toolCallId === "string" ? toolCallId : callId,
      inbox: null,
      mode: null,
      deadline: typeof deadline === "string" ? deadline : null,
      seq,
      pendingSeq: seq,
      state: "open",
      resultSeq: null,
      assigned: [],
      closedInbox: false,
      closedReason: null,
    });
  } else if (step["kind"] === "tool_result") {
    const key = typeof toolCallId === "string" ? toolCallId : callId;
    // A closer's result, or an agent result no pending step registered, answers nothing.
    for (const call of w.calls.values()) {
      if (call.nodeId === record.nodeId && call.resultKey === key && call.state === "open") {
        call.state = "answered";
        call.resultSeq = seq;
        return;
      }
    }
  }
}

/** Apply `node_waiting`: register its take or sleep (a take joins its inbox's absorbing takes) and move the node to the waiting set. */
function applyWaiting(w: WaitsState, seq: number, record: NodeWaitingRecord): void {
  for (const entry of record.waits) {
    w.calls.set(entry.callId, {
      callId: entry.callId,
      nodeId: record.nodeId,
      tool: entry.tool,
      resultKey: entry.callId,
      inbox: entry.inbox,
      mode: entry.mode,
      deadline: entry.deadline,
      seq,
      pendingSeq: null,
      state: "open",
      resultSeq: null,
      assigned: [],
      closedInbox: false,
      closedReason: null,
    });
    if (entry.tool === "take") inboxOf(w, entry.inbox!).absorbing.push(entry.callId);
  }
  w.waiting.set(record.nodeId, { seq, callIds: record.callIds, steerInbox: record.steerInbox });
}

/** Apply `node_woken`: time out its calls (a host call closes `timed_out`), end its takes' absorbing and move it back in flight. */
function applyWoken(w: WaitsState, seq: number, record: NodeWokenRecord): void {
  for (const callId of record.timedOut) {
    const call = w.calls.get(callId);
    if (call === undefined || call.state !== "open") continue;
    if (call.tool === "take" || call.tool === "sleep") call.state = "timed_out";
    else close(call, "timed_out");
  }
  for (const call of nodeCalls(w, record.nodeId)) {
    if (call.inbox !== null) remove(inboxOf(w, call.inbox).absorbing, call.callId);
  }
  w.waiting.delete(record.nodeId);
  w._lastWoken.set(record.nodeId, seq);
}

/** Apply `node_succeeded`: an inline take's events leave `unread`; then the node settles. */
function applySuccess(w: WaitsState, record: NodeSucceededRecord): void {
  const take = w._pendingTakes.get(record.nodeId);
  // A waiting take's events were assigned, never unread; only an inline take's are taken here.
  const waited = nodeCalls(w, record.nodeId).some((call) => call.tool === "take");
  if (take !== undefined && take.seq === record.commit && !waited) {
    const unread = inboxOf(w, take.inbox).unread;
    for (const event of take.events) remove(unread, event);
  }
  settle(w, record.nodeId);
}

/** Apply `inbox_closed`: absorbing takes with nothing assigned are answered by the close; none absorbs after it. */
function applyClose(w: WaitsState, name: string): void {
  const inbox = inboxOf(w, name);
  inbox.closed = true;
  for (const callId of inbox.absorbing) {
    const call = w.calls.get(callId);
    if (call !== undefined && call.assigned.length === 0) {
      call.state = "answered";
      call.closedInbox = true;
    }
  }
  inbox.absorbing.length = 0;
}

/** Close `call` with `reason` when it is open. */
function close(call: Call | undefined, reason: CloseReason): void {
  if (call === undefined || call.state !== "open") return;
  call.state = "closed";
  call.closedReason = reason;
}

/** Close every open call with `reason`. */
function closeOpenCalls(w: WaitsState, reason: CloseReason): void {
  for (const call of w.calls.values()) close(call, reason);
}

/** Forget a settled node's calls and bookkeeping. */
function settle(w: WaitsState, nodeId: string): void {
  for (const call of nodeCalls(w, nodeId)) {
    if (call.inbox !== null) remove(inboxOf(w, call.inbox).absorbing, call.callId);
    w.calls.delete(call.callId);
  }
  w.waiting.delete(nodeId);
  w._lastWoken.delete(nodeId);
  w._pendingTakes.delete(nodeId);
  w.interrupts.delete(nodeId);
}

/** Return `nodeId`'s registered calls, in registration order. */
export function nodeCalls(w: WaitsState, nodeId: string): readonly Call[] {
  return [...w.calls.values()].filter((call) => call.nodeId === nodeId);
}

/** Return whether `nodeId` waits and every call it waits on is settled, or an interrupt names it. */
export function isWakeable(w: WaitsState, nodeId: string): boolean {
  const waiting = w.waiting.get(nodeId);
  if (waiting === undefined) return false;
  if (w.interrupts.has(nodeId)) return true;
  return waiting.callIds.every((callId) => {
    const call = w.calls.get(callId);
    return call !== undefined && call.state !== "open";
  });
}

/** Return the wakeable nodes, oldest `node_waiting` first. */
export function wakeableNodes(w: WaitsState): readonly string[] {
  return [...w.waiting.keys()].filter((nodeId) => isWakeable(w, nodeId));
}

/** Return each inbox's unread send seqs, oldest first; an inbox with none is left out. */
export function unreadEvents(w: WaitsState): ReadonlyMap<string, readonly number[]> {
  return new Map([...w.inboxes].filter(([, inbox]) => inbox.unread.length > 0).map(([name, inbox]) => [name, [...inbox.unread]]));
}

/** Return `inbox`'s unread send seqs that no pending take commit holds: those a new take may read. */
export function availableUnread(w: WaitsState, inbox: string): readonly number[] {
  const held = new Set([...w._pendingTakes.values()].flatMap((take) => take.events));
  return (w.inboxes.get(inbox)?.unread ?? []).filter((seq) => !held.has(seq));
}

/** Return a call's deadline in epoch milliseconds, or `null` when it has none. */
function deadlineMs(call: Call): number | null {
  return call.deadline === null ? null : Date.parse(call.deadline);
}

/**
 * Return, by waiting node in waiting order, the open calls whose deadline is at or before
 * `nowMs`, for each node whose other calls are all settled, so timing these out makes it wakeable.
 */
export function dueTimeouts(w: WaitsState, nowMs: number): ReadonlyMap<string, readonly string[]> {
  const due = new Map<string, readonly string[]>();
  for (const [nodeId, waiting] of w.waiting) {
    const open = waiting.callIds.map((callId) => w.calls.get(callId)).filter((call) => call?.state === "open") as Call[];
    const past = open.filter((call) => {
      const ms = deadlineMs(call);
      return ms !== null && ms <= nowMs;
    });
    if (past.length > 0 && past.length === open.length) due.set(nodeId, past.map((call) => call.callId));
  }
  return due;
}

/** Return the earliest deadline of an open call of a waiting node, or `null` when none has one. */
export function wakeAt(w: WaitsState): string | null {
  let earliest: string | null = null;
  for (const waiting of w.waiting.values()) {
    for (const callId of waiting.callIds) {
      const call = w.calls.get(callId);
      // Record times have one fixed-width form, so text order is time order.
      if (call?.state === "open" && call.deadline !== null && (earliest === null || call.deadline < earliest)) {
        earliest = call.deadline;
      }
    }
  }
  return earliest;
}

/** Return the calls of each waiting node, in waiting order, then the order the node names them. */
export function waitInfos(w: WaitsState): readonly WaitInfo[] {
  const infos: WaitInfo[] = [];
  for (const [node, waiting] of w.waiting) {
    for (const callId of waiting.callIds) {
      const call = w.calls.get(callId);
      infos.push({
        node,
        callId,
        tool: call?.tool ?? "",
        inbox: call?.inbox ?? null,
        deadline: call?.deadline ?? null,
        settled: call !== undefined && call.state !== "open",
      });
    }
  }
  return infos;
}
