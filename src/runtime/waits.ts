/**
 * The inbox and call state a run's records build. One fold maintains it for the reducer,
 * `inspectRun` and the run index, so each reads the same waits from the same records; run status
 * stays with the run scan (`runtime/run_index.ts`).
 *
 * Knows about: runtime/log (record types, record times, `by`), events (`CloseReason`), nodes/base
 * (`WaitRequest`), nodes/contracts (`CALL_STATES`), nodes/start and nodes/end (the START and END
 * ids), memory/base (`Author`), diagnostics.
 */

import { reprText } from "../diagnostics.ts";
import type { Author } from "../memory/base.ts";
import type { WaitRequest } from "../nodes/base.ts";
import { CALL_STATES } from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import { YianRangeError } from "../typesys/errors.ts";
import {
  formatRecordTime,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
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
  recordTimeMs,
  requireRecordTime,
  type TakeCommit,
  type TakeInbox,
  takeInboxes,
  validatedBy,
  type WaitEntry,
} from "./log.ts";

import { CLOSE_REASONS, type CloseReason } from "../events.ts";

/** Where a registered call stands (`CALL_STATES`). */
export type CallStateKind = (typeof CALL_STATES)[number];

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
  /** The inbox a take reads, or the list its `from:` names; `null` for any other call. */
  readonly inbox: TakeInbox | null;
  /** A take's mode; `null` for any other call. */
  readonly mode: "one" | "all" | null;
  /** When the call times out, as a record time; `null` when it doesn't. */
  readonly deadline: string | null;
  /** The seq of the record that registered it: its `pending` step or its `node_waiting`. */
  readonly seq: number;
  /** A host call's `pending` step seq; `null` for a take or sleep. */
  readonly pendingSeq: number | null;
  /**
   * Whether it is an AGENT's call of a flow it grants as a tool, which the run answers from the
   * flow's scope, never a host.
   */
  readonly flow: boolean;
  /**
   * A flow call's scope, `<node>/<flow>#<k>`, whose END's success answers it; `null` for any
   * other call.
   */
  readonly scope: string | null;
  /**
   * Whether a host's check held the agent's call before its tool ran (its `pending` step says
   * `held: true`): the reply's later calls wait for it, and its settling wakes its node alone.
   */
  readonly held: boolean;
  state: CallStateKind;
  /** The seq of the `tool_result` step that answered it, or `null`. */
  resultSeq: number | null;
  /** A take's assigned send seqs, in assignment order. */
  assigned: number[];
  /** The inbox of each of `assigned`, in step with it. */
  assignedInboxes: string[];
  /** Whether a take was answered by its inbox (each of its inboxes) closing with nothing assigned. */
  closedInbox: boolean;
  /** Why the call was closed, when its state is `closed`. */
  closedReason: CloseReason | null;
  /** The seq of the record that closed it, when its state is `closed`. */
  closedSeq: number | null;
  /**
   * The seq of the `released` step that released it, kept once it registers again or settles;
   * `null` for a call never released.
   */
  releasedSeq: number | null;
  /** The `by` of the `released` step that released it, kept as `releasedSeq` is. */
  releasedBy: Author | null;
  /**
   * The `by` of the `tool_result` that answered it, which a host's `complete` records; `null`
   * while it is unanswered, or when its answer holds none.
   */
  answeredBy: Author | null;
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
  /** The steer inbox it opened, which an interrupt of that inbox reaches; `null` for none. */
  readonly steerInbox: string | null;
}

/** A remembered Stop on an inbox. */
export interface StopState {
  /** The seq of the host's `interrupted` record that left it, which the records the run applies keep. */
  readonly seq: number;
  /** Whether it has stopped an agent: an `interrupted` record on the inbox named one. */
  readonly stopped: boolean;
  /** Who made it: the `by` of the host's `interrupted` record, which the records the run applies copy. */
  readonly by: Author | null;
}

/** One call a host can see the run waiting on. */
export interface WaitInfo {
  readonly node: string;
  readonly callId: string;
  readonly tool: string;
  /** The inbox a take reads, or the list its `from:` names; `null` for any other call. */
  readonly inbox: TakeInbox | null;
  readonly deadline: string | null;
  /** Whether the call is already answered, timed out, closed or released, so its node is about to wake. */
  readonly settled: boolean;
  /** Whether a host's check held the agent's call before its tool ran (`Call.held`). */
  readonly held: boolean;
}

/** The waits state a run's records build: the waiting set, the registered calls and the inboxes. */
export class WaitsState {
  /** The waiting nodes, in `node_waiting` order. */
  readonly waiting = new Map<string, WaitingNode>();
  /** The registered calls, in registration order; kept until their node settles. */
  readonly calls = new Map<string, Call>();
  /** The inboxes, created on their first send, take wait or close. */
  readonly inboxes = new Map<string, InboxState>();
  /** Pending interrupts by node: the seq of the `interrupted` record each named node hasn't honored yet. */
  readonly interrupts = new Map<string, number>();
  /**
   * The remembered Stop by inbox: set by a standing `interrupted` record, ended by the next pickup
   * from the inbox (a take of it waits, a waiting take is answered or times out, an inline take
   * commits, a withdraw reopens a take), by the first `run_waiting` once it has stopped an agent,
   * or by run end. A steer reader that opens the inbox while it stands is stopped.
   */
  readonly stops = new Map<string, StopState>();
  /**
   * The wire kind of the last frontier record (start, wake, wait, resume, stop, run_waiting,
   * outcome); `null` before any.
   */
  lastFrontier: string | null = null;
  /** @internal Whether the run-start input commit was read. */
  _inputsCommitted = false;
  /** @internal Each unsettled node's last `node_woken` seq. */
  readonly _lastWoken = new Map<string, number>();
  /**
   * @internal Each in-flight node's take commit that no success has accepted yet. A resume keeps
   * it, so the node, run again, takes its events back (`reclaimedTake`).
   */
  readonly _pendingTakes = new Map<string, { readonly seq: number } & TakeCommit>();
  /** @internal Each unsettled node's latest `model_call` step seq: the calls registered after it are its latest reply's. */
  readonly _replies = new Map<string, number>();
  /** @internal The unsettled nodes that wrote an `interrupted` step, honoring an interrupt; their success says `interrupted`. */
  readonly _honored = new Set<string>();
  /** @internal Each unsettled node an interrupt named: that interrupt's inbox, held through its honor. */
  readonly _interruptInboxes = new Map<string, string>();
  /**
   * @internal Each in-flight node a record the run applied for a Stop named, while it holds that
   * interrupt: the Stop's seq (`StopState.seq`).
   */
  readonly _appliedStops = new Map<string, number>();

  /** Return the plain JSON form, each map as its `[key, value]` entries in order (`fromJSON` reads it back). */
  toJSON(): WaitsJson {
    return {
      waiting: [...this.waiting],
      calls: [...this.calls],
      inboxes: [...this.inboxes],
      interrupts: [...this.interrupts],
      stops: [...this.stops],
      lastFrontier: this.lastFrontier,
      inputsCommitted: this._inputsCommitted,
      lastWoken: [...this._lastWoken],
      pendingTakes: [...this._pendingTakes],
      replies: [...this._replies],
      honored: [...this._honored],
      interruptInboxes: [...this._interruptInboxes],
      appliedStops: [...this._appliedStops],
    };
  }

  /**
   * Return the state `json` (a `toJSON` form, read back by `JSON.parse` or `parseJson`) holds,
   * with fresh copies of its parts, keys in `toJSON`'s order. Ints may be numbers or `bigint`s.
   *
   * @throws RangeError when `json` isn't such a form.
   */
  static fromJSON(json: unknown): WaitsState {
    if (!isObject(json)) throw new YianRangeError("waits JSON must be an object");
    const keys = Object.keys(json);
    if (keys.length !== WAITS_JSON_KEYS.length || !WAITS_JSON_KEYS.every((key) => json[key] !== undefined)) {
      throw new YianRangeError(
        "waits JSON must hold exactly waiting, calls, inboxes, interrupts, stops, lastFrontier, inputsCommitted, lastWoken, pendingTakes, replies, honored, interruptInboxes and appliedStops",
      );
    }
    const w = new WaitsState();
    for (const [node, waiting] of entries(json, "waiting")) w.waiting.set(node, waitingFromJson(waiting, `waiting ${reprText(node)}`));
    for (const [callId, call] of entries(json, "calls")) w.calls.set(callId, callFromJson(call, `calls ${reprText(callId)}`));
    for (const [name, inbox] of entries(json, "inboxes")) w.inboxes.set(name, inboxFromJson(inbox, `inboxes ${reprText(name)}`));
    for (const [node, seq] of entries(json, "interrupts")) w.interrupts.set(node, int(seq, `interrupts ${reprText(node)}`));
    for (const [name, stop] of entries(json, "stops")) w.stops.set(name, stopFromJson(stop, `stops ${reprText(name)}`));
    if (json.lastFrontier !== null && typeof json.lastFrontier !== "string") {
      throw new YianRangeError("waits JSON lastFrontier must be a string or null");
    }
    w.lastFrontier = json.lastFrontier;
    if (typeof json.inputsCommitted !== "boolean") throw new YianRangeError("waits JSON inputsCommitted must be a boolean");
    w._inputsCommitted = json.inputsCommitted;
    for (const [node, seq] of entries(json, "lastWoken")) w._lastWoken.set(node, int(seq, `lastWoken ${reprText(node)}`));
    for (const [node, take] of entries(json, "pendingTakes")) w._pendingTakes.set(node, takeFromJson(take, `pendingTakes ${reprText(node)}`));
    for (const [node, seq] of entries(json, "replies")) w._replies.set(node, int(seq, `replies ${reprText(node)}`));
    const honored = json.honored;
    if (!Array.isArray(honored) || !honored.every((node) => typeof node === "string")) {
      throw new YianRangeError("waits JSON honored must be an array of node ids");
    }
    for (const node of honored) w._honored.add(node);
    for (const [node, inbox] of entries(json, "interruptInboxes")) {
      if (typeof inbox !== "string") throw new YianRangeError(`waits JSON interruptInboxes ${reprText(node)} must be an inbox name`);
      w._interruptInboxes.set(node, inbox);
    }
    for (const [node, seq] of entries(json, "appliedStops")) w._appliedStops.set(node, int(seq, `appliedStops ${reprText(node)}`));
    return w;
  }
}

/** The keys of a `WaitsJson`, in `toJSON`'s order. */
const WAITS_JSON_KEYS = [
  "waiting",
  "calls",
  "inboxes",
  "interrupts",
  "stops",
  "lastFrontier",
  "inputsCommitted",
  "lastWoken",
  "pendingTakes",
  "replies",
  "honored",
  "interruptInboxes",
  "appliedStops",
] as const;

/** Return whether `value` is a plain JSON object (not an array or null). */
function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Return `value`, an int as a number or a safe `bigint`, as a number; `null` when it isn't one. */
function intOrNull(value: unknown): number | null {
  const n = typeof value === "bigint" ? Number(value) : value;
  return typeof n === "number" && Number.isSafeInteger(n) ? n : null;
}

/** Return the int `value` as a number. @throws RangeError naming `label` when it isn't one. */
function int(value: unknown, label: string): number {
  const n = intOrNull(value);
  if (n === null) throw new YianRangeError(`waits JSON ${label} must be an integer`);
  return n;
}

/** Return the ints of a list of seqs, as numbers, or `null` when `value` isn't an array of ints. */
export function seqList(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  const items = value.map(intOrNull);
  return items.every((item): item is number => item !== null) ? items : null;
}

/** Return the strings of `value`, or `null` when it isn't an array of strings. */
function texts(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? [...value] : null;
}

/** Return `items` when it holds at least one item, else `undefined`. */
function nonEmpty(items: string[] | null): string[] | undefined {
  return items !== null && items.length > 0 ? items : undefined;
}

/** Return the `[key, value]` entries of the map field `name`. @throws RangeError when they aren't. */
function entries(json: Record<string, unknown>, name: string): [string, unknown][] {
  const value = json[name];
  if (!Array.isArray(value) || !value.every((entry) => Array.isArray(entry) && entry.length === 2 && typeof entry[0] === "string")) {
    throw new YianRangeError(`waits JSON ${name} must be an array of [key, value] entries`);
  }
  return value as [string, unknown][];
}

/** Return a stored waiting node. @throws RangeError when it isn't one. */
function waitingFromJson(value: unknown, label: string): WaitingNode {
  const seq = isObject(value) ? intOrNull(value["seq"]) : null;
  const callIds = isObject(value) ? texts(value["callIds"]) : null;
  const steer = isObject(value) ? value["steerInbox"] : undefined;
  if (seq === null || callIds === null || (steer !== null && typeof steer !== "string")) {
    throw new YianRangeError(`waits JSON ${label} must be a waiting node`);
  }
  return { seq, callIds, steerInbox: steer };
}

/** Return a stored call. @throws RangeError when it isn't one. */
function callFromJson(value: unknown, label: string): Call {
  const bad = () => new YianRangeError(`waits JSON ${label} must be a call`);
  if (!isObject(value)) throw bad();
  const text = (key: string) => (typeof value[key] === "string" ? (value[key] as string) : undefined);
  const textOrNull = (key: string) => (value[key] === null ? null : text(key));
  const intOrNullField = (key: string) => (value[key] === null ? null : (intOrNull(value[key]) ?? undefined));
  const call = {
    callId: text("callId"),
    nodeId: text("nodeId"),
    tool: text("tool"),
    resultKey: text("resultKey"),
    inbox: value["inbox"] === null ? null : (text("inbox") ?? nonEmpty(texts(value["inbox"]))),
    mode: textOrNull("mode"),
    deadline: textOrNull("deadline"),
    seq: intOrNull(value["seq"]) ?? undefined,
    pendingSeq: intOrNullField("pendingSeq"),
    state: text("state"),
    resultSeq: intOrNullField("resultSeq"),
    assigned: seqList(value["assigned"]) ?? undefined,
    assignedInboxes: texts(value["assignedInboxes"]) ?? undefined,
    closedInbox: typeof value["closedInbox"] === "boolean" ? value["closedInbox"] : undefined,
    closedReason: textOrNull("closedReason"),
    closedSeq: intOrNullField("closedSeq"),
    releasedSeq: intOrNullField("releasedSeq"),
    releasedBy: storedBy(value["releasedBy"]),
    answeredBy: storedBy(value["answeredBy"]),
    flow: typeof value["flow"] === "boolean" ? value["flow"] : undefined,
    scope: textOrNull("scope"),
    held: typeof value["held"] === "boolean" ? value["held"] : undefined,
  };
  if (
    Object.values(call).some((field) => field === undefined) ||
    (call.mode !== null && call.mode !== "one" && call.mode !== "all") ||
    call.assignedInboxes!.length !== call.assigned!.length ||
    !(CALL_STATES as readonly string[]).includes(call.state!) ||
    (call.closedReason !== null && !(CLOSE_REASONS as readonly string[]).includes(call.closedReason!)) ||
    // A closed call keeps the seq of the record that closed it, and a released one its release's.
    (call.state === "closed" && call.closedSeq === null) ||
    (call.state === "released" && call.releasedSeq === null)
  ) {
    throw bad();
  }
  return call as Call;
}

/** Return a stored inbox. @throws RangeError when it isn't one. */
function inboxFromJson(value: unknown, label: string): InboxState {
  const unread = isObject(value) ? seqList(value["unread"]) : null;
  const absorbing = isObject(value) ? texts(value["absorbing"]) : null;
  const closed = isObject(value) ? value["closed"] : undefined;
  if (unread === null || absorbing === null || typeof closed !== "boolean") throw new YianRangeError(`waits JSON ${label} must be an inbox`);
  return { unread, closed, absorbing };
}

/** Return a stored remembered Stop. @throws RangeError when it isn't one. */
function stopFromJson(value: unknown, label: string): StopState {
  const seq = isObject(value) ? intOrNull(value["seq"]) : null;
  const stopped = isObject(value) ? value["stopped"] : undefined;
  const by = isObject(value) ? storedBy(value["by"]) : undefined;
  if (seq === null || typeof stopped !== "boolean" || by === undefined) throw new YianRangeError(`waits JSON ${label} must be a stop`);
  return { seq, stopped, by };
}

/** Return a stored `by` (`validatedBy`), `null` included; `undefined` when it isn't one. */
function storedBy(value: unknown): Author | null | undefined {
  try {
    return value === undefined ? undefined : validatedBy(value, "by");
  } catch {
    return undefined;
  }
}

/** Return a stored pending take commit. @throws RangeError when it isn't one. */
function takeFromJson(value: unknown, label: string): { readonly seq: number } & TakeCommit {
  const seq = isObject(value) ? intOrNull(value["seq"]) : null;
  const events = isObject(value) ? seqList(value["events"]) : null;
  const inbox = isObject(value) ? value["inbox"] : undefined;
  const closed = isObject(value) ? value["closed"] : undefined;
  // One inbox closes as a bool; a list names its closed inboxes.
  const one = typeof inbox === "string" && typeof closed === "boolean";
  const listed = nonEmpty(texts(inbox)) !== undefined && texts(closed) !== null;
  if (seq === null || events === null || !(one || listed)) throw new YianRangeError(`waits JSON ${label} must be a take commit`);
  return { seq, inbox: inbox as TakeInbox, events, closed: closed as TakeCommit["closed"] };
}

/** The plain JSON form of a `WaitsState`. */
export interface WaitsJson {
  readonly waiting: readonly (readonly [string, WaitingNode])[];
  readonly calls: readonly (readonly [string, Call])[];
  readonly inboxes: readonly (readonly [string, InboxState])[];
  readonly interrupts: readonly (readonly [string, number])[];
  readonly stops: readonly (readonly [string, StopState])[];
  readonly lastFrontier: string | null;
  readonly inputsCommitted: boolean;
  readonly lastWoken: readonly (readonly [string, number])[];
  readonly pendingTakes: readonly (readonly [string, { readonly seq: number } & TakeCommit])[];
  readonly replies: readonly (readonly [string, number])[];
  readonly honored: readonly string[];
  readonly interruptInboxes: readonly (readonly [string, string])[];
  readonly appliedStops: readonly (readonly [string, number])[];
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
 * rule for. Of a step it reads only `kind`, `call_id`, `tool`, `tool_call_id`, `deadline`, `flow`,
 * `scope`, `held`, `closed`, `inbox` and `from`, and its record's `by`. A released call registered
 * again keeps its release.
 *
 * @param envelope The record and its seq; a `RecordEnvelope` fits.
 * @throws RangeError when a `pending` step's `deadline` is set but isn't a record time.
 * @throws TypeError when it is set but isn't a string.
 */
export function applyWaitsRecord(w: WaitsState, envelope: { readonly seq: number; readonly record: RunRecord }): void {
  const { seq, record } = envelope;
  if (record instanceof PoolCommitRecord) applyCommit(w, seq, record);
  else if (record instanceof NodeStepRecord) applyStep(w, seq, record);
  else if (record instanceof NodeWaitingRecord) applyWaiting(w, seq, record);
  else if (record instanceof NodeWokenRecord) applyWoken(w, seq, record);
  else if (record instanceof NodeSucceededRecord) applySuccess(w, record);
  else if (record instanceof NodeRoutedRecord) settle(w, record.nodeId);
  // A flow call's growth leaves its agent in flight.
  else if (record instanceof NodeExpandedRecord) {
    if (record.callId === null) settle(w, record.spawnerId);
  }
  else if (record instanceof NodeFailedRecord) {
    if (record.scope !== null) {
      // A failure inside a flow call's scope fails the call and ends the scope; the run goes on.
      close(w.calls.get(record.callId!), "failed", seq);
      endScope(w, record.scope);
    } else {
      closeOpenCalls(w, "failed", seq);
      settle(w, record.failure.nodeId!);
      // A node failure ends the run.
      w.stops.clear();
    }
  } else if (record instanceof RunWaitingRecord) {
    // The reply a Stop targeted is stopped; a Stop that stopped no one yet (a mid-turn ask or sleep) still stands.
    for (const [name, stop] of w.stops) if (stop.stopped) w.stops.delete(name);
  } else if (record instanceof RunResumedRecord) {
    // An orphan take commit still holds its events: its node runs again and takes them back, and no
    // other take reads them first, as without the crash.
    for (const callId of record.closedCalls) close(w.calls.get(callId), "crashed", seq);
  } else if (record instanceof InboxClosedRecord) applyClose(w, record.inbox);
  else if (record instanceof InboxWithdrawnRecord) applyWithdrawn(w, record);
  else if (record instanceof InterruptedRecord) applyInterrupted(w, seq, record);
  else if (record instanceof RunCancelledRecord) {
    closeOpenCalls(w, "cancelled", seq);
    w.stops.clear();
  } else if (record instanceof RunFailedRecord) {
    closeOpenCalls(w, "failed", seq);
    w.stops.clear();
  }
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

/** Return whether inbox `name` is closed. */
function isClosed(w: WaitsState, name: string): boolean {
  return w.inboxes.get(name)?.closed ?? false;
}

/** End the remembered Stop on each inbox of `inbox`: a take picked up from it. */
function pickup(w: WaitsState, inbox: TakeInbox): void {
  for (const name of takeInboxes(inbox)) w.stops.delete(name);
}

/** Take `call` out of the absorbing takes of each inbox it reads. */
function stopAbsorbing(w: WaitsState, call: Call): void {
  if (call.inbox !== null) for (const name of takeInboxes(call.inbox)) remove(inboxOf(w, name).absorbing, call.callId);
}

/** Assign `call` the send `seq` of inbox `name`, after those it holds. */
function hold(call: Call, seq: number, name: string): void {
  call.assigned.push(seq);
  call.assignedInboxes.push(name);
}

/** Return the unread events of inboxes `names` a new take may read (`availableUnread`), each with its inbox, in seq order. */
export function availableIn(w: WaitsState, names: readonly string[]): (readonly [number, string])[] {
  return names.flatMap((name) => availableUnread(w, name).map((seq) => [seq, name] as const)).sort(([a], [b]) => a - b);
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
    // An inline take is a pickup from the inbox; a waited take's commit isn't (its wait or answer was).
    if (!nodeCalls(w, record.nodeId).some((call) => call.tool === "take")) pickup(w, record.take.inbox);
    return;
  }
  if (!record.key.startsWith(SEND_PREFIX)) return;
  const name = record.key.slice(SEND_PREFIX.length);
  if (!assign(w, name, seq)) inboxOf(w, name).unread.push(seq);
}

/**
 * Assign the event `seq` of inbox `name` to its oldest absorbing take, which stops absorbing (from
 * each inbox it reads) after one event for `take: one`; return whether a take was absorbing.
 */
function assign(w: WaitsState, name: string, seq: number): boolean {
  const takeId = inboxOf(w, name).absorbing[0];
  const take = takeId === undefined ? undefined : w.calls.get(takeId);
  if (take === undefined) return false;
  // Answering a waiting take is a pickup; a second event for an answered `take: all` isn't.
  if (take.state === "open") pickup(w, take.inbox!);
  hold(take, seq, name);
  take.state = "answered";
  if (take.mode === "one") stopAbsorbing(w, take);
  return true;
}

/**
 * Offer the unread events of inboxes `names` no pending take commit holds, oldest first, to their
 * absorbing takes, as each would have gone had it been sent now: events a take commit no success
 * accepted held are free again once its node settles, and a take that waited meanwhile is first in
 * line.
 */
function reoffer(w: WaitsState, names: readonly string[]): void {
  for (const [seq, name] of availableIn(w, names)) if (assign(w, name, seq)) remove(inboxOf(w, name).unread, seq);
}

/**
 * Apply a node step: a `pending` step registers a host call, a `tool_result` answers one, and a
 * `released` step releases a held one.
 */
function applyStep(w: WaitsState, seq: number, record: NodeStepRecord): void {
  const step = record.step;
  const callId = step["call_id"];
  const toolCallId = step["tool_call_id"];
  if (step["kind"] === "pending") {
    const tool = step["tool"];
    if (typeof callId !== "string" || typeof tool !== "string") return;
    // `wakeAt` compares deadlines as text, which is time order only for record times.
    const deadline = step["deadline"] == null ? null : requireRecordTime(step["deadline"], `pending step ${reprText(callId)} deadline`);
    w.calls.set(callId, {
      callId,
      nodeId: record.nodeId,
      tool,
      resultKey: typeof toolCallId === "string" ? toolCallId : callId,
      inbox: null,
      mode: null,
      deadline,
      seq,
      pendingSeq: seq,
      state: "open",
      resultSeq: null,
      assigned: [],
      assignedInboxes: [],
      closedInbox: false,
      closedReason: null,
      closedSeq: null,
      releasedSeq: w.calls.get(callId)?.releasedSeq ?? null,
      releasedBy: w.calls.get(callId)?.releasedBy ?? null,
      answeredBy: null,
      flow: step["flow"] === true,
      scope: step["flow"] === true && typeof step["scope"] === "string" ? step["scope"] : null,
      held: step["held"] === true,
    });
    // The interrupt closed the node's calls when it landed; a tool whose group was already
    // running may pend after it, and its call is closed at once.
    if (w.interrupts.has(record.nodeId)) close(w.calls.get(callId), "interrupted", seq);
  } else if (step["kind"] === "released") {
    const call = typeof callId === "string" ? w.calls.get(callId) : undefined;
    if (call?.state === "open") {
      call.state = "released";
      call.releasedSeq = seq;
      call.releasedBy = record.by;
    }
  } else if (step["kind"] === "model_call") {
    w._replies.set(record.nodeId, seq);
  } else if (step["kind"] === "tool_result") {
    const key = typeof toolCallId === "string" ? toolCallId : callId;
    const closed = step["closed"];
    // Only a call of the node's latest reply takes a result, as the reducer checks: provider ids
    // repeat across replies. An agent's closer for a call a record closed is its result; the call
    // stays closed. A released call its agent didn't run takes `skipped` or `crashed` as its
    // answer. Any other closer, or an agent result no pending step registered, answers nothing.
    const reply = w._replies.get(record.nodeId);
    const latest = (call: Call) => call.pendingSeq !== null && (reply === undefined || call.pendingSeq > reply);
    if (typeof closed === "string") {
      for (const call of w.calls.values()) {
        const closes = call.state === "closed" && call.closedReason === closed && call.resultSeq === null;
        const answers = call.state === "released" && (closed === "skipped" || closed === "crashed");
        if (call.nodeId === record.nodeId && call.resultKey === key && latest(call) && (closes || answers)) {
          if (answers) {
            call.state = "answered";
            call.answeredBy = record.by;
          }
          call.resultSeq = seq;
          return;
        }
      }
      return;
    }
    for (const call of w.calls.values()) {
      // A flow call its scope's END answered takes its agent's result, once, and a released call
      // the result of its run.
      const flowAnswer = call.flow && call.state === "answered" && call.resultSeq === null;
      const takes = call.state === "open" || call.state === "released" || flowAnswer;
      if (call.nodeId === record.nodeId && call.resultKey === key && latest(call) && takes) {
        call.state = "answered";
        call.resultSeq = seq;
        call.answeredBy = record.by;
        return;
      }
    }
  } else if (step["kind"] === "steer") {
    const inbox = typeof step["inbox"] === "string" ? w.inboxes.get(step["inbox"]) : undefined;
    if (inbox !== undefined) for (const seq of seqList(step["from"]) ?? []) remove(inbox.unread, seq);
  } else if (step["kind"] === "interrupted") {
    w.interrupts.delete(record.nodeId);
    w._appliedStops.delete(record.nodeId);
    w._honored.add(record.nodeId);
  }
}

/**
 * Apply `interrupted`: each named node holds a pending interrupt, and its open calls close
 * `interrupted`, a waiting node's released calls too; the scope of each of its open flow calls
 * ends. A standing record remembers the Stop on its inbox; one that isn't leaves the stops alone.
 * One while a Stop stands is the run's, applying it for the in-flight reader that opened the inbox:
 * the Stop keeps its seq, which that reader remembers.
 */
function applyInterrupted(w: WaitsState, seq: number, record: InterruptedRecord): void {
  if (record.standing) {
    const stop = w.stops.get(record.inbox);
    // Once it has stopped an agent the Stop stays so: a replayed empty record can't extend its window.
    const stopped = (stop?.stopped ?? false) || record.nodes.length > 0;
    w.stops.set(record.inbox, { seq: stop?.seq ?? seq, stopped, by: record.by });
    if (stop !== undefined) for (const nodeId of record.nodes) if (!w.waiting.has(nodeId)) w._appliedStops.set(nodeId, stop.seq);
  }
  for (const nodeId of record.nodes) {
    w.interrupts.set(nodeId, seq);
    w._interruptInboxes.set(nodeId, record.inbox);
    for (const call of interruptedCalls(w, nodeId)) {
      const open = call.state === "open";
      shut(call, "interrupted", seq);
      if (open && call.scope !== null) endScope(w, call.scope);
    }
  }
}

/**
 * Return the calls of `nodeId` an interrupt naming it closes, in registration order: its open
 * calls, and a waiting node's released calls, which haven't started. A running node's released
 * calls are its agent's to settle.
 */
export function interruptedCalls(w: WaitsState, nodeId: string): readonly Call[] {
  const waiting = w.waiting.has(nodeId);
  return nodeCalls(w, nodeId).filter((call) => call.state === "open" || (waiting && call.state === "released"));
}

/** End flow call scope `scope`: each node of it, a nested scope's included, settles here. */
function endScope(w: WaitsState, scope: string): void {
  const inside = (nodeId: string) => nodeId.startsWith(`${scope}/`);
  const nodes = new Set([...w.calls.values()].map((call) => call.nodeId).filter(inside));
  for (const nodeId of w.waiting.keys()) if (inside(nodeId)) nodes.add(nodeId);
  for (const nodeId of w._pendingTakes.keys()) if (inside(nodeId)) nodes.add(nodeId);
  for (const nodeId of nodes) settle(w, nodeId);
}

/**
 * Apply `node_waiting`: register its take or sleep (a take joins the absorbing takes of each open
 * inbox it reads) and move the node to the waiting set.
 */
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
      assignedInboxes: [],
      closedInbox: false,
      closedReason: null,
      closedSeq: null,
      releasedSeq: null,
      releasedBy: null,
      answeredBy: null,
      flow: false,
      scope: null,
      held: false,
    });
    if (entry.tool === "take") {
      for (const name of takeInboxes(entry.inbox!)) {
        // A closed inbox gets no more sends; a take of several waits on the others.
        if (!isClosed(w, name)) inboxOf(w, name).absorbing.push(entry.callId);
      }
      // The run waits for the inbox's next message: a pickup.
      pickup(w, entry.inbox!);
    }
  }
  w.waiting.set(record.nodeId, { seq, callIds: record.callIds, steerInbox: record.steerInbox });
}

/** Apply `node_woken`: time out its calls (a host call closes `timed_out`), end its takes' absorbing and move it back in flight. */
function applyWoken(w: WaitsState, seq: number, record: NodeWokenRecord): void {
  for (const callId of record.timedOut) {
    const call = w.calls.get(callId);
    if (call === undefined || call.state !== "open") continue;
    if (call.tool === "take" || call.tool === "sleep") call.state = "timed_out";
    else close(call, "timed_out", seq);
    // A take timing out is a pickup (of nothing).
    if (call.tool === "take") pickup(w, call.inbox!);
  }
  for (const call of nodeCalls(w, record.nodeId)) stopAbsorbing(w, call);
  w.waiting.delete(record.nodeId);
  w._lastWoken.set(record.nodeId, seq);
}

/**
 * Apply `node_succeeded`: an inline take's events leave `unread`; then the node settles. The END of
 * a flow call's scope answers that call; its agent writes the result it reads.
 */
function applySuccess(w: WaitsState, record: NodeSucceededRecord): void {
  if (record.nodeId.endsWith(`/${EndNode.ID}`)) {
    for (const call of w.calls.values()) {
      if (call.scope === null || call.state !== "open" || `${call.scope}/${EndNode.ID}` !== record.nodeId) continue;
      call.state = "answered";
      break;
    }
  }
  const take = w._pendingTakes.get(record.nodeId);
  // A waiting take's events were assigned, never unread; only an inline take's are taken here.
  const waited = nodeCalls(w, record.nodeId).some((call) => call.tool === "take");
  if (take !== undefined && take.seq === record.commit && !waited) {
    for (const name of takeInboxes(take.inbox)) {
      const unread = inboxOf(w, name).unread;
      for (const event of take.events) remove(unread, event);
    }
  }
  settle(w, record.nodeId);
  // END's success ends the run.
  if (record.nodeId === EndNode.ID) w.stops.clear();
}

/**
 * Apply `inbox_closed`: absorbing takes with nothing assigned are answered by the close, a take of
 * several inboxes only once each of them is closed; none absorbs after it.
 */
function applyClose(w: WaitsState, name: string): void {
  const inbox = inboxOf(w, name);
  inbox.closed = true;
  for (const callId of inbox.absorbing) {
    const call = w.calls.get(callId);
    if (call === undefined || call.assigned.length > 0) continue;
    if (!takeInboxes(call.inbox!).every((other) => isClosed(w, other))) continue;
    call.state = "answered";
    call.closedInbox = true;
    // The close answers a waiting take: a pickup.
    pickup(w, call.inbox!);
  }
  inbox.absorbing.length = 0;
}

/**
 * Apply `inbox_withdrawn`: its sends leave `unread`, and each take of a waiting node that held one
 * gives up what it was assigned and waits again, absorbing in call order; the sends kept go to the
 * takes that lost theirs, oldest first, as new sends would, and a take that lost nothing keeps what
 * it holds. No take absorbs a closed inbox: there a take keeps the sends not named, and one left
 * with none reads what is still unread, as a new take would, and takes the close only when nothing
 * is. A take of several inboxes does the same across them: it keeps the sends of its closed
 * inboxes, and reads what is unread directly only once each of them is closed.
 */
function applyWithdrawn(w: WaitsState, record: InboxWithdrawnRecord): void {
  const withdrawn = new Set(record.events);
  for (const seq of record.events) remove(inboxOf(w, record.inbox).unread, seq);
  const offered = new Set([record.inbox]);
  for (const call of waitedTakes(w, record.inbox)) {
    const kept = call.assigned.flatMap((seq, index) => (withdrawn.has(seq) ? [] : [[seq, call.assignedInboxes[index]!] as const]));
    if (kept.length === call.assigned.length) continue;
    call.assigned.length = 0;
    call.assignedInboxes.length = 0;
    const names = takeInboxes(call.inbox!);
    const allClosed = names.every((name) => isClosed(w, name));
    if (allClosed && kept.length > 0) {
      for (const [seq, name] of kept) hold(call, seq, name);
      continue;
    }
    // The take waits for its inboxes' next message again (or reads what is left, or the close): a pickup.
    pickup(w, call.inbox!);
    if (allClosed) {
      const available = availableIn(w, names);
      const next = call.mode === "one" ? available.slice(0, 1) : available;
      for (const [seq, name] of next) {
        remove(inboxOf(w, name).unread, seq);
        hold(call, seq, name);
      }
      call.closedInbox = next.length === 0;
      continue;
    }
    for (const [seq, name] of kept) {
      if (isClosed(w, name)) hold(call, seq, name);
      else inboxOf(w, name).unread.push(seq);
    }
    if (call.assigned.length > 0) continue;
    call.state = "open";
    // A `take: all` never left absorbing; a `take: one` left it at its assignment.
    for (const name of names) {
      if (isClosed(w, name)) continue;
      absorbAgain(w, call, name);
      offered.add(name);
    }
  }
  for (const name of offered) inboxOf(w, name).unread.sort((a, b) => a - b);
  reoffer(w, [...offered]);
}

/** Put `call` back among the absorbing takes of inbox `name`, in call order, unless it is there. */
function absorbAgain(w: WaitsState, call: Call, name: string): void {
  const inbox = inboxOf(w, name);
  if (inbox.absorbing.includes(call.callId)) return;
  const later = inbox.absorbing.findIndex((id) => {
    const other = w.calls.get(id);
    return other !== undefined && other.seq > call.seq;
  });
  inbox.absorbing.splice(later < 0 ? inbox.absorbing.length : later, 0, call.callId);
}

/** Close `call` with `reason` when it is open, at the record at `seq`. */
function close(call: Call | undefined, reason: CloseReason, seq: number): void {
  if (call !== undefined && call.state === "open") shut(call, reason, seq);
}

/** Close `call` with `reason`, at the record at `seq`. */
function shut(call: Call, reason: CloseReason, seq: number): void {
  call.state = "closed";
  call.closedReason = reason;
  call.closedSeq = seq;
}

/** Close every open call with `reason`, at the record at `seq`. */
function closeOpenCalls(w: WaitsState, reason: CloseReason, seq: number): void {
  for (const call of w.calls.values()) close(call, reason, seq);
}

/** Forget a settled node's calls and bookkeeping; a take commit no success accepted frees its events. */
function settle(w: WaitsState, nodeId: string): void {
  for (const call of nodeCalls(w, nodeId)) {
    stopAbsorbing(w, call);
    w.calls.delete(call.callId);
  }
  w.waiting.delete(nodeId);
  w._lastWoken.delete(nodeId);
  const take = w._pendingTakes.get(nodeId);
  w._pendingTakes.delete(nodeId);
  if (take !== undefined) reoffer(w, takeInboxes(take.inbox));
  w.interrupts.delete(nodeId);
  w._replies.delete(nodeId);
  w._honored.delete(nodeId);
  w._interruptInboxes.delete(nodeId);
  w._appliedStops.delete(nodeId);
}

/** Return `nodeId`'s registered calls, in registration order. */
export function nodeCalls(w: WaitsState, nodeId: string): readonly Call[] {
  return [...w.calls.values()].filter((call) => call.nodeId === nodeId);
}

/**
 * Return whether `nodeId` waits and an interrupt names it, every call it waits on is settled, or a
 * held call it waits on is no longer open (answered, closed or released): the reply's later calls
 * wait only for its holds.
 */
export function isWakeable(w: WaitsState, nodeId: string): boolean {
  const waiting = w.waiting.get(nodeId);
  if (waiting === undefined) return false;
  if (w.interrupts.has(nodeId)) return true;
  const calls = waiting.callIds.map((callId) => w.calls.get(callId));
  if (calls.some((call) => call?.held === true && call.state !== "open")) return true;
  return calls.every((call) => call !== undefined && call.state !== "open");
}

/** Return the wakeable nodes, oldest `node_waiting` first. */
export function wakeableNodes(w: WaitsState): readonly string[] {
  return [...w.waiting.keys()].filter((nodeId) => isWakeable(w, nodeId));
}

/** Return each inbox's unread send seqs, oldest first; an inbox with none is left out. */
export function unreadEvents(w: WaitsState): ReadonlyMap<string, readonly number[]> {
  return new Map([...w.inboxes].filter(([, inbox]) => inbox.unread.length > 0).map(([name, inbox]) => [name, [...inbox.unread]]));
}

/**
 * Return `inbox`'s unread send seqs that no pending take commit holds, one a resume dropped
 * included: those a new take may read.
 */
export function availableUnread(w: WaitsState, inbox: string): readonly number[] {
  const held = new Set([...w._pendingTakes.values()].flatMap((take) => take.events));
  return (w.inboxes.get(inbox)?.unread ?? []).filter((seq) => !held.has(seq));
}

/**
 * Return what in-flight node `nodeId` takes back: the inbox and the still-unread events, in seq
 * order, of its take commit no success accepted, which a resume has dropped by the time the node
 * commits again. As without the crash, the node takes them, and no other take may read them first
 * (`availableUnread`). `null` when it holds no such commit, or none of its events is still unread.
 */
export function reclaimedTake(w: WaitsState, nodeId: string): { readonly inbox: TakeInbox; readonly events: readonly number[] } | null {
  const take = w._pendingTakes.get(nodeId);
  if (take === undefined) return null;
  const unread = new Set(takeInboxes(take.inbox).flatMap((name) => w.inboxes.get(name)?.unread ?? []));
  const events = take.events.filter((seq) => unread.has(seq));
  return events.length > 0 ? { inbox: take.inbox, events } : null;
}

/** Return the take calls reading inbox `name` whose node waits (hasn't woken), in registration order. */
function waitedTakes(w: WaitsState, name: string): Call[] {
  return [...w.calls.values()].filter(
    (call) => call.tool === "take" && takeInboxes(call.inbox!).includes(name) && w.waiting.has(call.nodeId),
  );
}

/**
 * Return the send seqs of inbox `name` a `withdraw` takes back, in seq order: its unread sends
 * (an orphan inline take commit's among them) and those assigned to a take whose node hasn't
 * woken. Not included: sends assigned to a woken take, taken by an accepted take, or read by a
 * steer.
 */
export function withdrawableEvents(w: WaitsState, name: string): readonly number[] {
  const unread = w.inboxes.get(name)?.unread ?? [];
  const held = waitedTakes(w, name).flatMap((call) => call.assigned.filter((_, index) => call.assignedInboxes[index] === name));
  return [...unread, ...held].sort((a, b) => a - b);
}

/**
 * Return whether the run's records say it waits for the next message on inbox `name`: an open
 * take of it waits, nothing of it is withdrawable, and no waiting node reads it as its steer
 * inbox. Running readers are the caller's to check.
 */
export function awaitsNextMessage(w: WaitsState, name: string): boolean {
  if (!waitedTakes(w, name).some((call) => call.state === "open")) return false;
  if (withdrawableEvents(w, name).length > 0) return false;
  return ![...w.waiting.values()].some((waiting) => waiting.steerInbox === name);
}

/** Return a call's deadline in epoch milliseconds, or `null` when it has none. */
function deadlineMs(call: Call): number | null {
  return call.deadline === null ? null : recordTimeMs(call.deadline, "deadline");
}

/**
 * Return, by waiting node in waiting order, the open calls whose deadline is at or before
 * `nowMs`. A waiting node is due once any open call it waits on is past its deadline; the wake
 * times those calls out, and its other calls may stay open.
 */
export function dueTimeouts(w: WaitsState, nowMs: number): ReadonlyMap<string, readonly string[]> {
  const due = new Map<string, readonly string[]>();
  for (const [nodeId, waiting] of w.waiting) {
    const open = waiting.callIds.map((callId) => w.calls.get(callId)).filter((call): call is Call => call?.state === "open");
    const past = open.filter((call) => {
      const ms = deadlineMs(call);
      return ms !== null && ms <= nowMs;
    });
    if (past.length > 0) due.set(nodeId, past.map((call) => call.callId));
  }
  return due;
}

/**
 * Return the earliest deadline of an open call of a waiting node, or `null` when none has one:
 * from the waits state, or from its `waitInfos` (a `RunResult`'s `waits`).
 */
export function wakeAt(w: WaitsState | readonly WaitInfo[]): string | null {
  let earliest: string | null = null;
  for (const wait of w instanceof WaitsState ? waitInfos(w) : w) {
    // Record times have one fixed-width form, so text order is time order.
    if (!wait.settled && wait.deadline !== null && (earliest === null || wait.deadline < earliest)) earliest = wait.deadline;
  }
  return earliest;
}

/**
 * Return the calls of each waiting node, in waiting order, then the order the node names them;
 * a flow call is the run's to answer, so it is left out.
 */
export function waitInfos(w: WaitsState): readonly WaitInfo[] {
  const infos: WaitInfo[] = [];
  for (const [node, waiting] of w.waiting) {
    for (const callId of waiting.callIds) {
      const call = w.calls.get(callId);
      if (call?.flow === true) continue;
      infos.push({
        node,
        callId,
        tool: call?.tool ?? "",
        inbox: call?.inbox ?? null,
        deadline: call?.deadline ?? null,
        settled: call !== undefined && call.state !== "open",
        held: call?.held ?? false,
      });
    }
  }
  return infos;
}

/** What a take or sleep does when its node commits: take now, end the sleep, or wait. */
export type WaitDecision =
  | { readonly kind: "take"; readonly commit: TakeCommit }
  | { readonly kind: "slept" }
  | { readonly kind: "wait"; readonly entry: WaitEntry };

/**
 * Decide the take or sleep `request` of node `nodeId`'s call `callId` when the node commits, at
 * `nowMs`.
 *
 * A call the node already waited on (a restart after its `node_woken`) is reused: an answered take
 * takes its assigned events, closed when its inbox is closed with nothing left to read; a timed-out
 * take takes nothing; a timed-out sleep is done. A take whose commit a resume dropped takes back
 * that commit's events still unread (`reclaimedTake`), closed when its inbox is closed with nothing
 * else to read. A new take reads the unread events no other pending take commit holds: `take: one`
 * takes the oldest, `take: all` takes them all, closed when that drains a closed inbox; with none,
 * a closed inbox or a zero timeout takes nothing at once, and otherwise the node waits, until
 * `nowMs` plus the timeout when there is one. A new sleep waits until its `until` or `nowMs` plus
 * its seconds, or is done when that time isn't after `nowMs`.
 *
 * A take of a list of inboxes reads their events together, in seq order, and its commit names the
 * listed inboxes that are closed with nothing left; with none to read it takes nothing at once only
 * when each of them is closed (or on a zero timeout). A timed-out one names them too.
 *
 * @throws RangeError when `callId` is registered but still open, closed, or not a take or sleep
 *   call, or a sleep's `until` isn't a record time.
 */
export function decideWait(w: WaitsState, nodeId: string, callId: string, request: WaitRequest, nowMs: number): WaitDecision {
  const held = w.calls.get(callId);
  if (held !== undefined) return heldDecision(w, held);
  if (request.tool === "take") {
    const names = takeInboxes(request.inbox);
    const available = availableIn(w, names);
    // A commit a resume dropped is taken again, as without the crash; nothing else is read.
    const back = reclaimedTake(w, nodeId);
    const read = back !== null ? [] : request.mode === "one" ? available.slice(0, 1) : available;
    const left = available.slice(read.length);
    const done = (name: string) => isClosed(w, name) && !left.some(([, of]) => of === name);
    const events = back?.events ?? read.map(([seq]) => seq);
    const allClosed = names.every((name) => isClosed(w, name));
    if (events.length > 0 || allClosed || (request.timeoutSeconds !== null && request.timeoutSeconds <= 0)) {
      return take(request.inbox, events, done);
    }
    const deadline = request.timeoutSeconds === null ? null : formatRecordTime(nowMs + request.timeoutSeconds * 1000);
    return { kind: "wait", entry: { callId, tool: "take", inbox: request.inbox, mode: request.mode, deadline } };
  }
  const deadline =
    request.until !== null ? request.until : formatRecordTime(nowMs + request.forSeconds! * 1000);
  if (recordTimeMs(deadline, "sleep until") <= nowMs) return { kind: "slept" };
  return { kind: "wait", entry: { callId, tool: "sleep", inbox: null, mode: null, deadline } };
}

/** Return the decision a held take or sleep call has already reached. */
function heldDecision(w: WaitsState, call: Call): WaitDecision {
  if (call.tool !== "take" && call.tool !== "sleep") {
    throw new YianRangeError(`call ${reprText(call.callId)} isn't a take or sleep (tool ${reprText(call.tool)})`);
  }
  if (call.state === "open") throw new YianRangeError(`call ${reprText(call.callId)} is still open; its node can't decide its wait`);
  if (call.state === "closed") throw new YianRangeError(`call ${reprText(call.callId)} was closed (${call.closedReason})`);
  if (call.tool === "sleep") return { kind: "slept" };
  const inbox = call.inbox!;
  const done = (name: string) => isClosed(w, name) && availableUnread(w, name).length === 0;
  // A take of one inbox can't time out on a closed one, so it takes the timeout as not closed.
  if (call.state === "timed_out") return take(inbox, [], (name) => typeof inbox !== "string" && done(name));
  return take(inbox, takenEvents(call), done);
}

/** Return the sends a settled take call takes: what it was assigned, in seq order; none once it timed out. */
export function takenEvents(call: Call): number[] {
  return call.state === "timed_out" ? [] : [...call.assigned].sort((a, b) => a - b);
}

/**
 * Return a take decision: `closed` says whether `done` holds for its inbox, or names each listed
 * inbox it holds for.
 */
function take(inbox: TakeInbox, events: readonly number[], done: (name: string) => boolean): WaitDecision {
  const closed = typeof inbox === "string" ? done(inbox) : inbox.filter(done);
  return { kind: "take", commit: { inbox, events, closed } };
}
