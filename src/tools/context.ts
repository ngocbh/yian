/** The per-call context yian passes to a host tool, the reserved handler ids and the pending marker. */

/** The reserved id of a run's `ask_user` handler. */
export const ASK_USER = "ask_user";

/** The built-in `take` tool: read an inbox. */
export const TAKE = "take";
/** The built-in `sleep` tool: wait until a time. */
export const SLEEP = "sleep";
/** The host-implemented `ask` tool: ask a person. */
export const ASK = "ask";
/** A built-in tool id. */
export type BuiltinToolId = typeof TAKE | typeof SLEEP | typeof ASK;
/** The built-in tool ids, which a TOOL node reaches with `tool_id:`. */
export const BUILTIN_TOOL_IDS: ReadonlySet<BuiltinToolId> = new Set<BuiltinToolId>([TAKE, SLEEP, ASK]);

/** A TOOL node's call id: the same on every attempt and start of the node. */
export function toolCallId(runId: string, nodeId: string): string {
  return `${runId}/${nodeId}`;
}

/** The brand `pending()` carries; a registry symbol, so a marker from another yian copy counts. */
export const PENDING_BRAND: unique symbol = Symbol.for("yian.pending") as never;

/** What a host handler returns to say "the answer comes later, through complete()". */
export interface PendingMarker {
  readonly [PENDING_BRAND]: true;
}

const PENDING: PendingMarker = Object.freeze({ [PENDING_BRAND]: true as const });

/** Return the pending marker. A handler returns it; yian records the call and the node waits. */
export function pending(): PendingMarker {
  return PENDING;
}

/** Whether `value` is a pending marker; never throws. */
export function isPending(value: unknown): value is PendingMarker {
  try {
    return typeof value === "object" && value !== null && (value as Record<symbol, unknown>)[PENDING_BRAND] === true;
  } catch {
    // A Proxy whose get trap throws isn't a marker.
    return false;
  }
}

/**
 * Who is calling a tool, the host's app object, and the call's abort signal.
 *
 * @example
 * const place = (args: { symbol: string }, ctx: ToolContext<Broker>) =>
 *   ctx.context.submit(args.symbol, { idempotencyKey: ctx.callId, signal: ctx.signal });
 */
export class ToolContext<T = unknown> {
  /**
   * An opaque id for this call. For a TOOL node it is `toolCallId(runId, node)`, the run id plus the
   * expanded node path, for example `run-7f3a/loop#3/place`, and stays the same on every attempt
   * and when a resumed run re-runs the same call.
   * For a call from an AGENT's tool loop it is `<run>/<node>/<n>/<index>` (the reply's `model_call`
   * step and the call's place in the reply), stable across resume. Outside a run (an unbound tool
   * set) each call gets a fresh id.
   */
  readonly callId: string;
  /**
   * The attempt number, starting at 1. For a TOOL node it counts the attempts at the same call: 2
   * on its first retry, and the next number when a resumed run re-runs an unfinished attempt.
   * Every other call, including one from an AGENT's tool loop, sees 1.
   */
  readonly attempt: number;
  /** The tool's own name: its id, or a group's tool's name without its group. */
  readonly toolName: string;
  /** The run's id, or `""` outside a run. */
  readonly runId: string;
  /**
   * The expanded id of the calling node, or `""` when the caller names none. A node called
   * directly, outside a run, still passes its id.
   */
  readonly node: string;
  /** The host's app object given to the run, or to its resume. It is never recorded. */
  readonly context: T;
  /**
   * Aborted when the call must end early: the run stops, or a sibling call fails the node. A tool
   * that waits should pass it on or watch it.
   */
  readonly signal: AbortSignal;

  /** Build a frozen context from its fields. */
  constructor(fields: {
    callId: string;
    attempt: number;
    toolName: string;
    runId: string;
    node: string;
    context: T;
    signal: AbortSignal;
  }) {
    this.callId = fields.callId;
    this.attempt = fields.attempt;
    this.toolName = fields.toolName;
    this.runId = fields.runId;
    this.node = fields.node;
    this.context = fields.context;
    this.signal = fields.signal;
    Object.freeze(this);
  }
}
