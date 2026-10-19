/** The per-call context yian passes to a host tool, and the reserved `ask_user` handler id. */

/** The reserved id of a run's `ask_user` handler. */
export const ASK_USER = "ask_user";

/**
 * Who is calling a tool, the host's app object, and the call's abort signal.
 *
 * @example
 * const place = (args: { symbol: string }, ctx: ToolContext<Broker>) =>
 *   ctx.context.submit(args.symbol, { idempotencyKey: ctx.callId, signal: ctx.signal });
 */
export class ToolContext<T = unknown> {
  /**
   * An opaque id for this call. For a TOOL node it is the run id plus the expanded node path, for
   * example `run-7f3a/loop#3/place`, and stays the same when a resumed run re-runs the same call.
   * For a call from an AGENT's tool loop it is unique per call but not stable across resume until
   * AGENT steps are recorded. Outside a run (an unbound tool set) each call gets a fresh id.
   */
  readonly callId: string;
  /**
   * The attempt number, starting at 1. For a TOOL node it counts the attempts at the same call: 2
   * on its first retry, and the next number when a resumed run re-runs an interrupted attempt.
   * Every other call, including one from an AGENT's tool loop, sees 1.
   */
  readonly attempt: number;
  /** The id the tool is registered under. */
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
