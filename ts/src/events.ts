/**
 * Transient engine observer vocabulary shared by the runtime and callers.
 *
 * Node observers describe evaluation after their corresponding durable transition
 * has been applied: `NodeStarted` followed by `NodeSucceeded`, `NodeRouted`,
 * `NodeExpanded`, or `NodeFailed`. `StreamChunk` is intentionally transient,
 * as are the in-loop AGENT observers (`ModelCallStarted`/`ModelCallFinished` with
 * `TokenUsage`, `ToolCallStarted`/`ToolCallFinished`, `ReplayDegraded`): they reach `onEvent`
 * between a node's start and outcome but are never yielded, recorded, or replayed,
 * and `NodeSkipped` is derived from a route's durable skip closure. `ModelCallRetried`
 * is transient too, published outside a forward node. A node only
 * describes its result; the reducer owns graph state and variable-pool writes.
 *
 * Run observers bracket a fresh or recovered execution segment with `RunStarted`
 * or `RunResumed`. `RunSucceeded` is derived at successful quiescence;
 * `RunFailed` is derived from a durable node failure or from a reduced transition
 * that makes the terminal output unreachable. Plain classes keep this low-level
 * observer module separate from the serialized record vocabulary in `runtime/log`;
 * canonical semantic failures come from `diagnostics`.
 *
 * Each event has a `type` discriminant equal to its class name and a `toJSON()`
 * giving its snake_case data form, described by `conformance/schema/events.schema.json`:
 * keys in field order, every key present (`null` for an absent optional). Values the
 * flow produced are kept as they are, so write the form with yian's JSON codec, not
 * `JSON.stringify`, which turns infinity into null and throws on a bigint.
 */

import { compareCodeUnits, type Failure } from "./diagnostics.ts";

// --------------------------------------------------------------------------- //
// Node-level signals and reducer observer notifications
// --------------------------------------------------------------------------- //

/** A node began executing. */
export class NodeStarted {
  readonly type = "NodeStarted" as const;
  readonly nodeId: string;

  constructor(options: { nodeId: string }) {
    this.nodeId = options.nodeId;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId };
  }
}

/** A streamed fragment of a node's output (e.g. an LLM token). */
export class StreamChunk {
  readonly type = "StreamChunk" as const;
  readonly nodeId: string;
  readonly key: string;
  readonly chunk: string;
  readonly final: boolean;
  /** The model exchange an AGENT text chunk belongs to; `null` for other streams. */
  readonly exchangeId: string | null;
  /** The AGENT model attempt the text belongs to, 1-based; `null` for other streams. */
  readonly attempt: number | null;

  constructor(options: {
    nodeId: string;
    key: string;
    chunk: string;
    final?: boolean;
    exchangeId?: string | null;
    attempt?: number | null;
  }) {
    this.nodeId = options.nodeId;
    this.key = options.key;
    this.chunk = options.chunk;
    this.final = options.final ?? false;
    this.exchangeId = options.exchangeId ?? null;
    this.attempt = options.attempt ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      key: this.key,
      chunk: this.chunk,
      final: this.final,
      exchange_id: this.exchangeId,
      attempt: this.attempt,
    };
  }
}

/**
 * A node finished and produced its single output value (the engine does the pool write).
 *
 * `commitAs` exposes the effective redirect retained by the success record: when set,
 * the reducer wrote the value under that target and fired its out-edges instead of those
 * of `nodeId`. The dispatcher resolves an explicit `Output.commitAs` before the node's
 * baked target; `null` means the value was committed under `nodeId`.
 */
export class NodeSucceeded {
  readonly type = "NodeSucceeded" as const;
  readonly nodeId: string;
  /** The node's single produced value. */
  readonly output: unknown;
  readonly commitAs: string | null;

  constructor(options: { nodeId: string; output?: unknown; commitAs?: string | null }) {
    this.nodeId = options.nodeId;
    this.output = options.output ?? null;
    this.commitAs = options.commitAs ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId, output: this.output, commit_as: this.commitAs };
  }
}

/**
 * A router (CASE) selected an out-edge handle; the engine takes it and skip-floods the
 * siblings. Stores no value.
 *
 * `handle` is the taken branch's handle: a `case`'s `then:` target id, or `"#else"`
 * (`ELSE_HANDLE`) for its `else:` route, also taken when no `when:` matches.
 */
export class NodeRouted {
  readonly type = "NodeRouted" as const;
  readonly nodeId: string;
  readonly handle: string;

  constructor(options: { nodeId: string; handle: string }) {
    this.nodeId = options.nodeId;
    this.handle = options.handle;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId, handle: this.handle };
  }
}

/** A post-durable observer that a route skip-flood made a node unreachable. */
export class NodeSkipped {
  readonly type = "NodeSkipped" as const;
  readonly nodeId: string;

  constructor(options: { nodeId: string }) {
    this.nodeId = options.nodeId;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId };
  }
}

/** A node failed with one canonical semantic failure. */
export class NodeFailed {
  readonly type = "NodeFailed" as const;
  readonly failure: Failure;

  constructor(options: { failure: Failure }) {
    this.failure = options.failure;
  }

  /** Return the data form, the failure in its JSON form. */
  toJSON() {
    return { failure: this.failure.toJSON() };
  }
}

/**
 * A spawner returned a seed-first `Grow`; the dispatcher builds and splices its fragment.
 *
 * The node ran; its eventual value is committed by the fragment terminal via `commitAs`.
 */
export class NodeExpanded {
  readonly type = "NodeExpanded" as const;
  readonly nodeId: string;
  // Structural, because `Grow` lives in `nodes/base.ts`, a layer above this one.
  readonly grow: { readonly seed: unknown; readonly prune: ReadonlySet<string> };

  constructor(options: { nodeId: string; grow: NodeExpanded["grow"] }) {
    this.nodeId = options.nodeId;
    this.grow = options.grow;
  }

  /** Return the data form, the growth's `prune` set as a sorted list. */
  toJSON() {
    return {
      node_id: this.nodeId,
      grow: { seed: this.grow.seed, prune: [...this.grow.prune].sort(compareCodeUnits) },
    };
  }
}

/**
 * A resume restarted `nodeId`, whose attempts were interrupted `interrupted` times so
 * far. Hosts that resume automatically own any limit on restarts.
 */
export class NodeAttemptResumed {
  readonly type = "NodeAttemptResumed" as const;
  readonly nodeId: string;
  readonly interrupted: number;

  constructor(options: { nodeId: string; interrupted: number }) {
    this.nodeId = options.nodeId;
    this.interrupted = options.interrupted;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId, interrupted: this.interrupted };
  }
}

// --------------------------------------------------------------------------- //
// In-loop AGENT observers (transient: callback-only, never yielded or recorded)
// --------------------------------------------------------------------------- //

/**
 * Provider-reported token counts for one model call.
 *
 * Counts come from the reply's usage metadata; a provider that reports no usage
 * yields no `TokenUsage`. Optional details are `null` when unreported.
 */
export class TokenUsage {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly cacheReadTokens: number | null;
  readonly cacheCreationTokens: number | null;
  readonly reasoningTokens: number | null;

  constructor(options: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    cacheReadTokens?: number | null;
    cacheCreationTokens?: number | null;
    reasoningTokens?: number | null;
  }) {
    this.inputTokens = options.inputTokens;
    this.outputTokens = options.outputTokens;
    this.totalTokens = options.totalTokens;
    this.cacheReadTokens = options.cacheReadTokens ?? null;
    this.cacheCreationTokens = options.cacheCreationTokens ?? null;
    this.reasoningTokens = options.reasoningTokens ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return {
      input_tokens: this.inputTokens,
      output_tokens: this.outputTokens,
      total_tokens: this.totalTokens,
      cache_read_tokens: this.cacheReadTokens,
      cache_creation_tokens: this.cacheCreationTokens,
      reasoning_tokens: this.reasoningTokens,
    };
  }
}

/** Why an AGENT node calls its model. */
export type ModelCallPurpose = "chat" | "structured_native" | "structured_fallback";

/**
 * An AGENT node is about to call its model.
 *
 * `purpose` is `"chat"`, `"structured_native"` or `"structured_fallback"`. `provider` and `model`
 * echo the effective config.
 */
export class ModelCallStarted {
  readonly type = "ModelCallStarted" as const;
  readonly nodeId: string;
  readonly exchangeId: string;
  readonly purpose: ModelCallPurpose;
  readonly provider: string | null;
  readonly model: string | null;
  /** The node's model attempt this call belongs to, 1-based. */
  readonly attempt: number;

  constructor(options: {
    nodeId: string;
    exchangeId: string;
    purpose: ModelCallPurpose;
    provider?: string | null;
    model?: string | null;
    attempt: number;
  }) {
    this.nodeId = options.nodeId;
    this.exchangeId = options.exchangeId;
    this.purpose = options.purpose;
    this.provider = options.provider ?? null;
    this.model = options.model ?? null;
    this.attempt = options.attempt;
  }

  /** Return the data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      exchange_id: this.exchangeId,
      purpose: this.purpose,
      provider: this.provider,
      model: this.model,
      attempt: this.attempt,
    };
  }
}

/**
 * One model call returned or raised.
 *
 * `usage` is the provider's own count, or `null` when it reported none. A
 * call that raised has no `usage` unless the provider replied first (a native
 * structured reply that failed to parse). A streamed reply cut by a host
 * interrupt has no `error` and the usage of the chunks already received.
 * `error` is the error's type name only, never provider text.
 */
export class ModelCallFinished {
  readonly type = "ModelCallFinished" as const;
  readonly nodeId: string;
  readonly exchangeId: string;
  readonly purpose: ModelCallPurpose;
  readonly provider: string | null;
  readonly model: string | null;
  readonly usage: TokenUsage | null;
  readonly error: string | null;
  /** The node's model attempt this call belongs to, 1-based. */
  readonly attempt: number;

  constructor(options: {
    nodeId: string;
    exchangeId: string;
    purpose: ModelCallPurpose;
    provider?: string | null;
    model?: string | null;
    usage?: TokenUsage | null;
    error?: string | null;
    attempt: number;
  }) {
    this.nodeId = options.nodeId;
    this.exchangeId = options.exchangeId;
    this.purpose = options.purpose;
    this.provider = options.provider ?? null;
    this.model = options.model ?? null;
    this.usage = options.usage ?? null;
    this.error = options.error ?? null;
    this.attempt = options.attempt;
  }

  /** Return the data form, the usage in its data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      exchange_id: this.exchangeId,
      purpose: this.purpose,
      provider: this.provider,
      model: this.model,
      usage: this.usage === null ? null : this.usage.toJSON(),
      error: this.error,
      attempt: this.attempt,
    };
  }
}

/** An AGENT node is about to run one granted tool call from a model reply. */
export class ToolCallStarted {
  readonly type = "ToolCallStarted" as const;
  readonly nodeId: string;
  readonly exchangeId: string;
  readonly callIndex: number;
  readonly toolCallId: string;
  readonly name: string;
  readonly arguments: unknown;

  constructor(options: {
    nodeId: string;
    exchangeId: string;
    callIndex: number;
    toolCallId: string;
    name: string;
    arguments?: unknown;
  }) {
    this.nodeId = options.nodeId;
    this.exchangeId = options.exchangeId;
    this.callIndex = options.callIndex;
    this.toolCallId = options.toolCallId;
    this.name = options.name;
    this.arguments = options.arguments ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      exchange_id: this.exchangeId,
      call_index: this.callIndex,
      tool_call_id: this.toolCallId,
      name: this.name,
      arguments: this.arguments,
    };
  }
}

/**
 * One tool call returned (`result` text) or raised (`error` type name).
 *
 * A call that a host interrupt skipped is reported as started and then finished
 * with `error: "interrupted"`; the tool did not run.
 */
export class ToolCallFinished {
  readonly type = "ToolCallFinished" as const;
  readonly nodeId: string;
  readonly exchangeId: string;
  readonly callIndex: number;
  readonly toolCallId: string;
  readonly name: string;
  readonly result: string | null;
  readonly error: string | null;

  constructor(options: {
    nodeId: string;
    exchangeId: string;
    callIndex: number;
    toolCallId: string;
    name: string;
    result?: string | null;
    error?: string | null;
  }) {
    this.nodeId = options.nodeId;
    this.exchangeId = options.exchangeId;
    this.callIndex = options.callIndex;
    this.toolCallId = options.toolCallId;
    this.name = options.name;
    this.result = options.result ?? null;
    this.error = options.error ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      exchange_id: this.exchangeId,
      call_index: this.callIndex,
      tool_call_id: this.toolCallId,
      name: this.name,
      result: this.result,
      error: this.error,
    };
  }
}

/**
 * An AGENT's model left out replay state it couldn't send back for `reason`
 * (`other_account`, `other_provider`, ...), published once per reason per invocation.
 * The replayed turns' text and tool calls were still sent.
 *
 * One published during AGENT backward (under the AGENT's node id) or with `nodeId`
 * `"__memory__"` (a memory's learner) comes from outside a forward node; in practice it
 * can't fire there, since those requests replay no assistant message.
 */
export class ReplayDegraded {
  readonly type = "ReplayDegraded" as const;
  readonly nodeId: string;
  readonly reason: string;

  constructor(options: { nodeId: string; reason: string }) {
    this.nodeId = options.nodeId;
    this.reason = options.reason;
  }

  /** Return the data form. */
  toJSON() {
    return { node_id: this.nodeId, reason: this.reason };
  }
}

/**
 * A model call outside a forward node failed with a retryable error and is sent again
 * after `retryIn` seconds; published before the wait. TS-only: Python yian publishes nothing.
 *
 * `consumer` names what the model was built for, as `ModelAccess.createModel`'s `consumer`
 * option does (`llm_clients/factory.ts`). `attempt` is the
 * failed attempt, from 1; `error` is the provider error's kind and message, scrubbed of
 * credentials. Transient and published outside a node, so it is not in `IN_LOOP_EVENTS`.
 */
export class ModelCallRetried {
  readonly type = "ModelCallRetried" as const;
  readonly nodeId: string;
  readonly consumer: string;
  readonly attempt: number;
  readonly error: { readonly kind: string; readonly message: string };
  readonly retryIn: number;

  constructor(options: {
    nodeId: string;
    consumer: string;
    attempt: number;
    error: ModelCallRetried["error"];
    retryIn: number;
  }) {
    this.nodeId = options.nodeId;
    this.consumer = options.consumer;
    this.attempt = options.attempt;
    this.error = options.error;
    this.retryIn = options.retryIn;
  }

  /** Return the data form. */
  toJSON() {
    return {
      node_id: this.nodeId,
      consumer: this.consumer,
      attempt: this.attempt,
      error: { kind: this.error.kind, message: this.error.message },
      retry_in: this.retryIn,
    };
  }
}

/**
 * The in-loop AGENT event classes, so hosts sharing one `onEvent` callback can
 * separate them from post-durable observers. `ModelCallRetried` is transient too, but it is
 * published outside a node, so it isn't listed here.
 */
export const IN_LOOP_EVENTS = [
  ModelCallStarted,
  ModelCallFinished,
  ToolCallStarted,
  ToolCallFinished,
  ReplayDegraded,
] as const;

// --------------------------------------------------------------------------- //
// Run-level events
// --------------------------------------------------------------------------- //

/** The lead event of a fresh run. */
export class RunStarted {
  readonly type = "RunStarted" as const;

  /** Return the data form, which is empty. */
  toJSON() {
    return {};
  }
}

/** A recovered run segment began after its unfinished nodes were requeued. */
export class RunResumed {
  readonly type = "RunResumed" as const;

  /** Return the data form, which is empty. */
  toJSON() {
    return {};
  }
}

/** The run reached its terminal; `output` is the flow's single committed value. */
export class RunSucceeded {
  readonly type = "RunSucceeded" as const;
  /** The flow's single (possibly object) terminal value. */
  readonly output: unknown;

  constructor(options: { output?: unknown } = {}) {
    this.output = options.output ?? null;
  }

  /** Return the data form. */
  toJSON() {
    return { output: this.output };
  }
}

/** The run ended on a node failure or an unreachable terminal output. */
export class RunFailed {
  readonly type = "RunFailed" as const;
  readonly failure: Failure;

  constructor(options: { failure: Failure }) {
    this.failure = options.failure;
  }

  /** Return the data form, the failure in its JSON form. */
  toJSON() {
    return { failure: this.failure.toJSON() };
  }
}

/** Every event `onEvent` and `onStream` receive (`onStream` receives only `StreamChunk`). */
export type RunEvent =
  | NodeStarted
  | StreamChunk
  | NodeSucceeded
  | NodeRouted
  | NodeSkipped
  | NodeFailed
  | NodeExpanded
  | NodeAttemptResumed
  | ModelCallStarted
  | ModelCallFinished
  | ToolCallStarted
  | ToolCallFinished
  | ReplayDegraded
  | ModelCallRetried
  | RunStarted
  | RunResumed
  | RunSucceeded
  | RunFailed;
