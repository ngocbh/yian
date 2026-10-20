/**
 * Node evaluation at the engine's pure read/execution boundary.
 *
 * A node is a pure function of its bound input record; the engine owns both boundaries: the read
 * (bind the inputs from the pool) and the write (the dispatcher stores `Output`). `evalNode` is
 * the append-only dispatcher's evaluation seam: it binds the record, pre-resolves reserved keys,
 * builds the trait-gated capabilities, runs the node through its assert-owning `run` template,
 * and resolves to an `EvaluatedOutcome` holding exactly `Output | Route | Grow |
 * EvaluationFailure` plus optional evidence and input capture. It publishes no observers; durable
 * observer order belongs to the dispatcher.
 *
 * An expected rejection is classified where its operation occurs: binding evaluation, supplied
 * input contracts, template assertions, node execution, or node-result validation. A stop (the
 * node's aborted signal), host capability faults, unexpected binder faults, and malformed compiled
 * or setup state reject `evalNode` instead of becoming an `EvaluationFailure`.
 *
 * Knows about:   compile/model (types), diagnostics, events, memory contracts, nodes, runtime
 *                contracts and log, skills, tools, typesys.
 * Never imports: runtime state or engine, the run boundary, compose, or the CLI.
 */

import { Failure, type FailureCode, type OriginRef } from "../diagnostics.ts";
import { EvaluationEvidence } from "../memory/contracts.ts";
import { CapabilityFault, Grow, markerMessage, type Node, Output, Route, type Steering } from "../nodes/base.ts";
import { BindingCapture } from "../nodes/binding.ts";
import { isExactly } from "../nodes/contracts.ts";
import { BackwardCapture } from "./contracts.ts";
import { isList, jsTypeName } from "../typesys/values.ts";

/**
 * One raw node-evaluation failure for durable dispatcher preparation. Frozen.
 *
 * `failure` is the sole semantic and presentation authority.
 */
export class EvaluationFailure {
  readonly failure: Failure;

  constructor(failure: Failure) {
    this.failure = failure;
    Object.freeze(this);
  }
}

/**
 * @internal Build one failure of `node` from a thrown value: its message, the `Error`'s `name`
 * (else the value's type name) as the cause type, and the `Error`'s stack as the traceback. The
 * engine uses it for its contract failures.
 *
 * @param origin The failing operation's authored origin; `null` falls back to the node's origin.
 */
export function evaluationFailure(
  node: Node,
  code: FailureCode,
  error: unknown,
  origin: OriginRef | null = null,
): EvaluationFailure {
  return new EvaluationFailure(
    new Failure({
      code,
      message: markerMessage(error),
      nodeId: node.id,
      trace: node.failureTrace(origin),
      causeType: error instanceof Error ? error.name : jsTypeName(error),
      traceback: error instanceof Error ? (error.stack ?? null) : null,
    }),
  );
}

/** Return `error` as a capability fault: a `CapabilityFault` as it is, anything else wrapped. */
function capabilityFault(error: unknown): CapabilityFault {
  return error instanceof CapabilityFault ? error : new CapabilityFault(error);
}

/**
 * Carry a host observer fault out of a node body as a capability fault, so a failing
 * `onStream`/`onEvent` stays a host fault, never a node failure.
 */
function observerSink<T>(sink: (item: T) => unknown): (item: T) => void {
  return (item) => {
    try {
      sink(item);
    } catch (error) {
      throw capabilityFault(error);
    }
  };
}

/**
 * Return a host steering inbox after checking its shape, or `null`.
 *
 * @throws TypeError when `steering` isn't `null` and lacks the `interruptSignal` and `drain`
 *   methods.
 */
export function requireSteering(steering: unknown): Steering | null {
  const inbox = steering as Partial<Steering> | null | undefined;
  if (steering !== null && !(typeof inbox?.interruptSignal === "function" && typeof inbox.drain === "function")) {
    throw new TypeError("steering must provide interruptSignal(nodeId) and drain(nodeId), or be null");
  }
  return steering as Steering | null;
}

/**
 * @internal The steering a node receives: it carries host inbox faults and malformed answers out
 * of the node body as capability faults, so a failing inbox stays a host fault, never a node
 * failure. The evaluator wraps the run's inbox in it.
 */
export class SteeringCapability implements Steering {
  readonly #inbox: Steering;

  constructor(inbox: Steering) {
    this.#inbox = inbox;
  }

  /**
   * Return the host's interrupt signal for `nodeId`.
   *
   * @throws CapabilityFault carrying the host's error, or a `TypeError` when the answer isn't an
   *   `AbortSignal`.
   */
  interruptSignal(nodeId: string): AbortSignal {
    let signal: unknown;
    try {
      signal = this.#inbox.interruptSignal(nodeId);
    } catch (error) {
      throw capabilityFault(error);
    }
    if (!(signal instanceof AbortSignal)) {
      throw new CapabilityFault(new TypeError("steering.interruptSignal must return an AbortSignal"));
    }
    return signal;
  }

  /**
   * Return the node's queued steer messages as a fresh array of strings, or a promise of one when
   * the host answers with a promise.
   *
   * @throws CapabilityFault (or rejects with one) carrying the host's error, or a `TypeError` when
   *   the answer isn't an array of strings.
   */
  drain(nodeId: string): readonly string[] | Promise<readonly string[]> {
    let messages: unknown;
    try {
      messages = this.#inbox.drain(nodeId);
    } catch (error) {
      throw capabilityFault(error);
    }
    if (!(messages instanceof Promise)) return checkedMessages(messages);
    const checked = messages.then(checkedMessages, (error: unknown) => {
      throw capabilityFault(error);
    });
    // A node may drop the promise; its fault must not become an unhandled rejection.
    checked.catch(() => {});
    return checked;
  }
}

/** Return a fresh copy of drained steer messages after checking they are strings. */
function checkedMessages(messages: unknown): string[] {
  if (!isList(messages) || messages.some((message) => typeof message !== "string")) {
    throw new CapabilityFault(new TypeError("steering.drain must return an array of strings"));
  }
  return [...(messages as string[])];
}

/** The closed raw worker result the append-only dispatcher consumes. */
export type WorkerOutcome = Output | Route | Grow | EvaluationFailure;

/**
 * One raw worker outcome with frozen evidence and optional backward capture. Frozen.
 *
 * Binding values and routes travel with the common inventories and static material, as separate
 * lower binding and runtime carriers. Neither is accepted history until the dispatcher records
 * this outcome.
 */
export class EvaluatedOutcome {
  readonly outcome: WorkerOutcome;
  readonly evidence: EvaluationEvidence | null;
  readonly bindings: BindingCapture | null;
  readonly backwardCapture: BackwardCapture | null;

  /**
   * Require one closed raw outcome and exact optional evidence and captures; the backward capture
   * is checked again by rebuilding it.
   *
   * @throws TypeError for an outcome outside `WorkerOutcome`, or evidence or a capture that isn't
   *   exactly its class.
   * @throws RangeError when only one of `bindings` and `backwardCapture` is present, or as
   *   `BackwardCapture` refuses the capture.
   */
  constructor(
    outcome: WorkerOutcome,
    evidence: EvaluationEvidence | null,
    bindings: BindingCapture | null = null,
    backwardCapture: BackwardCapture | null = null,
  ) {
    if (
      !(
        outcome instanceof Output ||
        outcome instanceof Route ||
        outcome instanceof Grow ||
        outcome instanceof EvaluationFailure
      )
    ) {
      throw new TypeError("outcome must be a WorkerOutcome");
    }
    if (evidence !== null && !isExactly(evidence, EvaluationEvidence)) {
      throw new TypeError("evidence must be an EvaluationEvidence or null");
    }
    if (bindings !== null && !isExactly(bindings, BindingCapture)) {
      throw new TypeError("bindings must be a BindingCapture or null");
    }
    if (backwardCapture !== null) {
      if (!isExactly(backwardCapture, BackwardCapture)) {
        throw new TypeError("backward_capture must be a BackwardCapture or null");
      }
      const { inputNames, variableNames, scope, nodeData } = backwardCapture;
      new BackwardCapture({ inputNames, variableNames, scope, nodeData });
    }
    if ((bindings === null) !== (backwardCapture === null)) {
      throw new RangeError("binding capture and backward invocation metadata must be retained together");
    }
    this.outcome = outcome;
    this.evidence = evidence;
    this.bindings = bindings;
    this.backwardCapture = backwardCapture;
    Object.freeze(this);
  }
}
