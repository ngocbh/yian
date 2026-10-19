/**
 * The node base contract: a node is a pure function of its bound input record.
 *
 * The base's async template method `run(inputs, caps)` fills defaults, checks the pre-asserts,
 * calls the kind's body, checks the post-asserts, and resolves to ONE of the closed sum
 * `Output | Route | Grow` (`NodeResult`). A kind supplies its body as `forward(inputs, caps)`,
 * which returns a result or a promise of one. The node never receives the pool: the engine's
 * evaluator binds its inputs (the read boundary) and hands it a record. Narrow effects enter
 * through one capabilities object, `caps`: the node's own abort `signal`, plus what its traits
 * ask for (`bindItem` for MAP, `steps` for a node that records durable steps). Failure is not a
 * variant: a failing node rejects, and the evaluator turns an expected rejection into a
 * canonical failure retained by the accepted failure record.
 *
 * Invariant: a node never writes the pool. It describes its one output value as
 * `new Output(value)`; the engine performs the write under the node's id. This keeps nodes pure.
 *
 * Knows about:   diagnostics, expr, memory contracts, typesys, nodes/binding, nodes/contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { reprText } from "../diagnostics.ts";
import { type FeedbackValue, validateFeedback } from "../expr/index.ts";
import { isList, isRecord, setField, type Value, type ValueRecord } from "../typesys/values.ts";

/** The closed vocabulary of node kinds. Dispatch is an explicit `switch`, never a registry. */
export const NodeKind = {
  AGENT: "agent",
  CODE: "code",
  TOOL: "tool",
  CASE: "case",
  /** Blocks through the host-owned `ask_user` tool. */
  HUMAN_INPUT: "human_input",
  /** Authorable: blocks until an absolute UTC date or datetime. */
  WAIT: "wait",
  /** Authorable: iterates a callable to a fixpoint (`('a -> 'a) -> 'a -> 'a`). */
  LOOP: "loop",
  /** Internal-only: the loader-synthesized input boundary (parameter binding). */
  START: "start",
  /** Internal-only: the loader-synthesized return boundary (record and list modes). */
  END: "end",
  /** Authorable: consults another flow once (`kind: call`). */
  CALL: "call",
  /** Authorable: maps a callable over a list (`kind: map` with `over:`). */
  MAP: "map",
} as const;

export type NodeKind = (typeof NodeKind)[keyof typeof NodeKind];

/**
 * A produced value that the dispatcher records before the reducer binds it. Frozen.
 *
 * `commitAs` is the node-chosen commit redirect: when set, the reducer writes `value` under
 * `commitAs` instead of the node's own id and fires that target's out-edges; `null` (the common
 * case) commits under the node's own id. LOOP uses it when its final driver commits under the
 * compiled origin; the baked `Node.commitAs` field covers child-flow terminal redirects.
 */
export class Output {
  /** The produced value. */
  readonly value: unknown;
  /** The target id to commit under instead of the node's own id, or `null`. */
  readonly commitAs: string | null;

  /** Describe one produced value; `value` and `commitAs` default to `null`. */
  constructor(value: unknown = null, options: { commitAs?: string | null } = {}) {
    this.value = value;
    this.commitAs = options.commitAs ?? null;
    Object.freeze(this);
  }
}

/**
 * A routing-only outcome (a terminal) carrying only the chosen case handle: no value, no
 * post-asserts. Frozen. A router (CASE) returns it; the dispatcher records the handle and the
 * reducer takes its out-edge while skip-flooding siblings.
 */
export class Route {
  /** The chosen case handle. */
  readonly handle: string;

  /** Describe one routing choice. */
  constructor(handle: string) {
    this.handle = handle;
    Object.freeze(this);
  }
}

// Labeled branches are named by their target node id, and no authored node id may hold `#`, so
// no labeled branch can share the else route.
/**
 * The handle a router takes when no labeled branch matches. Unlike any other handle it may have
 * no outgoing edge.
 */
export const ELSE_HANDLE = "#else";

/**
 * One deterministic graph expansion a spawner describes for the runtime to record. Frozen.
 *
 * `seed` is the pure input to the compile-owned growth builder: kind-shaped but opaque to the
 * engine (a CALL record, MAP records list, or LOOP record and index). The same canonical value
 * is persisted, so durable replay rebuilds identical topology without re-running the node that
 * produced it. `prune` holds the ids to retire in the same step (empty for CALL and MAP; a
 * self-respawning LOOP's finished iteration ids). The reducer retires a pruned id with its whole
 * grown subtree once every node in it is settled; until then it stays live.
 */
export class Grow {
  /** The pure input to the compile-owned growth builder. */
  readonly seed: unknown;
  /** The ids to retire in the same step; a private copy, read-only by contract. */
  readonly prune: ReadonlySet<string>;

  /** Describe one expansion; `prune` is copied into a new set and defaults to empty. */
  constructor(seed: unknown, prune: Iterable<string> = []) {
    this.seed = seed;
    this.prune = new Set(prune);
    Object.freeze(this);
  }
}

/** The closed sum a node's `run` resolves to. */
export type NodeResult = Output | Route | Grow;

/** Require one batch of independent textual messages. */
function validateTextFeedback(messages: unknown, name: string): void {
  if (!isList(messages)) throw new TypeError(`${name} must be an array`);
  for (const message of messages) {
    if (typeof message !== "string") throw new TypeError(`${name} messages must be strings`);
  }
}

/**
 * Independent feedback for an owner's inputs, behavior, and local variables. Frozen.
 *
 * Input messages keep ordinary records and positional lists. Every value is a deep-frozen copy
 * made by the constructor, so no caller can change the retained result: copy a value before you
 * modify it.
 */
export class BackwardResult {
  /** Authored input names mapped to ordered batches of independent feedback values. */
  readonly inputFeedback: Readonly<Record<string, readonly FeedbackValue[]>>;
  /** Observations about this owner's behavior. */
  readonly selfFeedback: readonly string[];
  /** Textual messages for declared local variables; these don't update variable values. */
  readonly variableFeedback: Readonly<Record<string, readonly string[]>>;

  /**
   * Validate the channels and retain frozen copies of them.
   *
   * @param options.inputFeedback Input message batches; `null` or omitted is an empty channel. An
   *   array inside a batch is one positional message.
   * @param options.selfFeedback Independent messages about the owner's behavior; defaults to
   *   empty.
   * @param options.variableFeedback Local variable message batches; `null` or omitted is an empty
   *   channel.
   * @throws TypeError for an invalid channel, batch, or message type.
   * @throws RangeError for an invalid nested feedback selection.
   */
  constructor(
    options: {
      inputFeedback?: Readonly<Record<string, readonly FeedbackValue[]>> | null;
      selfFeedback?: readonly string[];
      variableFeedback?: Readonly<Record<string, readonly string[]>> | null;
    } = {},
  ) {
    const { selfFeedback = [] } = options;
    const inputs: unknown = options.inputFeedback ?? {};
    const variables: unknown = options.variableFeedback ?? {};
    if (!isRecord(inputs)) throw new TypeError("input_feedback must be a record");
    for (const name of Object.keys(inputs)) {
      const messages = inputs[name];
      if (!isList(messages)) throw new TypeError(`input_feedback[${reprText(name)}] must be an array`);
      for (const message of messages) validateFeedback(message, message, [name]);
    }
    validateTextFeedback(selfFeedback, "self_feedback");
    if (!isRecord(variables)) throw new TypeError("variable_feedback must be a record");
    for (const name of Object.keys(variables)) {
      validateTextFeedback(variables[name], `variable_feedback[${reprText(name)}]`);
    }
    this.inputFeedback = copyBackwardValue(inputs, "input_feedback", new Map()) as typeof this.inputFeedback;
    this.selfFeedback = copyBackwardValue(selfFeedback, "self_feedback") as readonly string[];
    this.variableFeedback = copyBackwardValue(variables, "variable_feedback") as typeof this.variableFeedback;
    Object.freeze(this);
  }
}

/**
 * Check one retained JSON value and return a deep-frozen copy of it.
 *
 * JSON here is `null`, strings, booleans, `bigint`s, numbers other than NaN (infinity is a
 * value), plain arrays and plain records. `-0` is copied as `0`, since yian has no negative
 * zero, and a `__proto__` key stays data.
 *
 * @param value The JSON value to check and copy.
 * @param name The value's name in error messages; a nested value is named `name[0]` or
 *   `name['k']`.
 * @param memo Containers already copied, mapped to their copies. A container met again returns
 *   its earlier copy, so the result shares it. Omitted, every occurrence is copied separately.
 * @returns The copy, with fresh frozen arrays and records.
 * @throws TypeError for a value that isn't JSON (`<name> must contain only JSON values`).
 * @throws RangeError for NaN (`<name> must not contain NaN`).
 */
export function copyBackwardValue(value: unknown, name: string, memo?: Map<object, unknown>): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean" || typeof value === "bigint") {
    return value;
  }
  if (typeof value === "number") {
    if (Number.isNaN(value)) throw new RangeError(`${name} must not contain NaN`);
    return Object.is(value, -0) ? 0 : value;
  }
  let result: unknown[] | ValueRecord;
  if (isList(value)) {
    const shared = memo?.get(value);
    if (shared !== undefined) return shared;
    result = [];
    // An index loop, not `map`: `map` skips the holes of a sparse array, which must fail here.
    for (let index = 0; index < value.length; index++) {
      result.push(copyBackwardValue(value[index], `${name}[${index}]`, memo));
    }
  } else if (isRecord(value)) {
    const shared = memo?.get(value);
    if (shared !== undefined) return shared;
    result = {};
    for (const key of Object.keys(value)) {
      setField(result, key, copyBackwardValue(value[key], `${name}[${reprText(key)}]`, memo) as Value);
    }
  } else {
    throw new TypeError(`${name} must contain only JSON values`);
  }
  Object.freeze(result);
  // Only finished copies are stored, so a failing container fails again, with the same name,
  // wherever it reappears.
  memo?.set(value, result);
  return result;
}
