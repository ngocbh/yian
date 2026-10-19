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

import { Failure, OriginRef, reprText } from "../diagnostics.ts";
import { type FeedbackValue, hasFeedback, validateFeedback } from "../expr/index.ts";
import { isList, isRecord, setField, type Value, type ValueRecord } from "../typesys/values.ts";
import { ExecutionScope, isExactly } from "./contracts.ts";

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

/**
 * Require one inventory of distinct authored names. `BackwardContext` and the kinds that capture
 * backward inputs use it.
 *
 * @throws TypeError when `names` isn't an array of strings.
 * @throws RangeError for a repeated name.
 */
export function validateDeclaredNames(names: unknown, fieldName: string): void {
  if (!isList(names)) throw new TypeError(`${fieldName} must be an array`);
  for (const name of names) {
    if (typeof name !== "string") throw new TypeError(`${fieldName} must contain only strings`);
  }
  if (new Set(names).size !== names.length) throw new RangeError(`${fieldName} must contain unique names`);
}

/**
 * Validate the optional source-neutral paths from declared inputs into the output, held in
 * `data.input_output_paths`. `BackwardContext` and the structural backward hooks use it.
 *
 * A path part is a string, a non-negative `bigint`, or a non-negative safe-integer `number`.
 *
 * @throws TypeError when the paths aren't a record of arrays.
 * @throws RangeError for an undeclared input or an invalid path part.
 */
export function validateInputOutputPaths(data: Readonly<Record<string, unknown>>, inputNames: readonly string[]): void {
  if (!Object.hasOwn(data, "input_output_paths")) return;
  const paths = data.input_output_paths;
  if (!isRecord(paths)) throw new TypeError("input_output_paths must be a record");
  for (const name of Object.keys(paths)) {
    if (!inputNames.includes(name)) throw new RangeError("input_output_paths must name declared inputs");
    const path = paths[name];
    if (!isList(path)) throw new TypeError("input_output_paths values must be arrays");
    for (const part of path) {
      const valid =
        typeof part === "string" ||
        (typeof part === "bigint" && part >= 0n) ||
        (typeof part === "number" && Number.isSafeInteger(part) && part >= 0);
      if (!valid) throw new RangeError("output path components must be strings or non-negative integers");
    }
  }
}

/**
 * Detached forward material and attribution supplied to one backward hook. Frozen.
 *
 * Input declarations stay separate from recorded values: an early binding failure has
 * `inputs === null` even when the owner declares input names. JSON values are deep-frozen copies
 * made by the constructor; copy one before you modify it. The failure's trace keeps canonical
 * origin keys without subclass state.
 */
export class BackwardContext {
  /** The bound JSON input record, or `null` when no usable record was retained. */
  readonly inputs: Readonly<Record<string, unknown>> | null;
  /** Authored input names this owner may address, including defaults. */
  readonly inputNames: readonly string[];
  /** Declared local variables; an empty inventory rejects all targets. */
  readonly variableNames: readonly string[];
  /** The recorded JSON output, when the forward invocation produced one. */
  readonly output: unknown;
  /** The recorded forward failure, independent of backward's outcome. */
  readonly failure: Failure | null;
  /** The authored path and structured child position, when available. */
  readonly scope: ExecutionScope | null;
  /** The recorded run id. */
  readonly runId: string | null;
  /** The display id of the executing node; it is never parsed to infer scope. */
  readonly nodeId: string | null;
  /** The positive accepted node-start sequence, paired with the run id. */
  readonly nodeStartedSeq: number | null;
  /** Node-owned JSON material retained for backward. */
  readonly nodeData: Readonly<Record<string, unknown>>;
  /** Ordered JSON attribution accompanying feedback messages. */
  readonly downstreamContext: readonly Readonly<Record<string, unknown>>[];

  /**
   * Validate the invocation metadata and retain frozen copies of its values.
   *
   * @param inputs The canonical bound JSON input record; `null` means no usable input record was
   *   retained.
   * @param options.inputNames Authored input names; defaults to empty.
   * @param options.variableNames Declared local variables; defaults to empty.
   * @param options.output The recorded JSON output; defaults to `null`.
   * @param options.failure The recorded forward failure, rebuilt with plain `OriginRef`s; defaults
   *   to `null`.
   * @param options.scope The execution scope; defaults to `null`.
   * @param options.runId The recorded run id; defaults to `null`.
   * @param options.nodeId The executing node's display id; defaults to `null`.
   * @param options.nodeStartedSeq The positive node-start sequence; defaults to `null`.
   * @param options.nodeData Node-owned JSON material; `null` or omitted is `{}`.
   * @param options.downstreamContext Ordered JSON attribution records; defaults to empty. They
   *   are copied with one memo, so a subtree they share is copied once and stays shared.
   * @throws TypeError for malformed metadata or non-JSON values.
   * @throws RangeError for duplicate declarations, an invalid position or path, or NaN.
   */
  constructor(
    inputs: Readonly<Record<string, unknown>> | null,
    options: {
      inputNames?: readonly string[];
      variableNames?: readonly string[];
      output?: unknown;
      failure?: Failure | null;
      scope?: ExecutionScope | null;
      runId?: string | null;
      nodeId?: string | null;
      nodeStartedSeq?: number | null;
      nodeData?: Readonly<Record<string, unknown>> | null;
      downstreamContext?: readonly Readonly<Record<string, unknown>>[];
    } = {},
  ) {
    const {
      inputNames = [],
      variableNames = [],
      output = null,
      scope = null,
      runId = null,
      nodeId = null,
      nodeStartedSeq = null,
      downstreamContext = [],
    } = options;
    let { failure = null } = options;
    if (inputs !== null && !isRecord(inputs)) throw new TypeError("inputs must be a record or null");
    validateDeclaredNames(inputNames, "input_names");
    validateDeclaredNames(variableNames, "variable_names");
    if (failure !== null) {
      if (!isExactly(failure, Failure)) throw new TypeError("failure must be a Failure or null");
      failure = new Failure({ ...failure, trace: failure.trace.map((origin) => new OriginRef(origin.key)) });
      Object.freeze(failure.trace);
      Object.freeze(failure);
    }
    if (scope !== null && !isExactly(scope, ExecutionScope)) {
      throw new TypeError("scope must be an ExecutionScope or null");
    }
    if (runId !== null && typeof runId !== "string") throw new TypeError("run_id must be a string or null");
    if (nodeId !== null && typeof nodeId !== "string") throw new TypeError("node_id must be a string or null");
    if (nodeStartedSeq !== null) {
      if (!Number.isSafeInteger(nodeStartedSeq)) throw new TypeError("node_started_seq must be an integer or null");
      if (nodeStartedSeq <= 0) throw new RangeError("node_started_seq must be positive");
    }
    const data: unknown = options.nodeData ?? {};
    if (!isRecord(data)) throw new TypeError("node_data must be a record");
    validateInputOutputPaths(data, inputNames);
    if (!isList(downstreamContext)) throw new TypeError("downstream_context must be an array");
    for (const item of downstreamContext) {
      if (!isRecord(item)) throw new TypeError("downstream_context must contain records");
    }

    this.inputs = copyBackwardValue(inputs, "inputs") as typeof this.inputs;
    this.inputNames = Object.freeze([...inputNames]);
    this.variableNames = Object.freeze([...variableNames]);
    this.output = copyBackwardValue(output, "output");
    this.failure = failure;
    this.scope = scope;
    this.runId = runId;
    this.nodeId = nodeId;
    this.nodeStartedSeq = nodeStartedSeq;
    this.nodeData = copyBackwardValue(data, "node_data") as typeof this.nodeData;
    // Messages below one route node share their route contexts, so the items are copied with one
    // memo: each shared subtree is copied once.
    const memo = new Map<object, unknown>();
    this.downstreamContext = Object.freeze(
      downstreamContext.map((item, index) => copyBackwardValue(item, `downstream_context[${index}]`, memo)),
    ) as typeof this.downstreamContext;
    Object.freeze(this);
  }
}

/**
 * Validate every feedback channel before routing any input message.
 *
 * @param result The owner's proposed feedback channels.
 * @param context The recorded bound values and the owner's declared input and local variable
 *   inventories.
 * @throws TypeError for a result or context of another class, or an invalid message type.
 * @throws RangeError for an undeclared target, an unavailable input value, or a nested feedback
 *   shape mismatch.
 */
export function validateBackwardResult(result: BackwardResult, context: BackwardContext): void {
  if (!isExactly(result, BackwardResult)) throw new TypeError("result must be a BackwardResult");
  if (!isExactly(context, BackwardContext)) throw new TypeError("context must be a BackwardContext");
  const inputs = context.inputs;
  const names = Object.keys(result.inputFeedback);
  if (names.length > 0 && inputs === null) throw new RangeError("input_feedback requires usable recorded inputs");
  for (const name of names) {
    if (!context.inputNames.includes(name)) {
      throw new RangeError(`input_feedback selects undeclared input ${reprText(name)}`);
    }
    if (inputs === null || !Object.hasOwn(inputs, name)) {
      throw new RangeError(`input_feedback has no recorded input ${reprText(name)}`);
    }
    for (const message of result.inputFeedback[name]!) validateFeedback(message, inputs[name], [name]);
  }
  for (const name of Object.keys(result.variableFeedback)) {
    if (!context.variableNames.includes(name)) {
      throw new RangeError(`variable_feedback selects undeclared variable ${reprText(name)}`);
    }
  }
}

/**
 * Select input messages through the captured output paths without rewriting text. The
 * structural kinds (START, END, LOOP) use it as their backward.
 *
 * @param context The backward context; its `nodeData.input_output_paths` maps declared inputs to
 *   their paths into the output.
 * @param feedback Independent messages addressing the output.
 * @returns Each input's messages selected at its path; an input with no path or no selected text
 *   gets none, and no recorded inputs give an empty result.
 * @throws TypeError when `feedback` isn't an array.
 * @throws RangeError for missing captured paths, a path absent from the accepted output, or
 *   feedback that doesn't fit the output.
 */
export function projectStructuralFeedback(
  context: BackwardContext,
  feedback: readonly FeedbackValue[],
): BackwardResult {
  if (!isList(feedback)) throw new TypeError("feedback must be an array of independent messages");
  const inputs = context.inputs;
  if (inputs === null) return new BackwardResult();
  const data = context.nodeData;
  validateInputOutputPaths(data, context.inputNames);
  if (!Object.hasOwn(data, "input_output_paths")) {
    throw new RangeError("structural backward requires captured input_output_paths");
  }
  const output = context.failure !== null ? null : context.output;
  for (const message of feedback) validateFeedback(message, output);
  const projected: [string, FeedbackValue[]][] = [];
  const paths = data.input_output_paths as Readonly<Record<string, readonly (string | bigint | number)[]>>;
  for (const name of context.inputNames) {
    if (!Object.hasOwn(inputs, name) || !Object.hasOwn(paths, name)) continue;
    const path = paths[name]!;
    if (context.failure === null) {
      let value = output;
      for (const part of path) {
        if (typeof part === "string" && isRecord(value) && Object.hasOwn(value, part)) value = value[part];
        else if (typeof part !== "string" && isList(value) && Number(part) < value.length) value = value[Number(part)];
        else throw new RangeError("captured output path does not exist in the accepted output");
      }
    }
    const messages: FeedbackValue[] = [];
    for (const message of feedback) {
      let selected = message;
      for (const part of path) {
        if (selected === null || typeof selected === "string") break;
        const key = typeof part === "string" ? part : Number(part);
        // An own-key read: a partial message that omits the field selects nothing here.
        if (isRecord(selected)) selected = Object.hasOwn(selected, key) ? selected[key]! : null;
        else selected = selected[key as number]!;
      }
      if (hasFeedback(selected)) messages.push(selected);
    }
    if (messages.length > 0) projected.push([name, messages]);
  }
  // `fromEntries` defines each key as data, so an input named `__proto__` stays an entry.
  const result = new BackwardResult({ inputFeedback: Object.fromEntries(projected) });
  validateBackwardResult(result, context);
  return result;
}
