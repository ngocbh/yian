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
 * Knows about:   diagnostics, events, expr, memory contracts, skills, tools, typesys, nodes/binding,
 *                nodes/contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { Failure, type FailureCode, OriginRef, reprText } from "../diagnostics.ts";
import type {
  ModelCallFinished,
  ModelCallStarted,
  StreamChunk,
  ToolCallFinished,
  ToolCallStarted,
} from "../events.ts";
import {
  ExpressionError,
  type FeedbackValue,
  hasFeedback,
  parseCondition,
  promptRefs,
  validateFeedback,
} from "../expr/index.ts";
import { evaluateWhenRecord } from "../expr/expressions.ts";
import { sameFeedback } from "../expr/lineage.ts";
import type { MemoryScope } from "../memory/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import type { ToolSet } from "../tools/registry.ts";
import type { VariablePool } from "../typesys/pool.ts";
import {
  isList,
  isRecord,
  jsTypeName,
  setField,
  type Type,
  type Value,
  type ValueRecord,
  writeAsJson,
} from "../typesys/values.ts";
import { applyDefaults, BindingError, copyValue, type ReservedBinder, validateParameters } from "./binding.ts";
import {
  type Assertion,
  ExecutionScope,
  type HostToolCall,
  isExactly,
  iterOriginCarrier,
  mapOrigin,
  type NodeSteps,
  type OriginMapper,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  rewriteOriginCarrier,
  validateOptionalOrigin,
  ValueContract,
} from "./contracts.ts";

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
      if (!isList(messages)) throw new TypeError(`input_feedback[${writeAsJson(name)}] must be an array`);
      for (const message of messages) validateFeedback(message, message, [name]);
    }
    validateTextFeedback(selfFeedback, "self_feedback");
    if (!isRecord(variables)) throw new TypeError("variable_feedback must be a record");
    for (const name of Object.keys(variables)) {
      validateTextFeedback(variables[name], `variable_feedback[${writeAsJson(name)}]`);
    }
    // Only this channel copies with a memo: Python deep-copies it, so its shared subtrees stay shared.
    this.inputFeedback = copyBackwardValue(inputs, "input_feedback", new Map()) as typeof this.inputFeedback;
    this.selfFeedback = copyBackwardValue(selfFeedback, "self_feedback") as readonly string[];
    this.variableFeedback = copyBackwardValue(variables, "variable_feedback") as typeof this.variableFeedback;
    Object.freeze(this);
  }
}

/** Whether two results have equal input, self and variable feedback channels. */
export function sameBackwardResult(a: BackwardResult, b: BackwardResult): boolean {
  // Each channel is a record of arrays, itself a feedback value.
  return (
    sameFeedback(a.inputFeedback as FeedbackValue, b.inputFeedback as FeedbackValue) &&
    sameFeedback(a.selfFeedback as FeedbackValue, b.selfFeedback as FeedbackValue) &&
    sameFeedback(a.variableFeedback as FeedbackValue, b.variableFeedback as FeedbackValue)
  );
}

/**
 * Check one retained JSON value and return a deep-frozen copy of it. The runtime log, learning,
 * the backward hooks and the runtime contracts use it.
 *
 * JSON here is `null`, strings, booleans, `bigint`s, numbers other than NaN (infinity is a
 * value), plain arrays and plain records. `-0` is copied as `0`, since yian has no negative
 * zero, and a `__proto__` key stays data.
 *
 * @param value The JSON value to check and copy.
 * @param name The value's name in error messages; a nested value is named `name[0]` or
 *   `name["k"]`.
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
  const shared = memo?.get(value as object);
  if (shared !== undefined) return shared;
  let result: unknown[] | ValueRecord;
  if (isList(value)) {
    result = [];
    // An index loop, not `map`: `map` skips the holes of a sparse array, which must fail here.
    for (let index = 0; index < value.length; index++) {
      result.push(copyBackwardValue(value[index], `${name}[${index}]`, memo));
    }
  } else if (isRecord(value)) {
    result = {};
    for (const key of Object.keys(value)) {
      setField(result, key, copyBackwardValue(value[key], `${name}[${writeAsJson(key)}]`, memo) as Value);
    }
  } else {
    throw new TypeError(`${name} must contain only JSON values`);
  }
  Object.freeze(result);
  // Only finished copies are stored, so a failing container fails again, named by its own path,
  // wherever it reappears.
  memo?.set(value as object, result);
  return result;
}

/**
 * Require one inventory of distinct authored names. `BackwardContext`, the runtime trace and the
 * runtime contracts use it.
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
 * Parse a condition expression, prefixing its parse error with `subject`. The node kinds with
 * predicates use it.
 *
 * A `T(...)` constructor must name a type in `types`, the node's type table; with no table,
 * every constructor is refused.
 *
 * @throws ExpressionError `<subject> is not a valid boolean expression: <error>`, with the parse
 *   error as its `cause`.
 */
export function parseNamedCondition(
  expression: string,
  subject: string,
  types: ReadonlyMap<string, Type> | null = null,
): void {
  try {
    // An omitted table is an empty one, so every constructor is refused, as Python's `types or {}`.
    parseCondition(expression, types ?? new Map());
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw new ExpressionError(`${subject} is not a valid boolean expression: ${error.message}`, { cause: error });
  }
}

/**
 * Check one prompt template's `${...}` spans, prefixing an error with `subject`. AGENT and
 * HUMAN_INPUT use it.
 *
 * @throws ExpressionError `<subject>: <error>`, with the template error as its `cause`.
 */
export function checkNamedPrompt(text: string, subject: string): void {
  try {
    promptRefs(text);
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    throw new ExpressionError(`${subject}: ${error.message}`, { cause: error });
  }
}

/**
 * Validate the optional source-neutral paths from declared inputs into the output, held in
 * `data.input_output_paths`. `BackwardContext`, the runtime contracts and the structural
 * backward hooks use it.
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
 * Return a frozen copy of `failure`, checked by `Failure`'s constructor, whose trace holds plain
 * `OriginRef`s. `BackwardContext` and the runtime's `BackwardFailure` use it.
 *
 * @throws TypeError or RangeError when a field breaks `Failure`'s contract, such as a trace that
 *   isn't an array of `OriginRef`s.
 */
export function retainFailure(failure: Failure): Failure {
  // Rebuilt once to check the trace with Failure's own errors, as Python's `__post_init__`,
  // before its items are copied without subclass state.
  const checked = new Failure(failure);
  const trace = checked.trace.map((origin) => Object.freeze(new OriginRef(origin.key)));
  const retained = new Failure({ ...checked, trace });
  Object.freeze(retained.trace);
  return Object.freeze(retained);
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
      failure = retainFailure(failure);
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
 * @throws TypeError for a result or context of another class.
 * @throws RangeError for an undeclared target, an unavailable input value, or a nested feedback
 *   shape mismatch.
 */
export function validateBackwardResult(result: BackwardResult, context: BackwardContext): void {
  if (!isExactly(result, BackwardResult)) throw new TypeError("result must be a BackwardResult");
  if (!isExactly(context, BackwardContext)) throw new TypeError("context must be a BackwardContext");
  // Python rebuilds the result here. `isExactly` refuses a subclass, and a result built by the
  // constructor was validated and frozen then; an object forged with `Object.create` isn't
  // defended against.
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

/**
 * A node-local `asserts:` clause was false, or couldn't be evaluated: thrown by a node's own
 * `run` template around its kind body (START included, after binding).
 *
 * `assertion` is the exact carrier the template checked, or `null` when the error was thrown
 * outside the template. Only the template's instance, marked as an assertion stage, gets the
 * failure code `failureCode`; the same class thrown by node code stays `node.execute`.
 */
export class NodeAssertFailed extends Error {
  override name = "NodeAssertFailed";
  /** The assertion that failed, or `null` outside the run template. */
  readonly assertion: Assertion | null;
  /** `assertion.failed` for a false condition, `condition.evaluate` for one that threw. */
  readonly failureCode: "assertion.failed" | "condition.evaluate";

  /**
   * Retain the assertion that classifies this template-owned failure; `assertion` defaults to
   * `null` and `failureCode` to `assertion.failed`.
   */
  constructor(
    message: string,
    options: {
      assertion?: Assertion | null;
      failureCode?: "assertion.failed" | "condition.evaluate";
      cause?: unknown;
    } = {},
  ) {
    const { assertion = null, failureCode = "assertion.failed", cause } = options;
    super(message, cause === undefined ? undefined : { cause });
    this.assertion = assertion;
    this.failureCode = failureCode;
  }
}

/**
 * Return a stage marker's message: an `Error`'s own message, else the thrown value as text, else
 * its type name when it has no text.
 */
function markerMessage(original: unknown): string {
  if (original instanceof Error) return original.message;
  // A body may throw a value with no text (a null-prototype object, a throwing `toString`); the
  // marker must still wrap it.
  try {
    return String(original);
  } catch {
    return jsTypeName(original);
  }
}

/** A stage marker retaining one value thrown by a node's body. Used by the evaluator. */
export class NodeBodyError extends Error {
  override name = "NodeBodyError";
  /** The exact value the body threw. */
  readonly original: unknown;

  /** Keep the exact thrown value, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** A stage marker for a failure of an evaluator-owned binding stage. Used by the evaluator. */
export class BindingStageError extends Error {
  override name = "BindingStageError";
  /** The binder's error. */
  readonly original: BindingError;

  /** Keep the binder's error apart from any error a node body throws, also as the `cause`. */
  constructor(original: BindingError) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/**
 * A carrier for an unexpected error from the deferred per-item binder, which must escape the
 * node. Used by the evaluator.
 */
export class DeferredBindingFault extends Error {
  override name = "DeferredBindingFault";
  /** The exact value the injected binder threw. */
  readonly original: unknown;

  /** Keep the exact host or setup fault, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/** A carrier for an unexpected host fault from an injected capability. Used by the evaluator. */
export class CapabilityFault extends Error {
  override name = "CapabilityFault";
  /** The exact value the capability threw. */
  readonly original: unknown;

  /** Keep the exact host fault, so the evaluator can pass it on as it was, also as the `cause`. */
  constructor(original: unknown) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/**
 * A host inbox read by a running steerable node, keyed by its node id.
 *
 * The id is the node's expanded id: an expanded LOOP or MAP node has its own id per occurrence.
 * Steering is push-based: the host aborts the node's interrupt signal to stop its loop and
 * return now. `drain` may return the queued steer messages or a promise of them; the node awaits
 * it between its effects.
 * The interrupt signal is separate from the node's own `caps.signal`, so an interrupt never
 * aborts a running tool.
 */
export interface Steering {
  /** The signal the host aborts to interrupt node `nodeId` (its expanded id). */
  interruptSignal(nodeId: string): AbortSignal;
  /** Remove and return the steer messages queued for `nodeId`, oldest first. */
  drain(nodeId: string): readonly string[] | Promise<readonly string[]>;
}

/** A stage marker for an assertion failure of the base run template. Used by the evaluator. */
export class AssertionStageError extends Error {
  override name = "AssertionStageError";
  /** The template's assertion failure. */
  readonly original: NodeAssertFailed;

  /** Keep the template's assertion failure apart from node-body errors, also as the `cause`. */
  constructor(original: NodeAssertFailed) {
    super(markerMessage(original), { cause: original });
    this.original = original;
  }
}

/**
 * The capabilities a node's `run` and `forward` receive, as one object. The evaluator passes
 * every node its own `signal`, plus only what the node's traits ask for.
 */
export interface NodeCaps {
  /** The node's own abort signal; a node that blocks cooperatively stops when it aborts. */
  readonly signal?: AbortSignal;
  /** Binds one element's input record; supplied to a node that binds per item (MAP). */
  readonly bindItem?: (element: Value, index: number) => ValueRecord;
  /** The node's durable steps; supplied to a node that records them (`needsSteps`). */
  readonly steps?: NodeSteps;
  /** The run's tool set; supplied to a node that declares `needsTools`. */
  readonly tools?: ToolSet;
  /** The run's host-supplied skills by name; supplied to a node that declares `needsSkills`. */
  readonly skills?: ReadonlyMap<string, Skill>;
  /** The transient stream-chunk sink; supplied to a node that declares `needsLiveEvents`. */
  readonly emitStream?: (chunk: StreamChunk) => void;
  /** The transient model and tool call observer sink; supplied with `needsLiveEvents`. */
  readonly emitEvent?: (event: ModelCallStarted | ModelCallFinished | ToolCallStarted | ToolCallFinished) => void;
  /** Receives the node's render capture once; supplied to a node that declares `needsRenderCapture`. */
  readonly captureRender?: (captured: Readonly<Record<string, unknown>>) => void;
  /** The host inbox; supplied to a node that declares `needsSteering`. */
  readonly steering?: Steering;
}

/**
 * The base contract every node kind implements: a pure function of its input record.
 *
 * A node implements its `forward` body; the inherited `run` template resolves to one of the
 * closed sum `Output | Route | Grow`. The node never receives the pool (the engine's evaluator
 * binds its inputs and hands it a record) and never writes the pool: it describes its one output
 * as `new Output(value)` and the engine performs the write under the node id. Failure is a
 * rejection, not a variant.
 *
 * The traits are read-only fields a kind overrides (`override readonly isSpawner = true`). A
 * compile-owned node copy is shallow, so code that changes a node reassigns a field and never
 * mutates an array, record, map or carrier it read from one.
 */
export abstract class Node {
  /**
   * The closed-vocabulary tag used by compose/compile and node-owned diagnostics. Runtime never
   * dispatches on it. Set per kind.
   */
  abstract readonly kind: NodeKind;
  /**
   * Declares "I may grow the graph" (a spawner returns a `Grow`). Overridden by the spawner
   * kinds (CALL, MAP, LOOP); the evaluator gates the grow path on it.
   */
  readonly isSpawner: boolean = false;
  /**
   * Declares "my committed output is the run's input record" (START). That record is the
   * argument list around the input values, so it may go one level over the value nesting limit
   * while each input value stays within it.
   */
  readonly commitsInputRecord: boolean = false;
  /**
   * Declares "I bind my inputs per element through `caps.bindItem` rather than once up front"
   * (MAP). When set, the evaluator starts the record empty and supplies `caps.bindItem`;
   * otherwise it binds `params` once from the pool.
   */
  readonly bindsPerItem: boolean = false;
  /**
   * The reserved input key a per-item binder may project its local scope from; `null` when the
   * node has no recorded source collection for the local scope.
   */
  readonly itemSourceKey: string | null = null;
  /**
   * Nested-call depth policy for the engine's growth core: the depth stamped on a grow's subgraph
   * spawners and its terminal is `parentDepth + growDepthDelta`. `null` for a non-spawner (the
   * core does no depth work), `0` for LOOP (it inherits its parent's call depth), `1` for CALL
   * and MAP (each child is one level deeper).
   */
  readonly growDepthDelta: number | null = null;
  /**
   * Declares "I am the self-respawning fixpoint-iteration driver" (LOOP). The growth reducer
   * gates the origin-keyed single-live-record invariant on it: each iteration is a fresh driver
   * clone whose grow is attributed to `originId` and supersedes the prior live iteration.
   */
  readonly isLoop: boolean = false;
  /**
   * Declares access to notes and an optional structured transcript through one lazy memory
   * context capability. AGENT is the only built-in kind that opts in.
   */
  readonly needsMemoryContext: boolean = false;
  /**
   * Declares "I can use host-supplied skills": the evaluator supplies the run's skills only when
   * the run boundary injected them and this is set. AGENT is the only built-in kind that opts in.
   */
  readonly needsSkills: boolean = false;
  /**
   * Declares "I can emit visible prompt, model and tool evidence": the evaluator supplies a
   * write-only invocation-local trace only when evidence capture is on and this is set. AGENT is
   * the only built-in kind that opts in.
   */
  readonly needsEvaluationTrace: boolean = false;
  /**
   * Declares "I can publish actual render locals for retained backward context": the evaluator
   * supplies a single-write capture callable only during retention.
   */
  readonly needsRenderCapture: boolean = false;
  /**
   * Declares "I look up or invoke tools": the evaluator supplies the run's own tool set, so
   * per-run tools and the per-run `ask_user` handler never leak between runs. TOOL, HUMAN_INPUT
   * and AGENT opt in.
   */
  readonly needsTools: boolean = false;
  /**
   * Declares "I can publish transient in-loop observers": the evaluator supplies the stream and
   * event sinks the host has. AGENT is the only built-in kind that opts in.
   */
  readonly needsLiveEvents: boolean = false;
  /**
   * Declares "I record durable steps and read them back on a later start": the engine supplies
   * `caps.steps` only when this is set. The steps are opaque to the runtime.
   */
  readonly needsSteps: boolean = false;
  /**
   * Declares "I read the host's steering inbox": the evaluator supplies steering only when the
   * run boundary injected it and this is set. Writable: AGENT sets it from its `steerable:` flag.
   */
  needsSteering: boolean = false;

  /** The node's unique id within its (possibly namespaced) flow. */
  id: string;
  /** A human-friendly display title, or `null`. */
  title: string | null;
  /**
   * The canonical declared output contract: one exact value or a closed record with field-local
   * type evidence, shared by compiled and directly built nodes. `null` leaves the write
   * unenforced.
   */
  outputContract: OutputContract | null;
  /**
   * The node-side declared params, with no source: the flow owns the wiring. Stamped by the
   * compiler; the evaluator binds through `params` and the flow's wiring. `null` for a node that
   * declares no inputs.
   */
  params: readonly Parameter[] | null = null;
  /**
   * Ordered node-local contracts checked by the `run` template against the bound record before
   * `forward`. Stamped by the loader; empty for most nodes.
   */
  preAsserts: readonly Assertion[] = [];
  /** Ordered node-local contracts (reading `${output}`) checked after `forward`. */
  postAsserts: readonly Assertion[] = [];
  /**
   * Engine-baked commit redirect: write this node's output under `commitAs` instead of its own
   * id, and fire that target's out-edges. Set by the compile-owned growth builders on a
   * child-flow terminal (a CALL or MAP child END filler, or a MAP END-list filler) to point back
   * at the spawner id; `null` for an ordinary node.
   */
  commitAs: string | null = null;
  /**
   * "Attribute my grows and self-commit to this origin node": `null` on every ordinary node. Set
   * only on a self-respawning loop driver: the compiled loop `L` and each per-iteration clone
   * `L~k` set it to `L`, so the engine attributes every iteration's `Grow` and `Output` to the
   * one durable origin. A driver is the origin when `id === originId`, which gates "do not
   * self-prune".
   */
  originId: string | null = null;
  /**
   * Opaque authored-source identity for this node, for diagnostics; `null` for a source-free or
   * generated node. Unrelated to `originId`, which controls LOOP execution attribution.
   */
  origin: OriginRef | null;
  /**
   * Generated outer-to-inner authored ancestry for failures inside this node. Directly built
   * nodes start empty; compile-owned expansion derives it from enclosing spawners after the
   * authored origins are rewritten, so it stays outside local origin traversal.
   */
  originContext: readonly OriginRef[] = [];
  /**
   * Compile-derived static memory context; authors can't supply it. Memory layout resolution
   * stamps it when memory is on; otherwise `null`.
   */
  memoryScope: MemoryScope | null = null;
  /**
   * Compile-derived authored location, independent of memory declarations. Static layout
   * resolution stamps root-relative paths on private node templates; `null` when built directly.
   */
  executionScope: ExecutionScope | null = null;
  /**
   * Author-supplied static per-node config, baked at build time from the flow's `env:` default
   * merged with the node's own `env:` (the node wins). The engine never reads it; only a node's
   * own `forward` interprets the keys it understands (AGENT: `max_tool_iterations`), and unknown
   * keys are carried but ignored. `{}` when none is authored.
   */
  env: Readonly<Record<string, unknown>> = {};
  /**
   * The type table `isinstance(x, T)` looks `T` up in when this node evaluates its bindings,
   * asserts and predicates, keyed by type name. Filled at load from the names the node's
   * expressions use; empty when none are used.
   */
  types: ReadonlyMap<string, Type> = new Map();
  /**
   * Wiring param to the author-facing phrase for a reference error, for a synthesized param the
   * author never wrote (a CASE `__rN` from a `when:` ref, `__on` from `on:`), such as
   * `cases[1].when` or `on:`. Render-only and absent from run records; filled by the desugar that
   * mints the param. Empty for authored params, whose default label is `input '<param>'`.
   */
  wiringLabels: ReadonlyMap<string, string> = new Map();
  /**
   * Param to the full phrase a runtime binding error starts with in place of `input '<param>'`
   * or `binding '<param>'` (such as `node 'g' on`), where `wiringLabels` holds only the load-time
   * site phrase (`on:`). Not part of node identity or the manifest; filled by Compose for params
   * the author never wrote as inputs. Empty means the default text.
   */
  bindingLabels: ReadonlyMap<string, string> = new Map();

  /**
   * Build a node; each option defaults to `null`.
   *
   * @param nodeId The node's id within its flow.
   * @param options.title A display title.
   * @param options.outputContract The declared output contract: an exact `ValueContract` or
   *   `PartialRecordContract`.
   * @param options.origin The authored-source identity.
   * @throws TypeError for an output contract of another class, or an origin that isn't an
   *   `OriginRef`.
   */
  constructor(
    nodeId: string,
    options: { title?: string | null; outputContract?: OutputContract | null; origin?: OriginRef | null } = {},
  ) {
    const { title = null, outputContract = null, origin = null } = options;
    if (
      outputContract !== null &&
      !isExactly(outputContract, ValueContract) &&
      !isExactly(outputContract, PartialRecordContract)
    ) {
      throw new TypeError(
        "output_contract must be an exact ValueContract, PartialRecordContract, or null, " +
          `got ${jsTypeName(outputContract)}`,
      );
    }
    validateOptionalOrigin(origin, "node origin");
    this.id = nodeId;
    this.title = title;
    this.outputContract = outputContract;
    this.origin = origin;
  }

  /**
   * Return the generated ancestry plus the best local failure origin.
   *
   * `operationOrigin` wins when present; otherwise the node's `origin` is used. When neither
   * exists, only the ancestry is returned. Order and repeated origins are kept.
   *
   * @param operationOrigin A more precise authored identity for the failing operation.
   * @throws TypeError when the ancestry isn't an array of `OriginRef`s or the selected origin
   *   isn't an `OriginRef`.
   * @throws RangeError for an `OriginRef` whose key bypassed the constructor's check.
   */
  failureTrace(operationOrigin: OriginRef | null = null): readonly OriginRef[] {
    if (!Array.isArray(this.originContext)) throw new TypeError("node origin_context must be an array");
    for (const origin of this.originContext) {
      if (!(origin instanceof OriginRef)) throw new TypeError("node origin_context item must be an OriginRef");
      // Rebuilding checks the key, which an object made without the constructor never had checked.
      new OriginRef(origin.key);
    }
    const selected = operationOrigin ?? this.origin;
    validateOptionalOrigin(selected, "failure operation origin");
    if (selected === null) return this.originContext;
    new OriginRef(selected.key);
    return [...this.originContext, selected];
  }

  /**
   * Yield this node's source-neutral origins without following child flows: the node origin,
   * then the output contract's, the params', the pre-asserts' and the post-asserts', in
   * structural order.
   *
   * @param seen Carriers already visited by a graph traversal; a carrier is yielded once per
   *   `seen`. Omitted, a local traversal starts.
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    validateOptionalOrigin(this.origin, "node origin");
    if (this.origin !== null) yield this.origin;
    if (this.outputContract !== null) yield* iterOriginCarrier(this.outputContract, seen);
    if (this.params !== null) {
      for (const parameter of this.params) yield* iterOriginCarrier(parameter, seen);
    }
    for (const assertion of this.preAsserts) yield* iterOriginCarrier(assertion, seen);
    for (const assertion of this.postAsserts) yield* iterOriginCarrier(assertion, seen);
  }

  /**
   * Rewrite this node's local origins, keeping shared carriers shared.
   *
   * This mutating hook is for a compile-owned private node copy, which may be shallow: it
   * assigns `origin`, `outputContract` and fresh `params`, `preAsserts` and `postAsserts` arrays
   * of rebuilt carriers, and never changes a carrier or an array in place.
   *
   * @param mapper A pure transform for each present origin; its errors propagate.
   * @param seen Carriers already rewritten in this graph, mapped to their rebuilt forms. Omitted,
   *   a local rewrite starts.
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`, or `mapper` returns anything
   *   but an `OriginRef`.
   */
  rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    validateOptionalOrigin(this.origin, "node origin");
    this.origin = mapOrigin(this.origin, mapper);
    if (this.outputContract !== null) this.outputContract = rewriteOriginCarrier(this.outputContract, mapper, seen);
    if (this.params !== null) {
      this.params = this.params.map((parameter) => rewriteOriginCarrier(parameter, mapper, seen));
    }
    this.preAsserts = this.preAsserts.map((assertion) => rewriteOriginCarrier(assertion, mapper, seen));
    this.postAsserts = this.postAsserts.map((assertion) => rewriteOriginCarrier(assertion, mapper, seen));
  }

  /**
   * Return the static child flows whose origins belong to this node. The base has none; the
   * higher-order kinds override this without importing the compile layer that owns traversal.
   */
  originChildren(): readonly unknown[] {
    return [];
  }

  /**
   * Return the authored operation origin for one node-body error. The base returns the node's
   * `origin` (`null` for a source-free node); a kind with finer source-neutral mapping overrides
   * this.
   */
  exceptionOrigin(error: unknown): OriginRef | null {
    return this.origin;
  }

  /**
   * Return the failure code for one node-body error. The base returns `node.execute`; a kind
   * with a finer source-neutral classification overrides this.
   */
  exceptionCode(error: unknown): FailureCode {
    return "node.execute";
  }

  /**
   * Return the node whose successors observe this node's absent output.
   *
   * A successful output commits under the baked `commitAs` redirect when present, else under the
   * node's own id. Route skip propagation follows the same boundary, so generated completion
   * nodes don't strand their parent's outgoing edges. A kind whose output target is chosen by
   * other node-owned state overrides this.
   */
  skippedOutputTarget(): string {
    return this.commitAs ?? this.id;
  }

  /**
   * Reject malformed node-owned execution setup before the body runs.
   *
   * The base requires a kind's own `forward`, then parses every condition through
   * `validateConditions`, so a malformed post-assert or predicate is reported before the body
   * has effects. A composite kind extends this for its own invariants, such as requiring its
   * compiled child flow.
   *
   * @throws Error when the node's class defines no `forward`.
   * @throws ExpressionError when a condition doesn't parse.
   */
  validateExecutionSetup(): void {
    if (this.forward === Node.prototype.forward) {
      throw new Error(`node ${reprText(this.id)} (${this.constructor.name}) defines no forward`);
    }
    this.validateConditions();
  }

  /**
   * Parse every condition this node evaluates. The base parses the node's assertions; a kind
   * that evaluates its own predicate (CASE `when`, LOOP `while`/`until`) extends this.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   * @throws ExpressionError when a condition doesn't parse; the message names the node, the
   *   field and the expression.
   */
  validateConditions(options: { path?: string | null } = {}): void {
    this.validateAssertions(options);
  }

  /**
   * Parse every pre-assert and post-assert expression on this node.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   * @throws ExpressionError when an assertion doesn't parse; the message names the node, the
   *   assertion list and the assertion.
   */
  validateAssertions(options: { path?: string | null } = {}): void {
    const { path = null } = options;
    for (const assertion of this.preAsserts) this.parseAssertion("pre-assert", assertion, path);
    for (const assertion of this.postAsserts) this.parseAssertion("post-assert", assertion, path);
  }

  /**
   * Check every prompt template this node renders. The base renders none; AGENT and HUMAN_INPUT
   * override this. An override throws `ExpressionError` when a template's `${...}` span doesn't
   * parse, calls `isinstance` or a function that isn't a builtin, or calls a builtin with
   * arguments it doesn't take; the message names the node and the field.
   *
   * @param options.path The static node path to name in the error instead of the node's id,
   *   used by a pre-run check over a whole flow.
   */
  validatePrompts(options: { path?: string | null } = {}): void {}

  /** Parse one assertion's expression, naming the node and the assertion when it doesn't parse. */
  private parseAssertion(label: string, assertion: Assertion, where: string | null = null): void {
    // `||`, as Python's `or`: an empty path names the node by its id.
    parseNamedCondition(
      assertion.expression,
      `node ${reprText(where || this.id)} ${label} ${this.shownAssertion(assertion)}`,
      this.types,
    );
  }

  /** Show an assertion in a message: its display, then its expression when that differs. */
  private shownAssertion(assertion: Assertion): string {
    const shown = reprText(assertion.display);
    return assertion.expression === assertion.display ? shown : `${shown} (\`${assertion.expression}\`)`;
  }

  /**
   * Run the node's complete pure-execution template.
   *
   * Declared defaults are filled first, then node-owned setup is validated (including that every
   * assertion and kind predicate parses), the pre-asserts run against the bound record, `forward`
   * runs, and the post-asserts check an `Output` against the inputs as bound. The promise
   * resolves to exactly one `Output | Route | Grow`. Every failure rejects it, with the original
   * error; the checks and the synchronous part of `forward` (up to its first `await`) still run
   * before the promise is returned.
   *
   * @param inputs The bound input record; the template fills defaults into a fresh copy.
   * @param caps The capabilities the node's traits ask for, passed to `forward`.
   */
  run(inputs: Readonly<Record<string, unknown>>, caps: NodeCaps = {}): Promise<NodeResult> {
    return this.runTemplate(inputs, false, null, caps);
  }

  /**
   * Run the base template with the stage markers and input capture. Used by the evaluator.
   *
   * A default's `BindingError` rejects as a `BindingStageError`, an assertion failure as an
   * `AssertionStageError`, and anything else the body throws as a `NodeBodyError`, except four
   * markers, which pass through as they are: `BindingStageError`, `DeferredBindingFault`,
   * `CapabilityFault` and `NodeBodyError`. An `AssertionStageError` thrown by a body is wrapped.
   *
   * @param inputs The bound input record.
   * @param caps The capabilities the node's traits ask for, passed to `forward`.
   * @param captureInputs Receives the completed input record after defaults and before the
   *   assertions or the body, before this returns its promise; never passed to `forward`. The
   *   checks and the synchronous part of `forward` (up to its first `await`) also run before the
   *   promise is returned.
   */
  runForEvaluator(
    inputs: Readonly<Record<string, unknown>>,
    caps: NodeCaps = {},
    captureInputs: ((inputs: Record<string, unknown>) => void) | null = null,
  ): Promise<NodeResult> {
    return this.runTemplate(inputs, true, captureInputs, caps);
  }

  /**
   * The template: fill declared defaults for omitted inputs, check the pre-asserts, run the
   * kind's `forward`, then check the post-asserts against `{...inputs, output}`.
   *
   * Defaults come first, so a pre-assert may read an input the caller omitted but the node
   * declares a default for; a present key, `null` included, shadows its default. START uses this
   * same template, so the parameter seam is its only default authority. The assertions read only
   * the bound record, never the pool: every assert reaching a node is node-local after reference
   * lowering, so the evaluator binds what it reads into the record.
   *
   * @param classify Whether to wrap the stage failures in the evaluator's markers.
   */
  private async runTemplate(
    inputs: Readonly<Record<string, unknown>>,
    classify: boolean,
    captureInputs: ((inputs: Record<string, unknown>) => void) | null,
    caps: NodeCaps,
  ): Promise<NodeResult> {
    // Everything before the first `await` runs inside the caller's call, the synchronous part of
    // `forward` included, so `captureInputs` has run by the time `runForEvaluator` returns;
    // `async` still turns a throw here into a rejection.
    const params = this.params ?? [];
    if (!isList(params)) {
      throw new TypeError(`node ${reprText(this.id)} params must be an exact list, got ${jsTypeName(params)}`);
    }
    for (const parameter of params) {
      if (!isExactly(parameter, Parameter)) {
        throw new TypeError(
          `node ${reprText(this.id)} params must contain Parameter values, got ${jsTypeName(parameter)}`,
        );
      }
    }
    let record: Record<string, unknown>;
    try {
      record = applyDefaults(params, inputs);
    } catch (error) {
      if (classify && error instanceof BindingError) throw new BindingStageError(error);
      throw error;
    }
    if (captureInputs !== null) captureInputs(record);
    this.validateExecutionSetup();
    const preFailure = this.failingAssertion("pre-assert", this.preAsserts, record);
    if (preFailure !== null) throw classify ? new AssertionStageError(preFailure) : preFailure;
    // A snapshot for the post-asserts, taken before `forward`: a body may change the record it
    // receives (a CODE body transforming its input in place), and the post-asserts must see the
    // inputs as bound. Only paid when the node has post-asserts.
    const postRecord = this.postAsserts.length > 0 ? (copyValue(record as ValueRecord) as ValueRecord) : null;
    let result: NodeResult;
    try {
      result = await this.forward(record, caps);
    } catch (error) {
      // The evaluator's own capabilities and deferred binder raise these four inside the body; an
      // assertion marker comes only from this template, so one a body throws is node code.
      if (
        !classify ||
        error instanceof BindingStageError ||
        error instanceof DeferredBindingFault ||
        error instanceof CapabilityFault ||
        error instanceof NodeBodyError
      ) {
        throw error;
      }
      throw new NodeBodyError(error);
    }
    // Only an `Output` has post-asserts checked: a spawner's grow path owns its own.
    if (result instanceof Output && postRecord !== null) {
      // `setField`, so an input named `__proto__` stays data; the snapshot is this run's own.
      setField(postRecord, "output", result.value as Value);
      const postFailure = this.failingAssertion("post-assert", this.postAsserts, postRecord);
      if (postFailure !== null) throw classify ? new AssertionStageError(postFailure) : postFailure;
    }
    return result;
  }

  /**
   * Run the node as a pure function of its bound input record: the kind's body, which the `run`
   * template wraps.
   *
   * Every concrete kind defines it; the evaluator drives `run`, which checks the assertions
   * around this body exactly once. A kind that omits it gets this loud error (the setup check
   * reports it first). The evaluator gives a body error the code the node's `exceptionCode`
   * names (default `node.execute`).
   *
   * @param inputs The bound input record; the body may change it.
   * @param caps The capabilities the node's traits ask for.
   * @returns One `Output | Route | Grow`, or a promise of one.
   * @throws Error always, on the base.
   */
  forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult | Promise<NodeResult> {
    throw new Error(`node ${reprText(this.id)} (${this.constructor.name}) defines no forward`);
  }

  /** Return this node's local variable declarations; the base declares none. */
  declaredVariables(): readonly Parameter[] {
    return [];
  }

  /**
   * Return the tool and tool-group ids a model inside this node may call. The run boundary
   * checks them against the run's tools before any record is written. The base grants none.
   */
  grantedTools(): readonly string[] {
    return [];
  }

  /**
   * Return the tool call this node makes directly, without a model. The run boundary checks it
   * against the run's tools before any record is written. The base makes none.
   */
  hostToolCall(): HostToolCall | null {
    return null;
  }

  /**
   * Return the declared inputs eligible for feedback, in declaration order. Kinds with
   * control-only params override this.
   *
   * @throws TypeError when `params` isn't an exact array of `Parameter`s.
   */
  backwardInputNames(): readonly string[] {
    const parameters = this.params ?? [];
    validateParameters(parameters);
    return parameters.map((parameter) => parameter.name);
  }

  /**
   * Return static backward material before this invocation runs.
   *
   * This pure hook reads authored layout only: it doesn't resolve defaults, render prompts or
   * run the body. Input and variable names are captured separately. With retention on, the base
   * `run` template calls it after defaults and before the assertions or the body; a kind that
   * overrides `run` keeps the evaluator's pre-run boundary.
   *
   * @returns JSON material (infinity is a value; NaN is refused), empty by default. The reserved
   *   `input_output_paths` record addresses declared inputs by paths of string fields or
   *   non-negative list indices into the output.
   */
  captureBackwardContext(): Record<string, unknown> {
    return {};
  }

  /**
   * Return the bound inputs a model never sees (neither recalled on nor shown as evidence). They
   * stay in the bound record, so backward and resume read what ran. Empty by default.
   */
  modelHiddenInputs(): ReadonlySet<string> {
    return new Set();
  }

  /**
   * Return the model config this invocation bound as an input, if any. Backward then records it
   * as the invocation's `effective_llm_config` instead of the run's entry for the node's static
   * path.
   *
   * @param inputs The invocation's bound inputs, reserved keys included.
   * @returns The bound config; `null` by default, which reads the run's recorded map.
   */
  boundLlmConfig(inputs: Readonly<Record<string, unknown>>): Record<string, unknown> | null {
    return null;
  }

  /**
   * Stop feedback at the default behavioral boundary with empty channels. The base makes no
   * model call.
   *
   * @param context Detached recorded inputs, the optional accepted output or failure, the
   *   declared names and node-owned static material for this invocation. Read this snapshot
   *   rather than live node fields when interpreting feedback.
   * @param feedback Ordered independent messages: text, a partial record, an exact-length nested
   *   item list, or `null`; an array is one positional message.
   * @returns The input, self and variable feedback channels; the base returns all three empty.
   */
  backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult | Promise<BackwardResult> {
    return new BackwardResult();
  }

  /**
   * Return the reserved input keys the evaluator pre-resolves from `nodeWiring` and `pool`
   * before `run`, merged into the bound record. WAIT returns `{until: <ISO deadline>}`; MAP
   * returns `{over: <list>}`; the base reserves none and returns `{}`.
   *
   * @param nodeWiring The flow-owned wiring for this node.
   * @param pool The live pool the evaluator reads sources from.
   * @param options.bind Resolves a named source and applies the node-owned check before
   *   retaining its value; it takes the name, the source and the normalizing callable. Omitted
   *   for direct read-hook calls.
   */
  bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    return {};
  }

  /**
   * Return the reserved author-wiring key names this node owns beyond its declared `params`: the
   * keys the wiring parity check allows. They are the static names of the keys `bindReserved`
   * resolves (WAIT: `until`; MAP: `over`), read at load time. The base reserves none.
   */
  reservedWiringKeys(): ReadonlySet<string> {
    return new Set();
  }

  /**
   * Return the author-facing phrase naming a reserved wiring key in a message, without the node:
   * it follows the node, as in `node 'w' wait until`. Default: `<kind> <key>`.
   *
   * @param key One of `reservedWiringKeys()`.
   */
  reservedKeyLabel(key: string): string {
    return `${this.kind} ${key}`;
  }

  /**
   * Return the author-facing phrase for a wiring `param` in a reference error: the label the
   * desugar recorded in `wiringLabels` for a synthesized param (`cases[1].when`, `on:`), else
   * `input '<param>'`.
   */
  wiringRefLabel(param: string): string {
    return this.wiringLabels.get(param) ?? `input ${reprText(param)}`;
  }

  /**
   * Return the first assertion that doesn't hold on `record`, as the error to throw, or `null`.
   *
   * Each assertion is parsed before it is evaluated with this node's type table. A false
   * condition fails with `assertion.failed`; one that throws an `ExpressionError` while
   * evaluating fails with `condition.evaluate`, that error as the `cause`.
   *
   * @throws ExpressionError when an assertion doesn't parse; the message names the node.
   */
  private failingAssertion(
    label: string,
    assertions: readonly Assertion[],
    record: Record<string, unknown>,
  ): NodeAssertFailed | null {
    const subject = `node ${reprText(this.id)} ${label}`;
    for (const assertion of assertions) {
      this.parseAssertion(label, assertion);
      let holds: boolean;
      try {
        holds = evaluateWhenRecord(assertion.expression, record as ValueRecord, this.types);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        return new NodeAssertFailed(
          `${subject} ${this.shownAssertion(assertion)} could not be evaluated: ${error.message}`,
          { assertion, failureCode: "condition.evaluate", cause: error },
        );
      }
      if (!holds) return new NodeAssertFailed(`${subject} failed: ${assertion.display}`, { assertion });
    }
    return null;
  }
}
