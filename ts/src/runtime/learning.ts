/**
 * Pure backward transfers and accepted backward/memory reducer projections.
 *
 * This module shares exact route-transfer validation with the backward callback driver and owns
 * immutable attempt state. It projects owner-local contexts, the only episode projection, for
 * every enabled memory key from complete accepted history and backward results. Contracts,
 * records, and retained traces sit below it; state and callback drivers consume its pure
 * projections. It invokes no callbacks or backend operations.
 *
 * Knows about:   diagnostics, expr/lineage, memory/contracts, nodes/base, nodes/contracts,
 *                runtime/contracts, runtime/log, runtime/trace, typesys.
 * Never imports: runtime state or backward, engine or run, compile, compose, or the CLI.
 */

import { createHash } from "node:crypto";

import { compareCodeUnits, Failure, reprText } from "../diagnostics.ts";
import {
  type FeedbackValue,
  hasFeedback,
  RouteContext,
  type RoutePath,
  routeFeedback,
  SourceRef,
  sameFeedback,
  sameSource,
  sourceKey,
  validateFeedback,
  type ValueRoute,
} from "../expr/lineage.ts";
import {
  DeclSite,
  freezeFeedback,
  type JsonValue as FrozenJson,
  LearnContext,
  LearningFailure,
  LearningResult,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationFailure,
  type MemoryScope,
  MemorySpec,
  MemoryStack,
  PolicyCommitOutcome,
  sameLearningResult,
  sameMemoryMutation,
} from "../memory/contracts.ts";
import {
  BackwardContext,
  BackwardResult,
  copyBackwardValue,
  type LearningMaterial,
  sameBackwardResult,
  validateBackwardResult,
} from "../nodes/base.ts";
import { isExactly, sameExecutionScope, sameItems, sameOptional } from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { isList, isRecord } from "../typesys/values.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardFailure,
  BackwardOutcome,
  sameAttributedBackwardResult,
  sameBackwardAttribution,
  sameBackwardFailure,
  sameBackwardOutcome,
} from "./contracts.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalSeed,
  encodeRecord,
  executionScopePayload,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  routePayload,
  type RunRecord,
  SCHEMA_VERSION,
} from "./log.ts";
import { type OutcomeRecord, RetainedTrace, type TraceOccurrence, TraceTerminal, visibleInputs } from "./trace.ts";

/** The root flow's declaration site: the source-free root channel goes to the keys it owns. */
const ROOT_FLOW_SITE = new DeclSite([], ["root"]);

/** A node-data output path part: a field name, or an index as a decoded `bigint` or a `number`. */
type PathPart = string | bigint | number;

/** Pending messages by their target's `sourceKey`. */
export type PendingMessages = Map<string, BackwardMessage[]>;

/** One target and the message it receives. */
export type Transfer = readonly [SourceRef, BackwardMessage];

/** A transient ordered transfer, with contexts materialized only by consumers. Frozen. */
export class BackwardMessage {
  /** The message, held as given. */
  readonly feedback: FeedbackValue;
  /** Where the message came from. */
  readonly attribution: BackwardAttribution;
  /** The sort key among one receiver's messages, compared item by item. */
  readonly order: readonly number[];
  /** The route contexts the message passed through, in order. */
  readonly context: readonly RouteContext[];

  /** Hold the fields, with frozen copies of `order` and `context`. */
  constructor(
    feedback: FeedbackValue,
    attribution: BackwardAttribution,
    order: readonly number[],
    context: readonly RouteContext[] = [],
  ) {
    this.feedback = feedback;
    this.attribution = attribution;
    this.order = Object.freeze([...order]);
    this.context = Object.freeze([...context]);
    Object.freeze(this);
  }
}

/** Python's tuple order over two message orders. */
function compareOrder(a: readonly number[], b: readonly number[]): number {
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] !== b[index]) return a[index]! - b[index]!;
  }
  return a.length - b.length;
}

/** Remove `source`'s pending messages and return them sorted by `order`; none gives `[]`. */
export function takeMessages(pending: PendingMessages, source: SourceRef): BackwardMessage[] {
  const key = sourceKey(source);
  const messages = pending.get(key) ?? [];
  pending.delete(key);
  // JS sort is stable, as Python's `sorted` is.
  return [...messages].sort((a, b) => compareOrder(a.order, b.order));
}

/** Append `message` to `source`'s pending messages, starting its list when it has none. */
export function addMessage(pending: PendingMessages, source: SourceRef, message: BackwardMessage): void {
  const key = sourceKey(source);
  const messages = pending.get(key);
  if (messages === undefined) pending.set(key, [message]);
  else messages.push(message);
}

/** A path's JSON form: its indexes as `bigint`s, since a JS `number` is a float in learning's JSON. */
function pathData(path: readonly PathPart[]): JsonValue[] {
  return path.map((part) => (typeof part === "string" ? part : BigInt(part)));
}

/** Project a reference without materializing its accepted value. */
function sourceData(source: SourceRef | null): JsonObject | null {
  if (source === null) return null;
  return { run_id: source.runId, execution_seq: BigInt(source.executionSeq), path: pathData(source.path) };
}

/**
 * Materialize each backward route context once per backward pass.
 *
 * Route contexts, their `sources` arrays and snapshots are shared by every contribution below a
 * route node, so rebuilding them per contribution makes long operator chains quadratic. Contexts,
 * source arrays and snapshots are keyed by identity, lookups by `sourceKey`. Callers must not
 * modify the values it returns.
 */
export class ContextCache {
  private readonly trace: RetainedTrace;
  private readonly lookups = new Map<string, JsonValue>();
  private readonly contexts = new Map<RouteContext, JsonObject>();
  private readonly sources = new Map<readonly SourceRef[], JsonObject[]>();
  private readonly results = new Map<string, JsonValue>();
  private readonly snapshots = new Map<ValueRoute, JsonObject>();

  /** Start an empty cache over `trace`, whose accepted sources it looks up. */
  constructor(trace: RetainedTrace) {
    this.trace = trace;
  }

  /**
   * Return the trace's canonical data for `source`, looking it up once.
   *
   * @returns The shared canonical value; callers must not modify it.
   * @throws RangeError when the trace has no such accepted source; failures aren't cached.
   */
  lookup(source: SourceRef): JsonValue {
    const key = sourceKey(source);
    if (this.lookups.has(key)) return this.lookups.get(key)!;
    // Read the method from the instance on each miss: tests replace it there.
    const value = this.trace.lookupSource(source);
    this.lookups.set(key, value);
    return value;
  }

  /**
   * Return the shared materialized form of one route context: accepted source values resolved and
   * unavailable temporary snapshots kept. Callers must not modify it.
   */
  context(context: RouteContext): JsonObject {
    const cached = this.contexts.get(context);
    if (cached !== undefined) return cached;
    const sources = this.sourceValues(context.sources);
    let result: JsonValue;
    if (context.resultJson === null) {
      result = context.resultType === null && sources.length === 1 ? sources[0]!.value! : null;
    } else {
      if (!this.results.has(context.resultJson)) this.results.set(context.resultJson, context.result as JsonValue);
      result = this.results.get(context.resultJson)!;
    }
    const operand = context.operand;
    const detail: JsonObject = {
      expression: context.expression,
      result,
      sources,
      role: context.role,
      operand:
        operand === null || typeof operand === "string"
          ? operand
          : typeof operand === "number"
            ? BigInt(operand)
            : pathData(operand),
      positions: context.positions.map((position) => position.map(BigInt)),
      span: context.span === null ? null : context.span.map(BigInt),
      result_type: context.resultType,
      snapshot: this.snapshot(context.snapshot),
    };
    this.contexts.set(context, detail);
    return detail;
  }

  /** Return the shared source/value pairs for one `sources` array. */
  private sourceValues(sources: readonly SourceRef[]): JsonObject[] {
    const cached = this.sources.get(sources);
    if (cached !== undefined) return cached;
    const values = sources.map((source) => ({ source: sourceData(source), value: this.lookup(source) }));
    this.sources.set(sources, values);
    return values;
  }

  /** Return the shared route payload for one snapshot, or `null` without one. */
  private snapshot(snapshot: ValueRoute | null): JsonObject | null {
    if (snapshot === null) return null;
    const cached = this.snapshots.get(snapshot);
    if (cached !== undefined) return cached;
    const payload = routePayload(snapshot);
    this.snapshots.set(snapshot, payload);
    return payload;
  }
}

/** Resolve accepted values while keeping unavailable temporary snapshots, one per context. */
function routeContextData(contexts: readonly RouteContext[], cache: ContextCache): JsonObject[] {
  return contexts.map((context) => cache.context(context));
}

/**
 * Resolve only the accepted values one transient callback view needs: the message's attribution
 * and its materialized contexts, shared with other messages. Integers are `bigint`s.
 *
 * @param message The message to describe.
 * @param index The message's position among its receiver's messages.
 * @param cache This backward pass's cache over the trace whose accepted sources are resolved.
 */
export function materializeContext(message: BackwardMessage, index: number, cache: ContextCache): JsonObject {
  const attribution = message.attribution;
  return {
    message_index: BigInt(index),
    sender: sourceData(attribution.sender),
    input_name: attribution.inputName,
    sender_message_index: BigInt(attribution.messageIndex),
    contribution_index: attribution.contributionIndex === null ? null : BigInt(attribution.contributionIndex),
    parents: attribution.parents.map(BigInt),
    contexts: routeContextData(message.context, cache),
  };
}

/**
 * Walk the accepted output along a node-data path.
 *
 * @throws RangeError when the path is absent from the output.
 */
function structuralOutput(output: unknown, path: readonly PathPart[]): unknown {
  let value = output;
  for (const part of path) {
    if (typeof part === "string" && isRecord(value) && Object.hasOwn(value, part)) value = value[part];
    else if (typeof part !== "string" && isList(value) && Number(part) < value.length) value = value[Number(part)];
    else throw new RangeError("structural output path is absent from the accepted output");
  }
  return value;
}

/** The node data's `input_output_paths`, or `null` without them. */
function outputPaths(context: BackwardContext): Readonly<Record<string, readonly PathPart[]>> | null {
  const data = context.nodeData;
  return Object.hasOwn(data, "input_output_paths")
    ? (data.input_output_paths as Readonly<Record<string, readonly PathPart[]>>)
    : null;
}

/** Reference an accepted structural output without rebuilding or copying it. */
function outputContext(
  source: SourceRef,
  context: BackwardContext,
  name: string,
  hasOutput: boolean,
): RouteContext[] {
  if (!hasOutput) return [];
  const paths = outputPaths(context);
  if (paths === null || !Object.hasOwn(paths, name)) return [];
  const path = paths[name]!;
  // Only refuses a path absent from the output; the context references the output, not the value.
  structuralOutput(context.output, path);
  const operand: RoutePath = path.map((part) => (typeof part === "string" ? part : Number(part)));
  const positions = path.length > 0 && typeof path[0] !== "string" ? [[Number(path[0])]] : [];
  return [
    new RouteContext({ expression: "", resultJson: null, sources: [source], role: "output", operand, positions }),
  ];
}

/** Keep the ordered parent contexts before adding the actual structural output. */
function inheritedContext(
  occurrence: TraceOccurrence,
  context: BackwardContext,
  messages: readonly BackwardMessage[],
  parents: readonly number[],
  name: string,
  hasOutput: boolean,
): RouteContext[] {
  const inherited = parents.flatMap((index) => messages[index]!.context);
  return [...inherited, ...outputContext(occurrence.source, context, name, hasOutput)];
}

/**
 * Attach transient inherited and output contexts after the pure transfer step.
 *
 * @param trace Accepted terminal forward authority.
 * @param occurrence The occurrence sending feedback.
 * @param context The occurrence's backward context.
 * @param messages The occurrence's ordered incoming messages.
 * @param result The occurrence's validated backward result.
 * @param cache This backward pass's cache; its lookups route the feedback.
 * @returns Each target and its message.
 */
export function contextualTransfers(
  trace: RetainedTrace,
  occurrence: TraceOccurrence,
  context: BackwardContext,
  messages: readonly BackwardMessage[],
  result: BackwardResult,
  cache: ContextCache,
): Transfer[] {
  const hasOutput = trace.sources.has(sourceKey(occurrence.source));
  return transferBackward(trace, occurrence, context, messages, result, (source) => cache.lookup(source)).map(
    ([source, transfer]) => {
      const { attribution } = transfer;
      const inherited = inheritedContext(
        occurrence,
        context,
        messages,
        attribution.parents,
        attribution.inputName!,
        hasOutput,
      );
      const message = new BackwardMessage(transfer.feedback, attribution, transfer.order, [
        ...inherited,
        ...transfer.context,
      ]);
      return [source, message] as const;
    },
  );
}

/**
 * Validate the whole terminal message before deciding whether to seed work.
 *
 * Returns the pending messages (the root message for the terminal's source, or none) and the
 * direct root result: a source-free failed terminal keeps text feedback as self feedback.
 *
 * @throws RangeError when `trace` isn't exactly a `RetainedTrace` with an exact terminal, or the
 *   feedback doesn't fit the terminal output.
 * @throws TypeError when `feedback` isn't ordinary feedback.
 */
export function seedBackward(trace: RetainedTrace, feedback: FeedbackValue): [PendingMessages, BackwardResult] {
  if (!isExactly(trace, RetainedTrace) || !isExactly(trace.terminal, TraceTerminal)) {
    throw new RangeError("backward requires a retained terminal trace");
  }
  const terminal = trace.terminal;
  validateFeedback(feedback, terminal.output);
  const copy = copyBackwardValue(feedback, "root_feedback") as FeedbackValue;
  if (!hasFeedback(copy)) return [new Map(), new BackwardResult()];
  if (terminal.source === null) {
    // A failed terminal's output is null, so its feedback is text.
    const root = terminal.succeeded ? new BackwardResult() : new BackwardResult({ selfFeedback: [copy as string] });
    return [new Map(), root];
  }
  const message = new BackwardMessage(copy, new BackwardAttribution(null, null, 0, null, []), [0]);
  return [new Map([[sourceKey(terminal.source), [message]]]), new BackwardResult()];
}

/**
 * Verify the transfer the node data prescribes before assigning incoming ordinals.
 *
 * Returns each input's parent positions when the result is exactly the structural projection of
 * the messages, else `null`.
 *
 * @throws RangeError when a node-data path is absent from the accepted output.
 */
function structuralParents(
  context: BackwardContext,
  messages: readonly BackwardMessage[],
  result: BackwardResult,
  hasOutput: boolean,
): Map<string, number[]> | null {
  const paths = outputPaths(context);
  const inputs = context.inputs;
  if (paths === null || inputs === null || (!hasOutput && context.failure === null)) return null;
  const expected: [string, FeedbackValue[]][] = [];
  const parents = new Map<string, number[]>();
  for (const name of context.inputNames) {
    if (!Object.hasOwn(paths, name) || !Object.hasOwn(inputs, name)) continue;
    const path = paths[name]!;
    if (hasOutput) structuralOutput(context.output, path);
    const selected: FeedbackValue[] = [];
    const indices: number[] = [];
    messages.forEach((message, index) => {
      let feedback = message.feedback;
      for (const part of path) {
        if (feedback === null || typeof feedback === "string") break;
        const key = typeof part === "string" ? part : Number(part);
        // An own-key read: a partial message that omits the field selects nothing here.
        if (isRecord(feedback)) feedback = Object.hasOwn(feedback, key) ? feedback[key]! : null;
        else feedback = feedback[key as number]!;
      }
      if (hasFeedback(feedback)) {
        selected.push(feedback);
        indices.push(index);
      }
    });
    if (selected.length > 0) {
      expected.push([name, selected]);
      parents.set(name, indices);
    }
  }
  // `fromEntries` defines each key as data, so an input named `__proto__` stays an entry.
  return sameFeedback(result.inputFeedback as FeedbackValue, Object.fromEntries(expected)) ? parents : null;
}

/**
 * Derive exact transfers from accepted routes without invoking or formatting.
 *
 * @param lookup Resolves accepted sources while routing; defaults to `trace.lookupSource`.
 * @returns Each target and its message.
 * @throws RangeError when an input with feedback has no captured route, or for the structural
 *   walk's and `routeFeedback`'s errors.
 */
function transferBackward(
  trace: RetainedTrace,
  occurrence: TraceOccurrence,
  context: BackwardContext,
  messages: readonly BackwardMessage[],
  result: BackwardResult,
  lookup: (source: SourceRef) => unknown = (source) => trace.lookupSource(source),
): Transfer[] {
  const structural = structuralParents(context, messages, result, trace.sources.has(sourceKey(occurrence.source)));
  const route = occurrence.inputRoute;
  const routes = new Map(route.names.map((name, index) => [name, route.children[index]!]));
  const transfers: Transfer[] = [];
  context.inputNames.forEach((name, inputIndex) => {
    const batch = Object.hasOwn(result.inputFeedback, name) ? result.inputFeedback[name]! : [];
    batch.forEach((feedback, messageIndex) => {
      const parents = structural !== null ? [structural.get(name)![messageIndex]!] : messages.map((_, index) => index);
      const input = routes.get(name);
      // Python's `KeyError` on the route lookup.
      if (input === undefined) throw new RangeError(reprText(name));
      const routed = routeFeedback(input, feedback, lookup, { recordedValue: context.inputs![name] });
      routed.contributions.forEach((contribution, contributionIndex) => {
        const attribution = new BackwardAttribution(occurrence.source, name, messageIndex, contributionIndex, parents);
        const order = [...messages[parents[0]!]!.order, inputIndex, messageIndex, contributionIndex];
        const message = new BackwardMessage(contribution.feedback, attribution, order, contribution.context);
        transfers.push([contribution.source, message]);
      });
    });
  });
  return transfers;
}

/**
 * Project the validated START and actual source-zero fields without duplication.
 *
 * Starts from a fresh copy of `rootInputs` (START's input feedback) and appends each root input
 * field another sender's source-zero message selects.
 *
 * @throws RangeError when messages remain for another source: an unavailable or later occurrence.
 */
export function projectRootBackward(
  trace: RetainedTrace,
  pending: PendingMessages,
  rootInputs: Readonly<Record<string, readonly FeedbackValue[]>>,
  directRoot: BackwardResult,
): BackwardResult {
  const sourceZero = new SourceRef(trace.runId, 0);
  const inputs = new Map(Object.entries(rootInputs).map(([name, batch]) => [name, [...batch]]));
  const messages = takeMessages(pending, sourceZero);
  if (messages.length > 0) {
    const initial = trace.lookupSource(sourceZero);
    for (const message of messages) {
      const sender = message.attribution.sender;
      if (trace.rootStartSource !== null && sameOptional(sender, trace.rootStartSource, sameSource)) continue;
      for (const name of trace.rootInputNames) {
        if (!isRecord(initial) || !Object.hasOwn(initial, name)) continue;
        const feedback = message.feedback;
        let selected: FeedbackValue = null;
        if (typeof feedback === "string") selected = feedback;
        else if (isRecord(feedback) && Object.hasOwn(feedback, name)) selected = feedback[name]!;
        if (!hasFeedback(selected)) continue;
        // Each batch in `inputs` is a fresh copy, so it's appended to in place.
        const batch = inputs.get(name);
        if (batch === undefined) inputs.set(name, [selected]);
        else batch.push(selected);
      }
    }
  }
  if (pending.size > 0) throw new RangeError("feedback targets an unavailable or later accepted occurrence");
  return new BackwardResult({ inputFeedback: Object.fromEntries(inputs), selfFeedback: directRoot.selfFeedback });
}

/**
 * Validate a complete reverse result against exact accepted transfer streams.
 *
 * @param trace Accepted terminal forward authority.
 * @param submittedFeedback The original complete root message.
 * @param rootResult The submitted root projection.
 * @param attributedResults Every accepted occurrence in outcome order, unreached empty results
 *   included.
 * @throws TypeError for carriers of another class, results that aren't an array, or malformed
 *   feedback.
 * @throws RangeError for an inventory, transfer, local contract or root disagreement.
 */
export function validateAttributedBackward(
  trace: RetainedTrace,
  submittedFeedback: FeedbackValue,
  rootResult: BackwardResult,
  attributedResults: readonly AttributedBackwardResult[],
): void {
  const [pending, directRoot] = seedBackward(trace, submittedFeedback);
  if (!isExactly(rootResult, BackwardResult)) throw new TypeError("result must be an exact BackwardResult");
  if (!isList(attributedResults)) throw new TypeError("attributed_results must be an array");
  for (const item of attributedResults) {
    if (!isExactly(item, AttributedBackwardResult)) {
      throw new TypeError("attributed_results require exact AttributedBackwardResult values");
    }
  }
  const occurrences = trace.occurrences;
  const sources = (items: readonly { source: SourceRef }[]) => items.map((item) => item.source);
  if (!sameItems(sources(attributedResults), sources(occurrences), sameSource)) {
    throw new RangeError("backward results must match the complete accepted occurrence order");
  }
  let rootInputs: Readonly<Record<string, readonly FeedbackValue[]>> = {};
  for (let index = occurrences.length - 1; index >= 0; index--) {
    const occurrence = occurrences[index]!;
    const item = attributedResults[index]!;
    const messages = takeMessages(pending, occurrence.source);
    const received = messages.map((message) => message.feedback);
    const attribution = messages.map((message) => message.attribution);
    if (
      !sameItems(item.feedback, received, sameFeedback) ||
      !sameItems(item.attribution, attribution, sameBackwardAttribution) ||
      item.invoked !== (messages.length > 0)
    ) {
      throw new RangeError("backward received feedback and attribution must match exact transfers");
    }
    const context = trace.backwardContext(occurrence.source);
    validateBackwardResult(item.result, context);
    if (messages.length === 0) continue;
    if (trace.rootStartSource !== null && sameSource(occurrence.source, trace.rootStartSource)) {
      rootInputs = item.result.inputFeedback;
    }
    for (const [source, transfer] of transferBackward(trace, occurrence, context, messages, item.result)) {
      addMessage(pending, source, transfer);
    }
  }
  if (!sameBackwardResult(rootResult, projectRootBackward(trace, pending, rootInputs, directRoot))) {
    throw new RangeError("backward root result must match the accepted START and source-zero projection");
  }
}

/** A commit observation: the memory commit succeeded or failed. */
export type CommitObservation = MemoryCommitSucceededRecord | MemoryCommitFailedRecord;

/** Accepted computation and commit facts for one explicit attempt. Frozen. */
export class BackwardAttempt {
  /** The attempt's accepted start. */
  readonly started: BackwardStartedRecord;
  /** The attempt's accepted Ready, once computed. */
  readonly ready: BackwardReadyRecord | null;
  /** The attempt's accepted closing record, once closed. */
  readonly finished: BackwardFinishedRecord | null;
  /** The accepted commit requests, in order. */
  readonly requests: readonly MemoryCommitRequestedRecord[];
  /** The accepted commit observations, in order. */
  readonly observations: readonly CommitObservation[];

  /**
   * Require exact record classes of this attempt; the records are held by reference and the
   * arrays copied and frozen.
   *
   * @param started The attempt's accepted start, whose attempt id every other record must share.
   * @param options Each record defaults to `null`, each array to empty.
   * @throws TypeError when a record isn't exactly its class, or a commit array isn't an array.
   * @throws RangeError when a record belongs to another attempt.
   */
  constructor(
    started: BackwardStartedRecord,
    options: {
      ready?: BackwardReadyRecord | null;
      finished?: BackwardFinishedRecord | null;
      requests?: readonly MemoryCommitRequestedRecord[];
      observations?: readonly CommitObservation[];
    } = {},
  ) {
    const { ready = null, finished = null, requests = [], observations = [] } = options;
    const records: [unknown, abstract new (...args: never[]) => { readonly attemptId: number }][] = [
      [started, BackwardStartedRecord],
      [ready, BackwardReadyRecord],
      [finished, BackwardFinishedRecord],
    ];
    for (const [record, carrier] of records) {
      if (record === null && carrier !== BackwardStartedRecord) continue;
      if (!isExactly(record, carrier)) throw new TypeError("backward attempt requires exact accepted records");
      if (record.attemptId !== started.attemptId) throw new RangeError("backward attempt record identities disagree");
    }
    const commits = [
      [requests, [MemoryCommitRequestedRecord]],
      [observations, [MemoryCommitSucceededRecord, MemoryCommitFailedRecord]],
    ] as const;
    for (const [items, carriers] of commits) {
      if (!isList(items)) throw new TypeError("backward commit records must be arrays");
      for (const record of items) {
        if (!carriers.some((carrier) => isExactly(record, carrier))) {
          throw new TypeError("backward commit requires an exact accepted record");
        }
        if (record.attemptId !== started.attemptId) throw new RangeError("backward commit attempt identities disagree");
      }
    }
    this.started = started;
    this.ready = ready;
    this.finished = finished;
    this.requests = Object.freeze([...requests]);
    this.observations = Object.freeze([...observations]);
    Object.freeze(this);
  }
}

/** The failure of an attempt that stopped before it finished. */
function unfinishedFailure(): BackwardFailure {
  return new BackwardFailure("backward", "backward.unfinished", "backward attempt did not finish");
}

/** Project each memory in `memoryOrder` from the attempt's accepted requests and observations. */
function policyOutcomes(memoryOrder: readonly string[], attempt: BackwardAttempt): readonly PolicyCommitOutcome[] {
  const requests = new Map(attempt.requests.map((record) => [record.memory, record]));
  const observations = new Map(attempt.observations.map((record) => [record.memory, record]));
  const outcomes = memoryOrder.map((name) => {
    const request = requests.get(name);
    const observation = observations.get(name);
    if (request === undefined) return new PolicyCommitOutcome(name, "unrequested");
    const commitId = request.commitId;
    if (observation === undefined) return new PolicyCommitOutcome(name, "unknown", { commitId });
    if (isExactly(observation, MemoryCommitFailedRecord)) {
      return new PolicyCommitOutcome(name, "failed", { commitId, failure: observation.failure });
    }
    return new PolicyCommitOutcome(name, "succeeded", { commitId });
  });
  return Object.freeze(outcomes);
}

/**
 * The learning result of the current attempt's accepted prefix, or `null` without memories.
 *
 * Each failed commit is a failure, after `learnerFailure` when given; with no failure the status
 * is `succeeded` when `succeeded` is set, else `incomplete`.
 */
export function prefixLearning(
  backward: RuntimeBackwardState,
  options: { succeeded?: boolean; learnerFailure?: LearningFailure | null } = {},
): LearningResult | null {
  const { succeeded = false, learnerFailure = null } = options;
  if (backward.memoryOrder.length === 0) return null;
  const policies = backward.policyOutcomes;
  const failures = policies
    .filter((item) => item.status === "failed")
    .map((item) => new LearningFailure(item.memory, "commit", item.failure!));
  if (learnerFailure !== null) failures.unshift(learnerFailure);
  const status = failures.length > 0 ? "failed" : succeeded ? "succeeded" : "incomplete";
  return new LearningResult(status, { failures, policies });
}

/**
 * Immutable accepted backward attempt history and per-memory dispositions. Frozen.
 *
 * `memoryOrder` shares the RunStarted authority root. Completed proposals remain inspectable;
 * only new records can advance an attempt. No prefix view resumes work or invokes a callback.
 */
export class RuntimeBackwardState {
  /** Recorded memory keys in table order. */
  readonly memoryOrder: readonly string[];
  /** Accepted attempt history; a later start never inherits earlier proposals. */
  readonly attempts: readonly BackwardAttempt[];

  /**
   * Refuse a malformed key order and impossible accepted-history shapes; the arrays are copied
   * and frozen, the attempts held by reference.
   *
   * @throws TypeError when `memoryOrder` isn't an array of non-empty strings, `attempts` isn't an
   *   array, or an attempt isn't exactly a `BackwardAttempt`.
   * @throws RangeError when the keys aren't sorted and unique, or the attempts break the start,
   *   request or observation order.
   */
  constructor(memoryOrder: readonly string[] = [], attempts: readonly BackwardAttempt[] = []) {
    if (!isList(memoryOrder) || memoryOrder.some((key) => typeof key !== "string" || !key)) {
      throw new TypeError("backward memory order must be an array of non-empty strings");
    }
    if (!sameItems(memoryOrder, [...new Set<string>(memoryOrder)].sort(compareCodeUnits))) {
      throw new RangeError("backward memory order must be sorted and unique");
    }
    if (!isList(attempts)) throw new TypeError("backward history must be an array");
    attempts.forEach((attempt, index) => {
      if (!isExactly(attempt, BackwardAttempt)) throw new TypeError("backward history requires exact attempt states");
      if (attempt.started.attemptId !== index + 1) {
        throw new RangeError("backward attempt IDs must follow accepted start order");
      }
      if (index < attempts.length - 1 && attempt.finished === null) {
        throw new RangeError("an unfinished attempt cannot precede another start");
      }
      const { requests, observations } = attempt;
      const requested = requests.map((record) => record.memory);
      if (!sameItems(requested, memoryOrder.slice(0, requests.length))) {
        throw new RangeError("backward requests must follow recorded memory order");
      }
      if (observations.length > requests.length || requests.length > observations.length + 1) {
        throw new RangeError("backward requests and observations must alternate");
      }
      observations.forEach((observation, resultIndex) => {
        const request = requests[resultIndex]!;
        if (observation.memory !== request.memory || observation.commitId !== request.commitId) {
          throw new RangeError("backward observation must match its accepted request");
        }
        if (isExactly(observation, MemoryCommitFailedRecord) && resultIndex < requests.length - 1) {
          throw new RangeError("a failed commit cannot be followed by another request");
        }
      });
      if (requests.length > 0 && attempt.ready === null) {
        throw new RangeError("backward requests require accepted Ready proposals");
      }
    });
    this.memoryOrder = Object.freeze([...memoryOrder]);
    this.attempts = Object.freeze([...attempts]);
    Object.freeze(this);
  }

  /** The most recently accepted attempt, including a closed one. */
  get current(): BackwardAttempt | null {
    return this.attempts.at(-1) ?? null;
  }

  /** The current attempt's exact accepted submission. */
  get started(): BackwardStartedRecord | null {
    return this.current?.started ?? null;
  }

  /** Completed results and proposals, without treating them as resumable. */
  get ready(): BackwardReadyRecord | null {
    return this.current?.ready ?? null;
  }

  /** The accepted closing record, if this attempt was closed. */
  get finished(): BackwardFinishedRecord | null {
    return this.current?.finished ?? null;
  }

  /** Whether any attempt has an accepted commit request, whatever was observed later. */
  get everRequested(): boolean {
    return this.attempts.some((attempt) => attempt.requests.length > 0);
  }

  /** The attempt id folded into the current attempt's commit ids. */
  get commitAttemptKey(): number | null {
    return this.current?.started.attemptId ?? null;
  }

  /** Each configured memory's disposition from the current attempt's requests and observations. */
  get policyOutcomes(): readonly PolicyCommitOutcome[] {
    const current = this.current;
    if (current === null) {
      return Object.freeze(this.memoryOrder.map((name) => new PolicyCommitOutcome(name, "unrequested")));
    }
    return policyOutcomes(this.memoryOrder, current);
  }

  /**
   * Whether a new attempt may start; the terminal and learn gates stay with the run state.
   *
   * Any finished attempt, whatever its outcome or commit requests, admits a later start. Only an
   * unfinished attempt blocks one.
   */
  get canStart(): boolean {
    return this.current === null || this.finished !== null;
  }

  /** The accepted Finished outcome, or the unfinished outcome derived without appending anything. */
  get outcome(): BackwardOutcome | null {
    const current = this.current;
    if (current === null) return null;
    if (current.finished !== null) return current.finished.outcome;
    return new BackwardOutcome("failed", {
      feedback: current.ready?.rootResult ?? null,
      failure: unfinishedFailure(),
      learning: prefixLearning(this),
    });
  }
}

/** Whether two records have the same canonical text, the class included. */
function sameRecord(a: RunRecord, b: RunRecord): boolean {
  const text = (record: RunRecord) => encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r", 0, record));
  return text(a) === text(b);
}

/**
 * Whether two backward states have the same memory order and attempts, each attempt the same
 * object or with the same records by canonical text; Python's dataclass `==`.
 *
 * The reducer keeps earlier attempts by reference, so the identity check leaves only the changed
 * attempt to compare by text.
 */
export function sameRuntimeBackwardState(a: RuntimeBackwardState, b: RuntimeBackwardState): boolean {
  return (
    sameItems(a.memoryOrder, b.memoryOrder) &&
    sameItems(
      a.attempts,
      b.attempts,
      (x, y) =>
        x === y ||
        (sameRecord(x.started, y.started) &&
          sameOptional<RunRecord>(x.ready, y.ready, sameRecord) &&
          sameOptional<RunRecord>(x.finished, y.finished, sameRecord) &&
          sameItems<RunRecord>(x.requests, y.requests, sameRecord) &&
          sameItems<RunRecord>(x.observations, y.observations, sameRecord)),
    )
  );
}

/**
 * A state over history whose one changed attempt the reducer already checked, sharing
 * `backward`'s memory order.
 */
function withAttempts(backward: RuntimeBackwardState, attempts: readonly BackwardAttempt[]): RuntimeBackwardState {
  // Rechecking every earlier attempt on each record would make the Nth attempt cost O(N) and a
  // replayed log O(N^2), so the constructor is skipped.
  const state = Object.create(RuntimeBackwardState.prototype) as RuntimeBackwardState;
  Object.assign(state, { memoryOrder: backward.memoryOrder, attempts: Object.freeze([...attempts]) });
  return Object.freeze(state);
}

/**
 * Return each recorded memory's batch in table order, empty ones included.
 *
 * @throws RangeError when the proposals don't name the recorded memories in table order.
 */
export function proposalBatches(
  backward: RuntimeBackwardState,
  ready: BackwardReadyRecord,
): (readonly MemoryMutation[])[] {
  const memories = ready.proposals.map((proposal) => proposal.memory);
  if (!sameItems(memories, backward.memoryOrder)) {
    throw new RangeError("memory proposals must cover the complete recorded memory table in order");
  }
  return ready.proposals.map((proposal) => proposal.mutations);
}

/**
 * Require completed feedback and failure dispositions from accepted authority.
 *
 * @throws RangeError when the Finished outcome disagrees with the attempt's accepted records.
 */
function validateBackwardFinished(
  trace: RetainedTrace,
  backward: RuntimeBackwardState,
  record: BackwardFinishedRecord,
): void {
  const current = backward.current!;
  const outcome = record.outcome;
  const ready = current.ready;
  if (ready !== null) {
    if (!sameOptional(outcome.feedback, ready.rootResult, sameBackwardResult)) {
      throw new RangeError("Finished feedback must match accepted Ready");
    }
    if (
      record.attributedResults.length > 0 &&
      !sameItems(record.attributedResults, ready.attributedResults, sameAttributedBackwardResult)
    ) {
      throw new RangeError("Finished attribution must match accepted Ready");
    }
  }
  if (outcome.status === "succeeded") {
    if (ready === null || backward.policyOutcomes.some((policy) => policy.status !== "succeeded")) {
      throw new RangeError("successful Finished requires Ready and every memory confirmation");
    }
    if (!sameOptional(outcome.learning, prefixLearning(backward, { succeeded: true }), sameLearningResult)) {
      throw new RangeError("successful Finished learning must match accepted memory confirmations");
    }
    return;
  }
  // A failed outcome always carries a failure.
  const failure = outcome.failure!;
  if (sameBackwardFailure(failure, unfinishedFailure())) {
    if (!sameBackwardOutcome(outcome, backward.outcome!) || (ready === null && record.attributedResults.length > 0)) {
      throw new RangeError("the Finished of an unfinished attempt must exactly project the accepted prefix");
    }
    return;
  }
  if (ready !== null) {
    const failed = backward.policyOutcomes.filter((policy) => policy.status === "failed");
    if (failed.length !== 1) throw new RangeError("failed Finished after Ready requires a recorded commit failure");
    const policy = failed[0]!;
    const cause = policy.failure!;
    const expected = new BackwardFailure("commit", cause.code, cause.message, { memory: policy.memory, cause });
    if (
      !sameBackwardFailure(failure, expected) ||
      !sameOptional(outcome.learning, prefixLearning(backward), sameLearningResult)
    ) {
      throw new RangeError("commit Finished must match the exact accepted failure and memory states");
    }
    return;
  }
  if (failure.stage === "learn") {
    const { cause, memory } = failure;
    if (!isExactly(cause, MemoryOperationFailure) || memory === null || !backward.memoryOrder.includes(memory)) {
      throw new RangeError("learner failure requires one recorded memory key and memory-operation cause");
    }
    const expected = new BackwardFailure("learn", cause.code, cause.message, { memory, cause });
    let unattributed = failure;
    if (failure.nodeStartedSeq !== null) {
      validateContextFailureOwner(trace, failure);
      const { causeType, traceback } = failure;
      unattributed = new BackwardFailure(failure.stage, failure.code, failure.message, {
        memory,
        cause,
        causeType,
        traceback,
      });
    }
    if (!sameBackwardFailure(unattributed, expected) || outcome.feedback === null) {
      throw new RangeError("learner Finished must retain the exact attributed failure and completed feedback");
    }
    validateAttributedBackward(trace, current.started.feedback, outcome.feedback, record.attributedResults);
    const known = new LearningFailure(memory, "learn", cause);
    if (!sameOptional(outcome.learning, prefixLearning(backward, { learnerFailure: known }), sameLearningResult)) {
      throw new RangeError("learner Finished learning must contain its exact known failure");
    }
    return;
  }
  if (failure.stage !== "backward" || outcome.feedback !== null || record.attributedResults.length > 0) {
    throw new RangeError("incomplete backward computation cannot carry completed or partial results");
  }
  // Called only for its checks: the trace must be terminal and the submitted feedback must fit it.
  seedBackward(trace, current.started.feedback);
  if (
    !hasFeedback(current.started.feedback) ||
    (failure.code !== "backward.execute" && failure.code !== "backward.result")
  ) {
    throw new RangeError("backward computation failure requires effective feedback and a driver failure code");
  }
  if (failure.runId !== trace.runId || failure.nodeStartedSeq === null || failure.memory !== null) {
    throw new RangeError("backward computation failure requires accepted occurrence attribution");
  }
  const source = new SourceRef(trace.runId, failure.nodeStartedSeq);
  acceptedOccurrence(trace, source, "backward computation failure requires an accepted occurrence");
  const context = trace.backwardContext(source);
  if (failure.nodeId !== context.nodeId || !sameOptional(failure.scope, context.scope, sameExecutionScope)) {
    throw new RangeError("backward computation failure must match its recorded invocation");
  }
  if (!sameOptional(outcome.learning, prefixLearning(backward), sameLearningResult)) {
    throw new RangeError("failed computation learning must retain incomplete unrequested memories");
  }
}

/**
 * Return the accepted occurrence at `source`.
 *
 * @throws RangeError with `message` when the trace has no such occurrence.
 */
function acceptedOccurrence(trace: RetainedTrace, source: SourceRef, message: string): TraceOccurrence {
  try {
    return trace.occurrence(source);
  } catch (error) {
    // Python's `KeyError` from the lookup, replaced by the caller's message.
    if (error instanceof RangeError) throw new RangeError(message);
    throw error;
  }
}

/**
 * Require a context failure's node to be an accepted occurrence that receives its memory.
 *
 * @throws RangeError when the failure isn't a context failure of this run, or its occurrence isn't
 *   accepted, doesn't match, or doesn't receive the memory.
 */
function validateContextFailureOwner(trace: RetainedTrace, failure: BackwardFailure): void {
  if (failure.code !== "memory.invalid_context" || failure.runId !== trace.runId) {
    throw new RangeError("an attributed learner failure must be a context failure of this run");
  }
  const source = new SourceRef(trace.runId, failure.nodeStartedSeq!);
  const occurrence = acceptedOccurrence(trace, source, "a context failure must name an accepted occurrence");
  const context = trace.backwardContext(source);
  if (failure.nodeId !== context.nodeId || !sameOptional(failure.scope, context.scope, sameExecutionScope)) {
    throw new RangeError("a context failure must match its recorded occurrence");
  }
  const scope = occurrence.memoryScope;
  if (scope === null || failure.memory === null || !scope.memories.includes(failure.memory)) {
    throw new RangeError("a context failure must name an occurrence that receives its memory");
  }
}

/** A backward lifecycle or attributed commit record. */
export type BackwardRecord =
  | BackwardStartedRecord
  | BackwardReadyRecord
  | BackwardFinishedRecord
  | MemoryCommitRequestedRecord
  | CommitObservation;

/**
 * Validate and project one accepted backward record without effects.
 *
 * @param trace Recorded terminal forward authority.
 * @param backward The immutable accepted attempt prefix.
 * @param record A backward lifecycle or attributed commit record.
 * @returns The validated next immutable attempt projection.
 * @throws TypeError when `backward` isn't exactly a `RuntimeBackwardState`, or for malformed
 *   carriers or result channels.
 * @throws RangeError for an attempt, transfer, proposal or disposition disagreement.
 */
export function reduceBackwardRecord(
  trace: RetainedTrace,
  backward: RuntimeBackwardState,
  record: BackwardRecord,
): RuntimeBackwardState {
  if (!isExactly(backward, RuntimeBackwardState)) {
    throw new TypeError("backward reduction requires RuntimeBackwardState");
  }
  if (isExactly(record, BackwardStartedRecord)) {
    // Called only for its checks: the trace must be terminal and the feedback must fit it.
    seedBackward(trace, record.feedback);
    if (!backward.canStart) throw new RangeError("backward attempt cannot start from the accepted prefix");
    if (record.attemptId !== backward.attempts.length + 1) {
      throw new RangeError("backward attempt_id must be the next accepted start");
    }
    return withAttempts(backward, [...backward.attempts, new BackwardAttempt(record)]);
  }
  const current = backward.current;
  if (current === null || current.finished !== null || record.attemptId !== current.started.attemptId) {
    throw new RangeError("backward record requires its matching unfinished attempt");
  }
  let next: BackwardAttempt;
  if (isExactly(record, BackwardReadyRecord)) {
    if (current.ready !== null) throw new RangeError("backward attempt already has a Ready");
    validateAttributedBackward(trace, current.started.feedback, record.rootResult, record.attributedResults);
    // Called only for its check that the proposals cover the recorded memory table in order.
    proposalBatches(backward, record);
    // Requests need a Ready, so before one the attempt has no commit records to carry over.
    next = new BackwardAttempt(current.started, { ready: record });
  } else if (isExactly(record, BackwardFinishedRecord)) {
    validateBackwardFinished(trace, backward, record);
    const { ready, requests, observations } = current;
    next = new BackwardAttempt(current.started, { ready, finished: record, requests, observations });
  } else if (isExactly(record, MemoryCommitRequestedRecord)) {
    const blocked = backward.policyOutcomes.some((policy) => policy.status === "unknown" || policy.status === "failed");
    if (current.ready === null || blocked) {
      throw new RangeError("backward commit request requires Ready and no pending or failed memory");
    }
    const index = current.requests.length;
    if (index >= backward.memoryOrder.length || record.memory !== backward.memoryOrder[index]) {
      throw new RangeError("backward request must name the next recorded memory");
    }
    const batch = proposalBatches(backward, current.ready)[index]!;
    const expectedId = memoryCommitId(trace.runId, record.memory, batch, current.started.attemptId);
    if (!sameItems(record.mutations, batch, sameMemoryMutation) || record.commitId !== expectedId) {
      throw new RangeError("backward commit batch and identity must match the ordered Ready proposals");
    }
    const { ready, observations } = current;
    next = new BackwardAttempt(current.started, { ready, requests: [...current.requests, record], observations });
  } else if (isExactly(record, MemoryCommitSucceededRecord) || isExactly(record, MemoryCommitFailedRecord)) {
    if (current.requests.length !== current.observations.length + 1) {
      throw new RangeError("backward commit result requires a matching pending request");
    }
    const request = current.requests.at(-1)!;
    if (record.memory !== request.memory || record.commitId !== request.commitId) {
      throw new RangeError("backward commit result must match its pending memory and identity");
    }
    const { ready, requests } = current;
    next = new BackwardAttempt(current.started, { ready, requests, observations: [...current.observations, record] });
  } else {
    throw new RangeError("backward commit requires a completed memory proposal");
  }
  return withAttempts(backward, [...backward.attempts.slice(0, -1), next]);
}

/**
 * Project one canonical durable `Failure` into source-neutral JSON.
 *
 * @throws TypeError when `failure` isn't exactly a `Failure`, or a field has the wrong type.
 * @throws RangeError when its code is unknown.
 */
export function failureJson(failure: Failure): JsonObject {
  if (!isExactly(failure, Failure)) throw new TypeError("failure projection requires a Failure");
  // A Failure isn't frozen, so its fields are checked again before they enter durable JSON.
  new Failure(failure);
  return failure.toJSON();
}

/**
 * Project the actual accepted outcome without assuming model evidence.
 *
 * `committed` is a success's committed value, as `RetainedTrace.lookupSource` reads it from the
 * commit the success names, or an expansion's seed (`TraceOccurrence.seed`); other outcomes ignore
 * it. Returns the episode status, the success
 * output (`null` on failure) and the failure JSON (`null` on success).
 *
 * @throws TypeError when `record` isn't exactly one of the outcome record classes.
 */
export function episodeOutcome(
  record: OutcomeRecord,
  committed: JsonValue,
): [MemoryEpisodeStatus, JsonValue, JsonObject | null] {
  if (isExactly(record, NodeSucceededRecord)) return [MemoryEpisodeStatus.SUCCEEDED, canonicalSeed(committed)[1], null];
  if (isExactly(record, NodeRoutedRecord)) {
    if (record.terminalFailure === null) {
      return [MemoryEpisodeStatus.SUCCEEDED, { kind: "route", handle: record.handle }, null];
    }
    return [MemoryEpisodeStatus.FAILED, null, failureJson(record.terminalFailure)];
  }
  if (isExactly(record, NodeExpandedRecord)) {
    const output = { kind: "grow", seed: canonicalSeed(committed)[1], prune_ids: [...record.pruneIds] };
    return [MemoryEpisodeStatus.SUCCEEDED, output, null];
  }
  if (isExactly(record, NodeFailedRecord)) return [MemoryEpisodeStatus.FAILED, null, failureJson(record.failure)];
  throw new TypeError("unsupported graph outcome for episode projection");
}

/** A receiving memory's learn context can't be built from validated run material. */
export class OwnerContextError extends Error {
  override name = "OwnerContextError";
  /** The receiving memory key. */
  readonly memory: string;
  /** The context-construction error, also the `cause`. */
  readonly original: Error;
  /** The occurrence whose material failed, or `null` when only the source-free root feedback failed. */
  readonly context: BackwardContext | null;

  /**
   * Keep the receiving memory key, the original error and the owner occurrence; the message is
   * the original's.
   *
   * @throws TypeError when `memory` isn't a non-empty string or `context` isn't `null` or exactly a
   *   `BackwardContext`.
   */
  constructor(memory: string, original: Error, context: BackwardContext | null = null) {
    if (typeof memory !== "string" || !memory) {
      throw new TypeError("owner context failure requires a memory key string");
    }
    if (context !== null && !isExactly(context, BackwardContext)) {
      throw new TypeError("owner context failure context must be a BackwardContext or null");
    }
    super(original.message, { cause: original });
    this.memory = memory;
    this.original = original;
    this.context = context;
  }
}

/**
 * Project every local channel, including input messages stopped at literals. Integers are
 * `bigint`s.
 *
 * @param trace Accepted terminal forward authority.
 * @param occurrence The owning occurrence.
 * @param context The occurrence's backward context.
 * @param messages The occurrence's ordered incoming messages.
 * @param result The occurrence's validated backward result.
 * @param cache This backward pass's cache over `trace`.
 * @returns One entry per channel message, in channel order.
 */
function ownerFeedback(
  trace: RetainedTrace,
  occurrence: TraceOccurrence,
  context: BackwardContext,
  messages: readonly BackwardMessage[],
  result: BackwardResult,
  cache: ContextCache,
): JsonObject[] {
  const entries: JsonObject[] = messages.map((message, index) => ({
    channel: "output",
    target_name: null,
    value: message.feedback as JsonValue,
    contribution_context: materializeContext(message, index, cache),
  }));
  const hasOutput = trace.sources.has(sourceKey(occurrence.source));
  const structural = structuralParents(context, messages, result, hasOutput);
  const route = occurrence.inputRoute;
  const routes = new Map(route.names.map((name, index) => [name, route.children[index]!]));
  for (const name of context.inputNames) {
    const batch = Object.hasOwn(result.inputFeedback, name) ? result.inputFeedback[name]! : [];
    batch.forEach((feedback, index) => {
      const parents = structural !== null ? [structural.get(name)![index]!] : messages.map((_, parent) => parent);
      const inherited = inheritedContext(occurrence, context, messages, parents, name, hasOutput);
      // Validation already routed this feedback, so the input has a captured route.
      const routed = routeFeedback(routes.get(name)!, feedback, (source) => cache.lookup(source), {
        recordedValue: context.inputs![name],
      });
      entries.push({
        channel: "input",
        target_name: name,
        value: feedback as JsonValue,
        contribution_context: {
          message_index: BigInt(index),
          parents: parents.map(BigInt),
          contexts: routeContextData(inherited, cache),
          contributions: routed.contributions.map((contribution, contributionIndex) => ({
            source: sourceData(contribution.source),
            value: contribution.feedback as JsonValue,
            contribution_index: BigInt(contributionIndex),
            contexts: routeContextData(contribution.context, cache),
          })),
        },
      });
      routed.stopped.forEach((stopped, stopIndex) => {
        entries.push({
          channel: "stopped",
          target_name: name,
          value: stopped.feedback as JsonValue,
          contribution_context: {
            message_index: BigInt(index),
            stop_index: BigInt(stopIndex),
            parents: parents.map(BigInt),
            contexts: routeContextData([...inherited, ...stopped.context], cache),
          },
        });
      });
    });
  }
  result.selfFeedback.forEach((feedback, index) => {
    entries.push({
      channel: "self",
      target_name: null,
      value: feedback,
      message_index: BigInt(index),
      contribution_context: {},
    });
  });
  const variables = result.variableFeedback;
  for (const name of context.variableNames) {
    if (!Object.hasOwn(variables, name)) continue;
    variables[name]!.forEach((feedback, index) => {
      entries.push({
        channel: "variable",
        target_name: name,
        value: feedback,
        message_index: BigInt(index),
        contribution_context: {},
      });
    });
  }
  return entries;
}

/** Keep the source-free and root-input fallback channels without duplicating START. */
function rootOwnerFeedback(result: BackwardResult): JsonObject[] {
  const entries: JsonObject[] = [];
  for (const [name, batch] of Object.entries(result.inputFeedback)) {
    batch.forEach((feedback, index) => {
      entries.push({
        channel: "input",
        target_name: name,
        value: feedback as JsonValue,
        contribution_context: { message_index: BigInt(index) },
      });
    });
  }
  result.selfFeedback.forEach((feedback, index) => {
    entries.push({
      channel: "self",
      target_name: null,
      value: feedback,
      message_index: BigInt(index),
      contribution_context: {},
    });
  });
  for (const [name, batch] of Object.entries(result.variableFeedback)) {
    batch.forEach((feedback, index) => {
      entries.push({
        channel: "variable",
        target_name: name,
        value: feedback,
        message_index: BigInt(index),
        contribution_context: {},
      });
    });
  }
  const coordinates = { execution_seq: null, owner_static_path: null, occurrence_path: null, execution_scope: null };
  return entries.map((entry) => ({ ...coordinates, root: true, ...entry }));
}

/**
 * Build every enabled memory's context from a validated complete backward result.
 *
 * A memory receives an occurrence when the occurrence's recorded scope names its key. The
 * source-free root channel goes to every memory owned by the root flow.
 *
 * @param trace Accepted terminal executions and historical scopes.
 * @param memories The complete recorded key table, including keys without executed nodes.
 * @param stacks The recorded per-target key stacks.
 * @param submittedFeedback The original ordinary root message.
 * @param rootResult The completed root projection.
 * @param attributedResults The complete accepted occurrence inventory and validated local results.
 * @param options `cache` is the backward pass's route-context cache over `trace`, shared with the
 *   backward driver; without one a new one starts.
 * @returns Detached episode and feedback batches in table order. Feedback keeps the accepted
 *   execution coordinates; the memory scope selects receivers and episode ownership without
 *   relabeling that feedback.
 * @throws TypeError or RangeError for an invalid table, stacks, trace or backward authority, or
 *   missing accepted source authority while materializing.
 * @throws OwnerContextError when validated material can't satisfy a receiving memory's JSON
 *   contract. It names the occurrence whose episode or feedback failed, or none when only the root
 *   feedback failed. No callback or backend is invoked.
 */
export function projectOwnerContexts(
  trace: RetainedTrace,
  memories: readonly MemorySpec[],
  stacks: readonly MemoryStack[],
  submittedFeedback: FeedbackValue,
  rootResult: BackwardResult,
  attributedResults: readonly AttributedBackwardResult[],
  options: { cache?: ContextCache | null; materials?: ReadonlyMap<string, LearningMaterial> | null } = {},
): readonly LearnContext[] {
  if (!isList(memories) || memories.some((spec) => !isExactly(spec, MemorySpec))) {
    throw new TypeError("owner contexts require the complete recorded memory table");
  }
  if (!isList(stacks) || stacks.some((stack) => !isExactly(stack, MemoryStack))) {
    throw new TypeError("owner contexts require the recorded memory stacks");
  }
  const keys = memories.map((spec) => spec.key);
  if (!sameItems(keys, [...new Set(keys)].sort(compareCodeUnits))) {
    throw new RangeError("owner contexts require a memory table sorted by unique key");
  }
  const known = new Set(keys);
  const recorded = new Map(stacks.map((stack) => [JSON.stringify(stack.locator), stack.memories]));
  if (recorded.size !== stacks.length) throw new RangeError("recorded memory stacks must have unique locators");
  validateAttributedBackward(trace, submittedFeedback, rootResult, attributedResults);
  const occurrences = trace.occurrences;
  const scopes = new Map<string, MemoryScope>();
  for (const occurrence of occurrences) {
    const scope = occurrence.memoryScope;
    if (scope === null) continue;
    const stack = recorded.get(JSON.stringify(scope.staticPath));
    if (stack === undefined || !sameItems(stack, scope.memories)) {
      throw new RangeError("historical memory scope disagrees with its recorded memory stack");
    }
    if (!scope.memories.every((key) => known.has(key))) {
      throw new RangeError("historical memory scope names a key absent from the recorded table");
    }
    if (scope.memories.length > 0) scopes.set(sourceKey(occurrence.source), scope);
  }

  const cache = options.cache ?? new ContextCache(trace);
  const [pending, directRoot] = seedBackward(trace, submittedFeedback);
  const feedbackBySource = new Map<string, JsonObject[]>();
  for (let index = occurrences.length - 1; index >= 0; index--) {
    const occurrence = occurrences[index]!;
    const item = attributedResults[index]!;
    const messages = takeMessages(pending, occurrence.source);
    const context = trace.backwardContext(occurrence.source);
    const key = sourceKey(occurrence.source);
    if (scopes.has(key)) {
      feedbackBySource.set(key, ownerFeedback(trace, occurrence, context, messages, item.result, cache));
    }
    for (const [source, transfer] of contextualTransfers(trace, occurrence, context, messages, item.result, cache)) {
      addMessage(pending, source, transfer);
    }
  }
  const fallback = rootOwnerFeedback(projectRootBackward(trace, pending, {}, directRoot));

  const contexts: LearnContext[] = [];
  // One memo across every freeze here: occurrences and the root fallback share subtrees.
  const memo = new Map<object, FrozenJson>();
  for (const spec of memories) {
    const episodes: MemoryEpisode[] = [];
    const frozen: FrozenJson[] = [];
    for (const occurrence of occurrences) {
      const key = sourceKey(occurrence.source);
      const scope = scopes.get(key);
      if (scope === undefined || !scope.memories.includes(spec.key)) continue;
      const record = occurrence.record;
      const capture = occurrence.capture;
      const material = options.materials?.get(key) ?? null;
      // A success's committed value, or an expansion's seed, which neither record holds.
      const committed = isExactly(record, NodeSucceededRecord)
        ? trace.lookupSource(occurrence.source)
        : isExactly(record, NodeExpandedRecord)
          ? occurrence.seed
          : null;
      const [status, output, failure] = episodeOutcome(record, committed);
      const executionScope = executionScopePayload(capture.scope);
      const coordinates = {
        execution_seq: BigInt(occurrence.source.executionSeq),
        owner_static_path: executionScope === null ? null : executionScope.static_path!,
        occurrence_path: executionScope === null ? null : executionScope.occurrence_path!,
        execution_scope: executionScope,
      };
      const local = feedbackBySource.get(key)!;
      const entries = local.map((entry) => ({ ...coordinates, ...entry }));
      // Episodes reach learn prompts and storage; inputs that only configured the call stay in the
      // bound record for backward and resume.
      const inputs = visibleInputs(occurrence);
      try {
        episodes.push(
          new MemoryEpisode(trace.runId, scope.occurrencePath, scope.staticPath, status, inputs, {
            output,
            failure,
            interactions: material === null ? [] : material.interactions,
            baseSystemPrompt: material === null ? null : material.baseSystemPrompt,
            prompt: material === null ? null : material.prompt,
            memorySections: material === null ? [] : material.memorySections,
            executionSeq: occurrence.source.executionSeq,
          }),
        );
        // Freezing this occurrence's feedback alone attributes a failure to it, under the local
        // indices and run/key checks a context of only these entries reports.
        LearnContext.fromFrozen(trace.runId, spec.key, [], []);
        frozen.push(...freezeFeedback(entries, memo));
      } catch (error) {
        if (error instanceof TypeError || error instanceof RangeError) {
          throw new OwnerContextError(spec.key, error, trace.backwardContext(occurrence.source));
        }
        throw error;
      }
    }
    try {
      if (spec.owners.some((site) => DeclSite.compare(site, ROOT_FLOW_SITE) === 0)) {
        // A fresh context checks its run, key and episodes before its feedback.
        LearnContext.fromFrozen(trace.runId, spec.key, episodes, []);
        frozen.push(...freezeFeedback(fallback, memo, frozen.length));
      }
      contexts.push(LearnContext.fromFrozen(trace.runId, spec.key, episodes, Object.freeze(frozen)));
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) throw new OwnerContextError(spec.key, error);
      throw error;
    }
  }
  return Object.freeze(contexts);
}

/**
 * Return the sole deterministic ID for one memory's mutation batch in one attempt.
 *
 * The SHA-256 input is the UTF-8 run-log text (`logJson`: keys in code-unit order, ints of any
 * size, floats marked by `.0`, `-0` written as `0.0`) of
 * `[runId, attemptId, memory, [[mutationId, payload], ...]]`, the attempt id written as an int.
 * Batch order is semantic. The result is
 * `ac-memory-v3:<hex digest>`.
 *
 * @throws RangeError when `attemptId` isn't a positive safe integer, or `runId` or `memory` isn't
 *   a non-empty string.
 * @throws TypeError when `mutations` isn't an array or an item isn't exactly a `MemoryMutation`.
 */
export function memoryCommitId(
  runId: string,
  memory: string,
  mutations: readonly MemoryMutation[],
  attemptId: number,
): string {
  if (typeof attemptId !== "number" || !Number.isSafeInteger(attemptId) || attemptId < 1) {
    throw new RangeError("attempt_id must be a positive integer");
  }
  if (typeof runId !== "string" || runId === "") throw new RangeError("run_id must be a non-empty string");
  if (typeof memory !== "string" || memory === "") throw new RangeError("memory must be a non-empty string");
  if (!isList(mutations)) throw new TypeError("mutations must be an array");
  const batch = mutations.map((mutation) => {
    if (!isExactly(mutation, MemoryMutation)) throw new TypeError("mutations items must be MemoryMutation values");
    return [mutation.mutationId, mutation.payload];
  });
  // A plain number would be written as a float (`1.0`), so the attempt id is a bigint.
  // The run log's canonical form writes a carrier's -0 as `0.0`; logJson alone refuses -0.
  const text = logJson(canonicalSeed([runId, BigInt(attemptId), memory, batch])[0]);
  return `ac-memory-v3:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}
