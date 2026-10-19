/**
 * Pure backward transfers and accepted backward/memory reducer projections.
 *
 * This module shares exact route-transfer validation with the backward callback driver and owns
 * immutable attempt state. It projects owner-local contexts, the only episode projection, for
 * every enabled memory key from complete accepted history and backward results. Contracts,
 * records, and retained traces sit below it; state and callback drivers consume its pure
 * projections. It invokes no callbacks or backend operations.
 *
 * Knows about:   diagnostics, expr/lineage, memory, nodes/base, runtime/contracts, runtime/log,
 *                runtime/trace, typesys.
 * Never imports: runtime state or backward, engine or run, compile, compose, or the CLI.
 */

import { createHash } from "node:crypto";

import { Failure, reprText } from "../diagnostics.ts";
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
import { MemoryEpisodeStatus, MemoryMutation } from "../memory/index.ts";
import {
  type BackwardContext,
  BackwardResult,
  copyBackwardValue,
  sameBackwardResult,
  validateBackwardResult,
} from "../nodes/base.ts";
import { isExactly, sameItems, sameOptional } from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { isList, isRecord } from "../typesys/values.ts";
import { AttributedBackwardResult, BackwardAttribution, sameBackwardAttribution } from "./contracts.ts";
import {
  canonicalSeed,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  routePayload,
} from "./log.ts";
import { type OutcomeRecord, RetainedTrace, type TraceOccurrence, TraceTerminal } from "./trace.ts";

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

/** Append `message` to `source`'s pending messages. */
function addMessage(pending: PendingMessages, source: SourceRef, message: BackwardMessage): void {
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
 * @param _trace The trace whose accepted sources are resolved; `cache` reads it, so it's unused.
 * @param message The message to describe.
 * @param index The message's position among its receiver's messages.
 * @param cache This backward pass's cache over `trace`.
 */
export function materializeContext(
  _trace: RetainedTrace,
  message: BackwardMessage,
  index: number,
  cache: ContextCache,
): JsonObject {
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
        if (hasFeedback(selected)) inputs.set(name, [...(inputs.get(name) ?? []), selected]);
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
      item.invoked !== messages.length > 0
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
 * Returns the episode status, the success output (`null` on failure) and the failure JSON (`null`
 * on success).
 *
 * @throws RangeError when a success has no canonical typed value.
 * @throws TypeError when `record` isn't exactly one of the outcome record classes.
 */
export function episodeOutcome(record: OutcomeRecord): [MemoryEpisodeStatus, JsonValue, JsonObject | null] {
  if (isExactly(record, NodeSucceededRecord)) {
    const wire: unknown = record.value;
    if (!isRecord(wire) || !Object.hasOwn(wire, "value")) {
      throw new RangeError("successful record has no canonical typed value");
    }
    return [MemoryEpisodeStatus.SUCCEEDED, canonicalSeed(wire.value)[1], null];
  }
  if (isExactly(record, NodeRoutedRecord)) {
    if (record.terminalFailure === null) {
      return [MemoryEpisodeStatus.SUCCEEDED, { kind: "route", handle: record.handle }, null];
    }
    return [MemoryEpisodeStatus.FAILED, null, failureJson(record.terminalFailure)];
  }
  if (isExactly(record, NodeExpandedRecord)) {
    const output = { kind: "grow", seed: canonicalSeed(record.seed)[1], prune_ids: [...record.pruneIds] };
    return [MemoryEpisodeStatus.SUCCEEDED, output, null];
  }
  if (isExactly(record, NodeFailedRecord)) return [MemoryEpisodeStatus.FAILED, null, failureJson(record.failure)];
  throw new TypeError("unsupported graph outcome for episode projection");
}

/**
 * Return the sole deterministic ID for one memory's mutation batch in one attempt.
 *
 * The SHA-256 input is the UTF-8 run-log text (`logJson`: keys in code-unit order, ints of any
 * size, floats marked by `.0`) of `[runId, attemptId, memory, [[mutationId, payload], ...]]`, the
 * attempt id written as an int. Batch order is semantic. The result is
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
  const text = logJson([runId, BigInt(attemptId), memory, batch]);
  return `ac-memory-v3:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}
