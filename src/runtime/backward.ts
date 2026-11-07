/**
 * Reverse computation and explicit, record-authorized backward attempts.
 *
 * The fold consumes recorded routes and node-owned backward hooks. It does not interpret kinds,
 * evaluate expressions, drive child flows, or write logs/memory. Small durable attribution
 * references live in `runtime/contracts.ts` below this module; materialized interpretation data and
 * ordering keys remain transient. The shared learning projection owns formatting without
 * performing this callback fold. The internal attempt driver surrounds that pure computation with
 * accepted records and injected memory resources. It never resumes proposals or invokes public run
 * lifecycle operations.
 *
 * Never imports: compile, compose, the engine, run or the CLI.
 */

import { reprText } from "../diagnostics.ts";
import { ReplayDegraded } from "../events.ts";
import { type FeedbackValue, sameSource, sourceKey } from "../expr/lineage.ts";
import { ToolSpec } from "../llm/messages.ts";
import {
  LearningFailure,
  MemoryContext,
  MemoryOperationFailure,
  RecallContext,
  sameMemorySnapshotEnvelope,
  sameMemorySpec,
} from "../memory/contracts.ts";
import {
  type BackwardContext,
  type BackwardOptions,
  BackwardResult,
  copyBackwardValue,
  failureMessage,
  type LearningMaterial,
  type MaterialCaps,
  tracebackText,
  validateBackwardResult,
} from "../nodes/base.ts";
import { Skill } from "../skills/catalog.ts";
import { isExactly, sameItems, sameOptional } from "../nodes/contracts.ts";
import {
  expandGrants,
  GrantError,
  type GrantedTool,
  type GrantScope,
  requestToolSpecs,
  type ToolGrant,
} from "../tools/grants.ts";
import { jsTypeName } from "../typesys/values.ts";
import { AttributedBackwardResult, BackwardFailure, BackwardOutcome, type MemoryProposal } from "./contracts.ts";
import {
  addMessage,
  ContextCache,
  contextualTransfers,
  materializeContext,
  memoryCommitId,
  OwnerContextError,
  prefixLearning,
  projectOwnerContexts,
  projectRootBackward,
  proposalBatches,
  seedBackward,
  takeOccurrenceMessages,
} from "./learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  isRunLog,
  type RunLog,
  type RunRecord,
} from "./log.ts";
import { LearningProposalError, MemoryRuntime } from "./memory_runtime.ts";
import { commitPrepared, prepareRecord, RunState } from "./state.ts";
import { NodeDataError, occurrenceMaterial, type RetainedTrace, type TraceOccurrence, visibleInputs } from "./trace.ts";

/** @internal A callback or result-contract failure attributed to one occurrence; forward state is unchanged. */
export class BackwardComputeError extends Error {
  override name = "BackwardComputeError";
  /** The attributed failure; the error's message is its message. */
  readonly failure: BackwardFailure;

  /** Keep `failure`, and the original error as the `cause`. */
  constructor(failure: BackwardFailure, options: { cause?: unknown } = {}) {
    super(failure.message, options);
    this.failure = failure;
  }
}

/** Return `error` as a `code` failure of the occurrence `at` describes (its backward context's attribution). */
function computeFailure(
  at: Pick<BackwardContext, "scope" | "runId" | "nodeId" | "nodeStartedSeq">,
  code: string,
  error: unknown,
): BackwardComputeError {
  return new BackwardComputeError(
    new BackwardFailure("backward", code, failureMessage(error), {
      scope: at.scope,
      runId: at.runId,
      nodeId: at.nodeId,
      nodeStartedSeq: at.nodeStartedSeq,
      causeType: error instanceof Error ? error.name : jsTypeName(error),
      traceback: tracebackText(error),
    }),
    { cause: error },
  );
}

/**
 * Project root feedback through one accepted-outcome reverse walk.
 *
 * @param trace A retained terminal execution with private backward callbacks.
 * @param rootFeedback One ordinary text, record, positional array, or `null` value selecting the
 *   actual terminal output. Failed terminals accept text or `null`.
 * @param options.cache The pass's route-context cache over `trace`, so the owner projection can
 *   reuse it; `null` or omitted starts a new one.
 * @param options.models Builds the model a callback calls (AGENT backward); passed to each
 *   callback, which defaults it to `defaultModelAccess()`.
 * @param options.onEvent Receives the transient events callbacks publish; passed to each callback.
 * @returns Root input feedback and results in accepted-outcome order, including unreached
 *   occurrences whose callbacks were not invoked.
 * @throws TypeError (as a rejection, as are all below) for malformed feedback or trace carriers.
 * @throws RangeError for an unfinished trace, invalid feedback shape, or broken retained
 *   route/source authority.
 * @throws Error when an occurrence that receives feedback has no retained handler.
 * @throws BackwardComputeError for an attributed callback or result-contract error; any error a
 *   callback throws, a host capability fault included, is attributed, and so is node data an
 *   occurrence that receives feedback can't rebuild (`NodeDataError`, as `backward.execute`).
 */
export async function computeBackward(
  trace: RetainedTrace,
  rootFeedback: FeedbackValue,
  options: { cache?: ContextCache | null; materials?: LearningMaterials | null } & BackwardOptions = {},
): Promise<[BackwardResult, readonly AttributedBackwardResult[]]> {
  const { models, onEvent } = options;
  const materials = options.materials ?? new Map<string, LearningMaterial>();
  const cache = options.cache ?? new ContextCache(trace);
  const [pending, directRoot] = seedBackward(trace, rootFeedback);
  const occurrences = trace.occurrences;
  const results = new Map(
    occurrences.map((item) => [
      sourceKey(item.source),
      new AttributedBackwardResult(item.source, [], [], new BackwardResult(), false),
    ]),
  );
  let rootInputs: Readonly<Record<string, readonly FeedbackValue[]>> = {};
  for (const occurrence of [...occurrences].reverse()) {
    const source = occurrence.source;
    const messages = takeOccurrenceMessages(pending, occurrence);
    if (messages.length === 0) continue;
    // Trace lookups stay outside the callback's `try`: their faults aren't the callback's. Node
    // data that can't be rebuilt is the occurrence's failure, attributed as the callback's would be.
    let context: BackwardContext;
    try {
      context = trace.backwardContext(source, {
        downstreamContext: messages.map((message, i) => materializeContext(message, i, cache)),
        material: materials.get(sourceKey(source)) ?? null,
      });
    } catch (error) {
      if (!(error instanceof NodeDataError)) throw error;
      throw computeFailure(error, "backward.execute", error);
    }
    const received = messages.map((item) => item.feedback);
    const feedback = Object.freeze(received.map((message) => copyBackwardValue(message, "feedback") as FeedbackValue));
    const handler = trace.backwardHandlers.get(sourceKey(source));
    if (handler === undefined) throw new Error("backward occurrence has no retained handler");
    let result: BackwardResult;
    try {
      result = await handler(context, feedback, { models, onEvent });
    } catch (error) {
      throw computeFailure(context, "backward.execute", error);
    }
    try {
      validateBackwardResult(result, context);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) {
        throw computeFailure(context, "backward.result", error);
      }
      throw error;
    }
    results.set(
      sourceKey(source),
      new AttributedBackwardResult(source, received, messages.map((item) => item.attribution), result, true),
    );
    if (sameOptional(source, trace.rootStartSource, sameSource)) rootInputs = result.inputFeedback;
    for (const [target, transfer] of contextualTransfers(trace, occurrence, context, messages, result, cache)) {
      addMessage(pending, target, transfer);
    }
  }
  const root = projectRootBackward(trace, pending, rootInputs, directRoot);
  return [root, occurrences.map((item) => results.get(sourceKey(item.source))!)];
}

/**
 * Append and reduce one exact observer-free backward transition; its prepare and commit run in one
 * synchronous step, so no other drive can advance the state in between.
 */
function acceptBackward(state: RunState, log: RunLog, record: RunRecord): void {
  if (commitPrepared(log, state, prepareRecord(state, record)).length > 0) {
    throw new Error("backward records cannot emit observers");
  }
}

/**
 * Finish attempt `attemptId` and return only the outcome of the accepted Finished record. A drive
 * finishes only its own attempt: the reducer refuses a Finished whose attempt is no longer the
 * unfinished current one.
 */
function finishBackward(
  state: RunState,
  log: RunLog,
  attemptId: number,
  outcome: BackwardOutcome,
  attributed: readonly AttributedBackwardResult[] = [],
): BackwardOutcome {
  acceptBackward(state, log, new BackwardFinishedRecord(attemptId, outcome, attributed));
  return state.backwardState.finished!.outcome;
}

/** Each accepted occurrence's learning material by its source's `sourceKey`; one without any is absent. */
export type LearningMaterials = ReadonlyMap<string, LearningMaterial>;

/**
 * Rebuild each accepted occurrence's learning material once for a backward pass
 * (`occurrenceMaterial`), from the run's recorded skills and tool specs and, for an occurrence
 * with memories, a recall against `memory`, opened on the recorded snapshots.
 *
 * An occurrence whose rebuilt requests don't match its recorded digests has no material, so
 * learning reads no prompt, memory sections or interactions of it (its backward context's
 * `_evaluation_evidence` is `null`, and its episodes keep their inputs, output and feedback). It
 * is reported as a `ReplayDegraded` naming its node (reason `rebuilt_request_differs`). Besides a
 * changed request, a render or recall that fails now, a structured model that can't be built and
 * a tool the recorded specs lack fail the check too. A grant naming a tool group is expanded with
 * the groups the run recorded, so a run reopened without its tools learns the same.
 *
 * @param options `memory` recalls for scoped occurrences (`null` recalls nothing); `models`
 *   builds a structured call's model to read its plan off; `onEvent` receives the reports.
 */
export async function learningMaterials(
  state: RunState,
  options: {
    memory: MemoryRuntime | null;
    models: BackwardOptions["models"] | null;
    onEvent: BackwardOptions["onEvent"] | null;
  },
): Promise<LearningMaterials> {
  const { memory, models, onEvent } = options;
  const trace = state.trace!;
  const { skills, toolSpecs } = recordedGrants(state);
  // Recalls for `occurrence` against the recorded snapshots, as its run did.
  const recallFor =
    (occurrence: TraceOccurrence) =>
    async (systemPrompt: string, prompt: string): Promise<MemoryContext> => {
      const scope = occurrence.memoryScope;
      if (scope === null || scope.memories.length === 0 || memory === null) return new MemoryContext();
      const recall = new RecallContext(systemPrompt, prompt, visibleInputs(occurrence) ?? {}, scope.staticPath);
      return await memory.openContext(scope, recall, { signal: new AbortController().signal });
    };
  const materials = new Map<string, LearningMaterial>();
  for (const occurrence of trace.occurrences) {
    const material = await occurrenceMaterial(trace, occurrence.source, {
      skills,
      toolSpecs,
      models: models ?? null,
      openMemoryContext: recallFor(occurrence),
    });
    if (material === null) continue;
    if (!material.digestOk) {
      onEvent?.(new ReplayDegraded({ nodeId: occurrence.nodeId, reason: "rebuilt_request_differs" }));
      continue;
    }
    materials.set(sourceKey(occurrence.source), material);
  }
  return materials;
}

/**
 * Return what the run recorded of its grants, as a `MaterialCaps` reads them: the skills by name,
 * and a request's tool list for some grants, expanded as the live loop expands them but over the
 * groups and tools the run recorded (`null` when a grant names no recorded tool).
 */
export function recordedGrants(state: RunState): Pick<MaterialCaps, "skills" | "toolSpecs"> {
  const skills = new Map(
    Object.entries(state.recordedSkills).map(([name, { description, instructions }]) => [
      name,
      new Skill({ name, description, instructions }),
    ]),
  );
  const scope = recordedScope(state);
  const toolSpecs = (grants: readonly ToolGrant[], loadSkill: ToolSpec | null) => {
    let granted: GrantedTool[];
    try {
      granted = expandGrants(grants, [scope]);
    } catch (error) {
      if (error instanceof GrantError) return null;
      throw error;
    }
    const specOf = (tool: GrantedTool) => {
      if (!Object.hasOwn(state.recordedTools, tool.key)) return null;
      const { name, description, parameters } = state.recordedTools[tool.key]!;
      return new ToolSpec({ name, description, parameters });
    };
    return requestToolSpecs(granted, specOf, loadSkill);
  };
  return { skills, toolSpecs };
}

/**
 * Return the run's recorded tools as the grant scope its `ToolSet` was: the recorded groups by
 * name, each member by its own name; a top-level tool is a recorded tool no group lists; and a
 * tool's identity is its record name, so grants expand and dedupe as they did live.
 */
function recordedScope(state: RunState): GrantScope {
  const groups = state.recordedToolGroups;
  const tools = state.recordedTools;
  const members = (group: string): readonly string[] =>
    Object.hasOwn(groups, group) ? groups[group]! : [];
  const nameOf = (key: string) => (Object.hasOwn(tools, key) ? tools[key]!.name : null);
  const listed = new Set(Object.values(groups).flat());
  return {
    label: "this run",
    bareMembers: true,
    group: (name) =>
      Object.hasOwn(groups, name) ? members(name).map((key) => nameOf(key) ?? key) : null,
    hasTool: (name) => Object.hasOwn(tools, name) && !listed.has(name),
    groupsHolding: (name) => {
      const seen = new Set<string>();
      const holding: string[] = [];
      for (const group of Object.keys(groups)) {
        const key = members(group).find((member) => nameOf(member) === name);
        if (key === undefined || seen.has(key)) continue;
        seen.add(key);
        holding.push(group);
      }
      return holding;
    },
    groupNames: () => Object.keys(groups),
    identity: (group, tool) =>
      group === null ? tool : (members(group).find((key) => nameOf(key) === tool) ?? tool),
  };
}

/** Compute once, stage every memory, then request and observe each memory key in order. */
async function driveOpenedBackward(
  state: RunState,
  log: RunLog,
  memory: MemoryRuntime | null,
  options: BackwardOptions,
): Promise<BackwardOutcome> {
  const started = state.backwardState.started!;
  const attemptId = started.attemptId;
  const trace = state.trace!;
  const cache = new ContextCache(trace);
  const materials = await learningMaterials(state, {
    memory,
    models: options.models ?? null,
    onEvent: options.onEvent ?? null,
  });
  let root: BackwardResult;
  let attributed: readonly AttributedBackwardResult[];
  try {
    [root, attributed] = await computeBackward(trace, started.feedback, {
      cache,
      materials,
      models: options.models,
      onEvent: options.onEvent,
    });
  } catch (error) {
    if (!(error instanceof BackwardComputeError)) throw error;
    return finishBackward(
      state,
      log,
      attemptId,
      new BackwardOutcome("failed", { failure: error.failure, learning: prefixLearning(state.backwardState) }),
    );
  }
  let proposals: readonly MemoryProposal[];
  try {
    const contexts = projectOwnerContexts(
      trace,
      state.memories ?? [],
      state.memoryStacks ?? [],
      started.feedback,
      attributed,
      { cache, materials },
    );
    proposals = memory === null ? [] : await memory.proposeLearning(contexts);
  } catch (error) {
    let key: string;
    let context: BackwardContext | null = null;
    let cause: MemoryOperationFailure;
    if (error instanceof OwnerContextError) {
      key = error.memory;
      context = error.context;
      let message = "memory learn context could not be constructed";
      // A trace's backward context always names its occurrence's node.
      if (context !== null) message += ` for node ${reprText(context.nodeId!)}`;
      // A MemoryOperationFailure refuses a lone surrogate, which the original's text may hold.
      const text = `${message}: ${failureMessage(error.original)}`.toWellFormed();
      cause = new MemoryOperationFailure("learn", "memory.invalid_context", text, false);
    } else if (error instanceof LearningProposalError) {
      key = error.memory;
      cause = error.failure;
    } else {
      throw error;
    }
    const failure = new BackwardFailure("learn", cause.code, cause.message, {
      scope: context?.scope ?? null,
      runId: context?.runId ?? null,
      nodeId: context?.nodeId ?? null,
      nodeStartedSeq: context?.nodeStartedSeq ?? null,
      memory: key,
      cause,
    });
    const learning = prefixLearning(state.backwardState, { learnerFailure: new LearningFailure(key, "learn", cause) });
    return finishBackward(
      state,
      log,
      attemptId,
      new BackwardOutcome("failed", { feedback: root, failure, learning }),
      attributed,
    );
  }

  acceptBackward(state, log, new BackwardReadyRecord(attemptId, root, attributed, proposals));
  const backward = state.backwardState;
  const batches = proposalBatches(backward, backward.ready!);
  for (const [index, key] of backward.memoryOrder.entries()) {
    const batch = batches[index]!;
    const commitId = memoryCommitId(state.runId!, key, batch, backward.commitAttemptKey!);
    acceptBackward(state, log, new MemoryCommitRequestedRecord(key, commitId, batch, attemptId));
    // The backend receives the accepted request, re-read after its append.
    const cause = await memory!.commitLearning(state.backwardState.current!.requests.at(-1)!);
    if (cause !== null) {
      acceptBackward(state, log, new MemoryCommitFailedRecord(key, commitId, cause, attemptId));
      const failure = new BackwardFailure("commit", cause.code, cause.message, { memory: key, cause });
      return finishBackward(
        state,
        log,
        attemptId,
        new BackwardOutcome("failed", { feedback: root, failure, learning: prefixLearning(state.backwardState) }),
      );
    }
    acceptBackward(state, log, new MemoryCommitSucceededRecord(key, commitId, attemptId));
  }
  return finishBackward(
    state,
    log,
    attemptId,
    new BackwardOutcome("succeeded", {
      feedback: root,
      learning: prefixLearning(state.backwardState, { succeeded: true }),
    }),
  );
}

/**
 * Drive one new explicit whole backward attempt against an accepted terminal run.
 *
 * Every call starts a new attempt, including after a successful, failed, or committing earlier
 * attempt.
 *
 * @param state Verified retained forward and backward record authority.
 * @param log The open append authority for that state.
 * @param rootFeedback Ordinary feedback on the recorded terminal.
 * @param options.openMemory Fresh resource opener using the recorded memory table and snapshots;
 *   required only for enabled memories, and called only then. This driver closes the returned
 *   runtime on every exit.
 * @param options.models Builds the model a callback calls (AGENT backward); `computeBackward`'s.
 * @param options.onEvent Receives the transient events callbacks publish; `computeBackward`'s.
 * @returns The exact accepted Finished disposition. An unfinished earlier attempt, including one
 *   with unconfirmed commit requests, is first closed as unfinished; all callbacks and proposals
 *   are recomputed.
 * @throws TypeError (as a rejection, as are all below) when `state` isn't exactly a `RunState`,
 *   `log` isn't a `RunLog`, the feedback is malformed, or the opener returns something other than
 *   a `MemoryRuntime`.
 * @throws RangeError for invalid feedback or terminal state, a missing opener, an opened runtime
 *   that is closed or doesn't match the recorded memory table and snapshots, or a state another
 *   drive advanced meanwhile.
 * @throws Error when a backward record emits observers. Host, setup, unexpected backend, and
 *   append/reducer faults escape with their accepted prefix intact; cleanup never manufactures a
 *   completion record.
 */
export async function driveBackward(
  state: RunState,
  log: RunLog,
  rootFeedback: FeedbackValue,
  options: {
    openMemory?: (() => MemoryRuntime | Promise<MemoryRuntime>) | null;
  } & BackwardOptions = {},
): Promise<BackwardOutcome> {
  if (!isExactly(state, RunState) || !isRunLog(log)) {
    throw new TypeError("backward driving requires a RunState and RunLog");
  }
  const started = new BackwardStartedRecord(state.backwardState.attempts.length + 1, rootFeedback);
  seedBackward(state.trace!, started.feedback);
  const backward = state.backwardState;
  const unfinished = backward.started !== null && backward.finished === null;
  const first = unfinished ? new BackwardFinishedRecord(backward.started!.attemptId, backward.outcome!) : started;
  // A gate only: `seedBackward` already refused the learn-disabled and terminal cases; this
  // refuses a record that passes that check but can't be prepared (e.g. feedback that isn't
  // canonical JSON) before the opener runs. The record is prepared again with its commit, since
  // the opener's await lets another drive advance the state.
  prepareRecord(state, first);
  let memory: MemoryRuntime | null = null;
  // Boxed, so a thrown `undefined` still counts as the primary error.
  let primary: { error: unknown } | null = null;
  try {
    if (backward.memoryOrder.length > 0) {
      const { openMemory = null } = options;
      if (typeof openMemory !== "function") {
        throw new RangeError("enabled backward memories require a fresh memory runtime");
      }
      const opened: unknown = await openMemory();
      if (!(opened instanceof MemoryRuntime)) throw new TypeError("backward memory opener must return a MemoryRuntime");
      memory = opened;
      if (
        memory.closed ||
        !sameItems(memory.memories, state.memories ?? [], sameMemorySpec) ||
        !sameItems(memory.snapshots, state.memorySnapshots ?? [], sameMemorySnapshotEnvelope)
      ) {
        throw new RangeError("opened memory runtime does not match the recorded memory table and snapshots");
      }
    }
    acceptBackward(state, log, first);
    if (unfinished) acceptBackward(state, log, started);
    return await driveOpenedBackward(state, log, memory, {
      models: options.models,
      onEvent: options.onEvent,
    });
  } catch (error) {
    primary = { error };
    throw error;
  } finally {
    await memory?.closePreserving(primary);
  }
}
