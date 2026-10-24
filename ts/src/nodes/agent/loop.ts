/**
 * The AGENT loop: drive only newly returned model calls using current grants.
 *
 * Without tools or skills the loop makes a single model call (structured when the output type
 * needs it); with them it runs the tool loop until the model answers.
 *
 * A reply's tool calls run in groups (design 6.9): consecutive calls that are safe to run together
 * start together, an unsafe call runs alone, and results commit in call order whatever order the
 * calls finish in. Python runs them one at a time, which is what a reply without safe tools gets.
 *
 * Each model call is written as a `model_call` step before anything acts on its reply, and each
 * tool result as a `tool_result` step as soon as that call finishes, before the next model call
 * (`AgentSteps`). A resumed start replays them: each recorded reply is taken in place of its model
 * call, and each tool call with a recorded result gets it without running; a replayed call with no
 * result is closed as `crashed`, its outcome unknown. Live calls follow the last recorded one.
 *
 * A steerable agent reads the host's steers at the top of each iteration and stops at a host
 * interrupt between effects or before a later chunk of a streamed reply, returning its latest
 * reply's text (a cut reply's text so far); an interrupt never aborts a running tool call.
 *
 * Knows about:   diagnostics, events, llm, llm_clients connection, platform, nodes/base, skills,
 *                tools, typesys, peer AGENT modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import { ReplayDegraded } from "../../events.ts";
import { type ChatModel, parseArguments } from "../../llm/events.ts";
import { RouteModel } from "../../llm/executor.ts";
import { scrubCallCredentials } from "../../llm_clients/connection.ts";
import {
  Message,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
} from "../../llm/messages.ts";
import { realScheduler } from "../../platform.ts";
import type { Skill } from "../../skills/catalog.ts";
import { invokeLoadSkill, LOAD_SKILL, loadSkillTool } from "../../skills/tool.ts";
import { isReportedToolError } from "../../tools/errors.ts";
import { type CheckedCall, ToolSet } from "../../tools/registry.ts";
import { rejectDeepNesting, type Type, type ValueRecord, writeAsText } from "../../typesys/values.ts";
import {
  ModelTextInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
  type VisibleInteraction,
} from "../../memory/contracts.ts";
import { Output } from "../base.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  AgentSteps,
  checkStopped,
  drainSteers,
  interruptRequested,
  type RecordedResult,
  interruptSignal,
  requestSha256,
  sentCredentials,
  ToolIterationLimitError,
} from "./context.ts";
import { ReplyInterrupted } from "./interrupts.ts";
import {
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  observeSkippedToolCall,
  observeToolCall,
  type ToolCallIdentity,
} from "./live.ts";
import { reportedText, SKIPPED_TOOL_RESULT, systemPromptWithMemory, TOOL_OUTCOME_UNKNOWN, toolText } from "./memory.ts";
import { generateStructured, structuredMaterial, typeToSchema, userMessage } from "./structured.ts";
import { pushVisible, recordModelText, recordToolCall, recordToolResult, textOf } from "./text.ts";

/**
 * Run one AGENT invocation.
 *
 * Makes one model call without tools or skills, otherwise runs the tool loop until the model
 * answers. Sets `ctx.sent` to the built model's route credentials once the model is built. Each
 * reason the model gives for leaving out replay state reaches `ctx.emitEvent` once, as
 * `ReplayDegraded`. With `ctx.steps`, each model call and failed attempt is recorded as a
 * `model_call` step and each tool result as a `tool_result` step, a failed attempt's error
 * scrubbed of every credential the call may have sent; the steps earlier starts recorded are
 * replayed first (`callModel`, `replayToolCalls`), and the host's steers and interrupt are read
 * only once they end.
 *
 * The model reads the system prompt (with the recalled memory sections), the prompt's messages,
 * then each iteration's steers, replies and tool results; a final structured call after tools
 * reads them all.
 *
 * @returns The checked structured value, or the final reply's visible text; after a host
 *   interrupt, the latest reply's visible text (`""` before any).
 * @throws RangeError when a grant names no tool of the run (`ToolSet.resolve`).
 * @throws TypeCheckError when a reply's tool call arguments nest too deep.
 * @throws ToolIterationLimitError when a reply calls tools at the `maxToolIterations`-th turn.
 * @throws AgentLoopError when a reply calls a tool that isn't granted.
 * @throws CapabilityFault when a recorded call doesn't match the rebuilt request
 *   (`AgentSteps.takeRecorded`).
 * @throws Whatever building or calling the model throws (marked by `markModelFailure`), a stop's
 *   reason, a tool's own error other than a reported one, a failed argument check, a result
 *   `toolText` refuses, or `generateStructured`'s `OutputInvalidError`. The host inbox's faults
 *   pass through.
 */
export async function runAgent(ctx: AgentRunContext): Promise<Output> {
  const systemPrompt = systemPromptWithMemory(ctx.baseSystemPrompt, ctx.memoryContext.sections);
  const system = new Message({ role: "system", parts: [new TextPart(systemPrompt)] });
  const live = liveSinks(ctx);
  checkStopped(ctx);
  // Each turn's request replays every earlier one, so a reason would otherwise repeat per turn.
  const degraded = new Set<string>();
  const onReplayDegraded = (reason: string) => {
    if (degraded.has(reason)) return;
    degraded.add(reason);
    ctx.emitEvent?.(new ReplayDegraded({ nodeId: ctx.nodeId, reason }));
  };
  let model: ChatModel;
  try {
    model = await ctx.models.createModel(ctx.llmConfig ?? {}, {
      nodeId: ctx.nodeId,
      signal: ctx.signal ?? new AbortController().signal,
      onReplayDegraded,
    });
  } catch (error) {
    // Building the model resolves its credentials, which can fail as refused ones.
    markModelFailure(error);
    throw error;
  }
  ctx.sent = () => (model instanceof RouteModel ? model.route.auth.credentials() : []);
  checkStopped(ctx);
  const steps = new AgentSteps(
    ctx.steps,
    () => (ctx.scheduler ?? realScheduler).now(),
    ctx.runDir ?? null,
    (error) => scrubCallCredentials(error, ctx.llmConfig, { env: ctx.models.env, sent: sentCredentials(ctx) }),
    ctx.nodeId,
  );
  const retry = { scheduler: ctx.scheduler, random: ctx.random };
  const messages = [system, ...ctx.messages];
  ctx.evaluationTrace?.recordPrompt(ctx.baseSystemPrompt, ctx.promptText, ctx.memoryContext.sections);
  const schema = ctx.outputType !== null ? typeToSchema(ctx.outputType) : null;
  const structuredOptions = {
    retries: ctx.retries,
    live,
    signal: ctx.signal,
    evaluationTrace: ctx.evaluationTrace,
    retry,
    steps,
  };
  const toolset = ctx.toolset ?? new ToolSet();
  // A granted id may name a host-filled group; grants are checked by tool name.
  const granted = toolset.expand(ctx.tools);
  if (granted.length === 0 && ctx.skills.length === 0 && schema !== null) {
    return new Output(await generateStructured(model, messages, ctx.outputType!, structuredOptions));
  }

  // The schema goes out as the host gave it; each protocol adapter lowers it for its provider.
  const bound = toolset
    .resolve(granted)
    .map((tool) => new ToolSpec({ name: tool.name, description: tool.description, parameters: tool.parametersJsonSchema }));
  const skills = new Map(ctx.skills.map((skill) => [skill.name, skill]));
  if (skills.size > 0) bound.push(loadSkillTool());
  const interrupt = interruptSignal(ctx);
  let iterations = 0;
  let reply: Message | null = null;
  // The steers this start sent, which the request digest leaves out.
  const steered = new Set<Message>();

  while (true) {
    checkStopped(ctx);
    if (interruptRequested(interrupt, steps)) return interrupted(reply);
    // Steers join only here, never between a reply's tool calls and their results.
    const steers = await drainSteers(ctx, steps);
    // A stop during the drain wins; its steers are never sent.
    checkStopped(ctx);
    // So does an interrupt that lands while an async drain settles: no model call follows it.
    if (interruptRequested(interrupt, steps)) return interrupted(reply);
    for (const steer of steers) {
      const message = userMessage(steer);
      steered.add(message);
      messages.push(message);
    }
    const request = new Request({ messages, tools: bound });
    let called: Awaited<ReturnType<typeof callModel>>;
    try {
      called = await callModel(live, model, request, {
        purpose: "chat",
        signal: ctx.signal,
        interrupt,
        retry,
        record: { steps, digest: requestSha256(request, steered) },
      });
    } catch (error) {
      // The cut reply's text is kept; its calls, partial or not, never run.
      if (error instanceof ReplyInterrupted) {
        // A stop that lands after the cut, before this catch runs, still wins.
        checkStopped(ctx);
        return interrupted(error.reply);
      }
      throw error;
    }
    // With `record`, `n` is the reply's `model_call` step.
    const { response, n, replayed } = called as { response: Response; n: number; replayed: boolean };
    // A stop that lands during the call wins over its reply.
    checkStopped(ctx);
    rejectInvalidArguments(response.message);
    reply = response.message;
    iterations += 1;
    messages.push(reply);
    // A call whose arguments didn't parse is an invalid tool call, which the loop never runs.
    const calls = reply.parts.filter(
      (part): part is ToolCallPart => part instanceof ToolCallPart && part.arguments !== null,
    );
    if (calls.length === 0) {
      const text = textOf(reply);
      recordModelText(ctx.evaluationTrace, text);
      if (schema !== null) {
        return new Output(await generateStructured(model, messages, ctx.outputType!, structuredOptions));
      }
      return new Output(text);
    }

    if (iterations === ctx.maxToolIterations) {
      // No later model call could read these results, so running them would only add side
      // effects to a node that is already failing.
      throw new ToolIterationLimitError(
        `agent node ${reprText(ctx.nodeId)} hit the tool-iteration cap (${ctx.maxToolIterations}) without a final ` +
          `answer; the last reply's ${calls.length} tool call(s) were not run`,
      );
    }

    const turn = { ctx, toolset, granted, skills, live, n, steps, messages, calls, interrupt };
    if (await (replayed ? replayToolCalls(turn) : runToolCalls(turn))) return interrupted(reply);
  }
}

/**
 * Rebuild what one AGENT invocation showed a learner from the steps it recorded, and check each
 * recorded request against the one rebuilt from them, as `runAgent` builds it.
 *
 * Interactions follow step order: each chat reply's text when it makes no runnable tool call,
 * else its runnable calls, then the results recorded for them in call order (a `skipped` result
 * is left out); then the structured calls as `structuredMaterial` shows them. A failed attempt,
 * and a reply a host interrupt cut, show nothing. Steps of another kind are skipped. A request's
 * digest leaves steers out, as the writer does, so steers never break the check.
 *
 * @param steps The invocation's steps across its starts.
 * @param options `messages` are the system message (memory sections included) and the prompt's
 *   messages; `tools` the specs every chat request carries, `null` when they can't be rebuilt
 *   (no chat request then matches); `structured` builds the model the structured calls read
 *   their plan off and gives the output type, or `null` when it can't (no structured request
 *   then matches).
 * @returns The interactions, and whether every recorded request matched.
 */
export async function stepsMaterial(
  steps: readonly ValueRecord[],
  options: {
    messages: readonly Message[];
    tools: readonly ToolSpec[] | null;
    structured: () => Promise<{ model: ChatModel; typ: Type } | null>;
  },
): Promise<{ interactions: VisibleInteraction[]; digestOk: boolean }> {
  const messages = [...options.messages];
  const interactions: VisibleInteraction[] = [];
  let digestOk = true;
  // The latest reply's runnable calls and the results recorded for them, by call id.
  let calls: ToolCallPart[] = [];
  let results = new Map<string, ValueRecord>();
  const commit = () => {
    for (const call of calls) {
      const result = results.get(call.id);
      if (result === undefined) continue;
      const content = result["output"] as string;
      if (result["closed"] !== "skipped") {
        pushVisible(interactions, () => new ToolResultInteraction(call.name, content));
      }
      const part = new ToolResultPart({ callId: call.id, name: call.name, content, isError: result["is_error"] === true });
      messages.push(new Message({ role: "tool", parts: [part] }));
    }
    calls = [];
    results = new Map();
  };
  for (const [index, step] of steps.entries()) {
    if (step["kind"] === "tool_result") {
      results.set(step["call_id"] as string, step);
      continue;
    }
    if (step["kind"] !== "model_call") continue;
    commit();
    if (step["purpose"] !== "chat") {
      const rest = steps.slice(index).filter((later) => later["kind"] === "model_call");
      const structured = await options.structured();
      if (structured === null) return { interactions, digestOk: false };
      const material = structuredMaterial(structured.model, messages, structured.typ, rest);
      interactions.push(...material.interactions);
      return { interactions, digestOk: digestOk && material.digestOk };
    }
    const request = options.tools === null ? null : new Request({ messages, tools: options.tools });
    if (request === null || step["request_sha256"] !== requestSha256(request, new Set())) digestOk = false;
    if (Object.hasOwn(step, "error")) continue;
    const reply = Message.fromJSON(step["output"]);
    messages.push(reply);
    if (step["finish"] === "interrupted") continue;
    calls = reply.parts.filter((part): part is ToolCallPart => part instanceof ToolCallPart && part.arguments !== null);
    if (calls.length === 0) pushVisible(interactions, () => new ModelTextInteraction(reply.text()));
    for (const call of calls) pushVisible(interactions, () => new ToolCallInteraction(call.name, call.arguments));
  }
  commit();
  return { interactions, digestOk };
}

/** End an interrupted loop successfully with the latest reply's visible text (`""` before any). */
function interrupted(reply: Message | null): Output {
  return new Output(reply === null ? "" : textOf(reply));
}

/** What running one reply's tool calls reads and appends to. */
interface ToolTurn {
  readonly ctx: AgentRunContext;
  readonly toolset: ToolSet;
  /** The granted tool names, groups expanded. */
  readonly granted: string[];
  readonly skills: ReadonlyMap<string, Skill>;
  readonly live: LiveSinks | null;
  /** The `n` of the reply's `model_call` step, which each call's idempotency key and brackets hold. */
  readonly n: number;
  /** Where each call's `tool_result` step is written. */
  readonly steps: AgentSteps;
  /** The model's messages; each committed result is appended. */
  readonly messages: Message[];
  /** The reply's runnable tool calls, in call order. */
  readonly calls: readonly ToolCallPart[];
  /** The host's interrupt signal, or `undefined` for an agent that isn't steerable. */
  readonly interrupt: AbortSignal | undefined;
}

/**
 * How one tool call ended: the model reads `content` (a reported error's text when `reported`), or
 * it threw `error`, or its result had no text.
 */
type ToolOutcome = { readonly content: string; readonly reported: boolean } | { readonly error: unknown };

/** One started call of a group and how it ended. */
interface SettledCall {
  readonly call: ToolCallPart;
  readonly outcome: ToolOutcome;
}

/** One call whose check passed, ready to start. */
interface PlannedCall {
  readonly call: ToolCallPart;
  readonly index: number;
  /** Run the call under its group's signal. */
  readonly invoke: (signal: AbortSignal) => Promise<unknown>;
}

/** The outcome of checking one call before it may start. */
type CallCheck =
  | { readonly planned: PlannedCall; readonly safe: boolean }
  | { readonly error: unknown; readonly granted: boolean };

/**
 * Run one reply's tool calls in groups, appending a tool message per call in call order.
 *
 * Calls are checked in call order. Consecutive calls that are safe to run together (a
 * `load_skill` call, or a tool whose `isConcurrencySafe` says so) form a group; an unsafe call
 * runs alone, after the group before it. A failed check ends the calls before it as their own
 * group, which runs and commits; then the node fails at it, as Python's one-at-a-time loop does.
 *
 * A host interrupt seen before a call's check, before a group starts, or at a failed check skips
 * every call not started yet (`skipFrom`); a running group always finishes and commits.
 *
 * @returns Whether a host interrupt skipped the remaining calls.
 * @throws A stop's reason, the first failing call's error, a result `toolText` refuses, the
 *   ungranted `AgentLoopError`, or a failed argument check.
 */
async function runToolCalls(turn: ToolTurn): Promise<boolean> {
  let group: PlannedCall[] = [];
  for (const [index, call] of turn.calls.entries()) {
    // Not `checkStopped(ctx)`: without a context signal, the tools run under the tool set's bound
    // run signal, so a stop on it must stop the calls too.
    checkNodeStopped(turn);
    // After the stop, as in Python; nothing of `group` has started yet, so it is skipped whole.
    if (turn.interrupt?.aborted) return skipFrom(turn, group[0]?.index ?? index);
    const check = await checkCall(turn, call, index);
    if ("error" in check) {
      if (await runGroup(turn, group)) return true;
      return await failCall(turn, call, index, check);
    }
    if (check.safe) {
      group.push(check.planned);
      continue;
    }
    if (await runGroup(turn, group)) return true;
    group = [];
    if (await runGroup(turn, [check.planned])) return true;
  }
  return await runGroup(turn, group);
}

/** One replayed call and the result the model reads for it. */
interface ReplayedCall {
  readonly call: ToolCallPart;
  readonly result: RecordedResult;
}

/**
 * Give a replayed reply's tool calls their results without running any, appending a tool message
 * per call in call order. A call keeps the result an earlier start recorded. A call with none (its
 * run was cut off before its result was written) is checked as live: a failed check fails the
 * node as live; otherwise it is closed, its `tool_result` step written as `crashed`, an error
 * reading `TOOL_OUTCOME_UNKNOWN`. Unparsed calls are not in `turn.calls`, so they get nothing, as
 * live.
 *
 * Every call is checked, so the calls form the groups they formed live and the evidence is
 * recorded in the same order; a call with a recorded result keeps it whatever its check says now.
 *
 * @returns Whether a host interrupt skipped calls of this reply when it was live (a recorded
 *   `skipped` result): the loop ends as it did then.
 * @throws A stop's reason, or a failed check's error (`failCall`).
 */
async function replayToolCalls(turn: ToolTurn): Promise<boolean> {
  const { ctx, steps } = turn;
  let group: ReplayedCall[] = [];
  const commit = () => {
    for (const { call } of group) recordToolCall(ctx.evaluationTrace, call.name, call.arguments);
    for (const { call, result } of group) {
      const { id: callId, name } = call;
      recordToolResult(ctx.evaluationTrace, name, result.output);
      const part = new ToolResultPart({ callId, name, content: result.output, isError: result.isError });
      turn.messages.push(new Message({ role: "tool", parts: [part] }));
    }
    group = [];
  };
  for (const [index, call] of turn.calls.entries()) {
    checkNodeStopped(turn);
    let result = steps.recordedResult(turn.n, call.id);
    // Live, the interrupt skipped from this call's group on; the groups before it committed.
    if (result?.closed === "skipped") {
      commit();
      return true;
    }
    const check = await checkCall(turn, call, index);
    if (result === null) {
      if ("error" in check) {
        commit();
        return await failCall(turn, call, index, check);
      }
      result = { output: TOOL_OUTCOME_UNKNOWN, isError: true, closed: "crashed" };
      steps.toolResult({ callId: call.id, startedAt: null, output: result.output, isError: true, closed: "crashed" });
    }
    if ("planned" in check && check.safe) {
      group.push({ call, result });
      continue;
    }
    commit();
    group = [{ call, result }];
    commit();
  }
  commit();
  return false;
}

/**
 * Report the calls from call `start` on as skipped by a host interrupt, in call order: each gets a
 * `skipped` `tool_result` step (`SKIPPED_TOOL_RESULT`, an error) and its skipped events. None of
 * them runs or leaves evidence.
 *
 * @returns `true`, for the caller to return.
 */
function skipFrom(turn: ToolTurn, start: number): true {
  for (const [index, call] of turn.calls.entries()) {
    if (index < start) continue;
    turn.steps.toolResult({
      callId: call.id,
      startedAt: null,
      output: SKIPPED_TOOL_RESULT,
      isError: true,
      closed: "skipped",
    });
    observeSkippedToolCall(turn.live, identityOf(turn, call, index));
  }
  return true;
}

/**
 * Check one call before it may start: its grant, then its arguments and whether it is safe to run
 * beside others. A check the stop cut short throws the stop's reason. A `ReportedToolError` from
 * the argument check plans an unsafe call that rejects with it, so it runs alone and the model
 * reads it as the call's error result.
 */
async function checkCall(turn: ToolTurn, call: ToolCallPart, index: number): Promise<CallCheck> {
  const { ctx, toolset, skills } = turn;
  const args = call.arguments!;
  if (skills.size > 0 && call.name === LOAD_SKILL) {
    // `load_skill` only reads the granted skills.
    return { planned: { call, index, invoke: async () => invokeLoadSkill(skills, args) }, safe: true };
  }
  if (!turn.granted.includes(call.name)) {
    const error = new AgentLoopError(
      `agent node ${reprText(ctx.nodeId)} requested tool ${reprText(call.name)}, but it is not granted; ` +
        `granted tools: ${writeAsText(turn.granted)}`,
    );
    return { error, granted: false };
  }
  let checked: CheckedCall;
  try {
    checked = await toolset.validate(call.name, args, { signal: ctx.signal });
  } catch (error) {
    // A check the stop cut short is the stop, not a failed check.
    checkNodeStopped(turn);
    // Python checks the arguments inside the call, where a reported error is the call's result.
    if (isReportedToolError(error)) {
      return { planned: { call, index, invoke: () => Promise.reject(error) }, safe: false };
    }
    return { error, granted: true };
  }
  // The key names the reply's `model_call` step and the call's place in it. Steps outlive a start
  // and a resumed start numbers its steps after the recorded ones, so a key never names two
  // calls that ran: a reply a crash cut before its `model_call` was written ran no tool. Outside a
  // run (no run id) the tool set gives each call a fresh id.
  const runId = toolset.runId;
  const callId = runId ? `${runId}/${ctx.nodeId}/${turn.n}/${index}` : undefined;
  const invoke = (signal: AbortSignal) => toolset.invoke(checked, { node: ctx.nodeId, callId, signal });
  return { planned: { call, index, invoke }, safe: toolset.isConcurrencySafe(checked) };
}

/**
 * Fail the node at a call whose check failed, once the calls before it committed.
 *
 * As for a call about to start, a stop wins first, then a host interrupt skips it and every later
 * call. Otherwise the call's evidence is recorded; a granted call whose arguments failed gets the
 * brackets of a call that threw, since Python checks them inside the call, and an ungranted one
 * gets none.
 *
 * @returns `true` when a host interrupt skipped the call.
 * @throws The failed check's error, or what publishing its brackets throws.
 */
async function failCall(
  turn: ToolTurn,
  call: ToolCallPart,
  index: number,
  failed: { readonly error: unknown; readonly granted: boolean },
): Promise<true> {
  const { ctx, live } = turn;
  checkNodeStopped(turn);
  if (turn.interrupt?.aborted) return skipFrom(turn, index);
  recordToolCall(ctx.evaluationTrace, call.name, call.arguments);
  if (failed.granted) {
    // Rejects with the failed check's error once its brackets are published.
    await observeToolCall(live, () => Promise.reject(failed.error), identityOf(turn, call, index), {
      signal: nodeSignal(turn),
      text: (value) => toolText(call.name, value),
    });
  }
  throw failed.error;
}

/**
 * Run one group's calls together and commit their results in call order.
 *
 * The calls share one child signal of the node's. Each starts in call order; the first to fail
 * (in time; a `ReportedToolError` is the call's error result, not a failure; a result `toolText`
 * refuses is a failure) aborts the group's signal, and no further call starts. As each call
 * finishes with a result, its text is computed and written as a `tool_result` step at once, in
 * finish order, unless the node has stopped by then. Every started call is awaited; then a stop
 * wins, else the first failure is thrown with nothing of the group committed. A stop, then a host
 * interrupt, is checked before the group starts; the interrupt skips from the group's first call.
 *
 * @returns Whether a host interrupt skipped the group and every later call.
 */
async function runGroup(turn: ToolTurn, group: readonly PlannedCall[]): Promise<boolean> {
  if (group.length === 0) return false;
  const { ctx, live } = turn;
  checkNodeStopped(turn);
  if (turn.interrupt?.aborted) return skipFrom(turn, group[0]!.index);
  const signal = nodeSignal(turn);
  const controller = new AbortController();
  const relay = () => controller.abort(signal!.reason);
  // Removed once the group settles, so a long loop leaves no listeners on the node's signal.
  signal?.addEventListener("abort", relay, { once: true });
  // Set in a callback, so declared wide: a `null` initializer would narrow it for good.
  let failure = null as { error: unknown } | null;
  const fail = (error: unknown) => {
    // Only the first failure in time is kept and aborts; aborting again changes nothing.
    failure ??= { error };
    controller.abort(error);
  };
  /**
   * Take the text the model reads for one call's result and write its step, as the call finishes;
   * a result with no text, or a write that throws, fails the group.
   */
  const settle = (call: ToolCallPart, startedAt: number, value: unknown, reported: boolean): ToolOutcome => {
    try {
      const content = reported ? reportedText(call.name, value) : toolText(call.name, value);
      // Once the node stopped, the stop may be what ended the call, so its result isn't written.
      if (!signal?.aborted) {
        turn.steps.toolResult({ callId: call.id, startedAt, output: content, isError: reported }, { settled: true });
      }
      return { content, reported };
    } catch (error) {
      fail(error);
      return { error };
    }
  };
  try {
    for (const { call } of group) recordToolCall(ctx.evaluationTrace, call.name, call.arguments);
    const settling: Promise<SettledCall>[] = [];
    for (const planned of group) {
      if (controller.signal.aborted) break;
      const { call, index } = planned;
      let started = false;
      let startedAt = 0;
      const running = observeToolCall(
        live,
        () => {
          started = true;
          startedAt = turn.steps.now();
          return planned.invoke(controller.signal);
        },
        identityOf(turn, call, index),
        { signal, text: (value) => toolText(call.name, value) },
      );
      const outcome = running.then(
        (value): ToolOutcome => settle(call, startedAt, value, false),
        (error): ToolOutcome => {
          if (isReportedToolError(error)) return settle(call, startedAt, error, true);
          fail(error);
          return { error };
        },
      );
      settling.push(outcome.then((ended) => ({ call, outcome: ended })));
      // `observeToolCall` calls `invoke` synchronously unless publishing the started event threw;
      // its rejection is seen only later, so the next call must not start in the meantime.
      if (!started) break;
    }
    const settled = await Promise.all(settling);
    // A stop pending when the calls settle wins over the first failure and over their results.
    checkNodeStopped(turn);
    if (failure !== null) throw failure.error;
    for (const { call, outcome } of settled) {
      // No outcome is an error: a failure was thrown above.
      const { content, reported } = outcome as Extract<ToolOutcome, { content: string }>;
      const { id: callId, name } = call;
      recordToolResult(ctx.evaluationTrace, name, content);
      const part = new ToolResultPart({ callId, name, content, isError: reported });
      turn.messages.push(new Message({ role: "tool", parts: [part] }));
    }
  } finally {
    signal?.removeEventListener("abort", relay);
  }
  return false;
}

/** The node's stop signal: the context's, else the tool set's run signal. */
function nodeSignal(turn: ToolTurn): AbortSignal | undefined {
  return turn.ctx.signal ?? turn.toolset.signal;
}

/** Throw the node's stop reason if it stopped, whichever signal carries the stop. */
function checkNodeStopped(turn: ToolTurn): void {
  nodeSignal(turn)?.throwIfAborted();
}

/** One call's identity on its brackets. */
function identityOf(turn: ToolTurn, call: ToolCallPart, index: number): ToolCallIdentity {
  const { id: toolCallId, name } = call;
  return { n: turn.n, callIndex: index, toolCallId, name, arguments: call.arguments! };
}

/**
 * Throw `TypeCheckError` if a tool call's arguments nest too deep.
 *
 * An unparsed call's raw arguments text is parsed at any depth and checked the same way, since the
 * assembler leaves a call unparsed when its arguments nest too deep; text that is not JSON is left
 * to the invalid call's own handling.
 */
function rejectInvalidArguments(reply: Message): void {
  for (const part of reply.parts) {
    if (!(part instanceof ToolCallPart)) continue;
    // `ToolCallPart` already refuses parsed arguments past the limit; checking them again matters
    // only if the depth limit changed after the part was built.
    const args = part.arguments ?? parseArguments(part.rawArguments!, { anyDepth: true })[0];
    if (args !== null) rejectDeepNesting(args, `tool call ${reprText(part.name)} arguments`);
  }
}
