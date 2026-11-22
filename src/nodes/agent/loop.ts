/**
 * The AGENT loop: drive only newly returned model calls using current grants.
 *
 * A structured agent without tools or skills makes a single structured call; any other agent runs
 * the chat loop until the model answers (without tools or skills, every reply is an answer).
 *
 * A reply's tool calls run in groups: consecutive calls that are safe to run together
 * start together, an unsafe call runs alone, and results commit in call order whatever order the
 * calls finish in. Every call gets a result. The model's mistakes (unparsed arguments, a name it
 * wasn't granted, arguments that fail their check) never reach a tool: each gets an error result
 * the model reads, and the loop goes on. At `maxToolIterations`, a reply that still calls tools
 * runs none of them: each call gets the cap's error result (`toolLimitText`), and the model answers
 * once more, reading no steer. That reply is the last: calls it still makes get the same result,
 * and its text is the answer. A host interrupt pending when such a reply arrives skips its calls
 * instead, as it does any reply's. A reply the provider cut off at the token limit with no call
 * whose arguments parsed is no answer: the node fails with `OutputTruncatedError`.
 *
 * Each model call is written as a `model_call` step before anything acts on its reply, and each
 * tool result as a `tool_result` step as soon as that call finishes, before the next model call
 * (`AgentSteps`). A resumed start replays them: each recorded reply is taken in place of its model
 * call, and each tool call with a recorded result gets it without running. A replayed call with no
 * result is sent again when its tool is resumable or the reply alone refuses it (the same error
 * result again), pended again while the run still holds it open, or else closed with the word of
 * the record that closed it (`crashed` when none did: its outcome is unknown). A capped reply's
 * calls get the cap's result, or skips, as its first recorded result shows. Live calls follow the
 * last recorded one.
 *
 * A tool call whose handler returns `pending()` is written as a `pending` step, which registers it
 * with the run under its call id `<run>/<node>/<n>/<index>`; the reply's other calls still run, and
 * then the loop returns `Pending` naming every call of the reply that pended. A host's `complete`
 * writes the call's `tool_result`; the woken agent replays and calls the model with every result.
 * A call the host's check holds (`beforeToolCall` gives `pending()`) is registered the same way, by
 * a hold step written before its tool runs, and its tool never runs: `complete` gives the model
 * its result. The calls before it and in its group run; the reply's later calls wait until every
 * hold is answered, then run as live, checked and asked about again. An interrupt closes a hold,
 * whose closer reads as skipped, and skips the waiting calls. A hold the host releases runs once,
 * as the model asked it, checked again but not asked about again, at the first evaluation that
 * began after the release; a crash that may have cut it after it started sends it again under the
 * same call id when its tool is resumable, else closes it `crashed`. An interrupt before it starts
 * skips it.
 *
 * An agent with a steer inbox reads it at the top of each iteration (a recorded `steer` step) and
 * stops at a host interrupt of that inbox between effects, while a model call waits for its reply
 * to start (aborting the request), or before a later chunk of a streamed reply: the cut reply keeps
 * its text so far, calls not started are skipped, the reply's pending calls get `interrupted`
 * closers, and the loop writes `interrupted {by}` and returns `Interrupted`. An interrupt never
 * aborts a running tool call's signal: it aborts the call's own `ToolContext.interrupted`, and the
 * loop honors it once the call returns, its result recorded as usual. In the chat loop, a steer
 * queued when the model gives its final answer gets one more model call, within
 * `maxToolIterations`; a structured agent without tools or skills reads its steers once, before
 * its single call.
 *
 * Knows about: diagnostics, events, llm, llm_clients connection, platform, nodes/base,
 *              nodes/contracts, skills, tools, typesys, peer AGENT modules.
 */

import { reprText } from "../../diagnostics.ts";
import { ReplayDegraded } from "../../events.ts";
import { ArgumentsTooDeepError, type ChatModel, parseDeepArguments } from "../../llm/events.ts";
import { RouteModel } from "../../llm/executor.ts";
import { scrubCallCredentials } from "../../llm_clients/connection.ts";
import {
  copyJson,
  Message,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
} from "../../llm/messages.ts";
import { realScheduler, warn } from "../../platform.ts";
import { type Skill, SkillError } from "../../skills/catalog.ts";
import { invokeLoadSkill, LOAD_SKILL, loadSkillTool } from "../../skills/tool.ts";
import { ASK_USER, isPending } from "../../tools/context.ts";
import { isReportedToolError, isToolCheckError, ReportedToolError } from "../../tools/errors.ts";
import {
  type BeforeToolCall,
  checkGateText,
  type FlowCallScope,
  GATE_FAILED_TEXT,
  type GatedCall,
  type ToolSource,
} from "../../tools/gate.ts";
import {
  asToolGrant,
  expandGrants,
  GrantError,
  type GrantedTool,
  requestToolSpecs,
} from "../../tools/grants.ts";
import { type CheckedCall, ToolSet } from "../../tools/registry.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import { freezeParsed } from "../../typesys/json.ts";
import { formatRecordTime, parseDuration } from "../../typesys/temporal.ts";
import { isRecord, rejectDeepNesting, type Type, TypeCheckError, type ValueRecord } from "../../typesys/values.ts";
import { CapabilityFault, Output, Pending, type RebuiltRequest, type RebuiltRequests, type SteerInbox } from "../base.ts";
import type { CallView } from "../contracts.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  AgentSteps,
  checkStopped,
  interruptRequested,
  recordedResponse,
  RequestDigest,
  sentCredentials,
} from "./context.ts";
import { flowFailureText, isFlowCall } from "./flow_tools.ts";
import { Interrupted, ReplyInterrupted } from "./interrupts.ts";
import {
  CAPPED_TOOL_ERROR,
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  observeFlowCallFinished,
  observeSkippedToolCall,
  observeToolCall,
  REFUSED_TOOL_ERROR,
  type ToolCallIdentity,
} from "./live.ts";
import {
  type CloserReason,
  closerResult,
  reportedText,
  SKIPPED_TOOL_RESULT,
  TOOL_INTERRUPTED_OUTCOME,
  toolLimitText,
  toolText,
  unavailableText,
  unparsedText,
} from "./results.ts";
import { foldSteps, type SteerRender, steerMessages, steerValues } from "./messages.ts";
import type { StructuredTarget } from "../../llm/structured.ts";
import { generateStructured, OutputTruncatedError, rebuildStructuredRequests, typeToSchema } from "./structured.ts";

/**
 * Run one AGENT invocation.
 *
 * A structured agent without tools or skills reads its queued steers once and makes one structured
 * call. Any other agent runs the chat loop until the model answers; while the model answers with
 * steers queued it reads them and calls again (each reply, late steers included, counts toward
 * `maxToolIterations`; at the cap the answer stands and the steers stay queued). A reply at the
 * cap that still calls tools runs none: its calls get the cap's error result, and one more model
 * call, with no steer read, gives the answer (its text even when it calls tools again). Sets
 * `ctx.sent` to the built model's route credentials once the model is built. A reply cut off at
 * the token limit (`finishReason` `length`) with no call whose arguments parsed fails the node,
 * once any late steer it gets was read; a cut reply with a parsed call runs it. Each reason the
 * model gives for leaving out replay state reaches `ctx.emitEvent` once, as `ReplayDegraded`. With
 * `ctx.steps`, each model call and failed attempt is recorded as a `model_call` step and each
 * tool result as a `tool_result` step, a failed attempt's error scrubbed of every credential the
 * call may have sent; the steps earlier starts recorded are replayed first (`callModel`,
 * `replayToolCalls`), and the host's interrupt is read only once they end, a late steer recorded
 * after the last recorded call included (a recorded honor step ends the loop there); recorded
 * steers are fed back as replay reaches them.
 *
 * The model reads the system prompt, the prompt's messages, then each iteration's steers, replies
 * and tool results; a final structured call after tools reads them all.
 *
 * @returns The checked structured value, or the final reply's visible text; `Interrupted` once it
 *   honored a host interrupt (`honor`); or `Pending` naming the calls of the latest reply whose
 *   handler returned `pending()`, once the reply's other calls ran.
 * @throws RangeError when a grant names no tool of the run (`ToolSet.resolve`).
 * @throws TypeCheckError when a reply repeats a tool call id or its tool call arguments nest too deep.
 * @throws AgentLoopError when a tool returns `pending()` outside a durable run, or when the host's
 *   `beforeToolCall` would be asked about a call outside a run.
 * @throws OutputTruncatedError when a reply was cut off at the token limit with no runnable call.
 * @throws CapabilityFault when a recorded call doesn't match the rebuilt request
 *   (`AgentSteps.takeRecorded`), or the loop was cut with no pending interrupt.
 * @throws Whatever building or calling the model throws (marked by `markModelFailure`), a stop's
 *   reason, a tool's own error other than a reported one, what a parameters validator throws
 *   other than its issues, a result `toolText` refuses, or `generateStructured`'s
 *   `OutputInvalidError`. The host inbox's faults pass through.
 */
export async function runAgent(ctx: AgentRunContext): Promise<Output | Pending | Interrupted> {
  const system = new Message({ role: "system", parts: [new TextPart(ctx.systemPrompt)] });
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
  const schema = ctx.outputType !== null ? typeToSchema(ctx.outputType) : null;
  const interrupt = ctx.steerInbox?.interrupt;
  const structuredOptions = {
    retries: ctx.retries,
    live,
    signal: ctx.signal,
    interrupt,
    retry,
    steps,
  };
  /** The structured call's checked value, or the honor of an interrupt that cut it. */
  const structured = async (): Promise<Output | Interrupted> => {
    try {
      return new Output(await generateStructured(model, messages, ctx.outputType!, structuredOptions));
    } catch (error) {
      if (error instanceof ReplyInterrupted) return honor(ctx, steps);
      throw error;
    }
  };
  const onDrop = (event: number, part: number) =>
    warn(`node ${reprText(ctx.nodeId)}: steer event ${event} part ${part} is media, which no route sends yet; dropped`);
  // The events are already consumed, so a render that fails for one sends its default text instead.
  const onFallback = (event: number, error: unknown) =>
    warn(
      `node ${reprText(ctx.nodeId)}: steer event ${event} didn't render (${error instanceof Error ? error.message : String(error)}); ` +
        "sent as is",
    );
  const steerOptions = { render: ctx.renderSteer ?? null, onDrop, onFallback };
  const toolset = ctx.toolset ?? new ToolSet();
  // A grant may name a host-filled group, whose tools the model sees as `<group>_<tool>`.
  let granted: GrantedTool[];
  try {
    granted = expandGrants(ctx.tools.map(asToolGrant), toolset.scopes());
  } catch (error) {
    if (!(error instanceof GrantError)) throw error;
    throw new YianRangeError(`agent node ${reprText(ctx.nodeId)} ${error.message}`, { cause: error });
  }
  if (granted.length === 0 && ctx.skills.length === 0 && schema !== null) {
    // One structured request: steers join it as they would a loop's first call, and an interrupt
    // before it is honored, leaving the steers queued.
    if (interruptRequested(interrupt, steps)) return honor(ctx, steps);
    for (const values of steerReads(steps, ctx.steerInbox)) messages.push(...steerMessages(values, steerOptions));
    return await structured();
  }

  const skills = new Map(ctx.skills.map((skill) => [skill.name, skill]));
  // The schema goes out as the host gave it; each protocol adapter lowers it for its provider.
  const specOf = (each: GrantedTool) => {
    const tool = toolset.resolve([{ group: each.group, name: each.tool }])[0]!;
    const { name, description } = tool;
    return new ToolSpec({ name, description, parameters: tool.parametersJsonSchema });
  };
  const bound = requestToolSpecs(granted, specOf, skills.size > 0 ? loadSkillTool() : null)!;
  // A tool built by an older copy of yian has no `resumable` field, and isn't resumable.
  const resumable = new Set(
    granted
      .filter((each) => toolset.resolve([{ group: each.group, name: each.tool }])[0]!.resumable === true)
      .map((each) => each.visible),
  );
  let iterations = 0;
  // Set once the reply at the cap had its calls answered: the next reply is the last.
  let capped = false;
  let reply: Message | null = null;
  const digests = new RequestDigest();
  /** The answer: the structured call's value, or the last reply's visible text. */
  const answer = async (): Promise<Output | Interrupted> =>
    schema !== null ? await structured() : new Output(reply!.text());

  while (true) {
    checkStopped(ctx);
    // Before the steer read, so an interrupt leaves queued steers for the next reader.
    if (interruptRequested(interrupt, steps)) return honor(ctx, steps);
    // Steers join only here, never between a reply's tool calls and their results. The read is
    // synchronous, so neither a stop nor an interrupt lands between it and the model call. The
    // call after the cap reads none: queued steers stay queued, as for an answer at the cap.
    if (!capped) {
      for (const values of steerReads(steps, ctx.steerInbox)) messages.push(...steerMessages(values, steerOptions));
    }
    const request = new Request({ messages, tools: bound });
    let called: { response: Response; n: number; replayed: boolean };
    try {
      called = await callModel(live, model, request, {
        purpose: "chat",
        signal: ctx.signal,
        interrupt,
        retry,
        record: { steps, digest: digests.digest(request) },
      });
    } catch (error) {
      // The cut reply's text is recorded; its calls, partial or not, never run. A stop that lands
      // after the cut, before this catch runs, still wins (`honor`).
      if (error instanceof ReplyInterrupted) return honor(ctx, steps);
      throw error;
    }
    // With `record`, `n` is the reply's `model_call` step.
    const { response, n, replayed } = called;
    // A stop that lands during the call wins over its reply.
    checkStopped(ctx);
    rejectInvalidReply(response.message);
    reply = response.message;
    iterations += 1;
    messages.push(reply);
    // Every call gets a result, one whose arguments didn't parse included: a provider refuses a
    // conversation that leaves a call unanswered.
    const calls = reply.parts.filter((part): part is ToolCallPart => part instanceof ToolCallPart);
    if (calls.length === 0) {
      // A late steer an earlier start recorded here was its choice to continue, made before any
      // interrupt or cap check of this start: replay follows it, and the loop top honors an
      // interrupt that cut the next call or its retry wait. A resume runs with the run's recorded
      // settings and flow files, so a recorded steer never sits at the cap.
      if (steps.hasRecordedSteer()) continue;
      // A final answer that arrived before the loop saw an interrupt stands: nothing is left to
      // cut. An interrupt seen now beats a late steer: the answer stands as if it never came (no
      // steer, no honor step), and queued steers stay for the next reader. At the cap the answer
      // stands too: the cap is checked before anything is read, so nothing is consumed that won't
      // be sent. A structured agent's trailing call is cut like any model call, and never starts
      // once an interrupt is seen (an earlier start's honor included).
      const late = !capped && !interruptRequested(interrupt, steps) && iterations !== ctx.maxToolIterations;
      if (late && queuedSteer(steps, ctx.steerInbox)) continue;
      if (response.finishReason === "length") throw new OutputTruncatedError(model);
      return await answer();
    }
    // The cut arguments would only earn error results; a parsed call still runs.
    if (response.finishReason === "length" && calls.every((call) => call.arguments === null)) {
      throw new OutputTruncatedError(model);
    }

    const pended = new Map<number, string>();
    const turn: ToolTurn = {
      ctx, toolset, granted, resumable, skills, live, n, steps, messages, calls, interrupt, pended, waitingFrom: null,
    };
    if (capped || iterations === ctx.maxToolIterations) {
      // The model reads why its calls didn't run, then gives its answer.
      if (answerAtCap(turn, ctx.maxToolIterations)) return honor(ctx, steps);
      if (capped) return await answer();
      capped = true;
      continue;
    }
    const skipped = await (replayed ? replayToolCalls(turn) : runToolCalls(turn));
    // The interrupt closed the reply's pending calls, and the calls waiting for a hold never run;
    // the agent doesn't wait on them.
    if (skipped || (pended.size > 0 && interruptRequested(interrupt, steps))) {
      closePended(turn);
      if (turn.waitingFrom !== null) skipFrom(turn, turn.waitingFrom);
      return honor(ctx, steps);
    }
    if (pended.size > 0) return new Pending([...pended].sort(([a], [b]) => a - b).map(([, callId]) => callId));
  }
}

/**
 * Whether a live steer joins after a final answer: once replay is over and no recorded attempt
 * continues the call, the inbox has events queued now. The loop top reads them; nothing awaits
 * between this check and that read.
 */
function queuedSteer(steps: AgentSteps, inbox: SteerInbox | undefined): boolean {
  return inbox !== undefined && !steps.replaying() && !steps.continuesCall() && inbox.queued();
}

/**
 * Honor a host interrupt: once a stop is ruled out, write `interrupted {by}` (unless an earlier
 * start wrote it, which replay reached) and return `Interrupted`. `by` is the seq of the
 * `interrupted` record the node holds pending.
 *
 * @throws The stop's reason when the node stopped: a stop wins over an interrupt.
 * @throws CapabilityFault carrying a RangeError when no interrupt is pending: the loop was cut by
 *   a signal no `interrupted` record backs.
 */
function honor(ctx: AgentRunContext, steps: AgentSteps): Interrupted {
  checkStopped(ctx);
  const recorded = steps.honored();
  const by = recorded ?? ctx.steerInbox?.interruptedBy() ?? null;
  if (by === null) {
    throw new CapabilityFault(new YianRangeError(`node ${reprText(ctx.nodeId)} was cut with no pending interrupt`));
  }
  if (recorded === null) steps.interrupted(by);
  return new Interrupted(by);
}

/**
 * Write the agent's record of each call of the turn that pended and was settled since without one,
 * in call order (`recordSettled`): the interrupt closed it, it was registered closed, or its flow's
 * scope answered or failed. A call the host answered meanwhile keeps its answer, and one it
 * released is skipped: it never started.
 */
function closePended(turn: ToolTurn): void {
  const views = turn.ctx.steps?.currentCalls() ?? [];
  for (const [index, callId] of [...turn.pended].sort(([a], [b]) => a - b)) {
    const view = views.find((each) => each.callId === callId);
    const call = turn.calls[index]!;
    if (view === undefined || turn.steps.hasResult(turn.n, call.id)) continue;
    if (view.state === "released") skipCall(turn, call, index);
    else recordSettled(turn, call, index, view);
  }
}

/**
 * Write the agent's record of registered call `index`, which the run settled (`view`) and which
 * has no result: the answer its flow's scope gave, or its closer (`closeRecorded`). A flow call's
 * end is announced.
 *
 * @returns The text the model reads and whether it is an error; `null` when the record isn't the
 *   agent's to write here: the call is open, a host answered it, or it is released, for the agent
 *   to run.
 */
function recordSettled(
  turn: ToolTurn,
  call: ToolCallPart,
  index: number,
  view: CallView,
): { readonly output: string; readonly isError: boolean } | null {
  if (view.state === "closed") return { output: closeRecorded(turn, call, index, view), isError: true };
  // A host's answer is its own record; the agent records the answer the run shaped for a flow call.
  if (view.state !== "answered" || view.scope === null || view.result === null) return null;
  const output = view.result["output"] as string;
  const isError = view.result["is_error"] === true;
  turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError });
  observeFlowCallFinished(turn.live, identityOf(turn, call, index), { result: output });
  return { output, isError };
}

/**
 * Write the closer of call `index`, which a record closed (`view`), with that record's word; a
 * flow call's end is announced. A failed flow call's closer gives its failure. A hold never ran,
 * so the model reads it as skipped.
 *
 * @returns The text the model reads.
 */
function closeRecorded(turn: ToolTurn, call: ToolCallPart, index: number, view: CallView): string {
  if (view.closedReason === "failed" && view.failure !== null) {
    const output = flowFailureText(call.name, view.scope, view.failure);
    turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed: "failed" });
    if (view.scope !== null) observeFlowCallFinished(turn.live, identityOf(turn, call, index), { error: "failed" });
    return output;
  }
  const closer = closerResult(closerReason(view.closedReason));
  const output = view.held ? SKIPPED_TOOL_RESULT : closer.output;
  turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed: closer.closed });
  if (view.scope !== null) observeFlowCallFinished(turn.live, identityOf(turn, call, index), { error: closer.closed });
  return output;
}

/** The closer word for a call closed for `reason`: a deadline's or an interrupt's, else `crashed` (its outcome is unknown). */
function closerReason(reason: string | null): CloserReason {
  return reason === "timed_out" || reason === "interrupted" ? reason : "crashed";
}

/**
 * Return the steer reads that join the next model call, oldest first: each recorded before it,
 * then, once replay is over and no recorded attempt continues the call, a live read of everything
 * queued (a send made while the run was down joins the first live call).
 */
function steerReads(steps: AgentSteps, inbox: SteerInbox | undefined): (readonly unknown[])[] {
  const reads: (readonly unknown[])[] = steps.takeRecordedSteers().map((recorded) => recorded.values);
  if (inbox !== undefined && !steps.replaying() && !steps.continuesCall()) {
    const read = inbox.read();
    if (read !== null) reads.push(read.values);
  }
  return reads;
}

/**
 * Rebuild each request one AGENT invocation sent from the steps it recorded, as `runAgent` builds
 * it, checked against the recorded digest.
 *
 * Replies and tool results join the conversation in step order, as the live loop added them; a
 * `skipped` result is sent too. Steer steps add their messages where they were read; a steer step
 * without its values fails the check. Structured calls are rebuilt by
 * `rebuildStructuredRequests`. The steps are grouped as `foldSteps` groups them; steps of another
 * kind are skipped.
 *
 * @param steps The invocation's steps across its starts.
 * @param options `messages` are the system message and the prompt's messages; `tools` the specs
 *   every chat request carries, `null` when they can't be rebuilt (no chat request then matches);
 *   `structured` gives what the structured calls read their plan off (`structuredTarget`) and the
 *   output type, or `null` when it can't (no structured request then matches); `renderSteer` is the
 *   agent's steer render, which steer steps render through as the live loop did.
 * @returns Each `model_call` step's rebuilt request, by its index in `steps` (a step after a reply
 *   that failed the node, or after a steer step without its values, has none; with no `tools` or
 *   `structured` model, a step's request is `null`); and whether every recorded request matched.
 */
export async function rebuildStepRequests(
  steps: readonly ValueRecord[],
  options: {
    messages: readonly Message[];
    tools: readonly ToolSpec[] | null;
    structured: () => Promise<{ target: StructuredTarget; typ: Type } | null>;
    renderSteer?: SteerRender | null;
  },
): Promise<RebuiltRequests> {
  const messages = [...options.messages];
  const digests = new RequestDigest();
  const requests = new Map<number, RebuiltRequest>();
  let digestOk = true;
  const { exchanges, open } = foldSteps(steps);
  const turns = [...exchanges, open];
  for (const [turn, exchange] of turns.entries()) {
    for (const { step } of exchange.steers) {
      let values: readonly unknown[];
      try {
        values = steerValues(step);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        return { requests, digestOk: false };
      }
      messages.push(...steerMessages(values, { render: options.renderSteer ?? null }));
    }
    // The call's reply's tool calls, which its results answer.
    let calls: ToolCallPart[] = [];
    for (const [at, { n, step }] of exchange.attempts.entries()) {
      if (step["purpose"] !== "chat") {
        const rest = [...exchange.attempts.slice(at), ...turns.slice(turn + 1).flatMap((later) => later.attempts)];
        const structured = await options.structured();
        if (structured === null) {
          for (const later of rest) requests.set(later.n - 1, { request: null, digestOk: false });
          return { requests, digestOk: false };
        }
        const rebuilt = rebuildStructuredRequests(
          structured.target,
          messages,
          structured.typ,
          rest.map((later) => later.step),
        );
        rebuilt.requests.forEach((request, index) => requests.set(rest[index]!.n - 1, request));
        return { requests, digestOk: digestOk && rebuilt.digestOk };
      }
      const request = options.tools === null ? null : new Request({ messages, tools: options.tools });
      const digest = request === null ? null : digests.digest(request);
      const matched = digest !== null && step["request_sha256"] === digest;
      requests.set(n - 1, { request, digestOk: matched });
      if (!matched) digestOk = false;
      if (Object.hasOwn(step, "error")) continue;
      const reply = recordedResponse(step).message;
      messages.push(reply);
      if (step["finish"] === "interrupted") continue;
      try {
        rejectInvalidReply(reply);
      } catch (error) {
        // The run failed the node at this reply before acting on it.
        if (error instanceof TypeCheckError) return { requests, digestOk };
        throw error;
      }
      calls = reply.parts.filter((part): part is ToolCallPart => part instanceof ToolCallPart);
    }
    for (const call of calls) {
      const result = exchange.results.get(call.id)?.step;
      if (result === undefined) continue;
      const content = result["output"] as string;
      const isError = result["is_error"] === true;
      const part = new ToolResultPart({ callId: call.id, name: call.name, content, isError });
      messages.push(new Message({ role: "tool", parts: [part] }));
    }
  }
  return { requests, digestOk };
}

/** What running one reply's tool calls reads and appends to. */
interface ToolTurn {
  readonly ctx: AgentRunContext;
  readonly toolset: ToolSet;
  /** The granted tools, groups expanded, each under the name the model sees. */
  readonly granted: readonly GrantedTool[];
  /** The names the model sees of the granted tools that are resumable. */
  readonly resumable: ReadonlySet<string>;
  readonly skills: ReadonlyMap<string, Skill>;
  readonly live: LiveSinks | null;
  /** The `n` of the reply's `model_call` step, which each call's idempotency key and brackets hold. */
  readonly n: number;
  /** Where each call's `tool_result` step is written. */
  readonly steps: AgentSteps;
  /** The model's messages; each committed result is appended. */
  readonly messages: Message[];
  /** The reply's tool calls, in call order, unparsed ones included. */
  readonly calls: readonly ToolCallPart[];
  /** The host's interrupt signal, or `undefined` for an agent without a steer inbox. */
  readonly interrupt: AbortSignal | undefined;
  /** The calls whose handler returned `pending()`, or that the host's check held: each one's yian call id by its index. */
  readonly pended: Map<number, string>;
  /**
   * The first of the reply's calls that wait until a hold before them is answered, which an honor
   * skips; `null` when none waits.
   */
  waitingFrom: number | null;
}

/**
 * How one tool call ended: the model reads `content` (a reported error's text when `reported`), or
 * it threw `error`, or its result had no text; or its handler returned `pending()`, so the call
 * `pending` waits for a later answer.
 */
type ToolOutcome = CallText | { readonly error: unknown } | { readonly pending: string };

/** The text the model reads for one call's result, a reported error's when `reported`. */
interface CallText {
  readonly content: string;
  readonly reported: boolean;
}

/** One started call of a group and how it ended. */
interface SettledCall {
  readonly planned: PlannedCall;
  readonly outcome: ToolOutcome;
}

/** One call whose check passed, ready to start. */
interface PlannedCall {
  readonly call: ToolCallPart;
  readonly index: number;
  /**
   * The call's yian id, which a `pending` step names; `undefined` outside a run, for `load_skill`,
   * or for a call refused before its tool.
   */
  readonly callId: string | undefined;
  /** When an `ask_user` call times out (a record time), from the model's `timeout`; else `null`. */
  readonly deadline: string | null;
  /**
   * Whether a resume sends it again when a crash or a stop cut it off: its tool is resumable, or
   * it is refused for what the recorded reply alone shows.
   */
  readonly resumable: boolean;
  /** Whether it was refused before its tool, so it never starts: its result records no start. */
  readonly refused: boolean;
  /**
   * Where its tool comes from, which the host's `beforeToolCall` is asked about it with
   * (`ToolSet.sourceOf`); `null` when the hook is never asked about it: `load_skill`, the run's
   * `ask_user`, a call refused before its tool, and any call of a run without the hook.
   */
  readonly source: ToolSource | null;
  /** 1, or more for a call a crash or a stop cut off after its reply, which a resume sends again. */
  readonly attempt: number;
  /**
   * The call's own interrupt, which its tool reads as `ToolContext.interrupted`: made when the call
   * is planned, and aborted by the agent's interrupt from before the host is asked about it until
   * it settles.
   */
  readonly interrupted: AbortController;
  /** Run the call under its group's signal. */
  readonly invoke: (signal: AbortSignal) => Promise<unknown>;
}

/** The outcome of checking one call before it may start. */
type CallCheck = { readonly planned: PlannedCall; readonly safe: boolean } | { readonly error: unknown };

/**
 * Run one reply's tool calls in groups, appending a tool message per call in call order.
 *
 * Calls are checked in call order. Consecutive calls that are safe to run together (a
 * `load_skill` call, or a tool whose `isConcurrencySafe` says so) form a group; an unsafe call
 * runs alone, after the group before it, as does a call refused before its tool, which gets its
 * error result. A check that faults ends the calls before it as their own group, which runs and
 * commits; then the node fails at it. Once a group holds a call (`runGroup` gives `held`), the
 * later calls wait until the hold is answered: none of them starts, and a check that faulted is
 * made again then.
 *
 * A host interrupt seen before a call's check, before a group starts, or at a faulted check skips
 * every call not started yet (`skipFrom`); a running group always finishes and commits, its calls
 * told of the interrupt through `ToolContext.interrupted`.
 *
 * @param options `from` is the first call to run, the calls before it already answered;
 *   `attemptSince` the step after which the calls became runnable (a hold's answer or release),
 *   whose evaluations since count each call's attempts (`checkCall`).
 * @returns Whether a host interrupt skipped the remaining calls.
 * @throws A stop's reason, the first failing call's error, a result `toolText` refuses, or a
 *   faulted check's error.
 */
async function runToolCalls(
  turn: ToolTurn,
  { from = 0, attemptSince = null }: { from?: number; attemptSince?: number | null } = {},
): Promise<boolean> {
  let group: PlannedCall[] = [];
  for (const [index, call] of turn.calls.entries()) {
    if (index < from) continue;
    // Not `checkStopped(ctx)`: without a context signal, the tools run under the tool set's bound
    // run signal, so a stop on it must stop the calls too.
    checkNodeStopped(turn);
    // After the stop; nothing of `group` has started yet, so it is skipped whole.
    if (turn.interrupt?.aborted) return skipFrom(turn, group[0]?.index ?? index);
    const check = await checkCall(turn, call, index, attemptSince);
    if ("error" in check) {
      const before = await runGroup(turn, group);
      return before === "ran" ? await failCall(turn, call, index, check.error) : before === "skipped";
    }
    if (check.safe) {
      group.push(check.planned);
      continue;
    }
    const before = await runGroup(turn, group);
    if (before !== "ran") return before === "skipped";
    group = [];
    const alone = await runGroup(turn, [check.planned]);
    if (alone !== "ran") return alone === "skipped";
  }
  return (await runGroup(turn, group)) === "skipped";
}

/**
 * Give a replayed reply's tool calls their results, appending a tool message per call in call
 * order, as live commits them: a reply the host's check held a call of as `replayHeldCalls` does,
 * any other as `replayCalls` does.
 *
 * @returns Whether a host interrupt skipped calls of this reply, when it was live (a recorded
 *   `skipped` result) or now.
 * @throws As the walk it takes throws.
 */
async function replayToolCalls(turn: ToolTurn): Promise<boolean> {
  const holds = replyHolds(turn);
  return holds.ends.size > 0 ? await replayHeldCalls(turn, holds) : await replayCalls(turn);
}

/**
 * Give a replayed reply's tool calls their results, appending a tool message per call in call
 * order, as live commits them. A call keeps the result an earlier start recorded, unchecked. A
 * call with none is checked as live: a faulted check fails the node as live. Otherwise it is
 * answered by its view (`answerUnrecorded`): pended again while the run still holds it open, sent
 * again when no record closed it and its tool is resumable (or the reply alone refuses it), else
 * closed with the word of the record that closed it, `crashed` when none did.
 *
 * With a host interrupt pending, only the calls that may have run are sent again or closed, and
 * the rest are skipped from the first one on (`skipFrom`), as live skips the calls it hasn't
 * started. A call may have run when it belongs to the group the reply's own start was running at
 * its cut, which ran all its calls together. That group is formed as live formed it, from what the
 * reply's own start recorded (`NodeSteps.resumesSince`; a call's pending step marks the start that
 * ran it, even when a later one closed it): its first call is the first one that start didn't
 * finish, and a later call joins it while both are safe to run together. Any other call was sent,
 * if ever, by a later start, one call at a time: a resumable one may have run when the first start
 * to reach it did so before the interrupt, and one that isn't resumable never ran. A start reaches
 * a call when it writes the result of a call before it; the interrupt's place is its record's
 * (`NodeSteps.seqOf`), so a replay decides as an earlier one did. A call outside the group is
 * skipped before its check.
 *
 * @param after For the calls after a reply's holds (`replayHeldCalls`): they start at `from`; the
 *   evaluation `runner` (as `laterRunner` counts it) ran them as live, so it stands for the reply's
 *   own start; and each call's attempts count from the last hold's `answer`.
 * @returns Whether a host interrupt skipped calls of this reply, when it was live (a recorded
 *   `skipped` result) or now.
 * @throws A stop's reason, a faulted check's error (`failCall`), or what a call sent again throws
 *   (`resendCall`).
 */
async function replayCalls(
  turn: ToolTurn,
  after: { from: number; answer: number; runner: number } | null = null,
): Promise<boolean> {
  const { steps } = turn;
  const resumesSince = turn.ctx.steps?.resumesSince ?? (() => 0);
  const from = after?.from ?? 0;
  const attemptSince = after?.answer ?? null;
  /** Whether step `m` was recorded by the start that recorded the reply, or ran the calls after its holds. */
  const own =
    after === null
      ? (m: number) => resumesSince(m) === resumesSince(turn.n)
      : (m: number) => evaluationsSince(turn, m) === after.runner - 1;
  // The group the reply's own start was running at its cut: not reached yet, still taking safe
  // calls, or ended. Set in `joins`, so declared wide: an initializer alone would narrow it.
  let running = "ahead" as "ahead" | "open" | "ended";
  /** Place a call the reply's own start didn't finish: whether it is in that start's running group. */
  const joins = (safe: boolean): boolean => {
    const joined = running === "ahead" || (running === "open" && safe);
    running = joined && safe ? "open" : "ended";
    return joined;
  };
  /**
   * The step that shows call `index` ran: its pending step, written by the start that ran it, else
   * its result; `null` for neither.
   */
  const ranAt = (call: ToolCallPart, index: number): number | null => {
    const callId = agentCallId(turn, index);
    return (callId === undefined ? null : steps.recordedPending(callId)) ?? steps.recordedResult(turn.n, call.id)?.n ?? null;
  };
  // Only a call with neither can be skipped, so with none the groups needn't be formed.
  const unfinished = turn.calls.some((call, index) => index >= from && ranAt(call, index) === null);
  // The step at which a start after the reply's own last reached the next call: where it finished
  // the call before, with its result or, for a call it pended, its pending step (the answer or the
  // closer comes after, and a host writes the answer). `null` before any.
  let reached: number | null = null;
  /**
   * Whether a resumable call outside the reply's own start's running group may have run: the
   * first start to reach it did so before the interrupt recorded at `by`.
   */
  const mayHaveRun = (call: ToolCallPart, by: number): boolean =>
    turn.resumable.has(call.name) && (reached === null || turn.ctx.steps!.seqOf(reached) < by);
  for (const [index, call] of turn.calls.entries()) {
    if (index < from) continue;
    checkNodeStopped(turn);
    const result = steps.recordedResult(turn.n, call.id);
    // An interrupt skipped from this call's group on; the groups before it committed. A cut can
    // land among the skips, so the later calls without one are skipped now.
    if (result?.closed === "skipped") return skipFrom(turn, index + 1);
    const callId = agentCallId(turn, index);
    // Read at each call: an interrupt can arrive while a call is sent again.
    const by = turn.ctx.steerInbox?.interruptedBy() ?? null;
    const ran = ranAt(call, index);
    const pendedAt = callId === undefined ? null : steps.recordedPending(callId);
    // A later start that pended the call reached the next one there.
    const laterPending = pendedAt !== null && !own(pendedAt) ? pendedAt : null;
    if (laterPending !== null) reached = Math.max(reached ?? 0, laterPending);
    if (ran === null) {
      if (by !== null && running === "ended" && !mayHaveRun(call, by)) return skipFrom(turn, index);
    } else if (unfinished && running !== "ended" && !own(ran)) {
      // A later start finished it, so the reply's own start was still running it at its cut, or
      // never started it; its place in the groups bounds what may have run after it. Placed
      // whether or not an interrupt is pending yet, since one can arrive later in the replay.
      joins(await concurrencySafe(turn, call, index));
    }
    if (result !== null) {
      // A call the reply's own start pended was finished by the later start that closed it.
      if (!own(result.n) && (pendedAt === null || (laterPending === null && result.closed !== null))) {
        reached = Math.max(reached ?? 0, result.n);
      }
      pushToolMessage(turn, call, result.output, result.isError);
      continue;
    }
    const check = await checkCall(turn, call, index, attemptSince);
    if ("error" in check) return await failCall(turn, call, index, check.error);
    // Only a call nothing registered is placed: one the run registered has a pending step.
    if (ran === null && !joins(check.safe) && by !== null && !mayHaveRun(call, by)) return skipFrom(turn, index);
    const written = await answerUnrecorded(turn, check.planned);
    // A later start that pended the call reached the next one already.
    if (written !== null && laterPending === null) reached = written;
  }
  return false;
}

/**
 * Give a replayed reply the host's check held calls of its tool calls' results, as `replayCalls`
 * does, by its holds as the steps and the calls when the evaluation began show them
 * (`replyHolds`). First the holds the host released before the evaluation began run, once
 * (`runReleased`): before anything else is sent again, so a crash while another call is sent
 * never closes a released call that never started. Then, in call order:
 * 1. A call after the group of a hold undecided for this evaluation waits, unchecked: the walk
 *    stops there (`ToolTurn.waitingFrom`), or skips from it with a host interrupt pending.
 * 2. A held call keeps the host's answer, or what its release's run gave, and is pended again too
 *    while it is undecided and no interrupt is pending, so the node wakes at once to run the
 *    waiting calls.
 * 3. A held call with no result is pended again while open or released after the evaluation
 *    began (the node wakes at once to run it), or gets the closer of the interrupt that closed it.
 *    One its release's run registered again, its tool pending or its flow to grow, is answered as
 *    any other call (rule 5).
 * 4. The calls after every hold's group were run as live by the evaluation `laterRunner` finds,
 *    and `replayCalls` replays them from its records. When none ran them, they run as live now
 *    (`runToolCalls`), their attempts counting from the last answer or release, or are skipped
 *    with a host interrupt pending, or were skipped already.
 * 5. Any other call keeps its result, or is answered by its view (`answerUnrecorded`). One with no
 *    record is in a hold's group, since the groups before a hold's ran to their end, so it may have
 *    started with the hold: it is sent again or closed, never skipped. Its attempts count from the
 *    latest answer of the holds before its group.
 *
 * @returns Whether a host interrupt skipped calls of this reply.
 * @throws As `replayCalls`.
 */
async function replayHeldCalls(turn: ToolTurn, holds: ReplyHolds): Promise<boolean> {
  const released = await runReleased(turn, holds);
  for (const [index, call] of turn.calls.entries()) {
    checkNodeStopped(turn);
    // Read at each call: an interrupt can arrive while a call is sent again.
    const by = turn.ctx.steerInbox?.interruptedBy() ?? null;
    if (index > holds.waitAfter) {
      if (by !== null) return skipFrom(turn, index);
      turn.waitingFrom = index;
      return false;
    }
    if (index > holds.lastEnd) {
      // An interrupt skipped the calls from here on, live or as it was honored; a cut can land
      // among the skips, so the later calls without one are skipped now.
      if (turn.steps.recordedResult(turn.n, call.id)?.closed === "skipped") return skipFrom(turn, index + 1);
      const runner = laterRunner(turn, holds, by);
      if (runner !== null) return await replayCalls(turn, { from: index, answer: holds.decidedAt!, runner });
      if (by !== null) return skipFrom(turn, index);
      return await runToolCalls(turn, { from: index, attemptSince: holds.decidedAt });
    }
    const callId = agentCallId(turn, index)!;
    const result = turn.steps.recordedResult(turn.n, call.id);
    const view = turn.ctx.steps?.calls.find((each) => each.callId === callId);
    if (holds.ends.has(index) && (result !== null || view?.held !== false)) {
      const ran = released.get(index);
      if (result !== null) {
        pushToolMessage(turn, call, result.output, result.isError);
        if (holds.undecided.has(index) && by === null) turn.pended.set(index, callId);
      } else if (ran !== undefined) {
        pushToolMessage(turn, call, ran.content, ran.reported);
      } else if (view?.state === "closed") {
        pushToolMessage(turn, call, closeRecorded(turn, call, index, view), true);
      } else {
        // Open, released after the evaluation began, or registered again by its release's run now.
        turn.pended.set(index, callId);
      }
      continue;
    }
    if (result?.closed === "skipped") return skipFrom(turn, index + 1);
    if (result !== null) {
      pushToolMessage(turn, call, result.output, result.isError);
      continue;
    }
    const check = await checkCall(turn, call, index, answerBefore(holds, index));
    if ("error" in check) return await failCall(turn, call, index, check.error);
    await answerUnrecorded(turn, check.planned);
  }
  return false;
}

/**
 * Answer replayed call `planned`, which has no result, by its view as the evaluation began
 * (`NodeSteps.calls`), appending its tool message unless it pends. A call the run still holds
 * open (it pended and waits on, beside a sibling a deadline closed) is pended again, with no closer
 * and no handler call: the loop then returns `Pending`. A flow call its scope answered gets the
 * answer the run shaped for it. A call nothing registered was cut off by a crash or a stop: it is
 * sent again (`resendCall`), with the same call id, when it is resumable (its tool is, or the
 * reply alone refuses it, which gets the same error result again), else closed `crashed`, its
 * outcome unknown. Any other call is closed with the word of the record that closed it
 * (`closeRecorded`), a failed flow call with its failure; it never runs.
 *
 * @returns The `n` of the last step it wrote, or `null` when it pended the call.
 * @throws What a call sent again throws (`resendCall`).
 * @throws CapabilityFault carrying a RangeError for a view the agent has no record to write for,
 *   such as a host's answer its steps don't hold.
 */
async function answerUnrecorded(turn: ToolTurn, planned: PlannedCall): Promise<number | null> {
  const { call, index, callId } = planned;
  const view = turn.ctx.steps?.calls.find((each) => each.callId === callId);
  if (callId !== undefined && view?.state === "open") {
    turn.pended.set(index, callId);
    return null;
  }
  if (view === undefined) {
    if (!planned.resumable) return closeCrashed(turn, call);
    await resendCall(turn, planned);
    return turn.steps.lastWritten();
  }
  // Answered by its flow's scope, or closed: a host's answer is a recorded result, never here.
  const settled = recordSettled(turn, call, index, view);
  if (settled === null) {
    const what = `node ${reprText(turn.ctx.nodeId)}'s call ${reprText(view.callId)} is ${view.state} with no result recorded`;
    throw new CapabilityFault(new YianRangeError(what));
  }
  pushToolMessage(turn, call, settled.output, settled.isError);
  return turn.steps.lastWritten();
}

/** Close `call` `crashed`, its outcome unknown, and append its message; return the closer's `n`. */
function closeCrashed(turn: ToolTurn, call: ToolCallPart): number {
  const { output, closed } = closerResult("crashed");
  const n = turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed });
  pushToolMessage(turn, call, output, true);
  return n;
}

/** What a replay knows of a reply's holds, from its steps and the calls as the evaluation began. */
interface ReplyHolds {
  /** The last call of each held call's group, by the held call's index. */
  readonly ends: ReadonlyMap<number, number>;
  /**
   * The step that decided each hold before the evaluation began, by the held call's index: the
   * host's answer, or its release.
   */
  readonly answers: ReadonlyMap<number, number>;
  /** The `released` step of each hold the host released before the evaluation began, by the held call's index. */
  readonly released: ReadonlyMap<number, number>;
  /** The held calls undecided for this evaluation: open, or answered or released after it began. */
  readonly undecided: ReadonlySet<number>;
  /** Whether a record closed a hold before it ran. */
  readonly closed: boolean;
  /** The calls after this one wait for an undecided hold; `Infinity` when none waits. */
  readonly waitAfter: number;
  /** The last call of the last hold's group; -1 without a hold. */
  readonly lastEnd: number;
  /** The latest of `answers`; `null` when no answer decided a hold. */
  readonly decidedAt: number | null;
}

/**
 * Return what the steps and the calls as the evaluation began (`NodeSteps.calls`) show of the
 * reply's holds. A hold is decided by the host's release or answer before the evaluation began,
 * or by the interrupt that closed it before it ran; one answered or released after the evaluation
 * began (none of its evaluations since, `evaluationsSince`) is undecided for it, as an open one is.
 * A released hold stays decided by its release once its run settled it, or registered it again.
 */
function replyHolds(turn: ToolTurn): ReplyHolds {
  const ends = new Map<number, number>();
  const answers = new Map<number, number>();
  const released = new Map<number, number>();
  const undecided = new Set<number>();
  let closed = false;
  for (const [index, call] of turn.calls.entries()) {
    const callId = agentCallId(turn, index);
    const groupEnd = callId === undefined ? null : turn.steps.recordedHold(callId);
    if (groupEnd === null) continue;
    ends.set(index, groupEnd);
    const view = turn.ctx.steps?.calls.find((each) => each.callId === callId);
    const release = turn.steps.recordedRelease(callId!);
    const answer = view?.state === "answered" ? turn.steps.recordedResult(turn.n, call.id) : null;
    // A call registered again after its release isn't held: a record closing it closes no hold.
    if (view?.state === "closed" && view.held) closed = true;
    else if (release !== null && evaluationsSince(turn, release) > 0) {
      answers.set(index, release);
      released.set(index, release);
    } else if (release === null && answer !== null && evaluationsSince(turn, answer.n) > 0) answers.set(index, answer.n);
    else undecided.add(index);
  }
  const waitAfter = Math.min(...[...undecided].map((index) => ends.get(index)!));
  const decidedAt = answers.size === 0 ? null : Math.max(...answers.values());
  return { ends, answers, released, undecided, closed, waitAfter, lastEnd: Math.max(-1, ...ends.values()), decidedAt };
}

/**
 * Run the reply's holds the host released before this evaluation began (`ReplyHolds.released`)
 * that no evaluation settled or registered again, together, once, as the model asked them: each
 * checked again, the hook not asked again (the release covers it), its attempt the evaluations
 * since its release (`checkCall`). A call no evaluation can have started yet runs, or is skipped
 * with a host interrupt pending, read on the live signal, which a replay's walk doesn't read:
 * this evaluation is the first after its release, or each one after it began with the interrupt
 * pending, which skips the call at once. One an earlier evaluation after its release may have
 * started is sent again, whatever the interrupt, when its tool is resumable, else closed
 * `crashed`, its outcome unknown.
 *
 * @returns The text the model reads for each call, and whether it is an error, by index; none for
 *   a call whose tool pended, or whose flow is to grow, which registers it again.
 * @throws A stop's reason, a faulted check's error (`faultCall`), or what `runGroup` throws.
 */
async function runReleased(turn: ToolTurn, holds: ReplyHolds): Promise<ReadonlyMap<number, CallText>> {
  const texts = new Map<number, CallText>();
  const views = turn.ctx.steps?.calls ?? [];
  const members = [...holds.released].filter(([index]) => views.find((each) => each.callId === agentCallId(turn, index))?.state === "released");
  if (members.length === 0) return texts;
  const by = turn.ctx.steerInbox?.interruptedBy() ?? null;
  /** Whether no evaluation can have started the call `released` released. */
  const fresh = (released: number) => {
    const since = evaluationsSince(turn, released);
    return since === 1 || (by !== null && turn.ctx.steps!.evaluationsAfter(by) >= since);
  };
  const checks = new Map<number, CallCheck>();
  for (const [index, released] of members) {
    const call = turn.calls[index]!;
    if (fresh(released) || turn.resumable.has(call.name)) checks.set(index, await checkCall(turn, call, index, released));
  }
  checkNodeStopped(turn);
  for (const [index] of members) {
    if (checks.has(index)) continue;
    const { output, closed } = closerResult("crashed");
    turn.steps.toolResult({ callId: turn.calls[index]!.id, startedAt: null, output, isError: true, closed });
    texts.set(index, { content: output, reported: true });
  }
  // After the closers, whose writes a host's `onEvent` can deliver an interrupt during; nothing
  // awaits from here to the starts but a fault, which starts nothing.
  const interrupted = turn.interrupt?.aborted === true;
  const running: PlannedCall[] = [];
  for (const [index, released] of members) {
    const check = checks.get(index);
    const call = turn.calls[index]!;
    if (check === undefined) continue;
    if (interrupted && fresh(released)) texts.set(index, { content: skipCall(turn, call, index), reported: true });
    else if ("error" in check) await faultCall(turn, call, index, check.error);
    else running.push(check.planned);
  }
  await runGroup(turn, running, { interruptible: false, gated: false, texts });
  return texts;
}

/**
 * Return the evaluation that ran the calls after every hold's group as live, as the number of the
 * node's evaluations that began at or after it (this one counting 1, so at least 2); `null` when
 * none did.
 *
 * No call after a hold's group starts before every hold is decided, and none after a hold a record
 * closed. Of the evaluations after the last answer or release, the first one that got past every
 * call up to the last group's end ran them, and any later one only sends cut calls again. An
 * evaluation got past a call when it had the call's result, wrote it, or found it open: registered
 * before the evaluation began, and settled after (`NodeSteps.settledAt`). A hold the host answered
 * was past for each of them; one it released, settled at its release, only for an evaluation that
 * had or wrote its run's record. A skip's writer took the interrupt there, so it got past nothing.
 * One that began with the pending interrupt skipped them instead.
 */
function laterRunner(turn: ToolTurn, holds: ReplyHolds, by: number | null): number | null {
  const answer = holds.decidedAt;
  if (answer === null || holds.closed) return null;
  const steps = turn.ctx.steps!;
  /** Whether the evaluation `rank` evaluations back got past call `index`. */
  const passed = (rank: number, index: number): boolean => {
    const call = turn.calls[index]!;
    const callId = agentCallId(turn, index)!;
    const result = turn.steps.recordedResult(turn.n, call.id);
    if (result !== null && result.closed !== "skipped" && rank <= evaluationsSince(turn, result.n) + 1) return true;
    const registered = turn.steps.recordedPending(callId);
    if (registered === null || rank > evaluationsSince(turn, registered) + 1) return false;
    const view = steps.calls.find((each) => each.callId === callId);
    // Open now, or a host's answer, which came while it was open.
    if (view?.state === "open" || (view?.state === "answered" && view.scope === null)) return true;
    // Found open only when it began before the record that settled the call: a wake that timed
    // the call out began with it closed.
    const settled = steps.settledAt(callId);
    return settled === null || rank > steps.evaluationsAfter(settled - 1);
  };
  const answered = (index: number) => holds.ends.has(index) && !holds.released.has(index);
  for (let rank = evaluationsSince(turn, answer); rank > 1; rank--) {
    const past = turn.calls.every((_, index) => index > holds.lastEnd || answered(index) || passed(rank, index));
    if (past) return by !== null && steps.evaluationsAfter(by) >= rank ? null : rank;
  }
  return null;
}

/**
 * The latest answer of the reply's holds whose groups end before call `index`, where the call
 * became runnable; `null` when none does.
 */
function answerBefore(holds: ReplyHolds, index: number): number | null {
  const before = [...holds.answers].filter(([held]) => holds.ends.get(held)! < index).map(([, answer]) => answer);
  return before.length === 0 ? null : Math.max(...before);
}

/** How many of the node's evaluations began after step `n` (`NodeSteps.evaluationsSince`); 1 outside a run. */
function evaluationsSince(turn: ToolTurn, n: number): number {
  return turn.ctx.steps?.evaluationsSince(n) ?? 1;
}

/**
 * Answer each call of a reply at the tool-call cap with the cap's error result (`toolLimitText`),
 * in call order: none is checked, run or pended. A call an earlier start recorded a result for
 * keeps it; each other gets a `tool_result` step, then started and finished events
 * (`CAPPED_TOOL_ERROR`), as a skipped call does. A stop wins first; then a host interrupt skips
 * every call (`skipFrom`), as it would any reply's. The first call's recorded result shows which
 * an earlier start chose, so a replay finishes what it began.
 *
 * @returns Whether a host interrupt skipped the calls.
 */
function answerAtCap(turn: ToolTurn, cap: number): boolean {
  checkNodeStopped(turn);
  const first = turn.steps.recordedResult(turn.n, turn.calls[0]!.id);
  if (first === null ? turn.interrupt?.aborted === true : first.closed === "skipped") return skipFrom(turn, 0);
  const output = toolLimitText(cap);
  for (const [index, call] of turn.calls.entries()) {
    const recorded = turn.steps.recordedResult(turn.n, call.id);
    if (recorded !== null) {
      pushToolMessage(turn, call, recorded.output, recorded.isError);
      continue;
    }
    turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError: true });
    observeSkippedToolCall(turn.live, identityOf(turn, call, index), CAPPED_TOOL_ERROR);
    pushToolMessage(turn, call, output, true);
  }
  return false;
}

/** Append the tool message the model reads for `call`'s result. */
function pushToolMessage(turn: ToolTurn, call: ToolCallPart, content: string, isError: boolean): void {
  const part = new ToolResultPart({ callId: call.id, name: call.name, content, isError });
  turn.messages.push(new Message({ role: "tool", parts: [part] }));
}

/** Whether `call` may run beside other calls; a call whose check fails runs alone. */
async function concurrencySafe(turn: ToolTurn, call: ToolCallPart, index: number): Promise<boolean> {
  const check = await checkCall(turn, call, index);
  return !("error" in check) && check.safe;
}

/**
 * Report the calls from call `start` on as skipped by a host interrupt, in call order (`skipCall`).
 * None of them runs. A call with a result keeps it, whether an earlier start or this one wrote it;
 * a call the run registered keeps the host's answer, or takes its flow's answer or the closer of
 * the record that closed it (`recordSettled`). A held call the host released, which hasn't
 * started, is skipped.
 *
 * @returns `true`, for the caller to return.
 */
function skipFrom(turn: ToolTurn, start: number): true {
  const views = turn.ctx.steps?.currentCalls() ?? [];
  for (const [index, call] of turn.calls.entries()) {
    if (index < start || turn.steps.hasResult(turn.n, call.id)) continue;
    const callId = agentCallId(turn, index);
    const view = views.find((each) => each.callId === callId);
    if (view !== undefined && view.state !== "released") recordSettled(turn, call, index, view);
    else skipCall(turn, call, index);
  }
  return true;
}

/**
 * Write `call`'s `skipped` `tool_result` step (`SKIPPED_TOOL_RESULT`, an error) and its skipped
 * events; return the text the model reads.
 */
function skipCall(turn: ToolTurn, call: ToolCallPart, index: number): string {
  turn.steps.toolResult({ callId: call.id, startedAt: null, output: SKIPPED_TOOL_RESULT, isError: true, closed: "skipped" });
  observeSkippedToolCall(turn.live, identityOf(turn, call, index));
  return SKIPPED_TOOL_RESULT;
}

/**
 * Check one call before it may start: that its arguments parsed, its grant, then its arguments and
 * whether it is safe to run beside others. A check the stop cut short throws the stop's reason.
 * The model's mistakes plan a call refused before its tool (`refused`), whose error result the
 * model reads: arguments that didn't parse, a name it wasn't granted, a bad `ask_user` timeout,
 * and arguments the schema or the validator refuses (the check's message, or the message of a
 * `ReportedToolError` the validator threw).
 *
 * @param attemptSince The step after which the call became runnable (the answer or release of a
 *   hold it waited for, or its own release), whose evaluations since are its attempt
 *   (`NodeSteps.evaluationsSince`); `null` for a call runnable since its reply.
 * @returns The planned call, or the error of a check that failed otherwise (a validator that threw).
 */
async function checkCall(
  turn: ToolTurn,
  call: ToolCallPart,
  index: number,
  attemptSince: number | null = null,
): Promise<CallCheck> {
  const { ctx, toolset, skills } = turn;
  const args = call.arguments;
  if (args === null) return refused(call, index, new ReportedToolError(unparsedText(call.name)), true);
  if (skills.size > 0 && call.name === LOAD_SKILL) {
    // `load_skill` only reads the granted skills. Every `SkillError` it throws refuses the
    // model's arguments, so the model reads it as the call's result; any other error is a fault.
    const invoke = async () => {
      try {
        return invokeLoadSkill(skills, args);
      } catch (error) {
        if (error instanceof SkillError) throw new ReportedToolError(error.message, { cause: error });
        throw error;
      }
    };
    const planned = { call, index, callId: undefined, deadline: null, resumable: false, refused: false };
    const gate = { source: null, attempt: 1, interrupted: new AbortController() };
    return { planned: { ...planned, ...gate, invoke }, safe: true };
  }
  const grant = turn.granted.find((each) => each.visible === call.name);
  // The model sees the granted tools in the request, so the name is all it needs.
  if (grant === undefined) return refused(call, index, new ReportedToolError(unavailableText(call.name)), true);
  // The call names the tool as the model saw it; the grant names it in its group.
  const ref = { group: grant.group, name: grant.tool };
  const resumable = turn.resumable.has(call.name);
  // An ask's deadline is fixed before its handler runs, which gets it; the pending step records it.
  let deadline: string | null = null;
  if (call.name === ASK_USER && typeof args["timeout"] === "string") {
    try {
      deadline = formatRecordTime((ctx.scheduler ?? realScheduler).now() + parseDuration(args["timeout"]) * 1000);
    } catch (error) {
      if (!(error instanceof RangeError || error instanceof TypeError)) throw error;
      return refused(call, index, new ReportedToolError(`ask_user timeout: ${error.message}`), resumable);
    }
  }
  const inputs = deadline === null ? args : { ...args, deadline };
  try {
    // The schema check first, on its own: it reads only the model's arguments, so its failure is
    // the model's to read. The tool was resolved when the request was built, so it is there.
    toolset.checkArguments(ref, inputs, { trusted: deadline !== null });
  } catch (error) {
    return refused(call, index, new ReportedToolError((error as Error).message, { cause: error }), true);
  }
  let checked: CheckedCall;
  try {
    checked =
      deadline === null
        ? await toolset.validate(ref, inputs, { signal: ctx.signal })
        : await toolset.validateTrusted(ref, inputs, { signal: ctx.signal });
  } catch (error) {
    // A check the stop cut short is the stop, not a failed check.
    checkNodeStopped(turn);
    // The arguments are checked as part of the call, so the validator's issues and a reported
    // error are the call's result.
    // A validator may read outside state the tool changes, so a replay checking again tells
    // nothing of a call the cut ran: it is answered again only when its tool is resumable.
    if (isReportedToolError(error)) return refused(call, index, error, resumable);
    if (isToolCheckError(error)) {
      return refused(call, index, new ReportedToolError((error as Error).message, { cause: error }), resumable);
    }
    return { error };
  }
  // The key names the reply's `model_call` step and the call's place in it. Steps outlive a start
  // and a resumed start numbers its steps after the recorded ones, so a key never names two
  // calls that ran: a reply a crash cut before its `model_call` was written ran no tool. Outside a
  // run (no run id) the tool set gives each call a fresh id.
  const callId = agentCallId(turn, index);
  // Above 1 only for a call a crash or a stop cut off after its reply, which a resume sends again;
  // a call that waited for a hold counts from the hold's answer or release, a released one from
  // its own release.
  const attempt = attemptSince === null ? 1 + (ctx.steps?.resumesSince(turn.n) ?? 0) : evaluationsSince(turn, attemptSince);
  const interrupted = new AbortController();
  const invoke = (signal: AbortSignal) =>
    toolset.invoke(checked, { node: ctx.nodeId, callId, attempt, signal, interrupted: interrupted.signal });
  // Under the hook, every call is asked about but an `ask_user` call of the run's ask handler,
  // which waits for a person already. The tool the name resolves to decides, not the name: a
  // node's own tool comes first, under any name.
  const source = toolset.gate === null ? null : toolset.sourceOf(ref);
  const gate = { source: source === null || asksUser(source) ? null : source, attempt };
  const planned = { call, index, callId, deadline, resumable, refused: false, ...gate, interrupted, invoke };
  return { planned, safe: toolset.isConcurrencySafe(checked) };
}

/**
 * Whether `source` is the run's ask handler's: reserved names keep any other host, group, MCP or
 * registry tool from the top-level name `ask_user`.
 */
function asksUser(source: ToolSource): boolean {
  return source.kind === "host" && source.group === null && source.tool === ASK_USER;
}

/**
 * Plan a call refused before it reaches its tool: an unsafe call, so it runs alone, that rejects
 * with `error`, a reported error the model reads as the call's error result. `resumable` says
 * whether a replay that finds no result for it answers it again: `true` when the refusal follows
 * from the recorded reply alone (unparsed arguments, an ungranted name, the schema check), else
 * as its tool is.
 */
function refused(call: ToolCallPart, index: number, error: unknown, resumable: boolean): CallCheck {
  const invoke = () => Promise.reject(error);
  const planned = { call, index, callId: undefined, deadline: null, resumable, refused: true };
  const gate = { source: null, attempt: 1, interrupted: new AbortController() };
  return { planned: { ...planned, ...gate, invoke }, safe: false };
}

/**
 * Fail the node at a call whose check faulted (its validator threw), once the calls before it
 * committed.
 *
 * As for a call about to start, a stop wins first, then a host interrupt skips it and every later
 * call. Otherwise the call gets the brackets of a call that threw, since the arguments are checked
 * as part of the call.
 *
 * @returns `true` when a host interrupt skipped the call.
 * @throws The check's error, or what publishing its brackets throws.
 */
async function failCall(turn: ToolTurn, call: ToolCallPart, index: number, error: unknown): Promise<true> {
  checkNodeStopped(turn);
  if (turn.interrupt?.aborted) return skipFrom(turn, index);
  return await faultCall(turn, call, index, error);
}

/** Fail the node at a call whose check faulted: publish the brackets of a call that threw, then throw `error`. */
async function faultCall(turn: ToolTurn, call: ToolCallPart, index: number, error: unknown): Promise<never> {
  // Rejects with the check's error once its brackets are published.
  await observeToolCall(turn.live, () => Promise.reject(error), identityOf(turn, call, index), {
    signal: nodeSignal(turn),
    text: (value) => toolText(call.name, value),
  });
  throw error;
}

/**
 * Run one group's calls together and commit their results in call order.
 *
 * With the host's check (`ToolSet.gate`), the hook is asked about each call with a source
 * (`PlannedCall.source`), in call order, before any of the group's calls starts (`askGate`). Once
 * every answer is in, or the node stops or the agent takes an interrupt, nothing awaits until the
 * calls start: a stop or an interrupt ends the group (`caught`), writing nothing else. Then each
 * decision is written in call order: a refusal, or an answer that isn't a valid decision
 * (`decide`), as a `refused` result with its brackets; a hold (`pending()`) as its hold step,
 * whose `group_end` is the group's last call. A stop or an interrupt a host delivered while those
 * steps committed ends the group the same way, a written hold taking the closer of the interrupt
 * that closed it. Then the holds pend, the reply's later calls wait until they are answered (the
 * honor of an interrupt skips them from `ToolTurn.waitingFrom`), and the other calls start, each
 * allowed call's note going on its first record. A held call never starts; a refused call's text
 * joins the model's messages in call order with the others' results. A hold outside a durable run
 * fails the group before anything of it is written. A resend can't be held, since it may have run
 * before its cut: it is closed `crashed`.
 *
 * The calls share one child signal of the node's, and each call's own interrupt
 * (`PlannedCall.interrupted`) follows the agent's from before the hook is asked until the call
 * settles. Each starts in call order; the first to fail (in time; a `ReportedToolError` is the
 * call's error result, not a failure, and so is a rejection with the abort of the call's own
 * interrupt, which the model reads as `TOOL_INTERRUPTED_OUTCOME`; a result `toolText` refuses is a
 * failure) aborts the group's signal, and no further call starts. As each call finishes with a
 * result, its text is computed and written as a `tool_result` step at once, in finish order,
 * unless the node has stopped by then; a call whose handler returned `pending()` is written as a
 * `pending` step instead and gets no tool message (the loop returns `Pending`). An agent outside a
 * durable run can't pend: that fails the group. Every started call is awaited; then a stop wins,
 * else the first failure is thrown with nothing of the group committed. A stop, then a host
 * interrupt, is checked before the group starts; the interrupt skips from the group's first call,
 * unless `options.interruptible` is `false` and the host checks none of its calls.
 *
 * With `options.gated` false the hook is asked about none of the calls: the host released them.
 * With `options.texts`, each call's text goes there by its index instead of joining the model's
 * messages, for the caller to append in call order.
 *
 * @returns How the group ended (`GroupEnd`).
 * @throws AgentLoopError, before anything of the group is asked or written, when the host checks
 *   a call outside a run (`gatedCall`), or holds one outside a durable run.
 */
async function runGroup(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  {
    interruptible = true,
    gated = true,
    texts = null,
  }: { interruptible?: boolean; gated?: boolean; texts?: Map<number, CallText> | null } = {},
): Promise<GroupEnd> {
  if (group.length === 0) return "ran";
  const { live } = turn;
  checkNodeStopped(turn);
  const hook = gated ? turn.toolset.gate : null;
  const asking = hook === null ? [] : group.filter(isAsked).map((planned) => gatedCall(turn, planned));
  // A resend may have run before its cut, so an interrupt doesn't skip it; but the hook isn't
  // asked to let it run again once the agent holds one.
  let end = interruptible || asking.length > 0 ? caught(turn, group, interruptible) : null;
  if (end !== null) return end;
  const unlinks = new Map(group.map((planned) => [planned, link(turn.interrupt, planned.interrupted)]));
  let refusals: ReadonlyMap<PlannedCall, string> = NO_RULINGS;
  let notes: ReadonlyMap<PlannedCall, string> = NO_RULINGS;
  let holds: ReadonlySet<AskedCall> = NO_HOLDS;
  /** Whether `planned` is held. */
  const held = (planned: PlannedCall) => isAsked(planned) && holds.has(planned);
  const signal = nodeSignal(turn);
  const controller = new AbortController();
  const relay = () => controller.abort(signal!.reason);
  // Set in a callback, so declared wide: a `null` initializer would narrow it for good.
  let failure = null as { error: unknown } | null;
  const fail = (error: unknown) => {
    // Only the first failure in time is kept and aborts; aborting again changes nothing.
    failure ??= { error };
    controller.abort(error);
  };
  /**
   * Take the text the model reads for one call's result and write its step, with the host's
   * note, as the call finishes; a result with no text, or a write that throws, fails the group. A
   * pended call writes its `pending` step.
   */
  const settle = (planned: PlannedCall, startedAt: number | null, value: unknown, reported: boolean): ToolOutcome => {
    const { call, index, callId, deadline } = planned;
    const note = notes.get(planned) ?? null;
    try {
      // A flow call pends without a host: the run grows the flow's scope and answers it, so it
      // needs no durable log.
      if (!reported && isFlowCall(value)) {
        if (callId === undefined || turn.ctx.steps === undefined) {
          throw new AgentLoopError(`agent node ${reprText(turn.ctx.nodeId)}: flow tool ${reprText(call.name)} runs only in a run`);
        }
        if (signal?.aborted) return { pending: callId };
        const fields = {
          callId, tool: call.name, toolCallId: call.id, reply: turn.n, index, deadline, flowArguments: value.arguments, note,
        };
        // Arguments too large to record are the model's to fix, as its result.
        const tooLarge = turn.steps.pendingTooLarge(fields);
        if (tooLarge !== null) {
          turn.steps.toolResult({ callId: call.id, startedAt, output: tooLarge, isError: true, note }, { settled: true });
          observeFlowCallFinished(live, identityOf(turn, call, index), { error: "failed" });
          return { content: tooLarge, reported: true };
        }
        turn.steps.pending(fields);
        turn.pended.set(index, callId);
        return { pending: callId };
      }
      if (!reported && isPending(value)) {
        if (callId === undefined || turn.ctx.steps === undefined || !turn.ctx.steps.durable) {
          throw new AgentLoopError(
            `agent node ${reprText(turn.ctx.nodeId)}: tool ${reprText(call.name)} returned pending() outside a durable run; ` +
              "give the run a log",
          );
        }
        if (!signal?.aborted) {
          turn.steps.pending({ callId, tool: call.name, toolCallId: call.id, reply: turn.n, index, deadline, note });
          turn.pended.set(index, callId);
        }
        return { pending: callId };
      }
      const content = reported ? reportedText(call.name, value) : toolText(call.name, value);
      // Once the node stopped, the stop may be what ended the call, so its result isn't written.
      if (!signal?.aborted) {
        turn.steps.toolResult({ callId: call.id, startedAt, output: content, isError: reported, note }, { settled: true });
      }
      return { content, reported };
    } catch (error) {
      fail(error);
      return { error };
    }
  };
  try {
    if (hook !== null && asking.length > 0) {
      const { answers, wait } = askGate(turn, hook, asking);
      if (wait !== null) await wait;
      // From here to the starts nothing awaits, so no stop or interrupt lands between the last
      // check and a start.
      end = caught(turn, group, interruptible);
      if (end !== null) return end;
      ({ refusals, notes, holds } = decide(turn, asking, answers));
      if (holds.size > 0 && turn.ctx.steps?.durable !== true) {
        const name = reprText([...holds][0]!.call.name);
        throw new AgentLoopError(
          `agent node ${reprText(turn.ctx.nodeId)}: beforeToolCall held tool ${name} outside a durable run; give the run a log`,
        );
      }
      if (holds.size > 0 && !interruptible) {
        // A resend runs alone. It may have run before its cut, and a hold says its tool never ran.
        const where = `node ${reprText(turn.ctx.nodeId)}: beforeToolCall held tool ${reprText(group[0]!.call.name)}`;
        warn(`${where}, sent again after a cut; it may have run, so it is closed crashed`);
        return interruptGroup(turn, group, false);
      }
      const groupEnd = group.at(-1)!.index;
      for (const planned of group) {
        const text = refusals.get(planned);
        if (text !== undefined) refuse(turn, planned, text);
        else if (isAsked(planned) && holds.has(planned)) hold(turn, planned, groupEnd);
      }
      // A host's `onEvent` can deliver while those steps commit.
      end = caught(turn, group, interruptible, refusals);
      if (end !== null) return end;
      for (const planned of holds) turn.pended.set(planned.index, planned.callId!);
      if (holds.size > 0 && groupEnd + 1 < turn.calls.length) turn.waitingFrom = groupEnd + 1;
    }
    // Removed once the group settles, so a long loop leaves no listeners on the node's signal.
    signal?.addEventListener("abort", relay, { once: true });
    const settling: Promise<SettledCall>[] = [];
    for (const planned of group) {
      if (refusals.has(planned) || held(planned)) continue;
      if (controller.signal.aborted) break;
      const { call, index } = planned;
      let started = false;
      let startedAt: number | null = null;
      const running = observeToolCall(
        live,
        async () => {
          started = true;
          if (!planned.refused) startedAt = turn.steps.now();
          try {
            return await planned.invoke(controller.signal);
          } finally {
            // A call that returned never learns of a later interrupt.
            unlinks.get(planned)!();
          }
        },
        identityOf(turn, call, index),
        { signal, text: (value) => toolText(call.name, value) },
      );
      const outcome = running.then(
        (value): ToolOutcome => settle(planned, startedAt, value, false),
        (error): ToolOutcome => {
          // A call whose started event threw never ran: what was thrown is the host's failure.
          if (started && isReportedToolError(error)) return settle(planned, startedAt, error, true);
          // A tool that passed its interrupt on (to a timer, `fetch`, `spawn`) rejects with its
          // abort: the user's act, which the model reads, not the tool's failure.
          if (started && interruptAbort(planned.interrupted.signal, error)) {
            return settle(planned, startedAt, new ReportedToolError(TOOL_INTERRUPTED_OUTCOME), true);
          }
          fail(error);
          return { error };
        },
      );
      settling.push(outcome.then((ended) => ({ planned, outcome: ended })));
      // `observeToolCall` calls `invoke` synchronously unless publishing the started event threw;
      // its rejection is seen only later, so the next call must not start in the meantime.
      if (!started) break;
    }
    const settled = await Promise.all(settling);
    // A stop pending when the calls settle wins over the first failure and over their results.
    checkNodeStopped(turn);
    if (failure !== null) throw failure.error;
    const outcomes = new Map(settled.map(({ planned, outcome }) => [planned, outcome]));
    for (const planned of group) {
      // A held call has no result yet; the loop returns before the model reads any.
      if (held(planned)) continue;
      const refusal = refusals.get(planned);
      const outcome = refusal === undefined ? outcomes.get(planned)! : { content: refusal, reported: true };
      // Neither has a pended call. No outcome is an error: a failure was thrown above.
      if (!("content" in outcome)) continue;
      if (texts === null) pushToolMessage(turn, planned.call, outcome.content, outcome.reported);
      else texts.set(planned.index, outcome);
    }
  } finally {
    signal?.removeEventListener("abort", relay);
    for (const unlink of unlinks.values()) unlink();
  }
  return holds.size > 0 ? "held" : "ran";
}

/**
 * How a group ended: a host interrupt skipped it and every later call; it held a call, so the
 * reply's later calls wait until the hold is answered; or each of its calls has its result or
 * pended.
 */
type GroupEnd = "skipped" | "held" | "ran";

/** What the hook answered about one call: its value, or what it threw or rejected with. */
type GateAnswer = { readonly value: unknown } | { readonly fault: unknown };

/**
 * A valid decision of the hook: run the call, with its note (`null` for none), refuse it with the
 * text the model reads, or hold it.
 */
type GateRuling = { readonly note: string | null } | { readonly refuse: string } | { readonly hold: true };

/** One call the hook is asked about, and what it sees of it. */
interface Asked {
  readonly planned: AskedCall;
  readonly call: GatedCall;
}

/** A planned call the hook is asked about: one with a source. */
type AskedCall = PlannedCall & { readonly source: ToolSource };

/** Whether the hook is asked about `planned` (`PlannedCall.source`). */
function isAsked(planned: PlannedCall): planned is AskedCall {
  return planned.source !== null;
}

/** No refusals or notes: a group the hook isn't asked about. */
const NO_RULINGS: ReadonlyMap<PlannedCall, string> = new Map();

/** No holds: a group the hook isn't asked about. */
const NO_HOLDS: ReadonlySet<AskedCall> = new Set();

/**
 * End the group when a stop or an interrupt caught it before its calls started: a stop throws its
 * reason, and an interrupt answers the group (`interruptGroup`), with the texts of the refusals
 * this start wrote (`refusals`).
 *
 * @returns What `interruptGroup` returns; `null` when neither caught the group.
 */
function caught(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  interruptible: boolean,
  refusals: ReadonlyMap<PlannedCall, string> = NO_RULINGS,
): GroupEnd | null {
  checkNodeStopped(turn);
  return turn.interrupt?.aborted ? interruptGroup(turn, group, interruptible, refusals) : null;
}

/**
 * Ask `hook` about each of `asking`, in call order, before any await, until the group is stopped
 * or interrupted: once it is, the later calls aren't asked. Every call is asked with one signal,
 * which aborts once the node stops or the agent takes an interrupt. A thenable answer becomes a
 * Promise, whose rejection is a fault.
 *
 * @returns The answers by call, and `wait`, which resolves once every answer is in or the signal
 *   aborts; `null` when no answer is out. An answer still out then is ignored when it comes; there
 *   is no timeout.
 */
function askGate(
  turn: ToolTurn,
  hook: BeforeToolCall,
  asking: readonly Asked[],
): { readonly answers: ReadonlyMap<PlannedCall, GateAnswer>; readonly wait: Promise<void> | null } {
  const answers = new Map<PlannedCall, GateAnswer>();
  const stop = new AbortController();
  const unlinks = [link(nodeSignal(turn), stop), link(turn.interrupt, stop)];
  const ctx = Object.freeze({ runId: turn.toolset.runId, signal: stop.signal });
  const waits: Promise<void>[] = [];
  let open = true;
  const answer = (planned: PlannedCall, given: GateAnswer) => {
    if (open) answers.set(planned, given);
  };
  for (const { planned, call } of asking) {
    if (stop.signal.aborted) break;
    try {
      const value: unknown = hook(call, ctx);
      if (!isThenable(value)) answer(planned, { value });
      else {
        const given = Promise.resolve(value);
        waits.push(given.then((later) => answer(planned, { value: later }), (fault) => answer(planned, { fault })));
      }
    } catch (fault) {
      answer(planned, { fault });
    }
  }
  const close = () => {
    open = false;
    for (const unlink of unlinks) unlink();
  };
  if (waits.length === 0 || stop.signal.aborted) {
    close();
    return { answers, wait: null };
  }
  const stopped = new Promise<void>((resolve) => {
    stop.signal.addEventListener("abort", () => resolve(), { once: true });
  });
  return { answers, wait: Promise.race([Promise.all(waits), stopped]).then(close) };
}

/** Whether `value` has a `then` method, as a Promise does. Reading it may throw. */
function isThenable(value: unknown): value is PromiseLike<unknown> {
  const object = (typeof value === "object" && value !== null) || typeof value === "function";
  return object && typeof (value as { then?: unknown }).then === "function";
}

/** The `scopes` of a call outside a run's steps, which no flow call encloses. */
const NO_SCOPES: readonly FlowCallScope[] = Object.freeze([]);

/**
 * Return what the hook sees of `planned`, deep-frozen: its arguments are a copy of the model's, so
 * the hook can't change what the tool gets, and its scopes are the flow calls enclosing the agent
 * (`NodeSteps.flowCalls`).
 *
 * @throws AgentLoopError when `planned` has no call id: outside a run.
 */
function gatedCall(turn: ToolTurn, planned: AskedCall): Asked {
  const { call, callId, source, attempt } = planned;
  if (callId === undefined) {
    throw new AgentLoopError(`agent node ${reprText(turn.ctx.nodeId)}: beforeToolCall checks tool calls only in a run`);
  }
  const args = freezeParsed(copyJson(call.arguments!));
  const scopes = turn.ctx.steps?.flowCalls() ?? NO_SCOPES;
  const gated: GatedCall = { callId, node: turn.ctx.nodeId, tool: call.name, source, arguments: args, scopes, attempt };
  return { planned, call: Object.freeze(gated) };
}

/**
 * Return the hook's answers about `asking` as the group's refusals, notes and holds, in call order.
 * An answer that threw or rejected, or that isn't a valid decision (`checkDecision`), refuses its
 * call with `GATE_FAILED_TEXT` and warns with its error, which the model never reads.
 */
function decide(
  turn: ToolTurn,
  asking: readonly Asked[],
  answers: ReadonlyMap<PlannedCall, GateAnswer>,
): {
  refusals: ReadonlyMap<PlannedCall, string>;
  notes: ReadonlyMap<PlannedCall, string>;
  holds: ReadonlySet<AskedCall>;
} {
  const refusals = new Map<PlannedCall, string>();
  const notes = new Map<PlannedCall, string>();
  const holds = new Set<AskedCall>();
  for (const { planned } of asking) {
    let ruling: GateRuling;
    try {
      const answer = answers.get(planned)!;
      if ("fault" in answer) throw answer.fault;
      ruling = checkDecision(answer.value);
    } catch (error) {
      const where = `node ${reprText(turn.ctx.nodeId)}: beforeToolCall failed for tool ${reprText(planned.call.name)}`;
      warn(`${where}: ${faultText(error)}; the call is refused`);
      ruling = { refuse: GATE_FAILED_TEXT };
    }
    if ("refuse" in ruling) refusals.set(planned, ruling.refuse);
    else if ("hold" in ruling) holds.add(planned);
    else if (ruling.note !== null) notes.set(planned, ruling.note);
  }
  return { refusals, notes, holds };
}

/**
 * Return `value` as a ruling when it is a valid decision: `undefined`, a plain object
 * `{run: true}` or `{run: true, note}` (a `note` that is `undefined` is none), `{refuse}`, each
 * text passing `checkGateText`, or `pending()`.
 *
 * @throws TypeError or RangeError for anything else.
 */
function checkDecision(value: unknown): GateRuling {
  if (value === undefined) return { note: null };
  if (isPending(value)) return { hold: true };
  const shapes = "a decision must be undefined, {run: true}, {run: true, note} or {refuse}";
  if (!isRecord(value)) throw new YianTypeError(shapes);
  const keys = Object.keys(value);
  if (keys.length === 1 && keys[0] === "refuse") return { refuse: checkGateText(value["refuse"], "refuse") };
  if (!keys.every((key) => key === "run" || key === "note") || value["run"] !== true) throw new YianTypeError(shapes);
  const note = Object.hasOwn(value, "note") ? value["note"] : undefined;
  return { note: note === undefined ? null : checkGateText(note, "note") };
}

/** The message of `error` for a warning; never throws. */
function faultText(error: unknown): string {
  try {
    return error instanceof Error ? error.message : String(error);
  } catch {
    return typeof error;
  }
}

/** Write `planned`'s refusal, `text`, as its `refused` result, then its brackets. */
function refuse(turn: ToolTurn, { call, index }: PlannedCall, text: string): void {
  turn.steps.toolResult({ callId: call.id, startedAt: null, output: text, isError: true, closed: "refused" });
  observeSkippedToolCall(turn.live, identityOf(turn, call, index), REFUSED_TOOL_ERROR);
}

/** Write `planned`'s hold, in a group whose last call is `groupEnd`; it publishes no brackets. */
function hold(turn: ToolTurn, { call, index, callId, source }: AskedCall, groupEnd: number): void {
  const fields = { callId: callId!, tool: call.name, toolCallId: call.id, reply: turn.n, index, source, groupEnd };
  turn.steps.hold(fields);
}

/**
 * Answer a group a host interrupt caught before any of its calls started: skip it and every later
 * call (`skipFrom`), each call with a result this start wrote, such as a refusal, keeping it; or,
 * for a resend, which may have run before its cut, close each of its calls `crashed`
 * (`closeCrashed`), a call in `refusals` keeping the refusal this start wrote, whose text it
 * appends instead.
 *
 * @returns `skipped`, or `ran` for a resend.
 */
function interruptGroup(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  interruptible: boolean,
  refusals: ReadonlyMap<PlannedCall, string> = NO_RULINGS,
): GroupEnd {
  if (interruptible) {
    skipFrom(turn, group[0]!.index);
    return "skipped";
  }
  for (const planned of group) {
    const refusal = refusals.get(planned);
    if (refusal === undefined) closeCrashed(turn, planned.call);
    else pushToolMessage(turn, planned.call, refusal, true);
  }
  return "ran";
}

/**
 * Send again one call a crash or a stop cut off, as a group of its own: its started and finished
 * events, its `tool_result` step (or its `pending` step) and its tool message, a reported error as
 * the call's result, as live. The replay sends again only a call that may have started before the
 * cut, so a pending host interrupt doesn't skip it: its `ToolContext.interrupted` is aborted from
 * its start, and the loop honors the interrupt afterwards. The host's check is asked about it
 * again; a hold it gives closes the call `crashed` (`runGroup`).
 *
 * @throws As `runGroup` throws.
 */
async function resendCall(turn: ToolTurn, planned: PlannedCall): Promise<void> {
  await runGroup(turn, [planned], { interruptible: false });
}

/**
 * Abort `to` with `from`'s reason once `from` aborts, at once when it has; return a function that
 * unlinks them, so `to` never learns of a later abort. Without `from` it does nothing.
 */
function link(from: AbortSignal | undefined, to: AbortController): () => void {
  if (from === undefined) return () => {};
  const relay = () => to.abort(from.reason);
  if (from.aborted) relay();
  else from.addEventListener("abort", relay, { once: true });
  return () => from.removeEventListener("abort", relay);
}

/**
 * Whether a call's rejection `error` is the abort of its aborted `interrupted`: the signal's
 * reason, or an `AbortError`, which an API given the signal rejects with and which a module tool's
 * thread sends back by name.
 */
function interruptAbort(interrupted: AbortSignal, error: unknown): boolean {
  if (!interrupted.aborted) return false;
  try {
    return error === interrupted.reason || (error instanceof Error && error.name === "AbortError");
  } catch {
    // A thrown Proxy whose traps throw is no abort.
    return false;
  }
}

/** The yian id of call `index` of the turn's reply, `<run>/<node>/<n>/<index>`; `undefined` outside a run. */
function agentCallId(turn: ToolTurn, index: number): string | undefined {
  const runId = turn.toolset.runId;
  return runId ? `${runId}/${turn.ctx.nodeId}/${turn.n}/${index}` : undefined;
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
  return { n: turn.n, callIndex: index, toolCallId, name, arguments: call.arguments };
}

/**
 * Throw `TypeCheckError` if two of a reply's tool calls share an id, or a tool call's arguments
 * nest too deep.
 *
 * Results are paired with their calls by id, so calls sharing one couldn't be told apart (the
 * conversation sent next, a replay and an AGENT's new messages would all give each the last
 * result). An unparsed call's raw arguments text is parsed at any depth and checked the same way,
 * since the assembler leaves a call unparsed when its arguments nest too deep; text that is not
 * JSON, or that passes the received value bound before it nests too deep, is left to the invalid
 * call's own handling.
 */
function rejectInvalidReply(reply: Message): void {
  const names = new Map<string, string>();
  for (const part of reply.parts) {
    if (!(part instanceof ToolCallPart)) continue;
    const first = names.get(part.id);
    if (first !== undefined) {
      throw new TypeCheckError(
        `the reply has two tool calls with id ${reprText(part.id)} (${first} and ${part.name}); ` +
          "each tool call in a reply needs its own id",
      );
    }
    names.set(part.id, part.name);
  }
  for (const part of reply.parts) {
    if (!(part instanceof ToolCallPart)) continue;
    // `ToolCallPart` already refuses parsed arguments past the limit; checking them again matters
    // only if the depth limit changed after the part was built.
    const where = `tool call ${reprText(part.name)} arguments`;
    let args = part.arguments;
    try {
      args ??= parseDeepArguments(part.rawArguments!);
    } catch (error) {
      if (!(error instanceof ArgumentsTooDeepError)) throw error;
      throw new TypeCheckError(`${where}: ${error.message}`);
    }
    if (args !== null) rejectDeepNesting(args, where);
  }
}
