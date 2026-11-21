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
 * Knows about: diagnostics, events, llm, llm_clients connection, platform, nodes/base, skills,
 *              tools, typesys, peer AGENT modules.
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
 * @throws AgentLoopError when a tool returns `pending()` outside a durable run.
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
  // Where each granted tool comes from, which the host's check before each call reads.
  const sources = new Map(
    toolset.gate === null ? [] : granted.map((each) => [each.visible, toolset.sourceOf({ group: each.group, name: each.tool })] as const),
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
    const turn: ToolTurn = { ctx, toolset, granted, resumable, sources, skills, live, n, steps, messages, calls, interrupt, pended };
    if (capped || iterations === ctx.maxToolIterations) {
      // The model reads why its calls didn't run, then gives its answer.
      if (answerAtCap(turn, ctx.maxToolIterations)) return honor(ctx, steps);
      if (capped) return await answer();
      capped = true;
      continue;
    }
    const skipped = await (replayed ? replayToolCalls(turn) : runToolCalls(turn));
    // The interrupt closed the reply's pending calls; the agent doesn't wait on them.
    if (skipped || (pended.size > 0 && interruptRequested(interrupt, steps))) {
      closePended(turn);
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
 * Write a closer for each call of the turn that pended and was closed since (the interrupt closed
 * it, or it was registered closed), in call order, with the word of what closed it. A call the
 * host answered meanwhile keeps its answer.
 */
function closePended(turn: ToolTurn): void {
  const views = turn.ctx.steps?.currentCalls() ?? [];
  for (const [index, callId] of [...turn.pended].sort(([a], [b]) => a - b)) {
    const view = views.find((each) => each.callId === callId);
    if (view?.state !== "closed") continue;
    const { output, closed } = closerResult(closerReason(view.closedReason));
    const call = turn.calls[index]!;
    turn.steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed });
    if (view.scope !== null) observeFlowCallFinished(turn.live, identityOf(turn, call, index), { error: closed });
  }
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
  /**
   * Where each granted tool comes from, by the name the model sees, when the host checks each call
   * (`ToolSet.gate`); `null` for a node's own tool (`ToolSet.sourceOf`). Empty without the check.
   */
  readonly sources: ReadonlyMap<string, ToolSource | null>;
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
  /** The calls whose handler returned `pending()`: each one's yian call id by its index. */
  readonly pended: Map<number, string>;
}

/**
 * How one tool call ended: the model reads `content` (a reported error's text when `reported`), or
 * it threw `error`, or its result had no text; or its handler returned `pending()`, so the call
 * `pending` waits for a later answer.
 */
type ToolOutcome =
  | { readonly content: string; readonly reported: boolean }
  | { readonly error: unknown }
  | { readonly pending: string };

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
   * Where its tool comes from, when the host's `beforeToolCall` is asked about it; `null` for a
   * call the host isn't asked about: `load_skill`, `ask_user`, one refused before its tool, one of
   * the node's own tools, or any call of a run without the check.
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
 * commits; then the node fails at it.
 *
 * A host interrupt seen before a call's check, before a group starts, or at a faulted check skips
 * every call not started yet (`skipFrom`); a running group always finishes and commits, its calls
 * told of the interrupt through `ToolContext.interrupted`.
 *
 * @returns Whether a host interrupt skipped the remaining calls.
 * @throws A stop's reason, the first failing call's error, a result `toolText` refuses, or a
 *   faulted check's error.
 */
async function runToolCalls(turn: ToolTurn): Promise<boolean> {
  let group: PlannedCall[] = [];
  for (const [index, call] of turn.calls.entries()) {
    // Not `checkStopped(ctx)`: without a context signal, the tools run under the tool set's bound
    // run signal, so a stop on it must stop the calls too.
    checkNodeStopped(turn);
    // After the stop; nothing of `group` has started yet, so it is skipped whole.
    if (turn.interrupt?.aborted) return skipFrom(turn, group[0]?.index ?? index);
    const check = await checkCall(turn, call, index);
    if ("error" in check) {
      if (await runGroup(turn, group)) return true;
      return await failCall(turn, call, index, check.error);
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

/**
 * Give a replayed reply's tool calls their results, appending a tool message per call in call
 * order, as live commits them. A call keeps the result an earlier start recorded, unchecked. A
 * call with none is checked as live: a faulted check fails the node as live. Otherwise a call the
 * run still holds open (it pended and waits on, beside a sibling a deadline closed) is pended
 * again, with no closer and no handler call, and gets no tool message: the loop then returns
 * `Pending`. A call to a resumable tool that no record closed was cut off by a crash or a stop,
 * so it is sent again (`resendCall`), with the same call id; so is a call refused for what the
 * reply alone shows (unparsed arguments, an ungranted name, the schema check), which gets the same
 * error result again. Any other call is closed with the word of the record that closed it
 * (`closerResult`: `timed_out` for an ask whose deadline passed, `interrupted` for a call a host
 * interrupt closed), `crashed` when none did (its run was cut off before its result), the closer
 * written as its `tool_result` step, and never runs.
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
 * @returns Whether a host interrupt skipped calls of this reply, when it was live (a recorded
 *   `skipped` result) or now.
 * @throws A stop's reason, a faulted check's error (`failCall`), or what a call sent again throws
 *   (`resendCall`).
 */
async function replayToolCalls(turn: ToolTurn): Promise<boolean> {
  const { steps } = turn;
  const resumesSince = turn.ctx.steps?.resumesSince ?? (() => 0);
  /** Whether step `m` was recorded by the start that recorded the reply. */
  const own = (m: number) => resumesSince(m) === resumesSince(turn.n);
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
  const unfinished = turn.calls.some((call, index) => ranAt(call, index) === null);
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
    const check = await checkCall(turn, call, index);
    if ("error" in check) return await failCall(turn, call, index, check.error);
    const view = turn.ctx.steps?.calls.find((each) => each.callId === callId);
    if (callId !== undefined && view?.state === "open") {
      turn.pended.set(index, callId);
      continue;
    }
    // A flow call its scope answered: the agent records the answer the run shaped for it.
    if (view?.state === "answered" && view.result !== null) {
      const output = view.result["output"] as string;
      const isError = view.result["is_error"] === true;
      const written = steps.toolResult({ callId: call.id, startedAt: null, output, isError });
      observeFlowCallFinished(turn.live, identityOf(turn, call, index), { result: output });
      if (laterPending === null) reached = written;
      pushToolMessage(turn, call, output, isError);
      continue;
    }
    if (ran === null && !joins(check.safe) && by !== null && !mayHaveRun(call, by)) return skipFrom(turn, index);
    if (view === undefined && check.planned.resumable) {
      await resendCall(turn, check.planned);
      reached = steps.lastWritten() ?? reached;
      continue;
    }
    // A flow call that failed: the model reads the failure.
    const { output, closed } =
      view?.closedReason === "failed" && view.failure !== null
        ? { output: flowFailureText(call.name, view.scope, view.failure), closed: "failed" as const }
        : closerResult(closerReason(view?.closedReason ?? null));
    const closer = steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed });
    if (view?.scope != null) observeFlowCallFinished(turn.live, identityOf(turn, call, index), { error: closed });
    // A later start that pended the call reached the next one already.
    if (laterPending === null) reached = closer;
    pushToolMessage(turn, call, output, true);
  }
  return false;
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
 * Report the calls from call `start` on as skipped by a host interrupt, in call order: each gets a
 * `skipped` `tool_result` step (`SKIPPED_TOOL_RESULT`, an error) and its skipped events. None of
 * them runs. A call an earlier start recorded a result for keeps it, as does one at `answered`,
 * whose result this start wrote.
 *
 * @returns `true`, for the caller to return.
 */
function skipFrom(turn: ToolTurn, start: number, answered: ReadonlySet<number> = new Set()): true {
  for (const [index, call] of turn.calls.entries()) {
    if (index < start || answered.has(index) || turn.steps.recordedResult(turn.n, call.id) !== null) continue;
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
 * Check one call before it may start: that its arguments parsed, its grant, then its arguments and
 * whether it is safe to run beside others. A check the stop cut short throws the stop's reason.
 * The model's mistakes plan a call refused before its tool (`refused`), whose error result the
 * model reads: arguments that didn't parse, a name it wasn't granted, a bad `ask_user` timeout,
 * and arguments the schema or the validator refuses (the check's message, or the message of a
 * `ReportedToolError` the validator threw).
 *
 * @returns The planned call, or the error of a check that failed otherwise (a validator that threw).
 */
async function checkCall(turn: ToolTurn, call: ToolCallPart, index: number): Promise<CallCheck> {
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
    const planned = { call, index, callId: undefined, deadline: null, resumable: false, refused: false, source: null, attempt: 1 };
    return { planned: { ...planned, interrupted: new AbortController(), invoke }, safe: true };
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
  // Above 1 only for a call a crash or a stop cut off after its reply, which a resume sends again.
  const attempt = 1 + (ctx.steps?.resumesSince(turn.n) ?? 0);
  const interrupted = new AbortController();
  const invoke = (signal: AbortSignal) =>
    toolset.invoke(checked, { node: ctx.nodeId, callId, attempt, signal, interrupted: interrupted.signal });
  // `ask_user` waits for a person already; reserved names keep any other tool from showing as it.
  const source = call.name === ASK_USER ? null : (turn.sources.get(call.name) ?? null);
  const planned = { call, index, callId, deadline, resumable, refused: false, source, attempt, interrupted, invoke };
  return { planned, safe: toolset.isConcurrencySafe(checked) };
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
  const planned = { call, index, callId: undefined, deadline: null, resumable, refused: true, source: null, attempt: 1 };
  return { planned: { ...planned, interrupted: new AbortController(), invoke }, safe: false };
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
 * With the host's check (`ToolSet.gate`), the hook is asked about each call it checks
 * (`PlannedCall.source`), in call order, before any of the group's calls starts (`askGate`). Once
 * every answer is in, or the node stops or the agent takes an interrupt while one is out, nothing
 * awaits until the calls start: a stop throws its reason and an interrupt answers the group
 * (`interruptGroup`), writing nothing else; each refusal, and each answer that isn't a valid
 * decision (`decide`), is written in call order as a `refused` result with its brackets; a stop or
 * an interrupt a host delivered while those steps committed is honored the same way; then the
 * other calls start, each allowed call's note going on its first record. A refused call's text
 * joins the model's messages in call order with the others' results.
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
 * @returns Whether a host interrupt skipped the group and every later call.
 * @throws AgentLoopError when the host checks a call outside a run (`gatedCall`).
 */
async function runGroup(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  { interruptible = true }: { interruptible?: boolean } = {},
): Promise<boolean> {
  if (group.length === 0) return false;
  const { live } = turn;
  checkNodeStopped(turn);
  const hook = turn.toolset.gate;
  const gated = hook === null ? [] : group.filter((planned) => planned.source !== null);
  // A resend may have run before its cut, so an interrupt doesn't skip it; but the hook isn't
  // asked to let it run again once the agent holds one.
  if (turn.interrupt?.aborted && (interruptible || gated.length > 0)) return interruptGroup(turn, group, interruptible);
  const unlinks = new Map(group.map((planned) => [planned, link(turn.interrupt, planned.interrupted)]));
  const refusals = new Map<PlannedCall, string>();
  const notes = new Map<PlannedCall, string>();
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
        const fields = { callId, tool: call.name, toolCallId: call.id, reply: turn.n, index, deadline, flowArguments: value.arguments };
        // Arguments too large to record are the model's to fix, as its result.
        const tooLarge = turn.steps.pendingTooLarge(fields);
        if (tooLarge !== null) {
          turn.steps.toolResult({ callId: call.id, startedAt, output: tooLarge, isError: true }, { settled: true });
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
    if (hook !== null && gated.length > 0) {
      const asked = askGate(turn, hook, gated);
      const answers = asked instanceof Map ? asked : await asked;
      // From here to the starts nothing awaits, so no stop or interrupt lands between the last
      // check and a start.
      checkNodeStopped(turn);
      if (turn.interrupt?.aborted) return interruptGroup(turn, group, interruptible);
      for (const planned of gated) {
        const ruling = decide(turn, planned, answers.get(planned)!);
        if ("refuse" in ruling) refusals.set(planned, ruling.refuse);
        else if (ruling.note !== null) notes.set(planned, ruling.note);
      }
      for (const [planned, text] of refusals) refuse(turn, planned, text);
      // A host's `onEvent` can deliver while those steps commit.
      checkNodeStopped(turn);
      if (turn.interrupt?.aborted) return interruptGroup(turn, group, interruptible, refusals);
    }
    // Removed once the group settles, so a long loop leaves no listeners on the node's signal.
    signal?.addEventListener("abort", relay, { once: true });
    const settling: Promise<SettledCall>[] = [];
    for (const planned of group) {
      if (refusals.has(planned)) continue;
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
          if (isReportedToolError(error)) return settle(planned, startedAt, error, true);
          // A tool that passed its interrupt on (to a timer, `fetch`, `spawn`) rejects with its
          // abort: the user's act, which the model reads, not the tool's failure.
          if (interruptAbort(planned.interrupted.signal, error)) {
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
      const refusal = refusals.get(planned);
      const outcome = refusal === undefined ? outcomes.get(planned)! : { content: refusal, reported: true };
      // A pended call has no result yet; the loop returns before the model reads any. No outcome
      // is an error: a failure was thrown above.
      if (!("content" in outcome)) continue;
      pushToolMessage(turn, planned.call, outcome.content, outcome.reported);
    }
  } finally {
    signal?.removeEventListener("abort", relay);
    for (const unlink of unlinks.values()) unlink();
  }
  return false;
}

/** What the hook answered about one call: its value, or what it threw or rejected with. */
type GateAnswer = { readonly value: unknown } | { readonly fault: unknown };

/** A valid decision of the hook: run the call, with its note (`null` for none), or refuse it with the text the model reads. */
type GateRuling = { readonly note: string | null } | { readonly refuse: string };

/**
 * Ask `hook` about each of `gated`, in call order, before any await. Each is asked with a signal
 * that aborts once the node stops or the call's own interrupt does, which follows the agent's. A
 * thenable answer becomes a Promise, whose rejection is a fault.
 *
 * @returns The answers by call; or, when some answer is a Promise, a Promise of them, which
 *   resolves once every answer is in or a signal aborts. An answer still out then is left out, and
 *   ignored when it comes; there is no timeout.
 * @throws AgentLoopError when a call has no call id (`gatedCall`); the hook isn't asked then.
 */
function askGate(
  turn: ToolTurn,
  hook: BeforeToolCall,
  gated: readonly PlannedCall[],
): Map<PlannedCall, GateAnswer> | Promise<Map<PlannedCall, GateAnswer>> {
  const calls = gated.map((planned) => gatedCall(turn, planned));
  const answers = new Map<PlannedCall, GateAnswer>();
  const waits: Promise<void>[] = [];
  const stops: AbortSignal[] = [];
  const unlinks: (() => void)[] = [];
  let open = true;
  const answer = (planned: PlannedCall, given: GateAnswer) => {
    if (open) answers.set(planned, given);
  };
  for (const [at, planned] of gated.entries()) {
    const stop = new AbortController();
    stops.push(stop.signal);
    unlinks.push(link(nodeSignal(turn), stop), link(planned.interrupted.signal, stop));
    try {
      const value: unknown = hook(calls[at]!, Object.freeze({ runId: turn.toolset.runId, signal: stop.signal }));
      if (!isThenable(value)) answer(planned, { value });
      else waits.push(Promise.resolve(value).then((later) => answer(planned, { value: later }), (fault) => answer(planned, { fault })));
    } catch (fault) {
      answer(planned, { fault });
    }
  }
  const close = () => {
    open = false;
    for (const unlink of unlinks) unlink();
    return answers;
  };
  if (waits.length === 0) return close();
  const stopped = Promise.withResolvers<void>();
  for (const stop of stops) {
    if (stop.aborted) stopped.resolve();
    else stop.addEventListener("abort", () => stopped.resolve(), { once: true });
  }
  return Promise.race([Promise.all(waits), stopped.promise]).then(close);
}

/** Whether `value` has a `then` method, as a Promise does. Reading it may throw. */
function isThenable(value: unknown): value is PromiseLike<unknown> {
  const object = (typeof value === "object" && value !== null) || typeof value === "function";
  return object && typeof (value as { then?: unknown }).then === "function";
}

/** The `scopes` of every gated call: the hook isn't told of the flow calls enclosing an agent. */
const NO_SCOPES: readonly FlowCallScope[] = Object.freeze([]);

/**
 * Return the call the hook sees for `planned`, deep-frozen. Its arguments are a copy of the
 * model's, so the hook can't change what the tool gets.
 *
 * @throws AgentLoopError when `planned` has no call id: the hook is asked only in a run.
 */
function gatedCall(turn: ToolTurn, planned: PlannedCall): GatedCall {
  const { call, callId, source, attempt } = planned;
  if (callId === undefined) {
    throw new AgentLoopError(`agent node ${reprText(turn.ctx.nodeId)}: beforeToolCall checks tool calls only in a run`);
  }
  const args = freezeParsed(copyJson(call.arguments!));
  return Object.freeze({ callId, node: turn.ctx.nodeId, tool: call.name, source: source!, arguments: args, scopes: NO_SCOPES, attempt });
}

/**
 * Return the hook's answer about `planned` as a ruling. An answer that threw or rejected, or that
 * isn't a valid decision (`checkDecision`), refuses the call with `GATE_FAILED_TEXT` and warns
 * with its error, which the model never reads.
 */
function decide(turn: ToolTurn, planned: PlannedCall, answer: GateAnswer): GateRuling {
  try {
    if ("fault" in answer) throw answer.fault;
    return checkDecision(answer.value);
  } catch (error) {
    const node = reprText(turn.ctx.nodeId);
    warn(`node ${node}: beforeToolCall failed for tool ${reprText(planned.call.name)}: ${faultText(error)}; the call is refused`);
    return { refuse: GATE_FAILED_TEXT };
  }
}

/**
 * Return `value` as a ruling when it is a valid decision: `undefined`, a plain object
 * `{run: true}` or `{run: true, note}`, or `{refuse}`, each text passing `checkGateText`.
 *
 * @throws TypeError or RangeError for anything else, `pending()` included.
 */
function checkDecision(value: unknown): GateRuling {
  if (value === undefined) return { note: null };
  if (isPending(value)) throw new YianTypeError("a hold (pending()) isn't supported");
  const shapes = "a decision must be undefined, {run: true}, {run: true, note} or {refuse}";
  if (!isRecord(value)) throw new YianTypeError(shapes);
  const keys = Object.keys(value);
  if (keys.length === 1 && keys[0] === "refuse") return { refuse: checkGateText(value["refuse"], "refuse") };
  if (!keys.every((key) => key === "run" || key === "note") || value["run"] !== true) throw new YianTypeError(shapes);
  return { note: Object.hasOwn(value, "note") ? checkGateText(value["note"], "note") : null };
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

/**
 * Answer a group a host interrupt caught before any of its calls started: skip it and every later
 * call (`skipFrom`); or, for a resend, which may have run before its cut, close each of its calls
 * `crashed` and append its message. A call in `refusals` keeps the refusal this start wrote.
 *
 * @returns Whether the calls were skipped.
 */
function interruptGroup(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  interruptible: boolean,
  refusals: ReadonlyMap<PlannedCall, string> = new Map(),
): boolean {
  if (interruptible) return skipFrom(turn, group[0]!.index, new Set([...refusals.keys()].map((planned) => planned.index)));
  const { output, closed } = closerResult("crashed");
  for (const planned of group) {
    const refusal = refusals.get(planned);
    if (refusal === undefined) turn.steps.toolResult({ callId: planned.call.id, startedAt: null, output, isError: true, closed });
    pushToolMessage(turn, planned.call, refusal ?? output, true);
  }
  return false;
}

/**
 * Send again one call a crash or a stop cut off, as a group of its own: its started and finished
 * events, its `tool_result` step (or its `pending` step) and its tool message, a reported error as
 * the call's result, as live. The replay sends again only a call that may have started before the
 * cut, so a pending host interrupt doesn't skip it: its `ToolContext.interrupted` is aborted from
 * its start, and the loop honors the interrupt afterwards.
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
