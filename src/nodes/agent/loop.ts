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
 * instead, as it does any reply's.
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
 * aborts a running tool call. In the chat loop, a steer queued when the model gives its final
 * answer gets one more model call, within `maxToolIterations`; a structured agent without tools or
 * skills reads its steers once, before its single call.
 *
 * Knows about: diagnostics, events, llm, llm_clients connection, platform, nodes/base, skills,
 *              tools, typesys, peer AGENT modules.
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
import { realScheduler, warn } from "../../platform.ts";
import { type Skill, SkillError } from "../../skills/catalog.ts";
import { invokeLoadSkill, LOAD_SKILL, loadSkillTool } from "../../skills/tool.ts";
import { ASK_USER, isPending } from "../../tools/context.ts";
import { isReportedToolError, isToolCheckError, ReportedToolError } from "../../tools/errors.ts";
import {
  asToolGrant,
  expandGrants,
  GrantError,
  type GrantedTool,
  requestToolSpecs,
} from "../../tools/grants.ts";
import { type CheckedCall, ToolSet } from "../../tools/registry.ts";
import { YianRangeError } from "../../typesys/errors.ts";
import { formatRecordTime, parseDuration } from "../../typesys/temporal.ts";
import { rejectDeepNesting, type Type, TypeCheckError, type ValueRecord } from "../../typesys/values.ts";
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
import { Interrupted, ReplyInterrupted } from "./interrupts.ts";
import {
  CAPPED_TOOL_ERROR,
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  observeSkippedToolCall,
  observeToolCall,
  type ToolCallIdentity,
} from "./live.ts";
import {
  type CloserReason,
  closerResult,
  reportedText,
  SKIPPED_TOOL_RESULT,
  toolLimitText,
  toolText,
  unavailableText,
  unparsedText,
} from "./results.ts";
import { foldSteps, type SteerRender, steerMessages, steerValues } from "./messages.ts";
import type { StructuredTarget } from "../../llm/structured.ts";
import { generateStructured, rebuildStructuredRequests, typeToSchema } from "./structured.ts";

/**
 * Run one AGENT invocation.
 *
 * A structured agent without tools or skills reads its queued steers once and makes one structured
 * call. Any other agent runs the chat loop until the model answers; while the model answers with
 * steers queued it reads them and calls again (each reply, late steers included, counts toward
 * `maxToolIterations`; at the cap the answer stands and the steers stay queued). A reply at the
 * cap that still calls tools runs none: its calls get the cap's error result, and one more model
 * call, with no steer read, gives the answer (its text even when it calls tools again). Sets
 * `ctx.sent` to the built model's route credentials once the model is built. Each reason the
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
      return await answer();
    }

    const pended = new Map<number, string>();
    const turn: ToolTurn = { ctx, toolset, granted, resumable, skills, live, n, steps, messages, calls, interrupt, pended };
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
    turn.steps.toolResult({ callId: turn.calls[index]!.id, startedAt: null, output, isError: true, closed });
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
  readonly call: ToolCallPart;
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
 * every call not started yet (`skipFrom`); a running group always finishes and commits.
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
    if (ran === null && !joins(check.safe) && by !== null && !mayHaveRun(call, by)) return skipFrom(turn, index);
    if (view === undefined && check.planned.resumable) {
      await resendCall(turn, check.planned);
      reached = steps.lastWritten() ?? reached;
      continue;
    }
    const { output, closed } = closerResult(closerReason(view?.closedReason ?? null));
    const closer = steps.toolResult({ callId: call.id, startedAt: null, output, isError: true, closed });
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
 * them runs. A call an earlier start recorded a result for keeps it.
 *
 * @returns `true`, for the caller to return.
 */
function skipFrom(turn: ToolTurn, start: number): true {
  for (const [index, call] of turn.calls.entries()) {
    if (index < start || turn.steps.recordedResult(turn.n, call.id) !== null) continue;
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
    const planned = { call, index, callId: undefined, deadline: null, resumable: false, refused: false, invoke };
    return { planned, safe: true };
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
  const invoke = (signal: AbortSignal) => toolset.invoke(checked, { node: ctx.nodeId, callId, attempt, signal });
  const planned = { call, index, callId, deadline, resumable, refused: false, invoke };
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
  return { planned: { call, index, callId: undefined, deadline: null, resumable, refused: true, invoke }, safe: false };
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
 * The calls share one child signal of the node's. Each starts in call order; the first to fail
 * (in time; a `ReportedToolError` is the call's error result, not a failure; a result `toolText`
 * refuses is a failure) aborts the group's signal, and no further call starts. As each call
 * finishes with a result, its text is computed and written as a `tool_result` step at once, in
 * finish order, unless the node has stopped by then; a call whose handler returned `pending()` is
 * written as a `pending` step instead and gets no tool message (the loop returns `Pending`). An
 * agent outside a durable run can't pend: that fails the group.
 * Every started call is awaited; then a stop wins, else the first failure is thrown with nothing of
 * the group committed. A stop, then a host
 * interrupt, is checked before the group starts; the interrupt skips from the group's first call,
 * unless `options.interruptible` is `false`.
 *
 * @returns Whether a host interrupt skipped the group and every later call.
 */
async function runGroup(
  turn: ToolTurn,
  group: readonly PlannedCall[],
  { interruptible = true }: { interruptible?: boolean } = {},
): Promise<boolean> {
  if (group.length === 0) return false;
  const { live } = turn;
  checkNodeStopped(turn);
  if (interruptible && turn.interrupt?.aborted) return skipFrom(turn, group[0]!.index);
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
   * a result with no text, or a write that throws, fails the group. A pended call writes its
   * `pending` step.
   */
  const settle = (planned: PlannedCall, startedAt: number | null, value: unknown, reported: boolean): ToolOutcome => {
    const { call, index, callId, deadline } = planned;
    try {
      if (!reported && isPending(value)) {
        if (callId === undefined || turn.ctx.steps === undefined || !turn.ctx.steps.durable) {
          throw new AgentLoopError(
            `agent node ${reprText(turn.ctx.nodeId)}: tool ${reprText(call.name)} returned pending() outside a durable run; ` +
              "give the run a log",
          );
        }
        if (!signal?.aborted) {
          turn.steps.pending({ callId, tool: call.name, toolCallId: call.id, reply: turn.n, index, deadline });
          turn.pended.set(index, callId);
        }
        return { pending: callId };
      }
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
    const settling: Promise<SettledCall>[] = [];
    for (const planned of group) {
      if (controller.signal.aborted) break;
      const { call, index } = planned;
      let started = false;
      let startedAt: number | null = null;
      const running = observeToolCall(
        live,
        () => {
          started = true;
          if (!planned.refused) startedAt = turn.steps.now();
          return planned.invoke(controller.signal);
        },
        identityOf(turn, call, index),
        { signal, text: (value) => toolText(call.name, value) },
      );
      const outcome = running.then(
        (value): ToolOutcome => settle(planned, startedAt, value, false),
        (error): ToolOutcome => {
          if (isReportedToolError(error)) return settle(planned, startedAt, error, true);
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
      // A pended call has no result yet; the loop returns before the model reads any. No outcome
      // is an error: a failure was thrown above.
      if (!("content" in outcome)) continue;
      const { content, reported } = outcome;
      const { id: callId, name } = call;
      const part = new ToolResultPart({ callId, name, content, isError: reported });
      turn.messages.push(new Message({ role: "tool", parts: [part] }));
    }
  } finally {
    signal?.removeEventListener("abort", relay);
  }
  return false;
}

/**
 * Send again one call a crash or a stop cut off, as a group of its own: its started and finished
 * events, its `tool_result` step (or its `pending` step) and its tool message, a reported error as
 * the call's result, as live. The replay sends again only a call that may have started before the
 * cut, so a pending host interrupt doesn't skip it; the loop honors the interrupt afterwards.
 *
 * @throws As `runGroup` throws.
 */
async function resendCall(turn: ToolTurn, planned: PlannedCall): Promise<void> {
  await runGroup(turn, [planned], { interruptible: false });
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
 * JSON is left to the invalid call's own handling.
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
    const args = part.arguments ?? parseArguments(part.rawArguments!, { anyDepth: true })[0];
    if (args !== null) rejectDeepNesting(args, `tool call ${reprText(part.name)} arguments`);
  }
}
