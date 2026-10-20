/**
 * The AGENT loop: drive only newly returned model calls using current grants.
 *
 * Without tools or skills the loop makes a single model call (structured when the output type
 * needs it); with them it runs the tool loop until the model answers.
 *
 * An optional memory-owned turn retains actual exchanges. Restored provider context never drives
 * tool dispatch, and its synthetic missing-result repairs are not saved.
 *
 * A reply's tool calls run in groups (design 6.9): consecutive calls that are safe to run together
 * start together, an unsafe call runs alone, and results commit in call order whatever order the
 * calls finish in. Python runs them one at a time, which is what a reply without safe tools gets.
 *
 * A steerable agent reads the host's steers at the top of each iteration and stops at a host
 * interrupt between effects or before a later chunk of a streamed reply, returning its latest
 * reply's text (a cut reply's text so far); an interrupt never aborts a running tool call.
 *
 * Here: memory transcripts are not ported yet.
 *
 * Knows about:   diagnostics, llm, nodes/base, skills, tools, typesys, peer AGENT modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import { type ChatModel, parseArguments } from "../../llm/events.ts";
import { RouteModel } from "../../llm/executor.ts";
import {
  Message,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
} from "../../llm/messages.ts";
import type { Skill } from "../../skills/catalog.ts";
import { invokeLoadSkill, LOAD_SKILL, loadSkillTool } from "../../skills/tool.ts";
import { isReportedToolError } from "../../tools/errors.ts";
import { type CheckedCall, ToolSet } from "../../tools/registry.ts";
import { rejectDeepNesting, writeAsText } from "../../typesys/values.ts";
import { Output } from "../base.ts";
import {
  AgentLoopError,
  type AgentRunContext,
  checkStopped,
  drainSteers,
  interruptSignal,
  ToolIterationLimitError,
} from "./context.ts";
import { ReplyInterrupted } from "./interrupts.ts";
import {
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  newExchangeId,
  observeSkippedToolCall,
  observeToolCall,
  type ToolCallIdentity,
} from "./live.ts";
import { reportedText, systemPromptWithMemory, toolText } from "./memory.ts";
import { generateStructured, typeToSchema, userMessage } from "./structured.ts";
import { recordModelText, recordToolCall, recordToolResult, textOf } from "./text.ts";

/**
 * Run one AGENT invocation.
 *
 * Makes one model call without tools or skills, otherwise runs the tool loop until the model
 * answers. Sets `ctx.sent` to the built model's route credentials once the model is built.
 *
 * @returns The checked structured value, or the final reply's visible text; after a host
 *   interrupt, the latest reply's visible text (`""` before any).
 * @throws Error before the model is built when the memory context holds a transcript.
 * @throws RangeError when a grant names no tool of the run (`ToolSet.resolve`).
 * @throws TypeCheckError when a reply's tool call arguments nest too deep.
 * @throws ToolIterationLimitError when a reply calls tools at the `maxToolIterations`-th turn.
 * @throws AgentLoopError when a reply calls a tool that isn't granted.
 * @throws Whatever building or calling the model throws (marked by `markModelFailure`), a stop's
 *   reason, a tool's own error other than a reported one, a failed argument check, a result
 *   `toolText` refuses, or `generateStructured`'s `OutputInvalidError`. The host inbox's faults
 *   (`CapabilityFault`s) pass through.
 */
export async function runAgent(ctx: AgentRunContext): Promise<Output> {
  if (ctx.memoryContext.transcript !== null) {
    throw new Error("AGENT memory transcripts are not ported yet (T6-6)");
  }
  const systemPrompt = systemPromptWithMemory(ctx.baseSystemPrompt, ctx.memoryContext.sections);
  const messages = [new Message({ role: "system", parts: [new TextPart(systemPrompt)] }), userMessage(ctx.prompt)];
  const live = liveSinks(ctx);
  checkStopped(ctx);
  let model: ChatModel;
  try {
    model = await ctx.models.createModel(ctx.llmConfig ?? {}, {
      nodeId: ctx.nodeId,
      signal: ctx.signal ?? new AbortController().signal,
    });
  } catch (error) {
    // Building the model resolves its credentials, which can fail as refused ones.
    markModelFailure(error);
    throw error;
  }
  ctx.sent = () => (model instanceof RouteModel ? model.route.auth.credentials() : []);
  checkStopped(ctx);
  ctx.evaluationTrace?.recordPrompt(ctx.baseSystemPrompt, ctx.prompt, ctx.memoryContext.sections);
  const schema = ctx.outputType !== null ? typeToSchema(ctx.outputType) : null;
  const structuredOptions = {
    retries: ctx.retries,
    live,
    signal: ctx.signal,
    evaluationTrace: ctx.evaluationTrace,
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

  while (true) {
    checkStopped(ctx);
    if (interrupt?.aborted) return interrupted(reply);
    // Steers join only here, never between a reply's tool calls and their results.
    const steers = await drainSteers(ctx);
    // A stop during the drain wins; its steers are never sent.
    checkStopped(ctx);
    for (const steer of steers) messages.push(userMessage(steer));
    const exchangeId = newExchangeId();
    let response: Response;
    try {
      response = await callModel(live, model, new Request({ messages, tools: bound }), {
        purpose: "chat",
        exchangeId,
        signal: ctx.signal,
        interrupt,
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
    // A stop that lands during the call wins over its reply.
    checkStopped(ctx);
    reply = response.message;
    rejectInvalidArguments(reply);
    iterations += 1;
    messages.push(reply);
    // A call whose arguments didn't parse is an invalid tool call, which the loop never runs.
    const calls = reply.parts.filter((part): part is ToolCallPart => part instanceof ToolCallPart && part.arguments !== null);
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

    if (await runToolCalls({ ctx, toolset, granted, skills, live, exchangeId, messages, calls, interrupt })) {
      return interrupted(reply);
    }
  }
}

/** End an interrupted loop successfully with the latest reply's visible text; `""` before any. */
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
  readonly exchangeId: string;
  /** The model's messages; each committed result is appended. */
  readonly messages: Message[];
  /** The reply's runnable tool calls, in call order. */
  readonly calls: readonly ToolCallPart[];
  /** The host's interrupt signal, or `undefined` for an agent that isn't steerable. */
  readonly interrupt: AbortSignal | undefined;
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

/**
 * Report the calls from call `start` on as skipped by a host interrupt, in call order; none of
 * them runs or leaves evidence.
 *
 * @returns `true`, for the caller to return.
 */
function skipFrom(turn: ToolTurn, start: number): true {
  for (const [index, call] of turn.calls.entries()) {
    if (index >= start) observeSkippedToolCall(turn.live, identityOf(turn, call, index));
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
  // Unlike a TOOL node's id, this one is not stable across resume: a resumed AGENT asks its model
  // again, and reusing an id for a possibly different call would let an idempotency key dedupe the
  // wrong side effect. Outside a run (no run id) the tool set gives each call a fresh id.
  const runId = toolset.runId;
  const callId = runId ? `${runId}/${ctx.nodeId}/${turn.exchangeId}/${index}` : undefined;
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
 * @throws The failed check's error.
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
 * (in time; a `ReportedToolError` is the call's error result, not a failure) aborts the group's
 * signal, and no further call starts. Every started call is awaited; then a stop wins, else the
 * first failure is thrown with nothing of the group committed. A stop, then a host interrupt, is
 * checked before the group starts; the interrupt skips from the group's first call.
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
  try {
    for (const { call } of group) recordToolCall(ctx.evaluationTrace, call.name, call.arguments);
    const outcomes: Promise<{ value: unknown; reported: boolean } | null>[] = [];
    for (const planned of group) {
      if (controller.signal.aborted) break;
      const { call, index } = planned;
      let started = false;
      const running = observeToolCall(
        live,
        () => {
          started = true;
          return planned.invoke(controller.signal);
        },
        identityOf(turn, call, index),
        { signal, text: (value) => toolText(call.name, value) },
      );
      outcomes.push(
        running.then(
          (value) => ({ value, reported: false }),
          (error) => {
            if (isReportedToolError(error)) return { value: error, reported: true };
            // Only the first failure in time is kept and aborts; aborting again changes nothing.
            failure ??= { error };
            controller.abort(error);
            return null;
          },
        ),
      );
      // `observeToolCall` calls `invoke` synchronously unless publishing the started event threw;
      // its rejection is seen only later, so the next call must not start in the meantime.
      if (!started) break;
    }
    const settled = await Promise.all(outcomes);
    // A stop pending when the calls settle wins over the first failure and over their results.
    checkNodeStopped(turn);
    if (failure !== null) throw failure.error;
    for (const [position, outcome] of settled.entries()) {
      const { call } = group[position]!;
      // A `null` outcome is a failure, and a failure was thrown above.
      const { value, reported } = outcome!;
      const content = reported ? reportedText(call.name, value) : toolText(call.name, value);
      recordToolResult(ctx.evaluationTrace, call.name, content);
      const part = new ToolResultPart({ callId: call.id, name: call.name, content, isError: reported });
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
  return { exchangeId: turn.exchangeId, callIndex: index, toolCallId, name, arguments: call.arguments! };
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
