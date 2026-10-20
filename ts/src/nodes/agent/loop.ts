/**
 * The AGENT loop: drive only newly returned model calls using current grants.
 *
 * Without tools or skills the loop makes a single model call (structured when the output type
 * needs it); with them it runs the tool loop until the model answers.
 *
 * An optional memory-owned turn retains actual exchanges. Restored provider context never drives
 * tool dispatch, and its synthetic missing-result repairs are not saved.
 *
 * Here: the tool-free path; the tool loop, its tool-call argument checks and memory transcripts are
 * not ported yet.
 *
 * Knows about:   diagnostics, llm, memory contracts, nodes/base, typesys, peer AGENT modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import type { ChatModel } from "../../llm/events.ts";
import { RouteModel } from "../../llm/executor.ts";
import { Message, Request, TextPart, ToolCallPart } from "../../llm/messages.ts";
import { ToolCallInteraction } from "../../memory/contracts.ts";
import { writeAsText } from "../../typesys/values.ts";
import { Output } from "../base.ts";
import { AgentLoopError, type AgentRunContext, checkStopped, ToolIterationLimitError } from "./context.ts";
import { callModel, liveSinks, markModelFailure, newExchangeId } from "./live.ts";
import { systemPromptWithMemory } from "./memory.ts";
import { generateStructured, typeToSchema, userMessage } from "./structured.ts";
import { recordModelText, textOf } from "./text.ts";

/**
 * Run one AGENT invocation: one model call, structured when the output type has a schema.
 *
 * Sets `ctx.sent` to the built model's route credentials once the model is built.
 *
 * @returns The checked structured value, or the reply's visible text.
 * @throws Error before the model is built when a tool expands from the grants, a skill is
 *   granted, a steering inbox is present, or the memory context holds a transcript.
 * @throws ToolIterationLimitError when the reply calls tools at a cap of 1 model turn.
 * @throws AgentLoopError when the reply calls a tool, which is never granted here.
 * @throws Whatever building or calling the model throws (marked by `markModelFailure`), a stop's
 *   reason, or `generateStructured`'s `OutputInvalidError`.
 */
export async function runAgent(ctx: AgentRunContext): Promise<Output> {
  // A granted id may name a host-filled group; grants are checked by tool name.
  const granted = ctx.toolset?.expand(ctx.tools) ?? [...ctx.tools];
  if (granted.length > 0 || ctx.skills.length > 0 || ctx.steering !== undefined) {
    throw new Error("the AGENT tool loop is not ported yet (T5-11b)");
  }
  if (ctx.memoryContext.transcript !== null) {
    throw new Error("AGENT memory transcripts are not ported yet (T6-6)");
  }
  const systemPrompt = systemPromptWithMemory(ctx.baseSystemPrompt, ctx.memoryContext.sections);
  const messages = [new Message({ role: "system", parts: [new TextPart(systemPrompt)] }), userMessage(ctx.prompt)];
  const live = liveSinks(ctx);
  checkStopped(ctx);
  const signal = ctx.signal ?? new AbortController().signal;
  let model: ChatModel;
  try {
    // No replay observer: a single tool-free turn never replays an assistant turn.
    model = await ctx.models.createModel(ctx.llmConfig ?? {}, { nodeId: ctx.nodeId, signal });
  } catch (error) {
    // Building the model resolves its credentials, which can fail as refused ones.
    markModelFailure(error);
    throw error;
  }
  ctx.sent = () => (model instanceof RouteModel ? model.route.auth.credentials() : []);
  checkStopped(ctx);
  ctx.evaluationTrace?.recordPrompt(ctx.baseSystemPrompt, ctx.prompt, ctx.memoryContext.sections);
  if (ctx.outputType !== null && typeToSchema(ctx.outputType) !== null) {
    const value = await generateStructured(model, messages, ctx.outputType, {
      retries: ctx.retries,
      live,
      signal: ctx.signal,
      evaluationTrace: ctx.evaluationTrace,
    });
    return new Output(value);
  }

  checkStopped(ctx);
  const response = await callModel(live, model, new Request({ messages }), {
    purpose: "chat",
    exchangeId: newExchangeId(),
    signal: ctx.signal,
  });
  // A stop that lands during the call wins over its reply.
  checkStopped(ctx);
  const reply = response.message;
  // A call whose arguments didn't parse is Python's invalid tool call, which the loop never runs.
  const calls = reply.parts.filter((part): part is ToolCallPart => part instanceof ToolCallPart && part.arguments !== null);
  if (calls.length === 0) {
    const text = textOf(reply);
    recordModelText(ctx.evaluationTrace, text);
    return new Output(text);
  }
  if (ctx.maxToolIterations === 1) {
    // No later model call could read these results, so running them would only add side
    // effects to a node that is already failing.
    throw new ToolIterationLimitError(
      `agent node ${reprText(ctx.nodeId)} hit the tool-iteration cap (${ctx.maxToolIterations}) without a final ` +
        `answer; the last reply's ${calls.length} tool call(s) were not run`,
    );
  }
  const call = calls[0]!;
  if (ctx.evaluationTrace !== undefined) {
    let interaction: ToolCallInteraction | null = null;
    try {
      interaction = new ToolCallInteraction(call.name, call.arguments);
    } catch (error) {
      // Evidence stays observational for malformed arguments.
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    }
    if (interaction !== null) ctx.evaluationTrace.recordInteraction(interaction);
  }
  throw new AgentLoopError(
    `agent node ${reprText(ctx.nodeId)} requested tool ${reprText(call.name)}, but it is not granted; ` +
      `granted tools: ${writeAsText(granted)}`,
  );
}
