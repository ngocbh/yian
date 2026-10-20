/**
 * The AGENT loop: drive only newly returned model calls using current grants.
 *
 * Without tools or skills the loop makes a single model call (structured when the output type
 * needs it); with them it runs the tool loop until the model answers.
 *
 * An optional memory-owned turn retains actual exchanges. Restored provider context never drives
 * tool dispatch, and its synthetic missing-result repairs are not saved.
 *
 * Here: the tool loop runs a reply's calls one at a time, in call order; steering and memory
 * transcripts are not ported yet.
 *
 * Knows about:   diagnostics, llm, nodes/base, skills, tools, typesys, peer AGENT modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import { type ChatModel, parseArguments } from "../../llm/events.ts";
import { RouteModel } from "../../llm/executor.ts";
import { Message, Request, TextPart, ToolCallPart, ToolResultPart, ToolSpec } from "../../llm/messages.ts";
import { invokeLoadSkill, LOAD_SKILL, loadSkillTool } from "../../skills/tool.ts";
import { isReportedToolError } from "../../tools/errors.ts";
import { ToolSet } from "../../tools/registry.ts";
import { rejectDeepNesting, writeAsText } from "../../typesys/values.ts";
import { Output } from "../base.ts";
import { AgentLoopError, type AgentRunContext, checkStopped, ToolIterationLimitError } from "./context.ts";
import { callModel, liveSinks, markModelFailure, newExchangeId, observeToolCall } from "./live.ts";
import { reportedText, systemPromptWithMemory, toolText } from "./memory.ts";
import { generateStructured, typeToSchema, userMessage } from "./structured.ts";
import { recordModelText, recordToolCall, recordToolResult, textOf } from "./text.ts";

/**
 * Run one AGENT invocation.
 *
 * Makes one model call without tools or skills, otherwise runs the tool loop until the model
 * answers. Sets `ctx.sent` to the built model's route credentials once the model is built.
 *
 * @returns The checked structured value, or the final reply's visible text.
 * @throws Error before the model is built when a steering inbox is present or the memory context
 *   holds a transcript.
 * @throws RangeError when a grant names no tool of the run (`ToolSet.resolve`).
 * @throws TypeCheckError when a reply's tool call arguments nest too deep.
 * @throws ToolIterationLimitError when a reply calls tools at the `maxToolIterations`-th turn.
 * @throws AgentLoopError when a reply calls a tool that isn't granted.
 * @throws Whatever building or calling the model throws (marked by `markModelFailure`), a stop's
 *   reason, a tool's own error other than a reported one, a failed argument check, a result
 *   `toolText` refuses, or `generateStructured`'s `OutputInvalidError`.
 */
export async function runAgent(ctx: AgentRunContext): Promise<Output> {
  if (ctx.steering !== undefined) throw new Error("AGENT steering is not ported yet (T5-11b)");
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
  let iterations = 0;

  while (true) {
    checkStopped(ctx);
    const exchangeId = newExchangeId();
    const response = await callModel(live, model, new Request({ messages, tools: bound }), {
      purpose: "chat",
      exchangeId,
      signal: ctx.signal,
    });
    // A stop that lands during the call wins over its reply.
    checkStopped(ctx);
    const reply = response.message;
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

    for (const [callIndex, call] of calls.entries()) {
      checkStopped(ctx);
      const name = call.name;
      const args = call.arguments!;
      let invoke: () => Promise<unknown>;
      if (skills.size > 0 && name === LOAD_SKILL) {
        invoke = async () => invokeLoadSkill(skills, args);
      } else if (granted.includes(name)) {
        try {
          const checked = await toolset.validate(name, args, { signal: ctx.signal });
          // Unlike a TOOL node's id, this one is not stable across resume: a resumed AGENT asks
          // its model again, and reusing an id for a possibly different call would let an
          // idempotency key dedupe the wrong side effect. Outside a run (no run id) the tool set
          // gives each call a fresh id.
          const runId = toolset.runId;
          const callId = runId ? `${runId}/${ctx.nodeId}/${exchangeId}/${callIndex}` : undefined;
          invoke = () => toolset.invoke(checked, { node: ctx.nodeId, callId, signal: ctx.signal });
        } catch (error) {
          // A check the stop cut short is the stop, not a failed check.
          checkStopped(ctx);
          // Python checks the arguments inside the call, so a failed check gets the brackets of
          // a call that threw.
          invoke = () => Promise.reject(error);
        }
      } else {
        recordToolCall(ctx.evaluationTrace, name, args);
        throw new AgentLoopError(
          `agent node ${reprText(ctx.nodeId)} requested tool ${reprText(name)}, but it is not granted; ` +
            `granted tools: ${writeAsText(granted)}`,
        );
      }
      recordToolCall(ctx.evaluationTrace, name, args);
      const identity = { exchangeId, callIndex, toolCallId: call.id, name, arguments: args };
      let result: unknown;
      let reported = false;
      try {
        result = await observeToolCall(live, invoke, identity, {
          signal: ctx.signal,
          text: (value) => toolText(name, value),
        });
      } catch (error) {
        if (!isReportedToolError(error)) throw error;
        result = error;
        reported = true;
      }
      // A stop pending when the call settles wins over its result, a reported error included.
      checkStopped(ctx);
      const content = reported ? reportedText(name, result) : toolText(name, result);
      recordToolResult(ctx.evaluationTrace, name, content);
      const part = new ToolResultPart({ callId: call.id, name, content, isError: reported });
      messages.push(new Message({ role: "tool", parts: [part] }));
    }
  }
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
