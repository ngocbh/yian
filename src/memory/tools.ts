/**
 * A memory's granted tools as `FunctionTool`s over its run session.
 *
 * A read tool calls `session.call`; a write tool's call is a change, `apply`d with the call's id as
 * the change id, so a call sent again after a crash applies once. Every memory tool is resumable.
 * A session's outcomes and errors map to tool errors: a refusal is a permanent reported error the
 * model reads, a busy memory a reported one it may try again, and a reused change id a permanent
 * error. A transient provider error, such as an embedder's rate limit, is a reported error the model
 * may try again.
 *
 * It imports `tools`, `llm` (its errors), `typesys` and `diagnostics` only. The caller passes the author's kind and
 * static name, since it can't read node ids here.
 */

import { reprText } from "../diagnostics.ts";
import { ProviderError } from "../llm/errors.ts";
import type { ToolContext } from "../tools/context.ts";
import { PermanentToolError, ReportedToolError } from "../tools/errors.ts";
import { FunctionTool } from "../tools/function_tools.ts";
import { YianRangeError } from "../typesys/errors.ts";
import type { JsonObject } from "../typesys/json.ts";
import type { RecordedMemoryTool } from "../typesys/memory.ts";
import { fromHost, type MemoryRef, writeAsText } from "../typesys/values.ts";
import { type Applied, type Author, memoryErrorKind, type MemorySession } from "./base.ts";

/** @internal What a node's memory tools are, as a grant scope's messages name them. */
export const MEMORY_SCOPE_LABEL = "this node's memories";

/**
 * @internal Return a FunctionTool per tool `grant` grants of the memory `value`, in the value's
 * spec order, each over `session` and built from its recorded spec.
 *
 * @param grant The memory input the tools are reached through, and the tools it grants (`null`
 *   for all of them).
 * @param author The calling node's kind and static name; a change's author adds the call's run and
 *   expanded node.
 * @throws RangeError when `grant` names a tool the value doesn't have.
 */
export function memoryTools(
  value: MemoryRef,
  session: MemorySession,
  grant: { readonly input: string; readonly tools: readonly string[] | null },
  author: { readonly kind: "agent" | "tool"; readonly name: string },
): FunctionTool[] {
  return grantedMemorySpecs(value, grant).map((spec) => memoryTool(spec, session, grant.input, author));
}

/**
 * @internal Return the recorded specs of the tools `grant` grants of the memory `value`, in the
 * value's spec order: what `memoryTools` builds its tools from, and what a rebuilt request lists.
 *
 * @throws RangeError when `grant` names a tool the value doesn't have.
 */
export function grantedMemorySpecs(
  value: MemoryRef,
  grant: { readonly input: string; readonly tools: readonly string[] | null },
): readonly RecordedMemoryTool[] {
  const { input, tools } = grant;
  const missing = tools?.find((name) => !value.tools.some((tool) => tool.name === name));
  if (missing !== undefined) {
    const has = value.tools.map((tool) => tool.name).join(", ");
    throw new YianRangeError(`\`${input}\` has no tool \`${missing}\`; it has ${has}`);
  }
  return tools === null ? value.tools : value.tools.filter((tool) => tools.includes(tool.name));
}

/** Return the FunctionTool of one recorded memory tool `spec`. */
function memoryTool(
  spec: RecordedMemoryTool,
  session: MemorySession,
  input: string,
  author: { readonly kind: "agent" | "tool"; readonly name: string },
): FunctionTool {
  const { name, write } = spec;
  const execute = async (args: Record<string, unknown>, ctx: ToolContext): Promise<unknown> => {
    if (!write) {
      try {
        return await session.call(name, args as JsonObject, { signal: ctx.signal });
      } catch (error) {
        throw sessionError(input, ctx, error, name);
      }
    }
    // The tool's name goes last, so an argument can't change which change it is.
    const change = { ...args, op: name } as JsonObject;
    let applied: Applied;
    try {
      applied = await session.apply({ id: ctx.callId, change, author: callAuthor(author, ctx), signal: ctx.signal });
    } catch (error) {
      throw sessionError(input, ctx, error);
    }
    if (!applied.ok) throw new ReportedToolError(refusalText(input, applied), { permanent: true });
    return applied.result;
  };
  return new FunctionTool({
    name,
    description: spec.description,
    parametersJsonSchema: spec.parameters,
    // Reads may run beside other calls; writes run alone, in call order.
    isConcurrencySafe: write ? null : () => true,
    resumable: true,
    execute,
  });
}

/** Return a change's author: the caller's kind and name, and the call's run and node when it has them. */
function callAuthor(author: { readonly kind: "agent" | "tool"; readonly name: string }, ctx: ToolContext): Author {
  const made: Author = { kind: author.kind, name: author.name };
  if (ctx.runId !== "") made.run = ctx.runId;
  if (ctx.node !== "") made.node = ctx.node;
  return made;
}

/**
 * Return the error a call reports for a session's `error`: a busy memory, or a provider error
 * another attempt may fix (an embedder's rate limit, server error or timeout), as a reported error
 * the model may try again; a write's reused change id as a permanent error; and any other error as
 * it is.
 *
 * @param read The read tool's name for a read, which applies no change, so a reused change id from
 *   it is a fault of the memory; `null` for a write.
 */
function sessionError(input: string, ctx: ToolContext, error: unknown, read: string | null = null): unknown {
  const kind = memoryErrorKind(error);
  if (kind === "busy") return new ReportedToolError(`\`${input}\` is busy; try again`, { cause: error });
  // A transient fault of the memory's own model doesn't fail the run; one another attempt can't
  // fix (refused credentials) stays a fault, so the host sees it.
  if (error instanceof ProviderError && error.retryable) {
    return new ReportedToolError(`\`${input}\` couldn't embed: ${error.kind}; try again`, { cause: error });
  }
  if (kind === "change_id_reused" && read !== null) {
    return new Error(`\`${input}\`'s ${read} tool reported a reused change id, which a read can't have`, { cause: error });
  }
  if (kind === "change_id_reused") {
    // Named from the call, not the error: a host's error may say anything.
    return new PermanentToolError(`\`${input}\`: change id \`${ctx.callId}\` was used for another change`, {
      cause: error,
    });
  }
  return error;
}

/** Return the text the caller reads for a refused change: its reason, then the current state, if any. */
function refusalText(input: string, refused: Extract<Applied, { ok: false }>): string {
  if (refused.current === null) return refused.reason;
  const current = writeAsText(fromHost(refused.current, `memory ${reprText(input)} current state`));
  return `${refused.reason}\ncurrent: ${current}`;
}
