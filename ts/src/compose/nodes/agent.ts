/**
 * kind=agent: the LLM-reasoner node's descriptor.
 *
 * Owns the `agent` node's `AgentDescriptor`, which wraps the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) to reject the removed `mode:` key
 * and a variable named `inf`, and checks the shapes of its own fields. Registered in `KINDS` and
 * `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, compose source and helpers, memory contracts, nodes (contracts),
 *              typesys and defaults.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import { DeclSite, MemoryDecl } from "../../memory/contracts.ts";
import { isExactly, type ValueContract } from "../../nodes/contracts.ts";
import { isList, isRecord, type Value, valueRepr } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { parseMemoryDeclarations } from "../helpers/memory.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { formerBooleanFix } from "../helpers/yaml_words.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectInfinityName,
  rejectNonText,
  siteKey,
} from "./base.ts";

/** Shared by the AGENT and `adaptive_questions:` errors for the removed `mode:` key. */
export const MODE_REMOVED_FIX =
  "an AGENT with tools runs the tool loop, one without tools makes a single call; delete this line";

/**
 * Return why `value` is not a valid `retries:` count, or `null` when it is.
 *
 * A count is a YAML int (`bigint`) from 0 to `Number.MAX_SAFE_INTEGER`, so it converts to a
 * `number` exactly; a boolean is refused, so `retries: true` doesn't mean 1.
 *
 * Example: `retriesError(-1n)` is `"must be a non-negative integer, got -1"`.
 */
export function retriesError(value: unknown): string | null {
  if (typeof value === "bigint" && value >= 0n) {
    if (value <= BigInt(Number.MAX_SAFE_INTEGER)) return null;
    return `must be at most ${Number.MAX_SAFE_INTEGER}, got ${valueRepr(value)}`;
  }
  return `must be a non-negative integer, got ${valueRepr(value)}`;
}

/**
 * Return `"; "` and the fix for a YAML 1.1 boolean spelling in a boolean field, else `""`.
 *
 * Example: `formerBooleanHint("yes")` is ``"; `yes` is text; write `true`"``.
 */
function formerBooleanHint(value: unknown): string {
  const fix = formerBooleanFix(value);
  return fix === null ? "" : `; ${fix[1]}`;
}

/** The fields `AgentDescriptor`'s constructor takes: the common ones and the AGENT's own. */
export interface AgentDescriptorFields extends NodeDescriptorFields {
  prompt?: string | null;
  systemPrompt?: string | null;
  variables?: unknown;
  tools?: readonly string[];
  skills?: readonly string[] | "all" | null;
  llmConfig?: unknown;
  retries?: number;
  asserts?: readonly string[] | null;
  steerable?: boolean;
  memories?: readonly MemoryDecl[];
  outputContractOverride?: ValueContract | null;
}

/**
 * kind=agent: an LLM reasoner (a prompt, an optional system prompt, optional tools, skills and
 * model config).
 *
 * `allowed` adds `inputs`/`outputs` and the AGENT's own keys to the commons; nothing is required.
 * Only `inputs` is a binding field: the `${...}` references of `prompt:` and `system_prompt:` are
 * read by template scanning, not the inline-call desugar. The authored `long_term_memory:` is read
 * into `memories`.
 */
export class AgentDescriptor extends NodeDescriptor {
  static override readonly kind = "agent";
  static override readonly allowed = this.allow(
    "inputs",
    "outputs",
    "prompt",
    "system_prompt",
    "variables",
    "tools",
    "skills",
    "llm_config",
    "long_term_memory",
    "retries",
    "asserts",
    "steerable",
  );
  static override readonly required = this.require();
  static override readonly bindingFields: readonly string[] = ["inputs"];
  static override readonly keywordRemap: ReadonlyMap<string, string> = new Map([["long_term_memory", "memories"]]);

  /** The prompt template; `null` when absent. */
  readonly prompt: string | null;
  /** The system prompt template; `null` when absent. */
  readonly systemPrompt: string | null;
  /** The authored `variables:` as YAML read it; `build` reads the declarations. */
  readonly variables: unknown;
  /** The granted tool names. */
  readonly tools: readonly string[];
  /** The granted skill names, or `"all"`. */
  readonly skills: readonly string[] | "all";
  /** The authored `llm_config:` as YAML read it; `build` reads the values in it. */
  readonly llmConfig: unknown;
  /** The effective retry count, so the manifest holds it and a default flow keeps its identity. */
  readonly retries: number;
  /** The node asserts. */
  readonly asserts: readonly string[];
  /** Whether the node takes steering messages while it runs. */
  readonly steerable: boolean;
  /** The authored `long_term_memory:` declarations. */
  readonly memories: readonly MemoryDecl[];
  /**
   * Never parsed from YAML: set only by the `adaptive_questions` desugar, for a synthesized AGENT
   * whose output type is built in code and has no type text; it overrides the declared `output:`.
   */
  readonly outputContractOverride: ValueContract | null;

  /**
   * Create a descriptor from `fields`. `retries` defaults to `current().agentRetries`;
   * `variables` and `llmConfig` to `{}` (a `null` is kept as written for `build`, which refuses a
   * null `variables:` and reads a null `llm_config:` as the defaults); `skills` and `asserts` read
   * `null` as empty.
   *
   * @throws TypeError when `memories` isn't an array of exact `MemoryDecl` values, or `origin`
   *   isn't an `OriginRef` or `null`.
   */
  constructor(fields: AgentDescriptorFields) {
    super(fields);
    const memories = fields.memories ?? [];
    if (!isList(memories) || memories.some((decl) => !isExactly(decl, MemoryDecl))) {
      throw new TypeError("agent descriptor memories must be an array of exact MemoryDecl values");
    }
    this.prompt = fields.prompt ?? null;
    this.systemPrompt = fields.systemPrompt ?? null;
    this.variables = fields.variables === undefined ? {} : fields.variables;
    this.tools = fields.tools ?? [];
    this.skills = fields.skills ?? [];
    this.llmConfig = fields.llmConfig === undefined ? {} : fields.llmConfig;
    this.retries = fields.retries ?? current().agentRetries;
    this.asserts = fields.asserts ?? [];
    this.steerable = fields.steerable ?? false;
    this.memories = memories;
    this.outputContractOverride = fields.outputContractOverride ?? null;
  }

  /**
   * Reject the removed `mode:` key, parse as usual, then reject a variable named `inf`, since
   * `${inf}` reads infinity.
   *
   * @throws LoadError when the node authors `mode:` (at its key) or a variable named `inf` (at
   *   its key), or the template `parse` rejects it.
   */
  static override parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    if (cursor.has("mode")) {
      let message = `node ${reprText(nodeId)}: \`mode:\` was removed: ${MODE_REMOVED_FIX}`;
      const tools = cursor.get("tools");
      if (cursor.get("mode") === "plain" && isList(tools) && tools.length > 0) {
        message += " (its `tools:` will now be used)";
      }
      throw diagnostics.error(message, { range: cursor.keyRangeOf("mode") });
    }
    const desc = super.parse(nodeId, cursor) as AgentDescriptor;
    if (isRecord(desc.variables)) {
      for (const name of Object.keys(desc.variables)) {
        rejectInfinityName(
          `node ${reprText(nodeId)} variable ${reprText(name)}`,
          name,
          cursor.child("variables").keyRangeOf(name) ?? cursor.rangeOf("variables") ?? cursor.range,
          diagnostics,
        );
      }
    }
    return desc as InstanceType<T>;
  }

  /** Reject the retired `memory_policy:` key, at its key, before the generic key check. */
  static override _normalizeCursor(cursor: SourceCursor): SourceCursor {
    if (cursor.has("memory_policy")) {
      throw LoadDiagnostics.fromCursor(cursor).error(
        "node field `memory_policy:` was renamed to `long_term_memory:`",
        { range: cursor.keyRangeOf("memory_policy") },
      );
    }
    return cursor;
  }

  /**
   * Check the `prompt:`, `system_prompt:`, `tools:`, `retries:`, `skills:` and `steerable:`
   * shapes, store `retries` as a `number`, and parse a present `long_term_memory:` into
   * `memories` (`null` is none).
   *
   * @throws LoadError at the field for a bad shape, a lone surrogate in a prompt, or a bad
   *   memory declaration.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const id = reprText(nodeId);
    const rangeOf = (key: string) => ranges.get(siteKey("field", key)) ?? ownRange;
    const prompt = fields["prompt"] ?? null;
    rejectNonText(this.kind, nodeId, "prompt", prompt, rangeOf("prompt"), diagnostics);
    if (prompt !== null) checkWrittenValue(prompt, `node ${id} field 'prompt'`, rangeOf("prompt"), diagnostics);
    if (Object.hasOwn(fields, "systemPrompt")) {
      const systemPrompt = fields["systemPrompt"];
      if (typeof systemPrompt !== "string" || !systemPrompt) {
        throw diagnostics.error(`node ${id} (kind=${this.kind}): \`system_prompt:\` must be a non-empty string`, {
          range: rangeOf("system_prompt"),
        });
      }
      checkWrittenValue(systemPrompt, `node ${id} field 'system_prompt'`, rangeOf("system_prompt"), diagnostics);
    }
    if (Object.hasOwn(fields, "tools")) {
      const tools = fields["tools"];
      if (!isList(tools) || !tools.every((name) => typeof name === "string" && name)) {
        const example = typeof tools === "string" && tools ? tools : "search";
        throw diagnostics.error(
          `node ${id} (kind=${this.kind}): \`tools:\` must be a list of tool names; write \`tools: [${example}]\``,
          { range: rangeOf("tools") },
        );
      }
    }
    if (Object.hasOwn(fields, "retries")) {
      const problem = retriesError(fields["retries"]);
      if (problem !== null) {
        throw diagnostics.error(`node ${id} (kind=${this.kind}): \`retries:\` ${problem}`, {
          range: rangeOf("retries"),
        });
      }
      fields["retries"] = Number(fields["retries"]);
    }
    const skills = fields["skills"] ?? null;
    if (
      skills !== null &&
      skills !== "all" &&
      !(isList(skills) && skills.every((name) => typeof name === "string"))
    ) {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): \`skills:\` must be a list of skill names or \`all\``,
        { range: rangeOf("skills") },
      );
    }
    const steerable = Object.hasOwn(fields, "steerable") ? fields["steerable"] : false;
    if (typeof steerable !== "boolean") {
      throw diagnostics.error(
        `node ${id} (kind=${this.kind}): \`steerable:\` must be true or false${formerBooleanHint(steerable)}`,
        { range: rangeOf("steerable") },
      );
    }
    if (Object.hasOwn(fields, "memories")) {
      const value = fields["memories"] as Value;
      fields["memories"] =
        value === null
          ? []
          : parseMemoryDeclarations(value, {
              diagnostics,
              sourceRange: ranges.get(siteKey("field", "long_term_memory")) ?? null,
              site: new DeclSite([], ["root"], nodeId),
            });
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /** Attach one origin per authored `long_term_memory:` entry: its key, or the whole sugar scalar. */
  override withSourceOrigins(
    cursor: SourceCursor,
    options: { sitePath: readonly (string | number)[]; catalogBuilder: SourceCatalogBuilder },
  ): this {
    if (this.memories.length === 0 || !cursor.has("long_term_memory")) return this;
    const { sitePath, catalogBuilder } = options;
    const valueCursor = cursor.child("long_term_memory");
    const memories = this.memories.map((decl) => {
      const path = [...sitePath, "field", "memories", decl.name];
      const origin = valueCursor.has(decl.name)
        ? catalogBuilder.register(path, valueCursor.child(decl.name), "key")
        : // The sugar form: the whole scalar value is the one entry.
          catalogBuilder.register(path, valueCursor);
      return new MemoryDecl(decl.name, decl.type, decl.config, decl.site, origin);
    });
    return this.replace({ memories });
  }
}
