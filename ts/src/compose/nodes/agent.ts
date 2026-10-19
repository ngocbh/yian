/**
 * kind=agent: the LLM-reasoner node's descriptor.
 *
 * Owns the `agent` node's `AgentDescriptor`, which wraps the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) to reject the removed `mode:` key
 * and a variable named `inf`, checks the shapes of its own fields, and builds the runtime
 * `AgentNode`. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, compose source and helpers, llm_clients (config and provider
 *              blocks), memory contracts, nodes (agent, contracts and human_input), skills,
 *              typesys and defaults.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import { hostOnlyFieldError, LLMConfig } from "../../llm_clients/config.ts";
import { ValidationError } from "../../llm_clients/provider_blocks.ts";
import { DeclSite, MemoryDecl } from "../../memory/contracts.ts";
import { LLM_CONFIG_KEY } from "../../nodes/agent/llm_values.ts";
import { AgentNode, isMaxToolIterations, isTextOutput } from "../../nodes/agent/node.ts";
import { typeToSchema } from "../../nodes/agent/structured.ts";
import { Binding, isExactly, Parameter, ValueContract, ValueTemplate } from "../../nodes/contracts.ts";
import { siteLabel } from "../../nodes/human_input/node.ts";
import { SkillError } from "../../skills/index.ts";
import { isList, isRecord, typeLevels, type Value, valueRepr } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { parseMemoryDeclarations } from "../helpers/memory.ts";
import { readVariables } from "../helpers/types.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { formerBooleanFix } from "../helpers/yaml_words.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  notTextError,
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

/** Put `prefix` in place of the leading `llm_config` of a host-only field message. */
function hostOnlyMessage(prefix: string, message: string): string {
  // A message that does not start with `llm_config` is kept whole after the prefix.
  return message.startsWith("llm_config") ? prefix + message.slice("llm_config".length) : `${prefix}: ${message}`;
}

/** Return the field a `hostOnlyFieldError` message names (`llm_config.<field> ...`). */
function hostOnlyField(message: string): string {
  const rest = message.startsWith("llm_config.") ? message.slice("llm_config.".length) : message;
  return rest.split(" ", 1)[0]!;
}

/**
 * Return the field and value of an `LLMConfig` error that is a text field read as non-text, or
 * `null`.
 *
 * Only the first issue is read, so the field named agrees with the range the caller locates from
 * that same first issue. It is `null` when that issue is not a `string_type` failure, or is the
 * `str` member of a union field, which takes more than text. Issues carry no input, so the value
 * is read from `cfg` along the issue's path.
 */
function notTextDetail(error: ValidationError, cfg: Readonly<Record<string, unknown>>): [string, unknown] | null {
  const first = error.issues[0];
  if (first === undefined || first.type !== "string_type") return null;
  const { loc } = first;
  if (loc.at(-1) === "str") {
    // A union reports one issue per member, the `str` member first; any other issue under the same
    // parent (a list member's item issue extends it) marks a field that takes more than text.
    const parent = loc.slice(0, -1);
    if (error.issues.some((issue) => issue !== first && parent.every((step, i) => issue.loc[i] === step))) {
      return null;
    }
  }
  let value: unknown = cfg;
  for (const step of loc) {
    // Union member labels and variant tags are steps of the path but not keys of the value.
    if (typeof step === "number") {
      if (isList(value)) value = value[step];
    } else if (isRecord(value) && Object.hasOwn(value, step)) {
      value = value[step];
    }
  }
  return [siteLabel(loc), value];
}

/**
 * Check literal `llm_config` fields as `LLMConfig` fields.
 *
 * @param cfg The literal fields; a field holding a value is `null`.
 * @param prefix Message prefix naming the layer.
 * @param sourceRange The layer's location, when retained.
 * @param fieldRange The location of a top-level field by its key; `null` locates every error at
 *   the layer.
 * @throws LoadError when a field violates `LLMConfig`: a text field YAML read as non-text is named
 *   by `notTextError`, any other violation keeps the `LLMConfig` message.
 */
function checkLiteralFields(
  cfg: Readonly<Record<string, unknown>>,
  prefix: string,
  sourceRange: SourceRange | null,
  diagnostics: LoadDiagnostics,
  fieldRange: ((key: string) => SourceRange | null) | null = null,
): void {
  try {
    new LLMConfig(cfg);
  } catch (error) {
    if (error instanceof ValidationError) {
      const detail = notTextDetail(error, cfg);
      const key = error.issues[0]?.loc[0];
      const range = fieldRange !== null && typeof key === "string" ? fieldRange(key) : sourceRange;
      throw diagnostics.error(`${prefix}: ` + (detail !== null ? notTextError(...detail) : error.message), {
        range,
        cause: error,
      });
    }
    if (!(error instanceof RangeError)) throw error;
    throw diagnostics.error(`${prefix}: ${error.message}`, { range: sourceRange, cause: error });
  }
}

/**
 * Validate an authored `llm_config:` layer at load and split off `inherit`.
 *
 * `inherit` (the whole-node cascade opt-out, default `true`) is not an `LLMConfig` field, so it is
 * removed before the check that refuses unknown keys (such as `temparature`). A layer holding
 * `${...}` values (a `ValueTemplate`, or a `Binding` for the whole record) is checked by
 * `checkLlmConfigValue`; its values are checked when they resolve.
 *
 * @param cfg The authored layer, or `null` for defaults.
 * @param prefix Message prefix naming the layer, such as `node 'a': llm_config`.
 * @param sourceRange The layer's location, when retained.
 * @param fieldRanges For a layer holding values, the value range of each key written in it, by
 *   dotted path.
 * @returns The layer without `inherit`, then `inherit`.
 * @throws LoadError when the layer is not a mapping, sets a non-boolean `inherit`, sets a
 *   host-only field (`api_key`, `organization`, `project`, `headers`, or `base_url` other than
 *   beside `provider: azure`), or violates `LLMConfig`.
 */
function validateLlmConfig(
  cfg: unknown,
  prefix: string,
  sourceRange: SourceRange | null,
  diagnostics: LoadDiagnostics,
  fieldRanges: ReadonlyMap<string, SourceRange>,
): [Readonly<Record<string, unknown>> | ValueTemplate | Binding, boolean] {
  if (cfg === null) return [{}, true];
  if (cfg instanceof ValueTemplate || cfg instanceof Binding) {
    throw new Error("check_llm_config_value is not ported yet");
  }
  if (!isRecord(cfg)) throw diagnostics.error(`${prefix} must be a mapping`, { range: sourceRange });
  const { inherit = true, ...rest } = cfg;
  if (typeof inherit !== "boolean") {
    throw diagnostics.error(
      `${prefix} \`inherit:\` must be true or false, got ${valueRepr(inherit)}${formerBooleanHint(inherit)}`,
      { range: sourceRange },
    );
  }
  const hostOnly = hostOnlyFieldError(rest);
  if (hostOnly !== null) throw diagnostics.error(hostOnlyMessage(prefix, hostOnly), { range: sourceRange });
  checkLiteralFields(rest, prefix, sourceRange, diagnostics);
  return [rest, inherit];
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

  /**
   * Build the runtime `AgentNode` and its flow-owned wiring.
   *
   * Reads the `llm_config:` layer and the local variables, stamps each memory's site with
   * `ctx.owner`, sets the node's `params` from the authored `input:` and its output contract from
   * the override or the authored `output:`. `_finish` stamps the descriptor origin; the result
   * also ranges `__llm_config` at the authored `llm_config:`, where errors about its values point.
   *
   * @throws LoadError for a bad `env.max_tool_iterations`, a bad variable declaration, a tool or
   *   skill grant the node refuses, an `output:` nesting deeper than
   *   `current().maxAgentOutputNesting` or holding a type a model can't produce, `steerable` with
   *   non-text output, a variable named like an input, or an input named `__llm_config`.
   * @throws TypeError when `ctx.variableDeclarationReader` returns anything but an array of exact
   *   `Parameter`s or `null`.
   */
  override build(ctx: BuildContext): BuildResult {
    const id = reprText(this.id);
    const llmConfigRange = this.fieldRanges.get(siteKey("field", "llm_config")) ?? this.range;
    const llmFieldRanges = new Map<string, SourceRange>();
    const llmPrefix = siteKey("llm_config", "");
    for (const [key, range] of this.fieldRanges) {
      if (key.startsWith(llmPrefix)) llmFieldRanges.set(key.slice(llmPrefix.length), range);
    }
    const [ownCfg, inherit] = validateLlmConfig(
      this.llmConfig,
      this.fieldLabels.get("llm_config") || `node ${id}: llm_config`,
      llmConfigRange,
      ctx.diagnostics,
      llmFieldRanges,
    );
    const steerableRange = this.fieldRanges.get(siteKey("field", "steerable")) ?? this.range;
    if (Object.hasOwn(this.env, "max_tool_iterations") && !isMaxToolIterations(this.env["max_tool_iterations"])) {
      throw ctx.diagnostics.error(
        `node ${id}: \`env.max_tool_iterations\` must be a positive int or -1 (no cap), ` +
          `got ${valueRepr(this.env["max_tool_iterations"])}`,
        { range: this.fieldRanges.get(siteKey("field", "env")) ?? this.range },
      );
    }
    const variables = this.readVariables(ctx);
    const memories = this.memories.map(
      (decl) => new MemoryDecl(decl.name, decl.type, decl.config, new DeclSite([], ctx.owner, this.id), decl.origin),
    );
    let node: AgentNode;
    try {
      node = new AgentNode(this.id, {
        prompt: this.prompt ?? "",
        systemPrompt: this.systemPrompt,
        tools: [...this.tools],
        llmConfig: ownCfg,
        llmInherit: inherit,
        retries: this.retries,
        title: this.nodeName,
        memories,
        variables,
        skills: this.skills,
        steerable: this.steerable,
      });
    } catch (error) {
      // AgentNode checks skill grants and tool names; a refusal is the loader's one error type.
      if (!(error instanceof RangeError)) throw error;
      const skillsRange = this.fieldRanges.get(siteKey("field", "skills")) ?? this.range;
      const toolsRange = this.fieldRanges.get(siteKey("field", "tools")) ?? this.range;
      throw ctx.diagnostics.error(`node ${id}: ${error.message}`, {
        range: error instanceof SkillError ? skillsRange : toolsRange,
        cause: error,
      });
    }
    node.llmConfigLabel = this.fieldLabels.get("llm_config") ?? null;

    const override = this.outputContractOverride;
    node.outputContract = override ?? this.outputContract(ctx);
    if (isExactly(node.outputContract, ValueContract)) {
      const outputsRange = this.fieldRanges.get(siteKey("field", "outputs")) ?? this.range;
      // Schema generation, here and at run, recurses per level; deeper types aren't proven to
      // fit the stack. An engine-built output type is not the author's.
      const levels = typeLevels(node.outputContract.type);
      const limit = current().maxAgentOutputNesting;
      if (override === null && levels > limit) {
        throw ctx.diagnostics.error(
          `node ${id}: this AGENT's \`output:\` nests ${levels} levels, deeper than ${limit}, the limit ` +
            "for a model's structured reply (each record and list is a level); flatten it, or build the " +
            "deeper value in a CODE node",
          { range: outputsRange },
        );
      }
      try {
        typeToSchema(node.outputContract.type);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw ctx.diagnostics.error(
          `node ${id}: an AGENT can't produce this \`output:\`; a model generates only str, int, float, ` +
            "bool, date, datetime, Literal, record, union, object, Any, and list values, not `file` or " +
            "`None`. Produce the value in a CODE node.",
          { range: outputsRange, cause: error },
        );
      }
    }
    if (this.steerable && !isTextOutput(node.outputContract)) {
      // Checked here too so the error points at the `steerable:` line.
      throw ctx.diagnostics.error(`node ${id}: steerable needs \`str\` output`, { range: steerableRange });
    }
    node.params = this.sinkParams();
    const inputNames = new Set(node.params.map((parameter) => parameter.name));
    for (const variable of variables) {
      if (!inputNames.has(variable.name)) continue;
      const frame =
        ctx.resolveOrigin !== null && variable.origin !== null ? ctx.resolveOrigin(variable.origin) : null;
      const message = `node ${id}: variable ${reprText(variable.name)} collides with an input name`;
      if (frame !== null) throw ctx.diagnostics.error(message, { frames: [frame] });
      throw ctx.diagnostics.error(message, {
        range: this.fieldRanges.get(siteKey("field", "variables")) ?? this.range,
      });
    }
    if (inputNames.has(LLM_CONFIG_KEY)) {
      throw ctx.diagnostics.error(
        `node ${id}: input ${reprText(LLM_CONFIG_KEY)} is reserved for the values of \`llm_config\`; ` +
          "rename the input",
        { range: this.fieldRanges.get(siteKey("input", LLM_CONFIG_KEY)) ?? this.range },
      );
    }
    const result = this._finish(node, this.sinkWiring());
    // Errors about `llm_config` values point at the authored `llm_config:`.
    const fieldRanges = new Map(result.fieldRanges);
    if (llmConfigRange !== null) fieldRanges.set(siteKey("field", LLM_CONFIG_KEY), llmConfigRange);
    return new BuildResult({ ...result, fieldRanges });
  }

  /**
   * Compile this node's variable declarations at their authored site, through
   * `ctx.variableDeclarationReader`, or source-free when it declines or is unset.
   *
   * @throws LoadError for a bad declaration, ranged at `variables:` when the error has no frame.
   * @throws TypeError when the reader returns anything but an array of exact `Parameter`s or
   *   `null`.
   */
  private readVariables(ctx: BuildContext): Parameter[] {
    try {
      if (ctx.variableDeclarationReader !== null) {
        const variables: unknown = ctx.variableDeclarationReader(this.id, this.variables);
        if (variables !== null) {
          if (!isList(variables) || !variables.every((variable) => isExactly(variable, Parameter))) {
            throw new TypeError("variable declaration reader must return a list of exact Parameters or null");
          }
          return variables;
        }
      }
      // Only a node with no authored source gets here; a variable needs a default, which only a
      // tagged YAML declaration carries, so any declaration fails.
      return readVariables(this.variables, ctx.registry, { diagnostics: ctx.diagnostics });
    } catch (error) {
      if (!(error instanceof LoadError) || error.diagnostic.frames.length > 0) throw error;
      throw ctx.diagnostics.prepend(error, {
        range: this.fieldRanges.get(siteKey("field", "variables")) ?? this.range,
        cause: error,
      });
    }
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
