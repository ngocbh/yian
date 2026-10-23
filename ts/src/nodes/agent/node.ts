/**
 * AGENT: one authored leaf invoking a model.
 *
 * It makes one call without tools or skills, and runs the tool loop with them. The loop drives
 * local model and tool calls through the shared provider interface. Memories recall notes into
 * the system prompt; they hold no conversation. Tools, including the run's `ask_user`, are
 * ordinary explicitly authored tool grants. A crash or host interruption unwinds the node; no
 * conversation or continuation state is stored in the graph or variable pool.
 *
 * Backward makes one tool-free model call over the recorded invocation (`backward.ts`).
 *
 * Knows about:   nodes/base, nodes/binding, nodes/contracts, peer AGENT modules, memory contracts,
 *                llm_clients (config, connection, factory), expr (templates, lineage), skills,
 *                defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { compareCodeUnits, type FailureCode, type OriginRef, reprText } from "../../diagnostics.ts";
import type { FeedbackValue } from "../../expr/lineage.ts";
import { renderTemplateRecord } from "../../expr/template.ts";
import type { HostConnection } from "../../llm_clients/config.ts";
import { scrubCallCredentials } from "../../llm_clients/connection.ts";
import { defaultModelAccess, type ModelAccess } from "../../llm_clients/factory.ts";
import {
  iterMemoryDeclOrigins,
  MemoryContext,
  MemoryDecl,
  MemoryRecallError,
  rewriteMemoryDeclOrigins,
} from "../../memory/contracts.ts";
import { checkSkillName, type Skill, SkillError } from "../../skills/catalog.ts";
import { LOAD_SKILL, systemPromptWithSkills } from "../../skills/tool.ts";
import { canonicalJson } from "../../typesys/json.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isList,
  isRecord,
  type Type,
  ValueKind,
  type ValueRecord,
  valueRepr,
  writeAsText,
} from "../../typesys/values.ts";
import {
  type BackwardContext,
  type BackwardOptions,
  type BackwardResult,
  CapabilityFault,
  checkNamedPrompt,
  Node,
  type NodeCaps,
  NodeKind,
  type Output,
} from "../base.ts";
import { applyDefaults, BindingError, copyValue, type ReservedBinder } from "../binding.ts";
import {
  Binding,
  isExactly,
  iterOriginCarrier,
  type NodeSteps,
  type OriginMapper,
  type OutputContract,
  Parameter,
  rewriteOriginCarrier,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../contracts.ts";
import { backwardAgent } from "./backward.ts";
import { type AgentRunContext, checkStopped, DEFAULT_SYSTEM, sentCredentials, ToolIterationLimitError } from "./context.ts";
import { modelFailureCode } from "./live.ts";
import {
  type CallTimeLLM,
  callTimeConfig,
  invocationConfig,
  LLM_CONFIG_KEY,
  type Nullable,
  typesNullable,
} from "./llm_values.ts";
import { runAgent } from "./loop.ts";
import { OutputInvalidError } from "./structured.ts";

/** The step key under which a call-time agent records the config it resolved. */
const CONFIG_STEP = "effective_llm_config";

/** Whether an AGENT output contract is plain text: none, or an untagged `str`. */
export function isTextOutput(contract: OutputContract | null): boolean {
  return (
    contract === null ||
    (isExactly(contract, ValueContract) &&
      contract.type.kind === ValueKind.STRING &&
      (contract.type.tags === undefined || contract.type.tags.size === 0))
  );
}

/** Whether `value` is a valid `max_tool_iterations`: a positive int, or `-1n` for no cap. */
export function isMaxToolIterations(value: unknown): value is bigint {
  return typeof value === "bigint" && (value >= 1n || value === -1n);
}

/** Return `make()`; its error is a capability fault, since the render capture is the host's. */
function captureValue<T>(make: () => T): T {
  try {
    return make();
  } catch (error) {
    throw error instanceof CapabilityFault ? error : new CapabilityFault(error);
  }
}

/**
 * An LLM-backed leaf configured by a prompt and authored tools.
 *
 * `systemPrompt` is an optional template that replaces the default system message; it renders
 * against the same local record as `prompt`. `memories` holds the AGENT's authored memory
 * declarations; compile derives runtime scope. `variables` keeps typed local declarations with
 * mandatory literal defaults, resolved into a fresh prompt record for each invocation.
 */
export class AgentNode extends Node {
  readonly kind = NodeKind.AGENT;
  override readonly needsEvaluationTrace: boolean = true;
  override readonly needsMemoryContext: boolean = true;
  override readonly needsRenderCapture: boolean = true;
  override readonly needsTools: boolean = true;
  override readonly needsLiveEvents: boolean = true;
  override readonly needsSkills: boolean = true;
  override readonly needsModels: boolean = true;
  /**
   * Every agent records a `model_attempt` step before each model retry; a call-time agent also
   * records and reads back its `effective_llm_config`.
   */
  override readonly needsSteps: boolean = true;
  /** The prompt template. */
  prompt: string;
  /** The system prompt template replacing the default one; `null` keeps the default. */
  systemPrompt: string | null;
  /** The authored tool grants, which may name host-filled groups. */
  tools: readonly string[];
  /**
   * The authored source each cascade pass re-resolves: a record, or a template or binding for a
   * layer holding values.
   */
  ownLlmConfig: Readonly<Record<string, unknown>> | ValueTemplate | Binding;
  /** The effective config baked for this run; `null` while a layer's values are unresolved. */
  llmConfig: Readonly<Record<string, unknown>> | null;
  /** Whether this agent's config merges over the enclosing layers. */
  llmInherit: boolean;
  /** How many times a bad structured reply may be retried. */
  retries: number;
  /** The authored memory declarations, each with an AGENT `DeclSite`. */
  memories: readonly MemoryDecl[];
  /** The local variable declarations, each with a literal default. */
  variables: readonly Parameter[];
  /** The granted skill names, or `"all"` for every skill the host supplies. */
  skills: readonly string[] | "all";
  /** Whether the host may steer or interrupt this agent while it runs. */
  readonly steerable: boolean;
  /**
   * The author-facing phrase naming this agent's `llm_config` when a desugar synthesized the
   * agent from another node's text (e.g. `node 'ask' adaptive_questions.llm_config`).
   * Render-only and absent from run records and the manifest. `null` for an authored agent,
   * whose messages name its path.
   */
  llmConfigLabel: string | null;
  /** Set by Compose: the node and own-flow layers ride the reserved `__llm_config` wiring. */
  wiresLlmConfig: boolean;
  /**
   * Set by Compose: the static type of each value in `ownLlmConfig` by field path (`null` when
   * not proven), which decides whether it always overrides a lower layer's field.
   */
  llmValueTypes: readonly (readonly [ValuePath, Type | null])[];
  /**
   * Set per run on the run's private flow, never on the authored one: what run start fixed when
   * this agent's config resolves only when it runs; `null` when run start resolved it into
   * `llmConfig`.
   */
  callTime: CallTimeLLM | null;
  /**
   * Set per run on the run's private flow, only for an agent given `callTime`: the host key and
   * header values.
   */
  hostCredentials: HostConnection | null;

  /**
   * Build an AGENT node.
   *
   * @param nodeId The node's unique id.
   * @param options.prompt The prompt template; `null` or omitted means `""`.
   * @param options.systemPrompt A non-empty system prompt template, or `null` (the default).
   * @param options.tools The tool grants; defaults to `[]`.
   * @param options.llmConfig A config record, a `ValueTemplate` or `Binding` for a layer
   *   holding values, or `null` (the default).
   * @param options.llmInherit Whether the config merges over the enclosing layers; defaults to
   *   `true`.
   * @param options.retries Bad structured reply retries; `null` or omitted means
   *   `current().agentRetries`.
   * @param options.title A display title.
   * @param options.memories Exact `MemoryDecl`s with AGENT `DeclSite`s; defaults to `[]`.
   * @param options.variables Exact `Parameter`s with literal defaults; defaults to `[]`.
   * @param options.skills Skill names, or `"all"`; defaults to `[]`.
   * @param options.steerable Whether the host may steer this agent; defaults to `false`.
   * @throws TypeError for a bad `systemPrompt`, `llmConfig`, `memories`, `variables` or
   *   `steerable`.
   * @throws RangeError for the retired `builtin_tools` grant, a memory without an AGENT
   *   `DeclSite`, or a variable whose name repeats or whose default is missing or invalid.
   * @throws SkillError for a `load_skill` tool grant or a bad skill grant.
   */
  constructor(
    nodeId: string,
    options: {
      prompt?: string | null;
      systemPrompt?: string | null;
      tools?: readonly string[] | null;
      llmConfig?: Readonly<Record<string, unknown>> | ValueTemplate | Binding | null;
      llmInherit?: boolean;
      retries?: number | null;
      title?: string | null;
      memories?: readonly MemoryDecl[];
      variables?: readonly Parameter[] | null;
      skills?: readonly string[] | "all" | null;
      steerable?: boolean;
    } = {},
  ) {
    const {
      prompt = null,
      systemPrompt = null,
      tools = null,
      llmConfig = null,
      llmInherit = true,
      retries = null,
      title = null,
      memories = [],
      variables = null,
      skills = null,
      steerable = false,
    } = options;
    super(nodeId, { title });
    this.prompt = prompt ?? "";
    if (systemPrompt !== null && (typeof systemPrompt !== "string" || systemPrompt === "")) {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: system_prompt must be a non-empty string or null`);
    }
    this.systemPrompt = systemPrompt;
    this.tools = tools ?? [];
    if (this.tools.includes("builtin_tools")) {
      throw new RangeError(
        "builtin_tools belonged to the removed native Codex integration. " +
          "Remove it and grant registered Yian tools explicitly.",
      );
    }
    // ownLlmConfig is the authored source re-resolved by each cascade pass; llmConfig is the
    // effective config baked for this run. A layer holding values (a template, or a binding for
    // the whole record) has no config until they resolve.
    if (llmConfig instanceof ValueTemplate || llmConfig instanceof Binding) {
      this.ownLlmConfig = llmConfig;
      this.llmConfig = null;
    } else if (llmConfig === null || isRecord(llmConfig)) {
      this.ownLlmConfig = { ...llmConfig };
      this.llmConfig = llmConfig;
    } else {
      throw new TypeError(
        `AGENT node ${reprText(nodeId)}: llm_config must be a record, a ValueTemplate, a Binding or null`,
      );
    }
    this.llmInherit = llmInherit;
    this.retries = retries ?? current().agentRetries;
    if (!isList(memories) || memories.some((decl) => !isExactly(decl, MemoryDecl))) {
      throw new TypeError("memories must be an array of exact MemoryDecl values");
    }
    if (memories.some((decl) => decl.site.node === null)) {
      throw new RangeError("AGENT memories must carry AGENT DeclSites");
    }
    this.memories = memories;
    this.variables = variables ?? [];
    this.skills = skills ?? [];
    this.validateSkills();
    this.validateVariables();
    if (typeof steerable !== "boolean") {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: steerable must be a bool`);
    }
    this.steerable = steerable;
    this.needsSteering = steerable;
    this.llmConfigLabel = null;
    this.wiresLlmConfig = false;
    this.llmValueTypes = [];
    this.callTime = null;
    this.hostCredentials = null;
  }

  /** Declare `__llm_config` when values in the layers can reach this agent's config. */
  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(this.wiresLlmConfig ? [LLM_CONFIG_KEY] : []);
  }

  /** Hide `__llm_config` from recall and evidence, when this agent has it: it configures the call. */
  override modelHiddenInputs(): ReadonlySet<string> {
    return new Set(this.wiresLlmConfig ? [LLM_CONFIG_KEY] : []);
  }

  /**
   * Resolve and check a call-time agent's `llm_config` before its model call.
   *
   * The node and own-flow layers resolve against `pool`, merge over the base run start recorded,
   * and are checked as run start checks a static agent's config (`callTimeConfig`). Nothing is
   * rendered, recalled or sent to a model first. `options.bind` is unused: the bound value is the
   * resolved config, not the values it was read from.
   *
   * @returns `{__llm_config: <config>}`, the resolved config with no key or header value, the
   *   shape of a `RunStarted` map entry; `{}` for an agent run start resolved.
   * @throws Error when a call-time agent's wiring has no `__llm_config` binding.
   * @throws BindingError when a value breaks its tag (`binding.type`) or fails to resolve, or the
   *   resolved config is refused (`contract.input`); the message names the field.
   * @throws AuthError when the config is marked for the host key the run does not hold; the
   *   host's to fix, not a failure of this node.
   */
  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    if (this.callTime === null) return {};
    const template = nodeWiring.get(LLM_CONFIG_KEY);
    if (!(template instanceof Binding)) {
      throw new Error(`AGENT node ${reprText(this.id)} has no \`${LLM_CONFIG_KEY}\` wiring`);
    }
    return {
      [LLM_CONFIG_KEY]: callTimeConfig(template, pool, this.callTime, {
        label: this.llmConfigLabel,
        credentials: this.hostCredentials,
      }),
    };
  }

  /** Name the reserved `__llm_config` key as the authored `llm_config` field. */
  override reservedKeyLabel(key: string): string {
    return key === LLM_CONFIG_KEY ? "llm_config" : super.reservedKeyLabel(key);
  }

  /**
   * Return whether a value in `ownLlmConfig`, or a part of it, may be null. Answers from
   * `llmValueTypes`; a value with no proven type may be null.
   */
  llmNullable(): Nullable {
    return typesNullable(this.ownLlmConfig, this.llmValueTypes);
  }

  /**
   * Return the config an earlier start of this agent recorded in its steps, if any.
   *
   * @param steps The node's recorded steps, in order.
   * @returns A copy of the last recorded `effective_llm_config`, with no key or header value;
   *   `null` when no step records one.
   */
  recordedLlmConfig(steps: readonly unknown[]): Record<string, unknown> | null {
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      const step = steps[index];
      if (isRecord(step) && Object.hasOwn(step, CONFIG_STEP) && isRecord(step[CONFIG_STEP])) {
        return copyValue(step[CONFIG_STEP]) as Record<string, unknown>;
      }
    }
    return null;
  }

  /**
   * Return the declared inputs eligible for feedback, never `__llm_config`. The model config is
   * not material the model reasons about, so the node that produced a value in it never receives
   * this agent's input feedback.
   */
  override backwardInputNames(): readonly string[] {
    return super.backwardInputNames().filter((name) => name !== LLM_CONFIG_KEY);
  }

  /**
   * Return the config a call-time agent bound as `__llm_config`, if any.
   *
   * @param inputs The invocation's bound inputs.
   * @returns The resolved config, with no key or header value; `null` for an agent run start
   *   resolved, which binds none.
   */
  override boundLlmConfig(inputs: Readonly<Record<string, unknown>>): Record<string, unknown> | null {
    const config = Object.hasOwn(inputs, LLM_CONFIG_KEY) ? inputs[LLM_CONFIG_KEY] : null;
    return isRecord(config) ? config : null;
  }

  /**
   * Reject a steerable agent whose output contract is not plain text. The contract is assigned
   * after construction, so this is checked here as well as at YAML load.
   *
   * @throws RangeError when the agent is steerable and its output isn't an untagged `str`.
   */
  override validateExecutionSetup(): void {
    super.validateExecutionSetup();
    if (this.steerable && !isTextOutput(this.outputContract)) {
      // An interrupted node returns its text so far; a structured value can't be partial.
      throw new RangeError(`AGENT node ${reprText(this.id)}: steerable needs \`str\` output`);
    }
  }

  /**
   * Check the prompt and system prompt templates.
   *
   * @param options.path The static node path to name in the error instead of the node's id, used
   *   by a pre-run check over a whole flow.
   * @throws ExpressionError when a template's `${...}` span doesn't parse, calls `isinstance` or a
   *   function that isn't a builtin, or calls a builtin with arguments it doesn't take; the
   *   message names the node and the field.
   */
  override validatePrompts(options: { path?: string | null } = {}): void {
    const { path = null } = options;
    for (const [field, text] of [
      ["prompt", this.prompt],
      ["system_prompt", this.systemPrompt],
    ] as const) {
      // `||`, as Python's `or`: an empty path names the node by its id.
      if (typeof text === "string") checkNamedPrompt(text, `node ${reprText(path || this.id)} ${field}`);
    }
  }

  /** Require `"all"` or unique valid names, and no `load_skill` tool grant. */
  private validateSkills(): void {
    const where = `AGENT node ${reprText(this.id)}`;
    if (this.tools.includes(LOAD_SKILL)) {
      throw new SkillError(`${where}: ${reprText(LOAD_SKILL)} is a reserved tool name; remove it from tools`);
    }
    if (this.skills !== "all") {
      if (!isList(this.skills)) throw new SkillError(`${where}: skills must be a list of names or 'all'`);
      for (const name of this.skills) checkSkillName(name);
      if (new Set(this.skills).size !== this.skills.length) {
        throw new SkillError(`${where}: duplicate skill grant in ${writeAsText([...this.skills])}`);
      }
    }
  }

  /** Return the authored `tools:` grants, which may name host-filled groups. */
  override grantedTools(): readonly string[] {
    return [...this.tools];
  }

  /**
   * Select this node's grants from the run's host-supplied skills, by name, sorted.
   *
   * @param supplied The run's skills by name; `null` when the host supplied none.
   * @throws SkillError when the node grants skills and `supplied` is `null` or lacks one.
   */
  grantedSkills(supplied: ReadonlyMap<string, Skill> | null): readonly Skill[] {
    if (this.skills !== "all" && this.skills.length === 0) return [];
    if (supplied === null) {
      throw new SkillError(`AGENT node ${reprText(this.id)} grants skills, but the host supplied no skills= mapping`);
    }
    let names: string[];
    if (this.skills === "all") {
      names = [...supplied.keys()].sort(compareCodeUnits);
    } else {
      const missing = this.skills.filter((name) => !supplied.has(name));
      if (missing.length > 0) {
        throw new SkillError(
          `AGENT node ${reprText(this.id)}: granted skills not supplied by the host: ${writeAsText(missing)}`,
        );
      }
      names = [...this.skills].sort(compareCodeUnits);
    }
    return names.map((name) => supplied.get(name)!);
  }

  /** Require canonical, uniquely named declarations with valid typed defaults. */
  private validateVariables(): void {
    if (!isList(this.variables)) throw new TypeError("agent variables must be a list of Parameter values");
    const names = new Set<string>();
    for (const variable of this.variables) {
      if (!isExactly(variable, Parameter)) throw new TypeError("agent variables must contain exact Parameter values");
      if (typeof variable.name !== "string" || variable.name === "") {
        throw new TypeError("variable names must be non-empty strings");
      }
      if (names.has(variable.name)) throw new RangeError(`duplicate variable name ${reprText(variable.name)}`);
      names.add(variable.name);
      if (!isExactly(variable.contract, ValueContract)) {
        throw new TypeError(`variable ${reprText(variable.name)} requires an exact ValueContract`);
      }
      if (variable.hasDefault !== true || variable.required !== false) {
        throw new RangeError(`variable ${reprText(variable.name)} requires a default and required=false`);
      }
      variable.contract.type.validateShape();
      buildValueAs(variable.contract.type, variable.default);
    }
  }

  /**
   * Yield the common origins, then the variables', the memory declarations' and those of the
   * values in `ownLlmConfig`.
   */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    yield* super.iterLocalOrigins(seen);
    for (const variable of this.variables) yield* iterOriginCarrier(variable, seen);
    yield* iterMemoryDeclOrigins(this.memories, seen);
    if (this.ownLlmConfig instanceof ValueTemplate || this.ownLlmConfig instanceof Binding) {
      yield* iterOriginCarrier(this.ownLlmConfig, seen);
    }
  }

  /**
   * Rewrite the common, variable, memory declaration and `llmConfig` value origins on a copied
   * AGENT, reassigning each field with rebuilt carriers. Carrier aliases are kept.
   */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.variables = this.variables.map((variable) => rewriteOriginCarrier(variable, mapper, seen));
    this.memories = rewriteMemoryDeclOrigins(this.memories, mapper, seen);
    if (this.ownLlmConfig instanceof ValueTemplate || this.ownLlmConfig instanceof Binding) {
      this.ownLlmConfig = rewriteOriginCarrier(this.ownLlmConfig, mapper, seen);
    }
  }

  /**
   * Return the authored `env.max_tool_iterations` or the default loop bound, or `-1` for none.
   *
   * @throws RangeError when the authored value isn't a positive int or `-1`.
   */
  maxToolIterations(): number {
    if (!Object.hasOwn(this.env, "max_tool_iterations")) return current().agentMaxToolIterations;
    const raw = this.env.max_tool_iterations;
    if (!isMaxToolIterations(raw)) {
      throw new RangeError(
        `AGENT node ${reprText(this.id)}: env max_tool_iterations must be a positive int ` +
          `or -1 (no cap), got ${valueRepr(raw)}`,
      );
    }
    return Number(raw);
  }

  /**
   * Build the complete process-local context for one invocation.
   *
   * @param prompt The rendered prompt.
   * @param config The config this invocation's model call uses, host secrets included; it lives
   *   only in the returned context.
   * @throws RangeError when the authored `env.max_tool_iterations` is invalid.
   */
  private runContext(
    prompt: string,
    config: Readonly<Record<string, unknown>> | null,
    caps: NodeCaps,
    options: {
      memoryContext: MemoryContext;
      baseSystemPrompt: string;
      skills: readonly Skill[];
      models: ModelAccess;
    },
  ): AgentRunContext {
    return {
      nodeId: this.id,
      prompt,
      tools: [...this.tools],
      llmConfig: config,
      outputType: isExactly(this.outputContract, ValueContract) ? this.outputContract.type : null,
      retries: this.retries,
      maxToolIterations: this.maxToolIterations(),
      signal: caps.signal,
      evaluationTrace: caps.evaluationTrace,
      memoryContext: options.memoryContext,
      toolset: caps.tools ?? null,
      emitStream: caps.emitStream,
      emitEvent: caps.emitEvent,
      baseSystemPrompt: options.baseSystemPrompt,
      skills: options.skills,
      steering: caps.steering,
      models: options.models,
      sent: () => [],
      steps: caps.steps,
      scheduler: caps.scheduler,
      random: caps.random,
      runDir: caps.runDir ?? null,
    };
  }

  /**
   * Select a generated structured-output contract origin when proven.
   *
   * A memory recall or variable-default binding failure keeps its own origin. A structured reply
   * that failed the value check maps to its output contract's origin at the error's path (exact
   * or nearest ancestor). Other failures, and contracts without authored provenance, keep the
   * common node-origin fallback.
   */
  override exceptionOrigin(error: unknown): OriginRef | null {
    if (error instanceof MemoryRecallError || error instanceof BindingError) {
      return error.origin ?? super.exceptionOrigin(error);
    }
    const contract = this.outputContract;
    if (error instanceof OutputInvalidError && error.path !== null && isExactly(contract, ValueContract)) {
      return contract.origins.at(error.path) ?? super.exceptionOrigin(error);
    }
    return super.exceptionOrigin(error);
  }

  /**
   * Classify the tool-iteration cap, an exhausted structured reply and the agent's own model
   * failures; defer the rest.
   *
   * The cap is `agent.max_tool_iterations` and a structured reply still invalid once `retries:`
   * are spent is `agent.output_invalid`. An error the agent's own model build or model call
   * marked takes `modelFailureCode`'s code (`provider.auth` for refused credentials, a
   * `provider.*` code by provider error kind, else `node.execute`).
   */
  override exceptionCode(error: unknown): FailureCode {
    if (error instanceof ToolIterationLimitError) return "agent.max_tool_iterations";
    if (error instanceof OutputInvalidError) return "agent.output_invalid";
    return modelFailureCode(error) ?? super.exceptionCode(error);
  }

  /**
   * Render fresh locals once, publish the render capture, and run the loop.
   *
   * The authored `systemPrompt` renders against the same inputs and variables as `prompt`;
   * without one the loop receives `DEFAULT_SYSTEM`. `caps.captureRender` receives the actual
   * template, the resolved variable values and domains, and the rendered prompt once, with the
   * parts established so far when preparation fails; unknown variables or an unfinished prompt
   * stay `null`, and capture faults propagate. No invocation state is stored on the node.
   *
   * A call-time agent's model config is its bound `__llm_config` with the run's host key and
   * header values joined when it is marked for the host key; any other agent uses its run's
   * `llmConfig`. Before anything renders, a call-time agent checks the bound config against the
   * one an earlier start recorded in its steps, then records it as a step,
   * `{effective_llm_config: <config>}`, which holds no key or header value. Each model retry is
   * first recorded as a `model_attempt` step; a later start restarts the agent and ignores them.
   *
   * The model is built through `caps.models`, else `defaultModelAccess()`. Every error from the
   * loop has the credentials the call sent scrubbed before it leaves.
   *
   * @param inputs Delivered inputs, kept separate from local variables.
   * @param caps The node's signal, checked before recall and the model call, its scheduler and
   *   random draw for model-retry waits, and the capabilities its traits ask for: evidence, memory
   *   context, render capture, tools, live sinks, skills, steering, model access and steps.
   * @returns The model's answer.
   * @throws RangeError when a variable collides with a declared or delivered input.
   * @throws ExpressionError when the prompt or system prompt names a missing value or renders a
   *   null.
   * @throws SkillError when the node grants skills the host did not supply.
   * @throws CapabilityFault when capture or memory access fails, or the bound config
   *   differs from the one an earlier start recorded; the evaluator rethrows its original host
   *   error.
   * @throws The signal's reason once the host stopped the node.
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output> {
    if (caps.steps !== undefined && this.callTime !== null) {
      this.recordConfig(inputs[LLM_CONFIG_KEY] as Record<string, unknown>, caps.steps);
    }
    const captured: Record<string, unknown> | null =
      caps.captureRender === undefined
        ? null
        : { prompt_template: this.prompt, variables: null, variable_domains: null, rendered_prompt: null };
    let prompt: string;
    let systemPrompt: string;
    try {
      const inputNames = new Set([...Object.keys(inputs), ...(this.params ?? []).map((parameter) => parameter.name)]);
      for (const variable of this.variables) {
        if (inputNames.has(variable.name)) {
          throw new RangeError(`variable ${reprText(variable.name)} collides with an input name`);
        }
      }
      if (captured !== null) {
        captured["variable_domains"] = captureValue(() =>
          Object.fromEntries(
            this.variables.map((variable) => {
              const contract = variable.contract as ValueContract;
              return [variable.name, canonicalTypeLabel(contract.type)];
            }),
          ),
        );
      }
      const variables = this.resolveVariables();
      if (captured !== null) {
        captured["variables"] = captureValue(() => copyValue(variables));
      }
      const record = { ...inputs, ...variables } as ValueRecord;
      prompt = renderTemplateRecord(this.prompt, record);
      if (captured !== null) captured["rendered_prompt"] = prompt;
      systemPrompt = this.systemPrompt === null ? DEFAULT_SYSTEM : renderTemplateRecord(this.systemPrompt, record);
    } finally {
      if (captured !== null) caps.captureRender?.(captured);
    }
    checkStopped(caps);
    const granted = this.grantedSkills(caps.skills ?? null);
    const baseSystemPrompt = systemPromptWithSkills(systemPrompt, granted);
    const config = this.callConfig(inputs);
    const models = caps.models ?? defaultModelAccess();
    const env = models.env;
    // Recall sees the system prompt without the skill listing, so grants never change recall.
    const memoryContext =
      caps.openMemoryContext === undefined
        ? new MemoryContext()
        : await caps.openMemoryContext(systemPrompt, prompt);
    checkStopped(caps);
    const ctx = this.runContext(prompt, config, caps, { memoryContext, baseSystemPrompt, skills: granted, models });
    try {
      return await runAgent(ctx);
    } catch (error) {
      // Providers may echo any credential the call sent; the failure record must not carry it.
      throw scrubCallCredentials(error, config, { env, sent: sentCredentials(ctx) });
    }
  }

  /**
   * Check `config` against the config an earlier start recorded, then record it.
   *
   * @param config The bound config, with no secrets.
   * @param steps This start's steps.
   * @throws CapabilityFault carrying a RangeError when an earlier start recorded another config:
   *   the code that resolves it has changed since then, which is the host's to fix, not a failure
   *   of this node.
   */
  private recordConfig(config: Readonly<Record<string, unknown>>, steps: NodeSteps): void {
    const recorded = this.recordedLlmConfig(steps.prior);
    // By value: a recorded config comes back from the run log as a new record.
    if (recorded !== null && canonicalJson(recorded) !== canonicalJson(config)) {
      throw new CapabilityFault(
        new RangeError(
          `node ${reprText(this.callTime!.path)} llm_config: recorded config differs from the ` +
            "recomputed one; resume the run with the flow and yian version that started it",
        ),
      );
    }
    steps.record({ [CONFIG_STEP]: copyValue(config as ValueRecord) });
  }

  /**
   * Return the config this invocation's model call uses; never stored.
   *
   * @param inputs The delivered inputs; a call-time agent's hold its resolved `__llm_config`.
   * @returns For a call-time agent, a new config with the host secrets joined when it is marked;
   *   otherwise the run's `llmConfig`.
   */
  private callConfig(inputs: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> | null {
    if (this.callTime === null) return this.llmConfig;
    return invocationConfig(inputs[LLM_CONFIG_KEY] as Record<string, unknown>, this.hostCredentials);
  }

  /** Resolve the validated literal defaults into one detached invocation record. */
  private resolveVariables(): ValueRecord {
    return copyValue(applyDefaults(this.variables, {}) as ValueRecord) as ValueRecord;
  }

  /**
   * Return the host key and header values bound to this node for the run, if any.
   *
   * @param recorded The recorded config of the invocation backward revisits. A call-time agent
   *   gets the host's secrets only when it is marked (`invocationConfig`); any other agent, its
   *   own `llmConfig`'s `api_key` and `headers`.
   * @returns The key and the header values, each `null` when unbound.
   */
  private hostSecrets(recorded: unknown): { apiKey: string | null; headers: Readonly<Record<string, string>> | null } {
    let config: Readonly<Record<string, unknown>> | null = this.llmConfig;
    // Through `invocationConfig`, so the host-key marker rule lives in one place.
    if (this.callTime !== null) config = isRecord(recorded) ? invocationConfig(recorded, this.hostCredentials) : null;
    if (config === null) return { apiKey: null, headers: null };
    return {
      apiKey: (config["api_key"] as string | undefined) ?? null,
      headers: (config["headers"] as Record<string, string> | undefined) ?? null,
    };
  }

  /**
   * Use only recorded invocation material for one tool-free backward call (`backwardAgent`).
   *
   * Live node fields are unused, except a host `api_key` and `headers` bound for the run.
   *
   * @param context Accepted context, including the recorded model configuration and the
   *   render/evidence components.
   * @param feedback Ordered ordinary output messages. Empty effective feedback skips model
   *   construction.
   * @param options The model access (default `defaultModelAccess()`) and the observer for the
   *   call's retry and replay events.
   * @returns Independent input, self, and variable channels for common result validation by the
   *   caller before upstream routing.
   * @throws TypeError when the context, feedback, captured fields, or response channels are
   *   malformed.
   * @throws RangeError when the feedback shape, a required capture, or the response is invalid.
   *   Model and host errors propagate, model errors with every credential the call sent scrubbed.
   */
  override async backward(
    context: BackwardContext,
    feedback: readonly FeedbackValue[],
    options: BackwardOptions = {},
  ): Promise<BackwardResult> {
    // The only live fields: a host key and its header values, which are never recorded.
    const { apiKey, headers } = this.hostSecrets(context.nodeData["effective_llm_config"]);
    return backwardAgent(context, feedback, { ...options, apiKey, headers });
  }

  /** Return the canonical local declarations in their authored order. */
  override declaredVariables(): readonly Parameter[] {
    return [...this.variables];
  }
}
