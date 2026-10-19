/**
 * AGENT: one authored leaf invoking a model.
 *
 * It makes one call without tools or skills, and runs the tool loop with them. The loop drives
 * local model and tool calls through the shared provider interface. An optional memory backend
 * owns conversation admission and retained history; the evaluator owns the turn's lifetime.
 * Tools, including the run's `ask_user`, are ordinary explicitly authored tool grants. A crash or
 * host interruption unwinds the node; no conversation or continuation state is stored in the
 * graph or variable pool.
 *
 * Here: the node's construction and its checks, its load-time hooks (reserved `__llm_config`
 * wiring, prompt and setup checks, origins), the granted tools and skills, the tool-loop bound and
 * the local variable declarations. Not ported yet: `forward`, which brings `exceptionOrigin`
 * and `exceptionCode`, and `backward`.
 *
 * Knows about:   nodes/base, nodes/binding, nodes/contracts, nodes/agent/llm_values, memory contracts,
 *                llm_clients config, skills, defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { compareCodeUnits, type OriginRef, reprText } from "../../diagnostics.ts";
import type { HostConnection } from "../../llm_clients/config.ts";
import { iterMemoryDeclOrigins, MemoryDecl, rewriteMemoryDeclOrigins } from "../../memory/contracts.ts";
import { checkSkillName, type Skill, SkillError } from "../../skills/catalog.ts";
import { LOAD_SKILL } from "../../skills/tool.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import {
  buildValueAs,
  isList,
  isRecord,
  type Type,
  ValueKind,
  valueRepr,
  writeAsText,
} from "../../typesys/values.ts";
import { checkNamedPrompt, Node, NodeKind } from "../base.ts";
import { copyValue, type ReservedBinder } from "../binding.ts";
import {
  Binding,
  isExactly,
  iterOriginCarrier,
  type OriginMapper,
  type OutputContract,
  Parameter,
  rewriteOriginCarrier,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../contracts.ts";
import { type CallTimeLLM, callTimeConfig, LLM_CONFIG_KEY, type Nullable, typesNullable } from "./llm_values.ts";

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
  /** Receive node steps only when this agent's config resolves when it runs; follows `callTime`. */
  override needsSteps: boolean = false;
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
  /** The field behind `callTime`, an own field so a shallow node copy carries it. */
  private callTimeValue: CallTimeLLM | null = null;
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
   * @throws Error when a call-time agent's wiring has no `__llm_config` source.
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
    if (!nodeWiring.has(LLM_CONFIG_KEY)) {
      throw new Error(`AGENT node ${reprText(this.id)} has no \`${LLM_CONFIG_KEY}\` wiring`);
    }
    return {
      [LLM_CONFIG_KEY]: callTimeConfig(nodeWiring.get(LLM_CONFIG_KEY) as Binding, pool, this.callTime, {
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
   * Set per run on the run's private flow, never on the authored one: what run start fixed when
   * this agent's config resolves only when it runs; `null` when run start resolved it into
   * `llmConfig`. Setting it also sets `needsSteps`. It is an accessor because the base declares
   * `needsSteps` as a field, which a getter can't override, so the setter keeps that field in step.
   */
  get callTime(): CallTimeLLM | null {
    return this.callTimeValue;
  }

  set callTime(value: CallTimeLLM | null) {
    this.callTimeValue = value;
    this.needsSteps = value !== null;
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

  /** The model call or tool loop; not ported yet. */
  override forward(): never {
    throw new Error(`node ${reprText(this.id)} (AgentNode) forward is not ported yet (T5-11a)`);
  }

  /** The tool-free backward call; not ported yet. */
  override backward(): never {
    throw new Error(`node ${reprText(this.id)} (AgentNode) backward is not ported yet (T6-6)`);
  }

  /** Return the canonical local declarations in their authored order. */
  override declaredVariables(): readonly Parameter[] {
    return [...this.variables];
  }
}
