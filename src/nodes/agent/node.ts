/**
 * AGENT: one authored leaf invoking a model.
 *
 * It makes one call without tools or skills, and runs the tool loop with them. The loop drives
 * local model and tool calls through the shared provider interface. Tools, including the run's
 * `ask_user`, are ordinary explicitly authored tool grants. A crash or host interruption unwinds
 * the node; no conversation or continuation state is stored in the graph or variable pool.
 *
 * Knows about:   nodes/base, nodes/binding, nodes/contracts, nodes/ids, peer AGENT modules, llm
 *                (messages), llm_clients (config, connection, factory), expr (expressions,
 *                templates, lineage), memory (recall rendering, memory tools), skills, tools,
 *                defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { compareCodeUnits, type FailureCode, type OriginRef, reprText } from "../../diagnostics.ts";
import {
  type DeclaredTypes,
  ExpressionError,
  localDeclared,
  ResolveMode,
  resolveInRecord,
} from "../../expr/expressions.ts";
import { evalTemplate, renderTemplateRecord, scanTemplate } from "../../expr/template.ts";
import type { HostConnection } from "../../llm_clients/config.ts";
import { scrubCallCredentials } from "../../llm_clients/connection.ts";
import { defaultModelAccess, type ModelAccess } from "../../llm_clients/factory.ts";
import { MediaPart, Message as ModelMessage, MessageFormatError, TextPart } from "../../llm/messages.ts";
import { checkSkillName, type Skill, SkillError } from "../../skills/catalog.ts";
import { LOAD_SKILL, loadSkillTool, systemPromptWithSkills } from "../../skills/tool.ts";
import { renderRecall } from "../../memory/render.ts";
import { grantedMemorySpecs, MEMORY_SCOPE_LABEL, memoryTools } from "../../memory/tools.ts";
import type { FunctionTool } from "../../tools/function_tools.ts";
import { asToolGrant, type MemoryGrant, type ToolGrant, visibleToolName } from "../../tools/grants.ts";
import { ToolSet } from "../../tools/registry.ts";
import { type JsonObject, logJson } from "../../typesys/json.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import {
  buildValueAs,
  isList,
  inboxName,
  isMemoryType,
  isRecord,
  MemoryRef,
  setField,
  Type,
  TypeCheckError,
  TypedValue,
  ValueKind,
  type ValueRecord,
  valueRepr,
  writeAsText,
} from "../../typesys/values.ts";
import {
  bindMemoryGrants,
  CapabilityFault,
  checkNamedPrompt,
  type CompleteResult,
  MEMORY_KEY_PREFIX,
  memoryKey,
  memorySession,
  type RebuildCaps,
  type RebuiltRequests,
  Node,
  type NodeCaps,
  NodeKind,
  Output,
  type Pending,
  type SteerInbox,
} from "../base.ts";
import { applyDefaults, BindingError, copyValue, type ReservedBinder, resolveSource, sourceOrigin } from "../binding.ts";
import {
  acceptsNull,
  Binding,
  type CallView,
  isExactly,
  iterOriginCarrier,
  type NodeSteps,
  type OriginMapper,
  outputContractType,
  Parameter,
  rewriteOriginCarrier,
  ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../contracts.ts";
import { staticPathOf } from "../ids.ts";
import { type AgentRunContext, checkStopped, DEFAULT_SYSTEM, sentCredentials, ToolIterationLimitError } from "./context.ts";
import { Interrupted } from "./interrupts.ts";
import { modelFailureCode } from "./live.ts";
import {
  type CallTimeLLM,
  callTimeConfig,
  invocationConfig,
  LLM_CONFIG_KEY,
  type Nullable,
  typesNullable,
} from "./llm_values.ts";
import { rebuildStepRequests, runAgent } from "./loop.ts";
import { reportedText, toolText } from "./results.ts";
import {
  checkMessage,
  checkModelConversation,
  ConversationError,
  type ConversationMessage,
  messageSteps,
  rebuildMessages,
  type SteerRender,
  toModelMessage,
} from "./messages.ts";
import { OutputInvalidError } from "./structured.ts";

/** The reserved wiring key an agent's `steer_inbox:` rides; it binds to the inbox's name. */
export const STEER_INBOX_KEY = "__steer_inbox";

/** The step key under which a call-time agent records the config it resolved. */
const CONFIG_STEP = "effective_llm_config";
/** The `kind` of the step a call-time agent records its config in. */
const CONFIG_STEP_KIND = "llm_config";

/**
 * Open the steer inbox named `inbox` through `caps.openSteerInbox`.
 *
 * @throws CapabilityFault carrying a TypeError when the evaluator gave no steer inbox.
 */
function openSteerInbox(caps: NodeCaps, inbox: string): SteerInbox {
  if (caps.openSteerInbox === undefined) throw new CapabilityFault(new TypeError("the evaluator gave no steer inbox"));
  return caps.openSteerInbox(inbox);
}

/** Whether `value` is a valid `max_tool_iterations`: a positive int, or `-1n` for no cap. */
export function isMaxToolIterations(value: unknown): value is bigint {
  return typeof value === "bigint" && (value >= 1n || value === -1n);
}

/**
 * One item of an AGENT list prompt: a message whose text is a template, or the name of a local
 * input or variable holding a message or a list of messages, spliced in.
 */
export type PromptItem =
  | { readonly role: "user" | "assistant"; readonly text: string }
  | { readonly splice: string };

/**
 * Return a list prompt as authored: a splice as `${name}`, a message as `{<role>: <text>}`. The
 * manifest holds this form.
 */
export function authoredPrompt(items: readonly PromptItem[]): (string | Record<string, string>)[] {
  return items.map((item) => ("splice" in item ? `\${${item.splice}}` : { [item.role]: item.text }));
}

/**
 * Check a list prompt's items.
 *
 * @throws TypeError for an item other than `{role: "user" | "assistant", text: string}` or
 *   `{splice: string}`.
 * @throws RangeError for an empty list, or one whose last item is an assistant message.
 */
function checkPromptItems(nodeId: string, items: readonly unknown[]): void {
  const where = `AGENT node ${reprText(nodeId)}`;
  if (items.length === 0) throw new RangeError(`${where}: a list prompt must hold at least one item`);
  items.forEach((item, index) => {
    const fields: Readonly<Record<string, unknown>> = isRecord(item) ? item : {};
    const keys = Object.keys(fields).sort(compareCodeUnits).join(",");
    const message =
      keys === "role,text" &&
      (fields.role === "user" || fields.role === "assistant") &&
      typeof fields.text === "string";
    const splice = keys === "splice" && typeof fields.splice === "string";
    if (!message && !splice) {
      throw new TypeError(`${where}: prompt[${index}] must be {role: user or assistant, text} or {splice: <name>}`);
    }
  });
  if ((items.at(-1) as Readonly<Record<string, unknown>>).role === "assistant") {
    throw new RangeError(`${where}: the last message of a list prompt must be from the user`);
  }
}

/**
 * One rendered prompt message: a text message's role and text, or a spliced `Message`'s model
 * message; and the prompt item it came from.
 */
interface PromptMessage {
  readonly role: "user" | "assistant" | "tool";
  /** A text message's text, or the model message's text parts. */
  readonly text: string;
  /** A spliced `Message`'s model message; `null` for a text message, built when the prompt's messages are read. */
  readonly model: ModelMessage | null;
  /** The index of the prompt item it came from. */
  readonly item: number;
}

/** Return a text message of `role` from prompt item `item`. */
function textMessage(role: "user" | "assistant", text: string, item: number): PromptMessage {
  return { role, text, model: null, item };
}

/** Return what `checkModelConversation` reads of a rendered message; a text message is one text part. */
function conversationForm(rendered: PromptMessage): ConversationMessage {
  return rendered.model ?? { role: rendered.role, parts: [{ type: "text" }] };
}

/** Return the model message a rendered message sends, building a text message's here. */
function modelMessage(rendered: PromptMessage): ModelMessage {
  return rendered.model ?? new ModelMessage({ role: rendered.role, parts: [new TextPart(rendered.text)] });
}

/** Whether `value` is a `{role, text}` prompt message: a record with `role` user or assistant and a string `text`. */
function isTextMessage(value: unknown): value is { role: "user" | "assistant"; text: string } {
  return isRecord(value) && (value.role === "user" || value.role === "assistant") && typeof value.text === "string";
}

/**
 * Return a spliced value's message: a record with a `text` field is a `{role, text}` message
 * (other fields ignored); anything else must be a built-in `Message`, sent with all its parts.
 *
 * @throws ExpressionError when it is neither, or holds a media part, which no model route sends yet.
 */
function splicedMessage(nodeId: string, index: number, ref: string, value: unknown): PromptMessage {
  if (isRecord(value) && Object.hasOwn(value, "text")) {
    if (isTextMessage(value)) return textMessage(value.role, value.text, index);
    throw notAMessage(nodeId, index, ref, null);
  }
  let model: ModelMessage;
  try {
    model = toModelMessage(checkMessage(value));
  } catch (error) {
    if (error instanceof TypeCheckError || error instanceof MessageFormatError) throw notAMessage(nodeId, index, ref, error.message);
    throw error;
  }
  if (model.parts.some((part) => part instanceof MediaPart)) {
    throw new ExpressionError(`node ${reprText(nodeId)} prompt[${index}]: ${ref} holds a media part; no model route sends media yet`);
  }
  return { role: model.role as "user" | "assistant" | "tool", text: model.text(), model, item: index };
}

/** The error for a spliced value that isn't a message, with why a `Message` didn't fit, if one was read. */
function notAMessage(nodeId: string, index: number, ref: string, reason: string | null): ExpressionError {
  return new ExpressionError(
    `node ${reprText(nodeId)} prompt[${index}]: ${ref} must be a message (a \`Message\`, or a record with \`role\` user ` +
      "or assistant and `text: str`) or a list of them" + (reason === null ? "" : `; ${reason}`),
  );
}

/**
 * Render a list prompt against `record`: each message's text as a template, each splice's value
 * as one message or a list of them. A `{role, text}` record keeps only `role` and `text`; a
 * `Message` keeps every part.
 *
 * @throws ExpressionError when a template or splice names a missing value or reads a null, a
 *   spliced value isn't a message or a list of messages or holds a media part, or the list
 *   renders no message, ends in one that isn't the user's, or isn't a conversation a model route
 *   can send (`checkModelConversation`); that error names the prompt item holding the offending
 *   message, for an unanswered call the assistant message that made it.
 */
function renderPromptList(
  nodeId: string,
  items: readonly PromptItem[],
  record: ValueRecord,
  declared: DeclaredTypes | null = null,
): PromptMessage[] {
  const messages: PromptMessage[] = [];
  items.forEach((item, index) => {
    if (!("splice" in item)) {
      messages.push(textMessage(item.role, renderTemplateRecord(item.text, record, declared), index));
      return;
    }
    const ref = `\${${item.splice}}`;
    const value = evalTemplate(scanTemplate(ref), (names) => resolveInRecord(names, record), {
      mode: ResolveMode.STRICT_RAISE,
      declared,
    });
    const spliced: readonly unknown[] = isList(value) ? value : [value];
    for (const one of spliced) messages.push(splicedMessage(nodeId, index, ref, one));
  });
  const last = messages.at(-1);
  if (last === undefined) {
    throw new ExpressionError(`node ${reprText(nodeId)} prompt: the rendered list holds no message`);
  }
  if (last.role !== "user") {
    throw new ExpressionError(
      `node ${reprText(nodeId)} prompt: the last message must be from the user, got ${last.role}`,
    );
  }
  try {
    checkModelConversation(messages.map(conversationForm));
  } catch (error) {
    if (!(error instanceof ConversationError)) throw error;
    throw new ExpressionError(
      `node ${reprText(nodeId)} prompt[${messages[error.index]!.item}]: rendered message ${error.index}: ${error.reason}`,
    );
  }
  return messages;
}

/** The prompt one AGENT invocation sends, rendered from its inputs, variables and granted skills. */
export interface AgentPrompt {
  /** Build the prompt's model messages, after the system message; the last is the user's. */
  modelMessages(): readonly ModelMessage[];
  /** The authored system prompt rendered, or `DEFAULT_SYSTEM`, plus the granted skills' listing. */
  readonly systemPrompt: string;
  /** The granted skills, sorted by name. */
  readonly skills: readonly Skill[];
}

/** Return the first local variable of `node` named like a declared or delivered input, or `null`. */
function collidingVariable(node: AgentNode, inputs: Readonly<Record<string, unknown>>): string | null {
  const inputNames = new Set([...Object.keys(inputs), ...(node.params ?? []).map((parameter) => parameter.name)]);
  return node.variables.find((variable) => inputNames.has(variable.name))?.name ?? null;
}

/**
 * Whether `error` is a render's refusal (a template error, a missing skill), which leaves an
 * invocation's prompt unrebuilt; any other error is a fault and escapes.
 */
function isRebuildRefusal(error: unknown): boolean {
  return error instanceof ExpressionError || error instanceof SkillError;
}

/** An AGENT a host interrupted whose declared output doesn't allow null; its failure is `agent.interrupted`. */
export class AgentInterruptedError extends Error {
  override name = "AgentInterruptedError";
}

/**
 * An LLM-backed leaf configured by a prompt and authored tools.
 *
 * `systemPrompt` is an optional template that replaces the default system message; it renders
 * against the same local record as `prompt`. `variables` keeps typed local declarations with
 * mandatory literal defaults, resolved into a fresh prompt record for each invocation.
 */
export class AgentNode extends Node {
  readonly kind = NodeKind.AGENT;
  override readonly needsTools: boolean = true;
  override readonly needsMemories: boolean = true;
  override readonly needsLiveEvents: boolean = true;
  override readonly needsSkills: boolean = true;
  override readonly needsModels: boolean = true;
  /**
   * Every agent records its model calls and tool results as steps; a call-time agent also records
   * and reads back its config as an `llm_config` step.
   */
  override readonly needsSteps: boolean = true;
  /** An AGENT's success also commits its new messages, rebuilt from its steps. */
  override readonly producesMessages: boolean = true;
  /** The prompt template, or a list of messages and splices. */
  prompt: string | readonly PromptItem[];
  /** The system prompt template replacing the default one; `null` keeps the default. */
  systemPrompt: string | null;
  /** The authored tool grants, which may name host-filled groups and memory inputs. */
  tools: readonly ToolGrant[];
  /**
   * The memory inputs of this agent's flow that `tools` names, each riding the reserved
   * `__memory_<input>` wiring; a grant naming one is a memory grant, never a host one.
   */
  readonly memoryInputs: readonly string[];
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
  /** The local variable declarations, each with a literal default. */
  variables: readonly Parameter[];
  /** The granted skill names, or `"all"` for every skill the host supplies. */
  skills: readonly string[] | "all";
  /** Whether this agent reads a steer inbox (`steer_inbox:`, the reserved `__steer_inbox` wiring). */
  readonly steerInbox: boolean;
  /** The text template each steer event renders through, over `${event}`; `null` for the default rendering. */
  readonly steerRender: string | null;
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
   * @param options.prompt The prompt template or a list prompt; `null` or omitted means `""`.
   * @param options.systemPrompt A non-empty system prompt template, or `null` (the default).
   * @param options.tools The tool grants, a text read as a name grant; defaults to `[]`.
   * @param options.llmConfig A config record, a `ValueTemplate` or `Binding` for a layer
   *   holding values, or `null` (the default).
   * @param options.llmInherit Whether the config merges over the enclosing layers; defaults to
   *   `true`.
   * @param options.retries Bad structured reply retries; `null` or omitted means
   *   `current().agentRetries`.
   * @param options.title A display title.
   * @param options.variables Exact `Parameter`s with literal defaults; defaults to `[]`.
   * @param options.skills Skill names, or `"all"`; defaults to `[]`.
   * @param options.steerInbox Whether this agent reads a steer inbox; defaults to `false`.
   * @param options.steerRender The template each steer event renders through, over `${event}`;
   *   defaults to `null`, the default rendering.
   * @param options.memoryInputs The memory inputs of the agent's flow its grants name; defaults
   *   to none.
   * @throws TypeError for a bad `prompt` item, `systemPrompt`, `llmConfig`, `variables`,
   *   `steerInbox`, `steerRender` or `memoryInputs`.
   * @throws RangeError for an empty list prompt or one ending in an assistant message, the retired
   *   `builtin_tools` grant, a steer render without a steer inbox, or a variable whose name repeats
   *   or whose default is missing or invalid.
   * @throws SkillError for a `load_skill` tool grant or a bad skill grant.
   */
  constructor(
    nodeId: string,
    options: {
      prompt?: string | readonly PromptItem[] | null;
      systemPrompt?: string | null;
      tools?: readonly (string | ToolGrant)[] | null;
      llmConfig?: Readonly<Record<string, unknown>> | ValueTemplate | Binding | null;
      llmInherit?: boolean;
      retries?: number | null;
      title?: string | null;
      variables?: readonly Parameter[] | null;
      skills?: readonly string[] | "all" | null;
      steerInbox?: boolean;
      steerRender?: string | null;
      memoryInputs?: readonly string[] | null;
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
      variables = null,
      skills = null,
      steerInbox = false,
      steerRender = null,
      memoryInputs = null,
    } = options;
    super(nodeId, { title });
    if (isList(prompt)) checkPromptItems(nodeId, prompt);
    this.prompt = prompt ?? "";
    if (systemPrompt !== null && (typeof systemPrompt !== "string" || systemPrompt === "")) {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: system_prompt must be a non-empty string or null`);
    }
    this.systemPrompt = systemPrompt;
    this.tools = (tools ?? []).map(asToolGrant);
    if (memoryInputs !== null && !(isList(memoryInputs) && memoryInputs.every((input) => typeof input === "string"))) {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: memoryInputs must be a list of input names`);
    }
    this.memoryInputs = Object.freeze([...(memoryInputs ?? [])]);
    if (this.tools.some((grant) => grant.kind === "name" && grant.name === "builtin_tools")) {
      throw new RangeError(
        "builtin_tools belonged to the removed native Codex integration. " +
          "Remove it and grant registered Yian tools explicitly.",
      );
    }
    // A layer holding values (a template, or a binding for the whole record) has no config until
    // they resolve.
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
    this.variables = variables ?? [];
    this.skills = skills ?? [];
    this.validateSkills();
    this.validateVariables();
    if (typeof steerInbox !== "boolean") {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: steerInbox must be a bool`);
    }
    this.steerInbox = steerInbox;
    this.readsSteerInbox = steerInbox;
    if (steerRender !== null && (typeof steerRender !== "string" || steerRender === "")) {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: steer render must be non-empty text or null`);
    }
    if (steerRender !== null && !steerInbox) {
      throw new RangeError(`AGENT node ${reprText(nodeId)}: a steer render needs a steer inbox`);
    }
    this.steerRender = steerRender;
    this.llmConfigLabel = null;
    this.wiresLlmConfig = false;
    this.llmValueTypes = [];
    this.callTime = null;
    this.hostCredentials = null;
  }

  /**
   * Declare `__llm_config` when values in the layers can reach this agent's config,
   * `__steer_inbox` when it reads a steer inbox, and `__memory_<input>` for each memory it grants.
   */
  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(this.reservedKeys());
  }

  /** The reserved wiring keys this agent has. */
  private reservedKeys(): string[] {
    return [
      ...(this.wiresLlmConfig ? [LLM_CONFIG_KEY] : []),
      ...(this.steerInbox ? [STEER_INBOX_KEY] : []),
      ...this.memoryInputs.map(memoryKey),
    ];
  }

  /**
   * Resolve the steer inbox's name, and check a call-time agent's `llm_config`, before its model
   * call.
   *
   * The steer inbox binding (`__steer_inbox`) resolves through `options.bind` (else against
   * `pool`) to the inbox's name. The node and own-flow `llm_config` layers resolve against `pool`,
   * merge over the base run start recorded, and are checked as run start checks a static agent's
   * config (`callTimeConfig`). Nothing is rendered or sent to a model first.
   * `options.bind` is unused for the config: the bound value is the resolved config, not the values
   * it was read from.
   *
   * Each granted memory's binding (`__memory_<input>`) resolves the same way to its memory value,
   * or `null` for an Optional memory left out.
   *
   * @returns `{__steer_inbox: <name>}` for an agent with a steer inbox, `{__memory_<input>:
   *   <memory>}` per granted memory, and `{__llm_config: <config>}`, the resolved config with no
   *   key or header value, the shape of a `RunStarted` map entry, for a call-time agent; `{}` for
   *   none.
   * @throws Error when the wiring lacks a reserved binding the agent has.
   * @throws BindingError when the steer inbox binding isn't an inbox, a memory binding isn't a
   *   memory, a value breaks its tag (`binding.type`) or fails to resolve, or the resolved config
   *   is refused (`contract.input`); the message names the field.
   * @throws AuthError when the config is marked for the host key the run does not hold; the
   *   host's to fix, not a failure of this node.
   */
  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    const reserved: Record<string, unknown> = {};
    if (this.steerInbox) {
      if (!nodeWiring.has(STEER_INBOX_KEY)) {
        throw new Error(`AGENT node ${reprText(this.id)} has no \`${STEER_INBOX_KEY}\` wiring`);
      }
      const source = nodeWiring.get(STEER_INBOX_KEY);
      const normalize = (value: unknown): string => {
        const name = inboxName(value);
        if (name === null) {
          throw new BindingError(`AGENT node ${reprText(this.id)}: steer_inbox must be an inbox`, {
            failureCode: "contract.input",
            origin: sourceOrigin(source),
          });
        }
        return name;
      };
      const { bind = null } = options;
      reserved[STEER_INBOX_KEY] =
        bind === null ? normalize(resolveSource(source, pool)) : bind(STEER_INBOX_KEY, source, normalize);
    }
    const memories = bindMemoryGrants(this.id, this.memoryGrants(), "tools", nodeWiring, pool, options.bind ?? null);
    Object.assign(reserved, memories);
    if (this.callTime === null) return reserved;
    const template = nodeWiring.get(LLM_CONFIG_KEY);
    if (!(template instanceof Binding)) {
      throw new Error(`AGENT node ${reprText(this.id)} has no \`${LLM_CONFIG_KEY}\` wiring`);
    }
    reserved[LLM_CONFIG_KEY] = callTimeConfig(template, pool, this.callTime, {
      label: this.llmConfigLabel,
      credentials: this.hostCredentials,
    });
    return reserved;
  }

  /** Name the reserved `__llm_config`, `__steer_inbox` and `__memory_<input>` keys as the authored fields. */
  override reservedKeyLabel(key: string): string {
    if (key === LLM_CONFIG_KEY) return "llm_config";
    if (key === STEER_INBOX_KEY) return "steer_inbox";
    if (key.startsWith(MEMORY_KEY_PREFIX)) return "tools";
    return super.reservedKeyLabel(key);
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
   * @returns A copy of the `effective_llm_config` of the last `llm_config` step, with no key or
   *   header value; `null` when no step records one.
   */
  recordedLlmConfig(steps: readonly unknown[]): Record<string, unknown> | null {
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      const step = steps[index];
      if (isRecord(step) && step["kind"] === CONFIG_STEP_KIND && isRecord(step[CONFIG_STEP])) {
        return copyValue(step[CONFIG_STEP]) as Record<string, unknown>;
      }
    }
    return null;
  }

  /** Return the declared inputs, never the reserved `__llm_config` or `__steer_inbox`. */
  override dataInputNames(): readonly string[] {
    return super.dataInputNames().filter((name) => name !== LLM_CONFIG_KEY && name !== STEER_INBOX_KEY);
  }

  /**
   * Render one steer event through this agent's `render:` template over `${event}`, typed by
   * `localTypes` (which holds `event`); `null` without one.
   */
  private steerRenderer(localTypes: ReadonlyMap<string, Type | null> | null = null): SteerRender | null {
    const template = this.steerRender;
    if (template === null) return null;
    const declared = localDeclared(localTypes);
    return (value) => renderTemplateRecord(template, { event: value } as ValueRecord, declared);
  }

  /** Pick the steps of this agent's new messages (see `messageSteps` in `messages.ts`). */
  override messageSteps(steps: readonly Readonly<JsonObject>[]): number[] {
    return messageSteps(steps);
  }

  /** Rebuild this agent's new messages (see `rebuildMessages` in `messages.ts`). */
  override rebuildMessages(
    steps: readonly Readonly<ValueRecord>[],
    output: TypedValue | null,
    localTypes: ReadonlyMap<string, Type | null> | null = null,
  ): TypedValue {
    return rebuildMessages(steps, output, this.steerRenderer(localTypes));
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
    // `||`, not `??`: an empty path names the node by its id.
    const subject = `node ${reprText(path || this.id)}`;
    if (typeof this.prompt === "string") {
      checkNamedPrompt(this.prompt, `${subject} prompt`);
    } else {
      this.prompt.forEach((item, index) => {
        if ("text" in item) checkNamedPrompt(item.text, `${subject} prompt[${index}]`);
      });
    }
    if (this.systemPrompt !== null) checkNamedPrompt(this.systemPrompt, `${subject} system_prompt`);
    if (this.steerRender !== null) checkNamedPrompt(this.steerRender, `${subject} steer_inbox render`);
  }

  /** Require `"all"` or unique valid names, and no `load_skill` tool grant. */
  private validateSkills(): void {
    const where = `AGENT node ${reprText(this.id)}`;
    if (this.tools.some((grant) => grant.kind === "name" && grant.name === LOAD_SKILL)) {
      throw new SkillError(`${where}: ${reprText(LOAD_SKILL)} is a reserved tool name; remove it from tools`);
    }
    if (this.skills === "all") return;
    if (!isList(this.skills)) throw new SkillError(`${where}: skills must be a list of names or 'all'`);
    for (const name of this.skills) checkSkillName(name);
    if (new Set(this.skills).size !== this.skills.length) {
      throw new SkillError(`${where}: duplicate skill grant in ${writeAsText([...this.skills])}`);
    }
  }

  /** Return the authored `tools:` grants of host tools and groups, which may name host-filled groups. */
  override grantedTools(): readonly ToolGrant[] {
    return this.tools.filter((grant) => !this.memoryInputs.includes(grantName(grant)));
  }

  /** Return the authored `tools:` grants of memories, in grant order. */
  override memoryGrants(): readonly MemoryGrant[] {
    return this.tools
      .filter((grant) => this.memoryInputs.includes(grantName(grant)))
      .map((grant) => Object.freeze({ input: grantName(grant), tools: grant.kind === "name" ? null : grant.tools }));
  }

  /**
   * Shape the `tool_result` step for a host's `complete` on one of this agent's pending calls: the
   * text the model reads (`toolText`, or a reported error's text), keyed by the provider's tool
   * call id. Pure.
   *
   * @throws TypeCheckError when the output isn't a yian value.
   */
  override waitResultStep(call: CallView, result: CompleteResult): JsonObject {
    const reported = Object.hasOwn(result, "error");
    const output = reported
      ? reportedText(call.tool, (result as { error: string }).error)
      : toolText(call.tool, (result as { output: unknown }).output);
    return { kind: "tool_result", call_id: call.resultKey, started_at: null, output, is_error: reported };
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
   * Yield the common origins, then the variables' and those of the values in `ownLlmConfig`.
   */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    yield* super.iterLocalOrigins(seen);
    for (const variable of this.variables) yield* iterOriginCarrier(variable, seen);
    if (this.ownLlmConfig instanceof ValueTemplate || this.ownLlmConfig instanceof Binding) {
      yield* iterOriginCarrier(this.ownLlmConfig, seen);
    }
  }

  /**
   * Rewrite the common, variable and `llmConfig` value origins on a copied AGENT, reassigning each
   * field with rebuilt carriers. Carrier aliases are kept.
   */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.variables = this.variables.map((variable) => rewriteOriginCarrier(variable, mapper, seen));
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
   * @param messages The rendered prompt's messages.
   * @param config The config this invocation's model call uses, host secrets included; it lives
   *   only in the returned context.
   * @throws RangeError when the authored `env.max_tool_iterations` is invalid.
   */
  private runContext(
    messages: readonly ModelMessage[],
    config: Readonly<Record<string, unknown>> | null,
    caps: NodeCaps,
    options: {
      inputs: Readonly<Record<string, unknown>>;
      systemPrompt: string;
      skills: readonly Skill[];
      models: ModelAccess;
      steerInbox: SteerInbox | undefined;
    },
  ): AgentRunContext {
    const { tools, toolset } = this.invocationTools(options.inputs, caps);
    return {
      nodeId: this.id,
      messages,
      tools,
      llmConfig: config,
      outputType: isExactly(this.outputContract, ValueContract) ? this.outputContract.type : null,
      retries: this.retries,
      maxToolIterations: this.maxToolIterations(),
      signal: caps.signal,
      toolset,
      emitStream: caps.emitStream,
      emitEvent: caps.emitEvent,
      systemPrompt: options.systemPrompt,
      skills: options.skills,
      steerInbox: options.steerInbox,
      renderSteer: this.steerRenderer(caps.localTypes ?? null) ?? undefined,
      models: options.models,
      sent: () => [],
      steps: caps.steps,
      scheduler: caps.scheduler,
      random: caps.random,
      runDir: caps.runDir ?? null,
    };
  }

  /**
   * Return the grants one invocation's model is given and the tool set they resolve in: every
   * authored grant in its order, but none of a memory left out (an Optional one, null), and the
   * run's tools with each bound memory's granted tools as a local group over its run session.
   *
   * @throws CapabilityFault carrying the error when a memory is bound but the evaluator gave no
   *   memory sessions, or none on that memory.
   */
  private invocationTools(
    inputs: Readonly<Record<string, unknown>>,
    caps: NodeCaps,
  ): { tools: ToolGrant[]; toolset: ToolSet | null } {
    const { tools, memories } = this.memoryPlan(inputs);
    const groups = new Map<string, FunctionTool[]>();
    const author = { kind: "agent" as const, name: staticPathOf(this.id) };
    for (const [input, value] of memories) {
      const session = memorySession(caps, value.identity);
      groups.set(input, memoryTools(value, session, { input, tools: this.grantedMemoryTools(input) }, author));
    }
    const toolset = caps.tools ?? null;
    if (groups.size === 0) return { tools, toolset };
    return { tools, toolset: (toolset ?? new ToolSet()).withLocalGroups(groups, MEMORY_SCOPE_LABEL) };
  }

  /**
   * Return the grants an invocation's model is given, which the live loop and a rebuilt request
   * both expand: every authored grant in its order, but none of a memory left out (an Optional one,
   * null); and each bound memory a grant names, by input in grant order.
   */
  private memoryPlan(inputs: Readonly<Record<string, unknown>>): { tools: ToolGrant[]; memories: Map<string, MemoryRef> } {
    const bound = this.boundMemories(inputs);
    const tools = this.tools.filter((grant) => !bound.has(grantName(grant)) || bound.get(grantName(grant)) !== null);
    const memories = new Map<string, MemoryRef>();
    for (const [input, value] of bound) if (value !== null) memories.set(input, value);
    return { tools, memories };
  }

  /**
   * Return the memory each memory input this agent grants is bound to, by input in grant order:
   * its reserved `__memory_<input>` value, `null` for an Optional memory left out.
   */
  private boundMemories(inputs: Readonly<Record<string, unknown>>): Map<string, MemoryRef | null> {
    const bound = new Map<string, MemoryRef | null>();
    for (const { input } of this.memoryGrants()) {
      const value = inputs[memoryKey(input)];
      bound.set(input, MemoryRef.isMemoryRef(value) ? value : null);
    }
    return bound;
  }

  /** Return the tools this agent's grants give of memory input `input`, once each; `null` for all. */
  private grantedMemoryTools(input: string): readonly string[] | null {
    const listed: string[] = [];
    for (const grant of this.memoryGrants()) {
      if (grant.input !== input) continue;
      if (grant.tools === null) return null;
      for (const tool of grant.tools) if (!listed.includes(tool)) listed.push(tool);
    }
    return listed;
  }

  /**
   * Return the model-visible name of `value`'s hint tool when this agent holds that tool on a
   * memory of `value`'s identity, named by the grant's input; else `null`.
   */
  private hintTool(value: MemoryRef, inputs: Readonly<Record<string, unknown>>): string | null {
    const hint = value.recall.hint;
    if (hint === null) return null;
    for (const [input, bound] of this.boundMemories(inputs)) {
      if (bound === null || bound.identity !== value.identity) continue;
      const granted = this.grantedMemoryTools(input);
      const holds = granted === null ? bound.tools.some((tool) => tool.name === hint.tool) : granted.includes(hint.tool);
      if (holds) return visibleToolName(input, hint.tool);
    }
    return null;
  }

  /**
   * Replace, in `record`, each memory input with its rendered recall (`renderRecall`), and its
   * declared type in `types` with `str`. A memory input is one holding a `MemoryRef`, or one whose
   * declared type is a memory type, which then holds a memory or null (an Optional one left out,
   * which renders empty).
   *
   * @throws TypeError when an input declared a memory holds anything else: a bound value no
   *   prompt may show.
   */
  private renderMemories(
    record: ValueRecord,
    types: Map<string, Type | null>,
    inputs: Readonly<Record<string, unknown>>,
  ): void {
    for (const [name, value] of Object.entries(record)) {
      if (name.startsWith(MEMORY_KEY_PREFIX)) continue;
      const declared = isMemoryType(types.get(name));
      let text: string;
      if (MemoryRef.isMemoryRef(value)) text = renderRecall(value.recall, this.hintTool(value, inputs));
      else if (declared && value === null) text = renderRecall(null, null);
      else if (declared) throw new TypeError(`AGENT node ${reprText(this.id)}: memory input ${reprText(name)} holds no memory`);
      else continue;
      setField(record, name, text);
      types.set(name, Type.scalar(ValueKind.STRING));
    }
  }

  /**
   * Select a generated structured-output contract origin when proven.
   *
   * A variable-default binding failure keeps its own origin. A structured reply
   * that failed the value check maps to its output contract's origin at the error's path (exact
   * or nearest ancestor). Other failures, and contracts without authored provenance, keep the
   * common node-origin fallback.
   */
  override exceptionOrigin(error: unknown): OriginRef | null {
    if (error instanceof BindingError) {
      return error.origin ?? super.exceptionOrigin(error);
    }
    const contract = this.outputContract;
    if (error instanceof OutputInvalidError && error.path !== null && isExactly(contract, ValueContract)) {
      return contract.origins.at(error.path) ?? super.exceptionOrigin(error);
    }
    return super.exceptionOrigin(error);
  }

  /**
   * Classify an interrupt the output can't take, the tool-iteration cap, an exhausted structured
   * reply and the agent's own model failures; defer the rest.
   *
   * An interrupted agent whose output doesn't allow null is `agent.interrupted`, the cap is
   * `agent.max_tool_iterations` and a structured reply still invalid once `retries:` are spent is
   * `agent.output_invalid`. An error the agent's own model build or model call
   * marked takes `modelFailureCode`'s code (`provider.auth` for refused credentials, a
   * `provider.*` code by provider error kind, else `node.execute`).
   */
  override exceptionCode(error: unknown): FailureCode {
    if (error instanceof AgentInterruptedError) return "agent.interrupted";
    if (error instanceof ToolIterationLimitError) return "agent.max_tool_iterations";
    if (error instanceof OutputInvalidError) return "agent.output_invalid";
    return modelFailureCode(error) ?? super.exceptionCode(error);
  }

  /**
   * Render the prompt once (`renderPrompt`) and run the loop. No invocation state is stored on the
   * node.
   *
   * A call-time agent's model config is its bound `__llm_config` with the run's host key and
   * header values joined when it is marked for the host key; any other agent uses its run's
   * `llmConfig`. Before anything renders, a call-time agent checks the bound config against the
   * one an earlier start recorded in its steps, then records it as a step,
   * `{kind: "llm_config", effective_llm_config: <config>}`, which holds no key or header value.
   * Each model call is recorded as a `model_call` step and each tool result as a `tool_result`
   * step (`runAgent`); a later start replays them before calling the model again.
   *
   * The model is built through `caps.models`, else `defaultModelAccess()`. Every error from the
   * loop has the credentials the call sent scrubbed before it leaves.
   *
   * @param inputs Delivered inputs, kept separate from local variables.
   * @param caps The node's signal, checked before the model call, its scheduler and random draw for
   *   model-retry waits, and the capabilities its traits ask for: tools, live sinks, skills, the
   *   steer inbox (with its interrupt signal), model access and steps.
   * @returns The model's answer, or Pending when a tool call waits for a later answer; null after a
   *   host interrupt when the declared output allows it (`Optional(...)`).
   * @throws RangeError when a variable collides with a declared or delivered input.
   * @throws ExpressionError when the prompt or system prompt names a missing value or renders a
   *   null, a spliced value isn't a message (a `Message` or a `{role, text}` record) or holds a
   *   media part, or a list prompt renders no message, ends in one that isn't the user's, or isn't
   *   a conversation a model route can send.
   * @throws SkillError when the node grants skills the host did not supply.
   * @throws CapabilityFault when the bound config differs from the one an earlier start recorded,
   *   or a recorded model call doesn't match the rebuilt request; the evaluator rethrows its
   *   original host error.
   * @throws AgentInterruptedError after a host interrupt when the declared output doesn't allow
   *   null (no `output:` means `str`).
   * @throws The signal's reason once the host stopped the node.
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output | Pending> {
    if (caps.steps !== undefined && this.callTime !== null) {
      this.recordConfig(inputs[LLM_CONFIG_KEY] as Record<string, unknown>, caps.steps);
    }
    // Opened before anything renders, so a replaying agent already reads its inbox.
    const steerInbox = this.steerInbox ? openSteerInbox(caps, inputs[STEER_INBOX_KEY] as string) : undefined;
    const { signal, localTypes } = caps;
    const rendered = this.renderPrompt(inputs, caps.skills ?? null, { signal, localTypes });
    const config = this.callConfig(inputs);
    const models = caps.models ?? defaultModelAccess();
    const env = models.env;
    checkStopped(caps);
    const ctx = this.runContext(rendered.modelMessages(), config, caps, {
      inputs,
      systemPrompt: rendered.systemPrompt,
      skills: rendered.skills,
      models,
      steerInbox,
    });
    let result: Output | Pending | Interrupted;
    try {
      result = await runAgent(ctx);
    } catch (error) {
      // Providers may echo any credential the call sent; the failure record must not carry it.
      throw scrubCallCredentials(error, config, { env, sent: sentCredentials(ctx) });
    }
    if (!(result instanceof Interrupted)) return result;
    // An interrupted reply has no answer: null, which only an output that allows it takes. No
    // `output:` means `str`.
    const typ = outputContractType(this.outputContract);
    if (typ !== null && acceptsNull(typ)) return new Output(null);
    throw new AgentInterruptedError(
      `agent node ${reprText(this.id)} was interrupted and its output doesn't allow null; ` +
        "declare output: Optional(...) to route on an interrupt",
    );
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
    if (recorded !== null && logJson(recorded) !== logJson(config)) {
      throw new CapabilityFault(
        new RangeError(
          `node ${reprText(this.callTime!.path)} llm_config: recorded config differs from the ` +
            "recomputed one; resume the run with the flow and yian version that started it",
        ),
      );
    }
    steps.record({ kind: CONFIG_STEP_KIND, [CONFIG_STEP]: copyValue(config as ValueRecord) });
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

  /**
   * Render what one invocation sends: the pure half of `forward`, which `rebuildRequests` also
   * calls with the recorded skills.
   *
   * The authored `systemPrompt` renders against the same inputs and variables as `prompt`; without
   * one it is `DEFAULT_SYSTEM`. A text prompt renders into one user message, a list prompt into its
   * messages (`renderPromptList`).
   *
   * @param inputs Delivered inputs, kept separate from local variables.
   * @param skills The run's skills by name, or `null` when the host supplied none.
   * @param options `signal` is checked after rendering, before the skill grants are read.
   * @throws RangeError when a variable collides with a declared or delivered input.
   * @throws ExpressionError as `forward` documents for a prompt that can't render.
   * @throws SkillError when the node grants skills `skills` doesn't hold.
   * @throws The signal's reason once it aborted.
   */
  private renderPrompt(
    inputs: Readonly<Record<string, unknown>>,
    skills: ReadonlyMap<string, Skill> | null,
    options: {
      signal?: AbortSignal;
      localTypes?: ReadonlyMap<string, Type | null> | null;
    } = {},
  ): AgentPrompt {
    const { signal, localTypes = null } = options;
    const colliding = collidingVariable(this, inputs);
    if (colliding !== null) throw new RangeError(`variable ${reprText(colliding)} collides with an input name`);
    const variables = this.resolveVariables();
    const record = { ...inputs, ...variables } as ValueRecord;
    // The inputs' declared types, and each variable's contract.
    const types = new Map(localTypes ?? []);
    for (const variable of this.variables) types.set(variable.name, (variable.contract as ValueContract).type);
    this.renderMemories(record, types, inputs);
    const declared = localDeclared(types);
    const messages =
      typeof this.prompt === "string"
        ? [textMessage("user", renderTemplateRecord(this.prompt, record, declared), 0)]
        : renderPromptList(this.id, this.prompt, record, declared);
    const systemPrompt =
      this.systemPrompt === null ? DEFAULT_SYSTEM : renderTemplateRecord(this.systemPrompt, record, declared);
    signal?.throwIfAborted();
    const granted = this.grantedSkills(skills);
    return {
      // Built when called: forward builds them once it has checked for a stop, so a stopped node
      // fails as stopped before a text a model message refuses.
      modelMessages: () => messages.map(modelMessage),
      systemPrompt: systemPromptWithSkills(systemPrompt, granted),
      skills: granted,
    };
  }

  /** Resolve the validated literal defaults into one detached invocation record. */
  private resolveVariables(): ValueRecord {
    return copyValue(applyDefaults(this.variables, {}) as ValueRecord) as ValueRecord;
  }

  /**
   * Rebuild each request one accepted invocation sent: its prompt rendered with the recorded
   * skills (`renderPrompt`), then the requests read from its steps (`rebuildStepRequests`), each
   * checked against the recorded digest.
   *
   * An invocation with no `model_call` step sent no request. One whose render now fails (the run
   * rendered it before its first model call) has no requests either, and fails the digest check.
   */
  override async rebuildRequests(
    inputs: Readonly<Record<string, unknown>> | null,
    steps: readonly ValueRecord[],
    caps: RebuildCaps,
  ): Promise<RebuiltRequests> {
    const none = { requests: new Map(), digestOk: true };
    if (inputs === null || !steps.some((step) => step["kind"] === "model_call")) return none;
    // The run refused this render; any other RangeError while rebuilding is a fault.
    if (collidingVariable(this, inputs) !== null) return { ...none, digestOk: false };
    let rendered: AgentPrompt;
    try {
      rendered = this.renderPrompt(inputs, caps.skills, { localTypes: caps.localTypes });
    } catch (error) {
      if (!isRebuildRefusal(error)) throw error;
      return { ...none, digestOk: false };
    }
    const system = new ModelMessage({ role: "system", parts: [new TextPart(rendered.systemPrompt)] });
    // The memories' tools as the live loop built them, from the values the run recorded.
    const { tools: grants, memories } = this.memoryPlan(inputs);
    const local = new Map(
      [...memories].map(([input, value]) => [input, grantedMemorySpecs(value, { input, tools: this.grantedMemoryTools(input) })]),
    );
    const tools = caps.toolSpecs(grants, rendered.skills.length > 0 ? loadSkillTool() : null, local);
    const contract = this.outputContract;
    const structured = async () => {
      if (!isExactly(contract, ValueContract) || caps.models === null) return null;
      try {
        const config = this.callConfig(inputs) ?? {};
        const model = await caps.models.createModel(config, { nodeId: this.id, signal: new AbortController().signal });
        return { model, typ: contract.type };
      } catch (error) {
        // A TypeError is a fault in the builder, not a refused build.
        if (!(error instanceof Error) || error instanceof TypeError) throw error;
        // A model that can't be built now can't give the plan, so its structured requests can't
        // be checked: `null` fails the digest.
        return null;
      }
    };
    return await rebuildStepRequests(steps, {
      messages: [system, ...rendered.modelMessages()],
      tools,
      structured,
      renderSteer: this.steerRenderer(caps.localTypes ?? null),
    });
  }

  /** Return the canonical local declarations in their authored order. */
  override declaredVariables(): readonly Parameter[] {
    return [...this.variables];
  }
}

/** The name a grant gives: a tool or group's name, or the group of a group grant. */
function grantName(grant: ToolGrant): string {
  return grant.kind === "name" ? grant.name : grant.group;
}
