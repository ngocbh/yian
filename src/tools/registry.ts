/**
 * The tool registry and the tools one run can use.
 *
 * `registerTool` stores a host `FunctionTool` in `TOOL_REGISTRY`. A run's `ToolSet` reads this
 * registry when the host passes no `tools` mapping; the TOOL and AGENT nodes look tools up by id
 * through that `ToolSet`.
 *
 * This is the package's tool seam: the core ships no domain tools, and a host registers its own.
 * Tool ids must be unique across everything registered into the process. `ToolSet` holds one
 * run's tools: this registry, or a host mapping that replaces it and may hold named tool groups.
 * The reserved names never live here. `take` and `sleep` are engine built-ins a TOOL node reaches
 * with `tool_id:`; `load_skill` is bound by granting skills; each run gets its ask handler, which
 * serves both the agent's `ask_user` tool and the TOOL built-in `ask`, through
 * `new ToolSet({ ask: handler, ... })`.
 *
 * Knows about: `tools/context`, `tools/errors`, `tools/function_tools`, `tools/grants`,
 * `tools/mcp`, `tools/types`, `skills/tool` (the reserved `load_skill` name), `defaults`, typesys
 * and diagnostics.
 */

import { current } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { LOAD_SKILL } from "../skills/tool.ts";
import { type JsonObject, parseJson } from "../typesys/json.ts";
import { defaultRunId } from "../typesys/seeding.ts";
import {
  buildValueAs,
  fromHost,
  isRecord,
  jsTypeName,
  renderValuePath,
  toHost,
  TypeCheckError,
  type Value,
  type ValueRecord,
  writeAsText,
} from "../typesys/values.ts";
import { ASK, ASK_USER, isPending, SLEEP, TAKE, ToolContext, toolCallId } from "./context.ts";
import { isToolCheckError, ToolCheckError, toolErrorKinds } from "./errors.ts";
import { FunctionTool, isFunctionTool, type StandardIssue, type StandardResult } from "./function_tools.ts";
import { GrantError, type GrantScope, visibleToolName } from "./grants.ts";
import { isMcpServer, type McpServer } from "./mcp.ts";
import { inputTypeFromJsonSchema } from "./types.ts";

export { ASK, ASK_USER };

/** Maps tool id to the `FunctionTool` registered under it. */
export const TOOL_REGISTRY = new Map<string, FunctionTool>();

/** Names no host tool may take: the engine built-ins, the run's ask slot and the skills tool. */
export const RESERVED_TOOL_NAMES: ReadonlySet<string> = new Set([TAKE, SLEEP, ASK, ASK_USER, LOAD_SKILL]);
const ASK_HINT = "pass tools: {...registeredTools(), ask: handler} to runFlow or advance";

/** Return why a reserved name is reserved, for a refusal message. */
function reservedReason(name: string): string {
  if (name === TAKE || name === SLEEP) return `an engine built-in a TOOL node reaches with tool_id: ${name}`;
  if (name === ASK || name === ASK_USER) return ASK_HINT;
  return "granting skills binds it";
}

// Says what the tool is for rather than restricting it: a "use only when" rule kept models from
// calling it when the user asked to be asked something, so they drew question cards in text.
const ASK_USER_DESCRIPTION =
  "Ask the user a question and wait for their answer: when you need a decision or information " +
  "only they can give, or when they ask you to ask them something. Put the context they need " +
  "in one or two sentences. When there are clear choices, pass 2-4 short `options`; the user " +
  "can always type a different answer.";

// The schema LangChain's binding gives the model for `ask_user`, key order included, plus
// `timeout`.
const ASK_USER_PARAMETERS: JsonObject = {
  properties: {
    question: { type: "string" },
    options: {
      anyOf: [{ items: { type: "string" }, type: "array" }, { type: "null" }],
      default: null,
      description: "Optional 2-4 short answer choices. The user can always type a different answer.",
    },
    timeout: {
      anyOf: [{ type: "string" }, { type: "null" }],
      default: null,
      description:
        "Optional: how long to wait for an answer, such as 30m, 24h or 2d. After it the call returns 'timed out: no answer'.",
    },
  },
  required: ["question"],
  type: "object",
};

// Set by the agent from `timeout`, never by the model: checked on the trusted path only.
const ASK_USER_TRUSTED_PROPERTIES: JsonObject = {
  deadline: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
};

const ASK_DESCRIPTION = "Ask a person and wait for the answer.";

// yian's own `ask` surface: what a TOOL `ask` node may pass, before the host's extras.
const ASK_BASE_PROPERTIES: JsonObject = {
  prompt: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
  questions: { anyOf: [{ items: { type: "object" }, type: "array" }, { type: "null" }], default: null },
  // A literal duration arrives as integer seconds, a `${...}` as text.
  timeout: { anyOf: [{ type: "integer" }, { type: "string" }, { type: "null" }], default: null },
  __questions: { anyOf: [{ items: { type: "object" }, type: "array" }, { type: "null" }], default: null },
};

// Set by the TOOL node, never by an author: checked on the trusted path only.
const ASK_TRUSTED_PROPERTIES: JsonObject = {
  answerSchema: { anyOf: [{ items: { type: "object" }, type: "array" }, { type: "null" }], default: null },
  title: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
  nodeId: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
  deadline: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
};

/** Options for one tool call. */
export interface InvokeOptions {
  /** The expanded id of the calling node, passed as `ToolContext.node`; defaults to `""`. */
  node?: string;
  /**
   * The call's id. Absent, it is `"<runId>/<node>"` on a run-bound set given a `node`, and a fresh
   * random hex id otherwise.
   */
  callId?: string;
  /** The attempt number of this call, starting at 1; defaults to 1. */
  attempt?: number;
  /** The call's abort signal. Absent, the bound run's signal, else a signal that never aborts. */
  signal?: AbortSignal;
}

/** A tool by name: a top-level tool (`group: null`), or the tool `name` of group `group`. */
export interface ToolRef {
  readonly group: string | null;
  readonly name: string;
}

/** Whether `value` is a `ToolRef`: a `name` text and a `group` text or `null`. */
function isToolRef(value: unknown): value is ToolRef {
  if (typeof value !== "object" || value === null) return false;
  const { group, name } = value as { group?: unknown; name?: unknown };
  return typeof name === "string" && (group === null || typeof group === "string");
}

/** One tool call whose arguments `ToolSet.validate` checked; `invoke` and `isConcurrencySafe` take it. */
class CheckedCall {
  /** The tool's own name: its id, or a group's tool's name without its group. */
  readonly toolId: string;

  /** Build a frozen call; only `checkedCall` builds one, and records it in `CALLS`. */
  constructor(toolId: string) {
    this.toolId = toolId;
    Object.freeze(this);
  }
}

// A type only: a host gets a call from `validate`, so it can't forge one.
export type { CheckedCall };

/** What a `CheckedCall` stands for: the tool `validate` looked up and the argument `execute` gets. */
interface CallEntry {
  readonly tool: FunctionTool;
  /**
   * Return the argument for `execute`: the validator's output, the same object each time, or a
   * fresh host copy of the checked arguments when the tool has no validator.
   */
  readonly args: () => Record<string, unknown>;
}

// Keyed by the call, so nothing host-visible holds the validator's output.
const CALLS = new WeakMap<CheckedCall, CallEntry>();

/** The request a run's `ask_user` handler receives; an absent field is `null`. */
export interface AskUserRequest {
  question: string;
  options: string[] | null;
  questions: unknown[] | null;
  answerSchema: unknown[] | null;
  title: string | null;
  nodeId: string | null;
}

/** A run's `ask_user` handler: takes the request and the call's context, returns the answer. */
export type AskUserHandler = (request: AskUserRequest, ctx: ToolContext<any>) => unknown;

/** What a run's ask handler receives: the `ask_user` request plus the call's identity. */
export interface AskRequest extends AskUserRequest {
  /** The call's id, the same on every re-run of a TOOL `ask`; a host dedupes by it. */
  callId: string;
  /** When the call times out (a record time), for a TOOL `ask` with a `timeout`; else null. */
  deadline: string | null;
  /** The host-declared extra inputs the TOOL `ask` passed, by name; `{}` when none. */
  extra: Record<string, unknown>;
}

/** A run's ask handler. It may return `pending()`; the TOOL `ask` or the agent then waits for `complete`. */
export type AskHandler = (request: AskRequest, ctx: ToolContext<any>) => unknown;

/** An ask handler with the extra inputs it accepts on a TOOL `ask`, as JSON Schema properties. */
export interface AskTool {
  handler: AskHandler;
  inputs: JsonObject;
  required?: readonly string[];
}

/** One entry of a host `tools` mapping. */
export type ToolEntry = FunctionTool | McpServer | readonly FunctionTool[] | AskHandler | AskTool;

/**
 * Register a `FunctionTool` under `toolId`.
 *
 * The tool keeps its own schema and description unless `descriptionOverride` replaces the
 * description, which stores a copy and leaves `tool` unchanged.
 *
 * @param toolId The unique id to register under; the tool's name must equal it.
 * @param tool The tool to register.
 * @param options `descriptionOverride` is the description the model sees; `""`, `null` and
 *   `undefined` keep the tool's own.
 * @returns `tool`, unchanged.
 * @throws {RangeError} `toolId` is a reserved name (`take`, `sleep`, `ask`, `ask_user`,
 *   `load_skill`), is already registered, or differs from the tool's name.
 * @throws {TypeError} `tool` is not a `FunctionTool`, or `descriptionOverride` is not a string.
 *
 * @example
 * registerTool("add", new FunctionTool({
 *   name: "add",
 *   description: "Add two integers.",
 *   parametersJsonSchema: {
 *     type: "object",
 *     properties: { a: { type: "integer" }, b: { type: "integer" } },
 *     required: ["a", "b"],
 *   },
 *   execute: (args) => (args.a as number) + (args.b as number),
 * }));
 */
export function registerTool(
  toolId: string,
  tool: FunctionTool,
  options?: { descriptionOverride?: string },
): FunctionTool {
  if (RESERVED_TOOL_NAMES.has(toolId)) {
    throw new RangeError(`tool name ${reprText(toolId)} is reserved by yian: ${reservedReason(toolId)}`);
  }
  if (TOOL_REGISTRY.has(toolId)) throw new RangeError(`Duplicate tool id '${toolId}'`);
  if (!isFunctionTool(tool)) {
    throw new TypeError(
      `registerTool(${reprText(toolId)}) got a ${jsTypeName(tool)}; it accepts a FunctionTool. ` +
        "Pass anything else in the run's tools mapping instead",
    );
  }
  if (tool.name !== toolId) {
    throw new RangeError(
      `tool id ${reprText(toolId)} names a FunctionTool named ${reprText(tool.name)}; its name must match its id`,
    );
  }
  const description: unknown = options?.descriptionOverride;
  if (description !== undefined && description !== null && typeof description !== "string") {
    throw new TypeError(`registerTool(${reprText(toolId)}) descriptionOverride must be a string`);
  }
  TOOL_REGISTRY.set(
    toolId,
    description
      ? new FunctionTool({
          name: tool.name,
          description,
          parametersJsonSchema: tool.parametersJsonSchema,
          execute: tool.execute,
          isConcurrencySafe: tool.isConcurrencySafe,
          // A tool from an older copy of yian (the brand check accepts one) lacks the next three
          // fields; `FunctionTool` stores their `undefined` as `null`.
          parametersValidator: tool.parametersValidator,
          outputJsonSchema: tool.outputJsonSchema,
          outputValidator: tool.outputValidator,
        })
      : tool,
  );
  return tool;
}

/**
 * Return a copy of the tools registered with `registerTool`.
 *
 * A run given `tools` sees only those tools. A host that wants the registered tools as well builds
 * the mapping itself, for example `{...registeredTools(), ask: handler}`. Changing the copy
 * does not change the registry. The copy is taken when you call this, so register every tool the
 * run needs before the run starts.
 *
 * @returns A new record of tool id to tool.
 */
export function registeredTools(): Record<string, FunctionTool> {
  return Object.fromEntries(TOOL_REGISTRY);
}

/**
 * Look registered tools up by id.
 *
 * @param ids The ids to resolve, in order.
 * @returns The matching tools, in the same order as `ids`.
 * @throws {RangeError} An id is not registered.
 */
export function resolveTools(ids: readonly string[]): FunctionTool[] {
  return ids.map((id) => {
    const tool = TOOL_REGISTRY.get(id);
    if (tool === undefined) {
      throw new RangeError(`Unknown tool id '${id}'. Known tools: ${writeAsText(sortedIds(TOOL_REGISTRY))}`);
    }
    return tool;
  });
}

/**
 * Clean model-sent `ask_user` options; any shape is accepted, none throws.
 *
 * A JSON list string is parsed; any other string is one label. Non-string items are dropped.
 * Labels are collapsed to one line, empty and case-insensitive duplicate labels are dropped,
 * labels longer than `current().maxQuestionOptionChars` code points are cut, and at most
 * `current().maxQuestionOptions` are kept.
 *
 * @returns The cleaned labels, or `null` when none is left.
 */
export function normalizeOptions(value: unknown): string[] | null {
  if (typeof value === "string") {
    let parsed: unknown = null;
    try {
      parsed = parseJson(value, "plain");
    } catch {
      // Not JSON: the string is one label.
    }
    value = Array.isArray(parsed) ? parsed : [value];
  }
  if (!Array.isArray(value)) return null;
  const caps = current();
  const labels: string[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string") continue;
    let label = item.split(/\s+/).filter((part) => part !== "").join(" ");
    // Code points, not UTF-16 units.
    const chars = Array.from(label);
    if (chars.length > caps.maxQuestionOptionChars) {
      label = chars.slice(0, caps.maxQuestionOptionChars - 1).join("") + "…";
    }
    // Upper then lower case folds case: "Straße" and "STRASSE" are the same label.
    const key = label.toUpperCase().toLowerCase();
    if (label === "" || seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
    if (labels.length === caps.maxQuestionOptions) break;
  }
  return labels.length > 0 ? labels : null;
}

/** Return a map's keys sorted by code unit. */
function sortedIds(tools: ReadonlyMap<string, unknown>): string[] {
  return [...tools.keys()].sort(compareCodeUnits);
}

/**
 * Wrap a host handler in the fixed `ask_user` tool.
 *
 * The model sees `question`, optional `options` and optional `timeout`. The handler receives the
 * request fields, absent ones `null`, the call's id, the `deadline` the agent computed from
 * `timeout` (on the trusted path; `null` without one), no extras, and the call's context.
 */
function askUserTool(handler: AskHandler): FunctionTool {
  return new FunctionTool({
    name: ASK_USER,
    description: ASK_USER_DESCRIPTION,
    parametersJsonSchema: ASK_USER_PARAMETERS,
    execute: (args, ctx) =>
      handler(
        {
          question: args.question as string,
          options: (args.options ?? null) as string[] | null,
          questions: null,
          answerSchema: null,
          title: null,
          nodeId: null,
          callId: ctx.callId,
          // The agent turned the model's `timeout` into `deadline`; the handler never sees the text.
          deadline: (args.deadline ?? null) as string | null,
          extra: {},
        },
        ctx,
      ),
  });
}

/**
 * Wrap a host ask handler in the TOOL built-in `ask`.
 *
 * Its schema is yian's base surface plus the host's extras. The TOOL node has already parsed
 * `questions` and turned `timeout` into `deadline`, so the handler gets the request fields, the
 * extras the call passed in `extra`, and the call's context.
 */
function askTool(spec: AskTool): FunctionTool {
  const extras = Object.keys(spec.inputs);
  return new FunctionTool({
    name: ASK,
    description: ASK_DESCRIPTION,
    parametersJsonSchema: {
      properties: { ...ASK_BASE_PROPERTIES, ...spec.inputs },
      required: [...(spec.required ?? [])],
      type: "object",
    },
    execute: (args, ctx) => {
      const extra: Record<string, unknown> = {};
      for (const key of extras) if (Object.hasOwn(args, key)) extra[key] = args[key];
      return spec.handler(
        {
          question: (args.prompt ?? "") as string,
          options: null,
          questions: (args.questions ?? null) as unknown[] | null,
          answerSchema: (args.answerSchema ?? null) as unknown[] | null,
          title: (args.title ?? null) as string | null,
          nodeId: (args.nodeId ?? null) as string | null,
          callId: ctx.callId,
          deadline: (args.deadline ?? null) as string | null,
          extra,
        },
        ctx,
      );
    },
  });
}

/**
 * Return the run's ask slot entry, passed as `tools[key]`, as an `AskTool`: a function is a handler
 * with no extras.
 *
 * @throws {TypeError} The entry is neither a function nor `{handler, inputs, required?}`.
 * @throws {RangeError} An input shadows yian's ask surface, or a required name isn't an input.
 */
function askToolOf(key: string, entry: unknown): AskTool {
  const where = `tools[${JSON.stringify(key)}]`;
  if (typeof entry === "function") return { handler: entry as AskHandler, inputs: {}, required: [] };
  // A FunctionTool has no `handler`, so this also refuses one.
  const record = isRecord(entry) ? (entry as Record<string, unknown>) : null;
  const required = record?.required;
  if (
    record === null ||
    typeof record.handler !== "function" ||
    !isRecord(record.inputs) ||
    !(required === undefined || (Array.isArray(required) && required.every((name) => typeof name === "string")))
  ) {
    throw new TypeError(
      `${where} must be a handler function or {handler, inputs, required?}; yian supplies the base schema`,
    );
  }
  const inputs = record.inputs as JsonObject;
  for (const name of Object.keys(inputs)) {
    if (Object.hasOwn(ASK_BASE_PROPERTIES, name) || Object.hasOwn(ASK_TRUSTED_PROPERTIES, name)) {
      throw new RangeError(
        `${where} input ${reprText(name)} is part of yian's ask surface; name the extra input differently`,
      );
    }
  }
  for (const name of (required ?? []) as string[]) {
    if (!Object.hasOwn(inputs, name)) {
      throw new RangeError(`${where} requires ${reprText(name)}, which isn't one of its inputs`);
    }
  }
  return { handler: record.handler as AskHandler, inputs, required: (required ?? []) as string[] };
}

/** Return the items of the array passed as `tools[group]`, each checked to be a `FunctionTool`. */
function groupTools(group: string, items: readonly unknown[]): FunctionTool[] {
  const tools: FunctionTool[] = [];
  for (let index = 0; index < items.length; index++) {
    const item = items[index];
    const where = `tools[${JSON.stringify(group)}][${index}]`;
    if (isMcpServer(item)) {
      throw new TypeError(`${where} is an MCP server; pass it as its own tools entry, which becomes its group`);
    }
    if (!isFunctionTool(item)) throw new TypeError(`${where} must be a FunctionTool, got ${jsTypeName(item)}`);
    tools.push(item);
  }
  return tools;
}

/**
 * Return the tools of the MCP server passed as `tools[key]`, in the order it listed them.
 *
 * @throws {TypeError} The server isn't open.
 * @throws {RangeError} The server was open but has gone away.
 */
function serverTools(key: string, server: McpServer): readonly FunctionTool[] {
  if (!server.isOpen) {
    throw new TypeError(
      `tools[${JSON.stringify(key)}] is an MCP server that isn't open; open it with "await server.open()" ` +
        "or pass it to runFlow/advance, which open it",
    );
  }
  if (server.isClosed) throw new RangeError(`tools[${JSON.stringify(key)}]: MCP server '${server.label}' is closed`);
  return groupTools(key, server.tools);
}

/**
 * The tools one run can use.
 *
 * With no mapping, the run uses the global registry, read at lookup time. With a mapping, even an
 * empty one, the run sees only that mapping's entries; there is no fallback to the registry. Two
 * runs with different `ToolSet`s never see each other's entries.
 *
 * An entry whose value is an array of `FunctionTool`s, or an open `McpServer`, is a tool group.
 * Groups are namespaced: a group's tool is `{group, name}`, and two groups may hold different
 * tools of one name. A bare name finds a top-level tool, else the one group tool by that name. The
 * same tool object may sit in several groups; its record name is `<group>_<tool>` under the first
 * group that holds it. A tool may show to a model as its top-level name or as `<group>_<tool>`,
 * and two different tools that could show one name are refused. A group tool may not share its
 * name with a top-level key or take a reserved name (`take`, `sleep`, `ask`, `ask_user`,
 * `load_skill`). `scope()` is this set as a `GrantScope`, which `expandGrants` reads.
 *
 * A tool's `execute(args, ctx)` receives a `ToolContext` on each call: the call id, attempt, tool
 * name, run id, calling node, this set's `context` and the call's abort signal.
 *
 * `validate` checks a call's arguments, runs the tool's validator once, and returns a
 * `CheckedCall`; `invoke` takes that call, or a tool id and arguments to check first, and checks
 * the result of a tool that declares an output; `isConcurrencySafe` takes a call. `forRun` binds
 * a copy to one run's id and abort signal.
 *
 * @param tools The run's tools keyed by tool id. A `FunctionTool` must use its key as its name.
 *   An array is a group keyed by its group name; each item is a `FunctionTool` named by its
 *   `name`. An open `McpServer` is a group of the tools it listed when it opened. The `ask` entry
 *   (or `ask_user`, its old name) is the run's ask slot: an `AskHandler` `(request, ctx)`, or an
 *   `AskTool` `{handler, inputs, required?}` declaring extra inputs a TOOL `ask` may pass. The one
 *   handler serves the agent's `ask_user` tool, whose model surface is `question` and `options`,
 *   and the TOOL built-in `ask`; it receives an `AskRequest` and the call's context. `take`,
 *   `sleep` and `load_skill` are not keys a host may pass. `null` or `undefined` means the global
 *   registry. To keep the registered tools and add more, pass `{...registeredTools(), ...}`.
 * @param options `context` is the host's app object, passed to tools as `ToolContext.context`. It
 *   is never recorded.
 * @throws {TypeError} `tools` is not a plain object, `null` or `undefined`; a key is empty; an
 *   entry is not a `FunctionTool`, an `McpServer` or an array of `FunctionTool`s; a group item is
 *   an `McpServer` or not a `FunctionTool`; an `McpServer` entry isn't open; the ask entry is
 *   neither a function nor an `AskTool`; or both `ask` and `ask_user` are given.
 * @throws {RangeError} A key is `take`, `sleep` or `load_skill`; a `FunctionTool` entry's name
 *   differs from its key; an `McpServer` entry was open but has gone away; an `AskTool` input
 *   shadows yian's ask surface or a required name isn't an input; a group tool's name is
 *   reserved or is a top-level key; a group holds two different tools of one name; or two
 *   different tools could show one name.
 *
 * @example
 * const tools = new ToolSet({ toolbox: [weather, forecast] }); // two FunctionTools
 * tools.scope().group("toolbox"); // ["weather", "forecast"]
 * tools.resolve([{ group: "toolbox", name: "weather" }]); // [weather]
 * const runTools = new ToolSet({ place }, { context: broker }).forRun("run-1");
 * const call = await runTools.validate("place", { sym: "X" });
 * const parallel = runTools.isConcurrencySafe(call); // true: it may run beside other calls
 * await runTools.invoke(call, { node: "buy" }); // runs either way
 */
export class ToolSet {
  // Plain private fields, not `#` ones, so `forRun` can copy them into a subclass instance.
  private toolMap: ReadonlyMap<string, FunctionTool> | null;
  private groupMap: ReadonlyMap<string, ReadonlyMap<string, FunctionTool>>;
  // Each group tool's record name: `<group>_<tool>` under the first group holding the object.
  private recordNames: ReadonlyMap<FunctionTool, string>;
  private boundRunId: string;
  private boundSignal: AbortSignal | undefined;
  private hostContext: unknown;
  // The full schemas the trusted path checks the ask tools against; any other tool uses its own.
  private trustedSchemas: ReadonlyMap<string, JsonObject>;

  constructor(tools?: Readonly<Record<string, ToolEntry>> | null, options?: { context?: unknown }) {
    this.toolMap = null;
    this.groupMap = new Map();
    this.recordNames = new Map();
    this.boundRunId = "";
    this.boundSignal = undefined;
    this.hostContext = options?.context;
    this.trustedSchemas = new Map();
    if (tools === undefined || tools === null) return;
    if (!isRecord(tools)) throw new TypeError("tools must be a record of tool id to tool, or null");
    const keys = Object.keys(tools);
    if (Object.hasOwn(tools, ASK) && Object.hasOwn(tools, ASK_USER)) {
      throw new TypeError('pass the ask handler once, as tools["ask"] (tools["ask_user"] is its old name)');
    }
    const built = new Map<string, FunctionTool>();
    const groupItems = new Map<string, readonly FunctionTool[]>();
    const trusted = new Map<string, JsonObject>();
    for (const key of keys) {
      if (key === "") throw new TypeError("tool ids in tools must be non-empty strings");
      const entry: unknown = tools[key];
      if (key === ASK || key === ASK_USER) {
        const spec = askToolOf(key, entry);
        const ask = askTool(spec);
        built.set(ASK_USER, askUserTool(spec.handler));
        built.set(ASK, ask);
        const properties = { ...(ask.parametersJsonSchema.properties as JsonObject), ...ASK_TRUSTED_PROPERTIES };
        trusted.set(ASK, { ...ask.parametersJsonSchema, properties });
        const userProperties = { ...(ASK_USER_PARAMETERS.properties as JsonObject), ...ASK_USER_TRUSTED_PROPERTIES };
        trusted.set(ASK_USER, { ...ASK_USER_PARAMETERS, properties: userProperties });
      } else if (RESERVED_TOOL_NAMES.has(key)) {
        throw new RangeError(`tools[${JSON.stringify(key)}]: the name is reserved by yian: ${reservedReason(key)}`);
      } else if (isFunctionTool(entry)) {
        if (entry.name !== key) {
          throw new RangeError(
            `tool ${reprText(key)} is a FunctionTool named ${reprText(entry.name)}; its name must match its key`,
          );
        }
        built.set(key, entry);
      } else if (isMcpServer(entry)) {
        groupItems.set(key, serverTools(key, entry));
      } else if (Array.isArray(entry)) {
        groupItems.set(key, groupTools(key, entry));
      } else {
        throw new TypeError(
          `tools[${JSON.stringify(key)}] must be a FunctionTool, an MCP server, or an array of FunctionTools, ` +
            `got ${jsTypeName(entry)}`,
        );
      }
    }
    const topLevel = new Set(keys);
    const groups = new Map<string, ReadonlyMap<string, FunctionTool>>();
    const recordNames = new Map<FunctionTool, string>();
    // Every name a tool could show, with the tool and how a message names it.
    const shown = new Map<string, { item: FunctionTool; text: string }>();
    for (const [name, item] of built) shown.set(name, { item, text: `tool ${reprText(name)}` });
    for (const [group, items] of groupItems) {
      const members = new Map<string, FunctionTool>();
      for (const item of items) {
        const name = item.name;
        if (RESERVED_TOOL_NAMES.has(name)) {
          throw new RangeError(
            `group ${reprText(group)} has a tool named ${reprText(name)}, a reserved name; ${reservedReason(name)}`,
          );
        }
        if (topLevel.has(name)) {
          throw new RangeError(
            `tool ${reprText(name)} in group ${reprText(group)} clashes with tools[${JSON.stringify(name)}]`,
          );
        }
        const had = members.get(name);
        if (had !== undefined && had !== item) {
          const where = `appears twice in group ${reprText(group)}`;
          throw new RangeError(`tool ${reprText(name)} ${where} as different tools`);
        }
        if (had !== undefined) continue;
        members.set(name, item);
        // Host permission groups overlap, so one tool may sit in several of them; it keeps the
        // name of the first.
        if (!recordNames.has(item)) recordNames.set(item, visibleToolName(group, name));
        const visible = visibleToolName(group, name);
        const text = `${group}'s ${reprText(name)}`;
        if (RESERVED_TOOL_NAMES.has(visible)) {
          throw new RangeError(
            `${text} could show as ${reprText(visible)}, a reserved name; rename one`,
          );
        }
        const other = shown.get(visible);
        if (other !== undefined && other.item !== item) {
          throw new RangeError(
            `${other.text} and ${text} could both show as ${reprText(visible)}; rename one`,
          );
        }
        shown.set(visible, { item, text });
      }
      groups.set(group, members);
    }
    this.toolMap = built;
    this.groupMap = groups;
    this.recordNames = recordNames;
    this.trustedSchemas = trusted;
  }

  /**
   * Return this run's tool, or the registry's when the run has no mapping.
   *
   * A `ToolRef` with a group names that group's tool. A bare name is a top-level tool, else the one
   * group tool by that name. `ask` and `ask_user` never come from the registry: a run without its
   * own handler cannot ask. `take` and `sleep` are engine built-ins, never tools.
   *
   * @throws {GrantError} A bare name is a tool of two groups.
   * @throws {RangeError} No tool answers the ref.
   */
  private lookup(ref: string | ToolRef): FunctionTool {
    if (typeof ref !== "string" && ref.group !== null) {
      const members = this.groupMap.get(ref.group);
      if (members === undefined) {
        const known = `tool groups in this run: [${this.groups.join(", ")}]`;
        throw new RangeError(`unknown tool group ${reprText(ref.group)}; ${known}`);
      }
      const tool = members.get(ref.name);
      if (tool === undefined) {
        const has = members.size === 0 ? "no tools" : [...members.keys()].join(", ");
        throw new RangeError(`\`${ref.group}\` has no tool \`${ref.name}\`; it has ${has}`);
      }
      return tool;
    }
    const toolId = typeof ref === "string" ? ref : ref.name;
    if ((toolId === ASK || toolId === ASK_USER) && !this.toolMap?.has(toolId)) {
      throw new RangeError(`no ask handler for this run; ${ASK_HINT}`);
    }
    if (toolId === TAKE || toolId === SLEEP) {
      throw new RangeError(
        `${reprText(toolId)} is an engine built-in, not a tool; reach it with a TOOL node's tool_id: ${toolId}`,
      );
    }
    const group = this.groupMap.get(toolId);
    if (group !== undefined) {
      const [first] = group.keys();
      const one = `; name one of its tools, as \`{${toolId}: ${first}}\``;
      const what = first === undefined ? ", and it has no tools" : one;
      throw new RangeError(`${reprText(toolId)} is a tool group, not a tool${what}`);
    }
    const tool = (this.toolMap ?? TOOL_REGISTRY).get(toolId);
    if (tool !== undefined) return tool;
    const holders = this.holders(toolId);
    if (holders.length > 1) {
      const groups = holders.map(({ group }) => reprText(group));
      const shown = `${groups.slice(0, -1).join(", ")} and ${groups.at(-1)!}`;
      const fix = `name one as \`{${holders[0]!.group}: ${toolId}}\``;
      throw new GrantError(`${reprText(toolId)} is a tool of groups ${shown}; ${fix}`);
    }
    if (holders.length === 1) return holders[0]!.tool;
    const known = writeAsText([...this.toolIds]);
    throw new RangeError(`unknown tool id ${reprText(toolId)}; known tools: ${known}`);
  }

  /** The group tools named `name`, one per distinct tool, each under the first group holding it. */
  private holders(name: string): { group: string; tool: FunctionTool }[] {
    const found: { group: string; tool: FunctionTool }[] = [];
    for (const [group, members] of this.groupMap) {
      const tool = members.get(name);
      if (tool === undefined || found.some((each) => each.tool === tool)) continue;
      found.push({ group, tool });
    }
    return found;
  }

  /**
   * Return a granted tool's record name: a top-level tool's own name, or `<group>_<tool>` under
   * the first group holding a group tool.
   *
   * @throws {RangeError} No tool answers the ref, as `resolve` throws.
   */
  recordName(ref: ToolRef): string {
    const tool = this.lookup(ref);
    return this.recordNames.get(tool) ?? ref.name;
  }

  /**
   * This set as the run's grant scope: its groups by name, its top-level tools, bare names reaching
   * group tools, and record names as identities.
   */
  scope(): GrantScope {
    return {
      label: "this run",
      bareMembers: true,
      group: (name) => {
        const members = this.groupMap.get(name);
        return members === undefined ? null : [...members.keys()];
      },
      hasTool: (name) => {
        // `ask` and `ask_user` are tools only with the run's own handler.
        if (name === ASK || name === ASK_USER) return this.toolMap?.has(name) === true;
        return (this.toolMap ?? TOOL_REGISTRY).has(name);
      },
      groupsHolding: (name) => this.holders(name).map(({ group }) => group),
      groupNames: () => this.groups,
      identity: (group, tool) => (group === null ? tool : this.recordName({ group, name: tool })),
    };
  }

  /** The scopes this set's grants resolve in, in order: only `scope()`. */
  scopes(): GrantScope[] {
    return [this.scope()];
  }

  /**
   * Return a copy bound to one run, sharing this set's tools, groups and context.
   *
   * @param runId The run's id, passed to tools as `ToolContext.runId` and the base of each call's
   *   default `callId`.
   * @param options `signal` is the run's abort signal, aborted when the run stops. The copy holds
   *   exactly this value: none gives no signal, even when this set was bound to one.
   * @returns A new tool set of the same class holding this one's own fields, a subclass's
   *   included, with the run bound; this one stays as it was.
   */
  forRun(runId: string, options?: { signal?: AbortSignal }): this {
    const bound = Object.assign(Object.create(Object.getPrototypeOf(this)), this) as this;
    bound.boundRunId = runId;
    bound.boundSignal = options?.signal;
    return bound;
  }

  /** The bound run's id, or `""` when `forRun` has not bound one. */
  get runId(): string {
    return this.boundRunId;
  }

  /** The bound run's abort signal, or `undefined` when `forRun` has not bound one. */
  get signal(): AbortSignal | undefined {
    return this.boundSignal;
  }

  /** The host's app object passed to tools as `ToolContext.context`. */
  get context(): unknown {
    return this.hostContext;
  }

  /** The names of this run's tool groups, in mapping order. */
  get groups(): readonly string[] {
    return [...this.groupMap.keys()];
  }

  /**
   * The sorted ids of the tools this run can resolve, for messages: with a mapping, the mapping's
   * top-level tools and `{<group>: <tool>}` for each group's tools, as a TOOL node names one; with
   * no mapping, the global registry's tools.
   */
  get toolIds(): readonly string[] {
    if (this.toolMap === null) return sortedIds(TOOL_REGISTRY);
    const members = [...this.groupMap].flatMap(([group, tools]) =>
      [...tools.keys()].map((name) => `{${group}: ${name}}`),
    );
    return [...new Set([...this.toolMap.keys(), ...members])].sort(compareCodeUnits);
  }

  /**
   * Look tools up by id or ref.
   *
   * @param ids The bare names or refs to resolve, in order.
   * @returns The matching tools, in the same order as `ids`.
   * @throws {GrantError} A bare name is a tool of two groups.
   * @throws {RangeError} An id is not visible to this run.
   */
  resolve(ids: readonly (string | ToolRef)[]): FunctionTool[] {
    return ids.map((id) => this.lookup(id));
  }

  /**
   * Check one call's public arguments, without calling the tool, and return the checked call.
   *
   * yian checks the arguments against the tool's JSON Schema first. When the tool has a
   * `parametersValidator`, the validator then runs once, here, and every `invoke` of the returned
   * call gives `execute` its output; a caller that retries a call validates it once up front, so a
   * bad argument fails before the first attempt instead of inside the retries.
   *
   * @param toolId The id of the tool the call is for, or a group tool's ref.
   * @param inputs Public arguments keyed by parameter name.
   * @param options `signal` stops the call before and after the validator runs; absent, the
   *   bound run's signal.
   * @returns The checked call, for `invoke` and `isConcurrencySafe`.
   * @throws {RangeError} The id is not visible to this run, an argument is not in the schema, or
   *   a required argument is missing.
   * @throws {TypeError} `inputs` is not a record, or the validator returned no Standard Schema
   *   result.
   * @throws {TypeCheckError} An argument is not a yian value, or doesn't fit its property's type.
   * @throws {ToolCheckError} The validator reported issues.
   * @throws The signal's reason when it aborts, and whatever the validator throws, unchanged.
   *
   * @example
   * const tools = new ToolSet({ weather: lookup });
   * const call = await tools.validate("weather", { city: "Paris" }); // lookup is not called
   * await tools.invoke(call, { attempt: 1 });
   * await tools.validate("weather", { town: "Paris" }); // rejects with a RangeError
   */
  async validate(
    toolId: string | ToolRef,
    inputs: Record<string, unknown>,
    options?: { signal?: AbortSignal },
  ): Promise<CheckedCall> {
    return await this.checkedCall(toolId, inputs, false, options?.signal);
  }

  /**
   * Check one call as `validate` does, against the tool's full schema, host-injected fields
   * included; only the ask tools have any (`ask_user`'s `deadline`, which the agent sets from the
   * model's `timeout`).
   *
   * @returns The checked call, for `invoke` and `isConcurrencySafe`.
   * @throws As `validate` throws, against the full schema.
   */
  async validateTrusted(
    toolId: string | ToolRef,
    inputs: Record<string, unknown>,
    options?: { signal?: AbortSignal },
  ): Promise<CheckedCall> {
    return await this.checkedCall(toolId, inputs, true, options?.signal);
  }

  /**
   * Call one tool: a call `validate` returned, or a tool id and its public arguments.
   *
   * `invoke(call, options)` runs the tool `validate` looked up, with this set's context, run id
   * and signal; it checks no argument again. `invoke(toolId, inputs, options)` validates the
   * arguments first, as `validate` does, then calls the tool. Errors from the argument checks and
   * from the tool propagate unchanged; a tool that throws a non-`Error` rejects with an `Error`
   * wrapping it.
   *
   * When the tool declares an output (`outputJsonSchema` or `outputValidator`), the result must be
   * a yian value; it is checked against the type the output schema translates to, read open (an
   * object's fields are checked only when the schema sets `additionalProperties: false`), then
   * the output validator runs on it and its output must be a yian value too. `invoke` returns that
   * value in host form. A tool without an output returns its result as it is.
   *
   * @param call A call `validate` returned, on this set or any other.
   * @param toolId The id of the tool to call, or a group tool's ref.
   * @param inputs Public arguments keyed by parameter name.
   * @param options The calling node, call id, attempt and abort signal the tool's context carries.
   * @returns The tool's result, awaited, `undefined` read as `null`; for a tool that declares an
   *   output, the checked value in host form. A `pending()` marker comes back unchanged, unchecked.
   * @throws {TypeError} The first argument is neither a tool id nor a `CheckedCall`; and as
   *   `validate` throws.
   * @throws {ToolCheckError} A declared result is not a yian value or doesn't fit the output
   *   schema's type; or the output validator threw (the error is the `cause`), returned no
   *   Standard Schema result, reported issues or gave a value that isn't a yian value.
   * @throws Whatever the tool throws: an `Error` unchanged, any other value wrapped in an `Error`
   *   whose message is its text and whose `cause` is the value.
   */
  invoke(call: CheckedCall, options?: InvokeOptions): Promise<unknown>;
  invoke(
    toolId: string | ToolRef,
    inputs: Record<string, unknown>,
    options?: InvokeOptions,
  ): Promise<unknown>;
  async invoke(
    target: CheckedCall | string | ToolRef,
    inputsOrOptions?: Record<string, unknown> | InvokeOptions,
    options?: InvokeOptions,
  ): Promise<unknown> {
    if (typeof target === "string" || (!(target instanceof CheckedCall) && isToolRef(target))) {
      const inputs = inputsOrOptions as Record<string, unknown>;
      return await this.invoke(await this.validate(target, inputs, { signal: options?.signal }), options);
    }
    const entry = CALLS.get(target);
    if (entry === undefined) throw new TypeError("invoke takes a tool id or a CheckedCall from validate");
    return await this.call(entry.tool, target.toolId, entry.args(), inputsOrOptions as InvokeOptions | undefined);
  }

  /**
   * Check and call one tool through its full schema, including host-injected fields.
   *
   * Only the ask tools have injected fields: `ask_user` takes `deadline`, and `ask` takes
   * `answerSchema`, `title`, `nodeId` and `deadline`. Every other tool
   * is checked as `invoke` checks it, its validator included.
   *
   * @param toolId The id of the tool to call.
   * @param inputs Public and injected arguments keyed by parameter name.
   * @param options The calling node, call id, attempt and abort signal the tool's context carries.
   * @returns The tool's result, awaited, `undefined` read as `null`; for a tool that declares an
   *   output, the checked value in host form.
   * @throws {RangeError} The id is not visible to this run, an argument is not in the full schema,
   *   or a required argument is missing.
   * @throws {TypeError} `inputs` is not a record, or the validator returned no Standard Schema
   *   result.
   * @throws {TypeCheckError} An argument is not a yian value, or doesn't fit its property's type.
   * @throws {ToolCheckError} The validator reported issues, or a declared result failed its checks,
   *   a throw from the output validator included.
   * @throws Whatever the tool or the validator throws, unchanged.
   */
  async invokeTrusted(
    toolId: string | ToolRef,
    inputs: Record<string, unknown>,
    options?: InvokeOptions,
  ): Promise<unknown> {
    return await this.invoke(await this.checkedCall(toolId, inputs, true, options?.signal), options);
  }

  /**
   * Return whether a checked call may run beside other calls.
   *
   * `true` only when the tool's `isConcurrencySafe` returns exactly `true` for the argument
   * `execute` gets. It never throws: no such function, a value that isn't a `CheckedCall`, a
   * throw, a Promise or any other result give `false`.
   */
  isConcurrencySafe(call: CheckedCall): boolean {
    try {
      const entry = CALLS.get(call);
      if (entry === undefined) return false;
      const check = entry.tool.isConcurrencySafe;
      if (check === null) return false;
      const safe: unknown = check(entry.args());
      if (typeof (safe as { then?: unknown } | null)?.then === "function") {
        // A Promise isn't `true`; handle its rejection so it can't crash the process.
        Promise.resolve(safe).catch(() => {});
        return false;
      }
      return safe === true;
    } catch {
      // A call whose safety can't be read runs alone.
      return false;
    }
  }

  /**
   * Check one call as `validate` describes, against the trusted schema when `trusted` is set, and
   * return the call recorded with its tool and argument.
   */
  private async checkedCall(
    ref: string | ToolRef,
    inputs: Record<string, unknown>,
    trusted: boolean,
    signal: AbortSignal | undefined = this.boundSignal,
  ): Promise<CheckedCall> {
    const [tool, payload] = this.check(ref, inputs, trusted);
    // A tool's context and messages name it by its own name, as it names itself.
    const toolId = typeof ref === "string" ? ref : ref.name;
    // `??`: a tool built by an older copy of yian has no `parametersValidator` field.
    const validator = tool.parametersValidator ?? null;
    let args: () => Record<string, unknown>;
    if (validator === null) {
      args = () => toHost(payload) as Record<string, unknown>;
    } else {
      signal?.throwIfAborted();
      const returned: unknown = await validator["~standard"].validate(toHost(payload));
      // The validator may be slow; a run that stopped meanwhile doesn't call the tool.
      signal?.throwIfAborted();
      const result = standardResult(returned, `tool ${reprText(toolId)} arguments: the validator`, TypeError);
      if (result.issues !== undefined) {
        throw new ToolCheckError(
          validatorFailure(`tool ${reprText(toolId)} arguments failed its validator`, result.issues),
        );
      }
      // The validator's output is the tool's own argument type, not necessarily a record.
      const value = result.value as Record<string, unknown>;
      args = () => value;
    }
    const call = new CheckedCall(toolId);
    CALLS.set(call, { tool, args });
    return call;
  }

  /**
   * Look the tool up and check one call's arguments; return the tool and the checked arguments
   * as a fresh yian record, `ask_user`'s `options` cleaned.
   */
  private check(
    ref: string | ToolRef,
    inputs: Record<string, unknown>,
    trusted: boolean,
  ): [FunctionTool, ValueRecord] {
    const tool = this.lookup(ref);
    // A group tool never takes a reserved name, so `ask` and `ask_user` here are the top-level
    // ones.
    const toolId = typeof ref === "string" ? ref : ref.name;
    const where = `tool ${reprText(toolId)} arguments`;
    const payload = fromHost(inputs, where);
    if (!isRecord(payload)) throw new TypeError(`${where} must be a record, got ${jsTypeName(inputs)}`);
    if (toolId === ASK_USER && Object.hasOwn(payload, "options")) payload.options = normalizeOptions(payload.options);
    const schema = trusted ? (this.trustedSchemas.get(toolId) ?? tool.parametersJsonSchema) : tool.parametersJsonSchema;
    const scope = toolId === ASK || toolId === ASK_USER ? (trusted ? "trusted" : "public") : null;
    checkArguments(toolId, schema, payload, scope);
    return [tool, payload];
  }

  /**
   * Call a checked tool with its argument and this call's context, then check the result when the
   * tool declares an output, as `invoke` describes.
   */
  private async call(
    tool: FunctionTool,
    toolId: string,
    args: Record<string, unknown>,
    options: InvokeOptions = {},
  ): Promise<unknown> {
    const node = options.node ?? "";
    const ctx = new ToolContext({
      // Without a run and a node nothing makes two calls the same call, so each gets its own id.
      callId: options.callId ?? (this.boundRunId && node ? toolCallId(this.boundRunId, node) : defaultRunId()),
      attempt: options.attempt ?? 1,
      toolName: toolId,
      runId: this.boundRunId,
      node,
      context: this.hostContext,
      signal: options.signal ?? this.boundSignal ?? new AbortController().signal,
    });
    let result: unknown;
    try {
      result = await tool.execute(args, ctx);
    } catch (error) {
      throw asError(error);
    }
    if (result === undefined) result = null;
    // A pending call has no result yet, so there is nothing to check.
    if (isPending(result)) return result;
    // `??`: a tool built by an older copy of yian has no output fields.
    const schema = tool.outputJsonSchema ?? null;
    const validator = tool.outputValidator ?? null;
    if (schema === null && validator === null) return result;
    const where = `tool ${reprText(toolId)} result`;
    // A failing result check is a `ToolCheckError`, which the TOOL node never retries.
    const intake = (host: unknown): Value => {
      try {
        return fromHost(host, where);
      } catch (error) {
        // `fromHost`'s message already names the result and the path.
        if (error instanceof TypeCheckError) throw new ToolCheckError(error.message, { cause: error });
        throw error;
      }
    };
    let value = intake(result);
    // The open translation: the check never refuses a result the schema accepts.
    const typ = schema === null ? null : inputTypeFromJsonSchema(schema);
    if (typ !== null) {
      try {
        buildValueAs(typ, value);
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        const at = error.path.length > 0 ? ` at ${renderValuePath(error.path)}` : "";
        throw new ToolCheckError(`${where}${at}: ${error.message}`, { cause: error });
      }
    }
    if (validator !== null) {
      let output: unknown;
      try {
        const returned: unknown = await validator["~standard"].validate(toHost(value));
        const checked = standardResult(returned, `${where}: the output validator`, ToolCheckError);
        if (checked.issues !== undefined) {
          throw new ToolCheckError(validatorFailure(`${where} failed its output validator`, checked.issues));
        }
        output = checked.value;
      } catch (error) {
        // The guard, not `instanceof`, which would run a thrown Proxy's traps.
        if (isToolCheckError(error)) throw error;
        // `execute` has run, so a retry would repeat it: any other throw here, from the validator
        // or from reading its result, fails the call for good.
        throw new ToolCheckError(`${where}: the output validator threw: ${thrownText(error)}`, { cause: error });
      }
      // The validator's output is what yian records, so it must be a yian value too.
      value = intake(output);
    }
    return toHost(value);
  }
}

/**
 * Return a thrown value's text for a message: an `Error`'s message, else the value as text, else
 * its type name. Never throws, so a hostile thrown value (a Proxy, a Symbol message) can't escape.
 */
function thrownText(error: unknown): string {
  try {
    return String(error instanceof Error ? error.message : error);
  } catch {
    return jsTypeName(error);
  }
}

/**
 * Return a value a tool threw as an `Error`: an `Error`, or a value carrying a tool error brand,
 * as it is; anything else wrapped in an `Error` whose message is its text and whose `cause` is
 * the value.
 */
function asError(error: unknown): unknown {
  // The brand check keeps a branded error from another realm, which fails `instanceof`, unwrapped
  // so its kind still steers retries.
  if (toolErrorKinds(error).length > 0) return error;
  try {
    if (error instanceof Error) return error;
  } catch {
    // A Proxy whose prototype trap throws isn't an `Error`.
  }
  return new Error(thrownText(error), { cause: error });
}

/**
 * Return a validator's result as `{ issues }` or `{ value }`, each read once, when it is a Standard
 * Schema result: an object whose `issues` is an array, or whose `issues` is absent and which holds
 * `value`.
 *
 * @throws `kind`, with the message `<head> returned <type>, not a Standard Schema result`, for
 *   anything else.
 */
function standardResult(
  result: unknown,
  head: string,
  kind: new (message: string) => Error,
): StandardResult<unknown> {
  if (typeof result === "object" && result !== null) {
    const issues: unknown = (result as { issues?: unknown }).issues;
    if (Array.isArray(issues)) return { issues };
    if (issues === undefined && "value" in result) return { value: (result as { value: unknown }).value };
  }
  throw new kind(`${head} returned ${jsTypeName(result)}, not a Standard Schema result`);
}

/**
 * Return the message for a validator's issues: `head` (for example `tool 'x' arguments failed its
 * validator`), then each issue as `<path>: <message>`, or `<message>` when it has no path, joined
 * by `"; "`.
 */
function validatorFailure(head: string, issues: ReadonlyArray<StandardIssue>): string {
  const shown = issues.map((issue) => {
    // `Array.from`, not `.map`: a library's path may be an Array subclass whose species
    // constructor builds a different array (ArkType's empty path maps to `[0]`).
    const path = Array.from(issue.path ?? [], (segment) => {
      // `null` is outside the type, but a host validator may still send it.
      const key = typeof segment === "object" && segment !== null ? segment.key : segment;
      return typeof key === "symbol" ? String(key) : key;
    });
    return path.length > 0 ? `${renderValuePath(path)}: ${issue.message}` : issue.message;
  });
  return shown.length > 0 ? `${head}: ${shown.join("; ")}` : head;
}

/**
 * Check call arguments against a tool's JSON Schema.
 *
 * Unknown keys are rejected unless `additionalProperties` is `true` or a schema, in which case
 * extra values are checked against that schema. Missing required keys are rejected, in the
 * schema's `required` order. Each value is checked against the yian type its JSON Schema
 * translates to; a schema with no translation is not checked. `scope` names the invocation path
 * in the unknown-key message (`ask` and `ask_user` have two).
 *
 * @throws {RangeError} An unknown or missing key.
 * @throws {TypeCheckError} A value doesn't fit its property's translated type.
 */
function checkArguments(
  toolId: string,
  schema: JsonObject,
  payload: ValueRecord,
  scope: "public" | "trusted" | null,
): void {
  const name = reprText(toolId);
  // Own keys only: a schema is a plain object, so `toString` must not read as a property.
  const found = Object.hasOwn(schema, "properties") ? schema.properties : undefined;
  const properties: ValueRecord = isRecord(found) ? found : {};
  const extra = Object.hasOwn(schema, "additionalProperties") ? schema.additionalProperties : undefined;
  const extraSchema = isRecord(extra) ? extra : undefined;
  const unsupported = Object.keys(payload)
    .filter((key) => !Object.hasOwn(properties, key))
    .sort(compareCodeUnits);
  if (unsupported.length > 0 && extra !== true && extraSchema === undefined) {
    const path = scope === null ? "" : ` on its ${scope} invocation path`;
    const allowed = Object.keys(properties).sort(compareCodeUnits);
    throw new RangeError(
      `tool ${name} received unsupported argument(s) ${writeAsText(unsupported)}${path}; ` +
        `allowed: ${writeAsText(allowed)}`,
    );
  }
  const required = Object.hasOwn(schema, "required") ? schema.required : undefined;
  const missing = (Array.isArray(required) ? required : []).filter(
    (key): key is string => typeof key === "string" && !Object.hasOwn(payload, key),
  );
  if (missing.length > 0) throw new RangeError(`tool ${name} is missing required argument(s) ${writeAsText(missing)}`);
  for (const [key, value] of Object.entries(payload)) {
    const prop = Object.hasOwn(properties, key) ? properties[key] : extraSchema;
    if (prop === undefined) continue;
    const typ = inputTypeFromJsonSchema(prop, schema);
    if (typ === null) continue;
    try {
      buildValueAs(typ, value);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw new TypeCheckError(`tool ${name} argument ${reprText(key)}: ${error.message}`, {
        path: [key, ...error.path],
        cause: error,
      });
    }
  }
}

/**
 * Check and call one registered tool through its model-visible public schema.
 *
 * Errors from the check and the tool propagate unchanged. The tool gets a context outside any
 * run: `runId` and `node` are `""` and `callId` is a fresh random hex id.
 */
export function invokeTool(toolId: string, inputs: Record<string, unknown>): Promise<unknown> {
  return new ToolSet().invoke(toolId, inputs);
}

/**
 * Check and call one registered tool through its full schema, including host-injected fields.
 *
 * The tool gets a context outside any run, as with `invokeTool`.
 */
export function invokeToolTrusted(toolId: string, inputs: Record<string, unknown>): Promise<unknown> {
  return new ToolSet().invokeTrusted(toolId, inputs);
}
