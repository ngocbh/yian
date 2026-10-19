/**
 * The tool registry and the tools one run can use.
 *
 * `registerTool` stores a host `FunctionTool` in `TOOL_REGISTRY`. A run's `ToolSet` reads this
 * registry when the host passes no `tools` mapping; the TOOL, HUMAN_INPUT and AGENT nodes look
 * tools up by id through that `ToolSet`.
 *
 * This is the package's tool seam: the core ships no domain tools, and a host registers its own.
 * Tool ids must be unique across everything registered into the process. `ToolSet` holds one
 * run's tools: this registry, or a host mapping that replaces it and may hold named tool groups.
 * The reserved `ask_user` id never lives here: each run gets its handler through
 * `new ToolSet({ ask_user: handler, ... })`.
 *
 * Knows about: `tools/context`, `tools/errors`, `tools/function_tools`, `tools/mcp`, `tools/types`,
 * `skills/tool` (the reserved `load_skill` name), `defaults`, typesys and diagnostics.
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
import { ASK_USER, ToolContext } from "./context.ts";
import { ToolCheckError } from "./errors.ts";
import { FunctionTool, isFunctionTool, type StandardIssue, type StandardResult } from "./function_tools.ts";
import { isMcpServer, type McpServer } from "./mcp.ts";
import { inputTypeFromJsonSchema } from "./types.ts";

export { ASK_USER };

/** Maps tool id to the `FunctionTool` registered under it. */
export const TOOL_REGISTRY = new Map<string, FunctionTool>();

// Names a group tool may not take: the run's ask_user handler and the skills tool.
const RESERVED_TOOL_NAMES: ReadonlySet<string> = new Set([ASK_USER, LOAD_SKILL]);
const ASK_USER_HINT = "pass tools: {...registeredTools(), ask_user: handler} to runFlow or resumeFlow";

const ASK_USER_DESCRIPTION =
  "Ask the user one short question and wait for the answer. Use only when you need a decision " +
  "or information that only the user can provide. Put the context they need in one or two " +
  "sentences. When there are clear choices, pass 2-4 short `options`; the user can always type " +
  "their own answer.";

// The schema Python's LangChain binding gives the model for `ask_user`, key order included.
const ASK_USER_PARAMETERS: JsonObject = {
  properties: {
    question: { type: "string" },
    options: {
      anyOf: [{ items: { type: "string" }, type: "array" }, { type: "null" }],
      default: null,
      description: "Optional 2-4 short answer choices. The user can always type a different answer.",
    },
  },
  required: ["question"],
  type: "object",
};

// The full schema a trusted call (a HUMAN_INPUT node) is checked against: the public one plus the
// host-injected question and display metadata, which the model never sees.
const ASK_USER_TRUSTED_PARAMETERS: JsonObject = {
  ...ASK_USER_PARAMETERS,
  properties: {
    ...(ASK_USER_PARAMETERS.properties as JsonObject),
    questions: { anyOf: [{ items: { type: "object" }, type: "array" }, { type: "null" }], default: null },
    answerSchema: { anyOf: [{ items: { type: "object" }, type: "array" }, { type: "null" }], default: null },
    title: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
    nodeId: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
  },
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

/** One tool call whose arguments `ToolSet.validate` checked; `invoke` and `isConcurrencySafe` take it. */
class CheckedCall {
  /** The id of the tool the call is for. */
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

/** One entry of a host `tools` mapping. */
export type ToolEntry = FunctionTool | McpServer | readonly FunctionTool[] | AskUserHandler;

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
 * @throws {RangeError} `toolId` is the reserved `ask_user` id, is already registered, or differs
 *   from the tool's name.
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
  if (toolId === ASK_USER) throw new RangeError(`tool id ${reprText(ASK_USER)} is reserved; ${ASK_USER_HINT}`);
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
 * the mapping itself, for example `{...registeredTools(), ask_user: handler}`. Changing the copy
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
    // Code points, as Python's `len` counts them.
    const chars = Array.from(label);
    if (chars.length > caps.maxQuestionOptionChars) {
      label = chars.slice(0, caps.maxQuestionOptionChars - 1).join("") + "…";
    }
    // Close to Python's `casefold`: "Straße" and "STRASSE" are the same label.
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
 * The model sees `question` and optional `options`. The handler receives all six request fields,
 * absent ones `null`, and the call's context.
 */
function askUserTool(handler: AskUserHandler): FunctionTool {
  return new FunctionTool({
    name: ASK_USER,
    description: ASK_USER_DESCRIPTION,
    parametersJsonSchema: ASK_USER_PARAMETERS,
    execute: (args, ctx) =>
      handler(
        {
          question: args.question as string,
          options: (args.options ?? null) as string[] | null,
          questions: (args.questions ?? null) as unknown[] | null,
          answerSchema: (args.answerSchema ?? null) as unknown[] | null,
          title: (args.title ?? null) as string | null,
          nodeId: (args.nodeId ?? null) as string | null,
        },
        ctx,
      ),
  });
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
        "or pass it to runFlow/resumeFlow, which open it",
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
 * An entry whose value is an array of `FunctionTool`s, or an open `McpServer`, is a tool group. An
 * AGENT node grants a group by its key, and `expand` replaces that key with the group's tool
 * names. Every tool name in the run names one tool: the same tool object may sit in several
 * groups, but a group tool may not share its name with a different tool or with a top-level key,
 * or take a reserved name (`ask_user`, `load_skill`).
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
 *   `name`. An open `McpServer` is a group of the tools it listed when it opened. The `ask_user`
 *   entry must be a plain function `(request, ctx)`; it is wrapped so the model sees `question` and
 *   `options`, and it receives the six-field `AskUserRequest` and the call's context. `null` or
 *   `undefined` means the global registry. To keep the registered tools and add more, pass
 *   `{...registeredTools(), ...}`.
 * @param options `context` is the host's app object, passed to tools as `ToolContext.context`. It
 *   is never recorded.
 * @throws {TypeError} `tools` is not a plain object, `null` or `undefined`; a key is empty; an
 *   entry is not a `FunctionTool`, an `McpServer` or an array of `FunctionTool`s; a group item is
 *   an `McpServer` or not a `FunctionTool`; an `McpServer` entry isn't open; or the `ask_user`
 *   entry is not a function.
 * @throws {RangeError} A `FunctionTool` entry's name differs from its key, an `McpServer` entry was
 *   open but has gone away, or a group tool's name is `ask_user` or `load_skill`, is a top-level
 *   key, or names a different tool elsewhere in the groups.
 *
 * @example
 * const tools = new ToolSet({ toolbox: [weather, forecast] }); // two FunctionTools
 * tools.expand(["toolbox"]); // ["weather", "forecast"]
 * const runTools = new ToolSet({ place }, { context: broker }).forRun("run-1");
 * const call = await runTools.validate("place", { sym: "X" });
 * const parallel = runTools.isConcurrencySafe(call); // true: it may run beside other calls
 * await runTools.invoke(call, { node: "buy" }); // runs either way
 */
export class ToolSet {
  // Plain private fields, not `#` ones, so `forRun` can fill a copy of a subclass instance.
  private toolMap: ReadonlyMap<string, FunctionTool> | null;
  private groupMap: ReadonlyMap<string, readonly string[]>;
  private boundRunId: string;
  private boundSignal: AbortSignal | undefined;
  private hostContext: unknown;

  constructor(tools?: Readonly<Record<string, ToolEntry>> | null, options?: { context?: unknown }) {
    this.toolMap = null;
    this.groupMap = new Map();
    this.boundRunId = "";
    this.boundSignal = undefined;
    this.hostContext = options?.context;
    if (tools === undefined || tools === null) return;
    if (!isRecord(tools)) throw new TypeError("tools must be a record of tool id to tool, or null");
    const keys = Object.keys(tools);
    const built = new Map<string, FunctionTool>();
    const groupItems = new Map<string, readonly FunctionTool[]>();
    for (const key of keys) {
      if (key === "") throw new TypeError("tool ids in tools must be non-empty strings");
      const entry: unknown = tools[key];
      if (key === ASK_USER) {
        // A FunctionTool is an object, so this also rejects one.
        if (typeof entry !== "function") {
          throw new TypeError(`tools["ask_user"] must be a plain function handler; the engine supplies its schema`);
        }
        built.set(key, askUserTool(entry as AskUserHandler));
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
    const groups = new Map<string, readonly string[]>();
    const owners = new Map<string, { group: string; item: FunctionTool }>();
    for (const [group, items] of groupItems) {
      const names = new Set<string>();
      for (const item of items) {
        const name = item.name;
        if (RESERVED_TOOL_NAMES.has(name)) {
          const hint =
            name === ASK_USER
              ? `pass the handler as the top-level tools["ask_user"] entry`
              : "granting skills binds it";
          throw new RangeError(`group ${reprText(group)} has a tool named ${reprText(name)}, a reserved name; ${hint}`);
        }
        if (topLevel.has(name)) {
          throw new RangeError(
            `tool ${reprText(name)} in group ${reprText(group)} clashes with tools[${JSON.stringify(name)}]`,
          );
        }
        names.add(name);
        const owner = owners.get(name);
        if (owner === undefined) {
          owners.set(name, { group, item });
          built.set(name, item);
          continue;
        }
        // Host permission groups overlap, so one tool may sit in several of them; two different
        // tools under one name would be ambiguous.
        if (owner.item !== item) {
          const where =
            owner.group === group
              ? `appears twice in group ${reprText(group)}`
              : `is in both groups ${reprText(owner.group)} and ${reprText(group)}`;
          throw new RangeError(`tool ${reprText(name)} ${where} as different tools`);
        }
      }
      groups.set(group, Object.freeze([...names]));
    }
    this.toolMap = built;
    this.groupMap = groups;
  }

  /**
   * Return this run's tool, or the registry's when the run has no mapping.
   *
   * `ask_user` never comes from the registry: a run without its own handler cannot ask.
   */
  private lookup(toolId: string): FunctionTool {
    if (toolId === ASK_USER && !this.toolMap?.has(toolId)) {
      throw new RangeError(`no ask_user handler for this run; ${ASK_USER_HINT}`);
    }
    if (this.groupMap.has(toolId)) {
      throw new RangeError(`${reprText(toolId)} is a tool group, not a tool; grant it in an AGENT node's tools: list`);
    }
    const tools = this.toolMap ?? TOOL_REGISTRY;
    const tool = tools.get(toolId);
    if (tool === undefined) {
      throw new RangeError(`unknown tool id ${reprText(toolId)}; known tools: ${writeAsText(sortedIds(tools))}`);
    }
    return tool;
  }

  /**
   * Return a copy bound to one run, sharing this set's tools, groups and context.
   *
   * @param runId The run's id, passed to tools as `ToolContext.runId` and the base of each call's
   *   default `callId`.
   * @param options `signal` is the run's abort signal, aborted when the run stops. The copy holds
   *   exactly this value: none gives no signal, even when this set was bound to one.
   * @returns A new tool set of the same class; this one stays as it was.
   */
  forRun(runId: string, options?: { signal?: AbortSignal }): this {
    const bound = Object.create(Object.getPrototypeOf(this)) as this;
    bound.toolMap = this.toolMap;
    bound.groupMap = this.groupMap;
    bound.boundRunId = runId;
    bound.boundSignal = options?.signal;
    bound.hostContext = this.hostContext;
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
   * The sorted ids of the tools this run can resolve: with a mapping, the mapping's tools and group
   * members; with no mapping, the global registry's tools.
   */
  get toolIds(): readonly string[] {
    return sortedIds(this.toolMap ?? TOOL_REGISTRY);
  }

  /**
   * Replace each group id with the names of the tools in that group.
   *
   * @param ids Granted ids, in order. An id that is not a group is kept as is.
   * @returns The ids with every group replaced by its tool names, in group order. A name that
   *   repeats, from overlapping groups or a repeated grant, is kept only where it first appears.
   */
  expand(ids: readonly string[]): string[] {
    // Overlapping or repeated grants bind each tool once.
    const expanded = new Set<string>();
    for (const id of ids) for (const name of this.groupMap.get(id) ?? [id]) expanded.add(name);
    return [...expanded];
  }

  /**
   * Look tools up by id.
   *
   * @param ids The ids to resolve, in order.
   * @returns The matching tools, in the same order as `ids`.
   * @throws {RangeError} An id is not visible to this run.
   */
  resolve(ids: readonly string[]): FunctionTool[] {
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
   * @param toolId The id of the tool the call is for.
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
    toolId: string,
    inputs: Record<string, unknown>,
    options?: { signal?: AbortSignal },
  ): Promise<CheckedCall> {
    return await this.checkedCall(toolId, inputs, false, options?.signal);
  }

  /**
   * Call one tool: a call `validate` returned, or a tool id and its public arguments.
   *
   * `invoke(call, options)` runs the tool `validate` looked up, with this set's context, run id
   * and signal; it checks no argument again. `invoke(toolId, inputs, options)` validates the
   * arguments first, as `validate` does, then calls the tool. Errors from the argument checks and
   * from the tool propagate unchanged.
   *
   * When the tool declares an output (`outputJsonSchema` or `outputValidator`), the result must be
   * a yian value; it is checked against the type the output schema translates to, read open (an
   * object's fields are checked only when the schema sets `additionalProperties: false`), then
   * the output validator runs on it and its output must be a yian value too. `invoke` returns that
   * value in host form. A tool without an output returns its result as it is.
   *
   * @param call A call `validate` returned, on this set or any other.
   * @param toolId The id of the tool to call.
   * @param inputs Public arguments keyed by parameter name.
   * @param options The calling node, call id, attempt and abort signal the tool's context carries.
   * @returns The tool's result, awaited, `undefined` read as `null`; for a tool that declares an
   *   output, the checked value in host form.
   * @throws {TypeError} The first argument is neither a tool id nor a `CheckedCall`; and as
   *   `validate` throws.
   * @throws {ToolCheckError} A declared result is not a yian value or doesn't fit the output
   *   schema's type; or the output validator threw (the error is the `cause`), returned no
   *   Standard Schema result, reported issues or gave a value that isn't a yian value.
   * @throws Whatever the tool throws, unchanged.
   */
  invoke(call: CheckedCall, options?: InvokeOptions): Promise<unknown>;
  invoke(toolId: string, inputs: Record<string, unknown>, options?: InvokeOptions): Promise<unknown>;
  async invoke(
    target: CheckedCall | string,
    inputsOrOptions?: Record<string, unknown> | InvokeOptions,
    options?: InvokeOptions,
  ): Promise<unknown> {
    if (typeof target === "string") {
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
   * Only `ask_user` has injected fields (`questions`, `answerSchema`, `title`, `nodeId`); every
   * other tool is checked as `invoke` checks it, its validator included.
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
  async invokeTrusted(toolId: string, inputs: Record<string, unknown>, options?: InvokeOptions): Promise<unknown> {
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
    toolId: string,
    inputs: Record<string, unknown>,
    trusted: boolean,
    signal: AbortSignal | undefined = this.boundSignal,
  ): Promise<CheckedCall> {
    const [tool, payload] = this.check(toolId, inputs, trusted);
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
  private check(toolId: string, inputs: Record<string, unknown>, trusted: boolean): [FunctionTool, ValueRecord] {
    const tool = this.lookup(toolId);
    const where = `tool ${reprText(toolId)} arguments`;
    const payload = fromHost(inputs, where);
    if (!isRecord(payload)) throw new TypeError(`${where} must be a record, got ${jsTypeName(inputs)}`);
    const askUser = toolId === ASK_USER;
    if (askUser && Object.hasOwn(payload, "options")) payload.options = normalizeOptions(payload.options);
    const schema = askUser && trusted ? ASK_USER_TRUSTED_PARAMETERS : tool.parametersJsonSchema;
    checkArguments(toolId, schema, payload, askUser ? (trusted ? "trusted" : "public") : null);
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
      callId: options.callId ?? (this.boundRunId && node ? `${this.boundRunId}/${node}` : defaultRunId()),
      attempt: options.attempt ?? 1,
      toolName: toolId,
      runId: this.boundRunId,
      node,
      context: this.hostContext,
      signal: options.signal ?? this.boundSignal ?? new AbortController().signal,
    });
    let result = await tool.execute(args, ctx);
    if (result === undefined) result = null;
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
        if (error instanceof ToolCheckError) throw error;
        // `execute` has run, so a retry would repeat it: any other throw here, from the validator
        // or from reading its result, fails the call for good.
        let message: string;
        try {
          message = error instanceof Error ? error.message : String(error);
        } catch {
          // A thrown value `String` can't convert (a null-prototype object) must not escape as a
          // retried TypeError.
          message = jsTypeName(error);
        }
        throw new ToolCheckError(`${where}: the output validator threw: ${message}`, { cause: error });
      }
      // The validator's output is what yian records, so it must be a yian value too.
      value = intake(output);
    }
    return toHost(value);
  }
}

/**
 * Return a validator's result when it is a Standard Schema result: an object holding `issues` or
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
  if (typeof result === "object" && result !== null && ("issues" in result || "value" in result)) {
    return result as StandardResult<unknown>;
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
    const path = (issue.path ?? []).map((segment) => {
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
 * in the unknown-key message (`ask_user` has two).
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
