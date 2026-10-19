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
 * Knows about: `tools/context`, `tools/function_tools`, `tools/mcp`, `skills/tool` (the reserved
 * `load_skill` name), typesys and diagnostics.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { LOAD_SKILL } from "../skills/tool.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isRecord, jsTypeName, writeAsText } from "../typesys/values.ts";
import { ASK_USER, type ToolContext } from "./context.ts";
import { FunctionTool, isFunctionTool } from "./function_tools.ts";
import { isMcpServer, type McpServer } from "./mcp.ts";

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
    const where = `tools[${reprText(group)}][${index}]`;
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
      `tools[${reprText(key)}] is an MCP server that isn't open; open it with "await server.open()" ` +
        "or pass it to runFlow/resumeFlow, which open it",
    );
  }
  if (server.isClosed) throw new RangeError(`tools[${reprText(key)}]: MCP server '${server.label}' is closed`);
  return server.tools;
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
 * `forRun` binds a copy to one run's id and abort signal.
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
          throw new TypeError("tools['ask_user'] must be a plain function handler; the engine supplies its schema");
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
          `tools[${reprText(key)}] must be a FunctionTool, an MCP server, or an array of FunctionTools, ` +
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
            name === ASK_USER ? "pass the handler as the top-level tools['ask_user'] entry" : "granting skills binds it";
          throw new RangeError(`group ${reprText(group)} has a tool named ${reprText(name)}, a reserved name; ${hint}`);
        }
        if (topLevel.has(name)) {
          throw new RangeError(
            `tool ${reprText(name)} in group ${reprText(group)} clashes with tools[${reprText(name)}]`,
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
}
