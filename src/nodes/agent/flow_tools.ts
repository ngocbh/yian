/**
 * Flow tools: the `defs:` entries, `uses:` aliases and library defs an AGENT grants as tools.
 *
 * Compose bakes each granted flow onto the agent, with the description and argument schema the
 * model sees for it, as a CALL bakes its child. In the tool loop each is a local tool whose call
 * records a `pending` step holding its arguments (`flowCall`), so the run grows the flow's scope
 * and answers the call with the flow's output (`flowResultText`).
 *
 * Knows about: diagnostics, tools (the function tool, the node scope), typesys, and the peer AGENT
 * module `results`.
 */

import { type Failure, reprText } from "../../diagnostics.ts";
import { FunctionTool } from "../../tools/function_tools.ts";
import { type NodeScope, visibleToolName } from "../../tools/grants.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { cutText, isRecord } from "../../typesys/values.ts";
import { toolText } from "./results.ts";

/** The brand a flow call's marker carries. */
const FLOW_CALL: unique symbol = Symbol("yian.flowCall") as never;

/** What a flow tool's call returns to its loop: the checked arguments the flow's scope starts with. */
export interface FlowCall {
  readonly [FLOW_CALL]: true;
  readonly arguments: Readonly<Record<string, unknown>>;
}

/** Whether `value` is a flow call's marker. */
export function isFlowCall(value: unknown): value is FlowCall {
  return typeof value === "object" && value !== null && (value as Record<symbol, unknown>)[FLOW_CALL] === true;
}

/**
 * Return flow tool `tool` as a local tool named `name`: it takes the flow's description and schema,
 * and a call returns its arguments as a `FlowCall`. A call does nothing but record them, so a
 * resume that finds it cut off sends it again (`resumable`).
 */
export function flowToolFunction({ description, parameters }: FlowTool, name: string): FunctionTool {
  return new FunctionTool({
    name,
    description,
    parametersJsonSchema: parameters,
    execute: (args): FlowCall => Object.freeze({ [FLOW_CALL]: true as const, arguments: args }),
    resumable: true,
  });
}

/**
 * Return `tools` as a node scope holds them, each entry built by `make` under the name the scope
 * knows it by: a library's defs as a group under its alias, each by its def's name, so the model
 * sees `<alias>_<def>`; every other flow as a top-level tool by its name.
 */
export function flowToolScope<T extends { readonly name: string }>(
  tools: readonly FlowTool[],
  make: (tool: FlowTool, name: string) => T,
): NodeScope<T> {
  const groups = new Map<string, T[]>();
  const top: T[] = [];
  for (const tool of tools) {
    if (tool.library === null) {
      top.push(make(tool, tool.name));
      continue;
    }
    const { alias, def } = tool.library;
    groups.set(alias, [...(groups.get(alias) ?? []), make(tool, def)]);
  }
  return { groups, tools: top };
}

/**
 * Return the error text the model reads for flow tool `name`'s call that failed: the node of the
 * flow that failed (its id inside the call's `scope`), the failure's code and message; for a
 * call whose scope couldn't grow, why it didn't start.
 */
export function flowFailureText(name: string, scope: string | null, failure: Failure): string {
  const prefix = scope === null ? null : `${scope}/`;
  const node = failure.nodeId;
  if (prefix === null || node === null || !node.startsWith(prefix)) {
    return `Flow ${name} didn't start (${failure.code}): ${failure.message}`;
  }
  return `Flow ${name} failed at node ${node.slice(prefix.length)} (${failure.code}): ${failure.message}`;
}

/**
 * Return the text the model reads for flow tool `name`'s output: text as itself, any other value
 * as its JSON (`toolText`), cut to its first `maxChars` characters (`cutText`) with a note that the
 * run log holds all of it.
 *
 * @throws TypeCheckError when the output is not a yian value.
 */
export function flowResultText(name: string, output: unknown, maxChars: number): string {
  const text = toolText(name, output);
  const cut = cutText(text, maxChars);
  return cut === text ? text : `${cut}; the run log holds the whole output`;
}

/** One flow an AGENT grants as a tool. Frozen. */
export interface FlowTool {
  /**
   * The name the model sees: the `defs:` entry or `uses:` alias granted, or `<alias>_<def>` for a
   * library's def.
   */
  readonly name: string;
  /** The flow's `description:`, which the model reads to choose the tool. */
  readonly description: string;
  /** The JSON Schema of the tool's arguments, one property per input of the flow. */
  readonly parameters: JsonObject;
  /** The baked compiled flow, as a CALL's `child` holds one. */
  readonly child: unknown;
  /** The library def this tool runs, granted through the library's alias; `null` for a flow. */
  readonly library: { readonly alias: string; readonly def: string } | null;
}

/**
 * Return a frozen copy of `tools`, checked.
 *
 * @throws TypeError when an entry isn't a record with a non-empty `name`, a non-empty
 *   `description`, an object `parameters`, a `child` and a `library` that is `null` or a non-empty
 *   `alias` and `def` the tool is named `<alias>_<def>` by.
 * @throws RangeError when two entries share a name.
 */
export function checkedFlowTools(nodeId: string, tools: readonly FlowTool[]): readonly FlowTool[] {
  const names = new Set<string>();
  const checked = tools.map((tool) => {
    const { name, description, parameters, child, library = null } = isRecord(tool) ? tool : ({} as Partial<FlowTool>);
    if (typeof name !== "string" || name === "" || typeof description !== "string" || description === "") {
      throw new YianTypeError(`AGENT node ${reprText(nodeId)}: a flow tool needs a name and a description`);
    }
    if (!isRecord(parameters) || child === undefined || child === null) {
      throw new YianTypeError(`AGENT node ${reprText(nodeId)}: flow tool ${reprText(name)} needs its parameters and flow`);
    }
    if (library !== null) {
      const { alias, def } = isRecord(library) ? library : ({} as Record<string, unknown>);
      if (typeof alias !== "string" || alias === "" || typeof def !== "string" || def === "" || name !== visibleToolName(alias, def)) {
        throw new YianTypeError(`AGENT node ${reprText(nodeId)}: flow tool ${reprText(name)} needs a library alias and def it is named by`);
      }
    }
    if (names.has(name)) throw new YianRangeError(`AGENT node ${reprText(nodeId)} grants flow ${reprText(name)} twice`);
    names.add(name);
    const def = library === null ? null : Object.freeze({ alias: library.alias, def: library.def });
    return Object.freeze({ name, description, parameters, child, library: def });
  });
  return Object.freeze(checked);
}
