/**
 * Flow tools: the `defs:` entries and `uses:` aliases an AGENT grants as tools.
 *
 * Compose bakes each granted flow onto the agent, with the description and argument schema the
 * model sees for it, as a CALL bakes its child. In the tool loop each is a local tool whose call
 * records a `pending` step holding its arguments (`flowCall`), so the run grows the flow's scope
 * and answers the call with the flow's output (`flowResultText`).
 *
 * Knows about: diagnostics, tools (the function tool), typesys, and the peer AGENT module
 * `results`.
 */

import { reprText } from "../../diagnostics.ts";
import { FunctionTool } from "../../tools/function_tools.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { cutText, isRecord } from "../../typesys/values.ts";
import { toolText } from "./results.ts";

/** What the local scope of an agent's flow tools is called in messages. */
export const FLOW_SCOPE_LABEL = "this node's flows";

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
 * Return the local tools of `tools`: each takes the flow's name, description and schema, and a
 * call returns its arguments as a `FlowCall`. A call does nothing but record them, so a resume
 * that finds it cut off sends it again (`resumable`).
 */
export function flowToolFunctions(tools: readonly FlowTool[]): FunctionTool[] {
  return tools.map(
    ({ name, description, parameters }) =>
      new FunctionTool({
        name,
        description,
        parametersJsonSchema: parameters,
        execute: (args): FlowCall => Object.freeze({ [FLOW_CALL]: true as const, arguments: args }),
        resumable: true,
      }),
  );
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
  /** The grant: the `defs:` entry or `uses:` alias, which is the name the model sees. */
  readonly name: string;
  /** The flow's `description:`, which the model reads to choose the tool. */
  readonly description: string;
  /** The JSON Schema of the tool's arguments, one property per input of the flow. */
  readonly parameters: JsonObject;
  /** The baked compiled flow, as a CALL's `child` holds one. */
  readonly child: unknown;
}

/**
 * Return a frozen copy of `tools`, checked.
 *
 * @throws TypeError when an entry isn't a record with a non-empty `name`, a non-empty
 *   `description`, an object `parameters` and a `child`.
 * @throws RangeError when two entries share a name.
 */
export function checkedFlowTools(nodeId: string, tools: readonly FlowTool[]): readonly FlowTool[] {
  const names = new Set<string>();
  const checked = tools.map((tool) => {
    const { name, description, parameters, child } = isRecord(tool) ? tool : ({} as Partial<FlowTool>);
    if (typeof name !== "string" || name === "" || typeof description !== "string" || description === "") {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: a flow tool needs a name and a description`);
    }
    if (!isRecord(parameters) || child === undefined || child === null) {
      throw new TypeError(`AGENT node ${reprText(nodeId)}: flow tool ${reprText(name)} needs its parameters and flow`);
    }
    if (names.has(name)) throw new RangeError(`AGENT node ${reprText(nodeId)} grants flow ${reprText(name)} twice`);
    names.add(name);
    return Object.freeze({ name, description, parameters, child });
  });
  return Object.freeze(checked);
}
