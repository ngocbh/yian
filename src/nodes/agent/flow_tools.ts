/**
 * Flow tools: the `defs:` entries and `uses:` aliases an AGENT grants as tools.
 *
 * Compose bakes each granted flow onto the agent, with the description and argument schema the
 * model sees for it, as a CALL bakes its child.
 *
 * Knows about: diagnostics, typesys.
 */

import { reprText } from "../../diagnostics.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { isRecord } from "../../typesys/values.ts";

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
      throw new YianTypeError(`AGENT node ${reprText(nodeId)}: a flow tool needs a name and a description`);
    }
    if (!isRecord(parameters) || child === undefined || child === null) {
      throw new YianTypeError(`AGENT node ${reprText(nodeId)}: flow tool ${reprText(name)} needs its parameters and flow`);
    }
    if (names.has(name)) throw new YianRangeError(`AGENT node ${reprText(nodeId)} grants flow ${reprText(name)} twice`);
    names.add(name);
    return Object.freeze({ name, description, parameters, child });
  });
  return Object.freeze(checked);
}
