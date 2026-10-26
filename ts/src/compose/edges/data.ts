/**
 * Data-edge inference: a data `Edge` per `${<id>.output[.…]}` reference.
 *
 * The analogue of scope/dependency analysis: a sink binding that reads a free variable
 * `${<id>.output[.…]}` depends on the producing node `<id>`, so we emit a data
 * `Edge(<id> -> <consumer>, {inputGroup: <sink key>})`. `exprRefsOf` already flattens a coalesce
 * `${a | b}` to both ref paths, so the two alternatives of one sink share that sink's input group
 * (per-input readiness). An `input.X` ref is the `START_ID` input-reader edge; `system`/`item`
 * refs are not data edges.
 *
 * Boundary assembly owns the `END_ID` producer edges and the `START_ID` root edges, and runs after
 * the `case` control edges are built; this pass emits only the edges of the flow wiring.
 *
 * Knows about: the compile model (`Edge`), `expr` co-skip analysis, binding input groups, the
 * `case` descriptor, and `edges/producers.ts`.
 * Never imports: validate, build, the loader, or the runtime.
 */

import { Edge } from "../../compile/model.ts";
import { bindingCoSkips } from "../../expr/index.ts";
import { bindingInputGroups } from "../../nodes/contracts.ts";
import type { NodeDescriptor } from "../nodes/base.ts";
import { CaseDescriptor } from "../nodes/case.ts";
import { bindingProducers } from "./producers.ts";

/**
 * Infer the flow's data edges from the flow-owned wiring and the descriptors.
 *
 * For each consuming node, scan its sources and emit a data `Edge` per `${<id>.output[.…]}` ref,
 * tagged with the input group that identifies which sink it feeds (the param or arg name for a
 * leaf or call input; `over` for a map's `over:`). A coalesce sink's alternatives share one
 * group. Each string nested in a `ValueTemplate` source is its own group
 * (`bindingInputGroups`), with its own co-skip stance: the edge is optional unless the string
 * co-skips (`bindingCoSkips`). Edges are the projection of the flow wiring for leaf and call/map
 * nodes, named `<from>-><to>#<i>`.
 *
 * A `case` node is skipped: `CaseDescriptor.build` returns its data and control edges, so no
 * provisional into-case edge is inferred.
 *
 * @param descriptors The flow level's descriptors by node id.
 * @param flowWiring Node id to its `{param -> source}` map (a MAP carries its `over` under the
 *   reserved key).
 * @returns The data edges, in descriptor then wiring order.
 */
export function inferDataEdges(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
): Edge[] {
  const edges: Edge[] = [];
  const counts = new Map<string, number>(); // `from->to` -> next edge index

  const emitFor = (value: unknown, consumer: string, param: string): void => {
    // Each template string is its own group, and its co-skip stance is its own.
    for (const [group, text] of bindingInputGroups(param, value)) {
      const optional = !bindingCoSkips(text);
      for (const producer of bindingProducers(text)) {
        const pair = `${producer}->${consumer}`;
        const i = counts.get(pair) ?? 0;
        counts.set(pair, i + 1);
        edges.push(new Edge(`${pair}#${i}`, producer, consumer, { inputGroup: group, optional }));
      }
    }
  };

  for (const [nodeId, desc] of descriptors) {
    // A `case` owns its edges: `CaseDescriptor.build` returns them in its `BuildResult.edges`.
    if (desc instanceof CaseDescriptor) continue;
    // Leaf and call/map: a map's reserved `over` source rides the `"over"` key (over-then-inputs
    // order); `${item}` is body-local and yields no edge.
    for (const [param, src] of flowWiring.get(nodeId) ?? []) emitFor(src, nodeId, param);
  }
  return edges;
}
