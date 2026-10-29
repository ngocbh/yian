/**
 * Derive and atomically stamp memory-independent static execution scopes.
 *
 * This module assigns root-relative authored paths to private compiled node templates in one
 * walk: every scope is derived and checked before any node is stamped. LOOP templates begin at
 * driver zero; all other dynamic occurrence paths belong to growth preparation.
 *
 * Knows about:   compile/model, nodes/base, nodes/contracts, nodes/start, nodes/end and typesys.
 * Never imports: memory, compose, runtime, or the package-root run boundary.
 */

import type { Node } from "../nodes/base.ts";
import { ExecutionPosition, ExecutionScope, isExactly, sameExecutionScope } from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import { writeAsText } from "../typesys/values.ts";
import { Flow, flowNodes, nodeChild, validateUniqueTargets } from "./model.ts";

/** One static node, its root-relative authored path and the scope it receives. */
type ExecutionTarget = { readonly locator: readonly string[]; readonly scope: ExecutionScope; readonly node: Node };

/** The occurrence path a static `node` at `locator` starts with: a LOOP's initial driver position, else none. */
function initialOccurrence(node: Node, locator: readonly string[]): ExecutionPosition[] {
  return node.isLoop ? [new ExecutionPosition("loop_driver", locator, 0)] : [];
}

/** Traverse every callsite while detecting recursive child-flow cycles. */
function* walkTargets(flow: Flow, flowPath: readonly string[], activeFlows: Set<Flow>): Generator<ExecutionTarget> {
  if (activeFlows.has(flow)) throw new RangeError("recursive child-flow cycle is not execution-layout resolvable");
  activeFlows.add(flow);
  try {
    const nodes = flowNodes(flow);
    for (const [name, value] of [["start_id", flow.startId], ["end_id", flow.endId]] as const) {
      if (typeof value !== "string" || !value) throw new RangeError(`flow ${name} must be a non-empty exact string`);
    }
    for (const [nodeId, node] of nodes) {
      const locator = [...flowPath, nodeId];
      let boundary: "start" | "end" | null = null;
      if (nodeId === flow.startId && isExactly(node, StartNode)) boundary = "start";
      else if (nodeId === flow.endId && isExactly(node, EndNode)) boundary = "end";
      const occurrencePath = initialOccurrence(node, locator);
      const scope = new ExecutionScope(locator, { flowPath, boundary, occurrencePath });
      yield { locator, scope, node };
      const child = nodeChild(node);
      if (child !== null) yield* walkTargets(child, locator, activeFlows);
    }
  } finally {
    activeFlows.delete(flow);
  }
}

/**
 * Stamp every statically reachable node of `flow` with its execution scope, all or nothing.
 *
 * Scopes don't read memory declarations or flow ids, and missing artificial START/END nodes don't
 * prevent stamping. Stamping the same graph again is idempotent.
 *
 * @param flow The private graph (`Flow` or `CompiledFlow`) to stamp.
 * @throws TypeError if graph or static-child carriers have noncanonical types.
 * @throws RangeError if a graph identifier is malformed, static children cycle, one node object is
 *   reached through several paths, or a node already has a different execution scope.
 */
export function stampExecutionScopes(flow: Flow): void {
  const targets = [...walkTargets(flow, [], new Set())];
  validateUniqueTargets(targets, "execution layout");
  for (const { locator, scope, node } of targets) {
    const existing: unknown = node.executionScope;
    if (existing !== null && (!isExactly(existing, ExecutionScope) || !sameExecutionScope(existing, scope))) {
      throw new RangeError(`node at ${writeAsText([...locator])} already has a different execution scope`);
    }
  }
  for (const { scope, node } of targets) node.executionScope = scope;
}
