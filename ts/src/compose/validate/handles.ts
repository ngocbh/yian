/**
 * Validate: a `CaseNode`'s handles align with its outgoing control edges.
 *
 * The compile check's handle-alignment rule, applied to the desugared `CaseNode`s and their
 * control edges (the Compose path has no `FlowSpec`). Runs in the validate phase over the built
 * node map and the derived edges.
 *
 * Knows about: the compile model (`Edge`), nodes (`CaseNode`, `ELSE_HANDLE`), and load
 * diagnostics.
 * Never imports: build, the loader, or the runtime.
 */

import type { Edge } from "../../compile/model.ts";
import { reprText } from "../../diagnostics.ts";
import { ELSE_HANDLE, type Node } from "../../nodes/base.ts";
import { CaseNode } from "../../nodes/case/node.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";

/**
 * Check that every desugared `CaseNode`'s handles align with its outgoing control edges.
 *
 * Each outgoing control edge's handle must be a declared case handle or the else handle, and
 * every declared case must have an outgoing edge. A handle-less edge (`depends_on`, `runs_after`,
 * a data read) is not a branch, so it is not checked. The `case` desugar emits the
 * `gate -> <then|else>` control edges and the `CaseNode.cases`; this pins them consistent.
 * Out-edges are checked in edge order and cases in case order, so the first misalignment
 * reported is deterministic.
 *
 * @param options.diagnostics Builds the error (default: an unlocated one).
 * @throws LoadError on the first misalignment.
 */
export function checkCaseHandles(
  nodes: ReadonlyMap<string, Node>,
  edges: readonly Edge[],
  options: { diagnostics?: LoadDiagnostics | null } = {},
): void {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  for (const [nodeId, node] of nodes) {
    if (!(node instanceof CaseNode)) continue;
    const caseHandles = new Set(node.cases.map((c) => c.handle));
    const outHandles = new Set<string>();
    for (const e of edges) {
      if (e.from === nodeId && e.sourceHandle !== null) outHandles.add(e.sourceHandle);
    }
    for (const handle of outHandles) {
      if (handle !== ELSE_HANDLE && !caseHandles.has(handle)) {
        throw diagnostics.error(
          `case node ${reprText(nodeId)} has edge handle ${reprText(handle)} with no matching case`,
        );
      }
    }
    for (const handle of caseHandles) {
      if (!outHandles.has(handle)) {
        throw diagnostics.error(`case node ${reprText(nodeId)} case ${reprText(handle)} has no outgoing edge`);
      }
    }
  }
}
