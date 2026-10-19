/**
 * Child resolution: the load-time callable seam shared by `call`, `map` and `loop`.
 *
 * A `call`/`map`/`loop` node names a callable (a `defs:` entry or an external flow). The child
 * resolver is the load-time seam, `(flowId, version) -> ResolvedChild`; the loader composes the
 * defs-first resolution. Each kind's own `build` bakes the resolved child's compiled flow onto
 * its node; this module supplies the shared resolution and the call nesting limit it enforces.
 *
 * Like `base.ts` this is a leaf of the nodes package: it imports no other kind module and only
 * types from the compile layer, so `call`, `loop`, `base.ts` and the loader import it without
 * cycling.
 */

import type { CompiledFlow } from "../../compile/model.ts";
import { current } from "../../defaults.ts";
import { reprText, type SourceRange } from "../../diagnostics.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";

/** A resolved child flow; structural, so this module need not import the pipeline. */
export interface ResolvedChild {
  /** The child's compiled flow. */
  readonly compiled: CompiledFlow;
}

/**
 * Resolves a child flow by id and version for call, map and loop. `version` is an opaque tag
 * (a `uses:` ref's `@<version>`); an in-file def passes `null`.
 */
export type ChildResolver = (flowId: string, version: string | null) => ResolvedChild;

/** The shape `callNesting` walks: a flow's nodes and each node's static child flows. */
interface NestingFlow {
  readonly nodes: ReadonlyMap<string, { originChildren(): readonly unknown[] }>;
}

/** Return an author-facing callable description without changing named errors. */
function callableDisplay(flowId: string, childLabel: string | null = null): string {
  return childLabel || `callable ${reprText(flowId)}`;
}

// Calls being resolved around the current one in the file being loaded. A module-level number
// stands in for Python's ContextVar because assembly is synchronous.
let activeCalls = 0;

/**
 * Run `fn` with the active call count starting from zero while one flow file loads, restoring
 * the outer count afterwards, even when `fn` throws.
 *
 * Each file counts its own calls; a call into another file is checked against the built
 * callee's depth instead. `fn` must be synchronous: an `await` inside it would let concurrent
 * loads share the count.
 *
 * @param fn The file's load.
 * @returns What `fn` returns.
 */
export function callNestingScope<T>(fn: () => T): T {
  const saved = activeCalls;
  activeCalls = 0;
  try {
    return fn();
  } finally {
    activeCalls = saved;
  }
}

/**
 * Return the longest chain of static child flows under a compiled flow.
 *
 * Kind-blind: child flows come from each node's `originChildren()`. The walk is iterative and
 * visits each flow object once, so a deep or shared child neither overflows the stack nor costs
 * one visit per path.
 *
 * @param flow The flow to measure; read through `nodes` and each node's `originChildren()`.
 * @returns The number of nested bodies on the longest chain; 0 for a flow with no child flow.
 */
export function callNesting(flow: NestingFlow): number {
  // Keyed by identity: two equal-looking flows are still two bodies.
  const depths = new Map<object, number>();
  const children = new Map<object, NestingFlow[]>();
  const stack: NestingFlow[] = [flow];
  while (stack.length > 0) {
    const top = stack[stack.length - 1]!;
    if (depths.has(top)) {
      stack.pop();
      continue;
    }
    let kids = children.get(top);
    if (kids === undefined) {
      kids = [];
      for (const node of top.nodes.values()) {
        for (const child of node.originChildren()) kids.push(child as NestingFlow);
      }
      children.set(top, kids);
      for (const kid of kids) if (!depths.has(kid)) stack.push(kid);
      continue;
    }
    stack.pop();
    let depth = 0;
    for (const kid of kids) depth = Math.max(depth, 1 + depths.get(kid)!);
    depths.set(top, depth);
  }
  return depths.get(flow)!;
}

/**
 * Resolve a callable to its child flow through the resolver, loud on failure.
 *
 * @param nodeId The calling node's id, named in messages.
 * @param flowId The callable the node names.
 * @param resolver The load-time child resolver.
 * @param childLabel Author-facing name of the callable; `null` names it by `flowId`.
 * @param options.diagnostics The context errors are raised in; a fresh one when not given.
 * @param options.range Location of the call in the owning document.
 * @returns The resolved child flow.
 * @throws {LoadError} When the callable can't be resolved, or the call nests bodies deeper than
 *   `current().maxCallNesting` levels.
 */
export function resolveChild(
  nodeId: string,
  flowId: string,
  resolver: ChildResolver,
  childLabel: string | null = null,
  options: { diagnostics?: LoadDiagnostics | null; range?: SourceRange | null } = {},
): ResolvedChild {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const range = options.range ?? null;
  const display = callableDisplay(flowId, childLabel);
  const limit = current().maxCallNesting;

  const tooDeep = (levels: string): LoadError =>
    diagnostics.error(
      `call node ${reprText(nodeId)}: calling ${display} nests calls ${levels} deep, past the ` +
        `call nesting limit of ${limit} (each call, map, or loop body is a ` +
        "level); flatten the chain",
      { range },
    );

  const depth = activeCalls;
  // Checked before resolving: building the callee would recurse another level.
  if (depth >= limit) throw tooDeep(`more than ${limit} levels`);
  activeCalls = depth + 1;
  let child: ResolvedChild;
  try {
    child = resolver(flowId, null);
  } catch (error) {
    if (error instanceof LoadError) throw error;
    // Any other resolver failure means the callable is unknown.
    const text = error instanceof Error ? error.message : String(error);
    throw diagnostics.error(`call node ${reprText(nodeId)}: could not resolve ${display}: ${text}`, {
      range,
      cause: error,
    });
  } finally {
    activeCalls = depth;
  }
  // A callee built earlier (a cached def, an imported file) never ran on this count.
  const levels = depth + 1 + callNesting(child.compiled);
  if (levels > limit) throw tooDeep(`${levels} levels`);
  return child;
}
