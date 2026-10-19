/**
 * Derive and atomically apply memory-independent static execution scopes.
 *
 * This module assigns root-relative authored paths to private compiled node templates.
 * Resolution is pure; application validates the complete layout and its current targets before
 * stamping any node. LOOP templates begin at driver zero; all other dynamic occurrence paths
 * belong to growth preparation.
 *
 * Knows about:   compile/model, nodes/base, nodes/contracts, nodes/start, nodes/end, diagnostics and typesys.
 * Never imports: memory, compose, runtime, or the package-root run boundary.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { Node } from "../nodes/base.ts";
import {
  ExecutionPosition,
  ExecutionScope,
  isExactly,
  sameExecutionPosition,
  sameExecutionScope,
  sameItems,
} from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import { writeAsText } from "../typesys/values.ts";
import { CompiledFlow, Flow } from "./model.ts";

/** Replay exact nested invariants before accepting an existing frozen scope. */
function revalidateScope(scope: unknown): asserts scope is ExecutionScope {
  if (!isExactly(scope, ExecutionScope)) throw new TypeError("execution scope must be an exact ExecutionScope");
  new ExecutionScope(scope.staticPath, {
    flowPath: scope.flowPath,
    occurrencePath: scope.occurrencePath,
    boundary: scope.boundary,
  });
}

/** One exact static node target and its root-relative execution scope. Frozen. */
export class ResolvedExecutionTarget {
  /** The node's root-relative authored path; a frozen copy. */
  readonly locator: readonly string[];
  /** The node's execution scope. */
  readonly scope: ExecutionScope;
  /** The live node the scope is stamped on; not part of a layout's value. */
  readonly node: Node;

  /**
   * Validate the locator, the static scope and the live target class.
   *
   * @throws TypeError for a locator component that isn't a non-empty string, a scope that isn't
   *   an exact `ExecutionScope`, or a node that isn't a `Node`.
   * @throws RangeError for an empty or non-array locator, a scope whose paths don't match the
   *   locator, or an occurrence path other than a LOOP's initial driver position.
   */
  constructor(locator: readonly string[], scope: ExecutionScope, node: Node) {
    if (!Array.isArray(locator) || locator.length === 0) {
      throw new RangeError("target locator must be a non-empty array");
    }
    if (locator.some((part) => typeof part !== "string" || !part)) {
      throw new TypeError("target locator components must be non-empty exact strings");
    }
    revalidateScope(scope);
    if (!sameItems(scope.staticPath, locator)) throw new RangeError("target scope static_path must match its locator");
    if (!sameItems(scope.flowPath, locator.slice(0, -1))) {
      throw new RangeError("target scope flow_path must name its containing flow");
    }
    if (!(node instanceof Node)) throw new TypeError("target node must be a Node");
    const initial = node.isLoop ? [new ExecutionPosition("loop_driver", locator, 0)] : [];
    if (!sameItems(scope.occurrencePath, initial, sameExecutionPosition)) {
      throw new RangeError("static execution layout requires only the initial driver position");
    }
    this.locator = Object.freeze([...locator]);
    this.scope = scope;
    this.node = node;
    Object.freeze(this);
  }
}

/** Immutable static execution targets ready for atomic application. Frozen. */
export class ResolvedExecutionLayout {
  /** The targets in resolution order; a frozen copy. */
  readonly targets: readonly ResolvedExecutionTarget[];

  /**
   * Require an array of exact targets and replay every target's invariants.
   *
   * @throws TypeError when `targets` isn't an array of exact `ResolvedExecutionTarget`s, or as
   *   the target constructor does.
   * @throws RangeError as the target constructor does.
   */
  constructor(targets: readonly ResolvedExecutionTarget[]) {
    if (!Array.isArray(targets) || targets.some((target) => !isExactly(target, ResolvedExecutionTarget))) {
      throw new TypeError("targets must be an array of ResolvedExecutionTarget values");
    }
    for (const target of targets) new ResolvedExecutionTarget(target.locator, target.scope, target.node);
    this.targets = Object.freeze([...targets]);
    Object.freeze(this);
  }
}

/** Whether two layouts have equal locators and scopes, target by target; nodes aren't compared. */
function sameLayout(a: ResolvedExecutionLayout, b: ResolvedExecutionLayout): boolean {
  return sameItems(
    a.targets,
    b.targets,
    (x, y) => sameItems(x.locator, y.locator) && sameExecutionScope(x.scope, y.scope),
  );
}

/** Return a validated deterministic view of one flow's direct nodes. */
function flowNodes(flow: Flow): [string, Node][] {
  if (!isExactly(flow, Flow) && !isExactly(flow, CompiledFlow)) {
    throw new TypeError("execution layout expects an exact Flow or CompiledFlow");
  }
  if (!isExactly(flow.nodes, Map)) throw new TypeError("flow nodes must be an exact Map");
  for (const [name, value] of [["start_id", flow.startId], ["end_id", flow.endId]] as const) {
    if (typeof value !== "string" || !value) throw new RangeError(`flow ${name} must be a non-empty exact string`);
  }
  const ids = [...flow.nodes.keys()];
  for (const nodeId of ids) {
    if (typeof nodeId !== "string" || !nodeId) throw new RangeError("flow node ids must be non-empty exact strings");
  }
  return ids.sort(compareCodeUnits).map((nodeId) => {
    const node = flow.nodes.get(nodeId);
    if (!(node instanceof Node)) throw new TypeError(`flow node ${reprText(nodeId)} must be a Node`);
    if (typeof node.id !== "string" || node.id !== nodeId) {
      throw new RangeError(`flow node key ${reprText(nodeId)} must match its exact node id`);
    }
    return [nodeId, node];
  });
}

/** Read the existing node-owned static-child hook without kind dispatch. */
function nodeChild(node: Node): Flow | null {
  const children: unknown = node.originChildren();
  if (!Array.isArray(children)) throw new TypeError(`node ${reprText(node.id)} static children must be an exact array`);
  if (children.length > 1) throw new RangeError(`node ${reprText(node.id)} exposes more than one static child`);
  if (children.length === 0) return null;
  const child: unknown = children[0];
  if (!isExactly(child, Flow) && !isExactly(child, CompiledFlow)) {
    throw new TypeError(`node ${reprText(node.id)} static child must be an exact Flow`);
  }
  return child;
}

/** Traverse every callsite while detecting recursive child-flow cycles. */
function* walkTargets(
  flow: Flow,
  flowPath: readonly string[],
  activeFlows: Set<Flow>,
): Generator<ResolvedExecutionTarget> {
  if (activeFlows.has(flow)) throw new RangeError("recursive child-flow cycle is not execution-layout resolvable");
  activeFlows.add(flow);
  try {
    for (const [nodeId, node] of flowNodes(flow)) {
      const locator = [...flowPath, nodeId];
      let boundary: "start" | "end" | null = null;
      if (nodeId === flow.startId && isExactly(node, StartNode)) boundary = "start";
      else if (nodeId === flow.endId && isExactly(node, EndNode)) boundary = "end";
      yield new ResolvedExecutionTarget(
        locator,
        new ExecutionScope(locator, {
          flowPath,
          boundary,
          occurrencePath: node.isLoop ? [new ExecutionPosition("loop_driver", locator, 0)] : [],
        }),
        node,
      );
      const child = nodeChild(node);
      if (child !== null) yield* walkTargets(child, locator, activeFlows);
    }
  } finally {
    activeFlows.delete(flow);
  }
}

/**
 * Purely derive execution scopes without reading memory declarations or flow ids.
 *
 * @param flow The exact static graph (`Flow` or `CompiledFlow`) to inspect. Missing artificial
 *   START/END nodes do not prevent layout derivation.
 * @returns Deterministic root-relative node targets.
 * @throws TypeError if graph or static-child carriers have noncanonical types.
 * @throws RangeError if a graph identifier is malformed or static children cycle.
 */
export function resolveExecutionLayout(flow: Flow): ResolvedExecutionLayout {
  return new ResolvedExecutionLayout([...walkTargets(flow, [], new Set())]);
}

/** Reject duplicate paths or one node object reached through several paths. */
function validateUniqueTargets(layout: ResolvedExecutionLayout): void {
  // Keyed by the JSON of the locator, which no two distinct string paths share; never shown.
  const locators = new Set<string>();
  const nodes = new Set<Node>();
  for (const target of layout.targets) {
    const key = JSON.stringify(target.locator);
    if (locators.has(key)) {
      throw new RangeError(`execution layout has duplicate target path ${writeAsText([...target.locator])}`);
    }
    locators.add(key);
    if (nodes.has(target.node)) {
      throw new RangeError(
        "execution layout reaches one node object through multiple static paths; " +
          "use private child templates before applying it",
      );
    }
    nodes.add(target.node);
  }
}

/**
 * Apply one validated static layout to its exact graph, all or nothing.
 *
 * @param flow The private graph (`Flow` or `CompiledFlow`) used during resolution.
 * @param layout The layout to revalidate and stamp. Reapplying the same layout is idempotent.
 * @throws TypeError if any layout, graph, or existing scope is noncanonical.
 * @throws RangeError if targets are stale, aliased, or already scoped differently.
 */
export function applyExecutionLayout(flow: Flow, layout: ResolvedExecutionLayout): void {
  if (!isExactly(layout, ResolvedExecutionLayout)) {
    throw new TypeError("layout must be an exact ResolvedExecutionLayout");
  }
  // Replaying the constructor checks a layout that may have bypassed it.
  new ResolvedExecutionLayout(layout.targets);
  validateUniqueTargets(layout);
  const current = resolveExecutionLayout(flow);
  validateUniqueTargets(current);
  if (!sameLayout(current, layout)) throw new RangeError("execution layout no longer matches the compiled flow");

  for (const [i, target] of layout.targets.entries()) {
    const actual = current.targets[i]!;
    if (actual.node !== target.node) {
      throw new RangeError(
        `execution layout target ${writeAsText([...target.locator])} no longer names the same node`,
      );
    }
    const existing = actual.node.executionScope;
    if (existing !== null) {
      revalidateScope(existing);
      if (!sameExecutionScope(existing, target.scope)) {
        throw new RangeError(`node at ${writeAsText([...target.locator])} already has a different execution scope`);
      }
    }
  }

  for (const target of layout.targets) target.node.executionScope = target.scope;
}
