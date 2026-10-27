/**
 * Identity-safe traversal and rebasing across compiled origin graphs.
 *
 * The compile layer owns whole-graph shape: flow wiring, declared outputs, node identity, and
 * recursively baked child flows. Nodes own their local provenance slots and expose static
 * children through source-neutral hooks. This module joins those responsibilities without
 * importing Compose, resolving a source, or inspecting arbitrary semantic payloads.
 *
 * Knows about:   compile/model, nodes, memory contracts, typesys and diagnostics.
 * Never imports: compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { OriginRef, type OriginResolver, ResolverTable, reprText } from "../diagnostics.ts";
import { iterMemoryDeclOrigins, rewriteMemoryDeclOrigins } from "../memory/contracts.ts";
import { Node } from "../nodes/base.ts";
import {
  Binding,
  iterOriginCarrier,
  type OriginMapper,
  rewriteOriginCarrier,
  ValueTemplate,
} from "../nodes/contracts.ts";
import { jsTypeName } from "../typesys/values.ts";
import { CompiledFlow, copyGraph, Flow, FlowOutput } from "./model.ts";

/**
 * Iterate declared provenance slots in one static flow graph.
 *
 * Traversal follows wiring and outputs, then nodes and their baked children in deterministic
 * structural order; a compiled flow's memory declarations and its `flowLlmConfig` layer holding
 * values come first. Flow, node, and frozen-carrier identities are visited once, so shared
 * children and deliberate object cycles terminate. Equal origins in distinct carriers remain
 * distinct occurrences. Binding sources other than value templates, parameter defaults, node
 * environments, and render metadata stay opaque.
 *
 * @param flow Compiled topology or compile-owned fragment to traverse.
 * @returns The present semantic origins; a source-free graph yields none.
 * @throws TypeError eagerly when `flow` isn't a `Flow`; while iterating, when a node or a static
 *   child isn't a compiled graph value.
 */
export function iterFlowOrigins(flow: Flow): Generator<OriginRef> {
  if (!(flow instanceof Flow)) throw new TypeError(`flow must be a Flow, got ${jsTypeName(flow)}`);
  return walkFlowOrigins(flow, new Set(), new Set(), new Set());
}

/** Yield one flow identity's origins and recursively visit each unseen static child. */
function* walkFlowOrigins(
  flow: Flow,
  seenFlows: Set<Flow>,
  seenNodes: Set<Node>,
  seenCarriers: Set<object>,
): Generator<OriginRef> {
  if (seenFlows.has(flow)) return;
  seenFlows.add(flow);

  if (flow instanceof CompiledFlow) {
    yield* iterMemoryDeclOrigins(flow.memories, seenCarriers);
    if (flow.flowLlmConfig instanceof Binding || flow.flowLlmConfig instanceof ValueTemplate) {
      yield* iterOriginCarrier(flow.flowLlmConfig, seenCarriers);
    }
  }

  for (const nodeWiring of flow.wiring.values()) {
    for (const binding of nodeWiring.values()) yield* iterOriginCarrier(binding, seenCarriers);
  }

  if (flow instanceof CompiledFlow) {
    for (const output of flow.outputs) {
      if (!(output instanceof FlowOutput)) {
        throw new TypeError(`compiled flow outputs must contain FlowOutput values, got ${jsTypeName(output)}`);
      }
      yield* iterOriginCarrier(output.binding, seenCarriers);
    }
  }

  for (const node of flow.nodes.values()) {
    if (!(node instanceof Node)) {
      throw new TypeError(`flow nodes must contain Node values, got ${jsTypeName(node)}`);
    }
    if (seenNodes.has(node)) continue;
    seenNodes.add(node);
    yield* node.iterLocalOrigins(seenCarriers);
    for (const child of node.originChildren()) {
      if (!(child instanceof Flow)) {
        throw new TypeError(`node ${reprText(node.id)} origin child must be a Flow, got ${jsTypeName(child)}`);
      }
      yield* walkFlowOrigins(child, seenFlows, seenNodes, seenCarriers);
    }
  }
}

/**
 * Return an identity-preserving copy of `flow` with every origin transformed.
 *
 * The authored input graph is never mutated. A `copyGraph` working copy retains shared topology
 * and cycles while provenance carriers are rebuilt once per carrier identity, so the mapper is
 * called once for each origin reached through a unique carrier or node slot. Ordinary payloads
 * stay opaque and are shared by identity with the input.
 *
 * @param flow Compiled topology or compile-owned fragment to copy and rewrite.
 * @param mapper Transform for every present semantic origin. It must return an `OriginRef`; it
 *   may throw to abort the rewrite.
 * @returns A graph of the same concrete flow type with preserved shared-object topology and
 *   rewritten origins.
 * @throws TypeError when `flow` isn't a `Flow`, `mapper` isn't callable, a mapper result isn't an
 *   `OriginRef`, or the static graph shape is invalid.
 */
export function rewriteFlowOrigins<F extends Flow>(flow: F, mapper: OriginMapper): F {
  if (!(flow instanceof Flow)) throw new TypeError(`flow must be a Flow, got ${jsTypeName(flow)}`);
  if (typeof mapper !== "function") {
    throw new TypeError(`origin mapper must be callable, got ${jsTypeName(mapper)}`);
  }

  // Validate static shape before copying, so a structural error wins before any mapper call.
  for (const _origin of iterFlowOrigins(flow)) {
    // Draining is the check.
  }

  const working = copyGraph(flow);
  rewriteFlowOriginGraph(working, mapper, new Set(), new Set(), new Map());
  return working;
}

/**
 * Atomically rewrite reachable origins and retain their resolution routes.
 *
 * Equal old keys are one namespace identity even when their `OriginRef` values have distinct
 * classes, so `mapper` runs once per distinct opaque key. The returned table routes each mapped
 * identity back through `resolver` with its exact old identity. Keys remain opaque to this
 * operation.
 *
 * @param flow Compiled topology or compile-owned fragment to copy.
 * @param mapper Namespace transform called once per distinct reachable old key.
 * @param options.resolver Authority for the old identities. Required for an originful graph and
 *   required to be `null` for a source-free graph.
 * @returns An independent graph copy of the same concrete type and exact routes for all of its
 *   mapped origins. A source-free graph returns an empty table.
 * @throws TypeError when the flow, mapper, resolver, mapper result, or static graph shape is
 *   invalid.
 * @throws RangeError when an origin carries an invalid key, resolver presence disagrees with graph
 *   provenance, or distinct old keys map to the same new key.
 */
export function rebaseFlowOrigins<F extends Flow>(
  flow: F,
  mapper: OriginMapper,
  { resolver }: { resolver: OriginResolver | null },
): [F, ResolverTable] {
  if (!(flow instanceof Flow)) throw new TypeError(`flow must be a Flow, got ${jsTypeName(flow)}`);
  if (typeof mapper !== "function") {
    throw new TypeError(`origin mapper must be callable, got ${jsTypeName(mapper)}`);
  }
  if (resolver !== null && typeof resolver?.resolve !== "function") {
    throw new TypeError("origin resolver must implement resolve");
  }

  // Materializing validates the entire static graph before a user mapper can have side effects.
  // Map insertion order retains the first structural visit, while the single-key origin contract
  // stays independent of subclasses.
  const oldByKey = new Map<string, OriginRef>();
  for (const oldOrigin of iterFlowOrigins(flow)) {
    const oldKey = new OriginRef(oldOrigin.key).key;
    if (!oldByKey.has(oldKey)) oldByKey.set(oldKey, oldOrigin);
  }
  const oldOrigins = [...oldByKey.values()];
  if (oldOrigins.length > 0 && resolver === null) {
    throw new RangeError("originful flow requires an origin resolver");
  }
  if (oldOrigins.length === 0 && resolver !== null) {
    throw new RangeError("source-free flow cannot have an origin resolver");
  }

  const mappedByOldKey = new Map<string, OriginRef>();
  const mappedKeys = new Set<string>();
  for (const oldOrigin of oldOrigins) {
    const mappedOrigin: unknown = mapper(oldOrigin);
    if (!(mappedOrigin instanceof OriginRef)) {
      throw new TypeError(`origin mapper must return an OriginRef, got ${jsTypeName(mappedOrigin)}`);
    }
    const newOrigin = new OriginRef(mappedOrigin.key);
    if (mappedKeys.has(newOrigin.key)) {
      throw new RangeError(`mapped origin collision ${reprText(newOrigin.key)}`);
    }
    mappedByOldKey.set(oldOrigin.key, newOrigin);
    mappedKeys.add(newOrigin.key);
  }

  const table =
    resolver === null
      ? new ResolverTable()
      : ResolverTable.fromEntries(
          oldOrigins.map((oldOrigin) => [mappedByOldKey.get(oldOrigin.key)!, resolver, oldOrigin] as const),
        );
  const rewritten = rewriteFlowOrigins(flow, (origin) => mappedByOldKey.get(origin.key)!);
  return [rewritten, table];
}

/**
 * Rewrite one copied flow in place and recurse through unseen children.
 *
 * `iterFlowOrigins` has already checked the graph's shape, so this walk trusts it.
 */
function rewriteFlowOriginGraph(
  flow: Flow,
  mapper: OriginMapper,
  seenFlows: Set<Flow>,
  seenNodes: Set<Node>,
  seen: Map<object, object>,
): void {
  if (seenFlows.has(flow)) return;
  seenFlows.add(flow);

  if (flow instanceof CompiledFlow) {
    flow.memories = rewriteMemoryDeclOrigins(flow.memories, mapper, seen);
    if (flow.flowLlmConfig instanceof Binding || flow.flowLlmConfig instanceof ValueTemplate) {
      flow.flowLlmConfig = rewriteOriginCarrier(flow.flowLlmConfig, mapper, seen);
    }
  }

  // The copy owns its outer wiring map, but the inner maps are shared with the input: replace them.
  for (const [nodeId, nodeWiring] of flow.wiring) {
    flow.wiring.set(
      nodeId,
      new Map([...nodeWiring].map(([name, binding]) => [name, rewriteOriginCarrier(binding, mapper, seen)])),
    );
  }

  if (flow instanceof CompiledFlow) {
    flow.outputs = flow.outputs.map((output) => {
      const done = seen.get(output);
      if (done !== undefined) return done as FlowOutput;
      // Memoized so a FlowOutput shared by two flows stays one object.
      const rebuilt = new FlowOutput(output.name, rewriteOriginCarrier(output.binding, mapper, seen));
      seen.set(output, rebuilt);
      return rebuilt;
    });
  }

  for (const node of flow.nodes.values()) {
    if (seenNodes.has(node)) continue;
    seenNodes.add(node);
    node.rewriteLocalOrigins(mapper, seen);
    for (const child of node.originChildren()) {
      rewriteFlowOriginGraph(child as Flow, mapper, seenFlows, seenNodes, seen);
    }
  }
}
