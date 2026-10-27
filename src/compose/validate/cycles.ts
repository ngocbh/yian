/**
 * Graph stage: reject a directed cycle in the inferred (real-node) graph.
 *
 * A cross-node assembly pass. After all data and ordering edges are inferred, a flow graph must
 * be a DAG: the scheduler resolves nodes by settling their producers, so a loop (`a -> b -> a`)
 * can never make progress. This pass uses Kahn's algorithm to detect a cycle, then isolates the
 * cyclic strongly connected components so downstream blocked nodes are not blamed. The boundary
 * sentinels (`__start__`/`__end__`) are acyclic by construction, so passing the fully assembled
 * edge set is fine.
 *
 * The caller passes `nodeRanges` (node id to `SourceRange`). The cycle diagnostic carries one
 * frame per located named node in authored source order; a generated node can be named as the
 * authored host it came from. Its notes form a "why" legend: one line per dependency edge inside
 * a cyclic component, so the author sees which `${...}` reference closes the cycle.
 *
 * Knows about: the compile model (`Edge` and the boundary ids), the shared Kahn check, and load
 * diagnostics.
 * Never imports: build, the loader, or the runtime.
 */

import { END_ID, START_ID, type Edge } from "../../compile/model.ts";
import { FlowValidationError, rejectCycles as rejectGraphCycles } from "../../compile/validation.ts";
import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { writeAsText } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";

/**
 * Throw a `LoadError` if the inferred (real-node) graph has a directed cycle.
 *
 * Delegates detection to the shared Kahn check over the non-sentinel edges, then identifies the
 * actual cyclic strongly connected components: a node blocked downstream of a cycle is not a
 * cycle member. Sentinel `__start__`/`__end__` edges cannot close a cycle.
 *
 * Every located node named in the error gets one frame, ordered by authored line and column; a
 * synth node reported as its host shares the host's frame.
 *
 * The error also carries a "why" legend (`Diagnostic.notes`): one line per dependency edge
 * inside the loop, "<consumer> depends on <producer> (<consumer>.input.<group>)", so the author
 * sees which reference closes the cycle, not just which nodes are in it. A synth node reported as
 * its host is named as the host, its generated wire is left out, and duplicate lines collapse.
 *
 * @param options.nodeRanges Node id to its authored range (default none).
 * @param options.diagnostics Locates the error (default: an unlocated one).
 * @param options.gateIds The data gates a lifted node got from its host (group
 *   `<node>.<group>`); their notes cite the host's input, "(gate: <node>.input.<group>)".
 * @param options.synthHosts Synth node id to the authored host it was generated from. A synth
 *   node is reported as that host when only its generated wire into the host closes the cycle;
 *   otherwise it keeps its id and a note names its host.
 * @param options.synthWires Ids of those generated wire edges from a synth node into its host.
 * @throws LoadError naming the cycle's nodes, sorted, in the text form.
 */
export function rejectCycles(
  edges: readonly Edge[],
  nodeIds: ReadonlySet<string>,
  options: {
    nodeRanges?: ReadonlyMap<string, SourceRange> | null;
    diagnostics?: LoadDiagnostics | null;
    gateIds?: ReadonlySet<string>;
    synthHosts?: ReadonlyMap<string, string> | null;
    synthWires?: ReadonlySet<string>;
  } = {},
): void {
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const ranges = options.nodeRanges ?? new Map<string, SourceRange>();
  const synthWires = options.synthWires ?? new Set<string>();
  try {
    rejectGraphCycles(edges, nodeIds);
  } catch (error) {
    if (!(error instanceof FlowValidationError)) throw error;
    const components = cyclicComponents(edges, nodeIds);
    const cycleNodes = new Set(components.flatMap((component) => [...component]));
    const alias = new Map<string, string>();
    const synthNotes: string[] = [];
    for (const [synth, host] of options.synthHosts ?? []) {
      if (!cycleNodes.has(synth)) continue;
      const component = components.find((c) => c.has(synth))!;
      const inside = edges.filter((e) => e.from === synth && component.has(e.to));
      if (inside.every((e) => synthWires.has(e.id))) {
        alias.set(synth, host);
      } else {
        synthNotes.push(`${synth} is generated from node ${reprText(host)}`);
      }
    }
    const named = [...new Set([...cycleNodes].map((n) => alias.get(n) ?? n))].sort(compareCodeUnits);
    // A named host takes its own range; an aliased member stands in only without one.
    const located = new Map<string, SourceRange>();
    const members = [...cycleNodes].sort(
      (a, b) => Number(alias.has(a)) - Number(alias.has(b)) || compareCodeUnits(a, b),
    );
    for (const nodeId of members) {
      const range = ranges.get(nodeId);
      const name = alias.get(nodeId) ?? nodeId;
      if (range !== undefined && !located.has(name)) located.set(name, range);
    }
    const authored = [...located].sort(
      ([a, ra], [b, rb]) => ra.startLine - rb.startLine || ra.startCol - rb.startCol || compareCodeUnits(a, b),
    );
    const frames = authored.flatMap(([, range]) => diagnostics.frame(range) ?? []);
    const notes = cycleNotes(edges, components, { gateIds: options.gateIds, aliases: alias });
    for (const note of synthNotes) notes.add(note);
    throw diagnostics.error(`flow has a cycle involving ${writeAsText(named)}; flows must be acyclic`, {
      frames,
      notes: [...notes].sort(compareCodeUnits),
      cause: error,
    });
  }
}

/**
 * Return one note per dependency edge within a cyclic component, unsorted; the caller sorts.
 *
 * Each loop-internal edge `from -> to` reads as "`to` consumes `from`": the consumer depends on
 * the producer. An edge connecting two separate cycles is not itself cyclic and is omitted. A
 * data edge cites `to.input.<group>`; a pure ordering edge (`depends_on`/`runs_after`, no data)
 * is labelled as such. An edge whose id is in `gateIds` is a lifted node's data gate, whose group
 * is `<node>.<group>`: it cites `gate: <node>.input.<group>`, the host input it mirrors.
 *
 * @param options.aliases Synth node id to the host it is reported as. Notes name an aliased node
 *   as its host, and its generated wire into that host is omitted.
 */
function cycleNotes(
  edges: readonly Edge[],
  components: readonly ReadonlySet<string>[],
  options: { gateIds?: ReadonlySet<string>; aliases?: ReadonlyMap<string, string> } = {},
): Set<string> {
  const gateIds = options.gateIds ?? new Set<string>();
  const alias = options.aliases ?? new Map<string, string>();
  const componentByNode = new Map<string, number>();
  components.forEach((component, index) => {
    for (const nodeId of component) componentByNode.set(nodeId, index);
  });
  const notes = new Set<string>();
  for (const e of edges) {
    const component = componentByNode.get(e.from);
    if (component === undefined || componentByNode.get(e.to) !== component) continue;
    // The desugar's own wire from a synth node into its host, not an authored dependency.
    if (alias.get(e.from) === e.to) continue;
    const to = alias.get(e.to) ?? e.to;
    const from = alias.get(e.from) ?? e.from;
    let via: string;
    if (gateIds.has(e.id) && e.inputGroup) {
      // Only gate ids are split: an authored input key may itself contain a `.`.
      const [node, ...rest] = e.inputGroup.split(".");
      via = `gate: ${node}.input.${rest.join(".")}`;
    } else if (e.inputGroup) {
      via = `${to}.input.${e.inputGroup}`;
    } else {
      via = "ordering";
    }
    notes.add(`${to} depends on ${from} (${via})`);
  }
  return notes;
}

/**
 * Return the graph's cyclic strongly connected components, in the order the second pass finds them
 * (no caller depends on the order).
 *
 * Two iterative depth-first passes implement Kosaraju's algorithm with no recursion. Components
 * with several nodes are cyclic; a singleton is cyclic only when it has a self-edge. Sentinel
 * edges are ignored.
 */
function cyclicComponents(edges: readonly Edge[], nodeIds: ReadonlySet<string>): Set<string>[] {
  const adjacency = new Map<string, Set<string>>();
  const reverse = new Map<string, Set<string>>();
  for (const nodeId of nodeIds) {
    adjacency.set(nodeId, new Set());
    reverse.set(nodeId, new Set());
  }
  for (const edge of edges) {
    if (edge.from === START_ID || edge.from === END_ID || edge.to === START_ID || edge.to === END_ID) continue;
    adjacency.get(edge.from)!.add(edge.to);
    reverse.get(edge.to)!.add(edge.from);
  }
  const reverseSorted = (nodes: ReadonlySet<string>) => [...nodes].sort(compareCodeUnits).reverse();

  const visited = new Set<string>();
  const finishOrder: string[] = [];
  // Starts are visited sorted and neighbours pushed reverse-sorted so the finish order, and with it
  // the order the components come out in, is deterministic.
  for (const start of [...nodeIds].sort(compareCodeUnits)) {
    if (visited.has(start)) continue;
    const stack: [string, boolean][] = [[start, false]];
    while (stack.length > 0) {
      const [nodeId, expanded] = stack.pop()!;
      if (expanded) {
        finishOrder.push(nodeId);
        continue;
      }
      if (visited.has(nodeId)) continue;
      visited.add(nodeId);
      stack.push([nodeId, true]);
      for (const target of reverseSorted(adjacency.get(nodeId)!)) {
        if (!visited.has(target)) stack.push([target, false]);
      }
    }
  }

  const assigned = new Set<string>();
  const components: Set<string>[] = [];
  for (const start of finishOrder.reverse()) {
    if (assigned.has(start)) continue;
    const component = new Set<string>();
    const stack = [start];
    assigned.add(start);
    while (stack.length > 0) {
      const nodeId = stack.pop()!;
      component.add(nodeId);
      for (const source of reverseSorted(reverse.get(nodeId)!)) {
        if (!assigned.has(source)) {
          assigned.add(source);
          stack.push(source);
        }
      }
    }
    if (component.size > 1 || adjacency.get(start)!.has(start)) components.push(component);
  }

  return components;
}
