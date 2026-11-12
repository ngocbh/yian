/**
 * Graph stage: a lifted node takes its host's CASE routes, ordering, and data gates.
 *
 * A desugar pass that lifts a node out of an authored node records that node as the lifted node's
 * `host` (an inline `call(...)` in a node's binding, an adaptive gate's composer, the inner calls
 * of a nested branch `call(...)`). The lifted node runs only on its host's routes and after its
 * host's ordering: it gets a copy of every CASE route edge into the host (same `sourceHandle`, the
 * else handle included), and the ordering edges the host's `depends_on:` (co-skip) and
 * `runs_after:` (order only) declare. A host that is itself lifted passes on what it inherits, so
 * chains of lifts are followed.
 *
 * The lifted node also waits for the host's other data sources and is skipped exactly when the
 * host's co-skip rule would skip the host. For each data input group of the host it gets *data
 * gates*: ordinary data edges from that group's producers, all in one input group named
 * `<node>.<group>` (the node whose input it is, then that input's group). Grouped like the host's
 * input, the runtime join applies the host's rule to them: a required group is dead once all its
 * producers are skipped, a defaulted one (`${x.output :- d}`) only waits. They are data-shaped
 * because a coalesce (`${a.output | b.output}`, skipped only when every producer is) cannot be
 * written as ordering edges, which co-skip one by one. The lifted node binds its inputs from
 * wiring, so a gate never feeds a value.
 *
 * A producer is *family* when it is the lifted node, one of its hosts, or a node lifted out of one
 * of those hosts. A group whose only producer is family gives no gate: siblings lifted from the
 * same host don't wait on each other. A required group read from one family node is instead
 * looked through: the lifted node gets that node's own data gates (unless the lifted node itself
 * requires that node as an input, so its inputs already gate the lifted node). Each ancestor host
 * is visited the same way, so transitive lifts gate on every host up the chain.
 *
 * A gate group is dropped as redundant when the lifted node already depends on all its producers
 * and either the gate is optional or one of the lifted node's required groups lies inside it, so
 * those producers already skip the lifted node.
 *
 * One case is not an exact match: a family producer in a defaulted or multi-producer group, which
 * only happens when an author names a lifted node's id (`${__call_0.output | t.output}`). Its
 * other producers then only make the lifted node wait, so it may run while its host is skipped; it
 * never skips when the host would run.
 *
 * An unknown `depends_on`/`runs_after` source is skipped here; `inferOrderingEdges` reports it at
 * the host, which is the authored node. An unknown data producer gives no gate either.
 *
 * Knows about: the compile model (`Edge`), diagnostics (`reprText`), and descriptors.
 */

import { Edge } from "../../compile/model.ts";
import { reprText } from "../../diagnostics.ts";
import { YianRangeError } from "../../typesys/errors.ts";
import type { NodeDescriptor } from "../nodes/base.ts";

/** One inherited edge before it gets a target and an id. */
interface Gate {
  source: string;
  handle: string | null;
  ordering: boolean;
  optional: boolean;
  group: string | null;
}

/**
 * Return the route, ordering, and data-gate edges each lifted node inherits from its host.
 *
 * For each node with a `host`, in descriptor order, emits a copy of every edge into the host that
 * has a `sourceHandle` (a CASE route), then one ordering edge per host `dependsOn` source
 * (`optional: false`) and per `runsAfter` source (`optional: true`), then whatever the host
 * inherits from its own host, then the node's data gates: one data edge per producer of each host
 * input group it must wait on, in input group `<node>.<group>` with that group's `optional`,
 * visiting the host, then its hosts and any looked-through family node, in that order. A source
 * that is not a node, or is the lifted node itself, is skipped. Route copies and data gates are
 * named `<source>-><node>#<i>` and ordering edges `<source>~><node>#<i>`, with `i` the first index
 * no edge in `edges` or already emitted uses.
 *
 * @example
 * With a CASE `c` routing `then: publish` and `publish` reading `call(side)` lifted as `__call_0`
 * (host `publish`) and `${maybe.output}` as `notes`, the result holds
 * `new Edge("c->__call_0#0", "c", "__call_0", {sourceHandle: "publish"})` and
 * `new Edge("maybe->__call_0#0", "maybe", "__call_0", {inputGroup: "publish.notes"})`.
 *
 * @param descriptors The flow level's desugared descriptors by node id.
 * @param edges The edges built so far (data edges and each CASE's own edges). Route and data
 *   edges are read from it; every id in it is taken.
 * @returns The inherited edges, new ones only.
 * @throws RangeError when a host is not a node or hosts form a cycle, or when a lifted node has a
 *   route edge, `dependsOn`, or `runsAfter` of its own. Desugar never builds these, so each is a
 *   compiler bug.
 */
export function hostControlEdges(descriptors: ReadonlyMap<string, NodeDescriptor>, edges: readonly Edge[]): Edge[] {
  /** Append `edge` to `map`'s list under `key`. */
  const append = <K>(map: Map<K, Edge[]>, key: K, edge: Edge): void => {
    const list = map.get(key);
    if (list === undefined) map.set(key, [edge]);
    else list.push(edge);
  };

  const routesInto = new Map<string, Edge[]>();
  const dataInto = new Map<string, Edge[]>();
  for (const edge of edges) {
    if (edge.sourceHandle !== null) append(routesInto, edge.to, edge);
    else if (!edge.ordering) append(dataInto, edge.to, edge);
  }

  /** Whether every member of `inner` is in `outer`. */
  const within = (inner: ReadonlySet<string>, outer: ReadonlySet<string>): boolean =>
    [...inner].every((member) => outer.has(member));

  /** `host`, checked to be a node of this level. */
  const checkedHost = (nodeId: string, host: string): string => {
    if (!descriptors.has(host)) {
      throw new YianRangeError(`lifted node ${reprText(nodeId)} names host ${reprText(host)}, which is not a node`);
    }
    return host;
  };

  const memo = new Map<string, Gate[]>();

  /** The route and ordering gates of `nodeId`, inherited ones included. */
  const gating = (nodeId: string, chain: readonly string[]): Gate[] => {
    const known = memo.get(nodeId);
    if (known !== undefined) return known;
    if (chain.includes(nodeId)) {
      throw new YianRangeError(`lifted-node hosts form a cycle: ${[...chain, nodeId].join(" -> ")}`);
    }
    const desc = descriptors.get(nodeId)!;
    const gates: Gate[] = (routesInto.get(nodeId) ?? []).map((edge) => ({
      source: edge.from,
      handle: edge.sourceHandle,
      ordering: false,
      optional: edge.optional,
      group: null,
    }));
    for (const src of desc.dependsOn) {
      if (descriptors.has(src)) gates.push({ source: src, handle: null, ordering: true, optional: false, group: null });
    }
    for (const src of desc.runsAfter) {
      if (descriptors.has(src)) gates.push({ source: src, handle: null, ordering: true, optional: true, group: null });
    }
    if (desc.host !== null) gates.push(...gating(checkedHost(nodeId, desc.host), [...chain, nodeId]));
    memo.set(nodeId, gates);
    return gates;
  };

  /** `nodeId`'s incoming data edges grouped by input group, in edge order. */
  const inputGroups = (nodeId: string): Edge[][] => {
    const groups = new Map<string | null, Edge[]>();
    for (const edge of dataInto.get(nodeId) ?? []) append(groups, edge.inputGroup, edge);
    return [...groups.values()];
  };

  const chains = new Map<string, readonly string[]>();

  /** `nodeId`'s hosts, nearest first. */
  const ancestors = (nodeId: string): readonly string[] => {
    const known = chains.get(nodeId);
    if (known !== undefined) return known;
    const found: string[] = [];
    let current = nodeId;
    let host: string | null;
    while ((host = descriptors.get(current)!.host) !== null) {
      host = checkedHost(current, host);
      if (host === nodeId || found.includes(host)) {
        throw new YianRangeError(`lifted-node hosts form a cycle: ${[nodeId, ...found, host].join(" -> ")}`);
      }
      found.push(host);
      current = host;
    }
    chains.set(nodeId, found);
    return found;
  };

  /** The data gates of lifted node `nodeId`, in visiting order. */
  const dataGates = (nodeId: string): Gate[] => {
    const chain = ancestors(nodeId);
    const hosts = new Set(chain);
    const seen = new Set([nodeId, ...chain]);
    const queue = [...chain];
    const gates: Gate[] = [];
    const sources = new Set<string>();
    const required: ReadonlySet<string>[] = [];
    for (const group of inputGroups(nodeId)) {
      for (const edge of group) sources.add(edge.from);
      if (!group[0]!.optional) required.push(new Set(group.map((edge) => edge.from)));
    }

    /** Whether `source` is this node, a host of it, or lifted out of one of those. */
    const family = (source: string): boolean => {
      if (seen.has(source)) return true;
      const desc = descriptors.get(source);
      return desc !== undefined && desc.host !== null && ancestors(source).some((host) => hosts.has(host));
    };

    /** Gate on `producers` as one input group, unless that is redundant. */
    const add = (producers: readonly string[], group: string, optional: boolean): void => {
      const members = new Set(producers);
      if (members.size === 0) return;
      // The lifted node already waits on every producer, and either this gate only waits or a
      // required group of its own already skips it when this one would.
      if (within(members, sources) && (optional || required.some((own) => within(own, members)))) return;
      for (const member of members) sources.add(member);
      if (!optional) required.push(members);
      for (const source of producers) gates.push({ source, handle: null, ordering: false, optional, group });
    };

    for (let next = 0; next < queue.length; next++) {
      const current = queue[next]!;
      for (const group of inputGroups(current)) {
        const name = `${current}.${group[0]!.inputGroup}`;
        const optional = group[0]!.optional;
        const known = group.map((edge) => edge.from).filter((source) => descriptors.has(source));
        const kin = known.filter(family);
        if (kin.length === 0) {
          add(known, name, optional);
        } else if (group.length === 1 && !optional) {
          const source = group[0]!.from;
          if (!seen.has(source)) {
            seen.add(source);
            if (!required.some((own) => own.size === 1 && own.has(source))) queue.push(source);
          }
        } else {
          add(known.filter((source) => !kin.includes(source)), name, true);
        }
      }
    }
    return gates;
  };

  const taken = new Set(edges.map((edge) => edge.id));
  const inherited: Edge[] = [];
  for (const [nodeId, desc] of descriptors) {
    if (desc.host === null) continue;
    // Control edges join by OR, so an own route would loosen the inherited gate, and own ordering
    // fields would get ids from `inferOrderingEdges` that may collide with these.
    if ((routesInto.get(nodeId) ?? []).length > 0 || desc.dependsOn.length > 0 || desc.runsAfter.length > 0) {
      throw new YianRangeError(
        `lifted node ${reprText(nodeId)} has its own route or ordering edges; it may only inherit its host's`,
      );
    }
    const gates = [...gating(checkedHost(nodeId, desc.host), [nodeId]), ...dataGates(nodeId)];
    for (const gate of gates) {
      if (gate.source === nodeId) continue;
      const arrow = gate.ordering ? "~>" : "->";
      let index = 0;
      while (taken.has(`${gate.source}${arrow}${nodeId}#${index}`)) index += 1;
      const edgeId = `${gate.source}${arrow}${nodeId}#${index}`;
      taken.add(edgeId);
      inherited.push(
        new Edge(edgeId, gate.source, nodeId, {
          sourceHandle: gate.handle,
          ordering: gate.ordering,
          optional: gate.optional,
          inputGroup: gate.group,
        }),
      );
    }
  }
  return inherited;
}
