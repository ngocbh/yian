/** Deterministic identifiers shared by graph-expanding nodes and compilers. */

/**
 * Namespace a child node/edge id under its callsite: `<callsite>/<childId>`. `callsite` is the
 * spawner id (call/agent) or `` `${spawner}#${i}` `` (MAP element `i`); it nests.
 */
export function ns(callsite: string, childId: string): string {
  return `${callsite}/${childId}`;
}

/** The per-element callsite for MAP element `i`: `` `${spawnerId}#${i}` ``. */
export function mapCallsite(spawnerId: string, i: number): string {
  return `${spawnerId}#${i}`;
}

/**
 * Return the static path a run's node id stands for: the id with each `/` segment's MAP element
 * (`#<i>`) and LOOP iteration (`~<k>`) suffixes dropped, so `each#1/echo` is `each/echo` and
 * `node~2` is `node`. Authored ids are identifiers, so neither suffix is ever part of one.
 */
export function staticPathOf(nodeId: string): string {
  return nodeId
    .split("/")
    .map((segment) => segment.replace(/(?:[#~]\d+)+$/, ""))
    .join("/");
}
