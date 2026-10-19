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
