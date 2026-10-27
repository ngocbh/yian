/**
 * Small runtime-owned scheduler vocabulary shared by runtime modules.
 *
 * This module has no project dependencies so runtime state and its helpers can agree on execution
 * dispositions without assigning that runtime concern to the compiled-flow model.
 */

/**
 * Per-node and per-edge scheduling state tracked by the runtime.
 *
 * `UNKNOWN` is undecided, `TAKEN` is committed, `SKIPPED` will not run, and `EXPANDED` marks a
 * spawner whose returned growth was applied.
 */
export const NodeState = { UNKNOWN: "unknown", TAKEN: "taken", SKIPPED: "skipped", EXPANDED: "expanded" } as const;
/** One per-node or per-edge scheduling state. */
export type NodeState = (typeof NodeState)[keyof typeof NodeState];
