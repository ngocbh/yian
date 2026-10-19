/**
 * The node base contract: a node is a pure function of its bound input record.
 *
 * A node implements `async run(inputs, caps)` and resolves to ONE of the closed sum
 * `Output | Route | Grow` (`NodeResult`). The node never receives the pool: the engine's
 * evaluator binds its inputs (the read boundary) and hands it a record. Narrow effects enter
 * through one capabilities object, `caps`, holding only what the node's traits ask for: MAP
 * receives `bindItem`, a node that records durable steps receives `steps`, and every node may
 * read its own abort `signal`. Failure is not a variant: a failing node rejects, and the
 * evaluator turns an expected rejection into a canonical failure retained by the accepted
 * failure record.
 *
 * Invariant: a node never writes the pool. It describes its one output value as
 * `new Output(value)`; the engine performs the write under the node's id. This keeps nodes pure.
 *
 * Knows about:   diagnostics, expr, memory contracts, typesys, nodes/binding, nodes/contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

/** The closed vocabulary of node kinds. Dispatch is an explicit `switch`, never a registry. */
export const NodeKind = {
  AGENT: "agent",
  CODE: "code",
  TOOL: "tool",
  CASE: "case",
  /** Blocks through the host-owned `ask_user` tool. */
  HUMAN_INPUT: "human_input",
  /** Authorable: blocks until an absolute UTC date or datetime. */
  WAIT: "wait",
  /** Authorable: iterates a callable to a fixpoint (`('a -> 'a) -> 'a -> 'a`). */
  LOOP: "loop",
  /** Internal-only: the loader-synthesized input boundary (parameter binding). */
  START: "start",
  /** Internal-only: the loader-synthesized return boundary (record and list modes). */
  END: "end",
  /** Authorable: consults another flow once (`kind: call`). */
  CALL: "call",
  /** Authorable: maps a callable over a list (`kind: map` with `over:`). */
  MAP: "map",
} as const;

export type NodeKind = (typeof NodeKind)[keyof typeof NodeKind];

/**
 * A produced value that the dispatcher records before the reducer binds it. Frozen.
 *
 * `commitAs` is the node-chosen commit redirect: when set, the reducer writes `value` under
 * `commitAs` instead of the node's own id and fires that target's out-edges; `null` (the common
 * case) commits under the node's own id. LOOP uses it when its final driver commits under the
 * compiled origin; the baked `Node.commitAs` field covers child-flow terminal redirects.
 */
export class Output {
  /** The produced value. */
  readonly value: unknown;
  /** The target id to commit under instead of the node's own id, or `null`. */
  readonly commitAs: string | null;

  /** Describe one produced value; `value` and `commitAs` default to `null`. */
  constructor(value: unknown = null, options: { commitAs?: string | null } = {}) {
    this.value = value;
    this.commitAs = options.commitAs ?? null;
    Object.freeze(this);
  }
}

/**
 * A routing-only outcome (a terminal) carrying only the chosen case handle: no value, no
 * post-asserts. Frozen. A router (CASE) returns it; the dispatcher records the handle and the
 * reducer takes its out-edge while skip-flooding siblings.
 */
export class Route {
  /** The chosen case handle. */
  readonly handle: string;

  /** Describe one routing choice. */
  constructor(handle: string) {
    this.handle = handle;
    Object.freeze(this);
  }
}

// Labeled branches are named by their target node id, and no authored node id may hold `#`, so
// no labeled branch can share the else route.
/**
 * The handle a router takes when no labeled branch matches. Unlike any other handle it may have
 * no outgoing edge.
 */
export const ELSE_HANDLE = "#else";

/**
 * One deterministic graph expansion a spawner describes for the runtime to record. Frozen.
 *
 * `seed` is the pure input to the compile-owned growth builder: kind-shaped but opaque to the
 * engine (a CALL record, MAP records list, or LOOP record and index). The same canonical value
 * is persisted, so durable replay rebuilds identical topology without re-running the node that
 * produced it. `prune` holds the ids to retire in the same step (empty for CALL and MAP; a
 * self-respawning LOOP's finished iteration ids). The reducer retires a pruned id with its whole
 * grown subtree once every node in it is settled; until then it stays live.
 */
export class Grow {
  /** The pure input to the compile-owned growth builder. */
  readonly seed: unknown;
  /** The ids to retire in the same step; a private copy, read-only by contract. */
  readonly prune: ReadonlySet<string>;

  /** Describe one expansion; `prune` is copied into a new set and defaults to empty. */
  constructor(seed: unknown, prune: Iterable<string> = []) {
    this.seed = seed;
    this.prune = new Set(prune);
    Object.freeze(this);
  }
}

/** The closed sum a node's `run` resolves to. */
export type NodeResult = Output | Route | Grow;
