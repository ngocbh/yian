/**
 * CALL: the single-application driver (`kind: call`).
 *
 * `CallNode` applies a callable once (`'a flow -> 'b`). Its mapped counterpart is `MapNode` in
 * `nodes/map` (`kind: map` + `over:`, `list['a] -> list['b]`); the two are distinct typed
 * drivers. `CallNode` carries no `over` or `parallel`.
 *
 * `run` resolves to one seed-first `Grow` carrying the raw call-arg record. The compile-owned
 * injected builder constructs the child subgraph, and the generic growth reducer splices it. The
 * child START owns omitted-input defaulting (its params carry default and required), so the
 * driver doesn't pre-default. `child` and `childInputs` are baked at load by Compose;
 * `childInputs` is read by compile validation. The child's asserts ride its own compiled boundary
 * nodes and survive the deep copy of the spliced subgraph. The node also owns origin traversal
 * for its cached child signature and exposes its baked child through the source-neutral
 * `originChildren` hook.
 *
 * Knows about:   nodes/base, nodes/contracts and diagnostics.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { type OriginRef, reprText } from "../../diagnostics.ts";
import { Grow, Node, NodeKind } from "../base.ts";
import { iterOriginCarrier, type OriginMapper, type Parameter, rewriteOriginCarrier } from "../contracts.ts";

/**
 * The call driver (`kind: call`): apply a callable flow once (`'a flow -> 'b`).
 *
 * The single-call counterpart to `MapNode`. `run` resolves to one seed-first `Grow`; the
 * injected compile builder constructs the child fragment and the engine splices it. The spliced
 * child START owns omitted-input defaulting, so the driver passes call-args raw.
 */
export class CallNode extends Node {
  readonly kind = NodeKind.CALL;
  /** Grows the graph: `run` resolves to a seed-first `Grow`. */
  override readonly isSpawner: boolean = true;
  /** Each child is one call level deeper. */
  override readonly growDepthDelta: number | null = 1;
  /** Its seed only reads earlier pool values through its binding. */
  override readonly derived: boolean = true;
  /** The id of the child flow to call. */
  flowId: string;
  /** A pinned child flow version, if any. */
  flowVersion: string | null;
  /** The baked child flow (stamped at load); a `null` child rejects at run time. */
  child: unknown;
  /** The child's canonical parameters, read by compile validation. */
  childInputs: readonly Parameter[];

  /**
   * Build a CALL driver.
   *
   * @param nodeId The node's unique id.
   * @param options.flowId The id of the child flow to call.
   * @param options.flowVersion A pinned child flow version, if any.
   * @param options.child The baked child flow; `null` until Compose bakes it.
   * @param options.childInputs The child's canonical parameters; omitted, none.
   * @param options.title A display title.
   */
  constructor(
    nodeId: string,
    options: {
      flowId: string;
      flowVersion?: string | null;
      child?: unknown;
      childInputs?: readonly Parameter[] | null;
      title?: string | null;
    },
  ) {
    const { flowId, flowVersion = null, child = null, childInputs = null, title = null } = options;
    super(nodeId, { title });
    this.flowId = flowId;
    this.flowVersion = flowVersion;
    this.child = child;
    this.childInputs = childInputs ?? [];
  }

  /** Yield the common origins, then the cached child signature's. */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    yield* super.iterLocalOrigins(seen);
    for (const parameter of this.childInputs) yield* iterOriginCarrier(parameter, seen);
  }

  /** Rewrite the common and the cached signature's origins on a copied CALL. */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.childInputs = this.childInputs.map((parameter) => rewriteOriginCarrier(parameter, mapper, seen));
  }

  /** Return the baked child flow, or no child for an unbaked CALL. */
  override originChildren(): readonly unknown[] {
    return this.child === null ? [] : [this.child];
  }

  /**
   * Require the child flow that Compose must bake before execution.
   *
   * @throws Error when the child flow isn't baked.
   */
  override validateExecutionSetup(): void {
    super.validateExecutionSetup();
    if (this.child === null) {
      throw new Error(`CALL node ${reprText(this.id)}: child flow ${reprText(this.flowId)} not baked`);
    }
  }

  /**
   * Describe the call's growth: one `Grow` seeded with a copy of the call-arg record.
   *
   * @param inputs The bound call-args.
   * @returns A `Grow` seeded with a copy of the undefaulted call-arg record, with nothing to prune.
   */
  override forward(inputs: Record<string, unknown>): Grow {
    // The call-args go raw: the spliced child START fills omitted inputs from its params'
    // defaults, so the driver doesn't pre-default. The seed is the builder's only durable input;
    // the compile-owned builder constructs the child and lowers its post-asserts onto the child
    // END, for live execution and replay alike.
    return new Grow({ ...inputs });
  }
}
