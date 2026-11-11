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
 * driver doesn't pre-default. `child` and `childInputs` are baked at load by Compose
 * (`ChildFlowNode`); `childInputs` is read by compile validation. The child's asserts ride its own
 * compiled boundary nodes and survive the deep copy of the spliced subgraph.
 *
 * Knows about: nodes/base and nodes/child_flow.
 */

import { Grow, NodeKind } from "../base.ts";
import { ChildFlowNode } from "../child_flow.ts";

/**
 * The call driver (`kind: call`): apply a callable flow once (`'a flow -> 'b`).
 *
 * The single-call counterpart to `MapNode`. `run` resolves to one seed-first `Grow`; the
 * injected compile builder constructs the child fragment and the engine splices it. The spliced
 * child START owns omitted-input defaulting, so the driver passes call-args raw.
 */
export class CallNode extends ChildFlowNode {
  readonly kind = NodeKind.CALL;
  /** Grows the graph: `run` resolves to a seed-first `Grow`. */
  override readonly isSpawner: boolean = true;
  /** Each child is one call level deeper. */
  override readonly growDepthDelta: number | null = 1;
  /** Its seed only reads earlier pool values through its binding. */
  override readonly derived: boolean = true;

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
