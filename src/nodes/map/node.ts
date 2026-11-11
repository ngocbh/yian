/**
 * MAP: the authorable `List.map` driver (`kind: map` + `over:`).
 *
 * `MapNode` is the mapped-call node, distinct from the single-call `CallNode`. The two are
 * different typed drivers: a `CallNode` applies a callable once (`'a flow -> 'b`); a `MapNode`
 * maps it over a list (`list['a] -> list['b]`). The discriminator is the kind (`NodeKind.MAP`):
 * `MapNode` carries no `over` field and no `${...}` source on the node. The `over` source rides
 * `flow.wiring[id].over`, pre-resolved into `inputs.over` by the engine's evaluator through
 * `bindReserved` before `run`.
 *
 * `run` resolves to one seed-first `Grow` carrying the raw list of per-element records. The
 * compile-owned injected builder constructs the MAP fan-in, and the generic growth reducer
 * splices it. An empty `over` builds a synthetic start wired directly to the list END (which
 * emits `[]`). The per-element call-args go raw: the spliced child START owns omitted-input
 * defaulting. `child` and `childInputs` are baked at load by Compose (`ChildFlowNode`);
 * `childInputs` is read by compile validation. `parallel` is inert (concurrency is the engine's
 * worker count); it is carried for the `over` case.
 *
 * Knows about:   nodes/base, nodes/binding, nodes/child_flow, nodes/contracts, diagnostics and
 * typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { reprText } from "../../diagnostics.ts";
import type { VariablePool } from "../../typesys/pool.ts";
import { isList, type Value, valueRepr } from "../../typesys/values.ts";
import { Grow, type NodeCaps, NodeKind } from "../base.ts";
import { BindingError, rejectInvalidBinding, type ReservedBinder, resolveSource, sourceOrigin } from "../binding.ts";
import { ChildFlowNode } from "../child_flow.ts";
import { bindingSource, type Parameter } from "../contracts.ts";

/**
 * The `List.map` driver (`kind: map` + `over:`): map a callable flow over a list
 * (`list['a] -> list['b]`).
 *
 * The mapped counterpart to `CallNode`. The `over` source is pre-resolved into `inputs.over` by
 * the engine's bind seam; `run` resolves to one seed-first `Grow`. The injected compile builder
 * constructs the MAP fan-in and the engine splices it. The spliced child START owns omitted-input
 * defaulting, so per-element call-args go raw.
 */
export class MapNode extends ChildFlowNode {
  readonly kind = NodeKind.MAP;
  /** Grows the graph: `run` resolves to a seed-first `Grow`. */
  override readonly isSpawner: boolean = true;
  /** Binds call-args per element through the `bindItem` capability, not up front. */
  override readonly bindsPerItem: boolean = true;
  /** Each element is read from the bound `over` list. */
  override readonly itemSourceKey: string | null = "over";
  /** Each child is one call level deeper. */
  override readonly growDepthDelta: number | null = 1;
  /** Its seed only reads earlier pool values through its binding. */
  override readonly derived: boolean = true;
  /** Inert: concurrency is the engine's worker count; carried for the `over` case. */
  parallel: boolean;

  /**
   * Build a MAP driver.
   *
   * @param nodeId The node's unique id.
   * @param options.flowId The id of the child flow to map.
   * @param options.parallel Inert; defaults to `false`.
   * @param options.flowVersion A pinned child flow version, if any.
   * @param options.child The baked child flow; `null` until Compose bakes it.
   * @param options.childInputs The child's canonical parameters; omitted, none.
   * @param options.title A display title.
   */
  constructor(
    nodeId: string,
    options: {
      flowId: string;
      parallel?: boolean;
      flowVersion?: string | null;
      child?: unknown;
      childInputs?: readonly Parameter[] | null;
      title?: string | null;
    },
  ) {
    super(nodeId, options);
    this.parallel = options.parallel ?? false;
  }

  /**
   * Pre-resolve the MAP `over` source into the list `run` maps over.
   *
   * Returns `{over: <list>}`: the `over` source rides `nodeWiring.get("over")`, evaluated against
   * `pool`.
   *
   * @throws Error when the wiring has no `over` source.
   * @throws BindingError classified `contract.input` when the source resolves to `null` or a
   *   non-list; classified `binding.evaluate` when the list nests deeper than
   *   `YIAN_MAX_VALUE_NESTING` levels or holds a value host intake refuses, at any depth.
   */
  override bindReserved(
    nodeWiring: ReadonlyMap<string, unknown>,
    pool: VariablePool,
    options: { bind?: ReservedBinder | null } = {},
  ): Record<string, unknown> {
    const { bind = null } = options;
    if (!nodeWiring.has("over")) throw new Error(`MAP node ${reprText(this.id)} has no \`over\` wiring`);
    const overSrc = nodeWiring.get("over");
    const origin = sourceOrigin(overSrc);
    // A text source is shown as written, any other as its repr.
    const raw = bindingSource(overSrc);
    const label = `MAP node ${reprText(this.id)}: \`over\` (${typeof raw === "string" ? raw : valueRepr(raw)})`;
    const validate = (items: unknown): unknown => {
      if (items === null || !isList(items)) {
        throw new BindingError(`${label} did not resolve to a list`, { failureCode: "contract.input", origin });
      }
      rejectInvalidBinding(items, label, origin);
      return items;
    };
    const items = bind === null ? validate(resolveSource(overSrc, pool)) : bind("over", overSrc, validate);
    return { over: items };
  }

  /** A MAP reserves `over`, its iteration-source author-wiring key. */
  override reservedWiringKeys(): ReadonlySet<string> {
    return new Set(["over"]);
  }

  /**
   * Describe the map's growth: one `Grow` seeded with one call-arg record per `over` element.
   *
   * @param inputs The bound inputs; `inputs.over` is the list `bindReserved` resolved.
   * @param caps The evaluator's capabilities; `caps.bindItem` binds each element's call-args.
   * @returns A `Grow` whose seed lists a copy of each element's record in order, with nothing to
   *   prune; an empty `over` seeds an empty list.
   * @throws TypeError when `caps.bindItem` is missing, also for an empty `over`.
   */
  override forward(inputs: Record<string, unknown>, caps: NodeCaps): Grow {
    const { bindItem } = caps;
    if (bindItem === undefined) {
      throw new TypeError(`MAP node ${reprText(this.id)} was run without the bindItem capability`);
    }
    // Per-element call-args go raw: the spliced child START fills omitted inputs from its params'
    // defaults. The seed is the builder's only durable input; the compile-owned builder constructs
    // the MAP fan-in.
    return new Grow((inputs.over as readonly Value[]).map((element, index) => ({ ...bindItem(element, index) })));
  }
}
