/**
 * START: the flow's parameter-binding boundary, internal-only and pure.
 *
 * The loader synthesizes one `StartNode` per flow as its single root. The generic evaluator path
 * (bind params, then `Node.run`) owns validation, required inputs and defaults; START's body is
 * the identity over that bound record. Only the top-level START sees converted text, because
 * `runFlow` converts supplied inputs before seeding it.
 *
 * The flow's PRE `asserts:` are lowered onto START's ordinary `postAsserts` (each `${input.X}`
 * rewritten to `${output.X}`) and fire through the generic run template against START's bound
 * output record: START is an ordinary scheduled node, not a bypass, so top-level and child-flow
 * boundary asserts share one seam. Not START's job: system, clock or run id. Hidden from authors:
 * a reserved `__`-prefixed id, never parseable.
 *
 * Knows about:   nodes/base, nodes/contracts, expr (types) and typesys.
 * Never imports: anything above nodes; the loader synthesizes and seeds it.
 */

import type { FeedbackValue } from "../../expr/index.ts";
import { isList, isRecord, jsTypeName } from "../../typesys/values.ts";
import { type BackwardContext, BackwardResult, Node, NodeKind, Output, projectStructuralFeedback } from "../base.ts";
import { isExactly, Parameter } from "../contracts.ts";

/**
 * The flow's parameter-binding boundary: internal-only, loader-synthesized, pure.
 *
 * One per flow (the single root). Its params are the input names; the engine reads each source
 * from the flow's wiring for this node (the run-argument seed at top level, the call-argument
 * edges in a child). `run` returns the already-bound record as one object keyed by input name.
 */
export class StartNode extends Node {
  /**
   * Reserved id of the synthesized input boundary, the canonical source of truth. At top level
   * the loader mints one `StartNode` with this id; a child flow's is deep-namespaced
   * (`<callsite>/__start__`). Authors can never name a node this (the parser bans reserved ids).
   */
  static readonly ID = "__start__";

  readonly kind = NodeKind.START;
  /** Commits its bound input record as the node's output. */
  override readonly commitsInputRecord: boolean = true;

  /**
   * Build a START node.
   *
   * @param nodeId The node's id (`ID` at top level; deep-namespaced in a child).
   * @param options.params The flow's canonical semantic input signature; `null` is none.
   * @param options.title A display title.
   * @throws TypeError when `params` isn't an exact array of exact `Parameter`s.
   */
  constructor(
    nodeId: string,
    options: { params?: readonly Parameter[] | null; title?: string | null } = {},
  ) {
    const { params = null, title = null } = options;
    super(nodeId, { title });
    const parameters = params ?? [];
    if (!isList(parameters)) {
      throw new TypeError(`START params must be an exact list, got ${jsTypeName(parameters)}`);
    }
    for (const parameter of parameters) {
      if (!isExactly(parameter, Parameter)) {
        throw new TypeError(`START params must contain Parameter values, got ${jsTypeName(parameter)}`);
      }
    }
    this.params = parameters;
  }

  /** Return the record already bound by the generic node input seam. */
  override forward(inputs: Record<string, unknown>): Output {
    return new Output(inputs);
  }

  /**
   * Capture the named record projection independently of delivered presence: each declared input
   * maps to its same-named output field in `input_output_paths`, so an omitted optional input
   * keeps its path.
   */
  override captureBackwardContext(): Record<string, unknown> {
    return { input_output_paths: Object.fromEntries(this.backwardInputNames().map((name) => [name, [name]])) };
  }

  /**
   * Project feedback to the declared inputs present in the captured START record.
   *
   * @param context Recorded bound inputs, output or failure, the declared inventory and the
   *   captured named output paths.
   * @param feedback Independent whole-record text or structured selections; a failed START
   *   accepts text or `null`.
   * @returns Messages for delivered inputs, defaults included; omitted inputs and unavailable
   *   bindings get none, and a successful output that isn't a record gives an empty result.
   * @throws TypeError when a message batch or captured path has an invalid type.
   * @throws RangeError when feedback or a captured path doesn't match the recorded output, or the
   *   structural layout is missing.
   */
  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    if (context.failure === null && !isRecord(context.output)) return new BackwardResult();
    return projectStructuralFeedback(context, feedback);
  }
}
