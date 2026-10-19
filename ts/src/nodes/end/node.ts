/**
 * END: the flow's return boundary, internal-only.
 *
 * The loader synthesizes ONE `EndNode` per flow as its single terminal; the engine's run result
 * is END's committed value.
 *
 * Three modes:
 * - VALUE (a flow's `output: ${x}`, or no `output:`): at most one param; `run` returns
 *   `inputs[name]`, or `null` without a name.
 * - RECORD (a flow's `output: {...}`): one param per declared output; `run` returns
 *   `{name: inputs[name]}` for every name, so one field gives a one-key record and none gives `{}`.
 * - LIST (a MAP fan-in): params `e0`..`e{n-1}`, each fed by a MAP element's child END; `run`
 *   returns `[inputs.e0, ...]` in `over` order (index); `n = 0` gives `[]`.
 *
 * END carries the flow's POST asserts (`${X.output}`); the ordinary run template fires them after
 * END builds its output. A required terminal output co-skipped is a normal required-data-group
 * co-skip derived by the runtime reducer's edge dispositions; END has no special co-skip logic.
 * The loader gives an output whose producers a CASE may skip the explicit null default, so a
 * loaded flow never co-skips END through an output. Hidden from authors: a reserved
 * `__`-prefixed id, never parseable.
 *
 * Knows about:   nodes/base, nodes/contracts and expr (types).
 * Never imports: anything above nodes; the loader synthesizes it and the engine reads it as the
 *   terminal.
 */

import type { FeedbackValue } from "../../expr/index.ts";
import {
  type BackwardContext,
  type BackwardResult,
  Node,
  NodeKind,
  Output,
  projectStructuralFeedback,
} from "../base.ts";
import { Parameter } from "../contracts.ts";

/**
 * The flow's return boundary: internal-only, loader-synthesized.
 *
 * One per flow (the single terminal); the run result is this node's committed value. VALUE mode
 * returns its one input, or `null` without one; RECORD mode returns a record keyed by every
 * declared name, even for one name or none; LIST mode joins the element outputs into a list in
 * `over` order. Use the `value` / `record` / `list` factories rather than the constructor.
 */
export class EndNode extends Node {
  /**
   * Reserved id of the synthesized return boundary, the canonical source of truth. One per flow
   * (deep-namespaced in a child); the run result is the value stored under this id. Authors can
   * never name a node this (the parser bans reserved ids).
   */
  static readonly ID = "__end__";

  readonly kind = NodeKind.END;

  /** The terminal's mode: one value, a record of named outputs, or a MAP fan-in list. */
  mode: "value" | "record" | "list";
  /** The output names in value or record mode; empty in list mode. */
  names: readonly string[] = [];
  /** The element count in list mode; `null` otherwise. */
  n: number | null = null;

  /**
   * Build an END node from exactly one of `outputNames` (value or record mode) or `n` (list mode).
   *
   * @param nodeId The node's id (`ID` at top level; deep-namespaced in a child).
   * @param options.outputNames Record or value mode: one param per declared output.
   * @param options.n List mode: the element count.
   * @param options.title A display title.
   * @param options.value With `outputNames`, selects value mode instead of record mode.
   * @throws RangeError when not exactly one of `outputNames` / `n` is given, or when `value` is
   *   set with more than one output name or without `outputNames`.
   */
  constructor(
    nodeId: string,
    options: { outputNames?: readonly string[] | null; n?: number | null; title?: string | null; value?: boolean } = {},
  ) {
    const { outputNames = null, n = null, title = null, value = false } = options;
    super(nodeId, { title });
    if ((outputNames === null) === (n === null)) {
      throw new RangeError("EndNode takes exactly one of outputNames (value/record) or n (list)");
    }
    if (value && (outputNames === null || outputNames.length > 1)) {
      throw new RangeError("EndNode value mode takes at most one output name");
    }
    if (outputNames !== null) {
      this.mode = value ? "value" : "record";
      this.names = [...outputNames];
      this.params = this.names.map((name) => new Parameter(name));
    } else {
      this.mode = "list";
      this.n = n;
      this.params = Array.from({ length: n! }, (_, index) => new Parameter(`e${index}`));
    }
  }

  /**
   * Build a value-mode terminal that returns its one input as the flow's value.
   *
   * @param nodeId The node's id.
   * @param options.outputName The one param's name, or `null` for a terminal with no params that
   *   returns `null`.
   * @param options.title A display title.
   */
  static value(nodeId: string, options: { outputName: string | null; title?: string | null }): EndNode {
    const { outputName, title = null } = options;
    return new EndNode(nodeId, { outputNames: outputName === null ? [] : [outputName], title, value: true });
  }

  /** Build a record-mode terminal: one param per declared flow `output:` name. */
  static record(nodeId: string, options: { outputNames: readonly string[]; title?: string | null }): EndNode {
    const { outputNames, title = null } = options;
    return new EndNode(nodeId, { outputNames, title });
  }

  /** Build a list-mode terminal: `n` params (`e0`..`e{n-1}`) joined as a MAP fan-in. */
  static list(nodeId: string, options: { n: number; title?: string | null }): EndNode {
    const { n, title = null } = options;
    return new EndNode(nodeId, { n, title });
  }

  /** Whether this terminal is in record mode (a flow's `output: {...}`). */
  get returnsRecord(): boolean {
    return this.mode === "record";
  }

  /** Return the flow's value, record of named outputs, or list of element outputs. */
  override forward(inputs: Record<string, unknown>): Output {
    if (this.mode === "value") {
      const [name] = this.names;
      return new Output(name === undefined ? null : inputs[name]);
    }
    if (this.mode === "record") return new Output(Object.fromEntries(this.names.map((name) => [name, inputs[name]])));
    // List mode (the MAP fan-in): join in over order (index); n = 0 gives [].
    return new Output(this.backwardInputNames().map((name) => inputs[name]));
  }

  /**
   * Return the actual output contributors, excluding augmented assertion params: the value or
   * record output names, or the ordered `e0`..`eN` list slots. Empty layouts have none.
   */
  override backwardInputNames(): readonly string[] {
    if (this.mode !== "list") return [...this.names];
    return Array.from({ length: this.n! }, (_, index) => `e${index}`);
  }

  /**
   * Capture the output layout before the body runs, as `input_output_paths`: no paths for zero
   * outputs, `[]` for a value terminal's one input, a named path for each record field, or an
   * index path for each list position. Assertion inputs are excluded.
   */
  override captureBackwardContext(): Record<string, unknown> {
    const names = this.backwardInputNames();
    let paths: Record<string, unknown[]>;
    if (this.mode === "list") paths = Object.fromEntries(names.map((name, index) => [name, [BigInt(index)]]));
    else if (this.mode === "record") paths = Object.fromEntries(names.map((name) => [name, [name]]));
    else paths = Object.fromEntries(names.map((name) => [name, []]));
    return { input_output_paths: paths };
  }

  /**
   * Project feedback through the recorded layout without reading the live node fields.
   *
   * @param context Recorded inputs, output or failure, the output-name inventory and the captured
   *   output paths.
   * @param feedback Ordered text or structural selections over the accepted output. Text is sent
   *   unchanged to every contributor; a failed END accepts text or `null`.
   * @returns Independent input message batches. Zero-output and empty-list layouts have no
   *   contributors; missing bindings stop.
   * @throws TypeError when a message batch or captured path has an invalid type.
   * @throws RangeError when feedback or a captured path doesn't match the recorded output, or the
   *   structural layout is missing.
   */
  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    return projectStructuralFeedback(context, feedback);
  }
}
