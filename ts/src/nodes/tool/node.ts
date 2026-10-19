/**
 * TOOL: a direct, deterministic invocation of one tool from the run's `ToolSet`.
 *
 * No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
 * bindings like any other kind's, resolved by the engine's bind seam, so the tool sees only its
 * own record, never the live pool. An optional `output:` declaration sets the node's output
 * contract, which the engine checks against the tool's return value on every call. `retries:`
 * is how many times a failed call may be tried again.
 *
 * Only the node's shell is here. Its body (the call, its retries with backoff, and each retried
 * attempt recorded as a node step) is not ported yet and comes with `forward`.
 *
 * Knows about:   nodes/base, nodes/contracts, defaults, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { current } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { type Type, valueRepr } from "../../typesys/values.ts";
import { Node, NodeKind } from "../base.ts";
import { HostToolCall, outputContractType } from "../contracts.ts";

/**
 * A direct, deterministic invocation of one tool from the run's `ToolSet` (no LLM in the loop).
 *
 * A failed call is retried up to `retries` times. `inputTypes` maps each input name to the type
 * the loader proved its binding has (a literal's exact type, a typed binding's declared type, or
 * a reference's inferred type); an input whose type is not proven is absent.
 */
export class ToolNode extends Node {
  readonly kind = NodeKind.TOOL;
  /** Looks up and invokes its tool in the run's tool set. */
  override readonly needsTools: boolean = true;
  /** Records each retried attempt as a durable step, so a resumed run continues the count. */
  override readonly needsSteps: boolean = true;
  /** The tool id, looked up in the run's `ToolSet`; an unknown id rejects at run time. */
  toolId: string;
  /** How many times a failed tool call may be retried, at least 0. */
  retries: number;
  /** The proven type of each input's binding; an input whose type is not proven is absent. */
  inputTypes: ReadonlyMap<string, Type>;

  /**
   * Build a TOOL node.
   *
   * @param nodeId The node's unique id.
   * @param options.toolId The tool id, looked up in the run's `ToolSet`.
   * @param options.title A display title.
   * @param options.retries How many times a failed call may be retried, at least 0; `null` or
   *   omitted means `current().toolRetries`.
   * @throws TypeError when `retries` isn't a safe-integer number or `null`.
   * @throws RangeError when `retries` is negative.
   */
  constructor(nodeId: string, options: { toolId: string; title?: string | null; retries?: number | null }) {
    const { toolId, title = null, retries = null } = options;
    super(nodeId, { title });
    if (retries !== null) {
      const message = `node ${reprText(nodeId)}: retries must be a non-negative int, got ${valueRepr(retries)}`;
      if (!Number.isSafeInteger(retries)) throw new TypeError(message);
      if (retries < 0) throw new RangeError(message);
    }
    this.toolId = toolId;
    this.retries = retries ?? current().toolRetries;
    this.inputTypes = new Map();
  }

  /** Return this node's tool id, input names, proven input types, and output type. */
  override hostToolCall(): HostToolCall {
    return new HostToolCall(
      this.toolId,
      (this.params ?? []).map((parameter) => parameter.name),
      this.inputTypes,
      outputContractType(this.outputContract),
    );
  }

  /** The tool call and its retries; not ported yet. */
  override forward(): never {
    throw new Error(`node ${reprText(this.id)} (ToolNode) forward is not ported yet (T4-11)`);
  }
}
