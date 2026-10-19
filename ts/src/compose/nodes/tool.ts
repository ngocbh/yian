/**
 * kind=tool: the deterministic-tool node's descriptor.
 *
 * Owns the `tool` node's `ToolDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`), checks its own fields, and builds
 * the runtime `ToolNode`. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, compose source and helpers, nodes (tool) and defaults.
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { current } from "../../defaults.ts";
import { ToolNode } from "../../nodes/tool/node.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { retriesError } from "./agent.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  rejectNonText,
  siteKey,
} from "./base.ts";

/** The fields `ToolDescriptor`'s constructor takes: the common ones and the TOOL's own. */
export interface ToolDescriptorFields extends NodeDescriptorFields {
  toolId: string;
  retries?: number;
  asserts?: readonly string[] | null;
}

/**
 * kind=tool: a direct, deterministic call to one tool in the run's `ToolSet`.
 *
 * `allowed` adds `tool_id`, `inputs`/`outputs`, `retries` and `asserts` to the commons; `tool_id`
 * is required. A tool's authored `input:` is its sink-binding surface, so `inputs` is its one
 * binding field. `output:` is optional; when present it is the node's output contract, which the
 * engine checks the tool's return value against on every call.
 */
export class ToolDescriptor extends NodeDescriptor {
  static override readonly kind = "tool";
  static override readonly allowed = this.allow("tool_id", "inputs", "outputs", "retries", "asserts");
  static override readonly required = this.require("tool_id");
  static override readonly bindingFields: readonly string[] = ["inputs"];

  /** The tool id, looked up in the run's `ToolSet`. */
  readonly toolId: string;
  /** The effective retry count, so the manifest holds it; zero is left out of the manifest. */
  readonly retries: number;
  /** The node asserts. */
  readonly asserts: readonly string[];

  /**
   * Create a descriptor from `fields`. `retries` defaults to `current().toolRetries`; `asserts`
   * reads `null` as empty.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: ToolDescriptorFields) {
    super(fields);
    this.toolId = fields.toolId;
    this.retries = fields.retries ?? current().toolRetries;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Reject a non-text `tool_id:` (`null` included) and a bad `retries:`, store `retries` as a
   * `number`, then run the base checks.
   *
   * @throws LoadError at the field for a bad shape.
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const rangeOf = (key: string) => ranges.get(siteKey("field", key)) ?? ownRange;
    rejectNonText(this.kind, nodeId, "tool_id", fields["toolId"] ?? null, rangeOf("tool_id"), diagnostics, {
      required: true,
    });
    if (Object.hasOwn(fields, "retries")) {
      const problem = retriesError(fields["retries"]);
      if (problem !== null) {
        throw diagnostics.error(`node ${reprText(nodeId)} (kind=${this.kind}): \`retries:\` ${problem}`, {
          range: rangeOf("retries"),
        });
      }
      fields["retries"] = Number(fields["retries"]);
    }
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Build the runtime `ToolNode` and its flow-owned wiring.
   *
   * Sets the node's `params` from the authored `input:` and its output contract from the
   * optional `output:` (`null` when absent); `_finish` stamps the descriptor origin.
   *
   * @throws LoadError when `output:` isn't a valid type expression.
   */
  override build(ctx: BuildContext): BuildResult {
    const node = new ToolNode(this.id, { toolId: this.toolId, title: this.nodeName, retries: this.retries });
    node.outputContract = this.outputContract(ctx);
    node.params = this.sinkParams();
    return this._finish(node, this.sinkWiring());
  }
}
