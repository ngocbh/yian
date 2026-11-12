/**
 * The base of the nodes that run a child flow: CALL, MAP and LOOP.
 *
 * Each holds the child flow Compose bakes at load and the child's canonical parameters, walks
 * that cached signature's origins, exposes the baked child through the source-neutral
 * `originChildren` hook, and refuses to run unbaked.
 *
 * Knows about: nodes/base, nodes/contracts and diagnostics.
 */

import { type OriginRef, reprText } from "../diagnostics.ts";
import { Node } from "./base.ts";
import { iterOriginCarrier, type OriginMapper, type Parameter, rewriteOriginCarrier } from "./contracts.ts";

/** A node that runs a child flow: its id, pinned version, baked compiled flow and signature. */
export abstract class ChildFlowNode extends Node {
  /** The id of the child flow. */
  flowId: string;
  /** A pinned child flow version, if any. */
  flowVersion: string | null;
  /** The baked child flow (stamped at load); a `null` child rejects at run time. */
  child: unknown;
  /** The child's canonical parameters, read by compile validation; an omitted signature is `[]`. */
  childInputs: readonly Parameter[];

  /**
   * Build the node with its child flow.
   *
   * @param options.child The baked child flow; `null` until Compose bakes it.
   * @param options.childInputs The child's canonical parameters; omitted, none.
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

  /** Rewrite the common and the cached signature's origins on a copied node. */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.childInputs = this.childInputs.map((parameter) => rewriteOriginCarrier(parameter, mapper, seen));
  }

  /** Return the baked child flow, or no child for an unbaked node. */
  override originChildren(): readonly unknown[] {
    return this.child === null ? [] : [this.child];
  }

  /** Hold the baked child given. */
  override setOriginChildren(children: readonly unknown[]): void {
    super.setOriginChildren(children);
    if (children.length > 0) this.child = children[0];
  }

  /**
   * Require the child flow that Compose must bake before execution.
   *
   * @throws Error naming the node by its kind (`CALL node 'x'`) when the child flow isn't baked.
   */
  override validateExecutionSetup(): void {
    super.validateExecutionSetup();
    if (this.child === null) {
      throw new Error(`${this.kind.toUpperCase()} node ${reprText(this.id)}: child flow ${reprText(this.flowId)} not baked`);
    }
  }
}
