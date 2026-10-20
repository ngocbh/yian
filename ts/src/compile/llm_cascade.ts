/**
 * Effective LLM configuration over the static call tree: the load-time check for `llm_config`
 * values a child flow's agent would inherit.
 *
 * Static paths join local author node ids with `/`. The Compose parser reserves that separator in
 * node ids, so paths such as `review/summarize` are unambiguous.
 *
 * Knows about: diagnostics, typesys and nodes (base, contracts, the AGENT node and its
 * `llm_config` values).
 * Never imports: compose, runtime, the package-root run boundary, or the CLI.
 */

import type { OriginRef } from "../diagnostics.ts";
import {
  agentLabel,
  contractNullable,
  contributing,
  type Nullable,
  readsOnlyInputs,
  typesNullable,
  valueLeaves,
} from "../nodes/agent/llm_values.ts";
import { AgentNode } from "../nodes/agent/node.ts";
import type { Node } from "../nodes/base.ts";
import { Binding, type ValuePath, ValueTemplate } from "../nodes/contracts.ts";
import type { Type } from "../typesys/index.ts";

/** A flow as the walk reads it: its nodes and, for a composed flow, its own layer and its types. */
interface CascadeFlow {
  readonly nodes: ReadonlyMap<string, Node>;
  readonly flowLlmConfig?: unknown;
  readonly flowLlmValueTypes?: readonly (readonly [ValuePath, Type | null])[];
}

/**
 * An `llm_config` value in an enclosing flow's layer that a child flow's agent inherits. Frozen.
 *
 * Only the run's flow layer may pass a value down, and only one that reads just the run's inputs,
 * which are known at run start; any other value is read only when its flow runs, after the
 * child's agents have their config.
 */
export class InheritedValue {
  /** The static path of the callsite whose flow holds the layer; empty for the run's flow. */
  readonly owner: readonly string[];
  /** The CALL, MAP or LOOP node at `owner`; `null` for the run's flow. */
  readonly callsite: Node | null;
  /** The value's field path in its layer; empty for a whole record. */
  readonly field: ValuePath;
  /** The value's authored site, when known. */
  readonly origin: OriginRef | null;
  /** The static path of the agent that inherits it. */
  readonly agent: readonly string[];
  /** The agent's `llmConfigLabel`; `null` for an authored agent. */
  readonly label: string | null;

  /** Record one inherited value. */
  constructor(fields: {
    owner: readonly string[];
    callsite: Node | null;
    field: ValuePath;
    origin: OriginRef | null;
    agent: readonly string[];
    label?: string | null;
  }) {
    this.owner = Object.freeze([...fields.owner]);
    this.callsite = fields.callsite;
    this.field = Object.freeze([...fields.field]);
    this.origin = fields.origin;
    this.agent = Object.freeze([...fields.agent]);
    this.label = fields.label ?? null;
    Object.freeze(this);
  }

  /** Name a synthesized agent by its authored site, with its callsite; else `null`. */
  get gate(): string | null {
    return agentLabel(this.label, this.agent.slice(0, -1).join("/"));
  }
}

/**
 * Return the first `llm_config` value a child flow's agent would inherit, if any.
 *
 * The walk follows the cascade: a child's agent inherits every enclosing flow's layer unless it
 * has `inherit: false`. A value reaches it unless a layer closer to the agent (its own, its
 * flow's, a nearer enclosing flow's) always sets that field. A value answers from the static type
 * Compose proved for it, else from its tag.
 *
 * @param flow The run's flow; its static call tree is walked, unchanged.
 * @returns The first such value in static tree order, or `null`.
 */
export function findInheritedLlmValue(flow: CascadeFlow): InheritedValue | null {
  return findInherited(flow, [], null, []);
}

/** Name an `llm_config` field path in a message, such as `llm_config.model`. */
export function fieldLabel(field: ValuePath): string {
  return "llm_config" + field.map((part) => `.${part}`).join("");
}

/** Tell the author how to pass an inherited value to a child flow instead. */
export function inheritedValueFix(field: ValuePath): string {
  const name = field.length > 0 ? field.at(-1) : "llm_config";
  return `pass it to the child as an input and write \`\${input.${name}}\` in the child's llm_config`;
}

/** One enclosing flow's layer on the way down to a child flow. */
interface Enclosing {
  readonly layer: unknown;
  readonly nullable: Nullable;
  readonly owner: readonly string[];
  readonly callsite: Node | null;
}

/** Answer from the static types Compose proved for a flow layer's values, else from tags. */
function flowNullable(flow: CascadeFlow, layer: unknown): Nullable {
  const types = flow.flowLlmValueTypes;
  return types !== undefined && types.length > 0 ? typesNullable(layer, types) : contractNullable;
}

/** Search `flow`, reached at callsite `path` under `enclosing`, and its children. */
function findInherited(
  flow: CascadeFlow,
  path: readonly string[],
  callsite: Node | null,
  enclosing: readonly Enclosing[],
): InheritedValue | null {
  const layer = flow.flowLlmConfig ?? {};
  const own: Enclosing = { layer, nullable: flowNullable(flow, layer), owner: path, callsite };
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId];
    if (node instanceof AgentNode && node.llmInherit && enclosing.length > 0) {
      const found = inheritedBy(node, own, enclosing, nodePath);
      if (found !== null) return found;
    }
    const child = (node as { child?: unknown }).child;
    if (child !== undefined && child !== null) {
      const found = findInherited(child as CascadeFlow, nodePath, node, [...enclosing, own]);
      if (found !== null) return found;
    }
  }
  return null;
}

/** Return the first enclosing value that can reach `node`, innermost layer first. */
function inheritedBy(
  node: AgentNode,
  own: Enclosing,
  enclosing: readonly Enclosing[],
  agent: readonly string[],
): InheritedValue | null {
  const higher: [unknown, Nullable][] = [
    [node.ownLlmConfig, node.llmNullable()],
    [own.layer, own.nullable],
  ];
  for (const outer of [...enclosing].reverse()) {
    let remaining = outer.layer;
    for (const [layer, nullable] of higher) remaining = contributing(remaining, layer, nullable);
    for (const [field, leaf] of valueLeaves(remaining)) {
      // The run's inputs are known at run start.
      if (outer.owner.length === 0 && readsOnlyInputs(leaf)) continue;
      return new InheritedValue({
        owner: outer.owner,
        callsite: outer.callsite,
        field,
        origin: valueOrigin(outer.layer, field, leaf),
        agent,
        label: node.llmConfigLabel,
      });
    }
    higher.push([outer.layer, outer.nullable]);
  }
  return null;
}

/** Return the authored site of the value at `field` in `layer`. */
function valueOrigin(layer: unknown, field: ValuePath, leaf: unknown): OriginRef | null {
  if (layer instanceof Binding) return layer.origin;
  if (leaf instanceof Binding && leaf.origin !== null) return leaf.origin;
  return layer instanceof ValueTemplate ? layer.originAt(field) : null;
}
