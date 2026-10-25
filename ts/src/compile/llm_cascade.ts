/**
 * Resolve and replay effective LLM configuration over the static call tree.
 *
 * Resolution is a compile-layer operation: callers provide both the outermost authored layer and
 * the host's defaults, which may be empty; there is no built-in provider or model. An agent that no
 * layer gives a provider and model fails, naming its static path. The resulting map is therefore
 * complete and can be recorded and replayed without consulting process settings again.
 *
 * Static paths join local author node ids with `/`. The Compose parser reserves that separator in
 * node ids, so paths such as `review/summarize` are unambiguous. A child flow is copied at every
 * CALL/MAP/LOOP callsite before either resolution or replay. A root-level copy alone preserves
 * aliases between shared descendant defs and would let one parent's cascade overwrite another's.
 *
 * Knows about: diagnostics, typesys, llm_clients, nodes (base, binding, contracts, the AGENT node
 * and its `llm_config` values) and the compiled model (`copyGraph`). Defaults are injected by the
 * host.
 * Never imports: compose, runtime, the package-root run boundary, or the CLI.
 */

import { compareCodeUnits, type OriginRef, reprText } from "../diagnostics.ts";
import { HostConnection, mergeLlmConfig } from "../llm_clients/config.ts";
import {
  type AgentConfig,
  type ConnectionSelection,
  type ConnectionSnapshot,
  HOST_KEY_MARKER,
  recordSelection,
  restoreSelection,
  type SelectionRecord,
} from "../llm_clients/connection.ts";
import { requireConcreteSelection, resolveAgentConfig, validateLayer } from "../llm_clients/layers.ts";
import {
  agentLabel,
  CallTimeLLM,
  checkResolvedLayer,
  contractNullable,
  contributing,
  type FieldWhere,
  fieldWhere,
  keepValues,
  LLM_CONFIG_KEY,
  layerRefs,
  type Nullable,
  readsOnlyInputs,
  resolveLayer,
  resolveLayers,
  typesNullable,
  valueLeaves,
} from "../nodes/agent/llm_values.ts";
import { AgentNode } from "../nodes/agent/node.ts";
import type { Node } from "../nodes/base.ts";
import { BindingError, bindParams, copyValue } from "../nodes/binding.ts";
import { Binding, type Parameter, type ValuePath, ValueTemplate } from "../nodes/contracts.ts";
import { isRecord, type Type, TypeCheckError, type Value, type ValueRecord, VariablePool } from "../typesys/index.ts";
import { setField, writeAsText } from "../typesys/values.ts";
import { copyGraph, type Flow } from "./model.ts";

/**
 * A flow as the cascade reads it: its nodes, wiring and START id and, for a composed flow, its own
 * layer and its types.
 */
interface CascadeFlow {
  readonly nodes: ReadonlyMap<string, Node>;
  readonly wiring: ReadonlyMap<string, ReadonlyMap<string, Binding>>;
  readonly startId: string;
  readonly flowLlmConfig?: unknown;
  readonly flowLlmValueTypes?: readonly (readonly [ValuePath, Type | null])[];
}

/** One agent's configuration record. */
type Config = Record<string, unknown>;

/**
 * Bake and return every AGENT node's complete effective configuration.
 *
 * This is `resolveRunLlm` for a run given no inputs; its `effective` map is returned.
 *
 * @param flow The flow whose complete static call tree is resolved in place.
 * @param parentConfig The outermost gap-fill layer (the CLI/host layer).
 * @param options.defaults Lowest-precedence host defaults, possibly empty.
 * @param options.providerModels Saved models by provider.
 * @param options.strayModel Whether `YIAN_DEFAULT_MODEL` was set without a default provider;
 *   defaults to `false`.
 * @returns Complete effective configurations keyed by unambiguous static agent path.
 * @throws {RangeError} As `resolveRunLlm` throws it.
 */
export function resolveLlmCascade(
  flow: CascadeFlow,
  parentConfig: Readonly<Config>,
  options: {
    defaults: Readonly<Config>;
    providerModels?: Readonly<Record<string, string>> | null;
    strayModel?: boolean;
  },
): Record<string, Config> {
  return resolveRunLlm(flow, parentConfig, options).effective;
}

/** The configurations a run start resolves for its agents. Frozen. */
export class RunLLM {
  /** Each agent resolved at run start, its complete effective configuration keyed by static path. */
  readonly effective: Record<string, Config>;
  /**
   * Each agent whose configuration is resolved only when it runs, keyed by static agent path: its
   * base, the layers below its node and own-flow layers (`CallTimeLLM.base`).
   */
  readonly callTime: Record<string, Config>;

  /** Hold the two maps. */
  constructor(effective: Record<string, Config>, callTime: Record<string, Config>) {
    this.effective = effective;
    this.callTime = callTime;
    Object.freeze(this);
  }
}

/**
 * Bake every AGENT node's complete effective configuration at run start.
 *
 * The normal precedence is node, current flow, enclosing flows, caller layer, then `defaults`. An
 * AGENT node with `llmInherit` false skips the flow/enclosing/caller layers but still receives the
 * defaults. An unset model comes from `providerModels` for the agent's provider.
 *
 * Values (`${...}`) in the run's flow that read only its inputs are resolved here, in that flow's
 * layer and in the layers of its agents. Each input such a value reads is checked first, as START
 * checks it: present unless optional, its default when omitted, and its contract. A value
 * resolving to null leaves its field unset. A flow layer passes down only its literal and resolved
 * fields; Compose has refused any other value an inheriting agent could read
 * (`findInheritedLlmValue`). Any other agent whose layers carry values resolves its configuration
 * only when it runs; run start fixes its base: the enclosing flows' layers and the caller layer,
 * merged, or nothing under `inherit: false`. Its node and own-flow layers ride its reserved
 * `__llm_config` binding. Every returned value and every node-owned value is an independent deep
 * copy.
 *
 * @param flow The flow whose complete static call tree is resolved in place.
 * @param parentConfig The outermost gap-fill layer. At the root this is the CLI/host layer; in a
 *   child it is the already-accumulated enclosing flow layer.
 * @param options.defaults Lowest-precedence host defaults, possibly empty. `model` applies only to
 *   the default `provider`. No environment is read.
 * @param options.providerModels Saved models by provider. An authored model still wins per field;
 *   a provider with no saved model needs an authored one.
 * @param options.strayModel Whether `YIAN_DEFAULT_MODEL` was set without a default provider; an
 *   agent missing its provider or model then says so. Defaults to `false`.
 * @param options.runInputs The run's supplied inputs, coerced at the run boundary; `null` (the
 *   default) is none.
 * @returns The resolved configurations.
 * @throws {RangeError} If a provider or model selection is invalid, an agent has no provider or
 *   model, an input a value reads is missing or breaks its contract, a value fails to resolve, or a
 *   child flow's agent would inherit an `llm_config` value read only when its flow runs
 *   (`findInheritedLlmValue`).
 */
export function resolveRunLlm(
  flow: CascadeFlow,
  parentConfig: Readonly<Config>,
  options: {
    defaults: Readonly<Config>;
    providerModels?: Readonly<Record<string, string>> | null;
    strayModel?: boolean;
    runInputs?: Readonly<Record<string, unknown>> | null;
  },
): RunLLM {
  const { defaults, providerModels = null, strayModel = false, runInputs = null } = options;
  const inherited = findInheritedLlmValue(flow);
  if (inherited !== null) {
    const where =
      inherited.owner.length === 0
        ? "the run's flow"
        : `the flow node ${reprText(pathKey(inherited.owner))} calls`;
    const agent = inherited.gate || `node ${reprText(pathKey(inherited.agent))}`;
    throw new RangeError(
      `${agent} would inherit ${fieldLabel(inherited.field)} from ${where}, a value read only when that flow ` +
        `runs; ${inheritedValueFix(inherited.field)}`,
    );
  }
  validateLayer(parentConfig);
  validateLayer(defaults);
  const effective: Record<string, Config> = {};
  const callTime: Record<string, Config> = {};
  resolveFlow(flow, copied(parentConfig), {
    defaults: copied(defaults),
    providerModels: { ...providerModels },
    strayModel,
    path: [],
    effective,
    callTime,
    inputs: new RunInputs(flow, runInputs ?? {}),
  });
  return new RunLLM(effective, callTime);
}

/** The run's inputs that run-start values read, each checked once as START checks it. */
class RunInputs {
  private readonly startId: string;
  private readonly params: ReadonlyMap<string, Parameter>;
  private readonly supplied: ReadonlyMap<string, unknown>;
  private readonly checked = new Map<string, unknown>();

  /** Hold `flow`'s declared inputs and the supplied values, none checked yet. */
  constructor(flow: CascadeFlow, supplied: Readonly<Record<string, unknown>>) {
    this.startId = flow.startId;
    const params = flow.nodes.get(flow.startId)?.params ?? [];
    this.params = new Map(params.map((parameter) => [parameter.name, parameter]));
    this.supplied = new Map(Object.entries(supplied));
  }

  /**
   * Check each input the values in `layer` read; return a pool holding them.
   *
   * @param layer A layer whose values read only `input.*`.
   * @param where Names a value's field in a message.
   * @returns A pool whose START record holds every input checked so far; an omitted optional input
   *   without a default is null.
   * @throws {RangeError} An input is missing or breaks its contract; the message names the field
   *   and the input.
   */
  poolFor(layer: unknown, where: FieldWhere): VariablePool {
    for (const [path, leaf] of valueLeaves(layer)) {
      for (const name of this.readNames(leaf)) {
        if (!this.checked.has(name)) this.checked.set(name, this.check(name, where(path)));
      }
    }
    const record: ValueRecord = {};
    for (const [name, value] of this.checked) setField(record, name, copyValue(value as Value));
    const pool = new VariablePool(this.startId);
    pool.set(this.startId, record);
    return pool;
  }

  /** Return a pool with no inputs, for a layer whose values are all dropped. */
  emptyPool(): VariablePool {
    return new VariablePool(this.startId);
  }

  /** Return the names of the inputs a value reads; `${input}` reads them all. */
  private readNames(leaf: unknown): string[] {
    const names: string[] = [];
    for (const ref of layerRefs(leaf)) {
      const parts = ref.split(".");
      names.push(...(parts.length === 1 ? this.params.keys() : parts.slice(1, 2)));
    }
    return [...new Set(names)];
  }

  /** Bind one input as START binds it, and return its value. */
  private check(name: string, where: string): unknown {
    const parameter = this.params.get(name);
    if (parameter === undefined) throw new RangeError(`${where}: input ${reprText(name)} is not declared`);
    const present = this.supplied.has(name);
    if (!present && parameter.required) {
      throw new RangeError(`${where}: input ${reprText(name)} is required and was not given`);
    }
    const wiring = new Map<string, unknown>(present ? [[name, `\${input.${name}}`]] : []);
    let record: ValueRecord;
    try {
      // The pool checks values as they go in, where Python's binding checks them.
      const pool = new VariablePool(this.startId);
      pool.set(this.startId, present ? { [name]: this.supplied.get(name) } : {});
      record = bindParams([parameter], wiring, pool);
    } catch (error) {
      if (error instanceof BindingError) throw new RangeError(`${where}: ${error.message}`);
      if (error instanceof TypeCheckError) throw new RangeError(`${where}: input ${reprText(name)}: ${error.message}`);
      throw error;
    }
    return Object.hasOwn(record, name) ? record[name] : null;
  }
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
    const child = childOf(node);
    if (child !== null) {
      const found = findInherited(child, nodePath, node, [...enclosing, own]);
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

/**
 * Apply one recorded complete effective map, and call-time agents, to a static flow tree.
 *
 * The effective map and the call-time `agents` together must name exactly the flow's static agent
 * paths, each once. Validation finishes before mutation. Applying deep-copies each config onto its
 * agent's `llmConfig`, gives each call-time agent a `CallTimeLLM` with its base and the restored
 * selection (its `llmConfig` stays `null` until it runs), and de-aliases every static child
 * callsite, without importing settings or reading the environment.
 *
 * @param flow A clean compiled flow to mutate in place.
 * @param effective Recorded complete configurations keyed by static agent path.
 * @param callTime The recorded `RunStarted.call_time_llm`: `agents`, the base of each agent whose
 *   config resolves only when it runs, keyed by static agent path, and `selection`, the
 *   `recordSelection` record they bind from. `null` (the default) when there are none.
 * @throws {RangeError} If paths are missing, extra or in both maps, a config lacks provider/model,
 *   a call-time path names an agent with no `llm_config` value, a base is not a valid layer, or the
 *   selection is invalid.
 * @throws {TypeError} If a map or one of its entries is not a record.
 */
export function applyEffectiveLlmMap(flow: CascadeFlow, effective: unknown, callTime: unknown = null): void {
  if (!isRecord(effective)) throw new TypeError("effective LLM map must be a dict keyed by static agent path");
  let selection: ConnectionSelection | null = null;
  let agents: Record<string, unknown> = {};
  if (callTime !== null && callTime !== undefined) {
    if (!isRecord(callTime)) throw new TypeError("call-time LLM record must be a dict of selection and agents");
    const keys = Object.keys(callTime);
    if (keys.length !== 2 || !keys.includes("selection") || !keys.includes("agents")) {
      throw new RangeError("call-time LLM record must hold exactly selection and agents");
    }
    selection = restoreSelection(callTime["selection"]);
    if (!isRecord(callTime["agents"])) {
      throw new TypeError("call-time LLM agents must be a dict keyed by static agent path");
    }
    agents = callTime["agents"];
  }

  const statics = staticAgents(flow);
  const expected = new Set(statics.map(([path]) => path));
  const supplied = new Set([...Object.keys(effective), ...Object.keys(agents)]);
  const missing = [...expected].filter((path) => !supplied.has(path)).sort(compareCodeUnits);
  const extra = [...supplied].filter((path) => !expected.has(path)).sort(compareCodeUnits);
  const both = Object.keys(effective)
    .filter((path) => Object.hasOwn(agents, path))
    .sort(compareCodeUnits);
  if (missing.length > 0 || extra.length > 0 || both.length > 0) {
    const details: string[] = [];
    if (missing.length > 0) details.push(`missing paths: ${writeAsText(missing)}`);
    if (extra.length > 0) details.push(`extra paths: ${writeAsText(extra)}`);
    if (both.length > 0) details.push(`paths in both maps: ${writeAsText(both)}`);
    throw new RangeError("effective LLM map path mismatch; " + details.join("; "));
  }

  const labels = agentLabels(statics);
  for (const path of Object.keys(effective).sort(compareCodeUnits)) {
    const config = effective[path];
    if (!isRecord(config)) {
      throw new TypeError(`effective LLM config for ${labels.get(path) || reprText(path)} must be a dict`);
    }
    requireConcreteSelection(config, { path, label: labels.get(path) ?? null });
  }
  const carrying = carryingAgentPaths(statics);
  for (const path of Object.keys(agents).sort(compareCodeUnits)) {
    const subject = labels.get(path) || reprText(path);
    if (!carrying.has(path)) {
      throw new RangeError(`call-time LLM base for ${subject}: the agent has no llm_config value`);
    }
    const base = agents[path];
    if (!isRecord(base)) throw new TypeError(`call-time LLM base for ${subject} must be a dict`);
    try {
      validateLayer(base);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      throw new RangeError(`call-time LLM base for ${subject}: ${error.message}`);
    }
  }

  applyFlow(flow, effective as Record<string, Config>, agents as Record<string, Config>, selection, []);
}

/** Resolve one flow scope recursively into `effective` and `callTime`. */
function resolveFlow(
  flow: CascadeFlow,
  parentConfig: Readonly<Config>,
  scope: {
    defaults: Readonly<Config>;
    providerModels: Readonly<Record<string, string>>;
    strayModel: boolean;
    path: readonly string[];
    effective: Record<string, Config>;
    callTime: Record<string, Config>;
    inputs: RunInputs;
  },
): void {
  const { path, inputs } = scope;
  const flowLayer = mergeLlmConfig(runStartFlowLayer(flow, path.length === 0, inputs), parentConfig);

  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId];
    if (node instanceof AgentNode) {
      const key = pathKey(nodePath);
      const label = agentLabel(node.llmConfigLabel, pathKey(path));
      const carried = flow.wiring.get(nodeId)?.get(LLM_CONFIG_KEY) ?? null;
      if (carried !== null && (path.length > 0 || !readsOnlyInputs(carried))) {
        // Its own layers resolve when it runs; only the layers below them are known now.
        setField(scope.callTime as ValueRecord, key, (node.llmInherit ? copied(parentConfig) : {}) as Value);
        node.llmConfig = null;
        continue;
      }
      let authored: Config;
      if (carried === null) {
        const own = literalLayer(node.ownLlmConfig, inputs);
        validateLayer(own);
        authored = mergeLlmConfig(own, node.llmInherit ? flowLayer : {});
      } else {
        authored = resolveCarried(carried, node, parentConfig, { path: key, label, inputs });
      }
      const config = resolveAgentConfig(authored, {
        defaults: scope.defaults,
        providerModels: scope.providerModels,
        strayModel: scope.strayModel,
        path: key,
        label,
      });
      node.llmConfig = copied(config);
      setField(scope.effective as ValueRecord, key, copied(config) as Value);
    }

    const child = childOf(node);
    if (child !== null) {
      // Copy at EACH callsite. A copy of only the root preserves shared descendant aliases and lets
      // differently configured parents interfere.
      const copy = setChild(node, child);
      resolveFlow(copy, flowLayer, { ...scope, path: nodePath });
    }
  }
}

const FLOW_SUBJECT = "flow llm_config";

/**
 * Return a flow's layer as its agents inherit it at run start.
 *
 * The run's flow keeps its values that read only its inputs, resolved; every other value is
 * dropped with its field, since no agent that inherits the layer statically can read it.
 */
function runStartFlowLayer(flow: CascadeFlow, root: boolean, inputs: RunInputs): Config {
  const authored = flow.flowLlmConfig ?? {};
  let layer: Config;
  if (isRecord(authored)) {
    layer = copied(authored);
  } else {
    const where = (field: ValuePath) => fieldWhere(FLOW_SUBJECT, field);
    const known = keepValues(authored, root ? readsOnlyInputs : () => false);
    const pool = root ? inputs.poolFor(known, where) : inputs.emptyPool();
    layer = resolved(() => resolveLayer(known, pool, { where }));
    if (root) checkResolvedLayer(layer, FLOW_SUBJECT, { written: known });
  }
  validateLayer(layer);
  return layer;
}

/** Return an agent's own layer that holds no value as a plain record. */
function literalLayer(layer: unknown, inputs: RunInputs): Config {
  if (isRecord(layer)) return copied(layer);
  return resolveLayer(layer, inputs.emptyPool());
}

/**
 * Resolve an agent's reserved `__llm_config` layers and merge them over the caller's.
 *
 * The agent is in the run's flow, and its values read just the run's inputs.
 */
function resolveCarried(
  carried: Binding,
  node: AgentNode,
  parentConfig: Readonly<Config>,
  options: { path: string; label: string | null; inputs: RunInputs },
): Config {
  const subject = options.label || `node ${reprText(options.path)} llm_config`;
  const nodeWhere = (field: ValuePath) => fieldWhere(subject, field);
  const flowWhere = (field: ValuePath) => fieldWhere(FLOW_SUBJECT, field);
  const where = (field: ValuePath) => (field[0] === "node" ? nodeWhere : flowWhere)(field.slice(1));

  const pool = options.inputs.poolFor(carried.source, where);
  const [nodeLayer, flowLayer] = resolved(() => resolveLayers(carried, pool, { nodeWhere, flowWhere }));
  const written = (carried.source as ValueTemplate).value as Record<string, unknown>;
  checkResolvedLayer(nodeLayer, subject, { written: Object.hasOwn(written, "node") ? written["node"] : null });
  checkResolvedLayer(flowLayer, FLOW_SUBJECT, { written: Object.hasOwn(written, "flow") ? written["flow"] : null });
  if (!node.llmInherit) return nodeLayer;
  return mergeLlmConfig(nodeLayer, mergeLlmConfig(flowLayer, parentConfig));
}

/**
 * Return a call-time agent's provider when run start already knows it.
 *
 * It is known when a literal sets it and no value in a higher layer can override it: the first
 * provider set walking down the node layer, the own-flow layer (unless `inherit: false`), `base`
 * and `defaults`, with no provider or whole-record value above it.
 *
 * @param node The agent.
 * @param flow The flow `node` is in, whose layer is its own-flow layer.
 * @param base The agent's call-time base.
 * @param defaults The host defaults.
 * @returns The provider, or `null` when a value may set it or nothing does.
 */
export function knownProvider(
  node: AgentNode,
  flow: CascadeFlow,
  base: Readonly<Config>,
  defaults: Readonly<Config>,
): string | null {
  for (const layer of ownLayers(node, flow)) {
    if (valueLeaves(layer).some(([path]) => path.length === 0 || path[0] === "provider")) return null;
    const literal = literalFields(layer, flow);
    if (literal["provider"] !== undefined && literal["provider"] !== null) return literal["provider"] as string;
  }
  return (base["provider"] || defaults["provider"] || null) as string | null;
}

/**
 * Return the `base_url` a call-time agent's layers set, when one does.
 *
 * `base_url` is never a value, so the first layer setting it, walking down the node layer, the
 * own-flow layer (unless `inherit: false`) and `base`, gives it.
 *
 * @param node The agent.
 * @param flow The flow `node` is in, whose layer is its own-flow layer.
 * @param base The agent's call-time base.
 * @returns The endpoint, or `null` when no layer sets one.
 */
export function knownBaseUrl(node: AgentNode, flow: CascadeFlow, base: Readonly<Config>): string | null {
  for (const layer of ownLayers(node, flow)) {
    const literal = literalFields(layer, flow);
    if (literal["base_url"] !== undefined && literal["base_url"] !== null) return literal["base_url"] as string;
  }
  return (base["base_url"] ?? null) as string | null;
}

/** Return a call-time agent's node layer and, unless it opts out, its own-flow layer. */
function ownLayers(node: AgentNode, flow: CascadeFlow): unknown[] {
  const layers: unknown[] = [node.ownLlmConfig];
  if (node.llmInherit) layers.push(flow.flowLlmConfig ?? {});
  return layers;
}

/** Return a layer's literal fields, dropping every value in it. */
function literalFields(layer: unknown, flow: CascadeFlow): Readonly<Config> {
  if (isRecord(layer)) return layer;
  return resolveLayer(keepValues(layer, () => false), new VariablePool(flow.startId));
}

/**
 * Return each call-time agent's `knownProvider`, keyed by static agent path.
 *
 * @param flow The run's flow; its static call tree is walked, unchanged.
 * @param callTime Each call-time agent's base, keyed by static agent path.
 * @param defaults The host defaults.
 * @returns The provider, or `null` when not known, per path in `callTime`.
 */
export function knownCallTimeProviders(
  flow: CascadeFlow,
  callTime: Readonly<Record<string, Readonly<Config>>>,
  defaults: Readonly<Config>,
): Map<string, string | null> {
  const known = new Map<string, string | null>();
  for (const [path, node, owner] of staticAgents(flow)) {
    if (Object.hasOwn(callTime, path)) known.set(path, knownProvider(node, owner, callTime[path]!, defaults));
  }
  return known;
}

/**
 * Return the `knownBaseUrl` of each call-time agent that has one.
 *
 * @param flow The run's flow; its static call tree is walked, unchanged.
 * @param callTime Each call-time agent's base, keyed by static agent path.
 * @returns The endpoint, keyed by static agent path, for each agent in `callTime` whose layers set
 *   one.
 */
export function knownCallTimeBaseUrls(
  flow: CascadeFlow,
  callTime: Readonly<Record<string, Readonly<Config>>>,
): Map<string, string> {
  const urls = new Map<string, string>();
  for (const [path, node, owner] of staticAgents(flow)) {
    if (Object.hasOwn(callTime, path)) {
      const url = knownBaseUrl(node, owner, callTime[path]!);
      if (url !== null) urls.set(path, url);
    }
  }
  return urls;
}

/**
 * Return the config each started call-time agent recorded in its steps.
 *
 * An agent that has not started records nothing; its config is checked when it resolves.
 *
 * @param nodes The run's live nodes, keyed by node id.
 * @param steps The steps of each unsettled node, keyed by node id.
 * @returns Each recorded config, with no key or header value, keyed by node id.
 */
export function inFlightLlmConfigs(
  nodes: ReadonlyMap<string, unknown>,
  steps: ReadonlyMap<string, readonly unknown[]>,
): Record<string, Config> {
  const configs: Record<string, Config> = {};
  for (const [nodeId, nodeSteps] of steps) {
    const node = nodes.get(nodeId);
    if (node instanceof AgentNode && node.callTime !== null) {
      const recorded = node.recordedLlmConfig(nodeSteps);
      if (recorded !== null) setField(configs as ValueRecord, nodeId, recorded as Value);
    }
  }
  return configs;
}

/**
 * Record the connection selection the call-time agents may need.
 *
 * When every call-time agent's provider is known (`knownProvider`), only those providers'
 * connections are recorded; otherwise every connection is.
 *
 * @param snapshot The run's selection snapshot.
 * @param options.known Each call-time agent's `knownCallTimeProviders` entry, keyed by static
 *   agent path.
 * @param options.hostKeyProvider The host key's provider, if the host gave one.
 * @param options.hostBaseUrl Where the host key is sent, if the host gave it.
 * @param options.hostSettings The host's connection; only its nonsecret settings are recorded.
 * @returns The `recordSelection` record.
 */
export function callTimeSelection(
  snapshot: ConnectionSnapshot,
  options: {
    known: ReadonlyMap<string, string | null>;
    hostKeyProvider?: string | null;
    hostBaseUrl?: string | null;
    hostSettings?: HostConnection | null;
  },
): SelectionRecord {
  const known = [...options.known.values()];
  const providers = known.includes(null) ? null : new Set(known.filter((provider) => provider) as string[]);
  return recordSelection(snapshot, {
    providers,
    hostKeyProvider: options.hostKeyProvider ?? null,
    hostBaseUrl: options.hostBaseUrl ?? null,
    hostSettings: options.hostSettings ?? null,
  });
}

/**
 * Name the `llm_config` of each run-start agent whose config takes values.
 *
 * A run-start check of such an agent's effective config names the field at fault under this
 * subject. An agent of the run's flow whose layers carry values is named by its own layer. When
 * the run's flow layer holds values, every other agent that inherits it, at any depth, is named
 * with `(flow llm_config)`.
 *
 * @param flow The run's flow.
 * @returns The subject, such as `node 'chat' llm_config`, `Agent 'c/a' (flow llm_config)` or a
 *   synthesized agent's label, keyed by static agent path.
 */
export function valueConfigSubjects(flow: CascadeFlow): Map<string, string> {
  const flowValues = valueLeaves(flow.flowLlmConfig ?? null).length > 0;
  const subjects = new Map<string, string>();
  for (const [path, node, owner] of staticAgents(flow)) {
    const parts = path.split("/");
    const nodeId = parts.at(-1)!;
    if (owner.wiring.get(nodeId)?.has(LLM_CONFIG_KEY)) {
      if (owner === flow) {
        const label = agentLabel(node.llmConfigLabel, "");
        subjects.set(path, label || `node ${reprText(nodeId)} llm_config`);
      }
    } else if (flowValues && node.llmInherit) {
      const agent = agentLabel(node.llmConfigLabel, parts.slice(0, -1).join("/")) || `Agent ${reprText(path)}`;
      subjects.set(path, `${agent} (${FLOW_SUBJECT})`);
    }
  }
  return subjects;
}

/** Run one resolution, reporting a binding failure (it names its field) as a plain `RangeError`. */
function resolved<T>(resolve: () => T): T {
  try {
    return resolve();
  } catch (error) {
    if (!(error instanceof BindingError)) throw error;
    throw new RangeError(error.message);
  }
}

/** Return each static agent's path, node and the flow it is in, without mutating. */
function staticAgents(flow: CascadeFlow, path: readonly string[] = []): [string, AgentNode, CascadeFlow][] {
  const agents: [string, AgentNode, CascadeFlow][] = [];
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId];
    if (node instanceof AgentNode) agents.push([pathKey(nodePath), node, flow]);
    const child = childOf(node);
    if (child !== null) agents.push(...staticAgents(child, nodePath));
  }
  return agents;
}

/** Return the static paths of the `staticAgents` entries whose layers ride `__llm_config`. */
function carryingAgentPaths(statics: readonly [string, AgentNode, CascadeFlow][]): Set<string> {
  const paths = new Set<string>();
  for (const [path, , owner] of statics) {
    if (owner.wiring.get(path.split("/").at(-1)!)?.has(LLM_CONFIG_KEY)) paths.add(path);
  }
  return paths;
}

/**
 * Map each synthesized agent's static path to the authored site of its `llm_config`.
 *
 * Walks the static tree like the effective map does, without mutating it. An authored agent has no
 * label and is absent, so its messages keep naming its path.
 *
 * @param flow The flow whose static agents to label.
 * @returns The label of each labelled agent, keyed by static agent path.
 */
export function staticAgentLabels(flow: CascadeFlow): Map<string, string> {
  return agentLabels(staticAgents(flow));
}

/** Map each labelled `staticAgents` entry's path to its label, at the callsite holding it. */
function agentLabels(statics: readonly [string, AgentNode, CascadeFlow][]): Map<string, string> {
  const labels = new Map<string, string>();
  for (const [path, node] of statics) {
    const label = agentLabel(node.llmConfigLabel, path.split("/").slice(0, -1).join("/"));
    if (label !== null) labels.set(path, label);
  }
  return labels;
}

/** Apply already-validated configs and bases, and de-alias children recursively. */
function applyFlow(
  flow: CascadeFlow,
  effective: Readonly<Record<string, Config>>,
  callTime: Readonly<Record<string, Config>>,
  selection: ConnectionSelection | null,
  path: readonly string[],
): void {
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId];
    if (node instanceof AgentNode) {
      const key = pathKey(nodePath);
      if (Object.hasOwn(callTime, key)) {
        if (selection === null) {
          throw new RangeError(`call-time agent ${reprText(key)} has no recorded connection selection`);
        }
        node.llmConfig = null;
        // The selection is frozen, so the node may share it.
        node.callTime = new CallTimeLLM({ path: key, base: copied(callTime[key]!), selection });
      } else {
        node.llmConfig = copied(effective[key]!);
        node.callTime = null;
      }
    }

    const child = childOf(node);
    if (child !== null) applyFlow(setChild(node, child), effective, callTime, selection, nodePath);
  }
}

/**
 * Give every static AGENT marked for a host key that key and its header values.
 *
 * `ConnectionSnapshot.bind` marks the agents of the key's provider in the effective map. The key
 * and header values go on each marked node's run-local `llmConfig` only, never into the map that
 * `RunStarted` records. A call-time agent whose recorded selection names the key's provider gets
 * them as its run-local `hostCredentials`, never merged into a config; it uses them only if it
 * resolves to that provider. Call it after `applyEffectiveLlmMap`, which has already de-aliased
 * every child callsite.
 *
 * @param flow A flow whose static agents already carry their effective configs.
 * @param apiKey The host-supplied key.
 * @param headers The header values sent with that key; `null` (the default) or empty binds none.
 */
export function bindHostCredentials(
  flow: CascadeFlow,
  apiKey: string,
  headers: Readonly<Record<string, string>> | null = null,
): void {
  const given = headers !== null && Object.keys(headers).length > 0 ? headers : null;
  for (const node of flow.nodes.values()) {
    if (node instanceof AgentNode && node.callTime !== null) {
      if (node.callTime.selection.hostKeyProvider !== null) {
        node.hostCredentials = new HostConnection({ apiKey, headers: given });
      }
    } else if (node instanceof AgentNode && node.llmConfig?.[HOST_KEY_MARKER] === true) {
      // A fresh record: a copied flow's nodes share their configs with the original's.
      const config: AgentConfig = { ...node.llmConfig, api_key: apiKey };
      if (given !== null) config["headers"] = { ...given };
      node.llmConfig = config;
    }
    const child = childOf(node);
    if (child !== null) bindHostCredentials(child, apiKey, headers);
  }
}

/** Return the providers whose agents a recorded effective map marks for a host key. */
export function hostKeyProviders(effective: Readonly<Record<string, Readonly<Config>>>): Set<string> {
  const providers = new Set<string>();
  for (const config of Object.values(effective)) {
    if (config[HOST_KEY_MARKER] === true) providers.add(config["provider"] as string);
  }
  return providers;
}

/** Encode one non-empty static author path. */
function pathKey(path: readonly string[]): string {
  return path.join("/");
}

/** Return a node's static child flow (CALL, MAP, LOOP), or `null`. */
function childOf(node: Node): Flow | null {
  const child = (node as { child?: unknown }).child;
  return child === undefined || child === null ? null : (child as Flow);
}

/** Give `node` a private copy of its static child flow, and return the copy. */
function setChild(node: Node, child: Flow): Flow {
  const copy = copyGraph(child);
  (node as unknown as { child: Flow }).child = copy;
  return copy;
}

/** Return an independent deep copy of a config record. */
function copied<T>(config: T): T & Config {
  return copyValue(config as Value) as T & Config;
}
