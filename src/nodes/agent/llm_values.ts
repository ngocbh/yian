/**
 * Values (`${...}`) in an AGENT's `llm_config` layers: leaves, contribution, carrying, resolving.
 *
 * A layer is a plain record (all literal), a `ValueTemplate` (fields, block fields or whole
 * blocks from values), or a `Binding` (the whole record from one value). The helpers here find a
 * layer's value leaves and the references they read, decide which lower-layer values a higher
 * layer leaves able to contribute, carry an agent's node and own-flow layers as one reserved
 * binding, resolve that binding against a pool, check the resolved layers, and build a call-time
 * agent's config. Precedence and merge granularity follow `mergeLlmConfig`.
 *
 * Knows about:   diagnostics, expr, llm_clients, nodes/binding, nodes/contracts and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import { type OriginRef, reprText } from "../../diagnostics.ts";
import { ExpressionError, exprRefsOf, scanTemplate } from "../../expr/index.ts";
import { Span, type TemplateSegment } from "../../expr/template.ts";
import {
  type HostConnection,
  hostOnlyFieldError,
  LLMConfig,
  mergeLlmConfig,
  SUPPORTED_LLM_PROVIDERS,
} from "../../llm_clients/config.ts";
import {
  type AgentConfig,
  blockProblem,
  checkModelConfig,
  type ConnectionSelection,
  HOST_KEY_MARKER,
  refusedBlock,
} from "../../llm_clients/connection.ts";
import { requireConcreteSelection, resolveAgentConfig, validateLayer } from "../../llm_clients/layers.ts";
import { locatedBlockProblems, PROVIDER_BLOCKS, ValidationError } from "../../llm_clients/provider_blocks.ts";
import {
  isRecord,
  pythonTypeName,
  type Type,
  unionField,
  type Value,
  ValueKind,
  type ValueRecord,
  type VariablePool,
} from "../../typesys/index.ts";
import { setField } from "../../typesys/values.ts";
import { BindingError, resolveSource } from "../binding.ts";
import { Binding, iterSourceStrings, sameItems, type ValuePath, ValueTemplate } from "../contracts.ts";

/** Says whether a value leaf's value, or the part at a path inside it, may be null or absent. */
export type Nullable = (leaf: unknown, inner: ValuePath) => boolean;

/** Names a field of a layer in an error message, from its path in the layer. */
export type FieldWhere = (field: ValuePath) => string;

/** The reserved AGENT wiring key that carries the node and own-flow layers. */
export const LLM_CONFIG_KEY = "__llm_config";

/** Why a host-only field may not come from a value. */
export const HOST_ONLY_FROM_VALUE = "credentials and where they are sent stay with the host or yian auth";

/**
 * What run start fixes for an agent whose `llm_config` resolves only when it runs. Instances are
 * frozen.
 *
 * `path` is the agent's static path. `base` holds the layers below the agent's node and own-flow
 * layers: the enclosing flows' layers and the host layer, merged; empty under `inherit: false`.
 * It holds no key or header value. `selection` is the run's recorded selection: the defaults and
 * saved models the resolved config falls back to, and the connection it binds.
 */
export class CallTimeLLM {
  readonly path: string;
  readonly base: Readonly<Record<string, unknown>>;
  readonly selection: ConnectionSelection;

  constructor(fields: { path: string; base: Readonly<Record<string, unknown>>; selection: ConnectionSelection }) {
    this.path = fields.path;
    this.base = fields.base;
    this.selection = fields.selection;
    Object.freeze(this);
  }
}

/**
 * Return whether a layer leaf is a value rather than a literal.
 *
 * A value is a template string holding a `${...}` span, or a typed `Binding` whose source holds
 * one. A tagged literal and span-free text are literals.
 */
export function isValue(leaf: unknown): boolean {
  if (typeof leaf === "string") return hasSpan(leaf);
  return leaf instanceof Binding && dynamic(leaf.source);
}

/**
 * Pair each value in a layer with its field path.
 *
 * A whole record is the empty path, a whole block `[block]`, a field `[field]`, and a block
 * field `[block, field, ...]` down to the value. A plain record or `null` layer has no values.
 *
 * @param layer A layer: record, `ValueTemplate`, `Binding` or `null`.
 * @returns `[path, leaf]` pairs (the leaf is the template string or typed `Binding`), in tree
 *   order.
 */
export function valueLeaves(layer: unknown): [ValuePath, unknown][] {
  const leaves: [ValuePath, unknown][] = [];
  if (layer instanceof ValueTemplate) collectLeaves(layer.value, [], leaves);
  else if (layer instanceof Binding && isValue(layer)) leaves.push([[], layer]);
  return leaves;
}

/**
 * Return the reference paths a layer's values read, once each, in source order.
 *
 * @param layer A layer: record, `ValueTemplate`, `Binding` or `null`.
 * @returns Paths such as `input.m` or `pick.output`.
 * @throws {ExpressionError} A `${...}` span does not parse.
 */
export function layerRefs(layer: unknown): string[] {
  if (isRecord(layer) || layer === null) return [];
  const refs = new Set<string>();
  for (const [, text] of iterSourceStrings(layer)) {
    for (const ref of exprRefsOf(text)) refs.add(ref);
  }
  return [...refs];
}

/**
 * Return whether every value in a layer reads only its flow's inputs (`input.*`).
 *
 * A layer without values reads nothing, so it qualifies.
 *
 * @param layer A layer: record, `ValueTemplate`, `Binding` or `null`.
 * @returns `true` when no reference has a head other than `input`.
 * @throws {ExpressionError} A `${...}` span does not parse.
 */
export function readsOnlyInputs(layer: unknown): boolean {
  return layerRefs(layer).every((ref) => ref.split(".", 1)[0] === "input");
}

/**
 * Return whether a value of `typ` may be null, or lack the part at `inner`.
 *
 * Each step reads a record field, which must be required and non-`Optional`, or a field every
 * member of a union of records has. Any other step, and an unknown type, may be absent.
 *
 * @param typ The value's type; `null` when it is not known.
 * @param inner Field names inside the value; defaults to empty.
 * @returns `true` unless the value, and every step down `inner`, is present.
 */
export function typeNullable(typ: Type | null, inner: ValuePath = []): boolean {
  if (typ === null) return true;
  let current = typ;
  for (const segment of inner) {
    if (admitsNone(current)) return true;
    if (current.variants !== undefined) {
      const read = unionField(current, String(segment));
      if (read.kind !== "read" || read.type === null) return true;
      current = read.type;
      continue;
    }
    if (typeof segment !== "string") return true;
    const field = current.fields?.get(segment);
    if (field === undefined || !current.required?.has(segment)) return true;
    current = field;
  }
  return admitsNone(current);
}

/**
 * A `Nullable` that knows only the types tags declare.
 *
 * A typed `Binding` answers from its contract; any other leaf may be null.
 *
 * @returns Whether the value, or its part at `inner`, may be null or absent.
 */
export function contractNullable(leaf: unknown, inner: ValuePath): boolean {
  if (leaf instanceof Binding && leaf.contract !== null) return typeNullable(leaf.contract.type, inner);
  return true;
}

/**
 * A `Nullable` that answers from the static type of each value in `layer`.
 *
 * @param layer The layer whose value leaves are asked about.
 * @param types Each value leaf's type by its field path; `null` is not proven, so the value may
 *   be null. A leaf whose path is missing answers as `contractNullable` does.
 * @returns Whether a leaf of `layer`, or its part at a path, may be null or absent.
 */
export function typesNullable(layer: unknown, types: readonly (readonly [ValuePath, Type | null])[]): Nullable {
  // A `Binding` leaf is matched by identity, a text leaf by its text: equal texts read the same
  // value.
  const paths = new Map<unknown, ValuePath>(valueLeaves(layer).map(([path, leaf]) => [leaf, path]));

  return (leaf, inner) => {
    const path = paths.get(leaf);
    const entry = path === undefined ? undefined : types.find(([typed]) => sameItems(typed, path));
    if (entry === undefined) return contractNullable(leaf, inner);
    return typeNullable(entry[1], inner);
  };
}

/**
 * Return `lower` without the values `higher` always overrides.
 *
 * A value sets one merge field: a top-level key, or a provider block's field (a deeper value sets
 * the block field it is inside). It can't contribute when `higher` sets that field to a non-null
 * literal, or to a value that can't be null. A higher whole block or whole record shields the
 * field only when it can't be null and its type declares the field as present and non-null. A
 * lower whole block or whole record always contributes, since a partial higher layer leaves the
 * rest of it.
 *
 * @param lower The lower-precedence layer.
 * @param higher The layer above it for the same agent.
 * @param nullable Whether a higher value leaf, or a part of it, may be null or absent.
 * @returns `lower` itself when every value can contribute; otherwise a `ValueTemplate` without
 *   the shielded fields and their origins.
 */
export function contributing(lower: unknown, higher: unknown, nullable: Nullable): unknown {
  const shielded: ValuePath[] = [];
  for (const [path] of valueLeaves(lower)) {
    const field = mergeField(path);
    if (field !== null && shields(higher, field, nullable) && !shielded.some((seen) => sameItems(seen, field))) {
      shielded.push(field);
    }
  }
  if (shielded.length === 0) return lower;
  // Only template layers have field values.
  return withoutFields(lower as ValueTemplate, shielded);
}

/**
 * Build the reserved binding that carries an agent's node and own-flow layers.
 *
 * The template is `{node: <node layer>, flow: <flow layer>}`, each carried whole with its
 * literal fields; only flow values the node layer always overrides are dropped. Under
 * `inherit: false` the flow part is left out, as it is when nothing of the flow layer is left.
 * Literal text is escaped so it resolves to itself, and origins are re-keyed under `node` and
 * `flow`.
 *
 * @param nodeLayer The agent's own layer.
 * @param flowLayer The layer of the flow the agent is written in.
 * @param inherit The agent's literal `inherit`.
 * @param nullable Whether a node-layer value leaf may be null or absent.
 * @returns The binding, or `null` when no value can reach the agent.
 */
export function splitLayers(
  nodeLayer: unknown,
  flowLayer: unknown,
  inherit: boolean,
  nullable: Nullable,
): Binding | null {
  const flow = inherit ? contributing(flowLayer, nodeLayer, nullable) : null;
  if (valueLeaves(nodeLayer).length === 0 && valueLeaves(flow).length === 0) return null;
  const tree: Record<string, unknown> = {};
  const origins: [ValuePath, OriginRef][] = [];
  for (const [part, layer] of [["node", nodeLayer], ["flow", flow]] as const) {
    const [value, partOrigins] = carried(layer);
    if (part === "flow" && isRecord(value) && Object.keys(value).length === 0) continue;
    tree[part] = value;
    for (const [path, origin] of partOrigins) origins.push([[part, ...path], origin]);
  }
  return new Binding(new ValueTemplate(tree, origins));
}

/**
 * Resolve a `splitLayers` binding into the node and flow layers.
 *
 * Each layer resolves on its own, so an error names the field under that layer's subject. A null
 * field stays null, which `mergeLlmConfig` reads as unset; a null whole record is an empty layer.
 *
 * @param template The binding `splitLayers` built.
 * @param pool The pool the values read.
 * @param options.nodeWhere Names a node-layer field in an error message, from its path in the
 *   layer, such as `node 'chat' llm_config.model`.
 * @param options.flowWhere Names a flow-layer field the same way, such as `flow llm_config.model`.
 * @returns The node layer and the flow layer.
 * @throws {BindingError} A value fails to evaluate, a typed value violates its tag
 *   (`binding.type`), or a whole record resolves to something other than a mapping or null
 *   (`contract.input`).
 */
export function resolveLayers(
  template: Binding,
  pool: VariablePool,
  options: { nodeWhere: FieldWhere; flowWhere: FieldWhere },
): [Record<string, unknown>, Record<string, unknown>] {
  const source = template.source as ValueTemplate;
  return [resolvePart(source, "node", pool, options.nodeWhere), resolvePart(source, "flow", pool, options.flowWhere)];
}

/** Resolve the `part` layer of a `splitLayers` template, naming fields with `where`. */
function resolvePart(
  source: ValueTemplate,
  part: string,
  pool: VariablePool,
  where: FieldWhere,
): Record<string, unknown> {
  const tree = source.value as Record<string, unknown>;
  if (!Object.hasOwn(tree, part)) return {};
  const layer = tree[part];
  const origins = source.origins.filter(([path]) => path[0] === part);
  const template = new Binding(new ValueTemplate({ [part]: layer }, origins));
  const label = (path: ValuePath): string => where(path.slice(1));
  const resolved = resolveSource(template, pool, { name: LLM_CONFIG_KEY, label }) as Record<string, unknown>;
  return resolvedLayer(resolved[part], layer, where([]));
}

/**
 * Return `layer` without the merge fields that hold a value `keep` rejects.
 *
 * A rejected value takes its whole merge field with it (a top-level key, or a provider block's
 * field), so no partial field is left; a rejected whole block takes the block, and a rejected
 * whole record leaves an empty layer. Literal fields stay.
 *
 * @param layer A layer: record, `ValueTemplate`, `Binding` or `null`.
 * @param keep Whether a value leaf stays.
 * @returns `layer` itself when every value stays; otherwise a `ValueTemplate` without the
 *   dropped fields and their origins, or `{}`.
 */
export function keepValues(layer: unknown, keep: (leaf: unknown) => boolean): unknown {
  const dropped: ValuePath[] = [];
  for (const [path, leaf] of valueLeaves(layer)) {
    if (!keep(leaf)) {
      const field = mergeField(path) ?? path;
      if (!dropped.some((seen) => sameItems(seen, field))) dropped.push(field);
    }
  }
  if (dropped.length === 0) return layer;
  if (dropped.some((field) => field.length === 0)) return {};
  // Only template layers have field values.
  return withoutFields(layer as ValueTemplate, dropped);
}

/**
 * Resolve one layer against a pool into a plain configuration record.
 *
 * Literal text resolves to itself (`$$` reads as `$`), and typed values are checked against their
 * tags. A null whole record is an empty layer.
 *
 * @param layer A layer: record, `ValueTemplate`, `Binding` or `null`.
 * @param pool The pool the values read.
 * @param options.where Names a field in an error message, from its path in the layer, such as
 *   `flow llm_config.model`.
 * @returns The resolved layer.
 * @throws {BindingError} As `resolveLayers` throws it.
 */
export function resolveLayer(
  layer: unknown,
  pool: VariablePool,
  options: { where?: FieldWhere } = {},
): Record<string, unknown> {
  const { where } = options;
  const [part, origins] = carried(layer);
  const layerOrigins = origins.map(([path, origin]) => [["layer", ...path], origin] as const);
  const template = new Binding(new ValueTemplate({ layer: part }, layerOrigins));
  const label = where === undefined ? null : (path: ValuePath) => where(path.slice(1));
  const resolved = resolveSource(template, pool, { name: LLM_CONFIG_KEY, label }) as Record<string, unknown>;
  return resolvedLayer(resolved["layer"], part, where === undefined ? null : where([]));
}

/**
 * Return why a layer resolved from values sets a field no flow file may set, or `null`.
 *
 * `_connection` is internal to recorded runs. A host-only field (`api_key`, `base_url`,
 * `organization`, `project`, `headers`) that a value set, as the whole record, a whole block or
 * the field itself, is refused whatever it resolved beside, since the same value may have set the
 * provider too. A host-only field written as a literal key is checked as an authored layer is,
 * against the layer's literal keys only: `base_url` passes only beside a literal
 * `provider: azure`.
 *
 * @param written The layer as written: a layer (record, `ValueTemplate`, `Binding` or `null`), or
 *   its part of a `splitLayers` binding.
 * @param resolved What `written` resolved to.
 * @param subject Names the layer, such as `node 'chat' llm_config`.
 * @returns The message, naming the field under `subject`; it never echoes a value.
 */
export function refusedResolvedField(
  written: unknown,
  resolved: Readonly<Record<string, unknown>>,
  subject: string,
): string | null {
  const found = refusal(written, resolved);
  if (found === null) return null;
  const [field, text] = found;
  return `${fieldWhere(subject, field)}: ${text}`;
}

/** A layer resolved from values, refused at one of its fields. */
export class LayerFieldError extends RangeError {
  override name = "LayerFieldError";
  /**
   * The field at fault; empty for the layer as a whole. A provider block's problem names the
   * deepest key path it reports, such as `["anthropic", "thinking", "budget_tokens"]`.
   */
  readonly field: ValuePath;

  /** Hold the message and the field it names. */
  constructor(message: string, field: ValuePath) {
    super(message);
    this.field = field;
  }
}

/**
 * Check a layer resolved from values as an authored layer is checked, naming the field.
 *
 * Messages read `<subject>.<field><callsite>: ...` for the first field at fault (a provider
 * block's problems name their key path), or `<subject><callsite>: ...` for a problem of the layer
 * as a whole, and never echo a value. A host-only field or `_connection` is refused first
 * (`refusedResolvedField`).
 *
 * @param layer The resolved layer.
 * @param subject Names the layer, such as `node 'chat' llm_config`.
 * @param options.written The layer as written, which tells a literal field from one a value set.
 * @param options.callsite Follows the field, such as `` in 'sub'`` for a synthesized agent's layer
 *   inside a call; defaults to empty.
 * @throws {LayerFieldError} The layer sets a field a flow file may not set, an unknown or retired
 *   field, an unknown provider, or a field or block its schema refuses.
 */
export function checkResolvedLayer(
  layer: Readonly<Record<string, unknown>>,
  subject: string,
  options: { written: unknown; callsite?: string },
): void {
  const { written, callsite = "" } = options;
  const found = refusal(written, layer);
  if (found !== null) {
    const [field, text] = found;
    throw new LayerFieldError(`${fieldWhere(subject, field)}${callsite}: ${text}`, field);
  }
  try {
    validateLayer(layer);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new LayerFieldError(`${subject}${callsite}: ${error.message}`, []);
  }
  for (const block of Object.keys(PROVIDER_BLOCKS)) {
    const value = Object.hasOwn(layer, block) ? layer[block] : null;
    if (value === null || value === undefined) continue;
    const problems = locatedBlockProblems(block, value, () => false);
    if (problems.length > 0) {
      const first = problems[0]!;
      throw new LayerFieldError(blockProblem(subject, block, first.message, callsite), [block, ...first.keys]);
    }
  }
  try {
    new LLMConfig(layer);
  } catch (error) {
    if (!(error instanceof ValidationError)) throw error;
    const issue = error.issues[0]!;
    let field: ValuePath = issue.loc.slice(0, 1);
    let text: string;
    if (issue.loc.length === 1 && issue.loc[0] === "provider" && issue.type === "literal_error") {
      text = `not a known provider; use one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`;
    } else if (issue.type === "value_error") {
      text = issue.error!;
    } else {
      // A block's field is named in full, as in `<subject>.openai.temperature`.
      text = issue.msg;
      field = issue.loc;
    }
    throw new LayerFieldError(`${fieldWhere(subject, field)}${callsite}: ${text}`, field);
  }
}

/**
 * Return a synthesized agent's `llm_config` label, with its callsite when nested.
 *
 * @param label The agent's `llm_config` label; `null` for an authored agent.
 * @param parent The static path of the callsite the agent is in; empty at the run's flow.
 * @returns `label`, or `<label> in '<parent>'` when nested; `null` for an authored agent, whose
 *   messages name its path instead.
 */
export function agentLabel(label: string | null, parent: string): string | null {
  if (label === null) return null;
  return parent ? `${label} in ${reprText(parent)}` : label;
}

/** Name a field under an `llm_config` subject, as in `node 'a' llm_config.model`. */
export function fieldWhere(subject: string, field: ValuePath): string {
  return subject + field.map((part) => `.${part}`).join("");
}

/**
 * Return the config one model call uses: `canonical` with the host's secrets joined.
 *
 * The host key and header values are joined only when `canonical` is marked for the host key
 * (`HOST_KEY_MARKER`). The result is for that call alone; it is never stored or recorded.
 *
 * @param canonical A resolved config with no secrets.
 * @param credentials The run's host key and header values.
 * @returns A new config.
 */
export function invocationConfig(
  canonical: Readonly<Record<string, unknown>>,
  credentials: HostConnection | null,
): AgentConfig {
  const config = structuredClone(canonical) as AgentConfig;
  if (config[HOST_KEY_MARKER] === true && credentials !== null) {
    if (credentials.apiKey !== null) config["api_key"] = credentials.apiKey;
    const headers = credentials.headers;
    if (headers !== null && Object.keys(headers).length > 0) config["headers"] = { ...headers };
  }
  return config;
}

/**
 * Resolve and check a call-time agent's config; return it with no secrets.
 *
 * The node and own-flow layers resolve against `pool` and are each checked as an authored layer
 * is. They merge over the recorded base, the defaults and saved models of the recorded selection
 * fill the gaps, and the selection binds the connection: an agent of the host key's provider is
 * marked for it with the host's base_url and nonsecret settings, as run start marks a static
 * agent. The result has the shape of a `RunStarted` map entry. It is then checked as the model
 * call would check it, with the host's secrets joined only when it is marked.
 *
 * @param template The agent's reserved `__llm_config` binding.
 * @param pool The pool its values read.
 * @param callTime What run start fixed for the agent.
 * @param options.label The authored site of a synthesized agent's `llm_config`, named instead of
 *   its path.
 * @param options.credentials The run's host key and header values.
 * @returns The canonical config: no key or header value.
 * @throws {BindingError} A value breaks its tag (`binding.type`) or fails to resolve (its own
 *   code), or the resolved config is refused (`contract.input`). The message names the field,
 *   under `node '<path>' llm_config` or the label, and the origin is that field's authored site
 *   when known.
 * @throws {AuthError} The config is marked for the host key, but the run holds no key, or header
 *   names other than those run start recorded.
 */
export function callTimeConfig(
  template: Binding,
  pool: VariablePool,
  callTime: CallTimeLLM,
  options: { label: string | null; credentials: HostConnection | null },
): AgentConfig {
  const { label, credentials } = options;
  const path = callTime.path;
  const parent = path.slice(0, Math.max(0, path.lastIndexOf("/")));
  const site = label || `node ${reprText(path)} llm_config`;
  const subject = agentLabel(label, parent) || site;
  // A nested label names its callsite after the field: `<label>.model in 'sub'`.
  const callsite = subject.slice(site.length);
  const flowSubject = parent ? `flow ${reprText(parent)} llm_config` : "flow llm_config";
  const tree = (template.source as ValueTemplate).value as Record<string, unknown>;
  const [nodeLayer, flowLayer] = resolveLayers(template, pool, {
    nodeWhere: (field) => fieldWhere(site, field) + callsite,
    flowWhere: (field) => fieldWhere(flowSubject, field),
  });
  const parts = [
    ["node", nodeLayer, site, callsite],
    ["flow", flowLayer, flowSubject, ""],
  ] as const;
  for (const [part, layer, where, after] of parts) {
    try {
      checkResolvedLayer(layer, where, { written: Object.hasOwn(tree, part) ? tree[part] : null, callsite: after });
    } catch (error) {
      if (!(error instanceof LayerFieldError)) throw error;
      throw refused(error.message, fieldOrigin(template, [part, ...error.field]));
    }
  }
  const selection = callTime.selection;
  const snapshot = selection.snapshot;
  let canonical: AgentConfig;
  try {
    const config = resolveAgentConfig(mergeLlmConfig(nodeLayer, mergeLlmConfig(flowLayer, callTime.base)), {
      defaults: snapshot.defaults,
      providerModels: snapshot.providerModels,
      strayModel: snapshot.strayModel,
      path,
      label,
    });
    canonical = selection.bind(config);
    requireConcreteSelection(canonical, { path, label });
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw refused(error.message, null);
  }
  const probe = invocationConfig(canonical, credentials);
  try {
    checkModelConfig(probe);
  } catch (error) {
    // An `AuthError` (the host key or a header missing) propagates: it is the host's to fix by
    // resuming with the key, not a failure of this node.
    if (!(error instanceof RangeError)) throw error;
    const block = refusedBlock(probe);
    const origin = block
      ? (fieldOrigin(template, ["node", block.slice(1)]) ?? fieldOrigin(template, ["flow", block.slice(1)]))
      : null;
    throw refused(blockProblem(site, block.slice(1), error.message, callsite), origin);
  }
  return canonical;
}

/** Return the `contract.input` failure of a resolved config refused at a field. */
function refused(message: string, origin: OriginRef | null): BindingError {
  return new BindingError(message, { failureCode: "contract.input", origin });
}

/**
 * Return the authored site of `field` (under `node` or `flow`), or its nearest.
 *
 * The deepest written key on the way down that has a site is used; a part written as one
 * whole-record value is that value's site.
 */
function fieldOrigin(template: Binding, field: ValuePath): OriginRef | null {
  const source = template.source as ValueTemplate;
  for (let end = field.length; end > 1; end--) {
    const origin = source.originAt(field.slice(0, end));
    if (origin !== null) return origin;
  }
  const head = field[0];
  const tree = source.value as Record<string, unknown>;
  const part = typeof head === "string" && Object.hasOwn(tree, head) ? tree[head] : null;
  return part instanceof Binding ? part.origin : null;
}

/** Return the field `refusedResolvedField` refuses and why, or `null`. */
function refusal(written: unknown, resolved: Readonly<Record<string, unknown>>): [ValuePath, string] | null {
  if (Object.hasOwn(resolved, "_connection")) return [["_connection"], "internal to recorded runs"];
  const literal = literalFields(written);
  for (const field of Object.keys(resolved)) {
    if (hostOnlyFieldError({ [field]: null }) === null) continue;
    if (literal === null || !Object.hasOwn(literal, field)) return [[field], HOST_ONLY_FROM_VALUE];
  }
  const message = hostOnlyFieldError(literal ?? {});
  return message === null ? null : [[], message];
}

/**
 * Return a written layer's literal top-level fields, or `null` for a whole-record value.
 *
 * A field holding a value anywhere inside it is left out; a tagged literal counts as the literal
 * it tags.
 */
function literalFields(written: unknown): Record<string, unknown> | null {
  const tree = unwrapped(written);
  if (isValue(tree)) return null;
  if (!isRecord(tree)) return {};
  return Object.fromEntries(
    Object.entries(tree)
      .filter(([, item]) => !dynamic(item))
      .map(([key, item]) => [key, unwrapped(item)]),
  );
}

/** Look through templates and tagged literals to the tree they hold. */
function unwrapped(node: unknown): unknown {
  for (;;) {
    if (node instanceof ValueTemplate) node = node.value;
    else if (node instanceof Binding && !isValue(node)) node = node.source;
    else return node;
  }
}

/** Return `text`'s template segments, or `null` when it doesn't scan. */
function segmentsOf(text: string): TemplateSegment[] | null {
  try {
    return scanTemplate(text);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
}

/** Whether `text` holds a `${...}` span; malformed text counts, for later checks. */
function hasSpan(text: string): boolean {
  const segments = segmentsOf(text);
  return segments === null || segments.some((segment) => segment instanceof Span);
}

/** Whether `text` is exactly one `${...}` span, whose value keeps its type; malformed text counts. */
function wholeSpan(text: string): boolean {
  const segments = segmentsOf(text);
  return segments === null || (segments.length === 1 && segments[0] instanceof Span);
}

/** Whether a tree, looking through nested bindings, holds a `${...}` span. */
function dynamic(node: unknown): boolean {
  if (typeof node === "string") return hasSpan(node);
  if (node instanceof Binding) return dynamic(node.source);
  if (node instanceof ValueTemplate) return dynamic(node.value);
  if (isRecord(node)) return Object.values(node).some(dynamic);
  if (Array.isArray(node)) return node.some(dynamic);
  return false;
}

/** Add every value under `node` to `leaves`, with its path. */
function collectLeaves(node: unknown, path: ValuePath, leaves: [ValuePath, unknown][]): void {
  if (isValue(node)) {
    leaves.push([path, node]);
  } else if (isRecord(node)) {
    for (const [key, item] of Object.entries(node)) collectLeaves(item, [...path, key], leaves);
  } else if (Array.isArray(node)) {
    for (const [index, item] of node.entries()) collectLeaves(item, [...path, index], leaves);
  }
}

/** Whether `null` satisfies `typ`. */
function admitsNone(typ: Type): boolean {
  return typ.nullable || typ.kind === ValueKind.NONE;
}

/** Return the field a value at `path` sets, or `null` for a whole record or block. */
function mergeField(path: ValuePath): ValuePath | null {
  if (path.length === 0) return null;
  const head = path[0]!;
  if (typeof head === "string" && Object.hasOwn(PROVIDER_BLOCKS, head)) {
    return path.length >= 2 ? path.slice(0, 2) : null;
  }
  return path.slice(0, 1);
}

/** Whether `higher` always sets `field` to a non-null value. */
function shields(higher: unknown, field: ValuePath, nullable: Nullable): boolean {
  let node = higher instanceof ValueTemplate ? higher.value : higher;
  for (const [index, segment] of field.entries()) {
    if (isValue(node)) return !nullable(node, []) && !nullable(node, field.slice(index));
    if (node instanceof Binding) {
      // A tagged literal: its source is the value.
      node = node.source;
      if (node instanceof ValueTemplate) node = node.value;
    }
    if (!isRecord(node) || typeof segment !== "string" || !Object.hasOwn(node, segment)) return false;
    node = node[segment];
  }
  return alwaysSet(node, nullable);
}

/** Whether a higher layer's field value is never null. */
function alwaysSet(node: unknown, nullable: Nullable): boolean {
  if (node === null) return false;
  if (typeof node === "string") {
    // Text around a span stringifies the value, so the result is never null.
    return !hasSpan(node) || !wholeSpan(node) || !nullable(node, []);
  }
  if (node instanceof Binding) return isValue(node) ? !nullable(node, []) : node.source !== null;
  return true;
}

/** Return `template` without the entries at `fields` and the origins under them. */
function withoutFields(template: ValueTemplate, fields: readonly ValuePath[]): ValueTemplate {
  return new ValueTemplate(
    pruned(template.value, fields, []) as ValueTemplate["value"],
    template.origins.filter(([path]) => !fields.some((field) => sameItems(path.slice(0, field.length), field))),
  );
}

/** Rebuild a template tree without the entries at `fields`. */
function pruned(node: unknown, fields: readonly ValuePath[], path: ValuePath): unknown {
  if (!isRecord(node)) return node;
  const out: ValueRecord = {};
  for (const [key, item] of Object.entries(node)) {
    const here = [...path, key];
    if (fields.some((field) => sameItems(field, here))) continue;
    const inside = fields.some((field) => sameItems(field.slice(0, here.length), here));
    setField(out, key, (inside ? pruned(item, fields, here) : item) as Value);
  }
  return out;
}

/** Return a layer as a template part and the origins inside it. */
function carried(layer: unknown): [unknown, readonly (readonly [ValuePath, OriginRef])[]] {
  if (layer === null) return [{}, []];
  if (layer instanceof ValueTemplate) return [layer.value, layer.origins];
  if (layer instanceof Binding) {
    if (layer.source instanceof ValueTemplate) {
      return [
        new Binding(layer.source.value, { origin: layer.origin, contract: layer.contract }),
        layer.source.origins,
      ];
    }
    return [layer, []];
  }
  return [escaped(layer), []];
}

/** Escape every `$` in a literal tree's text, so the binder reads it back unchanged. */
function escaped(node: unknown): unknown {
  // A function replacement, since in a replacement string `"$$"` means one `$`.
  if (typeof node === "string") return node.replaceAll("$", () => "$$");
  if (isRecord(node)) {
    const out: ValueRecord = {};
    for (const [key, item] of Object.entries(node)) setField(out, key, escaped(item) as Value);
    return out;
  }
  if (Array.isArray(node)) return node.map(escaped);
  return node;
}

/** Return one resolved layer, with a null whole record as an empty layer. */
function resolvedLayer(value: unknown, carriedPart: unknown, label: string | null): Record<string, unknown> {
  if (value === null) return {};
  if (isRecord(value)) return value;
  const where = label ?? `binding ${reprText(LLM_CONFIG_KEY)}`;
  throw new BindingError(`${where}: llm_config must resolve to a mapping or null, got ${pythonTypeName(value)}`, {
    failureCode: "contract.input",
    origin: carriedPart instanceof Binding ? carriedPart.origin : null,
  });
}
