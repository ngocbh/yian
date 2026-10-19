/**
 * Values (`${...}`) in an AGENT's `llm_config` layers: leaves, contribution, carrying, resolving.
 *
 * A layer is a plain record (all literal), a `ValueTemplate` (fields, block fields or whole
 * blocks from values), or a `Binding` (the whole record from one value). The helpers here find a
 * layer's value leaves and the references they read, decide which lower-layer values a higher
 * layer leaves able to contribute, and carry an agent's node and own-flow layers as one reserved
 * binding. Precedence and merge granularity follow `mergeLlmConfig`.
 *
 * Knows about:   diagnostics, expr, llm_clients, nodes/contracts and typesys.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { OriginRef } from "../../diagnostics.ts";
import { ExpressionError, exprRefsOf, scanTemplate } from "../../expr/index.ts";
import { Span } from "../../expr/template.ts";
import { PROVIDER_BLOCKS } from "../../llm_clients/provider_blocks.ts";
import { isRecord, type Type, unionField, type Value, ValueKind, type ValueRecord } from "../../typesys/index.ts";
import { setField } from "../../typesys/values.ts";
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
    const field = typeof segment === "string" ? current.fields?.get(segment) : undefined;
    if (field === undefined || !current.required?.has(segment as string)) return true;
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
  // Leaves are matched by identity; two leaves with equal text read the same value.
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
  const template = lower as ValueTemplate;
  return new ValueTemplate(
    pruned(template.value, shielded, []) as ValueTemplate["value"],
    template.origins.filter(([path]) => !shielded.some((field) => sameItems(path.slice(0, field.length), field))),
  );
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
  const template = layer as ValueTemplate;
  return new ValueTemplate(
    pruned(template.value, dropped, []) as ValueTemplate["value"],
    template.origins.filter(([path]) => !dropped.some((field) => sameItems(path.slice(0, field.length), field))),
  );
}

/** Whether `text` holds a `${...}` span; malformed text counts, for later checks. */
function hasSpan(text: string): boolean {
  try {
    return scanTemplate(text).some((segment) => segment instanceof Span);
  } catch (error) {
    if (error instanceof ExpressionError) return true;
    throw error;
  }
}

/** Whether `text` is exactly one `${...}` span, whose value keeps its type. */
function wholeSpan(text: string): boolean {
  let segments;
  try {
    segments = scanTemplate(text);
  } catch (error) {
    if (error instanceof ExpressionError) return true;
    throw error;
  }
  return segments.length === 1 && segments[0] instanceof Span;
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
  if (typeof node === "string") return node.replaceAll("$", () => "$$");
  if (isRecord(node)) {
    const out: ValueRecord = {};
    for (const [key, item] of Object.entries(node)) setField(out, key, escaped(item) as Value);
    return out;
  }
  if (Array.isArray(node)) return node.map(escaped);
  return node;
}
