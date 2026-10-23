/**
 * Derive and atomically stamp source-neutral static memory scopes.
 *
 * Memory declarations remain authored identity on `CompiledFlow` and `AgentNode`. This module
 * resolves them into a sorted memory key table and one exact `MemoryScope` for every statically
 * reachable node. A scope names the node's root-relative locator and the keys it recalls and learns
 * into, outermost first. Resolution is pure; applying a resolved layout is a separate,
 * all-or-nothing mutation over a private compiled graph.
 *
 * Knows about:   compile/model, memory/contracts, nodes (base, contracts, agent, start, end),
 *                diagnostics and typesys.
 * Never imports: compose, runtime, or the package-root run boundary.
 */

import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";
import {
  DeclSite,
  type JsonObject,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryDecl,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySetupError,
  MemorySpec,
  MemoryStack,
  sameMemoryScope,
  sameMemorySpec,
} from "../memory/contracts.ts";
import { AgentNode } from "../nodes/agent/node.ts";
import { Node } from "../nodes/base.ts";
import { comparePaths, isExactly, sameItems, sameOptional } from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import { logJson } from "../typesys/json.ts";
import { isList, jsTypeName, writeAsText } from "../typesys/values.ts";
import { CompiledFlow, END_ID, Flow, START_ID } from "./model.ts";

/** Return a key no two distinct locators share; never shown. */
function locatorKey(locator: readonly string[]): string {
  return JSON.stringify(locator);
}

/** Return a key no two unequal sites share (Python hashes a `DeclSite` by value); never shown. */
function siteKey(site: DeclSite): string {
  return JSON.stringify([site.scope, site.owner, site.node]);
}

/**
 * Encode one JSON value for exact comparison, so `1n`, `1` (Python's `1.0`) and `true` differ.
 * Log JSON marks a float with `.0` and writes any float, where the JCS writer spells `1n` and `1`
 * alike and refuses a fraction.
 */
function exactJson(value: JsonObject): string {
  return logJson(value);
}

/**
 * Replay every exact nested invariant of one occurrence path.
 *
 * @throws TypeError when `value` isn't a plain array of exact `MemoryOccurrenceSegment`s, or as
 *   the segment constructor does.
 * @throws RangeError as the segment constructor does.
 */
export function revalidateMemoryOccurrencePath(value: unknown, name: string): readonly MemoryOccurrenceSegment[] {
  if (!isList(value)) throw new TypeError(`${name} must be an exact array`);
  for (const [index, segment] of value.entries()) {
    if (!isExactly(segment, MemoryOccurrenceSegment)) {
      throw new TypeError(`${name}[${index}] must be an exact MemoryOccurrenceSegment`);
    }
    new MemoryOccurrenceSegment(segment.kind, segment.staticPath, segment.index);
  }
  return value as readonly MemoryOccurrenceSegment[];
}

/**
 * Replay every exact invariant nested below one frozen memory scope.
 *
 * A value can bypass the constructor (an object of the class's prototype), so rebuilding each
 * layer keeps a noncanonical key or occurrence index from crossing the compile boundary.
 *
 * @throws TypeError when `value` isn't an exact `MemoryScope`, or as its constructor does.
 * @throws RangeError as its constructor does.
 */
export function revalidateMemoryScope(value: unknown, name = "memory_scope"): MemoryScope {
  if (!isExactly(value, MemoryScope)) throw new TypeError(`${name} must be an exact MemoryScope`);
  revalidateMemoryOccurrencePath(value.occurrencePath, `${name}.occurrence_path`);
  new MemoryScope(value.staticPath, { memories: value.memories, occurrencePath: value.occurrencePath });
  return value;
}

/**
 * One reachable memory key with its owners. Frozen.
 *
 * `origins` aligns with `owners` and serves diagnostics only.
 */
export class ResolvedMemory {
  readonly key: string;
  readonly type: string;
  /** The deep-frozen config snapshot. */
  readonly config: JsonObject;
  /** The declaring sites, sorted and unique by `DeclSite.compare`. */
  readonly owners: readonly DeclSite[];
  /** Each owner's declaration origin; not part of a layout's value. */
  readonly origins: readonly (OriginRef | null)[];

  /**
   * Validate the entry through `MemorySpec` and keep frozen copies.
   *
   * @throws TypeError as `MemorySpec` does, or for an origin that is neither `null` nor an
   *   `OriginRef`.
   * @throws RangeError as `MemorySpec` does, or for origins that don't align with the owners.
   */
  constructor(
    key: string,
    type: string,
    config: JsonObject,
    owners: readonly DeclSite[],
    origins: readonly (OriginRef | null)[],
  ) {
    const spec = new MemorySpec(key, type, config, owners);
    if (!isList(origins) || origins.length !== spec.owners.length) {
      throw new RangeError("origins must align with owners");
    }
    if (origins.some((origin) => origin !== null && !(origin instanceof OriginRef))) {
      throw new TypeError("origins must hold OriginRef values or null");
    }
    this.key = spec.key;
    this.type = spec.type;
    this.config = spec.config;
    this.owners = spec.owners;
    this.origins = Object.freeze([...origins]);
    Object.freeze(this);
  }

  /** The record-facing table entry. */
  get spec(): MemorySpec {
    return new MemorySpec(this.key, this.type, this.config, this.owners);
  }

  /** The first owner's origin, where key-level errors point. */
  get origin(): OriginRef | null {
    return this.origins[0]!;
  }
}

/** One exact static node target and the scope it must receive. Frozen. */
export class ResolvedMemoryTarget {
  /** The node's root-relative locator; a frozen copy. */
  readonly locator: readonly string[];
  readonly scope: MemoryScope | null;
  /** The live node the scope is stamped on; not part of a layout's value. */
  readonly node: Node;

  /**
   * Validate the locator, the scope and the target class.
   *
   * @throws TypeError for a locator component that isn't a non-empty string, a scope that is
   *   neither `null` nor an exact `MemoryScope`, or a node that isn't a `Node`.
   * @throws RangeError for an empty or non-array locator, or as the scope constructor does.
   */
  constructor(locator: readonly string[], scope: MemoryScope | null, node: Node) {
    if (!isList(locator) || locator.length === 0) throw new RangeError("target locator must be a non-empty array");
    if (locator.some((part) => typeof part !== "string" || !part)) {
      throw new TypeError("target locator components must be non-empty strings");
    }
    if (scope !== null && !isExactly(scope, MemoryScope)) {
      throw new TypeError("target scope must be an exact MemoryScope or null");
    }
    if (scope !== null) revalidateMemoryScope(scope, "target scope");
    if (!(node instanceof Node)) throw new TypeError("target node must be a Node");
    this.locator = Object.freeze([...locator]);
    this.scope = scope;
    this.node = node;
    Object.freeze(this);
  }
}

/**
 * Immutable static memory layout ready for atomic application. Frozen.
 *
 * `enabled` is true when a reachable flow or AGENT declares a memory. Then every target has a
 * scope whose `staticPath` is its locator and whose keys are all in `memories`. A disabled layout
 * has no scopes and no memories.
 */
export class ResolvedMemoryLayout {
  readonly enabled: boolean;
  /** The targets in resolution order; a frozen copy. */
  readonly targets: readonly ResolvedMemoryTarget[];
  /** The key table, sorted and unique by key; a frozen copy. */
  readonly memories: readonly ResolvedMemory[];

  /**
   * Validate exact containers, replay every target and memory, and check that every scope agrees
   * with the table.
   *
   * @throws TypeError for a non-boolean `enabled`, containers that aren't arrays of exact targets
   *   or memories, or as their constructors do.
   * @throws RangeError for keys out of order, a disabled layout with memories or scopes, an
   *   enabled layout without memories, or a scope that is missing, doesn't match its locator or
   *   names a key outside the table.
   */
  constructor(enabled: boolean, targets: readonly ResolvedMemoryTarget[], memories: readonly ResolvedMemory[] = []) {
    if (typeof enabled !== "boolean") throw new TypeError("enabled must be a bool");
    if (!isList(targets) || targets.some((target) => !isExactly(target, ResolvedMemoryTarget))) {
      throw new TypeError("targets must be an array of ResolvedMemoryTarget values");
    }
    if (!isList(memories) || memories.some((memory) => !isExactly(memory, ResolvedMemory))) {
      throw new TypeError("memories must be an array of ResolvedMemory values");
    }
    for (const target of targets) new ResolvedMemoryTarget(target.locator, target.scope, target.node);
    for (const memory of memories) {
      new ResolvedMemory(memory.key, memory.type, memory.config, memory.owners, memory.origins);
    }
    const keys = memories.map((memory) => memory.key);
    if (keys.some((key, i) => i > 0 && compareCodeUnits(keys[i - 1]!, key) >= 0)) {
      throw new RangeError("memories must be sorted and unique by key");
    }
    if (!enabled) {
      if (memories.length > 0) throw new RangeError("a disabled memory layout cannot contain memories");
      if (targets.some((target) => target.scope !== null)) {
        throw new RangeError("a disabled memory layout cannot contain scopes");
      }
    } else {
      if (memories.length === 0) throw new RangeError("an enabled memory layout must contain memories");
      const known = new Set(keys);
      for (const [index, target] of targets.entries()) {
        if (target.scope === null) throw new RangeError("an enabled memory layout must scope every target");
        if (!sameItems(target.scope.staticPath, target.locator)) {
          throw new RangeError(`targets[${index}] scope static_path must equal its locator`);
        }
        if (!target.scope.memories.every((key) => known.has(key))) {
          throw new RangeError(`targets[${index}] scope names a memory outside the table`);
        }
      }
    }
    this.enabled = enabled;
    this.targets = Object.freeze([...targets]);
    this.memories = Object.freeze([...memories]);
    Object.freeze(this);
  }
}

/**
 * Whether two layouts are equal by value: Python's dataclass equality, which leaves out each
 * target's node and each memory's origins.
 */
function sameLayout(a: ResolvedMemoryLayout, b: ResolvedMemoryLayout): boolean {
  return (
    a.enabled === b.enabled &&
    sameItems(
      a.targets,
      b.targets,
      (x, y) =>
        sameItems(x.locator, y.locator) &&
        sameOptional(x.scope, y.scope, sameMemoryScope),
    ) &&
    sameItems(a.memories, b.memories, (x, y) => sameMemorySpec(x.spec, y.spec))
  );
}

/** Build one layout failure with its first authored site, any others and its notes. */
function setupError(
  message: string,
  options: {
    origin?: OriginRef | null;
    related?: readonly (OriginRef | null)[];
    notes?: readonly string[];
    cause?: unknown;
  } = {},
): MemorySetupError {
  const { origin = null, related = [], notes = [] } = options;
  return new MemorySetupError(message, {
    origin,
    related: related.filter((item): item is OriginRef => item !== null),
    notes,
    ...("cause" in options ? { cause: options.cause } : {}),
  });
}

/** Replay the complete frozen layout contract on a layout that may have bypassed it. */
function revalidateResolvedLayout(layout: ResolvedMemoryLayout): void {
  if (!isExactly(layout, ResolvedMemoryLayout)) throw new TypeError("layout must be an exact ResolvedMemoryLayout");
  new ResolvedMemoryLayout(layout.enabled, layout.targets, layout.memories);
}

/** Require exact direct START/END targets before a flow owns memory. */
function validateMemoryFlowBoundaries(flow: Flow, decl: MemoryDecl): void {
  const origin = decl.origin;
  if (!isExactly(flow.nodes, Map)) {
    throw setupError("a memory-owning flow requires an exact node dictionary", { origin });
  }
  if (typeof flow.startId !== "string" || !flow.startId) {
    throw setupError("a memory-owning flow start_id must be a non-empty exact string", { origin });
  }
  if (typeof flow.endId !== "string" || !flow.endId) {
    throw setupError("a memory-owning flow end_id must be a non-empty exact string", { origin });
  }
  if (flow.startId === flow.endId) {
    throw setupError("a memory-owning flow must have distinct START and END boundaries", { origin });
  }
  if (flow.startId !== START_ID) {
    throw setupError(`a memory-owning flow start_id must be canonical ${reprText(START_ID)}`, { origin });
  }
  if (flow.endId !== END_ID) {
    throw setupError(`a memory-owning flow end_id must be canonical ${reprText(END_ID)}`, { origin });
  }
  const boundaries: [string, string][] = [["START", flow.startId], ["END", flow.endId]];
  const missing = boundaries.filter(([, nodeId]) => !flow.nodes.has(nodeId)).map(([label]) => label);
  if (missing.length > 0) {
    throw setupError(`a memory-owning flow is missing direct ${missing.join(" and ")} boundary node(s)`, { origin });
  }
  if (!isExactly(flow.nodes.get(START_ID), StartNode)) {
    throw setupError("a memory-owning flow START boundary must be an exact StartNode", { origin });
  }
  if (!isExactly(flow.nodes.get(END_ID), EndNode)) {
    throw setupError("a memory-owning flow END boundary must be an exact EndNode", { origin });
  }
}

/**
 * Return one owner's declarations in authored order, each key once.
 *
 * Every declaration's site must name `node`, the owning AGENT id or `null` for a flow, so a
 * hand-built graph cannot forge another owner.
 */
function ownerDecls(
  decls: unknown,
  label: string,
  origin: OriginRef | null,
  node: string | null,
): readonly MemoryDecl[] {
  if (!isList(decls) || decls.some((decl) => !isExactly(decl, MemoryDecl))) {
    throw setupError(`${label} memories must be an array of exact MemoryDecl values`, { origin });
  }
  const seen = new Set<string>();
  for (const decl of decls as readonly MemoryDecl[]) {
    if (decl.site.node !== node) {
      const named = decl.site.node === null ? "null" : reprText(decl.site.node);
      throw setupError(`${label} memory ${reprText(decl.key)} names site node ${named}`, { origin: decl.origin });
    }
    if (seen.has(decl.key)) {
      throw setupError(`${label} declares memory ${reprText(decl.key)} more than once`, { origin: decl.origin });
    }
    seen.add(decl.key);
  }
  return decls as readonly MemoryDecl[];
}

/** Return the declarations owned by an exact compiled flow. */
function flowDecls(flow: Flow): readonly MemoryDecl[] {
  if (!isExactly(flow, CompiledFlow)) return [];
  return ownerDecls(flow.memories, "flow", null, null);
}

/** Return the declarations owned by an exact AGENT node. */
function agentDecls(node: Node): readonly MemoryDecl[] {
  if (!isExactly(node, AgentNode)) return [];
  return ownerDecls(node.memories, `AGENT node ${reprText(node.id)}`, node.origin, node.id);
}

/**
 * Append each declared key not already in `stack`.
 *
 * A repeated key keeps its outermost position, so it yields one recall section and enters each
 * learning batch once.
 */
function extendStack(stack: readonly string[], decls: readonly MemoryDecl[]): readonly string[] {
  const keys = [...stack];
  for (const decl of decls) {
    if (!keys.includes(decl.key)) keys.push(decl.key);
  }
  return keys;
}

/** Return a validated deterministic view of one flow's direct nodes. */
function flowNodes(flow: Flow, origin: OriginRef | null = null): [string, Node][] {
  if (!isExactly(flow, Flow) && !isExactly(flow, CompiledFlow)) {
    throw setupError(`memory layout expected an exact Flow, got ${jsTypeName(flow)}`, { origin });
  }
  if (!isExactly(flow.nodes, Map)) throw setupError("flow nodes must be an exact dictionary", { origin });
  const ids: unknown[] = [...flow.nodes.keys()];
  for (const nodeId of ids) {
    if (typeof nodeId !== "string" || !nodeId) {
      throw setupError("flow node ids must be non-empty exact strings", { origin });
    }
  }
  return (ids as string[]).sort(compareCodeUnits).map((nodeId) => {
    const node: unknown = flow.nodes.get(nodeId);
    if (!(node instanceof Node)) throw setupError(`flow node ${reprText(nodeId)} must be a Node`, { origin });
    if (typeof node.id !== "string" || !node.id) {
      throw setupError(`flow node ${reprText(nodeId)} has an invalid node id`, { origin: node.origin });
    }
    if (node.id !== nodeId) {
      throw setupError(`flow node key ${reprText(nodeId)} does not match node id ${reprText(node.id)}`, {
        origin: node.origin,
      });
    }
    return [nodeId, node];
  });
}

/** Return the node's single exact static child, if present. */
function nodeChild(node: Node): Flow | null {
  const children: unknown = node.originChildren();
  if (!isList(children)) {
    throw setupError(`node ${reprText(node.id)} static children must be an exact array`, { origin: node.origin });
  }
  if (children.length > 1) {
    throw setupError(`node ${reprText(node.id)} exposes more than one static child`, { origin: node.origin });
  }
  if (children.length === 0) return null;
  const child: unknown = children[0];
  if (!isExactly(child, Flow) && !isExactly(child, CompiledFlow)) {
    throw setupError(`node ${reprText(node.id)} static child must be an exact Flow`, { origin: node.origin });
  }
  return child;
}

/**
 * Prefix `alias` to the scope of every reachable memory declaration, in place.
 *
 * Call this only on a graph the caller owns, such as the independent copy that `uses:` rebasing
 * returns. The walk follows the same static children as layout resolution and visits each flow
 * and node object once, so a def baked into several call sites, or a node shared by two copied
 * flows, is rewritten once. Each declaration array is replaced, never edited.
 *
 * @param flow Owned root of the graph to rewrite.
 * @param alias `uses:` alias that scopes the imported graph.
 * @returns `flow` itself, after the rewrite.
 */
export function scopeMemoryDeclarations(flow: Flow, alias: string): Flow {
  const seenFlows = new Set<object>();
  const seenNodes = new Set<object>();
  const pending: Flow[] = [flow];
  while (pending.length > 0) {
    const current = pending.pop()!;
    if (seenFlows.has(current)) continue;
    seenFlows.add(current);
    if (current instanceof CompiledFlow) current.memories = scoped(current.memories, alias);
    for (const [, node] of flowNodes(current)) {
      if (seenNodes.has(node)) continue;
      seenNodes.add(node);
      if (node instanceof AgentNode) node.memories = scoped(node.memories, alias);
      const child = nodeChild(node);
      if (child !== null) pending.push(child);
    }
  }
  return flow;
}

/** Return declarations whose site scope starts with `alias`. */
function scoped(memories: readonly MemoryDecl[], alias: string): readonly MemoryDecl[] {
  return Object.freeze(
    memories.map(
      (decl) =>
        new MemoryDecl(
          decl.name,
          decl.type,
          decl.config,
          new DeclSite([alias, ...decl.site.scope], decl.site.owner, decl.site.node),
          decl.origin,
        ),
    ),
  );
}

/** Render one declaration site for diagnostics, led by its import scope if any. */
function siteLabel(site: DeclSite): string {
  let label = site.owner.join("/");
  if (site.node !== null) label = `${label} AGENT ${reprText(site.node)}`;
  return site.scope.length > 0 ? `${site.scope.join(".")}: ${label}` : label;
}

/**
 * Group reachable declarations by key and enforce the per-key rules. Exported for tests.
 *
 * Each key has one type and config.
 *
 * @throws MemorySetupError when a rule fails; its notes name the other sites.
 */
export function memoryTable(declared: readonly MemoryDecl[]): readonly ResolvedMemory[] {
  const byKey = new Map<string, MemoryDecl[]>();
  for (const decl of declared) {
    const entries = byKey.get(decl.key);
    if (entries === undefined) byKey.set(decl.key, [decl]);
    else entries.push(decl);
  }
  const table: ResolvedMemory[] = [];
  for (const key of [...byKey.keys()].sort(compareCodeUnits)) {
    const entries = byKey.get(key)!;
    // Stable sort: declarations at one site keep walk order.
    const declarations = [...entries].sort((a, b) => DeclSite.compare(a.site, b.site));
    const first = declarations[0]!;
    const firstConfig = exactJson(first.config);
    const conflicts = declarations
      .slice(1)
      .filter((decl) => decl.type !== first.type || exactJson(decl.config) !== firstConfig);
    if (conflicts.length > 0) {
      // A def called from several nodes repeats its declaration, as one object or as per-call
      // copies; list each other declaration once, compared by site and value, not identity.
      const distinct = new Map<string, MemoryDecl>();
      for (const decl of declarations) {
        const identity = JSON.stringify([siteKey(decl.site), decl.type, exactJson(decl.config)]);
        if (!distinct.has(identity)) distinct.set(identity, decl);
      }
      const others = [...distinct.values()].slice(1);
      throw setupError(
        `memory ${reprText(key)} is declared with different type or config ` +
          `at ${siteLabel(first.site)} and ${siteLabel(conflicts[0]!.site)}`,
        {
          origin: first.origin,
          related: others.map((decl) => decl.origin),
          notes: others.map((decl) => `also declared at ${siteLabel(decl.site)}`),
        },
      );
    }
    const bySite = new Map<string, MemoryDecl>();
    for (const decl of declarations) {
      if (!bySite.has(siteKey(decl.site))) bySite.set(siteKey(decl.site), decl);
    }
    const ordered = [...bySite.values()];
    const owners = ordered.map((decl) => decl.site);
    table.push(new ResolvedMemory(key, first.type, first.config, owners, ordered.map((decl) => decl.origin)));
  }
  return Object.freeze(table);
}

/**
 * Reject unsafe scope segments before keys are grouped.
 *
 * A dotted alias renders the same key as a nested import, so this runs first to name the alias
 * rather than a key clash.
 */
function checkScopeSegments(declared: readonly MemoryDecl[]): void {
  for (const decl of declared) {
    for (const segment of decl.site.scope) {
      if (!MEMORY_IDENTIFIER_PATTERN.test(segment)) {
        throw setupError(
          `memory ${reprText(decl.key)}: uses: alias ${reprText(segment)} must ` +
            "match [A-Za-z_][A-Za-z0-9_-]* to scope memories",
          { origin: decl.origin },
        );
      }
    }
  }
}

/** Reject keys that differ only by case. */
function checkKeyCase(memories: readonly ResolvedMemory[]): void {
  // `toLowerCase` is Python's `casefold` here: key segments match the ASCII-only identifier grammar.
  const folded = new Map<string, ResolvedMemory>();
  for (const memory of memories) {
    const fold = memory.key.toLowerCase();
    const earlier = folded.get(fold);
    if (earlier === undefined) {
      folded.set(fold, memory);
      continue;
    }
    throw setupError(
      `memory keys ${reprText(earlier.key)} and ${reprText(memory.key)} differ only by ` +
        "case; storage paths would collide on case-insensitive filesystems",
      { origin: earlier.origin, related: memory.origins },
    );
  }
}

/**
 * Return every scoped target's key stack, sorted by locator.
 *
 * @throws TypeError or RangeError when `layout` breaks its contract.
 */
export function memoryStacks(layout: ResolvedMemoryLayout): readonly MemoryStack[] {
  revalidateResolvedLayout(layout);
  if (!layout.enabled) return Object.freeze([]);
  return Object.freeze(
    layout.targets
      .map((target) => new MemoryStack(target.locator, target.scope!.memories))
      .sort((a, b) => comparePaths(a.locator, b.locator)),
  );
}

/** Resolve one flow occurrence into node scopes and reachable declarations. */
function resolveFlow(
  flow: Flow,
  locator: readonly string[],
  inherited: readonly string[],
  targets: ResolvedMemoryTarget[],
  declared: MemoryDecl[],
  activeFlows: Set<object>,
  ownerOrigin: OriginRef | null = null,
): void {
  if (activeFlows.has(flow)) {
    throw setupError("recursive child-flow cycle is not memory-layout resolvable", { origin: ownerOrigin });
  }
  activeFlows.add(flow);
  try {
    const decls = flowDecls(flow);
    if (decls.length > 0) {
      validateMemoryFlowBoundaries(flow, decls[0]!);
      declared.push(...decls);
    }
    const stack = extendStack(inherited, decls);

    for (const [nodeId, node] of flowNodes(flow, ownerOrigin)) {
      const nodeLocator = [...locator, nodeId];
      const ownDecls = agentDecls(node);
      declared.push(...ownDecls);
      targets.push(
        new ResolvedMemoryTarget(
          nodeLocator,
          new MemoryScope(nodeLocator, { memories: extendStack(stack, ownDecls) }),
          node,
        ),
      );
      const child = nodeChild(node);
      if (child !== null) resolveFlow(child, nodeLocator, stack, targets, declared, activeFlows, node.origin);
    }
  } finally {
    activeFlows.delete(flow);
  }
}

/**
 * Purely derive the memory key table and static node scopes for `flow`.
 *
 * Existing `Node.memoryScope` values are ignored. A graph with no reachable flow or AGENT
 * declaration resolves to a disabled layout whose targets carry no scope. Otherwise every
 * reachable static node receives a scope, including memory-neutral composite drivers and flow
 * boundaries. No flow id is required.
 *
 * @throws TypeError when `flow` isn't an exact `CompiledFlow`.
 * @throws MemorySetupError when the graph or its declarations break a layout rule.
 */
export function resolveMemoryLayout(flow: CompiledFlow): ResolvedMemoryLayout {
  if (!isExactly(flow, CompiledFlow)) throw new TypeError("resolveMemoryLayout expects an exact CompiledFlow");
  const targets: ResolvedMemoryTarget[] = [];
  const declared: MemoryDecl[] = [];
  resolveFlow(flow, [], [], targets, declared, new Set());
  if (declared.length === 0) {
    return new ResolvedMemoryLayout(
      false,
      targets.map((target) => new ResolvedMemoryTarget(target.locator, null, target.node)),
    );
  }
  checkScopeSegments(declared);
  const memories = memoryTable(declared);
  checkKeyCase(memories);
  return new ResolvedMemoryLayout(true, targets, memories);
}

/** Reject duplicate locators or aliased node objects before any mutation. */
function validateUniqueTargets(layout: ResolvedMemoryLayout): void {
  const locators = new Set<string>();
  const nodes = new Set<object>();
  for (const target of layout.targets) {
    const key = locatorKey(target.locator);
    if (locators.has(key)) {
      throw setupError(`memory layout contains duplicate target path ${writeAsText([...target.locator])}`, {
        origin: target.node.origin,
      });
    }
    locators.add(key);
    if (nodes.has(target.node)) {
      throw setupError(
        "memory layout reaches one node object through multiple static paths; " +
          "the run setup must de-alias child flows before applying it",
        { origin: target.node.origin },
      );
    }
    nodes.add(target.node);
  }
}

/**
 * Atomically apply one previously resolved layout to its exact graph.
 *
 * The complete graph, semantic paths, the key table, exact node identities, duplicate targets,
 * and every existing scope are validated first. Only after that pass succeeds are scopes
 * assigned, so a stale or partially stamped graph cannot receive a partial layout. Reapplying the
 * same layout is a no-op.
 *
 * @throws TypeError when `flow` isn't an exact `CompiledFlow` or `layout` breaks its contract.
 * @throws MemorySetupError when the layout no longer matches the graph, targets are aliased, or a
 *   node already has a noncanonical or different scope.
 */
export function applyMemoryLayout(flow: CompiledFlow, layout: ResolvedMemoryLayout): void {
  if (!isExactly(flow, CompiledFlow)) throw new TypeError("applyMemoryLayout expects an exact CompiledFlow");
  revalidateResolvedLayout(layout);

  validateUniqueTargets(layout);
  const current = resolveMemoryLayout(flow);
  validateUniqueTargets(current);
  if (!sameLayout(current, layout)) throw setupError("memory layout no longer matches the compiled flow");

  // Equal layouts list the same locators in the same order.
  for (const [i, target] of layout.targets.entries()) {
    const actual = current.targets[i]!;
    const where = writeAsText([...target.locator]);
    if (actual.node !== target.node) {
      throw setupError(`memory layout target ${where} no longer names the same node`, { origin: actual.node.origin });
    }
    const existing = actual.node.memoryScope;
    if (existing === null) continue;
    try {
      revalidateMemoryScope(existing, `node at ${where} memory_scope`);
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
      throw setupError(`node at ${where} has a noncanonical memory scope`, { origin: actual.node.origin, cause: error });
    }
    if (target.scope === null || !sameMemoryScope(existing, target.scope)) {
      throw setupError(`node at ${where} already has a different memory scope`, { origin: actual.node.origin });
    }
  }

  for (const target of layout.targets) target.node.memoryScope = target.scope;
}
