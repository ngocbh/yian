/**
 * Union narrowing: what each node may assume about union values, from the CASEs that route it.
 *
 * A narrowing fact says which part of a union a value at a base path can be. A base path is
 * `<node>.output[.field...]` or `input.<name>[.field...]`. Facts come from CASE branches: an
 * `on:` CASE over a tag field of a union gives each `when:` edge the member with that tag and the
 * `else:` edge the unlisted members. A searched CASE gives each case's edge what its `when:`
 * proves true (`isinstance` checks, tag comparisons and null checks through `and`, `or` and
 * `not`); its `else:` edge carries none.
 *
 * Facts flow along the graph in topological order. A node's control edges join by union (it ran
 * because one of them fired), each required data group joins its producers by union, and every
 * contribution (control, required data groups, `depends_on`) meets by intersection. Each node
 * then gets a view of the producer and flow-input contracts with its narrowed bases rebuilt, so
 * the reference and type checks can read member fields inside a branch. A searched CASE's own
 * param whose every read is guarded by the checks before it in its `when:` gets a view narrowed
 * by what they prove too; a param read only after a check that can never be true is not checked,
 * since `checkImpossibleWhens` reports that check.
 *
 * The pass reads built, desugared CASE descriptors and their edges, so it runs after
 * `desugarCaseCallTargets` and the node builds.
 *
 * The fact algebra (`Members`, `Exact`, their union and intersection) lives in
 * `helpers/narrowing_facts.ts`; this module re-exports it.
 *
 * Knows about: descriptors, compiled edges and their topological order, output contracts, condition
 * guards, narrowing facts, and `typesys` unions.
 */

import { Edge } from "../../compile/model.ts";
import { topologicalOrder } from "../../compile/validation.ts";
import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode, Tree } from "../../expr/grammar.ts";
import { ExpressionError, parseCondition } from "../../expr/index.ts";
import { ELSE_HANDLE } from "../../nodes/base.ts";
import { isExactly, type OutputContract, PartialRecordContract, ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { jsTypeName, type Type, typeEquals, unionTagFields, ValueKind } from "../../typesys/values.ts";
import { guardChecks, guardedReads, proves } from "../helpers/condition_guards.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import {
  type BaseTypeOf,
  checkedRefs,
  type Env,
  Exact,
  type Fact,
  factType,
  guardEnv,
  impossibleCheckMessage,
  impossibleChecks,
  intersectEnvs,
  Members,
  membersType,
  NOT_NARROWABLE,
  unionEnvs,
} from "../helpers/narrowing_facts.ts";
import {
  combineOutputContracts,
  directReferencePathParts,
  projectOutputContract,
  recordFieldOrigins,
} from "../../compile/output_contracts.ts";
import type { NodeDescriptor } from "../nodes/base.ts";
import { CaseDescriptor } from "../nodes/case.ts";

export { Exact, intersectFacts, isinstanceFact, Members, unionFacts } from "../helpers/narrowing_facts.ts";
export type { Env, Fact } from "../helpers/narrowing_facts.ts";

/** The producer contracts and the flow-input contracts one node or param reads against. */
export type Views = readonly [
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
];

/** Per-node narrowing results for one flow; frozen. */
export class Narrowing {
  /** The unnarrowed producer contracts. */
  readonly producers: ReadonlyMap<string, OutputContract>;
  /** The unnarrowed flow-input contracts. */
  readonly inputs: ReadonlyMap<string, ValueContract>;
  /** Each ordered node's facts, keyed by base path. */
  readonly environments: ReadonlyMap<string, Env>;
  /** The narrowed contract maps of nodes that narrow something. */
  readonly views: ReadonlyMap<string, Views>;
  /** Nodes a topological order could not reach, on or after a cycle. They get no facts. */
  readonly unordered: ReadonlySet<string>;

  /** Build the results; every map and set left out is empty. */
  constructor(options: {
    producers: ReadonlyMap<string, OutputContract>;
    inputs: ReadonlyMap<string, ValueContract>;
    environments?: ReadonlyMap<string, Env>;
    views?: ReadonlyMap<string, Views>;
    unordered?: ReadonlySet<string>;
  }) {
    this.producers = options.producers;
    this.inputs = options.inputs;
    this.environments = options.environments ?? new Map();
    this.views = options.views ?? new Map();
    this.unordered = options.unordered ?? new Set();
    Object.freeze(this);
  }

  /**
   * Return the producer and flow-input contracts node `nodeId` sees.
   *
   * @param nodeId A node id. An unknown or unnarrowed node gets the unchanged maps.
   * @returns The producer contracts and the flow-input contracts.
   */
  forNode(nodeId: string): Views {
    return this.views.get(nodeId) ?? [this.producers, this.inputs];
  }

  /** Return node `nodeId`'s facts by base path; an empty map when it has none. */
  environment(nodeId: string): Env {
    return this.environments.get(nodeId) ?? new Map();
  }
}

// --------------------------------------------------------------------------- //
// Contract views
// --------------------------------------------------------------------------- //

/**
 * Replace the type at `path` through record fields, or `null` if unreachable. Exported for tests.
 *
 * When the narrowed value is never null, each record on the path is not null either and has each
 * step's field present, since a field read through a null or absent value gives null.
 *
 * @param typ The type to narrow inside.
 * @param path Field names from `typ` to the narrowed value. `[]` is `typ` itself.
 * @param narrowed The narrowed type.
 * @returns The narrowed type, or `null` when `path` names an absent field or crosses a union or a
 *   non-record.
 *
 * @example
 * Narrowing `Optional({b: Optional(str)})` at `["b"]` to `str` gives `{b: str}` with `b` required.
 */
export function narrowType(typ: Type, path: readonly string[], narrowed: Type): Type | null {
  if (path.length === 0) return narrowed;
  const [name, ...rest] = path as [string, ...string[]];
  // A union is never entered: rewriting its members would break member identity.
  if (typ.fields === undefined || typ.members !== undefined || !typ.fields.has(name)) return null;
  const inner = narrowType(typ.fields.get(name)!, rest, narrowed);
  if (inner === null) return null;
  const fields = new Map(typ.fields).set(name, inner);
  if (inner.nullable || inner.kind === ValueKind.NONE) return typ.replace({ fields });
  const required = new Set(typ.required ?? []).add(name);
  return typ.replace({ fields, required, nullable: false });
}

/**
 * Return `contract` with the type at `path` replaced by `typ`, keeping origins.
 *
 * The path is followed through record fields only; a `typ` that is never null makes each record
 * on the path non-null with the field below it present. An exact contract keeps its origins
 * unchanged, since narrowing does not move any path. Replacing a whole partial record gives an
 * exact contract with each proven field's origins rebased under its name; narrowing a partial
 * record's last opaque field gives the exact record.
 *
 * @param contract An exact or partial-record contract.
 * @param path Field names from the contract's root to the narrowed value. `[]` is the root.
 * @param typ The narrowed type.
 * @returns The narrowed contract, or `null` when `path` names an absent field, crosses a union or
 *   a non-record, or goes below an opaque field.
 * @throws TypeError when `contract` is neither an exact `ValueContract` nor an exact
 *   `PartialRecordContract`.
 *
 * @example
 * Narrowing `Union[Buy, Sell]` at `[]` to `Buy` gives a `Buy` contract with the same origins.
 */
export function narrowContractAt(contract: OutputContract, path: readonly string[], typ: Type): OutputContract | null {
  if (isExactly(contract, ValueContract)) {
    const narrowed = narrowType(contract.type, path, typ);
    if (narrowed === null) return null;
    return new ValueContract(narrowed, contract.origins);
  }
  if (!isExactly(contract, PartialRecordContract)) {
    throw new TypeError(
      `output contract must be an exact ValueContract or PartialRecordContract, got ${jsTypeName(contract)}`,
    );
  }
  if (path.length === 0) return new ValueContract(typ, recordFieldOrigins(contract.fields));
  const fields = new Map(contract.fields);
  const [name, ...rest] = path as [string, ...string[]];
  if (!fields.has(name)) return null;
  const fieldContract = fields.get(name)!;
  if (fieldContract === null) {
    if (rest.length > 0) return null;
    fields.set(name, new ValueContract(typ));
    return combineOutputContracts(fields);
  }
  const narrowedField = narrowContractAt(fieldContract, rest, typ);
  if (narrowedField === null) return null;
  // A proven field is a ValueContract, which narrows to a ValueContract.
  fields.set(name, narrowedField as ValueContract);
  return new PartialRecordContract([...fields]);
}

/** Split a base path into `["producer" | "input", root name, field path]`. */
function splitBase(base: string): [space: "producer" | "input", name: string, path: string[]] | null {
  const parts = base.split(".");
  if (parts.length >= 2 && parts[0] === "input") return ["input", parts[1]!, parts.slice(2)];
  if (parts.length >= 2 && parts[1] === "output") return ["producer", parts[0]!, parts.slice(2)];
  return null;
}

/** The contract at a base path, or `null` when it is unknown or unproven. */
function baseContract(
  base: string,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
): OutputContract | null {
  const split = splitBase(base);
  if (split === null) return null;
  const [space, name, path] = split;
  const root = (space === "producer" ? producers : inputs).get(name);
  if (root === undefined) return null;
  return projectOutputContract(root, path);
}

/**
 * Rebuild each narrowed base in the contract maps, outer bases first; the given maps themselves
 * when nothing narrows.
 */
function buildViews(
  env: Env,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
): Views {
  let producerView = producers;
  let inputView = inputs;
  const ordered: [string, NonNullable<ReturnType<typeof splitBase>>][] = [];
  for (const base of env.keys()) {
    const split = splitBase(base);
    if (split !== null) ordered.push([base, split]);
  }
  ordered.sort(([a, [, , pathA]], [b, [, , pathB]]) => pathA.length - pathB.length || compareCodeUnits(a, b));
  for (const [base, [space, name, path]] of ordered) {
    const target = factType(env.get(base)!);
    if (target === null) continue;
    const root = (space === "producer" ? producerView : inputView).get(name);
    if (root === undefined) continue;
    const current = projectOutputContract(root, path);
    if (isExactly(current, ValueContract) && typeEquals(current.type, target)) continue;
    const narrowed = narrowContractAt(root, path, target);
    if (narrowed === null) continue;
    if (space === "producer") {
      producerView = new Map(producerView).set(name, narrowed);
    } else {
      // A flow input is a ValueContract, which narrows to a ValueContract.
      inputView = new Map(inputView).set(name, narrowed as ValueContract);
    }
  }
  return [producerView, inputView];
}

// --------------------------------------------------------------------------- //
// CASE facts
// --------------------------------------------------------------------------- //

/** The union type of an exact contract, else `null`. */
function unionType(contract: OutputContract | null): Type | null {
  if (isExactly(contract, ValueContract) && contract.type.members !== undefined) return contract.type;
  return null;
}

/** The sorted tag values of a tagged union's tag field. */
function tags(union: Type, tagField: string): string[] {
  return union.members!.flatMap((member) => [...member.fields!.get(tagField)!.tags!]).sort(compareCodeUnits);
}

/** The fact that a value is one of `members`, relative to the base's own union. */
function membersFact(reference: Type | null, members: readonly Type[], nullable: boolean): Fact {
  if (members.length === 0) return new Members([], nullable);
  // The base has no union of its own outside this branch, so the set is only an exact type.
  if (reference === null) return new Exact(membersType(members, nullable)!);
  return new Members(members, nullable);
}

/**
 * The facts each handle of an `on:` CASE over a union tag field carries.
 *
 * A `when:` value that is not a tag gives its handle an empty member set;
 * `checkCaseWhenValues` reports it.
 */
function onCaseFacts(
  desc: CaseDescriptor,
  producersView: ReadonlyMap<string, OutputContract>,
  inputsView: ReadonlyMap<string, ValueContract>,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
): Map<string, Env> {
  const parts = directReferencePathParts(desc.on);
  if (parts === null || parts.length < 3) return new Map();
  const base = parts.slice(0, -1).join(".");
  const tagField = parts.at(-1)!;
  if (splitBase(base) === null) return new Map();
  // Members are taken from the base's own union, so facts from different CASEs on the same base
  // compare by member; the routing CASE's environment is met with them on each edge.
  let reference = unionType(baseContract(base, producers, inputs));
  if (reference !== null && !unionTagFields(reference).includes(tagField)) reference = null;
  const union = reference ?? unionType(baseContract(base, producersView, inputsView));
  if (union === null || !unionTagFields(union).includes(tagField)) return new Map();

  const tagValues = tags(union, tagField);
  const byHandle = new Map<string, Type[]>();
  const listed = new Set<string>();
  for (const spec of desc.cases) {
    // The `when:` value is the match value as read: a date is its text already.
    const value = spec["when"] ?? null;
    const then = spec["then"] ?? null;
    // A handle is a node name; any other `then:` routes no edge.
    if (value === null || typeof then !== "string") continue;
    let members = byHandle.get(then);
    if (members === undefined) byHandle.set(then, (members = []));
    if (typeof value === "string" && tagValues.includes(value)) {
      listed.add(value);
      members.push(...union.members!.filter((member) => member.fields!.get(tagField)!.tags!.has(value)));
    }
  }
  const facts = new Map<string, Env>();
  for (const [handle, members] of byHandle) {
    facts.set(handle, new Map([[base, membersFact(reference, members, false)]]));
  }
  if (desc.else_ !== null) {
    const unlisted = union.members!.filter(
      (member) => ![...member.fields!.get(tagField)!.tags!].some((tag) => listed.has(tag)),
    );
    facts.set(ELSE_HANDLE, new Map([[base, membersFact(reference, unlisted, union.nullable)]]));
  }
  return facts;
}

/**
 * The declared type of a base path at a CASE, for a check on it.
 *
 * Facts are taken against each base's unnarrowed type, so facts from different CASEs on the same
 * base compare by member. A base below a union, such as a field only some members have, takes its
 * type from `view`, the CASE node's routing view. A base that is not a node output or flow input
 * cannot be narrowed.
 */
function baseTypeOf(
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  view: Views,
): BaseTypeOf {
  return (base) => {
    if (splitBase(base) === null) return NOT_NARROWABLE;
    const contract = baseContract(base, producers, inputs) ?? baseContract(base, ...view);
    return isExactly(contract, ValueContract) ? contract.type : null;
  };
}

/** One read occurrence of a searched `when:`, with what it is name-checked against; frozen. */
export interface WhenRead {
  /** The read's refcall. */
  readonly node: Tree;
  /** The contract maps the read sees: the CASE's routing view, narrowed by the checks before it. */
  readonly views: Views;
  /** The facts behind `views`, by base path. */
  readonly facts: Env;
  /** The bases of `facts` that only the routing narrows, not the checks before the read. */
  readonly routed: ReadonlySet<string>;
}

/**
 * Return each read occurrence of a searched CASE's `when:` with the contracts it is name-checked
 * against; a read no value reaches (it follows a check `checkImpossibleWhens` reports) is left out.
 *
 * `and`, `or` and `not` short-circuit, so the checks before a read narrow what it sees, met with
 * the CASE node's own routing facts.
 *
 * @param narrowing The flow's narrowing results.
 * @param caseId The searched CASE.
 * @param tree Its `when:`, parsed.
 * @param registry The flow's typedefs.
 * @returns The reads, in source order.
 */
export function searchedWhenReads(
  narrowing: Narrowing,
  caseId: string,
  tree: ExprNode,
  registry: TypeRegistry,
): WhenRead[] {
  const { producers, inputs } = narrowing;
  const env = narrowing.environment(caseId);
  const view = narrowing.forNode(caseId);
  const typeOf = baseTypeOf(producers, inputs, view);
  const checked = checkedRefs(tree, typeOf, registry, { routing: env });
  const reads: WhenRead[] = [];
  guardedReads(tree).forEach((read, index) => {
    const guard = checked[index]![1];
    // A comprehension variable is no reference; the field-read checks check its reads.
    if (guard === null || read.node.data === "comp_var") return;
    const guarded = guardChecks(guard).length > 0 ? guardEnv(guard, typeOf, registry) : new Map();
    if (guarded.size === 0) {
      reads.push({ node: read.node, views: view, facts: new Map(), routed: new Set() });
      return;
    }
    const facts = intersectEnvs([env, guarded]);
    const routed = new Set([...facts.keys()].filter((base) => !guarded.has(base)));
    reads.push({ node: read.node, views: buildViews(facts, producers, inputs), facts, routed });
  });
  return reads;
}

/** A searched `when:` (or any condition) parsed, or `null` when it is no string or does not parse. */
export function parsedWhen(when: unknown): ExprNode | null {
  if (typeof when !== "string") return null;
  try {
    return parseCondition(when);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
}

/**
 * The facts each handle of a searched CASE carries: what its `when:` proves true.
 *
 * Cases routing to one target join by union. The `else:` handle carries no fact.
 */
function searchedCaseFacts(desc: CaseDescriptor, typeOf: BaseTypeOf, registry: TypeRegistry): Map<string, Env> {
  const byHandle = new Map<string, Env[]>();
  for (const spec of desc.cases) {
    const then = spec["then"] ?? null;
    // A handle is a node name; any other `then:` routes no edge.
    if (typeof then !== "string") continue;
    const tree = parsedWhen(spec["when"]);
    const env: Env = tree === null ? new Map() : guardEnv(proves(tree, true), typeOf, registry);
    let envs = byHandle.get(then);
    if (envs === undefined) byHandle.set(then, (envs = []));
    envs.push(env);
  }
  const facts = new Map<string, Env>();
  for (const [handle, envs] of byHandle) {
    const env = unionEnvs(envs);
    if (env.size > 0) facts.set(handle, env);
  }
  return facts;
}

/**
 * Reject a searched CASE `when:` with a check that can never be true.
 *
 * Each `when:` is checked with `impossibleChecks`, met with the facts of the branch that routes
 * its CASE node, so a check the routing rules out is reported too.
 *
 * @param descriptors The flow's node descriptors by id; only searched CASEs are checked.
 * @param producers Each node's output contract by id.
 * @param inputs The flow-input contracts by name.
 * @param narrowing The flow's narrowing, for each CASE node's routing facts and view.
 * @param options.registry The flow's `typedefs:` registry, for `isinstance` type names. `null`
 *   (the default) knows only the builtin names.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError when a `when:` has a check that can never be true, located at the first such
 *   `when:`; every such check in the flow is listed.
 *
 * @example
 * With `d` a `Union[Buy, Hold, Sell]`, the searched case `gate` with
 * `when: isinstance(d.output, Buy) and d.output.action == "hold"` fails with
 * ``case node 'gate': `cases[0].when`: `d.output.action == "hold"` can never be true here:
 * d.output is Buy, by the checks before it``.
 */
export function checkImpossibleWhens(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  narrowing: Narrowing,
  options: { registry?: TypeRegistry | null; diagnostics?: LoadDiagnostics | null } = {},
): void {
  const registry = options.registry ?? new Map();
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const problems: [message: string, range: SourceRange | null][] = [];
  for (const [nodeId, desc] of descriptors) {
    if (!(desc instanceof CaseDescriptor) || desc.on !== null) continue;
    const env = narrowing.environment(nodeId);
    const typeOf = baseTypeOf(producers, inputs, narrowing.forNode(nodeId));
    for (const [index, spec] of desc.cases.entries()) {
      const tree = parsedWhen(spec["when"]);
      if (tree === null) continue;
      const whenRange = desc.whenRanges[index] ?? null;
      for (const check of impossibleChecks(tree, typeOf, registry, { routing: env })) {
        problems.push([
          `case node ${reprText(nodeId)}: \`cases[${index}].when\`: ${impossibleCheckMessage(check)}`,
          whenRange ?? desc.range,
        ]);
      }
    }
  }
  if (problems.length > 0) {
    throw diagnostics.error(problems.map(([message]) => message).join("\n  "), { range: problems[0]![1] });
  }
}

// --------------------------------------------------------------------------- //
// Propagation
// --------------------------------------------------------------------------- //

/** The given edges plus each descriptor's ordering edges, between known nodes only. */
function knownEdges(descriptors: ReadonlyMap<string, NodeDescriptor>, edges: readonly Edge[]): Edge[] {
  const known = [...edges];
  for (const [nodeId, desc] of descriptors) {
    for (const source of desc.dependsOn) known.push(new Edge("", source, nodeId, { ordering: true, optional: false }));
    for (const source of desc.runsAfter) known.push(new Edge("", source, nodeId, { ordering: true, optional: true }));
  }
  return known.filter((edge) => descriptors.has(edge.from) && descriptors.has(edge.to));
}

/**
 * Compute each node's narrowing facts and contract views.
 *
 * A node's environment is the intersection of its contributions:
 *
 * - its control edges, joined by union, each giving the routing CASE's environment met with that
 *   edge's facts (an edge without a fact for a base leaves it unconstrained);
 * - each required data group, giving the union of its producers' environments; an optional group
 *   gives nothing;
 * - each required ordering edge (`depends_on`), giving its source's environment.
 *
 * A searched CASE's edge carries what its case's `when:` proves true, and its `else:` edge none.
 *
 * Edges and dependencies that name unknown nodes are ignored. Nodes on or after a cycle get no
 * facts and are reported in `unordered`.
 *
 * @param descriptors The flow's node descriptors by id. A `CaseDescriptor` produces facts;
 *   `dependsOn`/`runsAfter` give ordering edges.
 * @param edges The data edges plus each CASE's own data and control edges. Control edges carry the
 *   branch target, or the else handle for `else:`, as `sourceHandle`.
 * @param producers Each node's output contract by id.
 * @param inputs The flow-input contracts by name.
 * @param options.registry The flow's `typedefs:` registry, for `isinstance` type names. `null`
 *   (the default) knows only the builtin names.
 * @returns The environments, the views of narrowed nodes, and the unordered nodes.
 *
 * @example
 * With `d` a `Union[Buy, Sell]` and `c` an `on: ${d.output.action}` CASE routing `buy` to `b`,
 * `computeNarrowing(...).forNode("b")[0].get("d")` is a `Buy` contract.
 */
export function computeNarrowing(
  descriptors: ReadonlyMap<string, NodeDescriptor>,
  edges: readonly Edge[],
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  options: { registry?: TypeRegistry | null } = {},
): Narrowing {
  const registry = options.registry ?? new Map();
  const known = knownEdges(descriptors, edges);
  const { order } = topologicalOrder(descriptors.keys(), known);
  const incoming = new Map<string, Edge[]>();
  for (const nodeId of descriptors.keys()) incoming.set(nodeId, []);
  for (const edge of known) incoming.get(edge.to)!.push(edge);

  const environments = new Map<string, Env>();
  const views = new Map<string, Views>();
  const handleFacts = new Map<string, Map<string, Env>>();
  for (const nodeId of order) {
    const control: Env[] = [];
    const groups = new Map<string | null, [optional: boolean, members: Env[]]>();
    const contributions: Env[] = [];
    for (const edge of incoming.get(nodeId)!) {
      const sourceEnv = environments.get(edge.from) ?? new Map();
      if (edge.sourceHandle !== null) {
        const edgeFacts = handleFacts.get(edge.from)?.get(edge.sourceHandle) ?? new Map();
        control.push(intersectEnvs([sourceEnv, edgeFacts]));
      } else if (edge.ordering) {
        if (!edge.optional) contributions.push(sourceEnv);
      } else {
        // Like the runtime join, a group's first edge decides whether it is optional.
        let group = groups.get(edge.inputGroup);
        if (group === undefined) groups.set(edge.inputGroup, (group = [edge.optional, []]));
        group[1].push(sourceEnv);
      }
    }
    if (control.length > 0) contributions.push(unionEnvs(control));
    for (const [optional, members] of groups.values()) {
      if (!optional) contributions.push(unionEnvs(members));
    }
    const env = intersectEnvs(contributions);
    if (env.size > 0) {
      environments.set(nodeId, env);
      const view = buildViews(env, producers, inputs);
      if (view[0] !== producers || view[1] !== inputs) views.set(nodeId, view);
    }
    const desc = descriptors.get(nodeId)!;
    if (!(desc instanceof CaseDescriptor)) continue;
    const [producersView, inputsView] = views.get(nodeId) ?? [producers, inputs];
    if (desc.on !== null) {
      handleFacts.set(nodeId, onCaseFacts(desc, producersView, inputsView, producers, inputs));
      continue;
    }
    const typeOf = baseTypeOf(producers, inputs, [producersView, inputsView]);
    handleFacts.set(nodeId, searchedCaseFacts(desc, typeOf, registry));
  }

  const ordered = new Set(order);
  return new Narrowing({
    producers,
    inputs,
    environments,
    views,
    unordered: new Set([...descriptors.keys()].filter((nodeId) => !ordered.has(nodeId))),
  });
}
