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
 * Knows about: descriptors, compiled edges, output contracts, condition guards, narrowing facts,
 * and `typesys` unions.
 * Never imports: runtime.
 */

import { Edge } from "../../compile/model.ts";
import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import { ExpressionError, parseCondition } from "../../expr/index.ts";
import { ELSE_HANDLE } from "../../nodes/base.ts";
import { isExactly, type OutputContract, PartialRecordContract, ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { jsTypeName, type Type, typeEquals, unionTagFields, ValueKind } from "../../typesys/values.ts";
import { type Guard, guardChecks, proves } from "../helpers/condition_guards.ts";
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
} from "../helpers/output_contracts.ts";
import type { NodeDescriptor } from "../nodes/base.ts";
import { CaseDescriptor, type GuardedRead, searchedParamGuards } from "../nodes/case.ts";

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
  /**
   * Per searched CASE, the narrowed contract maps of its params whose every read is guarded by
   * checks before it in its `when:`.
   */
  readonly paramViews: ReadonlyMap<string, ReadonlyMap<string, Views>>;
  /**
   * Per searched CASE, the label and range to report a param at when its first read is guarded
   * and a later one isn't: the first unguarded read's `when:`.
   */
  readonly paramSites: ReadonlyMap<string, ReadonlyMap<string, readonly [label: string, range: SourceRange | null]>>;
  /** Per searched CASE, the facts each param in `paramViews` is read under. */
  readonly paramEnvs: ReadonlyMap<string, ReadonlyMap<string, Env>>;
  /**
   * Per searched CASE, the bases of each param's `paramEnvs` that only the routing narrows, not
   * the checks before the read.
   */
  readonly paramRouted: ReadonlyMap<string, ReadonlyMap<string, ReadonlySet<string>>>;
  /**
   * Per searched CASE, the params whose every read follows a check `checkImpossibleWhens`
   * reports, so no value reaches them.
   */
  readonly uncheckedParams: ReadonlyMap<string, ReadonlySet<string>>;
  /** Nodes a topological order could not reach, on or after a cycle. They get no facts. */
  readonly unordered: ReadonlySet<string>;

  /** Build the results; every map and set left out is empty. */
  constructor(options: {
    producers: ReadonlyMap<string, OutputContract>;
    inputs: ReadonlyMap<string, ValueContract>;
    environments?: ReadonlyMap<string, Env>;
    views?: ReadonlyMap<string, Views>;
    paramViews?: ReadonlyMap<string, ReadonlyMap<string, Views>>;
    paramSites?: ReadonlyMap<string, ReadonlyMap<string, readonly [label: string, range: SourceRange | null]>>;
    paramEnvs?: ReadonlyMap<string, ReadonlyMap<string, Env>>;
    paramRouted?: ReadonlyMap<string, ReadonlyMap<string, ReadonlySet<string>>>;
    uncheckedParams?: ReadonlyMap<string, ReadonlySet<string>>;
    unordered?: ReadonlySet<string>;
  }) {
    this.producers = options.producers;
    this.inputs = options.inputs;
    this.environments = options.environments ?? new Map();
    this.views = options.views ?? new Map();
    this.paramViews = options.paramViews ?? new Map();
    this.paramSites = options.paramSites ?? new Map();
    this.paramEnvs = options.paramEnvs ?? new Map();
    this.paramRouted = options.paramRouted ?? new Map();
    this.uncheckedParams = options.uncheckedParams ?? new Map();
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

  /**
   * Return the producer and flow-input contracts param `param` of `nodeId` reads against.
   *
   * @param nodeId A node id.
   * @param param One of the node's params. A searched CASE's guarded param gets its guarded view;
   *   any other param gets the node's view.
   * @returns The producer contracts and the flow-input contracts.
   */
  forParam(nodeId: string, param: string): Views {
    return this.paramViews.get(nodeId)?.get(param) ?? this.forNode(nodeId);
  }

  /**
   * Return the label and range to report param `param` of `nodeId` at, if moved.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The `cases[N].when` label and range of the param's first unguarded read when its
   *   first read is guarded and a later one isn't; otherwise `null`, and the param keeps the site
   *   of its first read.
   */
  paramSite(nodeId: string, param: string): readonly [label: string, range: SourceRange | null] | null {
    return this.paramSites.get(nodeId)?.get(param) ?? null;
  }

  /**
   * Return the facts a searched CASE's guarded param is read under; an empty map otherwise.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The facts behind `forParam`'s guarded view, by base path.
   */
  paramEnv(nodeId: string, param: string): Env {
    return this.paramEnvs.get(nodeId)?.get(param) ?? new Map();
  }

  /**
   * Return the bases of `paramEnv` that only the routing narrows.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The base paths whose facts come from the case branch that routes the node, not from
   *   the checks before the read.
   */
  paramRoutedBases(nodeId: string, param: string): ReadonlySet<string> {
    return this.paramRouted.get(nodeId)?.get(param) ?? new Set();
  }

  /**
   * Return whether no value reaches any read of a searched CASE's param.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns `true` when every read of the param follows a check that can never be true, which
   *   `checkImpossibleWhens` reports; its fields are then not checked.
   */
  paramUnchecked(nodeId: string, param: string): boolean {
    return this.uncheckedParams.get(nodeId)?.has(param) ?? false;
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
  if (typ.fields === undefined || typ.variants !== undefined || !typ.fields.has(name)) return null;
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
  if (isExactly(contract, ValueContract) && contract.type.variants !== undefined) return contract.type;
  return null;
}

/** The sorted tag values of a tagged union's tag field. */
function tags(union: Type, tagField: string): string[] {
  return union.variants!.flatMap((member) => [...member.fields!.get(tagField)!.tags!]).sort(compareCodeUnits);
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
      members.push(...union.variants!.filter((member) => member.fields!.get(tagField)!.tags!.has(value)));
    }
  }
  const facts = new Map<string, Env>();
  for (const [handle, members] of byHandle) {
    facts.set(handle, new Map([[base, membersFact(reference, members, false)]]));
  }
  if (desc.else_ !== null) {
    const unlisted = union.variants!.filter(
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

/** A searched `when:` parsed, or `null` when it is no string or does not parse. */
function parsedWhen(when: unknown): ExprNode | null {
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

/** Each searched CASE param's occurrences with the guard each is checked under. */
function checkedParamGuards(
  desc: CaseDescriptor,
  env: Env,
  typeOf: BaseTypeOf,
  registry: TypeRegistry,
): Map<string, GuardedRead[]> {
  const guardsOf = (tree: ExprNode): (Guard | null)[] =>
    checkedRefs(tree, typeOf, registry, { routing: env }).map(([, guard]) => guard);
  return searchedParamGuards(desc, guardsOf);
}

/**
 * The views and facts of a searched CASE's params whose every checked read is guarded.
 *
 * A param read under different guards gets the union of their facts, met with the CASE's own
 * environment. Occurrences that are not checked are left out. The third map holds each param's
 * bases that only the CASE's own environment (its routing) narrows.
 */
function guardedParamViews(
  params: ReadonlyMap<string, readonly GuardedRead[]>,
  env: Env,
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
  typeOf: BaseTypeOf,
  registry: TypeRegistry,
): [Map<string, Views>, Map<string, Env>, Map<string, Set<string>>] {
  const views = new Map<string, Views>();
  const envs = new Map<string, Env>();
  const routed = new Map<string, Set<string>>();
  for (const [param, occurrences] of params) {
    const guards = occurrences.flatMap(([, guard]) => (guard === null ? [] : [guard]));
    if (guards.length === 0 || !guards.every((guard) => guardChecks(guard).length > 0)) continue;
    const guarded = unionEnvs(guards.map((guard) => guardEnv(guard, typeOf, registry)));
    if (guarded.size === 0) continue;
    const met = intersectEnvs([env, guarded]);
    const view = buildViews(met, producers, inputs);
    if (view[0] !== producers || view[1] !== inputs) {
      views.set(param, view);
      envs.set(param, met);
      routed.set(param, new Set([...met.keys()].filter((base) => !guarded.has(base))));
    }
  }
  return [views, envs, routed];
}

/** The searched CASE params whose every read follows a reported impossible check. */
function uncheckedParams(params: ReadonlyMap<string, readonly GuardedRead[]>): Set<string> {
  const unchecked = new Set<string>();
  for (const [param, occurrences] of params) {
    if (occurrences.length > 0 && occurrences.every(([, guard]) => guard === null)) unchecked.add(param);
  }
  return unchecked;
}

/**
 * Where to report a searched CASE param whose first read is guarded and a later one isn't.
 *
 * The guarded read is fine, so an error on the param belongs to its first unguarded read: the
 * label and range of that read's `when:`. Reads that are not checked are left out.
 */
function unguardedReadSites(
  desc: CaseDescriptor,
  params: ReadonlyMap<string, readonly GuardedRead[]>,
): Map<string, readonly [label: string, range: SourceRange | null]> {
  const sites = new Map<string, readonly [string, SourceRange | null]>();
  for (const [param, occurrences] of params) {
    const checked = occurrences.flatMap(([index, guard]) => (guard === null ? [] : [[index, guard] as const]));
    if (checked.length === 0 || guardChecks(checked[0]![1]).length === 0) continue;
    const unguarded = checked.find(([, guard]) => guardChecks(guard).length === 0)?.[0];
    if (unguarded === undefined) continue;
    sites.set(param, [`cases[${unguarded}].when`, desc.whenRanges[unguarded] ?? null]);
  }
  return sites;
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

/** Kahn's order over `edges`, in descriptor order among ready nodes. */
function topologicalOrder(descriptors: ReadonlyMap<string, NodeDescriptor>, edges: readonly Edge[]): string[] {
  const indegree = new Map<string, number>();
  const successors = new Map<string, string[]>();
  for (const nodeId of descriptors.keys()) {
    indegree.set(nodeId, 0);
    successors.set(nodeId, []);
  }
  for (const edge of edges) {
    indegree.set(edge.to, indegree.get(edge.to)! + 1);
    successors.get(edge.from)!.push(edge.to);
  }
  const ready = [...indegree].filter(([, count]) => count === 0).map(([nodeId]) => nodeId);
  const order: string[] = [];
  // `ready` is a queue: nodes are taken from the front as they are appended at the back.
  for (let next = 0; next < ready.length; next++) {
    const nodeId = ready[next]!;
    order.push(nodeId);
    for (const successor of successors.get(nodeId)!) {
      const count = indegree.get(successor)! - 1;
      indegree.set(successor, count);
      if (count === 0) ready.push(successor);
    }
  }
  return order;
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
  const order = topologicalOrder(descriptors, known);
  const incoming = new Map<string, Edge[]>();
  for (const nodeId of descriptors.keys()) incoming.set(nodeId, []);
  for (const edge of known) incoming.get(edge.to)!.push(edge);

  const environments = new Map<string, Env>();
  const views = new Map<string, Views>();
  const paramViews = new Map<string, Map<string, Views>>();
  const paramSites = new Map<string, Map<string, readonly [string, SourceRange | null]>>();
  const paramEnvs = new Map<string, Map<string, Env>>();
  const paramRouted = new Map<string, Map<string, Set<string>>>();
  const unchecked = new Map<string, Set<string>>();
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
    const params = checkedParamGuards(desc, env, typeOf, registry);
    const [guarded, guardedEnvs, routed] = guardedParamViews(params, env, producers, inputs, typeOf, registry);
    if (guarded.size > 0) {
      paramViews.set(nodeId, guarded);
      paramEnvs.set(nodeId, guardedEnvs);
      paramRouted.set(nodeId, routed);
    }
    const uncheckedHere = uncheckedParams(params);
    if (uncheckedHere.size > 0) unchecked.set(nodeId, uncheckedHere);
    const sites = unguardedReadSites(desc, params);
    if (sites.size > 0) paramSites.set(nodeId, sites);
  }

  const ordered = new Set(order);
  return new Narrowing({
    producers,
    inputs,
    environments,
    views,
    paramViews,
    paramSites,
    paramEnvs,
    paramRouted,
    uncheckedParams: unchecked,
    unordered: new Set([...descriptors.keys()].filter((nodeId) => !ordered.has(nodeId))),
  });
}
