/**
 * Narrowing facts: which part of a union a value can be, and what checks prove about it.
 *
 * A fact says which part of its type a value at a base path can be: `Members` of the base's
 * union, or an `Exact` type it satisfies. Facts join by union (either holds) and meet by
 * intersection (both hold); losing a fact is always sound. The facts a check proves come from
 * `checkFact`: a true or false `isinstance`, a true or false comparison of a union's tag field
 * with a text, or a true null check. `guardEnv` evaluates a guard from `condition_guards.ts` into
 * the facts that hold whenever it does, `checkedRefs` gives the guard each read in a condition is
 * checked under, and `guardedReadTypes` the types a field read under a guard is checked against.
 * `impossibleChecks` finds the checks in a condition that are wrong whatever the values, and the
 * message helpers word those errors and a narrowed read's error.
 *
 * Knows about: `typesys` types and unions, `compile/validation`, `isinstance` type resolution,
 * condition guards, output-contract projection, and type labels.
 */

import { typesCompatible } from "../../compile/validation.ts";
import { compareCodeUnits, getCloseMatches, reprText } from "../../diagnostics.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import { isExactly, sameItems, ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import {
  canonicalUnionOrder,
  Type,
  TypeCheckError,
  typeEquals,
  typesOverlap,
  unionTagFields,
  ValueKind,
  valueRepr,
} from "../../typesys/values.ts";
import {
  AllOf,
  type Check,
  type CheckLiteral,
  checkText,
  conjuncts,
  falsified,
  type Guard,
  guardChecks,
  guardedRefs,
  Holds,
  IsinstanceCheck,
  literalsInContext,
  literalText,
  masked,
  NEVER,
  NOTHING,
  NotNullCheck,
  sameGuard,
  TagCheck,
  unknownCheck,
} from "./condition_guards.ts";
import { checkedMembers, isinstanceMembers, resolveIsinstanceType } from "./isinstance_types.ts";
import { projectOutputContract } from "../../compile/output_contracts.ts";
import { CLOSE_MATCH_CUTOFF, namedTypeLabel } from "./tags.ts";

/**
 * The value is one of `members`, all members of the base's union, or `null` if nullable; frozen.
 *
 * Members are deduplicated by `typeEquals` and kept in canonical union order. An empty set that
 * is not nullable means no value satisfies the facts, so the node never runs.
 */
export class Members {
  /** Non-nullable record members of the base's union. */
  readonly members: readonly Type[];
  /** Whether the value may also be `null`. */
  readonly nullable: boolean;

  constructor(members: readonly Type[], nullable = false) {
    const unique: Type[] = [];
    for (const member of members) {
      if (!unique.some((seen) => typeEquals(member, seen))) unique.push(member);
    }
    this.members = Object.freeze(canonicalUnionOrder(unique));
    this.nullable = nullable;
    Object.freeze(this);
  }
}

/** The value satisfies `type`; frozen. */
export class Exact {
  /** The type the value is known to satisfy. */
  readonly type: Type;

  constructor(type: Type) {
    this.type = type;
    Object.freeze(this);
  }
}

/** One narrowing fact for a base path. */
export type Fact = Members | Exact;

/** Facts keyed by dotted base path; an absent base is unconstrained. */
export type Env = Map<string, Fact>;

/** Whether `typ` equals one of `members`. */
function contains(members: readonly Type[], typ: Type): boolean {
  return members.some((member) => typeEquals(member, typ));
}

/** Whether two facts are equal: the same kind, with equal members and nullability, or equal types. */
function sameFact(a: Fact, b: Fact): boolean {
  if (a instanceof Exact) return b instanceof Exact && typeEquals(a.type, b.type);
  return b instanceof Members && a.nullable === b.nullable && sameItems(a.members, b.members, typeEquals);
}

/** Whether two environments have the same bases with equal facts, in any order. */
function sameEnv(a: Env, b: Env): boolean {
  if (a.size !== b.size) return false;
  for (const [base, fact] of a) {
    const other = b.get(base);
    if (other === undefined || !sameFact(fact, other)) return false;
  }
  return true;
}

/** Whether `fact` is an empty member set that is not nullable: no value has it. */
function isEmptyMembers(fact: Fact | null | undefined): boolean {
  return fact instanceof Members && fact.members.length === 0 && !fact.nullable;
}

/**
 * Return what holds when either fact holds, or `null` for no fact.
 *
 * @param a One fact; `null` is unconstrained.
 * @param b The other fact; `null` is unconstrained.
 * @returns The set union of two `Members`, an `Exact` both sides share, or `null`.
 *
 * @example
 * `unionFacts(new Members([Buy]), new Members([Sell]))` is `new Members([Buy, Sell])`;
 * `unionFacts(new Members([Buy]), null)` is `null`.
 */
export function unionFacts(a: Fact | null, b: Fact | null): Fact | null {
  if (a instanceof Members && b instanceof Members) {
    return new Members([...a.members, ...b.members], a.nullable || b.nullable);
  }
  if (a instanceof Exact && b instanceof Exact && typeEquals(a.type, b.type)) return a;
  return null;
}

/**
 * Return what holds when both facts hold, or `null` for no fact.
 *
 * Losing information is always sound, so a pair that cannot be combined gives `null`.
 *
 * @param a One fact; `null` is unconstrained.
 * @param b The other fact; `null` is unconstrained.
 * @returns The set intersection of two `Members`; `Members` of the one member an `Exact` names,
 *   else that `Exact`; of two `Exact`s, which both hold, so either is sound, the one whose type
 *   fits the other's (with null removed when the other rules it out); the other fact when one
 *   side is `null`; otherwise `null`.
 *
 * @example
 * `intersectFacts(new Members([Buy, Sell]), new Members([Sell, Hold]))` is
 * `new Members([Sell])`; `intersectFacts(new Exact(Buy), new Members([Buy, Sell]))` is
 * `new Members([Buy])`; `intersectFacts(new Exact(Any), new Exact(Buy))` is `new Exact(Buy)`,
 * and `intersectFacts(new Exact(Any), new Exact(Optional(Buy)))` is `new Exact(Buy)`.
 */
export function intersectFacts(a: Fact | null, b: Fact | null): Fact | null {
  if (a === null) return b;
  if (b === null) return a;
  if (a instanceof Members && b instanceof Members) {
    const kept = a.members.filter((member) => contains(b.members, member));
    return new Members(kept, a.nullable && b.nullable);
  }
  if (a instanceof Exact && b instanceof Exact) {
    // A non-null side rules out null, so the other holds without it.
    let [at, bt] = [a.type, b.type];
    if (at.nullable && !bt.nullable) at = at.replace({ nullable: false });
    else if (bt.nullable && !at.nullable) bt = bt.replace({ nullable: false });
    if (typesCompatible(at, bt)) return new Exact(at);
    return typesCompatible(bt, at) ? new Exact(bt) : null;
  }
  const [exact, members] = a instanceof Exact ? [a, b as Members] : [b as Exact, a];
  if (contains(members.members, exact.type)) return new Members([exact.type]);
  return exact;
}

/**
 * Return the fact that a value of `baseType` satisfies `typ`, or `null` for no fact.
 *
 * @param baseType The type of the value being tested.
 * @param typ The type it satisfies. A nullable `typ` also admits `null`.
 * @returns `null` when every `baseType` value already satisfies `typ`, since `typ` would only
 *   widen it (`object`, say, would turn off field checks); `Members` of the base union's members
 *   that `typ` (ignoring nullability) shares, when no other member could pass it; otherwise
 *   `new Exact(typ)`.
 *
 * @example
 * For `Union[Buy, Hold, Sell]`, `Buy` gives `new Members([Buy])`, `Union[Hold, Sell, Cancel]`
 * gives `new Members([Hold, Sell])`, an unrelated record gives `Exact` of that record, and
 * `object` gives `null`.
 */
export function isinstanceFact(baseType: Type, typ: Type): Fact | null {
  if (typesCompatible(baseType, typ)) return null;
  const kept = isinstanceMembers(baseType, typ);
  if (kept !== null) return new Members(kept, typ.nullable);
  return new Exact(typ);
}

/**
 * Return the fact that a value of `baseType` fails `isinstance(x, typ)`, or `null`.
 *
 * A base member equal to a member of `typ` (ignoring `typ`'s nullability) always passes the
 * check, so it is excluded. A member that only overlaps `typ`, or whose values `typ` would accept
 * with extra fields, may still fail, so it is kept.
 *
 * @param baseType The type of the value being tested.
 * @param typ The type the check names. A union names each of its members.
 * @returns `Members` of the base union's remaining members, nullable when the base is and `typ`
 *   is not; `null` when the base is no union or nothing is excluded.
 *
 * @example
 * For `Union[Buy, Hold, Sell]`, `Buy` gives `new Members([Hold, Sell])` and an unrelated record
 * gives `null`.
 */
export function notIsinstanceFact(baseType: Type, typ: Type): Fact | null {
  const members = baseType.members;
  if (members === undefined) return null;
  const wanted = checkedMembers(typ);
  const kept = members.filter((member) => !contains(wanted, member));
  const nullable = baseType.nullable && !typ.nullable;
  if (kept.length === members.length && nullable === baseType.nullable) return null;
  return new Members(kept, nullable);
}

/**
 * Return the fact that `<base>.<field> == value` is `holds`, or `null` for no fact.
 *
 * The members are chosen as an `on:` CASE's branches choose them.
 *
 * @param baseType The type of the value whose field is compared.
 * @param field The compared field.
 * @param value The text it is compared with.
 * @param holds Whether the field equals `value`.
 * @returns For a union whose `field` is a tag field: when `holds`, `Members` of the members whose
 *   tags contain `value`, not nullable (a null never equals a text); otherwise `Members` of the
 *   members whose tags are not exactly `{value}`, as nullable as the base. `null` for any other
 *   base or field.
 *
 * @example
 * For `Union[Buy, Hold, Sell]` tagged by `action`, `("action", "buy", true)` gives
 * `new Members([Buy])` and `("action", "buy", false)` gives `new Members([Hold, Sell])`.
 */
export function tagFact(baseType: Type, field: string, value: string, holds: boolean): Fact | null {
  const members = baseType.members;
  if (members === undefined || !unionTagFields(baseType).includes(field)) return null;
  // `unionTagFields` checked that every member has `field` with tags.
  const tagsOf = (member: Type): ReadonlySet<string> => member.fields!.get(field)!.tags!;
  if (holds) return new Members(members.filter((m) => tagsOf(m).has(value)));
  const kept = members.filter((m) => {
    const tags = tagsOf(m);
    return !(tags.size === 1 && tags.has(value));
  });
  return new Members(kept, baseType.nullable);
}

/**
 * Return the fact that a value of `baseType` is not null, or `null` for no fact.
 *
 * @param baseType The type of the value being tested.
 * @returns `Members` of every member, not nullable, for a nullable union; `Exact` of the type
 *   made non-nullable for any other nullable type (so `Optional(Any)` gives `Any`); `null` for a
 *   type that is not nullable, or is `null`'s own.
 *
 * @example
 * `notNullFact(Optional(str))` is `new Exact(str)`, and `notNullFact(Optional(Union[Buy, Sell]))`
 * is `new Members([Buy, Sell])`.
 */
export function notNullFact(baseType: Type): Fact | null {
  if (!baseType.nullable || baseType.kind === ValueKind.NONE) return null;
  if (baseType.members !== undefined) return new Members(baseType.members, false);
  return new Exact(baseType.replace({ nullable: false }));
}

/**
 * Return the fact a check proves about its base when it evaluates to `holds`.
 *
 * @param check The check.
 * @param baseType The base's declared type, or `null` when it is unproven.
 * @param holds What the check evaluated to.
 * @param registry The flow's typedefs, for an `isinstance` type name.
 * @returns `isinstanceFact` or `notIsinstanceFact` for an `isinstance`, `tagFact` for a tag
 *   comparison, `notNullFact` for a true null check and `null` for a false one (no fact says a
 *   value is null). On an unproven base a true `isinstance` gives `Exact` of its type and
 *   anything else gives `null`.
 * @throws TypeCheckError when an `isinstance` type name does not resolve.
 *
 * @example
 * `checkFact(new IsinstanceCheck("v", "Buy"), null, true, registry)` is `new Exact(Buy)`.
 */
export function checkFact(check: Check, baseType: Type | null, holds: boolean, registry: TypeRegistry): Fact | null {
  if (check instanceof IsinstanceCheck) {
    const typ = resolveIsinstanceType(check.typeName, registry);
    if (baseType === null) return holds ? new Exact(typ) : null;
    if (holds) return isinstanceFact(baseType, typ);
    return notIsinstanceFact(baseType, typ);
  }
  if (check instanceof TagCheck) {
    if (baseType === null) return null;
    return tagFact(baseType, check.field, check.value, holds);
  }
  if (check instanceof NotNullCheck) {
    if (baseType === null || !holds) return null;
    return notNullFact(baseType);
  }
  throw unknownCheck(check);
}

/**
 * Join environments by per-base union; a base missing from any is dropped.
 *
 * @param envs The environments. `[]` gives an empty environment.
 * @returns What holds when any one of them holds.
 */
export function unionEnvs(envs: readonly Env[]): Env {
  const joined: Env = new Map();
  if (envs.length === 0) return joined;
  for (const [base, first] of envs[0]!) {
    let fact: Fact | null = first;
    for (const env of envs.slice(1)) {
      fact = unionFacts(fact, env.get(base) ?? null);
      if (fact === null) break;
    }
    if (fact !== null) joined.set(base, fact);
  }
  return joined;
}

/**
 * Meet environments by per-base intersection.
 *
 * @param envs The environments.
 * @returns What holds when all of them hold.
 */
export function intersectEnvs(envs: readonly Env[]): Env {
  const met: Env = new Map();
  for (const env of envs) {
    for (const [base, fact] of env) {
      const combined = intersectFacts(met.get(base) ?? null, fact);
      if (combined === null) met.delete(base);
      else met.set(base, combined);
    }
  }
  return met;
}

/**
 * Return the type of a value that is one of `members`, or `null` if no value fits.
 *
 * @param members Record members of one union.
 * @param nullable Whether the value may also be `null`.
 * @returns The lone member, a union of the members, `null`'s type for a nullable empty set, or
 *   `null` for an empty set that is not nullable: no value fits.
 */
export function membersType(members: readonly Type[], nullable: boolean): Type | null {
  if (members.length === 0) return nullable ? Type.scalar(ValueKind.NONE) : null;
  if (members.length === 1) return members[0]!.replace({ nullable });
  return new Type({ kind: ValueKind.OBJECT, members: canonicalUnionOrder(members), nullable });
}

/**
 * Return whichever of two types admits every value of the other, or `null`.
 *
 * @param a One type; preferred when each admits the other's values.
 * @param b The other type.
 * @returns `a` when every `b` value fits it, else `b` when every `a` value fits it, else `null`:
 *   no type both name holds every value of either.
 *
 * @example
 * `widerType(Done, Optional(Done))` is `Optional(Done)`; `widerType(Done, null's type)` is
 * `null`.
 */
export function widerType(a: Type, b: Type): Type | null {
  if (typesCompatible(b, a)) return a;
  if (typesCompatible(a, b)) return b;
  return null;
}

/**
 * Return the type a value has under `fact`, or `null` when no value can.
 *
 * An empty member set on a nullable base can only be `null`; on a non-nullable base no value
 * fits, so there is no type.
 *
 * @param fact The fact.
 * @returns The type, or `null` for an empty member set that is not nullable.
 */
export function factType(fact: Fact): Type | null {
  if (fact instanceof Exact) return fact.type;
  return membersType(fact.members, fact.nullable);
}

// --------------------------------------------------------------------------- //
// Guards into facts
// --------------------------------------------------------------------------- //

/** A base the site cannot narrow, such as a path that is no pool path at a CASE. */
export const NOT_NARROWABLE: unique symbol = Symbol("NOT_NARROWABLE");

/** The answer of a `BaseTypeOf` for a base the site cannot narrow. */
export type NotNarrowable = typeof NOT_NARROWABLE;

/**
 * A site's declared type of a dotted base path: a `Type` when proven, `null` when unproven, or
 * `NOT_NARROWABLE` for a base the site cannot narrow.
 */
export type BaseTypeOf = (base: string) => Type | null | NotNarrowable;

/**
 * A narrowed field read: `[narrowed type, field names below it, narrowed base path]`.
 *
 * @example
 * `[Buy, ["leg", "price"], "v"]` for the read `v.leg.price` with `v` narrowed to `Buy`.
 */
export type NarrowedRead = readonly [Type, readonly string[], string];

/**
 * The type a check on `base` is taken against.
 *
 * A base whose declared type is unproven, such as a field only some members of an outer union
 * have, takes its type through the facts already proven for the longest outer base it is below.
 */
function baseTypeUnder(base: string, baseTypeOf: BaseTypeOf, outer: Env): Type | null | NotNarrowable {
  const declared = baseTypeOf(base);
  if (declared !== null) return declared;
  const parts = base.split(".");
  for (let cut = parts.length - 1; cut > 0; cut--) {
    const fact = outer.get(parts.slice(0, cut).join("."));
    if (fact === undefined) continue;
    const typ = factType(fact);
    if (typ === null) return null;
    const projected = projectOutputContract(new ValueContract(typ), parts.slice(cut));
    return isExactly(projected, ValueContract) ? projected.type : null;
  }
  return null;
}

/**
 * What `guardEnvUnder` has worked out for one `baseTypeOf` and registry, by guard identity.
 *
 * `proves` shares a sub-guard between the guards that contain it, so a guard is a small graph
 * whose tree grows exponentially with alternating `and`/`or` nesting; walking each sub-guard
 * once keeps the work linear in the graph.
 */
interface GuardEnvMemo {
  /** Each guard's outer bases: the bases whose facts in `outer` its facts can depend on. */
  readonly bases: Map<Guard, readonly string[]>;
  /** Each guard's facts, per `outer` seen, keyed by that `outer`'s facts for its outer bases. */
  readonly envs: Map<Guard, [readonly (Fact | undefined)[], Env][]>;
}

/** Return a new, empty `GuardEnvMemo`. */
function guardEnvMemo(): GuardEnvMemo {
  return { bases: new Map(), envs: new Map() };
}

/**
 * The bases whose facts in `outer` can change what `guard` proves: each strict prefix of a base
 * `baseTypeOf` leaves unproven, which takes its type through `outer` (`baseTypeUnder`).
 */
function outerBases(guard: Guard, baseTypeOf: BaseTypeOf, memo: GuardEnvMemo): readonly string[] {
  const known = memo.bases.get(guard);
  if (known !== undefined) return known;
  const found = new Set<string>();
  if (guard instanceof Holds) {
    const base = guard.check.base;
    if (baseTypeOf(base) === null) {
      const parts = base.split(".");
      for (let cut = 1; cut < parts.length; cut++) found.add(parts.slice(0, cut).join("."));
    }
  } else {
    for (const part of guard.parts) for (const base of outerBases(part, baseTypeOf, memo)) found.add(base);
  }
  const bases = [...found];
  memo.bases.set(guard, bases);
  return bases;
}

/** Whether two optional facts are both absent, or both present and equal. */
function sameOptionalFact(a: Fact | undefined, b: Fact | undefined): boolean {
  if (a === b) return true;
  return a !== undefined && b !== undefined && sameFact(a, b);
}

/** The facts `guard` proves, given that the facts in `outer` already hold; never mutate the result. */
function guardEnvUnder(
  guard: Guard,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  outer: Env,
  memo: GuardEnvMemo,
): Env {
  const key = outerBases(guard, baseTypeOf, memo).map((base) => outer.get(base));
  let seen = memo.envs.get(guard);
  if (seen === undefined) {
    seen = [];
    memo.envs.set(guard, seen);
  }
  const hit = seen.find(([facts]) => facts.every((fact, i) => sameOptionalFact(fact, key[i])));
  if (hit !== undefined) return hit[1];
  const env = guardEnvUncached(guard, baseTypeOf, registry, outer, memo);
  seen.push([key, env]);
  return env;
}

/** `guardEnvUnder` for a guard and `outer` not in `memo` yet. */
function guardEnvUncached(
  guard: Guard,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  outer: Env,
  memo: GuardEnvMemo,
): Env {
  if (guard instanceof Holds) {
    const base = guard.check.base;
    const typ = baseTypeUnder(base, baseTypeOf, outer);
    if (typ === NOT_NARROWABLE) return new Map();
    let fact: Fact | null;
    try {
      fact = checkFact(guard.check, typ, guard.value, registry);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      // The type-table pass reports the unknown name.
      return new Map();
    }
    return fact === null ? new Map() : new Map([[base, fact]]);
  }
  if (guard instanceof AllOf) {
    let env: Env = new Map();
    for (const part of guard.parts) {
      const partEnv = guardEnvUnder(part, baseTypeOf, registry, intersectEnvs([outer, env]), memo);
      env = intersectEnvs([env, partEnv]);
    }
    return env;
  }
  return unionEnvs(guard.parts.map((part) => guardEnvUnder(part, baseTypeOf, registry, outer, memo)));
}

/**
 * Return the facts that hold whenever `guard` holds.
 *
 * `Holds` gives its check's fact (`checkFact`), `AllOf` meets its parts and `AnyOf` joins them.
 * Facts are taken against each base's declared type, so facts from different checks compare by
 * member. The parts of an `AllOf` are taken left to right: a base whose declared type is unproven
 * because it sits below a union (a field only some members have) takes its type through the facts
 * the parts before it proved for an outer base, in nested parts too. A check whose base
 * `baseTypeOf` refuses, or whose type name does not resolve, proves nothing.
 *
 * @param guard A guard from `condition_guards.ts`.
 * @param baseTypeOf The site's declared type of a base path: a `Type`, `null` when unproven (a
 *   true `isinstance` then gives `Exact` of its type), or `NOT_NARROWABLE`.
 * @param registry The flow's typedefs, for `isinstance` type names.
 * @returns The facts by base path.
 *
 * @example
 * With `d.output` a `Union[Buy, Hold, Sell]` and `leg` only on `Buy`, the guard of
 * `d.output.action == "buy" and d.output.leg.type == "limit"` gives
 * `{"d.output": new Members([Buy]), "d.output.leg": new Members([Limit])}`.
 */
export function guardEnv(guard: Guard, baseTypeOf: BaseTypeOf, registry: TypeRegistry): Env {
  return guardEnvUnder(guard, baseTypeOf, registry, new Map(), guardEnvMemo());
}

/**
 * Return whether no value can satisfy the facts in `env`.
 *
 * @param env Facts by base path.
 * @returns `true` when some base's fact is an empty member set that is not nullable.
 */
export function envUnsatisfiable(env: Env): boolean {
  return [...env.values()].some(isEmptyMembers);
}

/**
 * Return whether an `isinstance` in `guard` names a type that does not resolve.
 *
 * @param guard A guard from `condition_guards.ts`.
 * @param registry The flow's typedefs.
 * @returns `true` when some `isinstance` type name is neither a typedef nor a builtin name. A
 *   site then leaves the read alone, so the type-table pass reports the name.
 */
export function unresolvedType(guard: Guard, registry: TypeRegistry): boolean {
  for (const check of guardChecks(guard)) {
    if (check instanceof TagCheck || check instanceof NotNullCheck) continue;
    if (!(check instanceof IsinstanceCheck)) throw unknownCheck(check);
    try {
      resolveIsinstanceType(check.typeName, registry);
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      return true;
    }
  }
  return false;
}

/**
 * Return the distinct facts of `guard`, and of each of its top-level conjuncts on the bases the
 * guard's meet lost: the candidates a read under `guard` is checked against, as for
 * `guardedReadTypes`, by base path.
 *
 * A meet drops a base whose facts can't be combined (two unrelated exact types), and a conjunct's
 * candidate keeps its fact there. Elsewhere the candidate keeps the meet's facts, which are at
 * least as narrow: a conjunct alone would forget the others (in `x != null and isinstance(x, T)`,
 * the null check alone says nothing of `T`).
 */
export function candidateEnvs(guard: Guard, baseTypeOf: BaseTypeOf, registry: TypeRegistry): Env[] {
  return candidateEnvsWith(guard, baseTypeOf, registry, guardEnvMemo());
}

/** `candidateEnvs`, reusing and adding to `memo`, which holds facts for `baseTypeOf` and `registry`. */
function candidateEnvsWith(guard: Guard, baseTypeOf: BaseTypeOf, registry: TypeRegistry, memo: GuardEnvMemo): Env[] {
  const met = guardEnvUnder(guard, baseTypeOf, registry, new Map(), memo);
  const envs: Env[] = [met];
  for (const part of conjuncts(guard)) {
    const env = new Map(met);
    for (const [base, fact] of guardEnvUnder(part, baseTypeOf, registry, new Map(), memo)) {
      if (!env.has(base)) env.set(base, fact);
    }
    if (!envs.some((seen) => sameEnv(seen, env))) envs.push(env);
  }
  return envs;
}

/**
 * The check sub-trees of the literals whose value the known type and the checks before them
 * decide.
 *
 * The known type is the declared type met with `routing`. Such a check, always true or always
 * false where it is evaluated, is no error by itself.
 */
function decidedLiterals(
  tree: ExprNode,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  routing: Env,
  memo: GuardEnvMemo,
): ExprNode[] {
  const decided: ExprNode[] = [];
  for (const literal of literalsInContext(tree)) {
    const contextEnv = intersectEnvs([routing, guardEnvUnder(literal.context, baseTypeOf, registry, new Map(), memo)]);
    if (envUnsatisfiable(contextEnv)) continue;
    const base = literal.check.base;
    const typ = baseTypeUnder(base, baseTypeOf, contextEnv);
    if (!(typ instanceof Type)) continue;
    for (const value of [true, false]) {
      let fact: Fact | null;
      try {
        fact = checkFact(literal.check, typ, value, registry);
      } catch (error) {
        if (!(error instanceof TypeCheckError)) throw error;
        break;
      }
      if (fact !== null && isEmptyMembers(intersectFacts(contextEnv.get(base) ?? null, fact))) {
        // `literalsInContext` gives every literal its sub-tree.
        decided.push(literal.node!);
        break;
      }
    }
  }
  return decided;
}

/** Whether some value satisfies `guard`'s facts and each top-level conjunct's, with `routing`. */
function satisfiable(
  guard: Guard,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  routing: Env,
  memo: GuardEnvMemo,
): boolean {
  const envs = candidateEnvsWith(guard, baseTypeOf, registry, memo);
  return !envs.some((env) => envUnsatisfiable(intersectEnvs([routing, env])));
}

/**
 * Return each reference occurrence in a condition with the guard its field read is checked under.
 *
 * A read that runs only when a check `impossibleChecks` reports has the value it can never have
 * is not checked (`null`), whatever the report's reason: the report explains it, and errors on
 * the code it makes dead would only repeat it. So is a read no value reaches because of a
 * reported check. Otherwise a guard some value satisfies is kept. A read no value reaches only
 * because an unreported check the known type or the checks before it always decide went the
 * other way (`isinstance(v, Decision) or v.nosuch > 0` with `v` declared `Decision`,
 * `isinstance(v, Buy) and (isinstance(v, Buy) or v.qty > 0)`) is no error by itself: such checks
 * prove nothing here, so the read is checked as without them (with `NOTHING` when the guard still
 * can't hold).
 *
 * @param tree A condition parsed with `parseCondition`, in any spelling.
 * @param baseTypeOf The site's declared type of a base path, as for `guardEnv`.
 * @param registry The flow's typedefs.
 * @param options.routing The facts that hold wherever the condition is evaluated, such as a CASE
 *   node's routing facts; with the declared type they make the known type. `null` (the default),
 *   or facts no value satisfies, is none.
 * @returns One `[reference path, guard or null]` pair per occurrence, in `guardedRefs` order.
 *
 * @example
 * With `v` a `Union[Buy, Hold, Sell]`, the `v.qty` read in `isinstance(v, Decision) or v.qty > 0`
 * gets `NOTHING`, in `isinstance(v, Buy) and (isinstance(v, Buy) or v.qty > 0)` the guard of
 * `isinstance(v, Buy)`, and in `isinstance(v, Buy) and v.action == 'hold' and v.qty > 0` and
 * `isinstance(v, Text) and v.qty > 0` gets `null`.
 */
export function checkedRefs(
  tree: ExprNode,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  options: { routing?: Env | null } = {},
): [string, Guard | null][] {
  const { routing = null } = options;
  const known: Env = routing !== null && !envUnsatisfiable(routing) ? routing : new Map();
  const refs = guardedRefs(tree);
  // Each read's guard shares sub-guards with the others', so their facts are worked out once.
  const memo = guardEnvMemo();
  const reported = reportedLiterals(tree, baseTypeOf, registry, known, memo);
  // `literalsInContext` gives every literal its sub-tree.
  const reportedNodes = reported.map(([literal]) => literal.node!);
  let unreported = refs;
  let needing = refs;
  if (reported.length > 0) {
    unreported = guardedRefs(masked(tree, reportedNodes));
    const never = reported.map(([literal, found]) => [literal.node!, neverValue(literal, found)] as const);
    needing = guardedRefs(falsified(tree, never));
  }
  if (unreported.length !== refs.length || needing.length !== refs.length) {
    throw new Error("masked and falsified copies of a condition must keep its reads");
  }
  let undecided: [string, Guard][] | null = null;
  const checked: [string, Guard | null][] = [];
  for (const [index, [ref, guard]] of refs.entries()) {
    const withoutReported = unreported[index]![1];
    const live = needing[index]![1];
    if (sameGuard(live, NEVER)) {
      checked.push([ref, null]);
    } else if (satisfiable(guard, baseTypeOf, registry, known, memo)) {
      checked.push([ref, guard]);
    } else if (satisfiable(withoutReported, baseTypeOf, registry, known, memo)) {
      checked.push([ref, null]);
    } else {
      if (undecided === null) {
        const decided = decidedLiterals(tree, baseTypeOf, registry, known, memo);
        undecided = guardedRefs(masked(tree, [...reportedNodes, ...decided]));
      }
      const trimmed = undecided[index]![1];
      checked.push([ref, satisfiable(trimmed, baseTypeOf, registry, known, memo) ? trimmed : NOTHING]);
    }
  }
  return checked;
}

/**
 * Return the value a reported literal's check sub-tree (`v.action != 'x'`, without any `not`
 * around it) never has; exported for tests; validators call `impossibleChecks`/`checkedRefs`.
 *
 * A text that is no tag never equals the tag field, whichever way the literal compares it; any
 * other reported literal can never be true.
 *
 * @param literal A literal `impossibleChecks` reports.
 * @param impossible Its report.
 * @returns The value the sub-tree never has.
 */
export function neverValue(literal: CheckLiteral, impossible: ImpossibleCheck): boolean {
  const check = literal.check;
  let checkNever: boolean;
  if (check instanceof TagCheck) {
    checkNever = impossible.reason === ImpossibleReason.NOT_A_TAG ? true : literal.value;
  } else if (check instanceof IsinstanceCheck || check instanceof NotNullCheck) {
    checkNever = literal.value;
  } else {
    throw unknownCheck(check);
  }
  // `!=` spells the check negatively: the sub-tree is true when the check is false.
  const positive = literal.value !== literal.negated;
  return checkNever === positive;
}

/**
 * Return the narrowed type a field read sees under `env`, or `null` when none applies.
 *
 * @param read The dotted reference path read, such as `"v.qty"`.
 * @param env Facts by base path.
 * @returns `[narrowed type, field names below it, narrowed base path]` for the longest base that
 *   is a strict prefix of `read`; `null` when there is no such base or no value can have its
 *   fact.
 *
 * @example
 * `envReadType("v.leg.price", new Map([["v", new Members([Buy])]]))` is
 * `[Buy, ["leg", "price"], "v"]`.
 */
export function envReadType(read: string, env: Env): NarrowedRead | null {
  const parts = read.split(".");
  let longest: string[] | null = null;
  for (const base of env.keys()) {
    const baseParts = base.split(".");
    if (
      baseParts.length < parts.length &&
      baseParts.every((part, i) => part === parts[i]) &&
      (longest === null || baseParts.length > longest.length)
    ) {
      longest = baseParts;
    }
  }
  if (longest === null) return null;
  const base = longest.join(".");
  const typ = factType(env.get(base)!);
  if (typ === null) return null;
  return [typ, parts.slice(longest.length), base];
}

/** Whether two narrowed reads are equal: equal types, the same fields and the same base. */
function sameRead(a: NarrowedRead, b: NarrowedRead): boolean {
  return typeEquals(a[0], b[0]) && sameItems(a[1], b[1]) && a[2] === b[2];
}

/**
 * Return the narrowed types a field read under `guard` is checked against.
 *
 * Each of the guard's own facts and the facts of each of its top-level conjuncts gives a
 * candidate (`envReadType`). A conjunct's candidate keeps what the whole guard's meet can lose
 * (two unrelated exact types). The read is fine when any candidate accepts it. Take the guard
 * from `checkedRefs`, which also says which reads are not checked at all.
 *
 * @param read The dotted reference path read, such as `"v.qty"`.
 * @param guard The guard the read is checked under.
 * @param baseTypeOf The site's declared type of a base path, as for `guardEnv`.
 * @param registry The flow's typedefs.
 * @returns `[narrowed type, field names below it, narrowed base path]` per distinct candidate, in
 *   order; `[]` when no fact applies to the read, so it is checked unnarrowed.
 *
 * @example
 * With `v` a `Union[Buy, Hold, Sell]`, the read `v.qty` under the guard of
 * `isinstance(v, Buy)` gives `[[Buy, ["qty"], "v"]]`.
 */
export function guardedReadTypes(
  read: string,
  guard: Guard,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
): NarrowedRead[] {
  const candidates: NarrowedRead[] = [];
  for (const env of candidateEnvs(guard, baseTypeOf, registry)) {
    const candidate = envReadType(read, env);
    if (candidate !== null && !candidates.some((seen) => sameRead(seen, candidate))) candidates.push(candidate);
  }
  return candidates;
}

// --------------------------------------------------------------------------- //
// Impossible checks and messages
// --------------------------------------------------------------------------- //

/** Why `impossibleChecks` reports a check. */
export const ImpossibleReason = Object.freeze({
  /** The checks evaluated before the literal rule it out. */
  CONTEXT: "context",
  /** The case branch that routes the node rules the literal out on its own. */
  ROUTING: "routing",
  /** An `isinstance` names a type no value of the base's type can be. */
  NO_OVERLAP: "no overlap",
  /** A negated `isinstance` names a type every value of the base's type is. */
  ALWAYS: "always",
  /**
   * A tag comparison's text is not a tag of the base union's tag field: a misspelling, reported
   * for `==` (never true) and `!=` (always true) alike.
   */
  NOT_A_TAG: "not a tag",
} as const);

export type ImpossibleReason = (typeof ImpossibleReason)[keyof typeof ImpossibleReason];

/** A check in a condition that is wrong whatever the values; frozen. */
export class ImpossibleCheck {
  /** The literal's canonical source, such as `not isinstance(v, Buy)`. */
  readonly text: string;
  /** The checked base path. */
  readonly base: string;
  /** Why it is reported. */
  readonly reason: ImpossibleReason;
  /** The base's type: narrowed for `CONTEXT` and `ROUTING`, else the type the base is checked against. */
  readonly baseLabel: string;
  /**
   * For `NO_OVERLAP` and `ALWAYS`, whether the base's type is its declared type; `false` when it
   * comes from the checks before the literal.
   */
  readonly declared: boolean;
  /** The type an `isinstance` names, for `NO_OVERLAP`. */
  readonly typeLabel: string;
  /**
   * The canonical source of the check that is always true where the literal needs it false, such
   * as `isinstance(v, Buy)` for `not isinstance(v, Buy)`; `""` when the literal needs it true.
   */
  readonly alwaysTrue: string;
  /** The compared text, for `NOT_A_TAG`. */
  readonly value: string;
  /** The compared field, for `NOT_A_TAG`. */
  readonly field: string;
  /** The tag field's tags, sorted, for `NOT_A_TAG`. */
  readonly tags: readonly string[];

  constructor(
    text: string,
    base: string,
    reason: ImpossibleReason,
    options: {
      baseLabel?: string;
      declared?: boolean;
      typeLabel?: string;
      alwaysTrue?: string;
      value?: string;
      field?: string;
      tags?: readonly string[];
    } = {},
  ) {
    const { baseLabel = "", declared = true, typeLabel = "", alwaysTrue = "", value = "", field = "", tags = [] } =
      options;
    this.text = text;
    this.base = base;
    this.reason = reason;
    this.baseLabel = baseLabel;
    this.declared = declared;
    this.typeLabel = typeLabel;
    this.alwaysTrue = alwaysTrue;
    this.value = value;
    this.field = field;
    this.tags = Object.freeze([...tags]);
    Object.freeze(this);
  }
}

/**
 * Return a narrowed type's label for a message, naming a union member by member.
 *
 * @param typ A narrowed type, such as a `factType` result.
 * @param registry The flow's typedefs.
 * @returns A union's members by typedef name joined by ` | ` (`Buy | Sell`), inside
 *   `Optional(...)` when it is nullable; any other type by `namedTypeLabel`, which falls back to
 *   the structural label for an anonymous type.
 */
export function narrowedTypeLabel(typ: Type, registry: TypeRegistry): string {
  if (typ.members === undefined) return namedTypeLabel(typ, registry);
  const label = typ.members.map((member) => namedTypeLabel(member, registry)).join(" | ");
  return typ.nullable ? `Optional(${label})` : label;
}

/** The label of the type a base has under `fact`, else of `fallback`. */
function factLabel(fact: Fact | null, fallback: Type | null, registry: TypeRegistry): string {
  const typ = (fact !== null ? factType(fact) : null) ?? fallback;
  return typ !== null ? narrowedTypeLabel(typ, registry) : "unknown";
}

/** Whether every value fits `typ`'s kind: `Any`, or a plain `object`. */
function isOpen(typ: Type): boolean {
  if (typ.kind === ValueKind.ANY) return true;
  return typ.kind === ValueKind.OBJECT && typ.fields === undefined && typ.members === undefined;
}

/** The check that is always true when `literal` can never be true, if it needs it false. */
function alwaysTrue(literal: CheckLiteral): string {
  return literal.value ? "" : checkText(literal.check, true);
}

/**
 * Return the report a literal gets against its base's type alone, else `null`; exported for
 * tests; validators call `impossibleChecks`/`checkedRefs`.
 *
 * @param literal A literal from `literalsInContext`.
 * @param baseType The type its base is checked against.
 * @param declared Whether `baseType` is the base's declared type.
 * @param registry The flow's typedefs.
 * @returns A `NOT_A_TAG`, `NO_OVERLAP` or `ALWAYS` report, or `null`.
 */
export function declaredImpossible(
  literal: CheckLiteral,
  baseType: Type,
  declared: boolean,
  registry: TypeRegistry,
): ImpossibleCheck | null {
  const check = literal.check;
  if (check instanceof NotNullCheck) {
    // Testing a value that is never null for null is ordinary defensive code.
    return null;
  }
  // The labels are built only for a report: a typedef's label parses and resolves its type.
  if (check instanceof TagCheck) {
    const members = baseType.members;
    if (members === undefined || !unionTagFields(baseType).includes(check.field)) return null;
    // `unionTagFields` checked that every member has the field with tags.
    const tags = members.flatMap((member) => [...member.fields!.get(check.field)!.tags!]).sort(compareCodeUnits);
    if (tags.includes(check.value)) return null;
    return new ImpossibleCheck(literalText(literal), check.base, ImpossibleReason.NOT_A_TAG, {
      baseLabel: namedTypeLabel(baseType, registry),
      value: check.value,
      field: check.field,
      tags,
    });
  }
  if (!(check instanceof IsinstanceCheck)) throw unknownCheck(check);
  if (isOpen(baseType)) return null;
  let typ: Type;
  try {
    typ = resolveIsinstanceType(check.typeName, registry);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    return null;
  }
  if (literal.value) {
    // A missing value is checked as null, which a nullable type admits.
    if (typ.nullable || typesOverlap(baseType, typ)) return null;
    return new ImpossibleCheck(literalText(literal), check.base, ImpossibleReason.NO_OVERLAP, {
      baseLabel: namedTypeLabel(baseType, registry),
      declared,
      typeLabel: namedTypeLabel(typ, registry),
    });
  }
  // A null base fails a non-nullable check, so its negation can hold.
  if (baseType.nullable && !typ.nullable) return null;
  const wanted = checkedMembers(typ);
  if (!checkedMembers(baseType).every((member) => contains(wanted, member))) return null;
  return new ImpossibleCheck(literalText(literal), check.base, ImpossibleReason.ALWAYS, {
    baseLabel: namedTypeLabel(baseType, registry),
    declared,
    alwaysTrue: alwaysTrue(literal),
  });
}

/**
 * Return the checks in a condition that are wrong whatever the values.
 *
 * Each literal in a boolean position (a check, or `not` applied to one) is tested against its
 * base's type and against what holds when it is evaluated:
 *
 * - Against the base's type alone, whatever the context: an `isinstance` naming a type no value
 *   of it can be (not a nullable type: a missing value is checked as null), a negated
 *   `isinstance` naming a type every value of it is, and a tag comparison whose text is not a tag
 *   of the union's tag field, reported as a misspelling for `==` and `!=` alike. A null check is
 *   never reported against its declared type alone, since testing a value that is never null for
 *   null is ordinary defensive code.
 * - Against the context, when the context can hold: the checks before the literal, met with what
 *   it proves, leave its base no member and no null. With `routing`, a literal the routing rules
 *   out together with the context is reported too, worded as routing only when the routing rules
 *   it out on its own.
 *
 * A check the types always decide (always true in an `or`, always false in an `and`) is no error
 * by itself, and neither is the code after it that no value reaches: sites check such a read as
 * without that check, and do not check a read after a reported check (`checkedRefs`), so the
 * report does not cascade into errors on the code it makes dead. Unproven, `object` and `Any`
 * bases, unknown type names, bases `baseTypeOf` refuses, and facts lost to an intersection prove
 * nothing. Each literal text is reported once.
 *
 * @param tree A condition parsed with `parseCondition`.
 * @param baseTypeOf The site's declared type of a base path, as for `guardEnv`.
 * @param registry The flow's typedefs.
 * @param options.routing The facts that hold wherever the condition is evaluated, such as a CASE
 *   node's routing facts. `null` (the default) is none. When no value satisfies them, the
 *   condition never runs and nothing is reported.
 * @returns The reported literals in source order.
 *
 * @example
 * With `v` a `Union[Buy, Hold, Sell]` tagged by `action`, `isinstance(v, Buy) and v.action ==
 * 'hold'` gives one `CONTEXT` check on `v.action == "hold"`, whose base is narrowed to `Buy`.
 */
export function impossibleChecks(
  tree: ExprNode,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  options: { routing?: Env | null } = {},
): ImpossibleCheck[] {
  const { routing = null } = options;
  const found: ImpossibleCheck[] = [];
  for (const [, impossible] of reportedLiterals(tree, baseTypeOf, registry, routing ?? new Map(), guardEnvMemo())) {
    if (found.every((seen) => impossible.text !== seen.text)) found.push(impossible);
  }
  return found;
}

/** Each literal `impossibleChecks` reports, with its report, before deduplication. */
function reportedLiterals(
  tree: ExprNode,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  routing: Env,
  memo: GuardEnvMemo,
): [CheckLiteral, ImpossibleCheck][] {
  if (envUnsatisfiable(routing)) return [];
  const found: [CheckLiteral, ImpossibleCheck][] = [];
  for (const literal of literalsInContext(tree)) {
    const contextEnv = guardEnvUnder(literal.context, baseTypeOf, registry, new Map(), memo);
    const routedEnv = intersectEnvs([routing, contextEnv]);
    const base = literal.check.base;
    const typ = baseTypeUnder(base, baseTypeOf, contextEnv);
    if (typ === NOT_NARROWABLE) continue;
    let impossible: ImpossibleCheck | null = null;
    if (typ !== null) {
      const declared = baseTypeOf(base) !== null;
      impossible = declaredImpossible(literal, typ, declared, registry);
    }
    if (impossible === null && !(envUnsatisfiable(contextEnv) || envUnsatisfiable(routedEnv))) {
      impossible = contextImpossible(literal, typ, contextEnv, routedEnv, baseTypeOf, registry, routing, memo);
    }
    if (impossible !== null) found.push([literal, impossible]);
  }
  return found;
}

/** The report a literal gets from what holds when it is evaluated, else `null`. */
function contextImpossible(
  literal: CheckLiteral,
  baseType: Type | null,
  contextEnv: Env,
  routedEnv: Env,
  baseTypeOf: BaseTypeOf,
  registry: TypeRegistry,
  routing: Env,
  memo: GuardEnvMemo,
): ImpossibleCheck | null {
  const base = literal.check.base;
  const own = new Holds(literal.check, literal.value);
  const text = literalText(literal);
  const met = intersectEnvs([contextEnv, guardEnvUnder(own, baseTypeOf, registry, contextEnv, memo)]);
  if (isEmptyMembers(met.get(base))) {
    const label = factLabel(contextEnv.get(base) ?? null, baseType, registry);
    return new ImpossibleCheck(text, base, ImpossibleReason.CONTEXT, {
      baseLabel: label,
      alwaysTrue: alwaysTrue(literal),
    });
  }
  if (!isEmptyMembers(intersectEnvs([routing, met]).get(base))) return null;
  const routedOwn = intersectEnvs([routing, guardEnvUnder(own, baseTypeOf, registry, routing, memo)]);
  let label: string;
  let reason: ImpossibleReason;
  if (isEmptyMembers(routedOwn.get(base))) {
    label = factLabel(routing.get(base) ?? null, baseType, registry);
    reason = ImpossibleReason.ROUTING;
  } else {
    label = factLabel(routedEnv.get(base) ?? null, baseType, registry);
    reason = ImpossibleReason.CONTEXT;
  }
  return new ImpossibleCheck(text, base, reason, { baseLabel: label, alwaysTrue: alwaysTrue(literal) });
}

/** The option closest to `word`, if one is close enough. */
function closeMatch(word: string, options: Iterable<string>): string | null {
  return getCloseMatches(word, [...options], 1, CLOSE_MATCH_CUTOFF)[0] ?? null;
}

/**
 * Return the load error text for an item of `impossibleChecks`.
 *
 * @param check An item of `impossibleChecks`.
 * @returns The message without a site prefix, such as: `` `isinstance(d.output, Text)` can never
 *   be true: d.output is declared Decision, and no Decision value fits Text ``.
 */
export function impossibleCheckMessage(check: ImpossibleCheck): string {
  const text = `\`${check.text}\``;
  const always = check.alwaysTrue ? `, so \`${check.alwaysTrue}\` is always true` : "";
  if (check.reason === ImpossibleReason.CONTEXT || check.reason === ImpossibleReason.ROUTING) {
    const why =
      check.reason === ImpossibleReason.CONTEXT ? "the checks before it" : "the case branch that routes this node";
    return `${text} can never be true here: ${check.base} is ${check.baseLabel}, by ${why}${always}`;
  }
  if (check.reason === ImpossibleReason.NO_OVERLAP || check.reason === ImpossibleReason.ALWAYS) {
    const base = check.declared
      ? `${check.base} is declared ${check.baseLabel}`
      : `${check.base} is ${check.baseLabel} here, by the checks before it`;
    if (check.reason === ImpossibleReason.ALWAYS) return `${text} can never be true: ${base}${always}`;
    return `${text} can never be true: ${base}, and no ${check.baseLabel} value fits ${check.typeLabel}`;
  }
  const message =
    `${text} compares ${check.base}.${check.field} with ${valueRepr(check.value)}, which is not ` +
    `one of its tags (${check.tags.join(", ")})`;
  const close = closeMatch(check.value, check.tags);
  return close === null ? message : `${message}; did you mean ${valueRepr(close)}?`;
}

/**
 * Return the load error text for a read of a field its narrowed type doesn't have.
 *
 * @param ref The dotted reference path read, such as `"r.answr"`.
 * @param field The field name the narrowed type lacks.
 * @param narrowedLabel The narrowed type's label, from `narrowedTypeLabel`.
 * @param base The narrowed base path.
 * @param fields The narrowed type's field names, for a close-match hint.
 * @param options.byRouting Whether the case branch that routes the node narrowed the base, rather
 *   than the checks before the read; defaults to `false`.
 * @returns Such as `reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done
 *   here, by the checks before this read; did you mean 'answer'?`.
 */
export function narrowedReadMessage(
  ref: string,
  field: string,
  narrowedLabel: string,
  base: string,
  fields: Iterable<string>,
  options: { byRouting?: boolean } = {},
): string {
  const why = options.byRouting ? "the case branch that routes this node" : "the checks before this read";
  const message =
    `reference \${${ref}} reads field ${reprText(field)}, which ${narrowedLabel} doesn't have; ` +
    `${base} is ${narrowedLabel} here, by ${why}`;
  const close = closeMatch(field, fields);
  return close === null ? message : `${message}; did you mean ${reprText(close)}?`;
}

/** The fields a record has, or any member of a union has; `null` for other types. */
function fieldNames(typ: Type): Set<string> | null {
  if (typ.members !== undefined) {
    const names = new Set<string>();
    for (const member of typ.members) {
      if (member.fields === undefined) return null;
      for (const name of member.fields.keys()) names.add(name);
    }
    return names;
  }
  return typ.fields === undefined ? null : new Set(typ.fields.keys());
}

/**
 * Return a narrowed read's error, naming the narrowed type when it lacks the field.
 *
 * @param ref The dotted reference path read, such as `"r.answr"`.
 * @param candidate The candidate the read failed on, from `guardedReadTypes`.
 * @param error The site's own error for the read on the candidate's type.
 * @param registry The flow's typedefs, for the type label.
 * @param options.byRouting Whether the case branch that routes the node narrowed the base, rather
 *   than the checks before the read; defaults to `false`.
 * @returns `narrowedReadMessage`'s text when no member of the narrowed type has the first field
 *   below the narrowed base; otherwise `error` unchanged (a field only some members have, or a
 *   field deeper down).
 *
 * @example
 * For `r.answr` with the candidate `[Done, ["answr"], "r"]`, the text is
 * `reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done here, by the
 * checks before this read; did you mean 'answer'?`.
 */
export function narrowedReadError(
  ref: string,
  candidate: NarrowedRead,
  error: string,
  registry: TypeRegistry,
  options: { byRouting?: boolean } = {},
): string {
  const [typ, fields, base] = candidate;
  const names = fieldNames(typ);
  const first = fields[0]!;
  if (names === null || names.has(first)) return error;
  const label = narrowedTypeLabel(typ, registry);
  return narrowedReadMessage(ref, first, label, base, [...names].sort(compareCodeUnits), options);
}
