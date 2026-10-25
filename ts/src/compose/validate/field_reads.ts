/**
 * Field-read checks of one condition or expression: the name checks of a read against its root's
 * contracts, narrowed by the checks before it, with the impossible-check report; and the nullable
 * field-read check (design 2.5), where a `.` step whose receiver may be null is a load error and a
 * `?.` step is not.
 *
 * Knows about: `expr` (reference steps), `nodes/contracts` (the output-contract sum),
 * `compile/output_contracts` (the type a contract gives field reads), `typesys`, and the compose
 * peers `helpers/condition_guards` and `helpers/narrowing_facts`.
 * Never imports: build, the loader, or the runtime.
 */

import { contractType, outputContractPathError, projectOutputContract } from "../../compile/output_contracts.ts";
import { type ExprNode, iterSubtreesTopdown, Tree } from "../../expr/grammar.ts";
import { type RefStep, splitReference, stepsText } from "../../expr/expressions.ts";
import { isExactly, type OutputContract, sameOutputContract, ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { fileFieldType, Type, typeEquals, unionField, unionTagFields, ValueKind } from "../../typesys/values.ts";
import {
  type Check,
  type CheckLiteral,
  checkText,
  type Guard,
  guardedReads,
  guardedRefs,
  IsinstanceCheck,
  literalsInContext,
  literalText,
  NotNullCheck,
  sameGuard,
  TagCheck,
  unknownCheck,
} from "../helpers/condition_guards.ts";
import {
  type BaseTypeOf,
  checkedRefs,
  type Env,
  guardedReadTypes,
  ImpossibleCheck,
  impossibleCheckMessage,
  impossibleChecks,
  ImpossibleReason,
  NOT_NARROWABLE,
  type NarrowedRead,
  narrowedReadError,
  type NotNarrowable,
  unresolvedType,
} from "../helpers/narrowing_facts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { ARITHMETIC } from "./skips.ts";

/** The code of a nullable field-read load error. */
export const NULLABLE_FIELD_READ = "expr.nullable_field_read";

/** One field-read error of a condition, without a site prefix, and its diagnostic code. */
export interface FieldReadError {
  readonly message: string;
  readonly code: string;
}

/** What a site knows of its reference roots. */
export interface ReadRoots {
  /**
   * A root's alternatives, the contracts it may have here (`null` for an unknown one);
   * `undefined` when the name is no root at this site. A read is an error only when every
   * alternative rejects it.
   */
  contracts(root: string): readonly (OutputContract | null)[] | undefined;
  /** How many names a reference's root takes at this site: `poolRootLength`, or `() => 1`. */
  rootLength(names: readonly string[]): number;
  /** Whether the root may be null besides its type: a producer that may skip while the reader runs. */
  nullable(root: string): boolean;
  /** Why a root may be null, for the message. */
  why(root: string): string;
  /** Whether the root's un-narrowed value is untyped, so a type it is narrowed to holds only at load. */
  untyped(root: string): boolean;
  /** The facts the CASE branch routing this site proves; `null` for none. */
  routing(): Env | null;
}

/** Return whether `typ` is untyped for these rules: unknown, `Any`, or `object` with no fields. */
export function isUntyped(typ: Type | null): boolean {
  return (
    typ === null ||
    typ.kind === ValueKind.ANY ||
    (typ.kind === ValueKind.OBJECT && typ.fields === undefined && typ.variants === undefined)
  );
}

/**
 * Return one message per read that steps `.` through a value that may be null, in source order,
 * without repeats.
 *
 * The root's receiver may be null when its type is `Optional` or `roots.nullable` says so; a later
 * step's receiver when the previous field is `Optional` or not required. Nullability an earlier
 * `?.` introduced doesn't count (TypeScript optional chains). A guard's narrowing lifts it.
 * Untyped receivers are never flagged, except a root that may be skipped: a `.` right after it is
 * flagged whatever the producer's type (design 2.6.6). An operand of arithmetic names the whole
 * fix, `(x?.k :- <default>)`, so the skipped-arithmetic check needn't add a second message. An
 * untyped root narrowed to `T` (by a guard or the CASE branch routing the site) needs `?.` for
 * each field `T` doesn't require (owner ruling, 2026-10-25). A read no value reaches (its guard
 * can never hold) is not checked; `checkImpossibleWhens` or the asserts' check reports the guard.
 *
 * @param tree A parsed expression.
 * @param roots What the site knows of its roots.
 * @param registry The flow's typedefs.
 * @returns The messages without a site prefix.
 */
export function nullableReadErrors(tree: ExprNode, roots: ReadRoots, registry: TypeRegistry): string[] {
  const messages: string[] = [];
  const operands = arithmeticOperands(tree);
  // What `checkedRefs` gives under each root's alternative, index-aligned with `guardedReads`.
  const checkedOf = new Map<string, (readonly [string, Guard | null][])[]>();
  for (const [index, { node, path }] of guardedReads(tree).entries()) {
    const { head, steps } = splitReference(node);
    const used = roots.rootLength([head, ...steps.map((step) => step.name)]);
    const rest = steps.slice(used - 1);
    if (rest.length === 0) continue;
    const root = [head, ...steps.slice(0, used - 1).map((step) => step.name)].join(".");
    const alternatives = roots.contracts(root);
    if (alternatives === undefined) continue;
    const read: Read = {
      root,
      rootText: head + stepsText(steps.slice(0, used - 1)),
      rest,
      arithmetic: operands.has(node),
    };
    let first: string | null = null;
    let accepted = alternatives.length === 0;
    if (!checkedOf.has(root)) checkedOf.set(root, []);
    for (const [which, contract] of alternatives.entries()) {
      const nullable = roots.nullable(root);
      // An opaque root walks as `Any`: untyped, unless a guard narrows it or it may be skipped.
      const chosen = contract ?? new ValueContract(Type.scalar(ValueKind.ANY));
      const baseTypeOf = rootBaseType(new Map([[root, chosen]]), nullable ? new Set([root]) : new Set());
      const checked = (checkedOf.get(root)![which] ??= checkedRefs(tree, baseTypeOf, registry, {
        routing: roots.routing(),
      }));
      const guard = checked[index]![1];
      if (guard === null) {
        // Reached by no value under this alternative: the impossible-check report explains it.
        accepted = true;
        break;
      }
      const candidates = guardedReadTypes(path, guard, baseTypeOf, registry);
      const error = alternativeError(read, chosen, candidates, roots, registry);
      if (error === null) {
        accepted = true;
        break;
      }
      first ??= error;
    }
    if (!accepted && first !== null) messages.push(first);
  }
  return [...new Set(messages)];
}

/** One read split at its root: the root's flat path, the root as written, and the steps after it. */
interface Read {
  readonly root: string;
  readonly rootText: string;
  readonly rest: readonly RefStep[];
  /** Whether the read is a direct operand of arithmetic. */
  readonly arithmetic: boolean;
}

/** Return the refcalls that are direct operands of arithmetic; parentheses make no node. */
function arithmeticOperands(tree: ExprNode): Set<ExprNode> {
  const operands = new Set<ExprNode>();
  if (!(tree instanceof Tree)) return operands;
  for (const node of iterSubtreesTopdown(tree)) {
    if (!ARITHMETIC.has(node.data)) continue;
    for (const child of node.children) {
      if (child instanceof Tree && child.data === "refcall") operands.add(child);
    }
  }
  return operands;
}

/**
 * Return the read's error under one alternative `contract` of its root, else `null`: it is an error
 * when every candidate's walk (or the one un-narrowed walk) gives one, and the first is returned.
 */
function alternativeError(
  read: Read,
  contract: OutputContract,
  candidates: readonly NarrowedRead[],
  roots: ReadRoots,
  registry: TypeRegistry,
): string | null {
  let first: string | null = null;
  for (const candidate of candidates.length === 0 ? [null] : candidates) {
    const error = walk(read, contract, candidate, roots, registry);
    if (error === null) return null;
    first ??= error;
  }
  return first;
}

/**
 * Return the first `.` step of `read` through a value that may be null, as a message, else `null`.
 *
 * A walk from an untyped root narrowed to a type (by a guard, or by the CASE branch routing the
 * site) is untyped at run time, so every field that type doesn't require may be absent there.
 */
function walk(
  read: Read,
  contract: OutputContract,
  candidate: NarrowedRead | null,
  roots: ReadRoots,
  registry: TypeRegistry,
): string | null {
  const nullable = roots.nullable(read.root);
  let typ = contractType(contract);
  if (typ !== null && nullable) typ = typ.replace({ nullable: true });
  let path = read.root;
  let prefix = read.rootText;
  if (candidate !== null && candidate[2] === path) typ = candidate[0];
  let why = roots.why(read.root);
  const narrowedUntyped = roots.untyped(read.root) && typ !== null && !isUntyped(typ);
  // A root that may be skipped is a missing root at run time, whatever its type: check its step
  // before the untyped stop.
  if (nullable && (typ === null || typ.nullable) && !read.rest[0]!.optional) {
    return nullableMessage(prefix, read.rest[0]!.name, why, read.arithmetic);
  }
  for (const step of read.rest) {
    if (typ === null || isUntyped(typ)) return null;
    if (typ.nullable && !step.optional) return nullableMessage(prefix, step.name, why, read.arithmetic);
    const field = fieldRead(typ, step.name);
    if (field === null) return null; // the name checks report it
    if (narrowedUntyped && field.nullable && !step.optional) {
      return narrowedUntypedMessage(prefix, step.name, namedTypeLabel(typ, registry));
    }
    typ = field.type?.replace({ nullable: field.nullable }) ?? null;
    why = field.optional ? `field '${step.name}' is Optional` : `field '${step.name}' is not required`;
    path += "." + step.name;
    prefix += (step.optional ? "?." : ".") + step.name;
    if (candidate !== null && candidate[2] === path) typ = candidate[0];
  }
  return null;
}

/**
 * Return the type of field `name` of a non-null `typ` (`null` when untyped), whether it may be null
 * and whether that is because it is `Optional`; `null` when `typ` has no such field.
 */
function fieldRead(typ: Type, name: string): { type: Type | null; nullable: boolean; optional: boolean } | null {
  if (typ.variants !== undefined) {
    const read = unionField(typ, name);
    if (read.kind === "absent") return null;
    if (read.kind === "read") {
      const optional = read.type?.nullable ?? false;
      return { type: read.type, nullable: optional, optional };
    }
    // A field only some members have reads null on the others.
    const types = read.members.map((member) => member.fields!.get(name)!.replace({ nullable: false }));
    const same = types.every((t) => typeEquals(t, types[0]!));
    return { type: same ? types[0]! : null, nullable: true, optional: false };
  }
  if (typ.kind === ValueKind.FILE) {
    const field = fileFieldType(name);
    return field === null ? null : { type: field, nullable: field.nullable, optional: field.nullable };
  }
  const field = typ.fields?.get(name);
  if (field === undefined) return null;
  const optional = field.nullable;
  return { type: field, nullable: optional || !(typ.required?.has(name) ?? false), optional };
}

/** Return the message for `.name` read on `prefix`, untyped and only narrowed to `label`. */
function narrowedUntypedMessage(prefix: string, name: string, label: string): string {
  return (
    `\${${prefix}.${name}} reads field '${name}', which ${label} doesn't require, of ${prefix}, ` +
    `which is untyped and only narrowed to ${label}; ` +
    `write ${prefix}?.${name}, or type ${prefix} as ${label} where it is bound (\`output: ${label}\` or \`!${label}\`)`
  );
}

/**
 * Return the message for `.name` read on `prefix`, which may be null because of `why`; an
 * `arithmetic` operand's fix gives the default too.
 */
function nullableMessage(prefix: string, name: string, why: string, arithmetic: boolean): string {
  const fix = arithmetic
    ? `write (${prefix}?.${name} :- <default>)`
    : `write ${prefix}?.${name}, or check ${prefix} != null first`;
  return `\${${prefix}.${name}} reads field '${name}' of ${prefix}, which may be null (${why}); ${fix}`;
}

/** Return the root `path` is at or below, else `null`. */
function rootOf(path: string, roots: Iterable<string>): string | null {
  for (const root of roots) {
    if (path === root || path.startsWith(root + ".")) return root;
  }
  return null;
}

/** Return the field names of `path` below `root`, which it is at or below. */
function below(path: string, root: string): string[] {
  return path === root ? [] : path.slice(root.length + 1).split(".");
}

/**
 * Return the declared type of a base path below one of `choice`'s roots, under that root's chosen
 * contract, made nullable below a root in `nullableRoots`; other bases are not narrowed. Exported
 * for tests.
 *
 * @param choice The contract chosen for each root.
 * @param nullableRoots The roots that may be null whatever their contracts say (default none).
 * @returns The site's `BaseTypeOf`: `null` for a base the contract doesn't prove, `NOT_NARROWABLE`
 *   for a base below no root.
 */
export function rootBaseType(
  choice: ReadonlyMap<string, OutputContract>,
  nullableRoots: ReadonlySet<string> = new Set(),
): BaseTypeOf {
  return (base: string): Type | null | NotNarrowable => {
    const root = rootOf(base, choice.keys());
    if (root === null) return NOT_NARROWABLE;
    const projected = projectOutputContract(choice.get(root)!, below(base, root));
    if (!isExactly(projected, ValueContract)) return null;
    if (nullableRoots.has(root)) return projected.type.replace({ nullable: true });
    return projected.type;
  };
}

/** Return `contracts` without repeats, in order. */
function distinct(contracts: readonly OutputContract[]): OutputContract[] {
  const kept: OutputContract[] = [];
  for (const contract of contracts) {
    if (!kept.some((seen) => sameOutputContract(contract, seen))) kept.push(contract);
  }
  return kept;
}

/**
 * Return whether `check` can prove or rule out anything about `root` under some contract. Exported
 * for tests.
 *
 * A tag comparison on a value that is no union with that tag field (`v.a == 'x'` on a record) is an
 * ordinary comparison, so it narrows nothing; nor does a null check on a value that is never null.
 *
 * @param check A check from a literal of the condition.
 * @param root The root path whose values the check may narrow.
 * @param contracts The contracts the root's value may have.
 * @param nullable Whether the root may be null whatever its contracts say, which makes every value
 *   below it nullable too (default `false`).
 * @returns Whether the check narrows the root.
 * @throws TypeError for a check of a kind this function doesn't know.
 */
export function narrows(
  check: Check,
  root: string,
  contracts: readonly OutputContract[],
  nullable = false,
): boolean {
  if (rootOf(check.base, [root]) === null) return false;
  if (check instanceof IsinstanceCheck) return true;
  if (!(check instanceof TagCheck || check instanceof NotNullCheck)) throw unknownCheck(check);
  // `rootBaseType` makes every base below such a root nullable, so a null check there narrows
  // even when no contract lets the value be null.
  if (check instanceof NotNullCheck && nullable) return true;
  const rest = below(check.base, root);
  for (const contract of contracts) {
    const projected = projectOutputContract(contract, rest);
    if (!isExactly(projected, ValueContract)) {
      // An unproven base takes its type from the checks on the value around it.
      return true;
    }
    const typ = projected.type;
    if (check instanceof NotNullCheck) {
      if (typ.nullable) return true;
    } else if (typ.variants !== undefined && unionTagFields(typ).includes(check.field)) {
      return true;
    }
  }
  return false;
}

/** What a condition proves under one choice of root contracts. */
interface ChoiceView {
  readonly choice: ReadonlyMap<string, OutputContract>;
  readonly baseTypeOf: BaseTypeOf;
  readonly impossible: readonly ImpossibleCheck[];
  readonly checked: readonly [string, Guard | null][];
}

/** How a site words a field-read error; frozen. */
export class ReadSite {
  /**
   * Whether the condition runs in a spawned child flow, where `isinstance` is not allowed, so a
   * member-only hint compares the tag instead.
   */
  readonly spawner: boolean;
  /** The pool path each root is bound from, named by a spawner's case hint. */
  readonly unionRoots: ReadonlyMap<string, string>;

  constructor(options: { spawner?: boolean; unionRoots?: ReadonlyMap<string, string> } = {}) {
    this.spawner = options.spawner ?? false;
    this.unionRoots = options.unionRoots ?? new Map();
    Object.freeze(this);
  }

  /** Return the read's error on `contract`, else `null`. */
  error(
    contract: OutputContract,
    fields: readonly string[],
    ref: string,
    options: { partialUnionFields: boolean; unionRoot?: string | null },
  ): string | null {
    return outputContractPathError(contract, fields, ref, {
      partialUnionFields: options.partialUnionFields,
      unionRoot: this.spawner ? (options.unionRoot ?? null) : null,
      isinstanceHint: !this.spawner,
      tagHint: this.spawner,
    });
  }
}

/** Return `check` rebuilt with `changes`, as its frozen class can't be updated in place. */
function replaceCheck(
  check: ImpossibleCheck,
  changes: { text?: string; base?: string; alwaysTrue?: string; baseLabel?: string },
): ImpossibleCheck {
  return new ImpossibleCheck(changes.text ?? check.text, changes.base ?? check.base, check.reason, {
    baseLabel: changes.baseLabel ?? check.baseLabel,
    declared: check.declared,
    typeLabel: check.typeLabel,
    alwaysTrue: changes.alwaysTrue ?? check.alwaysTrue,
    value: check.value,
    field: check.field,
    tags: check.tags,
  });
}

/**
 * Return a map from a report on `tree` back to how `authored`, the same condition before its
 * references were rewritten, spells the literal.
 */
function authoredSpelling(
  tree: ExprNode,
  authored: ExprNode | null,
): (check: ImpossibleCheck) => ImpossibleCheck {
  const spelled = new Map<string, CheckLiteral>();
  if (authored !== null) {
    const rewritten = literalsInContext(tree);
    const written = literalsInContext(authored);
    if (rewritten.length === written.length) {
      rewritten.forEach((literal, index) => {
        const text = literalText(literal);
        if (!spelled.has(text)) spelled.set(text, written[index]!);
      });
    }
  }
  return (check) => {
    const original = spelled.get(check.text);
    if (original === undefined) return check;
    const alwaysTrue = check.alwaysTrue ? checkText(original.check, true) : "";
    return replaceCheck(check, { text: literalText(original), base: original.check.base, alwaysTrue });
  };
}

/**
 * Return an assert's impossible checks and field-read errors, each message once.
 *
 * `roots` maps each path whose field reads are checked (a node-local name, `output`, `input.X`,
 * `<id>.output`) to the contracts its value may have; reads below a root in `unwalkedRoots` are
 * left to the site, and a root in `nullableRoots` may also be null whatever its contract says.
 * `and`, `or` and `not` short-circuit, so the checks before a read (`isinstance`, union tag
 * comparisons and null checks) narrow the values it can see (`guardedReadTypes`): a read is fine
 * when any narrowed type allows it, and otherwise the first one's error is reported, naming the
 * narrowed type when it lacks the field. A read nothing narrows is walked on the root's contract.
 * A read that runs only when a reported impossible check is true (`checkedRefs`), or guarded by an
 * `isinstance` whose type name doesn't resolve, is not checked. `authored` is the condition as
 * written when `tree` has its references rewritten; reports spell checks as written.
 *
 * A root with several distinct contracts (a node-local name's alternatives, which carry the branch
 * that routes the node) is checked under each: a check is impossible only when it is impossible
 * under every contract of its base's root, a read is skipped only when it is dead under every
 * contract of some root, and a read is an error only when every contract of its root where it runs
 * rejects it. Facts never cross roots, so each root's contracts are tried with every other such
 * root left unnarrowed, which costs the sum of the alternatives rather than their product. Only
 * roots some check can narrow are tried; reads below the others are walked on each contract. Each
 * distinct (read, guard) pair is checked, since one path can be read both narrowed and not.
 *
 * @param tree The parsed condition.
 * @param roots The checked roots and the contracts each may have.
 * @param registry The flow's typedefs.
 * @param options.partialUnionRoots Roots below which a field only some union members have can be
 *   read (default none).
 * @param options.unwalkedRoots Roots whose reads the site checks itself (default none).
 * @param options.nullableRoots Roots that may be null whatever their contracts say (default none).
 * @param options.site How the site words a field-read error (default a non-spawner site).
 * @param options.authored The condition as written, when `tree` is rewritten (default `null`).
 * @returns The messages without a site prefix, in order, without repeats.
 */
function nameErrors(
  tree: ExprNode,
  roots: ReadonlyMap<string, readonly OutputContract[]>,
  registry: TypeRegistry,
  options: {
    partialUnionRoots?: ReadonlySet<string>;
    unwalkedRoots?: ReadonlySet<string>;
    nullableRoots?: ReadonlySet<string>;
    site?: ReadSite;
    authored?: ExprNode | null;
  } = {},
): string[] {
  const {
    partialUnionRoots = new Set<string>(),
    unwalkedRoots = new Set<string>(),
    nullableRoots = new Set<string>(),
    site = new ReadSite(),
    authored = null,
  } = options;
  const refs = guardedRefs(tree);
  const used = [...roots.keys()].filter((root) => refs.some(([ref]) => rootOf(ref, [root]) !== null));
  if (used.length === 0) return [];
  const contracts = new Map(used.map((root) => [root, distinct(roots.get(root)!)]));
  const literals = literalsInContext(tree);
  const narrowed = used.filter((root) =>
    literals.some((literal) => narrows(literal.check, root, contracts.get(root)!, nullableRoots.has(root))),
  );
  const fixed = new Map<string, OutputContract>();
  for (const root of narrowed) {
    if (contracts.get(root)!.length === 1) fixed.set(root, contracts.get(root)![0]!);
  }
  const varied = narrowed.filter((root) => contracts.get(root)!.length > 1);
  let choices = varied.flatMap((root) =>
    contracts.get(root)!.map((contract) => new Map([...fixed, [root, contract]])),
  );
  if (narrowed.length > 0 && varied.length === 0) choices = [fixed];
  const views: ChoiceView[] = choices.map((choice) => {
    const baseTypeOf = rootBaseType(choice, nullableRoots);
    return {
      choice,
      baseTypeOf,
      impossible: impossibleChecks(tree, baseTypeOf, registry),
      checked: checkedRefs(tree, baseTypeOf, registry),
    };
  });

  /** The views that try each of `root`'s contracts; all views for any other root. */
  const viewsOf = (root: string | null): ChoiceView[] =>
    root === null || !varied.includes(root) ? views : views.filter((view) => view.choice.has(root));

  const spell = authoredSpelling(tree, authored);
  const order = new Map<string, number>();
  literals.forEach((literal, index) => {
    const text = literalText(literal);
    if (!order.has(text)) order.set(text, index);
  });
  const reported = new Map<string, ImpossibleCheck>();
  for (const view of views) {
    for (const check of view.impossible) {
      if (!reported.has(check.text)) reported.set(check.text, check);
    }
  }
  const messages: string[] = [];
  const rank = (text: string) => order.get(text) ?? order.size;
  for (const text of [...reported.keys()].sort((a, b) => rank(a) - rank(b))) {
    let check = reported.get(text)!;
    const group = viewsOf(rootOf(check.base, narrowed));
    if (group.every((view) => view.impossible.some((seen) => seen.text === text))) {
      check = declaredLabel(check, group[0]!.choice, nullableRoots, registry);
      messages.push(impossibleCheckMessage(spell(check)));
    }
  }
  // A read is dead under every combination of contracts when every contract of one root kills it,
  // since what kills it under one root's contract holds whatever the others are.
  const groups = varied.length > 0 ? varied.map((root) => viewsOf(root)) : [views];
  const seenReads = new Map<string, Guard[]>();
  refs.forEach(([ref, guard], index) => {
    const root = rootOf(ref, used);
    if (root === null || ref === root || unwalkedRoots.has(root)) return;
    const seenGuards = seenReads.get(ref) ?? [];
    if (seenGuards.some((seen) => sameGuard(seen, guard))) return;
    seenReads.set(ref, [...seenGuards, guard]);
    if (unresolvedType(guard, registry)) {
      // The type-table pass reports the unknown name.
      return;
    }
    if (views.length > 0 && groups.some((group) => group.every((view) => view.checked[index]![1] === null))) {
      // Reached by no value under any choice: a reported check explains it.
      return;
    }
    const message = readError(
      ref,
      root,
      index,
      viewsOf(root).filter((view) => view.checked[index]![1] !== null),
      narrowed.includes(root) ? null : contracts.get(root)!,
      registry,
      site,
      { partialUnionFields: partialUnionRoots.has(root), nullable: nullableRoots.has(root) },
    );
    if (message !== null) messages.push(message);
  });
  return [...new Set(messages)];
}

/**
 * Return `check` with its declared base type labelled as declared, for a base below a root that is
 * nullable only because its producer may be skipped.
 */
function declaredLabel(
  check: ImpossibleCheck,
  choice: ReadonlyMap<string, OutputContract>,
  nullableRoots: ReadonlySet<string>,
  registry: TypeRegistry,
): ImpossibleCheck {
  if (!check.declared || (check.reason !== ImpossibleReason.NO_OVERLAP && check.reason !== ImpossibleReason.ALWAYS)) {
    return check;
  }
  if (rootOf(check.base, nullableRoots) === null) return check;
  const declared = rootBaseType(choice)(check.base);
  if (!(declared instanceof Type)) return check;
  return replaceCheck(check, { baseLabel: namedTypeLabel(declared, registry) });
}

/**
 * Return why the `index`-th read `ref` below `root` can't work, else `null`.
 *
 * The read is an error when every view in `live` (those trying `root`'s contracts where it runs)
 * rejects it: on each of the `unnarrowed` contracts when no check narrows `root`, else on the
 * view's narrowed types or chosen contract. A narrowed error is preferred over an unnarrowed one.
 * When `root` is `nullable` only because its producer may be skipped, the error names the declared
 * type, not the nullable one the checks narrowed.
 */
function readError(
  ref: string,
  root: string,
  index: number,
  live: readonly ChoiceView[],
  unnarrowed: readonly OutputContract[] | null,
  registry: TypeRegistry,
  site: ReadSite,
  options: { partialUnionFields: boolean; nullable?: boolean },
): string | null {
  const { partialUnionFields, nullable = false } = options;
  const fields = below(ref, root);
  const unionRoot = site.unionRoots.get(root) ?? null;
  if (unnarrowed !== null) {
    let first: string | null = null;
    for (const contract of unnarrowed) {
      const error = site.error(contract, fields, ref, { partialUnionFields, unionRoot });
      if (error === null) return null;
      first = first ?? error;
    }
    return first;
  }
  const errors: [narrowed: boolean, error: string][] = [];
  for (const { choice, baseTypeOf, checked } of live) {
    const guard = checked[index]![1]!;
    const candidates = guardedReadTypes(ref, guard, baseTypeOf, registry);
    if (candidates.length === 0) {
      const error = site.error(choice.get(root)!, fields, ref, { partialUnionFields, unionRoot });
      if (error === null) return null;
      errors.push([false, error]);
      continue;
    }
    let narrowedError: string | null = null;
    for (const candidate of candidates) {
      const [typ, rest] = candidate;
      const error = site.error(new ValueContract(typ), rest, ref, { partialUnionFields });
      if (error === null) return null;
      if (narrowedError !== null) continue;
      if (nullable) {
        narrowedError = declaredReadError(
          ref,
          candidate,
          error,
          choice.get(root)!,
          fields,
          rootBaseType(choice)(candidate[2]),
          registry,
          site,
          { partialUnionFields, unionRoot },
        );
      } else {
        narrowedError = narrowedReadError(ref, candidate, error, registry);
      }
    }
    errors.push([true, narrowedError!]);
  }
  if (errors.length === 0) return null;
  return (errors.find(([narrowed]) => narrowed) ?? errors[0]!)[1];
}

/**
 * Return a narrowed read's error below a root that is nullable only because its producer may be
 * skipped, worded on the `declared` type of the candidate's base.
 *
 * The narrowed type loses that null, and when the checks leave only null the read (`fields` below
 * the root) is walked on the root's chosen `contract` instead.
 */
function declaredReadError(
  ref: string,
  candidate: NarrowedRead,
  error: string,
  contract: OutputContract,
  fields: readonly string[],
  declared: Type | null | NotNarrowable,
  registry: TypeRegistry,
  site: ReadSite,
  options: { partialUnionFields: boolean; unionRoot: string | null },
): string {
  const { partialUnionFields, unionRoot } = options;
  const [typ, rest, base] = candidate;
  if (!(declared instanceof Type) || declared.nullable) return narrowedReadError(ref, candidate, error, registry);
  if (typ.kind === ValueKind.NONE) {
    return site.error(contract, fields, ref, { partialUnionFields, unionRoot }) ?? error;
  }
  const stripped = typ.replace({ nullable: false });
  const strippedError = site.error(new ValueContract(stripped), rest, ref, { partialUnionFields }) ?? error;
  return narrowedReadError(ref, [stripped, rest, base], strippedError, registry);
}

/**
 * Return a condition's name errors (see `nameErrors`), then its nullable field-read errors
 * (`nullableReadErrors`), each message once.
 *
 * The nullable check reads each root's alternatives from `roots`, an `opaqueRoots` root as one
 * unknown contract (it gets no name check), and takes `nullableRoots` as the roots that may be
 * null besides their type.
 *
 * @param tree The parsed condition.
 * @param roots The checked roots and the contracts each may have.
 * @param registry The flow's typedefs.
 * @param options As for `nameErrors`, plus:
 * @param options.opaqueRoots Roots with no contract, still checked for nullable reads (default none).
 * @param options.rootLength How many names a reference's root takes (default one).
 * @param options.why Why a root in `nullableRoots` may be null (default "it is Optional").
 * @param options.untyped Whether a root's un-narrowed value is untyped (default: every one of its
 *   `roots` contracts is untyped, or it has none).
 * @returns The errors in order: name errors with code `compose.load`, then nullable ones.
 */
export function fieldReadErrors(
  tree: ExprNode,
  roots: ReadonlyMap<string, readonly OutputContract[]>,
  registry: TypeRegistry,
  options: {
    partialUnionRoots?: ReadonlySet<string>;
    unwalkedRoots?: ReadonlySet<string>;
    nullableRoots?: ReadonlySet<string>;
    site?: ReadSite;
    authored?: ExprNode | null;
    opaqueRoots?: ReadonlySet<string>;
    rootLength?: (names: readonly string[]) => number;
    why?: (root: string) => string;
    untyped?: (root: string) => boolean;
  } = {},
): FieldReadError[] {
  const {
    nullableRoots = new Set<string>(),
    opaqueRoots = new Set<string>(),
    rootLength = () => 1,
    why = () => "it is Optional",
    untyped = (root: string) => (roots.get(root) ?? []).every((contract) => isUntyped(contractType(contract))),
  } = options;
  const names = nameErrors(tree, roots, registry, options);
  const readRoots: ReadRoots = {
    contracts: (root) => roots.get(root) ?? (opaqueRoots.has(root) ? [null] : undefined),
    rootLength,
    nullable: (root) => nullableRoots.has(root),
    why,
    untyped,
    routing: () => null,
  };
  const nullable = nullableReadErrors(tree, readRoots, registry).filter((message) => !names.includes(message));
  return [
    ...names.map((message) => ({ message, code: "compose.load" })),
    ...nullable.map((message) => ({ message, code: NULLABLE_FIELD_READ })),
  ];
}
