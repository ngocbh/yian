/**
 * The nullable field-read check (design 2.5): a `.` step whose receiver may be null is a load error,
 * a `?.` step is not.
 *
 * Knows about: `expr` (reference steps), `nodes/contracts` (the output-contract sum),
 * `compile/output_contracts` (the type a contract gives field reads), `typesys`, and the compose
 * peers `helpers/condition_guards` and `helpers/narrowing_facts`.
 * Never imports: build, the loader, or the runtime.
 */

import { contractType } from "../../compile/output_contracts.ts";
import { type ExprNode, iterSubtreesTopdown, Tree } from "../../expr/grammar.ts";
import { type RefStep, splitReference, stepsText } from "../../expr/expressions.ts";
import { type OutputContract, ValueContract } from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { fileFieldType, Type, typeEquals, unionField, ValueKind } from "../../typesys/values.ts";
import { guardedReads } from "../helpers/condition_guards.ts";
import { type Env, guardedReadTypes, type NarrowedRead } from "../helpers/narrowing_facts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { rootBaseType } from "./asserts.ts";
import { ARITHMETIC } from "./skips.ts";

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
 * each field `T` doesn't require (owner ruling, 2026-10-25).
 *
 * @param tree A parsed expression.
 * @param roots What the site knows of its roots.
 * @param registry The flow's typedefs.
 * @returns The messages without a site prefix.
 */
export function nullableReadErrors(tree: ExprNode, roots: ReadRoots, registry: TypeRegistry): string[] {
  const messages: string[] = [];
  const operands = arithmeticOperands(tree);
  for (const { node, path, guard } of guardedReads(tree)) {
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
    for (const contract of alternatives) {
      const nullable = roots.nullable(root);
      // An opaque root walks as `Any`: untyped, unless a guard narrows it or it may be skipped.
      const chosen = contract ?? new ValueContract(Type.scalar(ValueKind.ANY));
      const baseTypeOf = rootBaseType(new Map([[root, chosen]]), nullable ? new Set([root]) : new Set());
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
