/**
 * Graph stage: reject `${T(...)}` constructor calls that can never succeed.
 *
 * A constructor `T(x)` checks the value of `x` against the `typedefs:` type `T`; the keyword form
 * `T(a=..., b=...)` builds a record from exactly the given fields and checks it. Most of what can
 * go wrong is visible in the call itself:
 *
 * - the call mixes keyword and positional arguments, or has several positional ones;
 * - the keyword form names a union (construct a member instead) or a type that is no record, a
 *   field `T` doesn't have, or leaves out a required field;
 * - `T` is an inbox type, which has no constructor;
 * - a field read on the result (`T(x).f`) names a field `T` can't have.
 *
 * `builtinFieldErrors` rejects a field read on a builtin whose result type is known
 * (`upper(x).f` reads a field of text), the same way.
 *
 * Given the types of the references an expression reads, a positional argument or a keyword field
 * whose inferred type is disjoint from the declared one is an error too. A narrower or unproven
 * value loads and is left to the runtime check.
 *
 * Knows about: the type relation (`compile/validation`), expressions, the type system, the
 * field-read message (`nodes/contracts`), and compose helpers. Callers are the reference
 * validator and the pipeline's type-table pass.
 * Never imports: build, the loader, or the runtime.
 */

import { typeRelation } from "../../compile/validation.ts";
import { reprText } from "../../diagnostics.ts";
import { builtinResultType, TEMPLATE_FNS } from "../../expr/builtins.ts";
import { constructorShapeError, inferExprType, ISINSTANCE } from "../../expr/expressions.ts";
import { callSuffixOf, type ExprNode, iterSubtreesTopdown, Token, Tree } from "../../expr/grammar.ts";
import { outputContractFieldError, ValueContract } from "../../nodes/contracts.ts";
import type { RefExpr, TypeRegistry } from "../../typesys/types.ts";
import { canonicalTypeLabel, isInboxType, type Type } from "../../typesys/values.ts";
import { constructorTypeFn } from "../helpers/isinstance_types.ts";
import { literalType, type TypeFn } from "../helpers/output_contracts.ts";

/** Resolves a reference path to its proven type, or `null` when it is unknown. */
export type RefType = (path: string) => Type | null;

/**
 * Return why each constructor call in one parsed expression can never succeed.
 *
 * @param tree A parsed `${...}` interior or condition.
 * @param registry The flow's `typedefs:` registry; a callee outside it is no constructor and is
 *   reported elsewhere.
 * @param refType Resolves a reference path to its proven type. `null` (the default) checks only
 *   what the call itself shows.
 * @returns One message per problem, in source order.
 */
export function constructorErrors(tree: ExprNode, registry: TypeRegistry, refType: RefType | null = null): string[] {
  if (!(tree instanceof Tree)) return [];
  const typeFn = constructorTypeFn(registry);
  const errors: string[] = [];
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall") continue;
    const callSuffix = callSuffixOf(node);
    const name = String(node.children[0]);
    if (callSuffix === undefined || name === ISINSTANCE || TEMPLATE_FNS.has(name)) continue;
    const typ = typeFn(name);
    if (typ === null) continue;
    if (isInboxType(typ)) {
      // A handle names a top-level inbox; building one from a value would forge it.
      errors.push(`\`${name}\` is an inbox type; it has no constructor`);
      continue;
    }
    const args = callSuffix.children as Tree[];
    const shapeError = constructorShapeError(name, callSuffix);
    if (shapeError !== null) {
      errors.push(shapeError);
      continue;
    }
    if (args.length === 1 && args[0]!.children.length === 1) {
      const error = valueError(name, typ, args[0]!.children[0]!, refType, typeFn);
      if (error !== null) errors.push(error);
    } else {
      errors.push(...recordErrors(name, typ, args, registry, refType, typeFn));
    }
    const error = resultFieldError(node, name, typ);
    if (error !== null) errors.push(error);
  }
  return errors;
}

/**
 * Return why each field read on a builtin call's result in one expression fails.
 *
 * A builtin whose result type is unknown (`builtinResultType` gives `null`) is left unchecked.
 *
 * @param tree A parsed `${...}` interior or condition.
 * @returns One message per field read the result type can't have, in source order.
 */
export function builtinFieldErrors(tree: ExprNode): string[] {
  if (!(tree instanceof Tree)) return [];
  const errors: string[] = [];
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall") continue;
    // A `${upper.f}` read of an input named `upper` is no call.
    if (callSuffixOf(node) === undefined) continue;
    const name = String(node.children[0]);
    const typ = builtinResultType(name);
    if (typ === null) continue;
    const error = resultFieldError(node, name, typ);
    if (error !== null) errors.push(error);
  }
  return errors;
}

/** Return why the field reads trailing call `name(...)`, whose result is `typ`, fail. */
function resultFieldError(node: Tree, name: string, typ: Type): string | null {
  const fields = node.children
    .filter((child): child is Tree => child instanceof Tree && child.data === "trailer")
    .map((child) => String(child.children[0]));
  if (fields.length === 0) return null;
  return outputContractFieldError(new ValueContract(typ), fields, `${name}(...).` + fields.join("."));
}

/** Return why `T(x)`'s value can never be a `T`, else `null`. */
function valueError(name: string, typ: Type, arg: ExprNode, refType: RefType | null, typeFn: TypeFn): string | null {
  const inferred = argumentType(arg, refType, typeFn);
  if (inferred === null || typeRelation(inferred, typ) !== "disjoint") return null;
  return `${name}(...): the value is ${canonicalTypeLabel(inferred)}, which can never be a ${name}`;
}

/** Return why `T(a=..., b=...)` can never build a valid `T`. */
function recordErrors(
  name: string,
  typ: Type,
  args: readonly Tree[],
  registry: TypeRegistry,
  refType: RefType | null,
  typeFn: TypeFn,
): string[] {
  if (typ.variants !== undefined) {
    const members = unionMemberNames(name, registry);
    const listed = members.length > 0 ? ` (${members.map((m) => `${m}(...)`).join(" or ")})` : "";
    return [`${name} is a union; construct a member${listed}, or check a value with ${name}(x)`];
  }
  if (typ.fields === undefined) {
    return [`${name} is not a record, so ${name}(...) takes one value to check, as in ${name}(x)`];
  }
  const given = new Map(args.map((arg) => [String(arg.children[0]), arg.children[1]!]));
  const errors: string[] = [];
  for (const fieldName of given.keys()) {
    if (!typ.fields.has(fieldName)) {
      errors.push(`${name} has no field ${reprText(fieldName)} (fields: ${[...typ.fields.keys()].join(", ")})`);
    }
  }
  const missing = [...typ.fields.keys()].filter(
    (fieldName) => (typ.required?.has(fieldName) ?? false) && !given.has(fieldName),
  );
  if (missing.length > 0) {
    const noun = missing.length === 1 ? "field" : "fields";
    errors.push(`${name}(...) is missing required ${noun} ` + missing.map((fieldName) => `\`${fieldName}\``).join(", "));
  }
  for (const [fieldName, value] of given) {
    const declared = typ.fields.get(fieldName);
    if (declared === undefined) continue;
    const inferred = argumentType(value, refType, typeFn);
    if (inferred !== null && typeRelation(inferred, declared) === "disjoint") {
      errors.push(
        `${name}(...) field ${reprText(fieldName)}: declared ${canonicalTypeLabel(declared)}, ` +
          `but the value is ${canonicalTypeLabel(inferred)}`,
      );
    }
  }
  return errors;
}

/** Infer one argument's type; a literal has its exact type, so a wrong one is disjoint. */
function argumentType(arg: ExprNode, refType: RefType | null, typeFn: TypeFn): Type | null {
  // `INF` and `NULL` tokens are inferred like any other expression.
  if (arg instanceof Token && (arg.type === "STRING" || arg.type === "NUMBER" || arg.type === "BOOL")) {
    const text = arg.value;
    let value: unknown;
    if (arg.type === "STRING") value = text.slice(1, -1);
    // An int has no digit limit here, as `evalExpr` reads it.
    else if (arg.type === "NUMBER") value = text.includes(".") ? Number(text) : BigInt(text);
    else value = text.toLowerCase() === "true";
    return literalType(value);
  }
  return inferExprType(arg, refType ?? (() => null), typeFn);
}

/** Return the member names a union typedef lists, or `[]` when they aren't all names. */
function unionMemberNames(name: string, registry: TypeRegistry): string[] {
  const definition = registry.get(name);
  if (definition?.kind !== "alias" || definition.target.kind !== "union") return [];
  const members = definition.target.members;
  if (!members.every((member): member is RefExpr => member.kind === "ref")) return [];
  return members.map((member) => member.name);
}
