/**
 * The type names `isinstance(x, T)` and `T(...)` constructors name, and their resolution.
 *
 * `isinstance` and constructors read their type from the evaluating node's `types` table.
 * Compose fills that table from the names a node's expressions use, resolved against the flow's
 * own `typedefs:` registry, so each flow (root, `defs:`, inline bodies, `uses:` children) sees
 * its own types. `isinstance` also takes a built-in type name (`int`, `list`, `Message`, ...); a
 * constructor takes only a `typedefs:` name. `isinstanceMembers` also says which members of a union
 * an `isinstance` check narrows it to.
 */

import { reprText } from "../../diagnostics.ts";
import {
  constructorNames,
  isinstanceTypeNames,
  parseCondition,
  unknownFunctionMessage,
} from "../../expr/expressions.ts";
import type { ExprNode } from "../../expr/grammar.ts";
import { scanTemplate, Span } from "../../expr/template.ts";
import { bindingSource, iterSourceStrings } from "../../nodes/contracts.ts";
import { type TypeRegistry, typeFor } from "../../typesys/types.ts";
import {
  canonicalUnionOrder,
  isMemoryType,
  type Type,
  TypeCheckError,
  typeEquals,
  typesOverlap,
} from "../../typesys/values.ts";

/** The built-in type names `isinstance` accepts besides a `typedefs:` name, each with its type text. */
const ISINSTANCE_BUILTIN_TYPES: ReadonlyMap<string, string> = new Map([
  ["str", "str"],
  ["int", "int"],
  ["float", "float"],
  ["bool", "bool"],
  ["date", "date"],
  ["datetime", "datetime"],
  ["object", "object"],
  // Any list, whatever its items.
  ["list", "list(Any)"],
  ["Message", "Message"],
]);

/**
 * Return the type names the `isinstance(x, T)` calls in `source` name.
 *
 * @param source A binding source (raw or carrier-wrapped) or a condition string. Every string
 *   nested in a value template is scanned; a literal names no types.
 * @param options.condition `true` parses `source` as one condition expression (`when:`,
 *   `while:`, an assert), where a bare `isinstance(x, T)` is an expression; `false` (the
 *   default) scans it as a template, where only `${...}` spans are expressions.
 * @returns The names in first-seen order, without duplicates.
 * @throws {ExpressionError} When `source` does not parse.
 *
 * @example
 * collectIsinstanceNames("${isinstance(d.output, Buy)}"); // ["Buy"]
 * collectIsinstanceNames("isinstance(x, int)", { condition: true }); // ["int"]
 */
export function collectIsinstanceNames(
  source: unknown,
  options: { condition?: boolean } = {},
): string[] {
  return collect(source, options.condition ?? false, isinstanceTypeNames);
}

/**
 * Return the names the `T(...)` constructor calls in `source` use.
 *
 * @param source A binding source or a condition string, as in `collectIsinstanceNames`.
 * @param options.condition `true` parses `source` as one condition expression; `false` (the
 *   default) scans it as a template.
 * @returns The names in first-seen order, without duplicates.
 * @throws {ExpressionError} When `source` does not parse.
 *
 * @example
 * collectConstructorNames("${Buy(d.output).conviction}"); // ["Buy"]
 */
export function collectConstructorNames(
  source: unknown,
  options: { condition?: boolean } = {},
): string[] {
  return collect(source, options.condition ?? false, constructorNames);
}

/**
 * Return the parsed expressions in `source`.
 *
 * @param source A binding source or a condition string, as in `collectIsinstanceNames`.
 * @param options.condition `true` parses `source` as one condition expression; `false` (the
 *   default) parses each `${...}` span of every string in it.
 * @returns The parse trees in source order.
 * @throws {ExpressionError} When `source` does not parse.
 */
export function expressionTrees(
  source: unknown,
  options: { condition?: boolean } = {},
): ExprNode[] {
  if (options.condition) {
    const value = bindingSource(source);
    return typeof value === "string" ? [parseCondition(value)] : [];
  }
  const trees: ExprNode[] = [];
  for (const [, text] of iterSourceStrings(source)) {
    for (const segment of scanTemplate(text)) {
      if (segment instanceof Span) trees.push(segment.tree);
    }
  }
  return trees;
}

/** Union `namesOf` over the expressions in `source`, in first-seen order. */
function collect(
  source: unknown,
  condition: boolean,
  namesOf: (tree: ExprNode) => string[],
): string[] {
  const names = new Set<string>();
  for (const tree of expressionTrees(source, { condition })) {
    for (const name of namesOf(tree)) names.add(name);
  }
  return [...names];
}

/**
 * Resolve one `isinstance` type name against a flow's `typedefs:` registry.
 *
 * @param name The bare type name written as the second `isinstance` argument: a `typedefs:` name,
 *   a scalar name, `object` (any record), `list` (any list) or `Message`.
 * @param registry The flow's `readTypedefs(...)` table.
 * @returns The resolved type.
 * @throws {TypeCheckError} When `name` is neither a `typedefs:` name nor a built-in type name, or
 *   names a host memory type: only the runner binds a memory, so no check narrows a value to one.
 */
export function resolveIsinstanceType(name: string, registry: TypeRegistry): Type {
  const builtin = ISINSTANCE_BUILTIN_TYPES.get(name);
  if (!registry.has(name) && builtin === undefined) {
    throw new TypeCheckError(
      `unknown type ${reprText(name)}; isinstance takes a typedefs: name or one of ` +
        [...ISINSTANCE_BUILTIN_TYPES.keys()].join(", "),
    );
  }
  const typ = typeFor(registry.has(name) ? name : builtin!, registry);
  if (isMemoryType(typ)) {
    throw new TypeCheckError(
      `\`${name}\` is a memory type; isinstance can't test for one (only the runner binds a memory)`,
    );
  }
  return typ;
}

/**
 * Resolve one constructor name against a flow's `typedefs:` registry.
 *
 * @param name The callee of a `T(...)` call that is no builtin.
 * @param registry The flow's `readTypedefs(...)` table.
 * @returns The resolved type.
 * @throws {TypeCheckError} When `name` is not a `typedefs:` name; a builtin type name (`int`)
 *   gets a hint that a constructor is no conversion.
 */
export function resolveConstructorType(name: string, registry: TypeRegistry): Type {
  if (!registry.has(name)) throw new TypeCheckError(unknownFunctionMessage(name));
  return typeFor(name, registry);
}

/**
 * Return the static-inference lookup of constructor types for one flow.
 *
 * @param registry The flow's `readTypedefs(...)` table.
 * @returns A lookup that maps a `typedefs:` name to its type and any other name to `null`.
 */
export function constructorTypeFn(registry: TypeRegistry): (name: string) => Type | null {
  const resolved = new Map<string, Type>();
  return (name) => {
    if (!registry.has(name)) return null;
    let typ = resolved.get(name);
    if (typ === undefined) {
      typ = typeFor(name, registry);
      resolved.set(name, typ);
    }
    return typ;
  };
}

/**
 * Return the members a check against `typ` names, ignoring its nullability: each member of a
 * union, else `typ` itself.
 */
export function checkedMembers(typ: Type): readonly Type[] {
  const strict = typ.replace({ nullable: false });
  return strict.members?.length ? strict.members : [strict];
}

/**
 * Return the members of a union `baseType` that pass `isinstance(x, typ)`, if known.
 *
 * The answer is the base members equal to a member of `typ` (ignoring nullability), and only
 * when no other base member could pass the check. It does not check whether every `baseType`
 * value already fits `typ`; callers check that first.
 *
 * @param baseType The type of the value being tested.
 * @param typ The type the check names. A union names each of its members.
 * @returns The shared members in canonical union order, or `null` when `baseType` is no union,
 *   shares no member with `typ`, or has another member that `typ` may admit.
 *
 * @example
 * For `Union[Buy, Hold, Sell]`, `Union[Hold, Sell, Cancel]` gives `[Hold, Sell]` and `Buy` gives
 * `[Buy]`; a `typ` whose member admits some `Buy` values but is no base member gives `null`.
 */
export function isinstanceMembers(baseType: Type, typ: Type): Type[] | null {
  const members = baseType.members;
  if (members === undefined) return null;
  const strict = typ.replace({ nullable: false });
  const wanted = checkedMembers(typ);
  const shared = members.filter((member) => wanted.some((w) => typeEquals(member, w)));
  // The runtime check is the write-boundary check, so a base member that is not a member of
  // `typ` but overlaps it can still pass. Leaving it out would be unsound (the branch could see
  // a value the answer excludes), and keeping it could make the answer less permissive than
  // `typ` (its values need not fit `typ`), so give no answer.
  const passing = members.filter((member) => typesOverlap(member, strict));
  if (
    shared.length > 0 &&
    shared.length === passing.length &&
    shared.every((member, i) => typeEquals(member, passing[i]!))
  ) {
    return canonicalUnionOrder(shared);
  }
  return null;
}
