/**
 * `${...}` evaluation plus conservative compile-time type/reference analysis.
 *
 * A resolver reads a reference's root (`VariablePool.readRoot`, or a field of a node's input
 * record) and the evaluator walks the field steps after it, so expressions read typed values and
 * can traverse into object outputs such as `${x.output.output.ratio}`.
 *
 * Grammar: comparisons (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over
 * references and literals, combined with `and` / `or` / `not`, conditionals and parentheses. The same parsed
 * tree feeds the evaluator, conservative type inference, and the reference walkers. Conditions
 * route through the one unified engine; a bare reference with no comparison operator is a
 * truthiness test (not an error). `isinstance(x, T)` checks a value against a type named in the
 * `types` table every evaluation entry point accepts, and a constructor `T(x)` /
 * `T(a=..., b=...)` checks (or builds and checks) a value of that type.
 */

import { compareCodeUnits, getCloseMatches, reprText } from "../diagnostics.ts";
import {
  BUILTIN_TYPE_NAMES,
  buildValueAs,
  FileRef,
  normalizeFiles,
  Type,
  TypeCheckError,
  unionField,
  type Value,
  ValueKind,
  type ValueRecord,
  type VariablePool,
} from "../typesys/index.ts";
import type { RootValue } from "../typesys/pool.ts";
import {
  fieldOf,
  fileFieldType,
  fromHost,
  isListKind,
  isRecord,
  joinedType,
  listKindForElement,
  valueRepr,
} from "../typesys/values.ts";
import {
  type ArgType,
  bindArguments,
  builtinResultType,
  floatParts,
  isShipped,
  kindName,
  roundQuotient,
  TEMPLATE_FNS,
  type TemplateFn,
  truthy,
} from "./builtins.ts";
import {
  callSuffixOf,
  type ExprNode,
  ExpressionError,
  ISINSTANCE,
  iterSubtreesTopdown,
  parseExpr,
  Token,
  Tree,
  valueArgs,
  wholeSpanInterior,
} from "./grammar.ts";

export { ExpressionError, ISINSTANCE, valueArgs };

/** A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError). */
export class RequiredError extends ExpressionError {
  override name = "RequiredError";
}

/** A `${T(...)}` constructor's value does not fit `T` (the binder's `binding.type`). */
export class ConstructorTypeError extends ExpressionError {
  override name = "ConstructorTypeError";
}

/**
 * Return the hint for calling a builtin type name that is no builtin, else `""`.
 *
 * `bool(x)` reads like a conversion, but constructors are `typedefs:` names that only check;
 * only `str`, `int` and `float` are conversions, as builtins.
 *
 * @param name The callee.
 * @returns `"; types in expressions are typedef names; ..."` for a builtin type name (`bool`,
 *   `list`, `Optional`) no builtin has, else empty.
 */
export function conversionHint(name: string): string {
  if (!BUILTIN_TYPE_NAMES.has(name) || TEMPLATE_FNS.has(name)) return "";
  return `; types in expressions are typedef names; \`${name}(...)\` is not a conversion`;
}

/**
 * Return the error text for a call to `name`, which is no builtin or type.
 *
 * @param name The callee.
 * @returns The message, with `conversionHint` for a builtin type name, else the closest builtin
 *   when one is close (`getCloseMatches`'s cutoff): `unknown expression function 'lenght'; did you
 *   mean 'len'?`.
 */
export function unknownFunctionMessage(name: string): string {
  const message = `unknown expression function ${reprText(name)}`;
  const hint = conversionHint(name);
  if (hint !== "") return message + hint;
  const [close] = getCloseMatches(name, [...TEMPLATE_FNS.keys(), ISINSTANCE], 1);
  return close === undefined ? message : `${message}; did you mean ${reprText(close)}?`;
}

/**
 * Return why a call to constructor `name` has the wrong arguments, else `null`.
 *
 * A constructor takes keyword fields (`Buy(action="buy", conviction=x)`) or exactly one
 * positional value (`Buy(x)`), never both.
 *
 * @param name The constructor name.
 * @param callSuffix The call's parsed `call_suffix`.
 * @returns The message, or `null` for a well-formed call.
 */
export function constructorShapeError(name: string, callSuffix: Tree): string | null {
  const args = callSuffix.children as Tree[];
  const keywords = args.filter((arg) => arg.children.length === 2).length;
  const positional = args.length - keywords;
  if (positional > 1 || (positional > 0 && keywords > 0)) {
    return (
      `${name}(...) takes keyword fields, as in ${name}(name=value, ...), or one ` +
      `value to check, as in ${name}(x)`
    );
  }
  return null;
}

/** Reads a reference's root, given all its names; the evaluator walks the names after the root. */
export type RootResolver = (names: readonly string[]) => RootValue;

/** The resolver for an evaluation with nothing in scope: every root is absent. */
export const MISSING_ROOT: RootResolver = () => ({ used: 1, present: false, value: null, why: null });

/** The declared type at the end of a reference's names (a root, or a root and fields); `null` is untyped. */
export type DeclaredTypes = (names: readonly string[]) => Type | null;

/**
 * A value's declared members, each non-null: none for an untyped type (`Any`, `object`, a file,
 * or unknown), a union's members, else the type itself.
 */
function declaredOf(typ: Type | null): readonly Type[] {
  if (typ === null || typ.kind === ValueKind.ANY || typ.kind === ValueKind.FILE) return [];
  if (typ.members !== undefined) return typ.members.flatMap(declaredOf);
  if (typ.kind === ValueKind.OBJECT && typ.fields === undefined) return [];
  return [typ.nullable ? typ.replace({ nullable: false }) : typ];
}

/** No declared members: an untyped value. */
const NO_MEMBERS = (): readonly Type[] => [];

/** The declared members of the receiver of `steps[index]`: `membersOf()` projected through the steps before it. */
function membersBefore(membersOf: () => readonly Type[], steps: readonly RefStep[], index: number): readonly Type[] {
  let members = membersOf();
  for (let i = 0; i < index; i++) members = projectDeclared(members, steps[i]!.name);
  return members;
}

/** Whether any of `members` declares field `name`. */
function declares(members: readonly Type[], name: string): boolean {
  return members.some((member) => member.fields?.has(name) === true);
}

/** The members after reading `name`: the declared members of the field in each member that declares it. */
function projectDeclared(members: readonly Type[], name: string): readonly Type[] {
  return members.flatMap((member) => declaredOf(member.fields?.get(name) ?? null));
}

/**
 * Return `typ` projected through `names` by declared fields: a step strips `Optional`; a union
 * projects through the members that declare the name, joined; from `Any`, `object`, an opaque
 * field or a file onward, everything is untyped (`null`).
 */
export function declaredAt(typ: Type | null, names: readonly string[]): Type | null {
  let members = declaredOf(typ);
  for (const name of names) members = projectDeclared(members, name);
  if (members.length === 0) return null;
  if (members.length === 1) return members[0]!;
  if (members.every((member) => member.fields !== undefined)) {
    return new Type({ kind: ValueKind.OBJECT, members: members });
  }
  return joinedType(members);
}

/** The `DeclaredTypes` of a record scope: a local's type from `types`, then its fields (`declaredAt`). */
export function localDeclared(types: ReadonlyMap<string, Type | null> | null): DeclaredTypes {
  return (names) => declaredAt(types?.get(names[0]!) ?? null, names.slice(1));
}

/** Throw when a reference has an empty name. */
function checkNames(names: readonly string[]): void {
  if (names.some((name) => !name)) throw new ExpressionError(`invalid variable path: '\${${names.join(".")}}'`);
}

/**
 * Read a reference's root from the pool (`VariablePool.readRoot`).
 *
 * @throws {ExpressionError} If a name is empty.
 */
export function resolveReference(names: readonly string[], pool: VariablePool): RootValue {
  checkNames(names);
  return pool.readRoot(names);
}

/**
 * Read a reference's root from a node's bound input record: the head, as an own field of
 * `record`, so a read never reaches a JS property such as `constructor`.
 *
 * Bare local-input refs only. The evaluator walks the steps after the head. Compose rejects
 * undeclared names everywhere, and field reads a known type can't have in bindings, CASE
 * conditions, LOOP predicates, and flow outputs. Asserts check less (a node assert's reads of its
 * inputs and a flow assert's `${input.X.f}` check only the head name).
 *
 * @throws {ExpressionError} If a name is empty.
 */
export function resolveInRecord(names: readonly string[], record: ValueRecord): RootValue {
  checkNames(names);
  const [present, value] = fieldOf(record, names[0]!);
  return { used: 1, present, value, why: present ? null : `input '${names[0]}' is not bound` };
}

/**
 * Apply a comparison operator to two resolved values.
 *
 * `==` and `!=` compare deeply (`equal`). `in` looks for text in text, an equal item in a
 * list, or an own key in a record. An ordered comparison with a `null` operand is `false`, so a
 * missing value routes to `else:`; otherwise it orders two numbers by exact value or two texts
 * by UTF-16 code units.
 *
 * @param op `==`, `!=`, `<`, `<=`, `>`, `>=`, `in` or `not in`.
 * @throws {ExpressionError} For `in` with a right side that isn't text, a list or a record, or a
 *   non-text left side in text; or ordering two values that aren't both numbers or both text.
 */
function evalComparison(op: string, lhs: Value, rhs: Value): boolean {
  switch (op) {
    case "==":
      return equal(lhs, rhs);
    case "!=":
      return !equal(lhs, rhs);
    case "in":
    case "not in": {
      let contained: boolean;
      if (typeof rhs === "string" && typeof lhs === "string") contained = rhs.includes(lhs);
      else if (Array.isArray(rhs)) contained = rhs.some((item) => equal(lhs, item));
      else if (isRecord(rhs)) contained = typeof lhs === "string" && Object.hasOwn(rhs, lhs);
      else throw operandKindsError(op, lhs, rhs);
      return op === "in" ? contained : !contained;
    }
  }
  if (lhs === null || rhs === null) return false;
  const x = numeric(lhs);
  const y = numeric(rhs);
  let order: number;
  // JS compares a bigint with a number by exact value.
  if (x !== null && y !== null) order = x < y ? -1 : x > y ? 1 : 0;
  else if (typeof lhs === "string" && typeof rhs === "string") order = compareCodeUnits(lhs, rhs);
  else throw operandKindsError(op, lhs, rhs);
  switch (op) {
    case "<":
      return order < 0;
    case "<=":
      return order <= 0;
    case ">":
      return order > 0;
    case ">=":
      return order >= 0;
  }
  throw new ExpressionError(`unsupported operator: ${reprText(op)}`);
}

/**
 * Return whether two values are equal: numbers by exact value (a bool as 0 or 1), texts by code
 * units, lists item by item, records by own keys and values, files by `uri`, `mime` and `name`.
 * Any other pair is unequal (`null` equals only `null`).
 */
function equal(a: Value, b: Value): boolean {
  const x = numeric(a);
  const y = numeric(b);
  // `==` between a bigint and a number compares their exact values.
  if (x !== null && y !== null) return x == y;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((item, i) => equal(item, b[i]!));
  }
  if (FileRef.isFileRef(a) && FileRef.isFileRef(b)) {
    return a.uri === b.uri && a.mime === b.mime && a.name === b.name;
  }
  if (isRecord(a) && isRecord(b)) {
    const keys = Object.keys(a);
    return (
      keys.length === Object.keys(b).length && keys.every((key) => Object.hasOwn(b, key) && equal(a[key]!, b[key]!))
    );
  }
  return a === b;
}

/** Return a number as operators read it: an int or a bool (0 or 1) as a `bigint`, a float as is; else `null`. */
function numeric(value: Value): bigint | number | null {
  if (typeof value === "boolean") return value ? 1n : 0n;
  return typeof value === "bigint" || typeof value === "number" ? value : null;
}

/** Return the error for `lhs op rhs` on a pair of operand kinds `op` doesn't take. */
function operandKindsError(op: string, lhs: Value, rhs: Value): ExpressionError {
  return new ExpressionError(
    `type error evaluating ${valueRepr(lhs)} ${op} ${valueRepr(rhs)}: ` +
      `unsupported operand kinds for \`${op}\`: \`${kindName(lhs)}\` and \`${kindName(rhs)}\``,
  );
}

/**
 * Parse an expression-field value (`when:`/`on:`/an `asserts:` entry) via the one unified
 * grammar, honouring the both-spellings entry rule.
 *
 * An expression field may be written two ways, which parse (and later evaluate) identically:
 *
 * - a bare expression `a > 5` (no braces), the canonical one;
 * - a whole `${a > 5}` (braces around the whole expression).
 *
 * If the whole trimmed value is a single `${...}` span, the outer braces are stripped and the
 * interior parsed; otherwise the whole value is parsed directly. A `${...}` inside the expression
 * (`${a} > 5`) is an error that shows the bare rewrite.
 *
 * Every call to a builtin (`TEMPLATE_FNS`) must pass arguments it takes. Any other callee except
 * the `isinstance` special form is a `T(...)` constructor: with `types`, it must name a type in
 * it; without, its name is left to the caller (compose resolves it against the flow's
 * `typedefs:`). The check is static: it applies even to a call `and`/`or` would never evaluate.
 *
 * @param expression The field value, in either spelling.
 * @param types The type table constructors are looked up in (only its names are read), or
 *   `null` when it is not known yet.
 * @returns The parsed expression.
 * @throws {ExpressionError} If the value does not parse or holds a `${...}` inside it, calls a
 *   builtin with arguments it doesn't take, or, with `types`, calls a function that is neither a
 *   builtin nor a type in `types`.
 */
export function parseCondition(expression: string, types: ReadonlyMap<string, unknown> | null = null): ExprNode {
  const interior = wholeSpanInterior(expression);
  const tree = parseExpr(interior === null ? expression.trim() : interior);
  checkCalls(tree, types);
  return tree;
}

/**
 * Require every builtin call in a parsed expression to pass arguments it takes.
 *
 * `isinstance` is skipped: the grammar already checks its shape. Any other callee that is no
 * builtin is a `T(...)` constructor, checked against `types` when given.
 *
 * @param tree A tree from `parseExpr`.
 * @param types The type table constructor names must be in, or `null` (the default) to leave
 *   them to the caller.
 * @throws {ExpressionError} If a builtin's parameters can't bind the call's positional count and
 *   keyword names, or, with `types`, a callee is neither a builtin nor in it.
 */
export function checkCalls(tree: ExprNode, types: ReadonlyMap<string, unknown> | null = null): void {
  if (!(tree instanceof Tree)) return;
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall") continue;
    const callSuffix = callSuffixOf(node);
    const callee = String(node.children[0]);
    if (callSuffix === undefined || callee === ISINSTANCE) continue;
    const fn = TEMPLATE_FNS.get(callee);
    if (fn === undefined) {
      // A constructor: one unknown-function message, here or where compose resolves the name
      // against the flow's typedefs.
      if (types !== null && !types.has(callee)) throw new ExpressionError(unknownFunctionMessage(callee));
      continue;
    }
    const args = callSuffix.children as Tree[];
    const positional = args.filter((arg) => arg.children.length === 1).map(() => null);
    const keywords = new Map(
      args.filter((arg) => arg.children.length === 2).map((arg) => [String(arg.children[0]), null]),
    );
    const bound = bindArguments(fn, positional, keywords);
    if (typeof bound === "string") {
      throw new ExpressionError(`expression function ${reprText(callee)} takes (${parameterList(fn)}): ${bound}`);
    }
  }
}

/** Render a builtin's parameters for a message: `value, indent=2`, each default in its text form. */
function parameterList(fn: TemplateFn): string {
  return fn.params
    .map((p) => (Object.hasOwn(p, "default") ? `${p.name}=${valueRepr(p.default)}` : p.name))
    .join(", ");
}

/**
 * Return the reference paths a condition-field value reads (`when:`/`until:`/`while:`/`asserts:`).
 *
 * The condition analog of the template and prompt reference walks: parse the whole value as one
 * expression (via `parseCondition`, honouring both spellings, bare `a > 5` and whole `${a > 5}`)
 * and collect its reference leaves via the shared `exprRefs`. Every compile-time consumer of a
 * condition's references (flow/node `asserts:`, a loop `while:`/`until:` predicate, a searched
 * `case` `when:`) routes here, so both spellings extract identically, unlike a flat `${...}`
 * regex, which reads a whole span as one bogus path and a bare form as no references at all.
 *
 * @param expression The condition source: any expression the grammar accepts, in any spelling.
 * @returns The reference paths (dotted, `#`/`/` preserved) in source order, not deduped.
 * @throws {ExpressionError} On a malformed condition, or one that calls a builtin with arguments it
 *   doesn't take (the caller frames it as a located load error). Any other callee is taken as a
 *   constructor and left to the caller's typedef check, as `parseCondition` with no `types` does.
 */
export function conditionRefs(expression: string): string[] {
  return exprRefs(parseCondition(expression));
}

/**
 * Parse and evaluate a `when:`/`asserts:` expression on the unified engine, resolving each
 * reference via `resolve` and each `isinstance` type name via `types`; the result is a boolean.
 *
 * Runs `evalExpr` in `CONDITION_FALSY` mode: a missing reference is `null` and so falsy through
 * comparisons, which lets a CASE's `else:` route handle absent values. The truthiness fold of
 * `and`/`or` becomes a plain boolean here, so a predicate always gives `true` or `false`.
 */
function evaluate(
  expression: string,
  resolve: RootResolver,
  types: ReadonlyMap<string, Type> | null,
  declared: DeclaredTypes | null = null,
): boolean {
  const tree = parseCondition(expression, types ?? new Map());
  return truthy(evalExpr(tree, resolve, { mode: ResolveMode.CONDITION_FALSY, types, declared }));
}

/**
 * Evaluate a `when:` boolean expression against the variable pool.
 *
 * Parses and folds the expression on the one unified engine (comparisons, arithmetic over
 * references and literals, `and` / `or` / `not`) and gives a single boolean. This is the public
 * pool-based path; record-scoped CASE and LOOP predicates use `evaluateWhenRecord`.
 *
 * The value may be written two ways, which evaluate identically: a bare expression `a > 5` or a
 * whole `${a > 5}`. A bare reference with no comparison operator is a truthiness test, not an
 * error.
 *
 * Example: `evaluateWhen("score.output >= 0.5 and flag.output", pool)`.
 *
 * @param expression The `when:` source: any expression the grammar accepts, in either spelling.
 * @param pool The variable pool each reference resolves against; a miss is a falsy `null`.
 * @param types The type table `isinstance(x, T)` and a constructor look `T` up in; `null` means
 *   no types.
 * @returns The truth value of the expression.
 * @throws {ExpressionError} If the expression is malformed, references an invalid path, applies
 *   an operator to operand kinds it doesn't take, or names a type absent from `types`. A call to
 *   a function that isn't a builtin, or to a builtin with arguments it doesn't take, throws even
 *   where `and`/`or` would not evaluate the call.
 */
export function evaluateWhen(
  expression: string,
  pool: VariablePool,
  types: ReadonlyMap<string, Type> | null = null,
): boolean {
  return evaluate(expression, (names) => resolveReference(names, pool), types);
}

/**
 * Evaluate a strict-CASE `when:` against the node's bound input record.
 *
 * `name` reads `record` (`resolveInRecord`) and `name.path` walks its fields by the step table
 * (see `evalExpr`). A missing `name` is `null`, compared as a plain value: `==` and ordered
 * comparisons against a non-null value are `false`, and `!=` is `true`. A bare missing `name` is
 * falsy, which gives `when:` its missing-is-falsy behavior. A `null` left side of `in`/`not in`
 * is an ordinary value (`miss in [...]` is `false`); a `null` right side throws, since `null` is
 * no container. Pool namespaces are not in scope (they belong in the node's input `from:`
 * bindings).
 *
 * @param types The type table `isinstance(x, T)` and a constructor look `T` up in; `null` means
 *   no types.
 * @param declared The declared type of each input, given a reference's names (`localDeclared`);
 *   `null` (the default) leaves every field read undeclared.
 * @throws {ExpressionError} On a field read the step table rejects, and as `evaluateWhen` does: a
 *   call to a function that isn't a builtin, or to a builtin with arguments it doesn't take,
 *   throws even where `and`/`or` would not evaluate the call.
 */
export function evaluateWhenRecord(
  expression: string,
  record: ValueRecord,
  types: ReadonlyMap<string, Type> | null = null,
  declared: DeclaredTypes | null = null,
): boolean {
  return evaluate(expression, (names) => resolveInRecord(names, record), types, declared);
}

// --------------------------------------------------------------------------------------- //
// The one evaluator: `evalExpr` walks a `parseExpr` tree with a `resolve` callable. It is the
// single `${...}` evaluator: the binding path (coalesce/default/required), the
// `when:`/`asserts:` boolean path, and the prompt builtin-call path all parse with `parseExpr`
// and evaluate here.
// --------------------------------------------------------------------------------------- //

/**
 * How a missing root (one `resolve` reads as absent, with no step after it) is treated. A field
 * read follows the step table in every mode (see `evalExpr`).
 *
 * - `STRICT_RAISE`: a missing root is an error: throw `ExpressionError`. For a context where
 *   every reference must be present (a strict render). A miss on a non-final operand of `|`,
 *   `:-`, or `:?` is still `null`, so the operator fires.
 * - `BINDING_NONE`: a missing root is `null`, the binding contract (a coalesce or default may
 *   then fire, or the whole value is `null`).
 * - `CONDITION_FALSY`: a missing root is `null`, which is falsy through comparisons, so a
 *   `when:` treats a miss as false. The same `null` as `BINDING_NONE`; the distinct name marks
 *   the intent at the call site (a `when:`/`asserts:` predicate).
 */
export const ResolveMode = Object.freeze({
  STRICT_RAISE: "strict-raise",
  BINDING_NONE: "binding-none",
  CONDITION_FALSY: "condition-falsy",
} as const);

export type ResolveMode = (typeof ResolveMode)[keyof typeof ResolveMode];

/**
 * Marker for an unresolved reference, telling a `refcall` miss apart from a resolved `null`, so
 * `default`/`required` can fire on a genuine miss. Non-strict modes turn it into `null` at the
 * operator boundary; strict mode throws.
 */
const MISSING: unique symbol = Symbol("missing");

/** What walking a subtree gives: a value, or `MISSING` for an unresolved reference. */
type Walked = Value | typeof MISSING;

/**
 * Apply one operator of an arithmetic chain to two resolved values.
 *
 * `+` concatenates two texts or two lists. Two ints (a bool counts as 0 or 1) give an exact
 * `bigint` for `+ - * %` and `**` with a non-negative exponent; `%` takes the sign of the left.
 * `/` always gives the float nearest the exact quotient. Any float operand, or `**` with a
 * negative int exponent, gives IEEE arithmetic on floats, where an int past the float range is
 * `±Infinity`.
 *
 * @param op `+`, `-`, `*`, `/`, `%` or `**`.
 * @param lhs The left value: the running value of a left fold, or an operand.
 * @param rhs The right value.
 * @throws {ExpressionError} For text on the left of `%`, operand kinds `op` doesn't take, a zero
 *   divisor or `0 ** negative`, a float result that would be NaN, or an int result the runtime
 *   can't hold; the message shows the operands as given.
 */
function applyStep(op: string, lhs: Value, rhs: Value): Value {
  if (op === "%" && typeof lhs === "string") {
    throw new ExpressionError("text formatting with `%` was removed; use a template");
  }
  if (op === "+") {
    if (typeof lhs === "string" && typeof rhs === "string") return lhs + rhs;
    if (Array.isArray(lhs) && Array.isArray(rhs)) return [...lhs, ...rhs];
  }
  const x = numeric(lhs);
  const y = numeric(rhs);
  if (x === null || y === null) throw operandKindsError(op, lhs, rhs);
  const divisionByZero = () => new ExpressionError(`division by zero: ${valueRepr(lhs)} ${op} ${valueRepr(rhs)}`);
  if (typeof x === "bigint" && typeof y === "bigint" && !(op === "**" && y < 0n)) {
    try {
      switch (op) {
        case "+":
          return x + y;
        case "-":
          return x - y;
        case "*":
          return x * y;
        case "/":
          if (y === 0n) throw divisionByZero();
          return trueDivide(x, y);
        case "%":
          if (y === 0n) throw divisionByZero();
          return x % y;
        case "**":
          // These bases have an exact power at any exponent, which `x ** y` may refuse to build.
          if (x === 0n) return y === 0n ? 1n : 0n;
          if (x === 1n) return 1n;
          if (x === -1n) return y % 2n === 0n ? 1n : -1n;
          return x ** y;
      }
    } catch (error) {
      // The runtime refuses a bigint too large with a RangeError; yian's own errors are
      // RangeErrors too and keep their messages.
      if (error instanceof RangeError && !(error instanceof ExpressionError)) {
        throw new ExpressionError("integer result is too large for this runtime");
      }
      throw error;
    }
  }
  const a = Number(x);
  const b = Number(y);
  let result: number;
  switch (op) {
    case "+":
      result = a + b;
      break;
    case "-":
      result = a - b;
      break;
    case "*":
      result = a * b;
      break;
    case "/":
      if (b === 0) throw divisionByZero();
      result = a / b;
      break;
    case "%":
      if (b === 0) throw divisionByZero();
      result = a % b;
      break;
    case "**":
      if (a === 0 && b < 0) throw divisionByZero();
      // IEEE `pow` gives 1 here; JS gives NaN.
      if (Math.abs(a) === 1 && !Number.isFinite(b)) result = 1;
      else if (Number.isFinite(a) && a !== 0 && Number.isInteger(b) && Math.abs(b) <= 1100) result = floatPow(a, b);
      else result = a ** b;
      break;
    default:
      throw new ExpressionError(`unsupported operator: ${reprText(op)}`);
  }
  if (Number.isNaN(result)) {
    throw new ExpressionError(`${op} on ${valueRepr(lhs)} and ${valueRepr(rhs)} has no numeric result`);
  }
  return result;
}

const EXACT_LIMIT = 2n ** 53n;

/** Return the float nearest `x / y` (ties to even); `0n / -5n` is `-0`. */
function trueDivide(x: bigint, y: bigint): number {
  // Both convert exactly, so the float division is correctly rounded. Beyond 2^53,
  // `Number(x) / Number(y)` would round twice, so the quotient is rounded once here.
  if (-EXACT_LIMIT <= x && x <= EXACT_LIMIT && -EXACT_LIMIT <= y && y <= EXACT_LIMIT) return Number(x) / Number(y);
  const quotient = roundQuotient(x < 0n ? -x : x, y < 0n ? -y : y);
  return x < 0n !== y < 0n ? -quotient : quotient;
}

/**
 * Return the float `x ** n` correctly rounded, for a finite non-zero `x` and an integral `n`
 * with `|n| <= 1100` (the bound keeps the exact work small).
 *
 * JS's `**` is off in the last bit for some of these (`10 ** -4`), and Bun and Node disagree.
 */
function floatPow(x: number, n: number): number {
  if (n === 0) return 1;
  const [m, e] = floatParts(x);
  const power = m ** BigInt(Math.abs(n));
  let num = n > 0 ? power : 1n;
  let den = n > 0 ? 1n : power;
  if (e * n >= 0) num <<= BigInt(e * n);
  else den <<= BigInt(-e * n);
  const result = roundQuotient(num, den);
  return x < 0 && n % 2 !== 0 ? -result : result;
}

/**
 * Return `-value`: an int (a bool as 0 or 1) as a `bigint`, a float as a float.
 *
 * @throws {ExpressionError} If `value` is not a number.
 */
function minus(value: Value): Value {
  const x = numeric(value);
  if (x === null) {
    throw new ExpressionError(
      `unary minus on non-number ${valueRepr(value)}: unsupported operand kind for \`-\`: \`${kindName(value)}\``,
    );
  }
  return -x;
}

// The operator tokens between the operands of a `sum` / `product` / `power` chain. A literal
// operand is a `Token` too, so operators are told apart by these types, never by `Token`.
const ARITH_OP_TOKEN_TYPES: ReadonlySet<string> = new Set(["ADD_OP", "MUL_OP", "POW_OP"]);

/**
 * Split an arithmetic chain node into its operands and operator symbols.
 *
 * @param node A `sum`, `product` or `power` node from `parseExpr`.
 * @returns The operand subtrees and tokens in source order, and the operator symbols between
 *   them (one fewer than the operands).
 */
function splitChain(node: Tree): [ExprNode[], string[]] {
  const operands: ExprNode[] = [];
  const operators: string[] = [];
  for (const child of node.children) {
    if (child instanceof Token && ARITH_OP_TOKEN_TYPES.has(child.type)) operators.push(child.value);
    else operands.push(child);
  }
  return [operands, operators];
}

/**
 * Name the base of a field read in a message: a reference or variable as written, a subscript as
 * `base[...]`, a field read as its base and steps, a literal as written, anything else `(...)`.
 */
export function subscriptText(node: ExprNode): string {
  if (node instanceof Token) return node.value;
  if ((node.data === "refcall" && callSuffixOf(node) === undefined) || node.data === "comp_var") {
    return referenceText(node);
  }
  if (node.data === "subscript") return `${subscriptText(node.children[0]!)}[...]`;
  if (node.data === "field_read") return subscriptText(node.children[0]!) + stepsText(splitReference(node).steps);
  return "(...)";
}

/** The error for indexing or slicing a value that is no list or text. */
function subscriptKindError(doing: "indexing" | "slicing", base: Value): ExpressionError {
  const hint = isRecord(base) ? "; read a record's field with a dot, as in `r.name`" : "";
  return new ExpressionError(`${doing} takes a list or text, got ${kindName(base)}${hint}`);
}

/** Name a subscript's base in a miss path: a literal's text, a comprehension variable's name, else `(...)`. */
function sliceBaseText(node: ExprNode): string {
  if (node instanceof Token) return node.value;
  if (node.data === "comp_var" && node.children.length === 1) return String(node.children[0]);
  return "(...)";
}

// The operator keyword tokens among a boolean combinator's operands. An operand may also be a
// top-level `Token` (a bare literal), so only these token types are skipped.
const OP_TOKEN_TYPES: ReadonlySet<string> = new Set(["NOT", "AND", "OR"]);

/**
 * Recursive walker for a `parseExpr` tree, parameterised by `resolve`, `item` and `mode`.
 * Hand-written so coalesce / default / required can short-circuit: the fallback is evaluated
 * only when the head misses. A comprehension runs its clauses in a loop rather than by
 * recursion, and keeps its variables' items while it runs.
 *
 * A reference evaluates to `MISSING` on an unresolved path; every value consumer (operators,
 * comparisons, the top-level entry) maps `MISSING` to `null` in the non-strict modes or throws
 * in `STRICT_RAISE`.
 */
class EvalExpr {
  readonly #resolve: RootResolver;
  readonly #item: Value | undefined;
  readonly #mode: ResolveMode;
  readonly #types: ReadonlyMap<string, Type> | null;
  readonly #declared: DeclaredTypes | null;
  readonly #itemType: Type | null;
  #missPath: string | null = null;
  // Set when the last miss is an index out of range: only `|`, `:-` and `:?` take that miss, and
  // every other consumer throws this error, in every mode.
  #outOfRange: ExpressionError | null = null;
  // Comprehension variables in scope: name -> its item and its declared type (the element type of
  // its un-narrowed iterable).
  readonly #vars = new Map<string, { value: Value; type: Type | null }>();

  constructor(
    resolve: RootResolver,
    item: Value | undefined,
    mode: ResolveMode,
    types: ReadonlyMap<string, Type> | null,
    declared: DeclaredTypes | null,
    itemType: Type | null,
  ) {
    this.#resolve = resolve;
    this.#item = item;
    this.#mode = mode;
    this.#types = types;
    this.#declared = declared;
    this.#itemType = itemType;
  }

  // --- top-level entry: turn a `MISSING` head into a value per mode --- //
  eval(node: ExprNode): Value {
    return this.asValue(this.walk(node));
  }

  /**
   * Turn `MISSING` into `null` in the non-strict modes; throw in `STRICT_RAISE`, and in every mode
   * for an index out of range.
   */
  asValue(value: Walked): Value {
    if (value === MISSING) {
      if (this.#outOfRange !== null) throw this.#outOfRange;
      if (this.#mode === ResolveMode.STRICT_RAISE) {
        const where = this.#missPath !== null ? ` \${${this.#missPath}}` : "";
        throw new ExpressionError(`unresolved reference${where} in expression`);
      }
      return null;
    }
    return value;
  }

  /**
   * Turn `MISSING` into `null` in every mode: a `|` / `:-` / `:?` operand, whose miss its operator
   * handles. A handled miss's path is dropped, so a later error can't name it.
   */
  orNone(value: Walked): Value {
    if (value !== MISSING) return value;
    this.#missPath = null;
    this.#outOfRange = null;
    return null;
  }

  /**
   * Turn `MISSING` into `null` for an operand that reports null itself (a comprehension iterable,
   * a slice bound, an index); an index out of range throws its own error instead.
   */
  orNull(value: Walked): Value {
    if (value === MISSING && this.#outOfRange !== null) throw this.#outOfRange;
    return this.orNone(value);
  }

  /** Record a miss at `path`, replacing any earlier one. */
  missAt(path: string): typeof MISSING {
    this.#missPath = path;
    this.#outOfRange = null;
    return MISSING;
  }

  // --- dispatch --- //
  /** Evaluate one subtree: the single hook every subtree goes through. */
  walk(node: ExprNode): Walked {
    if (node instanceof Token) return this.token(node);
    switch (node.data) {
      case "refcall":
        return this.refcall(node);
      case "coalesce":
        return this.coalesce(node);
      case "default_expr":
        return this.defaultExpr(node);
      case "required_expr":
        return this.requiredExpr(node);
      case "if_expr":
        return this.ifExpr(node);
      case "negate":
        return this.negate(node);
      case "or_expr":
        return this.orExpr(node);
      case "and_expr":
        return this.andExpr(node);
      case "compare":
        return this.comparison(node);
      case "compare_in":
        return this.comparison(node, "in");
      case "compare_notin":
        return this.comparison(node, "not in");
      case "sum":
      case "product":
        return this.sum(node);
      case "power":
        return this.power(node);
      case "neg":
        return this.neg(node);
      case "list_lit":
        return this.listLit(node);
      case "subscript":
        return this.subscript(node);
      case "field_read":
        return this.fieldRead(node);
      case "list_comp":
        return this.listComp(node);
      case "comp_var":
        return this.compVar(node);
    }
    throw new ExpressionError(`unsupported expression node ${reprText(node.data)}`);
  }

  /** A lone literal that inlined up to a top-level token. */
  token(tok: Token): Value {
    switch (tok.type) {
      case "NUMBER":
        return tok.value.includes(".") ? Number(tok.value) : BigInt(tok.value);
      case "STRING":
        return tok.value.slice(1, -1);
      case "BOOL":
        return tok.value === "true";
      case "NULL":
        return null;
      case "INF":
        return Infinity;
    }
    throw new ExpressionError(`unexpected token ${tok.type} ${reprText(tok.value)}`);
  }

  // --- references / calls (`refcall`) --- //
  /**
   * A `refcall` is a bare `NAME` and `trailer`s (a reference), or a `NAME`, a `call_suffix`
   * and trailing `trailer`s (a call, with dotted access on its result).
   *
   * A call is `isinstance(x, T)`, a `T(...)` constructor, or a builtin. A `.field` walk on its
   * result that misses is `MISSING`, the same as a reference's dotted miss, so every value
   * consumer treats the two alike (strict throws, the others read `null`).
   */
  refcall(node: Tree): Walked {
    const { head: name, steps } = splitReference(node);
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) return this.reference(name, steps, referenceText(node));
    const constructor = this.isConstructor(name);
    let value: Walked;
    if (name === ISINSTANCE) value = this.isinstanceCall(callSuffix);
    else if (constructor) value = this.construct(name, callSuffix);
    else value = this.callBuiltin(name, callSuffix);
    // A constructor's value has its type, as `inferExprType` types it.
    const membersOf = constructor ? () => declaredOf(this.#types!.get(name)!) : NO_MEMBERS;
    [value] = this.walkSteps(value, membersOf, steps, `${name}(...)`, null);
    return value;
  }

  /**
   * Read a reference: its root, then its steps by the step table (`walkSteps`). An `item`-headed
   * one reads the MAP-body `item` scope; any other root comes from `resolve`. Only a missing root
   * with no steps after it is `MISSING`.
   *
   * @param head The reference's first name.
   * @param steps Its steps as written.
   * @param text The read as written (`?.` included), for miss messages.
   * @throws {ExpressionError} When a step raises (see `walkSteps`).
   */
  reference(head: string, steps: readonly RefStep[], text: string): Walked {
    const stepNames = steps.map((step) => step.name);
    if (head === "item") {
      // MAP-body-local scope (not a resolver head); an absent item means no item scope.
      const item = this.#item;
      const membersOf = () => declaredOf(this.#itemType);
      const [walked] = this.walkSteps(item === undefined ? MISSING : item, membersOf, steps, "item", null);
      return walked === MISSING ? this.missAt(text) : walked;
    }
    const read = this.#resolve([head, ...stepNames]);
    const rootText = head + stepsText(steps.slice(0, read.used - 1));
    const rest = steps.slice(read.used - 1);
    // The declared type is looked up only when a record lacks a key.
    const membersOf = () => declaredOf(this.#declared?.([head, ...stepNames.slice(0, read.used - 1)]) ?? null);
    const [value] = this.walkSteps(read.present ? read.value : MISSING, membersOf, rest, rootText, read.why);
    return value === MISSING ? this.missAt(text) : value;
  }

  /**
   * Walk `steps` from a root's or a call's value by the step table, given the value's declared
   * members (`membersOf`, called only when a record lacks a key). `MISSING` is a missing root. `?.` on null or `MISSING` makes the whole read null,
   * where `.` raises. A record without the key gives null under `?.` or when a member declares the
   * key; else `.` raises. Anything but a record (or a file's `uri` / `mime` / `name`) raises.
   * Only `fieldOf` reads, so a step never reaches a JS property such as `constructor`
   * (safety-critical).
   *
   * @param head The root or call as written, for messages.
   * @param why Why a missing root is missing, for its message.
   * @returns The value, and whether every step found its key (no `?.` short-circuit or absent
   *   declared key on the way).
   * @throws {ExpressionError} When a step raises.
   */
  walkSteps(
    value: Walked,
    membersOf: () => readonly Type[],
    steps: readonly RefStep[],
    head: string,
    why: string | null,
  ): [Walked, boolean] {
    let shown = head;
    let complete = true;
    for (let index = 0; index < steps.length; index++) {
      const step = steps[index]!;
      const read = `${shown}${step.optional ? "?." : "."}${step.name}`;
      if (value === MISSING || value === null) {
        if (step.optional) return [null, false];
        const reason =
          value === MISSING
            ? `${shown} is missing${why !== null ? ` (${why})` : ""}`
            : `${shown} is null; write ${shown}?.${step.name} to read null instead, or check ${shown} != null first`;
        throw new ExpressionError(`cannot read \${${read}}: ${reason}`);
      }
      const [present, next] = fieldOf(value, step.name);
      if (present) {
        value = next;
      } else if (isRecord(value) && (step.optional || declares(membersBefore(membersOf, steps, index), step.name))) {
        value = null;
        complete = false;
      } else if (isRecord(value)) {
        throw new ExpressionError(`cannot read \${${read}}: ${shown} has no field ${reprText(step.name)}`);
      } else if (FileRef.isFileRef(value)) {
        throw new ExpressionError(`cannot read \${${read}}: ${shown} is a file, which has only uri, mime and name`);
      } else {
        throw new ExpressionError(`cannot read \${${read}}: ${shown} is ${kindName(value)}, which has no fields`);
      }
      shown = read;
    }
    return [value, complete];
  }

  /**
   * `base[i]` or `base[lower:upper]` on a list or a text, whose characters are code points, as
   * `len` counts them.
   *
   * A slice's omitted lower is 0 and omitted upper the length, a negative bound counts from the
   * end, and both clamp to `[0, n]`. A negative index counts from the end; an index out of range
   * is a miss that only `|`, `:-` and `:?` take (`${xs[0] :- ""}`), and an error anywhere else. A
   * null or missing base is a miss (a subscript of null misses, where a `.field` of null raises),
   * so `${xs[1:] :- []}` falls back, and its bounds aren't read; a null base's miss is named
   * `base[...]`, with a non-literal, non-variable base as `(...)`.
   *
   * @throws {ExpressionError} If an index or a bound is null or not an int (a bool counts as 0
   *   or 1), or the base is not a list or a text.
   */
  subscript(node: Tree): Walked {
    const [baseNode, part] = node.children as [ExprNode, Tree];
    const base = this.walk(baseNode);
    // A miss before the bounds are read, so `xs[n:] :- []` falls back whatever the bounds are.
    if (base === MISSING) return MISSING;
    if (base === null) return this.missAt(`${sliceBaseText(baseNode)}[...]`);
    if (!Array.isArray(base) && typeof base !== "string") {
      throw subscriptKindError(part.data === "index" ? "indexing" : "slicing", base);
    }
    if (part.data === "index") {
      const at = this.subscriptInt(part.children[0]!, "index");
      const items = typeof base === "string" ? [...base] : base;
      const n = BigInt(items.length);
      const k = at < 0n ? at + n : at;
      if (k < 0n || k >= n) {
        const what = typeof base === "string" ? "a text" : "a list";
        this.#missPath = null;
        this.#outOfRange = new ExpressionError(
          `index ${at} is out of range for ${what} of length ${n}; give a default with \`:-\``,
        );
        return MISSING;
      }
      return items[Number(k)]!;
    }
    const lower = this.sliceBound(part.children[0]!);
    const upper = this.sliceBound(part.children[1]!);
    const items = typeof base === "string" ? [...base] : base;
    const n = BigInt(items.length);
    const clamp = (bound: bigint | null, fallback: bigint): number => {
      let at = bound ?? fallback;
      if (at < 0n) at += n;
      return Number(at < 0n ? 0n : at > n ? n : at);
    };
    const start = clamp(lower, 0n);
    const stop = Math.max(start, clamp(upper, n));
    if (typeof base === "string") return items.slice(start, stop).join("");
    return base.slice(start, stop);
  }

  /**
   * A list comprehension, as nested loops: the first iterable in the enclosing scope,
   * each later clause once per binding of the clauses before it, an `if` keeping a binding when
   * its condition is truthy, and the elem appended after the last clause. The clauses run depth
   * first without recursion, so a long comprehension can't overflow the stack.
   *
   * @throws {ExpressionError} If an iterable is null, misses, or is not a list.
   */
  listComp(node: Tree): Value[] {
    const [elem, ...clauses] = node.children as [ExprNode, ...Tree[]];
    const out: Value[] = [];
    // One entry per active `for`: its clause index, its items, the next index, and its items' type.
    const loops: { at: number; items: Value[]; k: number; type: Type | null }[] = [];
    // Move to the next binding of the innermost loop that has one; return the next clause index, or -1.
    const next = (): number => {
      while (loops.length > 0) {
        const loop = loops.at(-1)!;
        const name = String(clauses[loop.at]!.children[0]);
        if (loop.k < loop.items.length) {
          this.#vars.set(name, { value: loop.items[loop.k++]!, type: loop.type });
          return loop.at + 1;
        }
        this.#vars.delete(name);
        loops.pop();
      }
      return -1;
    };
    let i = 0;
    while (i >= 0) {
      if (i === clauses.length) {
        out.push(this.asValue(this.walk(elem)));
        i = next();
      } else if (clauses[i]!.data === "comp_for") {
        const [target, iterable, source] = clauses[i]!.children as [Token, ExprNode, Token];
        const items = this.orNull(this.walk(iterable));
        if (items === null) {
          throw new ExpressionError(
            `comprehension \`for ${target} in ${source}\` iterates over null; give a default, e.g. \`(${source} :- [])\``,
          );
        }
        if (!Array.isArray(items)) {
          throw new ExpressionError(`comprehension \`for ${target} in ${source}\` takes a list, got ${kindName(items)}`);
        }
        loops.push({ at: i, items, k: 0, type: this.elementDeclared(iterable) });
        i = next();
      } else {
        i = truthy(this.asValue(this.walk(clauses[i]!.children[0]!))) ? i + 1 : next();
      }
    }
    return out;
  }

  /**
   * The declared element type of a comprehension's iterable: `elementType` of its type
   * (`declaredType`); `null` when untyped.
   */
  elementDeclared(iterable: ExprNode): Type | null {
    return elementType(this.declaredType(iterable));
  }

  /**
   * The declared type of `node`, inferred from the declared types of the references, `item`,
   * constructors and the comprehension variables in scope, without narrowing; `null` when untyped.
   */
  declaredType(node: ExprNode): Type | null {
    const untyped = this.#declared === null && this.#itemType === null && this.#types === null;
    if (untyped && this.#vars.size === 0) return null;
    const refType = (path: string): Type | null => {
      const names = path.split(".");
      if (names[0] === "item") return declaredAt(this.#itemType, names.slice(1));
      return this.#declared?.(names) ?? null;
    };
    const typeFn = (name: string): Type | null => this.#types?.get(name) ?? null;
    const vars = new Map([...this.#vars].map(([name, binding]): [string, Type | null] => [name, binding.type]));
    return inferExprType(node, refType, typeFn, vars);
  }

  /**
   * `base.k` and `base?.k` after a subscript or a parenthesized operand: the steps as `walkSteps`
   * walks them from the base's value, typed by the base's declared type. A `?.` on a miss reads
   * null; a `.` on an index out of range throws the index's error.
   */
  fieldRead(node: Tree): Walked {
    const baseNode = node.children[0]!;
    const { steps } = splitReference(node);
    const base = this.walk(baseNode);
    if (base === MISSING && this.#outOfRange !== null) {
      if (!steps[0]!.optional) throw this.#outOfRange;
      this.#outOfRange = null;
    }
    const membersOf = () => declaredOf(this.declaredType(baseNode));
    const [value] = this.walkSteps(base, membersOf, steps, subscriptText(baseNode), null);
    return value;
  }

  /** A comprehension variable read, then any `.field` trailers as `walkSteps` walks them. */
  compVar(node: Tree): Walked {
    const name = String(node.children[0]);
    const binding = this.#vars.get(name);
    // Only a hand-built tree can read a variable outside its comprehension.
    if (binding === undefined) throw new ExpressionError(`comprehension variable ${reprText(name)} read outside its comprehension`);
    const { steps } = splitReference(node);
    // A null item is returned as is; a `.field` on it follows the step table.
    const [value] = this.walkSteps(binding.value, () => declaredOf(binding.type), steps, name, null);
    return value;
  }

  /** One slice bound: `null` when omitted, else its int (a bool as 0 or 1). */
  sliceBound(node: ExprNode): bigint | null {
    if (node instanceof Tree && node.data === "omitted") return null;
    return this.subscriptInt(node, "slice bound");
  }

  /** A slice bound or an index (`what`): its int, a bool as 0 or 1. */
  subscriptInt(node: ExprNode, what: "slice bound" | "index"): bigint {
    const value = this.orNull(this.walk(node));
    if (value === null) {
      throw new ExpressionError(what === "index" ? "index is null" : "slice bound is null; omit it to mean the end");
    }
    if (typeof value === "boolean") return value ? 1n : 0n;
    if (typeof value !== "bigint") {
      const takes = what === "index" ? "an index takes" : "slice bounds take";
      throw new ExpressionError(`${takes} int, got ${kindName(value)}`);
    }
    return value;
  }

  /**
   * `isinstance(x, T)`: whether the value of `x` satisfies the type named `T`.
   *
   * `T` is looked up in the type table, never resolved as a reference. The check is the
   * variable pool's write-boundary check (`buildValueAs`), so a value passes exactly when it
   * could be written to a slot of type `T`. A missing `x` is checked as `null` in the
   * non-strict modes and throws in `STRICT_RAISE`.
   *
   * @throws {ExpressionError} If `T` is not in the type table.
   */
  isinstanceCall(callSuffix: Tree): boolean {
    const typeName = isinstanceTypeName(callSuffix);
    const typ = this.#types?.get(typeName);
    if (typ === undefined) throw new ExpressionError(`isinstance: unknown type ${reprText(typeName)}`);
    const value = this.asValue(this.walk((callSuffix.children[0] as Tree).children[0]!));
    try {
      buildValueAs(typ, value);
    } catch (error) {
      if (error instanceof TypeCheckError) return false;
      throw error;
    }
    return true;
  }

  /**
   * Whether a call to `name` is a `T(...)` constructor.
   *
   * A builtin wins over a type of the same name, and a builtin type name (`int`) is never a
   * constructor, even when an `isinstance` put it in the table.
   */
  isConstructor(name: string): boolean {
    return (
      name !== ISINSTANCE &&
      !TEMPLATE_FNS.has(name) &&
      !BUILTIN_TYPE_NAMES.has(name) &&
      this.#types !== null &&
      this.#types.has(name)
    );
  }

  /**
   * `T(x)` checks the value of `x` against `T`; `T(a=..., b=...)` builds a record.
   *
   * The keyword form's record holds exactly the given fields. Either value is checked with the
   * variable pool's write-boundary check and given back with only its file and object
   * representation read as `T` reads it (`normalizeFiles`), so it is exactly what `!T ${x}`
   * delivers. A missing `x` is checked as `null` in the non-strict modes and throws in
   * `STRICT_RAISE`.
   *
   * @throws {ExpressionError} If the call mixes keyword and positional arguments or has several
   *   positional ones (compose rejects this at load).
   * @throws {ConstructorTypeError} If the value does not fit `T`.
   */
  construct(name: string, callSuffix: Tree): Value {
    const shapeError = constructorShapeError(name, callSuffix);
    if (shapeError !== null) throw new ExpressionError(shapeError);
    const args = callSuffix.children as Tree[];
    let value: Value;
    if (args.length === 1 && args[0]!.children.length === 1) {
      value = this.asValue(this.walk(args[0]!.children[0]!));
    } else {
      // `fromEntries` defines each field as an own property, so a `__proto__` keyword is data.
      value = Object.fromEntries(
        args.map((arg): [string, Value] => [String(arg.children[0]), this.asValue(this.walk(arg.children[1]!))]),
      );
    }
    const typ = this.#types!.get(name)!;
    try {
      buildValueAs(typ, value);
    } catch (error) {
      if (error instanceof TypeCheckError) {
        throw new ConstructorTypeError(`${name}(...): ${error.message}`, { cause: error });
      }
      throw error;
    }
    return normalizeFiles(typ, value);
  }

  /**
   * Run a builtin call through `TEMPLATE_FNS`.
   *
   * The arguments are evaluated in source order (a keyword argument carries a leading `NAME`),
   * bound to the builtin's parameters with `bindArguments`, and passed in parameter order. A
   * shipped builtin's result is taken as it is (`isShipped`); a host function's is read as a host
   * value: a top-level `undefined` is `null`, anything else goes through `fromHost`.
   *
   * @throws {ExpressionError} If `callee` is no builtin, or the call fails: arguments that don't
   *   bind, an error the builtin throws, or a result that is no yian value
   *   (`expression function 'f' failed: <message>`). An `ExpressionError` the builtin throws
   *   passes through unchanged.
   */
  callBuiltin(callee: string, callSuffix: Tree): Value {
    const fn = TEMPLATE_FNS.get(callee);
    if (fn === undefined) throw new ExpressionError(unknownFunctionMessage(callee));
    const positional: Value[] = [];
    const keywords = new Map<string, Value>();
    for (const arg of callSuffix.children as Tree[]) {
      const kids = arg.children;
      if (kids.length === 2) keywords.set(String(kids[0]), this.asValue(this.walk(kids[1]!)));
      else positional.push(this.asValue(this.walk(kids[0]!)));
    }
    const failed = (message: string, options?: ErrorOptions) =>
      new ExpressionError(`expression function ${reprText(callee)} failed: ${message}`, options);
    const bound = bindArguments(fn, positional, keywords);
    if (typeof bound === "string") throw failed(bound);
    try {
      const result = fn.call(...bound);
      if (isShipped(fn)) return result as Value;
      return result === undefined ? null : fromHost(result, "result");
    } catch (error) {
      if (error instanceof ExpressionError) throw error;
      throw failed(error instanceof Error ? error.message : String(error), { cause: error });
    }
  }

  // --- coalesce / default / required (short-circuit) --- //
  /**
   * `a | b | c`: the first operand that resolves to a non-null value wins; a present falsy
   * value (`0`, `""`, `false`) wins. Only the last operand's miss is subject to `STRICT_RAISE`.
   */
  coalesce(node: Tree): Value {
    const last = node.children.length - 1;
    for (const [i, child] of node.children.entries()) {
      const walked = this.walk(child);
      const value = i === last ? this.asValue(walked) : this.orNone(walked);
      if (value !== null) return value;
    }
    return null;
  }

  /**
   * `head :- fallback`: the fallback is evaluated only when `head` misses or resolves to
   * `null`. Only the fallback's miss is subject to `STRICT_RAISE`.
   */
  defaultExpr(node: Tree): Value {
    const [head, fallback] = node.children as [ExprNode, ExprNode];
    let value = this.orNone(this.walk(head));
    if (value === null) value = this.asValue(this.walk(fallback));
    return value;
  }

  /**
   * `head :? message`: throw `RequiredError` with the message (text as is, any other value in
   * its text form) when `head` misses or is `null`, in every mode.
   */
  requiredExpr(node: Tree): Value {
    const [head, message] = node.children as [ExprNode, ExprNode];
    const value = this.orNone(this.walk(head));
    if (value === null) {
      const text = this.asValue(this.walk(message));
      throw new RequiredError(typeof text === "string" ? text : valueRepr(text));
    }
    return value;
  }

  /**
   * `then if test else otherwise`: the test is tested for truth, as `and` and `or` test their
   * operands, and only the branch it picks is evaluated. The value is that branch's, a miss
   * included, so `(a if c else b) :- d` falls back when the picked branch misses.
   */
  ifExpr(node: Tree): Walked {
    const [then, test, otherwise] = node.children as [ExprNode, ExprNode, ExprNode];
    return this.walk(truthy(this.asValue(this.walk(test))) ? then : otherwise);
  }

  // --- boolean combinators --- //
  /** The operand children of a boolean combinator node (the operator keywords dropped). */
  operands(node: Tree): ExprNode[] {
    return node.children.filter((c) => !(c instanceof Token && OP_TOKEN_TYPES.has(c.type)));
  }

  negate(node: Tree): boolean {
    return !truthy(this.asValue(this.walk(this.operands(node).at(-1)!)));
  }

  // `and` / `or` fold by truthiness, not bool-only, so values such as `0` and `"hit"` take part
  // directly; they stop at the first deciding operand and give a boolean.
  orExpr(node: Tree): boolean {
    return this.operands(node).some((c) => truthy(this.asValue(this.walk(c))));
  }

  andExpr(node: Tree): boolean {
    return this.operands(node).every((c) => truthy(this.asValue(this.walk(c))));
  }

  // --- comparisons (through `evalComparison`, missing-is-falsy) --- //
  /** A `compare` node by its operator token, or a `compare_in` / `compare_notin` node by `op`. */
  comparison(node: Tree, op?: "in" | "not in"): boolean {
    const [lhs, token, rhs] = node.children as [ExprNode, Token, ExprNode];
    return evalComparison(op ?? token.value, this.asValue(this.walk(lhs)), this.asValue(this.walk(rhs)));
  }

  // --- arithmetic over resolved values --- //
  /**
   * A `+ -` or `* / %` chain, folded left to right.
   *
   * Each operand is walked just before its step is applied, so evaluation order, error
   * precedence and each step's message (naming the running value) are those of the left-nested
   * binary steps. The fold is a loop, so a chain of any length adds no recursion.
   */
  sum(node: Tree): Value {
    const [operands, operators] = splitChain(node);
    let acc = this.asValue(this.walk(operands[0]!));
    // True while `acc` is the fresh list a list `+` step built, which nothing else holds: the
    // next list `+` can then append in place instead of copying the running list again.
    let owned = false;
    for (const [i, op] of operators.entries()) {
      const rhs = this.asValue(this.walk(operands[i + 1]!));
      if (owned && op === "+" && Array.isArray(rhs)) {
        // A loop, not `push(...rhs)`, which overflows the stack for a long list.
        for (const item of rhs) (acc as Value[]).push(item);
      } else {
        acc = applyStep(op, acc, rhs);
        owned = op === "+" && Array.isArray(acc);
      }
    }
    return acc;
  }

  /**
   * A `**` chain: every operand walked left to right, then folded right to left.
   *
   * This keeps the right-nested binary steps' order (`a ** (b ** c)`): a strict miss in an
   * earlier operand wins before a later one is walked, and the innermost step's error comes
   * first.
   */
  power(node: Tree): Value {
    const values = splitChain(node)[0].map((operand) => this.asValue(this.walk(operand)));
    return values.reduceRight((exponent, base) => applyStep("**", base, exponent));
  }

  neg(node: Tree): Value {
    return minus(this.asValue(this.walk(node.children[0]!)));
  }

  listLit(node: Tree): Value[] {
    return node.children.map((c) => this.asValue(this.walk(c)));
  }
}

/** The options of `evalExpr`. */
export interface EvalOptions {
  item?: Value;
  mode?: ResolveMode;
  types?: ReadonlyMap<string, Type> | null;
  /** The declared type of a reference's root, given its names; `null` leaves every step undeclared. */
  declared?: DeclaredTypes | null;
  /** The declared type of `item`; `null` leaves its steps undeclared. */
  itemType?: Type | null;
}

/**
 * Evaluate a unified `${...}` expression tree (from `parseExpr`).
 *
 * The one evaluator for every `${...}` construct: references (bare, dotted with `.` and `?.`,
 * `item`-scoped), arithmetic and comparisons over values (so text + text and list + list
 * concatenate), boolean combinators, list literals, indexes and slices of lists and texts
 * (`xs[i]`, `xs[a:b]`; a subscript of a miss or `null` is a miss, and so is an index out of range
 * for `|`, `:-` and `:?`, though an error anywhere else), list comprehensions (a `null` item is
 * returned as is), and coalesce / default / required. The operators follow the TS
 * number model: an int is a `bigint` at any size, a bool counts as 0 or 1, `/` always gives a
 * float, and a float result is IEEE's (`Infinity` is a value, NaN an error).
 *
 * A reference is a root, then `.` and `?.` steps. A missing root (a skipped node, an absent input
 * or local) is `null` in the non-strict modes. Each step follows the step table:
 *
 * | Receiver | `.k` | `?.k` |
 * |---|---|---|
 * | null, or a missing root | raises | null |
 * | record with `k` | its value | its value |
 * | record without `k`, its declared type declares `k` | null | null |
 * | record without `k`, untyped or not declared | raises | null |
 * | text, number, bool, list, file (`k` not `uri`/`mime`/`name`) | raises | raises |
 *
 * `?.` on null short-circuits the rest of the chain. `|`, `:-` and `:?` see only null and missing
 * roots, never an error from their left side. Arithmetic over `null` throws; an ordered
 * comparison with a `null` operand is `false`.
 *
 * Dotted access reads only a record's own field or a file's `uri`, `mime` or `name`
 * (`fieldOf`), never a JS property, so `${x.constructor}` raises "no field" (safety-critical).
 *
 * @param tree The parsed expression: a `Tree`, or a lone literal `Token`.
 * @param resolve Reads a reference's root from its names (`RootResolver`); the evaluator walks the
 *   names after the root. `item`-headed references bypass it.
 * @param options.item The MAP-body-local scope for `item` / `item.path` (record and file
 *   fields); `null` (the default) means no item scope.
 * @param options.mode How a missing root is treated (default `BINDING_NONE`).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in, keyed by type name; `null` (the default) means no types.
 * @param options.declared The declared type of a reference's root, given its names, so a declared
 *   field that is absent reads `null`; `null` (the default) leaves every step undeclared.
 * @param options.itemType The declared type of `item`; `null` (the default) leaves its steps
 *   undeclared.
 * @returns The value: `null`, a boolean, a `bigint`, a number, text, a list, a record or a file
 *   (`FileRef`).
 * @throws {RequiredError} When a `head :? message` head misses or is `null`.
 * @throws {ConstructorTypeError} When a `T(...)` constructor's value does not fit `T`.
 * @throws {ExpressionError} On a field read the step table rejects, an unsupported node, a call to
 *   an unknown function or one that fails, an `isinstance` type absent from `types`, arithmetic or
 *   ordering over operand kinds the operator doesn't take (`null` included), a zero divisor, a NaN
 *   result, an int too large for the runtime, a subscript of anything but a list or a text, or
 *   with an index or bound that is null or not an int, an index out of range outside `|`, `:-`
 *   and `:?`, a comprehension over anything but a list (`null` or a miss included), or a
 *   strict-mode miss.
 */
export function evalExpr(
  tree: ExprNode,
  resolve: RootResolver,
  { item, mode = ResolveMode.BINDING_NONE, types = null, declared = null, itemType = null }: EvalOptions = {},
): Value {
  return new EvalExpr(resolve, item, mode, types, declared, itemType).eval(tree);
}

// --------------------------------------------------------------------------------------- //
// Conservative compile-time type inference over the unified expression tree.
// --------------------------------------------------------------------------------------- //

// The kinds of a text value.
const TEXT_KINDS: ReadonlySet<ValueKind> = new Set([ValueKind.STRING, ValueKind.DATE, ValueKind.DATETIME]);

// The nodes whose value is always a boolean.
const BOOLEAN_NODES: ReadonlySet<string> = new Set([
  "compare",
  "compare_in",
  "compare_notin",
  "negate",
  "and_expr",
  "or_expr",
]);

/**
 * Return the statically proven type of one parsed expression, if any.
 *
 * Inference is deliberately conservative: literal tokens, exact references, comparisons,
 * boolean combinators, `isinstance(x, T)` and each shipped builtin (`builtinResultType`) have
 * fixed result types, and a `T(...)` constructor has type `T` (its value is checked against
 * `T`), with field reads on the result typed through `T`. A read with a `?.` step (`a?.b`) is
 * `Optional` of the field. Also typed:
 *
 * - a comprehension variable: its element type in scope (`elementType`), with field reads typed
 *   through it;
 * - `a | b`, `a :- b`: the non-null left operands joined with the last (`joinedType`, equal up
 *   to `Optional`), where an empty `[]` adds nothing, so `(xs :- [])` types like `xs` made
 *   non-null; `a :? m`: the non-null `a` (`m` is the error message);
 * - `a if c else b`: `a` and `b` joined, where a `null` branch makes the other `Optional` and an
 *   empty `[]` adds nothing;
 * - `xs[i]`: the item type of `xs`'s list type, `Optional` when `xs` may be null; `s[i]` and
 *   `s[a:b]` of a text: `str`, an index `Optional` as for a list;
 * - `xs[a:b]`: `xs`'s list type, non-null;
 * - `x.k` after a subscript or parentheses: typed through `x`'s type, as a call's result is;
 * - `[e for ...]`: `list` of `e`'s type in its scope;
 * - a list literal: `list` of its items joined.
 *
 * Other forms stay unknown until their runtime value semantics can be modeled without guessing.
 * Reference and constructor lookup are injected so this source-neutral expression layer knows
 * nothing of a flow graph or compose document.
 *
 * Example: `inferExprType(parseExpr("answer == 42"), () => Type.scalar(ValueKind.INTEGER))`
 * gives `Type.scalar(ValueKind.BOOLEAN)`.
 *
 * @param tree A tree or token from `parseExpr`.
 * @param refTypeFn Resolves one dotted reference path to its known type, or gives `null` when
 *   the type is not statically known. Anything it throws propagates unchanged.
 * @param typeFn Resolves a constructor name to its type, or gives `null` for a name that is no
 *   constructor; `null` leaves every constructor unknown.
 * @param vars The comprehension variables in scope, by name, to their element types (`null`
 *   when untyped).
 * @returns The proven expression type, or `null` when inference is inconclusive.
 */
export function inferExprType(
  tree: ExprNode,
  refTypeFn: (path: string) => Type | null,
  typeFn: ((name: string) => Type | null) | null = null,
  vars: ReadonlyMap<string, Type | null> = new Map(),
): Type | null {
  const infer = (node: ExprNode, scope = vars): Type | null => inferExprType(node, refTypeFn, typeFn, scope);
  if (tree instanceof Token) {
    switch (tree.type) {
      case "NUMBER":
        return Type.scalar(tree.value.includes(".") ? ValueKind.NUMBER : ValueKind.INTEGER);
      case "STRING":
        return Type.scalar(ValueKind.STRING);
      case "BOOL":
        return Type.scalar(ValueKind.BOOLEAN);
      case "NULL":
        return Type.scalar(ValueKind.NONE);
      case "INF":
        return Type.scalar(ValueKind.NUMBER);
    }
    return null;
  }
  if (tree.data === "refcall") {
    const { head: name, steps } = splitReference(tree);
    const callSuffix = callSuffixOf(tree);
    if (callSuffix !== undefined) {
      // `parseExpr` rejects dotted access on an `isinstance(...)` result.
      if (name === ISINSTANCE) return Type.scalar(ValueKind.BOOLEAN);
      if (TEMPLATE_FNS.has(name)) {
        const args = () => argumentTypes(name, callSuffix, (node) => infer(node));
        return stepsType(builtinResultType(name, args), steps);
      }
      return stepsType(typeFn === null ? null : typeFn(name), steps);
    }
    const typ = refTypeFn(referencePath(tree));
    return typ !== null && steps.some((step) => step.optional) ? typ.replace({ nullable: true }) : typ;
  }
  switch (tree.data) {
    case "comp_var": {
      const { head, steps } = splitReference(tree);
      return stepsType(vars.get(head) ?? null, steps);
    }
    case "coalesce":
    case "default_expr": {
      const last = tree.children.length - 1;
      const types: Type[] = [];
      for (const [i, operand] of tree.children.entries()) {
        if (operand instanceof Tree && operand.data === "list_lit" && operand.children.length === 0) continue;
        const typ = infer(operand);
        if (typ === null) return null;
        types.push(i === last ? typ : typ.replace({ nullable: false }));
      }
      return joinedType(types);
    }
    case "required_expr":
      return infer(tree.children[0]!)?.replace({ nullable: false }) ?? null;
    case "field_read":
      return stepsType(infer(tree.children[0]!), splitReference(tree).steps);
    case "if_expr": {
      // A `null` branch makes the other's type Optional; an empty `[]` adds nothing.
      const types: Type[] = [];
      let nullable = false;
      for (const branch of [tree.children[0]!, tree.children[2]!]) {
        if (branch instanceof Tree && branch.data === "list_lit" && branch.children.length === 0) continue;
        const typ = infer(branch);
        if (typ === null) return null;
        if (typ.kind === ValueKind.NONE && typ.members === undefined) nullable = true;
        else types.push(typ);
      }
      const joined = joinedType(types);
      return joined !== null && nullable ? joined.replace({ nullable: true }) : joined;
    }
    case "subscript": {
      const typ = infer(tree.children[0]!);
      if (typ === null) return null;
      const index = (tree.children[1] as Tree).data === "index";
      const text = typ.members === undefined && TEXT_KINDS.has(typ.kind);
      let item: Type | null;
      if (text) item = Type.scalar(ValueKind.STRING);
      else if (!isListKind(typ.kind)) return null;
      else if (!index) return typ.replace({ nullable: false });
      else item = typ.element ?? null;
      // A null base is a miss, so an index of a base that may be null may be null too.
      return item !== null && index && typ.nullable ? item.replace({ nullable: true }) : item;
    }
    case "list_comp": {
      const scope = new Map(vars);
      for (const clause of tree.children.slice(1) as Tree[]) {
        // A `comp_if` narrows only in the load checker, which has the facts.
        if (clause.data === "comp_for") {
          scope.set(String(clause.children[0]), elementType(infer(clause.children[1]!, scope)));
        }
      }
      return listOf(infer(tree.children[0]!, scope));
    }
    case "list_lit": {
      const items = tree.children.map((item) => infer(item));
      return items.some((typ) => typ === null) ? null : listOf(joinedType(items as Type[]));
    }
  }
  return BOOLEAN_NODES.has(tree.data) ? Type.scalar(ValueKind.BOOLEAN) : null;
}

/**
 * Return the type of each parameter's argument in a call to builtin `name`, in parameter order:
 * `infer`'s type for a given argument, `undefined` for one the call leaves out; `null` when the
 * arguments don't bind (`checkCalls` reports that).
 */
function argumentTypes(name: string, callSuffix: Tree, infer: (node: ExprNode) => Type | null): ArgType[] | null {
  const args = callSuffix.children as Tree[];
  const positional = args.filter((arg) => arg.children.length === 1).map((arg) => arg.children[0]!);
  const keywords = new Map(
    args.filter((arg) => arg.children.length === 2).map((arg) => [String(arg.children[0]), arg.children[1]!]),
  );
  const bound = bindArguments(TEMPLATE_FNS.get(name)!, positional, keywords);
  if (typeof bound === "string") return null;
  return bound.map((arg) => (arg instanceof Tree || arg instanceof Token ? infer(arg) : undefined));
}

/** Return the type of reading `steps` from a value of type `typ`, if proven; a `?.` step makes it `Optional`. */
export function stepsType(typ: Type | null, steps: readonly RefStep[]): Type | null {
  for (const step of steps) {
    if (typ === null) return null;
    typ = fieldReadType(typ, step.name);
    if (typ !== null && step.optional) typ = typ.replace({ nullable: true });
  }
  return typ;
}

/** Return the list type of elements of type `element`, or `null` when `element` is unknown. */
function listOf(element: Type | null): Type | null {
  return element === null ? null : new Type({ kind: listKindForElement(element), element });
}

/**
 * Return the element type a comprehension variable gets from an iterable of type `iterable`, or
 * `null` when untyped: `list(E)` and `Optional(list(E))` give `E`; `Any`, `object`, a union (its
 * members are records, which raise when iterated), an untyped list or an unknown type give `null`.
 */
export function elementType(iterable: Type | null): Type | null {
  if (iterable === null || !isListKind(iterable.kind)) return null;
  return iterable.element ?? null;
}

/**
 * Return the type of reading field `name` on a value of type `typ`, if proven.
 *
 * A record's field is `Optional` when the record is, or the field is not required; a union's
 * field is the members' shared type when every member has it; a file's `uri`, `mime` or `name`
 * is its `fileFieldType`, `Optional` when the file is.
 */
function fieldReadType(typ: Type, name: string): Type | null {
  let field: Type;
  if (typ.members !== undefined) {
    const read = unionField(typ, name);
    if (read.kind !== "read" || read.type === null) return null;
    field = read.type;
  } else if (typ.fields !== undefined && typ.fields.has(name)) {
    field = typ.fields.get(name)!;
    if (!typ.required?.has(name)) field = field.replace({ nullable: true });
  } else if (typ.fields === undefined && typ.kind === ValueKind.FILE) {
    const fileField = fileFieldType(name);
    if (fileField === null) return null;
    field = fileField;
  } else {
    return null;
  }
  return typ.nullable ? field.replace({ nullable: true }) : field;
}

/** Return the type name of a validated `isinstance(x, T)` call suffix. */
function isinstanceTypeName(callSuffix: Tree): string {
  const typeArg = callSuffix.children[1] as Tree;
  return String((typeArg.children[0] as Tree).children[0]);
}

/**
 * Return the type names the `isinstance(x, T)` calls in a parsed expression name.
 *
 * Example: `isinstanceTypeNames(parseExpr("isinstance(x, Buy) or isinstance(y, int)"))` gives
 * `["Buy", "int"]`.
 *
 * @param tree A tree from `parseExpr` or `parseCondition`, which have already checked each
 *   `isinstance` call's shape.
 * @returns Each `T` in source order, without duplicates; `[]` for a lone token.
 */
export function isinstanceTypeNames(tree: ExprNode): string[] {
  if (!(tree instanceof Tree)) return [];
  const names = new Set<string>();
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall" || String(node.children[0]) !== ISINSTANCE) continue;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) continue;
    names.add(isinstanceTypeName(callSuffix));
  }
  return [...names];
}

/**
 * Return the callees in a parsed expression that can only be `T(...)` constructors.
 *
 * A callee that is neither a builtin nor `isinstance` is a constructor name; compose resolves
 * each against the flow's `typedefs:` and rejects any other name at load.
 *
 * Example: `constructorNames(parseExpr("[Buy(d.output), upper(x), Sell(price=p)]"))` gives
 * `["Buy", "Sell"]`.
 *
 * @param tree A tree from `parseExpr` or `parseCondition`.
 * @returns Each name in source order, without duplicates; `[]` for a lone token.
 */
export function constructorNames(tree: ExprNode): string[] {
  if (!(tree instanceof Tree)) return [];
  const names = new Set<string>();
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall" || callSuffixOf(node) === undefined) continue;
    const name = String(node.children[0]);
    if (name !== ISINSTANCE && !TEMPLATE_FNS.has(name)) names.add(name);
  }
  return [...names];
}

/**
 * Return the tested expression `x` of an `isinstance(x, T)` call.
 *
 * Example: for `isinstance(d.output, Buy)` it is the `d.output` refcall; for
 * `isinstance(d.output, Buy) and x` it is `null`.
 *
 * @param tree A parsed expression from `parseExpr` or `parseCondition`, which have already
 *   checked each `isinstance` call's shape.
 * @returns The first argument's expression when `tree` itself is an `isinstance` call, else
 *   `null`.
 */
export function isinstanceSubject(tree: ExprNode): ExprNode | null {
  if (
    !(tree instanceof Tree) ||
    tree.data !== "refcall" ||
    tree.children.length !== 2 ||
    String(tree.children[0]) !== ISINSTANCE
  ) {
    return null;
  }
  const callSuffix = tree.children[1];
  if (!(callSuffix instanceof Tree) || callSuffix.data !== "call_suffix") return null;
  return (callSuffix.children[0] as Tree).children[0]!;
}

/** One `.name` or `?.name` step of a reference, a comprehension variable read, or a call result. */
export interface RefStep {
  readonly name: string;
  /** Written `?.name`. */
  readonly optional: boolean;
}

/**
 * Split a `refcall` or `comp_var` into its head name and its steps as written. For a call, the
 * steps are the trailers after its parentheses (the parser rejects trailers before them).
 */
export function splitReference(node: Tree): { head: string; steps: RefStep[] } {
  const steps = node.children
    .filter((child): child is Tree => child instanceof Tree && child.data === "trailer")
    .map((trailer) => ({ name: String(trailer.children[0]), optional: trailer.children.length === 2 }));
  return { head: String(node.children[0]), steps };
}

/**
 * Return the load error for the first reference in `tree` with a `?.` inside its root, else `null`.
 *
 * A comprehension variable is a one-name root, so its reads never have one.
 *
 * @param rootLength How many names a reference's root takes in this scope, given all its names.
 */
export function optionalInsideRoot(tree: ExprNode, rootLength: (names: readonly string[]) => number): string | null {
  if (!(tree instanceof Tree)) return null;
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall" || callSuffixOf(node) !== undefined) continue;
    const { head, steps } = splitReference(node);
    const used = rootLength([head, ...steps.map((step) => step.name)]);
    if (!steps.slice(0, used - 1).some((step) => step.optional)) continue;
    // The fix: no `?.` inside the root, and `?.` on the first step after it.
    const fixed = steps.map((step, i) => ({
      name: step.name,
      optional: i < used - 1 ? false : i === used - 1 ? true : step.optional,
    }));
    return (
      `reference \${${referenceText(node)}} puts ?. inside its root; ` +
      `?. goes after the root, as in \${${head + stepsText(fixed)}}`
    );
  }
  return null;
}

/**
 * Return `text` with `?.` on the first step after each reference root `isRoot` accepts; each such
 * reference is re-printed from its steps, and everything else is kept verbatim.
 *
 * @param text An expression (a bare expression, or a span interior).
 * @param rootLength How many names a reference's root takes, given all its names.
 * @param isRoot Whether a root, as a flat dotted path, gets the `?.`.
 */
export function optionalAfterRoots(
  text: string,
  rootLength: (names: readonly string[]) => number,
  isRoot: (root: string) => boolean,
): string {
  const edits: [start: number, end: number, replacement: string][] = [];
  const tree = parseExpr(text);
  for (const node of tree instanceof Tree ? iterSubtreesTopdown(tree) : []) {
    if (node.data !== "refcall" || callSuffixOf(node) !== undefined) continue;
    const { head, steps } = splitReference(node);
    const used = rootLength([head, ...steps.map((step) => step.name)]);
    const root = [head, ...steps.slice(0, used - 1).map((step) => step.name)].join(".");
    if (steps.length < used || !isRoot(root)) continue;
    const marked = steps.map((step, i) => (i === used - 1 ? { ...step, optional: true } : step));
    edits.push([(node.children[0] as Token).startPos, namesEnd(node, 1 + steps.length), head + stepsText(marked)]);
  }
  edits.sort((a, b) => b[0] - a[0]);
  for (const [start, end, replacement] of edits) text = text.slice(0, start) + replacement + text.slice(end);
  return text;
}

/** Return steps as written: `.a?.b`. */
export function stepsText(steps: readonly RefStep[]): string {
  return steps.map((step) => (step.optional ? "?." : ".") + step.name).join("");
}

/** Return a reference's flat dotted path, its head and step names with `?.` written as `.`. */
export function referencePath(node: Tree): string {
  const { head, steps } = splitReference(node);
  return [head, ...steps.map((step) => step.name)].join(".");
}

/** Return a reference as written, `?.` included. */
export function referenceText(node: Tree): string {
  const { head, steps } = splitReference(node);
  return head + stepsText(steps);
}

/**
 * Collect every reference-leaf path a unified `${...}` expression tree reads.
 *
 * The single compile-time reference walk over a tree from `parseExpr`. A `refcall` without a
 * `call_suffix` is a reference: its dotted path (leading `NAME` joined with each `trailer`
 * segment) is collected. A `refcall` with a `call_suffix` is a builtin call: the callee name
 * contributes no reference, but each `arg` value's references are collected (recursively).
 * `isinstance(x, T)` collects only `x`'s references; `T` is a type name. Literals contribute
 * nothing. Coalesce, default and required references are all collected, including a default's.
 *
 * `item`-headed references are collected here, matching the evaluator, which resolves them from
 * the MAP-body-local scope. The caller (compose, when minting edges) skips them.
 *
 * @param tree The parsed expression. A lone literal is a top-level `Token`; every other construct
 *   is a `Tree`; both are handled.
 * @returns The reference paths (dotted, `#` / `/` preserved) in source order, not deduped: every
 *   span-scanning caller unions them in source order.
 */
export function exprRefs(tree: ExprNode): string[] {
  const refs: string[] = [];
  collectExprRefs(tree, refs);
  return refs;
}

/** Walk `node`, collecting reference paths into `refs` (source order, no dedupe). */
function collectExprRefs(node: ExprNode, refs: string[]): void {
  if (node instanceof Token) return;
  if (node.data === "refcall") {
    // Keep in sync with the evaluator's refcall decomposition, `collectRefEdits` and
    // `conditionReadLeaves` in `template.ts`: each finds the call_suffix and takes the path from
    // `splitReference`.
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) {
      refs.push(referencePath(node));
      return;
    }
    // A builtin call: the callee contributes no reference; walk each argument value only.
    for (const arg of valueArgs(String(node.children[0]), callSuffix)) {
      // A keyword argument carries a leading NAME token; the value expression is the last child.
      collectExprRefs(arg.children.at(-1)!, refs);
    }
    return;
  }
  for (const child of node.children) collectExprRefs(child, refs);
}

/** A rename's answer: `text` replaces the read's first `used` names; the steps after them stay as written. */
export interface RefRenamed {
  readonly text: string;
  readonly used: number;
}

/** Rename one reference, given its flat dotted path; `null` keeps it. */
export type RefRename = (path: string) => RefRenamed | null;

/**
 * Return the end offset of a reference's `used`-th name (the NAME token or that trailer's NAME).
 *
 * @throws {RangeError} If `used` is not between 1 and the read's name count.
 */
export function namesEnd(node: Tree, used: number): number {
  const names = 1 + splitReference(node).steps.length;
  if (!Number.isInteger(used) || used < 1 || used > names) throw new RangeError(`a rename used ${used} of ${names} names`);
  const last = node.children[used - 1]!;
  return last instanceof Tree ? (last.children[0] as Token).endPos : (last as Token).endPos;
}

/**
 * Rewrite the reference-leaf paths of one unified `${...}` expression `text`.
 *
 * The rewrite analog of `exprRefs`: it walks the same parse tree (`refcall` reference /
 * builtin-call arguments), but instead of collecting each reference path it asks `rename(path)`
 * for a replacement and splices it in by source position. `rename` gets the flat dotted path and
 * returns the text for a prefix of the read, its first `used` names, or `null` to leave that
 * reference untouched. Only that prefix's source is replaced; the steps after it stay as written,
 * `?.` included. Everything that is not a reference leaf (operators, numbers, string and list
 * literals, a builtin call's callee name, parentheses, whitespace) is kept verbatim, as is the
 * type name of an `isinstance(x, T)`.
 *
 * @param text The expression source (a bare expression, or a span interior).
 * @param rename `path -> { text, used }`, or `null` to keep.
 * @returns `text` with the renamed reference leaves spliced in.
 * @throws {ExpressionError} If `text` does not parse (the same front end as `parseExpr`).
 */
export function rewriteExprRefs(text: string, rename: RefRename): string {
  const edits: [number, number, string][] = [];
  collectRefEdits(parseExpr(text), rename, edits);
  // Splice right to left so earlier offsets stay valid as later ones are replaced.
  edits.sort((a, b) => b[0] - a[0]);
  for (const [start, end, replacement] of edits) text = text.slice(0, start) + replacement + text.slice(end);
  return text;
}

/**
 * Walk `node`, collecting a `[start, end, replacement]` splice for each renamed reference leaf.
 * Keep in sync with `collectExprRefs` (and `rewriteConditionReads` in `template.ts`): the same
 * refcall / builtin-argument decomposition, but capturing token source positions to edit in place.
 */
function collectRefEdits(node: ExprNode, rename: RefRename, edits: [number, number, string][]): void {
  if (node instanceof Token) return;
  if (node.data === "refcall") {
    const children = node.children;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) {
      const renamed = rename(referencePath(node));
      if (renamed !== null) edits.push([(children[0] as Token).startPos, namesEnd(node, renamed.used), renamed.text]);
      return;
    }
    // A builtin call: the callee is not a reference; rewrite each argument value only.
    for (const arg of valueArgs(String(children[0]), callSuffix)) {
      collectRefEdits(arg.children.at(-1)!, rename, edits);
    }
    return;
  }
  for (const child of node.children) collectRefEdits(child, rename, edits);
}

/**
 * Rewrite the reference leaves of a condition-field value (`when:`/`until:`/`while:`/`asserts:`/
 * searched-`case` `when:`), honouring both spellings.
 *
 * The rewrite analog of `conditionRefs`: a whole-span `${a > 5}` keeps its outer braces and
 * rewrites the interior; a bare `a > 5` is rewritten in place. `rename` follows `rewriteExprRefs`:
 * it replaces a prefix of each read and keeps the steps after it. The value is trimmed first,
 * matching `parseCondition`.
 */
export function rewriteConditionRefs(expression: string, rename: RefRename): string {
  const interior = wholeSpanInterior(expression);
  if (interior !== null) return "${" + rewriteExprRefs(interior, rename) + "}";
  return rewriteExprRefs(expression.trim(), rename);
}
