/**
 * `${...}` evaluation plus conservative compile-time type/reference analysis.
 *
 * Variable resolution goes through `VariablePool.resolve`, so expressions read typed segments
 * and can traverse into object outputs such as `${x.output.output.ratio}`.
 *
 * Grammar: comparisons (`==` `!=` `<` `<=` `>` `>=` `in` `not in`) and arithmetic over
 * references and literals, combined with `and` / `or` / `not` and parentheses. The same parsed
 * tree feeds the evaluator, conservative type inference, and the reference walkers. Conditions
 * route through the one unified engine; a bare reference with no comparison operator is a
 * truthiness test (not an error). `isinstance(x, T)` checks a value against a type named in the
 * `types` table every evaluation entry point accepts, and a constructor `T(x)` /
 * `T(a=..., b=...)` checks (or builds and checks) a value of that type.
 *
 * With `trace`, evaluation also records a `ValueRoute` for the value: the source each read came
 * from, the arm a coalesce or default selected, list and constructor structure, `+` chains of
 * lists as concats, and every other operation as a scalar or (for a call) an opaque route over
 * its operands' routes, so feedback on the value can go back to its sources.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
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
import {
  fieldOf,
  fileFieldType,
  fromHost,
  isListKind,
  isRecord,
  valueRepr,
} from "../typesys/values.ts";
import { bindArguments, kindName, TEMPLATE_FNS, type TemplateFn } from "./builtins.ts";
import { ResolvedValue, SourceRef, type TracedValue, ValueRoute } from "./lineage.ts";
// grammar.ts imports this module too, so its exports are used only inside functions here.
import {
  callSuffixOf,
  type ExprNode,
  iterSubtreesTopdown,
  parseExpr,
  Token,
  Tree,
  wholeSpanInterior,
} from "./grammar.ts";

/** An expression could not be parsed or evaluated. */
export class ExpressionError extends RangeError {
  override name = "ExpressionError";
}

/** A `${ref:?message}` whose ref was unbound (the binder maps it to BindingError). */
export class RequiredError extends ExpressionError {
  override name = "RequiredError";
}

/** A `${T(...)}` constructor's value does not fit `T` (the binder's `binding.type`). */
export class ConstructorTypeError extends ExpressionError {
  override name = "ConstructorTypeError";
}

/**
 * The type-test special form `isinstance(x, T)`: its second argument is a type name
 * looked up in the node's type table, never a reference, so the evaluator, the reference
 * walkers and rewriters, and type inference treat only its first argument as an expression.
 */
export const ISINSTANCE = "isinstance";

/**
 * Return the hint for calling a builtin type name, else `""`.
 *
 * `int(x)` reads like a conversion, but constructors are `typedefs:` names that only check.
 *
 * @param name The callee.
 * @returns `"; types in expressions are typedef names; ..."` for a builtin type name (`int`,
 *   `list`, `Optional`), else empty.
 */
export function conversionHint(name: string): string {
  if (!BUILTIN_TYPE_NAMES.has(name)) return "";
  return `; types in expressions are typedef names; \`${name}(...)\` is not a conversion`;
}

/**
 * Return the error text for a call to `name`, which is no builtin or type.
 *
 * @param name The callee.
 * @returns The message, with `conversionHint` for a builtin type name.
 */
export function unknownFunctionMessage(name: string): string {
  return `unknown expression function ${reprText(name)}` + conversionHint(name);
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

/**
 * Resolve a `${path}` reference against the pool.
 *
 * @returns The value, or `null` for a miss.
 * @throws {ExpressionError} If a dotted segment of `path` is empty.
 */
export function resolveReference(path: string, pool: VariablePool): Value {
  const parts = path.trim().split(".");
  if (parts.some((part) => !part)) throw new ExpressionError(`invalid variable path: '\${${path}}'`);
  const [head, ...rest] = parts;
  return pool.resolve(head!, rest);
}

/**
 * Resolve a `${path}` against a node's bound input record (dotted walk).
 *
 * Bare local-input refs only. A missing input, a dotted miss, or a `null` step gives `null`,
 * which gives `when:` its missing-is-falsy behavior. Compose rejects undeclared names
 * everywhere, and field reads a known type can't have in bindings, CASE conditions, LOOP
 * predicates, and flow outputs. Asserts check less (a node assert's reads of its inputs and a
 * flow assert's `${input.X.f}` check only the head name), so there a `null` can also be a
 * misspelled field.
 *
 * The head is read only as an own field of `record`, and each step through `fieldOf`, so a
 * path never reaches a JS property such as `constructor`.
 *
 * @throws {ExpressionError} If a dotted segment of `path` is empty.
 */
export function resolveInRecord(path: string, record: ValueRecord): Value {
  const parts = path
    .trim()
    .split(".")
    .map((part) => part.trim());
  if (parts.some((part) => !part)) throw new ExpressionError(`invalid variable path: '\${${path}}'`);
  let value = fieldOf(record, parts[0]!)[1];
  for (const step of parts.slice(1)) value = fieldOf(value, step)[1];
  return value;
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
  // ordered comparisons: null on either side -> false (propagate falsy)
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
 * @throws {ExpressionError} On a malformed condition, or one that calls a function that isn't a
 *   builtin or calls a builtin with arguments it doesn't take (the caller frames it as a located
 *   load error).
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
  resolve: (path: string) => Value,
  types: ReadonlyMap<string, Type> | null,
): boolean {
  const tree = parseCondition(expression, types ?? new Map());
  return truthy(evalExpr(tree, resolve, { mode: ResolveMode.CONDITION_FALSY, types }));
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
  return evaluate(expression, (path) => resolveReference(path, pool), types);
}

/**
 * Evaluate a strict-CASE `when:` against the node's bound input record.
 *
 * `name` / `name.path` resolve against `record` (`resolveInRecord`); a miss is `null`, compared
 * as a plain value: `==` and ordered comparisons against a non-null value are `false`, and `!=`
 * is `true`. A bare `name` miss is falsy, which gives `when:` its missing-is-falsy behavior. A
 * `null` left side of `in`/`not in` is an ordinary value (`miss in [...]` is `false`); a `null`
 * right side throws, since `null` is no container. Pool namespaces are not in scope (they belong
 * in the node's input `from:` bindings).
 *
 * Unlike the strict prompt render, which throws on the same dotted miss, `when:` stays falsy so
 * the CASE's `else:` route can fire.
 *
 * @param types The type table `isinstance(x, T)` and a constructor look `T` up in; `null` means
 *   no types.
 * @throws {ExpressionError} As `evaluateWhen` does: a call to a function that isn't a builtin,
 *   or to a builtin with arguments it doesn't take, throws even where `and`/`or` would not
 *   evaluate the call.
 */
export function evaluateWhenRecord(
  expression: string,
  record: ValueRecord,
  types: ReadonlyMap<string, Type> | null = null,
): boolean {
  return evaluate(expression, (path) => resolveInRecord(path, record), types);
}

// --------------------------------------------------------------------------------------- //
// The one evaluator: `evalExpr` walks a `parseExpr` tree with a `resolve` callable. It is the
// single `${...}` evaluator: the binding path (coalesce/default/required), the
// `when:`/`asserts:` boolean path, and the prompt builtin-call path all parse with `parseExpr`
// and evaluate here.
// --------------------------------------------------------------------------------------- //

/**
 * How a missing reference (one `resolve` maps to `null`) is treated.
 *
 * - `STRICT_RAISE`: a missing reference is an error: throw `ExpressionError`. For a context
 *   where every reference must be present (a strict render). A miss on a non-final operand of
 *   `|`, `:-`, or `:?` is still `null`, so the operator fires.
 * - `BINDING_NONE`: a missing reference is `null`, the binding contract (a coalesce or default
 *   may then fire, or the whole value is `null`).
 * - `CONDITION_FALSY`: a missing reference is `null`, which is falsy through comparisons, so a
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
  // `Number(x) / Number(y)` would round twice, while Python's int / int is correctly rounded.
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
  // |x| = m · 2^e exactly, read off its bits.
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const bits = view.getBigUint64(0);
  const field = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  const m = field === 0 ? fraction : fraction | (1n << 52n);
  const e = field === 0 ? -1074 : field - 1075;
  const power = m ** BigInt(Math.abs(n));
  let num = n > 0 ? power : 1n;
  let den = n > 0 ? 1n : power;
  if (e * n >= 0) num <<= BigInt(e * n);
  else den <<= BigInt(-e * n);
  const result = roundQuotient(num, den);
  return x < 0 && n % 2 !== 0 ? -result : result;
}

/**
 * Return the float nearest `num / den` for a non-negative `num` and a positive `den`: ties to
 * even, subnormals included, `Infinity` past the float range.
 */
function roundQuotient(num: bigint, den: bigint): number {
  if (num === 0n) return 0;
  const bitLength = (value: bigint) => value.toString(2).length;
  // Scale so the integer quotient q has at least 55 bits: num / den = (q + a remainder) · 2^-s.
  const s = 55 - (bitLength(num) - bitLength(den));
  const [scaledNum, scaledDen] = s >= 0 ? [num << BigInt(s), den] : [num, den << BigInt(-s)];
  const q = scaledNum / scaledDen;
  const sticky = scaledNum % scaledDen !== 0n;
  const k = bitLength(q);
  // q's leading bit is worth 2^(k - 1 - s).
  if (k - 1 - s > 1023) return Infinity;
  // Keep 53 bits, or fewer for a subnormal, whose lowest bit is worth 2^-1074.
  const drop = Math.max(k - 53, s - 1074);
  const rest = q & ((1n << BigInt(drop)) - 1n);
  const half = 1n << BigInt(drop - 1);
  let mantissa = q >> BigInt(drop);
  if (rest > half || (rest === half && (sticky || (mantissa & 1n) === 1n))) mantissa += 1n;
  // Exact: mantissa <= 2^53 and the scale is a power of two in [2^-1074, 2^971], so only a
  // mantissa rounded up to 2^53 at the top overflows, to Infinity as IEEE rounding does.
  return Number(mantissa) * 2 ** (drop - s);
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

/** Return whether a value is truthy: all but `false`, `null`, a zero, `""`, `[]` and a record with no field. */
function truthy(value: Value): boolean {
  if (value === null || value === false || value === 0n || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return true;
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
 * Describe a chain from its operand routes, e.g. `(a + b - c)`.
 *
 * @param operators The operator symbols between the operands.
 * @param routes The operand routes, one more than `operators`.
 * @returns The operand expressions joined by their operators, in parentheses.
 */
function chainText(operators: readonly string[], routes: readonly ValueRoute[]): string {
  const parts = [routes[0]!.expression];
  operators.forEach((symbol, i) => parts.push(symbol, routes[i + 1]!.expression));
  return "(" + parts.join(" ") + ")";
}

/**
 * Trace a `+` chain of lists as a balanced tree of two-child `concat` routes.
 *
 * Two-child routes keep the run-record format of a single `left + right`, and the balanced split
 * keeps a long chain's route about `log2(n)` deep.
 *
 * @param routes The operand routes, in order.
 * @param sizes Each operand list's length.
 * @returns The lone route for one operand, else a `concat` of the two halves, whose expression is
 *   the halves' operands joined by `+`.
 */
function balancedConcat(routes: readonly ValueRoute[], sizes: readonly number[]): ValueRoute {
  if (routes.length === 1) return routes[0]!;
  const middle = Math.floor(routes.length / 2);
  const total = (part: readonly number[]) => part.reduce((sum, size) => sum + size, 0);
  return new ValueRoute("concat", {
    children: [
      balancedConcat(routes.slice(0, middle), sizes.slice(0, middle)),
      balancedConcat(routes.slice(middle), sizes.slice(middle)),
    ],
    expression: chainText(Array<string>(routes.length - 1).fill("+"), routes),
    sizes: [total(sizes.slice(0, middle)), total(sizes.slice(middle))],
  });
}

const DESCRIBE_SYMBOLS: Readonly<Record<string, string>> = {
  or_expr: "or",
  and_expr: "and",
  compare_in: "in",
  compare_notin: "not in",
};

/** Describe a node from its evaluated operands' routes, without another expression walk. */
function describe(node: Tree, children: readonly ValueRoute[]): string {
  const kind = node.data;
  if (kind === "sum" || kind === "product" || kind === "power") return chainText(splitChain(node)[1], children);
  const symbol = kind === "compare" ? String(node.children[1]) : DESCRIBE_SYMBOLS[kind];
  if (symbol !== undefined) return "(" + children.map((child) => child.expression).join(` ${symbol} `) + ")";
  if (kind === "neg" || kind === "negate") {
    return `(${kind === "neg" ? "-" : "not "}${children[0]!.expression})`;
  }
  return `${kind}(${children.map((child) => child.expression).join(", ")})`;
}

/** Name a slice's base in a miss path: a literal's text, a comprehension variable's name, else `(...)`. */
function sliceBaseText(node: ExprNode): string {
  if (node instanceof Token) return node.value;
  if (node.data === "comp_var" && node.children.length === 1) return String(node.children[0]);
  return "(...)";
}

/**
 * The route of a read: a `source` for a `SourceRef`, a `selected` over a prior route, else a
 * literal of the value.
 */
function resolvedRoute(resolved: ResolvedValue, path: string): ValueRoute {
  const source = resolved.source;
  if (source instanceof SourceRef) return new ValueRoute("source", { source, expression: path });
  if (source instanceof ValueRoute) return new ValueRoute("selected", { children: [source], expression: path });
  return ValueRoute.literal(resolved.value);
}

/** The routes collected by one traced subtree evaluation: its operands', and its own if it sets one. */
interface TraceFrame {
  children: ValueRoute[];
  route: ValueRoute | null;
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
  readonly #resolve: (path: string) => Value | ResolvedValue;
  readonly #item: Value | ResolvedValue;
  readonly #mode: ResolveMode;
  readonly #types: ReadonlyMap<string, Type> | null;
  readonly #trace: boolean;
  readonly #frames: TraceFrame[] = [];
  #lastRoute: ValueRoute | null = null;
  #missPath: string | null = null;
  // Comprehension variables in scope: name -> its item and, when tracing, the item's route.
  readonly #vars = new Map<string, { value: Value; route: ValueRoute | null }>();

  constructor(
    resolve: (path: string) => Value | ResolvedValue,
    item: Value | ResolvedValue,
    mode: ResolveMode,
    types: ReadonlyMap<string, Type> | null,
    trace: boolean,
  ) {
    this.#resolve = resolve;
    this.#item = item;
    this.#mode = mode;
    this.#types = types;
    this.#trace = trace;
  }

  // --- top-level entry: turn a `MISSING` head into a value per mode --- //
  eval(node: ExprNode): Value | TracedValue {
    const value = this.asValue(this.walk(node));
    return this.#trace ? { value, route: this.#lastRoute! } : value;
  }

  /** Turn `MISSING` into `null` in the non-strict modes; throw in `STRICT_RAISE`. */
  asValue(value: Walked): Value {
    if (value === MISSING) {
      if (this.#mode === ResolveMode.STRICT_RAISE) {
        const where = this.#missPath !== null ? ` \${${this.#missPath}}` : "";
        throw new ExpressionError(`unresolved reference${where} in expression`);
      }
      return null;
    }
    return value;
  }

  /**
   * Turn `MISSING` into `null` in every mode: an operand whose miss its consumer handles (a
   * `|` / `:-` / `:?` operand, a comprehension iterable, a slice bound). A handled miss's path is
   * dropped, so a later error can't name it.
   */
  orNone(value: Walked): Value {
    if (value !== MISSING) return value;
    this.#missPath = null;
    return null;
  }

  // --- dispatch --- //
  /**
   * Evaluate one subtree: the single hook every subtree goes through. When tracing, it also
   * builds the subtree's route: the one its handler set, else a `scalar` over its operands'
   * routes, else a literal of its value; and adds it to the enclosing subtree's operand routes.
   */
  walk(node: ExprNode): Walked {
    if (!this.#trace) return this.dispatch(node);
    const frame: TraceFrame = { children: [], route: null };
    this.#frames.push(frame);
    let value: Walked;
    try {
      value = this.dispatch(node);
    } finally {
      this.#frames.pop();
    }
    let route = frame.route;
    if (route === null) {
      const captured = value === MISSING ? null : value;
      // A token has no operands, so a frame with children is a tree's.
      route =
        frame.children.length > 0
          ? ValueRoute.literal(captured).with({
              kind: "scalar",
              children: frame.children,
              expression: describe(node as Tree, frame.children),
            })
          : ValueRoute.literal(captured);
    }
    this.#lastRoute = route;
    this.#frames.at(-1)?.children.push(route);
    return value;
  }

  dispatch(node: ExprNode): Walked {
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
    if (callSuffix === undefined) return this.reference(referencePath(node), referenceText(node));
    const constructor = this.isConstructor(name);
    let value: Walked;
    if (name === ISINSTANCE) value = this.isinstanceCall(callSuffix);
    else if (constructor) value = this.construct(name, callSuffix);
    else value = this.callBuiltin(name, callSuffix);
    for (const step of steps) value = this.dottedStep(value, step.name);
    if (value === MISSING) this.#missPath = `${name}(...)` + stepsText(steps);
    if (this.#trace) {
      const frame = this.#frames.at(-1)!;
      if (constructor) {
        frame.route = this.constructorRoute(name, callSuffix).project(steps.map((step) => step.name));
      } else {
        const routes = frame.children;
        const names = valueArgs(name, callSuffix).map((arg) =>
          arg.children.length === 2 ? String(arg.children[0]) : "",
        );
        let args = names.map((key, i) => (key ? `${key}=${routes[i]!.expression}` : routes[i]!.expression)).join(", ");
        if (name === ISINSTANCE) args += `, ${isinstanceTypeName(callSuffix)}`;
        frame.route = ValueRoute.literal(value === MISSING ? null : value).with({
          kind: "opaque",
          children: routes,
          names,
          expression: `${name}(${args})${stepsText(steps)}`,
        });
      }
    }
    return value;
  }

  /**
   * Resolve a dotted reference path. An `item`-headed path walks the MAP-body `item` scope
   * locally (record fields and file fields, see `dottedStep`); every other path goes whole to
   * `resolve`, which owns the dotted walk. A miss (an `item` miss, or `resolve` giving `null`)
   * is `MISSING`.
   *
   * @param path The flat dotted path, for the resolver and the `item` walk.
   * @param text The read as written (`?.` included), for miss messages and routes.
   */
  reference(path: string, text: string = path): Walked {
    const parts = path.split(".");
    if (parts[0] === "item") {
      // MAP-body-local scope (not a resolver head); a `null` item means no item scope.
      const item = this.#item;
      let present: boolean;
      let value: Walked;
      let source: SourceRef | ValueRoute | null = null;
      if (this.#trace && item instanceof ResolvedValue) {
        ({ present, value, source } = item);
      } else {
        present = item !== null;
        value = item as Value;
      }
      if (value === null || !present) value = MISSING;
      const steps = parts.slice(1);
      for (const step of steps) {
        if (source !== null) present = present && value !== MISSING && fieldOf(value, step)[0];
        value = this.dottedStep(value, step);
      }
      if (this.#trace && source === null) {
        this.#frames.at(-1)!.route = ValueRoute.literal(value === MISSING ? null : value).with({ expression: text });
      } else if (this.#trace && present) {
        this.#frames.at(-1)!.route = resolvedRoute(item as ResolvedValue, "item").project(steps);
      }
      if (value === MISSING) this.#missPath = text;
      return value;
    }
    let resolved = this.#resolve(path);
    if (this.#trace) {
      if (!(resolved instanceof ResolvedValue)) throw new TypeError("a traced resolver must return ResolvedValue");
      if (resolved.present) this.#frames.at(-1)!.route = resolvedRoute(resolved, text);
      resolved = resolved.present ? resolved.value : null;
    }
    if (resolved === null) {
      this.#missPath = text;
      return MISSING;
    }
    return resolved as Value;
  }

  /**
   * `base[lower:upper]` on a list, as Python's `slice.indices` with step 1: an omitted lower is 0
   * and an omitted upper the length, a negative bound counts from the end, and both clamp to
   * `[0, n]`. A null or missing base is a miss, as a `.field` on null is, and its bounds aren't
   * read; a null base's miss is named `base[...]`, with a non-literal, non-variable base as
   * `(...)`. When tracing, the route is a `list` of the base route projected to each kept index.
   *
   * @throws {ExpressionError} If a bound is null or not an int (a bool counts as 0 or 1), or the
   *   base is not a list.
   */
  subscript(node: Tree): Walked {
    const [baseNode, slice] = node.children as [ExprNode, Tree];
    const base = this.walk(baseNode);
    // A miss before the bounds are read, so `xs[n:] :- []` falls back whatever the bounds are.
    if (base === MISSING) return MISSING;
    if (base === null) {
      this.#missPath = `${sliceBaseText(baseNode)}[...]`;
      return MISSING;
    }
    const baseRoute = this.#trace ? this.#lastRoute! : null;
    const lower = this.sliceBound(slice.children[0]!);
    const upper = this.sliceBound(slice.children[1]!);
    if (!Array.isArray(base)) throw new ExpressionError(`slicing takes a list, got ${kindName(base)}`);
    const n = BigInt(base.length);
    const clamp = (bound: bigint | null, fallback: bigint): number => {
      let at = bound ?? fallback;
      if (at < 0n) at += n;
      return Number(at < 0n ? 0n : at > n ? n : at);
    };
    const start = clamp(lower, 0n);
    const stop = Math.max(start, clamp(upper, n));
    if (baseRoute !== null) {
      // Each kept item is its base item, so feedback on it reaches that item.
      const children = Array.from({ length: stop - start }, (_, k) => baseRoute.project([start + k]));
      this.#frames.at(-1)!.route = new ValueRoute("list", {
        children,
        expression: "[" + children.map((child) => child.expression).join(", ") + "]",
      });
    }
    return base.slice(start, stop);
  }

  /**
   * A list comprehension, as Python's nested loops: the first iterable in the enclosing scope,
   * each later clause once per binding of the clauses before it, an `if` keeping a binding when
   * its condition is truthy, and the elem appended after the last clause. The clauses run depth
   * first without recursion, so a long comprehension can't overflow the stack. When tracing, the
   * route is a `list` of the elems' routes, and a variable carries its item's route, the
   * iterable's route projected to the item, so feedback on an elem reaches that item. Conditions
   * aren't part of the route, as a CASE condition isn't.
   *
   * @throws {ExpressionError} If an iterable is null, misses, or is not a list.
   */
  listComp(node: Tree): Value[] {
    const [elem, ...clauses] = node.children as [ExprNode, ...Tree[]];
    const out: Value[] = [];
    const routes: ValueRoute[] = [];
    // One entry per active `for`: its clause index, its items, the next index, and its iterable's route.
    const loops: { at: number; items: Value[]; k: number; route: ValueRoute | null }[] = [];
    // Move to the next binding of the innermost loop that has one; return the next clause index, or -1.
    const next = (): number => {
      while (loops.length > 0) {
        const loop = loops.at(-1)!;
        const name = String(clauses[loop.at]!.children[0]);
        if (loop.k < loop.items.length) {
          const route = loop.route?.project([loop.k]) ?? null;
          this.#vars.set(name, { value: loop.items[loop.k++]!, route });
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
        if (this.#trace) routes.push(this.#lastRoute!);
        i = next();
      } else if (clauses[i]!.data === "comp_for") {
        const [target, iterable, source] = clauses[i]!.children as [Token, ExprNode, Token];
        const items = this.orNone(this.walk(iterable));
        if (items === null) {
          throw new ExpressionError(
            `comprehension \`for ${target} in ${source}\` iterates over null; give a default, e.g. \`(${source} :- [])\``,
          );
        }
        if (!Array.isArray(items)) {
          throw new ExpressionError(`comprehension \`for ${target} in ${source}\` takes a list, got ${kindName(items)}`);
        }
        loops.push({ at: i, items, k: 0, route: this.#trace ? this.#lastRoute : null });
        i = next();
      } else {
        i = truthy(this.asValue(this.walk(clauses[i]!.children[0]!))) ? i + 1 : next();
      }
    }
    if (this.#trace) {
      this.#frames.at(-1)!.route = new ValueRoute("list", {
        children: routes,
        expression: "[" + routes.map((route) => route.expression).join(", ") + "]",
      });
    }
    return out;
  }

  /**
   * A comprehension variable read, then any `.field` trailers as `dottedStep` walks them. When
   * tracing, its route is its item's route projected through the trailers.
   */
  compVar(node: Tree): Walked {
    const name = String(node.children[0]);
    const binding = this.#vars.get(name);
    // Only a hand-built tree can read a variable outside its comprehension.
    if (binding === undefined) throw new ExpressionError(`comprehension variable ${reprText(name)} read outside its comprehension`);
    const { steps } = splitReference(node);
    // A null item is returned as is, as in Python; a `.field` on it misses.
    let value: Walked = binding.value;
    for (const step of steps) value = this.dottedStep(value, step.name);
    if (value === MISSING) this.#missPath = name + stepsText(steps);
    else if (this.#trace && binding.route !== null) {
      this.#frames.at(-1)!.route = binding.route.project(steps.map((step) => step.name));
    }
    return value;
  }

  /** One slice bound: `null` when omitted, else its int (a bool as 0 or 1). */
  sliceBound(node: ExprNode): bigint | null {
    if (node instanceof Tree && node.data === "omitted") return null;
    const value = this.orNone(this.walk(node));
    if (value === null) throw new ExpressionError("slice bound is null; omit it to mean the end");
    if (typeof value === "boolean") return value ? 1n : 0n;
    if (typeof value !== "bigint") throw new ExpressionError(`slice bounds take int, got ${kindName(value)}`);
    return value;
  }

  /**
   * One dotted step: `fieldOf` only (a record's own field, or a file's `uri` / `mime` /
   * `name`), never a property read (safety-critical: so `${x.constructor}` can never reach a JS
   * property). Any other value, a missing field, or a `null` field is `MISSING`, and so is a
   * step into `MISSING`.
   */
  dottedStep(value: Walked, step: string): Walked {
    if (value === MISSING) return MISSING;
    const stepped = fieldOf(value, step)[1];
    return stepped === null ? MISSING : stepped;
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
   * The route of a constructor call, from its argument routes.
   *
   * `T(x)` selects its argument's route, so feedback passes through to `x`; the keyword form is a
   * record route, so feedback on a field reaches that field's argument.
   */
  constructorRoute(name: string, callSuffix: Tree): ValueRoute {
    const routes = this.#frames.at(-1)!.children;
    const args = callSuffix.children as Tree[];
    if (args.length === 1 && args[0]!.children.length === 1) {
      return new ValueRoute("selected", { children: routes, expression: `${name}(${routes[0]!.expression})` });
    }
    const names = args.map((arg) => String(arg.children[0]));
    const expression = names.map((key, i) => `${key}=${routes[i]!.expression}`).join(", ");
    return ValueRoute.record(names.map((key, i) => [key, routes[i]!] as const)).with({
      expression: `${name}(${expression})`,
    });
  }

  /**
   * Run a builtin call through `TEMPLATE_FNS`.
   *
   * The arguments are evaluated in source order (a keyword argument carries a leading `NAME`),
   * bound to the builtin's parameters with `bindArguments`, and passed in parameter order. The
   * result is read as a host value: a top-level `undefined` is `null`, anything else goes
   * through `fromHost`.
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
      return result === undefined ? null : fromHost(result, "result");
    } catch (error) {
      if (error instanceof ExpressionError) throw error;
      throw failed(error instanceof Error ? error.message : String(error), { cause: error });
    }
  }

  // --- coalesce / default / required (short-circuit) --- //
  /** When tracing, set the current subtree's route to a `selected` over the last walked subtree's. */
  selectRoute(expression: string): void {
    if (this.#trace) {
      this.#frames.at(-1)!.route = new ValueRoute("selected", { children: [this.#lastRoute!], expression });
    }
  }

  /**
   * `a | b | c`: the first operand that resolves to a non-null value wins; a present falsy
   * value (`0`, `""`, `false`) wins. Only the last operand's miss is subject to `STRICT_RAISE`.
   */
  coalesce(node: Tree): Value {
    const last = node.children.length - 1;
    for (const [i, child] of node.children.entries()) {
      const walked = this.walk(child);
      const value = i === last ? this.asValue(walked) : this.orNone(walked);
      if (value !== null) {
        this.selectRoute("|");
        return value;
      }
    }
    if (this.#trace) this.#frames.at(-1)!.route = ValueRoute.literal(null);
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
    this.selectRoute(":-");
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
    this.selectRoute(":?");
    return value;
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
    // The operand values, kept only when tracing (for the route).
    const values: Value[] | null = this.#trace ? [acc] : null;
    for (const [i, op] of operators.entries()) {
      const rhs = this.asValue(this.walk(operands[i + 1]!));
      acc = applyStep(op, acc, rhs);
      values?.push(rhs);
    }
    if (
      values !== null &&
      node.data === "sum" &&
      operators.every((op) => op === "+") &&
      values.every((value) => Array.isArray(value))
    ) {
      const frame = this.#frames.at(-1)!;
      frame.route = balancedConcat(frame.children, values.map((value) => (value as Value[]).length));
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
    const value = node.children.map((c) => this.asValue(this.walk(c)));
    if (this.#trace) {
      const frame = this.#frames.at(-1)!;
      frame.route = new ValueRoute("list", {
        children: frame.children,
        expression: "[" + frame.children.map((child) => child.expression).join(", ") + "]",
      });
    }
    return value;
  }
}

/** The options of `evalExpr` besides `trace`. */
export interface EvalOptions {
  item?: Value;
  mode?: ResolveMode;
  types?: ReadonlyMap<string, Type> | null;
}

/**
 * The options of an `evalExpr` that may trace: `item` may be a `ResolvedValue`, and `trace` turns
 * tracing on.
 */
export interface TraceOptions extends Omit<EvalOptions, "item"> {
  item?: Value | ResolvedValue;
  trace?: boolean;
}

/**
 * Evaluate a unified `${...}` expression tree (from `parseExpr`).
 *
 * The one evaluator for every `${...}` construct: references (bare, dotted, `item`-scoped; `?.`
 * steps read as `.` here, see `splitReference`),
 * arithmetic and comparisons over values (so text + text and list + list concatenate), boolean
 * combinators, list literals, list slices (`xs[a:b]`, as Python's with step 1; a slice of a
 * miss or `null` is a miss), list comprehensions (as Python's; an elem miss is a `null` item and
 * a condition miss is falsy outside strict mode, and a `null` item is returned as is), and
 * coalesce / default / required. The operators follow the TS
 * number model: an int is a `bigint` at any size, a bool counts as 0 or 1, `/` always gives a
 * float, and a float result is IEEE's (`Infinity` is a value, NaN an error).
 *
 * Three semantics support a CASE's `else:` routing: a missing reference is `null` (in the
 * non-strict modes); arithmetic over that `null` throws; an ordered comparison with a `null`
 * operand is `false`.
 *
 * Dotted access reads only a record's own field or a file's `uri`, `mime` or `name`
 * (`fieldOf`), never a JS property, so `${x.constructor}` is a miss (safety-critical).
 *
 * @param tree The parsed expression: a `Tree`, or a lone literal `Token`.
 * @param resolve Resolves a reference path to its value; `null` is a miss. `item`-headed paths
 *   bypass it. With `trace`, it returns a `ResolvedValue` holding the value, whether it is
 *   present, and its `SourceRef` or prior `ValueRoute`.
 * @param options.item The MAP-body-local scope for `item` / `item.path` (record and file
 *   fields); `null` (the default) means no item scope. With `trace`, a `ResolvedValue` may give
 *   the item's presence and its route through a composed collection.
 * @param options.mode How a missing reference is treated (default `BINDING_NONE`).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in, keyed by type name; `null` (the default) means no types.
 * @param options.trace Record the value's route while evaluating, and return both as a
 *   `TracedValue` (default `false`).
 * @returns The value: `null`, a boolean, a `bigint`, a number, text, a list, a record or a file
 *   (`FileRef`). With `trace`, the same value with its route.
 * @throws {RequiredError} When a `head :? message` head misses or is `null`.
 * @throws {ConstructorTypeError} When a `T(...)` constructor's value does not fit `T`.
 * @throws {ExpressionError} On an unsupported node, a call to an unknown function or one that
 *   fails, an `isinstance` type absent from `types`, arithmetic or ordering over operand kinds
 *   the operator doesn't take (`null` included), a zero divisor, a NaN result, an int too large
 *   for the runtime, a slice of anything but a list or with a bound that is null or not an int,
 *   a comprehension over anything but a list (`null` or a miss included), or a strict-mode miss.
 * @throws {TypeError} When a traced resolver doesn't return a `ResolvedValue`.
 */
export function evalExpr(
  tree: ExprNode,
  resolve: (path: string) => ResolvedValue,
  options: TraceOptions & { trace: true },
): TracedValue;
export function evalExpr(
  tree: ExprNode,
  resolve: (path: string) => Value,
  options?: EvalOptions & { trace?: false },
): Value;
export function evalExpr(
  tree: ExprNode,
  resolve: (path: string) => Value | ResolvedValue,
  options?: TraceOptions,
): Value | TracedValue;
export function evalExpr(
  tree: ExprNode,
  resolve: (path: string) => Value | ResolvedValue,
  {
    item = null,
    mode = ResolveMode.BINDING_NONE,
    types = null,
    trace = false,
  }: TraceOptions = {},
): Value | TracedValue {
  return new EvalExpr(resolve, item, mode, types, trace).eval(tree);
}

// --------------------------------------------------------------------------------------- //
// Conservative compile-time type inference over the unified expression tree.
// --------------------------------------------------------------------------------------- //

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
 * boolean combinators, and `isinstance(x, T)` have fixed result types, and a `T(...)`
 * constructor has type `T` (its value is checked against `T`), with field reads on the result
 * typed through `T`. A read with a `?.` step (`a?.b`) is `Optional` of the field. Other forms
 * stay unknown until their runtime value semantics can be modeled without guessing. Reference
 * and constructor lookup are injected so this source-neutral expression layer knows nothing of a
 * flow graph or compose document.
 *
 * Example: `inferExprType(parseExpr("answer == 42"), () => Type.scalar(ValueKind.INTEGER))`
 * gives `Type.scalar(ValueKind.BOOLEAN)`.
 *
 * @param tree A tree or token from `parseExpr`.
 * @param refTypeFn Resolves one dotted reference path to its known type, or gives `null` when
 *   the type is not statically known. Anything it throws propagates unchanged.
 * @param typeFn Resolves a constructor name to its type, or gives `null` for a name that is no
 *   constructor; `null` leaves every constructor unknown.
 * @returns The proven expression type, or `null` when inference is inconclusive.
 */
export function inferExprType(
  tree: ExprNode,
  refTypeFn: (path: string) => Type | null,
  typeFn: ((name: string) => Type | null) | null = null,
): Type | null {
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
    if (callSuffixOf(tree) !== undefined) {
      // `parseExpr` rejects dotted access on an `isinstance(...)` result.
      if (name === ISINSTANCE) return Type.scalar(ValueKind.BOOLEAN);
      let typ = typeFn === null || TEMPLATE_FNS.has(name) ? null : typeFn(name);
      for (const step of steps) {
        if (typ === null) return null;
        typ = fieldReadType(typ, step.name);
        if (typ !== null && step.optional) typ = typ.replace({ nullable: true });
      }
      return typ;
    }
    const typ = refTypeFn(referencePath(tree));
    return typ !== null && steps.some((step) => step.optional) ? typ.replace({ nullable: true }) : typ;
  }
  return BOOLEAN_NODES.has(tree.data) ? Type.scalar(ValueKind.BOOLEAN) : null;
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
  if (typ.variants !== undefined) {
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

/**
 * Return the `arg` subtrees of a call whose values are expressions.
 *
 * Every argument of a builtin is an expression; `isinstance(x, T)` has only `x`.
 */
export function valueArgs(callee: string, callSuffix: Tree): Tree[] {
  const args = callSuffix.children as Tree[];
  return callee === ISINSTANCE ? args.slice(0, 1) : args;
}

/** Return the type name of a validated `isinstance(x, T)` call suffix. */
export function isinstanceTypeName(callSuffix: Tree): string {
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
  const names: string[] = [];
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall" || String(node.children[0]) !== ISINSTANCE) continue;
    const callSuffix = node.children.at(-1);
    if (callSuffix instanceof Tree && callSuffix.data === "call_suffix") {
      const name = isinstanceTypeName(callSuffix);
      if (!names.includes(name)) names.push(name);
    }
  }
  return names;
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
  const names: string[] = [];
  for (const node of iterSubtreesTopdown(tree)) {
    if (node.data !== "refcall" || callSuffixOf(node) === undefined) continue;
    const name = String(node.children[0]);
    if (name !== ISINSTANCE && !TEMPLATE_FNS.has(name) && !names.includes(name)) names.push(name);
  }
  return names;
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
  if (node instanceof Token) return; // a lone NUMBER / STRING / BOOL / NULL literal names no reference
  if (node.data === "refcall") {
    // Keep in sync with the evaluator's refcall decomposition, `collectRefEdits` and
    // `conditionReadLeaves` in `template.ts`: each finds the call_suffix and takes the path from
    // `splitReference`.
    const children = node.children;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) {
      refs.push(referencePath(node));
      return;
    }
    // A builtin call: the callee contributes no reference; walk each argument value only.
    for (const arg of valueArgs(String(children[0]), callSuffix)) {
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
  if (node instanceof Token) return; // a lone literal holds no reference
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
