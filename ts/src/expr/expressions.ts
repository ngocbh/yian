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
 * So far this holds the error family, the condition parse with its call check, the evaluator
 * (`evalExpr`) without calls, and the static reference walkers and rewriters. Calls and type
 * inference land here later, and value tracing with `lineage.ts`.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { BUILTIN_TYPE_NAMES, FileRef, type Type, type Value } from "../typesys/index.ts";
import { fieldOf, isRecord, valueRepr } from "../typesys/values.ts";
import { bindArguments, TEMPLATE_FNS, type TemplateFn } from "./builtins.ts";
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

/** A `when:` expression could not be parsed or evaluated. */
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

/** Return a value's kind as messages name it: `null`, `bool`, `int`, `float`, `str`, `list`, `record` or `file`. */
function kindName(value: Value): string {
  if (value === null) return "null";
  switch (typeof value) {
    case "boolean":
      return "bool";
    case "bigint":
      return "int";
    case "number":
      return "float";
    case "string":
      return "str";
  }
  if (Array.isArray(value)) return "list";
  return FileRef.isFileRef(value) ? "file" : "record";
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
 * @param types The type table constructor names must be in, or `null` to leave them to the
 *   caller.
 * @throws {ExpressionError} If a builtin's parameters can't bind the call's positional count and
 *   keyword names, or, with `types`, a callee is neither a builtin nor in it.
 */
function checkCalls(tree: ExprNode, types: ReadonlyMap<string, unknown> | null): void {
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

// The operator keyword tokens among a boolean combinator's operands. An operand may also be a
// top-level `Token` (a bare literal), so only these token types are skipped.
const OP_TOKEN_TYPES: ReadonlySet<string> = new Set(["NOT", "AND", "OR"]);

/**
 * Recursive walker for a `parseExpr` tree, parameterised by `resolve`, `item` and `mode`.
 * Hand-written so coalesce / default / required can short-circuit: the fallback is evaluated
 * only when the head misses.
 *
 * A reference evaluates to `MISSING` on an unresolved path; every value consumer (operators,
 * comparisons, the top-level entry) maps `MISSING` to `null` in the non-strict modes or throws
 * in `STRICT_RAISE`.
 */
class EvalExpr {
  readonly #resolve: (path: string) => Value;
  readonly #item: Value;
  readonly #mode: ResolveMode;
  readonly #types: ReadonlyMap<string, Type> | null;
  #missPath: string | null = null;

  constructor(
    resolve: (path: string) => Value,
    item: Value,
    mode: ResolveMode,
    types: ReadonlyMap<string, Type> | null,
  ) {
    this.#resolve = resolve;
    this.#item = item;
    this.#mode = mode;
    this.#types = types;
  }

  // --- top-level entry: turn a `MISSING` head into a value per mode --- //
  eval(node: ExprNode): Value {
    return this.asValue(this.walk(node));
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

  /** Turn `MISSING` into `null` in every mode: a `|` / `:-` / `:?` operand whose miss the operator handles. */
  static orNone(value: Walked): Value {
    return value === MISSING ? null : value;
  }

  // --- dispatch --- //
  /**
   * Evaluate one subtree: the single hook every subtree goes through, where value tracing will
   * go with `lineage.ts`.
   */
  walk(node: ExprNode): Walked {
    return this.dispatch(node);
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
   */
  refcall(node: Tree): Walked {
    const children = node.children;
    const name = String(children[0]);
    if (callSuffixOf(node) === undefined) {
      const segments = children.slice(1).map((c) => String((c as Tree).children[0])); // each trailer's NAME
      return this.reference([name, ...segments].join("."));
    }
    throw new ExpressionError("expression calls are not supported yet");
  }

  /**
   * Resolve a dotted reference path. An `item`-headed path walks the MAP-body `item` scope
   * locally (record fields and file fields, see `dottedStep`); every other path goes whole to
   * `resolve`, which owns the dotted walk. A miss (an `item` miss, or `resolve` giving `null`)
   * is `MISSING`.
   */
  reference(path: string): Walked {
    const parts = path.split(".");
    if (parts[0] === "item") {
      // MAP-body-local scope (not a resolver head); a `null` item means no item scope.
      let value: Walked = this.#item === null ? MISSING : this.#item;
      for (const step of parts.slice(1)) value = this.dottedStep(value, step);
      if (value === MISSING) this.#missPath = path;
      return value;
    }
    const resolved = this.#resolve(path);
    if (resolved === null) {
      this.#missPath = path;
      return MISSING;
    }
    return resolved;
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

  // --- coalesce / default / required (short-circuit) --- //
  /**
   * `a | b | c`: the first operand that resolves to a non-null value wins; a present falsy
   * value (`0`, `""`, `false`) wins. Only the last operand's miss is subject to `STRICT_RAISE`.
   */
  coalesce(node: Tree): Value {
    const last = node.children.length - 1;
    for (const [i, child] of node.children.entries()) {
      const walked = this.walk(child);
      const value = i === last ? this.asValue(walked) : EvalExpr.orNone(walked);
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
    const value = EvalExpr.orNone(this.walk(head));
    return value === null ? this.asValue(this.walk(fallback)) : value;
  }

  /**
   * `head :? message`: throw `RequiredError` with the message (text as is, any other value in
   * its text form) when `head` misses or is `null`, in every mode.
   */
  requiredExpr(node: Tree): Value {
    const [head, message] = node.children as [ExprNode, ExprNode];
    const value = EvalExpr.orNone(this.walk(head));
    if (value === null) {
      const text = this.asValue(this.walk(message));
      throw new RequiredError(typeof text === "string" ? text : valueRepr(text));
    }
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
    for (const [i, op] of operators.entries()) {
      const rhs = this.asValue(this.walk(operands[i + 1]!));
      acc = applyStep(op, acc, rhs);
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

/**
 * Evaluate a unified `${...}` expression tree (from `parseExpr`).
 *
 * The one evaluator for every `${...}` construct: references (bare, dotted, `item`-scoped),
 * arithmetic and comparisons over values (so text + text and list + list concatenate), boolean
 * combinators, list literals, and coalesce / default / required. The operators follow the TS
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
 * @param tree The parsed expression: a `Tree`, or a lone literal or reference `Token`.
 * @param resolve Resolves a reference path to its value; `null` is a miss. `item`-headed paths
 *   bypass it.
 * @param options.item The MAP-body-local scope for `item` / `item.path` (record and file
 *   fields); `null` (the default) means no item scope.
 * @param options.mode How a missing reference is treated (default `BINDING_NONE`).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in, keyed by type name; `null` (the default) means no types.
 * @returns The value: `null`, a boolean, a `bigint`, a number, text, a list, a record or a file
 *   (`FileRef`).
 * @throws {RequiredError} When a `head :? message` head misses or is `null`.
 * @throws {ExpressionError} On an unsupported node, a call, arithmetic or ordering over operand
 *   kinds the operator doesn't take (`null` included), a zero divisor, a NaN result, an int too
 *   large for the runtime, or a strict-mode miss.
 */
export function evalExpr(
  tree: ExprNode,
  resolve: (path: string) => Value,
  {
    item = null,
    mode = ResolveMode.BINDING_NONE,
    types = null,
  }: { item?: Value; mode?: ResolveMode; types?: ReadonlyMap<string, Type> | null } = {},
): Value {
  return new EvalExpr(resolve, item, mode, types).eval(tree);
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
    // Keep in sync with the evaluator's refcall decomposition and `collectRefEdits`: each finds
    // the call_suffix and joins the path (leading NAME + trailer segments) the same way.
    const children = node.children;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) {
      const segments = children.slice(1).map((c) => String((c as Tree).children[0]));
      refs.push([String(children[0]), ...segments].join("."));
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

/**
 * Rewrite the reference-leaf paths of one unified `${...}` expression `text`.
 *
 * The rewrite analog of `exprRefs`: it walks the same parse tree (`refcall` reference /
 * builtin-call arguments), but instead of collecting each reference path it asks `rename(path)`
 * for a replacement and splices it in by source position. `rename` returns the new dotted path,
 * or `null` to leave that reference untouched. Everything that is not a reference leaf
 * (operators, numbers, string and list literals, a builtin call's callee name, parentheses,
 * whitespace) is kept verbatim, as is the type name of an `isinstance(x, T)`. A `refcall`
 * (`a.b`) is replaced with the new path.
 *
 * @param text The expression source (a bare expression, or a span interior).
 * @param rename `path -> newPath`, or `null` to keep.
 * @returns `text` with the renamed reference leaves spliced in.
 * @throws {ExpressionError} If `text` does not parse (the same front end as `parseExpr`).
 */
export function rewriteExprRefs(text: string, rename: (path: string) => string | null): string {
  const edits: [number, number, string][] = [];
  collectRefEdits(parseExpr(text), rename, edits);
  // Splice right to left so earlier offsets stay valid as later ones are replaced.
  edits.sort((a, b) => b[0] - a[0]);
  for (const [start, end, replacement] of edits) text = text.slice(0, start) + replacement + text.slice(end);
  return text;
}

/**
 * Walk `node`, collecting a `[start, end, replacement]` splice for each renamed reference leaf.
 * Keep in sync with `collectExprRefs`: the same refcall / builtin-argument decomposition, but
 * capturing token source positions to edit in place.
 */
function collectRefEdits(
  node: ExprNode,
  rename: (path: string) => string | null,
  edits: [number, number, string][],
): void {
  if (node instanceof Token) return; // a lone literal holds no reference
  if (node.data === "refcall") {
    const children = node.children;
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) {
      const segments = children.slice(1).map((c) => String((c as Tree).children[0]));
      const replacement = rename([String(children[0]), ...segments].join("."));
      if (replacement !== null) {
        const start = (children[0] as Token).startPos;
        const last = children.at(-1)!; // the last trailer, or the leading NAME token itself
        const end = last instanceof Tree ? (last.children[0] as Token).endPos : last.endPos;
        edits.push([start, end, replacement]);
      }
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
 * rewrites the interior; a bare `a > 5` is rewritten in place. `rename(path)` returns the
 * replacement path (or `null` to keep). The value is trimmed first, matching `parseCondition`.
 */
export function rewriteConditionRefs(expression: string, rename: (path: string) => string | null): string {
  const interior = wholeSpanInterior(expression);
  if (interior !== null) return "${" + rewriteExprRefs(interior, rename) + "}";
  return rewriteExprRefs(expression.trim(), rename);
}
