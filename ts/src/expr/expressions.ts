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
 * So far this holds the error family, the condition parse with its call check, and the static
 * reference walkers and rewriters. The evaluator and type inference land here later, and value
 * tracing with `lineage.ts`.
 */

import { reprText } from "../diagnostics.ts";
import { BUILTIN_TYPE_NAMES } from "../typesys/index.ts";
import { valueRepr } from "../typesys/values.ts";
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
