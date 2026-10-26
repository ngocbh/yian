/**
 * The `${...}` template language: binding values and strict prompt rendering.
 *
 * A binding value (a node `input:` entry or a flow `output:` entry) is a template: plain text
 * interspersed with `${...}` spans, with `$$` for a literal `$`.
 *
 * - A value that is exactly one `${...}` resolves to the typed value of that span.
 * - A `${...}` embedded in surrounding text is written into it in the text form (`writeAsText`);
 *   an embedded `null` is the empty string.
 * - A value with no `${...}` is a plain literal (after `$$` -> `$`); a lone `$` stays literal.
 *
 * Literal text is never parsed, so operator characters in free text (`stance (a|b|c)`) stay
 * literal. Only span interiors go through `parseExpr`, the one grammar shared with conditions:
 * references (`node.output.field`, `input.name`), literals, operators, pure builtins, coalesce
 * (`|`), default (`:-`) and required (`:?`). A `${...}` can't appear inside another one's
 * expression: inside a span, values are named directly and grouped with parentheses
 * (`${x :- (y :- "lit")}`); `scanTemplate` rejects a nested span with an error that shows the
 * fixed span.
 *
 * Evaluation is pool-agnostic: `evalBinding` and `evalTemplate` take a `resolve` callable, so
 * this stays a leaf that both the runtime binder and compose's compile-time reference walks may
 * import.
 *
 * Compose's compile-time walks over a binding's source live here too: the reference walk
 * (`exprRefsOf`) and its rewriter (`rewriteTemplateRefs`), the co-skip classifier
 * (`bindingCoSkips`: does a binding skip when all its producers are skipped?), the span call
 * checks (`checkTemplateCalls`, and `flowCallCalleesInSpans`, which finds a flow called inside a
 * span), and the per-occurrence condition reads (`conditionReads`, `rewriteConditionReads`),
 * which tell a gate which data edge each reference would take as a binding.
 *
 * This module also owns the strict AGENT prompt renderer (`renderTemplateRecord`)
 * and its compile-time companion (`promptRefs`), which lists the inputs a prompt reads and
 * rejects a call to anything but a formatting builtin (every builtin but the shipped `len`, which
 * computes a value and belongs in an input binding). A prompt is not a binding (it reads inputs
 * already bound to its node and mints no edge), but it is a `${...}` template and reuses this
 * scanner, so it lives here rather than in `expressions.ts`.
 *
 * With `trace`, `evalTemplate` and `evalBinding` also record the value's `ValueRoute`: a whole
 * single span's route, or an `interpolation` route over each embedded span's route with its
 * half-open span of code points in the rendered text, so feedback on the text can go back to its
 * sources.
 *
 * This module also holds the call-parsing helpers that compose's `call(...)` directive reuses:
 * `InlineCall`, the record a desugared call produces, and `findParenEnd` and `splitCallsAware`,
 * the paren-, quote- and span-aware splitters the directive's recognizer uses.
 *
 * Knows about: `expressions` (the evaluator, `ResolveMode`, `exprRefs`, `rewriteExprRefs`,
 * `checkCalls`, `valueArgs`, `conversionHint`, `ISINSTANCE`, `resolveInRecord`, `RootResolver`,
 * `ExpressionError`, `EvalOptions`, `TraceOptions`), `lineage` (`ValueRoute`, `TracedValue`),
 * `builtins` (`TEMPLATE_FNS`, `isNonPromptBuiltin`), `grammar` (`parseExpr`, `findSpanEnd`,
 * `rejectNestedSpans`, `iterSubtrees`, `callSuffixOf`, `wholeSpanInterior`, `ExprNode`, `Token`,
 * `Tree`), `typesys` (`writeAsText`) and `diagnostics` (`reprText`, `compareCodeUnits`). Never
 * imports: `nodes`, `compile`, `compose`, `runtime`, or the pool (pool-agnostic via `resolve`).
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import type { Value, ValueRecord } from "../typesys/index.ts";
import { writeAsText } from "../typesys/values.ts";
import { isNonPromptBuiltin, TEMPLATE_FNS } from "./builtins.ts";
import {
  checkCalls,
  conversionHint,
  type EvalOptions,
  evalExpr,
  ExpressionError,
  exprRefs,
  ISINSTANCE,
  namesEnd,
  type RefRename,
  type RefRenamed,
  ResolveMode,
  referencePath,
  referenceText,
  resolveInRecord,
  type RootResolver,
  rewriteExprRefs,
  type TraceOptions,
  valueArgs,
} from "./expressions.ts";
import {
  callSuffixOf,
  type ExprNode,
  findSpanEnd,
  iterSubtrees,
  parseExpr,
  rejectNestedSpans,
  type Token,
  Tree,
  wholeSpanInterior,
} from "./grammar.ts";
import { type TracedValue, ValueRoute } from "./lineage.ts";

/** A literal run of a template: raw text with `$$` already collapsed to `$`, never parsed. */
export class Literal {
  readonly text: string;

  constructor(text: string) {
    this.text = text;
  }
}

/** A `${...}` span of a template: its interior parsed by `parseExpr`. */
export class Span {
  readonly tree: ExprNode;

  constructor(tree: ExprNode) {
    this.tree = tree;
  }
}

/** A segment of a scanned template: a literal run or a parsed `${...}` span. */
export type TemplateSegment = Literal | Span;

/**
 * Scan a template into its literal runs and `${...}` spans.
 *
 * Only each span's interior is parsed; literal text between spans is kept raw, with `$$`
 * collapsed to `$`. A lone `$` that doesn't start a `${` stays literal.
 *
 * @param text The template: literal text interspersed with `${...}` spans.
 * @returns The segments in order: one `Literal` per literal run (never two in a row) and one
 *   `Span` per `${...}`. A pure-literal template is one `Literal`; an empty one has none.
 * @throws {ExpressionError} On an unbalanced `${`, a `${...}` nested in a span's expression, or
 *   a span interior that doesn't parse.
 */
export function scanTemplate(text: string): TemplateSegment[] {
  const segments: TemplateSegment[] = [];
  let literal = "";
  let i = 0;
  while (i < text.length) {
    if (text[i] === "$" && text[i + 1] === "$") {
      literal += "$";
      i += 2;
    } else if (text[i] === "$" && text[i + 1] === "{") {
      if (literal) {
        segments.push(new Literal(literal));
        literal = "";
      }
      const end = findSpanEnd(text, i + 2);
      if (end === null) throw new ExpressionError(`unbalanced '\${' in template ${reprText(text)}`);
      const interior = text.slice(i + 2, end);
      // Checked before parseExpr so the error shows the whole fixed span.
      rejectNestedSpans(interior, { inSpan: true });
      segments.push(new Span(parseExpr(interior)));
      i = end + 1;
    } else {
      literal += text[i];
      i += 1;
    }
  }
  if (literal) segments.push(new Literal(literal));
  return segments;
}

/**
 * One desugared inline call: the synth node `id`, the `callee` name and its keyword `args` (name
 * to a YAML-scalar literal value or a `${...}` binding text).
 */
export class InlineCall {
  readonly id: string;
  readonly callee: string;
  readonly args: ValueRecord;

  constructor(id: string, callee: string, args: ValueRecord) {
    this.id = id;
    this.callee = callee;
    this.args = args;
    Object.freeze(this);
  }
}

/**
 * Find the `)` closing a call whose arguments begin at `start`.
 *
 * Paren depth is counted, skipping `${...}` spans (their inner parens) and quoted text.
 *
 * @param text The text holding the call.
 * @param start The index just past the call's `(`.
 * @returns The index of the closing `)`, or `null` when the parens, a quote or a span never close.
 */
export function findParenEnd(text: string, start: number): number | null {
  let depth = 1;
  let quote: string | null = null;
  let i = start;
  while (i < text.length) {
    const c = text[i];
    if (quote !== null) {
      if (c === quote) quote = null;
      i += 1;
      continue;
    }
    if (c === "'" || c === '"') {
      quote = c;
    } else if (c === "$" && text[i + 1] === "{") {
      const end = findSpanEnd(text, i + 2);
      if (end === null) return null;
      i = end + 1;
      continue;
    } else if (c === "(") {
      depth += 1;
    } else if (c === ")") {
      depth -= 1;
      if (depth === 0) return i;
    }
    i += 1;
  }
  return null;
}

/**
 * Split `text` on the single character `sep` at top level.
 *
 * A `sep` inside quotes, a `${...}` span (brace depth) or parentheses (call arguments) doesn't
 * split, so a call's own arguments stay in one part.
 *
 * @param text The text to split.
 * @param sep The one-character separator.
 * @returns The parts in order, separators dropped; one part when nothing splits.
 */
export function splitCallsAware(text: string, sep: string): string[] {
  const out: string[] = [];
  let buf = "";
  let quote: string | null = null;
  let brace = 0;
  let paren = 0;
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (quote !== null) {
      buf += c;
      if (c === quote) quote = null;
      i += 1;
      continue;
    }
    if (c === "'" || c === '"') {
      quote = c;
      buf += c;
    } else if (c === "$" && text[i + 1] === "{") {
      brace += 1;
      buf += "${";
      i += 2;
      continue;
    } else if (c === "{" && brace > 0) {
      brace += 1;
      buf += c;
    } else if (c === "}" && brace > 0) {
      brace -= 1;
      buf += c;
    } else if (brace > 0) {
      buf += c;
    } else if (c === "(") {
      paren += 1;
      buf += c;
    } else if (c === ")") {
      // Clamped so a stray `)` doesn't disable the later top-level splits.
      paren = Math.max(0, paren - 1);
      buf += c;
    } else if (c === sep && paren === 0) {
      out.push(buf);
      buf = "";
    } else {
      buf += c;
    }
    i += 1;
  }
  out.push(buf);
  return out;
}

/**
 * Evaluate scanned template segments to a value.
 *
 * A template that is exactly one `${...}` span gives that span's typed value. Otherwise each
 * span's value is written in the text form and joined with the literal runs: an embedded `null`
 * is `""`, except in `STRICT_RAISE`, where a span whose final value is `null` raises, whole or
 * embedded. A template with no segments gives `""`.
 *
 * @param segments The scan from `scanTemplate`.
 * @param resolve Reads a reference's root from its names (see `evalExpr`).
 * @param options.item The MAP-body-local scope for `item` / `item.path`; `null` (the default)
 *   means no item scope. With `trace`, it may be a `ResolvedValue`.
 * @param options.mode How a missing reference is treated inside a span (default
 *   `BINDING_NONE`; see `evalExpr`).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in; `null` (the default) means no types.
 * @param options.trace Record the value's route, with each embedded span's half-open range of
 *   code points in the rendered text, and return both as a `TracedValue` (default `false`).
 * @returns The typed value of a whole single span; otherwise the rendered text. With `trace`, the
 *   same value with its route.
 * @throws {RequiredError} When a span's `head :? message` head misses or is `null`.
 * @throws {ExpressionError} On any error `evalExpr` raises for a span, a final `null` in
 *   `STRICT_RAISE` mode, or an embedded value the runtime can't write as text.
 */
export function evalTemplate(
  segments: readonly TemplateSegment[],
  resolve: RootResolver,
  options: TraceOptions & { trace: true },
): TracedValue;
export function evalTemplate(
  segments: readonly TemplateSegment[],
  resolve: RootResolver,
  options?: EvalOptions & { trace?: false },
): Value;
export function evalTemplate(
  segments: readonly TemplateSegment[],
  resolve: RootResolver,
  options?: TraceOptions,
): Value | TracedValue;
export function evalTemplate(
  segments: readonly TemplateSegment[],
  resolve: RootResolver,
  { item, mode = ResolveMode.BINDING_NONE, types = null, trace = false }: TraceOptions = {},
): Value | TracedValue {
  const evalSpan = (span: Span): Value | TracedValue => {
    const result = evalExpr(span.tree, resolve, { item, mode, types, trace });
    const value = trace ? (result as TracedValue).value : (result as Value);
    if (value === null && mode === ResolveMode.STRICT_RAISE) {
      throw new ExpressionError("null value in strict template");
    }
    return result;
  };

  if (segments.length === 1 && segments[0] instanceof Span) return evalSpan(segments[0]);
  let text = "";
  const routes: ValueRoute[] = [];
  const spans: [number, number][] = [];
  let expression = "";
  let offset = 0;
  for (const segment of segments) {
    if (segment instanceof Literal) {
      text += segment.text;
      if (trace) {
        expression += segment.text;
        offset += codePoints(segment.text);
      }
    } else {
      const evaluated = evalSpan(segment);
      const value = trace ? (evaluated as TracedValue).value : (evaluated as Value);
      const part = value === null ? "" : asText(value);
      text += part;
      if (trace) {
        const route = (evaluated as TracedValue).route;
        const end = offset + codePoints(part);
        routes.push(route);
        spans.push([offset, end]);
        expression += "${" + route.expression + "}";
        offset = end;
      }
    }
  }
  if (!trace) return text;
  if (routes.length === 0) return { value: text, route: ValueRoute.literal(text) };
  // `JSON.stringify`, as Python's `json.dumps`: literal text holding a lone surrogate then fails at
  // the slot's write check, traced or not, rather than here.
  const valueJson = JSON.stringify(text);
  return { value: text, route: new ValueRoute("interpolation", { valueJson, children: routes, expression, spans }) };
}

/** Count the code points of `text`: interpolation spans are in code points, not UTF-16 units. */
function codePoints(text: string): number {
  let n = 0;
  for (const _ of text) n++;
  return n;
}

/** Write `value` in the text form, turning the writer's runtime-limit error into an `ExpressionError`. */
function asText(value: Value): string {
  try {
    return writeAsText(value);
  } catch (error) {
    throw new ExpressionError(error instanceof Error ? error.message : String(error), { cause: error });
  }
}

/**
 * Evaluate a binding-value template to a value.
 *
 * Scans `source` and evaluates it in `BINDING_NONE` mode: a missing reference is `null`, so a
 * coalesce or default may fire. A whole single span gives its typed value; otherwise spans are
 * written as text into the literal runs. The full expression grammar is available inside a span.
 *
 * @param source The binding-value template. (A value that isn't text is a literal and never
 *   reaches here.)
 * @param resolve Reads a reference's root from its names (see `evalExpr`).
 * @param options.item The MAP-body-local scope for `item` / `item.path`; `null` (the default)
 *   means no item scope (a `null` element and no scope coincide). With `trace`, it may be a
 *   `ResolvedValue`.
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in; `null` (the default) means no types.
 * @param options.trace Record the value's route and return both as a `TracedValue` (default
 *   `false`; see `evalTemplate`).
 * @returns The typed value of a whole single span; otherwise the rendered text. With `trace`, the
 *   same value with its route.
 * @throws {RequiredError} When a `${ref :? message}` head is unbound.
 * @throws {ExpressionError} On a malformed span (unbalanced, nested, or not parsing) or any
 *   error a span's evaluation raises.
 */
export function evalBinding(
  source: string,
  resolve: RootResolver,
  options: Omit<TraceOptions, "mode"> & { trace: true },
): TracedValue;
export function evalBinding(
  source: string,
  resolve: RootResolver,
  options?: Omit<EvalOptions, "mode"> & { trace?: false },
): Value;
export function evalBinding(
  source: string,
  resolve: RootResolver,
  options?: Omit<TraceOptions, "mode">,
): Value | TracedValue;
export function evalBinding(
  source: string,
  resolve: RootResolver,
  { item, types = null, trace = false }: Omit<TraceOptions, "mode"> = {},
): Value | TracedValue {
  return evalTemplate(scanTemplate(source), resolve, { item, mode: ResolveMode.BINDING_NONE, types, trace });
}

/**
 * Return every reference path a template reads: `exprRefs` over its spans, in source order.
 *
 * Literal runs contribute nothing. Not deduped, as `exprRefs`.
 *
 * @param segments The scan from `scanTemplate`.
 * @returns The reference paths across all spans, in source order.
 */
export function templateRefs(segments: readonly TemplateSegment[]): string[] {
  const refs: string[] = [];
  for (const segment of segments) {
    if (segment instanceof Span) refs.push(...exprRefs(segment.tree));
  }
  return refs;
}

/**
 * Return every reference path a binding-value template reads, in source order (not deduped).
 *
 * The raw-string reference walk compose runs at every call site (edge inference, item capture,
 * output typing). `item`-headed references are collected; the caller skips them when minting
 * edges.
 *
 * @param source The binding-value template.
 * @returns The reference paths across all spans, in source order.
 * @throws {ExpressionError} On a malformed span (unbalanced, nested, or not parsing).
 */
export function exprRefsOf(source: string): string[] {
  return templateRefs(scanTemplate(source));
}

/**
 * Rewrite the reference leaves across all `${...}` spans of a binding or template value.
 *
 * The rewrite analog of `exprRefsOf`: each span's interior goes through `rewriteExprRefs`, so a
 * whole span, an embedded one and a computed one (`${x :- y}`) are handled alike. Unlike
 * `scanTemplate`, which collapses `$$` to `$`, this produces source, so `$$` and every literal
 * character are kept verbatim.
 *
 * @param source The binding or template value.
 * @param rename `path -> { text, used }`, or `null` to keep, as `rewriteExprRefs` takes it.
 * @returns `source` with the renamed reference leaves spliced in. A value that isn't text, or text
 *   with no `${`, comes back as is; an unbalanced `${` keeps the rest of the text as it is (the
 *   binding parser reports it later, located).
 * @throws {ExpressionError} If a span's interior does not parse.
 */
export function rewriteTemplateRefs<T>(source: T, rename: RefRename): T | string {
  if (typeof source !== "string" || !source.includes("${")) return source;
  let out = "";
  let i = 0;
  while (i < source.length) {
    if (source[i] === "$" && source[i + 1] === "$") {
      out += "$$"; // a literal `$`, never a span start
      i += 2;
    } else if (source[i] === "$" && source[i + 1] === "{") {
      const end = findSpanEnd(source, i + 2);
      if (end === null) {
        out += source.slice(i);
        break;
      }
      out += "${" + rewriteExprRefs(source.slice(i + 2, end), rename) + "}";
      i = end + 1;
    } else {
      out += source[i];
      i += 1;
    }
  }
  return out;
}

/**
 * Return whether a binding co-skips when all the producers it references are skipped.
 *
 * A hard data dependency: the whole value is one `${...}` span whose expression is a pure group
 * of references and reference defaults (see `operandCoSkips`). A literal operand runs the node
 * with the literal, a literal `:-` default supplies one, a `:?` runs it to fail loudly, and
 * embedded text writes an absent reference as `""`, so none of those co-skips.
 *
 * @param source The binding value.
 * @returns `false` for a value that isn't text, embedded text, a plain literal, a malformed span
 *   or a computed expression (arithmetic, comparison, builtin call).
 */
export function bindingCoSkips(source: unknown): boolean {
  if (typeof source !== "string") return false;
  let segments: TemplateSegment[];
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return false;
    throw error;
  }
  if (segments.length !== 1 || !(segments[0] instanceof Span)) return false;
  return operandCoSkips(segments[0].tree);
}

/**
 * Return whether one parsed operand co-skips: a pure reference or reference-default group.
 *
 * True for a bare reference, a positional constructor `T(x)` whose `x` co-skips, a coalesce whose
 * every operand co-skips, and a default whose fallback co-skips (`:-b.output`). False for a
 * literal, a builtin call, a keyword constructor, a `:?` required and any computed node.
 */
function operandCoSkips(node: ExprNode): boolean {
  if (!(node instanceof Tree)) return false; // a literal token
  if (node.data === "refcall") {
    // A reference co-skips; a builtin call is a computed value.
    const callSuffix = callSuffixOf(node);
    if (callSuffix === undefined) return true;
    // `T(x)` is `!T ${x}`, so it co-skips exactly when `x` does.
    const args = callSuffix.children as Tree[];
    return (
      isConstructorCallee(String(node.children[0])) &&
      args.length === 1 &&
      args[0]!.children.length === 1 &&
      operandCoSkips(args[0]!.children[0]!)
    );
  }
  if (node.data === "coalesce") return node.children.every((child) => operandCoSkips(child));
  if (node.data === "default_expr") return operandCoSkips(node.children[1]!); // the fallback must be a reference too
  return false;
}

// The sub-expressions that let a condition run on an absent reference (`|`, `:-`, `:?`).
const ESCAPES: ReadonlySet<string> = new Set(["coalesce", "default_expr", "required_expr"]);

/** One reference occurrence in a parsed condition. */
export interface ConditionRead {
  /** The reference's flat dotted path. */
  readonly path: string;
  /** The read as written, `?.` included. */
  readonly text: string;
  /**
   * The source-order index of the outermost `:-` / `:?` / `|` sub-expression around it, or `null`
   * outside any. Every outermost escape takes an index, with or without references.
   */
  readonly escape: number | null;
  /** Whether that sub-expression co-skips as a binding (`bindingCoSkips`); `false` outside any. */
  readonly coSkips: boolean;
}

/**
 * Return each reference leaf of a parsed condition with its `ConditionRead`, in source order.
 *
 * The one walk behind `conditionReads` and `rewriteConditionReads`, so their occurrence indexes
 * agree. Keep in sync with the reference and builtin-argument decomposition of `collectExprRefs`
 * (and `rewriteConditionReads` with the splice of `collectRefEdits`), both in `expressions.ts`: a
 * builtin call's callee and an `isinstance` type name are not reads; its argument values are.
 */
function conditionReadLeaves(tree: ExprNode): [Tree, ConditionRead][] {
  const leaves: [Tree, ConditionRead][] = [];
  let escapes = 0;

  const walk = (node: ExprNode, escape: number | null, coSkips: boolean): void => {
    if (!(node instanceof Tree)) return; // a literal token
    if (escape === null && ESCAPES.has(node.data)) {
      escape = escapes;
      coSkips = operandCoSkips(node);
      escapes += 1;
    }
    if (node.data === "refcall") {
      const children = node.children;
      const callSuffix = callSuffixOf(node);
      if (callSuffix === undefined) {
        leaves.push([node, { path: referencePath(node), text: referenceText(node), escape, coSkips }]);
        return;
      }
      for (const arg of valueArgs(String(children[0]), callSuffix)) {
        walk(arg.children.at(-1)!, escape, coSkips); // a keyword argument's value is its last child
      }
      return;
    }
    for (const child of node.children) walk(child, escape, coSkips);
  };

  walk(tree, null, false);
  return leaves;
}

/**
 * Return each reference occurrence of a parsed condition, in source order (not deduped).
 *
 * Unlike `exprRefs`, each occurrence also says which outermost `:-` / `:?` / `|` sub-expression it
 * sits in and whether that sub-expression co-skips as a binding, so a gate can take the same data
 * edge a binding of that sub-expression would.
 *
 * Example: `conditionReads(parseCondition("a.output > 5 or (b.output | c.output) == 2"))` gives
 * `a.output` (escape `null`, no co-skip), then `b.output` and `c.output` (escape `0`, co-skip).
 *
 * @param tree A parsed condition (from `parseCondition`).
 * @returns One read per reference occurrence.
 */
export function conditionReads(tree: ExprNode): ConditionRead[] {
  return conditionReadLeaves(tree).map(([, read]) => read);
}

/**
 * Rewrite each reference leaf of a condition, in either spelling, by occurrence.
 *
 * The per-occurrence analog of `rewriteConditionRefs`: a whole `${...}` keeps its braces and
 * rewrites the interior; a bare value is trimmed and rewritten in place. Only reference leaves
 * move; everything else is spliced through verbatim.
 *
 * @param expression The condition source, in either spelling.
 * @param rename Gets the occurrence's index in `conditionReads` order and its read; returns the
 *   text for a prefix of the read and how many names it used, as `rewriteExprRefs` takes it, or
 *   `null` to keep it. The steps after that prefix stay as written.
 * @returns `expression` with the renamed reference leaves spliced in.
 * @throws {ExpressionError} If the condition does not parse.
 */
export function rewriteConditionReads(
  expression: string,
  rename: (i: number, read: ConditionRead) => RefRenamed | null,
): string {
  const interior = wholeSpanInterior(expression);
  let text = interior === null ? expression.trim() : interior;
  const edits: [number, number, string][] = [];
  conditionReadLeaves(parseExpr(text)).forEach(([node, read], i) => {
    const renamed = rename(i, read);
    if (renamed !== null) edits.push([(node.children[0] as Token).startPos, namesEnd(node, renamed.used), renamed.text]);
  });
  // Splice right to left so earlier offsets stay valid.
  edits.sort((a, b) => b[0] - a[0]);
  for (const [start, end, replacement] of edits) text = text.slice(0, start) + replacement + text.slice(end);
  return interior === null ? text : "${" + text + "}";
}

/**
 * Render a strict AGENT prompt against its node's bound input record.
 *
 * Each `${...}` span is an expression whose references resolve against `record` (the node's
 * declared inputs; the pool namespaces aren't in scope), and whose calls run the `TEMPLATE_FNS`
 * builtins. Rendered in `STRICT_RAISE` mode: unlike a binding (a miss is `null`) and a condition
 * (a miss is falsy), a missing reference or a span whose final value is `null` raises. A prompt
 * is text, so a whole single span's value is written as text too. `$$` renders one `$`.
 *
 * @param text The prompt template.
 * @param record The node's bound input record.
 * @returns The rendered prompt.
 * @throws {RequiredError} When a span's `head :? message` head misses or is `null`.
 * @throws {ExpressionError} On a malformed span, an unresolved reference, a final `null` span
 *   value, an unknown builtin or one that fails, or a value the runtime can't write as text.
 */
export function renderTemplateRecord(text: string, record: ValueRecord): string {
  const value = evalTemplate(scanTemplate(text), (names) => resolveInRecord(names, record), {
    mode: ResolveMode.STRICT_RAISE,
  });
  return asText(value);
}

/**
 * Return every declared-input reference path a prompt reads.
 *
 * The compile-time companion to `renderTemplateRecord`, which compose uses to check a prompt's
 * names. Also rejects, so a bad prompt fails at load rather than at render: a callee that is no
 * formatting builtin (one not in `TEMPLATE_FNS`, or the shipped `len`, which computes),
 * `isinstance` (a prompt has no type table), and a builtin called with arguments it doesn't take.
 *
 * @param text The prompt template.
 * @returns Each plain-span path and each builtin-call argument path, in source order (not
 *   deduped); literals contribute nothing.
 * @throws {ExpressionError} On a malformed span, a call to anything but a formatting builtin, or
 *   a builtin called with arguments it doesn't take.
 */
export function promptRefs(text: string): string[] {
  const segments = scanTemplate(text);
  for (const segment of segments) {
    if (segment instanceof Span) {
      rejectUnknownPromptBuiltin(segment.tree);
      checkCalls(segment.tree);
    }
  }
  return templateRefs(segments);
}

/** Return whether a call to `callee` can be a `${T(...)}` constructor (no builtin, no `isinstance`). */
function isConstructorCallee(callee: string): boolean {
  return !TEMPLATE_FNS.has(callee) && callee !== ISINSTANCE;
}

/**
 * Return the callees in a parsed span that are neither a `TEMPLATE_FNS` builtin nor `isinstance`
 * (a flow or a constructor), in `callCallees` order (not deduped).
 */
function callCalleesAbsentFromBuiltins(tree: ExprNode): string[] {
  return callCallees(tree).filter(isConstructorCallee);
}

/** Return every callee a call inside a parsed span names, in Lark's `iter_subtrees` order. */
function callCallees(tree: ExprNode): string[] {
  if (!(tree instanceof Tree)) return [];
  return iterSubtrees(tree)
    .filter((node) => node.data === "refcall" && callSuffixOf(node) !== undefined)
    .map((node) => String(node.children[0]));
}

/**
 * Throw if a call in a scanned prompt span names anything but a formatting builtin.
 *
 * The compile-time mirror of the render-time unknown-function error, so compose rejects it
 * without evaluating. `isinstance`, `T(...)` constructors and the shipped `len`
 * (`isNonPromptBuiltin`) are rejected too: prompt builtins only format (a value is computed in an
 * input binding), and a prompt has no type table.
 */
function rejectUnknownPromptBuiltin(tree: ExprNode): void {
  if (callCallees(tree).includes(ISINSTANCE)) {
    throw new ExpressionError("isinstance is not a prompt function; compute it in an input binding");
  }
  const bad = callCallees(tree).filter((callee) => isConstructorCallee(callee) || isNonPromptBuiltin(callee));
  if (bad.length > 0) {
    const builtins = [...TEMPLATE_FNS.keys()]
      .filter((name) => !isNonPromptBuiltin(name))
      .sort(compareCodeUnits)
      .join(", ");
    const hint =
      conversionHint(bad[0]!) ||
      `; prompts call only formatting builtins (${builtins}); ` +
        "build or check a typed value with a `typedefs:` constructor `T(...)` in an input binding";
    throw new ExpressionError(`unknown prompt function ${reprText(bad[0]!)}${hint}`);
  }
}

/**
 * Return every flow callee named by a call inside a `${...}` span of `text`.
 *
 * Compose uses this to reject the retired inline `${ flow(args) }` form: a flow call belongs in
 * the whole-value `call(...)` directive. Builtins, `isinstance(x, T)` and a `T(...)` constructor
 * whose name is in `typeNames` stay legal inside a span and are not listed.
 *
 * @param text A binding or template value.
 * @param typeNames The flow's `typedefs:` names (default: none).
 * @returns The flow callees, in span order (not deduped); `[]` when there are none.
 * @throws {ExpressionError} If a span is malformed or doesn't parse.
 */
export function flowCallCalleesInSpans(text: string, typeNames: ReadonlySet<string> = new Set()): string[] {
  const callees: string[] = [];
  for (const segment of scanTemplate(text)) {
    if (segment instanceof Span) {
      callees.push(...callCalleesAbsentFromBuiltins(segment.tree).filter((callee) => !typeNames.has(callee)));
    }
  }
  return callees;
}

/**
 * Require every builtin call in `text`'s `${...}` spans to pass arguments it takes.
 *
 * Compose runs this on every binding and flow output; a prompt gets the same check through
 * `promptRefs`. Any other callee except `isinstance` is a `T(...)` constructor: with `types` it
 * must name a type in it; without, its name is left to the caller.
 *
 * @param text A binding or template value.
 * @param types The type table constructor names must be in (only its names are read), or `null`
 *   (the default) to leave them to the caller.
 * @throws {ExpressionError} If a span is malformed or doesn't parse, calls a builtin with
 *   arguments it doesn't take, or, with `types`, calls a function that is neither a builtin nor a
 *   type in `types`.
 */
export function checkTemplateCalls(text: string, types: ReadonlyMap<string, unknown> | null = null): void {
  for (const segment of scanTemplate(text)) {
    if (segment instanceof Span) checkCalls(segment.tree, types);
  }
}
