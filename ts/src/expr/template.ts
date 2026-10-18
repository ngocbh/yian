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
 * This module also owns the strict AGENT/HUMAN_INPUT prompt renderer (`renderTemplateRecord`)
 * and the compile-time walks over template spans. A prompt is not a binding (it reads inputs
 * already bound to its node and mints no edge), but it is a `${...}` template and reuses this
 * scanner, so it lives here rather than in `expressions.ts`.
 *
 * Value tracing lands here later, with `lineage.ts`; the call-parsing helpers land with compose's
 * `call(...)` directive.
 *
 * Knows about: `expressions` (the evaluator, `checkCalls`, `resolveInRecord`, `ExpressionError`),
 * `grammar` (`parseExpr`, `findSpanEnd`, `rejectNestedSpans`), `builtins` (the `TEMPLATE_FNS`
 * registry), `typesys` (`writeAsText`) and `diagnostics` (`reprText`).
 * Never imports: `nodes`, `compile`, `compose`, `runtime`, or the pool (pool-agnostic via
 * `resolve`).
 */

import { reprText } from "../diagnostics.ts";
import type { Type, Value, ValueRecord } from "../typesys/index.ts";
import { writeAsText } from "../typesys/values.ts";
import { evalExpr, ExpressionError, exprRefs, ResolveMode, resolveInRecord } from "./expressions.ts";
import { type ExprNode, findSpanEnd, parseExpr, rejectNestedSpans } from "./grammar.ts";

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
 * Evaluate scanned template segments to a value.
 *
 * A template that is exactly one `${...}` span gives that span's typed value. Otherwise each
 * span's value is written in the text form and joined with the literal runs: an embedded `null`
 * is `""`, except in `STRICT_RAISE`, where a span whose final value is `null` raises, whole or
 * embedded. A template with no segments gives `""`.
 *
 * @param segments The scan from `scanTemplate`.
 * @param resolve Resolves a reference path to its value; `null` is a miss.
 * @param options.item The MAP-body-local scope for `item` / `item.path`; `null` (the default)
 *   means no item scope.
 * @param options.mode How a missing reference is treated inside a span (default
 *   `BINDING_NONE`; see `evalExpr`).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in; `null` (the default) means no types.
 * @returns The typed value of a whole single span; otherwise the rendered text.
 * @throws {RequiredError} When a span's `head :? message` head misses or is `null`.
 * @throws {ExpressionError} On any error `evalExpr` raises for a span, a final `null` in
 *   `STRICT_RAISE` mode, or an embedded value the runtime can't write as text.
 */
export function evalTemplate(
  segments: readonly TemplateSegment[],
  resolve: (path: string) => Value,
  {
    item = null,
    mode = ResolveMode.BINDING_NONE,
    types = null,
  }: { item?: Value; mode?: ResolveMode; types?: ReadonlyMap<string, Type> | null } = {},
): Value {
  const evalSpan = (span: Span): Value => {
    const value = evalExpr(span.tree, resolve, { item, mode, types });
    if (value === null && mode === ResolveMode.STRICT_RAISE) {
      throw new ExpressionError("null value in strict template");
    }
    return value;
  };

  if (segments.length === 1 && segments[0] instanceof Span) return evalSpan(segments[0]);
  let text = "";
  for (const segment of segments) {
    if (segment instanceof Literal) {
      text += segment.text;
    } else {
      const value = evalSpan(segment);
      text += value === null ? "" : asText(value);
    }
  }
  return text;
}

/** Write `value` in the text form, turning the writer's runtime-limit error into an `ExpressionError`. */
function asText(value: Value): string {
  try {
    return writeAsText(value);
  } catch (error) {
    throw new ExpressionError((error as Error).message, { cause: error });
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
 * @param resolve Resolves a reference path to its value; `null` is a miss.
 * @param options.item The MAP-body-local scope for `item` / `item.path`; `null` (the default)
 *   means no item scope (a `null` element and no scope coincide).
 * @param options.types The type table `isinstance(x, T)` and a `T(...)` constructor look `T` up
 *   in; `null` (the default) means no types.
 * @returns The typed value of a whole single span; otherwise the rendered text.
 * @throws {RequiredError} When a `${ref :? message}` head is unbound.
 * @throws {ExpressionError} On a malformed span (unbalanced, nested, or not parsing) or any
 *   error a span's evaluation raises.
 */
export function evalBinding(
  source: string,
  resolve: (path: string) => Value,
  { item = null, types = null }: { item?: Value; types?: ReadonlyMap<string, Type> | null } = {},
): Value {
  return evalTemplate(scanTemplate(source), resolve, { item, mode: ResolveMode.BINDING_NONE, types });
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
 * Render a strict AGENT / HUMAN_INPUT prompt against its node's bound input record.
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
  const value = evalTemplate(scanTemplate(text), (path) => resolveInRecord(path, record), {
    mode: ResolveMode.STRICT_RAISE,
  });
  return asText(value);
}
