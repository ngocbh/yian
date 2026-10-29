/**
 * Recognize a whole-value `call(...)` directive in binding text.
 *
 * Also holds what the directive's desugar shares: `InlineCall`, the record a desugared call
 * produces, and `findParenEnd` and `splitCallsAware`, the paren-, quote- and span-aware splitters
 * the recognizer uses.
 */

import { reprText } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { callSuffixOf, findSpanEnd, iterSubtreesTopdown, type Token, Tree } from "../../expr/grammar.ts";
import { scanTemplate, Span } from "../../expr/template.ts";
import type { ValueRecord } from "../../typesys/index.ts";

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

/** An unquoted value whose trimmed text starts with this is a `call(...)` directive. */
export const CALL_HEAD = /^call\s*\(/;
const FLOW_ID = /^[A-Za-z_][A-Za-z0-9_-]*$/;

/**
 * Recognize a whole-value `call(...)` directive.
 *
 * A value whose trimmed text starts with `call(` must be exactly one directive: its parentheses
 * balance, nothing follows the closing `)`, and the first argument is a bare flow id (an
 * identifier, hyphens allowed). A positional argument after the flow id is kept in the remaining
 * text, so the keyword-argument parser rejects it with its own message.
 *
 * Example: `matchCallDirective("call(f, x=1)")` is `["f", " x=1"]`;
 * `matchCallDirective("pre call(x)")` is `null`.
 *
 * @param value The binding text.
 * @returns The flow id and the raw argument text after the flow id's comma (empty for
 *   `call(f)`), or `null` when the trimmed text doesn't start with `call(`.
 * @throws {ExpressionError} When the value starts with `call(` but its parentheses are
 *   unbalanced, text follows the closing `)`, or the first argument is not a bare flow id.
 */
export function matchCallDirective(value: string): [flowId: string, remaining: string] | null {
  const s = value.trim();
  const m = CALL_HEAD.exec(s);
  if (m === null) return null;
  const openIdx = m[0].length - 1;
  const closeIdx = findParenEnd(s, openIdx + 1);
  if (closeIdx === null) {
    throw new ExpressionError("`call(` is never closed (check its parentheses and quotes)");
  }
  const trailing = s.slice(closeIdx + 1).trim();
  if (trailing) {
    throw new ExpressionError(
      `text follows the closing \`)\`: ${reprText(trailing)}; a call(...) must be the whole value`,
    );
  }
  const argsStr = s.slice(openIdx + 1, closeIdx);
  const first = splitCallsAware(argsStr, ",")[0]!.trim();
  if (!FLOW_ID.test(first)) {
    throw new ExpressionError(
      `a call(...) must start with a flow id (a bare name) and a comma, got ${reprText(first)}`,
    );
  }
  // Everything after the first comma is the remaining (keyword) args; the parser over this text
  // never re-sees the positional flow id, so a positional arg among them (`call(f, 30)`) fails
  // loudly there.
  const comma = argsStr.indexOf(",");
  return [first, comma !== -1 ? argsStr.slice(comma + 1) : ""];
}

/**
 * Return the flow id to suggest for a `call(...)` written inside a `${...}` span of `text`.
 *
 * @param text A binding or branch value whose spans parse and call `call`.
 * @returns The first such call's flow id, read as `matchCallDirective` reads it, or `<flow>` when
 *   that call doesn't start with a flow id.
 */
export function spanCallFlowId(text: string): string {
  const call = spanCallDirectiveText(text);
  if (call === null) return "<flow>";
  try {
    return matchCallDirective(call)?.[0] ?? "<flow>";
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    return "<flow>";
  }
}

/**
 * Return the source text of the first `call(...)` inside a `${...}` span of `text`.
 *
 * Compose uses this to name the real flow when it rejects the `call(...)` directive written inside
 * a span, where it parses as a call to a function named `call`. Calls are visited outermost
 * first, so `${call(outer, x=call(inner))}` gives the outer call.
 *
 * @param text A binding or template value whose spans parse.
 * @returns The text from `call` through its closing `)`, or `null` when no span calls `call`.
 * @throws {ExpressionError} If a span is malformed or doesn't parse.
 */
export function spanCallDirectiveText(text: string): string | null {
  for (const segment of scanTemplate(text)) {
    if (!(segment instanceof Span) || !(segment.tree instanceof Tree)) continue;
    for (const node of iterSubtreesTopdown(segment.tree)) {
      if (node.data !== "refcall" || callSuffixOf(node) === undefined) continue;
      const name = node.children[0] as Token;
      if (name.value !== "call") continue;
      const open = segment.source.indexOf("(", name.endPos);
      const close = findParenEnd(segment.source, open + 1);
      if (close !== null) return segment.source.slice(name.startPos, close + 1);
    }
  }
  return null;
}
