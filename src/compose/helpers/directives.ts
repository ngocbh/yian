/** Recognize a whole-value `call(...)` directive in binding text. */

import { reprText } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { findParenEnd, spanCallDirectiveText, splitCallsAware } from "../../expr/template.ts";

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
