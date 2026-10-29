/**
 * YAML 1.1 spellings that the flow loader's YAML 1.2 core rules read as plain text, and names
 * for what YAML read a non-text value as.
 */

import { buildValueAs, isRecord, type Type, TypeCheckError } from "../../typesys/index.ts";

const BOOLEAN_WORDS: ReadonlyMap<string, boolean> = new Map([
  ["yes", true],
  ["y", true],
  ["on", true],
  ["no", false],
  ["n", false],
  ["off", false],
]);

/**
 * Return the boolean a YAML 1.1 boolean spelling meant and the fix text, else `null`.
 *
 * Matching ignores case and surrounding whitespace; the fix shows the stripped text as
 * written. Only a bare text value gets a fix; a list, record or other value returns `null`.
 * With `declared`, the fix is offered only if its boolean fits that type.
 *
 * Example: `formerBooleanFix("yes")` is ``[true, "`yes` is text; write `true`"]``;
 * `formerBooleanFix("true")` is `null`.
 */
export function formerBooleanFix(
  value: unknown,
  declared: Type | null = null,
): [boolean, string] | null {
  if (typeof value !== "string") return null;
  const fix = wordFix(value);
  if (fix === null || declared === null || valueFits(declared, fix[0])) return fix;
  return null;
}

/**
 * Name what YAML read a value as, for an error message about a value that must be text.
 *
 * Returns a phrase such as `a boolean`, `an integer` or `a list`. The loader keeps dates as
 * text and builds no bytes, so no phrase names a date, a datetime or bytes.
 *
 * Example: `yamlReadAs(5n)` is `"an integer"`; `yamlReadAs(null)` is `"null"`.
 */
export function yamlReadAs(value: unknown): string {
  if (typeof value === "boolean") return "a boolean";
  if (typeof value === "bigint") return "an integer";
  if (typeof value === "number") return "a float";
  if (value === null) return "null";
  if (Array.isArray(value)) return "a list";
  if (isRecord(value)) return "a mapping";
  if (typeof value === "object") return `a \`${value.constructor?.name ?? "object"}\``;
  return `a \`${typeof value}\``;
}

/** Return the boolean and fix text for one YAML 1.1 boolean word, else `null`. */
function wordFix(value: string): [boolean, string] | null {
  const text = value.trim();
  const flag = BOOLEAN_WORDS.get(text.toLowerCase());
  if (flag === undefined) return null;
  return [flag, `\`${text}\` is text; write \`${flag ? "true" : "false"}\``];
}

/** Return whether `value` passes the write-boundary check for `typ`. */
export function valueFits(typ: Type, value: unknown): boolean {
  try {
    buildValueAs(typ, value);
  } catch (error) {
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
  return true;
}
