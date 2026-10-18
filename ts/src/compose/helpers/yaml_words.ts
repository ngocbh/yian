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
const MISSING = Symbol("missing");

type Path = readonly (string | number)[];

/**
 * Return the boolean a YAML 1.1 boolean spelling meant and the fix text, else `null`.
 *
 * Matching ignores case and surrounding whitespace; the fix shows the stripped text as
 * written. With a declared type, a list or record is searched too: the word is the value the
 * type check rejects, and the fix is offered when its boolean clears that rejection. A bare
 * word's fix is offered only if its boolean fits `declared`.
 *
 * Example: `formerBooleanFix("yes")` is ``[true, "`yes` is text; write `true`"]``;
 * `formerBooleanFix("true")` is `null`.
 */
export function formerBooleanFix(
  value: unknown,
  declared: Type | null = null,
): [boolean, string] | null {
  if (typeof value === "string") {
    const fix = wordFix(value);
    if (fix === null || declared === null || rejection(declared, fix[0]) === null) return fix;
    return null;
  }
  if (declared === null || !(Array.isArray(value) || isRecord(value))) return null;
  const rejected = rejection(declared, value);
  if (rejected === null) return null;
  const fix = rejected.path.length > 0 ? wordFix(at(value, rejected.path)) : null;
  if (fix === null) return null;
  const again = rejection(declared, replaced(value, rejected.path, fix[0]));
  if (again !== null && samePath(again.path, rejected.path)) return null;
  return fix;
}

/**
 * Name what YAML read a value as, for an error message about a value that must be text.
 *
 * Returns a phrase such as `a boolean`, `an integer` or `a list`. The loader keeps dates as
 * text and builds no bytes, so Python's date, datetime and binary phrases have no case here.
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
function wordFix(value: unknown): [boolean, string] | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  const flag = BOOLEAN_WORDS.get(text.toLowerCase());
  if (flag === undefined) return null;
  return [flag, `\`${text}\` is text; write \`${flag ? "true" : "false"}\``];
}

/** Return the write-boundary check's error for `value`, or `null` when it fits. */
function rejection(declared: Type, value: unknown): TypeCheckError | null {
  try {
    buildValueAs(declared, value);
  } catch (error) {
    if (error instanceof TypeCheckError) return error;
    throw error;
  }
  return null;
}

/** Return the item a type-check path names inside `value`, or `MISSING`. */
function at(value: unknown, path: Path): unknown {
  for (const segment of path) {
    if (typeof segment === "string" && isRecord(value) && Object.hasOwn(value, segment)) {
      value = value[segment];
    } else if (
      typeof segment === "number" &&
      Array.isArray(value) &&
      0 <= segment &&
      segment < value.length
    ) {
      value = value[segment];
    } else {
      return MISSING;
    }
  }
  return value;
}

/** Return a copy of `value` with the item at `path` replaced by `leaf`. */
function replaced(value: unknown, path: Path, leaf: unknown): unknown {
  if (path.length === 0) return leaf;
  const [head, ...rest] = path as [string | number, ...Path];
  if (isRecord(value)) {
    // Spread and a computed key define own data properties, so a `__proto__` key stays data.
    return { ...value, [head]: replaced(value[head as string], rest, leaf) };
  }
  const copy = [...(value as unknown[])];
  copy[head as number] = replaced(copy[head as number], rest, leaf);
  return copy;
}

/** Whether two type-check paths name the same item. */
function samePath(left: Path, right: Path): boolean {
  return left.length === right.length && left.every((segment, index) => segment === right[index]);
}
