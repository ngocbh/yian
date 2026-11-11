/**
 * Spell a node's `output:` declaration from the plain type a test names, for tests that build a
 * flow's text from a type: `int` gives `!int`, `Literal(a, b)` gives `!Literal(a;b)`, and a record
 * such as `{qty: int}` gives `{qty: !int}`.
 */

import { parse } from "yaml";

import { tagSpelling } from "../../src/typesys/types.ts";

/**
 * Return the `output:` declaration for `type`: plain type text, or a record of them in YAML.
 *
 * @throws TypeError when `type` reads as neither text nor a record.
 */
export function outputTag(type: string): string {
  return spell(parse(type) as unknown);
}

function spell(value: unknown): string {
  if (typeof value === "string") return `!${tagSpelling(value)}`;
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const fields = Object.entries(value).map(([key, field]) => `${key}: ${spell(field)}`);
    return `{${fields.join(", ")}}`;
  }
  throw new TypeError(`outputTag: ${JSON.stringify(value)} is not a type`);
}
