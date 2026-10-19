/**
 * Remove API keys and tokens from provider error text before anything records it.
 *
 * Providers can echo a rejected key in their error text, in full, encoded, or masked with a
 * visible prefix and suffix (OpenAI's 401 says `Incorrect API key provided: sk-abc****wxyz`).
 * `redactKey` and `redactKeys` replace each key and its masked echoes with `REDACTED`. Walking
 * an error's chain to rewrite its messages is `scrubError`'s job, in `llm/errors.ts`.
 */

import { compareCodeUnits } from "../diagnostics.ts";
import { REDACTED, scrub } from "../llm/auth.ts";

export { REDACTED } from "../llm/auth.ts";

// A masked echo counts only when it shows at least this many key characters on one side.
const MIN_VISIBLE = 3;
const WORD = "A-Za-z0-9_\\-";
const MARK = "\\*{2,}|\\.{3}|…";

/**
 * Match masked echoes: visible head, mask marks, visible tail.
 *
 * Visible characters are word characters plus `extra`, the key's other characters. The
 * lookbehind starts a head only at the start of a run, which keeps matching linear.
 */
function maskedPattern(extra: string): RegExp {
  // In `u` mode a class may escape only syntax characters; these are the ones a class needs.
  const visible = `[${WORD}${extra.replace(/[\\\]\[^-]/g, "\\$&")}]`;
  return new RegExp(`(?<!${visible})${visible}*(?:${MARK})${visible}*`, "gu");
}

/** Return where the longest suffix of `head` that begins `key` starts, if long enough, in code points. */
function prefixStart(head: string[], key: string): number | null {
  for (let size = Math.min(head.length, [...key].length); size >= MIN_VISIBLE; size--) {
    if (key.startsWith(head.slice(-size).join(""))) return head.length - size;
  }
  return null;
}

/** Return where the longest prefix of `tail` that ends `key` stops, if long enough, in code points. */
function suffixStop(tail: string[], key: string): number | null {
  for (let size = Math.min(tail.length, [...key].length); size >= MIN_VISIBLE; size--) {
    if (key.endsWith(tail.slice(0, size).join(""))) return size;
  }
  return null;
}

/**
 * Replace `key`, its encoded forms and masked echoes of it in `text` with `REDACTED`.
 *
 * The encoded forms are `llm/auth.ts`'s `secretForms`. A masked echo is visible text around
 * `**`, `...` or `…`. It counts when at least three visible characters before the mask begin
 * the key or at least three after it end the key; the matching text and the mask are replaced,
 * and other masked text is left alone. Visible characters are letters, digits, `_`, `-`, and
 * any other character in the key, except a mask character; counts are in code points. A key
 * that contains `..` or `**` can leave fragments.
 *
 * @param text Text that may contain the key.
 * @param key The key; `null`, `undefined` or empty returns `text`.
 * @returns `text` without the key or its masked fragments.
 */
export function redactKey(text: string, key: string | null | undefined): string {
  if (!key || !text) return text;
  text = scrub(text, [key]);
  const others = new Set<string>();
  for (const char of key) {
    if (!/^[A-Za-z0-9_\-*…]$/u.test(char)) others.add(char);
  }
  const extra = [...others].sort(compareCodeUnits).join("");

  return text.replace(maskedPattern(extra), (token) => {
    const marks = [...token.matchAll(new RegExp(MARK, "gu"))];
    const last = marks.at(-1)!;
    const head = Array.from(token.slice(0, marks[0]!.index));
    const tail = Array.from(token.slice(last.index + last[0].length));
    const start = prefixStart(head, key);
    const stop = suffixStop(tail, key);
    if (start === null && stop === null) return token;
    // Text on a side that does not match the key is kept only when the other side matched.
    const left = start === null ? "" : head.slice(0, start).join("");
    const right = stop === null ? tail.join("") : tail.slice(stop).join("");
    return left + REDACTED + right;
  });
}

/**
 * Replace every key in `keys`, its encoded forms and masked echoes of it in `text`, longest
 * first, so a key that contains another is removed whole.
 *
 * @param text Text that may contain the keys.
 * @param keys The keys; `null`, `undefined` and empty entries are ignored.
 * @returns `text` without the keys or their masked fragments.
 */
export function redactKeys(text: string, keys: Iterable<string | null | undefined>): string {
  const plains = new Set<string>();
  for (const key of keys) if (key) plains.add(key);
  for (const key of [...plains].sort((a, b) => b.length - a.length)) text = redactKey(text, key);
  return text;
}
