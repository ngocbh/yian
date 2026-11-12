/**
 * Remove API keys and tokens from provider error text before anything records it.
 *
 * Providers can echo a rejected key in their error text, in full, encoded, or masked with a
 * visible prefix and suffix (OpenAI's 401 says `Incorrect API key provided: sk-abc****wxyz`).
 * `redactKey` and `redactKeys` replace each key and its masked echoes with `REDACTED`. Walking
 * an error's chain to rewrite its messages is `scrubError`'s job, in `llm/errors.ts`.
 */

import { compareCodeUnits } from "../diagnostics.ts";
import { foldCase, REDACTED, scrub } from "../llm/auth.ts";

export { REDACTED } from "../llm/auth.ts";

// A masked echo counts only when it shows at least this many key characters on one side.
const MIN_VISIBLE = 3;
const WORD = "A-Za-z0-9_\\-";
const MARK = /\*{2,}|\.{3}|…/gu;

/**
 * Match masked echoes: visible head, mask marks, visible tail.
 *
 * Visible characters are word characters plus `extra`, the key's other characters. The
 * lookbehind starts a head only at the start of a run, which keeps matching linear.
 */
function maskedPattern(extra: string): RegExp {
  // In `u` mode escaping anything but a syntax character (or `-` in a class) is a SyntaxError;
  // these are the ones a class needs.
  const visible = `[${WORD}${extra.replace(/[\\\]\[^-]/g, "\\$&")}]`;
  // Case-blind, as the echo may raise or lower the key's letters.
  return new RegExp(`(?<!${visible})${visible}*(?:${MARK.source})${visible}*`, "giu");
}

/** Return the UTF-16 index of each code-point boundary in `text`, from 0 to `text.length`. */
function boundaries(text: string): number[] {
  const indexes = [0];
  for (const char of text) indexes.push(indexes.at(-1)! + char.length);
  return indexes;
}

/**
 * Return where the longest suffix of `head` that begins `folded`, a key as `foldCase` folds it,
 * starts in any letter case, if it has at least `MIN_VISIBLE` code points; `keySize` is the key's
 * length in code points.
 */
function prefixStart(head: string, folded: string, keySize: number): number | null {
  const bounds = boundaries(head);
  const count = bounds.length - 1;
  for (let size = Math.min(count, keySize); size >= MIN_VISIBLE; size--) {
    const start = bounds[count - size]!;
    if (folded.startsWith(foldCase(head.slice(start)))) return start;
  }
  return null;
}

/**
 * Return where the longest prefix of `tail` that ends `folded`, a key as `foldCase` folds it,
 * stops in any letter case, if it has at least `MIN_VISIBLE` code points; `keySize` is the key's
 * length in code points.
 */
function suffixStop(tail: string, folded: string, keySize: number): number | null {
  const bounds = boundaries(tail);
  for (let size = Math.min(bounds.length - 1, keySize); size >= MIN_VISIBLE; size--) {
    const stop = bounds[size]!;
    if (folded.endsWith(foldCase(tail.slice(0, stop)))) return stop;
  }
  return null;
}

/**
 * Replace `key`, its encoded forms and masked echoes of it in `text` with `REDACTED`.
 *
 * The encoded forms are `llm/auth.ts`'s `secretForms`. A masked echo is visible text around
 * `**`, `...` or `…`. It counts when at least three visible characters before the mask begin
 * the key or at least three after it end the key, in any letter case, as the key itself is
 * matched; the matching text and the mask are replaced,
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
  const keySize = [...key].length;
  const folded = foldCase(key);

  return text.replace(maskedPattern(extra), (token) => {
    // `matchAll` scans a copy of `MARK`, so the shared regex's `lastIndex` stays 0.
    const marks = [...token.matchAll(MARK)];
    const last = marks.at(-1)!;
    const head = token.slice(0, marks[0]!.index);
    const tail = token.slice(last.index + last[0].length);
    const start = prefixStart(head, folded, keySize);
    const stop = suffixStop(tail, folded, keySize);
    if (start === null && stop === null) return token;
    // When only the tail matches, the whole head goes too, which over-redacts (a short key
    // prefix included). When only the head matches, the tail is kept, so up to two key
    // characters there may survive, under the three-character minimum.
    const left = start === null ? "" : head.slice(0, start);
    const right = stop === null ? tail : tail.slice(stop);
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
