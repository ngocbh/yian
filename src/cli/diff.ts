/**
 * A line diff of two texts in the unified format, as the chat's approval card shows a write.
 *
 * Knows about: nothing else.
 */

/** Unchanged lines shown around each change. */
const CONTEXT = 3;

/**
 * The most line pairs the longest-common-subsequence table holds; past it, the changed middle of
 * two texts is shown as all its old lines removed and all its new lines added.
 */
const MAX_TABLE = 4_000_000;

/** The note above a change too large to compare line by line. */
const UNMATCHED = "(lines not matched: too large to compare)";

/** One line of a diff: kept (` `), removed (`-`) or added (`+`), with its line ending. */
interface Edit {
  op: " " | "-" | "+";
  line: string;
  /** Set on the first edit of a change too large to compare line by line. */
  unmatched?: true;
}

/** `text`'s lines, each with its `\n` (the last may have none); none for an empty text. */
function splitLines(text: string): string[] {
  return text === "" ? [] : text.split(/(?<=\n)/);
}

/** The edits turning `a` into `b`, along a longest common subsequence of their lines. */
function lineEdits(a: readonly string[], b: readonly string[]): Edit[] {
  // The common head and tail are kept outright, so the table covers only the changed middle.
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  const fits = (): boolean => tail < a.length - head && tail < b.length - head;
  while (fits() && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const x = a.slice(head, a.length - tail);
  const y = b.slice(head, b.length - tail);
  const edits: Edit[] = a.slice(0, head).map((line) => ({ op: " ", line }));
  const n = x.length;
  const m = y.length;
  if (n * m > MAX_TABLE) {
    const first = edits.length;
    for (const line of x) edits.push({ op: "-", line });
    for (const line of y) edits.push({ op: "+", line });
    edits[first]!.unmatched = true;
  } else {
    // `table[i * (m + 1) + j]`: the longest common subsequence of `x[i..]` and `y[j..]`.
    const table = new Uint32Array((n + 1) * (m + 1));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        table[i * (m + 1) + j] =
          x[i] === y[j]
            ? table[(i + 1) * (m + 1) + j + 1]! + 1
            : Math.max(table[(i + 1) * (m + 1) + j]!, table[i * (m + 1) + j + 1]!);
      }
    }
    let i = 0;
    let j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && x[i] === y[j]) {
        edits.push({ op: " ", line: x[i]! });
        i++;
        j++;
      } else if (j === m || (i < n && table[(i + 1) * (m + 1) + j]! >= table[i * (m + 1) + j + 1]!)) {
        edits.push({ op: "-", line: x[i++]! });
      } else {
        edits.push({ op: "+", line: y[j++]! });
      }
    }
  }
  for (const line of a.slice(a.length - tail)) edits.push({ op: " ", line });
  return edits;
}

/** A hunk header's range: `start` alone for one line, else `start,count`. */
function range(start: number, count: number): string {
  return count === 1 ? `${start}` : `${start},${count}`;
}

/** One edit as a diff line: `\r\n` marked `[CRLF]`, and a missing final newline noted. */
function shown(edit: Edit): string {
  if (edit.line.endsWith("\r\n")) return `${edit.op}${edit.line.slice(0, -2)}[CRLF]\n`;
  if (edit.line.endsWith("\n")) return `${edit.op}${edit.line}`;
  return `${edit.op}${edit.line}\n\\ No newline at end of file\n`;
}

/**
 * Return the unified diff turning `oldText` into `newText`: `--- oldLabel` and `+++ newLabel`
 * headers, then `@@` hunks with three lines of context, or an empty string when the texts are
 * equal. A line ending in `\r\n` is shown with `[CRLF]`, so a change only in line endings shows,
 * and a change too large to compare line by line is shown whole under a note.
 */
export function unifiedDiff(
  oldText: string,
  newText: string,
  { oldLabel, newLabel }: { oldLabel: string; newLabel: string },
): string {
  const edits = lineEdits(splitLines(oldText), splitLines(newText));
  const changes = edits.flatMap((edit, at) => (edit.op === " " ? [] : [at]));
  if (changes.length === 0) return "";
  // Each edit's line numbers before it, in the old and the new text.
  const oldAt: number[] = [];
  const newAt: number[] = [];
  let oldLine = 0;
  let newLine = 0;
  for (const edit of edits) {
    oldAt.push(oldLine);
    newAt.push(newLine);
    if (edit.op !== "+") oldLine++;
    if (edit.op !== "-") newLine++;
  }
  let text = `--- ${oldLabel}\n+++ ${newLabel}\n`;
  let first = 0;
  while (first < changes.length) {
    // Changes with at most two contexts' worth of kept lines between them share a hunk.
    let last = first;
    while (last + 1 < changes.length && changes[last + 1]! - changes[last]! - 1 <= 2 * CONTEXT) last++;
    const start = Math.max(0, changes[first]! - CONTEXT);
    const end = Math.min(edits.length, changes[last]! + CONTEXT + 1);
    const hunk = edits.slice(start, end);
    const oldCount = hunk.filter((edit) => edit.op !== "+").length;
    const newCount = hunk.filter((edit) => edit.op !== "-").length;
    // An empty side starts at the line before the hunk, as `diff -u` writes it.
    const oldStart = oldCount === 0 ? oldAt[start]! : oldAt[start]! + 1;
    const newStart = newCount === 0 ? newAt[start]! : newAt[start]! + 1;
    text += `@@ -${range(oldStart, oldCount)} +${range(newStart, newCount)} @@\n`;
    for (const edit of hunk) {
      if (edit.unmatched) text += `${UNMATCHED}\n`;
      text += shown(edit);
    }
    first = last + 1;
  }
  return text;
}
