/**
 * Measuring and cutting text by the cells it takes on a terminal: a wide character takes two, a
 * combining mark none, and an ANSI sequence none.
 *
 * Knows about: the table of wide characters (`wide_cells.ts`).
 */

import { WIDE_RANGES } from "./wide_cells.ts";

/** A CSI sequence as the CLI writes it (a colour, a cursor move, a mode), which takes no cells. */
export const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

/** Whether `code` is a character a terminal draws two columns wide (`WIDE_RANGES`). */
function isWide(code: number): boolean {
  let low = 0;
  let high = WIDE_RANGES.length / 2 - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (code < WIDE_RANGES[2 * mid]!) high = mid - 1;
    else if (code > WIDE_RANGES[2 * mid + 1]!) low = mid + 1;
    else return true;
  }
  return false;
}

/** A combining mark, a zero-width joiner or a variation selector: drawn on the character before it. */
const ZERO_WIDTH = /^[\p{Mn}\p{Me}\u200d\ufe00-\ufe0f]$/u;

/** The columns one character takes: 0 for a zero-width one, 2 for a wide one, else 1. */
export function charCells(char: string): number {
  if (ZERO_WIDTH.test(char)) return 0;
  return isWide(char.codePointAt(0)!) ? 2 : 1;
}

/** The columns `text` takes on a terminal, ANSI sequences not counted. */
export function cells(text: string): number {
  let width = 0;
  for (const char of text.replace(ANSI, "")) width += charCells(char);
  return width;
}

/**
 * Return the UTF-16 ranges of `text`'s rows wrapped to `width` cells, each ending
 * before its row's trailing whitespace; an empty line is one empty row. Follows rich's
 * `divide_line`: a word keeps its trailing spaces, moves to a new row when it doesn't fit the rest
 * of this one, and is folded by characters when it is wider than a row.
 */
export function wrapLine(text: string, width: number): [start: number, end: number][] {
  const starts = [0];
  let used = 0;
  for (const match of text.matchAll(/\s*\S+\s*/g)) {
    const word = match[0];
    const length = cells(word.trimEnd());
    if (length <= width - used) {
      used += cells(word);
    } else if (length > width) {
      // Folded: each chunk of at most `width` cells starts a row; a character that doesn't fit
      // starts the next chunk.
      if (match.index > 0) starts.push(match.index);
      let at = match.index;
      let chunk = 0;
      for (const char of word) {
        const size = charCells(char);
        if (chunk > 0 && chunk + size > width) {
          starts.push(at);
          chunk = 0;
        }
        chunk += size;
        at += char.length;
      }
      used = chunk;
    } else {
      starts.push(match.index);
      used = cells(word);
    }
  }
  return starts.map((start, i) => {
    const row = text.slice(start, starts[i + 1] ?? text.length);
    return [start, start + row.trimEnd().length];
  });
}

/** `text` cut to at most `width` cells, ending in `…` when cut. */
export function fit(text: string, width: number): string {
  if (cells(text) <= width) return text;
  let kept = "";
  let used = 0;
  for (const char of text) {
    const size = charCells(char);
    if (used + size > width - 1) break;
    kept += char;
    used += size;
  }
  return `${kept}…`;
}

/**
 * `row`, which may hold ANSI sequences, cut to at most `width` cells; its styles, when it has
 * any, are reset when it is cut.
 */
export function clip(row: string, width: number): string {
  if (cells(row) <= width) return row;
  const sequence = new RegExp(`^${ANSI.source}`);
  let kept = "";
  let used = 0;
  let at = 0;
  while (at < row.length) {
    const found = sequence.exec(row.slice(at));
    if (found !== null) {
      kept += found[0];
      at += found[0].length;
      continue;
    }
    const char = String.fromCodePoint(row.codePointAt(at)!);
    if (used + charCells(char) > width) break;
    kept += char;
    used += charCells(char);
    at += char.length;
  }
  return row.search(ANSI) === -1 ? kept : `${kept}\x1b[0m`;
}

/** The columns `out` is wide: as it reports, else 80. */
export function columnsOf(out: { readonly columns?: number }): number {
  return out.columns || 80;
}

/** The rows `out` is high: as it reports, else 24. */
export function rowsOf(out: { readonly rows?: number }): number {
  return out.rows || 24;
}
