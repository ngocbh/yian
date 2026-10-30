/**
 * The live rows at the bottom of a terminal: the rows a prompt or the chat's view draws again as
 * they change, under the rows printed for good. One `LiveRows` per output, shared by everything
 * that draws on it (`liveRows`), so a prompt and the chat's view each repaint over what the other
 * left rather than erasing it first.
 *
 * Each frame is one write wrapped in synchronized output, the cursor hidden while it draws. It
 * writes only the rows that changed, each over the old one, clearing what is left of it, and
 * clears below only when the frame takes fewer rows than the last.
 *
 * Knows about: the terminal's cell widths.
 */

import { ANSI, charCells, columnsOf } from "./cells.ts";

const ESC = "\x1b[";
/** Starts synchronized output: the terminal shows what follows only once it ends. */
export const SYNC_ON = `${ESC}?2026h`;
/** Ends synchronized output. */
export const SYNC_OFF = `${ESC}?2026l`;
const SHOW_CURSOR = `${ESC}?25h`;
const HIDE_CURSOR = `${ESC}?25l`;

/** Where the cursor goes in a frame: on line `line`, after `before` (its text before the cursor). */
export interface RowCursor {
  readonly line: number;
  readonly before: string;
}

/**
 * Lay `line` out on a `columns`-wide terminal as the terminal wraps it: a character that doesn't
 * fit in the rest of a row (a wide one in the last column) starts the next row, and so does a `\n`.
 * Returns the rows it takes, the cells its last row uses, and where the character at code-point
 * offset `at` (ANSI sequences not counted; -1 for none) sits.
 */
export function layout(
  line: string,
  columns: number,
  at: number,
): { rows: number; end: number; cursor: { row: number; col: number } } {
  let row = 0;
  let col = 0;
  let cursor: { row: number; col: number } | undefined;
  for (const [i, char] of Array.from(line.replace(ANSI, "")).entries()) {
    if (i === at) cursor = { row, col: Math.min(col, columns - 1) };
    if (char === "\n") {
      row++;
      col = 0;
      continue;
    }
    const width = charCells(char);
    if (col + width > columns) {
      row++;
      col = 0;
      if (i === at) cursor = { row, col };
    }
    col += width;
  }
  // At the end of a line that fills its last row, the terminal holds the cursor in the last
  // column until the next character wraps it; the cursor stays there rather than adding a row.
  return { rows: row + 1, end: col, cursor: cursor ?? { row, col: Math.min(col, columns - 1) } };
}

/** What a `LiveRows` writes on. */
export type LiveOutput = { write(text: string): unknown; readonly columns?: number };

/**
 * The live rows drawn on one output. `stage` queues rows to print for good above them, `set` the
 * rows they should be, and `flush` writes the difference in one write; `draw` does both.
 */
export class LiveRows {
  readonly #output: LiveOutput;
  /** The lines on screen, from the live rows' top. */
  #lines: string[] = [];
  /** The cursor the lines on screen were drawn with. */
  #cursor: RowCursor | null = null;
  /** The screen row the cursor is on, counted from the live rows' top. */
  #row = 0;
  /** Where the cursor was placed last (a `RowCursor`'s screen row and column), if it was. */
  #placed: { row: number; col: number } | null = null;
  /** The width the lines on screen were drawn at. */
  #width: number | null = null;
  /** Set once the terminal may have reflowed the lines (a resize): the next frame starts over. */
  #lost = false;
  /** The rows to print for good above the live rows at the next flush. */
  #staged: string[] = [];
  /** The lines and cursor the next flush draws, or null to keep those on screen. */
  #want: { lines: string[]; cursor: RowCursor | null } | null = null;
  /** Whether the terminal shows the cursor now. */
  #visible = true;
  /** Whether a frame that doesn't place the cursor leaves it hidden. */
  hidden = false;

  constructor(output: LiveOutput) {
    this.#output = output;
  }

  /** The lines on screen, or those the next flush draws. */
  lines(): readonly string[] {
    return this.#want?.lines ?? this.#lines;
  }

  /** Print `rows` for good above the live rows at the next flush. */
  stage(rows: readonly string[]): void {
    this.#staged.push(...rows);
  }

  /** Make `lines` the live rows at the next flush, the cursor after `cursor` (null to hide it). */
  set(lines: readonly string[], cursor: RowCursor | null): void {
    this.#want = { lines: [...lines], cursor };
  }

  /** `set` then `flush`. */
  draw(lines: readonly string[], cursor: RowCursor | null, write?: (text: string) => void): void {
    this.set(lines, cursor);
    this.flush(write);
  }

  /** The terminal may have reflowed the rows on screen (it was resized): the next frame starts over. */
  lose(): void {
    this.#lost = true;
  }

  /**
   * Keep the cursor shown (`true`) or hidden when a frame doesn't place it; returns what shows or
   * hides it now, for the caller to write.
   */
  cursorMode(visible: boolean): string {
    this.hidden = !visible;
    this.#visible = visible;
    return visible ? SHOW_CURSOR : HIDE_CURSOR;
  }

  /**
   * Write the rows staged and the live rows as set, in one write (through `write`, else on the
   * output): the rows that changed, each over the old one; nothing when nothing changed.
   */
  flush(write: (text: string) => void = (text) => void this.#output.write(text)): void {
    const staged = this.#staged;
    const want = this.#want ?? { lines: this.#lines, cursor: this.#cursor };
    this.#staged = [];
    this.#want = null;
    const columns = columnsOf(this.#output);
    let out = "";
    let row = this.#row;
    // Moves to the start of screen row `target` of the live rows. Down goes by newlines, which
    // scroll the screen once past its last row.
    const moveTo = (target: number): void => {
      if (target < row) out += `${ESC}${row - target}A`;
      out += target > row ? "\r\n".repeat(target - row) : "\r";
      row = target;
    };
    const lost = this.#lost || (this.#width !== null && this.#width !== columns);
    this.#lost = false;
    let prev = this.#lines;
    let heights = prev.map((line) => layout(line, columns, -1).rows);
    // The screen rows from the live rows' top that hold what was drawn before.
    let stale = heights.reduce((sum, rows) => sum + rows, 0);
    if (lost) {
      moveTo(0);
      out += `${ESC}J`;
      prev = [];
      heights = [];
      stale = 0;
    }
    // Staged rows the screen shows already at the top of the live rows stay as they are: the live
    // rows start below them.
    let kept = 0;
    while (kept < staged.length && kept < prev.length && staged[kept] === prev[kept]) kept++;
    const shift = heights.slice(0, kept).reduce((sum, rows) => sum + rows, 0);
    row -= shift;
    stale -= shift;
    prev = prev.slice(kept);
    heights = heights.slice(kept);
    // Writes `line` from the start of the row the cursor is on, clearing what is left of its
    // last row. A row it fills is left whole: the cursor waits in its last column, where a clear
    // would take the last character.
    const put = (line: string): number => {
      const laid = layout(line, columns, -1);
      out += line.replaceAll("\n", "\r\n") + (laid.end < columns ? `${ESC}K` : "");
      return laid.rows;
    };
    for (const line of staged.slice(kept)) {
      moveTo(0);
      const rows = put(line);
      out += "\r\n";
      row = 0;
      stale = Math.max(0, stale - rows);
      // The rows below moved down: none of them is where it was.
      prev = [];
    }
    const lines = want.lines;
    const tops: number[] = [];
    let top = 0;
    let prevTop = 0;
    for (const [i, line] of lines.entries()) {
      tops.push(top);
      const same = i < prev.length && prev[i] === line && prevTop === top;
      if (!same) {
        moveTo(top);
        const rows = put(line);
        row = top + rows - 1;
        top += rows;
      } else top += heights[i]!;
      if (i < prev.length) prevTop += heights[i]!;
    }
    // Fewer rows than before: what is left below goes.
    if (top < stale) {
      moveTo(top);
      out += `${ESC}J`;
    }
    let placed: { row: number; col: number } | null = null;
    const cursor = want.cursor;
    if (cursor !== null && cursor.line < lines.length) {
      const at = Array.from(cursor.before.replace(ANSI, "")).length;
      const spot = layout(lines[cursor.line]!, columns, at).cursor;
      placed = { row: tops[cursor.line]! + spot.row, col: spot.col };
      const moved = placed.row !== this.#placed?.row || placed.col !== this.#placed.col;
      if (out !== "" || moved || !this.#visible) {
        moveTo(placed.row);
        out += `${ESC}${placed.col + 1}G`;
      }
    }
    const show = placed !== null || !this.hidden;
    this.#lines = lines;
    this.#cursor = cursor;
    this.#width = columns;
    this.#row = row;
    this.#placed = placed;
    if (out === "" && show === this.#visible) return;
    const hide = this.#visible ? HIDE_CURSOR : "";
    this.#visible = show;
    write(`${SYNC_ON}${hide}${out}${show ? SHOW_CURSOR : ""}${SYNC_OFF}`);
  }
}

/** The `LiveRows` of each output. */
const SHARED = new WeakMap<object, LiveRows>();

/** The live rows drawn on `output`: one for each output, shared by all that draw on it. */
export function liveRows(output: LiveOutput): LiveRows {
  let rows = SHARED.get(output);
  if (rows === undefined) {
    rows = new LiveRows(output);
    SHARED.set(output, rows);
  }
  return rows;
}
