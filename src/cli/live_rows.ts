/**
 * The live rows at the bottom of a terminal: the rows a prompt or the chat's view draws again as
 * they change, under the rows printed for good. One `LiveRows` per output, shared by everything
 * that draws on it (`liveRows`), so a prompt and the chat's view each repaint over what the other
 * left rather than erasing it first.
 *
 * Each frame is one write wrapped in synchronized output, the cursor hidden while it draws. It
 * writes only the rows that changed, each over the old one, clearing what is left of it, and
 * clears below only when the frame takes fewer rows than the last. After `rerender`, a frame goes
 * back up to the live rows' top, clears from there down and draws every row again, in place.
 *
 * Knows about: the terminal's cell widths.
 */

import { ANSI, charCells, columnsOf, rowsOf } from "./cells.ts";

const ESC = "\x1b[";
/** Starts synchronized output: the terminal shows what follows only once it ends. */
export const SYNC_ON = `${ESC}?2026h`;
/** Ends synchronized output. */
export const SYNC_OFF = `${ESC}?2026l`;
const SHOW_CURSOR = `${ESC}?25h`;
const HIDE_CURSOR = `${ESC}?25l`;

/** Where the cursor goes in a frame: on line `line`, after `before`, its text before the cursor. */
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

/**
 * `line` with a clear (`ESC[K`) before each wide character that doesn't fit in the rest of its
 * row: the terminal moves it to the next row and leaves the cell it skipped as it was.
 */
function wrapClears(line: string, columns: number): string {
  let out = "";
  let col = 0;
  for (const piece of line.split(/(\x1b\[[0-9;?]*[A-Za-z])/)) {
    if (piece.startsWith("\x1b[")) {
      out += piece;
      continue;
    }
    for (const char of piece) {
      const width = charCells(char);
      if (char === "\n") col = 0;
      else if (col + width > columns) {
        // A full row needs no clear, and one there would take its last character.
        if (col < columns) out += `${ESC}K`;
        col = width;
      } else col += width;
      out += char;
    }
  }
  return out;
}

/** What a `LiveRows` writes on. */
export type LiveOutput = {
  write(text: string): unknown;
  readonly columns?: number;
  readonly rows?: number;
};

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
  /** The height the output reported when they were drawn (`undefined` when it reports none). */
  #height: number | undefined | null = null;
  /** Set once the terminal may have reflowed the lines (a resize): the next frame starts over. */
  #lost = false;
  /**
   * Set by `rerender`: how many of the rows printed for good last the next frame draws again with
   * the live rows.
   */
  #rerender: number | null = null;
  /** The rows printed for good last, at most a screen's worth, oldest first. */
  #recent: string[] = [];
  /** How many rows were staged so far. */
  #printed = 0;
  /** The rows to print for good above the live rows at the next flush. */
  #staged: string[] = [];
  /** The lines and cursor the next flush draws, or null to keep those on screen. */
  #want: { lines: string[]; cursor: RowCursor | null } | null = null;
  /** Whether the terminal shows the cursor now. */
  #visible = true;
  /** Whether a frame that doesn't place the cursor leaves it hidden. */
  #hidden = false;

  constructor(output: LiveOutput) {
    this.#output = output;
  }

  /** Print `rows` for good above the live rows at the next flush. */
  stage(rows: readonly string[]): void {
    this.#staged.push(...rows);
    this.#printed += rows.length;
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

  /**
   * The terminal may have reflowed the rows on screen (a resize): the next frame starts over. A
   * frame at a new width or height starts over by itself.
   */
  lose(): void {
    this.#lost = true;
  }

  /** How many rows were printed for good above the live rows so far, those staged counted. */
  get printed(): number {
    return this.#printed;
  }

  /**
   * Forget what is on screen: the next frame goes back up to the live rows' top, and past the
   * `above` rows printed for good last (as `printed` counts them) as far as the screen still shows
   * them, clears from there down and draws those rows and every live row again, in place. A
   * `flush` with nothing set draws the rows on screen.
   */
  rerender(above = 0): void {
    this.#rerender = above;
  }

  /**
   * Keep the cursor hidden (`true`) or shown where a frame doesn't place it, from the next flush
   * on: a flush shows or hides it even when no row changed.
   */
  hideCursor(hidden: boolean): void {
    this.#hidden = hidden;
  }

  /** Whether the cursor is kept hidden where a frame doesn't place it (`hideCursor`). */
  get cursorHidden(): boolean {
    return this.#hidden;
  }

  /**
   * Write the rows staged and the live rows as set, in one write (through `write`, else on the
   * output): the rows that changed, each over the old one; nothing when nothing changed.
   */
  flush(write: (text: string) => void = (text) => void this.#output.write(text)): void {
    let staged = this.#staged;
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
    const above = this.#rerender;
    this.#rerender = null;
    // A new height is a resize too: a terminal that shrank may have scrolled the rows up.
    const height = this.#output.rows;
    const resized = this.#width !== null && (this.#width !== columns || this.#height !== height);
    const lost = this.#lost || resized;
    this.#lost = false;
    let prev = above !== null ? [] : this.#lines;
    let heights = prev.map((line) => layout(line, columns, -1).rows);
    // The screen rows from the live rows' top that hold what was drawn before.
    let stale = heights.reduce((sum, rows) => sum + rows, 0);
    // The rows printed before a resize may have been reflowed, or not: where they are is unknown.
    if (lost) this.#recent = [];
    if (above !== null) {
      // Only the rows above that still fit on the screen with the live rows are drawn again: the
      // cursor can't go up into the scrollback, and a frame taller than the screen would scroll
      // its top rows into the scrollback once more.
      const height = this.#lines.reduce((sum, line) => sum + layout(line, columns, -1).rows, 0);
      const room = rowsOf(this.#output) - Math.max(row + 1, height);
      let up = 0;
      let first = this.#recent.length;
      // The rows staged since are among the `above`, and are printed after these.
      const oldest = Math.max(0, this.#recent.length - (above - staged.length));
      while (first > oldest) {
        const rows = layout(this.#recent[first - 1]!, columns, -1).rows;
        if (up + rows > room) break;
        up += rows;
        first--;
      }
      if (row + up > 0) out += `${ESC}${row + up}A`;
      out += `\r${ESC}J`;
      row = 0;
      // Drawn again, they go back on the rows printed last with what is staged.
      staged = [...this.#recent.splice(first), ...staged];
      prev = [];
      heights = [];
      stale = 0;
    } else if (lost) {
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
    // Where the cursor was placed, counted from the live rows' new top.
    const last = above !== null ? null : this.#placed;
    let before = last === null ? null : { row: last.row - shift, col: last.col };
    stale -= shift;
    prev = prev.slice(kept);
    heights = heights.slice(kept);
    // Writes `line` from the start of the row the cursor is on, clearing what is left of its
    // last row. A row it fills is left whole: the cursor waits in its last column, where a clear
    // would take the last character.
    const put = (line: string): number => {
      const laid = layout(line, columns, -1);
      const clear = laid.end < columns ? `${ESC}K` : "";
      out += wrapClears(line, columns).replaceAll("\n", "\r\n") + clear;
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
      before = null;
    }
    this.#recent.push(...staged);
    this.#recent.splice(0, Math.max(0, this.#recent.length - rowsOf(this.#output)));
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
      const moved = placed.row !== before?.row || placed.col !== before.col;
      if (out !== "" || moved || !this.#visible) {
        moveTo(placed.row);
        out += `${ESC}${placed.col + 1}G`;
      }
    }
    const show = placed !== null || !this.#hidden;
    this.#lines = lines;
    this.#cursor = cursor;
    this.#width = columns;
    this.#height = height;
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
