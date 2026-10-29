/**
 * The CLI's terminal: the prompts a run shows on the host's terminal, drawn with questionary's look
 * from keypresses, one widget at a time, always giving the terminal back in cooked mode.
 */

import { emitKeypressEvents } from "node:readline";

import { compareCodePoints, reprText } from "../diagnostics.ts";
import { coerceParam } from "../nodes/binding.ts";
import type { Parameter } from "../nodes/contracts.ts";
import type { QuestionSpec } from "../nodes/tool/questions.ts";
import type { ToolContext } from "../tools/context.ts";
import type { AskUserRequest } from "../tools/registry.ts";
import { type JsonValue, JSONDecodeError, parseJson } from "../typesys/json.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isInboxType,
  isListKind,
  rejectDeepNesting,
  type Type,
  TypeCheckError,
  type Value,
  ValueKind,
  writeAsText,
} from "../typesys/values.ts";
import { ANSI, cells, charCells, clip, columnsOf, fit, rowsOf, wrapLine } from "./cells.ts";
import { box } from "./render.ts";
import { sanitize } from "./sanitize.ts";

/** Why a prompt stopped without an answer. */
type StopReason = "interrupt" | "eof" | "closed";

/** The input a `Terminal` reads keys from. */
type Input = NodeJS.ReadableStream & {
  isTTY?: boolean;
  isRaw?: boolean;
  setRawMode?(mode: boolean): unknown;
  readableEnded?: boolean;
  destroyed?: boolean;
};

/** The output a `Terminal` draws on: only `write`, `columns` and `rows` are used. */
type Output = Pick<NodeJS.WritableStream, "write"> & { columns?: number; rows?: number };

export interface TerminalOptions {
  /** The clock the keys after a prompt gives way are timed on, in milliseconds; `Date.now`. */
  now?: () => number;
  /** Where keys come from; defaults to process.stdin. Raw mode is set only when it is a TTY. */
  input?: Input;
  /** Where prompts are drawn; defaults to process.stderr (stdout stays the run's result). */
  output?: Output;
  /** Called once when Ctrl-C ("interrupt") or the end of input ("eof") closes the terminal. */
  onStop?: (reason: "interrupt" | "eof") => void;
  /**
   * Called with true when a prompt turn takes the terminal and false when it gives it back, with
   * `printed` true when the turn left rows on the screen (a collapsed answer, an approval's card).
   */
  onActive?: (active: boolean, printed: boolean) => void;
  /**
   * Write each prompt once as it opens, with no escape codes or carriage returns, and leave the
   * input in cooked mode: for an output or input that isn't a terminal.
   */
  plain?: boolean;
}

/** What `Terminal.approve` asks the user to approve. */
export interface ApproveRequest {
  /** What is to be done, such as `Write a.yaml`; the select's label. */
  title: string;
  /** More about it, written above the select; empty for none. */
  body: string;
  /** A unified diff of the change, or null for none. */
  diff: string | null;
  /** The answers, the approving one first and the rejecting one last. */
  choices: readonly string[];
  /** Ends the prompt, with no answer, when it aborts. */
  signal?: AbortSignal;
  /** Draw the body and diff as a framed card (`diffCard`), rather than as plain lines. */
  frame?: boolean;
  /** The choice preselected, one of `choices`; the first when absent. */
  preselect?: string;
}

/** A prompt ended because the terminal closed (Ctrl-C, end of input, or close()). */
export class TerminalStopped extends Error {
  override name = "TerminalStopped";
  /** What closed the terminal. */
  readonly reason: StopReason;

  /** Build the error for `reason`. */
  constructor(reason: StopReason) {
    super("terminal interaction stopped");
    this.reason = reason;
  }
}

/** One keypress, as the key loop hands it to a widget (or to a `watchKeys` editor). */
export interface Key {
  name: string | undefined;
  ctrl: boolean;
  meta: boolean;
  /** The text the key types, when it types any. */
  str: string | undefined;
  /**
   * Whether the key is a bracketed paste: one key, whose `str` holds all the text pasted, each
   * line break as `\n`.
   */
  paste: boolean;
}

/** How a widget or a turn ended without an answer: a stop, or its signal aborted. */
type Ending = StopReason | "aborted";

/** How a widget ended. */
type Outcome<T> = { value: T } | { stop: Ending } | { error: unknown };

/** One prompt's state and drawing; the key loop owns the terminal around it. */
interface Widget<T> {
  /** A multi-line label's lines above its last, written once above the frame (`labelPreface`). */
  preface: string;
  /** The frame's lines, styled; a `\n` inside one starts a new row. */
  lines(): string[];
  /** The cursor's line in the frame and that line's text before it, or null to hide the cursor. */
  cursor(): { row: number; before: string } | null;
  /** Apply one key; an outcome ends the widget. */
  key(key: Key): { value: T } | { stop: "interrupt" | "eof" } | undefined;
  /** The one line the frame collapses to once the widget has ended; null erases it and leaves none. */
  done(outcome: Outcome<T>): string | null;
  /**
   * For a plain terminal, after a key: the error a refused answer drew, with the typed text
   * dropped so the next line is a fresh answer; undefined when nothing was refused.
   */
  refused?(): string | undefined;
  /** Calls `redraw` when the widget's lines change without a key; returns what stops it. */
  watch?(redraw: () => void): () => void;
  /** Drawn on the whole (alternate) screen, which the terminal gives back as it was. */
  screen?: boolean;
  /** Called once the widget has left the screen, before the turn behind it can draw. */
  closed?(): void;
}

const ESC = "\x1b[";
const SHOW_CURSOR = `${ESC}?25h`;
const HIDE_CURSOR = `${ESC}?25l`;
const PASTE_ON = `${ESC}?2004h`;
const PASTE_OFF = `${ESC}?2004l`;
// The alternate screen: switching back restores the main screen and the cursor as they were.
const ALT_ON = `${ESC}?1049h`;
const ALT_OFF = `${ESC}?1049l`;

/**
 * `output` for a plain terminal: what is written loses its escape codes and carriage returns, and
 * a write left empty is skipped.
 */
function plainOutput(output: Output): Output {
  return {
    write(chunk: unknown, ...rest: unknown[]): boolean {
      const text = String(chunk).replace(ANSI, "").replaceAll("\r", "");
      if (text === "") return true;
      return (output.write as (...args: unknown[]) => boolean)(text, ...rest);
    },
    get columns() {
      return output.columns;
    },
  };
}

/** A unified diff's hunk header: where its old and new lines start. */
const HUNK = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/;

/**
 * The lines of an approval's card, `columns` wide: a box titled `title` holding `body` and then
 * `diff` (a unified diff) without its `---`/`+++` header, each line numbered (an old line's number
 * for a `-` line, the new one's otherwise), `-` lines red and `+` lines green, a dim `⋯` between
 * hunks; a title too long for the edge is cut with `…`. Long lines wrap at their spaces inside the
 * box (`wrapLine`). The text is drawn as it is given: sanitize it first.
 */
export function diffCard(title: string, body: string, diff: string, columns: number): string[] {
  const inner = Math.max(1, columns - 4);
  const rows: [string, string][] = [];
  const add = (text: string, codes: string | null, gutter = "") => {
    const lead = cells(gutter);
    for (const [i, [start, end]] of wrapLine(text, Math.max(1, inner - lead)).entries()) {
      const part = (i === 0 ? gutter : " ".repeat(lead)) + text.slice(start, end);
      rows.push([part, codes === null ? part : style(codes, part)]);
    }
  };
  if (body !== "") for (const line of body.replace(/\n$/, "").split("\n")) add(line, null);
  const lines = diff === "" ? [] : diff.replace(/\n$/, "").split("\n").filter((line) => !/^(---|\+\+\+) /.test(line));
  // The widest line number, for the gutter.
  let widest = 1;
  let oldLine = 0;
  let newLine = 0;
  for (const line of lines) {
    const hunk = HUNK.exec(line);
    if (hunk !== null) [oldLine, newLine] = [Number(hunk[1]), Number(hunk[2])];
    else if (line.startsWith("-")) widest = Math.max(widest, String(oldLine++).length);
    else if (line.startsWith("+") || line.startsWith(" ")) widest = Math.max(widest, String(newLine++).length);
  }
  if (body !== "" && lines.length > 0) add("", null);
  let hunks = 0;
  for (const line of lines) {
    const hunk = HUNK.exec(line);
    if (hunk !== null) {
      if (hunks++ > 0) add("⋯", "2");
      [oldLine, newLine] = [Number(hunk[1]), Number(hunk[2])];
    } else if (line.startsWith("-")) {
      add(line.slice(1), "31", `${String(oldLine++).padStart(widest)} - `);
    } else if (line.startsWith("+")) {
      add(line.slice(1), "32", `${String(newLine++).padStart(widest)} + `);
    } else if (line.startsWith(" ")) {
      oldLine++;
      add(line.slice(1), null, `${String(newLine++).padStart(widest)}   `);
    } else add(line, "2");
  }
  // The terminal's widgets draw in colour whatever the environment says, and so does the card.
  const card = box(rows, "cyan", { isTTY: true }, () => undefined, fit(title, Math.max(1, inner - 2)), inner);
  return card.split("\n");
}

/** `text` with each `\n` written as `\r\n`, since raw mode may not translate it. */
function crlf(text: string): string {
  return text.replaceAll("\n", "\r\n");
}

/** Wrap `text` in the SGR `codes`. */
function style(codes: string, text: string): string {
  return `${ESC}${codes}m${text}${ESC}0m`;
}

/**
 * Lay `line` out on a `columns`-wide terminal as the terminal wraps it: a character that doesn't
 * fit in the rest of a row (a wide one in the last column) starts the next row, and so does a `\n`.
 * Returns the rows it takes and where the character at code-point offset `at` (ANSI sequences not
 * counted) sits.
 */
function layout(line: string, columns: number, at: number): { rows: number; cursor: { row: number; col: number } } {
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
  return { rows: row + 1, cursor: cursor ?? { row, col: Math.min(col, columns - 1) } };
}

/**
 * A prompt's head `? <label>`, with the dim `hint` after it when given. Of a multi-line label it
 * holds only the last line: the mark and the lines before are its `labelPreface`.
 */
function promptHead(mark: string, label: string, hint?: string): string {
  const lines = label.split("\n");
  const head = lines.length === 1 ? `${style("1;36", mark)} ${style("1", label)}` : style("1", lines.at(-1)!);
  return hint ? `${head} ${style("2", hint)}` : head;
}

/**
 * The lines of a multi-line label before its last, the first after the mark, each ending in
 * `\r\n`: written once above the frame, so a long label (a plan to approve) is never redrawn.
 * Empty for a one-line label.
 */
function labelPreface(mark: string, label: string): string {
  const lines = label.split("\n").slice(0, -1);
  return lines.map((line, i) => `${i === 0 ? `${style("1;36", mark)} ` : ""}${style("1", line)}\r\n`).join("");
}

/** `index` moved by an up (up, `k`, Ctrl-P) or down (down, `j`, Ctrl-N) key among `count` rows, wrapping. */
function moved(key: Key, index: number, count: number): number {
  const up = key.ctrl ? key.name === "p" : !key.meta && (key.name === "up" || key.name === "k");
  const down = key.ctrl ? key.name === "n" : !key.meta && (key.name === "down" || key.name === "j");
  if (up) return (index - 1 + count) % count;
  return down ? (index + 1) % count : index;
}

/** The collapsed line `? <label> <answer>`, with nothing after the label when `answer` is empty. */
function collapsed(mark: string, label: string, answer: string): string {
  const head = promptHead(mark, label);
  return answer === "" ? head : `${head} ${style("36", answer)}`;
}

/** Whether `key` types text: printable characters, no control or meta modifier. */
function printable(key: Key): key is Key & { str: string } {
  return key.str !== undefined && key.str !== "" && !key.ctrl && !key.meta && !/[\x00-\x1f\x7f]/.test(key.str);
}

/** A paste longer than this many lines goes into a `LineEditor` as one token. */
const PASTE_LINES = 3;
/** A paste longer than this many characters goes into a `LineEditor` as one token. */
const PASTE_CHARS = 800;
/** How long a paste pauses before Escape or Ctrl-C, as a key of its own, can end it. */
const PASTE_IDLE_MS = 500;
/** The characters a paste keeps; past them the rest of it is dropped. */
const PASTE_MAX_CHARS = 16_000_000;

/** A paste a `LineEditor` holds as one token, `[Pasted text #<number> +<lines> lines]`. */
export interface PastedText {
  /** The text pasted, which the token stands for. */
  readonly text: string;
  /** The token's number among those of its message. */
  readonly number: number;
}

/** What a `LineEditor` holds: typed text and pasted tokens, in order. */
export type Draft = readonly (string | PastedText)[];

/** The lines `text` holds; a newline that ends it starts no line. */
function lineCount(text: string): number {
  return text.replace(/\n$/, "").split("\n").length;
}

/** The token a paste is drawn as: `[Pasted text #1 +40 lines]`. */
function pasteLabel(paste: PastedText): string {
  const lines = lineCount(paste.text);
  return `[Pasted text #${paste.number} +${lines} line${lines === 1 ? "" : "s"}]`;
}

/** Whether a paste of `text` goes into a `LineEditor` as a token rather than as text. */
function collapses(text: string): boolean {
  return lineCount(text) > PASTE_LINES || Array.from(text).length > PASTE_CHARS;
}

/**
 * `text` as a draft: one token when it is as long as a paste the input box collapses, else the
 * text.
 */
export function textDraft(text: string): Draft {
  return collapses(text) ? [{ text, number: 1 }] : [text];
}

/** The text `draft` stands for, each token expanded to its paste: what is sent. */
export function draftText(draft: Draft): string {
  return draft.map((part) => (typeof part === "string" ? part : part.text)).join("");
}

/** `draft` as it is shown, each token as its label: what is echoed. */
export function draftLabel(draft: Draft): string {
  return draft.map((part) => (typeof part === "string" ? part : pasteLabel(part))).join("");
}

/**
 * `drafts` as one draft: those holding anything, a space between them, their tokens numbered
 * afresh from 1, since they become one message.
 */
export function joinDrafts(drafts: readonly Draft[]): Draft {
  const joined: (string | PastedText)[] = [];
  let number = 0;
  for (const draft of drafts.filter((each) => each.length > 0 && draftText(each) !== "")) {
    if (joined.length > 0) joined.push(" ");
    for (const part of draft) {
      joined.push(typeof part === "string" ? part : { text: part.text, number: ++number });
    }
  }
  return joined;
}

/** One unit of a text widget: a character, or a pasted token, which the keys treat as one. */
type Unit = string | PastedText;

/** `draft` as units: each character of its text, each token whole. */
function unitsOf(draft: string | Draft): Unit[] {
  const parts: Draft = typeof draft === "string" ? [draft] : draft;
  return parts.flatMap((part): Unit[] => (typeof part === "string" ? Array.from(part) : [part]));
}

/** What `unit` draws: a newline as `⏎`, a tab as four spaces, a token as its label. */
function drawn(unit: Unit): string {
  if (typeof unit !== "string") return pasteLabel(unit);
  return unit === "\n" ? "⏎" : unit === "\t" ? "    " : unit;
}

/** The text `initial` stands for: itself, or a draft's text with its tokens expanded. */
function textOf(initial: string | Draft): string {
  return typeof initial === "string" ? initial : draftText(initial);
}

/** Whether `a` and `b` hold the same characters and the very same tokens. */
function sameUnits(a: readonly Unit[], b: readonly Unit[]): boolean {
  return a.length === b.length && a.every((unit, i) => unit === b[i]);
}

/**
 * A one-line text editor; `mask` draws `*` per character and keeps the answer off the screen. With
 * `collapse`, a long paste (more than `PASTE_LINES` lines or `PASTE_CHARS` characters) goes in as
 * one token, which the caret and the deleting keys treat as one character and the answer expands.
 */
function textWidget(options: {
  label: string;
  initial?: string;
  mask?: boolean;
  mark?: string;
  instruction?: string;
  validate?: (text: string) => string | undefined;
  collapse?: boolean;
}): Widget<string> & {
  text(): string;
  caret(): number;
  units(): readonly Unit[];
  set(text: string | Draft): void;
} {
  const mark = options.mark ?? "?";
  let chars: Unit[] = Array.from(options.initial ?? "");
  let pos = chars.length;
  let error: string | undefined;
  const shown = (part: readonly Unit[]): string =>
    options.mask ? "*".repeat(part.length) : part.map((unit) => drawn(unit)).join("");
  const head = (): string => `${promptHead(mark, options.label, options.instruction)} `;
  const text = (): string => draftText(chars);
  const insert = (added: readonly Unit[]): void => {
    chars.splice(pos, 0, ...added);
    pos += added.length;
  };
  const paste = (pasted: string): void => {
    // A paste keeps its newlines and tabs, and drops the other control characters.
    const text = pasted.replace(/[\x00-\x08\x0b-\x1f\x7f]/g, "");
    if (!(options.collapse && collapses(text))) return insert(Array.from(text));
    let last = 0;
    for (const unit of chars) if (typeof unit !== "string") last = Math.max(last, unit.number);
    insert([{ text, number: last + 1 }]);
  };
  return {
    preface: labelPreface(mark, options.label),
    lines: () => (error === undefined ? [head() + shown(chars)] : [head() + shown(chars), style("31", error)]),
    cursor: () => ({ row: 0, before: head() + shown(chars.slice(0, pos)) }),
    key(key) {
      const newline = key.name === "return" || key.name === "enter";
      error = undefined;
      if (key.paste) paste(key.str ?? "");
      else if (newline) {
        const answer = text();
        error = options.validate?.(answer);
        if (error === undefined) return { value: answer };
      } else if (key.ctrl && key.name === "d") {
        if (chars.length === 0) return { stop: "eof" };
      } else if (key.name === "home" || (key.ctrl && key.name === "a")) pos = 0;
      else if (key.name === "end" || (key.ctrl && key.name === "e")) pos = chars.length;
      else if (key.ctrl && key.name === "u") {
        chars.splice(0, pos);
        pos = 0;
      } else if (key.name === "left") pos = Math.max(0, pos - 1);
      else if (key.name === "right") pos = Math.min(chars.length, pos + 1);
      else if (key.name === "backspace") {
        if (pos > 0) chars.splice(--pos, 1);
      } else if (key.name === "delete") chars.splice(pos, 1);
      else if (printable(key)) insert(Array.from(key.str));
      return undefined;
    },
    done: (outcome) =>
      collapsed(mark, options.label, "value" in outcome && !options.mask ? shown(Array.from(outcome.value)) : ""),
    refused() {
      if (error === undefined) return undefined;
      chars.length = 0;
      pos = 0;
      return error;
    },
    text,
    caret: () => pos,
    units: () => chars.slice(),
    set(text) {
      // A new array rather than a splice, whose arguments a long text would overflow.
      chars = unitsOf(text);
      pos = chars.length;
      error = undefined;
    },
  };
}

/** One row of a `PromptList`: what names its item, and its text, drawn as given (styled). */
export interface PromptRow {
  readonly key: string;
  readonly text: string;
}

/**
 * A list of rows under a framed prompt (`LineEditor`), which Down moves into: each row is one
 * item, one row of the screen. No rows hides the list.
 */
export interface PromptList {
  rows(): readonly PromptRow[];
  /** Added to the prompt's hint while the list has rows (how to reach it, say). */
  hint?: string;
  /** Calls `listener` when the rows may have changed; returns what stops it. */
  subscribe?(listener: () => void): () => void;
}

/**
 * What a key did at a `LineEditor`: Enter in the box answers its text (`value`), Enter on a list
 * row `open`s it, Escape in the box is `escape`, Ctrl-D in an empty box the end of input; `null`
 * when the key only edited or moved.
 */
export type EditorAction =
  | { value: string }
  | { open: string }
  | { escape: true }
  | { stop: "eof" }
  | null;

/** The screen rows a prompt's list leaves for what is drawn above the box. */
const LIST_MARGIN = 4;

/**
 * How long a prompt that took a yielding `readLine`'s place drops keys, in milliseconds: an Enter
 * meant for the line mustn't answer it.
 */
const YIELD_GRACE_MS = 400;

/**
 * A line typed in a framed box `columns()` wide: a box holding `> ` and the text, folded onto as
 * many rows as it needs, the dim `hint` below it (and the list's own after it, when it fits), and
 * under that `list`'s rows when it has any. The box has the keys, as a line prompt does, until
 * Down on its last row moves them to the list; there Up and Down move between rows, and Up from
 * the top row, Escape or a typed character goes back to the box. On a short screen the list shows
 * as many rows as fit (`rows()` high, leaving `LIST_MARGIN` rows), the row with the keys among
 * them, and a dim `+N more` for the others. A long paste (more than `PASTE_LINES` lines or
 * `PASTE_CHARS` characters) goes in as one token, `[Pasted text #1 +40 lines]`, numbered among
 * the tokens of the box: the caret steps over it, Backspace, Delete and Ctrl-U take it whole, and
 * the text answered holds the paste. The text and where the keys are last across prompts, so one
 * editor can serve a chat's prompt and the box it keeps up while a reply runs.
 */
export class LineEditor {
  readonly #text = textWidget({ label: "", mark: ">", collapse: true });
  readonly #columns: () => number;
  readonly #rows: () => number;
  readonly #hint: string;
  /** The list under the hint, if any. */
  readonly list: PromptList | null;
  /** The list row with the keys, its key and place, or null while the box has them. */
  #focus: { key: string; index: number } | null = null;

  /** @param options.rows The screen's height, which the list keeps within; no bound when absent. */
  constructor(options: {
    columns: () => number;
    rows?: () => number;
    hint: string;
    list?: PromptList | null;
  }) {
    this.#columns = options.columns;
    this.#rows = options.rows ?? (() => Infinity);
    this.#hint = options.hint;
    this.list = options.list ?? null;
  }

  /** The text typed, each pasted token expanded to its paste. */
  text(): string {
    return this.#text.text();
  }

  /** The text typed and the pasted tokens, in order. */
  draft(): Draft {
    const draft: (string | PastedText)[] = [];
    for (const unit of this.#text.units()) {
      const last = draft.length - 1;
      if (typeof unit === "string" && typeof draft[last] === "string") draft[last] += unit;
      else draft.push(unit);
    }
    return draft;
  }

  /** Replace the text (and its tokens, given a draft), the cursor after it. */
  set(text: string | Draft): void {
    this.#text.set(text);
  }

  /** Where the list row with the keys is, or null while the box has them. */
  focus(): number | null {
    return this.#find(this.list?.rows() ?? []);
  }

  /** Where the focused row is among `rows`: by its key; once it went, the row now where it was. */
  #find(rows: readonly PromptRow[]): number | null {
    if (this.#focus === null) return null;
    const at = rows.findIndex((row) => row.key === this.#focus!.key);
    // A row that went (its run ended) gives the keys to the one in its place, or above, or the box.
    const last = rows.length - 1;
    const index = at !== -1 ? at : last < 0 ? null : Math.min(this.#focus.index, last);
    this.#focus = index === null ? null : { key: rows[index]!.key, index };
    return index;
  }

  /** Give the keys to row `index` of `rows`, or to the box for null. */
  #focusOn(rows: readonly PromptRow[], index: number | null): void {
    this.#focus = index === null ? null : { key: rows[index]!.key, index };
  }

  /** The list's rows as drawn: at most what fits, the focused one among them, then `+N more`. */
  #listRows(boxRows: number): string[] {
    const rows = this.list?.rows() ?? [];
    const focus = this.#find(rows);
    const room = Math.max(1, this.#rows() - boxRows - 1 - LIST_MARGIN);
    const pointer = (i: number): string => (i === focus ? `${style("1;36", "»")} ` : "  ");
    if (rows.length <= room) return rows.map((row, i) => pointer(i) + row.text);
    const shown = Math.max(1, room - 1);
    const start = Math.min(Math.max(0, (focus ?? 0) - shown + 1), rows.length - shown);
    const drawn = rows.slice(start, start + shown).map((row, i) => pointer(start + i) + row.text);
    return [...drawn, `  ${style("2", `+${rows.length - shown} more`)}`];
  }

  /** The rows, styled: the box, the hint and the list. */
  lines(): string[] {
    const { rows, width } = this.#fold();
    const { start, end } = this.#window();
    const border = (part: string): string => style("2", part);
    // A box cut to fit says on its edge how many rows it leaves out.
    const edge = (count: number, arrow: string): string => {
      if (count === 0) return "─".repeat(width + 4);
      const label = `─ ${arrow} ${count} more `;
      return label + "─".repeat(Math.max(0, width + 4 - cells(label)));
    };
    const list = this.#listRows(end - start + 2);
    const extra = list.length > 0 ? this.list?.hint : undefined;
    const both = extra ? `${this.#hint} · ${extra}` : this.#hint;
    // The list's hint only when it fits on the row: the box's hint matters more.
    const hint = cells(both) + 2 <= this.#columns() ? both : this.#hint;
    return [
      border(`╭${edge(start, "↑")}╮`),
      ...rows.slice(start, end).map((each, i) => {
        const body = each.chars.join("");
        const pad = " ".repeat(width - cells(body));
        return `${border("│")} ${lead(start + i)}${body}${pad} ${border("│")}`;
      }),
      border(`╰${edge(rows.length - end, "↓")}╯`),
      `  ${style("2", hint)}`,
      ...list,
    ];
  }

  /** The cursor's row among `lines()` and its text before it; null while the list has the keys. */
  cursor(): { row: number; before: string } | null {
    if (this.focus() !== null) return null;
    const { rows, row } = this.#fold();
    const before = rows[row]!.chars.slice(0, this.#text.caret() - rows[row]!.start).join("");
    return { row: row - this.#window().start + 1, before: `│ ${lead(row)}${before}` };
  }

  /**
   * The text rows the box draws, `start` to `end`: all of them, or on a short screen as many as
   * fit (`rows()` high, leaving `LIST_MARGIN` rows above it and three for its edges and the hint),
   * the cursor's row among them.
   */
  #window(): { start: number; end: number } {
    const { rows, row } = this.#fold();
    const room = Math.max(1, this.#rows() - LIST_MARGIN - 3);
    if (rows.length <= room) return { start: 0, end: rows.length };
    const start = Math.min(Math.max(0, row - room + 1), rows.length - room);
    return { start, end: start + room };
  }

  /** Apply one key. */
  key(key: Key): EditorAction {
    // A paste is text, whatever keys it holds: it goes into the box, its newlines kept, or a long
    // one as a token.
    if (key.paste) {
      this.#focus = null;
      this.#text.key(key);
      return null;
    }
    const list = this.list?.rows() ?? [];
    const focus = this.#find(list);
    const count = list.length;
    const enter = key.name === "return" || key.name === "enter";
    const plain = !key.ctrl && !key.meta;
    if (focus !== null) {
      if (enter) return { open: list[focus]!.key };
      if (key.name === "escape" || (plain && key.name === "up" && focus === 0)) this.#focus = null;
      else if (plain && key.name === "up") this.#focusOn(list, focus - 1);
      else if (plain && key.name === "down") this.#focusOn(list, Math.min(focus + 1, count - 1));
      else if (printable(key)) {
        this.#focus = null;
        this.#text.key(key);
      }
      return null;
    }
    if (plain && key.name === "down" && count > 0) {
      const { rows, row } = this.#fold();
      if (row === rows.length - 1) this.#focusOn(list, 0);
      return null;
    }
    if (key.name === "escape") return { escape: true };
    const ended = this.#text.key(key);
    if (ended === undefined) return null;
    return "value" in ended ? ended : ended.stop === "eof" ? { stop: "eof" } : null;
  }

  /**
   * The rows the text folds onto, each with the index of its first unit (a character or a token)
   * and what each unit draws, and the cursor's row.
   */
  #fold(): { rows: { start: number; chars: string[] }[]; row: number; width: number } {
    const width = Math.max(1, this.#columns() - 6);
    // A token is drawn whole on one row, cut to fit when it is wider than one.
    const chars = this.#text
      .units()
      .map((unit) => (typeof unit === "string" ? drawn(unit) : fit(pasteLabel(unit), width)));
    const rows = [{ start: 0, chars: [] as string[], used: 0 }];
    for (const [i, char] of chars.entries()) {
      let last = rows[rows.length - 1]!;
      const size = cells(char);
      if (last.used + size > width && last.chars.length > 0) {
        last = { start: i, chars: [], used: 0 };
        rows.push(last);
      }
      last.chars.push(char);
      last.used += size;
    }
    const caret = this.#text.caret();
    // A full last row leaves no cell for the cursor after it: it gets a row of its own.
    const last = rows[rows.length - 1]!;
    if (caret === chars.length && last.used >= width) rows.push({ start: chars.length, chars: [], used: 0 });
    const row = rows.findLastIndex((each) => each.start <= caret);
    return { rows, row, width };
  }
}

/** What a framed box's row `row` starts with: the bold `>` on the first. */
function lead(row: number): string {
  return row === 0 ? `${style("1", ">")} ` : "  ";
}

/** What a framed `readLine` answers when Enter opens a list row. */
type Opened = { open: string };

/**
 * `editor` as a prompt: once entered it collapses to `> <text>`; any other ending erases it.
 * Escape ends it only with `escape` (answering `ESCAPED`), and Enter on a list row answers
 * `{open}`.
 */
function editorWidget(
  editor: LineEditor,
  escape: boolean,
): Widget<string | typeof ESCAPED | Opened> {
  return {
    preface: "",
    lines: () => editor.lines(),
    cursor: () => editor.cursor(),
    key(key) {
      const action = editor.key(key);
      if (action === null) return undefined;
      if ("escape" in action) return escape ? { value: ESCAPED } : undefined;
      if ("open" in action) return { value: action };
      return action;
    },
    // The line entered as it was shown: its tokens, not the text they stand for.
    done: (outcome) =>
      "value" in outcome && typeof outcome.value === "string"
        ? `${style("1", ">")} ${draftLabel(editor.draft()).replaceAll("\n", "⏎")}`
        : null,
    watch: (redraw) => editor.list?.subscribe?.(redraw) ?? (() => {}),
  };
}

/** What a `readLine` prompt answers when Escape ends it. */
const ESCAPED: unique symbol = Symbol("escaped");

/** How `readLine` with `escape` ends on Escape: the text typed so far. */
type Escaped = { escape: true; draft: string };

/**
 * A picker over `titles`, answering the chosen index, so equal titles stay apart. `indicator`
 * draws `●`/`○` before each title.
 */
function selectWidget(options: {
  label: string;
  titles: readonly string[];
  initial?: number;
  indicator?: boolean;
}): Widget<number> {
  const count = options.titles.length;
  let index = options.initial ?? 0;
  const row = (title: string, i: number): string => {
    if (i !== index) return `   ${options.indicator ? "○ " : ""}${title}`;
    return ` ${style("1;36", "»")} ${style("36", `${options.indicator ? "● " : ""}${title}`)}`;
  };
  return {
    preface: labelPreface("?", options.label),
    lines: () => [promptHead("?", options.label, "(Use arrow keys)"), ...options.titles.map(row)],
    cursor: () => null,
    key(key) {
      // A paste reaches questionary's select as one paste event, which it ignores.
      if (key.paste) return undefined;
      if (key.name === "return" || key.name === "enter") return { value: index };
      // questionary binds Ctrl-Q like Ctrl-C in its pickers.
      if (key.ctrl && key.name === "q") return { stop: "interrupt" };
      index = moved(key, index, count);
      return undefined;
    },
    done: (outcome) => collapsed("?", options.label, "value" in outcome ? options.titles[outcome.value]! : ""),
  };
}

/**
 * A yes/no question: `y` or `n` answers at once (questionary's auto-enter), ENTER answers no. A
 * paste never answers it, as in questionary, where pasted text goes to a hidden buffer.
 */
function confirmWidget(label: string): Widget<boolean> {
  const line = `${promptHead("?", label, "(y/N)")} `;
  return {
    preface: labelPreface("?", label),
    lines: () => [line],
    cursor: () => ({ row: 0, before: line }),
    key(key) {
      if (key.paste) return undefined;
      if (key.name === "return" || key.name === "enter") return { value: false };
      // questionary's buffer stays empty, so Ctrl-D is always the end of input; Ctrl-Q is Ctrl-C.
      if (key.ctrl && key.name === "d") return { stop: "eof" };
      if (key.ctrl && key.name === "q") return { stop: "interrupt" };
      if (key.ctrl || key.meta) return undefined;
      if (key.str === "y" || key.str === "Y") return { value: true };
      if (key.str === "n" || key.str === "N") return { value: false };
      return undefined;
    },
    done: (outcome) => collapsed("?", label, "value" in outcome ? (outcome.value ? "Yes" : "No") : ""),
  };
}

/**
 * A multi-select over `titles`, answering the chosen indices in choice order: SPACE toggles the
 * marked row, `a` selects all (or none when all are selected), `i` inverts, ENTER submits.
 */
function checkboxWidget(options: { label: string; titles: readonly string[] }): Widget<number[]> {
  const count = options.titles.length;
  let index = 0;
  const selected = new Set<number>();
  const row = (title: string, i: number): string => {
    const pointer = i === index ? ` ${style("1;36", "»")} ` : "   ";
    if (selected.has(i)) return pointer + style("32", `● ${title}`);
    return `${pointer}○ ${i === index ? style("36", title) : title}`;
  };
  // questionary's answer line: "done", the one title in brackets, or the count.
  const summary = (picked: readonly number[]): string => {
    if (picked.length === 0) return "done";
    if (picked.length === 1) return `[${options.titles[picked[0]!]}]`;
    return `done (${picked.length} selections)`;
  };
  return {
    preface: labelPreface("?", options.label),
    lines: () => [
      promptHead("?", options.label, "(Use arrow keys to move, <space> to select, <a> to toggle, <i> to invert)"),
      ...options.titles.map(row),
    ],
    cursor: () => null,
    key(key) {
      // A paste reaches questionary's checkbox as one paste event, which it ignores.
      if (key.paste) return undefined;
      if (key.name === "return" || key.name === "enter") {
        return { value: options.titles.map((_, i) => i).filter((i) => selected.has(i)) };
      }
      if (key.ctrl && key.name === "q") return { stop: "interrupt" };
      const plain = !key.ctrl && !key.meta;
      if (plain && key.name === "space") {
        if (!selected.delete(index)) selected.add(index);
      } else if (plain && key.str === "a") {
        if (selected.size === count) selected.clear();
        else for (let i = 0; i < count; i++) selected.add(i);
      } else if (plain && key.str === "i") {
        for (let i = 0; i < count; i++) if (!selected.delete(i)) selected.add(i);
      } else index = moved(key, index, count);
      return undefined;
    },
    done: (outcome) => collapsed("?", options.label, "value" in outcome ? summary(outcome.value) : ""),
  };
}

/** One prompt turn: its signal, how it ended without an answer, and the active widget's wakeup. */
interface Turn {
  signal: AbortSignal | undefined;
  /**
   * Ctrl-C ends only this turn, and neither stop goes to `onStop`: the caller (`readLine`) gets
   * them as its answer.
   */
  keepOpen: boolean;
  /** The first way the turn ended without an answer; later ones don't replace it. */
  ending: Ending | undefined;
  /** Ends the active widget once `ending` is set; a no-op while no widget runs. */
  wake: () => void;
}

/** Runs one widget inside a turn: its answer, or a throw when it stops, fails or the turn aborts. */
type RunWidget = <T>(widget: Widget<T>) => Promise<T>;

/** Run `action`, dropping what it throws: a cleanup step or host hook never replaces an outcome. */
function quietly(action: () => unknown): void {
  try {
    action();
  } catch {}
}

/**
 * The host's terminal for a run's prompts. Prompts take turns in one FIFO: each turn waits for the
 * ones before it, takes the terminal for its widgets and gives it back restored. Ctrl-C (except at
 * `readLine`), the end of input or `close()` closes the terminal, so every queued and later prompt
 * gets no answer.
 *
 * @example
 * const terminal = new Terminal({ onStop: () => controller.abort() });
 * const name = await terminal.askText("Name:");
 */
export class Terminal {
  readonly #input: Input;
  readonly #output: Output;
  readonly #onStop: TerminalOptions["onStop"];
  readonly #onActive: TerminalOptions["onActive"];
  readonly #plain: boolean;
  #attached = false;
  #closed = false;
  /** Set once the input has ended or failed; the next widget ends with the EOF outcome. */
  #ended = false;
  /** Keys that arrived while no widget was active, fed to the next widget first. */
  #buffer: Key[] = [];
  /** The active widget's key and end handlers. */
  #active: { key: (key: Key) => void; end: () => void } | undefined;
  /** The turn holding the terminal. */
  #turn: Turn | undefined;
  /** Settles once every turn queued so far has finished. */
  #tail: Promise<void> = Promise.resolve();
  /**
   * The host's handler while `watchKeys` runs, and the raw mode to restore once nothing needs the
   * terminal raw.
   */
  #watch:
    | {
        onKey: (key: "escape" | "interrupt") => void;
        edit: ((key: Key) => boolean) | null;
        wasRaw: boolean;
      }
    | undefined;
  /** The last key was a `return`, so an `enter` now is the `\n` of the same `\r\n`. */
  #afterReturn = false;
  /** The text of the bracketed paste arriving, or null outside one. */
  #pasting: string[] | null = null;
  /** The characters `#pasting` holds. */
  #pasted = 0;
  /** Whether no key of the paste arriving came for `PASTE_IDLE_MS`. */
  #pasteStalled = false;
  /** Marks the paste arriving stalled once no key came for `PASTE_IDLE_MS`. */
  #pasteTimer: ReturnType<typeof setTimeout> | undefined;
  /** Whether the turn holding the terminal left rows on the screen. */
  #printed = false;
  /** Whether a widget has the whole (alternate) screen, which an exit must give back. */
  #onScreen = false;
  /** How many turns that want an answer (any prompt but a line prompt) wait in the queue. */
  #waiting = 0;
  /** Told each time `#waiting` changes. */
  readonly #onWaiting = new Set<() => void>();
  readonly #now: () => number;
  /** Set when a `readLine` yields: the next turn to start took its place. */
  #yielded = false;
  /** Until when keys are dropped (`#now` time), after a prompt took a yielding line's place. */
  #graceUntil = 0;

  /** Build a terminal over `options.input` and `options.output`; touches neither stream. */
  constructor(options: TerminalOptions = {}) {
    this.#input = options.input ?? process.stdin;
    this.#plain = options.plain === true;
    const output = options.output ?? process.stderr;
    this.#output = this.#plain ? plainOutput(output) : output;
    this.#onStop = options.onStop;
    this.#onActive = options.onActive;
    this.#now = options.now ?? Date.now;
  }

  /** Whether `approve` can ask: the terminal isn't plain or closed, and its input is a TTY. */
  get canApprove(): boolean {
    return !this.#plain && !this.#closed && this.#input.isTTY === true;
  }

  /** A `LineEditor` as wide as this terminal's output. */
  lineEditor(options: { hint: string; list?: PromptList | null }): LineEditor {
    const output = this.#output;
    const size = { columns: () => columnsOf(output), rows: () => rowsOf(output) };
    return new LineEditor({ ...options, ...size });
  }

  /** A free-text or hidden prompt; null when the user cancels or the terminal is closed. */
  async askText(
    label: string,
    options: { initial?: string; secret?: boolean; signal?: AbortSignal } = {},
  ): Promise<string | null> {
    const ended = await this.#runTurn(
      options.signal,
      (run) => run(textWidget({ label, initial: options.initial, mask: options.secret })),
      // A plain terminal leaves the tty cooked, which would echo the secret as it is typed.
      { raw: options.secret === true },
    );
    return "value" in ended ? ended.value : null;
  }

  /**
   * Read one line at a `> <label>` prompt, starting from `initial`: `{value}` when it is entered,
   * `{interrupt}` for Ctrl-C, `{eof}` for the end of input or a closed terminal, `{aborted}` with
   * the text typed so far (`draft`) when `signal` aborts first, which erases the prompt. With
   * `escape`, Escape ends the prompt as `{escape}` with the draft, erasing it the same way;
   * elsewhere Escape is ignored. Ctrl-C and Escape leave the terminal open, so the next
   * `readLine` works; keys typed after the prompt ends wait for the next one.
   *
   * With `frame`, on a terminal that isn't plain, the prompt is a box across the terminal holding
   * `> ` and the text, the dim `hint` below it, instead of the label; once entered it collapses
   * to `> <line>`, its pasted tokens shown as they were. Given `editor`, the box is that editor's
   * (`LineEditor`, its hint instead of `hint`), its text set to `initial` (a `Draft`, tokens and
   * all, when it holds pasted tokens) when it differs, and Enter on a row of its list answers
   * `{open}` with the row's key and the draft. An answer and a draft are text, each token
   * expanded; the editor's `draft()` keeps the tokens.
   *
   * With `yields`, the prompt gives way to a prompt that wants an answer (any but a `readLine`)
   * queued behind it while it is up: it ends as `{yielded}` with the draft, erased as an abort
   * erases it, and a `readLine` called next waits behind that prompt. The prompt that took its
   * place drops the keys typed before it opened, and those of its first `YIELD_GRACE_MS` but
   * Ctrl-C.
   */
  async readLine<E extends boolean = false>(
    label: string,
    options: {
      signal?: AbortSignal;
      initial?: string | Draft;
      escape?: E;
      frame?: boolean;
      hint?: string;
      editor?: LineEditor;
      yields?: boolean;
    } = {},
  ): Promise<
    | { value: string }
    | { eof: true }
    | { interrupt: true }
    | { aborted: true; draft: string }
    | { yielded: true; draft: string }
    | { open: string; draft: string }
    | (E extends true ? Escaped : never)
  > {
    const { initial = "" } = options;
    if (options.yields !== true) return this.#readLine(label, options);
    // Only prompts queued while this one is up are behind it; one counted before may be ahead.
    let up = false;
    const yielding = new AbortController();
    const onWaiting = () => {
      if (up && this.#waiting > 0) yielding.abort(new Error("a prompt waits behind the line"));
    };
    this.#onWaiting.add(onWaiting);
    const signal =
      options.signal === undefined
        ? yielding.signal
        : AbortSignal.any([options.signal, yielding.signal]);
    try {
      const read = await this.#readLine(label, { ...options, signal, initial }, () => {
        up = true;
        onWaiting();
      });
      if ("aborted" in read && yielding.signal.aborted && options.signal?.aborted !== true) {
        this.#yielded = true;
        return { yielded: true, draft: read.draft };
      }
      return read;
    } finally {
      this.#onWaiting.delete(onWaiting);
    }
  }

  /** `readLine` without `yields`; `onUp` is called once the prompt holds the terminal. */
  async #readLine<E extends boolean = false>(
    label: string,
    options: {
      signal?: AbortSignal;
      initial?: string | Draft;
      escape?: E;
      frame?: boolean;
      hint?: string;
      editor?: LineEditor;
    },
    onUp?: () => void,
  ): Promise<
    | { value: string }
    | { eof: true }
    | { interrupt: true }
    | { aborted: true; draft: string }
    | { open: string; draft: string }
    | (E extends true ? Escaped : never)
  > {
    const { signal, initial = "" } = options;
    let widget: Widget<string | typeof ESCAPED | Opened>;
    let draft: () => string;
    if (options.frame === true && !this.#plain) {
      const columns = () => columnsOf(this.#output);
      const editor = options.editor ?? new LineEditor({ columns, hint: options.hint ?? "" });
      if (!sameUnits(unitsOf(editor.draft()), unitsOf(initial))) editor.set(initial);
      widget = editorWidget(editor, options.escape === true);
      draft = () => editor.text();
    } else {
      const text = textWidget({ label, mark: ">", initial: textOf(initial) });
      widget = options.escape
        ? {
            ...text,
            key: (key) => (key.name === "escape" ? { value: ESCAPED } : text.key(key)),
            done: (outcome) =>
              "value" in outcome && outcome.value === ESCAPED
                ? null
                : text.done(outcome as Outcome<string>),
          }
        : text;
      draft = () => text.text();
    }
    if (signal?.aborted) return { aborted: true, draft: textOf(initial) };
    let ended: { value: string | typeof ESCAPED | Opened } | { stop: StopReason };
    try {
      ended = await this.#runTurn(
        signal,
        (run) => {
          onUp?.();
          return run(widget);
        },
        { keepOpen: true, wants: false },
      );
    } catch (error) {
      if (signal?.aborted && error === signal.reason) return { aborted: true, draft: draft() };
      throw error;
    }
    if ("value" in ended) {
      const { value } = ended;
      // Only an `escape` prompt answers ESCAPED, so E is true here.
      if (value === ESCAPED) {
        return { escape: true, draft: draft() } as E extends true ? Escaped : never;
      }
      if (typeof value !== "string") return { open: value.open, draft: draft() };
      return { value };
    }
    return ended.stop === "interrupt" ? { interrupt: true } : { eof: true };
  }

  /**
   * A single choice among strings; null when cancelled or closed. Each choice is drawn through
   * `visible` (a provider's model ids are choices) and answered as given.
   */
  async askChoice(
    label: string,
    choices: readonly string[],
    options: { initial?: string; signal?: AbortSignal } = {},
  ): Promise<string | null> {
    // questionary's select refuses both too.
    if (choices.length === 0) throw new RangeError("askChoice needs at least one choice");
    if (options.initial !== undefined && !choices.includes(options.initial)) {
      throw new RangeError(`askChoice's initial ${JSON.stringify(options.initial)} is not one of the choices`);
    }
    const initial = options.initial === undefined ? 0 : choices.indexOf(options.initial);
    const ended = await this.#runTurn(options.signal, (run) => run(selectWidget({ label, titles: choices.map(visible), initial })));
    return "value" in ended ? choices[ended.value]! : null;
  }

  /**
   * Show `draw`'s rows on the whole screen, the alternate one, until Escape, `q`, Enter or Ctrl-C,
   * or `signal` aborts; then give back the screen as it was and call `onClose`, before any prompt
   * queued behind takes the terminal. `draw` gets the columns, the rows and whether a prompt that
   * wants an answer (any but a `readLine`) waits behind the screen, and is called again on each
   * `subscribe` callback and each time that changes; rows past the screen's height are left out
   * and each is cut to the width. Does nothing on a plain terminal or an input that isn't a TTY,
   * and when the terminal is closed. Ctrl-C leaves the terminal open.
   */
  async showScreen(options: {
    draw: (columns: number, rows: number, waiting: boolean) => readonly string[];
    subscribe?: (redraw: () => void) => () => void;
    onClose?: () => void;
    signal?: AbortSignal;
  }): Promise<void> {
    if (this.#plain || this.#input.isTTY !== true || options.signal?.aborted) return;
    const output = this.#output;
    const widget: Widget<null> = {
      preface: "",
      screen: true,
      lines: () => [...options.draw(columnsOf(output), rowsOf(output), this.#waiting > 0)],
      cursor: () => null,
      key(key) {
        if (key.paste) return undefined;
        const plain = !key.ctrl && !key.meta;
        const leave = key.name === "return" || key.name === "enter" || key.str === "q";
        // A lone Escape can arrive marked meta.
        return key.name === "escape" || (plain && leave) ? { value: null } : undefined;
      },
      done: () => null,
      watch: (redraw) => {
        const stop = options.subscribe?.(redraw);
        this.#onWaiting.add(redraw);
        return () => {
          stop?.();
          this.#onWaiting.delete(redraw);
        };
      },
      closed: options.onClose,
    };
    try {
      await this.#runTurn(options.signal, (run) => run(widget), { keepOpen: true });
    } catch (error) {
      if (!(options.signal?.aborted && error === options.signal.reason)) throw error;
    }
  }

  /**
   * Ask the user to approve an action: write `body` and the unified `diff` when there is one (with
   * `frame`, as a card: `diffCard`), then select among `choices` under `title`, `preselect` (or
   * the first) preselected. All the text is drawn through
   * `sanitize`, its escape sequences shown rather than dropped. Returns the chosen choice; the
   * last one (the reject) for Ctrl-C, which leaves the terminal open; null without drawing or
   * reading when the terminal is plain, closed or its input isn't a TTY, and null when the input
   * ends or `signal` aborts.
   */
  async approve(request: ApproveRequest): Promise<string | null> {
    const { choices, signal } = request;
    if (choices.length === 0) throw new RangeError("approve needs at least one choice");
    // Only a person at a terminal can approve: piped input could approve on their behalf.
    if (this.#plain || this.#input.isTTY !== true || signal?.aborted) return null;
    // A sequence is shown, not dropped: what the user approves must not hide anything it holds.
    const shown = (text: string): string => sanitize(text, { sequences: "show" });
    const label = shown(request.title);
    const card = request.frame
      ? [
          diffCard(
            label,
            shown(request.body).replaceAll("\t", "    "),
            shown(request.diff ?? "").replaceAll("\t", "    "),
            columnsOf(this.#output),
          ).join("\n"),
        ]
      : [request.body, request.diff ?? ""].filter((text) => text !== "").map(shown);
    const titles = choices.map(shown);
    if (request.preselect !== undefined && !choices.includes(request.preselect)) {
      const named = JSON.stringify(request.preselect);
      throw new RangeError(`approve's preselect ${named} is not one of the choices`);
    }
    const initial = request.preselect === undefined ? 0 : choices.indexOf(request.preselect);
    let ended: { value: number } | { stop: StopReason };
    try {
      ended = await this.#runTurn(
        signal,
        (run) => {
          for (const text of card) this.#output.write(crlf(text.replace(/\n$/, "")) + "\r\n");
          return run(selectWidget({ label, titles, initial }));
        },
        { keepOpen: true },
      );
    } catch (error) {
      if (signal?.aborted && error === signal.reason) return null;
      throw error;
    }
    if ("value" in ended) return choices[ended.value]!;
    return ended.stop === "interrupt" ? choices[choices.length - 1]! : null;
  }

  /**
   * Prompt for each declaration not in `supplied` (key presence, `Object.hasOwn`), all in one turn;
   * an inbox input is never asked for.
   * Returns the explicit answers (omitted inputs absent, Set null a present null), `{}` without
   * drawing when nothing is missing, or null when cancelled or the terminal is closed. Throws what
   * a widget or the output throws.
   */
  async promptInputs(
    decls: readonly Parameter[],
    supplied: Readonly<Record<string, unknown>>,
  ): Promise<Record<string, unknown> | null> {
    // An inbox takes events through `send`, never a value at the start.
    const missing = decls.filter(
      (decl) => !isInboxType(decl.contract?.type ?? null) && !Object.hasOwn(supplied, decl.name),
    );
    if (missing.length === 0) return {};
    const ended = await this.#runTurn(undefined, async (run) => {
      const gathered: [string, unknown][] = [];
      for (const decl of missing) {
        const value = await promptInput(run, decl);
        if (value === OMIT_INPUT) continue;
        gathered.push([decl.name, value === NULL_INPUT ? null : value]);
      }
      // `fromEntries` defines each key as data, so an input named `__proto__` stays an entry.
      return Object.fromEntries(gathered);
    });
    return "value" in ended ? ended.value : null;
  }

  /**
   * The run's ask handler (an `AskUserHandler` from tools/registry.ts): asks `request` in one
   * turn and returns the answer. A question set (`questions`, an `ask` node's
   * `QuestionSpec`s) answers a record keyed by header; `options` a single choice; otherwise a
   * free-text prompt. `answerSchema` isn't used.
   *
   * @throws `ctx.signal.reason` when the signal aborts first, or when Ctrl-C or a closed terminal
   *   ends the call after the signal aborted (an `onStop` that aborts it, say).
   * @throws TerminalStopped `"interrupt"` for Ctrl-C, `"eof"` for the end of input (even when the
   *   signal then aborted), `"closed"` when the terminal is closed.
   */
  // A field, so a host can pass `terminal.askUser` unbound as the `ask` handler.
  readonly askUser = async (request: AskUserRequest, ctx: ToolContext): Promise<unknown> => {
    const signal = ctx.signal;
    const ended = await this.#runTurn(signal, (run) => promptRequest(run, this.#output, request));
    if ("value" in ended) return ended.value;
    if (ended.stop !== "eof" && signal.aborted) throw signal.reason;
    throw new TerminalStopped(ended.stop);
  };

  /**
   * Close the terminal: the active prompt ends, queued and later prompts are refused, listeners
   * detach, input pauses.
   */
  close(): void {
    this.#closed = true;
    const turn = this.#turn;
    if (turn !== undefined) {
      // The turn's own cleanup detaches and pauses, outside any key dispatch.
      turn.ending ??= "closed";
      turn.wake();
    } else if (this.#attached) {
      this.#detach();
      quietly(() => this.#input.pause());
    }
  }

  /**
   * Watch the keys while no prompt is up, until `signal` aborts: Escape calls `onKey("escape")`
   * and Ctrl-C `onKey("interrupt")`, rather than raising SIGINT, since the input is raw; other keys
   * wait for the next prompt. Given `edit`, every key but Ctrl-C goes to it first, and only one it
   * doesn't take (returns false for) goes on as above; bracketed paste is then on while the watch
   * runs, so a paste reaches it as one key holding the text, its newlines kept. A prompt meanwhile
   * takes the keys as usual, and the watch goes on after it. Does nothing on a plain terminal or an
   * input that isn't a TTY.
   */
  watchKeys(
    signal: AbortSignal,
    onKey: (key: "escape" | "interrupt") => void,
    edit: ((key: Key) => boolean) | null = null,
  ): void {
    const input = this.#input;
    if (this.#plain || this.#closed || signal.aborted) return;
    if (input.isTTY !== true || typeof input.setRawMode !== "function") return;
    this.#attach();
    const watch = { onKey, edit, wasRaw: input.isRaw === true };
    this.#watch = watch;
    const output = this.#output;
    // A `process.exit()` elsewhere skips the restore below.
    const onExit = (): void => {
      quietly(() => input.setRawMode!(false));
      if (edit !== null) quietly(() => output.write(PASTE_OFF));
    };
    process.once("exit", onExit);
    quietly(() => input.setRawMode!(true));
    if (edit !== null && this.#turn === undefined) quietly(() => output.write(PASTE_ON));
    input.resume();
    const end = (): void => {
      if (this.#watch === watch) this.#watch = undefined;
      quietly(() => process.removeListener("exit", onExit));
      // A turn holding the terminal restores it when it ends.
      if (this.#turn !== undefined) return;
      quietly(() => input.setRawMode!(watch.wasRaw));
      if (edit !== null) quietly(() => output.write(PASTE_OFF));
      quietly(() => input.pause());
    };
    signal.addEventListener("abort", end, { once: true });
  }

  /** Normalize one keypress and hand it to the active widget, or buffer it for the next one. */
  readonly #onKeypress = (
    str: string | undefined,
    raw: { name?: string; ctrl?: boolean; meta?: boolean; sequence?: string } | undefined,
  ): void => {
    const name = raw?.name;
    // A key typed soon after a lone Escape arrives as Alt+key. No prompt binds Alt, so the Escape
    // is dropped and the key stays the one typed.
    const alt = raw?.meta === true ? /^\x1b([^\x00-\x1f\x7f])$/su.exec(raw.sequence ?? "") : null;
    if (name === "paste-start") {
      this.#pasting ??= [];
      this.#pasteIdle();
      return;
    }
    // A paste whose end never came (a terminal dropped it) would keep every later key, Enter and
    // Ctrl-C too. Once it has paused, Escape ends it as it is and Ctrl-C drops it; a pause alone
    // doesn't end it, since a slow link can pause one that is still coming.
    const hatch = name === "escape" || (raw?.ctrl === true && name === "c");
    if (this.#pasting !== null && this.#pasteStalled && hatch) {
      return this.#endPaste(name === "escape");
    }
    if (name === "enter" && this.#afterReturn) {
      this.#afterReturn = false;
      return;
    }
    this.#afterReturn = name === "return";
    // A paste reaches the widget as one key holding its text, so a long one can become a token:
    // its newlines kept, the keys it holds (arrows, Escape, Ctrl-C) only text.
    if (this.#pasting !== null && name !== "paste-end") {
      const text = name === "return" || name === "enter" ? "\n" : (str ?? "");
      if (this.#pasted + text.length <= PASTE_MAX_CHARS) {
        this.#pasting.push(text);
        this.#pasted += text.length;
      }
      this.#pasteIdle();
      return;
    }
    if (name === "paste-end") return this.#endPaste(true);
    const meta = raw?.meta === true && alt === null;
    this.#dispatch({ name, ctrl: raw?.ctrl === true, meta, str: alt?.[1] ?? str, paste: false });
  };

  /** A key of the paste arrived: it hasn't paused, and its pause starts again. */
  #pasteIdle(): void {
    this.#pasteStalled = false;
    clearTimeout(this.#pasteTimer);
    this.#pasteTimer = setTimeout(() => (this.#pasteStalled = true), PASTE_IDLE_MS);
    // The pause only arms Escape and Ctrl-C, so it never keeps the process up.
    this.#pasteTimer.unref?.();
  }

  /** End the paste arriving: hand its text on as one paste key, or drop it unless `keep`. */
  #endPaste(keep: boolean): void {
    clearTimeout(this.#pasteTimer);
    const text = this.#pasting?.join("") ?? "";
    this.#pasting = null;
    this.#pasted = 0;
    this.#pasteStalled = false;
    if (!keep || text === "") return;
    this.#dispatch({ name: undefined, ctrl: false, meta: false, str: text, paste: true });
  }

  /** Hand `key` to the active widget or the watch, or buffer it for the next widget. */
  #dispatch(key: Key): void {
    const watch = this.#turn === undefined ? this.#watch : undefined;
    // Ctrl-C still works in the grace after a prompt took a line's place.
    const graced = this.#turn !== undefined && this.#now() < this.#graceUntil;
    if (graced && !(key.ctrl && key.name === "c")) return;
    if (this.#active) this.#active.key(key);
    else if (watch !== undefined && key.ctrl && key.name === "c") watch.onKey("interrupt");
    else if (watch?.edit?.(key) === true) return;
    else if (watch !== undefined && key.name === "escape") watch.onKey("escape");
    else this.#buffer.push(key);
  }

  /** The input ended or failed (EIO when the terminal goes away): the active or next widget ends. */
  readonly #onEnd = (): void => {
    this.#ended = true;
    this.#active?.end();
  };

  /** Start decoding keys and listen to the input, once for the terminal's life. */
  #attach(): void {
    if (!this.#attached) {
      this.#attached = true;
      emitKeypressEvents(this.#input);
      this.#input.on("keypress", this.#onKeypress);
      this.#input.on("end", this.#onEnd);
      this.#input.on("error", this.#onEnd);
    }
    // A stream that already emitted `end` never emits it again.
    if (this.#input.readableEnded === true || this.#input.destroyed === true) this.#ended = true;
  }

  /**
   * Stop listening to the input and drop buffered keys. `emitKeypressEvents` keeps its own `data`
   * listener until the next chunk arrives, on both runtimes, so only our listeners go.
   */
  #detach(): void {
    if (!this.#attached) return;
    this.#attached = false;
    this.#input.removeListener("keypress", this.#onKeypress);
    this.#input.removeListener("end", this.#onEnd);
    this.#input.removeListener("error", this.#onEnd);
    this.#buffer = [];
    this.#endPaste(false);
  }

  /**
   * Run `fn` as one turn of the FIFO: wait for the turns before it, take the terminal (raw mode,
   * listeners, input flowing), let `fn` run widgets, then restore the terminal on every outcome.
   * A plain terminal sets raw mode only with `raw`. While it waits behind another, it counts as
   * wanting an answer (`showScreen` says so) unless `wants` is false.
   * Returns `fn`'s value, or the stop that ended the turn (Ctrl-C, end of input, a closed
   * terminal); throws `signal`'s reason when it aborted first, else `fn`'s error.
   */
  async #runTurn<R>(
    signal: AbortSignal | undefined,
    fn: (run: RunWidget) => Promise<R>,
    options: { keepOpen?: boolean; raw?: boolean; wants?: boolean } = {},
  ): Promise<{ value: R } | { stop: StopReason }> {
    const { keepOpen = false, raw = false, wants = true } = options;
    if (this.#closed) return { stop: "closed" };
    signal?.throwIfAborted();
    const ahead = this.#tail;
    const { promise: finished, resolve: release } = Promise.withResolvers<void>();
    // The turn behind this one waits for the one ahead too, even when this one leaves early.
    this.#tail = ahead.then(() => finished);
    const turn: Turn = { signal, keepOpen, ending: undefined, wake: () => {} };
    const { promise: queuedAbort, resolve: leaveQueue } = Promise.withResolvers<void>();
    const onAbort = (): void => {
      // A queued turn whose signal aborts after the terminal closed is refused as closed.
      turn.ending ??= this.#closed ? "closed" : "aborted";
      leaveQueue();
      turn.wake();
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    const input = this.#input;
    const output = this.#output;
    const setsRaw = (!this.#plain || raw) && input.isTTY === true && typeof input.setRawMode === "function";
    // Read at turn start: while queued, the turn ahead may still hold the terminal raw.
    let wasRaw = false;
    // A `process.exit()` elsewhere skips the `finally` below.
    const onExit = (): void => {
      if (setsRaw) quietly(() => input.setRawMode!(false));
      quietly(() => output.write(SHOW_CURSOR + PASTE_OFF + (this.#onScreen ? ALT_OFF : "")));
    };
    let started = false;
    let result: { value: R } | { error: unknown } | undefined;
    try {
      // Counted while it waits behind another turn, so a screen up can say a prompt waits.
      const counted = wants && this.#turn !== undefined;
      if (counted) this.#countWaiting(1);
      try {
        await Promise.race([ahead, queuedAbort]);
      } finally {
        if (counted) this.#countWaiting(-1);
      }
      if (turn.ending === undefined && this.#closed) turn.ending = "closed";
      if (turn.ending === undefined) {
        started = true;
        this.#turn = turn;
        if (this.#yielded) {
          // The user may have been typing in the line: what they typed for it answers nothing here.
          this.#yielded = false;
          this.#buffer = [];
          this.#graceUntil = this.#now() + YIELD_GRACE_MS;
        }
        this.#printed = false;
        quietly(() => this.#onActive?.(true, false));
        this.#attach();
        // Under a watch the terminal is raw for it; the mode before the watch is the one to restore.
        wasRaw = this.#watch?.wasRaw ?? input.isRaw === true;
        // Even without raw mode the widgets hide the cursor and turn bracketed paste on.
        process.once("exit", onExit);
        if (setsRaw) input.setRawMode!(true);
        input.resume();
        result = { value: await fn((widget) => this.#runWidget(widget, turn)) };
      }
    } catch (error) {
      result = { error };
    } finally {
      signal?.removeEventListener("abort", onAbort);
      if (started) {
        this.#turn = undefined;
        // Each step runs on its own, cooked mode first, so a failing one never skips the rest.
        // Pausing here, in a promise continuation rather than inside the key dispatch, keeps a
        // piped stdin from being restarted by its read-ahead.
        // A watch still running keeps the terminal raw and the input flowing, and restores `wasRaw`
        // itself when it ends.
        const watch = this.#watch;
        if (watch !== undefined) watch.wasRaw = wasRaw;
        if (setsRaw) quietly(() => input.setRawMode!(watch !== undefined || wasRaw));
        // A watch with an editor still takes pastes.
        quietly(() => output.write(SHOW_CURSOR + (watch?.edit ? PASTE_ON : PASTE_OFF)));
        quietly(() => process.removeListener("exit", onExit));
        if (watch === undefined) quietly(() => input.pause());
        if (this.#closed) quietly(() => this.#detach());
        quietly(() => this.#onActive?.(false, this.#printed));
      }
    }
    const ending = turn.ending;
    // After the restore, so a handler that prints or exits never meets a raw terminal; before the
    // turns behind are released, so they see what `onStop` did.
    // Only the turn that closed the terminal records "interrupt" or "eof", so this runs once; a
    // `keepOpen` turn hands its stop to its caller instead.
    if ((ending === "interrupt" || ending === "eof") && !keepOpen) quietly(() => this.#onStop?.(ending));
    release();
    if (ending !== undefined && ending !== "aborted") return { stop: ending };
    if (signal?.aborted) throw signal.reason;
    if (result !== undefined && "value" in result) return result;
    throw result!.error;
  }

  /** Change the count of turns that wait for an answer by `by`, and tell the listeners. */
  #countWaiting(by: number): void {
    this.#waiting += by;
    for (const listener of [...this.#onWaiting]) quietly(listener);
  }

  /**
   * Run `widget` inside `turn` until it ends: draw it, feed it keys, collapse it. Ctrl-C or the end
   * of input records the turn's stop and closes the terminal; the turn restores everything after.
   */
  async #runWidget<T>(widget: Widget<T>, turn: Turn): Promise<T> {
    const stopped = (): unknown =>
      turn.ending === "aborted" ? turn.signal!.reason : new TerminalStopped(turn.ending!);
    if (turn.ending !== undefined) throw stopped();
    const output = this.#output;
    let outcome: Outcome<T> | undefined;
    const { promise: settled, resolve: settle } = Promise.withResolvers<void>();
    // The first outcome wins: a later abort, key or error can't replace it.
    const finish = (ended: Outcome<T>): void => {
      if (outcome !== undefined) return;
      outcome = ended;
      this.#active = undefined;
      settle();
    };
    const stop = (reason: "interrupt" | "eof"): void => {
      if (turn.ending === undefined) {
        turn.ending = reason;
        // A chat counts Ctrl-C presses at its line prompt, so the first one mustn't end the session.
        if (!(turn.keepOpen && reason === "interrupt")) this.#closed = true;
      }
      finish({ stop: turn.ending });
    };
    turn.wake = () => finish({ stop: turn.ending! });
    // Rows from the frame's top to where the cursor was left, so the next draw starts there.
    let cursorRow = 0;
    const erase = (): string => `\r${cursorRow > 0 ? `${ESC}${cursorRow}A` : ""}${ESC}J`;
    // Whether the cursor is shown: a widget can hide it for a while (a list with the keys).
    let shown = widget.cursor() !== null;
    const draw = (): void => {
      // A plain prompt is written once, as it opens; a refused line adds only its error, since
      // the cooked tty already echoed the line.
      if (this.#plain) {
        const refused = widget.refused?.();
        if (refused !== undefined) output.write(`${refused}\n`);
        return;
      }
      const columns = columnsOf(output);
      const lines = widget.lines();
      if (widget.screen) {
        // From the top, one screen row each, so the screen never scrolls.
        const rows = lines.slice(0, rowsOf(output)).map((line) => clip(line, columns));
        output.write(`${ESC}H${rows.map((row) => `${row}${ESC}K`).join("\r\n")}${ESC}J`);
        return;
      }
      const cursor = widget.cursor();
      let modes = "";
      if ((cursor !== null) !== shown) {
        shown = cursor !== null;
        modes = shown ? SHOW_CURSOR : HIDE_CURSOR;
      }
      const at = cursor === null ? -1 : Array.from(cursor.before.replace(ANSI, "")).length;
      const laid = lines.map((line, i) => layout(line, columns, i === cursor?.row ? at : -1));
      const total = laid.reduce((sum, line) => sum + line.rows, 0);
      let text = modes + erase() + crlf(lines.join("\n"));
      cursorRow = total - 1;
      if (cursor !== null) {
        const place = laid[cursor.row]!.cursor;
        const target = laid.slice(0, cursor.row).reduce((sum, line) => sum + line.rows, 0) + place.row;
        if (cursorRow > target) text += `${ESC}${cursorRow - target}A`;
        text += `\r${ESC}${place.col + 1}G`;
        cursorRow = target;
      }
      output.write(text);
    };
    const feed = (key: Key): void => {
      try {
        if (key.ctrl && key.name === "c") return stop("interrupt");
        const ended = widget.key(key);
        if (ended !== undefined) return "value" in ended ? finish(ended) : stop(ended.stop);
        draw();
      } catch (error) {
        finish({ error });
      }
    };
    let unwatch = (): void => {};
    try {
      if (this.#plain) output.write(`${widget.preface}${widget.lines().join("\n")}\n`);
      else {
        const screen = widget.screen ? ALT_ON : "";
        this.#onScreen = widget.screen === true;
        output.write(screen + widget.preface + PASTE_ON + (shown ? SHOW_CURSOR : HIDE_CURSOR));
      }
      draw();
      this.#active = { key: feed, end: () => stop("eof") };
      unwatch = widget.watch?.(() => {
        if (outcome !== undefined) return;
        try {
          draw();
        } catch (error) {
          finish({ error });
        }
      }) ?? unwatch;
      while (outcome === undefined && this.#buffer.length > 0) feed(this.#buffer.shift()!);
      if (outcome === undefined && this.#ended) stop("eof");
      await settled;
      const ended = outcome!;
      // A screen widget leaves no line: the screen it had is given back (below) as it was.
      if (!widget.screen) {
        // A `readLine` a deadline took down leaves no line: the prompt comes back after the reply.
        const vanish = turn.keepOpen && "stop" in ended && ended.stop === "aborted";
        const line = vanish ? null : widget.done(ended);
        if (line === null) output.write(erase());
        else if (!this.#plain) {
          output.write(erase() + crlf(line) + "\r\n");
          this.#printed = true;
        }
      }
    } catch (error) {
      finish({ error });
    } finally {
      this.#active = undefined;
      quietly(unwatch);
      if (widget.screen && !this.#plain) quietly(() => output.write(ALT_OFF));
      if (widget.screen) this.#onScreen = false;
      if (widget.closed) quietly(widget.closed);
    }
    const ended = outcome!;
    if ("value" in ended) return ended.value;
    if ("error" in ended) throw ended.error;
    throw stopped();
  }
}

/**
 * An example for the scalars whose accepted text isn't evident from the type name: a date or
 * datetime (a bare date is not a valid datetime).
 */
const FORMAT_EXAMPLE: ReadonlyMap<ValueKind, string> = new Map([
  [ValueKind.DATE, "2026-05-21"],
  [ValueKind.DATETIME, "2026-05-21T14:30"],
]);

/**
 * The prompt label for one declared input: its name, canonical type, `*` / `[optional]` /
 * `[default: <text form>]`, and a date/datetime example when there is no default.
 *
 * @example
 * inputLabel(asOf); // "as_of (Optional(date)) [optional] e.g. 2026-05-21"
 */
export function inputLabel(decl: Parameter): string {
  const parts = [decl.name];
  const type = decl.contract?.type;
  if (type !== undefined) parts.push(`(${canonicalTypeLabel(type)})`);
  // Keyed off the kind, so `Optional(date)` gets the example too.
  const example = type === undefined ? undefined : FORMAT_EXAMPLE.get(type.kind);
  const hint = example === undefined ? [] : [`e.g. ${example}`];
  if (decl.required) parts.push("*", ...hint);
  // A default is a yian value (an int is a bigint); it already shows the format, so no example.
  else if (decl.hasDefault) parts.push(`[default: ${writeAsText(decl.default as Value)}]`);
  else parts.push("[optional]", ...hint);
  return parts.join(" ");
}

/** A user mistake in a list/object editor answer, shown in the editor for correction. */
class StructuredInputError extends RangeError {
  override name = "StructuredInputError";
}

/**
 * Parse and recursively validate one list/object editor answer, read as yian's codec JSON.
 * The root shape is checked before the type, so even a nullable declaration refuses a textual
 * `null`: an explicit null is its own prompt action.
 *
 * @returns The native JSON array or object.
 * @throws StructuredInputError (a RangeError) for invalid JSON, a value nesting too deep, the wrong
 *   root shape or a value the type refuses (`$<path>: <reason>`).
 * @throws TypeError when `type` is neither a list nor an object.
 */
export function parseStructuredInput(raw: string, type: Type, name: string): JsonValue {
  let parsed: JsonValue;
  try {
    parsed = parseJson(raw, "codec");
    rejectDeepNesting(parsed, `input ${reprText(name)}`);
  } catch (error) {
    if (error instanceof JSONDecodeError) {
      throw new StructuredInputError(`Invalid JSON at line ${error.line}, column ${error.column}: ${error.reason}`);
    }
    if (error instanceof TypeCheckError) throw new StructuredInputError(error.message);
    throw error;
  }
  if (isListKind(type.kind)) {
    if (!Array.isArray(parsed)) throw new StructuredInputError("Expected a JSON array");
  } else if (type.kind === ValueKind.OBJECT) {
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new StructuredInputError("Expected a JSON object");
    }
  } else {
    throw new TypeError("structured prompt type must be a list or object");
  }
  try {
    buildValueAs(type, parsed);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    const path = error.path
      .map((segment) => (typeof segment === "number" ? `[${segment}]` : `[${JSON.stringify(segment)}]`))
      .join("");
    throw new StructuredInputError(`$${path}: ${error.message}`);
  }
  return parsed;
}

/** The prompt controls: leave the input absent, give it a present null, open its editor. */
const OMIT_INPUT = Symbol("omit input");
const NULL_INPUT = Symbol("set null");
const ENTER_INPUT = Symbol("enter value");

// Unicode whitespace: a text of only these characters is blank.
const BLANK_TEXT = /^[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u;

/** One select choice: its title and the value choosing it gives. */
interface Choice {
  title: string;
  value: unknown;
}

/** Whether a resolved input contract accepts an explicit null. */
function acceptsNull(type: Type | undefined): boolean {
  return type !== undefined && (type.nullable || type.kind === ValueKind.NONE);
}

/**
 * The omission/null/enter choices `decl` allows and the safe one to preselect: omission for a
 * non-required input, else Enter when `withEnter`, else none.
 */
function controlChoices(
  decl: Parameter,
  type: Type | undefined,
  withEnter = false,
): { choices: Choice[]; initial: number | undefined } {
  const choices: Choice[] = [];
  let initial: number | undefined;
  if (!decl.required) {
    choices.push({ title: decl.hasDefault ? "↩ Use declared default" : "↩ Omit input", value: OMIT_INPUT });
    initial = 0;
  }
  if (acceptsNull(type)) choices.push({ title: "Set null", value: NULL_INPUT });
  if (withEnter) {
    choices.push({ title: "Enter value", value: ENTER_INPUT });
    initial ??= choices.length - 1;
  }
  return { choices, initial };
}

/**
 * Run a select over `choices` (no indicator), preselecting `initial` (the first choice when
 * undefined), and return the chosen one's value.
 */
async function choose(run: RunWidget, label: string, choices: readonly Choice[], initial = 0): Promise<unknown> {
  const index = await run(selectWidget({ label, titles: choices.map((choice) => choice.title), initial }));
  return choices[index]!.value;
}

/**
 * Run the text editor for one open input. A list/object editor validates JSON in place and returns
 * the native value; a scalar editor checks its text as START would and returns it raw, coerced
 * later at the run boundary. An optional blank answer is `OMIT_INPUT`.
 */
async function promptOpenValue(
  run: RunWidget,
  decl: Parameter,
  type: Type | undefined,
  label: string,
): Promise<unknown> {
  const blankOmits = (text: string): boolean => !decl.required && BLANK_TEXT.test(text);
  if (type !== undefined && (isListKind(type.kind) || type.kind === ValueKind.OBJECT)) {
    const value = await run(
      textWidget({
        label,
        instruction: isListKind(type.kind) ? "Enter a JSON array" : "Enter a JSON object",
        validate: (text) => {
          if (blankOmits(text)) return undefined;
          try {
            parseStructuredInput(text, type, decl.name);
          } catch (error) {
            if (error instanceof StructuredInputError) return error.message;
            throw error;
          }
          return undefined;
        },
      }),
    );
    return blankOmits(value) ? OMIT_INPUT : parseStructuredInput(value, type, decl.name);
  }
  const value = await run(
    textWidget({
      label,
      validate: (text) => {
        if (blankOmits(text) || type === undefined) return undefined;
        // START's own check: the input's string coercion, then its contract.
        try {
          buildValueAs(type, coerceParam(decl, text));
        } catch (error) {
          if (error instanceof TypeCheckError) return error.message;
          throw error;
        }
        return undefined;
      },
    }),
  );
  return blankOmits(value) ? OMIT_INPUT : value;
}

/**
 * Ask for one declared input: its value, `OMIT_INPUT` or `NULL_INPUT`; a stop throws through
 * `run`. A required non-nullable bool confirms; other bools and Literals select among the allowed
 * controls and their values; a null-capable open input first selects omit, null or Enter, and an
 * optional blank in its editor returns to the palette; other inputs open their editor directly.
 */
async function promptInput(run: RunWidget, decl: Parameter): Promise<unknown> {
  const label = inputLabel(decl);
  const type = decl.contract?.type;

  if (type?.kind === ValueKind.BOOLEAN) {
    if (decl.required && !acceptsNull(type)) return run(confirmWidget(label));
    const { choices: controls, initial } = controlChoices(decl, type);
    const values = [
      { title: "true", value: true },
      { title: "false", value: false },
    ];
    return choose(run, label, [...controls, ...values], decl.required ? controls.length + 1 : initial);
  }

  if (type?.tags !== undefined && type.tags.size > 0) {
    const nullCapable = acceptsNull(type);
    const values = Array.from(type.tags)
      .sort(compareCodePoints)
      .map((tag) => ({ title: decl.required && !nullCapable ? tag : `Set value: ${tag}`, value: tag }));
    const { choices: controls, initial } = controlChoices(decl, type);
    return choose(run, label, [...controls, ...values], decl.required && nullCapable ? controls.length : initial);
  }

  if (type?.kind === ValueKind.NONE) {
    const { choices: controls, initial } = controlChoices(decl, type);
    return choose(run, label, controls, initial);
  }

  if (acceptsNull(type)) {
    const { choices: controls, initial } = controlChoices(decl, type, true);
    while (true) {
      const action = await choose(run, label, controls, initial);
      if (action !== ENTER_INPUT) return action;
      const value = await promptOpenValue(run, decl, type, label);
      if (value !== OMIT_INPUT) return value;
    }
  }

  return promptOpenValue(run, decl, type, label);
}

const OTHER = "✎ Other (write your own)";

/**
 * `text` safe to draw, through `sanitize`: escape sequences are shown rather than dropped (`^[`
 * for ESC), a tab as `^I`, and BIDI and zero-width characters kept. An ask's text comes from the
 * flow or a model, and a raw escape sequence in it could move the cursor, set the clipboard or
 * forge a link.
 */
function visible(text: string): string {
  return sanitize(text, { sequences: "show", tabs: "show", invisible: "keep" });
}

/**
 * Write the box that opens a question set: who needs the input (`title`), the prompt when there is
 * one, and how many questions follow. Rich's `Panel` look, with no wrapping.
 */
function writePanel(output: Output, title: string, prompt: string, count: number): void {
  const body = [
    style("1", "Your input is needed for ") + style("1;36", title),
    ...(prompt ? prompt.split("\n") : []),
    "",
    style("2", `${count} ${count === 1 ? "question" : "questions"} to answer.`),
  ];
  const head = " human input ";
  const headWidth = cells(head);
  const inner = Math.max(headWidth, ...body.map(cells)) + 2;
  const left = Math.floor((inner - headWidth) / 2);
  const lines = [
    style("36", `╭${"─".repeat(left)}${head}${"─".repeat(inner - headWidth - left)}╮`),
    ...body.map((line) => `${style("36", "│")} ${line}${" ".repeat(inner - 2 - cells(line))} ${style("36", "│")}`),
    style("36", `╰${"─".repeat(inner)}╯`),
  ];
  output.write(lines.join("\r\n") + "\r\n");
}

/** Write a blank line and the `Question i of N · <header>` rule across the terminal. */
function writeRule(output: Output, index: number, total: number, header: string): void {
  const title = style("1", `Question ${index} of ${total}`) + (header ? ` ${style("2", `· ${header}`)}` : "");
  const fill = Math.max(0, columnsOf(output) - cells(title) - 1);
  output.write(`\r\n${title} ${style("36", "─".repeat(fill))}\r\n`);
}

/**
 * Ask one question: free text without options; else a choice over `label  —  description` rows
 * and an Other row (a checkbox for `multiSelect`), answering the bare labels. Other asks for the
 * text that takes its place.
 */
async function askQuestion(run: RunWidget, question: Omit<QuestionSpec, "header">): Promise<unknown> {
  const options = question.options;
  const label = visible(question.question);
  if (options.length === 0) return run(textWidget({ label }));
  const titles = options.map((option) =>
    visible(option.description ? `${option.label}  —  ${option.description}` : option.label),
  );
  titles.push(OTHER);
  const other = (): Promise<string> => run(textWidget({ label: "Your answer", mark: "✎" }));
  if (question.multiSelect) {
    const labels: string[] = [];
    for (const i of await run(checkboxWidget({ label, titles }))) {
      labels.push(i === options.length ? await other() : options[i]!.label);
    }
    return labels;
  }
  const chosen = await run(selectWidget({ label, titles, indicator: true }));
  return chosen === options.length ? other() : options[chosen]!.label;
}

/**
 * Answer one ask request inside a turn: a question set (an `ask` node's `QuestionSpec`s)
 * opens with the panel and asks each question under its rule, answering a record keyed by header;
 * `options` are one choice; otherwise a text prompt labelled with the question. The request's
 * text is drawn through `visible`; the answers keep its labels and headers as they are.
 */
async function promptRequest(run: RunWidget, output: Output, request: AskUserRequest): Promise<unknown> {
  const questions = (request.questions ?? []) as QuestionSpec[];
  if (questions.length > 0) {
    const title = visible(request.title || request.nodeId || "this step");
    writePanel(output, title, visible(request.question), questions.length);
    const answers: [string, unknown][] = [];
    for (const [i, question] of questions.entries()) {
      writeRule(output, i + 1, questions.length, visible(question.header));
      answers.push([question.header, await askQuestion(run, question)]);
    }
    // `fromEntries` defines each key as data, so a header `__proto__` stays an entry.
    return Object.fromEntries(answers);
  }
  if (request.options && request.options.length > 0) {
    const options = request.options.map((label) => ({ label, description: "" }));
    return askQuestion(run, { question: request.question, options, multiSelect: false });
  }
  return run(textWidget({ label: visible(request.question || `input for ${request.nodeId}`) }));
}
