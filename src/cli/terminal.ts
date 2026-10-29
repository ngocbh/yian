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

/** The output a `Terminal` draws on: only `write` and `columns` are used. */
type Output = Pick<NodeJS.WritableStream, "write"> & { columns?: number };

export interface TerminalOptions {
  /** Where keys come from; defaults to process.stdin. Raw mode is set only when it is a TTY. */
  input?: Input;
  /** Where prompts are drawn; defaults to process.stderr (stdout stays the run's result). */
  output?: Output;
  /** Called once when Ctrl-C ("interrupt") or the end of input ("eof") closes the terminal. */
  onStop?: (reason: "interrupt" | "eof") => void;
  /** Called with true when a prompt turn takes the terminal and false when it gives it back. */
  onActive?: (active: boolean) => void;
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

/** One keypress, as the key loop hands it to a widget. */
interface Key {
  name: string | undefined;
  ctrl: boolean;
  meta: boolean;
  /** The text the key types, when it types any. */
  str: string | undefined;
  /** Whether the key arrived inside a bracketed paste. */
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
}

const ESC = "\x1b[";
const SHOW_CURSOR = `${ESC}?25h`;
const HIDE_CURSOR = `${ESC}?25l`;
const PASTE_ON = `${ESC}?2004h`;
const PASTE_OFF = `${ESC}?2004l`;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

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

/** `text` with each `\n` written as `\r\n`, since raw mode may not translate it. */
function crlf(text: string): string {
  return text.replaceAll("\n", "\r\n");
}

/** Wrap `text` in the SGR `codes`. */
function style(codes: string, text: string): string {
  return `${ESC}${codes}m${text}${ESC}0m`;
}

/** Whether `code` is an East Asian Wide or Fullwidth character or an emoji, two columns wide. */
function isWide(code: number): boolean {
  return (
    (code >= 0x1100 && code <= 0x115f) ||
    (code >= 0x2e80 && code <= 0x303e) ||
    (code >= 0x3041 && code <= 0x33ff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0xa000 && code <= 0xa4cf) ||
    (code >= 0xac00 && code <= 0xd7a3) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0xfe30 && code <= 0xfe4f) ||
    (code >= 0xff00 && code <= 0xff60) ||
    (code >= 0xffe0 && code <= 0xffe6) ||
    (code >= 0x1f300 && code <= 0x1f64f) ||
    (code >= 0x1f680 && code <= 0x1f6ff) ||
    (code >= 0x1f900 && code <= 0x1f9ff) ||
    (code >= 0x1fa70 && code <= 0x1faff) ||
    (code >= 0x20000 && code <= 0x3fffd)
  );
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

/** A one-line text editor; `mask` draws `*` per character and keeps the answer off the screen. */
function textWidget(options: {
  label: string;
  initial?: string;
  mask?: boolean;
  mark?: string;
  instruction?: string;
  validate?: (text: string) => string | undefined;
}): Widget<string> & { text(): string } {
  const mark = options.mark ?? "?";
  const chars = Array.from(options.initial ?? "");
  let pos = chars.length;
  let error: string | undefined;
  const shown = (part: readonly string[]): string =>
    options.mask ? "*".repeat(part.length) : part.map((char) => (char === "\n" ? "⏎" : char)).join("");
  const head = (): string => `${promptHead(mark, options.label, options.instruction)} `;
  const insert = (text: string): void => {
    const added = Array.from(text);
    chars.splice(pos, 0, ...added);
    pos += added.length;
  };
  return {
    preface: labelPreface(mark, options.label),
    lines: () => (error === undefined ? [head() + shown(chars)] : [head() + shown(chars), style("31", error)]),
    cursor: () => ({ row: 0, before: head() + shown(chars.slice(0, pos)) }),
    key(key) {
      const newline = key.name === "return" || key.name === "enter";
      error = undefined;
      if (newline && key.paste) insert("\n");
      else if (newline) {
        const text = chars.join("");
        error = options.validate?.(text);
        if (error === undefined) return { value: text };
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
      else if (printable(key)) insert(key.str);
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
    text: () => chars.join(""),
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
  #watch: { onKey: (key: "escape" | "interrupt") => void; wasRaw: boolean } | undefined;
  /** The last key was a `return`, so an `enter` now is the `\n` of the same `\r\n`. */
  #afterReturn = false;
  #pasting = false;

  /** Build a terminal over `options.input` and `options.output`; touches neither stream. */
  constructor(options: TerminalOptions = {}) {
    this.#input = options.input ?? process.stdin;
    this.#plain = options.plain === true;
    const output = options.output ?? process.stderr;
    this.#output = this.#plain ? plainOutput(output) : output;
    this.#onStop = options.onStop;
    this.#onActive = options.onActive;
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
   */
  async readLine<E extends boolean = false>(
    label: string,
    options: { signal?: AbortSignal; initial?: string; escape?: E } = {},
  ): Promise<
    | { value: string }
    | { eof: true }
    | { interrupt: true }
    | { aborted: true; draft: string }
    | (E extends true ? Escaped : never)
  > {
    const { signal, initial = "" } = options;
    const text = textWidget({ label, mark: ">", initial });
    const widget: Widget<string | typeof ESCAPED> = options.escape
      ? {
          ...text,
          key: (key) => (key.name === "escape" ? { value: ESCAPED } : text.key(key)),
          done: (outcome) =>
            "value" in outcome && outcome.value === ESCAPED ? null : text.done(outcome as Outcome<string>),
        }
      : text;
    if (signal?.aborted) return { aborted: true, draft: initial };
    let ended: { value: string | typeof ESCAPED } | { stop: StopReason };
    try {
      ended = await this.#runTurn(signal, (run) => run(widget), { keepOpen: true });
    } catch (error) {
      if (signal?.aborted && error === signal.reason) return { aborted: true, draft: text.text() };
      throw error;
    }
    if ("value" in ended) {
      // Only an `escape` prompt answers ESCAPED, so E is true here.
      if (ended.value === ESCAPED) return { escape: true, draft: text.text() } as E extends true ? Escaped : never;
      return { value: ended.value };
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
   * Ask the user to approve an action: write `body` and the unified `diff` when there is one, then
   * select among `choices` under `title`, the first preselected. All the text is drawn through
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
    const card = [request.body, request.diff ?? ""].filter((text) => text !== "").map(shown);
    const titles = choices.map(shown);
    let ended: { value: number } | { stop: StopReason };
    try {
      ended = await this.#runTurn(
        signal,
        (run) => {
          for (const text of card) this.#output.write(crlf(text.replace(/\n$/, "")) + "\r\n");
          return run(selectWidget({ label, titles }));
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
   * wait for the next prompt. A prompt meanwhile takes the keys as usual, and the watch goes on
   * after it. Does nothing on a plain terminal or an input that isn't a TTY.
   */
  watchKeys(signal: AbortSignal, onKey: (key: "escape" | "interrupt") => void): void {
    const input = this.#input;
    if (this.#plain || this.#closed || signal.aborted) return;
    if (input.isTTY !== true || typeof input.setRawMode !== "function") return;
    this.#attach();
    const watch = { onKey, wasRaw: input.isRaw === true };
    this.#watch = watch;
    // A `process.exit()` elsewhere skips the restore below.
    const onExit = (): void => quietly(() => input.setRawMode!(false));
    process.once("exit", onExit);
    quietly(() => input.setRawMode!(true));
    input.resume();
    const end = (): void => {
      if (this.#watch === watch) this.#watch = undefined;
      quietly(() => process.removeListener("exit", onExit));
      // A turn holding the terminal restores it when it ends.
      if (this.#turn !== undefined) return;
      quietly(() => input.setRawMode!(watch.wasRaw));
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
    if (name === "paste-start" || name === "paste-end") {
      this.#pasting = name === "paste-start";
      return;
    }
    if (name === "enter" && this.#afterReturn) {
      this.#afterReturn = false;
      return;
    }
    this.#afterReturn = name === "return";
    const key: Key = {
      name,
      ctrl: raw?.ctrl === true,
      meta: raw?.meta === true && alt === null,
      str: alt?.[1] ?? str,
      paste: this.#pasting,
    };
    const watching = this.#watch !== undefined && this.#turn === undefined;
    if (this.#active) this.#active.key(key);
    else if (watching && key.name === "escape") this.#watch!.onKey("escape");
    else if (watching && key.ctrl && key.name === "c") this.#watch!.onKey("interrupt");
    else this.#buffer.push(key);
  };

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
  }

  /**
   * Run `fn` as one turn of the FIFO: wait for the turns before it, take the terminal (raw mode,
   * listeners, input flowing), let `fn` run widgets, then restore the terminal on every outcome.
   * A plain terminal sets raw mode only with `raw`.
   * Returns `fn`'s value, or the stop that ended the turn (Ctrl-C, end of input, a closed
   * terminal); throws `signal`'s reason when it aborted first, else `fn`'s error.
   */
  async #runTurn<R>(
    signal: AbortSignal | undefined,
    fn: (run: RunWidget) => Promise<R>,
    { keepOpen = false, raw = false }: { keepOpen?: boolean; raw?: boolean } = {},
  ): Promise<{ value: R } | { stop: StopReason }> {
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
      quietly(() => output.write(SHOW_CURSOR + PASTE_OFF));
    };
    let started = false;
    let result: { value: R } | { error: unknown } | undefined;
    try {
      await Promise.race([ahead, queuedAbort]);
      if (turn.ending === undefined && this.#closed) turn.ending = "closed";
      if (turn.ending === undefined) {
        started = true;
        this.#turn = turn;
        quietly(() => this.#onActive?.(true));
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
        quietly(() => output.write(SHOW_CURSOR + PASTE_OFF));
        quietly(() => process.removeListener("exit", onExit));
        if (watch === undefined) quietly(() => input.pause());
        if (this.#closed) quietly(() => this.#detach());
        quietly(() => this.#onActive?.(false));
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
    const draw = (): void => {
      // A plain prompt is written once, as it opens; a refused line adds only its error, since
      // the cooked tty already echoed the line.
      if (this.#plain) {
        const refused = widget.refused?.();
        if (refused !== undefined) output.write(`${refused}\n`);
        return;
      }
      const columns = output.columns || 80;
      const lines = widget.lines();
      const cursor = widget.cursor();
      const at = cursor === null ? -1 : Array.from(cursor.before.replace(ANSI, "")).length;
      const laid = lines.map((line, i) => layout(line, columns, i === cursor?.row ? at : -1));
      const total = laid.reduce((sum, line) => sum + line.rows, 0);
      let text = erase() + crlf(lines.join("\n"));
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
    try {
      if (this.#plain) output.write(`${widget.preface}${widget.lines().join("\n")}\n`);
      else output.write(widget.preface + PASTE_ON + (widget.cursor() === null ? HIDE_CURSOR : SHOW_CURSOR));
      draw();
      this.#active = { key: feed, end: () => stop("eof") };
      while (outcome === undefined && this.#buffer.length > 0) feed(this.#buffer.shift()!);
      if (outcome === undefined && this.#ended) stop("eof");
      await settled;
      const ended = outcome!;
      // A `readLine` a deadline took down leaves no line: the prompt comes back after the reply.
      const vanish = turn.keepOpen && "stop" in ended && ended.stop === "aborted";
      const line = vanish ? null : widget.done(ended);
      if (line === null) output.write(erase());
      else if (!this.#plain) output.write(erase() + crlf(line) + "\r\n");
    } catch (error) {
      finish({ error });
    } finally {
      this.#active = undefined;
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
  const fill = Math.max(0, (output.columns || 80) - cells(title) - 1);
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
