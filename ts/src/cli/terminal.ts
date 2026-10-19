/**
 * The CLI's terminal: the prompts a run shows on the host's terminal, drawn with questionary's look
 * from keypresses, one widget at a time, always giving the terminal back in cooked mode.
 */

import { emitKeypressEvents } from "node:readline";

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

/** The output a `Terminal` draws on. */
type Output = NodeJS.WritableStream & { columns?: number };

export interface TerminalOptions {
  /** Where keys come from; defaults to process.stdin. Raw mode is set only when it is a TTY. */
  input?: Input;
  /** Where prompts are drawn; defaults to process.stderr (stdout stays the run's result). */
  output?: Output;
  /** Called once when Ctrl-C ("interrupt") or the end of input ("eof") closes the terminal. */
  onStop?: (reason: "interrupt" | "eof") => void;
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

/** How a widget ended. */
type Outcome<T> = { value: T } | { stop: StopReason } | { error: unknown };

/** One prompt's state and drawing; the key loop owns the terminal around it. */
interface Widget<T> {
  /** The frame's lines, styled, without newlines. */
  lines(): string[];
  /** The cursor's line and display column in the frame, or null to hide it. */
  cursor(): { row: number; col: number } | null;
  /** Apply one key; an outcome ends the widget. */
  key(key: Key): { value: T } | { stop: "eof" } | undefined;
  /** The one line the frame collapses to once the widget has ended. */
  done(outcome: Outcome<T>): string;
}

const ESC = "\x1b[";
const SHOW_CURSOR = `${ESC}?25h`;
const HIDE_CURSOR = `${ESC}?25l`;
const PASTE_ON = `${ESC}?2004h`;
const PASTE_OFF = `${ESC}?2004l`;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

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
    (code >= 0x1f900 && code <= 0x1f9ff) ||
    (code >= 0x20000 && code <= 0x3fffd)
  );
}

/** The columns `text` takes on a terminal, ignoring its ANSI sequences. */
function displayWidth(text: string): number {
  let width = 0;
  for (const char of text.replace(ANSI, "")) width += isWide(char.codePointAt(0)!) ? 2 : 1;
  return width;
}

/** The rows a line of `width` columns takes on a `columns`-wide terminal. */
function rowsOf(width: number, columns: number): number {
  return Math.max(1, Math.ceil(width / columns));
}

/** The collapsed line `? <label> <answer>`, with nothing after the label when `answer` is empty. */
function collapsed(mark: string, label: string, answer: string): string {
  const head = `${style("1;36", mark)} ${style("1", label)}`;
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
}): Widget<string> {
  const mark = options.mark ?? "?";
  const chars = Array.from(options.initial ?? "");
  let pos = chars.length;
  let error: string | undefined;
  const shown = (part: readonly string[]): string =>
    options.mask ? "*".repeat(part.length) : part.map((char) => (char === "\n" ? "⏎" : char)).join("");
  const head = (): string => {
    const instruction = options.instruction ? ` ${style("2", options.instruction)}` : "";
    return `${style("1;36", mark)} ${style("1", options.label)}${instruction} `;
  };
  const insert = (text: string): void => {
    const added = Array.from(text);
    chars.splice(pos, 0, ...added);
    pos += added.length;
  };
  return {
    lines: () => (error === undefined ? [head() + shown(chars)] : [head() + shown(chars), style("31", error)]),
    cursor: () => ({ row: 0, col: displayWidth(head()) + displayWidth(shown(chars.slice(0, pos))) }),
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
  };
}

/**
 * The host's terminal for a run's prompts. Each prompt takes the terminal for its widget and gives
 * it back restored; Ctrl-C or the end of input closes the terminal, so every later prompt gets no
 * answer.
 *
 * @example
 * const terminal = new Terminal({ onStop: () => controller.abort() });
 * const name = await terminal.askText("Name:");
 */
export class Terminal {
  readonly #input: Input;
  readonly #output: Output;
  readonly #onStop: ((reason: "interrupt" | "eof") => void) | undefined;
  #attached = false;
  #closed = false;
  #stopNotified = false;
  /** Set once the input has ended or failed; the next widget ends with the EOF outcome. */
  #ended = false;
  /** Keys that arrived while no widget was active, fed to the next widget first. */
  #buffer: Key[] = [];
  /** The active widget's key and end handlers. */
  #active: { key: (key: Key) => void; end: () => void } | undefined;
  /** The last key was a `return`, so an `enter` now is the `\n` of the same `\r\n`. */
  #afterReturn = false;
  #pasting = false;

  /** Build a terminal over `options.input` and `options.output`; touches neither stream. */
  constructor(options: TerminalOptions = {}) {
    this.#input = options.input ?? process.stdin;
    this.#output = options.output ?? process.stderr;
    this.#onStop = options.onStop;
  }

  /** A free-text or hidden prompt; null when the user cancels or the terminal is closed. */
  async askText(
    label: string,
    options: { initial?: string; secret?: boolean; signal?: AbortSignal } = {},
  ): Promise<string | null> {
    if (this.#closed) return null;
    const outcome = await this.#run(
      textWidget({ label, initial: options.initial, mask: options.secret }),
      options.signal,
    );
    return "value" in outcome ? outcome.value : null;
  }

  /** Normalize one keypress and hand it to the active widget, or buffer it for the next one. */
  readonly #onKeypress = (
    str: string | undefined,
    raw: { name?: string; ctrl?: boolean; meta?: boolean } | undefined,
  ): void => {
    const name = raw?.name;
    if (name === "paste-start" || name === "paste-end") {
      this.#pasting = name === "paste-start";
      return;
    }
    if (name === "enter" && this.#afterReturn) {
      this.#afterReturn = false;
      return;
    }
    this.#afterReturn = name === "return";
    const key: Key = { name, ctrl: raw?.ctrl === true, meta: raw?.meta === true, str, paste: this.#pasting };
    if (this.#active) this.#active.key(key);
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
   * Run `widget` on the terminal until it ends: draw it, feed it keys, collapse it, and restore the
   * terminal on every outcome. Ctrl-C or the end of input closes the terminal and then calls
   * `onStop`; an error, or `signal`'s reason when it aborts first, is thrown.
   */
  async #run<T>(widget: Widget<T>, signal?: AbortSignal): Promise<{ value: T } | { stop: StopReason }> {
    signal?.throwIfAborted();
    const input = this.#input;
    const output = this.#output;
    let outcome: Outcome<T> | undefined;
    let settle!: () => void;
    const settled = new Promise<void>((resolve) => {
      settle = resolve;
    });
    // The first outcome wins: a later abort, key or error can't replace it.
    const finish = (ended: Outcome<T>): void => {
      if (outcome !== undefined) return;
      outcome = ended;
      this.#active = undefined;
      if ("stop" in ended) this.#closed = true;
      settle();
    };
    // Rows from the frame's top to where the cursor was left, so the next draw starts there.
    let cursorRow = 0;
    const erase = (): string => `\r${cursorRow > 0 ? `${ESC}${cursorRow}A` : ""}${ESC}J`;
    const draw = (): void => {
      const columns = output.columns || 80;
      const lines = widget.lines();
      const rows = lines.map((line) => rowsOf(displayWidth(line), columns));
      const total = rows.reduce((sum, n) => sum + n, 0);
      let text = erase() + lines.join("\r\n");
      cursorRow = total - 1;
      const cursor = widget.cursor();
      if (cursor !== null) {
        const wrapped = Math.min(Math.floor(cursor.col / columns), (rows[cursor.row] ?? 1) - 1);
        const target = rows.slice(0, cursor.row).reduce((sum, n) => sum + n, 0) + wrapped;
        if (cursorRow > target) text += `${ESC}${cursorRow - target}A`;
        text += `\r${ESC}${cursor.col - wrapped * columns + 1}G`;
        cursorRow = target;
      }
      output.write(text);
    };
    const feed = (key: Key): void => {
      try {
        if (key.ctrl && key.name === "c") return finish({ stop: "interrupt" });
        const ended = widget.key(key);
        if (ended !== undefined) return finish(ended);
        draw();
      } catch (error) {
        finish({ error });
      }
    };
    const onAbort = (): void => finish({ error: signal!.reason });
    const wasRaw = input.isRaw === true;
    const raw = input.isTTY === true && typeof input.setRawMode === "function";
    // A `process.exit()` elsewhere skips the `finally` below.
    const onExit = (): void => {
      try {
        input.setRawMode!(false);
      } catch {}
      try {
        output.write(SHOW_CURSOR + PASTE_OFF);
      } catch {}
    };
    try {
      this.#attach();
      if (raw) {
        process.once("exit", onExit);
        input.setRawMode!(true);
      }
      output.write(PASTE_ON + (widget.cursor() === null ? HIDE_CURSOR : ""));
      draw();
      this.#active = { key: feed, end: () => finish({ stop: "eof" }) };
      signal?.addEventListener("abort", onAbort, { once: true });
      input.resume();
      while (outcome === undefined && this.#buffer.length > 0) feed(this.#buffer.shift()!);
      if (outcome === undefined && this.#ended) finish({ stop: "eof" });
      await settled;
      output.write(erase() + widget.done(outcome!) + "\r\n");
    } catch (error) {
      finish({ error });
    } finally {
      this.#active = undefined;
      signal?.removeEventListener("abort", onAbort);
      // Each step runs on its own, cooked mode first, so a failing one never skips the rest. Their
      // errors are dropped: the outcome recorded before them is what the prompt reports.
      const step = (action: () => unknown): void => {
        try {
          action();
        } catch {}
      };
      if (raw) step(() => input.setRawMode!(wasRaw));
      step(() => output.write(SHOW_CURSOR + PASTE_OFF));
      step(() => process.removeListener("exit", onExit));
      step(() => input.pause());
      if (this.#closed) this.#detach();
    }
    const ended = outcome!;
    // After the restore, so a handler that prints or exits never meets a raw terminal.
    if ("stop" in ended && ended.stop !== "closed" && !this.#stopNotified) {
      this.#stopNotified = true;
      try {
        this.#onStop?.(ended.stop);
      } catch {}
    }
    if ("error" in ended) throw ended.error;
    return ended;
  }
}
