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
  /** Called with true when a prompt turn takes the terminal and false when it gives it back. */
  onActive?: (active: boolean) => void;
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
  /** The frame's lines, styled, without newlines. */
  lines(): string[];
  /** The cursor's line in the frame and that line's text before it, or null to hide the cursor. */
  cursor(): { row: number; before: string } | null;
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
    (code >= 0x1f680 && code <= 0x1f6ff) ||
    (code >= 0x1f900 && code <= 0x1f9ff) ||
    (code >= 0x1fa70 && code <= 0x1faff) ||
    (code >= 0x20000 && code <= 0x3fffd)
  );
}

/**
 * Lay `line` out on a `columns`-wide terminal as the terminal wraps it: a character that doesn't
 * fit in the rest of a row (a wide one in the last column) starts the next row. Returns the rows
 * it takes and where the character at code-point offset `at` (ANSI sequences not counted) sits.
 */
function layout(line: string, columns: number, at: number): { rows: number; cursor: { row: number; col: number } } {
  let row = 0;
  let col = 0;
  let cursor: { row: number; col: number } | undefined;
  for (const [i, char] of Array.from(line.replace(ANSI, "")).entries()) {
    const width = isWide(char.codePointAt(0)!) ? 2 : 1;
    if (col + width > columns) {
      row++;
      col = 0;
    }
    if (i === at) cursor = { row, col };
    col += width;
  }
  // At the end of a line that fills its last row, the terminal holds the cursor in the last
  // column until the next character wraps it; the cursor stays there rather than adding a row.
  return { rows: row + 1, cursor: cursor ?? { row, col: Math.min(col, columns - 1) } };
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
  };
}

/**
 * A picker over `titles`, answering the chosen index, so equal titles stay apart. `indicator`
 * draws `●`/`○` before each title.
 */
function selectWidget(options: {
  label: string;
  titles: readonly string[];
  initial?: number;
  indicator?: boolean;
  mark?: string;
}): Widget<number> {
  const mark = options.mark ?? "?";
  const count = options.titles.length;
  let index = options.initial ?? 0;
  const row = (title: string, i: number): string => {
    if (i !== index) return `   ${options.indicator ? "○ " : ""}${title}`;
    return ` ${style("1;36", "»")} ${style("36", `${options.indicator ? "● " : ""}${title}`)}`;
  };
  return {
    lines: () => [
      `${style("1;36", mark)} ${style("1", options.label)} ${style("2", "(Use arrow keys)")}`,
      ...options.titles.map(row),
    ],
    cursor: () => null,
    key(key) {
      if (key.name === "return" || key.name === "enter") return { value: index };
      if (key.ctrl || key.meta) return undefined;
      if (key.name === "up" || key.name === "k") index = (index - 1 + count) % count;
      else if (key.name === "down" || key.name === "j") index = (index + 1) % count;
      return undefined;
    },
    done: (outcome) => collapsed(mark, options.label, "value" in outcome ? options.titles[outcome.value]! : ""),
  };
}

/** One prompt turn: its signal, how it ended without an answer, and the active widget's wakeup. */
interface Turn {
  signal: AbortSignal | undefined;
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
 * ones before it, takes the terminal for its widgets and gives it back restored. Ctrl-C, the end of
 * input or `close()` closes the terminal, so every queued and later prompt gets no answer.
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
  /** The last key was a `return`, so an `enter` now is the `\n` of the same `\r\n`. */
  #afterReturn = false;
  #pasting = false;

  /** Build a terminal over `options.input` and `options.output`; touches neither stream. */
  constructor(options: TerminalOptions = {}) {
    this.#input = options.input ?? process.stdin;
    this.#output = options.output ?? process.stderr;
    this.#onStop = options.onStop;
    this.#onActive = options.onActive;
  }

  /** A free-text or hidden prompt; null when the user cancels or the terminal is closed. */
  async askText(
    label: string,
    options: { initial?: string; secret?: boolean; signal?: AbortSignal } = {},
  ): Promise<string | null> {
    const ended = await this.#runTurn(options.signal, (run) =>
      run(textWidget({ label, initial: options.initial, mask: options.secret })),
    );
    return "value" in ended ? ended.value : null;
  }

  /** A single choice among strings; null when cancelled or closed. */
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
    const ended = await this.#runTurn(options.signal, (run) => run(selectWidget({ label, titles: choices, initial })));
    return "value" in ended ? choices[ended.value]! : null;
  }

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
   * Run `fn` as one turn of the FIFO: wait for the turns before it, take the terminal (raw mode,
   * listeners, input flowing), let `fn` run widgets, then restore the terminal on every outcome.
   * Returns `fn`'s value, or the stop that ended the turn (Ctrl-C, end of input, a closed
   * terminal); throws `signal`'s reason when it aborted first, else `fn`'s error.
   */
  async #runTurn<R>(
    signal: AbortSignal | undefined,
    fn: (run: RunWidget) => Promise<R>,
  ): Promise<{ value: R } | { stop: StopReason }> {
    if (this.#closed) return { stop: "closed" };
    signal?.throwIfAborted();
    const ahead = this.#tail;
    const { promise: finished, resolve: release } = Promise.withResolvers<void>();
    // The turn behind this one waits for the one ahead too, even when this one leaves early.
    this.#tail = ahead.then(() => finished);
    const turn: Turn = { signal, ending: undefined, wake: () => {} };
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
    const setsRaw = input.isTTY === true && typeof input.setRawMode === "function";
    // Read at turn start: while queued, the turn ahead may still hold the terminal raw.
    let wasRaw = false;
    // A `process.exit()` elsewhere skips the `finally` below.
    const onExit = (): void => {
      quietly(() => input.setRawMode!(false));
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
        wasRaw = input.isRaw === true;
        if (setsRaw) {
          process.once("exit", onExit);
          input.setRawMode!(true);
        }
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
        if (setsRaw) quietly(() => input.setRawMode!(wasRaw));
        quietly(() => output.write(SHOW_CURSOR + PASTE_OFF));
        quietly(() => process.removeListener("exit", onExit));
        quietly(() => input.pause());
        if (this.#closed) quietly(() => this.#detach());
        quietly(() => this.#onActive?.(false));
      }
    }
    const ending = turn.ending;
    // After the restore, so a handler that prints or exits never meets a raw terminal; before the
    // turns behind are released, so they see what `onStop` did.
    // Only the turn that closed the terminal records "interrupt" or "eof", so this runs once.
    if (ending === "interrupt" || ending === "eof") quietly(() => this.#onStop?.(ending));
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
        this.#closed = true;
      }
      finish({ stop: turn.ending });
    };
    turn.wake = () => finish({ stop: turn.ending! });
    // Rows from the frame's top to where the cursor was left, so the next draw starts there.
    let cursorRow = 0;
    const erase = (): string => `\r${cursorRow > 0 ? `${ESC}${cursorRow}A` : ""}${ESC}J`;
    const draw = (): void => {
      const columns = output.columns || 80;
      const lines = widget.lines();
      const cursor = widget.cursor();
      const at = cursor === null ? -1 : Array.from(cursor.before.replace(ANSI, "")).length;
      const laid = lines.map((line, i) => layout(line, columns, i === cursor?.row ? at : -1));
      const total = laid.reduce((sum, line) => sum + line.rows, 0);
      let text = erase() + lines.join("\r\n");
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
      output.write(PASTE_ON + (widget.cursor() === null ? HIDE_CURSOR : SHOW_CURSOR));
      draw();
      this.#active = { key: feed, end: () => stop("eof") };
      while (outcome === undefined && this.#buffer.length > 0) feed(this.#buffer.shift()!);
      if (outcome === undefined && this.#ended) stop("eof");
      await settled;
      output.write(erase() + widget.done(outcome!) + "\r\n");
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
