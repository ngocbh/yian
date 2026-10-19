/**
 * Terminal scenarios written once for both runtimes: `Terminal` driven through a fake TTY, with a
 * small screen replayer that turns the drawn output back into the rows a terminal would show. They
 * are plain async functions using `node:assert/strict`, so the same checks run as Bun tests
 * (`terminal.test.ts`) and on Node, where this module is the main one and prints one JSON line per
 * scenario.
 */

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { getEventListeners } from "node:events";
import { PassThrough, Writable } from "node:stream";
import { fileURLToPath } from "node:url";

import { Terminal, type TerminalOptions } from "../../src/cli/terminal.ts";

export const UP = "\x1b[A";
export const DOWN = "\x1b[B";
export const LEFT = "\x1b[D";
export const RIGHT = "\x1b[C";
export const HOME = "\x1b[H";
export const END = "\x1b[F";
export const DELETE = "\x1b[3~";
export const ENTER = "\r";
export const CTRL_A = "\x01";
export const CTRL_C = "\x03";
export const CTRL_D = "\x04";
export const CTRL_E = "\x05";
export const CTRL_U = "\x15";
export const BACKSPACE = "\x7f";
export const SPACE = " ";
export const PASTE_START = "\x1b[200~";
export const PASTE_END = "\x1b[201~";

/** A TTY-like input: a `PassThrough` that records every `setRawMode` call (none without `tty`). */
export class FakeInput extends PassThrough {
  isTTY: boolean;
  isRaw = false;
  readonly rawCalls: boolean[] = [];
  setRawMode?: (mode: boolean) => this;

  constructor({ tty = true }: { tty?: boolean } = {}) {
    super();
    this.isTTY = tty;
    if (tty) {
      this.setRawMode = (mode: boolean) => {
        this.rawCalls.push(mode);
        this.isRaw = mode;
        return this;
      };
    }
  }
}

/** An output collecting what is written, `columns` wide; with `throwOn`, that write (1-based) throws once. */
export class FakeOutput extends Writable {
  text = "";
  columns: number;
  writes = 0;
  readonly throwOn: number | undefined;
  readonly failure = new Error("the output broke");

  constructor({ columns = 80, throwOn }: { columns?: number; throwOn?: number } = {}) {
    super();
    this.columns = columns;
    this.throwOn = throwOn;
  }

  // Collected here rather than in `_write`, so the text is complete as soon as `write` returns.
  override write(chunk: unknown, encoding?: unknown, callback?: unknown): boolean {
    this.writes++;
    if (this.writes === this.throwOn) throw this.failure;
    this.text += String(chunk);
    if (typeof encoding === "function") encoding();
    else if (typeof callback === "function") callback();
    return true;
  }
}

/** What a terminal shows after `text`: its rows (trailing blanks dropped), the cursor and the modes set. */
export interface Screen {
  rows: string[];
  cursor: { row: number; col: number };
  /** Each private mode written, in order, such as `?25l` or `?2004h`. */
  modes: string[];
}

/** Display width of one character: 2 for the CJK and fullwidth ranges the scenarios use. */
function charWidth(char: string): number {
  const code = char.codePointAt(0)!;
  return (code >= 0x3000 && code <= 0x9fff) || (code >= 0xff00 && code <= 0xff60) ? 2 : 1;
}

/**
 * Replay `text` on a `columns`-wide terminal. It knows text, `\r`, `\n` (down one row, as with
 * output newline translation off), `ESC[nA`, `ESC[J`, `ESC[K` and `ESC[nG`; colours are ignored.
 * Writing past the last column wraps on the next character, as terminals do.
 */
export function screen(text: string, columns = 80): Screen {
  const rows: string[][] = [[]];
  const modes: string[] = [];
  let row = 0;
  let col = 0;
  const line = (): string[] => {
    while (rows.length <= row) rows.push([]);
    return rows[row]!;
  };
  const pattern = /\x1b\[([?]?)([0-9;]*)([A-Za-z])|([\s\S])/gu;
  for (const match of text.matchAll(pattern)) {
    const [, priv, args, command, char] = match;
    if (char !== undefined) {
      if (char === "\r") col = 0;
      else if (char === "\n") row++;
      else {
        const width = charWidth(char);
        if (col + width > columns) {
          row++;
          col = 0;
        }
        const cells = line();
        while (cells.length < col) cells.push(" ");
        cells.splice(col, width, char, ...(width === 2 ? [""] : []));
        col += width;
      }
      continue;
    }
    const n = args === "" ? 1 : Number(args);
    if (priv === "?") modes.push(`?${args}${command}`);
    else if (command === "A") row = Math.max(0, row - n);
    else if (command === "G") col = Math.max(1, Math.min(n, columns)) - 1;
    else if (command === "K") line().length = Math.min(line().length, col);
    else if (command === "J") {
      line().length = Math.min(line().length, col);
      rows.length = row + 1;
    } else if (command !== "m") throw new Error(`the replayer doesn't know ESC[${args}${command}`);
  }
  const shown = rows.map((cells) => cells.join("").trimEnd());
  while (shown.length > 0 && shown[shown.length - 1] === "") shown.pop();
  return { rows: shown, cursor: { row, col }, modes };
}

/** Let the streams deliver what was written: they do it in next-tick callbacks and microtasks. */
export function tick(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** A fresh fake terminal: its input, output and the `Terminal` over them. */
function setup(
  options: { tty?: boolean; columns?: number; throwOn?: number } & Pick<TerminalOptions, "onStop" | "onActive"> = {},
): { input: FakeInput; output: FakeOutput; terminal: Terminal; view: () => Screen; exitHooks: number } {
  const input = new FakeInput({ tty: options.tty });
  const output = new FakeOutput({ columns: options.columns, throwOn: options.throwOn });
  const terminal = new Terminal({ input, output, onStop: options.onStop, onActive: options.onActive });
  const exitHooks = process.listenerCount("exit");
  return { input, output, terminal, view: () => screen(output.text, output.columns), exitHooks };
}

/**
 * Raw mode was set once and restored, the exit hook is gone (`exitHooks` is the count before the
 * prompt), and, unless `modes` is false, the cursor and bracketed paste were the last modes written.
 */
function assertRestored(input: FakeInput, output: FakeOutput, exitHooks: number, modes = true): void {
  if (input.isTTY) assert.deepEqual(input.rawCalls, [true, false]);
  assert.equal(input.isRaw, false);
  assert.equal(process.listenerCount("exit"), exitHooks);
  if (modes) assert.deepEqual(screen(output.text).modes.slice(-2).sort(), ["?2004l", "?25h"]);
}

async function typingALineReturnsIt(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const answer = terminal.askText("Name:");
  input.write("hello");
  await tick();
  assert.deepEqual(view().rows, ["? Name: hello"]);
  assert.deepEqual(view().cursor, { row: 0, col: 13 });
  input.write(ENTER);
  assert.equal(await answer, "hello");
  assert.deepEqual(view().rows, ["? Name: hello"]);
  assert.deepEqual(view().cursor, { row: 1, col: 0 });
  assertRestored(input, output, exitHooks);
}

async function anInitialValueCanBeEdited(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const answer = terminal.askText("Name:", { initial: "world" });
  await tick();
  assert.deepEqual(view().rows, ["? Name: world"]);
  input.write(BACKSPACE + LEFT + LEFT + "X");
  await tick();
  assert.deepEqual(view().rows, ["? Name: woXrl"]);
  assert.deepEqual(view().cursor, { row: 0, col: 11 });
  input.write(CTRL_A + "<" + CTRL_E + ">");
  await tick();
  assert.deepEqual(view().rows, ["? Name: <woXrl>"]);
  input.write(HOME + DELETE + END + "!");
  await tick();
  assert.deepEqual(view().rows, ["? Name: woXrl>!"]);
  input.write(CTRL_A + RIGHT + RIGHT + CTRL_U);
  await tick();
  assert.deepEqual(view().rows, ["? Name: Xrl>!"]);
  assert.deepEqual(view().cursor, { row: 0, col: 8 });
  input.write(ENTER);
  assert.equal(await answer, "Xrl>!");
  assertRestored(input, output, exitHooks);
}

async function aBlankAnswerReturnsAnEmptyString(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askText("Name:");
  input.write(ENTER);
  assert.equal(await answer, "");
  assert.deepEqual(view().rows, ["? Name:"]);
}

async function aSecretIsMaskedAndNeverEchoed(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const answer = terminal.askText("Key:", { secret: true });
  input.write("s3cret");
  await tick();
  assert.deepEqual(view().rows, ["? Key: ******"]);
  input.write(ENTER);
  assert.equal(await answer, "s3cret");
  assert.deepEqual(view().rows, ["? Key:"]);
  for (const piece of ["s3", "3c", "cr", "re", "et"]) assert.ok(!output.text.includes(piece), piece);
  assertRestored(input, output, exitHooks);
}

async function ctrlCReturnsNullAndCallsOnStopOnce(): Promise<void> {
  const stops: string[] = [];
  const rawWhenStopped: boolean[] = [];
  const { input, output, terminal, view, exitHooks } = setup({
    onStop: (reason) => {
      stops.push(reason);
      rawWhenStopped.push(input.isRaw);
    },
  });
  const answer = terminal.askText("Name:");
  input.write("abc" + CTRL_C);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["interrupt"]);
  assert.deepEqual(rawWhenStopped, [false]);
  assert.deepEqual(view().rows, ["? Name:"]);
  assertRestored(input, output, exitHooks);
  const drawn = output.text;
  assert.equal(await terminal.askText("Again:"), null);
  assert.equal(output.text, drawn);
  assert.deepEqual(stops, ["interrupt"]);
}

async function ctrlDOnAnEmptyLineIsTheEndOfInput(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.write(CTRL_D);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output, exitHooks);
}

async function theEndOfInputReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.end();
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output, exitHooks);
}

async function ctrlDOnANonEmptyLineIsIgnored(): Promise<void> {
  const stops: string[] = [];
  const { input, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.write("ab" + CTRL_D + ENTER);
  assert.equal(await answer, "ab");
  assert.deepEqual(stops, []);
}

async function aThrowingWriteAtAnySiteRestoresTheTerminal(): Promise<void> {
  // A clean run counts the writes: the modes, the first frame, the redraw after "a", the
  // collapsed line and the restored modes.
  const clean = setup();
  const answered = clean.terminal.askText("Name:");
  clean.input.write("a" + ENTER);
  assert.equal(await answered, "a");
  const sites = clean.output.writes;
  assert.equal(sites, 5);
  for (let throwOn = 1; throwOn <= sites; throwOn++) {
    const { input, output, terminal, exitHooks } = setup({ throwOn });
    const answer = terminal.askText("Name:");
    input.write("a" + ENTER);
    // Up to the redraw the failure is the outcome; after it the answer is recorded and outranks it.
    if (throwOn <= 3) await assert.rejects(answer, (error) => error === output.failure, `write ${throwOn}`);
    else assert.equal(await answer, "a", `write ${throwOn}`);
    assertRestored(input, output, exitHooks, throwOn !== sites);
  }
}

async function theExitHookRestoresTheTerminalEvenWhenWritingFails(): Promise<void> {
  // Write 1 sets the modes and write 2 draws the first frame, so the hook's write is the one that throws.
  const { input, output, terminal, exitHooks } = setup({ throwOn: 3 });
  const before = process.listeners("exit");
  const answer = terminal.askText("Name:");
  await tick();
  const added = process.listeners("exit").filter((listener) => !before.includes(listener));
  assert.equal(added.length, 1);
  assert.doesNotThrow(() => added[0]!(0));
  assert.deepEqual(input.rawCalls, [true, false]);
  assert.equal(output.writes, 3);
  input.write(ENTER);
  assert.equal(await answer, "");
  assert.equal(process.listenerCount("exit"), exitHooks);
}

async function wideCharactersAtAWrapBoundaryStartTheNextRow(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 10 });
  output.write("above\r\n");
  const answer = terminal.askText("Name:");
  input.write("a日日日日日b");
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: a", "日日日日日", "b"]);
  assert.deepEqual(view().cursor, { row: 3, col: 1 });
  input.write(BACKSPACE);
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: a", "日日日日日"]);
  assert.deepEqual(view().cursor, { row: 2, col: 9 });
  input.write(LEFT.repeat(5));
  await tick();
  assert.deepEqual(view().cursor, { row: 2, col: 0 });
  input.write(ENTER);
  assert.equal(await answer, "a日日日日日");
  assert.deepEqual(view().rows, ["above", "? Name: a", "日日日日日"]);
}

async function aNonTtyInputStillReadsALine(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup({ tty: false });
  const answer = terminal.askText("Name:");
  input.write("answer\n");
  assert.equal(await answer, "answer");
  assert.deepEqual(input.rawCalls, []);
  assertRestored(input, output, exitHooks);
}

async function aThrowingOnStopStillRestoresTheTerminal(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup({
    onStop: () => {
      throw new Error("the handler broke");
    },
  });
  const answer = terminal.askText("Name:");
  input.write(CTRL_C);
  assert.equal(await answer, null);
  assertRestored(input, output, exitHooks);
}

async function aBracketedPasteKeepsItsNewlines(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askText("Name:");
  input.write(PASTE_START + "x\ny" + PASTE_END);
  await tick();
  assert.deepEqual(view().rows, ["? Name: x⏎y"]);
  input.write(ENTER);
  assert.equal(await answer, "x\ny");
}

async function wideCharactersRedrawWithoutStaleRows(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 10 });
  output.write("above\r\n");
  const answer = terminal.askText("Name:");
  input.write("日本");
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: 日", "本"]);
  input.write(BACKSPACE);
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: 日"]);
  input.write("本" + ENTER);
  assert.equal(await answer, "日本");
  assert.deepEqual(view().rows, ["above", "? Name: 日", "本"]);
}

async function aLongLineRedrawsWithoutStaleRows(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 10 });
  output.write("above\r\n");
  const answer = terminal.askText("Name:");
  input.write("abcdefghijklmnop");
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: ab", "cdefghijkl", "mnop"]);
  assert.deepEqual(view().cursor, { row: 3, col: 4 });
  input.write(CTRL_A);
  await tick();
  assert.deepEqual(view().cursor, { row: 1, col: 8 });
  input.write(CTRL_E + BACKSPACE.repeat(10));
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name: ab", "cdef"]);
  input.write(CTRL_A + CTRL_E + CTRL_U);
  await tick();
  assert.deepEqual(view().rows, ["above", "? Name:"]);
  input.write("ok" + ENTER);
  assert.equal(await answer, "ok");
  assert.deepEqual(view().rows, ["above", "? Name: ok"]);
}

async function aChoiceMovesWithWraparoundAndReturnsTheTitle(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const answer = terminal.askChoice("Pick", ["a", "b", "c"], { initial: "b" });
  await tick();
  assert.deepEqual(view().rows, ["? Pick (Use arrow keys)", "   a", " » b", "   c"]);
  assert.ok(view().modes.includes("?25l"));
  input.write(DOWN + DOWN);
  await tick();
  assert.deepEqual(view().rows, ["? Pick (Use arrow keys)", " » a", "   b", "   c"]);
  input.write(UP);
  await tick();
  assert.deepEqual(view().rows.slice(1), ["   a", "   b", " » c"]);
  input.write("j");
  await tick();
  assert.deepEqual(view().rows.slice(1), [" » a", "   b", "   c"]);
  input.write("j" + "j" + "k" + "\t");
  await tick();
  assert.deepEqual(view().rows.slice(1), ["   a", " » b", "   c"]);
  input.write(ENTER);
  assert.equal(await answer, "b");
  assert.deepEqual(view().rows, ["? Pick b"]);
  assertRestored(input, output, exitHooks);
}

async function aBadChoiceListIsRefusedWithoutDrawing(): Promise<void> {
  const { output, terminal } = setup();
  await assert.rejects(terminal.askChoice("Pick", []), RangeError);
  await assert.rejects(terminal.askChoice("Pick", ["a", "b"], { initial: "c" }), RangeError);
  assert.equal(output.text, "");
}

async function aPromptQueuedWhileAnotherIsRawRestoresCookedMode(): Promise<void> {
  const { input, terminal } = setup();
  const first = terminal.askText("First:");
  await tick();
  assert.equal(input.isRaw, true);
  const second = terminal.askText("Second:");
  input.write("1" + ENTER);
  assert.equal(await first, "1");
  input.write("2" + ENTER);
  assert.equal(await second, "2");
  assert.deepEqual(input.rawCalls, [true, false, true, false]);
  assert.equal(input.isRaw, false);
}

async function equalTitlesMapBackByIndex(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askChoice("Pick", ["same", "same", "other"]);
  input.write(DOWN);
  await tick();
  assert.deepEqual(view().rows.slice(1), ["   same", " » same", "   other"]);
  input.write(ENTER);
  assert.equal(await answer, "same");
  assert.deepEqual(view().rows, ["? Pick same"]);
}

async function concurrentPromptsDrawOneAfterTheOther(): Promise<void> {
  const active: boolean[] = [];
  const { input, output, terminal, view, exitHooks } = setup({ onActive: (on) => active.push(on) });
  const first = terminal.askText("First:");
  const second = terminal.askChoice("Second", ["x", "y"]);
  await tick();
  assert.deepEqual(view().rows, ["? First:"]);
  input.write("one");
  await tick();
  assert.deepEqual(view().rows, ["? First: one"]);
  input.write(ENTER);
  assert.equal(await first, "one");
  await tick();
  assert.deepEqual(view().rows, ["? First: one", "? Second (Use arrow keys)", " » x", "   y"]);
  input.write(DOWN + ENTER);
  assert.equal(await second, "y");
  assert.deepEqual(view().rows, ["? First: one", "? Second y"]);
  assert.deepEqual(active, [true, false, true, false]);
  assert.deepEqual(input.rawCalls, [true, false, true, false]);
  assert.equal(input.isRaw, false);
  assert.equal(process.listenerCount("exit"), exitHooks);
  assert.deepEqual(screen(output.text).modes.slice(-2).sort(), ["?2004l", "?25h"]);
}

async function typeaheadInOneChunkAnswersQueuedPrompts(): Promise<void> {
  const { input, terminal, view } = setup();
  const first = terminal.askText("First:");
  const second = terminal.askText("Second:");
  input.write("a\r\nb\r\n");
  assert.equal(await first, "a");
  assert.equal(await second, "b");
  assert.deepEqual(view().rows, ["? First: a", "? Second: b"]);
}

async function anAlreadyAbortedSignalThrowsWithoutDrawing(): Promise<void> {
  const { output, terminal } = setup();
  const reason = new Error("stopped");
  await assert.rejects(terminal.askText("Name:", { signal: AbortSignal.abort(reason) }), (error) => error === reason);
  await assert.rejects(
    terminal.askChoice("Pick", ["a"], { signal: AbortSignal.abort(reason) }),
    (error) => error === reason,
  );
  assert.equal(output.text, "");
}

async function aQueuedPromptWhoseSignalAbortsNeverDraws(): Promise<void> {
  const { input, terminal, view } = setup();
  const controller = new AbortController();
  const reason = new Error("stopped");
  const first = terminal.askText("First:");
  const second = terminal.askText("Second:", { signal: controller.signal });
  const third = terminal.askText("Third:");
  await tick();
  controller.abort(reason);
  await assert.rejects(second, (error) => error === reason);
  await tick();
  assert.deepEqual(view().rows, ["? First:"]);
  input.write("1" + ENTER);
  assert.equal(await first, "1");
  await tick();
  assert.deepEqual(view().rows, ["? First: 1", "? Third:"]);
  input.write("3" + ENTER);
  assert.equal(await third, "3");
  assert.deepEqual(view().rows, ["? First: 1", "? Third: 3"]);
}

async function anActivePromptWhoseSignalAbortsThrowsTheReason(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const controller = new AbortController();
  const reason = new Error("stopped");
  const answer = terminal.askChoice("Pick", ["a", "b"], { signal: controller.signal });
  await tick();
  controller.abort(reason);
  await assert.rejects(answer, (error) => error === reason);
  assertRestored(input, output, exitHooks);
  assert.deepEqual(stops, []);
  // The terminal stays open: the next prompt runs.
  const next = terminal.askText("Next:");
  input.write("ok" + ENTER);
  assert.equal(await next, "ok");
}

async function aWidgetErrorAfterTheAbortThrowsTheReason(): Promise<void> {
  // Write 1 sets the modes, write 2 draws the frame; the abort ends the widget and write 3, the
  // collapsed line, throws.
  const { input, output, terminal, exitHooks } = setup({ throwOn: 3 });
  const controller = new AbortController();
  const reason = new Error("stopped");
  const answer = terminal.askText("Name:", { signal: controller.signal });
  await tick();
  controller.abort(reason);
  await assert.rejects(answer, (error) => error === reason);
  assert.equal(output.writes, 4);
  assertRestored(input, output, exitHooks);
}

async function promptsOnOneSignalLeaveNoAbortListeners(): Promise<void> {
  const { input, terminal } = setup();
  const controller = new AbortController();
  for (let i = 0; i < 20; i++) {
    const answer = terminal.askText("Name:", { signal: controller.signal });
    input.write(`${i}` + ENTER);
    assert.equal(await answer, `${i}`);
  }
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
}

/** None of the terminal's listeners are on `input`, and it is paused. */
function assertDetached(input: FakeInput): void {
  for (const event of ["keypress", "end", "error"]) assert.equal(input.listenerCount(event), 0, event);
  assert.equal(input.isPaused(), true);
}

async function closeEndsTheActivePromptAndRefusesLaterOnes(): Promise<void> {
  const stops: string[] = [];
  const active: boolean[] = [];
  const { input, output, terminal, exitHooks } = setup({
    onStop: (reason) => stops.push(reason),
    onActive: (on) => active.push(on),
  });
  const answer = terminal.askText("Name:");
  const queued = terminal.askChoice("Pick", ["a"]);
  input.write("ab");
  await tick();
  terminal.close();
  assert.equal(await answer, null);
  assert.equal(await queued, null);
  assertRestored(input, output, exitHooks);
  assertDetached(input);
  const drawn = output.text;
  assert.equal(await terminal.askText("Again:"), null);
  assert.equal(await terminal.askChoice("Again", ["a"]), null);
  assert.equal(output.text, drawn);
  assert.deepEqual(stops, []);
  assert.deepEqual(active, [true, false]);
}

async function closeWithNoActivePromptDetachesAndPauses(): Promise<void> {
  const { input, terminal } = setup();
  const answer = terminal.askText("Name:");
  input.write("x" + ENTER);
  assert.equal(await answer, "x");
  assert.ok(input.listenerCount("keypress") > 0);
  terminal.close();
  assertDetached(input);
  assert.equal(await terminal.askText("Again:"), null);
}

async function afterCtrlCAQueuedPromptReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const controller = new AbortController();
  const { input, output, terminal, view } = setup({
    onStop: (reason) => {
      stops.push(reason);
      controller.abort(new Error("stopped"));
    },
  });
  const answer = terminal.askText("Name:", { signal: controller.signal });
  const queued = terminal.askText("Queued:", { signal: controller.signal });
  const plain = terminal.askChoice("Plain", ["a"]);
  await tick();
  input.write(CTRL_C);
  assert.equal(await answer, null);
  assert.equal(await queued, null);
  assert.equal(await plain, null);
  assert.deepEqual(stops, ["interrupt"]);
  assert.deepEqual(view().rows, ["? Name:"]);
  assertDetached(input);
  assert.ok(!output.text.includes("Queued"));
  // A closed terminal outranks a signal that is already aborted.
  assert.equal(await terminal.askText("Later:", { signal: controller.signal }), null);
  assert.deepEqual(view().rows, ["? Name:"]);
}

async function afterTheEndOfInputAQueuedPromptReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  const queued = terminal.askText("Queued:");
  await tick();
  input.end();
  assert.equal(await answer, null);
  assert.equal(await queued, null);
  assert.deepEqual(stops, ["eof"]);
  assert.ok(!output.text.includes("Queued"));
}

async function anUnusedTerminalTouchesNothing(): Promise<void> {
  const input = new FakeInput();
  const output = new FakeOutput();
  const before = input.eventNames();
  new Terminal({ input, output });
  input.write("typed");
  await tick();
  assert.deepEqual(input.eventNames(), before);
  assert.equal(input.readableFlowing, null);
  assert.equal(String(input.read()), "typed");
  assert.equal(output.text, "");
}

async function anInputThatAlreadyEndedGivesEofAtTheFirstPrompt(): Promise<void> {
  const stops: string[] = [];
  const { input, terminal } = setup({ tty: false, onStop: (reason) => stops.push(reason) });
  input.end();
  input.resume();
  await new Promise((resolve) => input.once("end", resolve));
  await tick();
  assert.deepEqual(stops, []);
  assert.equal(await terminal.askText("Name:"), null);
  assert.deepEqual(stops, ["eof"]);
  // No isPaused() check: on Bun, pausing a stream that already ended leaves it reporting not paused.
  for (const event of ["keypress", "end", "error"]) assert.equal(input.listenerCount(event), 0, event);
}

async function anInputErrorEndsTheActivePromptAsEndOfInput(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.write("ab");
  await tick();
  input.destroy(new Error("EIO"));
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output, exitHooks);
  for (const event of ["keypress", "end", "error"]) assert.equal(input.listenerCount(event), 0, event);
}

async function aThrowingOnActiveStillRestoresTheTerminal(): Promise<void> {
  const active: boolean[] = [];
  const { input, output, terminal, exitHooks } = setup({
    onActive: (on) => {
      active.push(on);
      throw new Error("the handler broke");
    },
  });
  const answer = terminal.askText("Name:");
  input.write("x" + ENTER);
  assert.equal(await answer, "x");
  assert.deepEqual(active, [true, false]);
  assertRestored(input, output, exitHooks);
}

/** A spawned child's exit, waited for at most 10 s with its stdin kept open; what it printed. */
async function runChild(send: string): Promise<{ code: number | null; stdout: string; stderr: string }> {
  // No execArgv forwarded: Node 22.18+ strips types by default, and Bun's execArgv isn't safe to pass on.
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), "child"], { stdio: "pipe" });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8").on("data", (chunk: string) => {
    stdout += chunk;
  });
  child.stderr.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  child.stdin.on("error", () => {});
  const exited = new Promise<number | null>((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) => resolve(code));
  });
  let timer: ReturnType<typeof setTimeout> | undefined;
  const limit = new Promise<"timeout">((resolve) => {
    timer = setTimeout(() => resolve("timeout"), 10_000);
  });
  try {
    // Never ended: the child must exit on its own with its stdin still open.
    child.stdin.write(send);
    const code = await Promise.race([exited, limit]);
    if (code === "timeout") assert.fail(`the child didn't exit within 10 s; stdout: ${stdout}; stderr: ${stderr}`);
    return { code, stdout, stderr };
  } finally {
    clearTimeout(timer);
    child.kill("SIGKILL");
    await exited.catch(() => {});
    child.stdin.destroy();
  }
}

/** The child side: one `askText` on the process's own stdin and stderr, then `close()`. */
async function child(): Promise<void> {
  const terminal = new Terminal();
  const answer = await terminal.askText("Name:");
  console.log(String(answer));
  terminal.close();
}

async function aChildExitsOnItsOwnAfterAnAnswerWithStdinOpen(): Promise<void> {
  const { code, stdout, stderr } = await runChild("answer\n");
  assert.equal(code, 0, stderr);
  assert.equal(stdout, "answer\n");
}

async function aChildExitsOnItsOwnAfterCtrlCWithStdinOpen(): Promise<void> {
  const { code, stdout, stderr } = await runChild(CTRL_C);
  assert.equal(code, 0, stderr);
  assert.equal(stdout, "null\n");
}

/** The scenarios by name, in the order `main` runs them. */
export const SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  "typing a line returns it": typingALineReturnsIt,
  "an initial value can be edited": anInitialValueCanBeEdited,
  "a blank answer returns an empty string": aBlankAnswerReturnsAnEmptyString,
  "a secret is masked and never echoed": aSecretIsMaskedAndNeverEchoed,
  "Ctrl-C returns null and calls onStop once": ctrlCReturnsNullAndCallsOnStopOnce,
  "Ctrl-D on an empty line is the end of input": ctrlDOnAnEmptyLineIsTheEndOfInput,
  "the end of input returns null": theEndOfInputReturnsNull,
  "Ctrl-D on a non-empty line is ignored": ctrlDOnANonEmptyLineIsIgnored,
  "a throwing write at any site restores the terminal": aThrowingWriteAtAnySiteRestoresTheTerminal,
  "the exit hook restores the terminal even when writing fails": theExitHookRestoresTheTerminalEvenWhenWritingFails,
  "a non-TTY input still reads a line": aNonTtyInputStillReadsALine,
  "a throwing onStop still restores the terminal": aThrowingOnStopStillRestoresTheTerminal,
  "a bracketed paste keeps its newlines": aBracketedPasteKeepsItsNewlines,
  "wide characters redraw without stale rows": wideCharactersRedrawWithoutStaleRows,
  "wide characters at a wrap boundary start the next row": wideCharactersAtAWrapBoundaryStartTheNextRow,
  "a long line redraws without stale rows": aLongLineRedrawsWithoutStaleRows,
  "a choice moves with wraparound and returns the title": aChoiceMovesWithWraparoundAndReturnsTheTitle,
  "a bad choice list is refused without drawing": aBadChoiceListIsRefusedWithoutDrawing,
  "equal titles map back by index": equalTitlesMapBackByIndex,
  "a prompt queued while another is raw restores cooked mode": aPromptQueuedWhileAnotherIsRawRestoresCookedMode,
  "concurrent prompts draw one after the other": concurrentPromptsDrawOneAfterTheOther,
  "typeahead in one chunk answers queued prompts": typeaheadInOneChunkAnswersQueuedPrompts,
  "an already-aborted signal throws without drawing": anAlreadyAbortedSignalThrowsWithoutDrawing,
  "a queued prompt whose signal aborts never draws": aQueuedPromptWhoseSignalAbortsNeverDraws,
  "an active prompt whose signal aborts throws the reason": anActivePromptWhoseSignalAbortsThrowsTheReason,
  "a widget error after the abort throws the reason": aWidgetErrorAfterTheAbortThrowsTheReason,
  "prompts on one signal leave no abort listeners": promptsOnOneSignalLeaveNoAbortListeners,
  "close ends the active prompt and refuses later ones": closeEndsTheActivePromptAndRefusesLaterOnes,
  "close with no active prompt detaches and pauses": closeWithNoActivePromptDetachesAndPauses,
  "after Ctrl-C a queued prompt returns null": afterCtrlCAQueuedPromptReturnsNull,
  "after the end of input a queued prompt returns null": afterTheEndOfInputAQueuedPromptReturnsNull,
  "an unused terminal touches nothing": anUnusedTerminalTouchesNothing,
  "an input that already ended gives EOF at the first prompt": anInputThatAlreadyEndedGivesEofAtTheFirstPrompt,
  "an input error ends the active prompt as end of input": anInputErrorEndsTheActivePromptAsEndOfInput,
  "a throwing onActive still restores the terminal": aThrowingOnActiveStillRestoresTheTerminal,
  "a child exits on its own after an answer with stdin open": aChildExitsOnItsOwnAfterAnAnswerWithStdinOpen,
  "a child exits on its own after Ctrl-C with stdin open": aChildExitsOnItsOwnAfterCtrlCWithStdinOpen,
};

/** Run every scenario, printing `{"name", "ok", "error"?}` per line; return how many failed. */
async function main(): Promise<number> {
  let failures = 0;
  for (const [name, run] of Object.entries(SCENARIOS)) {
    try {
      await run();
      console.log(JSON.stringify({ name, ok: true }));
    } catch (error) {
      failures++;
      const text = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.log(JSON.stringify({ name, ok: false, error: text }));
    }
  }
  return failures;
}

// `import.meta.main` doesn't exist on Node 22.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === "child") await child();
  // No process.exit: a leaked listener or timer then shows as a hang.
  else process.exitCode = (await main()) ? 1 : 0;
}
