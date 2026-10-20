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
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

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

// The `promptInputs` scenarios port the prompt-flow tests of `tests/engine/test_cli_prompt.py`,
// each named in the comment above its scenario. Not ported: the questionary-internal
// `test_nullable_open_repeated_palettes_use_fresh_choices` and
// `test_prompt_controls_are_fresh_for_each_widget` (no widget objects are shared here), and
// `test_gather_non_tty_*` (the run's prompt gate, not the terminal).

/** One source-free flow input; untyped when `type` is undefined. */
function decl(
  name: string,
  type?: Type,
  options: { required?: boolean; hasDefault?: boolean; default?: unknown } = {},
): Parameter {
  const contract = type === undefined ? null : new ValueContract(type);
  return new Parameter(name, { contract, coerceStrings: true, ...options });
}

const STR = Type.scalar(ValueKind.STRING);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const LIST_STR = Type.scalar(ValueKind.LIST_STRING);
const LIST_ANY = Type.scalar(ValueKind.LIST_ANY);

/** `kind`, nullable. */
function nullable(kind: ValueKind): Type {
  return new Type({ kind, nullable: true });
}

/** A `Literal(...)` of `tags`. */
function literal(tags: readonly string[], options: { nullable?: boolean } = {}): Type {
  return new Type({ kind: ValueKind.STRING, tags: new Set(tags), nullable: options.nullable ?? false });
}

/** A record with a required int `count`. */
const COUNT_RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["count", Type.scalar(ValueKind.INTEGER)]]),
  required: new Set(["count"]),
});

/** The active select's choice titles and the marked one, read off the screen. */
function frame(view: () => Screen): { label: string; titles: string[]; marked: number } {
  const rows = view().rows;
  const head = rows.findLastIndex((row) => row.endsWith(" (Use arrow keys)"));
  assert.ok(head >= 0, `no select on the screen: ${JSON.stringify(rows)}`);
  const choices = rows.slice(head + 1);
  return {
    label: rows[head]!.slice(2, -" (Use arrow keys)".length),
    titles: choices.map((row) => row.slice(3)),
    marked: choices.findIndex((row) => row.startsWith(" » ")),
  };
}

/** Wait for the active select, move its marker to `title` and press ENTER. */
async function choose(input: FakeInput, view: () => Screen, title: string): Promise<void> {
  await tick();
  const { titles, marked } = frame(view);
  const target = titles.indexOf(title);
  assert.ok(target >= 0, `no choice ${title} in ${JSON.stringify(titles)}`);
  input.write((target > marked ? DOWN : UP).repeat(Math.abs(target - marked)) + ENTER);
}

/** Wait for the active select and return its frame. */
async function select(view: () => Screen): Promise<{ label: string; titles: string[]; marked: number }> {
  await tick();
  return frame(view);
}

// test_prompt_preserves_required_false_enum_and_blank_text
async function aRequiredBoolLiteralAndStringAreAskedInOneTurn(): Promise<void> {
  const active: boolean[] = [];
  const { input, output, terminal, view, exitHooks } = setup({ onActive: (on) => active.push(on) });
  const answer = terminal.promptInputs(
    [
      decl("flag", BOOL, { required: true }),
      decl("mode", literal(["slow", "fast"]), { required: true }),
      decl("note", STR, { required: true }),
    ],
    {},
  );
  await tick();
  assert.deepEqual(view().rows, ["? flag (bool) * (y/N)"]);
  input.write("n");
  const mode = await select(view);
  assert.deepEqual(mode, { label: "mode (Literal(fast, slow)) *", titles: ["fast", "slow"], marked: 0 });
  await choose(input, view, "slow");
  input.write(ENTER);
  assert.deepEqual(await answer, { flag: false, mode: "slow", note: "" });
  assert.deepEqual(view().rows, ["? flag (bool) * No", "? mode (Literal(fast, slow)) * slow", "? note (str) *"]);
  assert.deepEqual(active, [true, false]);
  assertRestored(input, output, exitHooks);
}

async function aRequiredBoolConfirmsWithYOrNAtOnceAndEnterSaysNo(): Promise<void> {
  const cases: [string, boolean, string][] = [
    ["xy", true, "Yes"],
    ["Y", true, "Yes"],
    ["n", false, "No"],
    ["N", false, "No"],
    [ENTER, false, "No"],
  ];
  for (const [keys, expected, shown] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("ok", BOOL, { required: true })], {});
    await tick();
    assert.deepEqual(view().rows, ["? ok (bool) * (y/N)"]);
    input.write(keys);
    assert.deepEqual(await answer, { ok: expected }, keys);
    assert.deepEqual(view().rows, [`? ok (bool) * ${shown}`]);
  }
}

// test_structured_prompt_retries_nested_failure_and_returns_native_value
async function aListEditorShowsTheErrorUnderItAndStaysOpen(): Promise<void> {
  const { input, terminal, view } = setup();
  const rows = new Type({ kind: ValueKind.LIST_OBJECT, element: COUNT_RECORD });
  const answer = terminal.promptInputs([decl("rows", rows, { required: true })], {});
  const head = '? rows (list({"count":int})) * Enter a JSON array';
  input.write('[{"count": "bad"}]' + ENTER);
  await tick();
  assert.deepEqual(view().rows, [
    `${head} [{"count": "bad"}]`,
    '$[0]["count"]: "bad" does not match declared type int',
  ]);
  input.write(LEFT);
  await tick();
  assert.deepEqual(view().rows, [`${head} [{"count": "bad"}]`]);
  input.write(CTRL_E + CTRL_U + '[{"count": 2}]' + ENTER);
  assert.deepEqual(await answer, { rows: [{ count: 2n }] });
  assert.deepEqual(view().rows, ['? rows (list({"count":int})) * [{"count": 2}]']);
}

// test_object_prompt_returns_native_value_and_json_instruction
async function anObjectEditorReturnsTheNativeObject(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("profile", COUNT_RECORD, { required: true })], {});
  await tick();
  assert.deepEqual(view().rows, ['? profile ({"count":int}) * Enter a JSON object']);
  input.write('{"count": 2}' + ENTER);
  assert.deepEqual(await answer, { profile: { count: 2n } });
}

// test_optional_structured_whitespace_is_validated_then_omitted,
// test_nonnullable_defaulted_open_blank_stays_direct_omission
async function anOptionalOrDefaultedDirectEditorOmitsOnABlank(): Promise<void> {
  const decls = [
    decl("items", LIST_STR),
    decl("value", STR, { hasDefault: true, default: "fallback" }),
    decl("list", LIST_ANY, { hasDefault: true, default: [1n] }),
  ];
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(decls, {});
  input.write("   " + ENTER + "   " + ENTER + "   " + ENTER);
  assert.deepEqual(await answer, {});
  assert.equal(view().rows.length, 3);
}

// test_required_structured_whitespace_is_rejected_then_retried
async function aRequiredListEditorRefusesABlank(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("items", LIST_STR, { required: true })], {});
  input.write("   " + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(1), ["Invalid JSON at line 1, column 4: Expecting value"]);
  input.write(CTRL_U + "[]" + ENTER);
  assert.deepEqual(await answer, { items: [] });
}

// test_structured_prompt_cancellation_bypasses_validation
async function ctrlCInAListEditorReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.promptInputs(
    [decl("items", LIST_STR, { required: true }), decl("later", STR, { required: true })],
    {},
  );
  input.write("[" + CTRL_C);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["interrupt"]);
  assert.ok(!output.text.includes("later"));
  assertRestored(input, output, exitHooks);
}

// test_nullable_structured_enter_rejects_textual_null_then_returns_native
async function aNullableListOrObjectEditorRefusesATextualNull(): Promise<void> {
  const cases: [Type, string, unknown, string][] = [
    [nullable(ValueKind.LIST_ANY), "[null]", [null], "array"],
    [nullable(ValueKind.OBJECT), '{"value": null}', { value: null }, "object"],
  ];
  for (const [type, raw, expected, shape] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("items", type, { required: true })], {});
    const palette = await select(view);
    assert.deepEqual(palette.titles, ["Set null", "Enter value"]);
    assert.equal(palette.titles[palette.marked], "Enter value");
    input.write(ENTER + "null" + ENTER);
    await tick();
    assert.ok(view().rows.at(-2)!.endsWith(`Enter a JSON ${shape} null`), view().rows.at(-2));
    assert.equal(view().rows.at(-1), `Expected a JSON ${shape}`);
    input.write(CTRL_U + raw + ENTER);
    assert.deepEqual(await answer, { items: expected });
  }
}

// test_nullable_open_scalar_enter_preserves_literal_null
async function theTextNullInANullableStringEditorStaysAString(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("note", nullable(ValueKind.STRING))], {});
  const palette = await select(view);
  assert.deepEqual(palette, {
    label: "note (Optional(str)) [optional]",
    titles: ["↩ Omit input", "Set null", "Enter value"],
    marked: 0,
  });
  await choose(input, view, "Enter value");
  input.write("null" + ENTER);
  assert.deepEqual(await answer, { note: "null" });
  assert.deepEqual(view().rows, [
    "? note (Optional(str)) [optional] Enter value",
    "? note (Optional(str)) [optional] null",
  ]);
}

const NULLABLE_OPEN = [nullable(ValueKind.STRING), nullable(ValueKind.LIST_ANY), nullable(ValueKind.OBJECT)];

// test_nullable_open_prompt_cancellation_at_action, test_nullable_open_prompt_cancellation_in_editor
async function cancellingAtThePaletteOrItsEditorReturnsNull(): Promise<void> {
  for (const type of NULLABLE_OPEN) {
    const atPalette = setup();
    const first = atPalette.terminal.promptInputs([decl("value", type)], {});
    atPalette.input.write(CTRL_C);
    assert.equal(await first, null);
    const inEditor = setup();
    const second = inEditor.terminal.promptInputs([decl("value", type)], {});
    await choose(inEditor.input, inEditor.view, "Enter value");
    await tick();
    inEditor.input.end();
    assert.equal(await second, null);
  }
}

// test_nullable_open_blank_can_return_to_omit, test_nullable_open_blank_can_return_to_set_null,
// test_nullable_open_blank_can_reenter_real_value,
// test_prompt_returns_optional_whitespace_to_palette_then_omits
async function anOptionalBlankInThePaletteEditorReturnsToAFreshPalette(): Promise<void> {
  const answers: [Type, string, unknown][] = [
    [nullable(ValueKind.STRING), "ready", "ready"],
    [nullable(ValueKind.LIST_ANY), "[1]", [1n]],
    [nullable(ValueKind.OBJECT), "{}", {}],
  ];
  for (const [type, raw, value] of answers) {
    for (const [then, expected] of [
      ["↩ Omit input", {}],
      ["Set null", { value: null }],
      ["Enter value", { value }],
    ] as const) {
      const { input, terminal, view } = setup();
      const answer = terminal.promptInputs([decl("value", type)], {});
      await choose(input, view, "Enter value");
      input.write("   " + ENTER);
      const palette = await select(view);
      assert.deepEqual(palette.titles, ["↩ Omit input", "Set null", "Enter value"]);
      assert.equal(palette.marked, 0);
      await choose(input, view, then);
      if (then === "Enter value") input.write(raw + ENTER);
      assert.deepEqual(await answer, expected);
    }
  }
}

// test_nullable_open_set_null_does_not_open_editor
async function setNullInThePaletteOpensNoEditor(): Promise<void> {
  const cases: [Type, boolean][] = [
    ...NULLABLE_OPEN.map((type): [Type, boolean] => [type, false]),
    [nullable(ValueKind.STRING), true],
  ];
  for (const [type, required] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("value", type, { required })], {});
    await choose(input, view, "Set null");
    const gathered = await answer;
    assert.deepEqual(gathered, { value: null });
    assert.ok(Object.hasOwn(gathered!, "value"));
    assert.equal(view().rows.length, 1);
    assert.ok(view().rows[0]!.endsWith(" Set null"));
  }
}

// test_required_nullable_scalar_editor_preserves_blank
async function aRequiredNullableStringEditorKeepsABlank(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("value", nullable(ValueKind.STRING), { required: true })], {});
  const palette = await select(view);
  assert.deepEqual(palette.titles, ["Set null", "Enter value"]);
  assert.equal(palette.marked, 1);
  input.write(ENTER + ENTER);
  assert.deepEqual(await answer, { value: "" });
}

// test_required_nullable_structured_editor_retries_blank
async function aRequiredNullableListEditorRefusesABlank(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("value", nullable(ValueKind.LIST_ANY), { required: true })], {});
  input.write(ENTER + " " + ENTER);
  await tick();
  assert.equal(view().rows.at(-1), "Invalid JSON at line 1, column 2: Expecting value");
  input.write(CTRL_U + "[]" + ENTER);
  assert.deepEqual(await answer, { value: [] });
}

/** A required string, an optional date, a defaulted int and a required Literal. */
function labelDecls(): Parameter[] {
  return [
    decl("topic", STR, { required: true }),
    decl("as_of", nullable(ValueKind.DATE)),
    decl("window", Type.scalar(ValueKind.INTEGER), { hasDefault: true, default: 30n }),
    decl("mode", literal(["fast", "slow"]), { required: true }),
  ];
}

// test_prompt_labels_carry_marks
async function promptLabelsCarryTheTypeAndTheMark(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(labelDecls(), {});
  input.write("ACME" + ENTER + ENTER + ENTER);
  await choose(input, view, "fast");
  assert.deepEqual(await answer, { topic: "ACME", mode: "fast" });
  assert.deepEqual(view().rows, [
    "? topic (str) * ACME",
    "? as_of (Optional(date)) [optional] e.g. 2026-05-21 ↩ Omit input",
    "? window (int) [default: 30]",
    "? mode (Literal(fast, slow)) * fast",
  ]);
}

// test_prompt_skips_already_supplied, test_gather_treats_present_falsey_values_as_supplied_without_questionary
async function suppliedInputsAreSkippedAndNothingMissingDrawsNothing(): Promise<void> {
  const active: boolean[] = [];
  const { output, terminal } = setup({ onActive: (on) => active.push(on) });
  const supplied = { topic: "x", as_of: "2026-01-01", window: 5n, mode: "fast" };
  assert.deepEqual(await terminal.promptInputs(labelDecls(), supplied), {});
  const falsey = [
    decl("none", nullable(ValueKind.STRING), { required: true }),
    decl("false", BOOL, { required: true }),
    decl("zero", Type.scalar(ValueKind.INTEGER), { required: true }),
    decl("blank", STR, { required: true }),
  ];
  assert.deepEqual(await terminal.promptInputs(falsey, { none: null, false: false, zero: 0n, blank: "" }), {});
  assert.equal(output.text, "");
  assert.deepEqual(active, []);
  // Only the missing input is asked.
  const { input, terminal: other, view } = setup();
  const answer = other.promptInputs(labelDecls(), { topic: "x", window: 5n, mode: "fast" });
  input.write(ENTER);
  assert.deepEqual(await answer, {});
  assert.deepEqual(view().rows, ["? as_of (Optional(date)) [optional] e.g. 2026-05-21 ↩ Omit input"]);
}

// test_prompt_accepting_omission_preserves_start_owned_defaults (the prompt half; START applying
// the defaults is the run's)
async function acceptingEveryPreselectionOmitsEachInput(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(
    [
      decl("items", LIST_STR, { hasDefault: true, default: ["a", "b"] }),
      decl("mode", literal(["z", "a"]), { hasDefault: true, default: "z" }),
      decl("flag", nullable(ValueKind.BOOLEAN)),
    ],
    {},
  );
  await tick();
  assert.ok(view().rows[0]!.startsWith("? items (list(str)) [default: "), view().rows[0]);
  assert.ok(view().rows[0]!.endsWith(" Enter a JSON array"), view().rows[0]);
  input.write(ENTER);
  const mode = await select(view);
  assert.deepEqual(mode.titles, ["↩ Use declared default", "Set value: a", "Set value: z"]);
  assert.equal(mode.marked, 0);
  input.write(ENTER);
  const flag = await select(view);
  assert.deepEqual(flag.titles, ["↩ Omit input", "Set null", "true", "false"]);
  assert.equal(flag.marked, 0);
  input.write(ENTER);
  assert.deepEqual(await answer, {});
}

// test_prompt_collects_false_enum_and_text_overrides
async function optionalSelectsAndADefaultedStringCollectAnswers(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(
    [
      decl("flag", nullable(ValueKind.BOOLEAN)),
      decl("mode", literal(["fast", "slow"]), { hasDefault: true, default: "fast" }),
      decl("note", STR, { hasDefault: true, default: "original" }),
    ],
    {},
  );
  await choose(input, view, "false");
  await choose(input, view, "Set value: slow");
  await tick();
  assert.deepEqual(view().rows.at(-1), "? note (str) [default: original]");
  input.write("replacement" + ENTER);
  assert.deepEqual(await answer, { flag: false, mode: "slow", note: "replacement" });
}

// test_finite_prompt_set_null_gathers_present_none
async function setNullInAFiniteSelectGivesAPresentNull(): Promise<void> {
  const cases: [Type, boolean][] = [
    [nullable(ValueKind.BOOLEAN), false],
    [nullable(ValueKind.BOOLEAN), true],
    [literal(["fast", "null"], { nullable: true }), false],
    [literal(["fast", "null"], { nullable: true }), true],
    [Type.scalar(ValueKind.NONE), true],
    [nullable(ValueKind.NONE), false],
  ];
  for (const [type, required] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("value", type, { required })], {});
    await choose(input, view, "Set null");
    assert.deepEqual(await answer, { value: null });
    assert.equal(view().rows.length, 1);
  }
}

// test_null_only_prompt_offers_controls_but_no_value_editor
async function aNullOnlyInputOffersControlsAndNoEditor(): Promise<void> {
  const cases: [Parameter, string[]][] = [
    [decl("nothing", Type.scalar(ValueKind.NONE), { required: true }), ["Set null"]],
    [decl("optional_nothing", nullable(ValueKind.NONE)), ["↩ Omit input", "Set null"]],
    [
      decl("default_nothing", nullable(ValueKind.NONE), { hasDefault: true, default: null }),
      ["↩ Use declared default", "Set null"],
    ],
  ];
  for (const [parameter, titles] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([parameter], {});
    const palette = await select(view);
    assert.deepEqual(palette.titles, titles);
    assert.equal(palette.marked, 0);
    await choose(input, view, "Set null");
    assert.deepEqual(await answer, { [parameter.name]: null });
  }
}

// test_nullable_finite_prompt_omission_stays_absent
async function omissionInANullableFiniteSelectLeavesTheKeyAbsent(): Promise<void> {
  const parameters = [
    decl("flag", nullable(ValueKind.BOOLEAN), { hasDefault: true, default: true }),
    decl("mode", literal(["fast", "null"], { nullable: true }), { hasDefault: true, default: "fast" }),
    decl("nothing", nullable(ValueKind.NONE), { hasDefault: true, default: null }),
  ];
  for (const parameter of parameters) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([parameter], {});
    await choose(input, view, "↩ Use declared default");
    assert.deepEqual(await answer, {});
  }
}

// test_nullable_literal_distinguishes_null_tag_from_set_null
async function aLiteralTagNamedNullStaysTheString(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(
    [decl("mode", literal(["null", "other"], { nullable: true }), { hasDefault: true, default: "other" })],
    {},
  );
  const palette = await select(view);
  assert.deepEqual(palette.titles, ["↩ Use declared default", "Set null", "Set value: null", "Set value: other"]);
  await choose(input, view, "Set value: null");
  assert.deepEqual(await answer, { mode: "null" });
}

// test_nonnullable_finite_default_does_not_offer_set_null
async function aNonNullableBoolOrLiteralOffersNoSetNull(): Promise<void> {
  const cases: [Parameter, string[]][] = [
    [decl("flag", BOOL, { hasDefault: true, default: true }), ["↩ Use declared default", "true", "false"]],
    [
      decl("mode", literal(["fast", "slow"]), { hasDefault: true, default: "fast" }),
      ["↩ Use declared default", "Set value: fast", "Set value: slow"],
    ],
  ];
  for (const [parameter, titles] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([parameter], {});
    const palette = await select(view);
    assert.deepEqual(palette.titles, titles);
    assert.equal(palette.marked, 0);
    input.write(ENTER);
    assert.deepEqual(await answer, {});
  }
}

// test_required_null_capable_prompt_preserves_safe_default
async function aRequiredNullCapableSelectPreselectsASafeChoice(): Promise<void> {
  const cases: [Parameter, unknown, string][] = [
    [decl("flag", nullable(ValueKind.BOOLEAN), { required: true }), false, "false"],
    [decl("mode", literal(["null", "alpha"], { nullable: true }), { required: true }), "alpha", "Set value: alpha"],
    [decl("nothing", Type.scalar(ValueKind.NONE), { required: true }), null, "Set null"],
  ];
  for (const [parameter, expected, title] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([parameter], {});
    const palette = await select(view);
    assert.equal(palette.titles[palette.marked], title);
    input.write(ENTER);
    assert.deepEqual(await answer, { [parameter.name]: expected });
  }
}

// test_prompt_distinguishes_literal_values_from_omission_labels
async function aLiteralTagThatLooksLikeAControlIsAValue(): Promise<void> {
  const collision = "↩ Use declared default";
  const { input, terminal, view } = setup({ columns: 120 });
  const answer = terminal.promptInputs(
    [decl("mode", literal([collision, "ordinary"]), { hasDefault: true, default: "ordinary" })],
    {},
  );
  const palette = await select(view);
  assert.deepEqual(palette.titles, [collision, "Set value: ordinary", `Set value: ${collision}`]);
  await choose(input, view, `Set value: ${collision}`);
  assert.deepEqual(await answer, { mode: collision });
}

async function literalTagsAreSortedByCodePoint(): Promise<void> {
  const { input, terminal, view } = setup();
  // UTF-16 order would put the astral U+1F600 (a surrogate pair) before U+FF5E.
  const answer = terminal.promptInputs([decl("mode", literal(["😀", "～", "b", "a"]), { required: true })], {});
  const palette = await select(view);
  assert.deepEqual(palette.titles, ["a", "b", "～", "😀"]);
  input.write(ENTER);
  assert.deepEqual(await answer, { mode: "a" });
}

// test_prompt_distinguishes_cancellation_from_omission
async function ctrlCInABoolSelectReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.promptInputs([decl("flag", nullable(ValueKind.BOOLEAN))], {});
  input.write(CTRL_C);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["interrupt"]);
}

// test_prompt_leaves_explicit_null_defaults_omitted_for_every_widget_kind
async function explicitNullDefaultsAreOmittedByThePreselection(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs(
    [
      decl("maybe", nullable(ValueKind.INTEGER), { hasDefault: true, default: null }),
      decl("flag", nullable(ValueKind.BOOLEAN), { hasDefault: true, default: null }),
      decl("mode", literal(["fast", "slow"], { nullable: true }), { hasDefault: true, default: null }),
    ],
    {},
  );
  input.write(ENTER + ENTER + ENTER);
  assert.deepEqual(await answer, {});
  assert.deepEqual(view().rows, [
    "? maybe (Optional(int)) [default: null] ↩ Use declared default",
    "? flag (Optional(bool)) [default: null] ↩ Use declared default",
    "? mode (Optional(Literal(fast, slow))) [default: null] ↩ Use declared default",
  ]);
}

// test_any_prompt_uses_the_scalar_text_editor_without_set_null, test_optional_any_prompt_offers_set_null
async function anAnyInputIsFreeTextThatStaysText(): Promise<void> {
  for (const required of [true, false]) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("value", Type.scalar(ValueKind.ANY), { required })], {});
    await tick();
    assert.deepEqual(view().rows, [`? value (Any) ${required ? "*" : "[optional]"}`]);
    input.write("3" + ENTER);
    assert.deepEqual(await answer, { value: "3" });
  }
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("value", nullable(ValueKind.ANY))], {});
  const palette = await select(view);
  assert.deepEqual(palette.titles, ["↩ Omit input", "Set null", "Enter value"]);
  await choose(input, view, "Set null");
  assert.deepEqual(await answer, { value: null });
}

async function anUntypedInputIsFreeText(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.promptInputs([decl("name", undefined, { required: true }), decl("other")], {});
  input.write(" raw " + ENTER + "  " + ENTER);
  assert.deepEqual(await answer, { name: " raw " });
  assert.deepEqual(view().rows, ["? name *  raw", "? other [optional]"]);
}

async function theEndOfInputInAConfirmReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.promptInputs([decl("ok", BOOL, { required: true })], {});
  await tick();
  input.end();
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output, exitHooks);
}

async function aClosedTerminalPromptsNothing(): Promise<void> {
  const { output, terminal } = setup();
  terminal.close();
  assert.equal(await terminal.promptInputs([decl("topic", STR, { required: true })], {}), null);
  assert.equal(output.text, "");
}

async function aFailingOutputFailsTheCall(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup({ throwOn: 2 });
  await assert.rejects(
    terminal.promptInputs([decl("topic", STR, { required: true })], {}),
    (error) => error === output.failure,
  );
  assertRestored(input, output, exitHooks, false);
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
  "a required bool, Literal and string are asked in one turn": aRequiredBoolLiteralAndStringAreAskedInOneTurn,
  "a required bool confirms with y or n at once and ENTER says no": aRequiredBoolConfirmsWithYOrNAtOnceAndEnterSaysNo,
  "a list editor shows the error under it and stays open": aListEditorShowsTheErrorUnderItAndStaysOpen,
  "an object editor returns the native object": anObjectEditorReturnsTheNativeObject,
  "an optional or defaulted direct editor omits on a blank": anOptionalOrDefaultedDirectEditorOmitsOnABlank,
  "a required list editor refuses a blank": aRequiredListEditorRefusesABlank,
  "Ctrl-C in a list editor returns null": ctrlCInAListEditorReturnsNull,
  "a nullable list or object editor refuses a textual null": aNullableListOrObjectEditorRefusesATextualNull,
  "the text null in a nullable string editor stays a string": theTextNullInANullableStringEditorStaysAString,
  "cancelling at the palette or its editor returns null": cancellingAtThePaletteOrItsEditorReturnsNull,
  "an optional blank in the palette's editor returns to a fresh palette":
    anOptionalBlankInThePaletteEditorReturnsToAFreshPalette,
  "Set null in the palette opens no editor": setNullInThePaletteOpensNoEditor,
  "a required nullable string editor keeps a blank": aRequiredNullableStringEditorKeepsABlank,
  "a required nullable list editor refuses a blank": aRequiredNullableListEditorRefusesABlank,
  "prompt labels carry the type and the mark": promptLabelsCarryTheTypeAndTheMark,
  "supplied inputs are skipped and nothing missing draws nothing":
    suppliedInputsAreSkippedAndNothingMissingDrawsNothing,
  "accepting every preselection omits each input": acceptingEveryPreselectionOmitsEachInput,
  "optional selects and a defaulted string collect answers": optionalSelectsAndADefaultedStringCollectAnswers,
  "Set null in a finite select gives a present null": setNullInAFiniteSelectGivesAPresentNull,
  "a null-only input offers controls and no editor": aNullOnlyInputOffersControlsAndNoEditor,
  "omission in a nullable finite select leaves the key absent": omissionInANullableFiniteSelectLeavesTheKeyAbsent,
  "a Literal tag named null stays the string": aLiteralTagNamedNullStaysTheString,
  "a non-nullable bool or Literal offers no Set null": aNonNullableBoolOrLiteralOffersNoSetNull,
  "a required null-capable select preselects a safe choice": aRequiredNullCapableSelectPreselectsASafeChoice,
  "a Literal tag that looks like a control is a value": aLiteralTagThatLooksLikeAControlIsAValue,
  "Literal tags are sorted by code point": literalTagsAreSortedByCodePoint,
  "Ctrl-C in a bool select returns null": ctrlCInABoolSelectReturnsNull,
  "explicit null defaults are omitted by the preselection": explicitNullDefaultsAreOmittedByThePreselection,
  "an Any input is free text that stays text": anAnyInputIsFreeTextThatStaysText,
  "an untyped input is free text": anUntypedInputIsFreeText,
  "the end of input in a confirm returns null": theEndOfInputInAConfirmReturnsNull,
  "a closed terminal prompts nothing": aClosedTerminalPromptsNothing,
  "a failing output fails the call": aFailingOutputFailsTheCall,
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
