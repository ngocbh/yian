/**
 * Terminal scenarios written once for both runtimes: `Terminal` driven through a fake TTY, with a
 * small screen replayer that turns the drawn output back into the rows a terminal would show. They
 * are plain async functions using `node:assert/strict`, so the same checks run as Bun tests
 * (`terminal.test.ts`) and on Node, where this module is the main one and prints one JSON line per
 * scenario.
 */

import assert from "node:assert/strict";
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
  override write(chunk: any, encoding?: any, callback?: any): boolean {
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
    else if (command === "G") col = Math.min(n, columns) - 1;
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

/** Let the streams deliver what was written. */
export function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 10));
}

/** A fresh fake terminal: its input, output and the `Terminal` over them. */
function setup(
  options: { tty?: boolean; columns?: number; throwOn?: number } & Pick<TerminalOptions, "onStop"> = {},
): { input: FakeInput; output: FakeOutput; terminal: Terminal; view: () => Screen } {
  const input = new FakeInput({ tty: options.tty });
  const output = new FakeOutput({ columns: options.columns, throwOn: options.throwOn });
  const terminal = new Terminal({ input, output, onStop: options.onStop });
  return { input, output, terminal, view: () => screen(output.text, output.columns) };
}

/** Raw mode was set once and restored, and the cursor and bracketed paste were the last modes written. */
function assertRestored(input: FakeInput, output: FakeOutput): void {
  if (input.isTTY) assert.deepEqual(input.rawCalls, [true, false]);
  assert.equal(input.isRaw, false);
  assert.deepEqual(screen(output.text).modes.slice(-2).sort(), ["?2004l", "?25h"]);
}

async function typingALineReturnsIt(): Promise<void> {
  const { input, output, terminal, view } = setup();
  const answer = terminal.askText("Name:");
  input.write("hello");
  await tick();
  assert.deepEqual(view().rows, ["? Name: hello"]);
  assert.deepEqual(view().cursor, { row: 0, col: 13 });
  input.write(ENTER);
  assert.equal(await answer, "hello");
  assert.deepEqual(view().rows, ["? Name: hello"]);
  assert.deepEqual(view().cursor, { row: 1, col: 0 });
  assertRestored(input, output);
}

async function anInitialValueCanBeEdited(): Promise<void> {
  const { input, output, terminal, view } = setup();
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
  assertRestored(input, output);
}

async function aBlankAnswerReturnsAnEmptyString(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askText("Name:");
  input.write(ENTER);
  assert.equal(await answer, "");
  assert.deepEqual(view().rows, ["? Name:"]);
}

async function aSecretIsMaskedAndNeverEchoed(): Promise<void> {
  const { input, output, terminal, view } = setup();
  const answer = terminal.askText("Key:", { secret: true });
  input.write("s3cret");
  await tick();
  assert.deepEqual(view().rows, ["? Key: ******"]);
  input.write(ENTER);
  assert.equal(await answer, "s3cret");
  assert.deepEqual(view().rows, ["? Key:"]);
  for (const piece of ["s3", "3c", "cr", "re", "et"]) assert.ok(!output.text.includes(piece), piece);
  assertRestored(input, output);
}

async function ctrlCReturnsNullAndCallsOnStopOnce(): Promise<void> {
  const stops: string[] = [];
  const rawWhenStopped: boolean[] = [];
  const { input, output, terminal, view } = setup({
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
  assertRestored(input, output);
  assert.equal(await terminal.askText("Again:"), null);
  assert.deepEqual(stops, ["interrupt"]);
}

async function ctrlDOnAnEmptyLineIsTheEndOfInput(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.write(CTRL_D);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output);
}

async function theEndOfInputReturnsNull(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.end();
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output);
}

async function ctrlDOnANonEmptyLineIsIgnored(): Promise<void> {
  const stops: string[] = [];
  const { input, terminal } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.askText("Name:");
  input.write("ab" + CTRL_D + ENTER);
  assert.equal(await answer, "ab");
  assert.deepEqual(stops, []);
}

async function aThrowingWriteRestoresTheTerminalAndRejects(): Promise<void> {
  // Write 1 sets the modes, write 2 draws the first frame, write 3 is the redraw after a key.
  const { input, output, terminal } = setup({ throwOn: 3 });
  const answer = terminal.askText("Name:");
  input.write("a");
  await assert.rejects(answer, (error) => error === output.failure);
  assertRestored(input, output);
}

async function aNonTtyInputStillReadsALine(): Promise<void> {
  const { input, output, terminal } = setup({ tty: false });
  const answer = terminal.askText("Name:");
  input.write("answer\n");
  assert.equal(await answer, "answer");
  assert.deepEqual(input.rawCalls, []);
  assertRestored(input, output);
}

async function aThrowingOnStopStillRestoresTheTerminal(): Promise<void> {
  const { input, output, terminal } = setup({
    onStop: () => {
      throw new Error("the handler broke");
    },
  });
  const answer = terminal.askText("Name:");
  input.write(CTRL_C);
  assert.equal(await answer, null);
  assertRestored(input, output);
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
  "a throwing write restores the terminal and rejects": aThrowingWriteRestoresTheTerminalAndRejects,
  "a non-TTY input still reads a line": aNonTtyInputStillReadsALine,
  "a throwing onStop still restores the terminal": aThrowingOnStopStillRestoresTheTerminal,
  "a bracketed paste keeps its newlines": aBracketedPasteKeepsItsNewlines,
  "wide characters redraw without stale rows": wideCharactersRedrawWithoutStaleRows,
  "a long line redraws without stale rows": aLongLineRedrawsWithoutStaleRows,
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
  // No process.exit: a leaked listener or timer then shows as a hang.
  process.exitCode = (await main()) ? 1 : 0;
}
