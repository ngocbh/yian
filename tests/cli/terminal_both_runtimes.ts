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

import { LineEditor, Terminal, type TerminalOptions, TerminalStopped } from "../../src/cli/terminal.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { parseQuestions } from "../../src/nodes/tool/questions.ts";
import { ToolContext } from "../../src/tools/context.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";

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
export const CTRL_N = "\x0e";
export const CTRL_P = "\x10";
export const CTRL_Q = "\x11";
export const CTRL_U = "\x15";
export const BACKSPACE = "\x7f";
export const SPACE = " ";
export const ESCAPE = "\x1b";
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

/**
 * An output collecting what is written, `columns` wide and a TTY unless `tty` is false; with
 * `throwOn`, that write (1-based) throws once.
 */
export class FakeOutput extends Writable {
  text = "";
  columns: number;
  rows: number | undefined;
  isTTY: boolean;
  writes = 0;
  readonly throwOn: number | undefined;
  readonly failure = new Error("the output broke");

  constructor({ columns = 80, throwOn, tty = true }: { columns?: number; throwOn?: number; tty?: boolean } = {}) {
    super();
    this.columns = columns;
    this.isTTY = tty;
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
  /** Each private mode written, in order, such as `?25l` or `?2004h`; not synchronized output. */
  modes: string[];
  /** Each start (`?2026h`) and end (`?2026l`) of synchronized output, in order. */
  syncs: string[];
  /** The rows an erase of the whole screen (`ESC[2J`) moved off it, oldest first. */
  scrollback: string[];
}

/** Display width of one character: 2 for the CJK and fullwidth ranges the scenarios use. */
function charWidth(char: string): number {
  const code = char.codePointAt(0)!;
  if (/^[\p{Mn}\p{Me}\u200d\ufe00-\ufe0f]$/u.test(char)) return 0;
  return (code >= 0x3000 && code <= 0x9fff) || (code >= 0xff00 && code <= 0xff60) ? 2 : 1;
}

/**
 * Replay `text` on a `columns`-wide terminal. It knows text, `\r`, `\n` (down one row, as with
 * output newline translation off), `ESC[nA`, `ESC[J`, `ESC[2J` (its rows go to the scrollback, the
 * cursor stays), `ESC[K`, `ESC[nG`, `ESC[r;cH` and the alternate screen (`ESC[?1049h` and `l`, which gives back the main screen and its cursor);
 * colours are ignored. Writing past the last column wraps on the next character, as terminals do.
 */
export function screen(text: string, columns = 80): Screen {
  let rows: string[][] = [[]];
  const scrollback: string[][] = [];
  const modes: string[] = [];
  const syncs: string[] = [];
  let row = 0;
  let col = 0;
  // The main screen while the alternate one is up.
  let main: { rows: string[][]; row: number; col: number } | null = null;
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
        if (width === 0) {
          // Drawn on the character before it.
          if (col > 0) line()[col - 1] += char;
          continue;
        }
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
    if (priv === "?") {
      (args === "2026" ? syncs : modes).push(`?${args}${command}`);
      if (args === "1049" && command === "h" && main === null) {
        main = { rows, row, col };
        rows = [[]];
        row = 0;
        col = 0;
      } else if (args === "1049" && command === "l" && main !== null) {
        ({ rows, row, col } = main);
        main = null;
      }
    } else if (command === "H") {
      const [down = 1, across = 1] = (args ?? "").split(";").map((part) => (part === "" ? 1 : Number(part)));
      row = down - 1;
      col = across - 1;
    }
    else if (command === "A") row = Math.max(0, row - n);
    else if (command === "G") col = Math.max(1, Math.min(n, columns)) - 1;
    else if (command === "K") line().length = Math.min(line().length, col);
    else if (command === "J" && n === 2) {
      scrollback.push(...rows);
      rows = [[]];
    } else if (command === "J") {
      line().length = Math.min(line().length, col);
      rows.length = row + 1;
    } else if (command !== "m") throw new Error(`the replayer doesn't know ESC[${args}${command}`);
  }
  const shown = rows.map((cells) => cells.join("").trimEnd());
  while (shown.length > 0 && shown[shown.length - 1] === "") shown.pop();
  const scrolled = scrollback.map((cells) => cells.join("").trimEnd());
  return { rows: shown, cursor: { row, col }, modes, syncs, scrollback: scrolled };
}

/** Let the streams deliver what was written: they do it in next-tick callbacks and microtasks. */
export function tick(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** A fresh fake terminal: its input, output and the `Terminal` over them. */
function setup(
  options: { tty?: boolean; columns?: number; throwOn?: number } & Partial<Pick<
    TerminalOptions,
    "onStop" | "onActive" | "now" | "noColor"
  >> = {},
): { input: FakeInput; output: FakeOutput; terminal: Terminal; view: () => Screen; exitHooks: number } {
  const input = new FakeInput({ tty: options.tty });
  const output = new FakeOutput({ columns: options.columns, throwOn: options.throwOn });
  const terminal = new Terminal({
    input,
    output,
    onStop: options.onStop,
    onActive: options.onActive,
    noColor: options.noColor,
    ...(options.now === undefined ? {} : { now: options.now }),
  });
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

async function readLineReturnsEachLineThenTheEndOfInput(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, view, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const first = terminal.readLine("You:");
  input.write("hi");
  await tick();
  assert.deepEqual(view().rows, ["> You: hi"]);
  input.write(ENTER);
  assert.deepEqual(await first, { value: "hi" });
  assertRestored(input, output, exitHooks);
  const second = terminal.readLine("You:");
  input.end();
  assert.deepEqual(await second, { eof: true });
  // The caller handles its own stops: `onStop` hears of none.
  assert.deepEqual(stops, []);
  assert.deepEqual(await terminal.readLine("You:"), { eof: true });
}

async function ctrlCAtReadLineKeepsTheTerminalOpen(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const first = terminal.readLine("You:");
  input.write("abc" + CTRL_C);
  assert.deepEqual(await first, { interrupt: true });
  assert.deepEqual(stops, []);
  assertRestored(input, output, exitHooks);
  const second = terminal.readLine("You:");
  input.write("ok" + ENTER);
  assert.deepEqual(await second, { value: "ok" });
  // A prompt other than `readLine` still closes the terminal on Ctrl-C.
  const name = terminal.askText("Name:");
  input.write(CTRL_C);
  assert.equal(await name, null);
  assert.deepEqual(stops, ["interrupt"]);
  assert.deepEqual(await terminal.readLine("You:"), { eof: true });
}

async function anAbortedReadLineKeepsLaterKeys(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup();
  assert.deepEqual(await terminal.readLine("You:", { signal: AbortSignal.abort(new Error("woke")) }), {
    aborted: true,
    draft: "",
  });
  assert.equal(output.text, "");
  const controller = new AbortController();
  const line = terminal.readLine("You:", { signal: controller.signal });
  await tick();
  controller.abort(new Error("woke"));
  assert.deepEqual(await line, { aborted: true, draft: "" });
  assertRestored(input, output, exitHooks);
  // Keys typed while no prompt is up wait for the next one.
  input.write("later" + ENTER);
  await tick();
  assert.deepEqual(await terminal.readLine("You:"), { value: "later" });
}

async function anAbortedReadLineReturnsItsDraftForTheNextPrompt(): Promise<void> {
  const { input, terminal, view } = setup();
  const controller = new AbortController();
  const line = terminal.readLine("You:", { signal: controller.signal });
  input.write("half");
  await tick();
  controller.abort(new Error("woke"));
  assert.deepEqual(await line, { aborted: true, draft: "half" });
  // The aborted prompt is erased, not collapsed into a line.
  assert.deepEqual(view().rows, []);
  const next = terminal.readLine("You:", { initial: "half" });
  await tick();
  assert.deepEqual(view().rows, ["> You: half"]);
  input.write(" done" + ENTER);
  assert.deepEqual(await next, { value: "half done" });
  // Aborted before it draws, the draft is the initial text.
  const early = terminal.readLine("You:", { initial: "kept", signal: AbortSignal.abort(new Error("woke")) });
  assert.deepEqual(await early, { aborted: true, draft: "kept" });
}

async function aYieldingReadLineGivesWayToAPromptQueuedBehindIt(): Promise<void> {
  let clock = 0;
  const { input, terminal, view } = setup({ tty: true, now: () => clock });
  const line = terminal.readLine("You:", { yields: true });
  input.write("half");
  await tick();
  const answer = terminal.approve(approval());
  // An Enter typed as the prompt gives way, meant for the line, answers nothing.
  input.write(ENTER);
  assert.deepEqual(await line, { yielded: true, draft: "half" });
  await tick();
  assert.equal(view().rows.at(-3), "? Write a.yaml (Use arrow keys)");
  // The next prompt waits behind the approval, which was queued first.
  const next = terminal.readLine("You:", { initial: "half", yields: true });
  // Keys just after the card opens are dropped too; once it has been up a while, they count.
  input.write(ENTER);
  await tick();
  assert.equal(view().rows.at(-3), "? Write a.yaml (Use arrow keys)");
  clock += 1000;
  input.write(ENTER);
  assert.equal(await answer, "Approve");
  await tick();
  input.write(" done" + ENTER);
  assert.deepEqual(await next, { value: "half done" });
  // A line prompt behind it doesn't take it down, and neither does one without `yields`.
  const kept = terminal.readLine("You:", { yields: true });
  const behind = terminal.readLine("You:");
  await tick();
  input.write("one" + ENTER);
  assert.deepEqual(await kept, { value: "one" });
  const plain = terminal.readLine("You:");
  const asked = terminal.askText("Name:");
  input.write("two" + ENTER);
  assert.deepEqual(await behind, { value: "two" });
  input.write("three" + ENTER);
  assert.deepEqual(await plain, { value: "three" });
  input.write("ada" + ENTER);
  assert.equal(await asked, "ada");
  // A prompt queued behind it before it is up takes it down as it comes up; one ahead doesn't.
  const ahead = terminal.askText("Name:");
  await tick();
  const queued = terminal.readLine("You:", { initial: "kept", yields: true });
  const later = terminal.approve(approval());
  await tick();
  input.write("bo" + ENTER);
  assert.equal(await ahead, "bo");
  assert.deepEqual(await queued, { yielded: true, draft: "kept" });
  clock += 1000;
  input.write(ENTER);
  assert.equal(await later, "Approve");
}

async function aPromptOpeningWhileTheLiveBoxTakesKeysDropsThemForAWhile(): Promise<void> {
  let clock = 0;
  const { input, terminal, view } = setup({ tty: true, now: () => clock });
  const editor = new LineEditor({ columns: () => 60, hint: "" });
  const sent: string[] = [];
  const watch = new AbortController();
  terminal.watchKeys(
    watch.signal,
    () => {},
    (key) => {
      const action = editor.key(key);
      if (action !== null && "value" in action) sent.push(action.value);
      return true;
    },
  );
  input.write("steer");
  await tick();
  clock += 100;
  // The card opens as the user presses Enter, meant to send the line.
  const answer = terminal.approve(approval());
  await tick();
  input.write(ENTER);
  await tick();
  assert.equal(view().rows.at(-3), "? Write a.yaml (Use arrow keys)");
  clock += 1000;
  input.write(ENTER);
  assert.equal(await answer, "Approve");
  assert.deepEqual(sent, []);
  // Long after the last key in the box, while it is still up, a card drops keys at first too.
  clock += 1000;
  const later = terminal.approve(approval());
  await tick();
  input.write(ENTER);
  await tick();
  assert.equal(view().rows.at(-3), "? Write a.yaml (Use arrow keys)");
  // Ctrl-C still reaches it, as the reject.
  input.write(CTRL_C);
  assert.equal(await later, "Reject");
  // A paste that ends in the grace is dropped whole, like any key; after it, a paste counts.
  clock += 1000;
  const question = terminal.askText("Name:");
  await tick();
  input.write(PASTE_START + "meant\rfor the box" + PASTE_END);
  await tick();
  clock += 1000;
  input.write(PASTE_START + "ada" + PASTE_END + ENTER);
  assert.equal(await question, "ada");
  watch.abort();
  // With the box down, a card takes keys at once.
  const after = terminal.approve(approval());
  await tick();
  input.write(ENTER);
  assert.equal(await after, "Approve");
}

async function aLinePromptAfterAWithdrawnCardTakesKeysAtOnce(): Promise<void> {
  const { input, terminal } = setup({ tty: true, now: () => 0 });
  const line = terminal.readLine("You:", { yields: true });
  await tick();
  const withdrawn = new AbortController();
  const card = terminal.approve({ ...approval(), signal: withdrawn.signal });
  assert.deepEqual(await line, { yielded: true, draft: "" });
  // The card is withdrawn before it opens; the line comes back and takes the first keys.
  withdrawn.abort();
  assert.equal(await card, null);
  const next = terminal.readLine("You:", { yields: true });
  await tick();
  input.write("hi" + ENTER);
  assert.deepEqual(await next, { value: "hi" });
}

async function approvePreselectsTheChoiceItNames(): Promise<void> {
  const { input, terminal, view } = setup();
  const choices = ["Go", "Save", "Cancel"];
  const answer = terminal.approve({ ...approval(), choices, preselect: "Save" });
  await tick();
  assert.deepEqual(view().rows.slice(-3), ["   Go", " » Save", "   Cancel"]);
  input.write(ENTER);
  assert.equal(await answer, "Save");
}

async function onActiveSaysWhetherATurnLeftRows(): Promise<void> {
  const calls: [boolean, boolean][] = [];
  const { input, terminal } = setup({ onActive: (on, printed) => calls.push([on, printed]) });
  const controller = new AbortController();
  const line = terminal.readLine("You:", { signal: controller.signal });
  input.write("half");
  await tick();
  controller.abort(new Error("woke"));
  await line;
  // Erased, it leaves nothing; entered, it leaves its collapsed line.
  const next = terminal.readLine("You:");
  input.write(ENTER);
  await next;
  assert.deepEqual(calls, [
    [true, false],
    [false, false],
    [true, false],
    [false, true],
  ]);
}

async function escapeAtReadLineReturnsTheDraft(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, view, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const line = terminal.readLine("You:", { escape: true, initial: "half" });
  input.write("x");
  await tick();
  // A lone Escape: readline sends it once its escape timeout passes with no key after it.
  input.write(ESCAPE);
  assert.deepEqual(await line, { escape: true, draft: "halfx" });
  // The prompt is erased, as an aborted one is, and the terminal is given back.
  assert.deepEqual(view().rows, []);
  assertRestored(input, output, exitHooks);
  assert.deepEqual(stops, []);
  // The terminal stays open for the next prompt.
  const next = terminal.readLine("You:", { escape: true, initial: "halfx" });
  input.write(ENTER);
  assert.deepEqual(await next, { value: "halfx" });
}

async function escapeAtReadLineWithoutTheOptionIsIgnored(): Promise<void> {
  const { input, terminal, view } = setup();
  const line = terminal.readLine("You:", { initial: "half" });
  input.write(ESCAPE);
  // Past readline's escape timeout, so the Escape reaches the prompt on its own.
  await new Promise((resolve) => setTimeout(resolve, 600));
  await tick();
  assert.deepEqual(view().rows, ["> You: half"]);
  input.write("a" + ENTER);
  assert.deepEqual(await line, { value: "halfa" });
}

async function watchKeysSendsEscapeAndCtrlCAndBuffersTheRest(): Promise<void> {
  const { input, terminal, exitHooks } = setup();
  const keys: string[] = [];
  const watch = new AbortController();
  terminal.watchKeys(watch.signal, (key) => keys.push(key));
  assert.deepEqual(input.rawCalls, [true]);
  assert.equal(input.isPaused(), false);
  input.write(ESCAPE);
  // Past readline's escape timeout, so the Escape arrives on its own.
  await new Promise((resolve) => setTimeout(resolve, 600));
  input.write(CTRL_C);
  input.write("ab");
  await tick();
  assert.deepEqual(keys, ["escape", "interrupt"]);
  watch.abort();
  assert.deepEqual(input.rawCalls, [true, false]);
  assert.equal(input.isPaused(), true);
  assert.equal(process.listenerCount("exit"), exitHooks);
  // The other keys wait for the next prompt.
  const line = terminal.readLine("You:");
  input.write(ENTER);
  assert.deepEqual(await line, { value: "ab" });
  assert.deepEqual(keys, ["escape", "interrupt"]);
}

async function aPromptDuringWatchTakesTheKeysThenWatchingResumes(): Promise<void> {
  const { input, terminal } = setup();
  const keys: string[] = [];
  const watch = new AbortController();
  terminal.watchKeys(watch.signal, (key) => keys.push(key));
  const answer = terminal.askText("Sure?");
  input.write("yes" + ENTER);
  assert.equal(await answer, "yes");
  // The prompt gives the terminal back raw and flowing, to the watch.
  assert.equal(input.isRaw, true);
  assert.equal(input.isPaused(), false);
  input.write(ESCAPE);
  await new Promise((resolve) => setTimeout(resolve, 600));
  await tick();
  assert.deepEqual(keys, ["escape"]);
  watch.abort();
  assert.equal(input.isRaw, false);
  assert.equal(input.isPaused(), true);
  // A watch that ends while a prompt is up leaves the terminal cooked once the prompt ends.
  const again = new AbortController();
  terminal.watchKeys(again.signal, (key) => keys.push(key));
  const second = terminal.askText("Again?");
  await tick();
  again.abort();
  assert.equal(input.isRaw, true);
  input.write("no" + ENTER);
  assert.equal(await second, "no");
  assert.equal(input.isRaw, false);
  assert.equal(input.isPaused(), true);
  assert.deepEqual(keys, ["escape"]);
}

async function watchKeysOnAPlainOrNonTtyTerminalDoesNothing(): Promise<void> {
  const keys: string[] = [];
  for (const [tty, plain] of [
    [false, false],
    [true, true],
  ] as const) {
    const input = new FakeInput({ tty });
    const terminal = new Terminal({ input, output: new FakeOutput(), plain });
    const watch = new AbortController();
    terminal.watchKeys(watch.signal, (key) => keys.push(key));
    input.write(CTRL_C + ESCAPE);
    await new Promise((resolve) => setTimeout(resolve, 600));
    watch.abort();
    assert.deepEqual(input.rawCalls, []);
    assert.equal(input.listenerCount("keypress"), 0);
  }
  assert.deepEqual(keys, []);
}

async function aPlainTerminalWritesNoEscapesAndSetsNoRawMode(): Promise<void> {
  const input = new FakeInput();
  const output = new FakeOutput();
  const terminal = new Terminal({ input, output, plain: true });
  const line = terminal.readLine("You:");
  input.write("hi" + ENTER);
  assert.deepEqual(await line, { value: "hi" });
  const choice = terminal.askChoice("Pick", ["a", "b"]);
  input.write(DOWN + ENTER);
  assert.equal(await choice, "b");
  assert.deepEqual(input.rawCalls, []);
  assert.ok(!/[\x1b\r]/.test(output.text), JSON.stringify(output.text));
  // Each prompt is written once, as it opens.
  assert.equal(output.text, "> You: \n? Pick (Use arrow keys)\n » a\n   b\n");
}

async function aPlainTerminalSetsRawModeForASecretPrompt(): Promise<void> {
  const input = new FakeInput();
  const output = new FakeOutput();
  const terminal = new Terminal({ input, output, plain: true });
  const key = terminal.askText("Key:", { secret: true });
  input.write("sk-typed" + ENTER);
  assert.equal(await key, "sk-typed");
  // The tty doesn't echo the key, and the plain output draws none of it.
  assert.deepEqual(input.rawCalls, [true, false]);
  assert.equal(output.text, "? Key: \n");
}

async function aPlainTerminalWritesARefusedLinesErrorAndReadsTheNextLineAfresh(): Promise<void> {
  const input = new FakeInput();
  const output = new FakeOutput();
  const terminal = new Terminal({ input, output, plain: true });
  const answer = terminal.promptInputs([decl("items", LIST_STR, { required: true })], {});
  // The cooked tty already echoed each line, so only the prompt and the error are written.
  input.write("[\n");
  input.write('["b"]\n');
  assert.deepEqual(await answer, { items: ["b"] });
  assert.equal(
    output.text,
    "? items (list(str)) * Enter a JSON array \nInvalid JSON at line 1, column 2: Expecting value\n",
  );
}

async function aNonTtyReadLineReadsOneLinePerCall(): Promise<void> {
  const { input, terminal } = setup({ tty: false });
  input.write("a\nb\n");
  assert.deepEqual(await terminal.readLine("You:"), { value: "a" });
  assert.deepEqual(await terminal.readLine("You:"), { value: "b" });
  assert.deepEqual(input.rawCalls, []);
  input.end();
  assert.deepEqual(await terminal.readLine("You:"), { eof: true });
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

const HINT = "/exit to leave · ctrl-c interrupts · esc withdraws";
const FRAME_TOP = `╭${"─".repeat(58)}╮`;
const FRAME_BOTTOM = `╰${"─".repeat(58)}╯`;

/** A framed prompt's row holding `text`, of one-cell characters, 60 columns wide. */
const framed = (text: string) => `│ ${text.padEnd(56)} │`;

async function aFramedReadLineDrawsABoxAndAHintThenCollapsesToTheLine(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup({ columns: 60 });
  output.write("above\r\n");
  const line = terminal.readLine("You:", { frame: true, hint: HINT });
  await tick();
  assert.deepEqual(view().rows, ["above", FRAME_TOP, framed(">"), FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(view().cursor, { row: 2, col: 4 });
  input.write("run hello");
  await tick();
  assert.deepEqual(view().rows, ["above", FRAME_TOP, framed("> run hello"), FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(view().cursor, { row: 2, col: 13 });
  input.write(ENTER);
  assert.deepEqual(await line, { value: "run hello" });
  assert.deepEqual(view().rows, ["above", "> run hello"]);
  assertRestored(input, output, exitHooks);
}

async function aFramedReadLineWithAnEditorKeepsItsBoxEmptiedUnderTheLine(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 60 });
  const editor = new LineEditor({ columns: () => 60, hint: HINT });
  const line = terminal.readLine("You:", { frame: true, editor });
  input.write("hello" + ENTER);
  assert.deepEqual(await line, { value: "hello" });
  // The box stays, emptied, for what draws next to repaint over; the editor holds what was entered.
  assert.deepEqual(view().rows, ["> hello", FRAME_TOP, framed(">"), FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(editor.entered(), ["hello"]);
  assert.equal(editor.text(), "");
  // The line entered is on the user's gray band, across the width.
  const band = `\x1b[48;5;237;38;5;252m\x1b[1m>\x1b[22m hello${" ".repeat(53)}\x1b[0m`;
  assert.ok(output.text.includes(band));
}

async function aFramedReadLineEntersItsLineAtTheWidthOfTheTerminalThen(): Promise<void> {
  const { input, output, terminal } = setup({ columns: 60 });
  const editor = new LineEditor({ columns: () => output.columns, hint: HINT });
  const line = terminal.readLine("You:", { frame: true, editor });
  input.write("hello");
  await tick();
  output.columns = 40;
  input.write(ENTER);
  assert.deepEqual(await line, { value: "hello" });
  assert.ok(output.text.includes(`hello${" ".repeat(33)}\x1b[0m`));
}

async function aFramedReadLineWithNoColorEntersAPlainLine(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 60, noColor: true });
  const editor = new LineEditor({ columns: () => 60, hint: HINT });
  const line = terminal.readLine("You:", { frame: true, editor });
  input.write("hello" + ENTER);
  assert.deepEqual(await line, { value: "hello" });
  assert.equal(view().rows[0], "> hello");
  assert.ok(!output.text.includes("48;5;237"));
}

async function aQuestionsRuleIsAsWideAsTheTerminalWhenItIsDrawn(): Promise<void> {
  const { input, output, terminal } = setup({ columns: 60 });
  const questions = parseQuestions([
    { question: "Why?", header: "Why" },
    { question: "Notes?", header: "Notes" },
  ]);
  const answer = terminal.askUser(request({ question: "Q", questions }), toolContext());
  await tick();
  output.columns = 40;
  input.write("x" + ENTER);
  await tick();
  input.write("y" + ENTER);
  await answer;
  const rule = /Question 2 of 2[^─]*?(─+)/.exec(output.text)!;
  // `Question 2 of 2 · Notes`, a space, and the rule across the rest of the 40 columns.
  assert.equal(rule[1]!.length, 40 - 23 - 1);
}

async function aFramedReadLineFoldsALongLineInsideTheBox(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const line = terminal.readLine("You:", { frame: true, hint: HINT });
  // 54 cells fit a row: the 55th character starts the next, and a wide one that doesn't fit too.
  input.write(`${"a".repeat(53)}日本`);
  await tick();
  const wide = `│   日本${" ".repeat(50)} │`;
  assert.deepEqual(view().rows, [FRAME_TOP, framed(`> ${"a".repeat(53)}`), wide, FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(view().cursor, { row: 2, col: 8 });
  input.write(BACKSPACE.repeat(2) + "b");
  await tick();
  // A full row leaves the cursor a row of its own.
  assert.deepEqual(view().rows, [FRAME_TOP, framed(`> ${"a".repeat(53)}b`), framed(""), FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(view().cursor, { row: 2, col: 4 });
  input.write(ENTER);
  assert.deepEqual(await line, { value: `${"a".repeat(53)}b` });
}

async function aBracketedPasteKeepsItsNewlinesFramed(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write(PASTE_START + "x\ny" + PASTE_END);
  await tick();
  assert.deepEqual(view().rows, [FRAME_TOP, framed("> x⏎y"), FRAME_BOTTOM, `  ${HINT}`]);
  input.write(ENTER);
  assert.deepEqual(await answer, { value: "x\ny" });
}

/** A paste of `count` lines, `line 1` to `line <count>`, each ended as a terminal sends it. */
const pastedLines = (count: number) =>
  Array.from({ length: count }, (_, i) => `line ${i + 1}\r`).join("");

/** What `pastedLines(count)` reaches the editor as: its lines, each ended by a newline. */
const pastedText = (count: number) =>
  Array.from({ length: count }, (_, i) => `line ${i + 1}\n`).join("");

async function aLongPasteInAFramedReadLineIsOneTokenExpandedOnEnter(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write("look " + PASTE_START + pastedLines(40) + PASTE_END);
  await tick();
  const box = framed("> look [Pasted text #1 +40 lines]");
  assert.deepEqual(view().rows, [FRAME_TOP, box, FRAME_BOTTOM, `  ${HINT}`]);
  assert.deepEqual(view().cursor, { row: 1, col: 35 });
  input.write(" please" + ENTER);
  assert.deepEqual(await answer, { value: `look ${pastedText(40)} please` });
  assert.deepEqual(view().rows, ["> look [Pasted text #1 +40 lines] please"]);
}

async function aPastedTokenIsAtomicForTheCaretAndTheDeletingKeys(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write("a" + PASTE_START + pastedLines(5) + PASTE_END + "b");
  await tick();
  assert.deepEqual(view().rows[1], framed("> a[Pasted text #1 +5 lines]b"));
  // Left steps over the token in one press, and text goes in on either side of it.
  input.write(LEFT + LEFT + "<" + RIGHT + ">");
  await tick();
  assert.deepEqual(view().rows[1], framed("> a<[Pasted text #1 +5 lines]>b"));
  assert.deepEqual(view().cursor, { row: 1, col: 32 });
  // Backspace after it and Delete before it take it whole.
  input.write(LEFT + BACKSPACE);
  await tick();
  assert.deepEqual(view().rows[1], framed("> a<>b"));
  input.write(PASTE_START + pastedLines(5) + PASTE_END + LEFT + DELETE);
  await tick();
  assert.deepEqual(view().rows[1], framed("> a<>b"));
  // Ctrl-U takes the tokens before the cursor whole too.
  input.write(PASTE_START + pastedLines(5) + PASTE_END + CTRL_U);
  await tick();
  assert.deepEqual(view().rows[1], framed("> >b"));
  input.write(ENTER);
  assert.deepEqual(await answer, { value: ">b" });
}

async function aShortPasteStaysTextAndALongLineIsAToken(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  // Three lines go in as text.
  input.write(PASTE_START + "x\ry\rz" + PASTE_END + " ");
  await tick();
  assert.deepEqual(view().rows[1], framed("> x⏎y⏎z"));
  // One line longer than the threshold is a token.
  input.write(PASTE_START + "w".repeat(801) + PASTE_END);
  await tick();
  assert.deepEqual(view().rows[1], framed("> x⏎y⏎z [Pasted text #1 +1 line]"));
  input.write(ENTER);
  assert.deepEqual(await answer, { value: `x\ny\nz ${"w".repeat(801)}` });
}

async function twoPastesAreNumberedAndTypedTokenTextStaysText(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write(PASTE_START + pastedLines(4) + PASTE_END + " ");
  input.write(PASTE_START + pastedLines(6) + PASTE_END);
  await tick();
  assert.deepEqual(view().rows[1], framed("> [Pasted text #1 +4 lines] [Pasted text #2 +6 lines]"));
  input.write(CTRL_U + "[Pasted text #1 +4 lines]");
  await tick();
  assert.deepEqual(view().rows[1], framed("> [Pasted text #1 +4 lines]"));
  input.write(ENTER);
  assert.deepEqual(await answer, { value: "[Pasted text #1 +4 lines]" });
}

async function aTokenWiderThanTheBoxIsCut(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 20 });
  const answer = terminal.readLine("You:", { frame: true, hint: "" });
  input.write("ab" + PASTE_START + pastedLines(40) + PASTE_END);
  await tick();
  // 14 cells to a row: the token takes a row of its own, cut to fit, and the cursor the next.
  assert.deepEqual(view().rows.slice(1, 4), [
    `│ > ${"ab".padEnd(14)} │`,
    `│   ${"[Pasted text …".padEnd(14)} │`,
    `│   ${"".padEnd(14)} │`,
  ]);
  assert.deepEqual(view().cursor, { row: 3, col: 4 });
  input.write(ENTER);
  assert.deepEqual(await answer, { value: `ab${pastedText(40)}` });
}

/** Wait longer than a paste may pause before Escape or Ctrl-C can end it. */
const pastPastePause = () => new Promise((resolve) => setTimeout(resolve, 800));

async function aPasteSplitByAPauseGoesInWhole(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  let answered = false;
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  void answer.then(() => (answered = true));
  input.write(PASTE_START + "a\rb");
  await pastPastePause();
  // The rest of the paste: its newlines are still newlines, not Enter.
  input.write("\rc" + PASTE_END);
  await tick();
  assert.equal(answered, false);
  assert.deepEqual(view().rows[1], framed("> a⏎b⏎c"));
  input.write(ENTER);
  assert.deepEqual(await answer, { value: "a\nb\nc" });
}

async function escapeAfterAPauseEndsAPasteThatNeverEnds(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  // The terminal dropped the paste's end.
  input.write(PASTE_START + "a\rb");
  await pastPastePause();
  input.write(ESCAPE);
  await pastEscape();
  assert.deepEqual(view().rows[1], framed("> a⏎b"));
  input.write("c" + ENTER);
  assert.deepEqual(await answer, { value: "a\nbc" });
}

async function ctrlCAfterAPauseDropsAPasteThatNeverEnds(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write("x" + PASTE_START + "a\rb");
  await pastPastePause();
  // Ctrl-C drops the paste, and only that: the prompt stays.
  input.write(CTRL_C);
  await tick();
  assert.deepEqual(view().rows[1], framed("> x"));
  input.write("y" + ENTER);
  assert.deepEqual(await answer, { value: "xy" });
}

async function aPasteKeepsItsTabsShownAsSpaces(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const answer = terminal.readLine("You:", { frame: true, hint: HINT });
  input.write(PASTE_START + "a\tb" + PASTE_END);
  await tick();
  assert.deepEqual(view().rows[1], framed("> a    b"));
  assert.deepEqual(view().cursor, { row: 1, col: 10 });
  input.write(PASTE_START + "\tx\ry\rz\rw" + PASTE_END + ENTER);
  assert.deepEqual(await answer, { value: "a\tb\tx\ny\nz\nw" });
}

async function escapeErasesAFramedReadLine(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 60 });
  output.write("above\r\n");
  const line = terminal.readLine("You:", { frame: true, hint: HINT, escape: true });
  input.write("half");
  await tick();
  input.write(ESCAPE);
  assert.deepEqual(await line, { escape: true, draft: "half" });
  assert.deepEqual(view().rows, ["above"]);
}

/** Wait past readline's escape timeout, so a lone Escape written before arrives as the Escape key. */
function pastEscape(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 600));
}

/**
 * A list of `rows`, each keyed by its first word, that can change, telling its one subscriber with
 * `changed`.
 */
function rowList(rows: string[]) {
  let listener: (() => void) | null = null;
  return {
    rows: () => rows.map((text) => ({ key: text.split(" ")[0]!, text })),
    subscribe(next: () => void) {
      listener = next;
      return () => (listener = null);
    },
    changed: () => listener?.(),
    subscribed: () => listener !== null,
  };
}

async function aFramedReadLineWithAListMovesIntoItAndOpensARow(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const list = rowList(["r1  a.yaml", "r2  b.yaml"]);
  const editor = new LineEditor({ columns: () => 60, hint: HINT, list });
  const box = [FRAME_TOP, framed("> hi"), FRAME_BOTTOM, `  ${HINT}`];
  const line = terminal.readLine("You:", { frame: true, escape: true, editor, initial: "hi" });
  await tick();
  assert.deepEqual(view().rows, [...box, "  r1  a.yaml", "  r2  b.yaml"]);
  // Arrows and Escape inside a paste move nothing.
  input.write(PASTE_START + DOWN + ESCAPE + PASTE_END);
  await tick();
  assert.equal(editor.focus(), null);
  input.write(DOWN);
  await tick();
  assert.deepEqual(view().rows, [...box, "» r1  a.yaml", "  r2  b.yaml"]);
  // The list has the keys: the cursor is hidden, and Down stops at the last row.
  assert.equal(view().modes.at(-1), "?25l");
  input.write(DOWN + DOWN);
  await tick();
  assert.deepEqual(view().rows.slice(-2), ["  r1  a.yaml", "» r2  b.yaml"]);
  input.write(UP);
  await tick();
  assert.deepEqual(view().rows.slice(-2), ["» r1  a.yaml", "  r2  b.yaml"]);
  // Up from the top row gives the keys back to the box, and the text goes on.
  input.write(UP + "!");
  await tick();
  const typed = [FRAME_TOP, framed("> hi!"), FRAME_BOTTOM, `  ${HINT}`];
  assert.deepEqual(view().rows, [...typed, "  r1  a.yaml", "  r2  b.yaml"]);
  assert.equal(view().modes.at(-1), "?25h");
  // Escape in the list goes back to the box; it doesn't end the prompt.
  input.write(DOWN + ESCAPE);
  await pastEscape();
  assert.equal(editor.focus(), null);
  input.write(DOWN + DOWN + ENTER);
  assert.deepEqual(await line, { open: "r2", draft: "hi!" });
  // The editor's box stays drawn, for what draws next to repaint over.
  assert.deepEqual(view().rows, [...typed, "  r1  a.yaml", "» r2  b.yaml"]);
  // The editor keeps the text and the row for the next prompt.
  assert.equal(editor.focus(), 1);
  assert.equal(list.subscribed(), false);
}

async function aListRowThatGoesGivesTheKeysBackAndTheListRedrawsOnItsOwn(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const rows = ["r1  a.yaml"];
  const list = { ...rowList(rows), hint: "↓ runs" };
  const editor = new LineEditor({ columns: () => 60, hint: "esc withdraws", list });
  const line = terminal.readLine("You:", { frame: true, escape: true, editor });
  await tick();
  // The list's hint follows the box's while it has rows.
  assert.deepEqual(view().rows.slice(-2), ["  esc withdraws · ↓ runs", "  r1  a.yaml"]);
  input.write(DOWN);
  await tick();
  assert.equal(editor.focus(), 0);
  rows.pop();
  list.changed();
  await tick();
  assert.deepEqual(view().rows, [FRAME_TOP, framed(">"), FRAME_BOTTOM, "  esc withdraws"]);
  assert.equal(editor.focus(), null);
  // With no rows, Down stays in the box.
  input.write(DOWN + "x" + ENTER);
  assert.deepEqual(await line, { value: "x" });
}

async function aLongListShowsWhatFitsTheFocusedRowAmongThem(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  const rows = Array.from({ length: 10 }, (_, i) => `r${i + 1}`);
  // Twelve rows high: the box takes three and the hint one, and four are left above it.
  const editor = new LineEditor({ columns: () => 60, rows: () => 12, hint: "", list: rowList(rows) });
  const line = terminal.readLine("You:", { frame: true, editor });
  await tick();
  assert.deepEqual(view().rows.slice(4), ["  r1", "  r2", "  r3", "  +7 more"]);
  input.write(DOWN.repeat(5));
  await tick();
  assert.deepEqual(view().rows.slice(4), ["  r3", "  r4", "» r5", "  +7 more"]);
  // The focus follows its row's key: a row going above it leaves it on the same item.
  rows.splice(0, 1);
  input.write(ENTER);
  assert.deepEqual(await line, { open: "r5", draft: "" });
}

async function aPasteTallerThanTheScreenShowsTheRowsAroundTheCursor(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 60 });
  // Twelve rows high: five rows of text fit in the box, under four left above it.
  const editor = new LineEditor({ columns: () => 60, rows: () => 12, hint: "hint" });
  const line = terminal.readLine("You:", { frame: true, editor });
  const text = Array.from({ length: 8 }, (_, i) => String(i).repeat(54)).join("");
  input.write(PASTE_START + text + PASTE_END);
  await tick();
  const rows = view().rows;
  assert.equal(rows.length, 8);
  assert.equal(rows[0], `╭─ ↑ 4 more ${"─".repeat(47)}╮`);
  assert.equal(rows[1], framed(`  ${"4".repeat(54)}`));
  assert.equal(rows[5], framed(""));
  assert.deepEqual(rows.slice(-2), [FRAME_BOTTOM, "  hint"]);
  // The cursor stays on its row, and the window follows it.
  assert.deepEqual(view().cursor, { row: 5, col: 4 });
  input.write(HOME);
  await tick();
  assert.equal(view().rows[1], framed(`> ${"0".repeat(54)}`));
  // Eight rows now: the cursor needs no row of its own after the last.
  assert.equal(view().rows[6], `╰─ ↓ 3 more ${"─".repeat(47)}╯`);
  assert.deepEqual(view().cursor, { row: 1, col: 4 });
  input.write(ENTER);
  assert.deepEqual(await line, { value: text });
}

async function anEnterAfterTheListShiftedOpensTheRunItsPointerIsOn(): Promise<void> {
  const { input, terminal } = setup({ columns: 60 });
  const rows = ["r1  a.yaml", "r2  b.yaml", "r3  c.yaml"];
  const list = rowList(rows);
  const editor = new LineEditor({ columns: () => 60, hint: "", list });
  const line = terminal.readLine("You:", { frame: true, editor });
  input.write(DOWN + DOWN);
  await tick();
  assert.equal(editor.focus(), 1);
  // r1 ends and r4 starts at the head: the pointer stays on r2, now the second row again.
  rows.splice(0, 1, "r4  d.yaml");
  rows.splice(1, 0, "r5  e.yaml");
  list.changed();
  await tick();
  assert.equal(editor.focus(), 2);
  input.write(ENTER);
  assert.deepEqual(await line, { open: "r2", draft: "" });
}

async function downOnAFoldedLineMovesIntoTheListOnlyFromItsLastRow(): Promise<void> {
  const { input, terminal } = setup({ columns: 20 });
  const editor = new LineEditor({ columns: () => 20, hint: "", list: rowList(["r1"]) });
  const line = terminal.readLine("You:", { frame: true, editor, initial: "a".repeat(20) });
  // Twenty characters fold onto two rows of fourteen: the cursor at the end is on the last.
  input.write(HOME + DOWN);
  await tick();
  assert.equal(editor.focus(), null);
  input.write(END + DOWN);
  await tick();
  assert.equal(editor.focus(), 0);
  input.write(ENTER);
  assert.deepEqual(await line, { open: "r1", draft: "a".repeat(20) });
}

async function watchKeysWithAnEditorGivesItTheKeysButCtrlC(): Promise<void> {
  const { input, terminal } = setup();
  const keys: string[] = [];
  const edited: string[] = [];
  const watch = new AbortController();
  // The editor takes every key but Escape.
  terminal.watchKeys(
    watch.signal,
    (key) => keys.push(key),
    (key) => {
      if (key.name === "escape") return false;
      edited.push(key.name ?? key.str ?? "?");
      return true;
    },
  );
  input.write("ab" + ENTER);
  input.write(CTRL_C);
  input.write(ESCAPE);
  await pastEscape();
  assert.deepEqual(edited, ["a", "b", "return"]);
  assert.deepEqual(keys, ["interrupt", "escape"]);
  watch.abort();
  // Nothing was kept for the next prompt.
  const line = terminal.readLine("You:");
  input.write("c" + ENTER);
  assert.deepEqual(await line, { value: "c" });
}

async function showScreenDrawsOnTheAlternateScreenAndGivesTheScreenBack(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 20 });
  output.rows = 3;
  output.write("chat line\r\n");
  let shown = ["one", "a row longer than the screen is wide", "three", "four"];
  let redraw: (() => void) | null = null;
  const screenUp = terminal.showScreen({
    draw: (columns, rows) => [`${columns}x${rows}`, ...shown],
    subscribe: (next) => {
      redraw = next;
      return () => (redraw = null);
    },
  });
  await tick();
  // As many rows as the screen holds, each cut to the width.
  assert.deepEqual(view().rows, ["20x3", "one", "a row longer than th"]);
  assert.ok(view().modes.includes("?1049h"));
  shown = ["changed"];
  redraw!();
  assert.deepEqual(view().rows, ["20x3", "changed"]);
  input.write("x");
  await tick();
  assert.deepEqual(view().rows, ["20x3", "changed"]);
  input.write("q");
  await screenUp;
  assert.equal(redraw, null);
  assert.deepEqual(view().rows, ["chat line"]);
  assert.deepEqual(view().cursor, { row: 1, col: 0 });
  // Escape leaves it too, and so does Ctrl-C, which leaves the terminal open.
  const escaped = terminal.showScreen({ draw: () => ["x"] });
  input.write(ESCAPE);
  await pastEscape();
  await escaped;
  const again = terminal.showScreen({ draw: () => ["x"] });
  input.write(CTRL_C);
  await again;
  const line = terminal.readLine("You:");
  input.write("ok" + ENTER);
  assert.deepEqual(await line, { value: "ok" });
}

async function showScreenSaysWhenAPromptWaitsBehindItAndClosesFirst(): Promise<void> {
  const events: string[] = [];
  const { input, terminal, view } = setup({ columns: 40, onActive: (on) => events.push(on ? "active" : "inactive") });
  const screenUp = terminal.showScreen({
    draw: (_columns, _rows, waiting) => [waiting ? "a prompt waits" : "nothing waits"],
    onClose: () => events.push("closed"),
  });
  await tick();
  assert.deepEqual(view().rows, ["nothing waits"]);
  // A line prompt behind it waits on no one; an ask does.
  const line = terminal.readLine("You:", { signal: AbortSignal.timeout(60_000) });
  await tick();
  assert.deepEqual(view().rows, ["nothing waits"]);
  const answer = terminal.askText("Name:");
  await tick();
  assert.deepEqual(view().rows, ["a prompt waits"]);
  input.write("q");
  await screenUp;
  // The screen's owner is told it closed before the prompt behind takes the terminal.
  assert.deepEqual(events.slice(0, 3), ["active", "closed", "inactive"]);
  await tick();
  input.write(ENTER);
  assert.deepEqual(await line, { value: "" });
  input.write("ada" + ENTER);
  assert.equal(await answer, "ada");
}

async function aWatchWithAnEditorTurnsBracketedPasteOnAndAPasteStaysOneLine(): Promise<void> {
  let clock = 0;
  const { input, output, terminal } = setup({ columns: 60, now: () => clock });
  const modes = () => screen(output.text).modes.filter((mode) => mode.startsWith("?2004"));
  const editor = new LineEditor({ columns: () => 60, hint: "" });
  const sent: string[] = [];
  const watch = new AbortController();
  terminal.watchKeys(
    watch.signal,
    () => {},
    (key) => {
      const action = editor.key(key);
      if (action !== null && "value" in action) sent.push(action.value);
      return action === null || !("escape" in action);
    },
  );
  // The terminal brackets a paste only while bracketed paste is on.
  assert.deepEqual(modes(), ["?2004h"]);
  input.write(PASTE_START + "one\r/exit\r" + UP + "three" + PASTE_END);
  await tick();
  assert.deepEqual(sent, []);
  assert.equal(editor.text(), "one\n/exit\nthree");
  // A prompt meanwhile leaves it on for the watch; past its grace, it takes keys.
  const answer = terminal.askText("Name:");
  await tick();
  clock += 1000;
  input.write("x" + ENTER);
  assert.equal(await answer, "x");
  assert.equal(modes().at(-1), "?2004h");
  input.write(ENTER);
  await tick();
  assert.deepEqual(sent, ["one\n/exit\nthree"]);
  watch.abort();
  assert.equal(modes().at(-1), "?2004l");
}

async function theExitHookLeavesTheAlternateScreen(): Promise<void> {
  const { input, output, terminal } = setup();
  const before = process.listeners("exit");
  const screenUp = terminal.showScreen({ draw: () => ["x"] });
  await tick();
  const added = process.listeners("exit").filter((listener) => !before.includes(listener));
  assert.equal(added.length, 1);
  added[0]!(0);
  assert.equal(screen(output.text).modes.at(-1), "?1049l");
  input.write("q");
  await screenUp;
}

async function showScreenOffATtyDoesNothing(): Promise<void> {
  const { output, terminal } = setup({ tty: false });
  await terminal.showScreen({ draw: () => ["x"] });
  assert.equal(output.text, "");
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

// The `promptInputs` scenarios: the input prompts of a run, one widget per declared input.

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

/** A select as the screen shows it: its label, its choice titles and the marked one. */
type Picker = { label: string; titles: string[]; marked: number };

/** The active select, read off the screen. */
function frame(view: () => Screen): Picker {
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
  const { titles, marked } = await select(view);
  const target = titles.indexOf(title);
  assert.ok(target >= 0, `no choice ${title} in ${JSON.stringify(titles)}`);
  input.write((target > marked ? DOWN : UP).repeat(Math.abs(target - marked)) + ENTER);
}

/** Wait for the active select and return its frame. */
async function select(view: () => Screen): Promise<Picker> {
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
    [PASTE_START + "y" + PASTE_END + ENTER, false, "No"],
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

async function aScalarEditorChecksItsTextAsTheRunWouldAndStaysOpen(): Promise<void> {
  const cases: [Type, string, string, string][] = [
    [Type.scalar(ValueKind.INTEGER), "abc", '"abc" does not match declared type int', " 1_000 "],
    [Type.scalar(ValueKind.NUMBER), "1e999", "number is too large for a float", "inf"],
    [Type.scalar(ValueKind.DATE), "21/05/2026", '"21/05/2026" does not match declared type date: date must look like 2026-05-21', "2026-05-21"],
    [nullable(ValueKind.INTEGER), "", '"" does not match declared type int', "7"],
  ];
  for (const [type, bad, error, good] of cases) {
    const { input, terminal, view } = setup();
    const answer = terminal.promptInputs([decl("value", type, { required: true })], {});
    if (type.nullable) await choose(input, view, "Enter value");
    await tick();
    input.write(bad + ENTER);
    await tick();
    assert.deepEqual(view().rows.at(-1), error, bad);
    input.write(CTRL_E + CTRL_U + good + ENTER);
    // The text goes back as typed; the run reads it at START.
    assert.deepEqual(await answer, { value: good });
  }
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
    assert.deepEqual(await answer, { value: null });
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
async function anInboxInputIsNeverAsked(): Promise<void> {
  const { output, terminal } = setup();
  const inbox = decl("messages", inboxType(STR), { required: true });
  assert.deepEqual(await terminal.promptInputs([inbox], {}), {});
  assert.equal(output.text, "");
}

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

async function ctrlDInAConfirmIsTheEndOfInput(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.promptInputs([decl("ok", BOOL, { required: true })], {});
  input.write("x" + CTRL_D);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["eof"]);
  assertRestored(input, output, exitHooks);
}

async function ctrlQInterruptsAConfirm(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.promptInputs([decl("ok", BOOL, { required: true })], {});
  input.write(CTRL_Q);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["interrupt"]);
  assertRestored(input, output, exitHooks);
}

async function ctrlQInterruptsASelectButNotATextEditor(): Promise<void> {
  const stops: string[] = [];
  const { input, terminal, view } = setup({ onStop: (reason) => stops.push(reason) });
  const text = terminal.askText("Name:");
  input.write("a" + CTRL_Q + "b" + ENTER);
  assert.equal(await text, "ab");
  const answer = terminal.promptInputs([decl("flag", nullable(ValueKind.BOOLEAN))], {});
  await select(view);
  input.write(CTRL_Q);
  assert.equal(await answer, null);
  assert.deepEqual(stops, ["interrupt"]);
  assert.deepEqual(input.rawCalls, [true, false, true, false]);
}

// Blank is Unicode whitespace: U+0085 is blank, U+FEFF is not.
async function anOptionalStringBlankIsUnicodeWhitespace(): Promise<void> {
  const { input, terminal } = setup();
  const answer = terminal.promptInputs([decl("kept", STR), decl("omitted", STR)], {});
  input.write("\uFEFF" + ENTER + "\x85 " + ENTER);
  assert.deepEqual(await answer, { kept: "\uFEFF" });
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

// The `askUser` scenarios: an `ask_user` request's questions rendered and answered on the
// terminal, model options offered as a choice, and the terminal restored after each.

/** An `ask_user` request: `fields` over the absent ones (null). */
function request(fields: Partial<AskUserRequest>): AskUserRequest {
  return { question: "", options: null, questions: null, answerSchema: null, title: null, nodeId: null, ...fields };
}

/** An `ask_user` call's context over `signal` (a fresh, never-aborted one by default). */
function toolContext(signal: AbortSignal = new AbortController().signal): ToolContext {
  return new ToolContext({
    callId: "run-1/ask",
    attempt: 1,
    toolName: "ask_user",
    runId: "run-1",
    node: "ask",
    context: undefined,
    signal,
  });
}

/**
 * What `promise` threw, or `{ resolved }` when it resolved. The rejection is handled at once, so a
 * call that is checked later never counts as unhandled.
 */
function thrown(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    (resolved) => ({ resolved }),
    (error: unknown) => error,
  );
}

/** `error` is a `TerminalStopped` for `reason`. */
function assertStopped(error: unknown, reason: "interrupt" | "eof" | "closed"): void {
  assert.ok(error instanceof TerminalStopped, `not a TerminalStopped: ${String(error)}`);
  assert.equal(error.reason, reason);
}

/** A `Question i of N` rule as the screen shows it, `columns` wide. */
function rule(title: string, columns = 80): string {
  return `${title} ${"─".repeat(columns - Array.from(title).length - 1)}`;
}

const OTHER = "✎ Other (write your own)";
const CHECKBOX_HELP = "(Use arrow keys to move, <space> to select, <a> to toggle, <i> to invert)";

// test_multi_select_assembly, test_structured_prompt_metadata_reaches_handler_unchanged
async function aQuestionSetShowsThePanelAndARulePerQuestion(): Promise<void> {
  const active: boolean[] = [];
  const { input, output, terminal, view, exitHooks } = setup({ columns: 120, onActive: (on) => active.push(on) });
  const answer = terminal.askUser(
    request({
      question: "Complete the review",
      questions: parseQuestions([
        { question: "Choose areas", header: "Areas", options: [{ label: "API" }, { label: "UI" }], multi_select: true },
        { question: "Notes?", header: "Notes" },
      ]),
      answerSchema: [
        { name: "Areas", type: "list[str]" },
        { name: "Notes", type: "str" },
      ],
      title: "Release review",
      nodeId: "review/approval",
    }),
    toolContext(),
  );
  await tick();
  const panel = [
    "╭────────────── human input ──────────────╮",
    "│ Your input is needed for Release review │",
    "│ Complete the review                     │",
    "│                                         │",
    "│ 2 questions to answer.                  │",
    "╰─────────────────────────────────────────╯",
  ];
  assert.deepEqual(view().rows, [
    ...panel,
    "",
    rule("Question 1 of 2 · Areas", 120),
    `? Choose areas ${CHECKBOX_HELP}`,
    " » ○ API",
    "   ○ UI",
    `   ○ ${OTHER}`,
  ]);
  input.write(SPACE + DOWN + SPACE);
  await tick();
  assert.deepEqual(view().rows.slice(-3), ["   ● API", " » ● UI", `   ○ ${OTHER}`]);
  input.write(ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(panel.length), [
    "",
    rule("Question 1 of 2 · Areas", 120),
    "? Choose areas done (2 selections)",
    "",
    rule("Question 2 of 2 · Notes", 120),
    "? Notes?",
  ]);
  input.write("ship it" + ENTER);
  assert.deepEqual(await answer, { Areas: ["API", "UI"], Notes: "ship it" });
  assert.deepEqual(view().rows.slice(-1), ["? Notes? ship it"]);
  assert.deepEqual(active, [true, false]);
  assertRestored(input, output, exitHooks);
}

// test_single_select_assembly
async function aSingleSelectShowsDescriptionsAndReturnsTheBareLabel(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askUser(
    request({
      questions: parseQuestions([
        {
          question: "Which framework?",
          header: "Framework",
          options: [{ label: "React", description: "A UI library" }, { label: "Vue" }],
        },
      ]),
      nodeId: "pick/framework",
    }),
    toolContext(),
  );
  await tick();
  assert.deepEqual(view().rows, [
    "╭────────────── human input ──────────────╮",
    "│ Your input is needed for pick/framework │",
    "│                                         │",
    "│ 1 question to answer.                   │",
    "╰─────────────────────────────────────────╯",
    "",
    rule("Question 1 of 1 · Framework"),
    "? Which framework? (Use arrow keys)",
    " » ● React  —  A UI library",
    "   ○ Vue",
    `   ○ ${OTHER}`,
  ]);
  input.write(DOWN + DOWN + DOWN + ENTER);
  assert.deepEqual(await answer, { Framework: "React" });
  assert.deepEqual(view().rows.slice(-1), ["? Which framework? React  —  A UI library"]);
}

async function otherAndTextQuestionsFillARecordKeyedByHeader(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askUser(
    request({
      questions: [
        { question: "Why?", header: "__proto__", options: [], multiSelect: false },
        { question: "Pick", header: "", options: [{ label: "A", description: "" }], multiSelect: false },
        { question: "Anything else?", header: "Extra", options: [], multiSelect: false },
      ],
    }),
    toolContext(),
  );
  await tick();
  assert.deepEqual(view().rows.slice(0, 2), [
    "╭─────────── human input ────────────╮",
    "│ Your input is needed for this step │",
  ]);
  assert.deepEqual(view().rows.slice(-1), ["? Why?"]);
  input.write("because" + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(-5), [
    "",
    rule("Question 2 of 3"),
    "? Pick (Use arrow keys)",
    " » ● A",
    `   ○ ${OTHER}`,
  ]);
  input.write(DOWN + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(-2), [`? Pick ${OTHER}`, "✎ Your answer"]);
  input.write("mine" + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(-1), ["? Anything else?"]);
  input.write(ENTER);
  const record = (await answer) as Record<string, unknown>;
  assert.deepEqual(Object.entries(record), [
    ["__proto__", "because"],
    ["", "mine"],
    ["Extra", ""],
  ]);
  assert.equal(Object.getPrototypeOf(record), Object.prototype);
  assert.deepEqual(view().rows.slice(5), [
    "",
    rule("Question 1 of 3 · __proto__"),
    "? Why? because",
    "",
    rule("Question 2 of 3"),
    `? Pick ${OTHER}`,
    "✎ Your answer mine",
    "",
    rule("Question 3 of 3 · Extra"),
    "? Anything else?",
  ]);
}

async function aCheckboxTogglesInvertsAndReturnsChoiceOrder(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 120 });
  const ask = (): Promise<unknown> =>
    terminal.askUser(
      request({
        questions: [
          {
            question: "Pick",
            header: "P",
            options: [{ label: "API", description: "" }, { label: "UI", description: "" }],
            multiSelect: true,
          },
        ],
      }),
      toolContext(),
    );
  const rows = (): string[] => view().rows.slice(-3);

  let answer = ask();
  await tick();
  input.write(SPACE);
  await tick();
  assert.deepEqual(rows(), [" » ● API", "   ○ UI", `   ○ ${OTHER}`]);
  input.write("a");
  await tick();
  assert.deepEqual(rows(), [" » ● API", "   ● UI", `   ● ${OTHER}`]);
  input.write("a");
  await tick();
  assert.deepEqual(rows(), [" » ○ API", "   ○ UI", `   ○ ${OTHER}`]);
  input.write("i");
  await tick();
  assert.deepEqual(rows(), [" » ● API", "   ● UI", `   ● ${OTHER}`]);
  input.write(SPACE + UP);
  await tick();
  assert.deepEqual(rows(), ["   ○ API", "   ● UI", ` » ● ${OTHER}`]);
  input.write("j" + "k" + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(-2), ["? Pick done (2 selections)", "✎ Your answer"]);
  input.write("mine" + ENTER);
  assert.deepEqual(await answer, { P: ["UI", "mine"] });

  // Picked in reverse, the answer still follows the choice order, Other's text in its place.
  answer = ask();
  await tick();
  input.write(UP + SPACE + DOWN + SPACE + ENTER + "typed" + ENTER);
  assert.deepEqual(await answer, { P: ["API", "typed"] });

  answer = ask();
  await tick();
  input.write(DOWN + SPACE + ENTER);
  assert.deepEqual(await answer, { P: ["UI"] });
  assert.deepEqual(view().rows.slice(-1), ["? Pick [UI]"]);

  answer = ask();
  await tick();
  input.write(ENTER);
  assert.deepEqual(await answer, { P: [] });
  assert.deepEqual(view().rows.slice(-1), ["? Pick done"]);
}

async function ctrlQInterruptsACheckbox(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = thrown(
    terminal.askUser(
      request({
        questions: [{ question: "Pick", header: "P", options: [{ label: "A", description: "" }], multiSelect: true }],
      }),
      toolContext(),
    ),
  );
  await tick();
  input.write(CTRL_Q);
  assertStopped(await answer, "interrupt");
  assert.deepEqual(stops, ["interrupt"]);
  assertRestored(input, output, exitHooks);
}

// test_prompt_request_offers_model_options_as_a_choice, test_ask_user_options_reach_the_prompt_request
async function optionsAreOneSelectWithOtherAndNoPanel(): Promise<void> {
  const { input, output, terminal, view } = setup();
  let answer = terminal.askUser(request({ question: "Which?", options: ["A", "B"] }), toolContext());
  await tick();
  assert.deepEqual(view().rows, ["? Which? (Use arrow keys)", " » ● A", "   ○ B", `   ○ ${OTHER}`]);
  input.write(DOWN + ENTER);
  assert.equal(await answer, "B");
  assert.deepEqual(view().rows, ["? Which? B"]);
  answer = terminal.askUser(request({ question: "Which?", options: ["A", "B"] }), toolContext());
  input.write(UP + ENTER + "C" + ENTER);
  assert.equal(await answer, "C");
  assert.deepEqual(view().rows, ["? Which? B", `? Which? ${OTHER}`, "✎ Your answer C"]);
  assert.ok(!output.text.includes("human input"));
  assert.ok(!output.text.includes("Question"));
}

async function anAskShowsControlCharactersInsteadOfSendingThem(): Promise<void> {
  const { input, output, terminal } = setup();
  const osc52 = "\x1b]52;c;aGk=\x07";
  const link = "A\x1b]8;;https://x.example\x1b\\";
  const answer = terminal.askUser(
    request({
      question: `Prompt${osc52}\r\nSecond line`,
      questions: parseQuestions([
        { question: "Pick\x9b2J one", header: "H\x1b[1A", options: [{ label: link, description: "d\x7f" }] },
      ]),
      title: `Title${osc52}`,
    }),
    toolContext(),
  );
  input.write(ENTER);
  // The answer keeps the label and the header as the request wrote them.
  assert.deepEqual(await answer, { "H\x1b[1A": link });
  const text = terminal.askUser(request({ question: "Name\x1b]0;owned\x07?" }), toolContext());
  input.write(ENTER);
  assert.equal(await text, "");
  for (const sent of ["\x1b]", "\x07", "\x9b", "\x7f", "\x1b\\", "H\x1b[1A"]) {
    assert.ok(!output.text.includes(sent), JSON.stringify(sent));
  }
  const shown = [
    "Title^[]52;c;aGk=^G",
    "Prompt^[]52;c;aGk=^G",
    "H^[[1A",
    "Pick\uFFFD2J one",
    "A^[]8;;https://x.example^[\\",
    "d^?",
    "Name^[]0;owned^G?",
  ];
  for (const each of shown) {
    assert.ok(output.text.includes(each), JSON.stringify(each));
  }
  // A `\r\n` in the text is a line break.
  assert.ok(!output.text.includes("^M"));
}

async function aChoiceShowsControlCharactersAndAnswersTheChoiceAsGiven(): Promise<void> {
  const { input, output, terminal } = setup();
  const model = "gpt\x1b]52;c;aGk=\x07-x";
  const answer = terminal.askChoice("Model:", ["plain", model]);
  input.write(DOWN + ENTER);
  assert.equal(await answer, model);
  assert.ok(!output.text.includes("\x1b]52"), JSON.stringify(output.text));
  assert.ok(output.text.includes("gpt^[]52;c;aGk=^G-x"), JSON.stringify(output.text));
}

async function aPlainAskIsATextPrompt(): Promise<void> {
  const { input, terminal, view } = setup();
  let answer = terminal.askUser(request({ question: "Name?", options: [], questions: [] }), toolContext());
  input.write(ENTER);
  assert.equal(await answer, "");
  answer = terminal.askUser(request({ nodeId: "ask/name" }), toolContext());
  input.write("Ann" + ENTER);
  assert.equal(await answer, "Ann");
  answer = terminal.askUser(request({}), toolContext());
  input.write(ENTER);
  assert.equal(await answer, "");
  assert.deepEqual(view().rows, ["? Name?", "? input for ask/name Ann", "? input for null"]);
}

// test_stop_event_wakes_active_prompt_and_discards_its_late_answer,
// test_stop_event_set_before_prompt_completion_discards_late_answer
async function anAbortedAskUserThrowsTheReasonAndLeavesLaterKeysBuffered(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const controller = new AbortController();
  const reason = new Error("stopped");
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext(controller.signal)));
  input.write("ear");
  await tick();
  controller.abort(reason);
  assert.equal(await answer, reason);
  assertRestored(input, output, exitHooks);
  input.write("late" + ENTER);
  await tick();
  assert.equal(await terminal.askText("Next:"), "late");
  assert.deepEqual(stops, []);
}

// test_stop_event_wakes_queued_waiter_without_serving_prompt
async function aQueuedAskUserWhoseSignalAbortsNeverDraws(): Promise<void> {
  const { input, output, terminal } = setup();
  const controller = new AbortController();
  const reason = new Error("stopped");
  const first = terminal.askUser(request({ question: "First?" }), toolContext());
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext(controller.signal)));
  await tick();
  controller.abort(reason);
  assert.equal(await queued, reason);
  input.write("1" + ENTER);
  assert.equal(await first, "1");
  assert.ok(!output.text.includes("Queued"));
  await assert.rejects(
    terminal.askUser(request({ question: "Again?" }), toolContext(controller.signal)),
    (error) => error === reason,
  );
  assert.ok(!output.text.includes("Again"));
}

async function aQuestionSetAbortedMidwayDiscardsThePartialRecord(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const controller = new AbortController();
  const reason = new Error("stopped");
  const answer = thrown(
    terminal.askUser(
      request({
        questions: [
          { question: "A?", header: "a", options: [], multiSelect: false },
          { question: "B?", header: "b", options: [], multiSelect: false },
        ],
      }),
      toolContext(controller.signal),
    ),
  );
  input.write("x" + ENTER);
  await tick();
  assert.deepEqual(view().rows.slice(-1), ["? B?"]);
  controller.abort(reason);
  assert.equal(await answer, reason);
  assertRestored(input, output, exitHooks);
}

// test_non_eof_failure_after_a_stop_is_still_discarded
async function aFailingWriteAfterAnAbortStillThrowsTheReason(): Promise<void> {
  // Write 1 sets the modes, write 2 draws the frame; after the abort, write 3 collapses the frame
  // and write 4 restores the modes.
  for (const throwOn of [3, 4]) {
    const { input, output, terminal, exitHooks } = setup({ throwOn });
    const controller = new AbortController();
    const reason = new Error("stopped");
    const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext(controller.signal)));
    await tick();
    controller.abort(reason);
    assert.equal(await answer, reason, `write ${throwOn}`);
    assert.equal(output.writes, 4);
    assertRestored(input, output, exitHooks, false);
  }
}

async function aFailingOutputFailsAQuestionSet(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup({ throwOn: 1 });
  const answer = terminal.askUser(
    request({ questions: [{ question: "A?", header: "a", options: [], multiSelect: false }] }),
    toolContext(),
  );
  await assert.rejects(answer, (error) => error === output.failure);
  assertRestored(input, output, exitHooks);
}

// test_ctrl_c_during_active_prompt_preserves_keyboard_interrupt_and_joins,
// test_cancel_all_wakes_queued_waiters_and_rejects_later_prompt
async function ctrlCThrowsTheReasonOnStopAbortedAndRefusesQueuedCalls(): Promise<void> {
  const stops: string[] = [];
  const controller = new AbortController();
  const reason = new Error("stopped");
  const { input, output, terminal, view, exitHooks } = setup({
    onStop: (stop) => {
      stops.push(stop);
      controller.abort(reason);
    },
  });
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext(controller.signal)));
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext(controller.signal)));
  const plain = thrown(terminal.askUser(request({ question: "Plain?" }), toolContext()));
  await tick();
  input.write(CTRL_C);
  assert.equal(await answer, reason);
  assert.equal(await queued, reason);
  assertStopped(await plain, "closed");
  assert.deepEqual(stops, ["interrupt"]);
  assert.deepEqual(view().rows, ["? Name?"]);
  assertRestored(input, output, exitHooks);
  assert.equal(await thrown(terminal.askUser(request({ question: "Later?" }), toolContext(controller.signal))), reason);
  assertStopped(await thrown(terminal.askUser(request({ question: "Later?" }), toolContext())), "closed");
  assert.deepEqual(view().rows, ["? Name?"]);
}

async function ctrlCWithoutAnAbortThrowsInterrupt(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext()));
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext()));
  await tick();
  input.write(CTRL_C);
  assertStopped(await answer, "interrupt");
  assertStopped(await queued, "closed");
  assert.deepEqual(view().rows, ["? Name?"]);
  assertRestored(input, output, exitHooks);
}

// test_generic_eof_fails_current_prompt_and_cancels_queued_prompts,
// test_eof_that_closed_the_broker_outranks_a_stop_seen_later, test_pooled_prompt_eof_outranks_cancelled_sibling
async function theEndOfInputOutranksTheAbortItCausesAndRefusesQueuedCalls(): Promise<void> {
  const stops: string[] = [];
  const controller = new AbortController();
  const reason = new Error("stopped");
  const { input, output, terminal, exitHooks } = setup({
    onStop: (stop) => {
      stops.push(stop);
      controller.abort(reason);
    },
  });
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext(controller.signal)));
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext(controller.signal)));
  const plain = thrown(terminal.askUser(request({ question: "Plain?" }), toolContext()));
  await tick();
  input.end();
  assertStopped(await answer, "eof");
  assert.equal(await queued, reason);
  assertStopped(await plain, "closed");
  assert.deepEqual(stops, ["eof"]);
  assert.ok(!output.text.includes("Queued") && !output.text.includes("Plain"));
  assertRestored(input, output, exitHooks);
}

async function closeEndsAnActiveAskUserAndRefusesLaterOnes(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup();
  const controller = new AbortController();
  const reason = new Error("stopped");
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext()));
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext()));
  const queuedAborted = thrown(terminal.askUser(request({ question: "Aborted?" }), toolContext(controller.signal)));
  await tick();
  terminal.close();
  // Aborted after the close, so the closed terminal refuses it and the reason is what it throws.
  controller.abort(reason);
  assertStopped(await answer, "closed");
  assertStopped(await queued, "closed");
  assert.equal(await queuedAborted, reason);
  assertRestored(input, output, exitHooks);
  assert.ok(!output.text.includes("Queued") && !output.text.includes("Aborted"));
  const drawn = output.text;
  const aborted = toolContext(AbortSignal.abort(reason));
  assert.equal(await thrown(terminal.askUser(request({ question: "Later?" }), aborted)), reason);
  assertStopped(await thrown(terminal.askUser(request({ question: "Later?" }), toolContext())), "closed");
  assert.equal(output.text, drawn);
}

// test_cancel_all_invokes_every_hook_and_ignores_callback_failures
async function throwingHooksStillRestoreAndRefuseQueuedCalls(): Promise<void> {
  const calls: string[] = [];
  const { input, output, terminal, exitHooks } = setup({
    onStop: (stop) => {
      calls.push(stop);
      throw new Error("onStop broke");
    },
    onActive: (on) => {
      calls.push(String(on));
      throw new Error("onActive broke");
    },
  });
  const answer = thrown(terminal.askUser(request({ question: "Name?" }), toolContext()));
  const queued = thrown(terminal.askUser(request({ question: "Queued?" }), toolContext()));
  await tick();
  input.write(CTRL_C);
  assertStopped(await answer, "interrupt");
  assertStopped(await queued, "closed");
  assert.deepEqual(calls, ["true", "false", "interrupt"]);
  assertRestored(input, output, exitHooks);
}

// test_concurrent_prompts_are_served_in_fifo_order
async function concurrentAskUserCallsAreServedInFifoOrder(): Promise<void> {
  const { input, terminal, view } = setup();
  const answers = ["first", "second", "third"].map((question) =>
    terminal.askUser(request({ question }), toolContext()),
  );
  input.write("1" + ENTER + "2" + ENTER + "3" + ENTER);
  assert.deepEqual(await Promise.all(answers), ["1", "2", "3"]);
  assert.deepEqual(view().rows, ["? first 1", "? second 2", "? third 3"]);
}

const PLAN = "Here is the plan:\n\n1. a\n\nApprove it?";

async function aMultiLineTextLabelIsWrittenOnceAboveTheFrame(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askText(PLAN);
  input.write("apx");
  await tick();
  input.write(BACKSPACE + LEFT + RIGHT + "p");
  await tick();
  assert.deepEqual(view().rows, ["? Here is the plan:", "", "1. a", "", "Approve it? app"]);
  assert.deepEqual(view().cursor, { row: 4, col: 15 });
  input.write(ENTER);
  assert.equal(await answer, "app");
  assert.deepEqual(view().rows, ["? Here is the plan:", "", "1. a", "", "Approve it? app"]);
}

async function aMultiLineChoiceLabelIsWrittenOnceAboveTheFrame(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askChoice(PLAN, ["approve", "reject"]);
  input.write(DOWN + UP + DOWN);
  await tick();
  assert.deepEqual(view().rows, [
    "? Here is the plan:",
    "",
    "1. a",
    "",
    "Approve it? (Use arrow keys)",
    "   approve",
    " » reject",
  ]);
  input.write(ENTER);
  assert.equal(await answer, "reject");
  assert.deepEqual(view().rows, ["? Here is the plan:", "", "1. a", "", "Approve it? reject"]);
}

async function aChoiceTitleWithANewlineRedrawsWithoutStaleRows(): Promise<void> {
  const { input, output, terminal, view } = setup();
  output.write("above\r\n");
  const answer = terminal.askChoice("Pick", ["one\ntwo", "three"]);
  input.write(DOWN + UP + DOWN);
  await tick();
  assert.deepEqual(view().rows, ["above", "? Pick (Use arrow keys)", "   one", "two", " » three"]);
  input.write(DOWN + ENTER);
  assert.equal(await answer, "one\ntwo");
  assert.deepEqual(view().rows, ["above", "? Pick one", "two"]);
}

async function aPasteIntoASelectChangesNothing(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askChoice("Approve?", ["approve", "reject"]);
  input.write(PASTE_START + "just some notes\r" + PASTE_END);
  await tick();
  assert.deepEqual(view().rows, ["? Approve? (Use arrow keys)", " » approve", "   reject"]);
  input.write(ENTER);
  assert.equal(await answer, "approve");
}

async function combiningMarksTakeNoColumn(): Promise<void> {
  const { input, terminal, view } = setup();
  const nfd = "e\u0301".repeat(3);
  const answer = terminal.askText("Name:");
  input.write(nfd);
  await tick();
  assert.deepEqual(view().rows, [`? Name: ${nfd}`]);
  assert.deepEqual(view().cursor, { row: 0, col: 11 });
  input.write(ENTER);
  assert.equal(await answer, nfd);
}

async function aKeyTypedSoonAfterEscapeIsKept(): Promise<void> {
  const { input, terminal, view } = setup();
  const answer = terminal.askText("Name:");
  input.write(ESCAPE);
  await tick();
  input.write("a");
  await tick();
  assert.deepEqual(view().rows, ["? Name: a"]);
  input.write(ENTER);
  assert.equal(await answer, "a");
}

async function ctrlNAndCtrlPMoveInASelectAndACheckbox(): Promise<void> {
  const { input, terminal } = setup({ columns: 120 });
  const choice = terminal.askChoice("Pick", ["a", "b", "c"]);
  input.write(CTRL_N + CTRL_N + CTRL_N + CTRL_P + ENTER);
  assert.equal(await choice, "c");
  const options = [{ label: "API", description: "" }, { label: "UI", description: "" }];
  const question = { question: "Pick", header: "P", options, multiSelect: true };
  const picked = terminal.askUser(request({ questions: [question] }), toolContext());
  input.write(CTRL_N + SPACE + CTRL_P + CTRL_P + SPACE + ENTER + "x" + ENTER);
  assert.deepEqual(await picked, { P: ["UI", "x"] });
}

async function aNonTtyTurnStillHasTheExitHook(): Promise<void> {
  const { input, output, terminal, exitHooks } = setup({ tty: false });
  const before = process.listeners("exit");
  const answer = terminal.askChoice("Pick", ["a"]);
  await tick();
  const added = process.listeners("exit").filter((listener) => !before.includes(listener));
  assert.equal(added.length, 1);
  added[0]!(0);
  assert.deepEqual(screen(output.text).modes.slice(-2).sort(), ["?2004l", "?25h"]);
  input.write(ENTER);
  assert.equal(await answer, "a");
  assert.equal(process.listenerCount("exit"), exitHooks);
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

/** An approval request for `approve`, with the Approve and Reject choices. */
function approval(fields: { signal?: AbortSignal; diff?: string | null } = {}) {
  return {
    title: "Write a.yaml",
    body: "a new flow",
    diff: fields.diff ?? null,
    choices: ["Approve", "Reject"],
    signal: fields.signal,
  };
}

async function approveEnterPicksTheFirstChoiceAndDownTheNext(): Promise<void> {
  const { input, output, terminal, view, exitHooks } = setup();
  let answer = terminal.approve(approval({ diff: "--- a.yaml\n+++ a.yaml\n@@ -1 +1 @@\n-old\n+new\n" }));
  await tick();
  assert.deepEqual(view().rows, [
    "a new flow",
    "--- a.yaml",
    "+++ a.yaml",
    "@@ -1 +1 @@",
    "-old",
    "+new",
    "? Write a.yaml (Use arrow keys)",
    " » Approve",
    "   Reject",
  ]);
  input.write(ENTER);
  assert.equal(await answer, "Approve");
  assertRestored(input, output, exitHooks);
  answer = terminal.approve(approval());
  input.write(DOWN + ENTER);
  assert.equal(await answer, "Reject");
  assert.equal(view().rows.at(-1), "? Write a.yaml Reject");
}

async function approveShowsItsTextSanitized(): Promise<void> {
  const { input, output, terminal } = setup();
  const answer = terminal.approve({
    title: "Write\x1b[2J a.yaml",
    body: "body\x1b]52;c;aGk=\x07 \u202eend",
    diff: "+line\x1b[1A",
    choices: ["Approve\x07", "Reject"],
  });
  input.write(ENTER);
  assert.equal(await answer, "Approve\x07");
  for (const sent of ["\x1b[2J", "\x1b]", "\x07", "\u202e", "\x1b[1A"]) {
    assert.ok(!output.text.includes(sent), JSON.stringify(sent));
  }
  // A sequence is shown rather than dropped, so the card never hides what the text holds.
  for (const shown of ["Write^[[2J a.yaml", "body^[]52;c;aGk=^G \\u{202e}end", "+line^[[1A", "Approve^G"]) {
    assert.ok(output.text.includes(shown), JSON.stringify(shown));
  }
}

async function aFramedApprovalNumbersTheDiffAndColoursItsLines(): Promise<void> {
  const { input, output, terminal, view } = setup({ columns: 60 });
  const diff =
    "--- a.yaml\n+++ a.yaml\n@@ -1,3 +1,3 @@\n id: a\n-name: old\n+name: new\n nodes:\n" +
    "@@ -12 +12,2 @@\n-x\x1b[2J\n+y\n+z\n";
  const answer = terminal.approve({ ...approval({ diff }), frame: true });
  await tick();
  const row = (text: string) => `│ ${text}${" ".repeat(56 - text.length)} │`;
  assert.deepEqual(view().rows, [
    `╭─ Write a.yaml ${"─".repeat(43)}╮`,
    row("a new flow"),
    row(""),
    row(" 1   id: a"),
    row(" 2 - name: old"),
    row(" 2 + name: new"),
    row(" 3   nodes:"),
    row("⋯"),
    row("12 - x^[[2J"),
    row("12 + y"),
    row("13 + z"),
    `╰${"─".repeat(58)}╯`,
    "? Write a.yaml (Use arrow keys)",
    " » Approve",
    "   Reject",
  ]);
  // Removed lines red, added green; an escape sequence shown, never sent.
  assert.ok(output.text.includes("\x1b[31m 2 - name: old"));
  assert.ok(output.text.includes("\x1b[32m 2 + name: new"));
  assert.ok(!output.text.includes("\x1b[2J"));
  input.write(ENTER);
  assert.equal(await answer, "Approve");
}

async function aFramedApprovalFoldsALongLineInsideTheCard(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 30 });
  const answer = terminal.approve({ ...approval({ diff: `@@ -1 +1 @@\n+${"y".repeat(30)}\n` }), frame: true });
  await tick();
  assert.deepEqual(view().rows.slice(3, 6), [
    `│ 1 + ${"y".repeat(22)} │`,
    `│     ${"y".repeat(8)}${" ".repeat(14)} │`,
    `╰${"─".repeat(28)}╯`,
  ]);
  input.write(ENTER);
  assert.equal(await answer, "Approve");
}

async function aFramedApprovalCutsALongTitle(): Promise<void> {
  const { input, terminal, view } = setup({ columns: 30 });
  const answer = terminal.approve({ ...approval(), title: `Write ${"deep/".repeat(10)}a.yaml`, frame: true });
  await tick();
  assert.equal(view().rows[0], "╭─ Write deep/deep/deep/de… ─╮");
  input.write(ENTER);
  assert.equal(await answer, "Approve");
}

async function approveOffATtyReturnsNullWithoutReading(): Promise<void> {
  const { input, output, terminal } = setup({ tty: false });
  assert.equal(terminal.canApprove, false);
  assert.equal(setup().terminal.canApprove, true);
  input.write(ENTER);
  assert.equal(await terminal.approve(approval()), null);
  assert.equal(output.text, "");
  // The Enter is still there for the next prompt.
  assert.equal(await terminal.askText("Name:"), "");
}

async function ctrlCInApproveRejectsAndKeepsTheTerminalOpen(): Promise<void> {
  const stops: string[] = [];
  const { input, output, terminal, exitHooks } = setup({ onStop: (reason) => stops.push(reason) });
  const answer = terminal.approve(approval());
  input.write(CTRL_C);
  assert.equal(await answer, "Reject");
  assert.deepEqual(stops, []);
  assertRestored(input, output, exitHooks);
  const next = terminal.approve(approval());
  input.write(ENTER);
  assert.equal(await next, "Approve");
}

async function anAbortedApproveReturnsNull(): Promise<void> {
  const { input, terminal } = setup();
  const aborted = new AbortController();
  aborted.abort(new Error("stop"));
  assert.equal(await terminal.approve(approval({ signal: aborted.signal })), null);
  const later = new AbortController();
  const answer = terminal.approve(approval({ signal: later.signal }));
  await tick();
  later.abort(new Error("stop"));
  assert.equal(await answer, null);
  const next = terminal.approve(approval());
  input.write(ENTER);
  assert.equal(await next, "Approve");
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
  "readLine returns each line, then the end of input": readLineReturnsEachLineThenTheEndOfInput,
  "Ctrl-C at readLine keeps the terminal open": ctrlCAtReadLineKeepsTheTerminalOpen,
  "an aborted readLine keeps later keys": anAbortedReadLineKeepsLaterKeys,
  "a non-TTY readLine reads one line per call": aNonTtyReadLineReadsOneLinePerCall,
  "an aborted readLine returns its draft for the next prompt": anAbortedReadLineReturnsItsDraftForTheNextPrompt,
  "a yielding readLine gives way to a prompt queued behind it":
    aYieldingReadLineGivesWayToAPromptQueuedBehindIt,
  "approve preselects the choice it names": approvePreselectsTheChoiceItNames,
  "a prompt opening while the live box takes keys drops them for a while":
    aPromptOpeningWhileTheLiveBoxTakesKeysDropsThemForAWhile,
  "a line prompt after a withdrawn card takes keys at once":
    aLinePromptAfterAWithdrawnCardTakesKeysAtOnce,
  "onActive says whether a turn left rows on the screen": onActiveSaysWhetherATurnLeftRows,
  "Escape at readLine with escape returns the draft": escapeAtReadLineReturnsTheDraft,
  "Escape at readLine without escape is ignored": escapeAtReadLineWithoutTheOptionIsIgnored,
  "watchKeys sends Escape and Ctrl-C and buffers the rest": watchKeysSendsEscapeAndCtrlCAndBuffersTheRest,
  "a prompt during a watch takes the keys, then watching resumes": aPromptDuringWatchTakesTheKeysThenWatchingResumes,
  "watchKeys on a plain or non-TTY terminal does nothing": watchKeysOnAPlainOrNonTtyTerminalDoesNothing,
  "a plain terminal writes no escapes and sets no raw mode": aPlainTerminalWritesNoEscapesAndSetsNoRawMode,
  "a plain terminal sets raw mode for a secret prompt": aPlainTerminalSetsRawModeForASecretPrompt,
  "a plain terminal writes a refused line's error and reads the next line afresh":
    aPlainTerminalWritesARefusedLinesErrorAndReadsTheNextLineAfresh,
  "a bracketed paste keeps its newlines": aBracketedPasteKeepsItsNewlines,
  "a framed readLine draws a box and a hint, then collapses to the line":
    aFramedReadLineDrawsABoxAndAHintThenCollapsesToTheLine,
  "a framed readLine with an editor keeps its box, emptied, under the line entered":
    aFramedReadLineWithAnEditorKeepsItsBoxEmptiedUnderTheLine,
  "a framed readLine enters its line at the terminal's width then":
    aFramedReadLineEntersItsLineAtTheWidthOfTheTerminalThen,
  "a framed readLine with noColor enters a plain line": aFramedReadLineWithNoColorEntersAPlainLine,
  "a question's rule is as wide as the terminal when it is drawn":
    aQuestionsRuleIsAsWideAsTheTerminalWhenItIsDrawn,
  "a framed readLine folds a long line inside the box": aFramedReadLineFoldsALongLineInsideTheBox,
  "a bracketed paste keeps its newlines in a framed readLine": aBracketedPasteKeepsItsNewlinesFramed,
  "a long paste in a framed readLine is one token, expanded on Enter":
    aLongPasteInAFramedReadLineIsOneTokenExpandedOnEnter,
  "a pasted token is atomic for the caret and the deleting keys":
    aPastedTokenIsAtomicForTheCaretAndTheDeletingKeys,
  "a short paste stays text, and a long line is a token": aShortPasteStaysTextAndALongLineIsAToken,
  "two pastes are numbered, and typed token text stays text":
    twoPastesAreNumberedAndTypedTokenTextStaysText,
  "a token wider than the box is cut": aTokenWiderThanTheBoxIsCut,
  "a paste split by a pause goes in whole": aPasteSplitByAPauseGoesInWhole,
  "Escape after a pause ends a paste that never ends": escapeAfterAPauseEndsAPasteThatNeverEnds,
  "Ctrl-C after a pause drops a paste that never ends": ctrlCAfterAPauseDropsAPasteThatNeverEnds,
  "a paste keeps its tabs, shown as spaces": aPasteKeepsItsTabsShownAsSpaces,
  "Escape erases a framed readLine": escapeErasesAFramedReadLine,
  "a framed readLine with a list moves into it and opens a row": aFramedReadLineWithAListMovesIntoItAndOpensARow,
  "a list row that goes gives the keys back, and the list redraws on its own":
    aListRowThatGoesGivesTheKeysBackAndTheListRedrawsOnItsOwn,
  "a long list shows what fits, the focused row among them": aLongListShowsWhatFitsTheFocusedRowAmongThem,
  "a paste taller than the screen shows the rows around the cursor":
    aPasteTallerThanTheScreenShowsTheRowsAroundTheCursor,
  "an Enter after the list shifted opens the run its pointer is on":
    anEnterAfterTheListShiftedOpensTheRunItsPointerIsOn,
  "Down on a folded line moves into the list only from its last row":
    downOnAFoldedLineMovesIntoTheListOnlyFromItsLastRow,
  "watchKeys with an editor gives it the keys but Ctrl-C": watchKeysWithAnEditorGivesItTheKeysButCtrlC,
  "showScreen draws on the alternate screen and gives the screen back":
    showScreenDrawsOnTheAlternateScreenAndGivesTheScreenBack,
  "showScreen says when a prompt waits behind it, and closes before that prompt":
    showScreenSaysWhenAPromptWaitsBehindItAndClosesFirst,
  "showScreen off a TTY does nothing": showScreenOffATtyDoesNothing,
  "a watch with an editor turns bracketed paste on, and a paste stays one line":
    aWatchWithAnEditorTurnsBracketedPasteOnAndAPasteStaysOneLine,
  "the exit hook leaves the alternate screen": theExitHookLeavesTheAlternateScreen,
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
  "a scalar editor checks its text as the run would and stays open":
    aScalarEditorChecksItsTextAsTheRunWouldAndStaysOpen,
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
  "an inbox input is never asked": anInboxInputIsNeverAsked,
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
  "Ctrl-D in a confirm is the end of input": ctrlDInAConfirmIsTheEndOfInput,
  "Ctrl-Q interrupts a confirm": ctrlQInterruptsAConfirm,
  "Ctrl-Q interrupts a select but not a text editor": ctrlQInterruptsASelectButNotATextEditor,
  "an optional string's blank is Unicode whitespace": anOptionalStringBlankIsUnicodeWhitespace,
  "a closed terminal prompts nothing": aClosedTerminalPromptsNothing,
  "a failing output fails the call": aFailingOutputFailsTheCall,
  "a question set shows the panel and a rule per question": aQuestionSetShowsThePanelAndARulePerQuestion,
  "a single select shows descriptions and returns the bare label": aSingleSelectShowsDescriptionsAndReturnsTheBareLabel,
  "Other and text questions fill a record keyed by header": otherAndTextQuestionsFillARecordKeyedByHeader,
  "a checkbox toggles, inverts and returns choice order": aCheckboxTogglesInvertsAndReturnsChoiceOrder,
  "Ctrl-Q interrupts a checkbox": ctrlQInterruptsACheckbox,
  "options are one select with Other and no panel": optionsAreOneSelectWithOtherAndNoPanel,
  "a plain ask is a text prompt": aPlainAskIsATextPrompt,
  "an ask shows control characters instead of sending them": anAskShowsControlCharactersInsteadOfSendingThem,
  "a choice shows control characters and answers the choice as given":
    aChoiceShowsControlCharactersAndAnswersTheChoiceAsGiven,
  "an aborted askUser throws the reason and leaves later keys buffered":
    anAbortedAskUserThrowsTheReasonAndLeavesLaterKeysBuffered,
  "a queued askUser whose signal aborts never draws": aQueuedAskUserWhoseSignalAbortsNeverDraws,
  "a question set aborted midway discards the partial record": aQuestionSetAbortedMidwayDiscardsThePartialRecord,
  "a failing write after an abort still throws the reason": aFailingWriteAfterAnAbortStillThrowsTheReason,
  "a failing output fails a question set": aFailingOutputFailsAQuestionSet,
  "Ctrl-C throws the reason onStop aborted and refuses queued calls":
    ctrlCThrowsTheReasonOnStopAbortedAndRefusesQueuedCalls,
  "Ctrl-C without an abort throws interrupt": ctrlCWithoutAnAbortThrowsInterrupt,
  "the end of input outranks the abort it causes and refuses queued calls":
    theEndOfInputOutranksTheAbortItCausesAndRefusesQueuedCalls,
  "close ends an active askUser and refuses later ones": closeEndsAnActiveAskUserAndRefusesLaterOnes,
  "throwing hooks still restore and refuse queued calls": throwingHooksStillRestoreAndRefuseQueuedCalls,
  "concurrent askUser calls are served in FIFO order": concurrentAskUserCallsAreServedInFifoOrder,
  "a multi-line text label is written once above the frame": aMultiLineTextLabelIsWrittenOnceAboveTheFrame,
  "a multi-line choice label is written once above the frame": aMultiLineChoiceLabelIsWrittenOnceAboveTheFrame,
  "a choice title with a newline redraws without stale rows": aChoiceTitleWithANewlineRedrawsWithoutStaleRows,
  "a paste into a select changes nothing": aPasteIntoASelectChangesNothing,
  "combining marks take no column": combiningMarksTakeNoColumn,
  "a key typed soon after Escape is kept": aKeyTypedSoonAfterEscapeIsKept,
  "Ctrl-N and Ctrl-P move in a select and a checkbox": ctrlNAndCtrlPMoveInASelectAndACheckbox,
  "a non-TTY turn still has the exit hook": aNonTtyTurnStillHasTheExitHook,
  "a child exits on its own after an answer with stdin open": aChildExitsOnItsOwnAfterAnAnswerWithStdinOpen,
  "a child exits on its own after Ctrl-C with stdin open": aChildExitsOnItsOwnAfterCtrlCWithStdinOpen,
  "approve: Enter picks the first choice and Down the next": approveEnterPicksTheFirstChoiceAndDownTheNext,
  "approve shows its text sanitized": approveShowsItsTextSanitized,
  "a framed approval numbers the diff and colours its lines": aFramedApprovalNumbersTheDiffAndColoursItsLines,
  "a framed approval folds a long line inside the card": aFramedApprovalFoldsALongLineInsideTheCard,
  "a framed approval cuts a long title": aFramedApprovalCutsALongTitle,
  "approve off a TTY returns null without reading": approveOffATtyReturnsNullWithoutReading,
  "Ctrl-C in approve rejects and keeps the terminal open": ctrlCInApproveRejectsAndKeepsTheTerminalOpen,
  "an aborted approve returns null": anAbortedApproveReturnsNull,
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
