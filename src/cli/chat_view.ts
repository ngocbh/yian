/**
 * The look of `yian chat` on a terminal: a banner box, the assistant's replies as `⏺` blocks of
 * Markdown streamed as they come, its tool calls and its child runs' events as `⏺` lines with a
 * `⎿` result, a spinner while a turn runs, and a framed prompt (drawn by the terminal). Everything
 * the chat prints goes through one writer, which keeps the spinner's line clear of other text.
 *
 * Off a terminal, or with `NO_COLOR`, the chat keeps its plain lines and none of this is used.
 *
 * Knows about: the run's events, the chat's child events, the box, colour and Markdown helpers,
 * the terminal's cell widths, the clock the spinner ticks on and the sanitizing of text printed on
 * the terminal.
 */

import type { Env } from "../defaults.ts";
import {
  ModelCallFinished,
  ModelCallStarted,
  NodeSucceeded,
  type RunEvent,
  type StreamChunk,
  ToolCallFinished,
  ToolCallStarted,
} from "../events.ts";
import type { Scheduler } from "../platform.ts";
import type { ChildEvent } from "./chat_events.ts";
import { MarkdownLines, TAB } from "./markdown.ts";
import { box, type Colored, fit, paint, type Span, type Style, wrapLine, wrapSpans } from "./render.ts";
import { sanitize } from "./sanitize.ts";
import { cells, charCells } from "./terminal.ts";
import type { CliIo, Writer } from "./utils.ts";

/** The hint under the chat's framed prompt. */
export const CHAT_HINT = "/exit to leave · ctrl-c interrupts · esc withdraws";

/** What the chat's banner shows. */
export interface BannerInfo {
  /** `provider/model` of the assistant's replies. */
  model: string;
  /** How the model is reached (`ChatGPT login`, `API key`), or null when not known. */
  login: string | null;
  /** The folder the chat works in, as shown (`~/...`). */
  cwd: string;
  /** The flow in focus as the assistant names it, or null for none. */
  focus: string | null;
}

/** An output the view reads: whether it is a TTY, so colour is on, and its size. */
export type ViewOutput = Colored & { readonly columns?: number; readonly rows?: number };

/** An output the view draws on; a TTY stream emits `resize` when the terminal's size changes. */
export type ViewStream = Pick<NodeJS.WritableStream, "write"> &
  ViewOutput &
  Partial<Pick<NodeJS.EventEmitter, "on" | "off">>;

/**
 * Whether `yian chat` draws its look on `io`: stdin, stdout and stderr are all a TTY and
 * `NO_COLOR` is unset or empty.
 */
export function richChat(io: CliIo): boolean {
  const stderr = io.stderrStream as { isTTY?: boolean };
  return io.stdin.isTTY === true && stderr.isTTY === true && io.stdout.isTTY === true && !io.env("NO_COLOR");
}

/**
 * How a connection (an agent config's `_connection`) reaches the model: `ChatGPT login` for a
 * subscription, `API key` for a key that is there, null for anything else.
 */
export function loginLabel(connection: unknown): string | null {
  if (typeof connection !== "object" || connection === null) return null;
  const { method, source } = connection as { method?: unknown; source?: unknown };
  if (method === "subscription") return "ChatGPT login";
  return method === "api_key" && source !== "missing" ? "API key" : null;
}

/** `path` with `home` (when set and not the root) shown as `~`. */
export function homePath(path: string, home: string | undefined): string {
  const trimmed = home?.replace(/[\\/]+$/, "");
  if (!trimmed) return path;
  if (path === trimmed) return "~";
  for (const sep of ["/", "\\"]) {
    if (path.startsWith(trimmed + sep)) return `~${path.slice(trimmed.length)}`;
  }
  return path;
}

/** The columns `out` is wide: as it reports, else 80. */
function columnsOf(out: ViewOutput): number {
  return out.columns || 80;
}

/** The rows `out` is high: as it reports, else 24. */
function rowsOf(out: ViewOutput): number {
  return out.rows || 24;
}

/** Width of the banner's label column: `  focus   `. */
const LABEL_CELLS = 10;

/**
 * The banner box across `out`'s width: `✻ yian chat`, then the model and how it is reached, the
 * folder and the flow in focus, each sanitized and wrapped under its label; ends in a newline.
 */
export function chatBanner(info: BannerInfo, out: ViewOutput, env: Env): string {
  const inner = Math.max(LABEL_CELLS + 1, columnsOf(out) - 4);
  const rows: [string, string][] = [
    [`✻ yian chat`, `${paint("✻", "cyan", out, env)} ${paint("yian chat", "bold", out, env)}`],
  ];
  const field = (label: string, spans: Span[]) => {
    const head = `  ${label.padEnd(LABEL_CELLS - 2)}`;
    wrapSpans(spans, inner - LABEL_CELLS, out, env).forEach(([plain, printed], i) => {
      const lead = i === 0 ? head : " ".repeat(LABEL_CELLS);
      rows.push([lead + plain, (i === 0 ? paint(lead, "dim", out, env) : lead) + printed]);
    });
  };
  const model: Span[] = [[sanitize(info.model), null]];
  if (info.login !== null) model.push([` (${info.login})`, "dim"]);
  field("model", model);
  field("cwd", [[sanitize(info.cwd), null]]);
  if (info.focus !== null) field("focus", [[sanitize(info.focus), null]]);
  return `${box(rows, "dim", out, env, "", inner)}\n`;
}

/** The most characters of one argument value a tool line shows. */
const ARG_CHARS = 40;

/** One argument value, short: a string quoted and cut, a list or record shown inline. */
function shortValue(value: unknown): string {
  if (typeof value === "string") {
    const quoted = JSON.stringify(value);
    return quoted.length > ARG_CHARS ? `${quoted.slice(0, ARG_CHARS - 1)}…"` : quoted;
  }
  if (Array.isArray(value)) return `[${value.map(shortValue).join(", ")}]`;
  if (typeof value === "object" && value !== null) {
    return `{${Object.entries(value).map(([key, item]) => `${key}: ${shortValue(item)}`).join(", ")}}`;
  }
  return String(value);
}

/**
 * A tool call's arguments, short: each value in order, a `path` or `run` bare and the others as
 * `shortValue` shows them.
 */
export function shortArgs(args: unknown): string {
  if (typeof args !== "object" || args === null || Array.isArray(args)) return args === null ? "" : shortValue(args);
  return Object.entries(args)
    .map(([key, value]) => ((key === "path" || key === "run") && typeof value === "string" ? value : shortValue(value)))
    .join(", ");
}

/** A JSON object's fields as `key: value` lines, a string value bare; null for other text. */
function jsonFields(text: string): string[] | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
  return Object.entries(parsed).map(
    ([key, value]) => `${key}: ${typeof value === "string" ? value : shortValue(value)}`,
  );
}

/**
 * A tool's result or an event's body on at most two rows of `width` cells, the second ending in
 * `…` when there is more: sanitized, its blank lines dropped, a `key:` line joined with the
 * indented lines under it (a JSON object's fields as such lines), each of several cut to half the
 * width, and the lines joined by ` · `.
 */
export function summary(text: string, width: number): string[] {
  const lines = sanitize(jsonFields(text.trim())?.join("\n") ?? text)
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.trim() !== "");
  const items: string[] = [];
  // Set while the last item is a `key:` line taking the indented lines under it.
  let header = false;
  for (const line of lines) {
    if (header && /^\s/.test(line)) {
      const last = items.length - 1;
      items[last] = `${items[last]}${items[last]!.endsWith(":") ? " " : ", "}${line.trim()}`;
      continue;
    }
    items.push(line.trim());
    header = line.trim().endsWith(":");
  }
  // Of several, no one item takes more than half the room, so a long path doesn't hide the rest.
  const room = items.length > 1 ? Math.max(1, Math.floor(width / 2)) : Infinity;
  const joined = items.map((item) => fit(item, room)).join(" · ");
  const rows = wrapLine(joined, Math.max(1, width)).map(([start, end]) => joined.slice(start, end));
  return rows.length <= 2 ? rows : [rows[0]!, fit(`${rows[1]} ${rows[2]}`, width)];
}

/** An SGR sequence, which takes no cells. */
const SGR_SEQUENCE = /^\x1b\[[0-9;]*m/;

/** `row`, styled, cut to at most `width` cells; its styles are reset when it is cut. */
function clip(row: string, width: number): string {
  if (cells(row) <= width) return row;
  let kept = "";
  let used = 0;
  let at = 0;
  while (at < row.length) {
    const sequence = SGR_SEQUENCE.exec(row.slice(at));
    if (sequence !== null) {
      kept += sequence[0];
      at += sequence[0].length;
      continue;
    }
    const char = String.fromCodePoint(row.codePointAt(at)!);
    if (used + charCells(char) > width) break;
    kept += char;
    used += charCells(char);
    at += char.length;
  }
  return `${kept}\x1b[0m`;
}

/** Whether `nodeId` is a node named `reply`, at any depth: the chat's assistant. */
function isReply(nodeId: string): boolean {
  return nodeId.slice(nodeId.lastIndexOf("/") + 1) === "reply";
}

/** The spinner's frames, drawn in turn. */
const SPIN_FRAMES = ["·", "✢", "✳", "✶", "✻", "✽"];
/** How often the spinner is redrawn. */
const SPIN_MS = 250;

/** One streamed reply: a model call's text, drawn as it comes. */
interface Block {
  readonly node: string;
  readonly n: number;
  readonly markdown: MarkdownLines;
  /** Rendered rows still in the live region, which a retry can take back. */
  rows: string[];
  /** Rows of this block already printed for good. */
  printed: number;
  /** The text after the last newline. */
  partial: string;
}

/** The colour of a child event's card, by the status it came from. */
const CARD_STYLE: Readonly<Record<string, Style>> = {
  succeeded: "green",
  failed: "red",
  cancelled: "yellow",
  waiting: "cyan",
  running: "cyan",
};

/**
 * The chat's view on a terminal, drawn on `output`: what `inboxChat` and `yian chat` print goes
 * through `writer`, the run's events through `onEvent` and `onStream`.
 *
 * The bottom of the screen is live: the reply streaming in and, while a turn runs and no prompt
 * has the terminal, the spinner. Whatever else is printed goes above it, and the live rows are
 * drawn again below.
 */
export class ChatView {
  /** The hint under the framed prompt. */
  readonly hint = CHAT_HINT;
  /** A writer printing whole lines above the live rows; a `\n` is written as `\r\n`. */
  readonly writer: Writer;
  readonly #output: ViewStream;
  readonly #env: Env;
  readonly #clock: Scheduler;
  /** The cells of each live row on screen; the cursor is at the end of the last. */
  #drawn: number[] = [];
  /**
   * The failed attempt of the reply's model call: its rows already printed for good, if any.
   * `(retrying)` follows them once the next attempt starts, `(failed)` if the turn ends first.
   */
  #failed: { node: string; rows: string[] } | null = null;
  readonly #onResize = () => this.#resized();
  #block: Block | null = null;
  /** Whether the reply's last model call streamed text, so its output isn't printed again. */
  #streamed = false;
  /** What was printed last, for the blank line before the next item. */
  #last: "tool" | "text" | "other" | null = null;
  /** Whether the turn running printed anything. */
  #turnPrinted = false;
  /** Set while a turn runs: its start time on the clock, and what stops its spinner. */
  #turn: { start: number; stop: AbortController } | null = null;
  #frame = 0;
  /** Set while a prompt has the terminal. */
  #paused = false;

  /**
   * @param clock The spinner's own clock: never the run's, which a test may hold or drive.
   */
  constructor(output: ViewStream, env: Env, clock: Scheduler) {
    this.#output = output;
    this.#env = env;
    this.#clock = clock;
    this.writer = {
      write: async (text) => this.print(text),
      isTTY: true,
      get columns() {
        return columnsOf(output);
      },
    };
    output.on?.("resize", this.#onResize);
  }

  /** Stop listening for the terminal's resizes. */
  close(): void {
    this.#output.off?.("resize", this.#onResize);
  }

  /** Print `text`, whole lines, above the live rows. */
  print(text: string): void {
    if (text === "") return;
    this.#last = "other";
    this.#commit(text.replace(/\r?\n$/, "").split(/\r?\n/));
  }

  /** `text` (whole lines) dim, for the chat's notes. */
  note(text: string): string {
    return text.replace(/[^\n]+/g, (line) => this.#paint(line, "dim"));
  }

  /** A turn starts (`true`) or ends: the spinner runs meanwhile; the end prints what is left. */
  turn(active: boolean): void {
    if (active) {
      if (this.#turn !== null) return;
      this.#turn = { start: this.#clock.now(), stop: new AbortController() };
      this.#last = "other";
      void this.#spin(this.#turn.stop.signal);
      this.#redraw();
      return;
    }
    this.#finishBlock();
    this.#settleFailed("(failed)");
    this.#turn?.stop.abort();
    this.#turn = null;
    // A blank line between what the turn printed and the prompt.
    if (this.#turnPrinted) this.#commit([""]);
    this.#turnPrinted = false;
    this.#last = null;
    this.#redraw();
  }

  /** A prompt takes the terminal (`true`) or gives it back: the live rows go meanwhile. */
  active(active: boolean): void {
    this.#paused = active;
    this.#redraw();
  }

  /** A chunk of streamed text: a reply's is drawn after `⏺ `, a new model call starting a block. */
  onStream(chunk: StreamChunk): void {
    if (!isReply(chunk.nodeId) || chunk.n === null || chunk.chunk === "") return;
    let block = this.#block;
    if (block === null || block.node !== chunk.nodeId || block.n !== chunk.n) {
      this.#finishBlock();
      this.#gap("text");
      const markdown = new MarkdownLines(this.#width() - 2, this.#output, this.#env);
      block = { node: chunk.nodeId, n: chunk.n, markdown, rows: [], printed: 0, partial: "" };
      this.#block = block;
      this.#streamed = true;
    }
    block.partial += chunk.chunk;
    // A `\r` at the end may be half of a `\r\n`: it waits for the next chunk.
    const held = block.partial.endsWith("\r");
    const lines = (held ? block.partial.slice(0, -1) : block.partial).split(/\r\n|\n|\r/);
    block.partial = lines.pop()! + (held ? "\r" : "");
    for (const line of lines) block.rows.push(...block.markdown.line(line));
    // Rows beyond what the screen can hold are printed for good, the line still coming included;
    // only the rest can be taken back.
    const keep = Math.max(1, rowsOf(this.#output) - 4);
    const partial = this.#partialRows(block);
    let excess = block.rows.length + partial.length - keep;
    if (excess <= 0) return this.#redraw();
    const done = block.rows.splice(0, Math.min(excess, block.rows.length));
    excess -= done.length;
    if (excess > 0) {
      // The line's head goes as written, never as Markdown: it is past taking back.
      done.push(...partial.slice(0, excess).map(([, printed]) => printed));
      block.partial = this.#partialText(block).slice(partial[excess]![0]);
    }
    this.#commit(this.#prefixed(done, block.printed));
    block.printed += done.length;
  }

  /** The run's events: the reply's model and tool calls, and its end. */
  onEvent(event: RunEvent): void {
    if (!("nodeId" in event) || !isReply(event.nodeId)) return;
    if (event instanceof ModelCallStarted) {
      this.#streamed = false;
      if (this.#failed?.node === event.nodeId) this.#settleFailed("(retrying)");
    } else if (event instanceof ModelCallFinished) {
      const block = this.#block;
      if (block === null || block.node !== event.nodeId || block.n !== event.n) return;
      if (event.error === null) return this.#finishBlock();
      // A failed attempt's text is taken back; a next attempt streams a block of its own.
      this.#block = null;
      this.#streamed = false;
      this.#failed = { node: block.node, rows: block.printed > 0 ? this.#prefixed(block.rows, block.printed) : [] };
      this.#redraw();
    } else if (event instanceof ToolCallStarted) {
      this.#finishBlock();
      this.#gap("tool");
      const name = sanitize(event.name);
      const args = fit(sanitize(shortArgs(event.arguments)), Math.max(1, this.#width() - cells(name) - 4));
      this.#commit([`${this.#paint("⏺", "cyan")} ${this.#paint(name, "bold")}(${args})`]);
    } else if (event instanceof ToolCallFinished) {
      const { error, result } = event;
      const text = error === null ? (result ?? "") : error === "interrupted" ? error : `error: ${error}`;
      const style = error === null ? "dim" : error === "interrupted" ? "yellow" : "red";
      this.#result(summary(text, this.#width() - 5), style);
      this.#last = "tool";
    } else if (event instanceof NodeSucceeded) this.#finishBlock();
  }

  /** The reply's output, `text`: drawn as a block unless its model call streamed it already. */
  reply(text: string): void {
    if (this.#streamed) {
      this.#streamed = false;
      return;
    }
    this.#finishBlock();
    this.#gap("text");
    const markdown = new MarkdownLines(this.#width() - 2, this.#output, this.#env);
    const rows = [...text.split(/\r\n|\n|\r/).flatMap((line) => markdown.line(line)), ...markdown.end()];
    this.#commit(this.#prefixed(rows, 0));
  }

  /** A child event the chat sent its assistant: a `⏺ r1 <verb>` card with its body. */
  childEvent(event: ChildEvent): void {
    this.#finishBlock();
    this.#gap("tool");
    const card = event.card;
    const style = card === undefined ? "cyan" : CARD_STYLE[card.status]!;
    const head = card === undefined ? "yian" : `${sanitize(event.run ?? "")} ${card.verb}`;
    this.#commit([`${this.#paint("⏺", style)} ${this.#paint(fit(head, this.#width() - 2), "bold")}`]);
    const body = card === undefined ? event.text : card.body;
    if (body !== "") this.#result(summary(body, this.#width() - 5), "dim");
    this.#last = "tool";
  }

  /** Print what a failed attempt left, then `note` dim in place of its text. */
  #settleFailed(note: string): void {
    const failed = this.#failed;
    if (failed === null) return;
    this.#failed = null;
    this.#commit([...failed.rows, `  ${this.#paint(note, "dim")}`]);
  }

  /** The line still coming in `block`, as it is drawn: sanitized, a held `\r` left out, tabs expanded. */
  #partialText(block: Block): string {
    return sanitize(block.partial.replace(/\r$/, "")).replaceAll("\t", TAB);
  }

  /** The rows the line still coming in `block` takes: each one's start in `#partialText`, and its text. */
  #partialRows(block: Block): [number, string][] {
    const text = this.#partialText(block);
    if (text === "") return [];
    const width = Math.max(1, this.#width() - 2);
    return wrapLine(text, width).map(([start, end]) => [start, text.slice(start, end)]);
  }

  /** Print `rows` of a result under the line above: `  ⎿  ` before the first. */
  #result(rows: string[], style: Style): void {
    this.#commit(rows.map((row, i) => `${i === 0 ? "  ⎿  " : "     "}${this.#paint(row, style)}`));
  }

  /** A blank line before an item of `kind`, unless it follows one of its kind (tool lines). */
  #gap(kind: "tool" | "text"): void {
    const blank = kind === "text" || this.#last !== "tool";
    this.#last = kind;
    if (blank) this.#commit([""]);
  }

  /** End the block streaming in: its last line and what closes it are printed for good. */
  #finishBlock(): void {
    const block = this.#block;
    if (block === null) return;
    this.#block = null;
    const partial = block.partial.replace(/\r$/, "");
    if (partial !== "") block.rows.push(...block.markdown.line(partial));
    block.rows.push(...block.markdown.end());
    this.#commit(this.#prefixed(block.rows, block.printed));
  }

  /** `rows` of a block whose first `from` rows are printed: its first row after `⏺ `. */
  #prefixed(rows: readonly string[], from: number): string[] {
    return rows.map((row, i) => `${from + i === 0 ? "⏺ " : "  "}${row}`);
  }

  /** The live rows: the block streaming in, then the spinner. */
  #liveRows(): string[] {
    const rows: string[] = [];
    const block = this.#block;
    if (block !== null) {
      // The line still coming is shown as written; it is drawn as Markdown once it ends.
      const partial = this.#partialRows(block).map(([, text]) => text);
      rows.push(...this.#prefixed([...block.rows, ...partial], block.printed));
    }
    if (this.#turn !== null && !this.#paused) {
      const seconds = Math.floor((this.#clock.now() - this.#turn.start) / 1000);
      const frame = SPIN_FRAMES[this.#frame % SPIN_FRAMES.length]!;
      rows.push(`${this.#paint(frame, "cyan")} ${this.#paint(`Thinking… (${seconds}s · esc to interrupt)`, "dim")}`);
    }
    // Each row fits the width, so each takes one row of the screen.
    return this.#paused ? [] : rows.map((row) => clip(row, this.#width()));
  }

  /**
   * The text that erases the live rows, leaving the cursor at the start of where they began. They
   * take one screen row each at the width they were drawn at; at a narrower width, as many as a
   * terminal that reflows its lines wraps them onto.
   */
  #erase(): string {
    const width = this.#width();
    const taken = this.#drawn.reduce((sum, used) => sum + Math.max(1, Math.ceil(used / width)), 0);
    if (taken === 0) return "";
    return `\r${taken > 1 ? `\x1b[${taken - 1}A` : ""}\x1b[J`;
  }

  /**
   * The terminal changed size: the live rows are erased as it reflowed them and drawn again at the
   * new width. A terminal that cuts long lines rather than reflowing them may keep a stale row
   * above them; the rows printed for good are left as the terminal shows them.
   */
  #resized(): void {
    this.#redraw();
  }

  /** Print `rows` for good above the live rows, then draw those again. */
  #commit(rows: readonly string[]): void {
    if (rows.length === 0) return;
    if (this.#turn !== null) this.#turnPrinted = true;
    this.#write(this.#erase() + rows.map((row) => `${row}\r\n`).join(""));
    this.#drawn = [];
    this.#redraw();
  }

  /** Draw the live rows again in place of those on screen. */
  #redraw(): void {
    const rows = this.#liveRows();
    const text = this.#erase() + rows.join("\r\n");
    this.#drawn = rows.map(cells);
    if (text !== "") this.#write(text);
  }

  #write(text: string): void {
    this.#output.write(text);
  }

  /** Turn the spinner once each `SPIN_MS`, until `signal` aborts. */
  async #spin(signal: AbortSignal): Promise<void> {
    while (!signal.aborted) {
      try {
        await this.#clock.sleep(SPIN_MS, signal);
      } catch {
        return;
      }
      this.#frame += 1;
      if (!this.#paused) this.#redraw();
    }
  }

  #width(): number {
    return columnsOf(this.#output);
  }

  #paint(text: string, style: Style): string {
    return paint(text, style, this.#output, this.#env);
  }
}
