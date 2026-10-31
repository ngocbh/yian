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
import { TAKE } from "../tools/context.ts";
import type { ChildEvent } from "./chat_events.ts";
import type { ChildStatus, LiveChild } from "./chat_runs.ts";
import { MarkdownLines, renderMarkdown, TAB } from "./markdown.ts";
import { ANSI, cells, clip, columnsOf, fit, rowsOf, wrapLine } from "./cells.ts";
import { liveRows, type LiveRows, type RowCursor } from "./live_rows.ts";
import {
  box,
  type Colored,
  elapsedTag,
  paint,
  type Span,
  type Style,
  userRows,
  wrapSpans,
} from "./render.ts";
import { sanitize, sequenceSpans } from "./sanitize.ts";
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

/**
 * `path` with the home shown as `~`: `home`, or else `realHome` (the home's real path, for a home
 * reached through a symlink), each when set and not the root.
 */
export function homePath(path: string, home: string | undefined, realHome?: string): string {
  for (const each of [home, realHome]) {
    const trimmed = each?.replace(/[\\/]+$/, "");
    if (!trimmed) continue;
    if (path === trimmed) return "~";
    for (const sep of ["/", "\\"]) {
      if (path.startsWith(trimmed + sep)) return `~${path.slice(trimmed.length)}`;
    }
  }
  return path;
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

/**
 * What a child run does now, in a word or two, and its colour: `running`, `asks` (a question
 * waits for an answer), `waits for a message`, `waits` (for a timer), or how it ended.
 */
export function runStatusLabel(status: ChildStatus): [string, Style] {
  if (status.status === "running") return ["running", "cyan"];
  if (status.status !== "waiting") return [status.status, CARD_STYLE[status.status]!];
  if ((status.asks ?? []).length > 0) return ["asks", "yellow"];
  const waits = (status.waits ?? []) as readonly { tool?: unknown; settled?: unknown }[];
  const take = waits.some((wait) => wait.tool === TAKE && wait.settled !== true);
  return take ? ["waits for a message", "yellow"] : ["waits", "dim"];
}

/**
 * The run list's rows, one per child `children` lists (`ChildRuns.live`): its id, its flow and
 * `● <what it does> · <time since it started>`, coloured by what it does, the columns lined up
 * and each row cut to `width`.
 */
export function runRows(
  children: readonly LiveChild[],
  now: number,
  width: number,
  out: Colored,
  env: Env,
): string[] {
  const ids = Math.max(0, ...children.map((child) => cells(child.id)));
  const names = Math.max(0, ...children.map((child) => cells(sanitize(child.name))));
  return children.map((child) => {
    const [label, style] = runStatusLabel(child.status);
    const since = paint(`· ${elapsedTag(now - child.since).slice(1, -1)}`, "dim", out, env);
    const name = sanitize(child.name);
    const head = `${child.id.padEnd(ids)}  ${name}${" ".repeat(names - cells(name))}`;
    return clip(`${head}  ${paint(`● ${label}`, style, out, env)} ${since}`, width);
  });
}

/** Whether `nodeId` is a node named `reply`, at any depth: the chat's assistant. */
function isReply(nodeId: string): boolean {
  return nodeId.slice(nodeId.lastIndexOf("/") + 1) === "reply";
}

const SHOW_CURSOR = "\x1b[?25h";

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
  /**
   * Whether the block shows anything yet: its blank lines before its first text are dropped, and
   * it opens (with the blank line before it) only once text comes.
   */
  opened: boolean;
}

/** Whether `row` shows nothing but spaces. */
function blank(row: string): boolean {
  return row.replace(ANSI, "").trim() === "";
}

/**
 * The input box the view keeps under the spinner while a turn runs (the terminal's `LineEditor`):
 * its rows, and where its cursor is among them (null to hide it).
 */
export interface LiveInput {
  lines(): string[];
  cursor(): { row: number; before: string } | null;
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
 * The bottom of the screen is live: the reply streaming in, while a turn runs the spinner, and
 * while no prompt has the terminal the input box (`input`) under them. The live rows are the
 * output's shared `LiveRows`, which the prompts draw on too, so a prompt opens over the box as it
 * is and the box comes back where the prompt left it. Text reaches the screen three ways. `print`
 * (the writer) takes whole lines from the rest of the chat; `#commit` prints rows for good above
 * the live rows, in the same write that draws them again; `#redraw` repaints the live rows in
 * place, only those that changed. Before anything is printed for good, the rows of the reply
 * finished so far are printed for good too (`#settleBlock`), so nothing printed lands above text
 * it came after.
 *
 * The reply's model calls are tracked by `#block` (the call streaming in), `#streamed` (whether
 * the reply's last call streamed, so its output isn't printed again) and `#failed` (an attempt
 * that failed, waiting to be followed by `(retrying)` or `(failed)`).
 */
export class ChatView {
  /** The hint under the framed prompt. */
  readonly hint = CHAT_HINT;
  /** A writer printing whole lines above the live rows (`print`). */
  readonly writer: Writer;
  readonly #output: ViewStream;
  readonly #env: Env;
  readonly #clock: Scheduler;
  /** The live rows on the output, shared with the prompts drawn on it. */
  readonly #live: LiveRows;
  /** The input box drawn under the spinner, if any. */
  #input: LiveInput | null = null;
  /** What is written while held (`hold`), written once let go; null while not held. */
  #held: string[] | null = null;
  /** What `print` was given after its last newline, printed once the line ends. */
  #pending = "";
  #block: Block | null = null;
  #streamed = false;
  /**
   * The failed attempt of the reply's model call: the rows it left printed for good, if any.
   * `(retrying)` follows them once the next attempt starts, `(failed)` if the turn ends first.
   */
  #failed: { node: string; rows: string[] } | null = null;
  readonly #onResize = () =>
    this.#batch(() => {
      // The live rows were drawn one screen row each; a terminal that reflowed them took more,
      // which leaves fragments above, but erasing from higher could wipe text printed for good on
      // one that cut them instead.
      this.#live.lose();
      this.#trim();
      this.#redraw();
    });
  /** What was printed last, for the blank line before the next item. */
  #last: "tool" | "text" | "other" | null = null;
  /** Whether the last row printed for good is blank, so no second blank row follows it. */
  #blank = false;
  /** Whether the turn running printed anything. */
  #turnPrinted = false;
  /** Set while a turn runs: its start time on the clock, and what stops its spinner. */
  #turn: { start: number; stop: AbortController } | null = null;
  #frame = 0;
  /** The model the spinner names, if any (`model`). */
  #model: string | null = null;
  /** Set while a prompt has the terminal. */
  #paused = false;
  /** How deep the calls drawing one frame are (`#batch`), and whether that frame is due. */
  #depth = 0;
  #due = false;
  /** Shows the cursor if the process exits while a turn has it hidden. */
  readonly #showOnExit = () => this.#output.write(SHOW_CURSOR);

  /**
   * @param clock The spinner's own clock: never the run's, which a test may hold or drive.
   */
  constructor(output: ViewStream, env: Env, clock: Scheduler) {
    this.#output = output;
    this.#env = env;
    this.#clock = clock;
    this.#live = liveRows(output);
    this.writer = {
      write: async (text) => this.print(text),
      isTTY: true,
      get columns() {
        return columnsOf(output);
      },
    };
  }

  /**
   * End the view: a turn still running ends, a line left unfinished is printed, and the input box
   * goes.
   */
  close(): void {
    this.#batch(() => {
      if (this.#turn !== null) this.turn(false);
      this.#flushPending();
      this.#input = null;
      this.#live.hideCursor(false);
      this.#redraw();
    });
  }

  /**
   * Print `text` above the live rows, a line once it ends: a piece without a newline waits for the
   * rest of its line (or the turn's end, or `close`).
   */
  print(text: string): void {
    this.#batch(() => {
      const lines = (this.#pending + text).split(/\r?\n/);
      this.#pending = lines.pop()!;
      if (lines.length === 0) return;
      this.#last = "other";
      this.#commit(lines);
    });
  }

  /** Name `model` (`provider/model`) on the spinner from now on; none when null. */
  model(model: string | null): void {
    this.#batch(() => {
      this.#model = model;
      this.#redraw();
    });
  }

  /** `text` (whole lines) dim, for the chat's notes. */
  note(text: string): string {
    return text.replace(/[^\n]+/g, (line) => this.#paint(line, "dim"));
  }

  /**
   * A turn starts (`true`) or ends. Meanwhile the spinner runs, the cursor is hidden and a resize
   * redraws the live rows; the end prints what is left.
   */
  turn(active: boolean): void {
    this.#batch(() => {
      if (active) {
        if (this.#turn !== null) return;
        this.#turn = { start: this.#clock.now(), stop: new AbortController() };
        this.#last = "other";
        this.#output.on?.("resize", this.#onResize);
        process.once("exit", this.#showOnExit);
        this.#live.hideCursor(true);
        void this.#spin(this.#turn.stop.signal);
        this.#redraw();
        return;
      }
      if (this.#turn === null) return;
      try {
        this.#finishBlock();
        this.#settleFailed("(failed)");
        this.#flushPending();
        this.#turn.stop.abort();
        this.#turn = null;
        // A blank line between what the turn printed and the prompt.
        if (this.#turnPrinted) this.#blankLine();
        this.#turnPrinted = false;
        this.#last = null;
        this.#live.hideCursor(false);
        this.#redraw();
      } finally {
        this.#turn?.stop.abort();
        this.#turn = null;
        this.#output.off?.("resize", this.#onResize);
        process.removeListener("exit", this.#showOnExit);
        // The frame drawn as this call ends shows the cursor again.
        this.#live.hideCursor(false);
      }
    });
  }

  /**
   * A prompt takes the terminal (`true`) or gives it back, having left rows on the screen when
   * `printed`. Meanwhile the prompt draws over the live rows, and they come back as it ends.
   */
  active(active: boolean, printed = false): void {
    this.#batch(() => {
      this.#paused = active;
      if (active) return;
      if (printed) this.#blank = false;
      // The prompt shows the cursor as it ends; the turn hides it again.
      if (this.#turn !== null) this.#live.hideCursor(true);
      this.#redraw();
    });
  }

  /**
   * Keep `input` under the live rows (`null` for none), under the spinner while a turn runs, its
   * cursor shown there; call `refresh` once it changes. Between turns it stays where the prompt
   * that comes next draws the same box.
   */
  input(input: LiveInput | null): void {
    this.#batch(() => {
      this.#input = input;
      this.#redraw();
    });
  }

  /** Draw the live rows again: the input box changed. */
  refresh(): void {
    this.#batch(() => {
      this.#redraw();
    });
  }

  /**
   * Print `line`, a line sent from the input box, as the prompt leaves one (`userRows`): `> <line>`
   * on the user's gray band, wrapped at its spaces onto as many rows as it needs.
   */
  echo(line: string): void {
    this.#batch(() => {
      this.#last = "other";
      const colour = this.#output.isTTY === true && !this.#env("NO_COLOR");
      this.#commit(userRows(sanitize(line).replaceAll("\n", "⏎"), this.#width(), colour));
    });
  }

  /**
   * Hold what the view writes (`true`) while another screen is up, and write it all once let go,
   * so the screen it comes back to is drawn as if it had stayed.
   */
  hold(held: boolean): void {
    if (held) {
      this.#held ??= [];
      return;
    }
    const text = this.#held?.join("") ?? "";
    this.#held = null;
    if (text !== "") this.#output.write(text);
  }

  /** A chunk of streamed text: a reply's is drawn after `⏺ `, a new model call starting a block. */
  onStream(chunk: StreamChunk): void {
    this.#batch(() => {
      if (!isReply(chunk.nodeId) || chunk.n === null || chunk.chunk === "") return;
      let block = this.#block;
      if (block === null || block.node !== chunk.nodeId || block.n !== chunk.n) {
        this.#finishBlock();
        // Its width is fixed now: after a resize its rows are cut to the new width, not wrapped
        // again.
        const markdown = new MarkdownLines(this.#width() - 2, this.#output, this.#env);
        const fresh = { markdown, rows: [], printed: 0, partial: "", opened: false };
        block = { node: chunk.nodeId, n: chunk.n, ...fresh };
        this.#block = block;
        this.#streamed = true;
      }
      block.partial += chunk.chunk;
      // A `\r` at the end may be half of a `\r\n`: it waits for the next chunk.
      const held = block.partial.endsWith("\r");
      const lines = (held ? block.partial.slice(0, -1) : block.partial).split(/\r\n|\n|\r/);
      block.partial = lines.pop()! + (held ? "\r" : "");
      const rows = lines.flatMap((line) => block.markdown.line(line));
      if (!this.#open(block, rows)) return;
      block.rows.push(...rows);
      if (!this.#trim()) this.#redraw();
    });
  }

  /** The run's events: the reply's model and tool calls, and its end. */
  onEvent(event: RunEvent): void {
    this.#batch(() => {
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
        const rows = block.printed > 0 ? this.#prefixed(block.rows, block.printed) : [];
        this.#failed = { node: block.node, rows };
        this.#redraw();
      } else if (event instanceof ToolCallStarted) {
        this.#finishBlock();
        this.#gap("tool");
        const name = sanitize(event.name);
        const room = Math.max(1, this.#width() - cells(name) - 4);
        const args = fit(sanitize(shortArgs(event.arguments)), room);
        this.#commit([`${this.#paint("⏺", "cyan")} ${this.#paint(name, "bold")}(${args})`]);
      } else if (event instanceof ToolCallFinished) {
        const { error, result } = event;
        const text =
          error === null ? (result ?? "") : error === "interrupted" ? error : `error: ${error}`;
        const style = error === null ? "dim" : error === "interrupted" ? "yellow" : "red";
        this.#result(summary(text, this.#width() - 5), style);
        this.#last = "tool";
      } else if (event instanceof NodeSucceeded) this.#finishBlock();
    });
  }

  /** The reply's output, `text`: drawn as a block unless its model call streamed it already. */
  reply(text: string): void {
    this.#batch(() => {
      if (this.#streamed) {
        this.#streamed = false;
        return;
      }
      // A reply of only tool calls ends with no text: it has nothing to show.
      if (text.trim() === "") return;
      this.#finishBlock();
      this.#gap("text");
      const rows = renderMarkdown(text, this.#width() - 2, this.#output, this.#env).split("\n");
      this.#commit(this.#prefixed(rows, 0));
    });
  }

  /**
   * A child event the chat sent its assistant: a `⏺ r1 <verb>` card with its body, printed above
   * a reply still streaming, which goes on below it.
   */
  childEvent(event: ChildEvent): void {
    this.#batch(() => {
      this.#gap("tool");
      const card = event.card;
      const style = card === undefined ? "cyan" : CARD_STYLE[card.status]!;
      const head = card === undefined ? "yian" : `${sanitize(event.run ?? "")} ${card.verb}`;
      const title = this.#paint(fit(head, this.#width() - 2), "bold");
      this.#commit([`${this.#paint("⏺", style)} ${title}`]);
      const body = card === undefined ? event.text : card.body;
      if (body !== "") this.#result(summary(body, this.#width() - 5), "dim");
      this.#last = "tool";
    });
  }

  /** Print what a failed attempt left, then `note` dim in place of its text. */
  #settleFailed(note: string): void {
    const failed = this.#failed;
    if (failed === null) return;
    this.#failed = null;
    this.#commit([...failed.rows, `  ${this.#paint(note, "dim")}`]);
  }

  /** Print the line `print` holds unfinished, if any. */
  #flushPending(): void {
    if (this.#pending === "") return;
    this.print("\n");
  }

  /**
   * The line still coming, `raw`, as it is drawn: sanitized, tabs expanded, and a held `\r` or a
   * CSI sequence not finished yet at its end left out until the rest arrives.
   */
  #partialText(raw: string): string {
    return sanitize(raw.replace(/\r$/, "").replace(/\x1b(\[[0-?]*[ -/]*)?$/, "")).replaceAll("\t", TAB);
  }

  /** The rows the line still coming in `block` takes: each one's start in `#partialText`, and its text. */
  #partialRows(block: Block): [number, string][] {
    const text = this.#partialText(block.partial);
    if (text === "") return [];
    const width = Math.max(1, this.#width() - 2);
    return wrapLine(text, width).map(([start, end]) => [start, text.slice(start, end)]);
  }

  /**
   * Keep the streaming block's live rows within the screen: rows beyond what it can hold are
   * printed for good, the line still coming included, and only the rest can be taken back.
   * Returns whether it printed any (and so drew the live rows again).
   */
  #trim(): boolean {
    const block = this.#block;
    if (block === null) return false;
    // The spinner and the input box stay below the block.
    const input = this.#paused ? null : this.#input;
    const below = input === null ? 0 : input.lines().length;
    const keep = Math.max(1, rowsOf(this.#output) - 4 - below);
    const partial = this.#partialRows(block);
    let excess = block.rows.length + partial.length - keep;
    if (excess <= 0) return false;
    const done = block.rows.splice(0, Math.min(excess, block.rows.length));
    excess -= done.length;
    if (excess > 0) {
      // The line's head goes as written, never as Markdown: it is past taking back. What is left
      // stays raw, so a sequence or a `\r\n` split across chunks still reads whole.
      const [head, rest] = this.#splitRaw(block.partial, partial[excess]![0]);
      done.push(...wrapLine(head, Math.max(1, this.#width() - 2)).map(([start, end]) => head.slice(start, end)));
      block.partial = rest;
    }
    this.#write(this.#prefixed(done, block.printed));
    block.printed += done.length;
    this.#redraw();
    return true;
  }

  /**
   * Split the raw line `raw` where its drawn text (`#partialText`) reaches `cut`, or as near before
   * as a split leaves the drawn head a prefix of the whole: the head as drawn, and the raw rest.
   */
  #splitRaw(raw: string, cut: number): [string, string] {
    const whole = this.#partialText(raw);
    if (whole === raw) {
      const at = cut > 0 && /[\udc00-\udfff]/.test(raw[cut] ?? "") ? cut - 1 : cut;
      return [raw.slice(0, at), raw.slice(at)];
    }
    // A split goes between code points outside the whole line's sequences: there the head draws as
    // the start of the whole, longer the further in it is, so the last that fits is a binary search
    // away. A head ending in an ESC, or a CSI's start, that the whole shows draws as if it weren't
    // there, so a split after it would lose it.
    const inside = new Uint8Array(raw.length + 1);
    for (const [start, end] of sequenceSpans(raw)) inside.fill(1, start + 1, end);
    for (const match of raw.matchAll(/\x1b(?:\[[0-?]*[ -/]*)?/g)) {
      inside.fill(1, match.index + 1, Math.min(match.index + match[0].length + 1, raw.length));
    }
    const splits: number[] = [];
    for (let at = 0; at <= raw.length; at += at < raw.length ? String.fromCodePoint(raw.codePointAt(at)!).length : 1) {
      if (inside[at] === 0) splits.push(at);
    }
    let [low, high] = [0, splits.length - 1];
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (this.#partialText(raw.slice(0, splits[mid])).length <= cut) low = mid;
      else high = mid - 1;
    }
    const head = this.#partialText(raw.slice(0, splits[low]));
    return whole.startsWith(head) ? [head, raw.slice(splits[low])] : ["", raw];
  }

  /** Print `rows` of a result under the line above: `  ⎿  ` before the first. */
  #result(rows: string[], style: Style): void {
    this.#commit(rows.map((row, i) => `${i === 0 ? "  ⎿  " : "     "}${this.#paint(row, style)}`));
  }

  /** A blank line before an item of `kind`, unless it follows one of its kind (tool lines). */
  #gap(kind: "tool" | "text"): void {
    const gap = kind === "text" || this.#last !== "tool";
    this.#last = kind;
    if (gap) this.#blankLine();
  }

  /** Print a blank line, unless the last row printed is one. */
  #blankLine(): void {
    // The reply's finished rows go first: the row before the blank may be one of them.
    this.#settleBlock();
    if (!this.#blank) this.#commit([""]);
  }

  /**
   * Whether `block` shows anything once `rows` (rows to add to it) come: a block not opened yet
   * drops the blank rows at their head, and opens, after the blank line before it, once a row or
   * its line still coming has text.
   */
  #open(block: Block, rows: string[]): boolean {
    if (block.opened) return true;
    // A model's text may start with blank lines (before a tool call, say): they show nothing.
    while (rows.length > 0 && blank(rows[0]!)) rows.shift();
    if (rows.length === 0 && this.#partialText(block.partial).trim() === "") return false;
    // Opened after the gap, so the gap's redraw doesn't draw its rows yet.
    this.#gap("text");
    block.opened = true;
    return true;
  }

  /** Print the streaming block's finished rows for good; its unfinished line stays live. */
  #settleBlock(): void {
    const block = this.#block;
    if (block === null || block.rows.length === 0) return;
    const rows = this.#prefixed(block.rows, block.printed);
    block.printed += block.rows.length;
    block.rows = [];
    this.#write(rows);
  }

  /** End the block streaming in: its last line and what closes it are printed for good. */
  #finishBlock(): void {
    const block = this.#block;
    if (block === null) return;
    const partial = block.partial.replace(/\r$/, "");
    block.partial = "";
    const rows = [...(partial === "" ? [] : block.markdown.line(partial)), ...block.markdown.end()];
    // Blank lines at its end show nothing either.
    while (rows.length > 0 && blank(rows.at(-1)!)) rows.pop();
    if (rows.length === 0) {
      while (block.rows.length > 0 && blank(block.rows.at(-1)!)) block.rows.pop();
    }
    if (this.#open(block, rows)) block.rows.push(...rows);
    this.#settleBlock();
    this.#block = null;
    this.#redraw();
  }

  /** `rows` of a block whose first `from` rows are printed: its first row after `⏺ `. */
  #prefixed(rows: readonly string[], from: number): string[] {
    return rows.map((row, i) => `${from + i === 0 ? "⏺ " : "  "}${row}`);
  }

  /**
   * The live rows, each cut to the width: the block streaming in, the spinner while a turn runs,
   * and the input box; and the cursor in the box while a turn runs.
   */
  #liveRows(): { rows: string[]; cursor: RowCursor | null } {
    const rows: string[] = [];
    const block = this.#block;
    if (block !== null && block.opened) {
      // The line still coming is shown as written; it is drawn as Markdown once it ends.
      const partial = this.#partialRows(block).map(([, text]) => text);
      rows.push(...this.#prefixed([...block.rows, ...partial], block.printed));
    }
    if (this.#turn !== null) {
      const seconds = Math.floor((this.#clock.now() - this.#turn.start) / 1000);
      const frame = SPIN_FRAMES[this.#frame % SPIN_FRAMES.length]!;
      const model = this.#model === null ? "" : ` · ${sanitize(this.#model)}`;
      const status = `Thinking… (${seconds}s${model} · esc to interrupt)`;
      rows.push(`${this.#paint(frame, "cyan")} ${this.#paint(status, "dim")}`);
    }
    const input = this.#input?.lines() ?? [];
    rows.push(...input);
    // Each row fits the width, so each takes one row of the screen; and no more rows than the
    // screen has, less one, are drawn, or the cursor couldn't reach the top ones to repaint them.
    const kept = rows.slice(Math.max(0, rows.length - (rowsOf(this.#output) - 1)));
    const shown = this.#turn === null ? null : (this.#input?.cursor() ?? null);
    const line = shown === null ? -1 : kept.length - input.length + shown.row;
    // A cursor on a row cut off the top isn't shown.
    const cursor = shown !== null && line >= 0 ? { line, before: shown.before } : null;
    return { rows: kept.map((row) => clip(row, this.#width())), cursor };
  }

  /** Print `rows` for good above the live rows, after the reply's finished rows, then draw those again. */
  #commit(rows: readonly string[]): void {
    if (rows.length === 0) return;
    this.#settleBlock();
    this.#write(rows);
    this.#redraw();
  }

  /** Print `rows` for good above the live rows with the next draw of them (`#redraw`). */
  #write(rows: readonly string[]): void {
    if (rows.length === 0) return;
    if (this.#turn !== null) this.#turnPrinted = true;
    this.#blank = blank(rows.at(-1)!);
    this.#live.stage(rows);
  }

  /**
   * Draw the live rows again, in place of those on screen, with the rows to print for good above
   * them. While a prompt has the terminal, only those rows are printed, above the prompt.
   */
  #redraw(): void {
    if (this.#depth > 0) {
      this.#due = true;
      return;
    }
    const emit = (text: string) => this.#emit(text);
    if (this.#paused) return this.#live.flush(emit);
    const { rows, cursor } = this.#liveRows();
    this.#live.draw(rows, cursor, emit);
  }

  /**
   * Run `draw`, whose redraws make one frame drawn as the outermost such call ends: each call of
   * the view draws at most once.
   */
  #batch(draw: () => void): void {
    this.#depth += 1;
    try {
      draw();
    } finally {
      this.#depth -= 1;
      if (this.#depth === 0 && this.#due) {
        this.#due = false;
        this.#redraw();
      }
    }
  }

  /** Write `text` on the output, or keep it while held. */
  #emit(text: string): void {
    if (this.#held !== null) this.#held.push(text);
    else this.#output.write(text);
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
