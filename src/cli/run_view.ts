/**
 * A child run of `yian chat` shown on the whole screen, as its run list opens it: a header naming
 * the run, its flow, its status and its log, the run's progress read from its log and drawn as
 * `yian run` draws it (a `✓` or `✗` line per node that ended, a dim line per node still running
 * or waiting, an agent's text and tool calls as `⏺` lines with a `⎿` result), a dim `›` line per
 * record a host call on an inbox or the run wrote, with who made it, and the questions it waits on;
 * then a strip of its inboxes and a box. What the person does there presses the chat assistant's
 * own tools as the person (`ChildFlowOwner.press`): the run hears them relayed by the assistant,
 * and the assistant's conversation records each as their action.
 *
 * Knows about: the run log's records and its reader, the chat's child statuses, tool lines and
 * presses, the colour helpers, the terminal's cell widths, keys and input box, and the sanitizing
 * of text printed on the terminal.
 */

import type { ChildInterrupt, ChildStatus, PressArguments, Pressed } from "../child_flows.ts";
import { END_ID, START_ID } from "../compile/model.ts";
import type { Env } from "../defaults.ts";
import type { Author } from "../memory/base.ts";
import {
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  iterRunRecords,
  type NodeFailedRecord,
  type NodeRoutedRecord,
  type NodeStepRecord,
  PoolCommitRecord,
  recordTimeMs,
  RunCancelledRecord,
  RunLogChangedError,
} from "../runtime/log.ts";
import { TypedValue } from "../typesys/values.ts";
import { cells, fit, wrapLine } from "./cells.ts";
import { runStatusLabel, shortArgs, summary } from "./chat_view.ts";
import { TAB } from "./markdown.ts";
import {
  authorText,
  type Colored,
  elapsedTag,
  endedNodeText,
  failedNodeText,
  paint,
  type Style,
} from "./render.ts";
import { sanitize } from "./sanitize.ts";
import { joinDrafts, type Key, LineEditor, textDraft } from "./terminal.ts";

/** The most rows of a run's progress the view keeps: the last ones, the rest counted. */
const KEPT_ROWS = 2_000;

/**
 * One row of the progress: what its first line starts with (`lead`) and its later ones (`indent`)
 * when it wraps, its text, and its style.
 */
type Row = { lead: string; indent: string; text: string; style: Style | null };

/** Whether `nodeId` is a boundary node the engine adds, which has no authored meaning. */
function boundary(nodeId: string): boolean {
  const last = nodeId.slice(nodeId.lastIndexOf("/") + 1);
  return last === START_ID || last === END_ID;
}

/** `text` safe to draw on one row of the screen: sanitized, its tabs expanded. */
function shown(text: string): string {
  return sanitize(text).replaceAll("\t", TAB);
}

/**
 * A child run's progress, read from its log as it grows: each `rows` call reads the records
 * appended since the last one (the whole log again if it was rewritten under the reader). Each row
 * is wrapped and styled once for the screen's width, and only the last `KEPT_ROWS` are kept.
 */
export class RunProgress {
  readonly #log: string;
  readonly #columns: () => number;
  readonly #out: Colored;
  readonly #env: Env;
  /** Where the last read stopped: the end of its last record, its run id and seq. */
  #from: { afterOffset: number; runId: string; afterSeq: number } | null = null;
  /** The rows of the nodes that ended, and the agents' text and tool calls, in log order. */
  #rows: Row[] = [];
  /** Each row's lines, wrapped to `#width` and styled. */
  #lines: string[][] = [];
  #width = 0;
  /** How many lines the rows dropped from the head of `#rows` took. */
  #dropped = 0;
  /** The nodes started and not ended, in start order: when each started, and whether it waits. */
  #open = new Map<string, { at: number; waiting: boolean }>();
  /** Why the log couldn't be read, shown in place of the rows; null when it could. */
  #error: string | null = null;

  /** @param columns The screen's width, which the rows are wrapped to. */
  constructor(log: string, columns: () => number, out: Colored, env: Env) {
    this.#log = log;
    this.#columns = columns;
    this.#out = out;
    this.#env = env;
  }

  /**
   * The progress, as lines of the screen, styled: what ended, then each node still open with its
   * time so far at `now`; and how many lines before them are no longer kept (`earlier`).
   */
  rows(now: number): { lines: string[]; earlier: number } {
    const width = Math.max(1, this.#columns());
    if (width !== this.#width) {
      this.#width = width;
      this.#lines = this.#rows.map((row) => this.#wrap(row));
    }
    this.#read();
    if (this.#error !== null) {
      const why = this.#paint(`can't read the run's log: ${this.#error}`, "red");
      return { lines: [why], earlier: 0 };
    }
    const open = [...this.#open].map(([node, { at, waiting }]) => {
      const mark = waiting ? "◷" : "·";
      const text = `${mark} ${shown(node)} ${elapsedTag(now - at)}${waiting ? " waiting" : ""}`;
      return this.#paint(text, "dim");
    });
    return { lines: [...this.#lines.flat(), ...open], earlier: this.#dropped };
  }

  /** Read the records appended since the last read; all again, at once, if it was rewritten. */
  #read(again = true): void {
    try {
      // Every record moves the offset, the ones the progress skips too.
      for (const entry of iterRunRecords(this.#log, { ...this.#from })) {
        this.#from = { afterOffset: entry.end, runId: entry.runId, afterSeq: entry.seq };
        this.#take(entry.kind, entry.nodeId, recordTimeMs(entry.at), entry.envelope.record);
      }
      this.#error = null;
    } catch (error) {
      if (error instanceof RunLogChangedError) {
        // Rewritten under the reader (a failed append rolled back): read it all again.
        this.#from = null;
        this.#rows = [];
        this.#lines = [];
        this.#dropped = 0;
        this.#open.clear();
        if (again) this.#read(false);
        return;
      }
      // A log not written yet reads as nothing so far.
      if ((error as { code?: unknown }).code === "ENOENT") return;
      this.#error = error instanceof Error ? error.message : String(error);
    }
  }

  /** Add a row, keeping only the last `KEPT_ROWS`. */
  #push(row: Row): void {
    this.#rows.push(row);
    this.#lines.push(this.#wrap(row));
    if (this.#rows.length <= KEPT_ROWS) return;
    this.#rows.shift();
    this.#dropped += this.#lines.shift()!.length;
  }

  /** `row`'s lines at the screen's width, its later ones under its indent, styled. */
  #wrap(row: Row): string[] {
    const room = Math.max(1, this.#width - cells(row.lead));
    const ranges = row.text === "" ? [[0, 0] as const] : wrapLine(row.text, room);
    const parts = ranges.map(([start, end]) => row.text.slice(start, end));
    return parts.map((part, i) => {
      const line = `${i === 0 ? row.lead : row.indent}${part}`;
      return row.style === null ? line : this.#paint(line, row.style);
    });
  }

  /** Fold one record into the rows. */
  #take(kind: string, nodeId: string | null, at: number, record: unknown): void {
    const delivery = deliveryText(record);
    if (delivery !== null) {
      this.#push({ lead: "› ", indent: "  ", text: shown(delivery), style: "dim" });
      return;
    }
    if (kind === "node_failed") {
      const { failure } = record as NodeFailedRecord;
      const node = failure.nodeId ?? "";
      const text = failedNodeText(shown(node), this.#took(node, at));
      this.#push({ lead: "", indent: "  ", text, style: "red bold" });
      for (const line of shown(failure.message).split("\n")) {
        this.#push({ lead: "    ", indent: "    ", text: line, style: "red" });
      }
      return;
    }
    if (nodeId === null || boundary(nodeId)) return;
    if (kind === "node_started") this.#open.set(nodeId, { at, waiting: false });
    else if (kind === "node_waiting" || kind === "node_woken") {
      const open = this.#open.get(nodeId);
      if (open !== undefined) open.waiting = kind === "node_waiting";
    } else if (kind === "node_succeeded" || kind === "node_routed") {
      const handle = kind === "node_routed" ? shown((record as NodeRoutedRecord).handle) : null;
      const text = endedNodeText(shown(nodeId), this.#took(nodeId, at), handle);
      this.#push({ lead: "", indent: "  ", text, style: "green" });
    } else if (kind === "node_step") this.#step(record as NodeStepRecord);
  }

  /** An agent's step: a model call's text and tool calls, or a tool call's result. */
  #step({ step }: NodeStepRecord): void {
    if (step["kind"] === "tool_result") {
      const output = typeof step["output"] === "string" ? step["output"].replaceAll("\t", TAB) : "";
      const style = step["is_error"] === true ? "red" : "dim";
      for (const [i, row] of summary(output, this.#width - 5).entries()) {
        this.#push({ lead: i === 0 ? "  ⎿  " : "     ", indent: "     ", text: row, style });
      }
      return;
    }
    if (step["kind"] !== "model_call") return;
    const message = step["output"] as { parts?: unknown } | null;
    const parts = (message?.parts ?? []) as readonly Record<string, unknown>[];
    for (const part of parts) {
      const text = typeof part["text"] === "string" ? part["text"].trim() : "";
      if (part["type"] === "text" && text !== "") {
        for (const [i, line] of shown(text).split("\n").entries()) {
          this.#push({ lead: i === 0 ? "⏺ " : "  ", indent: "  ", text: line, style: null });
        }
      } else if (part["type"] === "tool_call" && typeof part["name"] === "string") {
        const text = `${shown(part["name"])}(${shown(shortArgs(part["arguments"] ?? null))})`;
        this.#push({ lead: "⏺ ", indent: "  ", text, style: null });
      }
    }
  }

  /** Close `nodeId` at `at`: ` [<time it took>]`, or "" when its start wasn't read. */
  #took(nodeId: string, at: number): string {
    const open = this.#open.get(nodeId);
    this.#open.delete(nodeId);
    return open === undefined ? "" : ` ${elapsedTag(at - open.at)}`;
  }

  #paint(text: string, style: Style): string {
    return paint(text, style, this.#out, this.#env);
  }
}

/**
 * A host call's record as a progress row, `<inbox>  <what>` then ` · <who>` when it says who made
 * it: a send's value (after the label of a message's author, in place of who sent it), a withdraw,
 * an interrupt and the agents it named, a close, or a cancel. An
 * interrupt the run applies for a Stop that stands is such a record too, so it draws a second
 * `interrupted` row, naming the agent it stopped. `null` for any other record. A step's `complete`
 * or `release` isn't one: its node's rows show it.
 */
function deliveryText(record: unknown): string | null {
  let what: string;
  let by: Author | null;
  if (record instanceof PoolCommitRecord && record.key.startsWith(`${START_ID}.`)) {
    const value = TypedValue.fromWire(record.value).toObject();
    const author = (value as { author?: unknown } | null)?.author;
    // A message names who it is from, as its reader sees it.
    const from = typeof author === "string" ? `${author}: ` : "";
    what = `${record.key.slice(START_ID.length + 1)}  ${from}${sentText(value)}`;
    by = from === "" ? record.by : null;
  } else if (record instanceof InboxWithdrawnRecord) {
    [what, by] = [`${record.inbox}  took back ${record.events.length}`, record.by];
  } else if (record instanceof InterruptedRecord) {
    [what, by] = [[`${record.inbox}  interrupted`, ...record.nodes].join(" "), record.by];
  } else if (record instanceof InboxClosedRecord) {
    [what, by] = [`${record.inbox}  closed`, record.by];
  } else if (record instanceof RunCancelledRecord) {
    [what, by] = ["cancelled", record.by];
  } else {
    return null;
  }
  return by === null ? what : `${what} · ${authorText(by)}`;
}

/**
 * A send's value on a row: a message's text, `[action]` for one that holds only actions, else the
 * value as a tool call's arguments show.
 */
function sentText(value: unknown): string {
  const parts = (value as { parts?: unknown } | null)?.parts;
  if (!Array.isArray(parts)) return shortArgs(value);
  const texts = (parts as { type?: unknown; text?: unknown }[]).filter((part) => part.type === "text");
  if (texts.length === 0 && parts.length > 0 && parts.every((part) => part?.type === "action")) return "[action]";
  return shortArgs(texts.map((part) => part.text).join(""));
}

/** What the run view's header names. */
interface RunViewInfo {
  readonly id: string;
  /** The flow it runs, as the run list names it. */
  readonly name: string;
  readonly status: ChildStatus;
}

/**
 * The run view's screen, `rows` high (each row is cut to the width when drawn): the header (the
 * run, its flow and its status, then its log), the `progress` lines, as many of the last as fit
 * under a dim count of those left out (and of the `earlier` ones no longer kept), the questions it
 * waits on, and `foot`.
 */
function runScreen(
  info: RunViewInfo,
  progress: { readonly lines: readonly string[]; readonly earlier: number },
  columns: number,
  rows: number,
  style: (text: string, s: Style) => string,
  foot: readonly string[],
): string[] {
  const { status } = info;
  const [label, labelStyle] = runStatusLabel(status);
  const name = fit(sanitize(info.name), Math.max(1, columns - info.id.length - label.length - 8));
  const head = [
    `${style(info.id, "bold")}  ${style(name, "bold")}  ${style(`● ${label}`, labelStyle)}`,
    `${style("log", "dim")}  ${sanitize(status.log)}`,
    "",
  ];
  const asks = (status.asks ?? []).map((ask) => style(`? ${sanitize(ask.question)}`, "yellow"));
  const tail = [...(asks.length > 0 ? ["", ...asks] : []), "", ...foot];
  const room = Math.max(1, rows - head.length - tail.length);
  const { lines, earlier } = progress;
  let body = lines.length > 0 ? [...lines] : [style("(nothing yet)", "dim")];
  if (body.length > room || earlier > 0) {
    const kept = body.slice(Math.max(0, body.length - (room - 1)));
    body = [style(`… ${earlier + lines.length - kept.length} earlier rows`, "dim"), ...kept];
  }
  return [...head, ...body, ...tail];
}

/** The tools of the chat's group a person presses from a run's view. */
type ViewTool = "message" | "interrupt" | "close_inbox" | "stop";

/** What a run's view shows and presses through: the chat's run and its owner, the assistant. */
export interface RunViewHost {
  /** The run's id among the chat's runs (`r1`). */
  readonly id: string;
  /** The flow it runs, as the run list names it. */
  readonly name: string;
  /** Its flow's inbox inputs, in the order declared (`LiveChild.inboxes`). */
  readonly inboxes: readonly string[];
  status(): ChildStatus;
  /** Its progress as `RunProgress.rows` gives it now. */
  progress(): { readonly lines: readonly string[]; readonly earlier: number };
  /** Press the assistant's tool `tool` on the run as the person (`ChildFlowOwner.press`). */
  press<K extends ViewTool>(
    tool: K,
    args: PressArguments[K],
  ): Promise<Pressed<K extends "interrupt" ? ChildInterrupt : ChildStatus>>;
}

/** The most rows what a press did takes under the strip. */
const SAID_ROWS = 2;

/** The rows the view keeps from its box, beside the box's own margin: the header, the strip and what a press did. */
const VIEW_ROWS = 6;

/** What each press is called while it is under way, and when it ran nothing. */
const PRESSING: Readonly<Record<ViewTool, { doing: string; not: string }>> = {
  message: { doing: "sending", not: "not sent" },
  interrupt: { doing: "interrupting", not: "not interrupted" },
  close_inbox: { doing: "closing", not: "not closed" },
  stop: { doing: "stopping", not: "not stopped" },
};

/** A line the view shows under the strip, and its style. */
type Said = { readonly text: string; readonly style: Style };

/** `count` of `noun`, with an `s` unless one. */
function counted(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** The words a send taken back held: a `str` as it is, a message's text parts joined. */
function sentWords(value: unknown): string {
  const parts = (value as { parts?: unknown } | null)?.parts;
  if (!Array.isArray(parts)) return typeof value === "string" ? value : "";
  return parts.map((part) => (part?.type === "text" && typeof part.text === "string" ? part.text : "")).join("");
}

/**
 * A child run of the chat on the whole screen (`Terminal.showScreen`): `runScreen`, then a strip of
 * the run's inboxes, the one picked in brackets (Tab and Shift-Tab pick), what the last press did,
 * and a box named after the inbox picked, which keeps its text and pastes as the chat's does.
 * Each key presses the assistant's tool as the person, never the chat's own: Enter sends the box's
 * text as it is (`message`), `/close` closes the inbox (`close_inbox`), `/stop` stops the run after
 * a `y` (`stop`), and Escape, while the status drawn is `running`, interrupts the inbox
 * (`interrupt`), the lines it takes back returning to the box. Escape at rest, or while a prompt of
 * the chat waits behind the view, closes it, pressing nothing.
 */
export class RunView {
  readonly #host: RunViewHost;
  readonly #style: (text: string, s: Style) => string;
  readonly #editor: LineEditor;
  /** The screen's size at the last draw, which the box keeps within: each draw sets it first. */
  #size = { columns: 0, rows: 0 };
  /** The strip's inbox the box sends to, by its place. */
  #picked = 0;
  /** The status the screen shows: Escape interrupts only a run it shows running. */
  #drawn: ChildStatus["status"] | null = null;
  /** Where the box's rows start among those drawn last. */
  #boxAt = 0;
  /** What the last press did, or what the view asks; `null` for nothing. */
  #said: Said | null = null;
  /** Whether a press is under way: until it is done, no other starts. */
  #busy = false;
  /** Whether `/stop` waits for its `y`. */
  #confirming = false;
  readonly #listeners = new Set<() => void>();

  constructor(host: RunViewHost, out: Colored, env: Env) {
    this.#host = host;
    this.#style = (text, s) => paint(text, s, out, env);
    const count = host.inboxes.length;
    const close = { name: "/close", description: "Close the inbox: the run reads no more from it" };
    const stop = { name: "/stop", description: "Stop the run: it ends" };
    this.#editor = new LineEditor({
      columns: () => this.#size.columns,
      rows: () => this.#size.rows - VIEW_ROWS,
      hint: count === 0 ? "/stop ends the run" : `enter sends${count > 1 ? " · tab picks" : ""} · /close · /stop`,
      commands: count === 0 ? [stop] : [close, stop],
      title: () => sanitize(this.#target() ?? ""),
    });
  }

  /** Call `listener` when what the view draws changes without a key: a press is done. */
  subscribe(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => void this.#listeners.delete(listener);
  }

  /** The screen, `rows` high and `columns` wide; `waiting` when a prompt of the chat waits behind it. */
  draw(columns: number, rows: number, waiting: boolean): string[] {
    this.#size = { columns, rows };
    const { id, name, inboxes } = this.#host;
    const status = this.#host.status();
    this.#drawn = status.status;
    const style = this.#style;
    const picked = (inbox: string, i: number) =>
      i === this.#picked ? style(`[${sanitize(inbox)}]`, "cyan bold") : sanitize(inbox);
    const strip = inboxes.length === 0 ? [] : [`${style("inboxes", "dim")}  ${inboxes.map(picked).join("  ")}`];
    const said = this.#said;
    const saidRows = said === null ? [] : this.#saidRows(said.text, columns).map((row) => style(row, said.style));
    const box = this.#editor.lines();
    const target = this.#target();
    const back = waiting
      ? style("the chat waits on you · esc to answer", "yellow")
      : this.#drawn === "running" && target !== null
        ? style(`esc interrupts ${sanitize(target)} · ctrl-c goes back to the chat`, "dim")
        : style("esc to go back to the chat", "dim");
    const foot = [...strip, ...saidRows, ...box, back];
    const screen = runScreen({ id, name, status }, this.#host.progress(), columns, rows, style, foot);
    this.#boxAt = screen.length - 1 - box.length;
    return screen;
  }

  /** Where the cursor shows among the rows drawn last: in the box. */
  cursor(): { row: number; before: string } | null {
    const at = this.#editor.cursor();
    return at === null ? null : { row: this.#boxAt + at.row, before: at.before };
  }

  /** Take one key, `waiting` when a prompt of the chat waits behind the view; return whether to close it. */
  key(key: Key, waiting: boolean): boolean {
    if (this.#confirming) {
      this.#confirming = false;
      this.#said = null;
      if (!key.paste && (key.str === "y" || key.str === "Y")) this.#press("stop", { run: this.#host.id });
      return false;
    }
    const count = this.#host.inboxes.length;
    // Tab completes a command while the menu is open.
    if (key.name === "tab" && !key.paste && !this.#editor.menuOpen()) {
      if (count > 0) this.#picked = (this.#picked + (key.shift === true ? count - 1 : 1)) % count;
      return false;
    }
    const action = this.#editor.key(key);
    if (action === null) return false;
    if ("escape" in action) return this.#escape(waiting);
    if ("value" in action && !this.#busy) this.#enter(action.value);
    return false;
  }

  /** The inbox the box sends to; `null` when the run has none. */
  #target(): string | null {
    return this.#host.inboxes[this.#picked] ?? null;
  }

  /** Escape out of the menu: interrupt the run the screen shows running, else close the view. */
  #escape(waiting: boolean): boolean {
    const inbox = this.#target();
    // A prompt of the chat behind the view comes first, and a run at rest has nothing to cut.
    if (waiting || this.#drawn !== "running" || inbox === null) return true;
    if (!this.#busy) this.#press("interrupt", { run: this.#host.id, inbox });
    return false;
  }

  /** Enter in the box with `text`: a command, or the text sent to the inbox picked as it is. */
  #enter(text: string): void {
    const command = text.trim();
    const inbox = this.#target();
    if (command === "") return;
    if (command === "/stop") {
      this.#editor.set("");
      this.#confirming = true;
      this.#said = { text: `stop ${this.#host.id}? its run ends (y/N)`, style: "yellow" };
    } else if (inbox === null) this.#said = { text: "this run has no inbox", style: "yellow" };
    else {
      this.#editor.set("");
      if (command === "/close") this.#press("close_inbox", { run: this.#host.id, inbox });
      else this.#press("message", { run: this.#host.id, inbox, text });
    }
  }

  /** Press `tool` with `args`, and say what it did once it is done. */
  #press<K extends ViewTool>(tool: K, args: PressArguments[K]): void {
    this.#busy = true;
    this.#said = { text: `${PRESSING[tool].doing}…`, style: "dim" };
    const failed = (error: unknown): Said => {
      const why = error instanceof Error ? error.message : String(error);
      return { text: `${PRESSING[tool].not}: ${why}`, style: "red" };
    };
    void this.#host
      .press(tool, args)
      .then(
        (pressed) => (this.#said = this.#outcome(tool, args, pressed)),
        (error: unknown) => (this.#said = failed(error)),
      )
      .finally(() => {
        this.#busy = false;
        for (const listener of this.#listeners) listener();
      });
  }

  /** What `pressed`, a press of `tool` with `args`, did: why nothing ran, the run's note, or what it did. */
  #outcome<K extends ViewTool>(tool: K, args: PressArguments[K], pressed: Pressed<ChildStatus | ChildInterrupt>): Said {
    const { result, recorded, reason } = pressed;
    if (result === null) return { text: `${PRESSING[tool].not}: ${reason}`, style: "yellow" };
    const inbox = sanitize((args as { inbox?: string }).inbox ?? "");
    let said: Said;
    if (tool === "interrupt") said = this.#interrupted(result as ChildInterrupt, inbox);
    else {
      const { note } = result as ChildStatus;
      const did =
        tool === "message" ? `sent to ${inbox}` : tool === "close_inbox" ? `closed ${inbox}` : `stopped ${this.#host.id}`;
      said = note === undefined ? { text: did, style: "dim" } : { text: note, style: "yellow" };
    }
    // The run has the press, but the assistant's conversation doesn't hold it.
    return recorded ? said : { text: `${said.text}; the assistant isn't told (${reason})`, style: "yellow" };
  }

  /** What `cut`, an interrupt of `inbox`, did; the lines it took back go back in the box, first. */
  #interrupted(cut: ChildInterrupt, inbox: string): Said {
    const back = cut.taken.map((event) => textDraft(sentWords(event.value)));
    if (back.length > 0) this.#editor.set(joinDrafts([...back, this.#editor.draft()]));
    const returned = back.length === 0 ? "" : `; ${counted(back.length, "message")} back in the box`;
    if (!cut.landed) return { text: `${cut.status.note ?? PRESSING.interrupt.not}${returned}`, style: "yellow" };
    if (cut.nodes.length > 0) return { text: `interrupted ${cut.nodes.join(", ")}${returned}`, style: "dim" };
    const others = counted(cut.queued, "queued message");
    const queued = cut.queued === 0 ? "" : `, and may use up ${others} from others unanswered`;
    const text = `nothing to stop now; the next turn on ${inbox} stops at once${queued}${returned}`;
    return { text, style: "yellow" };
  }

  /** `text` on at most `SAID_ROWS` rows `columns` wide, sanitized, the last cut to fit. */
  #saidRows(text: string, columns: number): string[] {
    const line = sanitize(text).replaceAll("\n", " ");
    const ranges = wrapLine(line, Math.max(1, columns));
    const rows = ranges.slice(0, SAID_ROWS).map(([start, end]) => line.slice(start, end));
    if (ranges.length > SAID_ROWS) rows[SAID_ROWS - 1] = fit(line.slice(ranges[SAID_ROWS - 1]![0]), columns);
    return rows;
  }
}
