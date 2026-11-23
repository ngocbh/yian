/**
 * A child run of `yian chat` shown on the whole screen, as its run list opens it: a header naming
 * the run, its flow, its status and its log, the run's progress read from its log and drawn as
 * `yian run` draws it (a `✓` or `✗` line per node that ended, a dim line per node still running
 * or waiting, an agent's text and tool calls as `⏺` lines with a `⎿` result), a dim `›` line per
 * record a host call on an inbox or the run wrote, with who made it; then a strip of its inboxes,
 * the questions it waits on and the tool calls it holds for approval, and a box. What the person
 * does there presses the chat assistant's own tools as the person (`ChildFlowOwner.press`): the run
 * hears them relayed by the assistant, and the assistant's conversation records each as their
 * action.
 *
 * Knows about: the run log's records and its reader, a run's open calls, the chat's child statuses,
 * tool lines and presses, the colour helpers, the terminal's cell widths, keys, input box and
 * question card, and the sanitizing of text printed on the terminal.
 */

import type { ChildInterrupt, ChildStatus, PressArguments, Pressed, PressUnrecorded } from "../child_flows.ts";
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
import type { AskQuestion, PendingCall } from "../run.ts";
import type { ToolSource } from "../tools/gate.ts";
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
import { joinDrafts, type Key, LineEditor, QuestionCard, textDraft } from "./terminal.ts";

/** The most rows of a run's progress the view keeps: the last ones, the rest counted. */
const KEPT_ROWS = 2_000;

/** The characters of a run id a progress row shows of who made a host call. */
const RUN_ID_SHOWN = 8;

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
  if (by === null) return what;
  // A run id is long and the run is named elsewhere: its start is enough.
  const run = by.run === undefined ? {} : { run: by.run.slice(0, RUN_ID_SHOWN) };
  return `${what} · ${authorText({ ...by, ...run })}`;
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
 * The run view's screen, at most `rows` high (each row is cut to the width when drawn): the header
 * (the run, its flow and its status, then its log), the `progress` lines, as many of the last as
 * fit under a dim count of those left out (and of the `earlier` ones no longer kept), and `foot`.
 * The foot always fits: on a short screen the progress is cut from the top, then the header from
 * its bottom.
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
  // The rows between the header and the foot; the progress gets what a blank leaves.
  const room = Math.max(0, rows - head.length - foot.length);
  const progressRoom = room - 1;
  const { lines, earlier } = progress;
  let body = lines.length > 0 ? [...lines] : [style("(nothing yet)", "dim")];
  if (progressRoom <= 0) body = [];
  else if (body.length > progressRoom || earlier > 0) {
    const kept = body.slice(Math.max(0, body.length - (progressRoom - 1)));
    body = [style(`… ${earlier + lines.length - kept.length} earlier rows`, "dim"), ...kept];
  }
  let middle = [...body, ""];
  if (middle.length > room) middle = middle.slice(0, -1).slice(Math.max(0, middle.length - 1 - room));
  const screen = [...head.slice(0, Math.max(0, rows - foot.length - middle.length)), ...middle, ...foot];
  return screen.slice(Math.max(0, screen.length - rows));
}

/**
 * What the box sends to, as the strip names it: one of the run's inboxes, a question it waits on,
 * or a tool call it holds for approval; `key` tells each apart.
 */
type Target =
  | { readonly kind: "inbox"; readonly key: string; readonly inbox: string }
  | { readonly kind: "question"; readonly key: string; readonly call: string; readonly question: string }
  | { readonly kind: "held"; readonly key: string; readonly call: string; readonly tool: string };

/** The strip's rows, in order: each kind of target and what its row is called. */
const STRIP: readonly (readonly [Target["kind"], string])[] = [
  ["inbox", "inboxes"],
  ["question", "questions"],
  ["held", "held calls"],
];

/** The most cells a question takes in the strip: the box's edge names it at more. */
const STRIP_CELLS = 24;

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
  /** Its open host call `call` as its log lists it now (`pendingCalls`); `null` when none is open. */
  pendingCall(call: string): PendingCall | null;
  /** Press the assistant's tool `tool` on the run as the person (`ChildFlowOwner.press`). */
  press<K extends keyof PressArguments>(
    tool: K,
    args: PressArguments[K],
  ): Promise<Pressed<K extends "interrupt" ? ChildInterrupt : ChildStatus>>;
}

/** The most rows what a press did takes under the strip. */
const SAID_ROWS = 2;

/** The most rows a held call's card takes under the strip. */
const CARD_ROWS = 6;

/** The rows the view keeps from its box, beside the box's own margin: the header, the strip and what a press did. */
const VIEW_ROWS = 6;

/** What each press is called while it is under way, and when it ran nothing. */
const PRESSING: Readonly<Record<keyof PressArguments, { doing: string; not: string }>> = {
  message: { doing: "sending", not: "not sent" },
  answer: { doing: "answering", not: "not answered" },
  interrupt: { doing: "interrupting", not: "not interrupted" },
  close_inbox: { doing: "closing", not: "not closed" },
  stop: { doing: "stopping", not: "not stopped" },
  approve: { doing: "deciding", not: "not decided" },
};

/** A line the view shows under the strip, its style, and the most rows it takes (`SAID_ROWS` when absent). */
type Said = { readonly text: string; readonly style: Style; readonly rows?: number };

/** The statuses of a run that has ended. */
const ENDED: ReadonlySet<ChildStatus["status"]> = new Set(["succeeded", "failed", "cancelled"]);

/** Why the chat, the assistant's run, didn't take a press or its action, in words. */
function chatRefusal(reason: PressUnrecorded | null): string {
  if (reason === "ended") return "the chat has ended";
  if (reason === "inbox_closed") return "the chat's inbox is closed";
  if (reason === "not_started") return "the chat hasn't started";
  if (reason === "busy") return "the chat's log stayed busy";
  return `the chat refused it (${reason})`;
}

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

/** `text` on one row: sanitized, its line breaks and tabs as spaces. */
function oneLine(text: string): string {
  return sanitize(text).replace(/[\n\t]/g, " ");
}

/** Where a held call's tool comes from, after a space, unless it is a host tool: ` from MCP server files`. */
function sourceText(source: ToolSource | null): string {
  if (source === null || source.kind === "host") return "";
  if (source.kind === "mcp") return ` from MCP server ${source.server}`;
  if (source.kind === "module") return ` from module ${source.alias}`;
  return source.kind === "memory" ? ` from memory ${source.input}` : ` from flow ${source.flow}`;
}

/**
 * A child run of the chat on the whole screen (`Terminal.showScreen`): `runScreen`, then a strip of
 * the run's inboxes, the questions it waits on and the tool calls it holds for approval, the one
 * picked in brackets (Tab and Shift-Tab pick), what the last press did, and a box named after it,
 * which keeps its text and pastes as the chat's does. Each key presses the assistant's tool as the
 * person, never the chat's own. Enter on an inbox sends the box's text as it is (`message`); on a
 * question, answers it with the text as it is (`answer`), or for a form (an ask's `questions`)
 * opens the chat's question card in place of the box, which Escape closes; on a held call, shows
 * the call as the run's log holds it (what it acts on, else its arguments) and asks: `a` approves
 * it and `r` refuses it (`approve`), the box's text as the reason. `/close` closes the inbox
 * (`close_inbox`) and `/stop` stops the run (`stop`), each after a `y`, and Escape, while the status
 * drawn is `running`, interrupts the inbox (`interrupt`), the lines it takes back returning to the
 * box. Escape at rest closes the view, pressing nothing. Once the run has ended, the box gives way
 * to a line saying so.
 */
export class RunView {
  readonly #host: RunViewHost;
  readonly #style: (text: string, s: Style) => string;
  readonly #editor: LineEditor;
  /** The screen's size at the last draw, which the box keeps within: each draw sets it first. */
  #size = { columns: 0, rows: 0 };
  /** The target picked, which the box sends to while the strip holds it; `null` before any is. */
  #picked: Target | null = null;
  /** The picked target's place in the strip: once it leaves, the box goes to the one now there. */
  #place = 0;
  /** The status the screen shows: keys act on its targets, and Escape interrupts only a run it shows running. */
  #shown: ChildStatus | null = null;
  /** Each question's call as the run's log listed it when first drawn, for its form; `null` once settled. */
  readonly #calls = new Map<string, PendingCall | null>();
  /** The questions and held calls pressed here, by key: one leaving the strip needs no word then. */
  readonly #pressed = new Set<string>();
  /** The card of the form being answered, in place of the box; `null` when none is up. */
  #card: QuestionCard | null = null;
  /** Where the box's rows, or the card's, start among those drawn last. */
  #boxAt = 0;
  /** What the last press did, or what the view asks; `null` for nothing. */
  #said: Said | null = null;
  /** Whether a press is under way: until it is done, no other starts. */
  #busy = false;
  /** What the next key does while the view asks something; `null` when it asks nothing. */
  #confirming: ((key: Key) => void) | null = null;
  readonly #listeners = new Set<() => void>();

  constructor(host: RunViewHost, out: Colored, env: Env) {
    this.#host = host;
    this.#style = (text, s) => paint(text, s, out, env);
    const close = { name: "/close", description: "Close the inbox: the run reads no more from it" };
    const stop = { name: "/stop", description: "Stop the run: it ends" };
    this.#editor = new LineEditor({
      columns: () => this.#size.columns,
      rows: () => this.#size.rows - VIEW_ROWS,
      hint: () => this.#hint(),
      commands: host.inboxes.length === 0 ? [stop] : [close, stop],
      title: () => this.#title(),
    });
  }

  /** Call `listener` when what the view draws changes without a key: a press is done, a card moves on. */
  subscribe(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => void this.#listeners.delete(listener);
  }

  /** The screen, `rows` high and `columns` wide; `waiting` when a prompt of the chat waits behind it. */
  draw(columns: number, rows: number, waiting: boolean): string[] {
    this.#size = { columns, rows };
    const { id, name } = this.#host;
    const status = this.#host.status();
    this.#shown = status;
    const style = this.#style;
    const target = this.#pick(status);
    const said = this.#said;
    const saidRows = said === null ? [] : this.#saidRows(said, columns).map((row) => style(row, said.style));
    // The rows the cursor is placed in: the box, or the card in its place.
    let lower: string[] = [];
    let foot: string[];
    if (this.#ended()) {
      this.#card = null;
      foot = [...saidRows, style("the run has ended · esc to go back", "dim")];
    } else if (this.#card !== null) {
      lower = this.#card.lines();
      foot = [...lower, style("esc closes the form", "dim")];
    } else {
      lower = this.#editor.lines();
      const back =
        status.status === "running" && target?.kind === "inbox"
          ? style(`esc interrupts ${sanitize(target.inbox)} · ctrl-c goes back to the chat`, "dim")
          : style("esc to go back to the chat", "dim");
      foot = [...this.#strip(status, target), ...saidRows, ...lower, back];
    }
    // Ctrl-C goes back to a prompt of the chat behind the view: Escape interrupts a run that works.
    if (waiting) foot[foot.length - 1] = style("the chat waits on you · ctrl-c to answer", "yellow");
    const screen = runScreen({ id, name, status }, this.#host.progress(), columns, rows, style, foot);
    this.#boxAt = screen.length - 1 - lower.length;
    return screen;
  }

  /** Where the cursor shows among the rows drawn last: in the box or the card, while the run hasn't ended. */
  cursor(): { row: number; before: string } | null {
    const at = this.#ended() ? null : (this.#card ?? this.#editor).cursor();
    if (at === null || this.#boxAt + at.row < 0) return null;
    return { row: this.#boxAt + at.row, before: at.before };
  }

  /** Take one key; return whether to close the view. */
  key(key: Key): boolean {
    if (this.#confirming !== null) {
      const answer = this.#confirming;
      this.#confirming = null;
      this.#said = null;
      answer(key);
      return false;
    }
    // Nothing more reaches an ended run: Escape goes back.
    if (this.#ended()) return key.name === "escape" && !key.paste;
    if (this.#card !== null) {
      this.#card.key(key);
      return false;
    }
    // Tab completes a command while the menu is open.
    if (key.name === "tab" && !key.paste && !this.#editor.menuOpen()) {
      this.#tab(key.shift === true);
      return false;
    }
    const action = this.#editor.key(key);
    if (action === null) return false;
    if ("escape" in action) return this.#escape();
    if ("value" in action && !this.#busy) this.#enter(action.value);
    return false;
  }

  /** Whether the status drawn last is an end. */
  #ended(): boolean {
    return this.#shown !== null && ENDED.has(this.#shown.status);
  }

  /** The run's targets in the status drawn last (now, before any draw): its inboxes, its questions, its held calls. */
  #targets(status = this.#shown ?? this.#host.status()): Target[] {
    return [
      ...this.#host.inboxes.map((inbox) => ({ kind: "inbox", key: `inbox ${inbox}`, inbox }) as const),
      ...(status.asks ?? []).map(({ call, question }) => ({ kind: "question", key: `call ${call}`, call, question }) as const),
      ...(status.held ?? []).map(({ call, tool }) => ({ kind: "held", key: `call ${call}`, call, tool }) as const),
    ];
  }

  /** The target the box sends to, in the status drawn last; `null` when the strip is empty. */
  #target(): Target | null {
    return this.#pick(this.#shown ?? this.#host.status());
  }

  /**
   * The target the box sends to among `status`'s: the one picked while the strip holds it, else the
   * one now in its place (the last, when the strip got shorter). A run that works lists no question
   * or held call, so one picked stays picked for when it waits again. One that left as the run
   * waits, and wasn't pressed here, is said to be no longer open.
   */
  #pick(status: ChildStatus): Target | null {
    const targets = this.#targets(status);
    const picked = this.#picked;
    const at = picked === null ? -1 : targets.findIndex((target) => target.key === picked.key);
    if (at >= 0) {
      this.#place = at;
      return targets[at]!;
    }
    const now = targets[Math.min(this.#place, targets.length - 1)] ?? null;
    if (picked !== null && status.status === "running") return now;
    if (picked !== null && picked.kind !== "inbox" && !ENDED.has(status.status) && !this.#pressed.has(picked.key)) {
      const what = picked.kind === "question" ? `the question "${this.#question(picked)}"` : `${picked.tool}'s held call`;
      this.#said = { text: `${what} is no longer open`, style: "yellow" };
    }
    this.#picked = now;
    return now;
  }

  /** Pick the next target in the strip, or the one before (`back`), around its ends. */
  #tab(back: boolean): void {
    const targets = this.#targets();
    const count = targets.length;
    if (count === 0) return;
    const now = this.#target();
    const at = Math.max(0, targets.findIndex((target) => target.key === now?.key));
    this.#place = (at + (back ? count - 1 : 1)) % count;
    this.#picked = targets[this.#place]!;
  }

  /** `target`'s ask when it is a form (`questions`), read from its call once; `null` for a plain question. */
  #form(target: Target): AskQuestion | null {
    if (target.kind !== "question") return null;
    if (!this.#calls.has(target.call)) {
      try {
        this.#calls.set(target.call, this.#host.pendingCall(target.call));
      } catch {
        // Unread, it is asked as a plain question, and read again next time.
        return null;
      }
    }
    const ask = this.#calls.get(target.call)?.ask ?? null;
    return (ask?.questions ?? []).length > 0 ? ask : null;
  }

  /** What a question asks, in words: a form's title, else its first question; else the question asked. */
  #question(target: Target & { kind: "question" }): string {
    const form = this.#form(target);
    const first = (form?.questions?.[0] as { question?: unknown } | undefined)?.question;
    return form?.title || (typeof first === "string" ? first : target.question);
  }

  /** The box's hint: what Enter does to the target picked, then the commands. */
  #hint(): string {
    const target = this.#target();
    if (target === null) return "/stop ends the run";
    const tab = this.#targets().length > 1 ? " · tab picks" : "";
    if (target.kind === "inbox") return `enter sends${tab} · /close · /stop`;
    if (target.kind === "held") return `enter approves or refuses${tab} · /stop`;
    return `${this.#form(target) === null ? "enter answers" : "enter opens the form"}${tab} · /stop`;
  }

  /** What the box's edge names: the inbox picked, the question, or what the text is for a held call. */
  #title(): string {
    const target = this.#target();
    if (target === null) return "";
    if (target.kind === "inbox") return oneLine(target.inbox);
    if (target.kind === "held") return `reason for ${oneLine(target.tool)}, if any`;
    return oneLine(this.#question(target));
  }

  /** The strip: a row for each kind of target the run has, the one `picked` in brackets. */
  #strip(status: ChildStatus, picked: Target | null): string[] {
    const style = this.#style;
    const targets = this.#targets(status);
    return STRIP.flatMap(([kind, label]) => {
      const named = targets
        .filter((target) => target.kind === kind)
        .map((target) => {
          let text: string;
          if (target.kind === "inbox") text = oneLine(target.inbox);
          else if (target.kind === "held") text = oneLine(target.tool);
          else text = fit(oneLine(this.#question(target)), STRIP_CELLS);
          return target.key === picked?.key ? style(`[${text}]`, "cyan bold") : text;
        });
      return named.length === 0 ? [] : [`${style(label, "dim")}  ${named.join("  ")}`];
    });
  }

  /** Escape out of the menu: interrupt the inbox picked of a run the screen shows running, else close the view. */
  #escape(): boolean {
    const target = this.#target();
    // A run at rest has nothing to cut.
    if (this.#shown?.status !== "running" || target?.kind !== "inbox") return true;
    if (!this.#busy) this.#press("interrupt", { run: this.#host.id, inbox: target.inbox }, "");
    return false;
  }

  /**
   * Enter in the box with `text`: a command, asked first; else, for the target picked, the text sent
   * to an inbox or answering a question as it is, a form's card, or a held call's decision.
   */
  #enter(text: string): void {
    const command = text.trim();
    const run = this.#host.id;
    const target = this.#target();
    if (command === "/stop") {
      return this.#confirm(`stop ${run}? its run ends (y/N)`, () => this.#press("stop", { run }, `stopped ${run}`));
    }
    if (target === null) {
      if (command !== "") this.#said = { text: "this run has no inbox", style: "yellow" };
      return;
    }
    if (command === "/close") {
      if (target.kind !== "inbox") {
        const why = this.#host.inboxes.length === 0 ? "this run has no inbox" : "/close closes an inbox: pick one with tab";
        this.#said = { text: why, style: "yellow" };
        return;
      }
      const { inbox } = target;
      const ask = `close ${inbox}? nothing more can be sent to it (y/N)`;
      return this.#confirm(ask, () => this.#press("close_inbox", { run, inbox }, `closed ${inbox}`));
    }
    if (target.kind === "held") return this.#decide(target, text);
    if (target.kind === "question" && this.#form(target) !== null) return this.#open(target);
    if (command === "") return;
    this.#editor.set("");
    if (target.kind === "inbox") this.#press("message", { run, inbox: target.inbox, text }, `sent to ${target.inbox}`);
    else this.#answer(target, text);
  }

  /** Press `answer` on `target` with `value`. */
  #answer(target: Target & { kind: "question" }, value: unknown): void {
    this.#pressed.add(target.key);
    this.#press("answer", { run: this.#host.id, call: target.call, answer: value }, `answered: ${this.#question(target)}`);
  }

  /** Open `target`'s form on the chat's question card, in place of the box; its answers go as one `answer`. */
  #open(target: Target & { kind: "question" }): void {
    const ask = this.#form(target)!;
    const nodeId = this.#calls.get(target.call)?.node ?? null;
    const request = { ...ask, options: null, nodeId };
    const card = new QuestionCard(request, () => this.#size.columns, () => this.#notify());
    this.#card = card;
    this.#said = null;
    void card.answer.then((answered) => {
      if (this.#card === card) this.#card = null;
      if (answered !== null && !this.#ended()) this.#answer(target, answered.value);
    });
  }

  /**
   * Show `target`'s held call as the run's log holds it now: what it acts on (its tool's `target`),
   * else its arguments. Then `a` approves it and `r` refuses it, with `text` as the reason when it
   * isn't blank; any other key leaves it held, and the text back in the box.
   */
  #decide(target: Target & { kind: "held" }, text: string): void {
    let call: PendingCall | null;
    try {
      call = this.#host.pendingCall(target.call);
    } catch (error) {
      this.#said = { text: `can't read the call: ${error instanceof Error ? error.message : String(error)}`, style: "red" };
      return;
    }
    if (call === null || !call.held) {
      this.#said = { text: `${target.tool}'s held call is no longer open`, style: "yellow" };
      return;
    }
    const run = this.#host.id;
    const reason = text.trim() === "" ? null : text;
    const card = [`approve ${run}'s ${call.tool} call${sourceText(call.source)}? a approves · r refuses · any other key leaves it held`];
    if (reason !== null) card.push(`reason: ${reason}`);
    card.push(call.target === null ? `arguments ${JSON.stringify(call.arguments ?? {})}` : `acts on ${JSON.stringify(call.target)}`);
    this.#editor.set("");
    this.#said = { text: card.join("\n"), style: "yellow", rows: CARD_ROWS };
    const { tool } = call;
    this.#confirming = (key) => {
      const letter = key.paste ? "" : (key.str ?? "").toLowerCase();
      const decision = letter === "a" ? "approve" : letter === "r" ? "refuse" : null;
      if (decision === null) return this.#editor.set(text);
      this.#pressed.add(target.key);
      const args = { run, call: target.call, decision, ...(reason === null ? {} : { reason }) } as const;
      this.#press("approve", args, `${decision === "approve" ? "approved" : "refused"} ${tool}`);
    };
  }

  /** Ask `question` in place of what the last press did; `y` makes `press`, any other key nothing. */
  #confirm(question: string, press: () => void): void {
    this.#editor.set("");
    this.#confirming = (key) => {
      if (!key.paste && (key.str === "y" || key.str === "Y")) press();
    };
    this.#said = { text: question, style: "yellow" };
  }

  /** Tell the listeners the view changed. */
  #notify(): void {
    for (const listener of this.#listeners) listener();
  }

  /** Press `tool` with `args`, and say what it did once it is done: `did`, unless the run's note says more. */
  #press<K extends keyof PressArguments>(tool: K, args: PressArguments[K], did: string): void {
    this.#busy = true;
    this.#said = { text: `${PRESSING[tool].doing}…`, style: "dim" };
    const failed = (error: unknown): Said => {
      const why = error instanceof Error ? error.message : String(error);
      return { text: `${PRESSING[tool].not}: ${why}`, style: "red" };
    };
    void this.#host
      .press(tool, args)
      .then(
        (pressed) => (this.#said = this.#outcome(tool, args, pressed, did)),
        (error: unknown) => (this.#said = failed(error)),
      )
      .finally(() => {
        this.#busy = false;
        this.#notify();
      });
  }

  /** What `pressed`, a press of `tool` with `args`, did: why nothing ran, the run's note, or `did`. */
  #outcome<K extends keyof PressArguments>(
    tool: K,
    args: PressArguments[K],
    pressed: Pressed<ChildStatus | ChildInterrupt>,
    did: string,
  ): Said {
    const { result, recorded, reason } = pressed;
    if (result === null) return { text: `${PRESSING[tool].not}: ${chatRefusal(reason)}`, style: "yellow" };
    let said: Said;
    if (tool === "interrupt") said = this.#interrupted(result as ChildInterrupt, sanitize((args as { inbox: string }).inbox));
    else {
      const { note } = result as ChildStatus;
      said = note === undefined ? { text: did, style: "dim" } : { text: note, style: "yellow" };
    }
    // The run has the press, but the assistant's conversation doesn't hold it.
    return recorded ? said : { text: `${said.text}; the assistant isn't told: ${chatRefusal(reason)}`, style: "yellow" };
  }

  /** What `cut`, an interrupt of `inbox`, did; the lines it took back go back in the box, first. */
  #interrupted(cut: ChildInterrupt, inbox: string): Said {
    const back = cut.taken.map((event) => textDraft(sentWords(event.value)));
    if (back.length > 0) this.#editor.set(joinDrafts([...back, this.#editor.draft()]));
    const returned = back.length === 0 ? "" : `; ${counted(back.length, "message")} back in the box`;
    if (!cut.landed) return { text: `${cut.status.note ?? PRESSING.interrupt.not}${returned}`, style: "yellow" };
    // Nothing was recorded: no reader to cut, and no Stop stands.
    if (cut.seq === null) {
      const text = `nothing to interrupt: the run waits for a message on ${inbox}, or is already stopping${returned}`;
      return { text, style: "dim" };
    }
    if (cut.nodes.length > 0) return { text: `interrupted ${cut.nodes.join(", ")}${returned}`, style: "dim" };
    const others = counted(cut.queued, "queued message");
    const queued = cut.queued === 0 ? "" : `, and may use up ${others} from others unanswered`;
    const text = `nothing to stop now; the next turn on ${inbox} stops at once${queued}${returned}`;
    return { text, style: "yellow" };
  }

  /** `said`'s text on at most its rows `columns` wide, sanitized, each of its lines wrapped, the last row cut to fit. */
  #saidRows(said: Said, columns: number): string[] {
    const width = Math.max(1, columns);
    const most = said.rows ?? SAID_ROWS;
    const rows = sanitize(said.text)
      .split("\n")
      .flatMap((line) => wrapLine(line, width).map(([start, end]) => ({ line, start, end })));
    const kept = rows.slice(0, most).map(({ line, start, end }) => line.slice(start, end));
    if (rows.length > most) {
      const { line, start, end } = rows[most - 1]!;
      // Cut within its line, or at its end with more lines after.
      kept[most - 1] = end < line.length ? fit(line.slice(start), width) : fit(`${line.slice(start, end)} …`, width);
    }
    return kept;
  }
}
