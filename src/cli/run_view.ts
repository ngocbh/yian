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
import type { PendingCall } from "../run.ts";
import { ASK_USER } from "../tools/context.ts";
import type { ToolSource } from "../tools/gate.ts";
import type { AskUserRequest } from "../tools/registry.ts";
import { TypedValue } from "../typesys/values.ts";
import { cells, fit, wrapLine } from "./cells.ts";
import { runStatusLabel, shortArgs, summary } from "./chat_view.ts";
import { TAB } from "./markdown.ts";
import {
  authorText,
  box,
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
      // A TOOL node's answer (an ask's, say) is a typed value its node's row already ends.
      if (typeof step["output"] !== "string") return;
      const output = step["output"].replaceAll("\t", TAB);
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

/** A question the strip names. */
type Question = Target & { readonly kind: "question" };

/** A held call the strip names. */
type Held = Target & { readonly kind: "held" };

/**
 * A held call being decided: the call as the run's log held it, the box's text and the reason it
 * gives, and whether its card was last drawn whole.
 */
type Deciding = { readonly target: Held; readonly call: PendingCall; readonly text: string; readonly reason: string | null; whole: boolean };

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
  /** Its open host calls as its log lists them now (`pendingCalls`). */
  pendingCalls(): readonly PendingCall[];
  /** Press the assistant's tool `tool` on the run as the person (`ChildFlowOwner.press`). */
  press<K extends keyof PressArguments>(
    tool: K,
    args: PressArguments[K],
  ): Promise<Pressed<K extends "interrupt" ? ChildInterrupt : ChildStatus>>;
}

/** The most rows what a press did takes under the strip. */
const SAID_ROWS = 2;

/**
 * The rows the view keeps from its box, beside the box's own margin: the header, a strip row per
 * kind of target, what a press did, and the line under the box.
 */
const VIEW_ROWS = 9;

/** What each press is called while it is under way, and when it ran nothing. */
const PRESSING: Readonly<Record<keyof PressArguments, { doing: string; not: string }>> = {
  message: { doing: "sending", not: "not sent" },
  answer: { doing: "answering", not: "not answered" },
  interrupt: { doing: "interrupting", not: "not interrupted" },
  close_inbox: { doing: "closing", not: "not closed" },
  stop: { doing: "stopping", not: "not stopped" },
  approve: { doing: "deciding", not: "not decided" },
};

/** A line the view shows under the strip, and its style; its own line breaks start rows. */
type Said = { readonly text: string; readonly style: Style };

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

/** Where a held call's tool comes from, unless it is a host tool: `from MCP server files`. */
function sourceText(source: ToolSource | null): string | null {
  if (source === null || source.kind === "host") return null;
  if (source.kind === "mcp") return `from MCP server ${source.server}`;
  if (source.kind === "module") return `from module ${source.alias}`;
  return source.kind === "memory" ? `from memory ${source.input}` : `from flow ${source.flow}`;
}

/**
 * The card asking whether to approve `call`, a held call of run `run`, `columns` wide: a box titled
 * with the question, holding what the call acts on (its tool's gate `target`, else its arguments)
 * as JSON, where its tool comes from unless it is a host tool, and last the person's `reason`. Each
 * label starts its row, and the text under it is indented on every row it wraps to, so no text from
 * the call or the reason reads as a label. Its text shows the sequences it holds, as an approval's
 * card does. `whole` is false when the title had to be cut.
 */
function heldCard(
  run: string,
  call: PendingCall,
  reason: string | null,
  columns: number,
  colors: { out: Colored; env: Env },
): { rows: string[]; whole: boolean } {
  const { out, env } = colors;
  const inner = Math.max(3, columns - 4);
  const shown = (text: string) => sanitize(text, { sequences: "show" }).replaceAll("\t", TAB);
  const rows: [string, string][] = [];
  const label = (text: string) => void rows.push([text, paint(text, "bold", out, env)]);
  const value = (text: string) => {
    for (const line of shown(text).split("\n")) {
      for (const [start, end] of wrapLine(line, inner - 2)) rows.push([`  ${line.slice(start, end)}`, `  ${line.slice(start, end)}`]);
    }
  };
  if (call.target === null) label("arguments");
  else label("acts on");
  value(JSON.stringify(call.target ?? call.arguments ?? {}));
  const from = sourceText(call.source);
  if (from !== null) label(shown(from));
  if (reason !== null) {
    label("reason");
    value(reason);
  }
  const title = `approve ${run}'s ${shown(call.tool)} call?`;
  const card = box(rows, "yellow", out, env, title, inner, columns).split("\n");
  return { rows: card, whole: cells(title) <= inner - 1 };
}

/**
 * A child run of the chat on the whole screen (`Terminal.showScreen`): `runScreen`, then a strip of
 * the run's inboxes, the questions it waits on and the tool calls it holds for approval, the one
 * picked in brackets (Tab and Shift-Tab pick), what the last press did, and a box named after it,
 * which keeps its text and pastes as the chat's does. Each key presses the assistant's tool as the
 * person, never the chat's own. Enter on an inbox sends the box's text as it is (`message`); on a
 * question, answers it with the text as it is (`answer`), or for a form (an ask's `questions`) or
 * an agent's question with options opens the chat's question card in place of the box, which
 * Escape closes; on a held call, shows the call on a card in place of the box, as the run's log
 * holds it, and asks: `a` approves it, once the card is drawn whole, and `r` refuses it
 * (`approve`), the box's text as the reason. While the run works, a question or held call picked
 * stays picked, and Enter on it waits for the run to wait. `/close` closes the inbox
 * (`close_inbox`) and `/stop` stops the run (`stop`), each after a `y`, and Escape, while the status
 * drawn is `running` and an inbox is picked, interrupts it (`interrupt`), the lines it takes back
 * returning to the box. Escape otherwise closes the view, pressing nothing. Once the run has ended,
 * the box gives way to a line saying so.
 */
export class RunView {
  readonly #host: RunViewHost;
  readonly #colors: { out: Colored; env: Env };
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
  /** The run's open calls as its log listed them, by id; a question's never changes. */
  readonly #calls = new Map<string, PendingCall>();
  /** The status (`ChildStatus.seq`) at which the run's open calls were last read; `null` before any read. */
  #readAt: number | null = null;
  /** The questions and held calls pressed here, by key: one leaving the strip needs no word then. */
  readonly #pressed = new Set<string>();
  /** The card of the question being answered, in place of the box; `null` when none is up. */
  #card: QuestionCard | null = null;
  /** The held call being decided, on its card in place of the box; `null` when none is. */
  #deciding: Deciding | null = null;
  /** Where the box's rows, or the card's, start among those drawn last. */
  #boxAt = 0;
  /** What the last press did, or what the view asks; `null` for nothing. */
  #said: Said | null = null;
  /** Whether a press is under way: until it is done, no other starts. */
  #busy = false;
  /** The press a command waits to make on a `y`, while its question is what the view says; `null` when none. */
  #confirming: (() => void) | null = null;
  readonly #listeners = new Set<() => void>();

  constructor(host: RunViewHost, out: Colored, env: Env) {
    this.#host = host;
    this.#colors = { out, env };
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
    // Ctrl-C goes back to a prompt of the chat behind the view: Escape interrupts a run that works.
    const behind = style("the chat waits on you · ctrl-c to answer", "yellow");
    // The rows the cursor is placed in: the box, or the question card in its place.
    let lower: string[] = [];
    let foot: string[];
    if (this.#ended()) {
      this.#card = null;
      this.#deciding = null;
      foot = [...this.#saidRows(columns), waiting ? behind : style("the run has ended · esc to go back", "dim")];
    } else if (this.#deciding !== null) {
      foot = [...this.#decision(this.#deciding, columns, rows - (waiting ? 1 : 0)), ...(waiting ? [behind] : [])];
    } else if (this.#card !== null) {
      lower = this.#card.lines();
      foot = [...lower, waiting ? behind : style("esc closes the form", "dim")];
    } else {
      lower = this.#editor.lines();
      const back =
        status.status === "running" && target?.kind === "inbox"
          ? style(`esc interrupts ${sanitize(target.inbox)} · ctrl-c goes back to the chat`, "dim")
          : style("esc to go back to the chat", "dim");
      foot = [...this.#strip(status, target, columns), ...this.#saidRows(columns), ...lower, waiting ? behind : back];
    }
    const screen = runScreen({ id, name, status }, this.#host.progress(), columns, rows, style, foot);
    this.#boxAt = screen.length - 1 - lower.length;
    return screen;
  }

  /** Where the cursor shows among the rows drawn last: in the box or the question card, while the run hasn't ended. */
  cursor(): { row: number; before: string } | null {
    const at = this.#ended() || this.#deciding !== null ? null : (this.#card ?? this.#editor).cursor();
    if (at === null || this.#boxAt + at.row < 0) return null;
    return { row: this.#boxAt + at.row, before: at.before };
  }

  /** Take one key; return whether to close the view. */
  key(key: Key): boolean {
    if (this.#confirming !== null) {
      const press = this.#confirming;
      this.#confirming = null;
      this.#said = null;
      if (!key.paste && (key.str === "y" || key.str === "Y")) press();
      return false;
    }
    // Nothing more reaches an ended run: Escape goes back.
    if (this.#ended()) return key.name === "escape" && !key.paste;
    if (this.#deciding !== null) {
      this.#decided(key);
      return false;
    }
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

  /** The status drawn last, or now, before any draw. */
  #status(): ChildStatus {
    return this.#shown ?? this.#host.status();
  }

  /**
   * `status`'s targets: the run's inboxes, its questions and its held calls. A run that works lists
   * no question or held call, so the one picked stays: its log still holds it open.
   */
  #targets(status = this.#status()): Target[] {
    const targets: Target[] = [
      ...this.#host.inboxes.map((inbox) => ({ kind: "inbox", key: `inbox ${inbox}`, inbox }) as const),
      ...(status.asks ?? []).map(({ call, question }) => ({ kind: "question", key: `call ${call}`, call, question }) as const),
      ...(status.held ?? []).map(({ call, tool }) => ({ kind: "held", key: `call ${call}`, call, tool }) as const),
    ];
    const picked = this.#picked;
    const kept = status.status === "running" && picked !== null && picked.kind !== "inbox";
    if (kept && !targets.some((target) => target.key === picked.key)) targets.push(picked);
    return targets;
  }

  /** The target the box sends to, in the status drawn last; `null` when the strip is empty. */
  #target(): Target | null {
    return this.#pick(this.#status());
  }

  /**
   * The target the box sends to among `status`'s: the one picked while the strip holds it, else the
   * one now in its place (the last, when the strip got shorter). A question or held call that left
   * as the run waits takes its card with it, and is said to be no longer open unless pressed here.
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
    if (picked !== null && picked.kind !== "inbox" && !ENDED.has(status.status)) {
      this.#card = null;
      this.#deciding = null;
      if (!this.#pressed.has(picked.key)) {
        const what = picked.kind === "question" ? `the question "${oneLine(this.#question(picked))}"` : `${oneLine(picked.tool)}'s held call`;
        this.#say({ text: `${what} is no longer open`, style: "yellow" });
      }
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

  /**
   * Open call `id` as the run's log lists it, `null` when it doesn't: the log is read at most once a
   * status, and only for a call not read before.
   */
  #call(id: string): PendingCall | null {
    const seq = this.#status().seq;
    if (!this.#calls.has(id) && this.#readAt !== seq) {
      this.#readAt = seq;
      try {
        for (const call of this.#host.pendingCalls()) this.#calls.set(call.callId, call);
      } catch {
        // Unread, it is a question the box answers until a later status reads it.
      }
    }
    return this.#calls.get(id) ?? null;
  }

  /**
   * The question card `target` opens, as the chat's own asks it: a form (an ask's `questions`), or
   * an agent's question with options (its choices and Other); `null` for one the box answers.
   */
  #asked(target: Target): AskUserRequest | null {
    if (target.kind !== "question") return null;
    const call = this.#call(target.call);
    if (call === null) return null;
    const { ask, node: nodeId } = call;
    if (ask !== null) return (ask.questions ?? []).length > 0 ? { ...ask, options: null, nodeId } : null;
    const options = call.arguments?.["options"];
    if (call.tool !== ASK_USER || !Array.isArray(options) || options.length === 0) return null;
    if (!options.every((option) => typeof option === "string")) return null;
    return { question: target.question, options, questions: null, answerSchema: null, title: null, nodeId };
  }

  /** What a question asks, in words: a form's title, else its first question; else the question asked. */
  #question(target: Question): string {
    const asked = this.#asked(target);
    const first = (asked?.questions?.[0] as { question?: unknown } | undefined)?.question;
    return asked?.title || (typeof first === "string" ? first : target.question);
  }

  /** The box's hint: what Enter does to the target picked, then the commands. */
  #hint(): string {
    const target = this.#target();
    if (target === null) return "/stop ends the run";
    const tab = this.#targets().length > 1 ? " · tab picks" : "";
    if (target.kind === "inbox") return `enter sends${tab} · /close · /stop`;
    if (target.kind === "held") return `enter approves or refuses${tab} · /stop`;
    const asked = this.#asked(target);
    const enter = asked === null ? "enter answers" : asked.questions === null ? "enter opens the choices" : "enter opens the form";
    return `${enter}${tab} · /stop`;
  }

  /** What the box's edge names: the inbox picked, the question, or what the text is for a held call. */
  #title(): string {
    const target = this.#target();
    if (target === null) return "";
    if (target.kind === "inbox") return oneLine(target.inbox);
    if (target.kind === "held") return `reason for ${oneLine(target.tool)}, if any`;
    return oneLine(this.#question(target));
  }

  /**
   * The strip, `columns` wide: a row for each kind of target the run has, the one `picked` in
   * brackets. A row too wide leaves out the items before the picked one until it shows.
   */
  #strip(status: ChildStatus, picked: Target | null, columns: number): string[] {
    const style = this.#style;
    const targets = this.#targets(status);
    return STRIP.flatMap(([kind, label]) => {
      const items = targets
        .filter((target) => target.kind === kind)
        .map((target) => {
          let text: string;
          if (target.kind === "inbox") text = oneLine(target.inbox);
          else if (target.kind === "held") text = oneLine(target.tool);
          else text = fit(oneLine(this.#question(target)), STRIP_CELLS);
          return { text, picked: target.key === picked?.key };
        });
      if (items.length === 0) return [];
      const at = items.findIndex((item) => item.picked);
      const named = (from: number) => `${from > 0 ? "…  " : ""}${items.slice(from, at + 1).map(({ text }) => text).join("  ")}`;
      let from = 0;
      // The picked item's brackets take two cells.
      while (from < at && cells(`${label}  ${named(from)}`) + 2 > columns) from++;
      const shown = items.slice(from).map((item) => (item.picked ? style(`[${item.text}]`, "cyan bold") : item.text));
      return [`${style(label, "dim")}  ${from > 0 ? "…  " : ""}${shown.join("  ")}`];
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
   * to an inbox or answering a question as it is, a question card, or a held call's card.
   */
  #enter(text: string): void {
    const command = text.trim();
    const run = this.#host.id;
    const target = this.#target();
    if (command === "/stop") {
      return this.#confirm(`stop ${run}? its run ends (y/N)`, () => this.#press("stop", { run }, `stopped ${run}`));
    }
    if (target === null) {
      if (command !== "") this.#say({ text: "this run has no inbox", style: "yellow" });
      return;
    }
    if (command === "/close") {
      if (target.kind !== "inbox") {
        const why = this.#host.inboxes.length === 0 ? "this run has no inbox" : "/close closes an inbox: pick one with tab";
        return this.#say({ text: why, style: "yellow" });
      }
      const { inbox } = target;
      const ask = `close ${inbox}? nothing more can be sent to it (y/N)`;
      return this.#confirm(ask, () => this.#press("close_inbox", { run, inbox }, `closed ${inbox}`));
    }
    if (target.kind !== "inbox" && this.#shown?.status === "running") {
      const what = target.kind === "question" ? "question" : "held call";
      return this.#say({ text: `the run is working; its ${what} comes back when it waits`, style: "yellow" });
    }
    if (target.kind === "held") return this.#decide(target, text);
    if (target.kind === "question" && this.#asked(target) !== null) return this.#open(target);
    if (command === "") return;
    this.#editor.set("");
    if (target.kind === "inbox") this.#press("message", { run, inbox: target.inbox, text }, `sent to ${target.inbox}`);
    else this.#answer(target, text);
  }

  /** Press `answer` on `target` with `value`. */
  #answer(target: Question, value: unknown): void {
    this.#pressed.add(target.key);
    const did = `answered: ${oneLine(this.#question(target))}`;
    this.#press("answer", { run: this.#host.id, call: target.call, answer: value }, did);
  }

  /** Open `target`'s question card in place of the box; its answers go as one `answer`. */
  #open(target: Question): void {
    const card = new QuestionCard(this.#asked(target)!, () => this.#size.columns, () => this.#notify());
    this.#card = card;
    this.#say(null);
    void card.answer.then((answered) => {
      // Gone with its question, or with the run: nothing to answer.
      if (this.#card !== card) return;
      this.#card = null;
      if (answered !== null) this.#answer(target, answered.value);
    });
  }

  /** Put `target`'s held call, as the run's log holds it now, on its card in place of the box, `text` its reason. */
  #decide(target: Held, text: string): void {
    let call: PendingCall | null;
    try {
      call = this.#host.pendingCalls().find((each) => each.callId === target.call) ?? null;
    } catch (error) {
      return this.#say({ text: `can't read the call: ${error instanceof Error ? error.message : String(error)}`, style: "red" });
    }
    if (call === null || !call.held) return this.#say({ text: `${oneLine(target.tool)}'s held call is no longer open`, style: "yellow" });
    this.#editor.set("");
    this.#say(null);
    this.#deciding = { target, call, text, reason: text.trim() === "" ? null : text, whole: false };
  }

  /**
   * The held call's card for `deciding`, `room` rows at most with the line under it saying which
   * keys do what. A card that doesn't fit keeps its head and says how many rows it leaves out, and
   * then `a` approves nothing: `deciding.whole` says whether the card was drawn whole.
   */
  #decision(deciding: Deciding, columns: number, room: number): string[] {
    const style = this.#style;
    const { rows: card, whole } = heldCard(this.#host.id, deciding.call, deciding.reason, columns, this.#colors);
    const fits = card.length <= room - 1;
    deciding.whole = whole && fits;
    const kept = Math.max(0, room - 2);
    const shown = fits ? card : [...card.slice(0, kept), style(`… ${card.length - kept} more rows`, "dim")];
    const keys = deciding.whole
      ? "a approves · r refuses · any other key leaves it held"
      : "the call is too long to show here; widen the terminal · r refuses · any other key leaves it held";
    return [...shown, style(keys, "yellow")];
  }

  /**
   * A key on a held call's card: `a` approves the call, once the card was drawn whole, and `r`
   * refuses it; any other key leaves it held, and the text back in the box.
   */
  #decided(key: Key): void {
    const { target, call, text, reason, whole } = this.#deciding!;
    const letter = key.paste ? "" : (key.str ?? "").toLowerCase();
    // Never from a card cut or not drawn yet: the line under it says why.
    if (letter === "a" && !whole) return;
    this.#deciding = null;
    if (letter !== "a" && letter !== "r") return this.#editor.set(text);
    const decision = letter === "a" ? "approve" : "refuse";
    this.#pressed.add(target.key);
    const args = { run: this.#host.id, call: target.call, decision, ...(reason === null ? {} : { reason }) } as const;
    this.#press("approve", args, `${decision === "approve" ? "approved" : "refused"} ${oneLine(call.tool)}`, text);
  }

  /** Ask `question` in place of what the last press did; `y` makes `press`, any other key nothing. */
  #confirm(question: string, press: () => void): void {
    this.#editor.set("");
    this.#said = { text: question, style: "yellow" };
    this.#confirming = press;
  }

  /** Say `said` in place of what the view said; a command's question it replaces is no longer asked. */
  #say(said: Said | null): void {
    this.#said = said;
    this.#confirming = null;
  }

  /** Tell the listeners the view changed. */
  #notify(): void {
    for (const listener of this.#listeners) listener();
  }

  /**
   * Press `tool` with `args`, and say what it did once it is done: `did`, unless the run's note says
   * more. A press that ran nothing puts `back`, its words, back in an empty box.
   */
  #press<K extends keyof PressArguments>(tool: K, args: PressArguments[K], did: string, back = ""): void {
    this.#busy = true;
    this.#say({ text: `${PRESSING[tool].doing}…`, style: "dim" });
    const giveBack = () => {
      if (back !== "" && this.#editor.text() === "") this.#editor.set(back);
    };
    void this.#host
      .press(tool, args)
      .then(
        (pressed) => {
          this.#say(this.#outcome(tool, args, pressed, did));
          if (pressed.result === null) giveBack();
        },
        (error: unknown) => {
          this.#say({ text: `${PRESSING[tool].not}: ${error instanceof Error ? error.message : String(error)}`, style: "red" });
          giveBack();
        },
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

  /** What the view says, on at most `SAID_ROWS` rows `columns` wide, sanitized, each line wrapped, the last row cut to fit. */
  #saidRows(columns: number): string[] {
    const said = this.#said;
    if (said === null) return [];
    const width = Math.max(1, columns);
    const rows = sanitize(said.text)
      .split("\n")
      .flatMap((line) => wrapLine(line, width).map(([start, end]) => ({ line, start, end })));
    const kept = rows.slice(0, SAID_ROWS).map(({ line, start, end }) => line.slice(start, end));
    if (rows.length > SAID_ROWS) {
      const { line, start, end } = rows[SAID_ROWS - 1]!;
      // Cut within its line, or at its end with more lines after.
      kept[SAID_ROWS - 1] = end < line.length ? fit(line.slice(start), width) : fit(`${line.slice(start, end)} …`, width);
    }
    return kept.map((row) => this.#style(row, said.style));
  }
}
