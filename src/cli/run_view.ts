/**
 * A child run of `yian chat` shown on the whole screen, as its run list opens it: a header naming
 * the run, its flow, its status and its log, the run's progress read from its log and drawn as
 * `yian run` draws it (a `✓` or `✗` line per node that ended, a dim line per node still running
 * or waiting, an agent's text and tool calls as `⏺` lines with a `⎿` result), and the questions it
 * waits on. Read-only: answers still go through the chat's assistant.
 *
 * Knows about: the run log's records and its reader, the chat's child statuses and tool lines,
 * the colour helpers, the terminal's cell widths and the sanitizing of text printed on the
 * terminal.
 */

import { END_ID, START_ID } from "../compile/model.ts";
import type { Env } from "../defaults.ts";
import {
  iterRunRecords,
  type NodeFailedRecord,
  type NodeRoutedRecord,
  type NodeStepRecord,
  recordTimeMs,
  RunLogChangedError,
} from "../runtime/log.ts";
import { cells, fit, wrapLine } from "./cells.ts";
import type { ChildStatus } from "./chat_runs.ts";
import { runStatusLabel, shortArgs, summary } from "./chat_view.ts";
import { TAB } from "./markdown.ts";
import {
  type Colored,
  elapsedTag,
  endedNodeText,
  failedNodeText,
  paint,
  type Style,
} from "./render.ts";
import { sanitize } from "./sanitize.ts";

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

/** What the run view's header names. */
export interface RunViewInfo {
  readonly id: string;
  /** The flow it runs, as the run list names it. */
  readonly name: string;
  readonly status: ChildStatus;
}

/**
 * The run view's screen, `rows` high (each row is cut to the width when drawn): the header (the
 * run, its flow and its status, then its log), the `progress` lines, as many of the last as fit
 * under a dim count of those left out (and of the `earlier` ones no longer kept), the questions it
 * waits on, and how to go back: in yellow, saying so, when the chat `waiting` waits on the user
 * (an approval or a question behind the view).
 */
export function runScreen(
  info: RunViewInfo,
  progress: { readonly lines: readonly string[]; readonly earlier: number },
  columns: number,
  rows: number,
  out: Colored,
  env: Env,
  waiting = false,
): string[] {
  const style = (text: string, s: Style) => paint(text, s, out, env);
  const { status } = info;
  const [label, labelStyle] = runStatusLabel(status);
  const name = fit(sanitize(info.name), Math.max(1, columns - info.id.length - label.length - 8));
  const head = [
    `${style(info.id, "bold")}  ${style(name, "bold")}  ${style(`● ${label}`, labelStyle)}`,
    `${style("log", "dim")}  ${sanitize(status.log)}`,
    "",
  ];
  const asks = (status.asks ?? []).map((ask) => style(`? ${sanitize(ask.question)}`, "yellow"));
  const back = waiting
    ? style("the chat waits on you · esc to answer", "yellow")
    : style("esc to go back to the chat", "dim");
  const foot = [...(asks.length > 0 ? ["", ...asks] : []), "", back];
  const room = Math.max(1, rows - head.length - foot.length);
  const { lines, earlier } = progress;
  let body = lines.length > 0 ? [...lines] : [style("(nothing yet)", "dim")];
  if (body.length > room || earlier > 0) {
    const kept = body.slice(Math.max(0, body.length - (room - 1)));
    body = [style(`… ${earlier + lines.length - kept.length} earlier rows`, "dim"), ...kept];
  }
  return [...head, ...body, ...foot];
}
