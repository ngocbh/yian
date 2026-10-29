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
import { fit } from "./cells.ts";
import type { ChildStatus } from "./chat_runs.ts";
import { runStatusLabel, shortArgs, summary } from "./chat_view.ts";
import { type Colored, paint, type Style } from "./render.ts";
import { elapsedTag } from "./run.ts";
import { sanitize } from "./sanitize.ts";

/** The record kinds the progress reads. */
const KINDS = [
  "node_started",
  "node_woken",
  "node_waiting",
  "node_succeeded",
  "node_routed",
  "node_failed",
  "node_step",
];

/** One row of the progress, before it is styled. */
type Row = { text: string; style: Style | null };

/** Whether `nodeId` is a boundary node the engine adds, which has no authored meaning. */
function boundary(nodeId: string): boolean {
  const last = nodeId.slice(nodeId.lastIndexOf("/") + 1);
  return last === START_ID || last === END_ID;
}

/**
 * A child run's progress, read from its log as it grows: each `rows` call reads the records
 * appended since the last one (the whole log again if it was rewritten under the reader).
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
  /** The nodes started and not ended, in start order: when each started, and whether it waits. */
  #open = new Map<string, { at: number; waiting: boolean }>();
  /** Why the log couldn't be read, shown in place of the rows; null when it could. */
  #error: string | null = null;

  /** @param columns The screen's width, which a tool result's summary is cut to as it is read. */
  constructor(log: string, columns: () => number, out: Colored, env: Env) {
    this.#log = log;
    this.#columns = columns;
    this.#out = out;
    this.#env = env;
  }

  /** The progress rows, styled: what ended, then each node still open with its time so far at `now`. */
  rows(now: number): string[] {
    this.#read();
    if (this.#error !== null) return [this.#paint(`can't read the run's log: ${this.#error}`, "red")];
    const open = [...this.#open].map(([node, { at, waiting }]) => {
      const mark = waiting ? "◷" : "·";
      return this.#paint(`${mark} ${sanitize(node)} ${elapsedTag(now - at)}${waiting ? " waiting" : ""}`, "dim");
    });
    const done = this.#rows.map((row) => (row.style === null ? row.text : this.#paint(row.text, row.style)));
    return [...done, ...open];
  }

  /** Read the records appended since the last read. */
  #read(): void {
    try {
      for (const entry of iterRunRecords(this.#log, { ...this.#from, kinds: KINDS })) {
        this.#from = { afterOffset: entry.end, runId: entry.runId, afterSeq: entry.seq };
        this.#take(entry.kind, entry.nodeId, recordTimeMs(entry.at), entry.envelope.record);
      }
      this.#error = null;
    } catch (error) {
      if (error instanceof RunLogChangedError) {
        // Rewritten under the reader (a failed append rolled back): read it all again next time.
        this.#from = null;
        this.#rows = [];
        this.#open.clear();
        return;
      }
      // A log not written yet reads as nothing so far.
      if ((error as { code?: unknown }).code === "ENOENT") return;
      this.#error = error instanceof Error ? error.message : String(error);
    }
  }

  /** Fold one record into the rows. */
  #take(kind: string, nodeId: string | null, at: number, record: unknown): void {
    if (kind === "node_failed") {
      const { failure } = record as NodeFailedRecord;
      const node = failure.nodeId ?? "";
      this.#rows.push({ text: `✗ ${sanitize(node)}${this.#took(node, at)}`, style: "red bold" });
      for (const line of sanitize(failure.message).split("\n")) {
        this.#rows.push({ text: `    ${line}`, style: "red" });
      }
      return;
    }
    if (nodeId === null || boundary(nodeId)) return;
    if (kind === "node_started") this.#open.set(nodeId, { at, waiting: false });
    else if (kind === "node_waiting" || kind === "node_woken") {
      const open = this.#open.get(nodeId);
      if (open !== undefined) open.waiting = kind === "node_waiting";
    } else if (kind === "node_succeeded") {
      this.#rows.push({ text: `✓ ${sanitize(nodeId)}${this.#took(nodeId, at)}`, style: "green" });
    } else if (kind === "node_routed") {
      const handle = sanitize((record as NodeRoutedRecord).handle);
      this.#rows.push({ text: `✓ ${sanitize(nodeId)} → ${handle}${this.#took(nodeId, at)}`, style: "green" });
    } else if (kind === "node_step") this.#step(record as NodeStepRecord);
  }

  /** An agent's step: a model call's text and tool calls, or a tool call's result. */
  #step({ step }: NodeStepRecord): void {
    if (step["kind"] === "tool_result") {
      const output = typeof step["output"] === "string" ? step["output"] : "";
      const style = step["is_error"] === true ? "red" : "dim";
      for (const [i, row] of summary(output, this.#columns() - 5).entries()) {
        this.#rows.push({ text: `${i === 0 ? "  ⎿  " : "     "}${row}`, style });
      }
      return;
    }
    if (step["kind"] !== "model_call") return;
    const message = step["output"] as { parts?: unknown } | null;
    const parts = (message?.parts ?? []) as readonly Record<string, unknown>[];
    for (const part of parts) {
      if (part["type"] === "text" && typeof part["text"] === "string" && part["text"].trim() !== "") {
        const lines = sanitize(part["text"].trim()).split("\n");
        for (const [i, line] of lines.entries()) {
          this.#rows.push({ text: `${i === 0 ? "⏺ " : "  "}${line}`, style: null });
        }
      } else if (part["type"] === "tool_call" && typeof part["name"] === "string") {
        const args = sanitize(shortArgs(part["arguments"] ?? null));
        this.#rows.push({ text: `⏺ ${sanitize(part["name"])}(${args})`, style: null });
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
 * run, its flow and its status, then its log), the `progress` rows, as many of the last as fit
 * under a dim count of those left out, the questions it waits on, and how to go back.
 */
export function runScreen(
  info: RunViewInfo,
  progress: readonly string[],
  columns: number,
  rows: number,
  out: Colored,
  env: Env,
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
  const foot = [...(asks.length > 0 ? ["", ...asks] : []), "", style("esc to go back to the chat", "dim")];
  const room = Math.max(1, rows - head.length - foot.length);
  let body = progress.length > 0 ? [...progress] : [style("(nothing yet)", "dim")];
  if (body.length > room) {
    const kept = body.slice(body.length - (room - 1));
    body = [style(`… ${body.length - kept.length} earlier rows`, "dim"), ...kept];
  }
  return [...head, ...body, ...foot];
}
