/**
 * The earlier `yian chat` sessions of a folder, which `/resume`, `--resume` and `--continue` go
 * on from, and the model choice messages `/model` sends.
 *
 * A session is a top-level run of the bundled chat flow whose recorded flow folder is the folder
 * the chat works in (`chat.ts` loads the flow there). The run index finds them; each one's log
 * gives its name, first user line, model and message count, and its fold the transcript a new
 * chat starts from.
 *
 * Knows about: the run index, the run log's records and its fold, the bundled chat flow's names
 * and the waiting children `chat.ts` prints.
 */

import { realpathSync } from "node:fs";
import { join } from "node:path";

import type { Env } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import { withWarnings } from "../platform.ts";
import { foldRunLog } from "../run.ts";
import {
  iterRunRecords,
  NodeStepRecord,
  PoolCommitRecord,
  RunStartedRecord,
} from "../runtime/log.ts";
import { readRunIndex, type RunIndexRow, updateRunIndex } from "../runtime/run_index.ts";
import { toHost } from "../typesys/values.ts";
import type { WaitingChild } from "./chat_runs.ts";
import { CHAT_FLOW_FILE, CHAT_FLOW_ID } from "./run.ts";
import type { PickChoice } from "./terminal.ts";

/** How many earlier chats `chatSessions` lists at most, the newest. */
const SESSIONS_LISTED = 20;

/** A provider and model the assistant's replies use. */
export interface ModelChoice {
  readonly provider: string;
  readonly model: string;
}

/** What a choice's text starts with, before its JSON (`chat.yaml`'s turn reads it so). */
const MODEL_PREFIX = "/model ";

/** The message `/model` sends to the chat's inbox for `choice`: authored `yian`, its JSON. */
export function modelMessage(choice: ModelChoice): Record<string, unknown> {
  const json = JSON.stringify({ provider: choice.provider, model: choice.model });
  const text = `${MODEL_PREFIX}${json}`;
  return { role: "user", author: "yian", parts: [{ type: "text", text }] };
}

/** The text of a message's single text part, or null. */
function onlyText(message: unknown): string | null {
  const parts = (message as { parts?: unknown } | null)?.parts;
  if (!Array.isArray(parts) || parts.length !== 1) return null;
  const part = parts[0] as { type?: unknown; text?: unknown };
  return part.type === "text" && typeof part.text === "string" ? part.text : null;
}

/** The model a message sent to the chat chooses, or null for any other message. */
export function chosenModel(message: unknown): ModelChoice | null {
  if ((message as { author?: unknown } | null)?.author !== "yian") return null;
  const text = onlyText(message) ?? "";
  if (!text.startsWith(`${MODEL_PREFIX}{`)) return null;
  try {
    return recordedModel(JSON.parse(text.slice(MODEL_PREFIX.length)));
  } catch {
    // Not a choice: text the chat didn't write as one.
    return null;
  }
}

/** An earlier chat, as `/resume` lists it. */
export interface ChatSession {
  readonly runId: string;
  readonly log: string;
  /** When it started, as its log records it. */
  readonly startedAt: string;
  /** The first line the user typed. */
  readonly firstLine: string;
  /** The name the chat gave itself, or kept from the chat it went on from; null until named. */
  readonly title: string | null;
  /**
   * The messages it started with (`spokenCount`), and the lines the user typed in it and the
   * replies to them.
   */
  readonly messages: number;
  /** The model it chose, or else the one its last reply used; null when the log doesn't say. */
  readonly model: ModelChoice | null;
  /**
   * The model it chose, with `/model` or as it started from an earlier chat's choice; null when it
   * ran on the defaults, which a chat going on from it then uses too.
   */
  readonly chosen: ModelChoice | null;
}

/**
 * The messages of `history` (a transcript's turns) the user and the assistant said: those with a
 * text part, not a tool call's or its result's alone; 0 for anything but a list of turns.
 */
export function spokenCount(history: unknown): number {
  if (!Array.isArray(history)) return 0;
  const spoken = (message: unknown): boolean => {
    const { role, parts } = (message ?? {}) as { role?: unknown; parts?: unknown };
    const said = role === "user" || role === "assistant";
    return said && Array.isArray(parts) && parts.some((part) => part?.type === "text");
  };
  return history.flat().filter(spoken).length;
}

/** The provider and model a recorded `llm_config` names, or null. */
function recordedModel(config: unknown): ModelChoice | null {
  const { provider, model } = (config ?? {}) as { provider?: unknown; model?: unknown };
  return typeof provider === "string" && typeof model === "string" ? { provider, model } : null;
}

/** Where a record the cheap pass over a log found sits, so it can be read alone. */
type Spot = { readonly after: { afterOffset: number; afterSeq: number; runId: string } | null };

/** The record at `spot` in `log`, built and checked. */
function recordAt(log: string, spot: Spot): unknown {
  for (const entry of iterRunRecords(log, spot.after ?? {})) return entry.envelope.record;
  return null;
}

/** A reply's model call records its model on a step of this kind. */
const LLM_STEP = "llm_config";

/** A turn's call that answers it, and the node that gives the chat its name, in its log. */
const ANSWERED = /^chat#\d+\/answered$/;
const TITLED = /^chat#\d+\/titled\/named$/;

/**
 * Read the session in `log`: its first user line and models; null when it isn't the bundled chat's
 * or the user typed nothing in it. A first pass reads only where the records are; then only the
 * run's start, the sends and the reply's last steps are built.
 */
function readSession(row: RunIndexRow, log: string): ChatSession | null {
  let start: Spot | null = null;
  const sends: Spot[] = [];
  const replySteps: Spot[] = [];
  let titled: Spot | null = null;
  let replies = 0;
  let after: Spot["after"] = null;
  for (const entry of iterRunRecords(log, { payloads: false })) {
    const spot = { after };
    const commit = entry.kind === "pool_commit" ? (entry.nodeId ?? "") : null;
    if (entry.kind === "run_started") start = spot;
    else if (commit === "__start__") sends.push(spot);
    else if (commit !== null && ANSWERED.test(commit)) replies++;
    else if (commit !== null && titled === null && TITLED.test(commit)) titled = spot;
    else if (entry.kind === "node_step" && entry.nodeId?.endsWith("/answered/reply")) {
      replySteps.push(spot);
    }
    after = { afterOffset: entry.end, afterSeq: entry.seq, runId: entry.runId };
  }
  const started = start === null ? null : recordAt(log, start);
  if (!(started instanceof RunStartedRecord)) return null;
  const text = started.flow?.files[started.flow.root]?.text ?? "";
  const ours = new RegExp(`^id: ${CHAT_FLOW_ID}$`, "m").test(text);
  if (started.flow?.root !== CHAT_FLOW_FILE || !ours) return null;
  let chosen: ModelChoice | null = null;
  let firstLine: string | null = null;
  let title: string | null = null;
  let lines = 0;
  // The messages it started with, from an earlier chat.
  let seeded = 0;
  for (const spot of sends) {
    const record = recordAt(log, spot);
    if (!(record instanceof PoolCommitRecord)) continue;
    const value = record.value?.value as Record<string, unknown> | undefined;
    if (record.key === "__start__") {
      chosen = recordedModel(value?.["model"]) ?? chosen;
      title = typeof value?.["title"] === "string" ? value["title"] : title;
      const transcript = value?.["transcript"] as { history?: unknown } | null | undefined;
      seeded = spokenCount(transcript?.history) || seeded;
    } else if (record.key === "__start__.owner_messages") {
      const choice = chosenModel(value);
      if (choice !== null) chosen = choice;
      else if (value?.["author"] == null) {
        lines++;
        firstLine ??= onlyText(value);
      }
    }
  }
  const named = titled === null ? null : recordAt(log, titled);
  if (named instanceof PoolCommitRecord && typeof named.value?.value === "string") {
    title = named.value.value;
  }
  if (firstLine === null || row.startedAt === null) return null;
  // The reply's last model call: its config step is among its last few steps.
  let used: ModelChoice | null = null;
  for (let i = replySteps.length - 1; i >= 0 && used === null; i--) {
    const step = recordAt(log, replySteps[i]!);
    if (step instanceof NodeStepRecord && step.step["kind"] === LLM_STEP) {
      used = recordedModel(step.step["effective_llm_config"]);
    }
  }
  const { runId, startedAt } = row;
  const messages = seeded + lines + replies;
  return { runId, log, startedAt, firstLine, title, messages, model: chosen ?? used, chosen };
}

/** The real path of `path`, or `path` when it doesn't resolve. */
function real(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return path;
  }
}

/**
 * Bring the run index of `runsDir` up to date, waiting at most a second for another update, and
 * return its rows; a failed update leaves the rows as they were.
 */
function indexRows(runsDir: string): RunIndexRow[] {
  try {
    withWarnings(() => {}, () => updateRunIndex(runsDir, { timeoutMs: 1000 }));
  } catch {
    // The rows as the last update left them still list every chat that ended before it.
  }
  try {
    return withWarnings(() => {}, () => readRunIndex(runsDir));
  } catch {
    return [];
  }
}

/**
 * The `yian chat` sessions started in the folder `root` whose runs are in `runsDir`, newest first,
 * that the user typed in and that no chat holds now, at most `SESSIONS_LISTED` of them. A log
 * that can't be read is left out.
 */
export function chatSessions(runsDir: string, root: string): ChatSession[] {
  const flow = join(real(root), CHAT_FLOW_FILE);
  const rows = indexRows(runsDir).filter(
    (row) => row.parentRunId === null && row.flow === flow && row.status !== "running",
  );
  // Newest first, so the logs are read only until enough chats are found.
  const newer = (a: RunIndexRow, b: RunIndexRow) =>
    compareCodeUnits(b.startedAt ?? "", a.startedAt ?? "") || compareCodeUnits(b.runId, a.runId);
  rows.sort(newer);
  const sessions: ChatSession[] = [];
  for (const row of rows) {
    if (sessions.length === SESSIONS_LISTED) break;
    let session: ChatSession | null;
    try {
      session = readSession(row, join(runsDir, row.runId, "log.jsonl"));
    } catch {
      // A log that can't be read isn't a chat to go on from.
      continue;
    }
    if (session !== null) sessions.push(session);
  }
  return sessions;
}

/** What a new chat starts from: an earlier one's transcript and model, and its children left. */
export interface SessionSeed {
  /** Its summary, and its messages as turns. */
  readonly transcript: { summary: string; history: unknown[][] };
  readonly model: ModelChoice | null;
  /** Its name, which the new chat keeps; null when it had none. */
  readonly title: string | null;
  /** The earlier chat's child runs left waiting or stopped; they aren't restarted. */
  readonly waiting: WaitingChild[];
}

/** The key of a turn's state in the chat's run log: the loop's `chat`, run `k`. */
const TURN_END = /^chat#\d+\/__end__$/;

/**
 * The seed a new chat takes from `session`: its transcript as its last turn left it (the summary,
 * and every turn's messages), read by folding its log, and the model it chose, if it chose one;
 * and its children in `runsDir` left waiting or stopped.
 *
 * @throws What `foldRunLog` throws for a log it can't replay.
 */
export async function sessionSeed(
  session: ChatSession,
  runsDir: string,
  env: Env,
): Promise<SessionSeed> {
  const folded = await foldRunLog(session.log, { env });
  let state: { summary: string; history: unknown[][] } = { summary: "", history: [] };
  for (const { entry, value } of folded.records) {
    const record = entry.envelope.record;
    if (value !== null && record instanceof PoolCommitRecord && TURN_END.test(record.key)) {
      state = toHost(value.value) as typeof state;
    }
  }
  const transcript = { summary: state.summary, history: state.history };
  const waiting = indexRows(runsDir).flatMap((row): WaitingChild[] => {
    if (row.parentRunId !== session.runId) return [];
    const log = join(runsDir, row.runId, "log.jsonl");
    if (row.status === "stopped" || row.status === "crashed") {
      return [{ run: row.runId, kind: "halted", log }];
    }
    if (row.status !== "waiting") return [];
    const tools = row.waits.filter((wait) => !wait.settled).map((wait) => wait.tool);
    const kind = tools.includes("ask") ? "question" : tools.includes("take") ? "inbox" : "timer";
    return [{ run: row.runId, kind, log }];
  });
  return { transcript, model: session.chosen, title: session.title, waiting };
}

/** Seconds in a minute, an hour and a day, and the days before a time ago is a date. */
const MINUTE_S = 60;
const HOUR_S = 3600;
const DAY_S = 86_400;
const DAYS_AGO = 30;

/** A UTC time as the run index records one, `2026-10-31T12:00:00.000Z`. */
const UTC_TIME = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d{1,3})\d*)?Z$/;

/**
 * How long before `now` (milliseconds since the epoch) the UTC time `startedAt` was: `<n>s ago`,
 * `<n>m ago`, `<n>h ago` or `<n>d ago`, and from `DAYS_AGO` days its UTC date. A time after `now`
 * is `0s ago`, and text that isn't such a time is given back.
 */
export function timeAgo(startedAt: string, now: number): string {
  const parts = UTC_TIME.exec(startedAt);
  if (parts === null) return startedAt;
  const [year, month, day, hour, minute, second] = parts.slice(1, 7).map(Number) as number[];
  const millis = Number((parts[7] ?? "0").padEnd(3, "0"));
  const at = Date.UTC(year!, month! - 1, day, hour, minute, second, millis);
  const seconds = Math.max(0, Math.floor((now - at) / 1000));
  if (seconds < MINUTE_S) return `${seconds}s ago`;
  if (seconds < HOUR_S) return `${Math.floor(seconds / MINUTE_S)}m ago`;
  if (seconds < DAY_S) return `${Math.floor(seconds / HOUR_S)}h ago`;
  const days = Math.floor(seconds / DAY_S);
  return days < DAYS_AGO ? `${days}d ago` : startedAt.slice(0, 10);
}

/**
 * `session` as `/resume`'s picker offers it at `now`: its name (its first line until named), then
 * when it started, its message count and its model; a search matches its first line too.
 */
export function sessionChoice(session: ChatSession, now: number): PickChoice {
  const firstLine = session.firstLine.replace(/\s+/g, " ").trim();
  const count = `${session.messages} message${session.messages === 1 ? "" : "s"}`;
  const model = session.model === null ? [] : [`${session.model.provider}/${session.model.model}`];
  const detail = [timeAgo(session.startedAt, now), count, ...model].join(" · ");
  if (session.title === null) return { title: firstLine, detail };
  return { title: session.title, detail, also: firstLine };
}
