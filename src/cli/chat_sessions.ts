/**
 * The earlier `yian chat` sessions of a folder, which `/resume`, `--resume` and `--continue` go
 * on from, and the model choice messages `/model` sends.
 *
 * A session is a top-level run of the bundled chat flow whose recorded flow folder is the folder
 * the chat works in (`chat.ts` loads the flow there). The run index finds them; each one's log
 * gives its first user line and model, and its fold the transcript a new chat starts from.
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
import { fit } from "./cells.ts";
import type { WaitingChild } from "./chat_runs.ts";
import { CHAT_FLOW_FILE, CHAT_FLOW_ID } from "./run.ts";

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
  /** The model it chose, or else the one its last reply used; null when the log doesn't say. */
  readonly model: ModelChoice | null;
  /**
   * The model it chose, with `/model` or as it started from an earlier chat's choice; null when it
   * ran on the defaults, which a chat going on from it then uses too.
   */
  readonly chosen: ModelChoice | null;
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

/**
 * Read the session in `log`: its first user line and models; null when it isn't the bundled chat's
 * or the user typed nothing in it. A first pass reads only where the records are; then only the
 * run's start, the sends and the reply's last steps are built.
 */
function readSession(row: RunIndexRow, log: string): ChatSession | null {
  let start: Spot | null = null;
  const sends: Spot[] = [];
  const replySteps: Spot[] = [];
  let after: Spot["after"] = null;
  for (const entry of iterRunRecords(log, { payloads: false })) {
    const spot = { after };
    if (entry.kind === "run_started") start = spot;
    else if (entry.kind === "pool_commit" && entry.nodeId === "__start__") sends.push(spot);
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
  for (const spot of sends) {
    const record = recordAt(log, spot);
    if (!(record instanceof PoolCommitRecord)) continue;
    const value = record.value?.value as Record<string, unknown> | undefined;
    if (record.key === "__start__") {
      chosen = recordedModel(value?.["model"]) ?? chosen;
    } else if (record.key === "__start__.owner_messages") {
      const choice = chosenModel(value);
      if (choice !== null) chosen = choice;
      else if (firstLine === null && value?.["author"] == null) firstLine = onlyText(value);
    }
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
  return { runId, log, startedAt, firstLine, model: chosen ?? used, chosen };
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
  return { transcript, model: session.chosen, waiting };
}

/**
 * `session` as a picker row at most `width` cells wide: when it started (UTC), its model and its
 * first line, cut to fit.
 */
export function sessionTitle(session: ChatSession, width: number): string {
  const when = session.startedAt.slice(0, 16).replace("T", " ");
  const model = session.model === null ? "" : ` · ${session.model.provider}/${session.model.model}`;
  return fit(`${when}${model} · ${session.firstLine.replace(/\s+/g, " ")}`, width);
}
