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
import { iterRunRecords, PoolCommitRecord, RunStartedRecord } from "../runtime/log.ts";
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

/** A message authored `yian` with this text chooses the model (`chat.yaml`'s turn reads it). */
const MODEL_TEXT = /^\/model (\S+) (\S+)$/;

/** The message `/model` sends to the chat's inbox for `choice`. */
export function modelMessage(choice: ModelChoice): Record<string, unknown> {
  const text = `/model ${choice.provider} ${choice.model}`;
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
  const match = MODEL_TEXT.exec(onlyText(message) ?? "");
  return match === null ? null : { provider: match[1]!, model: match[2]! };
}

/** An earlier chat, as `/resume` lists it. */
export interface ChatSession {
  readonly runId: string;
  readonly log: string;
  /** When it started, as its log records it. */
  readonly startedAt: string;
  /** The first line the user typed. */
  readonly firstLine: string;
  /** The model its last reply would use, or null when the log doesn't say. */
  readonly model: ModelChoice | null;
}

/** The agents whose recorded model is the chat's: the reply's when literal, else the default. */
const MODEL_PATHS = ["chat/answered/reply", "chat/compacted/folded/summarize"];

/** The provider and model a recorded `llm_config` names, or null. */
function recordedModel(config: unknown): ModelChoice | null {
  const { provider, model } = (config ?? {}) as { provider?: unknown; model?: unknown };
  return typeof provider === "string" && typeof model === "string" ? { provider, model } : null;
}

/**
 * Read the session in `log`: its first user line and model; null when it isn't the bundled chat's
 * or the user typed nothing in it.
 */
function readSession(row: RunIndexRow, log: string): ChatSession | null {
  let model: ModelChoice | null = null;
  let firstLine: string | null = null;
  for (const entry of iterRunRecords(log, { kinds: ["run_started", "pool_commit"] })) {
    const record = entry.envelope.record;
    if (record instanceof RunStartedRecord) {
      const text = record.flow?.files[record.flow.root]?.text ?? "";
      const ours = new RegExp(`^id: ${CHAT_FLOW_ID}$`, "m").test(text);
      if (record.flow?.root !== CHAT_FLOW_FILE || !ours) return null;
      const effective = record.effectiveLlmByStaticPath;
      const models = MODEL_PATHS.map((path) => recordedModel(effective[path]));
      model = models.find((each) => each !== null) ?? null;
      continue;
    }
    if (!(record instanceof PoolCommitRecord) || record.nodeId !== "__start__") continue;
    const value = record.value?.value as Record<string, unknown> | undefined;
    if (record.key === "__start__") {
      model = recordedModel(value?.["model"]) ?? model;
    } else if (record.key === "__start__.owner_messages") {
      const chosen = chosenModel(value);
      if (chosen !== null) model = chosen;
      else if (firstLine === null && value?.["author"] == null) firstLine = onlyText(value);
    }
  }
  if (firstLine === null || row.startedAt === null) return null;
  return { runId: row.runId, log, startedAt: row.startedAt, firstLine, model };
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
  const sessions = indexRows(runsDir).flatMap((row) => {
    if (row.parentRunId !== null || row.flow !== flow || row.status === "running") return [];
    const log = join(runsDir, row.runId, "log.jsonl");
    try {
      const session = readSession(row, log);
      return session === null ? [] : [session];
    } catch {
      return [];
    }
  });
  sessions.sort(
    (a, b) => compareCodeUnits(b.startedAt, a.startedAt) || compareCodeUnits(b.runId, a.runId),
  );
  return sessions.slice(0, SESSIONS_LISTED);
}

/** What a new chat starts from: an earlier one's transcript and model, and its children left. */
export interface SessionSeed {
  readonly transcript: { summary: string; history: unknown[] };
  readonly model: ModelChoice | null;
  /** The earlier chat's child runs left waiting or stopped; they aren't restarted. */
  readonly waiting: WaitingChild[];
}

/** The key of a turn's state in the chat's run log: the loop's `chat`, run `k`. */
const TURN_END = /^chat#\d+\/__end__$/;

/**
 * The seed a new chat takes from `session`: its transcript as its last turn left it (the summary,
 * and every turn's messages), read by folding its log, and the model it chose; and its children
 * in `runsDir` left waiting or stopped.
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
  const transcript = { summary: state.summary, history: state.history.flat() };
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
  return { transcript, model: session.model, waiting };
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
