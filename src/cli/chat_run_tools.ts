/**
 * The host tools `yian chat`'s assistant runs flows with (`chatRunTools`): start a flow in the
 * chat's folder as a child run once the user approves it on the terminal, and read, wait on,
 * answer, message and stop the runs it started. As with the file tools (`chat_tools.ts`), the
 * model always gets text back: a tool returns the run's status as JSON, or throws a
 * `ReportedToolError` it reads.
 *
 * Knows about: the host tool API and its errors, confined loading, the run boundary's flow readers
 * and input intake, CODE references, the chat's file tools' helpers, its child runs and its
 * events.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { relative, resolve } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { coerceInputs } from "../nodes/binding.ts";
import { splitReference } from "../nodes/code/worker.ts";
import { isSystemError, realScheduler, type Scheduler } from "../platform.ts";
import { hostInputs, skillGrants, staticNodes } from "../run.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { buildValueAs, isInboxType } from "../typesys/values.ts";
import { LIST_ITEMS } from "./chat_events.ts";
import type { ChildRuns, ChildStatus } from "./chat_runs.ts";
import {
  type ChatToolHost,
  confine,
  readFlow,
  refuseContent,
  renderedDiagnostic,
  resolveRoot,
  slashed,
  stringArg,
  stringArguments,
} from "./chat_tools.ts";
import { loadConfinedText } from "./confined_load.ts";
import { unifiedDiff } from "./diff.ts";
import { loadReportFlow } from "./report_issue.ts";

/** The longest `wait_flow` waits, in seconds. */
const WAIT_FLOW_MAX_S = 120;

/** The most characters of each text field a run tool's status shows the model. */
const STATUS_TEXT_CHARS = 4000;

/** The approval answers `start_flow` offers: the first two start, the last (Ctrl-C's) doesn't. */
const START_CHOICES = ["Approve", "Approve this flow for this chat", "Reject"] as const;

/** What `report_issue` starts its run with, beside what the model passes. */
export interface ReportOptions {
  /** The chat's own facts (`hostFacts`), first in the report's context. */
  facts: string;
  /** The home folder, as set and as its real path, which the report shows as `~`. */
  homes: readonly string[];
  /** The `file_issue` tool the report flow files with; no other run gets it. */
  fileIssue: FunctionTool;
}

/** What the run tools need beside the file tools' host. */
export interface ChatRunToolsOptions {
  /**
   * The chat's record of what the model has seen (its `EventQueue`): told each status a run tool
   * hands the model, by run and seq, so the child events up to it aren't relayed again; and asked
   * the last seq of a run the model has seen, a relayed event included. `null` for none.
   */
  seen?: { markSeen(run: string, seq: number): void; seen(run: string): number } | null;
  /** The clock `wait_flow` times out on; the real one by default. */
  scheduler?: Scheduler;
  /** What `report_issue` needs; without it, there is no `report_issue`. */
  report?: ReportOptions | null;
}

/** `text` cut to `STATUS_TEXT_CHARS` characters, with a note when it was cut. */
function cutText(text: string): string {
  if (text.length <= STATUS_TEXT_CHARS) return text;
  return `${text.slice(0, STATUS_TEXT_CHARS)}... (cut at ${STATUS_TEXT_CHARS} characters)`;
}

/**
 * @internal `status` as the model reads it: its text fields cut, its asks and waits listed up to
 * `LIST_ITEMS` each (`not_shown` says how many more), without the seq it doesn't need.
 */
export function shownStatus(status: ChildStatus): Record<string, unknown> {
  const { seq: _, output, failure, asks, replies, note, waits, ...rest } = status;
  const shown: Record<string, unknown> = { ...rest };
  const more: string[] = [];
  if (waits !== undefined) {
    shown["waits"] = waits.slice(0, LIST_ITEMS);
    if (waits.length > LIST_ITEMS) more.push(`${waits.length - LIST_ITEMS} more waits`);
  }
  if (output !== undefined) {
    // An output too long as JSON is shown as its cut JSON text.
    const json = JSON.stringify(output) ?? "null";
    if (typeof output === "string") shown["output"] = cutText(output);
    else shown["output"] = json.length > STATUS_TEXT_CHARS ? cutText(json) : output;
  }
  if (failure !== undefined) shown["failure"] = { ...failure, message: cutText(failure.message) };
  if (asks !== undefined) {
    shown["asks"] = asks.slice(0, LIST_ITEMS).map((ask) => ({ ...ask, question: cutText(ask.question) }));
    if (asks.length > LIST_ITEMS) more.unshift(`${asks.length - LIST_ITEMS} more asks`);
  }
  if (replies !== undefined) shown["replies"] = replies.map(cutText);
  if (note !== undefined) shown["note"] = cutText(note);
  if (more.length > 0) shown["not_shown"] = `${more.join(" and ")}; the run's log has all`;
  return shown;
}

/**
 * The resolved path of each CODE reference file of `flow` and the flows it uses, sorted: each
 * resolved in its flow's folder, as the CODE worker resolves it. The files those import aren't
 * among them.
 */
function codeFiles(flow: ComposedFlow): string[] {
  const files = new Set<string>();
  for (const [, node] of staticNodes(flow)) {
    // Read kind-blind, as the run boundary reads a grant: only a CODE node has these fields.
    const code = node as { mode?: unknown; code?: unknown; baseDir?: unknown };
    if (node.kind !== "code" || code.mode !== "reference" || typeof code.code !== "string") continue;
    const base = typeof code.baseDir === "string" ? code.baseDir : ".";
    files.add(resolve(base, splitReference(code.code).path));
  }
  return Array.from(files).sort(compareCodeUnits);
}

/**
 * The sha256 of everything that runs when `flow` starts: its flow files (`text`, the root's, and
 * each file it was loaded from), the bytes of each of its CODE reference `files` (or why one
 * can't be read), and `can`, what the approval card says it can do.
 */
function startDigest(flow: ComposedFlow, text: string, files: readonly string[], can: readonly string[]): string {
  const hash = createHash("sha256").update(text).update("\0");
  const sources = flow.source?.files ?? {};
  for (const key of Object.keys(sources).sort(compareCodeUnits)) {
    hash.update(key).update("\0").update(sources[key]!.text).update("\0");
  }
  for (const file of files) {
    hash.update(file).update("\0");
    try {
      hash.update(readFileSync(file));
    } catch (error) {
      if (!isSystemError(error)) throw error;
      hash.update(`unreadable: ${error.code}`);
    }
    hash.update("\0");
  }
  return hash.update(can.join("\n")).digest("hex");
}

/**
 * Check `inputs` against `flow`'s inputs before it starts, as its start would: no unknown names,
 * every required one given, each value of its declared type.
 *
 * @throws {ReportedToolError} `bad inputs: <why>`.
 */
function checkStartInputs(flow: ComposedFlow, inputs: unknown): void {
  try {
    const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
    for (const param of flow.input) {
      if (isInboxType(param.contract?.type ?? null)) continue;
      if (!Object.hasOwn(coerced, param.name)) {
        if (!param.required || param.hasDefault) continue;
        throw new RangeError(`required input ${reprText(param.name)} is missing`);
      }
      if (param.contract === null) continue;
      try {
        buildValueAs(param.contract.type, coerced[param.name]);
      } catch (error) {
        if (error instanceof Error) throw new TypeError(`input ${reprText(param.name)}: ${error.message}`);
        throw error;
      }
    }
  } catch (error) {
    if (error instanceof Error) throw new ReportedToolError(`bad inputs: ${error.message}`);
    throw error;
  }
}

/**
 * What the flow `flow` can do, over its own nodes and the flows it uses: the node kinds, the
 * tools its TOOL nodes call, the tools and skills its agents are granted, and whether it runs
 * code, with the files its CODE references import (`files`, as shown); one line each, for the
 * approval card.
 */
function capabilities(flow: ComposedFlow, files: readonly string[]): string[] {
  const kinds = new Set<string>();
  const called = new Set<string>();
  const granted = new Set<string>();
  for (const [, node] of staticNodes(flow)) {
    // Every flow has its start and end; they do nothing to approve.
    if (node.kind !== "start" && node.kind !== "end") kinds.add(node.kind);
    const call = node.hostToolCall();
    if (call !== null) called.add(call.toolId);
    for (const grant of node.grantedTools()) granted.add(grant);
  }
  const skills = new Set<string>();
  for (const [, grant] of skillGrants(flow.compiled)) {
    for (const name of grant === "all" ? ["all the skills the run has"] : grant) skills.add(name);
  }
  const listed = (names: ReadonlySet<string>) => Array.from(names).sort(compareCodeUnits).join(", ");
  const lines = [`node kinds: ${listed(kinds) || "none"}`];
  if (called.size > 0) lines.push(`tools it calls: ${listed(called)}`);
  if (granted.size > 0) lines.push(`tools its agents may call: ${listed(granted)}`);
  if (skills.size > 0) lines.push(`skills its agents get: ${listed(skills)}`);
  if (kinds.has("code")) {
    const named = files.length > 0 ? `; files: ${files.join(", ")}` : "";
    lines.push(`runs code on this machine (CODE nodes${named})`);
    lines.push("an approval for this chat doesn't cover changes to the files the code imports");
  }
  return lines;
}

/**
 * The child run `id` names, for a tool.
 *
 * @throws {ReportedToolError} When no child has the id, or it isn't a string.
 */
function knownRun(runs: ChildRuns, args: Record<string, unknown>): string {
  const id = stringArg(args, "run");
  try {
    runs.status(id);
  } catch (error) {
    if (error instanceof RangeError) throw new ReportedToolError(error.message);
    throw error;
  }
  return id;
}

/**
 * Load the flow file `path` in `host.root` for `start_flow`, or `content` as that file when
 * given: its real path, text and flow.
 *
 * @throws {ToolRefused} For a path that is refused, or a file that isn't a flow file or is too large.
 * @throws {ReportedToolError} For a file that can't be read, or a flow that doesn't load.
 */
async function loadForStart(
  host: ChatToolHost,
  path: string,
  content: string | undefined,
): Promise<{ real: string; shown: string; text: string; flow: ComposedFlow }> {
  const root = resolveRoot(host.root);
  const real = confine(root, path);
  if (content !== undefined) refuseContent(path, real, content);
  const text = content ?? readFlow(path, real);
  try {
    const flow = await loadConfinedText(root, real, text, host.env);
    return { real, shown: slashed(relative(root, real)), text, flow };
  } catch (error) {
    if (error instanceof LoadError) {
      throw new ReportedToolError(`invalid:\n${await renderedDiagnostic(error.diagnostic, host.env)}`);
    }
    throw error;
  }
}

/**
 * The chat assistant's run tools over `runs`: `start_flow`, which asks the user through
 * `host.approve` before it starts a flow in `host.root`; `flow_status` and `wait_flow`, which
 * read a child's status; and `answer_flow`, `message_flow`, `close_inbox` and `stop_flow`, which
 * relay to it. Each returns the child's status as JSON, its text fields cut at
 * `STATUS_TEXT_CHARS`, and marks it read (`runs.markRead`, `options.seen`). An unknown run, or an
 * argument of the wrong type, is a `ReportedToolError`. With `options.report`, also
 * `report_issue`, which starts the bundled report flow as a child run without asking: only that
 * run gets `file_issue`, whose own approval card guards anything that leaves the machine.
 */
export function chatRunTools(host: ChatToolHost, runs: ChildRuns, options: ChatRunToolsOptions = {}): FunctionTool[] {
  const { seen = null, scheduler = realScheduler, report = null } = options;
  // The approvals given "for this chat", by real path: the digest of what was approved.
  const approved = new Map<string, string>();
  // By child, the last seq a tool showed the model; with `seen`'s, what `wait_flow` waits past.
  const shown = new Map<string, number>();
  const lastSeen = (run: string) => Math.max(shown.get(run) ?? 0, seen?.seen(run) ?? 0);
  const show = (status: ChildStatus): string => {
    const last = lastSeen(status.run);
    runs.markRead(status.run, status.seq);
    seen?.markSeen(status.run, status.seq);
    shown.set(status.run, Math.max(status.seq, last));
    // The stored status keeps the replies it was published with; once shown, they are read.
    const { replies: _, ...unrepeated } = status;
    return JSON.stringify(shownStatus(status.seq <= last ? unrepeated : status));
  };
  const runArgument = (description: string, ...more: [name: string, description: string][]) =>
    stringArguments(
      ["run", description, true],
      ...more.map(([name, text]): [string, string, boolean] => [name, text, true]),
    );
  const RUN = "The run's id, as start_flow returned it (r1, r2, ...).";
  const STATUS =
    "Returns the run's status as JSON: status (running, waiting, succeeded, failed or cancelled), output, " +
    "failure, waits, asks (questions with their call ids), replies (what it said since you last looked), note " +
    "(something that couldn't be done) and log.";
  return [
    tool({
      name: "start_flow",
      description:
        "Start a flow file (.yaml, .yml) in the chat's folder as a background run of this chat, once the user " +
        "approves it on their terminal. Run check_flow first to learn its inputs, and ask the user for input " +
        "values you don't know rather than inventing them; an Inbox input is never given here (use " +
        "message_flow). With content, runs that text as the file at path would run, without " +
        `writing it: a draft. ${STATUS} The status's run is the id the other run tools take. ` +
        "Returns `not started: <why>` when the user rejects it or it can't start; don't retry a " +
        "rejected start unchanged.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "The flow file, relative to the chat's folder." },
          inputs: { type: "object", description: "The flow's inputs by name; {} for none." },
          content: {
            type: "string",
            description: "A draft's whole text, run in place of the file's.",
          },
        },
        required: ["path", "inputs"],
        additionalProperties: false,
      },
      execute: (args, ctx) => {
        const content = args["content"] === undefined ? undefined : stringArg(args, "content");
        return startFlow(stringArg(args, "path"), args["inputs"], content, ctx.signal);
      },
    }),
    tool({
      name: "flow_status",
      description: `Read where a run this chat started stands now, without waiting. ${STATUS}`,
      parameters: runArgument(RUN),
      isConcurrencySafe: () => true,
      execute: (args) => show(runs.status(knownRun(runs, args))),
    }),
    tool({
      name: "wait_flow",
      description:
        "Wait for a run this chat started to change (a reply, a question, its end), up to `seconds` " +
        `(at most ${WAIT_FLOW_MAX_S}), and return its status then, or at the timeout. Returns at once when ` +
        `it changed since you last saw it. ${STATUS}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: RUN },
          seconds: {
            type: "number",
            description: `How long to wait at most, in seconds (at most ${WAIT_FLOW_MAX_S}).`,
          },
        },
        required: ["run", "seconds"],
        additionalProperties: false,
      },
      isConcurrencySafe: () => true,
      execute: async (args, ctx) => {
        const id = knownRun(runs, args);
        const seconds = args["seconds"];
        if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
          throw new ReportedToolError("seconds must be a number of seconds, 0 or more");
        }
        const done = new AbortController();
        const end = () => done.abort();
        ctx.signal.addEventListener("abort", end, { once: true });
        if (ctx.signal.aborted) end();
        scheduler.sleep(Math.min(seconds, WAIT_FLOW_MAX_S) * 1000, done.signal).then(end, () => {});
        try {
          return show(await runs.next(id, lastSeen(id), { signal: done.signal }));
        } finally {
          ctx.signal.removeEventListener("abort", end);
          end();
        }
      },
    }),
    tool({
      name: "answer_flow",
      description:
        "Answer a question a run this chat started asks (its status's asks), by its call id, with the user's " +
        "answer: relay the question to the user and pass on what they say, and don't answer a question twice. " +
        `The answer is any JSON value but null; text for a plain question. ${STATUS}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: RUN },
          call: { type: "string", description: "The question's call id, from the status's asks." },
          answer: { description: "Any JSON value; text for a plain question." },
        },
        required: ["run", "call", "answer"],
        additionalProperties: false,
      },
      execute: async (args) => {
        const id = knownRun(runs, args);
        const call = stringArg(args, "call");
        const answer = args["answer"];
        if (answer === null || answer === undefined) throw new ReportedToolError("answer can't be null");
        return show(await runs.answer(id, call, answer));
      },
    }),
    tool({
      name: "message_flow",
      description:
        "Send text to an Inbox input of a run this chat started, as a chat message to a flow that takes them; " +
        `its status's waits name the inbox a run waits on. ${STATUS}`,
      parameters: runArgument(RUN, ["inbox", "The Inbox input's name."], ["text", "The message."]),
      execute: async (args) => {
        const id = knownRun(runs, args);
        return show(await runs.message(id, stringArg(args, "inbox"), stringArg(args, "text")));
      },
    }),
    tool({
      name: "close_inbox",
      description:
        "Close an Inbox input of a run this chat started, so the flow stops waiting for messages there and " +
        `can finish. ${STATUS}`,
      parameters: runArgument(RUN, ["inbox", "The Inbox input's name."]),
      execute: async (args) => {
        const id = knownRun(runs, args);
        return show(await runs.closeInbox(id, stringArg(args, "inbox")));
      },
    }),
    tool({
      name: "stop_flow",
      description: `Cancel a run this chat started; use it only when the user asks. ${STATUS}`,
      parameters: runArgument(RUN),
      execute: async (args) => show(await runs.stop(knownRun(runs, args))),
    }),
    ...(report === null ? [] : [reportTool(report)]),
  ];

  /** The `report_issue` tool over `report`. */
  function reportTool(options: ReportOptions): FunctionTool {
    return tool({
      name: "report_issue",
      description:
        "Help the user report something unexpected yian did, as an issue for its maintainers " +
        "at github.com/ngocbh/yian. Starts a background run that drafts the report, may ask the " +
        "user for missing facts (relay its questions as for any run), redacts what looks like a " +
        "secret, then shows the user the report on their terminal to file (gh, GITHUB_TOKEN or a " +
        "browser link) or only save. The chat adds yian's version, the runtime, the platform and " +
        "this chat's log path. In context, give what you know: the command or flow run, the " +
        `error text and a run's log path; never a log's contents, a token or a password. ${STATUS}`,
      parameters: stringArguments(
        ["problem", "What went wrong, in a sentence or two.", true],
        [
          "context",
          "What you know about it: the command or flow, the error text, a run's log path.",
          false,
        ],
      ),
      execute: async (args, ctx) => {
        const problem = stringArg(args, "problem");
        const known = args["context"] === undefined ? "" : stringArg(args, "context");
        const context = [options.facts, known].filter((part) => part !== "").join("\n");
        const inputs = { problem, context, homes: options.homes };
        let id: string;
        try {
          const flow = await loadReportFlow(host.env);
          const tools = { file_issue: options.fileIssue };
          ({ id } = await runs.start(flow, inputs, { signal: ctx.signal, tools }));
        } catch (error) {
          if (error instanceof Error) return `not started: ${error.message}`;
          throw error;
        }
        return show(runs.status(id));
      },
    });
  }

  /**
   * Start the flow at `path`, or the draft `content` as that file, with `inputs` once the user
   * approves it, as `start_flow` describes. Every outcome once the user is asked is returned.
   */
  async function startFlow(
    path: string,
    inputs: unknown,
    content: string | undefined,
    signal: AbortSignal,
  ): Promise<string> {
    const { real, shown: file, text, flow } = await loadForStart(host, path, content);
    checkStartInputs(flow, inputs);
    const files = codeFiles(flow);
    const root = resolveRoot(host.root);
    const can = capabilities(flow, files.map((each) => slashed(relative(root, each))));
    const digest = startDigest(flow, text, files, can);
    if (approved.get(real) !== digest) {
      // A draft is on no disk for the user to read, so the card shows all of it.
      const draft = content !== undefined;
      const answer = await host.approve({
        title: draft ? `Run ${file} (a draft, not written)` : `Run ${file}`,
        body: [
          `Starts ${draft ? "the draft " : ""}${file} in the background.`,
          `inputs: ${cutText(JSON.stringify(inputs))}`,
          ...can,
        ].join("\n"),
        diff: draft
          ? unifiedDiff("", text, { oldLabel: "/dev/null", newLabel: `${file} (draft)` })
          : null,
        choices: START_CHOICES,
        signal,
      });
      if (answer === null) return "not started: not approved (no terminal, or the prompt ended)";
      if (answer === START_CHOICES[1]) approved.set(real, digest);
      else if (answer !== START_CHOICES[0]) return "not started: rejected";
    }
    let id: string;
    try {
      ({ id } = await runs.start(flow, inputs as Record<string, unknown>, { signal, name: file }));
    } catch (error) {
      if (error instanceof Error) return `not started: ${error.message}`;
      throw error;
    }
    return show(runs.status(id));
  }
}
