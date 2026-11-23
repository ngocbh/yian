/**
 * The host tools `yian chat`'s assistant runs flows with (`chatRunTools`): the child-flow group,
 * whose `start` starts a flow in the chat's folder as a child run once the user approves it on the
 * terminal, and whose other tools read, wait on, answer, message and stop the runs it started; and
 * the tools that start the bundled flows. As with the file tools (`chat_tools.ts`), the model always
 * gets text back: a tool returns the run's status as JSON, or throws a `ReportedToolError` it reads.
 * The group's `start` asks the chat which flow a call names and whether the user lets it start
 * (`chatStart`). The runs are `ChatRuns`: the child-flow driver with the chat's hooks; the assistant
 * and each authoring run hold theirs through an owner.
 *
 * Knows about: the host tool API and its errors, confined loading, the run boundary's flow readers,
 * CODE references, the chat's file tools and their helpers, its child runs, their driver's hooks
 * and tool group, the CLI's key preflight and memory binding, the default embeddings, and its
 * bundled flows.
 */

import { createHash } from "node:crypto";
import { appendFileSync, readFileSync } from "node:fs";
import { relative, resolve } from "node:path";

import {
  type ChildFlowOwner,
  ChildFlows,
  childFlows,
  type ChildFlowStart,
  type ChildFlowsOptions,
  type ChildFlowTarget,
  type ChildStatus,
  cutStatusText,
  shownStatus,
} from "../child_flows.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import type { Embed } from "../llm/embeddings.ts";
import { defaultEmbedding } from "../llm_clients/connection.ts";
import { defaultEmbed } from "../llm_clients/factory.ts";
import { splitReference } from "../nodes/code/worker.ts";
import { isSystemError } from "../platform.ts";
import { type ConsoleHook, skillGrants, staticNodes } from "../run.ts";
import type { Skill } from "../skills/catalog.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, stringArg, stringArguments, tool } from "../tools/function_tools.ts";
import { grantText, visibleToolName } from "../tools/grants.ts";
import type { ToolEntry } from "../tools/registry.ts";
import {
  type ChatToolHost,
  chatFileTools,
  confine,
  confineFlow,
  readFlow,
  refuseContent,
  renderedDiagnostic,
  resolveRoot,
  slashed,
} from "./chat_tools.ts";
import { type AuthoredGrant, AUTHOR_TOOLS, loadAuthorFlow, loadReviewFlow, REVIEW_TOOLS } from "./authoring.ts";
import { loadConfinedText } from "./confined_load.ts";
import { unifiedDiff } from "./diff.ts";
import {
  bindMemories,
  type CliMemory,
  type CreatedMemory,
  createdLine,
  type DefaultMemory,
  defaultMemories,
  driftNote,
  memoryLocation,
  projectFolder,
  withoutLocations,
} from "./memory_bind.ts";
import { loadReportFlow } from "./report_issue.ts";
import { ensureProviderKeys } from "./startup.ts";
import { CommandEnv } from "./utils.ts";

/** The approval answers the group's `start` offers: the first two start, the last (Ctrl-C's) doesn't. */
const START_CHOICES = ["Approve", "Approve this flow for this chat", "Reject"] as const;

/** What the group's `start` starts in the chat, as its description begins. */
const START =
  "Start a flow file (.yaml, .yml) in the chat's folder as a background run of this chat, once the " +
  "user approves it on their terminal; flow is its path, relative to the chat's folder. Run check_flow " +
  "first to learn its inputs, and ask the user for input values you don't know rather than inventing " +
  "them. With content, runs that text as the file at flow would run, without writing it: a draft.";

/** The status the tools that start a bundled flow return. */
const STATUS =
  "Returns the run's status, a record: status (running, waiting, succeeded, failed or cancelled), " +
  "output, failure, waits, asks (questions with their call ids), replies (what it said since " +
  "you last looked), memory_notes (a memory it created or that changed, since you last looked), " +
  "note (something that couldn't be done) and log.";

/** What `author_flow` and `review_flow` start their runs with. */
export interface AuthoringOptions {
  /** The skills the authoring flows' agents load: the bundled `composing-agents`. */
  skills: ReadonlyMap<string, Skill>;
}

/** What `report_issue` starts its run with, beside what the model passes. */
export interface ReportOptions {
  /** The chat's own facts (`hostFacts`), first in the report's context. */
  facts: string;
  /** The home folder, as set and as its real path, which the report shows as `~`. */
  homes: readonly string[];
  /** The `file_issue` tool the report flow files with; no other run gets it. */
  fileIssue: FunctionTool;
}

/** What the run tools need beside the file tools' host and the assistant's owner. */
export interface ChatRunToolsOptions {
  /** What `report_issue` needs; without it, there is no `report_issue`. */
  report?: ReportOptions | null;
  /** What `author_flow` and `review_flow` need; without it, there are neither. */
  authoring?: AuthoringOptions | null;
}

/** What the chat's child runs need from the chat (`ChatRuns`). */
export interface ChatRunsOptions extends Omit<ChildFlowsOptions, "codeRoot" | "memories" | "preflight" | "onConsole"> {
  /** The folder the chat's flows are confined to: their CODE files must resolve inside it. */
  root: string;
  /** The home folder whose `~/.yian/memory` holds the children's default memories. */
  home: string;
  /** For tests: the embeddings call VectorDB memories make; `defaultEmbed` over `env` when absent. */
  embed?: Embed;
  /**
   * The file the console output of the children's CODE nodes is appended to, so it doesn't land on
   * the chat's terminal; `null` (the default) leaves it on the console.
   */
  consoleFile?: string | null;
}

/**
 * Return an `onConsole` that appends each line a child's CODE prints to `file`, so it doesn't
 * land on the chat's terminal.
 */
function consoleTo(file: string): ConsoleHook {
  return ({ text }) => {
    try {
      appendFileSync(file, `${text}\n`);
    } catch {
      // A line that can't be kept mustn't fail the node that printed it.
    }
  };
}

/**
 * The chat's child runs: the driver with the chat's hooks. A missing provider key is reported to
 * the assistant rather than prompted for mid-chat. A child's memory inputs are bound to their
 * defaults under `home`, as `yian run` binds them; a memory a pass creates is said by identity
 * alone, and an error the driver reports names no location, since a status reaches the chat's log
 * and its model, which never hold one.
 */
export class ChatRuns extends ChildFlows {
  readonly #home: string;
  readonly #env: Env;

  constructor({ root, home, embed, consoleFile = null, ...options }: ChatRunsOptions) {
    const { env } = options;
    super({
      ...options,
      codeRoot: root,
      preflight: (flow, llmConfig, inputs) =>
        ensureProviderKeys(flow, llmConfig, inputs, { env: new CommandEnv(env), terminal: null }),
      memories: {
        bind: (flow, project, note) => {
          const told = (created: CreatedMemory) => note(createdLine(created, false));
          const given = { home, note: told, project, memoryFlag: false, embed: embed ?? defaultEmbed(env) };
          return bindMemories(flow, [], { ...given, defaultEmbedding: () => defaultEmbedding({ env }) });
        },
        drift: driftNote,
        // A memory's open names its folder or store. A store's own folder, in an OS error on
        // making it, is under ~/.yian/memory, written from `~`.
        scrub: (error, memory) => {
          // The instances are the ones `bind` made.
          const roots = Object.values(memory as Readonly<Record<string, CliMemory>>).map(memoryLocation);
          return withoutLocations(error, home, roots);
        },
      },
      onConsole: consoleFile === null ? null : consoleTo(consoleFile),
    });
    this.#home = home;
    this.#env = env;
  }

  /**
   * Return the default memories `flow`'s start would bind, keyed by `project` as `start` keys
   * them, by identity and whether each exists yet: for the start's approval, never a location. A
   * new store's embedding model is the default `start` would use.
   *
   * @throws {RangeError} As `start` throws for a memory input with no default, or a new store with
   *   no embedding model, naming no location.
   */
  defaultMemories(flow: ComposedFlow, project: string | null): DefaultMemory[] {
    const home = this.#home;
    const env = this.#env;
    try {
      const options = { home, project, memoryFlag: false, defaultEmbedding: () => defaultEmbedding({ env }) };
      return defaultMemories(flow, options);
    } catch (error) {
      throw withoutLocations(error, home, []);
    }
  }
}

/**
 * A flow the group's `start` names in the chat's folder: the file it is listed by, its real path and
 * text, and whether it is a draft.
 */
interface ChatStartTarget extends ChildFlowTarget {
  readonly name: string;
  readonly real: string;
  readonly text: string;
  readonly draft: boolean;
}

/**
 * The start card's title for `file`: a draft says it isn't written, and a start a bundled flow
 * (`requester`) asks for names it.
 */
function startTitle(file: string, draft: boolean, requester: string | null): string {
  const notes = [draft ? "a draft" : null, requester === null ? null : `for ${requester}`];
  const shown = notes.filter((note) => note !== null);
  if (draft && requester === null) shown.push("not written");
  return shown.length === 0 ? `Run ${file}` : `Run ${file} (${shown.join(", ")})`;
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
    if (call !== null) called.add(visibleToolName(call.group, call.toolId));
    for (const grant of node.grantedTools()) granted.add(grantText(grant));
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
 * Load the flow file `path` in `host.root` for the group's `start`, or `content` as that file when
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
 * The group's `start` in the chat, for one owner's runs, started by `requester` (a bundled flow, or
 * `null` for the assistant): its `flow` names a flow file in `host.root`, or its `content` a draft
 * of that file, whose memories are keyed by the folder `flow` names, as `yian run` keys them; the
 * user approves each start on the terminal, unless they approved the same flow, unchanged, for this
 * chat. `approved` holds those approvals, by real path, as the digest of what was approved; every
 * owner of the chat shares it.
 */
function chatStart(
  host: ChatToolHost,
  runs: ChatRuns,
  approved: Map<string, string>,
  requester: string | null,
): ChildFlowStart<ChatStartTarget> {
  return {
    description: START,
    resolve: async ({ flow: path, content }) => {
      const { real, shown, text, flow } = await loadForStart(host, path, content ?? undefined);
      const project = projectFolder(resolve(resolveRoot(host.root), path));
      return { flow, name: shown, project, real, text, draft: content !== null };
    },
    approve: async ({ flow, name: file, project = null, real, text, draft }, inputs, signal) => {
      const files = codeFiles(flow);
      const root = resolveRoot(host.root);
      let memories: DefaultMemory[];
      try {
        memories = runs.defaultMemories(flow, project);
      } catch (error) {
        if (error instanceof Error) return error.message;
        throw error;
      }
      const can = capabilities(flow, files.map((each) => slashed(relative(root, each))));
      // The user sees when a flow (a draft, say) would use a memory another flow made. Which one it
      // binds is part of what is approved; whether it exists yet isn't.
      const named = memories.map(({ input, identity }) => `${input} → ${identity}`);
      const marked = memories.map(({ exists }, i) => `${named[i]} (${exists ? "existing" : "new"})`);
      const memoryLine = (list: readonly string[]) => (list.length === 0 ? [] : [`memories: ${list.join(", ")}`]);
      const digest = startDigest(flow, text, files, [...can, ...memoryLine(named)]);
      if (approved.get(real) === digest) return null;
      // A draft is on no disk for the user to read, so the card shows all of it.
      const answer = await host.approve({
        title: startTitle(file, draft, requester),
        body: [
          `Starts ${draft ? "the draft " : ""}${file} in the background.`,
          `inputs: ${cutStatusText(JSON.stringify(inputs))}`,
          ...can,
          ...memoryLine(marked),
        ].join("\n"),
        diff: draft
          ? unifiedDiff("", text, { oldLabel: "/dev/null", newLabel: `${file} (draft)` })
          : null,
        choices: START_CHOICES,
        signal,
      });
      if (answer === null) return "not approved (no terminal, or the prompt ended)";
      if (answer === START_CHOICES[1]) approved.set(real, digest);
      else if (answer !== START_CHOICES[0]) return "rejected";
      return null;
    },
  };
}

/**
 * The chat assistant's run tools over `runs`, which it holds through `owner`, by the key the chat
 * passes each under: the child-flow group under `owner.group` (`childFlows`), whose `start` asks the
 * user through `host.approve` before it starts a flow in `host.root` (`chatStart`). With
 * `options.report`, also `report_issue`, which starts the bundled report flow as a child run without
 * asking: only that run gets `file_issue`, whose own approval card guards anything that leaves the
 * machine. With `options.authoring`, also `author_flow` and `review_flow`, which start the bundled
 * authoring flows likewise (`authoringTools`). Each returns the child's status as JSON, as the
 * group's tools do, and marks it read through `owner`.
 */
export function chatRunTools(
  host: ChatToolHost,
  runs: ChatRuns,
  owner: ChildFlowOwner,
  options: ChatRunToolsOptions = {},
): Record<string, FunctionTool | FunctionTool[]> {
  const { report = null, authoring = null } = options;
  // The approvals given "for this chat", which every start in it shares.
  const approved = new Map<string, string>();
  const show = (id: string) => shownStatus(owner.show(owner.status(id)));
  const byName = (tools: readonly FunctionTool[]) => Object.fromEntries(tools.map((each) => [each.name, each]));
  return {
    [owner.group]: childFlows(owner, { start: chatStart(host, runs, approved, null) }),
    ...(report === null ? {} : byName([reportTool(report)])),
    ...(authoring === null ? {} : byName(authoringTools(authoring))),
  };

  /**
   * Once child `id` ends, cancel each run that `own`, its hold on the runs it started, still reaches,
   * such as a try it stopped waiting for. A child the chat stops without its end is left to the
   * chat's end.
   */
  async function cancelWhenEnded(id: string, own: ChildFlowOwner): Promise<void> {
    const ended = (status: ChildStatus) => !["running", "waiting"].includes(status.status);
    let status = runs.status(id);
    while (!ended(status)) {
      const next = await runs.next(id, status.seq);
      // The same status back: its driver stopped without ending it.
      if (next.seq === status.seq) return;
      status = next;
    }
    for (const child of runs.live()) {
      if (own.reaches(child.id)) await runs.stop(child.id);
    }
  }

  /**
   * `author_flow` and `review_flow`, which start the bundled authoring flows as child runs
   * without asking. The author run gets `AUTHOR_TOOLS` (the file tools over `host`, and the
   * `start` and `wait` of a group of its own), whose writes and starts ask the user as the
   * assistant's do; the review run only `REVIEW_TOOLS`; both `options.skills`. Once a run ends, the
   * runs it started that still go are cancelled. A path `author_flow` is given is refused at once
   * unless it names a flow file in the chat's folder.
   */
  function authoringTools(options: AuthoringOptions): FunctionTool[] {
    const start = async (
      name: string,
      grants: readonly AuthoredGrant[],
      load: () => Promise<ComposedFlow>,
      inputs: Record<string, unknown>,
      signal: AbortSignal,
    ): Promise<string | Record<string, unknown>> => {
      // Its starts are its own: the runs it starts are its children, through its own owner.
      let own: ChildFlowOwner | null = null;
      const tools = (child: string) => {
        const granted: Record<string, ToolEntry> = {};
        for (const each of chatFileTools(host)) if (grants.includes(each.name)) granted[each.name] = each;
        for (const grant of grants) {
          if (typeof grant === "string") continue;
          for (const [group, names] of Object.entries(grant)) {
            own = runs.owner({ runId: child, group });
            const hooks = chatStart(host, runs, approved, name);
            granted[group] = childFlows(own, { start: hooks }).filter((each) => names.includes(each.name));
          }
        }
        return granted;
      };
      let id: string;
      try {
        const flow = await load();
        ({ id } = await owner.start(flow, inputs, { signal, tools, skills: options.skills }));
      } catch (error) {
        if (error instanceof Error) return `not started: ${error.message}`;
        throw error;
      }
      // A failure here leaves the child's runs to the chat's end, which cancels them too. A run with
      // no group of its own starts none.
      if (own !== null) cancelWhenEnded(id, own).catch(() => {});
      return show(id);
    };
    return [
      tool({
        name: "author_flow",
        description:
          "Build a new flow with the user, or rework one, step by step: a background run pins " +
          "down the goal, agrees the typed inputs, output and nodes with the user, drafts the " +
          "flow, checks it until it loads, reviews it, tries it on sample inputs (the user " +
          "approves the run), refines it with the user, and writes it once they approve the " +
          "write. Relay its questions as for any run. For a small edit, use write_flow instead. " +
          STATUS,
        parameters: stringArguments(
          ["goal", "What the user wants the flow to do, in their words.", true],
          [
            "path",
            "The flow to rework, or where the new one goes, relative to the chat's folder.",
            false,
          ],
        ),
        execute: (args, ctx) => {
          const goal = stringArg(args, "goal");
          const path = args["path"] === undefined ? null : stringArg(args, "path");
          // Refused now, before the user answers anything for a flow that can't be written.
          if (path !== null) confineFlow(host.root, path);
          const load = () => loadAuthorFlow(host.env);
          return start("author_flow", AUTHOR_TOOLS, load, { goal, path }, ctx.signal);
        },
      }),
      tool({
        name: "review_flow",
        description:
          "Review a flow file in the chat's folder for security, reliability, cost and clarity, " +
          "in a background run whose output is its findings, the most severe first: each a " +
          `severity, an area, the node, the problem and a fix. ${STATUS}`,
        parameters: stringArguments([
          "path",
          "The flow file, relative to the chat's folder.",
          true,
        ]),
        execute: (args, ctx) => {
          const path = stringArg(args, "path");
          const text = readFlow(path, confine(resolveRoot(host.root), path));
          const load = () => loadReviewFlow(host.env);
          return start("review_flow", REVIEW_TOOLS, load, { path, text }, ctx.signal);
        },
      }),
    ];
  }

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
          ({ id } = await owner.start(flow, inputs, { signal: ctx.signal, tools }));
        } catch (error) {
          if (error instanceof Error) return `not started: ${error.message}`;
          throw error;
        }
        return show(id);
      },
    });
  }
}
