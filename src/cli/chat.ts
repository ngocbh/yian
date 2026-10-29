/**
 * `yian chat [FLOW]`: chat with yian's assistant, a bundled flow (`flows/chat.yaml`) that helps
 * the user discover, read, check, write and run flows in the folder the chat started in, and
 * hosts the flows it runs as child runs of the chat. With FLOW, the chat opens with that flow in
 * focus.
 *
 * The chat is an `inboxChat` of the bundled flow, given the assistant's file and run tools, its
 * bundled skill, and the events of its child runs (`ChildRuns`), which the chat relays to the
 * assistant as messages authored `yian`.
 *
 * Knows about: compose, the run boundary's LLM cascade, connections, skills, the bundle, the
 * chat's tools, child runs and events, the inbox chat and the CLI's start-up helpers.
 */

import { mkdirSync, realpathSync } from "node:fs";
import { basename, dirname, extname, join, relative } from "node:path";

import { copyGraph } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { loadFlow } from "../compose/loader.ts";
import { defaultSkillRoots, loadSkills } from "../compose/skills.ts";
import type { Env } from "../defaults.ts";
import { connectionSnapshot } from "../llm_clients/connection.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import { coerceInputs } from "../nodes/binding.ts";
import { isSystemError, leavesRoot, realScheduler } from "../platform.ts";
import { hostInputs, runStartLlm, skillGrants } from "../run.ts";
import { type Skill } from "../skills/catalog.ts";
import { DOCS, FLOWS } from "./bundled.ts";
import { bundledSkill } from "./bundled_skill.ts";
import { EventQueue, InFlight, trackInFlight } from "./chat_events.ts";
import { chatRunTools } from "./chat_run_tools.ts";
import { ChildRuns, type WaitingChild } from "./chat_runs.ts";
import { type ChatToolHost, chatFileTools, checkFocus, slashed } from "./chat_tools.ts";
import { inboxChat, type InboxChatDeps } from "./inbox_chat.ts";
import { printStyled } from "./render.ts";
import { removeEmptyDir, validateRunLogOptions } from "./run.ts";
import { sanitize } from "./sanitize.ts";
import { cliLayer, defaultRunsDir, freshLog } from "./startup.ts";
import type { CliIo } from "./utils.ts";

/** The label the bundled flow's diagnostics and recorded source go by. */
const CHAT_LABEL = "chat.yaml";

/** The path of the assistant's reply agent in the bundled flow, whose model the banner names. */
const REPLY_PATH = "chat/answered/reply";

/** The options `chat` reads; an absent option is `undefined`, an absent flag `false`. */
export interface ChatOptions {
  log?: string | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  engineTrace: boolean;
}

/**
 * @internal Test seams: the inbox chat's, the folder the chat works in (the current directory by
 * default), and the model builder the chat's and its children's agents use.
 */
export type ChatDeps = InboxChatDeps & { root?: string; models?: ModelAccess | null };

/**
 * Load the bundled chat flow, its `uses:` resolved to the bundled compaction flow.
 *
 * @throws {LoadError} Never for the bundle as committed; the bundle test loads it.
 */
export async function loadChatFlow(env: Env): Promise<ComposedFlow> {
  const compaction = await loadFlow(FLOWS["compact_history.yaml"]!, { sourceLabel: "compact_history.yaml", env });
  return loadFlow(FLOWS[CHAT_LABEL]!, { sourceLabel: CHAT_LABEL, env, childResolver: async () => compaction });
}

/** `provider/model` the assistant's replies use under `layer`, or `null` when no layer sets them. */
function replyModel(flow: ComposedFlow, layer: Readonly<Record<string, string>>, env: Env): string | null {
  try {
    const inputs = coerceInputs(flow.input, hostInputs(flow.input, {}));
    const resolved = runStartLlm(copyGraph(flow.compiled), inputs, { ...layer }, connectionSnapshot({ env }));
    const config = resolved.effective[REPLY_PATH];
    if (typeof config?.["provider"] !== "string" || typeof config["model"] !== "string") return null;
    return `${config["provider"]}/${config["model"]}`;
  } catch {
    // The bundled flow sets no model, so only a missing provider or model fails to resolve.
    return null;
  }
}

/** The flow in focus as the assistant names it: relative to `root` when inside it, else its real path. */
function focusName(root: string, path: string): string {
  let real: string;
  try {
    real = realpathSync(path);
  } catch {
    return path;
  }
  const inside = relative(realpathSync(root), real);
  return leavesRoot(inside) ? real : slashed(inside);
}

/** The file the children's CODE console output goes to: beside the chat's log, named after it. */
function consoleFileFor(log: string): string {
  return join(dirname(log), `${basename(log, extname(log))}.child-output.log`);
}

/** The lines telling the user how each child left waiting can go on. */
function waitingLines(children: readonly WaitingChild[]): string {
  return children
    .map((child) => {
      const what = { question: "an answer to its question", inbox: "a message", timer: "a timer" }[child.kind];
      const next =
        child.kind === "timer"
          ? `continue with: yian run --resume ${child.log}`
          : "it can't be continued outside this chat yet";
      return `run ${child.run} waits for ${what}; its log: ${child.log}\n  ${next}\n`;
    })
    .join("");
}

/**
 * `yian chat`: chat with yian's assistant in the folder `deps.root` (the current directory), with
 * `flowPath` in focus when given: its text and check go to the assistant each turn. Returns the
 * exit code.
 *
 * The chat runs the bundled flow as `inboxChat` runs a flow (the prompt, Ctrl-C, Esc and `/exit`
 * work as there; replies are printed on stdout through `sanitize`), its log at `options.log` or
 * in a new run folder under `~/.yian/runs`. Without a provider and model for the assistant, it
 * prints `chat failed: ...` and exits 1. A banner line on stderr names the assistant's model
 * and the flow in focus; when that flow has an `Inbox` input, a hint says `yian run` chats with
 * it. The assistant gets the file tools (`chatFileTools`, over the docs bundle) and the run tools
 * (`chatRunTools`) over the chat's child runs, whose logs go in new run folders under
 * `~/.yian/runs` and whose CODE console output goes to a file beside the chat's log; and the
 * bundled `composing-agents` skill (`bundledSkill`). A child granted skills gets them as
 * `yian run` reads them from the chat's folder.
 *
 * When the chat ends, the children still running are cancelled, and each child left waiting is
 * printed with its log and how to go on: one waiting for a timer resumes with `yian run --resume`;
 * one waiting for an answer or a message can't be continued outside the chat.
 *
 * @throws {UsageError} As `validateRunLogOptions` does for `--log`.
 */
export async function chat(
  flowPath: string | undefined,
  options: ChatOptions,
  io: CliIo,
  deps: ChatDeps = {},
): Promise<number> {
  validateRunLogOptions({ flow: CHAT_LABEL, inputValues: [], log: options.log });
  const root = deps.root ?? process.cwd();
  const env = io.env;
  const flow = await loadChatFlow(env);
  let skill: Skill;
  try {
    skill = bundledSkill(env);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    await printStyled(io, `chat failed: can't write the bundled skill: ${error.message}`, "red");
    return 1;
  }

  // Both always given, so neither is asked for at the terminal.
  const inputs: Record<string, unknown> = { flow: null, focus: null };
  let hint = "";
  if (flowPath !== undefined) {
    const focus = await checkFocus(flowPath, env);
    const name = focusName(root, flowPath);
    inputs["flow"] = name;
    inputs["focus"] =
      `The user opened this chat to get help with the flow ${name}.\n\n` +
      `Its text:\n${focus.text}\n\nIts check (check_flow):\n${focus.check}`;
    if (focus.inbox) hint = `to chat with this flow itself, run yian run ${flowPath}\n`;
  }
  const model = replyModel(flow, cliLayer(options), env);
  if (model === null) {
    const message =
      "chat failed: the assistant has no provider or model; run 'yian auth <provider>', or pass " +
      "--provider and --model";
    await printStyled(io, message, "red");
    return 1;
  }
  const banner = [`yian chat`, model, flowPath === undefined ? null : `focus: ${inputs["flow"]}`]
    .filter((part) => part !== null)
    .join(" · ");
  await io.stderr.write(`${sanitize(banner)}\n${sanitize(hint)}`);

  const fresh = freshLog(options.log, env);
  const logDir = defaultRunsDir(env);
  // `freshLog` made it for a default log; a child's run folder goes in it either way.
  mkdirSync(logDir, { recursive: true, mode: 0o700 });
  let rootSkills: Map<string, Skill> | null = null;
  const skillsFor = (child: ComposedFlow): ReadonlyMap<string, Skill> | null => {
    if (skillGrants(child.compiled).length === 0) return null;
    rootSkills ??= loadSkills(defaultSkillRoots({ cwd: root, env }), {
      skipInvalid: true,
      warn: (message) => void printStyled(io, `warning: ${message}`, "yellow").catch(() => {}),
    });
    return rootSkills;
  };
  const scheduler = deps.scheduler ?? realScheduler;
  const wake = new EventTarget();
  const runs: ChildRuns = new ChildRuns({
    root,
    logDir,
    env,
    llmConfig: cliLayer(options),
    models: deps.models ?? null,
    scheduler,
    chatRunId: deps.runId ?? fresh.runId,
    skillsFor,
    consoleFile: consoleFileFor(fresh.path),
    onStatus: (status) => queue.push(status),
  });
  const queue = new EventQueue({ wake, markRead: runs.markRead.bind(runs) });
  const inFlight = new InFlight(wake);
  const docs = new Map(Object.entries(DOCS));
  const tools: InboxChatDeps["tools"] = (terminal) => {
    const host: ChatToolHost = { root, approve: (request) => terminal.approve(request), env, docs };
    const all = trackInFlight(
      [...chatFileTools(host), ...chatRunTools(host, runs, { seen: queue, scheduler })],
      inFlight,
    );
    return Object.fromEntries(all.map((each) => [each.name, each]));
  };
  try {
    return await inboxChat(
      CHAT_LABEL,
      {
        inputValues: [],
        log: fresh.path,
        provider: options.provider,
        model: options.model,
        engineTrace: options.engineTrace,
      },
      io,
      {
        ...deps,
        runId: deps.runId ?? fresh.runId,
        inputs,
        skills: new Map([[skill.name, skill]]),
        tools,
        events: { queue, inFlight, wake, onInterrupt: () => runs.cancelWaits() },
        sanitizeReplies: true,
      },
      flow,
    );
  } finally {
    const waiting = await runs.dispose();
    await io.stderr.write(waitingLines(waiting));
    if (fresh.runDir !== null) removeEmptyDir(fresh.runDir);
  }
}
