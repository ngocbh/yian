/**
 * The host side of the issue reports `yian chat`'s assistant helps the user write: the bundled
 * `report_issue.yaml` flow (`loadReportFlow`), the facts the chat adds to a report (`hostFacts`),
 * and the `file_issue` tool the flow's last node calls (`fileIssueTool`). `file_issue` shows the
 * report on an approval card and files it the way the user picks there: with the `gh` CLI, with a
 * `GITHUB_TOKEN` from the environment, through a prefilled browser link, or not at all; every way
 * but Cancel saves it under `~/.yian/reports` too. The token is read here, by the host, and never
 * reaches the model or the run log.
 *
 * Knows about: compose, the bundle, the platform's home, fetch and runtime, the host tool API, the
 * chat's file tools' helpers, the terminal's approval prompt and the chat view's `~` paths.
 */

import { spawn } from "node:child_process";
import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { loadFlow } from "../compose/loader.ts";
import type { Env } from "../defaults.ts";
import {
  homeDir,
  hostPlatform,
  isSystemError,
  safeFetch,
  type Send,
  VERSION,
} from "../platform.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { FLOWS } from "./bundled.ts";
import { stringArg, stringArguments } from "./chat_tools.ts";
import { homePath } from "./chat_view.ts";
import type { ApproveRequest } from "./terminal.ts";
import { osReason } from "./utils.ts";

/** The repository yian's issues are filed in, and its host. */
const REPO = "ngocbh/yian";
const HOST = "github.com";

/** The bundled flow `report_issue` runs. */
const REPORT_FLOW_FILE = "report_issue.yaml";

/**
 * @internal The longest prefilled link `file_issue` gives, in characters: well under what GitHub
 * takes, and short enough that the whole outcome fits what the chat's model reads of a run's
 * output, so the assistant can pass the link on whole.
 */
export const LINK_CHARS = 3500;

/** The most of `LINK_CHARS` a link's title takes, encoded, so the body keeps room. */
const LINK_TITLE_CHARS = 1000;

/** How long `file_issue` waits for `gh auth status` before it leaves gh out, in milliseconds. */
const GH_CHECK_MS = 4000;

/** The most reports saved within one second before `file_issue` gives up on a free name. */
const SAME_SECOND_REPORTS = 100;

/** The approval answers, the ways to file first; only those that are available are offered. */
const GH = "File it with gh";
const TOKEN = "File it with GITHUB_TOKEN";
const LINK = "Open a prefilled link in the browser";
const SAVE = "Save it only";
const CANCEL = "Cancel";

/**
 * Runs `command` with `args`, `input` on its stdin; rejects when it can't start (not installed)
 * or `signal` aborts.
 */
export type Runner = (
  command: string,
  args: readonly string[],
  options?: { input?: string; signal?: AbortSignal },
) => Promise<{ code: number | null; stdout: string; stderr: string }>;

/** What `file_issue` works with. */
export interface FileIssueHost {
  /** Ask the user to approve, as `Terminal.approve` does. */
  approve: (request: ApproveRequest) => Promise<string | null>;
  /** Whether `approve` can ask now (`Terminal.canApprove`). */
  canApprove: () => boolean;
  /** The environment `GITHUB_TOKEN` and `HOME` are read from. */
  env: Env;
  /** The clock a saved report is named by, in epoch milliseconds. */
  now: () => number;
  /** Runs `gh`; `runCommand` by default. */
  run?: Runner;
  /** Sends the GitHub API request, as `safeFetch` sends it; the runtime's fetch by default. */
  send?: Send;
}

/**
 * Load the bundled report flow.
 *
 * @throws {LoadError} Never for the bundle as committed; its test loads it.
 */
export function loadReportFlow(env: Env): Promise<ComposedFlow> {
  return loadFlow(FLOWS[REPORT_FLOW_FILE]!, { sourceLabel: REPORT_FLOW_FILE, env });
}

/**
 * The chat's own facts for a report, one per line: yian's version, the runtime, the platform and
 * the chat's run log `chatLog`, shown with `~` for the home folder as `homes` gives it (as set,
 * then its real path).
 */
export function hostFacts(chatLog: string, homes: readonly string[]): string {
  const { runtime, platform } = hostPlatform();
  return [
    `yian version: ${VERSION}`,
    `runtime: ${runtime}`,
    `platform: ${platform}`,
    `chat run log: ${homePath(chatLog, homes[0], homes[1])}`,
  ].join("\n");
}

/**
 * Run `command` with `args` and no shell, `options.input` written to its stdin, and collect its
 * exit code and output; `options.signal` kills it.
 */
export const runCommand: Runner = (command, args, options = {}) =>
  new Promise((resolve, reject) => {
    const child = spawn(command, [...args], {
      stdio: ["pipe", "pipe", "pipe"],
      signal: options.signal,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk: string) => void (stdout += chunk));
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => void (stderr += chunk));
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    // A command that exits without reading its stdin breaks the pipe; its exit code says why.
    child.stdin.once("error", () => {});
    child.stdin.end(options.input ?? "");
  });

/**
 * The longest prefix of `text`, cut between characters, for which `fits` holds; `fits("")` is
 * taken to.
 */
function longestFitting(text: string, fits: (prefix: string) => boolean): string {
  if (fits(text)) return text;
  const chars = Array.from(text);
  let ok = 0;
  let over = chars.length;
  while (over - ok > 1) {
    const mid = Math.floor((ok + over) / 2);
    if (fits(chars.slice(0, mid).join(""))) ok = mid;
    else over = mid;
  }
  return chars.slice(0, ok).join("");
}

/**
 * @internal The prefilled new-issue link for `title` and `body`, at most `LINK_CHARS` long: a
 * title past `LINK_TITLE_CHARS` encoded is cut and ends in `…`, and a body that doesn't fit is
 * cut, each between characters; either way the body then ends with a note that the whole report
 * is in `saved`. `cut` says whether anything was.
 */
export function issueLink(
  title: string,
  body: string,
  saved: string,
): { url: string; cut: boolean } {
  const link = (heading: string, text: string) =>
    `https://${HOST}/${REPO}/issues/new?title=${encodeURIComponent(heading)}` +
    `&body=${encodeURIComponent(text)}`;
  const shortTitle = longestFitting(
    title,
    (prefix) => encodeURIComponent(`${prefix}…`).length <= LINK_TITLE_CHARS,
  );
  const shown = shortTitle === title ? title : `${shortTitle}…`;
  const whole = link(shown, body);
  if (shown === title && whole.length <= LINK_CHARS) return { url: whole, cut: false };
  const note = `\n\n(cut to fit a link; the whole report is in ${saved})`;
  const kept = longestFitting(body, (prefix) => link(shown, prefix + note).length <= LINK_CHARS);
  return { url: link(shown, kept + note), cut: true };
}

/**
 * Whether `gh` is installed and logged in to github.com, by `gh auth status`, which may go to the
 * network: given up on after `GH_CHECK_MS`.
 */
async function ghReady(run: Runner, signal: AbortSignal): Promise<boolean> {
  const bounded = AbortSignal.any([signal, AbortSignal.timeout(GH_CHECK_MS)]);
  try {
    const args = ["auth", "status", "--hostname", HOST, "--active"];
    return (await run("gh", args, { signal: bounded })).code === 0;
  } catch {
    return false;
  }
}

/** The last line of `text` that isn't blank, or `fallback`. */
function lastLine(text: string, fallback: string): string {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).at(-1) ?? fallback;
}

/**
 * Save the report under `~/.yian/reports` (the folder private to the user), named by `now` to the
 * second and numbered after the first of that second; return its path shown with `~`, or why it
 * couldn't be saved.
 */
function saveReport(
  env: Env,
  now: number,
  title: string,
  body: string,
): { saved: string } | { failed: string } {
  const home = homeDir(env);
  const dir = join(home, ".yian", "reports");
  // `2026-10-29T14-03-05Z`: no `:`, which some file systems refuse.
  const stamp = new Date(now).toISOString().replace(/\.\d+Z$/, "Z").replaceAll(":", "-");
  try {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    // `mode` sets only a folder mkdir makes, and the umask may loosen even that.
    chmodSync(dir, 0o700);
    for (let n = 1; n <= SAME_SECOND_REPORTS; n++) {
      const path = join(dir, `${stamp}${n === 1 ? "" : `-${n}`}.md`);
      try {
        writeFileSync(path, `# ${title}\n\n${body}\n`, { flag: "wx", mode: 0o600 });
        return { saved: homePath(path, home) };
      } catch (error) {
        if (!isSystemError(error) || error.code !== "EEXIST") throw error;
      }
    }
    return { failed: `${SAME_SECOND_REPORTS} reports already saved this second` };
  } catch (error) {
    if (!isSystemError(error)) throw error;
    return { failed: osReason(error) };
  }
}

/** File the issue with gh; the outcome's line. */
async function fileWithGh(
  run: Runner,
  title: string,
  body: string,
  signal: AbortSignal,
): Promise<string> {
  // `--title=`, so a title starting with `-` can't read as a flag.
  const args = ["issue", "create", "-R", `${HOST}/${REPO}`, `--title=${title}`, "--body-file", "-"];
  try {
    const result = await run("gh", args, { input: body, signal });
    if (result.code === 0) {
      return `filed with gh: ${lastLine(result.stdout, "(gh printed no link)")}`;
    }
    return `not filed: gh failed: ${lastLine(result.stderr, `exit code ${result.code}`)}`;
  } catch (error) {
    return `not filed: gh failed: ${(error as Error).message}`;
  }
}

/** File the issue with one GitHub API call authorized by `token`; the outcome's line. */
async function fileWithToken(
  token: string,
  send: Send | undefined,
  title: string,
  body: string,
  signal: AbortSignal,
): Promise<string> {
  const init: RequestInit = {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      accept: "application/vnd.github+json",
      "content-type": "application/json",
      "x-github-api-version": "2022-11-28",
    },
    body: JSON.stringify({ title, body }),
    signal,
  };
  let response: Response;
  try {
    response = await safeFetch(`https://api.${HOST}/repos/${REPO}/issues`, init, send);
  } catch (error) {
    // safeFetch's own errors never echo a header, so the token can't be in the message.
    return `not filed: can't reach GitHub: ${(error as Error).message}`;
  }
  const answer = (await response.json().catch(() => ({}))) as {
    html_url?: unknown;
    message?: unknown;
  };
  if (response.status === 201 && typeof answer.html_url === "string") {
    return `filed with GITHUB_TOKEN: ${answer.html_url}`;
  }
  const message = typeof answer.message === "string" ? `: ${answer.message}` : "";
  return `not filed: GitHub answered ${response.status}${message}`;
}

/**
 * Show the report on an approval card and file it the way the user picks, as `fileIssueTool`
 * describes; return how it was filed and where it was saved, for every outcome.
 */
async function fileIssue(
  host: FileIssueHost,
  title: string,
  body: string,
  signal: AbortSignal,
): Promise<string> {
  const unapproved =
    "not filed: not approved (no terminal, or the prompt ended); nothing was saved";
  // Nobody to ask: gh's check, which may go to the network, isn't worth making.
  if (!host.canApprove()) return unapproved;
  const run = host.run ?? runCommand;
  const token = host.env("GITHUB_TOKEN") || null;
  const choices = [
    ...((await ghReady(run, signal)) ? [GH] : []),
    ...(token === null ? [] : [TOKEN]),
    LINK,
    SAVE,
    CANCEL,
  ];
  const answer = await host.approve({
    title: `Report this issue to ${HOST}/${REPO}?`,
    body:
      `Title: ${title}\n\n${body}\n\n` +
      `Every choice but ${CANCEL} also saves it in ~/.yian/reports.`,
    diff: null,
    choices,
    // A stray Enter only saves: sending it anywhere takes a choice of its own.
    preselect: SAVE,
    signal,
  });
  if (answer === null) return unapproved;
  if (answer === CANCEL) return "not filed: cancelled; nothing was saved";
  const save = saveReport(host.env, host.now(), title, body);
  const where = "saved" in save ? `saved to ${save.saved}` : `not saved: ${save.failed}`;
  let outcome: string;
  if (answer === GH) outcome = await fileWithGh(run, title, body, signal);
  else if (answer === TOKEN) outcome = await fileWithToken(token!, host.send, title, body, signal);
  else if (answer === LINK) {
    const { url } = issueLink(title, body, "saved" in save ? save.saved : "the chat");
    outcome = `not filed yet: open this link to file it: ${url}`;
  } else outcome = "not filed: saved only";
  return `${outcome}\n${where}`;
}

/**
 * The `file_issue` tool over `host`: it shows the user the whole report on an approval card, then
 * files it the way they pick among those available: gh (when installed and logged in), the
 * environment's `GITHUB_TOKEN` (when set), a prefilled browser link (cut to fit, saying so),
 * saving it only (preselected), or Cancel. Every way but Cancel also saves it to
 * `~/.yian/reports/<time>.md`. Without a terminal to ask on, it asks nothing and files nothing.
 * Returns how it was filed and where it was saved.
 */
export function fileIssueTool(host: FileIssueHost): FunctionTool {
  return tool({
    name: "file_issue",
    description:
      `Show the user an issue report for ${HOST}/${REPO} to approve, then file it the way they ` +
      "pick (gh, GITHUB_TOKEN, a browser link) or save it only; returns how it was filed and " +
      "where it was saved.",
    parameters: stringArguments(
      ["title", "The issue's title.", true],
      ["body", "The issue's body, in Markdown.", true],
    ),
    execute: (args, ctx) =>
      fileIssue(host, stringArg(args, "title"), stringArg(args, "body"), ctx.signal),
  });
}
