/**
 * The flow commands, `yian run`, `yian check`, `yian status` and `yian skills check`, and `yian
 * run`'s helpers: what it prints on stderr, the inputs it reads from flags and its run-log options.
 * The start-up it shares with `yian chat` is in `startup.ts`, and the rendering both use in
 * `render.ts`.
 *
 * - `run` checks the run-log options, loads the flow, reads skills and MCP servers when it needs
 *   them, runs it on a fresh log or recovers the run recorded on a
 *   `--resume` log, checking its recorded keys, and prints its output or its failure; `yian run`
 *   reaches it through `runOrChat` in `inbox_chat.ts`, which hands a terminal chat to `inboxChat`;
 *   `check` loads each flow, or confined to `--root`, and reports each as text or one `--json`
 *   document; `status` reports where the run in a run log stands, or, given no log, updates and
 *   lists the run index of `~/.yian/runs`, as text or a `--json` document. A run whose log is in
 *   `~/.yian/runs` updates that index when it ends.
 * - `skillsCheck` loads skill directories and reports each as text or one `--json` document.
 * - `renderRunError` renders a failed run's diagnostic.
 * - `flowBanner` is the "what am I running" box; `ProgressReporter` draws per-node progress.
 * - `mergeInputs` reads `--inputs` then `--input k=v` (`readInputsJson`, `parseKv`).
 * - `validateRunLogOptions` checks the run-log options; `reportRunLog` prints the log's path.
 * - `checkSuppliedInputs` and `indexRun` are the steps of a fresh run's start and end that `yian
 *   chat` shares beside `startup.ts`'s.
 * - `readGrantedSkills` reads the skill folders when a flow's AGENTs grant skills.
 *
 * Output is raw ANSI on stderr, coloured only on a TTY without `NO_COLOR`.
 */

import { existsSync, lstatSync, realpathSync, rmdirSync } from "node:fs";
import { basename, dirname, join, relative } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { END_ID, START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { loadFlow } from "../compose/loader.ts";
import { defaultSkillRoots, loadSkillDir, loadSkills } from "../compose/skills.ts";
import { current, type Env } from "../defaults.ts";
import { compareCodeUnits, Diagnostic, reprText, shellQuote } from "../diagnostics.ts";
import type { RunEvent } from "../events.ts";
import { type Skill, SkillError } from "../skills/catalog.ts";
import { LOAD_SKILL } from "../skills/tool.ts";
import { JSONDecodeError, type JsonValue, parseJson } from "../typesys/json.ts";
import {
  canonicalTypeLabel,
  FileRef,
  isInboxType,
  isMemoryType,
  isRecord,
  type Value,
  writeAsText,
} from "../typesys/values.ts";
import { apiKeyEnvNames, OPTIONAL_API_KEY_PROVIDERS } from "../llm_clients/api_key_env.ts";
import { AuthError } from "../llm_clients/auth/index.ts";
import { defaultEmbed } from "../llm_clients/factory.ts";
import { StorageError } from "../llm_clients/auth/store.ts";
import {
  type AgentConfig,
  type Connection,
  connectionCredentials,
  defaultEmbedding,
  HEADER_NAMES_KEY,
  HOST_KEY_MARKER,
  keyIdentity,
} from "../llm_clients/connection.ts";
import { checkInputNames } from "../nodes/binding.ts";
import type { Parameter } from "../nodes/contracts.ts";
import { FileLockError, homeDir, isSystemError, sqliteErrorCode, statOrNull, withWarnings } from "../platform.ts";
import {
  advance,
  inspectRun,
  type LlmConfigsCheck,
  loadRecordedFlow,
  readRunIndex,
  type RunIndexRow,
  type RunInspection,
  type RunResult,
  runFlow,
  skillGrants,
  staticNodes,
  updateRunIndex,
} from "../run.ts";
import { wakeAt } from "../runtime/waits.ts";
import { ASK_USER, BUILTIN_TOOL_IDS } from "../tools/context.ts";
import { McpConnectError, McpSdkMissingError, type McpServer } from "../tools/mcp.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
import { loadConfined } from "./confined_load.ts";
import { loadMcpConfig, McpConfigError, userMcpConfig } from "./mcp_config.ts";
import {
  accountHome,
  bindMemories,
  type CreatedMemory,
  createdLine,
  driftNote,
  embeddingFlag,
  projectFolder,
  resumeMemories,
} from "./memory_bind.ts";
import {
  box,
  diagnosticJson,
  elapsedTag,
  endedNodeText,
  failedNodeText,
  paint,
  printStyled,
  renderDiagnostic,
  splitLines,
  type Style,
  tracebackBox,
  wrapSpans,
  writeEngineTrace,
} from "./render.ts";
import {
  cliLayer,
  defaultRunsDir,
  ensureProviderKeys,
  freshLog,
  loadFlowFile,
  NotUtf8Error,
  preflightKeys,
  readFlowText,
} from "./startup.ts";
import { cells, columnsOf } from "./cells.ts";
import { Terminal, TerminalStopped } from "./terminal.ts";
import {
  atTerminal,
  type CliIo,
  CommandEnv,
  forTerminal,
  HostStop,
  osReason,
  printJson,
  type PromptContext,
  stopOnSignals,
  UsageError,
  type Writer,
} from "./utils.ts";

/**
 * Write a failed run's diagnostic to `out` and return it: the flow's `diagnose` of its failure,
 * or an unlocated `(no detail)` when it has none. With `engineTrace`, the traceback the diagnostic
 * captured follows it, boxed, when it holds one.
 */
export async function renderRunError(
  result: RunResult,
  flow: ComposedFlow,
  engineTrace: boolean,
  out: Writer,
  env: Env,
): Promise<Diagnostic> {
  const diagnostic =
    result.failure === null
      ? new Diagnostic({ code: "node.execute", message: "(no detail)" })
      : flow.diagnose(result.failure);
  await renderDiagnostic(diagnostic, out, env);
  if (engineTrace && diagnostic.traceback) {
    await out.write(tracebackBox(diagnostic.traceback, out, env));
  }
  return diagnostic;
}

/**
 * Return a run output as JSON data: each file, at any depth, becomes `{uri, mime, name}`.
 *
 * Records shaped like yian's infinity tag or `$record` escape are left as they are: `printJson`
 * escapes them as it writes.
 */
export function jsonValue(value: unknown): unknown {
  if (FileRef.isFileRef(value)) return { uri: value.uri, mime: value.mime, name: value.name };
  if (Array.isArray(value)) return value.map(jsonValue);
  if (isRecord(value)) {
    const record: Record<string, unknown> = Object.create(null);
    for (const [key, item] of Object.entries(value)) record[key] = jsonValue(item);
    return record;
  }
  return value;
}

/**
 * Return a line's `spans` with each tab replaced by spaces up to the next 8-cell stop, counted
 * from the line's start across the spans, as rich's `expand_tabs` does.
 */
function expandTabs(spans: readonly [string, Style | null][]): [string, Style | null][] {
  let column = 0;
  return spans.map(([text, style]) => {
    let expanded = "";
    for (const char of text) {
      const piece = char === "\t" ? " ".repeat(8 - (column % 8)) : char;
      column += cells(piece);
      expanded += piece;
    }
    return [expanded, style];
  });
}

/**
 * Return the box printed when a run starts: `Running flow: <name>  (version: v)` and
 * `Description: …`, wrapped inside a box no wider than the terminal's columns (80 when unknown),
 * ending in a newline; `null` when the flow has no name, description or version.
 */
export function flowBanner(
  flow: { name: string | null; description: string | null; version: string | null },
  out: Writer,
  env: Env,
): string | null {
  const { name, description, version } = flow;
  if (!(name || description || version)) return null;
  // Each logical line is a list of [text, style] spans, wrapped as one plain text.
  const head: [string, Style | null][] = [["Running flow: ", "bold"], [name || "(unnamed)", null]];
  if (version) head.push([`  (version: ${version})`, "dim"]);
  const lines = [head];
  if (description) {
    const label = "Description: ";
    splitLines(label + description).forEach((line, i) => {
      lines.push(i === 0 ? [[label, "bold"], [line.slice(label.length), null]] : [[line, null]]);
    });
  }
  // Rich measures the box before it expands tabs, counting a tab as no cells.
  const widest = Math.max(...lines.map((spans) => cells(spans.map(([text]) => text.replaceAll("\t", "")).join(""))));
  const inner = Math.max(1, Math.min(widest, columnsOf(out) - 4));
  const rows = lines.flatMap((spans) => wrapSpans(expandTabs(spans), inner, out, env));
  return `${box(rows, "cyan", out, env, "", inner)}\n`;
}

/** Return the final namespace segment of `nodeId`, which marks a boundary node at any depth. */
function lastSegment(nodeId: string | null): string | null {
  return nodeId === null ? null : nodeId.slice(nodeId.lastIndexOf("/") + 1);
}

const SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
const SPIN_MS = 80;

/**
 * Per-node progress for `yian run`, drawn on stderr as the engine streams node events.
 *
 * A running node shows a spinner line, then a green `✓ <node>` on success (`✓ <node> → <handle>`
 * for a router) or a red `✗ <node>` with the error on the next line on failure. Each line ends
 * with the time since that node started (`elapsedTag`, on the `now` clock), counting up on a
 * spinner and frozen on the finished line; a node whose start wasn't seen has none. With
 * `verbose` each node's output follows its check. Several nodes can run at once (a fan-out), so
 * there is one spinner line per running node. On a TTY the spinners are redrawn every 80 ms and
 * each finished line is printed over them; off a TTY only the finished lines are printed. Outputs,
 * failure messages and warnings go through `forTerminal`.
 */
export class ProgressReporter {
  readonly #out: Writer;
  readonly #verbose: boolean;
  readonly #env: Env;
  readonly #now: () => number;
  // Each running node's start time on `#now`.
  readonly #running = new Map<string, number>();
  // Warning lines raised while paused, printed on `resume` or `stop`.
  readonly #held: string[] = [];
  #paused = false;
  #frame = 0;
  // Spinner lines on screen, which the next draw or finished line erases.
  #drawn = 0;
  #timer: ReturnType<typeof setInterval> | null = null;

  constructor(out: Writer, options: { verbose: boolean; env: Env; now?: () => number }) {
    this.#out = out;
    this.#verbose = options.verbose;
    this.#env = options.env;
    // A monotonic clock, so a wall-clock change mid-run doesn't skew a duration.
    this.#now = options.now ?? (() => performance.now());
  }

  /** Fold one engine event into the display; boundary nodes and other events are ignored. */
  handle(event: RunEvent): void {
    const nodeId =
      event.type === "NodeFailed" ? event.failure.nodeId : "nodeId" in event ? event.nodeId : null;
    // Synthesized boundary nodes carry no authored meaning, top-level or namespaced
    // (`polish#0/__start__`).
    const last = lastSegment(nodeId);
    if (last === START_ID || last === END_ID) return;
    const style = (text: string, s: Style) => paint(text, s, this.#out, this.#env);
    switch (event.type) {
      case "NodeStarted":
        this.#running.set(event.nodeId, this.#now());
        break;
      case "NodeSucceeded": {
        const lines = [style(endedNodeText(event.nodeId, this.#finish(event.nodeId)), "green")];
        if (this.#verbose) {
          const bodyLines = splitLines(forTerminal(writeAsText(event.output as Value), this.#out));
          for (const line of bodyLines.length > 0 ? bodyLines : [""]) lines.push(style(`    ${line}`, "dim"));
        }
        this.#print(lines.join("\n"));
        break;
      }
      case "NodeRouted": {
        // A router (CASE) produces no value: it selected an out-edge handle.
        const took = this.#finish(event.nodeId);
        this.#print(style(endedNodeText(event.nodeId, took, event.handle), "green"));
        break;
      }
      case "NodeExpanded":
        // A spawner grew the graph instead of returning a value; the spliced fragment's terminal
        // commits it, so it gets no line, only its spinner cleared.
        this.#running.delete(event.nodeId);
        break;
      case "NodeFailed": {
        const took = nodeId === null ? "" : this.#finish(nodeId);
        const message = forTerminal(event.failure.message, this.#out);
        const head = style(failedNodeText(String(nodeId), took), "red bold");
        this.#print(`${head}\n${style(`    ${message}`, "red")}`);
        break;
      }
      default:
        return;
    }
    this.#sync();
  }

  /**
   * Print `warning: <message>` on its own line above the spinners; while paused it is held until
   * `resume` or `stop`, so it can't land inside the prompt that has the terminal.
   */
  warn(message: string): void {
    const line = paint(`warning: ${forTerminal(message, this.#out)}`, "yellow", this.#out, this.#env);
    if (this.#paused) {
      this.#held.push(line);
      return;
    }
    this.#print(line);
    this.#sync();
  }

  /** Stop drawing spinners, erasing them, while a prompt has the terminal. */
  pause(): void {
    this.#paused = true;
    this.#sync();
  }

  /** Draw the spinners again after `pause`, after the warning lines held meanwhile. */
  resume(): void {
    this.#paused = false;
    this.#flushHeld();
    this.#sync();
  }

  /** End the display: print held warnings, forget the running nodes and erase their spinners. */
  stop(): void {
    this.#flushHeld();
    this.#running.clear();
    this.#sync();
  }

  /** Forget `nodeId` as running; return ` <elapsed>` since its start, or "" when none was seen. */
  #finish(nodeId: string): string {
    const start = this.#running.get(nodeId);
    this.#running.delete(nodeId);
    return start === undefined ? "" : ` ${elapsedTag(this.#now() - start)}`;
  }

  /** Print the warning lines held while paused. */
  #flushHeld(): void {
    for (const line of this.#held.splice(0)) this.#print(line);
  }

  #write(text: string): void {
    // Progress is advisory: a failed stderr write must not end the run as an unhandled rejection.
    this.#out.write(text).catch(() => {});
  }

  /** Return the text that erases the spinner lines on screen and leaves the cursor at column 0. */
  #erase(): string {
    return `\r\x1b[K${"\x1b[A\x1b[K".repeat(Math.max(0, this.#drawn - 1))}`;
  }

  /** Print a finished line over the spinners; the next `#sync` draws them again below it. */
  #print(text: string): void {
    const erase = this.#drawn > 0 ? this.#erase() : "";
    this.#drawn = 0;
    this.#write(`${erase}${text}\n`);
  }

  /** Draw one spinner line per running node in place of those on screen. */
  #draw(): void {
    const columns = this.#out.columns;
    const frame = SPINNER.charAt(this.#frame);
    const now = this.#now();
    const lines = [...this.#running].map(([nodeId, start]) => {
      const chars = [...nodeId];
      const tag = ` ${elapsedTag(now - start)}`;
      // A line wider than the terminal would wrap, and the erase would miss its first part: the
      // elapsed time goes whole, then the node id is cut.
      const width = columns === undefined ? Number.POSITIVE_INFINITY : Math.max(0, columns - 3);
      const fits = chars.length + tag.length <= width;
      const id = paint(chars.slice(0, width).join(""), "cyan", this.#out, this.#env);
      return `${frame} ${id}${fits ? tag : ""}`;
    });
    this.#write(this.#erase() + lines.join("\n"));
    this.#drawn = lines.length;
  }

  /** Start or stop the spinners to match the TTY, the pause and the running nodes. */
  #sync(): void {
    if (this.#out.isTTY && !this.#paused && this.#running.size > 0) {
      this.#draw();
      if (this.#timer === null) {
        this.#timer = setInterval(() => {
          this.#frame = (this.#frame + 1) % SPINNER.length;
          this.#draw();
        }, SPIN_MS);
        this.#timer.unref();
      }
      return;
    }
    if (this.#timer !== null) {
      clearInterval(this.#timer);
      this.#timer = null;
    }
    if (this.#drawn > 0) {
      this.#write(this.#erase());
      this.#drawn = 0;
    }
  }
}

/**
 * Return the `--input k=v` flags as a record of names to values; values stay text, which the
 * run coerces to each input's declared type. The key is stripped; the value is everything after
 * the first `=`.
 *
 * @throws {UsageError} For a flag without `=`.
 */
export function parseKv(pairs: readonly string[]): Record<string, string> {
  // A null prototype, so an input named `__proto__` is an ordinary key.
  const out: Record<string, string> = Object.create(null);
  for (const pair of pairs) {
    const at = pair.indexOf("=");
    if (at === -1) throw new UsageError(`Invalid value: --input must be k=v (got ${reprText(pair)})`);
    out[pair.slice(0, at).trim()] = pair.slice(at + 1);
  }
  return out;
}

/** Return the JSON name of a top-level value that isn't an object. */
function jsonTypeName(value: JsonValue): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "bigint" || typeof value === "number") return "number";
  return typeof value;
}

/**
 * Read the `--inputs` file: UTF-8 text holding one JSON object of input names to values. An int
 * reads as a `bigint`.
 *
 * @throws {UsageError} When the file is not UTF-8 text, is not valid JSON, or holds something
 *   other than an object.
 */
export function readInputsJson(path: string): Record<string, JsonValue> {
  // The BOM is kept, so the JSON reader refuses it. Universal newlines change only an error's line
  // and column: JSON's whitespace includes them all.
  let text: string;
  try {
    text = readFlowText(path);
  } catch (error) {
    if (error instanceof NotUtf8Error) throw new UsageError("Invalid value for --inputs: is not UTF-8 text");
    throw error;
  }
  let parsed: JsonValue;
  try {
    parsed = parseJson(text, "plain");
  } catch (error) {
    if (!(error instanceof JSONDecodeError)) throw error;
    throw new UsageError(
      `Invalid value for --inputs: invalid JSON at line ${error.line}, column ${error.column}: ${error.reason}`,
    );
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new UsageError(`Invalid value for --inputs: must contain a JSON object, got ${jsonTypeName(parsed)}`);
  }
  return parsed;
}

/**
 * Return a fresh run's supplied inputs: the `--inputs` file's, if given, then each `--input k=v`,
 * a later flag overriding an earlier value.
 *
 * @throws {UsageError} As `readInputsJson` and `parseKv` do.
 */
export function mergeInputs(inputs: string | undefined, inputValues: readonly string[]): Record<string, JsonValue> {
  const fromFile = inputs === undefined ? {} : readInputsJson(inputs);
  return Object.assign(Object.create(null), fromFile, parseKv(inputValues));
}

/** @internal Click's hint for an error in the inputs `--input` and `--inputs` supply together. */
export const INPUTS_HINT = "Invalid value for '--input' / '--inputs'";

/**
 * @internal Check that each name in `supplied` is one of `decls`' inputs that take a value at the
 * start; an inbox input takes events through `send`, and a memory input is bound with `--memory`,
 * so a value for either is refused too.
 *
 * @throws {UsageError} For a name of an inbox or memory input, or one no input has.
 */
export function checkSuppliedInputs(decls: readonly Parameter[], supplied: Readonly<Record<string, unknown>>): void {
  // An inbox takes events through `send`, never a value at the start.
  const isInbox = (decl: Parameter) => isInboxType(decl.contract?.type ?? null);
  const inbox = decls.filter((decl) => isInbox(decl) && Object.hasOwn(supplied, decl.name)).map((decl) => decl.name);
  if (inbox.length > 0) {
    const name = JSON.stringify(inbox.sort(compareCodeUnits)[0]);
    throw new UsageError(`${INPUTS_HINT}: input ${name} is an inbox; it takes messages through send, not a value`);
  }
  // A memory is bound to an instance, never given as a value.
  const memory = decls.find((decl) => isMemoryType(decl.contract?.type) && Object.hasOwn(supplied, decl.name));
  if (memory !== undefined) {
    throw new UsageError(`${INPUTS_HINT}: \`${memory.name}\` is a memory; bind it with --memory`);
  }
  const values = decls.filter((decl) => !isInbox(decl));
  try {
    checkInputNames(values, supplied);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new UsageError(`${INPUTS_HINT}: ${error.message}`);
  }
}

/**
 * @internal Return the inputs to add to `supplied`: each declaration it lacks (key presence)
 * answered at `terminal`'s prompts, or `{}` without a terminal, leaving optional and defaulted
 * inputs absent for START. An inbox or memory input is never asked for. Returns null when a prompt
 * is cancelled.
 *
 * @throws {UsageError} Without a terminal, when required inputs are missing, naming them all in
 *   declaration order.
 */
export async function gatherMissingInputs(
  decls: readonly Parameter[],
  supplied: Readonly<Record<string, unknown>>,
  terminal: Terminal | null,
): Promise<Record<string, unknown> | null> {
  // An inbox takes events through `send`, and a memory is bound, never a value given at the start.
  const values = decls.filter(
    (decl) => !isInboxType(decl.contract?.type ?? null) && !isMemoryType(decl.contract?.type),
  );
  if (terminal !== null) return terminal.promptInputs(values, supplied);
  const required = values.filter((decl) => decl.required && !Object.hasOwn(supplied, decl.name));
  if (required.length > 0) {
    const noun = required.length === 1 ? "input" : "inputs";
    const names = required.map((decl) => decl.name).join(", ");
    throw new UsageError(`${INPUTS_HINT}: missing required flow ${noun}: ${names}`);
  }
  return {};
}

/** Whether a directory entry exists at `path`, a dangling symlink included. */
function lexists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * The id and file name of `yian chat`'s bundled flow: `yian run --resume` refuses a log whose
 * recorded flow has both, as a chat of any yian version records it. A flow of your own resumes
 * unless it has both too.
 */
export const CHAT_FLOW_ID = "yian-chat";
export const CHAT_FLOW_FILE = "chat.yaml";

/**
 * Reject a flow file or fresh-only options on `--resume`, a fresh run without a flow file, and a
 * `--log` path that already exists. Absent options are `undefined`; `inputValues` is empty when
 * no `--input` is given.
 *
 * Run before input files are read, prompts are shown, or provider configuration is resolved.
 *
 * @throws {UsageError} When `resume` is given with `flow` (the resume runs the flow the log
 *   recorded), or with `--input`, `--inputs`, `--provider`, `--model`, `--embedding` or `--log`
 *   (each one given is named, in that order); when neither `flow` nor `resume` is given; or when a
 *   directory entry, even a dangling symlink, exists at `log`.
 */
export function validateRunLogOptions(options: {
  flow?: string | undefined;
  inputValues: readonly string[];
  inputs?: string | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  embedding?: string | undefined;
  log?: string | undefined;
  resume?: string | undefined;
}): void {
  const { flow, inputValues, inputs, provider, model, embedding, log, resume } = options;
  if (resume === undefined && flow === undefined) throw new UsageError("Missing argument 'FLOW'.");
  if (resume !== undefined) {
    // An edited flow file means a new run, so a resume never reads one.
    if (flow !== undefined) {
      throw new UsageError("Invalid value: --resume runs the flow recorded in the log; leave out <flow>");
    }
    const conflicts: string[] = [];
    if (inputValues.length > 0) conflicts.push("--input");
    if (inputs !== undefined) conflicts.push("--inputs");
    if (provider !== undefined) conflicts.push("--provider");
    if (model !== undefined) conflicts.push("--model");
    if (embedding !== undefined) conflicts.push("--embedding");
    if (log !== undefined) conflicts.push("--log");
    if (conflicts.length > 0) {
      throw new UsageError(`Invalid value: --resume cannot be combined with ${conflicts.join(", ")}`);
    }
    return;
  }
  if (log !== undefined && lexists(log)) {
    throw new UsageError(`Invalid value for --log: --log path already exists: ${log}`);
  }
}

/** Remove the directory `dir` if it is empty; otherwise, or when that fails, leave it as it is. */
export function removeEmptyDir(dir: string): void {
  try {
    rmdirSync(dir);
  } catch {
    // Not empty (the run created its log) or not removable: tidying must not replace the
    // command's own outcome.
  }
}

/**
 * @internal Update the run index of `~/.yian/runs` (`updateRunIndex`, waiting at most a second for
 * another update) after the run in `log` ended, when `log` is a run folder's log there; a failed
 * update is a `warning:` line on stderr, and the logs it skips aren't reported (`yian status`
 * reports them).
 */
export async function indexRun(log: string, io: CliIo): Promise<void> {
  const runsDir = defaultRunsDir(io.env);
  // Real paths, so a log reached through a symlinked HOME still counts; a missing folder can't.
  const real = (path: string) => {
    try {
      return realpathSync(path);
    } catch {
      return null;
    }
  };
  const folder = real(dirname(dirname(log)));
  if (folder === null || folder !== real(runsDir)) return;
  try {
    withWarnings(() => {}, () => updateRunIndex(runsDir, { timeoutMs: 1000 }));
  } catch (error) {
    if (!(error instanceof Error)) throw error;
    await printStyled(io, `warning: run index not updated: ${error.message}`, "yellow");
  }
}

/** Write the run's log path to `out`, even when progress is quiet. */
export async function reportRunLog(path: string, out: Writer, env: Env): Promise<void> {
  await out.write(`${paint(`Run log: ${path}`, "dim", out, env)}\n`);
}

/**
 * The options `run` reads; an absent option is `undefined`, an absent repeatable one an empty list,
 * an absent flag `false`.
 */
export interface RunOptions {
  inputValues: readonly string[];
  inputs?: string | undefined;
  log?: string | undefined;
  resume?: string | undefined;
  quiet: boolean;
  verbose: boolean;
  numWorkers?: number | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  engineTrace: boolean;
  mcpConfig: readonly string[];
  /** The `--memory NAME=PATH` flags. */
  memory: readonly string[];
  /** The `--embedding provider/model` a new VectorDB memory is made with. */
  embedding?: string | undefined;
  json: boolean;
}

/**
 * @internal Whether `error` is a host failure `run` reports as `run failed:` (`resume failed:`) rather than
 * with its stack: a bad value (a damaged or empty log or a defaults mismatch included), an
 * unreadable file, a run log another writer holds, an MCP server that didn't start, the missing MCP
 * SDK.
 */
export function isSetupError(error: unknown): error is Error {
  return (
    error instanceof TypeError ||
    error instanceof RangeError ||
    error instanceof AuthError ||
    error instanceof McpConnectError ||
    // A lock file that is unsafe, not private or changed; a locked log is a plain `Error` over
    // the lock's.
    error instanceof FileLockError ||
    (error instanceof Error && error.cause instanceof FileLockError) ||
    error instanceof McpSdkMissingError ||
    isSystemError(error)
  );
}

/**
 * The lines `openMcpServers` adds below a server failure it throws: the ids no started server
 * supplies and the other servers that failed. Kept apart from the error's `notes`, which a cleanup
 * failure also writes to.
 */
const selectionNotes = new WeakMap<Error, readonly string[]>();

/**
 * Return the bare tool ids `flow`'s static TOOL nodes call, which name single tools, and the names
 * its static nodes grant, which name tools or tool groups: an AGENT's grants (a group grant's
 * group) and the group of a TOOL node's `{group: tool}`.
 */
function mcpNeededNames(flow: ComposedFlow): [Set<string>, Set<string>] {
  const toolIds = new Set<string>();
  const grants = new Set<string>();
  for (const [, node] of staticNodes(flow.compiled)) {
    const call = node.hostToolCall();
    // A group names its server, as an AGENT's group grant does.
    if (call?.group != null) grants.add(call.group);
    else if (call !== null) toolIds.add(call.toolId);
    for (const grant of node.grantedTools()) {
      grants.add(grant.kind === "group" ? grant.group : grant.name);
    }
  }
  return [toolIds, grants];
}

/** Return the tool ids a run supplies without MCP: the registered, built-in and skill tools. */
function knownToolIds(): Set<string> {
  return new Set([...Object.keys(registeredTools()), ...BUILTIN_TOOL_IDS, ASK_USER, LOAD_SKILL]);
}

/** Whether some id in `toolIds` or `grants` isn't in `known`. */
function namesUnknown(toolIds: ReadonlySet<string>, grants: ReadonlySet<string>, known: ReadonlySet<string>): boolean {
  return [...toolIds, ...grants].some((id) => !known.has(id));
}

/**
 * @internal Whether a static TOOL node or AGENT grant of `flow` names a tool the run doesn't
 * supply without MCP, so `run` reads the MCP configs.
 */
export function needsMcp(flow: ComposedFlow): boolean {
  const [toolIds, grants] = mcpNeededNames(flow);
  return namesUnknown(toolIds, grants, knownToolIds());
}

/**
 * @internal Run `body` with the configured MCP servers `flow` needs, and close every server in
 * `servers` when it settles.
 *
 * A server a flow grants by name (an AGENT's tool group, or the group of a TOOL node's
 * `{group: tool}`) is selected unopened, so the run starts it after its own checks and closes it.
 * A TOOL node's bare id names one tool, never a group, so it counts as unresolved unless it is in
 * `known`; so does a grant that is neither in `known` nor a server name. When some id is
 * unresolved, the granted servers open here instead, and if their tools don't supply every
 * unresolved id, the other servers open in turn to show their tools until every id is supplied;
 * those that list a still-unresolved id stay open and the rest close before the run, so an unused
 * server's tool names can't collide with another's. A server opened only to show its tools that
 * fails to start is skipped with a `warn` call, unless the flow still lacks a needed id
 * afterwards. Once `signal` aborts, no further server opens, and only the servers already open are
 * selected. Nothing is selected when the flow needs no tools.
 *
 * @param servers The configured servers by name, none of them open.
 * @param toolIds The bare tool ids the flow's static TOOL nodes call.
 * @param grants The tool and tool-group names the flow's static nodes grant.
 * @param known The tool ids the run supplies without MCP.
 * @param options.signal The run's stop.
 * @param options.warn Called with `<error>; skipped, the flow doesn't need it` for each unneeded
 *   server that failed to start.
 * @param body Called with the selected servers by name, as tool groups for the run.
 * @returns What `body` resolves to.
 * @throws {McpConnectError} A server the flow names doesn't start or lists bad tools; or a needed
 *   id is still missing and some server failed to start: the first failure, with lines naming the
 *   missing ids and each other failure that `run` prints below it.
 * @throws {Error} A server must open here and the MCP SDK is not installed.
 * @throws The signal's reason, when it aborts while a server opens.
 */
export async function openMcpServers<T>(
  servers: Readonly<Record<string, McpServer>>,
  toolIds: ReadonlySet<string>,
  grants: ReadonlySet<string>,
  known: ReadonlySet<string>,
  { signal, warn }: { signal: AbortSignal; warn: (message: string) => void },
  body: (selected: Record<string, McpServer>) => Promise<T>,
): Promise<T> {
  const toolNames = (selected: Record<string, McpServer>) =>
    new Set(Object.values(selected).flatMap((server) => server.tools.map((tool) => tool.name)));
  try {
    let selected: Record<string, McpServer> = {};
    const named = Object.entries(servers).filter(([name]) => grants.has(name));
    const unresolved = new Set([
      ...[...toolIds].filter((id) => !known.has(id)),
      ...[...grants].filter((id) => !known.has(id) && !Object.hasOwn(servers, id)),
    ]);
    if (unresolved.size === 0) {
      selected = Object.fromEntries(named);
    } else {
      for (const [name, server] of named) {
        if (signal.aborted) break;
        selected[name] = await server.open({ signal });
      }
      // No other server needs to start for an id a granted server supplies.
      for (const name of toolNames(selected)) unresolved.delete(name);
      const failed: McpConnectError[] = [];
      for (const [name, server] of Object.entries(servers)) {
        if (unresolved.size === 0 || Object.hasOwn(selected, name) || signal.aborted) continue;
        try {
          await server.open({ signal });
        } catch (error) {
          if (!(error instanceof McpConnectError)) throw error;
          failed.push(error);
          continue;
        }
        if (server.tools.some((tool) => unresolved.has(tool.name))) {
          selected[name] = server;
          // A later server that lists the same id would collide with this one.
          for (const tool of server.tools) unresolved.delete(tool.name);
        } else {
          await server.close();
        }
      }
      if (signal.aborted) {
        // The run throws the stop; only what is already open goes to it.
        selected = Object.fromEntries(Object.entries(selected).filter(([, server]) => server.isOpen));
      } else {
        const missing = [...unresolved].sort(compareCodeUnits);
        if (missing.length > 0 && failed.length > 0) {
          // A server that failed to start may be the one that supplies them.
          const [error, ...others] = failed as [McpConnectError, ...McpConnectError[]];
          selectionNotes.set(error, [
            `the flow needs ${writeAsText(missing)}, which no started server supplies`,
            ...others.map((other) => other.message),
          ]);
          throw error;
        }
      }
      for (const error of failed) warn(`${error.message}; skipped, the flow doesn't need it`);
    }
    return await body(selected);
  } finally {
    // Closing a server that never opened, or that its run already closed, does nothing; this
    // also covers one stopped between opening and being recorded.
    for (const server of Object.values(servers)) await server.close();
  }
}

/**
 * Return `error`'s message with what lets the run resume. A resume needs the saved login it
 * pinned, so deleting a damaged file's logins and logging in again means a new run.
 */
function resumeAdvice(error: StorageError): string {
  if (error.damaged) {
    return (
      `${error.message}. Only restoring the file lets this run resume; after deleting its logins, log in ` +
      "again and start a new run"
    );
  }
  return `${error.message}. Resume again once it is fixed`;
}

/** @internal A resumed run's recorded provider credentials are not available. */
export class ResumeKeyError extends Error {
  override name = "ResumeKeyError";
}

/** The resume key checks' arguments: the command's environment, its terminal if any, the stop. */
interface ResumeKeyContext extends PromptContext {
  signal: AbortSignal;
}

/**
 * Require the first set variable of `envVars` to hold the key a recorded run started with.
 *
 * Only a missing key prompts, at `context.terminal`, for the first variable; a typed key that
 * passes is `set` there in `context.env`, as a fresh run's prompted key is.
 *
 * @param provider The provider name shown in prompts and errors.
 * @param envVars The variables the provider's client reads its key from, in its order.
 * @param identity The recorded `keyIdentity` of the key, or `null` when the record has none, in
 *   which case any non-empty key passes.
 * @throws {ResumeKeyError} The key is set but differs from the recorded one, is missing with no
 *   terminal, is declined or left empty `current().maxKeyPrompts` times, or a typed key differs
 *   from the recorded one.
 * @throws The signal's reason when the stop closes the prompt.
 */
async function resumeEnvKey(
  provider: string,
  envVars: readonly string[],
  identity: string | null,
  { env, terminal, signal }: ResumeKeyContext,
): Promise<void> {
  for (const name of envVars) {
    const existing = env.read(name);
    if (!existing) continue;
    if (identity !== null && keyIdentity(existing) !== identity) {
      throw new ResumeKeyError(`$${name} is not the key this run started with; set it to that key and resume again`);
    }
    return;
  }
  const envVar = envVars[0]!;
  if (terminal === null) {
    const unset = envVars.map((name) => `$${name}`).join(" or ");
    throw new ResumeKeyError(
      `provider ${reprText(provider)} needs an API key but ${unset} is not set; set it and resume again`,
    );
  }
  const cancelled = `no API key provided for ${reprText(provider)} ($${envVar}); set it and resume again`;
  for (let attempt = 0; attempt < current().maxKeyPrompts; attempt++) {
    const answer = await terminal.askText(`Enter ${envVar} for provider ${reprText(provider)}:`, {
      secret: true,
      signal,
    });
    // A prompt the stop closed (Ctrl-C, the end of input, a signal) is the stop, not a decline.
    signal.throwIfAborted();
    if (answer === null) throw new ResumeKeyError(cancelled);
    const key = answer.trim();
    if (!key) continue;
    if (identity !== null && keyIdentity(key) !== identity) {
      throw new ResumeKeyError(
        `the key entered is not the one this run started with; set $${envVar} to that key and resume again`,
      );
    }
    env.set(envVar, key);
    return;
  }
  throw new ResumeKeyError(cancelled);
}

/**
 * Return the error refusing a resume whose `provider` key came from a host's
 * `llm_config.api_key`, pointing to the host's `advance`; `headerNames` are the recorded names of
 * headers sent with the key, if any.
 */
function hostKeyRefusal(provider: string, headerNames: unknown): ResumeKeyError {
  // Header values the host sent with its key are not recorded either.
  const withHeaders = Array.isArray(headerNames) && headerNames.length > 0;
  const given = withHeaders ? "apiKey, headers" : "apiKey";
  return new ResumeKeyError(
    `this run's ${provider} API key came from a host's llm_config.api_key, which the CLI cannot ` +
      `supply; resume it from the host with advance({ log, ${given} })`,
  );
}

/**
 * Check that the recorded model configs of a resumed run can authenticate (`advance`'s
 * `checkLlmConfigs`).
 *
 * Like a fresh run's preflight, it requires each provider's environment key unless the agent
 * recorded a saved login; it also requires an OpenAI environment key to be the one the run started
 * with. Only a missing key prompts. Every saved login (any provider) and every OpenAI connection
 * then resolves its credentials, so a replaced saved credential fails here instead of at the first
 * model call. Checks run once per distinct saved or OpenAI connection and once per other provider;
 * providers without a key variable, and optional-key providers (`vllm`) whose variable is unset,
 * are skipped.
 *
 * @param effective The recorded effective model map, keyed by static agent path, with the config
 *   each started call-time agent recorded. Only each config's `provider`, `_connection`, host-key
 *   marker and recorded header names are read.
 * @param callTimeHost The host key the call-time agents' selection names (`provider` and any
 *   `header_names`); a call-time agent that hasn't started may still resolve to it, so it is
 *   refused like a marked agent.
 * @throws {ResumeKeyError} A key is missing, differs from the recorded one, was not given at the
 *   prompt, belongs to a host, or a recorded saved or OpenAI connection can no longer resolve its
 *   credentials.
 * @throws {RangeError} As `connectionCredentials` does, for invalid connection metadata.
 * @throws The signal's reason when the stop closes a prompt.
 */
async function checkResumeCredentials(
  effective: Readonly<Record<string, AgentConfig>>,
  callTimeHost: Readonly<Record<string, unknown>> | null,
  context: ResumeKeyContext,
): Promise<void> {
  const connections: [string, Connection][] = [];
  const providers: string[] = [];
  for (const config of Object.values(effective)) {
    const provider = config["provider"] as string;
    const connection = (config["_connection"] as Connection | undefined) ?? null;
    if (config[HOST_KEY_MARKER] || connection?.source === "host") {
      throw hostKeyRefusal(provider, config[HEADER_NAMES_KEY]);
    }
    if (connection !== null && (provider === "openai" || connection.source === "store")) {
      const pair: [string, Connection] = [provider, connection];
      if (!connections.some((other) => isDeepStrictEqual(other, pair))) connections.push(pair);
    } else if (!providers.includes(provider)) {
      providers.push(provider);
    }
  }
  if (callTimeHost !== null) {
    throw hostKeyRefusal(callTimeHost["provider"] as string, callTimeHost["header_names"]);
  }

  for (const [provider, connection] of connections) {
    const { source } = connection;
    if (provider === "openai" && (source === "environment" || source === "legacy_environment")) {
      await resumeEnvKey("openai", ["OPENAI_API_KEY"], connection.credential_id, context);
    }
  }
  for (const [provider, connection] of connections) {
    try {
      await connectionCredentials(connection, provider, { env: context.env.read });
    } catch (error) {
      if (!(error instanceof AuthError)) throw error;
      throw new ResumeKeyError(error instanceof StorageError ? resumeAdvice(error) : error.message, { cause: error });
    }
  }
  for (const provider of providers) {
    const envVars = apiKeyEnvNames(provider);
    if (envVars.length === 0) continue;
    if (OPTIONAL_API_KEY_PROVIDERS.has(provider) && !envVars.some((name) => context.env.read(name))) continue;
    await resumeEnvKey(provider, envVars, null, context);
  }
}

/**
 * @internal Return the `checkLlmConfigs` a resume from the CLI passes to `advance`:
 * `checkResumeCredentials` with `context`.
 */
export function resumeKeyCheck(context: ResumeKeyContext): LlmConfigsCheck {
  return (effective, { callTimeHost }) => checkResumeCredentials(effective, callTimeHost, context);
}

/**
 * Start a durable run of the flow at `flowPath`, or recover the one recorded at `resume`; returns
 * the exit code.
 *
 * The run-log options are checked before the flow is read. A flow that isn't UTF-8 text or doesn't
 * load is reported on stderr (a load error located, with its stack under `engineTrace`) and exits
 * 1. When an AGENT grants skills, the Claude Code and Codex skill folders are read, each invalid
 * skill a `warning:` line on stderr. When a static TOOL node or AGENT grant names a tool yian
 * lacks, `~/.yian/mcp.json` (when it is a file) and the `mcpConfig` files are read, a bad config
 * printed and exiting 1; just before the run, `openMcpServers` selects the servers the flow needs,
 * each unneeded server that failed to start a `warning:` line on stderr, and closes them all when
 * the run ends. Unless `quiet`, the flow's banner and per-node progress go to stderr; the log path
 * always does. A warning raised during the run prints as a `warning:` line on stderr, above the
 * spinners. On a TTY stdin, each input not supplied is prompted for on stderr, and a cancelled
 * prompt prints `run cancelled` and exits 1. Off a terminal (stdin or stderr isn't a TTY), each
 * prompt, an `ask` node's included, is written once, with no escape codes. `provider` and `model`,
 * those given, are the run's outermost `llmConfig` layer. Before the log exists, every key the
 * agents need is checked (`ensureProviderKeys`), prompted for on a TTY stdin; a missing one is
 * printed and exits 1. The log is `log`, else `~/.yian/runs/<run id>/log.jsonl` in a fresh folder.
 * A succeeded run prints its output on stdout, as text; a failed one prints its diagnostic on
 * stderr and exits 1. A run that ends waiting prints `run waiting: <node> (<tool>), ...` and the
 * `yian run --resume <log>` that continues it on stderr and exits 0; a cancelled one prints
 * `run cancelled` and exits 1 (`run.cancelled`). An inbox input is never asked for or supplied. A
 * host failure (a bad option value, a log that can't be created, an MCP server that doesn't start,
 * the missing MCP SDK) prints `run failed: <message>` on stderr, and when no started server
 * supplies a needed tool the missing ids and the other failed servers on the lines below, and
 * exits 1.
 *
 * With `resume` there is no `flowPath`: the flow the log recorded is compiled (`loadRecordedFlow`),
 * never a file on disk, and its skills and MCP config are read as for a fresh run; then the log
 * path is printed and `advance` recovers the run, with no input prompts, no `provider`/`model`
 * layer and no key preflight; `numWorkers` absent keeps the recorded count. A log whose flow was
 * built in code, or that `yian chat` wrote (its flow's id `CHAT_FLOW_ID` and its file
 * `CHAT_FLOW_FILE`), can't be resumed here and prints `resume failed: ...`. Before anything is
 * appended, the keys the recorded configs need are checked: a missing one is prompted for on a TTY
 * stdin; a missing, changed or declined key, a run whose key came from a host, or a saved login
 * that no longer resolves prints its message and exits 1 (`run.setup`). A host failure, an empty,
 * damaged or locked log included, prints `resume failed: <message>`. A finished log prints its
 * recorded outcome and appends nothing.
 *
 * Under `json`, stdout holds one
 * document: the output (`status: "succeeded"`), the failure (`status: "failed"`), the waits and
 * `wake_at` (`status: "waiting"`) or the error (`compose.load`, `run.cancelled`, `run.setup`).
 *
 * SIGINT, SIGTERM or SIGHUP from the command's start, Ctrl-C at a prompt during the run, or
 * `io.stop` aborting (with a `HostStop`) stops the run, closing a prompt that is up before it:
 * `cancelled by user` on stderr, exit 130, no document. The end of input at a prompt during the
 * run prints `run cancelled` and exits 1 (`run.cancelled`). The first stop wins.
 *
 * @throws {UsageError} As `validateRunLogOptions` and `mergeInputs` do; for an input the flow
 *   doesn't declare (or an inbox); off a TTY, for missing required inputs.
 */
export async function run(flowPath: string | undefined, options: RunOptions, io: CliIo): Promise<number> {
  const controller = new AbortController();
  // From the command's start, so a signal during the load or the prompts stops the run too.
  const disposeSignals = stopOnSignals(controller);
  const stop = io.stop;
  const forwardStop = () => controller.abort(stop?.reason);
  if (stop?.aborted) forwardStop();
  else stop?.addEventListener("abort", forwardStop, { once: true });
  // Building it has no side effects; the terminal pauses it while a prompt is up.
  const { quiet, verbose } = options;
  const reporter = quiet ? null : new ProgressReporter(io.stderr, { verbose, env: io.env });
  // `startRun` sets it just before `runFlow`.
  const started = { value: false };
  // Building it touches neither stream; it reads keys only when it prompts.
  const terminal = new Terminal({
    input: io.stdin,
    output: io.stderrStream,
    // Off a terminal, stderr gets the prompts once each, with no escape codes.
    plain: !atTerminal(io),
    // Before the run, a stopped prompt answers null and the command reports it.
    onStop: (reason) => {
      if (!started.value) return;
      const why = reason === "interrupt" ? new HostStop("SIGINT") : new TerminalStopped("eof");
      controller.abort(why);
    },
    onActive: (active) => {
      if (active) reporter?.pause();
      // The terminal calls this before `onStop`, so whether the run is stopping is known later.
      else queueMicrotask(() => {
        if (!controller.signal.aborted) reporter?.resume();
      });
    },
  });
  // A stop before the run ends a prompt that is up rather than waiting for its answer.
  const closePrompt = () => {
    if (!started.value) terminal.close();
  };
  controller.signal.addEventListener("abort", closePrompt, { once: true });
  try {
    const body = () => startRun(flowPath, options, io, { terminal, controller, reporter, started });
    // The spinner owns the cursor's line, so a warning raised during the run prints through the
    // reporter, above it; under `--quiet` there is none and `main`'s handler prints it.
    return await (reporter === null ? body() : withWarnings((message) => reporter.warn(message), body));
  } finally {
    controller.signal.removeEventListener("abort", closePrompt);
    terminal.close();
    disposeSignals();
    stop?.removeEventListener("abort", forwardStop);
    reporter?.stop();
  }
}

/** What `run` builds for its body: the terminal, the run's stop, progress, whether it started. */
interface RunHost {
  terminal: Terminal;
  controller: AbortController;
  reporter: ProgressReporter | null;
  started: { value: boolean };
}

/** `run`'s body, prompting at `host.terminal`, stopped through `host.controller`. */
async function startRun(
  flowPath: string | undefined,
  options: RunOptions,
  io: CliIo,
  host: RunHost,
): Promise<number> {
  const { terminal, controller, reporter } = host;
  validateRunLogOptions({ ...options, flow: flowPath });
  const { json, quiet, engineTrace } = options;
  const fail = async (diagnostic: Diagnostic) => {
    if (json) await printJson({ error: diagnosticJson(diagnostic) }, io.stdout);
    return 1;
  };
  const cancelled = async () => {
    await printStyled(io, "run cancelled", "yellow");
    return fail(new Diagnostic({ code: "run.cancelled", message: "run cancelled" }));
  };
  const refused = async (message: string) => {
    await printStyled(io, message, "red");
    return fail(new Diagnostic({ code: "run.setup", message }));
  };
  const writeStack = (error: Error) => writeEngineTrace(error, engineTrace, io);
  // The line a memory's `open` gives when it creates it, with its folder: stderr is the user's own.
  const memoryNote = (created: CreatedMemory) => {
    const line = paint(forTerminal(createdLine(created, true), io.stderr), "dim", io.stderr, io.env);
    void io.stderr.write(`${line}\n`).catch(() => {});
  };
  const action = options.resume === undefined ? "run" : "resume";
  // Only the MCP selection's lines follow the message; the error's own notes, such as a cleanup
  // failure's, stay out of it.
  const setupFailed = async (error: Error) => {
    const notes = selectionNotes.get(error) ?? [];
    const label = paint(`${action} failed: `, "red bold", io.stderr, io.env);
    const text = [error.message, ...notes].map((line) => paint(forTerminal(line, io.stderr), "red", io.stderr, io.env));
    await io.stderr.write(`${label}${text.join("\n")}\n`);
    await writeStack(error);
    return fail(new Diagnostic({ code: "run.setup", message: error.message, notes }));
  };
  let flow: ComposedFlow;
  if (options.resume !== undefined) {
    let recorded: ComposedFlow | null;
    try {
      // `advance` compiles it again: it refuses a flow for a run with recorded files.
      recorded = await loadRecordedFlow(options.resume, { env: io.env });
    } catch (error) {
      if (error instanceof LoadError) {
        await renderDiagnostic(error.diagnostic, io.stderr, io.env);
        await writeStack(error);
        return fail(error.diagnostic);
      }
      if (!isSetupError(error)) throw error;
      return setupFailed(error);
    }
    if (recorded === null) {
      return setupFailed(
        new RangeError("this run's flow was built in code; resume it with advance({log, flow})"),
      );
    }
    // A chat's run needs the chat to type into it and to host its tools.
    if (recorded.compiled.flowId === CHAT_FLOW_ID && recorded.source?.root === CHAT_FLOW_FILE) {
      const message = "this run is a yian chat's; yian run --resume can't continue it";
      return setupFailed(new RangeError(message));
    }
    flow = recorded;
  } else {
    const loaded = await loadFlowFile(flowPath!, io, { engineTrace });
    if (loaded instanceof Diagnostic) return fail(loaded);
    flow = loaded;
  }

  const commandEnv = new CommandEnv(io.env);
  const warned: string[] = [];
  const warn = (message: string) => void warned.push(message);
  const writeWarnings = async () => {
    for (const message of warned.splice(0)) {
      await printStyled(io, `warning: ${message}`, "yellow");
    }
  };

  let outcome: RunResult | number;
  try {
    outcome = await (async () => {
      const skills = await readGrantedSkills(flow, commandEnv.read, io);

      // MCP configs are read only when the flow needs a tool yian doesn't already have, so a
      // broken config file can't block flows that never use MCP. The servers stay unopened here;
      // `openMcpServers` selects the ones the flow needs just before the run.
      const known = knownToolIds();
      const [toolIds, grants] = mcpNeededNames(flow);
      let servers: Record<string, McpServer> = {};
      if (namesUnknown(toolIds, grants, known)) {
        const user = userMcpConfig(commandEnv.read);
        const paths = [...(statOrNull(user)?.isFile() ? [user] : []), ...options.mcpConfig];
        try {
          servers = loadMcpConfig(paths, commandEnv.read, { reserved: known, warn, userConfig: user });
        } catch (error) {
          if (!(error instanceof McpConfigError)) throw error;
          await writeWarnings();
          return refused(error.message);
        }
        await writeWarnings();
      }
      if (!quiet) {
        const banner = flowBanner(flow, io.stderr, io.env);
        if (banner !== null) await io.stderr.write(banner);
      }
      const onEvent = reporter === null ? null : (event: RunEvent) => reporter.handle(event);
      const selection = { signal: controller.signal, warn };
      type Start = (tools: Record<string, ToolEntry>) => Promise<RunResult>;
      /**
       * Print `log`, then start the run with `start`, given the run's tools and MCP servers. With
       * `runDir`, the run folder this command made for `log`, the folder is removed when the
       * command ends before the run created the log in it.
       */
      const startLogged = async (log: string, start: Start, runDir: string | null = null) => {
        try {
          // Not after the log path is printed: a stopped run creates no log.
          controller.signal.throwIfAborted();
          await reportRunLog(log, io.stderr, io.env);
          host.started.value = true;
          try {
            return await openMcpServers(servers, toolIds, grants, known, selection, async (selected) => {
              await writeWarnings();
              try {
                return await start({ ...registeredTools(), ...selected, ask: terminal.askUser });
              } finally {
                // Before anything else is printed: on a TTY, stopping erases the spinner lines on screen.
                reporter?.stop();
              }
            });
          } finally {
            await indexRun(log, io);
          }
        } finally {
          if (runDir !== null) removeEmptyDir(runDir);
        }
      };
      // Recovery takes no inputs, `--provider`/`--model` or key preflight: the log holds the run's
      // configuration, and its recorded keys are checked once it is replayed.
      if (options.resume !== undefined) {
        const log = options.resume;
        const memory = resumeMemories(flow, log, options.memory, {
          home: homeDir(io.env),
          note: memoryNote,
          embed: io.embed ?? defaultEmbed(commandEnv.read),
          accountHome: io.accountHome ?? accountHome,
        });
        const keyContext = {
          env: commandEnv,
          terminal: io.stdin.isTTY === true ? terminal : null,
          signal: controller.signal,
        };
        const checkLlmConfigs = resumeKeyCheck(keyContext);
        return startLogged(log, (tools) =>
          advance({
            log,
            numWorkers: options.numWorkers ?? null,
            onEvent,
            tools,
            skills,
            env: commandEnv.read,
            signal: controller.signal,
            checkLlmConfigs,
            memory,
          }),
        );
      }
      const supplied = mergeInputs(options.inputs, options.inputValues);
      // Before prompting, so a typo'd name fails instead of prompting for the real one.
      checkSuppliedInputs(flow.input, supplied);
      const memories = bindMemories(flow, options.memory, {
        home: homeDir(io.env),
        note: memoryNote,
        project: projectFolder(flowPath!),
        embedding: embeddingFlag(options.embedding),
        defaultEmbedding: () => defaultEmbedding({ env: commandEnv.read }),
        embed: io.embed ?? defaultEmbed(commandEnv.read),
        accountHome: io.accountHome ?? accountHome,
      });
      // A signal during the load stops the command before it prompts.
      controller.signal.throwIfAborted();
      const prompter = io.stdin.isTTY === true ? terminal : null;
      const gathered = await gatherMissingInputs(flow.input, supplied, prompter);
      // A stop closes the prompt, which then answers null: that is the stop, not a cancel.
      controller.signal.throwIfAborted();
      if (gathered === null) return cancelled();
      Object.assign(supplied, gathered);

      const cliCfg = cliLayer(options);
      // Any required key is asked for before the run creates its log.
      const keyRefusal = await preflightKeys(flow, cliCfg, supplied, io, {
        env: commandEnv,
        terminal: prompter,
        signal: controller.signal,
      });
      if (keyRefusal !== null) return fail(keyRefusal);

      const { runId, path: log, runDir } = freshLog(options.log, io.env);
      return startLogged(
        log,
        (tools) =>
          runFlow(flow, { ...supplied, ...memories }, {
            log,
            runId,
            llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
            numWorkers: options.numWorkers,
            onEvent,
            tools,
            skills,
            env: commandEnv.read,
            signal: controller.signal,
          }),
        runDir,
      );
    })();
  } catch (error) {
    if (error instanceof HostStop) {
      // No document: as Ctrl-C, the stop ends the command with nothing on stdout.
      await printStyled(io, "cancelled by user", "yellow");
      return 130;
    }
    // The end of input isn't Ctrl-C: the same outcome as a cancelled prompt before the run.
    if (error instanceof TerminalStopped && error.reason === "eof") return cancelled();
    if (error instanceof ResumeKeyError) return refused(error.message);
    if (!isSetupError(error)) throw error;
    return setupFailed(error);
  }
  if (typeof outcome === "number") return outcome;
  const result = outcome;
  for (const drift of result.memoryDrift) await printStyled(io, driftNote(drift), "yellow");

  if (result.status === "succeeded") {
    if (json) await printJson({ status: "succeeded", output: jsonValue(result.output) }, io.stdout);
    // A string prints as is, any other value in yian's text form; a TTY gets it through forTerminal.
    else await io.stdout.write(`${forTerminal(writeAsText(result.output as Value), io.stdout)}\n`);
    return 0;
  }
  if (result.status === "waiting") {
    if (json) await printJson(waitingDocument(result), io.stdout);
    else await io.stderr.write(waitingText(result));
    return 0;
  }
  if (result.status === "cancelled") return cancelled();
  const diagnostic = await renderRunError(result, flow, engineTrace, io.stderr, io.env);
  if (json) {
    const failure = { ...diagnosticJson(diagnostic), node: result.failure?.nodeId ?? null };
    await printJson({ status: "failed", failure }, io.stdout);
  }
  return 1;
}

/**
 * @internal Return the `--json` document of a waiting result, without `format`: each wait, and
 * `wake_at`, the earliest deadline of a call not yet settled (`null` when none has one).
 */
export function waitingDocument(result: RunResult): Record<string, JsonValue> {
  const waits = result.waits.map((wait) => ({
    node: wait.node,
    call_id: wait.callId,
    tool: wait.tool,
    inbox: wait.inbox,
    deadline: wait.deadline,
    settled: wait.settled,
  }));
  return { status: "waiting", waits, wake_at: wakeAt(result.waits) };
}

/** @internal Return the text a waiting result prints on stderr: what it waits on and how to go on. */
export function waitingText(result: RunResult): string {
  const waits = result.waits.map((wait) => `${wait.node} (${wait.tool})`).join(", ");
  return `run waiting: ${waits}\ncontinue with: yian run --resume ${shellQuote(result.log!)}\n`;
}

/** A loaded flow's input in a `yian check` result. */
interface CheckInput {
  name: string;
  type: string;
  required: boolean;
  has_default: boolean;
}

/** One `yian check` result: the loaded flow's id, name, description and inputs, or why it failed. */
type CheckResult =
  | {
      path: string;
      ok: true;
      id: string | null;
      name: string | null;
      description: string | null;
      inputs: CheckInput[];
    }
  | { path: string; ok: false; diagnostic: Diagnostic };

/**
 * Load the flow file `path` as `yian run` loads it, or confined to `root`, running nothing.
 *
 * A `LoadError`, a flow that isn't UTF-8 text, or one that can't be read is a failed result;
 * another error is thrown.
 */
async function checkFlow(path: string, root: string | undefined, env: Env): Promise<CheckResult> {
  let flow: ComposedFlow;
  try {
    if (root === undefined) {
      const text = readFlowText(path);
      flow = await loadFlow(text, { searchPaths: [dirname(path)], sourceLabel: basename(path), env });
    } else {
      flow = await loadConfined(root, path, new Set(), env);
    }
  } catch (error) {
    if (error instanceof LoadError) return { path, ok: false, diagnostic: error.diagnostic };
    let reason: string;
    if (error instanceof NotUtf8Error) reason = "not UTF-8 text";
    else if (isSystemError(error)) reason = osReason(error);
    else throw error;
    const diagnostic = new Diagnostic({ code: "compose.load", message: `can't read ${path}: ${reason}` });
    return { path, ok: false, diagnostic };
  }
  const inputs = flow.input.map((decl) => ({
    name: decl.name,
    // A flow's `input:` entry always declares its type, so its contract is set.
    type: canonicalTypeLabel(decl.contract!.type),
    required: decl.required,
    has_default: decl.hasDefault,
  }));
  const { name, description } = flow;
  return { path, ok: true, id: flow.compiled.flowId, name, description, inputs };
}

/**
 * Load each flow without running it and report whether it loads; returns the exit code, 1 when
 * any fails.
 *
 * With `root`, each flow, its `uses:` imports and its CODE references must be inside that
 * directory, a flow may not set `system.paths`, and its diagnostics are labelled with its path
 * relative to the root. As text, a loaded flow is a `<path>: ok` line on stdout and a failure its
 * `<path>:` line and diagnostic on stderr; with `json`, one document lists every result, each
 * path as given.
 */
export async function check(
  flows: readonly string[],
  { root, json }: { root?: string | undefined; json: boolean },
  io: CliIo,
): Promise<number> {
  const resolvedRoot = root === undefined ? undefined : realpathSync(root);
  const results: CheckResult[] = [];
  for (const path of flows) results.push(await checkFlow(path, resolvedRoot, io.env));
  if (json) {
    const documents = results.map((result) =>
      result.ok ? result : { path: result.path, ok: false, diagnostics: [diagnosticJson(result.diagnostic)] },
    );
    await printJson({ flows: documents }, io.stdout);
  } else {
    for (const result of results) {
      if (result.ok) {
        await io.stdout.write(`${result.path}: ok\n`);
      } else {
        await printStyled(io, `${result.path}:`, "red");
        await renderDiagnostic(result.diagnostic, io.stderr, io.env);
      }
    }
  }
  return results.every((result) => result.ok) ? 0 : 1;
}

/**
 * Report where the run recorded in the run log `log` stands (`inspectRun`), without appending to
 * it or taking its lock; returns the exit code. Without `log`, list the runs instead (`listRuns`).
 *
 * As text, a `status: <status>` line on stdout, then `in flight: <ids>` and `crashed attempts:
 * <id>=<n>` lines when there are any; with `json`, one document with `status`, `in_flight` (in
 * start order), `crashed_attempts`, and for a waiting run `ready` and `wake_at` (`false` and
 * `null` otherwise). A log that isn't a readable run log (empty, damaged,
 * not a run log, unreadable) prints its message on stderr and exits 1 (`run.setup`).
 */
export async function status(log: string | undefined, { json }: { json: boolean }, io: CliIo): Promise<number> {
  if (log === undefined) return listRuns({ json }, io);
  let inspection: RunInspection;
  try {
    inspection = inspectRun(log);
  } catch (error) {
    let message: string;
    if (error instanceof RangeError) message = error.message;
    else if (isSystemError(error)) message = `can't read ${log}: ${osReason(error)}`;
    else throw error;
    await printStyled(io, message, "red");
    if (json) {
      const diagnostic = new Diagnostic({ code: "run.setup", message });
      await printJson({ error: diagnosticJson(diagnostic) }, io.stdout);
    }
    return 1;
  }
  const { status: runStatus, inFlight, crashedAttempts, ready, wakeAt } = inspection;
  if (json) {
    // `printJson` writes a number as a float and a `bigint` as an int.
    const counts = Object.fromEntries([...crashedAttempts].map(([node, n]) => [node, BigInt(n)]));
    await printJson(
      { status: runStatus, in_flight: inFlight, crashed_attempts: counts, ready, wake_at: wakeAt },
      io.stdout,
    );
    return 0;
  }
  let text = `status: ${runStatus}\n`;
  if (inFlight.length > 0) text += `in flight: ${inFlight.join(", ")}\n`;
  if (crashedAttempts.size > 0) {
    const counts = [...crashedAttempts].map(([node, n]) => `${node}=${n}`);
    text += `crashed attempts: ${counts.join(", ")}\n`;
  }
  await io.stdout.write(text);
  return 0;
}

/**
 * Update the run index of `~/.yian/runs` (`updateRunIndex`) and list its rows (`readRunIndex`), by
 * start time; returns the exit code. A missing runs folder lists nothing.
 *
 * As text, a header and one line per run on stdout, in aligned columns: run id, status
 * (`unreadable at seq N` for a log yian can't read from seq N), start time and flow file (`-` for
 * none, or a flow built in code); nothing when there are no runs. With `json`, one document whose
 * `runs` holds each row: `run_id`, `flow`, `status`, `started_at`, `ended_at`, `parent_run_id`,
 * `memories` (each memory input's `input`, `identity` and `type`, never its location), `last_seq`,
 * `result` (the run's output, `null` until it succeeded), the token sums, for a waiting
 * run `ready` and `wake_at` (`false` and `null` otherwise), and `unreadable` (`seq` and `reason`,
 * or `null`). A log the update can't read is a `warning:` line on stderr: once until it changes, or
 * on every update for a file error such as no permission, which leaves the run's row as it was. An
 * index that can't be updated (another update held it too long, or it isn't a database) prints its
 * message on stderr and exits 1 (`run.setup`).
 */
async function listRuns({ json }: { json: boolean }, io: CliIo): Promise<number> {
  const runsDir = defaultRunsDir(io.env);
  let rows: RunIndexRow[] = [];
  try {
    if (existsSync(runsDir)) {
      updateRunIndex(runsDir);
      rows = readRunIndex(runsDir);
    }
  } catch (error) {
    if (!(error instanceof Error) || error instanceof TypeError) throw error;
    let message = error.message;
    if (sqliteErrorCode(error) !== undefined) message = `can't use ${join(runsDir, "index.sqlite")}: ${error.message}`;
    else if (isSystemError(error)) message = `can't read ${runsDir}: ${osReason(error)}`;
    await printStyled(io, message, "red");
    if (json) await printJson({ error: diagnosticJson(new Diagnostic({ code: "run.setup", message })) }, io.stdout);
    return 1;
  }
  if (json) {
    // `printJson` writes a number as a float and a `bigint` as an int.
    const runs = rows.map((row) => ({
      run_id: row.runId,
      flow: row.flow,
      status: row.status,
      started_at: row.startedAt,
      ended_at: row.endedAt,
      parent_run_id: row.parentRunId,
      memories: row.memories.map(({ input, identity, type }) => ({ input, identity, type })),
      last_seq: row.lastSeq === null ? null : BigInt(row.lastSeq),
      result: row.result === null ? null : parseJson(row.result, "codec"),
      input_tokens: BigInt(row.inputTokens),
      output_tokens: BigInt(row.outputTokens),
      cache_read_tokens: BigInt(row.cacheReadTokens),
      cache_creation_tokens: BigInt(row.cacheCreationTokens),
      reasoning_tokens: BigInt(row.reasoningTokens),
      ready: row.ready,
      wake_at: row.wakeAt,
      unreadable: row.unreadable === null ? null : { seq: BigInt(row.unreadable.seq), reason: row.unreadable.reason },
    }));
    await printJson({ runs }, io.stdout);
    return 0;
  }
  if (rows.length === 0) return 0;
  const table = [
    ["run id", "status", "started", "flow"],
    ...rows.map((row) => [
      row.runId,
      row.unreadable === null ? row.status : `unreadable at seq ${row.unreadable.seq}`,
      row.startedAt ?? "-",
      row.flow ?? "-",
    ]),
  ];
  const widths = [0, 1, 2].map((column) => Math.max(...table.map((cells) => cells[column]!.length)));
  const lines = table.map((cells) => cells.map((cell, column) => (column < 3 ? cell.padEnd(widths[column]!) : cell)).join("  "));
  await io.stdout.write(`${lines.join("\n")}\n`);
  return 0;
}

/**
 * Return the skills `flow`'s AGENTs can be granted, read from the Claude Code and Codex skill
 * folders (user, then project from the working directory up to the git root), or null, reading
 * nothing, when no AGENT grants skills. An invalid skill is skipped with a `warning:` line on
 * stderr rather than blocking the run.
 */
export async function readGrantedSkills(flow: ComposedFlow, env: Env, io: CliIo): Promise<Map<string, Skill> | null> {
  if (skillGrants(flow.compiled).length === 0) return null;
  const warned: string[] = [];
  const roots = defaultSkillRoots({ cwd: process.cwd(), env });
  const skills = loadSkills(roots, { skipInvalid: true, warn: (message) => void warned.push(message) });
  for (const message of warned) await printStyled(io, `warning: ${message}`, "yellow");
  return skills;
}

/** One `yian skills check` result: the skill's name and description, or why it failed to load. */
type SkillResult =
  | { path: string; ok: true; name: string; description: string }
  | { path: string; ok: false; error: { code: "skill.invalid"; message: string } };

/**
 * Load each skill directory and report whether it holds a valid `SKILL.md`; returns the exit code,
 * 1 when any fails.
 *
 * As text, each loaded skill is a `<path>: ok (<name>)` line on stdout and each failure its message
 * on stderr; with `json`, one document lists every result, each path as given. Two directories
 * with one skill name are not reported; that is the caller's concern.
 */
export async function skillsCheck(dirs: readonly string[], { json }: { json: boolean }, io: CliIo): Promise<number> {
  const results = dirs.map((path): SkillResult => {
    try {
      const skill = loadSkillDir(path);
      return { path, ok: true, name: skill.name, description: skill.description };
    } catch (error) {
      if (!(error instanceof SkillError) && !isSystemError(error)) throw error;
      return { path, ok: false, error: { code: "skill.invalid", message: error.message } };
    }
  });
  if (json) {
    await printJson({ skills: results }, io.stdout);
  } else {
    for (const result of results) {
      // The message already names the path.
      if (result.ok) await io.stdout.write(`${result.path}: ok (${result.name})\n`);
      else await io.stderr.write(`${result.error.message}\n`);
    }
  }
  return results.every((result) => result.ok) ? 0 : 1;
}
