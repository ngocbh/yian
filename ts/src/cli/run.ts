/**
 * The flow commands, `yian run`, `yian check`, `yian status` and `yian skills check`, and `yian
 * run`'s helpers: what it prints on stderr, the inputs it reads from flags, its run-log options and
 * default log path, and the signals that stop a run.
 *
 * - `run` checks the run-log options, loads the flow, builds its memories, reads skills and MCP
 *   servers when it needs them, runs it on a fresh log and prints its output or its failure;
 *   `check` loads each flow, or confined to `--root`, and reports each as text or one `--json`
 *   document; `status` inspects a run log.
 * - `skillsCheck` loads skill directories and reports each as text or one `--json` document.
 * - `renderDiagnostic` boxes each source frame of a diagnostic, its marked lines numbered, the
 *   message below; `diagnosticJson` gives the same diagnostic as a `--json` document's part;
 *   `renderRunError` renders a failed run's diagnostic.
 * - `flowBanner` is the "what am I running" box; `ProgressReporter` draws per-node progress.
 * - `mergeInputs` reads `--inputs` then `--input k=v` (`readInputsJson`, `parseKv`).
 * - `validateRunLogOptions` and `defaultRunLogPath` pick the run's log; `reportRunLog` prints it.
 * - `ensureProviderKeys` checks the keys of the connections `agentConnections` finds before a run.
 * - `stopOnSignals` stops a run's controller with a `HostStop` on SIGINT, SIGTERM or SIGHUP.
 *
 * Output is raw ANSI on stderr, coloured only on a TTY without `NO_COLOR`.
 */

import { chmodSync, lstatSync, mkdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { knownCallTimeBaseUrls, knownCallTimeProviders } from "../compile/llm_cascade.ts";
import { copyGraph, END_ID, START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { loadFlow } from "../compose/loader.ts";
import { parseSourceDocument } from "../compose/parse/file.ts";
import { defaultSkillRoots, loadSkillDir, loadSkills } from "../compose/skills.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits, Diagnostic, reprText } from "../diagnostics.ts";
import type { RunEvent } from "../events.ts";
import { isSystemError, resolvedPath, type Skill, SkillError, statOrNull } from "../skills/catalog.ts";
import { JSONDecodeError, type JsonValue, parseJson } from "../typesys/json.ts";
import {
  canonicalTypeLabel,
  FileRef,
  isRecord,
  type Value,
  writeAsText,
} from "../typesys/values.ts";
import { AuthError } from "../llm_clients/auth/index.ts";
import { type Connection, connectionCredentials, connectionSnapshot } from "../llm_clients/connection.ts";
import { MemoryOperationError } from "../memory/contracts.ts";
import { type MemoryRegistry, noteCleanup } from "../memory_host.ts";
import { checkInputNames, coerceInputs } from "../nodes/binding.ts";
import type { Parameter } from "../nodes/contracts.ts";
import { hostInputs, type RunResult, runFlow, runStartLlm, skillGrants, staticNodes } from "../run.ts";
import { McpConnectError, type McpServer } from "../tools/mcp.ts";
import { registeredTools } from "../tools/registry.ts";
import { defaultRunId } from "../typesys/seeding.ts";
import { loadMcpConfig, McpConfigError, userMcpConfig } from "./mcp_config.ts";
import { buildCliMemories, MemoryOptionError } from "./memory.ts";
import { Terminal, TerminalStopped } from "./terminal.ts";
import {
  type CliIo,
  CommandEnv,
  ensureEnvKey,
  MissingKeyError,
  osReason,
  printJson,
  type PromptContext,
  UsageError,
  type Writer,
} from "./utils.ts";

// The TS run inspection isn't ported yet; it throws until its module lands.
async function inspectRun(log: string): Promise<never> {
  throw new Error("inspectRun is not ported to TypeScript yet");
}

const SGR = { bold: 1, dim: 2, red: 31, green: 32, yellow: 33, cyan: 36 } as const;

/** A text style: a colour or weight, or two of them (`"red bold"`). */
type Style = keyof typeof SGR | `${keyof typeof SGR} ${keyof typeof SGR}`;

/** Return `text` in `style` when `out` is a TTY and `NO_COLOR` is unset or empty, else as is. */
function paint(text: string, style: Style, out: Writer, env: Env): string {
  if (!out.isTTY || env("NO_COLOR")) return text;
  const codes = style.split(" ").map((word) => SGR[word as keyof typeof SGR]);
  return `\x1b[${codes.join(";")}m${text}\x1b[0m`;
}

/** Return `text`'s lines as source ranges count them, with no empty line after a final break. */
function splitLines(text: string): string[] {
  const lines = text.split(/\r\n|\r|\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/**
 * Return `rows` in a box drawn in `border`, `title` set into its top edge. Each row is its plain
 * text, which sizes the box, and the text to print.
 */
function box(rows: readonly (readonly [string, string])[], border: Style, out: Writer, env: Env, title = ""): string {
  const heading = title ? `─ ${title} ` : "";
  const width = Math.max(heading.length - 1, ...rows.map(([plain]) => plain.length));
  const edge = (text: string) => paint(text, border, out, env);
  const lines = [edge(`╭${heading}${"─".repeat(width + 2 - heading.length)}╮`)];
  for (const [plain, printed] of rows) {
    lines.push(`${edge("│")} ${printed}${" ".repeat(width - plain.length)} ${edge("│")}`);
  }
  lines.push(edge(`╰${"─".repeat(width + 2)}╯`));
  return lines.join("\n");
}

// Lines of source shown above and below the marked lines, so a large flow isn't printed whole.
const ERR_CONTEXT = 5;

/**
 * Return one boxed source frame, or `null` when no line of `marks` is in `text`.
 *
 * The box shows the source from 5 lines above the first mark to 5 below the last, numbered, each
 * marked line flagged `❱`, and is titled `title:line[,line...]`.
 */
function renderFrameBox(text: string, marks: readonly number[], title: string, out: Writer, env: Env): string | null {
  const lines = splitLines(text);
  const shown = [...new Set(marks)].filter((mark) => 1 <= mark && mark <= lines.length).sort((a, b) => a - b);
  if (shown.length === 0) return null;
  const start = Math.max(1, shown[0]! - ERR_CONTEXT);
  const end = Math.min(lines.length, shown[shown.length - 1]! + ERR_CONTEXT);
  const width = String(end).length;
  const rows: [string, string][] = [];
  for (let line = start; line <= end; line++) {
    const marked = shown.includes(line);
    const gutter = `${marked ? "❱" : " "} ${String(line).padStart(width)} `;
    const code = lines[line - 1]!;
    rows.push([gutter + code, paint(gutter, marked ? "red bold" : "dim", out, env) + code]);
  }
  return box(rows, "red", out, env, `${title}:${shown.join(",")}`);
}

/**
 * Write `diag` to `out`: each frame, outermost first, then the message, then the notes.
 *
 * A frame is boxed around the lines its range covers (`SourceRange.coveredLines`), or shown as
 * its label when none is in its text; its note follows it. Several frames are headed `source
 * locations:`. A diagnostic without frames prints only its message.
 */
export async function renderDiagnostic(diag: Diagnostic, out: Writer, env: Env): Promise<void> {
  const parts: string[] = [];
  if (diag.frames.length > 1) parts.push(paint("source locations:", "red bold", out, env));
  for (const frame of diag.frames) {
    const framed = renderFrameBox(frame.text, frame.range.coveredLines(frame.text), frame.label, out, env);
    parts.push(framed ?? paint(frame.label, "red bold", out, env));
    if (frame.note !== null) parts.push(paint(`  ↳ ${frame.note}`, "yellow", out, env));
  }
  parts.push(paint(diag.message, "red", out, env));
  for (const note of diag.notes) parts.push(paint(`  ↳ ${note}`, "yellow", out, env));
  await out.write(`${parts.join("\n")}\n`);
}

/** A diagnostic in the `yian.cli.v1` form. */
export interface DiagnosticJson {
  code: string;
  message: string;
  frames?: { label: string; range: [bigint, bigint, bigint, bigint]; note?: string }[];
  notes?: string[];
}

/**
 * Return `diag` in the `yian.cli.v1` diagnostic form.
 *
 * A frame's range is `[startLine, startCol, endLine, endCol]`, 1-based with an exclusive end
 * column, as ints (`bigint`s, so `printJson` writes them without `.0`). `frames`, `notes` and a
 * frame's `note` are left out when empty.
 */
export function diagnosticJson(diag: Diagnostic): DiagnosticJson {
  const document: DiagnosticJson = { code: diag.code, message: diag.message };
  if (diag.frames.length > 0) {
    document.frames = diag.frames.map(({ label, range, note }) => ({
      label,
      range: [BigInt(range.startLine), BigInt(range.startCol), BigInt(range.endLine), BigInt(range.endCol)],
      ...(note ? { note } : {}),
    }));
  }
  if (diag.notes.length > 0) document.notes = [...diag.notes];
  return document;
}

/** Return `text`, a stack or captured traceback, in a box titled `traceback`, ending in a newline. */
function tracebackBox(text: string, out: Writer, env: Env): string {
  const lines = splitLines(text.trimEnd());
  const rows = lines.map((line) => [line, paint(line, "dim", out, env)] as const);
  return `${box(rows, "red", out, env, "traceback")}\n`;
}

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
 * Return the box printed when a run starts: `Running flow: <name>  (version: v)` and
 * `Description: …`, ending in a newline; `null` when the flow has no name, description or
 * version.
 */
export function flowBanner(
  flow: { name: string | null; description: string | null; version: string | null },
  out: Writer,
  env: Env,
): string | null {
  const { name, description, version } = flow;
  if (!(name || description || version)) return null;
  let plain = `Running flow: ${name || "(unnamed)"}`;
  let printed = paint("Running flow: ", "bold", out, env) + (name || "(unnamed)");
  if (version) {
    plain += `  (version: ${version})`;
    printed += paint(`  (version: ${version})`, "dim", out, env);
  }
  const rows: [string, string][] = [[plain, printed]];
  if (description) {
    const label = "Description: ";
    splitLines(label + description).forEach((line, i) => {
      rows.push([line, i === 0 ? paint(label, "bold", out, env) + line.slice(label.length) : line]);
    });
  }
  return `${box(rows, "cyan", out, env)}\n`;
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
 * for a router) or a red `✗ <node>` with the error on the next line on failure. With `verbose`
 * each node's output follows its check. Several nodes can run at once (a fan-out), so there is
 * one spinner line per running node. On a TTY the spinners are redrawn every 80 ms and each
 * finished line is printed over them; off a TTY only the finished lines are printed.
 */
export class ProgressReporter {
  readonly #out: Writer;
  readonly #verbose: boolean;
  readonly #env: Env;
  readonly #running = new Set<string>();
  #paused = false;
  #frame = 0;
  // Spinner lines on screen, which the next draw or finished line erases.
  #drawn = 0;
  #timer: ReturnType<typeof setInterval> | null = null;

  constructor(out: Writer, options: { verbose: boolean; env: Env }) {
    this.#out = out;
    this.#verbose = options.verbose;
    this.#env = options.env;
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
        this.#running.add(event.nodeId);
        break;
      case "NodeSucceeded": {
        this.#running.delete(event.nodeId);
        const lines = [style(`✓ ${event.nodeId}`, "green")];
        if (this.#verbose) {
          const bodyLines = splitLines(writeAsText(event.output as Value));
          for (const line of bodyLines.length > 0 ? bodyLines : [""]) lines.push(style(`    ${line}`, "dim"));
        }
        this.#print(lines.join("\n"));
        break;
      }
      case "NodeRouted":
        // A router (CASE) produces no value: it selected an out-edge handle.
        this.#running.delete(event.nodeId);
        this.#print(style(`✓ ${event.nodeId} → ${event.handle}`, "green"));
        break;
      case "NodeExpanded":
        // A spawner grew the graph instead of returning a value; the spliced fragment's terminal
        // commits it, so it gets no line, only its spinner cleared.
        this.#running.delete(event.nodeId);
        break;
      case "NodeFailed":
        if (nodeId !== null) this.#running.delete(nodeId);
        this.#print(`${style(`✗ ${nodeId}`, "red bold")}\n${style(`    ${event.failure.message}`, "red")}`);
        break;
      default:
        return;
    }
    this.#sync();
  }

  /** Stop drawing spinners, erasing them, while a prompt has the terminal. */
  pause(): void {
    this.#paused = true;
    this.#sync();
  }

  /** Draw the spinners again after `pause`. */
  resume(): void {
    this.#paused = false;
    this.#sync();
  }

  /** End the display: forget the running nodes and erase their spinners. */
  stop(): void {
    this.#running.clear();
    this.#sync();
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
    const lines = [...this.#running].map((nodeId) => {
      // A line wider than the terminal would wrap, and the erase would miss its first part.
      const fitted = columns === undefined ? nodeId : [...nodeId].slice(0, Math.max(0, columns - 3)).join("");
      return `${frame} ${paint(fitted, "cyan", this.#out, this.#env)}`;
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
  // The BOM is kept, so the JSON reader refuses it as Python's does. Universal newlines change only
  // an error's line and column, which then match Python's: JSON's whitespace includes them all.
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

// Click's hint for an error in the inputs `--input` and `--inputs` supply together.
const INPUTS_HINT = "Invalid value for '--input' / '--inputs'";

/**
 * Return the inputs to add to `supplied`: each declaration it lacks (key presence) answered at
 * `terminal`'s prompts, or `{}` without a terminal, leaving optional and defaulted inputs absent
 * for START. Returns null when a prompt is cancelled.
 *
 * @throws {UsageError} Without a terminal, when required inputs are missing, naming them all in
 *   declaration order.
 */
async function gatherMissingInputs(
  decls: readonly Parameter[],
  supplied: Readonly<Record<string, unknown>>,
  terminal: Terminal | null,
): Promise<Record<string, unknown> | null> {
  if (terminal !== null) return terminal.promptInputs(decls, supplied);
  const required = decls.filter((decl) => decl.required && !Object.hasOwn(supplied, decl.name));
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
 * Reject fresh-only options on `--resume`, and a `--log` path that already exists. Absent
 * options are `undefined`; `inputValues` is empty when no `--input` is given.
 *
 * Run before input files are read, prompts are shown, or provider configuration is resolved.
 *
 * @throws {UsageError} When `resume` is given with `--input`, `--inputs`, `--provider`,
 *   `--model` or `--log` (each one given is named, in that order), or a directory entry, even a
 *   dangling symlink, exists at `log`.
 */
export function validateRunLogOptions(options: {
  inputValues: readonly string[];
  inputs?: string | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  log?: string | undefined;
  resume?: string | undefined;
}): void {
  const { inputValues, inputs, provider, model, log, resume } = options;
  if (resume !== undefined) {
    const conflicts: string[] = [];
    if (inputValues.length > 0) conflicts.push("--input");
    if (inputs !== undefined) conflicts.push("--inputs");
    if (provider !== undefined) conflicts.push("--provider");
    if (model !== undefined) conflicts.push("--model");
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

/**
 * Create the private CLI directories and return a fresh run's default log path,
 * `~/.yian/runs/<runId>.jsonl`, with `HOME` from `env` (the process user's home directory,
 * `os.homedir()`, when it is unset or empty).
 *
 * Both directories are created with mode 0700, or set to it if they exist. The file itself is
 * not created: the run log creates it exclusively, owner-only.
 *
 * @throws {Error} When a directory can't be created or changed, or a non-directory holds its path.
 */
export function defaultRunLogPath(runId: string, env: Env): string {
  const stateDir = join(env("HOME") || homedir(), ".yian");
  const runsDir = join(stateDir, "runs");
  for (const dir of [stateDir, runsDir]) {
    try {
      mkdirSync(dir, { mode: 0o700 });
    } catch (error) {
      const taken = (error as NodeJS.ErrnoException).code === "EEXIST";
      if (!taken || !statSync(dir, { throwIfNoEntry: false })?.isDirectory()) throw error;
    }
    chmodSync(dir, 0o700);
  }
  return join(runsDir, `${runId}.jsonl`);
}

/** Write the run's log path to `out`, even when progress is quiet. */
export async function reportRunLog(path: string, out: Writer, env: Env): Promise<void> {
  await out.write(`${paint(`Run log: ${path}`, "dim", out, env)}\n`);
}

/**
 * The reason a run's controller is aborted with when the host stops it: a signal, or Ctrl-C at a
 * prompt (`SIGINT`).
 */
export class HostStop extends Error {
  override name = "HostStop";
  /** The signal that stopped the run. */
  readonly signal: NodeJS.Signals;

  constructor(signal: NodeJS.Signals) {
    super(`stopped by ${signal}`);
    this.signal = signal;
  }
}

const STOP_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

/**
 * Abort `controller` with a `HostStop` on the first SIGINT, SIGTERM or SIGHUP `target` emits, and
 * return a disposer that removes the listeners.
 *
 * The first signal removes all three listeners, so a second one gets the default action and ends
 * the process. An already aborted controller keeps its reason.
 */
export function stopOnSignals(
  controller: AbortController,
  target: Pick<NodeJS.EventEmitter, "on" | "off"> = process,
): () => void {
  const listeners = STOP_SIGNALS.map((signal) => {
    const listener = () => {
      dispose();
      controller.abort(new HostStop(signal));
    };
    return [signal, listener] as const;
  });
  const dispose = () => {
    for (const [signal, listener] of listeners) target.off(signal, listener);
  };
  for (const [signal, listener] of listeners) target.on(signal, listener);
  return dispose;
}

/** A flow or inputs file holding bytes that aren't UTF-8. */
class NotUtf8Error extends Error {
  override name = "NotUtf8Error";
}

/**
 * Read the flow or inputs file at `path` as Python's `read_text(encoding="utf-8")` does: strict UTF-8, a BOM
 * kept, `\r\n` and `\r` read as `\n`.
 *
 * @throws {NotUtf8Error} When the file isn't UTF-8 text.
 * @throws {Error} A system error when the file can't be read.
 */
function readFlowText(path: string): string {
  const bytes = readFileSync(path);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch (error) {
    if (error instanceof TypeError) throw new NotUtf8Error(path);
    throw error;
  }
  return text.replace(/\r\n?/g, "\n");
}

/**
 * The options `run` reads; an absent option is `undefined`, an absent repeatable one an empty list,
 * an absent flag `false`.
 */
interface RunOptions {
  inputValues: readonly string[];
  inputs?: string | undefined;
  log?: string | undefined;
  resume?: string | undefined;
  quiet: boolean;
  memoryRoot?: string | undefined;
  memoryStorage: readonly string[];
  verbose: boolean;
  numWorkers?: number | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  engineTrace: boolean;
  mcpConfig: readonly string[];
  json: boolean;
}

/**
 * Whether `error` is a host failure `run` reports as `run failed:` rather than with its stack: a
 * bad value, an unreadable file, a memory failure, an MCP server that didn't start, the missing MCP
 * SDK.
 */
function isSetupError(error: unknown): error is Error {
  return (
    error instanceof TypeError ||
    error instanceof RangeError ||
    error instanceof AuthError ||
    error instanceof MemoryOperationError ||
    error instanceof McpConnectError ||
    // The missing MCP SDK; any other plain `Error` is a bug and keeps its stack.
    (error instanceof Error && error.message.startsWith("MCP servers need the MCP SDK")) ||
    isSystemError(error)
  );
}

/**
 * Return the tool ids `flow`'s static TOOL nodes call, which name single tools, and the ids its
 * static AGENT nodes grant, which name tools or tool groups.
 */
function mcpNeededNames(flow: ComposedFlow): [Set<string>, Set<string>] {
  const toolIds = new Set<string>();
  const grants = new Set<string>();
  for (const [, node] of staticNodes(flow.compiled)) {
    const call = node.hostToolCall();
    if (call !== null) toolIds.add(call.toolId);
    for (const grant of node.grantedTools()) grants.add(grant);
  }
  return [toolIds, grants];
}

/**
 * @internal Run `body` with the configured MCP servers `flow` needs, and close every server in
 * `servers` when it settles.
 *
 * A server an AGENT grants by name (its tool group) is selected unopened, so the run starts it
 * after its own checks and closes it. A TOOL node's id names one tool, never a group, so it counts
 * as unresolved unless it is in `known`; so does a grant that is neither in `known` nor a server
 * name. When some id is unresolved, the granted servers open here instead, and if their tools don't
 * supply every unresolved id, every other server opens to show its tools; those that list a
 * still-unresolved id stay open and the rest close before the run, so an unused server's tool
 * names can't collide with another's. A server opened only to show its tools that fails to start is
 * skipped with a `warn` call, unless the flow still lacks a needed id afterwards. Once `signal`
 * aborts, no further server opens, and only the servers already open are selected. Nothing is
 * selected when the flow needs no tools.
 *
 * @param servers The configured servers by name, none of them open.
 * @param toolIds The tool ids the flow's static TOOL nodes call.
 * @param grants The tool and tool-group ids the flow's static AGENT nodes grant.
 * @param known The tool ids the run supplies without MCP.
 * @param options.signal The run's stop.
 * @param options.warn Called with `<error>; skipped, the flow doesn't need it` for each unneeded
 *   server that failed to start.
 * @param body Called with the selected servers by name, as tool groups for the run.
 * @returns What `body` resolves to.
 * @throws {McpConnectError} A server the flow names doesn't start or lists bad tools; or a needed
 *   id is still missing and some server failed to start, the first failure then noting the missing
 *   ids and each other failure.
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
        if (server.tools.some((tool) => unresolved.has(tool.name))) selected[name] = server;
        else await server.close();
      }
      if (signal.aborted) {
        // The run throws the stop; only what is already open goes to it.
        selected = Object.fromEntries(Object.entries(selected).filter(([, server]) => server.isOpen));
      } else {
        const supplied = toolNames(selected);
        const missing = [...unresolved].filter((id) => !supplied.has(id)).sort(compareCodeUnits);
        if (missing.length > 0 && failed.length > 0) {
          // A server that failed to start may be the one that supplies them.
          const [error, ...others] = failed as [McpConnectError, ...McpConnectError[]];
          const notes = ((error as { notes?: string[] }).notes ??= []);
          notes.push(`the flow needs ${writeAsText(missing)}, which no started server supplies`);
          for (const other of others) notes.push(other.message);
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
 * @internal Return the distinct provider and connection of every agent in `flow`, as a fresh run
 * would bind them.
 *
 * Snapshots the provider and model defaults and the connections once, resolves the `llm_config`
 * cascade as a fresh run would (`cliCfg` its outermost gap-fill layer, values read from `inputs`)
 * on a throwaway copy of the compiled flow, and binds each agent's connection as a fresh run would.
 * An agent whose config resolves only when it runs counts when its provider is already known, an
 * Azure one with the `base_url` its layers set; otherwise its key is checked at its model call.
 *
 * @param cliCfg The `--provider`/`--model` layer, possibly empty.
 * @param inputs The run's supplied inputs, before coercion; absent supplies none.
 * @returns The distinct `[provider, connection]` pairs in first-seen order, the connection `null`
 *   for a provider without one (`ollama`).
 * @throws {RangeError} If no layer gives an agent a provider and model, a supplied input can't be
 *   coerced, an input or a value the `llm_config` reads is invalid, or the saved logins or the
 *   default-provider environment are invalid.
 * @throws {TypeError} As `hostInputs` and `coerceInputs` do.
 */
export function agentConnections(
  flow: ComposedFlow,
  cliCfg: Readonly<Record<string, string>>,
  { inputs = {}, env }: { inputs?: Readonly<Record<string, unknown>>; env: Env },
): [string, Connection | null][] {
  const probe = copyGraph(flow.compiled);
  const snapshot = connectionSnapshot({ env });
  // Coerced as `runFlow` coerces, so an input selects the provider the run will use.
  const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
  const runLlm = runStartLlm(probe, coerced, { ...cliCfg }, snapshot);
  // A call-time agent whose provider no value can change needs its key as surely as a static
  // agent; finding out at its model call would waste the nodes before it.
  const known = knownCallTimeProviders(probe, runLlm.callTime, snapshot.defaults);
  const baseUrls = knownCallTimeBaseUrls(probe, runLlm.callTime);
  const effective: Record<string, Record<string, unknown>> = { ...runLlm.effective };
  for (const [path, provider] of known) {
    if (!provider) continue;
    effective[path] = { provider };
    // A saved Azure key covers only its own resource, so the endpoint picks the key.
    const baseUrl = baseUrls.get(path);
    if (provider === "azure" && baseUrl !== undefined) effective[path]["base_url"] = baseUrl;
  }
  const seen: [string, Connection | null][] = [];
  for (const config of Object.values(snapshot.bind(effective))) {
    const connection = (config["_connection"] as Connection | undefined) ?? null;
    const pair: [string, Connection | null] = [config["provider"] as string, connection];
    if (!seen.some((other) => isDeepStrictEqual(other, pair))) seen.push(pair);
  }
  return seen;
}

/**
 * @internal Check every key `flow`'s agents will need before the run starts, prompting at
 * `context.terminal` if one is missing.
 *
 * Each distinct agent connection (`agentConnections`) is checked once: a saved login resolves its
 * stored credential without a prompt; any other agent needs its provider's key variable, through
 * `ensureEnvKey`, which skips keyless and unknown providers and an unset optional key. An Azure
 * agent whose `base_url` names another resource than the saved login needs `AZURE_OPENAI_API_KEY`.
 * A prompted key is `set` in `context.env`.
 *
 * @param cliCfg The `--provider`/`--model` layer, possibly empty.
 * @param supplied The run's supplied inputs, which `llm_config` values may read to select a
 *   provider.
 * @throws {MissingKeyError} As `ensureEnvKey` does, for a required key that is missing.
 * @throws {AuthError} If a saved login can't be resolved.
 * @throws {RangeError} As `agentConnections` and `connectionCredentials` do.
 * @throws {TypeError} As `agentConnections` does, for inputs that can't be coerced.
 * @throws {Error} A system error when a saved login can't be read.
 */
export async function ensureProviderKeys(
  flow: ComposedFlow,
  cliCfg: Readonly<Record<string, string>>,
  supplied: Readonly<Record<string, unknown>>,
  context: PromptContext,
): Promise<void> {
  const env = context.env.read;
  for (const [provider, connection] of agentConnections(flow, cliCfg, { inputs: supplied, env })) {
    if (connection !== null && connection.source === "store") {
      await connectionCredentials(connection, provider, { env });
    } else {
      await ensureEnvKey(provider, context);
    }
  }
}

/**
 * Return what `body` resolves to once `registry` has closed. When `body` throws, its error is
 * rethrown, with a close failure, and that failure's notes, appended to its `notes`; otherwise a
 * close failure is thrown.
 */
async function closingAfter<T>(registry: MemoryRegistry, body: () => Promise<T>): Promise<T> {
  let value: T;
  try {
    value = await body();
  } catch (error) {
    try {
      await registry.close();
    } catch (cleanup) {
      noteCleanup(error, cleanup, "Memory cleanup failed");
    }
    throw error;
  }
  await registry.close();
  return value;
}

/**
 * Start a durable run of the flow at `flowPath`; returns the exit code. Recovering one (`resume`)
 * isn't ported yet and throws after the flow loads.
 *
 * The run-log options are checked before the flow is read. A flow that isn't UTF-8 text or doesn't
 * load is reported on stderr (a load error located, with its stack under `engineTrace`) and exits
 * 1. The flow's memories are built from `memoryRoot` and `memoryStorage`, passed to the run, and
 * closed when the command ends. When an AGENT grants skills, the Claude Code and Codex skill
 * folders are read, each invalid skill a `warning:` line on stderr. When a static TOOL node or
 * AGENT grant names a tool yian lacks, `~/.yian/mcp.json` (when it is a file) and the `mcpConfig`
 * files are read, a bad config printed and exiting 1; just before the run, `openMcpServers` selects
 * the servers the flow needs, each unneeded server that failed to start a `warning:` line on
 * stderr, and closes them all when the run ends. Unless `quiet`, the flow's banner and per-node
 * progress go to stderr; the log path always does. On a TTY stdin, each input not supplied is
 * prompted for on stderr, and a cancelled prompt prints `run cancelled` and exits 1. `provider` and
 * `model`, those given, are the run's outermost `llmConfig` layer. Before the log exists, every key
 * the agents need is checked (`ensureProviderKeys`), prompted for on a TTY stdin; a missing one is
 * printed and exits 1. The log is `log`, else a fresh one under `~/.yian/runs` named by the run id.
 * A succeeded run prints its output on stdout, as text; a failed one prints its diagnostic on
 * stderr and exits 1. A host failure (a bad option value, a log that can't be created, an MCP
 * server that doesn't start, the missing MCP SDK) prints `run failed: <message>` on stderr, an
 * `McpConnectError`'s notes on the lines below, and exits 1. Under `json`, stdout holds one
 * document: the output (`status: "succeeded"`), the failure (`status: "failed"`) or the error
 * (`compose.load`, `run.cancelled`, `run.setup`).
 *
 * SIGINT, SIGTERM or SIGHUP from the command's start, Ctrl-C at a prompt during the run, or
 * `io.stop` aborting (with a `HostStop`) stops the run, closing a prompt that is up before it:
 * `cancelled by user` on stderr, exit 130, no document. The end of input at a prompt during the
 * run prints `run cancelled` and exits 1 (`run.cancelled`). The first stop wins.
 *
 * @throws {UsageError} As `validateRunLogOptions` and `mergeInputs` do; for a memory flag value
 *   the flow can't take; for an input the flow doesn't declare; off a TTY, for missing required
 *   inputs.
 */
export async function run(flowPath: string, options: RunOptions, io: CliIo): Promise<number> {
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
    return await startRun(flowPath, options, io, { terminal, controller, reporter, started });
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
  flowPath: string,
  options: RunOptions,
  io: CliIo,
  host: RunHost,
): Promise<number> {
  const { terminal, controller, reporter } = host;
  validateRunLogOptions(options);
  const { json, quiet, engineTrace } = options;
  const fail = async (diagnostic: Diagnostic) => {
    if (json) await printJson({ error: diagnosticJson(diagnostic) }, io.stdout);
    return 1;
  };
  const writeStack = async (error: Error) => {
    if (engineTrace) await io.stderr.write(tracebackBox(error.stack ?? String(error), io.stderr, io.env));
  };
  // An McpConnectError's notes (the ids no started server supplies, the other servers that
  // failed) follow its message; other notes, such as a cleanup failure, stay out of it.
  const setupFailed = async (error: Error) => {
    const notes = error instanceof McpConnectError ? ((error as { notes?: string[] }).notes ?? []) : [];
    const label = paint("run failed: ", "red bold", io.stderr, io.env);
    const text = [error.message, ...notes].map((line) => paint(line, "red", io.stderr, io.env));
    await io.stderr.write(`${label}${text.join("\n")}\n`);
    await writeStack(error);
    return fail(new Diagnostic({ code: "run.setup", message: error.message, notes }));
  };
  let text: string;
  try {
    text = readFlowText(flowPath);
  } catch (error) {
    if (!(error instanceof NotUtf8Error)) throw error;
    const message = `can't read ${flowPath}: not UTF-8 text`;
    await io.stderr.write(`${paint(message, "red", io.stderr, io.env)}\n`);
    return fail(new Diagnostic({ code: "compose.load", message }));
  }
  let flow: ComposedFlow;
  try {
    const loading = { searchPaths: [dirname(flowPath)], sourceLabel: basename(flowPath), env: io.env };
    flow = await loadFlow(text, loading);
  } catch (error) {
    if (!(error instanceof LoadError)) throw error;
    // Where in the `.yaml` the flow broke, not the engine's internals, unless asked for.
    await renderDiagnostic(error.diagnostic, io.stderr, io.env);
    await writeStack(error);
    return fail(error.diagnostic);
  }

  const commandEnv = new CommandEnv(io.env);
  // Building opens nothing: built-in storage opens when the run starts.
  let registry: MemoryRegistry;
  try {
    const { memoryRoot: root, memoryStorage: storageOptions } = options;
    registry = await buildCliMemories(flow, { root, storageOptions, env: commandEnv.read });
  } catch (error) {
    if (error instanceof MemoryOptionError) {
      throw new UsageError(`Invalid value for ${error.paramHint}: ${error.message}`);
    }
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    return setupFailed(error);
  }
  const warned: string[] = [];
  const warn = (message: string) => void warned.push(message);
  const writeWarnings = async () => {
    for (const message of warned.splice(0)) {
      await io.stderr.write(`${paint(`warning: ${message}`, "yellow", io.stderr, io.env)}\n`);
    }
  };

  let outcome: RunResult | number;
  try {
    outcome = await closingAfter(registry, async () => {
      // Claude Code and Codex skill folders (user, then project up to the git root), read only
      // when an AGENT grants skills. An invalid skill is warned about and skipped rather than
      // blocking the run.
      let skills: Map<string, Skill> | null = null;
      if (skillGrants(flow.compiled).length > 0) {
        const roots = defaultSkillRoots({ cwd: process.cwd(), env: commandEnv.read });
        skills = loadSkills(roots, { skipInvalid: true, warn });
        await writeWarnings();
      }

      // MCP configs are read only when the flow needs a tool yian doesn't already have, so a
      // broken config file can't block flows that never use MCP. The servers stay unopened here;
      // `openMcpServers` selects the ones the flow needs just before the run.
      const known = new Set([...Object.keys(registeredTools()), "ask_user", "load_skill"]);
      const [toolIds, grants] = mcpNeededNames(flow);
      let servers: Record<string, McpServer> = {};
      if ([...toolIds, ...grants].some((id) => !known.has(id))) {
        const user = userMcpConfig(commandEnv.read);
        const paths = [...(statOrNull(user)?.isFile() ? [user] : []), ...options.mcpConfig];
        try {
          servers = loadMcpConfig(paths, commandEnv.read, { reserved: known, warn });
        } catch (error) {
          if (!(error instanceof McpConfigError)) throw error;
          await writeWarnings();
          await io.stderr.write(`${paint(error.message, "red", io.stderr, io.env)}\n`);
          return fail(new Diagnostic({ code: "run.setup", message: error.message }));
        }
        await writeWarnings();
      }
      if (!quiet) {
        const banner = flowBanner(flow, io.stderr, io.env);
        if (banner !== null) await io.stderr.write(banner);
      }
      // Recovering a run isn't ported yet; until then `--resume` runs nothing.
      if (options.resume !== undefined) {
        throw new Error("yian run --resume is not ported to TypeScript yet");
      }
      const supplied = mergeInputs(options.inputs, options.inputValues);
      // Before prompting, so a typo'd name fails instead of prompting for the real one.
      try {
        checkInputNames(flow.input, supplied);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw new UsageError(`${INPUTS_HINT}: ${error.message}`);
      }
      // A signal during the load stops the command before it prompts.
      controller.signal.throwIfAborted();
      const prompter = io.stdin.isTTY === true ? terminal : null;
      const gathered = await gatherMissingInputs(flow.input, supplied, prompter);
      // A stop closes the prompt, which then answers null: that is the stop, not a cancel.
      controller.signal.throwIfAborted();
      if (gathered === null) {
        await io.stderr.write(`${paint("run cancelled", "yellow", io.stderr, io.env)}\n`);
        return fail(new Diagnostic({ code: "run.cancelled", message: "run cancelled" }));
      }
      Object.assign(supplied, gathered);

      // The flags supply the outermost cascade layer (fill the gap), not an override: an agent's
      // own llm_config and the flow's still win.
      const cliCfg: Record<string, string> = {};
      if (options.provider) cliCfg["provider"] = options.provider;
      if (options.model) cliCfg["model"] = options.model;
      // Any required key is asked for before the run creates its log.
      try {
        await ensureProviderKeys(flow, cliCfg, supplied, { env: commandEnv, terminal: prompter });
      } catch (error) {
        // A key prompt the stop closed is the stop, not a missing key.
        controller.signal.throwIfAborted();
        if (!(error instanceof MissingKeyError || error instanceof AuthError)) throw error;
        await io.stderr.write(`${paint(error.message, "red", io.stderr, io.env)}\n`);
        return fail(new Diagnostic({ code: "run.setup", message: error.message }));
      }

      // Not after the log path is printed: a stopped run creates no log.
      controller.signal.throwIfAborted();
      const runId = defaultRunId();
      const log = options.log ?? defaultRunLogPath(runId, io.env);
      await reportRunLog(log, io.stderr, io.env);
      host.started.value = true;
      const selection = { signal: controller.signal, warn };
      return openMcpServers(servers, toolIds, grants, known, selection, async (selected) => {
        await writeWarnings();
        try {
          return await runFlow(flow, supplied, {
            log,
            runId,
            llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
            numWorkers: options.numWorkers ?? 8,
            onEvent: reporter === null ? null : (event) => reporter.handle(event),
            memory: registry.size > 0 ? registry : null,
            tools: { ...registeredTools(), ...selected, ask_user: terminal.askUser },
            skills,
            env: commandEnv.read,
            signal: controller.signal,
          });
        } finally {
          // Before anything else is printed: on a TTY, stopping erases the spinner lines on screen.
          reporter?.stop();
        }
      });
    });
  } catch (error) {
    if (error instanceof HostStop) {
      // No document: as Ctrl-C, the stop ends the command with nothing on stdout.
      await io.stderr.write(`${paint("cancelled by user", "yellow", io.stderr, io.env)}\n`);
      return 130;
    }
    // The end of input isn't Ctrl-C: the same outcome as a cancelled prompt before the run.
    if (error instanceof TerminalStopped && error.reason === "eof") {
      await io.stderr.write(`${paint("run cancelled", "yellow", io.stderr, io.env)}\n`);
      return fail(new Diagnostic({ code: "run.cancelled", message: "run cancelled" }));
    }
    if (!isSetupError(error)) throw error;
    return setupFailed(error);
  }
  if (typeof outcome === "number") return outcome;
  const result = outcome;

  if (result.status === "succeeded") {
    if (json) await printJson({ status: "succeeded", output: jsonValue(result.output) }, io.stdout);
    // A string prints as is; any other value in yian's text form.
    else await io.stdout.write(`${writeAsText(result.output as Value)}\n`);
    return 0;
  }
  const diagnostic = await renderRunError(result, flow, engineTrace, io.stderr, io.env);
  if (json) {
    const failure = { ...diagnosticJson(diagnostic), node: result.failure?.nodeId ?? null };
    await printJson({ status: "failed", failure }, io.stdout);
  }
  return 1;
}

/**
 * Return `path` resolved as Python's non-strict `Path.resolve()` does, except that a symlink loop
 * throws its `ELOOP` system error.
 */
function resolveFlowPath(path: string): string {
  try {
    return realpathSync(path);
  } catch (error) {
    if (!isSystemError(error) || error.code === "ELOOP") throw error;
    return resolvedPath(path);
  }
}

/** Whether `shown`, a path relative to the root, leaves it. */
function leavesRoot(shown: string): boolean {
  return shown === ".." || shown.startsWith(`..${sep}`) || isAbsolute(shown);
}

/**
 * Load the flow at `path` from inside `root`, labelled with its path relative to `root`; its
 * `uses:` children load the same way, each ref only from `<importing file's dir>/<ref>.yaml`.
 *
 * The loader puts the `uses:` alias frame on a child's `LoadError` and turns anything else a child
 * throws into a `compose.load` error there.
 *
 * @param root The resolved root directory.
 * @param loading The resolved paths of the flows importing this one.
 * @throws {LoadError} `compose.outside_root` for a flow outside `root`; `compose.load` for a flow
 *   whose YAML doesn't parse, that sets `system.paths`, or that doesn't load.
 * @throws {Error} A reference cycle.
 * @throws {NotUtf8Error} When the flow isn't UTF-8 text.
 * @throws {Error} A system error when the flow can't be read.
 */
async function loadConfined(
  root: string,
  path: string,
  loading: ReadonlySet<string>,
  env: Env,
): Promise<ComposedFlow> {
  // Resolved paths are compared, so a symlinked root (/var -> /private/var on macOS) works.
  const resolved = resolveFlowPath(path);
  const shown = relative(root, resolved) || ".";
  if (leavesRoot(shown)) {
    const message = `${shown} is outside the root`;
    throw new LoadError(new Diagnostic({ code: "compose.outside_root", message }));
  }
  if (loading.has(resolved)) throw new Error(`flow-reference cycle at ${shown}`);
  const text = readFlowText(resolved);
  const doc = parseSourceDocument(text, shown).root.value;
  // The loader doesn't search `system.paths` when given a child resolver, so such a flow would
  // load differently from `yian run`.
  if (isRecord(doc) && isRecord(doc["system"]) && Object.hasOwn(doc["system"], "paths")) {
    throw new LoadError(
      new Diagnostic({
        code: "compose.load",
        message: "system.paths is not supported with --root; use a relative uses: path",
      }),
    );
  }
  const dir = dirname(resolved);
  const importing = new Set([...loading, resolved]);
  // Unlike the default file resolver, this keeps no reuse cache, so a child imported along several
  // paths loads once per path; that is fine for a check.
  const childResolver = async (ref: string, version: string | null): Promise<ComposedFlow> => {
    const childPath = join(dir, `${ref}.yaml`);
    // A missing child outside the root is reported as outside, by `loadConfined`.
    const inside = !leavesRoot(relative(root, resolveFlowPath(childPath)));
    if (inside && !statOrNull(childPath)?.isFile()) {
      throw new Error(`flow ${reprText(ref)} not found: ${relative(root, childPath)}`);
    }
    const child = await loadConfined(root, childPath, importing, env);
    if (version !== null && child.version !== version) {
      const shownVersion = child.version === null ? "null" : reprText(child.version);
      throw new Error(`${ref} is version ${shownVersion}, not ${reprText(version)}`);
    }
    return child;
  };
  // `CodeNode.check()` resolves a `./file.ts:fn` reference against `baseDir`, here and in a child.
  return loadFlow(text, { childResolver, sourceLabel: shown, baseDir: dir, env });
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
 * With `root`, each flow must be inside that directory and may not set `system.paths`, and its
 * diagnostics are labelled with its path relative to the root. As text, a loaded flow is a
 * `<path>: ok` line on stdout and a failure its `<path>:` line and diagnostic on stderr; with
 * `json`, one document lists every result, each path as given.
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
        await io.stderr.write(`${paint(`${result.path}:`, "red", io.stderr, io.env)}\n`);
        await renderDiagnostic(result.diagnostic, io.stderr, io.env);
      }
    }
  }
  return results.every((result) => result.ok) ? 0 : 1;
}

/** Report the run recorded in the run log `log`; returns the exit code. */
export async function status(log: string): Promise<number> {
  return inspectRun(log);
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
