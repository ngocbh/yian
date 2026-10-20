/**
 * The flow commands, `yian run`, `yian check`, `yian status` and `yian skills check`, and `yian
 * run`'s helpers: what it prints on stderr, the inputs it reads from flags, its run-log options and
 * default log path, and the signals that stop a run.
 *
 * - `skillsCheck` loads skill directories and reports each as text or one `--json` document.
 * - `renderDiagnostic` boxes each source frame of a diagnostic, its marked lines numbered, the
 *   message below; `diagnosticJson` gives the same diagnostic as a `--json` document's part.
 * - `flowBanner` is the "what am I running" box; `ProgressReporter` draws per-node progress.
 * - `mergeInputs` reads `--inputs` then `--input k=v` (`readInputsJson`, `parseKv`).
 * - `validateRunLogOptions` and `defaultRunLogPath` pick the run's log; `reportRunLog` prints it.
 * - `stopOnSignals` stops a run's controller with a `HostStop` on SIGINT, SIGTERM or SIGHUP.
 *
 * Output is raw ANSI on stderr, coloured only on a TTY without `NO_COLOR`.
 */

import { chmodSync, lstatSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { END_ID, START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { loadSkillDir } from "../compose/skills.ts";
import type { Env } from "../defaults.ts";
import { type Diagnostic, reprText } from "../diagnostics.ts";
import type { RunEvent } from "../events.ts";
import { isSystemError, SkillError } from "../skills/catalog.ts";
import { JSONDecodeError, type JsonValue, parseJson } from "../typesys/json.ts";
import { FileRef, type Value, writeAsText } from "../typesys/values.ts";
import { type CliIo, printJson, UsageError, type Writer } from "./utils.ts";

// The TS loader and run inspection aren't ported yet; each throws until its module lands.
async function loadFlow(text: string, options: { searchPaths?: string[]; sourceLabel: string }): Promise<ComposedFlow> {
  throw new Error("loadFlow is not ported to TypeScript yet");
}

async function inspectRun(log: string): Promise<never> {
  throw new Error("inspectRun is not ported to TypeScript yet");
}

const SGR = { bold: 1, dim: 2, red: 31, green: 32, yellow: 33, cyan: 36 } as const;

/** A text style: a colour or weight, or two of them (`"red bold"`). */
type Style = keyof typeof SGR | `${keyof typeof SGR} ${keyof typeof SGR}`;

/** Return `text` in `style` when `out` is a TTY and `NO_COLOR` is unset or empty, else as is. */
export function paint(text: string, style: Style, out: Writer, env: Env): string {
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

/** Whether `value` is a record: a plain object whose prototype is `Object.prototype` or null. */
function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
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
  const bytes = readFileSync(path);
  let text: string;
  try {
    // The BOM is kept, so the JSON reader refuses it as Python's does.
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    throw new UsageError("Invalid value for --inputs: is not UTF-8 text");
  }
  // Python reads the file in text mode, which turns `\r\n` and `\r` into `\n`; JSON's whitespace
  // includes them all, so only an error's line and column change, and they then match.
  text = text.replace(/\r\n?/g, "\n");
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
 * `~/.yian/runs/<runId>.jsonl`, with `HOME` from `env` (the user's home directory when it is
 * unset or empty).
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

/** Start or recover a durable run of the flow at `flow`; returns the exit code. */
export async function run(flow: string): Promise<number> {
  await loadFlow(readFileSync(flow, "utf8"), { sourceLabel: flow });
  return 0;
}

/** Load each flow without running it and report whether it loads; returns the exit code. */
export async function check(flows: readonly string[]): Promise<number> {
  for (const flow of flows) await loadFlow(readFileSync(flow, "utf8"), { sourceLabel: flow });
  return 0;
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
