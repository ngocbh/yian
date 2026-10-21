/**
 * CLI helpers: the command I/O, the argument parser, the `--json` writer, provider credentials
 * and endpoints, and `osReason` (a system error's reason in Python's words).
 *
 * - `parseCommand` reads a command's arguments against its `CommandSpec` with `node:util`'s
 *   `parseArgs`, then converts int and float options and checks paths with Click's messages;
 *   `helpText` and `usageError` give the text a command prints for `--help` and for a `UsageError`.
 * - `printJson` writes the one `yian.cli.v1` document a `--json` command prints.
 * - `relay` copies the `run --json` child's document to the entry's stdout; `abortOnSignal` turns
 *   the stop the entry forwards to the child into its `CliIo.stop`, aborted with a `HostStop`.
 * - `ensureEnvKey` is the interactive key prompt `llm_clients/api_key_env.ts` points at; `yian run`
 *   calls it pre-flight for agents without a saved login, so a missing key surfaces as an upfront
 *   prompt instead of a failure on the first API call. `ensureApiKey` resolves a provider's saved
 *   login first, then falls back to `ensureEnvKey`. `confirmOllamaEndpoint` surfaces the resolved
 *   Ollama endpoint right after provider selection.
 *
 * The key and endpoint helpers prompt only when given a terminal; without one they return the
 * ambient value, or throw for a required key that is missing, so non-interactive runs stay
 * scriptable. They read the environment through a `CommandEnv`, and record a prompted value there
 * rather than in the process environment.
 */

import { accessSync, constants, statSync } from "node:fs";
import { parseArgs } from "node:util";

import { current, type Env, processEnv } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { getApiKeyEnv, OPTIONAL_API_KEY_PROVIDERS } from "../llm_clients/api_key_env.ts";
import { ApiKeyCredentials } from "../llm_clients/auth/index.ts";
import { connectionCredentials, connectionSnapshot } from "../llm_clients/connection.ts";
import { logJson } from "../typesys/json.ts";
import type { Terminal } from "./terminal.ts";

/** The `format` of every document a `--json` command prints. */
export const JSON_FORMAT = "yian.cli.v1";

// Mirror of the Ollama client's default base URL.
const OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434";

/** An output stream: `write` resolves once the stream has taken the text. */
export interface Writer {
  write(text: string): Promise<void>;
  readonly isTTY?: boolean;
  readonly columns?: number;
}

/**
 * A command's standard streams and environment: `stdin` and `stderrStream` are the raw streams a
 * `Terminal` prompts on, `stderrStream` being the stream under `stderr`.
 */
export interface CliIo {
  stdout: Writer;
  stderr: Writer;
  stdin: NodeJS.ReadableStream & { isTTY?: boolean };
  stderrStream: NodeJS.WritableStream & { columns?: number };
  env: Env;
  /**
   * A stop from outside the process's signals, aborted with a `HostStop`: aborted, or aborting
   * later, it stops the run.
   */
  stop?: AbortSignal;
}

/** The last write each `streamWriter` started, which `flush` awaits. */
const lastWrites = new WeakMap<Writer, Promise<void>>();
const quieted = new WeakSet<NodeJS.WritableStream>();

/** Whether `error` says the stream's reader has gone (a closed pipe, or the stream it destroyed). */
function isClosedPipe(error: Error): boolean {
  const code = (error as NodeJS.ErrnoException).code;
  return code === "EPIPE" || code === "ERR_STREAM_DESTROYED";
}

/**
 * Return a `Writer` over `stream`, whose writes resolve on the stream's callback.
 *
 * A write to a closed pipe (`yian check --json | head`) resolves quietly; another write error
 * rejects.
 */
export function streamWriter(stream: NodeJS.WritableStream & { isTTY?: boolean; columns?: number }): Writer {
  if (!quieted.has(stream)) {
    quieted.add(stream);
    // A failed write also emits `error`, which would end the process unheard; the write's
    // callback reports it instead.
    stream.on("error", () => {});
  }
  const writer: Writer = {
    write(text) {
      const done = new Promise<void>((resolve, reject) => {
        stream.write(text, (error) => (error && !isClosedPipe(error) ? reject(error) : resolve()));
      });
      lastWrites.set(writer, done);
      return done;
    },
    get isTTY() {
      return stream.isTTY === true;
    },
    get columns() {
      return stream.columns;
    },
  };
  return writer;
}

/** Return the process's streams and environment. */
export function processIo(): CliIo {
  return {
    stdout: streamWriter(process.stdout),
    stderr: streamWriter(process.stderr),
    stdin: process.stdin,
    stderrStream: process.stderr,
    env: processEnv,
  };
}

/**
 * Wait until the last write to `io.stdout` and `io.stderr` has been taken, so the process can exit.
 *
 * The last write's own promise is awaited because an empty write's callback can come before the
 * earlier writes are out on Bun. A failed write is the writer's caller's to report.
 */
export async function flush(io: CliIo): Promise<void> {
  await Promise.allSettled([lastWrites.get(io.stdout), lastWrites.get(io.stderr)]);
}

/** Write what `from` gives to `to`, resolving once `from` has ended and `to` took the last. */
export async function relay(from: NodeJS.ReadableStream, to: Writer): Promise<void> {
  from.setEncoding("utf8");
  for await (const text of from) await to.write(text as string);
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

/**
 * Return a signal aborted with `reason` on the first `name` signal `target` emits.
 *
 * The listener is on from the call, so a signal while the process is still starting isn't lost,
 * and it stays on: a later `name` signal does nothing instead of getting the default action.
 */
export function abortOnSignal(
  name: NodeJS.Signals,
  reason: unknown,
  target: Pick<NodeJS.EventEmitter, "on"> = process,
): AbortSignal {
  const controller = new AbortController();
  target.on(name, () => controller.abort(reason));
  return controller.signal;
}

/** A command's view of the environment: the values it `set`, over the base environment. */
export class CommandEnv {
  readonly #base: Env;
  readonly #values = new Map<string, string>();

  constructor(base: Env) {
    this.#base = base;
  }

  /** Return `name`'s value: the one this command set, else the base environment's. */
  readonly read: Env = (name) => (this.#values.has(name) ? this.#values.get(name) : this.#base(name));

  /** Record `value` for `name`, shadowing the base environment. */
  set(name: string, value: string): void {
    this.#values.set(name, value);
  }
}

/** A command line that doesn't match its command: the CLI prints the usage and exits with 2. */
export class UsageError extends Error {
  override name = "UsageError";
}

/**
 * What a path must be: `exists`; a directory (`dir: true`) or a file (`dir: false`); `readable`;
 * `writable`. A path that doesn't exist passes unless `exists` is set.
 */
export interface PathCheck {
  exists?: boolean;
  dir: boolean;
  readable?: boolean;
  writable?: boolean;
}

/**
 * An option: `--name` (and `-short`), a string, a flag, an int or a float, repeatable when
 * `multiple`.
 */
export interface OptionSpec {
  name: string;
  short?: string;
  type: "string" | "boolean" | "int" | "float";
  multiple?: boolean;
  min?: number;
  path?: PathCheck;
  metavar?: string;
  help: string;
}

/** A positional argument; a `variadic` one takes every remaining argument, at least one. */
export interface PositionalSpec {
  name: string;
  help: string;
  variadic?: boolean;
  /** May be omitted; then absent from `args`. */
  optional?: boolean;
  path?: PathCheck;
}

/** A command: `yian <name>`, its one-line `summary`, its positionals and options. */
export interface CommandSpec {
  name: string;
  summary: string;
  positionals: PositionalSpec[];
  options: OptionSpec[];
}

/** An option's value: absent, a string, a flag, a number, or a list of them for a repeatable one. */
export type OptionValue = string | boolean | number | string[] | number[] | undefined;

/**
 * A parsed command line: `--help`, or the positionals and options by name. A flag absent is
 * `false` and a repeatable option absent `[]`; another option absent is left out.
 */
export type ParsedCommand =
  | { help: true }
  | { help: false; args: Record<string, string | string[]>; options: Record<string, OptionValue> };

/** A parsed command line that isn't a `--help` request: what a command's handler receives. */
export type ParsedArgs = Extract<ParsedCommand, { help: false }>;

/** Return `positional`'s name as the usage text shows it. */
function metavar(positional: PositionalSpec): string {
  const name = positional.name.toUpperCase() + (positional.variadic ? "..." : "");
  return positional.optional ? `[${name}]` : name;
}

/** Return the parameter hint Click's messages show for `option`. */
function optionHint(option: OptionSpec): string {
  return option.short === undefined ? `'--${option.name}'` : `'--${option.name}' / '-${option.short}'`;
}

/** Throw a `UsageError` if `value` doesn't pass `check`, in Click's order and words. */
function checkPath(hint: string, value: string, check: PathCheck): void {
  const kind = check.dir ? "Directory" : "File";
  const fail = (what: string): never => {
    throw new UsageError(`Invalid value for ${hint}: ${kind} ${reprText(value)} ${what}.`);
  };
  const can = (mode: number) => {
    try {
      accessSync(value, mode);
      return true;
    } catch {
      return false;
    }
  };
  let stats;
  try {
    stats = statSync(value);
  } catch {
    if (check.exists) fail("does not exist");
    return;
  }
  if (check.dir && stats.isFile()) fail("is a file");
  if (!check.dir && stats.isDirectory()) fail("is a directory");
  if (check.readable && !can(constants.R_OK)) fail("is not readable");
  if (check.writable && !can(constants.W_OK)) fail("is not writable");
}

/** Return `text` as an int `option` accepts, or throw a `UsageError`. */
function toInt(option: OptionSpec, text: string): number {
  if (!/^-?[0-9]+$/.test(text)) {
    throw new UsageError(`Invalid value for ${optionHint(option)}: ${reprText(text)} is not a valid integer.`);
  }
  const value = Number(text);
  if (option.min !== undefined && value < option.min) {
    throw new UsageError(`Invalid value for ${optionHint(option)}: ${value} is not in the range x>=${option.min}.`);
  }
  return value;
}

// Python's float() syntax in ASCII: surrounding ASCII whitespace (float() refuses U+001C to
// U+001F, though `str.isspace` counts them), single underscores between digits, an optional
// fraction and exponent, or inf, infinity or nan in any case.
const FLOAT_TEXT =
  /^[\t-\r ]*([+-]?)(?:(inf|infinity|nan)|((?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:e[+-]?\d(?:_?\d)*)?))[\t-\r ]*$/i;

/** Return `text` as a float, as Python's `float()` reads it, or throw a `UsageError`. */
function toFloat(option: OptionSpec, text: string): number {
  const match = FLOAT_TEXT.exec(text);
  if (!match) {
    throw new UsageError(`Invalid value for ${optionHint(option)}: ${reprText(text)} is not a valid float.`);
  }
  const [, sign, word, digits] = match;
  const magnitude =
    digits === undefined ? (word!.toLowerCase() === "nan" ? NaN : Infinity) : Number(digits.replaceAll("_", ""));
  return sign === "-" ? -magnitude : magnitude;
}

/** Return `parseArgs`' error as a `UsageError` worded as Click words it. */
function parseArgsError(error: Error): UsageError {
  const message = error.message;
  let match = /^Unknown option '([^']*)'/.exec(message);
  if (match) return new UsageError(`No such option: ${match[1]}`);
  match = /--(\S+) <value>' argument missing/.exec(message);
  if (match) return new UsageError(`Option '--${match[1]}' requires an argument.`);
  // `-w -1`: parseArgs refuses a separate value starting with `-`, as Click finds none.
  match = /^Option '([^']*)' argument is ambiguous/.exec(message);
  if (match) return new UsageError(`Option '${match[1]}' requires an argument.`);
  // parseArgs names the option `-q, --quiet`; Click names the long form alone.
  match = /^Option '(?:-[^-], )?([^']*)' does not take an argument/.exec(message);
  if (match) return new UsageError(`Option '${match[1]}' does not take a value.`);
  return new UsageError(message);
}

/**
 * Parse `argv` (the arguments after the command's name) against `spec`.
 *
 * Options and positionals may mix; `--` ends the options. An int or float option's value is
 * converted, an int's checked against `min`; each path is checked once parsing is done,
 * positionals then options, in declared order. `--help` anywhere returns `{help: true}` before any
 * of those checks.
 *
 * @throws {UsageError} For an unknown option, a missing or refused value, a missing or extra
 *   positional, an invalid int or float, or a path that fails its check.
 */
export function parseCommand(spec: CommandSpec, argv: readonly string[]): ParsedCommand {
  const options: Record<string, { type: "string" | "boolean"; short?: string; multiple?: boolean }> = {
    help: { type: "boolean" },
  };
  for (const option of spec.options) {
    options[option.name] = {
      type: option.type === "boolean" ? "boolean" : "string",
      ...(option.short === undefined ? {} : { short: option.short }),
      ...(option.multiple ? { multiple: true } : {}),
    };
  }
  let parsed;
  try {
    parsed = parseArgs({ args: [...argv], options, strict: true, allowPositionals: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code?.startsWith("ERR_PARSE_ARGS_")) throw parseArgsError(error as Error);
    throw error;
  }
  const values = parsed.values as Record<string, string | boolean | string[] | undefined>;
  if (values["help"] === true) return { help: true };

  const rest = [...parsed.positionals];
  const args: Record<string, string | string[]> = {};
  for (const positional of spec.positionals) {
    if (rest.length === 0) {
      if (positional.optional) continue;
      throw new UsageError(`Missing argument '${metavar(positional)}'.`);
    }
    const taken = positional.variadic ? rest.splice(0) : rest.splice(0, 1);
    if (positional.path) for (const value of taken) checkPath(`'${metavar(positional)}'`, value, positional.path);
    args[positional.name] = positional.variadic ? taken : taken[0]!;
  }

  const result: Record<string, OptionValue> = {};
  for (const option of spec.options) {
    const raw = values[option.name];
    if (raw === undefined) {
      if (option.type === "boolean") result[option.name] = false;
      else if (option.multiple) result[option.name] = [];
      continue;
    }
    if (option.type === "boolean") {
      result[option.name] = raw;
      continue;
    }
    const texts = typeof raw === "string" ? [raw] : (raw as string[]);
    if (option.path) for (const text of texts) checkPath(optionHint(option), text, option.path);
    const converted =
      option.type === "int"
        ? texts.map((text) => toInt(option, text))
        : option.type === "float"
          ? texts.map((text) => toFloat(option, text))
          : texts;
    result[option.name] = option.multiple ? (converted as string[] | number[]) : converted[0];
  }

  if (rest.length > 0) {
    throw new UsageError(`Got unexpected extra argument(s) (${rest.join(" ")})`);
  }
  return { help: false, args, options: result };
}

/** Return the `Usage:` line of `spec`. */
function usageLine(spec: CommandSpec): string {
  return ["Usage: yian", spec.name, "[OPTIONS]", ...spec.positionals.map(metavar)].join(" ");
}

/** Return the name column of `option` in the help text. */
function optionLabel(option: OptionSpec): string {
  const names = option.short === undefined ? `--${option.name}` : `-${option.short}, --${option.name}`;
  if (option.type === "boolean") return names;
  const text = option.path ? (option.path.dir ? "DIRECTORY" : "FILE") : "TEXT";
  const fallback = option.type === "int" ? "INTEGER" : option.type === "float" ? "FLOAT" : text;
  return `${names} ${option.metavar ?? fallback}`;
}

/** Return `rows` as two aligned columns, each row indented by two spaces. */
function table(rows: [string, string][]): string {
  const width = Math.max(...rows.map(([label]) => label.length));
  return rows.map(([label, help]) => `  ${label.padEnd(width)}  ${help}`.trimEnd()).join("\n");
}

/** Return the help text of `spec`: the usage line, the summary, the arguments and the options. */
export function helpText(spec: CommandSpec): string {
  const parts = [usageLine(spec), `  ${spec.summary}`];
  if (spec.positionals.length > 0) {
    parts.push(`Arguments:\n${table(spec.positionals.map((p) => [metavar(p), p.help]))}`);
  }
  const options: [string, string][] = spec.options.map((option) => [optionLabel(option), option.help]);
  options.push(["--help", "Show this message and exit."]);
  parts.push(`Options:\n${table(options)}`);
  return `${parts.join("\n\n")}\n`;
}

/** Return the text a command prints on stderr for `error`: the usage line, a hint, the message. */
export function usageError(spec: CommandSpec, error: UsageError): string {
  return `${usageLine(spec)}\nTry 'yian ${spec.name} --help' for help.\n\nError: ${error.message}\n`;
}

/**
 * Write `document` to `out` as one `yian.cli.v1` JSON object and a newline.
 *
 * `format` comes first; the other keys follow sorted, each value written as a run-log line writes
 * it (`logJson`: a `bigint` as an int, a number as a float, infinity as the `$float` tag).
 *
 * @throws {TypeError} If `document` holds a value JSON can't represent, such as NaN.
 */
export async function printJson(document: Readonly<Record<string, unknown>>, out: Writer): Promise<void> {
  const doc: Record<string, unknown> = { format: JSON_FORMAT, ...document };
  const keys = ["format", ...Object.keys(doc).filter((key) => key !== "format").sort(compareCodeUnits)];
  // Key by key: a whole-document `logJson` would sort `format` among the other keys.
  const fields = keys.map((key) => `${logJson(key)}:${logJson(doc[key])}`);
  await out.write(`{${fields.join(",")}}\n`);
}

/** A provider key that is required but not set, and that no prompt supplied. */
export class MissingKeyError extends Error {
  override name = "MissingKeyError";
}

/** The key and endpoint helpers' arguments: the command's environment and terminal, if any. */
export interface PromptContext {
  env: CommandEnv;
  terminal: Pick<Terminal, "askText"> | null;
}

/**
 * Return the key `provider`'s saved login or environment supplies, prompting if neither does.
 *
 * A saved login resolves its stored credential, so it never prompts: an API-key login returns its
 * key, a subscription `null`. Otherwise the provider's key variable is checked with `ensureEnvKey`.
 *
 * @param provider The LLM provider name (such as `"anthropic"`), matched case-insensitively.
 * @returns The key, or `null` for a saved subscription or a provider that needs no key.
 * @throws {AuthError} If the saved login can't be read.
 * @throws {RangeError} If the saved logins or the default-provider environment are invalid.
 * @throws {MissingKeyError} As `ensureEnvKey` does, when there is no saved login.
 */
export async function ensureApiKey(provider: string, context: PromptContext): Promise<string | null> {
  const name = provider.toLowerCase();
  const { connections } = connectionSnapshot({ env: context.env.read });
  const connection = Object.hasOwn(connections, name) ? connections[name] : undefined;
  if (connection?.source === "store") {
    const credentials = await connectionCredentials(connection, name, { env: context.env.read });
    return credentials instanceof ApiKeyCredentials ? credentials.apiKey : null;
  }
  return ensureEnvKey(provider, context);
}

/**
 * Return `provider`'s API key from its environment variable, prompting for it if unset.
 *
 * Saved logins are not consulted. A keyless provider (`ollama`) or an unknown one has no key
 * variable, so `null` is returned; so is an optional key (`vllm`) left unset. A set variable's
 * value is returned as is. When it is unset and a terminal is given, the user is asked; the answer
 * is stripped and `set` in the command's environment so the run's clients read it. An empty answer
 * is a fingerslip and asks again, up to `current().maxKeyPrompts` times.
 *
 * @param provider The LLM provider name (such as `"anthropic"`), matched case-insensitively.
 * @returns The key, or `null` when the provider needs none.
 * @throws {MissingKeyError} If the provider needs a key, none is set, and there is no terminal,
 *   the user cancels the prompt, or every attempt was empty.
 */
export async function ensureEnvKey(provider: string, { env, terminal }: PromptContext): Promise<string | null> {
  const envVar = getApiKeyEnv(provider);
  if (envVar === null) return null; // keyless or unknown: nothing to ensure
  const existing = env.read(envVar);
  if (existing) return existing;
  if (OPTIONAL_API_KEY_PROVIDERS.has(provider.toLowerCase())) return null;
  if (terminal === null) {
    throw new MissingKeyError(
      `provider ${reprText(provider)} needs an API key but $${envVar} is not set ` +
        `(set it in the environment, or run interactively to be prompted). ` +
        `Or run 'yian auth ${provider.toLowerCase()}' to save a login`,
    );
  }
  // An empty answer can't be a wrong key (keys aren't checked here), so it asks again, capped so
  // a caller feeding empty answers forever can't hang. A cancel ends at once.
  const tries = current().maxKeyPrompts;
  for (let attempt = 0; attempt < tries; attempt++) {
    const answer = await terminal.askText(`Enter ${envVar} for provider ${reprText(provider)}:`, { secret: true });
    if (answer === null) throw new MissingKeyError(`no API key provided for ${reprText(provider)} ($${envVar})`);
    const key = answer.trim();
    if (key) {
      env.set(envVar, key);
      return key;
    }
  }
  throw new MissingKeyError(`no API key provided for ${reprText(provider)} ($${envVar}) after ${tries} attempts`);
}

/**
 * Confirm (or override) the Ollama base URL, prompting when a terminal is given.
 *
 * The current endpoint is `$OLLAMA_BASE_URL` if set, else `http://localhost:11434`. With a
 * terminal it is offered as an editable answer, and the confirmed endpoint is `set` as
 * `OLLAMA_BASE_URL` so the client resolves the same one; an empty answer keeps the current value.
 * Without one the current value is returned.
 */
export async function confirmOllamaEndpoint({ env, terminal }: PromptContext): Promise<string> {
  const endpoint = env.read("OLLAMA_BASE_URL") || OLLAMA_DEFAULT_BASE_URL;
  if (terminal === null) return endpoint;
  const answer = await terminal.askText("Ollama endpoint:", { initial: endpoint });
  const confirmed = (answer || endpoint).trim();
  env.set("OLLAMA_BASE_URL", confirmed);
  return confirmed;
}

/**
 * Return a system error's reason as Python's `strerror` words it: Node's message without its
 * `CODE: ` prefix and `, <syscall> '<path>'` suffix, first letter capitalised
 * (`No such file or directory`).
 */
export function osReason(error: NodeJS.ErrnoException): string {
  let reason = error.message;
  const prefix = `${error.code}: `;
  if (reason.startsWith(prefix)) reason = reason.slice(prefix.length);
  const at = error.syscall === undefined ? -1 : reason.indexOf(`, ${error.syscall}`);
  if (at !== -1) reason = reason.slice(0, at);
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}
