/**
 * Provider setup over staged authentication and an atomic store, `llm_config` checks, the provider
 * list, and model listings.
 *
 * `auth` works on the store at `--auth-file`, `$YIAN_AUTH_FILE` or the default path; with `--json`,
 * its removal prints one `yian.cli.v1` document and its OpenAI device sign-in prints JSON lines.
 * `configCheck` checks a host `llm_config` layer read from stdin, `providers` lists the providers
 * and the login kinds `auth` can save for each, and `models` (`yian models`) lists the models an
 * OpenAI subscription login or a provider's key offers.
 */

import { spawn } from "node:child_process";
import type { EventEmitter } from "node:events";

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { collect } from "../llm/events.ts";
import { Message, Request, TextPart } from "../llm/messages.ts";
import { apiKeyEnvNames, OPTIONAL_API_KEY_PROVIDERS, PROVIDER_API_KEY_ENV } from "../llm_clients/api_key_env.ts";
import { ApiKeyCredentials, AuthError, AuthStore, type OAuthCredentials, OpenAIAuth } from "../llm_clients/auth/index.ts";
import {
  LLM_CONFIG_KEYS,
  LLMConfig,
  popHostCredentials,
  SUPPORTED_LLM_PROVIDERS,
  validateBaseUrl,
} from "../llm_clients/config.ts";
import {
  validateHeaderName,
  validateHeaders,
  validateHeaderValue,
  validateOrganization,
} from "../llm_clients/connection_settings.ts";
import { callCredentials, ConnectionSnapshot } from "../llm_clients/connection.ts";
import { chatModelFromConfig, warnOnce } from "../llm_clients/factory.ts";
import { listModels, MODEL_LIST_TIMEOUT_SECONDS, ModelListError } from "../llm_clients/model_list.ts";
import { ValidationError } from "../llm_clients/provider_blocks.ts";
import { redactKeys } from "../llm_clients/redact.ts";
import { realScheduler, type Scheduler } from "../platform.ts";
import { isRecord } from "../typesys/index.ts";
import { parseJson } from "../typesys/json.ts";
import { Terminal } from "./terminal.ts";
import { type CliIo, type ParsedArgs, printJson, stopOnSignals, UsageError, type Writer } from "./utils.ts";

const KINDS: Readonly<Record<string, string>> = { api_key: "api-key", subscription: "subscription", none: "none" };
const CANCELLED = "Authentication cancelled; the previous login is unchanged";
const NO_DEFAULT = "No default provider is saved; choose one with 'yian auth <provider> --default'";
// Environment overrides of a local server's URL, read the way its client reads them.
const BASE_URL_ENV: Readonly<Record<string, string>> = { ollama: "OLLAMA_BASE_URL", vllm: "VLLM_BASE_URL" };
// Variables the provider's SDK reads to send runs to another host than the one listed; a gateway's
// key must not go to the public host's listing.
const ELSEWHERE_ENV: Readonly<Record<string, readonly string[]>> = {
  anthropic: ["ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"],
  google: ["GOOGLE_GEMINI_BASE_URL"],
};
const SETTINGS_NEED_API_KEY = "--organization, --project and --header apply to API-key authentication";
/** The message for an unknown top-level `llm_config` key, which never names the key. */
export const EXTRA_FIELD = `llm_config has an extra field; the fields are ${[...LLM_CONFIG_KEYS].sort(compareCodeUnits).join(", ")}`;

/** The prompts `auth` asks on a terminal. */
type AuthTerminal = Pick<Terminal, "askText" | "askChoice" | "close">;

/** What the handlers reach the outside through; each defaults to the real one. Tests pass fakes. */
export interface AuthDeps {
  /** Build the OpenAI authentication service; defaults to `new OpenAIAuth(options)`. */
  openAIAuth?: (options: {
    signal?: AbortSignal;
    requestTimeoutMs?: number;
  }) => Pick<OpenAIAuth, "apiModels" | "browserLogin" | "deviceLogin" | "refresh" | "subscriptionModels">;
  /** List a provider's models; defaults to `listModels`. */
  listModels?: typeof listModels;
  /** Build the model the reply check prompts; defaults to `chatModelFromConfig`. */
  chatModel?: typeof chatModelFromConfig;
  /**
   * The terminal `auth` prompts on, which it closes when done; `null` for none. Defaults to a
   * `Terminal` over stdin when stdin is a TTY, else none.
   */
  terminal?: AuthTerminal | null;
  /** Show the browser sign-in URL and open it; defaults to `openBrowser` on stdout. */
  openUrl?: (url: string) => Promise<void> | void;
  /** The clock deadlines run on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** Where `auth` listens for SIGINT (and SIGTERM, for the JSON device sign-in); defaults to `process`. */
  signals?: Pick<EventEmitter, "on" | "off">;
}

/** The real OpenAI authentication service, the default of `AuthDeps.openAIAuth`. */
const newOpenAIAuth: NonNullable<AuthDeps["openAIAuth"]> = (options) => new OpenAIAuth(options);

/** What a setup step reaches the outside through. */
interface SetupContext {
  io: CliIo;
  deps: AuthDeps;
  /** The terminal to prompt on; `null` when no interactive terminal is available. */
  terminal: AuthTerminal | null;
  /** The command's stop, which cancels every prompt and request. */
  signal: AbortSignal;
}

/** A deadline passed before the work it bounds was done. */
class TimeoutError extends Error {
  override name = "TimeoutError";
}

/**
 * Run `work` with a signal that aborts with `reason` once `ms` milliseconds pass on `scheduler`,
 * and return its result. The timer ends with the work, so none outlives it.
 */
async function withDeadline<T>(
  ms: number,
  reason: Error,
  scheduler: Scheduler,
  work: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const deadline = new AbortController();
  const cancel = new AbortController();
  // A cancelled sleep rejects; the handler keeps that rejection from ending the process.
  void scheduler.sleep(ms, cancel.signal).then(
    () => deadline.abort(reason),
    () => {},
  );
  try {
    return await work(deadline.signal);
  } finally {
    cancel.abort();
  }
}

/** Read `stdin` to its end and return its bytes. */
async function readStdin(stdin: CliIo["stdin"]): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stdin) chunks.push(Buffer.from(chunk as string | Uint8Array));
  return Buffer.concat(chunks);
}

/** Decode `bytes` as strict UTF-8, a BOM kept as text; `null` for bytes that aren't UTF-8. */
function stdinText(bytes: Buffer): string | null {
  try {
    return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    return null;
  }
}

/**
 * Refuse a prompt when no interactive terminal is available, and return the terminal.
 *
 * @throws {AuthError} No terminal is available; the message names `option`.
 */
function requireTerminal(option: string, ctx: SetupContext): AuthTerminal {
  if (ctx.terminal === null) {
    throw new AuthError(`Authentication needs ${option} when no interactive terminal is available`);
  }
  return ctx.terminal;
}

/** Select a value on a terminal, or require an explicit command option. */
async function choose(
  message: string,
  choices: readonly string[],
  options: { initial: string; option: string },
  ctx: SetupContext,
): Promise<string> {
  const terminal = requireTerminal(options.option, ctx);
  const answer = await terminal.askChoice(message, choices, { initial: options.initial, signal: ctx.signal });
  if (answer === null) throw new AuthError(CANCELLED);
  return answer;
}

/** Read a typed value on a terminal, or require an explicit command option. */
async function ask(message: string, option: string, ctx: SetupContext): Promise<string> {
  const terminal = requireTerminal(option, ctx);
  const answer = await terminal.askText(message, { signal: ctx.signal });
  if (answer === null || !answer.trim()) throw new AuthError(CANCELLED);
  return answer.trim();
}

/**
 * Read the provider's key from the first set variable of `variables`, or a hidden prompt.
 *
 * @param provider Provider name shown in the prompt.
 * @param variables Environment variables checked first, in order.
 * @param optional The provider can run without a key; off a terminal or with an empty answer, no
 *   key is returned.
 * @returns The key, or `null` when an optional key is not given.
 * @throws {AuthError} A required key is missing, or the prompt is cancelled.
 */
async function apiKey(
  provider: string,
  variables: readonly string[],
  ctx: SetupContext,
  { optional = false }: { optional?: boolean } = {},
): Promise<string | null> {
  for (const variable of variables) {
    const key = ctx.io.env(variable);
    if (key) return key;
  }
  if (ctx.terminal === null) {
    if (optional) return null;
    throw new AuthError(`Set ${variables.join(" or ")} before API-key setup, or run in an interactive terminal`);
  }
  const hint = optional ? " (leave empty for none)" : "";
  const answer = await ctx.terminal.askText(`${provider} API key${hint}:`, { secret: true, signal: ctx.signal });
  if (answer === null) throw new AuthError(CANCELLED);
  if (!answer.trim()) {
    if (optional) return null;
    throw new AuthError("Authentication cancelled; no API key was supplied");
  }
  return answer.trim();
}

/** Return what `fn` returns; a `RangeError` it throws is rethrown as an `AuthError`, its message after `prefix`. */
function asAuthError<T>(prefix: string, fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    if (error instanceof RangeError) throw new AuthError(prefix + error.message);
    throw error;
  }
}

/**
 * Check each `--header` and read the values given through environment variables.
 *
 * `NAME=VARIABLE` reads the value from that variable; `NAME` alone leaves it to a hidden prompt,
 * which needs a terminal. Messages never contain a value or the variable's name, since a mistyped
 * option may put a value where the variable goes.
 *
 * @param specs The `--header` options as given.
 * @param provider The provider the headers go to.
 * @returns Each header name and its value, or `null` to prompt.
 * @throws {AuthError} A name is not an HTTP token, is refused or repeated (names ignore case), a
 *   variable is empty or unset, a value is not allowed, or a value needs a prompt and no terminal
 *   is available.
 */
function headerSources(specs: readonly string[], provider: string, ctx: SetupContext): Map<string, string | null> {
  const sources = new Map<string, string | null>();
  const seen = new Set<string>();
  for (const spec of specs) {
    const split = spec.indexOf("=");
    const name = split === -1 ? spec : spec.slice(0, split);
    const variable = split === -1 ? null : spec.slice(split + 1);
    asAuthError("Invalid --header: ", () => validateHeaderName(name, provider));
    if (seen.has(name.toLowerCase())) throw new AuthError(`--header ${name} is given more than once (names ignore case)`);
    seen.add(name.toLowerCase());
    if (variable === null) {
      if (ctx.terminal === null) {
        throw new AuthError(
          `--header ${name} reads its value from a hidden prompt, which needs a terminal; ` +
            "use --header NAME=VARIABLE to read it from a variable",
        );
      }
      sources.set(name, null);
      continue;
    }
    if (!variable) throw new AuthError(`--header ${name}=VARIABLE needs an environment variable name`);
    const value = ctx.io.env(variable);
    if (!value) throw new AuthError(`The environment variable given for --header ${name} is not set`);
    sources.set(name, asAuthError("Invalid --header: ", () => validateHeaderValue(name, value)));
  }
  return sources;
}

/**
 * Prompt, hidden, for each header value not read yet, and check all of them.
 *
 * @param sources From `headerSources`.
 * @param provider The provider the headers go to.
 * @returns Each header name and its value.
 * @throws {AuthError} A prompt is cancelled or left empty, or a value or the number of headers is
 *   not allowed.
 */
async function headers(
  sources: ReadonlyMap<string, string | null>,
  provider: string,
  ctx: SetupContext,
): Promise<Record<string, string>> {
  const values: Record<string, string> = {};
  for (let [name, value] of sources) {
    if (value === null) {
      // headerSources refused a bare header without a terminal.
      value = await ctx.terminal!.askText(`${name} header value:`, { secret: true, signal: ctx.signal });
      if (value === null) throw new AuthError(CANCELLED);
      if (!value) throw new AuthError(`Authentication cancelled; no value was given for ${name}`);
    }
    values[name] = value;
  }
  return asAuthError("Invalid --header: ", () => validateHeaders(values, provider));
}

/** Start a program that opens a URL; `spawn` by default. */
type Launch = (
  command: string,
  args: readonly string[],
  options: { detached: boolean; stdio: "ignore" },
) => Pick<EventEmitter, "once"> & { unref(): void };

/**
 * Present the login URL on `out` and open it with the host's browser.
 *
 * The platform's opener (`open`, `rundll32 url.dll,FileProtocolHandler` or `xdg-open`) gets the URL
 * as an argument, never through a shell, and runs detached. Resolves once it has started or failed
 * to: the printed URL stays usable either way.
 */
export async function openBrowser(
  url: string,
  out: Writer,
  { platform = process.platform, launch = spawn }: { platform?: NodeJS.Platform; launch?: Launch } = {},
): Promise<void> {
  await out.write(`Open this URL to sign in:\n${url}\n`);
  const [command, args] =
    platform === "darwin"
      ? ["open", [url]]
      : platform === "win32"
        ? ["rundll32", ["url.dll,FileProtocolHandler", url]]
        : ["xdg-open", [url]];
  try {
    const child = launch(command, args, { detached: true, stdio: "ignore" });
    child.unref();
    await new Promise<void>((resolve) => {
      child.once("spawn", () => resolve());
      // Also the listener a later error finds, so it doesn't end the process.
      child.once("error", () => resolve());
    });
  } catch {
    // The displayed URL remains usable when the opener can't start.
  }
}

/** Offer general text-model families from the authenticated API catalog. */
function textModels(models: readonly string[]): string[] {
  return models.filter(
    (name) =>
      ["gpt-", "chatgpt-", "o1", "o3", "o4"].some((prefix) => name.startsWith(prefix)) &&
      !["realtime", "audio", "image", "transcribe", "tts"].some((part) => name.includes(part)),
  );
}

/** Print each saved login in `store` and the default without reading any credential. */
async function list(store: AuthStore, io: CliIo): Promise<void> {
  const state = store.read();
  const overrides = ["YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"].filter((name) => io.env(name));
  if (state.selections.length === 0) {
    await io.stdout.write("No login is saved; run 'yian auth <provider>' to add one\n");
  } else {
    await io.stdout.write("Saved logins:\n");
    for (const selection of state.selections) {
      const details = [KINDS[selection.method], `model ${selection.model}`];
      if (selection.endpoint) details.push(`endpoint ${selection.endpoint}`);
      if (selection.organization) details.push(`organization ${selection.organization}`);
      if (selection.project) details.push(`project ${selection.project}`);
      // Header values are secrets; only their names are shown.
      if (selection.headerNames.length > 0) details.push(`headers ${selection.headerNames.join(", ")}`);
      const defaultMark = selection.provider === state.defaultProvider ? " (default)" : "";
      await io.stdout.write(`  ${selection.provider}: ${details.join(", ")}${defaultMark}\n`);
    }
    if (state.defaultProvider === null) await io.stdout.write(`${NO_DEFAULT}\n`);
  }
  if (overrides.length > 0 && state.selections.length > 0) {
    const [is, override] = overrides.length === 1 ? ["is", "overrides"] : ["are", "override"];
    await io.stdout.write(`${overrides.join(" and ")} ${is} set in the environment and ${override} the saved choice\n`);
  }
}

/**
 * Refuse an Azure endpoint that a config's `base_url` could not name.
 *
 * @throws {AuthError} The endpoint is invalid.
 */
function checkEndpoint(endpoint: string): void {
  asAuthError("Invalid Azure endpoint: ", () => validateBaseUrl(endpoint));
}

/** The options of `yian auth` that `checkOptions` checks; an option not given is `undefined`. */
interface AuthOptions {
  method?: string;
  login?: string;
  model?: string;
  apiKeyEnv?: string;
  endpoint?: string;
  organization?: string;
  project?: string;
  header: readonly string[];
  default: boolean;
  remove: boolean;
}

/**
 * Validate the option combination and return the provider name and headers to use.
 *
 * Azure setup also needs `OPENAI_API_VERSION` and a valid `endpoint`, when given. `organization`
 * and `project` apply to an OpenAI API key, and `header` to every provider sent a key (see
 * `headerSources`).
 *
 * @returns The provider name, and each header name with its value from the environment or `null`
 *   to prompt.
 * @throws {AuthError} The provider or an option does not apply, a connection setting is invalid,
 *   or Azure setup lacks its API version or has an invalid endpoint.
 */
function checkOptions(provider: string, options: AuthOptions, ctx: SetupContext): [string, Map<string, string | null>] {
  const { method, login, model, apiKeyEnv, endpoint, organization, project, header } = options;
  const setup: [string, unknown][] = [
    ["--method", method],
    ["--login", login],
    ["--model", model],
    ["--api-key-env", apiKeyEnv],
    ["--endpoint", endpoint],
    ["--organization", organization],
    ["--project", project],
    ["--header", header.length > 0 ? header : undefined],
  ];
  const given = setup.filter(([, value]) => value !== undefined).map(([option]) => option);
  if (options.default && options.remove) throw new AuthError("Use either --default or --remove, not both");
  if ((options.default || options.remove) && given.length > 0) {
    const action = options.default ? "--default" : "--remove";
    throw new AuthError(`${action} does not take ${given.join(", ")}`);
  }
  // A login saved by another yian version may name any provider.
  if (options.remove) return [provider, new Map()];
  const name = provider.toLowerCase();
  if (!(SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(name)) {
    throw new AuthError(`Unsupported provider ${reprText(provider)}; choose one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`);
  }
  if (name !== "openai" && (method !== undefined || login !== undefined)) {
    throw new AuthError("--method and --login apply only to openai");
  }
  if (name !== "azure" && endpoint !== undefined) throw new AuthError("--endpoint applies only to azure");
  if (apiKeyEnv !== undefined && !apiKeyEnv.trim()) throw new AuthError("--api-key-env needs an environment variable name");
  if (name === "ollama" && apiKeyEnv !== undefined) throw new AuthError("ollama takes no API key; omit --api-key-env");
  if (method !== undefined && method !== "subscription" && method !== "api-key") {
    throw new AuthError("Choose --method subscription or --method api-key");
  }
  if (login !== undefined && login !== "browser" && login !== "device") {
    throw new AuthError("Choose --login browser or --login device");
  }
  if (method === "api-key" && login !== undefined) throw new AuthError("--login applies to subscription authentication");
  if (organization !== undefined || project !== undefined) {
    if (name !== "openai") throw new AuthError("--organization and --project apply only to openai");
    if (method === "subscription") throw new AuthError("--organization and --project apply to API-key authentication");
    for (const [field, value] of [
      ["organization", organization],
      ["project", project],
    ] as const) {
      if (value === undefined) continue;
      asAuthError(`Invalid --${field}: `, () => validateOrganization(value, field));
    }
  }
  if (header.length > 0) {
    if (name === "ollama") throw new AuthError("ollama takes no API key, so it takes no --header");
    if (method === "subscription") throw new AuthError("--header applies to API-key authentication");
  }
  if (name === "azure" && !options.default) {
    // The Azure client reads the API version only from the environment.
    if (!ctx.io.env("OPENAI_API_VERSION")) {
      throw new AuthError("Azure needs OPENAI_API_VERSION set to its API version (runs read it too)");
    }
    if (endpoint !== undefined) checkEndpoint(endpoint);
  }
  return [name, headerSources(header, name, ctx)];
}

/**
 * Sign in to OpenAI, choose an available model, and save the login.
 *
 * An API key is saved with the given organization, project and headers, which its model listing
 * sends too; a subscription takes none of them.
 *
 * @throws {AuthError} Sign-in, a setting, the listing, or the model choice fails.
 */
async function setupOpenai(
  store: AuthStore,
  method: string | undefined,
  login: string | undefined,
  model: string | undefined,
  apiKeyEnv: string,
  makeDefault: boolean,
  organization: string | undefined,
  project: string | undefined,
  sources: ReadonlyMap<string, string | null>,
  ctx: SetupContext,
): Promise<void> {
  method ??= await choose(
    "OpenAI authentication method:",
    ["subscription", "api-key"],
    { initial: "subscription", option: "--method" },
    ctx,
  );
  if (method === "subscription" && (organization || project || sources.size > 0)) {
    throw new AuthError(SETTINGS_NEED_API_KEY);
  }
  const service = (ctx.deps.openAIAuth ?? newOpenAIAuth)({ signal: ctx.signal });
  let credentials: OAuthCredentials | ApiKeyCredentials;
  let models: readonly string[];
  let choices: readonly string[];
  if (method === "subscription") {
    login ??= await choose(
      "Subscription sign-in:",
      ["browser", "device"],
      { initial: ctx.io.env("SSH_CONNECTION") ? "device" : "browser", option: "--login" },
      ctx,
    );
    const openUrl = ctx.deps.openUrl ?? ((url: string) => openBrowser(url, ctx.io.stdout));
    credentials =
      login === "device"
        ? await service.deviceLogin((url, code) => ctx.io.stdout.write(`Open ${url} and enter code: ${code}\n`))
        : await service.browserLogin(openUrl);
    models = await service.subscriptionModels(credentials);
    choices = models;
  } else {
    if (login !== undefined) throw new AuthError("--login applies to subscription authentication");
    // A required key is never null.
    const key = (await apiKey("openai", [apiKeyEnv], ctx))!;
    const values = await headers(sources, "openai", ctx);
    const apiCredentials = new ApiKeyCredentials(key, values);
    credentials = apiCredentials;
    try {
      // Runs with this login go to OPENAI_BASE_URL when it is set, so the key must go there too.
      const endpoint = ctx.io.env("OPENAI_BASE_URL") || undefined;
      models = await service.apiModels(apiCredentials, { organization, project, endpoint });
    } catch (error) {
      // The messages are fixed today; scrubbed so no sent secret can reach output.
      if (!(error instanceof AuthError)) throw error;
      throw new AuthError(redactKeys(error.message, [key, ...Object.values(values)]));
    }
    choices = textModels(models);
  }
  if (model === undefined) {
    if (choices.length === 0) {
      throw new AuthError("No general text models were returned; use --model with an available text model");
    }
    model = await choose("Model:", choices, { initial: choices[0]!, option: "--model" }, ctx);
  }
  if (!models.includes(model)) throw new AuthError("The selected model is not available for this connection");
  await store.save("openai", credentials, model, { organization, project, makeDefault });
  await ctx.io.stdout.write(`Saved OpenAI ${method} authentication with model ${model}\n`);
}

/**
 * Sign in to an OpenAI subscription by device code, writing JSON lines, and save the login.
 *
 * Writes a `code` line, a `waiting` line, and a `done` line once the login is saved with `model`,
 * or the subscription's first model; it becomes the default when the store has none. Aborting
 * `signal` stops the sign-in before the store's rename, rejecting with its reason: nothing is saved
 * and no further line is written.
 *
 * @throws {AuthError} The store is unusable, or sign-in, the listing or the model choice fails.
 */
async function signInLines(
  authFile: string | null,
  model: string | undefined,
  io: CliIo,
  deps: AuthDeps,
  signal: AbortSignal,
): Promise<void> {
  const store = new AuthStore(authFile, { env: io.env, signal });
  // Reading first also diagnoses an unusable store before any request.
  const makeDefault = store.read().defaultProvider === null;
  const service = (deps.openAIAuth ?? newOpenAIAuth)({ signal });
  const credentials = await service.deviceLogin(async (url, code) => {
    await printJson({ event: "code", user_code: code, verification_uri: url }, io.stdout);
    await printJson({ event: "waiting" }, io.stdout);
  });
  const models = await service.subscriptionModels(credentials);
  const chosen = model ?? models[0];
  if (chosen === undefined || !models.includes(chosen)) {
    throw new AuthError("The selected model is not available for this connection");
  }
  await store.save("openai", credentials, chosen, { makeDefault });
  await printJson({ event: "done", models: [...models], model: chosen }, io.stdout);
}

// The texts the reply check's other warnings have emitted, so each shows once per process.
const WARNED = new Set<string>();

/**
 * Send the model one short prompt and say why it failed, if it did.
 *
 * The model is built from the given key, endpoint and headers only, over an empty snapshot, so no
 * saved login or default selection takes part. Without a key, vLLM gets the dummy key its client
 * sends to a keyless server, and Ollama sends none. The build and the call get 30 seconds together
 * on `deps.scheduler`.
 *
 * @param provider Provider name.
 * @param model Model id, or the Azure deployment name.
 * @param key API key to send, or `null` for none.
 * @param endpoint Azure resource endpoint, or `null`.
 * @param headers Connection headers sent beside the key.
 * @param stop The command's stop.
 * @returns `null` when the model replied; otherwise the error's name and a short message without
 *   any credential the call sent.
 * @throws The stop's reason when `stop` aborts during the check.
 */
export async function modelAnswers(
  provider: string,
  model: string,
  key: string | null,
  endpoint: string | null,
  headers: Readonly<Record<string, string>>,
  io: CliIo,
  deps: AuthDeps,
  stop: AbortSignal,
): Promise<string | null> {
  const config: Record<string, unknown> = { provider, model };
  if (key !== null) {
    config["api_key"] = key;
  } else if (OPTIONAL_API_KEY_PROVIDERS.has(provider)) {
    // Passed explicitly so an ambient VLLM_API_KEY doesn't stand in for the keyless login being
    // saved.
    config["api_key"] = "EMPTY";
  }
  if (endpoint !== null) config["base_url"] = endpoint;
  if (Object.keys(headers).length > 0) config["headers"] = { ...headers };
  const timedOut = new TimeoutError("The model didn't reply within 30 seconds");
  try {
    await withDeadline(30_000, timedOut, deps.scheduler ?? realScheduler, async (deadline) => {
      try {
        const signal = AbortSignal.any([stop, deadline]);
        const chat = await (deps.chatModel ?? chatModelFromConfig)(config, {
          env: io.env,
          snapshot: new ConnectionSnapshot({ defaults: {}, providerModels: {}, connections: {} }),
          signal,
          onWarning: (message) => {
            // The reply decides whether the model is usable, not yian's catalog.
            if (!/^Model .* is not in the known model list/s.test(message)) warnOnce(WARNED, message);
          },
        });
        const prompt = new Message({ role: "user", parts: [new TextPart("Reply with OK.")] });
        await collect(chat.stream(new Request({ messages: [prompt] }), { signal }));
      } catch (error) {
        // Whatever a call cut short by the deadline rejects with, the deadline is the cause.
        throw deadline.aborted && !stop.aborted ? timedOut : error;
      }
    });
  } catch (error) {
    // A stopped call throws the signal's reason: the stop's is a cancel, rethrown; the deadline's
    // is `timedOut`, reported below like any failure.
    if (stop.aborted) throw error;
    // callCredentials also lists a set token variable (AZURE_OPENAI_AD_TOKEN, ANTHROPIC_AUTH_TOKEN),
    // though a call given a key doesn't send it. A keyless check sends no saved or ambient key, so
    // none is looked up.
    const sent = { provider, api_key: key, headers: { ...headers } };
    const secrets = key === null ? Object.values(headers) : callCredentials(sent, { env: io.env });
    const message = error instanceof Error ? error.message : String(error);
    const folded = redactKeys(message, secrets).split(/\s+/).filter(Boolean).join(" ");
    // Cut by code points, so no surrogate pair splits.
    const text = Array.from(folded).slice(0, 200).join("");
    return error instanceof Error ? `${error.name}: ${text}` : text;
  }
  return null;
}

/**
 * Read a provider's key, endpoint, and model, check the model, and save the login.
 *
 * A model in the provider's listing is accepted as is. Any other model, every Azure deployment, and
 * every model when a variable in `ELSEWHERE_ENV` sends runs to another host, must answer one short
 * prompt. vLLM's key is optional unless `apiKeyEnv` names its variable or headers are given, since
 * headers are sent only beside a key. The listing and the reply check send the headers, which are
 * saved with the key.
 *
 * @throws {AuthError} A value is missing or invalid, or the model did not answer.
 */
async function setupProvider(
  store: AuthStore,
  provider: string,
  model: string | undefined,
  apiKeyEnv: string | undefined,
  endpoint: string | undefined,
  makeDefault: boolean,
  sources: ReadonlyMap<string, string | null>,
  ctx: SetupContext,
): Promise<void> {
  if (provider === "azure" && endpoint === undefined) {
    endpoint = await ask("Azure OpenAI endpoint (https://<resource>.openai.azure.com/):", "--endpoint", ctx);
    checkEndpoint(endpoint);
  }
  let key: string | null = null;
  if (provider !== "ollama") {
    // Every provider but ollama has a key variable.
    const variables = apiKeyEnv === undefined ? apiKeyEnvNames(provider) : [apiKeyEnv];
    const optional = OPTIONAL_API_KEY_PROVIDERS.has(provider) && apiKeyEnv === undefined && sources.size === 0;
    key = await apiKey(provider, variables, ctx, { optional });
  }
  const values = await headers(sources, provider, ctx);
  let models: readonly string[] | null = null;
  if (provider === "azure") {
    model ||= await ask("Deployment name:", "--model", ctx);
  } else {
    if (model === undefined) requireTerminal("--model", ctx); // Before the listing request.
    const elsewhere = (ELSEWHERE_ENV[provider] ?? []).find((name) => ctx.io.env(name));
    const variable = BASE_URL_ENV[provider];
    const baseUrl = (variable && ctx.io.env(variable)) || null;
    let failure: string | null = null;
    if (elsewhere === undefined) {
      try {
        models = await (ctx.deps.listModels ?? listModels)(provider, {
          apiKey: key,
          baseUrl,
          headers: values,
          signal: ctx.signal,
        });
      } catch (error) {
        if (!(error instanceof ModelListError)) throw error;
        // The reasons are fixed today; scrubbed so no sent secret can reach output.
        failure = redactKeys(error.message, [key, ...Object.values(values)]);
      }
    }
    if (model === undefined) {
      if (elsewhere !== undefined) {
        await ctx.io.stdout.write(
          `${elsewhere} is set, so runs use another host than the ${provider} listing; type a model name\n`,
        );
        model = await ask("Model:", "--model", ctx);
      } else if (models !== null && models.length > 0) {
        model = await choose("Model:", models, { initial: models[0]!, option: "--model" }, ctx);
      } else {
        await ctx.io.stdout.write(
          `Couldn't list the ${provider} models (${failure || "none were returned"}); type a model name\n`,
        );
        model = await ask("Model:", "--model", ctx);
      }
    }
  }
  if (models === null || !models.includes(model)) {
    const problem = await modelAnswers(provider, model, key, endpoint ?? null, values, ctx.io, ctx.deps, ctx.signal);
    if (problem !== null) throw new AuthError(`The model didn't answer: ${problem}; the previous login is unchanged`);
  }
  await store.save(provider, key ? new ApiKeyCredentials(key, values) : null, model, { endpoint, makeDefault });
  await ctx.io.stdout.write(`Saved ${provider} login (${key ? "api-key" : "none"}) with model ${model}\n`);
}

/**
 * List, set up, choose the default among, or remove saved provider logins.
 *
 * `yian auth` lists the saved logins and the default.
 *
 * `yian auth PROVIDER` saves the provider's credential and model; it becomes the default only when
 * no default is saved. openai signs in with a subscription or an API key. Other providers read the
 * key from their environment variable or a hidden prompt; ollama takes none, and vllm's is optional
 * unless `--api-key-env` names its variable. A model the provider doesn't list, and every Azure
 * deployment, must answer one short prompt before the login is saved; with a base-URL variable
 * such as `ANTHROPIC_BASE_URL` set, the listing is skipped and the model must answer. azure also
 * needs `OPENAI_API_VERSION` set.
 *
 * An openai API key may be saved with `--organization` and `--project`, and every provider sent a
 * key with `--header`; the model listing and the reply check send them too. Values never come from
 * the command line: `--header NAME=VARIABLE` reads one from an environment variable, and
 * `--header NAME` from a hidden prompt. Each save states them again: a save without them keeps
 * none.
 *
 * `yian auth PROVIDER --default` makes a saved login the default. `yian auth PROVIDER --remove`
 * deletes one saved login. `yian auth --remove --auth-file F` deletes the whole file F.
 *
 * `--auth-file F` makes every form use the file F instead of `$YIAN_AUTH_FILE` or
 * `~/.config/yian/auth.json`. `--json` prints one `yian.cli.v1` document on stdout for `--remove`:
 * success, or the error that failed it. For `openai --method subscription --login device` it
 * prints JSON lines: the code to enter, a waiting line, then the saved models or the error;
 * SIGTERM or SIGINT stops the sign-in, saving nothing, with exit 130.
 *
 * A failed or cancelled setup leaves the saved logins unchanged.
 *
 * @returns 0; 1 when the command fails or is cancelled (saved logins stay unchanged); 130 when a
 *   `--json` command is stopped by a signal.
 * @throws {UsageError} `--json` is given to another form.
 */
export async function auth(command: ParsedArgs, io: CliIo, deps: AuthDeps = {}): Promise<number> {
  const provider = command.args["provider"] as string | undefined;
  const o = command.options;
  const options: AuthOptions = {
    method: o["method"] as string | undefined,
    login: o["login"] as string | undefined,
    model: o["model"] as string | undefined,
    apiKeyEnv: o["api-key-env"] as string | undefined,
    endpoint: o["endpoint"] as string | undefined,
    organization: o["organization"] as string | undefined,
    project: o["project"] as string | undefined,
    header: (o["header"] as string[] | undefined) ?? [],
    default: o["default"] === true,
    remove: o["remove"] === true,
  };
  const json = o["json"] === true;
  // A removal refuses setup options, so its error is the plain document even when they describe a
  // device sign-in.
  const device =
    (provider ?? "").toLowerCase() === "openai" &&
    options.method === "subscription" &&
    options.login === "device" &&
    !options.remove;
  if (json && !(options.remove || device)) {
    throw new UsageError("Invalid value: --json applies to --remove and to openai --method subscription --login device");
  }
  const stop = new AbortController();
  // Only the JSON device sign-in handles SIGTERM; every form takes Ctrl-C.
  const names: NodeJS.Signals[] = json && device ? ["SIGTERM", "SIGINT"] : ["SIGINT"];
  const authFile = (o["auth-file"] as string | undefined) ?? null;
  const open = (path: string | null) => new AuthStore(path, { env: io.env, signal: stop.signal });
  const terminal =
    deps.terminal !== undefined ? deps.terminal : io.stdin.isTTY === true ? new Terminal({ input: io.stdin }) : null;
  const ctx: SetupContext = { io, deps, terminal, signal: stop.signal };
  // Installed right before the `try` whose `finally` removes them.
  const off = stopOnSignals(stop, deps.signals ?? process, names);
  try {
    if (provider === undefined) {
      const { method, login, model, apiKeyEnv, endpoint, organization, project } = options;
      const setup =
        [method, login, model, apiKeyEnv, endpoint, organization, project].some((value) => value !== undefined) ||
        options.header.length > 0;
      if (options.remove && authFile !== null) {
        if (options.default || setup) {
          throw new AuthError(
            "--remove without a provider deletes the whole --auth-file; it takes no --default or setup option",
          );
        }
        await open(authFile).delete();
        if (json) await printJson({ ok: true }, io.stdout);
        else await io.stdout.write(`Removed the saved logins in ${authFile}\n`);
        return 0;
      }
      if (options.default || options.remove || setup) {
        throw new AuthError("Name a provider: yian auth <provider> [options]");
      }
      await list(open(authFile), io);
      return 0;
    }
    const [checked, sources] = checkOptions(provider, options, ctx);
    let name = checked;
    if (json && device) {
      await signInLines(authFile, options.model, io, deps, stop.signal);
      return 0;
    }
    const store = open(authFile);
    if (options.remove) {
      const state = store.read();
      const saved = new Set(state.selections.map((selection) => selection.provider));
      if (!saved.has(name) && saved.has(name.toLowerCase())) name = name.toLowerCase();
      const wasDefault = state.defaultProvider === name;
      await store.remove(name);
      if (json) {
        await printJson({ ok: true }, io.stdout);
        return 0;
      }
      await io.stdout.write(`Removed the saved ${name} login\n`);
      if (wasDefault) await io.stdout.write(`${NO_DEFAULT}\n`);
      return 0;
    }
    if (options.default) {
      await store.setDefault(name);
      await io.stdout.write(`Default provider: ${name}\n`);
      return 0;
    }
    // Reading first also diagnoses an unusable store before account sign-in.
    const current = store.read().defaultProvider;
    if (name === "openai") {
      const { method, login, model, apiKeyEnv, organization, project } = options;
      const keyEnv = apiKeyEnv ?? "OPENAI_API_KEY";
      await setupOpenai(store, method, login, model, keyEnv, current === null, organization, project, sources, ctx);
    } else {
      const { model, apiKeyEnv, endpoint } = options;
      await setupProvider(store, name, model, apiKeyEnv, endpoint, current === null, sources, ctx);
    }
    if (current === null || current === name) {
      await io.stdout.write(`Default provider: ${name}\n`);
    } else {
      await io.stdout.write(`Default provider: ${current} (switch with 'yian auth ${name} --default')\n`);
    }
    return 0;
  } catch (error) {
    if (stop.signal.aborted) {
      // A stopped --json command exits 130 and writes no document.
      if (json) return 130;
      await io.stderr.write(`${CANCELLED}\n`);
      return 1;
    }
    if (!(error instanceof AuthError)) throw error;
    if (json) {
      const failure = { code: "auth.failed", message: error.message };
      // Device sign-in writes JSON lines, so its error is an event line too.
      await printJson(device ? { event: "error", error: failure } : { error: failure }, io.stdout);
    } else {
      await io.stderr.write(`${error.message}\n`);
    }
    return 1;
  } finally {
    off();
    terminal?.close();
  }
}

/** One problem in an `llm_config` layer: the field it is at (`null` for none) and what is wrong. */
export interface ConfigProblem {
  field: string | null;
  message: string;
}

/**
 * Check a host `llm_config` layer as a run does and return its errors: the check `configCheck`
 * runs.
 *
 * The credentials are popped from a copy, then the rest is validated; the first failing step's
 * errors are returned. An unknown top-level key is reported without its name, since a pasted key
 * could be the name; a provider block reports its own field problems at its block's name.
 *
 * @returns Each distinct error, where `field` is the error's location joined with `.`, or `null`
 *   for an unknown top-level key, a plain `RangeError` or a model-level error; empty when the
 *   layer is valid.
 */
export function configErrors(layer: Readonly<Record<string, unknown>>): ConfigProblem[] {
  const rest = { ...layer };
  try {
    popHostCredentials(rest);
    new LLMConfig(rest);
  } catch (error) {
    if (error instanceof ValidationError) {
      const errors: ConfigProblem[] = [];
      for (const issue of error.issues) {
        const entry =
          issue.type === "extra_forbidden" && issue.loc.length === 1
            ? { field: null, message: EXTRA_FIELD }
            : { field: issue.loc.join(".") || null, message: issue.msg.replace(/^Value error, /, "") };
        if (!errors.some((seen) => seen.field === entry.field && seen.message === entry.message)) errors.push(entry);
      }
      return errors;
    }
    if (error instanceof RangeError) return [{ field: null, message: error.message }];
    throw error;
  }
  return [];
}

/**
 * Check the host llm_config layer given as a JSON object on stdin, key included.
 *
 * The layer is checked as a run checks the llm_config it is given; values never appear in the
 * errors. Prints a `yian.cli.v1` document `{ok, errors}` with `--json`, else "ok" or one error
 * per line on stderr.
 *
 * @returns 0, or 1 when the layer has errors.
 * @throws {UsageError} stdin is not a JSON object.
 */
export async function configCheck(command: ParsedArgs, io: CliIo): Promise<number> {
  const text = stdinText(await readStdin(io.stdin));
  let layer: unknown = null;
  if (text !== null) {
    try {
      layer = parseJson(text, "plain");
    } catch {
      // Not JSON: refused below like any other non-object.
    }
  }
  if (!isRecord(layer)) throw new UsageError("Invalid value: stdin must be a JSON object (an llm_config layer)");
  const errors = configErrors(layer);
  if (command.options["json"] === true) {
    await printJson({ ok: errors.length === 0, errors }, io.stdout);
  } else if (errors.length === 0) {
    await io.stdout.write("ok\n");
  } else {
    for (const { field, message } of errors) await io.stderr.write(`${field ? `${field}: ${message}` : message}\n`);
  }
  return errors.length === 0 ? 0 : 1;
}

/**
 * List the supported providers, the login kinds each takes, and its API-key variable.
 *
 * The login kinds are those `yian auth` can save: openai also takes a subscription, ollama takes
 * no key, and vllm's key is optional.
 */
export async function providers(command: ParsedArgs, io: CliIo): Promise<number> {
  const listed = SUPPORTED_LLM_PROVIDERS.map((name) => {
    let methods: string[];
    if (name === "openai") methods = ["api-key", "subscription"];
    else if (name === "ollama") methods = ["none"];
    else if (OPTIONAL_API_KEY_PROVIDERS.has(name)) methods = ["api-key", "none"];
    else methods = ["api-key"];
    return { name, methods, api_key_env: PROVIDER_API_KEY_ENV[name] ?? null };
  });
  if (command.options["json"] === true) {
    await printJson({ providers: listed }, io.stdout);
    return 0;
  }
  for (const provider of listed) {
    const suffix = provider.api_key_env ? ` (${apiKeyEnvNames(provider.name).join(" or ")})` : "";
    await io.stdout.write(`${provider.name}: ${provider.methods.join(", ")}${suffix}\n`);
  }
  return 0;
}

/**
 * Return the models the openai subscription login in `authFile` offers, in its order.
 *
 * An expiring token is refreshed first. The store's lock wait, each request and the whole listing
 * get `timeoutMs` milliseconds.
 *
 * @throws {AuthError} There is no openai subscription login, or the store, the refresh or the
 *   listing failed.
 * @throws {TimeoutError} `timeoutMs` passed first.
 */
async function subscriptionModels(
  authFile: string,
  timeoutMs: number,
  io: CliIo,
  deps: AuthDeps,
): Promise<readonly string[]> {
  const timedOut = new TimeoutError("The model listing timed out");
  return withDeadline(timeoutMs, timedOut, deps.scheduler ?? realScheduler, async (signal) => {
    try {
      const store = new AuthStore(authFile, { env: io.env, signal, lockTimeoutMs: timeoutMs });
      const selection = store.selection("openai");
      if (selection === null || selection.method !== "subscription") {
        throw new AuthError(`There is no saved openai subscription login in ${authFile}`);
      }
      const service = (deps.openAIAuth ?? newOpenAIAuth)({ signal, requestTimeoutMs: timeoutMs });
      const credentials = await store.credentials("openai", selection.credentialId, {
        refresh: (old) => service.refresh(old),
      });
      // A subscription selection's credential is OAuth, as `credentials` pins its id.
      return await service.subscriptionModels(credentials as OAuthCredentials);
    } catch (error) {
      // Whatever a call cut short by the deadline rejects with, the deadline is the cause.
      if (signal.aborted) throw timedOut;
      throw error;
    }
  });
}

/**
 * List the models an openai subscription login or a provider's key offers.
 *
 * `yian models --auth-file F` lists the models the openai subscription login saved in F offers, in
 * its order, refreshing an expiring token first. `yian models --provider P` lists the model ids
 * P's host gives the key read from stdin (none when stdin is empty or a terminal), sorted;
 * `--base-url` lists another host, and for openai `OPENAI_BASE_URL` does when it is set. azure has
 * no model listing. `--timeout` (default 8) bounds the whole listing, in seconds. Prints a
 * `yian.cli.v1` document `{models}` or `{error: {code, message, status}}` with `--json`, else one
 * id per line, or the failure on stderr.
 *
 * @returns 0, or 1 when the listing fails.
 * @throws {UsageError} Not exactly one of `--auth-file` and `--provider` is given, `--base-url`
 *   comes without `--provider`, the provider isn't supported, `--timeout` isn't a finite number
 *   above 0, or stdin isn't text.
 */
export async function models(command: ParsedArgs, io: CliIo, deps: AuthDeps = {}): Promise<number> {
  const authFile = command.options["auth-file"] as string | undefined;
  let provider = command.options["provider"] as string | undefined;
  const baseUrl = command.options["base-url"] as string | undefined;
  const timeout = (command.options["timeout"] as number | undefined) ?? MODEL_LIST_TIMEOUT_SECONDS;
  if ((authFile === undefined) === (provider === undefined)) {
    throw new UsageError("Invalid value: Give exactly one of --auth-file and --provider");
  }
  if (baseUrl !== undefined && provider === undefined) {
    throw new UsageError("Invalid value: --base-url applies only to --provider");
  }
  if (provider !== undefined) {
    provider = provider.toLowerCase();
    if (!(SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(provider)) {
      throw new UsageError(`Invalid value: Unsupported provider; choose one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`);
    }
  }
  if (!Number.isFinite(timeout) || timeout <= 0) {
    throw new UsageError("Invalid value: --timeout must be a finite number of seconds above 0");
  }
  let error: { code: string; message: string; status: bigint | null } | null = null;
  let listed: readonly string[] | null = null;
  if (provider !== undefined) {
    let key: string | null = null;
    if (io.stdin.isTTY !== true) {
      const text = stdinText(await readStdin(io.stdin));
      if (text === null) throw new UsageError("Invalid value: stdin must be the API key as text");
      key = text.trim() || null;
    }
    try {
      // Runs with an openai key go to OPENAI_BASE_URL when it is set, so its key must not go to OpenAI's host.
      const listedUrl = baseUrl ?? ((provider === "openai" && io.env("OPENAI_BASE_URL")) || null);
      listed = await (deps.listModels ?? listModels)(provider, { apiKey: key, baseUrl: listedUrl, timeout });
    } catch (failure) {
      if (!(failure instanceof ModelListError)) throw failure;
      error = {
        code: failure.timedOut ? "models.timeout" : "models.failed",
        message: `Couldn't list the ${provider} models: ${failure.message}`,
        status: failure.status === null ? null : BigInt(failure.status),
      };
    }
  } else {
    // Milliseconds past the safe integers (above about 1.8e305 s overflow to Infinity) are clamped.
    const timeoutMs = Math.min(timeout * 1000, Number.MAX_SAFE_INTEGER);
    try {
      listed = await subscriptionModels(authFile!, timeoutMs, io, deps);
    } catch (failure) {
      if (failure instanceof TimeoutError) {
        error = {
          code: "models.timeout",
          message: "Couldn't list the openai subscription models: timed out",
          status: null,
        };
      } else if (failure instanceof AuthError) {
        error = {
          code: "models.failed",
          message: `Couldn't list the openai subscription models: ${failure.message}`,
          status: failure.status === null ? null : BigInt(failure.status),
        };
      } else {
        throw failure;
      }
    }
  }
  const json = command.options["json"] === true;
  if (error !== null) {
    if (json) await printJson({ error }, io.stdout);
    else await io.stderr.write(`${error.message}\n`);
    return 1;
  }
  if (json) await printJson({ models: listed === null ? null : [...listed] }, io.stdout);
  else if (listed === null) await io.stderr.write(`${provider} has no model listing\n`);
  else for (const model of listed) await io.stdout.write(`${model}\n`);
  return 0;
}
