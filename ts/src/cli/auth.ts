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

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { OPTIONAL_API_KEY_PROVIDERS, PROVIDER_API_KEY_ENV } from "../llm_clients/api_key_env.ts";
import { AuthError, AuthStore, type OAuthCredentials, OpenAIAuth } from "../llm_clients/auth/index.ts";
import {
  LLM_CONFIG_KEYS,
  LLMConfig,
  popHostCredentials,
  SUPPORTED_LLM_PROVIDERS,
  validateBaseUrl,
} from "../llm_clients/config.ts";
import {
  validateHeaderName,
  validateHeaderValue,
  validateOrganization,
} from "../llm_clients/connection_settings.ts";
import { listModels, ModelListError } from "../llm_clients/model_list.ts";
import { ValidationError } from "../llm_clients/provider_blocks.ts";
import { realScheduler, type Scheduler } from "../platform.ts";
import { isRecord } from "../typesys/index.ts";
import { parseJson } from "../typesys/json.ts";
import { type CliIo, type ParsedArgs, printJson, UsageError } from "./utils.ts";

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

/** What the handlers reach the outside through; each defaults to the real one. Tests pass fakes. */
export interface AuthDeps {
  /** Build the OpenAI authentication service; defaults to `new OpenAIAuth(options)`. */
  openAIAuth?: (options: {
    signal?: AbortSignal;
    requestTimeoutMs?: number;
  }) => Pick<OpenAIAuth, "refresh" | "subscriptionModels">;
  /** List a provider's models; defaults to `listModels`. */
  listModels?: typeof listModels;
  /** The clock deadlines run on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
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
 * Refuse a prompt when no interactive terminal is available.
 *
 * @throws {AuthError} Standard input is not a terminal; the message names `option`.
 */
function requireTerminal(option: string, io: CliIo): void {
  if (io.stdin.isTTY !== true) {
    throw new AuthError(`Authentication needs ${option} when no interactive terminal is available`);
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
function headerSources(specs: readonly string[], provider: string, io: CliIo): Map<string, string | null> {
  const sources = new Map<string, string | null>();
  const seen = new Set<string>();
  for (const spec of specs) {
    const split = spec.indexOf("=");
    const name = split === -1 ? spec : spec.slice(0, split);
    const variable = split === -1 ? null : spec.slice(split + 1);
    try {
      validateHeaderName(name, provider);
    } catch (error) {
      if (error instanceof RangeError) throw new AuthError(`Invalid --header: ${error.message}`);
      throw error;
    }
    if (seen.has(name.toLowerCase())) throw new AuthError(`--header ${name} is given more than once (names ignore case)`);
    seen.add(name.toLowerCase());
    if (variable === null) {
      if (io.stdin.isTTY !== true) {
        throw new AuthError(
          `--header ${name} reads its value from a hidden prompt, which needs a terminal; ` +
            "use --header NAME=VARIABLE to read it from a variable",
        );
      }
      sources.set(name, null);
      continue;
    }
    if (!variable) throw new AuthError(`--header ${name}=VARIABLE needs an environment variable name`);
    const value = io.env(variable);
    if (!value) throw new AuthError(`The environment variable given for --header ${name} is not set`);
    try {
      sources.set(name, validateHeaderValue(name, value));
    } catch (error) {
      if (error instanceof RangeError) throw new AuthError(`Invalid --header: ${error.message}`);
      throw error;
    }
  }
  return sources;
}

/**
 * Refuse an Azure endpoint that a config's `base_url` could not name.
 *
 * @throws {AuthError} The endpoint is invalid.
 */
function checkEndpoint(endpoint: string): void {
  try {
    validateBaseUrl(endpoint);
  } catch (error) {
    if (error instanceof RangeError) throw new AuthError(`Invalid Azure endpoint: ${error.message}`);
    throw error;
  }
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
function checkOptions(provider: string, options: AuthOptions, io: CliIo): [string, Map<string, string | null>] {
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
      try {
        validateOrganization(value, field);
      } catch (error) {
        if (error instanceof RangeError) throw new AuthError(`Invalid --${field}: ${error.message}`);
        throw error;
      }
    }
  }
  if (header.length > 0) {
    if (name === "ollama") throw new AuthError("ollama takes no API key, so it takes no --header");
    if (method === "subscription") throw new AuthError("--header applies to API-key authentication");
  }
  if (name === "azure" && !options.default) {
    // The Azure client reads the API version only from the environment.
    if (!io.env("OPENAI_API_VERSION")) {
      throw new AuthError("Azure needs OPENAI_API_VERSION set to its API version (runs read it too)");
    }
    if (endpoint !== undefined) checkEndpoint(endpoint);
  }
  return [name, headerSources(header, name, io)];
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
 * @returns 0, or 1 when the command fails or is cancelled (saved logins stay unchanged).
 * @throws {UsageError} `--json` is given to another form.
 */
export async function auth(command: ParsedArgs, io: CliIo, _deps: AuthDeps = {}): Promise<number> {
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
  try {
    if (provider === undefined) {
      const { method, login, model, apiKeyEnv, endpoint, organization, project } = options;
      const setup =
        [method, login, model, apiKeyEnv, endpoint, organization, project].some((value) => value !== undefined) ||
        options.header.length > 0;
      if (options.default || options.remove || setup) {
        throw new AuthError("Name a provider: yian auth <provider> [options]");
      }
      throw new Error("not ported yet");
    }
    checkOptions(provider, options, io);
    throw new Error("not ported yet");
  } catch (error) {
    if (!(error instanceof AuthError)) throw error;
    if (json) {
      const failure = { code: "auth.failed", message: error.message };
      // Device sign-in writes JSON lines, so its error is an event line too.
      await printJson(device ? { event: "error", error: failure } : { error: failure }, io.stdout);
    } else {
      await io.stderr.write(`${error.message}\n`);
    }
    return 1;
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
      layer = null;
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
    const suffix = provider.api_key_env ? ` (${provider.api_key_env})` : "";
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
      const openAIAuth = deps.openAIAuth ?? ((options) => new OpenAIAuth(options));
      const service = openAIAuth({ signal, requestTimeoutMs: timeoutMs });
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
 * `--base-url` lists another host. azure has no model listing. `--timeout` (default 8) bounds the
 * whole listing, in seconds. Prints a `yian.cli.v1` document `{models}` or `{error: {code,
 * message, status}}` with `--json`, else one id per line, or the failure on stderr.
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
  const timeout = (command.options["timeout"] as number | undefined) ?? 8;
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
      listed = await (deps.listModels ?? listModels)(provider, { apiKey: key, baseUrl: baseUrl ?? null, timeout });
    } catch (failure) {
      if (!(failure instanceof ModelListError)) throw failure;
      const reason = failure.message;
      const status = /^HTTP (\d{3})$/.exec(reason);
      error = {
        code: reason === "timed out" ? "models.timeout" : "models.failed",
        message: `Couldn't list the ${provider} models: ${reason}`,
        status: status ? BigInt(status[1]!) : null,
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
        const status = /\(HTTP (\d{3})\)$/.exec(failure.message);
        error = {
          code: "models.failed",
          message: `Couldn't list the openai subscription models: ${failure.message}`,
          status: status ? BigInt(status[1]!) : null,
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
