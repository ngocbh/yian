/**
 * The start-up steps `yian run` and `yian chat` share before a run begins: reading and loading
 * the flow file (`readFlowText`, also for `--inputs`), the `--provider`/`--model` layer, the key
 * preflight (`ensureProviderKeys` over the connections `agentConnections` finds) and a fresh run's
 * log path under `defaultRunsDir`.
 *
 * Knows about: compose, the run boundary's input and LLM-cascade helpers, connections and saved
 * logins, the platform's home and folder helpers, the CLI's rendering and key prompt.
 */

import { chmodSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { isDeepStrictEqual } from "node:util";

import { knownCallTimeBaseUrls, knownCallTimeProviders } from "../compile/llm_cascade.ts";
import { copyGraph } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { loadFlow } from "../compose/loader.ts";
import type { Env } from "../defaults.ts";
import { Diagnostic } from "../diagnostics.ts";
import { AuthError } from "../llm_clients/auth/index.ts";
import { type Connection, connectionCredentials, connectionSnapshot } from "../llm_clients/connection.ts";
import { coerceInputs } from "../nodes/binding.ts";
import { homeDir, syncFolder } from "../platform.ts";
import { hostInputs, runLogPath, runStartLlm } from "../run.ts";
import { printStyled, renderDiagnostic, writeEngineTrace } from "./render.ts";
import { type CliIo, ensureEnvKey, MissingKeyError, type PromptContext } from "./utils.ts";

/**
 * @internal Read and load the flow file at `path` for a fresh run. A file that isn't UTF-8 text or
 * a flow that doesn't load is reported on stderr (a load error located, its stack below under
 * `engineTrace`), and its diagnostic is returned instead of the flow.
 *
 * @throws {Error} A system error when the file can't be read.
 */
export async function loadFlowFile(path: string, engineTrace: boolean, io: CliIo): Promise<ComposedFlow | Diagnostic> {
  let text: string;
  try {
    text = readFlowText(path);
  } catch (error) {
    if (!(error instanceof NotUtf8Error)) throw error;
    const message = `can't read ${path}: not UTF-8 text`;
    await printStyled(io, message, "red");
    return new Diagnostic({ code: "compose.load", message });
  }
  try {
    return await loadFlow(text, { searchPaths: [dirname(path)], sourceLabel: basename(path), env: io.env });
  } catch (error) {
    if (!(error instanceof LoadError)) throw error;
    // Where in the `.yaml` the flow broke, not the engine's internals, unless asked for.
    await renderDiagnostic(error.diagnostic, io.stderr, io.env);
    await writeEngineTrace(error, engineTrace, io);
    return error.diagnostic;
  }
}

/**
 * @internal Return the outermost `llmConfig` layer of a fresh run: `provider` and `model`, those
 * given. It fills the gaps rather than overriding: an agent's own llm_config and the flow's still
 * win.
 */
export function cliLayer(options: { provider?: string | undefined; model?: string | undefined }): Record<string, string> {
  const layer: Record<string, string> = {};
  if (options.provider) layer["provider"] = options.provider;
  if (options.model) layer["model"] = options.model;
  return layer;
}

/**
 * @internal Check every key `flow`'s agents need before a fresh run creates its log
 * (`ensureProviderKeys`), and return the message of a missing key or of a login that doesn't
 * resolve, else null.
 *
 * @throws `signal`'s reason once it has aborted: a key prompt the stop closed is the stop, not a
 *   missing key. Any other failure, as `ensureProviderKeys` throws it.
 */
export async function preflightKeys(
  flow: ComposedFlow,
  cliCfg: Readonly<Record<string, string>>,
  supplied: Readonly<Record<string, unknown>>,
  context: PromptContext,
  signal: AbortSignal,
): Promise<string | null> {
  try {
    await ensureProviderKeys(flow, cliCfg, supplied, context);
  } catch (error) {
    signal.throwIfAborted();
    if (!(error instanceof MissingKeyError || error instanceof AuthError)) throw error;
    return error.message;
  }
  return null;
}

/**
 * @internal Return the log a fresh run writes, `log`, else `defaultRunLogPath`'s in a fresh run
 * folder, and that folder (`runDir`; null for a given `log`), which the caller removes
 * (`removeEmptyDir`) once the command ends, so a run that never created its log leaves none.
 */
export function freshLog(log: string | undefined, runId: string, env: Env): { log: string; runDir: string | null } {
  if (log !== undefined) return { log, runDir: null };
  const path = defaultRunLogPath(runId, env);
  return { log: path, runDir: dirname(path) };
}

/** @internal A flow or inputs file holding bytes that aren't UTF-8. */
export class NotUtf8Error extends Error {
  override name = "NotUtf8Error";
}

/**
 * @internal Read the flow or inputs file at `path` as strict UTF-8, a BOM kept, `\r\n` and `\r` read as `\n`.
 *
 * @throws {NotUtf8Error} When the file isn't UTF-8 text.
 * @throws {Error} A system error when the file can't be read.
 */
export function readFlowText(path: string): string {
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
 * Create the private CLI directories and the run's folder, and return a fresh run's default log
 * path, `~/.yian/runs/<runId>/log.jsonl` (`runLogPath` under `~/.yian/runs`), with `HOME` from
 * `env` (the account's home directory when it is unset or empty).
 *
 * `~/.yian` and `~/.yian/runs` are created with mode 0700, or set to it if they exist; the run's
 * folder must be new and is created with mode 0700. The folder above each one created is synced.
 * The file itself is not created: the run log creates it exclusively, owner-only.
 *
 * @throws {Error} When a directory can't be created, changed or synced, a non-directory holds the
 *   path of `~/.yian` or `~/.yian/runs`, or anything holds the run folder's path.
 */
export function defaultRunLogPath(runId: string, env: Env): string {
  const runsDir = defaultRunsDir(env);
  for (const dir of [dirname(runsDir), runsDir]) {
    let made = true;
    try {
      mkdirSync(dir, { mode: 0o700 });
    } catch (error) {
      const taken = (error as NodeJS.ErrnoException).code === "EEXIST";
      if (!taken || !statSync(dir, { throwIfNoEntry: false })?.isDirectory()) throw error;
      made = false;
    }
    chmodSync(dir, 0o700);
    if (made) syncFolder(dirname(dir));
  }
  return runLogPath(runsDir, runId);
}

/** Return the CLI's runs folder, `~/.yian/runs`, with `HOME` from `env` as `defaultRunLogPath` reads it. */
export function defaultRunsDir(env: Env): string {
  return join(homeDir(env), ".yian", "runs");
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
