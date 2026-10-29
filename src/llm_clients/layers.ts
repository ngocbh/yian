/**
 * Resolve one agent's `llm_config` layers into its complete effective configuration.
 *
 * The compile-time cascade (`compile/llm_cascade.ts`) merges each agent's authored layers over the
 * static call tree and calls `resolveAgentConfig` per agent; `validateLayer` checks every layer
 * before it is merged and `requireConcreteSelection` checks a complete or recorded config before a
 * run uses it. Nothing here reads the environment or saved settings: the host passes its defaults
 * and saved models in. Every function takes yian values (integers as `bigint`); a key whose value
 * is `undefined` is absent, while `null` counts as written.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { writeAsText } from "../typesys/values.ts";
import {
  hostOnlyFieldError,
  LLM_CONFIG_KEYS,
  LLMConfig,
  mergeLlmConfig,
  RETIRED_LLM_CONFIG_KEYS,
  retiredKeyError,
  validateBaseUrl,
} from "./config.ts";
import {
  type AgentConfig,
  HEADER_NAMES_KEY,
  HOST_KEY_MARKER,
  missingModelMessage,
  validateConnection,
  validateHeaderNames,
} from "./connection.ts";
import { validateOrganization } from "./connection_settings.ts";

const REQUIRED_SELECTION_FIELDS: readonly string[] = ["provider", "model"];
// Recorded only on agents marked for a host key: the nonsecret settings sent with it.
const HOST_SETTING_KEYS = ["organization", "project", HEADER_NAMES_KEY];
// Left out of the layer check of a recorded map: only a resolved map carries them.
const RECORDED_ONLY_KEYS: ReadonlySet<string> = new Set([HOST_KEY_MARKER, "base_url", ...HOST_SETTING_KEYS]);

/**
 * Return one agent's complete effective configuration from its merged authored layers.
 *
 * `authored` is the agent's node layer already merged over its flow, enclosing and host layers (or
 * over nothing under `inherit: false`); it is not mutated. `defaults` other than `model` (the
 * lowest-precedence host defaults, possibly empty) fill the gaps under it. An unset model comes
 * from `providerModels` (saved models by provider) for the resolved provider, or from `defaults`
 * when that provider is the default one; a list or record provider has no model. The
 * result is then checked as a concrete selection. `strayModel` says `YIAN_DEFAULT_MODEL` was set
 * without a default provider. Every message names `path`, the agent's static path, unless `label`,
 * the authored site of a synthesized agent's `llm_config`, is given.
 *
 * Returns a new configuration that shares no value with the inputs. Throws `RangeError` if
 * `authored` holds a field internal to recorded runs, the agent has no provider or no model, or the
 * result is not a concrete, valid selection.
 */
export function resolveAgentConfig(
  authored: Readonly<AgentConfig>,
  options: {
    defaults: Readonly<AgentConfig>;
    providerModels: Readonly<Record<string, string>>;
    strayModel: boolean;
    path: string;
    label: string | null;
  },
): AgentConfig {
  const { defaults, providerModels, strayModel, path, label } = options;
  // Only resolution adds the marker and header names; an input layer never has them.
  rejectInternalKeys(authored);
  const written = structuredClone(authored) as AgentConfig;
  const fill = Object.fromEntries(Object.entries(structuredClone(defaults)).filter(([key]) => key !== "model"));
  const config = mergeLlmConfig(written, fill);
  const provider = own(config, "provider");
  if (!truthy(provider)) {
    throw new RangeError(missingModelMessage(path, null, { strayModel, label }));
  }
  if (!truthy(own(written, "model"))) {
    // Keyed by value: a primitive default provider matches its clone, a list or record one never
    // does.
    const models = new Map<unknown, unknown>(Object.entries(providerModels));
    const defaultProvider = own(defaults, "provider");
    const defaultModel = own(defaults, "model");
    if (truthy(defaultProvider) && truthy(defaultModel) && !models.has(defaultProvider)) {
      models.set(defaultProvider, defaultModel);
    }
    const model = models.get(provider);
    if (!truthy(model)) {
      throw new RangeError(missingModelMessage(path, provider, { strayModel, label }));
    }
    config["model"] = model;
  }
  requireConcreteSelection(config, { path, label });
  return config;
}

/**
 * Reject a config that could fall through to ambient provider/model settings.
 *
 * `config` is one agent's effective configuration, of yian values. Every message names `path`, the
 * agent's static path, unless `label`, the authored site of a synthesized agent's `llm_config`, is
 * given.
 *
 * Throws `RangeError` if the config has an invalid marker, base_url or host setting, a bad field,
 * or no concrete provider/model.
 */
export function requireConcreteSelection(
  config: Readonly<AgentConfig>,
  options: { path: string; label?: string | null },
): void {
  const subject = options.label || reprText(options.path);
  const marked = has(config, HOST_KEY_MARKER);
  if (marked && config[HOST_KEY_MARKER] !== true) {
    throw new RangeError(`effective LLM config for ${subject} has an invalid host key marker`);
  }
  const provider = own(config, "provider");
  for (const key of HOST_SETTING_KEYS) {
    if (!has(config, key)) continue;
    // Only the host's settings, sent with its key, are recorded beside the marker.
    if (!marked) throw new RangeError(`effective LLM config for ${subject} has ${key} without a host key`);
    if (key !== HEADER_NAMES_KEY && provider !== "openai") {
      throw new RangeError(`effective LLM config for ${subject}: ${key} applies only to provider openai`);
    }
    prefixed(subject, () => {
      if (key === HEADER_NAMES_KEY) validateHeaderNames(config[key], provider);
      else validateOrganization(config[key], key);
    });
  }
  if (has(config, "base_url")) {
    // An Azure base_url is its resource endpoint; any other serves only a host key.
    if (!marked && provider !== "azure") {
      throw new RangeError(
        `effective LLM config for ${subject} has a base_url without a host key; ` +
          "only provider azure takes a base_url from a flow file",
      );
    }
    // A recorded map is replayed on resume; check its URL before any node runs.
    prefixed(subject, () => validateBaseUrl(config["base_url"]));
  }
  // Only a resolved map carries the marker and the host key's base_url and settings; authored
  // layers may not (except an Azure endpoint, checked above). A retired key can only come from a
  // map recorded before it retired; such a finished run still replays, and resuming an unfinished
  // one refuses it per agent.
  validateLayer(
    Object.fromEntries(
      Object.entries(config).filter(([key]) => !RECORDED_ONLY_KEYS.has(key) && !RETIRED_LLM_CONFIG_KEYS.has(key)),
    ),
  );
  const missing = REQUIRED_SELECTION_FIELDS.filter((field) => {
    const value = own(config, field);
    return typeof value !== "string" || !value;
  });
  if (missing.length > 0) {
    throw new RangeError(`effective LLM config for ${subject} lacks concrete fields: ${writeAsText(missing)}`);
  }
}

/**
 * Reject unknown fields, retired options, and malformed host metadata before composition.
 *
 * `config` is one authored, host, or defaults layer, of yian values. Throws `RangeError` if the
 * layer sets a host-only or internal field, a retired or unknown key, or a malformed connection.
 */
export function validateLayer(config: Readonly<AgentConfig>): void {
  const hostOnly = hostOnlyFieldError(config);
  if (hostOnly !== null) {
    // Every layer here ends up in the recorded effective map, and a lower layer's base_url would
    // redirect the host key.
    throw new RangeError(hostOnly[1]);
  }
  rejectInternalKeys(config);
  const provider = own(config, "provider");
  if (
    (typeof provider === "string" && provider.toLowerCase() === "codex") ||
    has(config, "sandbox") ||
    has(config, "cwd")
  ) {
    new LLMConfig(config);
  }
  const retired = retiredKeyError(config);
  if (retired !== null) throw new RangeError(retired);
  // The loader checks authored values; a host layer is checked here only for unknown keys, which
  // would otherwise be recorded and silently ignored. Its values, including block contents, are
  // checked against the effective config before the run starts (`checkModelConfig`) and when the
  // model client is built.
  const unknown = Object.keys(config)
    .filter((key) => has(config, key) && !LLM_CONFIG_KEYS.has(key) && key !== "_connection")
    .sort(compareCodeUnits);
  for (const key of unknown) {
    const dashed = key.replaceAll("_", "-");
    if (LLM_CONFIG_KEYS.has(dashed)) throw new RangeError(`\`${key}\` is not an llm_config key; write \`${dashed}\``);
  }
  if (unknown.length > 0) {
    throw new RangeError(
      `llm_config has unknown field(s) ${writeAsText(unknown)} ` +
        `(allowed: ${[...LLM_CONFIG_KEYS].sort(compareCodeUnits).join(", ")})`,
    );
  }
  // `null`, not `undefined`, so a layer without a provider isn't checked as an openai one.
  if (has(config, "_connection")) validateConnection(config["_connection"], provider ?? null);
}

/** Reject the fields only a resolved, recorded config may hold. */
function rejectInternalKeys(config: Readonly<AgentConfig>): void {
  for (const internal of [HOST_KEY_MARKER, HEADER_NAMES_KEY]) {
    if (has(config, internal)) throw new RangeError(`llm_config.${internal} is internal to recorded runs`);
  }
}

/** Run `check`, prefixing the message of a `RangeError` it throws with the config's subject. */
function prefixed(subject: string, check: () => unknown): void {
  try {
    check();
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new RangeError(`effective LLM config for ${subject}: ${error.message}`);
  }
}

/** Return `config`'s own value for `key`, so inherited names such as `constructor` are absent. */
function own(config: Readonly<AgentConfig>, key: string): unknown {
  return Object.hasOwn(config, key) ? config[key] : undefined;
}

/** Whether `config` holds `key` as written: an own property that isn't `undefined` (`null` counts). */
function has(config: Readonly<AgentConfig>, key: string): boolean {
  return own(config, key) !== undefined;
}

/** Whether the yian value `value` counts as set: empty strings, lists and records don't. */
function truthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (value !== null && typeof value === "object") return Object.values(value).some((item) => item !== undefined);
  return Boolean(value);
}
