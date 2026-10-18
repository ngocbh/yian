/**
 * Every predefined default and limit yian uses, each overridable by environment.
 *
 * Each entry has one built-in value and one environment variable, `YIAN_` plus the
 * entry name: `YIAN_MAX_VALUE_NESTING` overrides `MAX_VALUE_NESTING`. Limits and input
 * caps start with `MAX_`; defaults for settings a flow author can set are named
 * `<OWNER>_<SETTING>` after the authored setting (`AGENT_RETRIES`). Units are suffixes
 * (`_BYTES`, `_CHARS`); counts have none. Code reads a value as the camelCase property
 * of `current()`, for example `current().maxValueNesting`.
 *
 * Values are read once per process: the first `current()` call reads and checks every
 * variable, and later changes to the environment have no effect. The environment is
 * injected as an `Env` function (`processEnv` reads `process.env`), and an empty
 * variable counts as unset. A run records the values that differ from the built-in
 * ones, and resuming it with different values is refused. `docs/defaults.md` lists
 * every entry.
 *
 * This module imports only `node:` builtins, so any layer may import it.
 */

import process from "node:process";

/** The kind of an entry: a language safety limit, a setting's default, or an input cap. */
export type Group = "limit" | "setting" | "input cap";

/** One default or limit and the range an environment value must fall in. */
export interface Entry<N extends string = string> {
  /** Upper snake case name, for example `MAX_VALUE_NESTING`. */
  readonly name: N;
  /** The environment variable that overrides this entry: `YIAN_` plus the name. */
  readonly variable: string;
  /** Built-in value. */
  readonly default: number;
  /** One line saying what the value controls. */
  readonly description: string;
  /** Whether this is a language safety limit, a setting's default, or an input cap. */
  readonly group: Group;
  /** Smallest value accepted from the environment. */
  readonly minimum: number;
  /** Largest value accepted, or `null` for any safe integer (up to `Number.MAX_SAFE_INTEGER`). */
  readonly maximum: number | null;
  /** A value accepted outside the range that means no cap. */
  readonly noLimit: number | null;
}

/** Return the frozen entry `name` with the given fields. */
function entry<N extends string>(
  name: N,
  fields: {
    default: number;
    description: string;
    group: Group;
    minimum: number;
    maximum?: number | null;
    noLimit?: number | null;
  },
): Entry<N> {
  const { default: value, description, group, minimum, maximum = null, noLimit = null } = fields;
  const variable = "YIAN_" + name;
  return Object.freeze({ name, variable, default: value, description, group, minimum, maximum, noLimit });
}

// The nesting limits keep the deepest accepted input the same on Node and Bun, whose
// stacks differ: the `yaml` package handles about 500 levels on Node but 5000 on Bun's
// main thread. Each maximum is re-measured once every recursive walker is ported.
export const ENTRIES = Object.freeze([
  // Language safety limits.
  // yian's JSON writer and reader are iterative, but walkers still to be ported recurse
  // per value level, so the limit can only be lowered.
  entry("MAX_VALUE_NESTING", {
    default: 100,
    description: "Deepest nesting a value may have; also bounds expression nesting",
    group: "limit",
    minimum: 1,
    maximum: 100,
  }),
  // The type walkers handle at least 500 levels on both runtimes. The limit also bounds
  // each typedef's levels, a chain of typedefs naming each other, and a type with its
  // typedef names expanded.
  entry("MAX_TYPE_NESTING", {
    default: 100,
    description: "Deepest nesting a type may have, counting record levels",
    group: "limit",
    minimum: 1,
    maximum: 125,
  }),
  // Providers limit the nesting of a structured-output schema, so a deeper `output:` type
  // is of no use; the variable can only lower it.
  entry("MAX_AGENT_OUTPUT_NESTING", {
    default: 16,
    description: "Deepest nesting an AGENT `output:` type may have (each record and list is a level)",
    group: "limit",
    minimum: 1,
    maximum: 16,
  }),
  // Twice a value's 100 levels, so a value at that limit fits inside a flow's own
  // structure; well under the `yaml` package's overflow depth on Node.
  entry("MAX_YAML_NESTING", {
    default: 200,
    description: "Deepest nesting of a YAML document yian reads",
    group: "limit",
    minimum: 1,
    maximum: 200,
  }),
  // Aliases make a document's expanded size exponential in its text, and every later
  // step, including the run log (JSON has no aliases), costs per expanded node.
  entry("MAX_YAML_NODES", {
    default: 100_000,
    description: "Most nodes a YAML document yian reads may expand to, counting each alias as a copy",
    group: "limit",
    minimum: 1,
  }),
  // Loading a body or a `uses:` file recurses through the loader, so the limit holds
  // both chains; it can only be lowered until the loader is ported and measured.
  entry("MAX_CALL_NESTING", {
    default: 50,
    description: "Deepest chain of call, map, and loop bodies a flow may nest, and of uses: files it may import",
    group: "limit",
    minimum: 1,
    maximum: 50,
  }),
  entry("MAX_TOTAL_NODES", {
    default: 10_000,
    description: "Most live nodes a run's graph may grow to",
    group: "limit",
    minimum: 1,
  }),
  entry("MAX_SKIP_WORLDS", {
    default: 10_000,
    description: "Most branch combinations the skip check explores before giving up",
    group: "limit",
    minimum: 1,
  }),
  // A cut keeps one character less than the cap and adds `…`, so the minimum is 2.
  entry("MAX_SHOWN_VALUE_CHARS", {
    default: 200,
    description: "Longest rendering of a value or key an error message shows",
    group: "limit",
    minimum: 2,
  }),
  // Defaults for node and memory behavior; a flow can override some of them.
  entry("AGENT_RETRIES", {
    default: 2,
    description: "Extra attempts an AGENT gets to correct an invalid structured reply when the node sets none",
    group: "setting",
    minimum: 0,
  }),
  entry("AGENT_MAX_TOOL_ITERATIONS", {
    default: -1,
    description: "AGENT tool-call turns when the node sets none; -1 is no cap",
    group: "setting",
    minimum: 1,
    noLimit: -1,
  }),
  // 0 keeps a TOOL node calling its tool once, since a retried call repeats any side
  // effect the failed attempt already had.
  entry("TOOL_RETRIES", {
    default: 0,
    description: "Extra attempts a TOOL node gets after a failed tool call when the node sets none",
    group: "setting",
    minimum: 0,
  }),
  // Backoff doubles from this base (1 s, 2 s, 4 s, ...) up to the cap below, and
  // jitter picks each wait between half and all of its step. No node sets the delays;
  // a run reads them when it waits, so they don't change the loaded flow.
  entry("TOOL_RETRY_DELAY_SECONDS", {
    default: 1,
    description: "Base wait in seconds before a TOOL node's first retry; each later wait doubles, up to the cap",
    group: "setting",
    minimum: 0,
  }),
  // The cap also bounds a tool's own `retry_after`, so a tool can't stall a run longer.
  entry("TOOL_RETRY_MAX_DELAY_SECONDS", {
    default: 30,
    description: "Longest wait in seconds before a TOOL node retries a failed tool call",
    group: "setting",
    minimum: 0,
  }),
  // A hung MCP server would otherwise stall the run forever; a run reads these when it
  // talks to a server, so they don't change the loaded flow.
  entry("MCP_CONNECT_TIMEOUT_SECONDS", {
    default: 30,
    description: "Longest wait in seconds for an MCP server to start and list its tools",
    group: "setting",
    minimum: 1,
  }),
  entry("MCP_CALL_TIMEOUT_SECONDS", {
    default: 300,
    description: "Longest wait in seconds for one MCP tool call",
    group: "setting",
    minimum: 1,
  }),
  entry("MODEL_RETRIES", {
    default: 6,
    description: "Extra attempts a model call gets after a transport failure (a network error, 5xx, 429 or timeout)",
    group: "setting",
    minimum: 0,
  }),
  entry("MODEL_FIRST_RESPONSE_TIMEOUT_SECONDS", {
    default: 240,
    description: "Longest wait in seconds from sending a model request to the first bytes of its response",
    group: "setting",
    minimum: 1,
  }),
  entry("MODEL_IDLE_TIMEOUT_SECONDS", {
    default: 240,
    description: "Longest wait in seconds between bytes of a streamed model response",
    group: "setting",
    minimum: 1,
  }),
  entry("REQUEST_TIMEOUT_SECONDS", {
    default: 30,
    description: "Longest wait in seconds for a short request (model listing, sign-in calls)",
    group: "setting",
    minimum: 1,
  }),
  entry("EPISODIC_MAX_EPISODES", {
    default: 100,
    description: "Episodes an episodic memory keeps when it sets no limit",
    group: "setting",
    minimum: 1,
  }),
  entry("EPISODIC_RECALL_LIMIT", {
    default: 10,
    description: "Episodes an episodic memory recalls when it sets no limit",
    group: "setting",
    minimum: 1,
  }),
  entry("REFLECTIVE_MAX_ITEMS", {
    default: 100,
    description: "Lessons a reflective memory keeps when it sets no limit",
    group: "setting",
    minimum: 1,
  }),
  entry("REFLECTIVE_RECALL_LIMIT", {
    default: 20,
    description: "Lessons a reflective memory recalls when it sets no limit",
    group: "setting",
    minimum: 1,
  }),
  // The three reflective budgets floor at 256: material over budget is cut to a
  // summary with a 64-char digest that is about 142 chars whatever the budget.
  entry("REFLECTIVE_MAX_LESSON_CHARS", {
    default: 8_000,
    description: "Characters of current lessons shown to the reflection model",
    group: "setting",
    minimum: 256,
  }),
  entry("REFLECTIVE_MAX_EPISODE_CHARS", {
    default: 20_000,
    description: "Characters of the episode shown to the reflection model",
    group: "setting",
    minimum: 256,
  }),
  entry("REFLECTIVE_MAX_LIMIT_CHARS", {
    default: 256,
    description: "Characters of the memory limits shown to the reflection model",
    group: "setting",
    minimum: 256,
  }),
  entry("SESSION_HISTORY_BUDGET_CHARS", {
    default: 32_000,
    description: "Characters of history a session memory sends to the model",
    group: "setting",
    minimum: 1,
  }),
  entry("SESSION_RECENT_TURNS", {
    default: 4,
    description: "Recent turns a session memory always keeps whole",
    group: "setting",
    minimum: 0,
  }),
  entry("PROFILE_MAX_CHARS", {
    default: 32_000,
    description: "Characters a profile memory may hold",
    group: "setting",
    minimum: 1,
  }),
  entry("PROFILE_INPUT_BUDGET_CHARS", {
    default: 24_000,
    description: "Characters of run material sent to a profile memory's learner",
    group: "setting",
    minimum: 1,
  }),
  // Input-size caps.
  entry("MAX_AUTH_RESPONSE_BYTES", {
    default: 1024 * 1024,
    description: "Largest response body read from a sign-in server",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_AUTH_STORE_BYTES", {
    default: 1024 * 1024,
    description: "Largest auth store file read or written",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_OAUTH_CALLBACK_BYTES", {
    default: 16 * 1024,
    description: "Largest request header block read from the sign-in callback",
    group: "input cap",
    minimum: 1,
  }),
  // OpenRouter's listing carries a description and pricing for every model.
  entry("MAX_MODEL_LIST_BYTES", {
    default: 16 * 1024 * 1024,
    description: "Largest model listing read from a provider",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_MCP_MESSAGE_BYTES", {
    default: 16 * 1024 * 1024,
    description: "Largest message read from an MCP server",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_BASE_URL_CHARS", {
    default: 2048,
    description: "Longest provider base URL accepted",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_CONNECTION_HEADERS", {
    default: 32,
    description: "Most extra request headers a provider connection may send",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_HEADER_VALUE_CHARS", {
    default: 8192,
    description: "Longest connection header name or value accepted",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_ORGANIZATION_CHARS", {
    default: 256,
    description: "Longest OpenAI organization or project id accepted",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_SKILL_FILE_BYTES", {
    default: 256 * 1024,
    description: "Largest SKILL.md or bundled skill file returned to a model",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_SKILL_NAME_CHARS", {
    default: 64,
    description: "Longest skill name accepted",
    group: "input cap",
    minimum: 1,
  }),
  // A cut keeps one character less than the cap and adds an ellipsis, so the
  // minimum is 2.
  entry("MAX_SKILL_DESCRIPTION_CHARS", {
    default: 1024,
    description: "Longest skill description shown in the skill listing",
    group: "input cap",
    minimum: 2,
  }),
  entry("MAX_QUESTIONS", {
    default: 4,
    description: "Most questions a HUMAN_INPUT node may ask at once",
    group: "input cap",
    minimum: 1,
  }),
  entry("MAX_QUESTION_OPTIONS", {
    default: 6,
    description: "Most options a question to the user may offer",
    group: "input cap",
    minimum: 1,
  }),
  // A cut keeps one character less than the cap and adds an ellipsis, so the
  // minimum is 2.
  entry("MAX_QUESTION_OPTION_CHARS", {
    default: 80,
    description: "Longest question option shown to the user",
    group: "input cap",
    minimum: 2,
  }),
  // A non-interactive caller feeding empty input forever can't hang; 3 mirrors
  // sudo's `passwd_tries` default.
  entry("MAX_KEY_PROMPTS", {
    default: 3,
    description: "Times the CLI asks for an API key before giving up",
    group: "input cap",
    minimum: 1,
  }),
] as const);

type CamelCase<S extends string> = S extends `${infer H}_${infer T}`
  ? `${Lowercase<H>}${Capitalize<CamelCase<T>>}`
  : Lowercase<S>;

/** The name of an entry, such as `"MAX_VALUE_NESTING"`. */
export type EntryName = (typeof ENTRIES)[number]["name"];

/** The value of every entry, one property per entry named in camelCase. */
export type Defaults = { readonly [N in EntryName as CamelCase<N>]: number };

/** Return the `Defaults` key of the entry `name`: `MAX_VALUE_NESTING` gives `maxValueNesting`. */
function keyOf(name: string): string {
  return name.toLowerCase().replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

/** Every entry's built-in value. */
export const BUILTIN: Defaults = Object.freeze(
  Object.fromEntries(ENTRIES.map((each) => [keyOf(each.name), each.default])),
) as Defaults;

/** An environment variable holds a value its entry does not accept. */
export class DefaultsError extends RangeError {
  override name = "DefaultsError";
}

/** A run started with values that differ from the ones in effect now. */
export class DefaultsMismatchError extends RangeError {
  override name = "DefaultsMismatchError";
}

const CANONICAL_INT = /^(?:0|-?[1-9][0-9]*)$/;
// Counted in code points, so a cut never splits a surrogate pair.
const SHOWN_CHARS = 40;

/** Describe the values `each` accepts. */
function rangeText(each: Entry): string {
  let text = `an integer from ${each.minimum} to ${each.maximum ?? Number.MAX_SAFE_INTEGER}`;
  if (each.noLimit !== null) text += `, or ${each.noLimit} for no cap`;
  return text;
}

/**
 * Return the value `raw` sets for `each`.
 *
 * @param each The entry the variable overrides.
 * @param raw The variable's non-empty value.
 * @returns The checked value, an exact safe integer.
 * @throws {DefaultsError} If `raw` is not a canonical decimal integer in range.
 */
function parse(each: Entry, raw: string): number {
  const fail = (problem: string): DefaultsError => {
    const points = Array.from(raw);
    const cut = points.length <= SHOWN_CHARS ? raw : points.slice(0, SHOWN_CHARS - 3).join("") + "...";
    const advice = `use ${rangeText(each)}, or unset it`;
    return new DefaultsError(`${each.variable}=${JSON.stringify(cut)} is ${problem}; ${advice}`);
  };
  if (!CANONICAL_INT.test(raw)) throw fail("not an integer");
  // Rounding is monotonic and every bound is an exact double, so a text above
  // `Number.MAX_SAFE_INTEGER` rounds to at least 2^53 (or `Infinity`) and is refused.
  const value = Number(raw);
  const inRange =
    value === each.noLimit || (each.minimum <= value && value <= (each.maximum ?? Number.MAX_SAFE_INTEGER));
  if (!inRange) throw fail("out of range");
  return value;
}

/** Read one environment variable: its text, or `undefined` when it is unset. */
export type Env = (name: string) => string | undefined;

/** Read `process.env`; a name that is not an own property, such as `"__proto__"`, is unset. */
export const processEnv: Env = (name) => (Object.hasOwn(process.env, name) ? process.env[name] : undefined);

/**
 * Return the values `env` sets, with built-in values for the rest.
 *
 * @param env The environment to read, such as `processEnv`; an empty value counts as unset.
 * @returns The checked values, frozen.
 * @throws {DefaultsError} If a variable holds a value its entry does not accept.
 */
export function read(env: Env): Defaults {
  const values: Record<string, number> = {};
  for (const each of ENTRIES) {
    const raw = env(each.variable);
    values[keyOf(each.name)] = raw ? parse(each, raw) : each.default;
  }
  return Object.freeze(values) as Defaults;
}
