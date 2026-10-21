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
 * Values are read once per process, unless a call's `env` scopes them: the first
 * `current()` call reads and checks every variable, and later changes to the environment
 * have no effect. Inside `withEnv(env, fn)`, `current()` returns the values `env` sets
 * instead, for `fn` and the async work it starts. The environment is injected as an `Env`
 * function (`processEnv` reads `process.env`), and an empty variable counts as unset. A run
 * records the values that differ from the built-in ones, and resuming it with different
 * values is refused. `docs/defaults.md` lists every entry.
 *
 * This module imports only `node:` builtins, so any layer may import it.
 */

import { AsyncLocalStorage } from "node:async_hooks";
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
// stacks differ: under a 1 MiB stack the `yaml` package composes 793 levels on Node and
// 818 on Bun, 3975 in a Bun Worker. Each limit's measured maximum is the deepest input that
// loads, runs, replays and resumes with every limit lifted; the AGENT output limit's builds
// the reply schema and reads a reply, with no run (docs/defaults.md has the method).
export const ENTRIES = Object.freeze([
  // Language safety limits.
  // yian's JSON writer and reader are iterative; the other value walkers recurse per
  // value level, and the limit can only be lowered.
  // Measured maximum under a 1 MiB stack (Node) / a Bun Worker: 1587 / 10500 (expressions; a plain value goes deeper).
  entry("MAX_VALUE_NESTING", {
    default: 100,
    description: "Deepest nesting a value may have; also bounds expression nesting",
    group: "limit",
    minimum: 1,
    maximum: 100,
  }),
  // The type walkers handle at least 893 levels on both runtimes. The limit also bounds
  // each typedef's levels, a chain of typedefs naming each other, and a type with its
  // typedef names expanded.
  // Measured maximum under a 1 MiB stack (Node) / a Bun Worker: 1387 / at least 3200; 893 under a 1 MiB Bun stack.
  entry("MAX_TYPE_NESTING", {
    default: 100,
    description: "Deepest nesting a type may have, counting record levels",
    group: "limit",
    minimum: 1,
    maximum: 125,
  }),
  // Providers limit the nesting of a structured-output schema, so a deeper `output:` type
  // is of no use; the variable can only lower it.
  // Measured maximum under a 1 MiB stack (Node) / a Bun Worker: at least 400 / at least 800 (building the schema
  // and reading a reply, not a run).
  entry("MAX_AGENT_OUTPUT_NESTING", {
    default: 16,
    description: "Deepest nesting an AGENT `output:` type may have (each record and list is a level)",
    group: "limit",
    minimum: 1,
    maximum: 16,
  }),
  // Twice a value's 100 levels, so a value at that limit fits inside a flow's own
  // structure; well under the `yaml` package's overflow depth on Node.
  // Measured maximum under a 1 MiB stack (Node) / a Bun Worker: 793 / 3975.
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
  // both chains; it can only be lowered.
  // Measured maximum under a 1 MiB stack (Node) / a Bun Worker: 268 / at least 400.
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
    description: "Largest message read from a stdio MCP server",
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

const NAMES: ReadonlySet<string> = new Set(ENTRIES.map((each) => each.name));

/** Compare two texts by UTF-16 code units, for sorting. */
function compareText(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Return the values in `values` that differ from the built-in ones.
 *
 * @param values The values to compare with `BUILTIN`.
 * @returns Entry name to value, sorted by name; values are `bigint`s so a run log records integers.
 */
export function nonDefaults(values: Defaults): Record<string, bigint> {
  const all: Readonly<Record<string, number>> = values;
  const changed = ENTRIES.flatMap((each) => {
    const value = all[keyOf(each.name)];
    return value === each.default ? [] : [[each.name, BigInt(value!)] as const];
  }).sort(([a], [b]) => compareText(a, b));
  return Object.fromEntries(changed);
}

/**
 * Return the built-in values overlaid with a run's recorded values.
 *
 * @param recorded Entry name to value, as a run recorded it; only own keys count.
 * @returns The values the run started with, frozen.
 * @throws {DefaultsMismatchError} If a recorded name is not an entry.
 */
export function fromRecorded(recorded: Readonly<Record<string, bigint | number>>): Defaults {
  const unknown = unknownNames(recorded);
  if (unknown.length > 0) throw new DefaultsMismatchError(unknown.join("\n"));
  const values: Record<string, number> = { ...BUILTIN };
  for (const [name, value] of Object.entries(recorded)) values[keyOf(name)] = Number(value);
  return Object.freeze(values) as Defaults;
}

/** Describe each recorded name that is not an entry, sorted by name. */
function unknownNames(recorded: Readonly<Record<string, bigint | number>>): string[] {
  return Object.keys(recorded)
    .filter((name) => !NAMES.has(name))
    .sort(compareText)
    .map((name) => `this run started with YIAN_${name}, which this version of yian does not know`);
}

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

/** Return whether `value` is an integer `each` accepts from the environment. */
function inRange(each: Entry, value: number): boolean {
  return (
    Number.isInteger(value) &&
    (value === each.noLimit || (each.minimum <= value && value <= (each.maximum ?? Number.MAX_SAFE_INTEGER)))
  );
}

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
  if (!inRange(each, value)) throw fail("out of range");
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

let cached: Defaults | undefined;

/** The values of the innermost enclosing `withEnv` call with a custom `env`, if any. */
const SCOPED = new AsyncLocalStorage<Defaults>();

/**
 * Return the values in effect: the enclosing `withEnv` call's, else this process's, reading
 * `processEnv` on the first call.
 *
 * A failed read is not kept, so every call throws until the environment is fixed.
 *
 * @returns The values in effect for this call.
 * @throws {DefaultsError} If a variable holds a value its entry does not accept.
 */
export function current(): Defaults {
  return SCOPED.getStore() ?? (cached ??= read(processEnv));
}

/**
 * Run `fn` with `current()` returning the values `env` sets, for `fn` and the async work it starts.
 *
 * `processEnv` keeps the values already in effect (the enclosing call's, else this process's).
 * A custom `env` is read once, before `fn` runs. The scope covers only `fn` and the work it
 * starts: an object returned from `fn` and used later (a run result's replay, a pooled
 * connection's callbacks) is outside it, so wrap that use in its own `withEnv` or capture the
 * values. Concurrent calls don't see each other's values.
 *
 * @param env The environment to read, such as `processEnv`; an empty value counts as unset.
 * @param fn The work to run, sync or async.
 * @returns What `fn` returns.
 * @throws {DefaultsError} If a variable holds a value its entry does not accept; then `fn` is not called.
 */
export function withEnv<T>(env: Env, fn: () => T): T {
  if (env === processEnv) {
    // Read now, so a bad process value throws before `fn` starts work, as a custom one does.
    current();
    return fn();
  }
  return SCOPED.run(read(env), fn);
}

/**
 * Check that a run's recorded values equal the ones in effect now.
 *
 * @param recorded Entry name to value, as the run recorded it; only own keys count.
 * @throws {DefaultsMismatchError} Naming every recorded name that is not an entry and every
 *   variable whose value differs, in one message.
 */
export function checkRecorded(recorded: Readonly<Record<string, bigint | number>>): void {
  const here: Readonly<Record<string, number>> = current();
  const problems = unknownNames(recorded);
  for (const each of ENTRIES) {
    const now = here[keyOf(each.name)];
    const then = Object.hasOwn(recorded, each.name) ? recorded[each.name]! : each.default;
    // Exact for any safe integer, which covers every value `read` accepts; a larger recorded
    // value rounds to at least 2^53 in magnitude, outside every range, so it is still invalid.
    const value = Number(then);
    if (now === value) continue;
    if (value !== each.default && !inRange(each, value)) {
      // Advising `set YIAN_X=<then>` would name a value the variable refuses.
      problems.push(
        `this run recorded an invalid value ${each.variable}=${String(then)}; ` +
          `${each.variable} accepts ${rangeText(each)}`,
      );
    } else if (value === each.default) {
      problems.push(
        `${each.variable} is ${now} here, but this run started with the default ${String(then)}; ` +
          `unset ${each.variable} to resume it`,
      );
    } else {
      problems.push(
        `${each.variable} is ${now} here, but this run started with ${String(then)}; ` +
          `set ${each.variable}=${String(then)} to resume it`,
      );
    }
  }
  if (problems.length > 0) throw new DefaultsMismatchError(problems.join("\n"));
}

/**
 * Drop this process's values, read `env` and keep the result; for tests and conformance.
 *
 * Inside a `withEnv` call with a custom `env`, `current()` still returns that call's values.
 *
 * @param env The environment to read.
 * @returns The new values.
 * @throws {DefaultsError} If a variable holds a value its entry does not accept; then no values are kept.
 */
export function reload(env: Env = processEnv): Defaults {
  cached = undefined;
  cached = read(env);
  return cached;
}

/**
 * Keep a frozen copy of `values` as this process's values, with no range check; for tests and conformance.
 *
 * Inside a `withEnv` call with a custom `env`, `current()` still returns that call's values.
 */
export function install(values: Defaults): void {
  cached = Object.freeze({ ...values });
}

/**
 * Drop this process's values so the next `current()` outside a custom `withEnv` scope reads
 * again; for tests and conformance.
 */
export function forget(): void {
  cached = undefined;
}
