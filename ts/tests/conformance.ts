/**
 * Run the language-neutral conformance cases in `conformance/cases/` against TypeScript yian.
 *
 * Each case file is one test, named by its id `<group>/<name>`. `conformance/README.md` defines
 * the case format and what each kind checks.
 *
 * Runs under Bun from source and under Node from `dist/`; only `node:` APIs and `yaml`.
 */

import { isMap, isNode, isSeq, parseDocument, visit } from "yaml";

/** A case value: an int is a `bigint`, a float a `number`. */
export type Value = null | boolean | bigint | number | string | Value[] | { [key: string]: Value };

/** A case file, as read. */
export type Case = { [key: string]: Value };

/** The kinds of case. */
export type Kind =
  | "load"
  | "run"
  | "resume"
  | "backward"
  | "expr"
  | "value"
  | "defaults"
  | "fingerprint"
  | "cli";

/** A mapping in a case. */
type Mapping = { [key: string]: Value };

// The closed format, as conformance/README.md defines it. Sets and maps, never object lookups,
// so a case value such as `kind: toString` never finds an `Object.prototype` member.
const COMMON: ReadonlySet<string> = new Set([
  "kind", "rules", "pending", "python_skip", "expect", "python_expect",
]);
const FLOW: ReadonlySet<string> = new Set(["flow", "python_flow", "files", "env"]);
const RUN: ReadonlySet<string> = new Set([...FLOW, "inputs", "run", "host"]);
const KIND_KEYS: ReadonlyMap<Kind, ReadonlySet<string>> = new Map<Kind, ReadonlySet<string>>([
  ["load", FLOW],
  ["run", RUN],
  ["resume", new Set([...RUN, "resume"])],
  ["backward", new Set([...RUN, "backward"])],
  ["expr", new Set(["expr", "template", "values", "item", "mode", "typedefs", "env"])],
  ["value", new Set(["type", "typedefs", "value", "env"])],
  ["defaults", new Set(["env"])],
  // Both runners load `flow` and `other` as written, so there is no `python_flow`.
  ["fingerprint", new Set([...[...FLOW].filter((key) => key !== "python_flow"), "other"])],
  ["cli", new Set([...FLOW, "args", "stdin"])],
]);
const REQUIRED: ReadonlyMap<Kind, readonly string[]> = new Map<Kind, readonly string[]>([
  ["load", ["flow"]], ["run", ["flow"]], ["resume", ["flow"]], ["backward", ["flow", "backward"]],
  ["value", ["type", "value"]], ["fingerprint", ["flow", "other"]],
]);
// Each kind's `expect` holds exactly one of its result groups, and any of its optional keys.
const RESULTS: ReadonlyMap<Kind, readonly (readonly string[])[]> = new Map<
  Kind,
  readonly (readonly string[])[]
>([
  ["load", [["load"]]],
  ["run", [["raises"], ["output"]]],
  ["resume", [["raises"], ["output"]]],
  ["backward", [["raises"], ["output"]]],
  ["expr", [["value"], ["raises"]]],
  ["value", [["value", "kind"], ["raises"]]],
  ["defaults", [["defaults"], ["raises"]]],
  ["fingerprint", [["fingerprint"]]],
  ["cli", [["exit"]]],
]);
const RUN_OPTIONAL: ReadonlySet<string> = new Set([
  "trace", "events", "stream", "tool_calls", "memory",
]);
const OPTIONAL: ReadonlyMap<Kind, ReadonlySet<string>> = new Map<Kind, ReadonlySet<string>>([
  ["run", RUN_OPTIONAL],
  ["resume", RUN_OPTIONAL],
  ["backward", new Set([...RUN_OPTIONAL, "backward"])],
  ["cli", new Set(["stdout", "stderr", "json", "json_lines"])],
]);
const EXPECT_KEYS: ReadonlyMap<Kind, ReadonlySet<string>> = new Map(
  [...RESULTS].map(([kind, groups]) => [
    kind,
    new Set([...(OPTIONAL.get(kind) ?? []), ...groups.flat()]),
  ]),
);
const ALL_KEYS: ReadonlySet<string> = new Set([
  ...COMMON, ...[...KIND_KEYS.values()].flatMap((keys) => [...keys]),
]);
const ALL_EXPECT_KEYS: ReadonlySet<string> = new Set(
  [...EXPECT_KEYS.values()].flatMap((keys) => [...keys]),
);
const RUN_KEYS: ReadonlySet<string> = new Set([
  "run_id", "num_workers", "learnable", "llm_config", "context",
]);
const RESUME_KEYS: ReadonlySet<string> = new Set(["num_workers", "api_key", "headers", "context"]);
// Host keys the format reserves and this runner doesn't support yet.
const RESERVED_HOST_KEYS: readonly string[] = ["models", "steering", "stop", "memory", "skills"];
const HOST_KEYS: ReadonlySet<string> = new Set(["tools", "human", "jitter", ...RESERVED_HOST_KEYS]);
const RESUME_HOST_KEYS: ReadonlySet<string> = new Set([...HOST_KEYS, "crash_after_record"]);
const RULES: ReadonlySet<string> = new Set([
  "numbers", "arithmetic", "compare", "text-form", "canonical", "columns", "messages",
  "text", "yaml", "dates", "infinity", "llm", "code",
]);
// Keys holding text that is not a case value: no codec form, no number checks.
const TEXT_KEYS: ReadonlySet<string> = new Set(["flow", "python_flow", "other", "files", "stdin"]);
const SECRET_KEYS: ReadonlySet<string> = new Set([
  "run", "resume", "backward", "env", "host", "stdin",
]);
const SECRET = /<key:([0-9]+)>/g;

/** Return `value` spelled for a message: JSON, a `bigint` as its digits, `null` when absent. */
export function spell(value: Value | undefined): string {
  if (value === undefined) return "null";
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number" && !Number.isFinite(value)) return String(value);
  if (Array.isArray(value)) return `[${value.map(spell).join(",")}]`;
  if (isMapping(value)) {
    const pairs = Object.entries(value).map(
      ([key, item]) => `${JSON.stringify(key)}:${spell(item)}`,
    );
    return `{${pairs.join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Return whether `value` is a mapping. */
function isMapping(value: Value | undefined): value is Mapping {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Return `mapping[key]` when it is an own key, else `fallback`. */
function get(mapping: Mapping, key: string, fallback?: Value): Value | undefined {
  return Object.hasOwn(mapping, key) ? mapping[key] : fallback;
}

/** Return whether `value` is true in Python's sense: not null, false, zero or empty. */
function truthy(value: Value | undefined): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  return Boolean(value);
}

/**
 * Return the case file `text` as YAML 1.2 core, untagged, with only text keys, each once.
 */
export function readCase(text: string): Case {
  const doc = parseDocument(text, {
    version: "1.2",
    schema: "core",
    intAsBigInt: true,
    uniqueKeys: false,
  });
  const error = doc.errors[0];
  if (error !== undefined) throw new Error(error.message);
  // YAML readers disagree on tags (`! 12` is text to some, `!!timestamp` a date), so a case
  // file has none. They are refused before any key, as Python's composer does.
  visit(doc, {
    Node(_, node) {
      if (node.tag !== undefined) {
        throw new Error(`a case file has no tags, found ${spell(node.tag)}`);
      }
    },
  });
  // Keys are checked level by level, mappings in document order, as Python's constructor does.
  const nodes: unknown[] = [doc.contents];
  for (const node of nodes) {
    if (isSeq(node)) {
      nodes.push(...node.items);
    } else if (isMap(node)) {
      const seen = new Set<string>();
      for (const pair of node.items) {
        const key = (isNode(pair.key) ? pair.key.toJS(doc) : pair.key) as Value;
        if (typeof key !== "string") throw new Error(`mapping key ${spell(key)} is not text`);
        if (seen.has(key)) throw new Error(`duplicate key ${spell(key)}`);
        seen.add(key);
        nodes.push(pair.value);
      }
    }
  }
  const value = doc.toJS() as Value;
  if (!isMapping(value)) throw new Error("a case file holds one mapping");
  return value;
}

/** Return `value`, throwing unless it is a mapping. */
function mapping(where: string, value: Value | undefined): Mapping {
  if (!isMapping(value)) throw new Error(`\`${where || "the case"}\` must be a mapping`);
  return value;
}

/**
 * Return `value`, throwing unless it is a mapping with only `allowed` keys and every `required`
 * one. A key in `known` but not in `allowed` belongs to another kind of case than `kind`.
 */
function keys(
  where: string,
  value: Value | undefined,
  allowed: ReadonlySet<string>,
  { known = new Set<string>(), kind = "", required = [] }: {
    known?: ReadonlySet<string>;
    kind?: string;
    required?: readonly string[];
  } = {},
): Mapping {
  const name = (key: string) => (where ? `${where}.${key}` : key);
  const map = mapping(where, value);
  for (const key of Object.keys(map)) {
    if (known.has(key) && !allowed.has(key)) {
      throw new Error(`\`${name(key)}\` is not a key of a \`${kind}\` case`);
    }
    if (!allowed.has(key)) throw new Error(`unknown key \`${name(key)}\``);
  }
  for (const key of required) {
    if (!Object.hasOwn(map, key)) throw new Error(`\`${name(key)}\` is required`);
  }
  return map;
}

/** Throw unless `value`, the host, its tools and their scripts use only the format's keys. */
function checkHost(value: Value | undefined, kind: Kind): void {
  const allowed = kind === "resume" ? RESUME_HOST_KEYS : HOST_KEYS;
  const host = keys("host", value, allowed, { known: RESUME_HOST_KEYS, kind });
  const tools = mapping("host.tools", get(host, "tools", {}));
  for (const [name, entry] of Object.entries(tools)) {
    const where = `host.tools.${name}`;
    const tool = keys(where, entry, new Set(["parameters", "description", "results"]), {
      required: ["parameters", "results"],
    });
    const results = tool["results"];
    const scripts: [string | null, Value | undefined][] = isMapping(results)
      ? Object.entries(results)
      : [[null, results]];
    for (const [node, script] of scripts) {
      if (!Array.isArray(script)) {
        throw new Error(`\`${where}.results\` must be a list or a mapping of lists`);
      }
      const label = node === null ? `${where}.results` : `${where}.results.${node}`;
      script.forEach((item, index) => {
        const at = `${label}[${index}]`;
        const step = keys(at, item, new Set(["return", "raise"]));
        if (Object.keys(step).length !== 1) {
          throw new Error(`\`${at}\` takes \`return\` or \`raise\``);
        }
        if (Object.hasOwn(step, "raise")) {
          const allowed = new Set(["class", "message", "retry_after"]);
          const error = keys(`${at}.raise`, step["raise"], allowed, {
            required: ["class", "message"],
          });
          if (Object.hasOwn(error, "retry_after") && error["class"] !== "TransientToolError") {
            throw new Error(`\`${at}.raise.retry_after\` is for TransientToolError`);
          }
        }
      });
    }
  }
  const human = mapping("host.human", get(host, "human", {}));
  if (Object.values(human).some((answers) => !Array.isArray(answers))) {
    throw new Error("`host.human` maps node ids to lists of answers");
  }
  if (Object.hasOwn(host, "human") && Object.hasOwn(tools, "ask_user")) {
    throw new Error("a case with `host.human` declares no `ask_user` tool");
  }
  const jitter = get(host, "jitter", []);
  if (
    !Array.isArray(jitter) ||
    jitter.some(
      (draw) => (typeof draw !== "bigint" && typeof draw !== "number") || !(draw >= 0 && draw < 1),
    )
  ) {
    throw new Error("`host.jitter` must be a list of numbers in [0, 1)");
  }
}

/** Return whether `value` names a kind of case. */
function isKind(value: Value | undefined): value is Kind {
  return typeof value === "string" && KIND_KEYS.has(value as Kind);
}

/** Return `case_` after checking it against the closed format; throw naming the problem. */
export function checkCase(case_: Case): Case {
  const kind = get(case_, "kind");
  if (!isKind(kind)) {
    const kinds = [...KIND_KEYS.keys()].join(", ");
    throw new Error(`\`kind\` must be one of ${kinds}, not ${spell(kind)}`);
  }
  keys("", case_, new Set([...COMMON, ...(KIND_KEYS.get(kind) ?? [])]), {
    known: ALL_KEYS,
    kind,
    required: ["kind", "expect", ...(REQUIRED.get(kind) ?? [])],
  });
  const optional = OPTIONAL.get(kind) ?? new Set<string>();
  for (const key of ["expect", "python_expect"]) {
    if (!Object.hasOwn(case_, key)) continue;
    const expect = keys(key, case_[key], EXPECT_KEYS.get(kind) ?? new Set<string>(), {
      known: ALL_EXPECT_KEYS,
      kind,
    });
    const groups = RESULTS.get(kind) ?? [];
    const results = Object.keys(expect).filter((name) => !optional.has(name));
    const matches = (group: readonly string[]) =>
      group.length === results.length && group.every((name) => results.includes(name));
    if (!groups.some(matches)) {
      const needs = groups
        .map((group) => group.map((name) => `\`${name}\``).join(" and "))
        .join(", or ");
      throw new Error(`\`${key}\` needs ${needs}`);
    }
  }
  if (Object.hasOwn(case_, "python_expect") && !truthy(get(case_, "rules"))) {
    throw new Error("`python_expect` needs `rules`");
  }
  for (const [key, what] of [["rules", "rule ids"], ["args", "text"]] as const) {
    const items = get(case_, key, []);
    if (!Array.isArray(items) || items.some((item) => typeof item !== "string")) {
      throw new Error(`\`${key}\` must be a list of ${what}`);
    }
  }
  const rules = get(case_, "rules", []) as string[];
  const unknown = rules.filter((rule) => !RULES.has(rule));
  if (unknown.length > 0) throw new Error(`unknown rule ids: ${unknown.join(", ")}`);
  for (const key of ["env", "files"]) {
    if (Object.values(mapping(key, get(case_, key, {}))).some((text) => typeof text !== "string")) {
      throw new Error(`\`${key}\` values must be text`);
    }
  }
  for (const key of TEXT_KEYS) {
    if (key !== "files" && typeof get(case_, key, "") !== "string") {
      throw new Error(`\`${key}\` must be text`);
    }
  }
  for (const [key, allowed] of [["run", RUN_KEYS], ["resume", RESUME_KEYS]] as const) {
    if (Object.hasOwn(case_, key)) {
      const args = keys(key, case_[key], allowed);
      if (get(args, "context", "present") !== "present") {
        throw new Error(`\`${key}.context\` must be \`present\``);
      }
    }
  }
  if (Object.hasOwn(case_, "backward")) {
    const backward = case_["backward"];
    if (!Array.isArray(backward) || backward.length === 0) {
      throw new Error("`backward` must be a non-empty list");
    }
    backward.forEach((entry, index) => {
      keys(`backward[${index}]`, entry, new Set(["feedback", "api_key", "headers"]), {
        required: ["feedback"],
      });
    });
  }
  if (Object.hasOwn(case_, "host")) checkHost(case_["host"], kind);
  // Both were checked as mappings above when present.
  const run = get(case_, "run", {}) as Mapping;
  if (kind === "resume") {
    const cut = get(get(case_, "host", {}) as Mapping, "crash_after_record");
    if (typeof cut !== "bigint" || cut < 0n) {
      throw new Error("a `resume` case needs `host.crash_after_record`, a record number");
    }
    // Python's `!= 0`, under which 0.0 and false equal 0 too.
    const workers = get(run, "num_workers");
    if (workers !== 0n && workers !== 0 && workers !== false) {
      throw new Error("`host.crash_after_record` needs `run.num_workers: 0`");
    }
  }
  if (kind === "backward" && get(run, "learnable") !== true) {
    throw new Error("a `backward` case needs `run.learnable: true`");
  }
  if (kind === "expr") {
    if (Object.hasOwn(case_, "expr") === Object.hasOwn(case_, "template")) {
      throw new Error("an `expr` case takes one of `expr` and `template`");
    }
    if (typeof get(case_, "expr", get(case_, "template")) !== "string") {
      throw new Error("`expr` or `template` must be text");
    }
    const mode = get(case_, "mode", "binding");
    if (mode !== "binding" && mode !== "condition" && mode !== "strict") {
      throw new Error(`\`mode\` must be one of binding, condition, strict, not ${spell(mode)}`);
    }
    if (
      mode === "condition" &&
      (Object.hasOwn(case_, "template") || Object.hasOwn(case_, "item"))
    ) {
      throw new Error("`mode: condition` takes an `expr` and no `item`");
    }
    if (
      mode === "strict" &&
      Object.hasOwn(case_, "template") &&
      (Object.hasOwn(case_, "item") || Object.hasOwn(case_, "typedefs"))
    ) {
      throw new Error("a strict `template` takes no `item` or `typedefs`");
    }
  }
  return case_;
}

/** Return a case value read from yian's JSON codec form; a bare non-finite float is an error. */
export function decode(value: Value): Value {
  if (typeof value === "number" && !Number.isFinite(value)) {
    // Spelled as Python spells the float.
    const bare = Number.isNaN(value) ? "nan" : value > 0 ? "inf" : "-inf";
    throw new Error(`a bare ${bare} is not a case value; write {$float: inf} or {$float: -inf}`);
  }
  if (Array.isArray(value)) return value.map(decode);
  if (!isMapping(value)) return value;
  const names = Object.keys(value);
  if (names.length === 1 && names[0] === "$float") {
    if (value["$float"] === "inf") return Infinity;
    if (value["$float"] === "-inf") return -Infinity;
  }
  let record = value;
  if (names.length === 1 && names[0] === "$record") {
    const inner = value["$record"];
    if (!isMapping(inner)) throw new Error("`$record` takes a mapping");
    record = inner;
  }
  return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, decode(item)]));
}

/** Return `value` with each `<key:N>` in its text replaced by `sk-conformance-N`. */
export function fillSecrets(value: Value): Value {
  if (typeof value === "string") return value.replace(SECRET, "sk-conformance-$1");
  if (Array.isArray(value)) return value.map(fillSecrets);
  if (isMapping(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fillSecrets(item)]));
  }
  return value;
}

/** Return the case as the TS runner runs it: values decoded and secrets filled in. */
export function prepare(case_: Case): Case {
  return Object.fromEntries(
    Object.entries(case_).map(([key, value]) => {
      const decoded = TEXT_KEYS.has(key) ? value : decode(value);
      return [key, SECRET_KEYS.has(key) ? fillSecrets(decoded) : decoded];
    }),
  );
}
