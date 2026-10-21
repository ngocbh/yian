/**
 * Run the language-neutral conformance cases in `conformance/cases/` against TypeScript yian.
 *
 * Each case file is one test, named by its id `<group>/<name>`. `conformance/README.md` defines
 * the case format and what each kind checks.
 *
 * Runs under Bun from source and under Node from `dist/`; only `node:` APIs and `yaml`.
 */

import { spawn } from "node:child_process";
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import type { IncomingHttpHeaders, IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { constants, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Document, isCollection, isMap, isNode, isSeq, parseDocument, visit } from "yaml";
import type { YAMLMap } from "yaml";

import {
  expressionTrees,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../src/compose/helpers/isinstance_types.ts";
import { flowFingerprint } from "../src/compile/manifest.ts";
import type { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { LoadError } from "../src/compose/helpers/errors.ts";
import { loadFlow } from "../src/compose/loader.ts";
import { ENTRIES, forget, nonDefaults, read, reload } from "../src/defaults.ts";
import type { Env } from "../src/defaults.ts";
import { compareCodeUnits, type Diagnostic, type Failure, reprText } from "../src/diagnostics.ts";
import type { RunEvent, StreamChunk } from "../src/events.ts";
import {
  constructorNames,
  evalExpr,
  evaluateWhenRecord,
  isinstanceTypeNames,
  ResolveMode,
  resolveInRecord,
} from "../src/expr/expressions.ts";
import { type ExprNode, parseExpr } from "../src/expr/grammar.ts";
import { evalBinding, renderTemplateRecord } from "../src/expr/template.ts";
import { virtualScheduler } from "../src/platform.ts";
import { type RunFlowOptions, type RunResult, runFlow } from "../src/run.ts";
import type { ToolContext } from "../src/tools/context.ts";
import {
  PermanentToolError,
  ReportedToolError,
  ToolCheckError,
  TransientToolError,
} from "../src/tools/errors.ts";
import { tool } from "../src/tools/function_tools.ts";
import type { AskUserHandler, AskUserRequest, ToolEntry } from "../src/tools/registry.ts";
import { canonicalJson, logJson, parseJson } from "../src/typesys/json.ts";
import type { JsonMode, JsonObject } from "../src/typesys/json.ts";
import { readTypedefs, typeFor } from "../src/typesys/types.ts";
import { buildValueAs, FileRef, isRecord } from "../src/typesys/values.ts";
import type { Type, Value as YianValue, ValueRecord } from "../src/typesys/values.ts";
import { validate } from "./schema.ts";
import type { Schema } from "./schema.ts";

/** A case value: an int is a `bigint`, a float a `number`. */
export type Value = null | boolean | bigint | number | string | Value[] | Case;

/** A case file, as read, and any mapping in it. */
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
  | "cli"
  | "json";

// The closed format, as conformance/README.md defines it. Sets, and tables keyed by a checked
// `Kind`, so a case value such as `kind: toString` never finds an `Object.prototype` member.
const COMMON: ReadonlySet<string> = new Set([
  "kind", "rules", "pending", "python_skip", "expect", "python_expect",
]);
const FLOW: ReadonlySet<string> = new Set(["flow", "python_flow", "files", "python_files", "env"]);
const RUN: ReadonlySet<string> = new Set([...FLOW, "inputs", "run", "host"]);
const KIND_KEYS = {
  load: FLOW,
  run: RUN,
  resume: new Set([...RUN, "resume"]),
  backward: new Set([...RUN, "backward"]),
  expr: new Set(["expr", "template", "values", "item", "mode", "typedefs", "env"]),
  value: new Set(["type", "typedefs", "value", "env"]),
  defaults: new Set(["env"]),
  // Both runners load `flow` and `other` as written: no `python_flow` or `python_files`.
  fingerprint: new Set(["flow", "files", "env", "other"]),
  cli: new Set([...FLOW, "args", "stdin", "signal", "http"]),
  json: new Set(["read", "mode", "write", "form"]),
} satisfies Record<Kind, ReadonlySet<string>>;
const REQUIRED: Partial<Record<Kind, readonly string[]>> = {
  load: ["flow"], run: ["flow"], resume: ["flow"], backward: ["flow", "backward"],
  value: ["type", "value"], fingerprint: ["flow", "other"],
};
// Each kind's `expect` holds exactly one of its result groups, and any of its optional keys.
const RESULTS = {
  load: [["load"]],
  run: [["raises"], ["output"]],
  resume: [["raises"], ["output"]],
  backward: [["raises"], ["output"]],
  expr: [["value"], ["raises"]],
  value: [["value", "kind"], ["raises"]],
  defaults: [["defaults"], ["raises"]],
  fingerprint: [["fingerprint"]],
  cli: [["exit"]],
  json: [["value"], ["text"], ["raises"]],
} satisfies Record<Kind, readonly (readonly string[])[]>;
const RUN_OPTIONAL: ReadonlySet<string> = new Set([
  "trace", "events", "stream", "tool_calls", "memory",
]);
const OPTIONAL: Partial<Record<Kind, ReadonlySet<string>>> = {
  run: RUN_OPTIONAL,
  resume: RUN_OPTIONAL,
  backward: new Set([...RUN_OPTIONAL, "backward"]),
  cli: new Set(["stdout", "stderr", "json", "json_lines", "exists"]),
};
const KIND_NAMES = Object.keys(KIND_KEYS) as Kind[];

/** Return the keys a `kind` case's `expect` may hold: its result groups' and optional keys. */
function expectKeys(kind: Kind): ReadonlySet<string> {
  return new Set([...(OPTIONAL[kind] ?? []), ...RESULTS[kind].flat()]);
}

const ALL_KEYS: ReadonlySet<string> = new Set([
  ...COMMON, ...KIND_NAMES.flatMap((kind) => [...KIND_KEYS[kind]]),
]);
const ALL_EXPECT_KEYS: ReadonlySet<string> = new Set(
  KIND_NAMES.flatMap((kind) => [...expectKeys(kind)]),
);
const RUN_KEYS: ReadonlySet<string> = new Set([
  "run_id", "num_workers", "learnable", "llm_config", "context",
]);
const RESUME_KEYS: ReadonlySet<string> = new Set(["num_workers", "api_key", "headers", "context"]);
// Host keys the format reserves and this runner doesn't support yet.
const RESERVED_HOST_KEYS: readonly string[] = ["models", "steering", "stop", "memory", "skills"];
const HOST_KEYS: ReadonlySet<string> = new Set(["tools", "human", "jitter", ...RESERVED_HOST_KEYS]);
const RESUME_HOST_KEYS: ReadonlySet<string> = new Set([...HOST_KEYS, "crash_after_record"]);
/** The `[section, key]` pairs a case may not use yet: it fails before it is set up. */
const RESERVED: readonly (readonly [string, string])[] = [
  ...RESERVED_HOST_KEYS.map((key) => ["host", key] as const),
  ["expect", "memory"],
];
const RULES: ReadonlySet<string> = new Set([
  "numbers", "int-in-float", "arithmetic", "compare", "text-form", "canonical", "columns",
  "messages", "text", "yaml", "dates", "infinity", "llm", "code", "failure-codes",
]);
// Keys holding text, or plain JSON (`http`), that is not a case value: no codec form, no number
// checks.
const TEXT_KEYS: ReadonlySet<string> = new Set([
  "flow", "python_flow", "other", "files", "python_files", "stdin", "read", "http",
]);
const SECRET_KEYS: ReadonlySet<string> = new Set([
  "run", "resume", "backward", "env", "host", "stdin", "http",
]);
const SECRET = /<key:([0-9]+)>/g;
const ENV_SUFFIXES: readonly string[] = ["_API_KEY", "_BASE_URL", "_ENDPOINT", "_API_VERSION"];

/** Return `value` spelled for a message: JSON, a `bigint` as its digits, `null` when absent. */
function spell(value: Value | undefined): string {
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

/** Return whether `value` is a mapping: a plain object, as a case file's mappings are. */
function isMapping(value: unknown): value is Case {
  if (typeof value !== "object" || value === null) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/** Return `mapping[key]` when it is an own key, else `fallback`. */
function get(mapping: Case, key: string, fallback?: Value): Value | undefined {
  return Object.hasOwn(mapping, key) ? mapping[key] : fallback;
}

/** Return whether `value` is true in Python's sense: not null, false, zero or empty. */
function truthy(value: Value | undefined): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  // NaN is true to Python.
  if (typeof value === "number") return value !== 0;
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
    if (isSeq(node)) nodes.push(...node.items);
    else if (isMap(node)) checkKeys(doc, node, (value) => nodes.push(value));
  }
  // Case files are trusted, and Python has no alias limit.
  const value = doc.toJS({ maxAliasCount: -1 }) as Value;
  if (!isMapping(value)) throw new Error("a case file holds one mapping");
  return value;
}

/** Throw unless every key of `map` is text and appears once; hand each value to `next`. */
function checkKeys(doc: Document, map: YAMLMap, next: (value: unknown) => void): void {
  const seen = new Set<string>();
  for (const pair of map.items) {
    // Python builds a collection key in full first, so a bad key inside it is reported first.
    if (isCollection(pair.key)) checkDeep(doc, pair.key);
    const key = (isNode(pair.key) ? pair.key.toJS(doc, { maxAliasCount: -1 }) : pair.key) as Value;
    if (typeof key !== "string") throw new Error(`mapping key ${spell(key)} is not text`);
    if (seen.has(key)) throw new Error(`duplicate key ${spell(key)}`);
    seen.add(key);
    next(pair.value);
  }
}

/** Throw unless every mapping in `node` has only text keys, each once, depth first. */
function checkDeep(doc: Document, node: unknown): void {
  if (isSeq(node)) node.items.forEach((item) => checkDeep(doc, item));
  else if (isMap(node)) checkKeys(doc, node, (value) => checkDeep(doc, value));
}

/** Return `value`, throwing unless it is a mapping. */
function mapping(where: string, value: Value | undefined): Case {
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
): Case {
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
    const toolKeys = new Set(["parameters", "description", "concurrency_safe", "results"]);
    const tool = keys(where, entry, toolKeys, { required: ["parameters", "results"] });
    if (Object.hasOwn(tool, "concurrency_safe") && typeof tool["concurrency_safe"] !== "boolean") {
      throw new Error(`\`${where}.concurrency_safe\` must be a boolean`);
    }
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
          const raiseKeys = new Set(["class", "message", "retry_after"]);
          const error = keys(`${at}.raise`, step["raise"], raiseKeys, {
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

/**
 * Throw unless `value` is a list of `{request, response, repeat}` exchanges. A request is
 * `{method, url, headers, json, form}` with at most one of `json` and `form`, a response
 * `{status, json, delay}`, and only the last exchange may repeat.
 */
function checkHttp(value: Value | undefined): void {
  if (!Array.isArray(value)) throw new Error("`http` must be a list");
  value.forEach((item, index) => {
    const where = `http[${index}]`;
    const exchange = keys(where, item, new Set(["request", "response", "repeat"]), {
      required: ["request", "response"],
    });
    const repeat = get(exchange, "repeat", false);
    if (typeof repeat !== "boolean") throw new Error(`\`${where}.repeat\` must be true or false`);
    if (repeat && index !== value.length - 1) {
      throw new Error(`\`${where}.repeat\`: only the last exchange repeats`);
    }
    const requestKeys = new Set(["method", "url", "headers", "json", "form"]);
    const request = keys(`${where}.request`, exchange["request"], requestKeys, {
      required: ["method", "url"],
    });
    if (typeof request["method"] !== "string" || typeof request["url"] !== "string") {
      throw new Error(`\`${where}.request\` \`method\` and \`url\` must be text`);
    }
    if (Object.hasOwn(request, "json") && Object.hasOwn(request, "form")) {
      throw new Error(`\`${where}.request\` takes at most one of \`json\` and \`form\``);
    }
    for (const key of ["headers", "form"]) {
      const texts = Object.values(mapping(`${where}.request.${key}`, get(request, key, {})));
      if (texts.some((text) => typeof text !== "string")) {
        throw new Error(`\`${where}.request.${key}\` values must be text`);
      }
    }
    const responseKeys = new Set(["status", "json", "delay"]);
    const response = keys(`${where}.response`, exchange["response"], responseKeys, {
      required: ["status"],
    });
    const status = response["status"];
    if (typeof status !== "bigint" || status < 100n || status > 599n) {
      throw new Error(`\`${where}.response.status\` must be an int from 100 to 599`);
    }
    const delay = get(response, "delay", 0n);
    // Finite, as Python's runner needs: a wait can't take an infinite timeout.
    const finite = typeof delay === "number" && delay >= 0 && delay < Infinity;
    if (!finite && !(typeof delay === "bigint" && delay >= 0n)) {
      throw new Error(`\`${where}.response.delay\` must be a finite non-negative number`);
    }
  });
}

/** Return whether `value` names a kind of case. */
function isKind(value: Value | undefined): value is Kind {
  return typeof value === "string" && Object.hasOwn(KIND_KEYS, value);
}

/** Return `case_` after checking it against the closed format; throw naming the problem. */
export function checkCase(case_: Case): Case {
  const kind = get(case_, "kind");
  if (!isKind(kind)) {
    const kinds = KIND_NAMES.join(", ");
    throw new Error(`\`kind\` must be one of ${kinds}, not ${spell(kind)}`);
  }
  keys("", case_, new Set([...COMMON, ...KIND_KEYS[kind]]), {
    known: ALL_KEYS,
    kind,
    required: ["kind", "expect", ...(REQUIRED[kind] ?? [])],
  });
  const optional = OPTIONAL[kind] ?? new Set<string>();
  for (const key of ["expect", "python_expect"]) {
    if (!Object.hasOwn(case_, key)) continue;
    const expect = keys(key, case_[key], expectKeys(kind), {
      known: ALL_EXPECT_KEYS,
      kind,
    });
    const groups: readonly (readonly string[])[] = RESULTS[kind];
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
  for (const key of ["env", "files", "python_files"]) {
    if (Object.values(mapping(key, get(case_, key, {}))).some((text) => typeof text !== "string")) {
      throw new Error(`\`${key}\` values must be text`);
    }
  }
  for (const key of TEXT_KEYS) {
    if (["files", "python_files", "http"].includes(key)) continue;
    if (typeof get(case_, key, "") !== "string") {
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
  if (Object.hasOwn(case_, "signal")) {
    const signal = keys("signal", case_["signal"], new Set(["name", "after"]), {
      required: ["name", "after"],
    });
    const name = get(signal, "name");
    if (name !== "SIGINT" && name !== "SIGTERM") {
      throw new Error("`signal.name` must be SIGINT or SIGTERM");
    }
    const after = get(signal, "after");
    if (typeof after !== "string" || after === "") {
      throw new Error("`signal.after` must be non-empty text");
    }
  }
  if (Object.hasOwn(case_, "http")) checkHttp(case_["http"]);
  for (const key of ["expect", "python_expect"]) {
    // Checked as a mapping above when present.
    const expect = get(case_, key, {}) as Case;
    const paths = mapping(`${key}.exists`, get(expect, "exists", {}));
    for (const [path, exists] of Object.entries(paths)) {
      if (typeof exists !== "boolean") {
        throw new Error(`\`${key}.exists\` values must be true or false`);
      }
      if (path.startsWith("/") || path.split("/").includes("..")) {
        throw new Error(`\`${key}.exists\`: \`${path}\` must be a relative path without \`..\``);
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
  const run = get(case_, "run", {}) as Case;
  if (kind === "resume") {
    const cut = get(get(case_, "host", {}) as Case, "crash_after_record");
    if (cut !== "end" && (typeof cut !== "bigint" || cut < 0n)) {
      throw new Error("a `resume` case needs `host.crash_after_record`, a record number or `end`");
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
  if (kind === "json") {
    const read = Object.hasOwn(case_, "read");
    if (read === Object.hasOwn(case_, "write")) {
      throw new Error("a `json` case takes one of `read` and `write`");
    }
    const mode = get(case_, "mode");
    const form = get(case_, "form");
    if (read && ((mode !== "codec" && mode !== "plain") || form !== undefined)) {
      throw new Error("`read` takes `mode` (codec or plain), and no `form`");
    }
    if (!read && ((form !== "canonical" && form !== "log") || mode !== undefined)) {
      throw new Error("`write` takes `form` (canonical or log), and no `mode`");
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
function fillSecrets(value: Value): Value {
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

/** Return the first directory above this file holding `conformance/cases`: the repository. */
function findRoot(): string {
  // `ts/tests/` from source, `ts/dist/tests/` when built.
  let dir = dirname(fileURLToPath(import.meta.url));
  while (!existsSync(join(dir, "conformance", "cases"))) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error("no `conformance/cases` above the conformance runner");
    dir = parent;
  }
  return dir;
}

const CASES = join(findRoot(), "conformance", "cases");

/** The runner's own "this case failed" error, carrying only a message. */
export class CaseFailure extends Error {
  override name = "CaseFailure";
}

/** Return `value` as Python's `str()` writes it into a message: text as is, else spelled. */
function str(value: Value | undefined): string {
  return typeof value === "string" ? value : spell(value);
}

/** How `same` compares. */
export interface SameFlags {
  /** An int and a float compare by value. */
  byValue?: boolean;
  /** An expected mapping without `message` doesn't compare it. */
  optionalMessage?: boolean;
}

/** Return whether `actual` equals `expected`, kinds included: a `bigint` int, a `number` float. */
export function same(actual: unknown, expected: unknown, flags: SameFlags = {}): boolean {
  if (isMapping(expected)) {
    if (!isMapping(actual)) return false;
    const dropMessage = flags.optionalMessage === true && !Object.hasOwn(expected, "message");
    const names = Object.keys(actual).filter((key) => !(dropMessage && key === "message"));
    const wanted = Object.keys(expected);
    return (
      names.length === wanted.length &&
      wanted.every((key) => Object.hasOwn(actual, key) && same(actual[key], expected[key], flags))
    );
  }
  if (Array.isArray(expected)) {
    return (
      Array.isArray(actual) &&
      actual.length === expected.length &&
      expected.every((item, index) => same(actual[index], item, flags))
    );
  }
  const numeric = (value: unknown) => typeof value === "bigint" || typeof value === "number";
  if (flags.byValue === true && numeric(actual) && numeric(expected)) {
    // `==` compares a bigint and a number exactly.
    return (actual as bigint | number) == (expected as bigint | number);
  }
  return actual === expected;
}

/** Return `value` as YAML, a float always with a fraction (`1.0`), lines never folded. */
function dump(value: unknown): string {
  const doc = new Document(value);
  visit(doc, {
    Scalar(_, node) {
      if (typeof node.value === "number") node.minFractionDigits = 1;
    },
  });
  return doc.toString({ lineWidth: 0 });
}

/** Throw a `CaseFailure` showing both sides as YAML unless `actual` is the same as `expected`. */
export function compare(
  what: string,
  actual: unknown,
  expected: unknown,
  flags: SameFlags = {},
): void {
  if (!same(actual, expected, flags)) {
    throw new CaseFailure(
      `${what} differs\n--- actual\n${dump(actual)}--- expected\n${dump(expected)}`,
    );
  }
}

/**
 * TS error class names that differ from the case names, mapped to the case names. A change that
 * introduces such a class adds its entry here and the matching row to the error-class table in
 * `conformance/README.md`.
 */
export const ERROR_CLASSES: ReadonlyMap<string, string> = new Map([
  // TS throws a RangeError wherever Python raises ValueError at a yian boundary.
  ["RangeError", "ValueError"],
]);

/**
 * Return `error` in the case's `raises` form: `{class, message}`, plus `path` for a
 * `TypeCheckError`. The class is `error.name` read through `classes`, so every yian error class
 * sets `override name = "<Class>"`. A thrown non-`Error` is named by its `typeof`.
 */
export function raised(error: unknown, classes: ReadonlyMap<string, string> = ERROR_CLASSES): Case {
  if (!(error instanceof Error)) return { class: typeof error, message: String(error) };
  const form: Case = { class: classes.get(error.name) ?? error.name, message: error.message };
  const path: unknown = (error as { path?: unknown }).path;
  if (error.name === "TypeCheckError" && Array.isArray(path)) {
    form["path"] = (path as unknown[]).map((step) =>
      typeof step === "number" ? BigInt(step) : (step as Value),
    );
  }
  return form;
}

/**
 * Return the error `call()` throws in `raises` form; throw a `CaseFailure` unless it is the one
 * expected. A `CaseFailure` from `call()` is the case's own failure and passes through.
 */
export async function expectRaises(expected: Case, call: () => unknown): Promise<Case> {
  try {
    await call();
  } catch (error) {
    if (error instanceof CaseFailure) throw error;
    const form = raised(error);
    compare("raises", form, expected);
    return form;
  }
  throw new CaseFailure(`expected ${str(get(expected, "class"))} to be raised, but nothing was`);
}

/**
 * Compare the results `compute()` returns, by name, with `expect`, as `flags` say. When `expect`
 * has `raises`, compare the error `compute()` throws instead.
 */
export async function results(
  expect: Case,
  compute: () => Record<string, unknown> | Promise<Record<string, unknown>>,
  flags: SameFlags = {},
): Promise<void> {
  if (Object.hasOwn(expect, "raises")) {
    await expectRaises(expect["raises"] as Case, compute);
    return;
  }
  for (const [name, actual] of Object.entries(await compute())) {
    if (!Object.hasOwn(expect, name)) throw new CaseFailure(`unexpected result ${name}`);
    compare(name, actual, expect[name], flags);
  }
}

/** A case's directory and the environment yian gets. */
export interface CaseContext {
  dir: string;
  env: Record<string, string>;
}

/** Return the `Env` that reads the case's environment; a name that is not an own key is unset. */
function caseEnv(ctx: CaseContext): Env {
  return (name) => (Object.hasOwn(ctx.env, name) ? ctx.env[name] : undefined);
}

/** Run one prepared case of a kind; throw (a `CaseFailure`, normally) when it fails. */
export type KindRunner = (case_: Case, ctx: CaseContext) => void | Promise<void>;

// `RESERVED` shrinks as the kinds that read `host` and `expect.memory` support its keys.
/**
 * The kind runners. A case whose kind has none fails. A kind is added by replacing its comment
 * line with `<kind>: run<Kind>,` and writing `run<Kind>` under the kind's section header below.
 */
export const KINDS: Partial<Record<Kind, KindRunner>> = {
  load: runLoad,
  run: runRun,
  // resume: not run yet
  // backward: not run yet
  expr: runExpr,
  value: runValue,
  defaults: runDefaults,
  fingerprint: runFingerprint,
  cli: runCli,
  json: runJson,
};

/**
 * Make the case directory under the OS temp directory: `flow.yaml`, the `files`, and a copy of
 * `fixturesDir` as `fixtures/` when it is a directory. Except for `cli`, the environment is the
 * process's without yian's own variables, plus `YIAN_AUTH_FILE` naming a missing file in the
 * directory, plus the case's `env`; for `cli` it is the case's `env` alone.
 */
export function setUp(case_: Case, fixturesDir: string): CaseContext {
  const dir = mkdtempSync(join(tmpdir(), "yian-conformance-"));
  try {
    const flow = get(case_, "flow");
    if (typeof flow === "string") writeFileSync(join(dir, "flow.yaml"), flow);
    // yian refuses an auth file, or a directory holding one, that others can read; `mkdtempSync`
    // makes the case directory 0700.
    for (const [name, text] of Object.entries(get(case_, "files", {}) as Record<string, string>)) {
      const path = join(dir, name);
      mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
      writeFileSync(path, text, { mode: 0o600 });
    }
    if (statSync(fixturesDir, { throwIfNoEntry: false })?.isDirectory()) {
      cpSync(fixturesDir, join(dir, "fixtures"), { recursive: true });
    }
  } catch (error) {
    rmSync(dir, { recursive: true, force: true });
    throw error;
  }
  const env = get(case_, "env", {}) as Record<string, string>;
  if (case_["kind"] === "cli") return { dir, env: { ...env } };
  const inherited = Object.entries(process.env).filter(
    (entry): entry is [string, string] =>
      entry[1] !== undefined &&
      !entry[0].startsWith("YIAN_") &&
      !ENV_SUFFIXES.some((suffix) => entry[0].endsWith(suffix)),
  );
  return {
    dir,
    env: { ...Object.fromEntries(inherited), YIAN_AUTH_FILE: join(dir, "auth.json"), ...env },
  };
}

/** Return the codes of a `load` case's expected diagnostics. */
function diagnosticCodes(case_: Case): Value[] {
  if (case_["kind"] !== "load") return [];
  const load = get(case_["expect"] as Case, "load");
  const diagnostics = isMapping(load) ? get(load, "diagnostics") : undefined;
  if (!Array.isArray(diagnostics)) return [];
  return diagnostics.flatMap((diagnostic) =>
    isMapping(diagnostic) && Object.hasOwn(diagnostic, "code") ? [diagnostic["code"] ?? null] : [],
  );
}

/** Return the message a failing case reports for `error`. */
function failureText(error: unknown): string {
  if (error instanceof CaseFailure) return error.message;
  if (!(error instanceof Error)) return String(error);
  const header = `${error.name}: ${error.message}`;
  const stack = error.stack ?? "";
  return stack.startsWith(header) ? stack : `${header}\n${stack}`;
}

/** Run one prepared case; return its failure message, or `undefined` when it passes. */
async function runCase(
  case_: Case,
  kinds: Partial<Record<Kind, KindRunner>>,
  fixturesDir: string,
): Promise<string | undefined> {
  for (const [section, key] of RESERVED) {
    if (Object.hasOwn(get(case_, section, {}) as Case, key)) {
      return `\`${section}.${key}\` is not supported by the TS runner yet`;
    }
  }
  const kind = case_["kind"] as Kind;
  const run = Object.hasOwn(kinds, kind) ? kinds[kind] : undefined;
  if (run === undefined) return `the TS runner doesn't run \`${kind}\` cases yet`;
  let ctx: CaseContext | undefined;
  try {
    ctx = setUp(case_, fixturesDir);
    await run(case_, ctx);
    return undefined;
  } catch (error) {
    return failureText(error);
  } finally {
    if (ctx !== undefined) rmSync(ctx.dir, { recursive: true, force: true });
  }
}

const USAGE =
  "usage: conformance [--group GROUP] [--kind KIND] [--case GROUP/NAME] [--rules RULE] " +
  "[--diagnostic-code CODE]";

/** What `main` reads cases from, runs them with and writes lines to. */
export interface MainOptions {
  /** The cases, one directory per group; the fixtures are its sibling `fixtures/`. */
  casesDir?: string;
  kinds?: Partial<Record<Kind, KindRunner>>;
  /** Write one line of the report. */
  write?: (line: string) => void;
  /** Write one line of a usage error. */
  writeError?: (line: string) => void;
}

/**
 * Run the cases the flags in `argv` select; report one `FAIL` line per failing case and a
 * summary. Return the exit code: 0 when no case fails, 1 when one does or none matches, 2 on a
 * usage error. Each flag is repeatable; values of one flag are alternatives, flags all apply.
 */
export async function main(
  argv: readonly string[],
  {
    casesDir = CASES,
    kinds = KINDS,
    write = (line) => {
      process.stdout.write(`${line}\n`);
    },
    writeError = (line) => {
      process.stderr.write(`${line}\n`);
    },
  }: MainOptions = {},
): Promise<number> {
  const option = { type: "string", multiple: true } as const;
  let flags: Partial<Record<"group" | "kind" | "case" | "rules" | "diagnostic-code", string[]>>;
  try {
    flags = parseArgs({
      args: [...argv],
      strict: true,
      options: {
        group: option, kind: option, case: option, rules: option, "diagnostic-code": option,
      },
    }).values;
  } catch {
    writeError(USAGE);
    return 2;
  }
  const allows = (values: string[] | undefined, matches: (wanted: string) => boolean) =>
    values === undefined || values.some(matches);
  // The default sort compares UTF-16 code units.
  const ids = readdirSync(casesDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((group) =>
      readdirSync(join(casesDir, group.name))
        .filter((name) => name.endsWith(".yaml"))
        .map((name) => `${group.name}/${name.slice(0, -".yaml".length)}`),
    )
    .sort();
  const selected: ({ id: string; case_: Case } | { id: string; unreadable: string })[] = [];
  for (const id of ids) {
    const group = id.slice(0, id.indexOf("/"));
    if (!allows(flags.group, (wanted) => wanted === group)) continue;
    if (!allows(flags.case, (wanted) => wanted === id)) continue;
    let case_: Case;
    try {
      case_ = prepare(checkCase(readCase(readFileSync(join(casesDir, `${id}.yaml`), "utf8"))));
    } catch (error) {
      // `--kind`, `--rules` and `--diagnostic-code` never hide a broken file; `--group` and
      // `--case`, read from its path, still select or skip it.
      selected.push({ id, unreadable: error instanceof Error ? error.message : String(error) });
      continue;
    }
    const rules = get(case_, "rules", []) as string[];
    const codes = diagnosticCodes(case_);
    if (
      allows(flags.kind, (wanted) => wanted === case_["kind"]) &&
      allows(flags.rules, (wanted) => rules.includes(wanted)) &&
      allows(flags["diagnostic-code"], (wanted) => codes.includes(wanted))
    ) {
      selected.push({ id, case_ });
    }
  }
  if (selected.length === 0) {
    write("no case matches");
    return 1;
  }
  let passed = 0;
  let failed = 0;
  let pending = 0;
  const fixturesDir = join(dirname(casesDir), "fixtures");
  for (const entry of selected) {
    let failure: string | undefined;
    if ("unreadable" in entry) {
      failure = entry.unreadable;
    } else {
      failure = await runCase(entry.case_, kinds, fixturesDir);
      if (Object.hasOwn(entry.case_, "pending")) {
        if (failure !== undefined) {
          pending += 1;
          continue;
        }
        const marker = str(entry.case_["pending"]);
        failure = `passes, but is marked pending (${marker}); remove the marker`;
      }
    }
    if (failure === undefined) {
      passed += 1;
      continue;
    }
    failed += 1;
    const [first, ...rest] = failure.replace(/\n$/, "").split("\n");
    write(`FAIL ${entry.id}: ${first}`);
    for (const line of rest) write(line === "" ? "" : `  ${line}`);
  }
  const cases = selected.length === 1 ? "case" : "cases";
  write(`${selected.length} ${cases}: ${passed} passed, ${failed} failed, ${pending} pending`);
  return failed > 0 ? 1 : 0;
}

// ---- load ----

/**
 * Load `text` as the case's `flow.yaml` under the defaults `env` sets, resolving `uses:` paths
 * and CODE references in the case directory.
 */
async function loadText(text: string, ctx: CaseContext): Promise<ComposedFlow> {
  // The defaults stay installed across the load's awaits; safe, since cases run one at a time.
  try {
    reload(caseEnv(ctx));
    const options = { searchPaths: [ctx.dir], sourceLabel: "flow.yaml", baseDir: ctx.dir };
    return await loadFlow(text, options);
  } finally {
    forget();
  }
}

/** Return `diagnostic` in the case's diagnostic form. */
function diagnosticForm(diagnostic: Diagnostic): Case {
  const frames = diagnostic.frames.map((frame): Case => {
    const { startLine, startCol, endLine, endCol } = frame.range;
    return {
      label: frame.label,
      range: [startLine, startCol, endLine, endCol].map(BigInt),
      ...(frame.note ? { note: frame.note } : {}),
    };
  });
  const form: Case = { code: diagnostic.code, message: diagnostic.message };
  if (frames.length > 0) form["frames"] = frames;
  if (diagnostic.notes.length > 0) form["notes"] = [...diagnostic.notes];
  return form;
}

/** Run a `load` case: load `flow`; the result is `ok` or the `LoadError`'s diagnostic. */
async function runLoad(case_: Case, ctx: CaseContext): Promise<void> {
  const compute = async () => {
    try {
      await loadText(case_["flow"] as string, ctx);
    } catch (error) {
      if (!(error instanceof LoadError)) throw error;
      return { load: { diagnostics: [diagnosticForm(error.diagnostic)] } };
    }
    return { load: "ok" };
  };
  await results(case_["expect"] as Case, compute, { optionalMessage: true });
}

// ---- run ----

/** The `runFlow` option each `run` key sets. */
const RUN_OPTIONS: Readonly<Record<string, keyof RunFlowOptions>> = {
  run_id: "runId",
  num_workers: "numWorkers",
  learnable: "learnable",
  llm_config: "llmConfig",
  context: "context",
};

/** Return the `run` arguments as `runFlow` options: an int as a JS number, `context` an object. */
function runOptions(run: Case): RunFlowOptions {
  return Object.fromEntries(
    Object.entries(run).map(([key, value]) => [
      RUN_OPTIONS[key],
      key === "context" ? {} : typeof value === "bigint" ? Number(value) : value,
    ]),
  );
}

/** An error class a script's `raise` can name. */
type ErrorClass = new (message: string, options?: ErrorOptions & { retryAfter?: number }) => Error;

/** The classes a stub tool's `raise` can name; any other name is a plain `Error` of that name. */
const TOOL_ERRORS: ReadonlyMap<string, ErrorClass> = new Map<string, ErrorClass>([
  ["PermanentToolError", PermanentToolError],
  ["TransientToolError", TransientToolError],
  ["ReportedToolError", ReportedToolError],
  ["ToolCheckError", ToolCheckError],
]);

/**
 * Return a script's error `{class, message, retry_after}`: an error of the class `classes` names,
 * else a plain `Error` named `class` (`TimeoutError` and `ConnectionError` among them).
 */
function scriptedError(error: Case, classes: ReadonlyMap<string, ErrorClass>): Error {
  const name = error["class"] as string;
  const message = error["message"] as string;
  const Class = classes.get(name);
  if (Class === undefined) return Object.assign(new Error(message), { name });
  const retryAfter = Object.hasOwn(error, "retry_after") ? Number(error["retry_after"]) : undefined;
  return new Class(message, retryAfter === undefined ? {} : { retryAfter });
}

/** Return `name`, a camelCase field name, in snake_case. */
function snakeCase(name: string): string {
  return name.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Return an `ask_user` request as a case writes its `args`: snake_case keys, each question's
 * too, null fields left out.
 */
function askUserArgs(request: AskUserRequest): Case {
  const args: Case = {};
  for (const [key, value] of Object.entries(request)) {
    if (value === null) continue;
    args[snakeCase(key)] =
      key === "questions"
        ? (value as Value[]).map((question) =>
            isMapping(question)
              ? Object.fromEntries(Object.entries(question).map(([name, item]) => [snakeCase(name), item]))
              : question,
          )
        : (value as Value);
  }
  return args;
}

/** Return how many records the log file at `path` holds: the `nextSequence` of its next append. */
function nextSequence(path: string): number {
  return existsSync(path) ? readFileSync(path, "utf8").split("\n").length - 1 : 0;
}

/** A case's scripted host: what `runFlow` gets from it, what it records, and its check. */
interface Host {
  /** A stub per `host.tools` entry, and `ask_user` for `host.human`. */
  tools: Record<string, ToolEntry>;
  /** The draws retry backoffs take, from `host.jitter`; `null` for the implementation's own. */
  random: (() => number) | null;
  /** Each tool's calls in order, by tool name, as `{args, call_id, attempt, node}`. */
  calls: Map<string, Case[]>;
  /** Each script entry taken, in order, with the log's `nextSequence` when it was taken. */
  taken: { label: string; entry: Value; sequence: number }[];
  /**
   * Throw a `CaseFailure` when a script ran out or was missing; return a message per script with
   * entries left.
   */
  check: () => string[];
}

/**
 * Return the case's host: each stub tool call is recorded and takes its script's next entry, as
 * each `ask_user` call takes the next `host.human` answer of its node and each backoff the next
 * `host.jitter` draw. `log` is the run's log file.
 */
function host(case_: Case, log: string): Host {
  const spec = get(case_, "host", {}) as Case;
  const calls = new Map<string, Case[]>();
  const problems: string[] = [];
  const taken: Host["taken"] = [];
  const scripts = new Map<string, Value[]>(
    Object.entries(get(spec, "human", {}) as Case).map(([node, answers]) => [
      `host.human.${node}`,
      [...(answers as Value[])],
    ]),
  );

  const take = (label: string, what: string): Value => {
    const script = scripts.get(label);
    if (script === undefined) {
      problems.push(`${what} found no \`${label}\` list`);
    } else if (script.length === 0) {
      problems.push(`${what} found \`${label}\` used up`);
    } else {
      const entry = script.shift()!;
      taken.push({ label, entry, sequence: nextSequence(log) });
      return entry;
    }
    // Permanent, so a TOOL node doesn't retry the overrun; `check()` reports it either way.
    throw new PermanentToolError(`\`${label}\` is used up`);
  };

  const call = (name: string, label: string, ctx: ToolContext, args: Case): Value => {
    const record = { args, call_id: ctx.callId, attempt: BigInt(ctx.attempt), node: ctx.node };
    calls.set(name, [...(calls.get(name) ?? []), record]);
    return take(label, `a call to \`${name}\` from node ${reprText(ctx.node)}`);
  };

  const tools: [string, ToolEntry][] = [];
  for (const [name, entry] of Object.entries(get(spec, "tools", {}) as Case)) {
    const stub = entry as Case;
    const results = stub["results"];
    const byNode = isMapping(results);
    if (byNode) {
      for (const [node, script] of Object.entries(results)) {
        scripts.set(`host.tools.${name}.results.${node}`, [...(script as Value[])]);
      }
    } else {
      scripts.set(`host.tools.${name}.results`, [...(results as Value[])]);
    }
    const execute = (args: Record<string, unknown>, ctx: ToolContext): unknown => {
      const label = `host.tools.${name}.results${byNode ? `.${ctx.node}` : ""}`;
      const step = call(name, label, ctx, args as Case) as Case;
      if (Object.hasOwn(step, "return")) return step["return"];
      throw scriptedError(step["raise"] as Case, TOOL_ERRORS);
    };
    const description = get(stub, "description", "") as string;
    const isConcurrencySafe = stub["concurrency_safe"] === true ? () => true : null;
    const parameters = stub["parameters"] as JsonObject;
    tools.push([name, tool({ name, description, parameters, execute, isConcurrencySafe })]);
  }
  if (Object.hasOwn(spec, "human")) {
    const askUser: AskUserHandler = (request, ctx) =>
      call("ask_user", `host.human.${ctx.node}`, ctx, askUserArgs(request));
    tools.push(["ask_user", askUser]);
  }
  let random: (() => number) | null = null;
  if (Object.hasOwn(spec, "jitter")) {
    scripts.set("host.jitter", [...(spec["jitter"] as Value[])]);
    random = () => Number(take("host.jitter", "a backoff draw"));
  }

  const check = (): string[] => {
    if (problems.length > 0) throw new CaseFailure(problems.join("\n"));
    return [...scripts]
      .filter(([, script]) => script.length > 0)
      .map(([label, script]) => {
        const entries = script.length === 1 ? "entry" : "entries";
        return `\`${label}\` has ${script.length} ${entries} left`;
      });
  };
  return { tools: Object.fromEntries(tools), random, calls, taken, check };
}

/** Return a run's `failure` in the case's failure form. */
function failureForm(failure: Failure, flow: ComposedFlow): Case {
  return { ...diagnosticForm(flow.diagnose(failure)), node: failure.nodeId };
}

/** The events and chunk whose `attempt` is left out when it is 1, as Python leaves it null. */
const ATTEMPT_LEFT_OUT_AT_1: ReadonlySet<string> = new Set([
  "ModelCallStarted", "ModelCallFinished", "StreamChunk",
]);

/**
 * Return the data form of an event or stream chunk, without its null fields: its `toJSON()`, with
 * a failure in failure form and every file as `{uri, mime, name}`.
 */
function fields(event: RunEvent, flow: ComposedFlow): Case {
  const data: Case = {};
  for (const [key, item] of Object.entries(event.toJSON())) {
    if (item === null || (key === "attempt" && item === 1 && ATTEMPT_LEFT_OUT_AT_1.has(event.type))) continue;
    const failure = key === "failure" ? (event as { failure: Failure }).failure : null;
    data[key] = failure !== null ? failureForm(failure, flow) : dataForm(item as YianValue);
  }
  return data;
}

/** Return `event` as `[<EventName>]` or `[<EventName>, <data form>]`. */
function eventForm(event: RunEvent, flow: ComposedFlow): Value[] {
  const data = fields(event, flow);
  return Object.keys(data).length > 0 ? [event.type, data] : [event.type];
}

/**
 * Return `entries` with run-level ones (node `null`) first, then each node's, nodes sorted; each
 * group keeps its order.
 */
function grouped<T>(entries: readonly T[], nodeOf: (entry: T) => string | null): T[] {
  const key = (entry: T): string | null => nodeOf(entry);
  // `sort` is stable, so each group keeps its order.
  return [...entries].sort((a, b) => {
    const [first, second] = [key(a), key(b)];
    if (first === null || second === null) return (first === null ? 0 : 1) - (second === null ? 0 : 1);
    return compareCodeUnits(first, second);
  });
}

/** Return `results` with the events, stream and each tool's calls `grouped` by node. */
function pooled(results: Record<string, unknown>): Record<string, unknown> {
  // A `NodeFailed` belongs to its failure's node; run events have no `node_id`.
  const eventNode = (event: Value[]): string | null => {
    const data = (event[1] ?? {}) as Case;
    const node = event[0] === "NodeFailed" ? (data["failure"] as Case)["node"] : data["node_id"];
    return (node ?? null) as string | null;
  };
  const toolCalls = results["tool_calls"] as Record<string, Case[]>;
  return {
    ...results,
    events: grouped(results["events"] as Value[][], eventNode),
    stream: grouped(results["stream"] as Case[], (chunk) => (get(chunk, "node_id") ?? null) as string | null),
    tool_calls: Object.fromEntries(
      Object.entries(toolCalls).map(([name, calls]) => [
        name,
        grouped(calls, (call) => call["node"] as string),
      ]),
    ),
  };
}

/** The fields whose text values are ids an implementation makes up. */
const ID_FIELDS: ReadonlySet<string> = new Set(["exchange_id", "commit_id", "mutation_id", "note_id"]);

/**
 * Return `results` with each made-up id in their text replaced by `<id:N>`.
 *
 * The ids are `runId` (unless `null`) and every text value of an `ID_FIELDS` key, numbered in
 * order of first appearance walking `results` in their order, with mapping keys in UTF-16 order.
 */
function placeholders(results: Record<string, unknown>, runId: string | null): Record<string, unknown> {
  const ids = new Set<string>(runId === null ? [] : [runId]);
  const collect = (value: unknown): void => {
    if (Array.isArray(value)) {
      value.forEach(collect);
    } else if (isMapping(value)) {
      for (const [key, item] of Object.entries(value)) {
        if (ID_FIELDS.has(key) && typeof item === "string" && item !== "") ids.add(item);
        collect(item);
      }
    }
  };
  collect(results);
  if (ids.size === 0) return results;
  // Longest first, so an id that holds another is replaced whole.
  const longestFirst = [...ids].sort((a, b) => b.length - a.length);
  const escaped = longestFirst.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(escaped.join("|"), "g");
  const numbers = new Map<string, number>();
  const number = (id: string): number => {
    if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
    return numbers.get(id)!;
  };
  const replace = (value: unknown): unknown => {
    if (typeof value === "string") return value.replace(pattern, (id) => `<id:${number(id)}>`);
    if (Array.isArray(value)) return value.map(replace);
    if (isMapping(value)) {
      const names = Object.keys(value).sort(compareCodeUnits);
      return Object.fromEntries(names.map((name) => [name, replace(value[name])]));
    }
    return value;
  };
  return Object.fromEntries(Object.entries(results).map(([name, value]) => [name, replace(value)]));
}

/** Return `result` in the form of `expect.output`. */
function output(result: RunResult, flow: ComposedFlow): Case {
  if (result.status === "succeeded") {
    return { status: "succeeded", output: dataForm(result.output as YianValue) };
  }
  return { status: result.status, failure: failureForm(result.failure!, flow) };
}

/**
 * Run the flow with `inputs` and the `run` arguments on the case's host, logged to `run.jsonl` in
 * the case directory, its waits on a virtual clock that starts at the real current time. Return
 * its results: `output`, or the host error as `raises` and `error` (never compared), and `left`, a
 * message per script with entries left.
 */
async function actualRun(case_: Case, ctx: CaseContext): Promise<Record<string, unknown>> {
  const flow = await loadText(case_["flow"] as string, ctx);
  const log = join(ctx.dir, "run.jsonl");
  const { tools, random, calls, check } = host(case_, log);
  const run = get(case_, "run", {}) as Case;
  const events: RunEvent[] = [];
  const chunks: StreamChunk[] = [];
  // `env` scopes the case's YIAN_* values to the run, so nothing is installed around it.
  const options: RunFlowOptions = {
    ...runOptions(run),
    log,
    env: caseEnv(ctx),
    tools,
    random,
    scheduler: virtualScheduler(Date.now()),
    onEvent: (event) => {
      events.push(event);
    },
    onStream: (chunk) => {
      chunks.push(chunk);
    },
  };
  let result: RunResult | undefined;
  let caught: Record<string, unknown> = {};
  try {
    result = await runFlow(flow, get(case_, "inputs", {}) as Case, options);
  } catch (error) {
    caught = { raises: raised(error), error };
  }
  // A call past a script's end is the case's own error, whatever it made differ.
  const left = check();
  const text = existsSync(log) ? readFileSync(log, "utf8") : "";
  const lines = text.split("\n").filter((line) => line !== "");
  // In placeholder order: ids are numbered walking the results in this order.
  let actual: Record<string, unknown> = {
    events: events.map((event) => eventForm(event, flow)),
    stream: chunks.map((chunk) => fields(chunk, flow)),
    tool_calls: Object.fromEntries([...calls].map(([name, list]) => [name, list.map(dataForm)])),
    ...(result !== undefined ? { output: output(result, flow) } : {}),
  };
  const error = caught["error"];
  checkSecrets(case_, {
    ...Object.fromEntries(Object.entries(actual).map(([name, value]) => [name, spell(value as Value)])),
    "the run log": text,
    "the raised error": error === undefined ? "" : error instanceof Error ? error.message : String(error),
  });
  // Python's `!= 0`, under which 0.0 and false equal 0 too.
  const workers = get(run, "num_workers", 8n);
  if (workers !== 0n && workers !== 0 && workers !== false) actual = pooled(actual);
  const given = Object.hasOwn(run, "run_id") || lines.length === 0;
  const runId = given ? null : ((JSON.parse(lines[0]!) as Case)["run_id"] as string);
  // A raised error is compared as is, without placeholders.
  actual = placeholders(actual, runId);
  return { ...actual, ...caught, left };
}

/**
 * Throw a `CaseFailure` unless the results in `actual` are the ones `expect` holds: the raised
 * error, or the output, which compares numbers by value for a success.
 */
function checkResults(expect: Case, actual: Record<string, unknown>): void {
  if (Object.hasOwn(expect, "raises") && !Object.hasOwn(actual, "raises")) {
    const expected = str(get(expect["raises"] as Case, "class"));
    throw new CaseFailure(`expected ${expected} to be raised, but nothing was`);
  }
  if (Object.hasOwn(actual, "raises") && !Object.hasOwn(expect, "raises")) {
    throw new CaseFailure(`unexpected error\n${failureText(actual["error"])}`);
  }
  if (Object.hasOwn(expect, "raises")) compare("raises", actual["raises"], expect["raises"]);
  if (Object.hasOwn(expect, "output")) {
    const form = actual["output"] as Case;
    compare("output", form, expect["output"], { byValue: form["status"] === "succeeded" });
  }
  for (const name of ["events", "stream", "tool_calls"]) {
    // A run's events, stream and tool calls compare numbers by value.
    if (Object.hasOwn(expect, name)) compare(name, actual[name], expect[name], { byValue: true });
  }
  // Last: a leftover entry usually follows from a divergence the comparisons show.
  const left = actual["left"] as string[];
  if (left.length > 0) throw new CaseFailure(left.join("\n"));
}

/** Run a `run` case: run the flow and compare its results with `expect`. */
async function runRun(case_: Case, ctx: CaseContext): Promise<void> {
  checkResults(case_["expect"] as Case, await actualRun(case_, ctx));
}

// ---- resume ----

// ---- backward ----

// ---- expr ----

/** Return `value` in data form: every `FileRef` in it as its mapping `{uri, mime, name}`. */
function dataForm(value: YianValue): Value {
  if (FileRef.isFileRef(value)) return { uri: value.uri, mime: value.mime, name: value.name };
  if (Array.isArray(value)) return value.map(dataForm);
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, dataForm(item)]));
  }
  return value;
}

/**
 * Run an `expr` case: evaluate `expr` or `template` in `mode` over `values` and `item`, with the
 * type table the loader would give its `isinstance` and constructor names, under the defaults
 * `env` sets. A strict `template` renders as a prompt.
 */
async function runExpr(case_: Case, ctx: CaseContext): Promise<void> {
  const mode = get(case_, "mode", "binding");
  const template = get(case_, "template", null) as string | null;
  const text = case_["expr"] as string;
  const values = get(case_, "values", {}) as ValueRecord;
  const item = get(case_, "item", null) as YianValue;
  await results(case_["expect"] as Case, () => {
    // Read inside the result, so a bad variable is raised as the case's error; forgotten
    // after, so a later case reads its own.
    try {
      reload(caseEnv(ctx));
      if (template !== null && mode === "strict") return { value: renderTemplateRecord(template, values) };
      const trees: ExprNode[] =
        template !== null
          ? expressionTrees(template)
          : mode === "condition"
            ? expressionTrees(text, { condition: true })
            : [parseExpr(text)];
      const registry = readTypedefs((case_["typedefs"] ?? {}) as Record<string, unknown>);
      const types = new Map<string, Type>();
      for (const tree of trees) {
        for (const name of isinstanceTypeNames(tree)) types.set(name, resolveIsinstanceType(name, registry));
        for (const name of constructorNames(tree)) types.set(name, resolveConstructorType(name, registry));
      }
      const resolve = (path: string) => resolveInRecord(path, values);
      if (template !== null) return { value: dataForm(evalBinding(template, resolve, { item, types })) };
      if (mode === "condition") {
        // What every `when:`, `while:` and assert runs.
        return { value: evaluateWhenRecord(text, values, types) };
      }
      const resolveMode = mode === "strict" ? ResolveMode.STRICT_RAISE : ResolveMode.BINDING_NONE;
      return { value: dataForm(evalExpr(trees[0]!, resolve, { item, mode: resolveMode, types })) };
    } finally {
      forget();
    }
  });
}

// ---- value ----

/**
 * Run a `value` case: build `value` as `type` under the defaults `env` sets and compare the
 * stored value and its kind.
 */
async function runValue(case_: Case, ctx: CaseContext): Promise<void> {
  await results(case_["expect"] as Case, () => {
    // Read inside the result, so a bad variable is raised as the case's error; forgotten
    // after, so a later case reads its own.
    try {
      reload(caseEnv(ctx));
      const typedefs = (case_["typedefs"] ?? {}) as Record<string, unknown>;
      const typ = typeFor(case_["type"] as string, readTypedefs(typedefs));
      const typed = buildValueAs(typ, case_["value"]);
      // A file in the value is compared in data form.
      return { value: typed.toWire().value, kind: typed.kind };
    } finally {
      forget();
    }
  });
}

// ---- defaults ----

/** Run a `defaults` case: read yian's defaults from `env`; compare the entries `expect` names. */
async function runDefaults(case_: Case, ctx: CaseContext): Promise<void> {
  const expect = case_["expect"] as Case;
  await results(expect, () => {
    const byName: Record<string, bigint> = {
      ...Object.fromEntries(ENTRIES.map((entry) => [entry.name, BigInt(entry.default)])),
      ...nonDefaults(read(caseEnv(ctx))),
    };
    const names = Object.keys(get(expect, "defaults", {}) as Case);
    const unknown = names.filter((name) => !Object.hasOwn(byName, name)).sort();
    if (unknown.length > 0) {
      throw new CaseFailure(`\`expect.defaults\` names unknown entries: ${unknown.join(", ")}`);
    }
    return { defaults: Object.fromEntries(names.map((name) => [name, byName[name]])) };
  });
}

// ---- fingerprint ----

/**
 * Run a `fingerprint` case: check that `flow` fingerprints the same twice; the result is whether
 * `other`'s is the `same` or `different`.
 */
async function runFingerprint(case_: Case, ctx: CaseContext): Promise<void> {
  const fingerprint = async (text: string) => flowFingerprint((await loadText(text, ctx)).compiled);
  await results(case_["expect"] as Case, async () => {
    const first = await fingerprint(case_["flow"] as string);
    if ((await fingerprint(case_["flow"] as string)) !== first) {
      throw new CaseFailure("two loads of `flow` have different fingerprints");
    }
    const matches = (await fingerprint(case_["other"] as string)) === first;
    return { fingerprint: matches ? "same" : "different" };
  });
}

// ---- cli ----

/**
 * The `yian` entry: `../src/cli/index` beside this file, `.ts` from source and `.js` when
 * built.
 */
const CLI_ENTRY = fileURLToPath(
  new URL(`../src/cli/index.${import.meta.url.endsWith(".ts") ? "ts" : "js"}`, import.meta.url),
);
// Read from the repository, not beside this file, so the built runner finds it too.
const CLI_SCHEMA = JSON.parse(
  readFileSync(join(findRoot(), "conformance", "schema", "cli.schema.json"), "utf8"),
) as Schema;
const FILLED = /sk-conformance-[0-9]+/g;

/**
 * Return `value`, a case's plain JSON, as JSON text with Python's spacing, floats with a
 * fraction.
 */
function jsonText(value: Value): string {
  if (typeof value === "number") {
    const text = String(value);
    return /^-?[0-9]+$/.test(text) ? `${text}.0` : text;
  }
  if (typeof value === "bigint") return value.toString();
  if (Array.isArray(value)) return `[${value.map(jsonText).join(", ")}]`;
  if (isMapping(value)) {
    const pairs = Object.entries(value).map(
      ([key, item]) => `${JSON.stringify(key)}: ${jsonText(item)}`,
    );
    return `{${pairs.join(", ")}}`;
  }
  return JSON.stringify(value);
}

/**
 * Return what differs between a request and the `request` of an `http` exchange, `expected`, or
 * `undefined` when it matches.
 */
function mismatch(
  expected: Case,
  method: string,
  url: string,
  headers: IncomingHttpHeaders,
  body: Buffer,
): string | undefined {
  if (method !== expected["method"] || url !== expected["url"]) {
    return `expected ${str(expected["method"])} ${str(expected["url"])}`;
  }
  for (const [name, value] of Object.entries(get(expected, "headers", {}) as Case)) {
    // Node gives header names in lower case.
    const got = headers[name.toLowerCase()];
    if (got !== value) return `header ${name}: expected ${spell(value)}, got ${spell(got ?? null)}`;
  }
  const key = ["json", "form"].find((name) => Object.hasOwn(expected, name));
  if (key === undefined) return undefined;
  const text = body.toString("utf8");
  try {
    const got =
      key === "json" ? parseJson(text, "plain") : Object.fromEntries(new URLSearchParams(text));
    if (same(got, expected[key])) return undefined;
  } catch {
    // A body that doesn't read is a mismatch.
  }
  const sent = JSON.stringify(text.slice(0, 500));
  return `body: expected ${key} ${jsonText(expected[key]!)}, got ${sent}`;
}

/**
 * A stub HTTP server on 127.0.0.1 for the exchanges `http`, answered in order. Each request names
 * the URL it was meant for in an `x-conformance-url` header. One that doesn't match the next
 * exchange, or an error in a handler, is recorded as a problem.
 */
class Stub {
  readonly url: string;
  readonly #http: Case[];
  readonly #server: Server;
  #taken = 0;
  readonly #problems: string[] = [];
  readonly #open = new Set<Promise<void>>();
  // Each ends a `delay` early.
  readonly #wakers = new Set<() => void>();
  #stopped = false;

  private constructor(http: Case[], server: Server) {
    this.#http = http;
    this.#server = server;
    this.url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
  }

  /** Start serving `http`. */
  static async start(http: Case[]): Promise<Stub> {
    // A connection the command left half-open can't keep `close` waiting for its handler.
    const server = createServer({ requestTimeout: 10_000, connectionsCheckingInterval: 1_000 });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const stub = new Stub(http, server);
    server.on("request", (request: IncomingMessage, response: ServerResponse) => {
      const handled = stub.#answer(request, response).finally(() => stub.#open.delete(handled));
      stub.#open.add(handled);
    });
    return stub;
  }

  /** Check the request against the next exchange, then answer it. */
  async #answer(request: IncomingMessage, response: ServerResponse): Promise<void> {
    try {
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(chunk as Buffer);
      const header = request.headers["x-conformance-url"];
      const url = typeof header === "string" ? header : (request.url ?? "");
      const method = request.method ?? "";
      let index = this.#taken;
      if (index === this.#http.length && this.#http.at(-1)?.["repeat"] === true) index -= 1;
      let answer: Case = { status: 500n };
      const exchange = this.#http[index];
      if (exchange === undefined) {
        this.#problems.push(`${method} ${url}: \`http\` has no exchange left for it`);
      } else {
        const expected = exchange["request"] as Case;
        const difference = mismatch(expected, method, url, request.headers, Buffer.concat(chunks));
        if (difference === undefined) {
          this.#taken = index + 1;
          answer = exchange["response"] as Case;
        } else {
          this.#problems.push(`${method} ${url} doesn't match \`http[${index}]\`: ${difference}`);
        }
      }
      await this.#delay(Number(get(answer, "delay", 0n)) * 1000);
      const hasJson = Object.hasOwn(answer, "json");
      const data = hasJson ? jsonText(answer["json"]!) : "";
      response.writeHead(Number(answer["status"]), {
        ...(hasJson ? { "content-type": "application/json" } : {}),
        "content-length": Buffer.byteLength(data),
      });
      response.end(data);
    } catch (error) {
      // A connection the command dropped is not a problem of the case.
      const code = (error as { code?: unknown }).code;
      if (code !== "ECONNRESET" && code !== "EPIPE") {
        const problem = error instanceof Error ? (error.stack ?? error.message) : String(error);
        this.#problems.push(problem);
      }
    }
  }

  /** Wait `ms` milliseconds, or less once `close` is called. */
  #delay(ms: number): Promise<void> {
    if (this.#stopped || ms <= 0) return Promise.resolve();
    return new Promise((resolve) => {
      const wake = () => {
        clearTimeout(timer);
        this.#wakers.delete(wake);
        resolve();
      };
      const timer = setTimeout(wake, ms);
      this.#wakers.add(wake);
    });
  }

  /**
   * Stop serving once every handler finishes; return the problems, then each exchange without
   * `repeat` that was never requested, with filled secrets written `<key:N>`.
   */
  async close(): Promise<string[]> {
    this.#stopped = true;
    // Usually lets a request that arrived just before the command exited reach its handler.
    await new Promise((resolve) => setImmediate(resolve));
    for (const wake of [...this.#wakers]) wake();
    while (this.#open.size > 0) await Promise.all(this.#open);
    await new Promise((resolve) => {
      this.#server.close(resolve);
      this.#server.closeAllConnections();
    });
    const unrequested = this.#http
      .map((exchange, index) => [exchange, index] as const)
      .filter(([exchange, index]) => index >= this.#taken && exchange["repeat"] !== true)
      .map(([, index]) => `\`http[${index}]\` was never requested`);
    return [...this.#problems, ...unrequested].map((problem) =>
      problem.replace(FILLED, (filled) => `<key:${filled.slice("sk-conformance-".length)}>`),
    );
  }
}

/**
 * Return the ES module the command imports first: it sends every `fetch` to the stub at `stub`,
 * with the URL it was meant for in an `x-conformance-url` header.
 */
function routeHook(stub: string): string {
  // `safeFetch` reads `globalThis.fetch` per call; the stub's URL is written in, not passed in
  // the environment, which stays exactly the case's.
  return `const send = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const request = new Request(input, init);
  const headers = new Headers(request.headers);
  headers.set("x-conformance-url", request.url);
  const body = ["GET", "HEAD"].includes(request.method) ? undefined : await request.arrayBuffer();
  return send(${JSON.stringify(stub)}, {
    method: request.method, headers, body, signal: request.signal, redirect: request.redirect,
  });
};
`;
}

/** What a command run gave: its exit code or the signal that ended it, and its output. */
interface Ran {
  code: number | null;
  signal: NodeJS.Signals | null;
  out: string;
  err: string;
}

/**
 * Run this runtime with `args` in `cwd`, `stdin` written to it; with `signal`, send it the
 * signal `signal.name` once `signal.after` is in its stdout or stderr. Fails the case when it
 * exits before `after` appears, `after` doesn't appear within 60 seconds, or it doesn't exit
 * within 120 seconds.
 */
async function runCommand(
  args: string[],
  cwd: string,
  env: Record<string, string>,
  stdin: string,
  signal: Case | undefined,
): Promise<Ran> {
  // The child's own signal listeners replace any disposition it inherits, so yian's handlers
  // work even when this process was started with SIGINT ignored.
  const child = spawn(process.execPath, args, { cwd, env, stdio: "pipe" });
  const ran: Ran = { code: null, signal: null, out: "", err: "" };
  child.stdout.setEncoding("utf8").on("data", (text: string) => {
    ran.out += text;
  });
  child.stderr.setEncoding("utf8").on("data", (text: string) => {
    ran.err += text;
  });
  let finished = false;
  let failed: Error | undefined;
  // Node still emits `close` after a spawn error.
  child.on("error", (error) => {
    failed = error;
  });
  const closed = new Promise<void>((resolve) => {
    child.on("close", (code, killedBy) => {
      [ran.code, ran.signal, finished] = [code, killedBy, true];
      resolve();
    });
  });
  const exited = new Promise<void>((resolve) => {
    child.on("exit", () => resolve());
    child.on("error", () => resolve());
  });
  const check = () => {
    if (failed !== undefined) throw new CaseFailure(`the command failed to run: ${failed.message}`);
  };
  // A command can exit, or stop reading, before taking all of its stdin.
  child.stdin.on("error", () => {});
  child.stdin.end(stdin);
  const streams = () => `--- stdout\n${ran.out}--- stderr\n${ran.err}`;
  try {
    let after = "";
    if (signal !== undefined) {
      const name = signal["name"] as NodeJS.Signals;
      const wanted = signal["after"] as string;
      after = ` after ${name}`;
      const deadline = Date.now() + 60_000;
      while (!ran.out.includes(wanted) && !ran.err.includes(wanted)) {
        check();
        if (finished) {
          const before = `the command exited before ${JSON.stringify(wanted)} appeared`;
          throw new CaseFailure(`${before}\n${streams()}`);
        }
        if (Date.now() > deadline) {
          throw new CaseFailure(`${JSON.stringify(wanted)} didn't appear within 60 seconds`);
        }
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      child.kill(name);
    }
    const limit = new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 120_000);
      void closed.then(() => clearTimeout(timer));
    });
    await Promise.race([closed, limit]);
    check();
    if (!finished) {
      throw new CaseFailure(`the command didn't exit within 120 seconds${after}\n${streams()}`);
    }
    return ran;
  } finally {
    // Ends a command the case gave up on, and waits for it, as Python's `process.wait()`, so the
    // stub and the hook outlive it.
    if (!finished) {
      child.kill("SIGKILL");
      await exited;
    }
  }
}

/** Throw a `CaseFailure` when a secret the case filled appears in a text of `texts`, by where. */
function checkSecrets(case_: Case, texts: Record<string, string>): void {
  const filled = [...SECRET_KEYS].map((key) => get(case_, key) ?? null);
  const secrets = new Set(spell(filled).match(FILLED) ?? []);
  const leaks = Object.entries(texts).flatMap(([where, text]) =>
    [...new Set(text.match(FILLED) ?? [])]
      .filter((secret) => secrets.has(secret))
      .sort()
      .map((secret) => `secret ${secret} appears in ${where}`),
  );
  if (leaks.length > 0) throw new CaseFailure(leaks.join("\n"));
}

/**
 * Return the document `text`, a JSON text the command wrote, in codec form, after checking it as
 * parsed against `cli.schema.json`; throw a `CaseFailure` showing `stdout` when it fails.
 */
function cliDocument(text: string, stdout: string): Value {
  let parsed: unknown;
  try {
    parsed = parseJson(text, "plain");
  } catch (error) {
    const reason = (error as Error).message;
    throw new CaseFailure(`stdout is not a JSON document (${reason})\n--- stdout\n${stdout}`);
  }
  // Checked as parsed: the schema describes the JSON, not the codec's values.
  if (validate(CLI_SCHEMA, CLI_SCHEMA, parsed).length > 0) {
    // The root is an `anyOf` of the documents, so each one's problems are listed.
    const reasons = (CLI_SCHEMA["anyOf"] as Schema[])
      .flatMap((branch) => validate(CLI_SCHEMA, branch, parsed))
      .map((problem) => `  ${problem}\n`)
      .join("");
    throw new CaseFailure(
      `stdout doesn't match conformance/schema/cli.schema.json\n${reasons}--- stdout\n${stdout}`,
    );
  }
  return parseJson(text, "codec") as Value;
}

/**
 * Whether `path` exists, a dangling symlink included; false on any system error, as Python's
 * `os.path.lexists`.
 */
function lexists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === undefined) throw error;
    return false;
  }
}

/**
 * Run a `cli` case: run the command `entry` with `args` in the case directory, its `fetch` routed
 * to a stub serving `http`, and compare its exit code, output, documents and the paths that exist.
 */
export async function runCli(
  case_: Case,
  ctx: CaseContext,
  { entry = CLI_ENTRY }: { entry?: string } = {},
): Promise<void> {
  const home = join(ctx.dir, "home");
  mkdirSync(home);
  const env = { PATH: process.env["PATH"] ?? "", HOME: home, COLUMNS: "1000", ...ctx.env };
  // Outside the case directory, so it is neither in `exists` nor scanned for secrets.
  const hookDir = mkdtempSync(join(tmpdir(), "yian-conformance-hook-"));
  let stub: Stub | undefined;
  let ran: Ran;
  let problems: string[] = [];
  try {
    stub = await Stub.start(get(case_, "http", []) as Case[]);
    const hook = join(hookDir, "route.mjs");
    writeFileSync(hook, routeHook(stub.url));
    // Bun reads `--import` as `--preload`; `--no-install` keeps it from installing packages, and
    // `--no-env-file` from loading a `.env` in the case directory into the exact environment.
    const runtime = process.versions.bun !== undefined ? ["--no-install", "--no-env-file"] : [];
    const args = [...runtime, "--import", hook, entry, ...(get(case_, "args", []) as string[])];
    const stdin = get(case_, "stdin", "") as string;
    ran = await runCommand(args, ctx.dir, env, stdin, get(case_, "signal") as Case | undefined);
  } finally {
    try {
      if (stub !== undefined) problems = await stub.close();
    } finally {
      rmSync(hookDir, { recursive: true, force: true });
    }
  }
  // Longest first: the real path can hold the given one (`/private/var` on macOS).
  const paths = [...new Set([ctx.dir, realpathSync(ctx.dir)])].sort((a, b) => b.length - a.length);
  const normal = (text: string) =>
    paths
      .reduce((result, path) => result.replaceAll(path, "<tmp>"), text)
      .replace(/ +$/gm, "");
  const stdout = normal(ran.out);
  const stderr = normal(ran.err);
  const streams = `--- stdout\n${stdout}--- stderr\n${stderr}`;
  if (problems.length > 0) throw new CaseFailure(`${problems.join("\n")}\n${streams}`);
  const logs = (readdirSync(ctx.dir, { recursive: true, encoding: "utf8" }) as string[])
    .filter((name) => name.endsWith(".jsonl"))
    .map((name) => [name, readFileSync(join(ctx.dir, name), "utf8")] as const);
  checkSecrets(case_, { stdout, stderr, ...Object.fromEntries(logs) });
  const expect = case_["expect"] as Case;
  if (ran.signal !== null) {
    throw new CaseFailure(`died from signal ${constants.signals[ran.signal]}\n${streams}`);
  }
  if (!same(BigInt(ran.code ?? 0), expect["exit"])) {
    throw new CaseFailure(`exit ${ran.code}, expected ${spell(expect["exit"])}\n${streams}`);
  }
  for (const [name, text] of [["stdout", stdout], ["stderr", stderr]] as const) {
    if (Object.hasOwn(expect, name) && text !== expect[name]) {
      const expected = str(expect[name]);
      throw new CaseFailure(`${name} differs\n--- actual\n${text}--- expected\n${expected}`);
    }
  }
  if (Object.hasOwn(expect, "json")) {
    compare("json", cliDocument(stdout, stdout), expect["json"], { optionalMessage: true });
  }
  if (Object.hasOwn(expect, "json_lines")) {
    const lines = stdout.split("\n").filter((line) => line.trim() !== "");
    const documents = lines.map((line) => cliDocument(line, stdout));
    compare("json_lines", documents, expect["json_lines"], { optionalMessage: true });
  }
  if (Object.hasOwn(expect, "exists")) {
    const exists = Object.fromEntries(
      Object.keys(expect["exists"] as Case).map((path) => [
        path,
        lexists(join(ctx.dir, path)),
      ]),
    );
    compare("exists", exists, expect["exists"]);
  }
}

// ---- json ----

/** Run a `json` case: read `read` in its `mode`, or write `write` in its `form`. */
async function runJson(case_: Case): Promise<void> {
  const expect = case_["expect"] as Case;
  if (Object.hasOwn(case_, "read")) {
    const text = case_["read"] as string;
    await results(expect, () => ({ value: parseJson(text, case_["mode"] as JsonMode) }));
    return;
  }
  const value = case_["write"] as Value;
  await results(expect, () => {
    if (case_["form"] !== "log") return { text: canonicalJson(value) };
    const text = logJson(value);
    // The log form is lossless: reading the line back gives the value, kind for kind.
    compare("read back", parseJson(text, "codec"), value);
    return { text };
  });
}

/** Return whether this file is the process's entry script. */
function isEntry(): boolean {
  // Node 22 has no `import.meta.main`; under `bun test`, `argv[1]` is the test file.
  const entry = process.argv[1];
  if (entry === undefined) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    // `argv[1]` names no file.
    return false;
  }
}

if (isEntry()) {
  // Not `process.exit()`, which can cut off piped stdout on Node.
  process.exitCode = await main(process.argv.slice(2));
}
