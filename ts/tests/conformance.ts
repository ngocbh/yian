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
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { Document, isCollection, isMap, isNode, isScalar, isSeq, parseDocument, visit } from "yaml";
import type { Node as YamlNode, YAMLMap } from "yaml";

import {
  expressionTrees,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../src/compose/helpers/isinstance_types.ts";
import { flowFingerprint } from "../src/compile/manifest.ts";
import { END_ID } from "../src/compile/model.ts";
import type { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { LoadError } from "../src/compose/helpers/errors.ts";
import { loadFlow } from "../src/compose/loader.ts";
import { ENTRIES, forget, nonDefaults, read, reload, withEnv } from "../src/defaults.ts";
import type { Env } from "../src/defaults.ts";
import {
  compareCodeUnits,
  type Diagnostic,
  Failure,
  type FailureCode,
  OriginRef,
  reprText,
} from "../src/diagnostics.ts";
import { type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "../src/events.ts";
import type { Steering } from "../src/nodes/base.ts";
import { ProviderError, kindForStatus } from "../src/llm/errors.ts";
import {
  type ChatModel,
  type Event,
  EventStream,
  Finish,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../src/llm/events.ts";
import {
  Message,
  MessageFormatError,
  type Request,
  Response,
  TextPart,
  ToolCallPart,
  Usage,
} from "../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../src/llm/testing.ts";
import { checkModelConfig, resolveModelConfig } from "../src/llm_clients/connection.ts";
import type { ModelAccess } from "../src/llm_clients/factory.ts";
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
import { buildMemories, type MemoryRegistry } from "../src/memory_host.ts";
import { virtualScheduler } from "../src/platform.ts";
import {
  inspectRun,
  replayRunLog,
  type ResumeFlowOptions,
  resumeFlow,
  type RunInspection,
  type RunFlowOptions,
  type RunResult,
  runFlow,
} from "../src/run.ts";
import { JsonlRunLog, MemoryRunLog, readRunLog, type RunStartedRecord } from "../src/runtime/log.ts";
import { addInterruptedAttempts } from "../src/runtime/state.ts";
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
import { buildValueAs, FileRef, fromHost, isRecord } from "../src/typesys/values.ts";
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
const RESERVED_HOST_KEYS: readonly string[] = ["skills"];
const HOST_KEYS: ReadonlySet<string> = new Set([
  "tools", "human", "jitter", "models", "memory", "steering", "stop", ...RESERVED_HOST_KEYS,
]);
const RESUME_HOST_KEYS: ReadonlySet<string> = new Set([...HOST_KEYS, "crash_after_record"]);
/** The `[section, key]` pairs a case may not use yet: it fails before it is set up. */
const RESERVED: readonly (readonly [string, string])[] = RESERVED_HOST_KEYS.map(
  (key) => ["host", key] as const,
);
// `host.models` consumers, besides `memory:<key>` under `LEARNER_NODE`.
const CONSUMERS: readonly string[] = ["agent", "structured_retry", "compose", "backward", "summary"];
// The node id memory learners' model calls are scripted under.
const LEARNER_NODE = "__memory__";
const USAGE_COUNTS: readonly string[] = ["input_tokens", "output_tokens", "total_tokens"];
const USAGE_DETAILS: readonly string[] = [
  "cache_read_tokens", "cache_creation_tokens", "reasoning_tokens",
];
/** The events and chunk whose `attempt` is left out when it is 1, as Python leaves it null. */
const ATTEMPT_LEFT_OUT_AT_1: ReadonlySet<string> = new Set([
  "ModelCallStarted", "ModelCallFinished", "StreamChunk",
]);
const RULES: ReadonlySet<string> = new Set([
  "numbers", "int-in-float", "arithmetic", "compare", "text-form", "canonical", "columns",
  "messages", "text", "yaml", "dates", "infinity", "llm", "code", "failure-codes", "run-log",
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

/** Return whether `value` is 0 in Python's `== 0` sense, under which 0.0 and false equal 0 too. */
function isZero(value: Value | undefined): boolean {
  return value === 0n || value === 0 || value === false;
}

/** How a case file is parsed: YAML 1.2 core, ints as `bigint`, duplicate keys left to `checkKeys`. */
const CASE_YAML = { version: "1.2", schema: "core", intAsBigInt: true, uniqueKeys: false } as const;

/**
 * Return the case file `text` as YAML 1.2 core, untagged, with only text keys, each once.
 */
export function readCase(text: string): Case {
  const doc = parseDocument(text, CASE_YAML);
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

/** Throw unless `usage` is a usage: the three counts and any optional ones, as ints. */
function checkUsage(where: string, usage: Value | undefined): void {
  const counts = keys(where, usage, new Set([...USAGE_COUNTS, ...USAGE_DETAILS]), {
    required: USAGE_COUNTS,
  });
  if (Object.values(counts).some((count) => typeof count !== "bigint" || count < 0n)) {
    throw new Error(`\`${where}\` counts must be non-negative ints`);
  }
}

/** Return `data` read as a yian.message.v1 message, throwing unless it is one. */
function readMessage(where: string, data: Value | undefined): Message {
  try {
    return Message.fromJSON(data);
  } catch (error) {
    if (!(error instanceof MessageFormatError)) throw error;
    throw new Error(`\`${where}\` is not a yian.message.v1 message: ${error.message}`);
  }
}

/**
 * Throw unless `reply` is a yian.message.v1 assistant message this runner can serve. This runner
 * serves only text and tool call parts.
 */
function checkReply(where: string, reply: Value | undefined): void {
  const message = readMessage(where, reply);
  if (message.role !== "assistant") throw new Error(`\`${where}.role\` must be \`assistant\``);
  message.parts.forEach((part, index) => {
    if (!(part instanceof TextPart || part instanceof ToolCallPart)) {
      throw new Error(
        `\`${where}.parts[${index}]\`: this runner doesn't script \`${part.type}\` parts yet`,
      );
    }
  });
}

/** Throw unless `chunk` is one text, tool call, usage or finish chunk; a finish comes last. */
function checkChunk(where: string, chunk: Value | undefined, last: boolean): void {
  const kinds = new Set(["text", "tool_call", "usage", "finish"]);
  const entries = Object.entries(keys(where, chunk, kinds));
  if (entries.length !== 1) {
    throw new Error(`\`${where}\` takes one of \`text\`, \`tool_call\`, \`usage\` and \`finish\``);
  }
  const [kind, value] = entries[0]!;
  if (kind === "text" && typeof value !== "string") throw new Error(`\`${where}.text\` must be text`);
  if (kind === "tool_call") {
    const call = keys(`${where}.tool_call`, value, new Set(["index", "id", "name", "arguments"]), {
      required: ["index", "arguments"],
    });
    if (typeof call["index"] !== "bigint") {
      throw new Error(`\`${where}.tool_call.index\` must be an int`);
    }
    if (["id", "name", "arguments"].some((key) => typeof get(call, key, "") !== "string")) {
      throw new Error(`\`${where}.tool_call\` \`id\`, \`name\` and \`arguments\` must be text`);
    }
  }
  if (kind === "usage") checkUsage(`${where}.usage`, value);
  if (kind === "finish" && !(value === null || typeof value === "string")) {
    throw new Error(`\`${where}.finish\` must be text or null`);
  }
  if (kind === "finish" && !last) throw new Error(`\`${where}\`: \`finish\` must be the last chunk`);
}

/**
 * Throw unless `request` is `{messages, tools, schema}` with `messages` required: yian.message.v1
 * messages, a non-empty list of tool specs and a mapping.
 */
function checkRequest(where: string, request: Value | undefined): void {
  const map = keys(where, request, new Set(["messages", "tools", "schema"]), {
    required: ["messages"],
  });
  for (const key of ["messages", "tools"]) {
    if (!Array.isArray(get(map, key, []))) throw new Error(`\`${where}.${key}\` must be a list`);
  }
  const tools = get(map, "tools", []) as Value[];
  if (Object.hasOwn(map, "tools") && tools.length === 0) {
    throw new Error(`\`${where}.tools\` is empty; omit \`tools\` when none are bound`);
  }
  (map["messages"] as Value[]).forEach((message, index) => {
    readMessage(`${where}.messages[${index}]`, message);
  });
  const spec = ["name", "description", "parameters"];
  tools.forEach((item, index) => {
    keys(`${where}.tools[${index}]`, item, new Set(spec), { required: spec });
  });
  mapping(`${where}.schema`, get(map, "schema", {}));
}

/**
 * Throw unless `entry` is a `reply` or `chunks` model entry. A `reply` may add a `usage`; `chunks`
 * may add a `raise`, and without one end in a `finish`. Either may add a `request`.
 */
function checkModelEntry(where: string, value: Value | undefined): void {
  const entry = keys(where, value, new Set(["reply", "usage", "chunks", "raise", "request"]));
  if (Object.hasOwn(entry, "reply") === Object.hasOwn(entry, "chunks")) {
    throw new Error(`\`${where}\` takes \`reply\` or \`chunks\``);
  }
  if (Object.hasOwn(entry, "request")) checkRequest(`${where}.request`, entry["request"]);
  if (Object.hasOwn(entry, "reply")) {
    if (Object.hasOwn(entry, "raise")) {
      throw new Error(`\`${where}.raise\` must come with \`chunks\`, not a \`reply\``);
    }
    checkReply(`${where}.reply`, entry["reply"]);
    if (Object.hasOwn(entry, "usage")) checkUsage(`${where}.usage`, entry["usage"]);
    return;
  }
  if (Object.hasOwn(entry, "usage")) {
    throw new Error(
      `\`${where}.usage\` must come with a \`reply\`; \`chunks\` carry a usage chunk`,
    );
  }
  const chunks = entry["chunks"];
  if (!Array.isArray(chunks)) throw new Error(`\`${where}.chunks\` must be a list`);
  chunks.forEach((chunk, index) => {
    checkChunk(`${where}.chunks[${index}]`, chunk, index === chunks.length - 1);
  });
  if (Object.hasOwn(entry, "raise")) {
    const error = keys(`${where}.raise`, entry["raise"], new Set(["class", "message", "status"]), {
      required: ["class", "message"],
    });
    // The replay builds the error from these.
    if (typeof error["class"] !== "string" || typeof error["message"] !== "string") {
      throw new Error(`\`${where}.raise.class\` and \`.message\` must be text`);
    }
    if (typeof get(error, "status", 0n) !== "bigint") {
      throw new Error(`\`${where}.raise.status\` must be an int`);
    }
  } else if (chunks.length === 0 || !Object.hasOwn(chunks.at(-1) as Case, "finish")) {
    throw new Error(
      `\`${where}.chunks\` must end in a \`finish\` chunk, unless a \`raise\` follows them`,
    );
  }
}

/** Throw unless `models` is `host.models`: node ids to consumers to lists of entries. */
function checkModels(models: Value | undefined): void {
  for (const [node, consumers] of Object.entries(mapping("host.models", models))) {
    for (const [consumer, script] of Object.entries(mapping(`host.models.${node}`, consumers))) {
      const label = `host.models.${node}.${consumer}`;
      if (consumer.startsWith("memory:") !== (node === LEARNER_NODE)) {
        throw new Error(
          `\`${label}\`: \`memory:<key>\` consumers go under \`${LEARNER_NODE}\`, and ` +
            `\`${LEARNER_NODE}\` holds only them`,
        );
      }
      if (node !== LEARNER_NODE && !CONSUMERS.includes(consumer)) {
        throw new Error(`\`${label}\`: the consumer must be one of ${CONSUMERS.join(", ")}`);
      }
      if (!Array.isArray(script)) throw new Error(`\`${label}\` must be a list`);
      script.forEach((entry, index) => {
        checkModelEntry(`${label}[${index}]`, entry);
      });
    }
  }
}

/**
 * Throw unless `steering` is `host.steering`: node ids to lists of actions. An action is
 * `{after, steer: [text, ...]}` or `{after, interrupt: true}`, `after` optional and one of
 * `{model_call: n, chunk: k}`, `{model_call: n}` and `{tool_call: i}`, counted from 1.
 */
function checkSteering(steering: Value | undefined): void {
  for (const [node, actions] of Object.entries(mapping("host.steering", steering))) {
    if (!Array.isArray(actions)) {
      throw new Error("`host.steering` maps node ids to lists of actions");
    }
    actions.forEach((item, index) => {
      const where = `host.steering.${node}[${index}]`;
      const action = keys(where, item, new Set(["after", "steer", "interrupt"]));
      if (Object.hasOwn(action, "steer") === Object.hasOwn(action, "interrupt")) {
        throw new Error(`\`${where}\` takes \`steer\` or \`interrupt\``);
      }
      const steer = action["steer"];
      if (
        Object.hasOwn(action, "steer") &&
        (!Array.isArray(steer) ||
        steer.length === 0 ||
        steer.some((text) => typeof text !== "string"))
      ) {
        throw new Error(`\`${where}.steer\` must be a non-empty list of text`);
      }
      if (Object.hasOwn(action, "interrupt") && action["interrupt"] !== true) {
        throw new Error(`\`${where}.interrupt\` must be \`true\``);
      }
      if (Object.hasOwn(action, "after")) checkPoint(`${where}.after`, action["after"]);
    });
  }
}

/** Throw unless `value` is run or resume arguments with only `allowed` keys and the `required`. */
function checkArguments(
  where: string,
  value: Value | undefined,
  allowed: ReadonlySet<string>,
  required: readonly string[] = [],
): void {
  const args = keys(where, value, allowed, { required });
  if (get(args, "context", "present") !== "present") {
    throw new Error(`\`${where}.context\` must be \`present\``);
  }
}

/** Throw unless `value` is a non-empty list of `{feedback, api_key, headers}`. */
function checkBackward(where: string, value: Value | undefined): void {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`\`${where}\` must be a non-empty list`);
  }
  value.forEach((entry, index) => {
    keys(`${where}[${index}]`, entry, new Set(["feedback", "api_key", "headers"]), {
      required: ["feedback"],
    });
  });
}

/**
 * Throw unless `value` is `host.memory`: `{before: [{inputs, run, backward}, ...]}`. Each earlier
 * run gives `run.run_id`, and one with `backward` is learnable.
 */
function checkMemory(value: Value | undefined): void {
  const memory = keys("host.memory", value, new Set(["before"]));
  const before = get(memory, "before", []);
  if (!Array.isArray(before)) throw new Error("`host.memory.before` must be a list");
  before.forEach((item, index) => {
    const where = `host.memory.before[${index}]`;
    const entry = keys(where, item, new Set(["inputs", "run", "backward"]), { required: ["run"] });
    mapping(`${where}.inputs`, get(entry, "inputs", {}));
    checkArguments(`${where}.run`, entry["run"], RUN_KEYS, ["run_id"]);
    if (Object.hasOwn(entry, "backward")) {
      checkBackward(`${where}.backward`, entry["backward"]);
      if (get(entry["run"] as Case, "learnable") !== true) {
        throw new Error(`\`${where}.backward\` needs \`${where}.run.learnable: true\``);
      }
    }
  });
}

/**
 * Throw unless `value`, at `where`, is a point: `{model_call: n, chunk: k}`, `{model_call: n}` or
 * `{tool_call: i}`, counted from 1.
 */
function checkPoint(where: string, value: Value | undefined): void {
  const point = mapping(where, value);
  const names = Object.keys(point).sort().join(",");
  if (!["chunk,model_call", "model_call", "tool_call"].includes(names)) {
    throw new Error(`\`${where}\` is \`{model_call, chunk}\`, \`{model_call}\` or \`{tool_call}\``);
  }
  if (Object.values(point).some((count) => typeof count !== "bigint" || count < 1n)) {
    throw new Error(`\`${where}\` counts must be ints from 1`);
  }
}

/** Throw unless `stop` is `host.stop`: node ids to `{after: <point>}`. */
function checkStop(stop: Value | undefined): void {
  for (const [node, item] of Object.entries(mapping("host.stop", stop))) {
    const where = `host.stop.${node}`;
    const action = keys(where, item, new Set(["after"]), { required: ["after"] });
    checkPoint(`${where}.after`, action["after"]);
  }
}

/**
 * Throw if an `expect` mapping's `events` or `stream` write the `attempt: 1` the data form leaves
 * out of a model call event or stream chunk; `key` names the mapping.
 */
function checkAttempts(key: string, expect: Case): void {
  const isOne = (data: Value | undefined) =>
    isMapping(data) && (data["attempt"] === 1n || data["attempt"] === 1);
  const rejectAttemptOne = (where: string): never => {
    throw new Error(`\`${where}\` writes \`attempt: 1\`, which the data form leaves out`);
  };
  const events = get(expect, "events", []);
  (Array.isArray(events) ? events : []).forEach((event, index) => {
    const named = Array.isArray(event) && ATTEMPT_LEFT_OUT_AT_1.has(String(event[0]));
    if (named && isOne(event[1])) rejectAttemptOne(`${key}.events[${index}]`);
  });
  const stream = get(expect, "stream", []);
  (Array.isArray(stream) ? stream : []).forEach((chunk, index) => {
    if (isOne(chunk)) rejectAttemptOne(`${key}.stream[${index}]`);
  });
}

/**
 * Throw unless `value`, the host, its tools, models and their scripts and its steering use only
 * the format's keys.
 */
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
          if (Object.hasOwn(error, "retry_after")) {
            if (error["class"] !== "TransientToolError") {
              throw new Error(`\`${at}.raise.retry_after\` is for TransientToolError`);
            }
            const seconds = error["retry_after"];
            const number = typeof seconds === "bigint" || typeof seconds === "number";
            // Finite, as `TransientToolError` requires.
            if (!number || !(seconds >= 0 && seconds < Infinity)) {
              throw new Error(`\`${at}.raise.retry_after\` must be a finite non-negative number`);
            }
          }
        }
      });
    }
  }
  checkModels(get(host, "models", {}));
  if (Object.hasOwn(host, "memory")) checkMemory(host["memory"]);
  checkSteering(get(host, "steering", {}));
  checkStop(get(host, "stop", {}));
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
    if (Object.hasOwn(case_, key)) checkArguments(key, case_[key], allowed);
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
    checkAttempts(key, expect);
  }
  if (Object.hasOwn(case_, "backward")) checkBackward("backward", case_["backward"]);
  if (Object.hasOwn(case_, "host")) checkHost(case_["host"], kind);
  for (const key of ["expect", "python_expect"]) {
    const expect = get(case_, key, {}) as Case;
    if (!Object.hasOwn(expect, "memory")) continue;
    if (!Object.hasOwn(get(case_, "host", {}) as Case, "memory")) {
      throw new Error(`\`${key}.memory\` needs \`host.memory\``);
    }
    const texts = Object.values(mapping(`${key}.memory`, expect["memory"]));
    if (texts.some((text) => typeof text !== "string")) {
      throw new Error(`\`${key}.memory\` values must be text`);
    }
  }
  // Both were checked as mappings above when present.
  const run = get(case_, "run", {}) as Case;
  if (kind === "resume") {
    const host = get(case_, "host", {}) as Case;
    const stops = Object.hasOwn(host, "stop");
    if (stops && Object.hasOwn(host, "crash_after_record")) {
      throw new Error("a `resume` case takes one of `host.crash_after_record` and `host.stop`");
    }
    const cut = get(host, "crash_after_record");
    if (!stops && cut !== "end" && (typeof cut !== "bigint" || cut < 0n)) {
      throw new Error(
        "a `resume` case needs `host.crash_after_record`, a record number or `end`, or `host.stop`",
      );
    }
    if (!isZero(get(run, "num_workers"))) {
      throw new Error(`\`host.${stops ? "stop" : "crash_after_record"}\` needs \`run.num_workers: 0\``);
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

/** Return both sides as YAML, or "" when `actual` is the same as `expected`. */
function difference(what: string, actual: unknown, expected: unknown, flags: SameFlags = {}): string {
  if (same(actual, expected, flags)) return "";
  return `${what} differs\n--- actual\n${dump(actual)}--- expected\n${dump(expected)}`;
}

/** Throw a `CaseFailure` showing both sides as YAML unless `actual` is the same as `expected`. */
export function compare(
  what: string,
  actual: unknown,
  expected: unknown,
  flags: SameFlags = {},
): void {
  const text = difference(what, actual, expected, flags);
  if (text) throw new CaseFailure(text);
}

/**
 * TS error class names that differ from the case names, mapped to the case names. A change that
 * introduces such a class adds its entry here and the matching row to the error-class table in
 * `conformance/README.md`.
 */
export const ERROR_CLASSES: ReadonlyMap<string, string> = new Map([
  // TS throws a RangeError wherever Python raises ValueError at a yian boundary.
  ["RangeError", "ValueError"],
  // A RangeError naming a run-log schema TS can't read; Python raises ValueError.
  ["RunLogSchemaError", "ValueError"],
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
  /** Whether `--crash-sweep` asks an eligible `run` case for its crash sweep too. */
  crashSweep?: boolean;
  /** The path of the case's golden log, `<group>/<name>.log.jsonl` beside it. */
  golden?: string;
  /** Whether `--write-logs` asks a `run`, `resume` or `backward` case to write its golden log. */
  writeLogs?: boolean;
  /**
   * With `--write-expect`, where a `run`, `resume` or `backward` case puts each `REWRITTEN` result
   * that differs from its `expect`, by name, instead of failing on it.
   */
  rewrites?: Map<string, unknown>;
}

/** Return the `Env` that reads the case's environment; a name that is not an own key is unset. */
function caseEnv(ctx: CaseContext): Env {
  return (name) => (Object.hasOwn(ctx.env, name) ? ctx.env[name] : undefined);
}

/** Run one prepared case of a kind; throw (a `CaseFailure`, normally) when it fails. */
export type KindRunner = (case_: Case, ctx: CaseContext) => void | Promise<void>;

// `RESERVED` shrinks as the kinds that read `host` support its keys.
/** The kind runners, one for every kind. */
export const KINDS: Record<Kind, KindRunner> = {
  load: runLoad,
  run: runRun,
  resume: runRun,
  backward: runRun,
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
  options: Pick<CaseContext, "crashSweep" | "golden" | "writeLogs" | "rewrites"> = {},
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
    ctx = { ...setUp(case_, fixturesDir), ...options };
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
  "[--diagnostic-code CODE] [--crash-sweep] [--write-logs] [--write-expect]";

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
 * `--crash-sweep` also runs each eligible `run` case's crash sweep (`crashSweep`); `--write-logs`,
 * which needs `--case`, writes each selected `run`, `resume` and `backward` case's golden log
 * (`goldenLog`); `--write-expect` writes back each such passing case's `REWRITTEN` results that
 * differ (`rewriteCase`) and reports a `wrote` line for it.
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
  let flags: Partial<Record<"group" | "kind" | "case" | "rules" | "diagnostic-code", string[]>> & {
    "crash-sweep"?: boolean;
    "write-logs"?: boolean;
    "write-expect"?: boolean;
  };
  try {
    flags = parseArgs({
      args: [...argv],
      strict: true,
      options: {
        group: option, kind: option, case: option, rules: option, "diagnostic-code": option,
        "crash-sweep": { type: "boolean" },
        "write-logs": { type: "boolean" },
        "write-expect": { type: "boolean" },
      },
    }).values;
  } catch {
    writeError(USAGE);
    return 2;
  }
  // Pinning a golden log is a choice per case, never a whole suite's.
  if (flags["write-logs"] === true && flags.case === undefined) {
    writeError("--write-logs needs --case: name each case whose golden log to write");
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
      // A pending case's expectation is the one it should come to meet, so it is never written.
      const rewrites =
        flags["write-expect"] === true && !Object.hasOwn(entry.case_, "pending")
          ? new Map<string, unknown>()
          : undefined;
      failure = await runCase(entry.case_, kinds, fixturesDir, {
        crashSweep: flags["crash-sweep"] === true,
        golden: join(casesDir, `${entry.id}.log.jsonl`),
        writeLogs: flags["write-logs"] === true,
        rewrites,
      });
      if (failure === undefined && rewrites !== undefined && rewrites.size > 0) {
        const path = join(casesDir, `${entry.id}.yaml`);
        try {
          writeFileSync(path, rewriteCase(readFileSync(path, "utf8"), rewrites));
          write(`wrote ${entry.id}: ${[...rewrites.keys()].join(", ")}`);
        } catch (error) {
          failure = `--write-expect can't write expect back: ${failureText(error)}`;
        }
      }
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

/** The `resumeFlow` option each `resume` key sets. */
const RESUME_OPTIONS: Readonly<Record<string, keyof ResumeFlowOptions>> = {
  num_workers: "numWorkers",
  api_key: "apiKey",
  headers: "headers",
  context: "context",
};

/**
 * Return the `run` or `resume` arguments `args` as the options `names` gives each key: an int as a
 * JS number, `context` an object.
 */
function runOptions<T>(args: Case, names: Readonly<Record<string, keyof T>>): T {
  return Object.fromEntries(
    Object.entries(args).map(([key, value]) => [
      names[key],
      key === "context" ? {} : typeof value === "bigint" ? Number(value) : value,
    ]),
  ) as T;
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
 * Return the error a script's `raise` mapping `{class, message, retry_after, status}` names. With
 * `status` (a model's), it is the `ProviderError` an HTTP reply with that status and the message
 * as its body gives; else an error of the class `classes` names, else a plain `Error` named
 * `class` (`TimeoutError` and `ConnectionError` among them).
 */
function scriptedError(error: Case, classes: ReadonlyMap<string, ErrorClass>): Error {
  const name = error["class"] as string;
  const message = error["message"] as string;
  if (Object.hasOwn(error, "status")) {
    const status = Number(error["status"]);
    return new ProviderError(kindForStatus(status, message), message, { status });
  }
  const Class = classes.get(name);
  if (Class === undefined) return Object.assign(new Error(message), { name });
  const options = Object.hasOwn(error, "retry_after") ? { retryAfter: Number(error["retry_after"]) } : {};
  return new Class(message, options);
}

// How the user message `generateStructured` adds after an invalid reply starts.
const CORRECTION = "Your previous output was invalid:";
// How the system message of an AGENT's history summary request starts.
const SUMMARY_PROMPT = "Summarize the historical conversation";

/**
 * Return the consumer a forward model call of node `nodeId` with `request` is scripted under:
 * `summary` for an AGENT's history summary (its first message is the summary system prompt),
 * `structured_retry` for a structured call's attempts after its first (each ends in the
 * correction an invalid reply adds), else `compose` for a `<gate>__compose` node and `agent` for
 * any other.
 */
function consumerOf(nodeId: string, request: Request): string {
  const first = request.messages[0];
  if (first?.role === "system" && first.text().startsWith(SUMMARY_PROMPT)) return "summary";
  const last = request.messages.at(-1);
  if (last?.role === "user" && last.text().startsWith(CORRECTION)) return "structured_retry";
  return nodeId.endsWith("__compose") ? "compose" : "agent";
}

/** Return a case usage as a `Usage`. */
function caseUsage(usage: Case): Usage {
  const count = (name: string) => (Object.hasOwn(usage, name) ? Number(usage[name]) : null);
  return new Usage({
    inputTokens: Number(usage["input_tokens"]),
    outputTokens: Number(usage["output_tokens"]),
    totalTokens: Number(usage["total_tokens"]),
    cacheReadTokens: count("cache_read_tokens"),
    cacheCreationTokens: count("cache_creation_tokens"),
    reasoningTokens: count("reasoning_tokens"),
  });
}

/**
 * Return a `host.models` entry as the reply a provider streams: a `reply` as a `Response`, which
 * streams as one chunk; `chunks` as events, then the `raise` error when there is one. All text
 * chunks build one text part, first when there is any, and each tool call index its own part,
 * in the order first seen.
 */
function scriptedReply(entry: Case): ScriptedReply {
  if (Object.hasOwn(entry, "reply")) {
    const usage = Object.hasOwn(entry, "usage") ? caseUsage(entry["usage"] as Case) : null;
    return new Response({ message: Message.fromJSON(entry["reply"]), usage });
  }
  const chunks = entry["chunks"] as Case[];
  const parts = new Map<string, number>();
  // Text first, as a LangChain message keeps its text apart from its tool calls.
  if (chunks.some((chunk) => Object.hasOwn(chunk, "text"))) parts.set("text", 0);
  const part = (key: string): number => {
    if (!parts.has(key)) parts.set(key, parts.size);
    return parts.get(key)!;
  };
  const events: (Event | Error)[] = chunks.map((chunk) => {
    if (Object.hasOwn(chunk, "text")) return new TextDelta(part("text"), chunk["text"] as string);
    if (Object.hasOwn(chunk, "usage")) return new UsageReport(caseUsage(chunk["usage"] as Case));
    if (Object.hasOwn(chunk, "finish")) {
      return new Finish({ reason: chunk["finish"] as string | null });
    }
    const delta = chunk["tool_call"] as Case;
    return new ToolCallDelta({
      index: part(`tool_call ${String(delta["index"])}`),
      id: (get(delta, "id") ?? null) as string | null,
      name: (get(delta, "name") ?? null) as string | null,
      arguments: delta["arguments"] as string,
    });
  });
  if (Object.hasOwn(entry, "raise")) events.push(scriptedError(entry["raise"] as Case, new Map()));
  return events;
}

/**
 * Return a model call's request in the case's form: its messages as yian.message.v1 messages,
 * its tools as `{name, description, parameters}` unless none are bound, and its native
 * structured output's schema unless it asks for none.
 */
function v1Request(request: Request): Case {
  const form: Case = { messages: request.messages.map((message) => message.toJSON() as Case) };
  if (request.tools.length > 0) {
    form["tools"] = request.tools.map(
      ({ name, description, parameters }) => ({ name, description, parameters }) as Case,
    );
  }
  if (request.outputFormat !== null) form["schema"] = request.outputFormat.schema as Case;
  return form;
}

/** A point of the run a `host.steering` action waits for: its `after`. */
type Point = { model_call?: bigint; chunk?: bigint; tool_call?: bigint };

/** The error the runner stops a run with at its `host.stop` point. */
class ConformanceStop extends Error {
  override name = "ConformanceStop";

  constructor() {
    super("conformance stop");
  }
}

/**
 * The `host.steering` inbox an AGENT reads (`Steering`): steers and interrupts by node id, and the
 * `host.stop` points.
 *
 * An action without `after` is queued at once, the others when the run reaches their point.
 * Steers are drained once; an interrupt aborts the node's interrupt signal, which stays aborted.
 * Reaching a stop calls `onStop`. Calls are counted, and points reached, only once `armed`, so
 * that runs before the main one neither count nor steer.
 */
class Inbox implements Steering {
  armed = false;
  readonly #steers = new Map<string, string[]>();
  readonly #interrupts = new Map<string, AbortController>();
  readonly #counts = new Map<string, number>();
  readonly #pending: { label: string; node: string; action: Case }[] = [];
  readonly #onStop: () => void;

  constructor(steering: Case, stop: Case = {}, onStop: () => void = () => {}) {
    this.#onStop = onStop;
    for (const [node, action] of Object.entries(stop)) {
      this.#pending.push({ label: `host.stop.${node}`, node, action: { ...(action as Case), stop: true } });
    }
    for (const [node, actions] of Object.entries(steering)) {
      (actions as Case[]).forEach((action, index) => {
        if (Object.hasOwn(action, "after")) {
          this.#pending.push({ label: `host.steering.${node}[${index}]`, node, action });
        } else {
          this.#act(node, action);
        }
      });
    }
  }

  /** Queue `action`'s steers for `node`, abort its interrupt signal, or stop the run. */
  #act(node: string, action: Case): void {
    if (action["stop"] === true) {
      this.#onStop();
      return;
    }
    const steers = this.#steers.get(node) ?? [];
    this.#steers.set(node, [...steers, ...(get(action, "steer", []) as string[])]);
    if (action["interrupt"] === true) this.#controller(node).abort();
  }

  /** Return `node`'s interrupt controller, made on first use. */
  #controller(node: string): AbortController {
    let controller = this.#interrupts.get(node);
    if (controller === undefined) this.#interrupts.set(node, (controller = new AbortController()));
    return controller;
  }

  /** Count a `model_call` or `tool_call` of `node`; return its number, `null` unarmed. */
  count(node: string, what: "model_call" | "tool_call"): bigint | null {
    if (!this.armed) return null;
    const key = `${what} ${node}`;
    const number = (this.#counts.get(key) ?? 0) + 1;
    this.#counts.set(key, number);
    return BigInt(number);
  }

  /** Act on the pending actions of `node` whose `after` is `point`. */
  reached(node: string, point: Point): void {
    const due = this.#pending.filter((each) => each.node === node && same(each.action["after"], point));
    for (const item of due) {
      this.#pending.splice(this.#pending.indexOf(item), 1);
      this.#act(node, item.action);
    }
  }

  /** Remove and return the steers queued for `nodeId`. */
  drain(nodeId: string): readonly string[] {
    const steers = this.#steers.get(nodeId) ?? [];
    this.#steers.delete(nodeId);
    return steers;
  }

  /** The signal aborted once `nodeId`'s interrupt is set. */
  interruptSignal(nodeId: string): AbortSignal {
    return this.#controller(nodeId).signal;
  }

  /** Return a message per action whose point was never reached. */
  left(): string[] {
    return this.#pending.map(({ label }) => `\`${label}\` was never reached`);
  }
}

/**
 * Yield `events`, a scripted call's, reaching its `inbox` points at `[node, number]`, its model
 * call: chunk `k` once its events are delivered (a `reply`, whose events are one chunk, after all
 * of them; a `chunks` entry's, one event each), then the end of the call when it doesn't raise.
 */
async function* reaching(
  events: EventStream,
  reply: boolean,
  inbox: Inbox,
  [node, number]: [string, bigint],
): AsyncGenerator<Event> {
  let chunk = 0n;
  while (true) {
    const result = await events.next();
    if (result.done) break;
    yield result.value;
    // Reached when the next event is asked for, which is after the AGENT has read this one.
    if (!reply) inbox.reached(node, { model_call: number, chunk: ++chunk });
  }
  if (reply) inbox.reached(node, { model_call: number, chunk: 1n });
  inbox.reached(node, { model_call: number });
}

/**
 * Return a builder of chat models that serve each call the next `host.models` entry of its node
 * and consumer, taken through `take` when the call is made: the consumer the model was built for
 * (`backward`, `memory:<key>`), else the call's own (`consumerOf`). A call's request that
 * differs from its entry's `request` is added to `problems`. A call `take` refuses fails at its
 * stream's first read, as a provider's failure does. An `agent` call reaches its `inbox` points
 * (`reaching`).
 */
function scriptedModels(
  take: (label: string, what: string) => Value,
  problems: string[],
  inbox: Inbox,
): (provider: string, model: string, nodeId: string, builtFor?: string) => ChatModel {
  const served = new Map<string, number>();
  return (provider, model, nodeId, builtFor) => ({
    provider,
    model,
    stream: (request, options = {}) => {
      const consumer = builtFor ?? consumerOf(nodeId, request);
      const label = `host.models.${nodeId}.${consumer}`;
      let entry: Case;
      try {
        entry = take(label, `a model call for \`${consumer}\` from node ${reprText(nodeId)}`) as Case;
      } catch (error) {
        return new EventStream({
          next: () => {
            throw error;
          },
        });
      }
      const index = (served.get(label) ?? -1) + 1;
      served.set(label, index);
      if (Object.hasOwn(entry, "request")) {
        const what = `\`${label}[${index}].request\``;
        // By value: TS holds a JSON number as a JS `number` whether it is written `1` or `1.0`.
        const text = difference(what, v1Request(request), entry["request"]!, { byValue: true });
        if (text) problems.push(text);
      }
      const scriptedModel = new ScriptedModel([scriptedReply(entry)], { provider, model });
      const events = scriptedModel.stream(request, options);
      const number = consumer === "agent" ? inbox.count(nodeId, "model_call") : null;
      if (number === null) return events;
      const reply = Object.hasOwn(entry, "reply");
      return new EventStream(reaching(events, reply, inbox, [nodeId, number]), () => events.close());
    },
  });
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
  /** Builds every model the run calls; no call reaches a provider. */
  models: ModelAccess;
  /** The `host.steering` inbox, which also reaches the `host.stop` points. */
  inbox: Inbox;
  /** With `host.stop`, the run's signal and the error it is aborted with at the point. */
  stop: { signal: AbortSignal; error: ConformanceStop } | null;
  /** Each tool's calls in order, by tool name, as `{args, call_id, attempt, node}`. */
  calls: Map<string, Case[]>;
  /** Each script entry taken, in order, with the log's `nextSequence` when it was taken. */
  taken: { label: string; entry: Value; sequence: number }[];
  /** Put each entry taken while the log's `nextSequence` was above `last` back at its script's front. */
  putBack: (last: number) => void;
  /**
   * Throw a `CaseFailure` when a script ran out or was missing or a model call's request differed
   * from its entry's; return a message per script with entries left and per `host.steering`
   * action never reached.
   */
  check: () => string[];
}

/**
 * Return the case's host: each stub tool call is recorded and takes its script's next entry, as
 * each `ask_user` call takes the next `host.human` answer of its node and each backoff the next
 * `host.jitter` draw and each model call the next `host.models` entry of its node and consumer.
 * Every model the run builds is the scripted one, after its config is resolved and checked with
 * `env`. Each stub call reaches its `host.steering` point once it takes its entry. `log` is the
 * run's log file.
 */
function host(case_: Case, log: string, env: Env): Host {
  const spec = get(case_, "host", {}) as Case;
  const calls = new Map<string, Case[]>();
  const problems: string[] = [];
  const taken: Host["taken"] = [];
  const stopper = new AbortController();
  const stopError = new ConformanceStop();
  const stop = Object.hasOwn(spec, "stop") ? { signal: stopper.signal, error: stopError } : null;
  const inbox = new Inbox(get(spec, "steering", {}) as Case, get(spec, "stop", {}) as Case, () =>
    stopper.abort(stopError),
  );
  const scripts = new Map<string, Value[]>(
    Object.entries(get(spec, "human", {}) as Case).map(([node, answers]) => [
      `host.human.${node}`,
      [...(answers as Value[])],
    ]),
  );

  const refuse = (problem: string, message: string): never => {
    problems.push(problem);
    // Permanent, so a TOOL node doesn't retry the overrun; a model call's overrun fails its
    // AGENT. `check()` reports it either way.
    throw new PermanentToolError(message);
  };

  const take = (label: string, what: string): Value => {
    const script = scripts.get(label);
    if (script === undefined) return refuse(`${what} found no \`${label}\` list`, `\`${label}\` is used up`);
    if (script.length === 0) return refuse(`${what} found \`${label}\` used up`, `\`${label}\` is used up`);
    const entry = script.shift()!;
    taken.push({ label, entry, sequence: nextSequence(log) });
    return entry;
  };

  const call = (name: string, label: string, ctx: ToolContext, args: Case): Value => {
    const record = { args, call_id: ctx.callId, attempt: BigInt(ctx.attempt), node: ctx.node };
    let list = calls.get(name);
    if (list === undefined) calls.set(name, (list = []));
    list.push(record);
    const number = inbox.count(ctx.node, "tool_call");
    const entry = take(label, `a call to \`${name}\` from node ${reprText(ctx.node)}`);
    if (number !== null) inbox.reached(ctx.node, { tool_call: number });
    return entry;
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

  for (const [node, consumers] of Object.entries(get(spec, "models", {}) as Case)) {
    for (const [consumer, script] of Object.entries(consumers as Case)) {
      scripts.set(`host.models.${node}.${consumer}`, [...(script as Value[])]);
    }
  }
  const scripted = scriptedModels(take, problems, inbox);
  // Installed even without `host.models`, so an unscripted model call never reaches a provider.
  const models: ModelAccess = {
    env,
    createModel: (config, { nodeId, consumer }) => {
      // The script replaces only the provider: the config is resolved and checked as yian's own
      // factory does, so its errors stay yian's. No credential is read.
      const values = fromHost(config, "llm_config") as Record<string, unknown>;
      const [validated] = checkModelConfig(resolveModelConfig(values, { env }));
      return scripted(validated.provider!, validated.model!, nodeId, consumer);
    },
  };

  const check = (): string[] => {
    if (problems.length > 0) throw new CaseFailure(problems.join("\n"));
    return [...scripts]
      .filter(([, script]) => script.length > 0)
      .map(([label, script]) => {
        const entries = script.length === 1 ? "entry" : "entries";
        return `\`${label}\` has ${script.length} ${entries} left`;
      })
      .concat(inbox.left());
  };
  const putBack = (last: number): void => {
    // Taken in `nextSequence` order, so the entries to put back are a tail.
    const first = taken.findIndex(({ sequence }) => sequence > last);
    if (first === -1) return;
    for (const item of taken.splice(first).reverse()) scripts.get(item.label)!.unshift(item.entry);
  };
  return { tools: Object.fromEntries(tools), random, models, inbox, stop, calls, taken, putBack, check };
}

/** Return a run's `failure` in the case's failure form. */
function failureForm(failure: Failure, flow: ComposedFlow): Case {
  return { ...diagnosticForm(flow.diagnose(failure)), node: failure.nodeId };
}

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

// The README's trace table: record kind -> [the payload field naming the node, the payload fields
// kept].
type TraceRow = readonly [string | null, readonly string[]];
const TRACE: ReadonlyMap<string, TraceRow> = new Map<string, TraceRow>([
  [
    "run_started",
    [null, ["learnable", "num_workers", "defaults", "effective_llm_by_static_path", "call_time_llm", "parent"]],
  ],
  ["run_resumed", [null, ["retry_node_ids", "num_workers"]]],
  ["run_stopped", [null, ["reason", "node_ids"]]],
  ["node_started", ["node_id", []]],
  ["node_step", ["node_id", ["n", "step"]]],
  ["pool_commit", ["node_id", ["key", "value"]]],
  ["node_succeeded", ["node_id", ["commit", "forward_capture"]]],
  ["node_routed", ["node_id", ["handle", "terminal_failure", "forward_capture"]]],
  ["node_expanded", ["spawner_id", ["seed", "prune_ids", "forward_capture"]]],
  ["node_failed", ["failure", ["failure", "forward_capture"]]],
  ...["memory_commit_requested", "memory_commit_succeeded", "memory_commit_failed"].map(
    (kind): [string, TraceRow] => [
      kind,
      [null, ["memory", "commit_id", "attempt_id", "mutations", "failure"]],
    ],
  ),
  ["backward_started", [null, ["attempt_id", "feedback"]]],
  ["backward_ready", [null, ["attempt_id", "root_result", "attributed_results", "proposals"]]],
  ["backward_finished", [null, ["attempt_id", "outcome", "attributed_results"]]],
]);
/** The run start's commits whose values are times, which the trace leaves out. */
const TIMED_KEYS: ReadonlySet<string> = new Set(["system.now", "system.today"]);
/** The fields of a node step that are times (an AGENT's model calls and tool results). */
const STEP_TIMES: readonly string[] = ["started_at", "first_token_at"];
const LEFT_OUT_WHEN_EMPTY: ReadonlySet<string> = new Set([
  "defaults", "effective_llm_by_static_path", "call_time_llm", "parent",
]);
const CAPTURE_LEFT_OUT_WHEN_EMPTY: ReadonlySet<string> = new Set([
  "evaluated_keys", "items", "node_data", "input_names", "variable_names",
]);
const FAILURE_FIELDS: ReadonlyMap<string, string> = new Map([
  ["node_failed", "failure"],
  ["node_routed", "terminal_failure"],
]);

/** Return a connection record with only its `method`, `source` and `endpoint`. */
function cut(connection: Case): Case {
  const names = ["method", "source", "endpoint"].filter((name) => Object.hasOwn(connection, name));
  return Object.fromEntries(names.map((name) => [name, connection[name]!]));
}

/**
 * Return an agent's LLM config projected to the trace: its `llm_config` fields (keys not
 * starting with `_`), `_header_names` and `_host_key` stay, and its `_connection` is cut; any
 * other `_` key is dropped.
 */
function llm(config: Case): Case {
  const kept = Object.fromEntries(
    Object.entries(config).filter(
      ([key]) => !key.startsWith("_") || key === "_header_names" || key === "_host_key",
    ),
  );
  const connection = get(config, "_connection") ?? null;
  if (connection !== null) kept["_connection"] = cut(connection as Case);
  return kept;
}

/** Return whether a route field holds its empty value, as Python's `in (None, [], 0, "")` reads. */
function emptyRouteValue(value: Value): boolean {
  // Python's `0` also equals `0.0` and `False`.
  return (
    value === null || value === "" || value === 0 || value === 0n || value === false ||
    (Array.isArray(value) && value.length === 0)
  );
}

/** Return a `forward_capture` projected to the trace, changing it in place. */
function capture(captured: Case): Case {
  const route = (node: Case): Case => {
    // `expression` and `result_type` are text each implementation spells its own way.
    const lean: Case = Object.fromEntries(
      Object.entries(node).filter(
        ([key, value]) =>
          key !== "expression" && key !== "result_type" && (key === "kind" || !emptyRouteValue(value)),
      ),
    );
    if (Object.hasOwn(lean, "value_json")) {
      lean["value_json"] = parseJson(lean["value_json"] as string, "codec") as Value;
    }
    if (Object.hasOwn(lean, "children")) lean["children"] = (lean["children"] as Case[]).map(route);
    if (Object.hasOwn(lean, "snapshot")) lean["snapshot"] = route(lean["snapshot"] as Case);
    return lean;
  };
  const data = get(captured, "node_data") ?? null;
  if (isMapping(data) && (get(data, "effective_llm_config") ?? null) !== null) {
    data["effective_llm_config"] = llm(data["effective_llm_config"] as Case);
  }
  captured["route"] = route(captured["route"] as Case);
  for (const item of captured["items"] as Case[]) item["capture"] = capture(item["capture"] as Case);
  return Object.fromEntries(
    Object.entries(captured).filter(
      ([key, value]) => truthy(value) || !CAPTURE_LEFT_OUT_WHEN_EMPTY.has(key),
    ),
  );
}

/** Return the run log's `lines` projected to the README's trace. */
function trace(lines: readonly string[], flow: ComposedFlow): Value[][] {
  const failure = (payload: Case): Case => {
    const origins = (payload["trace"] as Case[]).map(
      (origin) => new OriginRef(origin["key"] as string),
    );
    return failureForm(
      new Failure({
        code: payload["code"] as FailureCode,
        message: payload["message"] as string,
        nodeId: payload["node_id"] as string | null,
        trace: origins,
      }),
      flow,
    );
  };
  return lines.map((line) => {
    // Read with the codec, so ints stay ints and floats floats. Each record is parsed afresh, so
    // its payload is projected in place.
    const record = parseJson(line, "codec") as Case;
    const kind = record["kind"] as string;
    const payload = record["payload"] as Case;
    const row = TRACE.get(kind);
    if (row === undefined) {
      throw new CaseFailure(`the run log holds a \`${kind}\` record the trace has no row for`);
    }
    const [nodeField, kept] = row;
    let node = nodeField === null ? null : (payload[nodeField] ?? null);
    if (kind === "node_failed") node = (node as Case)["node_id"] ?? null;
    if (kind === "run_started") {
      const effective = payload["effective_llm_by_static_path"] as Case;
      payload["effective_llm_by_static_path"] = Object.fromEntries(
        Object.entries(effective).map(([path, config]) => [path, llm(config as Case)]),
      );
      const callTime = get(payload, "call_time_llm") ?? null;
      if (callTime !== null) {
        // `selection` is internal; its effect is each agent's `effective_llm_config` step.
        const agents = (callTime as Case)["agents"] as Case;
        payload["call_time_llm"] = {
          agents: Object.fromEntries(
            Object.entries(agents).map(([path, config]) => [path, llm(config as Case)]),
          ),
        };
      }
    }
    if (kind === "pool_commit") {
      if (payload["key"] === node) delete payload["key"];
      // The run start's clock values are times.
      else if (TIMED_KEYS.has(payload["key"] as string)) delete payload["value"];
    }
    const key = FAILURE_FIELDS.get(kind);
    if (key !== undefined && (payload[key] ?? null) !== null) {
      payload[key] = failure(payload[key] as Case);
    }
    if ((get(payload, "forward_capture") ?? null) !== null) {
      payload["forward_capture"] = capture(payload["forward_capture"] as Case);
    }
    if (kind === "node_step") {
      const step = payload["step"] as Case;
      if (Object.hasOwn(step, "retry_in") && step["retry_in"] !== null) step["retry_in"] = Number(step["retry_in"]);
      // Step times are times.
      for (const time of STEP_TIMES) delete step[time];
      if ((get(step, "effective_llm_config") ?? null) !== null) {
        step["effective_llm_config"] = llm(step["effective_llm_config"] as Case);
      }
      // A failed step's error class goes through the error-class table: a TOOL attempt's `type`,
      // a failed model call's `kind` (a provider error's kind is no class and stays).
      const error = get(step, "error");
      if (step["event"] === "failed" && isMapping(error) && typeof error["type"] === "string") {
        error["type"] = ERROR_CLASSES.get(error["type"]) ?? error["type"];
      }
      if (step["kind"] === "model_call" && isMapping(error) && typeof error["kind"] === "string") {
        error["kind"] = ERROR_CLASSES.get(error["kind"]) ?? error["kind"];
      }
    }
    if (kind === "backward_finished") {
      const backward = get(payload["outcome"] as Case, "failure") ?? null;
      if (backward !== null) {
        const failed = backward as Case;
        delete failed["cause_type"];
        delete failed["traceback"];
        // A backward failure's `cause` is a node failure when it has a `trace`.
        const cause = failed["cause"];
        if (isMapping(cause) && Object.hasOwn(cause, "trace")) failed["cause"] = failure(cause);
      }
    }
    const projected = Object.fromEntries(
      Object.entries(payload).filter(
        ([name, value]) =>
          kept.includes(name) && value !== null && !(LEFT_OUT_WHEN_EMPTY.has(name) && !truthy(value)),
      ),
    );
    return Object.keys(projected).length > 0 ? [kind, node, projected] : [kind, node];
  });
}

/** Return `result` in the form of `expect.output`. */
function output(result: RunResult, flow: ComposedFlow): Case {
  if (result.status === "succeeded") {
    return { status: "succeeded", output: dataForm(result.output as YianValue) };
  }
  return { status: result.status, failure: failureForm(result.failure!, flow) };
}

/**
 * Return `entries` with run-level ones (node `null`) first, then each node's, nodes sorted; each
 * group keeps its order.
 */
function grouped<T>(entries: readonly T[], nodeOf: (entry: T) => string | null): T[] {
  // `sort` is stable, so each group keeps its order.
  return [...entries].sort((a, b) => {
    const [first, second] = [nodeOf(a), nodeOf(b)];
    if (first === null || second === null) return (first === null ? 0 : 1) - (second === null ? 0 : 1);
    return compareCodeUnits(first, second);
  });
}

/** Return `results` with the trace, events, stream and each tool's calls `grouped` by node. */
function pooled(results: Record<string, unknown>): Record<string, unknown> {
  // A `NodeFailed` belongs to its failure's node; run events have no `node_id`.
  const eventNode = (event: Value[]): string | null => {
    const data = (event[1] ?? {}) as Case;
    const node = event[0] === "NodeFailed" ? (data["failure"] as Case)["node"] : data["node_id"];
    return (node ?? null) as string | null;
  };
  const toolCalls = results["tool_calls"] as Record<string, Case[]>;
  // A success's `commit` is the seq of the `pool_commit` just before it in its node's group, a seq
  // that depends on how the parallel nodes interleave, so a grouped trace leaves it out.
  const trace = (results["trace"] as Value[][]).map((entry) => {
    if (entry[0] !== "node_succeeded" || entry[2] === undefined) return entry;
    const { commit: _commit, ...rest } = entry[2] as Case;
    return Object.keys(rest).length > 0 ? [entry[0], entry[1], rest] : [entry[0], entry[1]];
  });
  return {
    ...results,
    trace: grouped(trace, (entry) => entry[1] as string | null),
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
 * order of first appearance walking `results` in their order (the `before` traces first, `memory`
 * last), with mapping keys in UTF-16 order.
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

/** The refused append that cuts a `resume` case's first run. */
class Cut extends Error {
  override name = "Cut";
}

/** What `settle` returns for a run that resolved. */
const ENDED = Symbol("ended");

/** Return what `start` rejects with, or `ENDED` when it resolves. */
async function settle(start: () => Promise<unknown>): Promise<unknown> {
  try {
    await start();
    return ENDED;
  } catch (error) {
    return error;
  }
}

/**
 * Run `start` with every append refused once the log holds record `last`; throw a `CaseFailure`
 * (`at` saying where) unless it rejects with that refusal.
 */
async function cutAt(last: number, start: () => Promise<unknown>, at = ""): Promise<void> {
  const append = JsonlRunLog.prototype.append;
  const refused = new Cut("the conformance runner refused this append");
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record) {
    if (this.nextSequence > last) throw refused;
    return append.call(this, record);
  };
  let outcome: unknown;
  try {
    outcome = await settle(start);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
  if (outcome === ENDED) throw new CaseFailure(`the run ended before record ${last + 1}${at}`);
  if (outcome !== refused) {
    throw new CaseFailure(`the run raised before record ${last + 1}${at}\n${failureText(outcome)}`);
  }
}

/**
 * Run a `resume` case's first run with `start`: cut after record `host.crash_after_record`, run to
 * its end for `end`, or stopped at its `host.stop` point. Throw a `CaseFailure` unless it ended as
 * the case says.
 */
async function cutFirstRun(
  spec: Case,
  start: () => Promise<unknown>,
  stop: { error: ConformanceStop } | null,
): Promise<void> {
  const last = get(spec, "crash_after_record");
  if (typeof last === "bigint") {
    await cutAt(Number(last), start);
    return;
  }
  const outcome = await settle(start);
  if (stop !== null && outcome === ENDED) throw new CaseFailure("the run ended before its `host.stop` point");
  if (outcome !== (stop === null ? ENDED : stop.error)) {
    const before = stop === null ? "its end" : "its stop";
    throw new CaseFailure(`the run raised before ${before}\n${failureText(outcome)}`);
  }
}

/**
 * Return the options every run of a case gets: its log, environment and host (`steering` only
 * with `host.steering`, as Python passes it), on a virtual clock that starts at `start`, with
 * `events` and `chunks` collecting what it publishes.
 */
function runHooks(
  case_: Case,
  log: string,
  env: Env,
  { tools, random, models, inbox }: Host,
  start: number,
  events: RunEvent[] = [],
  chunks: StreamChunk[] = [],
) {
  return {
    log,
    env,
    tools,
    random,
    models,
    ...(Object.hasOwn(get(case_, "host", {}) as Case, "steering") ? { steering: inbox } : {}),
    scheduler: virtualScheduler(start),
    onEvent: (event: RunEvent) => {
      events.push(event);
    },
    onStream: (chunk: StreamChunk) => {
      chunks.push(chunk);
    },
  };
}

/**
 * Return the whole recall text of memory `key` in `memory`, after the main run. Fails the case for
 * a key the flow doesn't declare.
 */
async function recall(memory: MemoryRegistry, key: string): Promise<string> {
  if (!memory.types.has(key)) {
    throw new CaseFailure(`\`expect.memory.${key}\` names no memory the flow declares`);
  }
  const session = await memory.get(key)!.openRun();
  try {
    const content = await session.recall(null);
    return typeof content === "string" ? content : content.text;
  } finally {
    await session.close();
  }
}

/**
 * Call `result.backward` once per `backward` entry, `{feedback, api_key, headers}`, in order;
 * return each call's outcome, as the `backward_finished` record it appended to `log` projects
 * (`trace`), or `{raises}` when the call raised. Fails the case when a call appends other than one
 * `backward_finished` record.
 */
async function callBackward(
  result: RunResult,
  log: string,
  entries: readonly Case[],
  flow: ComposedFlow,
): Promise<Value[]> {
  const finished = () => {
    const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
    return trace(lines, flow).filter(([kind]) => kind === "backward_finished");
  };
  const outcomes: Value[] = [];
  for (const [index, entry] of entries.entries()) {
    const before = finished().length;
    try {
      await result.backward(entry["feedback"], {
        apiKey: (get(entry, "api_key") ?? null) as string | null,
        headers: (get(entry, "headers") ?? null) as Record<string, string> | null,
      });
    } catch (error) {
      if (error instanceof CaseFailure) throw error;
      outcomes.push({ raises: raised(error) });
      continue;
    }
    const after = finished();
    if (after.length !== before + 1) {
      throw new CaseFailure(
        `backward call ${index + 1} appended ${after.length - before} \`backward_finished\` records, not one`,
      );
    }
    outcomes.push((after.at(-1)![2] as Case)["outcome"]!);
  }
  return outcomes;
}

/**
 * Run each `host.memory.before` entry in order, logged to `before-<n>.jsonl` in the case
 * directory, with `options` and memories from `memories()`, closed after the run's backward calls
 * (`callBackward`). Return the log texts by where they are reported and each run's trace (grouped
 * when the run is pooled). A raise, the run's or a backward call's, fails the case.
 */
async function beforeRuns(
  case_: Case,
  ctx: CaseContext,
  flow: ComposedFlow,
  options: RunFlowOptions,
  memories: () => Promise<MemoryRegistry | null>,
): Promise<{ logs: Record<string, string>; traces: Value[][][] }> {
  const logs: Record<string, string> = {};
  const traces: Value[][][] = [];
  const memory = get(get(case_, "host", {}) as Case, "memory", {}) as Case;
  const before = get(memory, "before", []) as Case[];
  for (const [index, entry] of before.entries()) {
    const where = `host.memory.before[${index}]`;
    const log = join(ctx.dir, `before-${index + 1}.jsonl`);
    const run = entry["run"] as Case;
    let outcomes: Value[];
    try {
      const registry = await memories();
      try {
        const inputs = get(entry, "inputs", {}) as Case;
        const args = runOptions<RunFlowOptions>(run, RUN_OPTIONS);
        const result = await runFlow(flow, inputs, { ...options, ...args, log, memory: registry });
        outcomes = await callBackward(result, log, get(entry, "backward", []) as Case[], flow);
      } finally {
        await registry?.close();
      }
    } catch (error) {
      if (error instanceof CaseFailure) throw error;
      throw new CaseFailure(`\`${where}\` raised\n${failureText(error)}`);
    }
    const refused = outcomes.find((outcome) => isMapping(outcome) && Object.hasOwn(outcome, "raises"));
    if (refused !== undefined) {
      throw new CaseFailure(`\`${where}\` backward raised ${spell((refused as Case)["raises"])}`);
    }
    const text = readFileSync(log, "utf8");
    logs[`the log of \`${where}\``] = text;
    const lines = trace(text.split("\n").filter((line) => line !== ""), flow);
    const isPooled = !isZero(get(run, "num_workers", 8n));
    traces.push(isPooled ? grouped(lines, (record) => record[1] as string | null) : lines);
  }
  return { logs, traces };
}

/**
 * Run the flow with `inputs` and the `run` arguments on the case's host, logged to `run.jsonl` in
 * the case directory, its waits on a virtual clock that starts at the real current time; with
 * `host.stop`, the run is stopped at its point. A `resume` case's first run is cut after record
 * `host.crash_after_record` (or, for `end`, runs to its end; with `host.stop`, is stopped) and
 * resumed with the `resume` arguments. Return the results: `trace`, `events`, `stream` (the
 * resume's), `tool_calls` (every run's), `output` or the host error as `raises` and `error` (never
 * compared), `backward` for a `backward` case (each call's outcome, `callBackward`), `memory` when
 * `expect.memory` lists keys, and `left`, a message per script with entries left.
 *
 * With `host.memory`, the `before` runs go first (`beforeRuns`), and every run (each `before` run,
 * the main run, and for a `resume` case the cut run and the resume) gets the flow's memories built
 * afresh under `<case dir>/memory`, closed after the run's backward calls; a failure building the
 * cut run's is the case's `raises`, and nothing resumes. `expect.memory` is read from memories
 * built after the main run's. Learner events are kept apart, never compared.
 */
async function actualRun(case_: Case, ctx: CaseContext): Promise<Record<string, unknown>> {
  const flow = await loadText(case_["flow"] as string, ctx);
  const log = join(ctx.dir, "run.jsonl");
  const env = caseEnv(ctx);
  const spec = get(case_, "host", {}) as Case;
  const scripted = host(case_, log, env);
  const { tools, models, inbox, stop, calls, check } = scripted;
  const run = get(case_, "run", {}) as Case;
  const resume = get(case_, "resume", {}) as Case;
  const events: RunEvent[] = [];
  const chunks: StreamChunk[] = [];
  // The learners' events (TS-only, from `buildMemories`' `onEvent`) are never read: collected here
  // so they stay out of the run's `events`, which `expect.events` compares.
  const apart: unknown[] = [];
  // A run without `host.memory` gets none, leaving the flow's declarations dormant.
  const memories = async (): Promise<MemoryRegistry | null> =>
    Object.hasOwn(spec, "memory")
      ? buildMemories(flow, {
          root: join(ctx.dir, "memory"),
          models,
          onEvent: (event) => apart.push(event),
        })
      : null;
  // One clock for every run of the case. `env` scopes the case's YIAN_* values to each run, so
  // nothing is installed around it.
  const hooks = runHooks(case_, log, env, scripted, Date.now(), events, chunks);
  // The `before` runs share the clock, but not the log, the steering or the observers.
  const { random, scheduler } = hooks;
  const shared: RunFlowOptions = { env, tools, random, models, scheduler };
  const { logs, traces } = await beforeRuns(case_, ctx, flow, shared, memories);
  const inputs = get(case_, "inputs", {}) as Case;
  const signal = stop === null ? {} : { signal: stop.signal };
  const runArguments = runOptions<RunFlowOptions>(run, RUN_OPTIONS);
  let start = (memory: MemoryRegistry | null) =>
    runFlow(flow, inputs, { ...runArguments, ...hooks, ...signal, memory });
  let result: RunResult | undefined;
  let caught: Record<string, unknown> = {};
  inbox.armed = true;
  if (case_["kind"] === "resume") {
    let cutMemory: MemoryRegistry | null = null;
    try {
      cutMemory = await memories();
    } catch (error) {
      // The cut run's memories failing to build is the case's `raises`; nothing resumes.
      caught = { raises: raised(error), error };
    }
    if (!Object.hasOwn(caught, "raises")) {
      // A close error escapes as the case's own error, as Python's `with` lets it.
      try {
        await cutFirstRun(spec, () => start(cutMemory), stop);
      } finally {
        await cutMemory?.close();
      }
      events.length = 0;
      chunks.length = 0;
      const resumeArguments = runOptions<ResumeFlowOptions>(resume, RESUME_OPTIONS);
      start = (memory) => resumeFlow({ ...resumeArguments, ...hooks, memory });
    }
  }
  let outcomes: Value[] = [];
  if (!Object.hasOwn(caught, "raises")) {
    let memory: MemoryRegistry | null = null;
    try {
      memory = await memories();
      result = await start(memory);
    } catch (error) {
      caught = { raises: raised(error), error };
    }
    // The memories stay open for the backward calls, and close even when one fails the case; the
    // close is outside the run's `try`, so a close error never replaces its `raises`.
    try {
      const entries = get(case_, "backward", []) as Case[];
      if (result !== undefined) outcomes = await callBackward(result, log, entries, flow);
    } finally {
      await memory?.close();
    }
  }
  const listed = get(get(case_, "expect", {}) as Case, "memory") as Case | undefined;
  let recalled: Record<string, string> | undefined;
  if (listed !== undefined) {
    const fresh = (await memories())!;
    try {
      recalled = {};
      for (const key of Object.keys(listed)) recalled[key] = await recall(fresh, key);
    } finally {
      await fresh.close();
    }
  }
  // A call past a script's end is the case's own error, whatever it made differ.
  const left = check();
  if (existsSync(log)) await checkInvariants(flow, log, { env, tools, models }, result ?? null);
  goldenLog(log, ctx);
  const text = existsSync(log) ? readFileSync(log, "utf8") : "";
  const lines = text.split("\n").filter((line) => line !== "");
  // In placeholder order: ids are numbered walking the results in this order. The `before`
  // traces come first and are dropped once numbered.
  let actual: Record<string, unknown> = {
    before: traces,
    trace: trace(lines, flow),
    events: events.map((event) => eventForm(event, flow)),
    stream: chunks.map((chunk) => fields(chunk, flow)),
    tool_calls: Object.fromEntries([...calls].map(([name, list]) => [name, list.map(dataForm)])),
    ...(result !== undefined ? { output: output(result, flow) } : {}),
    ...(case_["kind"] === "backward" ? { backward: outcomes } : {}),
    ...(recalled !== undefined ? { memory: recalled } : {}),
  };
  checkSecrets(case_, {
    ...Object.fromEntries(Object.entries(actual).map(([name, value]) => [name, spell(value as Value)])),
    "the run log": text,
    ...logs,
    "the raised error": str((caught["raises"] as Case | undefined)?.["message"] ?? ""),
  });
  // A `resume` case's results are grouped by the resumed run's `num_workers`.
  if (!isZero(get({ ...run, ...resume }, "num_workers", 8n))) actual = pooled(actual);
  const runId =
    Object.hasOwn(run, "run_id") || lines.length === 0
      ? null
      : ((parseJson(lines[0]!, "codec") as Case)["run_id"] as string);
  // A raised error is compared as is, without placeholders.
  actual = placeholders(actual, runId);
  delete actual["before"];
  return { ...actual, ...caught, left };
}

/** The results `--write-expect` writes back. */
const REWRITTEN: readonly string[] = ["trace", "events", "stream", "tool_calls"];
/** The results that compare numbers by value: a run's events, stream and tool calls. */
const BY_VALUE: ReadonlySet<string> = new Set(["events", "stream", "tool_calls"]);

/**
 * Throw a `CaseFailure` unless the results in `actual` are the ones `expect` holds: the raised
 * error, or the output, which compares numbers by value for a success. With `rewrites`, each
 * `REWRITTEN` result that differs goes there by name instead of failing.
 */
function checkResults(
  expect: Case,
  actual: Record<string, unknown>,
  rewrites?: Map<string, unknown>,
): void {
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
  for (const name of [...REWRITTEN, "backward", "memory"]) {
    if (!Object.hasOwn(expect, name)) continue;
    // The trace keeps kinds.
    const byValue = BY_VALUE.has(name);
    if (rewrites !== undefined && REWRITTEN.includes(name)) {
      if (!same(actual[name], expect[name], { byValue })) rewrites.set(name, actual[name]);
    } else {
      compare(name, actual[name], expect[name], { byValue });
    }
  }
  // Last: a leftover entry usually follows from a divergence the comparisons show.
  const left = actual["left"] as string[];
  if (left.length > 0) throw new CaseFailure(left.join("\n"));
}

/**
 * Return the run log text `text` as a golden log holds it: each line with every `traceback` value,
 * its time (`at`), the seeded clock values (the `value` of the `system.now` and `system.today`
 * commits), a step's times (`started_at` and `first_token_at` when set), the
 * recorded flow's folder (`flow.dir`) and the yian version (`yian_version`) replaced by
 * `<masked>`, written again with the log's own encoder. The times and the seeded values come from
 * the case's virtual clock, which starts at the real time; the folder is the case's temporary one;
 * the version changes with each release.
 */
function masked(text: string): string {
  const lines = text.split("\n").filter((line) => line !== "");
  const mask = (value: unknown): void => {
    if (Array.isArray(value)) value.forEach(mask);
    if (!isMapping(value)) return;
    for (const [key, item] of Object.entries(value)) {
      if (key === "traceback" && typeof item === "string") value[key] = "<masked>";
      else mask(item);
    }
  };
  return lines
    .map((line, seq) => {
      const record = parseJson(line, "codec") as Case;
      // Masking re-encodes the line, so it must encode back to itself first.
      if (logJson(record) !== line) {
        throw new CaseFailure(`run log record ${seq} doesn't encode back to its line`);
      }
      mask(record);
      record["at"] = "<masked>";
      const payload = record["payload"] as Case;
      if (record["kind"] === "pool_commit" && TIMED_KEYS.has(payload["key"] as string)) payload["value"] = "<masked>";
      if (record["kind"] === "node_step") {
        const step = payload["step"] as Case;
        for (const time of STEP_TIMES) if ((step[time] ?? null) !== null) step[time] = "<masked>";
      }
      if (record["kind"] === "run_started") {
        const flow = payload["flow"] as Case | null;
        if (flow !== null) flow["dir"] = "<masked>";
        payload["yian_version"] = "<masked>";
      }
      return `${logJson(record)}\n`;
    })
    .join("");
}

/**
 * Compare the run log at `log`, masked, with the case's golden log when it has one, byte for
 * byte; with `--write-logs`, write it instead.
 */
function goldenLog(log: string, ctx: CaseContext): void {
  const golden = ctx.golden;
  if (golden === undefined || !(ctx.writeLogs === true || existsSync(golden))) return;
  if (!existsSync(log)) {
    if (ctx.writeLogs === true) return;
    throw new CaseFailure(`the run wrote no log, but ${basename(golden)} pins one`);
  }
  const text = masked(readFileSync(log, "utf8"));
  if (ctx.writeLogs === true) {
    writeFileSync(golden, text);
    return;
  }
  const expected = readFileSync(golden, "utf8").split("\n");
  const actual = text.split("\n");
  const seq = actual.findIndex((line, index) => line !== expected[index]);
  if (seq === -1 && actual.length === expected.length) return;
  const at = seq === -1 ? actual.length : seq;
  const lines = `--- actual\n${actual[at] ?? ""}\n--- golden\n${expected[at] ?? ""}`;
  throw new CaseFailure(`the run log differs from ${basename(golden)} at record ${at}\n${lines}`);
}

/**
 * Throw a `CaseFailure` naming the first invariant the run log at `log` breaks (`at` says where in
 * a crash sweep): at every prefix, at most the segment's `num_workers` nodes are in flight (1 when
 * 0); after the run's terminal outcome only backward and memory records follow; after a
 * `run_stopped` only node steps until a `run_resumed`; `inspectRun` with the flow reads a replay's
 * status, in-flight nodes, interrupted counts and failure, and the log's run id, schema, flow
 * fingerprint, learnable flag and last seq, and without it the same except in its one blind
 * corner (nothing in flight, a node still ready, and END succeeded or proved unreachable); a finished
 * run's counts are `result`'s; and a finished log resumed with `options` gives the same status and
 * output and appends nothing.
 */
async function checkInvariants(
  flow: ComposedFlow,
  log: string,
  options: { env: Env; tools: Record<string, ToolEntry>; models: ModelAccess },
  result: RunResult | null,
  at = "",
): Promise<void> {
  const fail = (problem: string): never => {
    throw new CaseFailure(`invariant${at}: ${problem}`);
  };
  const envelopes = readRunLog(log);
  if (envelopes.length === 0) return;
  const inFlight = new Set<string>();
  let workers = 1;
  let stopped = false;
  let lastRun = -1;
  envelopes.forEach(({ kind, payload }, seq) => {
    if (stopped && kind !== "node_step" && kind !== "run_resumed") {
      fail(`record ${seq} (${kind}) follows a run_stopped`);
    }
    if (kind === "run_started" || kind === "run_resumed") {
      workers = Math.max(1, Number(payload["num_workers"]));
      inFlight.clear();
      stopped = false;
    }
    if (kind === "run_stopped") stopped = true;
    if (kind === "node_started") inFlight.add(payload["node_id"] as string);
    if (kind === "node_succeeded" || kind === "node_routed") inFlight.delete(payload["node_id"] as string);
    if (kind === "node_expanded") inFlight.delete(payload["spawner_id"] as string);
    if (kind === "node_failed") inFlight.delete((payload["failure"] as Readonly<Case>)["node_id"] as string);
    if (inFlight.size > workers) {
      fail(`${inFlight.size} nodes are in flight after record ${seq}, over ${workers}`);
    }
    if (!kind.startsWith("backward_") && !kind.startsWith("memory_")) lastRun = seq;
  });
  const replay = (count: number) =>
    withEnv(options.env, () => replayRunLog(flow, MemoryRunLog.fromRecords(envelopes.slice(0, count))));
  // A terminal result is never undone, so checking the prefix before the last run record covers
  // every record after it. A prefix inside the run start (through the START inputs' commit, the
  // first whose key is its node) can't be replayed, and holds no outcome.
  const inputsCommit = envelopes.findIndex(({ kind, payload }) => kind === "pool_commit" && payload["key"] === payload["node_id"]);
  if (inputsCommit === -1) return;
  if (lastRun > inputsCommit && replay(lastRun).result !== null) fail(`record ${lastRun} follows the terminal outcome`);
  // Mirrors `inspectRun`'s replay branch on purpose: the check is that it takes that branch and
  // agrees with a replay, not a second statement of its rules.
  const state = replay(envelopes.length);
  const terminal = state.result;
  const finished = terminal === null ? null : terminal instanceof RunSucceeded ? "succeeded" : "failed";
  const counts = new Map(state.interruptedAttempts);
  if (finished === null) addInterruptedAttempts(counts, state.acceptedStarts, state.lastStopSequence);
  const first = envelopes[0]!;
  const started = first.record as RunStartedRecord;
  const expected = {
    status: finished ?? (state.lastStopSequence === null ? "interrupted" : "stopped"),
    inFlight: finished === null ? [...state.acceptedStarts.keys()] : [],
    interruptedAttempts: Object.fromEntries(counts),
    runId: first.runId,
    schema: first.schema,
    flowFingerprint: started.flowFingerprint,
    learnable: started.learnable,
    lastSeq: envelopes.length - 1,
    startedAt: first.at,
    endedAt: finished === null ? null : envelopes.at(-1)!.at,
    failure: terminal instanceof RunFailed ? failureForm(terminal.failure, flow) : null,
  };
  const form = (inspection: RunInspection) => ({
    ...inspection,
    inFlight: [...inspection.inFlight],
    interruptedAttempts: Object.fromEntries(inspection.interruptedAttempts),
    failure: inspection.failure === null ? null : failureForm(inspection.failure, flow),
  });
  compare(`inspectRun with the flow${at}`, form(inspectRun(log, { flow, env: options.env })), expected);
  const blind = form(inspectRun(log));
  const endSucceeded = envelopes.some(
    ({ kind, payload }) => kind === "node_succeeded" && payload["node_id"] === END_ID,
  );
  const corner =
    finished === null &&
    state.acceptedStarts.size === 0 &&
    state.ready.length > 0 &&
    (endSucceeded || state.terminalUnreachableFailure !== null);
  const allowed = corner
    ? { ...blind, status: expected.status, endedAt: expected.endedAt, failure: expected.failure }
    : blind;
  compare(`inspectRun without the flow${at}`, allowed, expected);
  if (finished === null) return;
  if (result !== null) {
    const counted = Object.fromEntries(result.interruptedAttempts);
    compare(`interrupted counts${at}`, counted, expected.interruptedAttempts);
  }
  const before = readFileSync(log);
  const reopened = await resumeFlow({ log, ...options });
  if (!readFileSync(log).equals(before)) fail("resuming the finished log changed it");
  if (reopened.status !== finished) {
    fail(`resuming the finished log gives ${reopened.status}, not ${finished}`);
  }
  if (result !== null) compare(`resumed output${at}`, output(reopened, flow), output(result, flow));
}

/** The trace entries of node outcomes, which a crash sweep's strong oracle compares. */
const OUTCOMES: ReadonlySet<string> = new Set([
  "node_succeeded", "node_routed", "node_expanded", "node_failed",
]);

/**
 * Return the node outcome entries of a run log's trace `entries`, each capture source's
 * `execution_seq` replaced by the kind and node of the record it names, and each success's
 * `commit` by the trace fields of the `pool_commit` it names: captures and successes name records
 * by seq, and a resume adds records, so a crash sweep compares the records instead.
 */
export function sourcesByRecord(entries: readonly Value[][]): Value[][] {
  const rename = (value: unknown): void => {
    if (Array.isArray(value)) value.forEach(rename);
    if (!isMapping(value)) return;
    for (const [key, item] of Object.entries(value)) {
      if (key === "execution_seq" && typeof item === "bigint") {
        // The trace has one entry per log line, and a record's seq is its line index.
        const entry = entries[Number(item)];
        if (entry === undefined) {
          throw new CaseFailure(`a capture source names seq ${item}, which the log doesn't hold`);
        }
        value[key] = `${String(entry[0])} ${String(entry[1])}`;
      } else {
        rename(item);
      }
    }
  };
  const kept = entries.filter(([kind]) => OUTCOMES.has(kind as string));
  rename(kept);
  for (const [kind, , payload] of kept) {
    if (kind !== "node_succeeded") continue;
    const commit = (payload as Case)["commit"] as bigint;
    const named = entries[Number(commit)];
    if (named?.[0] !== "pool_commit") throw new CaseFailure(`a success names seq ${commit}, which isn't a pool_commit`);
    (payload as Case)["commit"] = named[2] ?? {};
  }
  return kept;
}

/**
 * Return whether `case_` gets a crash sweep: a serial `run` case expecting `output`, not
 * `pending`, with no `host.models`, `host.steering`, `host.memory` or `host.stop`.
 */
function sweepable(case_: Case): boolean {
  const spec = get(case_, "host", {}) as Case;
  return (
    case_["kind"] === "run" &&
    isZero(get(get(case_, "run", {}) as Case, "num_workers", 8n)) &&
    Object.hasOwn(case_["expect"] as Case, "output") &&
    !Object.hasOwn(case_, "pending") &&
    !["models", "steering", "memory", "stop"].some((key) => Object.hasOwn(spec, key))
  );
}

/**
 * Run a `run` case's crash sweep. An uncut run writes N records; then, for each k in 0..N-2, a run
 * on a fresh host is cut after record k, the script entries it took after the cut are put back (a
 * cut right after a node's commit puts back from the node's last record before it, since the node
 * runs again), the invariants are checked at the cut, `beforeResume` (for a test, to damage the log) runs, and
 * the run is resumed. A cut inside the run start (before the START inputs' commit) must instead be
 * refused by the resume, leaving the log as it is. The resume must end with a result whose invariants hold. With no
 * `host.human` or `host.jitter` and only `return` entries in its tool scripts, the strong oracle
 * also holds: the same output, and the same node outcome entries in the trace, in order, with a
 * capture's source named by its record's kind and node rather than its seq.
 */
export async function crashSweep(
  case_: Case,
  ctx: CaseContext,
  { beforeResume = (_log: string) => {} } = {},
): Promise<void> {
  const flow = await loadText(case_["flow"] as string, ctx);
  const env = caseEnv(ctx);
  const spec = get(case_, "host", {}) as Case;
  const tools = Object.values(get(spec, "tools", {}) as Case) as Case[];
  const entries = tools.flatMap(({ results }) =>
    (isMapping(results) ? Object.values(results) : [results]).flat(),
  );
  const strong =
    !Object.hasOwn(spec, "human") &&
    !Object.hasOwn(spec, "jitter") &&
    entries.every((entry) => Object.hasOwn(entry as Case, "return"));
  // Every run seeds the same clock, so `system.now` agrees.
  const start = Date.now();
  const inputs = get(case_, "inputs", {}) as Case;
  const args = runOptions<RunFlowOptions>(get(case_, "run", {}) as Case, RUN_OPTIONS);
  const uncutLog = join(ctx.dir, "sweep.jsonl");
  const uncutHooks = runHooks(case_, uncutLog, env, host(case_, uncutLog, env), start);
  const uncut = await runFlow(flow, inputs, { ...args, ...uncutHooks });
  const outcomes = (log: string) => {
    const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
    return sourcesByRecord(trace(lines, flow));
  };
  const expected = outcomes(uncutLog);
  const uncutRecords = readRunLog(uncutLog);
  const count = uncutRecords.length;
  // The run start's last record is the START inputs' commit, the first whose key is its node.
  const inputsCommit = uncutRecords.findIndex(({ kind, record }) => {
    const { nodeId, key } = record as { nodeId?: string; key?: string };
    return kind === "pool_commit" && key === nodeId;
  });
  for (let k = 0; k <= count - 2; k += 1) {
    const at = ` (crash sweep, cut after record ${k})`;
    const log = join(ctx.dir, `sweep-${k}.jsonl`);
    const scripted = host(case_, log, env);
    const hooks = runHooks(case_, log, env, scripted, start);
    // The uncut run's id, so the runs agree on it when the case gives none.
    const cut = () => runFlow(flow, inputs, { ...args, ...hooks, runId: uncut.runId });
    await cutAt(k, cut, at);
    // A node's commit and its success are written with nothing awaited between them, so a cut
    // right after a commit is a crash before the success: the node runs again and takes its
    // script entries again. Entries taken up to the node's last record before the commit stay,
    // since the resume replays that record's steps.
    scripted.putBack(k > inputsCommit && uncutRecords[k]!.kind === "pool_commit" ? k - 1 : k);
    if (k < inputsCommit) {
      // A log cut inside its run start doesn't hold the run's inputs yet, so it can't be resumed.
      const before = readFileSync(log);
      const refused = await settle(() => resumeFlow(hooks));
      const message = `run log ${log} ends inside its run start; start the run again`;
      if (!(refused instanceof RangeError) || refused.message !== message) {
        throw new CaseFailure(`the resume didn't refuse the incomplete run start${at}\n${failureText(refused)}`);
      }
      if (!readFileSync(log).equals(before)) throw new CaseFailure(`the refused resume changed the log${at}`);
      continue;
    }
    const reopen = { env, tools: scripted.tools, models: scripted.models };
    await checkInvariants(flow, log, reopen, null, at);
    beforeResume(log);
    let resumed: RunResult;
    try {
      // The cut run's `context`, which the log doesn't hold, as a `resume` case passes it again.
      const context = args.context === undefined ? {} : { context: args.context };
      resumed = await resumeFlow({ ...hooks, ...context });
    } catch (error) {
      throw new CaseFailure(`the resume raised${at}\n${failureText(error)}`);
    }
    if (strong) {
      compare(`output${at}`, output(resumed, flow), output(uncut, flow), { byValue: true });
      compare(`node outcomes${at}`, outcomes(log), expected);
    } else if (resumed.status !== uncut.status) {
      // Without the strong oracle, a resume still ends as the uncut run did.
      throw new CaseFailure(`the resume ${resumed.status}, the uncut run ${uncut.status}${at}`);
    }
    await checkInvariants(flow, log, reopen, resumed, at);
  }
}

/**
 * Run a `run` or `resume` case (whose first run `actualRun` cuts or stops and resumes): run the
 * flow and compare its results with `expect`; with `--crash-sweep`, an eligible case's crash sweep
 * (`crashSweep`) runs after. With `ctx.rewrites`, the `REWRITTEN` results that differ go there
 * instead of failing the case.
 */
async function runRun(case_: Case, ctx: CaseContext): Promise<void> {
  checkResults(case_["expect"] as Case, await actualRun(case_, ctx), ctx.rewrites);
  if (ctx.crashSweep === true && sweepable(case_)) await crashSweep(case_, ctx);
}

// ---- --write-expect ----

/** Return `value` in a case file's codec form: the inverse of `decode`. */
function encode(value: Value): Value {
  if (typeof value === "number" && !Number.isFinite(value)) {
    if (Number.isNaN(value)) throw new Error("NaN has no case form");
    return { $float: value > 0 ? "inf" : "-inf" };
  }
  if (Array.isArray(value)) return value.map(encode);
  if (!isMapping(value)) return value;
  const record = Object.fromEntries(Object.entries(value).map(([key, item]) => [key, encode(item)]));
  return codecMapping(value) ? { $record: record } : record;
}

/** Return whether `decode` reads the mapping `value` as something else: a `$float` or `$record`. */
function codecMapping(value: Value): boolean {
  if (!isMapping(value)) return false;
  const names = Object.keys(value);
  if (names.length !== 1) return false;
  return names[0] === "$record" || (names[0] === "$float" && ["inf", "-inf"].includes(value["$float"] as string));
}

/**
 * Return a new node holding `value` in case form: a float always with a fraction, and a
 * collection in flow style when `flow` says so or it holds only scalars.
 */
function caseNode(doc: Document, value: Value, flow = false): YamlNode {
  const node = doc.createNode(encode(value)) as YamlNode;
  visit(node, {
    Scalar(_, scalar) {
      if (typeof scalar.value === "number") scalar.minFractionDigits = 1;
    },
    Map(_, map) {
      map.flow = flow || map.items.every((pair) => isScalar(pair.value));
    },
    Seq(_, seq) {
      seq.flow = flow || seq.items.every((item) => isScalar(item));
    },
  });
  return node;
}

/** Return the case value a node holds, as `prepare` reads it; `null` for no node. */
function held(doc: Document, node: unknown): Value {
  return isNode(node) ? decode(node.toJS(doc, { maxAliasCount: -1 }) as Value) : null;
}

/**
 * Return `node` changed to hold `value`, as `same` with `flags` reads it. Parts that already hold
 * their value keep their nodes, so their style and comments stay: a mapping keeps its keys, in
 * order, and a list the longest run of items that match in order. Other parts are new nodes.
 */
function updated(doc: Document, node: unknown, value: Value, flags: SameFlags): unknown {
  if (same(held(doc, node), value, flags)) return node;
  const raw = isNode(node) ? (node.toJS(doc, { maxAliasCount: -1 }) as Value) : null;
  if (isMap(node) && isMapping(value) && !codecMapping(raw) && !codecMapping(value)) {
    const keyOf = (pair: { key: unknown }) => (isScalar(pair.key) ? pair.key.value : pair.key);
    node.items = node.items.filter((pair) => Object.hasOwn(value, keyOf(pair) as string));
    for (const [name, item] of Object.entries(value)) {
      const pair = node.items.find((old) => keyOf(old) === name);
      if (pair === undefined) node.items.push(doc.createPair(name, caseNode(doc, item)));
      else pair.value = updated(doc, pair.value, item, flags);
    }
    return node;
  }
  if (isSeq(node) && Array.isArray(value)) {
    const olds = node.items;
    const values = olds.map((old) => held(doc, old));
    // `kept[i][j]`: how many items of `olds` from `i` match items of `value` from `j`, in order.
    const kept = olds.map(() => new Array<number>(value.length + 1).fill(0));
    kept.push(new Array<number>(value.length + 1).fill(0));
    for (let i = olds.length - 1; i >= 0; i -= 1) {
      for (let j = value.length - 1; j >= 0; j -= 1) {
        kept[i]![j] = same(values[i], value[j], flags)
          ? kept[i + 1]![j + 1]! + 1
          : Math.max(kept[i + 1]![j]!, kept[i]![j + 1]!);
      }
    }
    // A new item is in flow style when an old item is a flow collection holding a collection.
    const flow = olds.some(
      (old) =>
        isCollection(old) &&
        old.flow &&
        old.items.some((inner) => isCollection(isMap(old) ? (inner as { value: unknown }).value : inner)),
    );
    const items: unknown[] = [];
    let gone: unknown[] = [];
    let i = 0;
    let j = 0;
    while (i < olds.length || j < value.length) {
      if (i < olds.length && j < value.length && same(values[i], value[j], flags)) {
        items.push(olds[i]);
        gone = [];
        i += 1;
        j += 1;
      } else if (j < value.length && (i === olds.length || kept[i]![j + 1]! > kept[i + 1]![j]!)) {
        // Between two matches, old items are dropped first, so a new item takes the place of a
        // dropped one, if any, and keeps what of its text still holds.
        const old = gone.shift();
        items.push(old === undefined ? caseNode(doc, value[j]!, flow) : updated(doc, old, value[j]!, flags));
        j += 1;
      } else {
        gone.push(olds[i]);
        i += 1;
      }
    }
    node.items = items;
    return node;
  }
  return caseNode(doc, value);
}

/**
 * Return the `[start, end)` offsets in `text` of its top-level `key` and value: from the start of
 * the key's line to the end of the value's last line. `undefined` when it has no such key.
 */
function section(text: string, key: string): [number, number] | undefined {
  const pair = (parseDocument(text, CASE_YAML).contents as YAMLMap).items.find(
    (item) => isScalar(item.key) && item.key.value === key,
  );
  if (pair === undefined) return undefined;
  const start = text.lastIndexOf("\n", (pair.key as YamlNode).range![0] - 1) + 1;
  const newline = text.indexOf("\n", (pair.value as YamlNode).range![1] - 1);
  return [start, newline === -1 ? text.length : newline + 1];
}

/**
 * Quote each text in a flow collection under `node` that holds a `?`. The writer leaves it bare,
 * as YAML 1.2 allows, but PyYAML ends the text there, so the Python runner couldn't read the case.
 */
function quoteQuestionMarks(node: unknown): void {
  visit(node as Parameters<typeof visit>[0], {
    Collection(_, collection) {
      if (!collection.flow) return;
      const parts = isMap(collection)
        ? collection.items.flatMap((pair) => [pair.key, pair.value])
        : collection.items;
      for (const part of parts) {
        if (isScalar(part) && typeof part.value === "string" && part.value.includes("?")) {
          part.type = "QUOTE_DOUBLE";
        }
      }
    },
  });
}

/**
 * Return the case file `text` with each result in `results` written into `expect` in case form
 * (as `updated` changes it), and `run-log` added to its `rules`; a new `rules` goes right after
 * `kind`. Only the lines of `rules` and `expect` change: they are cut from the whole file written
 * again and put in place of the old ones, because the writer spells some other text (block scalar
 * headers, for one) unlike the case.
 */
function rewriteCase(text: string, results: ReadonlyMap<string, unknown>): string {
  const doc = parseDocument(text, CASE_YAML);
  const options = (indentSeq: boolean) => ({
    lineWidth: 0, indentSeq, flowCollectionPadding: false, singleQuote: true, doubleQuotedAsJSON: true,
  });
  const cut = (written: string, key: string) => {
    const [start, end] = section(written, key)!;
    return written.slice(start, end);
  };
  // Cases indent a list under a key one way or the other; keep the way `expect` has.
  const indentSeq = [false, true].find(
    (indent) => cut(doc.toString(options(indent)), "expect") === cut(text, "expect"),
  );
  if (indentSeq === undefined) throw new CaseFailure("the writer doesn't reproduce its text");
  for (const pair of (doc.get("expect") as YAMLMap).items) {
    const name = (pair.key as { value: string }).value;
    const value = results.get(name) as Value | undefined;
    if (value !== undefined) pair.value = updated(doc, pair.value, value, { byValue: BY_VALUE.has(name) });
  }
  const rules = doc.get("rules");
  if (!isSeq(rules)) {
    const top = doc.contents as YAMLMap;
    const kind = top.items.findIndex((pair) => isScalar(pair.key) && pair.key.value === "kind");
    top.items.splice(kind + 1, 0, doc.createPair("rules", caseNode(doc, ["run-log"])));
  } else if (!rules.items.some((item) => isScalar(item) && item.value === "run-log")) {
    rules.add("run-log");
  }
  quoteQuestionMarks(doc.get("expect"));
  const written = doc.toString(options(indentSeq));
  let result = text;
  for (const key of ["expect", "rules"]) {
    // A new key goes right after `kind`, which is one line.
    const after = result.indexOf("\n", section(result, "kind")![0]) + 1;
    const [start, end] = section(result, key) ?? [after, after];
    result = result.slice(0, start) + cut(written, key) + result.slice(end);
  }
  if (!same(readCase(result), readCase(written))) {
    throw new Error("rewriting the case changed more than its rules and expect");
  }
  return result;
}

// ---- resume ----

// `resume` cases run through `runRun`: `actualRun` cuts or stops their first run and resumes it.

// ---- backward ----

// `backward` cases run through `runRun`: `actualRun` makes their backward calls (`callBackward`).

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

// True when Bun or Node (22.18 and later) runs this file; false under a test runner.
if (import.meta.main) {
  // Not `process.exit()`, which can cut off piped stdout on Node.
  process.exitCode = await main(process.argv.slice(2));
}
