/**
 * Run the language-neutral conformance cases in `tests/conformance/cases/` against TypeScript yian.
 *
 * Each case file is one test, named by its id `<group>/<name>`. `tests/conformance/SPEC.md` defines
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
} from "../../src/compose/helpers/isinstance_types.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { END_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ENTRIES, nonDefaults, read, withEnv } from "../../src/defaults.ts";
import type { Env } from "../../src/defaults.ts";
import {
  compareCodeUnits,
  type Diagnostic,
  Failure,
  type FailureCode,
  OriginRef,
  reprText,
} from "../../src/diagnostics.ts";
import { type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "../../src/events.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/results.ts";
import type { CompleteResult } from "../../src/nodes/base.ts";
import { ProviderError, kindForStatus } from "../../src/llm/errors.ts";
import {
  type ChatModel,
  type Event,
  EventStream,
  Finish,
  type StructuredOutput,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../../src/llm/events.ts";
import {
  Message,
  MessageFormatError,
  type Request,
  Response,
  TextPart,
  ToolCallPart,
  Usage,
} from "../../src/llm/messages.ts";
import type { EmbeddingModel } from "../../src/llm/embeddings.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { VectorDBMemory } from "../../src/memory/vectordb.ts";
import { checkModelConfig, resolveModelConfig } from "../../src/llm_clients/connection.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import {
  constructorNames,
  evalExpr,
  evaluateWhenRecord,
  isinstanceTypeNames,
  ResolveMode,
  resolveInRecord,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr } from "../../src/expr/grammar.ts";
import { evalBinding, renderTemplateRecord } from "../../src/expr/template.ts";
import { typeToSchema } from "../../src/nodes/agent/structured.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { type FileLock, lockFile, openSqlite, virtualScheduler } from "../../src/platform.ts";
import {
  inspectRun,
  replayRunLog,
  type AdvanceOptions,
  advance,
  close,
  complete,
  interrupt,
  type RunInspection,
  type RunFlowOptions,
  type RunResult,
  runFlow,
  send,
  staticNodes,
  withdraw,
} from "../../src/run.ts";
import {
  MemoryRunLog,
  NodeStepRecord,
  readRunLog,
  type RecordEnvelope,
  type RunLog,
  type RunStartedRecord,
} from "../../src/runtime/log.ts";
import { addCrashedAttempts } from "../../src/runtime/state.ts";
import { ASK, pending, type ToolContext } from "../../src/tools/context.ts";
import {
  PermanentToolError,
  ReportedToolError,
  ToolCheckError,
  TransientToolError,
} from "../../src/tools/errors.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { visibleToolName } from "../../src/tools/grants.ts";
import type { AskRequest, AskTool, ToolEntry } from "../../src/tools/registry.ts";
import { canonicalJson, logJson, parseJson } from "../../src/typesys/json.ts";
import type { JsonMode, JsonObject } from "../../src/typesys/json.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  FileRef,
  fromHost,
  InboxRef,
  isMemoryType,
  isRecord,
  jsonPart,
  MemoryRef,
} from "../../src/typesys/values.ts";
import type { Type, Value as YianValue, ValueRecord } from "../../src/typesys/values.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { validate } from "../helpers/schema.ts";
import type { Schema } from "../helpers/schema.ts";

/** A case value: an int is a `bigint`, a float a `number`. */
export type Value = null | boolean | bigint | number | string | Value[] | Case;

/** A case file, as read, and any mapping in it. */
export type Case = { [key: string]: Value };

/** The kinds of case. */
export type Kind =
  | "load"
  | "run"
  | "resume"
  | "expr"
  | "value"
  | "defaults"
  | "fingerprint"
  | "cli"
  | "json";

// The closed format, as tests/conformance/SPEC.md defines it. Sets, and tables keyed by a checked
// `Kind`, so a case value such as `kind: toString` never finds an `Object.prototype` member.
const COMMON: ReadonlySet<string> = new Set(["kind", "pending", "expect"]);
const FLOW: ReadonlySet<string> = new Set(["flow", "files", "env"]);
const RUN: ReadonlySet<string> = new Set([...FLOW, "inputs", "run", "host"]);
const KIND_KEYS = {
  load: new Set([...FLOW, "memory_types"]),
  run: RUN,
  resume: new Set([...RUN, "resume"]),
  expr: new Set(["expr", "template", "values", "item", "mode", "typedefs", "env"]),
  value: new Set(["type", "typedefs", "value", "env"]),
  defaults: new Set(["env"]),
  fingerprint: new Set(["flow", "files", "env", "other"]),
  cli: new Set([...FLOW, "args", "stdin", "signal", "http"]),
  json: new Set(["read", "mode", "write", "form"]),
} satisfies Record<Kind, ReadonlySet<string>>;
const REQUIRED: Partial<Record<Kind, readonly string[]>> = {
  load: ["flow"], run: ["flow"], resume: ["flow"],
  value: ["type", "value"], fingerprint: ["flow", "other"],
};
// Each kind's `expect` holds exactly one of its result groups, and any of its optional keys.
const RESULTS = {
  load: [["load"]],
  run: [["raises"], ["output"]],
  resume: [["raises"], ["output"]],
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
  resume: new Set([...RUN_OPTIONAL, "memory_drift"]),
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
  "run_id", "num_workers", "llm_config", "context",
]);
const RESUME_KEYS: ReadonlySet<string> = new Set(["num_workers", "api_key", "headers", "context", "memory"]);
// Host keys the format reserves and this runner doesn't support yet.
const RESERVED_HOST_KEYS: readonly string[] = ["skills"];
const HOST_KEYS: ReadonlySet<string> = new Set([
  "tools", "human", "jitter", "models", "stop", "sends", "memory", ...RESERVED_HOST_KEYS,
]);
const RESUME_HOST_KEYS: ReadonlySet<string> = new Set([...HOST_KEYS, "crash_after_record"]);
/** The `[section, key]` pairs a case may not use yet: it fails before it is set up. */
const RESERVED: readonly (readonly [string, string])[] = RESERVED_HOST_KEYS.map(
  (key) => ["host", key] as const,
);
// `host.models` consumers.
const CONSUMERS: readonly string[] = ["agent", "structured_retry", "compose", "summary"];
const USAGE_COUNTS: readonly string[] = ["input_tokens", "output_tokens", "total_tokens"];
const USAGE_DETAILS: readonly string[] = [
  "cache_read_tokens", "cache_creation_tokens", "reasoning_tokens",
];
// Keys holding text, or plain JSON (`http`), that is not a case value: no codec form, no number
// checks.
const TEXT_KEYS: ReadonlySet<string> = new Set([
  "flow", "other", "files", "stdin", "read", "http",
]);
const SECRET_KEYS: ReadonlySet<string> = new Set([
  "run", "resume", "env", "host", "stdin", "http",
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

/** Return whether `value` counts as true: not null, false, zero, empty text, or an empty list or mapping. */
function truthy(value: Value | undefined): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  // NaN counts as true.
  if (typeof value === "number") return value !== 0;
  return Boolean(value);
}

/** Return whether `value` is zero: 0, 0.0 and false all count. */
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
  // file has none. They are refused before any key is checked.
  visit(doc, {
    Node(_, node) {
      if (node.tag !== undefined) {
        throw new Error(`a case file has no tags, found ${spell(node.tag)}`);
      }
    },
  });
  // Keys are checked level by level, mappings in document order.
  const nodes: unknown[] = [doc.contents];
  for (const node of nodes) {
    if (isSeq(node)) nodes.push(...node.items);
    else if (isMap(node)) checkKeys(doc, node, (value) => nodes.push(value));
  }
  // Case files are trusted, so aliases have no limit.
  const value = doc.toJS({ maxAliasCount: -1 }) as Value;
  if (!isMapping(value)) throw new Error("a case file holds one mapping");
  return value;
}

/** Throw unless every key of `map` is text and appears once; hand each value to `next`. */
function checkKeys(doc: Document, map: YAMLMap, next: (value: unknown) => void): void {
  const seen = new Set<string>();
  for (const pair of map.items) {
    // A collection key is checked in full first, so a bad key inside it is reported first.
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
      if (!CONSUMERS.includes(consumer)) {
        throw new Error(`\`${label}\`: the consumer must be one of ${CONSUMERS.join(", ")}`);
      }
      if (!Array.isArray(script)) throw new Error(`\`${label}\` must be a list`);
      script.forEach((entry, index) => {
        checkModelEntry(`${label}[${index}]`, entry);
      });
    }
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

/**
 * Throw unless `value`, at `where`, is a point: `{model_call: n, chunk: k}`, `{model_call: n}` or
 * `{tool_call: i}`, counted from 1 (`chunk` from 0).
 */
function checkPoint(where: string, value: Value | undefined): void {
  const point = mapping(where, value);
  const names = Object.keys(point).sort().join(",");
  if (!["chunk,model_call", "model_call", "tool_call"].includes(names)) {
    throw new Error(`\`${where}\` is \`{model_call, chunk}\`, \`{model_call}\` or \`{tool_call}\``);
  }
  const counted = ([name, count]: [string, Value]) =>
    typeof count === "bigint" && count >= (name === "chunk" ? 0n : 1n);
  if (!Object.entries(point).every(counted)) {
    throw new Error(`\`${where}\` counts must be ints from 1, \`chunk\` from 0`);
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
 * Throw unless `value`, a `host.sends` list, holds only `{inbox, event}` items, optionally with
 * `after: {node, <point>}` (sent live when the run reaches the point) or, in a `resume` case,
 * `down: true` (sent after the crash, before the resume), `{close}` items, `{withdraw}` items
 * (`withdraw(run, inbox)`) and `{interrupt}` items (`interrupt(run, inbox)`), each optionally
 * with `after`, and `{complete, output}` or `{complete, error}` items.
 */
function checkSends(value: Value | undefined, kind: Kind): void {
  if (!Array.isArray(value)) throw new Error("`host.sends` must be a list");
  value.forEach((item, index) => {
    // `hold: true` is only for an item sent while the run waits, so not with `after` or `down`.
    const holds = isMapping(item) && Object.hasOwn(item, "hold");
    const held = holds && (item as Case)["hold"] === true && !Object.hasOwn(item, "after") && !Object.hasOwn(item, "down");
    const fields = isMapping(item)
      ? Object.keys(item)
          .filter((key) => key !== "hold")
          .sort(compareCodeUnits)
          .join(",")
      : null;
    const send = typeof (item as Case | null)?.["inbox"] === "string";
    const completes = typeof (item as Case | null)?.["complete"] === "string";
    const interrupts = typeof (item as Case | null)?.["interrupt"] === "string";
    const withdraws = typeof (item as Case | null)?.["withdraw"] === "string";
    const valid =
      (!holds || held) &&
      ((fields === "event,inbox" && send) ||
      (fields === "after,event,inbox" && send && isMapping((item as Case)["after"])) ||
      (fields === "down,event,inbox" && send && (item as Case)["down"] === true && kind === "resume") ||
      (fields === "close" && typeof (item as Case)["close"] === "string") ||
      (fields === "withdraw" && withdraws) ||
      (fields === "after,withdraw" && withdraws && isMapping((item as Case)["after"])) ||
      (fields === "interrupt" && interrupts) ||
      (fields === "after,interrupt" && interrupts && isMapping((item as Case)["after"])) ||
      (fields === "complete,output" && completes) ||
      (fields === "complete,error" && completes && typeof (item as Case)["error"] === "string"));
    if (!valid) {
      throw new Error(
        "`host.sends` items are {inbox, event, after?|down?}, {close}, {withdraw, after?}, {interrupt, after?} or {complete, output | error}; " +
          "`hold: true` marks an item sent while the run waits that isn't followed by an advance",
      );
    }
    if (fields === "after,event,inbox" || fields === "after,interrupt" || fields === "after,withdraw") {
      const { node, ...point } = (item as Case)["after"] as Case;
      if (typeof node !== "string") throw new Error(`\`host.sends[${index}].after.node\` must be a node id`);
      checkPoint(`host.sends[${index}].after`, point);
    }
  });
}

/** The answer a `host.sends` `{complete}` item gives: its `output`, or its `error`. */
function completion(item: Case): CompleteResult {
  return Object.hasOwn(item, "output") ? { output: item["output"] } : { error: item["error"] as string };
}

/** Throw unless `entry`, the stub tool at `where`, uses only the format's keys and scripts. */
function checkStub(where: string, entry: Value | undefined): void {
  const toolKeys = new Set(["parameters", "description", "concurrency_safe", "resumable", "results", "pending"]);
  const tool = keys(where, entry, toolKeys, { required: ["parameters"] });
  for (const flag of ["concurrency_safe", "resumable"]) {
    if (Object.hasOwn(tool, flag) && typeof tool[flag] !== "boolean") {
      throw new Error(`\`${where}.${flag}\` must be a boolean`);
    }
  }
  const waits = Object.hasOwn(tool, "pending");
  if (Object.hasOwn(tool, "results") === waits || (waits && tool["pending"] !== true)) {
    throw new Error(`\`${where}\` takes \`results\` or \`pending: true\``);
  }
  if (waits) return;
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

/** Throw unless `files`, at `where`, maps relative paths without `..` to text. */
function checkFiles(where: string, files: Value | undefined): void {
  for (const [path, text] of Object.entries(mapping(where, files))) {
    if (typeof text !== "string") throw new Error(`\`${where}\` values must be text`);
    if (path.startsWith("/") || path.split("/").includes("..")) {
      throw new Error(`\`${where}\`: \`${path}\` must be a relative path without \`..\``);
    }
  }
}

/** The keys a `host.memory` entry of each memory type takes. */
const MEMORY_KEYS: Readonly<Record<string, ReadonlySet<string>>> = {
  FileMemory: new Set(["type", "id", "files", "held"]),
  VectorDBMemory: new Set(["type", "id", "embedding", "items", "recall_recent"]),
};

/**
 * Throw unless `value` is `host.memory`: input names to `{type: FileMemory, id, files, held}`
 * (`held` a point with its `node`) or `{type: VectorDBMemory, id, embedding, items,
 * recall_recent}`, or to `{same_as: <input>}` naming an entry that isn't one.
 */
function checkMemory(value: Value | undefined): void {
  const entries = mapping("host.memory", value);
  for (const [input, entry] of Object.entries(entries)) {
    const where = `host.memory.${input}`;
    if (isMapping(entry) && Object.hasOwn(entry, "same_as")) {
      const same = keys(where, entry, new Set(["same_as"]))["same_as"];
      const named = typeof same === "string" && same !== input ? get(entries, same) : undefined;
      if (!isMapping(named) || Object.hasOwn(named, "same_as")) {
        throw new Error(`\`${where}.same_as\` must name another \`host.memory\` entry that isn't a \`same_as\``);
      }
      continue;
    }
    const type = get(mapping(where, entry), "type");
    if (type === undefined) throw new Error(`\`${where}.type\` is required`);
    if (typeof type !== "string" || !Object.hasOwn(MEMORY_KEYS, type)) {
      throw new Error(`\`${where}.type\` must be FileMemory or VectorDBMemory`);
    }
    const memory = keys(where, entry, MEMORY_KEYS[type]!, { required: type === "VectorDBMemory" ? ["embedding"] : [] });
    if (typeof get(memory, "id", "") !== "string") throw new Error(`\`${where}.id\` must be text`);
    if (type === "VectorDBMemory") {
      checkVectorEntry(where, memory);
      continue;
    }
    checkFiles(`${where}.files`, get(memory, "files", {}));
    if (Object.hasOwn(memory, "held")) {
      const { node, ...point } = mapping(`${where}.held`, memory["held"]);
      if (typeof node !== "string") throw new Error(`\`${where}.held.node\` must be a node id`);
      checkPoint(`${where}.held`, point);
    }
  }
}

/** Throw unless the VectorDBMemory entry `memory`, at `where`, holds an embedding model, items and a recall count. */
function checkVectorEntry(where: string, memory: Case): void {
  const embedding = memory["embedding"];
  const texts = (keys: string[]) => isMapping(embedding) && keys.every((key) => typeof embedding[key] === "string");
  if (!texts(["provider", "model"]) || Object.keys(embedding as Case).length !== 2) {
    throw new Error(`\`${where}.embedding\` must be \`{provider, model}\`, each text`);
  }
  const items = get(memory, "items", []);
  const isItem = (item: Value) =>
    isMapping(item) &&
    typeof item["text"] === "string" &&
    Object.keys(item).every((key) => key === "text" || (key === "fields" && isMapping(item["fields"])));
  if (!Array.isArray(items) || !items.every(isItem)) {
    const shape = "a list of `{text, fields}`, each `text` text and `fields` a mapping";
    throw new Error(`\`${where}.items\` must be ${shape}`);
  }
  const recent = get(memory, "recall_recent", 0n);
  if (typeof recent !== "bigint" || recent < 0n) {
    throw new Error(`\`${where}.recall_recent\` must be an int of 0 or more`);
  }
}

/**
 * Throw unless `value`, the host, its tools, models and their scripts use only the format's keys.
 */
function checkHost(value: Value | undefined, kind: Kind): void {
  const allowed = kind === "resume" ? RESUME_HOST_KEYS : HOST_KEYS;
  const host = keys("host", value, allowed, { known: RESUME_HOST_KEYS, kind });
  const tools = mapping("host.tools", get(host, "tools", {}));
  // Each earlier group's stub tools, which a later group's `same_as` may name.
  const stubs = new Set<string>();
  for (const [name, entry] of Object.entries(tools)) {
    const where = `host.tools.${name}`;
    if (!isMapping(entry) || !Object.hasOwn(entry, "group")) {
      checkStub(where, entry);
      continue;
    }
    const group = keys(where, entry, new Set(["group"]))["group"];
    if (!isMapping(group)) throw new Error(`\`${where}.group\` must be a mapping of stub tools`);
    for (const [tool, member] of Object.entries(group)) {
      const at = `${where}.group.${tool}`;
      if (!isMapping(member) || !Object.hasOwn(member, "same_as")) {
        checkStub(at, member);
        continue;
      }
      const same = keys(at, member, new Set(["same_as"]))["same_as"];
      if (typeof same !== "string" || !stubs.has(same)) {
        throw new Error(`\`${at}.same_as\` must name an earlier group's stub tool as \`<group>.<tool>\``);
      }
    }
    for (const [tool, member] of Object.entries(group)) {
      if (!Object.hasOwn(member as Case, "same_as")) stubs.add(`${name}.${tool}`);
    }
  }
  checkModels(get(host, "models", {}));
  checkMemory(get(host, "memory", {}));
  checkStop(get(host, "stop", {}));
  checkSends(get(host, "sends", []), kind);
  const human = mapping("host.human", get(host, "human", {}));
  if (Object.values(human).some((answers) => !Array.isArray(answers))) {
    throw new Error("`host.human` maps node ids to lists of answers");
  }
  if (Object.hasOwn(host, "human") && (Object.hasOwn(tools, "ask") || Object.hasOwn(tools, "ask_user"))) {
    throw new Error("a case with `host.human` declares no `ask` or `ask_user` tool");
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
    // Finite: `setTimeout` runs an infinite delay after 1 ms, so the case would not wait.
    const finite = typeof delay === "number" && delay >= 0 && delay < Infinity;
    if (!finite && !(typeof delay === "bigint" && delay >= 0n)) {
      throw new Error(`\`${where}.response.delay\` must be a finite non-negative number`);
    }
  });
}

/** Return the memory type of `host.memory` entry `input` in `memory`, through its `same_as`. */
function memoryType(memory: Case, input: string): Value | undefined {
  const entry = memory[input] as Case;
  const same = get(entry, "same_as");
  return get((typeof same === "string" ? memory[same] : entry) as Case, "type");
}

/**
 * Throw unless the case's memories are used as the format says: no `inputs` key binds a
 * `host.memory` input; `expect.memory` maps `host.memory` inputs to null or, for a FileMemory, a
 * mapping of paths to text and, for a VectorDBMemory, a list of items; `expect.memory_drift`
 * lists input names; `resume.memory` maps input names to null or `{id}`.
 */
function checkMemoryUse(case_: Case, expect: Case): void {
  const memory = get(get(case_, "host", {}) as Case, "memory", {}) as Case;
  const inputs = get(case_, "inputs", {});
  for (const input of Object.keys(memory)) {
    if (isMapping(inputs) && Object.hasOwn(inputs, input)) throw new Error(`\`inputs.${input}\`: \`host.memory\` binds it`);
  }
  for (const [input, files] of Object.entries(mapping("expect.memory", get(expect, "memory", {})))) {
    const where = `expect.memory.${input}`;
    if (!Object.hasOwn(memory, input)) throw new Error(`\`${where}\` names no \`host.memory\` input`);
    if (files === null) continue;
    if (memoryType(memory, input) === "VectorDBMemory") {
      const fields = new Set(["id", "text", "fields", "retired"]);
      const isItem = (item: Value) =>
        isMapping(item) &&
        typeof item["id"] === "bigint" &&
        typeof item["text"] === "string" &&
        Object.keys(item).every((key) => fields.has(key)) &&
        (!Object.hasOwn(item, "retired") || typeof item["retired"] === "string");
      if (!Array.isArray(files) || !files.every(isItem)) {
        throw new Error(`\`${where}\` must be null or a list of items \`{id, text, fields, retired}\``);
      }
      continue;
    }
    if (!isMapping(files) || Object.values(files).some((text) => typeof text !== "string")) {
      throw new Error(`\`${where}\` must be null or a mapping of paths to text`);
    }
    checkFiles(where, files);
  }
  const drift = get(expect, "memory_drift", []);
  if (!Array.isArray(drift) || drift.some((input) => typeof input !== "string")) {
    throw new Error("`expect.memory_drift` must be a list of input names");
  }
  const resume = get(case_, "resume", {}) as Case;
  for (const [input, binding] of Object.entries(mapping("resume.memory", get(resume, "memory", {})))) {
    const bound = isMapping(binding) && Object.keys(binding).join() === "id" && typeof binding["id"] === "string";
    if (binding !== null && !bound) throw new Error(`\`resume.memory.${input}\` must be null or \`{id}\``);
  }
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
  const expect = keys("expect", case_["expect"], expectKeys(kind), {
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
    throw new Error(`\`expect\` needs ${needs}`);
  }
  const args = get(case_, "args", []);
  if (!Array.isArray(args) || args.some((item) => typeof item !== "string")) {
    throw new Error("`args` must be a list of text");
  }
  for (const key of ["env", "files"]) {
    if (Object.values(mapping(key, get(case_, key, {}))).some((text) => typeof text !== "string")) {
      throw new Error(`\`${key}\` values must be text`);
    }
  }
  for (const key of TEXT_KEYS) {
    if (["files", "http"].includes(key)) continue;
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
  const paths = mapping("expect.exists", get(expect, "exists", {}));
  for (const [path, exists] of Object.entries(paths)) {
    if (typeof exists !== "boolean") {
      throw new Error("`expect.exists` values must be true or false");
    }
    if (path.startsWith("/") || path.split("/").includes("..")) {
      throw new Error(`\`expect.exists\`: \`${path}\` must be a relative path without \`..\``);
    }
  }
  if (Object.hasOwn(case_, "host")) checkHost(case_["host"], kind);
  checkMemoryUse(case_, expect);
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
    // The message names the float nan, inf or -inf.
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

/** Return the first directory above this file holding `tests/conformance/cases`: the repository. */
function findRoot(): string {
  // `tests/conformance/` from source, `dist/tests/conformance/` when built.
  let dir = dirname(fileURLToPath(import.meta.url));
  while (!existsSync(join(dir, "tests", "conformance", "cases"))) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error("no `tests/conformance/cases` above the conformance runner");
    dir = parent;
  }
  return dir;
}

const CASES = join(findRoot(), "tests", "conformance", "cases");

/** The runner's own "this case failed" error, carrying only a message. */
export class CaseFailure extends Error {
  override name = "CaseFailure";
}

/** Return `value` as a message shows it: text as is, else spelled. */
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
 * Return `error` in the case's `raises` form: `{class, message}`, plus `path` for a
 * `TypeCheckError`. The class is `error.name`, so every yian error class sets
 * `override name = "<Class>"`. A thrown non-`Error` is named by its `typeof`.
 */
export function raised(error: unknown): Case {
  if (!(error instanceof Error)) return { class: typeof error, message: String(error) };
  const form: Case = { class: error.name, message: error.message };
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
  /** Whether `--write-logs` asks a `run` or `resume` case to write its golden log. */
  writeLogs?: boolean;
  /**
   * With `--write-expect`, where a `run` or `resume` case puts each `REWRITTEN` result
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
  expr: runExpr,
  value: runValue,
  defaults: runDefaults,
  fingerprint: runFingerprint,
  cli: runCli,
  json: runJson,
};

/**
 * Make the case directory under the OS temp directory: `flow.yaml`, the `files`, and a copy of
 * `fixturesDir` as `fixtures/` when it is a directory. Except for `cli`, the environment is
 * `outer` (the process's) without yian's own variables, plus `YIAN_AUTH_FILE` naming a missing
 * file in the directory, plus the case's `env`; for `cli` it is the case's `env` alone.
 */
export function setUp(
  case_: Case,
  fixturesDir: string,
  outer: Readonly<Record<string, string | undefined>> = process.env,
): CaseContext {
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
  const inherited = Object.entries(outer).filter(
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
  "usage: conformance [--group GROUP] [--kind KIND] [--case GROUP/NAME] " +
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
 * which needs `--case`, writes each selected `run` and `resume` case's golden log
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
  let flags: Partial<Record<"group" | "kind" | "case" | "diagnostic-code", string[]>> & {
    "crash-sweep"?: boolean;
    "write-logs"?: boolean;
    "write-expect"?: boolean;
  };
  try {
    flags = parseArgs({
      args: [...argv],
      strict: true,
      options: {
        group: option, kind: option, case: option, "diagnostic-code": option,
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
      // `--kind` and `--diagnostic-code` never hide a broken file; `--group` and
      // `--case`, read from its path, still select or skip it.
      selected.push({ id, unreadable: error instanceof Error ? error.message : String(error) });
      continue;
    }
    const codes = diagnosticCodes(case_);
    if (
      allows(flags.kind, (wanted) => wanted === case_["kind"]) &&
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
 * and CODE references in the case directory, with the host memory classes `memoryTypes`.
 */
async function loadText(
  text: string,
  ctx: CaseContext,
  memoryTypes: readonly (typeof MemoryBase)[] = [],
): Promise<ComposedFlow> {
  // Scoped to the load's own async work, not installed, so a load still running after its test
  // timed out can't change what a later test reads.
  const options = { searchPaths: [ctx.dir], sourceLabel: "flow.yaml", baseDir: ctx.dir, memoryTypes };
  return await withEnv(caseEnv(ctx), () => loadFlow(text, options));
}

/**
 * Return a memory class per `memory_types` entry: a contract of that type whose read tools have
 * the names listed, in order. A load reads only the contract, so the classes never open.
 */
function memoryClasses(entries: Case): (typeof MemoryBase)[] {
  return Object.entries(mapping("memory_types", entries)).map(([type, tools]) => {
    const contract: MemoryContract = {
      type,
      tools: (tools as string[]).map((name) => ({ name, write: false, description: name, parameters: { type: "object" } })),
    };
    return class extends MemoryBase {
      static override readonly contract = contract;
      override open(): Promise<MemorySession> {
        throw new Error("a load case never opens a memory");
      }
      override async reset(): Promise<void> {}
    };
  });
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
      await loadText(case_["flow"] as string, ctx, memoryClasses(get(case_, "memory_types", {}) as Case));
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
  llm_config: "llmConfig",
  context: "context",
};

/** The `advance` option each `resume` key sets. */
const RESUME_OPTIONS: Readonly<Record<string, keyof AdvanceOptions>> = {
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

/** A point of the run a `host.stop` or `host.sends` item waits for: its `after`. */
type Point = { model_call?: bigint; chunk?: bigint; tool_call?: bigint };

/** The error the runner stops a run with at its `host.stop` point. */
class ConformanceStop extends Error {
  override name = "ConformanceStop";

  constructor() {
    super("conformance stop");
  }
}

/**
 * The run points an AGENT reaches: the `host.stop` points, the `host.sends` items delivered at
 * a point (`after`), a send, a withdraw or an interrupt, and the `host.memory` locks taken at a
 * point (`held`).
 *
 * Reaching a stop calls `onStop`; reaching a send calls `deliver`, a withdraw `withdraw`, an
 * interrupt `interrupt`, a lock `hold`. Calls are counted, and points reached, only once `armed`,
 * so that runs before the main one neither count nor act.
 */
class Inbox {
  armed = false;
  readonly #counts = new Map<string, number>();
  readonly #pending: { label: string; node: string; action: Case }[] = [];
  readonly #onStop: () => void;
  readonly #deliver: (inbox: string, event: Value) => void;
  readonly #interrupt: (inbox: string) => void;
  readonly #withdraw: (inbox: string) => void;
  readonly #hold: (input: string) => void;

  constructor(
    stop: Case = {},
    onStop: () => void = () => {},
    sends: readonly Case[] = [],
    deliver: (inbox: string, event: Value) => void = () => {},
    interrupt: (inbox: string) => void = () => {},
    withdraw: (inbox: string) => void = () => {},
    memory: Case = {},
    hold: (input: string) => void = () => {},
  ) {
    this.#onStop = onStop;
    this.#deliver = deliver;
    this.#interrupt = interrupt;
    this.#withdraw = withdraw;
    this.#hold = hold;
    for (const [node, action] of Object.entries(stop)) {
      this.#pending.push({ label: `host.stop.${node}`, node, action: { ...(action as Case), stop: true } });
    }
    for (const [input, entry] of Object.entries(memory)) {
      if (!Object.hasOwn(entry as Case, "held")) continue;
      const { node, ...point } = (entry as Case)["held"] as Case;
      this.#pending.push({ label: `host.memory.${input}.held`, node: node as string, action: { after: point, hold: input } });
    }
    sends.forEach((item, index) => {
      if (!Object.hasOwn(item, "after")) return;
      const { node, ...point } = item["after"] as Case;
      const action: Case = Object.hasOwn(item, "interrupt")
        ? { after: point, interrupt: item["interrupt"]! }
        : Object.hasOwn(item, "withdraw")
          ? { after: point, withdraw: item["withdraw"]! }
          : { after: point, send: { inbox: item["inbox"]!, event: item["event"]! } };
      this.#pending.push({ label: `host.sends[${index}]`, node: node as string, action });
    });
  }

  /**
   * Send `action`'s event, withdraw its inbox's unread sends, interrupt its readers, take its
   * memory's lock, or stop the run.
   */
  #act(action: Case): void {
    if (action["stop"] === true) {
      this.#onStop();
      return;
    }
    if (Object.hasOwn(action, "hold")) {
      this.#hold(action["hold"] as string);
      return;
    }
    if (Object.hasOwn(action, "send")) {
      const { inbox, event } = action["send"] as Case;
      this.#deliver(inbox as string, event as Value);
      return;
    }
    if (Object.hasOwn(action, "withdraw")) {
      this.#withdraw(action["withdraw"] as string);
      return;
    }
    this.#interrupt(action["interrupt"] as string);
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
      this.#act(item.action);
    }
  }

  /** Return a message per action whose point was never reached. */
  left(): string[] {
    return this.#pending.map(({ label }) => `\`${label}\` was never reached`);
  }
}

/**
 * Yield `events`, a scripted call's, reaching its `inbox` points at `[node, number]`, its model
 * call: chunk 0 as its first event is asked for, before it is sent; chunk `k` once its events are
 * delivered (a `reply`, whose events are one chunk, after all of them; a `chunks` entry's, one
 * event each), then the end of the call when it doesn't raise.
 */
async function* reaching(
  events: EventStream,
  reply: boolean,
  inbox: Inbox,
  [node, number]: [string, bigint],
): AsyncGenerator<Event> {
  let chunk = 0n;
  inbox.reached(node, { model_call: number, chunk });
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

/** A host problem `check()` reports: its text, and for a call past a script's end, the script's label. */
interface Problem {
  text: string;
  overrun?: string;
}

/**
 * Return the scripted `request` with each tool result for a call in `calls` read as the closer a
 * resume writes for a call cut off before its result.
 */
function closedCrashed(request: Value, calls: ReadonlySet<string> | undefined): Value {
  if (calls === undefined) return request;
  const messages = ((request as Case)["messages"] as Case[]).map((message) => {
    if (message["role"] !== "tool") return message;
    const parts = (message["parts"] as Case[]).map((part) =>
      part["type"] === "tool_result" && calls.has(part["call_id"] as string)
        ? { ...part, content: TOOL_OUTCOME_UNKNOWN, is_error: true }
        : part,
    );
    return { ...message, parts };
  });
  return { ...(request as Case), messages };
}

/**
 * Return a builder of chat models that serve each call the next `host.models` entry of its node
 * and consumer, taken through `take` when the call is made: the call's own consumer
 * (`consumerOf`). A call's request that
 * differs from its entry's `request` is added to `problems`, naming the entry by its index,
 * `taken` entries of its script before it; for a node with calls in `crashed`, the entry's request
 * reads each such call's result as the `crashed` closer (`closedCrashed`). A call `take` refuses fails at its
 * stream's first read, as a provider's failure does. An `agent` call reaches its `inbox` points
 * (`reaching`).
 */
function scriptedModels(
  take: (label: string, what: string) => Value,
  problems: Problem[],
  inbox: Inbox,
  taken: (label: string) => number,
  crashed: ReadonlyMap<string, ReadonlySet<string>>,
): (provider: string, model: string, nodeId: string, structuredOutput: StructuredOutput) => ChatModel {
  return (provider, model, nodeId, structuredOutput) => ({
    provider,
    model,
    structuredOutput,
    stream: (request, options = {}) => {
      const consumer = consumerOf(nodeId, request);
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
      // Counted from what is still taken, so an entry put back is named by its place again.
      const index = taken(label) - 1;
      if (Object.hasOwn(entry, "request")) {
        const what = `\`${label}[${index}].request\``;
        // By value: TS holds a JSON number as a JS `number` whether it is written `1` or `1.0`.
        const expected = closedCrashed(entry["request"]!, crashed.get(nodeId));
        const text = difference(what, v1Request(request), expected, { byValue: true });
        if (text) problems.push({ text });
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

/** The inputs of yian's own `ask` surface; any other input of a TOOL `ask` is a host extra. */
const ASK_SURFACE: ReadonlySet<string> = new Set(["prompt", "questions", "timeout", "__questions"]);

/**
 * Return an ask request as a case writes its `args`: snake_case keys, each question's too, null
 * fields left out, and the host's extras flattened in. The call id is left out: the call record
 * already keeps it.
 */
function askUserArgs(request: AskRequest): Case {
  const args: Case = {};
  for (const [key, value] of Object.entries(request)) {
    if (value === null || key === "callId") continue;
    if (key === "extra") {
      for (const [name, item] of Object.entries(value as Case)) args[name] = item;
      continue;
    }
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

/** The memories a case's host binds, each in a folder of its own under one directory. */
interface CaseMemories {
  /** Each `host.memory` input's folder, by input name; a `same_as` input's is its entry's. */
  readonly roots: ReadonlyMap<string, string>;
  /** Return `input`'s folder with symlinks resolved, as FileMemory resolves it. */
  real(input: string): string;
  /** Return a new instance per entry on its folder, by input name; a `same_as` input shares its entry's. */
  bind(): Record<string, MemoryBase>;
  /** Return a new instance of `input`'s entry in the folder `root`, with identity `id`. */
  make(input: string, root: string, id: string | undefined): MemoryBase;
  /** Return the JSON Schema of `input`'s item type, which a run opens its memory with, or `null`. */
  itemType(input: string): JsonObject | null;
  /**
   * Return what `expect.memory` compares for `input`: a FileMemory's files, or a VectorDBMemory's
   * items; `null` when there is no folder or store.
   */
  contents(input: string): Value;
}

/** Write `files`, relative paths to text, under `root`: folders 0700, files 0600. */
function writeFolder(root: string, files: Readonly<Record<string, string>>): void {
  mkdirSync(root, { recursive: true, mode: 0o700 });
  for (const [name, text] of Object.entries(files)) {
    const path = join(root, name);
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
    writeFileSync(path, text, { mode: 0o600 });
  }
}

/** The words of the runner's embedder, one dimension each. */
const EMBED_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"];

/** The runner's embedder: each text's vector counts the words `alpha` to `hotel` in it, one dimension each. */
async function wordCounts(texts: readonly string[]): Promise<Float32Array[]> {
  return texts.map((text) => {
    const words = text.toLowerCase().split(/[^a-z]+/);
    return Float32Array.from(EMBED_WORDS.map((word) => words.filter((each) => each === word).length));
  });
}

/** The file a VectorDBMemory entry's store is, in its folder. */
const STORE_FILE = "store.db";

/**
 * Return a new instance of the `host.memory` entry `entry`'s type in the folder `root`, with
 * identity `id`, its defaults read from `env`, as the run's are: a FileMemory on the folder, or a
 * VectorDBMemory on its store, embedded by the runner's embedder.
 */
function entryMemory(entry: Case, root: string, id: string | undefined, env: Env): MemoryBase {
  // A virtual clock of its own: its one sleep is a lock wait, which then takes no time.
  const scheduler = virtualScheduler(Date.now());
  const named = id === undefined ? {} : { id };
  return withEnv(env, () => {
    if (entry["type"] === "FileMemory") return new FileMemory({ root, ...named, scheduler });
    return new VectorDBMemory({
      path: join(root, STORE_FILE),
      ...named,
      embedding: entry["embedding"] as unknown as EmbeddingModel,
      embed: wordCounts,
      recallRecent: Number(get(entry, "recall_recent", 0n)),
      scheduler,
    });
  });
}

/** Return the JSON Schema of `flow`'s memory input `input`'s item type, as a run computes it, or `null`. */
function flowItemType(flow: ComposedFlow, input: string): JsonObject | null {
  const typ = flow.input.find((param) => param.name === input)?.contract?.type;
  return isMemoryType(typ) && typ.element !== undefined ? typeToSchema(typ.element) : null;
}

/** Make `memory`'s store, typed `itemType`, and add `items` to it in order, as a host: ids 1, 2, ... */
async function seedItems(memory: MemoryBase, itemType: JsonObject | null, items: readonly Case[]): Promise<void> {
  const signal = new AbortController().signal;
  const session = await memory.open({ create: true, itemType, signal });
  try {
    for (const [i, item] of items.entries()) {
      const change: JsonObject = { op: "add", text: item["text"] as string };
      if (Object.hasOwn(item, "fields")) change["fields"] = item["fields"] as JsonObject;
      const author = { kind: "host", name: "conformance" } as const;
      const applied = await session.apply({ id: `seed-${i + 1}`, change, author, signal });
      if (!applied.ok) throw new Error(`\`host.memory\` item ${i + 1} was refused: ${applied.reason}`);
    }
  } finally {
    await session.close();
  }
}

/**
 * Return the items of the VectorDBMemory store at `path`, in id order, each `{id, text, fields,
 * retired}` with `fields` and `retired` left out when it has none, or `null` when there is no store.
 */
function storeItems(path: string): Value {
  if (!existsSync(path)) return null;
  const db = openSqlite(path, { readonly: true });
  try {
    return db
      .prepare("SELECT id, text, fields, retired FROM items ORDER BY id")
      .all()
      .map((row) => {
        const item: Case = { id: BigInt(Number(row.id)), text: row.text as string };
        if (row.fields !== null) item["fields"] = parseJson(row.fields as string, "codec") as Value;
        if (row.retired !== null) item["retired"] = row.retired as string;
        return item;
      });
  } finally {
    db.close();
  }
}

/**
 * Return the memories `host.memory` in `spec` declares for `flow`, each entry's folder
 * `<dir>/<input>` holding its `files` (no folder without them), or its store `store.db` holding its
 * `items`, added with the input's item type (no store without them); its instances built with `env`.
 */
async function caseMemories(spec: Case, dir: string, env: Env, flow: ComposedFlow): Promise<CaseMemories> {
  const entries = get(spec, "memory", {}) as Case;
  const own = (input: string) => {
    const same = (entries[input] as Case | undefined)?.["same_as"];
    return typeof same === "string" ? same : input;
  };
  // A `resume.memory` input may have no entry (an Optional memory left out): it binds a FileMemory.
  const entry = (input: string) => (entries[own(input)] as Case | undefined) ?? { type: "FileMemory" };
  const roots = new Map(Object.keys(entries).map((input) => [input, join(dir, own(input))]));
  const make = (input: string, root: string, id: string | undefined) => entryMemory(entry(input), root, id, env);
  const itemType = (input: string) => flowItemType(flow, input);
  const idOf = (input: string) => get(entry(input), "id") as string | undefined;
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  for (const input of Object.keys(entries)) {
    if (own(input) !== input) continue;
    const { files, items } = entry(input);
    if (isMapping(files)) writeFolder(roots.get(input)!, files as Record<string, string>);
    if (Array.isArray(items)) {
      await seedItems(make(input, roots.get(input)!, idOf(input)), itemType(input), items as Case[]);
    }
  }
  const real = realpathSync(dir);
  return {
    roots,
    real: (input) => join(real, own(input)),
    bind: () => {
      const made = new Map<string, MemoryBase>();
      for (const input of Object.keys(entries)) {
        const name = own(input);
        if (!made.has(name)) made.set(name, make(name, roots.get(name)!, idOf(name)));
      }
      return Object.fromEntries(Object.keys(entries).map((input) => [input, made.get(own(input))!]));
    },
    make,
    itemType,
    contents: (input) =>
      entry(input)["type"] === "VectorDBMemory"
        ? storeItems(join(roots.get(input)!, STORE_FILE))
        : folderFiles(roots.get(input)!),
  };
}

/**
 * Return how a resume binds the case's `memories`: new instances on their folders, but as
 * `resume.memory` says: `null` binds nothing, and `{id}` binds a memory of that identity and its
 * entry's type in a folder of its own, `<dir>/resume-<input>`, made now. Also return those folders.
 */
async function resumeMemories(
  case_: Case,
  memories: CaseMemories,
  dir: string,
): Promise<{ bind: () => Record<string, MemoryBase>; roots: Map<string, string> }> {
  const overrides = get(get(case_, "resume", {}) as Case, "memory", {}) as Case;
  const roots = new Map<string, string>();
  for (const [input, binding] of Object.entries(overrides)) {
    if (binding === null) continue;
    const root = join(dir, `resume-${input}`);
    const made = memories.make(input, root, (binding as Case)["id"] as string);
    const signal = new AbortController().signal;
    await (await made.open({ create: true, itemType: memories.itemType(input), signal })).close();
    roots.set(input, root);
  }
  const bind = () => {
    const bound = memories.bind();
    for (const [input, binding] of Object.entries(overrides)) {
      if (binding === null) delete bound[input];
      else bound[input] = memories.make(input, roots.get(input)!, (binding as Case)["id"] as string);
    }
    return bound;
  };
  return { bind, roots };
}

/**
 * Return the files in the folder `root` but `.yian/`, by `/`-separated path sorted in code-unit
 * order, or `null` when there is no folder.
 */
function folderFiles(root: string): Case | null {
  if (!existsSync(root)) return null;
  const paths: string[] = [];
  const walk = (rel: string): void => {
    for (const entry of readdirSync(join(root, rel), { withFileTypes: true })) {
      const path = rel === "" ? entry.name : `${rel}/${entry.name}`;
      if (path === ".yian") continue;
      if (entry.isDirectory()) walk(path);
      else paths.push(path);
    }
  };
  walk("");
  // readdir's order is the file system's (unsorted on some, such as GPFS); the default sort
  // compares UTF-16 code units, so a listing reads the same everywhere.
  return Object.fromEntries(paths.sort().map((path) => [path, readFileSync(join(root, path), "utf8")]));
}

/**
 * Throw a `CaseFailure` when one of `texts` holds a memory's folder from `roots`, by input name,
 * as given or with symlinks resolved: a run's log, output and errors never hold a memory's
 * location.
 */
function checkNoRoots(roots: Iterable<readonly [string, string]>, texts: Record<string, string>): void {
  const leaks: string[] = [];
  for (const [input, root] of roots) {
    const forms = existsSync(root) ? [root, realpathSync(root)] : [root];
    for (const [where, text] of Object.entries(texts)) {
      if (forms.some((form) => text.includes(form))) leaks.push(`memory \`${input}\`'s folder appears in ${where}`);
    }
  }
  if (leaks.length > 0) throw new CaseFailure(leaks.join("\n"));
}

/** A case's scripted host: what `runFlow` gets from it, what it records, and its check. */
interface Host {
  /** A stub per `host.tools` entry, and the ask slot for `host.human`. */
  tools: Record<string, ToolEntry>;
  /** The draws retry backoffs take, from `host.jitter`; `null` for the implementation's own. */
  random: (() => number) | null;
  /** Builds every model the run calls; no call reaches a provider. */
  models: ModelAccess;
  /** The run points: the `host.stop` points and the `host.sends` `after` points. */
  inbox: Inbox;
  /** Wait for the `host.sends` items sent at run points; reject with the first refusal. */
  sent: () => Promise<void>;
  /** With `host.stop`, the run's signal and the error it is aborted with at the point. */
  stop: { signal: AbortSignal; error: ConformanceStop } | null;
  /** Each tool's calls in order, by tool name, as `{args, call_id, attempt, node}`. */
  calls: Map<string, Case[]>;
  /** Each script entry taken, in order, with the log's `nextSequence` when it was taken. */
  taken: { label: string; entry: Value; sequence: number }[];
  /**
   * Put each entry taken while the log's `nextSequence` was above `last` back at its script's
   * front, and forget the answers resumable stubs gave then.
   */
  putBack: (last: number) => void;
  /** By node, the AGENT calls a resume closes `crashed`, whose results its model requests read as the closer. */
  crashed: Map<string, ReadonlySet<string>>;
  /**
   * Throw a `CaseFailure` when a script ran out or was missing or a model call's request differed
   * from its entry's; return a message per script with entries left and per `host.stop` or
   * `host.sends` point never reached. `allowed` names the scripts that may run out, and how many
   * entries each may leave.
   */
  check: (allowed?: { overruns?: ReadonlySet<string>; left?: ReadonlyMap<string, number> }) => string[];
  /** Release the memory locks taken at `held` points. */
  release: () => void;
}

/**
 * Return the case's host: each stub tool call is recorded and takes its script's next entry, as
 * each ask (a TOOL `ask` or an agent's `ask_user`) takes the next `host.human` answer of its node,
 * recorded under the name it was called by, and each backoff the next
 * `host.jitter` draw and each model call the next `host.models` entry of its node and consumer.
 * Every model the run builds is the scripted one, after its config is resolved and checked with
 * `env`. Each stub call reaches its run points once it takes its entry. `log` is the
 * run's log file. The ask slot declares, as extras, every input the `flow`'s TOOL `ask` nodes pass
 * beyond yian's ask surface, such as an `adaptive_questions:` brief's context. At a `host.memory`
 * entry's `held` point, the lock of its memory, in its folder in `memories`, is taken and held
 * until `release`.
 */
function host(case_: Case, log: string, env: Env, flow: ComposedFlow, memories?: CaseMemories): Host {
  const spec = get(case_, "host", {}) as Case;
  const calls = new Map<string, Case[]>();
  const problems: Problem[] = [];
  const taken: Host["taken"] = [];
  // What each resumable stub answered, by `<where> <call id>`, with the log's `nextSequence` then.
  const answered = new Map<string, { entry: Value; sequence: number }>();
  const crashed: Host["crashed"] = new Map();
  const stopper = new AbortController();
  const stopError = new ConformanceStop();
  const stop = Object.hasOwn(spec, "stop") ? { signal: stopper.signal, error: stopError } : null;
  const delivered: Promise<unknown>[] = [];
  const locks: FileLock[] = [];
  const inbox = new Inbox(
    get(spec, "stop", {}) as Case,
    () => stopper.abort(stopError),
    get(spec, "sends", []) as Case[],
    // A live run takes the send through its engine at once; the promise settles after.
    (name, event) => {
      const sent = send(log, name, event, { env });
      sent.catch(() => {});
      delivered.push(sent);
    },
    (name) => {
      const cut = interrupt(log, name, { env });
      cut.catch(() => {});
      delivered.push(cut);
    },
    (name) => {
      const taken = withdraw(log, name, { env });
      taken.catch(() => {});
      delivered.push(taken);
    },
    get(spec, "memory", {}) as Case,
    // The run start made `.yian/`; the lock is the one FileMemory takes, so its writes wait.
    (input) => locks.push(lockFile(join(memories!.real(input), ".yian", "lock"), { private: true })),
  );
  const scripts = new Map<string, Value[]>(
    Object.entries(get(spec, "human", {}) as Case).map(([node, answers]) => [
      `host.human.${node}`,
      [...(answers as Value[])],
    ]),
  );

  const refuse = (label: string, problem: string, message: string): never => {
    problems.push({ text: problem, overrun: label });
    // Permanent, so a TOOL node doesn't retry the overrun; a model call's overrun fails its
    // AGENT. `check()` reports it either way.
    throw new PermanentToolError(message);
  };

  const take = (label: string, what: string): Value => {
    const script = scripts.get(label);
    if (script === undefined) return refuse(label, `${what} found no \`${label}\` list`, `\`${label}\` is used up`);
    if (script.length === 0) return refuse(label, `${what} found \`${label}\` used up`, `\`${label}\` is used up`);
    const entry = script.shift()!;
    taken.push({ label, entry, sequence: nextSequence(log) });
    return entry;
  };

  /** Record a call to `name`; take its script's next entry, or `null` with no `label` (a pending tool). */
  const call = (name: string, label: string | null, ctx: ToolContext, args: Case): Value => {
    const record = { args, call_id: ctx.callId, attempt: BigInt(ctx.attempt), node: ctx.node };
    let list = calls.get(name);
    if (list === undefined) calls.set(name, (list = []));
    list.push(record);
    const number = inbox.count(ctx.node, "tool_call");
    const entry = label === null ? null : take(label, `a call to \`${name}\` from node ${reprText(ctx.node)}`);
    if (number !== null) inbox.reached(ctx.node, { tool_call: number });
    return entry;
  };

  /**
   * Build the stub tool `name` declared at `where`: its calls are recorded under `shown` and take
   * from its `where.results` scripts. A resumable stub answers a call id it answered before with
   * the same entry and takes none, as an idempotent tool would.
   */
  const stubTool = (where: string, name: string, shown: string, stub: Case): FunctionTool => {
    const waits = stub["pending"] === true;
    const resumable = stub["resumable"] === true;
    const results = stub["results"];
    const byNode = isMapping(results);
    if (waits) {
      // A pending tool's calls wait; `host.sends` answers them.
    } else if (byNode) {
      for (const [node, script] of Object.entries(results)) {
        scripts.set(`${where}.results.${node}`, [...(script as Value[])]);
      }
    } else {
      scripts.set(`${where}.results`, [...(results as Value[])]);
    }
    const execute = (args: Record<string, unknown>, ctx: ToolContext): unknown => {
      if (waits) {
        call(shown, null, ctx, args as Case);
        return pending();
      }
      const label = `${where}.results${byNode ? `.${ctx.node}` : ""}`;
      const key = `${where} ${ctx.callId}`;
      const known = resumable ? answered.get(key) : undefined;
      let step: Case;
      if (known === undefined) {
        step = call(shown, label, ctx, args as Case) as Case;
        // The entry `call` just took, with the sequence it was taken at.
        if (resumable) answered.set(key, { entry: step, sequence: taken.at(-1)!.sequence });
      } else {
        call(shown, null, ctx, args as Case);
        step = known.entry as Case;
      }
      if (Object.hasOwn(step, "return")) return step["return"];
      throw scriptedError(step["raise"] as Case, TOOL_ERRORS);
    };
    const description = get(stub, "description", "") as string;
    const isConcurrencySafe = stub["concurrency_safe"] === true ? () => true : null;
    const parameters = stub["parameters"] as JsonObject;
    return tool({ name, description, parameters, execute, isConcurrencySafe, resumable });
  };
  const tools: [string, ToolEntry][] = [];
  // Each group's stub tools by `<group>.<tool>`, which a later `same_as` reuses.
  const groupStubs = new Map<string, FunctionTool>();
  for (const [name, entry] of Object.entries(get(spec, "tools", {}) as Case)) {
    if (!Object.hasOwn(entry as Case, "group")) {
      tools.push([name, stubTool(`host.tools.${name}`, name, name, entry as Case)]);
      continue;
    }
    const members: FunctionTool[] = [];
    for (const [member, stub] of Object.entries((entry as Case)["group"] as Case)) {
      const same = (stub as Case)["same_as"];
      if (typeof same === "string") {
        members.push(groupStubs.get(same)!);
        continue;
      }
      // A group's tool is called by its record name, `<group>_<tool>`.
      const built = stubTool(`host.tools.${name}.group.${member}`, member, visibleToolName(name, member), stub as Case);
      groupStubs.set(`${name}.${member}`, built);
      members.push(built);
    }
    tools.push([name, members]);
  }
  if (Object.hasOwn(spec, "human")) {
    const inputs: JsonObject = {};
    for (const [, node] of staticNodes(flow)) {
      if (!(node instanceof ToolNode) || node.builtin !== ASK) continue;
      for (const { name } of node.params ?? []) if (!ASK_SURFACE.has(name)) inputs[name] = {};
    }
    const ask: AskTool = {
      handler: (request, ctx) => call(ctx.toolName, `host.human.${ctx.node}`, ctx, askUserArgs(request)),
      inputs,
    };
    tools.push([ASK, ask]);
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
  const count = (label: string) => taken.filter((item) => item.label === label).length;
  const scripted = scriptedModels(take, problems, inbox, count, crashed);
  // Installed even without `host.models`, so an unscripted model call never reaches a provider.
  const models: ModelAccess = {
    env,
    createModel: (config, { nodeId }) => {
      // The script replaces only the provider: the config is resolved and checked as yian's own
      // factory does, so its errors stay yian's. No credential is read.
      const values = fromHost(config, "llm_config") as Record<string, unknown>;
      const [validated] = checkModelConfig(resolveModelConfig(values, { env }));
      return scripted(validated.provider!, validated.model!, nodeId, validated.structuredOutput ?? "native");
    },
  };

  const check: Host["check"] = ({ overruns = new Set(), left = new Map() } = {}) => {
    const failing = problems.filter(({ overrun }) => overrun === undefined || !overruns.has(overrun));
    if (failing.length > 0) throw new CaseFailure(failing.map(({ text }) => text).join("\n"));
    return [...scripts]
      .filter(([label, script]) => script.length > (left.get(label) ?? 0))
      .map(([label, script]) => {
        const entries = script.length === 1 ? "entry" : "entries";
        return `\`${label}\` has ${script.length} ${entries} left`;
      })
      .concat(inbox.left());
  };
  const putBack = (last: number): void => {
    // An answer given after the cut goes with its entry, so a call sent again takes it again.
    for (const [key, { sequence }] of answered) if (sequence > last) answered.delete(key);
    // Taken in `nextSequence` order, so the entries to put back are a tail.
    const first = taken.findIndex(({ sequence }) => sequence > last);
    if (first === -1) return;
    for (const item of taken.splice(first).reverse()) scripts.get(item.label)!.unshift(item.entry);
  };
  const sent = async (): Promise<void> => void (await Promise.all(delivered));
  const release = () => {
    for (const lock of locks.splice(0)) lock.release();
  };
  return {
    tools: Object.fromEntries(tools), random, models, inbox, stop, calls, taken, putBack, crashed, check, sent, release,
  };
}

/** Return a run's `failure` in the case's failure form. */
function failureForm(failure: Failure, flow: ComposedFlow): Case {
  return { ...diagnosticForm(flow.diagnose(failure)), node: failure.nodeId };
}

/**
 * Return the data form of an event or stream chunk, without its null fields and a model call's
 * `started_at` (a time): its `toJSON()`, with a failure in failure form and every file as
 * `{uri, mime, name}`.
 */
function fields(event: RunEvent, flow: ComposedFlow): Case {
  const data: Case = {};
  for (const [key, item] of Object.entries(event.toJSON())) {
    if (item === null || key === "started_at") continue;
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

// SPEC.md's trace table: record kind -> [the payload field naming the node, the payload fields
// kept].
type TraceRow = readonly [string | null, readonly string[]];
const TRACE: ReadonlyMap<string, TraceRow> = new Map<string, TraceRow>([
  [
    "run_started",
    [null, ["num_workers", "defaults", "effective_llm_by_static_path", "call_time_llm", "parent"]],
  ],
  ["run_resumed", [null, ["retry_node_ids", "num_workers", "closed_calls"]]],
  ["run_stopped", [null, ["reason", "node_ids"]]],
  ["node_waiting", ["node_id", ["call_ids", "waits", "steer_inbox"]]],
  ["node_woken", ["node_id", ["timed_out"]]],
  ["run_waiting", [null, ["node_ids"]]],
  ["inbox_closed", [null, ["inbox"]]],
  ["inbox_withdrawn", [null, ["inbox", "events"]]],
  ["interrupted", [null, ["inbox", "nodes"]]],
  ["run_cancelled", [null, []]],
  ["run_failed", [null, ["failure"]]],
  ["node_started", ["node_id", []]],
  ["node_step", ["node_id", ["n", "step"]]],
  ["pool_commit", ["node_id", ["key", "value", "from", "inbox", "events", "closed", "steps"]]],
  ["node_succeeded", ["node_id", ["commit", "messages"]]],
  ["node_routed", ["node_id", ["handle", "terminal_failure"]]],
  ["node_expanded", ["spawner_id", ["prune_ids"]]],
  ["node_failed", ["failure", ["failure"]]],
]);
/** The run start's commits whose values are times, which the trace leaves out. */
const TIMED_KEYS: ReadonlySet<string> = new Set(["system.now", "system.today"]);
/** The fields of a node step that are times (an AGENT's model calls and tool results). */
const STEP_TIMES: readonly string[] = ["started_at", "first_token_at"];
const LEFT_OUT_WHEN_EMPTY: ReadonlySet<string> = new Set([
  "defaults", "effective_llm_by_static_path", "call_time_llm", "parent", "waits", "timed_out", "closed_calls",
]);
const FAILURE_FIELDS: ReadonlyMap<string, string> = new Map([
  ["node_failed", "failure"],
  ["node_routed", "terminal_failure"],
  ["run_failed", "failure"],
]);

/**
 * Return the objects of a record's payload that may hold a `deadline` (a time): a `node_waiting`'s
 * wait entries and a `pending` step.
 */
function deadlineHolders(kind: string, payload: Case): Case[] {
  if (kind === "node_waiting") return payload["waits"] as Case[];
  const step = kind === "node_step" ? (payload["step"] as Case) : null;
  return step !== null && step["kind"] === "pending" ? [step] : [];
}

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

/** Return the run log's `lines` projected to SPEC.md's trace. */
export function trace(lines: readonly string[], flow: ComposedFlow): Value[][] {
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
    // Deadlines are times.
    for (const holder of deadlineHolders(kind, payload)) delete holder["deadline"];
    if (kind === "node_step") {
      const step = payload["step"] as Case;
      if (Object.hasOwn(step, "retry_in") && step["retry_in"] !== null) step["retry_in"] = Number(step["retry_in"]);
      // Step times are times.
      for (const time of STEP_TIMES) delete step[time];
      if ((get(step, "effective_llm_config") ?? null) !== null) {
        step["effective_llm_config"] = llm(step["effective_llm_config"] as Case);
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

/**
 * Return `result` in the form of `expect.output`. A waiting or cancelled run has no failure, so
 * its form is its status alone, which no expectation matches.
 */
function output(result: RunResult, flow: ComposedFlow): Case {
  if (result.status === "succeeded") {
    return { status: "succeeded", output: dataForm(result.output as YianValue) };
  }
  if (result.failure === null) return { status: result.status };
  return { status: result.status, failure: failureForm(result.failure, flow) };
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
export function pooled(results: Record<string, unknown>): Record<string, unknown> {
  // A `NodeFailed` belongs to its failure's node; run events have no `node_id`.
  const eventNode = (event: Value[]): string | null => {
    const data = (event[1] ?? {}) as Case;
    const node = event[0] === "NodeFailed" ? (data["failure"] as Case)["node"] : data["node_id"];
    return (node ?? null) as string | null;
  };
  const toolCalls = results["tool_calls"] as Record<string, Case[]>;
  // A success's `commit` and `messages` are seqs of `pool_commit`s just before it in its node's
  // group, and a messages commit's `steps` are seqs of its node's steps: seqs that depend on how the
  // parallel nodes interleave, so a grouped trace leaves them out. A derived commit's `from` seqs
  // point into other nodes' groups, so each is named by the kind and node of its record instead.
  const entries = results["trace"] as Value[][];
  const trace = entries.map((entry) => {
    if (!(entry[0] === "node_succeeded" || entry[0] === "pool_commit") || entry[2] === undefined) return entry;
    const { commit: _commit, messages: _messages, steps: _steps, ...rest } = entry[2] as Case;
    if (isMapping(rest["from"])) {
      rest["from"] = Object.fromEntries(
        Object.entries(rest["from"]).map(([name, seq]) => {
          // The trace has one entry per log line, and a record's seq is its line index.
          const source = entries[Number(seq)];
          if (source === undefined) throw new CaseFailure(`a derived commit names seq ${seq}, which the log doesn't hold`);
          // A node writes several commits, so the name alone can't tell a wrong one of its node's.
          const key = ((source[2] ?? {}) as Case)["key"] ?? source[1];
          if (source[0] !== "pool_commit" || key !== name) {
            throw new CaseFailure(`a derived commit names seq ${seq} for \`${name}\`, which isn't its pool_commit`);
          }
          return [name, `${String(source[0])} ${String(source[1])}`];
        }),
      );
    }
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

/**
 * Return `results` with the made-up run id `runId` in their text replaced by `<id:1>`; `null`
 * replaces nothing. Mapping keys come out in UTF-16 order.
 */
function placeholders(results: Record<string, unknown>, runId: string | null): Record<string, unknown> {
  if (runId === null) return results;
  const replace = (value: unknown): unknown => {
    if (typeof value === "string") return value.replaceAll(runId, "<id:1>");
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
 * Run `start` with every append it makes refused once the log holds record `last`; throw a
 * `CaseFailure` (`at` saying where) unless it rejects with that refusal.
 */
async function cutAt(last: number, start: () => Promise<unknown>, at = ""): Promise<void> {
  const refused = new Cut("the conformance runner refused this append");
  const refuse = (log: RunLog) => {
    if (log.nextSequence > last) throw refused;
  };
  // Scoped to `start` and the async work it starts, so a cut still running after its test timed
  // out can't refuse another test's appends.
  const outcome = await withAppendHook(refuse, () => settle(start));
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
 * Return the options every run of a case gets: its log, environment and host, on a virtual clock that starts at `start`, with
 * `events` and `chunks` collecting what it publishes.
 */
function runHooks(
  log: string,
  env: Env,
  { tools, random, models }: Host,
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
 * Run the flow with `inputs` and the `run` arguments on the case's host, logged to `run.jsonl` in
 * the case directory, its waits on a virtual clock that starts at the real current time; with
 * `host.stop`, the run is stopped at its point. A `resume` case's first run is cut after record
 * `host.crash_after_record` (or, for `end`, runs to its end; with `host.stop`, is stopped) and
 * resumed with the `resume` arguments. Each run and `advance` binds new instances of the
 * `host.memory` memories, in folders under `memory/` in the case directory. Return the results:
 * `trace`, `events`, `stream` (the resume's), `tool_calls` (every run's), `output` or the host
 * error as `raises` and `error` (never compared), `memory` (the folders `expect.memory` names,
 * after the run), `memory_drift` (the resume's), and `left`, a message per script with entries
 * left.
 */
async function actualRun(case_: Case, ctx: CaseContext): Promise<Record<string, unknown>> {
  const flow = await loadText(case_["flow"] as string, ctx);
  const log = join(ctx.dir, "run.jsonl");
  const env = caseEnv(ctx);
  const spec = get(case_, "host", {}) as Case;
  const folders = join(ctx.dir, "memory");
  const memories = await caseMemories(spec, folders, env, flow);
  const scripted = host(case_, log, env, flow, memories);
  const { tools, models, inbox, stop, calls, check, sent } = scripted;
  const run = get(case_, "run", {}) as Case;
  const resume = get(case_, "resume", {}) as Case;
  const events: RunEvent[] = [];
  const chunks: StreamChunk[] = [];
  // One clock for every run of the case. `env` scopes the case's YIAN_* values to each run, so
  // nothing is installed around it.
  const hooks = runHooks(log, env, scripted, Date.now(), events, chunks);
  const inputs = get(case_, "inputs", {}) as Case;
  let signal: { signal?: AbortSignal } = stop === null ? {} : { signal: stop.signal };
  const runArguments = runOptions<RunFlowOptions>(run, RUN_OPTIONS);
  let start = () => runFlow(flow, { ...inputs, ...memories.bind() }, { ...runArguments, ...hooks, ...signal });
  // Each `advance` binds the memories again, as a host does; a case without them passes none.
  const { memory: resumeMemory, ...resumeArgs } = resume;
  const binds = Object.hasOwn(spec, "memory") || resumeMemory !== undefined;
  let bound = () => (binds ? { memory: memories.bind() } : {});
  const roots: [string, string][] = [...memories.roots];
  // After each `host.sends` item, the run moves on with the arguments `advance` shares with `run`.
  const sharedArgs = Object.fromEntries(
    Object.entries(run).filter(([key]) => Object.hasOwn(RESUME_OPTIONS, key)),
  );
  let advanceArguments = runOptions<AdvanceOptions>(sharedArgs, RESUME_OPTIONS);
  let result: RunResult | undefined;
  let drift: string[] | null = null;
  let caught: Record<string, unknown> = {};
  // The items sent while the run waits; the others are sent at a point, or while it is down.
  const all = get(spec, "sends", []) as Case[];
  const sends = all.filter((item) => !Object.hasOwn(item, "after") && !Object.hasOwn(item, "down"));
  // A run must not end waiting when it was given items to send while it waits, or an empty list.
  const waitingSends = Object.hasOwn(spec, "sends") && (sends.length > 0 || all.length === 0);
  inbox.armed = true;
  try {
    if (case_["kind"] === "resume") {
      await cutFirstRun(spec, () => start(), stop);
      // The stop ended the first run; aborted for good, it would stop each later advance at once.
      signal = {};
      // The sends a host makes while the run is down.
      for (const item of get(spec, "sends", []) as Case[]) {
        if (item["down"] === true) await send(log, item["inbox"] as string, item["event"], { env });
      }
      events.length = 0;
      chunks.length = 0;
      const resumed = await resumeMemories(case_, memories, folders);
      roots.push(...resumed.roots);
      if (binds) bound = () => ({ memory: resumed.bind() });
      const resumeArguments = runOptions<AdvanceOptions>(resumeArgs, RESUME_OPTIONS);
      advanceArguments = resumeArguments;
      start = () => advance({ ...resumeArguments, ...hooks, ...bound() });
    }
    try {
      result = await start();
      if (case_["kind"] === "resume") drift = result.memoryDrift.map(({ input }) => input);
      await sent();
      // Each item is delivered while the run waits; the same collectors see every `advance`.
      while (result.status === "waiting" && sends.length > 0) {
        const item = sends.shift()!;
        if (Object.hasOwn(item, "close")) await close(log, item["close"] as string, { env });
        else if (Object.hasOwn(item, "withdraw")) await withdraw(log, item["withdraw"] as string, { env });
        else if (Object.hasOwn(item, "interrupt")) await interrupt(log, item["interrupt"] as string, { env });
        else if (Object.hasOwn(item, "complete")) await complete(log, item["complete"] as string, completion(item), { env });
        else await send(log, item["inbox"] as string, item["event"], { env });
        // A held item leaves the run waiting, so the next item lands on the same wait.
        if (item["hold"] === true) continue;
        result = await advance({ ...advanceArguments, ...hooks, ...signal, ...bound() });
        await sent();
      }
    } catch (error) {
      caught = { raises: raised(error), error };
    }
  } finally {
    scripted.release();
  }
  if (waitingSends && result?.status === "waiting") {
    throw new CaseFailure("the run is still waiting after host.sends ran out");
  }
  // A call past a script's end is the case's own error, whatever it made differ.
  const left = check();
  if (existsSync(log)) await checkInvariants(flow, log, { env, tools, models }, result ?? null);
  goldenLog(log, ctx);
  const text = existsSync(log) ? readFileSync(log, "utf8") : "";
  const lines = text.split("\n").filter((line) => line !== "");
  let actual: Record<string, unknown> = {
    trace: trace(lines, flow),
    events: events.map((event) => eventForm(event, flow)),
    stream: chunks.map((chunk) => fields(chunk, flow)),
    tool_calls: Object.fromEntries([...calls].map(([name, list]) => [name, list.map(dataForm)])),
    ...(result !== undefined ? { output: output(result, flow) } : {}),
  };
  const texts = {
    ...Object.fromEntries(Object.entries(actual).map(([name, value]) => [name, spell(value as Value)])),
    "the run log": text,
    "the raised error": str((caught["raises"] as Case | undefined)?.["message"] ?? ""),
  };
  checkSecrets(case_, texts);
  checkNoRoots(roots, texts);
  // A `resume` case's results are grouped by the resumed run's `num_workers`.
  if (!isZero(get({ ...run, ...resume }, "num_workers", 8n))) actual = pooled(actual);
  const runId =
    Object.hasOwn(run, "run_id") || lines.length === 0
      ? null
      : ((parseJson(lines[0]!, "codec") as Case)["run_id"] as string);
  // A raised error is compared as is, without placeholders.
  actual = placeholders(actual, runId);
  const named = Object.keys(get(case_["expect"] as Case, "memory", {}) as Case);
  const memory = Object.fromEntries(named.map((input) => [input, memories.contents(input)]));
  return { ...actual, memory, ...(drift === null ? {} : { memory_drift: drift }), ...caught, left };
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
  for (const name of ["memory", "memory_drift"]) {
    if (Object.hasOwn(expect, name)) compare(name, actual[name], expect[name]);
  }
  for (const name of REWRITTEN) {
    if (!Object.hasOwn(expect, name)) continue;
    // The trace keeps kinds.
    const byValue = BY_VALUE.has(name);
    if (rewrites !== undefined) {
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
 * commits), a step's times (`started_at` and `first_token_at` when set), a set `deadline` of a
 * `node_waiting` wait entry or a `pending` step, the recorded flow's folder (`flow.dir`) and the
 * yian version (`yian_version`) replaced by `<masked>`, written again with the log's own encoder.
 * The times and the seeded values come from the case's virtual clock, which starts at the real
 * time; the folder is the case's temporary one; the version changes with each release.
 */
export function masked(text: string): string {
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
      for (const holder of deadlineHolders(record["kind"] as string, payload)) {
        if ((holder["deadline"] ?? null) !== null) holder["deadline"] = "<masked>";
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
 * 0; a waiting node isn't, nor a woken TOOL node); no record follows the run's terminal outcome;
 * after a `run_stopped` only node steps until a `run_resumed`; `inspectRun` with the flow reads a
 * replay's status, in-flight nodes, crashed counts and failure, and the log's run id, schema, flow
 * fingerprint and last seq (and no waits), and without it the same except in its one blind
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
  const replay = (count: number) =>
    withEnv(options.env, () => replayRunLog(flow, MemoryRunLog.fromRecords(envelopes.slice(0, count))));
  // A woken AGENT calls a model and takes a worker; a woken TOOL node decides inline and takes
  // none (`wakesWithoutWorker`). The node is live in the graph the log replays to before its wake.
  const wakeTakesWorker = (seq: number, nodeId: string) => !replay(seq).flow.nodes.get(nodeId)!.wakesWithoutWorker;
  const inFlight = new Set<string>();
  let workers = 1;
  let stopped = false;
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
    if (kind === "node_woken" && wakeTakesWorker(seq, payload["node_id"] as string)) {
      inFlight.add(payload["node_id"] as string);
    }
    // A waiting node holds no worker, so other ready work runs beside it.
    if (kind === "node_succeeded" || kind === "node_routed" || kind === "node_waiting") {
      inFlight.delete(payload["node_id"] as string);
    }
    // A flow call's growth leaves its agent in flight.
    if (kind === "node_expanded" && payload["call_id"] === null) inFlight.delete(payload["spawner_id"] as string);
    if (kind === "node_failed") inFlight.delete((payload["failure"] as Readonly<Case>)["node_id"] as string);
    if (inFlight.size > workers) {
      fail(`${inFlight.size} nodes are in flight after record ${seq}, over ${workers}`);
    }
  });
  // A terminal result is never undone, so checking the prefix before the last record covers every
  // record after it. A prefix inside the run start (through the START inputs' commit, the first
  // whose key is its node) can't be replayed, and holds no outcome.
  const inputsCommit = envelopes.findIndex(({ kind, payload }) => kind === "pool_commit" && payload["key"] === payload["node_id"]);
  if (inputsCommit === -1) return;
  const last = envelopes.length - 1;
  if (last > inputsCommit && replay(last).result !== null) fail(`record ${last} follows the terminal outcome`);
  // Mirrors `inspectRun`'s replay branch on purpose: the check is that it takes that branch and
  // agrees with a replay, not a second statement of its rules.
  const state = replay(envelopes.length);
  const terminal = state.result;
  const finished = terminal === null ? null : terminal instanceof RunSucceeded ? "succeeded" : "failed";
  const counts = new Map(state.crashedAttempts);
  if (finished === null) addCrashedAttempts(counts, state.acceptedStarts, state.lastStopSequence);
  const first = envelopes[0]!;
  const started = first.record as RunStartedRecord;
  const expected = {
    status: finished ?? (state.lastStopSequence === null ? "crashed" : "stopped"),
    inFlight: finished === null ? [...state.acceptedStarts.keys()] : [],
    crashedAttempts: Object.fromEntries(counts),
    runId: first.runId,
    schema: first.schema,
    flowFingerprint: started.flowFingerprint,
    lastSeq: envelopes.length - 1,
    startedAt: first.at,
    endedAt: finished === null ? null : envelopes.at(-1)!.at,
    failure: terminal instanceof RunFailed ? failureForm(terminal.failure, flow) : null,
    // No case waits yet, so the replay's waits read as none.
    waits: [],
    ready: false,
    wakeAt: null,
  };
  const form = (inspection: RunInspection) => ({
    ...inspection,
    inFlight: [...inspection.inFlight],
    crashedAttempts: Object.fromEntries(inspection.crashedAttempts),
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
    const counted = Object.fromEntries(result.crashedAttempts);
    compare(`crashed counts${at}`, counted, expected.crashedAttempts);
  }
  const before = readFileSync(log);
  const reopened = await advance({ log, ...options });
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
 * Return the node outcome entries of a run log's trace `entries`, each success's `commit` replaced
 * by the trace fields of the `pool_commit` it names, whose `from` seqs are given as the kind and
 * node of the record each names, and its `messages` by the messages commit's fields, each of whose
 * `steps` is given as its step's node and kind (a resumed call-time agent records its config
 * again, so step numbers move): successes, `from` and `steps` name records by seq, and a resume
 * adds records, so a crash sweep compares the records instead.
 */
export function sourcesByRecord(entries: readonly Value[][]): Value[][] {
  const kept = entries.filter(([kind]) => OUTCOMES.has(kind as string));
  for (const [kind, , payload] of kept) {
    if (kind !== "node_succeeded") continue;
    const commit = (payload as Case)["commit"] as bigint;
    const named = entries[Number(commit)];
    if (named?.[0] !== "pool_commit") throw new CaseFailure(`a success names seq ${commit}, which isn't a pool_commit`);
    const fields = { ...((named[2] ?? {}) as Case) };
    if (isMapping(fields["from"])) {
      fields["from"] = Object.fromEntries(
        Object.entries(fields["from"]).map(([name, seq]) => {
          const source = entries[Number(seq)];
          return [name, `${String(source?.[0])} ${String(source?.[1])}`];
        }),
      );
    }
    (payload as Case)["commit"] = fields;
    const messages = (payload as Case)["messages"];
    if (typeof messages === "bigint") {
      const commit = entries[Number(messages)];
      if (commit?.[0] !== "pool_commit") throw new CaseFailure(`a success names messages ${messages}, which isn't a pool_commit`);
      const steps = ((commit[2] as Case)["steps"] as bigint[]).map((seq) => {
        const step = entries[Number(seq)];
        const body = (((step?.[2] ?? {}) as Case)["step"] ?? {}) as Case;
        return `${String(step?.[0])} ${String(step?.[1])} ${String(body["kind"])}`;
      });
      (payload as Case)["messages"] = { ...(commit[2] as Case), steps };
    }
  }
  return kept;
}

/**
 * Return, by node, the AGENT tool calls a cut after record `k` of the uncut run's `records` cuts
 * off: each call a reply up to `k` made, with no result up to `k`, that the uncut run made
 * (`made` holds the yian ids of its host calls, `<run>/<node>/<n>/<index>`), by call id with its
 * tool's name. These are the calls a resume closes `crashed`; a call the agent never made (an
 * ungranted tool, a reply past its tool-call limit) has nothing to cut off. A memory tool's call
 * isn't a host call, so it is never here: memory tools are resumable, and the resume sends it again.
 */
function cutCalls(
  records: readonly RecordEnvelope[],
  k: number,
  runId: string,
  made: ReadonlySet<string>,
): Map<string, Map<string, string>> {
  const cut = new Map<string, Map<string, string>>();
  for (const { record } of records.slice(0, k + 1)) {
    if (!(record instanceof NodeStepRecord)) continue;
    const { step, nodeId, n } = record;
    if (step["kind"] === "tool_result") cut.get(nodeId)?.delete(step["call_id"] as string);
    if (step["kind"] !== "model_call") continue;
    const parts = ((step["output"] as JsonObject | null)?.["parts"] ?? []) as JsonObject[];
    const calls = parts.filter((part) => part["type"] === "tool_call");
    for (const [index, part] of calls.entries()) {
      if (!made.has(`${runId}/${nodeId}/${n}/${index}`)) continue;
      let open = cut.get(nodeId);
      if (open === undefined) cut.set(nodeId, (open = new Map()));
      open.set(part["id"] as string, part["name"] as string);
    }
  }
  for (const [node, calls] of cut) if (calls.size === 0) cut.delete(node);
  return cut;
}

/** Return `calls`, `[node, call id]` pairs, as one sorted text for a message. */
function callNames(calls: readonly (readonly [string, string])[]): string {
  return calls
    .map(([node, id]) => `${node} ${id}`)
    .sort(compareCodeUnits)
    .join(", ");
}

/**
 * @internal Return the label of the script a call to tool `name` from AGENT `node` takes from:
 * its `host.tools` results, else `host.human`. A group's tool is called `<group>_<tool>`, or by
 * its own name when an agent granted it by its bare name; a `same_as` reads as the tool it names.
 */
export function scriptLabel(spec: Case, node: string, name: string): string {
  const found = stubCalled(spec, name);
  if (found === undefined) return `host.human.${node}`;
  return `${found.where}.results${isMapping(found.stub["results"]) ? `.${node}` : ""}`;
}

/**
 * Return the stub tool an AGENT's call to tool `name` reaches: a group's tool by `<group>_<tool>`,
 * or by its own name when an agent granted it by its bare name; `undefined` for an ask.
 */
function stubCalled(spec: Case, name: string): StubTool | undefined {
  const stubs = stubTools(spec);
  return stubs.find((each) => each.shown === name) ?? stubs.find((each) => each.own === name);
}

/** One stub tool `host.tools` declares: where, its stub, its visible name and its own name. */
interface StubTool {
  readonly where: string;
  readonly stub: Case;
  /** A top-level stub's name, or a group's tool's `<group>_<tool>`. */
  readonly shown: string;
  /** The tool's own name, which a bare grant of a group's tool shows. */
  readonly own: string;
}

/**
 * Return each stub tool `host.tools` declares. A `same_as` member gives the stub it names, under
 * the member's own `<group>_<tool>`.
 */
function stubTools(spec: Case): StubTool[] {
  const found: StubTool[] = [];
  const named = new Map<string, [string, Case]>();
  for (const [name, entry] of Object.entries(get(spec, "tools", {}) as Case)) {
    if (!Object.hasOwn(entry as Case, "group")) {
      found.push({ where: `host.tools.${name}`, stub: entry as Case, shown: name, own: name });
      continue;
    }
    for (const [member, stub] of Object.entries((entry as Case)["group"] as Case)) {
      const same = (stub as Case)["same_as"];
      const [where, declared]: [string, Case] =
        typeof same === "string" ? named.get(same)! : [`host.tools.${name}.group.${member}`, stub as Case];
      if (typeof same !== "string") named.set(`${name}.${member}`, [where, declared]);
      found.push({ where, stub: declared, shown: visibleToolName(name, member), own: member });
    }
  }
  return found;
}

/**
 * Return whether `case_` gets a crash sweep: a serial `run` case expecting `output`, not
 * `pending`, with no `host.stop`, `host.sends` (an interrupt item among them) or `held` memory.
 */
export function sweepable(case_: Case): boolean {
  const spec = get(case_, "host", {}) as Case;
  return (
    case_["kind"] === "run" &&
    isZero(get(get(case_, "run", {}) as Case, "num_workers", 8n)) &&
    Object.hasOwn(case_["expect"] as Case, "output") &&
    !Object.hasOwn(case_, "pending") &&
    !["stop", "sends"].some((key) => Object.hasOwn(spec, key)) &&
    !Object.values(get(spec, "memory", {}) as Case).some((entry) => Object.hasOwn(entry as Case, "held"))
  );
}

/**
 * Run a `run` case's crash sweep. An uncut run writes N records; then, for each k in 0..N-2, a run
 * on a fresh host is cut after record k, the script entries it took after the cut are put back (a
 * cut right after a node's commit puts back from the node's last record before it, since the node
 * runs again), the invariants are checked at the cut, `beforeResume` (for a test, to damage the log
 * or the cut's memory folders, by input name) runs, and the run is resumed. Each run binds new
 * memories in folders of its own, `sweep-memory/` for the uncut run and `sweep-<k>-memory/` for
 * cut `k`, and the resume binds new instances on the cut's folders. A cut inside the run start (before the START inputs' commit) must instead be
 * refused by the resume, leaving the log as it is. The resume must end with a result whose invariants hold,
 * close `crashed` exactly the AGENT tool calls the cut cut off before their result (`cutCalls`) but
 * those to resumable stubs, which it sends again, pass its
 * host's `check()` with no entry left, and leave no filled secret in its log, as the uncut run's
 * log must not. For each call closed `crashed`, the script its call took from may keep one entry,
 * the agent's `host.models` script may run out, and the agent's later scripted requests read the
 * call's result as the closer. The cut's memory folders must then hold what `expect.memory` says,
 * and no log a memory's folder. With no
 * `host.human` or `host.jitter` and only `return` entries in its tool scripts, the strong oracle
 * also holds: the same output, and the same node outcome entries in the trace, in order, with each
 * record a success names given by its kind and node rather than its seq.
 */
export async function crashSweep(
  case_: Case,
  ctx: CaseContext,
  {
    beforeResume = (_log: string, _memory: ReadonlyMap<string, string>) => {},
  }: { beforeResume?: (log: string, memory: ReadonlyMap<string, string>) => void } = {},
): Promise<void> {
  const flow = await loadText(case_["flow"] as string, ctx);
  const env = caseEnv(ctx);
  const spec = get(case_, "host", {}) as Case;
  // A `same_as` member shares its stub's scripts, so each stub counts once.
  const tools = [...new Map(stubTools(spec).map(({ where, stub }) => [where, stub])).values()];
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
  const uncutMemories = await caseMemories(spec, join(ctx.dir, "sweep-memory"), env, flow);
  const uncutHost = host(case_, uncutLog, env, flow, uncutMemories);
  const uncutHooks = runHooks(uncutLog, env, uncutHost, start);
  const uncut = await runFlow(flow, { ...inputs, ...uncutMemories.bind() }, { ...args, ...uncutHooks });
  const made = new Set([...uncutHost.calls.values()].flatMap((list) => list.map(({ call_id }) => call_id as string)));
  const uncutText = { "the uncut run log (crash sweep)": readFileSync(uncutLog, "utf8") };
  checkSecrets(case_, uncutText);
  checkNoRoots(uncutMemories.roots, uncutText);
  // A case without memories passes no memory option, as before.
  const binds = Object.hasOwn(spec, "memory");
  const expectMemory = get(case_["expect"] as Case, "memory");
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
    const memories = await caseMemories(spec, join(ctx.dir, `sweep-${k}-memory`), env, flow);
    const memory = () => (binds ? { memory: memories.bind() } : {});
    const scripted = host(case_, log, env, flow, memories);
    const hooks = runHooks(log, env, scripted, start);
    // The uncut run's id, so the runs agree on it when the case gives none.
    const cut = () => runFlow(flow, { ...inputs, ...memories.bind() }, { ...args, ...hooks, runId: uncut.runId });
    await cutAt(k, cut, at);
    // A node's commit and its success are written with nothing awaited between them, so a cut
    // right after a commit is a crash before the success: the node runs again and takes its
    // script entries again. Entries taken up to the node's last record before the commit stay,
    // since the resume replays that record's steps.
    scripted.putBack(k > inputsCommit && uncutRecords[k]!.kind === "pool_commit" ? k - 1 : k);
    if (k < inputsCommit) {
      // A log cut inside its run start doesn't hold the run's inputs yet, so it can't be resumed.
      const before = readFileSync(log);
      const refused = await settle(() => advance({ ...hooks, ...memory() }));
      const message = `run log ${log} ends inside its run start; start the run again`;
      if (!(refused instanceof RangeError) || refused.message !== message) {
        throw new CaseFailure(`the resume didn't refuse the incomplete run start${at}\n${failureText(refused)}`);
      }
      if (!readFileSync(log).equals(before)) throw new CaseFailure(`the refused resume changed the log${at}`);
      continue;
    }
    const reopen = { env, tools: scripted.tools, models: scripted.models };
    await checkInvariants(flow, log, reopen, null, at);
    beforeResume(log, memories.roots);
    const open = cutCalls(uncutRecords, k, uncut.runId, made);
    // The resume sends a resumable stub's cut calls again and reads their answers, so it closes
    // only the others, and only theirs get a script allowance and a model overrun.
    for (const [node, calls] of open) {
      for (const [id, name] of calls) if (stubCalled(spec, name)?.stub["resumable"] === true) calls.delete(id);
      if (calls.size === 0) open.delete(node);
    }
    for (const [node, calls] of open) scripted.crashed.set(node, new Set(calls.keys()));
    let resumed: RunResult;
    try {
      // The cut run's `context`, host key and header values, which the log doesn't hold, as a
      // `resume` case passes them again.
      const context = args.context === undefined ? {} : { context: args.context };
      const { api_key: apiKey, headers } = (args.llmConfig ?? {}) as Pick<AdvanceOptions, "headers"> & {
        api_key?: string;
      };
      resumed = await advance({ ...hooks, ...context, apiKey, headers, ...memory() });
    } catch (error) {
      throw new CaseFailure(`the resume raised${at}\n${failureText(error)}`);
    }
    // The calls the cut left open are closed `crashed`, not made again, and only they.
    const closed = callNames(
      readRunLog(log)
        .slice(k + 1)
        .flatMap(({ record }) =>
          record instanceof NodeStepRecord && record.step["closed"] === "crashed"
            ? [[record.nodeId, record.step["call_id"] as string] as const]
            : [],
        ),
    );
    const expectedClosed = callNames([...open].flatMap(([node, calls]) => [...calls.keys()].map((id) => [node, id] as const)));
    if (closed !== expectedClosed) {
      throw new CaseFailure(`the resume closed [${closed}] crashed, but the cut cut off [${expectedClosed}]${at}`);
    }
    // Given back what it took after the cut, the resume takes every entry with the requests
    // scripted, as the uncut run did, but for the calls closed `crashed`: an entry such a call
    // took stays taken, and its agent may make a model call more, having read the closer.
    const overruns = new Set<string>();
    const allowance = new Map<string, number>();
    for (const [node, calls] of open) {
      overruns.add(`host.models.${node}.agent`);
      for (const name of calls.values()) {
        const label = scriptLabel(spec, node, name);
        allowance.set(label, (allowance.get(label) ?? 0) + 1);
      }
    }
    let left: string[];
    try {
      left = scripted.check({ overruns, left: allowance });
    } catch (error) {
      throw new CaseFailure(`the resume's host calls differ from the script${at}\n${(error as Error).message}`);
    }
    if (left.length > 0) throw new CaseFailure(left.map((message) => `${message}${at}`).join("\n"));
    const text = { [`the run log${at}`]: readFileSync(log, "utf8") };
    checkSecrets(case_, text);
    checkNoRoots(memories.roots, text);
    // Each write landed once: the cut's folders hold what the uncut run's would.
    if (isMapping(expectMemory)) {
      const contents = Object.keys(expectMemory).map((input) => [input, memories.contents(input)]);
      compare(`memory${at}`, Object.fromEntries(contents), expectMemory);
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
 * Return a new node holding `value` in case form: a float always with a fraction (with `byValue`,
 * for a result that compares numbers by value, only a float that isn't whole), and a collection in
 * flow style when `flow` says so or it holds only scalars.
 */
function caseNode(doc: Document, value: Value, flow = false, byValue = false): YamlNode {
  const node = doc.createNode(encode(value)) as YamlNode;
  visit(node, {
    Scalar(_, scalar) {
      if (typeof scalar.value === "number" && !(byValue && Number.isInteger(scalar.value))) {
        scalar.minFractionDigits = 1;
      }
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
      if (pair === undefined) node.items.push(doc.createPair(name, caseNode(doc, item, false, flags.byValue)));
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
        items.push(
          old === undefined ? caseNode(doc, value[j]!, flow, flags.byValue) : updated(doc, old, value[j]!, flags),
        );
        j += 1;
      } else {
        gone.push(olds[i]);
        i += 1;
      }
    }
    node.items = items;
    return node;
  }
  return caseNode(doc, value, false, flags.byValue);
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
 * Return the case file `text` with each result in `results` written into `expect` in case form
 * (as `updated` changes it). Only the lines of `expect` change: they are cut from the whole file
 * written again and put in place of the old ones, because the writer spells some other text (block
 * scalar headers, for one) unlike the case.
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
  const written = doc.toString(options(indentSeq));
  const [start, end] = section(text, "expect")!;
  const result = text.slice(0, start) + cut(written, "expect") + text.slice(end);
  if (!same(readCase(result), readCase(written))) {
    throw new Error("rewriting the case changed more than its expect");
  }
  return result;
}

// ---- resume ----

// `resume` cases run through `runRun`: `actualRun` cuts or stops their first run and resumes it.

// ---- expr ----

/**
 * Return `value` in data form: every `FileRef` in it as its mapping `{uri, mime, name}`, every
 * `InboxRef` as `{inbox}`, and every `MemoryRef` as `{memory}`.
 */
function dataForm(value: YianValue): Value {
  if (FileRef.isFileRef(value)) return { uri: value.uri, mime: value.mime, name: value.name };
  if (InboxRef.isInboxRef(value)) return { inbox: value.inbox };
  if (MemoryRef.isMemoryRef(value)) return dataForm(jsonPart(value, [], "mapping"));
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
  // Read inside the result, so a bad variable is raised as the case's error; scoped, not
  // installed, so neither a later case nor the caller reads the case's values.
  await results(case_["expect"] as Case, () =>
    withEnv(caseEnv(ctx), () => {
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
      const resolve = (names: readonly string[]) => resolveInRecord(names, values);
      if (template !== null) return { value: dataForm(evalBinding(template, resolve, { item, types })) };
      if (mode === "condition") {
        // What every `when:`, `while:` and assert runs.
        return { value: evaluateWhenRecord(text, values, types) };
      }
      const resolveMode = mode === "strict" ? ResolveMode.STRICT_RAISE : ResolveMode.BINDING_NONE;
      return { value: dataForm(evalExpr(trees[0]!, resolve, { item, mode: resolveMode, types })) };
    }),
  );
}

// ---- value ----

/**
 * Run a `value` case: build `value` as `type` under the defaults `env` sets and compare the
 * stored value and its kind.
 */
async function runValue(case_: Case, ctx: CaseContext): Promise<void> {
  // Read inside the result, so a bad variable is raised as the case's error; scoped, not
  // installed, so neither a later case nor the caller reads the case's values.
  await results(case_["expect"] as Case, () =>
    withEnv(caseEnv(ctx), () => {
      const typedefs = (case_["typedefs"] ?? {}) as Record<string, unknown>;
      const typ = typeFor(case_["type"] as string, readTypedefs(typedefs));
      const typed = buildValueAs(typ, case_["value"]);
      // A file in the value is compared in data form.
      return { value: typed.toWire().value, kind: typed.kind };
    }),
  );
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
 * The `yian` entry: `../../src/cli/index` from this file, `.ts` from source and `.js` when
 * built.
 */
const CLI_ENTRY = fileURLToPath(
  new URL(`../../src/cli/index.${import.meta.url.endsWith(".ts") ? "ts" : "js"}`, import.meta.url),
);
// Read from the repository, not beside this file, so the built runner finds it too.
const CLI_SCHEMA = JSON.parse(
  readFileSync(join(findRoot(), "tests", "conformance", "schema", "cli.schema.json"), "utf8"),
) as Schema;
const FILLED = /sk-conformance-[0-9]+/g;

/**
 * Return `value`, a case's plain JSON, as JSON text with `, ` and `: ` separators, floats with a
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
    // Ends a command the case gave up on, and waits for it to exit, so the stub and the hook
    // outlive it.
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
      `stdout doesn't match tests/conformance/schema/cli.schema.json\n${reasons}--- stdout\n${stdout}`,
    );
  }
  return parseJson(text, "codec") as Value;
}

/**
 * Whether `path` exists, a dangling symlink included; false on any system error.
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
