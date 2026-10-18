/**
 * Contracts for the transient observer events, a port of
 * `tests/engine/test_in_loop_event_types.py` and `tests/engine/test_run_resumed_event.py`,
 * plus each event's `type` discriminant, its snake_case JSON data form, and
 * `conformance/schema/events.schema.json`, which describes that form.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

import { FAILURE_CODES, Failure, OriginRef } from "../../src/diagnostics.ts";
import * as events from "../../src/events.ts";
import {
  IN_LOOP_EVENTS,
  ModelCallFinished,
  type ModelCallPurpose,
  ModelCallStarted,
  NodeAttemptResumed,
  NodeExpanded,
  NodeFailed,
  NodeRouted,
  NodeSkipped,
  NodeStarted,
  NodeSucceeded,
  RunFailed,
  type RunEvent,
  RunResumed,
  RunStarted,
  RunSucceeded,
  StreamChunk,
  TokenUsage,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";

const FAILURE = new Failure({
  code: "node.execute",
  message: "boom",
  nodeId: "n",
  trace: [new OriginRef("flow"), new OriginRef("flow/n")],
  causeType: "Error",
});

const FAILURE_DATA = {
  code: "node.execute",
  message: "boom",
  node_id: "n",
  trace: [{ key: "flow" }, { key: "flow/n" }],
  cause_type: "Error",
  traceback: null,
};

const USAGE = new TokenUsage({
  inputTokens: 3,
  outputTokens: 2,
  totalTokens: 5,
  cacheReadTokens: 1,
  cacheCreationTokens: 0,
  reasoningTokens: 4,
});

/** One sample per event class with its exact data form, keys in field order. */
const SAMPLES: readonly { event: RunEvent; data: Record<string, unknown> }[] = [
  { event: new NodeStarted({ nodeId: "n" }), data: { node_id: "n" } },
  {
    event: new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }),
    data: {
      node_id: "n",
      key: "text",
      chunk: "hi",
      final: false,
      exchange_id: null,
      attempt: null,
    },
  },
  {
    event: new StreamChunk({
      nodeId: "n",
      key: "text",
      chunk: "",
      final: true,
      exchangeId: "x",
      attempt: 2,
    }),
    data: { node_id: "n", key: "text", chunk: "", final: true, exchange_id: "x", attempt: 2 },
  },
  {
    event: new NodeSucceeded({ nodeId: "n", output: "hi" }),
    data: { node_id: "n", output: "hi", commit_as: null },
  },
  {
    event: new NodeSucceeded({ nodeId: "n", output: { a: [1, null] }, commitAs: "t" }),
    data: { node_id: "n", output: { a: [1, null] }, commit_as: "t" },
  },
  {
    event: new NodeRouted({ nodeId: "c", handle: "#else" }),
    data: { node_id: "c", handle: "#else" },
  },
  { event: new NodeSkipped({ nodeId: "s" }), data: { node_id: "s" } },
  { event: new NodeFailed({ failure: FAILURE }), data: { failure: FAILURE_DATA } },
  {
    event: new NodeExpanded({
      nodeId: "loop",
      grow: { seed: [{ n: 0 }, 0], prune: new Set(["b", "a", "B"]) },
    }),
    data: { node_id: "loop", grow: { seed: [{ n: 0 }, 0], prune: ["B", "a", "b"] } },
  },
  {
    event: new NodeAttemptResumed({ nodeId: "n", interrupted: 2 }),
    data: { node_id: "n", interrupted: 2 },
  },
  {
    event: new ModelCallStarted({ nodeId: "n", exchangeId: "x", purpose: "chat", attempt: 1 }),
    data: {
      node_id: "n",
      exchange_id: "x",
      purpose: "chat",
      provider: null,
      model: null,
      attempt: 1,
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      exchangeId: "x",
      purpose: "structured_native",
      provider: "anthropic",
      model: "m",
      usage: USAGE,
      attempt: 3,
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      purpose: "structured_native",
      provider: "anthropic",
      model: "m",
      usage: {
        input_tokens: 3,
        output_tokens: 2,
        total_tokens: 5,
        cache_read_tokens: 1,
        cache_creation_tokens: 0,
        reasoning_tokens: 4,
      },
      error: null,
      attempt: 3,
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      exchangeId: "x",
      purpose: "history_summary",
      error: "TimeoutError",
      attempt: 1,
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      purpose: "history_summary",
      provider: null,
      model: null,
      usage: null,
      error: "TimeoutError",
      attempt: 1,
    },
  },
  {
    event: new ModelCallStarted({
      nodeId: "n",
      exchangeId: "x",
      purpose: "structured_fallback",
      provider: "openai",
      model: "m",
      attempt: 2,
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      purpose: "structured_fallback",
      provider: "openai",
      model: "m",
      attempt: 2,
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      exchangeId: "x",
      purpose: "structured_native",
      usage: new TokenUsage({ inputTokens: 0, outputTokens: 0, totalTokens: 0 }),
      error: "ValidationError",
      attempt: 1,
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      purpose: "structured_native",
      provider: null,
      model: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        cache_read_tokens: null,
        cache_creation_tokens: null,
        reasoning_tokens: null,
      },
      error: "ValidationError",
      attempt: 1,
    },
  },
  {
    event: new ToolCallStarted({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
      arguments: { query: "q", Raw_Key: 1 },
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      call_index: 0,
      tool_call_id: "call-1",
      name: "lookup",
      arguments: { query: "q", Raw_Key: 1 },
    },
  },
  {
    event: new ToolCallFinished({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 1,
      toolCallId: "call-2",
      name: "lookup",
      error: "interrupted",
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      call_index: 1,
      tool_call_id: "call-2",
      name: "lookup",
      result: null,
      error: "interrupted",
    },
  },
  {
    event: new ToolCallFinished({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
      result: "ok",
    }),
    data: {
      node_id: "n",
      exchange_id: "x",
      call_index: 0,
      tool_call_id: "call-1",
      name: "lookup",
      result: "ok",
      error: null,
    },
  },
  { event: new RunStarted(), data: {} },
  { event: new RunResumed(), data: {} },
  { event: new RunSucceeded({ output: { greeting: "hi" } }), data: { output: { greeting: "hi" } } },
  { event: new RunSucceeded(), data: { output: null } },
  { event: new RunFailed({ failure: FAILURE }), data: { failure: FAILURE_DATA } },
  {
    event: new RunFailed({
      failure: new Failure({
        code: "graph.terminal_unreachable",
        message: "",
        nodeId: null,
        traceback: "Error: x\n    at f",
      }),
    }),
    data: {
      failure: {
        code: "graph.terminal_unreachable",
        message: "",
        node_id: null,
        trace: [],
        cause_type: null,
        traceback: "Error: x\n    at f",
      },
    },
  },
];

/** The exported event class names, sorted. */
const EVENT_CLASS_NAMES = Object.entries(events)
  .filter(([, value]) => typeof value === "function" && value !== TokenUsage)
  .map(([name]) => name)
  .sort();

/** Data-form keys whose values the flow produced; their insides are not yian's structures. */
const FLOW_VALUE_KEYS = new Set(["output", "arguments", "seed"]);

/** Return every key in `value` at every depth, not walking into flow values. */
function yianKeys(value: unknown): string[] {
  if (Array.isArray(value)) return value.flatMap(yianKeys);
  if (value === null || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, inner]) =>
    FLOW_VALUE_KEYS.has(key) ? [key] : [key, ...yianKeys(inner)],
  );
}

describe("in-loop observers", () => {
  test("shapes and defaults", () => {
    const usage = new TokenUsage({ inputTokens: 3, outputTokens: 2, totalTokens: 5 });
    expect([usage.cacheReadTokens, usage.cacheCreationTokens, usage.reasoningTokens]).toEqual([
      null,
      null,
      null,
    ]);
    const finished = new ModelCallFinished({
      nodeId: "n",
      exchangeId: "x",
      purpose: "chat",
      provider: "anthropic",
      model: "m",
      usage,
      attempt: 1,
    });
    expect(finished.error).toBeNull();
    expect(
      new ModelCallStarted({ nodeId: "n", exchangeId: "x", purpose: "chat", attempt: 1 }).provider,
    ).toBeNull();
    const started = new ToolCallStarted({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
    });
    expect(started.arguments).toBeNull();
    const tool = new ToolCallFinished({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
      result: "ok",
    });
    expect(tool.error).toBeNull();
    if (false) {
      // @ts-expect-error fields are readonly
      usage.inputTokens = 1;
      // @ts-expect-error fields are readonly
      started.nodeId = "x";
      // @ts-expect-error fields are readonly
      finished.usage = null;
    }
  });

  test("IN_LOOP_EVENTS names exactly the four observers", () => {
    expect(IN_LOOP_EVENTS).toEqual([
      ModelCallStarted,
      ModelCallFinished,
      ToolCallStarted,
      ToolCallFinished,
    ]);
  });
});

describe("run events", () => {
  test("RunResumed needs no fields and is not RunStarted", () => {
    const resumed = new RunResumed();
    expect(resumed).toBeInstanceOf(RunResumed);
    expect(resumed.type).not.toBe(new RunStarted().type);
  });

  test("RunResumed and RunStarted are not terminals", () => {
    for (const lead of [new RunResumed(), new RunStarted()]) {
      expect(lead instanceof RunSucceeded).toBe(false);
      expect(lead instanceof RunFailed).toBe(false);
    }
  });
});

describe("type discriminant", () => {
  test("every event's type is its class name", () => {
    for (const { event } of SAMPLES) {
      expect(event.constructor.name).toBe(event.type);
    }
  });

  test("the samples cover every event class", () => {
    const sampled: string[] = [...new Set(SAMPLES.map(({ event }) => event.type))].sort();
    expect(sampled).toEqual(EVENT_CLASS_NAMES);
    expect(EVENT_CLASS_NAMES).toHaveLength(16);
  });

  test("TokenUsage is not an event and has no type", () => {
    expect("type" in USAGE).toBe(false);
  });
});

describe("toJSON", () => {
  test("each event gives its exact data form, keys in field order at every depth", () => {
    for (const { event, data } of SAMPLES) {
      const form = event.toJSON();
      expect(form).toEqual(data);
      expect(JSON.stringify(form)).toBe(JSON.stringify(data));
    }
  });

  test("TokenUsage gives its six snake_case keys, null kept", () => {
    expect(new TokenUsage({ inputTokens: 3, outputTokens: 2, totalTokens: 5 }).toJSON()).toEqual({
      input_tokens: 3,
      output_tokens: 2,
      total_tokens: 5,
      cache_read_tokens: null,
      cache_creation_tokens: null,
      reasoning_tokens: null,
    });
  });

  test("flow values are passed through untouched", () => {
    const output = { big: 10n, nested: [Infinity] };
    expect(new NodeSucceeded({ nodeId: "n", output }).toJSON().output).toBe(output);
    expect(new RunSucceeded({ output: 10n }).toJSON().output).toBe(10n);
    const args = { n: 1n };
    const started = new ToolCallStarted({
      nodeId: "n",
      exchangeId: "x",
      callIndex: 0,
      toolCallId: "c",
      name: "t",
      arguments: args,
    });
    expect(started.toJSON().arguments).toBe(args);
    const seed = [{ n: 0n }, 0];
    const expanded = new NodeExpanded({ nodeId: "l", grow: { seed, prune: new Set() } });
    expect(expanded.toJSON().grow.seed).toBe(seed);
  });

  test("JSON.stringify writes the data form, so type never reaches JSON", () => {
    for (const { event } of SAMPLES) {
      expect(JSON.parse(JSON.stringify(event))).toEqual(event.toJSON());
    }
  });

  test("every key of yian's own structures is snake_case; flow values are not walked", () => {
    const keys = SAMPLES.flatMap(({ event }) => yianKeys(event.toJSON()));
    expect(keys).toContain("tool_call_id");
    expect(keys).toContain("cause_type");
    expect(keys).toContain("cache_read_tokens");
    expect(keys).not.toContain("Raw_Key");
    expect(keys.filter((key) => !/^[a-z][a-z0-9_]*$/.test(key))).toEqual([]);
  });
});

// --------------------------------------------------------------------------- //
// The events schema
// --------------------------------------------------------------------------- //

/** A JSON Schema subschema, as parsed. */
type Schema = Record<string, unknown>;

const SCHEMA = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../../conformance/schema/events.schema.json", import.meta.url)),
    "utf8",
  ),
) as Schema;

/** The keywords the validator implements. */
const KEYWORDS = new Set([
  "$ref",
  "type",
  "const",
  "enum",
  "required",
  "properties",
  "items",
  "oneOf",
  "minimum",
]);
/** Annotations the validator ignores. */
const ANNOTATIONS = new Set(["$schema", "title", "description", "$defs"]);
const TYPES = new Set(["null", "boolean", "integer", "number", "string", "array", "object"]);

/** Every value of `ModelCallPurpose`; the type makes the key set exhaustive and exact. */
const PURPOSES: Record<ModelCallPurpose, null> = {
  chat: null,
  structured_native: null,
  structured_fallback: null,
  history_summary: null,
};

/** Return the JSON Schema type names `value` has (an integer is also a number). */
function jsonTypes(value: unknown): string[] {
  if (value === null) return ["null"];
  if (Array.isArray(value)) return ["array"];
  if (typeof value === "number") return Number.isInteger(value) ? ["integer", "number"] : ["number"];
  return [typeof value];
}

/** Return the `$defs` entry a `#/$defs/<name>` reference names, or `undefined`. */
function resolveRef(ref: unknown): Schema | undefined {
  if (typeof ref !== "string" || !ref.startsWith("#/$defs/")) return undefined;
  const defs = SCHEMA.$defs as Record<string, Schema>;
  const name = ref.slice("#/$defs/".length);
  return Object.hasOwn(defs, name) ? defs[name] : undefined;
}

/**
 * Validate `value` against `schema` for the keywords in `KEYWORDS`, returning
 * `path: problem` strings. Stricter than JSON Schema in one way: a subschema with
 * `properties` rejects keys it doesn't list.
 */
function validate(schema: Schema, value: unknown, path = "$"): string[] {
  const problems: string[] = [];
  const types = jsonTypes(value);
  if ("$ref" in schema) problems.push(...validate(resolveRef(schema.$ref)!, value, path));
  if ("type" in schema) {
    const allowed = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((type) => allowed.includes(type))) {
      return [...problems, `${path}: ${types[0]} is not ${allowed.join(" or ")}`];
    }
  }
  if ("const" in schema && !isDeepStrictEqual(value, schema.const)) {
    problems.push(`${path}: ${JSON.stringify(value)} is not ${JSON.stringify(schema.const)}`);
  }
  if ("enum" in schema) {
    const members = schema.enum as unknown[];
    if (!members.some((member) => isDeepStrictEqual(value, member))) {
      problems.push(`${path}: ${JSON.stringify(value)} is not one of the enum`);
    }
  }
  if ("minimum" in schema && typeof value === "number" && value < (schema.minimum as number)) {
    problems.push(`${path}: ${value} is below ${schema.minimum}`);
  }
  if (types[0] === "object") {
    const object = value as Record<string, unknown>;
    for (const key of (schema.required as string[] | undefined) ?? []) {
      if (!Object.hasOwn(object, key)) problems.push(`${path}: missing ${key}`);
    }
    if ("properties" in schema) {
      const properties = schema.properties as Record<string, Schema>;
      for (const [key, inner] of Object.entries(object)) {
        if (Object.hasOwn(properties, key)) {
          problems.push(...validate(properties[key]!, inner, `${path}.${key}`));
        } else {
          problems.push(`${path}: undeclared key ${key}`);
        }
      }
    }
  }
  if (types[0] === "array" && "items" in schema) {
    (value as unknown[]).forEach((item, index) => {
      problems.push(...validate(schema.items as Schema, item, `${path}[${index}]`));
    });
  }
  if ("oneOf" in schema) {
    const passing = (schema.oneOf as Schema[]).filter(
      (branch) => validate(branch, value, path).length === 0,
    ).length;
    if (passing !== 1) problems.push(`${path}: ${passing} oneOf branches match, not 1`);
  }
  return problems;
}

/** Return `path: problem` strings for keywords outside the validator's list and bad `$ref`s. */
function lintSchema(schema: unknown, path = "$"): string[] {
  if (schema === null || typeof schema !== "object" || Array.isArray(schema)) {
    return [`${path}: not a schema object`];
  }
  const problems: string[] = [];
  for (const [key, inner] of Object.entries(schema)) {
    const at = `${path}.${key}`;
    if (!KEYWORDS.has(key) && !ANNOTATIONS.has(key)) problems.push(`${at}: unknown keyword`);
    if (key === "$ref" && resolveRef(inner) === undefined) problems.push(`${at}: unresolved`);
    if (key === "type") {
      const types = Array.isArray(inner) ? inner : [inner];
      for (const type of types) if (!TYPES.has(type)) problems.push(`${at}: unknown type`);
    }
    if (key === "items") problems.push(...lintSchema(inner, at));
    if (key === "oneOf") {
      (inner as unknown[]).forEach((branch, index) => {
        problems.push(...lintSchema(branch, `${at}[${index}]`));
      });
    }
    if (key === "properties" || key === "$defs") {
      for (const [name, sub] of Object.entries(inner as Record<string, unknown>)) {
        problems.push(...lintSchema(sub, `${at}.${name}`));
      }
    }
  }
  return problems;
}

/** Return the `$defs` entry named `name`. */
function def(name: string): Schema {
  return (SCHEMA.$defs as Record<string, Schema>)[name]!;
}

/** Return the event names the envelope's `oneOf` lists, in order. */
function oneOfNames(): string[] {
  return (SCHEMA.oneOf as Schema[]).map(
    (branch) => ((branch.properties as Record<string, Schema>).name as Schema).const as string,
  );
}

describe("events schema", () => {
  test("uses only the keywords the validator knows, and every $ref resolves", () => {
    expect(lintSchema(SCHEMA)).toEqual([]);
  });

  test("the lint rejects an unknown keyword and an unresolved $ref", () => {
    expect(lintSchema({ requried: ["a"] })).toEqual(["$.requried: unknown keyword"]);
    expect(lintSchema({ properties: { a: { $ref: "#/$defs/nope" } } })).toEqual([
      "$.properties.a.$ref: unresolved",
    ]);
  });

  test("declares draft 2020-12 and yian.events.v1", () => {
    expect(SCHEMA.$schema).toBe("https://json-schema.org/draft/2020-12/schema");
    expect(SCHEMA.title).toBe("yian.events.v1");
  });

  test("every sample event validates as {name, data}", () => {
    for (const { event } of SAMPLES) {
      expect(validate(def(event.type), event.toJSON())).toEqual([]);
      expect(validate(SCHEMA, { name: event.type, data: event.toJSON() })).toEqual([]);
    }
  });

  const modelCallFinished = {
    node_id: "n",
    exchange_id: "x",
    purpose: "chat",
    provider: null,
    model: null,
    usage: null,
    error: null,
    attempt: 1,
  };

  test.each([
    ["a missing required key", "NodeStarted", {}, ["$: missing node_id"]],
    ["an undeclared key", "NodeStarted", { node_id: "n", extra: 1 }, ["$: undeclared key extra"]],
    ["node_id: 1", "NodeStarted", { node_id: 1 }, ["$.node_id: integer is not string"]],
    [
      "purpose: other",
      "ModelCallFinished",
      { ...modelCallFinished, purpose: "other" },
      ['$.purpose: "other" is not one of the enum'],
    ],
    [
      "attempt: 0",
      "ModelCallFinished",
      { ...modelCallFinished, attempt: 0 },
      ["$.attempt: 0 is below 1"],
    ],
    [
      "failure code: nope",
      "NodeFailed",
      { failure: { ...FAILURE_DATA, code: "nope" } },
      ['$.failure.code: "nope" is not one of the enum'],
    ],
  ] as const)("rejects %s", (_, name, data, problems) => {
    expect(validate(def(name), data)).toEqual([...problems]);
    expect(validate(SCHEMA, { name, data })).toEqual(["$: 0 oneOf branches match, not 1"]);
  });

  test("rejects an unknown event name", () => {
    expect(validate(SCHEMA, { name: "NodeExploded", data: { node_id: "n" } })).toEqual([
      "$: 0 oneOf branches match, not 1",
    ]);
  });

  test("minimum skips null where the type allows it", () => {
    const chunk = { node_id: "n", key: "k", chunk: "", final: false, exchange_id: null };
    expect(validate(def("StreamChunk"), { ...chunk, attempt: null })).toEqual([]);
    expect(validate(def("StreamChunk"), { ...chunk, attempt: 0 })).toEqual([
      "$.attempt: 0 is below 1",
    ]);
    expect(validate(def("StreamChunk"), { ...chunk, attempt: 1.5 })).toEqual([
      "$.attempt: number is not integer or null",
    ]);
  });

  test("oneOf passes when exactly one branch passes", () => {
    const branches = [{ type: "string" }, { type: ["string", "null"] }];
    expect(validate({ oneOf: branches }, null)).toEqual([]);
    expect(validate({ oneOf: branches }, "a")).toEqual(["$: 2 oneOf branches match, not 1"]);
    expect(validate({ oneOf: branches }, 1)).toEqual(["$: 0 oneOf branches match, not 1"]);
  });

  test("event $defs, oneOf names and exported event classes are one set", () => {
    const defs = Object.keys(SCHEMA.$defs as Schema);
    const names = oneOfNames();
    expect(defs.filter((name) => /^[A-Z]/.test(name)).sort()).toEqual(EVENT_CLASS_NAMES);
    expect([...names].sort()).toEqual(EVENT_CLASS_NAMES);
    expect(new Set(names).size).toBe(names.length);
    expect(defs.filter((name) => !/^[A-Z]/.test(name)).sort()).toEqual([
      "failure",
      "grow",
      "token_usage",
      "value",
    ]);
    for (const branch of SCHEMA.oneOf as Schema[]) {
      const { name, data } = branch.properties as Record<string, Schema>;
      expect(data).toEqual({ $ref: `#/$defs/${name!.const}` });
    }
  });

  test("every object form requires each key it lists", () => {
    const trace = (def("failure").properties as Record<string, Schema>).trace!;
    const defs = Object.values(SCHEMA.$defs as Record<string, Schema>);
    const objects = [...defs.filter((schema) => "properties" in schema), trace.items as Schema];
    expect(objects).toHaveLength(EVENT_CLASS_NAMES.length + 4);
    for (const schema of objects) {
      expect(schema.type).toBe("object");
      expect(schema.required).toEqual(Object.keys(schema.properties as Schema));
    }
  });

  test("the failure code enum is FAILURE_CODES and the purpose enum is ModelCallPurpose", () => {
    const failure = def("failure").properties as Record<string, Schema>;
    expect(failure.code!.enum).toEqual([...FAILURE_CODES]);
    for (const name of ["ModelCallStarted", "ModelCallFinished"]) {
      const properties = def(name).properties as Record<string, Schema>;
      expect(properties.purpose!.enum).toEqual(Object.keys(PURPOSES));
    }
  });
});
