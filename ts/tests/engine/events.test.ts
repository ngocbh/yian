/**
 * Contracts for the transient observer events, a port of
 * `tests/engine/test_in_loop_event_types.py` and `tests/engine/test_run_resumed_event.py`,
 * plus each event's `type` discriminant, its snake_case JSON data form, and
 * `conformance/schema/events.schema.json`, which describes that form. TS-only: `ReplayDegraded`,
 * an in-loop observer, and `ModelCallRetried`, a retry outside a forward node.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { FAILURE_CODES, Failure, OriginRef } from "../../src/diagnostics.ts";
import { ERROR_KINDS } from "../../src/llm/errors.ts";
import * as events from "../../src/events.ts";
import {
  IN_LOOP_EVENTS,
  ModelCallFinished,
  type ModelCallPurpose,
  ModelCallRetried,
  ModelCallStarted,
  NodeAttemptResumed,
  NodeExpanded,
  NodeFailed,
  NodeRouted,
  NodeSkipped,
  NodeStarted,
  NodeSucceeded,
  ReplayDegraded,
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
import { lintSchema, loadSchema, type Schema, validate } from "../schema.ts";
import { importStatements } from "./source_scan.ts";

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
/** A model call's start time, 2026-10-20T14:02:11.123Z. */
const STARTED_AT = Date.UTC(2026, 9, 20, 14, 2, 11, 123);

const SAMPLES: readonly { event: RunEvent; data: Record<string, unknown> }[] = [
  { event: new NodeStarted({ nodeId: "n" }), data: { node_id: "n" } },
  {
    event: new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }),
    data: {
      node_id: "n",
      key: "text",
      chunk: "hi",
      final: false,
      n: null,
      index: null,
    },
  },
  {
    event: new StreamChunk({
      nodeId: "n",
      key: "text",
      chunk: "",
      final: true,
      n: 2,
      index: 4,
    }),
    data: { node_id: "n", key: "text", chunk: "", final: true, n: 2, index: 4 },
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
    event: new NodeAttemptResumed({ nodeId: "n", crashed: 2 }),
    data: { node_id: "n", crashed: 2 },
  },
  {
    event: new ModelCallStarted({ nodeId: "n", n: 1, purpose: "chat", startedAt: STARTED_AT }),
    data: {
      node_id: "n",
      n: 1,
      purpose: "chat",
      provider: null,
      model: null,
      started_at: "2026-10-20T14:02:11.123Z",
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      n: 3,
      purpose: "structured_native",
      provider: "anthropic",
      model: "m",
      usage: USAGE,
    }),
    data: {
      node_id: "n",
      n: 3,
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
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      n: 1,
      purpose: "chat",
      error: "TimeoutError",
    }),
    data: {
      node_id: "n",
      n: 1,
      purpose: "chat",
      provider: null,
      model: null,
      usage: null,
      error: "TimeoutError",
    },
  },
  {
    event: new ModelCallStarted({
      nodeId: "n",
      n: 2,
      purpose: "structured_fallback",
      provider: "openai",
      model: "m",
      startedAt: STARTED_AT,
    }),
    data: {
      node_id: "n",
      n: 2,
      purpose: "structured_fallback",
      provider: "openai",
      model: "m",
      started_at: "2026-10-20T14:02:11.123Z",
    },
  },
  {
    event: new ModelCallFinished({
      nodeId: "n",
      n: 1,
      purpose: "structured_native",
      usage: new TokenUsage({ inputTokens: 0, outputTokens: 0, totalTokens: 0 }),
      error: "ValidationError",
    }),
    data: {
      node_id: "n",
      n: 1,
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
    },
  },
  {
    event: new ToolCallStarted({
      nodeId: "n",
      n: 1,
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
      arguments: { query: "q", Raw_Key: 1 },
    }),
    data: {
      node_id: "n",
      n: 1,
      call_index: 0,
      tool_call_id: "call-1",
      name: "lookup",
      arguments: { query: "q", Raw_Key: 1 },
    },
  },
  {
    event: new ToolCallFinished({
      nodeId: "n",
      n: 1,
      callIndex: 1,
      toolCallId: "call-2",
      name: "lookup",
      error: "interrupted",
    }),
    data: {
      node_id: "n",
      n: 1,
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
      n: 1,
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
      result: "ok",
    }),
    data: {
      node_id: "n",
      n: 1,
      call_index: 0,
      tool_call_id: "call-1",
      name: "lookup",
      result: "ok",
      error: null,
    },
  },
  {
    event: new ReplayDegraded({ nodeId: "n", reason: "other_account" }),
    data: { node_id: "n", reason: "other_account" },
  },
  {
    event: new ModelCallRetried({
      nodeId: "agent",
      consumer: "backward",
      attempt: 1,
      error: { kind: "rate_limit", message: "openai returned HTTP 429 (rate_limit): slow down" },
      retryIn: 0.5,
    }),
    data: {
      node_id: "agent",
      consumer: "backward",
      attempt: 1,
      error: { kind: "rate_limit", message: "openai returned HTTP 429 (rate_limit): slow down" },
      retry_in: 0.5,
    },
  },
  {
    event: new ModelCallRetried({
      nodeId: "__memory__",
      consumer: "memory:notes",
      attempt: 2,
      error: { kind: "server", message: "" },
      retryIn: 0,
    }),
    data: {
      node_id: "__memory__",
      consumer: "memory:notes",
      attempt: 2,
      error: { kind: "server", message: "" },
      retry_in: 0,
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
      n: 1,
      purpose: "chat",
      provider: "anthropic",
      model: "m",
      usage,
    });
    expect(finished.error).toBeNull();
    expect(
      new ModelCallStarted({ nodeId: "n", n: 1, purpose: "chat", startedAt: STARTED_AT }).provider,
    ).toBeNull();
    const started = new ToolCallStarted({
      nodeId: "n",
      n: 1,
      callIndex: 0,
      toolCallId: "call-1",
      name: "lookup",
    });
    expect(started.arguments).toBeNull();
    const tool = new ToolCallFinished({
      nodeId: "n",
      n: 1,
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

  test("IN_LOOP_EVENTS names exactly the five observers, not ModelCallRetried", () => {
    expect(IN_LOOP_EVENTS).toEqual([
      ModelCallStarted,
      ModelCallFinished,
      ToolCallStarted,
      ToolCallFinished,
      ReplayDegraded,
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
    expect(EVENT_CLASS_NAMES).toHaveLength(18);
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
      n: 1,
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

  test("type stays out of the JSON form (plain samples only: no bigint or infinity)", () => {
    // Not a serialization recipe: JSON.stringify turns infinity into null and throws on a bigint.
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

const SCHEMA = loadSchema("events.schema.json");
/** The events tests refuse any key a data form doesn't list. */
const CLOSED = { closed: true };

/** Every value of `ModelCallPurpose`; the type makes the key set exhaustive and exact. */
const PURPOSES: Record<ModelCallPurpose, null> = {
  chat: null,
  structured_native: null,
  structured_fallback: null,
};

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
    expect(lintSchema(SCHEMA, SCHEMA)).toEqual([]);
  });

  test("the lint rejects an unknown keyword and an unresolved $ref", () => {
    expect(lintSchema(SCHEMA, { requried: ["a"] })).toEqual(["$.requried: unknown keyword"]);
    expect(lintSchema(SCHEMA, { properties: { a: { $ref: "#/$defs/nope" } } })).toEqual([
      "$.properties.a.$ref: unresolved",
    ]);
  });

  test("declares draft 2020-12 and yian.events.v1", () => {
    expect(SCHEMA.$schema).toBe("https://json-schema.org/draft/2020-12/schema");
    expect(SCHEMA.title).toBe("yian.events.v1");
  });

  test("every sample event validates as {name, data}", () => {
    for (const { event } of SAMPLES) {
      expect(validate(SCHEMA, def(event.type), event.toJSON(), CLOSED)).toEqual([]);
      expect(validate(SCHEMA, SCHEMA, { name: event.type, data: event.toJSON() }, CLOSED)).toEqual([]);
    }
  });

  const modelCallFinished = {
    node_id: "n",
    n: 1,
    purpose: "chat",
    provider: null,
    model: null,
    usage: null,
    error: null,
  };

  const modelCallRetried = {
    node_id: "agent",
    consumer: "backward",
    attempt: 1,
    error: { kind: "timeout", message: "" },
    retry_in: 1.5,
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
      "n: 0",
      "ModelCallFinished",
      { ...modelCallFinished, n: 0 },
      ["$.n: 0 is below 1"],
    ],
    [
      "a retry's attempt: 0",
      "ModelCallRetried",
      { ...modelCallRetried, attempt: 0 },
      ["$.attempt: 0 is below 1"],
    ],
    [
      "retry_in: -1",
      "ModelCallRetried",
      { ...modelCallRetried, retry_in: -1 },
      ["$.retry_in: -1 is below 0"],
    ],
    [
      "error kind: nope",
      "ModelCallRetried",
      { ...modelCallRetried, error: { kind: "nope", message: "" } },
      ['$.error.kind: "nope" is not one of the enum'],
    ],
    [
      "an error without its message",
      "ModelCallRetried",
      { ...modelCallRetried, error: { kind: "timeout" } },
      ["$.error: missing message"],
    ],
    [
      "failure code: nope",
      "NodeFailed",
      { failure: { ...FAILURE_DATA, code: "nope" } },
      ['$.failure.code: "nope" is not one of the enum'],
    ],
  ] as const)("rejects %s", (_, name, data, problems) => {
    expect(validate(SCHEMA, def(name), data, CLOSED)).toEqual([...problems]);
    expect(validate(SCHEMA, SCHEMA, { name, data }, CLOSED)).toEqual(["$: 0 oneOf branches match, not 1"]);
  });

  test("rejects an unknown event name", () => {
    expect(validate(SCHEMA, SCHEMA, { name: "NodeExploded", data: { node_id: "n" } }, CLOSED)).toEqual([
      "$: 0 oneOf branches match, not 1",
    ]);
  });

  test("minimum skips null where the type allows it", () => {
    const chunk = { node_id: "n", key: "k", chunk: "", final: false, index: null };
    expect(validate(SCHEMA, def("StreamChunk"), { ...chunk, n: null }, CLOSED)).toEqual([]);
    expect(validate(SCHEMA, def("StreamChunk"), { ...chunk, n: 0 }, CLOSED)).toEqual(["$.n: 0 is below 1"]);
    expect(validate(SCHEMA, def("StreamChunk"), { ...chunk, n: 1.5 }, CLOSED)).toEqual([
      "$.n: number is not integer or null",
    ]);
  });

  test("oneOf passes when exactly one branch passes", () => {
    const branches = [{ type: "string" }, { type: ["string", "null"] }];
    expect(validate(SCHEMA, { oneOf: branches }, null, CLOSED)).toEqual([]);
    expect(validate(SCHEMA, { oneOf: branches }, "a", CLOSED)).toEqual(["$: 2 oneOf branches match, not 1"]);
    expect(validate(SCHEMA, { oneOf: branches }, 1, CLOSED)).toEqual(["$: 0 oneOf branches match, not 1"]);
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
    const retryError = (def("ModelCallRetried").properties as Record<string, Schema>).error!;
    const defs = Object.values(SCHEMA.$defs as Record<string, Schema>);
    const objects = [...defs.filter((schema) => "properties" in schema), trace.items as Schema, retryError];
    expect(objects).toHaveLength(EVENT_CLASS_NAMES.length + 5);
    for (const schema of objects) {
      expect(schema.type).toBe("object");
      expect(schema.required).toEqual(Object.keys(schema.properties as Schema));
    }
  });

  test("the failure code, purpose and retry error kind enums are their TS vocabularies", () => {
    const failure = def("failure").properties as Record<string, Schema>;
    expect(failure.code!.enum).toEqual([...FAILURE_CODES]);
    const retryError = (def("ModelCallRetried").properties as Record<string, Schema>).error!;
    expect((retryError.properties as Record<string, Schema>).kind!.enum).toEqual([...ERROR_KINDS]);
    for (const name of ["ModelCallStarted", "ModelCallFinished"]) {
      const properties = def(name).properties as Record<string, Schema>;
      expect(properties.purpose!.enum).toEqual(Object.keys(PURPOSES));
    }
  });
});

describe("the events module", () => {
  test('imports only "./diagnostics.ts"', () => {
    // The observer vocabulary sits just above the diagnostic value spine.
    const path = fileURLToPath(new URL("../../src/events.ts", import.meta.url));
    const specifiers = importStatements(readFileSync(path, "utf8")).map((i) => i.specifier);
    expect(specifiers).toEqual(["./diagnostics.ts"]);
  });
});
