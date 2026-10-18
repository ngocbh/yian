/**
 * Contracts for the transient observer events, a port of
 * `tests/engine/test_in_loop_event_types.py` and `tests/engine/test_run_resumed_event.py`,
 * plus each event's `type` discriminant and its snake_case JSON data form.
 */

import { describe, expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import * as events from "../../src/events.ts";
import {
  IN_LOOP_EVENTS,
  ModelCallFinished,
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
  { event: new RunStarted(), data: {} },
  { event: new RunResumed(), data: {} },
  { event: new RunSucceeded({ output: { greeting: "hi" } }), data: { output: { greeting: "hi" } } },
  { event: new RunSucceeded(), data: { output: null } },
  { event: new RunFailed({ failure: FAILURE }), data: { failure: FAILURE_DATA } },
];

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
    const classes = Object.values(events)
      .filter((value) => typeof value === "function" && value !== TokenUsage)
      .map((value) => (value as { name: string }).name)
      .sort();
    const sampled: string[] = [...new Set(SAMPLES.map(({ event }) => event.type))].sort();
    expect(sampled).toEqual(classes);
    expect(classes).toHaveLength(16);
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
