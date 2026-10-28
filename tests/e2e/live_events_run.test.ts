/**
 * End to end, a tool-calling AGENT publishes in-loop observers with token usage, strictly between
 * the node's `NodeStarted` and its outcome.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import {
  IN_LOOP_EVENTS,
  ModelCallFinished,
  ModelCallStarted,
  NodeStarted,
  NodeSucceeded,
  type RunEvent,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import { readRunLog } from "../../src/runtime/log.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { barrier, serving } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-live-events-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
afterEach(() => TOOL_REGISTRY.delete("lookup"));

const FLOW = `
id: live
name: live
llm_config: {provider: anthropic, model: claude-test}
input:
  topic: !str
nodes:
  writer:
    kind: agent
    tools: [lookup]
    input:
      topic: \${input.topic}
    output: str
    prompt: "research \${topic}"
output: \${writer.output}
`;

const PAIR = `
id: pair
name: pair
llm_config: {provider: anthropic, model: claude-test}
input:
  topic: !str
nodes:
  left:
    kind: agent
    tools: [lookup]
    input:
      topic: \${input.topic}
    output: str
    prompt: "left \${topic}"
  right:
    kind: agent
    tools: [lookup]
    input:
      topic: \${input.topic}
    output: str
    prompt: "right \${topic}"
output:
  left: \${left.output}
  right: \${right.output}
`;

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** One reply: `text`, then the `calls` (`[name, args, id]`), with `usage` when given. */
function reply(text: string, calls: [string, Record<string, string>, string][] = [], usage: Usage | null = null): Response {
  const parts = [
    ...(text === "" ? [] : [new TextPart(text)]),
    ...calls.map(([name, args, id]) => new ToolCallPart({ id, name, arguments: args })),
  ];
  return new Response({
    message: new Message({ role: "assistant", parts }),
    finishReason: calls.length > 0 ? "tool_calls" : "stop",
    usage,
  });
}

/** A usage report of `input` and `output` tokens, with `cacheRead` cached input tokens. */
const usage = (input: number, output: number, cacheRead: number | null = null) =>
  new Usage({ inputTokens: input, outputTokens: output, totalTokens: input + output, cacheReadTokens: cacheRead });

/** The writer's script: a `lookup` call, then the answer. */
const script = () => [
  reply("checking", [["lookup", { topic: "acme" }, "call-1"]], usage(10, 4, 6)),
  reply("acme looks fine", [], usage(25, 3)),
];

/** Register `lookup`, which awaits `before` when given and returns `found:<topic>`. */
function registerLookup(before?: () => Promise<void>): void {
  registerTool(
    "lookup",
    tool({
      name: "lookup",
      description: "Look one topic up.",
      parameters: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
      execute: async (args) => {
        if (before !== undefined) await before();
        return `found:${args.topic}`;
      },
    }),
  );
}

const isInLoop = (event: unknown) => IN_LOOP_EVENTS.some((cls) => event instanceof cls);
const nodeIdOf = (event: unknown) => (event as { nodeId?: string }).nodeId;

for (const numWorkers of [0, 2]) {
  test(`a tool-calling agent reports ordered events and usage (numWorkers ${numWorkers})`, async () => {
    registerLookup();
    const model = new ScriptedModel(script(), CONFIG);
    const timeline: RunEvent[] = [];
    const log = join(dir, `run-${numWorkers}.jsonl`);

    const result = await runFlow(await loadFlow(FLOW), { topic: "acme" }, {
      log,
      numWorkers,
      onEvent: (event) => void timeline.push(event),
      env: hostEnv(dir),
      models: serving(model),
    });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("acme looks fine");
    expect(model.requests).toHaveLength(2);

    const writer = timeline.filter((event) => nodeIdOf(event) === "writer");
    expect(writer[0]).toBeInstanceOf(NodeStarted);
    expect(writer.at(-1)).toBeInstanceOf(NodeSucceeded);
    expect(writer.slice(1, -1).map((event) => event.constructor)).toEqual([
      ModelCallStarted,
      ModelCallFinished,
      ToolCallStarted,
      ToolCallFinished,
      ModelCallStarted,
      ModelCallFinished,
    ]);
    // The second reply follows the first reply's step and its tool result's.
    const [first, second] = writer.filter((event) => event instanceof ModelCallStarted).map((event) => event.n);
    expect([first, second]).toEqual([1, 3]);
    expect(new Set(writer.filter((event) => event instanceof ToolCallStarted).map((event) => event.n))).toEqual(
      new Set([first!]),
    );

    const finished = writer.filter((event) => event instanceof ModelCallFinished);
    const sum = (field: "inputTokens" | "outputTokens" | "totalTokens") =>
      finished.reduce((total, event) => total + event.usage![field], 0);
    expect(sum("inputTokens")).toBe(35);
    expect(sum("outputTokens")).toBe(7);
    expect(sum("totalTokens")).toBe(42);
    expect(finished[0]!.usage!.cacheReadTokens).toBe(6);
    expect(new Set(finished.map((event) => `${event.provider}/${event.model}`))).toEqual(new Set(["anthropic/claude-test"]));
    const toolStarted = writer.find((event) => event instanceof ToolCallStarted)!;
    expect([toolStarted.name, toolStarted.arguments, toolStarted.toolCallId]).toEqual([
      "lookup",
      { topic: "acme" },
      "call-1",
    ]);

    expect(result.events).toEqual(timeline.filter((event) => !isInLoop(event)));
    const recorded = new Set(readRunLog(log).map((envelope) => envelope.record.constructor.name));
    for (const cls of IN_LOOP_EVENTS) expect(recorded.has(cls.name)).toBe(false);
  });
}

test("a run without callbacks is unchanged", async () => {
  registerLookup();
  const model = new ScriptedModel(script(), CONFIG);

  const result = await runFlow(await loadFlow(FLOW), { topic: "acme" }, { env: hostEnv(dir), models: serving(model) });

  expect(result.output).toBe("acme looks fine");
  expect(model.requests).toHaveLength(2);
  expect(result.events.some(isInLoop)).toBe(false);
});

test("parallel agents keep their own events in pooled runs", async () => {
  // Both AGENTs are inside a tool call at once.
  registerLookup(barrier(2));
  const model = new ScriptedModel(
    [
      reply("", [["lookup", { topic: "acme" }, "call-1"]]),
      reply("", [["lookup", { topic: "acme" }, "call-2"]]),
      reply("done"),
      reply("done"),
    ],
    CONFIG,
  );
  const timeline: RunEvent[] = [];

  const result = await runFlow(await loadFlow(PAIR), { topic: "acme" }, {
    numWorkers: 2,
    onEvent: (event) => void timeline.push(event),
    env: hostEnv(dir),
    models: serving(model),
  });

  expect(result.output).toEqual({ left: "done", right: "done" });
  for (const nodeId of ["left", "right"]) {
    const own = [...timeline.keys()].filter((i) => nodeIdOf(timeline[i]) === nodeId);
    const started = own.find((i) => timeline[i] instanceof NodeStarted)!;
    const done = own.find((i) => timeline[i] instanceof NodeSucceeded)!;
    const inLoop = own.filter((i) => isInLoop(timeline[i]));
    expect(inLoop).toHaveLength(6);
    expect(started).toBeLessThan(inLoop[0]!);
    expect(inLoop.at(-1)!).toBeLessThan(done);
    // Each node numbers its own steps; node id + n names the step.
    expect(inLoop.map((i) => (timeline[i] as { n: number }).n)).toEqual([1, 1, 1, 1, 3, 3]);
  }
  // The barrier holds both tool calls open, so the two nodes' events interleave.
  const toolStarted = [...timeline.keys()].filter((i) => timeline[i] instanceof ToolCallStarted);
  const toolFinished = [...timeline.keys()].filter((i) => timeline[i] instanceof ToolCallFinished);
  expect(Math.max(...toolStarted)).toBeLessThan(Math.min(...toolFinished));
});
