/**
 * Late steers: a steer queued when the model gives its final answer gets one more model call;
 * every reply counts toward max_tool_iterations, and at the cap the answer stands and the steers
 * stay queued. TS-only: Python yian has no steer inbox.
 */
import { expect, test } from "bun:test";

import type { ChatModel, EventStream } from "../../src/llm/events.ts";
import { type Request, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { type AgentRunContext, DEFAULT_SYSTEM, ToolIterationLimitError } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output, SteerInbox } from "../../src/nodes/base.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { callReply, serving, textReply } from "./run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A steer inbox double: `queue` is what a read takes; `reads` logs each read's values, `null` for none. */
function inboxDouble(queue: string[] = []) {
  const reads: (readonly unknown[] | null)[] = [];
  let seq = 100;
  const inbox: SteerInbox = {
    inbox: "notes",
    queued: () => queue.length > 0,
    read: () => {
      if (queue.length === 0) {
        reads.push(null);
        return null;
      }
      const values = queue.splice(0);
      reads.push(values);
      return { n: 0, from: values.map(() => seq++), values };
    },
  };
  return { inbox, queue, reads };
}

/** A scripted model that runs `during(i)` while it answers its `i`th request (from 1), as a steer sent mid-reply. */
function hooked(model: ScriptedModel, during: (request: number) => void): ChatModel & { requests: Request[] } {
  return {
    provider: model.provider,
    model: model.model,
    requests: model.requests,
    stream(request: Request, options?: { signal?: AbortSignal }): EventStream {
      const stream = model.stream(request, options);
      during(model.requests.length);
      return stream;
    },
  };
}

/** A lookup tool returning `found <topic>`. */
const lookup = () =>
  new FunctionTool({
    name: "lookup",
    description: "Look up a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `found ${String(args["topic"])}`,
  });

/** A direct loop context: agent `chat` prompted "go", granted `lookup` with `withLookup`. */
function loopContext(model: ChatModel, extra: Partial<AgentRunContext> & { withLookup?: boolean } = {}): AgentRunContext {
  const { withLookup = false, ...rest } = extra;
  return {
    nodeId: "chat",
    messages: [userMessage("go")],
    promptText: "go",
    tools: withLookup ? ["lookup"] : [],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: withLookup ? new ToolSet({ lookup: lookup() }) : null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
    ...rest,
  };
}

/** Each message of `request` as `[role, text]`. */
const shown = (request: Request) =>
  request.messages.map((message) => [message.role, message.parts.map((part) => (part as TextPart).text ?? "").join("|")]);

/** Run the loop and return its output value. */
const output = async (ctx: AgentRunContext) => ((await runAgent(ctx)) as Output).value;

test("a steer queued while the final reply streams gets one more model call", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) => {
    if (i === 1) queue.push("use 2024");
  });
  expect(await output(loopContext(model, { steerInbox: inbox, withLookup: true }))).toBe("final");
  expect(model.requests).toHaveLength(2);
  expect(shown(model.requests[1]!).slice(-2)).toEqual([
    ["assistant", "draft"],
    ["user", "use 2024"],
  ]);
  expect(reads).toEqual([null, ["use 2024"]]);
});

test("no steer queued at the final answer: one model call, no read after it", async () => {
  const { inbox, reads } = inboxDouble();
  const model = new ScriptedModel([textReply("final")], CONFIG);
  expect(await output(loopContext(model, { steerInbox: inbox, withLookup: true }))).toBe("final");
  expect(model.requests).toHaveLength(1);
  expect(reads).toEqual([null]);
});

test("a late steer on a tool-free agent works the same", async () => {
  const { inbox, queue } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("draft"), textReply("final")], CONFIG), (i) => {
    if (i === 1) queue.push("shorter");
  });
  expect(await output(loopContext(model, { steerInbox: inbox }))).toBe("final");
  expect(shown(model.requests[1]!).at(-1)).toEqual(["user", "shorter"]);
});

test("replies after a late steer count toward the cap", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("one"), textReply("two")], CONFIG), (i) => {
    queue.push(`steer ${i}`);
  });
  expect(await output(loopContext(model, { steerInbox: inbox, maxToolIterations: 2 }))).toBe("two");
  expect(model.requests).toHaveLength(2);
  expect(queue).toEqual(["steer 2"]);
  expect(reads).toEqual([null, ["steer 1"]]);
});

test("at the cap the answer stands and nothing is read", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const model = hooked(new ScriptedModel([textReply("one")], CONFIG), () => queue.push("late"));
  expect(await output(loopContext(model, { steerInbox: inbox, maxToolIterations: 1 }))).toBe("one");
  expect(queue).toEqual(["late"]);
  expect(reads).toEqual([null]);
});

test("-1 is unbounded", async () => {
  const { inbox, queue } = inboxDouble();
  const replies = ["a", "b", "c", "d"].map((text) => textReply(text));
  const model = hooked(new ScriptedModel(replies, CONFIG), (i) => {
    if (i <= 3) queue.push(`steer ${i}`);
  });
  expect(await output(loopContext(model, { steerInbox: inbox }))).toBe("d");
  expect(model.requests).toHaveLength(4);
});

test("a reply with tool calls at the cap still fails", async () => {
  const { inbox } = inboxDouble();
  const model = new ScriptedModel([callReply(["lookup", { topic: "a" }, "c1"])], CONFIG);
  const failure = await runAgent(loopContext(model, { steerInbox: inbox, withLookup: true, maxToolIterations: 1 })).then(
    () => null,
    (error: unknown) => error,
  );
  expect(failure).toBeInstanceOf(ToolIterationLimitError);
});

test("an interrupt that lands during the final reply beats a queued steer: the answer stands, the steer stays", async () => {
  const { inbox, queue, reads } = inboxDouble();
  const interrupt = new AbortController();
  const model = hooked(new ScriptedModel([textReply("final")], CONFIG), () => {
    queue.push("late");
    interrupt.abort();
  });
  const steering = { interruptSignal: () => interrupt.signal };
  expect(await output(loopContext(model, { steerInbox: inbox, steering }))).toBe("final");
  expect(model.requests).toHaveLength(1);
  expect(queue).toEqual(["late"]);
  expect(reads).toEqual([null]);
});
