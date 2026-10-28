/**
 * Every `FunctionTool` receives a `ToolContext` per call, including the call's abort signal.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import {
  type AskRequest,
  invokeTool,
  invokeToolTrusted,
  registerTool,
  TOOL_REGISTRY,
  ToolSet,
} from "../../src/tools/registry.ts";

const HEX32 = /^[0-9a-f]{32}$/;

let saved: [string, FunctionTool][] = [];
let seen: [ToolContext<any>, Record<string, unknown>][] = [];

// Keep process-global registrations isolated between tests.
beforeEach(() => {
  saved = [...TOOL_REGISTRY];
  seen = [];
});
afterEach(() => {
  TOOL_REGISTRY.clear();
  for (const [id, tool] of saved) TOOL_REGISTRY.set(id, tool);
});

class Db {}

/** Build the `place` tool under `name`: it records each call and returns `<symbol>:<qty>`. */
function place(name = "place"): FunctionTool {
  return new FunctionTool({
    name,
    description: "Place an order.",
    parametersJsonSchema: {
      type: "object",
      properties: { symbol: { type: "string" }, qty: { type: "integer", default: 1 } },
      required: ["symbol"],
    },
    execute: (args, ctx) => {
      const call = { symbol: args.symbol, qty: args.qty ?? 1 };
      seen.push([ctx, call]);
      return `${call.symbol}:${call.qty}`;
    },
  });
}

describe("the context a tool receives", () => {
  test("unbound toolset passes a fresh context", async () => {
    const tools = new ToolSet({ place: place() });

    expect(await tools.invoke("place", { symbol: "X" }, { node: "n" })).toBe("X:1");
    await tools.invoke("place", { symbol: "Y" }, { node: "n" });

    const [[first, args], [second]] = seen as [[ToolContext<any>, unknown], [ToolContext<any>, unknown]];
    expect(args).toEqual({ symbol: "X", qty: 1 });
    expect(first.toolName).toBe("place");
    expect(first.node).toBe("n");
    expect(first.runId).toBe("");
    expect(first.attempt).toBe(1);
    expect(first.context).toBeUndefined();
    expect(first.callId).toMatch(HEX32);
    expect(second.callId).toMatch(HEX32);
    expect(first.callId).not.toBe(second.callId);
  });

  test("run bound toolset passes run id context and a stable call id", async () => {
    const db = new Db();
    const tools = new ToolSet({ place: place() }, { context: db }).forRun("run-1");

    await tools.invoke("place", { symbol: "X" }, { node: "loop#3/place" });
    await tools.invoke("place", { symbol: "X" }, { node: "loop#3/place", callId: "x", attempt: 2 });

    const [[ctx], [explicit]] = seen as [[ToolContext<any>, unknown], [ToolContext<any>, unknown]];
    expect(ctx.context).toBe(db);
    expect(ctx.runId).toBe("run-1");
    expect(ctx.node).toBe("loop#3/place");
    expect(ctx.callId).toBe("run-1/loop#3/place");
    expect(explicit.callId).toBe("x");
    expect(explicit.attempt).toBe(2);
  });

  test("trusted invocation passes the same context", async () => {
    const tools = new ToolSet({ place: place() }).forRun("run-1");

    await tools.invokeTrusted("place", { symbol: "X" }, { node: "n", attempt: 3 });

    const [ctx] = seen[0]!;
    expect(ctx.callId).toBe("run-1/n");
    expect(ctx.attempt).toBe(3);
  });

  test("for run returns a bound copy and leaves the original unbound", async () => {
    const db = new Db();
    const original = new ToolSet({ place: place() }, { context: db });

    const bound = original.forRun("run-1");

    expect(bound).not.toBe(original);
    expect(original.runId).toBe("");
    expect(bound.runId).toBe("run-1");
    expect(original.context).toBe(db);
    expect(bound.context).toBe(db);
    await original.invoke("place", { symbol: "X" });
    expect(seen[0]![0].runId).toBe("");
    expect(seen[0]![0].callId).toMatch(HEX32);
  });

  test("bound copy of the global registry still reads it", async () => {
    const bound = new ToolSet().forRun("run-9");
    registerTool("ctx_probe", place("ctx_probe"));

    expect(await bound.invoke("ctx_probe", { symbol: "Z" }, { node: "t" })).toBe("Z:1");
    expect(seen[0]![0].callId).toBe("run-9/t");
    expect(seen[0]![0].toolName).toBe("ctx_probe");
  });

  test("group items receive the context", async () => {
    const tools = new ToolSet({ trading: [place()] }).forRun("r");

    await tools.invoke("place", { symbol: "X" }, { node: "n" });

    expect(seen[0]![0].toolName).toBe("place");
    expect(seen[0]![0].callId).toBe("r/n");
  });

  test("registered tools receive the context", async () => {
    registerTool("ctx_registered", place("ctx_registered"));

    await new ToolSet().invoke("ctx_registered", { symbol: "X" }, { node: "n" });

    expect(seen[0]![0].toolName).toBe("ctx_registered");
  });

  test("module level invocation passes an unbound context", async () => {
    registerTool("ctx_module", place("ctx_module"));

    expect(await invokeTool("ctx_module", { symbol: "X" })).toBe("X:1");
    expect(await invokeToolTrusted("ctx_module", { symbol: "Y" })).toBe("Y:1");

    const [[first], [second]] = seen as [[ToolContext<any>, unknown], [ToolContext<any>, unknown]];
    expect(first.runId).toBe("");
    expect(first.node).toBe("");
    expect(first.callId).toMatch(HEX32);
    expect(first.callId).not.toBe(second.callId);
  });

  test("test_ask_user_handler_with_context_receives_it_first (TS: second)", async () => {
    const asked: [AskRequest, ToolContext<any>][] = [];
    const tools = new ToolSet({
      ask_user: (request, ctx) => {
        asked.push([request, ctx]);
        return "yes";
      },
    }).forRun("r");

    expect(await tools.invoke("ask_user", { question: "Q?", options: ["A", "a"] }, { node: "agent" })).toBe("yes");

    const [request, ctx] = asked[0]!;
    expect(ctx.toolName).toBe("ask_user");
    expect(ctx.node).toBe("agent");
    expect(ctx.callId).toBe("r/agent");
    expect(Object.keys(request).sort()).toEqual([
      "answerSchema",
      "callId",
      "deadline",
      "extra",
      "nodeId",
      "options",
      "question",
      "questions",
      "title",
    ]);
    expect(request.options).toEqual(["A"]);
  });

  test("a run bound call without a node gets a fresh call id", async () => {
    const tools = new ToolSet({ place: place() }).forRun("run-1");

    await tools.invoke("place", { symbol: "X" });
    await tools.invoke("place", { symbol: "Y" });

    const [[first], [second]] = seen as [[ToolContext<any>, unknown], [ToolContext<any>, unknown]];
    expect(first.runId).toBe("run-1");
    expect(first.callId).toMatch(HEX32);
    expect(second.callId).toMatch(HEX32);
    expect(first.callId).not.toBe(second.callId);
  });
});

describe("the call's abort signal", () => {
  test("the option wins over the bound signal, which wins over none", async () => {
    const run = new AbortController();
    const call = new AbortController();
    const tools = new ToolSet({ place: place() }).forRun("r", { signal: run.signal });

    await tools.invoke("place", { symbol: "X" }, { signal: call.signal });
    await tools.invokeTrusted("place", { symbol: "X" });

    expect(seen[0]![0].signal).toBe(call.signal);
    expect(seen[1]![0].signal).toBe(run.signal);
  });

  test("without either, a fresh signal that isn't aborted", async () => {
    const tools = new ToolSet({ place: place() });

    await tools.invoke("place", { symbol: "X" });
    await tools.invoke("place", { symbol: "Y" });

    const [[first], [second]] = seen as [[ToolContext<any>, unknown], [ToolContext<any>, unknown]];
    expect(first.signal).toBeInstanceOf(AbortSignal);
    expect(first.signal.aborted).toBe(false);
    expect(first.signal).not.toBe(second.signal);
  });
});
