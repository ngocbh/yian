/**
 * Each tool call sees the signal of the run that made it: a run binds its tool set to its run
 * signal (`ToolSet.forRun`), and a call's `ToolContext.signal` is that signal unless the caller
 * passes one.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { aborted, engine, graph, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

/** Calls the `probe` tool once, naming itself as the calling node, and outputs its answer. */
class ProbeNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsTools = true;
  /** The signal the node received as `caps.signal`. */
  signal: AbortSignal | null = null;

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    this.signal = caps.signal!;
    return new Output(await caps.tools!.invoke("probe", {}, { node: this.id }));
  }
}

test("two concurrent runs' nodes and tool calls see their own signals", async () => {
  const seen = new Map<string, AbortSignal>();
  // Each call waits until both are in flight, so the two runs interleave.
  let arrived = 0;
  const bothInside = Promise.withResolvers<void>();
  const probe = new FunctionTool({
    name: "probe",
    description: "Record the call's signal.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: async (_args, ctx) => {
      arrived += 1;
      if (arrived === 2) bothInside.resolve();
      await bothInside.promise;
      seen.set(ctx.node, ctx.signal);
      return "ok";
    },
  });
  // One tool set shared by both runs, as one MCP server's tools are.
  const shared = new ToolSet({ probe });
  const nodes = [new ProbeNode("a"), new ProbeNode("b")];
  const runs = nodes.map((node) =>
    engine(graph([node], [[START_ID, node.id], [node.id, END_ID]]), { numWorkers: 0, tools: shared }),
  );

  const results = await Promise.all(runs.map((run) => run.run()));

  expect(results.map((events) => events.at(-1))).toEqual([expect.any(RunSucceeded), expect.any(RunSucceeded)]);
  expect(seen.get("a")).toBe(runs[0]!.signal);
  expect(seen.get("b")).toBe(runs[1]!.signal);
  expect(runs[0]!.signal).not.toBe(runs[1]!.signal);
  expect(nodes[0]!.signal).not.toBe(nodes[1]!.signal);
  expect([...seen.values(), ...nodes.map((node) => node.signal!)].some((signal) => signal.aborted)).toBe(false);
});

for (const numWorkers of [0, 8]) {
  test(`a TOOL node's call sees a live signal (numWorkers ${numWorkers})`, async () => {
    const seen: [AbortSignal, boolean][] = [];
    const probe = new FunctionTool({
      name: "probe",
      description: "Record the call's signal.",
      parametersJsonSchema: { type: "object", properties: {} },
      execute: (_args, ctx) => {
        seen.push([ctx.signal, ctx.signal.aborted]);
        return "ok";
      },
    });
    const flow = graph([new ToolNode("probe", { toolId: "probe" })], [
      [START_ID, "probe"],
      ["probe", END_ID],
    ]);

    const events = await engine(flow, { numWorkers, tools: new ToolSet({ probe }) }).run();

    expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
    expect(seen).toEqual([[expect.any(AbortSignal), false]]);
  });

  test(`a TOOL node's call signal aborts when its run stops (numWorkers ${numWorkers})`, async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const entered = Promise.withResolvers<AbortSignal>();
    const probe = new FunctionTool({
      name: "probe",
      description: "Wait for the call's signal to abort.",
      parametersJsonSchema: { type: "object", properties: {} },
      execute: async (_args, ctx) => {
        entered.resolve(ctx.signal);
        return await aborted(ctx.signal);
      },
    });
    const flow = graph([new ToolNode("probe", { toolId: "probe" })], [
      [START_ID, "probe"],
      ["probe", END_ID],
    ]);
    const running = engine(flow, { numWorkers, tools: new ToolSet({ probe }), signal: host.signal }).run();

    const signal = await entered.promise;
    expect(signal.aborted).toBe(false);
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(signal.aborted).toBe(true);
    expect(signal.reason).toBe(reason);
  });
}

test("an AGENT's tool call signal aborts when its run stops", async () => {
  const host = new AbortController();
  const reason = new Error("stop now");
  const entered = Promise.withResolvers<AbortSignal>();
  const probe = new FunctionTool({
    name: "probe",
    description: "Wait for the call's signal to abort.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: async (_args, ctx) => {
      entered.resolve(ctx.signal);
      return await aborted(ctx.signal);
    },
  });
  const config = { provider: "openai", model: "gpt-5.5" };
  const call = new ToolCallPart({ id: "c0", name: "probe", arguments: {} });
  const calling = new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
  const model = new ScriptedModel([calling, textReply("done")], config);
  const flow = graph([new AgentNode("agent", { prompt: "go", llmConfig: config, tools: ["probe"] })], [
    [START_ID, "agent"],
    ["agent", END_ID],
  ]);
  const running = engine(flow, { tools: new ToolSet({ probe }), models: serving(model), signal: host.signal }).run();

  const signal = await entered.promise;
  expect(signal.aborted).toBe(false);
  host.abort(reason);

  expect(await rejected(running)).toBe(reason);
  expect(signal.aborted).toBe(true);
  expect(signal.reason).toBe(reason);
  expect(model.requests).toHaveLength(1);
});
