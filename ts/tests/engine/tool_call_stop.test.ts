/**
 * Each tool call sees the signal of the run that made it: a port of
 * `tests/engine/test_tool_call_stop.py`.
 *
 * TS has no stop event and no `_current_stop`: a run binds its tool set to its run signal
 * (`ToolSet.forRun`), and a call's `ToolContext.signal` is that signal unless the caller passes one.
 *
 * Changed: `test_concurrent_runs_see_their_own_stop_events` is "two concurrent runs' nodes and tool
 * calls see their own signals": two engines share one tool set, and each node's call, made while
 * both calls are in flight, sees its own run's signal.
 *
 * Ported elsewhere: `test_for_run_binds_the_stop_event_seen_during_a_call`,
 * `test_an_unbound_set_and_for_run_without_stop_event_give_none` and
 * `test_for_run_sets_exactly_the_stop_event_passed` (`tool_registry.test.ts`'s `ToolSet.forRun` rows
 * and `tool_context.test.ts`'s call-signal rows).
 *
 * Not ported: `test_a_langchain_tool_sees_the_stop_event` (TS has no LangChain tools) and
 * `test_the_stop_event_is_cleared_after_a_tool_raises` (no context variable to clear).
 *
 * Later (T4-11): `test_a_tool_node_sees_the_engine_stop_event`,
 * `test_a_tool_node_sees_its_run_stop_event` (TOOL node). Later (T5-11a):
 * `test_an_agent_tool_call_sees_its_run_stop_event` (AGENT node).
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { engine, graph } from "./run_state_fixtures.ts";

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
