import { expect, test } from "bun:test";
import { NodeSteps } from "../../src/nodes/contracts.ts";

const record = () => 1;
const view = { callId: "r/n", tool: "ask", state: "answered", closedReason: null, result: { kind: "tool_result" }, deadline: null } as const;

test("NodeSteps carries the node's calls and the run's event limit", () => {
  const plain = new NodeSteps(1, [], record);
  expect([plain.calls, plain.maxEventBytes]).toEqual([[], 1024 * 1024]);
  const steps = new NodeSteps(1, [], record, [view], 64);
  expect([steps.calls, steps.maxEventBytes]).toEqual([[view], 64]);
  expect(Object.isFrozen(steps.calls) && Object.isFrozen(steps.calls[0])).toBe(true);
});
test.each<[unknown, number]>([[{}, 1], [[], 0], [[], 1.5]])("NodeSteps refuses calls %p with limit %p", (calls, limit) => {
  expect(() => new NodeSteps(1, [], record, calls as never, limit)).toThrow();
});

import { Pending } from "../../src/nodes/base.ts";
test("Pending holds frozen call ids, or a wait the engine decides", () => {
  const host = new Pending(new Set(["r/ask"]));
  expect([host.callIds, host.wait]).toEqual([["r/ask"], null]);
  expect(Object.isFrozen(host) && Object.isFrozen(host.callIds)).toBe(true);
  const take = new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: null } });
  expect(take.callIds).toEqual([]);
  expect(Object.isFrozen(take.wait)).toBe(true);
});
test.each<[string, () => unknown, ErrorConstructor]>([
  ["no ids, no wait", () => new Pending([]), RangeError],
  ["ids and a wait", () => new Pending(["a"], { wait: { tool: "sleep", until: null, forSeconds: 5 } }), RangeError],
  ["a repeated id", () => new Pending(["a", "a"]), RangeError],
  ["an empty id", () => new Pending([""]), RangeError],
  ["a non-string id", () => new Pending([1 as unknown as string]), TypeError],
  ["a plain string of ids", () => new Pending("r/ask"), TypeError],
  ["sleep with both", () => new Pending([], { wait: { tool: "sleep", until: "2026-10-23T00:00:00.000Z", forSeconds: 1 } }), RangeError],
  ["sleep with neither", () => new Pending([], { wait: { tool: "sleep", until: null, forSeconds: null } }), RangeError],
  ["take mode", () => new Pending([], { wait: { tool: "take", inbox: "m", mode: "many" as "one", timeoutSeconds: null } }), RangeError],
  ["unknown tool", () => new Pending([], { wait: { tool: "ask" } as never }), TypeError],
])("Pending refuses %s", (_n, build, type) => expect(build).toThrow(type));

import { Node, NodeKind, Output } from "../../src/nodes/base.ts";
class Plain extends Node {
  readonly kind = NodeKind.CODE;
  override forward() { return new Output(1); }
}
test("the base node registers no host calls, so it shapes no result step", () => {
  const shape = () => new Plain("n").waitResultStep({ ...view, state: "open", result: null }, { output: 1 });
  expect(shape).toThrow(TypeError);
  expect(shape).toThrow(/registers no host calls/);
});

import { type NodeCaps } from "../../src/nodes/base.ts";
import { evalNode, EvaluatedOutcome } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { engine, graph, rejected } from "./run_state_fixtures.ts";

class Waits extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsSteps = true;
  seen: NodeSteps | undefined = undefined;
  override forward(_inputs: Record<string, unknown>, caps: NodeCaps) {
    this.seen = caps.steps;
    return new Pending(["r/w"]);
  }
}
test("evalNode resolves a Pending, and the steps it sees carry calls and the limit", async () => {
  const node = new Waits("w");
  const evaluated = await evalNode(node, null, new VariablePool("__start__"), { steps: new NodeSteps(1, [], record, [view], 64) });
  expect(evaluated.outcome).toBeInstanceOf(Pending);
  expect([node.seen!.calls, node.seen!.maxEventBytes]).toEqual([[view], 64]);
  expect(new EvaluatedOutcome(new Pending(["x"]), null).outcome).toBeInstanceOf(Pending);
});
test("until the engine waits, a Pending outcome is an unsupported outcome", async () => {
  const run = engine(graph([new Waits("w")], [["__start__", "w"], ["w", "__end__"]]));
  expect(String(await rejected(run.run()))).toContain("unsupported outcome");
});
