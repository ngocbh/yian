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
