/**
 * The adaptive-questions desugar, end to end: a port of `tests/engine/test_compose_desugar.py`.
 *
 * Ported: `test_adaptive_questions_lowers_to_agent_plus_gate` and
 * `test_adaptive_questions_scope_is_validated_like_an_agent` (the reference validator's
 * prompt-scope check).
 *
 * Changed: the flow assembles with `assembleText`, where Python loads it. The gate is a TOOL
 * `ask`, whose `__questions` input the composer feeds.
 *
 * TS-only: the composer's `retries:` count and the composer id collision error.
 */

import { expect, test } from "bun:test";

import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import type { ValueContract } from "../../src/nodes/contracts.ts";
import type { ToolNode } from "../../src/nodes/tool/node.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { assembleText, loadError } from "../compose/declarations.ts";

const AUTO = `
id: f
name: f
input:
  ctx: !str
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      ctx: \${input.ctx}
      prompt: "Help me with \${input.ctx}."
    adaptive_questions:
      prompt: "Design 1-3 questions with options for: \${ctx}"
output: \${ask.output}
`;

test("adaptive questions lowers to agent plus gate", () => {
  const nodes = assembleText(AUTO).flow.compiled.nodes;
  expect([...nodes.keys()]).toEqual(["ask__compose", "ask", "__start__", "__end__"]);
  const gate = nodes.get("ask") as ToolNode;
  const agent = nodes.get("ask__compose") as AgentNode;
  expect(agent.kind).toBe(NodeKind.AGENT);
  expect((agent.outputContract as ValueContract).type.kind).toBe(ValueKind.LIST_OBJECT);
  expect(gate.builtin).toBe("ask");
  expect(gate.params!.map((param) => param.name)).toContain("__questions");
  expect((gate.outputContract as ValueContract).type.kind).toBe(ValueKind.OBJECT);
});

test("adaptive questions scope is validated like an agent", () => {
  // The composer agent inherits the agent prompt-scope check.
  const bad = AUTO.replace("for: ${ctx}", "for: ${undeclared}");
  expect(loadError(() => assembleText(bad)).message).toContain("${undeclared} is not a declared input or variable");
});

test("the block's retries count is the composer's", () => {
  const text = AUTO.replace("for: ${ctx}\"\n", "for: ${ctx}\"\n      retries: 2\n");
  const agent = assembleText(text).flow.compiled.nodes.get("ask__compose") as AgentNode;
  expect(agent.retries).toBe(2);
});

test("a composer id collision is located at the gate", () => {
  const text = AUTO.replace("output:", "  ask__compose:\n    kind: code\n    code: return 1\n    output: int\noutput:");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(
    "node 'ask' (tool_id: ask): synthesized composer id 'ask__compose' collides with an existing node id",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});
