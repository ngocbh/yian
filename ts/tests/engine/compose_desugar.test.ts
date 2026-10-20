/**
 * The adaptive-questions desugar, end to end: a port of `tests/engine/test_compose_desugar.py`.
 *
 * Ported: `test_adaptive_questions_lowers_to_agent_plus_gate` and
 * `test_adaptive_questions_scope_is_validated_like_an_agent` (the reference validator's
 * prompt-scope check), and its case
 * `load-core/engine_compose_desugar__adaptive_questions_scope_is_validated_like_an_agent`.
 *
 * Changed: the flow assembles with `assembleText`, where Python loads it.
 *
 * TS-only: the composer's `retries:` count, the composer id collision error, and the harvested
 * cases the adaptive desugar raises whose Python test file has no TS mirror, each named by its case
 * id, pinned at the end.
 */

import { expect, test } from "bun:test";

import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import type { ValueContract } from "../../src/nodes/contracts.ts";
import type { HumanInputNode } from "../../src/nodes/human_input/node.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../compose/declarations.ts";

const AUTO = `
id: f
name: f
input:
  ctx: !str
nodes:
  ask:
    kind: human_input
    input: {ctx: "\${input.ctx}"}
    prompt: "Help me with \${ctx}."
    adaptive_questions:
      prompt: "Design 1-3 questions with options for: \${ctx}"
output: \${ask.output}
`;

test("adaptive questions lowers to agent plus gate", () => {
  const nodes = assembleText(AUTO).flow.compiled.nodes;
  expect([...nodes.keys()]).toEqual(["ask__compose", "ask", "__start__", "__end__"]);
  const gate = nodes.get("ask") as HumanInputNode;
  const agent = nodes.get("ask__compose") as AgentNode;
  expect(agent.kind).toBe(NodeKind.AGENT);
  expect((agent.outputContract as ValueContract).type.kind).toBe(ValueKind.LIST_OBJECT);
  expect(gate.questionsInput).toBe("__questions");
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
    "node 'ask' (kind=human_input): synthesized composer id 'ask__compose' collides with an existing node id",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});

// ---------- harvested conformance cases ----------

test("assemble: load-core/engine_compose_desugar__adaptive_questions_scope_is_validated_like_an_agent", () => {
  runCase({
    id: "load-core/engine_compose_desugar__adaptive_questions_scope_is_validated_like_an_agent",
    call: assembled,
    full: true,
  });
});

// From `tests/compose/test_node_input_origins.py`.
test.each<CaseRow>([{ id: "load-core/compose_node_input_origins__adaptive_reserved_input", call: assembled }])(
  "assemble: $id",
  runCase,
);
