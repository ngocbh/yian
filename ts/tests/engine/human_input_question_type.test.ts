/**
 * The question-list type: mirrors `tests/engine/test_human_input_question_type.py`.
 *
 * Ported: `test_question_list_type_structure` and `test_question_type_drives_structured_schema`
 * (the exact schema is pinned in `type_to_schema.test.ts`).
 *
 * Ported through `loadFlow` and `runFlow`, in `hostEnv`:
 * `test_human_input_with_any_output_stores_a_free_text_answer_as_text`. Its run half also runs on
 * a hand-built HUMAN_INPUT node with the answer schema and `Any` output contract the loader gives
 * it, through `FlowEngine` (the `engine()` fixture) on START -> `ask` -> END.
 *
 * Changed: none.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { typeToSchema } from "../../src/nodes/agent/structured.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { HumanInputNode } from "../../src/nodes/human_input/node.ts";
import { questionListType } from "../../src/nodes/tool/questions.ts";
import { runFlow } from "../../src/run.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, graph } from "./run_state_fixtures.ts";

test("question list type structure", () => {
  const sh = questionListType();
  expect(sh.kind).toBe(ValueKind.LIST_OBJECT);
  const q = sh.element!;
  expect(q.kind).toBe(ValueKind.OBJECT);
  expect([...q.fields!.keys()]).toEqual(expect.arrayContaining(["question", "header", "options", "multi_select"]));
  expect([...q.required!]).toEqual(expect.arrayContaining(["question", "header"]));
  const opt = q.fields!.get("options")!;
  expect(opt.kind).toBe(ValueKind.LIST_OBJECT);
  expect([...opt.element!.fields!.keys()]).toEqual(expect.arrayContaining(["label", "description"]));
});

test("question type drives structured schema", () => {
  expect(typeToSchema(questionListType())).not.toBeNull();
});

const ANY_GATE_FLOW = `
id: gate
name: gate
nodes:
  ask:
    kind: human_input
    prompt: anything to add?
    output: Any
output: \${ask.output}
`;

test("human input with Any output stores a free text answer as text", async () => {
  const loaded = await loadFlow(ANY_GATE_FLOW);
  expect((loaded.compiled.nodes.get("ask") as HumanInputNode).answerSchema).toEqual([
    { name: "answer", type: "Any", required: true },
  ]);
  const dir = mkdtempSync(join(tmpdir(), "yian-human-input-question-"));
  try {
    const result = await runFlow(loaded, {}, { tools: { ask_user: () => "yes" }, env: hostEnv(dir) });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("yes");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("human input with Any output stores a free text answer as text (engine)", async () => {
  const node = new HumanInputNode("ask", {
    prompt: "anything to add?",
    answerSchema: [{ name: "answer", type: "Any", required: true }],
  });
  node.outputContract = new ValueContract(Type.scalar(ValueKind.ANY));
  const flow = graph([node], [[START_ID, "ask"], ["ask", END_ID]], {
    outputs: [new FlowOutput("result", "${ask.output}")],
  });
  let last: unknown;
  const built = engine(flow, {
    tools: new ToolSet({ ask_user: () => "yes" }),
    onEvent: (event) => {
      last = event;
    },
  });

  await built.run();

  expect(last).toBeInstanceOf(RunSucceeded);
  expect((last as RunSucceeded).output).toBe("yes");
});
