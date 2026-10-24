/** `adaptive_questions:` on a TOOL `ask` lowers onto it: a composer agent writes `__questions`. */
import { expect, test } from "bun:test";

import { desugarAdaptiveQuestions } from "../../src/compose/desugar/adaptive_questions.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { assembleText, loadError } from "./declarations.ts";

const ASK = `id: f
name: f
input:
  topic: !str
nodes:
  q:
    kind: tool
    tool_id: ask
    input:
      to: lead
      topic: \${input.topic}
      timeout: 24h
    adaptive_questions:
      prompt: "Ask about \${topic}"
output: \${q.output}
`;

/** Return the descriptors of flow `text`'s nodes, lowered. */
function lowered(text: string) {
  return desugarAdaptiveQuestions(parseNodes(SourceDocument.parse(text).root.child("nodes")));
}

test("the composer comes first, then the ask reading its questions", () => {
  const descriptors = lowered(ASK);
  expect([...descriptors.keys()]).toEqual(["q__compose", "q"]);
  const composer = descriptors.get("q__compose") as AgentDescriptor;
  expect(composer).toBeInstanceOf(AgentDescriptor);
  expect(composer.host).toBe("q");
  // The composer reads the ask's context, not its timeout.
  expect(Object.keys(composer.inputs)).toEqual(["to", "topic"]);
  const ask = descriptors.get("q") as ToolDescriptor;
  expect(ask.adaptiveQuestions).toBeNull();
  expect(ask.inputs["__questions"]).toBe("${q__compose.output}");
  expect(ask.inputs["timeout"]).toBe("24h");
});

test("an adaptive ask loads and answers an object", () => {
  const compiled = assembleText(ASK).flow.compiled;
  expect([...compiled.nodes.keys()]).toContain("q__compose");
  expect(compiled.nodes.get("q")!.outputContract).not.toBeNull();
});

test("a brief reading an undeclared name is refused like an agent prompt", () => {
  const text = ASK.replace("Ask about \${topic}", "Ask about \${undeclared}");
  expect(loadError(() => assembleText(text)).message).toBe(
    "flow has unresolved references:\n  node 'q' adaptive_questions.prompt: ${undeclared} is not a declared input or variable",
  );
});

test("a node named like the composer is refused", () => {
  const text = ASK.replace("output: ", "  q__compose:\n    kind: code\n    code: return 1\noutput: ");
  expect(loadError(() => assembleText(text)).message).toContain(
    "node 'q' (tool_id: ask): synthesized composer id 'q__compose' collides with an existing node id",
  );
});
