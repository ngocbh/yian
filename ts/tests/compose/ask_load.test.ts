/** A TOOL `ask` loads with `human_input`'s question surface as inputs. */
import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { assembleText, loadError } from "./declarations.ts";

/** Return a flow with one TOOL node `q` of tool id `toolId` and `body` lines. */
const askFlow = (body: string, toolId = "ask") => `id: f
name: f
input:
  m: !Inbox(str)
nodes:
  q:
    kind: tool
    tool_id: ${toolId}
${body}  keep:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
output: null
`;

const refused = (text: string) => loadError(() => assembleText(text)).message;

const ADAPTIVE = `    adaptive_questions:
      prompt: Ask what matters
`;

test.each([
  ["nothing to ask", "    input:\n      to: lead\n", "ask needs input prompt:, input questions:, or adaptive_questions:"],
  ["no input at all", "", "ask needs input prompt:, input questions:, or adaptive_questions:"],
  [
    "questions and adaptive_questions together",
    `    input:\n      questions:\n        - question: Ship?\n${ADAPTIVE}`,
    "input questions: and adaptive_questions: are mutually exclusive",
  ],
  [
    "an unknown adaptive_questions key",
    "    adaptive_questions:\n      prompt: Ask\n      model: x\n",
    "`adaptive_questions:` field 'model' is not allowed (allowed: llm_config, prompt, retries)",
  ],
  ["a non-text brief", "    adaptive_questions:\n      prompt: 5\n", "adaptive_questions.prompt"],
  ["adaptive_questions without a brief", "    adaptive_questions:\n      retries: 1\n", "requires a `prompt:`"],
  [
    "__questions authored",
    "    input:\n      prompt: Ship?\n      __questions: []\n",
    "input '__questions' is reserved for adaptive_questions",
  ],
])("ask refuses %s", (_label, body, message) => {
  expect(refused(askFlow(body))).toContain(message);
});

test("adaptive_questions is refused on a take", () => {
  const body = `    input:\n      from: \${input.m}\n${ADAPTIVE}`;
  expect(refused(askFlow(body, "take"))).toContain("`adaptive_questions:` applies only to tool_id: ask");
});

test("ask with adaptive_questions parses and ranges the brief and llm_config", () => {
  const text = `nodes:
  q:
    kind: tool
    tool_id: ask
    adaptive_questions:
      prompt: Ask what matters
      llm_config:
        model: m
`;
  const desc = ToolDescriptor.parse("q", SourceDocument.parse(text).root.child("nodes").child("q"));
  expect(desc.adaptiveQuestions).toEqual({ prompt: "Ask what matters", llm_config: { model: "m" } });
  expect(desc.fieldRanges.get(siteKey("text", "adaptive_questions.prompt"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "adaptive_questions.llm_config"))).toBeInstanceOf(SourceRange);
});
