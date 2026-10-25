/** A TOOL `ask` loads with `human_input`'s question surface as inputs. */
import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { bindingSource, outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
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
  ["a non-mapping input", "    input: Ship?\n", "node 'q' (kind=tool): `input:` must be a mapping of"],
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

test("tool_id: ask_user asks like ask: a lone question is a host extra, not something to ask", () => {
  const body = "    input:\n      question: Ship?\n";
  expect(refused(askFlow(body, "ask_user"))).toContain(
    "node 'q' (tool_id: ask): ask needs input prompt:, input questions:, or adaptive_questions:",
  );
});

test("tool_id: ask_user takes adaptive_questions as ask does", () => {
  const node = assembleText(askFlow(ADAPTIVE, "ask_user")).flow.compiled.nodes.get("q") as ToolNode;
  expect(node.builtin).toBe("ask");
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

const ESCALATION = `id: approve_plan
name: approve_plan
typedefs:
  Approval:
    decision: Literal(approve, reject)
    note: Optional(str)
input:
  plan: !str
nodes:
  lead:
    kind: tool
    tool_id: ask
    input:
      to: lead
      prompt: "Approve this plan?\\n\\n\${input.plan}"
      timeout: 24h
    output: Optional(Approval)
  escalate:
    kind: case
    cases:
      - when: \${lead.output == null}
        then: manager
  manager:
    kind: tool
    tool_id: ask
    input:
      to: manager
      prompt: "No answer from the lead in 24h. Approve this plan?\\n\\n\${input.plan}"
      timeout: 48h
    output: Optional(Approval)
output: !Optional(Approval) \${lead.output | manager.output}
`;

test("the escalation flow loads: answer schema, timeout seconds and host extras", () => {
  const compiled = assembleText(ESCALATION).flow.compiled;
  const lead = compiled.nodes.get("lead") as ToolNode;
  expect(lead.builtin).toBe("ask");
  expect(lead.answerSchema.map((field) => field["name"])).toEqual(["decision", "note"]);
  expect(lead.answerSchema[0]).toEqual({ name: "decision", type: "str", required: true, enum: ["approve", "reject"] });
  const wiring = compiled.wiring.get("lead")!;
  expect(bindingSource(wiring.get("timeout"))).toBe(86400n);
  expect(bindingSource(wiring.get("to"))).toBe("lead");
  expect((compiled.nodes.get("manager") as ToolNode).retries).toBe(0);
});

/** Return a flow whose one ask node `q` has `inputs` lines and `extra` body lines. */
const askWith = (inputs: string, extra = "") => `id: f
name: f
input:
  qs: !list(object)
  n: !int
nodes:
  q:
    kind: tool
    tool_id: ask
    input:
${inputs}${extra}output: \${q.output}
`;

test("a questions ask answers an object; a prompt ask is free form", () => {
  const questions = assembleText(askWith("      questions:\n        - question: Ship?\n          header: Ship\n"));
  const node = questions.flow.compiled.nodes.get("q") as ToolNode;
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe("object");
  expect(node.answerSchema).toEqual([{ name: "answer", type: "object", required: true }]);
  const prompt = assembleText(askWith("      prompt: Ship?\n")).flow.compiled.nodes.get("q") as ToolNode;
  expect(prompt.outputContract).toBeNull();
  expect(prompt.answerSchema).toEqual([]);
});

test("a questions reference loads", () => {
  assembleText(askWith("      questions: ${input.qs}\n"));
});

test.each([
  ["a question without its question", "      questions:\n        - header: Ship\n", "question[0] is invalid"],
  ["a non-text header", "      questions:\n        - question: Ship?\n          header: 5\n", "`questions[0].header:` must be text"],
  ["questions that aren't a list", "      questions: ship\n", "must be a list of questions"],
  ["a questions reference that isn't a list", "      questions: ${input.n}\n", "must be a list of questions, but the value is int"],
  ["prompt: 5", "      prompt: 5\n", "`prompt:` must be text, but YAML read it as an integer"],
  ["a prompt reference that isn't text", "      prompt: ${input.n}\n", "must be text, but the value is int"],
])("ask refuses %s", (_label, inputs, message) => {
  expect(refused(askWith(inputs))).toContain(message);
});

test.each([
  ["Optional(str)", [{ name: "answer", type: "str", required: true }]],
  ["Optional(Literal(yes, no))", [{ name: "answer", type: "str", required: true, enum: ["no", "yes"] }]],
])("the answer schema of output %s is its non-null type", (output, schema) => {
  const text = askWith("      prompt: Ship?\n      timeout: 1h\n", `    output: ${output}\n`);
  const node = assembleText(text).flow.compiled.nodes.get("q") as ToolNode;
  expect(node.answerSchema).toEqual(schema);
});
