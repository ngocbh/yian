/** `kind: tool` with `tool_id: take | sleep | ask` loads as a yian built-in. */
import { expect, test } from "bun:test";

import { ToolNode } from "../../src/nodes/tool/node.ts";
import { assembleText, loadError } from "./declarations.ts";

/** Return a flow with one TOOL node `t` of tool id `toolId` and `extra` body lines. */
const toolFlow = (toolId: string, extra: string) => `id: f
name: f
nodes:
  t:
    kind: tool
    tool_id: ${toolId}
${extra}output: null
`;

const refused = (text: string) => loadError(() => assembleText(text)).message;

test("tool_id: ask_user stays a plain tool call", () => {
  const text = toolFlow(
    "ask_user",
    `    retries: 2
    input:
      question: Ship it?
      options:
        - yes
        - no
`,
  );
  const node = assembleText(text).flow.compiled.nodes.get("t") as ToolNode;
  expect(node.toolId).toBe("ask_user");
  expect(node.builtin).toBeNull();
  expect(node.retries).toBe(2);
});

test("load_skill is refused as a tool id", () => {
  expect(refused(toolFlow("load_skill", ""))).toContain("tool id `load_skill` is reserved by yian");
});

const INBOX = `    input:
      from: \${input.m}
`;

/** Return a flow taking from inbox `m` in TOOL node `t`, with `extra` body lines. */
const takeFlow = (extra: string) => `id: f
name: f
input:
  m: !Inbox(str)
nodes:
  t:
    kind: tool
    tool_id: take
${INBOX}${extra}output: null
`;

test.each([
  ["retries", "    retries: 2\n", "`retries:` doesn't apply to take: there is no tool call to retry"],
  ["asserts", "    asserts:\n      - output != null\n", "`asserts:` doesn't apply to take"],
  ["output", "    output: str\n", "take's output is inferred ({events: list(T), closed: bool}); remove `output:`"],
])("take refuses %s", (_label, extra, message) => {
  expect(refused(takeFlow(extra))).toContain(message);
});

test("sleep refuses output", () => {
  const text = toolFlow("sleep", "    input:\n      for: 90s\n    output: str\n");
  expect(refused(text)).toContain("sleep's output is None; remove `output:`");
});

test("ask refuses retries", () => {
  const text = toolFlow("ask", "    retries: 1\n    input:\n      prompt: Ship it?\n");
  expect(refused(text)).toContain("`retries:` doesn't apply to ask: the host owns retries of a call it holds");
});

test("a built-in builds with no retries", () => {
  const text = toolFlow("ask", "    input:\n      prompt: Ship it?\n");
  const node = assembleText(text).flow.compiled.nodes.get("t") as ToolNode;
  expect(node.builtin).toBe("ask");
  expect(node.retries).toBe(0);
});
