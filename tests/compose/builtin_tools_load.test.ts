/** `call: take | sleep | ask` loads as a yian built-in. */
import { expect, test } from "bun:test";

import { outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
import { assembleText, loadError } from "../helpers/declarations.ts";

/** Return a flow with one node `t` calling the tool `toolId`, with `extra` body lines. */
const toolFlow = (toolId: string, extra: string) => `id: f
name: f
nodes:
  t:
    kind: call
    call: ${toolId}
${extra}output: null
`;

const refused = (text: string) => loadError(() => assembleText(text)).message;

test("call: ask_user is the ask built-in, its old name", () => {
  const text = toolFlow(
    "ask_user",
    `    input:
      prompt: Ship?
`,
  );
  const node = assembleText(text).flow.compiled.nodes.get("t") as ToolNode;
  expect(node.toolId).toBe("ask");
  expect(node.builtin).toBe("ask");
});

test("load_skill is refused as a tool id", () => {
  expect(refused(toolFlow("load_skill", ""))).toContain("`call: load_skill`: `load_skill` is reserved by yian");
});

const INBOX = `    input:
      from: \${input.m}
`;

/** Return a flow taking from inbox `m` in node `t`, with `extra` body lines. */
const takeFlow = (extra: string) => `id: f
name: f
input:
  m: !Inbox(str)
nodes:
  t:
    kind: call
    call: take
${INBOX}${extra}output: null
`;

test.each([
  ["retries", "    retries: 2\n", "`retries:` doesn't apply to take: there is no tool call to retry"],
  ["asserts", "    asserts:\n      - output != null\n", "`asserts:` doesn't apply to take"],
  ["output", "    output: !str\n", "take's output is inferred ({events: list(T), closed: bool}); remove `output:`"],
])("take refuses %s", (_label, extra, message) => {
  expect(refused(takeFlow(extra))).toContain(message);
});

// An untyped producer leaves the value's type unproven, but a container is never a date.
test.each([
  ["a list", "        - \${a.output}\n"],
  ["a mapping", "        at: \${a.output}\n"],
])("a sleep until written as %s holding a reference is refused", (shape, body) => {
  const text =
    "id: f\nname: f\nnodes:\n  a:\n    kind: code\n    code: return '2030-01-01'\n" +
    `  t:\n    kind: call\n    call: sleep\n    input:\n      until:\n${body}output: null\n`;
  expect(refused(text)).toContain(`input 'until': must be a date or datetime, but the value is ${shape}`);
});

test("sleep refuses output", () => {
  const text = toolFlow("sleep", "    input:\n      for: 90s\n    output: !str\n");
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

const CHAT = `id: chat
name: chat
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from: \${input.m}
      take: all
output: \${heard.output.events}
`;

test("take loads with its inferred output and reads its inbox", () => {
  const { flow } = assembleText(CHAT);
  const node = flow.compiled.nodes.get("heard") as ToolNode;
  expect(node.builtin).toBe("take");
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe('{"closed":bool,"events":list(str)}');
});

test("take's mode may be left out or read from a reference", () => {
  assembleText(CHAT.replace("      take: all\n", ""));
  const typed = CHAT.replace("input:\n", "typedefs:\n  Mode: Literal(one, all)\ninput:\n  mode: !Mode\n").replace(
    "take: all",
    "take: ${input.mode}",
  );
  assembleText(typed);
});

/** Return a flow with inbox `m`, text input `t` and take node `t1` with `inputs` lines. */
const takeWith = (inputs: string) => `id: f
name: f
input:
  m: !Inbox(str)
  t: !str
nodes:
  t1:
    kind: call
    call: take
    input:
${inputs}  keep:
    kind: call
    call: take
    input:
      from: \${input.m}
output: \${input.t}
`;

test.each([
  ["an unknown input", "      from: ${input.m}\n      x: 1\n", "take takes from, take and timeout; got 'x'"],
  ["a missing from", "      take: one\n", "take needs from: ${input.<inbox>}"],
  ["a text input in from", "      from: ${input.t}\n", "take's from must be an Inbox input of this flow, as ${input.<name>}"],
  ["take: some", "      from: ${input.m}\n      take: some\n", "input 'take': must be one or all"],
  ["an inbox in take", "      from: ${input.m}\n      take: ${input.m}\n", "input 'take': must be one or all"],
])("take refuses %s", (_label, inputs, message) => {
  expect(refused(takeWith(inputs))).toContain(message);
});

/** A take of inboxes `m` and `q`: its output names each event's inbox by its input name. */
const SEVERAL = `id: chat
name: chat
input:
  m: !Inbox(str)
  q: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from:
        - \${input.m}
        - \${input.q}
output: \${heard.output.events}
`;

test("a take of a list of inboxes names each event's inbox and the closed ones", () => {
  const node = assembleText(SEVERAL).flow.compiled.nodes.get("heard") as ToolNode;
  expect(node.fromNames).toEqual(["m", "q"]);
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe(
    '{"closed":bool,"closed_inboxes":list(Literal(m, q)),"events":list({"event":str,"inbox":Literal(m, q)})}',
  );
  // A take of one inbox names none.
  expect((assembleText(CHAT).flow.compiled.nodes.get("heard") as ToolNode).fromNames).toBeNull();
});

/** Return a flow with inboxes `m` and `q`, an `Inbox(int)` `n` and text `t`, whose take lists `items`. */
const listing = (items: string) => `id: f
name: f
input:
  m: !Inbox(str)
  q: !Inbox(str)
  n: !Inbox(int)
  t: !str
nodes:
  t1:
    kind: call
    call: take
    input:
      from: ${items}
  keep:
    kind: call
    call: take
    input:
      from:
        - \${input.m}
        - \${input.q}
  keep_n:
    kind: call
    call: take
    input:
      from: \${input.n}
output: \${input.t}
`;

test.each([
  ["an empty list", "[]", "take's from lists no inbox; list at least one, as ${input.<name>}"],
  ["an inbox twice", '["${input.m}", "${input.m}"]', "take's from lists inbox input 'm' twice"],
  ["a text input", '["${input.m}", "${input.t}"]', "take's from lists ${input.t}, which isn't an Inbox input of this flow"],
  ["a literal", '["${input.m}", m]', `take's from lists "m", which isn't an Inbox input of this flow`],
  ["inboxes of two types", '["${input.m}", "${input.n}"]', "take's from lists inboxes of two types, Inbox(str) and Inbox(int)"],
])("a take's list refuses %s", (_label, items, message) => {
  expect(refused(listing(items))).toContain(message);
});

test("a take of a list of one inbox names its events too", () => {
  const node = assembleText(listing('["${input.q}"]')).flow.compiled.nodes.get("t1") as ToolNode;
  expect(node.fromNames).toEqual(["q"]);
});
