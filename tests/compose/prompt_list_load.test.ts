/** An AGENT `prompt:` may be a list of messages: what loads, and each load error at its item. */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { forget } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { assembleText, loadError } from "./declarations.ts";

afterEach(() => forget());

/** The line of `chatFlow`'s first prompt line. */
const PROMPT_LINE = 20;

/**
 * A chat flow whose `reply` agent reads `history` (typed `historyType`) and `message`; `prompt`
 * holds the lines after `prompt:`, each indented under it, starting on `PROMPT_LINE`.
 */
function chatFlow(prompt: readonly string[], historyType = "!list(Msg)"): string {
  return `\
id: chat
name: chat
typedefs:
  Msg:
    role: Literal(user, assistant)
    text: str
input:
  history: ${historyType}
  message: !str
nodes:
  reply:
    kind: agent
    input:
      history: \${input.history}
      message: \${input.message}
    output: str
    system_prompt: You are a helpful conversational assistant.
    variables: {greeting: !str hello}
    prompt:
${prompt.map((line) => `      ${line}`).join("\n")}
output: \${reply.output}
`;
}

/** Assemble `text` and return its `reply` node. */
const replyOf = (text: string) => assembleText(text).flow.compiled.nodes.get("reply") as AgentNode;

/** Return the load error assembling `text` raises, with its message and start line. */
function failure(text: string): { message: string; line: number } {
  const error = loadError(() => assembleText(text));
  return { message: error.message, line: error.diagnostic.frames.at(-1)!.range.startLine };
}

test("a history then a user message loads into prompt items", () => {
  const node = replyOf(chatFlow(["- ${history}", "- user: ${message}"]));

  expect(node.prompt).toEqual([{ splice: "history" }, { role: "user", text: "${message}" }]);
});

test("literal turns, variables and a single message type load", () => {
  const text = chatFlow(
    ["- user: ${greeting}", "- assistant: Hi, how can I help?", "- ${history}", "- user: ${message}"],
    "!Msg",
  );

  expect(replyOf(text).prompt).toEqual([
    { role: "user", text: "${greeting}" },
    { role: "assistant", text: "Hi, how can I help?" },
    { splice: "history" },
    { role: "user", text: "${message}" },
  ]);
});

test.each(["!object", "!list(object)", "!Any", "!list(Any)"])("a %s history may hold messages", (type) => {
  expect(replyOf(chatFlow(["- ${history}"], type)).prompt).toEqual([{ splice: "history" }]);
});

test("a text prompt stays text", () => {
  const text = chatFlow([]).replace("    prompt:\n\n", "    prompt: ${message}\n");
  expect(replyOf(text).prompt).toBe("${message}");
});

test.each([
  ["a pool reference", ["- ${input.history}"], 0, "${input.history}"],
  ["a plain string", ["- user: hi", "- hello"], 1, "hello"],
  ["a span inside text", ["- 'Earlier: ${history}'"], 0, "Earlier: ${history}"],
  ["an expression span", ["- ${history[0]}"], 0, "${history[0]}"],
])("a text item that isn't a lone ${name} is refused at it: %s", (_, prompt, index, item) => {
  expect(failure(chatFlow(prompt))).toEqual({
    message:
      `node 'reply' (kind=agent): prompt list item \`${item}\` isn't a splice: a splice is a lone ` +
      "`${name}` naming one of this node's inputs or variables; write a message as `user: <text>`",
    line: PROMPT_LINE + index,
  });
});

test.each([
  ["two keys", ["- {user: a, assistant: b}"], 0],
  ["an unknown role", ["- tool: hi"], 0],
  ["a list", ["- [a]"], 0],
  ["a null", ["- null"], 0],
])("a prompt list item of another shape is refused at it: %s", (_, prompt, index) => {
  expect(failure(chatFlow(prompt))).toEqual({
    message: "node 'reply' (kind=agent): a prompt list item is `${name}` or `user:`/`assistant:` text",
    line: PROMPT_LINE + index,
  });
});

test("a non-text message is refused at it", () => {
  expect(failure(chatFlow(["- ${history}", "- user: 5"]))).toEqual({
    message:
      "node 'reply' (kind=agent): `prompt[1].user:` must be text, but YAML read it as an integer; " +
      "put it in quotes",
    line: PROMPT_LINE + 1,
  });
});

test("a system message is refused", () => {
  expect(failure(chatFlow(["- system: Be brief.", "- user: ${message}"]))).toEqual({
    message: "node 'reply' (kind=agent): a prompt list item can't be `system:`; use `system_prompt:`",
    line: PROMPT_LINE,
  });
});

test("an empty list is refused", () => {
  const text = chatFlow([]).replace("    prompt:\n\n", "    prompt: []\n");
  expect(failure(text)).toEqual({
    message: "node 'reply' (kind=agent): `prompt:` must hold at least one message",
    line: PROMPT_LINE - 1,
  });
});

test("a list ending in an assistant message is refused at it", () => {
  expect(failure(chatFlow(["- ${history}", "- assistant: Sure."]))).toEqual({
    message: "node 'reply' (kind=agent): the last message of `prompt:` must be from the user",
    line: PROMPT_LINE + 1,
  });
});

test("a splice naming no input or variable is refused at it", () => {
  const { message, line } = failure(chatFlow(["- user: ${message}", "- ${transcript}", "- user: ${message}"]));

  expect(message).toBe(
    "flow has unresolved references:\n  node 'reply' prompt[1]: ${transcript} is not a declared input or variable",
  );
  expect(line).toBe(PROMPT_LINE + 1);
});

test("an unknown reference in a message is refused at it", () => {
  const { message, line } = failure(chatFlow(["- ${history}", "- user: ${input.message}"]));

  expect(message).toContain("node 'reply' prompt[1]: ${input.message} is not a declared input or variable");
  expect(line).toBe(PROMPT_LINE + 1);
});

test.each([
  ["!str", "str"],
  ["!list(str)", "list(str)"],
  ["!int", "int"],
])("a splice of a %s input is refused, naming the type", (type, label) => {
  const { message, line } = failure(chatFlow(["- ${history}", "- user: ${message}"], type));

  expect(message).toBe(
    "flow has unresolved references:\n  " +
      `node 'reply' prompt[0]: \${history} is \`${label}\`, which can't hold messages; ` +
      "splice a `Message`, a record with `role` and `text: str`, or a list of them",
  );
  expect(line).toBe(PROMPT_LINE);
});

test("a splice of a record whose text may be null is refused", () => {
  const text = chatFlow(["- ${history}", "- user: ${message}"], "!list(Turn)").replace(
    "typedefs:\n",
    "typedefs:\n  Turn: {role: str, text: Optional(str)}\n",
  );
  expect(failure(text).message).toContain("${history} is `list(");
  expect(failure(text).message).toContain("which can't hold messages");
});

test("a splice of a record without text is refused", () => {
  const text = chatFlow(["- ${history}", "- user: ${message}"], "!Turn").replace(
    "typedefs:\n",
    "typedefs:\n  Turn: {role: str, content: str}\n",
  );
  expect(failure(text).message).toContain('${history} is `{"content":str,"role":str}`, which can\'t hold messages');
});

test("a list prompt is identity; a text prompt's manifest is unchanged", async () => {
  const listed = (await loadFlow(chatFlow(["- ${history}", "- user: ${message}"]))).compiled;
  const manifest = (flowManifest(listed).nodes as JsonObject).reply as JsonObject;
  expect(manifest.prompt).toEqual(["${history}", { user: "${message}" }]);

  const other = (await loadFlow(chatFlow(["- ${history}", "- user: '${message}!'"]))).compiled;
  expect(flowFingerprint(other)).not.toBe(flowFingerprint(listed));

  const texted = (await loadFlow(chatFlow([]).replace("    prompt:\n\n", "    prompt: ${message}\n"))).compiled;
  expect(((flowManifest(texted).nodes as JsonObject).reply as JsonObject).prompt).toBe("${message}");
});

test.each(["!list(Message)", "!Message", "!Optional(Message)", "!list(Optional(Message))"])("a %s history loads", (type) => {
  expect(replyOf(chatFlow(["- ${history}", "- user: ${message}"], type)).prompt).toEqual([
    { splice: "history" },
    { role: "user", text: "${message}" },
  ]);
});

test("a typedef holding Messages or text records loads", () => {
  const text = chatFlow(["- ${history}", "- user: ${message}"], "!list(Entry)").replace(
    "typedefs:\n",
    "typedefs:\n  Entry: Union(Message, Msg)\n",
  );
  expect(replyOf(text).prompt).toHaveLength(2);
});

test("an AGENT can't output a Message", () => {
  const { message } = failure(chatFlow(["- ${history}", "- user: ${message}"]).replace("    output: str\n", "    output: Message\n"));
  expect(message).toContain("an AGENT can't produce this `output:`");
});
