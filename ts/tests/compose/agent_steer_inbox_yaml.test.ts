/**
 * `steer_inbox:` on AGENT: the agent reads steers from a declared top-level inbox, named by a lone
 * `${input.<name>}`. The retired `steerable:` is a load error with a hint.
 *
 * TS-only: Python yian has no inboxes. Python's `tests/compose/test_agent_steerable_yaml.py` keeps
 * the old `steerable:` rows.
 */

import { afterEach, expect, test } from "bun:test";

import { forget } from "../../src/defaults.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

/** A one-agent flow reading inbox `notes`, with `steerLine` on line 9 and `output` after it. */
function steerFlow(steerLine: string, output = "    output: str", input = "  notes: !Inbox(str)"): string {
  return `\
id: steer
name: steer
input:
${input}
nodes:
  chat:
    kind: agent
    prompt: Help the user.
${steerLine}
${output}
output: \${chat.output}
`;
}

/** Assemble `text` and return its compiled flow. */
const flowOf = (text: string) => assembleText(text).flow.compiled;
/** Assemble `text` and return its `chat` node. */
const chatOf = (text: string) => flowOf(text).nodes.get("chat") as AgentNode;

test("steer_inbox reaches the agent node and its reserved wiring", () => {
  const flow = flowOf(steerFlow("    steer_inbox: ${input.notes}"));
  const chat = flow.nodes.get("chat") as AgentNode;
  expect(chat.steerInbox).toBe(true);
  expect(chat.readsSteerInbox).toBe(true);
  expect((flow.wiring.get("chat")!.get(STEER_INBOX_KEY) as Binding).source).toBe("${input.notes}");
});

test("an agent with a steer inbox and no declared output is text", () => {
  expect(chatOf(steerFlow("    steer_inbox: ${input.notes}", "")).steerInbox).toBe(true);
});

test("an agent with a steer inbox may declare an Optional(str) output", () => {
  const chat = chatOf(steerFlow("    steer_inbox: ${input.notes}", "    output: Optional(str)"));
  expect(chat.steerInbox).toBe(true);
  expect(chat.outputContract).toBeInstanceOf(ValueContract);
  expect((chat.outputContract as ValueContract).type.nullable).toBe(true);
});

test.each([
  [
    "    steerable: true",
    "    output: str",
    "node field `steerable:` was replaced by `steer_inbox:`: declare an `Inbox(...)` input and write " +
      "`steer_inbox: ${input.<name>}`; the host sends steers with `send`",
  ],
  ["    steer_inbox: 3", "    output: str", "`steer_inbox:` must be `${input.<name>}` naming an Inbox input"],
  ["    steer_inbox: ${input.notes}", "    output: int", "node 'chat': a steer inbox needs `str` output"],
  ["    steer_inbox: ${input.notes}", "    output: {answer: str}", "node 'chat': a steer inbox needs `str` output"],
])("a bad steer inbox points at its line: %p %p", (line, output, message) => {
  const error = loadError(() => assembleText(steerFlow(line, output)));

  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
});

test("a mapping steer_inbox is refused", () => {
  const error = loadError(() => assembleText(steerFlow("    steer_inbox:\n      from: ${input.notes}")));
  expect(error.message).toContain("`steer_inbox:` must be `${input.<name>}` naming an Inbox input");
});

test("an input named __steer_inbox is reserved", () => {
  const error = loadError(() =>
    assembleText(steerFlow("    steer_inbox: ${input.notes}\n    input:\n      __steer_inbox: ${input.notes}")),
  );
  expect(error.message).toContain("input '__steer_inbox' is reserved for `steer_inbox:`; rename the input");
});

test("agent node rejects a non-bool steerInbox", () => {
  expect(() => new AgentNode("chat", { prompt: "p", steerInbox: 1 as unknown as boolean })).toThrow(
    new TypeError("AGENT node 'chat': steerInbox must be a bool"),
  );
});

test.each([
  [
    "a str input",
    "    steer_inbox: ${input.text}",
    "  notes: !Inbox(str)\n  text: !str",
    "node 'chat' steer_inbox: `${input.text}` isn't an inbox; `steer_inbox:` names an Inbox input",
  ],
  [
    "a node output",
    "    steer_inbox: ${other.output}",
    "  notes: !Inbox(str)",
    "node 'chat' steer_inbox: `${other.output}` isn't an inbox; `steer_inbox:` names an Inbox input",
  ],
  ["text", '    steer_inbox: "${input.notes} x"', "  notes: !Inbox(str)", "not inside text"],
  ["a field read", "    steer_inbox: ${input.notes.x}", "  notes: !Inbox(str)", "has no fields"],
])("steer_inbox refuses %s", (_label, line, input, message) => {
  const flow = steerFlow(line, "    output: str", input).replace(
    "output: ${chat.output}",
    "  other:\n    kind: code\n    code: return 1\n  keep:\n    kind: tool\n    tool_id: take\n    input:\n" +
      "      from: ${input.notes}\noutput: ${chat.output}",
  );
  const error = loadError(() => assembleText(flow));
  expect(error.message).toContain(message);
});

test("an inbox read only by steer_inbox is read", () => {
  flowOf(steerFlow("    steer_inbox: ${input.notes}"));
});

test("a loop body's agent reads the body's Inbox(Message) input", () => {
  const body = `\
id: body
name: body
input:
  owner_messages: !Inbox(Message)
  done: !bool
nodes:
  chat:
    kind: agent
    prompt: Help the owner.
    steer_inbox: \${input.owner_messages}
    output: str
output:
  done: true
`;
  const chat = assembleText(body).flow.compiled.nodes.get("chat") as AgentNode;
  expect(chat.steerInbox).toBe(true);
});

test.each<CaseRow>([{ id: "load-agent/compose_agent_steer_inbox__steerable_is_replaced", call: parsedNodes }])(
  "parseNodes: $id",
  runCase,
);

test.each<CaseRow>([
  { id: "load-agent/compose_agent_steer_inbox__not_an_inbox", call: assembled },
  { id: "load-agent/compose_agent_steer_inbox__needs_str_output", call: assembled },
  { id: "load-agent/compose_agent_steer_inbox__an_optional_str_agent_loads", call: assembled },
])("assemble: $id", runCase);
