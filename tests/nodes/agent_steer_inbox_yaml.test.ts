/**
 * `steer_inbox:` on AGENT: the agent reads steers from a declared top-level inbox, named by a lone
 * `${input.<name>}`. The retired `steerable:` is a load error with a hint.
 */

import { afterEach, expect, test } from "bun:test";

import { forget } from "../../src/defaults.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "../helpers/declarations.ts";

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
])("a bad steer inbox points at its line: %p %p", (line, output, message) => {
  const error = loadError(() => assembleText(steerFlow(line, output)));

  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
});

test("steer_inbox: {from, render} loads with the render and the reserved wiring", () => {
  const flow = flowOf(steerFlow("    steer_inbox:\n      from: ${input.notes}\n      render: |-\n        From the owner: ${event}"));
  const chat = flow.nodes.get("chat") as AgentNode;
  expect(chat.steerInbox).toBe(true);
  expect(chat.steerRender).toBe("From the owner: ${event}");
  expect((flow.wiring.get("chat")!.get(STEER_INBOX_KEY) as Binding).source).toBe("${input.notes}");
});

test("steer_inbox: {from} alone loads with no render", () => {
  const chat = chatOf(steerFlow("    steer_inbox:\n      from: ${input.notes}"));
  expect([chat.steerInbox, chat.steerRender]).toEqual([true, null]);
});

test.each([
  ["no from", "    steer_inbox:\n      render: x", "`steer_inbox:` needs `from:` naming an Inbox input", 9],
  ["an unknown key", "    steer_inbox:\n      from: ${input.notes}\n      when: x", "`steer_inbox:` takes only `from:` and `render:`", 11],
  ["an empty render", '    steer_inbox:\n      from: ${input.notes}\n      render: ""', "`steer_inbox.render:` must be non-empty text", 11],
  ["a render that isn't text", "    steer_inbox:\n      from: ${input.notes}\n      render: 3", "`steer_inbox.render:` must be non-empty text", 11],
  ["an empty mapping", "    steer_inbox: {}", "`steer_inbox:` needs `from:` naming an Inbox input", 9],
])("a steer_inbox mapping with %s is refused at its line", (_label, line, message, startLine) => {
  const error = loadError(() => assembleText(steerFlow(line)));
  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(startLine);
});

/** A steer flow whose `steer_inbox:` reads `notes` with render `render`, under `input`. */
const renderFlow = (render: string, input = "  notes: !Inbox(str)") =>
  steerFlow(`    steer_inbox:\n      from: \${input.notes}\n      render: |-\n        ${render}`, "    output: str", input);

test.each([
  ["an input", "${input.x}", "node 'chat' steer_inbox render: `${input.x}` names `input`; a steer render reads only `${event}`"],
  ["a node output", "${reply.output}", "node 'chat' steer_inbox render: `${reply.output}` names `reply`; a steer render reads only `${event}`"],
  ["a malformed span", "${event.}", "node 'chat' steer_inbox render: "],
])("a render reading %s is refused at the render", (_label, render, message) => {
  const error = loadError(() => assembleText(renderFlow(render, "  notes: !Inbox(str)\n  x: !str")));
  expect(error.message).toContain(message);
  // The block scalar's text starts on the line after `render: |-`.
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(12);
});

const NOTE_TYPES = "typedefs:\n  Note:\n    text: str\n";

test("a render's field reads are checked against the inbox's element type", () => {
  const typed = (render: string) => NOTE_TYPES + renderFlow(render, "  notes: !Inbox(Note)");
  expect(chatOf(typed("Note: ${event.text}")).steerRender).toBe("Note: ${event.text}");
  const error = loadError(() => assembleText(typed("Note: ${event.body}")));
  expect(error.message).toContain("node 'chat' steer_inbox render: ");
  expect(error.message).toContain("body");
});

test("a render over an Inbox(Message) reads its fields", () => {
  const chat = chatOf(renderFlow("${event.author}: ${event.parts}", "  notes: !Inbox(Message)"));
  expect(chat.steerRender).toBe("${event.author}: ${event.parts}");
});

test.each(["int", "{answer: str}"])("a steer inbox loads with any output, here %p", (output) => {
  expect(() => assembleText(steerFlow("    steer_inbox: ${input.notes}", `    output: ${output}`))).not.toThrow();
});

test("an input named __steer_inbox is reserved", () => {
  const error = loadError(() =>
    assembleText(steerFlow("    steer_inbox: ${input.notes}\n    input:\n      __steer_inbox: ${input.notes}")),
  );
  expect(error.message).toBe("node 'chat' input '__steer_inbox' starts with '__', which yian keeps for the names it generates; rename it");
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
  { id: "load-agent/compose_agent_steer_inbox__any_output_type_loads", call: assembled },
  { id: "load-agent/compose_agent_steer_inbox__an_optional_str_agent_loads", call: assembled },
])("assemble: $id", runCase);
