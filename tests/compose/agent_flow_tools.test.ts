/**
 * An AGENT's `tools:` naming a `defs:` entry or a `uses:` alias bakes that flow onto the agent as
 * a flow tool: its name, its `description:`, the JSON Schema of its inputs and its compiled flow.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { CompiledFlow } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-flow-tools-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

writeFileSync(
  join(dir, "summarize.yaml"),
  `id: summarize
name: summarize
description: Summarize a text in one sentence.
input:
  text: !str
  words: !int(1..100) 20
  note: !Optional(str)
nodes:
  line: !str \${input.text}
output: \${line.output}
`,
);

const FLOW = `
id: f
name: f
uses:
  summarize: summarize
typedefs:
  Ticket: {title: str, tags: list(str)}
defs:
  classify:
    description: Sort a ticket.
    input:
      ticket: !Ticket
    nodes:
      pick: !str bug
    output: \${pick.output}
  shout:
    description: Shout the text back.
    kind: code
    input:
      text: !str
    output: !str
    code: return inputs.text.toUpperCase()
nodes:
  helper:
    kind: agent
    prompt: Help.
    output: !str
    tools: [search, summarize, classify, shout]
output: \${helper.output}
`;

/** The agent `helper` of `FLOW`, loaded with the fixture folder as its search path. */
async function helper(): Promise<AgentNode> {
  const flow = await loadFlow(FLOW, { searchPaths: [dir], sourceLabel: "flow.yaml" });
  return flow.compiled.nodes.get("helper") as AgentNode;
}

test("the flows an agent names in tools: become its flow tools, in grant order", async () => {
  const agent = await helper();
  expect(agent.flowTools.map((tool) => [tool.name, tool.description])).toEqual([
    ["summarize", "Summarize a text in one sentence."],
    ["classify", "Sort a ticket."],
    ["shout", "Shout the text back."],
  ]);
  for (const tool of agent.flowTools) expect(tool.child).toBeInstanceOf(CompiledFlow);
});

test("a flow grant is not a host grant, and its flow is a static child under its name", async () => {
  const agent = await helper();
  expect(agent.grantedTools()).toEqual([{ kind: "name", name: "search" }]);
  expect(agent.originChildSegments()).toEqual(["summarize", "classify", "shout"]);
  expect(agent.originChildren()).toEqual(agent.flowTools.map((tool) => tool.child));
});

test("a flow tool's parameters are a closed object of its inputs, required unless defaulted or Optional", async () => {
  const [summarize] = (await helper()).flowTools;
  expect(summarize!.parameters).toEqual({
    additionalProperties: false,
    properties: {
      text: { title: "Text", type: "string" },
      words: { maximum: 100n, minimum: 1n, title: "Words", type: "integer" },
      note: { anyOf: [{ type: "string" }, { type: "null" }], title: "Note" },
    },
    required: ["text"],
    type: "object",
  });
});

test("a record input's schema is closed and kept under $defs", async () => {
  const classify = (await helper()).flowTools[1]!;
  expect(classify.parameters).toEqual({
    $defs: {
      ticket: {
        additionalProperties: false,
        properties: {
          title: { title: "Title", type: "string" },
          tags: { items: { type: "string" }, title: "Tags", type: "array" },
        },
        required: ["title", "tags"],
        title: "ticket",
        type: "object",
      },
    },
    additionalProperties: false,
    properties: { ticket: { $ref: "#/$defs/ticket" } },
    required: ["ticket"],
    type: "object",
  });
});

test("a flow named by a group grant's tool is still the tool it names", async () => {
  // `{search: [summarize]}` names a group `search`, whose tool happens to share a flow's name.
  const flow = await loadFlow(FLOW.replace("tools: [search, summarize, classify, shout]", "tools: [{search: [summarize]}]"), {
    searchPaths: [dir],
  });
  const agent = flow.compiled.nodes.get("helper") as AgentNode;
  expect(agent.flowTools).toEqual([]);
  expect(agent.grantedTools()).toEqual([{ kind: "group", group: "search", tools: ["summarize"] }]);
});
