/**
 * TS-only: `${x.new_messages}`, an AGENT's second readable value, typed `list(Message)`: what
 * loads and where, and each load error.
 */

import { afterEach, expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import { classifyPath } from "../../src/compile/validation.ts";
import { refProducer } from "../../src/compose/edges/producers.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { referenceContract } from "../../src/compose/helpers/output_contracts.ts";
import { forget } from "../../src/defaults.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { messagesKey } from "../../src/typesys/pool.ts";
import { newMessagesType, typeFor } from "../../src/typesys/types.ts";
import { loadErrorAsync } from "./declarations.ts";

/** The load error for reading `node`'s new messages when it isn't an AGENT, or not here. */
const NOT_AN_AGENT = (node: string) =>
  `reference \${${node}.new_messages} reads new_messages, which only an AGENT node has, from a node input, ` +
  `flow output or flow assert; node '${node}' isn't one there`;

afterEach(() => forget());

test("refProducer gives the node for x.output and x.new_messages", () => {
  expect(refProducer("reply.new_messages")).toBe("reply");
  expect(refProducer("reply.new_messages.0.parts")).toBe("reply");
  expect(refProducer("reply.output")).toBe("reply");
  expect(refProducer("item.new_messages")).toBeNull();
  expect(refProducer("system.new_messages")).toBeNull();
});

test("rensInternal renames x.new_messages into the callsite's namespace", () => {
  expect(rensInternal(new Binding("${reply.new_messages :- []}"), "turn#3")).toEqual(
    new Binding("${turn#3/reply.new_messages :- []}"),
  );
  expect(rensInternal(new Binding("${reply.new_messages + more.output}"), "cs")).toEqual(
    new Binding("${cs/reply.new_messages + cs/more.output}"),
  );
});

test("classifyPath reads x.new_messages only for a messages target", () => {
  const targets = new Set(["reply", messagesKey("reply"), "code"]);
  expect(classifyPath("reply.new_messages", targets, new Set())).toBeNull();
  expect(classifyPath("code.new_messages", targets, new Set())).toBe(NOT_AN_AGENT("code"));
  expect(classifyPath("code.nope", targets, new Set())).toContain("must be code.output[.field]");
});

test("referenceContract reads x.new_messages as the producers entry under x.new_messages", () => {
  const messages = new ValueContract(newMessagesType());
  const producers = new Map([
    ["reply", new ValueContract(typeFor("str", new Map()))],
    [messagesKey("reply"), messages],
  ]);
  expect(referenceContract("reply.new_messages", producers, new Map())).toBe(messages);
  expect(referenceContract("code.new_messages", producers, new Map())).toBeNull();
});

/** A flow whose agent `reply` is read by the node lines `reader` (indented two spaces), output `${output}`. */
const FLOW = (reader: string, output = "${reader.output}") => `
id: f
name: f
nodes:
  reply:
    kind: agent
    prompt: hi
    output: str
${reader}
output: ${output}
`;

/** A CODE node `reader` with the input lines `input` (indented six spaces). */
const CODE_READER = (input: string, output = "int") => `  reader:
    kind: code
    input:
${input}
    output: ${output}
    code: return inputs['history'].length`;

test("a CODE input reading reply.new_messages loads, with an edge from reply", async () => {
  const flow = await loadFlow(FLOW(CODE_READER("      history: ${reply.new_messages}")));
  expect(flow.compiled.edges.some((edge) => edge.from === "reply" && edge.to === "reader")).toBe(true);
});

test("a typed !list(Message) input accepts it; !list(str) refuses it", async () => {
  await loadFlow(FLOW(CODE_READER("      history: !list(Message) ${reply.new_messages}")));
  const error = await loadErrorAsync(loadFlow(FLOW(CODE_READER("      history: !list(str) ${reply.new_messages}"))));
  expect(error.message).toContain("history");
});

test("reading new_messages of a CODE node is a load error naming the field", async () => {
  const flow = FLOW(`${CODE_READER("      history: ${reply.new_messages}")}
  second:
    kind: code
    input:
      history: \${reader.new_messages}
    output: int
    code: return 1`, "${second.output}");
  expect((await loadErrorAsync(loadFlow(flow))).message).toContain(NOT_AN_AGENT("reader"));
});

test("a field read into new_messages is checked", async () => {
  const error = await loadErrorAsync(loadFlow(FLOW(CODE_READER("      history: ${reply.new_messages.role}"))));
  expect(error.message).toContain("role");
});

test("a flow output may read it, typed list(Message)", async () => {
  const flow = await loadFlow(FLOW("", "${reply.new_messages}"));
  expect(flow.compiled.edges.some((edge) => edge.from === "reply")).toBe(true);
});

test("a second agent splices the first's new_messages into its prompt list", async () => {
  await loadFlow(FLOW(`  reader:
    kind: agent
    input:
      history: \${reply.new_messages}
    prompt:
      - \${history}
      - user: again
    output: str`));
});

test("an llm_config source reading reply.new_messages is refused with the same message", async () => {
  const flow = FLOW(`  reader:
    kind: agent
    prompt: hi
    llm_config:
      provider: anthropic
      model: \${reply.new_messages}
    output: str`);
  const message = (await loadErrorAsync(loadFlow(flow))).message;
  expect(message).toContain("llm_config");
  expect(message).toContain("reads new_messages, which only an AGENT node has");
});
