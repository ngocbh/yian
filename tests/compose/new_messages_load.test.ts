/**
 * `${x.new_messages}`, an AGENT's second readable value, typed `list(Message)`: what
 * loads and where, and each load error; a comprehension variable over it is not a reference.
 */

import { afterEach, expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import { classifyPath } from "../../src/compile/validation.ts";
import { refProducer } from "../../src/compose/edges/producers.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { referenceContract } from "../../src/compile/output_contracts.ts";
import { forget } from "../../src/defaults.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { messagesKey } from "../../src/typesys/pool.ts";
import { newMessagesType, typeFor } from "../../src/typesys/types.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";

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

/** A flow whose agent `reply` runs only when `input.go`, with output `${output}`. */
const MAYBE_SKIPPED = (output: string) => `
id: f
name: f
input:
  go: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: "\${input.go}"
        then: reply
    else: other
  reply:
    kind: agent
    prompt: hi
    output: str
  other:
    kind: code
    code: return 1
    output: int
output: ${output}
`;

test("a maybe-skipped agent's new_messages is null unless the read has a default", async () => {
  // As for `.output`, a flow output reading only a maybe-skipped agent gets the null default.
  const bare = await loadFlow(MAYBE_SKIPPED("${reply.new_messages}"));
  expect(bare.compiled.outputs.map((output) => output.binding.source)).toEqual(["${reply.new_messages:-null}"]);
  const defaulted = await loadFlow(MAYBE_SKIPPED("${reply.new_messages :- []}"));
  expect(defaulted.compiled.outputs.map((output) => output.binding.source)).toEqual(["${reply.new_messages :- []}"]);
});

test("a typed !list(Message) input takes a maybe-skipped agent's new_messages with :- []", async () => {
  await loadFlow(
    MAYBE_SKIPPED("${reader.output}").replace(
      "output: ${reader.output}",
      `  reader:
    kind: code
    input:
      history: !list(Message) \${reply.new_messages :- []}
    output: int
    code: return inputs['history'].length
output: \${reader.output}`,
    ),
  );
});

/** A flow whose agent `reply` is checked by the flow assert `check`. */
const ASSERTED = (check: string) => `${FLOW("", "${reply.output}")}asserts:
  - ${check}
`;

test("a flow assert reading reply.new_messages loads, as a post-assert", async () => {
  const flow = await loadFlow(ASSERTED("reply.new_messages != null"));
  const end = flow.compiled.nodes.get("__end__")!;
  expect(end.postAsserts.map((assertion) => assertion.expression)).toEqual(["__assert_reply_new_messages != null"]);
  expect(flow.compiled.edges.some((edge) => edge.from === "reply" && edge.to === "__end__")).toBe(true);
});

test("a flow assert's field read into new_messages is checked against list(Message)", async () => {
  const error = await loadErrorAsync(loadFlow(ASSERTED("reply.new_messages.nope == 'x'")));
  expect(error.message).toContain("reads field 'nope' on type list(");
});

test("a flow assert reading new_messages of a CODE node is refused", async () => {
  const flow = `${FLOW(CODE_READER("      history: ${reply.new_messages}"))}asserts:
  - reader.new_messages != null
`;
  expect((await loadErrorAsync(loadFlow(flow))).message).toContain(NOT_AN_AGENT("reader"));
});

test("a node's own assert reading its new_messages is refused: it reads only its locals", async () => {
  const flow = FLOW("", "${reply.output}").replace(
    "    output: str\n",
    "    output: str\n    asserts:\n      - reply.new_messages != null\n",
  );
  expect((await loadErrorAsync(loadFlow(flow))).message).toContain(
    "${reply.new_messages} is not a declared input or `${output}`",
  );
});

/** A flow whose CASE `gate` (lines indented two spaces) routes to `a` or `b` after agent `reply`. */
const ROUTED = (gate: string) => `${FLOW(`${gate}
  a:
    kind: code
    code: return 1
    output: int
  b:
    kind: code
    code: return 2
    output: int`, "${a.output :- b.output}")}`;

test("a CASE can't route on new_messages, in on: or in a when:", async () => {
  for (const gate of [
    `  gate:\n    kind: case\n    on: \${reply.new_messages}\n    cases:\n      - when: x\n        then: a\n    else: b`,
    `  gate:\n    kind: case\n    cases:\n      - when: "\${reply.new_messages != []}"\n        then: a\n    else: b`,
  ]) {
    const message = (await loadErrorAsync(loadFlow(ROUTED(gate)))).message;
    expect(message).toContain("node 'gate'");
    expect(message).toContain(
      "reference ${reply.new_messages}: a case can't route on an agent's new_messages yet; " +
        "read it in a node input, and route on that node's output",
    );
    expect(message).not.toContain("isn't one there");
  }
});

test("a reader behind a CASE branch reads new_messages", async () => {
  const flow = await loadFlow(
    ROUTED(`  gate:\n    kind: case\n    cases:\n      - when: "\${reply.output == 'x'}"\n        then: a\n    else: b`).replace(
      "  a:\n    kind: code\n    code: return 1\n",
      "  a:\n    kind: code\n    input:\n      history: !list(Message) ${reply.new_messages}\n    code: return inputs['history'].length\n",
    ),
  );
  expect(flow.compiled.edges.some((edge) => edge.from === "reply" && edge.to === "a")).toBe(true);
});

test("a comprehension over reply.new_messages reads the node, not its variables", async () => {
  // The variable `reply` shadows the node inside; only the first iterable reads the node.
  const flow = await loadFlow(FLOW(CODE_READER("      history: ${[reply.role for reply in reply.new_messages]}", "list(str)")));
  expect(flow.compiled.edges.some((edge) => edge.from === "reply" && edge.to === "reader")).toBe(true);
  expect(rensInternal(new Binding("${[r.output for r in reply.new_messages]}"), "cs")).toEqual(
    new Binding("${[r.output for r in cs/reply.new_messages]}"),
  );
});

test("a flow assert's comprehension over reply.new_messages renames only the iterable", async () => {
  const flow = await loadFlow(ASSERTED("len([m for m in reply.new_messages if m.role == 'assistant']) > 0"));
  const end = flow.compiled.nodes.get("__end__")!;
  expect(end.postAsserts.map((assertion) => assertion.expression)).toEqual([
    "len([m for m in __assert_reply_new_messages if m.role == 'assistant']) > 0",
  ]);
});

/** An inline CALL `reader` whose input `h` binds `value` and whose body returns `h`'s length. */
const INLINE_READER = (value: string) => `  reader:
    kind: call
    input:
      h: ${value}
    nodes:
      count:
        kind: code
        input:
          h: \${input.h}
        output: int
        code: return inputs['h'].length
    output: \${count.output}`;

test("an untagged inline call input reading reply.new_messages loads, typed list(Message)", async () => {
  const flow = await loadFlow(FLOW(INLINE_READER("${reply.new_messages}")));
  expect(flow.compiled.edges.some((edge) => edge.from === "reply" && edge.to === "reader")).toBe(true);
});

test("an untagged inline call input's field read into reply.new_messages is checked", async () => {
  const error = await loadErrorAsync(loadFlow(FLOW(INLINE_READER("${reply.new_messages.role}"))));
  expect(error.message).toContain("role");
});

test("an untagged inline call input reading a CODE node's new_messages names both forms", async () => {
  const flow = FLOW(`${CODE_READER("      history: ${reply.new_messages}")}
${INLINE_READER("${reader.new_messages}").replace("  reader:", "  second:")}`, "${second.output}");
  const message = (await loadErrorAsync(loadFlow(flow))).message;
  expect(message).toContain("new_messages");
  expect(message).toContain("AGENT");
});

test("an untagged inline call input reading another field names both forms", async () => {
  const message = (await loadErrorAsync(loadFlow(FLOW(INLINE_READER("${reply.nope}"))))).message;
  expect(message).toContain("must select reply.output[.field] or an AGENT's reply.new_messages");
});
