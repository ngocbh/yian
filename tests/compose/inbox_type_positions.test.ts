/** Where an Inbox(...) type may be written: only as a flow or def input. */
import { expect, test } from "bun:test";

import { START_ID } from "../../src/compile/model.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, isInboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "../helpers/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });
/** A child that takes inbox `m` and returns nothing. */
const SINK = handChild({ inputs: [inboxParam("m")], output: null });
/** The top-level keys every flow text starts with. */
const HEAD = "id: f\nname: f\n";
/** The node every refused row adds, so `m` is read once at an allowed site. */
const KEEP = `  keep:
    kind: call
    call: sink
    input:
      m: \${input.m}
`;

test("a flow input may be an inbox", () => {
  const { flow } = assembleText(`${HEAD}input:
  m: !Inbox(str)
nodes:
${KEEP}output: null
`, "flow.yaml", { children: { sink: SINK } });
  const param = flow.compiled.nodes.get(START_ID)!.params!.find((p) => p.name === "m")!;
  expect(isInboxType(param.contract!.type)).toBe(true);
  expect(param.required).toBe(true);
});

const refuse = (text: string) =>
  loadError(() => assembleText(HEAD + text, "flow.yaml", { children: { sink: SINK } })).message;

test.each([
  ["a default", "input:\n  m: !Inbox(str) x\nnodes:\n" + KEEP + "output: null\n", "an inbox input takes no default"],
  ["Optional", "input:\n  m: !Optional(Inbox(str))\nnodes: {}\noutput: null\n", "an inbox is never optional"],
  ["a node output", "nodes:\n  t:\n    kind: tool\n    tool_id: f\n    output: Inbox(str)\noutput: null\n", "an output can't be an inbox"],
  ["a node output's record field", "nodes:\n  t:\n    kind: tool\n    tool_id: f\n    output:\n      a: Inbox(str)\noutput: null\n", "an output can't be an inbox"],
  ["a typed binding", "input:\n  a: !str\nnodes:\n  t:\n    kind: tool\n    tool_id: f\n    input:\n      x: !Inbox(str) ${input.a}\noutput: null\n", "only a flow input can be"],
  ["a nested tag in a default", "input:\n  a: !object\n    x: !Inbox(str) y\nnodes: {}\noutput: null\n", "only a flow input can be"],
  ["a flow output", "nodes: {}\noutput: !Inbox(str) null\n", "only a flow input can be"],
  ["a variable", "nodes:\n  a:\n    kind: agent\n    prompt: hi\n    variables:\n      v: !Inbox(str) x\noutput: null\n", "only a flow input can be"],
])("refuses %s", (_label, text, message) => {
  expect(refuse(text)).toContain(message);
});
