/** A loop carries an inbox into its body and leaves it out of its output. */
import { expect, test } from "bun:test";

import { type CompiledFlow, START_ID } from "../../src/compile/model.ts";
import { carriedFieldTypes } from "../../src/compose/nodes/loop.ts";
import { type LoopNode } from "../../src/nodes/loop/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, isInboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "./declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });
/** A child that takes inbox `m` and returns nothing. */
const SINK = handChild({ inputs: [inboxParam("m")], output: null });
/** A loop body that takes inbox `m` and a bool `done`, and returns `{done}`. */
const BODY = handChild({
  inputs: [inboxParam("m"), new Parameter("done", { required: true, contract: new ValueContract(BOOL) })],
  output: { done: "${input.done}" },
});
const CHILDREN = { children: { sink: SINK, body: BODY } };

const named = (predicate: string) => `id: f
name: f
input:
  m: !Inbox(str)
nodes:
  l:
    kind: loop
    call: body
    input:
      m: \${input.m}
      done: false
    while: ${predicate}
output: null
`;

const loopOf = (text: string) => assembleText(text, "flow.yaml", CHILDREN).flow.compiled.nodes.get("l") as LoopNode;
const fieldsOf = (node: LoopNode) => [...carriedFieldTypes(node.outputContract).keys()];

test("a loop's output leaves out the inbox it carries", () => {
  const node = loopOf(named("not done"));
  expect(node.returnedFields).toEqual(["done"]);
  expect(fieldsOf(node)).toEqual(["done"]);
});

test("a loop predicate can't read an inbox", () => {
  expect(loadError(() => loopOf(named("m == null"))).message).toContain(
    "loop node 'l': `while:` reads `m`, an inbox; a loop predicate can't read an inbox",
  );
});

/** A flow whose inline loop body seeds `m` from the inbox; `body` is the body's extra nodes. */
const inline = (body: string) => `id: f
name: f
input:
  m: !Inbox(str)
nodes:
  l:
    kind: loop
    while: not done
    input:
      m: \${input.m}
      done: false
    nodes:
${body}      d:
        kind: code
        output:
          done: bool
        code: |
          return {"done": true}
    output: \${d.output}
output: null
`;

const READ_M = `      s:
        kind: call
        call: sink
        input:
          m: \${input.m}
`;

test("an inline body's seed infers its Inbox type", () => {
  const node = loopOf(inline(READ_M));
  const param = node.childInputs.find((p) => p.name === "m")!;
  expect(isInboxType(param.contract!.type)).toBe(true);
  const start = (node.child as CompiledFlow).nodes.get(START_ID)!;
  expect(isInboxType(start.params!.find((p) => p.name === "m")!.contract!.type)).toBe(true);
  expect(fieldsOf(node)).toEqual(["done"]);
});

test("an inline body that doesn't read its inbox still loads", () => {
  expect(fieldsOf(loopOf(inline("")))).toEqual(["done"]);
});
