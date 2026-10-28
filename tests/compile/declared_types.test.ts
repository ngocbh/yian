/**
 * The declared types the run time derives from a compiled flow: pool roots
 * (`poolDeclaredTypes`), a node's locals (`localTypes`) and a MAP body's `item`
 * (`itemDeclaredType`). A field these types declare reads null when it's absent.
 */

import { expect, test } from "bun:test";

import { itemDeclaredType, localTypes, poolDeclaredTypes } from "../../src/compile/declared_types.ts";
import { cloneChild } from "../../src/compile/expand.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { newMessagesType } from "../../src/typesys/types.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

// A declared type's nullability doesn't affect which fields it declares, so `declaredAt` drops it.
const str = Type.scalar(ValueKind.STRING);

const REC = "typedefs:\n  Rec:\n    k: str\n    opt: Optional(str)\n";

/** Load `text` and return its compiled flow. */
async function compiled(text: string): Promise<CompiledFlow> {
  return (await loadFlow(text)).compiled;
}

/** Expect `actual` to be a type equal to `expected`. */
function expectType(actual: Type | null, expected: Type): void {
  expect(actual).not.toBeNull();
  expect(typeEquals(actual!, expected)).toBe(true);
}

const POOL_FLOW = `id: f
name: f
llm_config: {provider: anthropic, model: claude-test}
${REC}input:
  k: !str
  note: !Optional(str)
nodes:
  a:
    kind: code
    output: Rec
    code: |
      return {k: "x"}
  b:
    kind: code
    code: |
      return {k: "x"}
  chat:
    kind: agent
    prompt: Help.
    output: str
output:
  x: \${a.output.k}
  y: \${b.output}
  z: \${chat.output}
`;

test("pool roots take their producers' and inputs' declared types", async () => {
  const flow = await compiled(POOL_FLOW);
  const declared = poolDeclaredTypes(flow);
  const rec = declared(["a", "output"]);
  expect(rec?.fields?.has("opt")).toBe(true);
  expectType(declared(["a", "output", "k"]), str);
  expectType(declared(["a", "output", "opt"]), str);
  expect(declared(["a", "output", "nope"])).toBeNull();
  expect(declared(["b", "output"])).toBeNull();
  expect(declared(["ghost", "output"])).toBeNull();
  expectType(declared(["input", "k"]), str);
  expectType(declared(["input", "note"]), str);
  expect(declared(["input", "absent"])).toBeNull();
  expectType(declared(["chat", "new_messages"]), newMessagesType());
  expect(declared(["a", "new_messages"])).toBeNull();
  expectType(declared(["system", "now"]), str);
  expect(declared(["x"])).toBeNull();
});

test("a START's output is the record of its params", async () => {
  const flow = await compiled(POOL_FLOW);
  const declared = poolDeclaredTypes(flow);
  expectType(declared([flow.startId, "output", "note"]), str);
  expect(declared([flow.startId, "output", "nope"])).toBeNull();
});

test("a cloned CALL child's START declares its params", async () => {
  const child = await compiled(`id: c
name: c
input:
  k: !str
  note: !Optional(str)
nodes: {}
output: \${input.note :- "none"}
`);
  const cloned = cloneChild(child, "cs", { k: "a" });
  const start = [...cloned.nodes.keys()].find((id) => id.endsWith("__start__"))!;
  const declared = poolDeclaredTypes(cloned);
  expectType(declared([start, "output", "note"]), str);
  expect(declared([start, "output", "nope"])).toBeNull();
});

const LOCALS_FLOW = `id: f
name: f
llm_config: {provider: anthropic, model: claude-test}
typedefs:
  Rec:
    k: str
    opt: Optional(str)
  Ev:
    text: str
    opt: Optional(str)
input:
  notes: !Inbox(Ev)
  xs: !list(Rec)
nodes:
  a:
    kind: code
    output: Rec
    code: |
      return {k: "x"}
  use:
    kind: code
    output: Rec
    input:
      r: \${a.output :- null}
      t: !Rec \${a.output}
    asserts:
      - "\${output.k != null}"
    code: |
      return r
  pick:
    kind: case
    cases:
      - when: \${a.output.opt == null}
        then: yes
    else: no
  "yes":
    kind: code
    code: |
      return 1
  "no":
    kind: code
    code: |
      return 2
  chat:
    kind: agent
    steer_inbox:
      from: \${input.notes}
      render: "\${event.text}"
    prompt: Help.
    output: str
  each:
    kind: map
    over: \${input.xs}
    input:
      v: !str \${item.opt :- "d"}
    nodes:
      one:
        kind: code
        code: |
          return input.v
    output: \${one.output}
asserts:
  - "\${a.output.k != null}"
output:
  x: \${use.output}
  e: \${a.output}
  c: \${chat.output}
  m: \${each.output}
`;

/** Expect `actual` to be a record type declaring `field`. */
function expectDeclares(actual: Type | null | undefined, field: string): void {
  expect(actual?.fields?.has(field)).toBe(true);
}

test("a node's locals take their params' or bindings' declared types", async () => {
  const flow = await compiled(LOCALS_FLOW);
  const use = localTypes(flow, flow.nodes.get("use")!);
  expectDeclares(use.get("r"), "opt"); // untyped, from its pass-through binding
  expectDeclares(use.get("t"), "opt"); // !Rec
  expectDeclares(use.get("output"), "opt"); // the node's own output, for a post assert
  expect(localTypes(flow, flow.nodes.get("use")!)).toBe(use); // memoized per node
  expect([...use.keys()].sort()).toEqual(["output", "r", "t"]); // iterating computes every entry
  expectDeclares(localTypes(flow, flow.nodes.get("pick")!).get("__r0"), "opt"); // a searched CASE's hoisted root
  const event = localTypes(flow, flow.nodes.get("chat")!).get("event");
  expectDeclares(event, "text"); // the steer inbox's element type
  expectDeclares(localTypes(flow, flow.nodes.get(flow.endId)!).get("__assert_a"), "opt"); // a flow assert's local
  expect(localTypes(flow, flow.nodes.get("yes")!).get("anything")).toBeUndefined();
});

test("item in a MAP has the element type of its item source", async () => {
  const flow = await compiled(LOCALS_FLOW);
  expectDeclares(itemDeclaredType(flow, flow.nodes.get("each")!), "opt");
  expect(itemDeclaredType(flow, flow.nodes.get("use")!)).toBeNull();
});
