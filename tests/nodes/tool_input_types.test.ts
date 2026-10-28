/** Loading stamps each TOOL node input's proven inferred type on the node. */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { Node } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import type { ToolNode } from "../../src/nodes/tool/node.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
import { assembleText } from "./declarations.ts";

/** Return the nodes of flow `text`'s root, assembled as the loader does. */
const loadedNodes = (text: string) => assembleText(text).flow.compiled.nodes;

/** Return a TOOL node's stamped input types as canonical labels. */
function labels(node: Node | undefined): Record<string, string> {
  const types = (node as ToolNode).inputTypes;
  return Object.fromEntries([...types].map(([name, typ]) => [name, canonicalTypeLabel(typ)]));
}

/** Return a flow whose TOOL node `t` binds `v` as `binding`. */
function toolFlow(binding: string): string {
  return `
id: stamped
name: stamped
input:
  n: !int
  a: !str
  b: !str
nodes:
  c:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.n}
  t:
    kind: tool
    tool_id: place_order
    input:
      v: ${binding}
output: \${t.output}
`;
}

test.each<[string, string]>([
  ["${input.n}", "int"],
  ['"${input.a}-${input.b}"', "str"],
  ["5", "int(5..5)"],
  ["true", "bool"],
  ['"x"', "Literal(x)"],
  ["[1, 2]", "list(int)"],
  ["!float ${input.n}", "float"],
  ['""', "str"],
  ['"$${x}"', 'Literal("${x}")'],
  ["1.5", "float(1.5..1.5)"],
  ["null", "None"],
])("a proven input %s is stamped with its inferred type %s", (binding, label) => {
  expect(labels(loadedNodes(toolFlow(binding)).get("t"))).toEqual({ v: label });
});

test("a non-finite number literal is absent", () => {
  expect(labels(loadedNodes(toolFlow(".inf")).get("t"))).toEqual({});
});

test("an unproven input is absent", () => {
  expect(labels(loadedNodes(toolFlow("${c.output}")).get("t"))).toEqual({});
});

test("a tool in a child flow is stamped", () => {
  const nodes = loadedNodes(`
id: outer
name: outer
input:
  n: !int
defs:
  sub:
    input:
      m: !int
    nodes:
      t:
        kind: tool
        tool_id: place_order
        input:
          v: \${input.m}
    output: \${t.output}
nodes:
  use:
    kind: call
    call: sub
    input:
      m: \${input.n}
output: \${use.output}
`);

  const child = (nodes.get("use") as CallNode).child as CompiledFlow;
  expect(labels(child.nodes.get("t"))).toEqual({ v: "int" });
});

test("a case-narrowed input is stamped with the member record", () => {
  const nodes = loadedNodes(`
id: narrowed
name: narrowed
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Decision: Union(Buy, Sell)
input:
  decision: !object
nodes:
  d:
    kind: code
    input:
      decision: \${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    on: \${d.output.action}
    cases:
      - when: sell
        then: t
    else: hold
  t:
    kind: tool
    tool_id: place_order
    input:
      order: \${d.output}
  hold:
    kind: code
    depends_on: [d]
    code: |
      return "hold"
output: \${t.output | hold.output}
`);

  const stamped = (nodes.get("t") as ToolNode).inputTypes.get("order")!;
  expect(stamped.variants).toBeUndefined();
  expect(new Set(stamped.fields!.keys())).toStrictEqual(new Set(["action", "qty", "price"]));
});

test("a map body tool reads the element type", () => {
  const nodes = loadedNodes(`
id: batch
name: batch
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: \${input.xs}
    input:
      x: \${item}
    nodes:
      t:
        kind: tool
        tool_id: place_order
        input:
          x: \${input.x}
    output: \${t.output}
output: \${each.output}
`);

  const child = (nodes.get("each") as MapNode).child as CompiledFlow;
  expect(labels(child.nodes.get("t"))).toEqual({ x: "int" });
});
