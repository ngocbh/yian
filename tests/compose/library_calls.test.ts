/**
 * Library calls: `call: {billing: quote}` on call, map and loop nodes and the directive
 * `call(billing.quote, ...)` run a library's def, baked as the node's child with `def` set, and
 * load errors beside the conformance cases': a missing def with no close match, a directive
 * naming a library, the nesting limit and the `call:` shape.
 */

import { expect, test } from "bun:test";

import { flowManifest } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import type { ChildFlowNode } from "../../src/nodes/child_flow.ts";
import { runFlow } from "../../src/run.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";

// `requote` calls its sibling `quote`, and `bump` steps a loop's carried record.
const LIBRARY = `id: billing
name: billing
defs:
  quote:
    description: Price an order.
    input: {qty: !int}
    kind: code
    code: 'return inputs.qty * 3'
    output: !int
  requote:
    input: {qty: !int}
    nodes:
      again: {kind: call, call: quote, input: {qty: "\${input.qty}"}}
    output: \${again.output}
  bump:
    input: {n: !int}
    nodes:
      next: {kind: code, input: {n: "\${input.n}"}, code: 'return {n: inputs.n + 1}', output: {n: !int}}
    output: {n: "\${next.output.n}"}
`;

const CLEAN = "id: clean\nname: clean\ninput:\n  text: !str\nnodes:\n  t: !str ${input.text}\noutput: ${t.output}\n";

/** Load `nodes` (YAML under `nodes:`) in a flow that imports the library as `billing` and a flow as `clean`. */
async function load(nodes: string, output = "${price.output}", env?: (name: string) => string | undefined) {
  const library = await loadFlow(LIBRARY, { sourceLabel: "billing.yaml" });
  const clean = await loadFlow(CLEAN);
  const uses = "uses:\n  billing: ./billing\n  clean: ./clean\n";
  const text = `id: app\nname: app\ninput:\n  qtys: !list(int)\n${uses}nodes:\n${nodes}output: ${output}\n`;
  return loadFlow(text, { childResolver: (path) => (path === "./billing" ? library : clean), env });
}

/** The manifest of `flow`'s node `id`. */
function manifestOf(flow: ComposedFlow, id: string): JsonObject {
  return (flowManifest(flow.compiled)["nodes"] as JsonObject)[id] as JsonObject;
}

/** The message loading `nodes` fails with. */
async function loadMessage(nodes: string): Promise<string> {
  return (await loadErrorAsync(load(nodes))).message;
}

test("call, map and loop nodes and the directive bake a library's def, with the alias in flowId", async () => {
  const flow = await load(
    "  price: {kind: call, call: {billing: requote}, input: {qty: 2}}\n" +
      '  each: {kind: map, over: "${input.qtys}", call: {billing: quote}, input: {qty: "${item}"}}\n' +
      "  count: {kind: loop, call: {billing: bump}, input: {n: 0}, until: n >= 3, max: 5}\n" +
      "  inline:\n    kind: code\n    input:\n      q: call(billing.quote, qty=4)\n    code: return inputs.q\n" +
      "    output: !int\n",
    '"${[price.output, each.output, count.output.n, inline.output]}"',
  );
  const node = (id: string) => flow.compiled.nodes.get(id) as ChildFlowNode;

  expect(["price", "each", "count", "__call_0"].map((id) => [node(id).flowId, node(id).def])).toEqual([
    ["billing", "requote"],
    ["billing", "quote"],
    ["billing", "bump"],
    ["billing", "quote"],
  ]);
  // The baked child is the def's own graph.
  expect((node("price").child as CompiledFlow).nodes.has("again")).toBe(true);
  expect(manifestOf(flow, "price")["def"]).toBe("requote");
  expect((await runFlow(flow, { qtys: [1, 2] })).output).toEqual([6n, [3n, 6n], 3n, 12n]);
});

test("a flow's call node has no def, and its manifest leaves the field out", async () => {
  const flow = await load("  price: {kind: call, call: clean, input: {text: hi}}\n");

  expect((flow.compiled.nodes.get("price") as CallNode).def).toBeNull();
  expect(Object.hasOwn(manifestOf(flow, "price"), "def")).toBe(false);
});

test("a missing def with no close match names the library's defs", async () => {
  expect(await loadMessage("  price: {kind: loop, call: {billing: zzz}, input: {n: 0}, times: 2}\n")).toBe(
    "loop node 'price': library 'billing' has no def 'zzz' (its defs: 'quote', 'requote', 'bump')",
  );
});

test("a directive naming a library without a def fails, showing the directive's def form", async () => {
  const nodes = "  price:\n    kind: code\n    input:\n      q: call(billing, qty=4)\n    code: return inputs.q\n";
  expect(await loadMessage(nodes)).toBe(
    "node 'price' input 'q' call(billing): callable 'billing' is a library; name a def: `call(billing.quote, ...)`",
  );
});

test("a library def's own calls count toward the nesting limit", async () => {
  const env = (name: string) => (name === "YIAN_MAX_CALL_NESTING" ? "1" : undefined);
  const nodes = "  price: {kind: call, call: {billing: requote}, input: {qty: 2}}\n";
  expect((await loadErrorAsync(load(nodes, undefined, env))).message).toContain(
    "call node 'price': calling def 'requote' of 'billing' nests calls 2 levels deep, past the call nesting limit of 1",
  );
});

test("a call: mapping to a list shows the one def to write", async () => {
  expect(await loadMessage("  price: {kind: call, call: {billing: [quote]}}\n")).toBe(
    "node 'price' (kind=call): `call: {billing: ...}` names one def; write `call: {billing: quote}`",
  );
});

test("a def named under a compiler-private callable is refused, as a call of one is", async () => {
  expect(await loadMessage("  price: {kind: call, call: {__inline_flow_x: quote}}\n")).toBe(
    "node 'price' (kind=call): call targets beginning with '__inline_flow_' are reserved for compiler-generated " +
      "inline child flows",
  );
});

test.each([
  ["billing", "quote"],
  ["{billing: requote}", "requote"],
])("tool_id: %s naming a library points to a call node", async (toolId, def) => {
  expect(await loadMessage(`  price: {kind: tool, tool_id: ${toolId}}\n`)).toBe(
    `node 'price': \`billing\` is a library; call its def with \`kind: call\`: \`call: {billing: ${def}}\``,
  );
});
