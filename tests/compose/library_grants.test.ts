/**
 * Library grants: an AGENT's `tools:` naming a library alias grants every def (`billing`) or the
 * listed ones (`{billing: [quote]}`), each baked as a flow tool shown as `<alias>_<def>` with its
 * library def set, once however often it is granted; and load errors beside the conformance cases'.
 */

import { expect, test } from "bun:test";

import { flowManifest } from "../../src/compile/manifest.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";

const LIBRARY = `id: billing
name: billing
defs:
  quote:
    description: Price an order.
    input: {qty: !int}
    kind: code
    code: 'return inputs.qty * 3'
    output: !int
  refund:
    description: Refund an order.
    input: {qty: !int}
    kind: code
    code: 'return -inputs.qty'
    output: !int
`;

const CLEAN = "id: clean\nname: clean\ndescription: Trim.\nnodes:\n  t: !str x\noutput: ${t.output}\n";

/**
 * Load a flow whose agent `a` grants `tools`, importing `library` as `billing` and a flow as
 * `clean`; `extra` goes under the flow's top level.
 */
async function load(tools: string, options: { library?: string; extra?: string } = {}) {
  const { library = LIBRARY, extra = "" } = options;
  const lib = await loadFlow(library, { sourceLabel: "billing.yaml" });
  const clean = await loadFlow(CLEAN);
  const text =
    `id: app\nname: app\n${extra}uses:\n  billing: ./billing\n  clean: ./clean\n` +
    `nodes:\n  a: {kind: agent, prompt: hi, output: !str, tools: ${tools}}\noutput: \${a.output}\n`;
  return loadFlow(text, { childResolver: (path) => (path === "./billing" ? lib : clean) });
}

/** The agent `a` of `flow`. */
const agentOf = (flow: ComposedFlow) => flow.compiled.nodes.get("a") as AgentNode;

/** Each flow tool of `tools`' agent as `[name, library]`. */
async function granted(tools: string) {
  return agentOf(await load(tools)).flowTools.map((tool) => [tool.name, tool.library]);
}

/** The message loading `tools` fails with. */
async function loadMessage(tools: string, options: Parameters<typeof load>[1] = {}): Promise<string> {
  return (await loadErrorAsync(load(tools, options))).message;
}

test("a whole grant bakes every def as `<alias>_<def>`, and a listed grant only the listed defs", async () => {
  const whole = agentOf(await load("[billing, search]"));
  expect(whole.flowTools.map((tool) => [tool.name, tool.description, tool.library])).toEqual([
    ["billing_quote", "Price an order.", { alias: "billing", def: "quote" }],
    ["billing_refund", "Refund an order.", { alias: "billing", def: "refund" }],
  ]);
  // The alias is no host grant, and each def's nodes sit under its shown name.
  expect(whole.grantedTools()).toEqual([{ kind: "name", name: "search" }]);
  expect(whole.originChildSegments()).toEqual(["billing_quote", "billing_refund"]);
  expect(await granted("[{billing: [refund]}]")).toEqual([["billing_refund", { alias: "billing", def: "refund" }]]);
  // A bare def name never reaches into a library: it is a host grant.
  expect(agentOf(await load("[quote]")).grantedTools()).toEqual([{ kind: "name", name: "quote" }]);
});

test("a def granted whole and listed is baked once, at its first grant", async () => {
  expect((await granted("[billing, {billing: [quote]}]")).map(([name]) => name)).toEqual(["billing_quote", "billing_refund"]);
  expect((await granted("[{billing: [refund]}, billing]")).map(([name]) => name)).toEqual(["billing_refund", "billing_quote"]);
});

test("the manifest writes a flow tool's library def only when it has one", async () => {
  const agent = await load("[billing, clean]");
  const tools = ((flowManifest(agent.compiled)["nodes"] as JsonObject)["a"] as JsonObject)["flow_tools"] as JsonObject[];
  expect(tools.map((tool) => tool["library"])).toEqual([{ alias: "billing", def: "quote" }, { alias: "billing", def: "refund" }, undefined]);
});

test("a listed def the library lacks is a load error that names the closest def", async () => {
  expect(await loadMessage("[{billing: [quot]}]")).toBe("agent node 'a': library 'billing' has no def 'quot'; did you mean 'quote'?");
});

test("a granted library alias that is also a memory input is refused", async () => {
  const extra = "input:\n  billing: !FileMemory\n";
  expect(await loadMessage("[billing]", { extra })).toBe("node 'a': `billing` is both a memory input and a library; rename one");
});

test("a granted library with no defs is refused", async () => {
  expect(await loadMessage("[billing]", { library: "id: billing\nname: billing\ndefs: {}\n" })).toBe(
    "node 'a': library `billing` has no defs to grant",
  );
});

test("a def that would show under a granted flow's name is refused", async () => {
  const extra = "defs:\n  billing_quote:\n    description: Also a quote.\n    nodes:\n      v: !str x\n    output: ${v.output}\n";
  expect(await loadMessage("[billing_quote, billing]", { extra })).toBe(
    "node 'a': grants billing: tools 'billing_quote' (tool billing_quote) and 'billing_quote' (billing's quote) " +
      "have one name; grant one",
  );
});
