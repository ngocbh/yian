/**
 * Library files: a file with `defs:` and no `nodes:`, loaded as its defs by name, the top-level
 * keys it refuses, and its import through `uses:`, rebased under the alias and held to a code root.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { iterFlowOrigins } from "../../src/compile/origins.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { CallableScope, loadFlow } from "../../src/compose/loader.ts";
import { parseFile } from "../../src/compose/parse/file.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs } from "../../src/typesys/types.ts";

import { loadError, loadErrorAsync } from "../helpers/declarations.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-library-files-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

// `requote` calls its sibling `quote`, whose output is the library's typedef `Quote`.
const LIBRARY = `id: billing
name: billing
version: "2"
typedefs:
  Quote: {total: float, currency: str}
defs:
  quote:
    description: Price an order.
    input: {qty: !int}
    kind: code
    code: 'return {total: inputs.qty * 2.5, currency: "EUR"}'
    output: !Quote
  requote:
    description: Price an order again.
    input: {qty: !int}
    nodes:
      again: {kind: call, call: quote, input: {qty: "\${input.qty}"}}
    output: \${again.output}
`;

/** A flow `app` that imports `ref` as `billing` and returns its input. */
const importer = (ref: string, node = "echo: !str ${input.x}") =>
  `id: app\nname: app\ninput:\n  x: !str\nuses:\n  billing: ${ref}\nnodes:\n  ${node}\noutput: \${echo.output}\n`;

test("a file with defs: and no nodes: is a library", () => {
  const library = parseFile(LIBRARY);
  expect(library.library).toBe(true);
  expect(library.nodes).toEqual({});
  // `nodes:` makes a flow, even an empty one beside `defs:`.
  expect(parseFile(LIBRARY.replace("\ndefs:", "\nnodes: {}\ndefs:")).library).toBe(false);
});

test.each(["input", "output", "asserts", "env", "llm_config"])("a library refuses a top-level %s:", (key) => {
  const value = key === "asserts" ? '["true"]' : key === "output" ? "${x}" : "{}";
  const error = loadError(() => parseFile(`${LIBRARY}${key}: ${value}\n`));
  expect(error.message).toBe(
    `a library (defs: and no nodes:) can't set \`${key}:\`, which its defs don't inherit; set it on each def`,
  );
  const line = LIBRARY.split("\n").length;
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(line, 1, line, key.length + 1));
});

test("a file with neither nodes: nor defs: says a flow needs one", () => {
  expect(loadError(() => parseFile("id: f\nname: f\n")).message).toBe(
    "a flow needs nodes:, or defs: to be a library",
  );
});

test("loadFlow gives a library's defs by name, in written order, and the empty flow", async () => {
  const flow = await loadFlow(LIBRARY, { sourceLabel: "billing.yaml" });
  const quote = flow.library!.get("quote")!;

  expect([...flow.library!.keys()]).toEqual(["quote", "requote"]);
  expect(quote).toBeInstanceOf(ComposedFlow);
  expect(quote.description).toBe("Price an order.");
  expect(quote.input.map((param) => param.name)).toEqual(["qty"]);
  expect([...flow.compiled.nodes.keys()]).toEqual(["__start__", "__end__"]);
  expect([flow.compiled.flowId, flow.version]).toEqual(["billing", "2"]);
  for (const def of flow.library!.values()) {
    for (const origin of iterFlowOrigins(def.compiled)) {
      expect(def.originResolver!.resolve(origin)!.label).toBe("billing.yaml");
      expect(flow.originResolver!.resolve(origin)!.label).toBe("billing.yaml");
    }
  }
  expect((await loadFlow(importer("x"), { childResolver: () => flow })).library).toBeNull();
});

test("a def sees the library's typedefs, its uses: and its sibling defs", async () => {
  const fx = "id: fx\nname: fx\ninput:\n  n: !int\nnodes:\n  r: !int ${input.n}\noutput: ${r.output}\n";
  writeFileSync(join(tmp, "fx.yaml"), fx);
  const text = LIBRARY.replace("\ndefs:", "\nuses:\n  fx: ./fx\ndefs:").replace(
    "  requote:",
    "  convert:\n    input: {n: !int}\n    kind: call\n    call: fx\n  requote:",
  );
  const flow = await loadFlow(text, { searchPaths: [tmp] });

  const result = await runFlow(flow.library!.get("requote")!, { qty: 3 });
  expect(result.output).toEqual({ total: 7.5, currency: "EUR" });
  expect((await runFlow(flow.library!.get("convert")!, { n: 7 })).output).toBe(7n);
});

test("a def that doesn't load fails the library's load", async () => {
  const error = await loadErrorAsync(loadFlow(`${LIBRARY}  broken:\n    kind: nope\n`));
  expect(error.message).toStartWith("defs entry 'broken': node 'broken': unknown kind 'nope'");
});

test("each def is held to the node limit, as a flow is", async () => {
  const env = (name: string) => (name === "YIAN_MAX_FLOW_NODES" ? "3" : undefined);
  // `quote` is 3 nodes, START and END included; `requote` holds a copy of it beside its own 3.
  const error = await loadErrorAsync(loadFlow(LIBRARY, { env }));
  expect(error.message).toBe(
    "node 'again': this flow expands to more than 3 nodes, counting each call of a def or uses: flow as a " +
      "copy; the limit for a flow is 3 (YIAN_MAX_FLOW_NODES)",
  );
  const line = LIBRARY.split("\n").findIndex((text) => text.startsWith("      again:")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(line);
});

test("an imported library's defs are rebased under the alias, and its pin applies", async () => {
  writeFileSync(join(tmp, "billing.yaml"), LIBRARY);
  const canonical = await loadFlow(LIBRARY, { sourceLabel: "billing.yaml" });
  const scope = new CallableScope({}, readTypedefs(null), () => canonical, { billing: "./billing@2" });
  const imported = await scope.resolveAlias("billing");

  expect([...imported.library!.keys()]).toEqual(["quote", "requote"]);
  for (const [name, def] of imported.library!) {
    const own = new Set([...iterFlowOrigins(canonical.library!.get(name)!.compiled)].map((origin) => origin.key));
    const origins = [...iterFlowOrigins(def.compiled)];
    expect(origins.length).toBe(own.size);
    for (const origin of origins) {
      expect(own.has(origin.key)).toBe(false);
      expect(imported.originResolver!.resolve(origin)!.label).toBe("billing.yaml");
      expect(def.originResolver!.resolve(origin)!.label).toBe("billing.yaml");
    }
  }

  expect((await loadFlow(importer("./billing@2"), { searchPaths: [tmp] })).compiled.flowId).toBe("app");
  const pinned = await loadErrorAsync(loadFlow(importer("./billing@3"), { searchPaths: [tmp] }));
  expect(pinned.message).toContain('pinned ./billing@3 but the resolved file declares version: "2"');
});

test("a host's library is held to the code root through its defs", async () => {
  const root = join(tmp, "root");
  const out = join(tmp, "out");
  for (const dir of [root, out]) {
    mkdirSync(dir);
    writeFileSync(join(dir, "h.ts"), "export function fn(inputs: { qty: unknown }) {\n  return inputs.qty;\n}\n");
  }
  const text = LIBRARY.replace(/code: 'return .*'/, "code: ./h.ts:fn").replace("output: !Quote", "output: !int");
  const outside = await loadFlow(text, { baseDir: out });
  const inside = await loadFlow(text, { baseDir: root });
  const options = { baseDir: root, codeRoot: root };

  const error = await loadErrorAsync(loadFlow(importer("lib"), { ...options, childResolver: () => outside }));
  expect(error.message).toBe("node 'quote': `code:` reference './h.ts:fn' is outside the code root");
  expect((await loadFlow(importer("lib"), { ...options, childResolver: () => inside })).compiled.flowId).toBe("app");
});

test("a library's defs must be flows", async () => {
  const empty = (await loadFlow("id: e\nname: e\nnodes: {}\n")).compiled;
  const bad = new ComposedFlow(empty, { library: new Map([["quote", "not a flow" as never]]) });
  const error = await loadErrorAsync(loadFlow(importer("lib"), { childResolver: () => bad }));
  expect(error.message).toContain("resolver returned a library whose defs aren't each a ComposedFlow");
});

test.each([
  ["a call", "echo: {kind: call, call: billing, input: {}}", "call node 'echo'"],
  ["a grant", "echo: {kind: agent, prompt: hi, tools: [billing]}", "agent node 'echo'"],
])("%s of a library alias is a load error", async (_, node, subject) => {
  const library = await loadFlow(LIBRARY);
  const error = await loadErrorAsync(loadFlow(importer("lib", node), { childResolver: () => library }));
  expect(error.message).toBe(`${subject}: callable 'billing' is a library (defs: and no nodes:), not a flow`);
});
