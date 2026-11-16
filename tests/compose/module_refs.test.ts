/**
 * Tool module refs in `uses:`: classified before a scheme is read, held to the path rules a CODE
 * reference follows, and baked onto the AGENT and TOOL nodes that grant or call them. A load
 * never imports a module, from a run's record either.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { ToolNode } from "../../src/nodes/tool/node.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";

let tmp: string;

// Its top level throws, so a load that imported it would fail.
const MODULE = 'throw new Error("imported");\nexport const lookup = {};\n';

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-module-refs-")));
  for (const file of ["crm.ts", "tools.ts"]) writeFileSync(join(tmp, file), MODULE);
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const USES = { crm: "./crm.ts", quote: "./tools.ts:quote" };
const AGENT = "a: {kind: agent, prompt: hi, tools: [crm]}";
// The node kind each field is on.
const KINDS: Record<string, string> = { call: "call", tools: "agent, prompt: hi", tool_id: "tool" };

/** A flow `app` importing `uses` (alias to ref), with `nodes`, one node per line, after `head`. */
function app(uses: Record<string, string>, nodes: string[], head = ""): string {
  const refs = Object.entries(uses).map(([alias, ref]) => `  ${alias}: ${JSON.stringify(ref)}\n`);
  return `id: app\nname: app\n${head}uses:\n${refs.join("")}nodes:\n${nodes.map((node) => `  ${node}\n`).join("")}`;
}

/** Load `text` from `tmp`. */
function load(text: string, options: Parameters<typeof loadFlow>[1] = {}) {
  return loadFlow(text, { searchPaths: [tmp], ...options });
}

/** Return the message loading `text` from `tmp` fails with. */
async function refusal(text: string, options: Parameters<typeof loadFlow>[1] = {}): Promise<string> {
  return (await loadErrorAsync(load(text, options))).message;
}

test("a module ref loads without importing the module, from a run's record too", async () => {
  const flow = await load(app({ crm: "./crm.ts" }, [AGENT]));
  const again = await loadFlowSource(flow.source!);

  expect((again.compiled.nodes.get("a") as AgentNode).moduleTools.map(({ alias, path }) => [alias, path])).toEqual([
    ["crm", join(tmp, "crm.ts")],
  ]);
});

test.each([
  ["tools.ts:quote", "a tool module ref is a relative path: write `./tools.ts:quote`"],
  ["crm.mjs", "a tool module ref is a relative path: write `./crm.mjs`"],
  ["/abs/crm.js", "a tool module ref is a relative path, starting with ./ or ../"],
  ["./crm.ts@2", "a module has no version; write `./crm.ts`"],
  ["./tools.ts:quote@2", "a module has no version; write `./tools.ts:quote`"],
  ["./tools.ts:bad-name", "`bad-name` isn't an export name; write `./tools.ts:<name>`, or `./tools.ts` for every tool"],
  ["./tools.ts:default", "a default export is never a tool; export it by name"],
])("uses: %s is a module ref, refused: %s", async (ref, problem) => {
  const error = await loadErrorAsync(load(app({ crm: ref }, [AGENT])));
  expect(error.message).toBe(`uses: alias 'crm': ${problem}`);
});

test("a module alias can't take a name yian reserves", async () => {
  const text = app({ ask: "./tools.ts:ask" }, ["a: {kind: agent, prompt: hi, tools: [ask]}"]);
  expect(await refusal(text)).toBe("uses: alias 'ask': `ask` is a name yian reserves; rename the alias");
});

test("a listed module tool that would show under a built-in's name is refused at load", async () => {
  const text = app({ load: "./crm.ts" }, ["a: {kind: agent, prompt: hi, tools: [{load: [skill]}]}"]);
  expect(await refusal(text)).toBe(
    "node 'a': tool `skill` of `load` would show as `load_skill`, a name yian reserves; rename the alias or the tool",
  );
});

test("a module resolves in the importing file's folder, never on system: paths", async () => {
  mkdirSync(join(tmp, "sub"));
  writeFileSync(join(tmp, "sub", "own.ts"), MODULE);
  writeFileSync(join(tmp, "sub", "child.yaml"), app({ crm: "./own.ts" }, [AGENT]));
  const flow = await load(app({ child: "./sub/child" }, ["c: {kind: call, call: child}"]));
  const child = (flow.compiled.nodes.get("c") as CallNode).child as CompiledFlow;

  expect((child.nodes.get("a") as AgentNode).moduleTools[0]!.path).toBe(join(tmp, "sub", "own.ts"));
  expect(await refusal(app({ crm: "./own.ts" }, [AGENT], "system:\n  paths: [sub]\n"))).toBe(
    `uses: alias 'crm': tool module './own.ts' does not exist in '${tmp}'`,
  );
});

test("a module file must exist, be a file and sit inside the code root", async () => {
  // A flow saved as `nope.ts.yaml` is no longer what `./nope.ts` names.
  writeFileSync(join(tmp, "nope.ts.yaml"), "id: nope\nname: nope\nnodes:\n  a: !str hi\n");
  mkdirSync(join(tmp, "dir.ts"));
  const root = join(tmp, "root");
  mkdirSync(root);
  const refused = (ref: string, options = {}) => refusal(app({ crm: ref }, [AGENT]), options);

  expect(await refused("./nope.ts")).toBe(`uses: alias 'crm': tool module './nope.ts' does not exist in '${tmp}'`);
  expect(await refused("./dir.ts")).toBe("uses: alias 'crm': tool module './dir.ts' is not a file");
  expect(await refused("../crm.ts", { searchPaths: [root], codeRoot: root })).toBe(
    "uses: alias 'crm': tool module '../crm.ts' is outside the code root",
  );
});

test("an agent bakes each module alias it grants once, and its host grants leave them out", async () => {
  const node = "a: {kind: agent, prompt: hi, tools: [crm, quote, {crm: [lookup]}, search]}";
  const flow = await load(app({ ...USES, idle: "./crm.ts" }, [node]));
  const agent = flow.compiled.nodes.get("a") as AgentNode;

  expect(agent.moduleTools).toEqual([
    { alias: "crm", ref: "./crm.ts", path: join(tmp, "crm.ts") },
    { alias: "quote", ref: "./tools.ts:quote", path: join(tmp, "tools.ts") },
  ]);
  expect(agent.toolModules()).toEqual(agent.moduleTools);
  expect(agent.grantedTools()).toEqual([{ kind: "name", name: "search" }]);
});

test("a tool node bakes the module its tool_id names", async () => {
  const flow = await load(
    app(USES, ["g: {kind: tool, tool_id: {crm: lookup}}", "s: {kind: tool, tool_id: quote}", "h: {kind: tool, tool_id: x}"]),
  );
  const tool = (id: string) => flow.compiled.nodes.get(id) as ToolNode;

  expect(tool("g").toolId).toBe("crm_lookup");
  expect(tool("g").module).toEqual({ alias: "crm", ref: "./crm.ts", path: join(tmp, "crm.ts") });
  expect(tool("s").toolModules()).toEqual([{ alias: "quote", ref: "./tools.ts:quote", path: join(tmp, "tools.ts") }]);
  expect(tool("h").module).toBeNull();
});

test.each([
  ["call: crm", "node 'a': `crm` is a tool module; name one of its tools: `call: {crm: <tool>}`"],
  ["tools: [{quote: [quote]}]", "node 'a': `quote` is one tool, ./tools.ts:quote; grant it as `quote`"],
  ["tool_id: crm", "node 'a': `crm` is a tool module; name one of its tools: `tool_id: {crm: <tool>}`"],
  ["tool_id: {quote: quote}", "node 'a': `quote` is one tool, ./tools.ts:quote; write `tool_id: quote`"],
])("%s is a load error", async (field, message) => {
  expect(await refusal(app(USES, [`a: {kind: ${KINDS[field.split(":")[0]!]}, ${field}}`]))).toBe(message);
});

test("a map naming a module alias alone names one of its tools", async () => {
  expect(await refusal(app(USES, ["a: {kind: map, over: [1], call: crm}"]))).toBe(
    "node 'a': `crm` is a tool module; name one of its tools: `call: {crm: <tool>}`",
  );
});

test.each(["tools: [crm]", "tool_id: {crm: lookup}"])("%s naming a memory input too is a load error", async (field) => {
  const text = app(USES, [`a: {kind: ${KINDS[field.split(":")[0]!]}, ${field}}`], "input:\n  crm: !FileMemory\n");
  expect(await refusal(text)).toBe("node 'a': `crm` is both a memory input and a tool module; rename one");
});

test("a host's flow is held to the code root through its module tools", async () => {
  const root = join(tmp, "root");
  mkdirSync(root);
  const child = await load(app({ crm: "./crm.ts" }, [AGENT]));
  const options = { baseDir: root, codeRoot: root, childResolver: () => child };

  expect(await refusal(app({ lib: "lib" }, ["c: {kind: call, call: lib}"]), options)).toBe(
    "node 'a': tool module './crm.ts' is outside the code root",
  );
});

test("the fingerprint holds each module's ref, not where it resolved", async () => {
  for (const dir of ["one", "two"]) {
    mkdirSync(join(tmp, dir));
    for (const file of ["crm.ts", "tools.ts"]) writeFileSync(join(tmp, dir, file), MODULE);
  }
  const text = app({ crm: "./crm.ts" }, [AGENT, "t: {kind: tool, tool_id: {crm: lookup}}"]);
  const fingerprint = async (flow: string, dir: string) =>
    flowFingerprint((await load(flow, { searchPaths: [join(tmp, dir)] })).compiled);

  expect(await fingerprint(text, "one")).toBe(await fingerprint(text, "two"));
  expect(await fingerprint(text.replace("./crm.ts", "./tools.ts"), "one")).not.toBe(await fingerprint(text, "one"));
});
