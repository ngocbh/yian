/**
 * A literal under any name is an ordinary value: a name like `password` or `api_key` loads in every
 * slot, and the literal is part of the manifest and fingerprint.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";
import { assembleText, loadConformanceFlow, loadError } from "./declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-any-name-literals-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/**
 * Assert two flows that differ only in a literal have different fingerprints and that `literalA`
 * is in the first flow's manifest.
 */
async function assertIdentity(textA: string, textB: string, literalA: string): Promise<void> {
  const flowA = (await loadFlow(textA)).compiled;
  const flowB = (await loadFlow(textB)).compiled;

  expect(flowFingerprint(flowA)).not.toBe(flowFingerprint(flowB));
  expect(JSON.stringify(flowManifest(flowA))).toContain(literalA);
}

const MULTI_NODE_DEF_ENV = (v: string) =>
  "id: f\nname: f\n" +
  "defs:\n  g:\n    env:\n" +
  `      secret: ${v}\n` +
  "    nodes:\n      run:\n        kind: code\n        output: str\n" +
  "        code: |\n          return 'x'\n" +
  "    output: ${run.output}\n" +
  "nodes:\n  b:\n    kind: call\n    call: g\n" +
  "output: ${b.output}\n";

const SINGLE_NODE_DEF_ENV = (v: string) =>
  "id: f\nname: f\n" +
  "defs:\n  g:\n    kind: code\n    env:\n" +
  `      password: ${v}\n` +
  "    output: str\n    code: |\n      return 'x'\n" +
  "nodes:\n  b:\n    kind: call\n    call: g\n" +
  "output: ${b.output}\n";

test.each([
  ["multi-node-def", MULTI_NODE_DEF_ENV],
  ["single-node-def", SINGLE_NODE_DEF_ENV],
])("a def env literal under a credential name loads (%s)", async (_id, template) => {
  await assertIdentity(template("hunter2"), template("hunter3"), "hunter2");
});

test("a def input default under a credential name loads", async () => {
  const flow = (value: string) =>
    "id: f\nname: f\n" +
    "defs:\n  g:\n    input:\n" +
    `      password: !str ${value}\n` +
    "    nodes:\n      run:\n        kind: code\n        output: str\n" +
    "        code: |\n          return 'x'\n" +
    "    output: ${run.output}\n" +
    "nodes:\n  b:\n    kind: call\n    call: g\n" +
    "output: ${b.output}\n";

  await assertIdentity(flow("hunter2"), flow("hunter3"), "hunter2");
});

test("a value entry call argument under a credential name loads", () => {
  assembleText(
    "id: f\nname: f\n" +
      "defs:\n  g:\n    input:\n      api_key: !str\n    nodes:\n      r: !str ok\n    output: ${r.output}\n" +
      "nodes:\n" +
      '  x: !str call(g, api_key="abc")\n' +
      "output: ${x.output}\n",
  );
});

test("a map over literal under a credential name loads", () => {
  const flow = (value: string) =>
    "id: p\nname: p\n" +
    "defs:\n  leaf:\n    input:\n      x: !object\n    nodes:\n" +
    "      up:\n        kind: code\n        input:\n          x: ${input.x}\n" +
    "        output: str\n        code: return 'x'\n    output: ${up.output}\n" +
    "nodes:\n  m:\n    kind: map\n" +
    `    over: [{password: ${value}}, {password: b}]\n` +
    "    call: leaf\n    input:\n      x: ${item}\n" +
    "output: ${m.output}\n";
  const load = (value: string) => assembleText(flow(value)).flow.compiled;

  const first = load("hunter2");

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(load("hunter3")));
  expect(JSON.stringify(flowManifest(first))).toContain("hunter2");
});

test("a value entry under a credential name is identity", async () => {
  const flow = (value: string) => `id: f\nname: f\nnodes:\n  api_key: !str ${value}\noutput: \${api_key.output}\n`;

  await assertIdentity(flow("hunter2"), flow("hunter3"), "hunter2");
  expect((await runFlow(await loadConformanceFlow(flow("hunter2")), {}, { env })).output).toBe("hunter2");
});

test("an inline call in a node assert shows the assert under any keyword", () => {
  const text =
    "id: f\nname: f\n" +
    "defs:\n  g:\n    input:\n      password: !Optional(str) null\n    nodes:\n" +
    "      run:\n        kind: code\n        output: str\n        code: |\n          return 'x'\n" +
    "    output: ${run.output}\n" +
    "nodes:\n  b:\n    kind: code\n    output: str\n    code: |\n      return 'x'\n" +
    "    asserts:\n      - call(g, password=x)\n" +
    "output: ${b.output}\n";

  expect(loadError(() => assembleText(text)).message).toContain("assert 'call(g, password=x)'");
});
