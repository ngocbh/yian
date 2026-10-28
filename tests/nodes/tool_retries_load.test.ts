/** TOOL `retries:` lands on the node. */

import { afterEach, expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { current, forget, reload } from "../../src/defaults.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

afterEach(() => forget());

// A flow recorded before TOOL nodes had `retries:`; its fingerprint is pinned below.
const FLOW = `id: old_tool_crash
name: old_tool_crash
nodes:
  place:
    kind: tool
    tool_id: place
    input: {symbol: "ACME"}
output: \${place.output}
`;

const OLD_FINGERPRINT = "fde8e299e6e345b350315fabc2cc6406390fcb79db37fd3cff4ed3a68d4e0f4e";

/** Return `FLOW` with `fieldLine` added to the TOOL node, as line 7. */
function withLine(fieldLine: string): string {
  return FLOW.replace("    tool_id: place\n", `    tool_id: place\n${fieldLine}\n`);
}

/** Return the retries of the built TOOL node of flow `text`. */
function retriesOf(text: string): number {
  return (assembleText(text).flow.compiled.nodes.get("place") as ToolNode).retries;
}

/** Return the fingerprint of flow `text`, loaded. */
const fingerprintOf = async (text: string) => flowFingerprint((await loadFlow(text)).compiled);

test("tool node retries defaults to the tool retries default", () => {
  expect(new ToolNode("t", { toolId: "x" }).retries).toBe(current().toolRetries);
  expect(new ToolNode("t", { toolId: "x", retries: 4 }).retries).toBe(4);
  try {
    reload((name) => (name === "YIAN_TOOL_RETRIES" ? "3" : undefined));
    expect(new ToolNode("t", { toolId: "x" }).retries).toBe(3);
    expect(new ToolNode("t", { toolId: "x", retries: 4 }).retries).toBe(4);
  } finally {
    forget();
  }
});

test("retries loads onto the node", () => {
  expect(retriesOf(withLine("    retries: 2"))).toBe(2);
});

test("omitted retries uses the default", () => {
  expect(retriesOf(FLOW)).toBe(current().toolRetries);
  reload((name) => (name === "YIAN_TOOL_RETRIES" ? "3" : undefined));
  expect(retriesOf(FLOW)).toBe(3);
});

test.each([
  ["    retries: -1", "-1"],
  ["    retries: true", "true"],
  ['    retries: "2"', '"2"'],
  ["    retries: 1.5", "1.5"],
  ["    retries: null", "null"],
])("bad retries is a load error at its line: %s", (line, got) => {
  const error = loadError(() => assembleText(withLine(line)));

  expect(error.message).toBe(`node 'place' (kind=tool): \`retries:\` must be a non-negative integer, got ${got}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(7);
});

test("zero retries keeps the flow fingerprint", async () => {
  expect(await fingerprintOf(FLOW)).toBe(OLD_FINGERPRINT);
  expect(await fingerprintOf(withLine("    retries: 0"))).toBe(OLD_FINGERPRINT);
});

test("nonzero retries changes the flow fingerprint", async () => {
  expect(await fingerprintOf(withLine("    retries: 1"))).not.toBe(OLD_FINGERPRINT);
});

test.each<CaseRow>([{ id: "load-leaves/compose_tool_retries_load__negative_retries", call: parsedNodes }])(
  "parseNodes: $id",
  runCase,
);

test.each<CaseRow>([{ id: "load-leaves/compose_tool_retries_load__retries_loads", call: assembled }])(
  "assembles: $id",
  runCase,
);
