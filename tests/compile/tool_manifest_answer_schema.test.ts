/** A TOOL ask's answer schema is part of its node manifest and the flow's identity. */
import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

/** Return an ask flow whose node `q` declares `output` and waits `timeout`. */
const askFlow = (output: string, timeout = "24h") => `id: f
name: f
nodes:
  q:
    kind: tool
    tool_id: ask
    input:
      prompt: Ship it?
      timeout: ${timeout}
    output: ${output}
output: \${q.output}
`;

// A tool flow from before built-ins; its fingerprint is pinned in `tests/nodes/tool_retries_load.test.ts`.
const PLAIN = `id: old_tool_crash
name: old_tool_crash
nodes:
  place:
    kind: tool
    tool_id: place
    input: {symbol: "ACME"}
output: \${place.output}
`;
const PLAIN_FINGERPRINT = "fde8e299e6e345b350315fabc2cc6406390fcb79db37fd3cff4ed3a68d4e0f4e";

/** Return node `id`'s manifest in flow `text`, loaded. */
async function nodeManifest(text: string, id: string): Promise<JsonObject> {
  const flow = (await loadFlow(text)).compiled;
  return (flowManifest(flow).nodes as JsonObject)[id] as JsonObject;
}

const fingerprintOf = async (text: string) => flowFingerprint((await loadFlow(text)).compiled);

test("an ask's manifest holds its answer schema; a plain tool's holds none", async () => {
  expect(Object.keys(await nodeManifest(askFlow("bool"), "q"))).toContain("answer_schema");
  expect(Object.keys(await nodeManifest(PLAIN, "place"))).not.toContain("answer_schema");
});

test("asks that differ only in output: have different fingerprints", async () => {
  expect(await fingerprintOf(askFlow("bool"))).not.toBe(await fingerprintOf(askFlow("str")));
});

test("a plain tool flow keeps its fingerprint", async () => {
  expect(await fingerprintOf(PLAIN)).toBe(PLAIN_FINGERPRINT);
});

test("duration spellings share a fingerprint", async () => {
  expect(await fingerprintOf(askFlow("bool", "24h"))).toBe(await fingerprintOf(askFlow("bool", "86400s")));
});

test("Optional(T) and T give one answer schema", async () => {
  const optional = await nodeManifest(askFlow("Optional(str)"), "q");
  const plain = await nodeManifest(askFlow("str"), "q");
  expect(optional["answer_schema"]).toEqual(plain["answer_schema"]);
});
