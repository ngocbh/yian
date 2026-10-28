/**
 * Render-only node metadata stays out of program identity and the run log.
 *
 * A node's `wiringLabels` and `bindingLabels` name synthesized params for error messages only:
 * they change neither the flow manifest nor any record of the run, and a resume rebuilds the same
 * terminal result from the authored flow.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowManifest } from "../../src/compile/manifest.ts";
import { advance, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-node-range-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const REMOVED_FIELDS = ["range", "fieldRanges", "preAssertRanges", "postAssertRanges"];

// A blocking flow whose successful records must exclude render-only node metadata.
const RANGE_FLOW = `
id: f
name: f
input:
  settle_at: !date
nodes:
  settle:
    kind: tool
    tool_id: ask
    input:
      prompt: "continue?"
    output: Optional(str)
  done:
    kind: code
    depends_on: [settle]
    output: str
    code: ./fixtures/seeds.ts:confirm_action
output: \${done.output}
`;

test.each(REMOVED_FIELDS)("manifest rejects retired node range fields: %s", async (field) => {
  // A dynamic legacy field cannot silently disappear from program identity.
  const loaded = await loadConformanceFlow(RANGE_FLOW, { env });
  Object.assign(loaded.compiled.nodes.get("done")!, { [field]: {} });
  expect(() => flowManifest(loaded.compiled)).toThrow(RangeError);
  expect(() => flowManifest(loaded.compiled)).toThrow(field);
});

test("render only node metadata does not ride run log", async () => {
  // Recovery rebuilds topology from the supplied authored flow, so synthetic labels are neither
  // execution identity nor durable state.
  const path = join(dir, "range-free.jsonl");
  const loaded = await loadConformanceFlow(RANGE_FLOW, { env });
  const node = loaded.compiled.nodes.get("done")!;
  for (const field of REMOVED_FIELDS) expect(Object.hasOwn(node, field)).toBe(false);
  const manifest = flowManifest(loaded.compiled);
  node.wiringLabels = new Map([["__r0", "cases[0].when"]]); // a synthesized param's label
  node.bindingLabels = new Map([["__on", "on: value"]]); // a binding error's label
  expect(flowManifest(loaded.compiled)).toEqual(manifest);

  const res = await runFlow(loaded, { settle_at: "2026-07-01" }, { log: path, env, tools: { ask: () => null } });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe("cancel");

  const blob = readFileSync(path, "utf8");
  for (const leaked of ["wiringLabels", "wiring_labels", "cases[0].when", "bindingLabels", "binding_labels", "on: value"]) {
    expect(blob).not.toContain(leaked);
  }

  // A terminal resume is read-only and rebuilds the same result from clean topology.
  const before = readFileSync(path);
  const res2 = await advance({ log: path, env });
  expect(res2.status).toBe("succeeded");
  expect(res2.output).toBe("cancel");
  expect(res2.events).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});
