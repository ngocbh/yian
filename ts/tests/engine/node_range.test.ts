/**
 * Render-only node metadata stays out of program identity and the run log: a port of
 * `tests/engine/test_node_range.py`.
 *
 * A node's `wiringLabels` and `bindingLabels` name synthesized params for error messages only:
 * they change neither the flow manifest nor any record of the run, and a resume rebuilds the same
 * terminal result from the authored flow.
 *
 * Ported: `test_manifest_rejects_retired_node_range_fields`, each of its four rows, and
 * `test_render_only_node_metadata_does_not_ride_run_log`. The CODE reference is TS's
 * (`./fixtures/seeds.ts:confirm_action`). The log is searched for both the TS property names
 * (`wiringLabels`) and Python's (`wiring_labels`). `flow_identity.test.ts`'s
 * `test_manifest_rejects_unknown_execution_fields_instead_of_omitting_them` has similar rows on
 * another flow.
 *
 * Changed: the retired node properties are camelCase (`fieldRanges`, not `field_ranges`), the TS
 * API spelling (decision 12), as in `flow_identity.test.ts`, and the manifest refuses them with a
 * `RangeError` where Python raises `ValueError`.
 *
 * Not ported: `test_node_has_no_compose_range_metadata`, which guards against Python-era range
 * fields TS nodes never had (listed in `start_node.test.ts` too).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowManifest } from "../../src/compile/manifest.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
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
    kind: human_input
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

  const res = await runFlow(loaded, { settle_at: "2026-07-01" }, { log: path, env, tools: { ask_user: () => null } });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe("cancel");

  const blob = readFileSync(path, "utf8");
  for (const leaked of ["wiringLabels", "wiring_labels", "cases[0].when", "bindingLabels", "binding_labels", "on: value"]) {
    expect(blob).not.toContain(leaked);
  }

  // A terminal resume is read-only and rebuilds the same result from clean topology.
  const before = readFileSync(path);
  const res2 = await resumeFlow({ log: path, env });
  expect(res2.status).toBe("succeeded");
  expect(res2.output).toBe("cancel");
  expect(res2.events).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});
