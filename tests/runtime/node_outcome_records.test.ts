/**
 * A node's outcome records (`node_succeeded`, `node_routed`, `node_expanded`, `node_failed`) hold
 * no forward capture: the run log keeps no retained trace.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { decodeRecord, encodeRecord, NodeSucceededRecord, RecordEnvelope, SCHEMA_VERSION } from "../../src/runtime/log.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));

/** A CODE node, a CASE that routes, and a MAP that grows, so every outcome kind is written. */
const FLOW = `
id: outcomes
name: outcomes
input:
  topic: !str
defs:
  echo:
    input:
      x: !int
    nodes: {}
    output: \${input.x}
nodes:
  plan:
    kind: code
    code: ./compose_codefns.ts:make_plan
    input:
      topic: \${input.topic}
    output:
      rating: str
      score: float
  gate:
    kind: case
    cases:
      - when: plan.output.score > 0
        then: each
    else: zero
  zero:
    kind: code
    output: int
    code: |
      return 0
  each:
    kind: map
    call: echo
    over: \${[1, 2]}
    input:
      x: \${item}
output: \${each.output | zero.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-node-outcome-records-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** Run `FLOW` to a durable log in the test's folder. */
async function runSimple() {
  const log = join(dir, "run.jsonl");
  const flow = await loadFlow(FLOW, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
  const result = await runFlow(flow, { topic: "owls" }, { log, env: hostEnv(dir) });
  return { result, log };
}

/** A `node_succeeded` line whose payload also holds `extra`. */
function succeededWith(extra: Record<string, unknown>): string {
  const line = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r", 3, new NodeSucceededRecord("a", 2)));
  const wire = JSON.parse(line);
  wire.payload = { ...wire.payload, ...extra };
  return JSON.stringify(wire);
}

test("node outcome records carry no forward_capture", async () => {
  const { result, log } = await runSimple();
  expect(result.status, result.failure?.message).toBe("succeeded");
  const kinds = new Set<string>();
  for (const line of readFileSync(log, "utf8").trim().split("\n")) {
    const record = JSON.parse(line);
    kinds.add(record.kind);
    expect(record.payload).not.toHaveProperty("forward_capture");
  }
  for (const kind of ["node_succeeded", "node_routed", "node_expanded"]) expect(kinds).toContain(kind);
});

test("a node_succeeded line holding forward_capture is refused", () => {
  expect(() => decodeRecord(succeededWith({ forward_capture: null }))).toThrow(
    'invalid node_succeeded payload fields; missing=[], extra=["forward_capture"]',
  );
});
