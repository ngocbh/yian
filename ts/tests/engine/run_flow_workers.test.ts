/**
 * `runFlow`'s `numWorkers` sets the engine's drive mode: a port of
 * `tests/engine/test_run_flow_workers.py`. A fan-out/fan-in CODE flow runs serial (`0`) and pooled
 * (`>= 1`) to the same output, since workers only execute nodes.
 *
 * Ported: `test_run_flow_serial_and_pooled_same_output`.
 *
 * Changed: `test_run_flow_defaults_to_eight_workers` runs `runFlow` without and with a `signal`,
 * where Python runs `run_flow` and `_run_flow_controlled` (its host stop event); the merge node is
 * an inline body, as the fixtures have no `join_two`.
 *
 * Later (T4-10): `test_pooled_dispatcher_drops_read_results` (a LOOP).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";
import { JsonlRunLog, RunStartedRecord } from "../../src/runtime/log.ts";

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));

// topic -> {positive, cautious} in parallel -> a CODE node fans them in.
const FANOUT = `
id: angles
name: angles
input:
  topic: !str
nodes:
  pro:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./compose_codefns.ts:positive
  con:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./compose_codefns.ts:cautious
  merge:
    kind: code
    input:
      pro: \${pro.output}
      con: \${con.output}
    output: str
    code: |
      return inputs.pro + " | " + inputs.con
output: \${merge.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-workers-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** `runFlow` with only the test's credential store in its environment, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  const env = new Map([["YIAN_AUTH_FILE", join(dir, "auth.json")]]);
  return runFlow(flow, inputs, { env: (name) => env.get(name), ...options });
}

test("run flow serial and pooled same output", async () => {
  const loaded = await loadFlow(FANOUT, { baseDir: FIXTURES });
  const serial = await run(loaded, { topic: "X" }, { numWorkers: 0 });
  const pooled = await run(loaded, { topic: "X" }, { numWorkers: 4 });
  expect(serial.status).toBe("succeeded");
  expect(pooled.status).toBe("succeeded");
  expect(serial.output).toBe(pooled.output);
});

for (const withSignal of [false, true]) {
  test(`run flow defaults to eight workers (${withSignal ? "with" : "without"} a signal)`, async () => {
    const loaded = await loadFlow(FANOUT, { baseDir: FIXTURES });
    const path = join(dir, "run.jsonl");
    const signal = withSignal ? new AbortController().signal : null;
    const result = await run(loaded, { topic: "X" }, { log: path, signal });
    expect(result.status).toBe("succeeded");
    using log = JsonlRunLog.resume(path);
    const first = log.records()[0]!.record;
    expect(first).toBeInstanceOf(RunStartedRecord);
    expect((first as RunStartedRecord).numWorkers).toBe(8);
  });
}
