/**
 * A port of `tests/engine/test_long_int_old_logs.py`: a run log edited to hold a 5000-digit int,
 * in three places: a `run_started` initial pool, a CODE node's `node_succeeded` value, and a
 * route's `value_json` text. Each log is written by a run here and then edited, as in Python.
 *
 * Ported: `test_resume_fails_clearly_and_leaves_the_log`, `test_replay_read_fails_clearly`,
 * `test_decode_record_fails_clearly` and `test_cli_resume_fails_clearly`, each over the three logs.
 *
 * Changed: ints are `bigint`s with no digit limit (decisions 4a, 23), so each log reads whole
 * where Python refuses it with its old-log message. `resumeFlow` returns the finished run's
 * recorded result and leaves the log unchanged; `JsonlRunLog.resume`, the reader replay and the
 * CLI use, and `decodeRecord` read the edited record with the whole int (the route keeps it as
 * its `value_json` text); `yian run --resume` exits 0, prints the recorded output and leaves the
 * log unchanged. The CODE body is JS.
 *
 * Already ported: `test_route_value_json_with_a_long_int_is_rejected` in `route_chain_costs.test.ts`
 * (`validateRoute accepts a literal holding a 5000-digit int`), and
 * `test_invalid_record_json_keeps_its_message` in `run_records.test.ts`.
 *
 * Not ported: `test_decode_record_fails_clearly_whatever_pythons_limit` (Python's
 * `sys.set_int_max_str_digits`).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import {
  decodeRecord,
  JsonlRunLog,
  NodeSucceededRecord,
  type RunRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-old-logs-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const D = "1".repeat(5000);
const SENTINEL = "123456789123";

// The sentinel reaches the log only through RunStarted's initial pool here: START's own
// NodeSucceeded carries it too, so only the first line is edited.
const STARTED_FLOW = `id: old_started
name: old_started
input:
  n: !int
nodes:
  c:
    kind: code
    output: str
    code: |
      return "done"
output: \${c.output}
`;

// The CODE node derives the sentinel from constants, so no input carries it.
const CODE_FLOW = `id: old_code
name: old_code
input:
  t: !str
nodes:
  c:
    kind: code
    input:
      t: \${input.t}
    output: int
    code: |
      return 123456789000 + 123
output: \${c.output}
`;

// A learnable run records `input.n + 0` as a scalar route whose result is JSON text.
const ROUTE_FLOW = `id: old_route
name: old_route
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: \${(input.n + 0) % 7}
    output: int
    code: |
      return inputs["x"]
output: \${c.output}
`;

/** An edited log: its flow, path and recorded output, and a reader of the edited value from its record (else `undefined`). */
interface OldLog {
  flow: string;
  path: string;
  output: unknown;
  read: (record: RunRecord) => unknown;
}

let logs = 0;

/** Run `flow` with `inputs` into a fresh log and return the log's path and text. */
async function written(flow: string, inputs: Record<string, unknown>, learnable = false): Promise<[string, string]> {
  const path = join(dir, `run-${logs++}.jsonl`);
  await runFlow(await loadFlow(flow, { env }), inputs, { env, log: path, learnable });
  return [path, readFileSync(path, "utf8")];
}

/** Return an edited log whose RunStarted record holds the long int. */
async function oldStartedLog(): Promise<OldLog> {
  const [path, text] = await written(STARTED_FLOW, { n: BigInt(SENTINEL) });
  const lines = text.split("\n");
  expect(lines[0]).toContain('"kind":"run_started"');
  expect(lines[0]).toContain(`"n":${SENTINEL}`);
  lines[0] = lines[0]!.replace(SENTINEL, D);
  writeFileSync(path, lines.join("\n"));
  const read = (record: RunRecord) =>
    record instanceof RunStartedRecord
      ? (record.initialPool.store["__start__"]!.value as Record<string, unknown>)["n"]
      : undefined;
  return { flow: STARTED_FLOW, path, output: "done", read };
}

/** Return an edited log whose CODE node NodeSucceeded holds the long int. */
async function oldCodeLog(): Promise<OldLog> {
  const [path, text] = await written(CODE_FLOW, { t: "a" });
  const recorded = `"node_id":"c","value":{"kind":"int","value":${SENTINEL}}`;
  expect(text.split(recorded)).toHaveLength(2);
  writeFileSync(path, text.replace(recorded, recorded.replace(SENTINEL, D)));
  const read = (record: RunRecord) =>
    record instanceof NodeSucceededRecord && record.nodeId === "c" ? record.value.value : undefined;
  return { flow: CODE_FLOW, path, output: BigInt(SENTINEL), read };
}

/** Return an edited log whose route value_json string holds the long int. */
async function oldRouteLog(): Promise<OldLog> {
  const [path, text] = await written(ROUTE_FLOW, { n: BigInt(SENTINEL) }, true);
  // The sentinel sits inside a JSON string here, so the outer record parse never sees it.
  const recorded = `"value_json":"${SENTINEL}"`;
  expect(text.split(recorded)).toHaveLength(2);
  writeFileSync(path, text.replace(recorded, recorded.replace(SENTINEL, D)));
  // The node's route is `{x: (input.n + 0) % 7}`; the edited text is `input.n + 0`'s.
  const read = (record: RunRecord) =>
    record instanceof NodeSucceededRecord && record.nodeId === "c"
      ? record.forwardCapture!.route.children[0]!.children[0]!.valueJson
      : undefined;
  return { flow: ROUTE_FLOW, path, output: BigInt(SENTINEL) % 7n, read };
}

const OLD_LOGS = [
  ["run_started", oldStartedLog, BigInt(D)],
  ["code_node_succeeded", oldCodeLog, BigInt(D)],
  ["route_value_json", oldRouteLog, D],
] as const;

test.each(OLD_LOGS)("resume fails clearly and leaves the log (%s)", async (_id, makeLog) => {
  const { flow, path, output } = await makeLog();
  const before = readFileSync(path);

  const result = await resumeFlow(await loadFlow(flow, { env }), { log: path, env });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(output as never);
  expect(readFileSync(path)).toEqual(before);
});

test.each(OLD_LOGS)("replay read fails clearly (%s)", async (_id, makeLog, long) => {
  const { path, read } = await makeLog();
  const before = readFileSync(path);

  const log = JsonlRunLog.resume(path);
  const records = log.records();
  log.close();

  expect(records.filter((envelope) => read(envelope.record) === long)).toHaveLength(1);
  expect(readFileSync(path)).toEqual(before);
});

test.each(OLD_LOGS)("decode record fails clearly (%s)", async (_id, makeLog, long) => {
  const { path, read } = await makeLog();
  const line = readFileSync(path, "utf8").split("\n").find((line) => line.includes(D))!;

  expect(read(decodeRecord(line).record)).toBe(long);
});

test.each(OLD_LOGS)("cli resume fails clearly (%s)", async (_id, makeLog) => {
  const { flow, path, output } = await makeLog();
  const flowPath = join(dir, `flow-${logs}.yaml`);
  writeFileSync(flowPath, flow);
  const before = readFileSync(path);
  const out: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async () => {} },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? dir : undefined),
  };

  const code = await main(["run", flowPath, "--resume", path, "--quiet"], io);

  expect(code).toBe(0);
  expect(out.join("")).toBe(`${String(output)}\n`);
  expect(readFileSync(path)).toEqual(before);
});
