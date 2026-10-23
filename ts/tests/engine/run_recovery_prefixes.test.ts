/**
 * Silent replay of a historical log prefix through the live reducer: a port of
 * `tests/engine/test_run_recovery_prefixes.py`. Python's `ValueError` is a `RangeError` here.
 *
 * Ported, through `run.ts`'s `replayRunLog` (exported `@internal`) and `resumeFlow`, with flows
 * loaded from YAML whose CODE references resolve from the conformance fixtures, each pinning its
 * one class and full message: `test_replay_preserves_ordered_unfinished_starts_for_later_resume`,
 * `test_replay_rejects_invalid_prefix_roots` (3 rows),
 * `test_replay_rejects_log_identity_and_length_disagreement` and
 * `test_replay_rejects_records_after_terminal_without_touching_source`.
 *
 * Changed:
 * - `test_replay_requires_pristine_state`: the log is a live run of the fixtures' `flow()`.
 * - `test_historical_cursor_matches_each_envelopes_schema`: the cursor holds live TS log lines,
 *   not Python's recorded schema-9 bytes, since schema 12 is the only TS schema. It also pins the
 *   append past the prefix and the consumed envelopes `records()` and the iterator return.
 * - `test_replay_every_prefix_matches_live_state_and_uses_commit_path`: ESM bindings can't be
 *   patched, so the count of `commit_prepared` calls is not checked; the snapshots, the read-only
 *   source and the unchanged authored flow are.
 * - `test_replay_applies_logged_llm_map_before_rebuilding_growth`: replay takes no environment, so
 *   the "never reads a default" half (Python patches `connection_snapshot`) has nothing to patch.
 * - `test_recorded_public_resume_keeps_completed_code_and_wire[6]`: the prefix is the first 6
 *   lines of a live TS run's log (schema 12 is the only TS schema), and an accepted CODE rerun is
 *   caught by spying on `CodeNode.prototype.forward`; a resume in an environment whose
 *   `YIAN_DEFAULT_PROVIDER` and credential-store reads fail stands for the patched
 *   `connection_snapshot`.
 * - `test_replay_rejects_changed_but_valid_growth_digest`: the growth builder can't be patched, so
 *   the recorded `NodeExpanded` holds another valid digest instead of the builder making another
 *   fragment; the message is checked for "topology digest mismatch", as Python matches it.
 * - `test_fingerprint_mismatch_fails_before_copy_or_llm_apply`: `copyGraph` and
 *   `applyEffectiveLlmMap` can't be patched; the row checks the error and that the flow didn't
 *   change.
 * - `test_run_started_replay_rejects_pool_and_envelope_identity_drift`: records are frozen, so the
 *   drifted `RunStarted` is built anew with the other run id in its pool; the message is checked
 *   for `system.run_id`, as Python matches it.
 *
 * TS-only, the reducer cores of the rows above, through `replayRecords`: every prefix of a live
 * log replays to the live state after it; a replayed RunStarted keeps the recorded defaults and
 * call-time LLM config; an empty prefix, a first record that isn't RunStarted, a first sequence
 * that isn't 0, a sequence gap, a run-id change and a record after the terminal are refused; a
 * route to a handle that matches no edge throws the `RangeError` itself, not the live
 * `NodeDataError`.
 *
 * TS-only, through `runFlow`, `inspectRun` and `resumeFlow`: a two-worker run whose log refuses
 * appends after both `NodeStarted` records counts each node once, and its resume publishes a
 * `NodeAttemptResumed` for each, in start order.
 *
 * Not ported: none.
 *
 * Covered by a case: `test_recorded_public_resume_keeps_completed_code_and_wire[5]`
 * (`resume/run_recovery_prefixes__a_settled_node_does_not_run_again`).
 */

import { expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildGrow } from "../../src/compile/expand.ts";
import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { copyGraph, END_ID, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, type Env } from "../../src/defaults.ts";
import { NodeAttemptResumed, type RunEvent, RunResumed } from "../../src/events.ts";
import { legacyConnectionConfig } from "../../src/llm_clients/connection.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Grow } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { inspectRun, replayRunLog, type RunResult, resumeFlow, runFlow } from "../../src/run.ts";
import {
  decodeRecord,
  encodeRecord,
  JsonlRunLog,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  RecordEnvelope,
  type RunLog,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  HistoricalLogCursor,
  prepareExpanded,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRunStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { hostEnv } from "./mcp_helpers.ts";
import {
  commitStartSuccess,
  crashAfter,
  flow,
  fragmentBuilder,
  progressOf,
  reducerSnapshot,
  routeFlow,
  routeInFlight,
  RUN_ID,
  startData,
  startedRun,
  thrownBy,
} from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` whose message is `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  const caught = thrownBy(fn);
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/**
 * Return the log of a terminal live run of `flow()` and its reducer snapshot after each prefix,
 * from its RunStarted on.
 */
function linearHistory(): { log: MemoryRunLog; snapshots: Record<string, unknown>[] } {
  const state = new RunState(flow());
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(log, state, prepareRunStarted(state, startData(state.flow)));
  const snapshots = [reducerSnapshot(state)];
  const transitions = [
    () => commitPrepared(log, state, prepareNodeStarted(state, START_ID)),
    () => commitStartSuccess(state, log),
    () => commitPrepared(log, state, prepareNodeStarted(state, "work")),
    () => commitPrepared(log, state, prepareNodeSucceeded(state, "work", "Ada")),
    () => commitPrepared(log, state, prepareNodeStarted(state, END_ID)),
    () => commitPrepared(log, state, prepareNodeSucceeded(state, END_ID, "Ada")),
  ];
  for (const transition of transitions) {
    transition();
    snapshots.push(reducerSnapshot(state));
  }
  expect(state.result).not.toBeNull();
  return { log, snapshots };
}

/** Replay `envelopes` over a fresh state of `flow()`. */
function replay(envelopes: readonly RecordEnvelope[]): RunState {
  return replayRecords(new RunState(flow()), envelopes, { growBuilder: fragmentBuilder });
}

test("every prefix of a live log replays to the live state after that prefix", () => {
  const { log, snapshots } = linearHistory();
  const envelopes = log.records();
  expect(snapshots).toHaveLength(envelopes.length);

  snapshots.forEach((expected, index) => {
    const length = index + 1;
    const recovered = replay(envelopes.slice(0, length));
    expect(reducerSnapshot(recovered)).toEqual(expected);
    expect(recovered.nextSequence).toBe(length);
  });
});

test("a replayed RunStarted keeps the recorded defaults and call-time LLM config", () => {
  const live = new RunState(flow());
  const log = new MemoryRunLog(RUN_ID);
  const data = {
    ...startData(live.flow),
    defaults: { MAX_TOTAL_NODES: 5000n },
    callTimeLlm: { selection: {}, agents: { chat: {} } },
  };
  commitPrepared(log, live, prepareRunStarted(live, data));
  expect(live.runDefaults).not.toEqual(BUILTIN);
  expect(live.callTimeLlm).not.toBeNull();

  const recovered = replay(log.records());

  expect(recovered.runDefaults).toEqual(live.runDefaults);
  expect(recovered.callTimeLlm).toEqual(live.callTimeLlm);
  expect(recovered.nextSequence).toBe(live.nextSequence);
});

test("the historical cursor matches each envelope's text", () => {
  const { log } = linearHistory();
  const wires = log.records().map(encodeRecord);
  const cursor = new HistoricalLogCursor(wires);
  expect(cursor.runId).toBe(RUN_ID);
  expect(cursor.path).toBeNull();

  for (const wire of wires) {
    const expected = decodeRecord(wire);
    const before = cursor.nextSequence;
    expectError(
      () => cursor.append(new NodeStartedRecord("wrong")),
      RangeError,
      "prepared transition does not match the historical run record",
    );
    expect(cursor.nextSequence).toBe(before);
    const accepted = cursor.append(expected.record);
    expect(accepted.schema).toBe(SCHEMA_VERSION);
    expect(encodeRecord(accepted)).toBe(wire);
  }
  expect(cursor.nextSequence).toBe(wires.length);
  expectError(
    () => cursor.append(new NodeStartedRecord("wrong")),
    RangeError,
    "replay attempted to append beyond the historical prefix",
  );
  expect(cursor.records().map(encodeRecord)).toEqual(wires);
  expect([...cursor].map(encodeRecord)).toEqual(wires);
});

test("replay refuses a malformed prefix root", () => {
  const { log } = linearHistory();
  const envelopes = log.records();
  const started = new NodeStartedRecord(START_ID);

  expectError(() => replay([]), RangeError, "cannot replay an empty run-record prefix");
  expectError(
    () => replay([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, started)]),
    RangeError,
    "the first historical run record must be RunStarted",
  );
  expectError(
    () => replay([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 1, started)]),
    RangeError,
    "replay sequence mismatch: expected 0, found 1",
  );
  expectError(
    () => replay([...envelopes.slice(0, 2), ...envelopes.slice(3)]),
    RangeError,
    "replay sequence mismatch: expected 2, found 3",
  );
  const last = envelopes.at(-1)!;
  expectError(
    () => replay([...envelopes.slice(0, -1), new RecordEnvelope(last.schema, "other-run", last.seq, last.record)]),
    RangeError,
    "replay run_id mismatch: expected 'run-123', found 'other-run'",
  );
});

test("replay requires a pristine state", () => {
  const { log } = linearHistory();
  const state = new RunState(flow());
  state.ready.push(START_ID);

  expectError(
    () => replayRecords(state, log.records(), { growBuilder: fragmentBuilder }),
    RangeError,
    "RunStarted requires a pristine RunState at sequence 0",
  );
});

test("replay refuses a record after the terminal", () => {
  const { log } = linearHistory();
  const trailing = MemoryRunLog.fromRecords(log.records());
  trailing.append(new NodeStartedRecord(START_ID));
  const envelopes = trailing.records();

  expectError(() => replay(envelopes), RangeError, "terminal run state cannot prepare another graph transition");
});

test("a replayed route to a handle that matches no edge throws the RangeError itself", () => {
  const { log } = routeInFlight();
  const envelopes = [
    ...log.records(),
    new RecordEnvelope(SCHEMA_VERSION, RUN_ID, log.nextSequence, new NodeRoutedRecord("router", "missing")),
  ];

  expectError(
    () => replayRecords(new RunState(routeFlow()), envelopes, { growBuilder: fragmentBuilder }),
    RangeError,
    "NodeRouted handle 'missing' matches no outgoing edge of 'router'",
  );
});

// ---- Through run.ts's log replay and resumeFlow ----

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));
const RECOVERY_RUN_ID = "recovery-run";

const LINEAR = `
id: recovery-linear
name: recovery_linear
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      x: \${input.x}
    output: str
    code: ./compose_codefns.ts:echo_x
output: \${n.output}
`;

const FANOUT = `
id: recovery-fanout
name: recovery_fanout
nodes:
  first:
    kind: code
    input: {topic: first}
    output: str
    code: ./compose_codefns.ts:cautious
  second:
    kind: code
    input: {topic: second}
    output: str
    code: ./compose_codefns.ts:cautious
output:
  first: \${first.output}
  second: \${second.output}
`;

const CALL_AGENT = `
id: recovery-call-agent
name: recovery_call_agent
defs:
  child:
    nodes:
      leaf:
        kind: agent
        prompt: replay me
    output: \${leaf.output}
nodes:
  gate:
    kind: call
    call: child
output: \${gate.output}
`;

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** A stable borrowed prefix whose append and close must stay unused. */
class ReadOnlyLog implements RunLog {
  readonly path = null;
  readonly runId: string;
  readonly nextSequence: number;
  appendCalls = 0;
  closeCalls = 0;
  readonly #envelopes: readonly RecordEnvelope[];

  constructor(envelopes: readonly RecordEnvelope[], options: { runId?: string; nextSequence?: number } = {}) {
    this.#envelopes = envelopes;
    this.runId = options.runId ?? envelopes[0]?.runId ?? RECOVERY_RUN_ID;
    this.nextSequence = options.nextSequence ?? envelopes.length;
  }

  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return this.#envelopes[Symbol.iterator]();
  }

  records(): readonly RecordEnvelope[] {
    return Object.freeze([...this.#envelopes]);
  }

  append(): RecordEnvelope {
    this.appendCalls += 1;
    throw new Error("historical source log must remain read-only");
  }

  close(): void {
    this.closeCalls += 1;
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

/** Every replay-relevant coordinate of `state`, the live topology included. */
function recoverySnapshot(state: RunState): Record<string, unknown> {
  return {
    ...reducerSnapshot(state),
    flow: flowManifest(state.flow),
    initialStaticIds: [...state.initialStaticIds],
  };
}

/** A terminal log of `LINEAR` and the live state's snapshot after every prefix. */
async function loadedLinearHistory(): Promise<{
  loaded: ComposedFlow;
  log: MemoryRunLog;
  snapshots: Record<string, unknown>[];
}> {
  const loaded = await load(LINEAR);
  const log = new MemoryRunLog(RECOVERY_RUN_ID);
  const state = startedRun(loaded, log, { inputs: { x: "Ada" }, numWorkers: 2 });
  const snapshots = [recoverySnapshot(state)];
  const transitions = [
    () => prepareNodeStarted(state, START_ID),
    () => prepareNodeSucceeded(state, START_ID, { x: "Ada" }),
    () => prepareNodeStarted(state, "n"),
    () => prepareNodeSucceeded(state, "n", "Ada"),
    () => prepareNodeStarted(state, END_ID),
    () => prepareNodeSucceeded(state, END_ID, "Ada"),
  ];
  for (const transition of transitions) {
    commitPrepared(log, state, transition());
    snapshots.push(recoverySnapshot(state));
  }
  expect(state.result).not.toBeNull();
  return { loaded, log, snapshots };
}

/** A log whose CALL expansion holds a configured AGENT clone, its live state and effective map. */
async function agentGrowthHistory(): Promise<{
  loaded: ComposedFlow;
  live: RunState;
  log: MemoryRunLog;
  effective: Record<string, Record<string, unknown>>;
}> {
  const loaded = await load(CALL_AGENT);
  const privateFlow = copyGraph(loaded.compiled);
  const effective = resolveLlmCascade(
    privateFlow,
    { openai: { temperature: 0.25 } },
    { defaults: { provider: "openai", model: "recorded-model" } },
  );
  const log = new MemoryRunLog(RECOVERY_RUN_ID);
  const state = startedRun(loaded, log, { effective, privateFlow });
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, {}));
  commitPrepared(log, state, prepareNodeStarted(state, "gate"));
  commitPrepared(log, state, prepareExpanded(state, "gate", new Grow({}), buildGrow));
  return { loaded, live: state, log, effective };
}

test("test_replay_every_prefix_matches_live_state_and_uses_commit_path", async () => {
  const { loaded, log, snapshots } = await loadedLinearHistory();
  const envelopes = log.records();
  const originalManifest = flowManifest(loaded.compiled);

  snapshots.forEach((expected, index) => {
    const length = index + 1;
    const source = new ReadOnlyLog(envelopes.slice(0, length));
    const recovered = replayRunLog(loaded, source);

    expect(recoverySnapshot(recovered)).toEqual(expected);
    expect(source.appendCalls).toBe(0);
    expect(source.closeCalls).toBe(0);
    expect(source.nextSequence).toBe(length);
  });
  expect(flowManifest(loaded.compiled)).toEqual(originalManifest);
});

test("test_replay_preserves_ordered_unfinished_starts_for_later_resume", async () => {
  const loaded = await load(FANOUT);
  const log = new MemoryRunLog(RECOVERY_RUN_ID);
  const state = startedRun(loaded, log, { numWorkers: 4 });
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, {}));
  expect(state.ready).toEqual(["first", "second"]);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareNodeStarted(state, "second"));
  expect([...state.inFlight]).toEqual(["first", "second"]);

  const recovered = replayRunLog(loaded, new ReadOnlyLog(log.records()));

  expect(recovered.ready).toEqual([]);
  expect([...recovered.inFlight]).toEqual(["first", "second"]);
  expect(recovered.numWorkers).toBe(4);
});

test("test_replay_applies_logged_llm_map_before_rebuilding_growth", async () => {
  const { loaded, live, log, effective } = await agentGrowthHistory();
  const originalManifest = flowManifest(loaded.compiled);
  const source = new ReadOnlyLog(log.records());

  const recovered = replayRunLog(loaded, source);

  expect(recoverySnapshot(recovered)).toEqual(recoverySnapshot(live));
  expect(recovered.effectiveLlmByStaticPath).toEqual(effective as never);
  const leaf = recovered.flow.nodes.get("gate/leaf");
  expect(leaf).toBeInstanceOf(AgentNode);
  expect((leaf as AgentNode).llmConfig).toEqual(legacyConnectionConfig(effective["gate/leaf"]!));
  expect(source.appendCalls).toBe(0);
  expect(source.closeCalls).toBe(0);
  expect(flowManifest(loaded.compiled)).toEqual(originalManifest);
});

test("test_recorded_public_resume_keeps_completed_code_and_wire[6]", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-recovery-prefixes-"));
  const env: Env = (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER" || name === "YIAN_AUTH_FILE") throw new Error("read current defaults");
    return undefined;
  };
  try {
    const flow = await load(LINEAR);
    const path = join(dir, "historical.jsonl");
    const fresh = await runFlow(flow, { x: "Ada" }, {
      log: path,
      runId: "historical-linear",
      numWorkers: 0,
      env: hostEnv(dir),
    });
    expect(fresh.status).toBe("succeeded");
    const lines = readFileSync(path, "utf8").split("\n").slice(0, -1);
    const prefixLength = 6;
    const prefix = `${lines.slice(0, prefixLength).join("\n")}\n`;
    writeFileSync(path, prefix);
    const forward = spyOn(CodeNode.prototype, "forward").mockImplementation(() => {
      throw new Error("accepted CODE reran");
    });
    let result: RunResult;
    try {
      result = await resumeFlow({ log: path, env });
    } finally {
      forward.mockRestore();
    }

    expect(result.status).toBe("succeeded");
    expect(result.output).toBe("Ada");
    expect(result.runId).toBe("historical-linear");
    expect(result.input).toEqual({ x: "Ada" });
    expect(readFileSync(path, "utf8").startsWith(prefix)).toBe(true);
    const log = JsonlRunLog.resume(path);
    let envelopes: readonly RecordEnvelope[];
    let replayed: RunState;
    try {
      envelopes = log.records();
      replayed = replayRunLog(flow, log);
    } finally {
      log.close();
    }
    expect(envelopes.slice(0, prefixLength).map(encodeRecord)).toEqual(lines.slice(0, prefixLength));
    expect(envelopes.every((envelope) => envelope.schema === SCHEMA_VERSION)).toBe(true);
    const start = decodeRecord(lines[0]!).record as RunStartedRecord;
    expect(replayed.flowFingerprint).toBe(start.flowFingerprint);
    expect(start.flowFingerprint).toBe(flowFingerprint(flow.compiled));
    expect(replayed.effectiveLlmByStaticPath).toEqual(start.effectiveLlmByStaticPath as never);
    expect(replayed.numWorkers).toBe(0);
    expect(start.numWorkers).toBe(0);
    expect(replayed.learnable).toBe(false);
    expect(replayed.trace).toBeNull();
    const complete = readFileSync(path, "utf8");
    const inspected = await resumeFlow({ log: path, env });
    expect(inspected.output).toBe(result.output);
    expect(inspected.events).toEqual([]);
    expect(readFileSync(path, "utf8")).toBe(complete);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("test_replay_rejects_changed_but_valid_growth_digest", async () => {
  const { loaded, log } = await agentGrowthHistory();
  const envelopes = [...log.records()];
  const last = envelopes.pop()!;
  const grown = last.record as NodeExpandedRecord;
  const changed = new NodeExpandedRecord(grown.spawnerId, grown.seed, grown.pruneIds, "0".repeat(64));
  expect(grown.topologyDigest).not.toBe(changed.topologyDigest);
  envelopes.push(new RecordEnvelope(SCHEMA_VERSION, last.runId, last.seq, changed));

  const caught = thrownBy(() => replayRunLog(loaded, new ReadOnlyLog(envelopes)));
  expect(caught).toBeInstanceOf(RangeError);
  expect((caught as Error).message).toContain("topology digest mismatch");
});

test("test_replay_rejects_invalid_prefix_roots", async () => {
  const loaded = await load(LINEAR);
  const started = new NodeStartedRecord(START_ID);

  expectError(() => replayRunLog(loaded, new ReadOnlyLog([])), RangeError, "cannot replay an empty run log");
  expectError(
    () => replayRunLog(loaded, new ReadOnlyLog([new RecordEnvelope(SCHEMA_VERSION, RECOVERY_RUN_ID, 0, started)])),
    RangeError,
    "the first run log record must be RunStarted",
  );
  expectError(
    () => replayRunLog(loaded, new ReadOnlyLog([new RecordEnvelope(SCHEMA_VERSION, RECOVERY_RUN_ID, 1, started)])),
    RangeError,
    "run log sequence mismatch: expected 0, found 1",
  );
});

test("test_replay_rejects_log_identity_and_length_disagreement", async () => {
  const { loaded, log } = await loadedLinearHistory();
  const envelopes = log.records();

  expectError(
    () => replayRunLog(loaded, new ReadOnlyLog(envelopes, { runId: "different-run" })),
    RangeError,
    "run log run_id mismatch: expected 'different-run', found 'recovery-run'",
  );
  expectError(
    () => replayRunLog(loaded, new ReadOnlyLog(envelopes, { nextSequence: envelopes.length + 1 })),
    RangeError,
    "run log sequence mismatch: next_sequence is 8, prefix length is 7",
  );
});

test("test_fingerprint_mismatch_fails_before_copy_or_llm_apply", async () => {
  const { log } = await loadedLinearHistory();
  const source = new ReadOnlyLog(log.records());
  const changed = await load(LINEAR);
  changed.compiled.nodes.get("n")!.env = { "semantic-change": true };
  const before = flowManifest(changed.compiled);

  const caught = thrownBy(() => replayRunLog(changed, source));
  expect(caught).toBeInstanceOf(RangeError);
  expect((caught as Error).message).toStartWith("flow fingerprint mismatch: recorded '");
  expect(flowManifest(changed.compiled)).toEqual(before);
  expect(source.appendCalls).toBe(0);
});

test("test_replay_rejects_records_after_terminal_without_touching_source", async () => {
  const { loaded, log } = await loadedLinearHistory();
  const trailing = MemoryRunLog.fromRecords(log.records());
  trailing.append(new NodeStartedRecord(START_ID));
  const source = new ReadOnlyLog(trailing.records());

  expectError(() => replayRunLog(loaded, source), RangeError, "terminal run state cannot prepare another graph transition");

  expect(source.appendCalls).toBe(0);
  expect(source.closeCalls).toBe(0);
  expect(source.nextSequence).toBe(trailing.nextSequence);
});

test("test_run_started_replay_rejects_pool_and_envelope_identity_drift", async () => {
  const { loaded, log } = await loadedLinearHistory();
  const first = log.records()[0]!;
  const start = first.record as RunStartedRecord;
  const pool = structuredClone(start.initialPool) as { system: Record<string, { value: unknown }> };
  pool.system["run_id"]!.value = "other-run";
  const drifted = new RunStartedRecord({ ...start, initialPool: pool as never });
  const source = new ReadOnlyLog([new RecordEnvelope(SCHEMA_VERSION, first.runId, 0, drifted)]);

  const caught = thrownBy(() => replayRunLog(loaded, source));
  expect(caught).toBeInstanceOf(RangeError);
  expect((caught as Error).message).toContain("system.run_id");
});

test("a pooled run frozen after both starts counts each, and its resume publishes both", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-recovery-prefixes-"));
  const env = hostEnv(dir);
  try {
    const flow = await load(FANOUT);
    const path = join(dir, "pooled.jsonl");
    // Records 0-4 are RunStarted, START's two and both NodeStarted records.
    await crashAfter(5, () => runFlow(flow, {}, { log: path, numWorkers: 2, env }));

    const counts = new Map([
      ["first", 1],
      ["second", 1],
    ]);
    const expected = { status: "interrupted" as const, inFlight: ["first", "second"], interruptedAttempts: counts };
    expect(progressOf(inspectRun(path))).toEqual(expected);
    expect(progressOf(inspectRun(path, { flow, env }))).toEqual(expected);

    const seen: RunEvent[] = [];
    const result = await resumeFlow({ log: path, env, onEvent: (event) => seen.push(event) });
    expect(result.status).toBe("succeeded");
    expect(seen.slice(0, 3)).toEqual([
      new RunResumed(),
      new NodeAttemptResumed({ nodeId: "first", interrupted: 1 }),
      new NodeAttemptResumed({ nodeId: "second", interrupted: 1 }),
    ]);
    expect(result.interruptedAttempts).toEqual(counts);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
