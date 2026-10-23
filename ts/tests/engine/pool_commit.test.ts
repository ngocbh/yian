/**
 * TS-only: every write to the variable pool is a `pool_commit` record. The run start writes the
 * system values and the START inputs as commits right after `run_started`; a node's output is a
 * commit its `node_succeeded` names by seq. An orphan commit (its node never succeeded) is ignored
 * on replay and its node reruns; a log cut inside the run start can't be resumed.
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { inspectRun, replayRunLog, resumeFlow, runFlow, runRecordOffsets } from "../../src/run.ts";
import { iterRunRecords, JsonlRunLog, MemoryRunLog, readRunLog, type RecordEnvelope } from "../../src/runtime/log.ts";
import { type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import { crashAfter, NO_ENV, rejected } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-pool-commit-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's temp folder. */
const freshLog = () => join(tmp, `log-${++logs}.jsonl`);

/** A loop whose body CALLs another def, so child ENDs commit under their spawners. */
const CALL_IN_LOOP = `
id: count
name: count
defs:
  add1:
    input: {x: !int}
    nodes:
      inc: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x + 1}"}
    output: {y: "\${inc.output.y}"}
  body:
    input: {x: !int}
    nodes:
      step: {kind: call, call: add1, input: {x: "\${input.x}"}}
    output: {x: "\${step.output.y}"}
nodes:
  count: {kind: loop, call: body, input: {x: 0}, times: 2}
output: \${count.output.x}
`;

/** A MAP over three items. */
const DOUBLES = `
id: doubles
name: doubles
input: {xs: !list(int)}
defs:
  double:
    input: {x: !int}
    nodes:
      times2: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x * 2}"}
    output: \${times2.output.y}
nodes:
  each: {kind: map, over: "\${input.xs}", call: double, input: {x: "\${item}"}}
output: \${each.output}
`;

const load = (text: string) => loadFlow(text, { sourceLabel: "flow.yaml" });

/** Run `flow` serially (or with `numWorkers`) into `log`. */
const run = (flow: ComposedFlow, log: string, inputs: Record<string, unknown> = {}, numWorkers = 0) =>
  runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers });

/** The log's lines, each without its newline. */
const lines = (log: string) => readFileSync(log, "utf8").split("\n").slice(0, -1);

/** The seq of every `pool_commit` a `node_succeeded` names, by naming record seq. */
function named(envelopes: readonly RecordEnvelope[]): Map<number, number> {
  const commits = new Map<number, number>();
  for (const { kind, seq, payload } of envelopes) {
    if (kind === "node_succeeded") commits.set(seq, Number(payload["commit"]));
  }
  return commits;
}

test("the run start writes its system values, then the START inputs, as commits after run_started", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  expect(envelopes[0]!.kind).toBe("run_started");
  expect(Object.hasOwn(envelopes[0]!.payload, "initial_pool")).toBe(false);
  const start = envelopes.slice(1, 5).map(({ kind, payload }) => [kind, payload["node_id"], payload["key"]]);
  expect(start).toEqual([
    ["pool_commit", StartNode.ID, "system.now"],
    ["pool_commit", StartNode.ID, "system.run_id"],
    ["pool_commit", StartNode.ID, "system.today"],
    ["pool_commit", StartNode.ID, StartNode.ID],
  ]);
  expect(envelopes[2]!.payload["value"]).toBe("r1");
  expect(envelopes[4]!.payload).toEqual({ node_id: StartNode.ID, key: StartNode.ID, kind: "object", value: {} });
  expect(envelopes[5]!.kind).toBe("node_started");
});

test("every node_succeeded names its own node's earlier commit, and each commit is named once", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  const commits = named(envelopes);
  expect(commits.size).toBeGreaterThan(5);
  for (const [seq, commit] of commits) {
    const named = envelopes[commit]!;
    expect(named.kind).toBe("pool_commit");
    expect(commit).toBeLessThan(seq);
    expect(named.payload["node_id"]).toBe(envelopes[seq]!.payload["node_id"]);
  }
  const nodeCommits = envelopes.filter(({ kind, seq }) => kind === "pool_commit" && seq > 4).map(({ seq }) => seq);
  expect([...commits.values()].sort((a, b) => a - b)).toEqual(nodeCommits);
});

test("each commit's key is the node's commit target: a CALL's child END and the stopping LOOP driver commit under their spawner", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const envelopes = readRunLog(log);
  const moved = envelopes
    .filter(({ kind, seq, payload }) => kind === "pool_commit" && seq > 4 && payload["key"] !== payload["node_id"])
    .map(({ payload }) => [payload["node_id"], payload["key"]]);
  // Schema 12 recorded these as the successes' `commit_target`s.
  expect(moved).toEqual([
    ["count#0/step/__end__", "count#0/step"],
    ["count#1/step/__end__", "count#1/step"],
    ["count~2", "count"],
  ]);
});

test("a cut between a node's commit and its success reruns the node, leaving the orphan unread", async () => {
  const flow = await load(CALL_IN_LOOP);
  const uncutLog = freshLog();
  const uncut = await run(flow, uncutLog);
  const uncutEnvelopes = readRunLog(uncutLog);
  const cuts = [...named(uncutEnvelopes).values()];
  for (const k of cuts) {
    const node = uncutEnvelopes[k]!.payload["node_id"] as string;
    const key = uncutEnvelopes[k]!.payload["key"] as string;
    const log = freshLog();
    await crashAfter(k + 1, () => run(flow, log));
    const cutEnvelopes = readRunLog(log);
    expect(cutEnvelopes.at(-1)!.kind).toBe("pool_commit");

    // The orphan is ignored on replay: the pool holds what it held before the commit.
    const inspection = inspectRun(log, { flow, env: NO_ENV });
    expect(inspection.status).toBe("interrupted");
    expect([...inspection.inFlight]).toContain(node);
    const replayed = (count: number) => replayRunLog(flow, MemoryRunLog.fromRecords(cutEnvelopes.slice(0, count)));
    const before = replayed(k).pool.store.get(key);
    const after = replayed(k + 1).pool.store.get(key);
    expect(after === undefined ? undefined : logJson(after.toWire())).toBe(
      before === undefined ? undefined : logJson(before.toWire()),
    );

    const orphan = lines(log)[k];
    const resumed = await resumeFlow({ log, env: NO_ENV });
    expect(resumed.status).toBe("succeeded");
    expect(resumed.output).toEqual(uncut.output);
    expect(lines(log)[k]).toBe(orphan);
    const envelopes = readRunLog(log);
    const starts = envelopes.filter(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node);
    expect(starts).toHaveLength(2);
    const commits = envelopes.filter(({ kind, payload }) => kind === "pool_commit" && payload["node_id"] === node);
    const rerun = commits.find(({ seq }) => seq > k)!;
    expect(rerun).toBeDefined();
    const success = envelopes.find(({ kind, seq, payload }) => kind === "node_succeeded" && seq > k && payload["node_id"] === node);
    expect(Number(success!.payload["commit"])).toBe(rerun.seq);
  }
}, 60_000);

/**
 * Return the log text `envelopes` would have with the record at `from` moved to just after the
 * record at `to`, cut after the record that followed `to`, every seq renumbered.
 */
function interleaved(text: readonly string[], to: number, from: number): string {
  const records = text.map((line) => parseJson(line, "codec") as JsonObject);
  const moved = records.splice(from, 1)[0]!;
  records.splice(to + 1, 0, moved);
  return records
    .slice(0, to + 3)
    .map((record, seq) => `${logJson({ ...record, seq: BigInt(seq) })}\n`)
    .join("");
}

test("parallel nodes whose commits interleave: only the node without its success reruns", async () => {
  const flow = await load(DOUBLES);
  const uncutLog = freshLog();
  await run(flow, uncutLog, { xs: [1, 2, 3] }, 2);
  const envelopes = readRunLog(uncutLog);
  // The first success whose commit comes after another node's start, and a later node's commit.
  const successes = envelopes.filter(({ kind }) => kind === "node_succeeded");
  let pair: [number, number] | null = null;
  for (const success of successes) {
    const node = success.payload["node_id"] as string;
    const commit = Number(success.payload["commit"]);
    const other = envelopes.find(
      ({ kind, seq, payload }) => kind === "pool_commit" && seq > success.seq && payload["node_id"] !== node &&
        envelopes.some((e) => e.kind === "node_started" && e.seq < commit && e.payload["node_id"] === payload["node_id"]),
    );
    if (other !== undefined) {
      pair = [commit, other.seq];
      break;
    }
  }
  expect(pair).not.toBeNull();
  const [first, second] = pair!;
  const finished = envelopes[first]!.payload["node_id"] as string;
  const rerun = envelopes[second]!.payload["node_id"] as string;
  const log = freshLog();
  writeFileSync(log, interleaved(lines(uncutLog), first, second));
  const cut = readRunLog(log);
  expect(cut.slice(-3).map(({ kind, payload }) => [kind, payload["node_id"]])).toEqual([
    ["pool_commit", finished],
    ["pool_commit", rerun],
    ["node_succeeded", finished],
  ]);

  const resumed = await resumeFlow({ log, env: NO_ENV });
  expect(resumed.output).toEqual([2n, 4n, 6n]);
  const after = readRunLog(log);
  const startsOf = (node: string) => after.filter(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node);
  expect(startsOf(finished)).toHaveLength(1);
  expect(startsOf(rerun)).toHaveLength(2);
  const success = after.find(({ kind, payload }) => kind === "node_succeeded" && payload["node_id"] === finished)!;
  expect(Number(success.payload["commit"])).toBe(first);
});

/** Replace the `commit` of the record at `seq` in the log `text` and return the new text. */
function withCommit(text: readonly string[], seq: number, commit: number): string {
  const record = parseJson(text[seq]!, "codec") as { payload: JsonObject };
  record.payload["commit"] = BigInt(commit);
  return [...text.slice(0, seq), logJson(record), ...text.slice(seq + 1)].map((line) => `${line}\n`).join("");
}

test("a node_succeeded naming a commit that isn't its node's pending one is refused on replay", async () => {
  const flow = await load(CALL_IN_LOOP);
  const log = freshLog();
  await run(flow, log);
  const text = lines(log);
  const envelopes = readRunLog(log);
  const successes = [...named(envelopes)];
  const [seq, own] = successes[2]!;
  const node = envelopes[seq]!.payload["node_id"] as string;
  const other = successes[1]![1];
  const started = envelopes.find(({ kind, payload }) => kind === "node_started" && payload["node_id"] === node)!.seq;
  for (const commit of [other, 4, started]) {
    expect(commit).not.toBe(own);
    const edited = freshLog();
    writeFileSync(edited, withCommit(text, seq, commit));
    const replay = () => replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(edited)));
    expect(replay).toThrow(RangeError);
    expect(replay).toThrow(`node_succeeded of '${node}' names commit ${commit}, not its pending pool_commit ${own}`);
  }
});

test("a pool_commit for a node that isn't running is refused on replay", async () => {
  const flow = await load(CALL_IN_LOOP);
  const log = freshLog();
  await run(flow, log);
  const text = lines(log);
  const envelopes = readRunLog(log);
  const first = envelopes.find(({ kind, seq }) => kind === "pool_commit" && seq > 4)!;
  const record = parseJson(text[first.seq]!, "codec") as { payload: JsonObject };
  record.payload["node_id"] = "count";
  const edited = freshLog();
  writeFileSync(edited, [...text.slice(0, first.seq), logJson(record)].map((line) => `${line}\n`).join(""));
  const replay = () => replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(edited)));
  expect(replay).toThrow(RangeError);
  expect(replay).toThrow("pool_commit node 'count' is not in flight");
});

test("a log cut inside its run start can't be resumed, and stays as it is", async () => {
  const flow = await load(CALL_IN_LOOP);
  for (const k of [1, 2, 3, 4]) {
    const log = freshLog();
    await crashAfter(k, () => run(flow, log));
    const before = readFileSync(log);
    const error = await rejected(resumeFlow({ log, env: NO_ENV }));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(`run log ${log} ends inside its run start; start the run again`);
    expect(readFileSync(log).equals(before)).toBe(true);
  }
});

test("runRecordOffsets maps each seq to its line's byte offset, without the writer's lock", async () => {
  const log = freshLog();
  await run(await load(CALL_IN_LOOP), log);
  const text = readFileSync(log);
  const expected: number[] = [];
  for (let offset = 0; offset < text.length; offset = text.indexOf(0x0a, offset) + 1) expected.push(offset);
  const writer = JsonlRunLog.resume(log);
  try {
    const offsets = runRecordOffsets(log);
    expect(offsets).toEqual(expected);
    for (const k of [0, 1, 5, offsets.length - 1]) {
      const first = iterRunRecords(log, { afterOffset: offsets[k]! }).next().value!;
      expect(first.seq).toBe(k);
    }
  } finally {
    writer.close();
  }
});
