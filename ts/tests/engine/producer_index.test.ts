/**
 * TS-only: the reducer's producer index and the `derived` node trait.
 *
 * `RunState.producers` maps each pool name to the seq of the accepted `pool_commit` that last
 * wrote it, in every run: run inputs and system values map to their run-start commits, any other
 * name to its node's commit once that node's `node_succeeded` names it. It is pruned with the
 * pool and rebuilt by replay. A derived node (START, END, LOOP, CALL, MAP: its value only reads
 * earlier values through its binding) gets an in-memory input route in a non-learnable run too,
 * whose source leaves name the commits its binding read; nothing more is written to the log.
 */

import { afterAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow } from "../../src/compile/expand.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { SourceRef } from "../../src/expr/lineage.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { MemoryRunLog, readRunLog, type RecordEnvelope, type RunRecord } from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";
import { crashAfter, NO_ENV, startedRun } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-producer-index-"));
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

/** A MAP over the run's input list. */
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

/** One node of every kind; loaded, never run. */
const EVERY_KIND = `
id: kinds
name: kinds
input: {xs: !list(int)}
defs:
  one:
    input: {x: !int}
    nodes:
      c: {kind: code, input: {x: "\${input.x}"}, output: int, code: "return inputs.x"}
    output: {x: "\${c.output}"}
nodes:
  ask: {kind: human_input, prompt: "q?", output: str}
  w: {kind: wait, until: "2030-01-01T00:00:00Z"}
  route:
    kind: case
    cases:
      - when: \${ask.output == "a"}
        then: yes
    else: no
  yes: {kind: agent, prompt: hi}
  no: {kind: tool, tool_id: echo, input: {n: "x"}, output: str}
  code: {kind: code, code: "return 1", output: int}
  one_call: {kind: call, call: one, input: {x: 1}}
  each: {kind: map, over: "\${input.xs}", call: one, input: {x: "\${item}"}}
  again: {kind: loop, call: one, input: {x: 0}, times: 2}
output: \${again.output}
`;

const load = (text: string) => loadFlow(text, { sourceLabel: "flow.yaml" });

/** Run `flow` serially into `log`, learnable or not. */
const run = (flow: ComposedFlow, log: string, inputs: Record<string, unknown> = {}, learnable = false) =>
  runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers: 0, learnable });

const DERIVED_KINDS = new Set<string>([NodeKind.START, NodeKind.END, NodeKind.LOOP, NodeKind.CALL, NodeKind.MAP]);

test("derived is true for START, END, LOOP, CALL and MAP and false for every other kind", async () => {
  const flow = await load(EVERY_KIND);
  const kinds = new Set<string>();
  for (const node of flow.compiled.nodes.values()) {
    kinds.add(node.kind);
    expect([node.kind, node.derived]).toEqual([node.kind, DERIVED_KINDS.has(node.kind)]);
  }
  expect([...kinds].sort()).toEqual(Object.values(NodeKind).sort());
});

test("a derived kind's forward is synchronous: it returns no promise", async () => {
  const spies = [StartNode, EndNode, LoopNode, CallNode, MapNode].map((kind) => spyOn(kind.prototype, "forward"));
  try {
    await run(await load(CALL_IN_LOOP), freshLog());
    await run(await load(DOUBLES), freshLog(), { xs: [1, 2] });
    for (const spy of spies) {
      expect(spy.mock.calls.length).toBeGreaterThan(0);
      for (const result of spy.mock.results) expect(result.value).not.toBeInstanceOf(Promise);
    }
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
});

/** The producer index as sorted entries, for comparing. */
const entries = (state: RunState) => [...state.producers].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));

/**
 * Run `text` live on an in-memory log; return the final state, the records, and the producer
 * index after each record from the run start's last commit on, by seq.
 */
async function liveRun(
  text: string,
  options: { inputs?: Record<string, unknown>; learnable?: boolean } = {},
): Promise<{ state: RunState; records: readonly RecordEnvelope[]; live: Map<number, unknown> }> {
  const log = new MemoryRunLog("r1");
  const state = startedRun(await load(text), log, options);
  const live = new Map<number, unknown>([[state.nextSequence - 1, entries(state)]]);
  const append = log.append.bind(log);
  // The state before each append is the state after the record before it.
  log.append = (record: RunRecord) => {
    live.set(state.nextSequence - 1, entries(state));
    return append(record);
  };
  await new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers: 0 }).run();
  live.set(state.nextSequence - 1, entries(state));
  return { state, records: log.records(), live };
}

/**
 * The seq of the accepted commit that last wrote each pool name: the run start's commits at once,
 * any other commit when a `node_succeeded` names it.
 */
function lastWriters(records: readonly RecordEnvelope[]): Map<string, number> {
  const writers = new Map<string, number>();
  let started = false;
  for (const { kind, seq, payload } of records) {
    if (kind === "pool_commit" && !started) {
      writers.set(payload["key"] as string, seq);
      started = payload["key"] === payload["node_id"];
    } else if (kind === "node_succeeded") {
      const commit = Number(payload["commit"]);
      writers.set(records[commit]!.payload["key"] as string, commit);
    }
  }
  return writers;
}

/** Every name `state`'s pool holds: its store keys and `system.<name>` for each system value. */
const poolNames = (state: RunState) =>
  [...state.pool.store.keys(), ...[...state.pool.system.keys()].map((name) => `system.${name}`)].sort();

test("each pool name maps to the seq of the accepted pool_commit that last wrote it", async () => {
  const { state, records } = await liveRun(CALL_IN_LOOP);
  expect(state.trace).toBeNull();
  const writers = lastWriters(records);
  expect([...state.producers.keys()].sort()).toEqual(poolNames(state));
  for (const [name, seq] of state.producers) {
    expect([name, seq]).toEqual([name, writers.get(name)!]);
    expect(records[seq]!.kind).toBe("pool_commit");
    expect(records[seq]!.payload["key"]).toBe(name);
  }
  // Run inputs and system values map to their run-start commits.
  expect(["system.now", "system.run_id", "system.today"].map((name) => state.producers.get(name))).toEqual([1, 2, 3]);
});

test("the run start's commits enter the index at once; START's own commit replaces its inputs'", async () => {
  const { live } = await liveRun(DOUBLES, { inputs: { xs: [1n, 2n] } });
  expect(live.get(4)).toEqual([
    [StartNode.ID, 4],
    ["system.now", 1],
    ["system.run_id", 2],
    ["system.today", 3],
  ]);
  expect(live.get(7)).toContainEqual([StartNode.ID, 6]);
});

test("an orphan commit never enters the index", async () => {
  const flow = await load(CALL_IN_LOOP);
  const uncut = freshLog();
  await run(flow, uncut);
  const envelopes = readRunLog(uncut);
  const commits = envelopes.filter(({ kind, seq }) => kind === "pool_commit" && seq > 4).map(({ seq }) => seq);
  for (const k of commits) {
    const log = freshLog();
    await crashAfter(k + 1, () => run(flow, log));
    const cut = readRunLog(log);
    const before = replayRunLog(flow, MemoryRunLog.fromRecords(cut.slice(0, k)));
    const after = replayRunLog(flow, MemoryRunLog.fromRecords(cut));
    expect([...after.producers.values()]).not.toContain(k);
    expect(entries(after)).toEqual(entries(before));
  }
}, 60_000);

test("in a learnable run, each name's commit is by the node whose start the retained trace names", async () => {
  const { state, records } = await liveRun(CALL_IN_LOOP, { learnable: true });
  const trace = state.trace!;
  let checked = 0;
  for (const [name, seq] of state.producers) {
    if (name.startsWith("system.")) continue;
    const source = trace.resolve(`${name}.output`).source as SourceRef;
    const started = records[source.executionSeq]!;
    expect(started.kind).toBe("node_started");
    expect([name, records[seq]!.payload["node_id"]]).toEqual([name, started.payload["node_id"]]);
    checked += 1;
  }
  expect(checked).toBeGreaterThan(1);
});
