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
import { RunSucceeded } from "../../src/events.ts";
import type { SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { type BindingCapture, BindingCollector } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import {
  MemoryRunLog,
  NodeStartedRecord,
  readRunLog,
  type RecordEnvelope,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { resolveFromPool, type RunState } from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
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
 * Run `text` live on an in-memory log; return the loaded flow, the final state, the records, and
 * `look(state)` after each record from the run start's last commit on, by seq (the producer index
 * by default).
 */
async function liveRun<T = unknown>(
  text: string,
  options: { inputs?: Record<string, unknown>; learnable?: boolean; look?: (state: RunState) => T } = {},
): Promise<{ loaded: ComposedFlow; state: RunState; records: readonly RecordEnvelope[]; live: Map<number, T> }> {
  const { look = entries as (state: RunState) => T, ...start } = options;
  const loaded = await load(text);
  const log = new MemoryRunLog("r1");
  const state = startedRun(loaded, log, start);
  const live = new Map<number, T>([[state.nextSequence - 1, look(state)]]);
  const append = log.append.bind(log);
  // The state before each append is the state after the record before it.
  log.append = (record: RunRecord) => {
    live.set(state.nextSequence - 1, look(state));
    return append(record);
  };
  await new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers: 0 }).run();
  live.set(state.nextSequence - 1, look(state));
  return { loaded, state, records: log.records(), live };
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

/** The run start's commits, by key: those after `run_started` through the START inputs'. */
function runStart(records: readonly RecordEnvelope[]): Map<string, number> {
  const commits = new Map<string, number>();
  for (const { kind, seq, payload } of records.slice(1)) {
    expect(kind).toBe("pool_commit");
    commits.set(payload["key"] as string, seq);
    if (payload["key"] === StartNode.ID) return commits;
  }
  throw new Error("the log has no START inputs commit");
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
  const start = runStart(records);
  for (const name of ["system.now", "system.run_id", "system.today"]) {
    expect([name, state.producers.get(name)]).toEqual([name, start.get(name)!]);
  }
});

test("the run start's commits enter the index at once; START's own commit replaces its inputs'", async () => {
  const { live, records } = await liveRun(DOUBLES, { inputs: { xs: [1n, 2n] } });
  const start = runStart(records);
  const inputs = start.get(StartNode.ID)!;
  expect(live.get(inputs)).toEqual([...start].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
  const success = records.find(({ kind, payload }) => kind === "node_succeeded" && payload["node_id"] === StartNode.ID)!;
  const commit = Number(success.payload["commit"]);
  expect(commit).toBeGreaterThan(inputs);
  expect(live.get(success.seq)).toContainEqual([StartNode.ID, commit]);
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

test("in a learnable run, each name's commit is by the node whose start the retained trace names, or its expansion's", async () => {
  const { state, records } = await liveRun(CALL_IN_LOOP, { learnable: true });
  const trace = state.trace!;
  let checked = 0;
  for (const [name, seq] of state.producers) {
    if (name.startsWith("system.")) continue;
    const source = trace.resolve(`${name}.output`).source as SourceRef;
    const started = records[source.executionSeq]!;
    expect(started.kind).toBe("node_started");
    // A child START's commit is its expansion's, by the spawner, under the START's key.
    const commit = records[seq]!.payload;
    const writer = commit["key"] !== commit["node_id"] && commit["key"] === started.payload["node_id"] ? commit["key"] : commit["node_id"];
    expect([name, writer]).toEqual([name, started.payload["node_id"]]);
    checked += 1;
  }
  expect(checked).toBeGreaterThan(1);
});

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}, false],
  ["a CALL in a loop, learnable", CALL_IN_LOOP, {}, true],
  ["a MAP", DOUBLES, { xs: [1n, 2n, 3n] }, false],
] as const)("replaying the log cut at any record restores the live index: %s", async (_label, text, inputs, learnable) => {
  const { loaded, records, live } = await liveRun(text, { inputs, learnable });
  expect(records.at(-1)!.kind).toBe("node_succeeded");
  // A cut inside the run start can't be replayed; every later cut can.
  const first = Math.min(...live.keys());
  expect(live.size).toBe(records.length - first);
  for (const [seq, expected] of live) {
    const replayed = replayRunLog(loaded, MemoryRunLog.fromRecords(records.slice(0, seq + 1)));
    expect<unknown>([seq, entries(replayed)]).toEqual([seq, expected]);
  }
}, 60_000);

/** A non-learnable loop of `times` iterations whose body bumps a counter. */
const counter = (times: number) => `
id: counter
name: counter
defs:
  bump:
    input: {x: !int}
    nodes:
      inc: {kind: code, input: {x: "\${input.x}"}, output: int, code: "return inputs.x + 1"}
    output: {x: "\${inc.output}"}
nodes:
  count: {kind: loop, call: bump, input: {x: 0}, times: ${times}}
output: \${count.output.x}
`;

test("a 200-iteration non-learnable loop keeps no trace and a bounded index", async () => {
  const { state, records, live } = await liveRun(counter(200), { look: (state) => state.producers.size });
  expect(state.trace).toBeNull();
  expect(state.result).toEqual(new RunSucceeded({ output: 200n }));
  // The index after each body expansion, in iteration order.
  const sizes = records
    .filter(({ kind, payload }) => kind === "node_expanded" && /^count(~\d+)?$/.test(payload["spawner_id"] as string))
    .map(({ seq }) => live.get(seq)!);
  expect(sizes).toHaveLength(200);
  expect(sizes[199]).toBe(sizes[9]!);
  expect(state.producers.size).toBe(state.pool.store.size + state.pool.system.size);
}, 60_000);

/** A loop inside a loop's body, counting to four. */
const NESTED = `
id: nested
name: nested
defs:
  bump_body:
    input: {n: !int}
    nodes:
      bump: {kind: code, input: {n: "\${input.n}"}, output: {n: int}, code: "return {n: inputs.n + 1}"}
    output: {n: "\${bump.output.n}"}
  outer_body:
    input: {n: !int}
    nodes:
      inner: {kind: loop, call: bump_body, input: {n: "\${input.n}"}, times: 2}
    output: {n: "\${inner.output.n}"}
nodes:
  outer: {kind: loop, call: outer_body, input: {n: 0}, times: 2}
output: \${outer.output.n}
`;

/** A chat loop whose turn END appends to the history with \`+\` and to the log with a template. */
const CHAT = `
id: chat
name: chat
typedefs:
  Msg: {role: str, text: str}
defs:
  turn:
    input: {history: !list(Msg), log: !str}
    nodes:
      ask: {kind: code, input: {log: "\${input.log}"}, output: str, code: "return 'q'"}
      reply: {kind: code, input: {history: "\${input.history}", message: "\${ask.output}"}, output: str, code: "return 'a'"}
    output:
      history: \${input.history + [Msg(role="user", text=ask.output), Msg(role="assistant", text=reply.output)]}
      log: "\${input.log}\\nUser: \${ask.output}\\nAssistant: \${reply.output}"
nodes:
  chat: {kind: loop, call: turn, input: {history: [], log: ""}, times: 3}
output: \${chat.output.log}
`;

/** The source leaves of `route`, its children and its snapshot. */
function leaves(route: ValueRoute, found: SourceRef[] = []): SourceRef[] {
  if (route.source !== null) found.push(route.source);
  for (const child of route.children) leaves(child, found);
  if (route.snapshot !== null) leaves(route.snapshot, found);
  return found;
}

/** The source leaves of a binding capture's route and of each per-item capture's. */
const captureLeaves = (capture: BindingCapture): SourceRef[] => [
  ...leaves(capture.route),
  ...capture.items.flatMap(([, item]) => captureLeaves(item)),
];

/** One node evaluation that captured its bindings, and the producer index when it ran. */
interface Captured {
  nodeId: string;
  derived: boolean;
  capture: BindingCapture;
  producers: ReadonlyMap<string, number>;
}

/**
 * Run `text` serially, collecting every evaluation's binding capture by node, and the derived
 * nodes started, in start order.
 */
async function capturedRun(
  text: string,
  options: { inputs?: Record<string, unknown>; learnable?: boolean } = {},
): Promise<{ captured: Captured[]; derivedStarts: string[]; records: readonly RecordEnvelope[]; state: RunState }> {
  const loaded = await load(text);
  const log = new MemoryRunLog("r1");
  const state = startedRun(loaded, log, options);
  const derivedStarts: string[] = [];
  const append = log.append.bind(log);
  log.append = (record: RunRecord) => {
    if (record instanceof NodeStartedRecord && state.flow.nodes.get(record.nodeId)!.derived) {
      derivedStarts.push(record.nodeId);
    }
    return append(record);
  };
  const captured: Captured[] = [];
  const freeze = BindingCollector.prototype.freeze;
  let depth = 0;
  // Each evaluation freezes its collector once, and a MAP's per-item collectors inside it.
  const spy = spyOn(BindingCollector.prototype, "freeze").mockImplementation(function (this: BindingCollector) {
    depth += 1;
    try {
      const capture = freeze.call(this);
      if (depth === 1) {
        // A serial run evaluates one node at a time, the one in flight.
        const [nodeId] = state.inFlight;
        const node = state.flow.nodes.get(nodeId!)!;
        captured.push({ nodeId: nodeId!, derived: node.derived, capture, producers: new Map(state.producers) });
      }
      return capture;
    } finally {
      depth -= 1;
    }
  });
  try {
    await new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers: 0 }).run();
  } finally {
    spy.mockRestore();
  }
  expect(state.result).toBeInstanceOf(RunSucceeded);
  return { captured, derivedStarts, records: log.records(), state };
}

/** The pool name a non-learnable leaf names: the key of the commit at its seq. */
const commitName = (records: readonly RecordEnvelope[], leaf: SourceRef): string => {
  expect(records[leaf.executionSeq]!.kind).toBe("pool_commit");
  return records[leaf.executionSeq]!.payload["key"] as string;
};

/**
 * The pool name a learnable leaf names: the key of the commit its node's success accepted after
 * the start at its seq; seq zero is the run's inputs, under the START's id.
 */
const startName = (records: readonly RecordEnvelope[], leaf: SourceRef): string => {
  if (leaf.executionSeq === 0) return StartNode.ID;
  const started = records[leaf.executionSeq]!;
  expect(started.kind).toBe("node_started");
  const success = records.find(
    ({ kind, seq, payload }) =>
      kind === "node_succeeded" && seq > started.seq && payload["node_id"] === started.payload["node_id"],
  )!;
  return records[Number(success.payload["commit"])]!.payload["key"] as string;
};

/** Each captured node's sorted read names, by node id in evaluation order. */
const readNames = (captured: readonly Captured[], name: (leaf: SourceRef) => string) =>
  captured.map(({ nodeId, capture }) => [nodeId, [...new Set(captureLeaves(capture).map(name))].sort()] as const);

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}],
  ["a MAP", DOUBLES, { xs: [1n, 2n, 3n] }],
  ["a nested loop", NESTED, {}],
  ["a chat appending with + and a template", CHAT, {}],
] as const)("a derived node's in-memory route names the commits its binding read: %s", async (_label, text, inputs) => {
  const plain = await capturedRun(text, { inputs });
  expect(plain.state.trace).toBeNull();
  // Only derived nodes capture, each once per start.
  expect(plain.captured.every(({ derived }) => derived)).toBe(true);
  expect(plain.captured.map(({ nodeId }) => nodeId)).toEqual(plain.derivedStarts);
  expect(plain.captured.length).toBeGreaterThan(3);
  for (const { capture, producers } of plain.captured) {
    for (const leaf of captureLeaves(capture)) {
      expect(leaf.runId).toBe("r1");
      expect(producers.get(commitName(plain.records, leaf))).toBe(leaf.executionSeq);
    }
  }
  const learnable = await capturedRun(text, { inputs, learnable: true });
  const learned = learnable.captured.filter(({ derived }) => derived);
  expect(readNames(plain.captured, (leaf) => commitName(plain.records, leaf))).toEqual(
    readNames(learned, (leaf) => startName(learnable.records, leaf)),
  );
}, 60_000);

test("a derived node's route names exactly the pool names its binding reads", async () => {
  const { captured, records } = await capturedRun(CALL_IN_LOOP);
  const names = new Map(readNames(captured, (leaf) => commitName(records, leaf)));
  expect(names.get(StartNode.ID)).toEqual([]);
  expect(names.get("count")).toEqual([]);
  expect(names.get("count#0/step")).toEqual(["count#0/__start__"]);
  expect(names.get("count#0/step/__end__")).toEqual(["count#0/step/inc"]);
  expect(names.get("count#0/__end__")).toEqual(["count#0/step"]);
  expect(names.get("count~1")).toEqual(["count#0/__end__"]);
});

/** A flow whose END reads a node's output and a system value. */
const READS_SYSTEM = `
id: sys
name: sys
nodes:
  c: {kind: code, code: "return 1", output: int}
output: {n: "\${c.output}", rid: "\${system.run_id}"}
`;

test("a derived binding reading a system value: without a trace its leaf names the run-start commit; the trace gives none", async () => {
  const plain = await capturedRun(READS_SYSTEM);
  const end = plain.captured.find(({ nodeId }) => nodeId === EndNode.ID)!;
  const runIdCommit = runStart(plain.records).get("system.run_id")!;
  const plainLeaves = captureLeaves(end.capture);
  expect(plainLeaves.map((leaf) => leaf.executionSeq)).toContain(runIdCommit);
  expect(readNames([end], (leaf) => commitName(plain.records, leaf))).toEqual([[EndNode.ID, ["c", "system.run_id"]]]);
  // The retained trace resolves a system value with no source, so the learnable route has no leaf for it.
  const learnable = await capturedRun(READS_SYSTEM, { learnable: true });
  const learnedEnd = learnable.captured.find(({ nodeId }) => nodeId === EndNode.ID)!;
  expect(readNames([learnedEnd], (leaf) => startName(learnable.records, leaf))).toEqual([[EndNode.ID, ["c"]]]);
});

test("a pool name missing from the producer index is a RangeError", () => {
  const pool = new VariablePool(StartNode.ID);
  pool.set("a", { x: 1n });
  pool.addSystem("run_id", "r1");
  expect(() => resolveFromPool("a.output.x", pool, new Map(), "r1")).toThrow(
    new RangeError("pool name 'a' has no pool_commit in the producer index"),
  );
  expect(() => resolveFromPool("system.run_id", pool, new Map(), "r1")).toThrow(RangeError);
  // A miss is still a miss, not an error.
  expect(resolveFromPool("b.output", pool, new Map(), "r1").present).toBe(false);
  expect(resolveFromPool("a.output.y", pool, new Map([["a", 7]]), "r1").present).toBe(false);
  const found = resolveFromPool("a.output.x", pool, new Map([["a", 7]]), "r1");
  expect([found.value, found.present, (found.source as SourceRef).executionSeq, (found.source as SourceRef).path]).toEqual([
    1n,
    true,
    7,
    ["x"],
  ]);
});
