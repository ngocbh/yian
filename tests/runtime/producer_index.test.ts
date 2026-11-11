/**
 * The reducer's producer index and the `derived` node trait.
 *
 * `RunState.producers` maps each pool name to the seq of the accepted `pool_commit` that last
 * wrote it, in every run: run inputs and system values map to their run-start commits, any other
 * name to its node's commit once that node's `node_succeeded` names it. It is pruned with the
 * pool and rebuilt by replay. A derived node (START, END, LOOP, CALL, MAP: its value only reads
 * earlier values through its binding) names in its commit's `from` the commits its binding read.
 */

import { afterAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow } from "../../src/compile/expand.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { RunSucceeded } from "../../src/events.ts";
import { NodeKind } from "../../src/nodes/base.ts";
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
import { referenceRoot, VariablePool } from "../../src/typesys/pool.ts";
import { crashAfter, NO_ENV, startedRun } from "../helpers/run_state_fixtures.ts";

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
  ask: {kind: tool, tool_id: ask, input: {prompt: "q?"}, output: str}
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

/** Run `flow` serially into `log`. */
const run = (flow: ComposedFlow, log: string, inputs: Record<string, unknown> = {}) =>
  runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers: 0 });

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
  options: { inputs?: Record<string, unknown>; look?: (state: RunState) => T } = {},
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

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}],
  ["a MAP", DOUBLES, { xs: [1n, 2n, 3n] }],
] as const)("replaying the log cut at any record restores the live index: %s", async (_label, text, inputs) => {
  const { loaded, records, live } = await liveRun(text, { inputs });
  expect(records.at(-1)!.kind).toBe("node_succeeded");
  // A cut inside the run start can't be replayed; every later cut can.
  const first = Math.min(...live.keys());
  expect(live.size).toBe(records.length - first);
  for (const [seq, expected] of live) {
    const replayed = replayRunLog(loaded, MemoryRunLog.fromRecords(records.slice(0, seq + 1)));
    expect<unknown>([seq, entries(replayed)]).toEqual([seq, expected]);
  }
}, 60_000);

/** A loop of `times` iterations whose body bumps a counter. */
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

test("a 200-iteration loop keeps a bounded index", async () => {
  const { state, records, live } = await liveRun(counter(200), { look: (state) => state.producers.size });
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

/** Each derived commit in log order: its node id and the seqs its `from` names, by pool name. */
function froms(records: readonly RecordEnvelope[]): [nodeId: string, from: Map<string, number>][] {
  return records
    .filter(({ kind, payload }) => kind === "pool_commit" && Object.hasOwn(payload, "from"))
    .map(({ payload }) => [
      payload["node_id"] as string,
      new Map(Object.entries(payload["from"] as Record<string, bigint>).map(([name, seq]) => [name, Number(seq)])),
    ]);
}

/** Run `text` serially into a fresh log and return its records. */
async function loggedRun(text: string, inputs: Record<string, unknown> = {}): Promise<readonly RecordEnvelope[]> {
  const log = freshLog();
  expect((await run(await load(text), log, inputs)).status).toBe("succeeded");
  return readRunLog(log);
}

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}],
  ["a MAP", DOUBLES, { xs: [1n, 2n, 3n] }],
  ["a nested loop", NESTED, {}],
  ["a chat appending with + and a template", CHAT, {}],
] as const)("a derived node's commit names the commits its binding read: %s", async (_label, text, inputs) => {
  const records = await loggedRun(text, inputs);
  const found = froms(records);
  expect(found.length).toBeGreaterThan(3);
  for (const [, from] of found) {
    for (const [name, seq] of from) {
      expect([name, records[seq]!.kind]).toEqual([name, "pool_commit"]);
      expect(records[seq]!.payload["key"]).toBe(name);
    }
  }
}, 60_000);

test("a derived node's commit names exactly the pool names its binding reads", async () => {
  const names = new Map(froms(await loggedRun(CALL_IN_LOOP)).map(([nodeId, from]) => [nodeId, [...from.keys()].sort()]));
  expect(names.get("count")).toEqual([]);
  expect(names.get("count#0/step")).toEqual(["count#0/__start__"]);
  expect(names.get("count#0/step/__end__")).toEqual(["count#0/step/inc"]);
  expect(names.get("count#0/__end__")).toEqual(["count#0/step"]);
  expect(names.get("count~1")).toEqual(["count#0/__end__"]);
});

/** A flow whose CALL's child END, a derived commit, reads a node's output and a system value. */
const READS_SYSTEM = `
id: sys
name: sys
defs:
  inner:
    nodes:
      c: {kind: code, code: "return 1", output: int}
    output: {n: "\${c.output}", rid: "\${system.run_id}"}
nodes:
  call_it: {kind: call, call: inner}
output: \${call_it.output}
`;

test("a derived binding reading a system value names the run-start commit", async () => {
  const records = await loggedRun(READS_SYSTEM);
  const [, from] = froms(records).find(([nodeId]) => nodeId === `call_it/${EndNode.ID}`)!;
  expect([...from.keys()].sort()).toEqual(["call_it/c", "system.run_id"]);
  expect(from.get("system.run_id")).toBe(runStart(records).get("system.run_id")!);
});

test("a pool name missing from the producer index is a RangeError", () => {
  const pool = new VariablePool(StartNode.ID);
  pool.set("a", { x: 1n });
  pool.addSystem("run_id", "r1");
  expect(() => resolveFromPool(["a", "output", "x"], pool, new Map(), "r1")).toThrow(
    new RangeError("pool name 'a' has no pool_commit in the producer index"),
  );
  expect(() => resolveFromPool(["system", "run_id"], pool, new Map(), "r1")).toThrow(RangeError);
  // A miss is still a miss, not an error.
  expect(resolveFromPool(["b", "output"], pool, new Map(), "r1").present).toBe(false);
  const found = resolveFromPool(["a", "output", "x"], pool, new Map([["a", 7]]), "r1");
  expect([found.used, found.present, found.value]).toEqual([2, true, { x: 1n }]);
});

test("referenceRoot reads x.new_messages[.path] as that pool name when it is held", () => {
  const holds = (name: string) => name === "x" || name === "x.new_messages";
  expect(referenceRoot(["x", "new_messages", "0"], "__start__", holds)).toEqual({ name: "x.new_messages", used: 2 });
  expect(referenceRoot(["x", "new_messages"], "__start__", holds)).toEqual({ name: "x.new_messages", used: 2 });
  expect(referenceRoot(["y", "new_messages"], "__start__", holds)).toEqual({ name: null, used: 2 });
  expect(referenceRoot(["x", "output"], "__start__", holds)).toEqual({ name: "x", used: 2 });
});
