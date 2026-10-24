/**
 * TS-only: derived commits hold `from` pointers, and the expansion commits the child STARTs.
 *
 * A derived commit (a non-top `__end__`, the stopping LOOP driver) holds `from`: each pool name its
 * binding read, to the seq of the `pool_commit` that wrote it. Before `node_expanded`, the spawner
 * writes one `pool_commit` per START of the fragment (`node_id` the spawner, `key` the child START),
 * whose `from` is the spawner's; the child START's success names that commit and it writes none of
 * its own. Replay rebuilds a derived value from the pool and checks each named seq against the
 * producer index.
 */

import { afterAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow } from "../../src/compile/expand.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Node, Output } from "../../src/nodes/base.ts";
import { replayRunLog, resumeFlow, runFlow } from "../../src/run.ts";
import { rebuildDerived } from "../../src/runtime/eval_node.ts";
import {
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  readRunLog,
  type RecordEnvelope,
} from "../../src/runtime/log.ts";
import { rebuildCommitValue, rebuildExpansion } from "../../src/runtime/state.ts";
import { logJson } from "../../src/typesys/json.ts";
import { TypedValue } from "../../src/typesys/values.ts";
import { crashAfter, NO_ENV, serving, textReply } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-derived-records-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's temp folder. */
const freshLog = () => join(tmp, `log-${++logs}.jsonl`);

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** PR 1's chat loop, three scripted turns: each turn asks, replies, and appends in its END binding. */
const CHAT = `
id: chat
name: chat
llm_config: {provider: anthropic, model: claude-test}
typedefs:
  Msg: {role: str, text: str}
defs:
  turn:
    input: {history: !list(Msg), log: !str}
    nodes:
      ask: {kind: human_input, prompt: "You:", output: str}
      reply:
        kind: agent
        input: {history: "\${input.history}", message: "\${ask.output}"}
        output: str
        prompt: |-
          Conversation so far: \${history}
          User: \${message}
    output:
      history: \${input.history + [Msg(role="user", text=ask.output), Msg(role="assistant", text=reply.output)]}
      log: "\${input.log}\\nUser: \${ask.output}\\nAssistant: \${reply.output}"
nodes:
  chat: {kind: loop, call: turn, input: {history: [], log: ""}, times: 3}
output: \${chat.output.log}
`;

const load = (text: string) => loadFlow(text, { sourceLabel: "flow.yaml" });

/** The run options for the chat: scripted answers and replies, non-learnable unless `learnable`. */
function chatOptions(log: string, learnable = false) {
  const answers = ["a", "b", "c"];
  return {
    env: NO_ENV,
    log,
    runId: "r1",
    numWorkers: 0,
    learnable,
    llmConfig: CONFIG,
    models: serving(new ScriptedModel([textReply("A"), textReply("B"), textReply("C")], CONFIG)),
    tools: { ask_user: () => answers.shift()! },
  };
}

/** Run the chat into a fresh log and return the log's envelopes. */
async function chatLog(learnable = false): Promise<readonly RecordEnvelope[]> {
  const log = freshLog();
  const result = await runFlow(await load(CHAT), {}, chatOptions(log, learnable));
  expect(result.status).toBe("succeeded");
  return readRunLog(log);
}

/** The `pool_commit` envelopes of `envelopes`. */
const commits = (envelopes: readonly RecordEnvelope[]) =>
  envelopes.filter((envelope) => envelope.record instanceof PoolCommitRecord);

/** The seq of the `node_succeeded` of `nodeId`. */
function successOf(envelopes: readonly RecordEnvelope[], nodeId: string): NodeSucceededRecord {
  const found = envelopes.find(
    ({ record }) => record instanceof NodeSucceededRecord && record.nodeId === nodeId,
  );
  if (found === undefined) throw new Error(`no node_succeeded for ${nodeId}`);
  return found.record as NodeSucceededRecord;
}

/** The seq of the commit a node's success names. */
const commitSeqOf = (envelopes: readonly RecordEnvelope[], nodeId: string) => successOf(envelopes, nodeId).commit;

/** The `from` of the commit at `seq`, as an object. */
function fromAt(envelopes: readonly RecordEnvelope[], seq: number): Record<string, number> | null {
  const record = envelopes[seq]!.record as PoolCommitRecord;
  return record.from === null ? null : Object.fromEntries(record.from);
}

test("each expansion of the chat loop commits its child START before its node_expanded", async () => {
  const envelopes = await chatLog();
  for (const k of [0, 1, 2]) {
    const spawner = k === 0 ? "chat" : `chat~${k}`;
    const expanded = envelopes.findIndex(
      ({ record }) => record instanceof NodeExpandedRecord && record.spawnerId === spawner,
    );
    expect(expanded).toBeGreaterThan(0);
    const commit = envelopes[expanded - 1]!.record as PoolCommitRecord;
    expect([commit.nodeId, commit.key]).toEqual([spawner, `chat#${k}/__start__`]);
    // The first expansion's inputs are literals; each later one reads the previous turn's END.
    const previous = k === 0 ? {} : { [`chat#${k - 1}/__end__`]: commitSeqOf(envelopes, `chat#${k - 1}/__end__`) };
    expect(fromAt(envelopes, expanded - 1)).toEqual(previous);
  }
});

test("a child START's success names its expansion commit and it writes no commit of its own", async () => {
  const envelopes = await chatLog();
  for (const k of [0, 1, 2]) {
    const start = `chat#${k}/__start__`;
    const named = envelopes[commitSeqOf(envelopes, start)]!.record as PoolCommitRecord;
    expect([named.nodeId, named.key]).toEqual([k === 0 ? "chat" : `chat~${k}`, start]);
    expect(commits(envelopes).some(({ record }) => (record as PoolCommitRecord).nodeId === start)).toBe(false);
  }
});

test("a non-top END's commit holds from naming the commits its binding read", async () => {
  const envelopes = await chatLog();
  const end = commitSeqOf(envelopes, "chat#1/__end__");
  expect(fromAt(envelopes, end)).toEqual({
    "chat#1/__start__": commitSeqOf(envelopes, "chat#1/__start__"),
    "chat#1/ask": commitSeqOf(envelopes, "chat#1/ask"),
    "chat#1/reply": commitSeqOf(envelopes, "chat#1/reply"),
  });
  // The stopping LOOP driver reads the last turn's END.
  const stop = commitSeqOf(envelopes, "chat~3");
  expect(fromAt(envelopes, stop)).toEqual({ "chat#2/__end__": commitSeqOf(envelopes, "chat#2/__end__") });
});

test.each([false, true])("the run start, top-level START and END and other nodes' commits hold no from (learnable %p)", async (learnable) => {
  const envelopes = await chatLog(learnable);
  const plain = ["__start__", "__end__", "chat#0/ask", "chat#0/reply"];
  for (const { record } of commits(envelopes)) {
    const commit = record as PoolCommitRecord;
    if (commit.key.startsWith("system.") || plain.includes(commit.nodeId)) expect(commit.from).toBeNull();
  }
});

test.each([false, true])("every from seq in the log names a pool_commit (learnable %p)", async (learnable) => {
  const envelopes = await chatLog(learnable);
  const seqs = commits(envelopes).flatMap(({ record }) => [...((record as PoolCommitRecord).from ?? new Map()).values()]);
  expect(seqs.length).toBeGreaterThan(0);
  for (const seq of seqs) expect(envelopes[seq]!.record).toBeInstanceOf(PoolCommitRecord);
});

test("learnable and non-learnable runs write the same from", async () => {
  const shape = (envelopes: readonly RecordEnvelope[]) =>
    commits(envelopes).map(({ record }) => {
      const commit = record as PoolCommitRecord;
      return [commit.nodeId, commit.key, commit.from === null ? null : Object.fromEntries(commit.from)];
    });
  const plain = await chatLog(false);
  const learned = await chatLog(true);
  // Learnable runs hold the same records with the same seqs: steps and captures add no lines.
  expect(learned.length).toBe(plain.length);
  expect(shape(learned)).toEqual(shape(plain));
});

/** Run the chat with a crash after the record `cut` picks, then resume it; return both logs' envelopes. */
async function cutChat(cut: (envelopes: readonly RecordEnvelope[]) => number) {
  const whole = await chatLog();
  const at = cut(whole);
  const log = freshLog();
  const flow: ComposedFlow = await load(CHAT);
  // One script for both runs: the resume gets the answers and replies the cut run didn't use.
  const options = chatOptions(log);
  await crashAfter(at + 1, () => runFlow(flow, {}, options));
  expect(readRunLog(log).length).toBe(at + 1);
  const resumed = await resumeFlow({ log, env: NO_ENV, models: options.models, tools: options.tools });
  return { whole, resumed, envelopes: readRunLog(log) };
}

test("a cut after a LOOP's expansion commit, before its node_expanded, reruns the spawner", async () => {
  const { resumed, envelopes } = await cutChat((whole) =>
    whole.findIndex(({ record }) => record instanceof PoolCommitRecord && record.key === "chat#1/__start__"),
  );
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("\nUser: a\nAssistant: A\nUser: b\nAssistant: B\nUser: c\nAssistant: C");
  const expansions = envelopes.filter(
    ({ record }) => record instanceof PoolCommitRecord && record.key === "chat#1/__start__",
  );
  // The orphan stays, unread; the rerun spawner writes a new commit, then its node_expanded.
  expect(expansions.length).toBe(2);
  const named = commitSeqOf(envelopes, "chat#1/__start__");
  expect(named).toBe(expansions[1]!.seq);
  expect(envelopes[named + 1]!.record).toBeInstanceOf(NodeExpandedRecord);
});

test("a cut after node_expanded, before the child START starts, resumes with the START naming its expansion commit", async () => {
  const { resumed, envelopes } = await cutChat((whole) =>
    whole.findIndex(({ record }) => record instanceof NodeExpandedRecord && record.spawnerId === "chat~1"),
  );
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe("\nUser: a\nAssistant: A\nUser: b\nAssistant: B\nUser: c\nAssistant: C");
  const named = envelopes[commitSeqOf(envelopes, "chat#1/__start__")]!.record as PoolCommitRecord;
  expect([named.nodeId, named.key]).toEqual(["chat~1", "chat#1/__start__"]);
  expect(envelopes.filter(({ record }) => record instanceof NodeStartedRecord && record.nodeId === "chat~1").length).toBe(1);
});

test("a child START that evaluates to another value than its expansion commit fails the run", async () => {
  // The double changes only the evaluation (the evaluator's template run); the expansion computes
  // the committed value from the START's pure forward.
  const original = Node.prototype.runForEvaluator;
  const spy = spyOn(StartNode.prototype, "runForEvaluator").mockImplementation(async function (
    this: StartNode,
    inputs,
    caps,
    capture,
  ) {
    const result = await original.call(this, inputs, caps, capture);
    return this.id === "chat#1/__start__" ? new Output({ ...((result as Output).value as object), log: "changed" }) : result;
  });
  try {
    const log = freshLog();
    const run = runFlow(await load(CHAT), {}, chatOptions(log));
    const error = await run.then(
      () => null,
      (rejection: unknown) => rejection,
    );
    expect(error).toBeInstanceOf(RangeError);
    const envelopes = readRunLog(log);
    const seq = envelopes.findIndex(({ record }) => record instanceof PoolCommitRecord && record.key === "chat#1/__start__");
    expect((error as Error).message).toContain("chat#1/__start__");
    expect((error as Error).message).toContain(`seq ${seq}`);
    // The run wrote no success for the START.
    expect(envelopes.some(({ record }) => record instanceof NodeSucceededRecord && record.nodeId === "chat#1/__start__")).toBe(false);
  } finally {
    spy.mockRestore();
  }
});

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

/** A MAP over the run's input list, reading a system value in its body END. */
const DOUBLES = `
id: doubles
name: doubles
input: {xs: !list(int)}
defs:
  double:
    input: {x: !int}
    nodes:
      times2: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x * 2}"}
    output: {y: "\${times2.output.y}", run: "\${system.run_id}"}
nodes:
  each: {kind: map, over: "\${input.xs}", call: double, input: {x: "\${item}"}}
output: \${each.output}
`;

/** Run `text` serially into a fresh log; return the flow and the log's envelopes. */
async function plainLog(text: string, inputs: Record<string, unknown>, learnable: boolean) {
  const flow = await load(text);
  const log = freshLog();
  const result = await runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers: 0, learnable });
  expect(result.status).toBe("succeeded");
  return { flow, envelopes: readRunLog(log) };
}

/**
 * Replay `envelopes` and rebuild each derived value from its `from` through the reducer's pool
 * resolver as its record is accepted; return how many were rebuilt, after checking each equals the
 * value the log holds (the expansion's seed included).
 */
function rebuildAll(flow: ComposedFlow, envelopes: readonly RecordEnvelope[], learnable: boolean): number {
  let rebuilt = 0;
  replayRunLog(flow, MemoryRunLog.fromRecords(envelopes), {
    onPrepared: (prepared, state) => {
      expect(state.trace === null).toBe(!learnable);
      const record = prepared.record;
      if (record instanceof NodeSucceededRecord) {
        const commit = envelopes[record.commit]!.record as PoolCommitRecord;
        if (commit.from === null || commit.nodeId !== record.nodeId) return;
        const { valueJson } = rebuildCommitValue(state, record.nodeId, commit.key, commit.from);
        expect(valueJson).toBe(logJson(commit.value));
        rebuilt += 1;
      } else if (record instanceof NodeExpandedRecord) {
        const pending = [...state.pendingCommits.values()].filter((commit) => commit.spawner === record.spawnerId);
        const { seed, commits } = rebuildExpansion(state, record.spawnerId, pending[0]!.from, buildGrow);
        expect(logJson(seed)).toBe(logJson(record.seed));
        expect(commits).toEqual(pending.map((commit) => [commit.key, commit.valueJson] as const));
        rebuilt += 1;
      }
    },
  });
  return rebuilt;
}

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}],
  ["a MAP reading a system value", DOUBLES, { xs: [1n, 2n, 3n] }],
] as const)("replay rebuilds every derived value from its from, learnable or not: %s", async (_label, text, inputs) => {
  for (const learnable of [false, true]) {
    const { flow, envelopes } = await plainLog(text, inputs, learnable);
    expect(rebuildAll(flow, envelopes, learnable)).toBeGreaterThan(3);
  }
});

test("replay rebuilds every derived value of the chat, learnable or not", async () => {
  for (const learnable of [false, true]) {
    const envelopes = await chatLog(learnable);
    // Three expansions, three turn ENDs and the stopping driver.
    expect(rebuildAll(await load(CHAT), envelopes, learnable)).toBe(3 + 3 + 1);
  }
});

test("a system read points at its run-start commit", async () => {
  const { envelopes } = await plainLog(DOUBLES, { xs: [1n] }, false);
  const runId = envelopes.find(({ record }) => record instanceof PoolCommitRecord && record.key === "system.run_id")!;
  const end = envelopes[commitSeqOf(envelopes, "each#0/__end__")]!.record as PoolCommitRecord;
  expect(end.from!.get("system.run_id")).toBe(runId.seq);
});

test("a tampered from seq makes the pool resolver refuse the rebuild", async () => {
  const { flow, envelopes } = await plainLog(CALL_IN_LOOP, {}, false);
  let refused: unknown = null;
  replayRunLog(flow, MemoryRunLog.fromRecords(envelopes), {
    onPrepared: (prepared, state) => {
      const record = prepared.record;
      if (refused !== null || !(record instanceof NodeSucceededRecord) || record.nodeId !== "count#0/__end__") return;
      const commit = envelopes[record.commit]!.record as PoolCommitRecord;
      const [name, seq] = [...commit.from!][0]!;
      try {
        rebuildCommitValue(state, record.nodeId, commit.key, new Map([[name, seq + 1]]));
      } catch (error) {
        refused = error;
      }
    },
  });
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("from names");
});

test("a child START rebuilt over its constant wiring equals the value its expansion committed", async () => {
  const { envelopes } = await plainLog(CALL_IN_LOOP, {}, false);
  const flow = await load(CALL_IN_LOOP);
  const spawner = flow.compiled.nodes.get("count")!;
  const fragment = buildGrow(spawner, [{ x: 0n }, 0n]);
  const start = fragment.nodes.get("count#0/__start__")!;
  const { outcome } = rebuildDerived(start, fragment, new Map(), () => {
    throw new Error("a child START reads no source");
  }, null);
  const committed = envelopes[commitSeqOf(envelopes, "count#0/__start__")]!.record as PoolCommitRecord;
  expect(outcome).toBeInstanceOf(Output);
  expect(TypedValue.fromWire(committed.value).toObject()).toEqual((outcome as Output).value as never);
});
