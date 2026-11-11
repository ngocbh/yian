/**
 * Derived commits hold `from` pointers, and the expansion commits the child STARTs.
 *
 * A derived commit (a non-top `__end__`, the stopping LOOP driver) holds `from`: each pool name its
 * binding read, to the seq of the `pool_commit` that wrote it. Before `node_expanded`, the spawner
 * writes one `pool_commit` per START of the fragment (`node_id` the spawner, `key` the child START),
 * whose `from` is the spawner's; the child START's success names that commit and it writes none of
 * its own. Replay rebuilds a derived value from the pool and checks each named seq against the
 * producer index.
 */

import { afterAll, beforeAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildGrow } from "../../src/compile/expand.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { checkConversation, type Message } from "../../src/nodes/agent/messages.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Grow, Node, Output } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { NodeExpanded, NodeSucceeded, type RunEvent } from "../../src/events.ts";
import { foldRunLog, replayRunLog, advance, runFlow } from "../../src/run.ts";
import { rebuildDerived } from "../../src/runtime/eval_node.ts";
import {
  canonicalSeed,
  decodeRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  readRunLog,
  type RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { rebuildCommit, rebuildGrowth } from "../../src/runtime/state.ts";
import { decodeTyped } from "../../src/runtime/sources.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { crashAfter, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-derived-records-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's temp folder. */
const freshLog = () => join(tmp, `log-${++logs}.jsonl`);

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A chat loop of three scripted turns: each turn asks, replies, and appends in its END binding. */
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
      ask: {kind: tool, tool_id: ask, input: {prompt: "You:"}, output: str}
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
/** The chat, loaded once for the tests that only run it. */
let CHAT_FLOW: ComposedFlow | null = null;
beforeAll(async () => {
  CHAT_FLOW = await load(CHAT);
});

/**
 * The run options for the chat: scripted answers and replies, each observer going to `events`.
 */
function chatOptions(log: string, events: RunEvent[] = []) {
  const answers = ["a", "b", "c"];
  return {
    onEvent: (event: RunEvent) => void events.push(event),
    env: NO_ENV,
    log,
    runId: "r1",
    numWorkers: 0,
    llmConfig: CONFIG,
    models: serving(new ScriptedModel([textReply("A"), textReply("B"), textReply("C")], CONFIG)),
    tools: { ask: () => answers.shift()! },
  };
}

/** Run the chat into a fresh log, its observers going to `events`, and return the log's envelopes. */
async function chatLog(events: RunEvent[] = []): Promise<readonly RecordEnvelope[]> {
  const log = freshLog();
  const result = await runFlow(await load(CHAT), {}, chatOptions(log, events));
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

test("the run start, top-level START and END and other nodes' commits hold no from", async () => {
  const envelopes = await chatLog();
  const plain = ["__start__", "__end__", "chat#0/ask", "chat#0/reply"];
  for (const { record } of commits(envelopes)) {
    const commit = record as PoolCommitRecord;
    if (commit.key.startsWith("system.") || plain.includes(commit.nodeId)) expect(commit.from).toBeNull();
  }
});

test("every from seq in the log names a pool_commit", async () => {
  const envelopes = await chatLog();
  const seqs = commits(envelopes).flatMap(({ record }) => [...((record as PoolCommitRecord).from ?? new Map()).values()]);
  expect(seqs.length).toBeGreaterThan(0);
  for (const seq of seqs) expect(envelopes[seq]!.record).toBeInstanceOf(PoolCommitRecord);
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
  const resumed = await advance({ log, env: NO_ENV, models: options.models, tools: options.tools });
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
  ) {
    const result = await original.call(this, inputs, caps);
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

/** Run `text` serially into a fresh log; return the flow, the log's envelopes and the run's observers. */
async function plainLog(text: string, inputs: Record<string, unknown>) {
  const flow = await load(text);
  const log = freshLog();
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => void events.push(event);
  const result = await runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers: 0, onEvent });
  expect(result.status).toBe("succeeded");
  return { flow, envelopes: readRunLog(log), events };
}

/**
 * Replay `envelopes` and rebuild each derived value from its `from` through the reducer's pool
 * resolver as its record is accepted; return how many were rebuilt, after checking each equals the
 * value the live run published (`events`): a commit's value, an expansion's seed and its child
 * STARTs' values.
 */
function rebuildAll(
  flow: ComposedFlow,
  envelopes: readonly RecordEnvelope[],
  events: readonly RunEvent[],
): number {
  const outputs = new Map<string, unknown>();
  const seeds = new Map<string, unknown>();
  for (const event of events) {
    if (event instanceof NodeSucceeded) outputs.set(event.nodeId, event.output);
    if (event instanceof NodeExpanded) seeds.set(event.nodeId, event.grow.seed);
  }
  let rebuilt = 0;
  replayRunLog(flow, MemoryRunLog.fromRecords(envelopes), {
    onPrepared: (prepared, state) => {
      const record = prepared.record;
      if (record instanceof NodeSucceededRecord) {
        const commit = envelopes[record.commit]!.record as PoolCommitRecord;
        if (commit.from === null) return;
        if (commit.nodeId !== record.nodeId) {
          // A child START: its expansion's admitted commit holds the value its START binds.
          const valueJson = state.pendingCommits.get(record.nodeId)!.valueJson!;
          expect([record.nodeId, decodeTyped(valueJson).toObject()]).toEqual([record.nodeId, outputs.get(record.nodeId) as never]);
          return;
        }
        const valueJson = rebuildCommit(state, record.nodeId, commit.key, commit.from);
        expect([record.nodeId, decodeTyped(valueJson).toObject()]).toEqual([record.nodeId, outputs.get(record.nodeId) as never]);
        rebuilt += 1;
      } else if (record instanceof NodeExpandedRecord) {
        const pending = [...state.pendingCommits.values()].filter((commit) => commit.spawner === record.spawnerId);
        const { seed } = rebuildGrowth(state, record.spawnerId, pending[0]!.from);
        expect([record.spawnerId, canonicalSeed(seed)[1]]).toEqual([record.spawnerId, seeds.get(record.spawnerId) as never]);
        rebuilt += 1;
      }
    },
  });
  return rebuilt;
}

test.each([
  ["a CALL in a loop", CALL_IN_LOOP, {}],
  ["a MAP reading a system value", DOUBLES, { xs: [1n, 2n, 3n] }],
] as const)("replay rebuilds every derived value from its from: %s", async (_label, text, inputs) => {
  const { flow, envelopes, events } = await plainLog(text, inputs);
  expect(rebuildAll(flow, envelopes, events)).toBeGreaterThan(3);
});

test("replay rebuilds every derived value of the chat", async () => {
  const events: RunEvent[] = [];
  const envelopes = await chatLog(events);
  // Three expansions, three turn ENDs and the stopping driver.
  expect(rebuildAll(await load(CHAT), envelopes, events)).toBe(3 + 3 + 1);
});

test("a system read points at its run-start commit", async () => {
  const { envelopes } = await plainLog(DOUBLES, { xs: [1n] });
  const runId = envelopes.find(({ record }) => record instanceof PoolCommitRecord && record.key === "system.run_id")!;
  const end = envelopes[commitSeqOf(envelopes, "each#0/__end__")]!.record as PoolCommitRecord;
  expect(end.from!.get("system.run_id")).toBe(runId.seq);
});

test("a tampered from seq makes the pool resolver refuse the rebuild", async () => {
  const { flow, envelopes } = await plainLog(CALL_IN_LOOP, {});
  let refused: unknown = null;
  replayRunLog(flow, MemoryRunLog.fromRecords(envelopes), {
    onPrepared: (prepared, state) => {
      const record = prepared.record;
      if (refused !== null || !(record instanceof NodeSucceededRecord) || record.nodeId !== "count#0/__end__") return;
      const commit = envelopes[record.commit]!.record as PoolCommitRecord;
      const [name, seq] = [...commit.from!][0]!;
      try {
        rebuildCommit(state, record.nodeId, commit.key, new Map([[name, seq + 1]]));
      } catch (error) {
        refused = error;
      }
    },
  });
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("from names");
});

test("a child START rebuilt over its constant wiring equals the value its expansion committed", async () => {
  const { flow, events } = await plainLog(CALL_IN_LOOP, {});
  const spawner = flow.compiled.nodes.get("count")!;
  const fragment = buildGrow(spawner, [{ x: 0n }, 0n]);
  const start = fragment.nodes.get("count#0/__start__")!;
  const outcome = rebuildDerived(start, fragment, new Map(), () => {
    throw new Error("a child START reads no source");
  }, null);
  const committed = events.find((event) => event instanceof NodeSucceeded && event.nodeId === "count#0/__start__");
  expect(outcome).toBeInstanceOf(Output);
  expect((outcome as Output).value).toEqual((committed as NodeSucceeded).output);
});

test("a derived commit holds from and no kind or value; node_expanded holds no seed", async () => {
  const log = freshLog();
  const result = await runFlow(await load(CHAT), {}, chatOptions(log));
  expect(result.status).toBe("succeeded");
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  const payloads = lines.map((line) => parseJson(line, "codec") as { kind: string; payload: Record<string, unknown> });
  let derived = 0;
  for (const { kind, payload } of payloads) {
    if (kind === "node_expanded") expect(Object.hasOwn(payload, "seed")).toBe(false);
    if (kind !== "pool_commit") continue;
    if (Object.hasOwn(payload, "steps")) {
      expect(Object.keys(payload).sort()).toEqual(["key", "node_id", "steps"]);
    } else if (Object.hasOwn(payload, "from")) {
      derived += 1;
      expect(Object.keys(payload).sort()).toEqual(["from", "key", "node_id"]);
    } else {
      expect(Object.keys(payload).sort()).toEqual(["key", "kind", "node_id", "value"]);
    }
  }
  // Three expansions, three turn ENDs and the stopping driver.
  expect(derived).toBe(7);
});

/** The chat's log lines, each payload changed by `edit` at the first record `pick` selects. */
async function editedChat(
  pick: (kind: string, payload: Record<string, unknown>) => boolean,
  edit: (payload: Record<string, unknown>) => void,
): Promise<string> {
  const log = freshLog();
  await runFlow(await load(CHAT), {}, chatOptions(log));
  const records = readFileSync(log, "utf8").split("\n").filter((line) => line !== "").map((line) => parseJson(line, "codec") as Record<string, any>);
  const target = records.find((record) => pick(record["kind"], record["payload"]))!;
  edit(target["payload"]);
  const edited = freshLog();
  writeFileSync(edited, records.map((record) => `${logJson(record)}\n`).join(""));
  return edited;
}

test.each([
  [
    "a derived node's commit holding a value and no from",
    (kind: string, payload: Record<string, unknown>) => kind === "pool_commit" && payload["node_id"] === "chat#0/__end__",
    (payload: Record<string, unknown>) => {
      delete payload["from"];
      payload["kind"] = "object";
      payload["value"] = { history: [], log: "" };
    },
    "pool_commit of 'chat#0/__end__' must hold from",
  ],
  [
    "another node's commit holding from",
    (kind: string, payload: Record<string, unknown>) => kind === "pool_commit" && payload["node_id"] === "chat#0/ask",
    (payload: Record<string, unknown>) => {
      delete payload["kind"];
      delete payload["value"];
      payload["from"] = {};
    },
    "pool_commit of 'chat#0/ask' holds from",
  ],
])("replay refuses %s", async (_label, pick, edit, message) => {
  const log = await editedChat(pick, edit);
  const flow = await load(CHAT);
  let error: unknown = null;
  try {
    replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain(message);
});

test("decoding refuses a pool_commit with both forms and a node_expanded with a seed", () => {
  const commit =
    `{"at":"2026-01-01T00:00:00.000Z","kind":"pool_commit","payload":{"from":{},"key":"e","kind":"int","node_id":"e","value":1},` +
    `"run_id":"r1","schema":${SCHEMA_VERSION},"seq":9}`;
  expect(() => decodeRecord(commit)).toThrow(RangeError);
  const expanded =
    `{"at":"2026-01-01T00:00:00.000Z","kind":"node_expanded","payload":{"prune_ids":[],` +
    `"seed":{},"spawner_id":"s","topology_digest":"${"a".repeat(64)}"},"run_id":"r1","schema":${SCHEMA_VERSION},"seq":9}`;
  expect(() => decodeRecord(expanded)).toThrow('extra=["seed"]');
});

test("foldRunLog yields each accepted commit's value, rebuilt for a derived commit, as the log with values held it", async () => {
  // The same chat run, written when derived commits also held their values (one line each) and
  // before an agent's success wrote a messages commit, so commits are compared in order, not by seq.
  const held = readFileSync(fileURLToPath(new URL("../fixtures/derived/chat_commits_both_forms.jsonl", import.meta.url)), "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => (parseJson(line, "codec") as Record<string, any>)["payload"] as Record<string, any>);
  const log = freshLog();
  await runFlow(await load(CHAT), {}, chatOptions(log));
  const folded = await foldRunLog(log, { env: NO_ENV });
  let derived = 0;
  let messages = 0;
  let index = 0;
  for (const { entry, value } of folded.records) {
    const record = entry.envelope.record;
    if (!(record instanceof PoolCommitRecord)) {
      expect(value).toBeNull();
      continue;
    }
    if (record.steps !== null) {
      messages += 1;
      checkConversation(value!.toObject() as Message[]);
      continue;
    }
    const payload = held[index++]!;
    expect([index, payload["key"]]).toEqual([index, record.key]);
    if (record.key === "system.now" || record.key === "system.today") continue;
    if (record.from !== null) derived += 1;
    expect(value).not.toBeNull();
    expect([index, logJson(value!.toWire())]).toEqual([index, logJson({ kind: payload["kind"], value: payload["value"] })]);
  }
  expect(index).toBe(held.length);
  expect(derived).toBe(7);
  expect(messages).toBe(3);
});

test("foldRunLog reads an inner LOOP __end__ value from a finished log", async () => {
  const log = freshLog();
  await runFlow(await load(CHAT), {}, chatOptions(log));
  const folded = await foldRunLog(log, { env: NO_ENV });
  const end = folded.records.find(
    ({ entry }) => entry.envelope.record instanceof PoolCommitRecord && entry.envelope.record.key === "chat#1/__end__",
  )!;
  expect((end.entry.envelope.record as PoolCommitRecord).from).not.toBeNull();
  expect(end.value!.toObject()).toEqual({
    history: [
      { role: "user", text: "a" },
      { role: "assistant", text: "A" },
      { role: "user", text: "b" },
      { role: "assistant", text: "B" },
    ],
    log: "\nUser: a\nAssistant: A\nUser: b\nAssistant: B",
  });
  expect(folded.flow.compiled.nodes.has("chat")).toBe(true);
});

test("foldRunLog yields null for a commit no success accepts", async () => {
  const whole = await chatLog();
  const at = commitSeqOf(whole, "chat#1/__end__");
  const log = freshLog();
  await crashAfter(at + 1, () => runFlow(CHAT_FLOW!, {}, chatOptions(log)));
  const folded = await foldRunLog(log, { env: NO_ENV });
  const pending = folded.records[at]!;
  expect(pending.entry.envelope.record).toBeInstanceOf(PoolCommitRecord);
  expect(pending.value).toBeNull();
});

test("a derived node whose value doesn't rebuild from its from fails the run at its success", async () => {
  // The double changes only the evaluation (the evaluator's template run), as a binding that read
  // a name its `from` missed would; the success rebuilds the value from `from` with `forward`.
  const original = Node.prototype.runForEvaluator;
  const spy = spyOn(EndNode.prototype, "runForEvaluator").mockImplementation(async function (
    this: EndNode,
    inputs,
    caps,
  ) {
    const result = await original.call(this, inputs, caps);
    return this.id === "chat#0/__end__" ? new Output({ ...((result as Output).value as object), log: "changed" }) : result;
  });
  try {
    const log = freshLog();
    const error = await runFlow(await load(CHAT), {}, chatOptions(log)).then(
      () => null,
      (rejection: unknown) => rejection,
    );
    expect(error).toBeInstanceOf(RangeError);
    const envelopes = readRunLog(log);
    const seq = envelopes.findIndex(({ record }) => record instanceof PoolCommitRecord && record.nodeId === "chat#0/__end__");
    expect((error as Error).message).toContain("chat#0/__end__");
    expect((error as Error).message).toContain(`seq ${seq}`);
    expect(envelopes.some(({ record }) => record instanceof NodeSucceededRecord && record.nodeId === "chat#0/__end__")).toBe(false);
  } finally {
    spy.mockRestore();
  }
});

test("a spawner whose growth doesn't rebuild from its from fails the run at its expansion", async () => {
  // As for the END double: only the evaluation changes; the expansion rebuilds the seed from the
  // expansion commits' `from` with the spawner's `forward`.
  const original = Node.prototype.runForEvaluator;
  const spy = spyOn(CallNode.prototype, "runForEvaluator").mockImplementation(async function (
    this: CallNode,
    inputs,
    caps,
  ) {
    const result = await original.call(this, inputs, caps);
    return this.id === "count#0/step" ? new Grow({ x: 99n }) : result;
  });
  try {
    const log = freshLog();
    const error = await runFlow(await load(CALL_IN_LOOP), {}, { env: NO_ENV, log, runId: "r1", numWorkers: 0 }).then(
      () => null,
      (rejection: unknown) => rejection,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("count#0/step");
    const envelopes = readRunLog(log);
    expect(envelopes.some(({ record }) => record instanceof NodeExpandedRecord && record.spawnerId === "count#0/step")).toBe(false);
  } finally {
    spy.mockRestore();
  }
});

test("a derived child END reads a declared field that is absent as null", async () => {
  const text = `
id: opt
name: opt
typedefs:
  Rec: {k: str, opt: Optional(str)}
defs:
  child:
    nodes:
      a:
        kind: code
        output: Rec
        code: |
          return {k: "x"}
    output: \${a.output.opt :- "d"}
nodes:
  c: {kind: call, call: child}
output: \${c.output}
`;
  const result = await runFlow(await load(text), {}, { log: freshLog(), runId: "r1", env: NO_ENV });
  expect(result.failure?.message).toBeUndefined();
  expect(result.output).toBe("d");
});
