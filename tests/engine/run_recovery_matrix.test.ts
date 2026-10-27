/**
 * Recovery equivalence across nested growth and unfinished live effects.
 *
 * The live state is captured after every committed record, each prefix is replayed through
 * `run.ts`'s `replayRunLog`, and the complete reducer state and topology are compared. The effect
 * rows stop the run right after the effect's durable `NodeStarted`, so no model, human interaction
 * or wait runs. CODE references resolve from the conformance fixtures.
 */

import { expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { buildGrow } from "../../src/compile/expand.ts";
import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted, type RunEvent, RunSucceeded } from "../../src/events.ts";
import { ValueRoute } from "../../src/expr/lineage.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { DeclSite, MemoryScope, MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../../src/memory/index.ts";
import { type Node, NodeKind, Pending } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";
import { replayRunLog, advance, runFlow } from "../../src/run.ts";
import { ForwardCapture } from "../../src/runtime/contracts.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import {
  decodeRecord,
  JsonlRunLog,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStartedRecord,
  type RecordEnvelope,
  type RunRecord,
  RunResumedRecord,
  type RunStartedRecord,
} from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  PreparedTransition,
  prepareNodeStarted,
  prepareRunResumed,
  prepareRunStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

import { hostEnv } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import {
  expectUnchanged,
  FuncNode,
  NO_ENV,
  prepareOutput,
  reducerSnapshot,
  replayTraceSnapshot,
  snapshot,
  startedRun,
  untimed,
} from "./run_state_fixtures.ts";

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));
const RUN_ID = "recovery-matrix";
const SYSTEM_TODAY = "2026-08-14";
const SYSTEM_NOW = "2026-08-14T12:00:00+00:00";
const WAIT_DEADLINE = "2026-08-14T12:00:10+00:00";

const NESTED_GROWTH = `
id: recovery-matrix-growth
name: recovery_matrix_growth
defs:
  echo_one:
    input:
      value: !int
    nodes:
      echo:
        kind: code
        input:
          x: \${input.value}
        output: int
        code: ./compose_codefns.ts:echo_x
    output: \${echo.output}
  bump_once:
    input:
      n: !int
      exited: !bool
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
        code: ./compose_codefns.ts:loop_bump
    output: \${bump.output}
  composite:
    nodes:
      mapped:
        kind: map
        call: echo_one
        over: \${[1, 2]}
        input:
          value: \${item}
      repeated:
        kind: loop
        call: bump_once
        input:
          n: 0
          exited: false
        times: 2
    output:
      mapped: \${mapped.output}
      repeated: \${repeated.output}
nodes:
  outer:
    kind: call
    call: composite
output: \${outer.output}
`;

const NESTED_INLINE_GROWTH = `
id: recovery-matrix-inline-growth
name: recovery_matrix_inline_growth
nodes:
  outer:
    kind: call
    input:
      items: [1, 2]
    nodes:
      mapped:
        kind: map
        over: \${input.items}
        input:
          value: \${item}
        nodes:
          echo:
            kind: code
            input:
              x: \${input.value}
            output: int
            code: ./compose_codefns.ts:echo_x
        output: \${echo.output}
      repeated:
        kind: loop
        input:
          n: 0
          exited: false
        times: 2
        nodes:
          bump:
            kind: code
            input:
              n: \${input.n}
              exited: \${input.exited}
            output:
              n: int
              exited: bool
            code: ./compose_codefns.ts:loop_bump
        output: \${bump.output}
    output:
      mapped: \${mapped.output}
      repeated: \${repeated.output}
output: \${outer.output}
`;

const NESTED_HUMAN_CALL = `
id: recovery-nested-human-call
name: recovery_nested_human_call
defs:
  gate:
    input:
      token: !str
    nodes:
      ask:
        kind: tool
        tool_id: ask
        input:
          prompt: nested call input
        output: str
    output: \${ask.output}
nodes:
  invoke:
    kind: call
    call: gate
    input:
      token: ready
output: \${invoke.output}
`;

const LOOP_BODY_SPAWNERS = `
id: recovery-loop-body-spawners
name: recovery_loop_body_spawners
defs:
  echo_one:
    input:
      value: !int
    nodes:
      echo:
        kind: code
        input:
          x: \${input.value}
        output: int
        code: ./compose_codefns.ts:echo_x
    output: \${echo.output}
  turn:
    input:
      n: !int
      exited: !bool
    nodes:
      nested_call:
        kind: call
        call: echo_one
        input:
          value: \${input.n}
      nested_map:
        kind: map
        call: echo_one
        over: \${[1, 2]}
        input:
          value: \${item}
      bump:
        kind: code
        depends_on: [nested_call, nested_map]
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
        code: ./compose_codefns.ts:loop_bump
    output: \${bump.output}
nodes:
  repeated:
    kind: loop
    call: turn
    input:
      n: 0
      exited: false
    times: 3
output: \${repeated.output}
`;

const LOOP_BODY_NESTED_LOOP_AND_CASE = `
id: recovery-loop-body-nested-loop-case
name: recovery_loop_body_nested_loop_case
defs:
  echo_one:
    input:
      value: !int
    nodes:
      echo:
        kind: code
        input:
          x: \${input.value}
        output: int
        code: ./compose_codefns.ts:echo_x
    output: \${echo.output}
  inc:
    input:
      n: !int
    nodes:
      c:
        kind: call
        call: echo_one
        input:
          value: \${input.n}
    output:
      n: \${c.output}
  turn:
    input:
      n: !int
      exited: !bool
    nodes:
      inner:
        kind: loop
        call: inc
        input:
          n: \${input.n}
        times: 2
      pick:
        kind: case
        cases:
          - when: "input.n % 2 == 0"
            then: call(echo_one, value=\${input.n})
        else: call(echo_one, value=\${input.n})
      bump:
        kind: code
        depends_on: [inner]
        input:
          n: \${pick.output}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
        code: ./compose_codefns.ts:loop_bump
    output: \${bump.output}
nodes:
  repeated:
    kind: loop
    call: turn
    input:
      n: 0
      exited: false
    times: 3
output: \${repeated.output}
`;

// The body returns only `{n}`; the omitted `tag` merges in at every LOOP respawn.
const PARTIAL_LOOP = `
id: recovery-partial-loop
name: recovery_partial_loop
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: |
          return {"n": inputs["n"] + 1}
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  repeated:
    kind: loop
    call: bump
    input:
      n: 0
      tag: kept
    times: 3
output: \${repeated.output}
`;

const EFFECT_FLOWS: Record<string, string> = {
  ask: `
id: recovery-human
name: recovery_human
nodes:
  effect:
    kind: tool
    tool_id: ask
    input:
      prompt: this host interaction must never be invoked
    output: str
output: \${effect.output}
`,
  sleep: `
id: recovery-wait
name: recovery_wait
nodes:
  effect:
    kind: tool
    tool_id: sleep
    input:
      until: '${WAIT_DEADLINE}'
output: \${effect.output}
`,
};

const ORDERED_EFFECTS = `
id: recovery-ordered-effects
name: recovery_ordered_effects
nodes:
  release:
    kind: code
    input: {x: ready}
    output: str
    code: ./compose_codefns.ts:echo_x
  wait:
    kind: tool
    tool_id: sleep
    depends_on: [release]
    input:
      until: '${WAIT_DEADLINE}'
  agent:
    kind: agent
    depends_on: [release]
    prompt: never invoked
    llm_config: {provider: openai, model: offline-test}
  human:
    kind: tool
    tool_id: ask
    depends_on: [release]
    input:
      prompt: never invoked
    output: str
  later:
    kind: code
    depends_on: [release]
    input: {x: still-ready}
    output: str
    code: ./compose_codefns.ts:echo_x
output:
  agent: \${agent.output}
  human: \${human.output}
  wait: \${wait.output}
  later: \${later.output}
`;

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** Every topology and reducer coordinate of `state` as detached values. */
function matrixSnapshot(state: RunState): Record<string, unknown> {
  const flow = state.flow;
  const ids = [...flow.nodes.keys()];
  return {
    ...reducerSnapshot(state),
    topology: flowManifest(flow),
    originContext: [...flow.nodes].map(([nodeId, node]) => [nodeId, node.originContext]),
    nodeOrder: ids,
    edges: flow.edges.map((edge) => [
      edge.id,
      edge.from,
      edge.to,
      edge.sourceHandle,
      edge.inputGroup,
      edge.optional,
      edge.ordering,
    ]),
    wiring: [...flow.wiring].map(([nodeId, wiring]) => [nodeId, [...wiring]]),
    incoming: ids.map((nodeId) => [nodeId, flow.incoming(nodeId).map((edge) => edge.id)]),
    outgoing: ids.map((nodeId) => [nodeId, flow.outgoing(nodeId).map((edge) => edge.id)]),
    initialStaticIds: [...state.initialStaticIds],
  };
}

/**
 * A memory log that snapshots `state`, once set, before each append (the state after the
 * previous commit), and that refuses every append once `frozen`.
 */
class CapturingLog extends MemoryRunLog {
  state: RunState | null = null;
  readonly snapshots: Record<string, unknown>[] = [];
  frozen = false;

  override append(record: RunRecord): RecordEnvelope {
    if (this.frozen) throw new Error("the run was paused");
    if (this.state !== null) this.snapshots.push(matrixSnapshot(this.state));
    return super.append(record);
  }
}

/** Create one private explicitly configured run of `loaded` through its first record. */
function newStartedRun(loaded: ComposedFlow, numWorkers: number): { state: RunState; log: CapturingLog } {
  const privateFlow = copyGraph(loaded.compiled);
  const effective = resolveLlmCascade(privateFlow, {}, { defaults: { provider: "openai", model: "offline-test" } });
  const log = new CapturingLog(RUN_ID);
  return { state: startedRun(loaded, log, { numWorkers, effective, privateFlow }), log };
}

/** Run `loaded` live to its end; return the state, its log and the snapshot after each record. */
async function liveRun(
  loaded: ComposedFlow,
  numWorkers: number,
): Promise<{ state: RunState; log: CapturingLog; observers: readonly RunEvent[] }> {
  const { state, log } = newStartedRun(loaded, numWorkers);
  log.state = state;
  const observers = await new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers }).run();
  log.snapshots.push(matrixSnapshot(state));
  return { state, log, observers };
}

/**
 * Expect every prefix of `log` past its run start to replay to the live snapshot after it. The run
 * start (RunStarted and its commits) is written before the first snapshot, and a prefix inside it
 * can't be replayed.
 */
function expectEveryPrefixReplays(loaded: ComposedFlow, log: CapturingLog): void {
  const envelopes = log.records();
  const started = envelopes.length - log.snapshots.length + 1;
  expect(envelopes[started - 1]!.kind).toBe("pool_commit");
  expect(envelopes[started]!.kind).toBe("node_started");
  log.snapshots.forEach((expected, index) => {
    const recovered = replayRunLog(loaded, MemoryRunLog.fromRecords(envelopes.slice(0, started + index)));
    expect(matrixSnapshot(recovered), `prefix ${started + index}/${envelopes.length}`).toEqual(expected);
  });
}

/** A tool set, model access and scheduler that fail the test if an effect runs. */
const NO_EFFECTS = {
  tools: new ToolSet({
    ask: () => {
      throw new Error("effect work ran after only its durable NodeStarted");
    },
  }),
  models: {
    env: NO_ENV,
    createModel: () => {
      throw new Error("effect work ran after only its durable NodeStarted");
    },
  } satisfies ModelAccess,
  scheduler: {
    now: () => Date.parse(SYSTEM_NOW),
    sleep: () => {
      throw new Error("effect work ran after only its durable NodeStarted");
    },
  } satisfies Scheduler,
};

/**
 * Run `loaded` serially until the first durable `NodeStarted` that `isTarget` accepts, then
 * stop with nothing more recorded; return the live state and its log.
 */
async function pauseAfterStarted(
  loaded: ComposedFlow,
  isTarget: (state: RunState, nodeId: string) => boolean,
): Promise<{ state: RunState; log: CapturingLog; target: string }> {
  const { state, log } = newStartedRun(loaded, 0);
  const pause = new Error("paused after the durable start");
  let target: string | null = null;
  const onEvent = (event: RunEvent) => {
    if (event instanceof NodeStarted && isTarget(state, event.nodeId)) {
      target = event.nodeId;
      log.frozen = true;
      throw pause;
    }
  };
  let caught: unknown;
  try {
    await new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers: 0, onEvent, ...NO_EFFECTS }).run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBe(pause);
  expect(target).not.toBeNull();
  expect(log.records().at(-1)!.record).toEqual(new NodeStartedRecord(target!));
  return { state, log, target: target! };
}

for (const [sourceId, source] of [
  ["named", NESTED_GROWTH],
  ["inline", NESTED_INLINE_GROWTH],
] as const) {
  for (const [workersId, numWorkers] of [
    ["serial", 0],
    ["pooled", 2],
  ] as const) {
    test(`test_every_nested_growth_prefix_replays_to_the_live_state[${workersId}-${sourceId}]`, async () => {
      const loaded = await load(source);
      const { state, log, observers } = await liveRun(loaded, numWorkers);

      expect(state.result).toBeInstanceOf(RunSucceeded);
      expect((state.result as RunSucceeded).output).toEqual({ mapped: [1n, 2n], repeated: { n: 2n, exited: false } });
      expect(observers.at(-1)).toEqual(state.result!);
      expectEveryPrefixReplays(loaded, log);
    });
  }
}

const CALL_MAP = [LOOP_BODY_SPAWNERS, { n: 3n, exited: true }, ["/nested_call", "/nested_map"]] as const;
const LOOP_CASE = [LOOP_BODY_NESTED_LOOP_AND_CASE, { n: 3n, exited: true }, ["/inner", "/c", "/__call_0"]] as const;

// Every prefix replays from scratch, so the long loop-case log runs serially only; the call-map
// case covers pooled replay.
for (const [id, [source, expectedOutput, spawnerSuffixes], numWorkers] of [
  ["call-map-serial", CALL_MAP, 0],
  ["call-map-pooled", CALL_MAP, 2],
  ["loop-case-serial", LOOP_CASE, 0],
] as const) {
  test(`test_every_loop_body_spawner_prefix_replays_to_the_live_state[${id}]`, async () => {
    const loaded = await load(source);
    const { state, log, observers } = await liveRun(loaded, numWorkers);

    expect(observers.at(-1)).toBeInstanceOf(RunSucceeded);
    expect((observers.at(-1) as RunSucceeded).output).toEqual(expectedOutput);
    const expandedIds = log
      .records()
      .map((envelope) => envelope.record)
      .filter((record) => record instanceof NodeExpandedRecord)
      .map((record) => (record as NodeExpandedRecord).spawnerId);
    for (const suffix of spawnerSuffixes) expect(expandedIds.some((nodeId) => nodeId.endsWith(suffix))).toBe(true);
    expectEveryPrefixReplays(loaded, log);
    // Only the last turn's body stays live at the end of the run.
    const ids = [...state.flow.nodes.keys()];
    expect(ids.filter((nodeId) => nodeId.startsWith("repeated#0/") || nodeId.startsWith("repeated#1/"))).toEqual([]);
    expect(ids.some((nodeId) => nodeId.startsWith("repeated#2/"))).toBe(true);
  });
}

for (const [id, numWorkers] of [
  ["serial", 0],
  ["pooled", 2],
] as const) {
  test(`test_every_partial_loop_prefix_replays_to_the_live_state[${id}]`, async () => {
    const loaded = await load(PARTIAL_LOOP);
    const { log, observers } = await liveRun(loaded, numWorkers);

    expect(observers.at(-1)).toBeInstanceOf(RunSucceeded);
    expect((observers.at(-1) as RunSucceeded).output).toEqual({ n: 3n, tag: "kept" });
    expectEveryPrefixReplays(loaded, log);
  });
}

for (const [id, source] of Object.entries(EFFECT_FLOWS)) {
  test(`test_effect_start_prefix_replays_without_running_effect_work[${id}]`, async () => {
    const loaded = await load(source);
    const { state: live, log } = await pauseAfterStarted(loaded, (_state, nodeId) => nodeId === "effect");

    const recovered = replayRunLog(loaded, log);

    expect(matrixSnapshot(recovered)).toEqual(matrixSnapshot(live));
    expect([...recovered.inFlight]).toEqual(["effect"]);
    expect(recovered.ready).toEqual([]);
    const prepared = prepareRunResumed(recovered, { numWorkers: 0 });
    expect(prepared.record).toEqual(new RunResumedRecord(["effect"], 0));
    log.frozen = false;
    commitPrepared(log, recovered, prepared);
    expect([...recovered.inFlight]).toEqual([]);
    expect(recovered.ready).toEqual(["effect"]);
  });
}

test("test_namespaced_human_input_start_replays_without_host_interaction[call]", async () => {
  const loaded = await load(NESTED_HUMAN_CALL);
  const { state, log, target } = await pauseAfterStarted(
    loaded,
    (live, nodeId) => live.flow.nodes.get(nodeId)!.kind === NodeKind.TOOL,
  );

  expect(target).toContain("/ask");
  expect([...state.inFlight]).toEqual([target]);

  const recovered = replayRunLog(loaded, log);
  expect(matrixSnapshot(recovered)).toEqual(matrixSnapshot(state));
  expect([...recovered.inFlight]).toEqual([target]);
  expect(prepareRunResumed(recovered, { numWorkers: 0 }).record).toEqual(new RunResumedRecord([target], 0));
});

test("recovered wait recomputes from the authored absolute deadline", async () => {
  const loaded = await load(EFFECT_FLOWS["sleep"]!);
  const { log } = await pauseAfterStarted(loaded, (_state, nodeId) => nodeId === "effect");
  const recovered = replayRunLog(loaded, log);
  const authored = loaded.compiled.wiring.get("effect")!.get("until") as Binding;
  const replayed = recovered.flow.wiring.get("effect")!.get("until") as Binding;
  expect(replayed).toEqual(authored);
  expect(replayed.source).toBe(WAIT_DEADLINE);
  expect(replayed.origin).not.toBeNull();

  log.frozen = false;
  commitPrepared(log, recovered, prepareRunResumed(recovered, { numWorkers: 0 }));
  const node = recovered.flow.nodes.get("effect")!;

  // Whatever the clock reads, the sleep asks for the authored deadline in UTC; the engine decides.
  for (const second of [4, 7]) {
    const scheduler = virtualScheduler(Date.UTC(2026, 7, 14, 12, 0, second));
    const { outcome } = await evalNode(node, recovered.flow, recovered.pool, { scheduler });
    expect(outcome).toEqual(
      new Pending([], { wait: { tool: "sleep", until: "2026-08-14T12:00:10.000Z", forSeconds: null } }),
    );
  }
});

test("recorded nested growth resumes with exact memory and trace", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-recovery-matrix-"));
  const env = hostEnv(dir);
  try {
    const source = NESTED_GROWTH.replace(
      "name: recovery_matrix_growth\n",
      "name: recovery_matrix_growth\nlong_term_memory: episodic\n",
    ).replace("  composite:\n    nodes:", "  composite:\n    long_term_memory: reflective\n    nodes:");
    const flow = await load(source);
    const path = join(dir, "historical-growth.jsonl");
    const memory = () => new Map([["episodic", new FakeMemory()], ["reflective", new FakeMemory()]]);
    const fresh = await runFlow(flow, {}, {
      log: path,
      runId: "historical-growth",
      learnable: true,
      numWorkers: 0,
      memory: memory(),
      env,
    });
    expect(fresh.status).toBe("succeeded");
    // Through outer/mapped#1/echo's NodeSucceeded.
    const all = readFileSync(path, "utf8").split("\n").slice(0, -1);
    const length =
      all.findIndex((line) => {
        const { kind, payload } = decodeRecord(line);
        return kind === "node_succeeded" && payload["node_id"] === "outer/mapped#1/echo";
      }) + 1;
    const lines = all.slice(0, length);
    expect(decodeRecord(lines.at(-1)!).payload["node_id"]).toBe("outer/mapped#1/echo");
    const prefix = `${lines.join("\n")}\n`;
    writeFileSync(path, prefix);
    const initial = decodeRecord(lines[0]!).record as RunStartedRecord;
    const snapshots = new Map(initial.memorySnapshots!.map((item) => [item.memory, item.snapshot]));
    expect([...snapshots.keys()].sort()).toEqual(["episodic", "reflective"]);
    const backends = memory();
    const calls: string[] = [];
    const original = CodeNode.prototype.forward;
    const forward = spyOn(CodeNode.prototype, "forward").mockImplementation(function (this: CodeNode, inputs) {
      calls.push(this.id);
      expect(this.id).toBe("outer/repeated#1/bump");
      return original.call(this, inputs);
    });
    try {
      const result = await advance({ log: path, memory: backends, env });
      expect(result.status).toBe("succeeded");
      expect(result.output).toEqual({ mapped: [1n, 2n], repeated: { n: 2n, exited: false } });
      expect(calls).toEqual(["outer/repeated#1/bump"]);
      for (const [key, backend] of backends) expect(backend.openSnapshots).toEqual([snapshots.get(key)!]);
      expect(readFileSync(path, "utf8").startsWith(prefix)).toBe(true);

      const log = JsonlRunLog.resume(path);
      let replayed: RunState;
      try {
        replayed = replayRunLog(flow, log);
      } finally {
        log.close();
      }
      expect(replayed.flowFingerprint).toBe(initial.flowFingerprint);
      expect(initial.flowFingerprint).toBe(flowFingerprint(flow.compiled));
      expect(replayed.effectiveLlmByStaticPath).toEqual(initial.effectiveLlmByStaticPath as never);
      expect(replayed.learnable).toBe(true);
      const historical = replayed.trace!.occurrences.filter((item) => item.outcomeSeq < length);
      expect(
        new Set(historical.filter((item) => item.record instanceof NodeExpandedRecord).map((item) => item.nodeId)),
      ).toEqual(new Set(["outer", "outer/mapped", "outer/repeated"]));
      for (const occurrence of historical) expect(occurrence.recordJson).toBe(untimed(lines[occurrence.outcomeSeq]!));

      const complete = readFileSync(path, "utf8");
      const inspected = await advance({
        log: path,
        memory: new Map([["episodic", {}], ["reflective", {}]]) as never,
        env,
      });
      expect(inspected.output).toEqual(result.output);
      expect(inspected.events).toEqual([]);
      expect(readFileSync(path, "utf8")).toBe(complete);
      expect(calls).toEqual(["outer/repeated#1/bump"]);
    } finally {
      forward.mockRestore();
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("resume prepends ordered effect retries before existing ready work", async () => {
  const loaded = await load(ORDERED_EFFECTS);
  const { state, log } = newStartedRun(loaded, 2);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareOutput(log, state, START_ID, {}));
  commitPrepared(log, state, prepareNodeStarted(state, "release"));
  commitPrepared(log, state, prepareOutput(log, state, "release", "ready"));
  expect(state.ready).toEqual(["wait", "agent", "human", "later"]);

  const retryOrder = ["wait", "agent", "human"];
  expect(retryOrder).not.toEqual([...retryOrder].sort());
  for (const nodeId of retryOrder) commitPrepared(log, state, prepareNodeStarted(state, nodeId));
  const startedOrder = log
    .records()
    .flatMap((envelope) => (envelope.record instanceof NodeStartedRecord ? [envelope.record.nodeId] : []));
  expect(startedOrder).toEqual([START_ID, "release", ...retryOrder]);
  expect([...state.inFlight]).toEqual(retryOrder);
  expect(state.ready).toEqual(["later"]);

  const recovered = replayRunLog(loaded, log);
  expect([...recovered.inFlight]).toEqual(retryOrder);
  expect(recovered.ready).toEqual(["later"]);

  const prepared = prepareRunResumed(recovered, { numWorkers: 2 });
  expect(prepared.record).toEqual(new RunResumedRecord(retryOrder, 2));
  commitPrepared(log, recovered, prepared);
  expect(log.records().at(-1)!.record).toEqual(prepared.record);
  expect([...recovered.inFlight]).toEqual([]);
  expect(recovered.ready).toEqual([...retryOrder, "later"]);
});

// ---- The retry frontier of unfinished invocations ----

/**
 * Return `START -> agent -> END` where `agent` has the memory scope `notes`; with `sibling`, also
 * `START -> sibling`.
 */
function agentFlow(options: { sibling?: boolean } = {}): CompiledFlow {
  const agent = new FuncNode("agent", () => {
    throw new Error("replay invoked a node");
  });
  agent.memoryScope = new MemoryScope(["agent"], { memories: ["notes"] });
  const built = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["agent", agent],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-agent", START_ID, "agent"), new Edge("agent-end", "agent", END_ID)],
  );
  if (options.sibling) {
    const sibling = new FuncNode("sibling", () => {
      throw new Error("node called");
    });
    built.addSubgraph(new Map([["sibling", sibling]]), [new Edge("start-sibling", START_ID, "sibling")], new Map([
      ["sibling", new Map()],
    ]));
  }
  return built;
}

/**
 * Return a state and log with START succeeded and `agent` in flight. The memory `notes` is
 * recorded unless `enabled` is false; `learnable` and `sibling` shape the run and the flow.
 */
function agentStarted(
  options: { learnable?: boolean; sibling?: boolean; enabled?: boolean } = {},
): { state: RunState; log: MemoryRunLog } {
  const enabled = options.enabled ?? true;
  const flow = agentFlow(options);
  const state = new RunState(flow);
  const pool = new VariablePool(flow.startId);
  pool.set(flow.startId, {});
  pool.addSystem("run_id", "run");
  const log = new MemoryRunLog("run");
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: "run",
      flowFingerprint: "a".repeat(64),
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 0,
      learnable: options.learnable ?? false,
      memories: enabled ? [new MemorySpec("notes", "episodic", {}, [new DeclSite([], ["root"], "agent")])] : null,
      memoryStacks: enabled ? [new MemoryStack(["agent"], ["notes"])] : null,
      memorySnapshots: enabled ? [new MemorySnapshotEnvelope("notes", {})] : null,
    }),
  );
  commitInitialPool(log, state, pool);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const forwardCapture = state.learnable ? new ForwardCapture(ValueRoute.record([])) : null;
  commitPrepared(log, state, prepareOutput(log, state, START_ID, {}, { forwardCapture }));
  commitPrepared(log, state, prepareNodeStarted(state, "agent"));
  return { state, log };
}

/** Return `reducerSnapshot(state)` with the memory, learning and backward state. */
function frontierSnapshot(state: RunState): Record<string, unknown> {
  return {
    ...reducerSnapshot(state),
    memories: state.memories,
    memoryStacks: state.memoryStacks,
    memorySnapshots: state.memorySnapshots,
    learnable: state.learnable,
    backwardState: state.backwardState,
  };
}

/** Return `state`'s topology and each node's class, scopes and `commitAs`. */
function frontierFlowSnapshot(state: RunState): Record<string, unknown> {
  const flow = state.flow;
  return {
    initialStaticIds: [...state.initialStaticIds],
    startId: flow.startId,
    endId: flow.endId,
    edges: [...flow.edges],
    wiring: [...flow.wiring],
    incoming: [...flow.incomingEdges],
    outgoing: [...flow.outgoingEdges],
    nodes: [...flow.nodes].map(([nodeId, node]) => [
      nodeId,
      node.constructor,
      node.memoryScope,
      node.executionScope,
      node.commitAs,
    ]),
  };
}

/** Replay `log` over a fresh flow and expect the replay to equal the live `state`; return it. */
function expectReplays(state: RunState, log: MemoryRunLog, options: { sibling?: boolean } = {}): RunState {
  const replayed = replayRecords(new RunState(agentFlow(options)), log.records(), {
    growBuilder: () => {
      throw new Error("builder called");
    },
  });
  expect(frontierSnapshot(replayed)).toEqual(frontierSnapshot(state));
  expect(frontierFlowSnapshot(replayed)).toEqual(frontierFlowSnapshot(state));
  const trace = (s: RunState) => (s.trace === null ? null : replayTraceSnapshot(s.trace));
  expect(trace(replayed)).toEqual(trace(state));
  return replayed;
}

test("a resume allows memory-free unfinished invocations", () => {
  const { state, log } = agentStarted({ enabled: false });
  const firstSequence = state.acceptedStarts.get("agent")!;
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  expect(state.acceptedStarts).toEqual(new Map());
  commitPrepared(log, state, prepareNodeStarted(state, "agent"));
  expect(state.acceptedStarts.get("agent")!).toBeGreaterThan(firstSequence);
  expectReplays(state, log);
});

test.each([false, true])("a resume retries every unfinished invocation (learnable: %p)", (learnable) => {
  const { state, log } = agentStarted({ learnable, sibling: true });
  commitPrepared(log, state, prepareNodeStarted(state, "sibling"));
  const before = [...state.inFlight];
  const prepared = prepareRunResumed(state, { numWorkers: 0 });
  expect((prepared.record as RunResumedRecord).retryNodeIds).toEqual(before);
  expect(before).toEqual(["agent", "sibling"]);
  commitPrepared(log, state, prepared);
  expect(state.ready).toEqual(before);
  expect(state.inFlight).toEqual(new Set());
  expect(state.acceptedStarts).toEqual(new Map());
  expectReplays(state, log, { sibling: true });
});

test("a forged resume plan can't change the retry frontier", () => {
  const { state, log } = agentStarted();
  const prepared = prepareRunResumed(state, { numWorkers: 0 });
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new Error(`expected a frontier plan, found ${plan.kind}`);
  const forged = new PreparedTransition(
    prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    Object.freeze({ ...plan, delta: Object.freeze({ ...plan.delta, retryNodeIds: Object.freeze([]) }) }),
    prepared.observers,
  );
  const before = snapshot(state);

  let caught: unknown;
  try {
    commitPrepared(log, state, forged);
  } catch (error) {
    caught = error;
  }

  expect((caught as Error).constructor).toBe(RangeError);
  expect((caught as Error).message).toBe("frontier resume delta disagrees with its record");
  expectUnchanged(state, before);
  expect([...state.inFlight]).toEqual(["agent"]);
});
