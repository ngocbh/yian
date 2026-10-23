/**
 * Public fresh and resume orchestration for host-injected memory policies: mirrors
 * `tests/engine/test_run_memory.py`.
 *
 * Ported, through `runFlow` and `resumeFlow`:
 * `test_memory_context_opener_uses_same_runtime_on_fresh_and_resume` (fresh and resume ×
 * `num_workers` 0 and 2 × `learnable` false and true; the AGENT recalls notes from a `FakeMemory`,
 * since TS yian has no session memory to open a turn in),
 * `test_authored_policy_is_dormant_when_memory_is_none`,
 * `test_recall_only_records_snapshot_without_evidence_or_learning`,
 * `test_unfinished_primary_resume_uses_static_layout_after_map_growth`,
 * `test_unfinished_primary_requires_recorded_memory_mapping_on_resume`,
 * `test_run_started_records_the_resolved_memory_table_and_stacks` and
 * `test_replay_rejects_a_recorded_key_table_or_stacks_that_differ_from_the_flow` (2 rows); with
 * `RunResult.backward` too: `test_learnable_true_retains_without_enabling_an_authored_policy`,
 * `test_learning_success_is_separate_from_the_primary_result`,
 * `test_learning_failure_does_not_replace_primary_success`,
 * `test_primary_failure_still_learns_without_changing_its_failure`,
 * `test_precommit_failure_requires_an_explicit_whole_pass_retry`,
 * `test_explicit_retry_uses_static_layout_after_map_growth`,
 * `test_code_node_occurrences_are_episodes_of_the_flow_key` and
 * `test_backward_rejects_a_tampered_recorded_stack_set_before_learning` (2 rows).
 *
 * Changed: `AgentNode.prototype.forward` is spied on, where Python patches `AgentNode.forward`.
 * `runFlow` builds its `FlowEngine` itself, so the
 * runtimes `MemoryRuntime.open` returns (spied on) stand in for the openers Python's patched
 * `FlowEngine` receives, and TS has no `memory_recall` option to refuse. The flow's int output is
 * `7n`. A tampered `RunStartedRecord` is rebuilt from its fields (Python's `dataclasses.replace`);
 * the mismatch is a `RangeError` (Python's `ValueError`). The failing CODE body is
 * TypeScript throwing an `Error` (Python raises `RuntimeError`). The settled resume of
 * `test_precommit_failure_requires_an_explicit_whole_pass_retry` passes an empty object as the
 * ignored memory (Python's `object()`).
 *
 * Not ported: `test_conversation_completes_before_dispatcher_output_contract_acceptance` (TS yian
 * has no session memory).
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { NodeExpanded, NodeStarted, RunStarted } from "../../src/events.ts";
import {
  DeclSite,
  type JsonObject,
  type Memory,
  MemoryOperationError,
  MemorySetupError,
  MemorySpec,
  MemoryStack,
} from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { resumeFlow, type RunResult, runFlow } from "../../src/run.ts";
import {
  BackwardFinishedRecord,
  BackwardStartedRecord,
  decodeRecord,
  encodeRecord,
  JsonlRunLog,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  RecordEnvelope,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { rejected } from "./run_state_fixtures.ts";

const FLOW = `id: memory-run
name: memory_run
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: \${work.output}
`;

const FAILED_FLOW = `id: memory-run-failed
name: memory_run_failed
long_term_memory: test
nodes:
  work:
    kind: code
    code: |
      if (true) throw new Error("boom");
      return 0;
    output: int
output: \${work.output}
`;

const MAP_FLOW = `id: memory-map
name: memory_map
long_term_memory: test
input:
  values: !list(int)
defs:
  increment:
    input:
      value: !int
    nodes: {}
    output: \${input.value + 1}
nodes:
  each:
    kind: map
    call: increment
    over: \${input.values}
    input:
      value: \${item}
output: \${each.output}
`;

const AGENT_FLOW = `id: memory-run
name: memory_run
llm_config: {provider: openai, model: fake-model}
nodes:
  work:
    kind: agent
    long_term_memory: test
    prompt: hi
    output: int
output: \${work.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-memory-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** The durable records of the log at `path`, without keeping its lock. */
function records(path: string): RunRecord[] {
  const log = JsonlRunLog.resume(path);
  try {
    return log.records().map((envelope) => envelope.record);
  } finally {
    log[Symbol.dispose]();
  }
}

/** An `onEvent` that throws `message` at the first event `stop` accepts. */
function stopAt(stop: (event: unknown) => boolean, message = "host stop"): (event: unknown) => void {
  return (event) => {
    if (stop(event)) throw new Error(message);
  };
}

for (const resume of [false, true]) {
  for (const numWorkers of [0, 2]) {
    for (const learnable of [false, true]) {
      const name = `resume ${resume}, numWorkers ${numWorkers}, learnable ${learnable}`;
      test(`memory context opener uses same runtime on fresh and resume (${name})`, async () => {
        const path = join(dir, "recall.jsonl");
        const memory = new FakeMemory({ recallText: "notes" });
        const registry = new Map([["test", memory]]);
        const entered: (readonly string[])[] = [];
        const runtimes: MemoryRuntime[] = [];
        const open = MemoryRuntime.open.bind(MemoryRuntime);
        const opening = spyOn(MemoryRuntime, "open").mockImplementation(async (options) => {
          const runtime = await open(options);
          runtimes.push(runtime);
          return runtime;
        });
        const forward = spyOn(AgentNode.prototype, "forward").mockImplementation(async (_inputs, caps) => {
          const context = await caps!.openMemoryContext!("system", "prompt");
          entered.push(context.sections.map((section) => `${section.memory}: ${section.text}`));
          return new Output(7n);
        });
        try {
          const flow = await loadFlow(AGENT_FLOW);
          let result: RunResult;
          if (resume) {
            const onEvent = stopAt(
              (event) => event instanceof NodeStarted && event.nodeId === "work",
              "interrupt before body",
            );
            const options = { log: path, memory: registry, numWorkers, onEvent, learnable };
            const stop = await rejected(runFlow(flow, {}, options));
            expect((stop as Error).message).toBe("interrupt before body");
            expect(entered).toEqual([]);
            const before = readFileSync(path);
            result = await resumeFlow({ log: path, memory: registry });
            expect(readFileSync(path).subarray(0, before.length)).toEqual(before);
          } else {
            result = await runFlow(flow, {}, { log: path, memory: registry, numWorkers, learnable });
          }
          expect(result.output).toBe(7n);
        } finally {
          forward.mockRestore();
          opening.mockRestore();
        }
        expect(entered).toEqual([["test: notes"]]);
        expect(memory.recallContexts.map((context) => context!.consumerStaticPath)).toEqual([["work"]]);
        expect(runtimes).toHaveLength(resume ? 2 : 1);
        expect(runtimes.every((runtime) => runtime.closed)).toBe(true);
      });
    }
  }
}

test("authored policy is dormant when memory is none", async () => {
  const path = join(dir, "dormant.jsonl");
  const result = await runFlow(await loadFlow(FLOW), {}, { log: path });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(7n);
  expect(result.learning).toBeNull();
  const start = records(path)[0];
  expect(start).toBeInstanceOf(RunStartedRecord);
  const started = start as RunStartedRecord;
  expect(started.memories).toBeNull();
  expect(started.memorySnapshots).toBeNull();
  expect(started.learnable).toBe(false);
});

test("learnable true retains without enabling an authored policy", async () => {
  const path = join(dir, "retained.jsonl");
  const result = await runFlow(await loadFlow(FLOW), {}, { log: path, learnable: true });
  expect(result.learning).toBeNull();
  expect(result.backwardOutcome).toBeNull();
  expect((records(path)[0] as RunStartedRecord).memories).toBeNull();
  expect((await result.backward()).learning).toBeNull();
});

test("recall only records snapshot without evidence or learning", async () => {
  const path = join(dir, "recall-only.jsonl");
  const memory = new FakeMemory();
  const result = await runFlow(await loadFlow(FLOW), {}, { log: path, memory: new Map([["test", memory]]) });
  expect(result.status).toBe("succeeded");
  expect(result.learning).toBeNull();
  expect(memory.sessions).toHaveLength(1);
  expect(memory.learnContexts).toEqual([]);
  expect(memory.commitCalls).toEqual([]);
  expect(memory.closeCount).toBe(1);
  const all = records(path);
  const start = all[0];
  expect(start).toBeInstanceOf(RunStartedRecord);
  const started = start as RunStartedRecord;
  expect(started.memories).toHaveLength(1);
  expect(started.memorySnapshots).not.toBeNull();
  expect(started.learnable).toBe(false);
  expect(all.some((record) => record instanceof BackwardStartedRecord)).toBe(false);
});

test("learning success is separate from the primary result", async () => {
  const path = join(dir, "learn.jsonl");
  const memory = new FakeMemory();
  const options = { log: path, memory: new Map([["test", memory]]), learnable: true };
  const result = await runFlow(await loadFlow(FLOW), {}, options);
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(7n);
  expect(result.learning).toBeNull();
  await result.backward();
  expect(result.learning).not.toBeNull();
  expect(result.learning!.status).toBe("succeeded");
  expect(memory.learnContexts).toHaveLength(1);
  expect(memory.learnContexts[0]!.episodes).toHaveLength(3);
  expect(memory.learnContexts[0]!.feedback).toEqual([]);
  expect(memory.commitCalls).toHaveLength(1);
  expect(memory.commitCalls[0]![1]).toEqual([]);
  const all = records(path);
  expect(all.at(-3)).toBeInstanceOf(MemoryCommitRequestedRecord);
  expect(all.at(-2)).toBeInstanceOf(MemoryCommitSucceededRecord);
  expect(all.at(-1)).toBeInstanceOf(BackwardFinishedRecord);
});

test("learning failure does not replace primary success", async () => {
  const path = join(dir, "learn-failed.jsonl");
  const memory = new FakeMemory({
    learnOutcomes: [new MemoryOperationError("learner", "declined", "Cannot learn.", true)],
  });
  const options = { log: path, memory: new Map([["test", memory]]), learnable: true };
  const result = await runFlow(await loadFlow(FLOW), {}, options);
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(7n);
  expect(result.failure).toBeNull();
  const outcome = await result.backward();
  expect(outcome.status).toBe("failed");
  expect(outcome.failure!.stage).toBe("learn");
  expect(result.learning).not.toBeNull();
  expect(result.learning!.status).toBe("failed");
  expect(result.learning!.failures).toHaveLength(1);
  expect(result.learning!.failures[0]!.operation).toBe("learn");
  expect(result.learning!.failures[0]!.failure.message).toBe("Cannot learn.");
  expect(records(path).some((record) => record instanceof BackwardFinishedRecord)).toBe(true);
});

test("primary failure still learns without changing its failure", async () => {
  const path = join(dir, "primary-failed.jsonl");
  const memory = new FakeMemory();
  const options = { log: path, memory: new Map([["test", memory]]), learnable: true };
  const result = await runFlow(await loadFlow(FAILED_FLOW), {}, options);
  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.code).toBe("node.execute");
  await result.backward("Review the failure.");
  expect(result.learning).not.toBeNull();
  expect(result.learning!.status).toBe("succeeded");
  expect(memory.learnContexts).toHaveLength(1);
  expect(memory.commitCalls).toHaveLength(1);
});

test("precommit failure requires an explicit whole pass retry", async () => {
  const path = join(dir, "retry.jsonl");
  const retryable = new MemoryOperationError("learn", "test.busy", "try later", true);
  const memory = new FakeMemory({ learnOutcomes: [retryable, []] });
  const registry = new Map([["test", memory]]);
  const flow = await loadFlow(FLOW);

  const first = await runFlow(flow, {}, { log: path, memory: registry, learnable: true });
  await first.backward();
  const primaryPrefix = records(path).length;

  expect(first.status).toBe("succeeded");
  expect(first.learning).not.toBeNull();
  expect(first.learning!.status).toBe("failed");
  const resumed = await resumeFlow({ log: path, memory: registry });
  expect(resumed.learning).toEqual(first.learning);
  expect(records(path)).toHaveLength(primaryPrefix);
  expect(memory.learnContexts).toHaveLength(1);
  await resumed.backward();

  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe(7n);
  expect(resumed.events).toEqual([]);
  expect(resumed.learning).not.toBeNull();
  expect(resumed.learning!.status).toBe("succeeded");
  expect(memory.learnContexts).toHaveLength(2);
  const suffix = records(path).slice(primaryPrefix);
  expect(suffix.some((record) => record instanceof RunResumedRecord)).toBe(false);

  const settledCount = records(path).length;
  // Ignored: the primary run and its learning are settled.
  const ignored = new Map([["test", {} as Memory]]);
  const silent = await resumeFlow({ log: path, memory: ignored });
  expect(silent.learning).not.toBeNull();
  expect(silent.learning!.status).toBe("succeeded");
  expect(records(path)).toHaveLength(settledCount);
});

test("explicit retry uses static layout after map growth", async () => {
  const path = join(dir, "map-learning.jsonl");
  const retryable = new MemoryOperationError("learn", "test.busy", "later", true);
  const memory = new Map([["test", new FakeMemory({ learnOutcomes: [retryable, []] })]]);
  const flow = await loadFlow(MAP_FLOW);

  const first = await runFlow(flow, { values: [1n, 4n] }, { log: path, memory, learnable: true });
  await first.backward();
  const resumed = await resumeFlow({ log: path, memory });

  expect(first.status).toBe("succeeded");
  expect(first.output).toEqual([2n, 5n]);
  expect(first.learning).not.toBeNull();
  expect(first.learning!.status).toBe("failed");
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toEqual([2n, 5n]);
  expect(resumed.events).toEqual([]);
  expect(resumed.learning).not.toBeNull();
  expect(resumed.learning).toEqual(first.learning);
  await resumed.backward();
  expect(resumed.learning!.status).toBe("succeeded");
});

test("unfinished primary resume uses static layout after map growth", async () => {
  const path = join(dir, "map-primary.jsonl");
  const memory = new Map([["test", new FakeMemory()]]);
  const flow = await loadFlow(MAP_FLOW);
  const onEvent = stopAt((event) => event instanceof NodeExpanded, "host stop after map growth");
  const stop = await rejected(runFlow(flow, { values: [1n, 4n] }, { log: path, memory, onEvent }));
  expect((stop as Error).message).toBe("host stop after map growth");
  const resumed = await resumeFlow({ log: path, memory });
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toEqual([2n, 5n]);
});

test("unfinished primary requires the recorded memory mapping on resume", async () => {
  const path = join(dir, "unfinished.jsonl");
  const memory = new FakeMemory();
  const onEvent = stopAt((event) => event instanceof RunStarted);
  const options = { log: path, memory: new Map([["test", memory]]), onEvent };
  const stop = await rejected(runFlow(await loadFlow(FLOW), {}, options));
  expect((stop as Error).message).toBe("host stop");
  const missing = await rejected(resumeFlow({ log: path }));
  expect(missing).toBeInstanceOf(MemorySetupError);
  expect((missing as Error).message).toContain("requires");
  const resumed = await resumeFlow({ log: path, memory: new Map([["test", memory]]) });
  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toBe(7n);
  expect(memory.openSnapshots).toEqual([null, { namespace: "test" }]);
});

test("run started records the resolved memory table and stacks", async () => {
  const path = join(dir, "table.jsonl");
  const result = await runFlow(await loadFlow(FLOW), {}, { log: path, memory: new Map([["test", new FakeMemory()]]) });
  expect(result.status).toBe("succeeded");
  const start = records(path)[0] as RunStartedRecord;
  expect(start.memories).toEqual([
    new MemorySpec("test", "test", {} as JsonObject, [new DeclSite([], ["root"], null)]),
  ]);
  expect(start.memoryStacks!.map((stack) => stack.locator)).toContainEqual(["work"]);
  expect(start.memoryStacks!.every((stack) => stack.memories.length === 1 && stack.memories[0] === "test")).toBe(true);
});

for (const tampered of ["stacks", "table"] as const) {
  test(`replay rejects a recorded key table or stacks that differ from the flow (${tampered})`, async () => {
    const path = join(dir, "tampered.jsonl");
    const memory = new Map([["test", new FakeMemory()]]);
    const onEvent = stopAt((event) => event instanceof RunStarted);
    const stop = await rejected(runFlow(await loadFlow(FLOW), {}, { log: path, memory, onEvent }));
    expect((stop as Error).message).toBe("host stop");
    const lines = readFileSync(path, "utf8").split("\n").filter((line) => line !== "");
    const envelope = decodeRecord(lines[0]!);
    const record = envelope.record as RunStartedRecord;
    let changed: RunStartedRecord;
    if (tampered === "stacks") {
      const emptied = record.memoryStacks!.map((stack) => new MemoryStack(stack.locator, []));
      changed = new RunStartedRecord({ ...record, memoryStacks: emptied });
    } else {
      expect(record.memories).toHaveLength(1);
      const [spec] = record.memories!;
      const edited = new MemorySpec(spec!.key, spec!.type, { recall_limit: 1n }, spec!.owners);
      changed = new RunStartedRecord({ ...record, memories: [edited] });
    }
    lines[0] = encodeRecord(new RecordEnvelope(envelope.schema, envelope.runId, envelope.seq, changed, envelope.at));
    writeFileSync(path, lines.join("\n") + "\n");
    const error = await rejected(resumeFlow({ log: path, memory }));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("memory table or stacks");
  });
}

test("code node occurrences are episodes of the flow key", async () => {
  const memory = new FakeMemory();
  const options = { learnable: true, log: join(dir, "code.jsonl"), memory: new Map([["test", memory]]) };
  const result = await runFlow(await loadFlow(FLOW), {}, options);
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(7n);

  const outcome = await result.backward("Prefer eight.");

  expect(outcome.status).toBe("succeeded");
  expect(memory.learnContexts).toHaveLength(1);
  const context = memory.learnContexts[0]!;
  expect(context.memory).toBe("test");
  const work = context.episodes.filter((episode) => episode.ownerStaticPath.join("/") === "work");
  expect(work).toHaveLength(1);
  expect(work[0]!.output).toBe(7n);
  expect(work[0]!.occurrencePath).toEqual([]);
  expect(context.feedback).toContainEqual(expect.objectContaining({ owner_static_path: ["work"] }));
  expect(memory.commitCalls).toHaveLength(1);
});

for (const tamper of ["emptied", "dropped"] as const) {
  test(`backward rejects a tampered recorded stack set before learning (${tamper})`, async () => {
    const path = join(dir, "tampered-backward.jsonl");
    const memory = new FakeMemory();
    const options = { learnable: true, log: path, memory: new Map([["test", memory]]) };
    const result = await runFlow(await loadFlow(FLOW), {}, options);
    expect(result.status).toBe("succeeded");

    const lines = readFileSync(path, "utf8").split("\n").filter((line) => line !== "");
    const envelope = decodeRecord(lines[0]!);
    const record = envelope.record as RunStartedRecord;
    const recorded = record.memoryStacks!;
    const stacks =
      tamper === "emptied"
        ? recorded.map((stack) => new MemoryStack(stack.locator, []))
        : recorded.filter((stack) => stack.locator.join("/") !== "work");
    const changed = new RunStartedRecord({ ...record, memoryStacks: stacks });
    lines[0] = encodeRecord(new RecordEnvelope(envelope.schema, envelope.runId, envelope.seq, changed, envelope.at));
    writeFileSync(path, lines.join("\n") + "\n");

    const error = await rejected(result.backward("Review."));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("memory table or stacks");
    expect(memory.learnContexts).toEqual([]);
    expect(memory.commitCalls).toEqual([]);
  });
}
