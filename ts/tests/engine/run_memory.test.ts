/**
 * Public fresh and resume orchestration for host-injected memory policies: mirrors
 * `tests/engine/test_run_memory.py`.
 *
 * Ported, through `runFlow`: the fresh rows of
 * `test_memory_context_opener_uses_same_runtime_on_fresh_and_resume` (`num_workers` 0 and 2 ×
 * `learnable` false and true), `test_conversation_completes_before_dispatcher_output_contract_acceptance`
 * (2 rows), `test_authored_policy_is_dormant_when_memory_is_none`,
 * `test_recall_only_records_snapshot_without_evidence_or_learning` and
 * `test_run_started_records_the_resolved_memory_table_and_stacks`.
 *
 * Changed: `AgentNode.prototype.forward` is spied on, where Python patches `AgentNode.forward`;
 * the turn's request is its private `turnRequest`. `runFlow` builds its `FlowEngine` itself, so the
 * runtimes `MemoryRuntime.open` returns (spied on) stand in for the openers Python's patched
 * `FlowEngine` receives, and TS has no `memory_recall` option to refuse. The flow's int output is
 * `7n`.
 *
 * Later (T6-6, with resumeFlow): the resume rows of
 * `test_memory_context_opener_uses_same_runtime_on_fresh_and_resume` (4 rows),
 * `test_unfinished_primary_resume_uses_static_layout_after_map_growth`,
 * `test_unfinished_primary_requires_recorded_memory_mapping_on_resume` and
 * `test_replay_rejects_a_recorded_key_table_or_stacks_that_differ_from_the_flow` (2 rows).
 *
 * Later (T6-6, with RunResult.backward): `test_learnable_true_retains_without_enabling_an_authored_policy`,
 * `test_learning_success_is_separate_from_the_primary_result`,
 * `test_learning_failure_does_not_replace_primary_success`,
 * `test_primary_failure_still_learns_without_changing_its_failure`,
 * `test_precommit_failure_requires_an_explicit_whole_pass_retry`,
 * `test_explicit_retry_uses_static_layout_after_map_growth`,
 * `test_code_node_occurrences_are_episodes_of_the_flow_key` and
 * `test_backward_rejects_a_tampered_recorded_stack_set_before_learning` (2 rows).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ConversationItem, DeclSite, type JsonObject, MemorySpec } from "../../src/memory/contracts.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { runFlow } from "../../src/run.ts";
import { BackwardStartedRecord, JsonlRunLog, type RunRecord, RunStartedRecord } from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { type Mutable, snapshot } from "./session_fakes.ts";

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
  using log = JsonlRunLog.resume(path);
  return log.records().map((envelope) => envelope.record);
}

/** The conversation `memory` holds now. */
function history(memory: SessionMemory): Mutable {
  return snapshot(memory)["conversation"];
}

for (const numWorkers of [0, 2]) {
  for (const learnable of [false, true]) {
    test(`memory context opener uses same runtime on fresh (numWorkers ${numWorkers}, learnable ${learnable})`, async () => {
      const path = join(dir, "conversation.jsonl");
      const memory = new SessionMemory();
      const entered: (readonly string[])[] = [];
      const runtimes: MemoryRuntime[] = [];
      const open = MemoryRuntime.open.bind(MemoryRuntime);
      const opening = spyOn(MemoryRuntime, "open").mockImplementation(async (options) => {
        const runtime = await open(options);
        runtimes.push(runtime);
        return runtime;
      });
      const forward = spyOn(AgentNode.prototype, "forward").mockImplementation(async (_inputs, caps) => {
        const turn = (await caps!.openMemoryContext!("system", "prompt", { format: "test-v1" })).transcript!;
        entered.push((turn as Mutable).turnRequest.consumerStaticPath);
        const ordinal = await turn.request("model", { input: "test" });
        await turn.observe(ordinal, "completed", { items: [new ConversationItem("reply", "test-v1", { value: 7 })] });
        return new Output(7n);
      });
      try {
        const result = await runFlow(await loadFlow(AGENT_FLOW), {}, {
          log: path,
          memory: new Map([["test", memory]]),
          numWorkers,
          learnable,
        });
        expect(result.output).toBe(7n);
      } finally {
        forward.mockRestore();
        opening.mockRestore();
      }
      expect(entered).toEqual([["work"]]);
      expect(runtimes).toHaveLength(1);
      expect(runtimes.every((runtime) => runtime.closed)).toBe(true);
      expect(records(path).some((record) => record.constructor.name.startsWith("Conversation"))).toBe(false);
      expect(history(memory).closed_invocations.at(-1).status).toBe("completed");
      expect(history(memory).items[0].payload).toEqual({ value: 7 });
    });
  }
}

for (const numWorkers of [0, 2]) {
  test(`conversation completes before dispatcher output contract acceptance (numWorkers ${numWorkers})`, async () => {
    const memory = new SessionMemory();
    const forward = spyOn(AgentNode.prototype, "forward").mockImplementation(async (_inputs, caps) => {
      await caps!.openMemoryContext!("system", "prompt", {});
      return new Output("not an integer");
    });
    try {
      const result = await runFlow(await loadFlow(AGENT_FLOW), {}, {
        log: join(dir, "rejected.jsonl"),
        memory: new Map([["test", memory]]),
        numWorkers,
      });
      expect(result.status).toBe("failed");
      expect(result.failure!.code).toBe("contract.output");
    } finally {
      forward.mockRestore();
    }
    expect(history(memory).closed_invocations.at(-1).status).toBe("completed");
  });
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

test("run started records the resolved memory table and stacks", async () => {
  const path = join(dir, "table.jsonl");
  const result = await runFlow(await loadFlow(FLOW), {}, { log: path, memory: new Map([["test", new FakeMemory()]]) });
  expect(result.status).toBe("succeeded");
  const start = records(path)[0] as RunStartedRecord;
  expect(start.memories).toEqual([
    new MemorySpec("test", "test", {} as JsonObject, [new DeclSite([], ["root"], null)], false),
  ]);
  expect(start.memoryStacks!.map((stack) => stack.locator)).toContainEqual(["work"]);
  expect(start.memoryStacks!.every((stack) => stack.memories.length === 1 && stack.memories[0] === "test")).toBe(true);
});
