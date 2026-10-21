/**
 * The package-root run lifecycle over one authoritative append-only log: a port of
 * `tests/engine/test_run_lifecycle.py`.
 *
 * Fresh runs own a process-local log or one durable path; a resume owns a durable writer from
 * the lock to the engine's drain. Recorded history is replayed silently; only records committed
 * in the current segment publish events. The flows use CODE through `./compose_codefns.ts:<fn>`
 * references (resolved from the conformance fixtures) or inline TS bodies, and every run sees only
 * the test's credential store in its environment.
 *
 * Ported: `test_fresh_memory_success_returns_current_segment_and_canonical_input`,
 * `test_fresh_durable_success_returns_path_and_releases_the_lock`,
 * `test_nullable_specialized_list_survives_durable_run_and_terminal_resume`,
 * `test_fresh_memory_flow_failure_returns_durable_diagnostics`,
 * `test_fresh_flow_failure_is_a_result_and_releases_the_lock`,
 * `test_terminal_resume_preserves_source_free_failure`,
 * `test_nonterminal_resume_commits_exact_retries_and_selected_workers[serial-override]`,
 * `test_resume_hook_failure_leaves_the_log_unchanged` and `test_terminal_resume_never_calls_the_hook`.
 *
 * Changed:
 * - Python's `ValueError` is a `RangeError`; a CODE failure's message and traceback name the JS
 *   error, `Error: boom` (Python: the message `boom`, the traceback `ValueError: boom`).
 * - `test_final_run_boundary_signatures_and_result_fields_are_exact`: `resumeFlow(flow, options)`
 *   takes an options record, so the row checks `ResumeFlowOptions`'s keys (TS-only: `signal`,
 *   `env`, `scheduler`, `random`, `models`; `checkLlmConfigs` is `@internal`, Python's
 *   `_resume_flow_controlled` parameter, and there is no `stop_event`: the stop is `signal`).
 *   `runFlow`'s keys and `RunResult`'s fields are checked in `retired_runtime_surface.test.ts`.
 * - `test_terminal_resume_is_read_only_silent_and_unlocks[succeeded]`,
 *   `test_nonterminal_resume_restores_recorded_llm_without_reading_defaults`: ESM bindings can't
 *   be patched, so the resume's environment fails any read of `YIAN_DEFAULT_PROVIDER` or the
 *   credential store instead of `connection_snapshot` being replaced; the AGENT's model comes from
 *   the `models` option (Python patches `model_from_config`).
 * - `test_terminal_resume_preserves_recorded_trace_across_origin_rebase`: the loaded flow's END
 *   output is set back to the source without `:-null`, and its edge to required (Python patches
 *   `null_skipped_outputs` during the load).
 * - `test_run_started_append_failure_prevents_execution_observation_and_closes` and
 *   `test_outcome_append_failure_leaves_started_work_recoverable_and_closes`: `JsonlRunLog.create`
 *   is spied on to return the tracking log (which keeps no path: the run boundary never reads it
 *   back); the action's runs are counted by spying on `CodeNode.prototype.forward`.
 * - `test_resume_callback_exception_closes_and_unlocks_nonterminal_log`: a public `resumeFlow`
 *   (Python's `_resume_flow_controlled`); TS appends one `run_stopped` (reason `callback`) after
 *   the second start.
 * - `test_resume_hook_sees_recorded_configs_before_run_resumed`: the hook is the `checkLlmConfigs`
 *   option, called with `{callTimeHost}`.
 * - `test_callback_exception_drains_waiter_reraises_exactly_and_unlocks`,
 *   `test_callback_exception_drains_real_blocked_human_interaction_and_unlocks` and
 *   `test_host_interruption_drains_then_durable_resume_retries_to_terminal`: the trigger is a TOOL
 *   node whose tool waits for the WAIT's sleep (or the `ask_user` call) to start, through a
 *   scheduler whose sleeps wait for their signal (Python's `_TrackingStopEvent`); a host stop is
 *   the run's `signal`, which the interrupting tool aborts, and the run rejects with its reason
 *   (Python: `_HostStop` raised by a CODE node); the recovered WAIT reads a clock past its deadline
 *   from `scheduler`. TS records the stop as `run_stopped`, and nothing is named `RunAborted`.
 *
 * TS-only: a resume after a crash publishes `RunResumed` then one `NodeAttemptResumed` per retried
 * node with its interrupted count (2 after a second crash), and the result holds the counts; a
 * resume with an aborted signal rejects with its reason before the lock is taken.
 *
 * Covered by a case: `test_terminal_resume_is_read_only_silent_and_unlocks[failed]` and
 * `test_terminal_resume_preserves_compose_failure_and_diagnosis`
 * (`resume/run_lifecycle__a_finished_failed_run_resumes_read_only_with_its_failure`),
 * `test_nonterminal_resume_commits_exact_retries_and_selected_workers[recorded]`
 * (`resume/run_lifecycle__a_resume_retries_in_flight_work_ahead_of_ready_work`) and
 * `[pooled-override]` (`resume/run_lifecycle__a_resume_num_workers_overrides_the_recorded_one`).
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { rebaseFlowOrigins } from "../../src/compile/origins.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import {
  NodeAttemptResumed,
  NodeFailed,
  NodeStarted,
  NodeSucceeded,
  type RunEvent,
  RunFailed,
  RunResumed,
  RunStarted,
  RunSucceeded,
} from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { legacyConnectionConfig, legacyConnectionMap } from "../../src/llm_clients/connection.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { Scheduler } from "../../src/platform.ts";
import {
  type ResumeFlowOptions,
  type RunFlowOptions,
  type RunResult,
  resumeFlow,
  runFlow,
} from "../../src/run.ts";
import {
  JsonlRunLog,
  MemoryRunLog,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  readRunLog,
  type RecordEnvelope,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
} from "../../src/runtime/log.ts";
import { commitPrepared, prepareNodeStarted, prepareNodeSucceeded, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { seedSystemClock } from "../../src/typesys/seeding.ts";

import { textReply } from "./run_state_fixtures.ts";

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));
const RUN_ID = "lifecycle-run";
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-lifecycle-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** An environment holding only the credential store in the test's directory. */
function hostEnv(): Env {
  const auth = join(dir, "auth.json");
  return (name) => (name === "YIAN_AUTH_FILE" ? auth : undefined);
}

/** An environment that fails any read of the current model defaults or the credential store. */
function noDefaultsEnv(): Env {
  return (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER" || name === "YIAN_AUTH_FILE") {
      throw new Error("recovery consulted an ambient LLM default");
    }
    return undefined;
  };
}

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** `runFlow` in `hostEnv()` unless `options` sets `env`. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(), ...options });
}

/** `resumeFlow` in `hostEnv()` unless `options` sets `env`. */
function resume(flow: ComposedFlow, options: ResumeFlowOptions) {
  return resumeFlow(flow, { env: hostEnv(), ...options });
}

/** The error `promise` rejects with; fails the test when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected the promise to reject");
}

/** Read one closed durable log's records, proving its lock was released. */
function durableRecords(path: string): RunRecord[] {
  const log = JsonlRunLog.resume(path);
  try {
    return log.records().map((envelope) => envelope.record);
  } finally {
    log.close();
  }
}

const SUCCESS_FLOW = `
id: lifecycle-success
name: lifecycle_success
input:
  n: !int
nodes:
  twice:
    kind: code
    code: ./compose_codefns.ts:double
    input:
      n: \${input.n}
    output: int
output: \${twice.output}
`;

const NULLABLE_LIST_FLOW = `
id: lifecycle_nullable_list
name: lifecycle_nullable_list
input:
  items: !list(Optional(int))
output: list(Optional(int))
kind: code
code: |
  return inputs["items"]
`;

const FAILURE_FLOW = `
id: lifecycle-failure
name: lifecycle_failure
nodes:
  fail:
    kind: code
    code: ./compose_codefns.ts:boom
    output: str
output: \${fail.output}
`;

const TERMINAL_SKIP_FLOW = `
id: lifecycle-terminal-skip
name: lifecycle_terminal_skip
input:
  score: !float
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 1.0"
        then: selected
  selected:
    kind: code
    code: ./compose_codefns.ts:positive
    input:
      topic: selected
    output: str
output: \${selected.output}
`;

const ACTION_FLOW = `
id: lifecycle-action
name: lifecycle_action
nodes:
  work:
    kind: code
    code: return "done"
    output: str
output: \${work.output}
`;

const PARALLEL_FLOW = `
id: lifecycle-parallel
name: lifecycle_parallel
input:
  n: !int
nodes:
  a:
    kind: code
    code: ./compose_codefns.ts:echo_x
    input:
      x: \${input.n}
    output: int
  b:
    kind: code
    code: ./compose_codefns.ts:echo_x
    input:
      x: \${input.n}
    output: int
  c:
    kind: code
    code: ./compose_codefns.ts:echo_x
    input:
      x: \${input.n}
    output: int
output:
  a: \${a.output}
  b: \${b.output}
  c: \${c.output}
`;

const AGENT_FLOW = `
id: lifecycle-agent
name: lifecycle_agent
input:
  topic: !str
nodes:
  answer:
    kind: agent
    input:
      topic: \${input.topic}
    prompt: Answer \${topic}
    output: str
output: \${answer.output}
`;

const CALLBACK_DRAIN_FLOW = `
id: lifecycle-callback-drain
name: lifecycle_callback_drain
nodes:
  trigger:
    kind: tool
    tool_id: trigger
  waiter:
    kind: wait
    until: "2099-01-01T00:00:00+00:00"
output:
  trigger: \${trigger.output}
  waiter: \${waiter.output}
`;

const HUMAN_DRAIN_FLOW = `
id: lifecycle-human-drain
name: lifecycle_human_drain
nodes:
  trigger:
    kind: tool
    tool_id: trigger
  human:
    kind: human_input
    prompt: "Approve?"
    output: str
output:
  trigger: \${trigger.output}
  human: \${human.output}
`;

const HOST_DRAIN_FLOW = `
id: lifecycle-host-drain
name: lifecycle_host_drain
nodes:
  interrupter:
    kind: tool
    tool_id: interrupter
  waiter:
    kind: wait
    until: "2099-01-01T00:00:00+00:00"
output:
  interrupter: \${interrupter.output}
  waiter: \${waiter.output}
`;

/** One programmatic flow whose failure has no authored origins. */
function sourceFreeFailureFlow(): ComposedFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["fail", new CodeNode("fail", { code: "throw new Error('boom')" })],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  const edges = [new Edge("start-fail", START_ID, "fail"), new Edge("fail-end", "fail", END_ID)];
  return new ComposedFlow(CompiledFlow.fromParts(nodes, edges));
}

/** Count the CODE evaluations while `fn` runs. */
async function codeCalls<T>(fn: () => Promise<T>): Promise<[T, number]> {
  const forward = spyOn(CodeNode.prototype, "forward");
  try {
    const value = await fn();
    return [value, forward.mock.calls.length];
  } finally {
    forward.mockRestore();
  }
}

/** Whether `record` is an outcome of one of `nodeIds`. */
function isOutcomeOf(record: RunRecord, nodeIds: readonly string[]): boolean {
  if (record instanceof NodeSucceededRecord) return nodeIds.includes(record.nodeId);
  return record instanceof NodeFailedRecord && nodeIds.includes(record.failure.nodeId!);
}

/**
 * `TERMINAL_SKIP_FLOW` as loaded before a flow output reading only skippable nodes was made
 * nullable: END requires `selected`, so a route that skips it makes the run fail.
 */
async function oldTerminalSkipFlow(): Promise<ComposedFlow> {
  const loaded = await load(TERMINAL_SKIP_FLOW);
  const compiled = loaded.compiled;
  const [key, binding] = [...compiled.wiring.get(END_ID)!][0]!;
  expect(binding.source).toBe("${selected.output:-null}");
  const required = new Binding("${selected.output}", { origin: binding.origin });
  compiled.wiring.set(END_ID, new Map([[key, required]]));
  compiled.outputs = compiled.outputs.map((output) => new FlowOutput(output.name, required));
  compiled.edges = compiled.edges.map((edge) =>
    edge.to === END_ID
      ? new Edge(edge.id, edge.from, edge.to, {
          sourceHandle: edge.sourceHandle,
          inputGroup: edge.inputGroup,
          ordering: edge.ordering,
        })
      : edge,
  );
  // The copy rebuilds the edge index from the new edges.
  return new ComposedFlow(copyGraph(compiled), {
    version: loaded.version,
    name: loaded.name,
    description: loaded.description,
    originResolver: loaded.originResolver,
  });
}

/** A tool `name` taking no arguments whose `execute` is `body`. */
function noArgTool(name: string, body: (ctx: { signal: AbortSignal }) => unknown) {
  return tool({
    name,
    description: name,
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) => body(ctx) as never,
  });
}

/**
 * A scheduler reading `now` whose sleeps wait for their signal: `entered` resolves at the first
 * sleep, and `released` turns true once a sleep ends on its signal.
 */
function blockingScheduler(now: number): { scheduler: Scheduler; entered: Promise<void>; released: () => boolean } {
  let enter!: () => void;
  const entered = new Promise<void>((resolve) => {
    enter = resolve;
  });
  let released = false;
  const scheduler: Scheduler = {
    now: () => now,
    sleep(_ms, signal) {
      enter();
      return new Promise<void>((_resolve, reject) => {
        const abort = () => {
          released = true;
          reject(signal!.reason);
        };
        if (signal?.aborted) abort();
        else signal?.addEventListener("abort", abort, { once: true });
      });
    },
  };
  return { scheduler, entered, released: () => released };
}

/** Write RunStarted, complete START, then leave the `started` nodes in flight, in order. */
function writeNonterminalPrefix(
  path: string,
  loaded: ComposedFlow,
  inputs: Record<string, unknown>,
  options: { started: readonly string[]; numWorkers: number; effective?: Record<string, Record<string, unknown>> },
): number {
  const privateFlow = copyGraph(loaded.compiled);
  const pool = new VariablePool(privateFlow.startId);
  pool.set(privateFlow.startId, structuredClone(inputs));
  seedSystemClock(pool);
  pool.addSystem("run_id", RUN_ID);
  const state = new RunState(privateFlow);
  const log = JsonlRunLog.create(path, RUN_ID);
  try {
    commitPrepared(
      log,
      state,
      prepareRunStarted(state, {
        runId: RUN_ID,
        flowFingerprint: flowFingerprint(loaded.compiled),
        initialPool: pool,
        startWiringOverride: Object.fromEntries(Object.keys(inputs).map((name) => [name, `\${input.${name}}`])),
        effectiveLlmByStaticPath: options.effective ?? {},
        numWorkers: options.numWorkers,
      }),
    );
    commitPrepared(log, state, prepareNodeStarted(state, START_ID));
    commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, structuredClone(inputs)));
    for (const nodeId of options.started) commitPrepared(log, state, prepareNodeStarted(state, nodeId));
    return log.nextSequence;
  } finally {
    log.close();
  }
}

/**
 * A memory log standing for a created durable one, which can refuse one append and counts its
 * closes. The run boundary never reads a log's path back, so it keeps none.
 */
class TrackingRunLog extends MemoryRunLog {
  readonly failure: unknown;
  readonly failWhen: (record: RunRecord) => boolean;
  closeCalls = 0;

  constructor(runId: string, failure: unknown, failWhen: (record: RunRecord) => boolean) {
    super(runId);
    this.failure = failure;
    this.failWhen = failWhen;
  }

  override append(record: RunRecord): RecordEnvelope {
    if (this.failWhen(record)) throw this.failure;
    return super.append(record);
  }

  override close(): void {
    this.closeCalls += 1;
    super.close();
  }
}

/** Run `fn` with `JsonlRunLog.create` returning `fake` for its path and run id. */
async function withCreatedLog<T>(fake: TrackingRunLog, path: string, fn: () => Promise<T>): Promise<T> {
  const create = spyOn(JsonlRunLog, "create").mockImplementation((given, runId) => {
    expect(given).toBe(path);
    expect(runId).toBe(fake.runId);
    return fake as unknown as JsonlRunLog;
  });
  try {
    return await fn();
  } finally {
    create.mockRestore();
  }
}

/** The process-loss marker a frozen log throws. */
class Crash extends Error {
  override name = "Crash";
}

/**
 * Run `fn` with every durable log refusing appends once it holds `k` records, as a crash after
 * record `k - 1` would; return the error `fn` rejects with, which must be the crash.
 */
async function crashAfter(k: number, fn: () => Promise<unknown>): Promise<void> {
  const append = JsonlRunLog.prototype.append;
  const crash = new Crash(`crash after record ${k - 1}`);
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (this.nextSequence >= k) throw crash;
    return append.call(this, record);
  };
  try {
    expect(await rejection(fn())).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
}

/** A model access over `hostEnv()` that records each config it builds from and answers `reply`. */
function scripted(built: Record<string, unknown>[], reply = "recorded reply"): ModelAccess {
  return {
    env: hostEnv(),
    createModel: (config) => {
      built.push({ ...config });
      return new ScriptedModel([textReply(reply)], {
        provider: config["provider"] as string,
        model: config["model"] as string,
      });
    },
  };
}

/** Leave the one-agent flow's `answer` in flight with a recorded pre-connection map. */
async function agentResumePrefix(path: string): Promise<Record<string, Record<string, unknown>>> {
  const loaded = await load(AGENT_FLOW);
  const effective = resolveLlmCascade(copyGraph(loaded.compiled), {}, {
    defaults: { provider: "openai", model: "recorded-model" },
  });
  writeNonterminalPrefix(path, loaded, { topic: "Ada" }, { started: ["answer"], numWorkers: 0, effective });
  return effective;
}

/** Whether `events` holds no RunStarted and no START NodeStarted, with RunResumed first. */
function expectResumedSegment(events: readonly RunEvent[]): void {
  expect(events[0]).toBeInstanceOf(RunResumed);
  expect(events.some((event) => event instanceof RunStarted)).toBe(false);
  expect(events.some((event) => event instanceof NodeStarted && event.nodeId === START_ID)).toBe(false);
}

test("test_final_run_boundary_signatures_and_result_fields_are_exact", () => {
  // The options' keys, exactly: a missing or extra key fails `tsc`.
  const options: Record<keyof ResumeFlowOptions, true> = {
    log: true,
    numWorkers: true,
    onEvent: true,
    onStream: true,
    memory: true,
    tools: true,
    skills: true,
    steering: true,
    apiKey: true,
    headers: true,
    context: true,
    signal: true,
    env: true,
    scheduler: true,
    random: true,
    models: true,
    checkLlmConfigs: true,
  };
  expect(Object.keys(options)).toHaveLength(17);
  expect(resumeFlow.length).toBe(2);
});

test("test_fresh_memory_success_returns_current_segment_and_canonical_input", async () => {
  const seen: RunEvent[] = [];

  const result = await run(await load(SUCCESS_FLOW), { n: "4" }, { log: null, runId: RUN_ID, onEvent: (e) => seen.push(e) });

  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ n: 4n });
  expect(result.output).toBe(8n);
  expect(result.failure).toBeNull();
  expect(result.runId).toBe(RUN_ID);
  expect(result.log).toBeNull();
  expect(result.events).toEqual(seen);
  expect(result.events[0]).toBeInstanceOf(RunStarted);
  expect(result.events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("test_fresh_durable_success_returns_path_and_releases_the_lock", async () => {
  const path = join(dir, "success.jsonl");

  const result = await run(await load(SUCCESS_FLOW), { n: 5 }, { log: path, runId: RUN_ID });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(10n);
  expect(result.runId).toBe(RUN_ID);
  expect(result.log).toBe(path);
  const records = durableRecords(path);
  expect(records[0]).toBeInstanceOf(RunStartedRecord);
  expect(records.at(-1)).toBeInstanceOf(NodeSucceededRecord);
});

test("test_nullable_specialized_list_survives_durable_run_and_terminal_resume", async () => {
  const path = join(dir, "nullable-list.jsonl");

  const fresh = await run(await load(NULLABLE_LIST_FLOW), { items: [1, null, 3] }, { log: path, runId: RUN_ID });

  expect(fresh.status).toBe("succeeded");
  expect(fresh.output).toEqual([1n, null, 3n]);
  const success = durableRecords(path).find(
    (record) => record instanceof NodeSucceededRecord && record.nodeId === "lifecycle_nullable_list",
  );
  expect(success).toEqual(
    new NodeSucceededRecord("lifecycle_nullable_list", "lifecycle_nullable_list", {
      kind: "list[int]",
      value: [1n, null, 3n],
    } as never),
  );
  const before = readFileSync(path);

  const resumed = await resume(await load(NULLABLE_LIST_FLOW), { log: path });

  expect(resumed.status).toBe(fresh.status);
  expect(resumed.output).toEqual(fresh.output);
  expect(resumed.events).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});

/** Check a failed result of `FAILURE_FLOW` whose log is `log`. */
function expectBoom(result: RunResult, log: string | null): void {
  expect(result.status).toBe("failed");
  expect(result.input).toEqual({});
  expect(result.output).toBeNull();
  expect(result.failure).not.toBeNull();
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.nodeId).toBe("fail");
  expect(result.failure!.message).toBe("Error: boom");
  expect(result.failure!.traceback).toContain("Error: boom");
  expect(result.runId).toBe(RUN_ID);
  expect(result.log).toBe(log);
  const nodeFailure = result.events.find((event) => event instanceof NodeFailed) as NodeFailed;
  expect(nodeFailure.failure).toBe(result.failure!);
  expect(result.events.at(-1)).toBeInstanceOf(RunFailed);
  expect((result.events.at(-1) as RunFailed).failure).toBe(result.failure!);
}

test("test_fresh_memory_flow_failure_returns_durable_diagnostics", async () => {
  const result = await run(await load(FAILURE_FLOW), {}, { log: null, runId: RUN_ID });

  expectBoom(result, null);
});

test("test_fresh_flow_failure_is_a_result_and_releases_the_lock", async () => {
  const path = join(dir, "failed.jsonl");

  const result = await run(await load(FAILURE_FLOW), {}, { log: path, runId: RUN_ID });

  expectBoom(result, path);
  const durable = durableRecords(path).at(-1);
  expect(durable).toBeInstanceOf(NodeFailedRecord);
  expect((durable as NodeFailedRecord).failure).toEqual(result.failure!);
});

test("test_terminal_resume_is_read_only_silent_and_unlocks[succeeded]", async () => {
  const path = join(dir, "terminal-succeeded.jsonl");
  const fresh = await run(await load(SUCCESS_FLOW), { n: 6 }, { log: path, runId: RUN_ID });
  const before = readFileSync(path);
  const seen: RunEvent[] = [];

  const resumed = await resume(await load(SUCCESS_FLOW), {
    log: path,
    onEvent: (event) => seen.push(event),
    env: noDefaultsEnv(),
  });

  expect(resumed.status).toBe("succeeded");
  expect(fresh.status).toBe("succeeded");
  expect(resumed.output).toBe(12n);
  expect(fresh.output).toBe(12n);
  expect(resumed.failure).toBeNull();
  expect(resumed.input).toEqual(fresh.input);
  expect(resumed.runId).toBe(RUN_ID);
  expect(resumed.log).toBe(path);
  expect(resumed.events).toEqual([]);
  expect(seen).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
  durableRecords(path);
});

test("test_terminal_resume_preserves_source_free_failure", async () => {
  const path = join(dir, "terminal-source-free-failure.jsonl");
  const freshFlow = sourceFreeFailureFlow();
  const fresh = await run(freshFlow, {}, { log: path, runId: RUN_ID });
  const before = readFileSync(path);

  expect(fresh.status).toBe("failed");
  expect(fresh.failure!.trace).toEqual([]);
  expect(freshFlow.diagnose(fresh.failure!).frames).toEqual([]);

  const resumedFlow = sourceFreeFailureFlow();
  const resumed = await resume(resumedFlow, { log: path });

  expect(resumed.failure).toEqual(fresh.failure!);
  expect(resumedFlow.diagnose(resumed.failure!)).toEqual(freshFlow.diagnose(fresh.failure!));
  expect(resumed.events).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});

test("test_terminal_resume_preserves_recorded_trace_across_origin_rebase", async () => {
  const loaded = await oldTerminalSkipFlow();
  const path = join(dir, "terminal-origin-rebase.jsonl");
  const fresh = await run(loaded, { score: 0.0 }, { log: path, runId: RUN_ID });
  const before = readFileSync(path);

  expect(fresh.status).toBe("failed");
  expect(fresh.failure!.code).toBe("graph.terminal_unreachable");
  expect(fresh.failure!.trace.length).toBeGreaterThan(0);
  const routed = durableRecords(path).filter((record) => record instanceof NodeRoutedRecord) as NodeRoutedRecord[];
  expect(routed).toHaveLength(1);
  expect(routed[0]!.terminalFailure).toEqual(fresh.failure!);

  const [rebasedCompiled, resolver] = rebaseFlowOrigins(loaded.compiled, (origin) => new OriginRef(`rebased:${origin.key}`), {
    resolver: loaded.originResolver,
  });
  const rebased = new ComposedFlow(rebasedCompiled, {
    version: loaded.version,
    name: loaded.name,
    description: loaded.description,
    originResolver: resolver,
  });
  expect(flowFingerprint(rebased.compiled)).toBe(flowFingerprint(loaded.compiled));

  const resumed = await resume(rebased, { log: path });

  expect(resumed.status).toBe("failed");
  expect(resumed.failure).toEqual(fresh.failure!);
  expect(resumed.failure!.trace).toEqual(fresh.failure!.trace);
  expect(resumed.events).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});

test("test_nonterminal_resume_commits_exact_retries_and_selected_workers[serial-override]", async () => {
  const path = join(dir, "resume-0.jsonl");
  const loaded = await load(PARALLEL_FLOW);
  const prefixLength = writeNonterminalPrefix(path, loaded, { n: 7n }, { started: ["a", "b"], numWorkers: 2 });
  const seen: RunEvent[] = [];

  const result = await resume(await load(PARALLEL_FLOW), { log: path, numWorkers: 0, onEvent: (e) => seen.push(e) });

  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ n: 7n });
  expect(result.output).toEqual({ a: 7n, b: 7n, c: 7n });
  expect(result.runId).toBe(RUN_ID);
  expect(result.log).toBe(path);
  expect(result.events).toEqual(seen);
  expectResumedSegment(result.events);
  expect(durableRecords(path)[prefixLength]).toEqual(new RunResumedRecord(["a", "b"], 0));
});

test("test_nonterminal_resume_restores_recorded_llm_without_reading_defaults", async () => {
  const path = join(dir, "agent-resume.jsonl");
  const effective = await agentResumePrefix(path);
  const built: Record<string, unknown>[] = [];

  const result = await resume(await load(AGENT_FLOW), {
    log: path,
    env: noDefaultsEnv(),
    models: scripted(built),
  });

  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ topic: "Ada" });
  expect(result.output).toBe("recorded reply");
  expect(built).toEqual([legacyConnectionConfig(effective["answer"]!)]);
  expectResumedSegment(result.events);
});

test("test_run_started_append_failure_prevents_execution_observation_and_closes", async () => {
  const path = join(dir, "start-append-failure.jsonl");
  const expected = new Error("RunStarted fsync failed");
  const fake = new TrackingRunLog(RUN_ID, expected, (record) => record instanceof RunStartedRecord);
  const seen: RunEvent[] = [];
  const flow = await load(ACTION_FLOW);

  const [error, calls] = await codeCalls(() =>
    withCreatedLog(fake, path, () =>
      rejection(run(flow, {}, { log: path, runId: RUN_ID, onEvent: (e) => seen.push(e) })),
    ),
  );

  expect(error).toBe(expected);
  expect(fake.records()).toEqual([]);
  expect(calls).toBe(0);
  expect(seen).toEqual([]);
  expect(fake.closeCalls).toBeGreaterThanOrEqual(1);
});

test("test_outcome_append_failure_leaves_started_work_recoverable_and_closes", async () => {
  const path = join(dir, "outcome-append-failure.jsonl");
  const expected = new Error("NodeSucceeded fsync failed");
  const fake = new TrackingRunLog(
    RUN_ID,
    expected,
    (record) => record instanceof NodeSucceededRecord && record.nodeId === "work",
  );
  const seen: RunEvent[] = [];
  const flow = await load(ACTION_FLOW);

  const [error, calls] = await codeCalls(() =>
    withCreatedLog(fake, path, () =>
      rejection(run(flow, {}, { log: path, runId: RUN_ID, onEvent: (e) => seen.push(e) })),
    ),
  );

  expect(error).toBe(expected);
  expect(calls).toBe(1);
  const records = fake.records().map((envelope) => envelope.record);
  expect(records.at(-1)).toEqual(new NodeStartedRecord("work"));
  expect(records.some((record) => record instanceof NodeSucceededRecord && record.nodeId === "work")).toBe(false);
  expect(seen.some((event) => event instanceof NodeStarted && event.nodeId === "work")).toBe(true);
  expect(seen.some((event) => event instanceof NodeSucceeded && event.nodeId === "work")).toBe(false);
  expect(fake.closeCalls).toBeGreaterThanOrEqual(1);
});

test("test_resume_callback_exception_closes_and_unlocks_nonterminal_log", async () => {
  const path = join(dir, "resume-callback-failure.jsonl");
  writeNonterminalPrefix(path, await load(ACTION_FLOW), {}, { started: ["work"], numWorkers: 0 });
  const expected = new Error("resume observer failed");
  const seen: RunEvent[] = [];
  const reject = (event: RunEvent) => {
    seen.push(event);
    if (event instanceof NodeStarted && event.nodeId === "work") throw expected;
  };
  const flow = await load(ACTION_FLOW);

  const [error, calls] = await codeCalls(() => rejection(resume(flow, { log: path, onEvent: reject })));

  expect(error).toBe(expected);
  expect(calls).toBe(0);
  expectResumedSegment(seen);
  const records = durableRecords(path);
  expect(records.at(-2)).toEqual(new NodeStartedRecord("work"));
  expect(records.at(-1)).toBeInstanceOf(RunStoppedRecord);
  expect((records.at(-1) as RunStoppedRecord).reason).toBe("callback");
  expect(records.filter((record) => record instanceof NodeStartedRecord && record.nodeId === "work")).toHaveLength(2);
  expect(records.some((record) => record instanceof RunResumedRecord)).toBe(true);
});

test("test_resume_hook_sees_recorded_configs_before_run_resumed", async () => {
  const path = join(dir, "hook.jsonl");
  const effective = await agentResumePrefix(path);
  const built: Record<string, unknown>[] = [];
  const seen: Record<string, Record<string, unknown>>[] = [];
  const hosts: (Record<string, unknown> | null)[] = [];
  const resumedAtHook: boolean[] = [];

  const result = await resume(await load(AGENT_FLOW), {
    log: path,
    models: scripted(built),
    checkLlmConfigs: (configs, { callTimeHost }) => {
      seen.push(structuredClone(configs));
      hosts.push(callTimeHost);
      resumedAtHook.push(readRunLog(path).some((envelope) => envelope.record instanceof RunResumedRecord));
      configs["answer"]!["model"] = "hook-edited";
      for (const key of Object.keys(configs)) delete configs[key];
    },
  });

  expect(result.status).toBe("succeeded");
  expect(seen).toEqual([legacyConnectionMap(effective)]);
  expect(hosts).toEqual([null]);
  expect((seen[0]!["answer"]!["_connection"] as Record<string, unknown>)["source"]).toBe("legacy_environment");
  expect(resumedAtHook).toEqual([false]);
  expect(built.map((config) => config["model"])).toEqual(["recorded-model"]);
});

test("test_resume_hook_failure_leaves_the_log_unchanged", async () => {
  const path = join(dir, "hook-failure.jsonl");
  await agentResumePrefix(path);
  const before = readFileSync(path);
  const expected = new Error("no key");

  const error = await rejection(
    resume(await load(AGENT_FLOW), {
      log: path,
      models: scripted([]),
      checkLlmConfigs: () => {
        throw expected;
      },
    }),
  );

  expect(error).toBe(expected);
  expect(readFileSync(path)).toEqual(before);
  const result = await resume(await load(AGENT_FLOW), { log: path, models: scripted([]) });
  expect(result.status).toBe("succeeded");
});

test("test_terminal_resume_never_calls_the_hook", async () => {
  const path = join(dir, "hook-terminal.jsonl");
  await agentResumePrefix(path);
  const finished = await resume(await load(AGENT_FLOW), { log: path, models: scripted([]) });
  expect(finished.status).toBe("succeeded");
  const before = readFileSync(path);
  const calls: unknown[] = [];

  const result = await resume(await load(AGENT_FLOW), {
    log: path,
    models: scripted([]),
    checkLlmConfigs: (configs) => {
      calls.push(configs);
    },
  });

  expect(result.status).toBe("succeeded");
  expect(calls).toEqual([]);
  expect(readFileSync(path)).toEqual(before);
});

test("test_callback_exception_drains_waiter_reraises_exactly_and_unlocks", async () => {
  const path = join(dir, "callback-interruption.jsonl");
  const { scheduler, entered, released } = blockingScheduler(Date.UTC(2026, 0, 1));
  const expected = new Error("observer rejected a durable outcome");
  const reject = (event: RunEvent) => {
    if (event instanceof NodeSucceeded && event.nodeId === "trigger") throw expected;
  };

  const error = await rejection(
    run(await load(CALLBACK_DRAIN_FLOW), {}, {
      log: path,
      runId: RUN_ID,
      numWorkers: 2,
      onEvent: reject,
      scheduler,
      tools: {
        trigger: noArgTool("trigger", async () => {
          await entered;
          return "triggered";
        }),
      },
    }),
  );

  expect(error).toBe(expected);
  expect(released()).toBe(true);
  const records = durableRecords(path);
  expect(records.some((record) => record instanceof NodeSucceededRecord && record.nodeId === "waiter")).toBe(false);
  expect(records.at(-1)).toBeInstanceOf(RunStoppedRecord);
});

test("test_callback_exception_drains_real_blocked_human_interaction_and_unlocks", async () => {
  const path = join(dir, "human-callback-interruption.jsonl");
  const expected = new Error("observer rejected while ask_user was blocked");
  let enter!: () => void;
  const entered = new Promise<void>((resolve) => {
    enter = resolve;
  });
  let interactionReleased = false;
  const answer = async (request: { nodeId: string }, ctx: { signal: AbortSignal }) => {
    expect(request.nodeId).toBe("human");
    enter();
    await new Promise((resolve) => ctx.signal.addEventListener("abort", resolve, { once: true }));
    interactionReleased = true;
    return "late answer";
  };
  const reject = (event: RunEvent) => {
    if (event instanceof NodeSucceeded && event.nodeId === "trigger") throw expected;
  };

  const error = await rejection(
    run(await load(HUMAN_DRAIN_FLOW), {}, {
      log: path,
      runId: RUN_ID,
      numWorkers: 2,
      onEvent: reject,
      tools: {
        ask_user: answer as never,
        trigger: noArgTool("trigger", async () => {
          await entered;
          return "triggered";
        }),
      },
    }),
  );

  expect(error).toBe(expected);
  expect(interactionReleased).toBe(true);
  const records = durableRecords(path);
  expect(records.some((record) => record instanceof NodeStartedRecord && record.nodeId === "human")).toBe(true);
  expect(records.some((record) => isOutcomeOf(record, ["human"]))).toBe(false);
  expect(records.at(-1)).toBeInstanceOf(RunStoppedRecord);
});

test("test_host_interruption_drains_then_durable_resume_retries_to_terminal", async () => {
  const path = join(dir, "host-interruption.jsonl");
  const { scheduler, entered, released } = blockingScheduler(Date.UTC(2026, 0, 1));
  const host = new AbortController();
  const interruption = new Error("host interrupted the run");
  let armed = true;
  const interrupter = noArgTool("interrupter", async (ctx) => {
    if (!armed) return "recovered";
    await entered;
    armed = false;
    host.abort(interruption);
    throw ctx.signal.reason;
  });
  const flow = await load(HOST_DRAIN_FLOW);

  const error = await rejection(
    run(flow, {}, { log: path, runId: RUN_ID, numWorkers: 2, signal: host.signal, scheduler, tools: { interrupter } }),
  );

  expect(error).toBe(interruption);
  expect(released()).toBe(true);
  const interrupted = durableRecords(path);
  expect(interrupted.some((record) => isOutcomeOf(record, ["interrupter", "waiter"]))).toBe(false);
  expect(interrupted.at(-1)).toBeInstanceOf(RunStoppedRecord);

  const later: Scheduler = { now: () => Date.UTC(2100, 0, 1), sleep: () => Promise.resolve() };
  const resumed = await resume(await load(HOST_DRAIN_FLOW), { log: path, scheduler: later, tools: { interrupter } });

  expect(resumed.status).toBe("succeeded");
  expect(resumed.output).toEqual({ interrupter: "recovered", waiter: null });
  expectResumedSegment(resumed.events);
  expect(durableRecords(path)).toContainEqual(new RunResumedRecord(["interrupter", "waiter"], 2));
});

test("a resume after a crash publishes RunResumed and NodeAttemptResumed with the counts", async () => {
  const path = join(dir, "crash.jsonl");
  const flow = await load(ACTION_FLOW);
  // Records 0-3 are RunStarted, START's two and work's NodeStarted: work's outcome is refused.
  await crashAfter(4, () => run(flow, {}, { log: path, runId: RUN_ID, numWorkers: 0 }));
  // RunResumed and work's second NodeStarted are kept; its outcome is refused again.
  const first: RunEvent[] = [];
  await crashAfter(6, () => resume(flow, { log: path, onEvent: (event) => first.push(event) }));
  expect(first.slice(0, 2)).toEqual([new RunResumed(), new NodeAttemptResumed({ nodeId: "work", interrupted: 1 })]);

  const seen: RunEvent[] = [];
  const result = await resume(flow, { log: path, onEvent: (event) => seen.push(event) });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("done");
  expect(result.events).toEqual(seen);
  expect(seen.slice(0, 3)).toEqual([
    new RunResumed(),
    new NodeAttemptResumed({ nodeId: "work", interrupted: 2 }),
    new NodeStarted({ nodeId: "work" }),
  ]);
  expect(result.interruptedAttempts).toEqual(new Map([["work", 2]]));
});

test("a resume with an aborted signal rejects before the lock is taken", async () => {
  const path = join(dir, "aborted.jsonl");
  writeNonterminalPrefix(path, await load(ACTION_FLOW), {}, { started: ["work"], numWorkers: 0 });
  const before = readFileSync(path);
  const stop = new AbortController();
  stop.abort(new Error("stopped before the resume"));
  // Held, so a resume that took the lock would fail as busy instead.
  const held = JsonlRunLog.resume(path);
  try {
    const error = await rejection(resume(await load(ACTION_FLOW), { log: path, signal: stop.signal }));
    expect(error).toBe(stop.signal.reason);
  } finally {
    held.close();
  }
  expect(readFileSync(path)).toEqual(before);
});
