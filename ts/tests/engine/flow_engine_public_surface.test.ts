/**
 * The public direct-engine contract: a port of `tests/engine/test_flow_engine_public_surface.py`.
 * A direct caller prepares and commits `RunStarted` first, then lends that state and log to
 * `FlowEngine`, which drives only new work and borrows the log.
 *
 * Ported (adapted), on `graph()` and `FuncNode` for Python's loaded CODE flows:
 * `test_growth_builder_is_a_required_constructor_dependency`,
 * `test_unprepared_state_is_rejected_before_any_node_executes`,
 * `test_mismatched_state_and_log_are_rejected_before_execution`,
 * `test_mismatched_durable_sequence_is_rejected_before_execution` and
 * `test_mismatched_durable_worker_count_is_rejected_before_execution`; each names its `RangeError`
 * and full message, where Python matches a part of its `ValueError`.
 *
 * TS-only: the constructor's other checks, the run's tool set bound to the run id and signal, a
 * terminal state returning no events, unfinished starts refused, and a second `run()` refused.
 *
 * Not ported: `test_constructor_is_exactly_the_recorded_state_log_surface` (`inspect` and type
 * hints; TS type-checks the options) and `test_checkpoint_era_engine_methods_are_absent` (TS never
 * had them).
 */

import { expect, test } from "bun:test";

import { CompiledFlow, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { FlowEngine, type FlowEngineOptions } from "../../src/runtime/engine.ts";
import { MemoryRunLog, NodeStartedRecord } from "../../src/runtime/log.ts";
import { commitPrepared, prepareNodeFailed, prepareNodeStarted, RunState } from "../../src/runtime/state.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { engine, FuncNode, graph, nodeFailure, noGrowth, RUN_ID, rejected } from "./run_state_fixtures.ts";

/** The invocations of the action flow's node. */
const probe: string[] = [];

/** Return `START -> work -> END`, whose `work` records each invocation in `probe`. */
function actionFlow(): CompiledFlow {
  const work = new FuncNode("work", () => {
    probe.push("ran");
    return "done";
  });
  return graph([work], [
    [START_ID, "work"],
    ["work", END_ID],
  ], { outputs: [new FlowOutput("output", "${work.output}")] });
}

/** Return the run's rejection message after checking its class. */
async function refused(run: Promise<unknown>, kind: ErrorConstructor): Promise<string> {
  const error = await rejected(run);
  expect(error).toBeInstanceOf(kind);
  return (error as Error).message;
}

test("growth builder is a required constructor dependency", () => {
  const { state, log } = engine(actionFlow());
  expect(() => new FlowEngine(state, log, { numWorkers: 0 } as FlowEngineOptions)).toThrow(
    new TypeError("grow_builder must be callable"),
  );
});

test("unprepared state is rejected before any node executes", async () => {
  probe.length = 0;
  const state = new RunState(actionFlow());
  const log = new MemoryRunLog(RUN_ID);
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("FlowEngine requires an initialized RunState");
  expect(probe).toEqual([]);
  expect(log.nextSequence).toBe(0);
});

test("mismatched state and log are rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  const other = new MemoryRunLog("another-run");
  const run = new FlowEngine(state, other, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/log run_id mismatch: state 'run-123', log 'another-run'");
  expect(probe).toEqual([]);
  expect(log.nextSequence).toBe(1);
  expect(other.nextSequence).toBe(0);
});

test("mismatched durable sequence is rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  log.append(new NodeStartedRecord("unapplied"));
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/log sequence mismatch: state 1, log 2");
  expect(probe).toEqual([]);
  expect(state.nextSequence).toBe(1);
  expect(log.nextSequence).toBe(2);
});

test("mismatched durable worker count is rejected before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow(), { numWorkers: 1 });
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe("run-state/engine worker mismatch: state 1, engine 0");
  expect(probe).toEqual([]);
  expect(state.numWorkers).toBe(1);
  expect(state.nextSequence).toBe(1);
  expect(log.nextSequence).toBe(1);
});

test("the constructor checks its arguments", () => {
  const { state, log } = engine(actionFlow());
  const build = (options: Record<string, unknown>, stateArg: unknown = state, logArg: unknown = log) => () =>
    new FlowEngine(stateArg as RunState, logArg as MemoryRunLog, {
      growBuilder: noGrowth,
      numWorkers: 0,
      ...options,
    } as FlowEngineOptions);

  expect(build({}, {})).toThrow(new TypeError("FlowEngine expects a RunState"));
  expect(build({}, state, { runId: RUN_ID, nextSequence: 1 })).toThrow(new TypeError("FlowEngine expects a RunLog"));
  expect(build({ numWorkers: 1.5 })).toThrow(new TypeError("num_workers must be a non-negative integer"));
  expect(build({ numWorkers: "1" })).toThrow(new TypeError("num_workers must be a non-negative integer"));
  expect(build({ numWorkers: -1 })).toThrow(new RangeError("num_workers must be a non-negative integer"));
  expect(build({ onEvent: 1 })).toThrow(new TypeError("on_event must be callable or null"));
  expect(build({ onStream: 1 })).toThrow(new TypeError("on_stream must be callable or null"));
  expect(build({ memoryContextOpener: 1 })).toThrow(new TypeError("memory_context_opener must be callable or null"));
  expect(build({ skills: {} })).toThrow(new TypeError("skills must be a map or null"));
  expect(build({ steering: {} })).toThrow(
    new TypeError("steering must provide interruptSignal(nodeId) and drain(nodeId), or be null"),
  );
});

test("the run's tool set is bound to the run id and the run signal", () => {
  const { state, log } = engine(actionFlow());
  const tools = new ToolSet({});
  const built = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0, tools });
  expect(built.tools).toBeInstanceOf(ToolSet);
  expect(built.tools).not.toBe(tools);
  expect(built.tools.runId).toBe(RUN_ID);
  expect(built.tools.signal).toBe(built.signal);
  expect(built.signal.aborted).toBe(false);
  expect(engine(actionFlow()).tools.runId).toBe(RUN_ID);
});

test("a terminal state returns no events, and a second run is refused", async () => {
  const { state, log } = engine(actionFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeFailed(state, nodeFailure(START_ID)));
  expect(state.result).not.toBeNull();
  const size = log.nextSequence;
  const terminal = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 });
  expect(await terminal.run()).toEqual([]);
  expect(log.nextSequence).toBe(size);
  expect(await refused(terminal.run(), Error)).toBe("FlowEngine.run() was already called");
});

test("unfinished starts must be requeued before execution", async () => {
  probe.length = 0;
  const { state, log } = engine(actionFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const run = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 }).run();
  expect(await refused(run, RangeError)).toBe(
    "FlowEngine requires unfinished starts to be requeued with RunResumed before execution",
  );
  expect(probe).toEqual([]);
  expect(log.nextSequence).toBe(2);
});
