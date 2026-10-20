/**
 * The TOOL node: a port of `tests/engine/test_tool_retries.py`.
 *
 * TS has no `_sleep` or `_jitter` to patch: the node waits on `caps.scheduler` with `caps.signal`
 * and draws from `caps.random`. The direct rows run `node.run` on a virtual scheduler that records
 * each sleep, in milliseconds where Python records seconds, with a draw of 1 as Python's `delays`
 * fixture. JS has no `TimeoutError`, so the rows throw their own.
 *
 * Ported: `test_tool_node_rejects_a_bad_retries_count`; `test_a_huge_failure_count_gets_the_cap`
 * (`backoff` takes the draw as an argument, and a failure count of 10 ** 6 gets the cap too);
 * `test_forward_without_steps_retries_in_memory`,
 * `test_a_resumed_start_continues_the_numbering_and_rewaits` and
 * `test_validate_checks_arguments_without_calling_the_tool`.
 *
 * Ported (adapted), through `FlowEngine` (the `engine()` fixture, run id `run-123`) on START ->
 * TOOL `call` -> END in place of `run_flow`, reading the events and the log's `node_step` records:
 * `test_retries_zero_fails_as_before_and_records_no_step`,
 * `test_two_failures_then_success_records_each_attempt` (serial and two workers; Python's third
 * row, the run's default worker count, has no engine counterpart and is dropped),
 * `test_jitter_draws_from_half_to_the_full_delay`,
 * `test_backoff_stops_at_the_cap`, `test_retry_after_replaces_the_backoff_up_to_the_cap` (the
 * draw throws, so none is taken), `test_an_output_contract_failure_is_not_retried`,
 * `test_exhausted_retries_list_every_attempt` (its `run_flow` half: the failure's code, cause type
 * and message) and `test_a_stop_during_the_backoff_wait_stops_the_run` (serial and two workers,
 * Python's third row, the default worker count, dropped as above; the scheduler's `sleep` aborts
 * the host's controller, then delegates, so the sleep rejects at once; `run()` rejects with the
 * host's reason, the steps are exactly one `failed`, and no `NodeFailed` is emitted). TS-only
 * through the engine: a `Date` result fails `contract.output`.
 *
 * Ported (adapted), each through `node.run` with recorded steps in place of `run_flow`:
 * `test_permanent_error_on_the_first_attempt_fails_unchanged` (the error is the same object),
 * `test_permanent_error_on_the_second_attempt_stops_the_retries`,
 * `test_an_error_both_permanent_and_transient_is_not_retried` (an error carrying both brands),
 * `test_exhausted_retries_list_every_attempt` (the `cause` is the last error),
 * `test_a_failure_after_100_attempts_lists_every_attempt` and `test_ten_attempts_are_all_listed`
 * (one row), `test_odd_error_text_is_recorded_safely` and `test_a_long_error_type_name_is_capped`
 * (one row; `<BadStr object>` is an error whose `message` getter throws, and the long type name is
 * the error's `name`). `test_a_host_stop_from_the_tool_is_neither_retried_nor_recorded`: the tool
 * aborts the node's signal, then throws, and that error is rethrown as is.
 *
 * Changed: the `1.0` row is `1.5`, since `1.0` is the integer `1` in JS; a negative count
 * throws `RangeError` where Python raises `ValueError`; each row checks the whole message, not
 * just `'call'`. `test_an_invalid_argument_is_not_retried` runs through the engine and passes a
 * value whose type doesn't fit the schema: the failure is `node.execute` with cause type
 * `TypeCheckError` (Python's is pydantic's `ValidationError` for `gt=0`), and nothing is called;
 * `test_an_unknown_argument_is_not_retried` rejects with
 * `RangeError` where Python raises `ValueError`. `test_an_awaitable_from_a_function_tool_is_not_retried`
 * is "an execute returning a rejected promise is retried like any error": TS awaits every result.
 *
 * TS-only: `hostToolCall()` names the tool, the inputs, their proven types and the output type;
 * the node's kind, title and traits read on an instance (`needsTools`, `needsSteps`). `backoff`
 * directly: the doubling and the jitter's upper half, the cap, `retryAfter` taking no draw, and a
 * zero base giving 0, never `NaN`. `ToolAttemptsFailed` is an `Error` named `ToolAttemptsFailed`
 * with its `cause`. `backoff` ignores a `retryAfter` that isn't a finite number of seconds, at
 * least 0, that throws, or that sits on an unbranded error. The direct invalid-argument row:
 * `validate`'s `TypeCheckError` rejects `node.run` before any call. A `ToolCheckError` from
 * the tool is not retried, also when branded transient too (check wins). A tool throwing a
 * `CapabilityFault` is called once and records nothing (Python's `_CapabilityFault` shares the
 * host stop's `except`). A thrown non-`Error` records its `jsTypeName` and text, and propagates,
 * or becomes the `cause`, unwrapped. A result is taken in as a yian value (`5` is `5n`), and one
 * that isn't (a `Date`) rejects with `HostOutputError` after one call.
 *
 * Not ported: `test_the_real_sleep_raises_at_once_on_a_set_event`,
 * `test_the_real_sleep_waits_without_an_event`, `test_a_wait_longer_than_timeout_max_waits_in_chunks`
 * and `test_a_stop_between_timeout_max_chunks_ends_the_wait`: TS has no `_sleep`; the node sleeps on
 * its scheduler, whose abort and chunks `platform.test.ts` covers for `realScheduler`.
 */

import { expect, test } from "bun:test";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { current, forget, install } from "../../src/defaults.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { NodeFailed, type RunEvent, RunFailed, RunSucceeded } from "../../src/events.ts";
import { CapabilityFault, HostOutputError, NodeKind, Output } from "../../src/nodes/base.ts";
import { NodeSteps, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { backoff, ToolAttemptsFailed, ToolNode } from "../../src/nodes/tool/node.ts";
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";
import { NodeStepRecord } from "../../src/runtime/log.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import {
  PermanentToolError,
  TOOL_ERROR,
  ToolCheckError,
  type ToolErrorKind,
  TransientToolError,
} from "../../src/tools/errors.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { typeFor } from "../../src/typesys/index.ts";
import { TypeCheckError, type ValueRecord } from "../../src/typesys/values.ts";
import { engine, graph, rejected } from "./run_state_fixtures.ts";

/** JS has no built-in `TimeoutError`; the rows throw this one, as Python's throw its own. */
class TimeoutError extends Error {
  override name = "TimeoutError";
}

/** What a tool's context showed for one call. */
interface SeenCall {
  callId: string;
  attempt: number;
  node: string;
  signal: AbortSignal;
}

/**
 * A `FunctionTool` named `flaky` that throws the scripted errors in order, then returns `value`;
 * `seen` records each call's context. (A `FunctionTool` is frozen, so this holds one.)
 */
class FlakyTool {
  readonly seen: SeenCall[] = [];
  readonly tool: FunctionTool;

  constructor(errors: unknown[], value: unknown = "ok") {
    const script = [...errors];
    this.tool = new FunctionTool({
      name: "flaky",
      description: "Do one flaky thing.",
      parametersJsonSchema: { type: "object", properties: {} },
      execute: (_args, ctx: ToolContext) => {
        this.seen.push({ callId: ctx.callId, attempt: ctx.attempt, node: ctx.node, signal: ctx.signal });
        if (script.length > 0) throw script.shift();
        return value;
      },
    });
  }
}

/** A virtual scheduler that records each sleep it is asked for, in milliseconds. */
function recordingScheduler(): { scheduler: Scheduler; sleeps: number[] } {
  const inner = virtualScheduler();
  const sleeps: number[] = [];
  const scheduler: Scheduler = {
    now: () => inner.now(),
    sleep(ms, signal) {
      sleeps.push(ms);
      return inner.sleep(ms, signal);
    },
  };
  return { scheduler, sleeps };
}

/**
 * Run a TOOL node calling `tool` directly, with `retries`, a recording scheduler, a draw of 1 and
 * a live signal unless given; return the outcome (or the error) with the sleeps and the steps
 * recorded through `NodeSteps(1, [], ...)` unless `steps` is `null` (no steps).
 */
async function runDirect(
  tool: FunctionTool,
  options: { retries?: number; inputs?: Record<string, unknown>; signal?: AbortSignal; steps?: null } = {},
): Promise<{ result: unknown; error: unknown; sleeps: number[]; recorded: ValueRecord[] }> {
  const { retries = 1, inputs = {}, signal = new AbortController().signal } = options;
  const { scheduler, sleeps } = recordingScheduler();
  const recorded: ValueRecord[] = [];
  const steps = options.steps === null ? undefined : new NodeSteps(1, [], (step) => recorded.push(step));
  const node = new ToolNode("call", { toolId: tool.name, retries });
  const tools = new ToolSet({ [tool.name]: tool });
  try {
    const result = await node.run(inputs, { tools, scheduler, random: () => 1, signal, steps });
    return { result, error: undefined, sleeps, recorded };
  } catch (error) {
    return { result: undefined, error, sleeps, recorded };
  }
}

/** A jitter draw that must not be taken. */
function noDraw(): number {
  throw new Error("drew a jitter");
}

/** Run `body` with the retry base and cap set, restoring the defaults once it settles. */
async function withDelays<T>(base: number, cap: number, body: () => T | Promise<T>): Promise<T> {
  install({ ...current(), toolRetryDelaySeconds: base, toolRetryMaxDelaySeconds: cap });
  try {
    return await body();
  } finally {
    forget();
  }
}

/** The steps, failure and sleeps of one engine run, and the error `run()` rejected with. */
interface EngineRun {
  events: readonly RunEvent[];
  steps: ValueRecord[];
  sleeps: number[];
  error: unknown;
}

/**
 * Run START -> TOOL `call` -> END through the engine, the tool set holding `tool`, on a recording
 * scheduler (or `options.scheduler`) with a draw of 1 unless given; END outputs `${call.output}`.
 */
async function runEngine(
  node: ToolNode,
  tool: FunctionTool,
  options: {
    numWorkers?: number;
    random?: () => number;
    signal?: AbortSignal;
    scheduler?: Scheduler;
    wiring?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  } = {},
): Promise<EngineRun> {
  const { numWorkers = 0, random = () => 1, signal, wiring } = options;
  const recording = recordingScheduler();
  const scheduler = options.scheduler ?? recording.scheduler;
  const flow = graph([node], [[START_ID, node.id], [node.id, END_ID]], {
    outputs: [new FlowOutput("result", `\${${node.id}.output}`)],
    ...(wiring === undefined ? {} : { wiring }),
  });
  const events: RunEvent[] = [];
  const built = engine(flow, {
    tools: new ToolSet({ [tool.name]: tool }),
    scheduler,
    random,
    numWorkers,
    ...(signal === undefined ? {} : { signal }),
    onEvent: (event) => {
      events.push(event);
    },
  });
  let error: unknown;
  try {
    await built.run();
  } catch (caught) {
    error = caught;
  }
  const steps = built.log
    .records()
    .map((envelope) => envelope.record)
    .filter((record) => record instanceof NodeStepRecord)
    .map((record) => (record as NodeStepRecord).step);
  return { events, steps, sleeps: recording.sleeps, error };
}

/** The run's failure: the `RunFailed` event's. */
function runFailure(events: readonly RunEvent[]): Failure {
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  return (last as RunFailed).failure;
}

/** The run's output: the `RunSucceeded` event's. */
function runOutput(events: readonly RunEvent[]): unknown {
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

test("tool node rejects a bad retries count", () => {
  const rows: [unknown, typeof TypeError | typeof RangeError, string][] = [
    ["2", TypeError, '"2"'],
    [true, TypeError, "true"],
    [1.5, TypeError, "1.5"],
    [-1, RangeError, "-1"],
  ];
  for (const [retries, error, got] of rows) {
    const build = () => new ToolNode("call", { toolId: "flaky", retries: retries as number });
    expect(build).toThrow(error);
    expect(build).toThrow(`node 'call': retries must be a non-negative int, got ${got}`);
  }
});

test("hostToolCall names the tool, the inputs, their proven types and the output type", () => {
  const int = typeFor("int", new Map());
  const node = new ToolNode("call", { toolId: "flaky" });
  node.params = [new Parameter("symbol"), new Parameter("qty")];
  node.inputTypes = new Map([["qty", int]]);
  node.outputContract = new ValueContract(int);
  const call = node.hostToolCall();
  expect(call.toolId).toBe("flaky");
  expect(call.inputs).toEqual(["symbol", "qty"]);
  expect([...call.inputTypes]).toEqual([["qty", int]]);
  expect(call.outputType).toBe(int);

  const bare = new ToolNode("bare", { toolId: "x" }).hostToolCall();
  expect(bare.inputs).toEqual([]);
  expect(bare.inputTypes.size).toBe(0);
  expect(bare.outputType).toBeNull();
});

test("a TOOL node's kind, title and traits", () => {
  const node = new ToolNode("call", { toolId: "flaky", title: "Call" });
  expect(node.kind).toBe(NodeKind.TOOL);
  expect(node.title).toBe("Call");
  expect(node.needsTools).toBe(true);
  expect(node.needsSteps).toBe(true);
});

test("the backoff doubles from the base and draws from its upper half", () => {
  const error = new TimeoutError("a");
  expect([1, 2, 3].map((failure) => backoff(failure, error, () => 1))).toEqual([1, 2, 4]);
  expect([1, 2].map((failure) => backoff(failure, error, () => 0))).toEqual([0.5, 1.0]);
});

test("the backoff stops at the cap", async () => {
  const delays = await withDelays(10, 15, () =>
    [1, 2, 3].map((failure) => backoff(failure, new TimeoutError(), () => 1)),
  );
  expect(delays).toEqual([10, 15, 15]);
});

test("retryAfter replaces the backoff up to the cap and takes no draw", () => {
  expect(backoff(1, new TransientToolError("busy", { retryAfter: 7 }), noDraw)).toBe(7);
  expect(backoff(1, new TransientToolError("busy", { retryAfter: 100 }), noDraw)).toBe(30);
});

test("a retryAfter that isn't a finite number of seconds, or on an unbranded error, gets the backoff", () => {
  const transient = (retryAfter: unknown) => ({ [TOOL_ERROR]: ["transient"], retryAfter });
  const throwing = {
    [TOOL_ERROR]: ["transient"],
    get retryAfter(): number {
      throw new Error("no retryAfter");
    },
  };
  const rows: unknown[] = [
    transient(-1),
    transient(Number.NaN),
    transient(Number.POSITIVE_INFINITY),
    transient("5"),
    throwing,
    { retryAfter: 7 },
  ];
  for (const error of rows) expect(backoff(1, error, () => 1)).toBe(1);
});

test("a huge failure count gets the cap", () => {
  expect(backoff(2000, new TimeoutError(), () => 1)).toBe(30);
  expect(backoff(10 ** 6, new TimeoutError(), () => 1)).toBe(30);
});

test("a zero base gives a zero backoff, never NaN", async () => {
  const delays = await withDelays(0, 30, () =>
    [1, 10 ** 6].map((failure) => backoff(failure, new TimeoutError(), () => 1)),
  );
  expect(delays).toEqual([0, 0]);
});

test("ToolAttemptsFailed is an Error named ToolAttemptsFailed with its cause", () => {
  const cause = new TimeoutError("b");
  const error = new ToolAttemptsFailed("tool 'flaky' failed after 2 attempts", { cause });
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("ToolAttemptsFailed");
  expect(error.message).toBe("tool 'flaky' failed after 2 attempts");
  expect(error.cause).toBe(cause);
});

test("forward without steps retries in memory", async () => {
  const flaky = new FlakyTool([new TimeoutError("a"), new TimeoutError("b")]);

  const { result, sleeps } = await runDirect(flaky.tool, { retries: 2, steps: null });

  expect(result).toEqual(new Output("ok"));
  expect(flaky.seen.map((ctx) => ctx.attempt)).toEqual([1, 2, 3]);
  expect(sleeps).toEqual([1000, 2000]);
});

test("a resumed start continues the numbering and re-waits", async () => {
  const recorded: ValueRecord[] = [];
  const prior: ValueRecord[] = [
    { event: "failed", attempt: 1n, error: { type: "TimeoutError", message: "a" }, retry_in: 1 },
    { event: "started", attempt: 2n },
    { event: "failed", attempt: 2n, error: { type: "TimeoutError", message: "b" }, retry_in: 2 },
  ];
  // Starts: the first evaluation, a crash inside attempt 3's start, and this one.
  const steps = new NodeSteps(3, prior, (step) => recorded.push(step));
  const timeout = new TimeoutError("d");
  const flaky = new FlakyTool([timeout]);
  const { scheduler, sleeps } = recordingScheduler();
  const node = new ToolNode("call", { toolId: "flaky", retries: 2 });
  const tools = new ToolSet({ flaky: flaky.tool });

  const error = await rejected(node.run({}, { tools, scheduler, random: () => 1, steps }));

  expect(flaky.seen.map((ctx) => ctx.attempt)).toEqual([4]);
  expect(sleeps).toEqual([2000]);
  expect(recorded).toEqual([]);
  expect(error).toBeInstanceOf(ToolAttemptsFailed);
  expect((error as Error).message).toBe(
    "tool 'flaky' failed after 4 attempts: attempt 1: TimeoutError: a; " +
      "attempt 2: TimeoutError: b; attempt 3: interrupted; attempt 4: TimeoutError: d",
  );
  expect((error as Error).cause).toBe(timeout);
});

test("a permanent error on the first attempt fails unchanged", async () => {
  const permanent = new PermanentToolError("no");
  const flaky = new FlakyTool([permanent]);

  const { error, sleeps, recorded } = await runDirect(flaky.tool, { retries: 3 });

  expect(error).toBe(permanent);
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
  expect(recorded).toEqual([]);
});

test("a permanent error on the second attempt stops the retries", async () => {
  const permanent = new PermanentToolError("no");
  const flaky = new FlakyTool([new TimeoutError("a"), permanent]);

  const { error, sleeps } = await runDirect(flaky.tool, { retries: 3 });

  expect(error).toBeInstanceOf(ToolAttemptsFailed);
  expect((error as Error).message).toBe(
    "tool 'flaky' failed after 2 attempts: attempt 1: TimeoutError: a; attempt 2: PermanentToolError: no",
  );
  expect((error as Error).cause).toBe(permanent);
  expect(flaky.seen).toHaveLength(2);
  expect(sleeps).toEqual([1000]);
});

test("an error both permanent and transient is not retried", async () => {
  class Both extends PermanentToolError {
    override name = "Both";
    override readonly [TOOL_ERROR]: readonly ToolErrorKind[] = Object.freeze(["permanent", "transient"]);
  }
  const both = new Both("both");
  const flaky = new FlakyTool([both]);

  const { error } = await runDirect(flaky.tool, { retries: 3 });

  expect(error).toBe(both);
  expect(flaky.seen).toHaveLength(1);
});

test("a ToolCheckError is not retried", async () => {
  const check = new ToolCheckError("bad result");
  const flaky = new FlakyTool([check]);

  const { error, sleeps } = await runDirect(flaky.tool, { retries: 3 });

  expect(error).toBe(check);
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
});

test("a check error also branded transient is not retried", async () => {
  class TransientCheck extends ToolCheckError {
    override readonly [TOOL_ERROR]: readonly ToolErrorKind[] = Object.freeze(["check", "transient"]);
  }
  const check = new TransientCheck("bad result");
  const flaky = new FlakyTool([check]);

  const { error, sleeps, recorded } = await runDirect(flaky.tool, { retries: 2 });

  expect(error).toBe(check);
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
  expect(recorded).toEqual([]);
});

test("an invalid argument is not retried", async () => {
  const calls: unknown[] = [];
  const counted = new FunctionTool({
    name: "flaky",
    description: "Count one call.",
    parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } }, required: ["n"] },
    execute: (args) => {
      calls.push(args.n);
      return "ok";
    },
  });

  const { error, sleeps } = await runDirect(counted, { retries: 3, inputs: { n: "x" } });

  expect(error).toBeInstanceOf(TypeCheckError);
  expect(calls).toEqual([]);
  expect(sleeps).toEqual([]);
});

test("an unknown argument is not retried", async () => {
  const flaky = new FlakyTool([]);

  const { error, sleeps } = await runDirect(flaky.tool, { retries: 3, inputs: { bogus: 1 } });

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("unsupported argument");
  expect(flaky.seen).toEqual([]);
  expect(sleeps).toEqual([]);
});

test("validate checks arguments without calling the tool", async () => {
  const flaky = new FlakyTool([]);
  const tools = new ToolSet({ flaky: flaky.tool });

  await tools.validate("flaky", {});
  await expect(tools.validate("flaky", { bogus: 1 })).rejects.toThrow("unsupported argument");

  expect(flaky.seen).toEqual([]);
});

test("a host stop from the tool is neither retried nor recorded", async () => {
  const controller = new AbortController();
  const thrown = new TimeoutError("stopped");
  let calls = 0;
  const stopping = new FunctionTool({
    name: "flaky",
    description: "Stop the run, then fail.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => {
      calls += 1;
      controller.abort(new Error("host stop"));
      throw thrown;
    },
  });

  const { error, sleeps, recorded } = await runDirect(stopping, { retries: 2, signal: controller.signal });

  expect(error).toBe(thrown);
  expect(calls).toBe(1);
  expect(sleeps).toEqual([]);
  expect(recorded).toEqual([]);
});

test("a capability fault from the tool is neither retried nor recorded", async () => {
  const fault = new CapabilityFault(new Error("log down"));
  const flaky = new FlakyTool([fault]);

  const { error, sleeps, recorded } = await runDirect(flaky.tool, { retries: 2 });

  expect(error).toBe(fault);
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
  expect(recorded).toEqual([]);
});

test("an execute returning a rejected promise is retried like any error", async () => {
  let calls = 0;
  const later = new FunctionTool({
    name: "flaky",
    description: "Fail once, later.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => {
      calls += 1;
      return calls === 1 ? Promise.reject(new TimeoutError("a")) : Promise.resolve("x");
    },
  });

  const { result, sleeps } = await runDirect(later, { retries: 3 });

  expect(result).toEqual(new Output("x"));
  expect(calls).toBe(2);
  expect(sleeps).toEqual([1000]);
});

test("exhausted retries list every attempt", async () => {
  const last = new TimeoutError("b");
  const flaky = new FlakyTool([new TimeoutError("a"), last]);

  const { error } = await runDirect(flaky.tool, { retries: 1 });

  expect(error).toBeInstanceOf(ToolAttemptsFailed);
  expect((error as Error).message).toBe(
    "tool 'flaky' failed after 2 attempts: attempt 1: TimeoutError: a; attempt 2: TimeoutError: b",
  );
  expect((error as Error).cause).toBe(last);
  expect(flaky.seen).toHaveLength(2);
});

test("a failure after 100 attempts, or ten, lists every attempt", async () => {
  for (const count of [100, 10]) {
    const numbers = Array.from({ length: count }, (_, index) => index + 1);
    const flaky = new FlakyTool(numbers.map((n) => new TimeoutError(`e${n}`)));

    const { error } = await runDirect(flaky.tool, { retries: count - 1 });

    expect(flaky.seen).toHaveLength(count);
    const every = numbers.map((n) => `attempt ${n}: TimeoutError: e${n}`).join("; ");
    expect((error as Error).message).toBe(`tool 'flaky' failed after ${count} attempts: ${every}`);
  }
});

test("odd error text is recorded safely", async () => {
  class BadStr extends Error {
    override name = "BadStr";
  }
  const bad = new BadStr();
  Object.defineProperty(bad, "message", {
    get() {
      throw new Error("no text");
    },
  });
  const longName = new Error("boom");
  longName.name = "E".repeat(5000);
  const rows: [unknown, { type: string; message: string }][] = [
    [bad, { type: "BadStr", message: "<BadStr object>" }],
    [new TimeoutError("a\uD800b"), { type: "TimeoutError", message: "a?b" }],
    [new TimeoutError("x".repeat(10_000)), { type: "TimeoutError", message: "x".repeat(199) + "…" }],
    [longName, { type: "E".repeat(199) + "…", message: "boom" }],
  ];
  for (const [thrown, shown] of rows) {
    const { result, recorded } = await runDirect(new FlakyTool([thrown]).tool);

    expect(result).toEqual(new Output("ok"));
    const [failed, started] = recorded;
    expect(failed!.error).toEqual(shown);
    expect(started).toEqual({ event: "started", attempt: 2n });
  }
});

test("a thrown non-Error is recorded by its type and text, and propagates unwrapped", async () => {
  const once = await runDirect(new FlakyTool(["boom"]).tool);
  expect(once.recorded[0]!.error).toEqual({ type: "string", message: "boom" });

  const alone = await runDirect(new FlakyTool(["boom"]).tool, { retries: 0 });
  expect(alone.error).toBe("boom");

  const twice = await runDirect(new FlakyTool(["a", "b"]).tool, { retries: 1 });
  expect(twice.error).toBeInstanceOf(ToolAttemptsFailed);
  expect((twice.error as Error).message).toBe(
    "tool 'flaky' failed after 2 attempts: attempt 1: string: a; attempt 2: string: b",
  );
  expect((twice.error as Error).cause).toBe("b");
});

test("a tool's result is taken in as a yian value", async () => {
  const { result } = await runDirect(new FlakyTool([], 5).tool);
  expect(result).toEqual(new Output(5n));
});

test("a result that isn't a yian value is refused after one call, never retried", async () => {
  const flaky = new FlakyTool([], { a: new Date(0) });

  const { error, sleeps } = await runDirect(flaky.tool, { retries: 2 });

  expect(error).toBeInstanceOf(HostOutputError);
  expect((error as Error).message).toBe("output at a: Date is not a yian value");
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
});

test("retries zero fails as before and records no step", async () => {
  const flaky = new FlakyTool([new TimeoutError("a")]);

  const { events, steps, sleeps } = await runEngine(new ToolNode("call", { toolId: "flaky" }), flaky.tool);

  const failure = runFailure(events);
  expect(failure.code).toBe("node.execute");
  expect(failure.message).toBe("a");
  expect(failure.causeType).toBe("TimeoutError");
  expect(flaky.seen).toHaveLength(1);
  expect(steps).toEqual([]);
  expect(sleeps).toEqual([]);
});

for (const numWorkers of [0, 2]) {
  test(`two failures then success records each attempt (numWorkers ${numWorkers})`, async () => {
    const flaky = new FlakyTool([new TimeoutError("a"), new TimeoutError("b")]);
    const node = new ToolNode("call", { toolId: "flaky", retries: 2 });

    const { events, steps, sleeps } = await runEngine(node, flaky.tool, { numWorkers });

    expect(runOutput(events)).toBe("ok");
    expect(flaky.seen.map((ctx) => ctx.attempt)).toEqual([1, 2, 3]);
    expect(new Set(flaky.seen.map((ctx) => ctx.callId))).toEqual(new Set(["run-123/call"]));
    expect(sleeps).toEqual([1000, 2000]);
    expect(steps).toEqual([
      { event: "failed", attempt: 1n, error: { type: "TimeoutError", message: "a" }, retry_in: 1 },
      { event: "started", attempt: 2n },
      { event: "failed", attempt: 2n, error: { type: "TimeoutError", message: "b" }, retry_in: 2 },
      { event: "started", attempt: 3n },
    ]);
  });
}

test("jitter draws from half to the full delay", async () => {
  const flaky = new FlakyTool([new TimeoutError("a"), new TimeoutError("b")]);
  const node = new ToolNode("call", { toolId: "flaky", retries: 2 });

  const { events, steps, sleeps } = await runEngine(node, flaky.tool, { random: () => 0 });

  expect(runOutput(events)).toBe("ok");
  expect(steps.filter((step) => step.event === "failed").map((step) => step.retry_in)).toEqual([0.5, 1.0]);
  expect(sleeps).toEqual([500, 1000]);
});

test("the backoff stops at the cap, through the engine", async () => {
  const flaky = new FlakyTool([new TimeoutError("a"), new TimeoutError("b"), new TimeoutError("c")]);

  const { events, steps, sleeps } = await withDelays(10, 15, () =>
    runEngine(new ToolNode("call", { toolId: "flaky", retries: 3 }), flaky.tool),
  );

  expect(runOutput(events)).toBe("ok");
  expect(steps.filter((step) => step.event === "failed").map((step) => step.retry_in)).toEqual([10, 15, 15]);
  expect(sleeps).toEqual([10_000, 15_000, 15_000]);
});

for (const [retryAfter, waited] of [
  [7, 7],
  [100, 30],
] as const) {
  test(`retryAfter replaces the backoff up to the cap (${retryAfter})`, async () => {
    const flaky = new FlakyTool([new TransientToolError("busy", { retryAfter })]);
    const node = new ToolNode("call", { toolId: "flaky", retries: 1 });

    const { events, steps, sleeps } = await runEngine(node, flaky.tool, { random: noDraw });

    expect(runOutput(events)).toBe("ok");
    expect(steps[0]!.retry_in).toBe(waited);
    expect(sleeps).toEqual([waited * 1000]);
  });
}

test("exhausted retries list every attempt, through the engine", async () => {
  const flaky = new FlakyTool([new TimeoutError("a"), new TimeoutError("b")]);

  const { events } = await runEngine(new ToolNode("call", { toolId: "flaky", retries: 1 }), flaky.tool);

  const failure = runFailure(events);
  expect(failure.code).toBe("node.execute");
  expect(failure.causeType).toBe("ToolAttemptsFailed");
  expect(failure.message).toBe(
    "tool 'flaky' failed after 2 attempts: attempt 1: TimeoutError: a; attempt 2: TimeoutError: b",
  );
  expect(flaky.seen).toHaveLength(2);
});

test("an output contract failure is not retried", async () => {
  const flaky = new FlakyTool([], "bad");
  const node = new ToolNode("call", { toolId: "flaky", retries: 3 });
  node.outputContract = new ValueContract(typeFor("int", new Map()));

  const { events, sleeps } = await runEngine(node, flaky.tool);

  expect(runFailure(events).code).toBe("contract.output");
  expect(flaky.seen).toHaveLength(1);
  expect(sleeps).toEqual([]);
});

test("a result that isn't a yian value fails contract.output, through the engine", async () => {
  const flaky = new FlakyTool([], { a: new Date(0) });

  const { events } = await runEngine(new ToolNode("call", { toolId: "flaky", retries: 2 }), flaky.tool);

  const failure = runFailure(events);
  expect(failure.code).toBe("contract.output");
  expect(failure.message).toBe("output at a: Date is not a yian value");
  expect(flaky.seen).toHaveLength(1);
});

test("an invalid argument fails node.execute with no call, through the engine", async () => {
  const calls: unknown[] = [];
  const counted = new FunctionTool({
    name: "flaky",
    description: "Count one call.",
    parametersJsonSchema: { type: "object", properties: { n: { type: "integer" } }, required: ["n"] },
    execute: (args) => {
      calls.push(args.n);
      return "ok";
    },
  });
  const node = new ToolNode("call", { toolId: "flaky", retries: 3 });
  node.params = [new Parameter("n")];

  const { events, sleeps } = await runEngine(node, counted, { wiring: { call: { n: "x" } } });

  const failure = runFailure(events);
  expect(failure.code).toBe("node.execute");
  expect(failure.causeType).toBe("TypeCheckError");
  expect(calls).toEqual([]);
  expect(sleeps).toEqual([]);
});

for (const numWorkers of [0, 2]) {
  test(`a stop during the backoff wait stops the run (numWorkers ${numWorkers})`, async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const inner = virtualScheduler();
    // The stop lands as the wait begins, so the delegated sleep rejects at once.
    const scheduler: Scheduler = {
      now: () => inner.now(),
      sleep(ms, signal) {
        host.abort(reason);
        return inner.sleep(ms, signal);
      },
    };
    const flaky = new FlakyTool([new TimeoutError("a")]);
    const node = new ToolNode("call", { toolId: "flaky", retries: 2 });

    const { events, steps, error } = await runEngine(node, flaky.tool, {
      numWorkers,
      scheduler,
      signal: host.signal,
    });

    expect(error).toBe(reason);
    expect(host.signal.aborted).toBe(true);
    expect(flaky.seen).toHaveLength(1);
    expect(steps.map((step) => step.event)).toEqual(["failed"]);
    expect(events.some((event) => event instanceof NodeFailed)).toBe(false);
  });
}
