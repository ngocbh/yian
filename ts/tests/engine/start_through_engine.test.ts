/**
 * The top-level START runs as an ordinary scheduled node: a port of
 * `tests/engine/test_start_through_engine.py`. START runs through `evalNode`, emits `NodeStarted`
 * and `NodeSucceeded`, and carries the flow's boundary asserts in its ordinary `postAsserts`, each
 * `input.X` lowered to `output.X` so it reads START's bound output record.
 *
 * Ported (adapted), on `graph()` with a typed `StartNode` whose `postAsserts` hold the lowered
 * boundary assert, for Python's loaded flows: `test_start_emits_started_and_succeeded`, and the
 * engine halves of `test_typed_int_default_is_used_when_omitted`,
 * `test_boundary_assert_violating_value_fails`, `test_boundary_assert_satisfying_value_succeeds`
 * and `test_boundary_assert_reads_typed_default`. Integers are bigints.
 *
 * Later (T4-8), through `runFlow`: the `run_flow` halves of those four tests, and
 * `test_loadedflow_reusable_across_runs_shared_ir_untouched`.
 * Later (T4-10), through CALL: `test_child_boundary_assert_violation_fails_run` and
 * `test_child_boundary_assert_satisfied_succeeds`.
 */

import { expect, test } from "bun:test";

import { type CompiledFlow, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { engine, FuncNode, graph } from "./run_state_fixtures.ts";

const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));

/**
 * Return `START -> <id> -> END` for an integer input `name`, where node `id` echoes it and its
 * value is the output; START asserts `assert` (over `input`) when given.
 */
function echoFlow(
  id: string,
  name: string,
  options: { default?: bigint; assert?: [expression: string, display: string] } = {},
): CompiledFlow {
  const parameter =
    options.default === undefined
      ? new Parameter(name, { contract: INT, required: true })
      : new Parameter(name, { contract: INT, hasDefault: true, default: options.default });
  const start = new StartNode(START_ID, { params: [parameter] });
  if (options.assert !== undefined) start.postAsserts = [new Assertion(...options.assert)];
  const echo = new FuncNode(id, (inputs) => inputs["x"], { reads: { x: `\${input.${name}}` }, outputContract: INT });
  return graph([start, echo], [
    [START_ID, id],
    [id, END_ID],
  ], { outputs: [new FlowOutput("output", `\${${id}.output}`)] });
}

/** The age flow: START requires `age` and asserts `input.age >= 18`. */
function ageFlow(): CompiledFlow {
  return echoFlow("n", "age", { assert: ["output.age >= 18", "input.age >= 18"] });
}

test("boundary assert violating value fails", async () => {
  const last = (await engine(ageFlow(), { inputs: { age: 10n } }).run()).at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.message).toContain("assert failed");
});

test("boundary assert satisfying value succeeds", async () => {
  expect((await engine(ageFlow(), { inputs: { age: 20n } }).run()).at(-1)).toEqual(new RunSucceeded({ output: 20n }));
});

test("start emits started and succeeded", async () => {
  // START is an ordinary scheduled node, so it emits both lifecycle events.
  const events = await engine(ageFlow(), { inputs: { age: 20n } }).run();
  expect(events.some((event) => event instanceof NodeStarted && event.nodeId === START_ID)).toBe(true);
  expect(events.some((event) => event instanceof NodeSucceeded && event.nodeId === START_ID)).toBe(true);
});

test("typed int default is used when omitted", async () => {
  // The caller omits `n`; parameter binding supplies the integer default before START runs.
  const events = await engine(echoFlow("echo", "n", { default: 5n })).run();
  const started = events.find((event) => event instanceof NodeSucceeded && event.nodeId === START_ID);
  expect((started as NodeSucceeded).output).toEqual({ n: 5n });
  expect(events.at(-1)).toEqual(new RunSucceeded({ output: 5n }));
});

test("boundary assert reads typed default", async () => {
  // `n` is omitted, so the boundary assert reads START's bound integer output.
  const flow = echoFlow("echo", "n", { default: 5n, assert: ["output.n >= 5", "input.n >= 5"] });
  expect((await engine(flow).run()).at(-1)).toEqual(new RunSucceeded({ output: 5n }));
});
