/**
 * End-to-end `while:`/`until:`/`times:` loop runs over pure-CODE bodies: a port of
 * `tests/engine/test_loop_run.py`.
 *
 * Each iteration is a fresh `LoopNode` driver clone: `L` is the compiled loop (iteration 0),
 * `L~k` the driver clone for iteration `k >= 1`, and `L#k/…` the cloned body of iteration `k`. A
 * stopping driver commits the carried record under `L`; a continuing one grows `body_k` and the
 * next driver, pruning the previous body and (except the origin) itself.
 *
 * Ported: `test_while_loop_runs_until_predicate_false`, `test_while_loop_max_exceeded_fails_run`,
 * `test_until_loop_max_exceeded_fails_run`, `test_body_failure_in_bounded_loop_keeps_the_body_code`,
 * `test_while_loop_max_permits_exactly_max_body_runs` (both runs),
 * `test_nested_call_loop_respawn_failure_keeps_each_authored_origin_once`,
 * `test_loop_body_blocks_for_each_human_turn`,
 * `test_loop_body_can_compute_carried_boolean_in_its_output_binding`,
 * `test_loop_keeps_declared_record_shape_when_one_field_type_is_opaque`,
 * `test_partial_loop_contract_survives_nested_child_projection`,
 * `test_partial_loop_contract_survives_transparent_call_passthrough`,
 * `test_while_loop_predicate_runtime_error_fails_run`, `test_until_runs_body_at_least_once`,
 * `test_loop_budget_exceeded_in_step_fails_run`, `test_long_loop_stays_within_node_budget` and
 * `test_one_field_body_keeps_the_other_carried_fields` (its run is also the harvested
 * `run-loop/engine_loop_run__one_field_body_keeps_the_other_carried_fields`; the case doesn't pin
 * `returnedFields`). Integers are bigints (decision 4a); the CODE references are the TS fixtures'
 * (`./fixtures/compose_codefns.ts:<fn>`), and inline CODE bodies are TS.
 *
 * Changed:
 * - `test_body_failure_in_bounded_loop_keeps_the_body_code`: `causeType` is `"CodeError"`, not
 *   `"ValueError"`: TS wraps whatever a CODE body throws (the TS `boom` throws an `Error`) in a
 *   `CodeError`.
 * - `test_loop_body_blocks_for_each_human_turn`: TS has no `chat_fold` fixture; the body folds
 *   each message in an inline TS CODE body.
 * - `test_while_loop_predicate_runtime_error_fails_run`: a `while:` evaluation error is code
 *   `condition.evaluate` (decision 36), not `node.execute`.
 * - `test_loop_budget_exceeded_in_step_fails_run`: `set_defaults(MAX_TOTAL_NODES=6)` is an `env`
 *   that sets `YIAN_MAX_TOTAL_NODES` to `"6"`. A budget of 6 now trips the first growth (at
 *   `loop`), not a later iteration's (the harvested
 *   `run-loop/engine_loop_run__node_budget_exceeded_in_a_later_iteration` uses 7 for that); the
 *   test asserts only a failed run and "node budget".
 *
 * Not ported:
 * - `test_while_loop_zero_iterations_commits_seed`: covered by
 *   `run-loop/engine_loop_run__while_loop_zero_iterations_commits_seed`.
 * - `test_partial_loop_contract_rejects_invalid_turn_zero_seed`: covered by
 *   `run-loop/engine_loop_run__partial_loop_rejects_an_invalid_turn_zero_seed`.
 * - `test_loop_feeding_downstream_node_commits_and_advances`: covered by
 *   `run-loop/engine_loop_run__loop_output_feeds_a_downstream_node`.
 * - `test_until_does_not_evaluate_predicate_on_seed`: covered by
 *   `run-loop/engine_loop_run__until_does_not_evaluate_predicate_on_seed`.
 * - `test_times_runs_exactly_n`: covered by `run-loop/engine_loop_run__times_runs_exactly_n`.
 * - `test_until_stops_when_predicate_becomes_true`: covered by
 *   `run-loop/engine_loop_run__until_stops_when_predicate_becomes_true`.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { RunSucceeded } from "../../src/events.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { PartialRecordContract, type ValueContract } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { runFlow } from "../../src/run.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { engine } from "./run_state_fixtures.ts";

const COUNTER = `
id: counter
name: counter
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 10
output: \${loop.output}
`;

// max: 2 (< the 3 iterations the body needs) -> the runaway guard fires.
const MAXED = COUNTER.replace("max: 10", "max: 2");

test("while loop runs until predicate false", async () => {
  const result = await runFlow(await loadConformanceFlow(COUNTER), {});
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 3n, exited: true });
});

test("while loop max exceeded fails run", async () => {
  const result = await runFlow(await loadConformanceFlow(MAXED), {});
  expect(result.status).not.toBe("succeeded");
  // The runaway guard specifically: a node-budget blowup also mentions "max".
  expect(result.failure!.message).toContain("exceeded max");
  expect(result.failure!.code).toBe("loop.max_exceeded");
  const causeTypes = new Set(result.events.map((event) => (event as { failure?: Failure }).failure?.causeType));
  expect(causeTypes.has("LoopMaxExceeded")).toBe(true);
});

test("until loop max exceeded fails run", async () => {
  const neverEnds = UNTIL_COUNTDOWN.replace("until: n <= 0", 'until: "false"').replace("max: 10", "max: 2");

  const result = await runFlow(await loadConformanceFlow(neverEnds), {});

  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.code).toBe("loop.max_exceeded");
  expect(result.failure!.message).toContain("exceeded max (2)");
});

test("body failure in bounded loop keeps the body code", async () => {
  const result = await runFlow(await loadConformanceFlow(COUNTER.replace("loop_bump", "boom")), {});

  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.causeType).toBe("CodeError");
});

test("while loop max permits exactly max body runs", async () => {
  // COUNTER needs 3 iterations: max 3 succeeds and max 2 fails, pinning the guard both ways.
  const atBoundary = COUNTER.replace("max: 10", "max: 3");
  const ok = await runFlow(await loadConformanceFlow(atBoundary), {});
  expect(ok.status).toBe("succeeded");
  expect(ok.output).toEqual({ n: 3n, exited: true });
  const below = COUNTER.replace("max: 10", "max: 2");
  expect((await runFlow(await loadConformanceFlow(below), {})).status).not.toBe("succeeded");
});

const NESTED_CALL_LOOP_MAXED = `
id: nested-call-loop-maxed
name: nested_call_loop_maxed
defs:
  keep:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
  runaway:
    nodes:
      repeat:
        kind: loop
        call: keep
        input:
          n: 0
        while: "true"
        max: 1
    output: \${repeat.output}
nodes:
  outer:
    kind: call
    call: runaway
output: \${outer.output}
`;

test("nested call loop respawn failure keeps each authored origin once", async () => {
  const loaded = await loadConformanceFlow(NESTED_CALL_LOOP_MAXED);

  const result = await runFlow(loaded, {});

  const outer = loaded.compiled.nodes.get("outer") as CallNode;
  const repeat = (outer.child as CompiledFlow).nodes.get("repeat")!;
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.nodeId).toBe("outer/repeat~1");
  expect(result.failure!.causeType).toBe("LoopMaxExceeded");
  expect(result.failure!.code).toBe("loop.max_exceeded");
  expect(result.failure!.trace).toEqual([outer.origin!, repeat.origin!]);
});

// The chat scenario: a body blocks each turn on a `human_input` leaf, folds the returned message
// into the carried {messages, exited} record, and loops until "bye".
const LOOP_CHAT = `
id: chat
name: chat
defs:
  chat_turn:
    input:
      messages: !list(str)
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
      fold:
        kind: code
        code: |
          return { messages: [...inputs.messages, inputs.msg], exited: inputs.msg === "bye" }
        input:
          messages: \${input.messages}
          msg: \${ask.output}
        output:
          messages: list(str)
          exited: bool
    output: \${fold.output}
nodes:
  loop:
    kind: loop
    call: chat_turn
    input:
      messages: []
      exited: false
    while: not exited
    max: 100
output: \${loop.output}
`;

test("loop body blocks for each human turn", async () => {
  const loaded = await loadConformanceFlow(LOOP_CHAT);
  const answers = ["hi", "bye"][Symbol.iterator]();
  const result = await runFlow(loaded, {}, { tools: { ask_user: () => answers.next().value } });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ messages: ["hi", "bye"], exited: true });
});

const DIRECT_EXIT_LOOP = `
id: direct-exit-chat
name: direct_exit_chat
input:
  initial_exited: !bool
defs:
  chat_turn:
    input:
      messages: !list(str)
      exited: !bool
    nodes:
      ask:
        kind: human_input
        prompt: "your message"
        output: str
    output:
      messages: \${input.messages}
      exited: \${ask.output == "/exit"}
nodes:
  loop:
    kind: loop
    call: chat_turn
    input:
      messages: []
      exited: \${input.initial_exited}
    while: not exited
    max: 10
output: \${loop.output}
`;

test("loop body can compute carried boolean in its output binding", async () => {
  const loaded = await loadConformanceFlow(DIRECT_EXIT_LOOP);
  const prompts: string[] = [];

  /** Record the single authored prompt and return the exit command. */
  const answer = (request: { question: string }) => {
    prompts.push(request.question);
    return "/exit";
  };

  const result = await runFlow(loaded, { initial_exited: false }, { tools: { ask_user: answer } });

  const bodyType = (loaded.compiled.nodes.get("loop")!.outputContract as ValueContract).type;
  expect(bodyType.fields!.get("exited")!.kind).toBe(ValueKind.BOOLEAN);
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ messages: [], exited: true });
  expect(prompts).toEqual(["your message"]);
});

const OPAQUE_ARITHMETIC_LOOP = `
id: opaque-arithmetic-loop
name: opaque_arithmetic_loop
input:
  initial_n: !int
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes: {}
    output:
      n: \${input.n + 1}
      exited: \${input.exited}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: \${input.initial_n}
      exited: false
    times: 1
output: \${loop.output}
`;

test("loop keeps declared record shape when one field type is opaque", async () => {
  const loaded = await loadConformanceFlow(OPAQUE_ARITHMETIC_LOOP);

  const loop = loaded.compiled.nodes.get("loop") as LoopNode;
  const contract = loop.outputContract as PartialRecordContract;
  expect(contract.constructor).toBe(PartialRecordContract);
  expect(contract.names).toEqual(["n", "exited"]);
  expect(contract.fields[0]).toEqual(["n", null]);
  expect(contract.fields[1]![1]).not.toBeNull();
  expect(loop.respawn(1).outputContract).toBe(contract);
  expect("outputFieldContracts" in loop).toBe(false);

  const result = await runFlow(loaded, { initial_n: 1 });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n, exited: false });
});

const NESTED_OPAQUE_TURN_ZERO_LOOP = `
id: nested-opaque-turn-zero
name: nested_opaque_turn_zero
input:
  initial_n: !str
defs:
  inner_body:
    input:
      n: !int
      note: !str
    nodes: {}
    output:
      n: \${input.n}
      note: \${input.note + "!"}
  outer_body:
    input:
      n: !int
      note: !str
    nodes:
      inner:
        kind: loop
        call: inner_body
        input:
          n: \${input.n}
          note: \${input.note}
        while: "false"
        max: 1
    output:
      n: \${inner.output.n}
      note: \${inner.output.note}
nodes:
  outer:
    kind: loop
    call: outer_body
    input:
      n: \${input.initial_n + ""}
      note: seed
    while: "false"
    max: 1
output: \${outer.output}
`;

test("partial loop contract survives nested child projection", async () => {
  const loaded = await loadConformanceFlow(NESTED_OPAQUE_TURN_ZERO_LOOP);

  const contract = loaded.compiled.nodes.get("outer")!.outputContract as PartialRecordContract;
  expect(contract.constructor).toBe(PartialRecordContract);
  expect(contract.fields[0]![0]).toBe("n");
  expect(contract.fields[0]![1]).not.toBeNull();

  const result = await runFlow(loaded, { initial_n: "not-an-int" });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message).toContain("int");
});

const PASSTHROUGH_OPAQUE_TURN_ZERO_LOOP = `
id: passthrough-opaque-turn-zero
name: passthrough_opaque_turn_zero
input:
  initial_n: !str
defs:
  inner_body:
    input:
      n: !int
      note: !str
    nodes: {}
    output:
      n: \${input.n}
      note: \${input.note + "!"}
  middle:
    input:
      n: !int
      note: !str
    nodes:
      inner:
        kind: loop
        call: inner_body
        input:
          n: \${input.n}
          note: \${input.note}
        while: "false"
        max: 1
    output: \${inner.output}
  outer_body:
    input:
      n: !int
      note: !str
    nodes:
      middle_call:
        kind: call
        call: middle
        input:
          n: \${input.n}
          note: \${input.note}
    output:
      n: \${middle_call.output.n}
      note: \${middle_call.output.note}
nodes:
  outer:
    kind: loop
    call: outer_body
    input:
      n: \${input.initial_n + ""}
      note: seed
    while: "false"
    max: 1
output: \${outer.output}
`;

test("partial loop contract survives transparent call passthrough", async () => {
  const loaded = await loadConformanceFlow(PASSTHROUGH_OPAQUE_TURN_ZERO_LOOP);

  const contract = loaded.compiled.nodes.get("outer")!.outputContract as PartialRecordContract;
  expect(contract.constructor).toBe(PartialRecordContract);
  expect(contract.fields[0]![1]).not.toBeNull();

  const result = await runFlow(loaded, { initial_n: "not-an-int" });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message).toContain("int");
});

// The predicate divides by the carried `n` and the body counts down: the seed check (n=2) and the
// iteration-0 check (n=1, on `loop~1`) pass, then iteration 1's check divides by 0 inside the
// driver's `run`, which must become a failed run, never an escape from `runFlow`.
const PREDICATE_RAISES = `
id: pred-raise
name: pred_raise
defs:
  down:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: down
    input:
      n: 2
    while: 10 / n > 0
    max: 10
output: \${loop.output}
`;

test("while loop predicate runtime error fails run", async () => {
  const result = await runFlow(await loadConformanceFlow(PREDICATE_RAISES), {});
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.code).toBe("condition.evaluate");
  expect(result.failure!.message).toContain("division by zero");
});

// `until:` is a do-while: the seed n=1 doesn't satisfy `until: n <= 0`, yet the body runs once
// (1 -> 0) before the predicate is consulted; a `while`-style pre-check would commit {n: 1}.
const UNTIL_SEED_TRUE = `
id: ust
name: ust
defs:
  countdown:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 1
    until: n <= 0
    max: 5
output: \${loop.output}
`;

test("until runs body at least once", async () => {
  const result = await runFlow(await loadConformanceFlow(UNTIL_SEED_TRUE), {});
  expect(result.output).toEqual({ n: 0n });
});

test("loop budget exceeded in step fails run", async () => {
  // A budget trip inside a driver's growth must become a failed run, not an escape.
  const env = (name: string) => (name === "YIAN_MAX_TOTAL_NODES" ? "6" : undefined);
  const result = await runFlow(await loadConformanceFlow(COUNTER), {}, { env });
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.message).toContain("node budget");
});

// `until:` do-while over a countdown: 3 -> 2 -> 1 -> 0, then `n <= 0` holds and the loop stops.
const UNTIL_COUNTDOWN = `
id: uc
name: uc
defs:
  countdown:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 3
    until: n <= 0
    max: 10
output: \${loop.output}
`;

// Twenty iterations far exceed a 40-node budget if nothing is pruned; each driver retires the
// previous body and itself, so about one iteration stays resident.
const TIMES_20 = `
id: t20
name: t20
defs:
  countdown:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_countdown
        input:
          n: \${input.n}
        output:
          n: int
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: countdown
    input:
      n: 30
    times: 20
output: \${loop.output}
`;

test("long loop stays within node budget", async () => {
  const eng = engine((await loadConformanceFlow(TIMES_20)).compiled, {
    inputs: {},
    growBuilder: buildGrow,
    defaults: { MAX_TOTAL_NODES: 40n },
  });
  const terminal = (await eng.run()).at(-1);

  expect(terminal).toBeInstanceOf(RunSucceeded);
  expect(eng.state.flow.nodes.size).toBeGreaterThan(0);
  expect(eng.state.flow.nodes.size).toBeLessThan(40);
  expect((terminal as RunSucceeded).output).toEqual({ n: 10n });
});

// A one-field body `output: {count: ...}` returns only `count`; `label` keeps its seed.
const ONE_FIELD_BODY = `
id: one-field-body
name: one_field_body
nodes:
  loop:
    kind: loop
    times: 3
    input:
      count: 0
      label: x
    nodes:
      bump:
        kind: code
        input:
          n: \${input.count}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      count: \${bump.output}
output: \${loop.output}
`;

test("one field body keeps the other carried fields", async () => {
  const loaded = await loadConformanceFlow(ONE_FIELD_BODY);
  expect((loaded.compiled.nodes.get("loop") as LoopNode).returnedFields).toEqual(["count"]);
  const result = await runFlow(loaded, {});
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ count: 3n, label: "x" });
});
