/**
 * End-to-end `while:`/`until:`/`times:` loop runs over pure-CODE bodies.
 *
 * Each iteration is a fresh `LoopNode` driver clone: `L` is the compiled loop (iteration 0),
 * `L~k` the driver clone for iteration `k >= 1`, and `L#k/…` the cloned body of iteration `k`. A
 * stopping driver commits the carried record under `L`; a continuing one grows `body_k` and the
 * next driver, pruning the previous body and (except the origin) itself.
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { RunSucceeded } from "../../src/events.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { PartialRecordContract, type ValueContract } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { replayRunLog, advance, runFlow } from "../../src/run.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { JsonlRunLog, NodeExpandedRecord } from "../../src/runtime/log.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, NO_ENV } from "./run_state_fixtures.ts";

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

// The chat scenario: a body blocks each turn on an `ask` leaf, folds the returned message
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
        kind: tool
        tool_id: ask
        input:
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
  const result = await runFlow(loaded, {}, { tools: { ask: () => answers.next().value } });

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
        kind: tool
        tool_id: ask
        input:
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

  const result = await runFlow(loaded, { initial_exited: false }, { tools: { ask: answer } });

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
  const env = (name: string) => (name === "YIAN_MAX_TOTAL_NODES" ? "6" : NO_ENV(name));
  const result = await runFlow(await loadConformanceFlow(COUNTER), {}, { env });
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.message).toContain("node budget");
});

// `times: N` is a fixed count: seed n=10, times 3 runs 10 -> 9 -> 8 -> 7, then the driver at k=3
// stops.
const TIMES_3 = `
id: t3
name: t3
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
      n: 10
    times: 3
output: \${loop.output}
`;

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

test("terminate leaves exactly one dead final body and driver", async () => {
  // STOP commits under `loop` and prunes nothing: for times 3 the bodies run at k=0,1,2 and
  // STOP fires on `loop~3`, which read `loop#2/__end__`, so exactly `loop#2/` and `loop~3` stay.
  const eng = engine((await loadConformanceFlow(TIMES_3)).compiled, { inputs: {}, growBuilder: buildGrow });
  const terminal = (await eng.run()).at(-1);
  expect(terminal).toBeInstanceOf(RunSucceeded);
  expect((terminal as RunSucceeded).output).toEqual({ n: 7n });
  const ids = [...eng.state.flow.nodes.keys()];
  const bodyNamespaces = new Set(ids.filter((id) => id.startsWith("loop#")).map((id) => id.split("/")[0]));
  expect(bodyNamespaces).toEqual(new Set(["loop#2"]));
  // The origin `loop` is the committed result node, not a `loop~<k>` clone.
  expect(new Set(ids.filter((id) => id.startsWith("loop~")))).toEqual(new Set(["loop~3"]));
  expect(eng.state.flow.nodes.has("loop")).toBe(true);
});

test("until loop without max runs until predicate ends it", async () => {
  const result = await runFlow(await loadConformanceFlow(UNTIL_COUNTDOWN.replace("    max: 10\n", "")), {});
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 0n });
});

// A partial LOOP body returns a subset of the carried record; omitted fields keep the previous
// iteration's value. This body reads the whole record but returns only `{n, done}`, so `person`
// and `talking_to` ride through every turn unchanged.
const PARTIAL_CHAT = `
id: partial-chat
name: partial_chat
typedefs:
  Person:
    name: str
input:
  person: !Person
  talking_to: !str
defs:
  turn:
    input:
      person: !Person
      talking_to: !str
      n: !int
    nodes:
      step:
        kind: code
        code: |
          if (inputs["person"]["name"] !== "Ada") throw new Error("assert")
          if (inputs["talking_to"] !== "Bob") throw new Error("assert")
          const n = inputs["n"] + 1
          return {"n": n, "done": n >= 3}
        input:
          person: \${input.person}
          talking_to: \${input.talking_to}
          n: \${input.n}
        output:
          n: int
          done: bool
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      person: \${input.person}
      talking_to: \${input.talking_to}
      n: 0
      done: false
    while: not done
output: \${loop.output}
`;

const PARTIAL_CHAT_INPUTS = { person: { name: "Ada" }, talking_to: "Bob" };

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

const PARTIAL_NOTE = `
id: partial-note
name: partial_note
input:
  note0: !Optional(str)
defs:
  clear_note:
    input:
      n: !int
    nodes:
      step:
        kind: code
        code: |
          return {"n": inputs["n"] + 1, "note": null}
        input:
          n: \${input.n}
        output:
          n: int
          note: Optional(str)
    output: \${step.output}
  bump_only:
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
  loop:
    kind: loop
    call: BODY
    input:
      n: 0
      note: \${input.note0}
    times: 2
output: \${loop.output}
`;

test("partial body explicit null sets nullable field", async () => {
  const cleared = await runFlow(await loadConformanceFlow(PARTIAL_NOTE.replace("BODY", "clear_note")), {
    note0: "seed",
  });
  expect(cleared.status).toBe("succeeded");
  expect(cleared.output).toEqual({ n: 2n, note: null });

  const kept = await runFlow(await loadConformanceFlow(PARTIAL_NOTE.replace("BODY", "bump_only")), { note0: "seed" });
  expect(kept.status).toBe("succeeded");
  expect(kept.output).toEqual({ n: 2n, note: "seed" });
});

/** Return `PARTIAL_CHAT` with the LOOP asserting each of `asserts`. */
function withLoopAsserts(...asserts: string[]): string {
  const lines = asserts.map((assertion) => `      - "${assertion}"\n`).join("");
  return PARTIAL_CHAT.replace("    while: not done\n", `    while: not done\n    asserts:\n${lines}`);
}

test("partial loop asserts see the merged record", async () => {
  const postOk = await runFlow(
    await loadConformanceFlow(withLoopAsserts("output.person.name == 'Ada'")),
    PARTIAL_CHAT_INPUTS,
  );
  expect(postOk.status).toBe("succeeded");
  expect((postOk.output as Record<string, unknown>)["person"]).toEqual({ name: "Ada" });

  const postBad = await runFlow(
    await loadConformanceFlow(withLoopAsserts("output.person.name == 'Bob'")),
    PARTIAL_CHAT_INPUTS,
  );
  expect(postBad.status).toBe("failed");
  expect(postBad.failure!.causeType).toBe("NodeAssertFailed");
  expect(postBad.failure!.message).toContain("post-assert failed");

  // A PRE assert reads the carried inputs; each respawned driver re-checks it against the merged
  // record, where the omitted `person` still holds its seed.
  const preOk = await runFlow(await loadConformanceFlow(withLoopAsserts("person.name == 'Ada'")), PARTIAL_CHAT_INPUTS);
  expect(preOk.status).toBe("succeeded");
  expect((preOk.output as Record<string, unknown>)["n"]).toBe(3n);
  // Respawned drivers do run PRE asserts: one on a returned field fails mid-loop.
  const preBad = await runFlow(await loadConformanceFlow(withLoopAsserts("n < 2")), PARTIAL_CHAT_INPUTS);
  expect(preBad.status).toBe("failed");
  expect(preBad.failure!.nodeId).toBe("loop~2");
  expect(preBad.failure!.message).toContain("pre-assert failed");
});

test("partial loop zero iterations commits seed", async () => {
  const src = PARTIAL_CHAT.replace("      done: false\n", "      done: true\n");
  const result = await runFlow(await loadConformanceFlow(src), PARTIAL_CHAT_INPUTS);
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ person: { name: "Ada" }, talking_to: "Bob", n: 0n, done: true });
});

test("partial until and times loops", async () => {
  const until = await runFlow(
    await loadConformanceFlow(PARTIAL_CHAT.replace("while: not done", "until: done")),
    PARTIAL_CHAT_INPUTS,
  );
  expect(until.status).toBe("succeeded");
  expect(until.output).toEqual({ person: { name: "Ada" }, talking_to: "Bob", n: 3n, done: true });

  const times = await runFlow(
    await loadConformanceFlow(PARTIAL_CHAT.replace("while: not done", "times: 2")),
    PARTIAL_CHAT_INPUTS,
  );
  expect(times.status).toBe("succeeded");
  expect(times.output).toEqual({ person: { name: "Ada" }, talking_to: "Bob", n: 2n, done: false });
});

const ONE = `
  one:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output: \${step.output}
`;

const LOOP = `
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 0
    times: TIMES
output: \${loop.output}
`;

const CALL_BODY =
  `
id: nested_call
name: nested_call
defs:` +
  ONE +
  `
  body:
    input:
      n: !int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: \${input.n}
    output:
      n: \${c.output}
` +
  LOOP;

const MAP_BODY =
  `
id: nested_map
name: nested_map
defs:` +
  ONE +
  `
  body:
    input:
      n: !int
    nodes:
      m:
        kind: map
        call: one
        over: \${[input.n, input.n]}
        input:
          n: \${item}
      pick:
        kind: code
        input:
          xs: \${m.output}
        output: int
        code: |
          return inputs["xs"][0]
    output:
      n: \${pick.output}
` +
  LOOP;

const LOOP_BODY =
  `
id: nested_loop
name: nested_loop
defs:` +
  ONE +
  `
  inc:
    input:
      n: !int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: \${input.n}
    output:
      n: \${c.output}
  body:
    input:
      n: !int
    nodes:
      inner:
        kind: loop
        call: inc
        input:
          n: \${input.n}
        times: 3
    output:
      n: \${inner.output.n}
` +
  LOOP;

const CASE_BODY =
  `
id: nested_case
name: nested_case
defs:
  up:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output: \${step.output}
  up2:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 2
    output: \${step.output}
  body:
    input:
      n: !int
    nodes:
      pick:
        kind: case
        cases:
          - when: "input.n % 2 == 0"
            then: call(up, n=\${input.n})
        else: call(up2, n=\${input.n})
    output:
      n: \${pick.output}
` +
  LOOP;

/** Return the CASE body's carried value after `times` iterations from zero. */
function caseTotal(times: number): number {
  let n = 0;
  for (let i = 0; i < times; i++) n += n % 2 === 0 ? 1 : 2;
  return n;
}

// Each body grows nodes inside its iteration, so pruning only the direct body ids leaks them.
const NESTED_BODIES: Record<string, readonly [source: string, expected: (times: number) => number]> = {
  call: [CALL_BODY, (times) => times],
  map: [MAP_BODY, (times) => times],
  nested: [LOOP_BODY, (times) => 3 * times],
  case: [CASE_BODY, caseTotal],
};

/** Run one nested-growth body `times` iterations and return the finished engine. */
async function finishNested(
  shape: string,
  times: number,
  engineOptions: Parameters<typeof engine>[1] = {},
): Promise<FlowEngine> {
  const [source, expected] = NESTED_BODIES[shape]!;
  const eng = engine((await loadConformanceFlow(source.replace("TIMES", String(times)))).compiled, {
    inputs: {},
    growBuilder: buildGrow,
    ...engineOptions,
  });
  const terminal = (await eng.run()).at(-1);
  expect(terminal).toBeInstanceOf(RunSucceeded);
  expect((terminal as RunSucceeded).output).toEqual({ n: BigInt(expected(times)) });
  return eng;
}

for (const shape of Object.keys(NESTED_BODIES).sort()) {
  for (const numWorkers of [0, 2]) {
    const mode = numWorkers === 0 ? "serial" : "pooled";
    test(`loop with nested growth keeps one iteration resident: ${shape} ${mode}`, async () => {
      // Pruning an iteration also retires the nodes its nested CALL/MAP/LOOP grew.
      const resident = (await finishNested(shape, 3, { numWorkers })).state.flow.nodes.size;
      // A tight budget fails mid-run if any earlier iteration's nested nodes leak.
      const eng = await finishNested(shape, 20, { numWorkers, defaults: { MAX_TOTAL_NODES: BigInt(resident + 20) } });
      expect(eng.state.flow.nodes.size).toBe(resident);
    });
  }
}

for (const shape of Object.keys(NESTED_BODIES).sort()) {
  test(`loop with nested growth keeps trace indexes bounded: ${shape}`, async () => {
    // A learnable run's current-name trace indexes shrink with the live graph.
    const small = (await finishNested(shape, 3, { learnable: true })).state.trace!;
    const large = (await finishNested(shape, 20, { learnable: true })).state.trace!;
    expect(large.currentWriters.size).toBe(small.currentWriters.size);
    expect(large.pendingConnections.size).toBe(small.pendingConnections.size);
  });
}

test("loop growth record keeps only the requested prune ids", async () => {
  // The log records the body's direct ids and the driver; nested ids are derived.
  const eng = await finishNested("call", 3);
  const pruned = [...eng.log]
    .map(({ record }) => record)
    .filter((record): record is NodeExpandedRecord => record.constructor === NodeExpandedRecord)
    .filter((record) => record.spawnerId === "loop~2")
    .map((record) => record.pruneIds);
  expect(pruned).toEqual([["loop#1/__end__", "loop#1/__start__", "loop#1/c", "loop~2"]]);
});

test("self pruning spawner hands its fragment to its owner", async () => {
  // A driver that prunes itself passes the nodes it grew to its own owner.
  const owners = (await finishNested("call", 3)).state.growthOwner;
  expect(new Set(owners.values())).toEqual(new Set(["loop", "loop#2/c"]));
  expect(owners.get("loop~3")).toBe("loop");
});

// The child's unread side chain can still be waiting when the next driver prunes its iteration.
const DANGLING_SIDE =
  `
id: dangling_side
name: dangling_side
defs:
  one:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      side:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] * 2
      side2:
        kind: code
        input:
          n: \${side.output}
        output: int
        code: |
          return inputs["n"] * 2
      side3:
        kind: code
        input:
          n: \${side2.output}
        output: int
        code: |
          return inputs["n"] * 2
      side4:
        kind: code
        input:
          n: \${side3.output}
        output: int
        code: |
          return inputs["n"] * 2
    output: \${step.output}
  body:
    input:
      n: !int
    nodes:
      c:
        kind: call
        call: one
        input:
          n: \${input.n}
    output:
      n: \${c.output}
` + LOOP.replace("TIMES", "4");

for (const numWorkers of [0, 3]) {
  test(`loop keeps a nested subtree with unfinished work: ${numWorkers === 0 ? "serial" : "pooled"}`, async () => {
    // A nested node still waiting when its iteration is pruned keeps its subtree alive until a
    // later growth finds it settled.
    const eng = engine((await loadConformanceFlow(DANGLING_SIDE)).compiled, {
      inputs: {},
      growBuilder: buildGrow,
      numWorkers,
    });
    const terminal = (await eng.run()).at(-1);
    expect(terminal).toBeInstanceOf(RunSucceeded);
    expect((terminal as RunSucceeded).output).toEqual({ n: 4n });
    // Every owner entry names live nodes.
    const nodes = eng.state.flow.nodes;
    expect([...eng.state.growthOwner].filter(([key, value]) => !nodes.has(key) || !nodes.has(value))).toEqual([]);
    // Serially, each iteration's side chain is still waiting when the next driver prunes it;
    // later drivers retire it, so only the last iteration stays. Pooled order varies.
    if (numWorkers === 0) {
      expect(new Set([...nodes.keys()].map((id) => id.split("/")[0]))).toEqual(
        new Set(["__start__", "__end__", "loop", "loop#3", "loop~4"]),
      );
      expect(eng.state.deferredPrune).toEqual(new Set());
    }
  });
}

// The body output reads only `bump`, so the next driver can start while the side chain still
// runs; `side2` also reads `bump`, which must then stay live with it. `SLEEP` is milliseconds.
const DIRECT_SIDE =
  `
id: direct_side
name: direct_side
defs:
  body:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      side:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          await new Promise((resolve) => setTimeout(resolve, SLEEP))
          return inputs["n"]
      side2:
        kind: code
        input:
          n: \${side.output}
          m: \${bump.output}
        output: int
        code: |
          return inputs["n"] + inputs["m"]
    output:
      n: \${bump.output}
` + LOOP.replace("TIMES", "4");

for (const numWorkers of [0, 1, 4]) {
  for (const sleep of ["0", "300"]) {
    test(`loop keeps an unfinished body node live: ${numWorkers} workers, sleep ${sleep} ms`, async () => {
      // A body node the output doesn't read may finish after the next driver starts.
      const result = await runFlow(await loadConformanceFlow(DIRECT_SIDE.replace("SLEEP", sleep)), {}, { numWorkers });
      expect(result.status).toBe("succeeded");
      expect(result.output).toEqual({ n: 4n });
    });
  }
}

test("loop retires a kept body node once settled", async () => {
  // A body node kept while unfinished is retired by a later prune once it has settled.
  const finish = async (times: number, defaults?: Record<string, bigint>): Promise<FlowEngine> => {
    const source = DIRECT_SIDE.replace("SLEEP", "50").replace("times: 4", `times: ${times}`);
    const eng = engine((await loadConformanceFlow(source)).compiled, {
      inputs: {},
      growBuilder: buildGrow,
      numWorkers: 4,
      defaults,
    });
    const terminal = (await eng.run()).at(-1);
    expect(terminal).toBeInstanceOf(RunSucceeded);
    expect((terminal as RunSucceeded).output).toEqual({ n: BigInt(times) });
    return eng;
  };

  const resident = (await finish(3)).state.flow.nodes.size;
  // A few iterations' side chains may linger at once; leaking every one would add ~80.
  const eng = await finish(20, { MAX_TOTAL_NODES: BigInt(resident + 40) });
  expect(eng.state.flow.nodes.size).toBeLessThanOrEqual(resident + 40);
});

// The unread `note` CALL is still waiting on `slow` when the next driver prunes its iteration,
// then grows its own child while deferred.
const DEFERRED_CALL =
  `
id: deferred_call
name: deferred_call
defs:
  echo:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"]
    output: \${step.output}
  body:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      slow:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          await new Promise((resolve) => setTimeout(resolve, 200))
          return inputs["n"]
      note:
        kind: call
        call: echo
        input:
          n: \${slow.output}
    output:
      n: \${bump.output}
` + LOOP.replace("TIMES", "3");

test("loop body call grows after its iteration was pruned", async () => {
  // A kept body CALL can still expand its child after the next driver started.
  const result = await runFlow(await loadConformanceFlow(DEFERRED_CALL), {}, { numWorkers: 4 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 3n });
});

// ---------- resumed from a cut log ----------

const dir = mkdtempSync(join(tmpdir(), "yian-loop-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/**
 * Run `source` on `inputs` into a new log, await `beforeCut` on it, cut the log to its first half
 * of records, resume it, and expect both to succeed with the same output; return the log's path.
 */
async function runCutAndResume(
  name: string,
  source: string,
  inputs: Record<string, unknown>,
  numWorkers?: number,
  beforeCut: (path: string) => Promise<void> = async () => {},
): Promise<string> {
  const path = join(dir, `${name}.jsonl`);
  const first = await runFlow(await loadConformanceFlow(source), inputs, { log: path, numWorkers, env: hostEnv(dir) });
  expect(first.failure).toBeNull();
  await beforeCut(path);
  // Cut mid-run, so the resume replays the recorded drivers and runs the rest live.
  const lines = readFileSync(path, "utf8").split(/(?<=\n)/);
  writeFileSync(path, lines.slice(0, Math.floor(lines.length / 2)).join(""));
  const resumed = await advance({ log: path, numWorkers, env: hostEnv(dir) });
  expect(resumed.failure).toBeNull();
  expect(resumed.output).toEqual(first.output);
  return path;
}

/** Replay the log at `path` over `source`. */
async function replayed(source: string, path: string) {
  const log = JsonlRunLog.resume(path);
  try {
    return replayRunLog(await loadConformanceFlow(source), log);
  } finally {
    log.close();
  }
}

test("a partial loop resumes from its log", async () => {
  await runCutAndResume("partial", PARTIAL_CHAT, PARTIAL_CHAT_INPUTS);
});

test("a nested growth loop resumes from a cut log", async () => {
  // The resumed log replays to one resident iteration.
  const source = NESTED_BODIES["nested"]![0].replace("TIMES", "6");
  const path = await runCutAndResume("nested", source, {});
  const state = await replayed(source, path);
  expect(state.flow.nodes.size).toBe((await finishNested("nested", 2)).state.flow.nodes.size);
});

test("a loop with a kept body node resumes from a cut log", async () => {
  // The replay re-derives which body nodes were kept, so a resumed run finishes.
  const source = DIRECT_SIDE.replace("SLEEP", "200");
  await runCutAndResume("kept", source, {}, 4, async (path) => {
    expect((await replayed(source, path)).deferredPrune.size).toBeGreaterThan(0);
  });
});
