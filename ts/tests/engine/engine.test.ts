/**
 * The serial append-only engine drain: a port of `tests/engine/test_engine.py`. Fake nodes (no
 * models or tools) exercise ordering, fan-out, the exactly-once diamond join, CASE branches and
 * skip floods, outputs before successors, failure and typed output commits.
 *
 * Ported: `test_linear_runs_in_order`, `test_end_assembles_declared_output_record`,
 * `test_no_declared_outputs_yields_none`, `test_end_output_coalesce_returns_first_non_none`,
 * `test_end_output_default_when_ref_none`, `test_end_output_literal_string_passthrough`,
 * `test_end_output_literal_numeric_passthrough` (a bigint), `test_end_output_flowoutput_carrier_builds_record`,
 * `test_end_output_whole_ref_preserves_typed_value`, `test_fan_out_runs_all_branches`,
 * `test_multiple_roots_all_start`, `test_diamond_join_fires_exactly_once`,
 * `test_outputs_written_before_successor_runs`, `test_case_takes_branch_and_skips_other`,
 * `test_case_else_fallback`, `test_skip_floods_through_unreachable_chain`,
 * `test_data_edged_branch_target_does_not_run_when_skipped`,
 * `test_skipped_terminal_producer_fails_run`, `test_coalesce_terminal_one_branch_skipped_succeeds`,
 * `test_does_not_misfire_on_optional_terminal`, `test_genuine_none_terminal_succeeds`,
 * `test_node_failure_produces_graph_failed`, `test_output_enforced_rejects_type_mismatch`,
 * `test_multi_output_record_is_a_closed_type`, `test_unresolvable_declared_type_is_not_enforced`,
 * `test_undeclared_output_is_not_enforced` and `test_fixture_engine_defaults_serial`.
 *
 * Changed: `test_output_enforced_coerces_and_types_storage`: an int filling a float slot stays an
 * int in TS, so the stored value is the int 3 with storage kind `int`, where Python coerces it to
 * the float 3.0.
 *
 * TS-only: nothing ready before a terminal result is nonterminal quiescence. The node failure row
 * also checks that a failed run aborts its signal.
 *
 * Later (pooled mode): `test_num_workers_0_and_4_same_terminal`, with the pooled engine.
 *
 * Not ported: `test_parallel_module_and_export_are_gone`; TS never had `ParallelFlowEngine`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import { ELSE_HANDLE, type Node } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { BranchNode, engine, FailNode, FuncNode, graph, RecordNode, rejected } from "./run_state_fixtures.ts";

/** Return the last event of a run of `flow`. */
async function last(flow: CompiledFlow): Promise<unknown> {
  return (await engine(flow).run()).at(-1);
}

/** Return the output of a run of `flow`, which must succeed. */
async function output(flow: CompiledFlow): Promise<unknown> {
  const event = await last(flow);
  expect(event).toBeInstanceOf(RunSucceeded);
  return (event as RunSucceeded).output;
}

// --- ordering / linear ---

test("linear runs in order", async () => {
  const log: string[] = [];
  const flow = graph([new RecordNode("a", log), new RecordNode("b", log), new RecordNode("c", log)], [
    [START_ID, "a"],
    ["a", "b"],
    ["b", "c"],
    ["c", END_ID],
  ]);
  expect(await last(flow)).toBeInstanceOf(RunSucceeded);
  expect(log).toEqual(["a", "b", "c"]);
});

/** Return `START -> a -> t -> END`, where `t` returns `{output: "final", n: 7}`. */
function terminalFlow(outputs?: readonly FlowOutput[]): CompiledFlow {
  return graph(
    [new FuncNode("a", () => ({})), new FuncNode("t", () => ({ output: "final", n: 7n }))],
    [
      [START_ID, "a"],
      ["a", "t"],
      ["t", END_ID],
    ],
    outputs === undefined ? {} : { outputs },
  );
}

test("end assembles declared output record", async () => {
  // Two outputs bound to the terminal node's value form an object keyed by output name.
  const outputs = [new FlowOutput("output", "${t.output.output}"), new FlowOutput("n", "${t.output.n}")];
  expect(await output(terminalFlow(outputs))).toEqual({ output: "final", n: 7n });
});

test("no declared outputs yields null", async () => {
  expect(await output(terminalFlow())).toBeNull();
});

// --- END output assembly via the binding layer ---

/** Return `START -> <node> -> END` with one output `result` bound to `binding`. */
function oneOutput(node: Node, binding: unknown): CompiledFlow {
  return graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ], { outputs: [new FlowOutput("result", binding)] });
}

test("end output coalesce returns first non null", async () => {
  // Only `b` produced a value (the `a` branch never ran), so the coalesce resolves to it.
  expect(await output(oneOutput(new FuncNode("b", () => "B-RAN"), "${a.output | b.output}"))).toBe("B-RAN");
});

test("end output default when ref null", async () => {
  expect(await output(oneOutput(new FuncNode("a", () => ({})), '${missing.output:-"fallback"}'))).toBe("fallback");
});

test("end output literal string passthrough", async () => {
  expect(await output(oneOutput(new FuncNode("a", () => ({})), "plain literal"))).toBe("plain literal");
});

test("end output literal numeric passthrough", async () => {
  expect(await output(oneOutput(new FuncNode("a", () => ({})), 7n))).toBe(7n);
});

test("end output FlowOutput carrier builds record", async () => {
  const outputs = [new FlowOutput("output", "${t.output.output}"), new FlowOutput("n", "${t.output.n}")];
  expect(await output(terminalFlow(outputs))).toEqual({ output: "final", n: 7n });
});

test("end output whole ref preserves typed value", async () => {
  expect(await output(oneOutput(new FuncNode("t", () => ({ output: "final" })), "${t.output.output}"))).toBe("final");
});

// --- fan-out / fan-in ---

/** Return the diamond `START -> a -> {b, c} -> d -> END`, recording into `log`. */
function diamond(log: string[]): CompiledFlow {
  return graph(["a", "b", "c", "d"].map((id) => new RecordNode(id, log)), [
    [START_ID, "a"],
    ["a", "b"],
    ["a", "c"],
    ["b", "d"],
    ["c", "d"],
    ["d", END_ID],
  ]);
}

test("fan out runs all branches", async () => {
  const log: string[] = [];
  await engine(diamond(log)).run();
  expect(new Set(log)).toEqual(new Set(["a", "b", "c", "d"]));
});

test("multiple roots all start", async () => {
  // Two entry nodes fan out from START (as multi-reviewer flows do).
  const log: string[] = [];
  const flow = graph(["a", "b", "j"].map((id) => new RecordNode(id, log)), [
    [START_ID, "a"],
    [START_ID, "b"],
    ["a", "j"],
    ["b", "j"],
    ["j", END_ID],
  ]);
  const built = engine(flow);
  const events = await built.run();
  expect(new Set(log)).toEqual(new Set(["a", "b", "j"]));
  expect(log.filter((id) => id === "j")).toEqual(["j"]);
  expect(built.state.flow.startId).toBe(START_ID);
  expect(new Set(built.state.flow.outgoing(START_ID).map((edge) => edge.to))).toEqual(new Set(["a", "b"]));
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("diamond join fires exactly once", async () => {
  const log: string[] = [];
  await engine(diamond(log)).run();
  expect(log.filter((id) => id === "d")).toEqual(["d"]);
});

// --- outputs visible to successors ---

test("outputs written before successor runs", async () => {
  const seen: Record<string, unknown> = {};
  const flow = graph(
    [
      new FuncNode("up", () => ({ output: "payload" })),
      new FuncNode(
        "down",
        (inputs) => {
          seen["upstream"] = inputs["upstream"];
          return {};
        },
        { reads: { upstream: "${up.output.output}" } },
      ),
    ],
    [
      [START_ID, "up"],
      ["up", "down"],
      ["down", END_ID],
    ],
  );
  await engine(flow).run();
  expect(seen["upstream"]).toBe("payload");
});

// --- CASE branch and skip ---

/** Return `cond` routing to `handle` between `b` (`yes`) and `c` (else), recording into `log`. */
function caseFlow(handle: string, log: string[]): CompiledFlow {
  return graph([new BranchNode("cond", handle), new RecordNode("b", log), new RecordNode("c", log)], [
    [START_ID, "cond"],
    ["cond", "b", "yes"],
    ["cond", "c", ELSE_HANDLE],
    ["b", END_ID],
    ["c", END_ID],
  ]);
}

test("case takes branch and skips other", async () => {
  const log: string[] = [];
  const built = engine(caseFlow("yes", log));
  const events = await built.run();
  expect(log).toEqual(["b"]);
  expect(built.state.nodeState.get("c")).toBe(NodeState.SKIPPED);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("case else fallback", async () => {
  const log: string[] = [];
  await engine(caseFlow(ELSE_HANDLE, log)).run();
  expect(log).toEqual(["c"]);
});

test("skip floods through unreachable chain", async () => {
  // cond -else-> b -> c; selecting "yes" (a dead end into END) skips b and c.
  const log: string[] = [];
  const flow = graph([new BranchNode("cond", "yes"), new RecordNode("b", log), new RecordNode("c", log)], [
    [START_ID, "cond"],
    ["cond", "b", ELSE_HANDLE],
    ["cond", END_ID, "yes"],
    ["b", "c"],
    ["c", END_ID],
  ]);
  const built = engine(flow);
  await built.run();
  expect(log).toEqual([]);
  expect(built.state.nodeState.get("b")).toBe(NodeState.SKIPPED);
  expect(built.state.nodeState.get("c")).toBe(NodeState.SKIPPED);
});

test("data edged branch target does not run when skipped", async () => {
  // pro_note and con_note each have a control edge from gate and a data edge from synth, upstream
  // of the gate: routing to pro_note skips con_note although synth -> con_note is taken.
  const log: string[] = [];
  const nodes: [string, Node][] = [
    [START_ID, new StartNode(START_ID, { params: [] })],
    ["synth", new RecordNode("synth", log, "pro")],
    ["gate", new BranchNode("gate", "pro_note")],
    ["pro_note", new RecordNode("pro_note", log, "B")],
    ["con_note", new RecordNode("con_note", log, "b")],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ];
  const edges = [
    new Edge("s0", START_ID, "synth"),
    new Edge("synth->gate#0", "synth", "gate", { inputGroup: "cond" }),
    new Edge("gate->pro_note#0", "gate", "pro_note", { sourceHandle: "pro_note" }),
    new Edge("gate->con_note#0", "gate", "con_note", { sourceHandle: "con_note" }),
    new Edge("synth->pro_note#0", "synth", "pro_note", { inputGroup: "base" }),
    new Edge("synth->con_note#0", "synth", "con_note", { inputGroup: "base" }),
    new Edge("pro_note->end", "pro_note", END_ID),
    new Edge("con_note->end", "con_note", END_ID),
  ];
  const built = engine(CompiledFlow.fromParts(new Map(nodes), edges));
  const events = await built.run();
  expect(log).toContain("pro_note");
  expect(log).not.toContain("con_note");
  expect(built.state.nodeState.get("pro_note")).toBe(NodeState.TAKEN);
  expect(built.state.nodeState.get("con_note")).toBe(NodeState.SKIPPED);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

// --- a co-skipped terminal fails the run ---

/** Return `gate` routing to `a`, so `t` is skipped, with one output bound to `binding`. */
function gateFlow(binding: string): CompiledFlow {
  const log: string[] = [];
  return graph([new BranchNode("gate", "a"), new RecordNode("a", log, "A"), new RecordNode("t", log, "T")], [
    [START_ID, "gate"],
    ["gate", "a", "a"],
    ["gate", "t", "t"],
    ["a", END_ID],
    ["t", END_ID],
  ], { outputs: [new FlowOutput("result", binding)] });
}

test("skipped terminal producer fails run", async () => {
  const event = await last(gateFlow("${t.output}"));
  expect(event).toBeInstanceOf(RunFailed);
  expect((event as RunFailed).failure.message).toContain("skipped");
  expect((event as RunFailed).failure.code).toBe("graph.terminal_unreachable");
});

test("coalesce terminal one branch skipped succeeds", async () => {
  expect(await last(gateFlow("${a.output | t.output}"))).toBeInstanceOf(RunSucceeded);
});

test("does not misfire on optional terminal", async () => {
  expect(await output(gateFlow("${t.output:-null}"))).toBeNull();
});

test("genuine null terminal succeeds", async () => {
  expect(await output(oneOutput(new FuncNode("n", () => null), "${n.output}"))).toBeNull();
});

// --- failure ---

test("node failure produces graph failed", async () => {
  const flow = graph([new FuncNode("a", () => ({})), new FailNode("b", "nope"), new FuncNode("c", () => ({}))], [
    [START_ID, "a"],
    ["a", "b"],
    ["b", "c"],
    ["c", END_ID],
  ]);
  const built = engine(flow);
  const event = (await built.run()).at(-1);
  expect(event).toBeInstanceOf(RunFailed);
  expect((event as RunFailed).failure.message).toBe("nope");
  // A failed run stops: its signal aborts with the default reason.
  expect(built.signal.aborted).toBe(true);
  expect(built.signal.reason).toBeInstanceOf(DOMException);
  expect((built.signal.reason as DOMException).name).toBe("AbortError");
});

// --- write-boundary output enforcement ---

const NUMBER = new ValueContract(Type.scalar(ValueKind.NUMBER));

/** Return `START -> <id> -> END` for a node returning `fn()` under `contract`. */
function contracted(id: string, fn: () => unknown, contract: ValueContract | null): CompiledFlow {
  return graph([new FuncNode(id, fn, { outputContract: contract })], [
    [START_ID, id],
    [id, END_ID],
  ]);
}

test("output enforced rejects type mismatch", async () => {
  expect(await last(contracted("a", () => "oops", NUMBER))).toBeInstanceOf(RunFailed);
});

test("output enforced coerces and types storage", async () => {
  const built = engine(contracted("a", () => 3n, NUMBER));
  expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
  const stored = built.state.pool.getValue("a")!;
  expect(stored.kind).toBe("int");
  expect(stored.value).toBe(3n);
});

test("multi output record is a closed type", async () => {
  // Two or more declared outputs form a closed record type: every field required, no extras. A
  // missing or extra field fails at success preparation; the exact object is stored whole.
  const record = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["a", Type.scalar(ValueKind.STRING)],
        ["b", Type.scalar(ValueKind.STRING)],
      ]),
      required: new Set(["a", "b"]),
    }),
  );
  expect(await last(contracted("n", () => ({ a: "x" }), record))).toBeInstanceOf(RunFailed);
  expect(await last(contracted("n", () => ({ a: "x", b: "y", c: "z" }), record))).toBeInstanceOf(RunFailed);
  const built = engine(contracted("n", () => ({ a: "x", b: "y" }), record));
  expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
  expect(built.state.pool.get("n")).toEqual({ a: "x", b: "y" });
});

test("unresolvable declared type is not enforced", async () => {
  // The compiler leaves an unresolvable declared type unenforced: no output contract.
  const built = engine(contracted("a", () => ({ k: 1n }), null));
  expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
  expect(built.state.pool.get("a")).toEqual({ k: 1n });
});

test("undeclared output is not enforced", async () => {
  const built = engine(contracted("a", () => ({ other: "x" }), null));
  expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
  expect(built.state.pool.resolve("a", ["output", "other"])).toBe("x");
});

// --- the num_workers knob ---

test("fixture engine defaults serial", () => {
  const built = engine(graph([new RecordNode("a", [])], [
    [START_ID, "a"],
    ["a", END_ID],
  ]));
  expect(built.numWorkers).toBe(0);
});

test("nothing ready before a terminal result is nonterminal quiescence", async () => {
  const built = engine(graph([new RecordNode("a", [])], [
    [START_ID, "a"],
    ["a", END_ID],
  ]));
  built.state.ready.length = 0;
  const error = await rejected(built.run());
  expect(error).toEqual(new Error("FlowEngine reached nonterminal quiescence"));
  expect(built.signal.aborted).toBe(true);
  expect(built.log.nextSequence).toBe(1);
});
