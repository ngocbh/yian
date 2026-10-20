/**
 * The pooled engine (`numWorkers` above zero): a port of `tests/engine/test_parallel.py`.
 * Scheduling stays correct with workers in flight (the single-writer dispatcher gives the serial
 * outcomes: join-once, branch and skip), and independent branches overlap.
 *
 * Ported: `test_linear_is_deterministic_under_workers`, `test_diamond_join_once_under_workers`,
 * `test_case_skip_under_workers`, `test_veto_under_workers_data_edged_branch_skips`,
 * `test_terminal_skipped_fails_under_workers`, `test_failure_under_workers`,
 * `test_end_output_under_workers` and `test_no_declared_outputs_yields_none_under_workers`.
 *
 * Ported (adapted): `test_pooled_late_sibling_publishes_persisted_causal_terminal_failure`: `slow`
 * blocks on a promise `onEvent` resolves once `a` succeeded, and `onEvent` checks the state where
 * Python checked it while iterating the run. `test_fan_out_actually_overlaps`: overlap is shown
 * with promises, not timing: each branch blocks until all four have entered.
 *
 * TS-only: nothing ready and nothing in flight before a terminal result is nonterminal quiescence.
 *
 * Later (AsyncLocalStorage isolation): `test_workers_inherit_host_context_without_leaking_node_changes`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { NodeRouted, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
import { ELSE_HANDLE, type Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { NodeRoutedRecord } from "../../src/runtime/log.ts";
import {
  BlockingNode,
  BranchNode,
  engine,
  FailNode,
  FuncNode,
  graph,
  RecordNode,
  rejected,
} from "./run_state_fixtures.ts";

test("linear is deterministic under workers", async () => {
  const log: string[] = [];
  const flow = graph([new RecordNode("a", log), new RecordNode("b", log), new RecordNode("c", log)], [
    [START_ID, "a"],
    ["a", "b"],
    ["b", "c"],
    ["c", END_ID],
  ]);
  const events = await engine(flow, { numWorkers: 4 }).run();
  // A chain serializes whatever the pool size.
  expect(log).toEqual(["a", "b", "c"]);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("diamond join once under workers", async () => {
  const log: string[] = [];
  const flow = graph(
    [new RecordNode("a", log), new RecordNode("b", log), new RecordNode("c", log), new RecordNode("d", log)],
    [
      [START_ID, "a"],
      ["a", "b"],
      ["a", "c"],
      ["b", "d"],
      ["c", "d"],
      ["d", END_ID],
    ],
  );
  await engine(flow, { numWorkers: 4 }).run();
  expect(new Set(log)).toEqual(new Set(["a", "b", "c", "d"]));
  // The exactly-once join holds with workers in flight.
  expect(log.filter((id) => id === "d").length).toBe(1);
});

test("case skip under workers", async () => {
  const log: string[] = [];
  const flow = graph([new BranchNode("cond", "yes"), new RecordNode("b", log), new RecordNode("c", log)], [
    [START_ID, "cond"],
    ["cond", "b", "yes"],
    ["cond", "c", ELSE_HANDLE],
    ["b", END_ID],
    ["c", END_ID],
  ]);
  const built = engine(flow, { numWorkers: 4 });
  await built.run();
  expect(log).toEqual(["b"]);
  expect(built.state.nodeState.get("c")).toBe(NodeState.SKIPPED);
});

test("veto under workers data edged branch skips", async () => {
  // A branch target with a data edge from upstream of the gate still skips when its branch loses.
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
  const built = engine(CompiledFlow.fromParts(new Map(nodes), edges), { numWorkers: 4 });
  await built.run();
  expect(log).toContain("pro_note");
  expect(log).not.toContain("con_note");
  expect(built.state.nodeState.get("con_note")).toBe(NodeState.SKIPPED);
});

test("terminal skipped fails under workers", async () => {
  const flow = graph([new BranchNode("gate", "a"), new RecordNode("a", []), new RecordNode("t", [])], [
    [START_ID, "gate"],
    ["gate", "a", "a"],
    ["gate", "t", "t"],
    ["a", END_ID],
    ["t", END_ID],
  ], { outputs: [new FlowOutput("result", "${t.output}")] });
  const last = (await engine(flow, { numWorkers: 4 }).run()).at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.code).toBe("graph.terminal_unreachable");
});

test("pooled late sibling publishes persisted causal terminal failure", async () => {
  const release = Promise.withResolvers<void>();
  const flow = graph(
    [
      new BranchNode("gate", "a"),
      new RecordNode("a", []),
      new RecordNode("t", []),
      // Ends on its signal too, so a failing `expect` in `onEvent` rejects the run instead of
      // leaving the drain waiting on it.
      new BlockingNode("slow", release.promise.then(() => ({ output: "slow" }))),
    ],
    [
      [START_ID, "gate"],
      [START_ID, "slow"],
      ["gate", "a", "a"],
      ["gate", "t", "t"],
    ],
    { outputs: [new FlowOutput("result", "${t.output}")] },
  );
  let latched: Failure | null = null;
  const built = engine(flow, {
    numWorkers: 2,
    onEvent: (event) => {
      if (event instanceof NodeRouted) {
        expect(event.nodeId).toBe("gate");
        expect([...built.state.inFlight.keys()]).toEqual(["slow"]);
        expect(built.state.result).toBeNull();
        latched = built.state.terminalUnreachableFailure;
        expect(latched?.nodeId).toBe("gate");
        const routes = built.log.records().filter((envelope) => envelope.record instanceof NodeRoutedRecord);
        expect(routes.length).toBe(1);
        expect((routes[0]!.record as NodeRoutedRecord).terminalFailure).toEqual(latched!);
      } else if (event instanceof NodeSucceeded && event.nodeId === "a") {
        expect(latched).not.toBeNull();
        expect([...built.state.inFlight.keys()]).toEqual(["slow"]);
        release.resolve();
      }
    },
  });
  const events = await built.run();

  expect(latched).not.toBeNull();
  expect(events.at(-2)).toBeInstanceOf(NodeSucceeded);
  expect((events.at(-2) as NodeSucceeded).nodeId).toBe("slow");
  expect(events.at(-1)).toBeInstanceOf(RunFailed);
  expect((events.at(-1) as RunFailed).failure).toBe(latched!);
});

test("fan out actually overlaps", async () => {
  // Each branch blocks until all four have entered: without four workers in flight at once, the
  // run never settles and the test times out.
  const allEntered = Promise.withResolvers<void>();
  const branches = [0, 1, 2, 3].map(
    (index) => new BlockingNode(`s${index}`, allEntered.promise.then(() => ({ output: `s${index}` }))),
  );
  void Promise.all(branches.map((branch) => branch.entered)).then(() => allEntered.resolve());
  const nodes: Node[] = [new FuncNode("a", () => ({})), ...branches, new FuncNode("join", () => ({ output: "ok" }))];
  const edges: [string, string][] = [[START_ID, "a"]];
  for (const branch of branches) edges.push(["a", branch.id], [branch.id, "join"]);
  edges.push(["join", END_ID]);

  const events = await engine(graph(nodes, edges), { numWorkers: 4 }).run();

  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  expect(branches.every((branch) => branch.released)).toBe(true);
});

test("failure under workers", async () => {
  const flow = graph([new FuncNode("a", () => ({})), new FailNode("b", "nope")], [
    [START_ID, "a"],
    ["a", "b"],
    ["b", END_ID],
  ]);
  const last = (await engine(flow, { numWorkers: 4 }).run()).at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.message).toBe("nope");
});

/** Return `START -> a -> t -> END`, where `t` returns `{output: "final"}`. */
function terminalFlow(outputs?: readonly FlowOutput[]): CompiledFlow {
  return graph(
    [new FuncNode("a", () => ({})), new FuncNode("t", () => ({ output: "final" }))],
    [
      [START_ID, "a"],
      ["a", "t"],
      ["t", END_ID],
    ],
    outputs === undefined ? {} : { outputs },
  );
}

test("end output under workers", async () => {
  // One declared output bound to `t`'s whole value: END returns that value.
  const flow = terminalFlow([new FlowOutput("result", "${t.output}")]);
  const last = (await engine(flow, { numWorkers: 4 }).run()).at(-1);
  expect((last as RunSucceeded).output).toEqual({ output: "final" });
});

test("no declared outputs yields null under workers", async () => {
  const last = (await engine(terminalFlow(), { numWorkers: 4 }).run()).at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  expect((last as RunSucceeded).output).toBeNull();
});

test("nothing ready and nothing in flight before a terminal result is nonterminal quiescence", async () => {
  const built = engine(graph([new RecordNode("a", [])], [
    [START_ID, "a"],
    ["a", END_ID],
  ]), { numWorkers: 2 });
  built.state.ready.length = 0;
  const error = await rejected(built.run());
  expect(error).toEqual(new Error("FlowEngine reached nonterminal quiescence"));
  expect(built.signal.aborted).toBe(true);
  expect(built.log.nextSequence).toBe(1);
});
