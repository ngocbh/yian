/**
 * The synthesized START/END boundary nodes: a port of `tests/engine/test_boundary_nodes.py`'s load
 * rows. The loader puts real `START_ID`/`END_ID` nodes into every flow (the strings persist as the
 * reserved node ids), and an `${input.X}` read is a data edge from START. The rows check that the
 * nodes exist and that edges carry real adjacency, never that the strings are absent.
 *
 * Ported: `test_loader_synthesizes_start_and_end_nodes`,
 * `test_inputs_read_mints_producer_edge_from_start` and
 * `test_input_reader_is_not_also_a_bare_root`.
 *
 * Changed: the flow assembles through `assembleText`, where Python calls `load_flow`; its CODE
 * reference is TS's (`./fixtures/compose_codefns.ts:echo_x`).
 *
 * Ported: `test_terminal_helpers_retired`, `test_inputs_resolve_to_start_node` and
 * `test_pool_inputs_namespace_retired` (the pool has no `addInputs` and no `inputs` field).
 *
 * Ported, after rewrite (`graph()` with a `FuncNode` reading `${input.x}`, for the loaded CODE
 * node): the engine half of `test_engine_schedules_start_and_emits_succeeded`.
 *
 * Later (T4-8), through `run_flow`: `test_run_result_is_end_node_value` and the `run_flow` half of
 * `test_engine_schedules_start_and_emits_succeeded`. Later (T3-20, T4-8):
 * `test_terminal_output_of_a_skipped_producer_is_null` (its load half needs `nullSkippedOutputs`).
 */

import { expect, test } from "bun:test";

import { Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, NodeSucceeded } from "../../src/events.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { assembleText } from "../compose/declarations.ts";
import { engine, FuncNode, graph } from "./run_state_fixtures.ts";

const FLOW = `
id: f
name: f
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      x: \${input.x}
    output: str
    code: ./fixtures/compose_codefns.ts:echo_x
output: \${n.output}
`;

test("the loader synthesizes START and END nodes", () => {
  const flow = assembleText(FLOW).flow.compiled;
  // START_ID and END_ID identify real synthesized nodes.
  expect(flow.nodes.get(START_ID)!.kind).toBe(NodeKind.START);
  expect(flow.nodes.get(END_ID)!.kind).toBe(NodeKind.END);
  // START_ID is the single root; END_ID is the terminal.
  expect(flow.startId).toBe(START_ID);
  expect(flow.terminalId).toBe(END_ID);
  // Ordinary edges to and from the real boundary nodes (n -> END_ID is a real adjacency).
  const pairs = flow.edges.map((edge) => `${edge.from} ${edge.to}`);
  expect(pairs).toContain(`${START_ID} n`);
  expect(pairs).toContain(`n ${END_ID}`);
  // The n -> END_ID edge carries the output's input group (a real data edge).
  expect(flow.edges).toContainEqual(new Edge(`n->${END_ID}#0`, "n", END_ID, { inputGroup: "result" }));
});

test("an input read mints a producer edge from START", () => {
  // `${input.x}` on node `n` depends on START_ID: a START_ID->n data edge in input group x.
  const flow = assembleText(FLOW).flow.compiled;
  expect(flow.edges.some((edge) => edge.from === START_ID && edge.to === "n" && edge.inputGroup === "x")).toBe(true);
});

test("an input reader is not also a bare root", () => {
  // A node reading `${input.x}` gets the data edge only, not a second bare START_ID->n root edge.
  const flow = assembleText(FLOW).flow.compiled;
  const startToN = flow.edges.filter((edge) => edge.from === START_ID && edge.to === "n");
  expect(startToN).toHaveLength(1);
  expect(startToN[0]!.inputGroup).toBe("x");
});

test("the terminal helpers are retired", () => {
  // END commits the run's value; the disposition handles a skip.
  for (const name of ["terminalOutput", "emitTerminal", "terminalCoskip"]) {
    expect(name in FlowEngine.prototype).toBe(false);
  }
});

test("inputs resolve to the START node", () => {
  const pool = new VariablePool();
  // START commits the bound input record, so `${input.x}` reads START's `x`.
  pool.set(START_ID, { x: "ACME" });
  expect(pool.resolve("input", ["x"])).toBe("ACME");
});

test("the pool inputs namespace is retired", () => {
  const pool = new VariablePool();
  expect("addInputs" in pool).toBe(false);
  expect("inputs" in pool).toBe(false);
});

test("the engine schedules START and emits its success", async () => {
  // START is an ordinary scheduled node: the engine primes it with the raw run arguments and it
  // commits its bound record through the evaluator like any node.
  const flow = graph([new FuncNode("n", (inputs) => inputs.x, { reads: { x: "${input.x}" } })], [
    [START_ID, "n"],
    ["n", END_ID],
  ]);
  const built = engine(flow, { inputs: { x: "ACME" } });

  const events = await built.run();

  expect(events.some((event) => event instanceof NodeStarted && event.nodeId === START_ID)).toBe(true);
  expect(events.some((event) => event instanceof NodeSucceeded && event.nodeId === START_ID)).toBe(true);
  expect(built.state.pool.get(START_ID)).toEqual({ x: "ACME" });
});
