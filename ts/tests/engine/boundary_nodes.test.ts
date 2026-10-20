/**
 * The synthesized START/END boundary nodes: a port of `tests/engine/test_boundary_nodes.py`'s load
 * rows. The loader puts real `START_ID`/`END_ID` nodes into every flow (the strings persist as the
 * reserved node ids), and an `${input.X}` read is a data edge from START. The rows check that the
 * nodes exist and that edges carry real adjacency, never that the strings are absent.
 *
 * Ported: `test_loader_synthesizes_start_and_end_nodes`,
 * `test_inputs_read_mints_producer_edge_from_start` and `test_input_reader_is_not_also_a_bare_root`.
 *
 * Changed: the flow assembles through `assembleText`, where Python calls `load_flow`; its CODE
 * reference is TS's (`./fixtures/compose_codefns.ts:echo_x`).
 *
 * Later (T4): the run rows (`test_run_result_is_end_node_value`, `test_terminal_helpers_retired`,
 * `test_terminal_output_of_a_skipped_producer_is_null`, `test_inputs_resolve_to_start_node`,
 * `test_pool_inputs_namespace_retired`, `test_engine_schedules_start_and_emits_succeeded`).
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { assembleText } from "../compose/declarations.ts";

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
  const pairs = new Set(flow.edges.map((edge) => `${edge.from} ${edge.to}`));
  expect(pairs.has(`${START_ID} n`) && pairs.has(`n ${END_ID}`)).toBe(true);
  // The n -> END_ID edge carries the output's input group (a real data edge).
  expect(flow.edges.some((edge) => edge.from === "n" && edge.to === END_ID && edge.inputGroup !== null)).toBe(true);
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
