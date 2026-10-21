/**
 * The synthesized START/END boundary nodes: a port of `tests/engine/test_boundary_nodes.py`'s load
 * rows. The loader puts real `START_ID`/`END_ID` nodes into every flow (the strings persist as the
 * reserved node ids), and an `${input.X}` read is a data edge from START. The rows check that the
 * nodes exist and that edges carry real adjacency, never that the strings are absent.
 *
 * Ported: `test_loader_synthesizes_start_and_end_nodes`,
 * `test_inputs_read_mints_producer_edge_from_start`, `test_input_reader_is_not_also_a_bare_root`,
 * `test_inputs_resolve_to_start_node`, and the load half of
 * `test_terminal_output_of_a_skipped_producer_is_null`.
 *
 * Changed: the flow assembles through `assembleText`, where Python calls `load_flow`; its CODE
 * reference is TS's (`./fixtures/compose_codefns.ts:echo_x`).
 *
 * Not ported: `test_terminal_helpers_retired` and `test_pool_inputs_namespace_retired`: they check
 * that Python-era engine helpers and a pool namespace are gone, and TS never had them.
 *
 * Ported, after rewrite (`graph()` with a `FuncNode` reading `${input.x}`, for the loaded CODE
 * node): the engine half of `test_engine_schedules_start_and_emits_succeeded`.
 *
 * Ported through `loadConformanceFlow` and `runFlow`, in `hostEnv`: `test_run_result_is_end_node_value`,
 * the `run_flow` half of `test_engine_schedules_start_and_emits_succeeded` and the run half of
 * `test_terminal_output_of_a_skipped_producer_is_null`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, NodeSucceeded } from "../../src/events.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { runFlow } from "../../src/run.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { assembleText, loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, FuncNode, graph } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-boundary-nodes-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Load `text` with the conformance fixtures and run it on `inputs` in `hostEnv`. */
async function run(text: string, inputs: Record<string, unknown>) {
  return runFlow(await loadConformanceFlow(text), inputs, { env: hostEnv(dir) });
}

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

test("the run result is the END node's value", async () => {
  const result = await run(FLOW, { x: "ACME" });
  expect(result.status).toBe("succeeded");
  // END commits the value the run returns.
  expect(result.output).toBe("ACME");
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

const SKIPPED_TERMINAL = `
id: e7
name: e7
input:
  seed: !float
nodes:
  score:
    kind: code
    code: ./fixtures/compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "score.output >= 1"
        then: kept
    else: cold
  kept:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      topic: \${input.seed}
    output: str
  cold:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      topic: \${input.seed}
    output: str
output: \${kept.output}
`;

test("the terminal output of a skipped producer is null (loads)", () => {
  // END's group for an output the loader null-defaults is optional, so a skipped producer commits
  // null.
  const flow = assembleText(SKIPPED_TERMINAL).flow.compiled;
  expect(flow.outputs[0]!.binding.source).toBe("${kept.output:-null}");
});

test("the terminal output of a skipped producer is null (runs)", async () => {
  // score = 0, so `gate` routes to `cold` and `kept` is skipped.
  const result = await run(SKIPPED_TERMINAL, { seed: 0.0 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
});

test("inputs resolve to the START node", () => {
  const pool = new VariablePool();
  // START commits the bound input record, so `${input.x}` reads START's `x`.
  pool.set(START_ID, { x: "ACME" });
  expect(pool.resolve("input", ["x"])).toBe("ACME");
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
  const result = await run(FLOW, { x: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("ACME");
});
