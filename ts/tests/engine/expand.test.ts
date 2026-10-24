/**
 * Runtime graph expansion: a port of `tests/engine/test_expand.py`.
 *
 * Ported: `test_rens_internal_input_and_output_heads` and
 * `test_rens_internal_computed_whole_span_renamespaces_all_leaves`.
 *
 * Ported, each for 0 and 4 workers, through `engine(..., { growBuilder: buildGrow })` (Python's
 * `_engine`, whose default builder is `build_grow`):
 * `test_call_grow_clones_namespaced_and_substitutes_spawner`,
 * `test_call_inside_map_expands_and_runs_uniformly`,
 * `test_deep_static_call_map_chain_loads_and_succeeds` and
 * `test_over_budget_expansion_trips_max_total_nodes`. The CALL rows build on the shared
 * `childFlow` and `parentWithSpawner` (Python's `_child_flow` and `_parent_with_spawner`).
 *
 * Changed:
 * - the CODE leaves' `tests.seeds.fns:confirm_action` and `tests.engine._compose_codefns:echo_x`
 *   references are inline bodies with the same behavior, since TS load refuses a Python reference;
 * - ints are `bigint`s (the child's value is `80n`), and the budget is `BUILTIN.maxTotalNodes`
 *   (Python's `Defaults()`).
 *
 * Ported, through `runFlow` and `advance` in an environment holding only a credential store
 * path in a fresh directory: `test_deep_static_call_map_chain_resumes_after_its_last_expansion`.
 */

import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow, rensInternal } from "../../src/compile/expand.ts";
import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN } from "../../src/defaults.ts";
import { NodeStarted, RunFailed, RunSucceeded } from "../../src/events.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Binding, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { advance, runFlow } from "../../src/run.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { childFlow, engine, intParameter, parentWithSpawner } from "./run_state_fixtures.ts";

test("rensInternal re-namespaces input and output heads", () => {
  // `${input.k}` reads the namespaced child START's output object; `${x.output}` is node-first.
  expect(rensInternal(new Binding("${input.k}"), "cs")).toEqual(new Binding("${cs/__start__.output.k}"));
  expect(rensInternal(new Binding("${x.output.field}"), "cs")).toEqual(new Binding("${cs/x.output.field}"));
  // `system.X` and a non-text source pass through untouched.
  expect(rensInternal(new Binding("${system.now}"), "cs")).toEqual(new Binding("${system.now}"));
  expect(rensInternal(new Binding(5n), "cs")).toEqual(new Binding(5n));

  expect(() => rensInternal("${input.k}" as unknown as Binding, "cs")).toThrow(
    new TypeError("compiled flow wiring must contain Binding values, got string"),
  );
});

test("rensInternal re-namespaces every leaf of a computed whole span", () => {
  expect(rensInternal(new Binding("${x.output - y.output}"), "cs")).toEqual(
    new Binding("${cs/x.output - cs/y.output}"),
  );
  // Embedded text and two separate spans: the literal text is kept, both spans re-namespaced.
  expect(rensInternal(new Binding("pre ${x.output} mid ${input.k} post"), "cs")).toEqual(
    new Binding("pre ${cs/x.output} mid ${cs/__start__.output.k} post"),
  );
});

// (7 + 1) * 10: a concrete child value flowing through the spawner substitution. The child has a
// single output `out`, so the committed spawner value unwraps to the bare scalar.
const EXPECTED_CHILD_VALUE = 80n;

test.each([0, 4])("CALL grow clones namespaced and substitutes the spawner (%p workers)", async (numWorkers) => {
  const built = engine(parentWithSpawner(childFlow()), { inputs: { x: 7n }, numWorkers, growBuilder: buildGrow });
  const events = await built.run();

  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  // `after` saw the child's value under `sp`.
  expect((events.at(-1) as RunSucceeded).output).toBe(EXPECTED_CHILD_VALUE);
  expect(built.state.nodeState.get("sp")).toBe(NodeState.EXPANDED);
  expect([...built.state.flow.nodes.keys()].some((nodeId) => nodeId.startsWith("sp/"))).toBe(true);
  expect(built.state.pool.get("sp")).toBe(EXPECTED_CHILD_VALUE);
  expect(built.state.flow.nodes.get("sp/__end__")!.commitAs).toBe("sp");
});

// A CALL-inside-MAP: the parent MAP `each` runs the child `mid`, which CALLs a CODE leaf. Uniform
// recursion namespaces the MAP element as `each#i` and the nested CALL's clone under it as
// `each#i/inner/...` (only the MAP layer adds the `#i` element suffix).
const CALL_INSIDE_MAP = `
id: parent
name: parent
input:
  xs: !list(str)
defs:
  leaf:
    input:
      v: !str
    nodes:
      e:
        kind: code
        input:
          answer: \${input.v}
        output: str
        code: return String(inputs["answer"] ?? "cancel")
    output: \${e.output}
  mid:
    input:
      v: !str
    nodes:
      inner:
        kind: call
        call: leaf
        input:
          v: \${input.v}
    output: \${inner.output}
nodes:
  each:
    kind: map
    over: \${input.xs}
    call: mid
    input:
      v: \${item}
output: \${each.output}
`;

test.each([0, 4])("CALL inside MAP expands and runs uniformly (%p workers)", async (numWorkers) => {
  const loaded = await loadFlow(CALL_INSIDE_MAP);
  const built = engine(loaded.compiled, { inputs: { xs: ["a", "b"] }, numWorkers, growBuilder: buildGrow });
  const events = await built.run();

  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  // Both elements ran through the nested CALL.
  expect((events.at(-1) as RunSucceeded).output).toEqual(["a", "b"]);
  // The doubly-namespaced clones exist in the live graph.
  const nodeIds = [...built.state.flow.nodes.keys()];
  expect(nodeIds.some((nodeId) => nodeId.startsWith("each#0/inner/"))).toBe(true);
  expect(nodeIds.some((nodeId) => nodeId.startsWith("each#1/inner/"))).toBe(true);
});

/**
 * A static chain `levels` CALL/MAP levels deep ending in a CODE leaf: the root CALLs `step1`, odd
 * steps MAP over a one-element list and even steps CALL. Each level is a distinct def, so the
 * chain is a genuine nesting the loader accepts as finite.
 */
function mixedChainSource(levels: number): string {
  const lines = [
    "id: chain",
    "name: chain",
    "input:",
    "  v: !str",
    "defs:",
    "  leaf:",
    "    input:",
    "      v: !str",
    "    nodes:",
    "      e:",
    "        kind: code",
    "        input:",
    "          answer: ${input.v}",
    "        output: str",
    '        code: return String(inputs["answer"] ?? "cancel")',
    "    output: ${e.output}",
  ];
  for (let k = 1; k < levels; k++) {
    const callee = k === levels - 1 ? "leaf" : `step${k + 1}`;
    lines.push(`  step${k}:`, "    input:", "      v: !str", "    nodes:");
    if (k % 2) {
      lines.push(
        "      next:",
        "        kind: map",
        // A whole-span list expression; a list literal would keep "${input.v}" as text.
        "        over: ${[input.v]}",
        `        call: ${callee}`,
        "        input:",
        "          v: ${item}",
      );
    } else {
      lines.push("      next:", "        kind: call", `        call: ${callee}`, "        input:", "          v: ${input.v}");
    }
    lines.push("    output: ${next.output}");
  }
  lines.push("nodes:", "  top:", "    kind: call", "    call: step1", "    input:", "      v: ${input.v}", "output: ${top.output}");
  return lines.join("\n");
}

const DEEP_LEVELS = 8;
// 4 MAP levels (odd steps 1, 3, 5, 7) wrap the leaf's text in 4 one-element lists.
const DEEP_OUTPUT = [[[["go"]]]];

test.each([0, 4])("deep static CALL/MAP chain loads and succeeds (%p workers)", async (numWorkers) => {
  // Nesting depth is static (the loader rejects recursion), so an 8-level chain runs to
  // completion; the growth core still stamps each level's depth.
  const loaded = await loadFlow(mixedChainSource(DEEP_LEVELS));
  const built = engine(loaded.compiled, { inputs: { v: "go" }, numWorkers, growBuilder: buildGrow });
  const events = await built.run();

  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
  expect((events.at(-1) as RunSucceeded).output).toEqual(DEEP_OUTPUT);
  expect(Math.max(...built.state.depth.values())).toBe(DEEP_LEVELS);
});

test("deep static CALL/MAP chain resumes after its last expansion", async () => {
  // Record replay re-validates every recorded growth (depth included): cut the deep run's log right
  // after its last `node_expanded` record, so resume must replay all 8 expansions and then run the
  // deepest leaf fresh.
  const dir = mkdtempSync(join(tmpdir(), "yian-expand-"));
  const env = hostEnv(dir);
  try {
    const source = mixedChainSource(DEEP_LEVELS);
    const path = join(dir, "deep.jsonl");
    const first = await runFlow(await loadFlow(source), { v: "go" }, { log: path, numWorkers: 0, env });
    expect(first.failure).toBeNull();
    const lines = readFileSync(path, "utf8").split("\n").slice(0, -1);
    const expanded = lines.flatMap((line, index) => (JSON.parse(line).kind === "node_expanded" ? [index] : []));
    expect(expanded).toHaveLength(DEEP_LEVELS);
    writeFileSync(path, `${lines.slice(0, expanded.at(-1)! + 1).join("\n")}\n`);

    const resumed = await advance({ log: path, env });

    expect(resumed.failure).toBeNull();
    expect(resumed.output).toEqual(DEEP_OUTPUT);
    const started = resumed.events.flatMap((event) => (event instanceof NodeStarted ? [event.nodeId] : []));
    // The leaf ran after replay.
    expect(started.some((nodeId) => nodeId.endsWith("/e"))).toBe(true);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

/**
 * A child flow whose node count alone exceeds the budget: one declared-output root `n0` and
 * `n - 1` extra root pads, all reading `${input.x}`. One clone busts the budget right after the
 * splice, before any node runs, so the pads never execute.
 */
function wideChild(n: number): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [intParameter("x")] })],
    [END_ID, EndNode.value(END_ID, { outputName: "out" })],
  ]);
  const edges: Edge[] = [];
  const wiring = new Map<string, Map<string, string>>([
    [START_ID, new Map([["x", "${input.x}"]])],
    [END_ID, new Map([["out", "${n0.output}"]])],
  ]);
  for (let k = 0; k < n; k++) {
    const nodeId = `n${k}`;
    const node = new CodeNode(nodeId, { code: 'return inputs["x"]' });
    node.params = [new Parameter("x")];
    nodes.set(nodeId, node);
    wiring.set(nodeId, new Map([["x", "${input.x}"]]));
    edges.push(new Edge(`__start__->${nodeId}#0`, START_ID, nodeId, { inputGroup: "x" }));
  }
  edges.push(new Edge("n0->__end__#0", "n0", END_ID, { inputGroup: "out" }));
  return CompiledFlow.fromParts(nodes, edges, { outputs: [new FlowOutput("out", "${n0.output}")], wiring });
}

test.each([0, 4])("an over-budget expansion trips the node budget (%p workers)", async (numWorkers) => {
  const parent = parentWithSpawner(wideChild(BUILTIN.maxTotalNodes + 10));
  const events = await engine(parent, { inputs: { x: 0n }, numWorkers, growBuilder: buildGrow }).run();

  expect(events.at(-1)).toBeInstanceOf(RunFailed);
  expect((events.at(-1) as RunFailed).failure.message).toContain("node budget");
});
