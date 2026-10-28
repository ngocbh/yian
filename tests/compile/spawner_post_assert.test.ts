/**
 * A `call` node's node-level post `asserts:` fire from the cloned child END. The grow builder
 * lowers them onto the cloned child END, so a failing assert fails the child END (e.g.
 * `callee/__end__`), not the spawner. `${output}` stays as written; a declared call input `${X}`
 * maps to a synthesized END parameter wired off the cloned child START's coerced output, allocated
 * so it never collides with a parameter the child already lowered, and shared by repeated
 * references.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { callSubgraph } from "../../src/compile/expand.ts";
import type { Flow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { RunFailed } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Assertion } from "../../src/nodes/contracts.ts";
import { advance, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { childResolver, rejected } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-spawner-post-assert-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// The prefix `callSubgraph` gives a synthesized assertion parameter (unexported in TS).
const SYNTH_PREFIX = "__assert_";

// A child that doubles its int input and re-exports it as `${output}` (an int).
const ECHO_CHILD = `
id: echo-child
name: echo_child
input:
  v: !int
nodes:
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
output: \${x.output}
`;

// --- 1. a `${output}` post-assert fires from the cloned child END --- //

// A parent that calls echo-child with a fixed value, carrying a node-level `output < 100`
// post-assert; the child doubles v, so output == 2 * value.
function outputParent(value: number): string {
  return `
id: out-parent
name: out_parent
uses:
  echo-child: echo-child
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: ${value}
    asserts:
      - output < 100
output: \${callee.output}
`;
}

function loadOutput(value: number): Promise<ComposedFlow> {
  return loadConformanceFlow(outputParent(value), { childResolver: childResolver({ "echo-child": ECHO_CHILD }) });
}

test("output post assert satisfying succeeds", async () => {
  // v=5 -> output 10 < 100: the assert holds and the run succeeds.
  const res = await runFlow(await loadOutput(5), {});
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe(10n);
});

test("output post assert violating fails on child end", async () => {
  // v=60 -> output 120 violates output < 100: a loud failure fired on the child END.
  const res = await runFlow(await loadOutput(60), {});
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.message.toLowerCase()).toContain("post-assert");
  // The failing node is the cloned child END (callsite + end id), not the spawner.
  expect(res.failure!.nodeId).toBe("callee/__end__");
});

// --- 2. a declared-input post-assert maps to the cloned child START's output --- //

// The call declares an input `v` seeded from the run arg; the post assert reads both that declared
// input (`${v}`) and `${output}`, so it fires on the cloned child END.
function inputParent(seedExpr: string): string {
  return `
id: in-parent
name: in_parent
uses:
  echo-child: echo-child
input:
  v: !int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: ${seedExpr}
    asserts:
      - v >= 0 and output > 0
output: \${callee.output}
`;
}

function loadInput(): Promise<ComposedFlow> {
  return loadConformanceFlow(inputParent("${input.v}"), {
    childResolver: childResolver({ "echo-child": ECHO_CHILD }),
  });
}

test("declared input post assert satisfying succeeds", async () => {
  // v=5 >= 0 holds; output 10.
  const res = await runFlow(await loadInput(), { v: 5n });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe(10n);
});

test("declared input post assert violating fails", async () => {
  // v=-3 < 0 violates v >= 0: a loud failure on the child END.
  const res = await runFlow(await loadInput(), { v: -3n });
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.nodeId).toBe("callee/__end__");
});

// --- 3. recovery safety: the lowered spawner post-assert survives log replay --- //

// A child whose live `ask` can be interrupted after its durable start.
const BLOCKING_CHILD = `
id: approver
name: approver
input:
  v: !int
nodes:
  approve:
    kind: tool
    tool_id: ask
    input:
      prompt: "value \${input.v}?"
    output: int
output: \${approve.output}
`;

// The call carries a node-level `output < 100` post-assert, which the rebuilt child END checks
// after recovery retries the unfinished ask.
const BLOCKING_PARENT = `
id: recovery-parent
name: recovery_parent
uses:
  approver: approver
input:
  v: !int
nodes:
  gate:
    kind: call
    call: approver
    input:
      v: \${input.v}
    asserts:
      - output < 100
output: \${gate.output}
`;

function loadBlocking(): Promise<ComposedFlow> {
  return loadConformanceFlow(BLOCKING_PARENT, { childResolver: childResolver({ approver: BLOCKING_CHILD }) });
}

test("spawner post assert fires after log recovery", async () => {
  // Interrupt the ask after its NodeStarted is durable; recovery replays the expansion,
  // retries the ask and delivers 120, which the rebuilt child END's post-assert rejects.
  const path = join(dir, "spawner-post-assert.jsonl");
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };
  const caught = await rejected(
    runFlow(await loadBlocking(), { v: 1 }, { log: path, signal: host.signal, tools: { ask: interrupt }, env: hostEnv(dir) }),
  );
  expect(caught).toBe(interruption);

  const res = await advance({ log: path, tools: { ask: () => 120 }, env: hostEnv(dir) });

  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.message.toLowerCase()).toContain("post-assert");
});

// --- collision-free synthesized assertion parameters --- //

// A child def that doubles its input and carries its own flow-level post assert reading
// `${input.v}` (and `${x.output}`, which makes it a post assert). Build lowers it onto the def's
// END as the synthesized `__assert_input_v`; the call's own post assert reads the same declared
// input `${v}`, so spawner lowering must allocate a distinct parameter.
const ECHO_CHILD_WITH_INPUT_ASSERT = `
id: echo-child
name: echo_child
input:
  v: !int
asserts:
  - "input.v >= 0 and x.output >= 0"
nodes:
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
output: \${x.output}
`;

const COLLISION_PARENT = `
id: collision-parent
name: collision_parent
uses:
  echo-child: echo-child
input:
  v: !int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: \${input.v}
    asserts:
      - "v < 100 and output > 0"
output: \${callee.output}
`;

function loadCollision(): Promise<ComposedFlow> {
  return loadConformanceFlow(COLLISION_PARENT, {
    childResolver: childResolver({ "echo-child": ECHO_CHILD_WITH_INPUT_ASSERT }),
  });
}

/**
 * Build the CALL's grow subgraph with `callSubgraph` (the builder the CALL grows through) and
 * return the cloned child END and the subgraph.
 */
function clonedChildEnd(loaded: ComposedFlow) {
  const callee = loaded.compiled.nodes.get("callee")! as CallNode;
  const flow = callSubgraph(callee.child as Flow, "callee", { v: "${input.v}" }, {
    postAsserts: callee.postAsserts,
    inputNames: new Set((callee.params ?? []).map((param) => param.name)),
  });
  return { end: flow.nodes.get(flow.endId)!, flow };
}

test("collision free cloned end has no duplicate params or edges", async () => {
  // The cloned child END carries no duplicate synthesized parameter or second producer edge
  // sharing its input group.
  const { end, flow } = clonedChildEnd(await loadCollision());

  const names = (end.params ?? []).map((param) => param.name);
  expect(new Set(names).size).toBe(names.length);

  // No two edges into the END share an input group (a duplicate would feed the join twice).
  const groups = flow.edges
    .filter((edge) => edge.to === end.id && edge.inputGroup !== null)
    .map((edge) => edge.inputGroup);
  expect(new Set(groups).size).toBe(groups.length);

  // Both the child's own `__assert_input_v` and the spawner's synthesized parameter are present.
  const synths = names.filter((name) => name.startsWith(`${SYNTH_PREFIX}input_v`));
  expect(synths).toHaveLength(2);
});

test("spawner lowering preserves presentation origin", async () => {
  // CALL expansion rewrites only the executable text and keeps the provenance.
  const loaded = await loadOutput(5);
  const callee = loaded.compiled.nodes.get("callee")! as CallNode;
  const origin = new OriginRef("nodes/callee/asserts/0");
  const assertion = new Assertion("v >= 0 and output > 0", "callee input and output must be positive", { origin });

  const flow = callSubgraph(callee.child as Flow, "callee", { v: 5n }, {
    postAsserts: [assertion],
    inputNames: new Set(["v"]),
  });
  const lowered = flow.nodes.get(flow.endId)!;

  expect(lowered.postAsserts.at(-1)!.expression).toBe("__assert_input_v >= 0 and output > 0");
  expect(lowered.postAsserts.at(-1)!.display).toBe(assertion.display);
  expect(lowered.postAsserts.at(-1)!.origin).toBe(origin);
});

test("collision case run satisfying succeeds", async () => {
  // v=5 satisfies both `input.v >= 0` (child) and `v < 100` (spawner); output 10.
  const res = await runFlow(await loadCollision(), { v: 5n });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe(10n);
});

test("collision case run violating spawner assert fails on child end", async () => {
  // v=120 satisfies the child's `>= 0` but violates the spawner's `< 100`: a post-assert failure
  // on the cloned child END.
  const res = await runFlow(await loadCollision(), { v: 120n });
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.message.toLowerCase()).toContain("post-assert");
  expect(res.failure!.nodeId).toBe("callee/__end__");
});

// --- 5. repeated refs to the same declared input share one synthesized parameter --- //

const REPEATED_REF_PARENT = `
id: repeat-parent
name: repeat_parent
uses:
  echo-child: echo-child
input:
  v: !int
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: \${input.v}
    asserts:
      - "v >= 0 and v < 100 and output > 0"
output: \${callee.output}
`;

function loadRepeated(): Promise<ComposedFlow> {
  // The plain child (no asserts of its own), so only the spawner's synthesized parameter appears.
  return loadConformanceFlow(REPEATED_REF_PARENT, { childResolver: childResolver({ "echo-child": ECHO_CHILD }) });
}

test("repeated input ref shares one synth param", async () => {
  const { end } = clonedChildEnd(await loadRepeated());
  const synths = (end.params ?? [])
    .map((param) => param.name)
    .filter((name) => name.startsWith(`${SYNTH_PREFIX}input_v`));
  expect(synths).toHaveLength(1);
});

test("repeated input ref run succeeds", async () => {
  const res = await runFlow(await loadRepeated(), { v: 5n });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe(10n);
});

test("a spawner post assert set in code that doesn't parse fails before the run", async () => {
  // A spawner's bad post-assert set in code is named before the run, not thrown bare at growth.
  const loaded = await loadOutput(5);
  loaded.compiled.nodes.get("callee")!.postAsserts = [new Assertion("${output} < 100", "under 100")];

  const error = await rejected(runFlow(loaded, {}));
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as ExpressionError).message).toMatch(
    /^node 'callee' post-assert 'under 100' \(`\$\{output\} < 100`\) is not a valid boolean expression: /,
  );
});
