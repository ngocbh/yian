/**
 * Run-result and static edge-id goldens for CODE-only flows: a port of
 * `tests/engine/test_golden_run_results.py`.
 *
 * An `ask` answered by its host and a `sleep` until a past date finish in one run; a single
 * `${input.x}` reader has only its START data edge and its END edge; a defaulted input fills inside
 * START while `RunResult.input` keeps what the caller passed; and the three output shapes (no
 * `output:`, a bare `output:`, a name map) give null, the value, and a record.
 *
 * Ported: `test_golden_human_input_then_past_wait_completes_in_one_run`,
 * `test_golden_static_edge_ids_start_end`,
 * `test_golden_inputs_defaults_and_multi_output_surface`, `test_golden_output_arity_0`,
 * `test_golden_output_arity_1` and `test_golden_output_arity_2`. Ints read as `bigint`s (decision
 * 4a); the CODE references are TS's (`./fixtures/compose_codefns.ts:make_report`,
 * `./fixtures/seeds.ts:confirm_action`). Changed: the first row's gate is a TOOL `ask` and its
 * wait a TOOL `sleep`, where Python's are HUMAN_INPUT and WAIT nodes (TS yian has neither kind).
 *
 * Already ported: `test_golden_call_run_result`, `test_golden_map_run_result` and
 * `test_golden_static_edge_ids_case_and_call_and_map`, in `graph_growth_e2e.test.ts`'s
 * `f0 goldens hold`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-golden-run-results-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Load `text` and run it on `inputs` in the test's environment. */
async function run(
  text: string,
  inputs: Record<string, unknown>,
  options: RunFlowOptions = {},
): Promise<RunResult> {
  return runFlow(await loadConformanceFlow(text, { env }), inputs, { env, ...options });
}

// --- an `ask` followed by a `sleep` until a UTC date: seed 17's shape, agent-free --- //

const EFFECTS = `
id: e
name: e
typedefs:
  Approval: Literal(approve, reject)
input:
  settle_at: !date
nodes:
  approve:
    kind: tool
    tool_id: ask
    input:
      prompt: "approve? (approve/reject)"
    output: Approval
  gate:
    kind: case
    on: \${approve.output}
    cases:
      - when: approve
        then: settle
    else: abort
  settle:
    kind: tool
    tool_id: sleep
    input:
      until: \${input.settle_at}
  confirm:
    kind: code
    depends_on: [settle]
    input:
      answer: \${approve.output}
    output: str
    code: ./fixtures/seeds.ts:confirm_action
  abort:
    kind: code
    input:
      answer: \${approve.output}
    output: str
    code: ./fixtures/seeds.ts:confirm_action
output: \${confirm.output | abort.output}
`;

test("golden human input then past wait completes in one run", async () => {
  // The ask completes through its host; the past sleep returns in the same run.
  const result = await run(EFFECTS, { settle_at: "2000-01-01" }, { tools: { ask: () => "approve" } });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("approve");
});

// --- a single `${input.x}` reader with a bare output: `START_ID -> body -> END_ID` --- //

const STATIC_EDGE_FLOW = `
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

test("golden static edge ids start end", async () => {
  const flow = (await loadConformanceFlow(STATIC_EDGE_FLOW, { env })).compiled;
  expect(flow.edges.map((edge) => edge.id).sort()).toEqual([
    "__start__->n#0", // the `${input.x}` data edge; no bare root edge
    "n->__end__#0", // the `output: ${n.output}` edge
  ]);
});

// --- the input/output surface: an input default and a name-map output --- //

const SURFACE = `
id: surface
name: surface
input:
  topic: !str
  window: !int 30
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
      window: \${input.window}
    output:
      report: str
      n: int
    code: ./fixtures/compose_codefns.ts:make_report
output:
  report: \${emit.output.report}
  n: \${emit.output.n}
`;

test("golden inputs defaults and multi output surface", async () => {
  // The omitted `window` fills inside START (n = 4 uses window 30), so `RunResult.input` holds
  // only what the caller passed.
  const res = await run(SURFACE, { topic: "ACME" });
  expect(res.status).toBe("succeeded");
  expect(res.input).toEqual({ topic: "ACME" });
  expect(res.output).toEqual({ report: "report for ACME", n: 4n });
});

// --- the declared output shapes: none, a bare value, a name map --- //

const ARITY_0 = `
id: arity0
name: arity0
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      topic: \${input.x}
    output: str
    code: ./fixtures/compose_codefns.ts:echo
`;

const ARITY_1 = `
id: arity1
name: arity1
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      topic: \${input.x}
    output: str
    code: ./fixtures/compose_codefns.ts:echo
output: \${n.output}
`;

const ARITY_2 = `
id: arity2
name: arity2
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      topic: \${input.x}
    output:
      report: str
      n: int
    code: ./fixtures/compose_codefns.ts:make_report
output:
  report: \${n.output.report}
  count: \${n.output.n}
`;

test("golden output arity 0", async () => {
  // No `output:` section: the run's output is null.
  const res = await run(ARITY_0, { x: "ACME" });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBeNull();
});

test("golden output arity 1", async () => {
  // A bare `output: ${n.output}`: the bare value.
  const res = await run(ARITY_1, { x: "ACME" });
  expect(res.status).toBe("succeeded");
  expect(res.output).toBe("ACME");
});

test("golden output arity 2", async () => {
  // A name-map `output:`: a record keyed by output name.
  const res = await run(ARITY_2, { x: "ACME" });
  expect(res.status).toBe("succeeded");
  expect(res.output).toEqual({ report: "report for ACME", count: 4n });
});
