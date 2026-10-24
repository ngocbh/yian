/**
 * A `call` node's node-level post `asserts:` fire on the cloned child END: a port of
 * `tests/engine/test_call_postasserts.py`.
 *
 * A `call` is a spawner: its `run` resolves to a `Grow`, and its `${output}` is committed when the
 * cloned child END finishes (`commitAs` the callsite). `callSubgraph` lowers the call's post
 * asserts onto that END, so a false assertion fails the child END (e.g. `research/__end__`), which
 * owns the committed call output and its postconditions.
 *
 * Ported: `test_call_post_assert_true_passes`, `test_call_post_assert_false_fails_loudly`,
 * `test_call_post_assert_reads_input_and_output_true`,
 * `test_call_post_assert_reads_input_and_output_false` and `test_call_post_assert_reads_input_true`.
 * Integers are bigints (decision 4a); the CODE reference is the TS fixture's
 * (`./fixtures/compose_codefns.ts:make_report`).
 *
 * Ported through `runFlow` and `advance`, in `hostEnv`: `test_call_post_assert_survives_recovery_true`
 * and `test_call_post_assert_survives_recovery_false`. Changed: the interruption before the human
 * answer is the run's `signal`, which the `ask` handler aborts (Python: the handler raises
 * `_HostStop`); the child's gate is a TOOL `ask` where Python's is a HUMAN_INPUT node (TS yian has
 * no HUMAN_INPUT kind).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { RunFailed } from "../../src/events.ts";
import { advance, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { childResolver, rejected } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-call-postasserts-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// A child that re-exports a {report, n} record; the parent's call `${output}` is that record,
// with `n == len(topic)`.
const CHILD = `
id: child-one
name: child_one
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output:
      report: str
      n: int
    code: ./fixtures/compose_codefns.ts:make_report
output:
  report: \${emit.output.report}
  n: \${emit.output.n}
`;

// A parent that calls child-one with a fixed topic, carrying the given `asserts:` block on the
// call node. `topic` ("ACME") is a declared call input, so n == 4.
function parent(asserts: string): string {
  return `
id: call-parent
name: call_parent
uses:
  child-one: child-one
nodes:
  research:
    kind: call
    call: child-one
    input:
      topic: ACME
    asserts:
${asserts}
output: \${research.output}
`;
}

function load(asserts: string) {
  return loadConformanceFlow(parent(asserts), { childResolver: childResolver({ "child-one": CHILD }) });
}

// --- 1. live, passing --- //

test("call post assert true passes", async () => {
  // A true `${output.field}` post-assert: the run succeeds.
  const res = await runFlow(await load("      - output.n == 4"), {}, { env: hostEnv(dir) });
  expect(res.status).toBe("succeeded");
  expect(res.output).toEqual({ report: "report for ACME", n: 4n });
});

// --- 2. live, false -> loud failure --- //

test("call post assert false fails loudly", async () => {
  // A false `${output}` post-assert fails on the cloned child END (`research/__end__`), where the
  // lowered assertion rides.
  const loaded = await load("      - output.n == 999");
  const res = await runFlow(loaded, {}, { env: hostEnv(dir) });
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.nodeId).toBe("research/__end__");
  expect(loaded.diagnose(res.failure!).frames.length).toBeGreaterThan(0);
});

// --- 3. live, input + output ref --- //

test("call post assert reads input and output true", async () => {
  // The baseline true assertion: n == len("ACME") == 4.
  const res = await runFlow(await load("      - output.n == 4"), {}, { env: hostEnv(dir) });
  expect(res.status).toBe("succeeded");
});

test("call post assert reads input and output false", async () => {
  // An input+output post-assert that is false: `topic` ("ACME") is the recovered call input, so
  // `topic == "OTHER"` fails loudly.
  const res = await runFlow(await load('      - topic == "OTHER" and output.n == 4'), {}, { env: hostEnv(dir) });
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.nodeId).toBe("research/__end__");
});

test("call post assert reads input true", async () => {
  // The input ref resolves from the call's canonical growth seed (not null or dropped).
  const res = await runFlow(await load('      - topic == "ACME" and output.n == 4'), {}, { env: hostEnv(dir) });
  expect(res.status).toBe("succeeded");
});

// --- 4. durable: the input record survives interruption and recovery --- //

// A child whose `ask` leaf the host interrupts; the parent's call post-assert reads both
// the declared input `action` and `${output}`.
const BLOCKING_CHILD = `
id: approver
name: approver
input:
  action: !str
nodes:
  approve:
    kind: tool
    tool_id: ask
    input:
      prompt: "Approve \${input.action}?"
    output: str
output: \${approve.output}
`;

function loadBlocking(asserts: string) {
  return loadConformanceFlow(
    `
id: gate-parent
name: gate_parent
uses:
  approver: approver
input:
  action: !str
nodes:
  gate:
    kind: call
    call: approver
    input:
      action: \${input.action}
    asserts:
${asserts}
output: \${gate.output}
`,
    { childResolver: childResolver({ approver: BLOCKING_CHILD }) },
  );
}

/** Interrupt the started ask, then recover it from its run log with `answer`. */
async function driveDurable(name: string, asserts: string, answer: string) {
  const path = join(dir, `${name}.jsonl`);
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };
  const caught = await rejected(
    runFlow(
      await loadBlocking(asserts),
      { action: "deploy" },
      { log: path, signal: host.signal, tools: { ask: interrupt }, env: hostEnv(dir) },
    ),
  );
  expect(caught).toBe(interruption);
  return advance({ log: path, tools: { ask: () => answer }, env: hostEnv(dir) });
}

test("call post assert survives recovery true", async () => {
  // A true input+output post-assert fires after recovery; `action` comes from the recorded
  // NodeExpanded seed.
  const res = await driveDurable("true", '      - action == "deploy" and output == "yes"', "yes");
  expect(res.status, res.failure?.message).toBe("succeeded");
  expect(res.output).toBe("yes");
});

test("call post assert survives recovery false", async () => {
  // A false post-assert after recovery fails, located at the cloned child END.
  const res = await driveDurable("false", '      - action == "deploy" and output == "no"', "yes");
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.nodeId).toBe("gate/__end__");
});
