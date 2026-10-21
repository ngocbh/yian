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
 * Later (T4-8): `test_call_post_assert_survives_recovery_true` and
 * `test_call_post_assert_survives_recovery_false`: needs `resumeFlow` (a host stop, then a resume).
 */

import { expect, test } from "bun:test";

import { RunFailed } from "../../src/events.ts";
import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { childResolver } from "./run_state_fixtures.ts";

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
  const res = await runFlow(await load("      - output.n == 4"), {});
  expect(res.status).toBe("succeeded");
  expect(res.output).toEqual({ report: "report for ACME", n: 4n });
});

// --- 2. live, false -> loud failure --- //

test("call post assert false fails loudly", async () => {
  // A false `${output}` post-assert fails on the cloned child END (`research/__end__`), where the
  // lowered assertion rides.
  const loaded = await load("      - output.n == 999");
  const res = await runFlow(loaded, {});
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
  const res = await runFlow(await load("      - output.n == 4"), {});
  expect(res.status).toBe("succeeded");
});

test("call post assert reads input and output false", async () => {
  // An input+output post-assert that is false: `topic` ("ACME") is the recovered call input, so
  // `topic == "OTHER"` fails loudly.
  const res = await runFlow(await load('      - topic == "OTHER" and output.n == 4'), {});
  expect(res.status).toBe("failed");
  const failed = res.events.filter((event) => event instanceof RunFailed);
  expect(failed.length).toBeGreaterThan(0);
  expect(failed[0]!.failure.causeType).toBe("NodeAssertFailed");
  expect(res.failure!.nodeId).toBe("research/__end__");
});

test("call post assert reads input true", async () => {
  // The input ref resolves from the call's canonical growth seed (not null or dropped).
  const res = await runFlow(await load('      - topic == "ACME" and output.n == 4'), {});
  expect(res.status).toBe("succeeded");
});
