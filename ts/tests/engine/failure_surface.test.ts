/**
 * Failure producers carry semantic origins, never YAML coordinates: a port of
 * `tests/engine/test_failure_surface.py`. Ported: the instance half of
 * `test_failure_producers_have_no_location_parameters_or_attributes`,
 * `test_source_span_is_absent_from_runtime_events`,
 * `test_failure_carriers_have_no_parallel_location_fields` and
 * `test_observers_share_the_same_failure_authority`.
 *
 * Changed: the carriers' own keys stand in for dataclass fields. The observers also hold their
 * `type` tag; a `RunResult` also holds `crashedAttempts`, reads `learning` and
 * `backwardOutcome` through getters, and keeps its backward handle outside the instance.
 *
 * Not ported: the `inspect.signature` half of
 * `test_failure_producers_have_no_location_parameters_or_attributes` (TypeScript has no keyword
 * parameters to inspect; an unknown option fails the type check).
 */

import { expect, test } from "bun:test";

import { Failure } from "../../src/diagnostics.ts";
import * as events from "../../src/events.ts";
import { NodeFailed, RunFailed } from "../../src/events.ts";
import { NodeAssertFailed } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { RunResult } from "../../src/run.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { NodeFailedRecord } from "../../src/runtime/log.ts";

test("source span is absent from runtime events", () => {
  expect("SourceSpan" in events).toBe(false);
});

test("failure carriers have no parallel location fields", () => {
  const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
  expect(Object.keys(new EvaluationFailure(failure))).toEqual(["failure"]);
  expect(Object.keys(new NodeFailed({ failure }))).toEqual(["type", "failure"]);
  expect(Object.keys(new RunFailed({ failure }))).toEqual(["type", "failure"]);
  expect(Object.keys(new NodeFailedRecord(failure))).toEqual(["failure", "forwardCapture"]);

  const result = new RunResult({ input: {}, status: "failed", runId: "r", log: null, failure });
  expect(Object.keys(result)).toEqual([
    "input",
    "status",
    "runId",
    "log",
    "output",
    "failure",
    "events",
    "crashedAttempts",
  ]);
  expect([result.learning, result.backwardOutcome]).toEqual([null, null]);
});

test("observers share the same failure authority", () => {
  const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
  expect(new NodeFailed({ failure }).failure).toBe(failure);
  expect(new RunFailed({ failure }).failure).toBe(failure);
});

test("failure producers have no location attributes", () => {
  const binding = new BindingError("bad", { failureCode: "binding.evaluate" });
  const assertion = new NodeAssertFailed("bad");

  for (const error of [binding, assertion]) {
    expect("locator" in error).toBe(false);
    expect("range" in error).toBe(false);
  }
});
