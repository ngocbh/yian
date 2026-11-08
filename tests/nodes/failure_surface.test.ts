/**
 * Failure producers carry semantic origins, never YAML coordinates: failure carriers and observers
 * hold no location fields and share one failure authority.
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
  expect(Object.keys(new NodeFailedRecord(failure))).toEqual(["failure"]);

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
    "memoryDrift",
    "waits",
    "unread",
  ]);
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
