/**
 * `RunResumed` is a lead event like `RunStarted`: a port of `tests/engine/test_run_resumed_event.py`.
 *
 * Ported: `test_run_resumed_constructible_no_required_fields` and
 * `test_run_resumed_is_not_a_terminal`.
 *
 * Covered by a case: none.
 */

import { expect, test } from "bun:test";

import { RunFailed, RunResumed, RunStarted, RunSucceeded } from "../../src/events.ts";

test("test_run_resumed_constructible_no_required_fields", () => {
  expect(new RunResumed()).toBeDefined();
  expect(Object.getPrototypeOf(new RunResumed())).not.toBe(Object.getPrototypeOf(new RunStarted()));
});

test("test_run_resumed_is_not_a_terminal", () => {
  const resumed = new RunResumed();
  expect(resumed).not.toBeInstanceOf(RunSucceeded);
  expect(resumed).not.toBeInstanceOf(RunFailed);
});
