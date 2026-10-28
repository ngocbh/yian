/**
 * `RunResumed` is a lead event like `RunStarted`: it needs no fields and is not a terminal.
 */

import { expect, test } from "bun:test";

import { RunFailed, RunResumed, RunStarted, RunSucceeded } from "../src/events.ts";

test("run resumed constructible no required fields", () => {
  expect(new RunResumed()).toBeDefined();
  expect(Object.getPrototypeOf(new RunResumed())).not.toBe(Object.getPrototypeOf(new RunStarted()));
});

test("run resumed is not a terminal", () => {
  const resumed = new RunResumed();
  expect(resumed).not.toBeInstanceOf(RunSucceeded);
  expect(resumed).not.toBeInstanceOf(RunFailed);
});
