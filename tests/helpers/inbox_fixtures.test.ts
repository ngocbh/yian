/** `inboxEngine`: `engine()`'s run start, with each inbox bound to its handle. */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import type { RunStartedRecord } from "../../src/runtime/log.ts";
import { inboxEngine, inboxFlow } from "./inbox_fixtures.ts";

const FLOW = () => inboxFlow(["mail"], [], [[START_ID, END_ID]]);

test("an inbox engine binds each inbox to its handle and each input to the input", () => {
  const run = inboxEngine(FLOW(), { inputs: { topic: "x" } });
  const started = run.log.records()[0]!.record as RunStartedRecord;
  expect(started.startWiringOverride).toEqual({ topic: "${input.topic}", mail: { inbox: "mail" } });
});

test("an inbox engine records the memory configuration it is given, and none by default", () => {
  const configured = inboxEngine(FLOW(), { memories: [], memoryStacks: [], memorySnapshots: [] });
  expect(configured.state.memories).toEqual([]);
  expect(configured.state.memoryStacks).toEqual([]);
  expect(configured.state.memorySnapshots).toEqual([]);
  const plain = inboxEngine(FLOW());
  expect(plain.state.memories).toBeNull();
  expect(plain.state.memoryStacks).toBeNull();
  expect(plain.state.memorySnapshots).toBeNull();
});
