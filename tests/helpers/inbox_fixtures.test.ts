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

