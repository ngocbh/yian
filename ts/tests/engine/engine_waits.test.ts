import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { FakeHost, inboxEngine, inboxFlow, recordNames } from "./inbox_fixtures.ts";
import { engine, graph, RecordNode, rejected } from "./run_state_fixtures.ts";

test("a host Pending waits, and the quiescent run records run_waiting and resolves", async () => {
  const host = new FakeHost("ask");
  const run = inboxEngine(inboxFlow([], [host], [[START_ID, "ask"], ["ask", END_ID]]), { defaults: { MAX_EVENT_BYTES: 4096n } });
  const events = await run.run();
  expect(recordNames(run.log).slice(-3)).toEqual(["NodeStepRecord", "NodeWaitingRecord", "RunWaitingRecord"]);
  expect(run.state.result).toBeNull();
  expect([...run.state.waits.waiting.keys()]).toEqual(["ask"]);
  expect(events.slice(-2).map((event) => event.constructor.name)).toEqual(["NodeWaiting", "RunWaiting"]);
  // The run's recorded limit, not the process's.
  expect(host.limits).toEqual([4096]);
});

test.each([0, 2])("with nothing waiting, nonterminal quiescence still throws (%p workers)", async (numWorkers) => {
  const built = engine(graph([new RecordNode("a", [])], [[START_ID, "a"], ["a", END_ID]]), { numWorkers });
  built.state.ready.length = 0;
  expect(await rejected(built.run())).toEqual(new Error("FlowEngine reached nonterminal quiescence"));
});
