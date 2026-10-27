/**
 * Blocking nodes take exactly the configured worker capacity: a port of
 * `tests/engine/test_blocking_workers.py`.
 *
 * Ported (adapted), each with `BlockingNode` (a node awaiting a promise the test settles) in place
 * of HUMAN_INPUT and its `ask_user` tool, and promises in place of thread events and polling:
 * - `test_independent_node_completes_while_one_worker_waits_for_human`: the sibling's success is
 *   committed, and seen by `onEvent`, while the blocked node is still in flight.
 * - `test_ready_work_waits_while_every_worker_is_blocked`: once both workers block, the third node
 *   is still in the ready queue, not started.
 * - `test_failure_reaches_on_event_while_a_sibling_prompt_is_pending`: the run's failure aborts its
 *   signal, which stands in for the host withdrawing the prompt; `NodeFailed` and `RunFailed` reach
 *   `onEvent` while the sibling still blocks, and `run()` returns only after it stopped.
 *
 * The WAIT rows are `blocking_wait.test.ts`, the port of `test_blocking_wait.py`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeFailed, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
import { BlockingNode, engine, FuncNode, graph } from "./run_state_fixtures.ts";

test("independent node completes while one worker waits for a human", async () => {
  const release = Promise.withResolvers<string>();
  const ask = new BlockingNode("ask", release.promise);
  // The sibling runs only once `ask` occupies the other worker.
  const sibling = new BlockingNode("sibling", ask.entered.then(() => "done"));
  const flow = graph([ask, sibling], [
    [START_ID, "ask"],
    [START_ID, "sibling"],
    ["ask", END_ID],
    ["sibling", END_ID],
  ]);
  const committedWhileBlocked: unknown[] = [];
  const built = engine(flow, {
    numWorkers: 2,
    onEvent: (event) => {
      if (event instanceof NodeSucceeded && event.nodeId === "sibling") {
        committedWhileBlocked.push(built.state.pool.get("sibling"), ask.released, built.state.inFlight.has("ask"));
        release.resolve("answer");
      }
    },
  });

  const events = await built.run();

  expect(committedWhileBlocked).toEqual(["done", false, true]);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("ready work waits while every worker is blocked", async () => {
  const release = Promise.withResolvers<string>();
  const askOne = new BlockingNode("ask_one", release.promise);
  const askTwo = new BlockingNode("ask_two", release.promise);
  let readyRan = false;
  const ready = new FuncNode("ready", () => {
    readyRan = true;
    return "done";
  });
  const flow = graph([askOne, askTwo, ready], [
    [START_ID, "ask_one"],
    [START_ID, "ask_two"],
    [START_ID, "ready"],
    ["ask_one", END_ID],
    ["ask_two", END_ID],
    ["ready", END_ID],
  ]);
  const built = engine(flow, { numWorkers: 2 });

  const running = built.run();
  await Promise.all([askOne.entered, askTwo.entered]);
  expect(readyRan).toBe(false);
  expect(built.state.ready).toEqual(["ready"]);
  expect([...built.state.inFlight.keys()]).toEqual(["ask_one", "ask_two"]);
  release.resolve("answer");
  const events = await running;

  expect(readyRan).toBe(true);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("failure reaches onEvent while a sibling prompt is pending", async () => {
  // A human who never answers: only the run's abort ends the prompt.
  const ask = new BlockingNode("ask", new Promise(() => {}));
  const bad = new BlockingNode(
    "bad",
    ask.entered.then(() => {
      throw new Error("boom");
    }),
  );
  const flow = graph([ask, bad], [
    [START_ID, "ask"],
    [START_ID, "bad"],
    ["ask", END_ID],
    ["bad", END_ID],
  ]);
  const pendingWhenObserved: Record<string, boolean> = {};
  const built = engine(flow, {
    numWorkers: 2,
    onEvent: (event) => {
      if (event instanceof NodeFailed || event instanceof RunFailed) {
        pendingWhenObserved[event.constructor.name] = !ask.released;
      }
    },
  });

  const events = await built.run();

  expect(events.at(-1)).toBeInstanceOf(RunFailed);
  expect((events.at(-1) as RunFailed).failure.message).toBe("boom");
  // The host hears of the failure while the prompt is still open, so it can withdraw it.
  expect(pendingWhenObserved).toEqual({ NodeFailed: true, RunFailed: true });
  // `run()` returned only after the drain: the blocked sibling has stopped.
  expect(ask.released).toBe(true);
});
