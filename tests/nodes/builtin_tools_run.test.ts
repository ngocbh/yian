/**
 * YAML `take`, `sleep` and `ask` flows, loaded and run through the engine: a run ends waiting, a
 * send, a complete or a passed deadline is recorded, and a fresh engine over the log (as `advance`
 * builds one) wakes the node to its result.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { type CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { RunSucceeded } from "../../src/events.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { MemoryRunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  type PreparedTransition,
  prepareComplete,
  prepareInboxClosed,
  prepareSend,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { type AskRequest, ToolSet } from "../../src/tools/registry.ts";
import { type HandClock, handClock, inboxEngine, recordNames } from "../helpers/inbox_fixtures.ts";
import { FIXED_CLOCK, RUN_ID } from "../helpers/run_state_fixtures.ts";

const T0 = Date.UTC(2026, 9, 23);

/** The compiled flow of YAML `text`. */
const compiled = async (text: string): Promise<CompiledFlow> => (await loadFlow(text)).compiled;

/**
 * A fresh engine over `run`'s records replayed into a new state, as `advance` builds one. `flow` is
 * the pristine flow: each run grows its own copy.
 */
function reopened(flow: CompiledFlow, run: FlowEngine, options: { tools?: ToolSet; scheduler?: HandClock } = {}) {
  const records = run.log.records();
  const state = replayRecords(new RunState(copyGraph(flow)), records, { growBuilder: buildGrow });
  const log = MemoryRunLog.fromRecords(records, { clock: FIXED_CLOCK });
  return new FlowEngine(state, log, { growBuilder: buildGrow, numWorkers: 0, ...options });
}

/** Append one prepared delivery to `run`'s log, as a host call does between runs. */
function record(run: FlowEngine, prepared: PreparedTransition | null): void {
  commitPrepared(run.log, run.state, prepared!);
}

/** The flow output `run` succeeded with. */
const outputOf = (run: FlowEngine): unknown => {
  expect(run.state.result).toBeInstanceOf(RunSucceeded);
  return (run.state.result as RunSucceeded).output;
};

const CHAT = `id: chat
name: chat
input:
  owner_messages: !Inbox(str)
defs:
  turn:
    input:
      all_messages: !list(str)
      closed: !bool
      owner_messages: !Inbox(str)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.owner_messages}
          take: all
    output:
      all_messages: \${input.all_messages + heard.output.events}
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      all_messages: []
      closed: false
      owner_messages: \${input.owner_messages}
    while: "not closed"
output: \${chat.output.all_messages}
`;

test("a loop of takes waits each turn and ends when its inbox closes", async () => {
  const flow = await compiled(CHAT);
  const first = inboxEngine(copyGraph(flow), { growBuilder: buildGrow });
  await first.run();
  expect(recordNames(first.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);

  record(first, prepareSend(first.state, "owner_messages", "hi"));
  const second = reopened(flow, first);
  await second.run();
  expect(recordNames(second.log).slice(first.log.records().length)).toContain("NodeWokenRecord");
  expect(recordNames(second.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);

  record(second, prepareSend(second.state, "owner_messages", "bye"));
  record(second, prepareInboxClosed(second.state, "owner_messages"));
  const third = reopened(flow, second);
  await third.run();

  expect(recordNames(third.log).slice(second.log.records().length)).toContain("NodeWokenRecord");
  expect(outputOf(third)).toEqual(["hi", "bye"]);
});

/** A flow that sleeps with input `input` (one YAML line), then returns the sleep's output. */
const sleepFlow = (input: string) => `id: nap
name: nap
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      ${input}
output: \${nap.output}
`;

test("a sleep until a past time commits inline and the run succeeds with null", async () => {
  const run = inboxEngine(await compiled(sleepFlow('until: "2026-05-21T14:30:00Z"')), { scheduler: handClock(T0) });
  await run.run();
  expect(recordNames(run.log)).not.toContain("NodeWaitingRecord");
  expect(outputOf(run)).toBeNull();
});

test("a sleep for 90s waits, and a fresh engine past its deadline wakes it to null", async () => {
  const flow = await compiled(sleepFlow("for: 90s"));
  const clock = handClock(T0);
  const first = inboxEngine(copyGraph(flow), { scheduler: clock });
  await first.run();
  expect(recordNames(first.log).at(-1)).toBe("RunWaitingRecord");

  clock.set(T0 + 91_000);
  const next = reopened(flow, first, { scheduler: clock });
  await next.run();

  expect(recordNames(next.log)).toContain("NodeWokenRecord");
  expect(outputOf(next)).toBeNull();
});

const ESCALATION = `id: approve_plan
name: approve_plan
typedefs:
  Approval:
    decision: Literal(approve, reject)
    note: Optional(str)
nodes:
  lead:
    kind: tool
    tool_id: ask
    input:
      to: lead
      prompt: Approve this plan?
      timeout: 24h
    output: Optional(Approval)
  escalate:
    kind: case
    cases:
      - when: \${lead.output == null}
        then: manager
  manager:
    kind: tool
    tool_id: ask
    input:
      to: manager
      prompt: No answer from the lead in 24h. Approve this plan?
      timeout: 48h
    output: Optional(Approval)
output: !Optional(Approval) \${lead.output | manager.output}
`;

/** The run's ask tools: a handler declaring `to:` that pushes each request and returns `pending()`. */
const pendingAsk = (seen: AskRequest[]) =>
  new ToolSet({
    ask: { handler: (request: AskRequest) => (seen.push(request), pending()), inputs: { to: { type: "string" } } },
  }).forRun(RUN_ID);

test("the escalation asks the lead; a complete answers it and the manager is skipped", async () => {
  const seen: AskRequest[] = [];
  const tools = pendingAsk(seen);
  const flow = await compiled(ESCALATION);
  const first = inboxEngine(copyGraph(flow), { tools, scheduler: handClock(T0) });
  await first.run();
  expect(recordNames(first.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
  expect(seen.map((request) => [request.callId, request.extra, request.deadline])).toEqual([
    [`${RUN_ID}/lead`, { to: "lead" }, "2026-10-24T00:00:00.000Z"],
  ]);
  // The host answers the non-null type: null means nobody answered.
  expect(seen[0]!.answerSchema).toEqual([
    { name: "decision", type: "str", required: true, enum: ["approve", "reject"] },
    { name: "note", type: "str", required: false },
  ]);

  record(first, prepareComplete(first.state, `${RUN_ID}/lead`, { output: { decision: "approve", note: null } }));
  const next = reopened(flow, first, { tools, scheduler: handClock(T0) });
  await next.run();

  expect(recordNames(next.log).slice(first.log.records().length)).toContain("NodeWokenRecord");
  expect(outputOf(next)).toEqual({ decision: "approve", note: null });
  expect(seen).toHaveLength(1);
});

test("the escalation asks the manager once the lead's 24h pass", async () => {
  const seen: AskRequest[] = [];
  const tools = pendingAsk(seen);
  const flow = await compiled(ESCALATION);
  const clock = handClock(T0);
  const first = inboxEngine(copyGraph(flow), { tools, scheduler: clock });
  await first.run();
  expect(recordNames(first.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);

  clock.set(T0 + 24 * 3_600_000 + 1);
  const second = reopened(flow, first, { tools, scheduler: clock });
  await second.run();
  // The lead's timeout wakes it to null; the manager is asked and waits.
  expect(recordNames(second.log).slice(first.log.records().length)).toContain("NodeWokenRecord");
  expect(recordNames(second.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
  expect(seen.map((request) => request.extra)).toEqual([{ to: "lead" }, { to: "manager" }]);

  record(second, prepareComplete(second.state, `${RUN_ID}/manager`, { output: { decision: "reject", note: "no" } }));
  const third = reopened(flow, second, { tools, scheduler: clock });
  await third.run();

  expect(recordNames(third.log).slice(second.log.records().length)).toContain("NodeWokenRecord");
  expect(outputOf(third)).toEqual({ decision: "reject", note: "no" });
});

test("a null answer to an Optional ask is refused: null means nobody answered", async () => {
  const tools = pendingAsk([]);
  const flow = await compiled(ESCALATION);
  const first = inboxEngine(copyGraph(flow), { tools, scheduler: handClock(T0) });
  await first.run();

  let refused: unknown = null;
  try {
    prepareComplete(first.state, `${RUN_ID}/lead`, { output: null });
  } catch (error) {
    refused = error;
  }
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect((refused as DeliveryRefused).reason).toBe("invalid_value");
  expect((refused as DeliveryRefused).message).toContain("null means nobody answered in time");
});
