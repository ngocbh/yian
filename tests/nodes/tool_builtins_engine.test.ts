/**
 * Real TOOL `take`, `sleep`, `ask` and host-`pending()` nodes through the engine: each run ends
 * waiting, a send, a complete or a passed deadline is recorded, and a fresh engine over the log
 * (as `advance` builds one) wakes the node to its result. Also: a restart after a crash between a
 * host call's `pending` step and `node_waiting` reuses the call, and the `pending` step is capped by
 * the run's recorded event limit.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, copyGraph, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { MemoryRunLog, NodeStepRecord, NodeWaitingRecord, NodeWokenRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type PreparedTransition,
  prepareComplete,
  prepareRunResumed,
  prepareSend,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type AskRequest, ToolSet } from "../../src/tools/registry.ts";
import { eventBytes } from "../../src/typesys/json.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { type HandClock, handClock, inboxEngine, recordNames } from "../helpers/inbox_fixtures.ts";
import { FIXED_CLOCK, noGrowth, RUN_ID } from "../helpers/run_state_fixtures.ts";

const T0 = Date.UTC(2026, 9, 23);

/**
 * START (one `Inbox(str)` param per name in `inboxes`) -> `node` -> END, with `node`'s params and
 * wiring from `wiring` (input name -> source).
 */
function toolFlow(node: ToolNode, wiring: Record<string, unknown>, inboxes: readonly string[] = []): CompiledFlow {
  const str = Type.scalar(ValueKind.STRING);
  const params = inboxes.map((name) => new Parameter(name, { contract: new ValueContract(inboxType(str)) }));
  node.params = Object.keys(wiring).map((name) => new Parameter(name));
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params })],
      [node.id, node],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-node", START_ID, node.id), new Edge("node-end", node.id, END_ID)],
    { wiring: new Map([[node.id, new Map(Object.entries(wiring))]]) },
  );
}

/** A fresh engine over `run`'s records replayed into a new state, `resume` adding `run_resumed`. */
function reopened(
  flow: CompiledFlow,
  run: FlowEngine,
  options: { resume?: boolean; upTo?: number; tools?: ToolSet; scheduler?: HandClock } = {},
): FlowEngine {
  const { resume = false, upTo, tools, scheduler } = options;
  const records = run.log.records().slice(0, upTo);
  const state = replayRecords(new RunState(copyGraph(flow)), records, { growBuilder: noGrowth });
  const log = MemoryRunLog.fromRecords(records, { clock: FIXED_CLOCK });
  if (resume) commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  return new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0, tools, scheduler });
}

/** Append one prepared delivery to `run`'s log, as a host call does between runs. */
function record(run: FlowEngine, prepared: PreparedTransition | null): void {
  commitPrepared(run.log, run.state, prepared!);
}

/** The `pending` steps in `run`'s log. */
const pendingSteps = (run: FlowEngine) =>
  run.log
    .records()
    .map((entry) => entry.record)
    .filter((entry): entry is NodeStepRecord => entry instanceof NodeStepRecord && entry.step["kind"] === "pending");

test("a take waits on an empty inbox; a recorded send wakes it to the event", async () => {
  const flow = toolFlow(new ToolNode("heard", { toolId: "take" }), { from: "${input.m}", take: "all" }, ["m"]);
  const first = inboxEngine(flow);
  await first.run();
  expect(recordNames(first.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
  const waiting = first.log.records().at(-2)!.record as NodeWaitingRecord;
  expect(waiting.waits).toEqual([{ callId: `${RUN_ID}/heard`, tool: "take", inbox: "m", mode: "all", deadline: null }]);

  record(first, prepareSend(first.state, "m", "hi"));
  const next = reopened(flow, first);
  await next.run();

  expect(next.state.result).toBeInstanceOf(RunSucceeded);
  expect(next.state.pool.get("heard")).toEqual({ events: ["hi"], closed: false });
});

test("a sleep waits until its deadline, then a fresh engine past it wakes it to null", async () => {
  const flow = toolFlow(new ToolNode("nap", { toolId: "sleep" }), { for: "90s" });
  const clock = handClock(T0);
  const first = inboxEngine(flow, { scheduler: clock });
  await first.run();
  const waiting = first.log.records().at(-2)!.record as NodeWaitingRecord;
  expect(waiting.waits).toEqual([
    { callId: `${RUN_ID}/nap`, tool: "sleep", inbox: null, mode: null, deadline: "2026-10-23T00:01:30.000Z" },
  ]);

  clock.set(T0 + 91_000);
  const next = reopened(flow, first, { scheduler: clock });
  await next.run();

  expect(next.log.records().some((entry) => entry.record instanceof NodeWokenRecord)).toBe(true);
  expect(next.state.result).toBeInstanceOf(RunSucceeded);
  expect(next.state.pool.get("nap")).toBeNull();
});

/** A flow whose `lead` asks with a one-hour timeout; `output` is the node's declared type, if any. */
function askFlow(output: Type | null = null): CompiledFlow {
  const node = new ToolNode("lead", { toolId: "ask" });
  if (output !== null) node.outputContract = new ValueContract(output);
  return toolFlow(node, { prompt: "Ship?", timeout: "1h" });
}

/** A run's ask tools whose handler pushes each request and returns `pending()`. */
const pendingAsk = (seen: AskRequest[]) =>
  new ToolSet({ ask: (request: AskRequest) => (seen.push(request), pending()) }).forRun(RUN_ID);

test("an ask that pends records its deadline; a recorded complete gives the answer", async () => {
  const seen: AskRequest[] = [];
  const tools = pendingAsk(seen);
  const flow = askFlow();
  const first = inboxEngine(flow, { tools, scheduler: handClock(T0) });
  await first.run();
  expect(recordNames(first.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
  expect(pendingSteps(first).map((entry) => entry.step)).toEqual([
    {
      kind: "pending",
      call_id: `${RUN_ID}/lead`,
      tool: "ask",
      arguments: { prompt: "Ship?", timeout: "1h" },
      deadline: "2026-10-23T01:00:00.000Z",
    },
  ]);
  expect(seen.map((request) => [request.callId, request.deadline])).toEqual([
    [`${RUN_ID}/lead`, "2026-10-23T01:00:00.000Z"],
  ]);

  record(first, prepareComplete(first.state, `${RUN_ID}/lead`, { output: "yes" }));
  const next = reopened(flow, first, { tools, scheduler: handClock(T0) });
  await next.run();

  expect(next.state.result).toBeInstanceOf(RunSucceeded);
  expect(next.state.pool.get("lead")).toBe("yes");
  expect(seen).toHaveLength(1);
});

test("an ask nobody answers by its deadline gives null for Optional(T), else fails tool.timeout", async () => {
  const approval = Type.scalar(ValueKind.STRING);
  // No `output:` reads as `str`, which a timeout can't give.
  for (const [output, expected] of [
    [approval.replace({ nullable: true }), "succeeded"],
    [approval, "failed"],
    [null, "failed"],
  ] as const) {
    const seen: AskRequest[] = [];
    const tools = pendingAsk(seen);
    const flow = askFlow(output);
    const clock = handClock(T0);
    const first = inboxEngine(flow, { tools, scheduler: clock });
    await first.run();

    clock.set(T0 + 3_600_001);
    const next = reopened(flow, first, { tools, scheduler: clock });
    await next.run();

    expect(seen).toHaveLength(1);
    if (expected === "succeeded") {
      expect(next.state.result).toBeInstanceOf(RunSucceeded);
      expect(next.state.pool.get("lead")).toBeNull();
    } else {
      expect(next.state.result).toBeInstanceOf(RunFailed);
      const failure = (next.state.result as RunFailed).failure;
      expect(failure.code).toBe("tool.timeout");
      expect(failure.message).toBe(
        "TOOL node 'lead' (ask): nobody answered by 2026-10-23T01:00:00.000Z; declare output: Optional(...) to route on a timeout",
      );
    }
  }
});

/** A host tool `codex_send` whose handler counts its calls and returns `pending()`. */
function sendTools(calls: string[]): ToolSet {
  const send = new FunctionTool({
    name: "codex_send",
    description: "",
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } } },
    execute: (_args, ctx) => (calls.push(ctx.callId), pending()),
  });
  return new ToolSet({ codex_send: send }).forRun(RUN_ID);
}

test("a restart after a crash between the pending step and node_waiting doesn't call the handler again", async () => {
  const calls: string[] = [];
  const tools = sendTools(calls);
  const flow = toolFlow(new ToolNode("send", { toolId: "codex_send" }), { text: "hi" });
  const first = inboxEngine(flow, { tools });
  await first.run();
  const crashAt = first.log.records().findIndex((entry) => entry.record instanceof NodeWaitingRecord);

  const next = reopened(flow, first, { resume: true, upTo: crashAt, tools });
  await next.run();

  expect(calls).toEqual([`${RUN_ID}/send`]);
  expect(pendingSteps(next)).toHaveLength(1);
  const waiting = next.log.records().findLast((entry) => entry.record instanceof NodeWaitingRecord)!.record as NodeWaitingRecord;
  expect(waiting.callIds).toEqual([`${RUN_ID}/send`]);

  // The host answers; the run ends with it and the handler is still called once.
  record(next, prepareComplete(next.state, `${RUN_ID}/send`, { output: "sent" }));
  const done = reopened(flow, next, { tools });
  await done.run();
  expect(done.state.result).toBeInstanceOf(RunSucceeded);
  expect(done.state.pool.get("send")).toBe("sent");
  expect(calls).toHaveLength(1);
});

test("a pending step over the run's recorded event limit fails the node and records nothing", async () => {
  const step = { kind: "pending", call_id: `${RUN_ID}/send`, tool: "codex_send", arguments: { text: "hi" } };
  const size = eventBytes(step);
  for (const [limit, recorded] of [
    [size - 1, 0],
    [size, 1],
  ] as const) {
    const flow = toolFlow(new ToolNode("send", { toolId: "codex_send" }), { text: "hi" });
    const run = inboxEngine(flow, { tools: sendTools([]), defaults: { MAX_EVENT_BYTES: BigInt(limit) } });
    await run.run();

    expect(pendingSteps(run)).toHaveLength(recorded);
    if (recorded === 0) {
      expect(run.state.result).toBeInstanceOf(RunFailed);
      expect((run.state.result as RunFailed).failure.message).toBe(
        `tool 'codex_send' pending call step is ${size} bytes, over the run's limit of ${limit} (YIAN_MAX_EVENT_BYTES)`,
      );
    } else {
      expect(recordNames(run.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
    }
  }
});
