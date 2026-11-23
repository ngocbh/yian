/**
 * The run reducer's rules for the records of a host's check before each agent tool call: a gated
 * run's start and resumes, a refused call's result, a note, a held call's `pending` step and its
 * `released` step; and a held or released call's run state, as the waits, `pendingCalls` and
 * events show it.
 */

import { describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { CallClosed, CallHeld, RunResumed } from "../../src/events.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { SKIPPED_TOOL_RESULT, TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/results.ts";
import { pendingCalls } from "../../src/run.ts";
import { prepareDelivery } from "../../src/runtime/engine.ts";
import { MemoryRunLog, NodeStepRecord, RunResumedRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  callResultStep,
  callView,
  commitPrepared,
  DeliveryRefused,
  prepareComplete,
  preparePoolCommit,
  prepareInterrupt,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareNodeWoken,
  prepareRecord,
  prepareRelease,
  prepareRunCancelled,
  prepareRunResumed,
  prepareRunStarted,
  type PreparedTransition,
  RunState,
} from "../../src/runtime/state.ts";
import { waitInfos, WaitsState, wakeableNodes } from "../../src/runtime/waits.ts";
import { GATE_TEXT_MAX_BYTES } from "../../src/tools/gate.ts";
import { FIXED_CLOCK, RUN_ID, startData } from "../helpers/run_state_fixtures.ts";
import { askPending, call, MSGS, paramFlow, replayed, started, takeEntry, tool, waitFlow } from "../helpers/waits_fixtures.ts";

const agent = (id: string) => new AgentNode(id, { prompt: "go", tools: ["ask_user", "lookup"] });

/** A recorded chat reply calling `name` `calls` times with `args`, as `c0`, `c1`, ... */
const reply = (calls: number, name = "lookup", args: Record<string, unknown> = {}) => ({
  kind: "model_call",
  purpose: "chat",
  model: "m",
  request_sha256: "0".repeat(64),
  started_at: "2026-10-20T14:02:11.123Z",
  first_token_at: null,
  output: {
    format: "yian.message.v1",
    role: "assistant",
    parts: Array.from({ length: calls }, (_, i) => ({ type: "tool_call", id: `c${i}`, name, arguments: args })),
  },
  response_id: null,
  finish: "tool_calls",
  usage: null,
});

/** The yian call id of call `index` of agent `a`'s reply 1. */
const callOf = (index: number) => `${RUN_ID}/a/1/${index}`;

/** Agent `a`'s `pending` step for call `index` of its reply 1, provider id `c<index>`. */
const pendingOf = (index: number, tool = "lookup") => ({
  kind: "pending",
  call_id: callOf(index),
  tool,
  tool_call_id: `c${index}`,
  reply: 1n,
  index: BigInt(index),
});

/** The hold of call `index`, whose group ends at call `groupEnd`. */
const holdOf = (index: number, groupEnd = index) => ({
  ...pendingOf(index),
  held: true,
  source: { kind: "host", group: null, tool: "lookup" },
  group_end: BigInt(groupEnd),
});

/** An agent's `tool_result` step for provider call `toolCallId`. */
const result = (toolCallId: string, extra: Record<string, unknown> = {}) => ({
  kind: "tool_result",
  call_id: toolCallId,
  started_at: "2026-10-20T14:02:11.123Z",
  output: "found",
  is_error: false,
  ...extra,
});

/** A refused call's result, as the gate writes it. */
const refusal = (toolCallId: string) =>
  result(toolCallId, { started_at: null, output: "Not allowed.", is_error: true, closed: "refused" });

/** Commit `body` as `node`'s next step. */
function step(log: RunLog, state: RunState, node: string, body: Record<string, unknown>): void {
  commitPrepared(log, state, prepareNodeStep(state, node, body));
}

/** A flow whose agent `a` grants the def `shout`. */
const SHOUT_FLOW = `
id: shouting
name: shouting
llm_config: {provider: openai, model: test-model}
defs:
  shout:
    description: Shout the text back.
    input:
      text: !str
    nodes:
      loud:
        kind: code
        input:
          text: \${input.text}
        output: !str
        code: return inputs.text.toUpperCase();
    output: \${loud.output}
nodes:
  a:
    kind: agent
    prompt: go
    tools: [shout]
output: \${a.output}
`;

/**
 * A run whose agent `a` is running after a reply calling `lookup` three times (c0, c1, c2); with
 * `steers`, `a` reads the steer inbox `msgs`, so an interrupt of it names `a`.
 */
function replied(options: { steers?: boolean } = {}) {
  const steers = options.steers === true;
  const a = new AgentNode("a", { prompt: "go", tools: ["ask_user", "lookup"], steerInbox: steers });
  const flow = steers
    ? paramFlow([MSGS], { wiring: new Map([["a", new Map([[STEER_INBOX_KEY, new Binding("${input.msgs}")]])]]) }, a, tool("t", "ask"))
    : paramFlow([], a, tool("t", "ask"));
  const run = started(flow);
  commitPrepared(run.log, run.state, prepareNodeStarted(run.state, "a"));
  step(run.log, run.state, "a", reply(3));
  return run;
}

describe("a gated run", () => {
  test("its start sets the run's gated state, and a replay rebuilds it", () => {
    const { state, log, path } = started(paramFlow([], agent("a")), { gated: true });
    log.close();
    expect([state.gated, replayed(state.flow, path).gated]).toEqual([true, true]);
    expect(started(paramFlow([], agent("a"))).state.gated).toBe(false);
  });

  test("every resume of a gated run says it is gated", () => {
    const { state, log } = started(paramFlow([], agent("a")), { gated: true });
    expect(() => prepareRecord(state, new RunResumedRecord([], 0))).toThrow(
      "a resume of a gated run must record gated: true",
    );
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0, gated: true }));
    expect(state.gated).toBe(true);
  });

  test("a gated resume makes an ungated run gated for good", () => {
    const { state, log, path } = started(paramFlow([], agent("a")));
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
    expect(state.gated).toBe(false);
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0, gated: true }));
    expect(state.gated).toBe(true);
    expect(() => prepareRunResumed(state, { numWorkers: 0 })).toThrow("a resume of a gated run must record gated: true");
    log.close();
    expect(replayed(state.flow, path).gated).toBe(true);
  });
});

describe("a refused call's result", () => {
  test("is a call no pending step registered, never started, an error the model reads", () => {
    const { log, state } = replied();
    step(log, state, "a", refusal("c0"));
    expect(state.nodeSteps.get("a")!.at(-1)!["closed"]).toBe("refused");
  });

  const SHAPE = "a refused tool_result has started_at null, is_error true and its text as output";
  test.each<[string, Record<string, unknown>, string]>([
    ["a start time", { started_at: "2026-10-20T14:02:11.123Z" }, SHAPE],
    ["no error", { is_error: false }, SHAPE],
    ["a non-text output", { output: 7n }, "a refused tool_result's output must be a string"],
    ["an empty output", { output: "" }, "a refused tool_result's output must be a non-empty string"],
    ["an oversized output", { output: "a".repeat(GATE_TEXT_MAX_BYTES + 1) }, "a refused tool_result's output is 4097 UTF-8 bytes"],
    ["a note", { note: "ok" }, "a refused tool_result holds no note"],
  ])("refuses %s", (_name, change, message) => {
    const { state } = replied();
    expect(() => prepareNodeStep(state, "a", { ...refusal("c0"), ...change })).toThrow(message);
  });

  test("refuses a call a pending step registered", () => {
    const { log, state } = replied();
    step(log, state, "a", pendingOf(0));
    expect(() => prepareNodeStep(state, "a", refusal("c0"))).toThrow(
      `a refused tool_result for call '${callOf(0)}', which a pending step registered; only a call that never pended is refused`,
    );
  });

  test("refuses a TOOL node's result", () => {
    const { log, state } = replied();
    commitPrepared(log, state, prepareNodeStarted(state, "t"));
    expect(() => prepareNodeStep(state, "t", refusal("c0"))).toThrow(
      "a refused tool_result on node 't', which isn't an agent",
    );
  });
});

describe("a note", () => {
  test("rides an allowed call's result or its pending step", () => {
    const { log, state } = replied();
    step(log, state, "a", result("c0", { note: "rule 4 allows reads" }));
    step(log, state, "a", { ...pendingOf(1), note: "a person reviews" });
    expect(state.waits.calls.get(callOf(1))!.state).toBe("open");
  });

  test("rides a flow call's pending step", async () => {
    const flow = await loadFlow(SHOUT_FLOW);
    const { log, state } = started(flow.compiled);
    commitPrepared(log, state, prepareNodeStarted(state, "a"));
    step(log, state, "a", reply(1, "shout", { text: "hi" }));
    const call = { ...pendingOf(0, "shout"), flow: true, arguments: { text: "hi" }, scope: "a/shout#0" };
    step(log, state, "a", { ...call, note: "a flow the rules allow" });
    expect(state.waits.calls.get(callOf(0))!.flow).toBe(true);
  });

  test.each(["ask_user", "load_skill"])("refuses a note on a %s call's pending step, which is never checked", (tool) => {
    const { state } = replied();
    expect(() => prepareNodeStep(state, "a", { ...pendingOf(0, tool), note: "ok" })).toThrow(
      `a ${tool} call's pending step holds no note; its calls are never checked`,
    );
  });

  test.each<[string, unknown, string]>([
    ["an empty note", "", "a tool_result's note must be a non-empty string"],
    ["a non-text note", 7n, "a tool_result's note must be a string"],
    ["an oversized note", "a".repeat(GATE_TEXT_MAX_BYTES + 1), "a tool_result's note is 4097 UTF-8 bytes, over the limit of 4096"],
  ])("refuses %s on a result", (_name, note, message) => {
    const { state } = replied();
    expect(() => prepareNodeStep(state, "a", result("c0", { note }))).toThrow(message);
  });

  test("refuses a note on a closer or a host's answer", () => {
    const { log, state } = replied();
    expect(() => prepareNodeStep(state, "a", result("c0", { note: "ok", closed: "skipped" }))).toThrow(
      "a tool_result with a note is no closer",
    );
    step(log, state, "a", pendingOf(0));
    expect(() => prepareNodeStep(state, "a", result("c0", { note: "ok" }))).toThrow(
      `a tool_result for call '${callOf(0)}', which a pending step registered, holds no note; its pending step does`,
    );
  });

  test("refuses an oversized note on a pending step, and any note on a TOOL node's", () => {
    const { log, state } = replied();
    const long = "a".repeat(GATE_TEXT_MAX_BYTES + 1);
    expect(() => prepareNodeStep(state, "a", { ...pendingOf(0), note: long })).toThrow(
      "a pending step's note is 4097 UTF-8 bytes, over the limit of 4096",
    );
    commitPrepared(log, state, prepareNodeStarted(state, "t"));
    for (const field of ["note", "held"]) {
      expect(() => prepareNodeStep(state, "t", { ...askPending("t"), [field]: true })).toThrow(
        `a TOOL node's pending step holds no ${field}`,
      );
    }
  });
});

describe("a hold's pending step", () => {
  test("registers its call, with exactly the hold's fields", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0, 1));
    expect(state.waits.calls.get(callOf(0))!.state).toBe("open");
  });

  const EXACT =
    "a hold's pending step holds exactly kind, call_id, tool, tool_call_id, reply, index, held, source and group_end, and may hold target";
  const ONLY_HOLD = "a pending step holds source, group_end and target only as a hold";
  const NEVER_HELD = (tool: string) => `a hold's tool can't be ${tool}, whose calls are never checked`;
  const GROUP_END = "a hold's group_end must be an int from its index 1 to its reply's last call 2";
  test.each<[string, (step: Record<string, unknown>) => void, string]>([
    ["a note", (s) => (s["note"] = "x"), EXACT],
    ["a deadline", (s) => (s["deadline"] = "2026-10-21T00:00:00.000Z"), EXACT],
    ["flow fields", (s) => Object.assign(s, { flow: true, arguments: {}, scope: "a/f#0" }), EXACT],
    ["no source", (s) => delete s["source"], EXACT],
    ["no held", (s) => delete s["held"], ONLY_HOLD],
    ["a target on a call that isn't held", (s) => {
      for (const key of ["held", "source", "group_end"]) delete s[key];
      s["target"] = {};
    }, ONLY_HOLD],
    ["a target that isn't an object", (s) => (s["target"] = ["x"]), "a hold's target must be a JSON object"],
    ["held false", (s) => (s["held"] = false), "a hold's pending step says held: true"],
    ["an ask_user", (s) => (s["tool"] = "ask_user"), NEVER_HELD("ask_user")],
    ["a load_skill", (s) => (s["tool"] = "load_skill"), NEVER_HELD("load_skill")],
    ["a source of no kind", (s) => (s["source"] = { kind: "skill", tool: "lookup" }), "a hold's source must be a tool source"],
    ["a group ending before it", (s) => (s["group_end"] = 0n), GROUP_END],
    ["a float group end", (s) => (s["group_end"] = 1.5), GROUP_END],
    ["a text group end", (s) => (s["group_end"] = "2"), GROUP_END],
    ["a group past the reply", (s) => (s["group_end"] = 3n), GROUP_END],
  ])("refuses %s", (_name, change, message) => {
    const { state } = replied();
    const hold: Record<string, unknown> = holdOf(1);
    change(hold);
    expect(() => prepareNodeStep(state, "a", hold)).toThrow(message);
  });

  test("refuses an index past its reply's calls", () => {
    const { state } = replied();
    expect(() => prepareNodeStep(state, "a", holdOf(3))).toThrow("a hold's index 3 is past its reply's last call 2");
  });

  test("is checked as any agent pending step: its reply and call id", () => {
    const { state } = replied();
    expect(() => prepareNodeStep(state, "a", { ...holdOf(0), reply: 2n })).toThrow("an agent's pending reply must be");
    expect(() => prepareNodeStep(state, "a", { ...holdOf(0), call_id: callOf(1) })).toThrow(
      "isn't call 0 of node 'a''s reply 1",
    );
  });
});

describe("a released step", () => {
  test("names its node's open hold, on a running or a waiting node, with an optional note", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", holdOf(1));
    step(log, state, "a", { kind: "released", call_id: callOf(0), note: "approved by Ada" });
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    step(log, state, "a", { kind: "released", call_id: callOf(1) });
    expect(state.nodeSteps.get("a")!.at(-1)).toEqual({ kind: "released", call_id: callOf(1) });
  });

  const notHold = (index: number) => `a released step names call '${callOf(index)}', which isn't a hold of node 'a'`;
  test.each<[string, Record<string, unknown>, string]>([
    ["an extra field", { kind: "released", call_id: callOf(0), by: 3n }, "a released step holds only kind, call_id and note"],
    ["no call", { kind: "released" }, "a released step holds only kind, call_id and note"],
    ["an unknown call", { kind: "released", call_id: callOf(2) }, notHold(2)],
    ["a call that pended", { kind: "released", call_id: callOf(1) }, notHold(1)],
    ["an empty note", { kind: "released", call_id: callOf(0), note: "" }, "a released step's note must be a non-empty string"],
    // The step's record refuses it before the reducer does.
    ["a lone surrogate", { kind: "released", call_id: callOf(0), note: "ok\uD800" }, "lone surrogate"],
    [
      "an oversized note",
      { kind: "released", call_id: callOf(0), note: "é".repeat(GATE_TEXT_MAX_BYTES / 2 + 1) },
      "a released step's note is 4098 UTF-8 bytes, over the limit of 4096",
    ],
  ])("refuses %s", (_name, body, message) => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    expect(() => prepareNodeStep(state, "a", body)).toThrow(message);
  });

  test("refuses a hold already answered or released, and a TOOL node's", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", holdOf(1));
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    expect(() => prepareNodeStep(state, "a", { kind: "released", call_id: callOf(0) })).toThrow(
      `a released step names call '${callOf(0)}', which is already answered`,
    );
    step(log, state, "a", { kind: "released", call_id: callOf(1) });
    expect(() => prepareNodeStep(state, "a", { kind: "released", call_id: callOf(1) })).toThrow(
      `a released step names call '${callOf(1)}', which was released already`,
    );
    commitPrepared(log, state, prepareNodeStarted(state, "t"));
    expect(() => prepareNodeStep(state, "t", { kind: "released", call_id: callOf(1) })).toThrow(
      "a released step on node 't', which isn't an agent",
    );
  });
});

describe("a held call's run state", () => {
  /** The observers committing `body` as `a`'s next step publishes. */
  const published = (log: RunLog, state: RunState, body: Record<string, unknown>) =>
    commitPrepared(log, state, prepareNodeStep(state, "a", body));

  test("its pending step registers it held and publishes CallHeld; a host's pending() call is neither", () => {
    const { log, state } = replied();
    expect(published(log, state, holdOf(0, 1))).toEqual([new CallHeld({ nodeId: "a", callId: callOf(0), tool: "lookup" })]);
    expect(published(log, state, pendingOf(1))).toEqual([]);
    expect([state.waits.calls.get(callOf(0))!.held, state.waits.calls.get(callOf(1))!.held]).toEqual([true, false]);
    const back = WaitsState.fromJSON(JSON.parse(JSON.stringify(state.waits.toJSON())));
    expect(back.calls.get(callOf(0))!.held).toBe(true);
  });

  test("a hold registered on a node an interrupt names is held, then closed", () => {
    const { log, state } = replied({ steers: true });
    commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
    expect(published(log, state, holdOf(0))).toEqual([
      new CallHeld({ nodeId: "a", callId: callOf(0), tool: "lookup" }),
      new CallClosed({ callId: callOf(0), reason: "interrupted" }),
    ]);
  });

  test("a resume keeps it open, publishes no CallHeld again, and closes only the reply's other open calls crashed", () => {
    const holds = replied();
    step(holds.log, holds.state, "a", holdOf(0));
    step(holds.log, holds.state, "a", holdOf(1));
    expect((prepareRunResumed(holds.state, { numWorkers: 0 }).record as RunResumedRecord).closedCalls).toEqual([]);

    const { log, state, path } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    expect(commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }))).toEqual([
      new RunResumed(),
      new CallClosed({ callId: callOf(1), reason: "crashed" }),
    ]);
    expect([state.waits.calls.get(callOf(0))!.state, state.waits.calls.get(callOf(1))!.closedReason]).toEqual(["open", "crashed"]);
    log.close();
    expect(replayed(state.flow, path).waits.calls.get(callOf(0))!.state).toBe("open");
  });

  test("complete answers it, and wakes its waiting node while the node's other call is open", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    expect(wakeableNodes(state.waits)).toEqual([]);
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    expect(state.waits.calls.get(callOf(0))!.state).toBe("answered");
    expect(wakeableNodes(state.waits)).toEqual(["a"]);
    commitPrepared(log, state, prepareNodeWoken(state, "a"));
    expect(state.waits.calls.get(callOf(1))!.state).toBe("open");
  });

  test("a wait may name a hold answered in flight, and its node wakes at once", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    expect(wakeableNodes(state.waits)).toEqual(["a"]);
  });

  test("an answer to a call that isn't held doesn't wake the node while another is open", () => {
    const { log, state } = replied();
    step(log, state, "a", pendingOf(0));
    step(log, state, "a", holdOf(1));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    expect(wakeableNodes(state.waits)).toEqual([]);
    expect(() => prepareNodeWoken(state, "a")).toThrow(`NodeWoken node 'a' still waits on open call '${callOf(1)}'`);
  });

  test("an interrupt closes it, and a cancel does", () => {
    const interrupted = replied({ steers: true });
    step(interrupted.log, interrupted.state, "a", holdOf(0));
    commitPrepared(interrupted.log, interrupted.state, prepareNodeWaiting(interrupted.state, "a", { callIds: [callOf(0)], steerInbox: "msgs" }));
    const interrupt = prepareInterrupt(interrupted.state, "msgs")!;
    expect(commitPrepared(interrupted.log, interrupted.state, interrupt).at(-1)).toEqual(
      new CallClosed({ callId: callOf(0), reason: "interrupted" }),
    );

    const cancelled = replied();
    step(cancelled.log, cancelled.state, "a", holdOf(0));
    expect(commitPrepared(cancelled.log, cancelled.state, prepareRunCancelled(cancelled.state)).at(-1)).toEqual(
      new CallClosed({ callId: callOf(0), reason: "cancelled" }),
    );
  });

  test("pendingCalls and the waits show it held, with its recorded source and target", () => {
    const { log, state, path } = replied();
    const target = { run: "r1", call: { tool: "push", n: 2n, at: 1.5 } };
    step(log, state, "a", { ...holdOf(0), target });
    step(log, state, "a", pendingOf(1));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    log.close();
    const listed = pendingCalls(path).map(({ callId, held, source, target, arguments: args }) => ({ callId, held, source, target, args }));
    expect(listed).toEqual([
      { callId: callOf(0), held: true, source: { kind: "host", group: null, tool: "lookup" }, target, args: {} },
      { callId: callOf(1), held: false, source: null, target: null, args: {} },
    ]);
    expect(Object.isFrozen(listed[0]!.target!["call"])).toBe(true);
    expect(waitInfos(state.waits).map(({ callId, held }) => [callId, held])).toEqual([
      [callOf(0), true],
      [callOf(1), false],
    ]);
  });
});

describe("a released call", () => {
  /** Release call `index` of agent `a`; return the `released` step's seq. */
  const release = (log: RunLog, state: RunState, index: number, note: string | null = null) => {
    const prepared = prepareRelease(state, callOf(index), note)!;
    commitPrepared(log, state, prepared);
    return prepared.expectedSequence;
  };
  /**
   * Agent `a` waiting on its hold of call 0, which the host then released, woken: the evaluation
   * that runs it.
   */
  const wokenToRelease = (options: { steers?: boolean } = {}) => {
    const run = replied(options);
    step(run.log, run.state, "a", holdOf(0));
    commitPrepared(run.log, run.state, prepareNodeWaiting(run.state, "a", { callIds: [callOf(0)], ...(options.steers ? { steerInbox: "msgs" } : {}) }));
    const released = release(run.log, run.state, 0, "first");
    commitPrepared(run.log, run.state, prepareNodeWoken(run.state, "a"));
    return { ...run, released };
  };
  /** The `DeliveryRefused` `run` throws. */
  const refusedBy = (run: () => unknown): DeliveryRefused => {
    try {
      run();
    } catch (error) {
      if (error instanceof DeliveryRefused) return error;
      throw error;
    }
    throw new Error("expected a refusal");
  };

  test("its release is a released step publishing nothing, and wakes its waiting node while another call is open", () => {
    const { log, state, path } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    const prepared = prepareRelease(state, callOf(0), "approved by Ada")!;
    expect((prepareDelivery(state, { kind: "release", callId: callOf(0), note: "approved by Ada" }) as PreparedTransition).record).toEqual(
      prepared.record,
    );
    expect([prepared.record, prepared.observers]).toEqual([
      new NodeStepRecord("a", 4, { kind: "released", call_id: callOf(0), note: "approved by Ada" }),
      [],
    ]);
    commitPrepared(log, state, prepared);
    const released = state.waits.calls.get(callOf(0))!;
    expect(released).toMatchObject({ state: "released", releasedSeq: prepared.expectedSequence, held: true, resultSeq: null });
    expect(callView(state, released)).toMatchObject({ state: "released", held: true, result: null });
    expect(wakeableNodes(state.waits)).toEqual(["a"]);
    expect(waitInfos(state.waits).map(({ callId, settled, held }) => [callId, settled, held])).toEqual([
      [callOf(0), true, true],
      [callOf(1), false, false],
    ]);
    expect(WaitsState.fromJSON(JSON.parse(JSON.stringify(state.waits.toJSON()))).calls.get(callOf(0))).toEqual(released);
    log.close();
    // It is no longer the host's to answer.
    expect(pendingCalls(path).map((each) => each.callId)).toEqual([callOf(1)]);
  });

  test("a second release is a no-op whatever the call's state, and complete is refused once it is released", () => {
    const { log, state } = wokenToRelease();
    expect(prepareRelease(state, callOf(0), "second")).toBeNull();
    const refused = refusedBy(() => prepareComplete(state, callOf(0), { output: "done" }));
    expect([refused.reason, refused.callState, refused.message]).toEqual(["call_settled", "released", `call '${callOf(0)}' is already released`]);
    step(log, state, "a", result("c0"));
    expect(prepareRelease(state, callOf(0))).toBeNull();
    expect(state.nodeSteps.get("a")!.filter((each) => each["kind"] === "released")).toEqual([
      { kind: "released", call_id: callOf(0), note: "first" },
    ]);
  });

  test("prepareRelease refuses, in order, a run that ended or hasn't started, an unknown call, a take or flow call, a call that wasn't held, and a settled hold", async () => {
    const ended = replied();
    step(ended.log, ended.state, "a", holdOf(0));
    commitPrepared(ended.log, ended.state, prepareRunCancelled(ended.state));
    expect(refusedBy(() => prepareRelease(ended.state, callOf(0))).reason).toBe("ended");

    const fresh = new RunState(paramFlow([], agent("a")));
    commitPrepared(new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK }), fresh, prepareRunStarted(fresh, startData()));
    expect(refusedBy(() => prepareRelease(fresh, callOf(0))).reason).toBe("not_started");

    const take = started(waitFlow(tool("heard", "take")));
    commitPrepared(take.log, take.state, prepareNodeStarted(take.state, "heard"));
    commitPrepared(take.log, take.state, prepareNodeWaiting(take.state, "heard", { waits: [takeEntry("heard")], callIds: [call("heard")] }));
    expect(refusedBy(() => prepareRelease(take.state, call("heard"))).reason).toBe("not_host_call");

    const flow = started((await loadFlow(SHOUT_FLOW)).compiled);
    commitPrepared(flow.log, flow.state, prepareNodeStarted(flow.state, "a"));
    step(flow.log, flow.state, "a", reply(1, "shout", { text: "hi" }));
    step(flow.log, flow.state, "a", { ...pendingOf(0, "shout"), flow: true, arguments: { text: "hi" }, scope: "a/shout#0" });
    expect(refusedBy(() => prepareRelease(flow.state, callOf(0))).reason).toBe("not_host_call");

    const { log, state } = replied({ steers: true });
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    step(log, state, "a", holdOf(2));
    expect(refusedBy(() => prepareRelease(state, "nobody")).reason).toBe("unknown_call");
    expect(refusedBy(() => prepareRelease(state, callOf(1))).reason).toBe("not_held");
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    expect(refusedBy(() => prepareRelease(state, callOf(0)))).toMatchObject({ reason: "call_settled", callState: "answered" });
    commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
    expect(refusedBy(() => prepareRelease(state, callOf(2)))).toMatchObject({ reason: "call_settled", callState: "closed" });
  });

  test("prepareRelease refuses a call id or note of the wrong type, and a note checkGateText refuses, before the run's refusals", () => {
    const { state } = replied();
    expect(() => prepareRelease(state, 7 as never)).toThrow(new TypeError("call_id must be a string"));
    expect(() => prepareRelease(state, "nobody", 7 as never)).toThrow(new TypeError("note must be a string or null"));
    expect(() => prepareRelease(state, "nobody", "")).toThrow(new RangeError("note must be a non-empty string"));
    expect(() => prepareRelease(state, "nobody", "a".repeat(GATE_TEXT_MAX_BYTES + 1))).toThrow("note is 4097 UTF-8 bytes, over the limit of 4096");
  });

  test("an interrupt closes a waiting node's released call, which its agent closes as skipped once woken", () => {
    const { log, state } = replied({ steers: true });
    step(log, state, "a", holdOf(0));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0)], steerInbox: "msgs" }));
    const released = release(log, state, 0);
    const events = commitPrepared(log, state, prepareInterrupt(state, "msgs")!);
    expect(events.filter((event) => event instanceof CallClosed)).toEqual([new CallClosed({ callId: callOf(0), reason: "interrupted" })]);
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ state: "closed", closedReason: "interrupted", releasedSeq: released });
    expect(prepareRelease(state, callOf(0))).toBeNull();
    commitPrepared(log, state, prepareNodeWoken(state, "a"));
    step(log, state, "a", result("c0", { started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "interrupted" }));
    expect(state.waits.calls.get(callOf(0))!.resultSeq).not.toBeNull();
  });

  test("an interrupt leaves a running node's released call to its agent, which skips it", () => {
    const { log, state } = replied({ steers: true });
    step(log, state, "a", holdOf(0));
    release(log, state, 0);
    const events = commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
    expect(events.filter((event) => event instanceof CallClosed)).toEqual([]);
    expect(state.waits.calls.get(callOf(0))!.state).toBe("released");
    step(log, state, "a", result("c0", { started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" }));
    expect(state.waits.calls.get(callOf(0))!.state).toBe("answered");
  });

  test.each<[string, Record<string, unknown>]>([
    ["its run's result", {}],
    ["skipped, as an interrupt skips it before it starts", { started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" }],
    ["crashed, as a cut that may have started it closes it", { started_at: null, output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" }],
  ])("its running agent settles it once, with %s", (_name, extra) => {
    const { log, state, released } = wokenToRelease();
    step(log, state, "a", result("c0", extra));
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ state: "answered", releasedSeq: released });
    expect(callResultStep(state, callOf(0))).toEqual(result("c0", extra));
    expect(() => prepareNodeStep(state, "a", result("c0", extra))).toThrow(`tool_result for call '${callOf(0)}', which is already answered`);
  });

  test.each(["interrupted", "timed_out", "failed"])("refuses a %s closer on it: only a record closes a call so", (word) => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    release(log, state, 0);
    expect(() => prepareNodeStep(state, "a", result("c0", { started_at: null, is_error: true, closed: word }))).toThrow(
      `tool_result for released call '${callOf(0)}' closes it only skipped or crashed, not ${word}`,
    );
  });

  test.each<[string, Record<string, unknown>]>([
    ["its run's result", result("c0")],
    ["a crashed closer", result("c0", { started_at: null, output: TOOL_OUTCOME_UNKNOWN, is_error: true, closed: "crashed" })],
    ["a skip with no interrupt pending", result("c0", { started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" })],
    ["a pending step registering it again", pendingOf(0)],
  ])("refuses %s from the evaluation that began before the release: it pends the call again", (_name, body) => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    release(log, state, 0);
    expect(() => prepareNodeStep(state, "a", body)).toThrow("released after its node's latest start or wake");
  });

  test.each<[string, (log: RunLog, state: RunState) => unknown, string]>([
    ["a model call", (log, state) => step(log, state, "a", reply(1)), "a model_call step on node 'a', whose released call"],
    ["its output", (log, state) => commitPrepared(log, state, preparePoolCommit(state, "a", "done")), "an output commit on node 'a', whose released call"],
    [
      "an honor",
      (log, state) => step(log, state, "a", { kind: "interrupted", by: BigInt(state.waits.interrupts.get("a")!) }),
      "an interrupted step on node 'a', whose released call",
    ],
  ])("refuses %s while the agent holds a released call with no result", (_name, move, message) => {
    const { log, state } = wokenToRelease({ steers: true });
    commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
    expect(() => move(log, state)).toThrow(message);
    step(log, state, "a", result("c0", { started_at: null, output: SKIPPED_TOOL_RESULT, is_error: true, closed: "skipped" }));
    move(log, state);
  });

  test("refuses a refusal on it, and any result on its waiting node", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    release(log, state, 0);
    expect(() => prepareNodeStep(state, "a", refusal("c0"))).toThrow("only a call that never pended is refused");
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0)] }));
    expect(() => prepareNodeStep(state, "a", result("c0"))).toThrow(`tool_result for call '${callOf(0)}', which is already released`);
  });

  test("its tool's pending() registers it again, not held, its release kept, as its own tool and call", () => {
    const { log, state, released } = wokenToRelease();
    expect(() => prepareNodeStep(state, "a", holdOf(0))).toThrow(`call '${callOf(0)}' is already registered`);
    const again = `released call '${callOf(0)}' registers again only as tool 'lookup', call 'c0', with no note`;
    expect(() => prepareNodeStep(state, "a", pendingOf(0, "ask_user"))).toThrow(again);
    expect(() => prepareNodeStep(state, "a", { ...pendingOf(0), note: "ok" })).toThrow(again);
    step(log, state, "a", pendingOf(0));
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ state: "open", held: false, releasedSeq: released });
    expect(prepareRelease(state, callOf(0))).toBeNull();
    expect(() => prepareNodeStep(state, "a", pendingOf(0))).toThrow(`call '${callOf(0)}' is already registered`);
    commitPrepared(log, state, prepareComplete(state, callOf(0), { output: "done" })!);
    expect(state.waits.calls.get(callOf(0))!.state).toBe("answered");
  });

  test("a released flow call registers again as a call of its flow, its node's first scope", async () => {
    const flow = await loadFlow(SHOUT_FLOW);
    const { log, state } = started(flow.compiled);
    commitPrepared(log, state, prepareNodeStarted(state, "a"));
    step(log, state, "a", reply(1, "shout", { text: "hi" }));
    step(log, state, "a", { ...holdOf(0), tool: "shout", source: { kind: "flow", flow: "shout", library: null } });
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ flow: false, scope: null });
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0)] }));
    release(log, state, 0);
    commitPrepared(log, state, prepareNodeWoken(state, "a"));
    step(log, state, "a", { ...pendingOf(0, "shout"), flow: true, arguments: { text: "hi" }, scope: "a/shout#0" });
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ state: "open", flow: true, scope: "a/shout#0", held: false });
  });

  test("an interrupt's CallClosed events follow the calls' registration order", () => {
    const { log, state } = replied({ steers: true });
    step(log, state, "a", holdOf(0, 1));
    step(log, state, "a", pendingOf(1));
    step(log, state, "a", holdOf(2));
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1), callOf(2)], steerInbox: "msgs" }));
    release(log, state, 0);
    const events = commitPrepared(log, state, prepareInterrupt(state, "msgs")!);
    expect(events.filter((event) => event instanceof CallClosed).map((event) => (event as CallClosed).callId)).toEqual([
      callOf(0),
      callOf(1),
      callOf(2),
    ]);
  });

  test("a resume's retried agent takes a host's release or answer before it starts again, and reads it once it does", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", holdOf(1));
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
    expect([state.inFlight.has("a"), state.ready.includes("a")]).toEqual([false, true]);
    const released = release(log, state, 0);
    const answered = prepareComplete(state, callOf(1), { output: "done" })!;
    commitPrepared(log, state, answered);
    expect(() => prepareNodeStep(state, "a", reply(1))).toThrow(
      "node 'a', not started again since its resume, records only a tool_result answering one of its open calls",
    );
    commitPrepared(log, state, prepareNodeStarted(state, "a"));
    expect(state.nodeStartSeqs.get("a")!.at(-1)).toBeGreaterThan(answered.expectedSequence);
    step(log, state, "a", result("c0"));
    expect(state.waits.calls.get(callOf(0))).toMatchObject({ state: "answered", releasedSeq: released });
  });

  test("a host's record for a node that doesn't run, wait or start again is a refused delivery", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    // No record leaves a call registered on such a node, so the state is changed by hand.
    state.inFlight.delete("a");
    for (const deliver of [() => prepareRelease(state, callOf(0)), () => prepareComplete(state, callOf(0), { output: "x" })]) {
      expect(refusedBy(deliver)).toMatchObject({ reason: "unknown_call" });
    }
  });

  test("a wait must name it when it was released in flight, and the node wakes at once", () => {
    const { log, state } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    release(log, state, 0);
    expect(() => prepareNodeWaiting(state, "a", { callIds: [callOf(1)] })).toThrow(`NodeWaiting must name the node's released call '${callOf(0)}'`);
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    expect(wakeableNodes(state.waits)).toEqual(["a"]);
  });

  test("a call released before the evaluation began is its agent's to settle, never to wait on", () => {
    const woken = replied();
    step(woken.log, woken.state, "a", holdOf(0));
    step(woken.log, woken.state, "a", pendingOf(1));
    commitPrepared(woken.log, woken.state, prepareNodeWaiting(woken.state, "a", { callIds: [callOf(0), callOf(1)] }));
    release(woken.log, woken.state, 0);
    commitPrepared(woken.log, woken.state, prepareNodeWoken(woken.state, "a"));
    const before = `NodeWaiting call '${callOf(0)}' was released before the node's latest start or wake; its agent runs it`;
    expect(() => prepareNodeWaiting(woken.state, "a", { callIds: [callOf(0), callOf(1)] })).toThrow(before);
    expect(() => prepareNodeWaiting(woken.state, "a", { callIds: [callOf(1)] })).toThrow(`NodeWaiting must name the node's released call '${callOf(0)}'`);

    // Released in flight, then cut before the wait: the resumed start must settle it too.
    const cut = replied();
    step(cut.log, cut.state, "a", holdOf(0));
    release(cut.log, cut.state, 0);
    commitPrepared(cut.log, cut.state, prepareRunResumed(cut.state, { numWorkers: 0 }));
    commitPrepared(cut.log, cut.state, prepareNodeStarted(cut.state, "a"));
    expect(() => prepareNodeWaiting(cut.state, "a", { callIds: [callOf(0)] })).toThrow(before);
  });

  test("a resume leaves it released, closing only the reply's other open calls", () => {
    const { log, state, path } = replied();
    step(log, state, "a", holdOf(0));
    step(log, state, "a", pendingOf(1));
    release(log, state, 0);
    expect(commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }))).toEqual([
      new RunResumed(),
      new CallClosed({ callId: callOf(1), reason: "crashed" }),
    ]);
    expect(state.waits.calls.get(callOf(0))!.state).toBe("released");
    log.close();
    expect(replayed(state.flow, path).waits.calls.get(callOf(0))!.state).toBe("released");
  });

  test("a log cut at any record around the release replays the waits the run had there: the hold open before it, released from it on", () => {
    const { log, state, path } = replied();
    // The waits after each record from the reply's hold on, by the record count.
    const live = new Map<number, string>();
    const keep = () => live.set(log.nextSequence, JSON.stringify(state.waits));
    keep();
    step(log, state, "a", holdOf(0));
    keep();
    step(log, state, "a", pendingOf(1));
    keep();
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [callOf(0), callOf(1)] }));
    keep();
    const releasedAt = release(log, state, 0, "approved");
    keep();
    commitPrepared(log, state, prepareNodeWoken(state, "a"));
    keep();
    step(log, state, "a", result("c0"));
    keep();
    log.close();
    for (const [count, waits] of live) {
      const cut = replayed(state.flow, path, count);
      expect(JSON.stringify(cut.waits), `cut at ${count}`).toBe(waits);
      const hold = cut.waits.calls.get(callOf(0));
      if (hold === undefined) continue;
      const released = count > releasedAt;
      expect([hold.releasedSeq !== null, prepareRelease(cut, callOf(0)) === null], `cut at ${count}`).toEqual([released, released]);
      if (!released) expect(hold.state).toBe("open");
      expect((prepareRunResumed(cut, { numWorkers: 0 }).record as RunResumedRecord).closedCalls).not.toContain(callOf(0));
    }
  });
});
