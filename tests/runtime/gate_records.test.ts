/**
 * The run reducer's rules for the records of a host's check before each agent tool call: a gated
 * run's start and resumes, a refused call's result, a note, a held call's `pending` step and its
 * `released` step.
 */

import { describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { RunResumedRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareComplete,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareRecord,
  prepareRunResumed,
  type RunState,
} from "../../src/runtime/state.ts";
import { GATE_TEXT_MAX_BYTES } from "../../src/tools/gate.ts";
import { RUN_ID } from "../helpers/run_state_fixtures.ts";
import { askPending, paramFlow, replayed, started, tool } from "../helpers/waits_fixtures.ts";

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

/** A run whose agent `a` is running after a reply calling `lookup` three times (c0, c1, c2). */
function replied() {
  const run = started(paramFlow([], agent("a"), tool("t", "ask")));
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
    "a hold's pending step holds exactly kind, call_id, tool, tool_call_id, reply, index, held, source and group_end";
  const NEVER_HELD = (tool: string) => `a hold's tool can't be ${tool}, whose calls are never checked`;
  const GROUP_END = "a hold's group_end must be an int from its index 1 to its reply's last call 2";
  test.each<[string, (step: Record<string, unknown>) => void, string]>([
    ["a note", (s) => (s["note"] = "x"), EXACT],
    ["a deadline", (s) => (s["deadline"] = "2026-10-21T00:00:00.000Z"), EXACT],
    ["flow fields", (s) => Object.assign(s, { flow: true, arguments: {}, scope: "a/f#0" }), EXACT],
    ["no source", (s) => delete s["source"], EXACT],
    ["no held", (s) => delete s["held"], "a pending step holds source and group_end only as a hold"],
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
