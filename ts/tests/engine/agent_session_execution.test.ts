/**
 * Real AGENT execution keeps memory ownership across runs and graph growth: mirrors
 * `tests/engine/test_agent_session_execution.py`, through `runFlow`.
 *
 * Ported: `test_same_authored_agent_queues_real_runs_in_admission_order`,
 * `test_two_session_agents_overlap_on_separate_instances`,
 * `test_cancelled_waiter_exits_but_active_run_keeps_lease_until_reply_is_saved`,
 * `test_agent_session_owns_history_and_a_flow_transcript_is_rejected` (3 rows),
 * `test_loop_iterations_share_one_authored_conversation`,
 * `test_parallel_map_shares_authored_history_without_changing_result_order`,
 * `test_two_callsites_of_one_child_share_one_conversation`,
 * `test_one_session_under_two_providers_fails_at_run_setup` and
 * `test_one_session_under_two_models_of_one_provider_runs`.
 *
 * Changed: Python's worker threads are concurrent `runFlow` promises, and its events and barrier
 * are promises the model awaits; the test's own timeout bounds a deadlock, where Python runs each
 * threaded scenario in a subprocess with a deadline. A queued ticket is observed on the memory's
 * private `queue`, as Python reads `_queue`. Python's `_run_flow_controlled` stop event is
 * `runFlow`'s `signal`, and the run rejects with its reason (Python: `_HostStop`). The model that
 * holds the owner's call ignores the stop, as Python's blocked thread can't be cancelled, so its
 * reply surfaces after the stop. An overlap of one conversation's model calls is recorded and
 * checked after the runs, since a throw inside the model would be the call's error. The inline
 * CODE body is Python's text, which is JavaScript too.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type ConversationView,
  type JsonObject,
  type Memory,
  MemoryOperationError,
  MemorySetupError,
} from "../../src/memory/contracts.ts";
import { conversationViewFromJson } from "../../src/memory/conversation.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { decodeMessage } from "../../src/nodes/agent/memory.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { rejected } from "./run_state_fixtures.ts";
import { caught, runWith, snapshot } from "./session_fakes.ts";

const FLOW = `id: execution-session
name: execution_session
llm_config: {provider: openai, model: fake-model}
input:
  question: !str
nodes:
  answer:
    kind: agent
    long_term_memory: session
    input: {question: "\${input.question}"}
    prompt: \${question}
    output: str
output: \${answer.output}
`;

/** A reply holding only `text`. */
const answer = (text: string): Response =>
  new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });

/**
 * Python's `_Model`: a chat model answering each request with `invoke(messages)`, which may wait.
 * Its reply streams whatever the call's signal does.
 */
class InvokeModel implements ChatModel {
  readonly provider = "openai";
  readonly model = "fake-model";
  readonly invoke: (messages: readonly Message[]) => Response | Promise<Response>;

  constructor(invoke: (messages: readonly Message[]) => Response | Promise<Response>) {
    this.invoke = invoke;
  }

  stream(request: Request): EventStream {
    const invoke = this.invoke;
    async function* events(): AsyncGenerator<Event> {
      const inner = new ScriptedModel([await invoke(request.messages)]).stream(request);
      for (let next = await inner.next(); !next.done; next = await inner.next()) yield next.value;
    }
    return new EventStream(events());
  }
}

/** The text of the last message of `messages`. */
const lastText = (messages: readonly Message[]): string => messages.at(-1)!.text();

/** The texts of the user messages of `messages`. */
const humans = (messages: readonly Message[]): string[] =>
  messages.filter((message) => message.role === "user").map((message) => message.text());

/** The conversation `memory` holds, as a list of none or one view (Python's `_views`). */
function views(memory: SessionMemory): ConversationView[] {
  const raw = snapshot(memory)["conversation"];
  return raw === null || raw === undefined ? [] : [conversationViewFromJson(raw as JsonObject)];
}

/** Wait until `memory` holds `count` queued tickets: actual reservations, not timing. */
async function waitQueue(memory: SessionMemory, count: number): Promise<void> {
  const queue = memory["queue"] as unknown[];
  for (let turn = 0; turn < 1000 && queue.length !== count; turn++) await Bun.sleep(1);
  expect(queue.length).toBe(count);
}

test("the same authored agent queues real runs in admission order", async () => {
  const memory = new SessionMemory();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const flow = await loadFlow(FLOW);
  const calls: Message[][] = [];
  let active = 0;
  let overlapped = false;
  const model = new InvokeModel(async (messages) => {
    active += 1;
    overlapped ||= active !== 1;
    calls.push([...messages]);
    try {
      const question = lastText(messages);
      if (question === "first") {
        entered.resolve();
        await release.promise;
      }
      return answer(`reply:${question}`);
    } finally {
      active -= 1;
    }
  });
  const runs = [runWith(flow, { question: "first" }, memory, model)];
  let settled: PromiseSettledResult<Awaited<(typeof runs)[number]>>[];
  try {
    await entered.promise;
    for (const [index, question] of ["second", "third"].entries()) {
      runs.push(runWith(flow, { question }, memory, model, { numWorkers: 2 }));
      await waitQueue(memory, index + 1);
    }
    const before = snapshot(memory);
    const error = caught(() => memory.reset());
    expect(error).toBeInstanceOf(MemoryOperationError);
    expect((error as MemoryOperationError).code).toBe("session.busy");
    expect(snapshot(memory)).toEqual(before);
    expect(calls).toHaveLength(1);
  } finally {
    release.resolve();
    settled = await Promise.allSettled(runs);
  }
  expect(settled.map((outcome) => outcome.status)).toEqual(["fulfilled", "fulfilled", "fulfilled"]);
  expect(settled.map((outcome) => (outcome.status === "fulfilled" ? outcome.value.output : null))).toEqual([
    "reply:first",
    "reply:second",
    "reply:third",
  ]);
  expect(overlapped).toBe(false);
  expect(calls.map(humans)).toEqual([["first"], ["first", "second"], ["first", "second", "third"]]);
  const [stored] = views(memory);
  expect(stored!.closedInvocations.map((closed) => closed.status)).toEqual(["completed", "completed", "completed"]);
  memory.reset();
  expect(views(memory)).toEqual([]);
});

/** A two-party barrier: each `wait` resolves once both parties arrived, or rejects after 5 s. */
function barrier(): { wait(): Promise<void> } {
  const both = Promise.withResolvers<void>();
  let arrived = 0;
  return {
    wait: () => {
      if (++arrived === 2) both.resolve();
      const timeout = new Promise<void>((_, reject) => setTimeout(() => reject(new Error("barrier broken")), 5000));
      return Promise.race([both.promise, timeout]);
    },
  };
}

test("two session agents overlap on separate instances", async () => {
  const left = new SessionMemory();
  const right = new SessionMemory();
  const meeting = barrier();
  const flow = await loadFlow(`id: two-leaves
name: two_leaves
llm_config: {provider: openai, model: fake-model}
nodes:
  left: {kind: agent, long_term_memory: left_chat, prompt: left, output: str}
  right: {kind: agent, long_term_memory: right_chat, prompt: right, output: str}
output: \${[left.output, right.output]}
`);
  const model = new InvokeModel(async (messages) => {
    await meeting.wait();
    return answer(lastText(messages));
  });
  const memory = new Map<string, Memory>([
    ["left_chat", left],
    ["right_chat", right],
  ]);
  const result = await runWith(flow, {}, memory, model, { numWorkers: 2 });
  expect(result.failure).toBeNull();
  expect(result.output).toEqual(["left", "right"]);
  expect([views(left).length, views(right).length]).toEqual([1, 1]);
});

test("a cancelled waiter exits, but the active run keeps its lease until its reply is saved", async () => {
  const memory = new SessionMemory();
  const flow = await loadFlow(FLOW);
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const ownerStop = new AbortController();
  const waiterStop = new AbortController();
  const calls: string[] = [];
  const model = new InvokeModel(async (messages) => {
    const question = lastText(messages);
    calls.push(question);
    if (question === "owner") {
      entered.resolve();
      await release.promise;
    }
    return answer(`reply:${question}`);
  });
  const controlled = (question: string, stop: AbortController) =>
    runWith(flow, { question }, memory, model, { signal: stop.signal, numWorkers: 2 });
  const ownerReason = new Error("owner stopped");
  const waiterReason = new Error("waiter stopped");
  const owner = rejected(controlled("owner", ownerStop));
  let next: ReturnType<typeof runWith> | null = null;
  try {
    await entered.promise;
    const waiter = rejected(controlled("cancelled", waiterStop));
    await waitQueue(memory, 1);
    waiterStop.abort(waiterReason);
    expect(await waiter).toBe(waiterReason);
    await waitQueue(memory, 0);
    ownerStop.abort(ownerReason);
    next = runWith(flow, { question: "next" }, memory, model);
    await waitQueue(memory, 1);
    expect(calls).toEqual(["owner"]);
    expect(() => memory.reset()).toThrow(/active or queued/);
  } finally {
    release.resolve();
  }
  expect(await owner).toBe(ownerReason);
  expect((await next!).output).toBe("reply:next");
  expect(calls).toEqual(["owner", "next"]);
  const [stored] = views(memory);
  const items = stored!.items.map(decodeMessage);
  expect(humans(items)).toEqual(["owner", "next"]);
  expect(stored!.effects.map((effect) => effect.status)).toEqual(["completed", "completed"]);
  expect(stored!.closedInvocations.map((closed) => closed.status)).toEqual(["interrupted", "completed"]);
  expect(items.some((message) => message.text() === "reply:owner")).toBe(true);
});

for (const [outerSession, localSession] of [
  [true, false],
  [false, true],
  [true, true],
] as const) {
  test(`an agent's session owns history and a flow transcript is rejected: outer ${
    outerSession ? "session" : "notes"
  }, local ${localSession ? "session" : "notes"}`, async () => {
    const outer = outerSession ? new SessionMemory() : new FakeMemory({ recallText: "outer recall" });
    const local = localSession ? new SessionMemory() : new FakeMemory({ recallText: "local recall" });
    const flow = await loadFlow(
      FLOW.replace("    long_term_memory: session\n", "    long_term_memory: local\n").replace(
        "name: execution_session\n",
        "name: execution_session\nlong_term_memory: outer\n",
      ),
    );
    const calls: Message[][] = [];
    const model = new InvokeModel((messages) => {
      calls.push([...messages]);
      return answer("answer");
    });
    const memory = new Map<string, Memory>([
      ["outer", outer],
      ["local", local],
    ]);
    if (outerSession) {
      const result = await runWith(flow, { question: "first" }, memory, model);
      expect(result.status).toBe("failed");
      expect(result.failure!.causeType).toBe("MemoryRecallError");
      expect(result.failure!.message).toContain("not conversation-eligible");
      expect(calls).toEqual([]);
      return;
    }
    for (const question of ["first", "second"]) {
      expect((await runWith(flow, { question }, memory, model)).status).toBe("succeeded");
    }
    expect(humans(calls[1]!)).toEqual(["first", "second"]);
    expect(calls[1]![0]!.text()).toContain("outer recall");
    expect((outer as FakeMemory).recallContexts).toHaveLength(2);
    expect(views(local as SessionMemory)).toHaveLength(1);
  });
}

/** A model answering each request with `reply(callIndex, messages)`, recording each request's messages. */
function recordingModel(reply: (count: number, messages: readonly Message[]) => string) {
  const calls: Message[][] = [];
  const model = new InvokeModel((messages) => {
    calls.push([...messages]);
    return answer(reply(calls.length, messages));
  });
  return { model, calls };
}

test("loop iterations share one authored conversation", async () => {
  const flow = await loadFlow(`id: session-loop
name: session_loop
llm_config: {provider: openai, model: fake-model}
defs:
  body:
    input:
      question: !str
    nodes:
      answer:
        kind: agent
        long_term_memory: session
        input: {question: "\${input.question}"}
        prompt: \${question}
        output: str
      carry:
        kind: code
        input: {answer: "\${answer.output}"}
        code: "return {'question': inputs['answer']}"
        output: {question: str}
    output: \${carry.output}
nodes:
  repeat:
    kind: loop
    call: body
    input: {question: first}
    times: 3
output: \${repeat.output.question}
`);
  const memory = new SessionMemory();
  const { model, calls } = recordingModel((count) => `reply-${count}`);
  const result = await runWith(flow, {}, memory, model);
  expect(result.failure).toBeNull();
  expect(result.output).toBe("reply-3");
  expect(calls.map(humans)).toEqual([["first"], ["first", "reply-1"], ["first", "reply-1", "reply-2"]]);
  const [stored] = views(memory);
  expect(stored!.closedInvocations).toHaveLength(3);
});

test("a parallel MAP shares authored history without changing the result order", async () => {
  const flow = await loadFlow(`id: session-map
name: session_map
llm_config: {provider: openai, model: fake-model}
input:
  questions: !list(str)
defs:
  body:
    input:
      question: !str
    nodes:
      answer:
        kind: agent
        long_term_memory: session
        input: {question: "\${input.question}"}
        prompt: \${question}
        output: str
    output: \${answer.output}
nodes:
  each:
    kind: map
    call: body
    parallel: true
    over: \${input.questions}
    input: {question: "\${item}"}
output: \${each.output}
`);
  const memory = new SessionMemory();
  const { model, calls } = recordingModel((_count, messages) => `reply:${lastText(messages)}`);
  const result = await runWith(flow, { questions: ["one", "two", "three"] }, memory, model, { numWorkers: 3 });
  expect(result.failure).toBeNull();
  expect(result.output).toEqual(["reply:one", "reply:two", "reply:three"]);
  // Admission order may differ from the MAP's input order; each admitted call sees every earlier
  // actual call in its observed execution order.
  const admitted = calls.map(lastText);
  expect([...admitted].sort()).toEqual(["one", "three", "two"]);
  expect(calls.map(humans)).toEqual([1, 2, 3].map((index) => admitted.slice(0, index)));
  const [stored] = views(memory);
  expect(stored!.closedInvocations).toHaveLength(3);
});

test("two callsites of one child share one conversation", async () => {
  const flow = await loadFlow(`id: session-calls
name: session_calls
llm_config: {provider: openai, model: fake-model}
defs:
  body:
    input:
      question: !str
    nodes:
      answer:
        kind: agent
        long_term_memory: session
        input: {question: "\${input.question}"}
        prompt: \${question}
        output: str
    output: \${answer.output}
nodes:
  left: {kind: call, call: body, input: {question: left}}
  right: {kind: call, call: body, input: {question: right}}
output: \${[left.output, right.output]}
`);
  const memory = new SessionMemory();
  const { model, calls } = recordingModel((_count, messages) => lastText(messages));
  for (let run = 0; run < 2; run++) {
    const result = await runWith(flow, {}, memory, model);
    expect(result.failure).toBeNull();
    expect(result.output).toEqual(["left", "right"]);
  }
  expect(calls.map((messages) => humans(messages).length)).toEqual([1, 2, 3, 4]);
  expect(humans(calls.at(-1)!)).toEqual(calls.map(lastText));
  const [stored] = views(memory);
  expect(stored!.closedInvocations).toHaveLength(4);
});

const PROVIDER_SPLIT = `id: provider-split
name: provider_split
defs:
  ask:
    kind: agent
    long_term_memory: session
    prompt: hi
    output: str
  via_one:
    llm_config: {provider: openai, model: one}
    nodes:
      inner: {kind: call, call: ask}
    output: \${inner.output}
  via_two:
    llm_config: {provider: PROVIDER, model: two}
    nodes:
      inner: {kind: call, call: ask}
    output: \${inner.output}
nodes:
  a: {kind: call, call: via_one}
  b: {kind: call, call: via_two}
output: \${[a.output, b.output]}
`;

test("one session under two providers fails at run setup", async () => {
  const flow = await loadFlow(PROVIDER_SPLIT.replace("PROVIDER", "anthropic"));
  const model = new InvokeModel(() => {
    throw new Error("model called before the setup check");
  });
  const error = await rejected(runWith(flow, {}, new SessionMemory(), model));
  expect(error).toBeInstanceOf(MemorySetupError);
  const message = (error as Error).message;
  expect(message).toContain("session memory 'session' resolves provider");
  expect(message).toContain("'a/inner/ask'");
  expect(message).toContain("'b/inner/ask'");
  expect(message).toContain("pin a provider on the AGENT");
});

test("one session under two models of one provider runs", async () => {
  const flow = await loadFlow(PROVIDER_SPLIT.replace("PROVIDER", "openai"));
  const result = await runWith(flow, {}, new SessionMemory(), new InvokeModel(() => answer("ok")));
  expect(result.failure).toBeNull();
  expect(result.output).toEqual(["ok", "ok"]);
});
