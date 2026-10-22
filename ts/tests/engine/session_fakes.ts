/**
 * A unit-only backend-owned turn harness without a provider or run log: a port of
 * `tests/engine/_session_fakes.py`, plus the promise helpers, the hand-built `runAgent` context
 * and the one-AGENT session flow run through `FlowEngine` that the session tests share, and the
 * `runFlow` and `resumeFlow` helpers of the run-level memory tests (`PLAIN_FLOW`, `runWith`,
 * `resumeWith`, `CrashAt`, `crashes`, `StopAfterFirstReply`, `NO_ENV`, `sentTexts`, `reply`,
 * `backwardReply`, `memorySections`, `recording`).
 */

import { expect } from "bun:test";

import { type CompiledFlow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { RunFailed, RunSucceeded } from "../../src/events.ts";
import { type ChatModel, EventStream } from "../../src/llm/events.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { Message, type Request, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import {
  ConversationInvocation,
  ConversationItem,
  ConversationRequest,
  type ConversationTurn,
  type ConversationView,
  DeclSite,
  type JsonObject,
  type Memory,
  MemoryContext,
  MemoryOperationError,
  MemoryScope,
  MemorySpec,
} from "../../src/memory/contracts.ts";
import { conversationViewFromJson } from "../../src/memory/conversation.ts";
import { type Session, SessionMemory, type Turn } from "../../src/memory/session.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { decodeMessage } from "../../src/nodes/agent/memory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import { type ResumeFlowOptions, resumeFlow, type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import { JsonlRunLog, type RunRecord } from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import type { Type } from "../../src/typesys/values.ts";
import { Crash, engine, graph, rejected, serving } from "./run_state_fixtures.ts";

export { rejected };

// Cloned snapshots and nested test values are edited freely.
export type Mutable = any;

/** The value `fn` throws. */
export function caught(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** The `MemoryOperationError` `fn` throws, checked to carry `code` and `message`. */
export function failure(fn: () => unknown, code: string, message: string): MemoryOperationError {
  const error = caught(fn);
  expect(error).toBeInstanceOf(MemoryOperationError);
  expect((error as MemoryOperationError).code).toBe(code);
  expect((error as MemoryOperationError).message).toBe(message);
  return error as MemoryOperationError;
}

/** A signal that never aborts, for Python's `check_stopped=lambda: None`. */
export const NEVER: AbortSignal = new AbortController().signal;

/** One fake-format request for invocation `[runId, sequence]`. */
export function request(
  runId = "a",
  sequence = 1,
  consumer: readonly string[] = ["root", "agent"],
  compatibility?: JsonObject,
): ConversationRequest {
  return new ConversationRequest(
    consumer,
    new ConversationInvocation(runId, sequence),
    compatibility ?? { format: "fake-v1" },
  );
}

/** One fake-format message item. */
export function item(itemId: string, text: string, role = "assistant"): ConversationItem {
  return new ConversationItem(itemId, "fake-v1", { role, content: text });
}

/** The current snapshot of `memory`, read through a fresh run. */
export function snapshot(memory: { openRun(): { readonly snapshot: JsonObject; close(): void } }): JsonObject {
  const opened = memory.openRun();
  try {
    return opened.snapshot;
  } finally {
    opened.close();
  }
}

/** One open run of a memory; `frame` is the session, as Python's harness names it. */
export interface Harness {
  memory: SessionMemory;
  session: Session;
  frame: Session;
}

/** Open a run of `memory` (a fresh one by default), await `fn` with it, then close the run. */
export async function withHarness<T>(
  fn: (harness: Harness) => T | Promise<T>,
  memory: SessionMemory = new SessionMemory(),
): Promise<T> {
  const session = memory.openRun();
  try {
    return await fn({ memory, session, frame: session });
  } finally {
    session.close();
  }
}

/** Persist one model request and its completed reply item. */
export function appendReply(turn: Turn, itemId: string, text: string): void {
  const ordinal = turn.request("model", { prompt: text });
  turn.observe(ordinal, "completed", { items: [item(itemId, text)] });
}

/**
 * Python's `with source.open_turn(request) as turn:` block: open the turn, await `fn`, then close
 * it with the thrown value (rethrown) or normally.
 */
export async function withTurn<T>(
  source: Session,
  conversationRequest: ConversationRequest,
  fn: (turn: Turn) => T | Promise<T>,
  signal: AbortSignal = NEVER,
): Promise<T> {
  const turn = await source.openTurn(conversationRequest, { signal });
  let result: T;
  try {
    result = await fn(turn);
  } catch (error) {
    turn.close(error);
    throw error;
  }
  turn.close();
  return result;
}

/** A hand-made turn of only `request` and `observe`, as Python's `SimpleNamespace` turns. */
export function fakeTurn(
  request: () => number | Promise<number>,
  observe: (ordinal: number, status: string) => void | Promise<void>,
): ConversationTurn {
  return { request, observe } as unknown as ConversationTurn;
}

/** A promise with its outcome recorded as it settles. */
export interface Settled<T> {
  readonly promise: Promise<T>;
  settled: boolean;
  value?: T;
  error?: unknown;
}

/**
 * Track `promise`, attaching its handlers at once, since Bun fails a test when a promise rejects
 * before a handler is attached.
 */
export function settle<T>(promise: Promise<T>): Settled<T> {
  const state: Settled<T> = { promise, settled: false };
  promise.then(
    (value) => {
      state.settled = true;
      state.value = value;
    },
    (error: unknown) => {
      state.settled = true;
      state.error = error;
    },
  );
  return state;
}

/** Wait one macrotask, so every pending promise reaction has run. */
export function macrotask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 10));
}

/**
 * A hand-built `runAgent` context: AGENT `agent` asked `current question` by `model` with
 * `transcript` as its memory turn, no tools, and `overrides`.
 */
export function agentContext(
  model: ChatModel,
  transcript: ConversationTurn,
  overrides: Partial<AgentRunContext> = {},
): AgentRunContext {
  return {
    nodeId: "agent",
    prompt: "current question",
    tools: [],
    llmConfig: null,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext({ transcript }),
    toolset: null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: { env: () => undefined, createModel: () => model },
    sent: () => [],
    ...overrides,
  };
}

/** The conversation view of `memory`'s current snapshot (Python's `_view`). */
export function view(memory: SessionMemory): ConversationView {
  return conversationViewFromJson(snapshot(memory)["conversation"] as JsonObject);
}

/** The kind and status of each effect `memory` holds. */
export function effectsOf(memory: SessionMemory): [string, string][] {
  return view(memory).effects.map((effect) => [effect.kind, effect.status]);
}

/** The assistant messages `memory` stores, in order. */
export function storedReplies(memory: SessionMemory): Message[] {
  return view(memory)
    .items.map(decodeMessage)
    .filter((message) => message.role === "assistant");
}

/** The LLM config of `sessionFlow`'s AGENT. */
export const SESSION_CONFIG = { provider: "openai", model: "test-model" };

/**
 * Python's `_FLOW`: one AGENT `nodeId` (`agent` by default) owning AGENT memory `session`,
 * prompted with the run's input `question`, its output (of type `output`, text by default) the
 * flow's; the other options go to the node.
 */
export function sessionFlow(
  options: ConstructorParameters<typeof AgentNode>[1] & { nodeId?: string; output?: Type } = {},
): CompiledFlow {
  const { nodeId = "agent", output, ...rest } = options;
  const node = new AgentNode(nodeId, { prompt: "${question}", llmConfig: SESSION_CONFIG, ...rest });
  node.params = [new Parameter("question")];
  if (output !== undefined) node.outputContract = new ValueContract(output);
  node.memoryScope = new MemoryScope([nodeId], { memories: ["session"] });
  return graph([node], [[START_ID, nodeId]], {
    wiring: { [nodeId]: { question: "${input.question}" } },
    outputs: [new FlowOutput("result", `\${${nodeId}.output}`)],
  });
}

/**
 * Await `fn` with a context opener over a `MemoryRuntime` whose AGENT memory `session`, declared on
 * node `nodeId`, is `memory`; the runtime is closed after.
 */
export async function withSessionOpener<T>(
  memory: SessionMemory,
  nodeId: string,
  fn: (opener: MemoryContextOpener) => Promise<T>,
): Promise<T> {
  const spec = new MemorySpec("session", "session", {}, [new DeclSite([], ["root"], nodeId)], true);
  const runtime = await MemoryRuntime.open({
    memories: [spec],
    origins: new Map([["session", null]]),
    memory: new Map([["session", memory]]),
  });
  try {
    return await fn((scope, opened, options) => runtime.openContext(scope, opened, options));
  } finally {
    await runtime.close();
  }
}

/**
 * Run `flow` through `FlowEngine` with `memory` backing AGENT memory `session` of node `nodeId`
 * (`agent` by default; `withSessionOpener`); the other options go to `engine`.
 *
 * @returns The run's terminal event.
 * @throws What the run throws, such as a stop's reason.
 */
export function runSessionFlow(
  flow: CompiledFlow,
  memory: SessionMemory,
  options: Parameters<typeof engine>[1] & { nodeId?: string } = {},
): Promise<RunSucceeded | RunFailed> {
  const { nodeId = "agent", ...rest } = options;
  return withSessionOpener(memory, nodeId, async (memoryContextOpener) => {
    const built = engine(flow, { ...rest, memoryContextOpener });
    await built.run();
    return built.state.result as RunSucceeded | RunFailed;
  });
}

/**
 * Python's `_FLOW` of `tests/engine/test_agent_session.py`: AGENT `agent` owning AGENT memory
 * `session`, prompted with the run's input `question`, its text output the flow's.
 */
export const PLAIN_FLOW = `id: plain-session
name: plain_session
llm_config: {provider: openai, model: test-model}
input:
  question: !str
nodes:
  agent:
    kind: agent
    long_term_memory: session
    prompt: \${question}
    input:
      question: \${input.question}
    output: str
output: \${agent.output}
`;

/** An environment holding nothing, so a run reads no host setting. */
export const NO_ENV = (): undefined => undefined;

/**
 * `runFlow` of `flow` with `inputs`, `memory` backing memory `session` (or each memory key of a
 * `Map`), `model` serving every AGENT, in `NO_ENV`; the other options go to `runFlow`.
 */
export function runWith(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  memory: SessionMemory | ReadonlyMap<string, Memory> | null,
  model: ChatModel,
  options: RunFlowOptions = {},
): Promise<RunResult> {
  const registry = memory instanceof SessionMemory ? new Map([["session", memory]]) : memory;
  return runFlow(flow, inputs, { memory: registry, models: serving(model), env: NO_ENV, ...options });
}

/**
 * `resumeFlow` of `flow` from `log`, with `runWith`'s `memory`, `model` and `NO_ENV`; the other
 * options go to `resumeFlow`.
 */
export function resumeWith(
  flow: ComposedFlow,
  log: string,
  memory: SessionMemory | ReadonlyMap<string, Memory> | null,
  model: ChatModel,
  options: Omit<ResumeFlowOptions, "log"> = {},
): Promise<RunResult> {
  const registry = memory instanceof SessionMemory ? new Map([["session", memory]]) : memory;
  return resumeFlow(flow, { log, memory: registry, models: serving(model), env: NO_ENV, ...options });
}

/**
 * A `ScriptedModel` that sets `frozen` as it serves request `at` (1-based); `crashes` reads it.
 * Its config is `SESSION_CONFIG` unless `config` is given.
 */
export class CrashAt extends ScriptedModel {
  readonly at: number;
  frozen = false;

  constructor(replies: ScriptedReply[], at: number, config: { provider?: string; model?: string } = SESSION_CONFIG) {
    super(replies, config);
    this.at = at;
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    if (this.requests.length === this.at - 1) this.frozen = true;
    return super.stream(request, options);
  }
}

/**
 * Await `run`, which must reject with the `Crash` every durable log throws once `model` froze
 * them: a process lost while serving request `at`, so no record of its outcome reaches the log.
 */
export async function crashes(model: CrashAt, run: () => Promise<unknown>): Promise<void> {
  const crash = new Crash("process lost");
  const append = JsonlRunLog.prototype.append;
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (model.frozen) throw crash;
    return append.call(this, record);
  };
  try {
    expect(await rejected(run())).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
}

/**
 * A `ScriptedModel` whose first call's stream aborts `controller` once its stream is read to the end: a
 * host stop landing after the reply arrived, before the node stores it.
 */
export class StopAfterFirstReply extends ScriptedModel {
  readonly controller: AbortController;

  constructor(
    replies: ScriptedReply[],
    controller: AbortController,
    options: { provider?: string; model?: string } = {},
  ) {
    super(replies, options);
    this.controller = controller;
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const first = this.requests.length === 0;
    const inner = super.stream(request, options);
    if (!first) return inner;
    const controller = this.controller;
    async function* events() {
      yield* inner;
      controller.abort(new Error("host stop"));
    }
    return new EventStream(events(), () => inner.close());
  }
}

/** The texts of the messages of `model`'s request `index` (the last by default) after its system prompt. */
export function sentTexts(model: { requests: readonly { messages: readonly Message[] }[] }, index = -1): string[] {
  return model.requests.at(index)!.messages.slice(1).map((message) => message.text());
}

/** A text-only assistant reply. */
export function reply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
}

/** An AGENT backward reply whose only feedback is the self-feedback `notes`. */
export function backwardReply(...notes: string[]): Response {
  return reply(JSON.stringify({ input_feedback: {}, self_feedback: notes, variable_feedback: {} }));
}

/** The memory sections the system prompt among `messages` shows; none without a memory context. */
export function memorySections(messages: readonly Message[]): { memory: string; text: string }[] {
  const system = messages.find((message) => message.role === "system")?.text() ?? "";
  if (!system.includes("Memory context (canonical JSON")) return [];
  return (JSON.parse(system.slice(system.lastIndexOf("\n") + 1)) as { sections: { memory: string; text: string }[] })
    .sections;
}

/** Model access over `NO_ENV` that records each config it builds from and serves `pick(config)`. */
export function recording(
  pick: (config: Readonly<Record<string, unknown>>) => ChatModel,
): ModelAccess & { configs: Readonly<Record<string, unknown>>[] } {
  const configs: Readonly<Record<string, unknown>>[] = [];
  return {
    configs,
    env: NO_ENV,
    createModel: (config) => {
      configs.push(config);
      return pick(config);
    },
  };
}
