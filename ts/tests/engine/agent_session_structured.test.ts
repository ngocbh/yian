/**
 * Structured AGENT calls with a memory turn: each attempt's request and raw reply are stored,
 * independently of parsing; mirrors `tests/engine/test_agent_session_structured.py`.
 *
 * Ported: `test_structured_exhaustion_retains_schema_error_identity_and_origin` (native and
 * prompt; the failing reply is stored with effect `completed`),
 * `test_native_carrier_without_a_parsed_value_is_corrected_and_retried` and
 * `test_native_carriers_without_a_parsed_value_fail_once_retries_are_spent` (TS's tool strategy,
 * over a stub `RouteModel`, with a reply that doesn't call the schema's tool), the `request` and
 * `observe` stages of `test_structured_memory_faults_do_not_retry_model` (native and prompt), and
 * the `before`, `after_reply`, `after_bad_reply` and `after_parse` stages of
 * `test_structured_stop_boundaries_preserve_surfaced_raw`. Through `runFlow` (native and prompt):
 * `test_structured_runs_restore_actual_history_without_schema_tool_execution`,
 * `test_structured_retries_store_raw_replies_and_only_new_correction_deltas` and
 * `test_tool_then_structured_purpose_distinguishes_schema_calls_from_real_grants`.
 *
 * Changed: a stop is the aborted signal, its reason what the call throws, rather than a
 * `_HostStop`; a stop after the reply is aborted by the live sink at the call's
 * `ModelCallFinished`, and one after parsing by a spied `StructuredPlan.parse`, as Python patches
 * `_unwrap`. `test_structured_stop_boundaries_preserve_surfaced_raw`'s `after_bind` stage: TS has
 * no binding step, so the `before` stage covers it. Exhausted corrections throw
 * `OutputInvalidError` with the schema error as `cause`, its location in `path`, rather than the
 * schema error itself. `test_provider_error_and_failed_publication_bypass_retry_with_original_public_identity`
 * calls `generateStructured` with a turn rather than `run_flow`, so it sees the `CapabilityFault`
 * carrying the original, as Python's `generate_structured` raises it (Python's test sees the
 * original only through `run_flow`'s evaluator unwrapping it); the provider error is a
 * `ProviderError`. Its `FutureCancelledError` row is not ported: TS has no `FutureCancelledError`,
 * and a stop during the call is covered by the stop rows.
 * `test_no_session_provider_correction_text_and_retry_behavior_are_unchanged`:
 * TS never corrects a provider error, with or without a turn; it propagates after one call
 * (transport failures are `callWithRetries`' job). In the `runFlow` rows, the native strategy is
 * `gpt-5.5`'s, whose plan answers in text (Python: LangChain's `with_structured_output` carrier with
 * a `format-id` tool call), so the stored native reply is checked by its text, not a tool call id,
 * and the native correction by the type error's text (Python: pydantic's error string), with
 * TS's `does not match declared type int` for Python's `valid integer`. One script serves the chat
 * and structured calls of both strategies (Python's native calls go to a separate bound model), so
 * the second run's chat request is request 3 for both.
 *
 * TS-only: the first request stores `newMessages` (the conversation by default) plus the prompt
 * strategy's instruction, and each later one only its correction, under the attempt's purpose.
 *
 * Not ported:
 * - the `encode` stage of `test_structured_memory_faults_do_not_retry_model`: a TS `Message`
 *   always encodes, and the `observe` stage covers a fault after the call.
 * - `test_native_malformed_carriers_fail_without_retry_and_keep_available_raw`: LangChain's
 *   `include_raw` carriers; TS's call returns a `Response`.
 * - `test_native_carrier_without_a_parsed_value_and_a_nonfinite_raw_is_retried_unstored` and
 *   `test_native_carrier_without_a_parsed_value_and_a_deep_raw_is_retried_unstored`: a `Message`
 *   can hold neither a non-finite number nor a reply nested past the value limit.
 * - `test_provider_cancellation_with_failed_publication_is_not_a_structured_retry` and
 *   `test_session_future_cancellation_with_published_interruption_does_not_retry`:
 *   `FutureCancelledError` and asyncio cancellation; TS's equivalent, a stop during the call, is
 *   covered by the stop rows and by `invokeModel`'s tests.
 * - `test_sanitized_provider_retry_exhaustion_preserves_original_error` and
 *   `test_session_provider_corrections_do_not_send_or_store_exception_secrets`: TS never corrects
 *   a provider error, so no provider error text reaches a correction.
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished } from "../../src/events.ts";
import { NoAuth } from "../../src/llm/auth.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { ChatModel, EventStream } from "../../src/llm/events.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import { Message, type Part, type Request, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { MissingToolCallError, StructuredPlan } from "../../src/llm/structured.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import type { ConversationView } from "../../src/memory/contracts.ts";
import { conversationViewFromJson } from "../../src/memory/conversation.ts";
import { SessionMemory, type Turn } from "../../src/memory/session.ts";
import type { LiveSinks } from "../../src/nodes/agent/live.ts";
import { decodeMessage } from "../../src/nodes/agent/memory.ts";
import { generateStructured, OutputInvalidError, userMessage } from "../../src/nodes/agent/structured.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { logJson } from "../../src/typesys/json.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import {
  fakeTurn,
  NEVER,
  PLAIN_FLOW,
  rejected,
  request,
  runWith,
  snapshot,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const STRATEGIES = [
  ["native", NATIVE],
  ["prompt", PROMPTED],
] as const;
const INT = Type.scalar(ValueKind.INTEGER);
const MISSING = "the reply did not call the ScalarWrapper tool; call it with the answer";

function reply(...parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "stop" });
}

function text(text: string): Response {
  return reply(new TextPart(text));
}

const QUESTION = userMessage("question");

/** A `RouteModel` on OpenAI Chat (`gpt-5.5`, so the tool strategy) that replays `replies`. */
class ToolRoute extends RouteModel {
  readonly script: ScriptedModel;

  constructor(replies: ScriptedReply[]) {
    super({
      route: new Route({ provider: "openai", baseUrl: "https://api.example", path: "x", auth: new NoAuth() }),
      protocol: new OpenAIChat({ model: "gpt-5.5", provider: "openai" }),
    });
    this.script = new ScriptedModel(replies, NATIVE);
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    return this.script.stream(request, options);
  }
}

/** The conversation `memory` holds, read through a fresh run (Python's `_view`). */
function view(memory: SessionMemory): ConversationView {
  return conversationViewFromJson(snapshot(memory)["conversation"] as never);
}

/** The roles of the messages `view` stores, in order. */
function roles(stored: ConversationView): string[] {
  return stored.items.map((item) => decodeMessage(item).role);
}

/**
 * Run `fn` in a turn of `memory` opened with `signal`, returning its outcome; a thrown value is
 * returned as `error`.
 */
async function inTurn<T>(
  memory: SessionMemory,
  fn: (turn: Turn) => Promise<T>,
  signal: AbortSignal = NEVER,
): Promise<{ result?: T; error?: unknown }> {
  try {
    return { result: await withHarness(({ session }) => withTurn(session, request(), fn, signal), memory) };
  } catch (error) {
    return { error };
  }
}

/** Generate an `int` from `model` answering `[QUESTION]` in a turn of `memory`. */
function generateInTurn(
  model: ChatModel,
  memory: SessionMemory,
  retries?: number,
): Promise<{ result?: unknown; error?: unknown }> {
  return inTurn(memory, (turn) => generateStructured(model, [QUESTION], INT, { retries, transcript: turn }));
}

describe.each(STRATEGIES)("%s", (_label, strategy) => {
  test("test_structured_exhaustion_retains_schema_error_identity_and_origin", async () => {
    const memory = new SessionMemory();
    const { error } = await generateInTurn(new ScriptedModel([text('{"value": "bad"}')], strategy), memory, 0);
    expect(error).toBeInstanceOf(OutputInvalidError);
    expect((error as OutputInvalidError).cause).toBeInstanceOf(TypeCheckError);
    expect((error as OutputInvalidError).path).toEqual([]);
    const stored = view(memory);
    expect(stored.effects.at(-1)!.status).toBe("completed");
    expect(stored.closedInvocations.at(-1)!.status).toBe("failed");
    expect(roles(stored).at(-1)).toBe("assistant");
  });

  for (const stage of ["request", "observe"] as const) {
    test(`test_structured_memory_faults_do_not_retry_model: ${stage}`, async () => {
      const model = new ScriptedModel([text('{"value": 1}'), text('{"value": 2}')], strategy);
      const fault = new RangeError("memory fault");
      const turn = fakeTurn(
        () => {
          if (stage === "request") throw fault;
          return 1;
        },
        () => {
          if (stage === "observe") throw fault;
        },
      );
      const error = await rejected(generateStructured(model, [], INT, { transcript: turn }));
      expect(error).toBeInstanceOf(CapabilityFault);
      expect((error as CapabilityFault).original).toBe(fault);
      expect(model.requests).toHaveLength(stage === "request" ? 0 : 1);
    });
  }

  test("test_provider_error_and_failed_publication_bypass_retry_with_original_public_identity", async () => {
    const primary = new ProviderError("server", "provider failed", { status: 500 });
    const secondary = new Error("cannot publish failure");
    const model = new ScriptedModel([primary, text('{"value": 2}')], strategy);
    const memory = new SessionMemory();
    const { error } = await inTurn(memory, (turn) => {
      const observe = turn.observe.bind(turn);
      spyOn(turn, "observe").mockImplementation((ordinal, status, options) => {
        if (status === "failed" || status === "interrupted") throw secondary;
        observe(ordinal, status, options);
      });
      return generateStructured(model, [QUESTION], INT, { transcript: turn });
    });
    expect(error).toBeInstanceOf(CapabilityFault);
    expect((error as CapabilityFault).original).toBe(primary);
    expect(model.requests).toHaveLength(1);
    expect((primary as Error & { notes?: string[] }).notes).toEqual([
      "Conversation observation failed: Error: cannot publish failure",
    ]);
    const stored = view(memory);
    expect(stored.effects.map((effect) => effect.status)).toEqual(["pending"]);
    expect(stored.closedInvocations.at(-1)!.status).toBe("failed");
  });

  for (const withTranscript of [false, true]) {
    const label = `a provider error is never corrected, ${withTranscript ? "with" : "without"} a turn`;
    test(`test_no_session_provider_correction_text_and_retry_behavior_are_unchanged: ${label}`, async () => {
      const primary = new ProviderError("server", "original provider diagnostic", { status: 500 });
      const model = new ScriptedModel([primary, text('{"value": 2}')], strategy);
      const memory = new SessionMemory();
      const { error } = withTranscript
        ? await generateInTurn(model, memory)
        : { error: await rejected(generateStructured(model, [QUESTION], INT)) };
      expect(error).toBe(primary);
      expect(model.requests).toHaveLength(1);
      if (withTranscript) expect(view(memory).effects.map((effect) => effect.status)).toEqual(["failed"]);
    });
  }

  for (const stage of ["before", "after_reply", "after_bad_reply", "after_parse"] as const) {
    test(`test_structured_stop_boundaries_preserve_surfaced_raw: ${stage}`, async () => {
      const controller = new AbortController();
      const stop = new Error("stop structured");
      const value = stage === "after_bad_reply" ? '"bad"' : "1";
      const model = new ScriptedModel([text(`{"value": ${value}}`)], strategy);
      // The successful call's `ModelCallFinished` is published once its reply has surfaced.
      const live: LiveSinks = {
        nodeId: "agent",
        provider: null,
        model: null,
        emitEvent: (event) => {
          if (event instanceof ModelCallFinished && stage.startsWith("after_") && stage !== "after_parse") {
            controller.abort(stop);
          }
        },
        emitStream: null,
      };
      const memory = new SessionMemory();
      // The turn's close throws a pending stop too, so the call's own outcome is checked.
      let thrown: unknown;
      const parse = StructuredPlan.prototype.parse;
      const spy = spyOn(StructuredPlan.prototype, "parse").mockImplementation(function (this: StructuredPlan, response) {
        const data = parse.call(this, response);
        if (stage === "after_parse") controller.abort(stop);
        return data;
      });
      try {
        const { error } = await inTurn(
          memory,
          async (turn) => {
            if (stage === "before") controller.abort(stop);
            thrown = await rejected(
              generateStructured(model, [QUESTION], INT, { signal: controller.signal, live, transcript: turn }),
            );
          },
          controller.signal,
        );
        expect(thrown).toBe(stop);
        expect(error).toBe(stop);
      } finally {
        spy.mockRestore();
      }
      const surfaced = stage !== "before";
      expect(model.requests).toHaveLength(surfaced ? 1 : 0);
      const stored = view(memory);
      expect(stored.closedInvocations.at(-1)!.status).toBe("interrupted");
      if (surfaced) {
        expect(stored.effects.at(-1)!.status).toBe("completed");
        expect(roles(stored).at(-1)).toBe("assistant");
      } else {
        expect([stored.effects, stored.items]).toEqual([[], []]);
      }
    });
  }

  test("the first request stores newMessages and the instruction, a later one only its correction", async () => {
    const model = new ScriptedModel([text("bad json"), text('{"value": 2}')], strategy);
    const memory = new SessionMemory();
    const earlier = userMessage("earlier, already stored");
    const { result } = await inTurn(memory, (turn) =>
      generateStructured(model, [earlier, QUESTION], INT, { retries: 1, transcript: turn, newMessages: [QUESTION] }),
    );
    expect(result).toBe(2n);
    const stored = view(memory);
    const purpose = strategy === PROMPTED ? "structured_fallback" : "structured_native";
    expect(stored.effects.map((effect) => [effect.status, effect.request["purpose"]])).toEqual([
      ["completed", purpose],
      ["completed", purpose],
    ]);
    const messages = stored.items.map(decodeMessage);
    const [first, second] = model.requests.map((sent) => sent.messages);
    const instruction = strategy === PROMPTED ? [first!.at(-1)!] : [];
    const replies = [text("bad json").message, text('{"value": 2}').message];
    expect(messages).toEqual([QUESTION, ...instruction, replies[0]!, second!.at(-1)!, replies[1]!]);
    expect(second!.at(-1)!.text()).toStartWith("Your previous output was invalid");
    expect(second!.some((message) => message.role === "assistant")).toBe(false);
  });
});

describe("a reply without the schema's tool call", () => {
  test("test_native_carrier_without_a_parsed_value_is_corrected_and_retried", async () => {
    const call = new ToolCallPart({ id: "format-id", name: "ScalarWrapper", arguments: { value: 7n } });
    const model = new ToolRoute([text("42"), reply(call)]);
    const memory = new SessionMemory();
    const { result } = await generateInTurn(model, memory, 2);
    expect(result).toBe(7n);
    const requests = model.script.requests;
    expect(requests).toHaveLength(2);
    const stored = view(memory);
    expect(roles(stored).slice(-3)).toEqual(["assistant", "user", "assistant"]);
    expect(stored.effects.every((effect) => effect.status === "completed")).toBe(true);
    expect(requests[1]!.messages.some((message) => message.role === "assistant")).toBe(false);
    const correction = requests[1]!.messages.at(-1)!;
    expect(correction.role).toBe("user");
    expect(correction.text()).toContain(MISSING);
  });

  test("test_native_carriers_without_a_parsed_value_fail_once_retries_are_spent", async () => {
    const model = new ToolRoute([text("42"), text("42")]);
    const memory = new SessionMemory();
    const { error } = await generateInTurn(model, memory, 1);
    expect(model.script.requests).toHaveLength(2);
    expect(error).toBeInstanceOf(OutputInvalidError);
    expect((error as Error).cause).toBeInstanceOf(MissingToolCallError);
    expect((error as Error).message).toBe(MISSING);
    expect(roles(view(memory)).filter((role) => role === "assistant")).toHaveLength(2);
  });
});

/** Python's `_typed_flow`: `PLAIN_FLOW` on `strategy`'s model with an `int` output. */
function typedFlow(strategy: typeof NATIVE | typeof PROMPTED, tools = ""): Promise<ComposedFlow> {
  const config = `{provider: ${strategy.provider}, model: ${strategy.model}}`;
  return loadFlow(
    PLAIN_FLOW.replace("{provider: openai, model: test-model}", config)
      .replace("    output: str", "    output: int")
      .replace("    kind: agent\n", `    kind: agent\n${tools}`),
  );
}

describe.each(STRATEGIES)("%s, through runFlow", (label, strategy) => {
  const native = label === "native";
  const purpose = native ? "structured_native" : "structured_fallback";

  test("test_structured_runs_restore_actual_history_without_schema_tool_execution", async () => {
    const model = new ScriptedModel([text('{"value":1}'), text('{"value":2}')], strategy);
    const memory = new SessionMemory();
    const flow = await typedFlow(strategy);
    expect((await runWith(flow, { question: "first" }, memory, model)).output).toBe(1n);
    expect((await runWith(flow, { question: "second" }, memory, model)).output).toBe(2n);
    const last = model.requests.at(-1)!.messages;
    expect(last.some((message) => message.text() === "first")).toBe(true);
    if (native) {
      expect(last.some((message) => message.role === "user" && message.text().includes("context only"))).toBe(true);
      expect(
        last.some((message) => message.role === "tool" || message.parts.some((part) => part instanceof ToolCallPart)),
      ).toBe(false);
    }
    const stored = view(memory);
    expect(stored.effects.map((effect) => [effect.status, effect.request["purpose"]])).toEqual([
      ["completed", purpose],
      ["completed", purpose],
    ]);
    expect(stored.items).toHaveLength(native ? 6 : 8);
    if (native) expect(decodeMessage(stored.items[2]!).text()).toBe('{"value":1}');
  });

  test("test_structured_retries_store_raw_replies_and_only_new_correction_deltas", async () => {
    const model = new ScriptedModel([text('{"value":"bad"}'), text('{"value":2}')], strategy);
    const memory = new SessionMemory();
    const result = await runWith(await typedFlow(strategy), { question: "first" }, memory, model);
    expect(result.output).toBe(2n);
    expect(model.requests).toHaveLength(2);
    const retried = model.requests[1]!.messages;
    expect(retried.some((message) => message.role === "assistant")).toBe(false);
    const correction = retried.at(-1)!.text();
    expect(correction).toContain("previous output was invalid");
    expect(correction).toContain("does not match declared type int");
    const stored = view(memory);
    expect(stored.effects.map((effect) => effect.status)).toEqual(["completed", "completed"]);
    expect(stored.items).toHaveLength(native ? 5 : 6);
    const messages = stored.items.map(decodeMessage);
    expect(messages.filter((message) => message.role === "assistant")).toHaveLength(2);
    const corrections = messages.filter((message) => message.text().includes("previous output was invalid"));
    expect(corrections.map((message) => message.text())).toEqual([correction]);
    expect(() => logJson(stored.items.map((item) => item.payload))).not.toThrow();
  });

  test("test_tool_then_structured_purpose_distinguishes_schema_calls_from_real_grants", async () => {
    const calls: unknown[] = [];
    const tool = new FunctionTool({
      name: "ScalarWrapper",
      description: "A real tool named as the schema is.",
      parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
      execute: (args) => {
        calls.push(args);
        return `result:${args.value}`;
      },
    });
    const call = new ToolCallPart({ id: "format-id", name: "ScalarWrapper", arguments: { value: 1 } });
    const model = new ScriptedModel(
      [reply(call), text("ready"), text('{"value":9}'), text("ready again"), text('{"value":10}')],
      strategy,
    );
    const memory = new SessionMemory();
    const flow = await typedFlow(strategy, "    tools: [ScalarWrapper]\n");
    const tools = { ScalarWrapper: tool };
    expect((await runWith(flow, { question: "first" }, memory, model, { tools })).output).toBe(9n);
    expect((await runWith(flow, { question: "second" }, memory, model, { tools })).output).toBe(10n);
    expect(calls).toEqual([{ value: 1 }]);
    // The second run's chat request.
    const current = model.requests[3]!.messages;
    expect(current.filter((message) => message.role === "tool")).toHaveLength(1);
    if (native) {
      const calling = current.filter((message) => message.parts.some((part) => part instanceof ToolCallPart));
      expect(calling).toHaveLength(1);
      expect(current.some((message) => message.role === "user" && message.text().includes("structured"))).toBe(true);
    }
    const stored = view(memory);
    expect(stored.items).toHaveLength(native ? 10 : 12);
    expect(stored.effects.map((effect) => effect.kind)).toEqual(["model", "tool", "model", "model", "model", "model"]);
  });
});
