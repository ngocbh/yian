/**
 * The AGENT memory adapter: mirrors `tests/engine/test_agent_session.py`.
 *
 * Ported: `test_compatibility_matches_factory_defaults_and_config_carriers`,
 * `test_message_codec_round_trip_preserves_exact_provider_fields` (4 rows),
 * `test_message_codec_rejects_non_json_provider_payloads` (4 rows),
 * `test_message_codec_rejects_unknown_or_corrupt_stored_format` (5 rows),
 * `test_message_codec_rejects_nested_data_loss_and_opaque_payload_changes`, the `request`,
 * `observe`, `history` and `view` stages of
 * `test_adapter_memory_faults_are_distinct_from_provider_failures`,
 * `test_failed_observation_does_not_replace_provider_or_host_error` (2 rows),
 * `test_admission_follows_render_and_recall_and_only_current_recall_is_reused` (through
 * `AgentNode.forward` with an async `openMemoryContext`, the model's build counted as `model`) and
 * `test_request_is_stored_before_provider_and_failed_effect_remains_truthful` (two `FlowEngine`
 * runs with distinct run ids over a `MemoryRuntime` backed by one `SessionMemory`).
 *
 * Changed: the expected `message_format` is TS's own `yian.message.v1`, not Python's
 * `yian.langchain-message.v1`, and the codec rows use `yian.message.v1` messages. A stored payload
 * is deep-frozen rather than a detached copy. A non-JSON provider payload is refused by the
 * `Message` constructor before encoding (`MessageFormatError`); JS has no non-string key, so a
 * symbol key stands in for `{1: ...}`. A corrupt stored message is a `RangeError` with Python's
 * text. The data-loss rows are what a TS message can lose: an unknown key `Message.fromJSON`
 * refuses, and a part `toJSON` writes back changed (a spied `Message.prototype.toJSON`, as Python
 * patches `message_to_dict`). A host stop is the aborted signal, its reason what the call throws,
 * rather than a `_HostStop`; an observation failure's note is on the error's own `notes` array.
 * The `history` stage's hand-made view has empty ledgers, so its unknown-format item is the fault
 * (a `RangeError`), not a missing field. The refused provider call throws an `Error`, so its
 * sanitized failure names `Error`; the engine row also checks the evaluator's close: the turn closes
 * `failed` with the node's error, and a clean close adds no cleanup note to it.
 *
 * TS-only: a record's `api_key` is not kept when the provider resolves from the defaults, a
 * passed `snapshot` is used, and no provider anywhere throws `RangeError`; each `decodeMessage`
 * error text; the memory boundary passes a `CapabilityFault` and the aborted signal's reason
 * through and wraps anything else, also an error other than the reason after an abort; a model
 * call is requested with only its new input and observed with its reply under one exchange id;
 * without a turn the stop is checked before the reply check; a cut reply is observed `interrupted`
 * with the partial reply; a reply the check rejects is not stored and is recorded `failed`, or
 * `interrupted` with the stop thrown when a stop is pending.
 *
 * Not ported: `test_message_codec_accepts_older_payloads_without_optional_defaults`,
 * `test_message_codec_accepts_nested_tool_call_discriminator_defaults` and
 * `test_message_codec_rejects_replaced_supplied_tool_call_discriminator`: LangChain's optional
 * defaults and tool-call discriminators; `yian.message.v1` has neither. The `encode_input` and
 * `encode_reply` stages of `test_adapter_memory_faults_are_distinct_from_provider_failures`: a TS
 * `Message` always encodes, and the `observe` stage covers a fault after the call.
 *
 * Later (T6-6, through runFlow): every other test of the file, which runs a flow with `run_flow` or resumes one.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Env } from "../../src/defaults.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import type { ChatModel, EventStream } from "../../src/llm/events.ts";
import {
  MediaPart,
  Message,
  MESSAGE_FORMAT,
  MessageFormatError,
  ReasoningPart,
  type Request,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import { ConnectionSnapshot } from "../../src/llm_clients/connection.ts";
import {
  ConversationItem,
  type ConversationTurn,
  type ConversationView,
  MemoryContext,
  MemorySection,
} from "../../src/memory/contracts.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import {
  compatibility,
  decodeMessage,
  encodeMessage,
  historyMessages,
  invokeModel,
  memoryBoundary,
} from "../../src/nodes/agent/memory.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type NodeCaps } from "../../src/nodes/base.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import { serving } from "./run_state_fixtures.ts";
import {
  fakeTurn,
  NEVER,
  rejected,
  request,
  runSessionFlow,
  SESSION_CONFIG,
  sessionFlow,
  snapshot,
  view,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

// Each test's environment: conftest's base (a private store in a fresh directory, the default
// model) with the default provider this test sets.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-agent-session-"));
  vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "OpenAI"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

const expected = { provider: "openai", message_format: MESSAGE_FORMAT };

test("compatibility matches factory defaults and config carriers", () => {
  expect(compatibility(null, { env })).toEqual(expected);
  expect(compatibility(new LLMConfig({}), { env })).toEqual(expected);
  expect(compatibility(new LLMConfig({ provider: "openai" }), { env })).toEqual(expected);
  expect(compatibility({ provider: "OpenAI", model: "other", api_key: "not-retained" }, { env })).toEqual(expected);
});

test("a record's null provider resolves from the defaults", () => {
  expect(compatibility({ provider: null, api_key: "k" }, { env })).toEqual(expected);
});

test("a passed snapshot is used", () => {
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
  const snapshot = new ConnectionSnapshot({
    defaults: { provider: "anthropic", model: "claude-test" },
    providerModels: { anthropic: "claude-test" },
    connections: {},
  });
  expect(compatibility(null, { snapshot, env })).toEqual({ provider: "anthropic", message_format: MESSAGE_FORMAT });
});

test("no provider anywhere throws RangeError", () => {
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
  expect(() => compatibility(null, { env })).toThrow(RangeError);
  expect(() => compatibility({ model: "m" }, { env })).toThrow(RangeError);
});

const codecMessages: [string, Message][] = [
  ["system", new Message({ role: "system", parts: [new TextPart("current rules")] })],
  [
    "user with media",
    new Message({
      role: "user",
      parts: [new TextPart("look"), new MediaPart({ mime: "image/png", data: "AA", name: "shot.png" })],
    }),
  ],
  [
    "assistant with reasoning and a tool call",
    new Message({
      role: "assistant",
      parts: [
        new ReasoningPart("needed by provider", { signature: "sig-1" }),
        new TextPart("thinking", { annotations: [] }),
        new ToolCallPart({
          id: "provider-call",
          name: "lookup",
          arguments: { value: 1 },
          providerPayload: { index: 0 },
        }),
      ],
      protocol: "openai_responses",
      providerPayload: { model_name: "test", usage: { tokens: 12, ratio: 0.5, big: 2n ** 70n } },
    }),
  ],
  [
    "tool result",
    new Message({
      role: "tool",
      parts: [new ToolResultPart({ callId: "provider-call", name: "lookup", content: "result", isError: true })],
    }),
  ],
];

for (const [label, message] of codecMessages) {
  test(`message codec round trip preserves exact provider fields: ${label}`, () => {
    const item = encodeMessage(message, "exchange");
    expect(item.format).toBe(MESSAGE_FORMAT);
    expect(item.itemId).toMatch(/^[0-9a-f]{32}$/);
    expect(item.payload["exchange_id"]).toBe("exchange");
    expect(decodeMessage(item).toJSON()).toEqual(message.toJSON());
    expect(encodeMessage(message, "exchange").itemId).not.toBe(item.itemId);
    // The stored payload cannot be changed afterwards, at any depth.
    const wire = item.payload["message"] as JsonObject;
    expect(Object.isFrozen(wire)).toBe(true);
    expect(Object.isFrozen(wire["parts"])).toBe(true);
    expect(Object.isFrozen((wire["parts"] as JsonObject[])[0])).toBe(true);
  });
}

const badPayloads: [string, unknown][] = [
  ["NaN", Number.NaN],
  ["infinity", Number.POSITIVE_INFINITY],
  ["an object", new Date(0)],
  ["a symbol key", { [Symbol("key")]: "not a JSON key" }],
];

for (const [label, bad] of badPayloads) {
  test(`a Message refuses non-JSON provider payloads before encoding: ${label}`, () => {
    expect(
      () => new Message({ role: "assistant", parts: [new TextPart("reply")], providerPayload: { bad } as never }),
    ).toThrow(MessageFormatError);
  });
}

/** Return a stored item of `format` with `payload`. */
function stored(payload: unknown, format: string = MESSAGE_FORMAT): ConversationItem {
  return new ConversationItem("stored-item", format, payload);
}

const input = new Message({ role: "user", parts: [new TextPart("input")] }).toJSON();

test("message codec rejects unknown or corrupt stored format", () => {
  expect(() => decodeMessage(stored({ exchange_id: "exchange", message: input }, "unknown-v9"))).toThrow(
    new RangeError("unsupported conversation message format; reset is required"),
  );
  expect(() => decodeMessage(stored({ exchange_id: false, message: input }))).toThrow(
    new RangeError("conversation exchange_id must be a nonempty string"),
  );
  const { parts: _, ...missing } = input;
  for (const message of [missing, { ...input, role: "not-a-role" }, { ...input, protocol: 7 }]) {
    let error: unknown = null;
    try {
      decodeMessage(stored({ exchange_id: "exchange", message }));
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe("invalid stored conversation message");
    expect((error as Error).cause).toBeInstanceOf(MessageFormatError);
  }
});

test("each decodeMessage error has Python's text", () => {
  expect(() => decodeMessage({ itemId: "i", format: MESSAGE_FORMAT, payload: {} } as never)).toThrow(
    new TypeError("stored messages must be exact ConversationItem values"),
  );
  for (const payload of [{ exchange_id: "exchange" }, { exchange_id: "exchange", message: input, extra: 1 }]) {
    expect(() => decodeMessage(stored(payload))).toThrow(
      new RangeError("conversation message fields do not match the supported format"),
    );
  }
  expect(() => decodeMessage(stored({ exchange_id: "", message: input }))).toThrow(
    new RangeError("conversation exchange_id must be a nonempty string"),
  );
  expect(() => encodeMessage(new Message({ role: "user", parts: [] }), "")).toThrow(
    new RangeError("conversation exchange_id must be a nonempty string"),
  );
});

test("message codec rejects nested data loss and opaque payload changes", () => {
  const wire = new Message({
    role: "assistant",
    parts: [
      new TextPart("reply"),
      new ReasoningPart("reasoning"),
      new ToolCallPart({ id: "call-1", name: "lookup", arguments: { id: 1 } }),
    ],
  }).toJSON();
  const unknownKey = { ...wire, parts: [{ type: "text", text: "reply", extra: "not in the format" }] };
  expect(() => decodeMessage(stored({ exchange_id: "exchange", message: unknownKey }))).toThrow(
    "invalid stored conversation message",
  );

  const item = stored({ exchange_id: "exchange", message: wire });
  const real = Message.prototype.toJSON;
  const part = (data: JsonObject, index: number) => (data["parts"] as JsonObject[])[index]!;
  const args = (data: JsonObject) => part(data, 2)["arguments"] as JsonObject;
  const changes: ((data: JsonObject) => void)[] = [
    (data) => (part(data, 2)["name"] = "different"),
    (data) => delete part(data, 2)["id"],
    (data) => (args(data)["id"] = true),
    (data) => (args(data)["extra"] = "not in the message"),
    (data) => (part(data, 0)["extra"] = "not in the message"),
  ];
  for (const change of changes) {
    const spy = spyOn(Message.prototype, "toJSON").mockImplementation(function (this: Message) {
      const data = real.call(this);
      change(data);
      return data;
    });
    let error: unknown = null;
    try {
      decodeMessage(item);
    } catch (caught) {
      error = caught;
    } finally {
      spy.mockRestore();
    }
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe("invalid stored conversation message");
    // The round-trip check refused it, not a change that throws on its own.
    const cause = (error as Error).cause;
    expect(cause).toBeInstanceOf(RangeError);
    expect((cause as Error).message).toBe("stored message would be changed by decoding");
  }
  expect(item.payload["message"]).toEqual(wire);
});

test("the memory boundary passes a fault and the stop through and wraps anything else", async () => {
  const fault = new CapabilityFault(new Error("backend"));
  const thrower = (error: unknown) => () => {
    throw error;
  };
  expect(await memoryBoundary(() => 1)).toBe(1);
  expect(await memoryBoundary(async () => 2)).toBe(2);
  await expect(memoryBoundary(thrower(fault))).rejects.toBe(fault);

  const ordinary = new Error("codec");
  const wrapped = await memoryBoundary(thrower(ordinary)).catch((error: unknown) => error);
  expect(wrapped).toBeInstanceOf(CapabilityFault);
  expect((wrapped as CapabilityFault).original).toBe(ordinary);

  const controller = new AbortController();
  const stop = new Error("stopped");
  controller.abort(stop);
  await expect(memoryBoundary(async () => Promise.reject(stop), controller.signal)).rejects.toBe(stop);
  // After an abort, only the signal's own reason is the stop.
  const late = await memoryBoundary(thrower(ordinary), controller.signal).catch((error: unknown) => error);
  expect(late).toBeInstanceOf(CapabilityFault);
  expect((late as CapabilityFault).original).toBe(ordinary);
  // The reason before an abort is an ordinary error.
  const early = await memoryBoundary(thrower(stop), new AbortController().signal).catch((error: unknown) => error);
  expect((early as CapabilityFault).original).toBe(stop);
});

const reply = (text: string): Response =>
  new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }) });
const newInput = new Message({ role: "user", parts: [new TextPart("new")] });

for (const stage of ["request", "observe"] as const) {
  test(`adapter memory faults are distinct from provider failures: ${stage}`, async () => {
    const calls: string[] = [];
    const expected = new RangeError("memory refused write");
    const turn = fakeTurn(
      () => {
        if (stage === "request") throw expected;
        return 1;
      },
      () => {
        if (stage === "observe") throw expected;
      },
    );
    const invoke = async () => {
      calls.push("provider");
      return reply("reply");
    };
    const error = await rejected(invokeModel(turn, invoke, { newMessages: [newInput], signal: NEVER }));
    expect(error).toBeInstanceOf(CapabilityFault);
    expect((error as CapabilityFault).original).toBe(expected);
    expect(calls).toHaveLength(stage === "observe" ? 1 : 0);
  });
}

test("adapter memory faults are distinct from provider failures: history", async () => {
  const view = { items: [new ConversationItem("old", "unknown", {})], effects: [], applied: [], closedInvocations: [] };
  const error = await rejected(historyMessages({ view } as unknown as ConversationTurn));
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBeInstanceOf(RangeError);
});

test("adapter memory faults are distinct from provider failures: view", async () => {
  const expected = new RangeError("memory refused write");
  const broken = {
    get view(): never {
      throw expected;
    },
  };
  const error = await rejected(historyMessages(broken as unknown as ConversationTurn));
  expect(error).toBeInstanceOf(CapabilityFault);
  expect((error as CapabilityFault).original).toBe(expected);
});

for (const hostStop of [false, true]) {
  test(`failed observation does not replace provider or host error: host stop ${hostStop}`, async () => {
    const controller = new AbortController();
    const primary = new Error(hostStop ? "provider cancelled" : "provider failed");
    const seen: string[] = [];
    const invoke = async (): Promise<Response> => {
      // The host stop is the node's aborted signal, its reason what the call throws.
      if (hostStop) controller.abort(primary);
      throw primary;
    };
    const turn = fakeTurn(
      () => 1,
      (_ordinal, status) => {
        seen.push(status);
        throw new Error("cannot save failure");
      },
    );
    const error = await rejected(invokeModel(turn, invoke, { newMessages: [newInput], signal: controller.signal }));
    if (hostStop) {
      expect(error).toBe(primary);
    } else {
      expect(error).toBeInstanceOf(CapabilityFault);
      expect((error as CapabilityFault).original).toBe(primary);
    }
    expect(seen).toEqual([hostStop ? "interrupted" : "failed"]);
    expect((primary as Error & { notes?: string[] }).notes).toEqual([
      "Conversation observation failed: Error: cannot save failure",
    ]);
  });
}

test("a model call is requested with its new input and observed with its reply; chat by default", async () => {
  await withHarness(async ({ session }) => {
    await withTurn(session, request(), async (turn) => {
      const response = reply("answer");
      const returned = await invokeModel(turn, async () => response, {
        newMessages: [newInput],
        signal: NEVER,
        exchangeId: "exchange",
        purpose: "structured_native",
      });
      expect(returned).toBe(response);
      const [effect] = turn.view.effects;
      expect(effect!.kind).toBe("model");
      expect(effect!.request).toEqual({ exchange_id: "exchange", purpose: "structured_native" });
      expect(effect!.status).toBe("completed");
      expect(turn.view.items.map(decodeMessage)).toEqual([newInput, response.message]);
      expect(turn.view.items.map((stored) => stored.payload["exchange_id"])).toEqual(["exchange", "exchange"]);
      // By default the purpose is `chat` and the exchange id fresh.
      await invokeModel(turn, async () => response, { newMessages: [], signal: NEVER });
      const defaults = turn.view.effects[1]!.request;
      expect(defaults["purpose"]).toBe("chat");
      expect(defaults["exchange_id"]).toMatch(/^[0-9a-f]{32}$/);
    });
  });
});

test("a model call without a turn is invoked, stop checked, then reply checked", async () => {
  const events: string[] = [];
  const response = reply("answer");
  const returned = await invokeModel(null, async () => response, {
    newMessages: [newInput],
    checkReply: (checked) => {
      events.push("check");
      expect(checked).toBe(response);
    },
  });
  expect(returned).toBe(response);
  expect(events).toEqual(["check"]);
  const controller = new AbortController();
  const stop = new Error("stopped");
  const stopped = invokeModel(
    null,
    async () => {
      controller.abort(stop);
      return response;
    },
    { newMessages: [], signal: controller.signal, checkReply: () => events.push("late check") },
  );
  expect(await rejected(stopped)).toBe(stop);
  expect(events).toEqual(["check"]);
});

test("a cut reply is observed interrupted with the partial reply, and ReplyInterrupted rethrown", async () => {
  await withHarness(async ({ session }) => {
    const partial = new Message({ role: "assistant", parts: [new TextPart("partial")] });
    const cut = new ReplyInterrupted(partial);
    let view: ConversationView | null = null;
    const error = await rejected(
      withTurn(session, request(), async (turn) => {
        try {
          return await invokeModel(turn, () => Promise.reject(cut), { newMessages: [newInput], signal: NEVER });
        } finally {
          view = turn.view;
        }
      }),
    );
    expect(error).toBe(cut);
    expect(view!.effects[0]!.status).toBe("interrupted");
    expect(decodeMessage(view!.items.at(-1)!)).toEqual(partial);
  });
});

for (const pending of [false, true]) {
  test(`a rejected reply is not stored and is recorded ${pending ? "interrupted under a stop" : "failed"}`, async () => {
    await withHarness(async ({ session }) => {
      const controller = new AbortController();
      const stop = new Error("stopped");
      const refusal = new TypeError("invalid tool arguments");
      let seen: { status: string; failure: string | null; items: number } | null = null;
      const error = await rejected(
        withTurn(
          session,
          request(),
          async (turn) => {
            try {
              return await invokeModel(turn, async () => reply("bad"), {
                newMessages: [newInput],
                signal: controller.signal,
                checkReply: () => {
                  if (pending) controller.abort(stop);
                  throw refusal;
                },
              });
            } finally {
              const [effect] = turn.view.effects;
              seen = { status: effect!.status, failure: effect!.failure?.message ?? null, items: turn.view.items.length };
            }
          },
          controller.signal,
        ),
      );
      expect(error).toBe(pending ? stop : refusal);
      expect(seen!).toEqual(
        pending
          ? { status: "interrupted", failure: "Model invocation interrupted (Error).", items: 1 }
          : { status: "failed", failure: "Model invocation failed (TypeError).", items: 1 },
      );
    });
  });
}


/** A `ScriptedModel` that calls `probe` with the call's index before answering it. */
class ProbedModel extends ScriptedModel {
  readonly probe: (index: number) => void;

  constructor(replies: ScriptedReply[], probe: (index: number) => void) {
    super(replies, SESSION_CONFIG);
    this.probe = probe;
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    this.probe(this.requests.length);
    return super.stream(request, options);
  }
}

/** The texts of the messages of `model`'s last request after its system prompt. */
function lastTexts(model: ScriptedModel): string[] {
  return model.requests.at(-1)!.messages.slice(1).map((message) => message.text());
}

test("admission follows render and recall, and only the current recall is reused", async () => {
  const memory = new SessionMemory();
  const model = new ScriptedModel([reply("one"), reply("two")], SESSION_CONFIG);
  const events: string[] = [];
  const models = {
    env: () => undefined,
    createModel: () => {
      events.push("model");
      return model;
    },
  };
  const node = new AgentNode("agent", { prompt: "question ${value}", llmConfig: SESSION_CONFIG });
  for (const [index, note] of ["old recall", "current recall"].entries()) {
    await withHarness(async ({ session }) => {
      let turn = null as ConversationTurn | null;
      const openMemoryContext: NodeCaps["openMemoryContext"] = async (_system, prompt, compatibility) => {
        expect(prompt).toBe(`question ${index}`);
        events.push("recall");
        const sections = [new MemorySection("session", note)];
        turn = await session.openTurn(request(String(index), 1, ["root", "agent"], compatibility), { signal: NEVER });
        events.push("admit");
        return new MemoryContext({ sections, transcript: turn });
      };
      try {
        await node.forward({ value: BigInt(index) }, {
          openMemoryContext,
          captureRender: () => void events.push("render"),
          models,
        });
      } finally {
        turn?.close();
      }
    }, memory);
  }
  expect(events).toEqual(["render", "recall", "admit", "model", "render", "recall", "admit", "model"]);
  const last = model.requests.at(-1)!.messages;
  expect(last[0]!.text()).toContain("current recall");
  expect(last.every((message) => !message.text().includes("old recall"))).toBe(true);
  expect(lastTexts(model)).toEqual(["question 0", "one", "question 1"]);
});

/** Run `sessionFlow` as run `runId` with `question`, `memory` backing `session` and `model` serving it. */
function runSession(memory: SessionMemory, model: ChatModel, runId: string, question: string) {
  return runSessionFlow(sessionFlow(), memory, { runId, inputs: { question }, models: serving(model) });
}

test("the request is stored before the provider, and a failed effect remains truthful", async () => {
  const error = new Error("provider refused; secret-marker");
  const memory = new SessionMemory();
  // Captured, not checked, in the probe: a failed check there would become the provider error.
  let atCall = null as ConversationView | null;
  const model = new ProbedModel([error, reply("later")], (index) => {
    if (index === 0) atCall = view(memory);
  });
  const failed = await runSession(memory, model, "first-run", "first");
  expect(atCall!.effects.map((effect) => effect.status)).toEqual(["pending"]);
  expect(atCall!.items).toHaveLength(2);
  expect(failed).toBeInstanceOf(RunFailed);
  const failure = (failed as RunFailed).failure;
  expect(failure.code).toBe("node.execute");
  // The evaluator closed the turn with the node's error, and the clean close left no note.
  expect(failure.traceback).not.toContain("cleanup");
  expect((error as Error & { notes?: string[] }).notes).toBeUndefined();
  const stored = view(memory);
  expect(stored.effects[0]!.status).toBe("failed");
  expect(stored.closedInvocations.at(-1)!.status).toBe("failed");
  expect(stored.effects[0]!.failure!.message).toBe("Model invocation failed (Error).");
  expect(logJson(snapshot(memory))).not.toContain("secret-marker");
  const later = await runSession(memory, model, "second-run", "second");
  expect(later).toEqual(new RunSucceeded({ output: "later" }));
  expect(lastTexts(model)).toEqual([
    "first",
    'Session history status (context only): [{"invocation":"failed"},{"effect":"model","status":"failed"}]',
    "second",
  ]);
});
