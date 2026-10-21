/**
 * The AGENT memory adapter: mirrors `tests/engine/test_agent_session.py`.
 *
 * Ported: `test_compatibility_matches_factory_defaults_and_config_carriers`,
 * `test_message_codec_round_trip_preserves_exact_provider_fields` (4 rows),
 * `test_message_codec_rejects_non_json_provider_payloads` (4 rows),
 * `test_message_codec_rejects_unknown_or_corrupt_stored_format` (5 rows) and
 * `test_message_codec_rejects_nested_data_loss_and_opaque_payload_changes`.
 *
 * Changed: the expected `message_format` is TS's own `yian.message.v1`, not Python's
 * `yian.langchain-message.v1`, and the codec rows use `yian.message.v1` messages. A stored payload
 * is deep-frozen rather than a detached copy. A non-JSON provider payload is refused by the
 * `Message` constructor before encoding (`MessageFormatError`); JS has no non-string key, so a
 * symbol key stands in for `{1: ...}`. A corrupt stored message is a `RangeError` with Python's
 * text. The data-loss rows are what a TS message can lose: an unknown key `Message.fromJSON`
 * refuses, and a part `toJSON` writes back changed (a spied `Message.prototype.toJSON`, as Python
 * patches `message_to_dict`).
 *
 * TS-only: a record's `api_key` is not kept when the provider resolves from the defaults, a
 * passed `snapshot` is used, and no provider anywhere throws `RangeError`; each `decodeMessage`
 * error text; the memory boundary passes a `CapabilityFault` and the aborted signal's reason
 * through and wraps anything else, also an error other than the reason after an abort.
 *
 * Not ported: `test_message_codec_accepts_older_payloads_without_optional_defaults`,
 * `test_message_codec_accepts_nested_tool_call_discriminator_defaults` and
 * `test_message_codec_rejects_replaced_supplied_tool_call_discriminator`: LangChain's optional
 * defaults and tool-call discriminators; `yian.message.v1` has neither.
 *
 * Later (T6-6): every other test of the file, which needs the rest of the adapter, session memory
 * and `run_flow`.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Env } from "../../src/defaults.ts";
import {
  MediaPart,
  Message,
  MESSAGE_FORMAT,
  MessageFormatError,
  ReasoningPart,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import { ConnectionSnapshot } from "../../src/llm_clients/connection.ts";
import { ConversationItem } from "../../src/memory/contracts.ts";
import { compatibility, decodeMessage, encodeMessage, memoryBoundary } from "../../src/nodes/agent/memory.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

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
