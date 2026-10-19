/**
 * The AGENT node's helpers: mirrors `tests/engine/test_agent.py`.
 *
 * TS-only: `textOf` of a string, of a message with text, reasoning, tool-call and refusal parts
 * (only the text, joined with `"\n"`), and of a message without text; `ReplyInterrupted`'s message,
 * `reply`, `usage` (default `null`) and that it is an `Error`.
 *
 * The AgentNode construction rows (124, 129, 144, 152) and the `YIAN_MAX_TOOL_ITERATIONS` rows
 * (531, 538) come with the node.
 *
 * Later (T5-11b): the tool loop rows (167, 180, 221, 243, 267, 395, the cap rows 442–522, 638).
 * Later (T5-11a): 162, the structured-origin rows (299, 351, 371, 426) and the `provider.auth`
 * rows (599–695).
 *
 * Not ported: 549 (`test_agent_run_has_no_scratch_kwarg`, signature introspection).
 */

import { expect, test } from "bun:test";

import { TokenUsage } from "../../src/events.ts";
import { Message, ReasoningPart, RefusalPart, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { textOf } from "../../src/nodes/agent/text.ts";

test("textOf returns a string as it is", () => {
  expect(textOf("hello\nworld")).toBe("hello\nworld");
  expect(textOf("")).toBe("");
});

test("textOf keeps only a message's text parts, joined with a newline", () => {
  const message = new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking"),
      new TextPart("first"),
      new ToolCallPart({ id: "c1", name: "search", arguments: {} }),
      new RefusalPart("no"),
      new TextPart("second"),
    ],
  });
  expect(textOf(message)).toBe("first\nsecond");
});

test("textOf of a message without text is empty", () => {
  const message = new Message({
    role: "assistant",
    parts: [new ReasoningPart("thinking"), new ToolCallPart({ id: "c1", name: "search", arguments: {} })],
  });
  expect(textOf(message)).toBe("");
});

test("ReplyInterrupted carries the partial reply and the usage so far", () => {
  const reply = new Message({ role: "assistant", parts: [new TextPart("partial")] });
  const usage = new TokenUsage({ inputTokens: 3, outputTokens: 2, totalTokens: 5 });
  const error = new ReplyInterrupted(reply, usage);
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("ReplyInterrupted");
  expect(error.message).toBe("model reply interrupted by the host");
  expect(error.reply).toBe(reply);
  expect(error.usage).toBe(usage);
});

test("ReplyInterrupted's usage defaults to null", () => {
  const reply = new Message({ role: "assistant", parts: [] });
  expect(new ReplyInterrupted(reply).usage).toBeNull();
});
