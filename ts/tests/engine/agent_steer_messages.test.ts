/**
 * An AGENT's steer inbox: the reserved `__steer_inbox` wiring, how a steer read becomes model
 * messages, the loop's reads before each model call, and replay of recorded steers. TS-only:
 * Python yian has no inboxes.
 */
import { expect, test } from "bun:test";

import { type Message as ModelMessage, TextPart } from "../../src/llm/messages.ts";
import { steerMessages } from "../../src/nodes/agent/messages.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FileRef, InboxRef } from "../../src/typesys/values.ts";

/** A pool whose START holds the handle of inbox `name`, as START binds an Inbox input. */
function poolWithInbox(name: string): VariablePool {
  const pool = new VariablePool("__start__");
  pool.set("__start__", { [name]: new InboxRef(name) });
  return pool;
}

test("an agent with a steer inbox reserves its key, hides it from the model and reads its inbox", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  expect(node.readsSteerInbox).toBe(true);
  expect([...node.reservedWiringKeys()]).toEqual([STEER_INBOX_KEY]);
  expect(node.reservedKeyLabel(STEER_INBOX_KEY)).toBe("steer_inbox");
  expect(node.modelHiddenInputs().has(STEER_INBOX_KEY)).toBe(true);
  expect(node.backwardInputNames()).not.toContain(STEER_INBOX_KEY);
  expect(new AgentNode("plain", { prompt: "hi" }).readsSteerInbox).toBe(false);
});

test("bindReserved resolves the steer inbox binding to the inbox name", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  const wiring = new Map([[STEER_INBOX_KEY, new Binding("${input.m}")]]);
  expect(node.bindReserved(wiring, poolWithInbox("m"))).toEqual({ [STEER_INBOX_KEY]: "m" });
});

test("bindReserved refuses a steer inbox binding that isn't an inbox", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  const pool = new VariablePool("__start__");
  pool.set("__start__", { m: "text" });
  const wiring = new Map([[STEER_INBOX_KEY, new Binding("${input.m}")]]);
  expect(() => node.bindReserved(wiring, pool)).toThrow("AGENT node 'chat': steer_inbox must be an inbox");
});

/** Each message of `messages` as `[role, author, texts]`. */
const shape = (messages: readonly ModelMessage[]) =>
  messages.map((message) => [message.role, message.author, message.parts.map((part) => (part as TextPart).text)]);

test("one read's str and record events share one user message, a text part each", () => {
  expect(shape(steerMessages(["use dataset B", { year: 2024n }]))).toEqual([
    ["user", null, ["use dataset B", '{"year":2024}']],
  ]);
});

test("a Message event is its own user message, author and parts kept", () => {
  const message = { role: "user", author: "ana", parts: [{ type: "text", text: "hi" }, { type: "text", text: "there" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", "ana", ["hi", "there"]]]);
});

test("Message events and other events keep their order", () => {
  const message = { role: "user", parts: [{ type: "text", text: "m" }] };
  expect(shape(steerMessages(["a", message, "b", "c"]))).toEqual([
    ["user", null, ["a"]],
    ["user", null, ["m"]],
    ["user", null, ["b", "c"]],
  ]);
});

test("two Message events in one read are two user messages, in send order, authors kept", () => {
  const from = (author: string, text: string) => ({ role: "user", author, parts: [{ type: "text", text }] });
  expect(shape(steerMessages([from("ana", "one"), from("bo", "two")]))).toEqual([
    ["user", "ana", ["one"]],
    ["user", "bo", ["two"]],
  ]);
});

test("a Message steer with another role becomes a user message, author kept", () => {
  const message = { role: "assistant", author: "ana", parts: [{ type: "text", text: "x" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", "ana", ["x"]]]);
});

test("a tool call part in a steer Message becomes text; nothing throws", () => {
  const message = {
    role: "assistant",
    parts: [
      { type: "reasoning", text: "thinking" },
      { type: "tool_call", id: "c1", name: "lookup", arguments: { q: "x" } },
    ],
  };
  expect(shape(steerMessages([message]))).toEqual([
    ["user", null, ["thinking", '{"type":"tool_call","id":"c1","name":"lookup","arguments":{"q":"x"}}']],
  ]);
});

test("a Message steer's media is dropped and reported, never a failure", () => {
  const media = { role: "user", parts: [{ type: "media", mime: "image/png", file: new FileRef({ uri: "files/a.png", mime: "image/png", name: null }) }] };
  const drops: [number, number][] = [];
  const messages = steerMessages(["first", media], (event, part) => drops.push([event, part]));
  expect(drops).toEqual([[1, 0]]);
  expect(messages).toHaveLength(2);
  expect(messages[1]!.parts).toHaveLength(1);
  expect((messages[1]!.parts[0] as TextPart).text).toContain("image/png");
});

test("an empty author is left out", () => {
  const message = { role: "user", author: "", parts: [{ type: "text", text: "x" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", null, ["x"]]]);
});
