/**
 * An AGENT's steer inbox: the reserved `__steer_inbox` wiring, how a steer read becomes model
 * messages, the loop's reads before each model call, and replay of recorded steers. TS-only:
 * Python yian has no inboxes.
 */
import { expect, test } from "bun:test";

import { Binding } from "../../src/nodes/contracts.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { InboxRef } from "../../src/typesys/values.ts";

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
