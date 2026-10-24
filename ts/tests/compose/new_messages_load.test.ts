/**
 * TS-only: `${x.new_messages}`, an AGENT's second readable value, typed `list(Message)`: what
 * loads and where, and each load error.
 */

import { afterEach, expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import { refProducer } from "../../src/compose/edges/producers.ts";
import { forget } from "../../src/defaults.ts";
import { Binding } from "../../src/nodes/contracts.ts";

afterEach(() => forget());

test("refProducer gives the node for x.output and x.new_messages", () => {
  expect(refProducer("reply.new_messages")).toBe("reply");
  expect(refProducer("reply.new_messages.0.parts")).toBe("reply");
  expect(refProducer("reply.output")).toBe("reply");
  expect(refProducer("item.new_messages")).toBeNull();
  expect(refProducer("system.new_messages")).toBeNull();
});

test("rensInternal renames x.new_messages into the callsite's namespace", () => {
  expect(rensInternal(new Binding("${reply.new_messages :- []}"), "turn#3")).toEqual(
    new Binding("${turn#3/reply.new_messages :- []}"),
  );
  expect(rensInternal(new Binding("${reply.new_messages + more.output}"), "cs")).toEqual(
    new Binding("${cs/reply.new_messages + cs/more.output}"),
  );
});
