/**
 * TS-only: `${x.new_messages}`, an AGENT's second readable value, typed `list(Message)`: what
 * loads and where, and each load error.
 */

import { afterEach, expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import { classifyPath } from "../../src/compile/validation.ts";
import { refProducer } from "../../src/compose/edges/producers.ts";
import { referenceContract } from "../../src/compose/helpers/output_contracts.ts";
import { forget } from "../../src/defaults.ts";
import { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { messagesKey } from "../../src/typesys/pool.ts";
import { newMessagesType, typeFor } from "../../src/typesys/types.ts";

/** The load error for reading `node`'s new messages when it isn't an AGENT, or not here. */
const NOT_AN_AGENT = (node: string) =>
  `reference \${${node}.new_messages} reads new_messages, which only an AGENT node has, from a node input, ` +
  `flow output or flow assert; node '${node}' isn't one there`;

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

test("classifyPath reads x.new_messages only for a messages target", () => {
  const targets = new Set(["reply", messagesKey("reply"), "code"]);
  expect(classifyPath("reply.new_messages", targets, new Set())).toBeNull();
  expect(classifyPath("code.new_messages", targets, new Set())).toBe(NOT_AN_AGENT("code"));
  expect(classifyPath("code.nope", targets, new Set())).toContain("must be code.output[.field]");
});

test("referenceContract reads x.new_messages as the producers entry under x.new_messages", () => {
  const messages = new ValueContract(newMessagesType());
  const producers = new Map([
    ["reply", new ValueContract(typeFor("str", new Map()))],
    [messagesKey("reply"), messages],
  ]);
  expect(referenceContract("reply.new_messages", producers, new Map())).toBe(messages);
  expect(referenceContract("code.new_messages", producers, new Map())).toBeNull();
});
