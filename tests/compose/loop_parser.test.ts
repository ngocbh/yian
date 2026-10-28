/**
 * Parsing `kind: loop` into a `LoopDescriptor`, through `parseFile`, `flowNodesCursor` and
 * `parseNodes` with the registered kinds.
 */

import { expect, test } from "bun:test";

import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { nodesOf } from "../compose/declarations.ts";

const LOOP_YAML = `
id: chat
name: chat
input:
  history: !list
output:
  messages: list
nodes:
  chat_loop:
    kind: loop
    call: chat_turn
    input:
      messages: \${input.history}
      exited: false
    while: not exited
    max: 1000
`;

/** Replace `old` in a fixture, failing if it is absent (a stale needle). */
function swap(src: string, old: string, replacement: string): string {
  expect(src).toContain(old);
  return src.replace(old, replacement);
}

/** Parse flow `text`'s nodes and return node `nodeId` as a `LoopDescriptor`. */
function loopNode(text: string, nodeId: string): LoopDescriptor {
  const desc = nodesOf(text).get(nodeId);
  expect(desc).toBeInstanceOf(LoopDescriptor);
  return desc as LoopDescriptor;
}

test("the parser reads a loop node", () => {
  const desc = loopNode(LOOP_YAML, "chat_loop");

  expect(desc.call).toBe("chat_turn");
  expect(desc.while_).toBe("not exited");
  expect(desc.max).toBe(1000n);
  expect(desc.inputs["exited"]).toBe(false);
});

const UNTIL_TIMES_YAML = `
id: p
name: p
input:
  n: !int
output:
  n: int
nodes:
  L:
    kind: loop
    call: body
    input:
      n: 3
    until: n <= 0
    max: 5
`;

test("the parser accepts the until and times keys", () => {
  expect(loopNode(UNTIL_TIMES_YAML, "L").until_).toBe("n <= 0");
  // `times:` is also a loop key; the build, not the parse, enforces exactly one driver.
  expect(loopNode(swap(UNTIL_TIMES_YAML, "until: n <= 0", "times: 4"), "L").times).toBe(4n);
});
