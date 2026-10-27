/** The `NodeKind` vocabulary of `nodes/base.ts`. */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";

test("start end call member values", () => {
  expect(NodeKind.START).toBe("start");
  expect(NodeKind.END).toBe("end");
  expect(NodeKind.CALL).toBe("call");
});

test("new members are distinct", () => {
  expect(new Set([NodeKind.START, NodeKind.END, NodeKind.CALL, NodeKind.MAP]).size).toBe(4);
  // The retired REF member stays absent; CALL and MAP are distinct current drivers.
  expect("REF" in NodeKind).toBe(false);
  expect("MAP" in NodeKind).toBe(true);
});
