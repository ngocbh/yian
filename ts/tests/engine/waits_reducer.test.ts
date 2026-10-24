/**
 * The run reducer's rules for inboxes, calls and waits: sends, `node_waiting`, `node_woken` and
 * `run_waiting`, `pending` and `tool_result` steps, `complete`, the take commit, orphan takes at a
 * resume, `inbox_closed`, `run_cancelled` and `run_failed`, and the refusals a host delivery
 * meets. TS-only: Python yian has no inboxes or waits.
 */
import { describe, expect, test } from "bun:test";

import { TypeCheckError } from "../../src/typesys/values.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";

describe("DeliveryRefused", () => {
  test("is a RangeError naming why, with the call's state and cause when given", () => {
    const cause = new TypeCheckError("expected str");
    const plain = new DeliveryRefused("unknown_inbox", "no inbox \"other\"");
    expect(plain).toBeInstanceOf(RangeError);
    expect([plain.name, plain.reason, plain.callState, plain.cause, plain.message]).toEqual([
      "DeliveryRefused",
      "unknown_inbox",
      null,
      undefined,
      "no inbox \"other\"",
    ]);
    const settled = new DeliveryRefused("call_settled", "settled", { callState: "answered", cause });
    expect([settled.reason, settled.callState, settled.cause]).toEqual(["call_settled", "answered", cause]);
  });
});
