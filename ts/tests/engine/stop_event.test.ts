/**
 * The stop at the raw node boundary: a port of `tests/engine/test_stop_event.py`.
 *
 * TS has no stop event and no `_HostStop`: a stop is the node's aborted signal, which every node
 * receives as `caps.signal`.
 *
 * Changed: `test_stop_cap_is_exact_event_and_only_reaches_opted_in_node` is "every node gets the
 * given signal as `caps.signal`" (there is no opt-in trait);
 * `test_direct_opted_in_eval_gets_private_unset_event_when_host_omits_one` is "`evalNode` without
 * a signal gives an un-aborted one"; `test_preset_stop_prevents_node_execution_and_propagates_host_control`
 * is "an aborted signal rejects with its reason and `forward` never runs" (the `primary` flag has
 * no counterpart); `test_node_host_stop_bypasses_evaluation_failure_with_identity` is "a node
 * rejecting after its signal aborted rejects `evalNode` with the reason, not a failure".
 *
 * Not ported: `test_public_run_flow_signature_has_no_stop_event`: TS reads no signatures, and
 * `runFlow` takes `{ signal }`.
 */

import { expect, test } from "bun:test";

import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { rejected } from "./run_state_fixtures.ts";

/** Record the capabilities supplied to one direct evaluation. */
class CapNode extends Node {
  readonly kind = NodeKind.CODE;
  caps: NodeCaps = {};

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.caps = caps;
    return new Output("done");
  }
}

/** Expose whether an aborted signal allowed the node body to start. */
class NeverRunsNode extends Node {
  readonly kind = NodeKind.CODE;
  ran = false;

  override forward(): NodeResult {
    this.ran = true;
    return new Output("unexpected");
  }
}

test("every node gets the given signal as caps.signal", async () => {
  const signal = new AbortController().signal;
  const first = new CapNode("first");
  const second = new CapNode("second");

  expect((await evalNode(first, null, new VariablePool(), { signal })).outcome).toEqual(new Output("done"));
  expect((await evalNode(second, null, new VariablePool(), { signal })).outcome).toEqual(new Output("done"));

  expect(first.caps).toEqual({ signal });
  expect(first.caps.signal).toBe(signal);
  expect(second.caps.signal).toBe(signal);
});

test("evalNode without a signal gives an un-aborted one", async () => {
  const node = new CapNode("node");

  expect((await evalNode(node, null, new VariablePool())).outcome).toEqual(new Output("done"));

  expect(Object.keys(node.caps)).toEqual(["signal"]);
  expect(node.caps.signal).toBeInstanceOf(AbortSignal);
  expect(node.caps.signal!.aborted).toBe(false);
});

test("an aborted signal rejects with its reason and forward never runs", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped");
  controller.abort(reason);
  const node = new NeverRunsNode("never");

  expect(await rejected(evalNode(node, null, new VariablePool(), { signal: controller.signal }))).toBe(reason);
  expect(node.ran).toBe(false);
});

test("a node rejecting after its signal aborted rejects evalNode with the reason, not a failure", async () => {
  const controller = new AbortController();
  const reason = new Error("terminal EOF");

  class AbortsThenThrows extends Node {
    readonly kind = NodeKind.CODE;

    override forward(): NodeResult {
      controller.abort(reason);
      throw new Error("interrupted");
    }
  }

  expect(
    await rejected(evalNode(new AbortsThenThrows("work"), null, new VariablePool(), { signal: controller.signal })),
  ).toBe(reason);
});
