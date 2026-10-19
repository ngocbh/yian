/**
 * TOOL `retries:` lands on the node: a port of `tests/compose/test_tool_retries_load.py`.
 *
 * Ported: `test_tool_node_retries_defaults_to_the_tool_retries_default`.
 *
 * Changed: the test also sets `YIAN_TOOL_RETRIES=3`, so the node is shown to read the default
 * rather than hard-code the built-in 0. `test_nonzero_retries_changes_the_flow_fingerprint` builds
 * its flows by hand and compares against the same flow with `retries: 0`, not a pinned digest.
 *
 * Later (T3-15, T3-22): `test_retries_loads_onto_the_node`,
 * `test_omitted_retries_uses_the_default` and `test_bad_retries_is_a_load_error_at_its_line`
 * need the loader.
 *
 * Later (T3-22): `test_zero_retries_keeps_the_flow_fingerprint` pins a loaded flow's fingerprint.
 */

import { expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { current, forget, reload } from "../../src/defaults.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";

/** Python's `FLOW` (a TOOL `place` called with `{symbol: "ACME"}`) with the TOOL's `retries`. */
function toolFlow(retries: number): CompiledFlow {
  const place = new ToolNode("place", { toolId: "place", retries });
  place.params = [new Parameter("symbol")];
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["place", place],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [new Edge(`place->${END_ID}`, "place", END_ID, { inputGroup: "result" })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["place", new Map([["symbol", "ACME"]])],
        [END_ID, new Map([["result", "${place.output}"]])],
      ]),
      flowId: "old_tool_crash",
    },
  );
}

test("tool node retries defaults to the tool retries default", () => {
  expect(new ToolNode("t", { toolId: "x" }).retries).toBe(current().toolRetries);
  expect(new ToolNode("t", { toolId: "x", retries: 4 }).retries).toBe(4);
  try {
    reload((name) => (name === "YIAN_TOOL_RETRIES" ? "3" : undefined));
    expect(new ToolNode("t", { toolId: "x" }).retries).toBe(3);
    expect(new ToolNode("t", { toolId: "x", retries: 4 }).retries).toBe(4);
  } finally {
    forget();
  }
});

test("nonzero retries changes the flow fingerprint", () => {
  expect(flowFingerprint(toolFlow(1))).not.toBe(flowFingerprint(toolFlow(0)));
});
