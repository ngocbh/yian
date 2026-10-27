/**
 * `Node.reservedWiringKeys()`, the static author-wiring keys a node owns beyond its declared
 * `params` (MAP `over`): a port of `tests/engine/test_reserved_wiring_keys.py`.
 *
 * Ported (adapted): `test_reserved_wiring_keys_by_kind`.
 *
 * Changed: the key sets are compared with `Set`s; the CODE row's reference is `./mod.ts:fn`. The
 * WAIT `until` row is gone: TS yian has no WAIT kind (`sleep` is a TOOL built-in).
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";

test("reserved wiring keys by kind", () => {
  expect(new MapNode("m", { flowId: "c", child: null, childInputs: [] }).reservedWiringKeys()).toEqual(
    new Set(["over"]),
  );
  // A plain leaf reserves nothing.
  expect(new CodeNode("c", { code: "./mod.ts:fn" }).reservedWiringKeys()).toEqual(new Set());
  // An AGENT reserves `__llm_config` only once Compose wires values in its layers.
  const agent = new AgentNode("a", { prompt: "p" });
  expect(agent.reservedWiringKeys()).toEqual(new Set());
  agent.wiresLlmConfig = true;
  expect(agent.reservedWiringKeys()).toEqual(new Set(["__llm_config"]));
});
