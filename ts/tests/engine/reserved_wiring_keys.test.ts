/**
 * `Node.reservedWiringKeys()`, the static author-wiring keys a node owns beyond its declared
 * `params` (MAP `over`, WAIT `until`): a port of `tests/engine/test_reserved_wiring_keys.py`.
 *
 * Ported (adapted): `test_reserved_wiring_keys_by_kind`, its MAP and WAIT rows.
 *
 * Changed: the key sets are compared with `Set`s.
 *
 * Later (T3-8, CodeNode): the CODE row of `test_reserved_wiring_keys_by_kind`.
 *
 * Later (T3-9): the AGENT rows of `test_reserved_wiring_keys_by_kind`.
 */

import { expect, test } from "bun:test";

import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";

test("reserved wiring keys by kind", () => {
  expect(new MapNode("m", { flowId: "c", child: null, childInputs: [] }).reservedWiringKeys()).toEqual(
    new Set(["over"]),
  );
  expect(new WaitNode("w").reservedWiringKeys()).toEqual(new Set(["until"]));
});
