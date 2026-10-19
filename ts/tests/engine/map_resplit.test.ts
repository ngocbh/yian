/**
 * CALL and MAP are distinct typed drivers, a port of `tests/engine/test_map_resplit.py`: the
 * retired ref kind stays absent from `NodeKind`.
 *
 * Changed: `NodeKind` is a const object of strings, so a member is its value and has no
 * `.value`; `hasattr` is the `in` operator.
 *
 * Later (T3-8): `test_retired_ref_package_is_absent` and `test_call_node_excludes_map_fields`.
 */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";

test("retired ref kind is absent and call map are distinct", () => {
  expect("REF" in NodeKind).toBe(false);
  expect(NodeKind.CALL).toBe("call");
  expect(NodeKind.MAP).toBe("map"); // MAP re-split into its own kind
  expect(NodeKind.CALL).not.toBe(NodeKind.MAP);
});
