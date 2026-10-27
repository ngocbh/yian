/**
 * The `Route` outcome of `nodes/base.ts`, a port of `tests/engine/test_route.py`: a router
 * (CASE) returns `Route(handle)`, with no value.
 *
 * Changed: Python's keyword form `Route(handle=...)` is the same positional constructor.
 *
 * Not ported: `test_route_is_in_node_result_union` (the union is a type);
 * `test_node_routed_event_fields` is in `events.test.ts`.
 */

import { expect, test } from "bun:test";

import { Route } from "../../src/nodes/base.ts";

test("route carries handle", () => {
  expect(new Route("x").handle).toBe("x");
  expect(new Route("case_a").handle).toBe("case_a");
});
