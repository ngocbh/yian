/**
 * The `Route` outcome of `nodes/base.ts`: a router (CASE) returns `Route(handle)`, with no value.
 */

import { expect, test } from "bun:test";

import { Route } from "../../src/nodes/base.ts";

test("route carries handle", () => {
  expect(new Route("x").handle).toBe("x");
  expect(new Route("case_a").handle).toBe("case_a");
});
