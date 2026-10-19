/**
 * Deterministic graph-expansion ids, a port of `tests/engine/test_expand_ids.py`: `ns` namespaces
 * a child id under its callsite and `mapCallsite` mints the per-element callsite of a MAP.
 */

import { expect, test } from "bun:test";

import { mapCallsite, ns } from "../../src/nodes/ids.ts";

test("ns joins the callsite and the child id", () => {
  expect(ns("each", "leaf")).toBe("each/leaf"); // both node AND edge ids
  expect(ns("each#0/inner#0", "leaf")).toBe("each#0/inner#0/leaf"); // nested composes
});

test("mapCallsite is the spawner, a hash, the index", () => {
  expect(mapCallsite("each", 0)).toBe("each#0");
  expect(mapCallsite("each", 12)).toBe("each#12");
});

test("minting is pure and re-callable", () => {
  // No emission counter: a re-clone on recovery re-keys identically.
  expect(ns("each#0", "x")).toBe(ns("each#0", "x"));
  expect(mapCallsite("each", 3)).toBe(mapCallsite("each", 3));
});
