/**
 * The `BuildContext` and `BuildResult` carriers a node `build` reads and returns: a port of
 * `tests/compose/test_build_context.py`. The frozen rows check `Object.isFrozen`, the TS form of
 * a frozen dataclass.
 *
 * TS-only: `baseDir`, the directory a CODE reference resolves against, defaults to `null`.
 *
 * Dropped: `test_build_result_range_metadata_does_not_affect_equality` (TS has no dataclass
 * `==`).
 *
 * Later (T3-22): the `load_flow` half of
 * `test_build_context_owner_defaults_to_root_and_is_threaded_by_assemble`, each flow's
 * descriptors seeing that flow's owner path.
 */

import { expect, test } from "bun:test";

import { BuildContext, BuildResult } from "../../src/compose/nodes/base.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

test("BuildContext requires only the registry; every other field defaults", () => {
  const ctx = new BuildContext({ registry: new Map() });
  expect(ctx.resolver).toBeNull();
  expect(ctx.sourceFile).toBe("<inline-code>");
  expect(ctx.resolveOrigin).toBeNull();
  expect(ctx.outputContractReader).toBeNull();
  expect(ctx.baseDir).toBeNull();
});

test("BuildResult defaults its graph and Compose metadata to empty values", () => {
  const result = new BuildResult({ node: new StartNode("n"), wiring: new Map([["x", "y"]]) });
  expect(result.edges).toEqual([]);
  expect(result.nodeRange).toBeNull();
  expect(result.fieldRanges).toEqual(new Map());
});

test("BuildContext is frozen", () => {
  const ctx = new BuildContext({ registry: new Map() });
  expect(Object.isFrozen(ctx)).toBe(true);
  expect(Object.isFrozen(ctx.owner)).toBe(true);
});

test("BuildResult is frozen", () => {
  const result = new BuildResult({ node: new StartNode("n"), wiring: new Map() });
  expect(Object.isFrozen(result)).toBe(true);
  expect(Object.isFrozen(result.edges)).toBe(true);
});

test("BuildContext owner defaults to root", () => {
  expect(new BuildContext({ registry: new Map() }).owner).toEqual(["root"]);
});
