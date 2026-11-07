/**
 * The `BuildContext` and `BuildResult` carriers a node `build` reads and returns; the frozen rows
 * check `Object.isFrozen`.
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
});

test("BuildResult is frozen", () => {
  const result = new BuildResult({ node: new StartNode("n"), wiring: new Map() });
  expect(Object.isFrozen(result)).toBe(true);
  expect(Object.isFrozen(result.edges)).toBe(true);
});
