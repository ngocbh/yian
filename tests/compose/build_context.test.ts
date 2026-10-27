/**
 * The `BuildContext` and `BuildResult` carriers a node `build` reads and returns; the frozen rows
 * check `Object.isFrozen`.
 */

import { expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
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

test("BuildContext owner defaults to root and is threaded by assemble", async () => {
  // Each flow's descriptors see that flow's lexical owner path.
  expect(new BuildContext({ registry: new Map() }).owner).toEqual(["root"]);

  const seen: (readonly string[])[] = [];
  const original = AgentDescriptor.prototype.build;
  const spy = spyOn(AgentDescriptor.prototype, "build").mockImplementation(function (
    this: AgentDescriptor,
    ctx: BuildContext,
  ) {
    seen.push(ctx.owner);
    return original.call(this, ctx);
  });
  try {
    await loadFlow(
      "id: p\nname: p\ndefs:\n  child:\n    kind: agent\n    prompt: hi\n" +
        "    output: str\nnodes:\n  top:\n    kind: agent\n    prompt: hi\n" +
        "    output: str\n  invoke:\n    kind: call\n    call: child\n" +
        "output: ${top.output}\n",
    );
  } finally {
    spy.mockRestore();
  }
  expect([...seen].sort((a, b) => (a.join("/") < b.join("/") ? -1 : 1))).toEqual([["def", "child"], ["root"]]);
});
