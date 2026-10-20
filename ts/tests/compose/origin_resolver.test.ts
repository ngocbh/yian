/**
 * Installing the complete origin resolver at the Compose boundary: a port of
 * `tests/compose/test_origin_resolver.py`.
 *
 * Ported: every row but the one below. A Python resolver class is a plain object with `resolve`,
 * and its broken property a getter. Changed: `ResolverTable` has no `origins` list in TS, so the
 * table's exact keys are checked as its `size` plus `has` for each reachable origin; Python's
 * `deepcopy` of a loaded child is `copyGraph` of its compiled flow; Python's
 * `dataclasses.replace` is a new `ComposedFlow`; the check that a loaded flow has no
 * `_local_catalog` attribute is dropped, since a TS `ComposedFlow` has no private fields to probe.
 *
 * Not ported: `test_resolver_metadata_is_immutable_copy_reused_and_equality_neutral`, which
 * pins dataclass `deepcopy` and equality; a TS `ComposedFlow` has neither.
 */

import { expect, test } from "bun:test";

import { type CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { iterFlowOrigins, rewriteFlowOrigins } from "../../src/compile/origins.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { finalizeComposedFlow, loadFlow } from "../../src/compose/loader.ts";
import { SourceCatalog } from "../../src/compose/source/index.ts";
import { type DiagnosticFrame, OriginRef, type OriginResolver, ResolverTable } from "../../src/diagnostics.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";

import { loadErrorAsync } from "./declarations.ts";

const CHILD = `id: child
name: child
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: \${worker.output}
`;

const PARENT = `id: parent
name: parent
nodes:
  first:
    kind: call
    call: left
  second:
    kind: call
    call: right
output: \${second.output}
uses:
  left: child
  right: child
`;

const ONE_ALIAS_PARENT = `id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: child_view
output: \${invoke.output}
uses:
  child_view: child
`;

/** Return one canonical reachable origin per opaque key. */
function origins(flow: CompiledFlow): Map<string, OriginRef> {
  return new Map([...iterFlowOrigins(flow)].map((origin) => [origin.key, new OriginRef(origin.key)]));
}

/** The baked child of `flow`'s CALL node `nodeId`. */
const childOf = (flow: ComposedFlow, nodeId: string) =>
  (flow.compiled.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** Return `flow` with some of its fields replaced. */
function replaced(
  flow: ComposedFlow,
  changes: { compiled?: CompiledFlow; originResolver?: OriginResolver | null },
): ComposedFlow {
  return new ComposedFlow(changes.compiled ?? flow.compiled, {
    version: flow.version,
    name: flow.name,
    description: flow.description,
    originResolver: changes.originResolver === undefined ? flow.originResolver : changes.originResolver,
    importDepth: flow.importDepth,
  });
}

/** Expect `resolver` to route exactly the reachable origins of `flow`. */
function expectExactRoutes(resolver: OriginResolver | null, flow: CompiledFlow): void {
  expect(resolver).toBeInstanceOf(ResolverTable);
  const table = resolver as ResolverTable;
  const reachable = origins(flow);
  expect(table.size).toBe(reachable.size);
  for (const origin of reachable.values()) expect(table.has(origin)).toBe(true);
}

/** Expect one external-child failure owned by the importing alias site. */
function expectChildViewUsesFrame(error: LoadError): void {
  expect(error.message).toContain("alias 'child_view'");
  const frames = error.diagnostic.frames;
  expect(frames.length).toBe(1);
  expect(frames[0]!.label).toBe("<flow>");
  expect(frames[0]!.range.startLine).toBe(9);
  expect(frames[0]!.note).toBe("uses alias 'child_view'");
}

/** Load `ONE_ALIAS_PARENT` with a resolver returning `child`, expecting a `LoadError`. */
const loadOneAlias = (child: unknown) =>
  loadErrorAsync(loadFlow(ONE_ALIAS_PARENT, { childResolver: () => child as ComposedFlow }));

test("loadFlow installs an exact resolver and filters unused defs", async () => {
  const text = `id: complete-local
name: complete-local
input:
  n: !int
defs:
  unused:
    kind: code
    code: return 2
    output: int
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: \${worker.output}
asserts:
  - input.n >= 0
  - worker.output >= 0
`;

  const loaded = await loadFlow(text);

  expectExactRoutes(loaded.originResolver, loaded.compiled);
  for (const origin of origins(loaded.compiled).values()) expect(loaded.originResolver!.resolve(origin)).not.toBeNull();
});

test("a source-free load has no resolver", async () => {
  const loaded = await loadFlow("id: empty\nname: empty\nnodes: {}\n");
  expect(origins(loaded.compiled).size).toBe(0);
  expect(loaded.originResolver).toBeNull();
});

test("one alias reuses one rebased child for several calls", async () => {
  const child = await loadFlow(CHILD);
  const text = PARENT.replace("    call: right\n", "    call: left\n");

  const loaded = await loadFlow(text, { childResolver: () => child });
  const first = childOf(loaded, "first");
  const second = childOf(loaded, "second");

  expect(first).toBe(second);
  expectExactRoutes(loaded.originResolver, loaded.compiled);
  for (const origin of origins(first).values()) expect(loaded.originResolver!.resolve(origin)).not.toBeNull();
});

test("two aliases rebase the canonical child without changing it", async () => {
  const child = await loadFlow(CHILD);
  const canonicalOrigins = origins(child.compiled);
  const canonicalFingerprint = flowFingerprint(child.compiled);

  const loaded = await loadFlow(PARENT, { childResolver: () => child });
  const firstOrigins = origins(childOf(loaded, "first"));
  const secondOrigins = origins(childOf(loaded, "second"));
  const childKeys = new Set([...firstOrigins.keys(), ...secondOrigins.keys()]);
  const parentKeys = [...origins(loaded.compiled).keys()].filter((key) => !childKeys.has(key));

  expect(childOf(loaded, "first")).not.toBe(childOf(loaded, "second"));
  expect([...firstOrigins.keys()].some((key) => secondOrigins.has(key))).toBe(false);
  expect(parentKeys.length).toBeGreaterThan(0);
  expect(origins(child.compiled)).toEqual(canonicalOrigins);
  expect(flowFingerprint(childOf(loaded, "first"))).toBe(canonicalFingerprint);
  expect(flowFingerprint(childOf(loaded, "second"))).toBe(canonicalFingerprint);
  for (const origin of [...firstOrigins.values(), ...secondOrigins.values()]) {
    expect(loaded.originResolver!.resolve(origin)!.text).toBe(CHILD);
  }
  for (const key of parentKeys) expect(loaded.originResolver!.resolve(new OriginRef(key))!.text).toBe(PARENT);
});

test("a delimiter-heavy alias rebases opaque child origins", async () => {
  const alias = "left/~: child";
  const parent = `id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: '${alias}'
output: \${invoke.output}
uses:
  '${alias}': child
`;

  const loaded = await loadFlow(parent, { childResolver: () => loadFlow(CHILD) });
  const embedded = origins(childOf(loaded, "invoke"));

  expect(embedded.size).toBeGreaterThan(0);
  for (const origin of embedded.values()) expect(loaded.originResolver!.resolve(origin)!.text).toBe(CHILD);
});

test.each([
  ["no resolver", null],
  ["an empty table", new ResolverTable()],
])("an originful custom child needs a complete resolver (%s)", async (_kind, replacement) => {
  const child = replaced(await loadFlow(CHILD), { originResolver: replacement });
  const error = await loadOneAlias(child);
  expect(error.message).toContain("origin resolver");
  expectChildViewUsesFrame(error);
});

test("a custom child resolver's failure is wrapped at the uses: boundary", async () => {
  const broken = {
    resolve(): DiagnosticFrame | null {
      throw new Error("resolver exploded");
    },
  };
  const error = await loadOneAlias(replaced(await loadFlow(CHILD), { originResolver: broken }));
  expect(error.message).toContain("resolver exploded");
  expectChildViewUsesFrame(error);
});

test("a custom child resolver's getter failure is wrapped at the uses: boundary", async () => {
  const broken = {
    get resolve(): OriginResolver["resolve"] {
      throw new Error("resolver getter exploded");
    },
  };
  const error = await loadOneAlias(replaced(await loadFlow(CHILD), { originResolver: broken }));
  expect(error.message).toContain("resolver getter exploded");
  expectChildViewUsesFrame(error);
});

test("a custom child's origin hook failure is wrapped at the uses: boundary", async () => {
  /** A node whose provenance hook fails. */
  class BrokenOriginNode extends Node {
    readonly kind = NodeKind.CODE;

    override *iterLocalOrigins(): Generator<OriginRef> {
      throw new Error("origin hook exploded");
    }
  }

  const canonical = await loadFlow(CHILD);
  const compiled = copyGraph(canonical.compiled);
  compiled.nodes.set("worker", new BrokenOriginNode("worker"));
  const error = await loadOneAlias(replaced(canonical, { compiled }));
  expect(error.message).toContain("origin hook exploded");
  expectChildViewUsesFrame(error);
});

test("validating a custom child keeps exact origin subclasses", async () => {
  class ChildOrigin extends OriginRef {}

  const canonical = await loadFlow(CHILD);
  const compiled = rewriteFlowOrigins(canonical.compiled, (origin) => new ChildOrigin(origin.key));
  const exact: OriginResolver = {
    resolve: (origin) =>
      Object.getPrototypeOf(origin) === ChildOrigin.prototype
        ? canonical.originResolver!.resolve(new OriginRef(origin.key))
        : null,
  };
  const child = replaced(canonical, { compiled, originResolver: exact });

  const loaded = await loadFlow(PARENT, { childResolver: () => child });
  for (const origin of iterFlowOrigins(loaded.compiled)) expect(loaded.originResolver!.resolve(origin)).not.toBeNull();
});

test("the external resolver must return a ComposedFlow", async () => {
  const error = await loadOneAlias({});
  expect(error.message).toContain("expected ComposedFlow");
  expectChildViewUsesFrame(error);
});

test("a source-free custom child refuses an origin resolver", async () => {
  const child = replaced(await loadFlow("id: empty-child\nname: empty-child\nnodes: {}\n"), {
    originResolver: new ResolverTable(),
  });
  const error = await loadOneAlias(child);
  expect(error.message).toContain("source-free child");
  expectChildViewUsesFrame(error);
});

test("a source-free custom child stays source-free when embedded", async () => {
  const child = await loadFlow("id: empty-child\nname: empty-child\nnodes: {}\n");
  const parent = `id: empty-parent
name: empty-parent
nodes:
  invoke:
    kind: call
    call: child
uses:
  child: empty-child
`;

  const loaded = await loadFlow(parent, { childResolver: () => child });
  expect(origins(childOf(loaded, "invoke")).size).toBe(0);
  expectExactRoutes(loaded.originResolver, loaded.compiled);
});

test("finalization refuses missing and overlapping routes", async () => {
  const loaded = await loadFlow(CHILD);
  const staged = replaced(loaded, { originResolver: null });
  const table = loaded.originResolver as ResolverTable;
  expect(table).toBeInstanceOf(ResolverTable);

  expect(() => finalizeComposedFlow(staged, new SourceCatalog(), [])).toThrow(/no resolver route/);
  expect(() => finalizeComposedFlow(staged, new SourceCatalog(), [table, table])).toThrow(
    /multiple resolver routes/,
  );
});
