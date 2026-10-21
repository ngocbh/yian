/**
 * The CODE node's whole-code and exact-line origins: a port of
 * `tests/compose/test_code_origins.py`.
 *
 * Ported: every row but the run below; the CODE bodies keep Python's text, which load doesn't
 * run.
 *
 * Ported (adapted): `test_code_origin_carriers_reject_invalid_shapes`, its node and descriptor
 * rows `whole-not-origin` and `line-item-not-origin`.
 *
 * Changed: the line origins are an array, not a tuple; a rewrite assigns a fresh array.
 *
 * TS-only: the node keeps a frozen copy of the line origins it is given, and a rewrite assigns a
 * fresh frozen array; a line-origins value that isn't an array is rejected, by the node and the
 * descriptor.
 *
 * Not ported: the `lines-not-tuple` row of `test_code_origin_carriers_reject_invalid_shapes` (TS
 * line origins are arrays only).
 *
 * Later (T4-8): `test_runtime_failure_diagnoses_exact_line_across_composition_paths`.
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { cloneChild } from "../../src/compile/expand.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { rewriteFlowOrigins } from "../../src/compile/origins.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { SourceCatalogBuilder, scopeOrigin } from "../../src/compose/source/catalog.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, type OriginResolver, SourceRange } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";

const CODE_ORIGIN = new OriginRef("/s:root/s:node/s:worker/s:field/s:code");
const LINE_ORIGINS = [
  new OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:0"),
  new OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:1"),
];

const LITERAL_FLOW = `id: code-origins
name: code-origins
nodes:
  worker:
    kind: code
    output: int
    code: |-
      value = 1
      return value
output: \${worker.output}
`;

/** Load `text` and return its CODE node `nodeId` and the resolver its origins resolve through. */
async function workerOf(text: string, nodeId = "worker"): Promise<{ node: CodeNode; resolver: OriginResolver }> {
  const loaded = await loadFlow(text, { sourceLabel: "code-origins.yaml" });
  return { node: loaded.compiled.nodes.get(nodeId) as CodeNode, resolver: loaded.originResolver! };
}

/** The CODE node `nodeId` of CALL node `callId`'s baked child in `loaded`. */
const childCode = (loaded: ComposedFlow, callId: string, nodeId: string) =>
  ((loaded.compiled.nodes.get(callId) as CallNode).child as CompiledFlow).nodes.get(nodeId) as CodeNode;

/** Expect a CODE node's origins to resolve to the one scalar occurrence at `cursor`. */
function expectResolvesCodeCursor(node: CodeNode, resolver: OriginResolver, cursor: SourceCursor): void {
  expect(resolver).not.toBeNull();
  const whole = resolver.resolve(node.codeOrigin!);
  expect(whole).not.toBeNull();
  expect(whole!.range).toEqual(cursor.range!);
  expect(whole!.text).toBe(cursor.document!.text);
  const lineRanges = cursor.scalarSource!.constructedLineRanges!;
  const lineFrames = node.codeLineOrigins.map((origin) => resolver.resolve(origin));
  expect(lineFrames.every((frame) => frame !== null && frame.text === cursor.document!.text)).toBe(true);
  expect(lineFrames.map((frame) => frame!.range)).toEqual([...lineRanges]);
}

test("literal code carries whole scalar and exact constructed lines", async () => {
  const { node, resolver } = await workerOf(LITERAL_FLOW);

  expect(node.codeOrigin).toEqual(CODE_ORIGIN);
  expect(node.codeLineOrigins).toEqual(LINE_ORIGINS);
  expect(resolver.resolve(node.codeOrigin!)!.range).toEqual(new SourceRange(7, 11, 10, 1));
  expect(node.codeLineOrigins.map((origin) => resolver.resolve(origin)!.range)).toEqual([
    new SourceRange(8, 7, 8, 16),
    new SourceRange(9, 7, 9, 19),
  ]);
});

test("compact root code origins retain the projected authored scalar", async () => {
  const text = `id: compact_worker
name: compact_worker
output: int
kind: code
code: |-
  value = 1
  return value
`;
  const codeCursor = SourceDocument.parse(text, "code-origins.yaml").root.child("code");
  const { node, resolver } = await workerOf(text, "compact_worker");

  expect(node.codeOrigin).toEqual(new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code"));
  expect(node.codeLineOrigins).toEqual([
    new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:0"),
    new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:1"),
  ]);
  expectResolvesCodeCursor(node, resolver, codeCursor);
});

test.each([
  ["quoted", '    code: "return 1"\n'],
  ["folded", "    code: >-\n      return 1\n"],
])("nonliteral code carries only whole scalar fallback: %s", async (_id, codeSource) => {
  const text =
    "id: code-origins\nname: code-origins\nnodes:\n  worker:\n    kind: code\n    output: int\n" +
    codeSource +
    "output: ${worker.output}\n";

  const { node, resolver } = await workerOf(text);

  expect(node.codeOrigin).toEqual(CODE_ORIGIN);
  expect(node.codeLineOrigins).toEqual([]);
  expect(resolver.resolve(node.codeOrigin!)).not.toBeNull();
});

test("source-free code accepts empty origin carriers", () => {
  const node = new CodeNode("worker", { code: "return 1", codeOrigin: null, codeLineOrigins: [] });

  expect(node.codeOrigin).toBeNull();
  expect(node.codeLineOrigins).toEqual([]);
  expect([...node.iterLocalOrigins()]).toEqual([]);
});

test("source-free attachment preserves explicit code origins", () => {
  const codeOrigin = new OriginRef("programmatic/code");
  const lineOrigins = [new OriginRef("programmatic/line/0")];
  const descriptor = new CodeDescriptor({ id: "worker", code: "return 1", codeOrigin, codeLineOrigins: lineOrigins });

  const attached = descriptor.withSourceOrigins(SourceCursor.virtual({ kind: "code", code: "return 1" }), {
    sitePath: ["root", "node", "worker"],
    catalogBuilder: new SourceCatalogBuilder(),
  });

  expect(attached.codeOrigin).toBe(codeOrigin);
  expect(attached.codeLineOrigins).toBe(descriptor.codeLineOrigins);
});

test("changed projection keeps explicit lines without claiming exact yaml lines", () => {
  const nodeCursor = SourceDocument.parse("node:\n  kind: code\n  code: |-\n    return 1\n").root.child("node");
  const projected = nodeCursor.projectMapping(
    new Map([
      ["kind", ["code", nodeCursor.child("kind")] as const],
      ["code", ["return 2", nodeCursor.child("code")] as const],
    ]),
  );
  const explicitLine = new OriginRef("programmatic/line/0");
  const descriptor = new CodeDescriptor({
    id: "worker",
    code: "return 2",
    codeOrigin: new OriginRef("programmatic/code"),
    codeLineOrigins: [explicitLine],
  });
  const builder = new SourceCatalogBuilder();

  const attached = descriptor.withSourceOrigins(projected, {
    sitePath: ["root", "node", "worker"],
    catalogBuilder: builder,
  });

  expect(attached.codeOrigin).toEqual(CODE_ORIGIN);
  expect(attached.codeLineOrigins).toEqual([explicitLine]);
  expect(builder.freeze().resolve(attached.codeOrigin!)).not.toBeNull();
});

test("code origins participate in local traversal and rewrite", () => {
  const nodeOrigin = new OriginRef("node");
  const codeOrigin = new OriginRef("code");
  const lineOrigins = [new OriginRef("line/0"), new OriginRef("line/1")];
  const node = new CodeNode("worker", {
    code: "const value = 1\nreturn value",
    codeOrigin,
    codeLineOrigins: lineOrigins,
  });
  node.origin = nodeOrigin;

  expect([...node.iterLocalOrigins()]).toEqual([nodeOrigin, codeOrigin, ...lineOrigins]);

  node.rewriteLocalOrigins((origin) => new OriginRef(`rewritten/${origin.key}`));

  expect([...node.iterLocalOrigins()]).toEqual([
    new OriginRef("rewritten/node"),
    new OriginRef("rewritten/code"),
    new OriginRef("rewritten/line/0"),
    new OriginRef("rewritten/line/1"),
  ]);
  expect(node.codeOrigin).toEqual(new OriginRef("rewritten/code"));
  expect(node.codeLineOrigins).toEqual([new OriginRef("rewritten/line/0"), new OriginRef("rewritten/line/1")]);
});

test.each([
  ["whole-not-origin", { codeOrigin: "not-an-origin" }, "CODE source origin must be an OriginRef or null"],
  [
    "line-item-not-origin",
    { codeLineOrigins: [new OriginRef("line/0"), "not-an-origin"] },
    "CODE line origins must contain only OriginRef values",
  ],
])("code origin carriers reject invalid shapes: %s", (_id, origins, message) => {
  expect(() => new CodeNode("worker", { code: "return 1", ...(origins as object) })).toThrow(new TypeError(message));
  expect(() => new CodeDescriptor({ id: "worker", code: "return 1", ...(origins as object) })).toThrow(
    new TypeError(message),
  );
});

test("code origins are noncanonical manifest and fingerprint metadata", async () => {
  const original = (await loadFlow(LITERAL_FLOW)).compiled;
  const rewritten = rewriteFlowOrigins(original, (origin) => new OriginRef(`rewritten/${origin.key}`));

  const originalNode = original.nodes.get("worker") as CodeNode;
  const rewrittenNode = rewritten.nodes.get("worker") as CodeNode;
  expect(rewrittenNode.codeOrigin).not.toEqual(originalNode.codeOrigin);
  expect(rewrittenNode.codeLineOrigins).not.toEqual(originalNode.codeLineOrigins);
  expect(flowManifest(rewritten)).toEqual(flowManifest(original));
  expect(flowFingerprint(rewritten)).toBe(flowFingerprint(original));
});

test("multi node def code origins use the definition lexical path", async () => {
  const text = `id: multi-def-code-origins
name: multi-def-code-origins
defs:
  child:
    nodes:
      worker:
        kind: code
        output: int
        code: |-
          value = 1
          return value
    output: \${worker.output}
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const node = childCode(loaded, "invoke", "worker");
  const codeCursor = document.root.child("defs").child("child").child("nodes").child("worker").child("code");

  expect(node.codeOrigin).toEqual(new OriginRef("/s:def/s:child/s:node/s:worker/s:field/s:code"));
  expect(node.codeLineOrigins).toEqual([
    new OriginRef("/s:def/s:child/s:node/s:worker/s:field/s:code/s:line/i:0"),
    new OriginRef("/s:def/s:child/s:node/s:worker/s:field/s:code/s:line/i:1"),
  ]);
  expectResolvesCodeCursor(node, loaded.originResolver!, codeCursor);
});

test("flat def code origins retain the projected authored scalar", async () => {
  const text = `id: flat-def-code-origins
name: flat-def-code-origins
defs:
  child:
    kind: code
    output: int
    code: |-
      value = 1
      return value
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const node = childCode(loaded, "invoke", "child");
  const codeCursor = document.root.child("defs").child("child").child("code");

  expect(node.codeOrigin).toEqual(new OriginRef("/s:def/s:child/s:node/s:child/s:field/s:code"));
  expect(node.codeLineOrigins).toEqual([
    new OriginRef("/s:def/s:child/s:node/s:child/s:field/s:code/s:line/i:0"),
    new OriginRef("/s:def/s:child/s:node/s:child/s:field/s:code/s:line/i:1"),
  ]);
  expectResolvesCodeCursor(node, loaded.originResolver!, codeCursor);
});

test("inline child code origins include the enclosing node path", async () => {
  const text = `id: inline-code-origins
name: inline-code-origins
nodes:
  invoke:
    kind: call
    nodes:
      worker:
        kind: code
        output: int
        code: |-
          value = 1
          return value
    output: \${worker.output}
output: \${invoke.output}
`;
  const codeCursor = SourceDocument.parse(text)
    .root.child("nodes")
    .child("invoke")
    .child("nodes")
    .child("worker")
    .child("code");
  const loaded = await loadFlow(text);
  const node = childCode(loaded, "invoke", "worker");

  expect(node.codeOrigin).toEqual(new OriginRef("/s:root/s:node/s:invoke/s:inline/s:node/s:worker/s:field/s:code"));
  expect(node.codeLineOrigins).toEqual([
    new OriginRef("/s:root/s:node/s:invoke/s:inline/s:node/s:worker/s:field/s:code/s:line/i:0"),
    new OriginRef("/s:root/s:node/s:invoke/s:inline/s:node/s:worker/s:field/s:code/s:line/i:1"),
  ]);
  expectResolvesCodeCursor(node, loaded.originResolver!, codeCursor);
});

test("external alias rebases every code origin without mutating child", async () => {
  const child = await loadFlow(LITERAL_FLOW, { sourceLabel: "child.yaml" });
  const parent = `id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: child_view
output: \${invoke.output}
uses:
  child_view: child
`;

  const loaded = await loadFlow(parent, { childResolver: () => child });
  const canonical = child.compiled.nodes.get("worker") as CodeNode;
  const embedded = childCode(loaded, "invoke", "worker");

  expect(canonical.codeOrigin).toEqual(CODE_ORIGIN);
  expect(canonical.codeLineOrigins).toEqual(LINE_ORIGINS);
  expect(embedded.codeOrigin).toEqual(scopeOrigin("child_view", canonical.codeOrigin!));
  expect(embedded.codeLineOrigins).toEqual(canonical.codeLineOrigins.map((origin) => scopeOrigin("child_view", origin)));
  const resolver = loaded.originResolver!;
  const childResolver = child.originResolver!;
  expect(resolver).not.toBeNull();
  expect(childResolver).not.toBeNull();
  expect(resolver.resolve(embedded.codeOrigin!)).toEqual(childResolver.resolve(canonical.codeOrigin!));
  expect(embedded.codeLineOrigins.map((origin) => resolver.resolve(origin))).toEqual(
    canonical.codeLineOrigins.map((origin) => childResolver.resolve(origin)),
  );
});

test("generated child clone preserves code origins for original resolver", async () => {
  const loaded = await loadFlow(LITERAL_FLOW);
  const original = loaded.compiled.nodes.get("worker") as CodeNode;

  const cloned = cloneChild(loaded.compiled, "invoke", {});
  const generated = cloned.nodes.get("invoke/worker") as CodeNode;

  expect(generated).not.toBe(original);
  expect(generated.codeOrigin).toEqual(original.codeOrigin);
  expect(generated.codeLineOrigins).toEqual(original.codeLineOrigins);
  const resolver = loaded.originResolver!;
  expect(resolver).not.toBeNull();
  expect(resolver.resolve(generated.codeOrigin!)).toEqual(resolver.resolve(original.codeOrigin!));
  expect(generated.codeLineOrigins.map((origin) => resolver.resolve(origin))).toEqual(
    original.codeLineOrigins.map((origin) => resolver.resolve(origin)),
  );
});

test("formatting moves resolved ranges without changing semantic keys", async () => {
  const moved = `id: code-origins
name: code-origins

# A formatting-only comment moves the authored scalar.
nodes:
  worker:
    kind: code
    output: int
    # Another formatting-only comment.
    code: |-
      value = 1
      return value
output: \${worker.output}
`;
  const codeCursorOf = (text: string) =>
    SourceDocument.parse(text, "code-origins.yaml").root.child("nodes").child("worker").child("code");
  const baselineCursor = codeCursorOf(LITERAL_FLOW);
  const movedCursor = codeCursorOf(moved);
  const baselineFlow = await loadFlow(LITERAL_FLOW, { sourceLabel: "code-origins.yaml" });
  const relocatedFlow = await loadFlow(moved, { sourceLabel: "code-origins.yaml" });
  const baselineNode = baselineFlow.compiled.nodes.get("worker") as CodeNode;
  const relocatedNode = relocatedFlow.compiled.nodes.get("worker") as CodeNode;

  expect(relocatedNode.codeOrigin).toEqual(baselineNode.codeOrigin);
  expect(relocatedNode.codeLineOrigins).toEqual(baselineNode.codeLineOrigins);
  expect(flowFingerprint(relocatedFlow.compiled)).toBe(flowFingerprint(baselineFlow.compiled));
  expectResolvesCodeCursor(baselineNode, baselineFlow.originResolver!, baselineCursor);
  expectResolvesCodeCursor(relocatedNode, relocatedFlow.originResolver!, movedCursor);
  expect(movedCursor.range).not.toEqual(baselineCursor.range);
  expect(movedCursor.scalarSource!.constructedLineRanges).not.toEqual(
    baselineCursor.scalarSource!.constructedLineRanges,
  );
});

// --- TS-only rows --- //

test("a line-origins value that isn't an array is rejected", () => {
  const origins = new OriginRef("line/0") as unknown as OriginRef[];
  const message = new TypeError("CODE line origins must be an array");

  expect(() => new CodeNode("worker", { code: "return 1", codeLineOrigins: origins })).toThrow(message);
  expect(() => new CodeDescriptor({ id: "worker", code: "return 1", codeLineOrigins: origins })).toThrow(message);
});

test("the line origins are a frozen copy, and a rewrite assigns a fresh frozen array", () => {
  const given = [new OriginRef("line/0")];
  const node = new CodeNode("worker", { code: "return 1", codeLineOrigins: given });
  given.push(new OriginRef("line/1"));

  expect(node.codeLineOrigins).toEqual([new OriginRef("line/0")]);
  expect(Object.isFrozen(node.codeLineOrigins)).toBe(true);

  const before = node.codeLineOrigins;
  node.rewriteLocalOrigins((origin) => origin);

  expect(node.codeLineOrigins).not.toBe(before);
  expect(Object.isFrozen(node.codeLineOrigins)).toBe(true);
  expect(before).toEqual([new OriginRef("line/0")]);
});
