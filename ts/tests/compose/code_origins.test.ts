/**
 * The CODE node's whole-code and exact-line origins: a port of
 * `tests/compose/test_code_origins.py`.
 *
 * Ported: `test_source_free_code_accepts_empty_origin_carriers` and
 * `test_code_origins_participate_in_local_traversal_and_rewrite`.
 *
 * Ported (adapted): `test_code_origin_carriers_reject_invalid_shapes`, its node and descriptor
 * rows `whole-not-origin` and `line-item-not-origin`.
 *
 * Changed: the line origins are an array, not a tuple; a rewrite assigns a fresh array.
 * `test_code_origins_are_noncanonical_manifest_and_fingerprint_metadata` builds its CompiledFlow by
 * hand instead of loading it. `test_literal_code_carries_whole_scalar_and_exact_constructed_lines`,
 * `test_compact_root_code_origins_retain_the_projected_authored_scalar` and
 * `test_nonliteral_code_carries_only_whole_scalar_fallback` assemble the root flow with
 * `assembleText`, where Python loads it, and resolve through the scope's catalog builder in place
 * of `loaded.origin_resolver`.
 *
 * TS-only: the node keeps a frozen copy of the line origins it is given, and a rewrite assigns a
 * fresh frozen array; a line-origins value that isn't an array is rejected, by the node and the
 * descriptor.
 *
 * Not ported: the `lines-not-tuple` row of `test_code_origin_carriers_reject_invalid_shapes` (TS
 * line origins are arrays only).
 *
 * Later (T3-22): `test_multi_node_def_code_origins_use_the_definition_lexical_path`,
 * `test_flat_def_code_origins_retain_the_projected_authored_scalar`,
 * `test_external_alias_rebases_every_code_origin_without_mutating_child`,
 * `test_generated_child_clone_preserves_code_origins_for_original_resolver` and
 * `test_formatting_moves_resolved_ranges_without_changing_semantic_keys`. Later (T3-17, T3-16):
 * `test_inline_child_code_origins_include_the_enclosing_node_path`.
 *
 * Later (T4-9): `test_runtime_failure_diagnoses_exact_line_across_composition_paths`.
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { rewriteFlowOrigins } from "../../src/compile/origins.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { assembleText } from "./declarations.ts";

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

/** Assemble `text` and return its CODE node `nodeId` and the catalog builder its origins resolve through. */
function workerOf(text: string, nodeId = "worker"): { node: CodeNode; builder: SourceCatalogBuilder } {
  const { flow, builder } = assembleText(text, "code-origins.yaml");
  return { node: flow.compiled.nodes.get(nodeId) as CodeNode, builder };
}

/** Expect a CODE node's origins to resolve to the one scalar occurrence at `cursor`. */
function expectResolvesCodeCursor(node: CodeNode, builder: SourceCatalogBuilder, cursor: SourceCursor): void {
  const whole = builder.resolveRegistered(node.codeOrigin!);
  expect(whole).not.toBeNull();
  expect(whole!.range).toEqual(cursor.range!);
  expect(whole!.text).toBe(cursor.document!.text);
  const lineRanges = cursor.scalarSource!.constructedLineRanges!;
  const lineFrames = node.codeLineOrigins.map((origin) => builder.resolveRegistered(origin));
  expect(lineFrames.every((frame) => frame !== null && frame.text === cursor.document!.text)).toBe(true);
  expect(lineFrames.map((frame) => frame!.range)).toEqual([...lineRanges]);
}

test("literal code carries whole scalar and exact constructed lines", () => {
  const { node, builder } = workerOf(LITERAL_FLOW);

  expect(node.codeOrigin).toEqual(CODE_ORIGIN);
  expect(node.codeLineOrigins).toEqual(LINE_ORIGINS);
  expect(builder.resolveRegistered(node.codeOrigin!)!.range).toEqual(new SourceRange(7, 11, 10, 1));
  expect(node.codeLineOrigins.map((origin) => builder.resolveRegistered(origin)!.range)).toEqual([
    new SourceRange(8, 7, 8, 16),
    new SourceRange(9, 7, 9, 19),
  ]);
});

test("compact root code origins retain the projected authored scalar", () => {
  const text = `id: compact_worker
name: compact_worker
output: int
kind: code
code: |-
  value = 1
  return value
`;
  const codeCursor = SourceDocument.parse(text, "code-origins.yaml").root.child("code");
  const { node, builder } = workerOf(text, "compact_worker");

  expect(node.codeOrigin).toEqual(new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code"));
  expect(node.codeLineOrigins).toEqual([
    new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:0"),
    new OriginRef("/s:root/s:node/s:compact_worker/s:field/s:code/s:line/i:1"),
  ]);
  expectResolvesCodeCursor(node, builder, codeCursor);
});

test.each([
  ["quoted", '    code: "return 1"\n'],
  ["folded", "    code: >-\n      return 1\n"],
])("nonliteral code carries only whole scalar fallback: %s", (_id, codeSource) => {
  const text =
    "id: code-origins\nname: code-origins\nnodes:\n  worker:\n    kind: code\n    output: int\n" +
    codeSource +
    "output: ${worker.output}\n";

  const { node, builder } = workerOf(text);

  expect(node.codeOrigin).toEqual(CODE_ORIGIN);
  expect(node.codeLineOrigins).toEqual([]);
  expect(builder.resolveRegistered(node.codeOrigin!)).not.toBeNull();
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

test("code origins are noncanonical manifest and fingerprint metadata", () => {
  const worker = new CodeNode("worker", {
    code: "const value = 1\nreturn value",
    codeOrigin: new OriginRef("worker/code"),
    codeLineOrigins: [new OriginRef("worker/code/line/0"), new OriginRef("worker/code/line/1")],
  });
  const original = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["worker", worker],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [new Edge(`worker->${END_ID}`, "worker", END_ID, { inputGroup: "result" })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["worker", new Map()],
        [END_ID, new Map([["result", "${worker.output}"]])],
      ]),
    },
  );
  const rewritten = rewriteFlowOrigins(original, (origin) => new OriginRef(`rewritten/${origin.key}`));

  const originalNode = original.nodes.get("worker") as CodeNode;
  const rewrittenNode = rewritten.nodes.get("worker") as CodeNode;
  expect(rewrittenNode.codeOrigin).not.toEqual(originalNode.codeOrigin);
  expect(rewrittenNode.codeLineOrigins).not.toEqual(originalNode.codeLineOrigins);
  expect(flowManifest(rewritten)).toEqual(flowManifest(original));
  expect(flowFingerprint(rewritten)).toBe(flowFingerprint(original));
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
