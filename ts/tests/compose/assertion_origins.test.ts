/**
 * Authored Compose assertion origins: a port of
 * `tests/compose/test_assertion_origins.py::test_alias_occurrences_mint_distinct_node_and_flow_assertion_origins`,
 * `::test_duplicate_loop_assertions_keep_sites_when_the_driver_respawns` and
 * `::test_virtual_assertions_do_not_mint_or_erase_programmatic_origins`.
 *
 * Changed: the flows assemble with `assembleText` where Python calls `load_flow`, and an origin
 * resolves through the catalog builder `assembleText` returns (`resolveRegistered`), where Python
 * reads the loaded flow's `origin_resolver`. Origins compare by `key`, as Python's set compares
 * them. The LOOP body's CODE reference is TS's (`./fixtures/compose_codefns.ts:loop_bump`). The
 * source-free row calls `assemble` with a scope that resolves nothing and an empty catalog
 * builder, where Python passes a `_CallableScope`.
 *
 * Later (T3-22): `test_root_flow_assertions_keep_indexed_sites_through_rewrites` (its `defs:`
 * entry), `test_same_node_assertion_text_has_distinct_root_def_inline_and_flat_sites` (`defs:` and
 * a flattened def), `test_compact_root_node_assertion_uses_its_projected_physical_item` (a compact
 * flow) and `test_def_boundary_and_call_relocation_reuse_authored_assertion_carriers` (`defs:`
 * and `call_subgraph`).
 */

import { expect, test } from "bun:test";

import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { assemble, type CallableScope, ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { Assertion, isExactly } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { assembleText } from "./declarations.ts";

/** Check that one semantic assertion resolves its exact authored list item; return its origin. */
function expectAssertionSite(
  builder: SourceCatalogBuilder,
  assertion: Assertion,
  cursor: SourceCursor,
): OriginRef {
  expect(isExactly(assertion, Assertion)).toBe(true);
  expect(assertion.origin).toBeInstanceOf(OriginRef);

  const frame = builder.resolveRegistered(assertion.origin!);
  expect(frame).not.toBeNull();
  expect(frame!.text).toBe(cursor.document!.text);
  expect(frame!.range).toEqual(cursor.range!);
  return assertion.origin!;
}

/** The number of distinct origins in `origins`. */
const distinct = (origins: readonly OriginRef[]) => new Set(origins.map((origin) => origin.key)).size;

test("alias occurrences mint distinct node and flow assertion origins", () => {
  // Alias uses remain distinct even when they share one definition range.
  const text = `id: aliased-assertion-origins
name: aliased-assertion-origins
input:
  n: !int
nodes:
  first: &shared_node
    kind: code
    code: return 1
    output: int
    asserts:
      - output >= 0
  second: *shared_node
output:
  first: \${first.output}
  second: \${second.output}
asserts:
  - &shared_rule input.n >= 0
  - *shared_rule
`;
  const document = SourceDocument.parse(text, "flow.yaml");
  const { flow: loaded, builder } = assembleText(text);

  const nodeAssert = (nodeId: string) => document.root.child("nodes").child(nodeId).child("asserts").child(0);
  const [firstCursor, secondCursor] = [nodeAssert("first"), nodeAssert("second")];
  const nodeOrigins = (
    [
      ["first", firstCursor],
      ["second", secondCursor],
    ] as const
  ).map(([nodeId, cursor]) => expectAssertionSite(builder, loaded.compiled.nodes.get(nodeId)!.postAsserts[0]!, cursor));
  expect(firstCursor.range).toEqual(secondCursor.range);
  expect(firstCursor.path).not.toEqual(secondCursor.path);
  expect(distinct(nodeOrigins)).toBe(2);

  const flowCursors = document.root.child("asserts");
  const startAssertions = loaded.compiled.nodes.get(START_ID)!.postAsserts;
  const flowOrigins = startAssertions.map((assertion, index) =>
    expectAssertionSite(builder, assertion, flowCursors.child(index)),
  );
  expect(flowCursors.child(0).range).toEqual(flowCursors.child(1).range);
  expect(flowCursors.child(0).path).not.toEqual(flowCursors.child(1).path);
  expect(distinct(flowOrigins)).toBe(2);
});

test("duplicate loop assertions keep their sites when the driver respawns", () => {
  // One node's duplicate authored carriers survive LOOP reconstruction.
  const text = `id: loop-assertion-origins
name: loop-assertion-origins
nodes:
  drive:
    kind: loop
    times: 1
    input:
      n: 0
      exited: false
    asserts:
      - output.n >= 0
      - output.n >= 0
    nodes:
      bump:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${bump.output}
output: \${drive.output}
`;
  const document = SourceDocument.parse(text, "flow.yaml");
  const { flow: loaded, builder } = assembleText(text);
  const drive = loaded.compiled.nodes.get("drive") as LoopNode;
  const cursors = document.root.child("nodes").child("drive").child("asserts");

  const origins = drive.postAsserts.map((assertion, index) =>
    expectAssertionSite(builder, assertion, cursors.child(index)),
  );
  expect(distinct(origins)).toBe(2);

  const respawned = drive.respawn(1);

  expect(respawned.postAsserts).toEqual(drive.postAsserts);
  expect(respawned.postAsserts).not.toBe(drive.postAsserts);
  expect(respawned.postAsserts.map((assertion) => assertion.origin)).toEqual(
    drive.postAsserts.map((assertion) => assertion.origin),
  );
});

test("virtual assertions do not mint or erase programmatic origins", () => {
  // Source-free assembly leaves raw assertions without an origin and explicit origins intact.
  const raw = "input.n >= 0";
  const explicit = new OriginRef("programmatic-assertion");
  const programmatic = new Assertion("worker.output >= 0", "programmatic worker assertion", { origin: explicit });
  const scope: CallableScope = {
    catalogBuilder: new SourceCatalogBuilder(),
    resolve: () => new ComposedFlow(new CompiledFlow(new Map(), [])),
    registerPrivate: () => {},
  };
  const loaded = assemble({
    inputsSection: { n: "int" },
    nodesCursor: SourceCursor.virtual({ worker: { kind: "code", code: "return 1", outputs: "int" } }),
    outputsSection: "${worker.output}",
    assertsSection: [raw, programmatic],
    assertsCursor: SourceCursor.virtual([raw, programmatic.expression]),
    registry: new Map(),
    scope,
    owner: ["root"],
  });

  const start = loaded.compiled.nodes.get(START_ID)!;
  const end = loaded.compiled.nodes.get(END_ID)!;
  expect(start.postAsserts[0]!.display).toBe(raw);
  expect(start.postAsserts[0]!.origin).toBeNull();
  expect(end.postAsserts[0]!.display).toBe(programmatic.display);
  expect(end.postAsserts[0]!.origin).toBe(explicit);
  expect(loaded.originResolver).toBeNull();
});
