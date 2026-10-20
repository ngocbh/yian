/**
 * The edges phase's ordering edges (`compose/edges/ordering.ts`): a port of
 * `tests/compose/test_edges.py`.
 *
 * `inferOrderingEdges` lowers each `depends_on:`/`runs_after:` declaration into a zero-data
 * ordering edge, and reports an unknown target as a `LoadError` located at the declaring node
 * descriptor's range.
 *
 * Changed: the CODE reference is TS's (`./m.ts:f`), where Python names `m:f`.
 */

import { expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";

import { inferOrderingEdges } from "../../src/compose/edges/ordering.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { loadError } from "./declarations.ts";

/** Return the physical cursor over a flow document's `nodes:` mapping. */
function nodesReader(text: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes");
}

test("an unknown ordering target is located", () => {
  // A `depends_on:`/`runs_after:` that names an unknown node is a LoadError located at the
  // DECLARING node descriptor's range.
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  a:\n    kind: code\n    code: ./m.ts:f\n    depends_on: [ghost]\n    output: str\n" +
    "output: ${a.output}\n";
  const nodesCursor = nodesReader(text);
  const descriptors = parseNodes(nodesCursor);

  const error = loadError(() =>
    inferOrderingEdges(descriptors, new Set(descriptors.keys()), {
      diagnostics: LoadDiagnostics.fromCursor(nodesCursor),
    }),
  );

  expect(error.message).toBe("node 'a': `depends_on` references unknown node 'ghost'");
  expect(error.diagnostic.frames.length).toBe(1);
  expect(error.diagnostic.frames[0]!.range).toStrictEqual(descriptors.get("a")!.range!);
  expect(error.diagnostic.frames[0]!.text).toBe(text);
});

test("ordering edges are built", () => {
  // A valid `depends_on:` lowers to a zero-data ordering edge (optional=false).
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  warm:\n    kind: code\n    code: ./m.ts:f\n    output: str\n" +
    "  a:\n    kind: code\n    code: ./m.ts:f\n    depends_on: [warm]\n    output: str\n" +
    "output: ${a.output}\n";
  const descriptors = parseNodes(nodesReader(text));

  const edges = inferOrderingEdges(descriptors, new Set(descriptors.keys()));

  expect(edges).toStrictEqual([new Edge("warm~>a#0", "warm", "a", { ordering: true, optional: false })]);
});
