/**
 * The host each desugar pass records on a node it lifts out of an authored node: a port of
 * `tests/compose/test_lifted_hosts.py`.
 *
 * A lifted node's `host` is the node whose text it was written in. A later edge step gives the
 * lifted node its host's CASE routes, `dependsOn`, and `runsAfter`. A node that is itself a route
 * target (a branch `call(...)`), a value entry, or a node that comes from the flow
 * `output:`/`asserts:` has no host.
 *
 * Changed: `test_host_is_not_an_authored_key` assembles the flow (`assembleText`) where Python
 * loads it.
 *
 * Later (T3-17): `test_nested_branch_call_records_the_outer_call_as_host` (the case-target
 * desugar) and `test_adaptive_composer_records_the_gate_as_host` (the adaptive-questions desugar).
 */

import { expect, test } from "bun:test";

import { desugarInlineCalls, synthIdGen } from "../../src/compose/desugar/inline_calls.ts";
import { readValueEntry } from "../../src/compose/desugar/value_entries.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import type { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { assembleText, loadError } from "./declarations.ts";

/** Parse a flow whose `nodes:` section is `nodesYaml` into descriptors. */
function descriptorsOf(nodesYaml: string): Map<string, NodeDescriptor> {
  const text = "id: f\nname: f\nnodes:\n" + nodesYaml;
  return parseNodes(SourceDocument.parse(text).root.child("nodes"));
}

/** The lifted `__call_N` descriptors, keyed by id. */
function synthOf(descriptors: ReadonlyMap<string, NodeDescriptor>): Map<string, CallDescriptor> {
  return new Map(
    [...descriptors].filter(([nodeId]) => nodeId.startsWith("__call_")) as [string, CallDescriptor][],
  );
}

test("authored descriptors have no host", () => {
  const descriptors = descriptorsOf("  a:\n    kind: code\n    code: m:f\n    output: str\n");
  expect(descriptors.get("a")!.host).toBeNull();
});

test("host is not an authored key", () => {
  const text = `
id: f
name: f
nodes:
  a:
    kind: code
    code: "return 1"
    host: b
    output: int
output: \${a.output}
`;
  expect(loadError(() => assembleText(text)).message).toBe(
    "node 'a' (kind=code): field 'host' is not allowed " +
      "(allowed: asserts, code, depends_on, env, input, kind, node_name, output, runs_after)",
  );
});

test("inline calls in a node input record the node as host", () => {
  const descriptors = descriptorsOf(
    "  publish:\n    kind: code\n    code: m:f\n" +
      "    input:\n      draft: call(f, x=call(g, y=1))\n    output: str\n",
  );
  const [newDescriptors] = desugarInlineCalls(descriptors, null, { nextId: synthIdGen() });
  const synth = synthOf(newDescriptors);
  expect([...synth.keys()].sort()).toEqual(["__call_0", "__call_1"]);
  expect([...synth.values()].map((desc) => desc.host)).toEqual(["publish", "publish"]);
  expect(newDescriptors.get("publish")!.host).toBeNull();
});

test("inline calls in map over and loop seed record their node", () => {
  const descriptors = descriptorsOf(
    "  each:\n    kind: map\n    call: work\n    over: call(items)\n" +
      "  turn:\n    kind: loop\n    call: body\n    times: 2\n" +
      "    input:\n      s: call(seed)\n",
  );
  const [newDescriptors] = desugarInlineCalls(descriptors, null, { nextId: synthIdGen() });
  const hosts = Object.fromEntries([...synthOf(newDescriptors).values()].map((desc) => [desc.call, desc.host]));
  expect(hosts).toEqual({ items: "each", seed: "turn" });
});

test("inline calls in flow output and asserts have no host", () => {
  const descriptors = descriptorsOf("  a:\n    kind: code\n    code: m:f\n    output: int\n");
  const [newDescriptors] = desugarInlineCalls(descriptors, { r: "call(f)" }, {
    assertsSection: ["call(g)"],
    nextId: synthIdGen(),
  });
  const synth = [...synthOf(newDescriptors).values()];
  expect(synth.map((desc) => desc.call).sort()).toEqual(["f", "g"]);
  expect(synth.map((desc) => desc.host)).toEqual([null, null]);
});

test("value entry has no host", () => {
  const nodes = SourceDocument.parse("id: f\nname: f\nnodes:\n  verdict: !str approve\n").root.child("nodes");
  const builder = new SourceCatalogBuilder();
  const descriptors = parseNodes(nodes, {
    readValue: (nodeId, cursor) =>
      readValueEntry(nodeId, cursor, {
        registry: new Map(),
        owner: ["f"],
        catalogBuilder: builder,
        diagnostics: LoadDiagnostics.fromCursor(cursor),
      }),
  });
  expect(descriptors.get("verdict")!.host).toBeNull();
});
