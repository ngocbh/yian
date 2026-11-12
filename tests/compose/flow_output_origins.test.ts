/** Authored Compose flow-output origins. */

import { expect, test } from "bun:test";

import { END_ID, type CompiledFlow, type FlowOutput } from "../../src/compile/model.ts";
import { assemble } from "../../src/compose/build/pipeline.ts";
import { CallableScope, loadFlow } from "../../src/compose/loader.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, type OriginResolver } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";

/** Return one declared output by name. */
function output(flow: CompiledFlow, name: string): FlowOutput {
  const matches = flow.outputs.filter((candidate) => candidate.name === name);
  expect(matches.length).toBe(1);
  return matches[0]!;
}

/** Expect one flow-output binding to resolve its authored value occurrence; return its origin. */
function expectOutputSite(
  catalog: OriginResolver,
  flow: CompiledFlow,
  name: string,
  cursor: SourceCursor,
  source: unknown,
): OriginRef {
  const found = output(flow, name);
  expect(found.binding.source).toEqual(source);
  expect(found.binding.origin).toBeInstanceOf(OriginRef);
  expect(flow.wiring.get(END_ID)!.get(name)).toBe(found.binding);

  const frame = catalog.resolve(found.binding.origin!)!;
  expect(frame).not.toBeNull();
  expect(frame.text).toBe(cursor.document!.text);
  expect(frame.range).toEqual(cursor.range!);
  return found.binding.origin!;
}

/** The baked child of `flow`'s CALL node `nodeId`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

test("a bare root output resolves its exact value and reaches END by identity", async () => {
  const text = `id: bare-output-origin
name: bare-output-origin
nodes:
  worker:
    kind: code
    code: return 1
    output: !int
output: \${worker.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);

  const cursor = document.root.child("output");
  expectOutputSite(loaded.originResolver!, loaded.compiled, "result", cursor, "${worker.output}");
});

test("mapped root, named def and inline outputs keep distinct value sites", async () => {
  const text = `id: lexical-output-origins
name: lexical-output-origins
defs:
  named:
    nodes:
      worker:
        kind: code
        code: return 2
        output: !int
    output: \${worker.output}
nodes:
  worker:
    kind: code
    code: return 1
    output: !int
  named_call:
    kind: call
    call: named
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: return 3
        output: !int
    output: \${worker.output}
output:
  root: \${worker.output}
  named: \${named_call.output}
  inline: \${inline_call.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const named = childOf(loaded.compiled, "named_call");
  const inline = childOf(loaded.compiled, "inline_call");
  const catalog = loaded.originResolver!;
  expect(catalog).not.toBeNull();
  const outputCursor = document.root.child("output");

  const origins = new Set(
    [
      expectOutputSite(catalog, loaded.compiled, "root", outputCursor.child("root"), "${worker.output}"),
      expectOutputSite(catalog, loaded.compiled, "named", outputCursor.child("named"), "${named_call.output}"),
      expectOutputSite(catalog, loaded.compiled, "inline", outputCursor.child("inline"), "${inline_call.output}"),
      expectOutputSite(
        catalog,
        named,
        "result",
        document.root.child("defs").child("named").child("output"),
        "${worker.output}",
      ),
      expectOutputSite(
        catalog,
        inline,
        "result",
        document.root.child("nodes").child("inline_call").child("output"),
        "${worker.output}",
      ),
    ].map((origin) => origin.key),
  );

  expect(origins.size).toBe(5);
});

test("output alias and merge sites keep distinct occurrences", async () => {
  const text = `id: aliased-output-origins
name: aliased-output-origins
x-shared: &shared
  merged: &value \${worker.output}
nodes:
  worker:
    kind: code
    code: return 1
    output: !int
output:
  <<: *shared
  alias: *value
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const outputCursor = document.root.child("output");
  const mergedCursor = outputCursor.child("merged");
  const aliasCursor = outputCursor.child("alias");

  const catalog = loaded.originResolver!;
  const mergedOrigin = expectOutputSite(catalog, loaded.compiled, "merged", mergedCursor, "${worker.output}");
  const aliasOrigin = expectOutputSite(catalog, loaded.compiled, "alias", aliasCursor, "${worker.output}");

  expect(mergedCursor.range).toEqual(aliasCursor.range!);
  expect(mergedCursor.path).not.toEqual(aliasCursor.path);
  expect(mergedOrigin.key).not.toBe(aliasOrigin.key);
});

test("output rewrites keep authored sites with the final binding sources", async () => {
  const text = `id: rewritten-output-origins
name: rewritten-output-origins
defs:
  helper:
    nodes:
      value:
        kind: code
        code: return 7
        output: !int
    output: \${value.output}
nodes:
  score:
    kind: code
    code: return 1
    output: !int
  gate:
    kind: case
    cases:
      - when: score.output > 0
        then: positive
    else: negative
  positive:
    kind: code
    code: return 1
    output: !int
  negative:
    kind: code
    code: return 0
    output: !int
output:
  via_call: call(helper)
  via_case: \${gate.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const outputCursor = document.root.child("output");

  const callOrigin = expectOutputSite(
    loaded.originResolver!,
    loaded.compiled,
    "via_call",
    outputCursor.child("via_call"),
    "${__call_0.output}",
  );
  const caseOrigin = expectOutputSite(
    loaded.originResolver!,
    loaded.compiled,
    "via_case",
    outputCursor.child("via_case"),
    "${positive.output | negative.output}",
  );
  expect(callOrigin.key).not.toBe(caseOrigin.key);
});

test("a compact root's generated re-export has no origin", async () => {
  const loaded = await loadFlow(`id: compact_root
name: compact_root
kind: code
code: return 1
output: !int
`);

  const found = output(loaded.compiled, "result");
  expect(found.binding.source).toBe("${compact_root.output}");
  expect(found.binding.origin).toBeNull();
  expect(loaded.compiled.wiring.get(END_ID)!.get("result")).toBe(found.binding);
});

test("a single-node def's generated re-export has no origin", async () => {
  const loaded = await loadFlow(`id: compact-def-output-origin
name: compact-def-output-origin
defs:
  compact:
    kind: code
    code: return 1
    output: !int
nodes:
  invoke:
    kind: call
    call: compact
output: \${invoke.output}
`);
  const child = childOf(loaded.compiled, "invoke");

  const found = output(child, "result");
  expect(found.binding.source).toBe("${compact.output}");
  expect(found.binding.origin).toBeNull();
  expect(child.wiring.get(END_ID)!.get("result")).toBe(found.binding);
});

test("virtual outputs neither mint nor erase programmatic origins", () => {
  const explicit = new OriginRef("programmatic-flow-output");
  const loaded = assemble({
    inputsSection: {},
    nodesCursor: SourceCursor.virtual({ worker: { kind: "code", code: "return 1", outputs: "int" } }),
    outputsSection: {
      virtual: "${worker.output}",
      programmatic: new Binding("${worker.output}", { origin: explicit }),
    },
    outputsCursor: SourceCursor.virtual({ virtual: "${worker.output}", programmatic: "${worker.output}" }),
    assertsSection: [],
    registry: new Map(),
    scope: new CallableScope({}, new Map(), null, {}),
    owner: ["root"],
  });

  const virtual = output(loaded.compiled, "virtual");
  const programmatic = output(loaded.compiled, "programmatic");
  expect(virtual.binding.origin).toBeNull();
  expect(programmatic.binding.origin).toEqual(explicit);
  expect(loaded.compiled.wiring.get(END_ID)!.get("virtual")).toBe(virtual.binding);
  expect(loaded.compiled.wiring.get(END_ID)!.get("programmatic")).toBe(programmatic.binding);
});

test("a physical output doesn't replace an explicit programmatic origin", () => {
  const explicit = new OriginRef("programmatic-flow-output");
  const document = SourceDocument.parse("output:\n  result: ${worker.output}\n");
  const loaded = assemble({
    inputsSection: {},
    nodesCursor: SourceCursor.virtual({ worker: { kind: "code", code: "return 1", outputs: "int" } }),
    outputsSection: { result: new Binding("${worker.output}", { origin: explicit }) },
    outputsCursor: document.root.child("output"),
    assertsSection: [],
    registry: new Map(),
    scope: new CallableScope({}, new Map(), null, {}),
    owner: ["root"],
  });

  const found = output(loaded.compiled, "result");
  expect(found.binding.origin).toEqual(explicit);
  expect(loaded.originResolver).toBeNull();
  expect(loaded.compiled.wiring.get(END_ID)!.get("result")).toBe(found.binding);
});
