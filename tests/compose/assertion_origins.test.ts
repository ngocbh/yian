/** Authored Compose assertion origins. */

import { expect, test } from "bun:test";

import { callSubgraph } from "../../src/compile/expand.ts";
import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { assemble } from "../../src/compose/build/pipeline.ts";
import { CallableScope, loadFlow } from "../../src/compose/loader.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, type OriginResolver } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Assertion, isExactly } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { loadConformanceFlow } from "./declarations.ts";

/** Check that one semantic assertion resolves its exact authored list item; return its origin. */
function expectAssertionSite(
  catalog: OriginResolver,
  assertion: Assertion,
  cursor: SourceCursor,
  options: { expression?: string; display?: string } = {},
): OriginRef {
  expect(isExactly(assertion, Assertion)).toBe(true);
  if (options.expression !== undefined) expect(assertion.expression).toBe(options.expression);
  if (options.display !== undefined) expect(assertion.display).toBe(options.display);
  expect(assertion.origin).toBeInstanceOf(OriginRef);

  const frame = catalog.resolve(assertion.origin!);
  expect(frame).not.toBeNull();
  expect(frame!.text).toBe(cursor.document!.text);
  expect(frame!.range).toEqual(cursor.range!);
  return assertion.origin!;
}

/** The number of distinct origins in `origins`. */
const distinct = (origins: readonly OriginRef[]) => new Set(origins.map((origin) => origin.key)).size;

/** The baked child of CALL node `nodeId` in `flow`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

test("root flow assertions keep indexed sites through rewrites", async () => {
  // Flow lowering keeps duplicate, pre/post, and inline-call occurrences exact.
  const text = `id: root-assertion-origins
name: root-assertion-origins
input:
  n: !int
defs:
  truthy:
    kind: code
    code: return true
    output: bool
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: \${worker.output}
asserts:
  - input.n >= 0
  - input.n >= 0
  - call(truthy)
  - worker.output >= 0
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const catalog = loaded.originResolver!;
  expect(catalog).not.toBeNull();
  const assertionCursors = document.root.child("asserts");
  const start = loaded.compiled.nodes.get(START_ID)!;
  const end = loaded.compiled.nodes.get(END_ID)!;

  expect(start.postAsserts.map((assertion) => assertion.display)).toEqual(["input.n >= 0", "input.n >= 0"]);
  expect(start.postAsserts.map((assertion) => assertion.expression)).toEqual(["output.n >= 0", "output.n >= 0"]);
  expect(end.postAsserts.map((assertion) => assertion.display)).toEqual(["call(truthy)", "worker.output >= 0"]);
  expect(end.postAsserts[0]!.expression).toBe("\${__assert___call_0}");

  const ordered = [
    [start.postAsserts[0]!, assertionCursors.child(0)],
    [start.postAsserts[1]!, assertionCursors.child(1)],
    [end.postAsserts[0]!, assertionCursors.child(2)],
    [end.postAsserts[1]!, assertionCursors.child(3)],
  ] as const;
  const origins = ordered.map(([assertion, cursor]) => expectAssertionSite(catalog, assertion, cursor));

  expect(distinct(origins)).toBe(4);
});

test("alias occurrences mint distinct node and flow assertion origins", async () => {
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
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const catalog = loaded.originResolver!;

  const nodeAssert = (nodeId: string) => document.root.child("nodes").child(nodeId).child("asserts").child(0);
  const [firstCursor, secondCursor] = [nodeAssert("first"), nodeAssert("second")];
  const nodeOrigins = (
    [
      ["first", firstCursor],
      ["second", secondCursor],
    ] as const
  ).map(([nodeId, cursor]) => expectAssertionSite(catalog, loaded.compiled.nodes.get(nodeId)!.postAsserts[0]!, cursor));
  expect(firstCursor.range).toEqual(secondCursor.range);
  expect(firstCursor.path).not.toEqual(secondCursor.path);
  expect(distinct(nodeOrigins)).toBe(2);

  const flowCursors = document.root.child("asserts");
  const startAssertions = loaded.compiled.nodes.get(START_ID)!.postAsserts;
  const flowOrigins = startAssertions.map((assertion, index) =>
    expectAssertionSite(catalog, assertion, flowCursors.child(index)),
  );
  expect(flowCursors.child(0).range).toEqual(flowCursors.child(1).range);
  expect(flowCursors.child(0).path).not.toEqual(flowCursors.child(1).path);
  expect(distinct(flowOrigins)).toBe(2);
});

test("same node assertion text has distinct root, def, inline and flat sites", async () => {
  // Lexically equal node assertions retain four physical identities.
  const assertionText = "output >= 0";
  const text = `id: lexical-node-assertions
name: lexical-node-assertions
defs:
  named:
    nodes:
      worker:
        kind: code
        code: return 2
        output: int
        asserts:
          - ${assertionText}
    output: \${worker.output}
  flat:
    kind: code
    code: return 3
    output: int
    asserts:
      - ${assertionText}
nodes:
  worker:
    kind: code
    code: return 1
    output: int
    asserts:
      - ${assertionText}
  named_call:
    kind: call
    call: named
  flat_call:
    kind: call
    call: flat
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: return 4
        output: int
        asserts:
          - ${assertionText}
    output: \${worker.output}
output:
  root: \${worker.output}
  named: \${named_call.output}
  flat: \${flat_call.output}
  inline: \${inline_call.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const catalog = loaded.originResolver!;
  expect(catalog).not.toBeNull();
  const root = document.root;
  const nodeAssert = (cursor: SourceCursor) => cursor.child("asserts").child(0);

  const sites = [
    [loaded.compiled.nodes.get("worker")!.postAsserts[0]!, nodeAssert(root.child("nodes").child("worker"))],
    [
      childOf(loaded.compiled, "named_call").nodes.get("worker")!.postAsserts[0]!,
      nodeAssert(root.child("defs").child("named").child("nodes").child("worker")),
    ],
    [
      childOf(loaded.compiled, "inline_call").nodes.get("worker")!.postAsserts[0]!,
      nodeAssert(root.child("nodes").child("inline_call").child("nodes").child("worker")),
    ],
    [
      childOf(loaded.compiled, "flat_call").nodes.get("flat")!.postAsserts[0]!,
      nodeAssert(root.child("defs").child("flat")),
    ],
  ] as const;
  const origins = sites.map(([assertion, cursor]) =>
    expectAssertionSite(catalog, assertion, cursor, { expression: assertionText, display: assertionText }),
  );

  expect(distinct(origins)).toBe(4);
});

test("compact root node assertion uses its projected physical item", async () => {
  // Compact projection gives its synthesized node the top-level assert site.
  const text = `id: compact_assertion_origin
name: compact_assertion_origin
kind: code
code: return 1
output: int
asserts:
  - output >= 0
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadFlow(text);
  const node = loaded.compiled.nodes.get("compact_assertion_origin")!;
  const cursor = document.root.child("asserts").child(0);

  expectAssertionSite(loaded.originResolver!, node.postAsserts[0]!, cursor, {
    expression: "output >= 0",
    display: "output >= 0",
  });
});

test("duplicate loop assertions keep their sites when the driver respawns", async () => {
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
  const document = SourceDocument.parse(text);
  const loaded = await loadConformanceFlow(text);
  const catalog = loaded.originResolver!;
  const drive = loaded.compiled.nodes.get("drive") as LoopNode;
  const cursors = document.root.child("nodes").child("drive").child("asserts");

  const origins = drive.postAsserts.map((assertion, index) =>
    expectAssertionSite(catalog, assertion, cursors.child(index)),
  );
  expect(distinct(origins)).toBe(2);

  const respawned = drive.respawn(1);

  expect(respawned.postAsserts).toEqual(drive.postAsserts);
  expect(respawned.postAsserts).not.toBe(drive.postAsserts);
  expect(respawned.postAsserts.map((assertion) => assertion.origin)).toEqual(
    drive.postAsserts.map((assertion) => assertion.origin),
  );
});

test("def boundary and call relocation reuse authored assertion carriers", async () => {
  // Def START/END and a relocated CALL post-assert keep their source sites.
  const text = `id: nested-assertion-origins
name: nested-assertion-origins
defs:
  child:
    input:
      n: !int
    nodes:
      worker:
        kind: code
        code: ./fixtures/compose_codefns.ts:double
        input:
          n: \${input.n}
        output: int
    output: \${worker.output}
    asserts:
      - input.n >= 0
      - worker.output >= 0
nodes:
  invoke:
    kind: call
    call: child
    input:
      n: 1
    asserts:
      - n >= 0 and output >= 0
output: \${invoke.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadConformanceFlow(text);
  const catalog = loaded.originResolver!;
  expect(catalog).not.toBeNull();
  const invoke = loaded.compiled.nodes.get("invoke") as CallNode;
  const child = invoke.child as CompiledFlow;
  const childAsserts = document.root.child("defs").child("child").child("asserts");
  const callCursor = document.root.child("nodes").child("invoke").child("asserts").child(0);

  const startAssertion = child.nodes.get(START_ID)!.postAsserts[0]!;
  const endAssertion = child.nodes.get(END_ID)!.postAsserts[0]!;
  const callAssertion = invoke.postAsserts[0]!;

  expectAssertionSite(catalog, startAssertion, childAsserts.child(0));
  expectAssertionSite(catalog, endAssertion, childAsserts.child(1));
  const callOrigin = expectAssertionSite(catalog, callAssertion, callCursor);

  const cloned = callSubgraph(child, "invoke", { n: 1 }, {
    postAsserts: invoke.postAsserts,
    inputNames: new Set(invoke.params!.map((parameter) => parameter.name)),
  });
  const relocated = cloned.nodes.get(cloned.endId)!.postAsserts.at(-1)!;

  expect(relocated).not.toBe(callAssertion);
  expect(relocated.origin).toEqual(callOrigin);
  expect(relocated.display).toBe(callAssertion.display);
  expect(relocated.expression).toBe("__assert_input_n >= 0 and output >= 0");
  expect(catalog.resolve(relocated.origin!)).toEqual(catalog.resolve(callOrigin));
});

test("virtual assertions do not mint or erase programmatic origins", () => {
  // Source-free assembly leaves raw assertions without an origin and explicit origins intact.
  const raw = "input.n >= 0";
  const explicit = new OriginRef("programmatic-assertion");
  const programmatic = new Assertion("worker.output >= 0", "programmatic worker assertion", { origin: explicit });
  const loaded = assemble({
    inputsSection: { n: "int" },
    nodesCursor: SourceCursor.virtual({ worker: { kind: "code", code: "return 1", outputs: "int" } }),
    outputsSection: "${worker.output}",
    assertsSection: [raw, programmatic],
    assertsCursor: SourceCursor.virtual([raw, programmatic.expression]),
    registry: new Map(),
    scope: new CallableScope({}, new Map(), null, {}),
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
