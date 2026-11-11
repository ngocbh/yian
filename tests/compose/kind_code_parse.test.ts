/** The `code` kind's parse side. */

import { expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { checkCodeNode, checkFlowCode, CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { rejected } from "../helpers/run_state_fixtures.ts";
import { assembled, assembleText, type CaseRow, parsedNodes, runCase } from "../helpers/declarations.ts";

const FLOW = `nodes:
  verdict:
    kind: code
    inputs:
      rating: \${input.rating}
    outputs: str
    code: |
      x = inputs["rating"]
      return x
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata. */
function fieldsOf(desc: CodeDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

/** A context over an empty registry, with `baseDir` when given. */
const reading = (baseDir?: string | null) => new BuildContext({ registry: new Map(), baseDir });

test("code parse matches descriptor and records body range", () => {
  const desc = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));

  expect(desc).toBeInstanceOf(CodeDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new CodeDescriptor({
        id: "verdict",
        code: 'x = inputs["rating"]\nreturn x\n',
        inputs: { rating: "${input.rating}" },
        outputs: "str",
      }),
    ),
  );
  // Ranges are recorded for the node body and its fields (the code body carries its own).
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "code"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("code build returns code node and retains compose ranges", () => {
  const desc = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));
  const result = desc.build(new BuildContext({ registry: new Map(), sourceFile: "flow.yaml" }));

  expect(result.node).toBeInstanceOf(CodeNode);
  // The label a failing body's stack frames are named after.
  expect((result.node as CodeNode).sourceFile).toBe("flow.yaml");
  expect(result.node.outputContract).not.toBeNull();
  expect(result.nodeRange).toBe(desc.range);
  expect(result.wiring).toEqual(new Map([["rating", "${input.rating}"]]));
});

test("code build stamps descriptor origin", () => {
  const parsed = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));
  const origin = new OriginRef("root/node/verdict");
  const located = parsed.replace({ origin });

  const result = located.build(reading());

  expect(result.node.origin).toBe(origin);
  expect(() => parsed.replace({ origin: "root/node/verdict" as unknown as OriginRef })).toThrow(
    new TypeError("node descriptor origin must be an OriginRef or null"),
  );
});

/**
 * Parse node `nodeId` of `text`, attach its source origins, build it and run the CODE load check
 * with the catalog builder's registered-origin resolver; return the `LoadError` it rejects with.
 */
async function checkError(text: string, nodeId: string): Promise<LoadError> {
  const reader = SourceDocument.parse(text, "flow.yaml").root.child("nodes").child(nodeId);
  const builder = new SourceCatalogBuilder();
  const desc = CodeDescriptor.parse(nodeId, reader).withSourceOrigins(reader, {
    sitePath: ["root", "node", nodeId],
    catalogBuilder: builder,
  });
  const node = desc.build(reading()).node as CodeNode;
  const error = await rejected(
    checkCodeNode(node, {
      diagnostics: LoadDiagnostics.fromCursor(reader),
      resolveOrigin: (origin) => builder.resolveRegistered(origin),
    }),
  );
  expect(error).toBeInstanceOf(LoadError);
  return error as LoadError;
}

test("code build frames inline syntax error at yaml line", async () => {
  // `code: |` indicator is on line 7; the body's first (and only) line is line 8.
  const text = `nodes:
  broken:
    kind: code
    inputs:
      rating: \${input.rating}
    outputs: str
    code: |
      return (
`;

  const error = await checkError(text, "broken");

  // Body line 1 selects the first exact constructed CODE line (YAML line 8).
  expect(error.diagnostic.frames.length).toBe(1);
  expect(error.diagnostic.frames[0]!.range!.startLine).toBe(8);
  expect(error.diagnostic.frames[0]!.text).toBe(text);
  expect(error.diagnostic.message.startsWith("node 'broken': ")).toBe(true);
});

test.each([
  ["quoted", '    code: "return (\\n"\n'],
  ["folded", "    code: >-\n      return (\n"],
])("nonliteral syntax error uses whole code scalar: %s", async (_id, codeField) => {
  const text =
    "id: syntax-fallback\nname: syntax-fallback\nnodes:\n  broken:\n    kind: code\n    outputs: str\n" +
    codeField +
    "output: ${broken.output}\n";
  const codeCursor = SourceDocument.parse(text, "flow.yaml").root.child("nodes").child("broken").child("code");

  const error = await checkError(text, "broken");

  expect(error.diagnostic.message.startsWith("node 'broken': ")).toBe(true);
  expect(error.diagnostic.frames[0]!.range).toEqual(codeCursor.range!);
});

test("build passes the base directory, the code origins and the title, and compiles nothing", () => {
  const codeOrigin = new OriginRef("code");
  const codeLineOrigins = [new OriginRef("code/line/0")];
  const desc = new CodeDescriptor({
    id: "a",
    code: "return (",
    nodeName: "Broken",
    codeOrigin,
    codeLineOrigins,
  });

  const node = desc.build(reading("/flows")).node as CodeNode;

  expect(node.baseDir).toBe("/flows");
  expect(node.codeOrigin).toBe(codeOrigin);
  expect(node.codeLineOrigins).toEqual(codeLineOrigins);
  expect(node.title).toBe("Broken");
  expect(node.code).toBe("return (");
  expect((desc.build(reading()).node as CodeNode).baseDir).toBeNull();
});

test("checkFlowCode reaches a CODE node inside a child flow", async () => {
  const text = `id: child-code-check
name: child-code-check
nodes:
  first:
    kind: code
    output: !int
    code: return 1
  invoke:
    kind: call
    nodes:
      worker:
        kind: code
        output: !int
        code: |-
          const value = 1
          return (
    output: \${worker.output}
output: \${invoke.output}
`;
  const document = SourceDocument.parse(text, "flow.yaml");
  const lineRange = document.root.child("nodes").child("invoke").child("nodes").child("worker").child("code")
    .scalarSource!.constructedLineRanges![1]!;
  const { flow, builder } = assembleText(text, "flow.yaml");

  const error = await rejected(
    checkFlowCode(flow.compiled, {
      diagnostics: LoadDiagnostics.fromDocument(document),
      resolveOrigin: (origin) => builder.resolveRegistered(origin),
    }),
  );

  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).diagnostic.message.startsWith("node 'worker': ")).toBe(true);
  expect((error as LoadError).diagnostic.frames.map((frame) => frame.range)).toEqual([lineRange]);
});

// ---------- harvested conformance cases ----------
// Covered by the conformance cases `load-leaves/seed_e30_inline_syntax_error`,
// `load-leaves/seed_e32_malformed_code_ref` and
// `load-leaves/compose_kind_code_parse__quoted_body_syntax_error_frames_the_whole_scalar`, whose
// errors come from compiling the CODE body, which the runner's `load` kind runs.

test.each<CaseRow>([
  { id: "load-leaves/seed_e11_field_not_allowed", call: parsedNodes },
  { id: "load-leaves/seed_e12_missing_required_field", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  { id: "load-agent/seed_01_structured_agent", call: assembled },
  { id: "load-agent/seed_03_research_one", call: assembled },
  { id: "load-leaves/seed_29_inline_code", call: assembled },
  { id: "load-leaves/seed_30_reference_code", call: assembled },
  { id: "load-leaves/seed_e31_inline_no_return", call: assembled },
])("assembles: $id", runCase);
