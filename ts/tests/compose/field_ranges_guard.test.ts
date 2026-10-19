/**
 * Compose keeps a node's YAML ranges and field labels on the transient build result, never on
 * the built node: a port of `tests/compose/test_field_ranges_guard.py` through `StandInCode`,
 * parsed from the node's cursor with `input:` renamed to `inputs` as the parse phase does.
 *
 * Adapted: row 1 checks the `inputs` field range the template `parse` records, since the
 * per-binding `input` ranges come from `parseNodes`, which is not ported yet. Row 3 drops its
 * equality half (TS has no dataclass `==`).
 *
 * Later (T3-16): `test_case_generated_input_range_stays_in_build_result`. Later (this item's
 * parse/nodes task): restore Python's `("input", "topic")` line-10 assertion once `parseNodes`
 * records input ranges.
 */

import { expect, test } from "bun:test";

import { BuildContext, BuildResult, siteKey } from "../../src/compose/nodes/base.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { StandInCode } from "./stand_ins.ts";

const FLOW = `
id: ranges
name: ranges
input:
  topic: !str
nodes:
  n:
    kind: code
    input:
      topic: \${input.topic}
    output: int
    code: "return {'output': 1}"
output: \${n.output}
`;

const SURFACE = new Map([
  ["input", "inputs"],
  ["output", "outputs"],
]);

/** Parse `FLOW`'s node `n` with `StandInCode`. */
function parseN(): StandInCode {
  const document = SourceDocument.parse(FLOW);
  return StandInCode.parse("n", document.root.child("nodes").child("n").renamed(SURFACE));
}

const CTX = new BuildContext({ registry: new Map() });

test("building moves the descriptor's ranges to the build result only", () => {
  const descriptor = parseN();

  const result = descriptor.build(CTX);

  expect(descriptor.fieldRanges).toBeInstanceOf(Map);
  expect(result.nodeRange).toBe(descriptor.range);
  expect(result.fieldRanges).toEqual(descriptor.fieldRanges);
  expect(result.fieldRanges).not.toBe(descriptor.fieldRanges);
  expect(result.fieldRanges.get(siteKey("field", "inputs"))!.startLine).toBe(10);
  expect("range" in result.node).toBe(false);
  expect("fieldRanges" in result.node).toBe(false);
});

test("building copies the descriptor's field labels to the build result only", () => {
  const plain = parseN();
  const labels = new Map([["prompt", "node 'h' adaptive_questions.prompt"]]);
  const labelled = plain.replace({ fieldLabels: labels });

  const result = labelled.build(CTX);

  expect(result.fieldLabels).toEqual(labels);
  expect(result.fieldLabels).not.toBe(labelled.fieldLabels);
  expect("fieldLabels" in result.node).toBe(false);
  expect(plain.build(CTX).fieldLabels).toEqual(new Map());
});

test("siteName prefers the site's label", () => {
  const node = parseN().build(CTX).node;
  const result = new BuildResult({
    node,
    wiring: new Map(),
    fieldLabels: new Map([["input 'topic'", "node 'h' input 'topic'"]]),
  });

  expect(result.siteName("n", "input 'topic'")).toBe("node 'h' input 'topic'");
  expect(result.siteName("n", "input 'other'")).toBe("node 'n' input 'other'");
  expect(result.siteName("n", "prompt")).toBe("node 'n' prompt");
});
