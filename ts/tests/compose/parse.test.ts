/**
 * The parse phase's `nodes:` mapping pass: a port of `tests/compose/test_parse.py`, with
 * `StandInAgent` and `StandInCase` for the agent and case kinds.
 *
 * Changed: `test_case_rejects_numeric_key` asserts "field '1' is not allowed", since TS keeps the
 * key `1` as the text written. Python compares the descriptor to an `AgentDescriptor` with
 * `field_ranges` excluded; TS compares its class and fields.
 *
 * TS-only: the `input:` binding ranges and positional `asserts:` ranges stamped after the kind
 * parse, which Python tests through later phases.
 */

import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { loadError } from "./declarations.ts";
import { StandInAgent, StandInCase, StandInCode, withKinds } from "./stand_ins.ts";

/** Return the physical cursor over a flow document's `nodes:` mapping. */
function nodesReader(text: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes");
}

const FLOW = `id: f
name: f
nodes:
  writer:
    kind: agent
    prompt: "Write about \${topic}"
    input:
      topic: \${input.topic}
    output: string
output: \${writer.output}
`;

test("parse nodes uses span tree", () => {
  const desc = withKinds([StandInAgent], () => parseNodes(nodesReader(FLOW)).get("writer")!);

  // The singular `input:`/`output:` source keys are back-mapped to the plural fields.
  expect(desc).toBeInstanceOf(StandInAgent);
  expect(desc.id).toBe("writer");
  expect((desc as StandInAgent).prompt).toBe("Write about ${topic}");
  expect(desc.inputs).toEqual({ topic: "${input.topic}" });
  expect(desc.outputs).toBe("string");
  // Located: the physical cursor stamps ranges onto the descriptor and its fields.
  expect(desc.range).not.toBeNull();
  expect(desc.fieldRanges.get(siteKey("field", "prompt"))!.startLine).toBe(6);
  // The back-mapped plural fields resolve their source ranges (via the rewritten key).
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeDefined();
  expect(desc.fieldRanges.get(siteKey("field", "outputs"))).toBeDefined();
});

test("parse nodes unknown kind is located", () => {
  const text = "id: f\nname: f\nnodes:\n  n:\n    kind: wizard\n    output: str\noutput: 1\n";
  const error = loadError(() => parseNodes(nodesReader(text)));

  expect(error.message).toContain("wizard");
  expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
});

test("parse nodes reserved node id is loud", () => {
  const text = "id: f\nname: f\nnodes:\n  input:\n    kind: agent\n    prompt: hi\noutput: 1\n";

  expect(loadError(() => parseNodes(nodesReader(text))).message).toContain("reserved");
});

test("parse nodes virtual cursor", () => {
  // A virtual cursor (defs / compact / in-memory) parses the already-plural body with no source
  // ranges: descriptors still build, ranges are null.
  const reader = SourceCursor.virtual({
    w: { kind: "agent", prompt: "hi", inputs: { x: "${input.x}" }, outputs: "str" },
  });
  const desc = withKinds([StandInAgent], () => parseNodes(reader).get("w")!);

  expect(desc).toBeInstanceOf(StandInAgent);
  expect(desc.id).toBe("w");
  expect((desc as StandInAgent).prompt).toBe("hi");
  expect(desc.inputs).toEqual({ x: "${input.x}" });
  expect(desc.outputs).toBe("str");
  expect(desc.range).toBeNull();
  expect(desc.fieldRanges.size).toBe(0);
});

test("parse nodes normalizes without mutating cursor", () => {
  const nodes = SourceDocument.parse(FLOW).root.child("nodes");
  const before = nodes.value;

  const desc = withKinds([StandInAgent], () => parseNodes(nodes).get("writer")!);

  expect(desc.inputs).toEqual({ topic: "${input.topic}" });
  expect(desc.outputs).toBe("string");
  expect(nodes.value).toEqual(before);
  expect(nodes.child("writer").has("input")).toBe(true);
  expect(nodes.child("writer").has("inputs")).toBe(false);
});

test("case on key is text", () => {
  const source = `nodes:
  route:
    kind: case
    on: \${score.output}
    cases:
      - when: high
        then: hot
`;
  const nodes = SourceDocument.parse(source).root.child("nodes");
  const route = nodes.child("route");

  const desc = withKinds([StandInCase], () => parseNodes(nodes).get("route")!) as StandInCase;

  expect(desc.on).toBe("${score.output}");
  expect(route.has("on")).toBe(true);
});

test("case rejects numeric key", () => {
  const source = `nodes:
  route:
    kind: case
    on: \${score.output}
    1: ignored
    cases: []
`;
  const nodes = SourceDocument.parse(source).root.child("nodes");
  const route = nodes.child("route");

  const error = loadError(() => withKinds([StandInCase], () => parseNodes(nodes)));

  // TS keeps the key `1` as the text written, so the message quotes it where Python's int key
  // reads `field 1`.
  expect(error.message).toContain("field '1' is not allowed");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(route.rangeOf("1")!);
});

test("input binding and assert ranges are stamped after the kind parse", () => {
  const source = `nodes:
  x:
    kind: code
    code: m:f
    input:
      n: \${input.v}
    asserts:
      - "output >= 0"
      - "output >= 0"
`;
  const nodes = SourceDocument.parse(source).root.child("nodes");

  const desc = withKinds([StandInCode], () => parseNodes(nodes).get("x")!);

  expect(desc.fieldRanges.get(siteKey("input", "n"))!.startLine).toBe(6);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeDefined();
  expect(desc.assertRanges.map((range) => range!.startLine)).toEqual([8, 9]);
  // The node-level range is the node's key line.
  expect(desc.range).toEqual(nodes.keyRangeOf("x"));
});
