/** The parse phase's `nodes:` mapping pass. */

import { afterEach, expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { siteKey } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { flowNodesCursor } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { loadError, nodesOf, runTextCase } from "./declarations.ts";

afterEach(() => forget());

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
  const desc = parseNodes(nodesReader(FLOW)).get("writer")!;

  // The singular `input:`/`output:` source keys are back-mapped to the plural fields.
  expect(desc).toBeInstanceOf(AgentDescriptor);
  expect(desc.id).toBe("writer");
  expect((desc as AgentDescriptor).prompt).toBe("Write about ${topic}");
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
  const desc = parseNodes(reader).get("w")!;

  expect(desc).toBeInstanceOf(AgentDescriptor);
  expect(desc.id).toBe("w");
  expect((desc as AgentDescriptor).prompt).toBe("hi");
  expect(desc.inputs).toEqual({ x: "${input.x}" });
  expect(desc.outputs).toBe("str");
  expect(desc.range).toBeNull();
  expect(desc.fieldRanges.size).toBe(0);
});

test("parse nodes normalizes without mutating cursor", () => {
  const nodes = SourceDocument.parse(FLOW).root.child("nodes");
  const before = nodes.value;

  const desc = parseNodes(nodes).get("writer")!;

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

  const desc = parseNodes(nodes).get("route") as CaseDescriptor;

  expect(desc).toBeInstanceOf(CaseDescriptor);
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

  const error = loadError(() => parseNodes(nodes));

  // The key `1` stays the text written, so the message quotes it.
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

  const desc = parseNodes(nodes).get("x")!;

  expect(desc.fieldRanges.get(siteKey("input", "n"))!.startLine).toBe(6);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeDefined();
  expect(desc.assertRanges.map((range) => range!.startLine)).toEqual([8, 9]);
  // The node-level range is the node's key line.
  expect(desc.range).toEqual(nodes.keyRangeOf("x"));
});

/** A node `a` of `kind` whose body is `body` and whose `input:` binds `name`. */
function inputNamed(kind: string, name: string, body = "code: return 1"): string {
  return `id: f
name: f
nodes:
  a:
    kind: ${kind}
    ${body}
    input:
      ${name}: \${input.x}
output: \${a.output}
`;
}

test("a node input named inf is refused at its key", () => {
  const text = inputNamed("code", "inf");
  const error = loadError(() => nodesOf(text));

  expect(error.message).toBe(
    "node 'a' input 'inf': `inf` is infinity in an expression, so ${inf} can't read it; rename it",
  );
  const key = SourceDocument.parse(text).root.child("nodes").child("a").child("input");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(key.keyRangeOf("inf")!);
});

test("a generated input named inf is refused at its node", () => {
  // A compact flow's auto-wired bindings are virtual, so the error falls back to the node body.
  const document = SourceDocument.parse("id: f\nname: f\nkind: code\ncode: return 1\n");
  const nodes = { f: { kind: "code", code: "return 1", inputs: { inf: "${input.inf}" } } };
  const cursor = flowNodesCursor(document, nodes);
  const error = loadError(() => parseNodes(cursor));

  expect(error.message).toContain("node 'f' input 'inf': `inf` is infinity");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.child("f").range!);
});

test("an input named inf parses where the inputs aren't locals", () => {
  // A map's input names are its child's parameters, never read as bare names.
  const text = inputNamed("map", "inf", "call: child\n    over: ${input.xs}");

  expect(nodesOf(text).get("a")!.inputs).toEqual({ inf: "${input.x}" });
});

test("an input named infinity parses", () => {
  const desc = nodesOf(inputNamed("code", "infinity")).get("a")!;

  expect(desc.inputs).toEqual({ infinity: "${input.x}" });
});

test("parseNodes: load-core/compose_infinity_names__a_node_input_named_inf", () =>
  runTextCase({
    id: "load-core/compose_infinity_names__a_node_input_named_inf",
    call: (text, label) => nodesOf(text, [], label),
  }));
