/** The `tool` kind's parse side. */

import { describe, expect, test } from "bun:test";

import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../helpers/declarations.ts";

const FLOW = `nodes:
  fetch:
    kind: tool
    tool_id: http_get
    inputs:
      url: \${input.url}
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata. */
function fieldsOf(desc: ToolDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

/** A context over an empty registry. */
const reading = () => new BuildContext({ registry: new Map() });

test("tool parse matches descriptor and records ranges", () => {
  const desc = ToolDescriptor.parse("fetch", readerFor(FLOW, "fetch"));

  expect(desc).toBeInstanceOf(ToolDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(new ToolDescriptor({ id: "fetch", toolRef: { group: null, name: "http_get" }, inputs: { url: "${input.url}" } })),
  );
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "tool_id"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("tool build returns tool node and wires inputs", () => {
  const desc = ToolDescriptor.parse("fetch", readerFor(FLOW, "fetch"));
  const result = desc.build(reading());

  expect(result.node).toBeInstanceOf(ToolNode);
  expect((result.node as ToolNode).toolId).toBe("http_get");
  // A tool's sink surface is the base's `inputs` map; the wiring carries those sources.
  expect(result.wiring).toEqual(new Map([["url", "${input.url}"]]));
  expect(result.nodeRange).toBe(desc.range);
});

test("a tool node with an authored output loads", () => {
  const { flow } = assembleText(`id: count
name: count
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: !int
output: \${place.output}
`);

  expect((flow.compiled.nodes.get("place")!.outputContract as ValueContract).type).toEqual(
    Type.scalar(ValueKind.INTEGER),
  );
});

/** Build a TOOL `place` whose `outputs:` is `outputYaml`. */
function builtTool(outputYaml: string): ToolNode {
  const text = `nodes:
  place:
    kind: tool
    tool_id: place_order
    inputs:
      qty: 5
    outputs: ${outputYaml}
`;
  return ToolDescriptor.parse("place", readerFor(text, "place")).build(reading()).node as ToolNode;
}

test("tool build sets a scalar output contract", () => {
  const node = builtTool("int");

  expect(node.outputContract).toBeInstanceOf(ValueContract);
  expect((node.outputContract as ValueContract).type).toEqual(Type.scalar(ValueKind.INTEGER));
});

test("tool build sets a record output contract", () => {
  const fields = (builtTool("{qty: int}").outputContract as ValueContract).type.fields!;

  expect([...fields.keys()]).toEqual(["qty"]);
  expect(fields.get("qty")).toEqual(Type.scalar(ValueKind.INTEGER));
});

// Refused at build, which knows whether the group is a memory input (`memory_grants.test.ts`).
test("a tool_id group tool that would show under a reserved name is refused in the form written", () => {
  const error = loadError(() =>
    assembleText("id: f\nname: f\nnodes:\n  a:\n    kind: tool\n    tool_id: {load: skill}\noutput: null\n"),
  );

  expect(error.message).toBe(
    "node 'a' (kind=tool): `tool_id: {load: skill}` would show as `load_skill`, a name yian reserves; " +
      "rename the group or the tool",
  );
});

describe("retries", () => {
  /** Parse node `a` of a TOOL flow with `line` added. */
  const parseA = (line: string) =>
    ToolDescriptor.parse("a", readerFor(`nodes:\n  a:\n    kind: tool\n    tool_id: t\n${line}`, "a"));

  test("a count past the largest safe integer is refused", () => {
    const error = loadError(() => parseA("    retries: 9007199254740992\n"));

    expect(error.message).toBe(
      "node 'a' (kind=tool): `retries:` must be at most 9007199254740991, got 9007199254740992",
    );
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(5);
  });
});

test("build passes the title and the retries to the node", () => {
  const node = new ToolDescriptor({ id: "a", toolRef: { group: null, name: "t" }, nodeName: "Fetch", retries: 2 }).build(reading())
    .node as ToolNode;

  expect(node.title).toBe("Fetch");
  expect(node.retries).toBe(2);
  expect(node.outputContract).toBeNull();
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-leaves/seed_08_tool_news", call: assembled },
  { id: "load-leaves/tool_grants__a_tool_node_names_a_group_tool", call: assembled },
  { id: "load-leaves/tool_grants__a_tool_id_mapping_with_two_keys_is_refused", call: assembled },
  { id: "load-leaves/tool_grants__a_tool_id_group_with_a_list_is_refused", call: assembled },
  { id: "load-leaves/tool_grants__a_built_in_in_a_group_is_refused", call: assembled },
])("assembles: $id", runCase);
