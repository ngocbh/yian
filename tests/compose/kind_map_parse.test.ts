/** The `map` kind's parse. */

import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { type CaseRow, loadError, parsedNodes, runCase } from "../helpers/declarations.ts";

const FLOW = `nodes:
  fan:
    kind: map
    call: child_flow
    over: \${input.items}
    inputs:
      x: \${item}
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata. */
function fieldsOf(desc: CallDescriptor): Record<string, unknown> {
  const {
    range: _range,
    fieldRanges: _fieldRanges,
    assertRanges: _assertRanges,
    childLabel: _childLabel,
    callSite: _callSite,
    ...fields
  } = desc;
  return fields;
}

/** Return the cursor over node `fan` of a map whose `over:` is written `over`. */
function overReader(over: string): SourceCursor {
  return readerFor(`nodes:\n  fan:\n    kind: map\n    call: child_flow\n    over: ${over}\n`, "fan");
}

/** Parse node `fan` of a map whose `over:` is written `over`. */
function parseOver(over: string): MapDescriptor {
  return MapDescriptor.parse("fan", overReader(over));
}

test("map parse matches the descriptor and records ranges", () => {
  const desc = MapDescriptor.parse("fan", readerFor(FLOW, "fan"));

  expect(desc).toBeInstanceOf(MapDescriptor);
  expect(desc).toBeInstanceOf(CallDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new MapDescriptor({ id: "fan", call: "child_flow", over: "${input.items}", inputs: { x: "${item}" } }),
    ),
  );
  expect(desc.kind).toBe("map");
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "call"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "over"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("map parse requires over", () => {
  const flow = "nodes:\n  fan:\n    kind: map\n    call: child_flow\n    inputs:\n      x: ${item}\n";

  const cursor = readerFor(flow, "fan");

  const error = loadError(() => MapDescriptor.parse("fan", cursor));

  expect(error.message).toBe("node 'fan' (kind=map): missing required field 'over'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
});

test("map parse records parallel", () => {
  const flow =
    "nodes:\n  fan:\n    kind: map\n    call: child_flow\n    over: ${input.items}\n    parallel: true\n";

  const desc = MapDescriptor.parse("fan", readerFor(flow, "fan"));

  expect(desc.parallel).toBe(true);
  expect(desc.fieldRanges.get(siteKey("field", "parallel"))).toBeInstanceOf(SourceRange);
});

test("an unquoted call( over passes the parse", () => {
  expect(parseOver("call(g").over).toBe("call(g");
});

test("a quoted call( over is text", () => {
  const cursor = overReader('"call(g)"');

  const error = loadError(() => MapDescriptor.parse("fan", cursor));

  expect(error.message).toBe(
    "node 'fan' (kind=map): `over:` must be a list, but YAML read it as text; " +
      "write a list, or a reference such as `over: ${input.items}`",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.rangeOf("over")!);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-map-call/compose_map_wait_source_types__date_over_literal", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__int_over_literal", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__null_over_is_empty", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__quoted_call_over_is_text", call: parsedNodes },
])("parseNodes: $id", runCase);
