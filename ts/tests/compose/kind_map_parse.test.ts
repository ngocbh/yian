/**
 * The `map` kind's parse: a port of `tests/compose/test_kind_map_parse.py`.
 *
 * Ported: `test_map_parse_matches_descriptor_and_records_ranges`,
 * `test_map_parse_requires_over`, `test_map_parse_records_parallel`.
 *
 * Changed: Python compares the descriptor to a `MapDescriptor` with the parse metadata and the
 * compiler-owned `child_label` and `call_site` excluded; TS compares its class and fields. The
 * requires-over row pins the whole message, where Python matches a fragment.
 *
 * TS-only: an unquoted `over:` starting with `call(` passes the parse (the loader lifts the
 * directive later, so even an unbalanced one passes here), while a quoted one is text and fails.
 *
 * Pinned at the end (`parsedNodes`): the harvested `over:` literal cases of
 * `tests/compose/test_map_wait_source_types.py`, which has no TS mirror;
 * `load-map-call/compose_map_wait_source_types__date_over_literal` reads the date as text (rule
 * `dates`).
 */

import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

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

/** Return a descriptor's fields, leaving out what Python's `compare=False` leaves out. */
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

/** Parse node `fan` of a map whose `over:` is written `over`. */
function parseOver(over: string): MapDescriptor {
  const text = `nodes:\n  fan:\n    kind: map\n    call: child_flow\n    over: ${over}\n`;
  return MapDescriptor.parse("fan", readerFor(text, "fan"));
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

  const error = loadError(() => MapDescriptor.parse("fan", readerFor(flow, "fan")));

  expect(error.message).toBe("node 'fan' (kind=map): missing required field 'over'");
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
  const error = loadError(() => parseOver('"call(g)"'));

  expect(error.message).toBe(
    "node 'fan' (kind=map): `over:` must be a list, but YAML read it as text; " +
      "write a list, or a reference such as `over: ${input.items}`",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  // tests/compose/test_map_wait_source_types.py
  { id: "load-map-call/compose_map_wait_source_types__date_over_literal", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__int_over_literal", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__null_over_is_empty", call: parsedNodes },
  { id: "load-map-call/compose_map_wait_source_types__quoted_call_over_is_text", call: parsedNodes },
])("parseNodes: $id", runCase);
