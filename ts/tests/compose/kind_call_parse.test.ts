/**
 * The `call` kind's parse: a port of `tests/compose/test_kind_call_parse.py`.
 *
 * Ported: `test_call_parse_matches_descriptor_and_records_ranges`,
 * `test_call_parse_requires_call`, `test_call_parse_rejects_non_string_asserts`.
 *
 * Changed: Python compares the descriptor to a `CallDescriptor` with the parse metadata and the
 * compiler-owned `child_label` and `call_site` excluded; TS compares its class and fields. The
 * requires-call and asserts rows pin the whole message and its range (the node body; the
 * `asserts:` value), where Python matches a fragment.
 *
 * The harvested cases whose error the CALL parse raises are pinned in the TS mirrors of the Python
 * files they come from: `non_text_fields.test.ts` (`call:`), `node_input_shape.test.ts`
 * (`input:`), `node_when_hint.test.ts` (`when:`) and `engine/inline_composite_bodies.test.ts`
 * (the key check).
 */

import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { loadError } from "./declarations.ts";

const FLOW = `nodes:
  step:
    kind: call
    call: child_flow
    inputs:
      x: \${input.x}
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

test("call parse matches the descriptor and records ranges", () => {
  const desc = CallDescriptor.parse("step", readerFor(FLOW, "step"));

  expect(desc).toBeInstanceOf(CallDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(new CallDescriptor({ id: "step", call: "child_flow", inputs: { x: "${input.x}" } })),
  );
  expect(desc.kind).toBe("call");
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "call"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("call parse requires call", () => {
  const flow = "nodes:\n  step:\n    kind: call\n    inputs:\n      x: ${input.x}\n";

  const cursor = readerFor(flow, "step");

  const error = loadError(() => CallDescriptor.parse("step", cursor));

  expect(error.message).toBe("node 'step' (kind=call): missing required field 'call'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
});

test("call parse rejects non-string asserts", () => {
  const flow = "nodes:\n  step:\n    kind: call\n    call: child_flow\n    asserts:\n      - 5\n";

  const cursor = readerFor(flow, "step");

  const error = loadError(() => CallDescriptor.parse("step", cursor));

  expect(error.message).toBe("node 'step' (kind=call): `asserts:` must be a list of strings");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.rangeOf("asserts")!);
});
