/**
 * The `loop` kind's parse: a port of `tests/compose/test_kind_loop_parse.py`.
 *
 * Ported: `test_loop_parse_matches_descriptor_and_records_ranges` (`max` is `10n`),
 * `test_loop_parse_requires_call`, `test_loop_parse_maps_until_keyword`,
 * `test_loop_rejects_a_non_string_predicate_at_its_field`.
 *
 * Changed: Python compares the descriptor to a `LoopDescriptor` with the parse metadata and the
 * compiler-owned `child_label` excluded; TS compares its class and fields. The requires-call row
 * pins the whole message and its range (the node body), where Python matches a fragment. The
 * non-string predicate rows parse the flow's nodes, where Python loads it, and spell the value in
 * the text form (`(got true)`, `(got ["x"])`).
 *
 * TS-only: a lone surrogate in `while:` is refused at the value.
 *
 * Later (T3-16, the LOOP build): `test_loop_rejects_an_empty_predicate_at_load`,
 * `test_loop_reports_a_malformed_predicate_at_its_field`.
 *
 * Pinned at the end (`parsedNodes`): the harvested
 * `load-loop/compose_kind_loop_parse__a_list_until_predicate_is_rejected` and
 * `__a_true_while_predicate_is_rejected`, both from this file's Python source.
 */

import { expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { type CaseRow, loadError, nodesOf, parsedNodes, runCase } from "./declarations.ts";

const FLOW = `nodes:
  drive:
    kind: loop
    call: body_flow
    inputs:
      acc: \${input.seed}
    while: \${carry.go}
    max: 10
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out what Python's `compare=False` leaves out. */
function fieldsOf(desc: LoopDescriptor): Record<string, unknown> {
  const {
    range: _range,
    fieldRanges: _fieldRanges,
    assertRanges: _assertRanges,
    childLabel: _childLabel,
    ...fields
  } = desc;
  return fields;
}

test("loop parse matches the descriptor and records ranges", () => {
  const desc = LoopDescriptor.parse("drive", readerFor(FLOW, "drive"));

  // `while:` lands in the `while_` field.
  expect(desc).toBeInstanceOf(LoopDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new LoopDescriptor({
        id: "drive",
        call: "body_flow",
        inputs: { acc: "${input.seed}" },
        while_: "${carry.go}",
        max: 10n,
      }),
    ),
  );
  expect(desc.kind).toBe("loop");
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "call"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
  // The range is keyed by the YAML key (`while`), not the field name (`while_`).
  expect(desc.fieldRanges.get(siteKey("field", "while"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "max"))).toBeInstanceOf(SourceRange);
});

test("loop parse requires call", () => {
  const cursor = readerFor("nodes:\n  drive:\n    kind: loop\n    times: 3\n", "drive");

  const error = loadError(() => LoopDescriptor.parse("drive", cursor));

  expect(error.message).toBe("node 'drive' (kind=loop): missing required field 'call'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
});

test("loop parse maps the until keyword", () => {
  const flow = "nodes:\n  drive:\n    kind: loop\n    call: body_flow\n    until: ${carry.done}\n    max: 5\n";

  const desc = LoopDescriptor.parse("drive", readerFor(flow, "drive"));

  expect(desc.until_).toBe("${carry.done}");
  expect(desc.while_).toBeNull();
  expect(desc.fieldRanges.get(siteKey("field", "until"))).toBeInstanceOf(SourceRange);
});

const PREDICATE_FLOW = `id: f
name: f
input:
  n: !int
defs:
  body:
    input:
      x: !int
    nodes:
      inc:
        kind: code
        code: return inputs["x"] + 1n
        input:
          x: \${input.x}
        output: int
    output:
      x: \${inc.output}
nodes:
  lp:
    kind: loop
    call: body
    input:
      x: \${input.n}
    PREDICATE
    max: 3
output: \${lp.output}
`;

test.each([
  ["while: true", "`while:` must be an expression string over the carried record, such as `while: not done` (got true)"],
  ["while: false", "`while:` must be an expression string over the carried record, such as `while: not done` (got false)"],
  ["until: 1", "`until:` must be an expression string over the carried record, such as `until: done` (got 1)"],
  ["until: [x]", '`until:` must be an expression string over the carried record, such as `until: done` (got ["x"])'],
])("a non-string predicate is refused at its field: %s", (predicate, message) => {
  const error = loadError(() => nodesOf(PREDICATE_FLOW.replace("PREDICATE", predicate)));

  expect(error.message).toBe(`loop node 'lp': ${message}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(24);
});

test("a lone surrogate in while is refused at the value", () => {
  const text = 'nodes:\n  lp:\n    kind: loop\n    call: body\n    while: "\\ud800"\n';

  const error = loadError(() => LoopDescriptor.parse("lp", readerFor(text, "lp")));

  expect(error.message).toBe("node 'lp' field 'while': text holding a lone surrogate is not a yian value");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(readerFor(text, "lp").rangeOf("while")!);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-loop/compose_kind_loop_parse__a_list_until_predicate_is_rejected", call: parsedNodes },
  { id: "load-loop/compose_kind_loop_parse__a_true_while_predicate_is_rejected", call: parsedNodes },
])("parseNodes: $id", runCase);
