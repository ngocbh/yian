/**
 * The `when:` hint on a node body's unknown-key error: a port of
 * `tests/compose/test_node_when_hint.py`, through `parseFile`, `flowNodesCursor` and `parseNodes`
 * with stand-in kinds.
 *
 * Changed: `test_when_on_non_case_node_suggests_a_case_node` runs on the `code` stand-in alone,
 * and `test_when_on_case_node_points_at_cases_entries` on a `case` stand-in carrying the case
 * kind's `whenHint`.
 *
 * Later (T3-15/T3-16): the per-kind sweep of `test_when_on_non_case_node_suggests_a_case_node`
 * over every kind but `case`, and the real `case` kind's hint.
 */

import { expect, test } from "bun:test";

import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { loadError, nodesOf } from "./declarations.ts";
import { StandInCase, StandInCode, withKinds } from "./stand_ins.ts";

const NON_CASE_HINT = "route to it from a `case` node";
const CASE_HINT = "goes in each entry of `cases:`";

/** A `case` stand-in with the case kind's `when:` hint. */
class HintedCase extends StandInCase {
  static override readonly whenHint =
    "; `when:` goes in each entry of `cases:` (`cases: [{when: ..., then: <node>}]`)";
}

/** Return the message of the `LoadError` parsing flow `text`'s nodes with `kinds` throws. */
function nodesMessage(text: string, kinds: readonly DescriptorClass[]): string {
  return loadError(() => nodesOf(text, kinds)).message;
}

test("when on a non-case node suggests a case node", () => {
  const text = `id: f
name: f
nodes:
  x:
    kind: code
    when: \${x.output}
output: \${x.output}
`;

  const message = nodesMessage(text, [StandInCode]);

  expect(message).toContain("field 'when' is not allowed");
  expect(message).toContain("(allowed: ");
  expect(message).toContain(NON_CASE_HINT);
});

test("when on a case node points at cases entries", () => {
  const text = `id: f
name: f
nodes:
  route:
    kind: case
    when: \${a.output}
    cases:
      - when: \${a.output}
        then: b
  a:
    kind: code
    code: "def run(): return True"
  b:
    kind: code
    code: "def run(): return 1"
output: \${b.output}
`;

  const message = nodesMessage(text, [HintedCase, StandInCode]);

  expect(message).toContain("field 'when' is not allowed");
  expect(message).toContain(CASE_HINT);
  expect(message).not.toContain("route to it from");
});

test("when error points at the when key", () => {
  const source = `nodes:
  x:
    kind: code
    code: "def run(): return 1"
    when: \${a.output}
`;
  const nodes = SourceDocument.parse(source).root.child("nodes");
  const node = nodes.child("x");

  const error = loadError(() => withKinds([StandInCode], () => parseNodes(nodes)));

  expect(error.message).toContain("field 'when' is not allowed");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(node.rangeOf("when")!);
});

test("other unknown key gets no case hint", () => {
  const text = `id: f
name: f
nodes:
  x:
    kind: code
    code: "def run(): return 1"
    bogus: 1
output: \${x.output}
`;

  const message = nodesMessage(text, [StandInCode]);

  expect(message).toContain("field 'bogus' is not allowed");
  expect(message.endsWith(")")).toBe(true);
  expect(message).not.toContain("route to it");
  expect(message).not.toContain("`case`");
});

test("when on a compact flow suggests a case node", () => {
  const text = `id: x
name: x
kind: code
code: "def run(): return 1"
when: \${input.flag}
`;

  const message = nodesMessage(text, [StandInCode]);

  expect(message).toContain("field 'when' is not allowed");
  expect(message).toContain(NON_CASE_HINT);
});
