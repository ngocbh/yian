/**
 * The `wait` kind's parse side: a port of `tests/compose/test_kind_wait_parse.py`.
 *
 * Changed: Python compares the descriptor to a `WaitDescriptor` with `field_ranges` excluded; TS
 * compares its class and fields.
 *
 * TS-only: one row per branch of the `until:` literal check, and one for the container check. TS YAML reads an unquoted date or
 * timestamp as text, so it is checked as text and bound as written, where Python bound the
 * timestamp it read in `isoformat()` spelling. `build` passes the title to the node.
 *
 * Held here until T3-22: the harvested cases at the end, whose error the WAIT parse raises, each
 * under a comment naming the Python file it comes from.
 *
 * Later (T3-16/T3-17): `load-leaves/compose_map_wait_source_types__literal_until_check_in_an_inline_body`,
 * whose WAIT sits in an inline `call` body.
 */

import { describe, expect, test } from "bun:test";

import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

const FLOW = `nodes:
  hold:
    kind: wait
    until: \${input.deadline}
`;

const FIX = "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`";

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata (Python's `compare=False`). */
function fieldsOf(desc: WaitDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

/** A context over an empty registry. */
const reading = () => new BuildContext({ registry: new Map() });

test("wait parse matches descriptor and records ranges", () => {
  const desc = WaitDescriptor.parse("hold", readerFor(FLOW, "hold"));

  expect(desc).toBeInstanceOf(WaitDescriptor);
  expect(fieldsOf(desc)).toEqual(fieldsOf(new WaitDescriptor({ id: "hold", until: "${input.deadline}" })));
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "until"))).toBeInstanceOf(SourceRange);
});

test("wait build returns blocking node with reserved until wiring", () => {
  const desc = WaitDescriptor.parse("hold", readerFor(FLOW, "hold"));
  const result = desc.build(reading());

  expect(result.node).toBeInstanceOf(WaitNode);
  // The `until` source rides the reserved wiring key; the node carries no author params.
  expect(result.node.params).toEqual([]);
  expect(result.wiring).toEqual(new Map([["until", "${input.deadline}"]]));
  expect(result.nodeRange).toBe(desc.range);
});

// --- TS-only rows --- //

describe("the until: literal", () => {
  /** Parse WAIT node `w` whose `until:` is written `value`. */
  const parseW = (value: string) =>
    WaitDescriptor.parse("w", readerFor(`nodes:\n  w:\n    kind: wait\n    until: ${value}\n`, "w"));

  test.each([
    ["quoted date text", "'2026-05-21'", "2026-05-21"],
    ["an unquoted date", "2026-05-21", "2026-05-21"],
    ["an unquoted timestamp", "2026-05-21T14:30:00Z", "2026-05-21T14:30:00Z"],
    ["an unquoted timestamp with an offset", "2026-05-21T14:30+07:00", "2026-05-21T14:30+07:00"],
    ["a space-separated timestamp", "2026-05-21 14:30:00", "2026-05-21 14:30:00"],
    ["text holding a reference", "at ${input.as_of}", "at ${input.as_of}"],
  ])("%s passes and binds its text unchanged", (_label, written, text) => {
    const desc = parseW(written);

    expect(desc.until).toBe(text);
    expect(desc.build(reading()).wiring).toEqual(new Map([["until", text]]));
  });

  test.each([
    ["true", "a boolean"],
    ["1.5", "a float"],
    ["{at: noon}", "a mapping"],
    ["[2026-05-21]", "a list"],
  ])("%s is refused as %s", (written, phrase) => {
    const error = loadError(() => parseW(written));

    expect(error.message).toBe(
      `node 'w' (kind=wait): \`until:\` must be a date or datetime, but YAML read it as ${phrase}; ${FIX}`,
    );
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
  });

  test.each(["ok", "'2026-02-30'", "'2026-05-21T24:00'"])("text %s is not a date or datetime", (written) => {
    const error = loadError(() => parseW(written));

    expect(error.message).toBe(
      "node 'w' (kind=wait): `until:` must be a date or datetime, but YAML read it as text that is not a " +
        `date or datetime; ${FIX}`,
    );
  });

  test("a list holding a reference gets the container message", () => {
    const error = loadError(() => parseW("[\"${a.output}\"]"));

    expect(error.message).toBe(
      "wait node 'w' until: a value holding ${...} can't be a list or mapping; write the list as an " +
        "expression, e.g. `${[a.output, b.output]}`",
    );
  });
});

test("build passes the title to the node", () => {
  const node = new WaitDescriptor({ id: "w", until: "2026-05-21", nodeName: "Settle" }).build(reading()).node;

  expect(node.title).toBe("Settle");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  // tests/compose/test_map_wait_source_types.py
  { id: "load-leaves/compose_map_wait_source_types__call_text_under_until_is_not_a_directive", call: parsedNodes },
  { id: "load-leaves/compose_map_wait_source_types__integer_until_literal", call: parsedNodes },
  { id: "load-leaves/compose_map_wait_source_types__month_13_text_until_is_not_a_date", call: parsedNodes },
  { id: "load-leaves/compose_map_wait_source_types__null_until_is_empty", call: parsedNodes },
])("parseNodes: $id", runCase);
