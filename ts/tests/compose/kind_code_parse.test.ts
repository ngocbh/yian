/**
 * The `code` kind's parse side: a port of `tests/compose/test_kind_code_parse.py`.
 *
 * Changed: Python compares the descriptor to a `CodeDescriptor` with `field_ranges` excluded; TS
 * compares its class and fields.
 * `test_code_build_stamps_descriptor_origin_without_affecting_equality` drops its `==` half (TS
 * has no dataclass `==`), and its message ends "or null" where Python's ends "or None".
 *
 * TS-only: `build` passes the context's `baseDir`, the descriptor's code origins and its title to
 * the node, and compiles nothing.
 *
 * Later (T4-9): `test_code_build_frames_inline_syntax_error_at_yaml_line` and
 * `test_nonliteral_syntax_error_uses_whole_code_scalar` (TS `build` compiles nothing at load, and
 * the message `'(' was never closed` is CPython's).
 *
 * Held here until T3-22: the harvested cases at the end, whose error the CODE parse raises, each
 * under a comment naming the Python file it comes from.
 */

import { expect, test } from "bun:test";

import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { type CaseRow, parsedNodes, runCase } from "./declarations.ts";

const FLOW = `nodes:
  verdict:
    kind: code
    inputs:
      rating: \${input.rating}
    outputs: str
    code: |
      x = inputs["rating"]
      return x
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata (Python's `compare=False`). */
function fieldsOf(desc: CodeDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

/** A context over an empty registry, with `baseDir` when given. */
const reading = (baseDir?: string | null) => new BuildContext({ registry: new Map(), baseDir });

test("code parse matches descriptor and records body range", () => {
  const desc = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));

  expect(desc).toBeInstanceOf(CodeDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new CodeDescriptor({
        id: "verdict",
        code: 'x = inputs["rating"]\nreturn x\n',
        inputs: { rating: "${input.rating}" },
        outputs: "str",
      }),
    ),
  );
  // Ranges are recorded for the node body and its fields (the code body carries its own).
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "code"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("code build returns code node and retains compose ranges", () => {
  const desc = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));
  const result = desc.build(reading());

  expect(result.node).toBeInstanceOf(CodeNode);
  expect(result.node.outputContract).not.toBeNull();
  expect(result.nodeRange).toBe(desc.range);
  expect(result.wiring).toEqual(new Map([["rating", "${input.rating}"]]));
});

test("code build stamps descriptor origin", () => {
  const parsed = CodeDescriptor.parse("verdict", readerFor(FLOW, "verdict"));
  const origin = new OriginRef("root/node/verdict");
  const located = parsed.replace({ origin });

  const result = located.build(reading());

  expect(result.node.origin).toBe(origin);
  expect(() => parsed.replace({ origin: "root/node/verdict" as unknown as OriginRef })).toThrow(
    new TypeError("node descriptor origin must be an OriginRef or null"),
  );
});

// --- TS-only rows --- //

test("build passes the base directory, the code origins and the title, and compiles nothing", () => {
  const codeOrigin = new OriginRef("code");
  const codeLineOrigins = [new OriginRef("code/line/0")];
  const desc = new CodeDescriptor({
    id: "a",
    code: "return (",
    nodeName: "Broken",
    codeOrigin,
    codeLineOrigins,
  });

  const node = desc.build(reading("/flows")).node as CodeNode;

  expect(node.baseDir).toBe("/flows");
  expect(node.codeOrigin).toBe(codeOrigin);
  expect(node.codeLineOrigins).toEqual(codeLineOrigins);
  expect(node.title).toBe("Broken");
  expect(node.code).toBe("return (");
  expect((desc.build(reading()).node as CodeNode).baseDir).toBeNull();
});

// ---------- harvested conformance cases ----------
// Later (T4-9): `load-leaves/seed_e30_inline_syntax_error`, `load-leaves/seed_e31_inline_no_return`,
// `load-leaves/seed_e32_malformed_code_ref` and
// `load-leaves/compose_kind_code_parse__quoted_body_syntax_error_frames_the_whole_scalar`, whose
// errors come from compiling the CODE body.

test.each<CaseRow>([
  // tests/seeds/errors/
  { id: "load-leaves/seed_e11_field_not_allowed", call: parsedNodes },
  { id: "load-leaves/seed_e12_missing_required_field", call: parsedNodes },
])("parseNodes: $id", runCase);
