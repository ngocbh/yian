/**
 * Compact (single-node) flows: a port of the parse rows of `tests/engine/test_compact.py`.
 *
 * A compact flow carries a node `kind:` at the top level and no `nodes:` map. The parser
 * desugars it into the canonical one-node flow: the flow `id:` names the node, the flow `input:`
 * is the node signature auto-wired by name, the flow `output:` is the node's output type
 * re-exported as the flow output, and every other top-level key is the node body. The
 * projection row uses `StandInCode` for the code kind.
 *
 * Python reaches `test_compact_digit_id_rejected_without_parser_dump` and the full-form half of
 * `test_full_form_flow_id_may_keep_hyphens` through `load_flow`; TS calls `parseFile`.
 *
 * Later (T3-22): `test_compact_code_runs`, `test_compact_record_output_runs`, which run flows,
 * and the compiled-nodes check of `test_full_form_flow_id_may_keep_hyphens`.
 *
 * The harvested cases whose error `parseFile` raises are pinned at the end, with
 * `load-types/compose_tagged_inputs__compact_input_list_shows_the_tagged_spelling` (from
 * `tests/compose/test_tagged_inputs.py`).
 */

import { afterEach, expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { fileOf, loadError, runTextCase } from "../compose/declarations.ts";
import { StandInCode, withKinds } from "../compose/stand_ins.ts";

afterEach(() => forget());

const COMPACT_CODE = `
id: echo_one
name: echo_one
input:
  topic: !str
output: str
kind: code
code: tests.engine._compose_codefns:echo
`;

test("a compact flow desugars to one node keyed by the flow id", () => {
  const f = parseFile(COMPACT_CODE);

  expect(Object.keys(f.nodes)).toEqual(["echo_one"]);
  const node = f.nodes["echo_one"] as Record<string, unknown>;
  expect(node["kind"]).toBe("code");
  expect(node["inputs"]).toEqual({ topic: "${input.topic}" }); // auto-wired by name
  expect(f.outputs).toBe("${echo_one.output}"); // single-value re-export
});

test("the compact projection keeps logic ranges without changing the root", () => {
  const document = parseSourceDocument(COMPACT_CODE, "compact.yaml");
  const original = document.root.value;
  const parsed = parseFile(document);

  const cursor = flowNodesCursor(document, parsed.nodes);
  const descriptor = withKinds([StandInCode], () => parseNodes(cursor)).get("echo_one")!;

  expect(cursor.document).toBe(document);
  expect(descriptor.range).toEqual(new SourceRange(2, 1, 2, 3));
  expect(cursor.child("echo_one").rangeOf("kind")).toEqual(new SourceRange(7, 7, 7, 11));
  expect(descriptor.fieldRanges.get(siteKey("field", "code"))).toEqual(
    new SourceRange(8, 7, 8, 41),
  );
  expect(descriptor.fieldRanges.has(siteKey("field", "inputs"))).toBe(false);
  expect(document.root.value).toEqual(original);
});

const COMPACT_NON_LEAF = `
id: router
name: router
input:
  x: !str
kind: case
on: \${input.x}
cases:
  - when: a
    then: somewhere
`;

test("a compact flow rejects a non-leaf kind", () => {
  // case/call/map reference other nodes a one-node flow has none of.
  const error = loadError(() => parseFile(COMPACT_NON_LEAF));

  expect(error.message).toContain("not allowed inline");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(6, 7, 6, 11));
});

const COMPACT_MODEL = `
id: rank
name: rank
input:
  topic: !str
output: float
kind: model
model_id: x
`;

test("the compact model kind is removed", () => {
  // A removed kind names its replacement instead of the non-leaf rejection.
  const error = loadError(() => parseFile(COMPACT_MODEL));

  expect(error.message).toContain("kind 'model' was removed");
  expect(error.message).not.toContain("not allowed inline");
  expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(7);
});

const COMPACT_NO_ID = `
name: noid
input:
  topic: !str
kind: code
code: tests.engine._compose_codefns:echo
`;

test("a compact flow requires an id", () => {
  const document = parseSourceDocument(COMPACT_NO_ID, "compact.yaml");
  const error = loadError(() => parseFile(document));

  expect(error.message).toContain("top-level `id:` is required");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(document.root.range!);
});

const COMPACT_BAD_INPUT = `
id: echo_one
name: echo_one
input: [topic]
kind: code
code: tests.engine._compose_codefns:echo
`;

test("a compact flow's invalid input is located at its value", () => {
  const error = loadError(() => parseFile(COMPACT_BAD_INPUT));

  expect(error.message).toContain("must be a param map");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(4, 8, 4, 15));
});

const COMPACT_HYPHEN_ID = `
id: hello-agent
name: hello
input:
  topic: !str
output: str
kind: code
code: tests.engine._compose_codefns:echo
`;

test("a compact flow's non-identifier id is rejected at the id", () => {
  const document = parseSourceDocument(COMPACT_HYPHEN_ID, "compact.yaml");
  const error = loadError(() => parseFile(document));

  expect(error.message).toContain("compact flow id 'hello-agent' also names its node");
  expect(error.message).toContain("use 'hello_agent'");
  expect(error.message).toContain("full `nodes:` form");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(document.root.rangeOf("id")!);
});

test("a compact flow's digit id is rejected without a parser dump", () => {
  const error = loadError(() =>
    parseFile(COMPACT_HYPHEN_ID.replace("id: hello-agent", "id: '2x'")),
  );

  expect(error.message).toMatch(/compact flow id '2x'.*use '_2x'/);
  expect(error.message).not.toContain("Unexpected token");
});

test("a full-form flow id may keep hyphens", () => {
  const f = parseFile(`
id: hello-agent
name: hello
input:
  topic: !str
nodes:
  greet:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: \${input.topic}
output: \${greet.output}
`);

  expect(f.id).toBe("hello-agent");
  expect(Object.keys(f.nodes)).toEqual(["greet"]);
});

test.each([
  "load-core/engine_compact__compact_digit_id_rejected_without_parser_dump",
  "load-core/engine_compact__compact_model_kind_is_removed",
  "load-core/engine_compact__compact_rejects_non_leaf_kind",
  // tests/compose/test_tagged_inputs.py
  "load-types/compose_tagged_inputs__compact_input_list_shows_the_tagged_spelling",
])("parseFile: %s", (id) => runTextCase({ id, call: fileOf }));
