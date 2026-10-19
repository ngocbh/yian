/**
 * The section parser, strict sections, `x-*` keys and anchors: a port of
 * `tests/engine/test_parser.py`.
 *
 * The seed rows read the conformance cases `load-core/seed_00_hello_agent` and
 * `load-core/seed_11_reuse_anchors`, which hold Python's `tests/seeds/` flows.
 *
 * Changed: TS checks the schema by hand and reports one failure, at the range Python's
 * `_validation_error_range` picks: the first located failure, else a missing required key at the
 * root. Its message is that failure's own; Python's error lists every failure (pydantic's text)
 * when a missing key comes first, as in `name: [bad]` with no `id:`.
 * `uses: {7: child}` now fails at the document, as a key YAML read as non-text, still at the key.
 * A recursive alias inside a tagged `nodes:` entry now parses; Python recursed until its generic
 * `except` reported the overflow with no location.
 *
 * TS-only: the messages of the schema rows, whose Python text is pydantic's except for a text
 * field read as non-text, and the tagged recursive-alias row.
 *
 * Held here until T3-22: the harvested cases at the end, whose error `parseSourceDocument` or
 * `parseFile` raises, each group under a comment naming the Python file it comes from.
 */

import { afterEach, expect, test } from "bun:test";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { ComposeFile, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { caseDocument, loadError, runTextCase, type TextCaseRow } from "../compose/declarations.ts";

afterEach(() => forget());

const MINIMAL = `
id: min
name: minimal
input:
  topic: !str
nodes:
  note:
    kind: agent
    input:
      topic: \${input.topic}
    output: str
    prompt: "Write a note on \${topic}."
output: \${note.output}
`;

/** Return the range of the last frame of `error`'s diagnostic. */
function lastRange(error: LoadError): SourceRange | null {
  return error.diagnostic.frames.at(-1)!.range;
}

test("a minimal flow parses", () => {
  const document = parseSourceDocument(MINIMAL, "<flow>");
  const f = parseFile(document);

  expect(f).toBeInstanceOf(ComposeFile);
  expect(f.id).toBe("min");
  expect(f.name).toBe("minimal");
  expect(f.description).toBeNull();
  // The section stays raw; the input's type is the tag on its source node.
  expect(f.inputs).toEqual({ topic: null });
  expect(document.root.child("input").child("topic").tag).toBe("str");
  expect(Object.keys(f.nodes)).toContain("note"); // nodes stays a raw mapping here
  expect((f.nodes["note"] as Record<string, unknown>)["kind"]).toBe("agent");
  expect(f.outputs).toBe("${note.output}");
  expect(f.asserts).toEqual([]); // optional, defaults empty
  expect(f.typedefs).toEqual({});
});

test("an unknown top-level key is loud", () => {
  const error = loadError(() => parseFile("id: x\nname: x\nnodes: {}\nbogus: 1\n"));

  expect(error.message).toContain("bogus");
  expect(lastRange(error)).toEqual(new SourceRange(4, 1, 4, 6));
});

test.each([
  ["id: x\nname: x\nnodes: {}\nasserts:\n  - true\n", new SourceRange(5, 5, 5, 9)],
  ["id: x\nname: x\nnodes: {}\ninput: []\n", new SourceRange(4, 8, 4, 10)],
  ["id: x\nname: x\nnodes: {}\nuses:\n  7: child\n", new SourceRange(5, 3, 5, 4)],
  // A missing required key never replays, so the later located failure wins.
  ["name: x\nnodes: [1]\n", new SourceRange(2, 8, 2, 11)],
  ["id: x\nname: x\nasserts: [1]\n", new SourceRange(3, 11, 3, 12)],
])("a schema error is located at its field: %#", (text, expected) => {
  expect(lastRange(loadError(() => parseFile(text)))).toEqual(expected);
});

test.each([
  [
    "id: x\nname: x\nnodes: {}\nasserts:\n  - true\n",
    "`asserts[0]:` must be text, but YAML read it as a boolean; put it in quotes",
  ],
  [
    "id: x\nname: x\nnodes: {}\ninput: []\n",
    "`input:` must be a mapping, but YAML read it as a list; write a mapping",
  ],
  [
    "id: x\nname: x\nnodes: {}\nuses:\n  7: child\n",
    "the key `7` was read by YAML as an integer, but names in a flow must be text. " +
      'Quote it ("7") or choose another name.',
  ],
  ["id:\nname: x\nnodes: {}\n", "`id:` must be text, but it is empty; write the text"],
  [
    "id: x\nname: x\nnodes: {}\nuses:\n  lib: 5\n",
    "`uses.lib:` must be text, but YAML read it as an integer; put it in quotes",
  ],
  [
    "id: x\nname: x\nnodes: {}\nasserts: {}\n",
    "`asserts:` must be a list, but YAML read it as a mapping; write a list",
  ],
  [
    "id: x\nname: x\ndescription: null\nversion: 2\nnodes: {}\n",
    "`version:` must be text, but YAML read it as an integer; put it in quotes",
  ],
])("a schema error's message: %#", (text, message) => {
  expect(loadError(() => parseFile(text)).message).toBe(message);
});

test("a schema error uses the first location that replays", () => {
  const document = parseSourceDocument("name: [bad]\nnodes: {}\n", "flow.yaml");
  const error = loadError(() => parseFile(document));

  expect(error.message).toBe("`name:` must be text, but YAML read it as a list; put it in quotes");
  expect(lastRange(error)).toEqual(new SourceRange(1, 7, 1, 12));
});

test("a missing schema field falls back to the root range", () => {
  const document = parseSourceDocument("id: x\nname: x\n", "flow.yaml");
  const error = loadError(() => parseFile(document));

  expect(error.message).toBe("missing required top-level key 'nodes'");
  expect(lastRange(error)).toEqual(document.root.range);
});

test("a recursive alias inside a tagged nodes entry parses", () => {
  const f = parseFile("id: x\nname: x\nnodes:\n  a: !int &m\n    nodes:\n      b: *m\n");

  expect(Object.keys(f.nodes)).toEqual(["a"]);
});

test("x-* keys are ignored", () => {
  const f = parseFile(`x-owner: ngocbh\nx-agent-defaults:\n  kind: agent\n${MINIMAL}`);

  expect(f).toBeInstanceOf(ComposeFile);
  expect(f.id).toBe("min");
  // x-* keys are not surfaced on the strict model
  expect(Object.hasOwn(f, "x-owner")).toBe(false);
});

test("asserts and typedefs are optional", () => {
  const f = parseFile(
    `${MINIMAL}asserts:\n  - "input.topic != \\"\\""\ntypedefs:\n  Rating:\n    category: str\n`,
  );

  expect(f.asserts).toHaveLength(1);
  expect(Object.keys(f.typedefs)).toContain("Rating");
});

test("a non-mapping top level is loud", () => {
  expect(() => parseFile("- just\n- a\n- list\n")).toThrow(LoadError);
});

test("seed 00 parses its sections", () => {
  const f = parseFile(caseDocument("load-core/seed_00_hello_agent").document);

  expect(f.id).toBe("hello-agent");
  expect(f.name).toBe("hello_agent");
  expect(Object.keys(f.nodes)).toContain("note");
  expect(f.outputs).toBe("${note.output}");
});

test("seed 11 loads its anchors", () => {
  // The document expands &anchor / *alias / <<: merge before the strict schema; the
  // x-agent-defaults anchor holder is stripped as an x-* key.
  const f = parseFile(caseDocument("load-core/seed_11_reuse_anchors").document);

  expect(f.id).toBe("reuse-anchors");
  for (const nid of ["pro", "con", "judge"]) {
    const node = f.nodes[nid] as Record<string, unknown>;
    expect(node["kind"]).toBe("agent"); // merged in from the anchor
    expect(node["outputs"]).toBe("str");
    expect((node["llm_config"] as Record<string, unknown>)["provider"]).toBe("anthropic");
  }
});

test("a flow parses an optional version", () => {
  const f = parseFile("id: x\nname: x\nversion: v1\nnodes:\n  a: {kind: code, code: m:f}\n");

  expect(f.version).toBe("v1");
});

test("a flow's version defaults to null", () => {
  expect(parseFile("id: x\nname: x\nnodes:\n  a: {kind: code, code: m:f}\n").version).toBeNull();
});

// ---------- harvested cases ----------

test.each<Omit<TextCaseRow, "call">>([
  // tests/compose/test_unbuildable_scalars.py
  { id: "yaml/compose_unbuildable_scalars__a_case_when" },
  { id: "yaml/compose_unbuildable_scalars__a_uses_file", file: "child.yaml" },
  { id: "yaml/compose_unbuildable_scalars__day_out_of_range" },
  { id: "yaml/compose_unbuildable_scalars__hour_out_of_range" },
  { id: "yaml/compose_unbuildable_scalars__month_out_of_range" },
  { id: "yaml/compose_unbuildable_scalars__offset_minutes_out_of_range" },
  { id: "yaml/compose_unbuildable_scalars__tagged_input_default" },
  { id: "yaml/compose_unbuildable_scalars__year_zero" },
  // tests/engine/test_long_int_literals.py
  { id: "yaml/engine_long_int_literals__long_int_literal" },
  // tests/engine/test_limit_overrides.py; the case leaves out Python's message.
  {
    id: "load-core/engine_limit_overrides__yaml_nesting_follows_the_variable",
    message:
      "invalid YAML: this list nests deeper than 5 levels, the limit for a YAML document " +
      "(each mapping and list is a level; a yian value nests at most 4); flatten it",
  },
  // tests/engine/test_uses_files.py
  { id: "load-core/engine_uses_files__malformed_external_child", file: "child.yaml" },
  // A sample case of the case format, with no Python test.
  { id: "yaml/unclosed_flow_sequence" },
])("parseSourceDocument: $id", (row) => runTextCase({ ...row, call: parseSourceDocument }));

test.each([
  // tests/engine/test_surface_vocab_golden.py
  "load-core/engine_surface_vocab_golden__legacy_inputs_section",
  // tests/seeds/errors/
  "load-core/seed_e09_unknown_top_level_key",
  // tests/engine/test_memory_authoring.py
  "load-agent/engine_memory_authoring__retired_memory_policy_at_the_root",
])("parseFile: %s", (id) =>
  runTextCase({ id, call: (text, label) => parseFile(parseSourceDocument(text, label)) }),
);
