/** The section parser, strict sections, `x-*` keys and anchors. */

import { afterEach, expect, test } from "bun:test";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { ComposeFile, flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import {
  caseDocument,
  fileOf,
  loadError,
  runTextCase,
  type TextCaseRow,
} from "../helpers/declarations.ts";

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
    output: !str
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
  const { document } = caseDocument("load-core/seed_11_reuse_anchors");
  const f = parseFile(document);
  // The merged `output: !str` tag is read from the source, as the loader reads it.
  const descriptors = parseNodes(flowNodesCursor(document, f.nodes));

  expect(f.id).toBe("reuse-anchors");
  for (const nid of ["pro", "con", "judge"]) {
    const node = f.nodes[nid] as Record<string, unknown>;
    expect(node["kind"]).toBe("agent"); // merged in from the anchor
    expect(descriptors.get(nid)!.outputs).toBe("str");
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

/** yian's message for a plain scalar that reads as a `what` but is not a valid one. */
const notValid = (what: string, reason: string) =>
  `invalid YAML: not a valid ${what}: ${reason}; quote it to keep it as text`;
const BAD_MONTH = notValid("date", "month must be in 1..12");

// A row with `message` pins yian's text where the case leaves the message out.
test.each<Omit<TextCaseRow, "call">>([
  { id: "yaml/compose_unbuildable_scalars__a_case_when", message: BAD_MONTH },
  { id: "yaml/compose_unbuildable_scalars__a_uses_file", file: "child.yaml", message: BAD_MONTH },
  {
    id: "yaml/compose_unbuildable_scalars__day_out_of_range",
    message: notValid("date", "day is out of range for month"),
  },
  {
    id: "yaml/compose_unbuildable_scalars__hour_out_of_range",
    message: notValid("datetime", "hour must be in 0..23"),
  },
  { id: "yaml/compose_unbuildable_scalars__month_out_of_range", message: BAD_MONTH },
  {
    id: "yaml/compose_unbuildable_scalars__offset_minutes_out_of_range",
    message: notValid("datetime", "UTC offset minutes must be under 60"),
  },
  { id: "yaml/compose_unbuildable_scalars__tagged_input_default", message: BAD_MONTH },
  {
    id: "yaml/compose_unbuildable_scalars__year_zero",
    message: notValid("date", "year 0 is out of range"),
  },
  { id: "yaml/engine_long_int_literals__long_int_literal" },
  { id: "load-core/engine_limit_overrides__yaml_nesting_follows_the_variable" },
  { id: "load-core/engine_uses_files__malformed_external_child", file: "child.yaml" },
  // A sample case of the case format.
  { id: "yaml/unclosed_flow_sequence" },
])("parseSourceDocument: $id", (row) => runTextCase({ ...row, call: parseSourceDocument }));

test.each([
  "load-core/engine_surface_vocab_golden__legacy_inputs_section",
  "load-core/seed_e09_unknown_top_level_key",
])("parseFile: %s", (id) => runTextCase({ id, call: fileOf }));
