/**
 * A mapping key YAML reads as non-text is a load error at the key: a port of
 * `tests/compose/test_non_text_keys.py`, through `parseSourceDocument` (and `parseFile` +
 * `parseNodes` for the quoted-key row).
 *
 * Dropped: the date key rows (`2026-10-13` and `2026-10-13 10:00:00` are text in TS) and the
 * `!!binary` rows (the TS YAML layer refuses `!!binary`).
 *
 * Later (T3-16): `test_case_on_needs_no_exemption`. Later (T3-22):
 * `test_the_flow_loads_with_text_names_everywhere`, `test_a_child_flow_file_reports_its_own_key`,
 * and the loading halves of `test_a_quoted_key_follows_the_usual_name_rules` and
 * `test_words_yaml_1_1_read_as_booleans_are_text_keys`; here a quoted key and a YAML 1.1 word
 * parse with no error.
 */

import { expect, test } from "bun:test";

import { parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { loadError, nodesOf } from "./declarations.ts";
import { StandInCode, standIn } from "./stand_ins.ts";

const SITES = [
  "field",
  "typedef",
  "def_node",
  "flow_input",
  "node_input",
  "record",
  "inline_node",
  "root_node",
  "flow_output",
  "x_block",
] as const;

type Site = (typeof SITES)[number];

/** One flow with a name at each site; `@site` marks where `flowWith` writes it. */
const FLOW = `id: f
name: f
typedefs:
  R:
    a: int
    @field: int
  @typedef:
    a: int
defs:
  child:
    nodes:
      @def_node:
        kind: code
        code: return 1
        output: int
    output: 1
input:
  q: !str
  @flow_input: !str
nodes:
  a:
    kind: code
    code: return inputs["v"]
    input:
      v: \${input.q}
      @node_input: x
      w: {@record: a}
    output: str
  b:
    kind: call
    nodes:
      inner:
        kind: code
        code: return 1
        output: int
      @inline_node:
        kind: code
        code: return 2
        output: int
    output: \${inner.output}
  @root_node:
    kind: code
    code: return 3
    output: int
output:
  r: \${a.output}
  @flow_output: \${a.output}
x-meta:
  @x_block: v
`;

/** Return the flow with `key` written at `site` and plain names elsewhere. */
function flowWith(site: Site, key: string): string {
  return FLOW.replace(/@(\w+)/g, (_, name: string) =>
    name === site ? key : name === "typedef" ? "T" : `n_${name}`,
  );
}

/** Return the range of `key` where `flowWith` wrote it. */
function keyRange(text: string, key: string, site: Site): SourceRange {
  const line = FLOW.split("\n").findIndex((written) => written.includes(`@${site}`)) + 1;
  const column = text.split("\n")[line - 1]!.indexOf(key) + 1;
  return new SourceRange(line, column, line, column + key.length);
}

/** Return the `LoadError` parsing `text` throws. */
function parseError(text: string) {
  return loadError(() => parseSourceDocument(text, "k.yaml"));
}

const KEYS = [
  ["True", "a boolean"],
  ["1", "an integer"],
  ["null", "null"],
  ["1.5", "a float"],
] as const;

test.each(SITES.flatMap((site) => KEYS.map(([key, readAs]) => [site, key, readAs] as const)))(
  "a non-text key is a load error at the key: %s %s",
  (site, key, readAs) => {
    const text = flowWith(site, key);

    const error = parseError(text);

    expect(error.diagnostic.message).toBe(
      `the key \`${key}\` was read by YAML as ${readAs}, but names in a flow must be text. ` +
        `Quote it ("${key}") or choose another name.`,
    );
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(keyRange(text, key, site));
  },
);

test("the message names what yaml read", () => {
  const error = parseError(flowWith("x_block", "0x1F"));

  expect(error.diagnostic.message).toBe(
    "the key `0x1F` was read by YAML as an integer, but names in a flow must be text. " +
      'Quote it ("0x1F") or choose another name.',
  );
});

test("an empty key asks for a name", () => {
  const text = flowWith("x_block", "n_x_block").replace("  n_x_block: v", "  ? \n  : v");

  expect(parseError(text).diagnostic.message).toBe(
    "an empty key was read by YAML as null, but names in a flow must be text. Write a name.",
  );
});

test("a quoted key follows the usual name rules", () => {
  // The loading halves are later; the quoted keys parse.
  parseFile(parseSourceDocument(flowWith("root_node", '"True"'), "k.yaml"));
  parseFile(parseSourceDocument(flowWith("x_block", "'2026-10-13'"), "k.yaml"));

  const call = standIn("call", { allow: ["nodes", "outputs"] });
  const error = loadError(() => nodesOf(flowWith("root_node", "'1'"), [StandInCode, call]));

  expect(error.diagnostic.message).not.toContain("names in a flow must be text");
  expect(error.diagnostic.message).toContain("_1");
});

test.each(["yes", "on", "off", "no"])("yaml 1.1 boolean words are text keys: %s", (key) => {
  parseSourceDocument(flowWith("root_node", key), "k.yaml");
  parseSourceDocument(flowWith("flow_input", key), "k.yaml");
});

test.each([
  ["same-integer", "x-meta:\n  017: a\n  17: b\n", new SourceRange(4, 3, 4, 6)],
  ["one-and-true", "x-meta: {1: a, true: b}\n", new SourceRange(3, 10, 3, 11)],
])("a non-text key is reported before a repeat it causes: %s", (_, meta, range) => {
  const text =
    "id: f\nname: f\n" +
    meta +
    "nodes:\n  a:\n    kind: code\n    code: return 1\n    output: int\n" +
    "output: ${a.output}\n";

  const error = parseError(text);

  expect(error.diagnostic.message).toContain("was read by YAML as an integer");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(range);
});

test("a numeric input key is a load error not a type error", () => {
  const text = `id: f
name: f
nodes:
  a:
    kind: code
    input: {1: 2}
    code: return 1
    output: int
output: \${a.output}
`;

  const error = parseError(text);

  expect(error.diagnostic.message.startsWith("the key `1` was read by YAML as an integer")).toBe(
    true,
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(6, 13, 6, 14));
});

test.each([
  ["tagged", "node_input", "!object {KEY: 2}"],
  ["tagged-list", "node_input", "!list(object) [{KEY: 2}]"],
  ["nested-template", "node_input", '{KEY: "${input.q}"}'],
  ["nested-tag", "node_input", "{d: !object {KEY: 2}, s: '${input.q}'}"],
  ["tagged-default", "flow_default", "!object {KEY: 2}"],
  ["tagged-list-default", "flow_default", "!list(object) [{KEY: 2}]"],
])("a non-text key in a tagged or nested value is the same error: %s", (_, where, value) => {
  const written = value.replace("KEY", "1");
  let line: string;
  let text: string;
  if (where === "node_input") {
    line = `      v: ${written}`;
    text =
      "id: f\nname: f\ninput:\n  q: !str\nnodes:\n  a:\n    kind: code\n" +
      `    code: return 1\n    input:\n${line}\n    output: int\n` +
      "output: ${a.output}\n";
  } else {
    line = `  v: ${written}`;
    text =
      `id: f\nname: f\ninput:\n${line}\nnodes:\n  a:\n    kind: code\n` +
      "    code: return 1\n    output: int\noutput: ${a.output}\n";
  }

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(
    "the key `1` was read by YAML as an integer, but names in a flow must be text. " +
      'Quote it ("1") or choose another name.',
  );
  const lineNumber = text.split("\n").indexOf(line) + 1;
  const column = line.indexOf("1:") + 1;
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(
    new SourceRange(lineNumber, column, lineNumber, column + 1),
  );
});
