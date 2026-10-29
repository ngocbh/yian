/** A mapping key YAML reads as non-text is a load error at the key. */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { parseSourceDocument } from "../../src/compose/parse/file.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { loadError, loadErrorAsync, nodesOf, runTextCase } from "../helpers/declarations.ts";

afterEach(() => forget());

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

/** Load `text` as `k.yaml`. */
const load = (text: string) => loadFlow(text, { sourceLabel: "k.yaml" });

test("the flow loads with text names everywhere", async () => {
  expect(await load(flowWith("x_block", "n_x_block"))).not.toBeNull();
});

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

test.each([
  ["{role: Literal(a, true, b)}", "true", "a boolean", "Literal(a"],
  ["{role: Literal(a, b, null, c)}", "null", "null", "Literal(a"],
  ["{n: Literal(1, 2, 3)}", "2", "an integer", "Literal(1"],
  ["{t: Dict(str, List(int)), u: Literal(x, 1, y)}", "1", "an integer", "Literal(x"],
])("a key a flow mapping's comma made names the split type: %s", (written, key, readAs, cut) => {
  // YAML splits `{role: Literal(a, true, b)}` at each comma, so `true` becomes a key.
  const text = flowWith("x_block", "n_x_block").replace("  n_x_block: v", `  R: ${written}`);

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(
    `the key \`${key}\` was read by YAML as ${readAs}, but names in a flow must be text. This ` +
      `\`{...}\` mapping's comma split the type that starts \`${cut}\`; quote the type or use ` +
      "block style.",
  );
});

test.each([
  ["a block mapping", "  a: Literal(x\n  true: 1"],
  ["a closed type", "  R: {a: Literal(x), true: 1}"],
  ["a mapping value", "  R: {a: {b: Literal(x}, true: 1}"],
])("a non-text key after %s keeps the plain message", (_label, block) => {
  const text = flowWith("x_block", "n_x_block").replace("  n_x_block: v", block);

  expect(parseError(text).diagnostic.message).toEndWith('Quote it ("true") or choose another name.');
});

test("an empty key asks for a name", () => {
  const text = flowWith("x_block", "n_x_block").replace("  n_x_block: v", "  ? \n  : v");

  expect(parseError(text).diagnostic.message).toBe(
    "an empty key was read by YAML as null, but names in a flow must be text. Write a name.",
  );
});

test("a quoted key follows the usual name rules", async () => {
  await load(flowWith("root_node", '"True"'));
  await load(flowWith("x_block", "'2026-10-13'"));

  const error = await loadErrorAsync(load(flowWith("root_node", "'1'")));

  expect(error.diagnostic.message).not.toContain("names in a flow must be text");
  expect(error.diagnostic.message).toContain("_1");
});

test("case on needs no exemption", async () => {
  const text = `id: f
name: f
input:
  x: !str
nodes:
  route:
    kind: case
    on: \${input.x}
    cases:
      - when: a
        then: done
    else: done
  done:
    kind: code
    code: return 1
    output: int
output: \${done.output}
`;

  expect((nodesOf(text, []).get("route") as CaseDescriptor).on).toBe("${input.x}");
  await load(text);
});

test.each(["yes", "on", "off", "no"])("yaml 1.1 boolean words are text keys: %s", async (key) => {
  await load(flowWith("root_node", key));
  await load(flowWith("flow_input", key));
});

test("a child flow file reports its own key", async () => {
  const tmp = mkdtempSync(join(tmpdir(), "yian-non-text-keys-"));
  try {
    writeFileSync(
      join(tmp, "child.yaml"),
      "id: child\nname: child\nnodes:\n  a:\n    1: x\n    kind: code\n" +
        "    code: return 1\n    output: int\noutput: ${a.output}\n",
    );
    const parent =
      "id: parent\nname: parent\nnodes:\n  c:\n    kind: call\n    call: kid\n" +
      "output: ${c.output}\nuses:\n  kid: child\n";

    const error = await loadErrorAsync(loadFlow(parent, { searchPaths: [tmp] }));

    const frame = error.diagnostic.frames.at(-1)!;
    expect(error.diagnostic.message).toContain("was read by YAML as an integer");
    expect(frame.label).toBe("child.yaml");
    expect(frame.range).toEqual(new SourceRange(5, 5, 5, 6));
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
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

test("an alias key is reported where it is written, not at its anchor", () => {
  const text =
    "id: f\nname: f\nnodes:\n  n: !int &k 1\n" +
    "output:\n  r: ${n.output}\n  *k : ${n.output}\n";

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(
    "the key `1` was read by YAML as an integer, but names in a flow must be text. " +
      'Quote it ("1") or choose another name.',
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(7, 3, 7, 5));
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

test.each([
  { id: "yaml/compose_non_text_keys__a_child_flow_file_reports_its_own_key", file: "child.yaml" },
  { id: "yaml/compose_non_text_keys__boolean_key_as_a_node_id" },
  { id: "yaml/compose_non_text_keys__date_key_in_a_record_literal" },
  { id: "yaml/compose_non_text_keys__datetime_key" },
  { id: "yaml/compose_non_text_keys__empty_key_asks_for_a_name" },
  { id: "yaml/compose_non_text_keys__float_key_as_a_typedef_field" },
  { id: "yaml/compose_non_text_keys__hex_key_is_shown_as_written" },
  { id: "yaml/compose_non_text_keys__integer_key_as_a_flow_input" },
  { id: "yaml/compose_non_text_keys__integer_key_in_a_tagged_value" },
  { id: "yaml/compose_non_text_keys__null_key_in_an_x_block" },
  { id: "yaml/compose_non_text_keys__one_then_true_reports_the_non_text_key" },
])("parseSourceDocument: $id", (row) => runTextCase({ ...row, call: parseSourceDocument }));
