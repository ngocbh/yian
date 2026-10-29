/** A mapping that repeats one of its own keys is a load error. */

import { afterEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { loadError, runTextCase } from "../helpers/declarations.ts";

afterEach(() => forget());

const NODE_ID = `id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    output: int
  a:
    kind: code
    code: return 2
    output: int
output: \${a.output}
`;

const NODE_FIELD = `id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    code: return 2
    output: int
output: \${a.output}
`;

const TOP_LEVEL = `id: f
name: f
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: \${a.output}
output: \${a.output}
`;

const FLOW_STYLE = `id: f
name: f
nodes:
  a:
    kind: code
    input: {x: 1, x: 2}
    code: return inputs["x"]
    output: int
output: \${a.output}
`;

test.each([
  ["node-id", NODE_ID, "a", 4, 8],
  ["node-field", NODE_FIELD, "code", 6, 7],
  ["top-level", TOP_LEVEL, "output", 8, 9],
  ["flow-style", FLOW_STYLE, "x", 6, 6],
] as const)("repeated key is a load error at the second pair: %s", (_, text, key, first, line) => {
  const error = loadError(() => parseSourceDocument(text, "dup.yaml"));

  expect(error.message).toContain(`duplicate key '${key}'`);
  expect(error.message).toContain(`first at line ${first}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("merged keys may be overridden", async () => {
  const text = `id: f
name: f
x-base: &base
  kind: code
  code: return 1
  output: int
x-other: &other
  code: return 3
nodes:
  a:
    <<: [*base, *other]
    code: return 2
output: \${a.output}
`;

  const loaded = await loadFlow(text);

  expect(loaded.compiled.nodes.has("a")).toBe(true);
  const f = parseFile(parseSourceDocument(text, "<flow>"));
  expect(f.nodes["a"]).toEqual({ kind: "code", code: "return 2", outputs: "int" });
});

test("a shared anchor reports its duplicate once at the definition", () => {
  const text = `id: f
name: f
x-base: &base
  kind: code
  code: return 1
  code: return 2
  output: int
nodes:
  a: *base
  b: *base
output: \${a.output}
`;

  const error = loadError(() => parseSourceDocument(text, "<flow>"));

  expect(error.message).toContain("duplicate key 'code'");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(6);
});

test("a node id repeated through an alias fails at the alias", () => {
  const text = `id: f
name: f
nodes:
  &k a:
    kind: code
    code: return 1
    output: int
  *k :
    kind: code
    code: return 2
    output: int
output: \${a.output}
`;

  const error = loadError(() => parseSourceDocument(text, "<flow>"));

  expect(error.message).toBe(
    "duplicate key 'a' (first at line 4); a YAML mapping's keys must be unique, so remove or rename one",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(8);
});

test("keys that read as the same integer fail as non-text first", () => {
  const text = `id: f
name: f
x-meta:
  017: a
  17: b
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: \${a.output}
`;

  const error = loadError(() => parseSourceDocument(text, "dup.yaml"));

  expect(error.message).toContain("the key `017` was read by YAML as an integer");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
  // The keys stay the text written, so `017` and `17` don't repeat.
  expect(SourceDocument.parse(text).repeatedKey).toBeNull();
});

test("on and true are two keys and true is not text", () => {
  const text = `id: f
name: f
x-meta:
  on: a
  true: b
nodes:
  a:
    kind: code
    code: return 1
    output: int
output: \${a.output}
`;

  const error = loadError(() => parseSourceDocument(text, "<flow>"));

  expect(error.message).toContain("the key `true` was read by YAML as a boolean");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(5);
  expect(SourceDocument.parse(text).root.child("x-meta").value).toEqual({ on: "a", true: "b" });
});

/** A flow's `nodes:` section, after the part a test writes. */
const NODES = "nodes:\n  a:\n    kind: code\n    code: return 1\n    output: int\n";

test.each([
  ["{a: Dict(str, int), b: Dict(str, int)}", "int)", "Dict(str", 4],
  ["{a: Dict(str, List(int)), b: Dict(str, List(int))}", "List(int))", "Dict(str", 4],
  ["{a: Union(A, B, C), b: Union(A, B, C)}", "B", "Union(A", 4],
  ["{status: Literal(ok, error, timeout), error: str}", "error", "Literal(ok", 4],
])("a repeat a flow mapping's comma made names the split value: %s", (written, key, cut, at) => {
  // YAML splits `{a: Dict(str, int), ...}` at each comma, so the type's tail `int)` becomes a
  // key that the next split type repeats; in the last row the first copy is the split piece.
  const text = `id: f\nname: f\ntypedefs:\n  R: ${written}\n${NODES}`;

  const error = loadError(() => parseSourceDocument(text, "dup.yaml"));

  expect(error.message).toBe(
    `duplicate key '${key}' (first at line ${at}); this \`{...}\` mapping's comma split the ` +
      `value that starts \`${cut}\`, so quote that value or use block style`,
  );
  // The second key, after the space before it, in 1-based columns.
  const column = text.split("\n")[3]!.lastIndexOf(` ${key}`) + 2;
  expect(error.diagnostic.frames.at(-1)!.range.startCol).toBe(column);
});

test("a repeat a flow mapping's comma made outside a type names the split value", () => {
  const text = `id: f\nname: f\nx-meta: {note: see (below, b, b}\n${NODES}`;

  const error = loadError(() => parseSourceDocument(text, "dup.yaml"));

  expect(error.message).toBe(
    "duplicate key 'b' (first at line 3); this `{...}` mapping's comma split the value that " +
      "starts `see (below`, so quote that value or use block style",
  );
});

test.each([
  ["a block mapping", "x-meta:\n  a: Dict(str\n  b: 1\n  b: 2\n"],
  ["a quoted value", "x-meta: {a: 'Dict(str', b: 1, b: 2}\n"],
  ["a balanced value", "x-meta: {a: Dict(str, int), b: 1, b: 2}\n"],
])("a repeat after %s keeps the plain message", (_label, block) => {
  const text = `id: f\nname: f\n${block}${NODES}`;

  const error = loadError(() => parseSourceDocument(text, "dup.yaml"));

  expect(error.message).toEndWith("a YAML mapping's keys must be unique, so remove or rename one");
});

test.each([
  "yaml/compose_duplicate_keys__on_and_true_are_two_keys_and_true_is_not_text",
  "yaml/compose_duplicate_keys__repeated_node_id",
  "yaml/compose_duplicate_keys__same_integer_keys_fail_as_non_text_first",
  "yaml/compose_duplicate_keys__shared_anchor_reports_once_at_the_definition",
])("parseSourceDocument: %s", (id) => runTextCase({ id, call: parseSourceDocument }));
