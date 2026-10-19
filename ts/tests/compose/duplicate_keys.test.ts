/**
 * A mapping that repeats one of its own keys is a load error: a port of
 * `tests/compose/test_duplicate_keys.py`, through `parseSourceDocument` and `parseFile`.
 *
 * Changed: TS keeps mapping keys as the text written, so `017` and `17` are two keys and the
 * document has no repeated key (Python's `repeated_key` is the int `17`); the non-text key still
 * fails first. The `x-meta` value of the `on`/`true` row has the text key `true`.
 *
 * Later (T3-22): the compiled-nodes check of `test_merged_keys_may_be_overridden`, which loads
 * the flow; its `parseFile` half is here.
 */

import { expect, test } from "bun:test";

import { parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { loadError } from "./declarations.ts";

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

test("merged keys may be overridden", () => {
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
  // Changed: the keys stay the text written, so `017` and `17` don't repeat.
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
