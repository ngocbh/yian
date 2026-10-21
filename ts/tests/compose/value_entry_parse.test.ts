/**
 * Classifying `nodes:` entries, a node has `kind:` and a value has a YAML tag: a port of
 * `tests/compose/test_value_entry_parse.py`. The Python tests read the flow with `load_flow`;
 * these read its `nodes:` mapping with `parseNodes`, so the entry under test comes first and the
 * `code` node after it is parsed, by the registered `CodeDescriptor`, only by the rows that load
 * every entry.
 *
 * Re-expected: TS reads unquoted dates as text (rule `dates`), so the four date rows of
 * `test_untagged_scalar_or_list_shows_the_tag_to_write` are deliberate TS rows, not Python's: a
 * list holding a date is a list of mixed items or of text, and a lone date is `!str`. Infinity is
 * a float value in TS (rule `infinity`), so an untagged `.inf` gets its example, and the YAML
 * layer refuses `.nan` before `parseNodes` sees it.
 *
 * TS-only: `!!str 3`, whose example quotes the text since plain `3` reads as an int (Python checks
 * it through `test_the_example_entry_loads`), and `exampleType` and `containsItself` called
 * directly.
 *
 * The harvested `load-case/compose_value_entry_parse__*` cases are pinned at the end; each fails
 * at its first entry, so no stand-in is registered.
 *
 * Later (T4-8): `test_the_example_entry_loads`, which loads the example and runs it.
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { containsItself, exampleType, parseNodes } from "../../src/compose/parse/nodes.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";
import { loadError, nodesOf, runTextCase } from "./declarations.ts";
import { standIn } from "./stand_ins.ts";

afterEach(() => forget());

const NODE_OR_VALUE =
  "node 'x': an entry in `nodes:` is a node or a value; add `kind:` for a node, " +
  "or a tag (`!object`, `!Buy`, ...) for a value";

/** A one-off kind whose descriptors stand in for read value entries. */
const ValueEntry = standIn("value_entry");

/** Return a flow whose `nodes:` holds `entry` (already indented) and a CODE node. */
function flow(entry: string): string {
  return (
    "id: f\n" +
    "name: f\n" +
    "nodes:\n" +
    entry +
    "  step:\n" +
    "    kind: code\n" +
    "    code: |\n" +
    "      return 1\n" +
    "output: ${step.output}\n"
  );
}

/** Return the 1-based line number of the first line containing `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Return the start line of the innermost frame of a load error. */
function errorLine(error: LoadError): number {
  const range = error.diagnostic.frames.at(-1)!.range;
  expect(range).not.toBeNull();
  return range!.startLine;
}

/** Return the `nodes:` cursor of a parsed document. */
function nodesCursor(text: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes");
}

/** Parse the `nodes:` of `text`, expecting a load error. */
function parseError(text: string): LoadError {
  return loadError(() => parseNodes(nodesCursor(text)));
}

test("untagged mapping without kind says add kind or a tag", () => {
  const text = flow("  x:\n    prompt: hi\n");

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(NODE_OR_VALUE);
  expect(errorLine(error)).toBe(lineOf(text, "prompt: hi"));
});

describe("untagged scalar or list shows the tag to write", () => {
  test.each([
    ["  x: 5\n", "`x: !int 5`"],
    ["  x: hello\n", "`x: !str hello`"],
    ['  x: "hi"\n', '`x: !str "hi"`'],
    ["  x: [1, 2]\n", "`x: !list(int) [1, 2]`"],
    ["  x: [[1, 2], [3]]\n", "`x: !list(list(int)) [[1, 2], [3]]`"],
    ["  x: [null, 1]\n", "`x: !list(Optional(int)) [null, 1]`"],
    ["  x: [1, 2.5]\n", "`x: !list(float) [1, 2.5]`"],
    ["  x: [[1], [null]]\n", "`x: !list(list(Optional(int))) [[1], [null]]`"],
    ["  x: [[], [1]]\n", "`x: !list(list(int)) [[], [1]]`"],
    ["  x: [1, a]\n", "`x: !list(Any) [1, a]`"],
    ["  x: [[1], [a]]\n", "`x: !list(Any) [[1], [a]]`"],
    ["  x: [{a: 1}, 2]\n", "`x: !list(Any) [{a: 1}, 2]`"],
    ["  x: [[1, null], a]\n", "`x: !list(Any) [[1, null], a]`"],
    ["  x: [null, 1, a]\n", "`x: !list(Optional(Any)) [null, 1, a]`"],
    ["  x: [[1, a], [b, 2]]\n", "`x: !list(list(Any)) [[1, a], [b, 2]]`"],
    // Dates are text in TS, so these lists hold mixed items and need no quotes, where Python
    // asks to quote the dates.
    ["  x: [1, 2024-01-01]\n", "`x: !list(Any) [1, 2024-01-01]`"],
    ["  x: [null, [2024-01-01], a]\n", "`x: !list(Optional(Any)) [null, [2024-01-01], a]`"],
    ["  x:\n", "`x: !Optional(str) null`"],
    // A lone date is text in TS, where Python suggests `!date` or `!datetime` with quotes.
    ["  x: 2024-01-01\n", "`x: !str 2024-01-01`"],
    ["  x: 2024-01-01T10:00:00\n", "`x: !str 2024-01-01T10:00:00`"],
    // Plain `3` would read back as an int, so the example quotes the text.
    ["  x: !!str 3\n", '`x: !str "3"`'],
    // Infinity is a float value in TS (rule `infinity`), and `x: !float .inf` loads; Python gives
    // no example.
    ["  x: .inf\n", "`x: !float .inf`"],
  ])("%j", (entry, example) => {
    const text = flow(entry);

    const error = parseError(text);

    expect(error.diagnostic.message).toBe(`${NODE_OR_VALUE}, e.g. ${example}`);
    expect(errorLine(error)).toBe(lineOf(text, "  x:"));
  });
});

describe("an untagged value no tag can hold gets no example", () => {
  test.each(["  x: &R [*R]\n", "  x: &R [{a: *R}]\n"])("%j", (entry) => {
    const text = flow(entry);

    const error = parseError(text);

    expect(error.diagnostic.message).toBe(NODE_OR_VALUE);
    expect(errorLine(error)).toBe(lineOf(text, "  x:"));
  });

  test("a NaN is refused before the entry is parsed", () => {
    // NaN is not a yian value, so the YAML layer refuses `.nan` while it parses the document;
    // Python reads it and gives the entry no example.
    expect(() => nodesCursor(flow("  x: [{a: .nan}]\n"))).toThrow("NaN is not a yian value");
  });
});

test("untagged block list shows the tag with a placeholder value", () => {
  const text = flow("  x:\n    - 1\n    - 2\n");

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(`${NODE_OR_VALUE}, e.g. \`x: !list(int) ...\``);
  expect(errorLine(error)).toBe(lineOf(text, "- 1"));
});

test("an untagged credential-named literal gets the example", () => {
  const text = flow("  api_key: sk-live-123\n");

  const error = parseError(text);

  expect(error.diagnostic.message).toBe(
    NODE_OR_VALUE.replace("node 'x'", "node 'api_key'") + ", e.g. `api_key: !str sk-live-123`",
  );
  expect(errorLine(error)).toBe(lineOf(text, "api_key:"));
});

describe("an untagged flow call or span gets no example", () => {
  test.each([
    ["  x: call(f)\n", "x"],
    ["  api_key: call(f)\n", "api_key"],
    ["  x: ${step.output}\n", "x"],
  ])("%j", (entry, nodeId) => {
    const text = flow(entry);

    const error = parseError(text);

    expect(error.diagnostic.message).toBe(NODE_OR_VALUE.replace("node 'x'", `node '${nodeId}'`));
    expect(errorLine(error)).toBe(lineOf(text, `  ${nodeId}:`));
  });
});

describe("reserved ids apply to a tagged entry", () => {
  test.each(["  input: !int 5\n", "  input:\n    kind: code\n"])("%j", (entry) => {
    const message =
      "node id 'input' is reserved (used as `${input.X}` as a resolver head — pick " +
      "a different node id)";

    expect(parseError(flow(entry)).diagnostic.message).toBe(message);
  });
});

test("tagged entries are read by the callback in source order", () => {
  const cursor = nodesCursor(
    "nodes:\n" +
      "  a:\n" +
      "    kind: code\n" +
      "    code: return 1\n" +
      "  x: !int 5\n" +
      "  d: !Buy {a: 1}\n" +
      "  b:\n" +
      "    kind: code\n" +
      "    code: return 2\n",
  );
  const calls: [string, string | null, unknown][] = [];
  const read = new Map<string, NodeDescriptor>();
  const readValue = (nodeId: string, entry: SourceCursor): NodeDescriptor => {
    calls.push([nodeId, entry.tag, entry.value]);
    const desc = new ValueEntry({ id: nodeId });
    read.set(nodeId, desc);
    return desc;
  };

  const descriptors = parseNodes(cursor, { readValue });

  expect([...descriptors.keys()]).toEqual(["a", "x", "d", "b"]);
  expect(calls).toEqual([
    ["x", "int", 5n],
    ["d", "Buy", { a: 1n }],
  ]);
  expect(descriptors.get("x")).toBe(read.get("x")!);
  expect(descriptors.get("d")).toBe(read.get("d")!);
  expect(descriptors.get("a")).toBeInstanceOf(CodeDescriptor);
  expect(descriptors.get("b")).toBeInstanceOf(CodeDescriptor);
});

test("a tagged mapping with kind is still a value", () => {
  const cursor = nodesCursor("nodes:\n  v: !object {kind: code, code: return 1}\n");
  const value = new ValueEntry({ id: "v" });

  const descriptors = parseNodes(cursor, { readValue: () => value });

  expect([...descriptors]).toEqual([["v", value]]);
});

test("a tagged entry without a reader is a type error", () => {
  const message = "node 'x' is a tagged value entry; parseNodes needs readValue to read it";

  expect(() => parseNodes(nodesCursor("nodes:\n  x: !int 5\n"))).toThrow(
    new TypeError(message),
  );
});

describe("exampleType and containsItself", () => {
  test.each([
    ["Infinity", Infinity, "float"],
    ["-Infinity", -Infinity, "float"],
    ["null", null, "Optional(str)"],
    ["[null]", [null], "list(Optional(str))"],
    ["[]", [], "list(Optional(Any))"],
    ["{a: 1}", { a: 1n }, "object"],
    ["[1, 2.5]", [1n, 2.5], "list(float)"],
  ])("exampleType(%s)", (_name, value, label) => {
    expect(canonicalTypeLabel(exampleType(value)!)).toBe(label);
  });

  test("a value no type holds has no example type", () => {
    expect(exampleType(new Date(0))).toBeNull();
    expect(exampleType([1n, new Date(0)])).toBeNull();
  });

  test("a list or record holding itself contains itself", () => {
    const list: unknown[] = [];
    list.push({ a: list });
    const shared = [1n];

    expect(containsItself(list)).toBe(true);
    expect(containsItself([shared, shared])).toBe(false);
    expect(containsItself("x")).toBe(false);
  });
});

test.each([
  "load-case/compose_value_entry_parse__an_empty_entry_shows_a_null_tag",
  "load-case/compose_value_entry_parse__an_untagged_block_list_shows_a_placeholder",
  "load-case/compose_value_entry_parse__an_untagged_int_shows_its_tag",
  "load-case/compose_value_entry_parse__an_untagged_list_shows_its_tag",
  "load-case/compose_value_entry_parse__an_untagged_mixed_list_shows_list_of_any",
  "load-case/compose_value_entry_parse__an_untagged_quoted_text_shows_its_tag",
  "load-case/compose_value_entry_parse__reserved_ids_apply_to_a_tagged_entry",
])("parseNodes: %s", (id) => runTextCase({ id, call: (text, label) => nodesOf(text, [], label) }));
