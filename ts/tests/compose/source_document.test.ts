/**
 * One-pass, privately retained YAML source documents: a port of
 * `tests/compose/test_source_document.py`.
 *
 * `test_document_uses_one_loader_instance` and the SafeLoader comparisons have no
 * counterpart: there is one compose pass and no SafeLoader to compare with. Python's three
 * `!!value` tests become refusal tests in TS, since YAML 1.2 has no `!!value`. Tags, merges,
 * anchors and the cursor-based checks come with the code that reads them. TS-era rows: a
 * mapping key YAML reads as non-text is kept in the value as written, lone CRs are line
 * breaks while NEL, U+2028 and U+2029 are not, and an undefined alias follows YAML 1.2.
 */

import { describe, expect, test } from "bun:test";

import { SourceSyntaxError } from "../../src/compose/source/common.ts";
import { type NonTextKey, SourceDocument } from "../../src/compose/source/document.ts";
import { SourceRange } from "../../src/diagnostics.ts";

/** Return the `SourceSyntaxError` parsing `text` throws, failing if it throws none. */
function refusal(text: string): SourceSyntaxError {
  try {
    SourceDocument.parse(text);
  } catch (error) {
    expect(error).toBeInstanceOf(SourceSyntaxError);
    return error as SourceSyntaxError;
  }
  throw new Error("expected a SourceSyntaxError");
}

test("a document has private retained slots", () => {
  const text = "count: 30\nflag: yes\n";
  const document = SourceDocument.parse(text, "flow.yaml");

  expect(document._value).toEqual({ count: 30n, flag: "yes" });
  expect(document._rootDefinition).toBe(0);
  expect(document._definitions.length).toBeGreaterThan(0);
  expect(Object.isFrozen(document._definitions)).toBe(true);
  expect(Object.isFrozen(document._definitions[0])).toBe(true);
  expect(document.text).toBe(text);
  expect(document.label).toBe("flow.yaml");
  expect("value" in document).toBe(false);
  expect("definitions" in document).toBe(false);
  expect(() => {
    (document as { label: string }).label = "other.yaml";
  }).toThrow(TypeError);
});

test("a syntax error has the problem's range", () => {
  expect(refusal("nodes:\n  broken: ]\n").range).toEqual(new SourceRange(2, 11, 2, 12));
});

describe("a character the reader refuses", () => {
  const MESSAGE = "unacceptable character #x0000: special characters are not allowed";

  test.each([
    ["lf", "\n"],
    ["cr", "\r"],
    ["crlf", "\r\n"],
  ])("is located after a line break: %s", (_, lineBreak) => {
    const error = refusal(`first: ok${lineBreak}bad:\x00 value\n`);

    expect(error.message).toBe(MESSAGE);
    expect(error.range).toEqual(new SourceRange(2, 5, 2, 6));
  });

  test.each([
    ["nel", "\x85"],
    ["ls", "\u2028"],
    ["ps", "\u2029"],
  ])("stays on line 1 after a character that is not a line break: %s", (_, lineBreak) => {
    const error = refusal(`first: ok${lineBreak}bad:\x00 value\n`);

    expect(error.message).toBe(MESSAGE);
    // Python gives (2, 5, 2, 6): PyYAML also breaks lines at NEL, U+2028 and U+2029
    // (`rule: columns`).
    expect(error.range).toEqual(new SourceRange(1, 15, 1, 16));
  });

  test("includes a lone surrogate", () => {
    const error = refusal("a: \uD800\n");

    expect(error.message).toBe(
      "unacceptable character #xd800: special characters are not allowed",
    );
    expect(error.range).toEqual(new SourceRange(1, 4, 1, 5));
  });
});

test("a document refuses a second document", () => {
  const error = refusal("first: 1\n---\nsecond: 2\n");

  expect(error.message).toContain("another document");
  expect(error.range).toEqual(new SourceRange(2, 1, 2, 2));
});

test("a document accepts an empty document", () => {
  const document = SourceDocument.parse("");

  expect(document._value).toBeNull();
  expect(document._rootDefinition).toBe(0);
  expect(document._definitions).toHaveLength(1);
});

test("an unclosed flow sequence is located at the end of the text", () => {
  expect(refusal("id: x\nname: x\nnodes: [\n").range).toEqual(new SourceRange(4, 1, 4, 2));
});

test("lone CRs are line breaks; the text is kept as written", () => {
  const text = 'a: x\r  y\rb: "p\r  q"\rc: |\r  m\r  n\r';
  const document = SourceDocument.parse(text);

  expect(document._value).toEqual({ a: "x y", b: "p q", c: "m\nn\n" });
  expect(document.text).toBe(text);
  const root = document._definitions[document._rootDefinition];
  if (root?.kind !== "mapping") throw new Error("expected a mapping root");
  const valueRange = (key: string) => {
    const occurrence = root.effectiveIndex.get(key);
    return document._definitions[occurrence!.arc.valueDefinition]?.range;
  };
  expect(valueRange("a")).toEqual(new SourceRange(1, 4, 2, 4));
  expect(valueRange("b")).toEqual(new SourceRange(3, 4, 4, 5));
});

test("an alias name runs up to a space, as in YAML 1.2", () => {
  expect(refusal("{*k: 1}\n").message).toBe("found undefined alias 'k:'");
});

describe("nonTextKey", () => {
  test("is the constructed key, its text and its range", () => {
    expect(SourceDocument.parse("a: 1\nTrue: x\n").nonTextKey).toEqual({
      key: true,
      text: "True",
      range: new SourceRange(2, 1, 2, 5),
    });
  });

  test.each<[string, NonTextKey["key"]]>([
    ["true", true],
    ["FALSE", false],
    ["1", 1n],
    ["-3", -3n],
    ["0x1F", 31n],
    ["1.5", 1.5],
    ["1e3", 1000],
    ["null", null],
    ["~", null],
  ])("reports each kind of scalar: %p", (raw, key) => {
    const document = SourceDocument.parse(`${raw}: x\n`);
    const found = document.nonTextKey;

    expect(found?.key).toBe(key);
    expect(found?.text).toBe(raw);
    expect([found?.range.startLine, found?.range.startCol]).toEqual([1, 1]);
    // The value keeps the key as written.
    expect(document._value).toEqual({ [raw]: "x" });
  });

  test.each([
    "yes",
    "on",
    "off",
    "no",
    "1:30",
    "1_000",
    '"True"',
    "'1'",
    "=",
    "2026-10-13",
    "2026-10-13 10:00:00",
    "2026-10-13T10:00:00Z",
  ])("leaves a text key out: %p", (raw) => {
    expect(SourceDocument.parse(`${raw}: x\n`).nonTextKey).toBeNull();
  });

  test.each<[string, string, NonTextKey["key"], SourceRange]>([
    ["sequence item", "a:\n  - {1: x}\n", 1n, new SourceRange(2, 6, 2, 7)],
    ["flow mapping value", "a: {b: {2.5: x}}\n", 2.5, new SourceRange(1, 9, 1, 12)],
    ["deep value", "a:\n  b:\n    c:\n      null: x\n", null, new SourceRange(4, 7, 4, 11)],
    ["earlier value beats later key", "a: {2: x}\n3: y\n", 2n, new SourceRange(1, 5, 1, 6)],
    ["earlier key beats later value", "1: x\nb: {2: y}\n", 1n, new SourceRange(1, 1, 1, 2)],
  ])("is found anywhere, in document order: %s", (_, text, key, range) => {
    const found = SourceDocument.parse(text).nonTextKey;

    expect(found?.key).toBe(key);
    expect(found?.range).toEqual(range);
  });

  // Python gives one column earlier: eemeli puts an empty node after the space, PyYAML
  // before it (`rule: yaml`).
  test.each([
    ["explicit", "? \n: x\n", new SourceRange(1, 3, 1, 3)],
    ["flow", "{? : x}\n", new SourceRange(1, 4, 1, 4)],
  ])("reports an empty key with empty text: %s", (_, text, range) => {
    expect(SourceDocument.parse(text).nonTextKey).toEqual({ key: null, text: "", range });
  });
});

describe("repeatedKey", () => {
  test("compares keys as text, so 017 and 17 differ", () => {
    const document = SourceDocument.parse("017: a\n17: b\n");

    expect(document.nonTextKey).toEqual({
      key: 17n,
      text: "017",
      range: new SourceRange(1, 1, 1, 4),
    });
    expect(document.repeatedKey).toBeNull();
    expect(document._value).toEqual({ "017": "a", "17": "b" });
  });

  test("is the first key a mapping repeats, with both ranges", () => {
    expect(SourceDocument.parse("a: 1\nb: 2\na: 3\n").repeatedKey).toEqual({
      key: "a",
      first: new SourceRange(1, 1, 1, 2),
      second: new SourceRange(3, 1, 3, 2),
    });
  });

  test("is not two pairs keyed by one alias target", () => {
    expect(SourceDocument.parse("x: &k a\ny: {*k : 1, *k : 2}").repeatedKey).toBeNull();
  });
});

test("a list or mapping key is refused", () => {
  const error = refusal("[a]: 1\n");

  expect(error.message).toBe("a list or mapping can't be a mapping key; mapping keys must be text");
  expect(error.range).toEqual(new SourceRange(1, 1, 1, 2));
});

test("a missing value is null", () => {
  expect(SourceDocument.parse("{a}")._value).toEqual({ a: null });
});
