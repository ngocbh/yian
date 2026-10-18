/**
 * One-pass, privately retained YAML source documents: a port of
 * `tests/compose/test_source_document.py`.
 *
 * `test_document_uses_one_loader_instance` and the SafeLoader comparisons have no
 * counterpart: there is one compose pass and no SafeLoader to compare with. Python's three
 * `!!value` tests become refusal tests in TS, since YAML 1.2 has no `!!value`; `!!binary`,
 * `!!set`, `!!omap` and `!!pairs` are refused too; the merge texts compare with hand-written
 * values. The cursor-based checks come with the cursor. The tag rows that read the document
 * (merge-source tags included) and the merge-rope size and literal-line rows of
 * `test_source_cursor.py` read definitions here; `test_source_tags.py` itself is ported in
 * `source_tags.test.ts`, with the cursor. TS-era rows: a mapping key YAML reads as
 * non-text is kept in the value as written, lone CRs are line breaks while NEL, U+2028 and
 * U+2029 are not, an undefined alias follows YAML 1.2, a local tag never changes how its
 * scalar reads, and the tag hint names a tag glued to `,`, `[`, `]`, `{` or `}`.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { parse as parseYaml } from "yaml";

import {
  type ArcSequence,
  type Definition,
  SourceSyntaxError,
} from "../../src/compose/source/common.ts";
import { SourceCursor } from "../../src/compose/source/cursor.ts";
import {
  type NonTextKey,
  SourceDocument,
  TAG_HINT,
} from "../../src/compose/source/document.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import type { Value } from "../../src/typesys/index.ts";

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

test("root is a cursor at the root definition", () => {
  const document = SourceDocument.parse("a: 1\n");
  const root = document.root;

  expect(root).toBeInstanceOf(SourceCursor);
  expect(root.document).toBe(document);
  expect(root._definition).toBe(0);
  expect(root.path).toEqual([]);
  expect(root.value).toEqual({ a: 1n });
});

/** Return the definition at `path` (mapping keys and list indexes) in `text`'s document. */
function definitionAt(text: string, ...path: (string | number)[]): Definition {
  const document = SourceDocument.parse(text);
  let definition = document._definitions[document._rootDefinition]!;
  for (const step of path) {
    const id =
      definition.kind === "mapping"
        ? definition.effectiveIndex.get(step as string)!.arc.valueDefinition
        : definition.kind === "sequence"
          ? definition.itemDefinitions[step as number]!
          : null;
    if (id === null) throw new Error(`no definition at ${String(step)}`);
    definition = document._definitions[id]!;
  }
  return definition;
}

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
    expect(Object.isFrozen(document.nonTextKey)).toBe(true);
  });

  test("is the first key a mapping repeats, with both ranges", () => {
    const { repeatedKey } = SourceDocument.parse("a: 1\nb: 2\na: 3\n");

    expect(repeatedKey).toEqual({
      key: "a",
      first: new SourceRange(1, 1, 1, 2),
      second: new SourceRange(3, 1, 3, 2),
    });
    expect(Object.isFrozen(repeatedKey)).toBe(true);
    expect(Object.isFrozen(repeatedKey!.first)).toBe(true);
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

test.each([
  ["!!seq a: 1", new SourceRange(1, 1, 1, 2)],
  ["!!map a: 1", new SourceRange(1, 1, 1, 2)],
  ["? !!seq a\n: 1", new SourceRange(1, 3, 1, 4)],
  ["{!!seq a: 1}", new SourceRange(1, 2, 1, 3)],
  ["!!map on: !!bool", new SourceRange(1, 1, 1, 2)],
])("a scalar key tagged as a list or mapping is refused before its value: %p", (text, range) => {
  const error = refusal(text);

  expect(error.message).toBe("a list or mapping can't be a mapping key; mapping keys must be text");
  expect(error.range).toEqual(range);
});

test("a missing value is null", () => {
  expect(SourceDocument.parse("{a}")._value).toEqual({ a: null });
});

describe("property ranges", () => {
  afterEach(() => install(BUILTIN));

  /** Return the range of the definition at `path`. */
  function rangeAt(text: string, ...path: (string | number)[]): SourceRange | null {
    return definitionAt(text, ...path).range;
  }

  /** Return the start line and column of the definition at `path`. */
  function startAt(text: string, ...path: (string | number)[]): [number, number] {
    const range = rangeAt(text, ...path)!;
    return [range.startLine, range.startCol];
  }

  test.each<[string, (string | number)[], SourceRange]>([
    ["a: &x !int 5\n", ["a"], new SourceRange(1, 4, 1, 13)],
    ["a: !int\n", ["a"], new SourceRange(1, 4, 1, 8)],
    ["a: !int   \n", ["a"], new SourceRange(1, 4, 1, 8)],
    ["a: !int # c\n", ["a"], new SourceRange(1, 4, 1, 8)],
    ["- !a\n- x\n", [0], new SourceRange(1, 3, 1, 5)],
  ])("include a node's anchor and tag: %p", (text, path, range) => {
    expect(rangeAt(text, ...path)).toEqual(range);
  });

  test.each<[string, (string | number)[], [number, number]]>([
    ["!top\nk: v\n", [], [1, 1]],
    ["- &x !a [1]\n", [0], [1, 3]],
  ])("start a list or mapping at its properties: %p", (text, path, start) => {
    expect(startAt(text, ...path)).toEqual(start);
  });

  test.each<[string, (string | number)[], SourceRange]>([
    ["{a: !int, b: !str}\n", ["a"], new SourceRange(1, 5, 1, 9)],
    ["{a: !int, b: !str}\n", ["b"], new SourceRange(1, 14, 1, 18)],
    ["{a: !int , b: 1}\n", ["a"], new SourceRange(1, 5, 1, 9)],
    ["[!a, b]\n", [0], new SourceRange(1, 2, 1, 4)],
    ["[!a , b]\n", [0], new SourceRange(1, 2, 1, 4)],
    ["[&x , b]\n", [0], new SourceRange(1, 2, 1, 4)],
    ["[!a, &b , !c]\n", [0], new SourceRange(1, 2, 1, 4)],
    ["[!a, &b , !c]\n", [1], new SourceRange(1, 6, 1, 8)],
    ["[!a, &b , !c]\n", [2], new SourceRange(1, 11, 1, 13)],
  ])("make an empty value in a flow collection span its properties: %p", (text, path, range) => {
    expect(rangeAt(text, ...path)).toEqual(range);
  });

  test.each<[string, (string | number)[], [number, number]]>([
    ["&k a: 1\n", [], [1, 1]],
    ["- &k a: 1\n", [0], [1, 3]],
    ["x:\n  !t b: 1\n", ["x"], [2, 3]],
    ["? &k a\n: 1\n", [], [1, 1]],
    ["- ? !t a\n  : 1\n", [0], [1, 3]],
  ])("start a block mapping at its first key's properties: %p", (text, path, start) => {
    expect(startAt(text, ...path)).toEqual(start);
  });

  test.each<[string, (string | number)[], [number, number]]>([
    ["[&k a: 1]\n", [0], [1, 2]],
    ["[!t a: 1, b]\n", [0], [1, 2]],
    ["[&k : 1]\n", [0], [1, 2]],
    ["- [&k a: 1]\n", [0, 0], [1, 4]],
  ])("start a one-pair flow mapping at its key's properties: %p", (text, path, start) => {
    expect(startAt(text, ...path)).toEqual(start);
  });

  test.each([
    ["&k 1: a\nb: {*k : c}\n", new SourceRange(1, 1, 1, 5)],
    ["a: &k 1\n*k : b\n", new SourceRange(1, 4, 1, 8)],
  ])("make a non-text key span its properties: %p", (text, range) => {
    expect(SourceDocument.parse(text).nonTextKey).toEqual({ key: 1n, text: "1", range });
  });

  test("locate an undefined alias at the alias", () => {
    const error = refusal("x: &a [*a, *b]");

    expect(error.message).toBe("found undefined alias 'b'");
    expect(error.range).toEqual(new SourceRange(1, 12, 1, 13));
  });

  test("locate a nesting-limit error at the list's anchor", () => {
    install({ ...BUILTIN, maxYamlNesting: 1 });
    const error = refusal("x: &a [1]\n");

    expect(error.message).toContain("this list nests deeper than 1 levels");
    expect(error.range).toEqual(new SourceRange(1, 4, 1, 5));
  });

  test("are frozen", () => {
    expect(Object.isFrozen(rangeAt("a: 1\n", "a"))).toBe(true);
    expect(Object.isFrozen(refusal("a: ]\n").range)).toBe(true);
  });
});

describe("range ends", () => {
  // PyYAML's ends: a block collection ends at the next token past blank lines and comments,
  // a block scalar after the blank lines that follow it.
  test.each<[string, (string | number)[], SourceRange]>([
    ["- &n0\n  <<: *n0\n  c: ''\n- on", [], new SourceRange(1, 1, 4, 5)],
    ["- &n0\n  <<: *n0\n  c: ''\n- on", [0], new SourceRange(1, 3, 4, 1)],
    ["root:\n  -\n    - x y\n    - \n", [], new SourceRange(1, 1, 5, 1)],
    ["root:\n  -\n    - x y\n    - \n", ["root"], new SourceRange(2, 3, 5, 1)],
    ["root:\n  -\n    - x y\n    - \n", ["root", 0], new SourceRange(3, 5, 5, 1)],
    ["root: !int |\n    t\n\n", ["root"], new SourceRange(1, 7, 4, 1)],
    ["a:\n  b: 1\n  # c\n# d\nc: 2\n", ["a"], new SourceRange(2, 3, 5, 1)],
    ["a: |-\n  t\n  \n\nb: 1", ["a"], new SourceRange(1, 4, 5, 1)],
    ["a:\r\n  b: 1\r\n  # x\r\n\r\nc: 2\r\n", ["a"], new SourceRange(2, 3, 5, 1)],
    ["a: [1,\n  2]\n\nb: 1", ["a"], new SourceRange(1, 4, 2, 5)],
    ["a: 'q'  # c\n\nb: 1", ["a"], new SourceRange(1, 4, 1, 7)],
  ])("are PyYAML's: %p at %p", (text, path, range) => {
    expect(definitionAt(text, ...path).range).toEqual(range);
  });

  test("frame a block record up to the next key in the shared depends_on case", () => {
    const url = new URL(
      "../../../conformance/cases/run-agent/seed_12_depends_on.yaml",
      import.meta.url,
    );
    const { flow } = parseYaml(readFileSync(url, "utf8")) as { flow: string };

    // The frame the case expects for `fetch`'s declared output record.
    expect(definitionAt(flow, "nodes", "fetch", "output").range).toEqual(
      new SourceRange(34, 7, 35, 5),
    );
  });
});

describe("core tags", () => {
  // Python's three `!!value` texts respell the alias `*equals :` in a flow mapping, where
  // `*equals:` is the alias `equals:` in YAML 1.2.
  test.each([
    ["a: !!binary YQ==\n", "binary"],
    ["a: !!set {x: null}\n", "set"],
    ["a: !!omap [{x: 1}]\n", "omap"],
    ["a: !!pairs [{x: 1}]\n", "pairs"],
    ["!!value =: x\n", "value"],
    ["!!binary aGVsbG8=: x\n", "binary"],
    ['!!binary "": x\n', "binary"],
    ["first: &equals !!value =\n? *equals\n: aliased-key\n", "value"],
    ["first: &equals !!value =\nnested: {*equals : aliased-key}\n", "value"],
    ["target:\n  first: &equals !!value =\n  <<: {*equals : inherited}\n", "value"],
  ])("without a yian constructor are refused: %p", (text, name) => {
    const error = refusal(text);

    expect(error.message).toBe(
      `could not determine a constructor for the tag 'tag:yaml.org,2002:${name}'`,
    );
    expect(error.range).not.toBeNull();
  });

  test("an alias glued to `:` in a flow mapping is undefined", () => {
    const error = refusal("first: &equals !!value =\nnested: {*equals: aliased-key}\n");

    expect(error.message).toBe("found undefined alias 'equals:'");
  });

  test.each([
    ["a: !!str [1]\n", "expected a scalar node, but found sequence"],
    ["a: !!seq x\n", "expected a sequence node, but found scalar"],
    ["a: !!map [1]\n", "expected a mapping node, but found sequence"],
    ["a: !!seq {b: 1}\n", "expected a sequence node, but found mapping"],
  ])("of the wrong kind are refused: %p", (text, message) => {
    expect(refusal(text).message).toBe(message);
  });

  test("a !!seq scalar is refused after its sibling scalars", () => {
    expect(refusal("[!!seq x, 2026-13-45]\n").message).toBe(
      "not a valid date: month must be in 1..12; quote it to keep it as text",
    );
  });

  test("!!seq and !!map pick the constructor", () => {
    expect(SourceDocument.parse("a: !!seq [1]\nb: !!map {c: 2}\n")._value).toEqual({
      a: [1n],
      b: { c: 2n },
    });
  });
});

describe("bare and unknown tags", () => {
  test("a bare tag is an error over its node", () => {
    const error = refusal("a: ! x\n");

    expect(error.message).toBe("a tag names a type, e.g. `!int`");
    expect(error.range).toEqual(new SourceRange(1, 4, 1, 7));
  });

  test.each([
    ["a: ! [1]\n", new SourceRange(1, 4, 1, 7)],
    ["root: ! ['q', &n1 {b: c}]\n", new SourceRange(1, 7, 1, 10)],
    ["root: !\n  1: x\n", new SourceRange(1, 7, 2, 3)],
    ["- !\n  - 1\n", new SourceRange(1, 3, 2, 3)],
  ])("a bare tag on a collection is an error up to its opening token: %p", (text, range) => {
    const error = refusal(text);

    expect(error.message).toBe("a tag names a type, e.g. `!int`");
    expect(error.range).toEqual(range);
  });

  test("a bare tag is checked after the YAML syntax", () => {
    expect(refusal("a: ! x\nb: ]\n").range).toEqual(new SourceRange(2, 4, 2, 5));
  });

  test.each([
    ["%TAG ! tag:example.com:\n---\na: !int 5\n", "'tag:example.com:int'", 3],
    ["a: !<tag:example.com:x> 5\n", "'tag:example.com:x'", 1],
  ])("an unknown global tag is an error: %p", (text, tag, line) => {
    const error = refusal(text);

    expect(error.message).toBe(`unknown YAML tag ${tag}; a type tag is local, e.g. \`!int\``);
    expect(error.range?.startLine).toBe(line);
  });

  test("a verbatim local tag is a local tag", () => {
    const definition = definitionAt("a: !<!int> 5\n", "a");

    expect(definition.tag).toBe("int");
    expect(definition.value).toBe(5n);
  });
});

describe("the tag hint", () => {
  test.each(["d: !Union(Buy, Sell) x\n", "d: !dict(str,int)\n", "d: !list[str] x\n"])(
    "follows an error at a character glued to a tag: %p",
    (text) => {
      expect(refusal(text).message.endsWith("\n" + TAG_HINT)).toBe(true);
    },
  );

  // An anchor glued to `,` and a tag glued to `"` get eemeli's MISSING_CHAR too.
  test.each(["m: {a: 1, b: c: d}\n", "{a: !int, b: c: d}\n", "a: &x, b\n", 'a: !x"q"\n'])(
    "is not added to another YAML error: %p",
    (text) => {
      expect(refusal(text).message).not.toContain(TAG_HINT);
    },
  );

  test("a tag ends at a flow indicator", () => {
    expect(SourceDocument.parse("m: {a: !int, b: !str}\n")._value).toEqual({
      m: { a: null, b: null },
    });
  });
});

describe("local tags", () => {
  test.each<[string, Value]>([
    ["a: !float 3\n", 3n],
    ["a: !int '5'\n", "5"],
    ['a: !int "5"\n', "5"],
    ["a: !x <<\n", "<<"],
    ["a: !str |\n  5\n", "5\n"],
    ["a: !int\n", null],
    ["a: !x ~\n", null],
    ["a: !bool yes\n", "yes"],
    ["a: !str 1.5\n", 1.5],
    ["a: !int 0x10\n", 16n],
  ])("read a scalar as if untagged: %p", (text, value) => {
    expect(SourceDocument.parse(text)._value).toEqual({ a: value });
  });

  test("are recorded on the definition", () => {
    expect(definitionAt("a: !float 3\n", "a").tag).toBe("float");
    expect(definitionAt("d: !Buy\n  conviction: 1\n", "d").tag).toBe("Buy");
    expect(definitionAt("h: !list(message) []\n", "h").tag).toBe("list(message)");
    expect(definitionAt("a: !!str 5\n", "a").tag).toBeNull();
    expect(definitionAt("a: 5\n", "a").tag).toBeNull();
  });

  test("read a tagged key as if untagged", () => {
    const document = SourceDocument.parse("!int 1: one\n!x <<: two\n");

    expect(document._value).toEqual({ "1": "one", "<<": "two" });
    expect(document.nonTextKey).toEqual({
      key: 1n,
      text: "1",
      range: new SourceRange(1, 1, 1, 7),
    });
  });

  test("a core tag can make a key non-text or text", () => {
    expect(SourceDocument.parse('!!int "7": x\n').nonTextKey?.key).toBe(7n);
    expect(SourceDocument.parse("!!str 1: x\n").nonTextKey).toBeNull();
  });
});

test.each([
  ["a: !!int &x 1_000\n", 1, 4],
  ["[1, 2026-13-45]\n", 1, 5],
  ["a:\n  2026-13-45: 1\n", 2, 3],
])("a scalar error is located at the node and its properties: %p", (text, line, column) => {
  const range = refusal(text).range;

  expect([range?.startLine, range?.startCol]).toEqual([line, column]);
});

describe("literal line ranges", () => {
  /** Return the constructed line ranges of the scalar at `code`, after checking its value. */
  function linesOf(text: string, value: string, key = "code"): readonly SourceRange[] | null {
    const definition = definitionAt(text, key);
    if (definition.kind !== "scalar") throw new Error("expected a scalar");
    expect(definition.value).toBe(value);
    return definition.constructedLineRanges;
  }

  test("lone CRs are line breaks", () => {
    expect(linesOf('a: x\r  y\rb: "p\r  q"\rc: |\r  m\r  n\r', "m\nn\n", "c")).toEqual([
      new SourceRange(6, 3, 6, 4),
      new SourceRange(7, 3, 7, 4),
    ]);
  });

  test.each<[string, string, string, SourceRange[]]>([
    [
      "clip",
      "code: |\n  alpha\n  beta\n",
      "alpha\nbeta\n",
      [new SourceRange(2, 3, 2, 8), new SourceRange(3, 3, 3, 7)],
    ],
    [
      "strip",
      "code: |-\n  alpha\n  beta\n",
      "alpha\nbeta",
      [new SourceRange(2, 3, 2, 8), new SourceRange(3, 3, 3, 7)],
    ],
    [
      "keep",
      "code: |+\n  alpha\n  beta\n\n\n",
      "alpha\nbeta\n\n\n",
      [
        new SourceRange(2, 3, 2, 8),
        new SourceRange(3, 3, 3, 7),
        new SourceRange(4, 1, 4, 1),
        new SourceRange(5, 1, 5, 1),
      ],
    ],
    [
      "explicit indent",
      "code: |2-\n    first\n      second\n",
      "  first\n    second",
      [new SourceRange(2, 3, 2, 10), new SourceRange(3, 3, 3, 13)],
    ],
    [
      "leading, interior and trailing blank lines",
      "code: |+\n\n  first\n  \n  second\n\n",
      "\nfirst\n\nsecond\n\n",
      [
        new SourceRange(2, 1, 2, 1),
        new SourceRange(3, 3, 3, 8),
        new SourceRange(4, 3, 4, 3),
        new SourceRange(5, 3, 5, 9),
        new SourceRange(6, 1, 6, 1),
      ],
    ],
    [
      "CRLF",
      "code: |-\r\n  first\r\n  second\r\n",
      "first\nsecond",
      [new SourceRange(2, 3, 2, 8), new SourceRange(3, 3, 3, 9)],
    ],
    [
      "anchored and tagged header",
      "code: &shared # fake | indicator\n  !!str |-\n    first\n",
      "first",
      [new SourceRange(3, 5, 3, 10)],
    ],
    ["content at EOF", "code: |-\n  final", "final", [new SourceRange(2, 3, 2, 8)]],
    ["empty clip", "code: |\n", "", []],
    ["empty strip", "code: |-\n", "", []],
    ["empty keep", "code: |+\n", "", []],
    ["empty clip at EOF", "code: |", "", []],
    ["empty strip at EOF", "code: |-", "", []],
    ["empty keep at EOF", "code: |+", "", []],
    [
      "kept blank lines",
      "code: |+\n\n\n",
      "\n\n",
      [new SourceRange(2, 1, 2, 1), new SourceRange(3, 1, 3, 1)],
    ],
  ])("map a literal's constructed lines: %s", (_, text, value, ranges) => {
    expect(linesOf(text, value)).toEqual(ranges);
  });

  test.each([
    ["folded", "code: >-\n  first\n  second\n", "first second"],
    ["quoted", 'code: "first\\nsecond"\n', "first\nsecond"],
    ["plain", "code: first\n  second\n", "first second"],
  ])("are null for another style: %s", (_, text, value) => {
    expect(linesOf(text, value)).toBeNull();
  });
});

describe("merges", () => {
  afterEach(() => install(BUILTIN));

  const unboundedNodes = () => install({ ...BUILTIN, maxYamlNodes: Infinity });

  /** Return the value at `key` of the root mapping of `text`'s document. */
  function valueAt(text: string, key: string): Value {
    const value = SourceDocument.parse(text)._value as Record<string, Value>;
    return value[key]!;
  }

  const MERGE_SOURCE_TAG =
    "a tag on a `<<:` merge source checks nothing; tag the mapping that merges it";

  test("a merge list's first mapping wins and own keys win", () => {
    const text =
      "first: &first {shared: first, first_only: 1}\n" +
      "second: &second {shared: second, second_only: 2}\n" +
      "target:\n" +
      "  <<: [*first, *second]\n" +
      "  local: 3\n";

    expect(valueAt(text, "target")).toEqual({
      shared: "first",
      first_only: 1n,
      second_only: 2n,
      local: 3n,
    });
  });

  test("of two merge keys the later wins, and own keys win", () => {
    const text =
      "first: &first {shared: first, first_only: 1}\n" +
      "second: &second {shared: second, second_only: 2}\n" +
      "target:\n" +
      "  <<: *first\n" +
      "  <<: *second\n" +
      "  shared: local\n";

    expect(valueAt(text, "target")).toEqual({
      shared: "local",
      first_only: 1n,
      second_only: 2n,
    });
  });

  test("merged keys keep their text and an own key wins", () => {
    // Python's `!!binary` and `!!value` lines are refused in TS.
    const text =
      "base: &base\n" +
      "  true: boolean\n" +
      "  when: 2001-12-15T02:59:43.1Z\n" +
      "target:\n" +
      "  <<: *base\n" +
      "  1: integer-wins\n";
    const document = SourceDocument.parse(text);

    expect((document._value as Record<string, Value>)["target"]).toEqual({
      true: "boolean",
      when: "2001-12-15T02:59:43.1Z",
      1: "integer-wins",
    });
    expect(document.nonTextKey).toEqual({
      key: true,
      text: "true",
      range: new SourceRange(2, 3, 2, 7),
    });
  });

  test("an inline anchored merge source with a core tag YAML 1.2 lacks is refused", () => {
    const text =
      "target:\n" +
      "  <<: &inline\n" +
      "    values: !!set {one: null, two: null}\n" +
      "    ordered: !!omap [{first: 1}, {second: 2}]\n";

    expect(refusal(text).message).toBe(
      "could not determine a constructor for the tag 'tag:yaml.org,2002:set'",
    );
  });

  test("an inline merge source keeps a recursive alias's identity", () => {
    const target = valueAt("target:\n  <<: &base\n    self: *base\n", "target") as Record<
      string,
      Record<string, Value>
    >;

    expect(target["self"]!["self"]).toBe(target["self"]!);
    expect(target["self"]).not.toBe(target);
  });

  test("a mapping that merges itself keeps its own pairs", () => {
    const document = SourceDocument.parse("root: &root\n  <<: *root\n  value: local\n");
    const occurrences = document.root.child("root").mappingOccurrences();

    expect(document._value).toEqual({ root: { value: "local" } });
    expect(occurrences.map((cursor) => cursor.value)).toEqual(["local", "local"]);
  });

  test("a local tag on a merge source is an error over the source", () => {
    const error = refusal("target: {<<: !mystery {value: inherited}}\n");

    expect(error.message).toBe(MERGE_SOURCE_TAG);
    expect(error.range).toEqual(new SourceRange(1, 14, 1, 41));
  });

  test.each([
    ["inline", "use: {<<: !Foo {x: 1}}\n", new SourceRange(1, 11, 1, 22)],
    ["in list", "a: &a {x: 1}\nuse:\n  <<: [!A {y: 1}, *a]\n", new SourceRange(3, 8, 3, 17)],
    ["on list", "a: &a {x: 1}\nuse:\n  <<: !L [*a]\n", new SourceRange(3, 7, 3, 14)],
    ["anchor then alias", "m:\n  <<: &b !Foo {x: 1}\nn: *b\n", new SourceRange(2, 7, 2, 21)],
    ["anchor in list", "m:\n  <<: [&a !A {x: 1}]\nn: *a\n", new SourceRange(2, 8, 2, 20)],
  ])("a tag written at a merge site is an error: %s", (_, text, range) => {
    const error = refusal(text);

    expect(error.message).toBe(MERGE_SOURCE_TAG);
    expect(error.range).toEqual(range);
  });

  test("an unknown global tag on a merge source is an error", () => {
    const error = refusal("use: {<<: !<tag:example.com:x> {x: 1}}\n");

    expect(error.message).toBe(
      "unknown YAML tag 'tag:example.com:x'; a type tag is local, e.g. `!int`",
    );
    expect(error.range).toEqual(new SourceRange(1, 11, 1, 38));
  });

  test("a core tag on a merge source keeps its YAML meaning", () => {
    expect(valueAt("use: {<<: !!map {x: 1}}\n", "use")).toEqual({ x: 1n });
  });

  test.each([
    ["merge into tagged", "buy1: &b !Buy {conviction: 1}\nbuy2: !Buy {<<: *b, qty: 2}\n"],
    ["alias", "n: &b !Foo {x: 1}\nm:\n  <<: *b\n"],
    ["alias in list", "a: &a !A {x: 1}\nuse: {<<: [*a]}\n"],
    ["x-key anchor", "x-base: &b !Buy {c: 1}\nn: {<<: *b}\n"],
    ["aliased list", "a: &a {x: 1}\nl: &l !L [*a]\nuse: {<<: *l}\n"],
  ])("a tagged anchor merged through an alias parses: %s", (_, text) => {
    expect(() => SourceDocument.parse(text)).not.toThrow();
  });

  test.each([
    [
      "nested first",
      "root:\n  <<:\n    - {<<: 7}\n    - 8\n",
      "expected a mapping or list of mappings for merging, but found scalar",
      new SourceRange(3, 12, 3, 13),
    ],
    [
      "outer first",
      "root:\n  <<:\n    - 8\n    - {<<: 7}\n",
      "expected a mapping for merging, but found scalar",
      new SourceRange(3, 7, 3, 8),
    ],
    [
      "a list item before a source's scalar",
      "m: {<<: [{x: 2026-13-45}, 5]}\n",
      "expected a mapping for merging, but found scalar",
      new SourceRange(1, 27, 1, 28),
    ],
    [
      "a later merge key before a source's scalar",
      "m: {<<: {x: 2026-13-45}, <<: 5}\n",
      "expected a mapping or list of mappings for merging, but found scalar",
      new SourceRange(1, 30, 1, 31),
    ],
    [
      "a source's tag before a source's scalar",
      "m: {<<: [{x: 2026-13-45}, !T {y: 1}]}\n",
      MERGE_SOURCE_TAG,
      new SourceRange(1, 27, 1, 36),
    ],
    [
      "a later item's tag before an item's kind",
      "m: {<<: [5, !T {y: 1}]}\n",
      MERGE_SOURCE_TAG,
      new SourceRange(1, 13, 1, 22),
    ],
    [
      "a tag before the value's kind",
      "m: {<<: !T 5}\n",
      MERGE_SOURCE_TAG,
      new SourceRange(1, 9, 1, 13),
    ],
  ])("merge errors come first, in written order: %s", (_, text, message, range) => {
    const error = refusal(text);

    expect(error.message).toBe(message);
    expect(error.range).toEqual(range);
  });

  test.each([
    ["root: {<<}\n", new SourceRange(1, 10, 1, 11)],
    ["root: {a: 1, <<}\n", new SourceRange(1, 16, 1, 17)],
    ["root:\n  <<:\n  a: 1\n", new SourceRange(2, 6, 2, 7)],
  ])("a merge key with no value is refused where the key ends: %p", (text, range) => {
    const error = refusal(text);

    expect(error.message).toBe(
      "expected a mapping or list of mappings for merging, but found scalar",
    );
    expect(error.range).toEqual(range);
  });

  test("a merge key is not a non-text key", () => {
    const document = SourceDocument.parse("base: &b {a: x}\nc:\n  <<: *b\n  d: y\n");

    expect(document.nonTextKey).toBeNull();
  });

  test("a merged-in non-text key is reported where it is written", () => {
    expect(SourceDocument.parse("base: &b {1: x}\nc:\n  <<: *b\n").nonTextKey).toEqual({
      key: 1n,
      text: "1",
      range: new SourceRange(1, 11, 1, 12),
    });
  });

  test("a merged key is not a repeated key", () => {
    const document = SourceDocument.parse("base: &b {a: x}\nc:\n  <<: *b\n  a: y\n");

    expect(document.repeatedKey).toBeNull();
  });

  /** Return a short document with exponentially many merged pairs (Python's builder). */
  function binaryMergeDocument(levels: number): SourceDocument {
    const lines = ["level_0: &level_0", "  value: leaf"];
    for (let level = 1; level <= levels; level++) {
      const source = `*level_${level - 1}`;
      lines.push(`level_${level}: &level_${level}`, `  <<: [${source}, ${source}]`);
    }
    lines.push(`root: *level_${levels}`);
    return SourceDocument.parse(lines.join("\n") + "\n");
  }

  test("a binary merge chain keeps only distinct pairs and ropes", () => {
    unboundedNodes();
    const levels = 16;
    const document = binaryMergeDocument(levels);
    const mappings = document._definitions.filter((definition) => definition.kind === "mapping");
    const ropes = new Set<ArcSequence>();
    const pending = mappings.map((definition) => definition.pairSequence);
    let arcs = 0;
    let largest = 0;
    while (pending.length > 0) {
      const rope = pending.pop()!;
      if (ropes.has(rope)) continue;
      ropes.add(rope);
      arcs += rope.arcs.length;
      largest = Math.max(largest, rope.flatPairCount);
      for (const part of rope.parts) pending.push(part);
    }
    const effective = mappings.reduce((sum, definition) => sum + definition.effectiveIndex.size, 0);

    expect((document._value as Record<string, Record<string, Value>>)["root"]!["value"]).toBe(
      "leaf",
    );
    expect(arcs).toBeLessThan(levels * 8);
    expect(effective).toBeLessThan(levels * 8);
    expect(ropes.size).toBeLessThan(levels * 8);
    expect(largest).toBeGreaterThanOrEqual(2 ** levels);
  });

  test("a binary merge chain of 120 levels parses", () => {
    unboundedNodes();
    const value = binaryMergeDocument(120)._value as Record<string, Record<string, Value>>;

    expect(value["root"]!["value"]).toBe("leaf");
  });
});
