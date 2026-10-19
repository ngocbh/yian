/**
 * Local YAML tags kept by the source layer: a port of `tests/compose/test_source_tags.py`.
 *
 * The merge-source, bare-tag, unknown-tag and tag-hint errors live in `source_document.test.ts`.
 * `test_space_ends_the_tag`: a tag written with `,` is now a YAML syntax error with the tag hint
 * (YAML 1.2 tag syntax; `source_document.test.ts`, "the tag hint").
 * `test_bare_tag_check_does_not_shrink_nesting_depth` and
 * `test_parse_leaves_the_composed_node_tags_unchanged` have no counterpart: the TS build walks
 * with an explicit stack and keeps no composed tree.
 *
 * Later: `test_compact_flow_generated_binding_has_no_tag`, which needs `flowNodesCursor`.
 * Later (T3-22): `test_single_node_def_generated_binding_has_no_tag`.
 */

import { describe, expect, test } from "bun:test";

import { normalizeNodeSignatureCursor } from "../../src/compose/parse/nodes.ts";
import { type MapStep, type TaggedNode } from "../../src/compose/source/common.ts";
import { SourceCursor } from "../../src/compose/source/cursor.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { SourceRange } from "../../src/diagnostics.ts";

/** Return the cursor at `path` below the root of `text`'s document. */
function cur(text: string, ...path: (string | number)[]): SourceCursor {
  let cursor = SourceDocument.parse(text).root;
  for (const step of path) cursor = cursor.child(step);
  return cursor;
}

/** Return the first mapping step of `node`'s path. */
function firstKey(node: TaggedNode): string {
  return (node.path[0] as MapStep).key;
}

describe("tagged values", () => {
  test("a tagged scalar's value follows the flow YAML rules", () => {
    const c = cur("a: !int 5\n", "a");

    expect(c.value).toBe(5n);
    expect(c.tag).toBe("int");
    expect(c.hasValue).toBe(true);
  });

  test("tagged plain scalars resolve like untagged ones", () => {
    const tagged = cur(
      "a: !bool yes\nb: !float 1e3\nc: !str 1.5\nd: !int 0x10\ne: !x ~\nf: !bool true\n",
    );
    const untagged = cur("a: yes\nb: 1e3\nc: 1.5\nd: 0x10\ne: ~\nf: true\n");

    expect(tagged.value).toEqual(untagged.value);
    expect(tagged.value).toEqual({ a: "yes", b: 1000, c: 1.5, d: 16n, e: null, f: true });
  });

  test("a tagged empty scalar has no value", () => {
    const c = cur("a: !int\n", "a");

    expect(c.tag).toBe("int");
    expect(c.value).toBeNull();
    expect(c.hasValue).toBe(false);
  });

  test("a tagged null has a value", () => {
    const c = cur("a: !Optional(str) null\n", "a");

    expect(c.value).toBeNull();
    expect(c.hasValue).toBe(true);
  });

  test("a tagged empty quoted scalar has a value", () => {
    const c = cur("a: !str ''\n", "a");

    expect(c.value).toBe("");
    expect(c.tag).toBe("str");
    expect(c.hasValue).toBe(true);
  });

  test("a quoted tagged scalar stays text", () => {
    expect(cur("a: !int '5'\n", "a").value).toBe("5");
    expect(cur('a: !int "5"\n', "a").value).toBe("5");
  });

  test("a tagged block scalar stays text", () => {
    const c = cur("a: !str |\n  5\n", "a");

    expect(c.value).toBe("5\n");
    expect(c.tag).toBe("str");
  });

  test("a tagged mapping and sequence", () => {
    const m = cur("d: !Buy\n  conviction: 1\n", "d");
    const s = cur("h: !list(message) []\n", "h");

    expect(m.tag).toBe("Buy");
    expect(m.value).toEqual({ conviction: 1n });
    expect(s.tag).toBe("list(message)");
    expect(s.value).toEqual([]);
  });

  test("a template scalar with a tag", () => {
    expect(cur("q: !int ${size.output.qty}\n", "q").value).toBe("${size.output.qty}");
  });

  test("untagged and core tags report null", () => {
    expect(cur("a: 5\n", "a").tag).toBeNull();
    expect(cur("a: 5\n", "a").hasValue).toBe(true);
    const c = cur("a: !!str 5\n", "a");
    expect(c.tag).toBeNull();
    expect(c.value).toBe("5");
    expect(cur("a:\n", "a").hasValue).toBe(true);
  });

  test("untagged containers report null", () => {
    const root = cur("m: {x: 1}\ns: [1]\n");

    expect(root.tag).toBeNull();
    expect(root.child("m").tag).toBeNull();
    expect(root.child("s").tag).toBeNull();
  });

  test("a verbatim local tag is a local tag", () => {
    const c = cur("a: !<!int> 5\n", "a");

    expect(c.tag).toBe("int");
    expect(c.value).toBe(5n);
  });

  test("a tagged key constructs like an untagged one", () => {
    const document = SourceDocument.parse("!int 1: one\n!x <<: two\n");

    expect(document.root.value).toEqual({ "1": "one", "<<": "two" });
    expect(document.nonTextKey?.key).toBe(1n);
  });

  test("tags ending at a flow indicator are two tagged empty values", () => {
    const m = cur("m: {a: !int, b: !str}\n", "m");

    expect(m.child("a").tag).toBe("int");
    expect(m.child("a").hasValue).toBe(false);
    expect(m.child("b").tag).toBe("str");
    expect(m.child("b").hasValue).toBe(false);
  });
});

const NODE_FLOW = `nodes:
  n:
    kind: code
    input:
      d: !Buy
        conviction: 1
      h: !list(str) []
`;

describe("the node signature projection", () => {
  test.each([
    ["d", "Buy"],
    ["h", "list(str)"],
  ])("tagged node inputs keep tags through signature projection: %s", (name, tag) => {
    const nodes = cur(NODE_FLOW, "nodes");
    const entry = normalizeNodeSignatureCursor(nodes.child("n")).child("inputs").child(name);

    expect(entry.tag).toBe(tag);
    expect(entry.withValue(structuredClone(entry.value)).tag).toBe(tag);
    expect(entry.withValue({ other: 2n }).tag).toBeNull();
  });
});

describe("tags through anchors, aliases and merges", () => {
  test("a tag survives an anchor, an alias and a merge", () => {
    expect(cur("base: &b {x: !int 1}\nuse:\n  <<: *b\n", "use", "x").tag).toBe("int");
  });

  // The tag-on-a-merge-source errors are in source_document.test.ts ("a tag written at a
  // merge site is an error"); these texts parse, and each listed node replays to its tag.
  test.each([
    ["merge into tagged", "buy1: &b !Buy {conviction: 1}\nbuy2: !Buy {<<: *b, qty: 2}\n"],
    ["alias", "n: &b !Foo {x: 1}\nm:\n  <<: *b\n"],
    ["alias in list", "a: &a !A {x: 1}\nuse: {<<: [*a]}\n"],
    ["x-key anchor", "x-base: &b !Buy {c: 1}\nn: {<<: *b}\n"],
    ["aliased list", "a: &a {x: 1}\nl: &l !L [*a]\nuse: {<<: *l}\n"],
  ])("a tagged anchor merged through an alias keeps its tag: %s", (_, text) => {
    const document = SourceDocument.parse(text);
    const nodes = document.taggedNodes();

    for (const node of nodes) expect(document._resolveOccurrence(node.path).tag).toBe(node.tag);
    expect(nodes.length).toBeGreaterThan(0);
  });

  test("a merge into a tagged mapping reads both tags", () => {
    const root = cur("buy1: &b !Buy {conviction: 1}\nbuy2: !Buy {<<: *b, qty: 2}\n");

    expect(root.child("buy2").value).toEqual({ conviction: 1n, qty: 2n });
    expect(root.child("buy1").tag).toBe("Buy");
    expect(root.child("buy2").tag).toBe("Buy");
  });

  test("a core tag on a merge source keeps its YAML meaning", () => {
    expect(cur("use: {<<: !!map {x: 1}}\n", "use").value).toEqual({ x: 1n });
  });

  test("a tag on an anchor is shared by its aliases", () => {
    const root = cur("a: &v !int 1\nb: *v\n");

    expect(root.child("a").tag).toBe("int");
    expect(root.child("b").tag).toBe("int");
  });
});

describe("value gating", () => {
  test("a tag is value-gated through projections", () => {
    const c = cur("a: !int 5\n", "a");

    expect(c.withValue(5n).tag).toBe("int");
    expect(c.withValue("${input.a}").tag).toBeNull();
    expect(c.withValue("${input.a}").hasValue).toBe(true);
  });

  test("a tag is value-gated by type", () => {
    const c = cur("a: !int 1\n", "a");
    const m = cur("m: !Buy {x: 1}\n", "m");

    expect(c.withValue(true).tag).toBeNull();
    expect(m.withValue({ x: true }).tag).toBeNull();
    expect(m.withValue({ x: 1n }).tag).toBe("Buy");
  });

  test("a changed empty tagged scalar has a value", () => {
    const c = cur("a: !int\n", "a");

    expect(c.withValue(null).tag).toBe("int");
    expect(c.withValue(null).hasValue).toBe(false);
    expect(c.withValue(3n).tag).toBeNull();
    expect(c.withValue(3n).hasValue).toBe(true);
  });

  test("a virtual cursor has no tag and has a value", () => {
    const v = SourceCursor.virtual(null);

    expect(v.tag).toBeNull();
    expect(v.hasValue).toBe(true);
  });
});

describe("tagged nodes", () => {
  test("a document lists its tagged nodes", () => {
    const d = SourceDocument.parse("a: !int 1\nb: [!str x]\n");

    expect(d.taggedNodes().map((t) => t.tag).sort()).toEqual(["int", "str"]);
  });

  test("tagged nodes carry a range and a path, in document order", () => {
    const d = SourceDocument.parse("b: [!str x]\na: !int 1\n");
    const nodes = d.taggedNodes();

    expect(nodes.map((t) => t.tag)).toEqual(["str", "int"]);
    expect(nodes[0]!.range).toEqual(new SourceRange(1, 5, 1, 11));
    expect(d._resolveOccurrence(nodes[0]!.path).tag).toBe("str");
    expect(d._resolveOccurrence(nodes[1]!.path).value).toBe(1n);
  });

  test("an aliased node is listed once, at its anchor", () => {
    const d = SourceDocument.parse("base: &b {x: !int 1}\nuse:\n  <<: *b\n  y: *b\n");
    const nodes = d.taggedNodes();

    expect(nodes.length).toBe(1);
    const only = nodes[0]!;
    expect(only.tag).toBe("int");
    expect(d._resolveOccurrence(only.path).path).toEqual(only.path);
    expect(firstKey(only)).toBe("base");
  });

  test("an untagged document lists no tagged nodes", () => {
    expect(SourceDocument.parse("a: 1\nb: [x]\n").taggedNodes()).toEqual([]);
    expect(SourceDocument.parse("").taggedNodes()).toEqual([]);
  });

  test("a shared merge rope is walked once", () => {
    const keys = 200;
    const users = 200;
    let base = "";
    for (let i = 0; i < keys; i++) base += `  k${i}: !int ${i}\n`;
    let uses = "";
    for (let j = 0; j < users; j++) uses += `u${j}:\n  <<: *b\n`;
    const document = SourceDocument.parse(`b: &b\n${base}${uses}`);
    const prototype = SourceCursor.prototype;
    const descend = prototype._descendMappingOccurrence;
    let calls = 0;
    prototype._descendMappingOccurrence = function (this: SourceCursor, occurrence) {
      calls += 1;
      return descend.call(this, occurrence);
    };
    try {
      expect(document.taggedNodes().length).toBe(keys);
    } finally {
      prototype._descendMappingOccurrence = descend;
    }
    // One step per authored pair: the shared rope is walked once, not once per user.
    expect(calls).toBeLessThanOrEqual(keys + 2 * users + 1);
  });

  test("a cursor's tag definition id matches its tagged node", () => {
    const d = SourceDocument.parse("a: !int 1\nb: [!str x, &s !int 2, *s]\nc: 3\n");
    const ids = d.taggedNodes().map((node) => node.definitionId);
    const b = d.root.child("b");
    const cursors = [d.root.child("a"), b.child(0), b.child(1)];

    expect(cursors.map((c) => c.tagDefinitionId)).toEqual(ids);
    expect(b.child(2).tagDefinitionId).toBe(b.child(1).tagDefinitionId);
    expect(d.root.child("c").tagDefinitionId).toBeNull();
    expect(d.root.child("a").withValue(2n).tagDefinitionId).toBeNull();
    expect(SourceCursor.virtual(1n).tagDefinitionId).toBeNull();
  });

  test("scalarText is the authored text of an unchanged scalar", () => {
    const root = SourceDocument.parse("a: !str yes\nb: 30\nc: 'q'\nd: [1]\ne: 2026-01-01\n").root;

    expect(root.child("a").scalarText).toBe("yes");
    expect(root.child("b").scalarText).toBe("30");
    expect(root.child("c").scalarText).toBe("q");
    expect(root.child("e").scalarText).toBe("2026-01-01");
    expect(root.child("d").scalarText).toBeNull();
    expect(root.child("b").withValue(31n).scalarText).toBeNull();
    expect(SourceCursor.virtual(30n).scalarText).toBeNull();
  });

  test("keyText is the authored key", () => {
    const x = SourceDocument.parse("x: {true: 1, 'q': 2, 3: 4}\n").root.child("x");

    expect(x.child("true").keyText).toBe("true");
    expect(x.child("q").keyText).toBe("q");
    expect(x.child("3").keyText).toBe("3");
    expect(x.keyText).toBe("x");
    expect(SourceDocument.parse("- 1\n").root.child(0).keyText).toBeNull();
    expect(SourceCursor.virtual({ a: 1n }).child("a").keyText).toBeNull();
  });

  test("picked top-level keys are skipped", () => {
    const d = SourceDocument.parse("x-a: &k !str hidden\nx-b: !int 1\nkind: *k\nn: {m: !int 2}\n");
    const nodes = d.taggedNodes({ skipTopLevel: (key) => key.startsWith("x-") });

    expect(nodes.map((node) => node.tag)).toEqual(["str", "int"]);
    expect(firstKey(nodes[0]!)).toBe("kind");
    expect(d.taggedNodes().length).toBe(3);
  });

  test("a tagged mapping key is marked", () => {
    const d = SourceDocument.parse("a:\n  !int q: 1\n  r: !int 2\n");
    const nodes = d.taggedNodes();

    expect(nodes.length).toBe(2);
    const [key, value] = nodes as [TaggedNode, TaggedNode];
    expect(key.isKey).toBe(true);
    expect((key.path.at(-1) as MapStep).key).toBe("q");
    expect(value.isKey).toBe(false);
  });

  test("key paths are spelled as written", () => {
    const d = SourceDocument.parse("n:\n  !str on: 1\n  l: [0, {null: !int 3}]\n");
    const nodes = d.taggedNodes();

    expect(nodes.length).toBe(2);
    const [key, value] = nodes as [TaggedNode, TaggedNode];
    expect(key.keyPath).toEqual(["n", "on"]);
    expect(value.keyPath).toEqual(["n", "l", 1, "null"]);
  });
});
