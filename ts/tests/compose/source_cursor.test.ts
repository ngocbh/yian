/**
 * Immutable, occurrence-aware YAML source cursors: a port of
 * `tests/compose/test_source_cursor.py`.
 *
 * Dropped: the tuple, set, frozenset, date and timezone snapshot cases and the container
 * subclasses other than arrays (a YAML value here is text, a number, a boolean, null, a plain
 * array or a plain record), the `_value_view` set operations, and the hash-call-count test
 * (ropes are cached by identity, with no hashing). The `scalar_style` tests that need Compose
 * come with Compose; the typed accessors, range helpers, scalar facts and projections come with
 * those methods. Mapping keys are text, so the constructed-key cases read keys as written.
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  type ArcSequence,
  freezeValue,
  type MapStep,
  type OccurrencePath,
  setRecordKey,
  snapshotValue,
  SourceAccessError,
} from "../../src/compose/source/common.ts";
import { SourceCursor } from "../../src/compose/source/cursor.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";

/** Lift the YAML node limit for one test; `afterEach` restores it. */
const unboundedNodes = () => install({ ...BUILTIN, maxYamlNodes: Infinity });

afterEach(() => install(BUILTIN));

/** Return the last step of `cursor`'s path as a mapping step. */
function lastStep(cursor: SourceCursor): MapStep {
  return cursor.path.at(-1) as MapStep;
}

/** Return the `SourceAccessError` `action` throws, failing if it throws none. */
function accessError(action: () => unknown): SourceAccessError {
  try {
    action();
  } catch (error) {
    expect(error).toBeInstanceOf(SourceAccessError);
    return error as SourceAccessError;
  }
  throw new Error("expected a SourceAccessError");
}

/** Return a document whose root aliases a binary DAG of `levels` lists (Python's builder). */
function binaryAliasDocument(levels = 24): SourceDocument {
  const lines = ["level_0: &level_0 [leaf]"];
  for (let level = 1; level <= levels; level++) {
    const source = `*level_${level - 1}`;
    lines.push(`level_${level}: &level_${level} [${source}, ${source}]`);
  }
  lines.push(`root: *level_${levels}`);
  return SourceDocument.parse(lines.join("\n") + "\n");
}

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

/** Return every distinct rope reachable from the document's mapping definitions. */
function ropesOf(document: SourceDocument): Set<ArcSequence> {
  const ropes = new Set<ArcSequence>();
  const pending: ArcSequence[] = [];
  for (const definition of document._definitions) {
    if (definition.kind === "mapping") pending.push(definition.pairSequence);
  }
  while (pending.length > 0) {
    const rope = pending.pop()!;
    if (ropes.has(rope)) continue;
    ropes.add(rope);
    for (const part of rope.parts) pending.push(part);
  }
  return ropes;
}

describe("value snapshots", () => {
  test("a copy keeps shared and recursive aliases and is not the original", () => {
    const shared: ValueRecord = { value: 1n };
    const root: ValueRecord = {};
    root["self"] = root;
    const original: ValueRecord = { shared, left: shared, right: shared, root };

    const copied = snapshotValue(original) as ValueRecord;

    expect(copied).not.toBe(original);
    expect(copied["shared"]).not.toBe(shared);
    expect(copied["shared"]).toBe(copied["left"]!);
    expect(copied["left"]).toBe(copied["right"]!);
    const copiedRoot = copied["root"] as ValueRecord;
    expect(copiedRoot).not.toBe(root);
    expect(copiedRoot["self"]).toBe(copiedRoot);
    expect(copied["shared"]).toEqual({ value: 1n });
  });

  test("deep lists, records and alias DAGs copy without stack overflow", () => {
    // Deep enough to overflow a recursive copy; Bun's stack survives 20,000 levels.
    const depth = 100_000;
    const levels = 2_000;
    let list: Value = "leaf";
    let record: Value = "leaf";
    let dag: Value = ["leaf"];
    for (let level = 0; level < depth; level++) {
      list = [list];
      record = { next: record };
    }
    for (let level = 0; level < levels; level++) dag = [dag, dag];

    let copiedList = snapshotValue(list);
    let copiedRecord = snapshotValue(record);
    let copiedDag = snapshotValue(dag);

    expect(copiedList).not.toBe(list);
    expect(copiedRecord).not.toBe(record);
    expect(copiedDag).not.toBe(dag);
    for (let level = 0; level < depth; level++) {
      copiedList = (copiedList as Value[])[0]!;
      copiedRecord = (copiedRecord as ValueRecord)["next"]!;
    }
    for (let level = 0; level < levels; level++) {
      const pair = copiedDag as Value[];
      expect(pair[0]).toBe(pair[1]!);
      copiedDag = pair[0]!;
    }
    expect(copiedList).toBe("leaf");
    expect(copiedRecord).toBe("leaf");
    expect(copiedDag).toEqual(["leaf"]);
  });

  test("anything but a YAML value throws TypeError", () => {
    class Custom {}
    class ListSubclass extends Array<Value> {}
    let deep: unknown = new Map();
    for (let level = 0; level < 100_000; level++) deep = [deep];
    const invalid: unknown[] = [
      new Custom(),
      new Map(),
      new Set(),
      new Date(),
      undefined,
      () => null,
      { nested: [new Map()] },
      new ListSubclass(),
      [1, , 3],
      deep,
    ];

    for (const value of invalid) {
      expect(() => snapshotValue(value)).toThrow(TypeError);
      expect(() => snapshotValue(value)).toThrow(/source cursors accept only YAML values/);
    }
    expect(() => snapshotValue(new Date())).toThrow(/got Date$/);
    expect(() => snapshotValue(undefined)).toThrow(/got undefined$/);
  });

  test("a __proto__ key stays data", () => {
    const record: ValueRecord = {};
    setRecordKey(record, "__proto__", 1n);

    const copied = snapshotValue(record) as ValueRecord;

    expect(Object.getPrototypeOf(copied)).toBe(Object.prototype);
    expect(Object.keys(copied)).toEqual(["__proto__"]);
    expect(Object.getOwnPropertyDescriptor(copied, "__proto__")?.value).toBe(1n);
  });

  test("freezeValue freezes every container of a recursive value", () => {
    const items: Value[] = [1n];
    const record: ValueRecord = { items, name: "x" };
    items.push(record);
    const value: ValueRecord = { record, list: [items] };

    expect(freezeValue(value)).toBe(value);
    for (const container of [value, record, items, value["list"]!]) {
      expect(Object.isFrozen(container)).toBe(true);
    }
    expect(() => {
      record["name"] = "y";
    }).toThrow(TypeError);
  });
});

describe("values", () => {
  test("keys read as written and ranges locate them", () => {
    // Python's `!!binary` line is refused in TS, so the lines below it shift up by one.
    const text =
      "values:\n" +
      "  0x10: hexadecimal\n" +
      "  1.50: floating\n" +
      "  2001-12-15T02:59:43.1Z: timestamp\n" +
      "collision:\n" +
      "  true: first\n" +
      "  1:\n" +
      "    nested: second\n";
    const root = SourceDocument.parse(text).root;
    const values = root.child("values");

    expect(values.get("0x10")).toBe("hexadecimal");
    expect(values.get("1.50")).toBe("floating");
    expect(values.get("2001-12-15T02:59:43.1Z")).toBe("timestamp");
    expect(values.child("0x10").range).toEqual(new SourceRange(2, 9, 2, 20));

    // Text keys: `true:` and `1:` no longer collide.
    const collision = root.child("collision");
    expect(collision.child("1").range).toEqual(new SourceRange(8, 5, 9, 1));
    expect(collision.child("1").get("nested")).toBe("second");
    expect(collision.child("true").value).toBe("first");

    const copied = values.value as ValueRecord;
    copied["0x10"] = "mutated";
    expect(values.get("0x10")).toBe("hexadecimal");
  });

  test("the root value is a defensive copy", () => {
    const document = SourceDocument.parse("root:\n  items:\n    - name: first\n");

    const first = document.root.value as Record<string, Record<string, ValueRecord[]>>;
    (first as ValueRecord)["added"] = true;
    first["root"]!["items"]![0]!["name"] = "changed";

    expect(document.root.value).toEqual({ root: { items: [{ name: "first" }] } });
  });

  test("a copy keeps alias identity and recursive aliases", () => {
    const document = SourceDocument.parse(
      "shared: &shared\n  value: 1\nleft: *shared\nright: *shared\nroot: &root\n  self: *root\n",
    );

    const copied = document.root.value as Record<string, ValueRecord>;

    expect(copied["shared"]).toBe(copied["left"]!);
    expect(copied["left"]).toBe(copied["right"]!);
    expect(copied["root"]!["self"]).toBe(copied["root"]!);
  });

  test("holds compares exactly and ends on a recursive alias", () => {
    const cursor = SourceDocument.parse("p: &a [1, *a]\nq: [1, true]\n").root;
    const recursive = cursor.child("p");

    expect(recursive.holds(recursive.value)).toBe(true);
    expect(recursive.holds([1n, [1n]])).toBe(false);
    expect(cursor.child("q").holds([1n, true])).toBe(true);
    expect(cursor.child("q").holds([1n, 1n])).toBe(false);
    expect(cursor.child("q").holds([1, true])).toBe(false);
  });

  test("holds compares records by keys in any order and never equals a list", () => {
    const cursor = SourceCursor.virtual({ a: 1n, b: [true] });

    expect(cursor.holds({ b: [true], a: 1n })).toBe(true);
    expect(cursor.holds({ a: 1n })).toBe(false);
    expect(cursor.holds({ a: 1n, c: [true] })).toBe(false);
    expect(cursor.holds({ a: 1n, b: [true], c: null })).toBe(false);
    expect(SourceCursor.virtual([]).holds({})).toBe(false);
    expect(SourceCursor.virtual({}).holds([])).toBe(false);
    expect(SourceCursor.virtual(1n).holds(true)).toBe(false);
  });

  test("a deep alias DAG copies without stack overflow", () => {
    unboundedNodes();
    const levels = 2_000;
    let copied = binaryAliasDocument(levels).root.child("root").value;

    for (let level = 0; level < levels; level++) {
      const pair = copied as Value[];
      expect(pair[0]).toBe(pair[1]!);
      copied = pair[0]!;
    }
    expect(copied).toEqual(["leaf"]);
  });

  test("the value view is frozen and keeps aliases", () => {
    const document = SourceDocument.parse(
      "shared: &shared\n  values: [1, 2]\nleft: *shared\nright: *shared\n" +
        "root: &root\n  self: *root\n",
    );

    const view = document.root._valueView() as Record<string, ValueRecord>;

    expect(view["shared"]).toBe(view["left"]!);
    expect(view["left"]).toBe(view["right"]!);
    expect(view["root"]!["self"]).toBe(view["root"]!);
    expect(Object.isFrozen(view)).toBe(true);
    expect(Object.isFrozen(view["shared"]!["values"])).toBe(true);
    expect(() => {
      view["left"]!["new"] = 2n;
    }).toThrow(TypeError);
    expect(() => {
      (view["shared"]!["values"] as Value[]).push(3n);
    }).toThrow(TypeError);
  });

  test("a virtual value is snapshotted and copied on read", () => {
    const recursive: ValueRecord = { items: [1n] };
    recursive["self"] = recursive;
    const original: ValueRecord = { root: recursive };

    const cursor = SourceCursor.virtual(original);
    (recursive["items"] as Value[]).push(2n);
    const first = cursor.value as Record<string, ValueRecord>;
    (first["root"]!["items"] as Value[]).push(3n);
    const second = cursor.value as Record<string, ValueRecord>;

    expect(second["root"]!["items"]).toEqual([1n]);
    expect(second["root"]!["self"]).toBe(second["root"]!);
    expect(Object.isFrozen(cursor._valueView())).toBe(true);
    expect(cursor.document).toBeNull();
    expect(cursor.path).toEqual([]);
    expect(cursor.range).toBeNull();
    expect(cursor.keyRange).toBeNull();
  });

  test("a cursor is frozen", () => {
    const cursor = SourceDocument.parse("a: 1\n").root;

    expect(Object.isFrozen(cursor)).toBe(true);
    expect(() => {
      (cursor as unknown as { _definition: number })._definition = 1;
    }).toThrow(TypeError);
  });
});

describe("merges and aliases", () => {
  test("the binary alias graph keeps a linear definition DAG", () => {
    unboundedNodes();
    const document = binaryAliasDocument();

    expect(document._definitions.length).toBeLessThan(200);
  });

  test("a repeated cached merge source keeps the rightmost winner's ordinal", () => {
    const root = SourceDocument.parse(
      "a: &a {clash: A, a_only: true}\n" +
        "b: &b {clash: B, b_only: true}\n" +
        "root:\n" +
        "  <<: [*a, *b, *a]\n",
    ).root.child("root");

    const winner = root.child("clash");

    expect(root.value).toEqual({ clash: "A", a_only: true, b_only: true });
    expect(lastStep(winner).pairOrdinal).toBe(4);
  });

  test("repeated merge occurrences keep logical ordinals and replay", () => {
    const document = binaryMergeDocument(5);
    const target = document.root.child("root");

    const occurrences = target.mappingOccurrences("value");

    expect(occurrences.length).toBe(32);
    expect(occurrences.map((item) => lastStep(item).pairOrdinal)).toEqual(
      Array.from({ length: 32 }, (_, index) => index),
    );
    expect(new Set(occurrences.map((item) => item.range)).size).toBe(1);
    expect(occurrences[0]!.range).toEqual(new SourceRange(2, 10, 2, 14));
    for (const item of [occurrences[0]!, occurrences.at(-1)!]) {
      expect(document._resolveOccurrence(item.path).path).toEqual(item.path);
      expect(document._resolveOccurrence(item.path).value).toBe("leaf");
    }
  });

  test("a recursive sequence merge keeps order, occurrences and replay", () => {
    const document = SourceDocument.parse(
      "base: &base {base_only: B, clash: base}\n" +
        "root: &root\n" +
        "  <<: [*base, *root]\n" +
        "  root_only: R\n" +
        "  clash: root\n",
    );
    const root = document.root.child("root");
    const occurrences = root.mappingOccurrences();

    expect(Object.keys(root.value as ValueRecord)).toEqual(["root_only", "clash", "base_only"]);
    expect(occurrences.map((item) => lastStep(item).key)).toEqual([
      "root_only",
      "clash",
      "base_only",
      "clash",
      "root_only",
      "clash",
    ]);
    expect(occurrences.map((item) => lastStep(item).pairOrdinal)).toEqual([0, 1, 2, 3, 4, 5]);
    for (const item of occurrences) {
      const replayed = document._resolveOccurrence(item.path);
      expect(replayed.path).toEqual(item.path);
      expect(replayed.value).toEqual(item.value);
    }
  });

  test("a double recursive merge keeps three local occurrences", () => {
    // A merge source still being flattened contributes its own pairs once per merge key,
    // then the mapping's own: three, where Python's in-place flattening keeps four.
    const root = SourceDocument.parse(
      "root: &root\n  <<: *root\n  <<: *root\n  value: local\n",
    ).root.child("root");

    const occurrences = root.mappingOccurrences("value");

    expect(occurrences.length).toBe(3);
    expect(occurrences.map((item) => lastStep(item).pairOrdinal)).toEqual([0, 1, 2]);
  });

  test("a repeated self source in one merge list keeps three occurrences", () => {
    const root = SourceDocument.parse(
      "root: &root\n  <<: [*root, *root]\n  value: local\n",
    ).root.child("root");

    const occurrences = root.mappingOccurrences("value");

    expect(occurrences.length).toBe(3);
    expect(occurrences.map((item) => lastStep(item).pairOrdinal)).toEqual([0, 1, 2]);
  });

  test("many recursive merges keep metadata linear", () => {
    const width = 1_200;
    const lines = ["root: &root"];
    for (let index = 0; index < width; index++) lines.push("  <<: *root");
    for (let index = 0; index < width; index++) lines.push(`  key_${index}: ${index}`);

    const document = SourceDocument.parse(lines.join("\n") + "\n");
    const root = document.root.child("root");
    const ropes = ropesOf(document);
    let arcs = 0;
    for (const rope of ropes) arcs += rope.arcs.length;
    let effective = 0;
    for (const definition of document._definitions) {
      if (definition.kind === "mapping") effective += definition.effectiveIndex.size;
    }

    expect(root.get("key_0")).toBe(0n);
    expect(root.get(`key_${width - 1}`)).toBe(BigInt(width - 1));
    expect(root._mappingDefinition()!.pairSequence.flatPairCount).toBe(width * (width + 1));
    expect(arcs).toBeLessThan(width * 3);
    expect(effective).toBeLessThan(width * 3);
  });

  test("a nested mapping can merge its recursive ancestor", () => {
    const document = SourceDocument.parse(
      "root: &root\n  child: &child\n    <<: *root\n    leaf: child\n  own: root\n",
    );

    const copied = document.root.value as Record<string, Record<string, ValueRecord>>;
    const child = copied["root"]!["child"]!;

    expect(child["child"]).toBe(child);
    expect(child["own"]).toBe("root");
    expect(child["leaf"]).toBe("child");
  });

  test("a deep alias path is lazy", () => {
    unboundedNodes();
    let cursor = binaryAliasDocument().root.child("root");

    for (let level = 0; level < 24; level++) cursor = cursor.child(0);

    expect(cursor._pathLink!.step).toEqual({ index: 0 });
    expect(cursor.path.length).toBe(25);
  });
});

describe("navigation", () => {
  test("mapping child and children use effective order", () => {
    const root = SourceDocument.parse(
      "first: original\nsecond:\n  nested: true\nfirst: final\n",
    ).root;

    const first = root.child("first");
    const children = root.children();

    expect(first.value).toBe("final");
    expect(first.range).toEqual(new SourceRange(4, 8, 4, 13));
    expect(first.keyRange).toEqual(new SourceRange(4, 1, 4, 6));
    expect(children.map((child) => child.value)).toEqual(["final", { nested: true }]);
  });

  test("wide mappings and many merge sources look keys up by the effective index", () => {
    const width = 200;
    const wide = SourceDocument.parse(
      Array.from({ length: width }, (_, index) => `key_${index}: ${index}\n`).join(""),
    ).root;
    const lines = Array.from(
      { length: width },
      (_, index) => `source_${index}: &source_${index} {key_${index}: ${index}}`,
    );
    const sources = Array.from({ length: width }, (_, index) => `*source_${index}`).join(", ");
    lines.push("root:", `  <<: [${sources}]`);
    const merged = SourceDocument.parse(lines.join("\n") + "\n").root.child("root");
    const prototype = SourceCursor.prototype;
    const { _iterFlatOccurrences: flat, _iterUniqueOccurrences: unique } = prototype;
    const refuse = (): never => {
      throw new Error("a key lookup must not enumerate occurrences");
    };

    prototype._iterFlatOccurrences = refuse;
    prototype._iterUniqueOccurrences = refuse;
    try {
      expect(wide.children().map((child) => child.value)).toEqual(
        Array.from({ length: width }, (_, index) => BigInt(index)),
      );
      for (let index = 0; index < width; index++) {
        const key = `key_${index}`;
        expect(wide.has(key)).toBe(true);
        expect(wide.child(key).value).toBe(BigInt(index));
        expect(merged.has(key)).toBe(true);
        expect(merged.child(key).value).toBe(BigInt(index));
      }
    } finally {
      prototype._iterFlatOccurrences = flat;
      prototype._iterUniqueOccurrences = unique;
    }
    expect(merged._mappingDefinition()!.effectiveIndex.size).toBe(width);
  });

  test("sequence child and children follow indexes", () => {
    const sequence = SourceDocument.parse("items:\n  - first\n  - name: second\n").root.child(
      "items",
    );

    expect(sequence.child(0).value).toBe("first");
    expect(sequence.child(0).range).toEqual(new SourceRange(2, 5, 2, 10));
    expect(sequence.child(0).keyRange).toBeNull();
    expect(sequence.children().map((child) => child.value)).toEqual([
      "first",
      { name: "second" },
    ]);
    expect(accessError(() => sequence.child(-1)).message).toBe(
      "sequence index must be a non-negative integer",
    );
    expect(accessError(() => sequence.child(2)).message).toBe("sequence index 2 is out of range");
    expect(accessError(() => sequence.child("0")).message).toBe(
      "sequence index must be a non-negative integer",
    );
    expect(accessError(() => sequence.child(1.5)).message).toBe(
      "sequence index must be a non-negative integer",
    );
    expect(accessError(() => sequence.child(2)).range).toEqual(sequence.range);
  });

  test("a mapping lookup needs a text key it holds", () => {
    const root = SourceDocument.parse("name: first\n1: one\n").root;

    expect(accessError(() => root.child("missing")).message).toBe(
      "missing required field 'missing'",
    );
    expect(accessError(() => root.child(1)).message).toBe("missing required field 1");
    expect(accessError(() => root.child("toString")).message).toBe(
      "missing required field 'toString'",
    );
    expect(accessError(() => root.child("missing")).range).toEqual(root.range);
    expect(root.child("1").value).toBe("one");
    expect(accessError(() => root.child("name").child("x")).message).toBe(
      "cannot select a child from a scalar value",
    );
    expect(root.child("name").children()).toEqual([]);
    expect(root.child("name").mappingOccurrences()).toEqual([]);
    expect(root.child("name").has("x")).toBe(false);
  });

  test("has and get read effective keys and copy values or the fallback", () => {
    const root = SourceDocument.parse("name: compose\nconfig: {enabled: true}\n").root;
    const fallback: ValueRecord = { fallback: [] };

    expect(root.has("name")).toBe(true);
    expect(root.has("missing")).toBe(false);
    expect(root.has("toString")).toBe(false);
    expect(root.get("name")).toBe("compose");
    expect(root.get("missing")).toBeNull();
    const got = root.get("missing", fallback);
    expect(got).toEqual({ fallback: [] });
    expect(got).not.toBe(fallback);
    const config = root.get("config") as ValueRecord;
    expect(Object.isFrozen(config)).toBe(false);
    config["enabled"] = false;
    expect(root.get("config")).toEqual({ enabled: true });
  });

  test("a value-only cursor navigates its value", () => {
    const cursor = SourceCursor.virtual({ items: ["first", "second"], name: "x" });
    const items = cursor.child("items");

    expect(items.child(1).value).toBe("second");
    expect(items.child(1).document).toBeNull();
    expect(items.children().map((child) => child.value)).toEqual(["first", "second"]);
    expect(cursor.children().map((child) => child.value)).toEqual([["first", "second"], "x"]);
    expect(cursor.mappingOccurrences("name").map((child) => child.value)).toEqual(["x"]);
    expect(cursor.mappingOccurrences().length).toBe(2);
    expect(items.mappingOccurrences()).toEqual([]);
    expect(cursor.has("name")).toBe(true);
    expect(cursor.has("toString")).toBe(false);
    expect(items.has("0")).toBe(false);
    expect(cursor.get("name")).toBe("x");
    expect(cursor.get("missing", "fallback")).toBe("fallback");
    expect(accessError(() => cursor.child("missing")).message).toBe(
      "missing required field 'missing'",
    );
    expect(accessError(() => cursor.child(0)).message).toBe("missing required field 0");
    expect(accessError(() => items.child("0")).message).toBe(
      "sequence index must be a non-negative integer",
    );
    expect(accessError(() => items.child(2)).message).toBe("sequence index 2 is out of range");
    expect(accessError(() => items.child(0).child(0)).message).toBe(
      "cannot select a child from a scalar value",
    );
    expect(accessError(() => items.child(0).child(0)).range).toBeNull();
    expect(SourceCursor.virtual("x").children()).toEqual([]);
  });

  test("a duplicate lookup is last while occurrences keep both", () => {
    const root = SourceDocument.parse("name: first\nother: middle\nname: final\n").root;

    const occurrences = root.mappingOccurrences("name");

    expect(root.child("name").value).toBe("final");
    expect(occurrences.map((cursor) => cursor.value)).toEqual(["first", "final"]);
    expect(occurrences.map((cursor) => cursor.range!.startLine)).toEqual([1, 3]);
    expect(occurrences.map((cursor) => lastStep(cursor).pairOrdinal)).toEqual([0, 2]);
    expect(root.mappingOccurrences().map((cursor) => cursor.value)).toEqual([
      "first",
      "middle",
      "final",
    ]);
  });

  test("a merge-local value wins but the inherited occurrence remains", () => {
    const root = SourceDocument.parse(
      "base: &base\n  value: inherited\ntarget:\n  <<: *base\n  value: local\n",
    ).root.child("target");

    const occurrences = root.mappingOccurrences("value");

    expect(root.child("value").value).toBe("local");
    expect(occurrences.map((cursor) => cursor.value)).toEqual(["inherited", "local"]);
    expect(occurrences.map((cursor) => cursor.range!.startLine)).toEqual([2, 5]);
  });

  test("the `~` key filter is distinct from no filter", () => {
    const root = SourceDocument.parse("~: null-key\nnamed: named-value\n").root;

    expect(root.mappingOccurrences("~").map((cursor) => cursor.value)).toEqual(["null-key"]);
    expect(root.mappingOccurrences().map((cursor) => cursor.value)).toEqual([
      "null-key",
      "named-value",
    ]);
    expect(root.mappingOccurrences("missing")).toEqual([]);
  });

  test("occurrence steps hold keys as written", () => {
    const root = SourceDocument.parse(
      "bool_map: {on: bool-value}\n" +
        "int_map: {1: int-value}\n" +
        'str_map: {"1": str-value}\n' +
        "null_map: {~: null-value}\n",
    ).root;

    const steps = ["bool_map", "int_map", "str_map", "null_map"].map(
      (name) => root.child(name).mappingOccurrences()[0]!.path.at(-1),
    );

    expect(steps).toEqual([
      { key: "on", pairOrdinal: 0 },
      { key: "1", pairOrdinal: 0 },
      { key: "1", pairOrdinal: 0 },
      { key: "~", pairOrdinal: 0 },
    ]);
  });

  test("alias occurrences have distinct paths and the anchor's range", () => {
    const document = SourceDocument.parse(
      "shared: &shared {value: 1}\nleft: *shared\nright: *shared\n",
    );
    const saved = (() => {
      const left = document.root.child("left");
      const right = document.root.child("right");
      expect(left.path).not.toEqual(right.path);
      expect(left.range).toEqual(new SourceRange(1, 9, 1, 27));
      expect(right.range).toEqual(left.range);
      expect(left.keyRange).toEqual(new SourceRange(2, 1, 2, 5));
      expect(right.keyRange).toEqual(new SourceRange(3, 1, 3, 6));
      return { path: left.path, range: left.range };
    })();

    const replayed = document._resolveOccurrence(saved.path);

    expect(replayed.path).toEqual(saved.path);
    expect(replayed.range).toEqual(saved.range);
  });

  test("cursors are distinct objects", () => {
    const document = SourceDocument.parse(
      "shared: &shared {value: 1}\nleft: *shared\nright: *shared\n",
    );
    const left = document.root.child("left");
    const right = document.root.child("right");
    const virtual = SourceCursor.virtual(left.value);

    expect(left).not.toBe(right);
    expect(left).not.toBe(virtual);
    expect(new Set([left, right, virtual]).size).toBe(3);
    expect(
      new Map<SourceCursor, string>([
        [left, "left"],
        [right, "right"],
        [virtual, "virtual"],
      ]).size,
    ).toBe(3);
  });

  test("a recursive alias path stops at its ancestor", () => {
    const root = SourceDocument.parse(
      "root: &root\n  name: recursive\n  self: *root\n",
    ).root.child("root");

    const recursive = root.child("self");
    const copied = recursive.value as ValueRecord;

    expect(copied["self"]).toBe(copied);
    expect(lastStep(recursive).key).toBe("self");
    expect(recursive.range).toEqual(root.range);
    expect(recursive.keyRange).toEqual(new SourceRange(3, 3, 3, 7));
    const error = accessError(() => recursive.child("name"));
    expect(error.message).toBe("cannot descend through a recursive alias occurrence");
    expect(error.range).toBe(recursive.range);
  });

  test("an occurrence path that does not fit the document is refused", () => {
    const document = SourceDocument.parse("a: [x]\nb: y\n");
    const refused = (path: OccurrencePath): string =>
      accessError(() => document._resolveOccurrence(path)).message;

    expect(document._resolveOccurrence([{ key: "a", pairOrdinal: 0 }, { index: 0 }]).value).toBe(
      "x",
    );
    expect(refused([{ key: "a", pairOrdinal: 2 }])).toBe("invalid mapping occurrence path");
    expect(refused([{ key: "b", pairOrdinal: 0 }])).toBe(
      "mapping occurrence path key does not match",
    );
    expect(refused([{ key: "a", pairOrdinal: 0 }, { index: 1 }])).toBe(
      "invalid sequence occurrence path",
    );
    expect(refused([{ index: 0 }])).toBe("invalid sequence occurrence path");
    expect(refused([{ key: "a", pairOrdinal: 0 }, { key: "x", pairOrdinal: 0 }])).toBe(
      "invalid mapping occurrence path",
    );
  });
});
