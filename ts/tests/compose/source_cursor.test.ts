/**
 * Immutable, occurrence-aware YAML source cursors: a port of
 * `tests/compose/test_source_cursor.py`.
 *
 * Dropped: the tuple, set, frozenset, date and timezone snapshot cases and the container
 * subclasses other than arrays (a YAML value here is text, a number, a boolean, null, a plain
 * array or a plain record), the `_value_view` set operations, and the hash-call-count test
 * (ropes are cached by identity, with no hashing). The `scalar_style` tests that need Compose
 * come with Compose. Mapping keys are text, so the constructed-key cases read keys as written,
 * and the wide projection refuses occurrence walks rather than counting key comparisons.
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  type ArcSequence,
  freezeValue,
  type MapStep,
  type OccurrencePath,
  type ScalarStyle,
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
    expect(values.rangeOf("0x10")).toEqual(new SourceRange(2, 9, 2, 20));

    // Text keys: `true:` and `1:` no longer collide.
    const collision = root.child("collision");
    expect(collision.child("1").range).toEqual(new SourceRange(8, 5, 9, 1));
    expect(collision.child("1").str("nested")).toBe("second");
    expect(collision.child("1").sourceKeyValue).toBe(1n);
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

  test("virtual deep lists and records copy without stack overflow", () => {
    const levels = 2_000;
    let list: Value = "leaf";
    let record: Value = "leaf";
    for (let level = 0; level < levels; level++) {
      list = [list];
      record = { next: record };
    }

    let copiedList = SourceCursor.virtual(list).value;
    let copiedRecord = SourceCursor.virtual(record).value;

    expect(copiedList).not.toBe(list);
    expect(copiedRecord).not.toBe(record);
    for (let level = 0; level < levels; level++) {
      copiedList = (copiedList as Value[])[0]!;
      copiedRecord = (copiedRecord as ValueRecord)["next"]!;
    }
    expect(copiedList).toBe("leaf");
    expect(copiedRecord).toBe("leaf");
  });

  test("the get fallback copies without stack overflow", () => {
    let fallback: Value = "leaf";
    for (let level = 0; level < 2_000; level++) fallback = [fallback];

    let copied = SourceCursor.virtual({}).get("missing", fallback);

    expect(copied).not.toBe(fallback);
    for (let level = 0; level < 2_000; level++) copied = (copied as Value[])[0]!;
    expect(copied).toBe("leaf");
  });

  test("virtual, withValue and projectMapping refuse anything but a YAML value", () => {
    class Custom {}
    class ListSubclass extends Array<Value> {}
    const invalid = [
      new Custom(),
      new Map(),
      new ListSubclass(),
      { nested: new ListSubclass() },
    ] as unknown as Value[];
    const base = SourceCursor.virtual({});

    for (const value of invalid) {
      expect(() => SourceCursor.virtual(value)).toThrow(/source cursors accept only YAML values/);
      expect(() => base.withValue(value)).toThrow(/source cursors accept only YAML values/);
      expect(() => base.projectMapping(new Map([["value", [value, null]]]))).toThrow(
        /source cursors accept only YAML values/,
      );
    }
    const notACursor = () => base.projectMapping(new Map([["value", [1n, {} as SourceCursor]]]));
    expect(notACursor).toThrow(TypeError);
    expect(notACursor).toThrow("projected entry source must be a SourceCursor or null");
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
    const projected = wide.projectMapping(
      new Map(
        Array.from({ length: width }, (_, index) => {
          const key = `key_${index}`;
          return [key, [BigInt(index), wide.child(key)] as const];
        }),
      ),
    );
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
        expect(projected.has(key)).toBe(true);
        expect(projected.child(key).value).toBe(BigInt(index));
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

  test("projected entries are distinct objects", () => {
    const projected = SourceCursor.virtual({}).projectMapping(
      new Map([
        ["first", [1n, null]],
        ["second", [2n, null]],
      ]),
    );

    const [first, second] = projected._projectedEntries!;

    expect(first).not.toBe(second);
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

describe("typed accessors", () => {
  test("typed accessors read required fields and copy containers", () => {
    const root = SourceDocument.parse(
      "name: compose\ncount: 3\nconfig:\n  enabled: true\nitems: [one, two]\n",
    ).root;

    expect(root.str("name")).toBe("compose");
    expect(root.int("count")).toBe(3n);
    expect(root.mapping("config")).toEqual({ enabled: true });
    expect(root.sequence("items")).toEqual(["one", "two"]);
    expect(root.rangeOf("name")).toEqual(new SourceRange(1, 7, 1, 14));
    expect(root.keyRangeOf("name")).toEqual(new SourceRange(1, 1, 1, 5));
    expect(root.keyRangeOf("missing")).toBeNull();

    const mapping = root.mapping("config");
    const sequence = root.sequence("items");
    expect(Object.isFrozen(mapping)).toBe(false);
    expect(Object.isFrozen(sequence)).toBe(false);
    mapping["enabled"] = false;
    sequence.push("three");
    expect(root.mapping("config")).toEqual({ enabled: true });
    expect(root.sequence("items")).toEqual(["one", "two"]);

    expect(accessError(() => root.str("missing")).message).toBe("missing required field 'missing'");
    const notText = accessError(() => root.str("count"));
    expect(notText.message).toBe("field 'count' must be a string, got int");
    expect(notText.range).toEqual(root.rangeOf("count"));
    expect(accessError(() => root.mapping("items")).message).toBe(
      "field 'items' must be a mapping, got list",
    );
    expect(accessError(() => root.sequence("config")).message).toBe(
      "field 'config' must be a sequence, got dict",
    );
  });

  test("int rejects a boolean and str names null as NoneType", () => {
    const root = SourceDocument.parse("flag: true\nnothing:\n").root;

    const error = accessError(() => root.int("flag"));
    expect(error.message).toBe("field 'flag' must be an integer, got bool");
    expect(error.range).toEqual(new SourceRange(1, 7, 1, 11));
    expect(accessError(() => root.str("nothing")).message).toBe(
      "field 'nothing' must be a string, got NoneType",
    );
  });
});

describe("range helpers", () => {
  test("item range lists keep duplicates and item ranges keep the last", () => {
    const root = SourceDocument.parse("asserts:\n  - output > 0\n  - output > 0\n").root;

    expect(root.itemRangeList("asserts")).toEqual([
      new SourceRange(2, 5, 2, 15),
      new SourceRange(3, 5, 3, 15),
    ]);
    expect(root.itemRanges("asserts")).toEqual(
      new Map([["output > 0", new SourceRange(3, 5, 3, 15)]]),
    );
    expect(root.itemRangeList("missing")).toEqual([]);
    expect(root.itemRanges("missing")).toEqual(new Map());
  });

  test("sequence field ranges follow item order", () => {
    const root = SourceDocument.parse(
      "cases:\n" +
        '  - when: "a >= 1"\n' +
        "    then: x\n" +
        '  - when: "b >= 2"\n' +
        "    then: y\n" +
        "  - scalar\n",
    ).root;

    expect(root.sequenceFieldRanges("cases", "when")).toEqual([
      new SourceRange(2, 11, 2, 19),
      new SourceRange(4, 11, 4, 19),
      null,
    ]);
    expect(root.sequenceFieldRanges("cases", "then")).toEqual([
      new SourceRange(3, 11, 3, 12),
      new SourceRange(5, 11, 5, 12),
      null,
    ]);
    expect(root.mappingValueRanges("missing")).toEqual(new Map());
  });

  test("mapping value ranges keep each key's last range in first-key order", () => {
    const root = SourceDocument.parse("m:\n  a: 1\n  b: 2\n  a: 3\n").root;

    const ranges = root.mappingValueRanges("m");
    expect([...ranges.keys()]).toEqual(["a", "b"]);
    expect(ranges.get("a")).toEqual(new SourceRange(4, 6, 4, 7));
  });

  test("a merge range summary folds a shared rope without enumerating", () => {
    unboundedNodes();
    const document = binaryMergeDocument(24);
    const original = SourceCursor.prototype._iterFlatOccurrences;
    SourceCursor.prototype._iterFlatOccurrences = () => {
      throw new Error("an effective range lookup must not flatten occurrences");
    };
    try {
      expect(document.root.mappingValueRanges("root")).toEqual(
        new Map([["value", new SourceRange(2, 10, 2, 14)]]),
      );
    } finally {
      SourceCursor.prototype._iterFlatOccurrences = original;
    }
  });

  test("range helpers delegate through renamed to the child sources", () => {
    const base = SourceDocument.parse(
      "input:\n" +
        "  value: ${start.output}\n" +
        "asserts:\n" +
        "  - output > 0\n" +
        "cases:\n" +
        "  - when: ready\n" +
        "    then: finish\n",
    ).root;

    const renamed = base.renamed(new Map([["input", "inputs"]]));

    expect(renamed.mappingValueRanges("inputs")).toEqual(base.mappingValueRanges("input"));
    expect(renamed.mappingValueRanges("inputs").size).toBe(1);
    expect(renamed.itemRanges("asserts")).toEqual(base.itemRanges("asserts"));
    expect(renamed.itemRanges("asserts").size).toBe(1);
    expect(renamed.itemRangeList("asserts")).toEqual(base.itemRangeList("asserts"));
    for (const field of ["when", "then"]) {
      expect(renamed.sequenceFieldRanges("cases", field)).toEqual(
        base.sequenceFieldRanges("cases", field),
      );
      expect(renamed.sequenceFieldRanges("cases", field)[0]).not.toBeNull();
    }
  });
});

describe("projections", () => {
  test("renamed doesn't change the base and keeps child paths and ranges", () => {
    const base = SourceDocument.parse("input: source\nkeep: unchanged\n").root;

    const renamed = base.renamed(new Map([["input", "inputs"]]));

    expect(renamed.value).toEqual({ inputs: "source", keep: "unchanged" });
    expect(base.value).toEqual({ input: "source", keep: "unchanged" });
    expect(base.has("inputs")).toBe(false);
    expect(renamed.child("inputs").path).toEqual(base.child("input").path);
    expect(renamed.child("inputs").range).toEqual(base.child("input").range);
    expect(renamed.child("inputs").keyRange).toEqual(base.child("input").keyRange);
    expect(renamed.keyRangeOf("inputs")).toEqual(base.keyRangeOf("input"));
  });

  test("renamed refuses a key that collides with another", () => {
    const base = SourceDocument.parse("input: singular\ninputs: plural\n").root;

    const error = accessError(() => base.renamed(new Map([["input", "inputs"]])));

    expect(error.message).toBe("key `input` and key `inputs` both mean 'inputs'; keep one");
    expect(error.range).toEqual(base.keyRangeOf("input"));
  });

  test("a rename collision names keys as authored", () => {
    const base = SourceDocument.parse('0x1F: plain\n"x": quoted\n').root;

    const error = accessError(() => base.renamed(new Map([["0x1F", "x"]])));

    expect(error.message.startsWith("key `0x1F` and key `\"x\"` both mean 'x'; keep one")).toBe(
      true,
    );
    expect(error.message).not.toContain("31");
    expect(error.range).toEqual(base.keyRangeOf("0x1F"));
  });

  test("renamed can swap two keys and refuses a non-mapping", () => {
    const base = SourceDocument.parse("a: 1\nb: 2\nc: [1]\n").root;

    const swapped = base.renamed(
      new Map([
        ["a", "b"],
        ["b", "a"],
      ]),
    );

    expect(swapped.value).toEqual({ b: 1n, a: 2n, c: [1n] });
    expect(Object.keys(swapped.value as ValueRecord)).toEqual(["b", "a", "c"]);
    expect(accessError(() => base.child("c").renamed(new Map())).message).toBe(
      "cannot rename keys on a non-mapping value",
    );
  });

  test("withValue aligns nested mapping and sequence sources", () => {
    const source = SourceDocument.parse(
      "body:\n  mapping:\n    field: old\n  sequence:\n    - field: first\n",
    ).root.child("body");

    const aligned = source.withValue({
      mapping: { field: "new", generated: true },
      sequence: [{ field: "changed" }, { field: "generated" }],
    });

    const mapping = aligned.child("mapping");
    const sequence = aligned.child("sequence");
    expect(mapping.child("field").range).toEqual(source.child("mapping").child("field").range);
    expect(mapping.child("field").value).toBe("new");
    expect(mapping.child("generated").range).toBeNull();
    expect(sequence.child(0).child("field").range).toEqual(
      source.child("sequence").child(0).child("field").range,
    );
    expect(sequence.child(1).document).toBeNull();
    expect(sequence.child(1).child("field").range).toBeNull();
    expect(sequence.children().map((item) => item.value)).toEqual([
      { field: "changed" },
      { field: "generated" },
    ]);
    expect(aligned.sequenceFieldRanges("sequence", "field")).toEqual([
      source.child("sequence").child(0).child("field").range,
      null,
    ]);
  });

  test("a projected child keeps the source's document, path and range", () => {
    const base = SourceDocument.parse("original:\n  nested: old\n").root;
    const source = base.child("original");

    const projected = base.projectMapping(
      new Map<string, readonly [Value, SourceCursor | null]>([
        ["renamed", [{ nested: ["new"] }, source]],
        ["generated", [{ virtual: true }, null]],
      ]),
    );
    const retained = projected.child("renamed");
    const generated = projected.child("generated");

    expect(projected.document).toBe(base.document);
    expect(projected.path).toEqual(base.path);
    expect(projected.range).toEqual(base.range);
    expect(retained.value).toEqual({ nested: ["new"] });
    expect(retained.document).toBe(source.document);
    expect(retained.path).toEqual(source.path);
    expect(retained.range).toEqual(source.range);
    expect(retained.keyRange).toEqual(source.keyRange);
    expect(generated.document).toBeNull();
    expect(generated.path).toEqual([]);
    expect(generated.range).toBeNull();
    expect(generated.keyRange).toBeNull();
    expect(projected.mappingOccurrences("renamed").map((child) => child.range)).toEqual([
      source.range,
    ]);
    expect(projected.mappingOccurrences().length).toBe(2);
    expect(projected.children().map((child) => child.document)).toEqual([base.document, null]);
  });

  test("projected mutation can't change the overlay", () => {
    const base = SourceDocument.parse("source: original\n").root;
    const original: ValueRecord = { nested: [1n] };
    const projected = base.projectMapping(new Map([["source", [original, base.child("source")]]]));

    (original["nested"] as Value[]).push(2n);
    const first = projected.value as Record<string, ValueRecord>;
    (first["source"]!["nested"] as Value[]).push(3n);
    const childValue = projected.child("source").value as ValueRecord;
    (childValue["nested"] as Value[]).push(4n);

    expect(projected.value).toEqual({ source: { nested: [1n] } });
    expect(Object.isFrozen(projected._valueView())).toBe(true);
    expect(base.value).toEqual({ source: "original" });
  });
});

describe("scalar facts", () => {
  test("scalarSource is a frozen view of an unchanged text scalar", () => {
    const cursor = SourceDocument.parse("code: |-\n  first\n").root.child("code");

    const source = cursor.scalarSource!;
    expect(Object.isFrozen(source)).toBe(true);
    expect(source.range).toEqual(cursor.range!);
    expect(source.constructedLineRanges).toEqual([new SourceRange(2, 3, 2, 8)]);
    expect(Object.isFrozen(source.constructedLineRanges)).toBe(true);
    expect(Object.isFrozen(source.constructedLineRanges![0])).toBe(true);
    expect(Object.isFrozen(source.range)).toBe(true);
    const empty = SourceDocument.parse("code: |").root.child("code").scalarSource!;
    expect(empty.constructedLineRanges).toEqual([]);
    expect(Object.isFrozen(empty.constructedLineRanges)).toBe(true);
    expect(SourceCursor.virtual("first\nsecond").scalarSource).toBeNull();
    expect(SourceDocument.parse("n: 3\n").root.child("n").scalarSource).toBeNull();
  });

  /** One literal case: name, text, built value, and each constructed line's range bounds. */
  type LiteralCase = [string, string, string, [number, number, number, number][]];
  const literalCases: LiteralCase[] = [
    ["clip", "code: |\n  alpha\n  beta\n", "alpha\nbeta\n", [[2, 3, 2, 8], [3, 3, 3, 7]]],
    ["strip", "code: |-\n  alpha\n  beta\n", "alpha\nbeta", [[2, 3, 2, 8], [3, 3, 3, 7]]],
    [
      "keep",
      "code: |+\n  alpha\n  beta\n\n\n",
      "alpha\nbeta\n\n\n",
      [[2, 3, 2, 8], [3, 3, 3, 7], [4, 1, 4, 1], [5, 1, 5, 1]],
    ],
    [
      "explicit indent",
      "code: |2-\n    first\n      second\n",
      "  first\n    second",
      [[2, 3, 2, 10], [3, 3, 3, 13]],
    ],
    [
      "blank lines",
      "code: |+\n\n  first\n  \n  second\n\n",
      "\nfirst\n\nsecond\n\n",
      [[2, 1, 2, 1], [3, 3, 3, 8], [4, 3, 4, 3], [5, 3, 5, 9], [6, 1, 6, 1]],
    ],
    ["CRLF", "code: |-\r\n  first\r\n  second\r\n", "first\nsecond", [[2, 3, 2, 8], [3, 3, 3, 9]]],
    [
      "anchored tagged header",
      "code: &shared # fake | indicator\n  !!str |-\n    first\n",
      "first",
      [[3, 5, 3, 10]],
    ],
    ["EOF", "code: |-\n  final", "final", [[2, 3, 2, 8]]],
    ["empty clip", "code: |\n", "", []],
    ["empty strip", "code: |-\n", "", []],
    ["empty keep", "code: |+\n", "", []],
    ["empty clip at EOF", "code: |", "", []],
    ["empty strip at EOF", "code: |-", "", []],
    ["empty keep at EOF", "code: |+", "", []],
    ["kept blank lines", "code: |+\n\n\n", "\n\n", [[2, 1, 2, 1], [3, 1, 3, 1]]],
  ];

  test.each(literalCases)(
    "scalarSource maps literal lines: %s",
    (_name, text, value, expected) => {
      const cursor = SourceDocument.parse(text).root.child("code");

      expect(cursor.value).toBe(value);
      const source = cursor.scalarSource!;
      expect(source.range).toEqual(cursor.range!);
      expect(source.constructedLineRanges).toEqual(
        expected.map((bounds) => new SourceRange(...bounds)),
      );
    },
  );

  test.each([
    ["folded", "code: >-\n  first\n  second\n"],
    ["quoted", 'code: "first\\nsecond"\n'],
    ["plain", "code: first\n  second\n"],
  ])("scalarSource falls back to the whole range: %s", (_name, text) => {
    const cursor = SourceDocument.parse(text).root.child("code");

    const source = cursor.scalarSource!;
    expect(source.range).toEqual(cursor.range!);
    expect(source.constructedLineRanges).toBeNull();
  });

  test.each([
    ["plain", "value: x = 1\n", "plain"],
    ["single", "value: 'x = 1'\n", "single"],
    ["double", 'value: "x = 1"\n', "double"],
    ["literal", "value: |\n  x = 1\n", "literal"],
    ["folded", "value: >\n  x = 1\n", "folded"],
    ["multi-line plain", "value: first\n  second\n", "plain"],
    ["plain int", "value: 3\n", "plain"],
    ["plain null", "value:\n", "plain"],
  ] as [string, string, ScalarStyle][])(
    "scalarStyle reports the authored style: %s",
    (_name, text, expected) => {
      expect(SourceDocument.parse(text).root.child("value").scalarStyle).toBe(expected);
    },
  );

  test("scalarStyle of sequence items and aliases; null for containers", () => {
    const root = SourceDocument.parse(
      "items: [plain, 'single', \"double\"]\n" +
        "anchored: &a 'x = 1'\n" +
        "alias: *a\n" +
        "map: {a: 1}\n",
    ).root;

    expect(root.child("items").children().map((item) => item.scalarStyle)).toEqual([
      "plain",
      "single",
      "double",
    ]);
    expect(root.child("alias").scalarStyle).toBe("single");
    expect(root.child("map").scalarStyle).toBeNull();
    expect(root.child("items").scalarStyle).toBeNull();
    expect(SourceCursor.virtual("x").scalarStyle).toBeNull();
  });

  test("scalarText, keyText and sourceKeyValue read the authored source", () => {
    const root = SourceDocument.parse("a: !str yes\nb: 30\nc: 'q'\nd: [1]\n0x10: hex\n").root;

    expect(root.child("a").scalarText).toBe("yes");
    expect(root.child("b").scalarText).toBe("30");
    expect(root.child("c").scalarText).toBe("q");
    expect(root.child("d").scalarText).toBeNull();
    expect(root.child("0x10").keyText).toBe("0x10");
    expect(root.child("0x10").sourceKeyValue).toBe(16n);
    expect(root.child("d").child(0).keyText).toBeNull();
    expect(root.child("d").child(0).sourceKeyValue).toBeNull();
    expect(root.keyText).toBeNull();
    expect(SourceCursor.virtual(1).keyText).toBeNull();
  });

  test("tags, tag definition ids and hasValue on physical cursors", () => {
    const root = SourceDocument.parse("a: !int 5\nb: !int\nc: !int null\nd: 7\n").root;

    expect(root.child("a").tag).toBe("int");
    expect(root.child("a").hasValue).toBe(true);
    expect(root.child("b").tag).toBe("int");
    expect(root.child("b").hasValue).toBe(false);
    expect(root.child("c").hasValue).toBe(true);
    expect(root.child("d").tag).toBeNull();
    expect(root.child("d").tagDefinitionId).toBeNull();
    expect(root.child("a").tagDefinitionId).toBe(root.child("a")._definition);
    expect(SourceCursor.virtual(1).tag).toBeNull();
    expect(SourceCursor.virtual(null).hasValue).toBe(true);
  });

  test("scalarSource and scalarStyle hold through withValue while unchanged", () => {
    const code = SourceDocument.parse("code: |-\n  first\n  second\n").root.child("code");
    const value = SourceDocument.parse("value: x\n").root.child("value");

    expect(code.withValue("first\nsecond").scalarSource).toEqual(code.scalarSource);
    expect(code.withValue("first\nchanged").scalarSource).toBeNull();
    expect(value.withValue("x").scalarStyle).toBe("plain");
    expect(value.withValue("other").scalarStyle).toBeNull();
  });
});
