/**
 * Immutable, occurrence-aware YAML source cursors: a port of
 * `tests/compose/test_source_cursor.py`.
 */

import { describe, expect, test } from "bun:test";

import { freezeValue, setRecordKey, snapshotValue } from "../../src/compose/source/common.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";

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
    const levels = 2_000;
    let list: Value = "leaf";
    let record: Value = "leaf";
    let dag: Value = ["leaf"];
    for (let level = 0; level < levels; level++) {
      list = [list];
      record = { next: record };
      dag = [dag, dag];
    }

    let copiedList = snapshotValue(list);
    let copiedRecord = snapshotValue(record);
    let copiedDag = snapshotValue(dag);

    expect(copiedList).not.toBe(list);
    expect(copiedRecord).not.toBe(record);
    expect(copiedDag).not.toBe(dag);
    for (let level = 0; level < levels; level++) {
      copiedList = (copiedList as Value[])[0]!;
      copiedRecord = (copiedRecord as ValueRecord)["next"]!;
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
    for (let level = 0; level < 2_000; level++) deep = [deep];
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
