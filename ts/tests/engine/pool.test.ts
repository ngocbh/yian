/**
 * `VariablePool`: typed reads and writes of a node's single value, the `${...}` resolver, the
 * declared-type write check and the lossless round trip, a port of `tests/engine/test_pool.py`.
 *
 * TS-only: the default and a custom `startId`, read rows for `system` with a longer path and
 * `input` with none, a dotted read through a `FileRef` field, a failed `set` keeping the
 * previous value, the exact `dumps()` text, the int-in-float and infinity round trips, node ids
 * `constructor` and `__proto__`, `fromWire` reading `toWire`'s object, the `fromWire` refusal
 * table (Python's strict reader is `runtime/log.py::decode_pool`, whose messages these keep) and
 * `loads` of bad JSON.
 *
 * Not ported: Python's carrier-class check (`type(stored) is ListObjectValue`); TS has one
 * `TypedValue` class, so the round trip checks the kind and `instanceof TypedValue`.
 */

import { describe, expect, test } from "bun:test";

import { evalExpr, optionalInsideRoot, resolveReference } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { JSONDecodeError } from "../../src/typesys/json.ts";
import { poolRootLength, referenceRoot, VariablePool } from "../../src/typesys/pool.ts";
import { FileRef, Type, TypeCheckError, TypedValue, type Value, ValueKind } from "../../src/typesys/values.ts";

// The START node's id; `boundary_ids.test.ts` pins the pool's default to `StartNode.ID`.
const START_ID = "__start__";

/** Read `${path}` from `pool` as a binding does: the pool reads the root, the evaluator walks the rest. */
function read(pool: VariablePool, path: string): Value {
  return evalExpr(parseExpr(path), (names) => resolveReference(names, pool));
}

describe("VariablePool reads and writes", () => {
  test("set and get a node's value (test_set_and_get)", () => {
    const pool = new VariablePool();
    pool.set("n1", "hello");
    expect(pool.get("n1")).toBe("hello");
    expect(pool.getValue("n1")!.kind).toBe(ValueKind.STRING);
    // a missing node resolves to the fallback
    expect(pool.get("missing", "d")).toBe("d");
  });

  test("get reads an object value whole (test_get_whole_object_value)", () => {
    // "multiple outputs" are fields of one object value, read back whole.
    const pool = new VariablePool();
    pool.set("n1", { answer: "txt", score: 0.8 });
    expect(pool.get("n1")).toEqual({ answer: "txt", score: 0.8 });
    expect(pool.get("nope")).toBeNull();
  });

  test("a declared type rejects a drifting value (test_declared_type_drift_raises)", () => {
    const pool = new VariablePool();
    // a node declaring an integer output but emitting a string fails at the write
    expect(() => pool.set("n1", "twelve", ValueKind.INTEGER)).toThrow(TypeCheckError);
    // the good case writes through
    pool.set("n1", 12n, ValueKind.INTEGER);
    expect(pool.get("n1")).toBe(12n);
  });

  test("set takes a Type and enforces it (test_pool_set_accepts_type_and_enforces)", () => {
    const pool = new VariablePool();
    const action = new Type({ kind: ValueKind.STRING, tags: new Set(["Approve", "Reject"]) });
    pool.set("n", "Approve", action);
    expect(pool.get("n")).toBe("Approve");
    expect(() => pool.set("n", "approve", action)).toThrow(TypeCheckError);
  });

  test("a failed set keeps the previous value", () => {
    const pool = new VariablePool();
    pool.set("n", 1n, ValueKind.INTEGER);
    expect(() => pool.set("n", "one", ValueKind.INTEGER)).toThrow(TypeCheckError);
    expect(pool.get("n")).toBe(1n);
  });
});

describe("VariablePool.readRoot through the evaluator", () => {
  test("a node-first head walks the value (test_resolve_node_first_and_structured)", () => {
    // `.output` is a syntactic discriminator the resolver skips: store the node's value
    // directly (wrapping it as `{output: ...}` would only test the record-key arm by accident).
    const pool = new VariablePool();
    pool.set("reviewer", { ratio: 21.5, nested: { k: "v" } });
    // ${reviewer.output} (2-segment) returns the whole stored value
    expect(read(pool, "reviewer.output")).toEqual({ ratio: 21.5, nested: { k: "v" } });
    // ${reviewer.output.ratio} (3-segment) walks one level in
    expect(read(pool, "reviewer.output.ratio")).toBe(21.5);
    expect(read(pool, "reviewer.output.nested.k")).toBe("v");
    // missing path -> null, not an error
    expect(read(pool, "reviewer.output.absent")).toBeNull();
    expect(read(pool, "ghost.output")).toBeNull();
  });

  test("retired heads resolve to null (test_retired_heads_resolve_none)", () => {
    // `node` and `ref_outputs` were unified into `<id>.output`; they are now just unknown heads.
    const pool = new VariablePool();
    pool.set("reviewer", "txt");
    expect(read(pool, "node.reviewer.output")).toBeNull();
    expect(read(pool, "ref_outputs.reviewer.output")).toBeNull();
  });

  test("input and system heads (test_resolve_inputs_and_system)", () => {
    const pool = new VariablePool();
    pool.set(START_ID, { topic: "ACME" }); // ${input.X} ≡ store[START_ID].X
    expect(read(pool, "input.topic")).toBe("ACME"); // the flow's run args
    expect(read(pool, "input.missing")).toBeNull();
    expect(read(pool, "system.anything")).toBeNull(); // ambient namespace, unseeded
    // `parent` is retired: it is now just an unknown head
    expect(read(pool, "parent.threshold")).toBeNull();
    expect("parent" in pool).toBe(false);
    expect(read(pool, "unknown_ns.x")).toBeNull();
  });

  test(
    "a subflow's value resolves via its node-first head " +
      "(test_subflow_outputs_resolve_via_node_first_head)",
    () => {
      // A CALL/subflow node's value lives under its own id in `store`.
      const pool = new VariablePool();
      pool.set("child_ref", "child-result");
      expect(read(pool, "child_ref.output")).toBe("child-result");
    },
  );

  test("input reads the START record (test_inputs_resolve_via_start_store)", () => {
    // The standalone `inputs` namespace (and `add_inputs`) is retired: ${input.X} resolves to
    // store[START_ID].X, the START node's committed bound-input record.
    const pool = new VariablePool();
    pool.set(START_ID, { window: 30n, note: "anything" });
    expect(read(pool, "input.window")).toBe(30n);
    expect(read(pool, "input.note")).toBe("anything");
    expect("addInputs" in pool).toBe(false);
    expect("inputs" in pool).toBe(false);
  });

  test("startId defaults to __start__ and a custom one is where input reads", () => {
    expect(new VariablePool().startId).toBe("__start__");
    const pool = new VariablePool("c/__start__");
    pool.set("c/__start__", { x: "inner" });
    pool.set(START_ID, { x: "outer" });
    expect(read(pool, "input.x")).toBe("inner");
  });

  test("system reads the whole ambient; a field of its text misses", () => {
    const pool = new VariablePool();
    pool.addSystem("today", "2026-10-18");
    expect(read(pool, "system.today")).toBe("2026-10-18");
    expect(read(pool, "system.today.x")).toBeNull();
    expect(read(pool, "system")).toBeNull();
  });

  test("input with no key resolves to null", () => {
    const pool = new VariablePool();
    pool.set(START_ID, { x: 1n });
    expect(read(pool, "input")).toBeNull();
  });

  test("a dotted read through a FileRef reads its field", () => {
    const pool = new VariablePool();
    pool.set("doc", new FileRef({ uri: "s3://bucket/a.txt" }), ValueKind.FILE);
    expect(read(pool, "doc.output.uri")).toBe("s3://bucket/a.txt");
    expect(read(pool, "doc.output.mime")).toBeNull();
    expect(read(pool, "doc.output.size")).toBeNull();
  });
});

describe("VariablePool round trip", () => {
  test("dumps and loads keep every value's type (test_full_pool_round_trip)", () => {
    const pool = new VariablePool();
    pool.set("n1", { a: 1n, b: [1.5, 2.5] });
    pool.set("flag_node", true);
    pool.set("n2", ["ACME", "BETA"]);
    pool.set(START_ID, { as_of_date: "2026-06-05" }); // the flow's run args (the START record)

    const back = VariablePool.loads(pool.dumps());

    expect(back.get("n1")).toEqual({ a: 1n, b: [1.5, 2.5] });
    expect(back.getValue("flag_node")!.kind).toBe(ValueKind.BOOLEAN);
    expect(back.getValue("n2")!.kind).toBe(ValueKind.LIST_STRING);
    // store[START_ID] survives dumps/loads
    expect(read(back, "input.as_of_date")).toBe("2026-06-05");
    expect(back.get("flag_node")).toBe(true);
  });

  test(
    "nullable specialized list items survive " +
      "(test_pool_round_trip_preserves_nullable_specialized_list_elements)",
    () => {
      const row = new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([["score", Type.scalar(ValueKind.NUMBER)]]),
        required: new Set(["score"]),
        nullable: true,
      });
      const rows = new Type({ kind: ValueKind.LIST_OBJECT, element: row });
      const pool = new VariablePool();
      pool.set("rows", [{ score: 0.9 }, null], rows);

      const restored = VariablePool.loads(pool.dumps());

      const stored = restored.getValue("rows");
      expect(stored).toBeInstanceOf(TypedValue);
      expect(stored!.kind).toBe(ValueKind.LIST_OBJECT);
      expect(restored.get("rows")).toEqual([{ score: 0.9 }, null]);
    },
  );

  test("dumps writes the wire object with sorted keys", () => {
    const pool = new VariablePool();
    pool.set("n", 2.5);
    expect(pool.dumps()).toBe(
      '{"start_id":"__start__","store":{"n":{"kind":"float","value":2.5}},"system":{}}',
    );
  });

  test("an int in a float slot and a list(float) holding an int round-trip", () => {
    const pool = new VariablePool();
    pool.set("n", 2n, ValueKind.NUMBER);
    pool.set("xs", [1n, 2.5], ValueKind.LIST_NUMBER);
    const back = VariablePool.loads(pool.dumps());
    expect(back.getValue("n")!.kind).toBe(ValueKind.INTEGER);
    expect(back.get("n")).toBe(2n);
    expect(back.getValue("xs")!.kind).toBe(ValueKind.LIST_NUMBER);
    expect(back.get("xs")).toEqual([1n, 2.5]);
  });

  test("infinity round-trips", () => {
    const pool = new VariablePool();
    pool.set("n", Infinity);
    pool.addSystem("low", -Infinity);
    const back = VariablePool.loads(pool.dumps());
    expect(back.get("n")).toBe(Infinity);
    expect(read(back, "system.low")).toBe(-Infinity);
  });

  test("node ids constructor and __proto__ round-trip as own keys", () => {
    const pool = new VariablePool();
    pool.set("constructor", 1n);
    pool.set("__proto__", "p");
    const wire = pool.toWire();
    expect(Object.hasOwn(wire.store, "constructor")).toBe(true);
    expect(Object.hasOwn(wire.store, "__proto__")).toBe(true);
    const back = VariablePool.loads(pool.dumps());
    expect(back.get("constructor")).toBe(1n);
    expect(back.get("__proto__")).toBe("p");
  });

  test("fromWire reads toWire's object", () => {
    const pool = new VariablePool("c/__start__");
    pool.set("n", "v");
    pool.addSystem("run_id", "r1");
    const back = VariablePool.fromWire(pool.toWire());
    expect(back.startId).toBe("c/__start__");
    expect(back.get("n")).toBe("v");
    expect(read(back, "system.run_id")).toBe("r1");
  });

  const entry = { kind: "str", value: "v" };
  test.each([
    ["not a record", ["store"], "variable-pool wire must be a JSON object"],
    [
      "a missing and an extra key",
      { store: {}, start_id: "s", x: 1n },
      'invalid variable pool fields; missing=["system"], extra=["x"]',
    ],
    [
      "store an array",
      { store: [], system: {}, start_id: "s" },
      "variable-pool store and system must be JSON objects",
    ],
    [
      "a store entry that isn't a record",
      { store: { n: "v" }, system: {}, start_id: "s" },
      "variable-pool store entries must be typed objects",
    ],
    [
      "a system entry that isn't a record",
      { store: { n: entry }, system: { k: [entry] }, start_id: "s" },
      "variable-pool system entries must be typed objects",
    ],
    [
      "a bad entry",
      { store: { n: { kind: "int", value: "x" } }, system: {}, start_id: "s" },
      `invalid int typed value: expected a bigint, got "x"`,
    ],
    [
      "start_id not text",
      { store: {}, system: {}, start_id: 1n },
      "variable-pool start_id must be a string",
    ],
  ])("fromWire refuses %s", (_label, data, message) => {
    const read = () => VariablePool.fromWire(data);
    expect(read).toThrow(RangeError);
    expect(read).toThrow(new RangeError(message));
  });

  test("loads of bad JSON throws JSONDecodeError", () => {
    expect(() => VariablePool.loads("{")).toThrow(JSONDecodeError);
  });
});

test("a pool root takes two names under a namespace, else one", () => {
  expect(poolRootLength(["a", "output", "k"])).toBe(2);
  expect(poolRootLength(["a", "new_messages"])).toBe(2);
  expect(poolRootLength(["input", "k", "x"])).toBe(2);
  expect(poolRootLength(["system", "now"])).toBe(2);
  expect(poolRootLength(["item", "output"])).toBe(1);
  expect(poolRootLength(["x", "k"])).toBe(1);
  expect(poolRootLength(["a"])).toBe(1);
});

test("a comprehension variable is a one-name root, so ?. after it is never inside its root", () => {
  expect(optionalInsideRoot(parseExpr("[m?.output for m in xs]"), poolRootLength)).toBeNull();
  expect(optionalInsideRoot(parseExpr("[m.output for m in a?.output]"), poolRootLength)).toContain("${a.output}");
});

test("referenceRoot names the pool entry and how many names the root used", () => {
  const holds = (name: string) => name === "a" || name === "a.new_messages";
  expect(referenceRoot(["a", "output", "k"], START_ID, holds)).toEqual({ name: "a", used: 2 });
  expect(referenceRoot(["a", "new_messages"], START_ID, holds)).toEqual({ name: "a.new_messages", used: 2 });
  expect(referenceRoot(["input", "k", "x"], START_ID, holds)).toEqual({ name: START_ID, used: 2, key: "k" });
  expect(referenceRoot(["system", "now"], START_ID, holds)).toEqual({ name: "system.now", used: 2 });
  expect(referenceRoot(["b", "output", "k"], START_ID, holds)).toEqual({ name: null, used: 2 });
  expect(referenceRoot(["x", "k"], START_ID, holds)).toEqual({ name: null, used: 1 });
});

test("the pool reads a root: a present null is not missing", () => {
  const pool = new VariablePool();
  pool.set(START_ID, { k: null });
  pool.set("a", { k: 1n });
  pool.addSystem("now", "2026-10-25");
  expect(pool.readRoot(["input", "k", "x"])).toEqual({ used: 2, present: true, value: null, why: null });
  expect(pool.readRoot(["input", "j"])).toEqual({ used: 2, present: false, value: null, why: "input 'j' was not given" });
  expect(pool.readRoot(["a", "output", "k"])).toEqual({ used: 2, present: true, value: { k: 1n }, why: null });
  expect(pool.readRoot(["b", "output"])).toEqual({ used: 2, present: false, value: null, why: "node 'b' was skipped" });
  expect(pool.readRoot(["b", "new_messages"])).toEqual({ used: 2, present: false, value: null, why: "node 'b' was skipped" });
  expect(pool.readRoot(["system", "now"])).toEqual({ used: 2, present: true, value: "2026-10-25", why: null });
  expect(pool.readRoot(["system", "later"])).toEqual({
    used: 2,
    present: false,
    value: null,
    why: "system value 'later' is not set",
  });
  expect(pool.readRoot(["x", "k"])).toEqual({ used: 1, present: false, value: null, why: null });
});
