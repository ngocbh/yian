/**
 * `VariablePool`: typed reads and writes of a node's single value, the `${...}` resolver, the
 * declared-type write check and the lossless round trip, a port of `tests/engine/test_pool.py`.
 *
 * TS-only: the default and a custom `startId`, `resolve` rows for `system` with a longer path
 * and `input` with none, a dotted read through a `FileRef` field, a failed `set` keeping the
 * previous value, the exact `dumps()` text, the int-in-float and infinity round trips, node ids
 * `constructor` and `__proto__`, `fromWire` reading `toWire`'s object, the `fromWire` refusal
 * table (Python's strict reader is `runtime/log.py::decode_pool`, whose messages these keep) and
 * `loads` of bad JSON.
 *
 * Not ported: Python's carrier-class check (`type(stored) is ListObjectValue`); TS has one
 * `TypedValue` class, so the round trip checks the kind and `instanceof TypedValue`.
 */

import { describe, expect, test } from "bun:test";

import { JSONDecodeError } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FileRef, Type, TypeCheckError, TypedValue, ValueKind } from "../../src/typesys/values.ts";

// The START node's id; the `nodes/` port will pin it to `StartNode.ID`.
const START_ID = "__start__";

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

describe("VariablePool.resolve", () => {
  test("a node-first head walks the value (test_resolve_node_first_and_structured)", () => {
    // `.output` is a syntactic discriminator the resolver skips: store the node's value
    // directly (wrapping it as `{output: ...}` would only test the record-key arm by accident).
    const pool = new VariablePool();
    pool.set("reviewer", { ratio: 21.5, nested: { k: "v" } });
    // ${reviewer.output} (2-segment) returns the whole stored value
    expect(pool.resolve("reviewer", ["output"])).toEqual({ ratio: 21.5, nested: { k: "v" } });
    // ${reviewer.output.ratio} (3-segment) walks one level in
    expect(pool.resolve("reviewer", ["output", "ratio"])).toBe(21.5);
    expect(pool.resolve("reviewer", ["output", "nested", "k"])).toBe("v");
    // missing path -> null, not an error
    expect(pool.resolve("reviewer", ["output", "absent"])).toBeNull();
    expect(pool.resolve("ghost", ["output"])).toBeNull();
  });

  test("retired heads resolve to null (test_retired_heads_resolve_none)", () => {
    // `node` and `ref_outputs` were unified into `<id>.output`; they are now just unknown heads.
    const pool = new VariablePool();
    pool.set("reviewer", "txt");
    expect(pool.resolve("node", ["reviewer", "output"])).toBeNull();
    expect(pool.resolve("ref_outputs", ["reviewer", "output"])).toBeNull();
  });

  test("input and system heads (test_resolve_inputs_and_system)", () => {
    const pool = new VariablePool();
    pool.set(START_ID, { topic: "ACME" }); // ${input.X} ≡ store[START_ID].X
    expect(pool.resolve("input", ["topic"])).toBe("ACME"); // the flow's run args
    expect(pool.resolve("input", ["missing"])).toBeNull();
    expect(pool.resolve("system", ["anything"])).toBeNull(); // ambient namespace, unseeded
    // `parent` is retired: it is now just an unknown head
    expect(pool.resolve("parent", ["threshold"])).toBeNull();
    expect("parent" in pool).toBe(false);
    expect(pool.resolve("unknown_ns", ["x"])).toBeNull();
  });

  test(
    "a subflow's value resolves via its node-first head " +
      "(test_subflow_outputs_resolve_via_node_first_head)",
    () => {
      // A CALL/subflow node's value lives under its own id in `store`.
      const pool = new VariablePool();
      pool.set("child_ref", "child-result");
      expect(pool.resolve("child_ref", ["output"])).toBe("child-result");
    },
  );

  test("input reads the START record (test_inputs_resolve_via_start_store)", () => {
    // The standalone `inputs` namespace (and `add_inputs`) is retired: ${input.X} resolves to
    // store[START_ID].X, the START node's committed bound-input record.
    const pool = new VariablePool();
    pool.set(START_ID, { window: 30n, note: "anything" });
    expect(pool.resolve("input", ["window"])).toBe(30n);
    expect(pool.resolve("input", ["note"])).toBe("anything");
    expect("addInputs" in pool).toBe(false);
    expect("inputs" in pool).toBe(false);
  });

  test("startId defaults to __start__ and a custom one is where input reads", () => {
    expect(new VariablePool().startId).toBe("__start__");
    const pool = new VariablePool("c/__start__");
    pool.set("c/__start__", { x: "inner" });
    pool.set(START_ID, { x: "outer" });
    expect(pool.resolve("input", ["x"])).toBe("inner");
  });

  test("system returns the whole ambient and ignores the rest of the path", () => {
    const pool = new VariablePool();
    pool.addSystem("today", "2026-10-18");
    expect(pool.resolve("system", ["today"])).toBe("2026-10-18");
    expect(pool.resolve("system", ["today", "x"])).toBe("2026-10-18");
    expect(pool.resolve("system", [])).toBeNull();
  });

  test("input with no key resolves to null", () => {
    const pool = new VariablePool();
    pool.set(START_ID, { x: 1n });
    expect(pool.resolve("input", [])).toBeNull();
  });

  test("a dotted read through a FileRef reads its field", () => {
    const pool = new VariablePool();
    pool.set("doc", new FileRef({ uri: "s3://bucket/a.txt" }), ValueKind.FILE);
    expect(pool.resolve("doc", ["output", "uri"])).toBe("s3://bucket/a.txt");
    expect(pool.resolve("doc", ["output", "mime"])).toBeNull();
    expect(pool.resolve("doc", ["output", "size"])).toBeNull();
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
    expect(back.resolve("input", ["as_of_date"])).toBe("2026-06-05");
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
    expect(back.resolve("system", ["low"])).toBe(-Infinity);
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
    expect(back.resolve("system", ["run_id"])).toBe("r1");
  });

  const entry = { kind: "str", value: "v" };
  test.each([
    ["not a record", ["store"], "variable-pool wire must be a JSON object"],
    [
      "a missing and an extra key",
      { store: {}, start_id: "s", x: 1n },
      "invalid variable pool fields; missing=['system'], extra=['x']",
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
