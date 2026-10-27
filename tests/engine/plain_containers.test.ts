/**
 * Host intake, `fromHost`, a port of `tests/engine/test_plain_containers.py`, and its reverse,
 * `toHost`. TS yian reads a host value (a tool or CODE result, an input) through `fromHost`
 * where Python reads a node output into plain containers: it returns a fresh value made of
 * plain arrays and records, and refuses what isn't a yian value. Each row that differs from
 * Python carries a comment naming its rule.
 *
 * Ported: plain values and leaves (a `FileRef` kept; a class instance refused), shared values
 * copied once near the root and deep, the deep chain copied without recursion, and a
 * `TypedValue` (Python copies its payload; refused here). TS-only: the host-intake table
 * (scalars, numbers, `undefined`, lone surrogates, non-plain objects, an array whose prototype
 * was swapped), a record inside itself refused, `__proto__` kept as an own data key, an accessor
 * property (a getter or a setter) and a `Proxy` refused without running host code, skipped
 * symbol and non-enumerable keys (accessors included) and extra array properties, and a
 * depth-30 diamond copied in linear time.
 *
 * Also here: the cycle row of
 * `test_untyped_output_json.py::test_reject_non_json_locates_the_bad_part`.
 *
 * The `toHost` rows (TS-only; Python hands a host a deep copy of its own values): an int within
 * ±(2^53−1) handed over as a `number` and beyond as a `bigint`, floats kept, fresh arrays and
 * records, `__proto__` kept as an own data key, a `FileRef` kept, sharing kept, a 100 000-deep
 * list without recursion, and the `fromHost` round trip.
 *
 * Not ported: the list, tuple and dict subclass rows read through their own iteration
 * (`HiddenList`, `HiddenDict`, `YieldsOneNan`, the namedtuple, the tuple rows), which are
 * refused here as non-plain objects (one row checks an `Array` subclass); the cycle rows,
 * since intake refuses a container inside itself (an array row and a record row check it);
 * the unhashable-key row (a record key is always a string); and
 * `test_iteration_error_propagates`, since intake runs no host code (it refuses getters and
 * Proxies instead of reading them).
 *
 * Not ported from `tests/engine/test_hidden_container_outputs.py` (no TS file): its rows run
 * Python CODE returning list, tuple and dict subclasses whose iteration differs from their
 * storage, and sets; TS host intake takes only plain arrays and records and refuses the rest (the
 * non-plain-object, `Array` subclass and cycle rows here check it):
 * `test_stored_output_is_what_iteration_yields`,
 * `test_a_cycle_through_a_subclass_reads_it_once`, `test_a_yielded_nan_is_located`,
 * `test_a_yielded_set_is_located`, `test_a_yielded_text_key_is_required`,
 * `test_a_yielded_unhashable_key_is_located` and
 * `test_ordinary_subclasses_store_like_their_plain_equivalents`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  buildValue,
  FileRef,
  fromHost,
  jsTypeName,
  toHost,
  TypeCheckError,
  type Value,
  type ValueRecord,
} from "../../src/typesys/values.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

class Foo {}
class Sub extends Array<unknown> {}

/** Return the `TypeCheckError` `fromHost(value, where)` throws. */
function intakeError(value: unknown, where = "output"): TypeCheckError {
  try {
    fromHost(value, where);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("fromHost did not throw");
}

/** Return Proxy traps that all throw, so a test fails if intake runs one. */
function throwingTraps(): ProxyHandler<object> {
  const boom = (): never => {
    throw new RangeError("boom");
  };
  return {
    get: boom,
    ownKeys: boom,
    getPrototypeOf: boom,
    getOwnPropertyDescriptor: boom,
    has: boom,
  };
}

describe("fromHost", () => {
  test("a plain value becomes an equal fresh value", () => {
    // Host intake returns a fresh value; Python returns a plain value itself.
    const value = { a: [1n, [2.5, "x"]], b: null, c: { d: true } };
    const out = fromHost(value, "output") as ValueRecord;
    expect(out).toStrictEqual(value);
    expect(out).not.toBe(value);
    expect(out["a"]).not.toBe(value.a);
  });

  test("a FileRef is kept", () => {
    const file = new FileRef({ uri: "file:///x" });
    expect(fromHost(file, "output")).toBe(file);
    expect((fromHost({ f: file }, "output") as ValueRecord)["f"]).toBe(file);
  });

  test.each<[string, unknown, Value]>([
    ["null", null, null],
    ["true", true, true],
    ["text", "x", "x"],
    ["a bigint", 10n ** 400n, 10n ** 400n],
    ["a whole number", 3, 3n],
    ["a fraction", 2.5, 2.5],
  ])("a scalar is taken as a yian value: %s", (_id, value, expected) => {
    expect(fromHost(value, "output")).toBe(expected);
  });

  test("a null-prototype record becomes a record", () => {
    const value = Object.assign(Object.create(null) as object, { a: 1n });
    expect(fromHost(value, "output")).toStrictEqual({ a: 1n });
  });

  test.each<[string, unknown, string, (string | number)[]]>([
    ["undefined at the root", undefined, "output: undefined is not a yian value; use null", []],
    [
      "undefined in a record",
      { a: { b: undefined } },
      "output at a.b: undefined is not a yian value; use null",
      ["a", "b"],
    ],
    ["a list hole", [1, , 2], "output at [1]: undefined is not a yian value; use null", [1]],
    ["NaN", { a: [0, NaN] }, "output at a[1]: NaN is not a yian value", ["a", 1]],
    [
      "text holding a lone surrogate",
      ["\ud800"],
      "output at [0]: text holding a lone surrogate is not a yian value",
      [0],
    ],
    [
      "a key holding a lone surrogate",
      { a: { "\udc00": 1 } },
      "output at a: text holding a lone surrogate is not a yian value",
      ["a"],
    ],
    ["a Date", { d: new Date(0) }, "output at d: Date is not a yian value", ["d"]],
    ["a Map", [new Map()], "output at [0]: Map is not a yian value", [0]],
    ["a Set", new Set(), "output: Set is not a yian value", []],
    [
      "a class instance",
      { "my key": new Foo() },
      'output at ["my key"]: Foo is not a yian value',
      ["my key"],
    ],
    ["an Array subclass", Sub.from([1]), "output: Sub is not a yian value", []],
    // An array is never a record, whatever its prototype.
    [
      "an array whose prototype was swapped",
      Object.setPrototypeOf([1], Object.prototype),
      "output: Object is not a yian value",
      [],
    ],
    ["a function", [() => 1], "output at [0]: function is not a yian value", [0]],
    ["a symbol", { s: Symbol("s") }, "output at s: symbol is not a yian value", ["s"]],
    // Hosts don't build carriers: a TypedValue is refused like any class instance; Python
    // copies its payload.
    ["a TypedValue", { t: buildValue(1n) }, "output at t: TypedValue is not a yian value", ["t"]],
    // A FileRef forged from the prototype never ran the constructor's checks.
    [
      "a forged FileRef",
      { f: Object.create(FileRef.prototype) as unknown },
      "output at f: FileRef is not a yian value",
      ["f"],
    ],
  ])("refuses %s", (_id, value, message, path) => {
    const error = intakeError(value);
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });

  // From test_untyped_output_json.py::test_reject_non_json_locates_the_bad_part (the cycle row).
  test("refuses a container inside itself at the entry holding it", () => {
    // Intake refuses a cycle; Python's plain-container read keeps it.
    const value: unknown[] = [];
    value.push(value);
    const error = intakeError(value);
    expect(error.message).toBe("output at [0]: value contains itself");
    expect(error.path).toEqual([0]);
  });

  test("refuses a record inside itself at the entry holding it", () => {
    // Intake refuses a cycle; Python's plain-container read keeps it.
    const inner: Record<string, unknown> = {};
    const value = { a: inner };
    inner["b"] = value;
    const error = intakeError(value);
    expect(error.message).toBe("output at a.b: value contains itself");
    expect(error.path).toEqual(["a", "b"]);
  });

  test("a __proto__ key stays an own data key and no prototype changes", () => {
    const value = JSON.parse('{"__proto__": {"polluted": 1}, "a": 2}') as object;
    const out = fromHost(value, "output") as ValueRecord;
    expect(Object.getPrototypeOf(out)).toBe(Object.prototype);
    expect(Object.keys(out)).toEqual(["__proto__", "a"]);
    const own = Object.getOwnPropertyDescriptor(out, "__proto__");
    expect(own?.value).toStrictEqual({ polluted: 1n });
    expect((out as { polluted?: unknown }).polluted).toBeUndefined();
    expect(({} as { polluted?: unknown }).polluted).toBeUndefined();
  });

  // Intake reads only own data properties, so no host code runs during the check.
  test("a getter is refused without being run", () => {
    let reads = 0;
    const value = {
      get a() {
        reads += 1;
        return 1;
      },
    };
    const error = intakeError(value);
    expect(error.message).toBe("output at a: accessor property is not a yian value");
    expect(error.path).toEqual(["a"]);
    expect(reads).toBe(0);
  });

  test.each<[string, unknown, string, (string | number)[]]>([
    [
      "a setter-only property",
      Object.defineProperty({}, "a", { set() {}, enumerable: true }),
      "output at a: accessor property is not a yian value",
      ["a"],
    ],
    [
      "an accessor on an array index",
      Object.defineProperty([0], 0, { get: () => 1 }),
      "output at [0]: accessor property is not a yian value",
      [0],
    ],
    [
      "an accessor in a record inside an array",
      [Object.defineProperty({}, "x", { get: () => 1, enumerable: true })],
      "output at [0].x: accessor property is not a yian value",
      [0, "x"],
    ],
  ])("refuses %s", (_id, value, message, path) => {
    const error = intakeError(value);
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });

  test("a getter that throws is refused, not run", () => {
    const value = {
      s: {
        get x(): unknown {
          throw new RangeError("boom");
        },
      },
    };
    const error = intakeError(value);
    expect(error.message).toBe("output at s.x: accessor property is not a yian value");
    expect(error.path).toEqual(["s", "x"]);
  });

  test("symbol and non-enumerable keys and extra array properties are skipped", () => {
    // Like JSON.stringify, these are not part of the value.
    const record: Record<string | symbol, unknown> = { a: 1 };
    record[Symbol("s")] = 2;
    Object.defineProperty(record, "hidden", { value: 3, enumerable: false });
    const list = Object.assign([1], { extra: 2 });
    expect(fromHost(record, "output")).toStrictEqual({ a: 1n });
    expect(fromHost(list, "output")).toStrictEqual([1n]);
  });

  test("a symbol-keyed or non-enumerable accessor is skipped", () => {
    // Those keys aren't part of the value, so their accessors are never read.
    const boom = (): never => {
      throw new RangeError("boom");
    };
    const record = { a: 1 };
    Object.defineProperty(record, Symbol("s"), { get: boom, enumerable: true });
    Object.defineProperty(record, "hidden", { get: boom, enumerable: false });
    expect(fromHost(record, "output")).toStrictEqual({ a: 1n });
  });

  test("a shared value is copied once", () => {
    const shared = { x: 1 };
    const out = fromHost([shared, shared], "output") as Value[];
    expect(out[0]).toBe(out[1]);
    expect(out[0]).toStrictEqual({ x: 1n });
  });

  test("a value shared near the root and deep is copied once", () => {
    const shared = [1];
    let deep: unknown = shared;
    for (let i = 0; i < 150; i++) deep = [deep];
    const out = fromHost([shared, deep], "output") as Value[];
    let inner = out[1] as Value;
    for (let i = 0; i < 150; i++) inner = (inner as Value[])[0]!;
    expect(inner).toBe(out[0]!);
  });

  test("a depth-30 diamond is copied in linear time", () => {
    let node: unknown = 1;
    for (let i = 0; i < 30; i++) node = { l: node, r: node };
    const start = performance.now();
    let out = fromHost(node, "output");
    expect(performance.now() - start).toBeLessThan(2000);
    for (let i = 0; i < 30; i++) {
      const record = out as ValueRecord;
      expect(record["l"]).toBe(record["r"]!);
      out = record["l"]!;
    }
    expect(out).toBe(1n);
  });

  test("a 100 000-deep array is copied without recursion", () => {
    let deep: unknown = [];
    for (let i = 0; i < 100_000; i++) deep = [deep];
    let node = fromHost(deep, "output") as Value[];
    let depth = 0;
    while (node.length > 0) {
      node = node[0] as Value[];
      depth += 1;
    }
    expect(depth).toBe(100_000);
  });

  test("a Proxy is refused without running a trap", () => {
    const error = intakeError({ s: new Proxy({}, throwingTraps()) });
    expect(error.message).toBe("output at s: Proxy is not a yian value");
    expect(error.path).toEqual(["s"]);
  });

  test("an object whose prototype is a Proxy is named object, running no trap", () => {
    const value = { a: Object.create(new Proxy({}, throwingTraps())) as unknown };
    const error = intakeError(value);
    expect(error.message).toBe("output at a: object is not a yian value");
    expect(error.path).toEqual(["a"]);
  });

  test("an instance whose constructor is a Proxy is named object, running no trap", () => {
    function F(): void {}
    F.prototype.constructor = new Proxy(() => 1, throwingTraps());
    const value = { a: new (F as unknown as new () => object)() };
    const error = intakeError(value);
    expect(error.message).toBe("output at a: object is not a yian value");
    expect(error.path).toEqual(["a"]);
  });

  test("jsTypeName names a Proxy without running a trap", () => {
    expect(jsTypeName(new Proxy({}, throwingTraps()))).toBe("Proxy");
  });

  test.each<[string, unknown, string, (string | number)[]]>([
    [
      "a Proxy over an array",
      [new Proxy([1], {})],
      "output at [0]: Proxy is not a yian value",
      [0],
    ],
    ["a top-level Proxy", new Proxy({ a: 1 }, {}), "output: Proxy is not a yian value", []],
    [
      "a Proxy over a function",
      { f: new Proxy(() => 1, {}) },
      "output at f: Proxy is not a yian value",
      ["f"],
    ],
  ])("refuses %s", (_id, value, message, path) => {
    const error = intakeError(value);
    expect(error.message).toBe(message);
    expect(error.path).toEqual(path);
  });
});

describe("toHost", () => {
  test.each<[string, Value, unknown]>([
    ["a small int", 1n, 1],
    ["the largest safe int", 2n ** 53n - 1n, 2 ** 53 - 1],
    ["the smallest safe int", -(2n ** 53n - 1n), -(2 ** 53 - 1)],
    ["an int just beyond 2^53-1", 2n ** 53n, 2n ** 53n],
    ["an int just beyond -(2^53-1)", -(2n ** 53n), -(2n ** 53n)],
    ["a float", 2.5, 2.5],
    ["Infinity", Infinity, Infinity],
    ["-Infinity", -Infinity, -Infinity],
    ["text", "x", "x"],
    ["true", true, true],
    ["null", null, null],
  ])("a scalar is handed over: %s", (_id, value, expected) => {
    expect(toHost(value)).toBe(expected);
  });

  test("arrays and records are converted into fresh copies", () => {
    const value: ValueRecord = { a: [1n, [2.5, "x"]], b: { c: 2n ** 60n } };
    const out = toHost(value) as { a: unknown[]; b: Record<string, unknown> };
    expect(out).toStrictEqual({ a: [1, [2.5, "x"]], b: { c: 2n ** 60n } });
    expect(out).not.toBe(value);
    out.a.push(3);
    (out.a[1] as unknown[]).push(4);
    out.b["d"] = 5;
    expect(value).toStrictEqual({ a: [1n, [2.5, "x"]], b: { c: 2n ** 60n } });
  });

  test("a __proto__ key stays an own data key and no prototype changes", () => {
    const value = fromHost(JSON.parse('{"__proto__": {"polluted": 1}, "a": 2}'), "x");
    const out = toHost(value) as object;
    expect(Object.getPrototypeOf(out)).toBe(Object.prototype);
    expect(Object.keys(out)).toEqual(["__proto__", "a"]);
    expect(Object.getOwnPropertyDescriptor(out, "__proto__")?.value).toStrictEqual({ polluted: 1 });
    expect((out as { polluted?: unknown }).polluted).toBeUndefined();
  });

  test("a FileRef is the same object", () => {
    const file = new FileRef({ uri: "file:///x" });
    expect(toHost(file)).toBe(file);
    expect((toHost([file]) as unknown[])[0]).toBe(file);
  });

  test("a shared value is copied once", () => {
    const shared: Value[] = [1n];
    const out = toHost({ l: shared, r: [shared] }) as { l: unknown; r: unknown[] };
    expect(out.l).toBe(out.r[0]);
    expect(out.l).toStrictEqual([1]);
  });

  test("a 100 000-deep list converts without recursion", () => {
    let deep: Value = [];
    for (let i = 0; i < 100_000; i++) deep = [deep];
    let node = toHost(deep) as unknown[];
    let depth = 0;
    while (node.length > 0) {
      node = node[0] as unknown[];
      depth += 1;
    }
    expect(depth).toBe(100_000);
  });

  test("fromHost reads toHost's result back as the same value", () => {
    const value: ValueRecord = { a: [1n, 2.5, -(2n ** 70n)], b: { c: 0n, d: "x", e: null } };
    expect(fromHost(toHost(value), "x")).toStrictEqual(value);
  });
});
