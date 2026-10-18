/**
 * Host intake, `fromHost`, a port of `tests/engine/test_plain_containers.py`. TS yian reads a
 * host value (a tool or CODE result, an input) through `fromHost` where Python reads a node
 * output into plain containers: it returns a fresh value made of plain arrays and records, and
 * refuses what isn't a yian value. Each row that differs from Python carries a comment naming
 * its rule.
 *
 * Ported: plain values and leaves (a `FileRef` kept; a class instance refused), shared values
 * copied once near the root and deep, the deep chain copied without recursion, an error from
 * the host's own code propagating, and a `TypedValue` (Python copies its payload; refused
 * here). TS-only: the host-intake table (scalars, numbers, `undefined`, lone surrogates,
 * non-plain objects), `__proto__` kept as an own data key, a getter read once, skipped
 * symbol and non-enumerable keys and extra array properties, and a depth-30 diamond copied in
 * linear time.
 *
 * Also here: the cycle row of
 * `test_untyped_output_json.py::test_reject_non_json_locates_the_bad_part`.
 *
 * Not ported: the list, tuple and dict subclass rows read through their own iteration
 * (`HiddenList`, `HiddenDict`, `YieldsOneNan`, the namedtuple, the tuple rows), which are
 * refused here as non-plain objects (one row checks an `Array` subclass); the cycle rows,
 * since intake refuses a container inside itself (one row checks it); and the unhashable-key
 * row (a record key is always a string).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  buildValue,
  FileRef,
  fromHost,
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
      "output at ['my key']: Foo is not a yian value",
      ["my key"],
    ],
    ["an Array subclass", Sub.from([1]), "output: Sub is not a yian value", []],
    ["a function", [() => 1], "output at [0]: function is not a yian value", [0]],
    ["a symbol", { s: Symbol("s") }, "output at s: symbol is not a yian value", ["s"]],
    // Hosts don't build carriers: a TypedValue is refused like any class instance; Python
    // copies its payload.
    ["a TypedValue", { t: buildValue(1n) }, "output at t: TypedValue is not a yian value", ["t"]],
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

  test("a getter is read once", () => {
    let reads = 0;
    const value = {
      get a() {
        reads += 1;
        return reads;
      },
    };
    expect(fromHost(value, "output")).toStrictEqual({ a: 1n });
    expect(reads).toBe(1);
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

  test("a shared value is copied once", () => {
    let reads = 0;
    const shared = {
      get x() {
        reads += 1;
        return 1;
      },
    };
    const out = fromHost([shared, shared], "output") as Value[];
    expect(out[0]).toBe(out[1]);
    expect(out[0]).toStrictEqual({ x: 1n });
    expect(reads).toBe(1);
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

  test("an error a getter throws propagates unchanged", () => {
    const boom = new RangeError("boom");
    const value = {
      s: {
        get x(): unknown {
          throw boom;
        },
      },
    };
    let thrown: unknown;
    try {
      fromHost(value, "output");
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBe(boom);
  });
});
