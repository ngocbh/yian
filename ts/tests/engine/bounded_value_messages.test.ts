/**
 * Messages that show a value or key stay within the shown-value cap (200), a port of the
 * `value_repr` and unknown-record-field tests of `tests/engine/test_bounded_value_messages.py`.
 *
 * Also here: `test_long_ints.py::test_value_repr_of_a_small_int`, and the path renderings of
 * `test_nonfinite_values.py::test_reject_nonfinite_names_where_and_path`. The spelling,
 * placeholder and bounded-rendering rows are TS-only: a message shows a value in its text form
 * (JSON style, JS number spelling), not Python's `repr`. Also TS-only: a test pinning
 * `valueRepr` to `writeAsText`, the text form builtins and templates write, for every value that
 * isn't text.
 *
 * The unknown-field rows check `buildValueAs`;
 * `test_record_sorts_mixed_unknown_keys_by_type_without_crashing` keeps its two long names (a
 * record key is always a string, so the int and broken-repr keys have no TS counterpart). The
 * unknown-input row checks `checkInputNames`.
 *
 * Ported through `loadFlow` and `runFlow`, each CODE body the JavaScript of the
 * `_bounded_codefns.py` function it stands for:
 * `test_too_deep_value_under_long_keys_shows_a_bounded_path`,
 * `test_output_record_with_many_unknown_fields_shows_a_bounded_message`,
 * `test_long_declared_field_keeps_its_origin` and `test_deep_output_is_located` (`deep_list` and
 * `dict_of_deep_list`). Changed: `test_list_hiding_a_deep_list_stores_what_its_iteration_yields`
 * and `test_list_yielding_a_deep_list_fails_the_output_contract` return an `Array` subclass with
 * its own iterator, which fails `contract.output` as "`<Name>` is not a yian value": host value
 * intake reads only plain arrays and records and runs no host iterator (design 3.2.1). Changed
 * the same way: `test_output_holding_a_big_set_shows_a_cut_repr` and
 * `test_deep_frozenset_output_is_not_a_json_value` return a `Set`, which fails `contract.output`
 * as "Set is not a yian value", not Python's cut rendering or "is not a JSON value" (design 3.2.1
 * host intake reads no `Set`).
 *
 * Not ported, runs: `test_too_deep_value_under_a_non_text_key_shows_a_bounded_message` (a
 * JavaScript record key is always text) and the `deep_tuple` rows of `test_deep_output_is_located`
 * (no tuples; an array is the only list).
 *
 * Not ported: the long-int placeholders (`<int with N digits>`; an int is a `bigint` with no
 * digit limit, and a message shows its digits cut at the cap, which `long_ints.test.ts` checks as
 * `test_long_ints.py::test_value_repr_shows_a_digit_count_for_a_long_int`), the set rendering (no
 * set is a yian value) and `_sort_repr` (Python-only: it orders
 * names whose cut renderings are equal, which TS sorts uncut).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import process from "node:process";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, DefaultsError, forget, install, reload } from "../../src/defaults.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { checkInputNames } from "../../src/nodes/binding.ts";
import { outputContractOrigin, Parameter } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import {
  buildValueAs,
  FileRef,
  renderValuePath,
  shownText,
  Type,
  TypeCheckError,
  type Value,
  ValueKind,
  valueRepr,
  writeAsText,
} from "../../src/typesys/values.ts";

const CAP = 200;
const MESSAGE_BOUND = 1200;

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

class Foo {}

describe("valueRepr", () => {
  // A message shows text JSON-quoted, so the cut rendering starts with `"`, not Python's `'`.
  test("cuts a long string to the cap", () => {
    expect(valueRepr("x".repeat(500))).toBe('"' + "x".repeat(CAP - 2) + "…");
  });

  test("keeps a rendering exactly at the cap", () => {
    const value = "x".repeat(CAP - 2);
    expect(valueRepr(value)).toBe(`"${value}"`);
    expect(valueRepr(value)).toHaveLength(CAP);
  });

  test("follows the configured cap", () => {
    reload((name) => (name === "YIAN_MAX_SHOWN_VALUE_CHARS" ? "10" : undefined));
    expect(valueRepr("abcdefghijklmnop")).toBe('"abcdefgh…');
  });

  test("throws a bad defaults variable rather than showing a placeholder", () => {
    const variable = "YIAN_MAX_SHOWN_VALUE_CHARS";
    const saved = process.env[variable];
    process.env[variable] = "1";
    forget();
    try {
      expect(() => valueRepr("abc")).toThrow(DefaultsError);
    } finally {
      if (saved === undefined) delete process.env[variable];
      else process.env[variable] = saved;
    }
  });

  test("shows a small int as its digits", () => {
    expect(valueRepr(5n)).toBe("5");
  });

  test.each<[string, unknown, string]>([
    ["true", true, "true"],
    ["null", null, "null"],
    ["a float", 2.5, "2.5"],
    ["a whole float", 2, "2"],
    ["infinity", Infinity, "Infinity"],
    ["a long int", 12345678901234567890123n, "12345678901234567890123"],
    ["a list", ["a"], '["a"]'],
    ["a record", { k: 1n }, '{"k":1}'],
    ["a file", new FileRef({ uri: "s3://b/k" }), '{"uri":"s3://b/k","mime":null,"name":null}'],
    ["JS key order", { b: 1n, 2: 2n }, '{"2":2,"b":1}'],
    ["a null-prototype record", Object.assign(Object.create(null), { k: "v" }), '{"k":"v"}'],
    ["a Date", new Date(0), "<Date object>"],
    ["a Map", new Map(), "<Map object>"],
    ["a class instance", new Foo(), "<Foo object>"],
    // The prototype's own `constructor` names it, not one the value fakes.
    ["a faked constructor", Object.assign(new Foo(), { constructor: Map }), "<Foo object>"],
    ["a symbol", Symbol("s"), "<symbol object>"],
    ["undefined inside a list", [1n, undefined], "[1,<undefined object>]"],
  ])("spells %s", (_id, value, expected) => {
    expect(valueRepr(value)).toBe(expected);
  });

  test("gives a placeholder when a getter throws", () => {
    const value = {
      get a(): never {
        throw new Error("boom");
      },
    };
    expect(valueRepr(value)).toBe("<Object object>");
  });

  test("renders a cyclic and a very deep value within the cap without recursion", () => {
    const cyclic: unknown[] = [];
    cyclic.push(cyclic);
    expect(valueRepr(cyclic)).toBe("[".repeat(CAP - 1) + "…");

    let deep: unknown = "leaf";
    for (let i = 0; i < 100_000; i++) deep = { a: deep };
    expect(valueRepr(deep)).toBe('{"a":'.repeat(39) + '{"a"…');
  });

  test("renders a huge list in time bounded by the cap", () => {
    const huge: unknown[] = [];
    huge.length = 2 ** 32 - 1;
    const shown = valueRepr(huge);
    expect(shown).toHaveLength(CAP);
    expect(shown.startsWith("[<undefined object>,<undefined object>,")).toBe(true);
  });

  test("a message shows a value that isn't text in its text form", () => {
    const values: Value[] = [
      true,
      12n,
      10n ** 25n,
      1.5,
      1,
      Infinity,
      -Infinity,
      null,
      [1n, "a", null],
      { b: 1n, "2": [true] },
      new FileRef({ uri: "s3://b/k", mime: "text/plain" }),
    ];
    for (const value of values) expect(valueRepr(value)).toBe(writeAsText(value));
  });
});

describe("shownText", () => {
  test("counts code points and never splits a surrogate pair", () => {
    expect(shownText("😀".repeat(CAP))).toBe("😀".repeat(CAP));
    expect(shownText("😀".repeat(CAP + 1))).toBe("😀".repeat(CAP - 1) + "…");
  });
});

describe("renderValuePath", () => {
  test.each<[string, (string | number)[], string]>([
    ["a list index", ["a", 1], "a[1]"],
    ["a field after an index", ["a", 1, "b"], "a[1].b"],
    ["a key that isn't an identifier", [0, "my key"], '[0]["my key"]'],
  ])("renders %s", (_id, path, expected) => {
    expect(renderValuePath(path)).toBe(expected);
  });

  test("cuts each key to the cap", () => {
    expect(renderValuePath(["x".repeat(301)])).toBe("x".repeat(CAP - 1) + "…");
    expect(renderValuePath([" ".repeat(301)])).toBe('["' + " ".repeat(CAP - 2) + "…]");
  });

  test("keeps the start and end of a path longer than four times the cap", () => {
    const shown = renderValuePath(Array.from({ length: 1000 }, (_, i) => `k${i}`));
    expect([...shown]).toHaveLength(4 * CAP);
    expect(shown.startsWith("k0.k1.k2.")).toBe(true);
    expect(shown.endsWith(".k998.k999")).toBe(true);
    expect(shown.slice(2 * CAP, 2 * CAP + 1)).toBe("…");
  });
});

describe("unknown record fields", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["a", Type.scalar(ValueKind.INTEGER)]]),
    required: new Set(["a"]),
  });

  /** Return the `TypeCheckError` `buildValueAs(record, value)` throws. */
  function buildError(value: unknown): TypeCheckError {
    try {
      buildValueAs(record, value);
    } catch (error) {
      expect(error).toBeInstanceOf(TypeCheckError);
      return error as TypeCheckError;
    }
    throw new Error("buildValueAs did not throw");
  }

  test("a record lists many unknown fields within the cap", () => {
    const value: Record<string, bigint> = {};
    for (let i = 0; i < 200_000; i++) value[`key${i}`] = BigInt(i);
    value["a"] = 1n;
    const error = buildError(value);
    expect(error.message.startsWith("record has unknown fields: [")).toBe(true);
    expect(error.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
    expect(error.path).toEqual(["key0"]);
  });

  test("a record sorts long unknown names past the cap", () => {
    // Equal up to the cap, so a sort on the cut text could not order them.
    const later = "x".repeat(300) + "b";
    const earlier = "x".repeat(300) + "a";
    const error = buildError({ a: 1n, [later]: 1n, [earlier]: 2n });
    expect(error.message.startsWith('record has unknown fields: ["xxx')).toBe(true);
    expect(error.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
    expect(error.path).toEqual([earlier]);
  });
});

test("unknown inputs list within the cap", () => {
  const raw: Record<string, bigint> = {};
  for (let i = 0; i < 200_000; i++) raw[`name${i}`] = BigInt(i);
  let message = "";
  try {
    checkInputNames([new Parameter("topic")], raw);
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    message = (error as RangeError).message;
  }
  expect(message.startsWith('unknown inputs "name0", "name1", ')).toBe(true);
  expect(message.endsWith("; declared inputs: topic")).toBe(true);
  expect(message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
});

// ---------- runs ----------

/** Nest `leaf` in `levels` levels of `wrap`, as JavaScript a CODE body runs. */
const NEST =
  "const nest = (levels, wrap, leaf = 1) => { let v = leaf; for (let i = 0; i < levels; i++) v = wrap(v); return v; };";

/** The CODE bodies Python's `tests/engine/_bounded_codefns.py` functions stand for. */
const FNS: Record<string, string> = {
  long_str_keys: `${NEST} return {s: nest(200, (v) => ({["k".repeat(10000)]: v}))};`,
  big_set: "const s = new Set(); for (let i = 0; i < 200000; i++) s.add(i); return {s};",
  many_unknown_fields: "const v = {}; for (let i = 0; i < 200000; i++) v[`key${i}`] = i; v.a = 1; return v;",
  wrong_long_field: `return {["f".repeat(250)]: "not an int", g: 1};`,
  deep_list: `${NEST} return nest(200000, (v) => [v]);`,
  dict_of_deep_list: `${NEST} return {s: nest(200000, (v) => [v])};`,
  deep_frozenset: `${NEST} return nest(200000, (v) => new Set([v]));`,
  hidden_deep_list:
    `${NEST} class HidingList extends Array { *[Symbol.iterator]() {} }` +
    " const s = new HidingList(); s.push(nest(200000, (v) => [v])); return {s};",
  yielded_deep_list:
    `${NEST} const deep = [nest(200000, (v) => [v])];` +
    " class YieldingList extends Array { *[Symbol.iterator]() { yield* deep; } } return {s: new YieldingList()};",
};

const LONG_FIELD = "f".repeat(250);

/** A flow whose one CODE node runs `fn`'s body with an optional declared output. */
function bounded(fn: string, output: string | null = null): string {
  return (
    "id: bounded\nname: bounded\nnodes:\n  b:\n    kind: code\n" +
    `    code: ${JSON.stringify(FNS[fn])}\n` +
    "    input:\n      topic: x\n" +
    (output === null ? "" : `    output: ${output}\n`) +
    "output: ${b.output}\n"
  );
}

/** Run the flow for `fn` and return its failure, which must be a `contract.output`. */
async function failureOf(fn: string, output: string | null = null): Promise<Failure> {
  const result = await runFlow(await loadFlow(bounded(fn, output)), {}, { numWorkers: 0 });
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("contract.output");
  return result.failure!;
}

// Host value intake reads no `Set` (design 3.2.1), so no cut rendering of one is shown.
test.each([null, "Any"])("an output holding a big set is not a yian value: %p", async (output) => {
  const failure = await failureOf("big_set", output);
  expect(failure.nodeId).toBe("b");
  expect(failure.message).toBe("output at s: Set is not a yian value");
});

test.each([null, "Any", "{s: Any}"])("a too-deep value under long keys shows a bounded path: %p", async (output) => {
  const failure = await failureOf("long_str_keys", output);
  expect(failure.message).toContain("value nests deeper");
  expect(failure.message).toContain("…");
  expect(failure.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
});

test("an output record with many unknown fields shows a bounded message", async () => {
  const failure = await failureOf("many_unknown_fields", "{a: int}");
  expect(failure.message).toContain("unknown fields");
  expect(failure.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
});

test("a long declared field keeps its origin", async () => {
  const flow = await loadFlow(bounded("wrong_long_field", `{${LONG_FIELD}: int, g: int}`));
  const contract = flow.compiled.nodes.get("b")!.outputContract;
  const fieldOrigin = outputContractOrigin(contract, [LONG_FIELD]);
  expect(fieldOrigin).not.toBeNull();
  expect(fieldOrigin).not.toEqual(outputContractOrigin(contract, []));

  const result = await runFlow(flow, {}, { numWorkers: 0 });
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("contract.output");
  expect(result.failure?.trace).toEqual([fieldOrigin!]);
});

test.each(["deep_list", "dict_of_deep_list"].flatMap((fn) => [null, "Any"].map((output) => [fn, output] as const)))(
  "a deep output is located: %s, %p",
  async (fn, output) => {
    const failure = await failureOf(fn, output);
    expect(failure.message).toContain("value nests deeper than 100 levels");
    expect(failure.message).toContain(" at ");
    expect(failure.message).not.toContain("canonical data nesting");
  },
);

test.each([null, "Any", "list(Any)"])("a deep set output is not a yian value: %p", async (output) => {
  // Refused as a `Set` before its nesting is walked (design 3.2.1).
  const failure = await failureOf("deep_frozenset", output);
  expect(failure.nodeId).toBe("b");
  expect(failure.message).toBe("output: Set is not a yian value");
  expect(failure.message).not.toContain("canonical data nesting");
});

test.each([
  ["hiding", "hidden_deep_list", "HidingList"],
  ["yielding", "yielded_deep_list", "YieldingList"],
])("a list %s a deep list is not a yian value", async (_how, fn, name) => {
  // Host value intake reads only plain arrays and never runs a host iterator (design 3.2.1).
  const failure = await failureOf(fn);
  expect(failure.nodeId).toBe("b");
  expect(failure.message).toBe(`output at s: ${name} is not a yian value`);
});
