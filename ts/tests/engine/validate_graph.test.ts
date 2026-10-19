/**
 * Recursive type compatibility, the cycle check and reference head checks: a port of the pure
 * `types_compatible` tests of `tests/engine/test_validate_graph.py`
 * (`test_numeric_domain_compatibility_requires_interval_containment`,
 * `test_enum_compatibility_requires_tag_subset`,
 * `test_numeric_domain_compatibility_recurses_through_records_and_lists`,
 * `test_list_element_compatibility_preserves_nullable_direction`,
 * `test_none_source_is_compatible_exactly_with_none_or_nullable_sinks`,
 * `test_bare_list_compatibility_uses_its_storage_element_contract`,
 * `test_list_element_compatibility_recurses_before_storage_kind_rejection`).
 *
 * TS-only: `rejectCycles` on bare edges (the cycle message and `errors` for a two-node cycle,
 * as conformance `load-core/seed_e02_cycle`; the stuck nodes sorted; edges touching START and
 * END ignored; an acyclic graph passes; an edge naming an unknown node throws, as Python's
 * `KeyError`), `FlowValidationError`'s `errors`, and `classifyPath`'s
 * messages and its `null` for good paths.
 *
 * Later (T3-18): the cycle and CASE handle tests, which build seeds through compose.
 */

import { describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import {
  classifyPath,
  FlowValidationError,
  rejectCycles,
  typesCompatible,
} from "../../src/compile/validation.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

/** Resolve a type expression with no typedefs. */
const t = (text: string): Type => typeFor(text, new Map());

/** A record whose fields are all required. */
function rec(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

describe("recursive type compatibility", () => {
  test.each([
    ["int(2..4)", "int(1..12)", true],
    ["int(1..12)", "int(2..4)", false],
    ["int(1..12)", "int(1..12)", true],
    ["int(-2..4)", "int(0..12)", false],
    ["int(2..13)", "int(1..12)", false],
    ["int(4..4)", "int(1..12)", true],
    ["int", "int(1..12)", false],
    ["int(1..12)", "int", true],
    ["float(0.25..0.75)", "float(0..1)", true],
    ["float(0..1)", "float(0.25..0.75)", false],
    ["float", "float(0..1)", false],
    ["float(0..1)", "float", true],
    ["int(1..2)", "float(0.5..2.5)", true],
    ["int(0..2)", "float(0.5..2.5)", false],
    ["int", "float(0..1)", false],
    ["int(0..1)", "float", true],
    ["float(1..1)", "int(1..1)", false],
    ["Optional(int(1..2))", "int(0..3)", false],
    ["int(1..2)", "Optional(int(0..3))", true],
    ["Optional(int(1..2))", "Optional(float(0..3))", true],
    ["Optional(int)", "Optional(int(0..3))", false],
    ["None", "Optional(int(0..3))", true],
    ["None", "int(0..3)", false],
    // An open end is unbounded on that side.
    ["int(5..)", "int(0..)", true],
    ["int(0..10)", "int(0..)", true],
    ["int(0..10)", "int(..100)", true],
    ["int(0..)", "int(0..)", true],
    ["int(0..)", "int(0..100)", false],
    ["int(..5)", "int(0..)", false],
    ["int", "int(0..)", false],
    ["int(0..)", "int", true],
    ["int(..5)", "int(..9)", true],
    ["int(..9)", "int(..5)", false],
    ["int(1..)", "float(0.5..)", true],
    ["float(0.0..)", "int(0..)", false],
    ["Optional(int(0..))", "Optional(int(-1..))", true],
  ])(
    "numeric domain compatibility requires interval containment: %s into %s",
    (source, sink, ok) => {
      expect(typesCompatible(t(source), t(sink))).toBe(ok);
    },
  );

  // An untagged source cannot prove membership, like an unbounded numeric source.
  test.each([
    ["Literal(a, b)", "Literal(a, b)", true],
    ["Literal(a)", "Literal(a, b)", true],
    ["Literal(a, c)", "Literal(a, b)", false],
    ["str", "Literal(a, b)", false],
    ["Literal(a, b)", "str", true],
    ["Optional(str)", "Optional(Literal(a, b))", false],
    ["list(str)", "list(Literal(a, b))", false],
    ["list(Literal(a))", "list(Literal(a, b))", true],
  ])("enum compatibility requires a tag subset: %s into %s", (source, sink, ok) => {
    expect(typesCompatible(t(source), t(sink))).toBe(ok);
  });

  test.each([
    ["list(int(2..4))", "list(float(1..12))", true],
    ["list(int)", "list(int(1..12))", false],
    ["list(int(0..4))", "list(float(1..12))", false],
  ])(
    "numeric domain compatibility recurses through records and lists: %s into %s",
    (source, sink, ok) => {
      expect(typesCompatible(rec({ values: t(source) }), rec({ values: t(sink) }))).toBe(ok);
    },
  );

  test("list element compatibility preserves the nullable direction", () => {
    // A source list may narrow but cannot widen the sink's element contract.
    const strictElement = Type.scalar(ValueKind.INTEGER);
    const nullableElement = new Type({ kind: ValueKind.INTEGER, nullable: true });
    const strict = new Type({ kind: ValueKind.LIST_INTEGER, element: strictElement });
    const nullable = new Type({ kind: ValueKind.LIST_INTEGER, element: nullableElement });

    expect(typesCompatible(strict, nullable)).toBe(true);
    expect(typesCompatible(nullable, nullable)).toBe(true);
    expect(typesCompatible(nullable, strict)).toBe(false);
  });

  test("a None source is compatible exactly with None or nullable sinks", () => {
    // Static compatibility mirrors runtime acceptance of the singleton None value.
    const none = Type.scalar(ValueKind.NONE);
    const strictString = Type.scalar(ValueKind.STRING);
    const nullableString = new Type({ kind: ValueKind.STRING, nullable: true });

    expect(typesCompatible(none, none)).toBe(true);
    expect(typesCompatible(none, nullableString)).toBe(true);
    expect(typesCompatible(none, strictString)).toBe(false);
    expect(typesCompatible(rec({ note: none }), rec({ note: nullableString }))).toBe(true);
  });

  test("a bare list's compatibility uses its storage element contract", () => {
    // Bare specialized lists stay strict while a bare LIST_ANY stays unconstrained.
    const strictElement = Type.scalar(ValueKind.INTEGER);
    const nullableElement = new Type({ kind: ValueKind.INTEGER, nullable: true });
    const bareInteger = Type.scalar(ValueKind.LIST_INTEGER);
    const strictInteger = new Type({ kind: ValueKind.LIST_INTEGER, element: strictElement });
    const nullableInteger = new Type({ kind: ValueKind.LIST_INTEGER, element: nullableElement });
    const bareAny = Type.scalar(ValueKind.LIST_ANY);
    const refinedAny = new Type({ kind: ValueKind.LIST_ANY, element: strictElement });

    expect(typesCompatible(bareInteger, nullableInteger)).toBe(true);
    expect(typesCompatible(strictInteger, bareInteger)).toBe(true);
    expect(typesCompatible(nullableInteger, bareInteger)).toBe(false);
    expect(typesCompatible(refinedAny, bareAny)).toBe(true);
    expect(typesCompatible(bareAny, refinedAny)).toBe(false);
  });

  test("list element compatibility recurses before storage kind rejection", () => {
    // A list of None values may fill a list whose elements are nullable.
    const source = new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.NONE) });
    const sink = new Type({
      kind: ValueKind.LIST_STRING,
      element: new Type({ kind: ValueKind.STRING, nullable: true }),
    });
    expect(typesCompatible(source, sink)).toBe(true);
  });
});

/** Return bare edges for `from->to` pairs. */
function edges(...pairs: [string, string][]): { from: string; to: string }[] {
  return pairs.map(([from, to]) => ({ from, to }));
}

/** Return the `FlowValidationError` `run` throws. */
function flowError(run: () => unknown): FlowValidationError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(FlowValidationError);
    return error as FlowValidationError;
  }
  throw new Error("expected a FlowValidationError");
}

describe("TS-only: rejectCycles", () => {
  test("a two-node cycle names both nodes", () => {
    const error = flowError(() => rejectCycles(edges(["a", "b"], ["b", "a"]), new Set(["a", "b"])));
    const message = 'flow has a cycle involving ["a","b"]; flows must be acyclic';
    expect(error).toBeInstanceOf(RangeError);
    expect(error.name).toBe("FlowValidationError");
    expect(error.message).toBe(message);
    expect(error.errors).toEqual([message]);
  });

  test("the stuck nodes are sorted, and nodes behind the cycle are stuck too", () => {
    const error = flowError(() =>
      rejectCycles(
        edges(["start", "b9"], ["b9", "b10"], ["b10", "b9"], ["b10", "Z"]),
        new Set(["start", "b9", "b10", "Z"]),
      ),
    );
    expect(error.message).toBe(
      'flow has a cycle involving ["Z","b10","b9"]; flows must be acyclic',
    );
  });

  test("edges touching START and END are ignored", () => {
    expect(() =>
      rejectCycles(
        edges([START_ID, "a"], ["a", "b"], ["b", END_ID], [END_ID, START_ID]),
        new Set(["a", "b"]),
      ),
    ).not.toThrow();
  });

  test("an acyclic graph passes", () => {
    expect(() =>
      rejectCycles(
        edges(["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]),
        new Set(["a", "b", "c", "d"]),
      ),
    ).not.toThrow();
  });

  test.each<[string, string]>([
    ["x", "a"],
    ["a", "x"],
  ])("edge %s->%s naming an unknown node throws", (from, to) => {
    expect(() => rejectCycles(edges([from, to]), new Set(["a"]))).toThrow(
      "edge names unknown node 'x'",
    );
  });
});

describe("TS-only: FlowValidationError", () => {
  test("keeps the errors it is given", () => {
    const error = new FlowValidationError("2 problems", ["x", "y"]);
    expect(error.message).toBe("2 problems");
    expect(error.errors).toEqual(["x", "y"]);
  });
});

describe("TS-only: classifyPath", () => {
  const targets = new Set(["a"]);
  const inputs = new Set(["q"]);

  test.each<[string, string[], string]>([
    ["input", [], "reference ${input} must be input.<name>"],
    ["input.zz", [], "reference ${input.zz} points to unknown flow input 'zz'"],
    [
      "inputs.q",
      [],
      "reference ${inputs.q} uses the retired plural head 'inputs' (did you mean `input`?)",
    ],
    [
      "outputs.a",
      [],
      "reference ${outputs.a} uses the retired plural head 'outputs' " +
        "(did you mean `<node>.output`?)",
    ],
    ["item.x", [], "reference ${item.x} uses unknown namespace 'item'"],
    ["system", [], "reference ${system} uses unknown system ambient '' (only today/now/run_id)"],
    [
      "system.clock",
      [],
      "reference ${system.clock} uses unknown system ambient 'clock' (only today/now/run_id)",
    ],
    [
      "system.now.year",
      [],
      "reference ${system.now.year} reads field 'year' on type datetime, which has no fields",
    ],
    [
      "system.run_id.x",
      [],
      "reference ${system.run_id.x} reads field 'x' on type str, which has no fields",
    ],
    ["a", [], "reference ${a} must be a.output[.path]"],
    [
      "a.out",
      [],
      "reference ${a.out} on node 'a' must be a.output[.field] " +
        "(`.output` is the node-value selector)",
    ],
    ["zz.output", [], "reference ${zz.output} uses unknown namespace 'zz'"],
    ["toString.output", [], "reference ${toString.output} uses unknown namespace 'toString'"],
  ])("%s is refused", (path, extraHeads, message) => {
    expect(classifyPath(path, targets, inputs, extraHeads)).toBe(message);
  });

  test.each<[string, string[]]>([
    ["input.q", []],
    ["input.q.deep", []],
    ["item", ["item"]],
    ["item.x.y", ["item"]],
    ["system.today", []],
    ["system.now", []],
    ["system.run_id", []],
    ["a.output", []],
    ["a.output.x.y", []],
  ])("%s is accepted", (path, extraHeads) => {
    expect(classifyPath(path, targets, inputs, extraHeads)).toBeNull();
  });

  test("extra heads default to none", () => {
    expect(classifyPath("item", targets, inputs)).toBe(
      "reference ${item} uses unknown namespace 'item'",
    );
  });
});
