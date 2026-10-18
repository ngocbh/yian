/**
 * The one spelling every message uses to print a `Literal` type, a port of the hand-built
 * half of `tests/engine/test_literal_label.py`. Later: the read-back and `format_type_expr`
 * asserts (with the parser), `test_nested_literal_labels` (with `typeFor`), the two hint
 * tests (with `readTypedefs`) and `test_case_message_prints_the_parenthesized_literal` (with
 * flow loading).
 */

import { describe, expect, test } from "bun:test";

import { canonicalTypeLabel, literalLabel, Type, ValueKind } from "../../src/typesys/values.ts";

function lit(...tags: string[]): Type {
  return new Type({ kind: ValueKind.STRING, tags: new Set(tags) });
}

/** A closed record with one required field `a`. */
function rec(a: Type): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["a", a]]),
    required: new Set(["a"]),
  });
}

describe("literal labels", () => {
  test.each([
    [["no_go", "go"], "Literal(go, no_go)"],
    [["a b", "z", "é"], 'Literal("a b", z, é)'],
    [["True", "x-y", "1a", 'say "hi"'], 'Literal("1a", "True", "say \\"hi\\"", "x-y")'],
    [["a,b"], 'Literal("a,b")'],
    // a NAME token that `str.isidentifier` rejects must be quoted to read back
    [["a²"], 'Literal("a²")'],
  ])("canonical label spells %p in the parenthesized form", (tags, label) => {
    expect(canonicalTypeLabel(lit(...tags))).toBe(label);
  });

  test("both renderers share one spelling", () => {
    expect(literalLabel(["go", "no_go"])).toBe("Literal(go, no_go)");
    expect(canonicalTypeLabel(lit("go", "no_go"))).toBe("Literal(go, no_go)");
  });

  test("prints a name bare and quotes the rest, in the given order", () => {
    expect(literalLabel(["go", "no go", "True"])).toBe('Literal(go, "no go", "True")');
  });

  // A member prints bare only when it is both a NAME token (`[^\W\d]\w*`) and an identifier
  // (`str.isidentifier()`); each spelling is the one Python's `literal_label` gives.
  test.each([
    // a non-ASCII name
    ["é", "Literal(é)"],
    // a letter number starts a name and an identifier
    ["ⅰx", "Literal(ⅰx)"],
    // a NAME token that `str.isidentifier` rejects
    ["²x", 'Literal("²x")'],
    // an identifier that is not a NAME token
    ["a·", 'Literal("a·")'],
    // a constant the parser reserves
    ["None", 'Literal("None")'],
    // not a name at all
    ["1a", 'Literal("1a")'],
    ['say "hi"', 'Literal("say \\"hi\\"")'],
  ])("spells %p as %p", (tag, label) => {
    expect(literalLabel([tag])).toBe(label);
  });

  test("the order key keeps the bracket spelling", () => {
    // By label `{"a":Literal("b c")}` sorts first (`"` < `a`); by the bracket key
    // `{"a":Literal["a","z"]}` does, and Python orders the members that way.
    const variants = [rec(lit("b c")), rec(lit("a", "z"))];
    const union = new Type({ kind: ValueKind.OBJECT, variants });
    expect(canonicalTypeLabel(union)).toBe('Union({"a":Literal(a, z)},{"a":Literal("b c")})');
  });
});
