/**
 * The one spelling every message uses to print a `Literal` type, a port of
 * `tests/engine/test_literal_label.py`. `test_case_message_prints_the_parenthesized_literal`
 * assembles its flow (`assembleText`), where Python loads it; its harvested case is pinned at the
 * end.
 */

import { describe, expect, test } from "bun:test";

import { formatTypeExpr, parseType, readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  literalLabel,
  Type,
  TypeCheckError,
  typeEquals,
  ValueKind,
} from "../../src/typesys/values.ts";
import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

/** Return the message of the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Readonly<Record<string, unknown>>): string {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error("readTypedefs did not throw");
}

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
    // the label reads back
    expect(typeEquals(typeFor(label, new Map()), lit(...tags))).toBe(true);
  });

  test("both renderers share one spelling", () => {
    expect(formatTypeExpr({ kind: "enum", tags: ["go", "no_go"] })).toBe("Literal(go, no_go)");
    expect(literalLabel(["go", "no_go"])).toBe("Literal(go, no_go)");
    expect(canonicalTypeLabel(lit("go", "no_go"))).toBe("Literal(go, no_go)");
    // authored order kept
    expect(formatTypeExpr(parseType("Literal(b, a)"))).toBe("Literal(b, a)");
  });

  test("nested literal labels", () => {
    expect(canonicalTypeLabel(typeFor("Optional(list(Literal(go, no_go)))", new Map()))).toBe(
      "Optional(list(Literal(go, no_go)))",
    );
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

  test("the tag-only sequence hint spells a parseable Literal", () => {
    expect(typedefsError({ Mode: ["a b", "True"] })).toBe(
      "typedef 'Mode': a tag-only union must be a Literal(...) enum, not a sequence " +
        '(e.g. Mode: Literal("a b", "True"))',
    );
    expect(typeEquals(typeFor('Literal("a b", "True")', new Map()), lit("a b", "True"))).toBe(
      true,
    );
  });

  test("the union overlap hint spells a parseable Literal", () => {
    const message = typedefsError({
      Buy: { qty: "int" },
      Sell: { qty: "int" },
      Order: "Union(Buy, Sell)",
    });
    expect(message.endsWith("e.g. kind: Literal(buy)")).toBe(true);
    expect(typeEquals(typeFor("Literal(buy)", new Map()), lit("buy"))).toBe(true);
  });

  test("the order key keeps the bracket spelling", () => {
    // By label `{"a":Literal("b c")}` sorts first (`"` < `a`); by the bracket key
    // `{"a":Literal["a","z"]}` does, and Python orders the members that way.
    const variants = [rec(lit("b c")), rec(lit("a", "z"))];
    const union = new Type({ kind: ValueKind.OBJECT, variants });
    expect(canonicalTypeLabel(union)).toBe('Union({"a":Literal(a, z)},{"a":Literal("b c")})');
  });
});

test("case message prints the parenthesized literal", () => {
  const flow = `id: f
name: f
input:
  s: !Verdict
typedefs:
  Verdict: Literal(go, no_go)
nodes:
  route:
    kind: case
    on: \${input.s}
    cases:
      - when: nogo
        then: a
    else: a
  a:
    kind: code
    code: return 1
output: \${a.output}
`;

  expect(loadError(() => assembleText(flow)).message).toContain("(`Literal(go, no_go)`); use one of: go, no_go");
});

// ---------- harvested conformance cases ----------

test("assembled: load-core/engine_literal_label__case_message_prints_the_parenthesized_literal", () => {
  runCase({ id: "load-core/engine_literal_label__case_message_prints_the_parenthesized_literal", call: assembled });
});
