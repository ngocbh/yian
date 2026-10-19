/**
 * Reading a slot's type through its YAML tag and spelling tags for messages: a port of the
 * `read_tag_type` and `suggest_tag` tests in `tests/compose/test_tags.py`.
 *
 * TS-only: direct rows for `splitTypedText`, `oldTypedText`, `spellTypeText`,
 * `closeTypeName`, `suggestedLiteral` (a literal that doesn't parse is kept as written, where
 * Python reads an unbound name), `yamlQuoted`, `pathLabel`, `hasSpan` and `namedTypeLabel`,
 * which Python tests only through `load_flow`; and the fragments of conformance messages these
 * helpers produce, each the exact text in its case.
 *
 * Later (the next task): the `read_tagged_value` tests.
 */

import { describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import {
  anySourceHint,
  closeTypeName,
  hasSpan,
  namedTypeLabel,
  oldFormSuggestion,
  oldTypedText,
  pathLabel,
  readTagType,
  spellTypeText,
  splitTypedText,
  suggestedLiteral,
  suggestTag,
  yamlQuoted,
} from "../../src/compose/helpers/tags.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

const REG = readTypedefs({
  Buy: { action: "Literal(buy)", conviction: "float" },
  Sell: { action: "Literal(sell)", conviction: "float" },
  Decision: "Union(Buy, Sell)",
  Order: { decision: "Buy", qty: "int" },
});

const EMPTY: TypeRegistry = new Map();

const rt = (tag: string): Type =>
  readTagType(tag, REG, { label: "x", diagnostics: new LoadDiagnostics() });

/** Return the `LoadError` reading `tag` throws. */
function rejected(tag: string): LoadError {
  try {
    rt(tag);
  } catch (error) {
    if (error instanceof LoadError) return error;
    throw error;
  }
  throw new Error(`expected ${tag} to be rejected`);
}

const t = (text: string, registry: TypeRegistry = REG): Type => typeFor(text, registry);

describe("readTagType", () => {
  test("reads simple and one-argument tags", () => {
    expect(rt("int").kind).toBe(ValueKind.INTEGER);
    expect(rt("Optional(str)").nullable).toBe(true);
    expect(rt("int(1..12)").bounds).toEqual([1n, 12n]);
    expect(rt("Decision").variants).toBeTruthy();
    expect(rt("list(Buy)").element?.fields).toBeTruthy();
  });

  test.each([
    ["Union(Buy,Sell)", "name it in `typedefs:`"],
    ["int,b:", "a tag can't contain `,`"],
    ["Optional[str]", "`!Optional(str)`"],
    ["list[Buy]", "(strict YAML 1.2); write `!list(Buy)`"],
    ["Union[Buy, Sell]", "name `Union(Buy, Sell)` in `typedefs:`"],
    ["Union(Buy,", "ends at the first space"],
    ["Literal(a", "ends at the first space"],
    ["Strr", "Did you mean `str`?"],
    ["Bye", "Did you mean `Buy`?"],
  ])("rejects %p with %p", (tag, fragment) => {
    expect(rejected(tag).message).toContain(fragment);
  });

  test("an incomplete tag's message names the tag and the fix", () => {
    const message = rejected("Union(Buy,").message;
    expect(message).toContain("`!Union(Buy,` is not a complete type");
    expect(message).toContain("(e.g. `Decision: Union(Buy, Sell)`)");
    expect(message).toContain("(`!Decision`)");
  });

  test("an unknown inner name gets no misleading head hint", () => {
    expect(rejected("list(Strr)").message).not.toContain("Did you mean");
  });

  test.each([
    ["dict[str, int]", "`!object`"],
    ["list[dict[str, int]]", "`!list(object)`"],
  ])("a bracket dict %p suggests %p", (tag, spelled) => {
    expect(rejected(tag).message).toContain(`write ${spelled}`);
  });

  test("the bracket hint is offered only for a type that resolves", () => {
    const message = rejected("list[Nope]").message;
    expect(message).not.toContain("write");
    expect(message).not.toContain("typedefs:");
  });
});

describe("suggestTag", () => {
  test("spells a tagged slot or advises a typedef", () => {
    expect(suggestTag("qty", t("int"), "${x}")).toBe("`qty: !int ${x}`");
    expect(suggestTag("qty", t("Optional(int(1..12))"), null)).toBe(
      "`qty: !Optional(int(1..12))`",
    );
    expect(suggestTag("a", t("Literal(buy)"), "buy")).toBe("`a: !Literal(buy) buy`");
    expect(suggestTag("d", t("Union(Buy, Sell)"), "${x}")).toContain("typedefs:");
    expect(suggestTag("d", t("Literal(a, b)"), null)).toBe(
      "name `Literal(a, b)` in `typedefs:` and tag with its name",
    );
    expect(suggestTag("d", t("Buy"), null)).toContain("typedefs:");
    expect(suggestTag("d", t('Literal("a b")'), null)).toContain("typedefs:");
  });

  test("prefers a typedef name", () => {
    expect(suggestTag("d", t("Buy"), null, { registry: REG })).toBe("`d: !Buy`");
    expect(suggestTag("d", t("Optional(Buy)"), "${x}", { registry: REG })).toBe(
      "`d: !Optional(Buy) ${x}`",
    );
    expect(suggestTag("d", t("list(Decision)"), null, { registry: REG })).toBe(
      "`d: !list(Decision)`",
    );
    expect(suggestTag("d", t("int"), null, { registry: REG })).toBe("`d: !int`");
    expect(suggestTag("d", t("Buy"), null, { typeText: "Buy" })).toBe("`d: !Buy`");
    expect(suggestTag("d", t("Buy"), null, { typeText: "{a: int}" })).toContain("typedefs:");
  });

  test("spells a type directly before naming a scalar alias", () => {
    const registry = readTypedefs({
      Price: "float",
      Qty: "int(1..12)",
      Buy: { action: "Literal(buy)" },
    });
    expect(suggestTag("p", t("float", registry), null, { registry })).toBe("`p: !float`");
    expect(suggestTag("p", t("list(float)", registry), null, { registry })).toBe(
      "`p: !list(float)`",
    );
    expect(suggestTag("q", t("int(1..12)", registry), null, { registry })).toBe(
      "`q: !int(1..12)`",
    );
    expect(suggestTag("b", t("list(Buy)", registry), null, { registry })).toBe(
      "`b: !list(Buy)`",
    );
  });

  test("needs a type or its text", () => {
    expect(() => suggestTag("d", null, null)).toThrow(
      new TypeError("suggestTag needs a type or its text"),
    );
  });
});

describe("old typed text", () => {
  test.each([
    ["int = 3", ["int", "3"]],
    ["int =", ["int", ""]],
    [" a b  =  c = d ", ["a b", "c = d"]],
    ["call(f, a = 1)", null],
    ["no separator", null],
    [5n, null],
  ])("splitTypedText(%p) is %p", (value, expected) => {
    expect(splitTypedText(value)).toEqual(expected as [string, string] | null);
  });

  test("oldTypedText recognizes a type on the left in either spelling", () => {
    const [typ, spelled, after] = oldTypedText("Optional[int] = 3", REG)!;
    expect(typeEquals(typ, t("Optional(int)"))).toBe(true);
    expect([spelled, after]).toEqual(["Optional(int)", "3"]);
    expect(oldTypedText("hello = 3", REG)).toBeNull();
    expect(oldTypedText("plain", REG)).toBeNull();
  });

  test("spellTypeText spells parsed type text without resolving names", () => {
    expect(spellTypeText("list[message]")).toBe("list(message)");
    expect(spellTypeText(" Optional(int) ")).toBe("Optional(int)");
    expect(spellTypeText("not a type (")).toBeNull();
  });

  test("closeTypeName", () => {
    expect(closeTypeName("INT", REG)).toBe("int");
    expect(closeTypeName("list(Strr)", REG)).toBeNull();
    expect(closeTypeName("(int)", REG)).toBeNull();
  });

  test.each([
    ["30", "str", ['"30"', null]],
    ["no", "bool", ["false", "`no` is text; write `false`"]],
    ["5", "int", ["5", null]],
    ["[", "int", ["[", null]],
    ["${x}", "int", ["${x}", null]],
    ["call(f)", "int", ["call(f)", null]],
  ])("suggestedLiteral(%p) for %p", (text, typeText, expected) => {
    expect(suggestedLiteral(text, t(typeText))).toEqual(expected as [string, string | null]);
  });
});

describe("message pieces", () => {
  test.each([
    ["plain", '"plain"'],
    ["", '""'],
    ['say "hi"', `'say "hi"'`],
    ["it's \\ here", "'it''s \\ here'"],
    ["a\nb", '"a\\nb"'],
    ['a\t"b"', '"a\\t\\"b\\""'],
    ["a\u00a0b", '"a\\u00a0b"'],
    ["\u{1F600}", '"\u{1F600}"'],
    ["\ud800", '"\\ud800"'],
    ["\u{E0001}", '"\\U000E0001"'],
  ])("yamlQuoted(%p) is %p", (text, expected) => {
    expect(yamlQuoted(text)).toBe(expected);
  });

  test("pathLabel", () => {
    expect(pathLabel("x", [])).toBe("x");
    expect(pathLabel("x", ["items", 0, "a"])).toBe("x.items[0].a");
  });

  test.each([
    ["${x}", true],
    ["a ${x.y} b", true],
    ["plain", false],
    ["$${x}", false],
    ["${", true],
  ])("hasSpan(%p) is %p", (text, expected) => {
    expect(hasSpan(text)).toBe(expected);
  });

  test("namedTypeLabel names a record or union by its typedef", () => {
    expect(namedTypeLabel(t("Buy"), REG)).toBe("Buy");
    expect(namedTypeLabel(t("Optional(Decision)"), REG)).toBe("Optional(Decision)");
    expect(namedTypeLabel(t("list(Buy)"), REG)).toBe("list(Buy)");
    expect(namedTypeLabel(t("Optional(int(1..12))"), REG)).toBe("Optional(int(1..12))");
    expect(namedTypeLabel(t("Buy"), EMPTY)).toBe('{"action":Literal(buy),"conviction":float}');
  });
});

describe("conformance message fragments", () => {
  test("anySourceHint: an Any into a loop-carried int", () => {
    expect(anySourceHint("n", t("Any"), t("int"))).toBe(
      "tag the slot (`n: !int ${...}`) to check it at runtime, or narrow it with `isinstance` " +
        "in a `case`",
    );
  });

  test("suggestTag fragments", () => {
    expect(suggestTag("acc", t("Optional(Any)"), "${mk.output}")).toBe(
      "`acc: !Optional(Any) ${mk.output}`",
    );
    expect(suggestTag("route_then_0", t("str"), "hot")).toBe("`route_then_0: !str hot`");
    expect(suggestTag("cursor", t("str"), yamlQuoted("p1"))).toBe('`cursor: !str "p1"`');
    expect(yamlQuoted("1")).toBe('"1"');
    expect(suggestTag("replyy_text", t("str"), "replyy")).toBe("`replyy_text: !str replyy`");
  });

  test("old typed binding and flow output text", () => {
    const options = { registry: EMPTY, placeholder: "${...}" };
    expect(oldFormSuggestion("n", t("int"), "int", "5", options)).toBe("`n: !int 5`");
    expect(yamlQuoted("int = 5")).toBe('"int = 5"');
    expect(oldFormSuggestion("r", t("int"), "int", "${use.output}", options)).toBe(
      "`r: !int ${use.output}`",
    );
    expect(yamlQuoted("int = ${use.output}")).toBe('"int = ${use.output}"');
  });
});
