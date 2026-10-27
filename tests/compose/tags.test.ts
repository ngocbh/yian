/**
 * Reading a slot's type and value through its YAML tags, spelling tags for messages, and rejecting
 * tags outside a slot.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import {
  anySourceHint,
  closeTypeName,
  hasSpan,
  namedTypeLabel,
  oldFormSuggestion,
  oldTypedText,
  pathLabel,
  readTaggedValue,
  readTagType,
  rejectTaggedOutput,
  rejectUnreadTags,
  spellTypeText,
  splitTypedText,
  suggestedLiteral,
  suggestTag,
  TaggedSlot,
  yamlQuoted,
} from "../../src/compose/helpers/tags.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { assembled, type CaseCall, type CaseRow, cursorAt, loadError, runCase } from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

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

describe("closeTypeName", () => {
  test("offers the name closest to the leading name", () => {
    expect(closeTypeName("INT", REG)).toBe("int");
    expect(closeTypeName("list(Strr)", REG)).toBeNull();
    expect(closeTypeName("(int)", REG)).toBeNull();
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

  test.each([
    ["str", "int"],
    ["Any", "Optional(Any)"],
    ["Any", "Any"],
    ["Any", "file"],
  ])("anySourceHint has no advice from %p into %p", (source, sink) => {
    expect(anySourceHint("n", t(source), t(sink))).toBeNull();
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

  test("oldFormSuggestion notes a changed literal and fills in a missing value", () => {
    const options = { registry: EMPTY, placeholder: "${...}" };
    expect(oldFormSuggestion("f", t("bool"), "bool", "no", options)).toBe(
      "`f: !bool false` (`no` is text; write `false`)",
    );
    expect(oldFormSuggestion("n", t("int"), "int", "", options)).toBe("`n: !int ${...}`");
    expect(
      oldFormSuggestion("n", t("int"), "int", "", { registry: EMPTY, placeholder: null }),
    ).toBe("`n: !int`");
  });
});

// ---------- readTaggedValue ----------

/** Read slot `key` of `text` (or of `options.document`) as a node input. */
function read(
  text: string,
  key = "x",
  options: { ledger?: Set<number>; document?: SourceDocument; literalOnly?: boolean } = {},
): TaggedSlot {
  const document = options.document ?? SourceDocument.parse(text);
  return readTaggedValue(document.root.child(key), REG, {
    label: key,
    sitePath: ["node", "n", "input", key],
    catalogBuilder: new SourceCatalogBuilder(),
    diagnostics: LoadDiagnostics.fromDocument(document),
    ledger: options.ledger ?? new Set(),
    literalOnly: options.literalOnly,
  });
}

/** Assert reading `text` fails with a message holding `fragment`. */
function fails(text: string, fragment: string, options: { literalOnly?: boolean } = {}): void {
  expect(loadError(() => read(text, "x", options)).message).toContain(fragment);
}

/** Return `slot`'s value as a `ValueTemplate`, failing if it isn't one. */
function templateOf(slot: TaggedSlot): ValueTemplate {
  expect(slot.value).toBeInstanceOf(ValueTemplate);
  return slot.value as ValueTemplate;
}

/** Return `node` as a `Binding`, failing if it isn't one. */
function bindingOf(node: unknown): Binding {
  expect(node).toBeInstanceOf(Binding);
  return node as Binding;
}

describe("readTaggedValue", () => {
  test("the top-level tag gives the contract", () => {
    const slot = read("x: !int ${a.output}\n");
    expect([slot.value, slot.hasValue]).toEqual(["${a.output}", true]);
    expect(typeEquals(slot.contract!.type, t("int"))).toBe(true);
    expect(slot.contract!.origins.default).not.toBeNull();
  });

  test("an untagged value has no contract and is unchanged", () => {
    expect(read("x: ${a.output}\n")).toEqual(new TaggedSlot("${a.output}", null, true));
    expect(read("x: {a: 1, b: [y, 2.5], c: null}\n")).toEqual(
      new TaggedSlot({ a: 1n, b: ["y", 2.5], c: null }, null, true),
    );
    expect(read("x: '$$5'\n").value).toBe("$$5");
  });

  test("every tag read is recorded in the ledger", () => {
    const document = SourceDocument.parse(
      "x: !Order {decision: !Buy {action: buy, conviction: !float 1}, qty: !int 1}\n" +
        "y: !int 2\n",
    );
    const ledger = new Set<number>();
    read("", "x", { document, ledger });
    const unread = document.root.child("y").tagDefinitionId;
    const all = document.taggedNodes().map((node) => node.definitionId);
    expect(ledger).toEqual(new Set(all.filter((id) => id !== unread)));
    expect(ledger.size).toBe(4);
  });

  test("an empty tagged value has no value", () => {
    const slot = read("x: !Optional(int)\n");
    expect([slot.value, slot.hasValue]).toEqual([null, false]);
    expect(slot.contract!.type.nullable).toBe(true);
  });

  test.each(["x:\n  - !int\n", "x:\n  d: !Buy\n", "x: !list(int)\n  - 1\n  - !int\n"])(
    "a nested empty tag is an error: %p",
    (text) => {
      fails(text, "a nested tag needs a value");
    },
  );

  test.each([
    ["x: !int (1..12) 5\n", "`!int(1..12)`"],
    ["x: !Optional (str)\n", "`!Optional(str)`"],
    ["x: [!int (1..12) 5]\n", "`!int(1..12)`"],
    ["x: !Optional (list(int))\n", "`!Optional(list(int))`"],
    ["x: !list (Optional(int))\n", "`!list(Optional(int))`"],
  ])("a space after a tag is an error: %p", (text, spelled) => {
    fails(text, `a YAML tag ends at the first space; write ${spelled} without spaces`);
  });

  test("quoted text starting with a parenthesis is text", () => {
    expect(read("x: !str '(a) b'\n").value).toBe("(a) b");
  });

  test("top-level strings are kept as written", () => {
    expect(read('x: !str "Status: ${a.output}"\n').value).toBe("Status: ${a.output}");
    expect(read("x: !int call(f, a=1)\n").value).toBe("call(f, a=1)");
    expect(read("x: !str '$$5'\n").value).toBe("$$5");
  });

  test("a nested tagged template becomes a nested binding", () => {
    const slot = read('x: {d: !Buy "${d.output}", n: 1}\n');
    const template = templateOf(slot);
    expect(slot.contract).toBeNull();
    const value = template.value as Record<string, unknown>;
    const nested = bindingOf(value["d"]);
    expect(nested.source).toBe("${d.output}");
    expect(typeEquals(nested.contract!.type, t("Buy"))).toBe(true);
    expect(nested.origin).not.toBeNull();
    expect(nested.contract!.origins.default).toBe(nested.origin);
    expect(value["n"]).toBe(1n);
    const leaf = template.originAt(["d"]);
    expect(leaf).not.toBeNull();
    expect(leaf!.key).not.toBe(nested.origin!.key);
  });

  test("a nested tagged literal is checked and kept as a binding", () => {
    const value = templateOf(read("x: {q: !int 5, s: !str '$$x'}\n")).value as Record<
      string,
      unknown
    >;
    expect(bindingOf(value["q"]).source).toBe(5n);
    expect(typeEquals(bindingOf(value["q"]).contract!.type, t("int"))).toBe(true);
    // The binder unescapes `$$` in a template, so the raw text is kept.
    expect(bindingOf(value["s"]).source).toBe("$$x");
    fails("x: {q: !int '5'}\n", "'x.q': `\"5\"` is not a valid int");
  });

  test("a list of tagged items", () => {
    const items = templateOf(read('x: !list(int) [!int "${a.output}", !int 2]\n'))
      .value as unknown[];
    const [first, second] = items.map(bindingOf);
    expect([first!.source, second!.source]).toEqual(["${a.output}", 2n]);
    for (const binding of [first!, second!]) {
      expect(typeEquals(binding.contract!.type, t("int"))).toBe(true);
    }
  });

  test.each(["x: {a: call(f)}\n", "x: {a: 'call(f, b=1)'}\n", "x: [' call(f)']\n"])(
    "a nested call is an error: %p",
    (text) => {
      fails(
        text,
        "a `call(...)` inside a value isn't supported; bind it at the top level, " +
          "or reword the text",
      );
    },
  );

  test("nested spans make a template with raw leaf strings", () => {
    const template = templateOf(read('x: {a: "${n.output}", b: [1, "${m.output}"], c: "$$x"}\n'));
    expect(template.value).toEqual({ a: "${n.output}", b: [1n, "${m.output}"], c: "$$x" });
    expect(template.originAt(["a"])).not.toBeNull();
    expect(template.originAt(["b", 1])).not.toBeNull();
    expect(template.originAt(["c"])).not.toBeNull();
  });

  test("a span-free container is a literal unescaped once", () => {
    expect(read('x: {c: "$$x", d: ["$$$${y}"]}\n').value).toEqual({ c: "$x", d: ["$${y}"] });
  });

  test("a top-level tagged container with spans is checked later", () => {
    templateOf(read('x: !Buy {action: buy, conviction: "${c.output}"}\n'));
  });

  test("a wrong literal is an error", () => {
    fails("x: !int '5'\n", "'x': `\"5\"` is not a valid int");
    fails("x: !int(1..12) 13\n", "`13` is not a valid int(1..12)");
    fails(
      "x: !date 2026/01/01\n",
      "'x': `2026/01/01` is not a valid date (date must look like 2026-05-21)",
    );
    fails("x: !Buy {action: sell, conviction: 1}\n", "not a valid Buy");
    fails('x: !Order {decision: !Buy {action: buy, conviction: 1}, qty: "1"}\n', "(at `x.qty`)");
  });

  test("a str literal read as a number gets the quote hint", () => {
    fails(
      "x: !str 30\n",
      "'x': `30` is not a valid str; quote it to keep it as text: `x: !str \"30\"`",
    );
    const message = loadError(() => read("x: !str true\n")).message;
    expect(message).toContain('`x: !str "true"`');
    expect(message).toContain("(YAML reads unquoted true/false as booleans)");
    expect(read("x: !str yes\n").value).toBe("yes");
    // An unquoted date is text, so it fits `str` as written.
    const items = templateOf(read("x: [!str 2026-01-01]\n")).value as unknown[];
    expect(bindingOf(items[0]).source).toBe("2026-01-01");
  });

  test.each([
    ["x: !Buy {conviction: 1}\n", "x.action"],
    ["x: !list(Buy) [{conviction: 1}]\n", "x[0].action"],
    ["x: !Order {decision: {conviction: 1}, qty: 1}\n", "x.decision.action"],
    ["x: {d: !Buy {conviction: 1}}\n", "x.d.action"],
  ])("a tag never fills in a missing field: %p", (text, at) => {
    fails(text, `record missing required fields: ["action"] (at \`${at}\`)`);
  });

  test("a valid union member loads", () => {
    expect(read("x: !Decision {action: sell, conviction: 0.5}\n").value).toEqual({
      action: "sell",
      conviction: 0.5,
    });
  });

  test("literalOnly folds nested tags into a plain value", () => {
    const slot = read("x: !Order {decision: !Buy {action: buy, conviction: 1}, qty: 1}\n", "x", {
      literalOnly: true,
    });
    expect(slot.value).toEqual({ decision: { action: "buy", conviction: 1n }, qty: 1n });
    expect(read("x: !str ${y}\n", "x", { literalOnly: true }).value).toBe("${y}");
    expect(read('x: {s: "$$y", t: call(f)}\n', "x", { literalOnly: true }).value).toEqual({
      s: "$$y",
      t: "call(f)",
    });
    expect(read("x: !int\n", "x", { literalOnly: true }).hasValue).toBe(false);
    fails("x: {d: !Buy {conviction: 1}}\n", "record missing required fields", {
      literalOnly: true,
    });
  });

  test("text starting with a parenthesis after a tag must be quoted", () => {
    fails(
      "x: !str (555) 123-4567\n",
      "'x': a YAML tag ends at the first space; quote the text: " +
        '`x: !str "(555) 123-4567"`',
    );
    fails("x: [!str (555) 1]\n", 'quote the text: `!str "(555) 1"`');
    fails("x: !Strr (a)\n", "Did you mean `str`?");
  });

  test("a mismatch at the root does not repeat the path", () => {
    const message = loadError(() => read("x: !int [1]\n")).message;
    expect(message).toContain("'x': the value is not a valid int:");
    expect(message).not.toContain("(at");
  });

  test("block scalar text is quoted in messages", () => {
    fails("x: !int |\n  5\n", '`"5\\n"` is not a valid int');
  });

  test("a recursive value error names the slot", () => {
    fails("x: &a {b: *a}\n", "'x': cannot descend through a recursive alias");
  });

  test("text starting with a call skips the load check", () => {
    // The inline-call desugar knows the YAML style: unquoted, the text must be one well-formed
    // directive; quoted, it is text.
    expect(read("x: !str call(f) and more\n").value).toBe("call(f) and more");
    expect(read("x: !int call(f) and more\n").value).toBe("call(f) and more");
    expect(read("x: !int call(f, a=1)\n").value).toBe("call(f, a=1)");
    fails("x: !int calls(f)\n", "is not a valid int");
  });

  test("a verbatim tag carries brackets and commas into the tag", () => {
    fails(
      "x: !<!list[Buy]> v\n",
      "'x': a tag can't contain `[`, `]`, `{` or `}` (strict YAML 1.2); write `!list(Buy)`",
    );
    fails(
      "x: !<!Union(Buy,Sell)> v\n",
      "'x': `!Union(Buy,Sell)`: a tag can't contain `,`; name it in `typedefs:`",
    );
  });

  test("a tag in a flow collection stops at the comma", () => {
    fails("x: {d: !Union(Buy, Sell) y}\n", "'x.d': `!Union(Buy` is not a complete type");
  });

  test("a %XX escape in a tag is kept raw", () => {
    fails("x: !e%2Cf v\n", "'x': `!e%2Cf` is not a type (");
  });

  test("!Any takes a date as text and refuses a null", () => {
    expect(read("x: !Any 2026-01-01\n").value).toBe("2026-01-01");
    fails(
      "x: !Any null\n",
      "'x': `null` is not a valid Any; to allow null, write `x: !Optional(Any) null`",
    );
  });
});

// ---------- rejectTaggedOutput, rejectUnreadTags ----------

/** Return the message of the `LoadError` checking `output:` in `text` throws. */
function outputRejection(text: string): string {
  const document = SourceDocument.parse(text);
  const diagnostics = LoadDiagnostics.fromDocument(document);
  return loadError(() => rejectTaggedOutput(document.root.child("output"), diagnostics)).message;
}

/** Check `text` for unread tags with an empty ledger. */
function checkUnread(text: string, options: { typedefsOnly?: boolean } = {}): void {
  const document = SourceDocument.parse(text);
  rejectUnreadTags(document, new Set(), LoadDiagnostics.fromDocument(document), options);
}

describe("tag positions", () => {
  test("rejectTaggedOutput names a nested record key and a list item", () => {
    expect(outputRejection("output:\n  a:\n    b: !int\n")).toBe(
      "`output:` declares a type; write `b: int`, without a tag",
    );
    expect(outputRejection("output:\n  - str\n  - !int\n")).toBe(
      "`output:` declares a type; an item of `output:` is tagged `!int`; write the type " +
        "without a tag",
    );
  });

  test("rejectUnreadTags with typedefsOnly checks only typedefs", () => {
    checkUnread("kind: !str code\ntypedefs:\n  A: int\n", { typedefsOnly: true });
    expect(
      loadError(() => checkUnread("typedefs:\n  A: !int x\n", { typedefsOnly: true })).message,
    ).toBe("typedef 'A': a type definition is written as a type, without a tag");
    expect(loadError(() => checkUnread("kind: !str code\n")).message).toContain(
      "`kind:` is not one",
    );
  });

  test("rejectUnreadTags skips a tag under an x- key", () => {
    checkUnread("x-a: !int 1\nb: 2\n");
  });

  test("rejectUnreadTags gives a tagged mapping key its own message", () => {
    expect(loadError(() => checkUnread("a:\n  !int b: 1\n")).message).toBe(
      "`!int b:` tags a mapping key; a tag types a value, so put it after the colon: " +
        "`b: !int <value>`",
    );
  });
});

// ---------- Harvested cases ----------

/**
 * A slot `readTaggedValue` reads: its path from the root and its options. Its label is the
 * path's last step, as the loader's labels are for every harvested slot.
 */
type Slot = [path: string[], options?: { literalOnly?: boolean; nestedCallFix?: string }];

const LLM_CONFIG_CALL_FIX =
  "compute it in a node and read that node's output with `${...}`, or reword the text";
const LITERAL = { literalOnly: true };

/** Read `slot` as the loader does, with the file's consumed-tag ledger. */
function readSlot(
  [path, options = {}]: Slot,
  document: SourceDocument,
  registry: TypeRegistry,
  diagnostics: LoadDiagnostics,
  builder: SourceCatalogBuilder,
): void {
  readTaggedValue(cursorAt(document, path), registry, {
    label: path.at(-1)!,
    sitePath: path,
    catalogBuilder: builder,
    diagnostics,
    ledger: builder.consumedTags,
    ...options,
  });
}

const slot =
  (...read: Slot): CaseCall =>
  (document, registry, diagnostics, builder) =>
    readSlot(read, document, registry, diagnostics, builder);

const output =
  (path: string[]): CaseCall =>
  (document, _registry, diagnostics) =>
    rejectTaggedOutput(cursorAt(document, path), diagnostics);

/**
 * Check for unread tags as the loader does, after reading the `ledger` slots, so the ledger
 * holds every legitimately read tag.
 */
const unread =
  (options: { typedefsOnly?: boolean; ledger?: Slot[] } = {}): CaseCall =>
  (document, registry, diagnostics, builder) => {
    for (const read of options.ledger ?? []) readSlot(read, document, registry, diagnostics, builder);
    rejectUnreadTags(document, builder.consumedTags, diagnostics, { typedefsOnly: options.typedefsOnly });
  };

describe("harvested cases", () => {
  test.each<CaseRow>([
    {
      id: "types/compose_open_range_flows__no_ends_in_a_tag",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "types/compose_open_range_flows__tagged_literal_outside_an_open_range",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "load-types/compose_any_flows__a_null_any_literal_is_a_load_error",
      call: slot(["nodes", "n", "input", "x"]),
    },
    {
      id: "load-types/compose_tagged_bindings__tagged_mapping_output_that_does_not_fit_fails",
      call: slot(["output"]),
    },
    {
      id: "load-types/compose_tagged_bindings__unknown_tag_on_an_input_offers_a_close_match",
      call: slot(["nodes", "use", "input", "qty"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__null_literal_for_an_output_field",
      call: slot(["output", "x"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__number_literal_for_str_suggests_quoting",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__text_literal_for_int_fails_at_its_line",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__true_literal_for_str_explains_yaml_booleans",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_list_under_an_object_tag_is_rejected",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_mistyped_literal_under_a_credential_name",
      call: slot(["nodes", "api_key"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_nested_tag_is_checked_at_its_own_line",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_number_under_a_str_tag_asks_for_quotes",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_spaced_tag_is_rejected",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__a_tag_does_not_fill_in_a_missing_field",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-case/compose_value_entries_load__an_unknown_tag_suggests_a_type",
      call: slot(["nodes", "x"]),
    },
    {
      id: "load-agent/compose_llm_config_values_load__a_call_inside_an_llm_config_string_fails",
      call: slot(["nodes", "a", "llm_config"], { nestedCallFix: LLM_CONFIG_CALL_FIX }),
    },
    {
      id: "yaml/compose_untyped_output_dates__any_tag_on_a_date",
      call: slot(["output", "as_of"]),
    },
    {
      id: "yaml/compose_yaml_node_limit__a_recursive_alias_keeps_its_error",
      call: slot(["nodes", "use", "input", "topic"]),
    },
    {
      id: "yaml/compose_yaml_words__former_boolean_in_a_bool_slot",
      call: slot(["output", "ok"]),
    },
    {
      id: "yaml/compose_yaml_words__former_boolean_inside_a_list_gets_no_fix",
      call: slot(["output", "ok"]),
    },
    {
      id: "types/compose_any_name_literals__a_typed_binding_under_a_credential_name_loads",
      call: slot(["nodes", "fetch", "input", "password"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__a_null_literal_fits_an_optional_tag",
      call: slot(["nodes", "use", "input", "x"]),
    },
    {
      id: "load-types/compose_typed_bindings_check__str_into_literal_loads",
      call: slot(["nodes", "use", "input", "x"]),
    },
  ])("readTaggedValue: $id", runCase);

  test.each<CaseRow>([
    {
      id: "load-types/compose_tag_positions__tagged_agent_output_is_rejected",
      call: output(["nodes", "n", "output"]),
    },
    {
      id: "load-types/compose_tag_positions__tagged_compact_output_is_rejected_at_its_line",
      call: output(["output"]),
    },
    {
      id: "load-types/compose_tag_positions__tagged_list_item_in_a_node_output_says_the_item_is_tagged",
      call: output(["nodes", "n", "output"]),
    },
    {
      id: "load-types/compose_tag_positions__tagged_tool_output_is_rejected",
      call: output(["nodes", "n", "output"]),
    },
    {
      id: "load-types/compose_tag_positions__tagged_single_node_def_output_is_rejected",
      call: output(["defs", "echo", "output"]),
      prefix: "defs entry 'echo': ",
    },
  ])("rejectTaggedOutput: $id", runCase);

  test.each<CaseRow>([
    {
      id: "load-types/compose_tag_positions__tag_on_kind_is_rejected_at_its_line",
      call: unread(),
    },
    {
      id: "load-types/compose_tag_positions__tag_on_prompt_is_rejected_at_its_line",
      call: unread(),
    },
    {
      id: "load-types/compose_tag_positions__tag_outside_a_slot_in_a_def_is_rejected",
      call: unread({ ledger: [[["defs", "sub", "input", "x"], LITERAL]] }),
    },
    {
      id: "load-types/compose_tag_positions__tag_outside_a_slot_in_a_uses_child_is_rejected",
      call: unread({ ledger: [[["input", "x"], LITERAL]] }),
      file: "child.yaml",
    },
    {
      id: "load-types/compose_tag_positions__tag_outside_a_slot_in_an_inline_body_is_rejected",
      call: unread(),
    },
    {
      id: "load-types/compose_tag_positions__tagged_mapping_key_gets_its_own_message",
      call: unread(),
    },
    {
      id: "load-types/compose_tag_positions__tagged_typedef_is_rejected_at_its_line",
      call: unread({ typedefsOnly: true }),
    },
    {
      id: "load-types/compose_tag_positions__tags_in_flow_inputs_load",
      call: unread({
        ledger: [
          [["input", "n"], LITERAL],
          [["input", "m"], LITERAL],
        ],
      }),
    },
    {
      id: "load-types/compose_tag_positions__tags_in_a_def_slot_load",
      call: unread({
        ledger: [
          [["defs", "sub", "input", "x"], LITERAL],
          [["defs", "sub", "nodes", "echo", "input", "value"]],
        ],
      }),
    },
    {
      id: "load-types/compose_tag_positions__tag_under_an_extension_key_aliased_into_a_slot_loads",
      call: unread({ ledger: [[["nodes", "n", "input", "qty"]]] }),
    },
  ])("rejectUnreadTags: $id", runCase);

  // The inline body's slots are read by its child's assembly.
  test.each<CaseRow>([{ id: "load-types/compose_tag_positions__tags_in_an_inline_body_slot_load", call: assembled }])(
    "assemble: $id",
    runCase,
  );
});
