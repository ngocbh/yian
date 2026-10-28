/**
 * `readTypedefs`, the `typedefs:` registry builder.
 */

import { describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { readTypedefs, type RecordDef, typeFor } from "../../src/typesys/types.ts";
import { buildValueAs, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";

/** Return the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Readonly<Record<string, unknown>>): TypeCheckError {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("readTypedefs did not throw");
}

/** Return the `TypeCheckError` `run` throws. */
function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

/** Expect `readTypedefs(raw)` to throw exactly `message` at `path`. */
function expectRejected(
  raw: Readonly<Record<string, unknown>>,
  message: string,
  path: readonly string[],
): void {
  const error = typedefsError(raw);
  expect(error.message).toBe(message);
  expect(error.path).toEqual(path);
}

describe("building", () => {
  test("records, enums and aliases", () => {
    const reg = readTypedefs({
      Topic: "str",
      Amount: "float",
      Category: "Literal(pro, con, mixed)",
      Rating: { category: "Category", score: "float", note: "Optional(str)" },
    });
    expect(reg.get("Topic")?.kind).toBe("alias");
    expect(reg.get("Amount")?.kind).toBe("alias");
    expect(reg.get("Category")).toEqual({ kind: "variant", tags: ["pro", "con", "mixed"] });
    expect(reg.get("Rating")?.kind).toBe("record");
  });

  test("an alias composes an alias", () => {
    const reg = readTypedefs({ Topic: "str", Bundle: "list(Topic)" });
    expect(typeFor("Bundle", reg).kind).toBe(ValueKind.LIST_STRING);
  });

  test("bounded aliases keep Optional and nested value paths", () => {
    const registry = readTypedefs({
      Score: "float(0..1)",
      MaybeScore: "Optional(Score)",
      Rating: { scores: "list(MaybeScore)", count: "int(1..12)" },
      Ratings: "list(Rating)",
    });
    const typ = typeFor("Ratings", registry);
    typ.validateShape();
    // The int 1 fills a float slot and stays the int 1.
    const value = [{ scores: [null, 0.5, 1n], count: 4n }];
    expect(buildValueAs(typ, value).toObject()).toEqual([{ scores: [null, 0.5, 1n], count: 4n }]);
    expect(buildValueAs(typeFor("MaybeScore", registry), null).toObject()).toBeNull();
    for (const [bad, path] of [
      [[{ scores: [null, 1.1], count: 4n }], [0, "scores", 1]],
      [[{ scores: [0n], count: 0n }], [0, "count"]],
    ] as const) {
      expect(typeCheckError(() => buildValueAs(typ, bad)).path).toEqual(path);
    }
  });

  test("a record resolves with an enum and an Optional field", () => {
    const reg = readTypedefs({
      Category: "Literal(pro, con)",
      Rating: { category: "Category", score: "float", note: "Optional(str)" },
    });
    const sh = typeFor("Rating", reg);
    // note (Optional) is not required
    expect(sh.required).toEqual(new Set(["category", "score"]));
    expect(sh.fields!.get("category")!.tags).toEqual(new Set(["pro", "con"]));
    expect(sh.fields!.get("note")!.nullable).toBe(true);
  });

  test("a record resolves through other typedefs", () => {
    const reg = readTypedefs({
      Wrapper: { signal: "Signal", all: "list(Signal)" },
      Signal: { score: "float", tags: "list(str)" },
    });
    const signal = typeFor("Wrapper", reg).fields!.get("signal")!;
    expect(signal.fields!.get("tags")!.kind).toBe(ValueKind.LIST_STRING);
  });

  test.each([null, undefined, {}])("no typedefs give an empty registry: %p", (raw) => {
    expect(readTypedefs(raw).size).toBe(0);
  });
});

describe("rejected definitions", () => {
  test("an all-bare-tags sequence must be a Literal", () => {
    expectRejected(
      { Category: ["pro", "con"] },
      "typedef 'Category': a tag-only union must be a Literal(...) enum, not a sequence " +
        "(e.g. Category: Literal(pro, con))",
      ["Category"],
    );
  });

  test("an empty sequence gets the hint without an example", () => {
    expectRejected(
      { X: [] },
      "typedef 'X': a tag-only union must be a Literal(...) enum, not a sequence",
      ["X"],
    );
  });

  test("a payload union sequence is rejected", () => {
    expectRejected(
      { Choice: ["defer", { approve: { count: "int" } }] },
      "typedef 'Choice': a tagged/payload union sequence is not supported; declare each " +
        "member as a record and write Choice: Union(A, B, ...)",
      ["Choice"],
    );
  });

  test("an alias cycle is rejected eagerly", () => {
    expectRejected({ A: "B", B: "A" }, "alias cycle through 'A' in typedefs", ["A"]);
  });

  test("a recursive record is rejected eagerly", () => {
    expectRejected(
      { R: { x: "R" } },
      "typedef 'R' field 'x': recursive type reference: 'R'",
      ["R", "x"],
    );
  });

  test("mutually recursive records are rejected", () => {
    expectRejected(
      { A: { b: "B" }, B: { a: "A" } },
      "typedef 'B' field 'a': recursive type reference: 'B'",
      ["B", "a"],
    );
  });

  test("an unused record with an unknown constructor is rejected", () => {
    expectRejected(
      { Bad: { x: "lst(int)" } },
      "typedef 'Bad' field 'x': unknown type constructor 'lst' in 'lst(int)'",
      ["Bad", "x"],
    );
  });

  test("an unused record with an unknown name is rejected", () => {
    expectRejected(
      { Order: { pick: "Nope", note: "str" } },
      "typedef 'Order' field 'pick': unknown type 'Nope'",
      ["Order", "pick"],
    );
  });

  test("an unused record with bad bounds is rejected", () => {
    expectRejected(
      { Order: { qty: "int(5..1)" } },
      "typedef 'Order' field 'qty': invalid bounds in 'int(5..1)': type bounds must be " +
        "ordered from minimum to maximum",
      ["Order", "qty"],
    );
  });

  test("an unused alias to an unknown name is rejected", () => {
    expectRejected({ Al: "Nope" }, "typedef 'Al': unknown type 'Nope'", ["Al"]);
  });

  test("the error names the broken typedef, not its user", () => {
    // `Wrapper` is listed first and uses `Signal`; the fault is Signal's own field.
    expectRejected(
      { Wrapper: { signal: "Signal" }, Signal: { score: "float", tags: "lst(str)" } },
      "typedef 'Signal' field 'tags': unknown type constructor 'lst' in 'lst(str)'",
      ["Signal", "tags"],
    );
  });

  test.each([
    ["5n", 5n, "int"],
    ["5 (a YAML float such as 5.0)", 5, "float"],
    ["1.5", 1.5, "float"],
    ["true", true, "bool"],
    ["null", null, "NoneType"],
    ["a Map", new Map(), "Map"],
  ])("a definition of another type names its type: %s", (_label, value, name) => {
    expectRejected({ X: value }, `typedef 'X': unsupported definition of type ${name}`, ["X"]);
  });
});

describe("typedef names", () => {
  test("a builtin name is rejected (PascalCase first, then shadowing)", () => {
    expectRejected(
      { str: "int" },
      "typedef name 'str' must be PascalCase (an uppercase identifier)",
      ["str"],
    );
    expectRejected(
      { Optional: "int" },
      "typedef name 'Optional' shadows a builtin scalar or typing constructor",
      ["Optional"],
    );
  });

  test.each(["rating", "ǅx", "_A"])("%p is not PascalCase", (name) => {
    expectRejected(
      { [name]: { x: "int" } },
      `typedef name '${name}' must be PascalCase (an uppercase identifier)`,
      [name],
    );
  });

  test("a Unicode PascalCase name loads", () => {
    expect(readTypedefs({ Größe: "int" }).get("Größe")?.kind).toBe("alias");
  });

  test("Message is reserved", () => {
    expectRejected({ Message: { role: "str" } }, "typedef name 'Message' shadows the built-in type 'Message'", ["Message"]);
  });

  test("the part record names stay free", () => {
    expect(readTypedefs({ Text: "str", ToolCall: { id: "str" } }).size).toBe(2);
  });
});

describe("typedefs naming Message", () => {
  test("a field, an alias and a union member may be Message", () => {
    const registry = readTypedefs({
      Turn: { message: "Message", at: "str" },
      History: "list(Message)",
      Note: { kind: "Literal(note)", text: "str" },
      Entry: "Union(Message, Note)",
    });
    expect(typeFor("Turn", registry).fields!.get("message")).toBe(typeFor("Message", new Map()));
    expect(typeFor("History", registry).element).toBe(typeFor("Message", new Map()));
    expect(typeFor("Entry", registry).variants).toHaveLength(2);
  });

  test("Message's levels count toward a typedef's nesting", () => {
    // Message is 3 levels, so `Deep` is 1 + 1 + 3.
    install({ ...BUILTIN, maxTypeNesting: 4 });
    try {
      expect(() => readTypedefs({ Deep: { history: "list(Message)" } })).toThrow("type nests deeper than 4 levels");
    } finally {
      forget();
    }
  });
});

describe("record fields", () => {
  test("a field type must be a string", () => {
    // An inline nested object is not allowed here.
    expectRejected(
      { Bad: { x: { nested: "int" } } },
      "typedef 'Bad' field 'x': type must be a string expression (got dict)",
      ["Bad", "x"],
    );
  });

  test.each([
    ["5n", 5n, "int"],
    ["1.5", 1.5, "float"],
    ["true", true, "bool"],
    ["null", null, "NoneType"],
    ["an array", ["int"], "list"],
  ])("a field type of another type names its type: %s", (_label, value, name) => {
    expectRejected(
      { R: { a: value } },
      `typedef 'R' field 'a': type must be a string expression (got ${name})`,
      ["R", "a"],
    );
  });

  test.each([
    ["first", { role: "Literal(user", "assistant)": null, text: "str" }, "Literal(user"],
    ["not first", { text: "str", role: "List(Dict(str", "int))": null }, "List(Dict(str"],
  ])("a type split by a flow mapping's comma names the split type: %s", (_label, fields, cut) => {
    // YAML reads `{role: Literal(user, assistant)}` as the fields `Literal(user` and a
    // null-valued `assistant)`; the error points at the cut type, not the stray field.
    expectRejected(
      { Msg: fields },
      `typedef 'Msg' field 'role': malformed type expression '${cut}': unbalanced ` +
        "parentheses — if this type sits inside a `{...}` mapping, its comma split it; quote " +
        "it or use block style",
      ["Msg", "role"],
    );
  });

  test("a missing type is reported before another field's bad type", () => {
    // Only a type that opens more `(` than it closes can be one a comma cut.
    expectRejected(
      { Msg: { role: null, text: "Lst(str)" } },
      "typedef 'Msg' field 'role': type must be a string expression (got NoneType)",
      ["Msg", "role"],
    );
  });

  test.each([
    ["", "typedef 'R' field '': field names must be non-empty strings"],
    ["\ud800", "typedef 'R' field '\\ud800': field names must not contain a Unicode surrogate"],
  ])("field name %p is rejected", (field, message) => {
    expectRejected({ R: { [field]: "int" } }, message, ["R", field]);
  });

  test("__proto__ and constructor are ordinary fields", () => {
    // A `__proto__` key in an object literal sets the prototype, so JSON.parse builds it.
    const raw = JSON.parse('{"R": {"__proto__": "int", "constructor": "str"}}');
    const fields = typeFor("R", readTypedefs(raw)).fields!;
    expect([...fields.keys()]).toEqual(["__proto__", "constructor"]);
    expect(fields.get("__proto__")!.kind).toBe(ValueKind.INTEGER);
  });

  test("a record's fields follow property order, integer-like keys first", () => {
    const record = readTypedefs({ R: { b: "int", 1: "str", a: "bool" } }).get("R") as RecordDef;
    expect([...record.fields.keys()]).toEqual(["1", "b", "a"]);
  });

  test("a null-prototype object is a mapping", () => {
    const fields = Object.assign(Object.create(null), { a: "int" });
    const raw = Object.assign(Object.create(null), { R: fields });
    expect(readTypedefs(raw).get("R")?.kind).toBe("record");
  });
});

describe("enum tags", () => {
  test.each([
    [
      'Literal("", fast)',
      "typedef 'Mode': empty Literal member in 'Literal(\"\", fast)' (tags must be non-empty " +
        "names or quoted strings)",
    ],
    [
      'Literal("\ud800", fast)',
      "typedef 'Mode': type expression 'Literal(\"\\ud800\", fast)' contains a Unicode " +
        "surrogate (types must be valid UTF-8 text)",
    ],
  ])("%p is rejected", (expr, message) => {
    expectRejected({ Mode: expr }, message, ["Mode"]);
  });
});
