/**
 * `Union(...)` of records in the type grammar and in `typedefs:`, a port of
 * `tests/engine/test_union_typedefs.py`.
 *
 * Covered by the `types` cases of the same names loading the same flows (the CODE node naming the
 * conformance fixture), the bad union's message and typedef-key location pinned:
 * `test_load_flow_locates_bad_union_at_its_typedef`
 * (`engine_union_typedefs__load_flow_locates_bad_union_at_its_typedef`) and
 * `test_load_flow_accepts_union_typedef`
 * (`engine_union_typedefs__load_flow_accepts_union_typedef`).
 */

import { describe, expect, test } from "bun:test";

import {
  parseType,
  readTypedefs,
  resolveType,
  type TypeRegistry,
  typeFor,
} from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  canonicalUnionOrder,
  type Type,
  TypeCheckError,
  typeEquals,
  unionTagFields,
  ValueKind,
} from "../../src/typesys/values.ts";

type RawTypedefs = Record<string, string | Record<string, string>>;

const TRADES: RawTypedefs = {
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Hold: { action: "Literal(hold)" },
};

/** The trades registry with `extra` typedefs added. */
function trades(extra: RawTypedefs = {}): TypeRegistry {
  return readTypedefs({ ...TRADES, ...extra });
}

/** Each member's `action` tag, in order. */
function tags(members: readonly Type[]): string[] {
  return members.map((member) => [...member.fields!.get("action")!.tags!][0]!);
}

/** Expect two lists of types to be equal member by member. */
function expectTypesEqual(a: readonly Type[], b: readonly Type[]): void {
  expect(a.length).toBe(b.length);
  a.forEach((typ, i) => expect(typeEquals(typ, b[i]!)).toBe(true));
}

/** Return the message of the `TypeCheckError` `parseType(text)` throws. */
function parseError(text: string): string {
  try {
    parseType(text);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error(`parseType(${JSON.stringify(text)}) did not throw`);
}

describe("parsing", () => {
  test("parses a union", () => {
    expect(parseType("Union(Buy, Sell)")).toEqual({
      kind: "union",
      members: [
        { kind: "ref", name: "Buy" },
        { kind: "ref", name: "Sell" },
      ],
    });
  });

  test("parses a union inside Optional", () => {
    expect(parseType("Optional(Union(Buy, Sell))")).toEqual({
      kind: "optional",
      inner: {
        kind: "union",
        members: [
          { kind: "ref", name: "Buy" },
          { kind: "ref", name: "Sell" },
        ],
      },
    });
  });

  test.each([
    [
      "Buy | Sell",
      "malformed type expression 'Buy | Sell': use Union(Buy, Sell) (| is not a type operator)",
    ],
    [
      "Optional(Buy | Sell)",
      "malformed type expression 'Optional(Buy | Sell)': " +
        "use Union(Buy, Sell) (| is not a type operator)",
    ],
  ])("the bar operator hints Union: %p", (text, message) => {
    expect(parseError(text)).toBe(message);
  });

  test("a bar chain hints every member", () => {
    expect(parseError("Buy | Sell | Hold")).toBe(
      "malformed type expression 'Buy | Sell | Hold': " +
        "use Union(Buy, Sell, Hold) (| is not a type operator)",
    );
  });

  test("a None member hints Optional", () => {
    expect(parseError("Union(Buy, None)")).toBe(
      "None is not a union member in 'Union(Buy, None)'; use Optional(Union(...))",
    );
  });

  test.each([
    ["Union()", "malformed type expression 'Union()': Union(...) needs a type"],
    ["Union(())", "malformed type expression 'Union(())'"],
  ])("an empty union is malformed: %p", (text, message) => {
    expect(parseError(text)).toBe(message);
  });
});

describe("resolution", () => {
  test("a typedef union orders its members by tag", () => {
    const reg = trades({ Decision: "Union(Buy, Sell, Hold)" });
    const decision = typeFor("Decision", reg);
    expect(decision.kind).toBe(ValueKind.OBJECT);
    expect(decision.nullable).toBe(false);
    expect(tags(decision.variants!)).toEqual(["buy", "hold", "sell"]);
    expect(typeEquals(decision.variants![0]!, typeFor("Buy", reg))).toBe(true);
    expect(unionTagFields(decision)).toEqual(["action"]);
  });

  test("a union resolves in every position", () => {
    const reg = trades({ Decision: "Union(Buy, Sell, Hold)", Order: { decision: "Decision" } });
    const decision = typeFor("Decision", reg);

    const optional = typeFor("Optional(Decision)", reg);
    expect(optional.nullable).toBe(true);
    expectTypesEqual(optional.variants!, decision.variants!);

    const listed = typeFor("list(Decision)", reg);
    expect(listed.kind).toBe(ValueKind.LIST_OBJECT);
    expect(typeEquals(listed.element!, decision)).toBe(true);

    const order = typeFor("Order", reg);
    expect(typeEquals(order.fields!.get("decision")!, decision)).toBe(true);
    expect(order.required).toEqual(new Set(["decision"]));

    expect(typeFor("Optional(Union(Buy, Sell))", reg).nullable).toBe(true);

    expect(tags(typeFor("Union(Buy, Sell)", reg).variants!)).toEqual(["buy", "sell"]);
  });

  test("an Optional union field is not required", () => {
    const reg = trades({ Order: { decision: "Optional(Union(Buy, Sell))", id: "str" } });
    expect(typeFor("Order", reg).required).toEqual(new Set(["id"]));
  });

  test("nested unions and aliases flatten without duplicates", () => {
    const reg = trades({
      Purchase: "Buy",
      Trade: "Union(Buy, Sell)",
      Decision: "Union(Trade, Purchase, Hold, Union(Sell, Hold))",
    });
    const decision = typeFor("Decision", reg);
    expect(tags(decision.variants!)).toEqual(["buy", "hold", "sell"]);
    expect(typeEquals(decision, typeFor("Union(Buy, Sell, Hold)", reg))).toBe(true);
  });

  test("a single-member union is the member", () => {
    const reg = trades();
    expect(typeEquals(typeFor("Union(Buy)", reg), typeFor("Buy", reg))).toBe(true);
    expect(typeEquals(typeFor("Union(Buy, Buy)", reg), typeFor("Buy", reg))).toBe(true);
  });

  test("authored order does not affect equality", () => {
    const reg = trades();
    expect(
      typeEquals(typeFor("Union(Buy, Sell, Hold)", reg), typeFor("Union(Hold, Sell, Buy)", reg)),
    ).toBe(true);
  });

  test("an untagged disjoint union resolves", () => {
    const reg = readTypedefs({
      Text: { text: "str" },
      Call: { name: "str", input: "object" },
      Reply: "Union(Text, Call)",
    });
    const reply = typeFor("Reply", reg);
    expect(unionTagFields(reply)).toEqual([]);
    const members = [typeFor("Text", reg), typeFor("Call", reg)];
    expectTypesEqual(reply.variants!, canonicalUnionOrder(members));
    expect(new Set(reply.variants!.map(canonicalTypeLabel))).toEqual(
      new Set(members.map(canonicalTypeLabel)),
    );
  });

  test("resolveType accepts a union expression directly", () => {
    const union = resolveType(
      {
        kind: "union",
        members: [
          { kind: "ref", name: "Sell" },
          { kind: "ref", name: "Buy" },
        ],
      },
      trades(),
    );
    expect(tags(union.variants!)).toEqual(["buy", "sell"]);
  });

  test("the canonical label uses the canonical order", () => {
    const reg = trades();
    const buy = canonicalTypeLabel(typeFor("Buy", reg));
    const sell = canonicalTypeLabel(typeFor("Sell", reg));
    expect(canonicalTypeLabel(typeFor("Union(Sell, Buy)", reg))).toBe(`Union(${buy},${sell})`);
    expect(canonicalTypeLabel(typeFor("Optional(Union(Sell, Buy))", reg))).toBe(
      `Optional(Union(${buy},${sell}))`,
    );
  });
});

/** Return the message of the `TypeCheckError` `typeFor(text, reg)` throws. */
function typeForError(text: string, reg: TypeRegistry = trades()): string {
  try {
    typeFor(text, reg);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error(`typeFor(${JSON.stringify(text)}) did not throw`);
}

/** Python's overlap error for `label`, members `a` and `b`, with the `file` note if `files`. */
function overlap(label: string, a: string, b: string, files = false): string {
  const why = files
    ? " (a `file` slot reads `{uri, mime, name}`, so `file` overlaps `object`, `Any` and a " +
      "record of that shape)"
    : "";
  return (
    `${label}: members '${a}' and '${b}' can match the same value${why}; add a tag field ` +
    "every member declares as a single-value Literal, with a different value in each " +
    `member, e.g. kind: Literal(${a.toLowerCase()})`
  );
}

describe("resolution errors", () => {
  test("a member must be a record", () => {
    expect(typeForError("Union(Buy, Ticker)", trades({ Ticker: "str" }))).toBe(
      "Union(Buy, Ticker): member 'Ticker' is not a record",
    );
  });

  test.each(["str", "list(Buy)", "object", "Literal(a, b)"])(
    "a non-record member is rejected: %p",
    (member) => {
      expect(typeForError(`Union(Buy, ${member})`)).toBe(
        `Union(Buy, ${member}): member '${member}' is not a record`,
      );
    },
  );

  test("a nullable member is rejected", () => {
    expect(typeForError("Union(Buy, MaybeSell)", trades({ MaybeSell: "Optional(Sell)" }))).toBe(
      "Union(Buy, MaybeSell): member 'MaybeSell' is nullable; use Optional(Union(...)) instead",
    );
  });

  test("overlapping members are rejected", () => {
    const reg = readTypedefs({ Bull: { conviction: "float" }, Bear: { conviction: "float" } });
    expect(typeForError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("overlapping shapes that differ are rejected", () => {
    const reg = readTypedefs({
      Bull: { conviction: "float" },
      Bear: { conviction: "float", note: "Optional(str)" },
    });
    expect(typeForError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("an Optional Literal is not a tag", () => {
    const reg = readTypedefs({
      Bull: { side: "Optional(Literal(bull))", conviction: "float" },
      Bear: { side: "Optional(Literal(bear))", conviction: "float" },
    });
    expect(typeForError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("an unknown member", () => {
    expect(typeForError("Union(Buy, Nope)")).toBe("unknown type 'Nope'");
  });

  test("every member resolves before any is checked", () => {
    const reg = readTypedefs({ A: { x: "int" } });
    expect(typeForError("Union(Optional(A), Nope)", reg)).toBe("unknown type 'Nope'");
  });

  test("an alias chain names the member by its first alias", () => {
    expect(typeForError("Union(Buy, P)", trades({ P: "Q", Q: "Optional(Sell)" }))).toBe(
      "Union(Buy, P): member 'P' is nullable; use Optional(Union(...)) instead",
    );
    expect(typeForError("Union(Buy, P)", trades({ P: "Q", Q: "str" }))).toBe(
      "Union(Buy, P): member 'P' is not a record",
    );
    // A union between the alias and the member names the member as that union writes it.
    const reg = trades({ P: "Q", Q: "Union(X, Sell)", X: { action: "Literal(buy)", qty: "int" } });
    expect(typeForError("Union(Buy, P)", reg)).toBe(overlap("Union(Buy, P)", "Buy", "X"));
  });

  test("two differently named records of one shape overlap", () => {
    // The members are equal, so the error comes before the overlap check and has no
    // `file` note.
    const reg = readTypedefs({ F: { f: "file" }, G: { f: "file" } });
    expect(typeForError("Union(F, G)", reg)).toBe(overlap("Union(F, G)", "F", "G"));
  });

  test("overlapping records with a file field explain the file overlap", () => {
    const reg = readTypedefs({ F: { f: "file" }, G: { f: "object" } });
    expect(typeForError("Union(F, G)", reg)).toBe(overlap("Union(F, G)", "F", "G", true));
  });
});

/** Return the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: RawTypedefs): TypeCheckError {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("readTypedefs did not throw");
}

describe("eager checks in typedefs", () => {
  test("a recursive member is rejected", () => {
    const error = typedefsError({
      Leaf: { kind: "Literal(leaf)" },
      Node: { kind: "Literal(node)", child: "Union(Node, Leaf)" },
    });
    expect(error.message).toBe("typedef 'Node' field 'child': recursive type reference: 'Node'");
    expect(error.path).toEqual(["Node", "child"]);
  });

  test("an unused overlapping union fails in readTypedefs", () => {
    const error = typedefsError({
      Bull: { conviction: "float" },
      Bear: { conviction: "float" },
      Mood: "Union(Bull, Bear)",
    });
    expect(error.message).toBe(`typedef 'Mood': ${overlap("Union(Bull, Bear)", "Bull", "Bear")}`);
    expect(error.path).toEqual(["Mood"]);
  });

  test("an unused record with a bad union field fails in readTypedefs", () => {
    const error = typedefsError({
      ...TRADES,
      Ticker: "str",
      Order: { pick: "Union(Buy, Ticker)" },
    });
    expect(error.message).toBe(
      "typedef 'Order' field 'pick': Union(Buy, Ticker): member 'Ticker' is not a record",
    );
    expect(error.path).toEqual(["Order", "pick"]);
  });

  test("a malformed union field is prefixed", () => {
    const error = typedefsError({ ...TRADES, Order: { pick: "Union(Buy, Sell" } });
    expect(error.message).toBe(
      "typedef 'Order' field 'pick': malformed type expression 'Union(Buy, Sell': unbalanced " +
        "parentheses — if this type sits inside a `{...}` mapping, its comma split it; quote " +
        "it or use block style",
    );
    expect(error.path).toEqual(["Order", "pick"]);
  });

  test("an alias cycle through a union is rejected", () => {
    const error = typedefsError({ ...TRADES, Decision: "Union(Buy, Again)", Again: "Decision" });
    expect(error.message).toBe("alias cycle through 'Decision' in typedefs");
    expect(error.path).toEqual(["Decision"]);
  });

  test("a self-referencing union alias is rejected", () => {
    const error = typedefsError({ ...TRADES, Decision: "Union(Buy, Decision)" });
    expect(error.message).toBe("alias cycle through 'Decision' in typedefs");
    expect(error.path).toEqual(["Decision"]);
  });
});
