/**
 * `Union(...)` in the type grammar and its resolution, a port of the parsing and resolution
 * tests of `tests/engine/test_union_typedefs.py`, with registries built by hand. Later:
 * `test_recursive_member_rejected` and the eager checks (with `readTypedefs`) and the flow
 * loading tests.
 */

import { describe, expect, test } from "bun:test";

import {
  parseType,
  resolveType,
  type TypeDef,
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

/**
 * Build a registry by hand, without checks: a mapping is a record, `Literal(...)` text a
 * variant and other text an alias.
 */
function registry(raw: RawTypedefs): TypeRegistry {
  return new Map(
    Object.entries(raw).map(([name, value]): [string, TypeDef] => {
      if (typeof value !== "string") {
        return [name, { kind: "record", fields: new Map(Object.entries(value)) }];
      }
      const target = parseType(value);
      if (target.kind === "enum") return [name, { kind: "variant", tags: target.tags }];
      return [name, { kind: "alias", target }];
    }),
  );
}

/** The trades registry with `extra` typedefs added. */
function trades(extra: RawTypedefs = {}): TypeRegistry {
  return registry({ ...TRADES, ...extra });
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
    const reg = registry({
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
function resolveError(text: string, reg: TypeRegistry = trades()): string {
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
    expect(resolveError("Union(Buy, Ticker)", trades({ Ticker: "str" }))).toBe(
      "Union(Buy, Ticker): member 'Ticker' is not a record",
    );
  });

  test.each(["str", "list(Buy)", "object", "Literal(a, b)"])(
    "a non-record member is rejected: %p",
    (member) => {
      expect(resolveError(`Union(Buy, ${member})`)).toBe(
        `Union(Buy, ${member}): member '${member}' is not a record`,
      );
    },
  );

  test("a nullable member is rejected", () => {
    expect(resolveError("Union(Buy, MaybeSell)", trades({ MaybeSell: "Optional(Sell)" }))).toBe(
      "Union(Buy, MaybeSell): member 'MaybeSell' is nullable; use Optional(Union(...)) instead",
    );
  });

  test("overlapping members are rejected", () => {
    const reg = registry({ Bull: { conviction: "float" }, Bear: { conviction: "float" } });
    expect(resolveError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("overlapping shapes that differ are rejected", () => {
    const reg = registry({
      Bull: { conviction: "float" },
      Bear: { conviction: "float", note: "Optional(str)" },
    });
    expect(resolveError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("an Optional Literal is not a tag", () => {
    const reg = registry({
      Bull: { side: "Optional(Literal(bull))", conviction: "float" },
      Bear: { side: "Optional(Literal(bear))", conviction: "float" },
    });
    expect(resolveError("Union(Bull, Bear)", reg)).toBe(
      overlap("Union(Bull, Bear)", "Bull", "Bear"),
    );
  });

  test("an unknown member", () => {
    expect(resolveError("Union(Buy, Nope)")).toBe("unknown type 'Nope'");
  });

  test("every member resolves before any is checked", () => {
    const reg = registry({ A: { x: "int" } });
    expect(resolveError("Union(Optional(A), Nope)", reg)).toBe("unknown type 'Nope'");
  });

  test("an alias chain names the member by its first alias", () => {
    expect(resolveError("Union(Buy, P)", trades({ P: "Q", Q: "Optional(Sell)" }))).toBe(
      "Union(Buy, P): member 'P' is nullable; use Optional(Union(...)) instead",
    );
    expect(resolveError("Union(Buy, P)", trades({ P: "Q", Q: "str" }))).toBe(
      "Union(Buy, P): member 'P' is not a record",
    );
    // A union between the alias and the member names the member as that union writes it.
    expect(resolveError("Union(Buy, P)", trades({ P: "Q", Q: "Union(X, Sell)", X: "int" }))).toBe(
      "Union(Buy, P): member 'X' is not a record",
    );
  });

  test("two differently named records of one shape overlap", () => {
    // The members are equal, so the error comes before the overlap check and has no
    // `file` note.
    const reg = registry({ F: { f: "file" }, G: { f: "file" } });
    expect(resolveError("Union(F, G)", reg)).toBe(overlap("Union(F, G)", "F", "G"));
  });

  test("overlapping records with a file field explain the file overlap", () => {
    const reg = registry({ F: { f: "file" }, G: { f: "object" } });
    expect(resolveError("Union(F, G)", reg)).toBe(overlap("Union(F, G)", "F", "G", true));
  });
});
