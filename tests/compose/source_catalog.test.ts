/** Immutable, occurrence-backed Compose source catalogs. Site paths are arrays. */

import { describe, expect, test } from "bun:test";

import {
  SourceCatalog,
  SourceCatalogBuilder,
  type SourceSitePath,
  scopeOrigin,
} from "../../src/compose/source/catalog.ts";
import { SourceAccessError } from "../../src/compose/source/common.ts";
import { SourceCursor } from "../../src/compose/source/cursor.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import * as sourcePackage from "../../src/compose/source/index.ts";
import type {
  NonTextKey,
  OccurrencePath,
  RepeatedKey,
  ScalarSource,
  TaggedNode,
} from "../../src/compose/source/index.ts";
import { DiagnosticFrame, OriginRef, SourceRange } from "../../src/diagnostics.ts";

/** Return the cursor at `value` in a one-key document. */
function valueCursor(): SourceCursor {
  return SourceDocument.parse("value: present\n").root.child("value");
}

/** Return the cursor at a two-line literal block `value`. */
function literalCursor(): SourceCursor {
  return SourceDocument.parse("value: |-\n  first\n  second\n").root.child("value");
}

/** Run `body`, counting its `SourceDocument.parse` calls; the original is restored after. */
function countParses<T>(body: () => T): { parses: number; result: T } {
  const original = SourceDocument.parse;
  let parses = 0;
  SourceDocument.parse = (text: string, label?: string) => {
    parses += 1;
    return original.call(SourceDocument, text, label);
  };
  try {
    const result = body();
    return { parses, result };
  } finally {
    SourceDocument.parse = original;
  }
}

describe("site keys", () => {
  test("are typed, escaped and deterministic", () => {
    const cursor = valueCursor();
    const paths: SourceSitePath[] = [
      ["nodes", 1, "input"],
      ["nodes", "1", "input"],
      ["nodes", "a/b"],
      ["nodes", "a~1b"],
      ["nodes", "a", "b"],
    ];

    const firstBuilder = new SourceCatalogBuilder();
    const first = paths.map((path) => firstBuilder.register(path, cursor));
    const secondBuilder = new SourceCatalogBuilder();
    const second = paths.map((path) => secondBuilder.register(path, cursor));

    expect(first).toEqual(second);
    expect(first.every((origin) => origin instanceof OriginRef)).toBe(true);
    expect(first.every((origin) => origin !== null && origin.key !== "")).toBe(true);
    expect(new Set(first.map((origin) => origin!.key)).size).toBe(paths.length);
  });

  test.each([
    [true, "boolean"],
    [1.5, "non-integer number 1.5"],
  ])("reject the ambiguous segment %p", (segment, got) => {
    expect(() =>
      new SourceCatalogBuilder().register([segment as unknown as string], valueCursor()),
    ).toThrow(
      new TypeError(`source site path segments must be text or integer values, got ${got}`),
    );
  });

  test("must be non-empty arrays", () => {
    const cursor = valueCursor();
    expect(() =>
      new SourceCatalogBuilder().register("value" as unknown as SourceSitePath, cursor),
    ).toThrow(new TypeError("source site path must be an array"));
    expect(() => new SourceCatalogBuilder().register([], cursor)).toThrow(
      new RangeError("source site path cannot be empty"),
    );
  });
});

describe("scoped origins", () => {
  test("are domain-separated, injective and deterministic", () => {
    const aliases = ["child", "a/b", "a~1b", "@scope", "子"];
    const oldKeys = ["local", "/s:root/s:node", "a/b", "a~1b", "@scope/s:uses"];
    const scopeAll = () =>
      aliases.flatMap((alias) =>
        oldKeys.map((oldKey) => scopeOrigin(alias, new OriginRef(oldKey))),
      );

    const first = scopeAll();
    const second = scopeAll();

    expect(first).toEqual(second);
    expect(new Set(first.map((origin) => origin.key)).size).toBe(
      aliases.length * oldKeys.length,
    );
    expect(first.every((origin) => origin.key.startsWith("@scope/"))).toBe(true);
    expect(first.every((origin) => !origin.key.startsWith("/"))).toBe(true);
  });

  test("nest the previous key as one opaque segment", () => {
    const inner = scopeOrigin("inner/alias", new OriginRef("old/~key"));
    const nested = scopeOrigin("outer~alias", inner);

    expect(nested).toEqual(scopeOrigin("outer~alias", new OriginRef(inner.key)));
    expect(nested).not.toEqual(scopeOrigin("outer/inner", new OriginRef("old/~key")));
    expect(nested).not.toEqual(scopeOrigin("outer~alias", new OriginRef("old/~key")));
  });

  test("escaping prevents segment boundary spoofing", () => {
    expect(scopeOrigin("a/s:origin/s:b", new OriginRef("c"))).not.toEqual(
      scopeOrigin("a", new OriginRef("b/s:origin/s:c")),
    );
  });

  test.each([
    ["an empty alias", "", new OriginRef("old"), RangeError],
    ["a null alias", null, new OriginRef("old"), TypeError],
    ["a text origin", "child", "old", TypeError],
  ] as const)("reject %s", (_case, alias, origin, error) => {
    expect(() =>
      scopeOrigin(alias as unknown as string, origin as unknown as OriginRef),
    ).toThrow(error);
  });

  test("revalidate the origin's key", () => {
    // An origin that skipped its constructor's check.
    const invalid = Object.assign(Object.create(OriginRef.prototype) as OriginRef, { key: "" });

    expect(() => scopeOrigin("child", invalid)).toThrow(
      new RangeError("origin key must be a non-empty string"),
    );
  });
});

describe("registration", () => {
  test("rejects a duplicate site", () => {
    const root = SourceDocument.parse("first: one\nsecond: two\n").root;
    const builder = new SourceCatalogBuilder();
    builder.register(["nodes", "worker", "input"], root.child("first"));

    let error: unknown;
    try {
      builder.register(["nodes", "worker", "input"], root.child("second"));
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(SourceAccessError);
    expect((error as SourceAccessError).message).toBe(
      "duplicate source catalog site ('nodes', 'worker', 'input')",
    );
    expect((error as SourceAccessError).range).toEqual(root.child("second").range);
    expect(() =>
      builder.register(["nodes", "worker", "input"], SourceCursor.virtual("source-free")),
    ).toThrow(SourceAccessError);
  });

  test("spells a one-segment duplicate as a one-element tuple", () => {
    const builder = new SourceCatalogBuilder();
    builder.register(["a"], valueCursor());

    expect(() => builder.register(["a"], valueCursor())).toThrow(
      "duplicate source catalog site ('a',)",
    );
  });

  test("rejects invalid resolver and registration inputs", () => {
    const cursor = valueCursor();
    const builder = new SourceCatalogBuilder();

    expect(() => builder.register(["value"], "not a cursor" as unknown as SourceCursor)).toThrow(
      new TypeError("source catalog registration requires a SourceCursor"),
    );
    expect(() => builder.register(["value"], cursor, "line" as "value")).toThrow(
      new TypeError("source catalog anchor must be 'value' or 'key'"),
    );
    expect(() => builder.freeze().resolve("not an origin" as unknown as OriginRef)).toThrow(
      new TypeError("source catalog resolve requires an OriginRef"),
    );
  });
});

describe("resolution", () => {
  test("value and key anchors resolve complete frames", () => {
    const text = "node:\n  field: value\n";
    const document = SourceDocument.parse(text, "flow.yaml");
    const cursor = document.root.child("node").child("field");
    const builder = new SourceCatalogBuilder();

    const valueOrigin = builder.register(["nodes", "node", "field", "value"], cursor);
    const keyOrigin = builder.register(["nodes", "node", "field", "key"], cursor, "key");
    const catalog = builder.freeze();

    expect(catalog).toBeInstanceOf(SourceCatalog);
    expect(catalog.resolve(valueOrigin!)).toEqual(
      new DiagnosticFrame({ label: "flow.yaml", text, range: new SourceRange(2, 10, 2, 15) }),
    );
    expect(catalog.resolve(keyOrigin!)).toEqual(
      new DiagnosticFrame({ label: "flow.yaml", text, range: new SourceRange(2, 3, 2, 8) }),
    );
    expect(catalog.resolve(new OriginRef("unknown"))).toBeNull();
  });

  test("duplicate pairs and aliases replay exact occurrences", () => {
    const text =
      "name: first\n" +
      "name: final\n" +
      "shared: &shared {value: 1}\n" +
      "left: *shared\n" +
      "right: *shared\n";
    const root = SourceDocument.parse(text, "aliases.yaml").root;
    const [firstName, finalName] = root.mappingOccurrences("name");
    const left = root.child("left");
    const right = root.child("right");
    const builder = new SourceCatalogBuilder();

    const origins = {
      firstName: builder.register(["names", 0], firstName!),
      finalName: builder.register(["names", 1], finalName!),
      leftValue: builder.register(["aliases", "left", "value"], left),
      rightValue: builder.register(["aliases", "right", "value"], right),
      leftKey: builder.register(["aliases", "left", "key"], left, "key"),
      rightKey: builder.register(["aliases", "right", "key"], right, "key"),
    };
    const catalog = builder.freeze();
    const rangeOf = (origin: OriginRef | null) => catalog.resolve(origin!)!.range;

    expect(rangeOf(origins.firstName)).toEqual(new SourceRange(1, 7, 1, 12));
    expect(rangeOf(origins.finalName)).toEqual(new SourceRange(2, 7, 2, 12));
    expect(rangeOf(origins.leftValue)).toEqual(new SourceRange(3, 9, 3, 27));
    expect(rangeOf(origins.rightValue)).toEqual(new SourceRange(3, 9, 3, 27));
    expect(rangeOf(origins.leftKey)).toEqual(new SourceRange(4, 1, 4, 5));
    expect(rangeOf(origins.rightKey)).toEqual(new SourceRange(5, 1, 5, 6));
  });

  test("projected and renamed cursors keep physical backing", () => {
    const renamedText = "original: old\n";
    const renamedDocument = SourceDocument.parse(renamedText, "renamed.yaml");
    const renamed = renamedDocument.root
      .renamed(new Map([["original", "displayed"]]))
      .child("displayed");

    const backingText = "backing: old\n";
    const backingDocument = SourceDocument.parse(backingText, "backing.yaml");
    const hostDocument = SourceDocument.parse("host: local\n", "host.yaml");
    const projected = hostDocument.root
      .projectMapping(
        new Map([["displayed", ["replacement", backingDocument.root.child("backing")]]]),
      )
      .child("displayed");

    const builder = new SourceCatalogBuilder();
    const renamedOrigin = builder.register(["renamed", "value"], renamed);
    const projectedValueOrigin = builder.register(["projected", "value"], projected);
    const projectedKeyOrigin = builder.register(["projected", "key"], projected, "key");
    const catalog = builder.freeze();

    expect(catalog.resolve(renamedOrigin!)).toEqual(
      new DiagnosticFrame({
        label: "renamed.yaml",
        text: renamedText,
        range: renamedDocument.root.child("original").range!,
      }),
    );
    expect(catalog.resolve(projectedValueOrigin!)).toEqual(
      new DiagnosticFrame({
        label: "backing.yaml",
        text: backingText,
        range: backingDocument.root.child("backing").range!,
      }),
    );
    expect(catalog.resolve(projectedKeyOrigin!)).toEqual(
      new DiagnosticFrame({
        label: "backing.yaml",
        text: backingText,
        range: backingDocument.root.child("backing").keyRange!,
      }),
    );
  });

  test("source-free and unanchored sites are ignored", () => {
    const document = SourceDocument.parse("items:\n  - first\n");
    const builder = new SourceCatalogBuilder();

    expect(builder.register(["virtual"], SourceCursor.virtual("value"))).toBeNull();
    expect(builder.register(["root-key"], document.root, "key")).toBeNull();
    expect(
      builder.register(["sequence-key"], document.root.child("items").child(0), "key"),
    ).toBeNull();
    expect(builder.register(["empty-value"], SourceDocument.parse("").root)).toBeNull();
    const reusedOrigin = builder.register(["virtual"], document.root.child("items").child(0));

    const catalog = builder.freeze();
    expect(catalog.resolve(reusedOrigin!)).not.toBeNull();
    expect(catalog.resolve(new OriginRef("not-registered"))).toBeNull();
  });

  test("does not reparse the source", () => {
    const { parses } = countParses(() => {
      const document = SourceDocument.parse("value: present\n");
      const builder = new SourceCatalogBuilder();
      const origin = builder.register(["value"], document.root.child("value"))!;
      const currentFrame = builder.resolveRegistered(origin);
      const catalog = builder.freeze();
      expect(currentFrame).toEqual(catalog.resolve(origin));
      expect(catalog.resolve(origin)).not.toBeNull();
      expect(catalog.resolve(origin)).not.toBeNull();
    });

    expect(parses).toBe(1);
  });
});

describe("snapshots", () => {
  test("freeze isolates snapshots", () => {
    const root = SourceDocument.parse("first: one\nsecond: two\n").root;
    const builder = new SourceCatalogBuilder();
    const firstOrigin = builder.register(["first"], root.child("first"))!;
    const firstSnapshot = builder.freeze();

    const secondOrigin = builder.register(["second"], root.child("second"))!;
    const secondSnapshot = builder.freeze();

    expect(firstSnapshot.resolve(firstOrigin)).not.toBeNull();
    expect(firstSnapshot.resolve(secondOrigin)).toBeNull();
    expect(secondSnapshot.resolve(firstOrigin)).not.toBeNull();
    expect(secondSnapshot.resolve(secondOrigin)).not.toBeNull();
  });

  test("membership is exact and snapshot-isolated", () => {
    const root = SourceDocument.parse("first: one\nsecond: two\n").root;
    const builder = new SourceCatalogBuilder();
    const firstOrigin = builder.register(["first"], root.child("first"))!;
    const firstSnapshot = builder.freeze();
    const secondOrigin = builder.register(["second"], root.child("second"))!;
    const secondSnapshot = builder.freeze();

    const unexpectedResolve = () => {
      throw new Error("membership must not replay a source occurrence");
    };
    firstSnapshot.resolve = unexpectedResolve;
    secondSnapshot.resolve = unexpectedResolve;

    expect(firstSnapshot.has(firstOrigin)).toBe(true);
    expect(firstSnapshot.has(new OriginRef(firstOrigin.key))).toBe(true);
    expect(firstSnapshot.has(secondOrigin)).toBe(false);
    expect(secondSnapshot.has(firstOrigin)).toBe(true);
    expect(secondSnapshot.has(secondOrigin)).toBe(true);
    expect(secondSnapshot.has(new OriginRef("unknown"))).toBe(false);
    expect(secondSnapshot.has("not-an-origin")).toBe(false);
  });

  test("builder lookup reads current sites without freezing", () => {
    const document = SourceDocument.parse("first: one\nsecond: two\n", "flow.yaml");
    const builder = new SourceCatalogBuilder();
    const first = builder.register(["first"], document.root.child("first"))!;
    builder.freeze = () => {
      throw new Error("registered-site lookup must not freeze");
    };

    expect(builder.resolveRegistered(first)!.range).toEqual(new SourceRange(1, 8, 1, 11));
    expect(builder.resolveRegistered(new OriginRef("unknown"))).toBeNull();
  });
});

describe("constructed lines", () => {
  test("a literal registers zero-based exact frames", () => {
    const text = "code: |-\r\n  first\r\n\r\n  second\r\n";
    const document = SourceDocument.parse(text, "literal.yaml");
    const cursor = document.root.child("code");
    const builder = new SourceCatalogBuilder();

    const origins = [0, 1, 2].map((index) =>
      builder.registerConstructedLine(["code", "line", index], cursor, index),
    );
    const catalog = builder.freeze();

    expect(origins.map((origin) => catalog.resolve(origin!))).toEqual([
      new DiagnosticFrame({ label: "literal.yaml", text, range: new SourceRange(2, 3, 2, 8) }),
      new DiagnosticFrame({ label: "literal.yaml", text, range: new SourceRange(3, 1, 3, 1) }),
      new DiagnosticFrame({ label: "literal.yaml", text, range: new SourceRange(4, 3, 4, 9) }),
    ]);
  });

  test("an unchanged projection retains literal provenance", () => {
    const text = "code: |-\n  first\n  second\n";
    const cursor = SourceDocument.parse(text, "projected.yaml").root.child("code");
    const unchanged = cursor.withValue(cursor.value);
    const builder = new SourceCatalogBuilder();

    const origin = builder.registerConstructedLine(["projected", "code", "line", 1], unchanged, 1);

    expect(builder.freeze().resolve(origin!)).toEqual(
      new DiagnosticFrame({ label: "projected.yaml", text, range: new SourceRange(3, 3, 3, 9) }),
    );
  });

  test.each([
    ["folded", "value: >-\n  first\n  second\n", null],
    ["quoted", 'value: "first\\nsecond"\n', null],
    ["plain-multiline", "value: first\n  second\n", null],
    ["virtual", null, "first\nsecond"],
    ["changed", "value: |-\n  first\n  second\n", "changed\nvalue"],
  ] as const)("%s: unavailable lines don't reserve the site", (_id, text, replacement) => {
    let cursor: SourceCursor;
    if (text === null) {
      cursor = SourceCursor.virtual(replacement);
    } else {
      cursor = SourceDocument.parse(text).root.child("value");
      if (replacement !== null) cursor = cursor.withValue(replacement);
    }
    const builder = new SourceCatalogBuilder();
    const sitePath = ["value", "line", 0];

    expect(builder.registerConstructedLine(sitePath, cursor, 0)).toBeNull();

    let wholeOrigin = builder.register(sitePath, cursor);
    if (wholeOrigin === null) {
      const fallback = SourceDocument.parse("fallback: located\n").root.child("fallback");
      wholeOrigin = builder.register(sitePath, fallback);
    }
    expect(builder.freeze().resolve(wholeOrigin!)).not.toBeNull();
  });

  test.each([[true], ["0"], [1.5], [null]])("reject the non-integer index %p", (index) => {
    expect(() =>
      new SourceCatalogBuilder().registerConstructedLine(
        ["value", "bad-index"],
        literalCursor(),
        index as unknown as number,
      ),
    ).toThrow(new TypeError("constructed line index must be an integer"));
  });

  test.each([
    [-1, "constructed line index must be non-negative"],
    [2, "constructed line index 2 is out of range"],
  ])("reject the invalid index %p", (index, message) => {
    expect(() =>
      new SourceCatalogBuilder().registerConstructedLine(
        ["value", "bad-index"],
        literalCursor(),
        index,
      ),
    ).toThrow(new SourceAccessError(message));
  });

  test("resolution rejects a malformed negative selector", () => {
    const document = SourceDocument.parse("value: |-\n  first\n  second\n");
    const cursor = document.root.child("value");
    const origin = new OriginRef("malformed");
    const catalog = new SourceCatalog(
      new Map([[origin.key, { document, occurrencePath: cursor.path, selector: { index: -1 } }]]),
    );

    expect(catalog.resolve(origin)).toBeNull();
  });

  test("a snapshot keeps its own copy of a mutable input site", () => {
    const document = SourceDocument.parse("value: |-\n  first\n  second\n");
    const cursor = document.root.child("value");
    const origin = new OriginRef("site");
    const occurrencePath: { key?: string; index?: number }[] = cursor.path.map((step) => ({
      ...step,
    }));
    const selector = { index: 1 };
    const site = { document, occurrencePath: occurrencePath as OccurrencePath, selector };
    const catalog = new SourceCatalog(new Map([[origin.key, site]]));
    const before = catalog.resolve(origin);

    occurrencePath[0]!.key = "missing";
    occurrencePath.push({ index: 0 });
    selector.index = 0;

    expect(before!.range).toEqual(new SourceRange(3, 3, 3, 9));
    expect(catalog.resolve(origin)).toEqual(before);
  });

  test("duplicate detection crosses selector kinds", () => {
    const cursor = literalCursor();
    const folded = SourceDocument.parse("value: >-\n  first\n  second\n").root.child("value");
    const sitePath = ["value", "site"];
    const lineFirst = new SourceCatalogBuilder();
    lineFirst.registerConstructedLine(sitePath, cursor, 0);

    expect(() => lineFirst.register(sitePath, cursor)).toThrow(SourceAccessError);
    expect(() => lineFirst.registerConstructedLine(sitePath, cursor, 1)).toThrow(
      SourceAccessError,
    );
    expect(() => lineFirst.registerConstructedLine(sitePath, folded, 0)).toThrow(
      SourceAccessError,
    );

    const wholeFirst = new SourceCatalogBuilder();
    wholeFirst.register(sitePath, cursor);
    expect(() => wholeFirst.registerConstructedLine(sitePath, cursor, 0)).toThrow(
      SourceAccessError,
    );
  });

  test("alias occurrences replay the shared definition", () => {
    const text = "shared: &code |-\n  first\n  second\nleft: *code\nright: *code\n";
    const root = SourceDocument.parse(text, "aliases.yaml").root;
    const builder = new SourceCatalogBuilder();

    const left = builder.registerConstructedLine(
      ["aliases", "left", "line", 1],
      root.child("left"),
      1,
    );
    const right = builder.registerConstructedLine(
      ["aliases", "right", "line", 0],
      root.child("right"),
      0,
    );
    const catalog = builder.freeze();

    expect(catalog.resolve(left!)!.range).toEqual(new SourceRange(3, 3, 3, 9));
    expect(catalog.resolve(right!)!.range).toEqual(new SourceRange(2, 3, 2, 8));
  });

  test("keys survive formatting while frames move", () => {
    const firstText = "wrapper:\n  code: |-\n    first\n    second\n";
    const secondText = "# shifted\nwrapper:\n    code: |-\n      first\n      second\n";
    const firstCursor = SourceDocument.parse(firstText, "first.yaml")
      .root.child("wrapper")
      .child("code");
    const secondCursor = SourceDocument.parse(secondText, "second.yaml")
      .root.child("wrapper")
      .child("code");
    const sitePath = ["nodes", "worker", "code", "line", 1];
    const firstBuilder = new SourceCatalogBuilder();
    const secondBuilder = new SourceCatalogBuilder();

    const firstOrigin = firstBuilder.registerConstructedLine(sitePath, firstCursor, 1)!;
    const secondOrigin = secondBuilder.registerConstructedLine(sitePath, secondCursor, 1)!;

    expect(firstOrigin).toEqual(secondOrigin);
    expect(firstBuilder.freeze().resolve(firstOrigin)).toEqual(
      new DiagnosticFrame({
        label: "first.yaml",
        text: firstText,
        range: new SourceRange(4, 5, 4, 11),
      }),
    );
    expect(secondBuilder.freeze().resolve(secondOrigin)).toEqual(
      new DiagnosticFrame({
        label: "second.yaml",
        text: secondText,
        range: new SourceRange(5, 7, 5, 13),
      }),
    );
  });

  test("resolution does not reparse the source", () => {
    const { parses } = countParses(() => {
      const document = SourceDocument.parse("value: |-\n  first\n  second\n");
      const builder = new SourceCatalogBuilder();
      const origin = builder.registerConstructedLine(
        ["value", "line", 1],
        document.root.child("value"),
        1,
      );
      const catalog = builder.freeze();
      expect(catalog.resolve(origin!)).not.toBeNull();
      expect(catalog.resolve(origin!)).not.toBeNull();
    });

    expect(parses).toBe(1);
  });
});

describe("the package index", () => {
  test("exports the public names, and not the builder or scoping", () => {
    expect(Object.keys(sourcePackage).sort()).toEqual([
      "SourceAccessError",
      "SourceCatalog",
      "SourceCursor",
      "SourceDocument",
      "SourceSyntaxError",
      "parseYamlLiteral",
      "readPlainScalar",
    ]);
  });

  test("exports the public types", () => {
    // Compile-time only: the type check fails if a type isn't exported.
    const types: [ScalarSource?, TaggedNode?, OccurrencePath?, NonTextKey?, RepeatedKey?] = [];
    expect(types).toEqual([]);
  });
});
