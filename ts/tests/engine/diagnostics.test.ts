/**
 * Contracts for source-neutral diagnostic values and origin resolution, a port of
 * `tests/engine/test_diagnostics.py`, plus the text helpers yian's messages need.
 */

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  compareCodeUnits,
  Diagnostic,
  DiagnosticFrame,
  FAILURE_CODES,
  Failure,
  type FailureCode,
  getCloseMatches,
  OriginRef,
  type OriginResolver,
  ResolverTable,
  reprText,
  SourceRange,
} from "../../src/diagnostics.ts";
import { importStatements } from "./source_scan.ts";

const EXPECTED_FAILURE_CODES: readonly FailureCode[] = [
  "binding.evaluate",
  "binding.type",
  "contract.input",
  "assertion.failed",
  "node.execute",
  "contract.node_result",
  "contract.output",
  "contract.route",
  "contract.growth",
  "graph.terminal_unreachable",
  "loop.max_exceeded",
  "agent.max_tool_iterations",
  "provider.auth",
  "agent.output_invalid",
  "provider.rate_limit",
  "provider.quota",
  "provider.unavailable",
  "provider.context_length",
  "provider.refused",
  "condition.evaluate",
];

/** Return what `fn` throws; fail the test if it returns. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

describe("compareCodeUnits", () => {
  test("sorts by UTF-16 code units", () => {
    // UTF-16 order puts the surrogate pair of U+1F600 before U+E000.
    expect(["b", "a", "B", "\u{1F600}", "\uE000", "aa"].sort(compareCodeUnits)).toEqual([
      "B",
      "a",
      "aa",
      "b",
      "\u{1F600}",
      "\uE000",
    ]);
  });

  test("returns zero for equal strings and a negative number for a smaller one", () => {
    expect(compareCodeUnits("a", "a")).toBe(0);
    expect(Math.sign(compareCodeUnits("a", "b"))).toBe(-1);
  });
});

describe("reprText", () => {
  // `expected` is CPython 3.12.14's `repr(input)`; regenerate with
  // `print(json.dumps([[s, repr(s)] for s in inputs]))`, where `inputs` is the first column.
  const cases: [string, string][] = [
    ["", "''"],
    ["abc", "'abc'"],
    ["it's", "\"it's\""],
    ["say \"hi\"", "'say \"hi\"'"],
    ["both ' and \"", "'both \\' and \"'"],
    ["back\\slash", "'back\\\\slash'"],
    ["tab\there", "'tab\\there'"],
    ["nl\ncr\r", "'nl\\ncr\\r'"],
    ["\u0000\u0001\u001f\u007f", "'\\x00\\x01\\x1f\\x7f'"],
    ["\u0080\u009f\u00a0\u00ad", "'\\x80\\x9f\\xa0\\xad'"],
    ["caf\u00e9", "'caf\u00e9'"],
    ["e\u0301", "'e\u0301'"],
    ["\u200b\u2028\u2029\u3000", "'\\u200b\\u2028\\u2029\\u3000'"],
    ["\u0378", "'\\u0378'"],
    ["\ue000", "'\\ue000'"],
    ["\ud800", "'\\ud800'"],
    ["\udfff", "'\\udfff'"],
    ["\ud83d\ude00", "'\ud83d\ude00'"],
    ["\udbff\udfff", "'\\U0010ffff'"],
    ["\udb40\udc01", "'\\U000e0001'"],
    ["\udb40\udd00", "'\udb40\udd00'"],
    ["\ufeff", "'\\ufeff'"],
    ["\uffff", "'\\uffff'"],
    ["\ud7ff", "'\\ud7ff'"],
    ["\ud840\udc00", "'\ud840\udc00'"],
    ["\ud884\udf4a", "'\ud884\udf4a'"],
    ["\ud884\udf4b", "'\\U0003134b'"],
    ["x' \"y\\", "'x\\' \"y\\\\'"],
  ];

  test.each(cases)("reprText(%j) is %j", (input, expected) => {
    expect(reprText(input)).toBe(expected);
  });

  test("its printable table is CPython 3.12.14's str.isprintable", () => {
    // CPython 3.12.14's
    // hashlib.sha256(''.join('1' if chr(c).isprintable() else '0'
    //                        for c in range(0x110000)).encode()).hexdigest()
    // `\` is printable but escaped, and surrogates are never printable, as in Python.
    const bits: string[] = [];
    for (let cp = 0; cp < 0x110000; cp++) {
      const ch = String.fromCodePoint(cp);
      if (ch === "\\") bits.push("1");
      else if (cp >= 0xd800 && cp <= 0xdfff) bits.push("0");
      else bits.push(reprText(ch).slice(1, -1) === ch ? "1" : "0");
    }
    expect(createHash("sha256").update(bits.join("")).digest("hex")).toBe(
      "0326b64c15e91a0583a1d9f8bc53f4bb534adc91a5ae096b55adaf19525ed98d",
    );
  });
});

describe("getCloseMatches", () => {
  // Strings of 200 or more code points, long enough for the autojunk rule.
  const a120 = "a".repeat(120);
  const b120 = "b".repeat(120);
  const c120 = "c".repeat(120);
  const b240 = "b".repeat(240);
  const ab120 = "ab".repeat(120);
  const xyz69 = "xyz".repeat(69);
  const xyz70 = "xyz".repeat(70);
  const zyx70 = "zyx".repeat(70);
  // `expected` is CPython 3.12.14's `difflib.get_close_matches(word, possibilities, n, cutoff)`.
  const cases: [string, string[], number, number, string[]][] = [
    ["appel", ["ape", "apple", "peach", "puppy"], 3, 0.6, ["apple", "ape"]],
    ["wheel", ["while", "when", "whale", "wheat"], 3, 0.6, ["when", "while", "wheat"]],
    ["place_ordr", ["place_order", "list_orders", "cancel"], 1, 0.6, ["place_order"]],
    ["abc", ["abc", "acb", "bac", "cab", "bca", "cba"], 6, 0.0, ["abc", "cab", "bca", "bac", "acb", "cba"]],
    ["ab", ["ba", "ab", "ba"], 3, 0.0, ["ab", "ba", "ba"]],
    ["", ["", "a"], 3, 0.0, ["", "a"]],
    ["x", [], 3, 0.6, []],
    ["Strng", ["String", "string", "str", "Sting"], 2, 0.6, ["String", "Sting"]],
    ["aaaa", ["aaab", "abaa", "baaa"], 3, 0.0, ["baaa", "aaab", "abaa"]],
    [a120 + b120, [a120 + c120, b240, ab120], 3, 0.0, [a120 + c120, ab120, b240]],
    [xyz70, [xyz69 + "q", zyx70], 2, 0.0, [xyz69 + "q", zyx70]],
    ["lookup", ["look_up", "lookups", "Lookup", "kool"], 4, 0.6, ["lookups", "look_up", "Lookup"]],
    ["int", ["Int", "integer", "in", "nit"], 4, 0.5, ["in", "nit", "Int", "integer"]],
    ["\ud83d\ude00\ud83d\ude01", ["\ud83d\ude00", "\ud83d\ude01x", "\ud83d"], 3, 0.0, ["\ud83d\ude00", "\ud83d\ude01x", "\ud83d"]],
    // Python gives `["a", "\ud83d\ude00b", "\ue000a"]`; TS breaks the 0.5 tie by UTF-16
    // code units.
    ["\ud83d\ude00a", ["\ud83d\ude00b", "\ue000a", "a"], 3, 0.0, ["a", "\ue000a", "\ud83d\ude00b"]],
  ];

  test.each(cases)(
    "getCloseMatches(%j, %j, %d, %d) is %j",
    (word, possibilities, n, cutoff, expected) => {
      expect(getCloseMatches(word, possibilities, n, cutoff)).toEqual(expected);
    },
  );

  test("defaults to n = 3 and cutoff = 0.6 and takes any iterable", () => {
    expect(getCloseMatches("appel", ["ape", "apple", "peach", "puppy"])).toEqual(["apple", "ape"]);
    expect(getCloseMatches("appel", new Set(["ape", "apple", "peach", "puppy"]))).toEqual([
      "apple",
      "ape",
    ]);
  });

  test.each([
    [0, 0.6, "n must be > 0: 0"],
    [3, 1.5, "cutoff must be in [0.0, 1.0]: 1.5"],
    [3, -0.1, "cutoff must be in [0.0, 1.0]: -0.1"],
  ])("rejects n = %d, cutoff = %d", (n, cutoff, message) => {
    const error = thrown(() => getCloseMatches("appel", ["apple"], n, cutoff));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(message);
  });
});

describe("SourceRange", () => {
  test("keeps its four fields", () => {
    const range = new SourceRange(2, 3, 4, 5);

    expect(Object.keys(range)).toEqual(["startLine", "startCol", "endLine", "endCol"]);
    expect([range.startLine, range.startCol, range.endLine, range.endCol]).toEqual([2, 3, 4, 5]);
    if (false) {
      // @ts-expect-error fields are readonly
      range.endCol = 1;
    }
  });

  const block = "a:\n  b: 1\nc: 2\n";
  const cases: [string, SourceRange, string, number[]][] = [
    ["a single-line span", new SourceRange(1, 2, 1, 5), "abc\n", [1]],
    ["an end in the next line's indentation", new SourceRange(1, 1, 3, 1), block, [1, 2]],
    ["an end after real text", new SourceRange(1, 1, 2, 4), block, [1, 2]],
    ["a zero-width caret", new SourceRange(2, 3, 2, 3), "a\nbcd\ne\n", [2]],
    ["a start outside the text", new SourceRange(9, 1, 9, 2), "a\nb\nc\n", [9]],
    ["an end beyond the last line", new SourceRange(2, 1, 9, 1), "a\nb\nc\n", [2, 3]],
    ["\\r\\n line breaks", new SourceRange(1, 1, 3, 1), "a:\r\n  b: 1\r\nc: 2\r\n", [1, 2]],
    ["\\r line breaks", new SourceRange(1, 1, 3, 1), "a:\r  b: 1\rc: 2\r", [1, 2]],
    // Python gives [2, 3]: its `splitlines` also breaks at U+2028 (`rule: columns`).
    ["U+2028 inside a line", new SourceRange(2, 1, 9, 1), "a\u2028b\nc", [2]],
    // Python gives [1, 2]: its `strip` keeps U+FEFF, JS `trim` strips it (`rule: text`).
    ["a U+FEFF-only prefix", new SourceRange(1, 1, 2, 2), "a\n\ufeff\n", [1]],
    // Python gives [1]: its `strip` removes U+001F, JS `trim` keeps it (`rule: text`).
    ["a U+001F-only prefix", new SourceRange(1, 1, 2, 2), "a\n\u001f\n", [1, 2]],
  ];

  test.each(cases)("coveredLines: %s", (_, range, text, expected) => {
    expect(range.coveredLines(text)).toEqual(expected);
  });
});

describe("DiagnosticFrame", () => {
  test("is frozen presentation data", () => {
    const range = new SourceRange(2, 3, 2, 8);
    const frame = new DiagnosticFrame({
      label: "flow.yaml",
      text: "nodes:\n  ask: {}\n",
      range,
      note: "inside call 'review'",
    });

    expect(Object.keys(frame)).toEqual(["label", "text", "range", "note"]);
    expect(frame.label).toBe("flow.yaml");
    expect(frame.text).toBe("nodes:\n  ask: {}\n");
    expect(frame.range).toBe(range);
    expect(frame.note).toBe("inside call 'review'");
    expect(new DiagnosticFrame({ label: "flow.yaml", text: "", range }).note).toBeNull();
    if (false) {
      // @ts-expect-error fields are readonly
      frame.note = null;
    }
  });
});

describe("Diagnostic", () => {
  test("has one frozen presentation shape", () => {
    const frame = new DiagnosticFrame({
      label: "flow.yaml",
      text: "nodes:\n",
      range: new SourceRange(1, 1, 1, 7),
      note: "outer context",
    });
    const frames = [frame];
    const notes = ["try another value"];
    const diagnostic = new Diagnostic({
      code: "node.execute",
      message: "boom",
      frames,
      notes,
      nodeId: "work",
      causeType: "ValueError",
      traceback: "Traceback ...",
    });
    frames.push(frame);
    notes.push("changed");

    expect(Object.keys(diagnostic)).toEqual([
      "code",
      "message",
      "frames",
      "notes",
      "nodeId",
      "causeType",
      "traceback",
    ]);
    expect(diagnostic.code).toBe("node.execute");
    expect(diagnostic.message).toBe("boom");
    expect(diagnostic.frames).toEqual([frame]);
    expect(diagnostic.frames[0]).toBe(frame);
    expect(diagnostic.notes).toEqual(["try another value"]);
    expect(diagnostic.nodeId).toBe("work");
    expect(diagnostic.causeType).toBe("ValueError");
    expect(diagnostic.traceback).toBe("Traceback ...");
    if (false) {
      // @ts-expect-error fields are readonly
      diagnostic.message = "x";
      // @ts-expect-error the frames are a readonly array
      diagnostic.frames.push(frame);
      // @ts-expect-error the notes are a readonly array
      diagnostic.notes.push("x");
    }
  });

  test("defaults its optional fields", () => {
    const diagnostic = new Diagnostic({ code: "node.execute", message: "" });

    expect(diagnostic.frames).toEqual([]);
    expect(diagnostic.notes).toEqual([]);
    expect(diagnostic.nodeId).toBeNull();
    expect(diagnostic.causeType).toBeNull();
    expect(diagnostic.traceback).toBeNull();
  });
});

describe("Failure", () => {
  test("has one closed source-neutral shape", () => {
    const outer = new OriginRef("root:call");
    const operation = new OriginRef("child:input:topic");
    const failure = new Failure({
      code: "contract.input",
      message: "expected integer",
      nodeId: "review/score",
      trace: [outer, operation, operation],
      causeType: "TypeCheckError",
      traceback: "Traceback ...",
    });

    expect([...FAILURE_CODES]).toEqual([...EXPECTED_FAILURE_CODES]);
    for (const code of FAILURE_CODES) expect(code).toMatch(/^[a-z]+\.[a-z_]+$/);
    expect(Object.keys(failure)).toEqual([
      "code",
      "message",
      "nodeId",
      "trace",
      "causeType",
      "traceback",
    ]);
    expect(failure.code).toBe("contract.input");
    expect(failure.message).toBe("expected integer");
    expect(failure.nodeId).toBe("review/score");
    expect(failure.trace).toEqual([outer, operation, operation]);
    expect(failure.trace[0]).toBe(outer);
    expect(failure.causeType).toBe("TypeCheckError");
    expect(failure.traceback).toBe("Traceback ...");
    if (false) {
      // @ts-expect-error fields are readonly
      failure.message = "changed";
      // @ts-expect-error the trace is a readonly array
      failure.trace.push(outer);
    }
  });

  test("copies its trace", () => {
    const trace = [new OriginRef("root:node")];
    const failure = new Failure({ code: "node.execute", message: "boom", nodeId: "work", trace });
    trace.push(new OriginRef("root:other"));

    expect(failure.trace).toEqual([new OriginRef("root:node")]);
  });

  test("accepts every closed code and source-free optional values", () => {
    for (const code of EXPECTED_FAILURE_CODES) {
      const failure = new Failure({
        code,
        message: "",
        nodeId: null,
        causeType: "",
        traceback: "",
      });

      expect(failure.code).toBe(code);
      expect(failure.trace).toEqual([]);
      expect(failure.causeType).toBe("");
      expect(failure.traceback).toBe("");
    }

    const bare = new Failure({ code: "node.execute", message: "", nodeId: "" });
    expect(bare.nodeId).toBe("");
    expect(bare.trace).toEqual([]);
    expect(bare.causeType).toBeNull();
    expect(bare.traceback).toBeNull();
  });

  test.each([
    [{ code: "unknown" }, RangeError, "unknown failure code 'unknown'"],
    [{ code: 1 }, TypeError, "failure code must be a string"],
    [{ message: null }, TypeError, "failure message must be a string"],
    [{ nodeId: 1 }, TypeError, "failure node_id must be a string or null"],
    [{ trace: "x" }, TypeError, "failure trace must be an array"],
    [{ trace: ["origin"] }, TypeError, "failure trace item must be an OriginRef"],
    [{ causeType: 1 }, TypeError, "failure cause_type must be a string or null"],
    [{ traceback: 1 }, TypeError, "failure traceback must be a string or null"],
  ] as const)("rejects malformed fields: %j", (changes, errorType, message) => {
    const options = { code: "node.execute", message: "boom", nodeId: "work", ...changes };
    const error = thrown(
      () => new Failure(options as unknown as ConstructorParameters<typeof Failure>[0]),
    );

    expect(error).toBeInstanceOf(errorType);
    expect((error as Error).message).toBe(message);
  });

  test("toJSON keeps every key, nulls included, in field order", () => {
    const full = new Failure({
      code: "contract.input",
      message: "expected integer",
      nodeId: "review/score",
      trace: [new OriginRef("root:call"), new OriginRef("child:input:topic")],
      causeType: "TypeError",
      traceback: "TypeError: x",
    });
    const bare = new Failure({ code: "node.execute", message: "boom", nodeId: null });

    expect(full.toJSON()).toEqual({
      code: "contract.input",
      message: "expected integer",
      node_id: "review/score",
      trace: [{ key: "root:call" }, { key: "child:input:topic" }],
      cause_type: "TypeError",
      traceback: "TypeError: x",
    });
    expect(bare.toJSON()).toEqual({
      code: "node.execute",
      message: "boom",
      node_id: null,
      trace: [],
      cause_type: null,
      traceback: null,
    });
    expect(Object.keys(bare.toJSON())).toEqual([
      "code",
      "message",
      "node_id",
      "trace",
      "cause_type",
      "traceback",
    ]);
    expect(Object.keys(full.toJSON().trace[0]!)).toEqual(["key"]);
  });
});

describe("OriginRef", () => {
  test("is a validated frozen single key", () => {
    const origin = new OriginRef("root:node");

    expect(origin.key).toBe("root:node");
    expect(Object.keys(origin)).toEqual(["key"]);
    if (false) {
      // @ts-expect-error the key is readonly
      origin.key = "changed";
    }

    for (const invalid of ["", null, 0, false]) {
      const error = thrown(() => new OriginRef(invalid as unknown as string));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toBe("origin key must be a non-empty string");
    }
  });
});

describe("OriginResolver", () => {
  test("is structural and may return null", () => {
    // Any source front end can implement resolution without a compose type.
    const resolver: OriginResolver = {
      resolve(origin) {
        if (origin.key !== "known") return null;
        return new DiagnosticFrame({
          label: "memory",
          text: "value\n",
          range: new SourceRange(1, 1, 1, 6),
        });
      },
    };

    expect(resolver.resolve(new OriginRef("known"))).not.toBeNull();
    expect(resolver.resolve(new OriginRef("missing"))).toBeNull();
  });
});

describe("ResolverTable", () => {
  test("routes each new origin to one exact delegate origin", () => {
    // One table can route colliding delegate-local keys without precedence.
    const firstFrame = new DiagnosticFrame({
      label: "first",
      text: "first\n",
      range: new SourceRange(1, 1, 1, 6),
    });
    const secondFrame = new DiagnosticFrame({
      label: "second",
      text: "second\n",
      range: new SourceRange(1, 1, 1, 7),
    });
    const calls: [string, OriginRef][] = [];

    class Resolver implements OriginResolver {
      readonly name: string;
      readonly frame: DiagnosticFrame;

      constructor(name: string, frame: DiagnosticFrame) {
        this.name = name;
        this.frame = frame;
      }

      resolve(origin: OriginRef): DiagnosticFrame | null {
        calls.push([this.name, origin]);
        return origin.key === "local" ? this.frame : null;
      }
    }

    const firstNew = new OriginRef("call/a");
    const secondNew = new OriginRef("call/b");
    const table = ResolverTable.fromEntries([
      [firstNew, new Resolver("first", firstFrame), new OriginRef("local")],
      [secondNew, new Resolver("second", secondFrame), new OriginRef("local")],
    ]);

    expect(table.size).toBe(2);
    expect(table.has(firstNew)).toBe(true);
    expect(table.has(secondNew)).toBe(true);
    expect(table.has(new OriginRef("missing"))).toBe(false);
    expect(table.has("call/a" as unknown as OriginRef)).toBe(false);
    expect(table.resolve(firstNew)).toBe(firstFrame);
    expect(table.resolve(secondNew)).toBe(secondFrame);
    expect(table.resolve(new OriginRef("missing"))).toBeNull();
    expect(calls).toEqual([
      ["first", new OriginRef("local")],
      ["second", new OriginRef("local")],
    ]);
  });

  test("merge is strict and preserves delegate behavior", () => {
    // Merging uses exact disjoint routes and never adds fallback semantics.
    class Resolver implements OriginResolver {
      readonly label: string;

      constructor(label: string) {
        this.label = label;
      }

      resolve(origin: OriginRef): DiagnosticFrame | null {
        if (origin.key !== "old") return null;
        return new DiagnosticFrame({
          label: this.label,
          text: "value\n",
          range: new SourceRange(1, 1, 1, 6),
        });
      }
    }

    const first = ResolverTable.fromEntries([
      [new OriginRef("first"), new Resolver("first"), new OriginRef("old")],
    ]);
    const second = ResolverTable.fromEntries([
      [new OriginRef("second"), new Resolver("second"), new OriginRef("old")],
    ]);

    const merged = ResolverTable.merge(first, second);

    expect(merged.size).toBe(2);
    expect(merged.resolve(new OriginRef("first"))?.label).toBe("first");
    expect(merged.resolve(new OriginRef("second"))?.label).toBe("second");
    expect(ResolverTable.merge().size).toBe(0);
    expect(new ResolverTable().size).toBe(0);
  });

  test("rejects duplicate routes before information is lost", () => {
    // Duplicate public keys fail even when their resolution targets are identical.
    const resolver: OriginResolver = { resolve: () => null };
    const same = new OriginRef("same");
    const old = new OriginRef("old");

    /** A distinct class that still names the same opaque identity. */
    class DerivedOrigin extends OriginRef {}

    const duplicate = thrown(() =>
      ResolverTable.fromEntries([
        [same, resolver, old],
        [new DerivedOrigin("same"), resolver, old],
      ]),
    );
    expect(duplicate).toBeInstanceOf(RangeError);
    expect((duplicate as Error).message).toBe("duplicate resolver table origin 'same'");

    const first = ResolverTable.fromEntries([[same, resolver, old]]);
    const second = ResolverTable.fromEntries([[same, resolver, old]]);
    const merged = thrown(() => ResolverTable.merge(first, second));
    expect(merged).toBeInstanceOf(RangeError);
    expect((merged as Error).message).toBe("duplicate resolver table origin 'same'");
  });

  test("snapshots its entries and propagates delegate failures", () => {
    // Delegates retain authority, so their errors reach the caller unchanged.
    const resolver: OriginResolver = {
      resolve(origin) {
        throw new Error(`cannot resolve ${origin.key}`);
      },
    };
    const entries: [OriginRef, OriginResolver, OriginRef][] = [
      [new OriginRef("new"), resolver, new OriginRef("old")],
    ];
    const table = ResolverTable.fromEntries(entries);
    entries.length = 0;

    expect(table.size).toBe(1);
    expect(() => table.resolve(new OriginRef("new"))).toThrow(new Error("cannot resolve old"));
  });

  test("rejects delegate results outside the OriginResolver contract", () => {
    const resolver = { resolve: () => "not-a-frame" } as unknown as OriginResolver;
    const table = ResolverTable.fromEntries([
      [new OriginRef("new"), resolver, new OriginRef("old")],
    ]);

    const error = thrown(() => table.resolve(new OriginRef("new")));
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toBe(
      "resolver table delegate must return a DiagnosticFrame or null",
    );
  });

  type Entries = Parameters<typeof ResolverTable.fromEntries>[0];

  test.each([
    [
      "a new origin that isn't an OriginRef",
      () => ResolverTable.fromEntries([["new", {}, new OriginRef("old")]] as unknown as Entries),
      "resolver table new origin must be an OriginRef",
    ],
    [
      "a delegate without resolve",
      () =>
        ResolverTable.fromEntries([
          [new OriginRef("new"), {}, new OriginRef("old")],
        ] as unknown as Entries),
      "resolver table delegate must implement resolve",
    ],
    [
      "a null delegate",
      () =>
        ResolverTable.fromEntries([
          [new OriginRef("new"), null, new OriginRef("old")],
        ] as unknown as Entries),
      "resolver table delegate must implement resolve",
    ],
    [
      "an old origin that isn't an OriginRef",
      () =>
        ResolverTable.fromEntries([
          [new OriginRef("new"), new ResolverTable(), "old"],
        ] as unknown as Entries),
      "resolver table old origin must be an OriginRef",
    ],
    [
      "a two-item entry",
      () =>
        ResolverTable.fromEntries([
          [new OriginRef("new"), new ResolverTable()],
        ] as unknown as Entries),
      "resolver table entry must be a three-item tuple",
    ],
    [
      "merging a non-table",
      () => ResolverTable.merge(new ResolverTable(), {} as unknown as ResolverTable),
      "resolver table merge requires ResolverTable values",
    ],
    [
      "resolving a string",
      () => new ResolverTable().resolve("new" as unknown as OriginRef),
      "resolver table resolve requires an OriginRef",
    ],
  ] as const)("rejects malformed routes: %s", (_, operation, message) => {
    // Malformed routing values fail at the public boundary with stable errors.
    const error = thrown(operation);

    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toBe(message);
  });

  test("takes no routes in its constructor", () => {
    // Raw target maps are deliberately not part of the public constructor.
    if (false) {
      // @ts-expect-error the constructor takes no arguments
      new ResolverTable(new Map());
    }
  });
});

describe("the diagnostics module", () => {
  test("imports nothing", () => {
    // The diagnostic value spine must stay below every other module.
    const path = fileURLToPath(new URL("../../src/diagnostics.ts", import.meta.url));
    expect(importStatements(readFileSync(path, "utf8"))).toEqual([]);
  });
});
