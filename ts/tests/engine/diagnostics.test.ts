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
  OriginRef,
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

describe("the diagnostics module", () => {
  test("imports nothing", () => {
    // The diagnostic value spine must stay below every other module.
    const path = fileURLToPath(new URL("../../src/diagnostics.ts", import.meta.url));
    expect(importStatements(readFileSync(path, "utf8"))).toEqual([]);
  });
});
