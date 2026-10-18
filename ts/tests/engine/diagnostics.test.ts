/**
 * Contracts for source-neutral diagnostic values and origin resolution, a port of
 * `tests/engine/test_diagnostics.py`, plus the text helpers yian's messages need.
 */

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";

import {
  compareCodeUnits,
  FAILURE_CODES,
  Failure,
  type FailureCode,
  OriginRef,
  reprText,
} from "../../src/diagnostics.ts";

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
