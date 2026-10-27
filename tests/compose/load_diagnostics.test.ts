/** Canonical diagnostic construction at the Compose load boundary, and the source helpers. */

import { describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError, sourceText } from "../../src/compose/helpers/errors.ts";
import {
  boundaryFieldRanges,
  isExtensionKey,
  nth,
  oneLineText,
} from "../../src/compose/helpers/source.ts";
import { SourceAccessError, SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { locatedError } from "../../src/compose/validate/typed_bindings.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { Binding } from "../../src/nodes/contracts.ts";

const broken = () => new Diagnostic({ code: "compose.load", message: "broken" });

describe("LoadError", () => {
  test("preserves one canonical diagnostic", () => {
    const diagnostic = broken();

    const error = new LoadError(diagnostic);

    expect(error.diagnostic).toBe(diagnostic);
    expect(error.message).toBe("broken");
    expect(error.name).toBe("LoadError");
    expect(error).toBeInstanceOf(RangeError);
    for (const name of ["lines", "range", "notes"]) expect(name in error).toBe(false);
    // Freezing the error must leave its stack and throwing intact.
    expect(error.stack).toContain("LoadError: broken");
    expect(() => {
      throw error;
    }).toThrow(LoadError);
  });

  test.each([
    ["diagnostic", new Diagnostic({ code: "compose.load", message: "replacement" })],
    ["_diagnostic", new Diagnostic({ code: "compose.load", message: "replacement" })],
    ["line", 7],
    ["lines", [7]],
    ["range", {}],
    ["notes", ["replacement"]],
    ["other", "enrichment"],
  ])("rejects a write to %p", (name, value) => {
    const error = new LoadError(broken()) as unknown as Record<string, unknown>;

    expect(() => {
      error[name] = value;
    }).toThrow(TypeError);
  });

  test("rejects a defined property", () => {
    const error = new LoadError(broken());

    expect(() => Object.defineProperty(error, "line", { value: 7 })).toThrow(TypeError);
  });

  test("rejects a non-diagnostic payload", () => {
    expect(() => new LoadError("broken" as never)).toThrow(
      new TypeError("LoadError requires a Diagnostic"),
    );
  });

  test("keeps a cause, and has none without one", () => {
    const inner = new Error("inner");
    const document = SourceDocument.parse("value: broken\n", "child.yaml");
    const range = document.root.child("value").range;
    const diagnostics = LoadDiagnostics.fromDocument(document);

    expect(new LoadError(broken(), { cause: inner }).cause).toBe(inner);
    expect(diagnostics.error("m", { cause: inner }).cause).toBe(inner);
    expect(diagnostics.prepend(diagnostics.error("m"), { range, cause: inner }).cause).toBe(
      inner,
    );

    expect(Object.hasOwn(new LoadError(broken()), "cause")).toBe(false);
    expect(Object.hasOwn(diagnostics.error("m"), "cause")).toBe(false);
    expect(Object.hasOwn(diagnostics.prepend(diagnostics.error("m"), { range }), "cause")).toBe(
      false,
    );
  });
});

describe("LoadDiagnostics", () => {
  test("document factory owns the frame source", () => {
    const document = SourceDocument.parse("value: broken\n", "child.yaml");
    const value = document.root.child("value");

    const error = LoadDiagnostics.fromDocument(document).error("bad value", {
      range: value.range,
      notes: ["expected an integer"],
    });

    expect(error.diagnostic).toEqual(
      new Diagnostic({
        code: "compose.load",
        message: "bad value",
        frames: [
          new DiagnosticFrame({ label: "child.yaml", text: document.text, range: value.range! }),
        ],
        notes: ["expected an integer"],
      }),
    );
  });

  test("prepend adds outer context without mutating the inner error", () => {
    const parent = SourceDocument.parse("uses:\n  child: child\n", "parent.yaml");
    const child = SourceDocument.parse("value: broken\n", "child.yaml");
    const childValue = child.root.child("value");
    const inner = LoadDiagnostics.fromDocument(child).error("bad child", {
      range: childValue.range,
      notes: ["inner note"],
    });
    const original = inner.diagnostic;

    const wrapped = LoadDiagnostics.fromDocument(parent).prepend(inner, {
      range: parent.root.child("uses").child("child").range,
      message: "uses alias 'child': bad child",
      note: "imported here",
    });

    expect(inner.diagnostic).toBe(original);
    expect(wrapped).not.toBe(inner);
    expect(wrapped.diagnostic.message).toBe("uses alias 'child': bad child");
    expect(wrapped.diagnostic.frames.map((frame) => frame.label)).toEqual([
      "parent.yaml",
      "child.yaml",
    ]);
    expect(wrapped.diagnostic.frames[0]!.note).toBe("imported here");
    expect(wrapped.diagnostic.notes).toEqual(["inner note"]);
  });

  test("prepend copies the inner diagnostic's other fields and keeps its message", () => {
    const inner = new LoadError(
      new Diagnostic({
        code: "x.y",
        message: "inner",
        notes: ["n"],
        nodeId: "a",
        causeType: "T",
        traceback: "tb",
      }),
    );

    const wrapped = new LoadDiagnostics().prepend(inner, { range: null });

    expect(wrapped.diagnostic).toEqual(inner.diagnostic);
    expect(wrapped.diagnostic).not.toBe(inner.diagnostic);
    expect(() => new LoadDiagnostics().prepend("e" as never, { range: null })).toThrow(
      new TypeError("load diagnostic context requires a LoadError"),
    );
  });

  test("source-free factory keeps the diagnostic unlocated", () => {
    const document = SourceDocument.parse("value: broken\n", "unused.yaml");

    const error = new LoadDiagnostics().error("source-free failure", {
      range: document.root.range,
    });

    expect(error.diagnostic.frames).toEqual([]);
  });

  test("the constructor checks its source", () => {
    expect(() => new LoadDiagnostics("l")).toThrow(
      new RangeError("load diagnostic source requires both label and text"),
    );
    expect(() => new LoadDiagnostics(null, "t")).toThrow(
      new RangeError("load diagnostic source requires both label and text"),
    );
    expect(() => new LoadDiagnostics(3 as never, "t")).toThrow(
      new TypeError("load diagnostic label must be a string"),
    );
    expect(() => new LoadDiagnostics("l", 3 as never)).toThrow(
      new TypeError("load diagnostic text must be a string"),
    );
    expect(Object.isFrozen(new LoadDiagnostics("l", "t"))).toBe(true);
  });

  test("fromCursor owns the cursor's document, or none", () => {
    const document = SourceDocument.parse("value: 1\n", "flow.yaml");

    for (const cursor of [null, SourceCursor.virtual(1n)]) {
      const diagnostics = LoadDiagnostics.fromCursor(cursor);
      expect([diagnostics.label, diagnostics.text]).toEqual([null, null]);
    }
    const owned = LoadDiagnostics.fromCursor(document.root);
    expect([owned.label, owned.text]).toEqual(["flow.yaml", document.text]);
  });

  test("diagnostic checks its arguments", () => {
    const document = SourceDocument.parse("value: 1\n", "flow.yaml");
    const range = document.root.child("value").range!;
    const diagnostics = LoadDiagnostics.fromDocument(document);
    const frame = diagnostics.frame(range)!;

    expect(() => diagnostics.diagnostic("m", { range, frames: [frame] })).toThrow(
      new RangeError("load diagnostic accepts a range or frames, not both"),
    );
    expect(() => diagnostics.diagnostic(3 as never)).toThrow(
      new TypeError("load diagnostic message must be a string"),
    );
    expect(() => diagnostics.diagnostic("m", { frames: ["f" as never] })).toThrow(
      new TypeError("load diagnostic frames must be DiagnosticFrame values"),
    );
    expect(() => diagnostics.diagnostic("m", { notes: [3 as never] })).toThrow(
      new TypeError("load diagnostic notes must be strings"),
    );
    expect(() => diagnostics.diagnostic("m", { range: "r" as never })).toThrow(
      new TypeError("load diagnostic range must be a SourceRange or null"),
    );
    expect(() => diagnostics.frame(range, { note: 3 as never })).toThrow(
      new TypeError("load diagnostic frame note must be a string or null"),
    );
    expect(diagnostics.diagnostic("m", { frames: [frame] }).frames).toEqual([frame]);
  });

  test("a source-free frame is null before its arguments are checked", () => {
    const diagnostics = new LoadDiagnostics();

    expect(diagnostics.frame("r" as never)).toBeNull();
    expect(diagnostics.frame(new SourceRange(1, 1, 1, 2), { note: 3 as never })).toBeNull();
    expect(diagnostics.diagnostic("m", { range: "r" as never }).frames).toEqual([]);
  });
});

describe("sourceText", () => {
  const text = "value: broken  \nkey: [a, b]\n";
  const diagnostics = new LoadDiagnostics("flow.yaml", text);

  test("returns the stripped text of a one-line range", () => {
    expect(sourceText(diagnostics, new SourceRange(1, 7, 1, 16))).toBe("broken");
  });

  test.each([
    ["no source", new LoadDiagnostics(), new SourceRange(1, 1, 1, 6)],
    ["no range", diagnostics, null],
    ["a multi-line range", diagnostics, new SourceRange(1, 1, 2, 4)],
    ["a line past the text", diagnostics, new SourceRange(3, 1, 3, 4)],
    ["an all-space slice", diagnostics, new SourceRange(1, 14, 1, 16)],
  ])("is null for %s", (_, owner, range) => {
    expect(sourceText(owner, range)).toBeNull();
  });

  test.each([
    ["CRLF", "a: x\r\nkey: yy\r\n"],
    ["lone-CR", "a: x\rkey: yy\r"],
  ])("indexes a %s text's lines as the ranges do", (_, source) => {
    const document = SourceDocument.parse(source, "flow.yaml");

    expect(sourceText(LoadDiagnostics.fromDocument(document), document.root.rangeOf("key"))).toBe(
      "yy",
    );
  });
});

describe("source helpers", () => {
  test("isExtensionKey", () => {
    expect(isExtensionKey("x-anchors")).toBe(true);
    expect(isExtensionKey("xanchors")).toBe(false);
  });

  test("nth", () => {
    expect(nth([1, 2], 1)).toBe(2);
    expect(nth([1, 2], -1)).toBeNull();
    expect(nth([1, 2], 2)).toBeNull();
  });

  test("oneLineText returns the unstripped slice, or null", () => {
    const document = SourceDocument.parse("value:  broken \nnext: 1\n", "flow.yaml");

    expect(oneLineText(document, new SourceRange(1, 7, 1, 16))).toBe("  broken ");
    expect(oneLineText(null, new SourceRange(1, 1, 1, 6))).toBeNull();
    expect(oneLineText(document, null)).toBeNull();
    expect(oneLineText(document, new SourceRange(1, 1, 2, 4))).toBeNull();
    expect(oneLineText(document, new SourceRange(3, 1, 3, 4))).toBeNull();
  });

  test.each(["input", "inputs"])("boundaryFieldRanges maps %s: declarations", (key) => {
    const document = SourceDocument.parse(`${key}:\n  a: int\n  b: {type: str}\n`, "flow.yaml");
    const declarations = document.root.child(key);

    expect(boundaryFieldRanges(document.root)).toEqual(
      new Map([
        ["a", declarations.rangeOf("a")!],
        ["b", declarations.rangeOf("b")!],
      ]),
    );
  });

  test.each([
    ["a list root", "- input\n"],
    ["an input: that isn't a mapping", "input: [a]\n"],
  ])("boundaryFieldRanges is empty for %s", (_, text) => {
    expect(boundaryFieldRanges(SourceDocument.parse(text, "flow.yaml").root)).toEqual(new Map());
  });

  test("boundaryFieldRanges is empty for a recursive alias input:", () => {
    const root = SourceDocument.parse("&r {input: *r}\n", "flow.yaml").root;

    expect(() => root.mappingValueRanges("input")).toThrow(SourceAccessError);
    expect(boundaryFieldRanges(root)).toEqual(new Map());
  });

  test("boundaryFieldRanges is empty for a virtual cursor", () => {
    expect(boundaryFieldRanges(SourceCursor.virtual({ input: { a: "int" } }))).toEqual(new Map());
  });
});

test("a load diagnostic keeps compose.load unless given a code", () => {
  const d = new LoadDiagnostics("flow.yaml", "x: 1\n");
  expect(d.diagnostic("m").code).toBe("compose.load");
  expect(d.error("m", { code: "expr.nullable_field_read" }).diagnostic.code).toBe("expr.nullable_field_read");
  expect(locatedError("m", new Binding("${x}"), null, null, d).diagnostic.code).toBe("compose.load");
  expect(locatedError("m", new Binding("${x}"), null, null, d, { code: "expr.nullable_field_read" }).diagnostic.code).toBe(
    "expr.nullable_field_read",
  );
});
