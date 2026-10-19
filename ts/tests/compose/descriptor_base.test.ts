/**
 * The base descriptor contract shared by every node kind: a port of
 * `tests/compose/test_descriptor_base.py`.
 *
 * TS-only: the `message builders` rows, which Python tests through `load_flow` or its
 * docstrings: `valueTypeError` (its docstring examples), `rejectNonText`,
 * `rejectContainerSpans`, `inputShapeProblem` (the shapes of
 * `tests/compose/test_node_input_shape.py`; a text value is spelled by `valueRepr`, so `"abc"`
 * and `true` where Python shows `'abc'` and `True`) and `siteKey`.
 *
 * Dropped: the date row of `test_non_mapping_input_is_a_load_error` (TS reads an unquoted date
 * as text).
 *
 * Later (T3-14, the descriptor base): the 8 rows of `tests/compose/test_descriptor_base.py`.
 */

import { describe, expect, test } from "bun:test";

import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import {
  inputShapeProblem,
  rejectContainerSpans,
  rejectNonText,
  siteKey,
  valueTypeError,
} from "../../src/compose/nodes/base.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { loadError } from "./declarations.ts";

const RANGE = new SourceRange(2, 5, 2, 9);
const DIAGNOSTICS = new LoadDiagnostics("f.yaml", "nodes:\n  a: 1234\n");

describe("message builders", () => {
  test("valueTypeError names what YAML read the value as", () => {
    expect(valueTypeError("over", "a list", 5n, "write a list")).toBe(
      "`over:` must be a list, but YAML read it as an integer; write a list",
    );
    expect(valueTypeError("over", "a list", null, "write a list")).toBe(
      "`over:` must be a list, but it is empty; write a list",
    );
  });

  test("rejectNonText rejects a non-text value at its range", () => {
    const error = loadError(() => rejectNonText("code", "a", "code", 5n, RANGE, DIAGNOSTICS));
    expect(error.diagnostic.message).toBe(
      "node 'a' (kind=code): `code:` must be text, but YAML read it as an integer; put it in quotes",
    );
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([RANGE]);
  });

  test("rejectNonText lets text and an absent value through", () => {
    rejectNonText("code", "a", "code", "return 1", RANGE, DIAGNOSTICS);
    rejectNonText("code", "a", "code", null, RANGE, DIAGNOSTICS);
  });

  test("rejectNonText with required rejects an empty value", () => {
    const error = loadError(() =>
      rejectNonText("tool", "t", "tool_id", null, RANGE, DIAGNOSTICS, { required: true }),
    );
    expect(error.diagnostic.message).toBe(
      "node 't' (kind=tool): `tool_id:` must be text, but it is empty; write the text",
    );
  });

  test("rejectNonText takes a fix", () => {
    const error = loadError(() =>
      rejectNonText("case", "c", "on", [1n], RANGE, DIAGNOSTICS, { fix: "write an expression" }),
    );
    expect(error.diagnostic.message).toBe(
      "node 'c' (kind=case): `on:` must be text, but YAML read it as a list; write an expression",
    );
  });

  test.each([
    ["a list", ["${a.output}", "b"]],
    ["a record", { x: 1n, y: { z: "x ${a.output}" } }],
  ])("rejectContainerSpans rejects %s holding a span", (_label, value) => {
    const error = loadError(() => rejectContainerSpans("map", "m", "over", value, RANGE, DIAGNOSTICS));
    expect(error.diagnostic.message).toBe(
      "map node 'm' over: a value holding ${...} can't be a list or mapping; write the list as " +
        "an expression, e.g. `${[a.output, b.output]}`",
    );
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([RANGE]);
  });

  test.each([
    ["a list without a span", ["a", 1n, ["b"]]],
    ["a record without a span", { x: "a", y: null }],
    ["a whole span", "${a.output}"],
  ])("rejectContainerSpans lets %s through", (_label, value) => {
    rejectContainerSpans("map", "m", "over", value, RANGE, DIAGNOSTICS);
  });

  const LOOP_TAIL = "list each carried field, such as `<field>: ...`";
  test.each([
    ["a number", 5n, `(got 5); ${LOOP_TAIL}`],
    ["a list", [1n], `(got [1]); ${LOOP_TAIL}`],
    ["text", "abc", `(got "abc"); ${LOOP_TAIL}`],
    ["a boolean", true, `(got true); ${LOOP_TAIL}`],
    ["an empty list", [], "(got []); write `input: {}` or remove `input:`"],
    ["empty text", "", '(got ""); write `input: {}` or remove `input:`'],
    [
      "a tagged reference",
      new Binding("${seed.output}"),
      "(got the reference '${seed.output}'); list each carried field, such as " +
        "`<field>: ${seed.output.<field>}`",
    ],
    [
      "a nested reference",
      "${seed.output.n}",
      "(got the reference '${seed.output.n}'); list each carried field, such as " +
        "`<field>: ${seed.output.n.<field>}`",
    ],
    [
      "an expression",
      "${seed.output.n + 1}",
      "(got the expression '${seed.output.n + 1}'); list each carried field, such as " +
        "`<field>: ${...}`",
    ],
    [
      "an embedded span",
      "x ${seed.output}",
      '(got "x ${seed.output}"); list each carried field, such as `<field>: ${...}`',
    ],
  ])("inputShapeProblem describes %s", (_label, value, tail) => {
    expect(inputShapeProblem("loop", value)).toBe(`\`input:\` must be a mapping of carried fields ${tail}`);
  });

  test.each([
    ["loop", "carried fields", "list each carried field"],
    ["call", "the child's arguments", "bind each argument"],
    ["agent", "parameter names to bindings", "bind each parameter"],
  ])("inputShapeProblem names %s's binding noun", (kind, noun, verb) => {
    expect(inputShapeProblem(kind, "${seed.output}")).toBe(
      `\`input:\` must be a mapping of ${noun} (got the reference '\${seed.output}'); ` +
        `${verb}, such as \`<field>: \${seed.output.<field>}\``,
    );
    expect(inputShapeProblem(kind, "${seed.output.n + 1}")).toBe(
      `\`input:\` must be a mapping of ${noun} (got the expression '\${seed.output.n + 1}'); ` +
        `${verb}, such as \`<field>: \${...}\``,
    );
  });

  test("siteKey joins the range kind and the name", () => {
    expect(siteKey("field", "inputs")).toBe("field:inputs");
    expect(siteKey("input", "n")).toBe("input:n");
    expect(siteKey("llm_config", "model")).toBe("llm_config:model");
  });
});
