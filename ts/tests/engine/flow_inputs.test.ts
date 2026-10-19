/**
 * Flow `input:` declarations and supplied top-level inputs at the run boundary, a port of
 * `tests/engine/test_flow_inputs.py`: compose's `readFlowInputs` reads source-free type text
 * and `readSourceParameters` (through `parametersFromYaml`) reads tagged YAML declarations;
 * `coerceInputs` converts text under each parameter's `coerceStrings` policy and
 * `applyDefaults` fills omitted inputs. The binder tests build their parameters by hand with
 * `coerceStrings: true`, as the readers build them. Ints are `bigint` (`30n`).
 *
 * Ported: `test_required_scalar`, `test_int_with_default_and_coercion` (its declaration half
 * here, its `coerceInputs` half with a hand-built parameter), `test_optional_nullable`,
 * `test_nullable_null_default_is_distinct_from_no_default`,
 * `test_nonnullable_null_default_is_rejected_during_load`,
 * `test_source_free_default_assignment_is_rejected_during_load`,
 * `test_quoted_integer_default_is_not_coerced`, the three former-boolean tests,
 * `test_number_default_retains_compatible_integer_payload`,
 * `test_input_declaration_errors_retain_their_source_range`, `test_float_default`,
 * `test_list_default`, `test_bare_word_string_default`, `test_apply_defaults_fills_omitted`,
 * `test_reader_returns_only_exact_parameter_carriers`, the three bool spelling tests and
 * `test_bad_number_passes_through_unchanged`.
 *
 * Changed (TS yian): a default holding a lone surrogate (a double-quoted `\ud800` escape) is
 * refused before its tag is read, where Python accepts it.
 *
 * TS-only: the text rules for `float` and `int` inputs (the rules `int-in-float` and `text`). Text for a `float` input
 * reads an int spelling as that int at any size (`"2"` is `2n`, kept an int in the float slot),
 * a float spelling as a float, `inf`/`-inf` as infinity, and refuses a spelling beyond the
 * float range; Python's `float()` reads `"2"` as `2.0` and also takes `nan`, `Infinity` and
 * `+inf`, which stay text here. An `int` input's text has no digit limit and its digits are
 * ASCII. Also TS-only: a parameter without `coerceStrings`, an `Any` input and a non-record
 * `raw`; a 5000-digit `bigint` default, the counterpart of `test_long_int_boundaries.py`'s two
 * `apply_defaults` tests (no digit limit, so it is accepted rather than rejected); and the
 * harvested-case table, which makes the loader's call on the `input:` declarations of the
 * conformance cases these readers decide, comparing the message and the last frame.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { readFlowInputs } from "../../src/compose/helpers/types.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { applyDefaults, checkInputNames, coerceInputs } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { buildValueAs, Type, TypeCheckError, ValueKind } from "../../src/typesys/index.ts";
import type { TypeRegistry } from "../../src/typesys/types.ts";
import {
  type CaseRow,
  inputsAt,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const EMPTY: TypeRegistry = new Map();

/** Return the parameters keyed by name. */
function byName(parameters: Parameter[]): Map<string, Parameter> {
  return new Map(parameters.map((parameter) => [parameter.name, parameter]));
}

/** Read flow `input:` declarations from their YAML body, one per line. */
function inputs(body: string): Parameter[] {
  const lines = body
    .split("\n")
    .map((line) => `  ${line}\n`)
    .join("");
  return parametersFromYaml("input:\n" + lines, EMPTY);
}

/** Return the one parameter `name` the source-free `mapping` declares. */
function sourceFree(mapping: Record<string, unknown>, name: string): Parameter {
  return byName(readFlowInputs(mapping, EMPTY)).get(name)!;
}

/** Return the one tagged parameter `name` the YAML `body` declares. */
function tagged(body: string, name: string): Parameter {
  return byName(inputs(body)).get(name)!;
}

/** Return one parameter's semantic type. */
function typeOf(parameter: Parameter): Type {
  expect(parameter.contract?.constructor).toBe(ValueContract);
  return parameter.contract!.type;
}

describe("declarations", () => {
  test("a required scalar", () => {
    const d = sourceFree({ topic: "str" }, "topic");
    expect(d).toBeInstanceOf(Parameter);
    expect([d.required, d.hasDefault, d.default]).toEqual([true, false, null]);
    expect(typeOf(d).kind).toBe(ValueKind.STRING);
  });

  test("an int with a default", () => {
    const d = tagged("window: !int 30", "window");
    expect(typeOf(d).kind).toBe(ValueKind.INTEGER);
    expect(d.default).toBe(30n);
    expect([d.hasDefault, d.required, d.coerceStrings]).toEqual([true, false, true]);
  });

  test("an optional input is nullable and not required", () => {
    const d = sourceFree({ as_of: "Optional(date)" }, "as_of");
    expect(typeOf(d).nullable).toBe(true);
    expect([d.required, d.hasDefault, d.default]).toEqual([false, false, null]);
  });

  test("a nullable input's null default is distinct from no default", () => {
    const absent = sourceFree({ value: "Optional(int)" }, "value");
    const explicitNull = tagged("value: !Optional(int) null", "value");
    expect(absent.hasDefault).toBe(false);
    expect([explicitNull.hasDefault, explicitNull.default, explicitNull.required]).toEqual([
      true,
      null,
      false,
    ]);
    expect(explicitNull).not.toEqual(absent);
  });

  test("a non-nullable input's null default is rejected", () => {
    expect(loadError(() => inputs("value: !int null")).message).toContain(
      "'value': `null` is not a valid int",
    );
  });

  test("a source-free default assignment is rejected", () => {
    expect(loadError(() => readFlowInputs({ value: "int =" }, EMPTY)).message).toContain(
      "input 'value': source-free declarations take a type only; write defaults in YAML with a tag",
    );
  });

  test("a quoted integer default is not coerced", () => {
    expect(loadError(() => inputs('value: !int "30"')).message).toContain(
      "'value': `\"30\"` is not a valid int",
    );
  });

  test("a former boolean word as a bool default names the boolean to write", () => {
    expect(loadError(() => inputs("flag: !bool no")).message).toBe(
      "'flag': `no` is not a valid bool; `no` is text; write `false`",
    );
  });

  test.each(["flag: !int no", "flags: !list(bool) [yes, no]"])(
    "a former boolean word in %p gets no fix",
    (body) => {
      expect(loadError(() => inputs(body)).message).not.toContain("is text");
    },
  );

  test("a float default keeps a compatible int", () => {
    const parameter = tagged("value: !float 30", "value");
    expect(typeOf(parameter).kind).toBe(ValueKind.NUMBER);
    expect(parameter.hasDefault).toBe(true);
    expect(parameter.default).toBe(30n);
  });

  test.each([
    { declaration: "list(", message: "bad type expression" },
    { declaration: { field: 3n }, message: "a source-free declaration is type text, got dict" },
    { declaration: [{ field: "int" }], message: "a source-free declaration is type text, got list" },
    { declaration: 3n, message: "a source-free declaration is type text, got int" },
  ])("a declaration error keeps its source range: $message", ({ declaration, message }) => {
    const sourceRange = new SourceRange(4, 10, 4, 18);
    const diagnostics = new LoadDiagnostics("flow.yaml", (" ".repeat(40) + "\n").repeat(5));

    const error = loadError(() =>
      readFlowInputs({ value: declaration }, EMPTY, new Map([["value", sourceRange]]), { diagnostics }),
    );

    expect(error.message).toContain(message);
    expect(error.diagnostic.frames.length).toBe(1);
    expect(error.diagnostic.frames[0]!.range).toBe(sourceRange);
    expect(error.diagnostic.frames[0]!.label).toBe("flow.yaml");
  });

  test("a float default", () => {
    const d = tagged("budget: !float 1000.0", "budget");
    expect(typeOf(d).kind).toBe(ValueKind.NUMBER);
    expect(d.default).toBe(1000);
  });

  test("a list default", () => {
    const d = tagged('bundle: !list(str) ["ACME"]', "bundle");
    expect(typeOf(d).kind).toBe(ValueKind.LIST_STRING);
    expect(d.default).toEqual(["ACME"]);
    expect([d.hasDefault, d.required]).toEqual([true, false]);
  });

  test("a bare word string default", () => {
    const d = tagged("style: !str relevance", "style");
    expect(typeOf(d).kind).toBe(ValueKind.STRING);
    expect(d.default).toBe("relevance");
    expect([d.hasDefault, d.required]).toEqual([true, false]);
  });

  test("the source-free reader returns exact parameter carriers without origins", () => {
    for (const d of readFlowInputs({ topic: "str" }, EMPTY)) {
      expect(d.constructor).toBe(Parameter);
      expect(d.contract?.constructor).toBe(ValueContract);
      expect([d.origin, d.defaultOrigin]).toEqual([null, null]);
      expect(d.contract!.origins.entries).toEqual([]);
      expect(d.contract!.origins.default).toBeNull();
    }
  });

  // A double-quoted `\ud800` escape writes a lone surrogate.
  test.each([
    ['x: !str "\\ud800"', 19],
    ['x: !Nope "\\ud800"', 20],
  ] as const)("a default holding a lone surrogate in %p is refused before its tag is read", (body, end) => {
    const error = loadError(() => inputs(body));
    expect(error.message).toBe("input 'x' default: text holding a lone surrogate is not a yian value");
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(2, 6, 2, end));
  });
});

/** Return a flow input `name` of scalar `kind`, with a default when `fallback` is given. */
function input(name: string, kind: ValueKind, fallback?: unknown): Parameter {
  const contract = new ValueContract(Type.scalar(kind));
  if (fallback === undefined) return new Parameter(name, { contract, required: true, coerceStrings: true });
  return new Parameter(name, { contract, hasDefault: true, default: fallback, coerceStrings: true });
}

/** Return what `coerceInputs` makes of `text` supplied for an input `x` of `kind`. */
function coerced(kind: ValueKind, text: string): unknown {
  return coerceInputs([input("x", kind)], { x: text })["x"];
}

test("an int input coerces its text", () => {
  const params = [input("window", ValueKind.INTEGER, 30n)];
  expect(coerceInputs(params, { window: "5" })).toEqual({ window: 5n });
});

test("applyDefaults fills omitted inputs", () => {
  const params = [input("topic", ValueKind.STRING), input("window", ValueKind.INTEGER, 30n)];
  expect(applyDefaults(params, coerceInputs(params, { topic: "ACME" }))).toEqual({ topic: "ACME", window: 30n });
});

describe("bool spellings", () => {
  test.each(["1", "true", "TRUE", "yes", "y", "on", " True "])("%p is true", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(true);
  });

  test.each(["0", "false", "FALSE", "no", "n", "off", " False "])("%p is false", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(false);
  });

  // Like a bad number, an unrecognized spelling stays text so the contract rejects it.
  test.each(["maybe", "ture", "approve", "No thanks", ""])("%p stays text", (raw) => {
    expect(coerced(ValueKind.BOOLEAN, raw)).toBe(raw);
  });
});

test("a bad number passes through unchanged", () => {
  expect(coerced(ValueKind.INTEGER, "not-a-number")).toBe("not-a-number");
});

describe("float input text", () => {
  test.each([
    ["2", 2n],
    ["2.0", 2],
    [" 1_000 ", 1000n],
    ["1e3", 1000],
    [".5", 0.5],
    ["inf", Infinity],
    ["-inf", -Infinity],
    ["-0.0", 0],
  ] as const)("%p reads as %p", (text, value) => {
    expect(coerced(ValueKind.NUMBER, text)).toBe(value);
  });

  test("an int spelling is an int at any size", () => {
    const text = "1" + "0".repeat(400);
    const value = coerced(ValueKind.NUMBER, text);
    expect(value).toBe(10n ** 400n);
    // The float contract keeps it as given.
    expect(buildValueAs(Type.scalar(ValueKind.NUMBER), value).value).toBe(10n ** 400n);
  });

  test("a spelling beyond the float range is refused", () => {
    const call = () => coerced(ValueKind.NUMBER, "1e400");
    expect(call).toThrow(TypeCheckError);
    expect(call).toThrow(new TypeCheckError("input 'x': number is too large for a float"));
  });

  test.each(["nan", "Infinity", "+inf", "0x10", "1__0", "", " abc "])("%p stays text as given", (text) => {
    expect(coerced(ValueKind.NUMBER, text)).toBe(text);
  });
});

describe("int input text", () => {
  test("leading zeros are dropped", () => {
    expect(coerced(ValueKind.INTEGER, "007")).toBe(7n);
  });

  test("an int has no digit limit", () => {
    const text = "9".repeat(5000);
    expect(coerced(ValueKind.INTEGER, text)).toBe(BigInt(text));
  });

  test("a float spelling stays text", () => {
    expect(coerced(ValueKind.INTEGER, "2.0")).toBe("2.0");
  });
});

test("text stays text without coerceStrings or for an Any input", () => {
  const plain = new Parameter("n", { contract: new ValueContract(Type.scalar(ValueKind.INTEGER)) });
  expect(coerceInputs([plain], { n: "5" })).toEqual({ n: "5" });
  expect(coerced(ValueKind.ANY, "5")).toBe("5");
});

test("inputs must be a record", () => {
  const params = [input("x", ValueKind.INTEGER)];
  expect(() => checkInputNames(params, [["x", "1"]])).toThrow(
    new TypeError("inputs must be a mapping of input names to values, got Array"),
  );
  const fromMap = () => coerceInputs(params, new Map([["x", "1"]]) as never);
  // Bun's toThrow(error) compares only the message, so the class is checked on its own.
  expect(fromMap).toThrow(TypeError);
  expect(fromMap).toThrow(new TypeError("inputs must be a mapping of input names to values, got Map"));
});

test("unknown input names are listed in the text form", () => {
  const params = [input("x", ValueKind.INTEGER)];
  expect(() => checkInputNames(params, { zz: 1n })).toThrow(new RangeError('unknown input "zz"; declared inputs: x'));
  expect(() => checkInputNames(params, { b: 1n, a: 2n })).toThrow(
    new RangeError('unknown inputs "a", "b"; declared inputs: x'),
  );
});

test("applyDefaults accepts a long int default", () => {
  const big = 10n ** 5000n - 1n;
  expect(applyDefaults([new Parameter("x", { hasDefault: true, default: big })], {})).toEqual({ x: big });
});

// ---------- harvested conformance cases ----------

const rootInput = inputsAt(["input"]);
const childInput = inputsAt(["defs", "child", "input"], ["def", "child"]);
const CHILD = "defs entry 'child': ";

test.each<CaseRow>([
  { id: "load-core/engine_errors__invalid_authored_default_carries_input_range", call: rootInput },
  { id: "load-core/compose_source_ownership__recursive_root_alias", call: rootInput },
  { id: "load-core/engine_value_nesting_boundaries__input_default_over_the_limit", call: rootInput },
  {
    id: "load-core/compose_input_contract_origins__empty_definition_input_name",
    call: childInput,
    prefix: CHILD,
  },
  {
    id: "load-map-call/engine_defs__invalid_def_default_keeps_its_range",
    call: childInput,
    prefix: CHILD,
  },
])("readSourceFlowInputs: $id", runCase);
