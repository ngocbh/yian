/**
 * A named LOOP body's returned fields must fit the inputs the body declares: a port of
 * `tests/compose/test_named_loop_carried_types.py`, loading each flow with `assembleText`
 * (`assemble` as the loader calls it) where Python calls `load_flow`.
 *
 * A `call:` body's declared input is the parameter each returned field is bound to on the next
 * iteration. A returned field that can never fit it is a load error; over a typed seed, a nullable
 * returned field whose non-null type fits is left to the runtime input check. A field a `while:`
 * null check proves not null never enters the body as null, so it may be returned as nullable into
 * a strict input.
 *
 * Ported: `test_untyped_seed_with_an_optional_returned_field_is_rejected` (3),
 * `test_float_returned_into_an_int_input_is_rejected` (2),
 * `test_any_returned_into_an_int_input_is_rejected`,
 * `test_optional_any_returned_into_an_int_input_is_rejected`,
 * `test_optional_any_returned_into_an_any_input_over_an_untyped_seed_is_rejected`,
 * `test_optional_any_returned_into_an_any_input_over_a_typed_seed_loads`,
 * `test_int_returned_into_a_float_input_loads`,
 * `test_untyped_seed_with_an_optional_typedef_field_is_rejected`,
 * `test_typed_seed_with_an_optional_typedef_field_loads`,
 * `test_a_returned_field_the_body_does_not_take_loads`,
 * `test_a_seed_that_misses_the_input_reports_the_binding` and `test_the_error_points_at_the_seed`;
 * the load halves of `test_typed_seed_leaves_a_returned_null_to_the_input_check`,
 * `test_typed_seed_with_an_optional_returned_field_runs`,
 * `test_while_loop_ending_on_null_over_an_untyped_seed_runs`,
 * `test_while_loop_ending_on_null_over_a_typed_seed_runs`,
 * `test_while_loop_ending_on_null_into_an_optional_input_runs`,
 * `test_optional_returned_into_an_optional_input_runs`, `test_null_seed_into_an_optional_input_runs`
 * and `test_a_partial_loop_runs`.
 *
 * Changed: the `step` def is built by hand (`handChild`: its input `n` read from the def's
 * `input:` text, its `bump` node a producer with the record contract `{n: <body output>}`, and
 * `${bump.output}` returned), since def bodies load in T3-22; `test_a_returned_field_the_body_does_not_take_loads`
 * and `test_a_partial_loop_runs` build theirs the same way. The error rows pin the whole message
 * where Python matches fragments.
 *
 * Pinned at the end (`assembledWith`, the `step` def built by hand):
 * `load-loop/compose_named_loop_carried_types__{any_returned_into_an_int_input_is_rejected,
 * untyped_seed_rejects_an_optional_typedef_field}` and the `ok` cases
 * `load-loop/compose_named_loop_carried_types__{a_returned_field_the_body_does_not_take_loads,
 * int_returned_into_a_float_input_loads,typed_seed_with_an_optional_typedef_field_loads}`.
 *
 * Later (T3-22): `test_a_loop_inside_a_called_body_is_rejected` (the `defs entry` prefix comes
 * from the loader). Later (T4): the run halves of the load-half rows above.
 */

import { expect, test } from "bun:test";

import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  assembledWith,
  assembleText,
  type CaseRow,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

/**
 * A flow whose LOOP `repeat` calls the `defs:` body `step`, carrying `n`.
 *
 * @param bodyIn The type `step` declares for its input `n`.
 * @param bodyOut The type the body's CODE node declares for its output `n`.
 * @param ret The TS expression the body's CODE node returns as `n`.
 * @param seed The LOOP's `n:` seed binding, e.g. `${input.z}`.
 * @param options.driver The LOOP's driver lines (default `times: 2`).
 * @param options.flowInput The flow's input line (default `z: !int`).
 */
function namedLoop(
  bodyIn: string,
  bodyOut: string,
  ret: string,
  seed: string,
  options: { driver?: string; flowInput?: string } = {},
): string {
  const { driver = "times: 2", flowInput = "z: !int" } = options;
  return `
id: named
name: named
input:
  ${flowInput}
defs:
  step:
    input:
      n: !${bodyIn}
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: ${bodyOut}
        code: |
          return {"n": ${ret}}
    output: \${bump.output}
nodes:
  repeat:
    kind: loop
    ${driver}
    call: step
    input:
      n: ${seed}
output: \${repeat.output}
`;
}

const PAGINATE = 'inputs["n"] >= 2n ? null : inputs["n"] + 1n';
const WHILE = "while: ${n != null}\n    max: 10";
const P_TYPEDEFS = { P: { x: "int" } };

/**
 * A `step` def built by hand: inputs `inputText` (the lines under its `input:` key), a `bump`
 * producer whose record contract has `fields` (each a required field), returned whole.
 *
 * @param typedefs The typedefs the types name (default none).
 */
function stepOf(
  inputText: string,
  fields: Readonly<Record<string, string>>,
  typedefs: Readonly<Record<string, unknown>> = {},
): ChildFlows {
  const registry = readTypedefs({ ...typedefs, Rec: fields });
  return {
    step: handChild({
      inputs: parametersFromYaml(`input:\n${inputText}`, registry),
      producers: { bump: new ValueContract(typeFor("Rec", registry)) },
      output: "${bump.output}",
    }),
  };
}

/** The `step` def of `namedLoop`: input `n: !<bodyIn>`, a `bump: {n: <bodyOut>}` producer, returned whole. */
const step = (bodyIn: string, bodyOut: string, typedefs: Readonly<Record<string, unknown>> = {}) =>
  stepOf(`  n: !${bodyIn}\n`, { n: bodyOut }, typedefs);

/** Assemble `namedLoop(bodyIn, bodyOut, ...)` with its `step` def built by hand. */
function load(
  bodyIn: string,
  bodyOut: string,
  seed: string,
  options: { driver?: string; flowInput?: string; ret?: string } = {},
): void {
  const { ret = "null", ...loop } = options;
  assembleText(namedLoop(bodyIn, bodyOut, ret, seed, loop), "flow.yaml", { children: step(bodyIn, bodyOut) });
}

/** The message of the load error `load(...)` raises. */
const message = (bodyIn: string, bodyOut: string, seed: string) =>
  loadError(() => load(bodyIn, bodyOut, seed)).message;

/** The load error for a body returning `Optional(int)` into `n: int`, its tag hint echoing `seed`. */
const nullableMessage = (seed: string) =>
  "loop node 'repeat': the body 'step' returns 'n' as Optional(int), but takes 'n' as int; make the input " +
  `Optional(int), or tag the seed to leave the check to runtime: \`n: !int ${seed}\``;

test.each(["${input.z}", "0", "${input.z :- 0}"])(
  "an untyped seed with an optional returned field is rejected (%s)",
  (seed) => {
    expect(message("int", "Optional(int)", seed)).toBe(nullableMessage(seed));
  },
);

test("a typed seed leaves a returned null to the input check", () => {
  // The load half; the run half (the next turn's input check fails) is T4's.
  load("int", "Optional(int)", "!int ${input.z}");
});

test("a typed seed with an optional returned field loads", () => {
  // Python's test_typed_seed_with_an_optional_returned_field_runs; the run half is T4's.
  load("int", "Optional(int)", "!int ${input.z}", { ret: 'inputs["n"] + 1n' });
});

test("a while loop ending on null over an untyped seed loads", () => {
  // Python's test_while_loop_ending_on_null_over_an_untyped_seed_runs: the `while:` null check
  // keeps the returned null out of the body's `n: int`. The run half is T4's.
  assembleText(namedLoop("int", "Optional(int)", PAGINATE, "${input.z}", { driver: WHILE }), "flow.yaml", {
    children: step("int", "Optional(int)"),
  });
});

test("a while loop ending on null over a typed seed loads", () => {
  // Python's test_while_loop_ending_on_null_over_a_typed_seed_runs; the run half is T4's.
  assembleText(namedLoop("int", "Optional(int)", PAGINATE, "!int ${input.z}", { driver: WHILE }), "flow.yaml", {
    children: step("int", "Optional(int)"),
  });
});

test("a while loop ending on null into an optional input loads", () => {
  // Python's test_while_loop_ending_on_null_into_an_optional_input_runs; the run half is T4's.
  assembleText(namedLoop("Optional(int)", "Optional(int)", PAGINATE, "${input.z}", { driver: WHILE }), "flow.yaml", {
    children: step("Optional(int)", "Optional(int)"),
  });
});

test.each(["${input.z}", "!int ${input.z}"])("a float returned into an int input is rejected (%s)", (seed) => {
  expect(message("int", "float", seed)).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as float, but takes 'n' as int; " +
      "change the body's input or output type so they agree",
  );
});

test("an Any returned into an int input is rejected", () => {
  expect(message("int", "Any", "${input.z}")).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as Any, but takes 'n' as int; " +
      "narrow it in the body with isinstance in a case, or declare its producer's type",
  );
});

test("an Optional(Any) returned into an int input is rejected", () => {
  expect(message("int", "Optional(Any)", "${input.z}")).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as Optional(Any), but takes 'n' as int; " +
      "narrow it in the body with isinstance in a case, or declare its producer's type",
  );
});

test("an Optional(Any) returned into an Any input over an untyped seed is rejected", () => {
  expect(message("Any", "Optional(Any)", "${input.z}")).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as Optional(Any), but takes 'n' as Any; " +
      "make the input Optional(Any), or tag the seed to leave the check to runtime: `n: !Any ${input.z}`",
  );
});

test("an Optional(Any) returned into an Any input over a typed seed loads", () => {
  load("Any", "Optional(Any)", "!Any ${input.z}");
});

test("an optional returned into an optional input loads", () => {
  // Python's test_optional_returned_into_an_optional_input_runs; the run half is T4's.
  load("Optional(int)", "Optional(int)", "${input.z}");
});

test("a null seed into an optional input loads", () => {
  // Python's test_null_seed_into_an_optional_input_runs; the run half is T4's.
  load("Optional(int)", "Optional(int)", "${null}");
});

test("an int returned into a float input loads", () => {
  load("float", "int", "${input.z}");
});

/** A flow whose named body takes `n: P` and returns it as `Optional(P)`, seeded by `seed`. */
function recordLoop(seed: string): string {
  return "typedefs:\n  P:\n    x: int\n" + namedLoop("P", "Optional(P)", 'inputs["n"]', seed, { flowInput: "p: !P" });
}

test("an untyped seed with an optional typedef field is rejected", () => {
  const error = loadError(() =>
    assembleText(recordLoop("${input.p}"), "flow.yaml", { children: step("P", "Optional(P)", P_TYPEDEFS) }),
  );

  expect(error.message).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as Optional({\"x\":int}), but takes 'n' as {\"x\":int}; " +
      "make the input Optional(P), or tag the seed to leave the check to runtime: `n: !P ${input.p}`",
  );
});

test("a typed seed with an optional typedef field loads", () => {
  assembleText(recordLoop("!P ${input.p}"), "flow.yaml", { children: step("P", "Optional(P)", P_TYPEDEFS) });
});

test("a returned field the body does not take loads", () => {
  assembleText(
    `
id: extra
name: extra
input:
  z: !int
defs:
  step:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
          m: int
        code: |
          return {"n": inputs["n"] + 1n, "m": 0}
    output: \${bump.output}
nodes:
  repeat:
    kind: loop
    times: 2
    call: step
    input:
      n: \${input.z}
      m: 0
output: \${repeat.output}
`,
    "flow.yaml",
    { children: stepOf("  n: !int\n", { n: "int", m: "int" }) },
  );
});

test("a partial loop loads", () => {
  // Python's test_a_partial_loop_runs; the run half is T4's.
  assembleText(
    `
id: partial
name: partial
input:
  z: !int
defs:
  step:
    input:
      n: !int
      m: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1n}
    output: \${bump.output}
nodes:
  repeat:
    kind: loop
    times: 2
    call: step
    input:
      n: \${input.z}
      m: 5
output: \${repeat.output}
`,
    "flow.yaml",
    { children: stepOf("  n: !int\n  m: !int\n", { n: "int" }) },
  );
});

test("a seed that misses the input reports the binding", () => {
  const error = message("int", "Optional(int)", "!Optional(int) ${input.z}");

  expect(error).toContain("binding 'n' — child expects 'int'");
  expect(error).not.toContain("the body 'step' returns");
});

test("the error points at the seed", () => {
  const text = namedLoop("int", "Optional(int)", "null", "${input.z}");
  const line = text.split("\n").indexOf("      n: ${input.z}") + 1;

  const frames = loadError(() =>
    assembleText(text, "flow.yaml", { children: step("int", "Optional(int)") }),
  ).diagnostic.frames;

  expect(frames).toHaveLength(1);
  expect(frames[0]!.range.startLine).toBe(line);
});

// ---------- Pinned harvested cases ----------

test.each<CaseRow>([
  {
    id: "load-loop/compose_named_loop_carried_types__any_returned_into_an_int_input_is_rejected",
    call: assembledWith(step("int", "Any")),
  },
  {
    id: "load-loop/compose_named_loop_carried_types__untyped_seed_rejects_an_optional_typedef_field",
    call: assembledWith(step("P", "Optional(P)", P_TYPEDEFS)),
  },
  {
    id: "load-loop/compose_named_loop_carried_types__a_returned_field_the_body_does_not_take_loads",
    call: assembledWith(stepOf("  n: !int\n", { n: "int", m: "int" })),
  },
  {
    id: "load-loop/compose_named_loop_carried_types__int_returned_into_a_float_input_loads",
    call: assembledWith(step("float", "int")),
  },
  {
    id: "load-loop/compose_named_loop_carried_types__typed_seed_with_an_optional_typedef_field_loads",
    call: assembledWith(step("P", "Optional(P)", P_TYPEDEFS)),
  },
])("$id", runCase);
