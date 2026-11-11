/**
 * A named LOOP body's returned fields must fit the inputs the body declares.
 *
 * A `call:` body's declared input is the parameter each returned field is bound to on the next
 * iteration. A returned field that can never fit it is a load error; over a typed seed, a nullable
 * returned field whose non-null type fits is left to the runtime input check. A field a `while:`
 * null check proves not null never enters the body as null, so it may be returned as nullable into
 * a strict input.
 */

import { expect, test } from "bun:test";

import { runFlow } from "../../src/run.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  runCase,
} from "../helpers/declarations.ts";
import { outputTag } from "../helpers/output_tags.ts";

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
          n: ${outputTag(bodyOut)}
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

const PAGINATE = 'inputs["n"] >= 2 ? null : inputs["n"] + 1';
const WHILE = "while: ${n != null}\n    max: 10";

/** Assemble `namedLoop(bodyIn, bodyOut, ...)`. */
function load(
  bodyIn: string,
  bodyOut: string,
  seed: string,
  options: { driver?: string; flowInput?: string; ret?: string } = {},
): void {
  const { ret = "null", ...loop } = options;
  assembleText(namedLoop(bodyIn, bodyOut, ret, seed, loop));
}

/** Run `text` on `{z: 0}`. */
const run = async (text: string) => runFlow(await loadConformanceFlow(text), { z: 0n });

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
  // The load half; the run half (the next turn's input check fails) is the harvested
  // run-loop/compose_named_loop_carried_types__a_returned_null_fails_the_next_turn.
  load("int", "Optional(int)", "!int ${input.z}");
});

test("a typed seed with an optional returned field loads", () => {
  load("int", "Optional(int)", "!int ${input.z}", { ret: 'inputs["n"] + 1' });
});

test("typed seed with an optional returned field runs", async () => {
  const result = await run(namedLoop("int", "Optional(int)", 'inputs["n"] + 1', "!int ${input.z}"));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n });
});

test("a while loop ending on null over an untyped seed loads", () => {
  // The `while:` null check
  // keeps the returned null out of the body's `n: int`. The run half is the harvested
  // run-loop/compose_named_loop_carried_types__while_loop_ends_on_a_returned_null.
  assembleText(namedLoop("int", "Optional(int)", PAGINATE, "${input.z}", { driver: WHILE }));
});

test("a while loop ending on null over a typed seed loads", () => {
  assembleText(namedLoop("int", "Optional(int)", PAGINATE, "!int ${input.z}", { driver: WHILE }));
});

test("while loop ending on null over a typed seed runs", async () => {
  const result = await run(namedLoop("int", "Optional(int)", PAGINATE, "!int ${input.z}", { driver: WHILE }));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: null });
});

test("a while loop ending on null into an optional input loads", () => {
  assembleText(namedLoop("Optional(int)", "Optional(int)", PAGINATE, "${input.z}", { driver: WHILE }));
});

test("while loop ending on null into an optional input runs", async () => {
  const result = await run(namedLoop("Optional(int)", "Optional(int)", PAGINATE, "${input.z}", { driver: WHILE }));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: null });
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
  load("Optional(int)", "Optional(int)", "${input.z}");
});

test("optional returned into an optional input runs", async () => {
  const result = await run(namedLoop("Optional(int)", "Optional(int)", "null", "${input.z}"));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: null });
});

test("a null seed into an optional input loads", () => {
  load("Optional(int)", "Optional(int)", "${null}");
});

test("null seed into an optional input runs", async () => {
  const result = await run(namedLoop("Optional(int)", "Optional(int)", "null", "${null}"));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: null });
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
    assembleText(recordLoop("${input.p}")),
  );

  expect(error.message).toBe(
    "loop node 'repeat': the body 'step' returns 'n' as Optional({\"x\":int}), but takes 'n' as {\"x\":int}; " +
      "make the input Optional(P), or tag the seed to leave the check to runtime: `n: !P ${input.p}`",
  );
});

test("a typed seed with an optional typedef field loads", () => {
  assembleText(recordLoop("!P ${input.p}"));
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
          n: !int
          m: !int
        code: |
          return {"n": inputs["n"] + 1, "m": 0}
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
  );
});

/** A loop whose body takes `n` and `m` but returns only `n`. */
const PARTIAL = `
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
          n: !int
        code: |
          return {"n": inputs["n"] + 1}
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
`;

test("a partial loop loads", () => {
  assembleText(PARTIAL);
});

test("a partial loop runs", async () => {
  const result = await run(PARTIAL);

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n, m: 5n });
});

test("a loop inside a called body is rejected", () => {
  const text = `
id: nested
name: nested
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
          n: !Optional(int)
        code: |
          return {"n": null}
    output: \${bump.output}
  outer:
    input:
      z: !int
    nodes:
      repeat:
        kind: loop
        times: 2
        call: step
        input:
          n: \${input.z}
    output: \${repeat.output}
nodes:
  run:
    kind: call
    call: outer
    input:
      z: \${input.z}
output: \${run.output}
`;

  const error = loadError(() => assembleText(text)).message;

  expect(error).toContain("defs entry 'outer': ");
  expect(error).toContain(nullableMessage("${input.z}"));
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
    assembleText(text),
  ).diagnostic.frames;

  expect(frames).toHaveLength(1);
  expect(frames[0]!.range.startLine).toBe(line);
});

// ---------- Pinned harvested cases ----------

test.each<CaseRow>([
  {
    id: "load-loop/compose_named_loop_carried_types__any_returned_into_an_int_input_is_rejected",
    call: assembled,
  },
  {
    id: "load-loop/compose_named_loop_carried_types__untyped_seed_rejects_an_optional_typedef_field",
    call: assembled,
  },
  {
    id: "load-loop/compose_named_loop_carried_types__a_returned_field_the_body_does_not_take_loads",
    call: assembled,
  },
  {
    id: "load-loop/compose_named_loop_carried_types__int_returned_into_a_float_input_loads",
    call: assembled,
  },
  {
    id: "load-loop/compose_named_loop_carried_types__typed_seed_with_an_optional_typedef_field_loads",
    call: assembled,
  },
  { id: "load-loop/compose_named_loop_carried_types__a_loop_in_a_def_gets_the_carried_type_check", call: assembled, full: true },
])("$id", runCase);
