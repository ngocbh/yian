/**
 * An inline LOOP body's returned fields must fit the parameters generated from its seeds: a port
 * of `tests/compose/test_inline_loop_carried_types.py`, loading each flow with `assembleText`
 * (`assemble` as the loader calls it) where Python calls `load_flow`.
 *
 * An inline body becomes a child flow whose parameters are typed from the seeds, and each returned
 * field is bound to its parameter again on the next iteration. A returned field that can never fit
 * that parameter is a load error, so loading accepts only flows that can run.
 *
 * Ported: `test_untyped_null_seed_with_a_nullable_list_body_is_rejected_with_a_hint`,
 * `test_untyped_null_seed_error_points_at_the_seed`, `test_int_seed_with_a_float_body_field_is_rejected`,
 * `test_a_typed_seed_names_its_declared_type`,
 * `test_untyped_seed_with_an_optional_body_field_is_rejected_with_a_hint`,
 * `test_typed_str_seed_with_an_int_body_field_is_rejected`,
 * `test_literal_seed_with_a_str_body_field_is_rejected`; the load halves of
 * `test_typed_null_seed_with_a_nullable_list_body_runs`,
 * `test_int_seed_with_an_int_body_field_loads_and_runs`, `test_a_field_the_body_omits_is_not_checked`,
 * `test_typed_seed_with_an_optional_body_field_loads` and
 * `test_typed_seed_with_a_returned_null_fails_the_next_iteration`.
 *
 * Changed: the inline CODE bodies are TS, as the harvested cases write them. The error rows pin the
 * whole message where Python matches fragments.
 *
 * Pinned at the end (`assembled`, the body inline):
 * `load-loop/compose_inline_loop_carried_types__{literal_seed_with_a_str_body_field_is_rejected,
 * typed_str_seed_rejects_an_int_body_field}`.
 *
 * Pinned in skipped_flow_outputs.test.ts, its file's mirror:
 * `load-case/engine_skipped_flow_outputs__inline_loop_over_an_untyped_seed_asks_for_its_type` (its
 * `Optional(str)` comes from skipping the CASE-gated body node).
 * Later (T4): the run halves of the load-half rows above.
 */

import { expect, test } from "bun:test";

import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

/** A flow whose inline LOOP body returns `seen` as `Optional(list(str))`, seeded by `seed`. */
function seenLoop(seed: string): string {
  return `
id: seen
name: seen
nodes:
  repeat:
    kind: loop
    while: "n < 2"
    max: 5
    input:
      seen: ${seed}
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          seen: Optional(list(str))
          n: int
        code: |
          return {"seen": ["a", "b"].slice(0, Number(inputs["n"]) + 1), "n": inputs["n"] + 1n}
    output: \${bump.output}
output: \${repeat.output}
`;
}

/**
 * A flow whose inline LOOP carries `n`, seeded from an `int` flow input.
 *
 * @param bodyOutput The body CODE node's declared output type.
 * @param bodyValue The TS expression the body node returns as `n`.
 */
function numberLoop(bodyOutput: string, bodyValue: string): string {
  return `
id: number
name: number
input:
  z: !int
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: \${input.z}
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: ${bodyOutput}
        code: |
          return {"n": ${bodyValue}}
    output: \${bump.output}
output: \${repeat.output}
`;
}

/**
 * A flow whose inline LOOP body returns `note` from one CODE node.
 *
 * @param seed The LOOP's `note:` seed binding, e.g. `!str "start"`.
 * @param bodyOutput The body CODE node's declared output type.
 * @param bodyValue The TS expression the body node returns.
 */
function noteLoop(seed: string, bodyOutput: string, bodyValue: string): string {
  return `
id: note
name: note
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: ${seed}
    nodes:
      mk:
        kind: code
        input:
          note: \${input.note}
        output: ${bodyOutput}
        code: |
          return ${bodyValue}
    output:
      note: \${mk.output}
output: \${repeat.output}
`;
}

/** The message of the load error assembling `text` raises. */
const message = (text: string) => loadError(() => assembleText(text)).message;

test("an untyped null seed with a nullable list body is rejected with a hint", () => {
  expect(message(seenLoop("${null}"))).toBe(
    "loop node 'repeat': the body returns 'seen' as Optional(list(str)), but the carried field is typed " +
      "None from its seed; declare the seed's type: `seen: !Optional(list(str)) ${null}`",
  );
});

test("an untyped null seed error points at the seed", () => {
  const text = seenLoop("${null}");
  const line = text.split("\n").indexOf("      seen: ${null}") + 1;

  const frames = loadError(() => assembleText(text)).diagnostic.frames;

  expect(frames).toHaveLength(1);
  expect(frames[0]!.range.startLine).toBe(line);
});

test("a typed null seed with a nullable list body loads", () => {
  // Python's test_typed_null_seed_with_a_nullable_list_body_runs; the run half is T4's.
  assembleText(seenLoop("!Optional(list(str)) ${null}"));
});

test("an int seed with a float body field is rejected", () => {
  expect(message(numberLoop("float", 'inputs["n"] + 1.5'))).toBe(
    "loop node 'repeat': the body returns 'n' as float, but the carried field is typed int from its seed; " +
      "declare the seed's type: `n: !float ${input.z}`",
  );
});

test("an int seed with an int body field loads", () => {
  // Python's test_int_seed_with_an_int_body_field_loads_and_runs; the run half is T4's.
  assembleText(numberLoop("int", 'inputs["n"] + 1n'));
});

test("a typed seed names its declared type", () => {
  const text = numberLoop("float", 'inputs["n"] + 1.5').replace("n: ${input.z}", "n: !int ${input.z}");

  expect(message(text)).toBe(
    "loop node 'repeat': the body returns 'n' as float, but the carried field is declared int; " +
      "declare the seed's type: `n: !float ${input.z}`",
  );
});

test("a field the body omits is not checked", () => {
  // The load half; the run half is T4's.
  assembleText(`
id: partial
name: partial
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: \${null}
      n: 0
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
output: \${repeat.output}
`);
});

test("a typed seed with an optional body field loads", () => {
  assembleText(noteLoop('!str "start"', "Optional(str)", 'inputs["note"] + "!"'));
});

test("a typed seed with a returned null loads", () => {
  // Python's test_typed_seed_with_a_returned_null_fails_the_next_iteration: the null fails the
  // next iteration's input check at run time (T4).
  assembleText(noteLoop('!str "start"', "Optional(str)", "null"));
});

test("an untyped seed with an optional body field is rejected with a hint", () => {
  expect(message(noteLoop('"start"', "Optional(str)", 'inputs["note"] + "!"'))).toBe(
    "loop node 'repeat': the body returns 'note' as Optional(str), but the carried field is typed str " +
      'from its seed; declare the seed\'s type: `note: !Optional(str) "start"`',
  );
});

test("a typed str seed with an int body field is rejected", () => {
  expect(message(noteLoop('!str "start"', "int", "1"))).toBe(
    "loop node 'repeat': carried field 'note' — body output is 'int', carried source is 'str' " +
      "(a returned field must keep its carried type)",
  );
});

test("a literal seed with a str body field is rejected", () => {
  expect(message("typedefs:\n  AB: Literal(a, b)\n" + noteLoop("!AB a", "str", '"a"'))).toBe(
    "loop node 'repeat': the body returns 'note' as str, but the carried field is declared Literal(a, b); " +
      'declare the seed\'s type: `note: !str "a"`',
  );
});

// ---------- Pinned harvested cases ----------

test.each<CaseRow>([
  {
    id: "load-loop/compose_inline_loop_carried_types__literal_seed_with_a_str_body_field_is_rejected",
    call: assembled,
  },
  { id: "load-loop/compose_inline_loop_carried_types__typed_str_seed_rejects_an_int_body_field", call: assembled },
])("$id", runCase);
