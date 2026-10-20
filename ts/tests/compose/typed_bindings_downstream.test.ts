/**
 * Downstream checks see a typed binding's declared type: a port of
 * `tests/compose/test_typed_bindings_downstream.py`, assembling each flow with `assembleText` where
 * Python calls `load_flow`.
 *
 * A binding tagged with a type on a CALL, MAP or LOOP node declares the type its child receives,
 * so the cross-flow check compares that declared type with the child's parameters instead of the
 * type inferred from the value.
 *
 * Ported: `test_untyped_nullable_source_into_strict_child_fails_to_load`,
 * `test_typed_nullable_binding_into_strict_child_fails_to_load`,
 * `test_declared_float_does_not_fit_a_strict_int_child`, `test_map_binding_uses_the_declared_type`,
 * `test_declared_seed_type_is_checked_against_the_body_input`, and the load halves of
 * `test_typed_binding_narrows_a_nullable_source_into_a_strict_child` and
 * `test_typed_binding_rejects_a_null_at_runtime`.
 *
 * Changed: the `child` and `bump` defs are built by hand (`handChild`: the def's parameters read
 * from its `input:` text, its typed node as a producer with its output contract), since def bodies
 * load in T3-22. The error rows pin the whole message, where Python matches a fragment. CODE
 * bodies are TS; load doesn't run them.
 *
 * Not ported here, decided by T3-17's inline inputs (their run halves are T4's):
 * `test_typed_inline_binding_skips_inference_of_an_opaque_expression`,
 * `test_typed_inline_binding_types_an_empty_list`,
 * `test_untyped_inline_map_item_over_an_opaque_list_fails_to_load` (its case is pinned in
 * `engine/inline_composite_bodies.test.ts`) and `test_typed_inline_map_item_binding_runs`.
 *
 * Later (T3-19 loop_types): `test_declared_seed_type_is_compared_with_the_returned_field` and its
 * case `load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field`;
 * the load halves of `test_declared_seed_type_types_an_opaque_producer`,
 * `test_declared_seed_type_is_checked_on_the_loop_node`,
 * `test_typed_inline_loop_seed_types_the_body_parameter` and
 * `test_typed_inline_loop_seed_types_an_opaque_producer`.
 * Later (T4): the run halves of the rows above.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError, parametersFromYaml } from "./declarations.ts";

afterEach(() => forget());

const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));

const STRICT_CHILD = `defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: \${input.x}
        output: int
        code: |
          return inputs["x"] * 2n
    output: \${twice.output}
`;

/** The strict `child` def built by hand: input `x: !int`, one `twice: int` producer, returned. */
const CHILD = handChild({
  inputs: parametersFromYaml("input:\n  x: !int\n", new Map()),
  producers: { twice: INT },
  output: "${twice.output}",
});

/** A flow that calls the strict `child` with `x` bound as `binding`, after `producer`'s nodes. */
function callFlow(binding: string, producer: string): string {
  return `id: downstream
name: downstream
${STRICT_CHILD}nodes:
${producer}  run:
    kind: call
    call: child
    input:
      x: ${binding}
output: \${run.output}
`;
}

/** A `maybe` CODE producer declaring `Optional(int)` that returns `value`. */
const maybe = (value: string) =>
  `  maybe:\n    kind: code\n    output: Optional(int)\n    code: |\n      return ${value}\n`;

/** An `i` CODE producer declaring `int`. */
const INT_PRODUCER = "  i:\n    kind: code\n    output: int\n    code: |\n      return 3n\n";

/** An `opaque` CODE producer with no declared output that returns `value`. */
const opaque = (value: string) => `  opaque:\n    kind: code\n    code: |\n      return ${value}\n`;

/** Assemble `text` with `children` as its defs (default: the strict `child`). */
const load = (text: string, children: Record<string, ComposedFlow> = { child: CHILD }) =>
  assembleText(text, "flow.yaml", { children });

// ---------- CALL ----------

test("an untyped nullable source into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("${maybe.output}", maybe("3n"))));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'Optional(int)'");
});

test.each([
  ["narrows a nullable source into a strict child", "3n"],
  ["rejects a null at runtime", "null"],
])("a typed binding %s (loads)", (_name, value) => {
  load(callFlow("!int ${maybe.output}", maybe(value)));
});

test("a typed nullable binding into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("!Optional(int) ${maybe.output}", maybe("3n"))));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'Optional(int)'");
});

test("a declared float does not fit a strict int child", () => {
  const error = loadError(() => load(callFlow("!float ${i.output}", INT_PRODUCER)));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'float'");
});

test("a map binding uses the declared type", () => {
  const text = `id: downstream-map
name: downstream_map
${STRICT_CHILD}nodes:
${INT_PRODUCER}  each:
    kind: map
    over: [1, 2]
    call: child
    input:
      x: !float \${i.output}
output: \${each.output}
`;

  const error = loadError(() => load(text));

  expect(error.message).toBe("call node 'each': binding 'x' — child expects 'int', source is 'float'");
});

// ---------- LOOP ----------

/** A flow whose LOOP seeds `n` as `seed` into a named body `bump` that returns `n: int`. */
function loopFlow(seed: string, producer: string, bodyInput = "!int"): string {
  return `id: downstream-loop
name: downstream_loop
defs:
  bump:
    input:
      n: ${bodyInput}
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1n
    output:
      n: \${step.output}
nodes:
${producer}  loop:
    kind: loop
    times: 2
    input:
      n: ${seed}
    call: bump
output: \${loop.output}
`;
}

/** The `bump` def built by hand: input `n: <bodyInput>`, one `step: int` producer, returned as `n`. */
const bump = (bodyInput = "!int") =>
  handChild({
    inputs: parametersFromYaml(`input:\n  n: ${bodyInput}\n`, new Map()),
    producers: { step: INT },
    output: { n: "${step.output}" },
  });

test("a declared seed type is checked against the body input", () => {
  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1n")), { bump: bump() }));

  expect(error.message).toBe("loop node 'loop': binding 'n' — child expects 'int', source is 'str'");
});
