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
 * `test_declared_seed_type_is_checked_against_the_body_input`,
 * `test_declared_seed_type_is_compared_with_the_returned_field`, and the load halves of
 * `test_typed_binding_narrows_a_nullable_source_into_a_strict_child`,
 * `test_typed_binding_rejects_a_null_at_runtime`,
 * `test_declared_seed_type_types_an_opaque_producer`,
 * `test_declared_seed_type_is_checked_on_the_loop_node` (both flows),
 * `test_typed_inline_loop_seed_types_the_body_parameter` and
 * `test_typed_inline_loop_seed_types_an_opaque_producer` (each reading the inline body's parameters
 * from the LOOP node's `childInputs`, where Python reads the child's START parameters); and the
 * harvested case
 * `load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field`.
 *
 * Changed: the error rows pin the whole message, where Python matches a fragment. CODE bodies are
 * TS; load doesn't run them.
 *
 * Not ported (decided by T3-17's inline inputs, not T3-19 rows; the run halves of the others are
 * T4's): `test_typed_inline_binding_skips_inference_of_an_opaque_expression`,
 * `test_typed_inline_binding_types_an_empty_list`,
 * `test_untyped_inline_map_item_over_an_opaque_list_fails_to_load` (its case is pinned in
 * `engine/inline_composite_bodies.test.ts`) and `test_typed_inline_map_item_binding_runs`.
 *
 * Later (T4-8 follow-up): the run halves of the rows above.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import type { Parameter } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { assembled, assembleText, loadError, runCase } from "./declarations.ts";

afterEach(() => forget());

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

/** Assemble `text`. */
const load = (text: string) => assembleText(text);

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

test("a declared seed type is checked against the body input", () => {
  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1n"))));

  expect(error.message).toBe("loop node 'loop': binding 'n' — child expects 'int', source is 'str'");
});

test("a declared seed type is compared with the returned field", () => {
  // The producer is untyped, so only the declaration gives the seed a type: the same seed written
  // without one loads. The body takes `n: Any`, so both the seed and the returned `int` fit its input.
  load(loopFlow("${opaque.output}", opaque("1n"), "!Any"));

  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1n"), "!Any")));

  expect(error.message).toBe(
    "loop node 'loop': carried field 'n' — body output is 'int', carried source is 'str' " +
      "(a returned field must keep its carried type)",
  );
});

test("a declared seed type types an opaque producer (loads)", () => {
  load(loopFlow("!int ${opaque.output}", opaque("1n")));
});

test.each([
  ["untyped", "${opaque.output}"],
  ["declared int", "!int ${opaque.output}"],
])("a declared seed type is checked on the loop node (loads): %s", (_name, seed) => {
  load(loopFlow(seed, opaque('"1"')));
});

// ---------- inline bodies ----------

/** The parameters of the inline body of LOOP `nodeId` in `flow`, keyed by name. */
function bodyInputs(flow: ComposedFlow, nodeId: string): Map<string, Parameter> {
  const node = flow.compiled.nodes.get(nodeId) as LoopNode;
  return new Map(node.childInputs.map((param) => [param.name, param]));
}

test("a typed inline loop seed types the body parameter (loads)", () => {
  // A `float` seed lets an inline body carry a float past its int literal seed.
  const { flow } = assembleText(`id: inline-float-loop
name: inline_float_loop
nodes:
  loop:
    kind: loop
    times: 2
    input:
      r: !float 0
    nodes:
      half:
        kind: code
        input:
          r: \${input.r}
        output: float
        code: |
          return inputs["r"] + 0.5
    output:
      r: \${half.output}
output: \${loop.output}
`);

  expect(bodyInputs(flow, "loop").get("r")!.contract!.type).toStrictEqual(typeFor("float", new Map()));
});

test("a typed inline loop seed types an opaque producer (loads)", () => {
  // An `Optional(str)` seed over an untyped producer loads.
  const { flow } = assembleText(`id: inline-optional-seed
name: inline_optional_seed
nodes:
  p:
    kind: code
    code: |
      return "seed"
  loop:
    kind: loop
    times: 2
    input:
      n: 0
      label: !Optional(str) \${p.output}
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
output: \${loop.output}
`);

  const param = bodyInputs(flow, "loop").get("label")!;
  expect(param.contract!.type).toStrictEqual(typeFor("Optional(str)", new Map()));
  expect(param.required).toBe(false);
});

// ---------- harvested conformance cases ----------

test.each([
  {
    id: "load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field",
    call: assembled,
  },
])("assemble: $id", runCase);
