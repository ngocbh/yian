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
 * `test_declared_seed_type_is_compared_with_the_returned_field` and the harvested case
 * `load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field`.
 *
 * Ported through `loadFlow` and `runFlow`:
 * `test_typed_binding_narrows_a_nullable_source_into_a_strict_child`,
 * `test_typed_binding_rejects_a_null_at_runtime`, `test_declared_seed_type_types_an_opaque_producer`,
 * the untyped flow of `test_declared_seed_type_is_checked_on_the_loop_node` (its declared-int flow
 * only loads here; its run is covered by
 * `run-core/compose_typed_bindings_downstream__declared_seed_type_fails_on_the_loop_node`), and the
 * inline-body rows `test_typed_inline_binding_skips_inference_of_an_opaque_expression`,
 * `test_typed_inline_binding_types_an_empty_list`,
 * `test_typed_inline_loop_seed_types_the_body_parameter`,
 * `test_typed_inline_loop_seed_types_an_opaque_producer` and
 * `test_typed_inline_map_item_binding_runs`, each reading the inline body's parameters from the
 * node's `childInputs`, where Python reads the child's START parameters.
 *
 * Pinned in `engine/inline_composite_bodies.test.ts` through its case:
 * `test_untyped_inline_map_item_over_an_opaque_list_fails_to_load`.
 *
 * Changed: the error rows pin the whole message, where Python matches a fragment. CODE bodies are
 * TS; ints reach them as numbers and a flow's int output is a `bigint`. In
 * `test_typed_inline_loop_seed_types_the_body_parameter` the carried `r` ends as the int `1n`
 * (Python `1.0`): the body's number 1 is read as an int, and a whole number in a `float` slot
 * stays an int (decisions 30 and 38).
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { forget } from "../../src/defaults.ts";
import type { FailureCode } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { Parameter } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
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
          return inputs["x"] * 2
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
const INT_PRODUCER = "  i:\n    kind: code\n    output: int\n    code: |\n      return 3\n";

/** An `opaque` CODE producer with no declared output that returns `value`. */
const opaque = (value: string) => `  opaque:\n    kind: code\n    code: |\n      return ${value}\n`;

/** Assemble `text`. */
const load = (text: string) => assembleText(text);

/** Load `text` with `loadFlow` and run it with no inputs. */
const run = async (text: string) => runFlow(await loadFlow(text), {});

/** Load and run `text`, returning its output and expecting success. */
async function ok(text: string): Promise<unknown> {
  const result = await run(text);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/** Expect `result` to have failed with `code` at `nodeId`. */
function expectFailed(result: RunResult, code: FailureCode, nodeId: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe(code);
  expect(result.failure?.nodeId).toBe(nodeId);
}

// ---------- CALL ----------

test("an untyped nullable source into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("${maybe.output}", maybe("3"))));

  expect(error.message).toBe("call node 'run': binding 'x' — child expects 'int', source is 'Optional(int)'");
});

test("a typed binding narrows a nullable source into a strict child", async () => {
  expect(await ok(callFlow("!int ${maybe.output}", maybe("3")))).toBe(6n);
});

test("a typed binding rejects a null at runtime", async () => {
  expectFailed(await run(callFlow("!int ${maybe.output}", maybe("null"))), "binding.type", "run");
});

test("a typed nullable binding into a strict child fails to load", () => {
  const error = loadError(() => load(callFlow("!Optional(int) ${maybe.output}", maybe("3"))));

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
          return inputs["n"] + 1
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
  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1"))));

  expect(error.message).toBe("loop node 'loop': binding 'n' — child expects 'int', source is 'str'");
});

test("a declared seed type is compared with the returned field", () => {
  // The producer is untyped, so only the declaration gives the seed a type: the same seed written
  // without one loads. The body takes `n: Any`, so both the seed and the returned `int` fit its input.
  load(loopFlow("${opaque.output}", opaque("1"), "!Any"));

  const error = loadError(() => load(loopFlow("!str ${opaque.output}", opaque("1"), "!Any")));

  expect(error.message).toBe(
    "loop node 'loop': carried field 'n' — body output is 'int', carried source is 'str' " +
      "(a returned field must keep its carried type)",
  );
});

test("a declared seed type types an opaque producer", async () => {
  expect(await ok(loopFlow("!int ${opaque.output}", opaque("1")))).toEqual({ n: 3n });
});

test("a declared seed type is checked on the loop node", async () => {
  // Untyped, the text reaches the body's own `int` parameter, which does not convert it across
  // the LOOP edge, so the body's start fails.
  expectFailed(await run(loopFlow("${opaque.output}", opaque('"1"'))), "contract.input", "loop#0/__start__");
  load(loopFlow("!int ${opaque.output}", opaque('"1"')));
});

// ---------- inline bodies ----------

/** The parameters of the inline body of CALL, MAP or LOOP `nodeId` in `flow`, keyed by name. */
function bodyInputs(flow: ComposedFlow, nodeId: string): Map<string, Parameter> {
  const node = flow.compiled.nodes.get(nodeId) as CallNode | MapNode | LoopNode;
  return new Map(node.childInputs.map((param) => [param.name, param]));
}

/** Load `text`, expect `nodeId`'s body parameter `name` typed `type`, and return the loaded flow. */
async function typedBody(text: string, nodeId: string, name: string, type: string): Promise<ComposedFlow> {
  const flow = await loadFlow(text);
  expect(bodyInputs(flow, nodeId).get(name)!.contract!.type).toStrictEqual(typeFor(type, new Map()));
  return flow;
}

/** Run the loaded `flow` with no inputs, returning its output and expecting success. */
async function okLoaded(flow: ComposedFlow): Promise<unknown> {
  const result = await runFlow(flow, {});
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

test("a typed inline binding skips inference of an opaque expression", async () => {
  // A declared type types the inline parameter even when its value cannot be inferred.
  const flow = await typedBody(
    `id: inline-typed
name: inline_typed
nodes:
  a:
    kind: code
    code: |
      return 4
  wrap:
    kind: call
    input:
      n: !int \${a.output + 1}
    nodes:
      twice:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] * 2
    output: \${twice.output}
output: \${wrap.output}
`,
    "wrap",
    "n",
    "int",
  );

  expect(bodyInputs(flow, "wrap").get("n")!.required).toBe(true);
  expect(await okLoaded(flow)).toBe(10n);
});

test("a typed inline binding types an empty list", async () => {
  // An empty list literal takes its element type from the declaration.
  const flow = await typedBody(
    `id: inline-empty
name: inline_empty
nodes:
  wrap:
    kind: call
    input:
      items: !list(str) []
    nodes:
      count:
        kind: code
        input:
          items: \${input.items}
        output: int
        code: |
          return inputs["items"].length
    output: \${count.output}
output: \${wrap.output}
`,
    "wrap",
    "items",
    "list(str)",
  );

  expect(await okLoaded(flow)).toBe(0n);
});

test("a typed inline loop seed types the body parameter", async () => {
  // A `float` seed lets an inline body carry a float past its int literal seed.
  const flow = await loadFlow(`id: inline-float-loop
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
  // The body returns the number 1, which is read as the int 1 and stays an int in the `float` slot.
  expect(await okLoaded(flow)).toEqual({ r: 1n });
});

test("a typed inline loop seed types an opaque producer", async () => {
  // An `Optional(str)` seed over an untyped producer loads and carries its value.
  const flow = await loadFlow(`id: inline-optional-seed
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
          return inputs["n"] + 1
    output:
      n: \${step.output}
output: \${loop.output}
`);

  const param = bodyInputs(flow, "loop").get("label")!;
  expect(param.contract!.type).toStrictEqual(typeFor("Optional(str)", new Map()));
  expect(param.required).toBe(false);
  expect(await okLoaded(flow)).toEqual({ n: 2n, label: "seed" });
});

test("a typed inline map item binding runs", async () => {
  // A typed `${item}` binding types an inline MAP body over an untyped producer.
  const flow = await typedBody(
    `id: inline-map-item
name: inline_map_item
nodes:
${opaque('["a", "b"]')}  each:
    kind: map
    over: \${opaque.output}
    input:
      t: !str \${item}
    nodes:
      shout:
        kind: code
        input:
          t: \${input.t}
        output: str
        code: |
          return inputs["t"].toUpperCase()
    output: \${shout.output}
output: \${each.output}
`,
    "each",
    "t",
    "str",
  );

  expect(await okLoaded(flow)).toEqual(["A", "B"]);
});

// ---------- harvested conformance cases ----------

test.each([
  {
    id: "load-types/compose_typed_bindings_downstream__seed_type_is_compared_with_the_returned_field",
    call: assembled,
  },
])("assemble: $id", runCase);
