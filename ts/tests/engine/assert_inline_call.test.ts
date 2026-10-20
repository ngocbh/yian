/**
 * A named `call` node whose `.output` a flow `asserts:` expression reads: a port of
 * `tests/engine/test_assert_inline_call.py`.
 *
 * `dbl_call` runs the `dbl` def, and `dbl_call.output >= 0` is checked after the run. No synth node
 * is minted: the call is an ordinary, author-named node in the flow. The load half assembles the
 * flow (`assembleText`), the `dbl` def built by hand (`handChild`).
 *
 * Later (T4): `test_inline_call_assert_passes`, `test_inline_call_assert_fails_post` (runs).
 *
 * TS-only: the harvested `load-map-call/engine_assert_inline_call__named_call_node_assert_loads`,
 * pinned at the end.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembledWith, assembleText, handChild, parametersFromYaml, runCase } from "../compose/declarations.ts";

afterEach(() => forget());

// An in-file `dbl` def (the assert callee): doubles its int input via the `double` CODE function.
const DBL_DEF = `
defs:
  dbl:
    input:
      n: !int
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:double
        input:
          n: \${input.n}
        output: int
    output: \${x.output}
`;

const FLOW = `
id: ai
name: ai
input:
  v: !int
${DBL_DEF}
nodes:
  main:
    kind: code
    code: ./fixtures/compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
  dbl_call:
    kind: call
    call: dbl
    input:
      n: \${input.v}
output: \${main.output}
asserts:
  - "dbl_call.output >= 0"
`;

/** The `dbl` def built by hand: an int input `n` and an int output. */
function dbl(): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n", new Map()),
    producers: { x: new ValueContract(Type.scalar(ValueKind.INTEGER)) },
    output: "${x.output}",
  });
}

test("named call node assert loads", () => {
  const { flow } = assembleText(FLOW, "flow.yaml", { children: { dbl: dbl() } });
  const nodeIds = [...flow.compiled.nodes.keys()];
  expect(nodeIds).toContain("dbl_call"); // the assert reads an author-named call node
  expect(nodeIds.some((nodeId) => nodeId.startsWith("__call_"))).toBe(false); // no synth node minted
});

// ---------- harvested conformance cases ----------

test("assembled: load-map-call/engine_assert_inline_call__named_call_node_assert_loads", () => {
  runCase({
    id: "load-map-call/engine_assert_inline_call__named_call_node_assert_loads",
    call: assembledWith({ dbl: dbl() }),
  });
});
