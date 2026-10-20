/**
 * `case` `then:/else: call(...)` inline-call branch targets: a port of
 * `tests/engine/test_case_call.py`.
 *
 * A branch target may be an inline call (a fresh owned call) instead of a placed node id; it
 * desugars to a synth `__call_<n>` node and the then:/else: is rewritten to that synth id, so the
 * case veto skip-floods it when the branch is not taken (a nested call's inner calls follow it).
 * Accepted grammar: a whole-value `call(...)` directive (a route target is one node id).
 *
 * Changed: the flows assemble with `assembleText`, each `defs:` callee built by hand (`handChild`)
 * with the signature the row reads, where Python loads them.
 * `test_private_callee_in_case_target_is_rejected_at_branch_before_resolution` pins the whole
 * message; the scope's resolver throws a plain `Error` for any callee, so a private callee that
 * reached resolution would fail the row without a monkeypatch.
 *
 * Merged: `test_inline_binding_and_then_call_directive_share_minter_no_collision` into
 * `test_inline_binding_and_then_call_share_minter_no_collision`; both assemble the same flow.
 *
 * TS-only: the harvested `load-case/engine_case_call__a_private_callee_branch_is_rejected`,
 * pinned at the end.
 *
 * Later (T3-20, `case_branches`, which runs before this desugar):
 * `test_non_bare_then_call_is_rejected`, `test_non_bare_then_call_directive_is_rejected`,
 * `test_malformed_branch_call_directive_is_a_located_load_error`,
 * `test_quoted_malformed_branch_call_is_a_value_branch`,
 * `test_then_span_flow_call_target_is_rejected` and
 * `test_malformed_branch_call_holding_a_span_offers_no_text_rewrite`.
 * Later (T4): `test_then_call_directive_branch_runs_taken_value`,
 * `test_else_call_directive_branch_runs_taken_value`,
 * `test_case_output_over_call_directive_then_taken`,
 * `test_case_output_over_call_directive_else_taken`,
 * `test_nested_then_call_directive_branch_runs_taken_value` and
 * `test_the_call_directive_a_span_flow_call_points_at_loads` (runs).
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { INLINE_FLOW_PREFIX } from "../../src/compose/desugar/inline_bodies.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

/** A def built by hand: `input:` text `inputs`, one producer `x` of type `str`. */
function def(inputs: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n${inputs}`, new Map()),
    producers: { x: new ValueContract(typeFor("str", new Map())) },
    output: "${x.output}",
  });
}

const TAKE = def("  stance: !str\n");

// An in-file `take` def (the branch callable): echoes its stance via the `took` CODE fn.
const TAKE_DEF = `
defs:
  take:
    input:
      stance: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:took
        input:
          stance: \${input.stance}
        output: str
    output: \${x.output}
`;

/** Assemble `text` with the hand-built callees `children`; return the compiled flow. */
const load = (text: string, children: Record<string, ComposedFlow> = { take: TAKE }) =>
  assembleText(text, "flow.yaml", { children }).flow.compiled;

/** The ids of the synth call nodes of a compiled flow. */
const synthIds = (compiled: CompiledFlow) =>
  [...compiled.nodes.keys()].filter((nodeId) => nodeId.startsWith("__call_"));

// An inline-binding call(...) AND a then: call(...) in one flow: the two share one minter.
const BOTH = `
id: both
name: both
input:
  score: !float
  topic: !str
defs:
  take:
    input:
      stance: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:took
        input:
          stance: \${input.stance}
        output: str
    output: \${x.output}
  enrich:
    input:
      topic: !str
    nodes:
      y:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${y.output}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${input.topic})
    output: str
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance="pro")
    else: use
output: \${gate.output}
`;

test("inline binding and then call share minter no collision", () => {
  const synth = synthIds(load(BOTH, { take: TAKE, enrich: def("  topic: !str\n") }));
  expect(synth).toEqual(["__call_0", "__call_1"]); // distinct ids, no collision
});

// ---------- the `call(...)` directive spelling of a then:/else: branch target ----------

const FLOW_DIRECTIVE = `
id: cc
name: cc
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance="pro")
    else: call(take, stance="con")
output: \${gate.output}
`;

test("then else call directive desugars to synth branch nodes", () => {
  const compiled = load(FLOW_DIRECTIVE);
  // Both then: and else: call(...) directives became synth call nodes.
  const synth = synthIds(compiled);
  expect(synth).toEqual(["__call_0", "__call_1"]);
  // The gate's control edges target the synth ids (not a placed user node).
  const controlTargets = compiled.edges
    .filter((edge) => edge.sourceHandle !== null && edge.from === "gate")
    .map((edge) => edge.to);
  expect(controlTargets).toEqual(["__call_0", "__call_1"]);
});

test("nested then call directive lifts two synth nodes", () => {
  // The nested then: target lifts BOTH the inner and outer call as synth nodes; the else: lifts one
  // more: three synth call nodes in all, the gate's then-edge on the outer.
  const text = FLOW_DIRECTIVE.replace(
    'then: call(take, stance="pro")',
    'then: call(take, stance=call(take, stance="pro"))',
  );
  expect(synthIds(load(text))).toEqual(["__call_0", "__call_1", "__call_2"]);
});

test.each(["then", "else"])("private callee in case target is rejected at branch before resolution: %s", (branch) => {
  const privateName = INLINE_FLOW_PREFIX + "c".repeat(64);
  const thenTarget = branch === "then" ? `call(${privateName})` : "safe";
  const elseTarget = branch === "else" ? `call(${privateName})` : "safe";
  const text = `id: private_case
name: private_case
nodes:
  safe:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: safe
    output: str
  gate:
    kind: case
    cases:
      - when: "true"
        then: ${thenTarget}
    else: ${elseTarget}
output: \${gate.output}
`;
  const error = loadError(() => load(text, {}));
  const expectedLine = text.split("\n").findIndex((line) => line.includes(privateName)) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(expectedLine);
  expect(error.message).toBe(
    `case node 'gate' ${branch}:: call targets beginning with '__inline_flow_' are reserved for ` +
      "compiler-generated inline child flows",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([{ id: "load-case/engine_case_call__a_private_callee_branch_is_rejected", call: assembled }])(
  "assemble: $id",
  runCase,
);
