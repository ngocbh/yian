/**
 * The CALL child-run seam, end to end over CODE-only children: a port of
 * `tests/engine/test_call_run.py`.
 *
 * A plain `call` binds its declared args against the parent pool, seeds a child pool (plus the
 * child's own declared defaults for omitted args), runs the grown child and re-exports the
 * child's single value.
 *
 * Ported: `test_call_runs_child_and_reexports_value`,
 * `test_call_output_projection_preserves_nullable_parent_path` (its run is also the harvested
 * `run-map-call/engine_call_run__call_output_reads_through_a_null_parent`; the case doesn't pin
 * the output contract), `test_call_child_failure_fails_the_run`,
 * `test_call_child_applies_its_own_default`, `test_call_boundary_assert_sees_omitted_default`,
 * `test_call_arg_text_is_not_converted_for_a_typed_child_input` and
 * `test_call_run_expands_not_child_engine`. Integers are bigints (decision 4a); the CODE
 * references are the TS fixtures' (`./fixtures/compose_codefns.ts:<fn>`).
 *
 * Not ported: `test_call_accepts_none_elements_for_nullable_list_sink`: covered by
 * `run-map-call/engine_call_run__call_accepts_null_elements_for_a_nullable_list`.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { Grow } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import type { CodeNode } from "../../src/nodes/code/node.ts";
import type { ValueContract } from "../../src/nodes/contracts.ts";
import { ns } from "../../src/nodes/ids.ts";
import { runFlow } from "../../src/run.ts";
import type { JsonValue } from "../../src/typesys/json.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { childFlow, childResolver } from "./run_state_fixtures.ts";

// A CODE-only child: topic -> {report, n}.
const CHILD = `
id: child-one
name: child_one
input:
  topic: !str
  suffix: !str "!"
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output:
      report: str
      n: int
    code: ./fixtures/compose_codefns.ts:make_report
output:
  report: \${emit.output.report}
  n: \${emit.output.n}
`;

const BOOM_CHILD = `
id: boom-child
name: boom_child
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./fixtures/compose_codefns.ts:boom
output: \${emit.output}
`;

const CALL_PARENT = `
id: ref-parent
name: ref_parent
input:
  topic: !str
uses:
  child-one: child-one
nodes:
  research:
    kind: call
    call: child-one
    input:
      topic: \${input.topic}
output:
  report: \${research.output.report}
  n: \${research.output.n}
`;

const CALL_BOOM = `
id: ref-boom
name: ref_boom
input:
  topic: !str
uses:
  boom-child: boom-child
nodes:
  research:
    kind: call
    call: boom-child
    input:
      topic: \${input.topic}
output: \${research.output}
`;

test("call runs child and reexports value", async () => {
  const parent = await loadConformanceFlow(CALL_PARENT, { childResolver: childResolver({ "child-one": CHILD }) });
  const result = await runFlow(parent, { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  // The CALL node re-exports the child's {report, n}; the parent re-keys it.
  expect(result.output).toEqual({ report: "report for ACME", n: 4n });
});

test("call output projection preserves nullable parent path", async () => {
  // A dotted read through a nullable record may itself evaluate to null.
  const parent = await loadConformanceFlow(`id: nullable-projection
name: nullable_projection
typedefs:
  Payload:
    value: int
input:
  payload: !Optional(Payload)
defs:
  pick:
    input:
      payload: !Optional(Payload)
    nodes: {}
    output: \${input.payload.value}
nodes:
  invoke:
    kind: call
    call: pick
    input:
      payload: \${input.payload}
output: \${invoke.output}
`);

  const contract = parent.compiled.nodes.get("invoke")!.outputContract;
  expect(contract).not.toBeNull();
  expect((contract as ValueContract).type.nullable).toBe(true);

  const result = await runFlow(parent, { payload: null });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBeNull();
});

test("call child failure fails the run", async () => {
  const parent = await loadConformanceFlow(CALL_BOOM, { childResolver: childResolver({ "boom-child": BOOM_CHILD }) });
  const result = await runFlow(parent, { topic: "ACME" });
  expect(result.status).not.toBe("succeeded");
  expect(result.failure!.message).toContain("boom");
  const call = parent.compiled.nodes.get("research") as CallNode;
  expect(result.failure!.nodeId).toBe("research/emit");
  const emit = (call.child as CompiledFlow).nodes.get("emit") as CodeNode;
  expect(result.failure!.trace).toEqual([call.origin!, emit.codeOrigin!]);
});

test("call child applies its own default", async () => {
  // The child declares `suffix: !str "!"`, which the parent does not bind; the child's own
  // default fills it (the default-seeding path runs without the parent supplying it).
  const parent = await loadConformanceFlow(CALL_PARENT, { childResolver: childResolver({ "child-one": CHILD }) });
  const result = await runFlow(parent, { topic: "X" });
  expect(result.status).toBe("succeeded");
  expect((result.output as Record<string, unknown>).n).toBe(1n);
});

const CHILD_BOUNDARY_DEFAULT = `
id: child-bd
name: child_bd
input:
  topic: !str
  window: !int 30
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: ./fixtures/compose_codefns.ts:echo
output: \${emit.output}
asserts:
  - input.window >= 1
`;

const CALL_PARENT_BD = `
id: ref-parent-bd
name: ref_parent_bd
input:
  topic: !str
uses:
  child-bd: child-bd
nodes:
  research:
    kind: call
    call: child-bd
    input:
      topic: \${input.topic}
output: \${research.output}
`;

test("call boundary assert sees omitted default", async () => {
  // The parent omits `window` (default 30), yet the child's boundary assert `input.window >= 1`
  // must see the effective input (30), not null: the child START applies the default before its
  // asserts run.
  const parent = await loadConformanceFlow(CALL_PARENT_BD, {
    childResolver: childResolver({ "child-bd": CHILD_BOUNDARY_DEFAULT }),
  });
  const result = await runFlow(parent, { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("ACME");
});

const CALL_PARENT_BD_TEXT = `
id: ref-parent-bdc
name: ref_parent_bdc
input:
  topic: !str
uses:
  child-bd: child-bd
nodes:
  research:
    kind: call
    call: child-bd
    input:
      topic: \${input.topic}
      window: "30"
output: \${research.output}
`;

test("call arg text is not converted for a typed child input", async () => {
  // A CALL edge is not a flow boundary: the literal "30" into the child's `int` input is not
  // converted, so the flow fails to load instead of the boundary assert ever comparing
  // `"30" >= 1`.
  const loading = loadConformanceFlow(CALL_PARENT_BD_TEXT, {
    childResolver: childResolver({ "child-bd": CHILD_BOUNDARY_DEFAULT }),
  });
  await expect(loading).rejects.toThrow(LoadError);
  await expect(loading).rejects.toThrow("binding 'window' — child expects 'int', but the value \"30\" does not fit");
});

test("call run expands not child engine", async () => {
  // The CallNode returns only a seed; compile reconstructs the child fragment for the engine.
  const child = childFlow();
  const n = new CallNode("r", { flowId: "c", child, childInputs: [] });
  const out = await n.run({ topic: "ACME" });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual({ topic: "ACME" });
  expect(buildGrow(n, (out as Grow).seed as JsonValue).startId).toBe(ns("r", child.startId));
});
