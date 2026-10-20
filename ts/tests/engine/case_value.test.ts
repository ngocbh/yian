/**
 * `${<case>.output}` = the taken-branch value: a port of `tests/engine/test_case_value.py`.
 *
 * A `case` read as a VALUE: `${<case>.output}` desugars at load to a COALESCE over the case's
 * branch targets (`${t1.output | t2.output | … | else.output}`). Pure loader sugar — skip-flood
 * makes every non-taken branch null, so the coalesce yields the taken value (exactly seed 02's
 * hand-written join). Runtime IR unchanged.
 *
 * Descriptor-level coverage exercises `expandCaseOutputs` over constructed descriptors: whole,
 * dotted, embedded, `on:` form, nested cases flattened to leaves, non-case, and fast-path
 * references. It also covers loud rejection of a case value in a non-clean binding position, a
 * condition, or an assert. The load rows assemble CODE branches with `output: ${gate.output}` and
 * with a node consuming `${gate.output}`.
 *
 * (The `then:/else: call(...)` inline-call branch form is covered by `case_call.test.ts`.)
 *
 * Changed: the load rows assemble with `assembleText` (a `defs:` callee built by hand with
 * `handChild`), where Python loads and runs; `test_seed_22_case_value_output_expanded` reads seed
 * 22 from the `load-case/seed_22_case_value` case. A rejection row pins the whole message where
 * Python matches a fragment. `test_no_case_fast_path_is_identity` checks the same descriptors in a
 * fresh map, since `expandCaseOutputs` returns a mutable `Map` for its read-only input. The load
 * halves of the `_e2e` rows are named for what they check, and the coalesce-position rejection row
 * uses a variant of the shared flow (the harvested case holds Python's `cv_bad` flow word for word).
 *
 * TS-only: the harvested `load-case/engine_case_value__case_value_in_a_coalesce_is_rejected` and
 * `load-case/seed_22_case_value`, pinned at the end.
 *
 * Later (T4): the run halves of `test_case_value_returns_taken_branch`,
 * `test_node_consuming_case_value_runs`, `test_case_of_case_value_runs_taken_nested_leaf` and
 * `test_case_value_in_inline_call_arg_runs`.
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { expandCaseOutputs } from "../../src/compose/desugar/case_outputs.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { Assertion, Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  caseText,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

// ---------- descriptor-level `expandCaseOutputs` coverage ----------

const REJECTED =
  "a case value can't be read in a coalesce operand, a default, an expression, a condition, or an assert; " +
  "read it as a whole binding or inside text";

function gate(then = "pro", else_ = "caut"): CaseDescriptor {
  return new CaseDescriptor({ id: "gate", cases: [{ when: "s.output >= 0.5", then }], else_ });
}

function code(nodeId: string, inputs: Record<string, unknown>): CodeDescriptor {
  return new CodeDescriptor({ id: nodeId, code: "m:f", inputs });
}

/** Run `expandCaseOutputs` over `descriptors` with no outputs or asserts; return the new map. */
function expanded(descriptors: Record<string, NodeDescriptor>): Map<string, NodeDescriptor> {
  return expandCaseOutputs(new Map(Object.entries(descriptors)), null, [])[0];
}

/** Return node `nodeId`'s input `name` after `expanded`. */
const inputOf = (descriptors: Map<string, NodeDescriptor>, nodeId: string, name: string) =>
  descriptors.get(nodeId)!.inputs[name];

/** Return the `LoadError` expanding `descriptors` raises. */
const expandFails = (descriptors: Record<string, NodeDescriptor>, asserts: readonly Assertion[] = []) =>
  loadError(() => expandCaseOutputs(new Map(Object.entries(descriptors)), null, asserts));

/** Return an assertion whose expression and display are `expression`. */
const assertion = (expression: string) => new Assertion(expression, expression);

test("whole ref expands to coalesce", () => {
  const out = expanded({ gate: gate(), c: code("c", { v: "${gate.output}" }) });
  expect(inputOf(out, "c", "v")).toBe("${pro.output | caut.output}");
});

test("dotted case ref expands per branch", () => {
  const out = expanded({ gate: gate(), c: code("c", { v: "${gate.output.report}" }) });
  expect(inputOf(out, "c", "v")).toBe("${pro.output.report | caut.output.report}");
});

test("embedded case ref expands", () => {
  const out = expanded({ gate: gate(), c: code("c", { v: "pre ${gate.output} post" }) });
  expect(inputOf(out, "c", "v")).toBe("pre ${pro.output | caut.output} post");
});

test("on form case value uses then targets", () => {
  const onGate = new CaseDescriptor({
    id: "gate",
    on: "${x.output.tag}",
    cases: [
      { when: "a", then: "na" },
      { when: "b", then: "nb" },
    ],
  });
  const out = expanded({ gate: onGate, c: code("c", { v: "${gate.output}" }) });
  expect(inputOf(out, "c", "v")).toBe("${na.output | nb.output}"); // no else -> just thens
});

test("outputs section ref expands", () => {
  const [, outputs] = expandCaseOutputs(new Map([["gate", gate()]]), "${gate.output}", []);
  expect(outputs).toBe("${pro.output | caut.output}");
});

test("case of case value flattens to leaves", () => {
  // A case whose target is itself a case: ${<outer>.output} flattens through the nested case to its
  // LEAF (non-case) targets (sound via the veto, which skip-floods the nested gate when the outer
  // branch loses).
  const outer = new CaseDescriptor({ id: "outer", cases: [{ when: "w", then: "inner" }], else_: "x" });
  const inner = new CaseDescriptor({ id: "inner", cases: [{ when: "w", then: "a" }], else_: "b" });
  const out = expanded({ outer, inner, c: code("c", { v: "${outer.output}" }) });
  expect(inputOf(out, "c", "v")).toBe("${a.output | b.output | x.output}");
});

test("case of case cycle drops arm is loud", () => {
  // g1 -> g2 -> g1: the cycle-guarded flatten drops the cyclic arm -> empty leaves -> loud.
  const g1 = new CaseDescriptor({ id: "g1", cases: [{ when: "w", then: "g2" }] }); // no else
  const g2 = new CaseDescriptor({ id: "g2", cases: [{ when: "w", then: "g1" }] }); // no else
  expect(expandFails({ g1, g2, c: code("c", { v: "${g1.output}" }) }).message).toBe(
    "node 'c': ${g1.output} — case 'g1' has no resolvable branch value (a case needs a `then:` or " +
      "`else:`, and its targets must not form a cycle)",
  );
});

test("non case ref untouched", () => {
  const out = expanded({ gate: gate(), c: code("c", { v: "${s.output}" }) });
  expect(inputOf(out, "c", "v")).toBe("${s.output}");
});

test("multiple case values in one binding both expand", () => {
  const g1 = new CaseDescriptor({ id: "g1", cases: [{ when: "w", then: "a" }], else_: "b" });
  const g2 = new CaseDescriptor({ id: "g2", cases: [{ when: "w", then: "c" }], else_: "d" });
  const out = expanded({ g1, g2, n: code("n", { v: "pre ${g1.output} mid ${g2.output} post" }) });
  expect(inputOf(out, "n", "v")).toBe("pre ${a.output | b.output} mid ${c.output | d.output} post");
});

test("case value in tool inputs expands", () => {
  const tool = new ToolDescriptor({ id: "t", toolId: "x", inputs: { q: "${gate.output}" } });
  const out = expanded({ gate: gate(), t: tool });
  expect(inputOf(out, "t", "q")).toBe("${pro.output | caut.output}");
});

test("case value in human input and loop inputs expands", () => {
  const hosts = [
    new HumanInputDescriptor({ id: "human", prompt: "Continue?", inputs: { q: "${gate.output}" } }),
    new LoopDescriptor({ id: "loop", call: "body", inputs: { q: "${gate.output}" } }),
  ];
  for (const host of hosts) {
    const out = expanded({ gate: gate(), [host.id]: host });
    expect(inputOf(out, host.id, "q")).toBe("${pro.output | caut.output}");
  }
});

test("case with no branch target is loud", () => {
  // A malformed case (no then:, no else:) referenced as a value -> a clear LoadError, not `${}`.
  const bad = new CaseDescriptor({ id: "gate", cases: [{ when: "w" }] }); // no then, no else
  expect(expandFails({ gate: bad, c: code("c", { v: "${gate.output}" }) }).message).toBe(
    "node 'c': ${gate.output} — case 'gate' has no resolvable branch value (a case needs a `then:` or " +
      "`else:`, and its targets must not form a cycle)",
  );
});

test("no case fast path is identity", () => {
  const descriptors = new Map<string, NodeDescriptor>([["c", code("c", { v: "${x.output}" })]]);
  const outputs = { r: "${c.output}" };
  const [newDescriptors, newOutputs] = expandCaseOutputs(descriptors, outputs, []);
  // Untouched (no case node): the same descriptors in a fresh map, and the same outputs.
  expect([...newDescriptors]).toEqual([...descriptors]);
  expect(newDescriptors.get("c")).toBe(descriptors.get("c")!);
  expect(newOutputs).toBe(outputs);
});

test.each([
  ["coalesce", "${gate.output | other.output}"],
  ["default", "${gate.output:-fallback}"],
  // A case ref as the default VALUE must be LOUD, not silently expanded.
  ["default value", "${x :- gate.output}"],
  // The `:?` message is a full expression, so a case ref in it is a genuine ref.
  ["required message", "${input.t :? gate.output}"],
])("case ref in %s is loud", (_position, value) => {
  expect(expandFails({ gate: gate(), c: code("c", { v: value }) }).message).toBe(
    `node 'c': \${gate.output} — ${REJECTED}`,
  );
});

test("case ref in expression is loud", () => {
  const message = expandFails({ gate: gate(), c: code("c", { v: "${gate.output + 1}" }) }).message;
  expect(message).toBe(`node 'c': \${gate.output} — ${REJECTED}`);
  expect(message).not.toContain("whole single reference");
  expect(message).not.toContain("call(...)");
});

test("case ref in assert is located", () => {
  // The assert-position rejection locates at the `asserts:` line, not the `output:` line.
  const diagnostics = new LoadDiagnostics("flow.yaml", (" ".repeat(80) + "\n").repeat(50));
  const assertionRange = new SourceRange(42, 1, 42, 1);
  const error = loadError(() =>
    expandCaseOutputs(new Map([["gate", gate()]]), null, [assertion('gate.output != ""')], {
      assertionRangeAt: () => assertionRange,
      diagnostics,
    }),
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(42);
});

const OTHER_CONDITION = `case node 'other' condition: \${gate.output} — ${REJECTED}`;

test("case ref in condition is loud", () => {
  // Another case whose when: reads gate.output (a case value) -> rejected.
  const other = new CaseDescriptor({ id: "other", cases: [{ when: "gate.output == 1", then: "z" }] });
  expect(expandFails({ gate: gate(), other }).message).toBe(OTHER_CONDITION);
});

test("case ref in assert is loud", () => {
  expect(expandFails({ gate: gate() }, [assertion('gate.output != ""')]).message).toBe(
    `assert: \${gate.output} — ${REJECTED}`,
  );
});

test("case ref in whole span condition is loud", () => {
  const other = new CaseDescriptor({ id: "other", cases: [{ when: "${gate.output == 1}", then: "z" }] });
  expect(expandFails({ gate: gate(), other }).message).toBe(OTHER_CONDITION);
});

test("on form match value reading a case value is loud", () => {
  const other = new CaseDescriptor({ id: "other", on: "${s.output}", cases: [{ when: "${gate.output}", then: "z" }] });
  expect(expandFails({ gate: gate(), other }).message).toBe(OTHER_CONDITION);
});

test("on form match value naming a case is not a read", () => {
  // An `on:` case's `when:` is a match value (text), not a condition.
  const other = new CaseDescriptor({ id: "other", on: "${s.output}", cases: [{ when: "gate.output", then: "z" }] });
  expect(expanded({ gate: gate(), other }).get("other")).toBe(other);
});

// ---------- load coverage with CODE branches ----------

const CASE_VALUE_FLOW = `
id: cv
name: case_value
input:
  seed: !float
  topic: !str
nodes:
  s:
    kind: code
    code: ./fixtures/compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "s.output >= 0.5"
        then: pro
    else: caut
  pro:
    kind: code
    code: ./fixtures/compose_codefns.ts:positive
    input:
      topic: \${input.topic}
    output: str
  caut:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      topic: \${input.topic}
    output: str
output: \${gate.output}
`;

/** Assemble `text`; return its compiled flow. */
const load = (text: string, children = {}) => assembleText(text, "flow.yaml", { children }).flow.compiled;

/** Return the source of the compiled flow's first output. */
const firstOutput = (compiled: CompiledFlow) => compiled.outputs[0]!.binding.source;

test("case value flow output is expanded", () => {
  // `output: ${gate.output}` desugared to the branch coalesce (= seed 02's hand form).
  expect(firstOutput(load(CASE_VALUE_FLOW))).toBe("${pro.output | caut.output}");
});

test("node consuming case value loads with data edges from both branches", () => {
  const text = CASE_VALUE_FLOW.replace(
    "output: ${gate.output}\n",
    "  summary:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
      "    input:\n      topic: ${gate.output}\n    output: str\noutput: ${summary.output}\n",
  );
  const edges = load(text).edges.map((edge) => `${edge.from}->${edge.to}`);
  expect(edges).toContain("pro->summary");
  expect(edges).toContain("caut->summary");
});

test("seed 22 case value output expanded", () => {
  // Seed 22's `output: ${gate.output}` desugars to seed 02's hand-written join.
  expect(firstOutput(load(caseText("load-case/seed_22_case_value").text))).toBe("${positive.output | cautious.output}");
});

test("case of case value expands through the nested leaf", () => {
  const compiled = load(`
id: coc
name: coc
input:
  a: !float
  b: !float
nodes:
  outer:
    kind: case
    cases:
      - when: "input.a >= 0.5"
        then: inner
    else: x
  inner:
    kind: case
    cases:
      - when: "input.b >= 0.5"
        then: hi
    else: lo
  hi:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: "HI"
    output: str
  lo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: "LO"
    output: str
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: "X"
    output: str
output: \${outer.output}
`);
  // ${outer.output} flattened through `inner` to its leaves (+ outer's else `x`).
  expect(firstOutput(compiled)).toBe("${hi.output | lo.output | x.output}");
});

test("case value in inline call arg expands", () => {
  // The desugar ordering contract: an inline call arg reading a case value
  // (call(passthru, v=${gate.output})) desugars to a synth call whose arg is then expanded to the
  // branch coalesce.
  const text = CASE_VALUE_FLOW.replace(
    "output: ${gate.output}\n",
    "  wrap:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
      "    input:\n      topic: call(passthru, v=${gate.output})\n    output: str\noutput: ${wrap.output}\n",
  );
  const passthru = handChild({
    inputs: parametersFromYaml("input:\n  v: !str\n", new Map()),
    producers: { x: new ValueContract(typeFor("str", new Map())) },
    output: "${x.output}",
  });
  const compiled = load(text, { passthru });
  const argument = compiled.wiring.get("__call_0")!.get("v");
  expect(argument).toBeInstanceOf(Binding);
  expect((argument as Binding).source).toBe("${pro.output | caut.output}");
});

test("case value in coalesce position rejected", () => {
  const text = CASE_VALUE_FLOW.replace("output: ${gate.output}\n", "output: ${gate.output | s.output}\n");
  expect(loadError(() => load(text)).message).toBe(`flow outputs: \${gate.output} — ${REJECTED}`);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/engine_case_value__case_value_in_a_coalesce_is_rejected", call: assembled },
  { id: "load-case/seed_22_case_value", call: assembled },
])("assemble: $id", runCase);
