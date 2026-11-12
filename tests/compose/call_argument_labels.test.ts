/**
 * Load errors about an inline `call(...)` argument name the authored argument.
 *
 * An inline `call(child, y=...)` desugars to a synth `__call_<n>` node. An error about one of its
 * arguments reads `<site> call(<callee>) argument '<name>'`, where the site is the authored binding
 * the call was written in, and never names the synth node.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  assembledWith,
  assembleText,
  type CaseRow,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../helpers/declarations.ts";
import { outputTag } from "../helpers/output_tags.ts";

/** A def over the one input `name: !<type>` whose node `x` returns `output`, built by hand. */
const def = (name: string, type: string, output: string): ComposedFlow =>
  handChild({
    inputs: parametersFromYaml(`input:\n  ${name}: !${type}\n`, new Map()),
    producers: { x: new ValueContract(typeFor(output, new Map())) },
    output: "${x.output}",
  });

/** The `child` def, which echoes its one input `y`, as `{child}`. */
const child = (yType = "str"): ChildFlows => ({ child: def("y", yType, yType) });

/** Return a `defs:` block holding `child`, which echoes its one input `y`. */
function childDef(yType = "str"): string {
  return `
defs:
  child:
    input:
      y: !${yType}
    nodes:
      x:
        kind: code
        code: "return inputs['y']"
        input:
          y: \${input.y}
        output: ${outputTag(yType)}
    output: \${x.output}
`;
}

/** Return a flow whose CODE node `n` binds input `a` to `binding`. */
function nodeInputFlow(binding: string, yType = "str"): string {
  return `
id: f
name: f
input:
  s: !str
${childDef(yType)}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: ${binding}
    output: ${outputTag(yType)}
output: \${n.output}
`;
}

/** Assert loading `text` with `children` as its defs fails with `phrase` and never names a synth node. */
function expectNames(text: string, phrase: string, children: ChildFlows = child()): void {
  const message = loadError(() => assembleText(text, "flow.yaml", { children })).message;
  expect(message).toContain(phrase);
  expect(message).not.toContain("__call_");
}

const BAD_REF = "${nosuch.output}";

test("a bad ref in a node input call names the argument", () => {
  expectNames(
    nodeInputFlow(`call(child, y=${BAD_REF})`),
    `node 'n' input 'a' call(child) argument 'y' from: reference ${BAD_REF}`,
  );
});

test("a bad ref in a value entry call names the argument", () => {
  const text = `
id: f
name: f
${childDef()}
nodes:
  v: !str call(child, y=${BAD_REF})
output: \${v.output}
`;
  expectNames(text, "value entry 'v' call(child) argument 'y' from:");
});

/** Return a flow whose CASE `g` routes to `then` or `elseTarget`. */
function caseFlow(then: string, elseTarget: string): string {
  return `
id: f
name: f
input:
  s: !str
${childDef()}
nodes:
  g:
    kind: case
    cases:
      - when: "input.s == 'a'"
        then: ${then}
    else: ${elseTarget}
  a: {kind: code, code: "return 'a'", output: !str}
output: \${g.output}
`;
}

test("a bad ref in a case else call names the argument", () => {
  expectNames(caseFlow("a", `call(child, y=${BAD_REF})`), "node 'g' else call(child) argument 'y' from:");
});

test("a bad ref in a case then call names the argument", () => {
  expectNames(caseFlow(`call(child, y=${BAD_REF})`, "a"), "node 'g' cases[0].then call(child) argument 'y' from:");
});

/** Return a flow with one CODE node and the flow `output:` section `output`. */
function outputFlow(output: string): string {
  return `
id: f
name: f
${childDef()}
nodes:
  n: {kind: code, code: "return 'a'", output: !str}
output: ${output}
`;
}

test("a bad ref in a single flow output call names the argument", () => {
  expectNames(outputFlow(`call(child, y=${BAD_REF})`), "flow output call(child) argument 'y' from:");
});

test("a bad ref in a record flow output call names the argument", () => {
  expectNames(outputFlow(`\n  r: call(child, y=${BAD_REF})`), "flow output 'r' call(child) argument 'y' from:");
});

/** The `lister` and `one` defs of the MAP flow. */
const MAP_DEFS: ChildFlows = { lister: def("n", "str", "list(str)"), one: def("topic", "str", "str") };

test("a bad ref in a map over call names the argument", () => {
  const text = `
id: f
name: f
defs:
  lister:
    input:
      n: !str
    nodes:
      x:
        kind: code
        code: "return [inputs['n']]"
        input:
          n: \${input.n}
        output: !list(str)
    output: \${x.output}
  one:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: "return inputs['topic']"
        input:
          topic: \${input.topic}
        output: !str
    output: \${x.output}
nodes:
  m:
    kind: map
    call: one
    over: call(lister, n=${BAD_REF})
    input:
      topic: \${item}
output: \${m.output}
`;
  expectNames(text, "node 'm' map over call(lister) argument 'n' from:", MAP_DEFS);
});

/** The `check` def of the flow-assert flow. */
const CHECK: ChildFlows = { check: def("y", "str", "bool") };

test("a bad ref in a flow assert call names the argument", () => {
  const text = `
id: f
name: f
defs:
  check:
    input:
      y: !str
    nodes:
      x:
        kind: code
        code: "return True"
        input:
          y: \${input.y}
        output: !bool
    output: \${x.output}
nodes:
  n: {kind: code, code: "return 'a'", output: !str}
output: \${n.output}
asserts:
  - call(check, y=${BAD_REF})
`;
  expectNames(text, `flow assert 'call(check, y=${BAD_REF})' call(check) argument 'y' from:`, CHECK);
});

/** The `child` and `inner` defs of the nested-call flow. */
const NESTED: ChildFlows = { ...child(), inner: def("z", "str", "str") };

test("a nested call shares its enclosing binding site", () => {
  const text = nodeInputFlow(`call(child, y=call(inner, z=${BAD_REF}))`).replace(
    "defs:\n",
    `defs:
  inner:
    input:
      z: !str
    nodes:
      x:
        kind: code
        code: "return inputs['z']"
        input:
          z: \${input.z}
        output: !str
    output: \${x.output}
`,
  );
  expectNames(text, "node 'n' input 'a' call(inner) argument 'z' from:", NESTED);
});

test("a mis-called builtin in an argument names the argument", () => {
  expectNames(
    nodeInputFlow("call(child, y=${upper(input.s, 1)})"),
    "node 'n' input 'a' call(child) argument 'y' from: expression function 'upper'",
  );
});

test("an argument source type mismatch names the argument", () => {
  expectNames(
    nodeInputFlow("call(child, y=${input.s})", "int"),
    "node 'n' input 'a' call(child) argument 'y' — child expects 'int', source is 'str'",
    child("int"),
  );
});

test("an argument literal type mismatch names the argument", () => {
  expectNames(
    nodeInputFlow("call(child, y=abc)", "int"),
    "node 'n' input 'a' call(child) argument 'y' — child expects 'int', but the value \"abc\" does not fit",
    child("int"),
  );
});

test("an undeclared argument names the argument", () => {
  expectNames(
    nodeInputFlow("call(child, z=1, y=1)", "int"),
    "node 'n' input 'a' call(child) argument 'z' is not a declared input of callable 'child'",
    child("int"),
  );
});

test("a missing argument names the call", () => {
  expectNames(
    nodeInputFlow("call(child)", "int"),
    "node 'n' input 'a' call(child): required callable input 'y' is unbound",
    child("int"),
  );
});

/** Return a flow whose authored `kind: call` node `c` binds `inputs`. */
const callNodeFlow = (inputs: string) => `
id: f
name: f
input:
  s: !str
${childDef("int")}
nodes:
  c: {kind: call, call: child, input: ${inputs}}
output: \${c.output}
`;

test("an authored call node keeps its binding wording", () => {
  const load = (text: string) => loadError(() => assembleText(text, "flow.yaml", { children: child("int") })).message;

  expect(load(callNodeFlow("{z: 1, y: 1}"))).toContain(
    "call node 'c': binding 'z' is not a declared input of callable 'child'",
  );
  expect(load(callNodeFlow("{y: '${input.s}'}"))).toContain(
    "call node 'c': binding 'y' — child expects 'int', source is 'str'",
  );
});

test("an undeclared argument in a value entry call names the argument", () => {
  const text = `
id: f
name: f
${childDef("int")}
nodes:
  v: !int call(child, z=1, y=1)
output: \${v.output}
`;
  expectNames(
    text,
    "value entry 'v' call(child) argument 'z' is not a declared input of callable 'child'",
    child("int"),
  );
});

test("a cycle through an inline call names its host in a note", () => {
  const text = `
id: f
name: f
${childDef()}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=\${n.output})
    output: !str
output: \${n.output}
`;
  const { diagnostic } = loadError(() => assembleText(text, "flow.yaml", { children: child() }));
  expect(diagnostic.message).toBe('flow has a cycle involving ["__call_0","n"]; flows must be acyclic');
  expect(diagnostic.notes).toContain("__call_0 is generated from node 'n'");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  (
    [
      ["a_bad_ref_in_a_flow_assert_call_names_the_argument", CHECK],
      ["a_bad_ref_in_a_flow_output_call_names_the_argument", child()],
      ["a_bad_ref_in_a_map_over_call_names_the_argument", MAP_DEFS],
      ["a_bad_ref_in_a_value_entry_call_names_the_argument", child()],
      ["a_nested_call_shares_its_enclosing_binding_site", NESTED],
      ["a_cycle_through_an_inline_call_names_its_host", child()],
      ["an_argument_literal_type_mismatch", child("int")],
    ] as const
  ).map(([name, children]) => ({
    id: `load-map-call/compose_call_argument_labels__${name}`,
    call: assembledWith(children),
    full: true,
  })),
)("assemble: $id", runCase);
