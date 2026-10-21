/**
 * Loading typed node input bindings, `name: !Type value` carries a contract: a port of
 * `tests/compose/test_typed_bindings_load.py`.
 *
 * Ported: every row. Types compare with `typeEquals`, as Python's `==` compares them; a typed
 * literal `!int 0` holds the int `0n` (decision 4a). The CODE references are TS's
 * (`./fixtures/compose_codefns.ts:echo`), and the inline CODE bodies are Python's, which read the
 * same in JS (decision 11). `test_typed_binding_changes_the_flow_fingerprint` is ported too: the
 * case `fingerprint/compose_typed_bindings_load__typed_binding` harvested from it fingerprints
 * TOOL nodes, not this flow's CODE nodes.
 */

import { expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { Failure } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, bindingSource, type ValueContract } from "../../src/nodes/contracts.ts";
import type { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "./declarations.ts";

const INT = typeFor("int", new Map());

/** One node's exact wiring binding by parameter name. */
function bindingOf(compiled: CompiledFlow, nodeId: string, name: string): Binding {
  const binding = compiled.wiring.get(nodeId)!.get(name);
  expect(binding?.constructor).toBe(Binding);
  return binding!;
}

/** The compiled child flow baked onto CALL, MAP or LOOP node `nodeId`. */
const childOf = (compiled: CompiledFlow, nodeId: string) =>
  (compiled.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** Expect `contract` to be present with a type equal to `expected`. */
function expectContractType(contract: ValueContract | null, expected: Type): void {
  expect(contract).not.toBeNull();
  expect(typeEquals(contract!.type, expected)).toBe(true);
}

/** A flow whose `use` node binds `qty` as `binding`. */
const typedFlow = (binding: string) => `
id: typed
name: typed
input:
  n: !int
nodes:
  size:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.n}
    output:
      qty: int
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      qty: ${binding}
    output: int
output: \${use.output}
`;

test("typed ref binding carries its contract", async () => {
  const loaded = await loadConformanceFlow(typedFlow("!int ${size.output.qty}"));

  const binding = bindingOf(loaded.compiled, "use", "qty");
  expect(binding.source).toBe("${size.output.qty}");
  expectContractType(binding.contract, INT);
});

test("typed ref binding infers the same edges as the plain binding", async () => {
  const typed = await loadConformanceFlow(typedFlow("!int ${size.output.qty}"));
  const plain = await loadConformanceFlow(typedFlow("${size.output.qty}"));

  const edgesFromSize = (loaded: ComposedFlow) => loaded.compiled.edges.filter((edge) => edge.from === "size");

  const typedEdges = edgesFromSize(typed);
  expect(typedEdges.length).toBeGreaterThan(0);
  expect(typedEdges.some((edge) => edge.optional)).toBe(false);
  expect(typedEdges).toEqual(edgesFromSize(plain));
});

test("plain binding has no contract", async () => {
  const loaded = await loadConformanceFlow(typedFlow("${size.output.qty}"));

  expect(bindingOf(loaded.compiled, "use", "qty").contract).toBeNull();
});

test("typed binding changes the flow fingerprint", async () => {
  const typed = await loadConformanceFlow(typedFlow("!int ${size.output.qty}"));
  const plain = await loadConformanceFlow(typedFlow("${size.output.qty}"));

  expect(flowFingerprint(typed.compiled)).not.toBe(flowFingerprint(plain.compiled));
});

test("typed literal binding reads the right side as yaml", async () => {
  const loaded = await loadConformanceFlow(typedFlow("!int 0"));

  const binding = bindingOf(loaded.compiled, "use", "qty");
  expect(binding.source).toBe(0n);
  expectContractType(binding.contract, INT);
});

test("contract origin points at the typed binding line", async () => {
  const text = typedFlow("!int ${size.output.qty}");
  const loaded = await loadConformanceFlow(text);
  const contract = bindingOf(loaded.compiled, "use", "qty").contract!;
  const line = text.split("\n").findIndex((row) => row.includes("qty: !int ")) + 1;

  const diagnostic = loaded.diagnose(
    new Failure({ code: "binding.type", message: "x", nodeId: "use", trace: [contract.origins.default!] }),
  );

  expect(diagnostic.frames.length).toBe(1);
  expect(diagnostic.frames[0]!.range.startLine).toBe(line);
});

test("typed inline call binding references the synthetic call", async () => {
  const loaded = await loadConformanceFlow(`
id: typed_call
name: typed_call
defs:
  f:
    input:
      a: !int
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.a}
        output: int
    output: \${x.output}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      x: !int call(f, a=1)
    output: int
output: \${use.output}
`);

  const binding = bindingOf(loaded.compiled, "use", "x");
  expect(loaded.compiled.nodes.has("__call_0")).toBe(true);
  expect(binding.source).toBe("${__call_0.output}");
  expectContractType(binding.contract, INT);
});

test("typed case output binding expands and keeps its contract", async () => {
  const loaded = await loadConformanceFlow(`
id: typed_case
name: typed_case
typedefs:
  Decision: Literal(go, stop)
input:
  flag: !bool
nodes:
  route:
    kind: case
    cases:
      - when: \${input.flag}
        then: go
    else: stop
  go:
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_pro
    input:
      flag: \${input.flag}
    output: Decision
  stop:
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_pro
    input:
      flag: \${input.flag}
    output: Decision
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      x: !Decision \${route.output}
    output: str
output: \${use.output}
`);

  const binding = bindingOf(loaded.compiled, "use", "x");
  expect(binding.source).toBe("${go.output | stop.output}");
  const decision = (loaded.compiled.nodes.get("go")!.outputContract as ValueContract).type;
  expectContractType(binding.contract, decision);
  expect(decision.tags?.size).toBeGreaterThan(0);
});

test.each([
  ["double", '"a = b"'],
  ["single", "'a = b'"],
  ["literal", "|\n        a = b"],
])("quoted and block values stay plain text (%s)", async (_id, note) => {
  const loaded = await loadConformanceFlow(`
id: plain
name: plain
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      note: ${note}
    output: str
output: \${use.output}
`);

  const binding = bindingOf(loaded.compiled, "use", "note");
  expect((binding.source as string).replace(/\n+$/, "")).toBe("a = b");
  expect(binding.contract).toBeNull();
});

/** A flow whose `use` node returns its `v` input, bound as `binding`. */
const valueFlow = (binding: string) => `
id: value
name: value
input:
  a: !int
nodes:
  use:
    kind: code
    input:
      v: ${binding}
    code: |
      return inputs["v"]
output: \${use.output}
`;

test.each<[string, string, unknown]>([
  ["quoted-template", '!str "${input.a} = 2"', "1 = 2"],
  ["single-quoted-template", "!str 'n=${input.a}'", "n=1"],
  ["literal-list", '!list(str) ["$${input.a}"]', ["${input.a}"]],
])("quoted or bracketed right side is read as yaml first (%s)", async (_id, binding, expected) => {
  const result = await runFlow(await loadConformanceFlow(valueFlow(binding)), { a: 1 });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(expected);
});

test("plain value whose left side is not a type is text", async () => {
  const text = `
id: plain
name: plain
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      note: a = b
    output: str
output: \${use.output}
`;
  const binding = (await loadConformanceFlow(text)).compiled.wiring.get("use")!.get("note");

  expect(bindingSource(binding)).toBe("a = b");
  expect(binding instanceof Binding ? binding.contract : null).toBeNull();
});

test("typed inputs inside a multi node def carry contracts", async () => {
  const loaded = await loadConformanceFlow(`
id: typed_def
name: typed_def
defs:
  f:
    input:
      a: !int
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: !int \${input.a}
        output: int
    output: \${x.output}
nodes:
  use:
    kind: call
    call: f
    input:
      a: 1
output: \${use.output}
`);

  const binding = bindingOf(childOf(loaded.compiled, "use"), "x", "topic");
  expect(binding.source).toBe("${input.a}");
  expectContractType(binding.contract, INT);
});

test("typed inputs inside an inline call body carry contracts", async () => {
  const loaded = await loadConformanceFlow(`
id: typed_inline_call
name: typed_inline_call
input:
  n: !int
nodes:
  wrapper:
    kind: call
    input:
      n: \${input.n}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          value: !int \${input.n}
        output: int
    output: \${echo.output}
output: \${wrapper.output}
`);

  const binding = bindingOf(childOf(loaded.compiled, "wrapper"), "echo", "value");
  expect(binding.source).toBe("${input.n}");
  expectContractType(binding.contract, INT);
});

test("typed inputs inside an inline map body carry contracts", async () => {
  const loaded = await loadConformanceFlow(`
id: typed_inline_map
name: typed_inline_map
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: \${item}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["value"]
        input:
          value: !int \${input.x}
        output: int
    output: \${echo.output}
output: \${each.output}
`);

  const binding = bindingOf(childOf(loaded.compiled, "each"), "echo", "value");
  expect(binding.source).toBe("${input.x}");
  expectContractType(binding.contract, INT);
});

test("single node def parameter defaults are not split", async () => {
  const loaded = await loadConformanceFlow(`
id: single
name: single
defs:
  f:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      p: !int 3
    output: int
nodes:
  use:
    kind: call
    call: f
output: \${use.output}
`);

  const child = childOf(loaded.compiled, "use");
  const start = child.nodes.get(child.startId) as StartNode;
  const parameters = new Map(start.params!.map((parameter) => [parameter.name, parameter]));
  expect(parameters.get("p")!.hasDefault).toBe(true);
  expect(parameters.get("p")!.default).toBe(3n);
  const nodeId = [...child.wiring].find(([, wiring]) => wiring.has("p"))![0];
  const binding = bindingOf(child, nodeId, "p");
  expect(binding.source).toBe("${input.p}");
  expect(binding.contract).toBeNull();
});

const QUOTED_INLINE_CALL = `
id: quoted_inline_call
name: quoted_inline_call
nodes:
  host:
    kind: call
    input:
      n: 1
    nodes:
      use:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          note: "a = b"
        output: str
    output: \${use.output}
output: \${host.output}
`;

const QUOTED_INLINE_LOOP = `
id: quoted_inline_loop
name: quoted_inline_loop
nodes:
  host:
    kind: loop
    times: 1
    input:
      n: 1
    nodes:
      use:
        kind: code
        code: |
          return inputs["n"]
        input:
          n: \${input.n}
          note: "a = b"
        output: int
    output:
      n: \${use.output}
output: \${host.output}
`;

const QUOTED_DEF = `
id: quoted_def
name: quoted_def
defs:
  f:
    nodes:
      use:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          note: "a = b"
        output: str
    output: \${use.output}
nodes:
  host:
    kind: call
    call: f
output: \${host.output}
`;

test.each([
  ["inline-call", QUOTED_INLINE_CALL],
  ["inline-loop", QUOTED_INLINE_LOOP],
  ["multi-node-def", QUOTED_DEF],
])("quoted input stays plain wherever it is written (%s)", async (_id, text) => {
  const loaded = await loadConformanceFlow(text);

  const binding = bindingOf(childOf(loaded.compiled, "host"), "use", "note");
  expect(binding.source).toBe("a = b");
  expect(binding.contract).toBeNull();
});
