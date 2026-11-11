/** Loading typed flow outputs, `output: !Type value` and typed record fields. */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow, FlowOutput } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import type { EndNode } from "../../src/nodes/end/node.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-flow-outputs-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const REGISTRY = readTypedefs({ Size: { qty: "int" }, Decision: "Literal(go, stop)" });
const SIZE = typeFor("Size", REGISTRY);
const INT = typeFor("int", new Map());
const STR = typeFor("str", new Map());

/** The compiled flow's outputs keyed by name. */
const outputsOf = (compiled: CompiledFlow): Map<string, FlowOutput> =>
  new Map(compiled.outputs.map((output) => [output.name, output]));

/** The compiled child flow baked onto CALL node `nodeId`. */
const childOf = (compiled: CompiledFlow, nodeId: string) =>
  (compiled.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** The output contract of `compiled`'s node `nodeId`. */
const contractOf = (compiled: CompiledFlow, nodeId: string) => compiled.nodes.get(nodeId)!.outputContract;

/** Expect `actual` to be a type equal to `expected`. */
function expectType(actual: Type | undefined, expected: Type): void {
  expect(actual).toBeDefined();
  expect(typeEquals(actual!, expected)).toBe(true);
}

/** Expect `contract` to be exactly a `ValueContract` whose record type has exactly `fields`. */
function expectRecordFields(contract: unknown, fields: Record<string, Type>): void {
  expect((contract as object | null)?.constructor).toBe(ValueContract);
  const actual = (contract as ValueContract).type.fields!;
  expect([...actual.keys()].sort()).toEqual(Object.keys(fields).sort());
  for (const [name, type] of Object.entries(fields)) expectType(actual.get(name), type);
}

/** A flow whose untyped `size` node feeds `output:` written as `output`. */
const sizeFlow = (output: string) => `
id: sized
name: sized
typedefs:
  Size:
    qty: int
nodes:
  size:
    kind: code
    code: |
      return {"qty": 1}
output: ${output}
`;

test("typed single output carries its contract", async () => {
  const loaded = await loadFlow(sizeFlow("!Size ${size.output}"));

  const output = outputsOf(loaded.compiled).get("result")!;
  expect(output.binding.source).toBe("${size.output}");
  expectType(output.binding.contract?.type, SIZE);
  const end = loaded.compiled.nodes.get("__end__") as EndNode;
  expect(end.returnsRecord).toBe(false);
  expect(end.outputContract?.constructor).toBe(ValueContract);
  expectType((end.outputContract as ValueContract).type, SIZE);
});

test("untyped single output leaves end untyped", async () => {
  const loaded = await loadFlow(sizeFlow("${size.output}"));

  expect(outputsOf(loaded.compiled).get("result")!.binding.contract).toBeNull();
  expect(contractOf(loaded.compiled, "__end__")).toBeNull();
});

test("typed output changes the flow fingerprint", async () => {
  const typed = await loadFlow(sizeFlow("!Size ${size.output}"));
  const plain = await loadFlow(sizeFlow("${size.output}"));

  expect(flowFingerprint(typed.compiled)).not.toBe(flowFingerprint(plain.compiled));
});

test("record output types only its typed fields", async () => {
  const loaded = await loadFlow(`
id: decided
name: decided
typedefs:
  Decision: Literal(go, stop)
nodes:
  d:
    kind: code
    code: |
      return "go"
  n:
    kind: code
    code: |
      return "fine"
output:
  decision: !Decision \${d.output}
  note: \${n.output}
  label: "a = b"
`);

  const outputs = outputsOf(loaded.compiled);
  const decision = typeFor("Decision", REGISTRY);
  expect(outputs.get("decision")!.binding.source).toBe("${d.output}");
  expectType(outputs.get("decision")!.binding.contract?.type, decision);
  expect(outputs.get("note")!.binding.contract).toBeNull();
  expect(outputs.get("label")!.binding.source).toBe("a = b");
  expect(outputs.get("label")!.binding.contract).toBeNull();
  const end = loaded.compiled.nodes.get("__end__") as EndNode;
  expect(end.returnsRecord).toBe(true);
  const contract = end.outputContract;
  expect(contract?.constructor).toBe(PartialRecordContract);
  const fields = new Map((contract as PartialRecordContract).fields);
  expectType(fields.get("decision")?.type, decision);
  expect(fields.get("note")).toBeNull();
  expect(fields.get("label")).toBeNull();
});

test("fully typed record output gives end an exact record", async () => {
  const loaded = await loadFlow(`
id: exact
name: exact
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  a: !int \${n.output}
  b: !str \${n.output}
`);

  expectRecordFields(contractOf(loaded.compiled, "__end__"), { a: INT, b: STR });
});

test("untyped record output leaves end untyped", async () => {
  const loaded = await loadFlow(`
id: plain
name: plain
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  a: \${n.output}
`);

  expect(contractOf(loaded.compiled, "__end__")).toBeNull();
});

test("record output field whose left side is not a type is text", async () => {
  const text = `
id: plain
name: plain
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  note: a = b
`;
  const result = await runFlow(await loadFlow(text), {}, { env });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ note: "a = b" });
});

test("compact flow output stays the node output type", async () => {
  const loaded = await loadFlow(`
id: compact
name: compact
typedefs:
  Size:
    qty: int
kind: code
code: |
  return {"qty": 1}
output: !Size
`);

  expectType((contractOf(loaded.compiled, "compact") as ValueContract).type, SIZE);
  const output = outputsOf(loaded.compiled).get("result")!;
  expect(output.binding.source).toBe("${compact.output}");
  expect(output.binding.contract).toBeNull();
  expect(contractOf(loaded.compiled, "__end__")).toBeNull();
});

/** A flow calling and mapping def `sized`, whose `output:` is `childOutput`. */
const childFlow = (childOutput: string) => `
id: parent
name: parent
typedefs:
  Size:
    qty: int
defs:
  sized:
    input:
      n: !int
    nodes:
      size:
        kind: code
        input:
          n: \${input.n}
        code: |
          return {"qty": inputs["n"]}
    output: ${childOutput}
nodes:
  use:
    kind: call
    call: sized
    input:
      n: 1
  each:
    kind: map
    over: [1, 2]
    call: sized
    input:
      n: \${item}
output:
  one: \${use.output}
  many: \${each.output}
`;

test("call parent sees the child declared output type", async () => {
  const loaded = await loadFlow(childFlow("!Size ${size.output}"));

  expectType((contractOf(loaded.compiled, "use") as ValueContract).type, SIZE);
});

test("map parent sees a list of the child declared output type", async () => {
  const loaded = await loadFlow(childFlow("!Size ${size.output}"));

  expectType((contractOf(loaded.compiled, "each") as ValueContract).type, typeFor("list(Size)", REGISTRY));
});

test("call parent sees a partial record for a partly typed child", async () => {
  const childOutput = "\n      size: !Size ${size.output}\n      raw: ${size.output}";
  const loaded = await loadFlow(childFlow(childOutput));

  const contract = contractOf(loaded.compiled, "use");
  expect(contract?.constructor).toBe(PartialRecordContract);
  const fields = new Map((contract as PartialRecordContract).fields);
  expectType(fields.get("size")?.type, SIZE);
  expect(fields.get("raw")).toBeNull();
});

test("inline call body output field carries its declared type", async () => {
  const loaded = await loadFlow(`
id: inline_call
name: inline_call
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
        input:
          n: \${input.n}
        code: |
          return inputs["n"]
    output:
      qty: !int \${echo.output}
output: \${wrapper.output}
`);

  const child = childOf(loaded.compiled, "wrapper");
  expectType(outputsOf(child).get("qty")!.binding.contract?.type, INT);
  expectRecordFields(contractOf(loaded.compiled, "wrapper"), { qty: INT });
});

/** A flow whose inline LOOP body returns `rounds` written as `field`. */
const loopFlow = (field: string) => `
id: inline_loop
name: inline_loop
nodes:
  loop:
    kind: loop
    times: 2
    input:
      rounds: 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.rounds}
        code: |
          return inputs["n"] + 1
    output:
      rounds: ${field}
output: \${loop.output}
`;

test("inline loop body output field types the carried record", async () => {
  const typed = await loadFlow(loopFlow("!int ${bump.output}"));
  const plain = await loadFlow(loopFlow("${bump.output}"));

  const contract = contractOf(typed.compiled, "loop");
  expectRecordFields(contract, { rounds: INT });
  // Compared by value, the untyped body's contract is not that exact record.
  const other = contractOf(plain.compiled, "loop");
  expect(
    other instanceof ValueContract && typeEquals(other.type, (contract as ValueContract).type),
  ).toBe(false);
});
