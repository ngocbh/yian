/**
 * A binding's declared contract: the four carrier tests of
 * `tests/engine/test_binding_contract_identity.py` (the contract is the third, optional field;
 * only an exact `ValueContract` is accepted; `mapOrigins` rebases the contract and keeps its type;
 * `iterOrigins` yields the binding's origin before its contract's).
 *
 * Also ported: the identity tests `test_typed_wiring_binding_changes_flow_identity_only_when_typed`,
 * `test_binding_contract_origins_stay_outside_identity`, `test_typed_value_output_changes_fingerprint`
 * and `test_typed_record_output_field_changes_fingerprint`.
 *
 * Changed:
 * - `fields()` order is the own-key order of an instance.
 * - The identity tests are hand-built instead of `load_flow`: `twoStep()` and `recordOutput()`
 *   build the `_TWO_STEP` and `_RECORD_OUTPUT` flows with `CompiledFlow.fromParts`. Python's
 *   `_typed` deep-copies a loaded flow; the TS helpers type a fresh build, and "a deep copy keeps
 *   the fingerprint" is "a rebuild has the same fingerprint".
 *
 * Ported through `loadFlow` and `buildGrow`: `test_renamespacing_keeps_the_binding_contract`
 * (`rensInternal`), `test_flow_origin_traversal_reaches_the_binding_contract`,
 * `test_call_expansion_keeps_a_typed_child_binding` and `test_record_typed_binding_is_never_hashed`
 * (Python's hashability has no TS counterpart, "a deep copy keeps the fingerprint" is "a second
 * load, typed the same, has the same fingerprint", and its run is a `runFlow` in `hostEnv`). The
 * parent's `child` is served by a `childResolver` over the loaded `_CHILD`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { buildGrow, rensInternal } from "../../src/compile/expand.ts";
import { CompiledFlow, Edge, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { iterFlowOrigins, rewriteFlowOrigins } from "../../src/compile/origins.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { hostEnv } from "./mcp_helpers.ts";

/** Return an exact int contract with an optional root origin. */
function intContract(origin: OriginRef | null = null): ValueContract {
  return new ValueContract(typeFor("int", new Map()), new ContractOrigins({ default: origin }));
}

/** Return a record contract (Python's is unhashable, as its fields are a dict). */
function recordContract(): ValueContract {
  return new ValueContract(typeFor("Payload", readTypedefs({ Payload: { value: "int" } })));
}

test("binding fields end with an optional contract", () => {
  expect(Object.keys(new Binding("${x}"))).toEqual(["source", "origin", "contract"]);
  expect(new Binding("${x}").contract).toBeNull();
  const contract = intContract();
  expect(new Binding("${x}", { contract }).contract).toBe(contract);
});

test("a binding rejects a non-value contract", () => {
  expect(() => new Binding("${x}", { contract: typeFor("int", new Map()) as never })).toThrow(TypeError);
  expect(() => new Binding("${x}", { contract: typeFor("int", new Map()) as never })).toThrow(/ValueContract/);
  expect(() => new Binding("${x}", { contract: "int" as never })).toThrow(TypeError);
  expect(() => new Binding("${x}", { contract: "int" as never })).toThrow(/ValueContract/);
});

test("mapOrigins keeps and rebases the contract", () => {
  const contract = intContract(new OriginRef("type-site"));
  const binding = new Binding("${x}", { origin: new OriginRef("binding-site"), contract });

  const mapped = binding.mapOrigins((origin) => new OriginRef(`mapped:${origin.key}`));

  expect(mapped.source).toBe("${x}");
  expect(mapped.origin).toEqual(new OriginRef("mapped:binding-site"));
  expect(mapped.contract).not.toBeNull();
  expect(mapped.contract!.type).toBe(contract.type);
  expect(mapped.contract!.origins.default).toEqual(new OriginRef("mapped:type-site"));
  expect(new Binding("${x}").mapOrigins((origin) => origin).contract).toBeNull();
});

test("iterOrigins yields the binding's origin, then its contract's", () => {
  const contract = new ValueContract(
    typeFor("int", new Map()),
    new ContractOrigins({ default: new OriginRef("type-site") }),
  );
  const binding = new Binding("${x}", { origin: new OriginRef("binding-site"), contract });

  expect([...binding.iterOrigins()]).toEqual([new OriginRef("binding-site"), new OriginRef("type-site")]);
  expect([...new Binding("${x}", { contract }).iterOrigins()]).toEqual([new OriginRef("type-site")]);
});

/** The hand-built `_TWO_STEP` flow: `first` adds 1 to `${input.x}`, `second` doubles it. */
function twoStep(): CompiledFlow {
  const first = new CodeNode("first", { code: 'return inputs["x"] + 1' });
  first.params = [new Parameter("x")];
  first.outputContract = intContract();
  const second = new CodeNode("second", { code: 'return inputs["y"] * 2' });
  second.params = [new Parameter("y")];
  second.outputContract = intContract();
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [new Parameter("x", { contract: intContract() })] })],
      ["first", first],
      ["second", second],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [
      new Edge(`${START_ID}->first`, START_ID, "first", { inputGroup: "x" }),
      new Edge("first->second", "first", "second", { inputGroup: "y" }),
      new Edge(`second->${END_ID}`, "second", END_ID, { inputGroup: "result" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["first", new Map([["x", "${input.x}"]])],
        ["second", new Map([["y", "${first.output}"]])],
        [END_ID, new Map([["result", "${second.output}"]])],
      ]),
      outputs: [new FlowOutput("result", "${second.output}")],
      flowId: "two-step",
    },
  );
}

/** The hand-built `_RECORD_OUTPUT` flow: outputs `a` (`${first.output}`) and `b` (`${input.x}`). */
function recordOutput(): CompiledFlow {
  const first = new CodeNode("first", { code: 'return inputs["x"] + 1' });
  first.params = [new Parameter("x")];
  first.outputContract = intContract();
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [new Parameter("x", { contract: intContract() })] })],
      ["first", first],
      [END_ID, EndNode.record(END_ID, { outputNames: ["a", "b"] })],
    ]),
    [
      new Edge(`${START_ID}->first`, START_ID, "first", { inputGroup: "x" }),
      new Edge(`first->${END_ID}`, "first", END_ID, { inputGroup: "a" }),
      new Edge(`${START_ID}->${END_ID}`, START_ID, END_ID, { inputGroup: "b" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["first", new Map([["x", "${input.x}"]])],
        [END_ID, new Map([["a", "${first.output}"], ["b", "${input.x}"]])],
      ]),
      outputs: [new FlowOutput("a", "${first.output}"), new FlowOutput("b", "${input.x}")],
      flowId: "record-out",
    },
  );
}

/** Return `build()`'s flow with its one wiring binding `nodeId`.`param` carrying `contract`. */
function typed(build: () => CompiledFlow, nodeId: string, param: string, contract: ValueContract): CompiledFlow {
  const flow = build();
  const nodeWiring = flow.wiring.get(nodeId) as Map<string, Binding>;
  const binding = nodeWiring.get(param)!;
  nodeWiring.set(param, new Binding(binding.source, { origin: binding.origin, contract }));
  return flow;
}

/** Return `build()`'s flow with its declared output `index` carrying `contract`. */
function typedOutput(build: () => CompiledFlow, index: number, contract: ValueContract): CompiledFlow {
  const flow = build();
  const outputs = [...flow.outputs];
  const output = outputs[index]!;
  const { source, origin } = output.binding;
  outputs[index] = new FlowOutput(output.name, new Binding(source, { origin, contract }));
  flow.outputs = outputs;
  return flow;
}

test("a typed wiring binding changes flow identity only when typed", () => {
  // A wiring contract is canonical, and an untyped flow keeps its identity.
  const flow = twoStep();
  const originalFingerprint = flowFingerprint(flow);
  const originalTopology = expansionDigest(flow);
  expect(flowManifest(flow)).not.toHaveProperty("binding_types");

  const typedFlow = typed(twoStep, "second", "y", intContract(new OriginRef("type-site")));

  expect(flowFingerprint(typedFlow)).not.toBe(originalFingerprint);
  expect(expansionDigest(typedFlow)).not.toBe(originalTopology);
  const bindingTypes = flowManifest(typedFlow).binding_types as JsonObject;
  expect(Object.keys(bindingTypes)).toEqual(["second"]);
  expect(Object.keys(bindingTypes.second as JsonObject)).toEqual(["y"]);
  expect(((bindingTypes.second as JsonObject).y as JsonObject).kind).toBe("int");
  expect(flowFingerprint(flow)).toBe(originalFingerprint);
  expect(flowFingerprint(twoStep())).toBe(originalFingerprint);
});

test("binding contract origins stay outside identity", () => {
  // Two typed bindings that differ only in contract origins share identity.
  const left = typed(twoStep, "second", "y", intContract(new OriginRef("left")));
  const right = typed(twoStep, "second", "y", intContract(new OriginRef("right")));

  expect(flowFingerprint(left)).toBe(flowFingerprint(right));
  expect(expansionDigest(left)).toBe(expansionDigest(right));
});

test("a typed value output changes the fingerprint", () => {
  // A value flow's one declared output records its contract type.
  const flow = twoStep();
  expect(flow.outputs.map((output) => output.name)).toEqual(["result"]);
  expect((flowManifest(flow).outputs as JsonObject[])[0]).not.toHaveProperty("type");

  const typedFlow = typedOutput(twoStep, 0, intContract());

  expect(flowFingerprint(typedFlow)).not.toBe(flowFingerprint(flow));
  expect(((flowManifest(typedFlow).outputs as JsonObject[])[0]!.type as JsonObject).kind).toBe("int");
});

test("a typed record output field changes the fingerprint", () => {
  // A record flow's typed output field records its contract type.
  const flow = recordOutput();
  expect(flow.outputs.map((output) => output.name)).toEqual(["a", "b"]);

  const typedFlow = typedOutput(recordOutput, 1, intContract());

  expect(flowFingerprint(typedFlow)).not.toBe(flowFingerprint(flow));
  const outputs = flowManifest(typedFlow).outputs as JsonObject[];
  expect(outputs[0]).not.toHaveProperty("type");
  expect((outputs[1]!.type as JsonObject).kind).toBe("int");
});

const TWO_STEP = `
id: two-step
name: two_step
input:
  x: !int
nodes:
  first:
    kind: code
    input:
      x: \${input.x}
    output: int
    code: |
      return inputs["x"] + 1n
  second:
    kind: code
    input:
      y: \${first.output}
    output: int
    code: |
      return inputs["y"] * 2n
output: \${second.output}
`;

const CHILD = `
id: child
name: child
input:
  x: !int
nodes:
  inner:
    kind: code
    input:
      x: \${input.x}
    output: int
    code: |
      return inputs["x"] + 1n
output: \${inner.output}
`;

const PARENT = `
id: parent
name: parent
input:
  x: !int
uses:
  child: child
nodes:
  invoke:
    kind: call
    call: child
    input:
      x: \${input.x}
output: \${invoke.output}
`;

/** Load `PARENT` with its `child` served by the loaded `CHILD`. */
async function loadParent() {
  const child = await loadFlow(CHILD);
  return loadFlow(PARENT, { childResolver: () => child });
}

/** Set `contract` on `nodeWiring`'s binding `param`, keeping its source and origin. */
function setContract(nodeWiring: Map<string, Binding>, param: string, contract: ValueContract): void {
  const binding = nodeWiring.get(param)!;
  nodeWiring.set(param, new Binding(binding.source, { origin: binding.origin, contract }));
}

test("renamespacing keeps the binding contract", () => {
  // Re-namespacing a source or a literal keeps origin and contract.
  const contract = intContract(new OriginRef("type-site"));
  const expression = new Binding("${inner.output}", { origin: new OriginRef("binding-site"), contract });
  const literal = new Binding(3n, { origin: new OriginRef("binding-site"), contract });

  const renamed = rensInternal(expression, "invoke");
  expect(renamed).toEqual(new Binding("${invoke/inner.output}", { origin: new OriginRef("binding-site"), contract }));
  expect(renamed.contract).toBe(contract);
  expect(rensInternal(literal, "invoke").contract).toBe(contract);
});

test("flow origin traversal reaches the binding contract", async () => {
  // Flow-wide origin iteration and rewriting cover a typed binding's contract.
  const flow = (await loadFlow(TWO_STEP)).compiled;
  const typedFlow = typed(() => flow, "second", "y", intContract(new OriginRef("type-site")));

  expect([...iterFlowOrigins(typedFlow)]).toContainEqual(new OriginRef("type-site"));

  const rewritten = rewriteFlowOrigins(typedFlow, (origin) => new OriginRef(`new:${origin.key}`));

  const contract = rewritten.wiring.get("second")!.get("y")!.contract;
  expect(contract).not.toBeNull();
  expect(contract!.origins.default).toEqual(new OriginRef("new:type-site"));
  expect(typedFlow.wiring.get("second")!.get("y")!.contract!.origins.default).toEqual(new OriginRef("type-site"));
});

test("CALL expansion keeps a typed child binding", async () => {
  // A CALL child's typed wiring binding keeps its contract in the grown fragment.
  const parent = (await loadParent()).compiled;
  const call = parent.nodes.get("invoke") as CallNode;
  const contract = intContract();
  setContract((call.child as CompiledFlow).wiring.get("inner") as Map<string, Binding>, "x", contract);

  const fragment = buildGrow(call, { x: 2n });

  expect(fragment.wiring.get("invoke/inner")!.get("x")!.contract).toBe(contract);
});

test("a record typed binding is never hashed", async () => {
  /** Load the parent and type its CALL's child binding and its own `x` binding with `contract`. */
  async function typedParent(contract: ValueContract) {
    const loaded = await loadParent();
    const call = loaded.compiled.nodes.get("invoke") as CallNode;
    setContract((call.child as CompiledFlow).wiring.get("inner") as Map<string, Binding>, "x", contract);
    setContract(loaded.compiled.wiring.get("invoke") as Map<string, Binding>, "x", contract);
    return { loaded, compiled: loaded.compiled, call };
  }
  const contract = recordContract();
  const { loaded, compiled, call } = await typedParent(contract);
  const again = (await typedParent(recordContract())).compiled;

  expect([...again.wiring.get("invoke")!.get("x")!.contract!.type.fields!.keys()]).toEqual(["value"]);
  const manifest = flowManifest(compiled);
  expect((((manifest.binding_types as JsonObject).invoke as JsonObject).x as JsonObject).kind).toBe("object");
  expect(flowFingerprint(again)).toBe(flowFingerprint(compiled));

  const fragment = buildGrow(call, { x: 2n });
  expect(fragment.wiring.get("invoke/inner")!.get("x")!.contract).toBe(contract);
  expansionDigest(fragment);

  // The binder checks the record contract and rejects the int.
  const dir = mkdtempSync(join(tmpdir(), "yian-binding-contract-"));
  try {
    const result = await runFlow(loaded, { x: 2 }, { env: hostEnv(dir) });
    expect(result.status).toBe("failed");
    expect(result.failure).not.toBeNull();
    expect(result.failure!.code).toBe("binding.type");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
