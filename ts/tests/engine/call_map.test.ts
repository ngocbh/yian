/**
 * CALL and MAP build over a child's signature: a port of `tests/engine/test_call_map.py`.
 *
 * Ported: the signature half of `test_seed03_loads_as_child_flow`; the build halves of
 * `test_seed04_call_loads_and_reexports_child_output` and
 * `test_seed05_map_loads_with_list_output_and_parallel`; `test_map_input_named_over_is_reserved`,
 * `test_call_binding_to_undeclared_child_input_is_loud`, `test_map_child_with_no_output_is_loud`
 * and the contract half of `test_call_accepts_proven_null_for_nullable_child_input`.
 *
 * Changed: every child is built by hand (`handChild`) where Python loads seed 03 or a def. Seed
 * 03's child takes the seed's two parameters (read from its `input:` text) and returns a record
 * over its `synth` producer and `input.as_of`, with `fetch` and `synth` typed as the seed types
 * them. The seed 04 and 05 build halves parse the seed's node (`nodesOf`) and build it over a
 * `BuildContext` whose resolver returns that child; their origin halves assemble the seed
 * (`assembleText` with `children`), where Python loads it with a search path. The seeds come from
 * the conformance cases `load-map-call/seed_04_call` (with its file `03-research-one.yaml`) and
 * `load-map-call/seed_05_call_map`. The reserved-`over`, undeclared-binding (both joined errors)
 * and no-output rows pin the whole message, where Python matches fragments.
 *
 * TS-only: a MAP over a child whose output is partial leaves its contract unset, and a
 * descriptor's `fieldLabels` name an inline call's argument and the call in the binding errors.
 *
 * Pinned at the end (`assembled`):
 * `load-map-call/engine_call_map__map_input_named_over_is_reserved`, whose check runs before the
 * child is resolved.
 *
 * Later (T3-22): the seed and def loads of the rows above, `test_call_map_without_resolver_is_loud`
 * (the loader's `uses:` error) and `test_flow_without_calls_loads_without_resolver`. Later (T3-18,
 * T3-22): `test_seed04_downstream_dotted_ref_validates_against_child_record`. Later (T3-19): the
 * acceptance half of `test_call_accepts_proven_null_for_nullable_child_input` (the binding type
 * check).
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { childSignature } from "../../src/compose/nodes/resolver.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, isExactly, Parameter, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  caseText,
  type ChildFlows,
  handChild,
  loadError,
  nodesOf,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/** Seed 03 (`research-one`), built by hand: its parameters, producers and record output. */
function researchOne(): ComposedFlow {
  const { text } = caseText("load-map-call/seed_04_call", { file: "03-research-one.yaml" });
  const facts = typeFor("Facts", readTypedefs({ Facts: { values: "list(float)", news: "list(str)" } }));
  return handChild({
    inputs: parametersFromYaml(text, new Map()),
    producers: { fetch: new ValueContract(facts), synth: STR },
    output: { report: "${synth.output}", asof: "${input.as_of}" },
  });
}

/** A build context whose resolver hands back `children[flowId]`. */
function contextWith(children: ChildFlows): BuildContext {
  return new BuildContext({
    registry: new Map(),
    resolver: (flowId) => {
      const child = children[flowId];
      if (child === undefined) throw new Error(`unknown child flow ${flowId}`);
      return child;
    },
  });
}

/** Parse node `nodeId` of the flow of conformance case `id`. */
function seedNode(id: string, nodeId: string): NodeDescriptor {
  const { text, label } = caseText(id);
  return nodesOf(text, [], label).get(nodeId)!;
}

/** Return the field names of record type `type`, in order. */
function fieldNames(type: Type | undefined): string[] {
  return [...type!.fields!.keys()];
}

test("seed 03's child signature", () => {
  const sig = childSignature(researchOne());

  const parameters = new Map(sig.inputs.map((parameter) => [parameter.name, parameter]));
  expect([...parameters.keys()]).toEqual(["topic", "as_of"]);
  expect(
    sig.inputs.every(
      (parameter) => isExactly(parameter, Parameter) && isExactly(parameter.contract, ValueContract),
    ),
  ).toBe(true);
  expect(parameters.get("topic")!.required).toBe(true);
  expect(parameters.get("topic")!.contract!.type.kind).toBe(ValueKind.STRING);
  const asOf = parameters.get("as_of")!;
  expect(asOf.required).toBe(false);
  expect(asOf.hasDefault).toBe(true);
  expect(asOf.default).toBe("2026-01-01");
  expect(asOf.contract!.type.kind).toBe(ValueKind.DATE);
  expect(sig.inputs.every((parameter) => parameter.coerceStrings)).toBe(true);
  // Two outputs make a closed record.
  expect(isExactly(sig.output, ValueContract)).toBe(true);
  const output = sig.output as ValueContract;
  expect(output.type.kind).toBe(ValueKind.OBJECT);
  expect(fieldNames(output.type)).toEqual(["report", "asof"]);
});

test("seed 04's call builds and re-exports the child's output", () => {
  const child = researchOne();

  const result = seedNode("load-map-call/seed_04_call", "research").build(contextWith({ "research-one": child }));

  const node = result.node as CallNode;
  expect(node).toBeInstanceOf(CallNode);
  expect(node).not.toBeInstanceOf(MapNode);
  expect("over" in node).toBe(false);
  expect("parallel" in node).toBe(false);
  expect(node.flowId).toBe("research-one");
  expect(node.child).toBe(child.compiled);
  expect(node.childInputs).toStrictEqual(child.input);
  expect(node.outputContract).toBeInstanceOf(ValueContract);
  const contract = node.outputContract as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.OBJECT);
  expect(fieldNames(contract.type)).toEqual(["report", "asof"]);
  expect(result.wiring).toEqual(new Map([["topic", "${input.topic}"]]));
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["topic"]);
});

test("seed 04's call wiring and parameter carry origins", () => {
  const { text, label } = caseText("load-map-call/seed_04_call");
  const research = SourceDocument.parse(text, label).root.child("nodes").child("research");

  const { flow, builder } = assembleText(text, label, { children: { "research-one": researchOne() } });

  const wiring = flow.compiled.wiring.get("research")!;
  expect([...wiring].map(([name, binding]) => [name, (binding as Binding).source])).toEqual([
    ["topic", "${input.topic}"],
  ]);
  const topic = wiring.get("topic") as Binding;
  expect(topic.origin).toBeInstanceOf(OriginRef);
  expect(builder.resolveRegistered(topic.origin!)!.range).toEqual(research.child("input").rangeOf("topic")!);
  const node = flow.compiled.nodes.get("research")!;
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["topic"]);
  expect(node.params![0]!.origin).toBeInstanceOf(OriginRef);
});

test("seed 05's map builds a list of the child's output, parallel", () => {
  const child = researchOne();

  const result = seedNode("load-map-call/seed_05_call_map", "research_each").build(
    contextWith({ "research-one": child }),
  );

  const node = result.node as MapNode;
  expect(node).toBeInstanceOf(MapNode);
  expect("over" in node).toBe(false);
  expect(node.flowId).toBe("research-one");
  expect(node.parallel).toBe(true);
  expect(node.title).toBe("Research each topic");
  expect(node.child).toBe(child.compiled);
  const contract = node.outputContract as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.LIST_OBJECT);
  expect(contract.type.element!.kind).toBe(ValueKind.OBJECT);
  expect(fieldNames(contract.type.element)).toEqual(["report", "asof"]);
  // `over` is the iteration source, wired first; it is not a parameter.
  expect(result.wiring).toEqual(
    new Map([
      ["over", "${input.topics}"],
      ["topic", "${item}"],
      ["as_of", '${input.as_of:-"2026-01-01"}'],
    ]),
  );
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["topic", "as_of"]);
});

test("seed 05's map wiring and parameters carry origins", () => {
  const { text, label } = caseText("load-map-call/seed_05_call_map");

  const { flow } = assembleText(text, label, { children: { "research-one": researchOne() } });

  const wiring = flow.compiled.wiring.get("research_each")!;
  expect([...wiring].map(([name, binding]) => [name, (binding as Binding).source])).toEqual([
    ["over", "${input.topics}"],
    ["topic", "${item}"],
    ["as_of", '${input.as_of:-"2026-01-01"}'],
  ]);
  expect([...wiring.values()].every((binding) => (binding as Binding).origin instanceof OriginRef)).toBe(true);
  const node = flow.compiled.nodes.get("research_each")!;
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["topic", "as_of"]);
  expect(node.params!.every((parameter) => parameter.origin instanceof OriginRef)).toBe(true);
});

test("a map input named over is reserved", () => {
  // Checked before the child is resolved, so no resolver is needed.
  const desc = new MapDescriptor({ id: "m", call: "child", over: "${input.xs}", inputs: { over: "${item}" } });

  const error = loadError(() => desc.build(new BuildContext({ registry: new Map() })));

  expect(error.message).toBe(
    "map node 'm': input name 'over' is reserved (the iteration source); rename the `input:` entry",
  );
});

test("a call binding to an undeclared child input is loud", () => {
  const desc = new CallDescriptor({ id: "research", call: "research-one", inputs: { symbol: "${input.topic}" } });

  const error = loadError(() => desc.build(contextWith({ "research-one": researchOne() })));

  expect(error.message).toBe(
    "call node 'research': binding 'symbol' is not a declared input of callable 'research-one'\n" +
      "  call node 'research': required callable input 'topic' is unbound",
  );
});

test("a map child with no output is loud", () => {
  const child = handChild({
    inputs: parametersFromYaml("input:\n  topic: !str\n", new Map()),
    producers: { emit: null },
    output: null,
  });
  const signature = childSignature(child);
  expect(signature.output).toBeNull();
  expect(signature.outputNames).toEqual([]);
  expect(signature.declaredOutputs).toEqual([]);
  const desc = new MapDescriptor({
    id: "each",
    call: "no-output",
    over: "${input.topics}",
    inputs: { topic: "${item}" },
  });

  const error = loadError(() => desc.build(contextWith({ "no-output": child })));

  expect(error.message).toBe(
    "map node 'each': callable 'no-output' must declare a non-empty `output:`; " +
      "a mapped call (over:) returns a list of the child's results",
  );
});

test("a call over a nullable child input re-exports the nullable output", () => {
  const echo = handChild({
    inputs: parametersFromYaml("input:\n  maybe: !Optional(str)\n", new Map()),
    output: "${input.maybe}",
  });
  const desc = new CallDescriptor({ id: "invoke", call: "echo", inputs: { maybe: "${null}" } });

  const contract = desc.build(contextWith({ echo })).node.outputContract as ValueContract;

  expect(contract.type.kind).toBe(ValueKind.STRING);
  expect(contract.type.nullable).toBe(true);
});

test("a map over a partial child output leaves its contract unset", () => {
  // An unproven field leaves the element unknown, so the list isn't enforced.
  const child = handChild({ producers: { step: STR }, output: { n: "${step.output + 1}", s: "${step.output}" } });
  expect(childSignature(child).output).toBeInstanceOf(PartialRecordContract);
  const desc = new MapDescriptor({ id: "each", call: "child", over: "${input.items}" });

  expect(desc.build(contextWith({ child })).node.outputContract).toBeNull();
});

test("field labels name an inline call's argument and the call", () => {
  const child = handChild({ inputs: parametersFromYaml("input:\n  q: !str\n", new Map()), output: null });
  const desc = new CallDescriptor({
    id: "__call_0",
    call: "child",
    inputs: { y: "${input.s}" },
    fieldLabels: new Map([
      ["input 'y'", "node 'n' input 'a' call(child) argument 'y'"],
      ["call", "node 'n' input 'a' call(child)"],
    ]),
  });

  const error = loadError(() => desc.build(contextWith({ child })));

  expect(error.message).toBe(
    "node 'n' input 'a' call(child) argument 'y' is not a declared input of callable 'child'\n" +
      "  node 'n' input 'a' call(child): required callable input 'q' is unbound",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([{ id: "load-map-call/engine_call_map__map_input_named_over_is_reserved", call: assembled }])(
  "assemble: $id",
  runCase,
);
