/**
 * CALL and MAP build over a child's signature: the seed flows, the reserved `over` input,
 * undeclared bindings, children without output, nullable inputs, and the resolver requirement.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, expect, test } from "bun:test";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { ChildSignature, childSignature } from "../../src/compose/nodes/resolver.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, isExactly, Parameter, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  type CaseRow,
  caseText,
  type ChildFlows,
  handChild,
  linkConformanceFixtures,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/** Seed 03 (`research-one`), the child seeds 04 and 05 call. */
const RESEARCH_ONE = caseText("load-map-call/seed_04_call", { file: "03-research-one.yaml" }).text;

/** The search path holding seed 03, as `03-research-one.yaml`. */
let seeds: string;

beforeAll(() => {
  seeds = mkdtempSync(join(tmpdir(), "yian-call-map-"));
  writeFileSync(join(seeds, "03-research-one.yaml"), RESEARCH_ONE);
  linkConformanceFixtures(seeds);
});

afterAll(() => rmSync(seeds, { recursive: true, force: true }));

/** Load the flow of conformance case `id` with seed 03 on the search path. */
function loadSeed(id: string): Promise<ComposedFlow> {
  const { text, label } = caseText(id);
  return loadFlow(text, { searchPaths: [seeds], sourceLabel: label });
}

/** A child resolver serving seed 03 as `research-one`, and failing on any other id. */
async function makeResolver() {
  const children: Record<string, ComposedFlow> = { "research-one": await loadConformanceFlow(RESEARCH_ONE) };
  return (flowId: string) => {
    const child = children[flowId];
    if (child === undefined) throw new Error(`unknown child flow ${flowId}`);
    return child;
  };
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

/** Return the field names of record type `type`, in order. */
function fieldNames(type: Type | undefined): string[] {
  return [...type!.fields!.keys()];
}

/** Return each binding's source in `wiring`, in order. */
const sources = (wiring: ReadonlyMap<string, unknown>) =>
  [...wiring].map(([name, binding]) => [name, (binding as Binding).source]);

// ---------- seed 03: the child flow loads and exposes a signature ----------

test("seed 03 loads as a child flow", async () => {
  const loaded = await loadConformanceFlow(RESEARCH_ONE);
  expect(loaded).toBeInstanceOf(ComposedFlow);
  const sig = childSignature(loaded);
  expect(sig).toBeInstanceOf(ChildSignature);

  // topic is a required str; as_of has a typed date default, so it isn't required.
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

// ---------- seed 04: CALL loads and re-exports the child's output contract ----------

test("seed 04's call loads and re-exports the child's output", async () => {
  const loaded = await loadSeed("load-map-call/seed_04_call");
  expect(loaded).toBeInstanceOf(ComposedFlow);

  const research = loaded.compiled.nodes.get("research") as CallNode;
  expect(research).toBeInstanceOf(CallNode);
  expect(research).not.toBeInstanceOf(MapNode);
  // CALL carries no over or parallel.
  expect("over" in research).toBe(false);
  expect("parallel" in research).toBe(false);
  expect(research.flowId).toBe("research-one");
  // The exact contract re-exports the child's one output, a `{report, asof}` record.
  expect(research.outputContract).toBeInstanceOf(ValueContract);
  const contract = research.outputContract as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.OBJECT);
  expect(fieldNames(contract.type)).toEqual(["report", "asof"]);
  // CALL sources live on the flow's wiring; the node carries only the parameter names.
  const wiring = loaded.compiled.wiring.get("research")!;
  expect(sources(wiring)).toEqual([["topic", "${input.topic}"]]);
  const topic = wiring.get("topic") as Binding;
  expect(topic.origin).not.toBeNull();
  const { text, label } = caseText("load-map-call/seed_04_call");
  const authored = SourceDocument.parse(text, label).root.child("nodes").child("research");
  expect(loaded.originResolver!.resolve(topic.origin!)!.range).toEqual(authored.child("input").rangeOf("topic")!);
  expect(research.params!.map((parameter) => parameter.name)).toEqual(["topic"]);
  expect(research.params![0]!.origin).not.toBeNull();
});

test("seed 04's downstream dotted ref validates against the child record", async () => {
  // `take` reads `${research.output.report}` and `${research.output.asof}`; the CALL node's output
  // contract (a checked record) lets the dotted-field walk pass.
  const { compiled } = await loadSeed("load-map-call/seed_04_call");

  const wiring = compiled.wiring.get("take")!;
  expect(sources(wiring)).toEqual([
    ["brief", "${research.output.report}"],
    ["asof", "${research.output.asof}"],
  ]);
  expect([...wiring.values()].every((binding) => (binding as Binding).origin !== null)).toBe(true);
  expect(compiled.nodes.get("take")!.params!.every((parameter) => parameter.origin !== null)).toBe(true);
});

// ---------- seed 05: MAP loads with a list of the child's output, parallel ----------

test("seed 05's map loads with a list output, parallel", async () => {
  const loaded = await loadSeed("load-map-call/seed_05_call_map");
  expect(loaded).toBeInstanceOf(ComposedFlow);

  const node = loaded.compiled.nodes.get("research_each") as MapNode;
  expect(node).toBeInstanceOf(MapNode);
  // The source rides the flow's wiring, not the node.
  expect("over" in node).toBe(false);
  expect(node.flowId).toBe("research-one");
  expect(node.parallel).toBe(true);
  expect(node.title).toBe("Research each topic");
  const contract = node.outputContract as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.LIST_OBJECT);
  expect(contract.type.element!.kind).toBe(ValueKind.OBJECT);
  expect(fieldNames(contract.type.element)).toEqual(["report", "asof"]);
  // `over` is the iteration source, wired first; it is not a parameter.
  const wiring = loaded.compiled.wiring.get("research_each")!;
  expect(sources(wiring)).toEqual([
    ["over", "${input.topics}"],
    ["topic", "${item}"],
    ["as_of", '${input.as_of:-"2026-01-01"}'],
  ]);
  expect([...wiring.values()].every((binding) => (binding as Binding).origin !== null)).toBe(true);
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["topic", "as_of"]);
  expect(node.params!.every((parameter) => parameter.origin !== null)).toBe(true);
});

test("a map input named over is reserved", async () => {
  // A map's `over` is the iteration source (the reserved wiring key); an `input:` parameter named
  // `over` collides and is rejected at load.
  const text =
    "id: bad-over\nname: bad\ninput:\n  xs: !list(str)\n" +
    "nodes:\n  m:\n    kind: map\n    call: child\n    over: ${input.xs}\n" +
    "    input:\n      over: ${item}\n" +
    "output: ${m.output}\n";

  const error = await loadErrorAsync(loadFlow(text));

  expect(error.message).toBe(
    "map node 'm': input name 'over' is reserved (the iteration source); rename the `input:` entry",
  );
});

// ---------- name and arity checks: loud on a bad binding or a bad MAP child ----------

const CALL_BAD_NAME = `
id: bad-ref
name: bad_ref
input:
  topic: !str
uses:
  research-one: research-one
nodes:
  research:
    kind: call
    call: research-one
    input:
      symbol: \${input.topic}     # \`symbol\` is NOT a declared callable input
output:
  out: \${research.output}
`;

test("a call binding to an undeclared child input is loud", async () => {
  const error = await loadErrorAsync(loadFlow(CALL_BAD_NAME, { childResolver: await makeResolver() }));

  expect(error.message).toBe(
    "call node 'research': binding 'symbol' is not a declared input of callable 'research-one'\n" +
      "  call node 'research': required callable input 'topic' is unbound",
  );
});

test("a map binding to an undeclared child input names the map node", async () => {
  const text = CALL_BAD_NAME.replace("kind: call", "kind: map\n    over: ${[input.topic]}");

  const error = await loadErrorAsync(loadFlow(text, { childResolver: await makeResolver() }));

  expect(error.message).toBe(
    "map node 'research': binding 'symbol' is not a declared input of callable 'research-one'\n" +
      "  map node 'research': required callable input 'topic' is unbound",
  );
});

const CHILD_NO_OUTPUT = `
id: no-output
name: no_output
input:
  topic: !str
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
    code: ./fixtures/compose_codefns.ts:echo
`;

const MAP_OVER_NO_OUTPUT = `
id: bad-map
name: bad_map
input:
  topics: !list(str)
uses:
  no-output: no-output
nodes:
  each:
    kind: map
    call: no-output
    over: \${input.topics}
    input:
      topic: \${item}
output:
  out: \${each.output}
`;

test("a map child with no output is loud", async () => {
  const child = await loadConformanceFlow(CHILD_NO_OUTPUT);
  const signature = childSignature(child);
  expect(signature.output).toBeNull();
  // No declaration, so a MAP cannot form list(U).
  expect(signature.outputNames).toEqual([]);
  expect(signature.declaredOutputs).toEqual([]);
  const resolver = (flowId: string) => {
    expect(flowId).toBe("no-output");
    return child;
  };

  const error = await loadErrorAsync(loadFlow(MAP_OVER_NO_OUTPUT, { childResolver: resolver }));

  expect(error.message).toBe(
    "map node 'each': callable 'no-output' must declare a non-empty `output:`; " +
      "a mapped call (over:) returns a list of the child's results",
  );
});

const NULL_TO_NULLABLE_CALL = `
id: null-to-nullable-call
name: null_to_nullable_call
defs:
  echo:
    input:
      maybe: !Optional(str)
    nodes: {}
    output: \${input.maybe}
nodes:
  invoke:
    kind: call
    call: echo
    input:
      maybe: \${null}
output: \${invoke.output}
`;

test("a call accepts a proven null for a nullable child input", async () => {
  // Static binding compatibility mirrors runtime nullable coercion.
  const loaded = await loadFlow(NULL_TO_NULLABLE_CALL);

  const contract = loaded.compiled.nodes.get("invoke")!.outputContract as ValueContract;
  expect(contract).not.toBeNull();
  expect(contract.type.kind).toBe(ValueKind.STRING);
  expect(contract.type.nullable).toBe(true);
});

// ---------- a composed call or map needs a resolver ----------

test("a call or map without a resolver is loud", async () => {
  const error = await loadErrorAsync(loadFlow(caseText("load-map-call/seed_04_call").text));

  expect(error.message).toBe(
    "callable 'research-one' (uses: '03-research-one') needs a search path or child_resolver to resolve " +
      "(load with search_paths=[...])",
  );
});

test("a flow without calls loads without a resolver", async () => {
  expect(await loadConformanceFlow(caseText("load-agent/seed_01_structured_agent").text)).toBeInstanceOf(
    ComposedFlow,
  );
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

test.each<CaseRow>([
  { id: "load-map-call/engine_call_map__map_input_named_over_is_reserved", call: assembled },
  { id: "load-map-call/engine_call_map__call_accepts_proven_null_for_nullable_child_input", call: assembled },
])("assemble: $id", runCase);
