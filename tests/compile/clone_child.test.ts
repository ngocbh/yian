/**
 * The pure child cloner `cloneChild` of `compile/expand.ts`, and a cloned child END's lowered
 * post-assert firing record-scoped through `evalNode`.
 *
 * A cross-node ref (`${<callsite>/X.output}`) is lowered to a synth param (`__assert_tail`) fed by
 * a `${.../X.output:-null}` wiring, so binding puts the value into the record and the assert reads
 * it node-locally; nothing reads the pool around the record.
 */

import { expect, test } from "bun:test";

import { cloneChild } from "../../src/compile/expand.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import {
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySetupError,
} from "../../src/memory/contracts.ts";
import { type Node, NodeKind, Output } from "../../src/nodes/base.ts";
import { Assertion, Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { EvaluationFailure, type WorkerOutcome } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, FuncNode, nodeFlow } from "../helpers/run_state_fixtures.ts";

/** A required text parameter that coerces supplied strings. */
function parameter(name: string): Parameter {
  return new Parameter(name, {
    contract: new ValueContract(Type.scalar(ValueKind.STRING)),
    required: true,
    coerceStrings: true,
  });
}

/**
 * A child shaped START -> head (reads `${input.topic}`) -> tail (reads `${head.output.r}`) -> END,
 * with one END producer edge per declared output, as the loader mints them.
 */
function child(outputs: readonly FlowOutput[]): CompiledFlow {
  const head = new FuncNode("head", (inputs) => ({ r: inputs["topic"] }), { reads: { topic: "${input.topic}" } });
  const tail = new FuncNode("tail", (inputs) => inputs["r"], { reads: { r: "${head.output.r}" } });
  const parameters = [parameter("topic")];
  // These tests use a single output as a value; two or more form a record.
  const end =
    outputs.length <= 1
      ? EndNode.value(END_ID, { outputName: outputs[0]?.name ?? null })
      : EndNode.record(END_ID, { outputNames: outputs.map((output) => output.name) });
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: parameters })],
    ["head", head],
    ["tail", tail],
    [END_ID, end],
  ]);
  const edges = [
    new Edge("__start__->head#0", START_ID, "head", { inputGroup: "topic" }),
    new Edge("head->tail#0", "head", "tail", { inputGroup: "r" }),
  ];
  for (const [i, output] of outputs.entries()) {
    const producer = (output.binding.source as string).slice(2, -1).split(".")[0]!;
    edges.push(new Edge(`${producer}->${END_ID}#${i}`, producer, END_ID, { inputGroup: output.name }));
  }
  const wiring = new Map<string, ReadonlyMap<string, unknown>>([
    ["head", new Map(Object.entries(head.reads))],
    ["tail", new Map(Object.entries(tail.reads))],
    [START_ID, new Map(parameters.map((p) => [p.name, `\${input.${p.name}}`]))],
    [END_ID, new Map(outputs.map((output) => [output.name, output.binding]))],
  ]);
  return new CompiledFlow(nodes, edges, { outputs, wiring });
}

/** A child with one declared output. */
function single(): CompiledFlow {
  return child([new FlowOutput("out", "${tail.output}")]);
}

/** Set one wiring entry of `flow`'s node `nodeId`, replacing the shared inner map. */
function setWiring(flow: Flow, nodeId: string, param: string, source: unknown): void {
  flow.wiring.set(nodeId, new Map([...(flow.wiring.get(nodeId) ?? []), [param, new Binding(source)]]));
}

test("clone child namespaces ids", () => {
  const cloned = cloneChild(single(), "each", { topic: "ACME" });

  expect(cloned).toBeInstanceOf(Flow);
  expect(cloned.nodes.has("each/head") && cloned.nodes.has("each/tail")).toBe(true);
  // The child START and END are cloned too (namespaced).
  expect(cloned.nodes.has("each/__start__") && cloned.nodes.has("each/__end__")).toBe(true);
  // Internal `${head.output.r}` is re-namespaced to the node-first `${each/head.output.r}`.
  expect(cloned.wiring.get("each/tail")!.get("r")).toEqual(new Binding("${each/head.output.r}"));
  expect(cloned.edges.some((e) => e.id === "each/head->tail#0")).toBe(true);
  // The start id is the namespaced child START (the sole seed point), not `each/head`.
  expect(cloned.startId).toBe("each/__start__");
});

test("clone child preserves node origin while namespacing runtime ids", () => {
  const flow = single();
  const origin = new OriginRef("def/child/node/head");
  flow.nodes.get("head")!.origin = origin;

  const cloned = cloneChild(flow, "each", { topic: "ACME" });

  expect(cloned.nodes.get("each/head")!.origin).toBe(origin);
  expect(cloned.nodes.get("each/head")!.id).toBe("each/head");
  expect(flow.nodes.get("head")!.id).toBe("head");
});

test("clone child prefixes existing generated origin context", () => {
  const flow = single();
  const outer = new OriginRef("root/nodes/call");
  const nested = new OriginRef("child/nodes/map");
  flow.nodes.get("head")!.originContext = [nested];

  const cloned = cloneChild(flow, "each", { topic: "ACME" }, { originContext: [outer] });

  expect(cloned.nodes.get("each/head")!.originContext).toEqual([outer, nested]);
  expect(flow.nodes.get("head")!.originContext).toEqual([nested]);
});

test("clone child rejects a dynamic static memory template atomically", () => {
  const flow = single();
  const occurrence = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["call"], 0);
  flow.nodes.get("head")!.memoryScope = new MemoryScope(["call", "head"], { occurrencePath: [occurrence] });

  expect(() => cloneChild(flow, "call", { topic: "ACME" })).toThrow(MemorySetupError);
  expect(() => cloneChild(flow, "call", { topic: "ACME" })).toThrow(/static memory template/);
  expect(flow.nodes.get("head")!.memoryScope!.occurrencePath).toEqual([occurrence]);
});

test("clone child refuses a memory scope that isn't a scope", () => {
  const other = single();
  Object.assign(other.nodes.get("head")!, { memoryScope: "notes" });
  expect(() => cloneChild(other, "call", { topic: "ACME" })).toThrow(MemorySetupError);
  expect(() => cloneChild(other, "call", { topic: "ACME" })).toThrow(/'head' has a noncanonical static memory scope/);
});

test("clone child rewrites one binding layer and preserves its origin", () => {
  const flow = single();
  const origin = new OriginRef("child-tail-r");
  const authored = new Binding("${head.output.r}", { origin });
  flow.wiring.set("tail", new Map([["r", authored]]));

  const cloned = cloneChild(flow, "each", { topic: "ACME" });

  const rewritten = cloned.wiring.get("each/tail")!.get("r")!;
  expect(rewritten).toEqual(new Binding("${each/head.output.r}", { origin }));
  expect(rewritten.origin).toBe(origin);
  expect(rewritten).not.toBe(authored);
});

test("clone child does not rewrite a nested binding payload", () => {
  const flow = single();
  const inner = new Binding("${head.output.r}", { origin: new OriginRef("inner") });
  flow.wiring.set("tail", new Map([["r", new Binding(inner, { origin: new OriginRef("outer") })]]));

  const cloned = cloneChild(flow, "each", { topic: "ACME" });

  const rewritten = cloned.wiring.get("each/tail")!.get("r")!;
  expect(rewritten).toBeInstanceOf(Binding);
  expect(rewritten.source).toBe(inner);
});

test("clone child seeds start with call args as constants", () => {
  // A record value is resolved data: a `${...}` inside it is escaped, not a forward ref, so the
  // child START reads the exact text and no producer edge is minted.
  const cloned = cloneChild(single(), "each", { topic: "${upstream.output}" });
  expect(cloned.wiring.get("each/__start__")!.get("topic")).toEqual(new Binding("$${upstream.output}"));
  expect(cloned.edges.some((e) => e.to === "each/__start__")).toBe(false);
  // `each/head`'s `${input.topic}` is re-pointed to the namespaced START, not baked to a literal.
  const cloned2 = cloneChild(single(), "each", { topic: "ACME" });
  expect(cloned2.wiring.get("each/__start__")!.get("topic")).toEqual(new Binding("ACME"));
  expect(cloned2.edges.some((e) => e.to === "each/__start__")).toBe(false);
  expect(cloned2.wiring.get("each/head")!.get("topic")).toEqual(new Binding("${each/__start__.output.topic}"));
});

test("clone child alias target is child end", () => {
  const cloned = cloneChild(single(), "each", { topic: "ACME" });

  expect(cloned.endId).toBe("each/__end__");
  const end = cloned.nodes.get("each/__end__")!;
  expect(end).toBeInstanceOf(EndNode);
  expect(end.kind).toBe(NodeKind.END);
});

test("clone child end carries producer edges", () => {
  // The cloned child END keeps its producer->END edges (re-keyed under the callsite), so the alias
  // filler fires only after its producers settle.
  const flow = child([new FlowOutput("report", "${tail.output}"), new FlowOutput("n", "${head.output.r}")]);

  const cloned = cloneChild(flow, "each", { topic: "ACME" });

  const intoEnd = cloned.edges.filter((e) => e.to === "each/__end__").map((e) => [e.from, e.to, e.inputGroup]);
  expect(intoEnd).toEqual([
    ["each/tail", "each/__end__", "report"],
    ["each/head", "each/__end__", "n"],
  ]);
  expect(cloned.wiring.get("each/__end__")).toEqual(
    new Map([
      ["report", new Binding("${each/tail.output}")],
      ["n", new Binding("${each/head.output.r}")],
    ]),
  );
});

test("clone child lowered post asserts survive clone", () => {
  // A def's flow-level `asserts:` are lowered onto its END at build: each `${producer.output}`
  // post-assert becomes a synth END param (`__assert_<pid>`) fed by a `${pid.output:-null}` wiring
  // and a producer->END edge. Cloning deep-copies the END, re-namespaces its wiring and re-keys its
  // edges, so the whole structural dependency survives.
  const flow = single();
  const end = flow.nodes.get(END_ID)!;
  const origin = new OriginRef("flow/asserts/0");
  end.postAsserts = [new Assertion("__assert_tail != ''", "tail.output != ''", { origin })];
  end.params = [...end.params!, new Parameter("__assert_tail")];
  setWiring(flow, END_ID, "__assert_tail", "${tail.output:-null}");
  flow.edges.push(
    new Edge(`tail->${END_ID}#assert`, "tail", END_ID, { inputGroup: "__assert_tail", optional: true }),
  );

  const cloned = cloneChild(flow, "each", { topic: "ACME" });

  const clonedEnd = cloned.nodes.get("each/__end__")!;
  // The lowered post-assert is node-local (it names a synth param, not a node), so unchanged.
  expect(clonedEnd.postAsserts).toEqual([new Assertion("__assert_tail != ''", "tail.output != ''", { origin })]);
  expect(clonedEnd.postAsserts[0]!.origin).toBe(origin);
  expect(cloned.wiring.get("each/__end__")!.get("__assert_tail")).toEqual(new Binding("${each/tail.output:-null}"));
  expect(
    cloned.edges.some((e) => e.from === "each/tail" && e.to === "each/__end__" && e.inputGroup === "__assert_tail"),
  ).toBe(true);
});

test("spawner assert append preserves child and spawner origins", () => {
  const flow = single();
  const childOrigin = new OriginRef("child/asserts/0");
  const spawnerOrigin = new OriginRef("parent/nodes/call/asserts/0");
  flow.nodes.get(END_ID)!.postAsserts = [
    new Assertion("output != null", "child output exists", { origin: childOrigin }),
  ];

  const cloned = cloneChild(
    flow,
    "each",
    { topic: "ACME" },
    { postAsserts: [new Assertion("output != null", "call output exists", { origin: spawnerOrigin })] },
  );
  const clonedEnd = cloned.nodes.get(cloned.endId)!;

  expect(clonedEnd.postAsserts.map((a) => a.display)).toEqual(["child output exists", "call output exists"]);
  expect(clonedEnd.postAsserts.map((a) => a.origin)).toEqual([childOrigin, spawnerOrigin]);
});

/** Evaluate an END whose post-assert reads the synth param bound from `${each/tail.output}`. */
function runTailAssertEnd(pool: VariablePool): Promise<WorkerOutcome> {
  const end = EndNode.value("each/__end__", { outputName: "out" });
  end.postAsserts = [new Assertion("__assert_tail != ''", "each/tail.output != ''")];
  end.params = [...end.params!, new Parameter("__assert_tail")];
  const flow = nodeFlow(end, { out: "${each/tail.output}", __assert_tail: "${each/tail.output:-null}" });
  return drive(end, { pool, flow });
}

test("child end satisfied synth param post assert succeeds", async () => {
  const pool = new VariablePool();
  pool.set("each/tail", "ok");

  expect(await runTailAssertEnd(pool)).toBeInstanceOf(Output);
});

test("child end violated synth param post assert fails", async () => {
  const pool = new VariablePool();
  pool.set("each/tail", "");

  const result = await runTailAssertEnd(pool);

  expect(result).toBeInstanceOf(EvaluationFailure);
  expect((result as EvaluationFailure).failure.message).toContain("post-assert");
});
