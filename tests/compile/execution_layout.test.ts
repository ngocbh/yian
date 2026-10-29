/**
 * Memory-independent static execution paths and atomic scope stamping
 * (`compile/execution_layout.ts`): scopes with and without memory, rejected graphs stamping
 * nothing, static children and loops, and authored agent-variable owners.
 */

import { describe, expect, test } from "bun:test";

import { stampExecutionScopes } from "../../src/compile/execution_layout.ts";
import { callSubgraph, loopContinueSubgraph, mapSubgraph } from "../../src/compile/expand.ts";
import { applyEffectiveLlmMap } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { resolveMemoryLayout, stampMemoryLayout } from "../../src/compile/memory_layout.ts";
import { CompiledFlow, END_ID, Flow, nodeChild, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

/** A CODE node returning 1. */
function code(id: string): CodeNode {
  return new CodeNode(id, { code: "return 1" });
}

/** Expect `fn` to throw exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** Return every statically reachable node by its locator, joined with `/`. */
function nodesByLocator(flow: Flow, path: readonly string[] = [], found = new Map<string, Node>()): Map<string, Node> {
  for (const [nodeId, node] of flow.nodes) {
    const locator = [...path, nodeId];
    found.set(locator.join("/"), node);
    const child = nodeChild(node);
    if (child !== null) nodesByLocator(child, locator, found);
  }
  return found;
}

test("execution scope has an explicit manifest policy", () => {
  const node = code("plain");
  const flow = new CompiledFlow(new Map<string, Node>([["plain", node]]), []);
  expect(node.executionScope).toBeNull();
  const before = flowManifest(flow);
  const fingerprint = flowFingerprint(flow);

  node.executionScope = new ExecutionScope(["plain"]);

  expect(flowManifest(flow)).toEqual(before);
  expect(flowFingerprint(flow)).toBe(fingerprint);
  Object.assign(node, { misspelledExecutionScope: node.executionScope });
  expectError(
    () => flowManifest(flow),
    RangeError,
    'flow.nodes.plain: unknown noncanonical execution field(s) for code: ["misspelledExecutionScope"]',
  );
});

describe.each([
  ["Flow", (nodes: Map<string, Node>) => new Flow(nodes, [])],
  ["CompiledFlow", (nodes: Map<string, Node>) => new CompiledFlow(nodes, [])],
])("execution scopes exist without memory policies (%s)", (_name, makeFlow) => {
  test("stamps every node", () => {
    const node = new CodeNode("authored~3/step", { code: "return 1", title: "display name" });
    node.origin = new OriginRef("unrelated:source");
    const flow = makeFlow(
      new Map<string, Node>([
        [node.id, node],
        [START_ID, new StartNode(START_ID, { params: [] })],
        [END_ID, EndNode.value(END_ID, { outputName: null })],
      ]),
    );
    const expected = new Map<string, ExecutionScope>([
      [END_ID, new ExecutionScope([END_ID], { boundary: "end" })],
      [START_ID, new ExecutionScope([START_ID], { boundary: "start" })],
      [node.id, new ExecutionScope([node.id])],
    ]);

    stampExecutionScopes(flow);
    const stamped = new Map([...flow.nodes].map(([id, item]) => [id, item.executionScope]));
    expect(stamped).toEqual(expected);
    expect([...flow.nodes.values()].every((item) => item.memoryScope === null)).toBe(true);
    // Titles and origins aren't part of a scope, so stamping again changes nothing.
    node.title = "another display name";
    node.origin = new OriginRef("different:source");
    stampExecutionScopes(flow);
    expect(new Map([...flow.nodes].map(([id, item]) => [id, item.executionScope]))).toEqual(stamped);

    const plain = makeFlow(new Map<string, Node>([["plain", code("plain")]]));
    stampExecutionScopes(plain);
    expect(plain.nodes.get("plain")!.executionScope).toEqual(new ExecutionScope(["plain"]));
    expect(() => stampExecutionScopes(makeFlow(new Map()))).not.toThrow();
  });
});

test("aliased child targets are rejected before any scope is stamped", () => {
  const child = new Flow(new Map<string, Node>([["answer", code("answer")]]), []);
  const flow = new CompiledFlow(
    new Map<string, Node>([
      ["left", new CallNode("left", { flowId: "child", child })],
      ["right", new CallNode("right", { flowId: "child", child })],
    ]),
    [],
  );
  expect(() => stampExecutionScopes(flow)).toThrow(RangeError);
  expect(() => stampExecutionScopes(flow)).toThrow(/multiple static paths/);
  expect([...flow.nodes.values(), ...child.nodes.values()].every((node) => node.executionScope === null)).toBe(true);
});

test("a different existing scope is rejected before any scope is stamped", () => {
  const first = code("first");
  const last = new CodeNode("last", { code: "return 2" });
  const flow = new CompiledFlow(new Map<string, Node>([["first", first], ["last", last]]), []);
  const other = new ExecutionScope(["other"]);
  last.executionScope = other;

  expectError(() => stampExecutionScopes(flow), RangeError, 'node at ["last"] already has a different execution scope');
  expect(first.executionScope).toBeNull();
  expect(last.executionScope).toBe(other);
});

test.each([
  ["an object", {}],
  ["a string", "last"],
])("an existing scope that is %s is rejected before stamping", (_name, value) => {
  const first = code("first");
  const last = new CodeNode("last", { code: "return 2" });
  const flow = new CompiledFlow(new Map<string, Node>([["first", first], ["last", last]]), []);
  Object.assign(last, { executionScope: value });

  expectError(() => stampExecutionScopes(flow), RangeError, 'node at ["last"] already has a different execution scope');
  expect(first.executionScope).toBeNull();
  expect(last.executionScope as unknown).toBe(value);
});

test("recursive static child cycle is rejected without mutation", () => {
  const flow = new Flow(new Map(), []);
  const node = new CallNode("recurse", { flowId: "child", child: flow });
  flow.nodes.set(node.id, node);
  expect(() => stampExecutionScopes(flow)).toThrow(RangeError);
  expect(() => stampExecutionScopes(flow)).toThrow(/recursive child-flow cycle/);
  expect(node.executionScope).toBeNull();
});

test("static children are discovered through the node-owned hook", () => {
  const child = new Flow(new Map<string, Node>([["leaf", code("leaf")]]), []);

  class Wrapper extends Node {
    readonly kind = NodeKind.CODE;

    override originChildren(): readonly unknown[] {
      return [child];
    }
  }

  const flow = new Flow(new Map<string, Node>([["wrapper", new Wrapper("wrapper")]]), []);
  stampExecutionScopes(flow);
  expect(child.nodes.get("leaf")!.executionScope).toEqual(
    new ExecutionScope(["wrapper", "leaf"], { flowPath: ["wrapper"] }),
  );
});

test("only the containing flow's actual boundaries are marked", () => {
  const flow = new Flow(
    new Map<string, Node>([
      ["entry", new StartNode("entry", { params: [] })],
      ["result", EndNode.value("result", { outputName: null })],
      ["helper", new StartNode("helper", { params: [] })],
    ]),
    [],
    { startId: "entry", endId: "result" },
  );
  stampExecutionScopes(flow);
  expect([...flow.nodes].map(([id, node]) => [id, node.executionScope!.boundary])).toEqual([
    ["entry", "start"],
    ["result", "end"],
    ["helper", null],
  ]);
});

test("static loop scope starts at driver zero and restamps idempotently", () => {
  const node = new LoopNode("loop", { flowId: "body", times: 0, maxIters: 0, predicateKind: "times" });
  const flow = new Flow(new Map<string, Node>([["first", code("first")], ["loop", node]]), []);
  stampExecutionScopes(flow);
  const scope = node.executionScope;
  expect(scope!.occurrencePath).toEqual([new ExecutionPosition("loop_driver", ["loop"], 0)]);
  stampExecutionScopes(flow);
  expect(node.executionScope).toEqual(scope);
  expect(flow.nodes.get("first")!.executionScope!.occurrencePath).toEqual([]);
});

// ---------- loaded flows ----------

const REPEATED_DEFINITION = `id: root
name: root
defs:
  child:
    nodes:
      answer:
        kind: code
        code: |
          return 1
        output: int
    output: \${answer.output}
nodes:
  left: {kind: call, call: child}
  right: {kind: call, call: child}
output: \${right.output}
`;

const AGENT_ANSWER = "kind: agent\n        variables: {instruction: !str Summarize clearly}\n" +
  "        prompt: \${instruction}\n        output: str";

test("callsites to one definition have distinct paths", async () => {
  const flow = (await loadFlow(REPEATED_DEFINITION)).compiled;
  applyEffectiveLlmMap(flow, {});
  stampExecutionScopes(flow);
  const nodes = nodesByLocator(flow);
  const left = nodes.get("left/answer")!;
  const right = nodes.get("right/answer")!;
  expect(left).not.toBe(right);
  expect(left.executionScope).toEqual(new ExecutionScope(["left", "answer"], { flowPath: ["left"] }));
  expect(right.executionScope).toEqual(new ExecutionScope(["right", "answer"], { flowPath: ["right"] }));
  expect(nodes.get(`left/${START_ID}`)!.executionScope!.boundary).toBe("start");
  expect(nodes.get(`right/${END_ID}`)!.executionScope!.boundary).toBe("end");
});

test("agent variable owners are authored callsites and survive call cloning", async () => {
  const source = REPEATED_DEFINITION.replace(
    "kind: code\n        code: |\n          return 1\n        output: int",
    AGENT_ANSWER,
  );
  const flow = (await loadFlow(source)).compiled;
  applyEffectiveLlmMap(
    flow,
    Object.fromEntries(["left/answer", "right/answer"].map((path) => [path, { provider: "test", model: "test" }])),
  );
  stampExecutionScopes(flow);
  const owners = new Set(
    [...nodesByLocator(flow).values()].flatMap((node) =>
      node.declaredVariables().map((variable) => `${node.executionScope!.staticPath.join("/")}:${variable.name}`),
    ),
  );
  expect(owners).toEqual(new Set(["left/answer:instruction", "right/answer:instruction"]));

  for (const callsite of ["left", "right"]) {
    const spawner = flow.nodes.get(callsite) as CallNode;
    const child = spawner.child as Flow;
    const fragment = callSubgraph(child, callsite, {}, { executionScope: spawner.executionScope });
    const template = child.nodes.get("answer")!;
    const clone = [...fragment.nodes.values()].find((node) => node instanceof AgentNode)!;
    const [variable, ...rest] = clone.declaredVariables();
    expect(rest).toEqual([]);
    expect(owners.has(`${clone.executionScope!.staticPath.join("/")}:${variable!.name}`)).toBe(true);
    expect(clone.executionScope!.flowPath).toEqual([callsite]);
    expect(clone.executionScope!.occurrencePath).toEqual([new ExecutionPosition("call", [callsite])]);
    expect(variable).toBe(template.declaredVariables()[0]);
    expect(variable!.default).toBe("Summarize clearly");
    expect(clone.memoryScope).toBeNull();
  }
});

test("map items and loop iterations share the authored agent variable owner", async () => {
  const flow = (
    await loadFlow(`id: variable_owners
name: variable_owners
defs:
  body:
    input:
      n: !int
    nodes:
      writer:
        kind: agent
        input: {n: '\${input.n}'}
        variables:
          instruction: !str Summarize clearly
        prompt: '\${instruction}: \${n}'
        output: {n: int}
    output: \${writer.output}
  repeated:
    input:
      n: !int
    nodes:
      repeat:
        kind: loop
        call: body
        times: 2
        input: {n: '\${input.n}'}
    output: \${repeat.output}
nodes:
  outer:
    kind: map
    call: repeated
    over: [3, 3]
    input: {n: '\${item}'}
output: \${outer.output}
`)
  ).compiled;
  applyEffectiveLlmMap(flow, { "outer/repeat/writer": { provider: "test", model: "test" } });
  stampExecutionScopes(flow);
  const authored = [...nodesByLocator(flow).values()].find((node) => node instanceof AgentNode)!;
  const owner = [authored.executionScope!.staticPath, authored.declaredVariables()[0]!.name];
  expect(owner).toEqual([["outer", "repeat", "writer"], "instruction"]);
  const spawner = flow.nodes.get("outer") as MapNode;
  const fragment = mapSubgraph(spawner.child as Flow, spawner.id, [{ n: 3n }, { n: 3n }], {
    executionScope: spawner.executionScope,
  });
  const drivers = [...fragment.nodes.values()]
    .filter((node): node is LoopNode => node instanceof LoopNode)
    .sort((a, b) => a.executionScope!.occurrencePath[0]!.index! - b.executionScope!.occurrencePath[0]!.index!);
  expect(drivers).toHaveLength(2);
  const occurrences = new Set<string>();
  const variables = new Set<unknown>();
  drivers.forEach((driver, item) => {
    let current = driver;
    for (let iteration = 0; iteration < 2; iteration++) {
      const nextDriver = current.respawn(iteration + 1);
      const body = loopContinueSubgraph(current.child as Flow, current.originId!, { n: 3n }, iteration, nextDriver, {
        executionScope: current.executionScope,
      });
      const clone = [...body.nodes.values()].find((node) => node instanceof AgentNode)!;
      const [variable, ...rest] = clone.declaredVariables();
      expect(rest).toEqual([]);
      expect([clone.executionScope!.staticPath, variable!.name]).toEqual(owner);
      expect(clone.executionScope!.occurrencePath).toEqual([
        new ExecutionPosition("map_item", ["outer"], item),
        new ExecutionPosition("loop_body", ["outer", "repeat"], iteration),
      ]);
      expect(variable!.default).toBe("Summarize clearly");
      expect(clone.memoryScope).toBeNull();
      occurrences.add(JSON.stringify(clone.executionScope!.occurrencePath));
      variables.add(variable);
      current = nextDriver;
    }
  });
  expect(occurrences.size).toBe(4);
  expect(variables.size).toBe(1);
  expect(variables.has(authored.declaredVariables()[0])).toBe(true);
  expect(authored.executionScope!.occurrencePath).toEqual([]);
});

test.each([true, false])("execution scope agrees with existing memory ownership paths (child policy %p)", async (childPolicy) => {
  let source = REPEATED_DEFINITION.replace("name: root\n", "name: root\nlong_term_memory: outer\n").replace(
    "kind: code\n        code: |\n          return 1",
    "kind: agent\n        prompt: answer\n        long_term_memory: local",
  );
  if (childPolicy) source = source.replace("  child:\n", "  child:\n    long_term_memory: inner\n");
  const flow = (await loadFlow(source)).compiled;
  applyEffectiveLlmMap(
    flow,
    Object.fromEntries(
      ["left/answer", "right/answer"].map((path) => [path, { provider: "openai", model: "test-model" }]),
    ),
  );
  const memoryLayout = stampMemoryLayout(flow);
  const oldMemory = new Map(memoryLayout.targets.map((target) => [target.locator.join("/"), target.node.memoryScope]));
  stampExecutionScopes(flow);

  expect(resolveMemoryLayout(flow)).toEqual(memoryLayout);
  const nodes = nodesByLocator(flow);
  expect(new Set(nodes.keys())).toEqual(new Set(oldMemory.keys()));
  for (const [locator, node] of nodes) {
    const memoryScope = oldMemory.get(locator)!;
    expect(node.memoryScope).toBe(memoryScope);
    expect(memoryScope.staticPath).toEqual(node.executionScope!.staticPath);
    expect(memoryScope.memories[0]).toBe("outer");
    expect(node.executionScope!.occurrencePath).toEqual([]);
  }
  const childStart = nodes.get(`left/${START_ID}`)!.executionScope!;
  expect(childStart.flowPath).toEqual(["left"]);
  expect(childStart.boundary).toBe("start");
});
