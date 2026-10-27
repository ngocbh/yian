/**
 * Memory-independent static execution paths and atomic scope application
 * (`compile/execution_layout.ts`): rows of `tests/engine/test_execution_layout.py`.
 *
 * Ported: `test_execution_scope_has_an_explicit_manifest_policy`,
 * `test_execution_layout_exists_without_memory_policies` (both flow classes),
 * `test_aliased_child_targets_are_rejected_before_any_scope_is_stamped`,
 * `test_stale_graph_is_rejected_without_stamping_any_target`,
 * `test_invalid_layout_is_rejected_without_stamping_any_target`,
 * `test_scope_validation_is_atomic_even_for_forged_frozen_values`,
 * `test_recursive_static_child_cycle_is_rejected_without_mutation`,
 * `test_static_children_are_discovered_through_the_node_owned_hook`,
 * `test_only_the_containing_flows_actual_boundaries_are_marked`,
 * `test_static_loop_scope_starts_at_driver_zero_and_reapplies_atomically` and
 * `test_static_loop_layout_rejects_nonbootstrap_positions_before_stamping`.
 *
 * Changed:
 * - Python's `ValueError` is a `RangeError`; layout `==` compares each target's locator and scope
 *   (the node isn't compared), here through `view`.
 * - The targets, layouts, scopes and positions are frozen in TS, where Python uses
 *   `object.__setattr__`: the rows forge them with `forge` (an object of the class's prototype
 *   that never ran the constructor), and a forged scope reaches the layout through a forged target
 *   inside a forged layout.
 * - The stale-graph row's replaced target is `copyGraph(node)` (Python's `copy.deepcopy`).
 * - The invalid-layout row's "mutable-targets" is a forged layout whose `targets` isn't an array.
 * - The atomic-scope row's string subclass is a path holding `new String("last")`.
 *
 * Ported through `loadFlow`, `applyEffectiveLlmMap` and the expansion builders:
 * `test_callsites_to_one_definition_have_distinct_paths`,
 * `test_agent_variable_owners_are_authored_callsites_and_survive_call_cloning`,
 * `test_map_items_and_loop_iterations_share_the_authored_agent_variable_owner` and
 * `test_execution_scope_agrees_with_existing_memory_ownership_paths`. Changed: a clone keeps the
 * template's frozen variable `Parameter` (`copyGraph` shares carriers), where Python's clone holds
 * a copy, so the CALL row checks the clone's variable is the template's and the MAP/LOOP row that
 * all four occurrences hold the one authored variable.
 */

import { describe, expect, test } from "bun:test";

import {
  applyExecutionLayout,
  ResolvedExecutionLayout,
  ResolvedExecutionTarget,
  resolveExecutionLayout,
} from "../../src/compile/execution_layout.ts";
import { callSubgraph, loopContinueSubgraph, mapSubgraph } from "../../src/compile/expand.ts";
import { applyEffectiveLlmMap } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { applyMemoryLayout, resolveMemoryLayout } from "../../src/compile/memory_layout.ts";
import { CompiledFlow, copyGraph, END_ID, Flow, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { compareCodeUnits, OriginRef } from "../../src/diagnostics.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

/** Make an object of `proto`'s class holding `fields`, without running its constructor. */
function forge<T extends object>(proto: T, fields: object): T {
  return Object.assign(Object.create(proto), fields) as T;
}

/** A layout's comparable part: each target's locator and scope, in order. */
function view(layout: ResolvedExecutionLayout): unknown[] {
  return layout.targets.map((target) => [target.locator, target.scope]);
}

/** A CODE node returning 1. */
function code(id: string): CodeNode {
  return new CodeNode(id, { code: "return 1" });
}

/** Expect `fn` to throw a `TypeError` or `RangeError` (Python's `TypeError`/`ValueError`). */
function expectRejected(fn: () => void): void {
  let caught: unknown = null;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught instanceof TypeError || caught instanceof RangeError).toBe(true);
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

/** A two-node flow and its resolved layout. */
function twoNodeLayout(): [CompiledFlow, ResolvedExecutionLayout] {
  const flow = new CompiledFlow(
    new Map<string, Node>([
      ["first", code("first")],
      ["last", new CodeNode("last", { code: "return 2" })],
    ]),
    [],
  );
  return [flow, resolveExecutionLayout(flow)];
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
])("execution layout exists without memory policies (%s)", (_name, makeFlow) => {
  test("resolves and applies", () => {
    const node = new CodeNode("authored~3/step", { code: "return 1", title: "display name" });
    node.origin = new OriginRef("unrelated:source");
    const flow = makeFlow(
      new Map<string, Node>([
        [node.id, node],
        [START_ID, new StartNode(START_ID, { params: [] })],
        [END_ID, EndNode.value(END_ID, { outputName: null })],
      ]),
    );
    const layout = resolveExecutionLayout(flow);
    const expected = new Map<string, ExecutionScope>([
      [END_ID, new ExecutionScope([END_ID], { boundary: "end" })],
      [START_ID, new ExecutionScope([START_ID], { boundary: "start" })],
      [node.id, new ExecutionScope([node.id])],
    ]);
    const sortedIds = [...expected.keys()].sort(compareCodeUnits);
    expect(view(layout)).toEqual(sortedIds.map((id) => [[id], expected.get(id)]));
    expect(layout.targets.every((target) => target.node.executionScope === null)).toBe(true);
    node.title = "another display name";
    node.origin = new OriginRef("different:source");
    expect(view(resolveExecutionLayout(flow))).toEqual(view(layout));

    applyExecutionLayout(flow, layout);
    expect(layout.targets.every((target) => target.node.executionScope === target.scope)).toBe(true);
    expect(layout.targets.every((target) => target.node.memoryScope === null)).toBe(true);
    applyExecutionLayout(flow, layout);
    expect(view(resolveExecutionLayout(flow))).toEqual(view(layout));

    const plain = makeFlow(new Map<string, Node>([["plain", code("plain")]]));
    applyExecutionLayout(plain, resolveExecutionLayout(plain));
    expect(plain.nodes.get("plain")!.executionScope).toEqual(new ExecutionScope(["plain"]));
    expect(resolveExecutionLayout(makeFlow(new Map())).targets).toEqual([]);
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
  const layout = resolveExecutionLayout(flow);
  expect(() => applyExecutionLayout(flow, layout)).toThrow(RangeError);
  expect(() => applyExecutionLayout(flow, layout)).toThrow(/multiple static paths/);
  expect(layout.targets.every((target) => target.node.executionScope === null)).toBe(true);
});

test.each(["new-target", "replaced-target", "invalid-key"])(
  "stale graph is rejected without stamping any target (%s)",
  (change) => {
    const [flow, layout] = twoNodeLayout();
    if (change === "new-target") {
      flow.nodes.set("new", new CodeNode("new", { code: "return 3" }));
    } else if (change === "replaced-target") {
      flow.nodes.set("last", copyGraph(flow.nodes.get("last")!));
    } else {
      const last = flow.nodes.get("last")!;
      flow.nodes.delete("last");
      flow.nodes.set("bad-key", last);
    }
    expect(() => applyExecutionLayout(flow, layout)).toThrow(RangeError);
    expect([...flow.nodes.values()].every((node) => node.executionScope === null)).toBe(true);
    expect(layout.targets.every((target) => target.node.executionScope === null)).toBe(true);
  },
);

test.each(["missing-target", "duplicate-target", "mutable-targets"])(
  "invalid layout is rejected without stamping any target (%s)",
  (change) => {
    const [flow, real] = twoNodeLayout();
    let layout: ResolvedExecutionLayout;
    if (change === "missing-target") {
      layout = new ResolvedExecutionLayout(real.targets.slice(0, -1));
    } else if (change === "duplicate-target") {
      layout = new ResolvedExecutionLayout([...real.targets, real.targets.at(-1)!]);
    } else {
      layout = forge(ResolvedExecutionLayout.prototype, { targets: new Set(real.targets) });
    }
    expectRejected(() => applyExecutionLayout(flow, layout));
    expect([...flow.nodes.values()].every((node) => node.executionScope === null)).toBe(true);
  },
);

describe.each(["layout", "existing"])("scope validation is atomic even for forged frozen values (%s)", (location) => {
  test.each(["different-path", "path-subclass", "position-index"])("%s", (change) => {
    const [flow, layout] = twoNodeLayout();
    const [first, final] = layout.targets as [ResolvedExecutionTarget, ResolvedExecutionTarget];
    let fields: object;
    if (change === "different-path") {
      fields = { staticPath: ["other"] };
    } else if (change === "path-subclass") {
      fields = { staticPath: [new String("last")] };
    } else {
      const position = forge(ExecutionPosition.prototype, { kind: "map_item", staticPath: ["map"], index: true });
      fields = { occurrencePath: [position] };
    }
    const scope = forge(ExecutionScope.prototype, { ...final.scope, ...fields });
    let applied = layout;
    if (location === "layout") {
      const target = forge(ResolvedExecutionTarget.prototype, { ...final, scope });
      applied = forge(ResolvedExecutionLayout.prototype, { targets: [first, target] });
    } else {
      final.node.executionScope = scope;
    }
    expectRejected(() => applyExecutionLayout(flow, applied));
    expect(flow.nodes.get("first")!.executionScope).toBeNull();
    expect(final.node.executionScope).toBe(location === "existing" ? scope : null);
  });
});

test("recursive static child cycle is rejected without mutation", () => {
  const flow = new Flow(new Map(), []);
  const node = new CallNode("recurse", { flowId: "child", child: flow });
  flow.nodes.set(node.id, node);
  expect(() => resolveExecutionLayout(flow)).toThrow(RangeError);
  expect(() => resolveExecutionLayout(flow)).toThrow(/recursive child-flow cycle/);
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
  applyExecutionLayout(flow, resolveExecutionLayout(flow));
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
  const layout = resolveExecutionLayout(flow);
  expect(layout.targets.map((target) => [target.locator, target.scope.boundary])).toEqual([
    [["entry"], "start"],
    [["helper"], null],
    [["result"], "end"],
  ]);
});

test("static loop scope starts at driver zero and reapplies atomically", () => {
  const node = new LoopNode("loop", { flowId: "body", times: 0, maxIters: 0, predicateKind: "times" });
  const flow = new Flow(new Map<string, Node>([["first", code("first")], ["loop", node]]), []);
  const layout = resolveExecutionLayout(flow);
  const target = layout.targets.find((item) => item.node === node)!;
  expect(target.scope.occurrencePath).toEqual([new ExecutionPosition("loop_driver", ["loop"], 0)]);
  expect(flow.nodes.get("first")!.executionScope).toBeNull();
  applyExecutionLayout(flow, layout);
  applyExecutionLayout(flow, layout);
  expect(view(resolveExecutionLayout(flow))).toEqual(view(layout));
  expect(flow.nodes.get("first")!.executionScope!.occurrencePath).toEqual([]);
});

test.each([
  ["empty", []],
  ["driver one", [new ExecutionPosition("loop_driver", ["loop"], 1)]],
  ["other driver", [new ExecutionPosition("loop_driver", ["other"], 0)]],
  ["body zero", [new ExecutionPosition("loop_body", ["loop"], 0)]],
  ["nested driver", [new ExecutionPosition("call", ["outer"]), new ExecutionPosition("loop_driver", ["loop"], 0)]],
])("static loop layout rejects nonbootstrap positions before stamping (%s)", (_name, position) => {
  const node = new LoopNode("loop", { flowId: "body" });
  const flow = new Flow(new Map<string, Node>([["first", code("first")], ["loop", node]]), []);
  const layout = resolveExecutionLayout(flow);
  const target = layout.targets.find((item) => item.node === node)!;
  const scope = forge(ExecutionScope.prototype, { ...target.scope, occurrencePath: position });
  const forged = forge(ResolvedExecutionTarget.prototype, { ...target, scope });
  const applied = forge(ResolvedExecutionLayout.prototype, {
    targets: layout.targets.map((item) => (item === target ? forged : item)),
  });
  expect(() => applyExecutionLayout(flow, applied)).toThrow(RangeError);
  expect(() => applyExecutionLayout(flow, applied)).toThrow(/initial driver position/);
  expect([...flow.nodes.values()].every((item) => item.executionScope === null)).toBe(true);
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

/** Return the layout's targets by locator, joined with `/`. */
function targetsByLocator(layout: ResolvedExecutionLayout): Map<string, ResolvedExecutionTarget> {
  return new Map(layout.targets.map((target) => [target.locator.join("/"), target]));
}

test("callsites to one definition have distinct paths", async () => {
  const flow = (await loadFlow(REPEATED_DEFINITION)).compiled;
  applyEffectiveLlmMap(flow, {});
  const layout = resolveExecutionLayout(flow);
  const targets = targetsByLocator(layout);
  const left = targets.get("left/answer")!;
  const right = targets.get("right/answer")!;
  expect(left.node).not.toBe(right.node);
  expect(left.scope).toEqual(new ExecutionScope(["left", "answer"], { flowPath: ["left"] }));
  expect(right.scope).toEqual(new ExecutionScope(["right", "answer"], { flowPath: ["right"] }));
  expect(targets.get(`left/${START_ID}`)!.scope.boundary).toBe("start");
  expect(targets.get(`right/${END_ID}`)!.scope.boundary).toBe("end");

  applyExecutionLayout(flow, layout);
  expect(left.node.executionScope).toEqual(left.scope);
  expect(right.node.executionScope).toEqual(right.scope);
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
  const layout = resolveExecutionLayout(flow);
  const owners = new Set(
    layout.targets.flatMap((target) =>
      target.node.declaredVariables().map((variable) => `${target.scope.staticPath.join("/")}:${variable.name}`),
    ),
  );
  expect(owners).toEqual(new Set(["left/answer:instruction", "right/answer:instruction"]));
  expect(layout.targets.every((target) => target.node.executionScope === null)).toBe(true);

  applyExecutionLayout(flow, layout);
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
  const layout = resolveExecutionLayout(flow);
  const authored = layout.targets.find((target) => target.node instanceof AgentNode)!;
  const owner = [authored.scope.staticPath, authored.node.declaredVariables()[0]!.name];
  expect(owner).toEqual([["outer", "repeat", "writer"], "instruction"]);
  applyExecutionLayout(flow, layout);
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
  expect(variables.has(authored.node.declaredVariables()[0])).toBe(true);
  expect(authored.node.executionScope!.occurrencePath).toEqual([]);
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
  const memoryLayout = resolveMemoryLayout(flow);
  applyMemoryLayout(flow, memoryLayout);
  const oldMemory = new Map(memoryLayout.targets.map((target) => [target.locator.join("/"), target.node.memoryScope]));
  const executionLayout = resolveExecutionLayout(flow);
  applyExecutionLayout(flow, executionLayout);

  expect(resolveMemoryLayout(flow)).toEqual(memoryLayout);
  for (const target of executionLayout.targets) {
    const memoryScope = oldMemory.get(target.locator.join("/"))!;
    expect(target.node.memoryScope).toBe(memoryScope);
    expect(memoryScope.staticPath).toEqual(target.scope.staticPath);
    expect(memoryScope.memories[0]).toBe("outer");
    expect(target.scope.occurrencePath).toEqual([]);
  }
  const childStart = executionLayout.targets.find((target) => target.locator.join("/") === `left/${START_ID}`)!;
  expect(childStart.scope.flowPath).toEqual(["left"]);
  expect(childStart.scope.boundary).toBe("start");
});
