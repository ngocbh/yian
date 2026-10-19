/**
 * Memory-independent static execution paths and atomic scope application
 * (`compile/execution_layout.ts`): rows of `tests/engine/test_execution_layout.py`.
 *
 * Ported: `test_execution_layout_exists_without_memory_policies` (both flow classes),
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
 * Later (T3-21): `test_execution_scope_has_an_explicit_manifest_policy`.
 *
 * Later (T3-22): `test_callsites_to_one_definition_have_distinct_paths`,
 * `test_agent_variable_owners_are_authored_callsites_and_survive_call_cloning`,
 * `test_map_items_and_loop_iterations_share_the_authored_agent_variable_owner` and
 * `test_execution_scope_agrees_with_existing_memory_ownership_paths` (they load flows).
 */

import { describe, expect, test } from "bun:test";

import {
  applyExecutionLayout,
  ResolvedExecutionLayout,
  ResolvedExecutionTarget,
  resolveExecutionLayout,
} from "../../src/compile/execution_layout.ts";
import { CompiledFlow, copyGraph, END_ID, Flow, START_ID } from "../../src/compile/model.ts";
import { compareCodeUnits, OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
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
